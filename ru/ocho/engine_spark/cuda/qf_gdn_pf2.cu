// qf_gdn_pf2.cu - two-level chunked Gated DeltaNet for the prefill chunk (exact
// reformulation of prefill.cu k_gdn_prefill_chunked, which walked the 32-position
// chunks SEQUENTIALLY per head on 48 blocks).
//   phase 1 (parallel over heads x chunks): everything that does not depend on the
//     incoming state: normalized q/k, decays, A = beta_i e^{cp_i-cp_j} k_i.k_j,
//     B = q_i.k_j, and the two UT-transform solves
//        U = (I+A)^{-1} (beta o v)          W = (I+A)^{-1} (beta e^{cp} o k)
//     so that the state-dependent delta is the linear form  sd = U - W S.
//   phase 2 (one block per head, state S in shared memory, chunks in order):
//        sd_i = U_i - W_i S,   o_i = e^{cp_i} q_i S + sum_{j<=i} e^{cp_i-cp_j} B_ij sd_j,
//        S <- e^{cl} S + sum_j e^{cl-cp_j} k_j (x) sd_j
// Same formulas, same layouts as the sequential kernel; only the order of work changes.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define G2_VH 48
#define G2_D 128
#define G2_L 32
#define G2_DINN 10240
#define G2_VDIM (G2_VH * G2_D)

static __device__ __forceinline__ float g2_sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

// scratch per (head, chunk): U, W, Q, K [L][128]; B [L][L]; cp [L]
struct G2Scratch { float *U, *W, *Q, *K, *B, *cp; int nc; };

__global__ void __launch_bounds__(128) k_gdn2_intra(const float *__restrict__ qkv, const float *__restrict__ a_in, const float *__restrict__ b_in,
                                                    const float *__restrict__ A_log, const float *__restrict__ dt_bias, int T, G2Scratch sc) {
    __shared__ float sq[G2_L * G2_D], sk[G2_L * G2_D], sv[G2_L * G2_D];   // 48 KB: 2 blocks per SM under the 100 KB limit
    __shared__ float sA[G2_L * G2_L], sB[G2_L * G2_L];
    __shared__ float snq[G2_L], snk[G2_L], sLam[G2_L], sbeta[G2_L], scp[G2_L];
    const int h = blockIdx.x, c = blockIdx.y, kh = h / 3, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int s0 = c * G2_L;
    if (s0 >= T) return;
    const int L = (T - s0) < G2_L ? (T - s0) : G2_L;
    const float alog = A_log[h], dtb = dt_bias[h];
    if (tid < L) {
        const float av = a_in[(size_t)(s0 + tid) * G2_VH + h], bv = b_in[(size_t)(s0 + tid) * G2_VH + h];
        const float dt = av + dtb;
        const float sp = dt > 20.f ? dt : log1pf(expf(dt));
        sLam[tid] = -__expf(alog) * sp;
        sbeta[tid] = g2_sigmoid(bv);
    }
    __syncthreads();
    if (tid == 0) { float p = 0.f; for (int i = 0; i < L; i++) { p += sLam[i]; scp[i] = p; } }
    for (int r = wid; r < L; r += 4) {
        const float *qr = qkv + (size_t)(s0 + r) * G2_DINN + kh * G2_D, *kr = qr + 2048;
        float q2 = 0.f, k2 = 0.f;
        for (int d = lane; d < G2_D; d += 32) { const float q = qr[d], k = kr[d]; q2 += q * q; k2 += k * k; }
        #pragma unroll
        for (int off = 16; off; off >>= 1) { q2 += __shfl_down_sync(~0u, q2, off); k2 += __shfl_down_sync(~0u, k2, off); }
        if (lane == 0) { snq[r] = rsqrtf(q2 + 1e-6f); snk[r] = rsqrtf(k2 + 1e-6f); }
    }
    __syncthreads();
    for (int idx = tid; idx < L * G2_D; idx += 128) {
        const int r = idx >> 7, d = idx & 127;
        const float *row = qkv + (size_t)(s0 + r) * G2_DINN;
        sq[idx] = row[kh * G2_D + d] * snq[r] * (1.f / sqrtf((float)G2_D));
        sk[idx] = row[2048 + kh * G2_D + d] * snk[r];
        sv[idx] = row[4096 + h * G2_D + d];
    }
    __syncthreads();
    for (int p = wid; p < L * L; p += 4) {
        const int i = p / L, j = p % L;
        float dA = 0.f, dB = 0.f;
        for (int d = lane; d < G2_D; d += 32) { dA += sk[i * G2_D + d] * sk[j * G2_D + d]; dB += sq[i * G2_D + d] * sk[j * G2_D + d]; }
        #pragma unroll
        for (int off = 16; off; off >>= 1) { dA += __shfl_down_sync(~0u, dA, off); dB += __shfl_down_sync(~0u, dB, off); }
        if (lane == 0) { sA[i * L + j] = (j < i) ? sbeta[i] * __expf(scp[i] - scp[j]) * dA : 0.f; sB[i * L + j] = (j <= i) ? dB : 0.f; }
    }
    __syncthreads();
    // U = (I+A)^{-1} (beta o v): thread = v-column; in place in sv
    const int col = tid;
    for (int i = 0; i < L; i++) {
        float val = sbeta[i] * sv[i * G2_D + col];
        for (int j = 0; j < i; j++) val -= sA[i * L + j] * sv[j * G2_D + col];
        sv[i * G2_D + col] = val;
        __syncthreads();
    }
    const size_t o = ((size_t)h * sc.nc + c);
    float *U = sc.U + o * (G2_L * G2_D), *W = sc.W + o * (G2_L * G2_D), *Q = sc.Q + o * (G2_L * G2_D), *K = sc.K + o * (G2_L * G2_D);
    float *B = sc.B + o * (G2_L * G2_L), *cp = sc.cp + o * G2_L;
    for (int idx = tid; idx < L * G2_D; idx += 128) { U[idx] = sv[idx]; K[idx] = sk[idx]; }
    for (int idx = tid; idx < L * G2_D; idx += 128) { const int i = idx >> 7, r = idx & 127; Q[r * G2_L + i] = sq[idx]; }   // Q^T [r][i]
    __syncthreads();
    // W = (I+A)^{-1} (beta e^{cp} o k): thread = k-column; IN PLACE of sk (row i is read before written; rows j<i already hold W_j)
    for (int i = 0; i < L; i++) {
        float val = sbeta[i] * __expf(scp[i]) * sk[i * G2_D + col];
        for (int j = 0; j < i; j++) val -= sA[i * L + j] * sk[j * G2_D + col];
        __syncthreads();
        sk[i * G2_D + col] = val;
        __syncthreads();
    }
    for (int idx = tid; idx < L * G2_D; idx += 128) { const int i = idx >> 7, r = idx & 127; W[r * G2_L + i] = sk[idx]; }   // W^T [r][i]
    for (int idx = tid; idx < L * L; idx += 128) B[idx] = sB[idx];
    if (tid < L) cp[tid] = scp[tid];
}

// one block per head; 256 threads: thread = (half = tid>>7, column c = tid&127)
__global__ void __launch_bounds__(256) k_gdn2_scan(float *__restrict__ S_g, float *__restrict__ out48, int T, G2Scratch sc) {
    extern __shared__ float g2sm[];
    float *S = g2sm;                                   // [128][128]
    float *sd = S + G2_D * G2_D;                       // [L][128]
    float *sB = sd + G2_L * G2_D;                      // [L][L]
    float *scp = sB + G2_L * G2_L;                     // [L]
    float *sdec = scp + G2_L;                          // e^{cl - cp_j} [L]
    const int h = blockIdx.x, tid = threadIdx.x, c = tid & 127, half = tid >> 7;
    float *Sh = S_g + (size_t)h * G2_D * G2_D;
    for (int i = tid; i < G2_D * G2_D; i += 256) S[i] = Sh[i];
    const int nc = (T + G2_L - 1) / G2_L;
    for (int ch = 0; ch < nc; ch++) {
        const int s0 = ch * G2_L, L = (T - s0) < G2_L ? (T - s0) : G2_L;
        const size_t o = ((size_t)h * sc.nc + ch);
        const float *U = sc.U + o * (G2_L * G2_D), *W = sc.W + o * (G2_L * G2_D), *Q = sc.Q + o * (G2_L * G2_D), *K = sc.K + o * (G2_L * G2_D);
        for (int i = tid; i < L * L; i += 256) sB[i] = sc.B[o * (G2_L * G2_L) + i];
        if (tid < L) scp[tid] = sc.cp[o * G2_L + tid];
        __syncthreads();
        if (tid < L) sdec[tid] = __expf(scp[L - 1] - scp[tid]);
        // Row-outer products against the state column c: 16 independent accumulators
        // per thread (rows i = half, half+2, ...), W/Q rows read as warp broadcasts.
        {
            float acc[G2_L / 2];
            #pragma unroll
            for (int k = 0; k < G2_L / 2; k++) acc[k] = 0.f;
            for (int r = 0; r < G2_D; r++) {
                const float sv = S[r * G2_D + c];
                const float *Wr = W + r * G2_L;                       // W^T row r: the 32 i-values contiguous (2 sectors, broadcast)
                #pragma unroll
                for (int k = 0; k < G2_L / 2; k++) { const int i = half + 2 * k; if (i < L) acc[k] = fmaf(Wr[i], sv, acc[k]); }
            }
            #pragma unroll
            for (int k = 0; k < G2_L / 2; k++) { const int i = half + 2 * k; if (i < L) sd[i * G2_D + c] = U[i * G2_D + c] - acc[k]; }
        }
        __syncthreads();
        {
            float acc[G2_L / 2];
            #pragma unroll
            for (int k = 0; k < G2_L / 2; k++) acc[k] = 0.f;
            for (int r = 0; r < G2_D; r++) {
                const float sv = S[r * G2_D + c];
                const float *Qr = Q + r * G2_L;
                #pragma unroll
                for (int k = 0; k < G2_L / 2; k++) { const int i = half + 2 * k; if (i < L) acc[k] = fmaf(Qr[i], sv, acc[k]); }
            }
            #pragma unroll
            for (int k = 0; k < G2_L / 2; k++) {
                const int i = half + 2 * k;
                if (i < L) {
                    float ov = __expf(scp[i]) * acc[k];
                    for (int j = 0; j <= i; j++) ov = fmaf(__expf(scp[i] - scp[j]) * sB[i * G2_L + j], sd[j * G2_D + c], ov);
                    out48[((size_t)(s0 + i) * G2_VH + h) * G2_D + c] = ov;
                }
            }
        }
        __syncthreads();
        // S <- e^{cl} S + sum_j e^{cl-cp_j} k_j (x) sd_j : thread (half, c) owns rows r = half, half+2, ...
        const float cle = __expf(scp[L - 1]);
        for (int r = half; r < G2_D; r += 2) {
            float acc = cle * S[r * G2_D + c];
            #pragma unroll 8
            for (int j = 0; j < L; j++) acc = fmaf(sdec[j] * K[j * G2_D + r], sd[j * G2_D + c], acc);
            S[r * G2_D + c] = acc;
        }
        __syncthreads();
    }
    for (int i = tid; i < G2_D * G2_D; i += 256) Sh[i] = S[i];
}

extern "C" {
size_t qf_gdn2_scratch_floats(int maxT) {
    const size_t nc = (size_t)(maxT + G2_L - 1) / G2_L;
    return (size_t)G2_VH * nc * (4 * G2_L * G2_D + G2_L * G2_L + G2_L);
}
// scratch: qf_gdn2_scratch_floats(maxT) floats; maxT >= T.
int qf_gdn2_prefill(const float *qkv, const float *a, const float *b, const float *A_log, const float *dt_bias,
                    float *S, float *out48, int T, float *scratch, int maxT, cudaStream_t s) {
    if (T < 1 || T > maxT) return -1;
    static int smem_set = 0;
    const int smem = (G2_D * G2_D + G2_L * G2_D + G2_L * G2_L + 2 * G2_L) * 4;
    if (!smem_set) { if (cudaFuncSetAttribute(k_gdn2_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return -1; smem_set = 1; }
    G2Scratch sc; sc.nc = (maxT + G2_L - 1) / G2_L;
    const size_t per = (size_t)G2_VH * sc.nc;
    sc.U = scratch; sc.W = sc.U + per * G2_L * G2_D; sc.Q = sc.W + per * G2_L * G2_D; sc.K = sc.Q + per * G2_L * G2_D;
    sc.B = sc.K + per * G2_L * G2_D; sc.cp = sc.B + per * G2_L * G2_L;
    const int nc = (T + G2_L - 1) / G2_L;
    static int prof = -1; static cudaEvent_t ev[3]; static double ms_intra = 0, ms_scan = 0; static int calls = 0;
    if (prof < 0) { prof = (getenv("QF_PF_TIMING") && getenv("QF_PF_TIMING")[0] == '1') ? 1 : 0; if (prof) for (int i = 0; i < 3; i++) cudaEventCreate(&ev[i]); }
    if (prof) cudaEventRecord(ev[0], s);
    k_gdn2_intra<<<dim3(G2_VH, nc), 128, 0, s>>>(qkv, a, b, A_log, dt_bias, T, sc);
    if (prof) cudaEventRecord(ev[1], s);
    k_gdn2_scan<<<G2_VH, 256, smem, s>>>(S, out48, T, sc);
    if (prof) { cudaEventRecord(ev[2], s); cudaEventSynchronize(ev[2]); float a1 = 0, a2 = 0; cudaEventElapsedTime(&a1, ev[0], ev[1]); cudaEventElapsedTime(&a2, ev[1], ev[2]); ms_intra += a1; ms_scan += a2;
                if (++calls % 24 == 0) { fprintf(stderr, "gdn2 (24 layers, T=%d): intra %.1f scan %.1f ms\n", T, ms_intra, ms_scan); ms_intra = ms_scan = 0; } }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
}
