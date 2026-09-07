// prefill.cu - chunked prefill path (T tokens at once), batch 1.
// Mirrors qf.cu decode numerics: same decay/beta/sigmoid/silu formulas, same memory
// layouts. Final states must match T sequential decode steps (see test_prefill.cu).
#include "prefill.h"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

#define GDN_VH 48
#define GDN_KD 128
#define GDN_VD 128
#define DINN 10240
#define NHEAD 24
#define NKV 2
#define HDIM 256
#define HCC 4
#define HCL 320
#define NEMBD 2560

__device__ __forceinline__ float pf_sigmoid(float v) { return 1.f / (1.f + expf(-v)); }

// ---------------------------------------------------------------------------
// 1. GDN chunked parallel delta rule
// ---------------------------------------------------------------------------
// shared layout (floats): sq, sk, sv, sd, sqS [CH][128] each, sA, sB [CH][CH],
// saux [5*CH] = invq | invk | lambda | beta | cumprod
#define PF_SMEM_FLOATS (5 * PF_CHUNK * 128 + 2 * PF_CHUNK * PF_CHUNK + 5 * PF_CHUNK)
#define PF_SMEM_BYTES (PF_SMEM_FLOATS * 4)

__global__ void k_gdn_prefill_chunked(const float *__restrict__ qkv,
                                      const float *__restrict__ a_in,
                                      const float *__restrict__ b_in,
                                      const float *__restrict__ A_log,
                                      const float *__restrict__ dt_bias,
                                      float *__restrict__ S,
                                      float *__restrict__ out48, int T) {
    extern __shared__ float sm[];
    float *sq  = sm;                              // normed, scaled q
    float *sk  = sq + PF_CHUNK * 128;             // normed k
    float *sv  = sk + PF_CHUNK * 128;             // raw v, then rhs
    float *sd  = sv + PF_CHUNK * 128;             // delta
    float *sqS = sd + PF_CHUNK * 128;             // S0^T q_i
    float *sA  = sqS + PF_CHUNK * 128;            // [i][j], j<i
    float *sB  = sA + PF_CHUNK * PF_CHUNK;        // q_i.k_j, j<=i
    float *snq = sB + PF_CHUNK * PF_CHUNK;
    float *snk = snq + PF_CHUNK;
    float *sLam = snk + PF_CHUNK;
    float *sbeta = sLam + PF_CHUNK;
    float *scp = sbeta + PF_CHUNK;                // inclusive cumulative decay product

    const int h = blockIdx.x, kh = h / 3, tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    float *Sh = S + (size_t)h * GDN_KD * GDN_VD;
    const float alog = A_log[h], dtb = dt_bias[h];

    for (int s0 = 0; s0 < T; s0 += PF_CHUNK) {
        const int L = min(PF_CHUNK, T - s0);
        // per-token decay lambda (same formula as k_gdn_decode: S *= g, g multiplicative)
        if (tid < L) {
            float av = a_in[(size_t)(s0 + tid) * GDN_VH + h];
            float bv = b_in[(size_t)(s0 + tid) * GDN_VH + h];
            // sLam must hold the DECAY exp(g_log), not g_log itself: scp below
            // is a cumulative PRODUCT of decays and is used as ratios
            // scp[i]/scp[j]. Storing the log here made scp a product of logs,
            // which is not the decay of anything.
            //   k_gdn_decode: g_log = -exp(A_log) * softplus(a + dt_bias)
            //                 decay = exp(g_log);  S *= decay
            // Also carries k_gdn_decode's softplus overflow guard: expf(dt) for
            // large dt is inf, and log1pf(inf) is inf.
            float dt = av + dtb;
            float sp = dt > 20.f ? dt : log1pf(expf(dt));
            sLam[tid] = -__expf(alog) * sp;          // g_log (negative)
            sbeta[tid] = pf_sigmoid(bv);
        }
        __syncthreads();
        // scp holds the cumulative SUM of log-decays. The chunked form needs
        // the ratio prod_{u=j+1..i} decay_u, which is exp(scp[i] - scp[j]).
        // A cumulative PRODUCT of decays underflows: A_log covers A in
        // [0.01,16], so a single decay can be ~e^-16 and 16 of them is ~1e-112,
        // which flushes to zero and makes every ratio 0/0.
        if (tid == 0) {
            float p = 0.f;
            for (int i = 0; i < L; i++) { p += sLam[i]; scp[i] = p; }
        }
        // q/k l2 norms, warp per row
        for (int r = wid; r < L; r += 4) {
            const float *qr = qkv + (size_t)(s0 + r) * DINN + kh * GDN_KD;
            const float *kr = qr + 2048;
            float q2 = 0.f, k2 = 0.f;
            for (int d = lane; d < GDN_KD; d += 32) {
                float q = qr[d], k = kr[d];
                q2 += q * q; k2 += k * k;
            }
            for (int off = 16; off; off >>= 1) {
                q2 += __shfl_down_sync(~0u, q2, off);
                k2 += __shfl_down_sync(~0u, k2, off);
            }
            if (lane == 0) { snq[r] = rsqrtf(q2 + 1e-6f); snk[r] = rsqrtf(k2 + 1e-6f); }
        }
        __syncthreads();
        // pack normalized q (scaled by 1/sqrt(128)), k and raw v
        for (int idx = tid; idx < L * 128; idx += 128) {
            int r = idx >> 7, d = idx & 127;
            const float *row = qkv + (size_t)(s0 + r) * DINN;
            sq[idx] = row[kh * GDN_KD + d] * snq[r] * (1.f / sqrtf((float)GDN_KD));
            sk[idx] = row[2048 + kh * GDN_KD + d] * snk[r];
            sv[idx] = row[4096 + h * GDN_VD + d];
        }
        // A[i][j] = beta_i * cp[i]/cp[j] * (k_i.k_j) for j<i;  B[i][j] = q_i.k_j for j<=i
        // (decay-first ordering as in k_gdn_decode: delta_t uses lam_t * S_{t-1})
        for (int p = wid; p < L * L; p += 4) {
            int i = p / L, j = p % L;
            float dA = 0.f, dB = 0.f;
            for (int d = lane; d < 128; d += 32) {
                dA += sk[i * 128 + d] * sk[j * 128 + d];
                dB += sq[i * 128 + d] * sk[j * 128 + d];
            }
            for (int off = 16; off; off >>= 1) {
                dA += __shfl_down_sync(~0u, dA, off);
                dB += __shfl_down_sync(~0u, dB, off);
            }
            if (lane == 0) {
                if (j < i) sA[i * L + j] = sbeta[i] * __expf(scp[i] - scp[j]) * dA;
                if (j <= i) sB[i * L + j] = dB;
            }
        }
        __syncthreads();
        // rhs_i = beta_i (v_i - cp[i] * S0^T k_i); sqS_i = S0^T q_i. thread = v-column c.
        const int c = tid;
        for (int i = 0; i < L; i++) {
            float kvv = 0.f, qvv = 0.f;
            for (int r = 0; r < 128; r++) {
                float s = Sh[(size_t)r * 128 + c];
                kvv += s * sk[i * 128 + r];
                qvv += s * sq[i * 128 + r];
            }
            sv[i * 128 + c] = sbeta[i] * (sv[i * 128 + c] - __expf(scp[i]) * kvv);
            sqS[i * 128 + c] = qvv;
        }
        __syncthreads();
        // UT transform: (I + A) delta = rhs, forward substitution over i
        for (int i = 0; i < L; i++) {
            float val = sv[i * 128 + c];
            for (int j = 0; j < i; j++) val -= sA[i * L + j] * sd[j * 128 + c];
            sd[i * 128 + c] = val;
            __syncthreads();
        }
        // o_i = cp[i] * S0^T q_i + sum_{j<=i} cp[i]/cp[j] * (q_i.k_j) * delta_j
        for (int i = 0; i < L; i++) {
            float o = __expf(scp[i]) * sqS[i * 128 + c];
            for (int j = 0; j <= i; j++) o += __expf(scp[i] - scp[j]) * sB[i * L + j] * sd[j * 128 + c];
            out48[((size_t)(s0 + i) * GDN_VH + h) * GDN_VD + c] = o;
        }
        // S_end = cp[L-1] S0 + sum_j cp[L-1]/cp[j] * k_j outer delta_j
        const float cl = scp[L - 1];                 // cumulative LOG decay
        const float cle = __expf(cl);
        for (int r = 0; r < 128; r++) {
            float acc = cle * Sh[(size_t)r * 128 + c];
            for (int j = 0; j < L; j++)
                acc += __expf(cl - scp[j]) * sk[j * 128 + r] * sd[j * 128 + c];
            Sh[(size_t)r * 128 + c] = acc;
        }
        __syncthreads();
    }
}

int qf_gdn_prefill_f32w(const float *qkv, const float *a, const float *b,
                   const float *A_log, const float *dt_bias,
                   float *S, float *out, int T, cudaStream_t st) {
    if (T <= 0 || T > PF_MAX_T) return -1;
    static int smem_set = 0;
    if (!smem_set) {
        if (cudaFuncSetAttribute(k_gdn_prefill_chunked,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 PF_SMEM_BYTES) != cudaSuccess)
            return -1;
        smem_set = 1;
    }
    k_gdn_prefill_chunked<<<GDN_VH, 128, PF_SMEM_BYTES, st>>>(qkv, a, b, A_log, dt_bias, S, out, T);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// 2. causal SDPA with QSA selected-token mask
// ---------------------------------------------------------------------------
__global__ void k_attn_prefill(const float *__restrict__ q, int qstride,
                               const __nv_bfloat16 *__restrict__ K,
                               const __nv_bfloat16 *__restrict__ V,
                               const int *__restrict__ mask,
                               float *__restrict__ out, int T) {
    __shared__ float prob[PF_MAX_T];
    __shared__ float red[8];
    __shared__ float mx, sum;
    const int h = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
    const int kh = h / (NHEAD / NKV);
    const float *qi = q + (size_t)i * NHEAD * qstride + (size_t)h * qstride;
    const float scale = 1.f / sqrtf((float)HDIM);

    float local = -1e30f;
    for (int j = tid; j <= i; j += blockDim.x) {
        if (mask[j] == -1) { prob[j] = -1e30f; continue; }
        const __nv_bfloat16 *kj = K + (size_t)j * (NKV * HDIM) + kh * HDIM;
        float acc = 0.f;
        for (int d = 0; d < HDIM; d++) acc += qi[d] * __bfloat162float(kj[d]);
        prob[j] = acc * scale;
        local = fmaxf(local, prob[j]);
    }
    // Proper warp reduction. This was `for (off = 128; off; off >>= 1)` with
    // `min(off, 31)`, i.e. shuffle distances 31,31,31,16,8,4,2,1 - three
    // spurious shuffles by 31 before the real tree, and __shfl never crosses
    // warps anyway. The cross-warp step is the red[] array below.
    for (int off = 16; off; off >>= 1) local = fmaxf(local, __shfl_down_sync(~0u, local, off));
    if ((tid & 31) == 0) red[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
        float m2 = -1e30f;
        for (int w = 0; w < (int)(blockDim.x + 31) / 32; w++) m2 = fmaxf(m2, red[w]);
        mx = m2; sum = 0.f;
    }
    __syncthreads();
    float e = 0.f;
    for (int j = tid; j <= i; j += blockDim.x) {
        float ev = (prob[j] <= -1e29f) ? 0.f : expf(prob[j] - mx);
        prob[j] = ev;
        e += ev;                      // ACCUMULATE: was `e =`, which only
    }                                 // happened to work while T <= blockDim
    for (int off = 16; off; off >>= 1) e += __shfl_down_sync(~0u, e, off);
    if ((tid & 31) == 0) atomicAdd(&sum, e);
    __syncthreads();
    const float inv = (sum > 0.f) ? 1.f / (sum + 1e-9f) : 0.f;
    for (int j = tid; j <= i; j += blockDim.x) prob[j] *= inv;
    __syncthreads();
    const int d = tid;  // blockDim.x == HDIM
    float o = 0.f;
    for (int j = 0; j <= i; j++)
        o += prob[j] * __bfloat162float(V[(size_t)j * (NKV * HDIM) + kh * HDIM + d]);
    out[((size_t)i * NHEAD + h) * HDIM + d] = o;
}

int qf_attn_prefill(const float *q, int qstride, const void *kc, const void *vc,
                    const int *mask, float *out, int T, cudaStream_t st) {
    if (T <= 0 || T > PF_MAX_T) return -1;
    dim3 g(NHEAD, T);
    k_attn_prefill<<<g, HDIM, 0, st>>>(q, qstride, (const __nv_bfloat16 *)kc,
                                       (const __nv_bfloat16 *)vc, mask, out, T);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// 3. causal conv1d (kernel 4, dilation 1) + silu, same ring semantics as k_conv_step
// ---------------------------------------------------------------------------
__global__ void k_conv_prefill(float *__restrict__ raw, float *__restrict__ ring,
                               const float *__restrict__ w, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DINN) return;
    float p0 = ring[0 * DINN + i], p1 = ring[1 * DINN + i], p2 = ring[2 * DINN + i];
    const float w0 = w[i * 4 + 0], w1 = w[i * 4 + 1], w2 = w[i * 4 + 2], w3 = w[i * 4 + 3];
    for (int t = 0; t < T; t++) {
        float cur = raw[(size_t)t * DINN + i];
        float o = w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur;
        p0 = p1; p1 = p2; p2 = cur;
        raw[(size_t)t * DINN + i] = o / (1.f + expf(-o));
    }
    ring[0 * DINN + i] = p0; ring[1 * DINN + i] = p1; ring[2 * DINN + i] = p2;
}

int qf_conv_prefill_f32w(float *raw, float *ring, const float *w, int T, cudaStream_t st) {
    if (T <= 0 || T > PF_MAX_T) return -1;
    k_conv_prefill<<<(DINN + 255) / 256, 256, 0, st>>>(raw, ring, w, T);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// 4. hc wrappers: row-batched GEMM via cublas
// ---------------------------------------------------------------------------
static cublasHandle_t pf_cublas = NULL;

int qf_prefill_init(void) {
    if (!pf_cublas && cublasCreate(&pf_cublas) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cudaFuncSetAttribute(k_gdn_prefill_chunked,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PF_SMEM_BYTES) != cudaSuccess)
        return -1;
    return 0;
}

// Y[T][out] = X[T][in] * W[out][in]^T. Row-major everywhere: in cublas col-major terms
// Y_cm[out][T] = W_cm[in][out]^T * X_cm[in][T], where W_cm/X_cm/Y_cm are the same buffers.
int qf_gemm_bf16(const void *W, const float *X, float *Y, int out, int in, int T,
                 cudaStream_t st) {
    if (!pf_cublas && qf_prefill_init()) return -1;
    cublasSetStream(pf_cublas, st);
    const float alpha = 1.f, beta0 = 0.f;
    cublasStatus_t rc = cublasGemmEx(pf_cublas, CUBLAS_OP_T, CUBLAS_OP_N, out, T, in,
                                     &alpha, W, CUDA_R_16BF, in, X, CUDA_R_32F, in,
                                     &beta0, Y, CUDA_R_32F, out,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return rc == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int qf_gemm_f32(const float *W, const float *X, float *Y, int out, int in, int T,
                cudaStream_t st) {
    if (!pf_cublas && qf_prefill_init()) return -1;
    cublasSetStream(pf_cublas, st);
    const float alpha = 1.f, beta0 = 0.f;
    cublasStatus_t rc = cublasSgemm(pf_cublas, CUBLAS_OP_T, CUBLAS_OP_N, out, T, in,
                                    &alpha, W, in, X, in, &beta0, Y, out);
    return rc == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

// rmsnorm over 10240 with group_size 2560, zero-centered weight (matches k_rmsnorm)
__global__ void k_rmsnorm_rows(float *out, const float *x, const float *w) {
    __shared__ float ssq[HCC];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float *xr = x + (size_t)t * DINN;
    float *orow = out + (size_t)t * DINN;
    if (tid < HCC) ssq[tid] = 0.f;
    __syncthreads();
    for (int idx = tid; idx < DINN; idx += blockDim.x) {
        float v = xr[idx];
        atomicAdd(&ssq[idx / NEMBD], v * v);
    }
    __syncthreads();
    for (int idx = tid; idx < DINN; idx += blockDim.x) {
        int g = idx / NEMBD;
        orow[idx] = xr[idx] * rsqrtf(ssq[g] / (float)NEMBD + 1e-6f) * (1.f + w[idx]);
    }
}

__global__ void k_hc_dact(float *d, int n) {   // d = silu(d/4)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = d[i] / 4.f;
    d[i] = v / (1.f + expf(-v));
}

__global__ void k_hc_mix(float *mixed, const float *up, const float *normed, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NEMBD) return;
    int t = i / NEMBD, e = i % NEMBD;
    float acc = 0.f;
    for (int c = 0; c < HCC; c++)
        acc += pf_sigmoid(up[(size_t)t * DINN + c * NEMBD + e]) *
               normed[(size_t)t * DINN + c * NEMBD + e];
    mixed[i] = acc / (float)HCC;
}

__global__ void k_hc_inj(float *inj, int n) {   // inj = 2*sigmoid(inj/4)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inj[i] = 2.f * pf_sigmoid(inj[i] / 4.f);
}

int qf_hc_prefill_f32w(const float *R, const float *w_norm, const float *w_down,
                  const float *w_up, const float *w_inject,
                  float *mixed, float *inj,
                  float *w_normed, float *w_d320, float *w_up10240,
                  int T, cudaStream_t st) {
    if (T <= 0 || T > PF_MAX_T) return -1;
    k_rmsnorm_rows<<<T, 256, 0, st>>>(w_normed, R, w_norm);
    if (qf_gemm_f32(w_down, w_normed, w_d320, HCL, DINN, T, st)) return -1;
    k_hc_dact<<<(T * HCL + 255) / 256, 256, 0, st>>>(w_d320, T * HCL);
    if (qf_gemm_f32(w_up, w_d320, w_up10240, DINN, HCL, T, st)) return -1;
    k_hc_mix<<<(T * NEMBD + 255) / 256, 256, 0, st>>>(mixed, w_up10240, w_normed, T);
    if (w_inject && inj) {
        if (qf_gemm_f32(w_inject, w_normed, inj, HCC, DINN, T, st)) return -1;
        k_hc_inj<<<(T * HCC + 255) / 256, 256, 0, st>>>(inj, T * HCC);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
