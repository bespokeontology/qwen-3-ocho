// qf8_gdn_chunk.hip - Gated DeltaNet over a CHUNK of T positions on gfx906.
// HIP port of the engine's own chunked parallel delta rule (Spark
// src/cuda/prefill.cu k_gdn_prefill_chunked + k_conv_prefill), with the
// intra-chunk length cut to 16 so the shared footprint (44 KB) fits the 64 KB
// LDS of an MI50 (the CUDA kernel used 32 = 90 KB). Same formulas as the
// per-position kernel k8_gdn_fused: decay exp(-exp(A_log) * softplus(a + dt_bias)),
// beta = sigmoid(b), l2-normalized q/k, q scaled by 1/sqrt(128), v raw.
//   conv:   causal conv1d (kernel 4) + silu over T positions, ring carried
//   delta:  per v-head block, chunks of 16: A (k.k decays), B (q.k), S0^T k/q,
//           forward substitution, outputs, state update S <- cp*S + sum k (x) delta
//   gnorm:  out = w * rms(o) * sigmoid(z) per (position, head)
// Layouts (identical to the decode path): qkv_raw [T][10240] (q 0..2047, k
// 2048..4095, v 4096..10239), a/b [T][48], S [48][128][128], ring [3][10240],
// out [T][6144] (head h at h*128), z [T][6144].
#include "qf8_dense.h"
#include <math.h>

#ifndef QF_HOST_CHECK
#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#endif

#define GC_CH   16          // intra-chunk positions
#define GC_D    128
#define GC_VH   48
#define GC_DINN 10240
#define GC_SMEM_FLOATS (5 * GC_CH * GC_D + 2 * GC_CH * GC_CH + 5 * GC_CH)

__device__ __forceinline__ float gc_sigmoid(float v) { return 1.f / (1.f + expf(-v)); }
// 32-lane group sum on a wave64: offsets 16..1 stay inside each half.
__device__ __forceinline__ float gc_half_sum(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(~0u, v, off);
    return v;
}

__global__ void k_gc_conv(float *__restrict__ raw, float *__restrict__ ring,
                          const uint16_t *__restrict__ w, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= GC_DINN) return;
    float p0 = ring[0 * GC_DINN + i], p1 = ring[1 * GC_DINN + i], p2 = ring[2 * GC_DINN + i];
    const float w0 = qf8_bf2f(w[i * 4 + 0]), w1 = qf8_bf2f(w[i * 4 + 1]),
                w2 = qf8_bf2f(w[i * 4 + 2]), w3 = qf8_bf2f(w[i * 4 + 3]);
    for (int t = 0; t < T; t++) {
        const float cur = raw[(size_t)t * GC_DINN + i];
        const float o = w0 * p0 + w1 * p1 + w2 * p2 + w3 * cur;
        p0 = p1; p1 = p2; p2 = cur;
        raw[(size_t)t * GC_DINN + i] = o / (1.f + expf(-o));      // silu
    }
    ring[0 * GC_DINN + i] = p0; ring[1 * GC_DINN + i] = p1; ring[2 * GC_DINN + i] = p2;
}

__global__ void __launch_bounds__(128)
k_gc_delta(const float *__restrict__ qkv, const float *__restrict__ a_in,
           const float *__restrict__ b_in, const float *__restrict__ A_log,
           const float *__restrict__ dt_bias, float *__restrict__ S,
           float *__restrict__ out, int T) {
    extern __shared__ float sm[];
    float *sq  = sm;
    float *sk  = sq + GC_CH * GC_D;
    float *sv  = sk + GC_CH * GC_D;
    float *sd  = sv + GC_CH * GC_D;
    float *sqS = sd + GC_CH * GC_D;
    float *sA  = sqS + GC_CH * GC_D;
    float *sB  = sA + GC_CH * GC_CH;
    float *snq = sB + GC_CH * GC_CH;
    float *snk = snq + GC_CH;
    float *sLam = snk + GC_CH;
    float *sbeta = sLam + GC_CH;
    float *scp = sbeta + GC_CH;

    const int h = blockIdx.x, kh = h / 3, tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    float *Sh = S + (size_t)h * GC_D * GC_D;
    const float alog = A_log[h], dtb = dt_bias[h];

    for (int s0 = 0; s0 < T; s0 += GC_CH) {
        const int L = (T - s0) < GC_CH ? (T - s0) : GC_CH;
        if (tid < L) {
            const float av = a_in[(size_t)(s0 + tid) * GC_VH + h];
            const float bv = b_in[(size_t)(s0 + tid) * GC_VH + h];
            const float dt = av + dtb;
            const float sp = dt > 20.f ? dt : log1pf(expf(dt));
            sLam[tid] = -__expf(alog) * sp;              // g_log (negative)
            sbeta[tid] = gc_sigmoid(bv);
        }
        __syncthreads();
        if (tid == 0) { float p = 0.f; for (int i = 0; i < L; i++) { p += sLam[i]; scp[i] = p; } }
        for (int r = wid; r < L; r += 4) {
            const float *qr = qkv + (size_t)(s0 + r) * GC_DINN + kh * GC_D;
            const float *kr = qr + 2048;
            float q2 = 0.f, k2 = 0.f;
            for (int d = lane; d < GC_D; d += 32) { const float q = qr[d], k = kr[d]; q2 += q * q; k2 += k * k; }
            q2 = gc_half_sum(q2); k2 = gc_half_sum(k2);
            if (lane == 0) { snq[r] = rsqrtf(q2 + 1e-6f); snk[r] = rsqrtf(k2 + 1e-6f); }
        }
        __syncthreads();
        for (int idx = tid; idx < L * GC_D; idx += 128) {
            const int r = idx >> 7, d = idx & 127;
            const float *row = qkv + (size_t)(s0 + r) * GC_DINN;
            sq[idx] = row[kh * GC_D + d] * snq[r] * (1.f / sqrtf((float)GC_D));
            sk[idx] = row[2048 + kh * GC_D + d] * snk[r];
            sv[idx] = row[4096 + h * GC_D + d];
        }
        __syncthreads();
        for (int p = wid; p < L * L; p += 4) {
            const int i = p / L, j = p % L;
            float dA = 0.f, dB = 0.f;
            for (int d = lane; d < GC_D; d += 32) {
                dA += sk[i * GC_D + d] * sk[j * GC_D + d];
                dB += sq[i * GC_D + d] * sk[j * GC_D + d];
            }
            dA = gc_half_sum(dA); dB = gc_half_sum(dB);
            if (lane == 0) {
                if (j < i) sA[i * L + j] = sbeta[i] * __expf(scp[i] - scp[j]) * dA;
                if (j <= i) sB[i * L + j] = dB;
            }
        }
        __syncthreads();
        const int c = tid;                                     // v column
        for (int i = 0; i < L; i++) {
            float kvv = 0.f, qvv = 0.f;
            for (int r = 0; r < GC_D; r++) {
                const float s = Sh[(size_t)r * GC_D + c];
                kvv += s * sk[i * GC_D + r];
                qvv += s * sq[i * GC_D + r];
            }
            sv[i * GC_D + c] = sbeta[i] * (sv[i * GC_D + c] - __expf(scp[i]) * kvv);
            sqS[i * GC_D + c] = qvv;
        }
        __syncthreads();
        for (int i = 0; i < L; i++) {
            float val = sv[i * GC_D + c];
            for (int j = 0; j < i; j++) val -= sA[i * L + j] * sd[j * GC_D + c];
            sd[i * GC_D + c] = val;
            __syncthreads();
        }
        for (int i = 0; i < L; i++) {
            float o = __expf(scp[i]) * sqS[i * GC_D + c];
            for (int j = 0; j <= i; j++) o += __expf(scp[i] - scp[j]) * sB[i * L + j] * sd[j * GC_D + c];
            out[(size_t)(s0 + i) * (GC_VH * GC_D) + h * GC_D + c] = o;
        }
        const float cl = scp[L - 1];
        const float cle = __expf(cl);
        for (int r = 0; r < GC_D; r++) {
            float acc = cle * Sh[(size_t)r * GC_D + c];
            for (int j = 0; j < L; j++) acc += __expf(cl - scp[j]) * sk[j * GC_D + r] * sd[j * GC_D + c];
            Sh[(size_t)r * GC_D + c] = acc;
        }
        __syncthreads();
    }
}

// out = w * rms(o) * sigmoid(z), per (position, head); grid (T, 48), 64 threads (2 elems each)
__global__ void __launch_bounds__(64) k_gc_gnorm(float *__restrict__ out, const uint16_t *__restrict__ w,
                                                 const float *__restrict__ z) {
    const int t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    float *o = out + (size_t)t * (GC_VH * GC_D) + h * GC_D;
    const float *zz = z + (size_t)t * (GC_VH * GC_D) + h * GC_D;
    const float o0 = o[lane], o1 = o[lane + 64];
    float ss = qf8_wave_sum(o0 * o0 + o1 * o1);
    ss = __shfl_sync(QF8_WMASK, ss, 0);
    const float rs = rsqrtf(ss / GC_D + 1e-6f);
    o[lane]      = qf8_bf2f(w[lane])      * (o0 * rs) * gc_sigmoid(zz[lane]);
    o[lane + 64] = qf8_bf2f(w[lane + 64]) * (o1 * rs) * gc_sigmoid(zz[lane + 64]);
}

extern "C" int qf8_gdn_chunk(const Qf8GdnW *w, float *gdnS, float *convring,
                             float *qkv_raw, const float *z6144, const float *a48, const float *b48,
                             float *out6144, int T, hipStream_t s) {
    static int smem_set = 0;
    if (!smem_set) {
        if (hipFuncSetAttribute((const void *)k_gc_delta, hipFuncAttributeMaxDynamicSharedMemorySize,
                                GC_SMEM_FLOATS * 4) != hipSuccess) return -1;
        smem_set = 1;
    }
    k_gc_conv<<<(GC_DINN + 255) / 256, 256, 0, s>>>(qkv_raw, convring, w->conv1d, T);
    k_gc_delta<<<GC_VH, 128, GC_SMEM_FLOATS * 4, s>>>(qkv_raw, a48, b48, w->a_log, w->dt_bias, gdnS, out6144, T);
    k_gc_gnorm<<<dim3(T, GC_VH), 64, 0, s>>>(out6144, w->gdn_norm, z6144);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
#else
extern "C" int qf8_gdn_chunk(const Qf8GdnW *w, float *gdnS, float *convring, float *qkv_raw, const float *z6144,
                             const float *a48, const float *b48, float *out6144, int T, hipStream_t s) {
    (void)w;(void)gdnS;(void)convring;(void)qkv_raw;(void)z6144;(void)a48;(void)b48;(void)out6144;(void)T;(void)s; return 0; }
#endif
