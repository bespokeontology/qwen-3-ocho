// qf8_hc.hip - fused hyper-connection (GatedResidual) pre/post mixers, gfx906.
//
// Per SEMANTICS.md (GatedResidual, every layer, attn and ffn side):
//   1. n = rmsnorm(R)          zero-centered (1+w), group_size 2560 over 10240
//   2. m = sigmoid(up(silu(down(n)/4)))    down [10240->320], up [320->10240]
//   3. x = mean_c(m_c * n_c)               per-stream mix -> token mixer input
//   4. s_c = 2*sigmoid(block_inject(n)/4)  inject [10240->4]
//   5. R'_c = R_c + s_c * y                after the mixer (qf8_stream_inject)
//
// The old stage used 5 launches with a k_zero + atomicAdd accumulation in
// k_hc_up. Here: 3 launches, no atomics, no zeroing. k8_hc_upmix writes each
// mixed[j] exactly once (one wave owns output j, d[320] cached in LDS).
#include "qf8_dense.h"

// (1) zero-centered grouped RMSNorm, one 256-thread block per 2560 group.
__device__ __forceinline__ void k8_hc_norm_body(float *__restrict__ out, const float *__restrict__ x,
                           const uint16_t *__restrict__ w) {
    int g0 = blockIdx.x * QF8_NEMBD;
    float acc = 0.f;
    for (int i = threadIdx.x; i < QF8_NEMBD; i += blockDim.x) {
        float v = x[g0 + i];
        acc += v * v;
    }
    acc = qf8_wave_sum(acc);
    __shared__ float red[4];
    __shared__ float rstd;
    int lane = threadIdx.x % QF8_WAVE, wid = threadIdx.x / QF8_WAVE;
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        rstd = rsqrtf((red[0] + red[1] + red[2] + red[3]) / QF8_NEMBD + 1e-6f);
    __syncthreads();
    float r = rstd;
    for (int i = threadIdx.x; i < QF8_NEMBD; i += blockDim.x)
        out[g0 + i] = x[g0 + i] * r * (1.f + qf8_bf2f(w[g0 + i]));
}
__global__ void k8_hc_norm(float *__restrict__ out, const float *__restrict__ x,
                           const uint16_t *__restrict__ w) { k8_hc_norm_body(out, x, w); }


// (2) down + inject in one launch: grid 324 waves. Rows 0..319:
// d = silu(dot(wd[row], n)/4); rows 320..323: inj[c] = 2*sigmoid(dot/4).
// n (40 KiB) stays L2-hot across the 324 blocks; weights are streamed once.
__device__ __forceinline__ void k8_hc_down_inj_body(const float *__restrict__ n,
                               const uint16_t *__restrict__ wd,
                               const uint16_t *__restrict__ wi,
                               float *__restrict__ d320, float *__restrict__ inj) {
    int row = blockIdx.x, lane = threadIdx.x;
    const uint16_t *wr;
    if (row < QF8_HCL) wr = wd + (size_t)row * QF8_RSIZE;
    else               wr = wi + (size_t)(row - QF8_HCL) * QF8_RSIZE;
    float acc = 0.f;
    for (int i = lane; i < QF8_RSIZE; i += QF8_WAVE)
        acc += qf8_bf2f(wr[i]) * n[i];
    acc = qf8_wave_sum(acc);
    if (lane == 0) {
        if (row < QF8_HCL) d320[row] = qf8_silu(acc / 4.f);
        else               inj[row - QF8_HCL] = 2.f * qf8_sig(acc / 4.f);
    }
}
__global__ void k8_hc_down_inj(const float *__restrict__ n,
                               const uint16_t *__restrict__ wd,
                               const uint16_t *__restrict__ wi,
                               float *__restrict__ d320, float *__restrict__ inj) { k8_hc_down_inj_body(n, wd, wi, d320, inj); }


// (3) up + per-stream mean, one wave per output j (grid 2560 x 64 lanes).
// mixed[j] = (1/4) * sum_c sigmoid(up[c*2560+j] . d) * n[c*2560+j].
// d is tiny (320 floats) -> LDS once per block. No atomics: single writer.
__device__ __forceinline__ void k8_hc_upmix_body(const float *__restrict__ n,
                            const uint16_t *__restrict__ wu,
                            const float *__restrict__ d,
                            float *__restrict__ mixed) {
    __shared__ float ds[QF8_HCL];
    int lane = threadIdx.x;
    for (int l = lane; l < QF8_HCL; l += QF8_WAVE) ds[l] = d[l];
    __syncthreads();
    int j = blockIdx.x;
    float sum = 0.f;
    #pragma unroll
    for (int c = 0; c < QF8_HCC; c++) {
        const uint16_t *wr = wu + (size_t)(c * QF8_NEMBD + j) * QF8_HCL;
        float acc = 0.f;
        #pragma unroll
        for (int l5 = 0; l5 < QF8_HCL / QF8_WAVE; l5++)
            acc += qf8_bf2f(wr[lane + l5 * QF8_WAVE]) * ds[lane + l5 * QF8_WAVE];
        acc = qf8_wave_sum(acc);
        float m = qf8_sig(__shfl_sync(QF8_WMASK, acc, 0));
        if (lane == 0) sum += m * n[c * QF8_NEMBD + j];
    }
    if (lane == 0) mixed[j] = sum / 4.f;
}
__global__ void k8_hc_upmix(const float *__restrict__ n,
                            const uint16_t *__restrict__ wu,
                            const float *__restrict__ d,
                            float *__restrict__ mixed) { k8_hc_upmix_body(n, wu, d, mixed); }


__global__ void k8_stream_inject(float *__restrict__ R, const float *__restrict__ y,
                                 const float *__restrict__ inj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < QF8_RSIZE) R[i] += inj[i / QF8_NEMBD] * y[i % QF8_NEMBD];
}

// ---- host wrappers ----------------------------------------------------------

// ---- M-row hyper-connection (blockIdx.y = request row) ---------------------
__global__ void k8_hc_norm_M(float *__restrict__ out, const float *__restrict__ x,
                             const uint16_t *__restrict__ w) {
    const size_t off = (size_t)blockIdx.y * QF8_RSIZE;
    k8_hc_norm_body(out + off, x + off, w);
}
__global__ void k8_hc_down_inj_M(const float *__restrict__ n, const uint16_t *__restrict__ wd,
                                 const uint16_t *__restrict__ wi, float *__restrict__ d320,
                                 float *__restrict__ inj) {
    const size_t rs = (size_t)blockIdx.y * QF8_RSIZE;
    k8_hc_down_inj_body(n + rs, wd, wi, d320 + (size_t)blockIdx.y * QF8_HCL,
                        inj + (size_t)blockIdx.y * QF8_HCC);
}
__global__ void k8_hc_upmix_M(const float *__restrict__ n, const uint16_t *__restrict__ wu,
                              const float *__restrict__ d320, float *__restrict__ mixed) {
    k8_hc_upmix_body(n + (size_t)blockIdx.y * QF8_RSIZE, wu,
                     d320 + (size_t)blockIdx.y * QF8_HCL,
                     mixed + (size_t)blockIdx.y * QF8_NEMBD);
}
void qf8_hc_pre_M(float *normed, float *hc_d, float *mixed, float *inj,
                  const float *R, const uint16_t *wn, const uint16_t *wd,
                  const uint16_t *wu, const uint16_t *wi, int M, hipStream_t s) {
#ifdef QF_HOST_CHECK
    (void)normed;(void)hc_d;(void)mixed;(void)inj;(void)R;(void)wn;(void)wd;(void)wu;(void)wi;(void)M;(void)s;
#else
    k8_hc_norm_M<<<dim3(QF8_RSIZE / QF8_NEMBD, M), 256, 0, s>>>(normed, R, wn);
    k8_hc_down_inj_M<<<dim3(QF8_HCL + (wi ? QF8_HCC : 0), M), QF8_WAVE, 0, s>>>(
        normed, wd, wi ? wi : wd, hc_d, inj ? inj : hc_d);
    k8_hc_upmix_M<<<dim3(QF8_NEMBD, M), QF8_WAVE, 0, s>>>(normed, wu, hc_d, mixed);
#endif
}

void qf8_hc_pre(float *normed, float *hc_d, float *mixed, float *inj,
                const float *R, const uint16_t *wn, const uint16_t *wd,
                const uint16_t *wu, const uint16_t *wi, hipStream_t s) {
    QF8_LAUNCH(k8_hc_norm, QF8_RSIZE / QF8_NEMBD, 256, 0, s, normed, R, wn);
    QF8_LAUNCH(k8_hc_down_inj, QF8_HCL + (wi ? QF8_HCC : 0), QF8_WAVE, 0, s,
               normed, wd, wi ? wi : wd, hc_d, inj ? inj : hc_d);
    QF8_LAUNCH(k8_hc_upmix, QF8_NEMBD, QF8_WAVE, 0, s, normed, wu, hc_d, mixed);
}

void qf8_stream_inject(float *R, const float *y, const float *inj, hipStream_t s) {
    QF8_LAUNCH(k8_stream_inject, QF8_RSIZE / 256, 256, 0, s, R, y, inj);
}
