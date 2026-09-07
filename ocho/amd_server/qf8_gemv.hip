// qf8_gemv.hip - vectorized wave64 bf16 GEMV library for gfx906 (ROCm 5.7).
//
// All dense projections in the decode path are M=1 GEMVs over bf16 weights:
// pure HBM-bandwidth work. This beats the stage's original scalar u16 kernel
// by reading 8 bf16 per lane-load (uint4) and staging x in LDS once per block
// instead of re-reading it from global for every row.
//
// rocBLAS is used ONLY where it can win on gfx906: the 248320x2560 lm_head
// (1.27 GB bf16 per token, enough blocks to saturate; built with
// -DQF8_ROCBLAS, runtime default, QF8_LMHEAD=custom to override). Everything
// smaller stays on the custom kernel: rocBLAS gemv launch overhead and its
// non-LDS x access lose at these shapes.
#include "qf8_dense.h"
#include <stdlib.h>
#include <string.h>

#ifdef QF8_ROCBLAS
#include <rocblas/rocblas.h>
static rocblas_handle g_rb = NULL;
#endif

// ---- kernels ---------------------------------------------------------------
// One row per wave, 4 waves per block (256 threads). x staged in LDS.
// in % 64 == 0 required (640, 2560, 6144 all qualify). Dynamic smem = in*4.
__global__ void k8_gemv_vec(const uint16_t *__restrict__ W,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int rows, int in, float alpha, float beta) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int row = blockIdx.x * 4 + threadIdx.x / QF8_WAVE;
    int lane = threadIdx.x % QF8_WAVE;
    if (row >= rows) return;
    const uint4 *wr = (const uint4 *)(W + (size_t)row * in);
    float acc = 0.f;
    int nv = in >> 3;                       // uint4s per row (in%8==0)
    for (int i = lane; i < nv; i += QF8_WAVE) {
        uint4 u = wr[i];
        const uint16_t *e = (const uint16_t *)&u;
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc += qf8_bf2f(e[j]) * xs[(i << 3) + j];
    }
    acc = qf8_wave_sum(acc);
    if (lane == 0)
        y[row] = (beta != 0.f) ? alpha * acc + beta * y[row] : alpha * acc;
}

// Fused multi-projection: grid.y = matrix index, grid.x covers the largest
// row count; blocks past a matrix's rows exit immediately. One x staging per
// block, one launch for up to 4 projections that share x and `in`.
__global__ void k8_gemv_multi(const uint16_t *__restrict__ W0, float *__restrict__ y0, int r0,
                              const uint16_t *__restrict__ W1, float *__restrict__ y1, int r1,
                              const uint16_t *__restrict__ W2, float *__restrict__ y2, int r2,
                              const uint16_t *__restrict__ W3, float *__restrict__ y3, int r3,
                              const float *__restrict__ x, int in) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < in; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const uint16_t *W;
    float *y;
    int rows;
    switch (blockIdx.y) {
    case 0: W = W0; y = y0; rows = r0; break;
    case 1: W = W1; y = y1; rows = r1; break;
    case 2: W = W2; y = y2; rows = r2; break;
    default: W = W3; y = y3; rows = r3; break;
    }
    int row = blockIdx.x * 4 + threadIdx.x / QF8_WAVE;
    int lane = threadIdx.x % QF8_WAVE;
    if (row >= rows) return;
    const uint4 *wr = (const uint4 *)(W + (size_t)row * in);
    float acc = 0.f;
    int nv = in >> 3;
    for (int i = lane; i < nv; i += QF8_WAVE) {
        uint4 u = wr[i];
        const uint16_t *e = (const uint16_t *)&u;
        #pragma unroll
        for (int j = 0; j < 8; j++)
            acc += qf8_bf2f(e[j]) * xs[(i << 3) + j];
    }
    acc = qf8_wave_sum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void k8_zero(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}
__global__ void k8_cvt(float *out, const uint16_t *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = qf8_bf2f(in[i]);
}
__global__ void k8_cvt_back(uint16_t *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = qf8_f2bf(in[i]);
}
__global__ void k8_repeat4(float *R, const float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < QF8_RSIZE) R[i] = x[i % QF8_NEMBD];
}

// ---- host wrappers ----------------------------------------------------------
static int smem_for(int in) { return in * 4; }

void qf8_gemv_bf16(const uint16_t *W, const float *x, float *y,
                   int rows, int in, float alpha, float beta, hipStream_t s) {
    QF8_LAUNCH(k8_gemv_vec, (rows + 3) / 4, 256, smem_for(in), s,
               W, x, y, rows, in, alpha, beta);
}

void qf8_gemv_multi(const Qf8GemvMulti *m, const float *x, int in, hipStream_t s) {
    const uint16_t *W[4] = {0, 0, 0, 0};
    float *y[4] = {0, 0, 0, 0};
    int r[4] = {0, 0, 0, 0}, mx = 1;
    for (int i = 0; i < m->n && i < 4; i++) {
        W[i] = m->W[i]; y[i] = m->y[i]; r[i] = m->rows[i];
        if (r[i] > mx) mx = r[i];
    }
    dim3 g((mx + 3) / 4, m->n < 4 ? m->n : 4);
    QF8_LAUNCH(k8_gemv_multi, g, 256, smem_for(in), s,
               W[0], y[0], r[0], W[1], y[1], r[1],
               W[2], y[2], r[2], W[3], y[3], r[3], x, in);
}

void qf8_lm_head(const uint16_t *W, const float *x, float *logits,
                 uint16_t *x_bf16_scratch, hipStream_t s) {
#ifdef QF8_ROCBLAS
    static int use_rb = -1;
    if (use_rb < 0) {
        const char *e = getenv("QF8_LMHEAD");
        use_rb = (e && !strcmp(e, "custom")) ? 0 : 1;
    }
    if (use_rb) {
        if (!g_rb) rocblas_create_handle(&g_rb);
        rocblas_set_stream(g_rb, s);
        // gemv_ex: y = W.x, W bf16 row-major [248320,2560] seen by rocBLAS as
        // column-major [2560,248320] transposed problem.
        QF8_LAUNCH(k8_cvt_back, (QF8_NEMBD + 255) / 256, 256, 0, s,
                   x_bf16_scratch, x, QF8_NEMBD);
        float one = 1.f, zero = 0.f;
        rocblas_gemv_ex(g_rb, rocblas_operation_transpose,
                        QF8_NVOCAB, QF8_NEMBD,
                        &one, W, rocblas_datatype_bf16_r, QF8_NVOCAB,
                        x_bf16_scratch, rocblas_datatype_bf16_r, 1,
                        &zero, logits, rocblas_datatype_f32_r, 1,
                        logits, rocblas_datatype_f32_r, 1,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        return;
    }
#else
    (void)x_bf16_scratch;
#endif
    qf8_gemv_bf16(W, x, logits, QF8_NVOCAB, QF8_NEMBD, 1.f, 0.f, s);
}

void qf8_zero(float *p, int n, hipStream_t s) {
    QF8_LAUNCH(k8_zero, (n + 255) / 256, 256, 0, s, p, n);
}
void qf8_cvt_bf16_f32(float *out, const uint16_t *in, int n, hipStream_t s) {
    QF8_LAUNCH(k8_cvt, (n + 255) / 256, 256, 0, s, out, in, n);
}
void qf8_repeat4(float *R, const float *x, hipStream_t s) {
    QF8_LAUNCH(k8_repeat4, QF8_RSIZE / 256, 256, 0, s, R, x);
}
