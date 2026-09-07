// qf8_gemv_m.hip - M-row wave64 GEMV for the AMD executor (batch-first).
// Same arithmetic as k8_gemv_vec, but each weight row is loaded ONCE and dotted
// against ALL M activation rows: weight traffic scales with the matrix, not with
// M. Activations are read from global (L2-cached, M*in floats is small) so LDS
// stays near zero and wave residency stays high on gfx906 (the lesson from the
// MoE kernels: a 40 KB LDS stage collapses occupancy).
//   W [rows, in] bf16 row-major; x [M][in] f32; y [M][rows] f32.
#include "qf8_dense.h"

#ifndef QF8_M_MAX
#define QF8_M_MAX 8
#endif

__global__ void k8_gemv_vec_M(const uint16_t *__restrict__ W,
                              const float *__restrict__ x,
                              float *__restrict__ y,
                              int rows, int in, int M, float alpha, float beta) {
    int row  = blockIdx.x * 4 + threadIdx.x / QF8_WAVE;
    int lane = threadIdx.x % QF8_WAVE;
    if (row >= rows) return;
    const uint4 *wr = (const uint4 *)(W + (size_t)row * in);
    float acc[QF8_M_MAX];
    #pragma unroll
    for (int m = 0; m < QF8_M_MAX; m++) acc[m] = 0.f;
    int nv = in >> 3;
    for (int i = lane; i < nv; i += QF8_WAVE) {
        uint4 u = wr[i];                                   // weight read ONCE
        const uint16_t *e = (const uint16_t *)&u;
        float w[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) w[j] = qf8_bf2f(e[j]);
        const int xb = i << 3;
        #pragma unroll
        for (int m = 0; m < QF8_M_MAX; m++) {
            if (m < M) {
                const float *xm = x + (size_t)m * in + xb;  // global (L2)
                float a = acc[m];
                #pragma unroll
                for (int j = 0; j < 8; j++) a += w[j] * xm[j];
                acc[m] = a;
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < QF8_M_MAX; m++)
        if (m < M) {
            float v = qf8_wave_sum(acc[m]);
            if (lane == 0) {
                float *ym = y + (size_t)m * rows;
                ym[row] = (beta != 0.f) ? alpha * v + beta * ym[row] : alpha * v;
            }
        }
}

// Fused multi-projection, M rows: up to 4 matrices sharing x and `in`.
// y[k] is [M][rows[k]].
__global__ void k8_gemv_multi_M(const uint16_t *__restrict__ W0, float *__restrict__ y0, int r0,
                                const uint16_t *__restrict__ W1, float *__restrict__ y1, int r1,
                                const uint16_t *__restrict__ W2, float *__restrict__ y2, int r2,
                                const uint16_t *__restrict__ W3, float *__restrict__ y3, int r3,
                                const float *__restrict__ x, int in, int M) {
    const uint16_t *W; float *y; int rows;
    switch (blockIdx.y) {
    case 0: W = W0; y = y0; rows = r0; break;
    case 1: W = W1; y = y1; rows = r1; break;
    case 2: W = W2; y = y2; rows = r2; break;
    default: W = W3; y = y3; rows = r3; break;
    }
    int row  = blockIdx.x * 4 + threadIdx.x / QF8_WAVE;
    int lane = threadIdx.x % QF8_WAVE;
    if (row >= rows) return;
    const uint4 *wr = (const uint4 *)(W + (size_t)row * in);
    float acc[QF8_M_MAX];
    #pragma unroll
    for (int m = 0; m < QF8_M_MAX; m++) acc[m] = 0.f;
    int nv = in >> 3;
    for (int i = lane; i < nv; i += QF8_WAVE) {
        uint4 u = wr[i];
        const uint16_t *e = (const uint16_t *)&u;
        float w[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) w[j] = qf8_bf2f(e[j]);
        const int xb = i << 3;
        #pragma unroll
        for (int m = 0; m < QF8_M_MAX; m++) {
            if (m < M) {
                const float *xm = x + (size_t)m * in + xb;
                float a = acc[m];
                #pragma unroll
                for (int j = 0; j < 8; j++) a += w[j] * xm[j];
                acc[m] = a;
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < QF8_M_MAX; m++)
        if (m < M) {
            float v = qf8_wave_sum(acc[m]);
            if (lane == 0) y[(size_t)m * rows + row] = v;
        }
}

extern "C" void qf8_gemv_bf16_M(const uint16_t *W, const float *x, float *y,
                                int rows, int in, int M, float alpha, float beta, hipStream_t s) {
#ifdef QF_HOST_CHECK
    (void)W;(void)x;(void)y;(void)rows;(void)in;(void)M;(void)alpha;(void)beta;(void)s;
#else
    k8_gemv_vec_M<<<dim3((rows + 3) / 4), 4 * QF8_WAVE, 0, s>>>(W, x, y, rows, in, M, alpha, beta);
#endif
}

extern "C" void qf8_gemv_multi_M(const Qf8GemvMulti *mm, const float *x, int in, int M, hipStream_t s) {
#ifdef QF_HOST_CHECK
    (void)mm;(void)x;(void)in;(void)M;(void)s;
#else
    int maxr = 0;
    for (int k = 0; k < mm->n; k++) if (mm->rows[k] > maxr) maxr = mm->rows[k];
    dim3 g((maxr + 3) / 4, mm->n);
    k8_gemv_multi_M<<<g, 4 * QF8_WAVE, 0, s>>>(
        mm->W[0], mm->y[0], mm->rows[0],
        mm->n > 1 ? mm->W[1] : mm->W[0], mm->n > 1 ? mm->y[1] : mm->y[0], mm->n > 1 ? mm->rows[1] : 0,
        mm->n > 2 ? mm->W[2] : mm->W[0], mm->n > 2 ? mm->y[2] : mm->y[0], mm->n > 2 ? mm->rows[2] : 0,
        mm->n > 3 ? mm->W[3] : mm->W[0], mm->n > 3 ? mm->y[3] : mm->y[0], mm->n > 3 ? mm->rows[3] : 0,
        x, in, M);
#endif
}
