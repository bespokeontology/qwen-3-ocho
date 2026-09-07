// qf_fp4mma.cu - grouped routed-expert GEMV on native block-scaled FP4 MMA.
//
// Built on tools/mma_probe.cu, which validates ONE
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
// against a scalar reference: cosine 1.000000, maxdiff 0.0000, with
// ADJACENT-PAIR nibble packing and NO repack. That is the checkpoint's own
// layout (verified against the BF16 original: corr 0.9955 adjacent vs 0.1420
// interleaved), so the weights are already in operand order.
//
// Why the previous attempt (qf_fp4tc.cu) computes exactly zero: it fed the
// warp-wide instruction operands from a subset of lanes -
//     if (lane == 0) sfb = ...;        // one lane supplies the B scale
//     if (g == 0)    { b0 = ..; b1 = ..; }
// on the theory that a batch-1 GEMV only needs column 0. mma.sync is warp-wide:
// every lane must supply valid operands for its own quad's column. Here the
// activation is replicated across all 8 B columns, so it does.
//
// Fragment map (confirmed empirically by the probe), lane l = 4*g + p:
//   A: a0 = row g,   K 8p..8p+7      a2 = row g,   K 32+8p..
//      a1 = row g+8, K 8p..8p+7      a3 = row g+8, K 32+8p..
//   B: b0 = col g, K 8p..            b1 = col g, K 32+8p..
//   SFA: lane supplies row (p&1 ? g+8 : g); SFB: lane supplies col g
//   D:  d0=(g,2p) d1=(g,2p+1) d2=(g+8,2p) d3=(g+8,2p+1)
#include "../qwenflash.h"
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NEMBD
#define NEMBD 2560
#endif
#ifndef NFF
#define NFF 640
#endif
#ifndef NEXPUSED
#define NEXPUSED 10
#endif

__device__ __forceinline__ void qf_mma_fp4(float &d0, float &d1, float &d2, float &d3,
                                           uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1,
                                           uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0,0}, {%11}, {0,0};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
#else
    // NOT zero: a dead path must be visible, not indistinguishable from a real
    // result. nvcc -arch=sm_121a also compiles a generic compute_121 PTX pass
    // without the arch-feature macro, and qf_fp4tc.cu writing 0 here is exactly
    // how a dead MMA shipped as production.
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    d0 = d1 = d2 = d3 = nanf("");
#endif
}

// E2M1 / UE4M3 quantization of the activation into the MMA's B operand form.
__device__ __forceinline__ uint32_t qf_e2m1_enc(float x) {
    float a = fabsf(x);
    uint32_t c = a < 0.25f ? 0u : a < 0.75f ? 1u : a < 1.25f ? 2u
               : a < 1.75f ? 3u : a < 2.5f  ? 4u : a < 3.5f  ? 5u
               : a < 5.f   ? 6u : 7u;
    return ((__float_as_uint(x) >> 28) & 8u) | c;
}
__device__ __forceinline__ uint32_t qf_ue4m3_enc_ceil(float v) {
    if (!(v > 0.f)) return 0u;
    uint32_t b = __float_as_uint(v);
    int e32 = (int)(b >> 23) - 127;
    if (e32 < -6) { uint32_t m = (uint32_t)ceilf(v * 512.f); if (!m) m = 1; return m > 7 ? 8u : m; }
    int e = e32 + 7;
    uint32_t m = ((b & 0x7FFFFFu) + 0xFFFFFu) >> 20;
    if (m == 8u) { e++; m = 0; }
    return e > 15 ? 0x7Eu : (uint32_t)((e << 3) | m);
}
__device__ __forceinline__ float qf_ue4m3_dec(uint32_t b) {
    uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
}

// Quantize x[K] into packed E2M1 (K/2 bytes) + UE4M3 block scales (K/16 bytes),
// replicated across all 8 MMA columns so every lane holds a valid B fragment.
template<int K>
__global__ void k_fp4mma_quant(const float *__restrict__ x,
                               uint8_t *__restrict__ xq, uint8_t *__restrict__ xs) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;   // one 16-value block
    if (blk >= K / 16) return;
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; j++) a = fmaxf(a, fabsf(x[blk * 16 + j]));
    uint32_t sb = qf_ue4m3_enc_ceil(a * (1.f / 6.f));
    float inv = 1.f / fmaxf(qf_ue4m3_dec(sb), 1e-30f);
    uint8_t packed[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        packed[j] = (uint8_t)(qf_e2m1_enc(x[blk * 16 + 2 * j] * inv)
                            | (qf_e2m1_enc(x[blk * 16 + 2 * j + 1] * inv) << 4));
    // replicate into all 8 columns: col c row-major stride K/2
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        uint8_t *dst = xq + (size_t)c * (K / 2) + blk * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) dst[j] = packed[j];
        xs[(size_t)c * (K / 16) + blk] = (uint8_t)sb;
    }
}

// Grouped expert GEMV. grid = (rows/16, NEXPUSED); one warp per 16-row tile.
// y_all is [NEXPUSED][rows].
template<int K>
__global__ __launch_bounds__(128)
void k_fp4mma_gemv(const uint8_t *__restrict__ W, size_t w_stride,
                   const uint8_t *__restrict__ S, size_t s_stride,
                   const int *__restrict__ sel, const int *__restrict__ slots,
                   const float *__restrict__ s2tab,
                   const uint8_t *__restrict__ xq, const uint8_t *__restrict__ xs,
                   float *__restrict__ y_all, int rows, int x_per_expert) {
    const int kk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x / 32) + warp) * 16;
    if (row0 >= rows) return;
    const int slot = slots[kk];
    const uint8_t *Wp = W + (size_t)slot * w_stride;
    const uint8_t *Sp = S + (size_t)slot * s_stride;
    const uint8_t *bq = xq + (x_per_expert ? (size_t)kk * 8 * (K / 2) : 0);
    const uint8_t *bs = xs + (x_per_expert ? (size_t)kk * 8 * (K / 16) : 0);

    const int g = lane >> 2, p = lane & 3;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const int rA = row0 + g, rB = row0 + g + 8;
    for (int k0 = 0; k0 < K; k0 += 64) {
        const size_t ka = (size_t)k0 / 2;
        uint32_t a0 = *(const uint32_t *)(Wp + (size_t)rA * (K / 2) + ka + 4 * p);
        uint32_t a1 = *(const uint32_t *)(Wp + (size_t)rB * (K / 2) + ka + 4 * p);
        uint32_t a2 = *(const uint32_t *)(Wp + (size_t)rA * (K / 2) + ka + 16 + 4 * p);
        uint32_t a3 = *(const uint32_t *)(Wp + (size_t)rB * (K / 2) + ka + 16 + 4 * p);
        uint32_t b0 = *(const uint32_t *)(bq + (size_t)g * (K / 2) + ka + 4 * p);
        uint32_t b1 = *(const uint32_t *)(bq + (size_t)g * (K / 2) + ka + 16 + 4 * p);
        uint32_t sfa = *(const uint32_t *)(Sp + (size_t)((p & 1) ? rB : rA) * (K / 16) + k0 / 16);
        uint32_t sfb = *(const uint32_t *)(bs + (size_t)g * (K / 16) + k0 / 16);
        qf_mma_fp4(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb);
    }
    // Column 0 of D holds the GEMV result: lanes with p == 0 own (g,0) and (g+8,0).
    if (p == 0) {
        const float s2 = s2tab[sel[kk]];
        float *y = y_all + (size_t)kk * rows;
        if (rA < rows) y[rA] = d0 * s2;
        if (rB < rows) y[rB] = d2 * s2;
    }
}

extern "C" {
int qf_fp4mma_available(void) {
    int dev = -1, maj = 0, min = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev);
    return (maj == 12 && min <= 1);
}

void qf_fp4mma_quant_launch(const float *x, int K, uint8_t *xq, uint8_t *xs, cudaStream_t s) {
    if (K == 2560) k_fp4mma_quant<2560><<<(2560 / 16 + 127) / 128, 128, 0, s>>>(x, xq, xs);
    else if (K == 640) k_fp4mma_quant<640><<<(640 / 16 + 127) / 128, 128, 0, s>>>(x, xq, xs);
}

void qf_fp4mma_gemv_launch(const void *W, size_t w_stride, const void *S, size_t s_stride,
                           const int *sel, const int *slots, const float *s2tab,
                           const void *xq, const void *xs, float *y_all,
                           int rows, int K, int x_per_expert, cudaStream_t s) {
    dim3 g((rows + 63) / 64, NEXPUSED);   // 4 warps/block x 16 rows
    if (K == 2560)
        k_fp4mma_gemv<2560><<<g, 128, 0, s>>>((const uint8_t *)W, w_stride,
            (const uint8_t *)S, s_stride, sel, slots, s2tab,
            (const uint8_t *)xq, (const uint8_t *)xs, y_all, rows, x_per_expert);
    else if (K == 640)
        k_fp4mma_gemv<640><<<g, 128, 0, s>>>((const uint8_t *)W, w_stride,
            (const uint8_t *)S, s_stride, sel, slots, s2tab,
            (const uint8_t *)xq, (const uint8_t *)xs, y_all, rows, x_per_expert);
}
}
