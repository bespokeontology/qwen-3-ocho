// qf_fp4tc.cu - Native GB10 (sm_121a) block-scaled FP4 tensor-core MoE backend.
//
// Production expert GEMV/GEMM path for the NVFP4 checkpoint: every routed
// expert projection runs through the warp-level block-scaled FP4 MMA
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
//     .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//
// which consumes packed E2M1 pairs plus per-16-value UE4M3 scale factors in
// hardware (weights are never software-dequantized). One kernel launch covers
// all NEXPUSED selected experts for a projection stage; the per-token layer
// MoE is three launches total (quant + gate/up + down-accum).
//
// Fragment layouts (PTX ISA 9.x "Block Scaling for mma.sync" / m16n8k64
// fragment figures; cross-checked against CUTLASS mma_traits_sm120.hpp):
//   lane l = 4*g + p (g = quad 0..7, p = lane-in-quad 0..3)
//   A (16x64 e2m1, row-major, 8 nibbles per .b32):
//     a0 = row g,   K 8p..8p+7        a2 = row g,   K 32+8p..32+8p+7
//     a1 = row g+8, K 8p..8p+7        a3 = row g+8, K 32+8p..32+8p+7
//   B (64x8 e2m1, col-major):
//     b0 = col g, K 8p..8p+7          b1 = col g, K 32+8p..32+8p+7
//   SFA (.b32 = 4 UE4M3 bytes, byte j = K-block 16j..16j+15 of the K64 tile):
//     with {byte-id-a=0, thread-id-a=0}: lane p==0 supplies row g, p==1 row g+8
//   SFB: with {byte-id-b=0, thread-id-b=0}: lane p==0 supplies col g
//   C/D (16x8 f32): d0=(g,2p) d1=(g,2p+1) d2=(g+8,2p) d3=(g+8,2p+1)
// Decode is M=1 (one token), so B column 0 carries the activation and columns
// 1..7 are zero; the kernel is memory-bound and the tensor core's job is
// hardware dequant + MAC of the FP4 stream, not FLOP density.
#include "../qwenflash.h"
#include "../qf_fp4tc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NEMBD
#define NEMBD 2560
#endif
#ifndef NFF
#define NFF 640
#endif
#ifndef NEXPUSED
#define NEXPUSED 10
#endif

static_assert(NEMBD % 64 == 0 && NFF % 64 == 0, "NVFP4 MMA needs K%64==0 and rows%16==0");
static_assert(NFF % 64 == 0, "gate/up CTA covers 64 rows");

// ---------------------------------------------------------------------------
// Scalar helpers (quant path only; weights are never dequantized in software)
// ---------------------------------------------------------------------------

// UE4M3 decode, bit-identical to ue4m3_bits() in qf.cu.
__device__ __forceinline__ float ue4m3_dec(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                : (float)m * 0x1p-9f;
    return (b == 0x7Fu) ? __int_as_float(0x7F800000) : v;
}

// UE4M3 encode, round-UP so that |x / dec(s)| <= 6 always holds (e2m1 max
// magnitude is 6; satfinite conversion is then a no-op). 0x7F (INF) is never
// emitted; inputs above 448 saturate at 0x7E.
__device__ __forceinline__ uint32_t ue4m3_enc_ceil(float v) {
    if (!(v > 0.f)) return 0u;
    uint32_t b = __float_as_uint(v);
    int e32 = (int)(b >> 23) - 127;          // v in [2^e32, 2^(e32+1))
    if (e32 < -6) {                          // subnormals: m * 2^-9, m in 1..7
        uint32_t m = (uint32_t)ceilf(v * 512.f);
        if (m == 0) m = 1;
        if (m > 7) return 0x08u;             // == 2^-6, smallest normal
        return m;
    }
    int e = e32 + 7;                         // normal (8+m)*2^(e-10), base 2^e32
    uint32_t m = ((b & 0x7FFFFFu) + 0xFFFFFu) >> 20;  // ceil(8 * frac(v / 2^e32))
    if (m == 8u) { e++; m = 0; }
    if (e > 15) return 0x7Eu;                // saturate below INF
    return ((uint32_t)e << 3) | m;
}

// E2M1 encode, round-to-nearest (ties away) over {0,.5,1,1.5,2,3,4,6} + sign.
// Integer-only so the quant path has no dependency on cvt.e2m1x2 availability.
__device__ __forceinline__ uint32_t e2m1_enc(float x) {
    float ax = fabsf(x);
    uint32_t c = ax < 0.25f ? 0u : ax < 0.75f ? 1u : ax < 1.25f ? 2u
               : ax < 1.75f ? 3u : ax < 2.5f  ? 4u : ax < 3.5f  ? 5u
               : ax < 5.f   ? 6u : 7u;
    return ((__float_as_uint(x) >> 28) & 8u) | c;   // sign bit -> e2m1 bit 3
}

// ---------------------------------------------------------------------------
// The native block-scaled FP4 MMA (sm_120a / sm_121a only)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_mxf4nvf4(float &d0, float &d1, float &d2, float &d3,
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
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)sfa; (void)sfb;
    d0 = d1 = d2 = d3 = 0.f;   // unreachable: gated by qf_fp4tc_supported()
#endif
}

// Doorbell wait, same protocol as k_nvfp4_gemv_slot: spin on the slot's ready
// flag with a bounded guard, acquire-fence before touching the data.
__device__ __forceinline__ void fp4tc_wait_slot(const uint32_t *flag, int *route_err) {
    unsigned long long spins = 0;
    while (*(const volatile uint32_t *)flag == 0u) {
        if (++spins > (1ull << 26)) { *route_err = 1; break; }
        __nanosleep(256);
    }
    __threadfence();
}

// ---------------------------------------------------------------------------
// One-time nibble repack: checkpoint -> native MMA operand order.
//
// Checkpoint packs each 16-value scale group as 8 bytes with value j in the
// low nibble of byte j and value j+8 in the high nibble (interleaved). The
// MMA consumes adjacent pairs: byte i = {v[2i] lo, v[2i+1] hi}. Per 8-byte
// group the transform is:
//   out[i]   = lo(in[2i]) | lo(in[2i+1]) << 4      (i = 0..3)
//   out[4+i] = hi(in[2i]) | hi(in[2i+1]) << 4
// One thread owns one 8-byte group and reads it fully before writing it back,
// so the in-place update is race-free. NOT idempotent: apply once per fill.
// ---------------------------------------------------------------------------
__global__ void k_fp4tc_repack(uint8_t *__restrict__ data, size_t n8) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint64_t v = ((uint64_t *)data)[i];
    uint64_t o = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint64_t b0 = (v >> (16 * j)) & 0xFFu;
        uint64_t b1 = (v >> (16 * j + 8)) & 0xFFu;
        o |= ((b0 & 0xFu) | ((b1 & 0xFu) << 4)) << (8 * j);
        o |= (((b0 >> 4) | (b1 & 0xF0u))) << (8 * (j + 4));
    }
    ((uint64_t *)data)[i] = o;
}

// The repack is DISABLED by default (QF_FP4TC_REPACK=1 to force it on).
//
// It exists because the checkpoint was believed to pack each 16-value group as
// "value j in the low nibble of byte j, value j+8 in the high nibble". That is
// false - verified against the BF16 original (models/Qwen3.8-Flash-Next):
//     adjacent pairs {v[2i] lo, v[2i+1] hi} -> corr 0.9955
//     interleaved    {v[j] lo,  v[j+8] hi}  -> corr 0.1420
//
// And adjacent pairs is exactly what the MMA wants. Per this file's own
// fragment map, a0 is a .b32 holding K values 8p..8p+7 of row g; with
// adjacent-pair packing that is literally *(uint32_t *)(row + 4p). So the
// checkpoint is already in operand order and the repack SCRAMBLES it.
static int qf_fp4tc_repack_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QF_FP4TC_REPACK"); v = (e && e[0] == '1'); }
    return v;
}

void qf_fp4tc_repack_slab(void *w, size_t bytes_per_expert, int nslots, cudaStream_t s) {
    if (!qf_fp4tc_repack_on()) return;
    size_t n8 = bytes_per_expert * (size_t)nslots / 8;
    k_fp4tc_repack<<<(unsigned)((n8 + 255) / 256), 256, 0, s>>>((uint8_t *)w, n8);
}

void qf_fp4tc_repack_slot3(void *gate, void *up, void *down,
                           size_t bytes_per_expert, int slot, cudaStream_t s) {
    if (!qf_fp4tc_repack_on()) return;
    size_t n8 = bytes_per_expert / 8;
    unsigned nb = (unsigned)((n8 + 255) / 256);
    size_t off = (size_t)slot * bytes_per_expert;
    k_fp4tc_repack<<<nb, 256, 0, s>>>((uint8_t *)gate + off, n8);
    k_fp4tc_repack<<<nb, 256, 0, s>>>((uint8_t *)up + off, n8);
    k_fp4tc_repack<<<nb, 256, 0, s>>>((uint8_t *)down + off, n8);
}

// ---------------------------------------------------------------------------
// Activation quantization: fp32 x[N] -> packed E2M1 pairs + UE4M3 block
// scales, dynamic per-16-value absmax. Thread t handles one (block, j) pair
// where j indexes the byte inside the 8-byte group; the 8 threads of a group
// are warp-consecutive so the absmax reduction is a width-8 shfl.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fp4tc_quant_block(const float *x, int n,
                                                  uint8_t *xq, uint8_t *xs) {
    const int nblk = n >> 4;
    const int iters = (nblk * 8 + (int)blockDim.x - 1) / (int)blockDim.x;
    for (int it = 0; it < iters; it++) {
        int t = it * (int)blockDim.x + (int)threadIdx.x;
        int g = t >> 3, j = t & 7;
        float x0 = 0.f, x1 = 0.f;
        if (g < nblk) { x0 = x[g * 16 + 2 * j]; x1 = x[g * 16 + 2 * j + 1]; }
        float a = fmaxf(fabsf(x0), fabsf(x1));
        #pragma unroll
        for (int o = 4; o; o >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o, 8));
        if (g < nblk) {
            uint32_t sb = ue4m3_enc_ceil(a * (1.f / 6.f));
            float sd = ue4m3_dec(sb);
            float inv = sd > 0.f ? 1.f / sd : 0.f;
            xq[g * 8 + j] = (uint8_t)(e2m1_enc(x0 * inv) | (e2m1_enc(x1 * inv) << 4));
            if (j == 0) xs[g] = (uint8_t)sb;
        }
    }
}

__global__ void k_fp4tc_quant(const float *__restrict__ x, int n,
                              uint8_t *__restrict__ xq, uint8_t *__restrict__ xs) {
    fp4tc_quant_block(x, n, xq, xs);
}

// ---------------------------------------------------------------------------
// Grouped MMA core: one warp computes a 16-row strip of (W_slot @ x_q) for
// the full K. Weights/scales stream from global (read exactly once per token,
// evict-first); the quantized activation comes from shared memory.
// ---------------------------------------------------------------------------
template<int K>
__device__ __forceinline__ void fp4tc_mma_strip(const uint8_t *__restrict__ W,
                                                const uint8_t *__restrict__ S,
                                                const uint8_t *__restrict__ xq,
                                                const uint8_t *__restrict__ xs,
                                                int row0, int lane,
                                                float &out0, float &out1) {
    const int g = lane >> 2, p = lane & 3;
    const uint8_t *wr0 = W + (size_t)(row0 + g) * (K / 2);
    const uint8_t *wr1 = W + (size_t)(row0 + g + 8) * (K / 2);
    const uint8_t *sr0 = S + (size_t)(row0 + g) * (K / 16);
    const uint8_t *sr1 = S + (size_t)(row0 + g + 8) * (K / 16);
    // Two independent accumulator sets: even and odd K-blocks. One chained set
    // serializes 40 MMAs on the accumulator dependency; two sets let the MMA
    // pipeline overlap with the next fragment loads. Summed at the end.
    float d0a = 0.f, d1a = 0.f, d2a = 0.f, d3a = 0.f;
    float d0b = 0.f, d1b = 0.f, d2b = 0.f, d3b = 0.f;
    const int NK = K / 64;
    int kb = 0;
    for (; kb + 1 < NK; kb += 2) {
        uint32_t a0 = __ldcs((const uint32_t *)(wr0 + kb * 32 + 4 * p));
        uint32_t a1 = __ldcs((const uint32_t *)(wr1 + kb * 32 + 4 * p));
        uint32_t a2 = __ldcs((const uint32_t *)(wr0 + kb * 32 + 16 + 4 * p));
        uint32_t a3 = __ldcs((const uint32_t *)(wr1 + kb * 32 + 16 + 4 * p));
        uint32_t sfa = __ldcs((const uint32_t *)((p & 1 ? sr1 : sr0) + kb * 4));
        uint32_t b0 = 0, b1 = 0, sfb = 0;
        if (g == 0) {
            b0 = *(const uint32_t *)(xq + kb * 32 + 4 * p);
            b1 = *(const uint32_t *)(xq + kb * 32 + 16 + 4 * p);
        }
        if (lane == 0)
            sfb = *(const uint32_t *)(xs + kb * 4);
        mma_mxf4nvf4(d0a, d1a, d2a, d3a, a0, a1, a2, a3, b0, b1, sfa, sfb);

        uint32_t c0 = __ldcs((const uint32_t *)(wr0 + (kb + 1) * 32 + 4 * p));
        uint32_t c1 = __ldcs((const uint32_t *)(wr1 + (kb + 1) * 32 + 4 * p));
        uint32_t c2 = __ldcs((const uint32_t *)(wr0 + (kb + 1) * 32 + 16 + 4 * p));
        uint32_t c3 = __ldcs((const uint32_t *)(wr1 + (kb + 1) * 32 + 16 + 4 * p));
        uint32_t sfc = __ldcs((const uint32_t *)((p & 1 ? sr1 : sr0) + (kb + 1) * 4));
        uint32_t e0 = 0, e1 = 0, sfd = 0;
        if (g == 0) {
            e0 = *(const uint32_t *)(xq + (kb + 1) * 32 + 4 * p);
            e1 = *(const uint32_t *)(xq + (kb + 1) * 32 + 16 + 4 * p);
        }
        if (lane == 0)
            sfd = *(const uint32_t *)(xs + (kb + 1) * 4);
        mma_mxf4nvf4(d0b, d1b, d2b, d3b, c0, c1, c2, c3, e0, e1, sfc, sfd);
    }
    for (; kb < NK; kb++) {
        uint32_t a0 = __ldcs((const uint32_t *)(wr0 + kb * 32 + 4 * p));
        uint32_t a1 = __ldcs((const uint32_t *)(wr1 + kb * 32 + 4 * p));
        uint32_t a2 = __ldcs((const uint32_t *)(wr0 + kb * 32 + 16 + 4 * p));
        uint32_t a3 = __ldcs((const uint32_t *)(wr1 + kb * 32 + 16 + 4 * p));
        uint32_t sfa = __ldcs((const uint32_t *)((p & 1 ? sr1 : sr0) + kb * 4));
        uint32_t b0 = 0, b1 = 0, sfb = 0;
        if (g == 0) {
            b0 = *(const uint32_t *)(xq + kb * 32 + 4 * p);
            b1 = *(const uint32_t *)(xq + kb * 32 + 16 + 4 * p);
        }
        if (lane == 0)
            sfb = *(const uint32_t *)(xs + kb * 4);
        mma_mxf4nvf4(d0a, d1a, d2a, d3a, a0, a1, a2, a3, b0, b1, sfa, sfb);
    }
    out0 = d0a + d0b;
    out1 = d2a + d2b;
}

// ---------------------------------------------------------------------------
// Stage 1: grouped gate/up. grid = (NFF/64, NEXPUSED, 2), block = 128.
// blockIdx.y = k-th selected expert, blockIdx.z = 0 gate / 1 up.
// Writes eg_all[k][NFF] / eu_all[k][NFF] (fp32, per-expert scale_2 applied).
// ---------------------------------------------------------------------------
template<int K>   // K = NEMBD
__global__ __launch_bounds__(128) void k_fp4tc_gate_up(
    const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Wu,
    const uint8_t *__restrict__ Sg, const uint8_t *__restrict__ Su,
    size_t w_stride, size_t s_stride,
    const int *__restrict__ sel, const int *__restrict__ slots,
    const float *__restrict__ s2g_tab, const float *__restrict__ s2u_tab,
    const uint32_t *__restrict__ ready, int *__restrict__ route_err,
    const uint8_t *__restrict__ xq_g, const uint8_t *__restrict__ xs_g,
    float *__restrict__ eg_all, float *__restrict__ eu_all, int rows) {
    const int k = blockIdx.y, proj = blockIdx.z;
    const int slot = slots[k];
    if (threadIdx.x == 0) fp4tc_wait_slot(ready + slot, route_err);
    __syncthreads();
    __shared__ __align__(16) uint8_t xq[K / 2];
    __shared__ __align__(16) uint8_t xs[K / 16];
    for (int i = threadIdx.x * 4; i < K / 2; i += blockDim.x * 4)
        *(uint32_t *)(xq + i) = *(const uint32_t *)(xq_g + i);
    for (int i = threadIdx.x * 4; i < K / 16; i += blockDim.x * 4)
        *(uint32_t *)(xs + i) = *(const uint32_t *)(xs_g + i);
    __syncthreads();
    if (*route_err) return;   // token aborts on the latched error downstream

    const uint8_t *W = (proj ? Wu : Wg) + (size_t)slot * w_stride;
    const uint8_t *S = (proj ? Su : Sg) + (size_t)slot * s_stride;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 64 + warp * 16;
    float o0, o1;
    fp4tc_mma_strip<K>(W, S, xq, xs, row0, lane, o0, o1);
    if ((lane & 3) == 0) {
        float s2 = (proj ? s2u_tab : s2g_tab)[sel[k]];
        float *out = (proj ? eu_all : eg_all) + k * rows;
        out[row0 + (lane >> 2)] = o0 * s2;
        out[row0 + (lane >> 2) + 8] = o1 * s2;
    }
}

// ---------------------------------------------------------------------------
// Stage 2a: prepare and quantize the routed hidden vector exactly once per
// expert. The old down kernel repeated these 640 SiLU/multiply operations and
// the dynamic quantization in every one of its NEMBD/64 row-group CTAs (40x).
// Keeping this as a separate ten-CTA launch makes the down MMA purely a weight
// stream plus accumulation.
// ---------------------------------------------------------------------------
template<int K>   // K = NFF
__global__ __launch_bounds__(128) void k_fp4tc_down_prepare(
    const int *__restrict__ sel,
    const float *__restrict__ s2g_tab, const float *__restrict__ s2u_tab,
    const float *__restrict__ s2d_tab, const float *__restrict__ wts,
    const int *__restrict__ route_err,
    const float *__restrict__ eg_all, const float *__restrict__ eu_all,
    uint8_t *__restrict__ xq_all, uint8_t *__restrict__ xs_all) {
    const int k = blockIdx.x;
    if (*route_err) return;

    __shared__ float h[K];
    const int e = sel[k];
    const float fold = wts[k] * s2d_tab[e];
    const float s2g = s2g_tab[e], s2u = s2u_tab[e];
    const float *gA = eg_all + k * K, *uA = eu_all + k * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float gv = gA[i] * s2g, uv = uA[i] * s2u;
        h[i] = fold * (gv / (1.f + __expf(-gv))) * uv;   // silu(gate) * up
    }
    __syncthreads();
    fp4tc_quant_block(h, K, xq_all + (size_t)k * (K / 2),
                            xs_all + (size_t)k * (K / 16));
}

// Stage 2b: grouped down + route-weighted accumulation. grid = (NEMBD/64,
// NEXPUSED), block = 128. Every CTA consumes the already-prepared per-expert
// FP4 hidden vector and only streams its unique 64-row weight strip.
template<int K>   // K = NFF
__global__ __launch_bounds__(128) void k_fp4tc_down_accum(
    const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
    size_t w_stride, size_t s_stride,
    const int *__restrict__ slots,
    const uint32_t *__restrict__ ready, int *__restrict__ route_err,
    const uint8_t *__restrict__ xq_all, const uint8_t *__restrict__ xs_all,
    float *__restrict__ y, int rows) {
    const int k = blockIdx.y;
    const int slot = slots[k];
    if (threadIdx.x == 0) fp4tc_wait_slot(ready + slot, route_err);
    __syncthreads();
    if (*route_err) return;

    const uint8_t *W = Wd + (size_t)slot * w_stride;
    const uint8_t *S = Sd + (size_t)slot * s_stride;
    const uint8_t *xq = xq_all + (size_t)k * (K / 2);
    const uint8_t *xs = xs_all + (size_t)k * (K / 16);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 64 + warp * 16;
    float o0, o1;
    fp4tc_mma_strip<K>(W, S, xq, xs, row0, lane, o0, o1);
    if ((lane & 3) == 0) {
        atomicAdd(y + row0 + (lane >> 2), o0);
        atomicAdd(y + row0 + (lane >> 2) + 8, o1);
    }
}

// ---------------------------------------------------------------------------
// Host side: capability gate, workspace, launches, timing hooks
// ---------------------------------------------------------------------------
static struct {
    int supported;          // -1 undecided, 0/1
    int inited;
    uint8_t *xq, *xs;       // quantized activation: NEMBD/2 + NEMBD/16 bytes
    uint8_t *down_xq, *down_xs; // [NEXPUSED] prepared NFF hidden vectors
    float *eg_all, *eu_all; // [NEXPUSED][NFF] fp32 gate/up results
    int timing;             // QF_FP4TC_TIMING
    cudaEvent_t ev[4];
    double us[3];           // quant, gate_up, down accumulators
    unsigned long calls;
} g_fp4 = { -1, 0, NULL, NULL, NULL, NULL, NULL, NULL, 0,
            {NULL, NULL, NULL, NULL}, {0, 0, 0}, 0 };

// DISABLED BY DEFAULT - this path computes zero. Opt in with QF_FP4TC=1.
//
// Measured 2026-08-28 with the production-geometry canary (synth_canary, same
// NFF=640 / NEMBD=2560 as production, so this is not a small-shape artifact):
//
//   routed-expert output, layer 0     this path: absum 0.0000 (every element
//                                                exactly 0, std 0, no error
//                                                latched, OMMA present in the
//                                                sm_121a cubin)
//                                     legacy  : absum 47.6864, ratio to the HF
//                                                reference 1.0138
//
// Two defects are known in this file and neither is fixed yet:
//
//  1. fp4tc_mma_strip loads the B-operand block scale only on lane 0 (`if
//     (lane == 0) sfb = ...`) and the B fragment only on lanes 0-3 (`if (g ==
//     0)`). mma.sync...block_scale is a warp-wide instruction whose scale and
//     B operands must be supplied by the lanes the PTX ISA designates; the
//     other 31 lanes currently pass 0.
//  2. k_fp4tc_repack assumes the checkpoint packs each 16-value group as
//     "value j in the low nibble of byte j, value j+8 in the high nibble".
//     That is false. Dequantizing layers.0.mlp.experts.0.gate_proj from the
//     NVFP4 checkpoint and comparing against the BF16 original
//     (models/Qwen3.8-Flash-Next, same weights unquantized) gives:
//         byte i = {v[2i] lo, v[2i+1] hi}  -> corr 0.9955, maxerr 0.0066
//         byte j = {v[j] lo, v[j+8] hi}    -> corr 0.1420, maxerr 0.0809
//     The checkpoint is already in the first (adjacent-pair) order, so the
//     repack permutes correct weights into the wrong order.
//
// Because this path was the default, every previously reported throughput
// number was measured with the routed experts contributing nothing. Those
// numbers do not describe a working engine and must be re-measured.
//
// Do not re-enable by default until the canary reports routed/moe_out within
// tolerance of the reference.
int qf_fp4tc_supported(void) {
    if (g_fp4.supported < 0) {
        int ok = 0;
        const char *e = getenv("QF_FP4TC");
        if (e && e[0] == '1') {
            int dev = -1, major = 0, minor = 0;
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
                ok = (major == 12 && minor <= 1);   // sm_120a / sm_121a
        }
        g_fp4.supported = ok;
        if (!ok)
            fprintf(stderr, "qf_fp4tc: native NVFP4 MMA path disabled (default; set "
                            "QF_FP4TC=1 to opt in) - using the legacy NVFP4 GEMV path\n");
        else
            fprintf(stderr, "qf_fp4tc: WARNING - native NVFP4 MMA path enabled by "
                            "QF_FP4TC=1; this path is known to compute zero routed "
                            "experts. See the comment above qf_fp4tc_supported().\n");
    }
    return g_fp4.supported;
}

int qf_fp4tc_init(void) {
    if (!qf_fp4tc_supported()) return 0;
    if (g_fp4.inited) return 0;
    if (cudaMalloc((void **)&g_fp4.xq, NEMBD / 2) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&g_fp4.xs, NEMBD / 16) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&g_fp4.down_xq, (size_t)NEXPUSED * (NFF / 2)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&g_fp4.down_xs, (size_t)NEXPUSED * (NFF / 16)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&g_fp4.eg_all, (size_t)NEXPUSED * NFF * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&g_fp4.eu_all, (size_t)NEXPUSED * NFF * sizeof(float)) != cudaSuccess) return -1;
    const char *t = getenv("QF_FP4TC_TIMING");
    g_fp4.timing = (t && t[0] == '1');
    if (g_fp4.timing)
        for (int i = 0; i < 4; i++)
            if (cudaEventCreate(&g_fp4.ev[i]) != cudaSuccess) return -1;
    g_fp4.inited = 1;
    return 0;
}

void qf_fp4tc_shutdown(void) {
    if (g_fp4.xq) cudaFree(g_fp4.xq);
    if (g_fp4.xs) cudaFree(g_fp4.xs);
    if (g_fp4.down_xq) cudaFree(g_fp4.down_xq);
    if (g_fp4.down_xs) cudaFree(g_fp4.down_xs);
    if (g_fp4.eg_all) cudaFree(g_fp4.eg_all);
    if (g_fp4.eu_all) cudaFree(g_fp4.eu_all);
    for (int i = 0; i < 4; i++)
        if (g_fp4.ev[i]) cudaEventDestroy(g_fp4.ev[i]);
    memset(&g_fp4, 0, sizeof(g_fp4));
    g_fp4.supported = -1;
}

int qf_fp4tc_enabled(void) { return g_fp4.inited && g_fp4.supported == 1; }

void qf_fp4tc_moe(const QfLayer *L, const int *sel_dev, const float *wts_dev,
                  const float *x, float *y, int *route_err_dev, cudaStream_t s) {
    const size_t w_stride = (size_t)NFF * NEMBD / 2;    // 819200 B, all 3 projections
    const size_t s_stride = (size_t)NFF * NEMBD / 16;   // 102400 B, all 3 scale tensors
    const int tm = g_fp4.timing;
    if (tm) cudaEventRecord(g_fp4.ev[0], s);
    k_fp4tc_quant<<<1, 256, 0, s>>>(x, NEMBD, g_fp4.xq, g_fp4.xs);
    if (tm) cudaEventRecord(g_fp4.ev[1], s);
    dim3 gg(NFF / 64, NEXPUSED, 2);
    k_fp4tc_gate_up<NEMBD><<<gg, 128, 0, s>>>(
        (const uint8_t *)L->exp_gate, (const uint8_t *)L->exp_up,
        (const uint8_t *)L->exp_scale, (const uint8_t *)L->exp_scale_up,
        w_stride, s_stride, sel_dev, L->route_slot_dev,
        L->s2_gate_dev, L->s2_up_dev, L->slot_ready_dev, route_err_dev,
        g_fp4.xq, g_fp4.xs, g_fp4.eg_all, g_fp4.eu_all, NFF);
    if (tm) cudaEventRecord(g_fp4.ev[2], s);
    k_fp4tc_down_prepare<NFF><<<NEXPUSED, 128, 0, s>>>(
        sel_dev, L->s2_gate_dev, L->s2_up_dev, L->s2_down_dev, wts_dev,
        route_err_dev, g_fp4.eg_all, g_fp4.eu_all, g_fp4.down_xq, g_fp4.down_xs);
    dim3 gd(NEMBD / 64, NEXPUSED, 1);
    k_fp4tc_down_accum<NFF><<<gd, 128, 0, s>>>(
        (const uint8_t *)L->exp_down, (const uint8_t *)L->exp_scale_down,
        w_stride, s_stride, L->route_slot_dev, L->slot_ready_dev, route_err_dev,
        g_fp4.down_xq, g_fp4.down_xs, y, NEMBD);
    if (tm) {
        cudaEventRecord(g_fp4.ev[3], s);
        g_fp4.calls++;
    }
}

void qf_fp4tc_timing_report(FILE *out) {
    if (!g_fp4.timing || g_fp4.calls == 0) return;
    if (cudaEventSynchronize(g_fp4.ev[3]) != cudaSuccess) return;
    float ms;
    const char *names[3] = { "quant", "gate_up", "down_accum" };
    for (int i = 0; i < 3; i++) {
        if (cudaEventElapsedTime(&ms, g_fp4.ev[i], g_fp4.ev[i + 1]) == cudaSuccess)
            g_fp4.us[i] += (double)ms * 1000.0;
    }
    fprintf(out, "qf_fp4tc: %lu layer-calls, avg us: quant %.1f gate_up %.1f down %.1f (total %.1f)\n",
            g_fp4.calls, g_fp4.us[0] / g_fp4.calls, g_fp4.us[1] / g_fp4.calls,
            g_fp4.us[2] / g_fp4.calls,
            (g_fp4.us[0] + g_fp4.us[1] + g_fp4.us[2]) / g_fp4.calls);
    g_fp4.us[0] = g_fp4.us[1] = g_fp4.us[2] = 0.0;
    g_fp4.calls = 0;
}
