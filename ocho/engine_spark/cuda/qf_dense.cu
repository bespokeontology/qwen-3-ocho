// qf_dense.cu - fused dense BF16 projection kernels for batch=1 decode on
// DGX Spark GB10 (sm_121a, CUDA 13).
//
// Why not cuBLASLt here: at batch=1 every dense projection is a purely
// bandwidth-bound GEMV (the weight matrix is streamed exactly once; the
// activation is a few KB). cuBLASLt's n=1 paths for these shapes resolve to
// splitK GEMV-class kernels that add a reduction pass and workspace traffic,
// use no tensor-core throughput that matters at this arithmetic intensity,
// and still need the activation converted f32 -> bf16 before every call. The
// wave-2 Lt plan cache remains as the QF_DENSE=legacy fallback; the default
// path is this file.
//
// What this file does instead:
//   * One kernel launch per projection GROUP. All members of a group consume
//     the same activation x (e.g. GDN qkv+z+a+b), so x is read once per block
//     into shared memory (fp32, no quantization of the activation) and the
//     weights of every member stream through the same launch.
//   * One warp per output row; weights are read with 16-byte uint4 loads
//     (8 BF16 per lane per step), fully coalesced across the warp, fp32 FMA
//     accumulation. Numerics match the naive fallback exactly in operand
//     precision (BF16 weight x fp32 activation, fp32 accumulate); only the
//     summation order differs.
//   * No descriptors, no plans, no workspace, no heuristic queries: the kernel
//     is stateless, so there is nothing to create or cache per call. Launch
//     geometry is a closed-form function of the row count.
//
// Per-token dense launch count on a GDN layer drops from 10 GEMV + up to 10
// conversion launches to 4 single launches (in-proj, out-proj, MoE-in, shexp
// down); a QSA layer from 7+conversions to 4 (in-proj, wo, MoE-in, shexp
// down). lm_head is one launch with 31040 blocks of pure weight streaming.
#include "qf_dense.h"
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Same -D-overridable model dims as qf.cu (defaults = production model).
#ifndef NEMBD
#define NEMBD 2560
#endif
#ifndef NHEAD
#define NHEAD 24
#endif
#ifndef NKV
#define NKV 2
#endif
#ifndef HDIM
#define HDIM 256
#endif
#ifndef DINN
#define DINN 10240
#endif
#ifndef GDN_VDIM
#define GDN_VDIM 6144
#endif
#ifndef DTRANK
#define DTRANK 48
#endif
#ifndef NEXP
#define NEXP 512
#endif
#ifndef NFF
#define NFF 640
#endif
#ifndef NVOCAB
#define NVOCAB 248320
#endif
#ifndef QGATE
#define QGATE 512
#endif

#define QFD_BLOCK 256
#define QFD_WARPS (QFD_BLOCK / 32)
#define QFD_MAX_PROJ 4
#define QFD_MAX_IN 8192          // fp32 shared staging cap: 32 KiB per block

// Device-side group descriptor, passed by value (kernel parameter space).
typedef struct {
    const float *x;
    int in;
    int rows_total;
    QfDenseProj p[QFD_MAX_PROJ];
} QfDenseGroupK;

static __device__ __forceinline__ float qfd_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(~0u, v, off);
    return v;
}

// One warp per output row across all projections in the group. The activation
// is staged once per block into shared memory as fp32; each warp then streams
// its weight row with coalesced uint4 loads (8 BF16 = 16 B per lane, 512 B
// per warp per step). Shared-memory reads of xs are warp-uniform (broadcast,
// conflict-free).
template<int NPROJ>
__global__ __launch_bounds__(QFD_BLOCK)
void k_gemv_bf16_group(const __grid_constant__ QfDenseGroupK g) {
    extern __shared__ float xs[];
    {
        const int n4 = g.in >> 2;
        const float4 *x4 = (const float4 *)g.x;
        for (int i = threadIdx.x; i < n4; i += QFD_BLOCK)
            ((float4 *)xs)[i] = __ldg(x4 + i);
        for (int i = (g.in & ~3) + threadIdx.x; i < g.in; i += QFD_BLOCK)
            xs[i] = __ldg(g.x + i);
    }
    __syncthreads();

    const int r = blockIdx.x * QFD_WARPS + (threadIdx.x >> 5);
    if (r >= g.rows_total) return;
    const int lane = threadIdx.x & 31;

    // Map the flat row to (projection, local row). NPROJ <= 4, fully unrolled;
    // the r < rows_total guard above guarantees a match before the last arm.
    const __nv_bfloat16 *W = NULL;
    float *y = NULL;
    int lr = r;
    #pragma unroll
    for (int p = 0; p < NPROJ; p++) {
        if (lr < g.p[p].rows || p == NPROJ - 1) {
            W = (const __nv_bfloat16 *)g.p[p].W;
            y = g.p[p].y;
            break;
        }
        lr -= g.p[p].rows;
    }

    const uint4 *wr = (const uint4 *)(W + (size_t)lr * g.in);
    const int n8 = g.in >> 3;
    float acc = 0.f;
    for (int i = lane; i < n8; i += 32) {
        uint4 pk = __ldg(wr + i);
        const __nv_bfloat162 *h2 = (const __nv_bfloat162 *)&pk;
        const int o = i << 3;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(h2[j]);
            acc = fmaf(f.x, xs[o + 2 * j], acc);
            acc = fmaf(f.y, xs[o + 2 * j + 1], acc);
        }
    }
    acc = qfd_warp_sum(acc);
    if (lane == 0) y[lr] = acc;
}

// ---------------- FP8 E4M3 dense weights (Blackwell-class storage) ---------
// Dense weights are stored as OCP E4M3 with a UE4M3 scale byte per 64 input
// dims (the checkpoint's own routed-expert scheme; dense is 80% of per-token
// bytes and BF16 storage left the memory bandwidth of the GB10 on the table).
// Per-tensor slab layout built by qfd_quant_fp8: [scales rows*(in/64) bytes |
// pad to 16 | E4M3 weights rows*in bytes]; the kernels recover the scale base
// from W deterministically, so QfDenseProj and every call site stay unchanged.

__device__ static __forceinline__ float qfd_ue4m3_dec(uint8_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    float v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return b == 0x7F ? __int_as_float(0x7F800000) : v;
}

__device__ static __forceinline__ uint8_t qfd_ue4m3_enc_ceil(float v) {
    if (!(v > 0.f)) return 0u;
    uint32_t b = __float_as_uint(v);
    int e32 = (int)(b >> 23) - 127;
    if (e32 < -6) {
        uint32_t m = (uint32_t)ceilf(v * 512.f);
        if (m == 0) m = 1;
        if (m > 7) return 0x08u;
        return (uint8_t)m;
    }
    int e = e32 + 7;
    uint32_t m = ((b & 0x7FFFFFu) + 0xFFFFFu) >> 20;
    if (m == 8u) { e++; m = 0; }
    if (e > 15) return 0x7Eu;
    return (uint8_t)((e << 3) | m);
}

// Round-to-nearest E4M3 encode (OCP: bias 7, no inf; |x| > 448 saturates).
__device__ static __forceinline__ uint8_t qfd_e4m3_enc_rn(float x) {
    uint32_t ub = __float_as_uint(x);
    uint32_t sg = (ub >> 24) & 0x80u;
    uint32_t b = ub & 0x7FFFFFFFu;
    if (b < 0x3B000000u) return (uint8_t)sg;   // |x| < 2^-9 rounds to zero
    int e32 = (int)(b >> 23) - 127;
    uint32_t mant = (b & 0x7FFFFFu) | 0x800000u;
    uint32_t out;
    if (e32 < -6) {
        float m = rintf(__uint_as_float(b) * 512.f);
        if (m >= 8.f) { out = 0x08u; }         // rounded up into 2^-6
        else out = (uint32_t)m;
    } else {
        uint32_t q8 = (mant + 0x040000u) >> 20;
        if (q8 >= 16u) { e32++; q8 = 8u; }
        if (e32 > 15) out = 0x7Eu;             // saturate at 448
        else out = ((uint32_t)(e32 + 7) << 3) | (q8 & 7u);
    }
    return (uint8_t)(sg | out);
}

// One block per row: stage the BF16 row, take per-64 absmax -> UE4M3 scale,
// then quantize. Grid = rows; dynamic smem = in*4 + (in/64)*8 bytes.
__global__ void k_qfd_quant_bf16(const __nv_bfloat16 *__restrict__ src,
                                 uint8_t *__restrict__ q,
                                 uint8_t *__restrict__ sc, int in) {
    extern __shared__ float sm[];              // [in] floats then in/64 decs
    float *xs = sm;
    float *sdec = sm + in;
    const __nv_bfloat16 *row = src + (size_t)blockIdx.x * in;
    for (int i = threadIdx.x; i < in; i += blockDim.x)
        xs[i] = __bfloat162float(row[i]);
    __syncthreads();
    const int n64 = in >> 6;
    for (int b = threadIdx.x; b < n64; b += blockDim.x) {
        float am = 0.f;
        for (int j = 0; j < 64; j++) {
            float v = fabsf(xs[b * 64 + j]);
            am = fmaxf(am, v);
        }
        uint8_t sb = qfd_ue4m3_enc_ceil(am * (1.f / 448.f));
        sc[(size_t)blockIdx.x * n64 + b] = sb;
        sdec[b] = qfd_ue4m3_dec(sb);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < in; d += blockDim.x)
        q[(size_t)blockIdx.x * in + d] = qfd_e4m3_enc_rn(xs[d] / sdec[d >> 6]);
}

// Quantize one BF16 device tensor to the FP8 slab layout. Returns 0 and fills
// *W8 with the E4M3 weight base (scales live immediately before it) on success.
int qfd_quant_fp8(const void *src_bf16, int rows, int in, cudaStream_t s, void **W8) {
    if (!src_bf16 || rows <= 0 || in <= 0 || (in & 63)) return -1;
    const size_t n64 = (size_t)rows * (in >> 6);
    const size_t qbytes = (size_t)rows * in;
    const size_t pad = (16 - (n64 & 15u)) & 15u;
    uint8_t *slab = NULL;
    if (cudaMalloc(&slab, n64 + pad + qbytes) != cudaSuccess) return -1;
    uint8_t *sc = slab;
    uint8_t *q = slab + n64 + pad;
    const size_t smem = (size_t)in * 4 + (size_t)(in >> 6) * 8;
    k_qfd_quant_bf16<<<rows, 256, smem, s>>>((const __nv_bfloat16 *)src_bf16, q, sc, in);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(slab); return -1; }
    *W8 = q;
    return 0;
}

template<int NPROJ>
__global__ __launch_bounds__(QFD_BLOCK)
void k_gemv_fp8_group(const __grid_constant__ QfDenseGroupK g) {
    extern __shared__ float xs[];
    __shared__ float lut[256];
    {
        const int n4 = g.in >> 2;
        const float4 *x4 = (const float4 *)g.x;
        for (int i = threadIdx.x; i < n4; i += QFD_BLOCK)
            ((float4 *)xs)[i] = __ldg(x4 + i);
        for (int i = (g.in & ~3) + threadIdx.x; i < g.in; i += QFD_BLOCK)
            xs[i] = __ldg(g.x + i);
        for (int i = threadIdx.x; i < 256; i += QFD_BLOCK) {
            uint32_t e = (i >> 3) & 15u, m = i & 7u;
            float v = (e == 0) ? (float)m * 0x1p-9f
                               : __uint_as_float(((e + 120u) << 23) | (m << 20));
            lut[i] = (i & 0x80u) ? -v : v;
        }
    }
    __syncthreads();

    const int r = blockIdx.x * QFD_WARPS + (threadIdx.x >> 5);
    if (r >= g.rows_total) return;
    const int lane = threadIdx.x & 31;

    const uint8_t *W = NULL;
    const uint8_t *S = NULL;
    float *y = NULL;
    int lr = r;
    #pragma unroll
    for (int p = 0; p < NPROJ; p++) {
        if (lr < g.p[p].rows || p == NPROJ - 1) {
            W = (const uint8_t *)g.p[p].W;
            y = g.p[p].y;
            // Slab layout (qfd_quant_fp8): [scales rows*(in/64) | pad->16 | W]
            const size_t sbytes = (size_t)g.p[p].rows * (g.in >> 6);
            S = W - (sbytes + ((16 - (sbytes & 15u)) & 15u));
            break;
        }
        lr -= g.p[p].rows;
    }

    const int n64 = g.in >> 6;
    const uint2 *wr = (const uint2 *)(W + (size_t)lr * g.in);
    const uint8_t *sr = S + (size_t)lr * n64;
    float acc = 0.f;
    for (int c = lane; c < (g.in >> 3); c += 32) {
        uint2 pk = __ldg(wr + c);
        float scv = lut[sr[c >> 3]];
        const int o = c << 3;
        acc = fmaf(lut[pk.x & 0xFFu], scv * xs[o + 0], acc);
        acc = fmaf(lut[(pk.x >> 8) & 0xFFu], scv * xs[o + 1], acc);
        acc = fmaf(lut[(pk.x >> 16) & 0xFFu], scv * xs[o + 2], acc);
        acc = fmaf(lut[(pk.x >> 24) & 0xFFu], scv * xs[o + 3], acc);
        acc = fmaf(lut[pk.y & 0xFFu], scv * xs[o + 4], acc);
        acc = fmaf(lut[(pk.y >> 8) & 0xFFu], scv * xs[o + 5], acc);
        acc = fmaf(lut[(pk.y >> 16) & 0xFFu], scv * xs[o + 6], acc);
        acc = fmaf(lut[(pk.y >> 24) & 0xFFu], scv * xs[o + 7], acc);
    }
    acc = qfd_warp_sum(acc);
    if (lane == 0) y[lr] = acc;
}

// ---------------------------------------------------------------------------
// BATCHED dense group GEMV: T activations against ONE weight read.
//
// Decode is memory-bound - per token the engine moves ~4.27 GB of dense weights
// and does one multiply per byte. A T-token speculative verify that ran T
// sequential forwards would re-read all of it T times. Here each thread loads
// the weight byte once and issues T fused-multiply-adds, so the weight traffic
// of a T-token verify equals that of a single decode step. That is where the
// speedup in speculative decoding actually comes from; the routed experts do
// NOT amortize (each token routes to its own 10) and are looped per token.
//
// x is [T][in] row-major, y is [T][rows] row-major. Identical arithmetic to
// k_gemv_fp8_group per (token, row).
#define QFD_MAX_T 16   // rows per batched step (== QF_SPEC_MAXT)
template<int NPROJ>
__global__ __launch_bounds__(QFD_BLOCK)
void k_gemv_fp8_group_T(const __grid_constant__ QfDenseGroupK g,
                        const float *__restrict__ xT, float *const *__restrict__ yT,
                        int T) {
    // The activations are read straight from global. Staging [T][in] in shared
    // would be 98 KB at T=4, in=6144 - past the default per-block limit - and a
    // measured A/B on the expert kernel showed shared staging of the activation
    // is neutral anyway (-1.1%, inside the band): it is a small vector read by
    // every block, so it stays in L2 and is broadcast.
    const float *xs = xT;
    __shared__ float lut[256];
    {
        for (int i = threadIdx.x; i < 256; i += QFD_BLOCK) {
            uint32_t e = (i >> 3) & 15u, m = i & 7u;
            float v = (e == 0) ? (float)m * 0x1p-9f
                               : __uint_as_float(((e + 120u) << 23) | (m << 20));
            lut[i] = (i & 0x80u) ? -v : v;
        }
    }
    __syncthreads();

    const int r = blockIdx.x * QFD_WARPS + (threadIdx.x >> 5);
    if (r >= g.rows_total) return;
    const int lane = threadIdx.x & 31;

    const uint8_t *W = NULL, *S = NULL;
    int lr = r, pidx = 0;
    #pragma unroll
    for (int p = 0; p < NPROJ; p++) {
        if (lr < g.p[p].rows || p == NPROJ - 1) {
            W = (const uint8_t *)g.p[p].W;
            pidx = p;
            const size_t sbytes = (size_t)g.p[p].rows * (g.in >> 6);
            S = W - (sbytes + ((16 - (sbytes & 15u)) & 15u));
            break;
        }
        lr -= g.p[p].rows;
    }

    const int n64 = g.in >> 6;
    const uint2 *wr = (const uint2 *)(W + (size_t)lr * g.in);
    const uint8_t *sr = S + (size_t)lr * n64;
    float acc[QFD_MAX_T];
    #pragma unroll
    for (int t = 0; t < QFD_MAX_T; t++) acc[t] = 0.f;

    for (int c = lane; c < (g.in >> 3); c += 32) {
        uint2 pk = __ldg(wr + c);            // <- the single weight read
        float scv = lut[sr[c >> 3]];
        float w[8];
        w[0] = lut[pk.x & 0xFFu];        w[1] = lut[(pk.x >> 8) & 0xFFu];
        w[2] = lut[(pk.x >> 16) & 0xFFu]; w[3] = lut[(pk.x >> 24) & 0xFFu];
        w[4] = lut[pk.y & 0xFFu];        w[5] = lut[(pk.y >> 8) & 0xFFu];
        w[6] = lut[(pk.y >> 16) & 0xFFu]; w[7] = lut[(pk.y >> 24) & 0xFFu];
        const int o = c << 3;
        for (int t = 0; t < T; t++) {
            const float *xt = xs + (size_t)t * g.in;
            float a = acc[t];
            #pragma unroll
            for (int j = 0; j < 8; j++) a = fmaf(w[j], scv * xt[o + j], a);
            acc[t] = a;
        }
    }
    for (int t = 0; t < T; t++) {
        float v = qfd_warp_sum(acc[t]);
        if (lane == 0) yT[pidx][(size_t)t * g.p[pidx].rows + lr] = v;
    }
}


// ---------------------------------------------------------------------------
// v2 of the batched group GEMV (2026-09-01). The v1 kernel above takes T at
// run time: nvcc cannot keep acc[QFD_MAX_T] in registers under a runtime-bound
// loop, so the SASS round-trips every accumulator through local memory
// (LDL/STL per row per 8 weights), reads the activation row with EIGHT scalar
// 4-byte loads at a 32-byte lane stride, and re-multiplies the block scale
// into every FMA. Per (8 weights, row) that is ~28 instructions; at T=16 the
// kernel is issue/L1-bound and its cost grows ~linearly with T (measured
// 18.9 ms per batched row against a 36 ms weight stream for the whole step).
//
// v2 makes T a template parameter (accumulators in registers, loop fully
// unrolled), folds the block scale into the 8 decoded weights ONCE (both are
// FP8-derived, so the product is exact in fp32 - more exact than v1, which
// rounded scv*x), and reads each activation row as two float4 (2 loads
// instead of 8). Per (8 weights, row): 2 LDG.128 + 8 FFMA. The weight read
// stays single; results differ from v1 only by fp32 rounding order.
// Alignment: xT rows start at t*in floats with in % 8 == 0 and every caller
// passes a cudaMalloc'd base, so the float4 loads are 16-byte aligned.
template<int NPROJ, int T>
__global__ __launch_bounds__(QFD_BLOCK)
void k_gemv_fp8_group_T2(const __grid_constant__ QfDenseGroupK g,
                         const float *__restrict__ xT, float *const *__restrict__ yT) {
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += QFD_BLOCK) {
        uint32_t e = (i >> 3) & 15u, m = i & 7u;
        float v = (e == 0) ? (float)m * 0x1p-9f
                           : __uint_as_float(((e + 120u) << 23) | (m << 20));
        lut[i] = (i & 0x80u) ? -v : v;
    }
    __syncthreads();

    const int r = blockIdx.x * QFD_WARPS + (threadIdx.x >> 5);
    if (r >= g.rows_total) return;
    const int lane = threadIdx.x & 31;

    const uint8_t *W = NULL, *S = NULL;
    int lr = r, pidx = 0;
    #pragma unroll
    for (int p = 0; p < NPROJ; p++) {
        if (lr < g.p[p].rows || p == NPROJ - 1) {
            W = (const uint8_t *)g.p[p].W;
            pidx = p;
            const size_t sbytes = (size_t)g.p[p].rows * (g.in >> 6);
            S = W - (sbytes + ((16 - (sbytes & 15u)) & 15u));
            break;
        }
        lr -= g.p[p].rows;
    }

    const int n64 = g.in >> 6;
    const int in4 = g.in >> 2;                       // float4 per activation row
    const uint2 *wr = (const uint2 *)(W + (size_t)lr * g.in);
    const uint8_t *sr = S + (size_t)lr * n64;
    const float4 *x4 = (const float4 *)xT;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;

    for (int c = lane; c < (g.in >> 3); c += 32) {
        const uint2 pk = __ldg(wr + c);              // <- the single weight read
        const float scv = lut[sr[c >> 3]];
        float w[8];
        w[0] = scv * lut[pk.x & 0xFFu];         w[1] = scv * lut[(pk.x >> 8) & 0xFFu];
        w[2] = scv * lut[(pk.x >> 16) & 0xFFu]; w[3] = scv * lut[(pk.x >> 24) & 0xFFu];
        w[4] = scv * lut[pk.y & 0xFFu];         w[5] = scv * lut[(pk.y >> 8) & 0xFFu];
        w[6] = scv * lut[(pk.y >> 16) & 0xFFu]; w[7] = scv * lut[(pk.y >> 24) & 0xFFu];
        const float4 *xp = x4 + (c << 1);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4 a4 = __ldg(xp + (size_t)t * in4);
            const float4 b4 = __ldg(xp + (size_t)t * in4 + 1);
            float a = acc[t];
            a = fmaf(w[0], a4.x, a); a = fmaf(w[1], a4.y, a);
            a = fmaf(w[2], a4.z, a); a = fmaf(w[3], a4.w, a);
            a = fmaf(w[4], b4.x, a); a = fmaf(w[5], b4.y, a);
            a = fmaf(w[6], b4.z, a); a = fmaf(w[7], b4.w, a);
            acc[t] = a;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        const float v = qfd_warp_sum(acc[t]);
        if (lane == 0) yT[pidx][(size_t)t * g.p[pidx].rows + lr] = v;
    }
}

template<int NPROJ>
static int qfd_launch_group_T2(int T, const QfDenseGroupK &g, const float *xT,
                               float *const *d_y, int blocks, cudaStream_t s) {
#define QFD_T2_CASE(N) case N: k_gemv_fp8_group_T2<NPROJ, N><<<blocks, QFD_BLOCK, 0, s>>>(g, xT, d_y); return 0
    switch (T) {
        QFD_T2_CASE(1);  QFD_T2_CASE(2);  QFD_T2_CASE(3);  QFD_T2_CASE(4);
        QFD_T2_CASE(5);  QFD_T2_CASE(6);  QFD_T2_CASE(7);  QFD_T2_CASE(8);
        QFD_T2_CASE(9);  QFD_T2_CASE(10); QFD_T2_CASE(11); QFD_T2_CASE(12);
        QFD_T2_CASE(13); QFD_T2_CASE(14); QFD_T2_CASE(15); QFD_T2_CASE(16);
        default: return -1;
    }
#undef QFD_T2_CASE
}
// QF_DENSE_T_V1=1 selects the v1 kernel (the 43.93 tok/s bank's arithmetic) for A/B.
static int qfd_group_T_v1(void) {
    static int v1 = -1;
    if (v1 < 0) {
        const char *e = getenv("QF_DENSE_T_V1");
        v1 = (e && atoi(e)) ? 1 : 0;
        fprintf(stderr, "dense T: batched group GEMV %s\n", v1 ? "v1 (runtime-T, local-memory accumulators)"
                                                              : "v2 (template-T, register accumulators, float4 activations)");
    }
    return v1;
}

static int g_qfd_fp8 = 0;     // 1 once dense weights are E4M3 (qfd_quant_fp8)
extern "C" int qfd_fp8_active(void) { return g_qfd_fp8; }

// Batched group GEMV: T activations, one weight read. See k_gemv_fp8_group_T.
// projs[].y must point at [T][rows] fp32. Returns 0 on success, -1 if the shape
// is ineligible (caller should fall back to T sequential qfd_gemv_group calls).
// Output-pointer staging for the batched kernel, as a POOL with a per-pass
// cursor rather than one shared buffer.
//
// The kernel needs the nproj output pointers in device memory, and the obvious
// implementation stages them through one static host array copied to one device
// slot. That is correct eagerly and wrong under CUDA-graph capture, twice over:
// the copy node would read a PAGEABLE host array, and every call site in the
// captured pass would point at the SAME device slot, so at replay all of them
// would see whichever pointers were staged last.
//
// A pool fixes both. qfd_gemv_group_T_rewind() resets the cursor at the top of
// each batched pass, so call site k always takes slot k and always stages the
// identical pointers (they are fixed weight/workspace addresses). A captured
// copy node therefore re-reads a pinned slot whose contents never change.
// The pool is a pinned mirror + ASYNC copy per slot: a slot must not be
// re-staged by the host until the device has consumed its copy. The captured
// passes below 512 keep fixed pointers; the EAGER prefill chunk pass runs
// tens of layers ahead of the device, and its recurrent and attention layers
// stage DIFFERENT pointers into the same slot index - rewinding it per layer
// let the host overwrite slots the device had not copied yet (garbage prefill
// that only the launch-queue throttling of per-token loops masked). It now
// owns its own region above the captured slots, rewound once per chunk after
// the previous chunk's stream sync.
#define QFD_GT_SLOTS 4096
#define QFD_GT_EAGER_BASE 512
static float **g_gt_dev = NULL;     // [QFD_GT_SLOTS][4] device
static float **g_gt_host = NULL;    // pinned mirror
static int     g_gt_cursor = 0;

void qfd_gemv_group_T_rewind(void) { g_gt_cursor = 0; }
void qfd_gemv_group_T_rewind_eager(void) { g_gt_cursor = QFD_GT_EAGER_BASE; }

int qfd_gemv_group_T(const QfDenseProj *projs, int nproj, const float *xT, int in,
                     int T, cudaStream_t s) {
    if (nproj < 1 || nproj > 4 || T < 1 || T > QFD_MAX_T) return -1;
    if (in <= 0 || (in & 7) || in > 8192) return -1;
    if (!g_qfd_fp8) return -1;                     // FP8 slab layout required
    QfDenseGroupK g;
    g.in = in; g.rows_total = 0;
    if (!g_gt_dev) {
        if (cudaMalloc(&g_gt_dev, (size_t)QFD_GT_SLOTS * 4 * sizeof(float *)) != cudaSuccess) return -1;
        if (cudaHostAlloc((void **)&g_gt_host, (size_t)QFD_GT_SLOTS * 4 * sizeof(float *),
                          cudaHostAllocDefault) != cudaSuccess) return -1;
    }
    if (g_gt_cursor >= QFD_GT_SLOTS) return -1;    // pass longer than the pool: no aliasing, fail
    float **slot_host = g_gt_host + (size_t)g_gt_cursor * 4;
    float **slot_dev  = g_gt_dev  + (size_t)g_gt_cursor * 4;
    g_gt_cursor++;
    for (int p = 0; p < nproj; p++) {
        g.p[p].W = projs[p].W; g.p[p].y = projs[p].y; g.p[p].rows = projs[p].rows;
        g.rows_total += projs[p].rows;
        slot_host[p] = projs[p].y;
    }
    for (int p = nproj; p < 4; p++) { g.p[p].W = NULL; g.p[p].y = NULL; g.p[p].rows = 0; }
    if (cudaMemcpyAsync(slot_dev, slot_host, nproj * sizeof(float *),
                        cudaMemcpyHostToDevice, s) != cudaSuccess)
        return -1;
    float **d_y = slot_dev;
    const int blocks = (g.rows_total + QFD_WARPS - 1) / QFD_WARPS;
    if (!qfd_group_T_v1()) {
        switch (nproj) {
            case 1: return qfd_launch_group_T2<1>(T, g, xT, d_y, blocks, s);
            case 2: return qfd_launch_group_T2<2>(T, g, xT, d_y, blocks, s);
            case 3: return qfd_launch_group_T2<3>(T, g, xT, d_y, blocks, s);
            default: return qfd_launch_group_T2<4>(T, g, xT, d_y, blocks, s);
        }
    }
    switch (nproj) {
        case 1: k_gemv_fp8_group_T<1><<<blocks, QFD_BLOCK, 0, s>>>(g, xT, d_y, T); break;
        case 2: k_gemv_fp8_group_T<2><<<blocks, QFD_BLOCK, 0, s>>>(g, xT, d_y, T); break;
        case 3: k_gemv_fp8_group_T<3><<<blocks, QFD_BLOCK, 0, s>>>(g, xT, d_y, T); break;
        default: k_gemv_fp8_group_T<4><<<blocks, QFD_BLOCK, 0, s>>>(g, xT, d_y, T); break;
    }
    return 0;
}


// ---------------- host dispatch ----------------

static int g_qfd_mode = -1;   // -1 unset, 0 legacy per-proj dispatch, 1 fused

void qfd_set_fp8_weights(int on) { g_qfd_fp8 = on; }
int qfd_fp8_is_on(void) { return g_qfd_fp8; }

static int qfd_fused_enabled(void) {
    if (g_qfd_mode < 0) {
        const char *e = getenv("QF_DENSE");
        g_qfd_mode = (e && (!strcmp(e, "legacy") || !strcmp(e, "0"))) ? 0 : 1;
    }
    return g_qfd_mode;
}

static int qfd_group_eligible(const QfDenseProj *projs, int nproj, const float *x, int in) {
    if (nproj < 1 || nproj > QFD_MAX_PROJ) return 0;
    if (in <= 0 || (in & 7) || in > QFD_MAX_IN) return 0;
    if (((uintptr_t)x & 15) != 0) return 0;
    for (int i = 0; i < nproj; i++) {
        if (projs[i].rows <= 0 || !projs[i].W || !projs[i].y) return 0;
        // in % 8 == 0 makes every row 16-byte aligned when the base is.
        if (((uintptr_t)projs[i].W & 15) != 0) return 0;
    }
    return 1;
}

// ---------------- timing counters (opt-in, QF_TIMING=1) ----------------
// Events are recorded around each region on the decode stream and folded
// (one device-wide sync) only at report time or when a ring fills. The decode
// hot path with timing disabled costs one cached integer compare per region.

enum {
    QFD_T_GDN_IN = 0,   // GDN fused qkv+z+a+b
    QFD_T_QSA_IN,       // QSA fused q+k+v
    QFD_T_OUT_PROJ,     // gdn_out / wo / shexp_down
    QFD_T_MOE_IN,       // router + shared-expert gate/up + gate_inp
    QFD_T_LM_HEAD,      // output head
    QFD_T_N
};
static const char *g_qfd_t_names[QFD_T_N] = {
    "gdn_inproj", "qsa_inproj", "out_proj", "moe_inproj", "lm_head"
};
#define QFD_T_RING 4096
static cudaEvent_t  g_qfd_ev[QFD_T_N][QFD_T_RING][2];   // lazily created
static int          g_qfd_n[QFD_T_N];
static double       g_qfd_ms[QFD_T_N];
static uint64_t     g_qfd_calls[QFD_T_N];
static int          g_qfd_timing = -1;

static void qfd_t_fold(void) {
    cudaDeviceSynchronize();
    for (int r = 0; r < QFD_T_N; r++) {
        for (int i = 0; i < g_qfd_n[r]; i++) {
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, g_qfd_ev[r][i][0], g_qfd_ev[r][i][1]) == cudaSuccess)
                g_qfd_ms[r] += (double)ms;
        }
        g_qfd_n[r] = 0;
    }
}

static void qfd_t_begin(int reg, cudaStream_t s) {
    if (g_qfd_timing < 0) {
        const char *e = getenv("QF_TIMING");
        g_qfd_timing = (e && e[0] == '1') ? 1 : 0;
    }
    if (!g_qfd_timing) return;
    if (g_qfd_n[reg] >= QFD_T_RING) qfd_t_fold();
    cudaEvent_t *ev = g_qfd_ev[reg][g_qfd_n[reg]];
    if (!ev[0]) {
        cudaEventCreateWithFlags(&ev[0], cudaEventDefault);  // timing enabled
        cudaEventCreateWithFlags(&ev[1], cudaEventDefault);
    }
    cudaEventRecord(ev[0], s);
}

static void qfd_t_end(int reg, cudaStream_t s) {
    if (g_qfd_timing != 1) return;
    cudaEventRecord(g_qfd_ev[reg][g_qfd_n[reg]][1], s);
    g_qfd_n[reg]++;
    g_qfd_calls[reg]++;
}

void qfd_timing_report(FILE *fp) {
    if (g_qfd_timing != 1) return;   // silent no-op unless QF_TIMING=1
    if (!fp) fp = stderr;
    qfd_t_fold();
    double total = 0.0;
    fprintf(fp, "qfd dense-projection timing (GPU ms, accumulated since start/reset):\n");
    for (int r = 0; r < QFD_T_N; r++) {
        total += g_qfd_ms[r];
        fprintf(fp, "  %-12s %10.3f ms  %8llu calls  %8.2f us/call\n",
                g_qfd_t_names[r], g_qfd_ms[r],
                (unsigned long long)g_qfd_calls[r],
                g_qfd_calls[r] ? 1e3 * g_qfd_ms[r] / (double)g_qfd_calls[r] : 0.0);
    }
    fprintf(fp, "  %-12s %10.3f ms\n", "TOTAL", total);
}

void qfd_timing_reset(void) {
    if (g_qfd_timing == 1) qfd_t_fold();
    for (int r = 0; r < QFD_T_N; r++) { g_qfd_ms[r] = 0.0; g_qfd_calls[r] = 0; }
}

void qfd_shutdown(void) {
    for (int r = 0; r < QFD_T_N; r++)
        for (int i = 0; i < QFD_T_RING; i++)
            for (int j = 0; j < 2; j++)
                if (g_qfd_ev[r][i][j]) { cudaEventDestroy(g_qfd_ev[r][i][j]); g_qfd_ev[r][i][j] = NULL; }
    for (int r = 0; r < QFD_T_N; r++) { g_qfd_n[r] = 0; g_qfd_ms[r] = 0.0; g_qfd_calls[r] = 0; }
    g_qfd_timing = -1;
    g_qfd_mode = -1;
}

// ---------------- fused group entry point ----------------

int qfd_gemv_group(const QfDenseProj *projs, int nproj, const float *x, int in,
                   cudaStream_t s) {
    if (qfd_fused_enabled() && qfd_group_eligible(projs, nproj, x, in)) {
        QfDenseGroupK g;
        memset(&g, 0, sizeof(g));
        g.x = x;
        g.in = in;
        g.rows_total = 0;
        for (int i = 0; i < nproj; i++) {
            g.p[i] = projs[i];
            g.rows_total += projs[i].rows;
        }
        const int grid = (g.rows_total + QFD_WARPS - 1) / QFD_WARPS;
        const size_t smem = (size_t)in * sizeof(float);
        if (g_qfd_fp8 && !(in & 63)) {
            switch (nproj) {
            case 1: k_gemv_fp8_group<1><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 2: k_gemv_fp8_group<2><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 3: k_gemv_fp8_group<3><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 4: k_gemv_fp8_group<4><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            }
            if (cudaGetLastError() == cudaSuccess) return 0;
        } else {
            switch (nproj) {
            case 1: k_gemv_bf16_group<1><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 2: k_gemv_bf16_group<2><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 3: k_gemv_bf16_group<3><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            case 4: k_gemv_bf16_group<4><<<grid, QFD_BLOCK, smem, s>>>(g); break;
            }
            if (cudaGetLastError() == cudaSuccess)  // launch-time config check only
                return 0;                           // (async; no sync in decode)
        }
        fprintf(stderr, "qfd: fused group launch failed (%dx%d, %d proj), using per-proj fallback\n",
                g.rows_total, in, nproj);
        // fall through: recompute every member with the legacy dispatcher
    }
    for (int i = 0; i < nproj; i++)
        qf_gemv_bf16(projs[i].W, x, projs[i].y, projs[i].rows, in, s);
    return 1;
}

// ---------------- decode call-site wrappers ----------------

int qfd_gdn_inproj(const void *wqkv, const void *wz, const void *wa, const void *wb,
                   const float *x, float *yqkv, float *yz, float *ya, float *yb,
                   cudaStream_t s) {
    const QfDenseProj p[4] = {
        { wqkv, yqkv, DINN }, { wz, yz, GDN_VDIM }, { wa, ya, DTRANK }, { wb, yb, DTRANK }
    };
    qfd_t_begin(QFD_T_GDN_IN, s);
    int rc = qfd_gemv_group(p, 4, x, NEMBD, s);
    qfd_t_end(QFD_T_GDN_IN, s);
    return rc;
}

int qfd_qsa_inproj4(const void *wq, const void *wk, const void *wv, const void *widx,
                    const float *x, float *yq, float *yk, float *yv, float *yidx, cudaStream_t s) {
    const QfDenseProj p[4] = {
        { wq, yq, NHEAD * QGATE }, { wk, yk, NKV * HDIM }, { wv, yv, NKV * HDIM }, { widx, yidx, 640 }
    };
    qfd_t_begin(QFD_T_QSA_IN, s);
    int rc = qfd_gemv_group(p, 4, x, NEMBD, s);
    qfd_t_end(QFD_T_QSA_IN, s);
    return rc;
}
int qfd_qsa_inproj(const void *wq, const void *wk, const void *wv,
                   const float *x, float *yq, float *yk, float *yv, cudaStream_t s) {
    const QfDenseProj p[3] = {
        { wq, yq, NHEAD * QGATE }, { wk, yk, NKV * HDIM }, { wv, yv, NKV * HDIM }
    };
    qfd_t_begin(QFD_T_QSA_IN, s);
    int rc = qfd_gemv_group(p, 3, x, NEMBD, s);
    qfd_t_end(QFD_T_QSA_IN, s);
    return rc;
}

int qfd_moe_inproj(const void *wrouter, const void *wshg, const void *wshu, const void *wgi,
                   const float *x, float *yr, float *yg, float *yu, float *ygi,
                   cudaStream_t s) {
    const QfDenseProj p[4] = {
        { wrouter, yr, NEXP }, { wshg, yg, NFF }, { wshu, yu, NFF }, { wgi, ygi, 1 }
    };
    qfd_t_begin(QFD_T_MOE_IN, s);
    int rc = qfd_gemv_group(p, 4, x, NEMBD, s);
    qfd_t_end(QFD_T_MOE_IN, s);
    return rc;
}

int qfd_out_proj(const void *w, const float *x, float *y, int out, int in, cudaStream_t s) {
    const QfDenseProj p[1] = { { w, y, out } };
    qfd_t_begin(QFD_T_OUT_PROJ, s);
    int rc = qfd_gemv_group(p, 1, x, in, s);
    qfd_t_end(QFD_T_OUT_PROJ, s);
    return rc;
}

// Output head strategy: one fused launch, 248320 rows / 8 warps = 31040
// blocks streaming 1.27 GB of BF16 weights with zero activation conversion
// and zero workspace. This is the single largest dense read per token; at
// GB10 LPDDR5x bandwidth it bounds decode at ~200 tok/s from the head alone,
// far above the >20 tok/s target, so no split-K or vocab sharding is needed.
int qfd_lm_head(const void *w, const float *x, float *logits, cudaStream_t s) {
    const QfDenseProj p[1] = { { w, logits, NVOCAB } };
    qfd_t_begin(QFD_T_LM_HEAD, s);
    int rc = qfd_gemv_group(p, 1, x, NEMBD, s);
    qfd_t_end(QFD_T_LM_HEAD, s);
    return rc;
}
