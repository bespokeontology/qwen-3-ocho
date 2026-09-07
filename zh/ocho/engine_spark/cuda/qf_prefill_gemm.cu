// qf_prefill_gemm.cu - dense projections of a PREFILL CHUNK on the fp8 tensor
// core (sm_121a): y[T][rows] = W[rows][K] . x[T][K]^T with the production FP8
// slab weights (E4M3 [rows][K], UE4M3 scale per row per 64-K block, scales
// immediately before the base pointer - qfd_quant_fp8) and the activations
// quantized per token in the SAME form (E4M3 + UE4M3 per 64-K block).
// mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32, fragment map pinned by
// tools/fp8probe (maxerr 0 vs scalar). CTA = 4 warps x 16 rows, PG_NG groups of
// 8 tokens per weight pass (64 tokens): every weight byte is read once per 64
// tokens instead of once per 16 (the old qfd_gemv_group_T sub-tiles) and the
// multiply is hardware. Per-K-tile block scales are applied on the tile's own
// accumulator (d = A.B for the tile, acc += s_row * s_tok * d).
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qf_dense.h"
#ifndef PG_NG
#define PG_NG 8
#endif

static __device__ __forceinline__ float pg_ue4m3_dec(uint32_t b) {
    uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
}
static __device__ __forceinline__ uint32_t pg_ue4m3_enc_ceil(float v) {
    if (!(v > 0.f)) return 0u;
    uint32_t b = __float_as_uint(v);
    int e32 = (int)(b >> 23) - 127;
    if (e32 < -6) { uint32_t m = (uint32_t)ceilf(v * 512.f); if (!m) m = 1; return m > 7 ? 8u : m; }
    int e = e32 + 7;
    uint32_t m = ((b & 0x7FFFFFu) + 0xFFFFFu) >> 20;
    if (m == 8u) { e++; m = 0; }
    return e > 15 ? 0x7Eu : (uint32_t)((e << 3) | m);
}
static __device__ __forceinline__ void pg_mma_e4m3(float &d0, float &d1, float &d2, float &d3,
                                                   uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


// Quad-cooperative fragment loads: the 4 lanes of a quad each load 8 contiguous
// bytes of the 32-byte K-tile row (one full sector per row per instruction, no
// evict-first re-fetch of the other half), then two shuffles hand every lane its
// own 4-byte fragment pieces (bytes [4p,4p+4) and [16+4p,16+4p+4)).
static __device__ __forceinline__ uint2 qld8(const uint8_t *row_tile, int p) { return __ldcs((const uint2 *)(row_tile + 8 * p)); }
static __device__ __forceinline__ void qsplit(uint2 w, int lane, int p, uint32_t &lo_frag, uint32_t &hi_frag) {
    const int base = lane & ~3;
    const uint32_t x0 = __shfl_sync(0xffffffffu, w.x, base + (p >> 1)), y0 = __shfl_sync(0xffffffffu, w.y, base + (p >> 1));
    const uint32_t x1 = __shfl_sync(0xffffffffu, w.x, base + 2 + (p >> 1)), y1 = __shfl_sync(0xffffffffu, w.y, base + 2 + (p >> 1));
    lo_frag = (p & 1) ? y0 : x0;      // bytes [4p, 4p+4)
    hi_frag = (p & 1) ? y1 : x1;      // bytes [16+4p, 16+4p+4)
}

// x [T][K] fp32 -> xq [T][K] E4M3 (round-to-nearest, satfinite), xs [T][K/64] UE4M3 (ceil, |v|/448).
// grid (ceil(K/64/128), T), block 128: thread = one 64-block of one row.
__global__ void k_pg_quant_rows(const float *__restrict__ x, int K, uint8_t *__restrict__ xq, uint8_t *__restrict__ xs) {
    const int t = blockIdx.y, blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= K / 64) return;
    const float *xr = x + (size_t)t * K + blk * 64;
    float am = 0.f;
    #pragma unroll 8
    for (int j = 0; j < 64; j++) am = fmaxf(am, fabsf(xr[j]));
    const uint32_t sb = pg_ue4m3_enc_ceil(am * (1.f / 448.f));
    const float inv = 1.f / fmaxf(pg_ue4m3_dec(sb), 1e-30f);
    uint8_t *q = xq + (size_t)t * K + blk * 64;
    #pragma unroll 8
    for (int j = 0; j < 64; j += 4) {
        __nv_fp8x4_e4m3 v(make_float4(xr[j] * inv, xr[j + 1] * inv, xr[j + 2] * inv, xr[j + 3] * inv));
        *(uint32_t *)(q + j) = *(uint32_t *)&v;
    }
    xs[(size_t)t * (K / 64) + blk] = (uint8_t)sb;
}

// grid (ceil(rows/64), ceil(T/(8*PG_NG))), block 128. y [T][rows] fp32.
__global__ __launch_bounds__(128, 4) void k_pg_gemm(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S, int rows, int K,
                                                 const uint8_t *__restrict__ xq, const uint8_t *__restrict__ xs, int T,
                                                 float *__restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, p = lane & 3;
    const int row0 = blockIdx.x * 64 + warp * 16;
    if (row0 >= rows) return;
    const int rA = row0 + g, rB = row0 + g + 8;
    const int rAc = rA < rows ? rA : rows - 1, rBc = rB < rows ? rB : rows - 1;   // clamped loads, guarded stores
    const int n64 = K >> 6;
    const uint8_t *wA = W + (size_t)rAc * K, *wB = W + (size_t)rBc * K;
    const uint8_t *sA = S + (size_t)rAc * n64, *sB = S + (size_t)rBc * n64;
    const int t0 = blockIdx.y * (8 * PG_NG);
    const int ng0 = (T - t0 + 7) / 8, ng = ng0 < PG_NG ? ng0 : PG_NG;
    const uint8_t *bq[PG_NG]; const uint8_t *bsA[PG_NG], *bsB[PG_NG];
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) {
        int tk = t0 + q * 8 + g; if (tk > T - 1) tk = T - 1;                // B column g = token tk
        bq[q] = xq + (size_t)tk * K;
        int ta = t0 + q * 8 + 2 * p; if (ta > T - 1) ta = T - 1;           // D columns 2p, 2p+1
        int tb = ta + 1; if (tb > T - 1) tb = T - 1;
        bsA[q] = xs + (size_t)ta * n64; bsB[q] = xs + (size_t)tb * n64;
    }
    float acc[PG_NG][4];
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = 0.f;
    // Software pipeline: the fragments of K-tile k+32 are in flight while tile k's
    // MMAs run (the plain load->mma loop was latency-bound at 1024-token chunks).
    uint2 fA = qld8(wA, p), fB = qld8(wB, p);
    uint32_t b0[PG_NG], b1[PG_NG];
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) { b0[q] = *(const uint32_t *)(bq[q] + 4 * p); b1[q] = *(const uint32_t *)(bq[q] + 16 + 4 * p); }
    for (int k0 = 0; k0 < K; k0 += 32) {
        const int kn = k0 + 32;
        uint2 nA = make_uint2(0u, 0u), nB = nA; uint32_t nb0[PG_NG], nb1[PG_NG];
        if (kn < K) {
            nA = qld8(wA + kn, p); nB = qld8(wB + kn, p);
            #pragma unroll
            for (int q = 0; q < PG_NG; q++) { nb0[q] = *(const uint32_t *)(bq[q] + kn + 4 * p); nb1[q] = *(const uint32_t *)(bq[q] + kn + 16 + 4 * p); }
        } else {
            #pragma unroll
            for (int q = 0; q < PG_NG; q++) { nb0[q] = 0u; nb1[q] = 0u; }
        }
        const int kb = k0 >> 6;
        const float sa = pg_ue4m3_dec(sA[kb]), sb = pg_ue4m3_dec(sB[kb]);
        uint32_t a0, a1, a2, a3;
        qsplit(fA, lane, p, a0, a2); qsplit(fB, lane, p, a1, a3);
        #pragma unroll
        for (int q = 0; q < PG_NG; q++) {
            if (q < ng) {
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                pg_mma_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0[q], b1[q]);
                const float xa = pg_ue4m3_dec(bsA[q][kb]), xb = pg_ue4m3_dec(bsB[q][kb]);
                acc[q][0] = fmaf(sa * xa, d0, acc[q][0]); acc[q][1] = fmaf(sa * xb, d1, acc[q][1]);
                acc[q][2] = fmaf(sb * xa, d2, acc[q][2]); acc[q][3] = fmaf(sb * xb, d3, acc[q][3]);
            }
        }
        fA = nA; fB = nB;
        #pragma unroll
        for (int q = 0; q < PG_NG; q++) { b0[q] = nb0[q]; b1[q] = nb1[q]; }
    }
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) {
        if (q < ng) {
            const int ta = t0 + q * 8 + 2 * p, tb = ta + 1;
            if (ta < T) { if (rA < rows) y[(size_t)ta * rows + rA] = acc[q][0]; if (rB < rows) y[(size_t)ta * rows + rB] = acc[q][2]; }
            if (tb < T) { if (rA < rows) y[(size_t)tb * rows + rA] = acc[q][1]; if (rB < rows) y[(size_t)tb * rows + rB] = acc[q][3]; }
        }
    }
}

// ============================================================================
// Pipelined form (cp.async, PG_ST stages of PG_KT K32-tiles per 64-row strip):
// the register-prefetch loop was memory-latency-bound (0.1 G mma/s against a
// measured 30 G mma/s tensor rate). Row stride 144 B for 128 B of nibbles keeps
// the fragment reads conflict-free (bank = (row*4 + p) mod 32 within a tile).
// ============================================================================
#define PG_ST 3
#define PG_KT 4
#define PG_RS 144
static __device__ __forceinline__ void pg_cp16(void *smem, const void *gmem) {
    const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}
static __device__ __forceinline__ void pg_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> static __device__ __forceinline__ void pg_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
#define PG_TH 2                                     // token halves per CTA (8 warps): one weight stage serves 128 tokens
struct PgStage { uint8_t w[64 * PG_RS]; uint8_t x[PG_TH * 64 * PG_RS]; };
static __device__ __forceinline__ void pg_issue_x(PgStage *st, const uint8_t *xq, int K, int T, int t0, int kt0, int tid) {
    // the CTA's 64 tokens x PG_KT*32 B (each token row of xq is K bytes); rows past T clamp to T-1
    for (int c = tid; c < PG_TH * 64 * (PG_KT * 2); c += 256) {
        const int r = c / (PG_KT * 2), ch = c % (PG_KT * 2);
        int tk = t0 + r; if (tk > T - 1) tk = T - 1;
        pg_cp16(st->x + r * PG_RS + ch * 16, xq + (size_t)tk * K + kt0 * 32 + ch * 16);
    }
}
static __device__ __forceinline__ void pg_issue(PgStage *st, const uint8_t *W, int K, int rows, int kt0, int tid) {
    // 64 rows x PG_KT*32 B at column kt0*32: 8 x 16 B chunks per row = 512 chunks over 256 threads
    for (int c = tid; c < 64 * (PG_KT * 2); c += 256) {
        const int r = c / (PG_KT * 2), ch = c % (PG_KT * 2);
        const int rr = r < rows ? r : rows - 1;
        pg_cp16(st->w + r * PG_RS + ch * 16, W + (size_t)rr * K + kt0 * 32 + ch * 16);
    }
}
__global__ __launch_bounds__(256) void k_pg_gemm2(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S, int rows, int K,
                                                  const uint8_t *__restrict__ xq, const uint8_t *__restrict__ xs, int T,
                                                  float *__restrict__ y) {
    extern __shared__ __align__(16) uint8_t pg_smem[];
    PgStage *stg = (PgStage *)pg_smem;
    const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, p = lane & 3;
    const int warp = (tid >> 5) & 3, th = tid >> 7;                  // 8 warps: warp = row strip of 16, th = token half
    const int row0 = blockIdx.x * 64;
    if (row0 >= rows) return;
    const int rA = warp * 16 + g, rB = rA + 8;                       // rows within the strip
    const int rAo = row0 + rA, rBo = row0 + rB;
    const int rAc = rAo < rows ? rAo : rows - 1, rBc = rBo < rows ? rBo : rows - 1;
    const int n64 = K >> 6;
    const uint8_t *Wstrip = W + (size_t)row0 * K;
    const int rows_left = rows - row0 < 64 ? rows - row0 : 64;
    const uint8_t *sA = S + (size_t)rAc * n64, *sB = S + (size_t)rBc * n64;
    const int tbase = blockIdx.y * (PG_TH * 8 * PG_NG);              // the CTA's 128 tokens; this warp's half starts at t0
    const int t0 = tbase + th * (8 * PG_NG);
    const int ng0 = (T - t0 + 7) / 8, ng = ng0 < PG_NG ? (ng0 < 0 ? 0 : ng0) : PG_NG;
    const uint8_t *bsA[PG_NG], *bsB[PG_NG];
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) {
        int ta = t0 + q * 8 + 2 * p; if (ta > T - 1) ta = T - 1;
        int tb = ta + 1; if (tb > T - 1) tb = T - 1;
        bsA[q] = xs + (size_t)ta * n64; bsB[q] = xs + (size_t)tb * n64;
    }
    float acc[PG_NG][4];
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = 0.f;
    const int NT = K / 32, NSTG = (NT + PG_KT - 1) / PG_KT;
    #pragma unroll
    for (int st = 0; st < PG_ST - 1; st++) {
        if (st < NSTG) { pg_issue(&stg[st], Wstrip, K, rows_left, st * PG_KT, tid); pg_issue_x(&stg[st], xq, K, T, tbase, st * PG_KT, tid); }
        pg_commit();
    }
    for (int sidx = 0; sidx < NSTG; sidx++) {
        pg_wait<PG_ST - 2>();
        __syncthreads();
        const int nxt = sidx + PG_ST - 1;
        if (nxt < NSTG) { pg_issue(&stg[nxt % PG_ST], Wstrip, K, rows_left, nxt * PG_KT, tid); pg_issue_x(&stg[nxt % PG_ST], xq, K, T, tbase, nxt * PG_KT, tid); }
        pg_commit();
        const PgStage *cs = &stg[sidx % PG_ST];
        #pragma unroll
        for (int j = 0; j < PG_KT; j++) {
            const int kt = sidx * PG_KT + j;
            if (kt < NT) {
                const int k0 = kt * 32, kb = k0 >> 6;
                uint32_t b0[PG_NG], b1[PG_NG];
                #pragma unroll
                for (int q = 0; q < PG_NG; q++) {           // B column g of group q = token row th*64 + q*8+g of the staged tile
                    b0[q] = *(const uint32_t *)(cs->x + (th * 64 + q * 8 + g) * PG_RS + j * 32 + 4 * p);
                    b1[q] = *(const uint32_t *)(cs->x + (th * 64 + q * 8 + g) * PG_RS + j * 32 + 16 + 4 * p);
                }
                const uint32_t a0 = *(const uint32_t *)(cs->w + rA * PG_RS + j * 32 + 4 * p), a1 = *(const uint32_t *)(cs->w + rB * PG_RS + j * 32 + 4 * p);
                const uint32_t a2 = *(const uint32_t *)(cs->w + rA * PG_RS + j * 32 + 16 + 4 * p), a3 = *(const uint32_t *)(cs->w + rB * PG_RS + j * 32 + 16 + 4 * p);
                const float sa = pg_ue4m3_dec(sA[kb]), sb = pg_ue4m3_dec(sB[kb]);
                #pragma unroll
                for (int q = 0; q < PG_NG; q++) {
                    if (q < ng) {
                        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                        pg_mma_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0[q], b1[q]);
                        const float xa = pg_ue4m3_dec(bsA[q][kb]), xb = pg_ue4m3_dec(bsB[q][kb]);
                        acc[q][0] = fmaf(sa * xa, d0, acc[q][0]); acc[q][1] = fmaf(sa * xb, d1, acc[q][1]);
                        acc[q][2] = fmaf(sb * xa, d2, acc[q][2]); acc[q][3] = fmaf(sb * xb, d3, acc[q][3]);
                    }
                }
            }
        }
        __syncthreads();
    }
    pg_wait<0>();
    #pragma unroll
    for (int q = 0; q < PG_NG; q++) {
        if (q < ng) {
            const int ta = t0 + q * 8 + 2 * p, tb = ta + 1;
            if (ta < T) { if (rAo < rows) y[(size_t)ta * rows + rAo] = acc[q][0]; if (rBo < rows) y[(size_t)ta * rows + rBo] = acc[q][2]; }
            if (tb < T) { if (rAo < rows) y[(size_t)tb * rows + rAo] = acc[q][1]; if (rBo < rows) y[(size_t)tb * rows + rBo] = acc[q][3]; }
        }
    }
}


// =============================================================================
// cuBLASLt MXFP8 bulk GEMM path (QF_PF_GEMM3, default on). Stolen bulk prefill on the
// NVIDIA side: the chunk pass's dense projections run as block-scaled fp8 GEMMs (e4m3 data,
// ue8m0 per-32 scales on BOTH operands) on the GB10 tensor cores through cuBLASLt:
// 118-166 TFLOPS measured on the Qwen shapes (tools/cublaslt_mx_probe.cu) against ~28 for
// the hand-written m16n8k32 kernel above. Scale tensor layout (found by the probe against an
// exact reference): tiles of 128 rows x 4 k-blocks = 512 B, tile grid row-major with row
// tiles outer, in-tile offset (r%32)*16 + ((r/32)%4)*4 + (kb%4). Rows and K must be
// multiples of 128: weights are zero-padded at registration (the hyper-connection's 320
// becomes 384 - its callers use the padded stride), projections whose row count is not a
// multiple of 128 (the 48-row GDN gates) stay on the kernel above. Weights are MX-quantized
// ONCE at load from the bf16 originals (qfd_mx_register, called by the slab conversion before
// the bf16 is freed); activations are MX-quantized per call. Decode keeps its per-64 UE4M3
// slabs: prefill and decode differ in rounding, as they already did.
// =============================================================================
#include <cublasLt.h>
#include <cuda_bf16.h>

static __device__ __forceinline__ size_t pg_mx_sfoff(int r, int kb, int KBp) {
    return ((size_t)(r >> 7) * (KBp >> 2) + (kb >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (kb & 3);
}
// smallest power-of-two scale with absmax/scale <= 448, as the ue8m0 exponent byte
static __device__ __forceinline__ int pg_mx_exp(float a) {
    if (!(a > 0.f)) return -127;
    int e = (int)ceilf(log2f(a * (1.f / 448.f)));
    if (a * exp2f((float)-e) > 448.f) e++;
    if (e > -127 && a * exp2f((float)-(e - 1)) <= 448.f) e--;
    if (e < -127) e = -127; if (e > 127) e = 127;
    return e;
}
// grid (K_p/128, rows_p), block 128: warp = one 32-block, lane = one element
template<typename SRC>
__global__ void k_pg_mx_quant(const SRC *__restrict__ X, int rows, int in, int ldx, int K_p,
                              uint8_t *__restrict__ q, uint8_t *__restrict__ sc) {
    const int r = blockIdx.y, kb = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int k = kb * 32 + lane;
    float v = 0.f;
    if (r < rows && k < in) {
        if constexpr (sizeof(SRC) == 2) v = __bfloat162float(((const __nv_bfloat16 *)X)[(size_t)r * ldx + k]);
        else v = ((const float *)X)[(size_t)r * ldx + k];
    }
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const int e = pg_mx_exp(a);
    const float inv = exp2f((float)-e);
    q[(size_t)r * K_p + k] = (uint8_t)__nv_cvt_float_to_fp8(v * inv, __NV_SATFINITE, __NV_E4M3);
    if (lane == 0) sc[pg_mx_sfoff(r, kb, K_p >> 5)] = (uint8_t)(e + 127);
}
struct PgMxW { const void *key; uint8_t *q, *sc; int rows, in, rows_p, K_p; };
static int g_pg_mx_pad_rows = 0;   // hc path: accept an output written with this padded row stride
static PgMxW g_mxw[1024]; static int g_nmxw = 0;
static inline int pg_ceil128(int v) { return (v + 127) & ~127; }
static int pg_mx_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_PF_GEMM3"); v = (e && e[0] == '0') ? 0 : 1; } return v; }
// Called at load with the bf16 tensor still resident; key = the fp8 slab pointer the prefill callers pass.
int qfd_mx_register(const void *bf16, const void *key, int rows, int in) {   // C++ linkage: declared extern int in qf.cu
    if (!pg_mx_on() || !bf16 || !key || rows <= 0 || in <= 0 || g_nmxw >= 1024) return -1;
    if (rows > 65535) return 1;                                  // lm_head: never a chunk GEMM (grid.y limit); no copy
    if (rows % 128 != 0 && rows != 320) return 1;               // stays on the kernel path (48-row GDN gates)

    PgMxW w; w.key = key; w.rows = rows; w.in = in; w.rows_p = pg_ceil128(rows); w.K_p = pg_ceil128(in);
    if (cudaMalloc(&w.q, (size_t)w.rows_p * w.K_p) != cudaSuccess) return -1;
    if (cudaMalloc(&w.sc, (size_t)(w.rows_p / 128) * (w.K_p / 128) * 512) != cudaSuccess) { cudaFree(w.q); return -1; }
    k_pg_mx_quant<__nv_bfloat16><<<dim3(w.K_p / 128, w.rows_p), 128, 0, 0>>>((const __nv_bfloat16 *)bf16, rows, in, in, w.K_p, w.q, w.sc);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(w.q); cudaFree(w.sc); return -1; }
    g_mxw[g_nmxw++] = w;
    if (g_nmxw % 100 == 0 || g_nmxw < 3) fprintf(stderr, "pf mx: %d MX weight copies registered (last %dx%d -> %dx%d)\n", g_nmxw, rows, in, w.rows_p, w.K_p);
    return 0;
}
static const PgMxW *pg_mx_find(const void *key) {
    for (int i = 0; i < g_nmxw; i++) if (g_mxw[i].key == key) return &g_mxw[i];
    return NULL;
}
// per-shape cuBLASLt objects
struct PgMxPlan { int rows_p, K_p, T; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t lA, lB, lD; cublasLtMatmulAlgo_t algo; size_t ws; };
static PgMxPlan g_mxplan[256]; static int g_nmxplan = 0;
static cublasLtHandle_t g_lt = NULL; static void *g_ltws = NULL; static const size_t g_ltws_bytes = (size_t)64 << 20;
static uint8_t *g_mxq = NULL, *g_mxs = NULL; static int g_mxq_T = 0;   // activation MX scratch [T_p][K_p<=10240]
static PgMxPlan *pg_mx_plan(int rows_p, int K_p, int T) {
    for (int i = 0; i < g_nmxplan; i++) if (g_mxplan[i].rows_p == rows_p && g_mxplan[i].K_p == K_p && g_mxplan[i].T == T) return &g_mxplan[i];
    if (g_nmxplan >= 256) return NULL;
    if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return NULL;
    if (!g_ltws && cudaMalloc(&g_ltws, g_ltws_bytes) != cudaSuccess) return NULL;
    PgMxPlan pl; pl.rows_p = rows_p; pl.K_p = K_p; pl.T = T;
    if (cublasLtMatmulDescCreate(&pl.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return NULL;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA);
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm);
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm);
    cublasLtMatrixLayoutCreate(&pl.lA, CUDA_R_8F_E4M3, K_p, rows_p, K_p);
    cublasLtMatrixLayoutCreate(&pl.lB, CUDA_R_8F_E4M3, K_p, T, K_p);
    cublasLtMatrixLayoutCreate(&pl.lD, CUDA_R_32F, rows_p, T, rows_p);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ltws_bytes, sizeof g_ltws_bytes);
    cublasLtMatmulHeuristicResult_t heur[1]; int nres = 0;
    // scale pointers are set per call; the heuristic needs the modes only
    const uint8_t *dummy = g_mxs ? g_mxs : (const uint8_t *)g_ltws;
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof dummy);
    cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof dummy);
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, pl.op, pl.lA, pl.lB, pl.lD, pl.lD, pref, 1, heur, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs != CUBLAS_STATUS_SUCCESS || nres < 1) { fprintf(stderr, "pf mx: no cuBLASLt algo for %d x %d x T=%d (status %d)\n", rows_p, K_p, T, (int)hs); return NULL; }
    pl.algo = heur[0].algo; pl.ws = heur[0].workspaceSize;
    g_mxplan[g_nmxplan] = pl;
    return &g_mxplan[g_nmxplan++];
}
// activations: X fp32 [T][ldx] (in valid columns) -> g_mxq [T_p][K_p] + g_mxs tiles
static int pg_mx_act_ensure(int T, int K_p);
static int pg_mx_quant_act(const float *X, int in, int ldx, int T, int K_p, cudaStream_t s) {
    const int T_p = pg_ceil128(T);
    if (pg_mx_act_ensure(T, K_p)) { fprintf(stderr, "pf mx: activation scratch (T %d K_p %d) failed\n", T, K_p); return -1; }
    { cudaError_t pe = cudaGetLastError(); if (pe != cudaSuccess) fprintf(stderr, "pf mx: pending error BEFORE act quant (in %d T %d): %s\n", in, T, cudaGetErrorString(pe)); }
    k_pg_mx_quant<float><<<dim3(K_p / 128, T_p), 128, 0, s>>>(X, T, in, ldx, K_p, g_mxq, g_mxs);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "pf mx: act quant launch (in %d ldx %d T %d K_p %d T_p %d): %s\n", in, ldx, T, K_p, T_p, cudaGetErrorString(e)); return -1; } }
    return 0;
}
static int pg_mx_gemm(const PgMxW *w, int T, float *y, cudaStream_t s) {
    PgMxPlan *pl = pg_mx_plan(w->rows_p, w->K_p, T);
    if (!pl) return -1;
    const uint8_t *sa = w->sc, *sb = g_mxs;
    cublasLtMatmulDescSetAttribute(pl->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof sa);
    cublasLtMatmulDescSetAttribute(pl->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof sb);
    const float one = 1.f, zero = 0.f;
    cublasStatus_t st = cublasLtMatmul(g_lt, pl->op, &one, w->q, pl->lA, g_mxq, pl->lB, &zero, y, pl->lD, y, pl->lD,
                                       &pl->algo, g_ltws, g_ltws_bytes, s);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "pf mx: cublasLtMatmul status %d (%d x %d x T=%d)\n", (int)st, w->rows_p, w->K_p, T); return -1; }
    return 0;
}

// ---- fused hyper-connection / router glue (operator 09-06: fewer HBM traversals) ----
// Every kernel below consumes what the previous one produced while it is still in registers or
// L2, and writes the MX activation (e4m3 + ue8m0 tiles) the following cuBLASLt GEMM reads, so
// the separate quantization passes over [T][10240] / [T][2560] disappear. Exact Qwen shapes.
static int pg_mx_act_ensure(int T, int K_p) {
    const int T_p = pg_ceil128(T);
    if (g_mxq_T < T_p) {
        if (g_mxq) cudaFree(g_mxq); if (g_mxs) cudaFree(g_mxs);
        if (cudaMalloc(&g_mxq, (size_t)T_p * 10240) != cudaSuccess) { g_mxq = NULL; g_mxq_T = 0; return -1; }
        if (cudaMalloc(&g_mxs, (size_t)(T_p / 128) * (10240 / 128) * 512) != cudaSuccess) { g_mxs = NULL; g_mxq_T = 0; return -1; }
        g_mxq_T = T_p;
    }
    return K_p > 10240 ? -1 : 0;
}
extern "C" int qfd_gemm_mx_prequant(const QfDenseProj *projs, int nproj, int in, int T, int pad_rows, cudaStream_t s) {
    if (!pg_mx_on() || nproj < 1 || (in % 128) || !g_mxq) return -1;
    for (int p = 0; p < nproj; p++) {                           // all-or-nothing: the caller falls back as a group
        const PgMxW *w = pg_mx_find(projs[p].W);
        if (!w || w->K_p != in || (w->rows_p != w->rows && w->rows_p != pad_rows)) return -1;
    }
    for (int p = 0; p < nproj; p++)
        if (pg_mx_gemm(pg_mx_find(projs[p].W), T, projs[p].y, s)) return -1;
    return 0;
}
static __device__ __forceinline__ float pg_bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
static __device__ __forceinline__ float pg_e4m3_dec(uint8_t v) {
    const uint32_t sgn = v >> 7, e = (v >> 3) & 15u, m = v & 7u;
    float f = (e == 0) ? ((float)m * 0x1p-9f) : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return sgn ? -f : f;
}
static __device__ __forceinline__ float pg_blocksum256(float v, float (*red)[8], int slot) {   // 256 threads, 8 warps
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) red[slot][threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    #pragma unroll
    for (int w = 0; w < 8; w++) t += red[slot][w];
    return t;
}
static __device__ __forceinline__ void pg_mx_store_warp(float n, int t, int idx, int b, int KBp, uint8_t *__restrict__ q, uint8_t *__restrict__ sc, int K_p) {
    float a = fabsf(n);
    #pragma unroll
    for (int off = 16; off; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const int e = pg_mx_exp(a);
    q[(size_t)t * K_p + idx] = (uint8_t)__nv_cvt_float_to_fp8(n * exp2f((float)-e), __NV_SATFINITE, __NV_E4M3);
    if ((threadIdx.x & 31) == 0) sc[pg_mx_sfoff(t, b, KBp)] = (uint8_t)(e + 127);
}
// F1: [lazy inject R += inj_in (x) y_in] -> per-2560-group RMS norm (x * inv * (1 + w)) -> normed (fp32,
// needed by the mix) + MX(normed) for the down GEMM + injection gate inj[c] = 2 sigmoid(w_inject[c].normed / 4).
// One block per row, 256 threads x 40 elements; a warp holds one 32-element MX block per step.
__global__ void __launch_bounds__(256) k_hc_front_rows(float *__restrict__ R, const uint16_t *__restrict__ w_norm,
                                                       const uint16_t *__restrict__ w_inject, float *__restrict__ normed,
                                                       uint8_t *__restrict__ q, uint8_t *__restrict__ sc, float *__restrict__ inj,
                                                       const float *__restrict__ yin, const float *__restrict__ injin) {
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5;
    float *Rr = R + (size_t)t * 10240; float *nr = normed + (size_t)t * 10240;
    __shared__ float ys[2560]; __shared__ float red[4][8]; __shared__ float inv[4]; __shared__ float injv[4]; __shared__ float red2[4][8];
    if (yin) { for (int j = tid; j < 2560; j += 256) ys[j] = yin[(size_t)t * 2560 + j]; if (tid < 4) injv[tid] = injin[(size_t)t * 4 + tid]; }
    __syncthreads();
    float v[40]; float ss0 = 0.f, ss1 = 0.f, ss2 = 0.f, ss3 = 0.f;
    #pragma unroll
    for (int k = 0; k < 40; k++) {
        const int idx = k * 256 + tid, c = k / 10;
        float x = Rr[idx];
        if (yin) { x += injv[c] * ys[idx - c * 2560]; Rr[idx] = x; }
        v[k] = x;
        if (c == 0) ss0 += x * x; else if (c == 1) ss1 += x * x; else if (c == 2) ss2 += x * x; else ss3 += x * x;
    }
    const float t0 = pg_blocksum256(ss0, red, 0), t1 = pg_blocksum256(ss1, red, 1), t2 = pg_blocksum256(ss2, red, 2), t3 = pg_blocksum256(ss3, red, 3);
    if (tid == 0) { inv[0] = rsqrtf(t0 / 2560.f + 1e-6f); inv[1] = rsqrtf(t1 / 2560.f + 1e-6f); inv[2] = rsqrtf(t2 / 2560.f + 1e-6f); inv[3] = rsqrtf(t3 / 2560.f + 1e-6f); }
    __syncthreads();
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    #pragma unroll
    for (int k = 0; k < 40; k++) {
        const int idx = k * 256 + tid, c = k / 10;
        const float n = v[k] * inv[c] * (1.f + pg_bf2f(w_norm[idx]));
        nr[idx] = n;
        pg_mx_store_warp(n, t, idx, k * 8 + warp, 320, q, sc, 10240);
        a0 += pg_bf2f(w_inject[idx]) * n; a1 += pg_bf2f(w_inject[10240 + idx]) * n;
        a2 += pg_bf2f(w_inject[20480 + idx]) * n; a3 += pg_bf2f(w_inject[30720 + idx]) * n;
    }
    __syncthreads();
    const float g0 = pg_blocksum256(a0, red2, 0), g1 = pg_blocksum256(a1, red2, 1), g2 = pg_blocksum256(a2, red2, 2), g3 = pg_blocksum256(a3, red2, 3);
    if (tid == 0) { inj[(size_t)t * 4 + 0] = 2.f / (1.f + expf(-g0 * 0.25f)); inj[(size_t)t * 4 + 1] = 2.f / (1.f + expf(-g1 * 0.25f));
                    inj[(size_t)t * 4 + 2] = 2.f / (1.f + expf(-g2 * 0.25f)); inj[(size_t)t * 4 + 3] = 2.f / (1.f + expf(-g3 * 0.25f)); }
}
// F3: d (hc-down output, padded stride 384) -> silu(d/4) -> MX for the up GEMM (K = 384). Block = row, 384 threads.
__global__ void __launch_bounds__(384) k_hc_silu4_quant(const float *__restrict__ d, uint8_t *__restrict__ q, uint8_t *__restrict__ sc) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const float x = d[(size_t)t * 384 + tid] * 0.25f;
    const float sv = x / (1.f + expf(-x));
    pg_mx_store_warp(sv, t, tid, tid >> 5, 12, q, sc, 384);
}
// F5: mixed[j] = (1/4) sum_c sigmoid(up[c][j]) normed[c][j] -> mixed (fp32) + MX(mixed) for the following
// projection groups (K = 2560) + optional shared-expert gate logit shgi = w_sgi . mixed (fp8 slab weight).
__global__ void __launch_bounds__(256) k_hc_mix_rows2(const float *__restrict__ up, const float *__restrict__ normed,
                                                      float *__restrict__ mixed, uint8_t *__restrict__ q, uint8_t *__restrict__ sc,
                                                      const uint8_t *__restrict__ w_sgi, float *__restrict__ shgi) {
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5;
    const float *ur = up + (size_t)t * 10240, *nr = normed + (size_t)t * 10240;
    __shared__ float red[4][8];
    const uint8_t *S = w_sgi ? w_sgi - 48 : NULL;                      // slab: 40 scale bytes padded to 48 before the data
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 10; k++) {
        const int j = k * 256 + tid;
        float m = 0.f;
        #pragma unroll
        for (int c = 0; c < 4; c++) { const float u = ur[c * 2560 + j]; m += nr[c * 2560 + j] / (1.f + expf(-u)); }
        m *= 0.25f;
        mixed[(size_t)t * 2560 + j] = m;
        pg_mx_store_warp(m, t, j, k * 8 + warp, 80, q, sc, 2560);
        if (w_sgi) acc += pg_e4m3_dec(w_sgi[j]) * pg_ue4m3_dec(S[j >> 6]) * m;
    }
    if (w_sgi) { const float tot = pg_blocksum256(acc, red, 0); if (tid == 0) shgi[t] = tot; }
}
// F7: shared expert activation silu(g) * u -> MX for the down GEMM (K = 640). Block = row, 640 threads.
__global__ void __launch_bounds__(640) k_shexp_act_quant(const float *__restrict__ g, const float *__restrict__ u, uint8_t *__restrict__ q, uint8_t *__restrict__ sc) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const float gv = g[(size_t)t * 640 + tid];
    const float h = gv / (1.f + expf(-gv)) * u[(size_t)t * 640 + tid];
    pg_mx_store_warp(h, t, tid, tid >> 5, 20, q, sc, 640);
}
// Fused hyper-connection over T rows: F1 -> down GEMM (MX, 384-padded) -> F3 -> up GEMM (MX) -> F5.
// Leaves MX(mixed) in the activation scratch for the caller's next projection group (qfd_gemm_mx_prequant).
extern "C" int qfd_hc_fused_rows(float *R, const void *w_norm, const void *w_down, const void *w_up, const void *w_inject,
                                 float *d, float *up, float *normed, float *mixed, float *inj,
                                 const float *yin, const float *injin, const void *w_sgi, float *shgi, int T, cudaStream_t s) {
    if (!pg_mx_on() || T < 1) return -1;
    const PgMxW *wd = pg_mx_find(w_down), *wu = pg_mx_find(w_up);
    if (!wd || !wu || wd->rows_p != 384 || wd->K_p != 10240 || wu->K_p != 384 || wu->rows_p != 10240) return -1;
    if (pg_mx_act_ensure(T, 10240)) return -1;
    k_hc_front_rows<<<T, 256, 0, s>>>(R, (const uint16_t *)w_norm, (const uint16_t *)w_inject, normed, g_mxq, g_mxs, inj, yin, injin);
    if (pg_mx_gemm(wd, T, d, s)) return -1;
    k_hc_silu4_quant<<<T, 384, 0, s>>>(d, g_mxq, g_mxs);
    if (pg_mx_gemm(wu, T, up, s)) return -1;
    k_hc_mix_rows2<<<T, 256, 0, s>>>(up, normed, mixed, g_mxq, g_mxs, (const uint8_t *)w_sgi, shgi);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
// GDN post: FLA output (bf16 [T][48][128]) -> per-head RMS norm x weight[i] x sigmoid(z) -> MX for the
// out-projection (K = 6144). Replaces unpack (bf16->fp32), gated norm (fp32->bf16), bf16->fp32 and the
// GEMM's own quantization pass: one read of o and z, one MX write. Grid (48, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_post_quant(const __nv_bfloat16 *__restrict__ o, const float *__restrict__ z,
                                                        const __nv_bfloat16 *__restrict__ w, uint8_t *__restrict__ q, uint8_t *__restrict__ sc) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const float v = __bfloat162float(o[((size_t)t * 48 + h) * 128 + i]);
    __shared__ float red[4];
    float ss = v * v;
    #pragma unroll
    for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / 128.f + 1e-6f);
    const float g = z[(size_t)t * 6144 + h * 128 + i];
    const float y = __bfloat162float(w[i]) * (v * inv) * (1.f / (1.f + expf(-g)));
    pg_mx_store_warp(y, t, h * 128 + i, h * 4 + (i >> 5), 192, q, sc, 6144);
}
extern "C" int qfd_gdn_post_quant(const void *o_bf16, const float *z, const void *w, int T, cudaStream_t s) {
    if (pg_mx_act_ensure(T, 6144)) return -1;
    k_gdn_post_quant<<<dim3(48, T), 128, 0, s>>>((const __nv_bfloat16 *)o_bf16, z, (const __nv_bfloat16 *)w, g_mxq, g_mxs);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
extern "C" int qfd_shexp_act_quant(const float *g, const float *u, int T, cudaStream_t s) {
    if (pg_mx_act_ensure(T, 640)) return -1;
    k_shexp_act_quant<<<T, 640, 0, s>>>(g, u, g_mxq, g_mxs);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int qfd_gemm_fp8_rows(const QfDenseProj *projs, int nproj, const float *xT, int in, int T,
                                 uint8_t *xq, uint8_t *xs, cudaStream_t s);
// ---- hyper-connection pieces over T rows (the per-row arithmetic of qf_hc_T / k_hc_up_fp8_col_T) ----
__global__ void k_pg_silu4_rows(float *__restrict__ d, int n) {           // d = silu(d/4)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = d[i] * 0.25f; d[i] = v / (1.f + expf(-v)); }
}
// mixed[t][j] = (1/HCC) sum_c sigmoid(up[t][c*NE+j]) * normed[t][c*NE+j]; grid (NE/256, T)
__global__ void k_pg_hc_mix_rows(const float *__restrict__ up, const float *__restrict__ normed,
                                 float *__restrict__ mixed, int NE, int HC) {
    const int t = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x;
    if (j >= NE) return;
    const float *u = up + (size_t)t * NE * HC, *nn = normed + (size_t)t * NE * HC;
    float sum = 0.f;
    for (int c = 0; c < HC; c++) sum += (1.f / (1.f + expf(-u[c * NE + j]))) * nn[c * NE + j];
    mixed[(size_t)t * NE + j] = sum / (float)HC;
}
// inj[t][c] = 2 sigmoid((w_inject[c] . normed[t]) / HC); grid (HC, T), block 256
__global__ void k_pg_hc_inj_rows(const float *__restrict__ normed, const uint16_t *__restrict__ w_inject,
                                 float *__restrict__ inj, int IN, int HC) {
    const int t = blockIdx.y, c = blockIdx.x;
    const float *nr = normed + (size_t)t * IN;
    const uint16_t *wr = w_inject + (size_t)c * IN;
    float a = 0.f;
    for (int j = threadIdx.x; j < IN; j += 256) a = fmaf(__uint_as_float((uint32_t)wr[j] << 16), nr[j], a);
    #pragma unroll
    for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f; for (int w = 0; w < 8; w++) tot += red[w];
        inj[(size_t)t * HC + c] = 2.f / (1.f + expf(-tot / (float)HC));
    }
}

extern "C" {
int qfd_fp8_active(void);
// Hyper-connection over T rows: normed [T][HC*NE] (already normed by the caller),
// w_down fp8 slab [HCL][HC*NE], w_up fp8 slab [HC*NE][HCL], w_inject bf16 [HC][HC*NE] (or NULL).
// d [T][HCL] and up [T][HC*NE] are scratch; mixed [T][NE], inj [T][HC] outputs.
int qfd_hc_fp8_rows(const float *normed, const void *w_down, const void *w_up, const void *w_inject,
                    float *d, float *up, float *mixed, float *inj, int T, int NE, int HC, int hcl,
                    uint8_t *xq, uint8_t *xs, cudaStream_t s) {
    const int IN = NE * HC;
    // MX path: w_down is registered padded to 384 rows and w_up to K = 384; d is written and read
    // with the padded stride (d has T x 384 room; its padded columns are silu(0) = 0 against zero weights).
    const PgMxW *wd = pg_mx_on() ? pg_mx_find(w_down) : NULL, *wu = pg_mx_on() ? pg_mx_find(w_up) : NULL;
    const int hclp = (wd && wu && wd->rows_p == pg_ceil128(hcl) && wu->K_p == pg_ceil128(hcl)) ? pg_ceil128(hcl) : hcl;
    g_pg_mx_pad_rows = hclp;
    QfDenseProj pd[1] = {{w_down, d, hcl}};
    const int rd = qfd_gemm_fp8_rows(pd, 1, normed, IN, T, xq, xs, s);
    g_pg_mx_pad_rows = 0;
    if (rd) return -1;
    k_pg_silu4_rows<<<(T * hclp + 255) / 256, 256, 0, s>>>(d, T * hclp);
    QfDenseProj pu[1] = {{w_up, up, IN}};
    if (qfd_gemm_fp8_rows(pu, 1, d, hclp, T, xq, xs, s)) return -1;
    k_pg_hc_mix_rows<<<dim3((NE + 255) / 256, T), 256, 0, s>>>(up, normed, mixed, NE, HC);
    if (w_inject && inj) k_pg_hc_inj_rows<<<dim3(HC, T), 256, 0, s>>>(normed, (const uint16_t *)w_inject, inj, IN, HC);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
// xq [T][K], xs [T][K/64] scratch. Returns 0, or -1 if the slabs are not FP8 / shape ineligible.
int qfd_gemm_fp8_rows(const QfDenseProj *projs, int nproj, const float *xT, int in, int T,
                      uint8_t *xq, uint8_t *xs, cudaStream_t s) {
    if (nproj < 1 || T < 1 || in <= 0 || (in & 63) || !qfd_fp8_active()) return -1;
    int done[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nleft = nproj;
    if (pg_mx_on() && nproj <= 8 && (in % 128) == 0) {
        int quant = 0;
        for (int p = 0; p < nproj; p++) {
            const PgMxW *w = pg_mx_find(projs[p].W);
            if (!w || w->K_p != in || (w->rows_p != w->rows && w->rows_p != g_pg_mx_pad_rows)) continue;
            if (!quant) { if (pg_mx_quant_act(xT, in, in, T, in, s)) { fprintf(stderr, "pf mx: activation quant failed (in %d T %d): %s\n", in, T, cudaGetErrorString(cudaGetLastError())); return -1; } quant = 1; }
            if (pg_mx_gemm(w, T, projs[p].y, s)) return -1;
            done[p] = 1; nleft--;
        }
    }
    if (!nleft) { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) fprintf(stderr, "pf mx: sticky CUDA error after GEMM (in %d T %d): %s\n", in, T, cudaGetErrorString(e)); return e == cudaSuccess ? 0 : -1; }
    k_pg_quant_rows<<<dim3((in / 64 + 127) / 128, T), 128, 0, s>>>(xT, in, xq, xs);
    for (int p = 0; p < nproj; p++) {
        if (done[p]) continue;
        const uint8_t *W = (const uint8_t *)projs[p].W;
        const int rows = projs[p].rows;
        const size_t sbytes = (size_t)rows * (in >> 6);
        const uint8_t *S = W - (sbytes + ((16 - (sbytes & 15u)) & 15u));
        static int gemm2 = -1;
        if (gemm2 < 0) { const char *e = getenv("QF_PF_GEMM2"); gemm2 = (e && e[0] == '0') ? 0 : 1;
                         if (gemm2) cudaFuncSetAttribute(k_pg_gemm2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(PG_ST * sizeof(PgStage))); }
        if (gemm2) {
            dim3 grid2((rows + 63) / 64, (T + PG_TH * 8 * PG_NG - 1) / (PG_TH * 8 * PG_NG));
            k_pg_gemm2<<<grid2, 256, PG_ST * sizeof(PgStage), s>>>(W, S, rows, in, xq, xs, T, projs[p].y);
        } else {
            dim3 grid((rows + 63) / 64, (T + 8 * PG_NG - 1) / (8 * PG_NG));
            k_pg_gemm<<<grid, 128, 0, s>>>(W, S, rows, in, xq, xs, T, projs[p].y);
        }
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) fprintf(stderr, "pf gemm: CUDA error (in %d T %d): %s\n", in, T, cudaGetErrorString(e)); return e == cudaSuccess ? 0 : -1; }
}
}
