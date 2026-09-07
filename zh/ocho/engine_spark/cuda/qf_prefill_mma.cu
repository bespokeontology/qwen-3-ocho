// qf_prefill_mma.cu - routed experts of a PREFILL CHUNK on the native
// block-scaled FP4 tensor core (sm_121a), grouped by expert.
//
// The decode GEMV in qf_fp4mma.cu proved the fragment map of
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
// against a scalar reference with the checkpoint's own adjacent-pair nibble
// packing (no repack). Its B operand has EIGHT columns and decode fills one.
// Here the eight columns are eight tokens routed to the same expert (the CSR
// inversion of qf_prefill_moe.cu), so one weight pass serves up to
// 8*PFM_G tokens and the weights are never software-dequantized:
//   1. quantize the T activations to E2M1 + UE4M3 block scales (once per layer)
//   2. gate+up: per (expert, 64-row strip) CTA, 4 warps x 16 rows, both
//      projections in one weight pass, h = silu(s2g*g) * (s2u*u) -> hidden[pair]
//   3. quantize the T*10 hidden rows
//   4. down: per (expert, 64-row strip) CTA, y[token] += wt * s2d * (Wd . h)
// Slot layout identical to the M=1 path: gate/up slot = 640 rows x K/2 bytes,
// scales 640 x K/16; down slot = 2560 rows x 320 bytes, scales 2560 x 40.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PFM_NEMBD 2560
#define PFM_NFF   640
#define PFM_NEXP  512
#define PFM_K     10
#ifndef PFM_G
#define PFM_G 8                 // token groups of 8 per weight pass (64 tokens; 16 measured worse: register pressure)
#endif

// ---- device helpers: same arithmetic as qf_fp4mma.cu (kept local so that the
// opt-in decode file stays untouched) ----------------------------------------
static __device__ __forceinline__ void pfm_mma(float &d0, float &d1, float &d2, float &d3,
                                               uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
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
    d0 = d1 = d2 = d3 = nanf("");          // a dead path must be visible (see qf_fp4mma.cu)
#endif
}
static __device__ __forceinline__ uint32_t pfm_e2m1_enc(float x) {
    float a = fabsf(x);
    uint32_t c = a < 0.25f ? 0u : a < 0.75f ? 1u : a < 1.25f ? 2u
               : a < 1.75f ? 3u : a < 2.5f  ? 4u : a < 3.5f  ? 5u
               : a < 5.f   ? 6u : 7u;
    return ((__float_as_uint(x) >> 28) & 8u) | c;
}
static __device__ __forceinline__ uint32_t pfm_ue4m3_enc_ceil(float v) {
    if (!(v > 0.f)) return 0u;
    uint32_t b = __float_as_uint(v);
    int e32 = (int)(b >> 23) - 127;
    if (e32 < -6) { uint32_t m = (uint32_t)ceilf(v * 512.f); if (!m) m = 1; return m > 7 ? 8u : m; }
    int e = e32 + 7;
    uint32_t m = ((b & 0x7FFFFFu) + 0xFFFFFu) >> 20;
    if (m == 8u) { e++; m = 0; }
    return e > 15 ? 0x7Eu : (uint32_t)((e << 3) | m);
}
static __device__ __forceinline__ float pfm_ue4m3_dec(uint32_t b) {
    uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
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

// ---- 1./3. row quantizer: x [rows][K] fp32 -> xq [rows][K/2] E2M1 pairs, xs [rows][K/16] UE4M3.
// grid (ceil(K/16/128), rows), block 128: one thread per 16-value block.
template<int K>
__global__ void k_pfm_quant_rows(const float *__restrict__ x, uint8_t *__restrict__ xq, uint8_t *__restrict__ xs) {
    const int r = blockIdx.y;
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= K / 16) return;
    const float *xr = x + (size_t)r * K + blk * 16;
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; j++) a = fmaxf(a, fabsf(xr[j]));
    const uint32_t sb = pfm_ue4m3_enc_ceil(a * (1.f / 6.f));
    const float inv = 1.f / fmaxf(pfm_ue4m3_dec(sb), 1e-30f);
    uint32_t lo = 0, hi = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        lo |= (pfm_e2m1_enc(xr[2 * j] * inv) | (pfm_e2m1_enc(xr[2 * j + 1] * inv) << 4)) << (8 * j);
        hi |= (pfm_e2m1_enc(xr[8 + 2 * j] * inv) | (pfm_e2m1_enc(xr[8 + 2 * j + 1] * inv) << 4)) << (8 * j);
    }
    *(uint2 *)(xq + (size_t)r * (K / 2) + blk * 8) = make_uint2(lo, hi);
    xs[(size_t)r * (K / 16) + blk] = (uint8_t)sb;
}

// ---- 2. gate + up for every (expert, token) pair -> hidden[pair][NFF].
// grid (NFF/64, n_exp_max), block 128 (warp = 16-row strip). Lane l = 4g+p:
// A fragments a0/a2 = row g, a1/a3 = row g+8 (K 8p.. and 32+8p..); SFA from
// row (p&1 ? g+8 : g); B fragments = column g = token j0+q*8+g of the group;
// D: d0=(g,2p) d1=(g,2p+1) d2=(g+8,2p) d3=(g+8,2p+1).
template<int K>   // K = NEMBD
__global__ __launch_bounds__(128, 4) void k_pfm_gateup(
    const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
    const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
    const float *__restrict__ s2g, const float *__restrict__ s2u,
    const uint8_t *__restrict__ xq, const uint8_t *__restrict__ xs,
    const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
    const int *__restrict__ pair_tok, const int *__restrict__ counts,
    float *__restrict__ hidden) {
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, p = lane & 3;
    const int row0 = blockIdx.x * 64 + warp * 16, rA = row0 + g, rB = row0 + g + 8;
    const uint8_t *wgA = Wg + (size_t)slot * (PFM_NFF * (K / 2)) + (size_t)rA * (K / 2), *wgB = wgA + 8 * (K / 2);
    const uint8_t *wuA = Wu + (size_t)slot * (PFM_NFF * (K / 2)) + (size_t)rA * (K / 2), *wuB = wuA + 8 * (K / 2);
    const uint8_t *sgX = Sg + (size_t)slot * (PFM_NFF * (K / 16)) + (size_t)((p & 1) ? rB : rA) * (K / 16);
    const uint8_t *suX = Su + (size_t)slot * (PFM_NFF * (K / 16)) + (size_t)((p & 1) ? rB : rA) * (K / 16);
    const float g2 = s2g[slot], u2 = s2u[slot];
    for (int j0 = 0; j0 < n; j0 += 8 * PFM_G) {
        const int ng0 = (n - j0 + 7) / 8, ng = ng0 < PFM_G ? ng0 : PFM_G;
        const uint8_t *bq[PFM_G], *bs[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            int j = j0 + q * 8 + g; if (j > n - 1) j = n - 1;           // padded columns read a valid token, output dropped
            const int tok = pair_tok[base + j];
            bq[q] = xq + (size_t)tok * (K / 2); bs[q] = xs + (size_t)tok * (K / 16);
        }
        float dg[PFM_G][4], du[PFM_G][4];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) { dg[q][0] = dg[q][1] = dg[q][2] = dg[q][3] = 0.f; du[q][0] = du[q][1] = du[q][2] = du[q][3] = 0.f; }
        // Software pipeline: tile k+64's weight/activation fragments and scales are loaded
        // before tile k's MMAs (the plain loop was latency-bound at 1024-token chunks).
        uint2 wA = qld8(wgA, p), wB = qld8(wgB, p), uA = qld8(wuA, p), uB = qld8(wuB, p);   // tile 0, 8 B per lane per row
        uint32_t sfg = __ldcs((const uint32_t *)sgX), sfu = __ldcs((const uint32_t *)suX);
        uint32_t b0[PFM_G], b1[PFM_G], sfb[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) { b0[q] = *(const uint32_t *)(bq[q] + 4 * p); b1[q] = *(const uint32_t *)(bq[q] + 16 + 4 * p); sfb[q] = *(const uint32_t *)(bs[q]); }
        for (int k0 = 0; k0 < K; k0 += 64) {
            const int kn = k0 + 64, kna = kn / 2;
            uint2 nA = make_uint2(0u, 0u), nB = nA, mA = nA, mB = nA; uint32_t nsg = 0, nsu = 0, nb0[PFM_G], nb1[PFM_G], nsb[PFM_G];
            if (kn < K) {
                nA = qld8(wgA + kna, p); nB = qld8(wgB + kna, p); mA = qld8(wuA + kna, p); mB = qld8(wuB + kna, p);
                nsg = __ldcs((const uint32_t *)(sgX + kn / 16)); nsu = __ldcs((const uint32_t *)(suX + kn / 16));
                #pragma unroll
                for (int q = 0; q < PFM_G; q++) { nb0[q] = *(const uint32_t *)(bq[q] + kna + 4 * p); nb1[q] = *(const uint32_t *)(bq[q] + kna + 16 + 4 * p); nsb[q] = *(const uint32_t *)(bs[q] + kn / 16); }
            } else {
                #pragma unroll
                for (int q = 0; q < PFM_G; q++) { nb0[q] = 0u; nb1[q] = 0u; nsb[q] = 0u; }
            }
            uint32_t a0, a1, a2, a3, c0, c1, c2, c3;
            qsplit(wA, lane, p, a0, a2); qsplit(wB, lane, p, a1, a3); qsplit(uA, lane, p, c0, c2); qsplit(uB, lane, p, c1, c3);
            #pragma unroll
            for (int q = 0; q < PFM_G; q++) {
                if (q < ng) {
                    pfm_mma(dg[q][0], dg[q][1], dg[q][2], dg[q][3], a0, a1, a2, a3, b0[q], b1[q], sfg, sfb[q]);
                    pfm_mma(du[q][0], du[q][1], du[q][2], du[q][3], c0, c1, c2, c3, b0[q], b1[q], sfu, sfb[q]);
                }
            }
            wA = nA; wB = nB; uA = mA; uB = mB; sfg = nsg; sfu = nsu;
            #pragma unroll
            for (int q = 0; q < PFM_G; q++) { b0[q] = nb0[q]; b1[q] = nb1[q]; sfb[q] = nsb[q]; }
        }
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            if (q < ng) {
                const int jA = j0 + q * 8 + 2 * p, jB = jA + 1;
                if (jA < n) {
                    float gv = dg[q][0] * g2, uv = du[q][0] * u2;
                    hidden[(size_t)(base + jA) * PFM_NFF + rA] = gv / (1.f + __expf(-gv)) * uv;
                    gv = dg[q][2] * g2; uv = du[q][2] * u2;
                    hidden[(size_t)(base + jA) * PFM_NFF + rB] = gv / (1.f + __expf(-gv)) * uv;
                }
                if (jB < n) {
                    float gv = dg[q][1] * g2, uv = du[q][1] * u2;
                    hidden[(size_t)(base + jB) * PFM_NFF + rA] = gv / (1.f + __expf(-gv)) * uv;
                    gv = dg[q][3] * g2; uv = du[q][3] * u2;
                    hidden[(size_t)(base + jB) * PFM_NFF + rB] = gv / (1.f + __expf(-gv)) * uv;
                }
            }
        }
    }
}

// ---- 4. down: part[token][k][NEMBD] = wt * s2d * (Wd . h[pair]) for the pair's slot k
// in its token's top-10 (plain stores, every (token,k) written exactly once), then
// k_pfm_combine sums the 10 partials per token. No atomics: 26 MB of partials
// written once and read once instead of 6.5M contended read-modify-writes.
// grid (NEMBD/64, n_exp_max), block 128. B columns = the expert's pairs.
template<int K>   // K = NFF
__global__ __launch_bounds__(128, 4) void k_pfm_down(
    const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd, const float *__restrict__ s2d,
    const uint8_t *__restrict__ hq, const uint8_t *__restrict__ hs,
    const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
    const int *__restrict__ pair_tok, const float *__restrict__ pair_wt, const int *__restrict__ counts,
    const int *__restrict__ sel, float *__restrict__ part) {
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, p = lane & 3;
    const int row0 = blockIdx.x * 64 + warp * 16, rA = row0 + g, rB = row0 + g + 8;
    const uint8_t *wdA = Wd + (size_t)slot * (PFM_NEMBD * (K / 2)) + (size_t)rA * (K / 2), *wdB = wdA + 8 * (K / 2);
    const uint8_t *sdX = Sd + (size_t)slot * (PFM_NEMBD * (K / 16)) + (size_t)((p & 1) ? rB : rA) * (K / 16);
    const float d2 = s2d[slot];
    for (int j0 = 0; j0 < n; j0 += 8 * PFM_G) {
        const int ng0 = (n - j0 + 7) / 8, ng = ng0 < PFM_G ? ng0 : PFM_G;
        const uint8_t *bq[PFM_G], *bs[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            int j = j0 + q * 8 + g; if (j > n - 1) j = n - 1;
            bq[q] = hq + (size_t)(base + j) * (K / 2); bs[q] = hs + (size_t)(base + j) * (K / 16);
        }
        float dd[PFM_G][4];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) dd[q][0] = dd[q][1] = dd[q][2] = dd[q][3] = 0.f;
        uint2 wA = qld8(wdA, p), wB = qld8(wdB, p);
        uint32_t sfa = __ldcs((const uint32_t *)sdX);
        uint32_t b0[PFM_G], b1[PFM_G], sfb[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) { b0[q] = *(const uint32_t *)(bq[q] + 4 * p); b1[q] = *(const uint32_t *)(bq[q] + 16 + 4 * p); sfb[q] = *(const uint32_t *)(bs[q]); }
        for (int k0 = 0; k0 < K; k0 += 64) {
            const int kn = k0 + 64, kna = kn / 2;
            uint2 nA = make_uint2(0u, 0u), nB = nA; uint32_t nsa = 0, nb0[PFM_G], nb1[PFM_G], nsb[PFM_G];
            if (kn < K) {
                nA = qld8(wdA + kna, p); nB = qld8(wdB + kna, p);
                nsa = __ldcs((const uint32_t *)(sdX + kn / 16));
                #pragma unroll
                for (int q = 0; q < PFM_G; q++) { nb0[q] = *(const uint32_t *)(bq[q] + kna + 4 * p); nb1[q] = *(const uint32_t *)(bq[q] + kna + 16 + 4 * p); nsb[q] = *(const uint32_t *)(bs[q] + kn / 16); }
            } else {
                #pragma unroll
                for (int q = 0; q < PFM_G; q++) { nb0[q] = 0u; nb1[q] = 0u; nsb[q] = 0u; }
            }
            uint32_t a0, a1, a2, a3;
            qsplit(wA, lane, p, a0, a2); qsplit(wB, lane, p, a1, a3);
            #pragma unroll
            for (int q = 0; q < PFM_G; q++)
                if (q < ng) pfm_mma(dd[q][0], dd[q][1], dd[q][2], dd[q][3], a0, a1, a2, a3, b0[q], b1[q], sfa, sfb[q]);
            wA = nA; wB = nB; sfa = nsa;
            #pragma unroll
            for (int q = 0; q < PFM_G; q++) { b0[q] = nb0[q]; b1[q] = nb1[q]; sfb[q] = nsb[q]; }
        }
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            if (q < ng) {
                const int jA = j0 + q * 8 + 2 * p, jB = jA + 1;
                if (jA < n) {
                    const int tok = pair_tok[base + jA];
                    int k = 0; for (int kk = 0; kk < PFM_K; kk++) if (sel[tok * PFM_K + kk] == slot) { k = kk; break; }
                    const float w = pair_wt[base + jA] * d2; float *pt = part + ((size_t)tok * PFM_K + k) * PFM_NEMBD;
                    pt[rA] = w * dd[q][0]; pt[rB] = w * dd[q][2];
                }
                if (jB < n) {
                    const int tok = pair_tok[base + jB];
                    int k = 0; for (int kk = 0; kk < PFM_K; kk++) if (sel[tok * PFM_K + kk] == slot) { k = kk; break; }
                    const float w = pair_wt[base + jB] * d2; float *pt = part + ((size_t)tok * PFM_K + k) * PFM_NEMBD;
                    pt[rA] = w * dd[q][1]; pt[rB] = w * dd[q][3];
                }
            }
        }
    }
}

// ============================================================================
// Pipelined forms (cp.async, PFM_ST stages of PFM_KT K-tiles): the plain
// load->mma loop was memory-latency-bound (0.4 G mma/s against a measured
// 30 G mma/s tensor rate); here every CTA keeps two stages of its 64-row
// weight strip in flight in shared memory while it computes the third.
// Row stride 80 B for the 64 B of two K64 tiles keeps the fragment reads
// bank-conflict-free (bank = (row*20 + tile*8 + p) mod 32 is distinct over
// the quad's 8 rows). Scales (4 B per row per tile) ride along.
// ============================================================================
#define PFM_ST 3
#define PFM_KT 2
#define PFM_RS 80                                    // smem row stride (bytes) for PFM_KT tiles (KT=4/RS=144 measured worse: 1 CTA per SM)
static __device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}
static __device__ __forceinline__ void cp_async8(void *smem, const void *gmem) {
    const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(gmem));
}
static __device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> static __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// one stage of one matrix: 64 rows x (PFM_KT*32) B of nibbles + 64 rows x (PFM_KT*4) B of scales
struct PfmStage { uint8_t w[64 * PFM_RS]; uint8_t s[64 * 16]; };   // scale row stride 16 B (8 used)
static __device__ __forceinline__ void pfm_issue(PfmStage *st, const uint8_t *W, const uint8_t *S, int K, int kt0, int tid, int rows_valid) {
    // W rows: 64 x (PFM_KT*32 B) at column kt0*32; 4 x 16 B per row -> 256 chunks over 128 threads
    for (int c = tid; c < 64 * (PFM_KT * 2); c += 128) {
        const int r = c / (PFM_KT * 2), ch = c % (PFM_KT * 2);
        const int rr = r < rows_valid ? r : rows_valid - 1;
        cp_async16(st->w + r * PFM_RS + ch * 16, W + (size_t)rr * (K / 2) + kt0 * 32 + ch * 16);
    }
    if (tid < 64) {                                   // scales: PFM_KT*4 = 16 B per row
        const int rr = tid < rows_valid ? tid : rows_valid - 1;
        cp_async8(st->s + tid * 16, S + (size_t)rr * (K / 16) + kt0 * 4);
    }
}

// gate + up over the expert's pairs: same contract as k_pfm_gateup.
template<int K>
__global__ __launch_bounds__(128) void k_pfm_gateup2(
    const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
    const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
    const float *__restrict__ s2g, const float *__restrict__ s2u,
    const uint8_t *__restrict__ xq, const uint8_t *__restrict__ xs,
    const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
    const int *__restrict__ pair_tok, const int *__restrict__ counts,
    float *__restrict__ hidden) {
    extern __shared__ __align__(16) uint8_t pfm_smem[];
    PfmStage *sg_ = (PfmStage *)pfm_smem;                 // [PFM_ST] gate stages then [PFM_ST] up stages
    PfmStage *su_ = sg_ + PFM_ST;
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, p = lane & 3;
    const int row0 = blockIdx.x * 64;
    const int rA = warp * 16 + g, rB = rA + 8;             // rows within the strip
    const uint8_t *Wg_e = Wg + (size_t)slot * (PFM_NFF * (K / 2)) + (size_t)row0 * (K / 2);
    const uint8_t *Wu_e = Wu + (size_t)slot * (PFM_NFF * (K / 2)) + (size_t)row0 * (K / 2);
    const uint8_t *Sg_e = Sg + (size_t)slot * (PFM_NFF * (K / 16)) + (size_t)row0 * (K / 16);
    const uint8_t *Su_e = Su + (size_t)slot * (PFM_NFF * (K / 16)) + (size_t)row0 * (K / 16);
    const float g2 = s2g[slot], u2 = s2u[slot];
    constexpr int NT = K / 64, NSTG = (NT + PFM_KT - 1) / PFM_KT;
    for (int j0 = 0; j0 < n; j0 += 8 * PFM_G) {
        const int ng0 = (n - j0 + 7) / 8, ng = ng0 < PFM_G ? ng0 : PFM_G;
        const uint8_t *bq[PFM_G], *bs[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            int j = j0 + q * 8 + g; if (j > n - 1) j = n - 1;
            const int tok = pair_tok[base + j];
            bq[q] = xq + (size_t)tok * (K / 2); bs[q] = xs + (size_t)tok * (K / 16);
        }
        float dg[PFM_G][4], du[PFM_G][4];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) { dg[q][0] = dg[q][1] = dg[q][2] = dg[q][3] = 0.f; du[q][0] = du[q][1] = du[q][2] = du[q][3] = 0.f; }
        // prologue: stages 0..ST-2 in flight
        #pragma unroll
        for (int st = 0; st < PFM_ST - 1; st++) {
            if (st < NSTG) { pfm_issue(&sg_[st], Wg_e, Sg_e, K, st * PFM_KT, tid, 64); pfm_issue(&su_[st], Wu_e, Su_e, K, st * PFM_KT, tid, 64); }
            cp_commit();
        }
        for (int stg = 0; stg < NSTG; stg++) {
            cp_wait<PFM_ST - 2>();
            __syncthreads();
            const int nxt = stg + PFM_ST - 1;
            if (nxt < NSTG) { pfm_issue(&sg_[nxt % PFM_ST], Wg_e, Sg_e, K, nxt * PFM_KT, tid, 64); pfm_issue(&su_[nxt % PFM_ST], Wu_e, Su_e, K, nxt * PFM_KT, tid, 64); }
            cp_commit();
            const PfmStage *cg = &sg_[stg % PFM_ST], *cu = &su_[stg % PFM_ST];
            #pragma unroll
            for (int j = 0; j < PFM_KT; j++) {
                const int kt = stg * PFM_KT + j;
                if (kt < NT) {
                    const int ka = kt * 32;
                    uint32_t b0[PFM_G], b1[PFM_G], sfb[PFM_G];
                    #pragma unroll
                    for (int q = 0; q < PFM_G; q++) if (q < ng) { b0[q] = *(const uint32_t *)(bq[q] + ka + 4 * p); b1[q] = *(const uint32_t *)(bq[q] + ka + 16 + 4 * p); sfb[q] = *(const uint32_t *)(bs[q] + kt * 4); }
                    const uint32_t a0 = *(const uint32_t *)(cg->w + rA * PFM_RS + j * 32 + 4 * p), a1 = *(const uint32_t *)(cg->w + rB * PFM_RS + j * 32 + 4 * p);
                    const uint32_t a2 = *(const uint32_t *)(cg->w + rA * PFM_RS + j * 32 + 16 + 4 * p), a3 = *(const uint32_t *)(cg->w + rB * PFM_RS + j * 32 + 16 + 4 * p);
                    const uint32_t c0 = *(const uint32_t *)(cu->w + rA * PFM_RS + j * 32 + 4 * p), c1 = *(const uint32_t *)(cu->w + rB * PFM_RS + j * 32 + 4 * p);
                    const uint32_t c2 = *(const uint32_t *)(cu->w + rA * PFM_RS + j * 32 + 16 + 4 * p), c3 = *(const uint32_t *)(cu->w + rB * PFM_RS + j * 32 + 16 + 4 * p);
                    const int srow = (p & 1) ? rB : rA;
                    const uint32_t sfg = *(const uint32_t *)(cg->s + srow * 16 + j * 4), sfu = *(const uint32_t *)(cu->s + srow * 16 + j * 4);
                    #pragma unroll
                    for (int q = 0; q < PFM_G; q++) {
                        if (q < ng) {
                            pfm_mma(dg[q][0], dg[q][1], dg[q][2], dg[q][3], a0, a1, a2, a3, b0[q], b1[q], sfg, sfb[q]);
                            pfm_mma(du[q][0], du[q][1], du[q][2], du[q][3], c0, c1, c2, c3, b0[q], b1[q], sfu, sfb[q]);
                        }
                    }
                }
            }
            __syncthreads();                             // the stage buffer is refilled next iteration
        }
        cp_wait<0>();
        const int rAo = row0 + rA, rBo = row0 + rB;
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            if (q < ng) {
                const int jA = j0 + q * 8 + 2 * p, jB = jA + 1;
                if (jA < n) {
                    float gv = dg[q][0] * g2, uv = du[q][0] * u2;
                    hidden[(size_t)(base + jA) * PFM_NFF + rAo] = gv / (1.f + __expf(-gv)) * uv;
                    gv = dg[q][2] * g2; uv = du[q][2] * u2;
                    hidden[(size_t)(base + jA) * PFM_NFF + rBo] = gv / (1.f + __expf(-gv)) * uv;
                }
                if (jB < n) {
                    float gv = dg[q][1] * g2, uv = du[q][1] * u2;
                    hidden[(size_t)(base + jB) * PFM_NFF + rAo] = gv / (1.f + __expf(-gv)) * uv;
                    gv = dg[q][3] * g2; uv = du[q][3] * u2;
                    hidden[(size_t)(base + jB) * PFM_NFF + rBo] = gv / (1.f + __expf(-gv)) * uv;
                }
            }
        }
    }
}

// down over the expert's pairs into the owned partials: same contract as k_pfm_down.
template<int K>
__global__ __launch_bounds__(128) void k_pfm_down2(
    const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd, const float *__restrict__ s2d,
    const uint8_t *__restrict__ hq, const uint8_t *__restrict__ hs,
    const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
    const int *__restrict__ pair_tok, const float *__restrict__ pair_wt, const int *__restrict__ counts,
    const int *__restrict__ sel, float *__restrict__ part) {
    extern __shared__ __align__(16) uint8_t pfm_smem[];
    PfmStage *sd_ = (PfmStage *)pfm_smem;
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, p = lane & 3;
    const int row0 = blockIdx.x * 64;
    const int rA = warp * 16 + g, rB = rA + 8;
    const uint8_t *Wd_e = Wd + (size_t)slot * (PFM_NEMBD * (K / 2)) + (size_t)row0 * (K / 2);
    const uint8_t *Sd_e = Sd + (size_t)slot * (PFM_NEMBD * (K / 16)) + (size_t)row0 * (K / 16);
    const float d2 = s2d[slot];
    constexpr int NT = K / 64, NSTG = (NT + PFM_KT - 1) / PFM_KT;
    for (int j0 = 0; j0 < n; j0 += 8 * PFM_G) {
        const int ng0 = (n - j0 + 7) / 8, ng = ng0 < PFM_G ? ng0 : PFM_G;
        const uint8_t *bq[PFM_G], *bs[PFM_G];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            int j = j0 + q * 8 + g; if (j > n - 1) j = n - 1;
            bq[q] = hq + (size_t)(base + j) * (K / 2); bs[q] = hs + (size_t)(base + j) * (K / 16);
        }
        float dd[PFM_G][4];
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) dd[q][0] = dd[q][1] = dd[q][2] = dd[q][3] = 0.f;
        #pragma unroll
        for (int st = 0; st < PFM_ST - 1; st++) { if (st < NSTG) pfm_issue(&sd_[st], Wd_e, Sd_e, K, st * PFM_KT, tid, 64); cp_commit(); }
        for (int stg = 0; stg < NSTG; stg++) {
            cp_wait<PFM_ST - 2>();
            __syncthreads();
            const int nxt = stg + PFM_ST - 1;
            if (nxt < NSTG) pfm_issue(&sd_[nxt % PFM_ST], Wd_e, Sd_e, K, nxt * PFM_KT, tid, 64);
            cp_commit();
            const PfmStage *cd = &sd_[stg % PFM_ST];
            #pragma unroll
            for (int j = 0; j < PFM_KT; j++) {
                const int kt = stg * PFM_KT + j;
                if (kt < NT) {
                    const int ka = kt * 32;
                    uint32_t b0[PFM_G], b1[PFM_G], sfb[PFM_G];
                    #pragma unroll
                    for (int q = 0; q < PFM_G; q++) if (q < ng) { b0[q] = *(const uint32_t *)(bq[q] + ka + 4 * p); b1[q] = *(const uint32_t *)(bq[q] + ka + 16 + 4 * p); sfb[q] = *(const uint32_t *)(bs[q] + kt * 4); }
                    const uint32_t a0 = *(const uint32_t *)(cd->w + rA * PFM_RS + j * 32 + 4 * p), a1 = *(const uint32_t *)(cd->w + rB * PFM_RS + j * 32 + 4 * p);
                    const uint32_t a2 = *(const uint32_t *)(cd->w + rA * PFM_RS + j * 32 + 16 + 4 * p), a3 = *(const uint32_t *)(cd->w + rB * PFM_RS + j * 32 + 16 + 4 * p);
                    const uint32_t sfa = *(const uint32_t *)(cd->s + ((p & 1) ? rB : rA) * 16 + j * 4);
                    #pragma unroll
                    for (int q = 0; q < PFM_G; q++) {
                        if (q < ng) {
                            pfm_mma(dd[q][0], dd[q][1], dd[q][2], dd[q][3], a0, a1, a2, a3, b0[q], b1[q], sfa, sfb[q]);
                        }
                    }
                }
            }
            __syncthreads();
        }
        cp_wait<0>();
        const int rAo = row0 + rA, rBo = row0 + rB;
        #pragma unroll
        for (int q = 0; q < PFM_G; q++) {
            if (q < ng) {
                const int jA = j0 + q * 8 + 2 * p, jB = jA + 1;
                if (jA < n) {
                    const int tok = pair_tok[base + jA];
                    int k = 0; for (int kk = 0; kk < PFM_K; kk++) if (sel[tok * PFM_K + kk] == slot) { k = kk; break; }
                    const float w = pair_wt[base + jA] * d2; float *pt = part + ((size_t)tok * PFM_K + k) * PFM_NEMBD;
                    pt[rAo] = w * dd[q][0]; pt[rBo] = w * dd[q][2];
                }
                if (jB < n) {
                    const int tok = pair_tok[base + jB];
                    int k = 0; for (int kk = 0; kk < PFM_K; kk++) if (sel[tok * PFM_K + kk] == slot) { k = kk; break; }
                    const float w = pair_wt[base + jB] * d2; float *pt = part + ((size_t)tok * PFM_K + k) * PFM_NEMBD;
                    pt[rAo] = w * dd[q][1]; pt[rBo] = w * dd[q][3];
                }
            }
        }
    }
}

// ---- 5. combine: y[token][NEMBD] = sum_k part[token][k][NEMBD]. grid T, block 256.
__global__ void k_pfm_combine(const float *__restrict__ part, float *__restrict__ y) {
    const int t = blockIdx.x;
    const float *pt = part + (size_t)t * PFM_K * PFM_NEMBD;
    for (int i = threadIdx.x; i < PFM_NEMBD; i += blockDim.x) {
        float a = 0.f;
        #pragma unroll
        for (int k = 0; k < PFM_K; k++) a += pt[(size_t)k * PFM_NEMBD + i];
        y[(size_t)t * PFM_NEMBD + i] = a;
    }
}

extern "C" {
void qf_pf_csr_build_launch(const int *sel, const float *wt, int T, int *exp_slot, int *exp_ptr,
                            int *pair_tok, float *pair_wt, int *counts, cudaStream_t s);
// Same contract as qf_prefill_experts_csr (y ZEROED by the caller) plus the
// quantized-activation scratch: xq [T][1280], xs [T][160], hq [T*10][320], hs [T*10][40],
// and the owned down partials part [T*10][2560] fp32.
void qf_prefill_experts_mma(const void *Wg, const void *Sg, const void *Wu, const void *Su,
                            const void *Wd, const void *Sd,
                            const float *s2g, const float *s2u, const float *s2d,
                            const int *sel, const float *wt, int T,
                            const float *x, float *hidden, float *y,
                            int *exp_slot, int *exp_ptr, int *pair_tok, float *pair_wt, int *counts,
                            uint8_t *xq, uint8_t *xs, uint8_t *hq, uint8_t *hs, float *part, cudaStream_t s) {
    static int prof = -1; static cudaEvent_t ev[7]; static double acc[6]; static int calls = 0;
    if (prof < 0) { prof = (getenv("QF_PF_TIMING") && getenv("QF_PF_TIMING")[0] == '1') ? 1 : 0; if (prof) for (int i = 0; i < 7; i++) cudaEventCreate(&ev[i]); }
    if (prof) cudaEventRecord(ev[0], s);
    qf_pf_csr_build_launch(sel, wt, T, exp_slot, exp_ptr, pair_tok, pair_wt, counts, s);
    if (prof) cudaEventRecord(ev[1], s);
    const int nmax = (T * PFM_K) < PFM_NEXP ? (T * PFM_K) : PFM_NEXP;
    k_pfm_quant_rows<PFM_NEMBD><<<dim3((PFM_NEMBD / 16 + 127) / 128, T), 128, 0, s>>>(x, xq, xs);
    if (prof) cudaEventRecord(ev[2], s);
    static int mma2 = -1;
    if (mma2 < 0) {
        const char *e = getenv("QF_PF_MMA2"); mma2 = (e && e[0] == '0') ? 0 : 1;
        if (mma2) {
            cudaFuncSetAttribute(k_pfm_gateup2<PFM_NEMBD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2 * PFM_ST * sizeof(PfmStage)));
            cudaFuncSetAttribute(k_pfm_down2<PFM_NFF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(PFM_ST * sizeof(PfmStage)));
        }
    }
    if (mma2)
        k_pfm_gateup2<PFM_NEMBD><<<dim3(PFM_NFF / 64, nmax), 128, 2 * PFM_ST * sizeof(PfmStage), s>>>((const uint8_t *)Wg, (const uint8_t *)Sg,
            (const uint8_t *)Wu, (const uint8_t *)Su, s2g, s2u, xq, xs, exp_slot, exp_ptr, pair_tok, counts, hidden);
    else
    k_pfm_gateup<PFM_NEMBD><<<dim3(PFM_NFF / 64, nmax), 128, 0, s>>>((const uint8_t *)Wg, (const uint8_t *)Sg,
        (const uint8_t *)Wu, (const uint8_t *)Su, s2g, s2u, xq, xs, exp_slot, exp_ptr, pair_tok, counts, hidden);
    if (prof) cudaEventRecord(ev[3], s);
    k_pfm_quant_rows<PFM_NFF><<<dim3((PFM_NFF / 16 + 127) / 128, T * PFM_K), 128, 0, s>>>(hidden, hq, hs);
    if (prof) cudaEventRecord(ev[4], s);
    if (mma2)
        k_pfm_down2<PFM_NFF><<<dim3(PFM_NEMBD / 64, nmax), 128, PFM_ST * sizeof(PfmStage), s>>>((const uint8_t *)Wd, (const uint8_t *)Sd, s2d,
            hq, hs, exp_slot, exp_ptr, pair_tok, pair_wt, counts, sel, part);
    else
    k_pfm_down<PFM_NFF><<<dim3(PFM_NEMBD / 64, nmax), 128, 0, s>>>((const uint8_t *)Wd, (const uint8_t *)Sd, s2d,
        hq, hs, exp_slot, exp_ptr, pair_tok, pair_wt, counts, sel, part);
    if (prof) cudaEventRecord(ev[5], s);
    k_pfm_combine<<<T, 256, 0, s>>>(part, y);
    if (prof) {
        cudaEventRecord(ev[6], s); cudaEventSynchronize(ev[6]);
        for (int i = 0; i < 6; i++) { float ms = 0.f; cudaEventElapsedTime(&ms, ev[i], ev[i + 1]); acc[i] += ms; }
        if (++calls % 32 == 0) {
            fprintf(stderr, "prefill experts (32 layers, T=%d): csr %.1f xquant %.1f gateup %.1f hquant %.1f down %.1f combine %.1f ms\n",
                    T, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]);
            for (int i = 0; i < 6; i++) acc[i] = 0.0;
        }
    }
}
}
