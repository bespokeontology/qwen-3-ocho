// qf_prefill_moe.cu - Spark routed experts for a PREFILL CHUNK of T tokens:
// routing inversion (expert -> its tokens) + token-blocked NVFP4 kernels that
// read every expert weight row ONCE and dot it against all tokens routed to
// that expert. Same slot layout, same dequant as k_nvfp4_gemv_slot (qf.cu):
//   W row = K/2 bytes as 16-byte words (two 16-value sub-blocks each), scale
//   row = K/16 UE4M3 bytes, per-expert weight_scale_2 from the s2 tables,
//   gate/up slot stride 640*1280 B (scales 640*160), down 2560*320 B (scales 2560*40).
// The decode path amortizes nothing (one token); here the weight stream is
// shared by up to QF_PF_TX tokens per pass, so at chunk sizes of a few hundred
// tokens the expert bytes per token drop by the average tokens-per-expert.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define PF_NEMBD 2560
#define PF_NFF   640
#define PF_NEXP  512
#define PF_K     10
#ifndef QF_PF_TX
#define QF_PF_TX 16                 // tokens per weight pass (register accumulators)
#endif

static __device__ __forceinline__ float pf_e2m1(uint32_t nib) {
    uint32_t e = (nib >> 1) & 3u, m = nib & 1u;
    uint32_t mag = e ? (((e + 126u) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
    return __uint_as_float(mag | ((nib & 8u) << 28));
}
static __device__ __forceinline__ float pf_ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
    return (b == 0x7Fu) ? __int_as_float(0x7F800000) : v;
}
static __device__ __forceinline__ float pf_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(~0u, v, off);
    return v;
}

// ---- routing inversion, ONE block of 512 threads (thread = expert) -----------
// sel [T][K], wt [T][K] -> exp_slot[n] (ascending expert id), exp_ptr[n+1],
// pair_tok/pair_wt in (expert, token) order. Deterministic. counts[0] = n_exp,
// counts[1] = n_pairs. T*K pairs are scanned by every thread (T <= 1024).
__global__ void __launch_bounds__(512) k_pf_csr_build(const int *__restrict__ sel, const float *__restrict__ wt,
                                                      int T, int *__restrict__ exp_slot, int *__restrict__ exp_ptr,
                                                      int *__restrict__ pair_tok, float *__restrict__ pair_wt,
                                                      int *__restrict__ counts) {
    __shared__ int cnt[PF_NEXP], off[PF_NEXP], idx[PF_NEXP];
    __shared__ int red[512];
    const int e = threadIdx.x;
    int c = 0;
    for (int p = 0; p < T * PF_K; p++) c += (sel[p] == e);
    cnt[e] = c;
    __syncthreads();
    // exclusive scan over experts (Hillis-Steele in smem) + compact expert list
    red[e] = c;
    __syncthreads();
    for (int s = 1; s < 512; s <<= 1) {
        int v = (e >= s) ? red[e - s] : 0;
        __syncthreads();
        red[e] += v;
        __syncthreads();
    }
    const int incl = red[e];
    off[e] = incl - c;                        // exclusive offset in pair order
    __syncthreads();
    // dense index of this expert among the touched ones (ascending id)
    int nz = 0;
    for (int i = 0; i < e; i++) nz += (cnt[i] > 0);
    idx[e] = nz;
    __syncthreads();
    if (c > 0) {
        exp_slot[nz] = e;                     // slot == expert (fully resident)
        exp_ptr[nz] = off[e];
        int w = off[e];
        for (int p = 0; p < T * PF_K; p++)
            if (sel[p] == e) { pair_tok[w] = p / PF_K; pair_wt[w] = wt[p]; w++; }
    }
    if (e == 511) {
        int n = 0; for (int i = 0; i < PF_NEXP; i++) n += (cnt[i] > 0);
        exp_ptr[n] = incl;                    // total pairs
        counts[0] = n; counts[1] = incl;
    }
}

// ---- token-blocked NVFP4 dot: one warp per weight row, TX tokens per pass ----
// words u = lane, lane+32, ... of the row (K/32 words); token activations are
// read from global (L2) per word.
template<int K>
__device__ __forceinline__ void pf_row_dot(const uint8_t *__restrict__ wrow, const uint8_t *__restrict__ srow,
                                           const float *__restrict__ x, const int *__restrict__ toks, int nt,
                                           int lane, float *acc /*[QF_PF_TX]*/) {
    const int U4 = K / 32;
    const uint4 *wr4 = (const uint4 *)wrow;
    const uint16_t *sr2 = (const uint16_t *)srow;
    #pragma unroll
    for (int j = 0; j < QF_PF_TX; j++) acc[j] = 0.f;
    for (int u = lane; u < U4; u += 32) {
        const uint4 d = __ldg(wr4 + u);
        const uint32_t sp = (uint32_t)__ldg(sr2 + u);
        const float sc0 = pf_ue4m3(sp & 0xFFu), sc1 = pf_ue4m3(sp >> 8);
        const int base = (u >> 1) * 64 + (u & 1) * 32;
        float wv[32];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t q0 = (d.x >> (8 * j)) & 0xFFu, q1 = (d.y >> (8 * j)) & 0xFFu;
            uint32_t q2 = (d.z >> (8 * j)) & 0xFFu, q3 = (d.w >> (8 * j)) & 0xFFu;
            wv[2 * j] = pf_e2m1(q0 & 15u) * sc0;      wv[2 * j + 1] = pf_e2m1(q0 >> 4) * sc0;
            wv[8 + 2 * j] = pf_e2m1(q1 & 15u) * sc0;  wv[8 + 2 * j + 1] = pf_e2m1(q1 >> 4) * sc0;
            wv[16 + 2 * j] = pf_e2m1(q2 & 15u) * sc1; wv[16 + 2 * j + 1] = pf_e2m1(q2 >> 4) * sc1;
            wv[24 + 2 * j] = pf_e2m1(q3 & 15u) * sc1; wv[24 + 2 * j + 1] = pf_e2m1(q3 >> 4) * sc1;
        }
        #pragma unroll
        for (int j = 0; j < QF_PF_TX; j++) {
            if (j < nt) {
                const float4 *xv = (const float4 *)(x + (size_t)toks[j] * K + base);
                float a = 0.f;
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    const float4 v = __ldg(xv + q);
                    a = fmaf(wv[4 * q], v.x, a); a = fmaf(wv[4 * q + 1], v.y, a);
                    a = fmaf(wv[4 * q + 2], v.z, a); a = fmaf(wv[4 * q + 3], v.w, a);
                }
                acc[j] += a;
            }
        }
    }
}

// gate + up + silu*up for every (expert, token) pair: hidden[pair][640].
// grid (640/8, n_exp_max), block 256 = 8 rows (warp per row).
__global__ __launch_bounds__(256)
void k_pf_gateup_csr(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                     const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                     const float *__restrict__ s2g, const float *__restrict__ s2u,
                     const float *__restrict__ x, float *__restrict__ hidden,
                     const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
                     const int *__restrict__ pair_tok, const int *__restrict__ counts) {
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    __shared__ int stok[QF_PF_TX];
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    const uint8_t *wg = Wg + (size_t)slot * (640 * 1280) + (size_t)row * 1280;
    const uint8_t *sg = Sg + (size_t)slot * (640 * 160) + (size_t)row * 160;
    const uint8_t *wu = Wu + (size_t)slot * (640 * 1280) + (size_t)row * 1280;
    const uint8_t *su = Su + (size_t)slot * (640 * 160) + (size_t)row * 160;
    const float g2 = s2g[slot], u2 = s2u[slot];
    for (int j0 = 0; j0 < n; j0 += QF_PF_TX) {
        const int nt = (n - j0) < QF_PF_TX ? (n - j0) : QF_PF_TX;
        __syncthreads();
        if (threadIdx.x < nt) stok[threadIdx.x] = pair_tok[base + j0 + threadIdx.x];
        __syncthreads();
        float ag[QF_PF_TX], au[QF_PF_TX];
        pf_row_dot<PF_NEMBD>(wg, sg, x, stok, nt, lane, ag);
        pf_row_dot<PF_NEMBD>(wu, su, x, stok, nt, lane, au);
        #pragma unroll
        for (int j = 0; j < QF_PF_TX; j++) {
            if (j < nt) {
                const float g = pf_warp_sum(ag[j]) * g2, u = pf_warp_sum(au[j]) * u2;
                if (lane == 0) hidden[(size_t)(base + j0 + j) * PF_NFF + row] = g / (1.f + expf(-g)) * u;
            }
        }
    }
}

// down + router-weighted accumulate: y[tok][2560] += wt * (down_row . hidden[pair]).
// grid (2560/8, n_exp_max), block 256 = 8 rows. y is zeroed by the caller.
__global__ __launch_bounds__(256)
void k_pf_down_csr(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
                   const float *__restrict__ s2d, const float *__restrict__ hidden,
                   float *__restrict__ y, const int *__restrict__ exp_slot,
                   const int *__restrict__ exp_ptr, const int *__restrict__ pair_tok,
                   const float *__restrict__ pair_wt, const int *__restrict__ counts) {
    const int e = blockIdx.y;
    if (e >= counts[0]) return;
    __shared__ int spair[QF_PF_TX];
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    const uint8_t *wd = Wd + (size_t)slot * ((size_t)PF_NEMBD * PF_NFF / 2) + (size_t)row * (PF_NFF / 2);
    const uint8_t *sd = Sd + (size_t)slot * ((size_t)PF_NEMBD * 40) + (size_t)row * 40;
    const float d2 = s2d[slot];
    for (int j0 = 0; j0 < n; j0 += QF_PF_TX) {
        const int nt = (n - j0) < QF_PF_TX ? (n - j0) : QF_PF_TX;
        __syncthreads();
        if (threadIdx.x < nt) spair[threadIdx.x] = base + j0 + threadIdx.x;   // hidden row index = pair
        __syncthreads();
        float acc[QF_PF_TX];
        pf_row_dot<PF_NFF>(wd, sd, hidden, spair, nt, lane, acc);
        #pragma unroll
        for (int j = 0; j < QF_PF_TX; j++) {
            if (j < nt) {
                const float a = pf_warp_sum(acc[j]) * d2;
                if (lane == 0) {
                    const int pair = base + j0 + j;
                    atomicAdd(&y[(size_t)pair_tok[pair] * PF_NEMBD + row], pair_wt[pair] * a);
                }
            }
        }
    }
}

extern "C" {
// Routed experts of a T-token chunk: x [T][2560] fp32 (mixed), y [T][2560] fp32 (ZEROED by
// the caller), hidden [T*10][640] scratch, CSR scratch: exp_slot[T*10], exp_ptr[T*10+1],
// pair_tok[T*10], pair_wt[T*10], counts[2]. sel/wt: [T][10] from the router top-k.
// CSR build in three bulk launches (the single-block kernel above scanned all T*10 pairs from every
// thread: ~1.7 ms per layer at T=2048). Pair order inside an expert's segment is whatever the atomic
// scatter yields; the down kernel finds a token's partial slot by searching its own selection list,
// so every per-token result is independent of that order.
__global__ void k_pf_csr_count(const int *__restrict__ sel, int np, int *__restrict__ cnt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < np) atomicAdd(&cnt[sel[p]], 1);
}
__global__ void __launch_bounds__(512) k_pf_csr_scan(const int *__restrict__ cnt, int *__restrict__ exp_slot, int *__restrict__ exp_ptr,
                                                     int *__restrict__ counts, int *__restrict__ cursor) {
    __shared__ int red[512], nzp[512];
    const int e = threadIdx.x;
    const int c = cnt[e];
    red[e] = c; nzp[e] = c > 0;
    __syncthreads();
    for (int s = 1; s < 512; s <<= 1) {
        const int v = (e >= s) ? red[e - s] : 0, z = (e >= s) ? nzp[e - s] : 0;
        __syncthreads();
        red[e] += v; nzp[e] += z;
        __syncthreads();
    }
    const int off = red[e] - c, nz = nzp[e] - (c > 0);
    cursor[e] = off;
    if (c > 0) { exp_slot[nz] = e; exp_ptr[nz] = off; }
    if (e == 511) { exp_ptr[nzp[511]] = red[511]; counts[0] = nzp[511]; counts[1] = red[511]; }
}
__global__ void k_pf_csr_scatter(const int *__restrict__ sel, const float *__restrict__ wt, int np, int *__restrict__ cursor,
                                 int *__restrict__ pair_tok, float *__restrict__ pair_wt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;
    const int w = atomicAdd(&cursor[sel[p]], 1);
    pair_tok[w] = p / PF_K; pair_wt[w] = wt[p];
}
static int *g_csr_cnt = NULL, *g_csr_cur = NULL;
void qf_pf_csr_build_launch(const int *sel, const float *wt, int T, int *exp_slot, int *exp_ptr,
                            int *pair_tok, float *pair_wt, int *counts, cudaStream_t s) {
    if (!g_csr_cnt && (cudaMalloc(&g_csr_cnt, PF_NEXP * sizeof(int)) != cudaSuccess || cudaMalloc(&g_csr_cur, PF_NEXP * sizeof(int)) != cudaSuccess)) {
        k_pf_csr_build<<<1, 512, 0, s>>>(sel, wt, T, exp_slot, exp_ptr, pair_tok, pair_wt, counts); return; }
    const int np = T * PF_K;
    cudaMemsetAsync(g_csr_cnt, 0, PF_NEXP * sizeof(int), s);
    k_pf_csr_count<<<(np + 255) / 256, 256, 0, s>>>(sel, np, g_csr_cnt);
    k_pf_csr_scan<<<1, 512, 0, s>>>(g_csr_cnt, exp_slot, exp_ptr, counts, g_csr_cur);
    k_pf_csr_scatter<<<(np + 255) / 256, 256, 0, s>>>(sel, wt, np, g_csr_cur, pair_tok, pair_wt);
}
void qf_prefill_experts_csr(const void *Wg, const void *Sg, const void *Wu, const void *Su,
                            const void *Wd, const void *Sd,
                            const float *s2g, const float *s2u, const float *s2d,
                            const int *sel, const float *wt, int T,
                            const float *x, float *hidden, float *y,
                            int *exp_slot, int *exp_ptr, int *pair_tok, float *pair_wt, int *counts,
                            cudaStream_t s) {
    qf_pf_csr_build_launch(sel, wt, T, exp_slot, exp_ptr, pair_tok, pair_wt, counts, s);
    const int nmax = (T * PF_K) < PF_NEXP ? (T * PF_K) : PF_NEXP;   // distinct experts <= 512
    k_pf_gateup_csr<<<dim3(PF_NFF / 8, nmax), 256, 0, s>>>((const uint8_t *)Wg, (const uint8_t *)Sg,
        (const uint8_t *)Wu, (const uint8_t *)Su, s2g, s2u, x, hidden, exp_slot, exp_ptr, pair_tok, counts);
    k_pf_down_csr<<<dim3(PF_NEMBD / 8, nmax), 256, 0, s>>>((const uint8_t *)Wd, (const uint8_t *)Sd,
        s2d, hidden, y, exp_slot, exp_ptr, pair_tok, pair_wt, counts);
}
}
