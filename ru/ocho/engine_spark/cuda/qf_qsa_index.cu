// qf_qsa_index.cu - the QSA indexer on Spark: the model's sparse top-2048-token
// attention past the 2048-token budget (previously DENSE on this box; the
// output of any prompt past ~2K positions was garbage).
// Semantics (SEMANTICS.md, same as the AMD reference qf8_qsa.hip): the index
// projection gives per token [4 heads x 128 q | 128 token_k]; q gets a
// zero-centered RMS norm per head and full-128 rope (pairs i, i+64) at its
// position; token_k of every 4-token block is pooled (fp32 mean), k-layernormed
// and roped at the block start into pool_key[block]; a query at position pos
// scores the nfull = pos>>2 complete blocks before its own block with
// sum_h relu(q_h . pool_key[b]) / sqrt(128), selects the top-512 by a 48-step
// threshold bisection (deterministic, tie fill by ascending block), and the
// attention kernels attend the selected blocks plus the current block. With
// nfull <= 512 the mask is all ones (dense = exact). One code path serves the
// decode step (T = 1, positions read from device params so the CUDA graph
// stays valid) and the prefill chunk (T rows).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "qf_attn_fused.h"
#define IDX_H 4
#define IDX_D 128
#define IDX_C 4
#define IDX_TOPK 512
#define IDX_W 640

static __device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

// q norm + rope in place, one block (128 threads) per row.
__global__ void __launch_bounds__(128) k_idx_q_rows(float *__restrict__ idx640, const uint16_t *__restrict__ qnorm,
                                                    const float *__restrict__ invf, const QfDecodeParams *__restrict__ params) {
    const int t = blockIdx.x, tid = threadIdx.x, h = tid >> 5, sub = tid & 31;
    float *x = idx640 + (size_t)t * IDX_W;
    const long pos = (long)params[t].pos;
    __shared__ float qi[IDX_H * IDX_D];
    __shared__ float sq[IDX_H][32];
    const float v0 = x[h * 128 + sub], v1 = x[h * 128 + sub + 32], v2 = x[h * 128 + sub + 64], v3 = x[h * 128 + sub + 96];
    sq[h][sub] = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    __syncthreads();
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) acc += sq[h][i];
    const float rs = rsqrtf(acc / IDX_D + 1e-6f);
    qi[h * 128 + sub]      = v0 * rs * (1.f + bf(qnorm[sub]));
    qi[h * 128 + sub + 32] = v1 * rs * (1.f + bf(qnorm[sub + 32]));
    qi[h * 128 + sub + 64] = v2 * rs * (1.f + bf(qnorm[sub + 64]));
    qi[h * 128 + sub + 96] = v3 * rs * (1.f + bf(qnorm[sub + 96]));
    __syncthreads();
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int tt = tid + it * 128, hh = tt >> 6, i = tt & 63;
        const float a = qi[hh * 128 + i], b = qi[hh * 128 + i + 64];
        const float f = (float)pos * invf[i];
        qi[hh * 128 + i]      = a * cosf(f) - b * sinf(f);
        qi[hh * 128 + i + 64] = b * cosf(f) + a * sinf(f);
    }
    __syncthreads();
    x[tid] = qi[tid]; x[256 + tid] = qi[256 + tid];
}

// Pool token_k of the positions [pos0, pos0+T) into their 4-token blocks (in
// order), finalize blocks that reach 4 tokens. One block (128 threads) per
// touched 4-token block.
__global__ void __launch_bounds__(128) k_idx_pool_rows(const float *__restrict__ idx640, int T, const QfDecodeParams *__restrict__ params,
                                                       float *__restrict__ pool_sum, float *__restrict__ pool_key, int *__restrict__ pool_cnt,
                                                       const uint16_t *__restrict__ knorm, const float *__restrict__ invf) {
    const int tid = threadIdx.x;
    const long pos0 = (long)params[0].pos;
    const long b = (pos0 >> 2) + blockIdx.x;
    const long p_lo = pos0 > b * 4 ? pos0 : b * 4, p_hi = (pos0 + T) < (b * 4 + 4) ? (pos0 + T) : (b * 4 + 4);
    if (p_lo >= p_hi) return;
    float s = pool_sum[b * IDX_D + tid];
    for (long p = p_lo; p < p_hi; p++) s += idx640[(size_t)(p - pos0) * IDX_W + 512 + tid];
    pool_sum[b * IDX_D + tid] = s;
    __shared__ int cnt; __shared__ float red[4];
    if (tid == 0) { cnt = pool_cnt[b] + (int)(p_hi - p_lo); pool_cnt[b] = cnt; }
    __syncthreads();
    if (cnt != IDX_C) return;
    const float mv = s * (1.f / IDX_C);
    float a2 = mv * mv;
    #pragma unroll
    for (int off = 16; off; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
    if ((tid & 31) == 0) red[tid >> 5] = a2;
    __syncthreads();
    const float rsk = rsqrtf((red[0] + red[1] + red[2] + red[3]) / IDX_D + 1e-6f);
    __shared__ float pk[IDX_D];
    pk[tid] = mv * rsk * (1.f + bf(knorm[tid]));
    __syncthreads();
    if (tid < 64) {
        const float a = pk[tid], bb = pk[tid + 64];
        const float f = (float)(b * IDX_C) * invf[tid];
        pool_key[b * IDX_D + tid]      = a * cosf(f) - bb * sinf(f);
        pool_key[b * IDX_D + tid + 64] = bb * cosf(f) + a * sinf(f);
    }
}

// Per row: scores over the nfull complete blocks, top-512 by threshold
// bisection, bitmask (1 bit per block; all ones when nfull <= 512).
// The attention kernels iterate a COMPACT per-row list of the selected complete
// blocks (<= 512) plus the current block, so their cost is bounded by the 2048-token
// budget at any context (scanning every position and testing the mask cost O(ctx)
// per query: 83 -> 560 ms per 1024-token chunk from 2K to 7K context).
__device__ __forceinline__ void idx_compact_list(const uint32_t *__restrict__ mrow, int nfull, int *__restrict__ list, int *__restrict__ nlist) {
    // warp 0 only, deterministic: ascending block order via ballot + rank
    const int lane = threadIdx.x & 31;
    int count = 0;
    for (int base = 0; base < nfull; base += 32) {
        const int b = base + lane;
        const int sel = (b < nfull) ? (int)((mrow[b >> 5] >> (b & 31)) & 1u) : 0;
        const uint32_t bal = __ballot_sync(0xffffffffu, sel);
        if (sel) list[count + __popc(bal & ((1u << lane) - 1u))] = b;
        count += __popc(bal);
    }
    if (lane == 0) *nlist = count;
}
__global__ void __launch_bounds__(256) k_idx_select_rows(const float *__restrict__ idx640, const float *__restrict__ pool_key,
                                                         const QfDecodeParams *__restrict__ params, float *__restrict__ blk_score, int nb_max,
                                                         uint32_t *__restrict__ mask, int mw, int *__restrict__ list, int *__restrict__ nlist,
                                                         int pre) {   // pre = scores already in blk_score (GEMM path): only the row max is needed
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const long pos = (long)params[t].pos;
    int nfull = (int)(pos >> 2);
    if (nfull > nb_max) nfull = nb_max;
    uint32_t *mrow = mask + (size_t)t * mw;
    int *lrow = list + (size_t)t * (IDX_TOPK + 1);
    if (nfull <= IDX_TOPK) {
        for (int w = tid; w < mw; w += 256) mrow[w] = 0xFFFFFFFFu;
        for (int b = tid; b < nfull; b += 256) lrow[b] = b;
        if (tid == 0) nlist[t] = nfull;
        return;
    }
    for (int w = tid; w < mw; w += 256) mrow[w] = 0u;
    __shared__ float q[IDX_H * IDX_D];
    __shared__ float red[8];
    __shared__ float sh_lo, sh_hi; __shared__ int sh_cnt;
    for (int i = tid; i < IDX_H * IDX_D; i += 256) q[i] = idx640[(size_t)t * IDX_W + i];
    __syncthreads();
    float *sc = blk_score + (size_t)t * nb_max;
    const float iscale = rsqrtf((float)IDX_D);
    float mx = 0.f;
    if (pre) { for (int b = tid; b < nfull; b += 256) mx = fmaxf(mx, sc[b]); }
    else
    for (int b = tid; b < nfull; b += 256) {
        const float *kb = pool_key + (size_t)b * IDX_D;
        float s = 0.f;
        #pragma unroll
        for (int h = 0; h < IDX_H; h++) {
            float d = 0.f;
            #pragma unroll 8
            for (int i = 0; i < IDX_D; i++) d = fmaf(q[h * IDX_D + i], kb[i], d);
            s += fmaxf(d * iscale, 0.f);
        }
        sc[b] = s; mx = fmaxf(mx, s);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (tid == 0) { float m = red[0]; for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]); sh_lo = 0.f; sh_hi = m; }
    __syncthreads();
    for (int it = 0; it < 48; it++) {
        if (tid == 0) sh_cnt = 0;
        __syncthreads();
        const float mid = 0.5f * (sh_lo + sh_hi);
        int c = 0;
        for (int b = tid; b < nfull; b += 256) c += (sc[b] > mid);
        #pragma unroll
        for (int off = 16; off; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
        if (lane == 0) atomicAdd(&sh_cnt, c);
        __syncthreads();
        if (tid == 0) { if (sh_cnt >= IDX_TOPK) sh_lo = mid; else sh_hi = mid; }
        __syncthreads();
    }
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    const float hi = sh_hi, lo = sh_lo;
    int c = 0;
    for (int b = tid; b < nfull; b += 256) if (sc[b] > hi) { atomicOr(&mrow[b >> 5], 1u << (b & 31)); c++; }
    #pragma unroll
    for (int off = 16; off; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    if (lane == 0) atomicAdd(&sh_cnt, c);
    __syncthreads();
    if (tid == 0) {
        int r = IDX_TOPK - sh_cnt;
        for (int b = 0; b < nfull && r > 0; b++) { const float v = sc[b]; if (v > lo && v <= hi) { mrow[b >> 5] |= 1u << (b & 31); r--; } }
    }
    __syncthreads();
    if (wid == 0) idx_compact_list(mrow, nfull, lrow, nlist + t);
}

// ---- scoring as GEMMs: planes[h][t][b] = q_h[t] . k[b] for b < nfull_max (all rows, one strided-batched
// SGEMM over the 4 heads); the fold writes blk_score[t][b] = sum_h relu(plane/sqrt(D)) for b < nfull_t only.
// Causality is unchanged: the GEMM computes blocks a row may not read; the fold and the selector read only
// that row's complete blocks (nfull_t = pos_t >> 2), all of which were finalized by k_idx_pool_rows above.
__global__ void __launch_bounds__(256) k_idx_fold_rows(const float *__restrict__ planes, int nfull_max, int T,
                                                       const QfDecodeParams *__restrict__ params, float *__restrict__ blk_score, int nb_max) {
    const int t = blockIdx.x;
    int nfull = (int)(((long)params[t].pos) >> 2);
    if (nfull > nb_max) nfull = nb_max;
    if (nfull <= IDX_TOPK) return;
    const float iscale = rsqrtf((float)IDX_D);
    const size_t plane = (size_t)nfull_max * T;
    const float *row = planes + (size_t)t * nfull_max;
    float *sc = blk_score + (size_t)t * nb_max;
    for (int b = threadIdx.x; b < nfull; b += 256) {
        float sum = 0.f;
        #pragma unroll
        for (int h = 0; h < IDX_H; h++) sum += fmaxf(row[h * plane + b] * iscale, 0.f);
        sc[b] = sum;
    }
}
static cublasHandle_t g_idx_cublas = NULL;
static float *g_idx_planes = NULL; static size_t g_idx_planes_n = 0;
static int idx_gemm_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_IDX_GEMM"); v = (e && e[0] == '0') ? 0 : 1; } return v; }

extern "C" int qf_qsa_index_rows(float *idx640, int T, const QfDecodeParams *params, const void *idx_qnorm, const void *idx_knorm,
                                 const float *inv_freq_idx, float *pool_sum, float *pool_key, int *pool_cnt,
                                 float *blk_score, int nb_max, uint32_t *mask, int mw, int *list, int *nlist, long pos0_host, cudaStream_t s) {
    if (T < 1) return -1;
    k_idx_q_rows<<<T, 128, 0, s>>>(idx640, (const uint16_t *)idx_qnorm, inv_freq_idx, params);
    const int nblk = (int)(((pos0_host + T - 1) >> 2) - (pos0_host >> 2)) + 1;   // blocks touched (host-known count; positions from device)
    k_idx_pool_rows<<<nblk, 128, 0, s>>>(idx640, T, params, pool_sum, pool_key, pool_cnt, (const uint16_t *)idx_knorm, inv_freq_idx);
    int pre = 0;
    {   // scores for every row as 4 GEMMs (heads) over the chunk's largest complete-block count
        int nfull_max = (int)((pos0_host + T) >> 2); if (nfull_max > nb_max) nfull_max = nb_max;
        if (idx_gemm_on() && nfull_max > IDX_TOPK) {
            const size_t need = (size_t)IDX_H * T * nfull_max;
            if (need > g_idx_planes_n) {
                if (g_idx_planes) cudaFree(g_idx_planes);
                if (cudaMalloc(&g_idx_planes, need * sizeof(float)) != cudaSuccess) { g_idx_planes = NULL; g_idx_planes_n = 0; return -1; }
                g_idx_planes_n = need;
            }
            if (!g_idx_cublas && cublasCreate(&g_idx_cublas) != CUBLAS_STATUS_SUCCESS) return -1;
            cublasSetStream(g_idx_cublas, s);
            const float one = 1.f, zero = 0.f;
            // column-major: C_h (nfull_max x T) = K^T (nfull_max x 128, from pool_key [nb][128]) * Q_h (128 x T, from idx640 [T][640] at column h*128)
            cublasStatus_t rc = cublasSgemmStridedBatched(g_idx_cublas, CUBLAS_OP_T, CUBLAS_OP_N, nfull_max, T, IDX_D, &one,
                                                          pool_key, IDX_D, 0, idx640, IDX_W, IDX_D, &zero,
                                                          g_idx_planes, nfull_max, (long long)nfull_max * T, IDX_H);
            if (rc != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "idx gemm: cublas status %d\n", (int)rc); return -1; }
            k_idx_fold_rows<<<T, 256, 0, s>>>(g_idx_planes, nfull_max, T, params, blk_score, nb_max);
            pre = 1;
        }
    }
    k_idx_select_rows<<<T, 256, 0, s>>>(idx640, pool_key, params, blk_score, nb_max, mask, mw, list, nlist, pre);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
