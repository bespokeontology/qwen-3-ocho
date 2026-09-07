// qf_attn_fused.cu - fused multi-head QSA decode attention (GB10/sm_121a)
#include "../qwenflash.h"
#include "qf_attn_fused.h"
#include <math.h>
#include <stdio.h>

#ifndef NHEAD
#define NHEAD 24
#endif
#ifndef NKV
#define NKV 2
#endif
#ifndef HDIM
#define HDIM 256
#endif
#ifndef QGATE
#define QGATE 512
#endif
#ifndef KVDIM
#define KVDIM (NKV * HDIM)
#endif
#ifndef ROTD
#define ROTD 64      // head_dim * partial_rotary_factor = 256 * 0.25
#endif
// Compile-time limit now lives in qf_attn_fused.h as QF_ATTN_MAXPOS.
#define MAXPOS QF_ATTN_MAXPOS
#define KVREP (NHEAD / NKV)

#define QF_ATTN_THREADS 128
#define QF_ATTN_WARPS (QF_ATTN_THREADS / 32)
#define QF_ATTN_CHUNK ((MAXPOS + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT)

static float *g_part = NULL;
static float2 *g_ml = NULL;

int qf_attn_init(void) {
    if (g_part) return 0;
    if (cudaMalloc(&g_part, (size_t)NHEAD * QF_ATTN_NSPLIT * HDIM * sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&g_ml, (size_t)NHEAD * QF_ATTN_NSPLIT * sizeof(float2)) != cudaSuccess) {
        cudaFree(g_part);
        g_part = NULL;
        return -1;
    }
    return 0;
}

void qf_attn_shutdown(void) {
    if (g_part) cudaFree(g_part);
    if (g_ml) cudaFree(g_ml);
    g_part = NULL;
    g_ml = NULL;
}

__device__ __forceinline__ float block_mean_sq(const float *row) {
    __shared__ float part[32];
    __shared__ float total;
    float v = row[threadIdx.x];
    float sum = v * v;
    #pragma unroll
    for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(~0u, sum, off);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < (HDIM + 31) / 32; w++) t += part[w];
        total = t / (float)HDIM;
    }
    __syncthreads();
    return total;
}

#ifdef QF_CANARY_TAPS
extern int qf_canary_record;
extern int qf_canary_layer;   // set by qf.cu's decode loop
// Convert a slice of the bf16 KV cache to fp32 so it can be tapped.
__global__ void k_tap_bf16(const __nv_bfloat16 *__restrict__ src,
                           float *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
static float *g_tapbuf = NULL;
void qf_canary_tap(const char *tag, int il, const float *dev, int n);
#define QF_TAP(tag, il, dev, n) do { \
    if (qf_canary_record) qf_canary_tap((tag), (il), (dev), (n)); \
} while (0)
#else
#define QF_TAP(tag, il, dev, n) ((void)0)
#endif

__global__ void k_qsa_prep(float *__restrict__ q6144, float *__restrict__ k512,
                           const float *__restrict__ v512,
                           __nv_bfloat16 *__restrict__ kc, __nv_bfloat16 *__restrict__ vc,
                           const __nv_bfloat16 *__restrict__ q_norm,
                           const __nv_bfloat16 *__restrict__ k_norm,
                           const float *__restrict__ inv_freq,
                           const QfDecodeParams *__restrict__ params) {
    static_assert(HDIM % 32 == 0 && HDIM >= 64 && HDIM <= 1024, "prep kernel geometry");
    const int h = blockIdx.x, tid = threadIdx.x;
    const long pos = params->pos;
    // Rope over the first ROTD dims, pairing dim i with dim i+ROTD/2.
    //
    // Each thread computes its OWN output from the two NORMALIZED inputs of its
    // pair. The previous form had threads 0..31 write both row[tid] and
    // row[tid+32], and then every thread store row[tid] to the KV cache - so
    // threads 32..63 stored a value another thread had just overwritten.
    // Despite the __syncthreads(), the compiler kept each thread's own earlier
    // load of row[tid] (from the normalization step) live in a register and
    // reused it: k512/q6144 are __restrict__, so nothing told it to reload.
    // The K cache therefore held PRE-rope values in dims 32..63.
    //
    // Position 0 hid the bug completely (rope at pos 0 is the identity), which
    // is why a one-token prompt matched the reference exactly and every longer
    // prompt did not.
    const int HALF = ROTD / 2;
    float *row = (h < NHEAD) ? q6144 + (size_t)h * QGATE
                             : k512 + (size_t)(h - NHEAD) * HDIM;
    const __nv_bfloat16 *nw = (h < NHEAD) ? q_norm : k_norm;
    float ms = block_mean_sq(row);
    const float nv = row[tid] * rsqrtf(ms + 1e-6f) * (1.f + __bfloat162float(nw[tid]));
    row[tid] = nv;
    __syncthreads();
    float out = nv;
    if (tid < HALF) {
        float f = (float)pos * inv_freq[tid];
        out = nv * cosf(f) - row[tid + HALF] * sinf(f);
    } else if (tid < ROTD) {
        float f = (float)pos * inv_freq[tid - HALF];
        out = nv * cosf(f) + row[tid - HALF] * sinf(f);
    }
    __syncthreads();
    row[tid] = out;
    if (h >= NHEAD) {
        const int kh = h - NHEAD;
        kc[(size_t)pos * KVDIM + kh * HDIM + tid] = __float2bfloat16(out);
        vc[(size_t)pos * KVDIM + kh * HDIM + tid] = __float2bfloat16(v512[kh * HDIM + tid]);
    }
}
// M-row clone: blockIdx.y = request row; each row has its own q/k/v slice, KV
// cache (row stride kvspan) and decode params. Identical arithmetic.
__global__ void k_qsa_prep_M(float *__restrict__ q6144, float *__restrict__ k512,
                           const float *__restrict__ v512,
                           __nv_bfloat16 *__restrict__ kc, __nv_bfloat16 *__restrict__ vc,
                           const __nv_bfloat16 *__restrict__ q_norm,
                           const __nv_bfloat16 *__restrict__ k_norm,
                           const float *__restrict__ inv_freq,
                           const QfDecodeParams *__restrict__ params, size_t kvspan) {
    { const int mrow = blockIdx.y;
      q6144 += (size_t)mrow * NHEAD * QGATE; k512 += (size_t)mrow * KVDIM; v512 += (size_t)mrow * KVDIM;
      kc += (size_t)mrow * kvspan; vc += (size_t)mrow * kvspan; params += mrow; }
    static_assert(HDIM % 32 == 0 && HDIM >= 64 && HDIM <= 1024, "prep kernel geometry");
    const int h = blockIdx.x, tid = threadIdx.x;
    const long pos = params->pos;
    // Rope over the first ROTD dims, pairing dim i with dim i+ROTD/2.
    //
    // Each thread computes its OWN output from the two NORMALIZED inputs of its
    // pair. The previous form had threads 0..31 write both row[tid] and
    // row[tid+32], and then every thread store row[tid] to the KV cache - so
    // threads 32..63 stored a value another thread had just overwritten.
    // Despite the __syncthreads(), the compiler kept each thread's own earlier
    // load of row[tid] (from the normalization step) live in a register and
    // reused it: k512/q6144 are __restrict__, so nothing told it to reload.
    // The K cache therefore held PRE-rope values in dims 32..63.
    //
    // Position 0 hid the bug completely (rope at pos 0 is the identity), which
    // is why a one-token prompt matched the reference exactly and every longer
    // prompt did not.
    const int HALF = ROTD / 2;
    float *row = (h < NHEAD) ? q6144 + (size_t)h * QGATE
                             : k512 + (size_t)(h - NHEAD) * HDIM;
    const __nv_bfloat16 *nw = (h < NHEAD) ? q_norm : k_norm;
    float ms = block_mean_sq(row);
    const float nv = row[tid] * rsqrtf(ms + 1e-6f) * (1.f + __bfloat162float(nw[tid]));
    row[tid] = nv;
    __syncthreads();
    float out = nv;
    if (tid < HALF) {
        float f = (float)pos * inv_freq[tid];
        out = nv * cosf(f) - row[tid + HALF] * sinf(f);
    } else if (tid < ROTD) {
        float f = (float)pos * inv_freq[tid - HALF];
        out = nv * cosf(f) + row[tid - HALF] * sinf(f);
    }
    __syncthreads();
    row[tid] = out;
    if (h >= NHEAD) {
        const int kh = h - NHEAD;
        kc[(size_t)pos * KVDIM + kh * HDIM + tid] = __float2bfloat16(out);
        vc[(size_t)pos * KVDIM + kh * HDIM + tid] = __float2bfloat16(v512[kh * HDIM + tid]);
    }
}

__device__ __forceinline__ int qsa_selected(const uint32_t *mask, long pos_idx, long pos) {
    if (!mask) return 1;
    long block = pos_idx >> 2;
    if (block == (pos >> 2)) return 1;
    return (int)((mask[block >> 5] >> (block & 31)) & 1u);
}

__global__ void k_attn_partial(const float *__restrict__ q6144,
                               const __nv_bfloat16 *__restrict__ K,
                               const __nv_bfloat16 *__restrict__ V,
                               float *__restrict__ part, float2 *__restrict__ ml,
                               const QfDecodeParams *__restrict__ params,
                               const uint32_t *__restrict__ mask) {
    static_assert(HDIM % 8 == 0, "partial kernel geometry");
    const int h = blockIdx.x, split = blockIdx.y;
    const int kh = h / KVREP;
    const long pos = params->pos, ntok = pos + 1;
    const long chunk = (ntok + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT;
    const long lo = (long)split * chunk;
    const long hi = (lo + chunk < ntok) ? lo + chunk : ntok;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float *part_out = part + ((size_t)h * QF_ATTN_NSPLIT + split) * HDIM;

    __shared__ float qs[HDIM];
    __shared__ float scores[QF_ATTN_CHUNK];
    __shared__ float reduce[QF_ATTN_WARPS];
    __shared__ float broadcast;

    if (lo >= hi) {
        if (tid == 0) ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(-1e30f, 0.f);
        for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) part_out[d] = 0.f;
        return;
    }
    const long len = hi - lo;
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS)
        qs[d] = q6144[(size_t)h * QGATE + d];
    __syncthreads();

    const float scale = rsqrtf((float)HDIM);
    const __nv_bfloat16 *Khead = K + kh * HDIM;
    for (long p = lo + warp; p < hi; p += QF_ATTN_WARPS) {
        if (mask && !qsa_selected(mask, p, pos)) {
            if (lane == 0) scores[p - lo] = -1e30f;
            continue;
        }
        const __nv_bfloat16 *key = Khead + (size_t)p * KVDIM;
        float acc = 0.f;
        #pragma unroll
        for (int d0 = lane * 8; d0 < HDIM; d0 += 32 * 8) {
            uint4 raw = *(const uint4 *)(key + d0);
            const __nv_bfloat16 *kb = (const __nv_bfloat16 *)&raw;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc = fmaf(qs[d0 + j], __bfloat162float(kb[j]), acc);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
        if (lane == 0) scores[p - lo] = acc * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (long i = tid; i < len; i += QF_ATTN_THREADS) local_max = fmaxf(local_max, scores[i]);
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, off));
    if (lane == 0) reduce[warp] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = reduce[0];
        #pragma unroll
        for (int w = 1; w < QF_ATTN_WARPS; w++) m = fmaxf(m, reduce[w]);
        broadcast = m;
    }
    __syncthreads();
    const float max_score = broadcast;

    float local_sum = 0.f;
    for (long i = tid; i < len; i += QF_ATTN_THREADS) {
        float e = expf(scores[i] - max_score);
        scores[i] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_sum += __shfl_down_sync(~0u, local_sum, off);
    if (lane == 0) reduce[warp] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < QF_ATTN_WARPS; w++) sum += reduce[w];
        broadcast = sum;
        ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(max_score, sum);
    }
    __syncthreads();

    const __nv_bfloat16 *Vhead = V + kh * HDIM;
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) {
        float acc = 0.f;
        for (long p = 0; p < len; p++)
            acc = fmaf(scores[p], __bfloat162float(Vhead[(size_t)(lo + p) * KVDIM + d]), acc);
        part_out[d] = acc;
    }
}
// M-row clone: blockIdx.z = request row; per-row KV slice and split workspace.
__global__ void k_attn_partial_M(const float *__restrict__ q6144,
                               const __nv_bfloat16 *__restrict__ K,
                               const __nv_bfloat16 *__restrict__ V,
                               float *__restrict__ part, float2 *__restrict__ ml,
                               const QfDecodeParams *__restrict__ params,
                               const uint32_t *__restrict__ mask, size_t kvspan, size_t mask_stride) {
    { const int mrow = blockIdx.z;
      q6144 += (size_t)mrow * NHEAD * QGATE; K += (size_t)mrow * kvspan; V += (size_t)mrow * kvspan;
      part += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT * HDIM; ml += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT;
      params += mrow; if (mask) mask += (size_t)mrow * mask_stride; }
    static_assert(HDIM % 8 == 0, "partial kernel geometry");
    const int h = blockIdx.x, split = blockIdx.y;
    const int kh = h / KVREP;
    const long pos = params->pos, ntok = pos + 1;
    const long chunk = (ntok + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT;
    const long lo = (long)split * chunk;
    const long hi = (lo + chunk < ntok) ? lo + chunk : ntok;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float *part_out = part + ((size_t)h * QF_ATTN_NSPLIT + split) * HDIM;

    __shared__ float qs[HDIM];
    __shared__ float scores[QF_ATTN_CHUNK];
    __shared__ float reduce[QF_ATTN_WARPS];
    __shared__ float broadcast;

    if (lo >= hi) {
        if (tid == 0) ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(-1e30f, 0.f);
        for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) part_out[d] = 0.f;
        return;
    }
    const long len = hi - lo;
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS)
        qs[d] = q6144[(size_t)h * QGATE + d];
    __syncthreads();

    const float scale = rsqrtf((float)HDIM);
    const __nv_bfloat16 *Khead = K + kh * HDIM;
    for (long p = lo + warp; p < hi; p += QF_ATTN_WARPS) {
        if (mask && !qsa_selected(mask, p, pos)) {
            if (lane == 0) scores[p - lo] = -1e30f;
            continue;
        }
        const __nv_bfloat16 *key = Khead + (size_t)p * KVDIM;
        float acc = 0.f;
        #pragma unroll
        for (int d0 = lane * 8; d0 < HDIM; d0 += 32 * 8) {
            uint4 raw = *(const uint4 *)(key + d0);
            const __nv_bfloat16 *kb = (const __nv_bfloat16 *)&raw;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc = fmaf(qs[d0 + j], __bfloat162float(kb[j]), acc);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
        if (lane == 0) scores[p - lo] = acc * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (long i = tid; i < len; i += QF_ATTN_THREADS) local_max = fmaxf(local_max, scores[i]);
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, off));
    if (lane == 0) reduce[warp] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = reduce[0];
        #pragma unroll
        for (int w = 1; w < QF_ATTN_WARPS; w++) m = fmaxf(m, reduce[w]);
        broadcast = m;
    }
    __syncthreads();
    const float max_score = broadcast;

    float local_sum = 0.f;
    for (long i = tid; i < len; i += QF_ATTN_THREADS) {
        float e = expf(scores[i] - max_score);
        scores[i] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_sum += __shfl_down_sync(~0u, local_sum, off);
    if (lane == 0) reduce[warp] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < QF_ATTN_WARPS; w++) sum += reduce[w];
        broadcast = sum;
        ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(max_score, sum);
    }
    __syncthreads();

    const __nv_bfloat16 *Vhead = V + kh * HDIM;
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) {
        float acc = 0.f;
        for (long p = 0; p < len; p++)
            acc = fmaf(scores[p], __bfloat162float(Vhead[(size_t)(lo + p) * KVDIM + d]), acc);
        part_out[d] = acc;
    }
}

__global__ void k_attn_combine(const float *__restrict__ part,
                               const float2 *__restrict__ ml,
                               const float *__restrict__ q6144,
                               float *__restrict__ attn_out) {
    const int h = blockIdx.x, tid = threadIdx.x;
    __shared__ float reduce[32];
    __shared__ float broadcast;
    float local_max = -1e30f;
    for (int split = tid; split < QF_ATTN_NSPLIT; split += blockDim.x)
        local_max = fmaxf(local_max, ml[(size_t)h * QF_ATTN_NSPLIT + split].x);
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, off));
    if ((tid & 31) == 0) reduce[tid >> 5] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -1e30f;
        for (int w = 0; w < (HDIM + 31) / 32; w++) m = fmaxf(m, reduce[w]);
        broadcast = m;
    }
    __syncthreads();
    const float max_score = broadcast;

    float weights[QF_ATTN_NSPLIT];
    float sum = 0.f;
    #pragma unroll
    for (int split = 0; split < QF_ATTN_NSPLIT; split++) {
        float2 partial = ml[(size_t)h * QF_ATTN_NSPLIT + split];
        weights[split] = expf(partial.x - max_score);
        sum += partial.y * weights[split];
    }
    const float *head_part = part + (size_t)h * QF_ATTN_NSPLIT * HDIM + tid;
    float acc = 0.f;
    #pragma unroll
    for (int split = 0; split < QF_ATTN_NSPLIT; split++)
        acc = fmaf(weights[split], head_part[(size_t)split * HDIM], acc);
    float gate = q6144[(size_t)h * QGATE + HDIM + tid];
    attn_out[(size_t)h * HDIM + tid] = (acc / (sum + 1e-9f)) / (1.f + expf(-gate));
}
// M-row clone: blockIdx.y = request row.
__global__ void k_attn_combine_M(const float *__restrict__ part,
                               const float2 *__restrict__ ml,
                               const float *__restrict__ q6144,
                               float *__restrict__ attn_out) {
    { const int mrow = blockIdx.y;
      part += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT * HDIM; ml += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT;
      q6144 += (size_t)mrow * NHEAD * QGATE; attn_out += (size_t)mrow * NHEAD * HDIM; }
    const int h = blockIdx.x, tid = threadIdx.x;
    __shared__ float reduce[32];
    __shared__ float broadcast;
    float local_max = -1e30f;
    for (int split = tid; split < QF_ATTN_NSPLIT; split += blockDim.x)
        local_max = fmaxf(local_max, ml[(size_t)h * QF_ATTN_NSPLIT + split].x);
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, off));
    if ((tid & 31) == 0) reduce[tid >> 5] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -1e30f;
        for (int w = 0; w < (HDIM + 31) / 32; w++) m = fmaxf(m, reduce[w]);
        broadcast = m;
    }
    __syncthreads();
    const float max_score = broadcast;

    float weights[QF_ATTN_NSPLIT];
    float sum = 0.f;
    #pragma unroll
    for (int split = 0; split < QF_ATTN_NSPLIT; split++) {
        float2 partial = ml[(size_t)h * QF_ATTN_NSPLIT + split];
        weights[split] = expf(partial.x - max_score);
        sum += partial.y * weights[split];
    }
    const float *head_part = part + (size_t)h * QF_ATTN_NSPLIT * HDIM + tid;
    float acc = 0.f;
    #pragma unroll
    for (int split = 0; split < QF_ATTN_NSPLIT; split++)
        acc = fmaf(weights[split], head_part[(size_t)split * HDIM], acc);
    float gate = q6144[(size_t)h * QGATE + HDIM + tid];
    attn_out[(size_t)h * HDIM + tid] = (acc / (sum + 1e-9f)) / (1.f + expf(-gate));
}

// ---- List-driven partial kernel: the row attends its compact list of selected
// complete blocks (4 positions each) plus its current (tail) block, split across
// QF_ATTN_NSPLIT CTAs by list index. Same online-softmax/partial contract as
// k_attn_partial(_M): part/ml per (head, split), combined by k_attn_combine(_M).
// blockIdx.z = row (chunk) or 0 (decode); list row stride list_stride ints.
#define QF_ATTN_LIST_MAX 513
__global__ void k_attn_partial_list(const float *__restrict__ q6144,
                                    const __nv_bfloat16 *__restrict__ K,
                                    const __nv_bfloat16 *__restrict__ V,
                                    float *__restrict__ part, float2 *__restrict__ ml,
                                    const QfDecodeParams *__restrict__ params,
                                    const int *__restrict__ list, const int *__restrict__ nlist,
                                    size_t list_stride) {
    { const int mrow = blockIdx.z;
      q6144 += (size_t)mrow * NHEAD * QGATE;
      part += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT * HDIM; ml += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT;
      params += mrow; list += (size_t)mrow * list_stride; nlist += mrow; }
    const int h = blockIdx.x, split = blockIdx.y;
    const int kh = h / KVREP;
    const long pos = params->pos;
    const int nl = *nlist;                       // selected complete blocks
    const int nb = nl + 1;                       // + the current block
    const int per = (nb + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT;
    const int lo = split * per;
    const int hi = (lo + per < nb) ? lo + per : nb;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float *part_out = part + ((size_t)h * QF_ATTN_NSPLIT + split) * HDIM;
    __shared__ float qs[HDIM];
    __shared__ float scores[QF_ATTN_CHUNK];
    __shared__ int   spos[QF_ATTN_CHUNK];
    __shared__ float reduce[QF_ATTN_WARPS];
    __shared__ float broadcast;
    if (lo >= hi) {
        if (tid == 0) ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(-1e30f, 0.f);
        for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) part_out[d] = 0.f;
        return;
    }
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) qs[d] = q6144[(size_t)h * QGATE + d];
    // positions of this split's blocks, packed
    __shared__ int len_sh;
    if (tid == 0) {
        int n = 0;
        for (int i = lo; i < hi; i++) {
            const long b = (i < nl) ? (long)list[i] : (pos >> 2);
            const long p0 = b * 4, p1 = (i < nl) ? p0 + 4 : pos + 1;
            for (long p = p0; p < p1 && n < QF_ATTN_CHUNK; p++) spos[n++] = (int)p;
        }
        len_sh = n;
    }
    __syncthreads();
    const int len = len_sh;
    const float scale = rsqrtf((float)HDIM);
    const __nv_bfloat16 *Khead = K + kh * HDIM;
    for (int i = warp; i < len; i += QF_ATTN_WARPS) {
        const __nv_bfloat16 *key = Khead + (size_t)spos[i] * KVDIM;
        float acc = 0.f;
        #pragma unroll
        for (int d0 = lane * 8; d0 < HDIM; d0 += 32 * 8) {
            uint4 raw = *(const uint4 *)(key + d0);
            const __nv_bfloat16 *kb = (const __nv_bfloat16 *)&raw;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc = fmaf(qs[d0 + j], __bfloat162float(kb[j]), acc);
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
        if (lane == 0) scores[i] = acc * scale;
    }
    __syncthreads();
    float local_max = -1e30f;
    for (int i = tid; i < len; i += QF_ATTN_THREADS) local_max = fmaxf(local_max, scores[i]);
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, off));
    if (lane == 0) reduce[warp] = local_max;
    __syncthreads();
    if (tid == 0) { float m = reduce[0]; for (int w = 1; w < QF_ATTN_WARPS; w++) m = fmaxf(m, reduce[w]); broadcast = m; }
    __syncthreads();
    const float max_score = broadcast;
    float local_sum = 0.f;
    for (int i = tid; i < len; i += QF_ATTN_THREADS) { const float e = expf(scores[i] - max_score); scores[i] = e; local_sum += e; }
    #pragma unroll
    for (int off = 16; off; off >>= 1) local_sum += __shfl_down_sync(~0u, local_sum, off);
    if (lane == 0) reduce[warp] = local_sum;
    __syncthreads();
    if (tid == 0) { float sum = 0.f; for (int w = 0; w < QF_ATTN_WARPS; w++) sum += reduce[w]; broadcast = sum;
                    ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(max_score, sum); }
    __syncthreads();
    const __nv_bfloat16 *Vhead = V + kh * HDIM;
    for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) {
        float acc = 0.f;
        for (int i = 0; i < len; i++) acc = fmaf(scores[i], __bfloat162float(Vhead[(size_t)spos[i] * KVDIM + d]), acc);
        part_out[d] = acc;
    }
}

void qf_attn_qsa_layer(float *q6144, float *k512, float *v512,
                       __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                       const void *q_norm, const void *k_norm, const float *inv_freq,
                       const QfDecodeParams *params, const uint32_t *qsa_mask,
                       float *attn_out, cudaStream_t stream) {
    if (!g_part && qf_attn_init() != 0) {
        fprintf(stderr, "qf_attn: workspace alloc failed\n");
        return;
    }
    QF_TAP("qsa_qraw", qf_canary_layer, q6144, NHEAD * QGATE);
    k_qsa_prep<<<NHEAD + NKV, HDIM, 0, stream>>>(q6144, k512, v512, kc, vc,
        (const __nv_bfloat16 *)q_norm, (const __nv_bfloat16 *)k_norm, inv_freq, params);
    QF_TAP("qsa_qprep", qf_canary_layer, q6144, NHEAD * QGATE);
    QF_TAP("qsa_kprep", qf_canary_layer, k512, NKV * HDIM);
#ifdef QF_CANARY_TAPS
    if (qf_canary_record) {
        const int NP = 8;                       // canary prompt length
        const int n = NP * KVDIM;
        if (!g_tapbuf) cudaMalloc(&g_tapbuf, (size_t)n * sizeof(float));
        if (g_tapbuf) {
            k_tap_bf16<<<(n + 255) / 256, 256, 0, stream>>>(kc, g_tapbuf, n);
            qf_canary_tap("qsa_kc", qf_canary_layer, g_tapbuf, n);
            k_tap_bf16<<<(n + 255) / 256, 256, 0, stream>>>(vc, g_tapbuf, n);
            qf_canary_tap("qsa_vc", qf_canary_layer, g_tapbuf, n);
        }
    }
#endif
    dim3 grid(NHEAD, QF_ATTN_NSPLIT);
    k_attn_partial<<<grid, QF_ATTN_THREADS, 0, stream>>>(q6144, kc, vc, g_part, g_ml, params, qsa_mask);
    k_attn_combine<<<NHEAD, HDIM, 0, stream>>>(g_part, g_ml, q6144, attn_out);
}
// ---- GQA-grouped list kernel (2026-09-07): one CTA per (KV head, split) computes the KVREP query heads
// that share that KV head, loading each selected K/V row ONCE instead of once per query head (12x fewer
// K/V bytes requested). Same positions in the same order, the same per-head arithmetic (lane-8 partial
// sums, warp shuffle reduce, sequential PV accumulation) and the same part/ml contract as
// k_attn_partial_list, so k_attn_combine is unchanged. QF_ATTN_LIST_GQA=0 restores the per-head kernel.
#define QF_ATTN_LIST_CHUNK (4 * ((QF_ATTN_LIST_MAX + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT))
__global__ void __launch_bounds__(QF_ATTN_THREADS) k_attn_partial_list_gqa(const float *__restrict__ q6144,
                                    const __nv_bfloat16 *__restrict__ K,
                                    const __nv_bfloat16 *__restrict__ V,
                                    float *__restrict__ part, float2 *__restrict__ ml,
                                    const QfDecodeParams *__restrict__ params,
                                    const int *__restrict__ list, const int *__restrict__ nlist,
                                    size_t list_stride) {
    { const int mrow = blockIdx.z;
      q6144 += (size_t)mrow * NHEAD * QGATE;
      part += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT * HDIM; ml += (size_t)mrow * NHEAD * QF_ATTN_NSPLIT;
      params += mrow; list += (size_t)mrow * list_stride; nlist += mrow; }
    const int kh = blockIdx.x, split = blockIdx.y;
    const long pos = params->pos;
    const int nl = *nlist;                       // selected complete blocks
    const int nb = nl + 1;                       // + the current block
    const int per = (nb + QF_ATTN_NSPLIT - 1) / QF_ATTN_NSPLIT;
    const int lo = split * per;
    const int hi = (lo + per < nb) ? lo + per : nb;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float qs[KVREP][HDIM];
    __shared__ float scores[KVREP][QF_ATTN_LIST_CHUNK];
    __shared__ int   spos[QF_ATTN_LIST_CHUNK];
    __shared__ float reduce[KVREP][QF_ATTN_WARPS];
    __shared__ float bmax[KVREP];
    if (lo >= hi) {
        for (int j = 0; j < KVREP; j++) {
            const int h = kh * KVREP + j;
            if (tid == 0) ml[(size_t)h * QF_ATTN_NSPLIT + split] = make_float2(-1e30f, 0.f);
            float *po = part + ((size_t)h * QF_ATTN_NSPLIT + split) * HDIM;
            for (int d = tid; d < HDIM; d += QF_ATTN_THREADS) po[d] = 0.f;
        }
        return;
    }
    for (int idx = tid; idx < KVREP * HDIM; idx += QF_ATTN_THREADS) {
        const int j = idx / HDIM, d = idx - j * HDIM;
        qs[j][d] = q6144[(size_t)(kh * KVREP + j) * QGATE + d];
    }
    // positions of this split's blocks, packed in list order; only the last entry (the current block, index
    // nl) can be partial, so entry i starts at 4*(i-lo) - the same sequence the per-head kernel builds.
    for (int i = lo + tid; i < hi; i += QF_ATTN_THREADS) {
        const long b = (i < nl) ? (long)list[i] : (pos >> 2);
        const long p0 = b * 4, p1 = (i < nl) ? p0 + 4 : pos + 1;
        int n = 4 * (i - lo);
        for (long p = p0; p < p1; p++) spos[n++] = (int)p;
    }
    __syncthreads();
    int len = 4 * (hi - lo);
    if (hi == nb) len -= 3 - (int)(pos & 3);   // the current block holds (pos & 3) + 1 positions
    const float scale = rsqrtf((float)HDIM);
    const __nv_bfloat16 *Khead = K + kh * HDIM;
    for (int i0 = warp; i0 < len; i0 += 4 * QF_ATTN_WARPS) {
        uint4 raw[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const int i = i0 + u * QF_ATTN_WARPS;
            if (i < len) raw[u] = *(const uint4 *)(Khead + (size_t)spos[i] * KVDIM + lane * 8);
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const int i = i0 + u * QF_ATTN_WARPS;
            if (i >= len) break;
            const __nv_bfloat16 *kb = (const __nv_bfloat16 *)&raw[u];
            float kf[8];
            #pragma unroll
            for (int t = 0; t < 8; t++) kf[t] = __bfloat162float(kb[t]);
            const int d0 = lane * 8;
            #pragma unroll
            for (int j = 0; j < KVREP; j++) {
                float acc = 0.f;
                #pragma unroll
                for (int t = 0; t < 8; t++) acc = fmaf(qs[j][d0 + t], kf[t], acc);
                #pragma unroll
                for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
                if (lane == 0) scores[j][i] = acc * scale;
            }
        }
    }
    __syncthreads();
    float lred[KVREP];
    #pragma unroll
    for (int j = 0; j < KVREP; j++) lred[j] = -1e30f;
    for (int i = tid; i < len; i += QF_ATTN_THREADS) {
        #pragma unroll
        for (int j = 0; j < KVREP; j++) lred[j] = fmaxf(lred[j], scores[j][i]);
    }
    #pragma unroll
    for (int j = 0; j < KVREP; j++) {
        float v = lred[j];
        #pragma unroll
        for (int off = 16; off; off >>= 1) v = fmaxf(v, __shfl_down_sync(~0u, v, off));
        if (lane == 0) reduce[j][warp] = v;
    }
    __syncthreads();
    if (tid < KVREP) { float m = reduce[tid][0]; for (int w = 1; w < QF_ATTN_WARPS; w++) m = fmaxf(m, reduce[tid][w]); bmax[tid] = m; }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < KVREP; j++) lred[j] = 0.f;
    for (int i = tid; i < len; i += QF_ATTN_THREADS) {
        #pragma unroll
        for (int j = 0; j < KVREP; j++) { const float e = expf(scores[j][i] - bmax[j]); scores[j][i] = e; lred[j] += e; }
    }
    #pragma unroll
    for (int j = 0; j < KVREP; j++) {
        float v = lred[j];
        #pragma unroll
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(~0u, v, off);
        if (lane == 0) reduce[j][warp] = v;
    }
    __syncthreads();
    if (tid < KVREP) { float sum = 0.f; for (int w = 0; w < QF_ATTN_WARPS; w++) sum += reduce[tid][w];
                       ml[(size_t)(kh * KVREP + tid) * QF_ATTN_NSPLIT + split] = make_float2(bmax[tid], sum); }
    const __nv_bfloat16 *Vhead = V + kh * HDIM;
    float acc[KVREP][HDIM / QF_ATTN_THREADS];
    #pragma unroll
    for (int j = 0; j < KVREP; j++)
        #pragma unroll
        for (int c = 0; c < HDIM / QF_ATTN_THREADS; c++) acc[j][c] = 0.f;
    #pragma unroll 4
    for (int i = 0; i < len; i++) {
        const __nv_bfloat16 *vrow = Vhead + (size_t)spos[i] * KVDIM;
        float v[HDIM / QF_ATTN_THREADS];
        #pragma unroll
        for (int c = 0; c < HDIM / QF_ATTN_THREADS; c++) v[c] = __bfloat162float(vrow[tid + c * QF_ATTN_THREADS]);
        #pragma unroll
        for (int j = 0; j < KVREP; j++) {
            const float sc = scores[j][i];
            #pragma unroll
            for (int c = 0; c < HDIM / QF_ATTN_THREADS; c++) acc[j][c] = fmaf(sc, v[c], acc[j][c]);
        }
    }
    #pragma unroll
    for (int j = 0; j < KVREP; j++) {
        float *po = part + ((size_t)(kh * KVREP + j) * QF_ATTN_NSPLIT + split) * HDIM;
        #pragma unroll
        for (int c = 0; c < HDIM / QF_ATTN_THREADS; c++) po[tid + c * QF_ATTN_THREADS] = acc[j][c];
    }
}
// 2026-09-07 receipt (8K recipe, clean AMD server): 300/300 tokens identical to the per-head kernel; Spark tail 25.0 -> 24.7 ms.
// Operator: KEEP as default ("correct; 0-0.3 ms observed benefit; magnitude not yet established"); the per-head kernel
// stays as the A/B fallback: QF_ATTN_LIST_GQA=0.
static int attn_list_gqa_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_ATTN_LIST_GQA"); v = (e && e[0] == '0') ? 0 : 1; } return v; }
// Decode step over the compact selected-block list (bounded cost at any context).
void qf_attn_qsa_layer_list(float *q6144, float *k512, float *v512,
                            __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                            const void *q_norm, const void *k_norm, const float *inv_freq,
                            const QfDecodeParams *params, const int *list, const int *nlist,
                            float *attn_out, cudaStream_t stream) {
    if (!g_part && qf_attn_init() != 0) { fprintf(stderr, "qf_attn: workspace alloc failed\n"); return; }
    k_qsa_prep<<<NHEAD + NKV, HDIM, 0, stream>>>(q6144, k512, v512, kc, vc,
        (const __nv_bfloat16 *)q_norm, (const __nv_bfloat16 *)k_norm, inv_freq, params);
    if (attn_list_gqa_on())
        k_attn_partial_list_gqa<<<dim3(NKV, QF_ATTN_NSPLIT, 1), QF_ATTN_THREADS, 0, stream>>>(
            q6144, kc, vc, g_part, g_ml, params, list, nlist, (size_t)0);
    else
        k_attn_partial_list<<<dim3(NHEAD, QF_ATTN_NSPLIT, 1), QF_ATTN_THREADS, 0, stream>>>(
            q6144, kc, vc, g_part, g_ml, params, list, nlist, (size_t)0);
    k_attn_combine<<<NHEAD, HDIM, 0, stream>>>(g_part, g_ml, q6144, attn_out);
}

// ---- M-row QSA attention: the three kernels above with the request row in the
// grid. One launch each for all M rows instead of 3*M launches. Rows are
// independent sequences: their own q/k/v slices, KV caches (row stride kvspan
// elements), positions (params[row]) and a per-row split workspace. ----
static float  *g_part_M = NULL;
static float2 *g_ml_M   = NULL;
static int qf_attn_init_M(void) {
    if (g_part_M) return 0;
    if (cudaMalloc(&g_part_M, (size_t)QF_ATTN_MAXM * NHEAD * QF_ATTN_NSPLIT * HDIM * sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&g_ml_M, (size_t)QF_ATTN_MAXM * NHEAD * QF_ATTN_NSPLIT * sizeof(float2)) != cudaSuccess) {
        cudaFree(g_part_M); g_part_M = NULL; return -1;
    }
    return 0;
}
int qf_attn_qsa_layer_Mgrid(float *q6144, float *k512, float *v512,
                            __nv_bfloat16 *kc, __nv_bfloat16 *vc, size_t kvspan,
                            const void *q_norm, const void *k_norm, const float *inv_freq,
                            const QfDecodeParams *params, float *attn_out, int M,
                            cudaStream_t stream) {
    if (M < 1 || M > QF_ATTN_MAXM) return -1;
    if (!g_part_M && qf_attn_init_M() != 0) {
        fprintf(stderr, "qf_attn: M-row workspace alloc failed\n");
        return -1;
    }
    k_qsa_prep_M<<<dim3(NHEAD + NKV, M), HDIM, 0, stream>>>(q6144, k512, v512, kc, vc,
        (const __nv_bfloat16 *)q_norm, (const __nv_bfloat16 *)k_norm, inv_freq, params, kvspan);
    k_attn_partial_M<<<dim3(NHEAD, QF_ATTN_NSPLIT, M), QF_ATTN_THREADS, 0, stream>>>(
        q6144, kc, vc, g_part_M, g_ml_M, params, NULL, kvspan, (size_t)0);
    k_attn_combine_M<<<dim3(NHEAD, M), HDIM, 0, stream>>>(g_part_M, g_ml_M, q6144, attn_out);
    return 0;
}

// ---- Chunk-causal attention for the prefill chunk (see qf_attn_fused.h): the
// M-row kernels with a ZERO cache row stride. k_qsa_prep_M writes the T
// keys/values of the chunk (stream order), then k_attn_partial_M row t reads
// [0, params[t].pos] of that same cache. 3 launches per layer instead of 3*T. ----
#define QF_ATTN_CHUNK_MAXT 256
static float  *g_part_C = NULL;
static float2 *g_ml_C   = NULL;
extern "C" int qf_fi_attn_chunk(const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, const uint32_t *, int, long, int, float *, cudaStream_t);
int qf_attn_qsa_chunk(float *q6144, float *k512, float *v512,
                      __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                      const void *q_norm, const void *k_norm, const float *inv_freq,
                      const QfDecodeParams *params, const uint32_t *mask, int mask_words,
                      const int *list, const int *nlist, int list_stride,
                      float *attn_out, int T, long pos0, cudaStream_t stream) {
    if (T < 1 || T > 65535) return -1;
    if (!g_part_C) {
        if (cudaMalloc(&g_part_C, (size_t)QF_ATTN_CHUNK_MAXT * NHEAD * QF_ATTN_NSPLIT * HDIM * sizeof(float)) != cudaSuccess)
            return -1;
        if (cudaMalloc(&g_ml_C, (size_t)QF_ATTN_CHUNK_MAXT * NHEAD * QF_ATTN_NSPLIT * sizeof(float2)) != cudaSuccess) {
            cudaFree(g_part_C); g_part_C = NULL; return -1;
        }
    }
    k_qsa_prep_M<<<dim3(NHEAD + NKV, T), HDIM, 0, stream>>>(q6144, k512, v512, kc, vc,
        (const __nv_bfloat16 *)q_norm, (const __nv_bfloat16 *)k_norm, inv_freq, params, (size_t)0);
    // FlashInfer tensor-core prefill (QF_PF_FI=0 restores the list kernels): the per-row selected-block
    // masks become FlashInfer's custom bit mask; K/V are read straight from the cache (NHD layout).
    {   static int fi = -1;
        if (fi < 0) { const char *e = getenv("QF_PF_FI"); fi = (e && e[0] == '0') ? 0 : 1; }
        if (fi && mask && list) {
            if (qf_fi_attn_chunk(q6144, kc, vc, mask, mask_words, pos0, T, attn_out, stream) == 0) return 0;
            fprintf(stderr, "qf_attn_qsa_chunk: FlashInfer path failed, falling back to the list kernels\n");
        }
    }
    // All T keys/values are in the cache now; queries run in slices of
    // QF_ATTN_CHUNK_MAXT rows (the split workspace), each row over [0, pos_t].
    for (int q0 = 0; q0 < T; q0 += QF_ATTN_CHUNK_MAXT) {
        const int ts = (T - q0) < QF_ATTN_CHUNK_MAXT ? (T - q0) : QF_ATTN_CHUNK_MAXT;
        if (list)
            k_attn_partial_list<<<dim3(NHEAD, QF_ATTN_NSPLIT, ts), QF_ATTN_THREADS, 0, stream>>>(
                q6144 + (size_t)q0 * NHEAD * QGATE, kc, vc, g_part_C, g_ml_C, params + q0,
                list + (size_t)q0 * list_stride, nlist + q0, (size_t)list_stride);
        else
        k_attn_partial_M<<<dim3(NHEAD, QF_ATTN_NSPLIT, ts), QF_ATTN_THREADS, 0, stream>>>(
            q6144 + (size_t)q0 * NHEAD * QGATE, kc, vc, g_part_C, g_ml_C, params + q0,
            mask ? mask + (size_t)q0 * mask_words : NULL, (size_t)0, (size_t)mask_words);
        k_attn_combine_M<<<dim3(NHEAD, ts), HDIM, 0, stream>>>(g_part_C, g_ml_C, q6144 + (size_t)q0 * NHEAD * QGATE,
                                                               attn_out + (size_t)q0 * NHEAD * HDIM);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
