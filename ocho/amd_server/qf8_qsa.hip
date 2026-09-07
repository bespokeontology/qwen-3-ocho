// qf8_qsa.hip - fused all-head QSA full-attention layer for gfx906, T=1.
//
// Replaces the old per-head launch loop (24 heads x {scores, softmax, out} +
// separate norm/rope/cache kernels ~= 80 launches/layer) with:
//
//   1. fused {q[12288], k[512], v[512], idx_qk[640]} multi-GEMV (1 launch)
//   2. k8_attn_prep: q_norm + rope for all 24 q heads, k_norm + rope + KV
//      cache write for both kv heads, v cache write -- ONE launch, 27 blocks
//   3. k8_idx_step: indexer q norm+rope, pooled-key update for the current
//      4-token block, block finalization (mean -> k_layernorm -> rope at
//      block start) -- ONE 128-thread block
//   4. (only when complete blocks > 512, i.e. past the 2048-token budget --
//      host branch on pos, no device sync):
//      k8_idx_scores (q_idx . pooled keys, relu, /sqrt(128)) and
//      k8_idx_topk (single-block bitonic sort of <=2048 block scores, writes
//      a 1-byte-per-block selection mask)
//   5. k8_attn_flash: ONE kernel, 24 blocks (one per q head) x 256 threads
//      (4 waves split positions strided, fp32 online softmax, QSA block mask,
//      sigmoid out-gate in the epilogue)
//   6. o_proj GEMV
//
// Semantics per SEMANTICS.md: q/gate layout [24][512] (gate in the upper 256
// of each head row), kv repeat x12, zero-centered q/k norms over 256, partial
// rope on the first 64 dims (pairs i,i+32), scale 1/sqrt(256). Indexer:
// index_qk_proj [640,2560] -> 4 heads x 128 + token_k[128]; keys pooled per
// 4-token block (fp32 mean), k_layernorm, rope at block start; scores
// sum_h relu(q_h . k_pool)/sqrt(128); top-512 blocks; tail (incomplete)
// block always attended.
#include "qf8_dense.h"
#include "qf8_dense_i8.h"
#include <string.h>

// ---- KV element type: FP8 E4M3 storage (default) --------------------------
// Established policy: 8-bit storage with wider accumulation. The M-fold makes
// KV the dominant per-card state (M rows x ctx x 512), so KV is stored as
// E4M3 and widened to fp32 in registers at gather time. Build with
// -DQF8_KV_BF16 to fall back to the bf16 representation.
#ifndef QF8_KV_BF16
typedef uint8_t qf8_kv_t;
__device__ __forceinline__ qf8_kv_t qf8_kv_store(float v) {
    // E4M3 (bias 7), round-to-nearest-even on the mantissa, saturating.
    uint32_t u; __builtin_memcpy(&u, &v, 4);
    uint32_t sgn = (u >> 24) & 0x80u;
    int e = (int)((u >> 23) & 0xFFu) - 127;
    uint32_t m = u & 0x7FFFFFu;
    if (e < -9) return (qf8_kv_t)sgn;                 // underflow -> +-0
    if (e >= 8) return (qf8_kv_t)(sgn | 0x7Eu);       // saturate below NaN
    if (e < -6) {                                     // subnormal
        int sh = -6 - e;
        uint32_t mm = (m | 0x800000u) >> (sh + 20);
        return (qf8_kv_t)(sgn | (mm & 0x7u));
    }
    uint32_t mm = m >> 20, rem = m & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (mm & 1u))) {
        mm++;
        if (mm == 8u) { mm = 0; e++; if (e >= 8) return (qf8_kv_t)(sgn | 0x7Eu); }
    }
    return (qf8_kv_t)(sgn | (uint32_t)((e + 7) << 3) | (mm & 0x7u));
}
__device__ __forceinline__ float qf8_kv_load(qf8_kv_t b) {
    uint32_t x = (uint32_t)b, sgn = (x & 0x80u) << 24, e = (x >> 3) & 0xFu, m = x & 0x7u;
    if (e == 0) { float v = (float)m * 0.001953125f; /* 2^-9 */
                  uint32_t u; __builtin_memcpy(&u, &v, 4); u |= sgn; __builtin_memcpy(&v, &u, 4); return v; }
    uint32_t u = sgn | ((e + 120u) << 23) | (m << 20); float f; __builtin_memcpy(&f, &u, 4); return f;
}
#else
typedef uint16_t qf8_kv_t;
__device__ __forceinline__ qf8_kv_t qf8_kv_store(float v) { return qf8_f2bf(v); }
__device__ __forceinline__ float    qf8_kv_load(qf8_kv_t b) { return qf8_bf2f(b); }
#endif

// ---- (2) q/k norm + rope + cache write, all heads, one launch -------------
__global__ void k8_attn_prep(float *__restrict__ q12288,
                             float *__restrict__ k512,
                             const float *__restrict__ v512,
                             const uint16_t *__restrict__ q_norm,
                             const uint16_t *__restrict__ k_norm,
                             qf8_kv_t *__restrict__ kc,
                             qf8_kv_t *__restrict__ vc,
                             const float *__restrict__ inv_freq, long pos) {
    {   // rows form (chunk2): block row y handles position pos + y on row y of q/k/v
        const size_t ty = blockIdx.y;
        q12288 += ty * 12288; k512 += ty * (QF8_NKV * QF8_HDIM); v512 += ty * (QF8_NKV * QF8_HDIM);
        pos += (long)ty; }
    int b = blockIdx.x, lane = threadIdx.x;    // 64 lanes
    if (b < QF8_NHEAD + QF8_NKV) {
        // q head h (stride QGATE) or kv head h-NHEAD (stride HDIM)
        int is_q = b < QF8_NHEAD;
        float *base = is_q ? q12288 + (size_t)b * QF8_QGATE
                           : k512 + (size_t)(b - QF8_NHEAD) * QF8_HDIM;
        const uint16_t *wn = is_q ? q_norm : k_norm;
        float x0 = base[lane], x1 = base[lane + 64],
              x2 = base[lane + 128], x3 = base[lane + 192];
        float acc = qf8_wave_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
        acc = __shfl_sync(QF8_WMASK, acc, 0);
        float rs = rsqrtf(acc / QF8_HDIM + 1e-6f);
        base[lane]       = x0 * rs * (1.f + qf8_bf2f(wn[lane]));
        base[lane + 64]  = x1 * rs * (1.f + qf8_bf2f(wn[lane + 64]));
        base[lane + 128] = x2 * rs * (1.f + qf8_bf2f(wn[lane + 128]));
        base[lane + 192] = x3 * rs * (1.f + qf8_bf2f(wn[lane + 192]));
        __syncthreads();
        if (lane < 32) {                        // partial rope: first 64 dims
            float a = base[lane], bb = base[lane + 32];
            float f = (float)pos * inv_freq[lane];
            base[lane]      = a * cosf(f) - bb * sinf(f);
            base[lane + 32] = bb * cosf(f) + a * sinf(f);
        }
        __syncthreads();
        if (!is_q) {                            // write normed+roped k to cache
            int kh = b - QF8_NHEAD;
            #pragma unroll
            for (int j = 0; j < 4; j++)
                kc[(size_t)pos * (QF8_NKV * QF8_HDIM) + kh * QF8_HDIM + lane + 64 * j] =
                    qf8_kv_store(base[lane + 64 * j]);
        }
    } else {                                    // block 26: v cache write
        #pragma unroll
        for (int j = 0; j < 8; j++)
            vc[(size_t)pos * (QF8_NKV * QF8_HDIM) + lane + 64 * j] =
                qf8_kv_store(v512[lane + 64 * j]);
    }
}

// ---- (3) indexer per-token step -------------------------------------------
__global__ void k8_idx_step(float *__restrict__ idx640,
                            const uint16_t *__restrict__ idx_qnorm,
                            const uint16_t *__restrict__ idx_knorm,
                            const float *__restrict__ inv_freq_idx,
                            float *__restrict__ pool_sum,
                            float *__restrict__ pool_key,
                            int *__restrict__ pool_cnt, long pos) {
    __shared__ float qi[QF8_IDX_H * QF8_IDX_D];
    __shared__ float pk[QF8_IDX_D];
    __shared__ float red[2];
    __shared__ float sq[QF8_IDX_H][32];
    __shared__ int done;
    int tid = threadIdx.x;                      // 128 threads
    // idx q: zero-centered RMS norm per 128-dim head. The 4 heads live in
    // 32-lane subwaves; a shuffle tree with offsets <= 16 would still read
    // across subwave boundaries (lane 48 reads lane 64), so reduce through
    // smem instead: 4 x 32 partials, each thread re-sums its head's 32.
    int h = tid / 32, sub = tid % 32;
    float v0 = idx640[h * 128 + sub],        v1 = idx640[h * 128 + sub + 32],
          v2 = idx640[h * 128 + sub + 64],   v3 = idx640[h * 128 + sub + 96];
    sq[h][sub] = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    // token_k: add into the current block's pool (raw, pre-norm)
    int blk = (int)(pos >> 2);
    pool_sum[(size_t)blk * QF8_IDX_D + tid] += idx640[512 + tid];
    __syncthreads();
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) acc += sq[h][i];
    float rs = rsqrtf(acc / QF8_IDX_D + 1e-6f);
    qi[h * 128 + sub]      = v0 * rs * (1.f + qf8_bf2f(idx_qnorm[sub]));
    qi[h * 128 + sub + 32] = v1 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 32]));
    qi[h * 128 + sub + 64] = v2 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 64]));
    qi[h * 128 + sub + 96] = v3 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 96]));
    if (tid == 0) {
        int c = ++pool_cnt[blk];
        done = (c == QF8_IDX_C);
    }
    // rope on idx q at the current position (full 128: pairs i,i+64)
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        int t = tid + it * 128;                 // 256 pair-ops
        int hh = t >> 6, i = t & 63;
        float a = qi[hh * 128 + i], bb = qi[hh * 128 + i + 64];
        float f = (float)pos * inv_freq_idx[i];
        qi[hh * 128 + i]      = a * cosf(f) - bb * sinf(f);
        qi[hh * 128 + i + 64] = bb * cosf(f) + a * sinf(f);
    }
    __syncthreads();
    idx640[tid]       = qi[tid];                // publish normed+roped q
    idx640[256 + tid] = qi[256 + tid];
    // block finalize: mean -> k_layernorm (zero-centered) -> rope at blk*4
    if (done) {
        float mv = pool_sum[(size_t)blk * QF8_IDX_D + tid] * (1.f / QF8_IDX_C);
        pk[tid] = mv;
        float a2 = qf8_wave_sum(mv * mv);       // 128 threads = 2 waves
        int lane = tid % QF8_WAVE, wid = tid / QF8_WAVE;
        if (lane == 0) red[wid] = a2;
        __syncthreads();
        float rsk = rsqrtf((red[0] + red[1]) / QF8_IDX_D + 1e-6f);
        pk[tid] = mv * rsk * (1.f + qf8_bf2f(idx_knorm[tid]));
        __syncthreads();
        if (tid < 64) {
            float a = pk[tid], bb = pk[tid + 64];
            float f = (float)(blk * QF8_IDX_C) * inv_freq_idx[tid];
            pool_key[(size_t)blk * QF8_IDX_D + tid]      = a * cosf(f) - bb * sinf(f);
            pool_key[(size_t)blk * QF8_IDX_D + tid + 64] = bb * cosf(f) + a * sinf(f);
        }
    }
}

// ---- (4a) indexer scores over complete blocks ------------------------------
__global__ void k8_idx_scores(const float *__restrict__ idx640,
                              const float *__restrict__ pool_key,
                              float *__restrict__ blk_score, int nfull) {
    const float iscale = 1.f / sqrtf((float)QF8_IDX_D);
    for (int b = threadIdx.x; b < nfull; b += blockDim.x) {
        const float *kb = pool_key + (size_t)b * QF8_IDX_D;
        float sc = 0.f;
        #pragma unroll
        for (int h = 0; h < QF8_IDX_H; h++) {
            const float *qh = idx640 + h * QF8_IDX_D;
            float d = 0.f;
            for (int i = 0; i < QF8_IDX_D; i++) d += qh[i] * kb[i];
            sc += fmaxf(d * iscale, 0.f);       // relu
        }
        blk_score[b] = sc;
    }
}

// ---- (4b) top-512 block selection: threshold search, O(1) shared memory ----
//
// The previous implementation loaded EVERY block's (score, index) into LDS and
// bitonic-sorted all of them to answer a top-512 question. That made shared
// memory O(context):
//     ctx   8192 ->  2048 blocks ->  16 KiB   ok
//     ctx  32768 ->  8192 blocks ->  64 KiB   exactly at the gfx906 limit
//     ctx  65536 -> 16384 blocks -> 128 KiB   does not fit
//     ctx 262144 -> 65536 blocks -> 512 KiB   8x over
// so the model's 262144-token context was capped at 8192 by ONE kernel doing a
// full sort where a selection was wanted. Nothing else in the indexer scales
// badly - its HBM at 262144 is only ~193 MiB per GPU.
//
// This selects by bisecting a score threshold instead. Scores are relu'd sums,
// so they are >= 0 and bounded by the block max. f(t) = #{score > t} is
// non-increasing, so ~48 bisection steps drive lo/hi to adjacent floats around
// the TOPK-th largest value. Shared memory is a handful of scalars - CONSTANT in
// context length - and the work is O(nfull) per step with every thread
// participating.
//
// Determinism: the step count is fixed, the counting reduction is integer (so
// order does not matter), and the tie fill walks candidates in ascending block
// index. Same input -> same mask, always.
__global__ void k8_idx_topk(const float *__restrict__ blk_score,
                            uint8_t *__restrict__ mask, int nfull,
                            int *__restrict__ blk_list, int *__restrict__ nsel) {
    __shared__ float sh_lo, sh_hi;
    __shared__ int   sh_cnt;
    const int tid = threadIdx.x, nthr = blockDim.x;

    // block max -> upper bound for the search
    float mx = 0.f;
    for (int i = tid; i < nfull; i += nthr) mx = fmaxf(mx, blk_score[i]);
    #pragma unroll
    for (int off = QF8_WAVE / 2; off; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(QF8_WMASK, mx, off));
    __shared__ float red_mx[16];
    const int lane = tid % QF8_WAVE, wid = tid / QF8_WAVE;
    if (lane == 0) red_mx[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = red_mx[0];
        for (int w = 1; w < (int)(blockDim.x / QF8_WAVE); w++) m = fmaxf(m, red_mx[w]);
        sh_lo = 0.f;
        sh_hi = m;
    }
    __syncthreads();

    // bisect: invariant f(lo) >= TOPK, f(hi) <= TOPK
    for (int it = 0; it < 48; it++) {
        if (tid == 0) sh_cnt = 0;
        __syncthreads();
        const float mid = 0.5f * (sh_lo + sh_hi);
        int c = 0;
        for (int i = tid; i < nfull; i += nthr) c += (blk_score[i] > mid);
        c = (int)qf8_wave_sum((float)c);
        if (lane == 0) atomicAdd(&sh_cnt, c);
        __syncthreads();
        if (tid == 0) {
            if (sh_cnt >= QF8_IDX_TOPK) sh_lo = mid; else sh_hi = mid;
        }
        __syncthreads();
    }

    // everything strictly above hi is in; count it
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int c = 0;
    for (int i = tid; i < nfull; i += nthr) {
        uint8_t sel = (blk_score[i] > sh_hi);
        mask[i] = sel;
        c += sel;
    }
    c = (int)qf8_wave_sum((float)c);
    if (lane == 0) atomicAdd(&sh_cnt, c);
    __syncthreads();

    // fill the remainder from the tie band (lo, hi], lowest block index first
    if (tid == 0) {
        int r = QF8_IDX_TOPK - sh_cnt;
        for (int i = 0; i < nfull && r > 0; i++) {
            float v = blk_score[i];
            if (v > sh_lo && v <= sh_hi) { mask[i] = 1; r--; }
        }
    }
    __syncthreads();
    // compact selected-block list (ascending, deterministic): wave 0, ballot + rank.
    // The flash kernels walk this list (+ the tail block) instead of scanning every
    // position and testing the mask, which cost O(context) per query (34 -> 211 ms
    // per 256-position stage from 0 to 7.7K context).
    if (wid == 0) {
        int count = 0;
        for (int base = 0; base < nfull; base += QF8_WAVE) {
            const int b = base + lane;
            const int sel = (b < nfull) ? (int)mask[b] : 0;
            const unsigned long long bal = __ballot(sel);
            if (sel) blk_list[count + __popcll(bal & ((1ull << lane) - 1ull))] = b;
            count += __popcll(bal);
        }
        if (lane == 0) *nsel = count;
    }
}

// ---- (5) all-head flash-decode attention, one block per q head -------------
// 4 waves per block, wave w takes positions p = w, w+4, ...; per-wave online
// softmax state (m, l, acc[4 dims per lane]) merged across waves in smem.
// mask == NULL attends to everything (context within the 2048-token budget).
__global__ void k8_attn_flash(const float *__restrict__ q12288,
                              const qf8_kv_t *__restrict__ kc,
                              const qf8_kv_t *__restrict__ vc,
                              const uint8_t *__restrict__ mask, int nfull,
                              long ntok, float *__restrict__ attn_out,
                              const int *__restrict__ blk_list, const int *__restrict__ nsel_p) {
    const int h = blockIdx.x;
    const int kh = h / (QF8_NHEAD / QF8_NKV);
    const int wave = threadIdx.x / QF8_WAVE;
    const int lane = threadIdx.x % QF8_WAVE;
    __shared__ float qs[QF8_HDIM], gt[QF8_HDIM];
    __shared__ float accs[4][QF8_HDIM];
    __shared__ float mm[4], ll[4];
    int tid = threadIdx.x;
    qs[tid] = q12288[(size_t)h * QF8_QGATE + tid];
    gt[tid] = q12288[(size_t)h * QF8_QGATE + QF8_HDIM + tid];
    __syncthreads();

    float m = -1e30f, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const float scale = 1.f / sqrtf((float)QF8_HDIM);
    const qf8_kv_t *kb = kc + kh * QF8_HDIM;
    const qf8_kv_t *vb = vc + kh * QF8_HDIM;
    // Entries: with a selection, the listed complete blocks (4 positions each) plus the
    // tail block; otherwise the positions themselves. Bounded by the 2048-token budget.
    const int nsel = (mask && blk_list) ? *nsel_p : 0;
    const long nent = (mask && blk_list) ? (long)nsel + 1 : ntok;
    for (long ei = wave; ei < nent; ei += 4) {
        long pa, pb;
        if (mask && blk_list) { const long b = (ei < nsel) ? (long)blk_list[ei] : ((ntok - 1) >> 2); pa = b * 4; pb = pa + 4; if (pb > ntok) pb = ntok; }
        else { pa = ei; pb = ei + 1; }
        for (long p = pa; p < pb; p++) {
        const qf8_kv_t *kr = kb + (size_t)p * (QF8_NKV * QF8_HDIM);
        float dot = qs[lane] * qf8_kv_load(kr[lane])
                  + qs[lane + 64] * qf8_kv_load(kr[lane + 64])
                  + qs[lane + 128] * qf8_kv_load(kr[lane + 128])
                  + qs[lane + 192] * qf8_kv_load(kr[lane + 192]);
        dot = qf8_wave_sum(dot);
        float sc = __shfl_sync(QF8_WMASK, dot, 0) * scale;
        float mn = fmaxf(m, sc);
        // expf, NOT __expf. The reference softmax (src/cuda/qf.cu k_softmax)
        // uses full-precision expf; __expf is a hardware approximation whose
        // error lands directly on the attention weights of all 24 heads, in
        // every attention layer, and accumulates through the residual. It is a
        // silent accuracy-for-speed trade in the one place we cannot afford
        // one. (GDN keeps __expf because the reference uses __expf there too.)
        float corr = expf(m - mn);              // 0 on first (m=-1e30, mn=sc)
        float e = expf(sc - mn);
        const qf8_kv_t *vr = vb + (size_t)p * (QF8_NKV * QF8_HDIM);
        a0 = a0 * corr + e * qf8_kv_load(vr[lane]);
        a1 = a1 * corr + e * qf8_kv_load(vr[lane + 64]);
        a2 = a2 * corr + e * qf8_kv_load(vr[lane + 128]);
        a3 = a3 * corr + e * qf8_kv_load(vr[lane + 192]);
        l = l * corr + e;
        m = mn;
        }
    }
    // cross-wave merge
    accs[wave][lane]       = a0;
    accs[wave][lane + 64]  = a1;
    accs[wave][lane + 128] = a2;
    accs[wave][lane + 192] = a3;
    if (lane == 0) { mm[wave] = m; ll[wave] = l; }
    __syncthreads();
    if (wave == 0) {
        float M = fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3]));
        float c0 = expf(mm[0] - M), c1 = expf(mm[1] - M),
              c2 = expf(mm[2] - M), c3 = expf(mm[3] - M);
        float lt = ll[0] * c0 + ll[1] * c1 + ll[2] * c2 + ll[3] * c3 + 1e-9f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int d = lane + 64 * j;
            float o = (accs[0][d] * c0 + accs[1][d] * c1 +
                       accs[2][d] * c2 + accs[3][d] * c3) / lt;
            attn_out[h * QF8_HDIM + d] = o * qf8_sig(gt[d]);
        }
    }
}

// ---- (5b) split-position flash decode: grid (head, chunk) + merge ----------
// The one-block-per-head kernel above walks every position with 4 waves, so at
// a 1000-token context each wave serializes ~250 dependent K-row loads. Here
// QF8_FLASH_SPLIT chunks of positions run as independent blocks (8x the waves
// in flight) and write an unnormalized online-softmax partial (m, l, acc[256])
// per (head, chunk); a 24-block merge normalizes and applies the head gate.
#define QF8_FLASH_SPLIT 8
#define QF8_FLASH_PART  (QF8_HDIM + 2)
__global__ void __launch_bounds__(256) k8_attn_flash_split(const float *__restrict__ q12288,
                              const qf8_kv_t *__restrict__ kc,
                              const qf8_kv_t *__restrict__ vc,
                              const uint8_t *__restrict__ mask, int nfull,
                              long ntok, float *__restrict__ part,
                              const int *__restrict__ blk_list, const int *__restrict__ nsel_p) {
    const int h = blockIdx.x, chunk = blockIdx.y;
    const int kh = h / (QF8_NHEAD / QF8_NKV);
    const int wave = threadIdx.x / QF8_WAVE;
    const int lane = threadIdx.x % QF8_WAVE;
    __shared__ float qs[QF8_HDIM];
    __shared__ float accs[4][QF8_HDIM];
    __shared__ float mm[4], ll[4];
    const int tid = threadIdx.x;
    qs[tid] = q12288[(size_t)h * QF8_QGATE + tid];
    __syncthreads();
    const int nsel = (mask && blk_list) ? *nsel_p : 0;
    const long nent = (mask && blk_list) ? (long)nsel + 1 : ntok;   // list entries (+ tail block) or positions
    const long per = (nent + QF8_FLASH_SPLIT - 1) / QF8_FLASH_SPLIT;
    const long e0 = (long)chunk * per;
    long e1 = e0 + per; if (e1 > nent) e1 = nent;
    float m = -1e30f, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const float scale = 1.f / sqrtf((float)QF8_HDIM);
    const qf8_kv_t *kb = kc + kh * QF8_HDIM;
    const qf8_kv_t *vb = vc + kh * QF8_HDIM;
    for (long ei = e0 + wave; ei < e1; ei += 4) {
        long pa, pb;
        if (mask && blk_list) { const long b = (ei < nsel) ? (long)blk_list[ei] : ((ntok - 1) >> 2); pa = b * 4; pb = pa + 4; if (pb > ntok) pb = ntok; }
        else { pa = ei; pb = ei + 1; }
        for (long p = pa; p < pb; p++) {
        const qf8_kv_t *kr = kb + (size_t)p * (QF8_NKV * QF8_HDIM);
        const qf8_kv_t *vr = vb + (size_t)p * (QF8_NKV * QF8_HDIM);
        const float v0 = qf8_kv_load(vr[lane]), v1 = qf8_kv_load(vr[lane + 64]),
                    v2 = qf8_kv_load(vr[lane + 128]), v3 = qf8_kv_load(vr[lane + 192]);
        float dot = qs[lane] * qf8_kv_load(kr[lane])
                  + qs[lane + 64] * qf8_kv_load(kr[lane + 64])
                  + qs[lane + 128] * qf8_kv_load(kr[lane + 128])
                  + qs[lane + 192] * qf8_kv_load(kr[lane + 192]);
        dot = qf8_wave_sum(dot);
        float sc = __shfl_sync(QF8_WMASK, dot, 0) * scale;
        float mn = fmaxf(m, sc);
        float corr = expf(m - mn);
        float e = expf(sc - mn);
        a0 = a0 * corr + e * v0; a1 = a1 * corr + e * v1;
        a2 = a2 * corr + e * v2; a3 = a3 * corr + e * v3;
        l = l * corr + e;
        m = mn;
        }
    }
    accs[wave][lane] = a0; accs[wave][lane + 64] = a1;
    accs[wave][lane + 128] = a2; accs[wave][lane + 192] = a3;
    if (lane == 0) { mm[wave] = m; ll[wave] = l; }
    __syncthreads();
    float *pp = part + ((size_t)h * QF8_FLASH_SPLIT + chunk) * QF8_FLASH_PART;
    if (wave == 0) {
        float M = fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3]));
        float c0 = expf(mm[0] - M), c1 = expf(mm[1] - M), c2 = expf(mm[2] - M), c3 = expf(mm[3] - M);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int d = lane + 64 * j;
            pp[2 + d] = accs[0][d] * c0 + accs[1][d] * c1 + accs[2][d] * c2 + accs[3][d] * c3;
        }
        if (lane == 0) { pp[0] = M; pp[1] = ll[0] * c0 + ll[1] * c1 + ll[2] * c2 + ll[3] * c3; }
    }
}
__global__ void __launch_bounds__(256) k8_attn_flash_merge(const float *__restrict__ part,
                              const float *__restrict__ q12288, float *__restrict__ attn_out) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float *pp = part + (size_t)h * QF8_FLASH_SPLIT * QF8_FLASH_PART;
    float M = -1e30f;
    #pragma unroll
    for (int c = 0; c < QF8_FLASH_SPLIT; c++) M = fmaxf(M, pp[c * QF8_FLASH_PART]);
    float lt = 1e-9f, o = 0.f;
    #pragma unroll
    for (int c = 0; c < QF8_FLASH_SPLIT; c++) {
        const float w = expf(pp[c * QF8_FLASH_PART] - M);
        lt += pp[c * QF8_FLASH_PART + 1] * w;
        o  += pp[c * QF8_FLASH_PART + 2 + d] * w;
    }
    const float gt = q12288[(size_t)h * QF8_QGATE + QF8_HDIM + d];
    attn_out[h * QF8_HDIM + d] = (o / lt) * qf8_sig(gt);
}

// =============================================================================
// CHUNK2 (prefill): QSA STAGE -> ALL POSITIONS. The per-position decode form above
// is untouched. Every stage below is ONE launch over the T rows of the chunk:
//   int8 rows projections -> prep rows -> idx q rows -> pool blocks -> score rows
//   -> top-k rows (mask + compact list per row) -> split flash rows -> merge rows
//   -> int8 wo rows.
// Causality is a per-row bound, not launch order: row t (position pos0+t) scores and
// selects only its own nfull_t = (pos0+t+1)>>2 complete blocks (every token of those
// blocks is <= its position) and its tail walk stops at its own position. Pool keys
// finalized "early" by the pool launch (blocks completed by later rows of this chunk)
// sit at indices >= nfull_t of every earlier row and are never read by it.
// =============================================================================

// idx q rows: zero-centered RMS norm + rope of the 4 indexer q heads, one block per
// row (128 threads). Reads/writes idx640[0..511] of the row only; token_k (512..639)
// is left raw for the pool launch.
__global__ void k8_idx_q_rows(float *__restrict__ idx640, const uint16_t *__restrict__ idx_qnorm,
                              const float *__restrict__ inv_freq_idx, long pos0) {
    __shared__ float qi[QF8_IDX_H * QF8_IDX_D];
    __shared__ float sq[QF8_IDX_H][32];
    const int tid = threadIdx.x;                        // 128
    const long pos = pos0 + blockIdx.x;
    idx640 += (size_t)blockIdx.x * 640;
    const int h = tid / 32, sub = tid % 32;
    const float v0 = idx640[h * 128 + sub],      v1 = idx640[h * 128 + sub + 32],
                v2 = idx640[h * 128 + sub + 64], v3 = idx640[h * 128 + sub + 96];
    sq[h][sub] = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    __syncthreads();
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) acc += sq[h][i];
    const float rs = rsqrtf(acc / QF8_IDX_D + 1e-6f);
    qi[h * 128 + sub]      = v0 * rs * (1.f + qf8_bf2f(idx_qnorm[sub]));
    qi[h * 128 + sub + 32] = v1 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 32]));
    qi[h * 128 + sub + 64] = v2 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 64]));
    qi[h * 128 + sub + 96] = v3 * rs * (1.f + qf8_bf2f(idx_qnorm[sub + 96]));
    __syncthreads();                                    // rope pairs cross the two waves
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int t = tid + it * 128, hh = t >> 6, i = t & 63;
        const float a = qi[hh * 128 + i], bb = qi[hh * 128 + i + 64];
        const float f = (float)pos * inv_freq_idx[i];
        qi[hh * 128 + i]      = a * cosf(f) - bb * sinf(f);
        qi[hh * 128 + i + 64] = bb * cosf(f) + a * sinf(f);
    }
    __syncthreads();
    idx640[tid] = qi[tid]; idx640[256 + tid] = qi[256 + tid];
}

// pool blocks: one block per 4-token pool touched by the chunk. Adds the chunk's
// positions of that pool into pool_sum in position order (raw token_k), advances the
// count, and finalizes (mean -> zero-centered k-layernorm -> rope at the block start)
// when the pool completes. Identical arithmetic to the decode step, one launch.
__global__ void k8_idx_pool_rows(const float *__restrict__ idx640, int T, long pos0,
                                 const uint16_t *__restrict__ idx_knorm,
                                 const float *__restrict__ inv_freq_idx,
                                 float *__restrict__ pool_sum, float *__restrict__ pool_key,
                                 int *__restrict__ pool_cnt) {
    __shared__ float red[2];
    __shared__ float pk[QF8_IDX_D];
    const int tid = threadIdx.x;                        // 128 = 2 waves
    const long blk = (pos0 >> 2) + blockIdx.x;
    long p0 = blk * QF8_IDX_C, p1 = p0 + QF8_IDX_C;
    if (p0 < pos0) p0 = pos0;
    if (p1 > pos0 + T) p1 = pos0 + T;
    float sum = pool_sum[(size_t)blk * QF8_IDX_D + tid];
    const int cprev = pool_cnt[blk];
    for (long p = p0; p < p1; p++) sum += idx640[(size_t)(p - pos0) * 640 + 512 + tid];
    pool_sum[(size_t)blk * QF8_IDX_D + tid] = sum;
    const int cnt = cprev + (int)(p1 - p0);
    __syncthreads();
    if (tid == 0) pool_cnt[blk] = cnt;
    if (cnt != QF8_IDX_C) return;                       // uniform across the block
    const float mv = sum * (1.f / QF8_IDX_C);
    const float a2 = qf8_wave_sum(mv * mv);
    const int lane = tid % QF8_WAVE, wid = tid / QF8_WAVE;
    if (lane == 0) red[wid] = a2;
    __syncthreads();
    const float rsk = rsqrtf((red[0] + red[1]) / QF8_IDX_D + 1e-6f);
    pk[tid] = mv * rsk * (1.f + qf8_bf2f(idx_knorm[tid]));
    __syncthreads();
    if (tid < 64) {
        const float a = pk[tid], bb = pk[tid + 64];
        const float f = (float)(blk * QF8_IDX_C) * inv_freq_idx[tid];
        pool_key[(size_t)blk * QF8_IDX_D + tid]      = a * cosf(f) - bb * sinf(f);
        pool_key[(size_t)blk * QF8_IDX_D + tid + 64] = bb * cosf(f) + a * sinf(f);
    }
}

// score rows: one block per row; row t scores its nfull_t complete blocks into
// blk_score[t][0..nfull_t). Rows with nfull_t <= TOPK are dense and skip.
__global__ void k8_idx_scores_rows(const float *__restrict__ idx640,
                                   const float *__restrict__ pool_key,
                                   float *__restrict__ blk_score, int nb_max, long pos0) {
    __shared__ float qs[QF8_IDX_H * QF8_IDX_D];
    const int t = blockIdx.x;
    const int nfull = (int)((pos0 + t + 1) >> 2);
    if (nfull <= QF8_IDX_TOPK) return;
    const float *q = idx640 + (size_t)t * 640;
    for (int i = threadIdx.x; i < QF8_IDX_H * QF8_IDX_D; i += blockDim.x) qs[i] = q[i];
    __syncthreads();
    float *out = blk_score + (size_t)t * nb_max;
    const float iscale = 1.f / sqrtf((float)QF8_IDX_D);
    for (int b = threadIdx.x; b < nfull; b += blockDim.x) {
        const float *kb = pool_key + (size_t)b * QF8_IDX_D;
        float sc = 0.f;
        #pragma unroll
        for (int h = 0; h < QF8_IDX_H; h++) {
            const float *qh = qs + h * QF8_IDX_D;
            float d = 0.f;
            for (int i = 0; i < QF8_IDX_D; i++) d += qh[i] * kb[i];
            sc += fmaxf(d * iscale, 0.f);
        }
        out[b] = sc;
    }
}

// top-k rows: one block per row, the same 48-step threshold bisection as the decode
// selector over that row's scores; writes the row's mask, ascending compact list and
// count. Dense rows (nfull_t <= TOPK) get nsel = -1 and no list.
__global__ void k8_idx_topk_rows(const float *__restrict__ blk_score_rows,
                                 uint8_t *__restrict__ mask_rows, int nb_max, long pos0,
                                 int *__restrict__ list_rows, int *__restrict__ nsel_rows) {
    __shared__ float sh_lo, sh_hi;
    __shared__ int   sh_cnt;
    __shared__ float red_mx[16];
    const int t = blockIdx.x;
    const int nfull = (int)((pos0 + t + 1) >> 2);
    const int tid = threadIdx.x, nthr = blockDim.x;
    if (nfull <= QF8_IDX_TOPK) { if (tid == 0) nsel_rows[t] = -1; return; }
    const float *blk_score = blk_score_rows + (size_t)t * nb_max;
    uint8_t *mask = mask_rows + (size_t)t * nb_max;
    int *blk_list = list_rows + (size_t)t * (QF8_IDX_TOPK + 1);
    float mx = 0.f;
    for (int i = tid; i < nfull; i += nthr) mx = fmaxf(mx, blk_score[i]);
    #pragma unroll
    for (int off = QF8_WAVE / 2; off; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(QF8_WMASK, mx, off));
    const int lane = tid % QF8_WAVE, wid = tid / QF8_WAVE;
    if (lane == 0) red_mx[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = red_mx[0];
        for (int w = 1; w < (int)(blockDim.x / QF8_WAVE); w++) m = fmaxf(m, red_mx[w]);
        sh_lo = 0.f; sh_hi = m;
    }
    __syncthreads();
    for (int it = 0; it < 48; it++) {
        if (tid == 0) sh_cnt = 0;
        __syncthreads();
        const float mid = 0.5f * (sh_lo + sh_hi);
        int c = 0;
        for (int i = tid; i < nfull; i += nthr) c += (blk_score[i] > mid);
        c = (int)qf8_wave_sum((float)c);
        if (lane == 0) atomicAdd(&sh_cnt, c);
        __syncthreads();
        if (tid == 0) { if (sh_cnt >= QF8_IDX_TOPK) sh_lo = mid; else sh_hi = mid; }
        __syncthreads();
    }
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int c = 0;
    for (int i = tid; i < nfull; i += nthr) {
        const uint8_t sel = (blk_score[i] > sh_hi);
        mask[i] = sel;
        c += sel;
    }
    c = (int)qf8_wave_sum((float)c);
    if (lane == 0) atomicAdd(&sh_cnt, c);
    __syncthreads();
    if (tid == 0) {
        int r = QF8_IDX_TOPK - sh_cnt;
        for (int i = 0; i < nfull && r > 0; i++) {
            const float v = blk_score[i];
            if (v > sh_lo && v <= sh_hi) { mask[i] = 1; r--; }
        }
    }
    __syncthreads();
    if (wid == 0) {
        int count = 0;
        for (int base = 0; base < nfull; base += QF8_WAVE) {
            const int b = base + lane;
            const int sel = (b < nfull) ? (int)mask[b] : 0;
            const unsigned long long bal = __ballot(sel);
            if (sel) blk_list[count + __popcll(bal & ((1ull << lane) - 1ull))] = b;
            count += __popcll(bal);
        }
        if (lane == 0) nsel_rows[t] = count;
    }
}

// split flash rows: grid (head, split, row). Row t attends over its own compact list
// (+ its tail block, cut at its own position) or, for a dense row, over all of its
// positions. Partials go to part_rows[t][head][split].
__global__ void __launch_bounds__(256) k8_attn_flash_split_rows(const float *__restrict__ q12288,
                              const qf8_kv_t *__restrict__ kc,
                              const qf8_kv_t *__restrict__ vc,
                              long pos0, float *__restrict__ part_rows,
                              const int *__restrict__ list_rows, const int *__restrict__ nsel_rows) {
    const int t = blockIdx.z;
    const int h = blockIdx.x, chunk = blockIdx.y;
    const long ntok = pos0 + t + 1;
    const int nsel = nsel_rows[t];
    const int use_list = nsel >= 0;
    const int *blk_list = list_rows + (size_t)t * (QF8_IDX_TOPK + 1);
    const float *q = q12288 + (size_t)t * 12288;
    float *part = part_rows + (size_t)t * QF8_NHEAD * QF8_FLASH_SPLIT * QF8_FLASH_PART;
    const int kh = h / (QF8_NHEAD / QF8_NKV);
    const int wave = threadIdx.x / QF8_WAVE;
    const int lane = threadIdx.x % QF8_WAVE;
    __shared__ float qs[QF8_HDIM];
    __shared__ float accs[4][QF8_HDIM];
    __shared__ float mm[4], ll[4];
    const int tid = threadIdx.x;
    qs[tid] = q[(size_t)h * QF8_QGATE + tid];
    __syncthreads();
    const long nent = use_list ? (long)nsel + 1 : ntok;
    const long per = (nent + QF8_FLASH_SPLIT - 1) / QF8_FLASH_SPLIT;
    const long e0 = (long)chunk * per;
    long e1 = e0 + per; if (e1 > nent) e1 = nent;
    float m = -1e30f, l = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const float scale = 1.f / sqrtf((float)QF8_HDIM);
    const qf8_kv_t *kb = kc + kh * QF8_HDIM;
    const qf8_kv_t *vb = vc + kh * QF8_HDIM;
    for (long ei = e0 + wave; ei < e1; ei += 4) {
        long pa, pb;
        if (use_list) { const long b = (ei < nsel) ? (long)blk_list[ei] : ((ntok - 1) >> 2); pa = b * 4; pb = pa + 4; if (pb > ntok) pb = ntok; }
        else { pa = ei; pb = ei + 1; }
        for (long p = pa; p < pb; p++) {
        const qf8_kv_t *kr = kb + (size_t)p * (QF8_NKV * QF8_HDIM);
        const qf8_kv_t *vr = vb + (size_t)p * (QF8_NKV * QF8_HDIM);
        const float v0 = qf8_kv_load(vr[lane]), v1 = qf8_kv_load(vr[lane + 64]),
                    v2 = qf8_kv_load(vr[lane + 128]), v3 = qf8_kv_load(vr[lane + 192]);
        float dot = qs[lane] * qf8_kv_load(kr[lane])
                  + qs[lane + 64] * qf8_kv_load(kr[lane + 64])
                  + qs[lane + 128] * qf8_kv_load(kr[lane + 128])
                  + qs[lane + 192] * qf8_kv_load(kr[lane + 192]);
        dot = qf8_wave_sum(dot);
        const float sc = __shfl_sync(QF8_WMASK, dot, 0) * scale;
        const float mn = fmaxf(m, sc);
        const float corr = expf(m - mn);
        const float e = expf(sc - mn);
        a0 = a0 * corr + e * v0; a1 = a1 * corr + e * v1;
        a2 = a2 * corr + e * v2; a3 = a3 * corr + e * v3;
        l = l * corr + e;
        m = mn;
        }
    }
    accs[wave][lane] = a0; accs[wave][lane + 64] = a1;
    accs[wave][lane + 128] = a2; accs[wave][lane + 192] = a3;
    if (lane == 0) { mm[wave] = m; ll[wave] = l; }
    __syncthreads();
    float *pp = part + ((size_t)h * QF8_FLASH_SPLIT + chunk) * QF8_FLASH_PART;
    if (wave == 0) {
        const float M = fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3]));
        const float c0 = expf(mm[0] - M), c1 = expf(mm[1] - M), c2 = expf(mm[2] - M), c3 = expf(mm[3] - M);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = lane + 64 * j;
            pp[2 + d] = accs[0][d] * c0 + accs[1][d] * c1 + accs[2][d] * c2 + accs[3][d] * c3;
        }
        if (lane == 0) { pp[0] = M; pp[1] = ll[0] * c0 + ll[1] * c1 + ll[2] * c2 + ll[3] * c3; }
    }
}

// merge rows: grid (head, row)
__global__ void __launch_bounds__(256) k8_attn_flash_merge_rows(const float *__restrict__ part_rows,
                              const float *__restrict__ q12288, float *__restrict__ attn_out) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float *pp = part_rows + (size_t)t * QF8_NHEAD * QF8_FLASH_SPLIT * QF8_FLASH_PART
                    + (size_t)h * QF8_FLASH_SPLIT * QF8_FLASH_PART;
    float M = -1e30f;
    #pragma unroll
    for (int c = 0; c < QF8_FLASH_SPLIT; c++) M = fmaxf(M, pp[c * QF8_FLASH_PART]);
    float lt = 1e-9f, o = 0.f;
    #pragma unroll
    for (int c = 0; c < QF8_FLASH_SPLIT; c++) {
        const float w = expf(pp[c * QF8_FLASH_PART] - M);
        lt += pp[c * QF8_FLASH_PART + 1] * w;
        o  += pp[c * QF8_FLASH_PART + 2 + d] * w;
    }
    const float gt = q12288[(size_t)t * 12288 + (size_t)h * QF8_QGATE + QF8_HDIM + d];
    attn_out[(size_t)t * QF8_NHEAD * QF8_HDIM + h * QF8_HDIM + d] = (o / lt) * qf8_sig(gt);
}

// ---- state + host wrappers --------------------------------------------------
// Sized by the RUNTIME context, not QF8_MAXPOS: at 8192 this is 2 MiB per QSA
// layer, at 262144 it is 64 MiB. Allocating the compile-time maximum would
// waste 190 MiB per GPU whenever the server runs at a smaller context.
int qf8_qsa_idx_init(Qf8QsaIdx *ix, long maxpos) {
    if (maxpos < QF8_IDX_C) maxpos = QF8_IDX_C;
    if (maxpos > QF8_MAXPOS) maxpos = QF8_MAXPOS;
    size_t nb = (size_t)maxpos / QF8_IDX_C;
    if (hipMalloc((void **)&ix->pool_sum, nb * QF8_IDX_D * 4) != hipSuccess ||
        hipMalloc((void **)&ix->pool_key, nb * QF8_IDX_D * 4) != hipSuccess ||
        hipMalloc((void **)&ix->pool_cnt, nb * 4) != hipSuccess ||
        hipMalloc((void **)&ix->blk_score, nb * 4) != hipSuccess ||
        hipMalloc((void **)&ix->blk_mask, nb) != hipSuccess ||
        hipMalloc((void **)&ix->blk_list, (nb + 1) * sizeof(int)) != hipSuccess ||
        hipMalloc((void **)&ix->nsel, sizeof(int)) != hipSuccess) return -1;
    if (hipMemset(ix->pool_sum, 0, nb * QF8_IDX_D * 4) != hipSuccess ||
        hipMemset(ix->pool_cnt, 0, nb * 4) != hipSuccess) return -1;
    return 0;
}
void qf8_qsa_idx_free(Qf8QsaIdx *ix) {
    if (!ix) return;
    if (ix->pool_sum) hipFree(ix->pool_sum);
    if (ix->pool_key) hipFree(ix->pool_key);
    if (ix->pool_cnt) hipFree(ix->pool_cnt);
    if (ix->blk_score) hipFree(ix->blk_score);
    if (ix->blk_mask) hipFree(ix->blk_mask);
    if (ix->blk_list) hipFree(ix->blk_list);
    if (ix->nsel) hipFree(ix->nsel);
    ix->pool_sum = ix->pool_key = ix->blk_score = 0;
    ix->pool_cnt = 0; ix->blk_mask = 0;
}

void qf8_qsa_mixer(const Qf8AttnW *w, const float *mixed,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq,
                   const float *inv_freq_idx, float *y2560, hipStream_t s) {
    Qf8GemvMulti m;
    m.W[0] = w->wq;     m.y[0] = q12288; m.rows[0] = QF8_NHEAD * QF8_QGATE;
    m.W[1] = w->wk;     m.y[1] = k512;   m.rows[1] = QF8_NKV * QF8_HDIM;
    m.W[2] = w->wv;     m.y[2] = v512;   m.rows[2] = QF8_NKV * QF8_HDIM;
    m.W[3] = w->idx_qk; m.y[3] = idx640; m.rows[3] = 640;
    m.n = 4;
    qf8_gemv_multi(&m, mixed, QF8_NEMBD, s);
    QF8_LAUNCH(k8_attn_prep, QF8_NHEAD + QF8_NKV + 1, QF8_WAVE, 0, s,
               q12288, k512, v512, w->q_norm, w->k_norm, kc, vc, inv_freq, pos);
    QF8_LAUNCH(k8_idx_step, 1, 128, 0, s, idx640, w->idx_qnorm, w->idx_knorm,
               inv_freq_idx, ix->pool_sum, ix->pool_key, ix->pool_cnt, pos);
    long ntok = pos + 1;
    int nfull = (int)(ntok >> 2);               // complete 4-token blocks
    const uint8_t *mask = 0;
    if (nfull > QF8_IDX_TOPK) {                 // budget exceeded: select
        QF8_LAUNCH(k8_idx_scores, 1, 256, 0, s, idx640, ix->pool_key,
                   ix->blk_score, nfull);
        QF8_LAUNCH(k8_idx_topk, 1, 1024, 0, s, ix->blk_score, ix->blk_mask, nfull, ix->blk_list, ix->nsel);
        mask = ix->blk_mask;
    }
    QF8_LAUNCH(k8_attn_flash, QF8_NHEAD, 256, 0, s,
               q12288, kc, vc, mask, nfull, ntok, attn_out, ix->blk_list, ix->nsel);
    qf8_gemv_bf16(w->wo, attn_out, y2560, QF8_NEMBD, QF8_NHEAD * QF8_HDIM,
                  1.f, 0.f, s);
}


// ---- M=1 int8 twin of qf8_qsa_mixer: the four input projections and the
// output projection on the int8 copies (X8 activations, dot4); the attention
// kernels, KV write, indexer and selection are the same launches as above.
void qf8_qsa_mixer_i8(const Qf8AttnW *w, const Qf8I8Mat *iq, const Qf8I8Mat *ik,
                      const Qf8I8Mat *iv, const Qf8I8Mat *iidx, const Qf8I8Mat *io,
                      const float *mixed, int8_t *x8q, float *x8s, int8_t *a8q, float *a8s,
                      qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos,
                      float *q12288, float *k512, float *v512, float *idx640,
                      float *attn_out, const float *inv_freq,
                      const float *inv_freq_idx, float *y2560, int quant_x, float *attn_part,
                      hipStream_t s) {
    if (quant_x) qf8_i8_x8(mixed, x8q, x8s, QF8_NEMBD, s);   // else x8q/x8s already hold X8(mixed)
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 4;
    m.m[0] = iq;   m.y[0] = q12288;
    m.m[1] = ik;   m.y[1] = k512;
    m.m[2] = iv;   m.y[2] = v512;
    m.m[3] = iidx; m.y[3] = idx640;
    qf8_i8_gemv_multi(&m, x8q, x8s, s);
    QF8_LAUNCH(k8_attn_prep, QF8_NHEAD + QF8_NKV + 1, QF8_WAVE, 0, s,
               q12288, k512, v512, w->q_norm, w->k_norm, kc, vc, inv_freq, pos);
    QF8_LAUNCH(k8_idx_step, 1, 128, 0, s, idx640, w->idx_qnorm, w->idx_knorm,
               inv_freq_idx, ix->pool_sum, ix->pool_key, ix->pool_cnt, pos);
    long ntok = pos + 1;
    int nfull = (int)(ntok >> 2);
    const uint8_t *mask = 0;
    if (nfull > QF8_IDX_TOPK) {
        QF8_LAUNCH(k8_idx_scores, 1, 256, 0, s, idx640, ix->pool_key,
                   ix->blk_score, nfull);
        QF8_LAUNCH(k8_idx_topk, 1, 1024, 0, s, ix->blk_score, ix->blk_mask, nfull, ix->blk_list, ix->nsel);
        mask = ix->blk_mask;
    }
    if (attn_part) {
        QF8_LAUNCH(k8_attn_flash_split, dim3(QF8_NHEAD, QF8_FLASH_SPLIT), 256, 0, s,
                   q12288, kc, vc, mask, nfull, ntok, attn_part, ix->blk_list, ix->nsel);
        QF8_LAUNCH(k8_attn_flash_merge, QF8_NHEAD, 256, 0, s, attn_part, q12288, attn_out);
    } else
    QF8_LAUNCH(k8_attn_flash, QF8_NHEAD, 256, 0, s,
               q12288, kc, vc, mask, nfull, ntok, attn_out, ix->blk_list, ix->nsel);
    qf8_i8_x8(attn_out, a8q, a8s, QF8_NHEAD * QF8_HDIM, s);        // 6144 = 24 x 256
    Qf8I8Multi mo; memset(&mo, 0, sizeof mo); mo.n = 1; mo.m[0] = io; mo.y[0] = y2560;
    qf8_i8_gemv_multi(&mo, a8q, a8s, s);
}


// ---- CHUNK form: T positions of one sequence. The q/k/v/indexer projections and
// the output projection run over the T rows (bf16 M-row GEMVs in groups of 8 =
// QF8_M_MAX accumulator rows); the KV write, indexer pooling/selection and the
// split flash decode run per position in order on this row's caches.
void qf8_qsa_chunk(const Qf8AttnW *w, const float *mixed, int T,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos0,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq,
                   const float *inv_freq_idx, float *y2560, float *attn_part, hipStream_t s) {
    for (int r0 = 0; r0 < T; r0 += 8) {
        const int nr = (T - r0) < 8 ? (T - r0) : 8;
        Qf8GemvMulti m;
        m.W[0] = w->wq;     m.y[0] = q12288 + (size_t)r0 * 12288;          m.rows[0] = QF8_NHEAD * QF8_QGATE;
        m.W[1] = w->wk;     m.y[1] = k512 + (size_t)r0 * QF8_NKV * QF8_HDIM; m.rows[1] = QF8_NKV * QF8_HDIM;
        m.W[2] = w->wv;     m.y[2] = v512 + (size_t)r0 * QF8_NKV * QF8_HDIM; m.rows[2] = QF8_NKV * QF8_HDIM;
        m.W[3] = w->idx_qk; m.y[3] = idx640 + (size_t)r0 * 640;             m.rows[3] = 640;
        m.n = 4;
        qf8_gemv_multi_M(&m, mixed + (size_t)r0 * QF8_NEMBD, QF8_NEMBD, nr, s);
    }
    for (int t = 0; t < T; t++) {
        const long pos = pos0 + t;
        float *q = q12288 + (size_t)t * 12288, *k = k512 + (size_t)t * QF8_NKV * QF8_HDIM,
              *v = v512 + (size_t)t * QF8_NKV * QF8_HDIM, *ix640 = idx640 + (size_t)t * 640,
              *ao = attn_out + (size_t)t * QF8_NHEAD * QF8_HDIM;
        QF8_LAUNCH(k8_attn_prep, QF8_NHEAD + QF8_NKV + 1, QF8_WAVE, 0, s,
                   q, k, v, w->q_norm, w->k_norm, kc, vc, inv_freq, pos);
        QF8_LAUNCH(k8_idx_step, 1, 128, 0, s, ix640, w->idx_qnorm, w->idx_knorm,
                   inv_freq_idx, ix->pool_sum, ix->pool_key, ix->pool_cnt, pos);
        long ntok = pos + 1;
        int nfull = (int)(ntok >> 2);
        const uint8_t *mask = 0;
        if (nfull > QF8_IDX_TOPK) {
            QF8_LAUNCH(k8_idx_scores, 1, 256, 0, s, ix640, ix->pool_key, ix->blk_score, nfull);
            QF8_LAUNCH(k8_idx_topk, 1, 1024, 0, s, ix->blk_score, ix->blk_mask, nfull, ix->blk_list, ix->nsel);
            mask = ix->blk_mask;
        }
        if (attn_part) {
            QF8_LAUNCH(k8_attn_flash_split, dim3(QF8_NHEAD, QF8_FLASH_SPLIT), 256, 0, s,
                       q, kc, vc, mask, nfull, ntok, attn_part, ix->blk_list, ix->nsel);
            QF8_LAUNCH(k8_attn_flash_merge, QF8_NHEAD, 256, 0, s, attn_part, q, ao);
        } else
        QF8_LAUNCH(k8_attn_flash, QF8_NHEAD, 256, 0, s, q, kc, vc, mask, nfull, ntok, ao, ix->blk_list, ix->nsel);
    }
    for (int r0 = 0; r0 < T; r0 += 8) {
        const int nr = (T - r0) < 8 ? (T - r0) : 8;
        qf8_gemv_bf16_M(w->wo, attn_out + (size_t)r0 * QF8_NHEAD * QF8_HDIM, y2560 + (size_t)r0 * QF8_NEMBD,
                        QF8_NEMBD, QF8_NHEAD * QF8_HDIM, nr, 1.f, 0.f, s);
    }
}


// =============================================================================
// UNION-TILE QSA (prefill bulk form, gfx906): one wave per (8 consecutive queries x 4 heads of one
// kv head). The 32 (query, head) pairs share the tile's UNION of selected 4-token blocks (built
// from the per-row selection masks + each row's own tail block; dense rows contribute all their
// complete blocks), so every K/V block is read from HBM ONCE per wave instead of once per
// (query, head, split). Two lanes per pair split the 256 dims: q lives in registers as packed
// halves (dot2 with the K tile in LDS, one xor-shuffle completes the dot), the online softmax and
// the 128-dim output half accumulate in registers; V is read from LDS broadcast. Exactness: a pair
// takes key p of block b iff p <= its position and (b selected for its row or b == its tail block),
// which is the split-flash kernel's key set; masked keys get -inf before the softmax.
// =============================================================================
#include <hip/hip_fp16.h>
#define QU_QT 8
#define QU_MAXW 128                                  // 4096 blocks = 16384-token context per row mask
__global__ void __launch_bounds__(128) k8_attn_union8(const float *__restrict__ q12288, const qf8_kv_t *__restrict__ kc,
        const qf8_kv_t *__restrict__ vc, const uint8_t *__restrict__ mask_rows, int nb_max,
        const int *__restrict__ nsel_rows, long pos0, int T, float *__restrict__ attn_out) {
    // 128 threads = 32 (query, head) pairs x 4 lanes; lane quarter q4 owns dims [q4*64, q4*64+64)
    const int tile = blockIdx.x, kh = blockIdx.y, hg = blockIdx.z;
    const int tid = threadIdx.x, pair = tid >> 2, q4 = tid & 3;
    const int qi = pair >> 2, hl = pair & 3;
    const int t = tile * QU_QT + qi, h = kh * 12 + hg * 4 + hl;
    const bool live = t < T;
    const long pos = pos0 + t;
    __shared__ uint32_t ubits[QU_MAXW];
    __shared__ uint32_t rbits[QU_QT][QU_MAXW];
    __shared__ uint16_t ulist[QU_MAXW * 32];
    __shared__ int ucnt_sh;
    __shared__ __half ktile[4][256];
    __shared__ __half vtile[4][256];
    const int NW = (nb_max + 31) >> 5;
    for (int w = tid; w < NW; w += 128) ubits[w] = 0u;
    __syncthreads();
    // 1. per-row selection bits (+ own tail block) and their union
    for (int r = 0; r < QU_QT; r++) {
        const int tr = tile * QU_QT + r;
        const long prow = pos0 + tr;
        const int nf = (tr < T) ? (int)((prow + 1) >> 2) : 0;
        const int dense = (tr < T) ? (nsel_rows[tr] < 0) : 1;
        const int tailb = (int)(prow >> 2);
        for (int w = tid; w < NW; w += 128) {
            uint32_t bits = 0u;
            if (tr < T) {
                if (dense) { const int lo = w << 5; bits = (nf >= lo + 32) ? 0xffffffffu : (nf > lo ? ((1u << (nf - lo)) - 1u) : 0u); }
                else {
                    const uint8_t *mb = mask_rows + (size_t)tr * nb_max + (w << 5);
                    const int lim = (nb_max - (w << 5)) < 32 ? (nb_max - (w << 5)) : 32;
                    for (int i = 0; i < lim; i++) bits |= (uint32_t)(mb[i] & 1u) << i;
                }
                if ((tailb >> 5) == w) bits |= 1u << (tailb & 31);
            }
            rbits[r][w] = bits;
            if (bits) atomicOr(&ubits[w], bits);
        }
    }
    __syncthreads();
    // 2. compact the union (ascending) with wave 0: lanes 0-31 take word 2s, lanes 32-63 word 2s+1
    if (tid < 64) {
        int ucnt = 0;
        for (int w0 = 0; w0 < NW; w0 += 2) {
            const int w = w0 + (tid >> 5);
            const int sel = (w < NW) ? (int)((ubits[w] >> (tid & 31)) & 1u) : 0;
            const unsigned long long bal = __ballot(sel);
            if (sel) ulist[ucnt + __popcll(bal & ((1ull << tid) - 1ull))] = (uint16_t)((w << 5) + (tid & 31));
            ucnt += __popcll(bal);
        }
        if (tid == 0) ucnt_sh = ucnt;
    }
    __syncthreads();
    const int ucnt = ucnt_sh;
    // 3. this lane's 64 q dims (scaled by 1/sqrt(256) exactly) as packed halves
    __half2 qv[32];
    const float *qrow = q12288 + (size_t)t * 12288 + (size_t)h * 512;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const int d = q4 * 64 + 2 * i;
        qv[i] = live ? __floats2half2_rn(qrow[d] * 0.0625f, qrow[d + 1] * 0.0625f) : __floats2half2_rn(0.f, 0.f);
    }
    float m = -1e30f, l = 0.f;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.f;
    const qf8_kv_t *kb = kc + (size_t)kh * 256, *vb = vc + (size_t)kh * 256;
    const long pend = pos0 + T;                          // keys < pend exist in the cache
    // 4. walk the union: 4 keys per block, K/V tiles loaded once per block of 128 threads
    for (int u = 0; u < ucnt; u++) {
        const int b = ulist[u];
        __syncthreads();
        for (int e = tid; e < 1024; e += 128) {
            const int j = e >> 8, d = e & 255;
            const long p = (long)b * 4 + j;
            const bool ok = p < pend;
            ktile[j][d] = __float2half(ok ? qf8_kv_load(kb[(size_t)p * 512 + d]) : 0.f);
            vtile[j][d] = __float2half(ok ? qf8_kv_load(vb[(size_t)p * 512 + d]) : 0.f);
        }
        __syncthreads();
        const int selb = (rbits[qi][b >> 5] >> (b & 31)) & 1;
        float sc[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const __half2 *kh2 = (const __half2 *)&ktile[j][q4 * 64];
            float d = 0.f;
            #pragma unroll
            for (int i = 0; i < 32; i++) d = __builtin_amdgcn_fdot2(qv[i], kh2[i], d, false);
            d += __shfl_xor(d, 1, 64);
            d += __shfl_xor(d, 2, 64);
            const long p = (long)b * 4 + j;
            sc[j] = (live && selb && p <= pos) ? d : -1e30f;
        }
        const float mx4 = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
        if (mx4 <= -1e29f) continue;                      // nothing of this block belongs to this pair
        const float mn = fmaxf(m, mx4);
        const float corr = __expf(m - mn);
        const float p0 = __expf(sc[0] - mn), p1 = __expf(sc[1] - mn), p2 = __expf(sc[2] - mn), p3 = __expf(sc[3] - mn);
        l = l * corr + (p0 + p1 + p2 + p3);
        m = mn;
        const __half *v0 = &vtile[0][q4 * 64], *v1 = &vtile[1][q4 * 64], *v2 = &vtile[2][q4 * 64], *v3 = &vtile[3][q4 * 64];
        #pragma unroll
        for (int i = 0; i < 64; i++)
            acc[i] = acc[i] * corr + p0 * __half2float(v0[i]) + p1 * __half2float(v1[i]) + p2 * __half2float(v2[i]) + p3 * __half2float(v3[i]);
    }
    // 5. output: o / l * sigmoid(gate)
    if (live) {
        const float inv = 1.f / (l + 1e-9f);
        float *ao = attn_out + (size_t)t * 6144 + (size_t)h * 256;
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            const int d = q4 * 64 + i;
            ao[d] = acc[i] * inv * qf8_sig(qrow[256 + d]);
        }
    }
}

// ---- CHUNK2 entry: nine launches for T rows (see the kernels above) ----------
// x8q/x8s = X8(mixed) rows [T]; a8q/a8s = X8 scratch for attn_out rows [T][6144].
// Row scratch: blk_score_rows [T][nb_max] f32, mask_rows [T][nb_max] u8,
// list_rows [T][TOPK+1] int, nsel_rows [T] int, part_rows [T][24][8][258] f32.
int qf8_qsa_chunk2(const Qf8AttnW *w, const Qf8I8Mat *iq, const Qf8I8Mat *ik,
                   const Qf8I8Mat *iv, const Qf8I8Mat *iidx, const Qf8I8Mat *io,
                   const int8_t *x8q, const float *x8s, int8_t *a8q, float *a8s, int T,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos0,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq, const float *inv_freq_idx,
                   float *y2560, float *blk_score_rows, uint8_t *mask_rows, int *list_rows,
                   int *nsel_rows, int nb_max, float *part_rows, hipStream_t s) {
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 4;
    m.m[0] = iq;   m.y[0] = q12288;
    m.m[1] = ik;   m.y[1] = k512;
    m.m[2] = iv;   m.y[2] = v512;
    m.m[3] = iidx; m.y[3] = idx640;
    if (qf8_i8_gemv_rows(&m, x8q, x8s, T, s)) return -1;
    QF8_LAUNCH(k8_attn_prep, dim3(QF8_NHEAD + QF8_NKV + 1, T), QF8_WAVE, 0, s,
               q12288, k512, v512, w->q_norm, w->k_norm, kc, vc, inv_freq, pos0);
    QF8_LAUNCH(k8_idx_q_rows, T, 128, 0, s, idx640, w->idx_qnorm, inv_freq_idx, pos0);
    const int nblk = (int)(((pos0 + T - 1) >> 2) - (pos0 >> 2) + 1);
    QF8_LAUNCH(k8_idx_pool_rows, nblk, 128, 0, s, idx640, T, pos0, w->idx_knorm, inv_freq_idx,
               ix->pool_sum, ix->pool_key, ix->pool_cnt);
    QF8_LAUNCH(k8_idx_scores_rows, T, 256, 0, s, idx640, ix->pool_key, blk_score_rows, nb_max, pos0);
    QF8_LAUNCH(k8_idx_topk_rows, T, 256, 0, s, blk_score_rows, mask_rows, nb_max, pos0, list_rows, nsel_rows);
    static int uni = -1;
    if (uni < 0) { const char *e = getenv("QF_QSA_UNION"); uni = (e && e[0] == '0') ? 0 : 1; }
    if (uni && nb_max <= QU_MAXW * 32) {
        QF8_LAUNCH(k8_attn_union8, dim3((T + QU_QT - 1) / QU_QT, QF8_NKV, 3), 128, 0, s,
                   q12288, kc, vc, mask_rows, nb_max, nsel_rows, pos0, T, attn_out);
    } else {
    QF8_LAUNCH(k8_attn_flash_split_rows, dim3(QF8_NHEAD, QF8_FLASH_SPLIT, T), 256, 0, s,
               q12288, kc, vc, pos0, part_rows, list_rows, nsel_rows);
    QF8_LAUNCH(k8_attn_flash_merge_rows, dim3(QF8_NHEAD, T), 256, 0, s, part_rows, q12288, attn_out);
    }
    qf8_i8_x8(attn_out, a8q, a8s, T * QF8_NHEAD * QF8_HDIM, s);
    Qf8I8Multi mo; memset(&mo, 0, sizeof mo); mo.n = 1; mo.m[0] = io; mo.y[0] = y2560;
    if (qf8_i8_gemv_rows(&mo, a8q, a8s, T, s)) return -1;
    return 0;
}
