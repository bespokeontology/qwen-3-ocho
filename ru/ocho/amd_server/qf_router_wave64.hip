// qf_router_wave64.hip - device-resident MoE router / top-k / cache dispatch
// for gfx906 (MI50), wave64-safe (wave4 session 09).
//
// HIP mirror of the wave2/01 CUDA device-resident router (see
// prior/wave2/01_router_dispatch/fix.patch). Three kernels per MoE layer:
//   qf_rtr_gemv     router GEMV, logits[512] = Router_bf16[512,2560] . x
//   qf_rtr_topk10   full-512 fp32 softmax + top-10 (norm_topk_prob = false)
//   qf_rtr_dispatch device slot-map/LRU update + pinned miss mailbox
// After dispatch the host runs qf4_expcache_service (mailbox spin -> pinned
// staging -> H2D on the transfer stream -> compute stream waits the event),
// then the device-indexed expert kernels (qf5_nvfp4_*_d) consume
// sel_ids/route_slot/wts without any host routing data.
//
// gfx906 notes: bf16 decoded by exact integer construction (u32 << 16); every
// cross-lane reduction uses the full 64-lane mask (offsets 32..1) with
// cross-wave combines through shared memory.
//
// Host syntax check (no ROCm):
//   g++ -DQF_HOST_CHECK -I. -fsyntax-only -x c++ qf_router_wave64.hip
#include "qf_router_wave64.h"

#include <math.h>

#if !defined(QF_HOST_CHECK) && defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#endif

#define WAVE  64
#define WMASK 0xFFFFFFFFFFFFFFFFULL   // full 64-lane wave
#define NWAVE 8                       // 512 threads / 64

// bf16 -> f32 by exact integer bit construction (no bf16 hardware on gfx906).
__device__ __forceinline__ float qfr_bf2f(uint16_t v) {
    return __uint_as_float(((uint32_t)v) << 16);
}

// ---------------------------------------------------------------------------
// Kernel A: router GEMV. One row per lane (8 blocks x 64 lanes = 512 rows),
// 2560-element serial fp32 dot per lane (~2.6 MFLOP total; launch overhead
// dominates, so no intra-row split). fp32 accumulate, bf16 via u32<<16.
// ---------------------------------------------------------------------------
__global__ void qf_rtr_gemv(const uint16_t *__restrict__ router,
                            const float *__restrict__ x,
                            float *__restrict__ logits) {
    int row = blockIdx.x * WAVE + threadIdx.x;      // 0..511
    const uint16_t *wr = router + (size_t)row * QF_RTR_NEMBD;
    float acc = 0.f;
    for (int i = 0; i < QF_RTR_NEMBD; i++)
        acc = fmaf(qfr_bf2f(wr[i]), x[i], acc);
    logits[row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel B: softmax over ALL 512 logits in fp32 + top-10 by iterative argmax.
// Single block of 512 threads (8 waves), one logit per thread. The selected
// weights start as plain softmax probabilities (exp(l-max)/sum_all) and are
// then RENORMALIZED to sum to 1 (norm_topk_prob = true).
//
// Qwen4ExpTextTopKRouter does
//     if self.norm_topk_prob: router_top_value /= router_top_value.sum(-1)
// norm_topk_prob is absent from the checkpoint's config.json, so the
// Qwen4ExpTextConfig default applies - configuration_qwen4_exp.py:163 = True.
// The kimi-lineage "permanent record" claiming false is wrong on the
// checkpoint's own authority. Without this the routed experts are scaled by
// the mass of the top-10 slice of a 512-way softmax instead of by 1,
// attenuating the whole routed contribution relative to the shared expert by
// a token-dependent factor.
// ---------------------------------------------------------------------------
__global__ void qf_rtr_topk10(const float *__restrict__ logits,
                              int *__restrict__ sel_ids,
                              float *__restrict__ wts) {
    __shared__ float p[QF_RTR_NEXP];    // softmax probs (masked during select)
    __shared__ float red_v[NWAVE];
    __shared__ int   red_i[NWAVE];
    __shared__ float sh_mx, sh_sum, sh_wsum;
    __shared__ int   sh_best;
    // Rows form: block b handles row b of [rows][NEXP] logits -> [rows][TOPK]
    // sel/wts. The single-row launches (<<<1, 512>>>) are block 0, unchanged.
    { const size_t row = blockIdx.x; logits += row * QF_RTR_NEXP; sel_ids += row * QF_RTR_TOPK; wts += row * QF_RTR_TOPK; }
    const int t = threadIdx.x;
    const int lane = t & (WAVE - 1), wid = t / WAVE;
    if (t == 0) sh_wsum = 0.f;

    // block-wide max (wave reduce, full 64-lane mask, shared combine)
    float lv = logits[t];
    float m = lv;
    #pragma unroll
    for (int off = WAVE / 2; off; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(WMASK, m, off));
    if (lane == 0) red_v[wid] = m;
    __syncthreads();
    if (t == 0) {
        float mm = red_v[0];
        #pragma unroll
        for (int w = 1; w < NWAVE; w++) mm = fmaxf(mm, red_v[w]);
        sh_mx = mm;
    }
    __syncthreads();

    // exp(l - max) and block-wide sum -> plain softmax probability per logit
    float e = expf(lv - sh_mx);
    float s = e;
    #pragma unroll
    for (int off = WAVE / 2; off; off >>= 1)
        s += __shfl_down_sync(WMASK, s, off);
    if (lane == 0) red_v[wid] = s;
    __syncthreads();
    if (t == 0) {
        float ss = red_v[0];
        #pragma unroll
        for (int w = 1; w < NWAVE; w++) ss += red_v[w];
        sh_sum = ss;
    }
    __syncthreads();
    p[t] = e / sh_sum;
    __syncthreads();

    // 10 rounds of block-wide argmax over (value, index); the winner is
    // masked to -1 (probs are >= 0) so it cannot be re-selected.
    for (int k = 0; k < QF_RTR_TOPK; k++) {
        float v = p[t];
        int   i = t;
        #pragma unroll
        for (int off = WAVE / 2; off; off >>= 1) {
            float ov = __shfl_down_sync(WMASK, v, off);
            int   oi = __shfl_down_sync(WMASK, i, off);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { red_v[wid] = v; red_i[wid] = i; }
        __syncthreads();
        if (t == 0) {
            float bv = red_v[0];
            int   bi = red_i[0];
            #pragma unroll
            for (int w = 1; w < NWAVE; w++)
                if (red_v[w] > bv || (red_v[w] == bv && red_i[w] < bi)) {
                    bv = red_v[w]; bi = red_i[w];
                }
            sel_ids[k] = bi;
            wts[k]     = bv;
            sh_wsum   += bv;
            sh_best    = bi;
        }
        __syncthreads();
        if (t == sh_best) p[t] = -1.f;
        __syncthreads();
    }

    // norm_topk_prob = true: renormalize the selected weights to sum to 1.
    if (t == 0) {
        float inv = sh_wsum > 0.f ? 1.f / sh_wsum : 0.f;
        #pragma unroll
        for (int k = 0; k < QF_RTR_TOPK; k++) wts[k] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Kernel C: device-resident cache dispatch. Single block, 128 threads
// (2 waves). For each selected expert in order:
//   hit  -> bump the slot's LRU age, hand the slot to the expert kernels via
//           route_slot[k];
//   miss -> block-wide min-age victim search (hit slots were just
//           clock-bumped, so this round's selection is never evicted),
//           device map/age update, and (expert, victim_slot) appended to the
//           pinned mailbox.
// miss_n is written, then __threadfence_system(), then seq last: the host
// polls seq with a plain volatile read of the mapped pinned mailbox (no D2H
// copy anywhere in the protocol).
// ---------------------------------------------------------------------------
#define QFRTR_DTHREADS 128
#define QFRTR_DWAVES   (QFRTR_DTHREADS / WAVE)

// Per-expert selection histogram. Counts are order-independent so atomics are
// fine here (they are NOT on any accumulate path). Answers the question LRU
// cannot: is routing SKEWED - a hot set that would stay resident under
// frequency pinning - or uniform, in which case only full residency helps.
__global__ void qf_rtr_hist(const int *__restrict__ sel_ids,
                            unsigned int *__restrict__ hist) {
    int k = threadIdx.x;
    if (k < QF_RTR_TOPK) atomicAdd(&hist[sel_ids[k]], 1u);
}

__global__ void qf_rtr_dispatch(const int *__restrict__ sel_ids,
                                int *__restrict__ slot_for_expert,
                                int *__restrict__ expert_in_slot,
                                uint64_t *__restrict__ age,
                                uint64_t *__restrict__ clock,
                                int *__restrict__ route_slot,
                                volatile QfRouteMailbox *mb,
                                uint32_t seq, int nslots) {
    __shared__ int miss_n_sh, cur_e, cur_slot;
    __shared__ unsigned long long red_v[QFRTR_DWAVES];
    __shared__ int red_i[QFRTR_DWAVES];
    const int lane = threadIdx.x & (WAVE - 1), wid = threadIdx.x / WAVE;

    if (threadIdx.x == 0) miss_n_sh = 0;
    __syncthreads();

    for (int k = 0; k < QF_RTR_TOPK; k++) {
        if (threadIdx.x == 0) {
            cur_e    = sel_ids[k];
            cur_slot = slot_for_expert[cur_e];
        }
        __syncthreads();
        if (cur_slot >= 0) {
            // resident hit: touch the LRU clock, hand the slot to the GEMVs
            if (threadIdx.x == 0) {
                age[cur_slot]  = ++(*clock);
                route_slot[k]  = cur_slot;
            }
        } else {
            // miss: coldest-slot victim search, min (age, slot) pair
            unsigned long long bv = ~0ull;
            int bi = 0x7fffffff;
            for (int i = threadIdx.x; i < nslots; i += QFRTR_DTHREADS) {
                unsigned long long a2 = age[i];
                if (a2 < bv || (a2 == bv && i < bi)) { bv = a2; bi = i; }
            }
            #pragma unroll
            for (int off = WAVE / 2; off; off >>= 1) {
                unsigned long long ov = __shfl_down_sync(WMASK, bv, off);
                int oi = __shfl_down_sync(WMASK, bi, off);
                if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { red_v[wid] = bv; red_i[wid] = bi; }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned long long mv = red_v[0];
                int mi = red_i[0];
                #pragma unroll
                for (int w = 1; w < QFRTR_DWAVES; w++)
                    if (red_v[w] < mv || (red_v[w] == mv && red_i[w] < mi)) {
                        mv = red_v[w]; mi = red_i[w];
                    }
                int ev = expert_in_slot[mi];
                if (ev >= 0) slot_for_expert[ev] = -1;  // evict victim expert
                slot_for_expert[cur_e] = mi;
                expert_in_slot[mi]     = cur_e;
                age[mi]                = ++(*clock);
                route_slot[k]          = mi;
                int n2 = miss_n_sh;
                if (n2 < QF_ROUTE_MB_CAP) {
                    mb->miss_expert[n2] = cur_e;
                    mb->miss_slot[n2]   = mi;
                    miss_n_sh = n2 + 1;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mb->miss_n = (uint32_t)miss_n_sh;
        __threadfence_system();     // payload visible to the host before seq
        mb->seq = seq;              // written last: host spins on this
    }
}
