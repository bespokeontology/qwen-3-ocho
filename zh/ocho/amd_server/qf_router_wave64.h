// qf_router_wave64.h - device-resident MoE router + top-k + expert-cache
// dispatch for the 4x gfx906 (MI50) backend (wave4 session 09).
//
// Mirrors the wave2/01 CUDA design (prior/wave2/01_router_dispatch) in HIP:
// the router GEMV, softmax top-10, and slot-map/LRU dispatch all run on
// device; the only host interaction is the zero-copy pinned QfRouteMailbox
// (qwenflash.h) written by qf_rtr_dispatch and polled by
// qf4_expcache_service (qf_expert_slots.cpp). No router-logit D2H, no host
// top-k, no host slot lookups in the decode hot path.
//
// Semantics (SEMANTICS.md, norm_topk_prob = false): logits = router @ x,
// softmax over ALL 512 logits in fp32, top-10 selection, and the weights are
// the plain softmax probabilities of the selected experts -- NOT renormalized
// among the 10.
//
// gfx906: no native bf16 -- router weights are decoded by exact integer bit
// construction (uint32 << 16); all cross-lane reductions use the full 64-lane
// wave mask with offsets 32..1.
//
// Host syntax check (no ROCm):
//   g++ -DQF_HOST_CHECK -I. -fsyntax-only -x c++ qf_router_wave64.hip
#pragma once

#include <stdint.h>
#include "qwenflash.h"

#ifndef QF_HOST_CHECK
#include <hip/hip_runtime.h>
#else
#include "qf5_host_check.h"
#endif

#define QF_RTR_NEXP 512     // router rows / expert count
#define QF_RTR_NEMBD 2560   // router columns
#define QF_RTR_TOPK 10      // experts per token

// ---------------------------------------------------------------------------
// Kernels (qf_router_wave64.hip). Fixed launch geometry:
//   qf_rtr_gemv     <<<8, 64>>>   one router row per lane (8x64 = 512)
//   qf_rtr_topk10   <<<1, 512>>>  single block, 8 waves
//   qf_rtr_dispatch <<<1, 128>>>  single block, 2 waves
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// logits[512] = Router_bf16[512,2560] . mixed_f32[2560], fp32 accumulate.
__global__ void qf_rtr_gemv(const uint16_t *router, const float *x, float *logits);

// Full-512 fp32 softmax + iterative argmax top-10 (norm_topk_prob = false).
// Outputs sel_ids[10] (int) and wts[10] (plain softmax probs, unnormalized
// among the 10).
__global__ void qf_rtr_topk10(const float *logits, int *sel_ids, float *wts);
__global__ void qf_rtr_hist(const int *sel_ids, unsigned int *hist);

// Resolve a cache slot for every selected expert, entirely on device. Hit:
// bump the slot's LRU age, emit the slot into route_slot[k]. Miss: pick the
// min-age victim slot on device, update both maps and ages on device, append
// (expert, victim_slot) to the pinned mailbox. miss_n is written, then
// __threadfence_system(), then seq (monotone per-layer-per-token stamp
// provided by the host) is written last.
__global__ void qf_rtr_dispatch(const int *sel_ids,
                                int *slot_for_expert, int *expert_in_slot,
                                uint64_t *age, uint64_t *clock,
                                int *route_slot,
                                volatile QfRouteMailbox *mb,
                                uint32_t seq, int nslots);

#ifdef __cplusplus
}
#endif

// Host-side launch helpers (fixed geometry). Collapse to plain calls under
// QF_HOST_CHECK.
#ifdef QF_HOST_CHECK
#define QFRTR_LAUNCH(k, g, b, s, ...) k(__VA_ARGS__)
#else
#define QFRTR_LAUNCH(k, g, b, s, ...) k<<<g, b, 0, s>>>(__VA_ARGS__)
#endif

static inline void qf_rtr_gemv_launch(const uint16_t *router, const float *x,
                                      float *logits, hipStream_t s) {
    QFRTR_LAUNCH(qf_rtr_gemv, 8, 64, s, router, x, logits);
}
// Selection histogram: hist[e] += 1 per selected expert, per layer per token.
// Reveals routing skew, which decides whether pinning a hot set beats LRU.
static inline void qf_rtr_hist_launch(const int *sel_ids, unsigned int *hist,
                                      hipStream_t s) {
    QFRTR_LAUNCH(qf_rtr_hist, 1, QF_RTR_TOPK, s, sel_ids, hist);
}
static inline void qf_rtr_topk10_launch(const float *logits, int *sel_ids,
                                        float *wts, hipStream_t s) {
    QFRTR_LAUNCH(qf_rtr_topk10, 1, 512, s, logits, sel_ids, wts);
}
// T rows in ONE launch (grid = T blocks of 512): logits [T][NEXP] -> sel/wts [T][TOPK].
static inline void qf_rtr_topk10_rows_launch(const float *logits, int *sel_ids, float *wts, int T, hipStream_t s) {
    QFRTR_LAUNCH(qf_rtr_topk10, T, 512, s, logits, sel_ids, wts);
}
static inline void qf_rtr_dispatch_launch(const int *sel_ids,
                                          int *slot_for_expert, int *expert_in_slot,
                                          uint64_t *age, uint64_t *clock,
                                          int *route_slot,
                                          volatile QfRouteMailbox *mb,
                                          uint32_t seq, int nslots, hipStream_t s) {
    QFRTR_LAUNCH(qf_rtr_dispatch, 1, 128, s, sel_ids, slot_for_expert,
                 expert_in_slot, age, clock, route_slot, mb, seq, nslots);
}
