// qf_moe_m8_wave64.h - batched M=8 routed-expert MoE for gfx906 (wave64).
//
// Canonical workload is M=8 (M=1 is banned as an evaluation target). The
// single-token kernels in qf_nvfp4_wave64.hip load each NVFP4 weight group for
// exactly ONE dot product: an expert visited by w tokens re-streams its whole
// weight matrix w times. These kernels load each weight group ONCE and dot it
// against every token routed to that expert in the batch, so weight traffic
// (the 819 KB/expert roofline term) scales with DISTINCT experts, not with
// (expert,token) pairs. Combined with the four-card partition in
// qf_moe_4card.h, the routed-expert weight stream is spread across ~4 TB/s of
// aggregate MI50 HBM instead of one card's ~1 TB/s.
//
// Each card is self-contained: it holds its experts' weights AND their scalar
// weight_scale_2 folds (s2*_res, indexed by LOCAL resident slot) resident, so
// the per-wave hot path pushes only the tiny CSR + the activation - never the
// scale tables.
//
// Grouping key: the batch's top-k routing is inverted on the host into, per
// card, a CSR over the distinct experts that card owns:
//   exp_slot[e]           resident weight slot of the e-th distinct expert
//   exp_ptr[e], exp_ptr[e+1]   half-open range of this expert's pairs
//   pair_tok[p]           which of the M tokens pair p feeds (0..M-1)
//   pair_wt [p]           that token's routing weight for this expert
// A "pair" is one (expert,token) assignment. hidden is indexed by pair.
#pragma once
#include "qf_nvfp4_wave64.h"   // QF5_NEMBD/NFF/NEXP, EXP_W/S_BYTES, QfStore

#ifndef QF5_M8_TX
#define QF5_M8_TX 16           // tokens per expert per weight load (>= rows per request:
                               // a pair beyond TX would be silently dropped).
#endif                        // gate/up stages TX*NEMBD floats = 40 KB at TX=4.

// Per-card, per-batch routing view (device pointers). No scale tables here -
// those are resident on the card (s2*_res), read by exp_slot.
typedef struct {
    int  n_exp;               // (host view; unused when the CSR is device-built)
    int  n_pairs;
    int  M;                   // tokens in the batch (== 8 canonical)
    int  n_exp_max;           // static launch extent = M*K (blocks early-exit)
    const int *n_exp_dev;     // device-computed distinct-expert count
    const int   *exp_slot;    // [n_exp]  resident slot of each distinct expert
    const int   *exp_ptr;     // [n_exp+1] CSR offsets into pair_*
    const int   *pair_tok;    // [n_pairs] token index (0..M-1) for each pair
    const float *pair_wt;     // [n_pairs] routing weight for each pair
} Qf5M8Route;

#ifdef __cplusplus
extern "C" {
#endif

// gate+up+silu-mul for every pair -> hidden[pair][NFF]. s2g_res/s2u_res are the
// card's resident [EPC] scale folds, indexed by exp_slot.
void qf5_m8_gateup(const uint8_t *Wg, const uint8_t *Sg,
                   const uint8_t *Wu, const uint8_t *Su,
                   const float *s2g_res, const float *s2u_res,
                   const float *x, float *hidden,
                   const Qf5M8Route *rt, hipStream_t stream);

// down projection + weighted accumulate of every pair into y_partial[token].
// y_partial must be pre-zeroed [M][NEMBD]; pairs accumulate with atomicAdd.
void qf5_m8_downacc(const uint8_t *Wd, const uint8_t *Sd, const float *s2d_res,
                    const float *hidden, float *y_partial,
                    const Qf5M8Route *rt, hipStream_t stream);

// Device-side routing inversion: build this card's expert->rows CSR on device
// from the batch top-k (GLM k_moe_gather8 mechanism). counts[0]=n_exp, [1]=n_pairs.
// The host never touches routing in the decode hot path.
void qf5_m8_build_csr(const int *sel_dev, const float *wt_dev, int M, int K,
                      int card, int epc, int il, int *exp_slot, int *exp_ptr,
                      int *pair_tok, float *pair_wt, int *counts, hipStream_t s);

// y_out[i] = y0[i] + y1[i] + y2[i] + y3[i], n = M*NEMBD. Single-pass four-card
// combine so no card serializes the reduction (partials arrive concurrently).
void qf5_m8_reduce4(float *y_out, const float *y0, const float *y1,
                    const float *y2, const float *y3, int n, hipStream_t stream);

#ifdef __cplusplus
}
#endif
