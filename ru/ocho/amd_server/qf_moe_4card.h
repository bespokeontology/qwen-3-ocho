// qf_moe_4card.h - four-MI50 routed-expert tier for the M=8 QWE MoE fork.
//
// Experts are partitioned across the four cards (expert e lives on card
// e/EPC, EPC = QF5_NEXP/QF5_NCARD = 128) and stay resident for the process
// lifetime. A batch's ~44 distinct experts (M=8, ~45% overlap) therefore
// stream from four HBM stacks at once instead of one - the reason the box has
// four cards. Each card produces a PARTIAL routed output for the tokens whose
// experts it happens to own; the four partials are summed (P2P) into y_routed.
//
// This is the AMD half of the heterogeneous fork. Spark owns attention + the
// dense side layers + the shared expert; AMD owns the routed experts. The two
// meet only at the shared/routed combine (qf_moe_handoff.h).
//
// nwaves > 1 keeps several M=8 micro-batches outstanding so the cards stay
// occupied while Spark works and while activations cross the wire.
#pragma once
#include "qf_moe_m8_wave64.h"

#define QF5_NCARD 4
#define QF5_EPC   (QF5_NEXP / QF5_NCARD)     // 128 experts per card
#define QF5_NLAYER 48                         // MoE layers (slot = il*EPC + local_expert)

// One card's resident NVFP4 weights (local slot = global expert - card*EPC).
typedef struct {
    int dev;                 // hip device ordinal
    hipStream_t comp;        // this card's compute stream (per wave; see pool)
    const uint8_t *Wg, *Sg, *Wu, *Su, *Wd, *Sd;   // device [NLAYER*EPC][EXP_*_BYTES]
    const float   *s2g, *s2u, *s2d;               // device [NLAYER*EPC] weight_scale_2 folds
} Qf5Card;

typedef struct Qf5MoePool Qf5MoePool;

#ifdef __cplusplus
extern "C" {
#endif

// cards[QF5_NCARD] describe residency (dev + weight bases). s2*_all are host
// arrays [QF5_NEXP] of the weight_scale_2 folds, gathered per batch. M is the
// canonical micro-batch (8). nwaves is the ring depth (outstanding batches).
Qf5MoePool *qf5_moe_pool_init(const Qf5Card cards[QF5_NCARD], int M, int nwaves);
void        qf5_moe_pool_free(Qf5MoePool *p);

// Submit one M=8 batch on ring slot `wave`. x is [M][NEMBD] (host). sel/wt are
// [M][K] top-k routing (K == QF5_NEXPUSED). Returns after launch (async): all
// four cards + the reduce run on-device; nothing is synced here.
void qf5_moe_m8_submit(Qf5MoePool *p, int wave, int il, const float *x_host,
                       const int *sel, const float *wt, int K);

// Wait for ring slot `wave` and copy the reduced routed output to
// y_routed_host [M][NEMBD]. Pairs with a prior submit on the same slot.
void qf5_moe_m8_wait(Qf5MoePool *p, int wave, float *y_routed_host);

#ifdef __cplusplus
}
#endif
