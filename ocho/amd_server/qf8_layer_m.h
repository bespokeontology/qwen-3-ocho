// qf8_layer_m.h - M=8/M=4 row-subset execution for the existing gfx906 AMD
// executor. The arithmetic kernels (qf8_hc_pre, qf8_gdn_mixer, qf8_qsa_mixer,
// qf8_shexp, qf8_ple_apply, qf8_gemv*) are REUSED unchanged; this layer adds the
// scheduling vocabulary Spark already has: a batch of M independent rows, and
// the ability to run a row subset [r0, r0+nr) so A={0..3} / B={4..7} can be
// scheduled independently against the Spark domain.
//
// STAGE CLASSIFICATION (traced from qf_hip4_stage.hip's decode loop):
//   qf8_ple_apply    state per row (history ring)      -> row-indexed, per-row call
//   qf8_hc_pre       dense down/up + elementwise       -> per-row now; M-GEMV drop-in next
//   qf8_gdn_mixer    recurrent state per row           -> row-indexed state, per-row call
//   qf8_qsa_mixer    KV/indexer state per row          -> row-indexed state, per-row call
//   qf8_stream_inject elementwise                      -> per-row (trivial)
//   router GEMV      pure weight stream                -> qf8_gemv_bf16_M (BATCHED)
//   routed MoE       already M-capable (qf5_m8_*)      -> BATCHED
//   qf8_shexp        dense                             -> per-row now; M-GEMV next
//   out_head/lm_head pure weight stream                -> qf8_gemv_bf16_M (BATCHED)
// Row-indexed state must be allocated M-fold by the caller (Qf8StateM).
#pragma once
#include "qf8_dense.h"

#define QF8_M_MAXROWS 16                 // rows per request / per stage step
// Row strides for the M-folded executor state (row r at base + r*stride).
#ifndef QF8_MROWS
#define QF8_MROWS 32                     // rows RESIDENT on this card: two 16-row slots
#endif                                   // (W2 pipeline). A request carries <= QF8_M_MAXROWS rows.
#define QF8_GDNS_STRIDE (48 * 128 * 128) // floats per row of GDN S
#define QF8_RING_STRIDE (3 * 10240)      // floats per row of the conv ring

// M-fold per-row state for one layer range on this card.
typedef struct {
    int    M;                       // rows resident (8 canonical)
    float *R;                       // [M][HCC*NEMBD]
    float *normed, *hc_d, *mixed, *inj, *y2560;
    float *qkv_raw, *z6144, *a48, *b48, *q12288, *k512, *v512, *idx640, *attn_out8;
    float *router, *eg, *eu, *g1, *ed, *logits;
    float **gdnS;                   // [nlayer][M * 48*128*128]
    float **convring;               // [nlayer][M * 3*10240]
    uint16_t **kc, **vc;            // [nlayer][M * maxpos*512]
    Qf8QsaIdx *idx;                 // [nlayer*M]
} Qf8StateM;

#ifdef __cplusplus
extern "C" {
#endif
// Row-subset layer step: executes layer `il` for rows [r0, r0+nr) of the batch.
// Reuses every existing gfx906 kernel; only indexing/batching is new.
int qf8_layer_step_M(Qf8StateM *S, void *layerw, int il, int ll, long *posM,
                     int r0, int nr, hipStream_t s);
// Batched output head for all M rows (one weight stream for the batch).
int qf8_out_head_M(Qf8StateM *S, const uint16_t *ohc_norm, const uint16_t *ohc_down,
                   const uint16_t *ohc_up, const uint16_t *lm_head, int M, hipStream_t s);
#ifdef __cplusplus
}
#endif
