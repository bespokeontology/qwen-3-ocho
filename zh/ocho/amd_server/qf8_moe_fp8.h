// qf8_moe_fp8.h - gfx906 production FP8 routed-expert MoE.
//
// The FORMAT is designed for the wave, not inherited from a generic slab.
// gfx906: wave64, no native FP4/FP8 exec, ~894 GB/s measured streaming ceiling
// needing 8-16 waves/CU. E4M3 (1 byte -> 1 half) beats E2M1 (1 byte -> 2 halves)
// per op here and the extra bytes are free at 93% bandwidth headroom.
//
// LANE-CHUNK LAYOUT (packed offline, once, at load):
//   every weight row is cut into exactly QF8_WAVE (64) contiguous chunks, so
//   lane L of a wave owns chunk L of the row and needs EXACTLY ONE scale.
//     in = 2560 (gate/up) -> 40 bytes/lane ; in = 640 (down) -> 10 bytes/lane
//   weights: [slot][row][in]        E4M3, lane-contiguous => one coalesced
//                                   wave transaction per pass, no address math
//                                   in the inner loop
//   scales : [slot][row][64]        fp32, one per (row, lane), loaded once
// Padding is unnecessary: 2560 and 640 are both exact multiples of 64.
//
// EXECUTION: device-built expert descriptors + a PERSISTENT work queue. A fixed
// grid of workgroups pulls (expert, output-row-tile) work items with an atomic
// counter until the layer's routed work is exhausted, so uneven routing cannot
// leave CUs idle and the tiny-expert case never becomes a launch per expert.
// One resident expert tile services every row routed to it before it leaves
// registers (expert -> rows inversion, never row -> expert).
#pragma once
#include "qf8_dense.h"

#define QF8_FP8_MAXROW 16           // rows a descriptor can carry (16-row step)
#define QF8_FP8_RPT    4            // output rows per work item (one wave each)
#define QF8_FP8_WGS    240          // persistent workgroups (60 CUs x 4)
// Per-stage control block (device int[QF8_FP8_CTL]): [0] = descriptor count,
// [1] = gate/up work cursor, [2] = down work cursor. build_desc writes all
// three on the stream, so the persistent kernels never reset a counter
// themselves (a block-0 in-kernel reset races the other blocks' first
// atomicAdd and can double-accumulate in downacc) and two in-flight batches
// never share a cursor.
#define QF8_FP8_CTL    4
// Resident bytes per expert slot in this format (gate+up+down E4M3 + scales).
#define QF8_FP8_SLOT_BYTES ((size_t)3 * QF8_NFF * QF8_NEMBD + \
                            (size_t)2 * QF8_NFF * QF8_WAVE * 4 + (size_t)QF8_NEMBD * QF8_WAVE * 4)

// Device expert descriptor: built by the routing-inversion kernel, consumed by
// the persistent scheduler. Route coefficients travel WITH the descriptor so the
// down kernel needs no second routing lookup.
typedef struct {
    int   slot;                          // resident expert slot
    int   nrow;                          // rows routed to this expert
    int   row[QF8_FP8_MAXROW];           // which batch rows
    float wt[QF8_FP8_MAXROW];            // router weights
} Qf8MoeDesc;

#ifdef __cplusplus
extern "C" {
#endif
// Offline packer: NVFP4 expert (checkpoint form) -> lane-chunk E4M3 + scales.
// rows x in, called once per (expert, projection) at load.
int  qf8_fp8_pack_expert(const uint8_t *nvfp4_w, const uint8_t *nvfp4_s, float s2,
                         int rows, int in, uint8_t *w8_out, float *scale_out);
// Device routing inversion: top-k -> expert descriptors, writes ctl[0..2].
// Requires slot == expert id + slot_base (fully resident cache).
void qf8_fp8_build_desc(const int *sel_dev, const float *wt_dev, int M, int K,
                        int slot_base, Qf8MoeDesc *desc, int *ctl, hipStream_t s);
// Fused Gate+Up+SiLU*Up over all descriptors. hidden is [desc][row][NFF].
void qf8_fp8_gateup(const uint8_t *Wg, const float *Sg, const uint8_t *Wu, const float *Su,
                    const float *x, float *hidden, const Qf8MoeDesc *desc,
                    int *ctl, hipStream_t s);
// Fused Down + router-weighted accumulate into y (no separate combine pass).
// y must be pre-zeroed [M][NEMBD] on the stream: pairs accumulate with atomicAdd.
void qf8_fp8_downacc(const uint8_t *Wd, const float *Sd, const float *hidden,
                     float *y, const Qf8MoeDesc *desc, int *ctl, hipStream_t s);
#ifdef __cplusplus
}
#endif
