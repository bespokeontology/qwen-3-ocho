// qf_nvfp4_wave64.h - gfx906 (MI50) wave64-safe fused NVFP4 kernels and the
// fixed-slot HBM expert residency contract for the 4x gfx906 AMD backend.
//
// Session 05 deliverable. Companion to prior/09_amd_hip (qf_hip4.h /
// qf_hip4_stage.hip): these kernels replace k_nvfp4_gemv and the
// whole-expert-blob loading with slot-indexed kernels + a bounded LRU cache
// that mirrors the CUDA-side contract in source/src/model.cpp
// (qf_prepare_expert, QF_EXPERT_CACHE_SLOTS, default 192 slots/layer).
//
// Shapes covered (per SEMANTICS.md, MoE experts):
//   gate/up : W [640, 2560]  NVFP4  -> k_nvfp4_gemv_640x2560 / fused gate+up
//   down    : W [2560, 640]  NVFP4  -> k_nvfp4_gemv_2560x640 / fused acc
//
// Checkpoint packing (must match source/src/cuda/qf.cu and
// prior/09_amd_hip/qf_hip4_stage.hip exactly):
//   row of K values = K/2 packed bytes + K/16 UE4M3 scale bytes.
//   Scale group = 16 values = 8 packed bytes + 1 scale byte.
//   Byte j (0..7) of a group holds value j in the LOW nibble and value j+8
//   in the HIGH nibble (interleaved, NOT sequential 2j/2j+1).
//   val = sign * E2M1[nib & 7] * UE4M3(scale_byte) * scale2   (scale2 f32,
//   per expert per projection, host-resident, passed by value).
//
// gfx906 constraints honored everywhere:
//   - No native FP4/FP8/bf16 instructions: E2M1 and UE4M3 are decoded with
//     exact integer bit construction into f32 (no libm ldexpf on the hot
//     path).
//   - wave64: all cross-lane reductions use the full 64-lane mask with
//     offsets <= half the sub-group width, so 32-lane (half-wave) and 8-lane
//     (eighth-wave) sub-reductions stay independent inside one wave64.
//   - Identical trip counts across the whole wave: no divergence around
//     __shfl_down_sync.
#pragma once

#include <stddef.h>
#include <stdint.h>

// qwenflash.h from the engine tree (QfStore, qf_find). Added to the include
// path by the Makefile (-I$(ENGINESRC)).
#include "qwenflash.h"

#ifndef QF_HOST_CHECK
#include <hip/hip_runtime.h>
#else
#include "qf5_host_check.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Dimensions (baked, from SEMANTICS.md / qwenflash.h)
// ---------------------------------------------------------------------------
#define QF5_NEMBD   2560
#define QF5_NFF     640
#define QF5_NEXP    512
#define QF5_NEXPUSED 10
// Threads per block in the fused gate/up kernel; rows per block = this/32.
// Wider blocks stage the activation vector fewer times (see qf5_gateup_body).
#ifndef QF5_GATEUP_THREADS
#define QF5_GATEUP_THREADS 256
#endif

// Per-expert-per-projection byte strides. Identical for gate [640,2560],
// up [640,2560] and down [2560,640] (same element count), so ONE slot stride
// serves all three weight arrays and all three scale arrays.
#define QF5_EXP_W_BYTES ((size_t)QF5_NFF * QF5_NEMBD / 2)   // 819200 packed
#define QF5_EXP_S_BYTES ((size_t)QF5_NFF * QF5_NEMBD / 16)  // 102400 scales
#define QF5_EXP_SLOT_BYTES (3 * QF5_EXP_W_BYTES + 3 * QF5_EXP_S_BYTES) // 2764800

// ---------------------------------------------------------------------------
// Kernels (qf_nvfp4_wave64.hip). All take the cache BASE pointer plus a slot
// index; the slot offset is computed on device. Launch geometry is exact
// (640 = 80 blocks x 8 rows, 2560 = 80 blocks x 32 rows); no tail guards.
// All are stream-ordered; x/hidden/y are device f32 buffers.
// ---------------------------------------------------------------------------

// y[640]  = dequant(W[slot]) @ x[2560] * scale2          (gate or up, plain)
void qf5_nvfp4_gemv_640x2560(const uint8_t *W, const uint8_t *S,
                             const float *x, float *y, float scale2,
                             int slot, hipStream_t stream);

// y[2560] = dequant(W[slot]) @ x[640] * scale2           (down, plain)
void qf5_nvfp4_gemv_2560x640(const uint8_t *W, const uint8_t *S,
                             const float *x, float *y, float scale2,
                             int slot, hipStream_t stream);

// Fused gate+up+silu-mul (halves x staging and launch count vs two plain
// gemvs + k_silu_mul):
//   hidden[r] = silu(g[slot]_r . x * s2g) * (u[slot]_r . x * s2u), r<640
void qf5_nvfp4_gateup_640x2560(const uint8_t *Wg, const uint8_t *Sg,
                               const uint8_t *Wu, const uint8_t *Su,
                               const float *x, float *hidden,
                               float s2g, float s2u, int slot,
                               hipStream_t stream);

// Fused down + weighted accumulate (replaces gemv into scratch + k_axpy;
// y must be zeroed once per token before the top-10 loop):
//   y[r] += w * (d[slot]_r . hidden * s2d), r<2560
void qf5_nvfp4_downacc_2560x640(const uint8_t *W, const uint8_t *S,
                                const float *hidden, float w, float *y,
                                float s2d, int slot, hipStream_t stream);

// ---------------------------------------------------------------------------
// Device-indexed variants (wave4 session 09): identical fused bodies, but the
// expert id, cache slot, scale2 (and down weight) are read ON DEVICE from the
// router outputs (sel_ids / slot_ids / wts) and the device scale_2 tables, so
// the host launch loop needs no routing data. slot_ids[k] < 0 fails safe:
// gateup writes zeros to hidden, downacc leaves y untouched (must be
// impossible: the compute stream waits the miss-transfer event first).
// The value-based entry points above are kept for A/B.
// ---------------------------------------------------------------------------
void qf5_nvfp4_gateup_640x2560_d(const uint8_t *Wg, const uint8_t *Sg,
                                 const uint8_t *Wu, const uint8_t *Su,
                                 const float *x, float *hidden,
                                 const float *s2g_dev, const float *s2u_dev,
                                 const int *sel_ids, const int *slot_ids,
                                 int k, hipStream_t stream);
void qf5_nvfp4_downacc_2560x640_d(const uint8_t *W, const uint8_t *S,
                                  const float *hidden, const float *wts,
                                  float *y, const float *s2d_dev,
                                  const int *sel_ids, const int *slot_ids,
                                  int k, hipStream_t stream);

// ---------------------------------------------------------------------------
// Fixed-slot HBM expert cache (qf_expert_slots.cpp).
// Bounded residency: `slots` fixed expert slots per layer, all in HBM
// (hipMalloc). Host keeps the maps; device memory is addressed purely by
// slot. Misses go through the fixed pinned double-buffer in Qf4ExpXfer and
// overlap with compute via a transfer stream + events. No unbounded
// full-expert load, no host-mapped zero-copy in the decode hot path.
// ---------------------------------------------------------------------------
typedef struct {
    int      layer;                 // global layer index (tensor names)
    int      slots;                 // fixed slot count for this layer
    // device, slot-strided blobs:
    uint8_t *w_gate, *w_up, *w_down;        // slots * QF5_EXP_W_BYTES each
    // gfx906 production FP8 path (QF8_MOE_FP8=1): lane-chunk E4M3 + one scale
    // per (row, lane). Replaces the NVFP4 upload rather than duplicating it.
    uint8_t *w8_gate, *w8_up, *w8_down;     // slots * rows * in bytes
    float   *sc_gate, *sc_up, *sc_down;     // slots * rows * 64 floats
    // M=1 gfx906-native int8 representation (QF_M1_INT8=1, qf8_moe_i8.h):
    // built on device from the NVFP4 slots at init; per-row fp32 scales.
    int8_t  *i8_gate, *i8_up, *i8_down;     // slots * rows * in bytes
    float   *i8s_gate, *i8s_up, *i8s_down;  // slots * rows floats
    int      nv_ring;                       // QF_M1_INT8_ONLY=1: the NVFP4 arrays are a
                                            // nv_ring-slot staging ring (converted per expert
                                            // at load), not a resident arena; 0 = full arena
    uint8_t *s_gate, *s_up, *s_down;        // slots * QF5_EXP_S_BYTES each
    // host bookkeeping (INIT-SEED ONLY during decode; ground truth is device):
    int      *slot_for_expert;      // [QF5_NEXP], -1 when not resident
    int      *expert_in_slot;       // [slots], -1 when empty
    uint64_t *age;                  // [slots] LRU timestamps
    uint64_t  clock;
    // host, loaded once at init (tiny): per-expert scale_2 per projection
    float    *s2g, *s2u, *s2d;      // [QF5_NEXP] each
    // wave4 session 09: device-resident routing state (mirrors the wave2/01
    // CUDA contract in qwenflash.h). During decode the slot map, reverse map,
    // LRU clocks and miss bookkeeping live on device and are updated by
    // qf_rtr_dispatch (qf_router_wave64.hip); the host maps above only seed
    // these at init.
    int       fully_resident;       // 1 = slot_for_expert is the identity map;
                                    // no LRU, no mailbox, no host servicing
    int      *slot_for_expert_dev;  // [QF5_NEXP], -1 when not resident
    int      *expert_in_slot_dev;   // [slots]
    uint64_t *age_dev;              // [slots] device LRU timestamps
    uint64_t *clock_dev;            // [1] device LRU clock
    float    *s2g_dev, *s2u_dev, *s2d_dev;  // [QF5_NEXP] device scale_2 tables
    int      *route_slot_dev;       // [QF5_NEXPUSED] slots for current top-10
    QfRouteMailbox *mb_host;        // pinned miss mailbox (host view)
    QfRouteMailbox *mb_dev;         // device alias of the same mailbox
    uint32_t  expected_seq;         // host: next mailbox seq stamp to wait for
} Qf4ExpCache;

// Per-GPU transfer context: fixed pinned staging (double buffer) + transfer
// stream + one event per buffer. Created once per gfx906 card and shared by
// that card's 12 layer caches, so pinned memory stays bounded regardless of
// layer count.
typedef struct {
    void        *buf[2];            // pinned, QF5_EXP_SLOT_BYTES each
    hipEvent_t   ev[2];
    int          cur;
    hipStream_t  xfer;
} Qf4ExpXfer;

// Transfer context lifecycle (call after hipSetDevice(gpu)).
int  qf4_expxfer_init(Qf4ExpXfer *xf);
void qf4_expxfer_free(Qf4ExpXfer *xf);

// Allocates the fixed slot blobs in HBM and loads the per-expert scale_2
// tables from the store. slots <= 0 -> QF_EXPERT_CACHE_SLOTS env, else 192.
int  qf4_expcache_init(Qf4ExpCache *c, const QfStore *st, int layer, int slots);
void qf4_expcache_free(Qf4ExpCache *c);

// Returns the resident slot for `expert`, loading it on miss:
// gather mmap -> fixed pinned staging (waits for that buffer's previous
// transfer), async H2D on xf->xfer, hipEventRecord, compute stream waits on
// the event. The host gather overlaps in-flight device compute; the decode
// stream never blocks on the CPU. Returns -1 on error.
int  qf4_expcache_prepare(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                          int expert, hipStream_t compute);

// wave4 session 09: host side of the device-resident dispatch protocol.
// Spins on the pinned mailbox until qf_rtr_dispatch stamps seq ==
// c->expected_seq (a volatile read of zero-copy mapped memory -- the one
// sanctioned host<->device interaction per layer, NOT a memcpy readback),
// then for each recorded miss gathers the expert bytes out of the mmap'd
// store into the fixed pinned double buffer (waiting on that buffer's prior
// event), hipMemcpyAsync H2D on xf->xfer, hipEventRecord, and
// hipStreamWaitEvent(compute, ev) so the compute stream is ordered after the
// slot fill. All slot-map/LRU metadata was already updated ON DEVICE by the
// dispatch kernel; this function is a pure DMA engine. Returns 0 on success.
// Cumulative expert-cache accounting: total lookups, misses, and bytes
// streamed H2D. A miss costs a full expert (2.7 MB) over PCIe.
// Fused all-expert MoE: replaces the NEXPUSED x {gateup, downacc} launch loop
// with two launches. hidden must be nexp * QF5_NFF floats. y is SEEDED (no
// separate zeroing pass needed).
void qf5_nvfp4_moe_all(const uint8_t *Wg, const uint8_t *Sg,
                       const uint8_t *Wu, const uint8_t *Su,
                       const uint8_t *Wd, const uint8_t *Sd,
                       const float *x, float *hidden, float *y,
                       const float *s2g_dev, const float *s2u_dev,
                       const float *s2d_dev, const float *wts,
                       const int *sel_ids, const int *slot_ids,
                       int nexp, hipStream_t stream);

// The two halves of qf5_nvfp4_moe_all as separate launches (section-marked M=1 path).
void qf5_nvfp4_gateup_all_launch(const uint8_t *Wg, const uint8_t *Sg,
                                 const uint8_t *Wu, const uint8_t *Su,
                                 const float *x, float *hidden,
                                 const float *s2g_dev, const float *s2u_dev,
                                 const int *sel_ids, const int *slot_ids,
                                 int nexp, hipStream_t stream);
void qf5_nvfp4_downacc_all_launch(const uint8_t *Wd, const uint8_t *Sd,
                                  const float *hidden, const float *wts, float *y,
                                  const float *s2d_dev, const int *sel_ids,
                                  const int *slot_ids, int nexp, hipStream_t stream);
// Fill every slot at init when slots == QF5_NEXP, making the expert->slot map
// a permanent identity. Returns 1 if fully resident, 0 if not eligible, -1 on
// error. Removes ALL per-token host work for experts.
int qf4_expcache_prefill(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                         hipStream_t compute);

void qf4_expcache_stats(unsigned long long *calls, unsigned long long *miss,
                        unsigned long long *bytes);
int  qf4_expcache_service(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                          hipStream_t compute);

#ifdef __cplusplus
}
#endif
