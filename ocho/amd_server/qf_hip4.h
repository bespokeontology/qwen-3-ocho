// qf_hip4.h - 4x AMD gfx906 (MI50) pipeline-parallel decode engine for
// Qwen3.8-Flash-Next (48 layers = 4 GPUs x 12-layer stages).
//
// Wave-4 session 09 (persistent 4-GPU server engine core):
//  - Device-resident MoE router: qf_rtr_gemv / qf_rtr_topk10 /
//    qf_rtr_dispatch (qf_router_wave64.h) run the router GEMV, the full-512
//    fp32 softmax top-10 (norm_topk_prob = true) and the expert-cache slot
//    dispatch entirely on device. The only host<->device interaction per
//    layer is the zero-copy pinned QfRouteMailbox (qwenflash.h) written by
//    the dispatch kernel and serviced by qf4_expcache_service. No router
//    logits D2H, no host top-k, no host slot lookups in the decode path.
//  - Device-indexed expert kernels (qf5_nvfp4_*_d in qf_nvfp4_wave64.h):
//    expert id, cache slot and scale_2 are read on device.
//  - Inter-GPU residual handoff is pinned-host-staged OR peer-to-peer, chosen
//    at init from actual gfx906 capabilities (qf4_peer_available). ROCm 5.7
//    MI50 P2P works only with large-BAR PCIe (and xGMI where present), so
//    auto-probe is the correct runtime decision and pinned host staging is
//    the guaranteed fallback. No RCCL.
//  - Persistent state (GDN S matrices, conv rings, KV caches) lives on the
//    owning GPU between requests; qf_hip4_reset clears it via an in-band
//    sentinel.
//
// Design contract (unchanged from prior/09_amd_hip):
//  - Layer-parallel split: GPU g owns layers [12g, 12g+12).
//  - Token-level pipelining: while GPU g+1 runs token t, GPU g runs token
//    t+1. One host thread per GPU; bounded 2-deep mailboxes per boundary.
//  - All reductions are wave64-safe (gfx906 executes 64-lane waves): every
//    cross-lane reduce uses the full 64-lane mask, and cross-wave combines go
//    through shared memory. Sub-wave reductions (the NVFP4 kernels) use
//    offsets <= half the sub-group width.
//  - GDN conv1d weights are consumed as BF16 exactly as stored in the
//    checkpoint ([10240,1,4] squeezed to [10240,4]).
//
// qwenflash.h ships with this tree (src/qwenflash.h); the Makefile's
// ENGINESRC include path can point at the engine tree to override it.
#pragma once

#include "qwenflash.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QF4_NGPU        4
#define QF4_STAGE_NL    12      // layers per GPU (48 / 4)
#define QF4_NEMBD       2560
#define QF4_RSIZE       10240   // hc_count 4 x n_embd 2560, fp32
#define QF4_NVOCAB      248320

// ---------------------------------------------------------------------------
// Public API (qf_hip4_pipeline.cpp; pure host C++, no HIP headers needed).
// ---------------------------------------------------------------------------
typedef struct QfHip4 QfHip4;

#define QF4_HANDOFF_AUTO   0
#define QF4_HANDOFF_PINNED 1
#define QF4_HANDOFF_P2P    2

typedef struct {
    double   load_seconds;              // wall time of qf_hip4_init
    uint64_t tokens_decoded;            // decode steps completed (logits produced)
    double   last_step_ms;              // last submit->logits latency
    size_t   hbm_bytes[QF4_NGPU];       // committed per card (budget guard)
    int      handoff_mode;              // QF4_HANDOFF_* actually in use
    long     max_context;               // QF_MAX_CONTEXT in effect
} QfHip4Stats;

int  qf_hip4_init(QfHip4 **out, const char *model_dir);
int  qf_hip4_submit(QfHip4 *p, int token, long pos);                  // want_logits=1
int  qf_hip4_submit_prefill(QfHip4 *p, int token, long pos);          // want_logits=0, no logits wait needed
int  qf_hip4_wait(QfHip4 *p, long pos, float *logits_out);            // logits: QF4_NVOCAB fp32
int  qf_hip4_decode_step(QfHip4 *p, int token, long pos, float *logits_out);
void qf_hip4_reset(QfHip4 *p);       // between requests only; clears KV/GDN/conv/PLE state
int  qf_hip4_stats(QfHip4 *p, QfHip4Stats *out);
void qf_hip4_free(QfHip4 *p);
// Per-stage wall-clock breakdown (wait/recv/run/send). Inert unless
// /tmp/qf_prof.on exists at process start. Writes to `out`.
void qf_hip4_prof_dump(void *out);

// ---------------------------------------------------------------------------
// Stage API (qf_hip4_stage.hip). All entry points hipSetDevice() first, so
// they are safe to call from the per-stage worker thread regardless of which
// device that thread last used. All calls are synchronizing (they drain the
// stage stream before returning) unless documented otherwise.
// ---------------------------------------------------------------------------
typedef struct Qf4Stage Qf4Stage;

// gpu: physical device ordinal. st: shared, read-only, host-mmapped weight
// store. l0: first global layer index owned by this stage (owns 12).
// Computes the per-GPU byte commitment (weights + expert cache + recurrent
// state + KV at QF_MAX_CONTEXT) BEFORE any device allocation and fails with
// -1 if it exceeds QF4_HBM_BUDGET_MIB (default 15872).
int  qf4_stage_init(Qf4Stage **out, int gpu, const QfStore *st, int l0);
void qf4_stage_free(Qf4Stage *s);

// First stage: embed token -> x[2560], R = 4x repeat of x.
int  qf4_stage_set_token(Qf4Stage *s, int token);
// Non-first stages: H2D the residual handed off from the previous GPU
// (pinned-host handoff path).
int  qf4_stage_push_residual(Qf4Stage *s, const float *hR /*QF4_RSIZE*/);
// Runs this stage's 12 layers. Last stage additionally runs the output HC
// mixer + lm_head. pos is the absolute token position (rope / KV cache).
int  qf4_stage_run(Qf4Stage *s, long pos);
// Identical to qf4_stage_run except the last stage skips the output-HC +
// lm_head GEMV and produces no logits (prompt/prefill tokens: a
// 248320x2560 GEMV per prompt token is pure waste).
int  qf4_stage_run_nologits(Qf4Stage *s, long pos);
// Zeroes all recurrent state: GDN S matrices and conv history rings (they
// carry token history), KV cache occupancy bookkeeping (NOT the cache
// buffers' bytes), and reseeds PLE history to EOS (248044).
int  qf4_stage_reset(Qf4Stage *s);
int  qf4_stage_fork_rows(Qf4Stage *S, int M, long P);   // row 0 -> rows 1..M-1 (fork)
int  qf4_stage_commit_rows(Qf4Stage *S, int r, long P); // row r -> row 0 (decision/commit loop)
void qf4_stage_head_calls_reset(void);
// Non-last stages: D2H the residual for the next GPU (pinned-host path).
int  qf4_stage_pull_residual(Qf4Stage *s, float *hR /*QF4_RSIZE*/);
// Last stage: D2H the logits.
int  qf4_stage_pull_logits(Qf4Stage *s, float *h /*QF4_NVOCAB*/);
// ---- M=8 / M=4 row-subset execution (the heterogeneous scheduler's vocabulary)
// Rows are independent sequences; every row-indexed mutable state (GDN S, conv
// ring, KV cache, QSA indexer) is allocated M-fold and addressed by row stride.
// r0/nr express A={0..3} / B={4..7}.
int  qf4_stage_set_tokens_M(Qf4Stage *s, const int *tokens, int M);
int  qf4_stage_push_residual_M(Qf4Stage *s, const float *hR, int M);
int  qf4_stage_pull_residual_M(Qf4Stage *s, float *hR, int M);
int  qf4_stage_pull_logits_M(Qf4Stage *s, float *h, int M);
int  qf4_stage_head_ids_M(Qf4Stage *s, int *ids, int M);   // device argmax -> M ints
int  qf4_stage_handoff_M(Qf4Stage *src, Qf4Stage *dst, int r0, int nr);
int  qf4_stage_layer_M(Qf4Stage *s, int ll, const long *posM, int r0, int nr);
int  qf4_stage_run_M(Qf4Stage *s, const long *posM, int r0, int nr);
// CHUNK (prefill): T consecutive positions of the sequence in row 0 (T <= 16).
int  qf4_chunk_rows(void);                                                  // QF_CHUNK_ROWS (0 = off)
int  qf4_stage_set_chunk_tokens(Qf4Stage *s, const int *tokens, int T);
// pipelined chunk plumbing (async on the stage stream; events chain the stages)
int qf4_stage_push_rows_async(Qf4Stage *S, const float *hR, int nr);
int qf4_stage_pull_rows_async(Qf4Stage *S, float *hR, int nr);
int qf4_stage_record(Qf4Stage *S, void *ev);          // ev = hipEvent_t (opaque here)
int qf4_stage_wait(Qf4Stage *S, void *ev);
int qf4_stage_event_create(Qf4Stage *S, void **ev);   // embed rows 0..T-1 (stage 0)
int  qf4_stage_run_chunk(Qf4Stage *s, const int *tokens, long pos0, int T);
// Row-subset boundary ops [r0, r0+nr) (tokens/hR are indexed 0..nr-1; posM for
// qf4_stage_run_M is indexed by ABSOLUTE row).
int  qf4_stage_set_tokens_rows(Qf4Stage *s, const int *tokens, int r0, int nr);
int  qf4_stage_push_residual_rows(Qf4Stage *s, const float *hR, int r0, int nr);
int  qf4_stage_pull_residual_rows(Qf4Stage *s, float *hR, int r0, int nr);
int  qf4_stage_head_M(Qf4Stage *s, int M);
// 1 when every owned layer's expert cache is fully resident (slot == expert),
// which the M-row region path requires.
int  qf4_stage_fully_resident(const Qf4Stage *s);

// Committed device bytes computed by the init budget guard.
size_t qf4_stage_hbm_bytes(const Qf4Stage *s);
// Print this stage's accumulated per-section GPU timings (inert unless armed).
void qf4_stage_prof_print(Qf4Stage *s, void *out, const char *label);
// Routing skew census - decides whether pinning a hot set beats LRU.
void qf4_stage_hist_print(Qf4Stage *s, void *out, const char *label);

// ---- P2P handoff (used when qf4_peer_available probes true on all
// boundaries; see qf_hip4_pipeline.cpp). Each stage owns TWO fixed 40 KiB
// device handoff buffers + events; mailbox slots carry slot = pos % 2 so
// producer and consumer agree on buffer/event pairing.
// Device pointer of handoff buffer `slot` (slot is taken mod 2).
const void *qf4_stage_handoff_dev(Qf4Stage *s, int slot);
// Producer: copy R into handoff buffer `slot` on the stage stream and record
// the slot event (stream-ordered; not synchronizing).
int  qf4_stage_emit_residual(Qf4Stage *s, int slot);
// Consumer: wait the producer's slot event, then hipMemcpyPeerAsync the
// 40 KiB residual from src_dev (on src_gpu) into this stage's R
// (stream-ordered; not synchronizing).
int  qf4_stage_recv_residual_dev(Qf4Stage *s, int src_gpu, const void *src_dev,
                                 int src_slot, void *ev);
// Producer's slot event (hipEvent_t, as void* to keep HIP out of the header).
void *qf4_stage_handoff_event(Qf4Stage *s, int slot);
// Probe + enable P2P between two cards: hipDeviceCanAccessPeer and, when
// available, hipDeviceEnablePeerAccess in both directions.
// Returns 1 = P2P usable, 0 = not available, -1 = API error.
int  qf4_peer_available(int src_gpu, int dst_gpu);
// Pinned host memory for the pipeline's staging buffers (portable).
void *qf4_host_pinned(size_t bytes);
void  qf4_host_pinned_free(void *p);

#ifdef __cplusplus
}
#endif
