// qf8_dense.h - gfx906 (MI50) per-layer NON-MoE fast path for
// Qwen3.8-Flash-Next (RadixArk NVFP4 checkpoint), ROCm 5.7, wave64.
//
// Session 08 deliverable (dense / GDN / fused QSA attention / PLE).
// Replaces, inside qf_hip4_stage.hip's decode loop:
//   * the 5-launch atomic hyper-connection pre-mixer   -> qf8_hc_pre (3 launches, no atomics)
//   * the 8-launch GDN path                            -> qf8_gdn_mixer (3 launches)
//   * the ~80-launch per-head attention loop           -> qf8_qsa_mixer (5-7 launches, all heads
//     in ONE flash-decode kernel, QSA indexer + top-512 block selection on GPU)
//   * host-round-trip PLE                              -> qf8_ple_step / qf8_ple_apply
//     (hash + 2.5 KiB fp8 row gather on host via the mmap'd store, ALL math on GPU,
//      H2D overlapped through a pinned double buffer + transfer stream + event)
//   * the scalar u16 GEMV                              -> vectorized uint4 wave64 GEMV
//     (x staged in LDS, 8 bf16 per lane-load), optional rocBLAS for the lm_head.
//
// The MoE routed-expert path is NOT here: it stays with session 05's
// qf5_nvfp4_* fused kernels and fixed-slot cache. This API hands the expert
// loop a ready y2560 accumulator and the router logits on device.
//
// Hard rules honored:
//   - decode (T=1) only; every entry point is stream-ordered, ZERO device
//     synchronization and ZERO host readbacks anywhere in the layer path;
//   - all cross-lane reductions are full-wave64 (mask 0xFFFFFFFFFFFFFFFF,
//     offsets 32..1); the indexer q-norm uses 32-lane sub-reductions with
//     offsets <= 16 (wave64-safe, identical trip counts, no divergence);
//   - bf16 handled by exact bit-shift conversion (gfx906 has no bf16 HW);
//   - fp32 accumulation everywhere (production precision, no bit-identical
//     requirement).
#pragma once

#include "qwenflash.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifndef QF_HOST_CHECK
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
// ROCm 5.7 names the gfx906 wave64 shuffle intrinsics without CUDA's
// redundant active-mask parameter.
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#define __shfl_sync(mask, value, lane)        __shfl((value), (lane), warpSize)
#endif
#else
#include "qf8_host_check.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- dims (SEMANTICS.md / qwenflash.h, baked) ------------------------------
#define QF8_NEMBD    2560
#define QF8_RSIZE    10240        // hc_count 4 x 2560
#define QF8_HCC      4
#define QF8_HCL      320
#define QF8_NHEAD    24
#define QF8_NKV      2
#define QF8_HDIM     256
#define QF8_QGATE    512          // q_proj row per head: 256 q + 256 gate
#define QF8_GDN_KH   16
#define QF8_GDN_VH   48
#define QF8_GDN_D    128
#define QF8_DINN     10240
#define QF8_DTRANK   48
#define QF8_NEXP     512
#define QF8_NFF      640
#define QF8_NVOCAB   248320
// Context ceiling. Was 8192 purely because k8_idx_topk held every block's
// (score,index) in LDS for a full bitonic sort - 512 KiB at 262144. That kernel
// now selects by threshold with O(1) shared memory, so the only thing scaling
// with this is the indexer's pooled-key arrays in HBM (~64 MiB per QSA layer at
// 262144, ~193 MiB per GPU), which the init budget guard accounts for.
#define QF8_MAXPOS   262144
#define QF8_IDX_H    4            // indexer heads
#define QF8_IDX_D    128
#define QF8_IDX_C    4            // tokens per pooled block
#define QF8_IDX_TOPK 512          // selected blocks (2048 token budget / 4)
#define QF8_MAXBLK   (QF8_MAXPOS / QF8_IDX_C)
#define QF8_PLE_H    16
#define QF8_PLE_ROW  160          // fp8 bytes per head row
#define QF8_WAVE     64
#define QF8_WMASK    0xFFFFFFFFFFFFFFFFULL

// ---- launch macro (collapses for the host syntax check) --------------------
#ifdef QF_HOST_CHECK
#define QF8_LAUNCH(k, g, b, sm, s, ...) k(__VA_ARGS__)
#else
#define QF8_LAUNCH(k, g, b, sm, s, ...) k<<<g, b, sm, s>>>(__VA_ARGS__)
#endif

// ---- shared device helpers (pure, inlined into every TU) -------------------
#ifdef __cplusplus
__device__ __forceinline__ float qf8_bf2f(uint16_t v) {
    return __uint_as_float(((uint32_t)v) << 16);
}
__device__ __forceinline__ uint16_t qf8_f2bf(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);   // round-to-nearest-even
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float qf8_sig(float v) { return 1.f / (1.f + expf(-v)); }
__device__ __forceinline__ float qf8_silu(float v) { return v / (1.f + expf(-v)); }
// Full-wave64 sum. Caller: a full, non-divergent wave. Result valid in lane 0.
__device__ __forceinline__ float qf8_wave_sum(float v) {
    #pragma unroll
    for (int off = QF8_WAVE / 2; off; off >>= 1)
        v += __shfl_down_sync(QF8_WMASK, v, off);
    return v;
}
#endif

// ---- GEMV library (qf8_gemv.hip) -------------------------------------------
// y[r] = alpha * (W[r] . x) + beta * y[r]. W bf16 [rows, in] row-major.
// x f32 staged in LDS once per block. Vectorized: 8 bf16 per uint4 lane-load.
// in must be a multiple of 64 (all model shapes: 640/2560/6144 qualify).
void qf8_gemv_bf16(const uint16_t *W, const float *x, float *y,
                   int rows, int in, float alpha, float beta, hipStream_t s);

// Fused multi-projection: up to 4 weight matrices sharing the same x and the
// same `in`, ONE launch (grid.y picks the matrix). Used for GDN
// {qkv,z,a,b}, attention {q,k,v,idx_qk} and the shared expert
// {gate,up,gate_inp}.
typedef struct {
    const uint16_t *W[4];
    float          *y[4];
    int             rows[4];
    int             n;
} Qf8GemvMulti;
void qf8_gemv_multi(const Qf8GemvMulti *m, const float *x, int in, hipStream_t s);

// ---- M-row (batch-first) variants: one weight read serves all M rows --------
// x is [M][in], y is [M][rows] (multi: y[k] is [M][rows[k]]). Canonical M=8.
void qf8_gemv_bf16_M(const uint16_t *W, const float *x, float *y,
                     int rows, int in, int M, float alpha, float beta, hipStream_t s);
void qf8_gemv_multi_M(const Qf8GemvMulti *m, const float *x, int in, int M, hipStream_t s);

// lm_head [248320, 2560] -> logits. Uses rocBLAS gemv_ex when built with
// -DQF8_ROCBLAS and QF8_LMHEAD=rocblas (default), else the custom kernel.
// (rocBLAS path converts x to bf16 once per token into ctx-owned scratch.)
void qf8_lm_head(const uint16_t *W, const float *x, float *logits,
                 uint16_t *x_bf16_scratch, hipStream_t s);

void qf8_zero(float *p, int n, hipStream_t s);
void qf8_cvt_bf16_f32(float *out, const uint16_t *in, int n, hipStream_t s);
void qf8_repeat4(float *R, const float *x, hipStream_t s);

// ---- hyper-connections (qf8_hc.hip) ----------------------------------------
// Full GatedResidual pre-mixer in 3 launches, no atomics, no zeroing:
//   1. n = rmsnorm_group(R) (zero-centered (1+w), group 2560)
//   2. d = silu(down(n)/4) [320]  (+ inj[c] = 2*sigmoid(inject(n)/4) if wi)
//   3. mixed[j] = mean_c( sigmoid(up_{c,j} . d) * n_{c,j} )
// inj may be NULL (output HC mixer).
void qf8_hc_pre(float *normed, float *hc_d, float *mixed, float *inj,
                const float *R, const uint16_t *wn, const uint16_t *wd,
                const uint16_t *wu, const uint16_t *wi, hipStream_t s);
// R_c += inj[c] * y  (post-mixer residual inject)
void qf8_stream_inject(float *R, const float *y, const float *inj, hipStream_t s);
// M-row HC: blockIdx.y selects the request row (one launch per stage, not per row).
void qf8_hc_pre_M(float *normed, float *hc_d, float *mixed, float *inj,
                  const float *R, const uint16_t *wn, const uint16_t *wd,
                  const uint16_t *wu, const uint16_t *wi, int M, hipStream_t s);

// ---- per-layer weight views (filled by the stage's existing loaders) -------
typedef struct {
    const uint16_t *norm, *down, *up, *inject;   // bf16
} Qf8HcW;
typedef struct {
    const uint16_t *qkv, *zgate, *in_a, *in_b;   // bf16 [.,2560]
    const uint16_t *conv1d;                      // bf16 [10240,4]
    const float    *a_log, *dt_bias;             // f32 [48]
    const uint16_t *gdn_norm;                    // bf16 [128], ONE-based
    const uint16_t *gdn_out;                     // bf16 [2560,6144]
} Qf8GdnW;
typedef struct {
    const uint16_t *wq, *wk, *wv, *wo;           // bf16
    const uint16_t *q_norm, *k_norm;             // bf16 [256], zero-centered
    const uint16_t *idx_qk;                      // bf16 [640,2560]
    const uint16_t *idx_qnorm, *idx_knorm;       // bf16 [128], zero-centered
} Qf8AttnW;

// ---- QSA indexer per-layer state (device) ----------------------------------
typedef struct {
    float   *pool_sum;    // [QF8_MAXBLK, 128] running fp32 sum
    float   *pool_key;    // [QF8_MAXBLK, 128] normed+roped pooled keys
    int     *pool_cnt;    // [QF8_MAXBLK]
    float   *blk_score;   // [QF8_MAXBLK]
    uint8_t *blk_mask;    // [QF8_MAXBLK], 1 = selected
    int *blk_list, *nsel;              // compact selected-block list + count (the flash kernels walk it)
} Qf8QsaIdx;
int  qf8_qsa_idx_init(Qf8QsaIdx *ix, long maxpos);   // call after hipSetDevice
void qf8_qsa_idx_free(Qf8QsaIdx *ix);

// ---- GDN mixer (qf8_gdn.hip) -----------------------------------------------
// mixed[2560] -> y2560. 3 launches: fused {qkv,z,a,b} projection, fused
// conv1d+recurrence+gated-norm (16 blocks x 192 thr), out_proj GEMV.
// gdnS [48,128,128] f32 and convring [3,10240] f32 are the layer's recurrent
// state (zeroed at init by the caller).
// The same mixer as three separately callable steps (section-marked M=1 path).
void qf8_gdn_proj(const Qf8GdnW *w, const float *mixed,
                  float *qkv_raw, float *z6144, float *a48, float *b48, hipStream_t s);
void qf8_gdn_recur(const Qf8GdnW *w, float *gdnS, float *convring,
                   float *qkv_raw, const float *z6144, const float *a48, const float *b48,
                   float *out6144, hipStream_t s);
void qf8_gdn_outproj(const Qf8GdnW *w, const float *out6144, float *y2560, hipStream_t s);
void qf8_gdn_mixer(const Qf8GdnW *w, const float *mixed,
                   float *gdnS, float *convring,
                   float *qkv_raw, float *z6144, float *a48, float *b48,
                   float *out6144, float *y2560, hipStream_t s);

// KV cache element: FP8 E4M3 storage (8-bit policy) unless -DQF8_KV_BF16.
#ifndef QF8_KV_BF16
typedef uint8_t qf8_kv_t;
#else
typedef uint16_t qf8_kv_t;
#endif

// ---- fused QSA attention (qf8_qsa.hip) -------------------------------------
// mixed[2560] -> y2560. Launches: fused {q,k,v,idx} projection; qk-norm+rope+
// cache write (ONE kernel for all 24 q heads + 2 kv heads + v); indexer step
// (norm/rope of idx q, pool update, block finalize); [scores + top-512 block
// selection only when the 2048-token budget is exceeded -- host branch on pos,
// no sync]; ONE all-head flash-decode kernel (online softmax fp32, QSA block
// mask, output gate) ; o_proj GEMV. No per-head launch loop.
// inv_freq: [32] rope table (dim 64, theta 1e7); inv_freq_idx: [64] (dim 128).
void qf8_qsa_mixer(const Qf8AttnW *w, const float *mixed,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq,
                   const float *inv_freq_idx, float *y2560, hipStream_t s);

// ---- PLE (qf8_ple.hip) ------------------------------------------------------
// Layer-1 n-gram injection. The fp8 table (~tens of GB) stays in the mmap'd
// store (it cannot fit 16 GiB HBM); per token the host performs the 16-op
// integer hash and a 16 x 160 B row gather (pure memcpy, same contract as the
// expert-cache miss gather), staged through a pinned double buffer with its
// own transfer stream + event so the copy overlaps in-flight compute.
// EVERYTHING else -- fp8 dequant, key/value projections, norms, gating,
// dilated conv, residual add -- runs on GPU, stream-ordered, no syncs.
typedef struct Qf8Ple Qf8Ple;
// Loads PLE weights/constants for `layer` (1) onto the CURRENT device.
// Reads layer_multipliers / ngram_heads_vocab_sizes / ngram_heads_offsets
// (I64) from the checkpoint; falls back to the splitmix64/prime derivation if
// absent. Returns 0 on success, 1 if the PLE tensors are absent (PLE disabled,
// not an error), -1 on a real error.
int  qf8_ple_init(Qf8Ple **out, const QfStore *st, int layer);
void qf8_ple_free(Qf8Ple *p);
// Host side of one decode token: push `token` into the history ring, hash,
// gather 16 rows out of the mmap'd table, async H2D on the transfer stream.
// Call once per token BEFORE the layer loop reaches the PLE layer.
int  qf8_ple_step(Qf8Ple *p, const QfStore *st, int token);
// Per-row (independent request) variants: each row owns its n-gram history,
// conv ring, staging buffers and completion event.
int  qf8_ple_step_row(Qf8Ple *p, const QfStore *st, int token, int row);
void qf8_ple_apply_row(Qf8Ple *p, float *R, int row, hipStream_t s);
// Chunk form (row 0, T consecutive positions): host gather of all positions + one
// H2D, then the batched device apply on R rows [T][RSIZE] (see qf8_ple.hip).
int  qf8_ple_step_chunk(Qf8Ple *p, const QfStore *st, const int *tokens, int T);
void qf8_ple_apply_chunk(Qf8Ple *p, float *R, int T, hipStream_t s);
// per-slot chunk form: gather + H2D at token intake (overlaps the GPU's previous chunk), apply waits the slot
#define QF8_PLE_SLOTS 4
int  qf8_ple_prefetch_chunk(Qf8Ple *p, const QfStore *st, const int *tokens, int T, int slot);
void qf8_ple_apply_chunk_slot(Qf8Ple *p, float *R, int T, int slot, hipStream_t s);
int  qf8_ple_reset(Qf8Ple *p);
int  qf8_ple_fork_rows(Qf8Ple *p, int M, hipStream_t s);
int  qf8_ple_commit_rows(Qf8Ple *p, int r, hipStream_t s);
// Device side: R += PLE(R). Stream-waits the prefetch event; no host sync.
void qf8_ple_apply(Qf8Ple *p, float *R, hipStream_t s);

// Correctness taps only: device pointers to PLE intermediates.
// 0=emb 1=key 2=val 3=kn 4=qn 5=gate 6=gv 7=nc. NULL if unavailable.
const float *qf8_ple_dbg(const Qf8Ple *p, int which);

// ---- shared expert fast path (qf8_layer.hip) --------------------------------
typedef struct {
    const uint16_t *gate, *up, *down, *gate_inp;   // bf16
} Qf8ShexpW;
// y2560 += sigmoid(gate_inp.x) * down(silu(gate(x)) * up(x)).
// 4 launches: fused {gate,up,gate_inp} projection, silu-mul, down GEMV, add.
void qf8_shexp(const Qf8ShexpW *w, const float *mixed,
               float *eg, float *eu, float *g1, float *ed, float *y2560,
               hipStream_t s);
// M-row form: one weight stream serves the whole row subset.
void qf8_shexp_M(const Qf8ShexpW *w, const float *mixed, int M,
                 float *eg, float *eu, float *g1, float *ed, float *y2560,
                 int eg_stride, int y_stride, hipStream_t s);

// ---- output head (last stage) ----------------------------------------------
void qf8_out_head(const uint16_t *ohc_norm, const uint16_t *ohc_down,
                  const uint16_t *ohc_up, const uint16_t *lm_head,
                  const float *R, float *normed, float *hc_d, float *mixed,
                  uint16_t *x_bf16_scratch, float *logits, hipStream_t s);

// ---- timing hooks (opt-in; QF8_PROFILE=1) -----------------------------------
// Section tags for qf8_prof_mark; intervals are consecutive event pairs on
// the stage stream. qf8_prof_report is called OFF the hot path (end of a
// token batch / shutdown) and is the only place that synchronizes.
enum {
    QF8_SEC_HC_ATTN = 0, QF8_SEC_MIXER, QF8_SEC_INJECT1, QF8_SEC_HC_FFN,
    QF8_SEC_ROUTER, QF8_SEC_EXPSVC, QF8_SEC_EXPERTS, QF8_SEC_SHEXP,
    QF8_SEC_OUT_HEAD, QF8_SEC_PLE,
    QF8_SEC_EXP_DOWN, QF8_SEC_GDN_RECUR, QF8_SEC_GDN_OUT, QF8_SEC_EXP_GU2, QF8_SEC_EXP_DN2, QF8_SEC_N
};
#define QF8_PROF_SLOTS 4096
typedef struct {
    int       on;
    int       n;                       // marks recorded since last report
    hipEvent_t ev[QF8_PROF_SLOTS];
    int       tag[QF8_PROF_SLOTS];
    double    acc_ms[QF8_SEC_N];       // accumulated by report
    long      acc_n[QF8_SEC_N];
} Qf8Prof;
int   qf8_prof_init(Qf8Prof *p);
void  qf8_prof_free(Qf8Prof *p);
void  qf8_prof_mark(Qf8Prof *p, int tag, hipStream_t s);
// Synchronizes the stream, folds pending intervals into acc_*, prints if
// `print`, resets the mark ring. Off hot path only.
void  qf8_prof_report(Qf8Prof *p, hipStream_t s, FILE *out, int print);
// Print accumulated section totals only (no sync, no ring reset).
void  qf8_prof_print(Qf8Prof *p, FILE *out, const char *label);

#ifdef __cplusplus
}
#endif
