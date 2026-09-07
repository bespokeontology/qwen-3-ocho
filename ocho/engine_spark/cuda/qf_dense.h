// qf_dense.h - fused dense BF16 projections for batch=1 decode (GB10 / sm_121a)
//
// One launch per projection *group*: every projection in a group consumes the
// same activation vector, so the activation is read from HBM once per block
// and the whole group is a single kernel launch. Groups used by qf.cu:
//   GDN in-proj:  qkv[10240] + z[6144] + a[48] + b[48]   (in = 2560)
//   QSA in-proj:  q[12288]  + k[512] + v[512]            (in = 2560)
//   MoE in-proj:  router[512] + shexp gate[640] + up[640] + gate_inp[1]
//   out-proj:     gdn_out / wo / shexp_down              (single-proj group)
//   lm_head:      logits[248320]                         (single-proj group)
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// One projection in a fused group: y[0..rows) = W[rows, in] * x[in].
// W is BF16 row-major, base 16-byte aligned (any cudaMalloc'd tensor).
typedef struct {
    const void *W;   // BF16 [rows, in] row-major
    float *y;        // fp32 output [rows]
    int rows;
} QfDenseProj;

// Fused multi-projection GEMV. Eligibility: 1..4 projections, in % 8 == 0,
// in <= 8192 (fp32 shared staging), x and every W 16-byte aligned.
// Returns 0 when the fused kernel was queued, 1 when the call was dispatched
// per-projection to the legacy gemv_bf16 path (ineligible shape or
// QF_DENSE=legacy). Both paths compute the same result.
int qfd_gemv_group(const QfDenseProj *projs, int nproj, const float *x, int in,
                   cudaStream_t s);

// Quantize one BF16 device tensor to E4M3 + UE4M3 per-64-block scales
// (slab layout: [scales rows*(in/64) | pad->16 | E4M3 weights]). Returns 0
// and the E4M3 weight base in *W8. Call qfd_set_fp8_weights(1) once every
// dense tensor is converted; the group GEMV then uses the FP8 kernel.
// Batched group GEMV: T activations against ONE weight read. xT is [T][in],
// each projs[p].y is [T][rows]. Returns 0 on success, -1 if ineligible (caller
// falls back to T sequential qfd_gemv_group calls). Same arithmetic per
// (token, row) as qfd_gemv_group.
int qfd_gemv_group_T(const QfDenseProj *projs, int nproj, const float *xT, int in,
                     int T, cudaStream_t s);
// Reset the output-pointer slot cursor. Must be called at the top of every
// batched pass so a given call site always takes the same slot - that is what
// lets the pass be CUDA-graph captured (see the pool comment in qf_dense.cu).
void qfd_gemv_group_T_rewind(void);
void qfd_gemv_group_T_rewind_eager(void);   // eager prefill chunk: own slot region, rewind once per chunk

int qfd_quant_fp8(const void *src_bf16, int rows, int in, cudaStream_t s, void **W8);
void qfd_set_fp8_weights(int on);
// Returns 1 when dense weights were converted to E4M3 slabs at load.
int qfd_fp8_is_on(void);

// Decode call-site wrappers; dims are bound to the same -D macros as qf.cu.
int qfd_gdn_inproj(const void *wqkv, const void *wz, const void *wa, const void *wb,
                   const float *x, float *yqkv, float *yz, float *ya, float *yb,
                   cudaStream_t s);
int qfd_qsa_inproj4(const void *wq, const void *wk, const void *wv, const void *widx,
                    const float *x, float *yq, float *yk, float *yv, float *yidx, cudaStream_t s);
int qfd_qsa_inproj(const void *wq, const void *wk, const void *wv,
                   const float *x, float *yq, float *yk, float *yv, cudaStream_t s);
int qfd_moe_inproj(const void *wrouter, const void *wshg, const void *wshu, const void *wgi,
                   const float *x, float *yr, float *yg, float *yu, float *ygi,
                   cudaStream_t s);
int qfd_out_proj(const void *w, const float *x, float *y, int out, int in, cudaStream_t s);
int qfd_lm_head(const void *w, const float *x, float *logits, cudaStream_t s);

// Timing counters (opt-in: QF_TIMING=1). CUDA events are recorded around each
// fused region on the decode stream; no per-token synchronization is added.
// qfd_timing_report folds and prints accumulated per-region GPU times.
void qfd_timing_report(FILE *fp);
void qfd_timing_reset(void);
void qfd_shutdown(void);

// Implemented in qf.cu (existing dispatcher: persistent cuBLASLt plan, else
// naive kernel). Used for per-projection fallback when a group is ineligible.
void qf_gemv_bf16(const void *W, const float *x, float *y, int out, int in,
                  cudaStream_t s);

#ifdef __cplusplus
}
#endif
