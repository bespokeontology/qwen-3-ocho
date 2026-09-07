// qf_fp4tc.h - Native GB10 (sm_121a) block-scaled FP4 tensor-core MoE backend.
//
// Replaces the per-expert scalar NVFP4 GEMV loop in qf_decode_step with two
// grouped tensor-core kernels per layer built on the sm_120a/sm_121a native
// instruction:
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
//     .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//
// Contract with the rest of the engine:
//   - Weights stay resident in the existing per-layer slot slabs
//     (QfLayer.exp_gate/exp_up/exp_down + exp_scale*), repacked ONCE on GPU at
//     slot-fill time (checkpoint interleaved {j, j+8} nibble order -> native
//     adjacent-pair {2i, 2i+1} order). Never per token, never on CPU.
//   - Routing stays fully device-resident: the kernels read sel_dev,
//     route_slot_dev, s2_*_dev, wts_dev and spin on slot_ready_dev doorbells
//     exactly like k_nvfp4_gemv_slot did (bounded spin -> route_err latch).
//   - Math contract preserved: y2560 += sum_k wts[k] * s2d_e *
//     Wdown_e @ (silu((Wgate_e @ x) * s2g_e) * ((Wup_e @ x) * s2u_e)),
//     fp32 accumulation in the tensor core, no weight renormalization
//     (norm_topk_prob = false is honored by using wts_dev as-is).
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "qwenflash.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static process-wide capability decision: device is sm_120a/sm_121a-class
// (compute 12.x) and QF_FP4TC != 0. Memoized on first call. This is the flag
// the load/miss-service paths consult for the one-time repack; it is decided
// before any weights are repacked and never changes afterwards.
int  qf_fp4tc_supported(void);

// Workspace init (once, from qf_forward_init, AFTER model load). Returns 0 on
// success or when unsupported (the legacy NVFP4 GEMV loop then runs and no
// repack has happened). Returns -1 on allocation failure: callers must treat
// that as a fatal startup error, because supported() may already have caused
// weights to be repacked.
int  qf_fp4tc_init(void);
void qf_fp4tc_shutdown(void);

// supported() && workspace ready. This is the decode-time gate.
int  qf_fp4tc_enabled(void);

// One-time in-place nibble repack of NVFP4 weight slabs, checkpoint order ->
// native mma.mxf4nvf4 operand order. NOT idempotent: run exactly once per
// slot fill (miss service) or once per slab (full-resident load). Scale
// tensors need no repack; the kernels read the checkpoint [row][K/16] UE4M3
// layout directly.
void qf_fp4tc_repack_slab(void *w, size_t bytes_per_expert, int nslots,
                          cudaStream_t s);
void qf_fp4tc_repack_slot3(void *gate, void *up, void *down,
                           size_t bytes_per_expert, int slot, cudaStream_t s);

// Grouped top-10 routed-expert MoE for one layer. Four launches total:
//   1. quantize x (NEMBD fp32 -> NVFP4 + UE4M3 block scales, dynamic absmax)
//   2. grouped gate/up MMA for all 10 experts -> eg_all/eu_all (fp32)
//   3. prepare/quantize the routed hidden vector once per expert
//   4. grouped down MMA + route-weighted atomic accumulate into y
// x = ctx.mixed [NEMBD] fp32; y = ctx.y2560 [NEMBD] fp32, PRE-ZEROED by the
// caller; the kernel accumulates into it. All routing state stays on device.
void qf_fp4tc_moe(const QfLayer *L, const int *sel_dev, const float *wts_dev,
                  const float *x, float *y, int *route_err_dev,
                  cudaStream_t s);

// Timing hooks: with QF_FP4TC_TIMING=1 the three launches are bracketed by
// CUDA events on the decode stream; qf_fp4tc_timing_report prints accumulated
// per-stage averages (us) since the last report and resets the accumulators.
// Intended to be called by the host between tokens / at shutdown. With the
// env unset the hook is two integer compares on the host side.
void qf_fp4tc_timing_report(FILE *out);

#ifdef __cplusplus
}
#endif
