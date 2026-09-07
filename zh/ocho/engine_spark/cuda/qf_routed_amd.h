// qf_routed_amd.h - Spark-side feature gate + glue for the AMD routed-expert
// tier. This is what the qf.cu MoE fork calls. When QF_ROUTED_AMD=1 the routed
// branch is served by the four-MI50 pool over the wire (qf_moe_handoff.h);
// otherwise the existing local nvfp4_gemv_grouped path runs unchanged.
//
// Env:
//   QF_ROUTED_AMD=1            enable the offload (default off -> local path)
//   QF_ROUTED_AMD_HOST=a.b.c.d AMD box address (default 127.0.0.1)
//   QF_ROUTED_AMD_PORT=n       server port (default 5577)
//
// submit()/wait() are split so Spark's shared/attention branch overlaps the
// AMD routed branch: the fork submits, computes shared locally, then waits.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 1 if QF_ROUTED_AMD=1 and a connection is (or becomes) available. Cached.
int  qf_routed_amd_enabled(void);

// Non-blocking beyond the wire send. sel/wt are [M*K] host; x is [M*NEMBD]
// host (the MoE input, ctx.mixed copied D2H). Returns 0 on success; on any
// wire error returns non-zero and disables the offload for the rest of the
// run (caller must fall back to the local routed path -> dead-arm safety).
int  qf_routed_amd_submit(int il, long long pos, int M, int K,
                          const int *sel, const float *wt, const float *x);

// Collect the routed output y [M*NEMBD] host (H2D into ctx.y2560 by caller).
int  qf_routed_amd_wait(int M, float *y);

#ifdef __cplusplus
}
#endif
