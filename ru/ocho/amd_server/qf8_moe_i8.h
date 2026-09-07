// qf8_moe_i8.h - gfx906 M=1 routed-expert path on a ONE-TIME-PREPARED int8
// representation (the GLM W8 form ported to the Qwen expert shapes).
//
// Why: gfx906 has no FP4/FP8 execution. The NVFP4 kernels spend ~12 VALU ops
// per weight on nibble/E2M1/UE4M3 decoding and measured 777 us (gate/up) +
// 615 us (down) per layer for 24.6 MB of weights - ALU-bound at ~20 GB/s.
// v_dot4_i32_i8 does 4 weights per VALU op, so int8 weights (2x the bytes,
// ~1/50 the ALU) run at the HBM roofline like the GLM W8 GEMVs on these cards.
//
// Representation (built on device at server init from the resident NVFP4
// slots; slot == expert, fully resident):
//   i8_gate, i8_up : [slot][640][2560] int8, row-major, ld = 2560
//   i8_down        : [slot][2560][640] int8, ld = 640
//   i8s_*          : [slot][rows] fp32 per-row scale (row absmax / 127);
//                    the NVFP4 group scales and weight_scale_2 are folded in.
// Activations are int8 blocks of 32 with one fp32 scale (X8), quantized by a
// tiny kernel once per layer (x) and once per layer for the 10 hidden rows.
//
// Kernels (M=1, exact Qwen shapes, nothing generic):
//   k_i8_x8quant       fp32[n] -> int8 q[n] + fp32 s[n/32]   (n % 256 == 0)
//   k_i8_gateup_all    grid (80, nexp) x 256: half-wave per row, 5 x 16-byte
//                      nontemporal loads per matrix per lane, dot4, no LDS.
//   k_i8_downacc_all   grid 80 x 256: 32 rows x 8 lanes, loops the experts in
//                      fixed order, router weight folded in, y SEEDED.
//   k_nvfp4_to_i8      one-time conversion, grid (rows, slots).
#pragma once
#include "qf_nvfp4_wave64.h"

#ifdef __cplusplus
extern "C" {
#endif
// Build the int8 arenas of a fully resident layer cache from its NVFP4 slots
// (c->i8_* must be allocated; s2*_dev must be uploaded). Stream-ordered.
int  qf8_i8_build(Qf4ExpCache *c, hipStream_t s);
// Convert ONE expert: NVFP4 slot src_slot -> int8 slot dst_slot, using the
// scale_2 of `expert` (ring loader, QF_M1_INT8_ONLY=1).
int  qf8_i8_build_slot(Qf4ExpCache *c, int src_slot, int dst_slot, int expert, hipStream_t s);
// gate/up for the top-k experts of ONE row: x[2560] fp32 -> hidden[nexp][640] fp32.
// x8q/x8s: X8 scratch for x (2560 int8 + 80 fp32).
// quant_x = 0: x8q/x8s already hold X8(x) (produced by the router / hc combine).
void qf8_i8_gateup_m1(const Qf4ExpCache *c, const float *x, int8_t *x8q, float *x8s,
                      float *hidden, const int *sel_ids, int nexp, int quant_x, hipStream_t s);
// down + router-weighted accumulate: hidden[nexp][640] -> y[2560] (seeded).
// h8q/h8s: X8 scratch for the hidden rows (nexp*640 int8 + nexp*20 fp32).
void qf8_i8_down_m1(const Qf4ExpCache *c, const float *hidden, int8_t *h8q, float *h8s,
                    const float *wts, const int *sel_ids, float *y, int nexp, hipStream_t s);
// CHUNK (prefill) form: T token rows with distinct routing, inverted on device into a
// CSR (qf5_m8_build_csr: exp_slot/exp_ptr/pair_tok/pair_wt, counts[0] = n_exp). Every
// expert weight row is read ONCE and dotted against all tokens routed to it (<= 16).
// xq/xs: X8 of the T activation rows [T][2560]; hidden: [pairs][640] fp32; h8q/h8s: X8
// scratch for n_pairs_max * 640; y: [T][2560] fp32, ZEROED by the caller (atomic adds).
#define QF8_I8_CSR_TX 8     // tokens per LDS tile (20 KB + scales) -> 2 blocks/CU
void qf8_i8_csr_build(const int *sel, const float *wt, int T, int *exp_slot, int *exp_ptr,
                      int *pair_tok, float *pair_wt, int *counts, hipStream_t s);
void qf8_i8_experts_chunk(const Qf4ExpCache *c, const int8_t *xq, const float *xs,
                          float *hidden, int8_t *h8q, float *h8s,
                          const int *exp_slot, const int *exp_ptr, const int *pair_tok,
                          const float *pair_wt, const int *n_exp_dev, int n_exp_max,
                          int n_pairs_max, const int *sel, float *part, int T, float *y, hipStream_t s);
#ifdef __cplusplus
}
#endif
