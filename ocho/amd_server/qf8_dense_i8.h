// qf8_dense_i8.h - gfx906 M=1 int8 dense projections for the Qwen3.8 AMD
// prefix (GLM W8 form: int8 [rows][K] + fp32 per-row scale, X8 activations,
// v_dot4_i32_i8, 16-byte nontemporal weight loads, no LDS staging).
//
// Covers the per-layer dense GEMVs on the M=1 path that were BF16 and ran at
// 260-360 GB/s: GDN in_proj (qkv 10240 + z 6144 + a 48 + b 48 rows, K 2560,
// ONE launch), GDN out_proj (2560 x 6144), router (512 x 2560), shared expert
// gate/up/gate_inp (640/640/1 x 2560) and down (2560 x 640). The int8 copies
// are built ONCE at layer load from the resident bf16 tensors (per-row absmax,
// GLM's k_w8_quant); the bf16 tensors stay for the M-row / single-row paths.
//
// Geometry: LPR lanes per row, 16 bytes per lane per step, so a row of K
// bytes is K/(16*LPR) steps: K=2560 -> LPR 32 (5 steps), K=6144 -> LPR 32
// (12 steps), K=640 -> LPR 8 (5 steps). Block 256 = 256/LPR rows. Up to 4
// matrices sharing the same activation batch on blockIdx.y.
#pragma once
#include "qf8_dense.h"

typedef struct { int8_t *w; float *s; int rows, K; } Qf8I8Mat;

#ifdef __cplusplus
extern "C" {
#endif
// bf16 [rows][K] on device -> int8 + per-row scale (allocates m->w / m->s).
int  qf8_i8_quant_bf16(const uint16_t *W, int rows, int K, Qf8I8Mat *m, hipStream_t s);
// Same, from a HOST bf16 matrix (mmap'd store) through a staging buffer: the bf16
// copy never lands on the device (the 1.27 GB lm_head on the head card).
int  qf8_i8_quant_bf16_host(const uint16_t *Whost, int rows, int K, Qf8I8Mat *m, hipStream_t s);
// fp32 x[n] -> X8 (q int8[n], s fp32[n/32]); n % 32 == 0.
void qf8_i8_x8(const float *x, int8_t *q, float *sc, int n, hipStream_t s);
// y[i][rows_i] = M_i . x  for up to 4 int8 matrices sharing x (X8 form).
// act[i]: 0 = none, 1 = silu(v/4) (hc down), 2 = 2*sigmoid(v/4) (hc inject).
typedef struct { const Qf8I8Mat *m[4]; float *y[4]; int act[4]; int n; } Qf8I8Multi;
void qf8_i8_gemv_multi(const Qf8I8Multi *mm, const int8_t *xq, const float *xs, hipStream_t s);
// CHUNK (rows) forms: xq/xs are [T][K] X8 rows; y[i] is [T][rows_i]. K in {2560, 6144, 640, 320, 10240}.
int  qf8_i8_gemv_rows(const Qf8I8Multi *mm, const int8_t *xq, const float *xs, int T, hipStream_t s);
int  qf8_i8_hc_rows(float *normed, float *hc_d, float *mixed, float *inj, float *R,
                    const uint16_t *wn, const Qf8I8Mat *down, const Qf8I8Mat *inject, const Qf8I8Mat *up,
                    int8_t *n8q, float *n8s, int8_t *d8q, float *d8s, float *hcy,
                    const float *y_prev, const float *inj_prev, int8_t *xq_out, float *xs_out, int T, hipStream_t s);
int  qf8_i8_shexp_rows(const Qf8I8Mat *gate, const Qf8I8Mat *up, const Qf8I8Mat *gate_inp, const Qf8I8Mat *down,
                       const int8_t *xq, const float *xs, float *eg, float *eu, float *g1, float *ed,
                       int8_t *e8q, float *e8s, float *y2560, int T, hipStream_t s);
// Shared expert, M=1: gate/up/gate_inp (one launch) -> silu*up -> X8 -> down
// -> y2560 += sigmoid(gate_inp . x) * down. eg/eu: [640] fp32 scratch,
// g1: 1 float scratch, ed: [2560] fp32 scratch, e8q/e8s: X8 scratch for eg.
// Hyper-connection pre-mixer, M=1, int8: normed = zero-centered grouped RMSNorm(R)
// (fp32 out + X8 out), d = silu(down.n/4), inj = 2*sigmoid(inject.n/4) (ONE launch),
// up: mixed[j] = 1/4 sum_c sigmoid(up[c*2560+j].d) * n[c*2560+j].
// Scratch: n8q/n8s X8 of normed (10240 int8 + 320 fp32), d8q/d8s X8 of d (320 + 10),
// hcy fp32 [10240] (the up outputs before the sigmoid mix).
// y_prev/inj_prev (optional): the previous mixer's stream inject R += inj[c]*y is
// applied inside the norm kernel (R updated in place). xq_out/xs_out: X8(mixed).
void qf8_i8_hc_m1(float *normed, float *hc_d, float *mixed, float *inj, float *R,
                  const uint16_t *wn, const Qf8I8Mat *down, const Qf8I8Mat *inject,
                  const Qf8I8Mat *up, int8_t *n8q, float *n8s, int8_t *d8q, float *d8s,
                  float *hcy, const float *y_prev, const float *inj_prev,
                  int8_t *xq_out, float *xs_out, hipStream_t s);
// M=1 int8 twin of qf8_qsa_mixer (qf8_qsa.hip): int8 q/k/v/indexer and o projections.
// x8q/x8s: X8 scratch for mixed (2560 + 80), a8q/a8s: X8 scratch for attn_out (6144 + 192).
void qf8_qsa_mixer_i8(const Qf8AttnW *w, const Qf8I8Mat *iq, const Qf8I8Mat *ik,
                      const Qf8I8Mat *iv, const Qf8I8Mat *iidx, const Qf8I8Mat *io,
                      const float *mixed, int8_t *x8q, float *x8s, int8_t *a8q, float *a8s,
                      qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos,
                      float *q12288, float *k512, float *v512, float *idx640,
                      float *attn_out, const float *inv_freq,
                      const float *inv_freq_idx, float *y2560, int quant_x, float *attn_part,
                      hipStream_t s);
#define QF8_FLASH_PART_FLOATS (24 * 8 * (256 + 2))   // attn_part scratch: heads x chunks x (m, l, acc)
// CHUNK form of the GDN mixer's recurrent part (qf8_gdn_chunk.hip): conv + chunked delta
// rule + gated norm over T positions, in place of T per-position recurrence launches.
int  qf8_gdn_chunk(const Qf8GdnW *w, float *gdnS, float *convring, float *qkv_raw, const float *z6144,
                   const float *a48, const float *b48, float *out6144, int T, hipStream_t s);
// CHUNK form of the QSA mixer (T positions, batched projections, per-position attention).
int qf8_qsa_chunk2(const Qf8AttnW *w, const Qf8I8Mat *iq, const Qf8I8Mat *ik,
                   const Qf8I8Mat *iv, const Qf8I8Mat *iidx, const Qf8I8Mat *io,
                   const int8_t *x8q, const float *x8s, int8_t *a8q, float *a8s, int T,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos0,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq, const float *inv_freq_idx,
                   float *y2560, float *blk_score_rows, uint8_t *mask_rows, int *list_rows,
                   int *nsel_rows, int nb_max, float *part_rows, hipStream_t s);
void qf8_qsa_chunk(const Qf8AttnW *w, const float *mixed, int T,
                   qf8_kv_t *kc, qf8_kv_t *vc, Qf8QsaIdx *ix, long pos0,
                   float *q12288, float *k512, float *v512, float *idx640,
                   float *attn_out, const float *inv_freq,
                   const float *inv_freq_idx, float *y2560, float *attn_part, hipStream_t s);
void qf8_i8_shexp_m1(const Qf8I8Mat *gate, const Qf8I8Mat *up, const Qf8I8Mat *gate_inp,
                     const Qf8I8Mat *down, const int8_t *xq, const float *xs,
                     float *eg, float *eu, float *g1, float *ed, int8_t *e8q, float *e8s,
                     float *y2560, hipStream_t s);
#ifdef __cplusplus
}
#endif
