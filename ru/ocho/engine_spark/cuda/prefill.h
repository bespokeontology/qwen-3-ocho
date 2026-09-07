// prefill.h - chunked prefill path (T tokens at once), batch 1 only.
// Correctness contract: final states (GDN S, conv ring, KV cache contents implied by
// outputs) must match running the qf.cu decode kernels T times on the same inputs.
#pragma once
#include <cuda_runtime.h>

#define PF_CHUNK 32     // GDN intra-chunk length (fla-style chunking)
#define PF_MAX_T 512

#ifdef __cplusplus
extern "C" {
#endif

// one-time init: cublas handle + dynamic smem opt-in. Safe to call repeatedly.
// NOTE ON THE _f32w SUFFIX. These three take their WEIGHTS as `const float *`.
// Production weights are BF16, or FP8 slabs once g_hc_fp8/g_qfd_fp8 is set, so
// casting a real weight pointer to `const float *` reads twice the bytes - and
// for an FP8 slab the scales sit BEFORE the base pointer, so it reads off the
// front of the allocation. That cost a full debugging cycle on 2026-08-29 when
// the batched speculative forward called them on real weights: an illegal
// access from qf_hc_prefill, then silently wrong GDN. They are exercised only
// by cuda/test_prefill.cu, which builds f32 test weights. The suffix is the
// warning: if your weights came from the checkpoint, these are the wrong
// functions - use the trunk's own BF16/FP8 kernels one token at a time.

int qf_prefill_init(void);

// 1. chunked parallel delta rule (GDN), SEMANTICS recurrence with in-kernel q/k l2norm:
//      per head h (48 v-heads, k-head = h/3), per token t:
//        lam  = -exp(A_log[h]) * softplus(a[t][h] + dt_bias[h])   (multiplicative decay,
//              same as k_gdn_decode's S *= g)
//        beta = sigmoid(b[t][h])
//        S   *= lam;  delta = (v - S^T k) * beta;  S += k outer delta;  o = S^T (q/sqrt(128))
//    Chunked form (chunk = PF_CHUNK): intra-chunk UT transform of (I + A) with
//    cumulative decay products, inter-chunk state passing via S.
// qkv: [T][10240] fp32, post conv+silu (q[2048] | k[2048] | v[6144]); a,b: [T][48] fp32
// S: [48][128][128] fp32, initial state in / final state out. out: [T][48][128] fp32.
int qf_gdn_prefill_f32w(const float *qkv, const float *a, const float *b,
                   const float *A_log, const float *dt_bias,
                   float *S, float *out, int T, cudaStream_t st);

// 2. causal SDPA, T<=512, 24 q heads / 2 kv heads (kh = h/12), head_dim 256, fp32 softmax.
// q: fp32, token-major [T][24][qstride] (qstride = 512 when gate-interleaved as in
// decode's q6144 buffer, else 256). kc/vc: bf16 [T][2][256] (post rope/norm, as cached).
// mask: [T] int, mask[j] == -1 -> key j ignored (QSA selected-token mask). out: [T][24][256].
int qf_attn_prefill(const float *q, int qstride, const void *kc, const void *vc,
                    const int *mask, float *out, int T, cudaStream_t st);

// 3. causal conv1d over T tokens, kernel 4, dilation 1, silu. raw: [T][10240] in/out,
// ring: [3][10240] raw pre-conv history (same layout/update rule as k_conv_step).
int qf_conv_prefill_f32w(float *raw, float *ring, const float *w, int T, cudaStream_t st);

// 4. row-batched GEMM wrappers (cublas): Y[T][out] = X[T][in] * W[out][in]^T.
// W row-major [out][in]; X, Y row-major, fp32 in/out. bf16 variant for W bf16.
int qf_gemm_bf16(const void *W, const float *X, float *Y, int out, int in, int T,
                 cudaStream_t st);
int qf_gemm_f32(const float *W, const float *X, float *Y, int out, int in, int T,
                cudaStream_t st);

// hyper-connection front-end, batched over T rows:
//   normed = rmsnorm_rows(R, w)   (group 2560, zero-centered weight)
//   d      = silu(down(normed)/4)              [T][320]
//   mixed  = mean_c sigmoid(up(d)) * normed    [T][2560]
//   inj    = 2*sigmoid(inject(normed)/4)       [T][4] (skipped if w_inject == NULL)
// workspaces (device): w_normed [T][10240], w_d320 [T][320], w_up [T][10240].
int qf_hc_prefill_f32w(const float *R, const float *w_norm, const float *w_down,
                  const float *w_up, const float *w_inject,
                  float *mixed, float *inj,
                  float *w_normed, float *w_d320, float *w_up10240,
                  int T, cudaStream_t st);

#ifdef __cplusplus
}
#endif
