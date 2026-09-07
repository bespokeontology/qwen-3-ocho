// qf_attn_fused.h - fused QSA decode attention + device decode params
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Longest context the FUSED QSA decode attention can serve. Its per-block score
// buffer is shared memory sized at COMPILE time (scores[QF_ATTN_MAXPOS /
// QF_ATTN_NSPLIT]) while each block's chunk is computed from the RUNTIME token
// count, so a longer context writes past that array. Raising QF_ATTN_NSPLIT to
// compensate is not available either: k_attn_combine holds one float PER SPLIT
// in per-thread registers. qf.cu carries its own MAXPOS (the model's 262144
// context) - these are different numbers and must not be confused, which is
// exactly what happened before this constant existed.
#define QF_ATTN_NSPLIT 16
#define QF_ATTN_MAXPOS 8192

typedef struct {
    int      token;
    int      pos;
    uint32_t seq;
    uint32_t flags;
} QfDecodeParams;

int  qf_attn_init(void);
void qf_attn_shutdown(void);

void qf_attn_qsa_layer(float *q6144, float *k512, float *v512,
                       __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                       const void *q_norm, const void *k_norm, const float *inv_freq,
                       const QfDecodeParams *params, const uint32_t *qsa_mask,
                       float *attn_out, cudaStream_t s);
// M-row form: rows are independent sequences (own q/k/v slice at row stride,
// own KV cache at kvspan elements, own params[row]); the row is a grid
// dimension, so it is 3 launches per layer for all M rows. No QSA mask (decode).
#define QF_ATTN_MAXM 16
int  qf_attn_qsa_layer_Mgrid(float *q6144, float *k512, float *v512,
                             __nv_bfloat16 *kc, __nv_bfloat16 *vc, size_t kvspan,
                             const void *q_norm, const void *k_norm, const float *inv_freq,
                             const QfDecodeParams *params, float *attn_out, int M,
                             cudaStream_t s);
// Chunk-causal form for the prefill chunk: T consecutive positions of ONE
// sequence (params[t].pos = pos0 + t), all sharing the sequence's KV cache.
// Prep writes every key/value first; each query row then attends [0, pos_t].
// Dense (no QSA mask), exactly what the per-token chunk path computed.
int  qf_attn_qsa_chunk(float *q6144, float *k512, float *v512,
                       __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                       const void *q_norm, const void *k_norm, const float *inv_freq,
                       const QfDecodeParams *params, const uint32_t *mask, int mask_words,
                       const int *list, const int *nlist, int list_stride,
                       float *attn_out, int T, long pos0, cudaStream_t stream);
void qf_attn_qsa_layer_list(float *q6144, float *k512, float *v512,
                            __nv_bfloat16 *kc, __nv_bfloat16 *vc,
                            const void *q_norm, const void *k_norm, const float *inv_freq,
                            const QfDecodeParams *params, const int *list, const int *nlist,
                            float *attn_out, cudaStream_t stream);
