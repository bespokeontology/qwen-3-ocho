// FlashInfer (0.6.18, header-only C++) tensor-core prefill attention for the chunk pass.
// STOLEN BULK MATH (operator 09-06): Spark uses NVIDIA-native kernels; QSA semantics are kept by
// feeding FlashInfer's masked single-request prefill a bit mask built from our per-row selected-block
// masks: bit(i, j) = (j <= pos_i) && (block(j) selected for row i || block(j) == row i's own tail block).
// Dense rows (indexer count <= 512) carry all-ones block masks, so the same formula yields causal
// attention for them. The K/V cache is already in FlashInfer's NHD layout ([pos][NKV][HDIM] bf16).
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define FI_NHEAD 24
#define FI_NKV 2
#define FI_HDIM 256
#define FI_QGATE 512

__global__ void __launch_bounds__(256) k_fi_pack_q(const float *__restrict__ q6144, __nv_bfloat16 *__restrict__ qb, int T) {
    const int t = blockIdx.x, d = threadIdx.x;
    const float *qr = q6144 + (size_t)t * FI_NHEAD * FI_QGATE;
    __nv_bfloat16 *ob = qb + (size_t)t * FI_NHEAD * FI_HDIM;
    #pragma unroll 4
    for (int h = 0; h < FI_NHEAD; h++) ob[h * FI_HDIM + d] = __float2bfloat16(qr[h * FI_QGATE + d]);
}
// packed mask, flat bit index i*kv_len + j (FlashInfer: byte offset/8, bit offset%8); one thread per byte
__global__ void k_fi_mask(const uint32_t *__restrict__ idx_mask, int mw, long pos0, int T, int kv_len, uint8_t *__restrict__ out, size_t nbytes) {
    const size_t B = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (B >= nbytes) return;
    const size_t total = (size_t)T * kv_len;
    uint32_t byte = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        const size_t flat = B * 8 + k;
        if (flat >= total) break;
        const int i = (int)(flat / kv_len), j = (int)(flat - (size_t)i * kv_len);
        const long pos = pos0 + i;
        if ((long)j > pos) continue;
        const int b = j >> 2;
        const int sel = (idx_mask[(size_t)i * mw + (b >> 5)] >> (b & 31)) & 1u;
        if (sel || b == (int)(pos >> 2)) byte |= 1u << k;
    }
    out[B] = (uint8_t)byte;
}
__global__ void __launch_bounds__(256) k_fi_unpack_o(const __nv_bfloat16 *__restrict__ o, const float *__restrict__ q6144, float *__restrict__ attn_out, int T) {
    const int t = blockIdx.x, d = threadIdx.x;
    const __nv_bfloat16 *orow = o + (size_t)t * FI_NHEAD * FI_HDIM;
    const float *qr = q6144 + (size_t)t * FI_NHEAD * FI_QGATE;
    float *ao = attn_out + (size_t)t * FI_NHEAD * FI_HDIM;
    #pragma unroll 4
    for (int h = 0; h < FI_NHEAD; h++) {
        const float gate = qr[h * FI_QGATE + FI_HDIM + d];
        ao[h * FI_HDIM + d] = __bfloat162float(orow[h * FI_HDIM + d]) / (1.f + expf(-gate));
    }
}
static __nv_bfloat16 *g_fq = NULL, *g_fo = NULL; static size_t g_fq_rows = 0;
static uint8_t *g_fmask = NULL; static size_t g_fmask_n = 0;

extern "C" int qf_fi_attn_chunk(const float *q6144, const __nv_bfloat16 *kc, const __nv_bfloat16 *vc,
                                const uint32_t *idx_mask, int mw, long pos0, int T, float *attn_out, cudaStream_t s) {
    const long kv_len_l = pos0 + T;
    if (T < 1 || kv_len_l > 0x7fffffffL) return -1;
    const int kv_len = (int)kv_len_l;
    if ((size_t)T > g_fq_rows) {
        if (g_fq) cudaFree(g_fq); if (g_fo) cudaFree(g_fo);
        if (cudaMalloc(&g_fq, (size_t)T * FI_NHEAD * FI_HDIM * 2) != cudaSuccess) { g_fq = NULL; g_fq_rows = 0; return -1; }
        if (cudaMalloc(&g_fo, (size_t)T * FI_NHEAD * FI_HDIM * 2) != cudaSuccess) { g_fo = NULL; g_fq_rows = 0; return -1; }
        g_fq_rows = T;
    }
    const size_t nbytes = ((size_t)T * kv_len + 7) / 8;
    if (nbytes > g_fmask_n) {
        if (g_fmask) cudaFree(g_fmask);
        if (cudaMalloc(&g_fmask, nbytes + 64) != cudaSuccess) { g_fmask = NULL; g_fmask_n = 0; return -1; }
        g_fmask_n = nbytes;
    }
    k_fi_pack_q<<<T, FI_HDIM, 0, s>>>(q6144, g_fq, T);
    k_fi_mask<<<(unsigned)((nbytes + 255) / 256), 256, 0, s>>>(idx_mask, mw, pos0, T, kv_len, g_fmask, nbytes);
    using namespace flashinfer;
    using Params = SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
    Params p(g_fq, (__nv_bfloat16 *)kc, (__nv_bfloat16 *)vc, g_fmask, g_fo, nullptr, nullptr,
             FI_NHEAD, FI_NKV, (uint32_t)T, (uint32_t)kv_len,
             FI_NHEAD * FI_HDIM, FI_HDIM, FI_NKV * FI_HDIM, FI_HDIM, FI_HDIM,
             -1, 0.f, 1.f / 16.f, 1.f, 1e4f);
    using Variant = DefaultAttention<true, false, false, false>;
    cudaError_t e = SinglePrefillWithKVCacheDispatched<FI_HDIM, FI_HDIM, PosEncodingMode::kNone, false, MaskMode::kCustom, Variant, Params>(p, nullptr, s);
    if (e != cudaSuccess) { fprintf(stderr, "fi attn: %s\n", cudaGetErrorString(e)); return -1; }
    k_fi_unpack_o<<<T, FI_HDIM, 0, s>>>(g_fo, q6144, attn_out, T);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
