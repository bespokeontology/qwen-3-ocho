// qf.cu - GPU-native decode path, coherent v1
#include "../qwenflash.h"
#include "../qf_fp4tc.h"
#include "qf_dense.h"
#include "qf_attn_fused.h"
// QF_QSA_INDEX=0: attend densely (masks ignored) while still updating the indexer state - bisection only.
static int qsa_index_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_QSA_INDEX"); v = (e && e[0] == '0') ? 0 : 1; } return v; }

extern "C" int qf_qsa_index_rows(float *idx640, int T, const QfDecodeParams *params, const void *idx_qnorm, const void *idx_knorm,
                                 const float *inv_freq_idx, float *pool_sum, float *pool_key, int *pool_cnt,
                                 float *blk_score, int nb_max, uint32_t *mask, int mw, int *list, int *nlist, long pos0_host, cudaStream_t s);
#include "qf_decode_graph.h"
#include "../planner.h"   // QF_SPARK_MODE_FULL

// qf_fp4mma.cu - native block-scaled FP4 MMA expert GEMV (opt-in: QF_FP4MMA=1)
extern "C" int  qf_fp4mma_available(void);
extern "C" void qf_fp4mma_quant_launch(const float *x, int K, uint8_t *xq, uint8_t *xs, cudaStream_t s);
extern "C" void qf_fp4mma_gemv_launch(const void *W, size_t w_stride, const void *S, size_t s_stride,
                                      const int *sel, const int *slots, const float *s2tab,
                                      const void *xq, const void *xs, float *y_all,
                                      int rows, int K, int x_per_expert, cudaStream_t s);
static int qf_fp4mma_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QF_FP4MMA");
        v = (e && e[0] == '1') ? (qf_fp4mma_available() ? 1 : 0) : 0;
        if (v) fprintf(stderr, "qf_fp4mma: native block-scaled FP4 MMA expert path ENABLED\n");
    }
    return v;
}
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NEMBD
#ifndef NEMBD
#define NEMBD 2560
#endif
#endif
#ifndef NHEAD
#define NHEAD 24
#endif
#ifndef NKV
#ifndef NKV
#define NKV 2
#endif
#endif
#ifndef HDIM
#define HDIM 256
#endif
#ifndef NLAYER
#ifndef NLAYER
#define NLAYER 48
#endif
#endif
#ifndef NEXP
#define NEXP 512
#endif
#ifndef NEXPUSED
#ifndef NEXPUSED
#define NEXPUSED 10
#endif
#endif
#ifndef NFF
#define NFF 640
#endif
#ifndef HCC
#ifndef HCC
#define HCC 4
#endif
#endif
#ifndef HCL
#ifndef HCL
#define HCL 320
#endif
#endif
#ifndef GDN_KH
#ifndef GDN_KH
#define GDN_KH 16
#endif
#endif
#ifndef GDN_VH
#define GDN_VH 48
#endif
#ifndef GDN_KD
#ifndef GDN_KD
#define GDN_KD 128
#endif
#endif
#ifndef GDN_VD
#define GDN_VD 128
#endif
#ifndef DINN
#ifndef DINN
#define DINN 10240
#endif
#endif
#ifndef GDN_KV_RATIO
#define GDN_KV_RATIO (GDN_VH / GDN_KH)
#endif
#ifndef GDN_VDIM
#define GDN_VDIM (GDN_VH * GDN_VD)
#endif
#ifndef QKVDIM
#define QKVDIM (NHEAD * QGATE)
#endif
#ifndef KVDIM
#define KVDIM (NKV * HDIM)
#endif
#ifndef ATTN_OUT_DIM
#define ATTN_OUT_DIM (NHEAD * HDIM)
#endif
#ifndef DTRANK
#define DTRANK 48
#endif
#ifndef NVOCAB
#ifndef NVOCAB
#define NVOCAB 248320
// Rows per batched step (and the spec-verify T cap). 16: the Spark step is
// launch-bound, so its cost barely moves with rows - 16 rows per graph launch
// is the freeze's 117 tok/s configuration (two such slots in flight).
#define QF_SPEC_MAXT 16
#endif
#endif
#define KVREP (NHEAD / NKV)
#ifndef MAXPOS
#ifndef MAXPOS
// The MODEL's context limit. It is NOT what this engine can serve: the fused
// QSA attention tops out at QF_ATTN_MAXPOS (qf_attn_fused.h). qf_attn_fused.cu
// used to carry its own `#define MAXPOS 8192` under the same name in a separate
// translation unit, so this file believed 262144 while the attention kernel was
// sized for 8192 - and nothing said so. The default below is now what the
// engine can actually serve; asking for more is refused at init, not corrupted.
#define MAXPOS 262144
#endif
#endif
// Runtime context cap (server mode): QF_MAX_CONTEXT overrides the compile-time
// MAXPOS. Resolved once at qf_forward_init; KV caches and the score buffer are
// sized from it. Must be >= the server's --max-context (the server exports the
// same value before model load, and planner.cpp budgets from it).
static long g_maxpos = 0;
static long qf_maxpos(void) {
    if (g_maxpos <= 0) {
        const char *e = getenv("QF_MAX_CONTEXT");
        long v = (e && atol(e) > 0) ? atol(e) : (long)QF_ATTN_MAXPOS;
        g_maxpos = v;
    }
    return g_maxpos;
}
extern "C" long qf_context_cap(void) { return qf_maxpos(); }   // for the driver's generation cap
#ifndef QGATE
#define QGATE 512   // q_proj row width per head (256 q + 256 gate)
#endif

#define CHK(x) do { cudaError_t e=(x); if (e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return -1;} } while (0)

__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }
__device__ __forceinline__ float ue4m3f(uint8_t b) {
    uint32_t e = b >> 3, m = b & 7;
    if (e == 0) return ldexpf((float)m, -9);
    if (e == 15 && m == 7) return INFINITY;
    return ldexpf((float)(8 + m), (int)e - 10);
}
__device__ __forceinline__ float e2m1f(uint8_t n) {
    const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (n & 8) ? -t[n & 7] : t[n & 7];
}

__global__ void k_bf16_to_f32(float *out, const __nv_bfloat16 *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}
__global__ void k_repeat4(float *R, const float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCC * NEMBD) R[i] = x[i % NEMBD];
}
__global__ void k_embed_dev(float *x, float *R, const __nv_bfloat16 *embd,
                            const QfDecodeParams *__restrict__ params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NEMBD) return;
    float value = __bfloat162float(embd[(size_t)params->token * NEMBD + i]);
    x[i] = value;
    #pragma unroll
    for (int c = 0; c < HCC; c++) R[c * NEMBD + i] = value;
}

// Embedding of the current token into a plain [NEMBD] vector (no hc replication).
// Used by the MTP draft head, whose fusion needs the raw embedding.
__global__ void k_embed_1(float *x, const __nv_bfloat16 *embd,
                          const QfDecodeParams *__restrict__ params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NEMBD) x[i] = __bfloat162float(embd[(size_t)params->token * NEMBD + i]);
}
__global__ void k_gemv_bf16(const __nv_bfloat16 *__restrict__ W, const float *__restrict__ x,
                            float *__restrict__ y, int out, int in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out) return;
    float acc = 0.f;
    for (int i = threadIdx.x; i < in; i += blockDim.x)
        acc += __bfloat162float(W[(size_t)row * in + i]) * x[i];
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    __shared__ float ws[4][4];   // [warp][row-in-block]
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid][threadIdx.y] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < 4; w++) t += ws[w][threadIdx.y];
        y[row] = t;
    }
}
// ---------------- tensor-core BF16 GEMV (GB10 / sm_121a, cuBLASLt) ----------------
// k_gemv_bf16 above stays as the fallback for shapes without an Lt plan. The Lt
// path computes y[out] = W[out,in] * x[in] as a skinny col-major GEMM:
//   W row-major [out,in] is the same memory as col-major A[in,out] (ld=in), and
//   y = op_T(A) * x  with alpha=1, beta=0, fp32 accumulate/output.
// The activation is converted f32 -> bf16 once per call into g_lt.xbf.
//
// Wave-2 (per-shape plan cache): every cuBLASLt object a GEMV needs -- matmul
// descriptor, matrix layouts, heuristic-selected algorithm, and the algorithm's
// workspace -- is built ONCE per exact decode shape in qf_forward_init and kept
// in g_lt.plans. The per-token dispatch is a small table lookup plus the
// conversion kernel plus cublasLtMatmul: no descriptor/layout/preference is
// created or destroyed per GEMV and no heuristic query runs in the hot path.
//
// Stream binding: plans are bound to the decode stream captured at init
// (g_lt.stream). A GEMV arriving on any other stream takes the naive kernel, so
// per-plan workspaces never alias across streams. All decode call sites pass
// ctx.s, so this guard never fires in the engine as shipped.
//
// Workspace ownership: each plan owns exactly the workspace its algorithm
// requested (capped at QF_LT_WS_MAX via the heuristic preference), allocated at
// init and freed in qf_forward_shutdown. A plan whose setup fails is dropped
// (its shape uses the naive kernel); a plan whose matmul fails at runtime is
// marked dead and that shape permanently falls back.
#define QF_XBF_CAP 8192                    // max activation length converted to bf16
#define QF_GEMV_TC_MIN (1 << 20)           // min out*in elements to justify an Lt plan
#define QF_LT_WS_MAX ((size_t)32 << 20)    // per-plan workspace cap (splitK headroom)
#define QF_LT_MAX_PLANS 16                 // distinct decode shapes (see g_lt_shapes)

typedef struct {
    int out, in;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t layW, layX, layY;
    cublasLtMatmulAlgo_t algo;
    void *ws;             // owned by this plan; NULL when the algo needs none
    size_t ws_bytes;
    int ready;            // 1: usable; 0: runtime failure -> permanent fallback
} QfLtPlan;

static struct {
    cublasLtHandle_t h;
    __nv_bfloat16 *xbf;
    int xbf_cap;
    cudaStream_t stream;  // decode stream the plans are bound to
    QfLtPlan plans[QF_LT_MAX_PLANS];
    int nplans;
    int ok;
} g_lt;                   // static: zero-initialized

static int g_tc_enable = -1;   // -1: unset (read QF_GEMV_TC once), 0/1: off/on

// Every distinct (out, in) GEMV shape issued by qf_decode_step, written in the
// same macros the call sites use so -D overrides (e.g. tiny synth dims) track
// automatically. Duplicates are removed at plan build. Deliberately absent:
// DTRANK x NEMBD (48x2560, below QF_GEMV_TC_MIN) and 1 x NEMBD (shexp_gate_inp,
// out % 8 != 0) -- both always take the naive kernel.
static const struct { int out, in; } g_lt_shapes[] = {
    { DINN,     NEMBD },        // qkv (GDN layers)
    { GDN_VDIM, NEMBD },        // zgate
    { NEMBD,    GDN_VDIM },     // gdn_out / wo (NHEAD*HDIM == GDN_VDIM == 6144)
    { QKVDIM,   NEMBD },        // wq
    { KVDIM,    NEMBD },        // wk / wv / router (NEXP == KVDIM == 512)
    { NFF,      NEMBD },        // expert + shared-expert gate/up
    { NEMBD,    NFF },          // expert + shared-expert down
    { NVOCAB,   NEMBD },        // lm_head
};

__global__ void k_f32_to_bf16(__nv_bfloat16 *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

static QfLtPlan *lt_plan_find(int out, int in) {
    for (int i = 0; i < g_lt.nplans; i++)
        if (g_lt.plans[i].out == out && g_lt.plans[i].in == in)
            return &g_lt.plans[i];
    return NULL;
}

// Build one persistent plan: descriptors, layouts, algorithm, owned workspace.
// Returns 0 on success (p->ready set); on failure everything created so far is
// destroyed and the caller leaves no trace in the table.
static int lt_plan_build(QfLtPlan *p, int out, int in) {
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heur[4];
    int nres = 0, rc = -1;
    memset(p, 0, sizeof(*p));
    p->out = out;
    p->in = in;
    if (cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) goto done;
    {
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    }
    if (cublasLtMatrixLayoutCreate(&p->layW, CUDA_R_16BF, in, out, in) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatrixLayoutCreate(&p->layX, CUDA_R_16BF, in, 1, in) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatrixLayoutCreate(&p->layY, CUDA_R_32F, out, 1, out) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) goto done;
    {
        size_t wsmax = QF_LT_WS_MAX;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wsmax, sizeof(wsmax));
    }
    if (cublasLtMatmulAlgoGetHeuristic(g_lt.h, p->desc, p->layW, p->layX, p->layY, p->layY,
                                       pref, 4, heur, &nres) != CUBLAS_STATUS_SUCCESS || nres < 1)
        goto done;
    {
        int pick = -1;
        for (int i = 0; i < nres; i++)
            if (heur[i].workspaceSize <= QF_LT_WS_MAX) { pick = i; break; }
        if (pick < 0) goto done;
        p->algo = heur[pick].algo;
        p->ws_bytes = heur[pick].workspaceSize;
    }
    if (p->ws_bytes > 0 && cudaMalloc(&p->ws, p->ws_bytes) != cudaSuccess) {
        p->ws = NULL;
        p->ws_bytes = 0;
        goto done;
    }
    p->ready = 1;
    rc = 0;
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (rc != 0) {
        if (p->layY) { cublasLtMatrixLayoutDestroy(p->layY); p->layY = NULL; }
        if (p->layX) { cublasLtMatrixLayoutDestroy(p->layX); p->layX = NULL; }
        if (p->layW) { cublasLtMatrixLayoutDestroy(p->layW); p->layW = NULL; }
        if (p->desc) { cublasLtMatmulDescDestroy(p->desc); p->desc = NULL; }
    }
    return rc;
}

// Prebuild one plan per exact decode shape. Called once from qf_forward_init
// after the handle, xbf scratch, and decode stream exist. Per-shape failure is
// a per-shape fallback, never fatal.
static void lt_plans_init(void) {
    int n = (int)(sizeof(g_lt_shapes) / sizeof(g_lt_shapes[0]));
    for (int i = 0; i < n; i++) {
        int out = g_lt_shapes[i].out, in = g_lt_shapes[i].in;
        if ((in & 7) || (out & 7) || (int64_t)out * in < QF_GEMV_TC_MIN || in > g_lt.xbf_cap)
            continue;                        // not Lt-eligible: naive kernel handles it
        if (lt_plan_find(out, in)) continue; // duplicate shape
        if (g_lt.nplans >= QF_LT_MAX_PLANS) break;
        if (lt_plan_build(&g_lt.plans[g_lt.nplans], out, in) == 0)
            g_lt.nplans++;
        else
            fprintf(stderr, "qf: no Lt plan for GEMV %dx%d, using fallback kernel\n", out, in);
    }
}

static void gemv_bf16(const void *W, const float *x, float *y, int out, int in, cudaStream_t s) {
    if (g_tc_enable < 0) {
        const char *e = getenv("QF_GEMV_TC");
        g_tc_enable = (e && e[0] == '0') ? 0 : 1;
    }
    if (g_tc_enable && g_lt.ok && s == g_lt.stream) {
        QfLtPlan *p = lt_plan_find(out, in);
        if (p && p->ready) {
            const float alpha = 1.f, beta = 0.f;
            k_f32_to_bf16<<<(in + 255) / 256, 256, 0, s>>>(g_lt.xbf, x, in);
            if (cublasLtMatmul(g_lt.h, p->desc, &alpha, W, p->layW, g_lt.xbf, p->layX,
                               &beta, y, p->layY, y, p->layY,
                               &p->algo, p->ws, p->ws_bytes, s) == CUBLAS_STATUS_SUCCESS)
                return;
            // A plan that passed the heuristic failed at runtime: kill it and
            // recompute with the naive kernel (xbf is scratch, nothing to undo).
            p->ready = 0;
            fprintf(stderr, "qf: Lt GEMV %dx%d failed at runtime, using fallback kernel\n", out, in);
        }
    }
    dim3 blk(128, 4);
    k_gemv_bf16<<<(out + 3) / 4, blk, 0, s>>>((const __nv_bfloat16 *)W, x, y, out, in);
}
void qf_gemv_bf16(const void *W, const float *x, float *y, int out, int in, cudaStream_t s) {
    gemv_bf16(W, x, y, out, in, s);
}
__global__ void k_rmsnorm(float *out, const float *x, const __nv_bfloat16 *w, int n, float eps, int group) {
    const int glen = group ? group : n;
    const int g = group ? blockIdx.x * group : 0;
    if (g >= n) return;
    float acc = 0.f;
    for (int j = threadIdx.x; j < glen; j += blockDim.x) {
        float v = x[g + j];
        acc += v * v;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp_sum[8];
    __shared__ float inv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int wi = 0; wi < blockDim.x / 32; wi++) total += warp_sum[wi];
        inv = rsqrtf(total / glen + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < glen; j += blockDim.x) {
        const int i = g + j;
        out[i] = x[i] * inv * (1.f + __bfloat162float(w[i]));
    }
}
// M-row clone of k_rmsnorm (identical arithmetic; the row comes from the grid).
__global__ void k_rmsnorm_M(float *out, const float *x, const __nv_bfloat16 *w, int n, float eps, int group) {
    // M-row form: blockIdx.y = request row, [M][n]-contiguous in/out (W1 launch elimination)
    { const int mrow = blockIdx.y; out += (size_t)mrow * n; x += (size_t)mrow * n; }
    const int glen = group ? group : n;
    const int g = group ? blockIdx.x * group : 0;
    if (g >= n) return;
    float acc = 0.f;
    for (int j = threadIdx.x; j < glen; j += blockDim.x) {
        float v = x[g + j];
        acc += v * v;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp_sum[8];
    __shared__ float inv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int wi = 0; wi < blockDim.x / 32; wi++) total += warp_sum[wi];
        inv = rsqrtf(total / glen + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < glen; j += blockDim.x) {
        const int i = g + j;
        out[i] = x[i] * inv * (1.f + __bfloat162float(w[i]));
    }
}
__global__ void k_hc_down(const float *normed, const __nv_bfloat16 *w_down, float *d320) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (i >= HCL) return;
    float acc = 0.f;
    const __nv_bfloat16 *row = w_down + (size_t)i * (HCC * NEMBD);
    for (int j = lane; j < HCC * NEMBD; j += 32)
        acc = fmaf(__bfloat162float(row[j]), normed[j], acc);
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float v = acc / 4.f;
        d320[i] = v / (1.f + expf(-v));
    }
}
__global__ void k_hc_up(const float *normed, const __nv_bfloat16 *w_up, const float *d320, float *mixed,
                        const __nv_bfloat16 *w_inject, float *inj, int write_inject) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (i >= HCC * NEMBD) return;
    float acc = 0.f;
    const __nv_bfloat16 *row = w_up + (size_t)i * HCL;
    for (int l = lane; l < HCL; l += 32)
        acc = fmaf(__bfloat162float(row[l]), d320[l], acc);
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float m = sigmoidf_(acc);
        atomicAdd(&mixed[i % NEMBD], m * normed[i] / (float)HCC);
    }
    if (write_inject && i < HCC) {
        float a2 = 0.f;
        const __nv_bfloat16 *irow = w_inject + (size_t)i * (HCC * NEMBD);
        for (int j = lane; j < HCC * NEMBD; j += 32)
            a2 = fmaf(__bfloat162float(irow[j]), normed[j], a2);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a2 += __shfl_down_sync(0xffffffffu, a2, off);
        if (lane == 0) inj[i] = 2.f * sigmoidf_(a2 / (float)HCC);
    }
}

// E4M3 (OCP, bias 7) and UE4M3 decode helpers for FP8 dense slabs.
__device__ __forceinline__ float qf_e4m3_dec(uint8_t b) {
    uint32_t sg = (b & 0x80u) ? 0x80000000u : 0u;
    uint32_t e = (b >> 3) & 0xFu, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f * (sg ? -1.f : 1.f);
    return __uint_as_float(sg | ((e + 120u) << 23) | (m << 20));
}
__device__ __forceinline__ float qf_ue4m3_dec2(uint8_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}
// Slab layout from qfd_quant_fp8: [scales rows*(in/64) | pad->16 | E4M3 base].
static __device__ __forceinline__ const uint8_t *qf_fp8_scales(const uint8_t *w8, int rows, int in) {
    size_t n64 = (size_t)rows * (in >> 6);
    return w8 - (n64 + ((16 - (n64 & 15u)) & 15u));
}
// One BLOCK per output row instead of one warp.
//
// The warp-per-row form launched (HCL + 7) / 8 = 41 blocks for a 320-output,
// 10240-input GEMV, so each warp ground 320 iterations per lane and 41 blocks
// could not fill GB10. Measured 2026-08-28: 46.8 us x 97 calls/token =
// 4.54 ms/token at 83 GB/s, 30% of peak - the same occupancy starvation the
// routed experts had before grouping (which took them 30% -> 53%).
//
// 320 blocks x 256 threads: 40 elements per thread, then a block reduction.
// Arithmetic is unchanged, including the /4 and the silu on the output.
// Arm selector for the vectorized/scalar A/B. A __device__ variable, NOT a
// kernel argument or a host branch: decode replays a CAPTURED CUDA GRAPH, so
// an argument would be baked in at capture and a host branch would never be
// re-evaluated. Reading it inside the kernel lets one resident load produce
// interleaved samples of both arms - the only way to resolve a difference
// smaller than this engine's ~5-12% inter-load spread.
__device__ int d_hc_vec = 1;
__global__ __launch_bounds__(256)
void k_hc_down_fp8(const float *__restrict__ normed, const uint8_t *__restrict__ w8,
                   float *__restrict__ d320) {
    const int IN = HCC * NEMBD;
    const uint8_t *sc = qf_fp8_scales(w8, HCL, IN);
    const int i = blockIdx.x;
    if (i >= HCL) return;
    const uint8_t *row = w8 + (size_t)i * IN;
    const uint8_t *srow = sc + (size_t)i * (IN >> 6);
    const int tid = threadIdx.x;
    float acc = 0.f;
    // Vectorized inner loop. The scalar form read ONE BYTE per thread per
    // iteration (row[j], stride blockDim.x) plus a separate scale byte and a
    // separate float - about three memory instructions per FMA, 120 per thread
    // for 40 multiply-adds. nsys put this stage at 19% of the token while
    // moving 15% of dense's bytes, i.e. ~36% of dense's per-byte efficiency:
    // instruction-bound, not bandwidth-bound.
    //
    // Each thread now takes a CONTIGUOUS uchar4 of weights and the matching
    // float4 of activations, so a warp covers 128 B of weights and 512 B of
    // activations per step, and the scale is decoded ONCE per 64-element group
    // instead of re-read per element (j >> 6 was constant across a group but
    // reloaded every iteration because j strode by blockDim.x).
    //
    // IN is HCC*NEMBD = 10240, divisible by 4*256, so the vector path is exact
    // with no tail. The accumulation ORDER changes (thread t now owns
    // 4t..4t+3, +1024 stride, instead of t, +256), so results are not bitwise
    // equal to the scalar form - gated on greedy token identity, not on bits.
    if (d_hc_vec && (IN & (4 * 256 - 1)) == 0 && blockDim.x == 256) {
        const uchar4 *row4 = (const uchar4 *)row;
        const float4 *nrm4 = (const float4 *)normed;
        const int n4 = IN >> 2;                       // 4-byte groups
        for (int j4 = tid; j4 < n4; j4 += 256) {
            const uchar4 wq = row4[j4];
            const float4 nv = nrm4[j4];
            // 4 consecutive elements share a 64-element scale group unless the
            // group boundary falls inside them - it cannot, since 64 % 4 == 0.
            const float s = qf_ue4m3_dec2(srow[(j4 << 2) >> 6]);
            acc = fmaf(qf_e4m3_dec(wq.x) * s, nv.x, acc);
            acc = fmaf(qf_e4m3_dec(wq.y) * s, nv.y, acc);
            acc = fmaf(qf_e4m3_dec(wq.z) * s, nv.z, acc);
            acc = fmaf(qf_e4m3_dec(wq.w) * s, nv.w, acc);
        }
    } else {
        for (int j = tid; j < IN; j += blockDim.x)
            acc = fmaf(qf_e4m3_dec(row[j]) * qf_ue4m3_dec2(srow[j >> 6]), normed[j], acc);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float red[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) t += red[w];
        float v = t / 4.f;
        d320[i] = v / (1.f + expf(-v));
    }
}
// One warp per OUTPUT COLUMN of mixed[]: the four streams are summed in a
// register instead of through atomicAdd. k_hc_up_fp8 runs one warp per output
// ELEMENT and does atomicAdd(&mixed[i % NEMBD], ...) - 10240 atomics into 2560
// slots, 4-way contention on every address, ~1M atomics per token over the 97
// calls. Access pattern is preserved: the 8 warps of a block still read 8
// consecutive 320-byte rows (2560 contiguous bytes) per stream, so this trades
// nothing away for the atomics.
//
// The inject rows are kept in the SAME launch - splitting them into their own
// kernel was measured and was slower (docs section 8) - but spread across four
// BLOCKS rather than four warps of one block, so the 10240-wide tail lands on
// four SMs instead of one.
__global__ void k_hc_up_fp8_col(const float *normed, const uint8_t *w8, const float *d320,
                                float *mixed, const __nv_bfloat16 *w_inject, float *inj,
                                int write_inject) {
    const uint8_t *sc = qf_fp8_scales(w8, HCC * NEMBD, HCL);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = blockIdx.x * (blockDim.x >> 5) + warp;

    if (col < NEMBD) {
        float total = 0.f;
        #pragma unroll
        for (int c = 0; c < HCC; c++) {
            const int i = c * NEMBD + col;
            const uint8_t *row = w8 + (size_t)i * HCL;
            const uint8_t *srow = sc + (size_t)i * (HCL >> 6);
            float acc = 0.f;
            for (int l = lane; l < HCL; l += 32)
                acc = fmaf(qf_e4m3_dec(row[l]) * qf_ue4m3_dec2(srow[l >> 6]), d320[l], acc);
            #pragma unroll
            for (int off = 16; off; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) total += sigmoidf_(acc) * normed[i] / (float)HCC;
        }
        if (lane == 0) mixed[col] = total;
    }

    if (write_inject && blockIdx.x < HCC && warp == 0) {
        const int r = blockIdx.x;
        const __nv_bfloat16 *irow = w_inject + (size_t)r * (HCC * NEMBD);
        float a2 = 0.f;
        for (int j = lane; j < HCC * NEMBD; j += 32)
            a2 = fmaf(__bfloat162float(irow[j]), normed[j], a2);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a2 += __shfl_down_sync(0xffffffffu, a2, off);
        if (lane == 0) inj[r] = 2.f * sigmoidf_(a2 / (float)HCC);
    }
}

static int g_hc_col = -1;   // QF_HC_COL=0 restores the old atomic kernel


__global__ void k_hc_up_fp8(const float *normed, const uint8_t *w8, const float *d320, float *mixed,
                            const __nv_bfloat16 *w_inject, float *inj, int write_inject) {
    const uint8_t *sc = qf_fp8_scales(w8, HCC * NEMBD, HCL);
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (i >= HCC * NEMBD) return;
    const uint8_t *row = w8 + (size_t)i * HCL;
    const uint8_t *srow = sc + (size_t)i * (HCL >> 6);
    float acc = 0.f;
    for (int l = lane; l < HCL; l += 32)
        acc = fmaf(qf_e4m3_dec(row[l]) * qf_ue4m3_dec2(srow[l >> 6]), d320[l], acc);
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float m = sigmoidf_(acc);
        atomicAdd(&mixed[i % NEMBD], m * normed[i] / (float)HCC);
    }
    if (write_inject && i < HCC) {
        float a2 = 0.f;
        const __nv_bfloat16 *irow = w_inject + (size_t)i * (HCC * NEMBD);
        for (int j = lane; j < HCC * NEMBD; j += 32)
            a2 = fmaf(__bfloat162float(irow[j]), normed[j], a2);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            a2 += __shfl_down_sync(0xffffffffu, a2, off);
        if (lane == 0) inj[i] = 2.f * sigmoidf_(a2 / (float)HCC);
    }
}

// ---- batched (T-token) hyper-connection ------------------------------------
//
// The verify pass exists to read each weight once for T candidates, and every
// dense projection in it does - except the hyper-connection, which ran the
// per-token kernels T times. Each layer has two HCs and each HC reads a
// 320x10240 down slab and a 10240x320 up slab: 6.6 MB per HC, 13.1 MB per
// layer, 635 MB per token over 48 layers. That is as large as the entire
// output head, and at T=4 it was being read four times over.
//
// These kernels keep the arithmetic and the memory access pattern of the
// single-token forms exactly - same /4, same silu, same sigmoid, same
// column-major mixing that replaced the atomics - and only hold T accumulators
// per weight element instead of one. T is a template parameter so the
// accumulators stay in registers.
template <int T>
__global__ __launch_bounds__(256)
void k_hc_down_fp8_T(const float *__restrict__ normedT, const uint8_t *__restrict__ w8,
                     float *__restrict__ d320T) {
    const int IN = HCC * NEMBD;
    const uint8_t *sc = qf_fp8_scales(w8, HCL, IN);
    const int i = blockIdx.x;
    if (i >= HCL) return;
    const uint8_t *row = w8 + (size_t)i * IN;
    const uint8_t *srow = sc + (size_t)i * (IN >> 6);
    const int tid = threadIdx.x;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    for (int j = tid; j < IN; j += 256) {
        const float w = qf_e4m3_dec(row[j]) * qf_ue4m3_dec2(srow[j >> 6]);
        #pragma unroll
        for (int t = 0; t < T; t++)
            acc[t] = fmaf(w, normedT[(size_t)t * IN + j], acc[t]);
    }
    __shared__ float red[32];
    const int lane = tid & 31, warp = tid >> 5;
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float a = acc[t];
        #pragma unroll
        for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) red[warp] = a;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.f;
            for (int w = 0; w < 8; w++) sum += red[w];
            const float v = sum / 4.f;
            d320T[(size_t)t * HCL + i] = v / (1.f + expf(-v));
        }
        __syncthreads();
    }
}

template <int T>
__global__ void k_hc_up_fp8_col_T(const float *__restrict__ normedT, const uint8_t *__restrict__ w8,
                                  const float *__restrict__ d320T, float *__restrict__ mixedT,
                                  const __nv_bfloat16 *__restrict__ w_inject,
                                  float *__restrict__ injT, int write_inject) {
    const int IN = HCC * NEMBD;
    const uint8_t *sc = qf_fp8_scales(w8, IN, HCL);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = blockIdx.x * (blockDim.x >> 5) + warp;

    if (col < NEMBD) {
        float total[T];
        #pragma unroll
        for (int t = 0; t < T; t++) total[t] = 0.f;
        #pragma unroll
        for (int c = 0; c < HCC; c++) {
            const int i = c * NEMBD + col;
            const uint8_t *row = w8 + (size_t)i * HCL;
            const uint8_t *srow = sc + (size_t)i * (HCL >> 6);
            float acc[T];
            #pragma unroll
            for (int t = 0; t < T; t++) acc[t] = 0.f;
            for (int l = lane; l < HCL; l += 32) {
                const float w = qf_e4m3_dec(row[l]) * qf_ue4m3_dec2(srow[l >> 6]);
                #pragma unroll
                for (int t = 0; t < T; t++)
                    acc[t] = fmaf(w, d320T[(size_t)t * HCL + l], acc[t]);
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                float a = acc[t];
                #pragma unroll
                for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
                if (lane == 0) total[t] += sigmoidf_(a) * normedT[(size_t)t * IN + i] / (float)HCC;
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < T; t++) mixedT[(size_t)t * NEMBD + col] = total[t];
        }
    }

    if (write_inject && blockIdx.x < HCC && warp == 0) {
        const int r = blockIdx.x;
        const __nv_bfloat16 *irow = w_inject + (size_t)r * IN;
        float a2[T];
        #pragma unroll
        for (int t = 0; t < T; t++) a2[t] = 0.f;
        for (int j = lane; j < IN; j += 32) {
            const float w = __bfloat162float(irow[j]);
            #pragma unroll
            for (int t = 0; t < T; t++)
                a2[t] = fmaf(w, normedT[(size_t)t * IN + j], a2[t]);
        }
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float a = a2[t];
            #pragma unroll
            for (int off = 16; off; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) injT[(size_t)t * HCC + r] = 2.f * sigmoidf_(a / (float)HCC);
        }
    }
}

static inline void hc_up_fp8_launch(const float *normed, const uint8_t *w8, const float *d320,
                                    float *mixed, const __nv_bfloat16 *w_inject, float *inj,
                                    int write_inject, cudaStream_t s) {
    if (g_hc_col)
        k_hc_up_fp8_col<<<NEMBD / 8, 256, 0, s>>>(normed, w8, d320, mixed, w_inject, inj, write_inject);
    else
        k_hc_up_fp8<<<(HCC * NEMBD + 7) / 8, 256, 0, s>>>(normed, w8, d320, mixed, w_inject, inj, write_inject);
}
// MEASURED DEAD END (2026-08-28): splitting the hyper-connection inject out of
// k_hc_up/k_hc_up_fp8 into a block-parallel kernel REGRESSED decode from 21.66
// to 20.17 tok/s and was reverted the same cycle.
//
// VERDICT: NOT_ADJUDICATED, not "wrong". One sample, one prompt, no variance
// band, and the mechanism below was never profiled. Recorded so the number is
// not mistaken for a settled result.
//
// Hypothesis (UNCONFIRMED): the four warps computing the 10240-wide inject dot
// may already be overlapped with the other 10236 warps in the same launch, so
// separating them would serialize hidden work and add 96 launches per token.
// No profile of that build was taken. Do not cite this as measured.
//
__global__ void k_zero(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}
__global__ void k_stream_inject(float *R, const float *y, const float *inj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCC * NEMBD) R[i] += inj[i / NEMBD] * y[i % NEMBD];
}
__global__ void k_rope(float *q, float *k, const float *inv_freq, long pos) {
    // grid (32, 24+2): standard rope on first 64 dims (interleaved mrope text = T grid)
    int half = blockIdx.x, r = blockIdx.y;
    float *dst = (r < NHEAD) ? q + (size_t)r * QGATE : k + (size_t)(r - NHEAD) * HDIM;
    float a = dst[half], b = dst[half + 32];
    float f = pos * inv_freq[half];
    dst[half] = a * cosf(f) - b * sinf(f);
    dst[half + 32] = b * cosf(f) + a * sinf(f);
}
// Zero-centered RMSNorm over one 256-wide q/k head. q rows have a 512-value
// stride because each row is [q256 | gate256]; k rows are contiguous.
__global__ void k_qknorm(float *x, const __nv_bfloat16 *w, int stride) {
    int h = blockIdx.x, i = threadIdx.x;
    float *row = x + (size_t)h * stride;
    float v = row[i], sum = v * v;
    for (int off = 16; off; off >>= 1)
        sum += __shfl_down_sync(~0u, sum, off);
    __shared__ float partial[(HDIM + 31) / 32], mean;
    if ((i & 31) == 0) partial[i >> 5] = sum;
    __syncthreads();
    if (i == 0) {
        float total = 0.f;
        for (int j = 0; j < HDIM / 32; j++) total += partial[j];
        mean = total / HDIM;
    }
    __syncthreads();
    row[i] = v * rsqrtf(mean + 1e-6f) * (1.f + __bfloat162float(w[i]));
}
__global__ void k_conv_step(float *raw, float *ring, const __nv_bfloat16 *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DINN) return;
    float cur = raw[i];
    float o = __bfloat162float(w[i * 4 + 0]) * ring[0 * DINN + i] + __bfloat162float(w[i * 4 + 1]) * ring[1 * DINN + i]
            + __bfloat162float(w[i * 4 + 2]) * ring[2 * DINN + i] + __bfloat162float(w[i * 4 + 3]) * cur;
    ring[0 * DINN + i] = ring[1 * DINN + i];
    ring[1 * DINN + i] = ring[2 * DINN + i];
    ring[2 * DINN + i] = cur;
    raw[i] = o / (1.f + expf(-o));
}
__global__ void k_gdn_decode(const float *qkv, const float *a_in, const float *b_in,
                             const __nv_bfloat16 *A_log, const __nv_bfloat16 *dt_bias, float *S, float *out48) {
    static_assert(GDN_KD == GDN_VD, "decode kernel requires equal key and value head dimensions");
    static_assert((GDN_KD & (GDN_KD - 1)) == 0, "GDN head dimension must be a power of two");
    int h = blockIdx.x, kh = h / GDN_KV_RATIO, tid = threadIdx.x;
    float dt = a_in[h] + __bfloat162float(dt_bias[h]);
    float softplus_dt = dt > 20.f ? dt : log1pf(expf(dt));
    float g_log = -__expf(__bfloat162float(A_log[h])) * softplus_dt;
    float decay = __expf(g_log);
    float beta = sigmoidf_(b_in[h]);
    __shared__ float ks[GDN_KD], qs[GDN_KD], qred[GDN_KD], kred[GDN_KD];
    float q = qkv[kh * GDN_KD + tid], k = qkv[(GDN_KH * GDN_KD) + kh * GDN_KD + tid], v = qkv[(2 * GDN_KH * GDN_KD) + h * GDN_VD + tid];
    qred[tid] = q * q;
    kred[tid] = k * k;
    __syncthreads();
    for (int stride = GDN_KD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            qred[tid] += qred[tid + stride];
            kred[tid] += kred[tid + stride];
        }
        __syncthreads();
    }
    q *= rsqrtf(qred[0] + 1e-6f);
    k *= rsqrtf(kred[0] + 1e-6f);
    ks[tid] = k;
    qs[tid] = q * (1.f / sqrtf((float)GDN_KD));
    __syncthreads();
    float *Sh = S + (size_t)h * GDN_KD * GDN_VD;
    float *row = Sh + (size_t)tid * GDN_VD;
    for (int col = 0; col < GDN_VD; col++) row[col] *= decay;
    __syncthreads();
    for (int col = tid; col < GDN_VD; col += GDN_VD) {
        float kv = 0.f;
        for (int r = 0; r < GDN_KD; r++) kv += Sh[r * GDN_VD + col] * ks[r];
        float d = (v - kv) * beta;
        for (int r = 0; r < GDN_KD; r++) Sh[r * GDN_VD + col] += ks[r] * d;
    }
    __syncthreads();
    float o = 0.f;
    for (int r = 0; r < GDN_KD; r++) o += Sh[r * GDN_VD + tid] * qs[r];
    out48[h * GDN_VD + tid] = o;
}

// ---------------------------------------------------------------------------
// M=8 multi-request batched GDN. Canonical workload is M=8 (M=1 banned). One
// block per (head, request-row); each row carries its OWN recurrent state S and
// conv ring, so M independent sequences advance in parallel from one weight
// read of A_log/dt_bias/conv1d. Strides: qkv DINN, a/b DTRANK, out GDN_VDIM,
// S = GDN_VH*GDN_KD*GDN_VD, ring 3*DINN.
__global__ void k_conv_step_M(float *raw, float *ring, const __nv_bfloat16 *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DINN) return;
    int m = blockIdx.y;
    float *rawm = raw + (size_t)m * DINN;
    float *rgm  = ring + (size_t)m * (3 * DINN);
    float cur = rawm[i];
    float o = __bfloat162float(w[i * 4 + 0]) * rgm[0 * DINN + i] + __bfloat162float(w[i * 4 + 1]) * rgm[1 * DINN + i]
            + __bfloat162float(w[i * 4 + 2]) * rgm[2 * DINN + i] + __bfloat162float(w[i * 4 + 3]) * cur;
    rgm[0 * DINN + i] = rgm[1 * DINN + i];
    rgm[1 * DINN + i] = rgm[2 * DINN + i];
    rgm[2 * DINN + i] = cur;
    rawm[i] = o / (1.f + expf(-o));
}
__global__ void k_gdn_decode_M(const float *qkv, const float *a_in, const float *b_in,
                               const __nv_bfloat16 *A_log, const __nv_bfloat16 *dt_bias,
                               float *Sbase, float *out48base) {
    int h = blockIdx.x, m = blockIdx.y, kh = h / GDN_KV_RATIO, tid = threadIdx.x;
    const float *qkvm = qkv + (size_t)m * DINN;
    const float *am   = a_in + (size_t)m * DTRANK;
    const float *bm   = b_in + (size_t)m * DTRANK;
    float *S    = Sbase    + (size_t)m * (GDN_VH * GDN_KD * GDN_VD);
    float *out48 = out48base + (size_t)m * GDN_VDIM;
    float dt = am[h] + __bfloat162float(dt_bias[h]);
    float softplus_dt = dt > 20.f ? dt : log1pf(expf(dt));
    float g_log = -__expf(__bfloat162float(A_log[h])) * softplus_dt;
    float decay = __expf(g_log);
    float beta = sigmoidf_(bm[h]);
    __shared__ float ks[GDN_KD], qs[GDN_KD], qred[GDN_KD], kred[GDN_KD];
    float q = qkvm[kh * GDN_KD + tid], k = qkvm[(GDN_KH * GDN_KD) + kh * GDN_KD + tid], v = qkvm[(2 * GDN_KH * GDN_KD) + h * GDN_VD + tid];
    qred[tid] = q * q;
    kred[tid] = k * k;
    __syncthreads();
    for (int stride = GDN_KD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { qred[tid] += qred[tid + stride]; kred[tid] += kred[tid + stride]; }
        __syncthreads();
    }
    q *= rsqrtf(qred[0] + 1e-6f);
    k *= rsqrtf(kred[0] + 1e-6f);
    ks[tid] = k;
    qs[tid] = q * (1.f / sqrtf((float)GDN_KD));
    __syncthreads();
    float *Sh = S + (size_t)h * GDN_KD * GDN_VD;
    float *row = Sh + (size_t)tid * GDN_VD;
    for (int col = 0; col < GDN_VD; col++) row[col] *= decay;
    __syncthreads();
    for (int col = tid; col < GDN_VD; col += GDN_VD) {
        float kv = 0.f;
        for (int r = 0; r < GDN_KD; r++) kv += Sh[r * GDN_VD + col] * ks[r];
        float d = (v - kv) * beta;
        for (int r = 0; r < GDN_KD; r++) Sh[r * GDN_VD + col] += ks[r] * d;
    }
    __syncthreads();
    float o = 0.f;
    for (int r = 0; r < GDN_KD; r++) o += Sh[r * GDN_VD + tid] * qs[r];
    out48[h * GDN_VD + tid] = o;
}

// M-fold recurrent state: one GDN S matrix and conv ring PER request row, per
// recurrent layer. Allocated resident before the first batched token; a
// completing slot is zeroed and reused. Full-attention layers ((il+1)%4==0)
// have no GDN state.
static float **g_gdnS_M = nullptr, **g_convring_M = nullptr;
static __nv_bfloat16 **g_kc_M = nullptr, **g_vc_M = nullptr;  // M-fold QSA KV caches
static int   *g_selM = nullptr;   // [M*NEXPUSED] routed expert ids per row
static float *g_wtsM = nullptr;   // [M*NEXPUSED] routing weights per row
static float *g_shgM = nullptr, *g_shuM = nullptr, *g_edM = nullptr, *g_ginpM = nullptr; // M-fold shared-expert scratch
// M-fold routed-expert scratch (W1): rows stop sharing ctx.router / route_slot /
// eg_all / eu_all / ed_all, so the routed section is M launches -> 1 launch.
static float *g_routerM = nullptr;                 // [M][NEXP] router logits
static int   *g_rslotM  = nullptr;                 // [M][NEXPUSED] resolved slots
static int *g_permM = NULL;      // [M*NEXPUSED] pair order sorted by expert (k_moe_sort_pairs_M)
static float *g_egM_all = nullptr, *g_euM_all = nullptr;   // [M][NEXPUSED][NFF]
static float *g_edM_all = nullptr;                 // [M][NEXPUSED][NEMBD]
static int     g_reqbatch_M = 0;

extern "C" int qf_reqbatch_state_init(int M) {
    fprintf(stderr, "[state-init] M=%d enter\n", M);
    if (g_reqbatch_M >= M) return 0;
    if (g_reqbatch_M) return -1;                 // one-shot sizing
    g_gdnS_M     = (float **)calloc(NLAYER, sizeof(void *));
    g_convring_M = (float **)calloc(NLAYER, sizeof(void *));
    if (!g_gdnS_M || !g_convring_M) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    for (int il = 0; il < NLAYER; il++) {
        if (((il + 1) % 4) == 0) continue;
        if (cudaMalloc(&g_gdnS_M[il], (size_t)M * sS * sizeof(float)) != cudaSuccess) return -1;
        cudaMemset(g_gdnS_M[il], 0, (size_t)M * sS * sizeof(float));
        if (cudaMalloc(&g_convring_M[il], (size_t)M * 3 * DINN * sizeof(float)) != cudaSuccess) return -1;
        cudaMemset(g_convring_M[il], 0, (size_t)M * 3 * DINN * sizeof(float));
    }
    fprintf(stderr, "[state-init] GDN done\n");
    // M-fold QSA KV caches (full-attention layers only): one cache per request
    // row, resident before the first batched token (AGENTS rule 0).
    g_kc_M = (__nv_bfloat16 **)calloc(NLAYER, sizeof(void *));
    g_vc_M = (__nv_bfloat16 **)calloc(NLAYER, sizeof(void *));
    if (!g_kc_M || !g_vc_M) return -1;
    const size_t kvspan = (size_t)qf_maxpos() * KVDIM;
    for (int il = 0; il < NLAYER; il++) {
        if (((il + 1) % 4) != 0) continue;                 // GDN layer: no KV cache
        if (cudaMalloc(&g_kc_M[il], (size_t)M * kvspan * 2) != cudaSuccess) return -1;
        if (cudaMalloc(&g_vc_M[il], (size_t)M * kvspan * 2) != cudaSuccess) return -1;
    }
    fprintf(stderr, "[state-init] KV done qf_maxpos=%d\n", qf_maxpos());
    if (cudaMalloc(&g_selM,  (size_t)M * NEXPUSED * sizeof(int))   != cudaSuccess) return -1;
    if (cudaMalloc(&g_wtsM,  (size_t)M * NEXPUSED * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_shgM,  (size_t)M * NFF * sizeof(float))      != cudaSuccess) return -1;
    if (cudaMalloc(&g_shuM,  (size_t)M * NFF * sizeof(float))      != cudaSuccess) return -1;
    if (cudaMalloc(&g_edM,   (size_t)M * NEMBD * sizeof(float))    != cudaSuccess) return -1;
        if (cudaMalloc(&g_ginpM, (size_t)M * sizeof(float))            != cudaSuccess) return -1;
    if (cudaMalloc(&g_routerM, (size_t)M * NEXP * sizeof(float))              != cudaSuccess) return -1;
    if (cudaMalloc(&g_rslotM,  (size_t)M * NEXPUSED * sizeof(int))            != cudaSuccess) return -1;
    if (cudaMalloc(&g_permM,   (size_t)M * NEXPUSED * sizeof(int))            != cudaSuccess) return -1;
    if (cudaMalloc(&g_egM_all, (size_t)M * NEXPUSED * NFF * sizeof(float))    != cudaSuccess) return -1;
    if (cudaMalloc(&g_euM_all, (size_t)M * NEXPUSED * NFF * sizeof(float))    != cudaSuccess) return -1;
    if (cudaMalloc(&g_edM_all, (size_t)M * NEXPUSED * NEMBD * sizeof(float))  != cudaSuccess) return -1;
    g_reqbatch_M = M;
    return 0;
}

// Zero one request row's recurrent state across all layers (slot reuse on EOS).
extern "C" int qf_reqbatch_state_reset_row(int m, cudaStream_t s) {
    if (m < 0 || m >= g_reqbatch_M) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    for (int il = 0; il < NLAYER; il++) {
        if (((il + 1) % 4) == 0) continue;
        cudaMemsetAsync(g_gdnS_M[il] + (size_t)m * sS, 0, sS * sizeof(float), s);
        cudaMemsetAsync(g_convring_M[il] + (size_t)m * 3 * DINN, 0, (size_t)3 * DINN * sizeof(float), s);
    }
    return 0;
}
float *qf_reqbatch_gdnS(int il)     { return g_gdnS_M ? g_gdnS_M[il] : nullptr; }
float *qf_reqbatch_convring(int il) { return g_convring_M ? g_convring_M[il] : nullptr; }
__nv_bfloat16 *qf_reqbatch_kc(int il) { return g_kc_M ? g_kc_M[il] : nullptr; }
__nv_bfloat16 *qf_reqbatch_vc(int il) { return g_vc_M ? g_vc_M[il] : nullptr; }

__global__ void k_rmsnorm_gated(__nv_bfloat16 *out, const float *x, const __nv_bfloat16 *w, const float *z, float eps) {
    static_assert((GDN_VD & (GDN_VD - 1)) == 0, "GDN value head dimension must be a power of two");
    int h = blockIdx.x, i = threadIdx.x;
    float v = x[h * GDN_VD + i];
    __shared__ float red[GDN_VD];
    red[i] = v * v;
    __syncthreads();
    for (int stride = GDN_VD / 2; stride > 0; stride >>= 1) {
        if (i < stride) red[i] += red[i + stride];
        __syncthreads();
    }
    // Gate activation is SIGMOID, not silu. The checkpoint's config.json sets
    // "output_gate_type": "sigmoid", and Qwen4ExpTextGatedDeltaNet builds its
    // Qwen4ExpTextRMSNormGated with activation=config.output_gate_type, so the
    // reference computes weight * rmsnorm(v) * sigmoid(g).
    // This kernel previously used g/(1+exp(-g)) = g*sigmoid(g) = silu(g),
    // multiplying every GDN output by an extra factor of g. That is 36 of the
    // 48 production layers.
    float g = z[h * GDN_VD + i];
    out[h * GDN_VD + i] = __float2bfloat16(__bfloat162float(w[i]) * (v * rsqrtf(red[0] / GDN_VD + eps)) * (1.f / (1.f + expf(-g))));
}
// M-row clone of k_rmsnorm_gated: blockIdx.y = row over [M][GDN_VH*GDN_VD].
__global__ void k_rmsnorm_gated_M(__nv_bfloat16 *out, const float *x, const __nv_bfloat16 *w, const float *z, float eps) {
    { const int mrow = blockIdx.y; out += (size_t)mrow * GDN_VH * GDN_VD;
      x += (size_t)mrow * GDN_VH * GDN_VD; z += (size_t)mrow * GDN_VH * GDN_VD; }
    static_assert((GDN_VD & (GDN_VD - 1)) == 0, "GDN value head dimension must be a power of two");
    int h = blockIdx.x, i = threadIdx.x;
    float v = x[h * GDN_VD + i];
    __shared__ float red[GDN_VD];
    red[i] = v * v;
    __syncthreads();
    for (int stride = GDN_VD / 2; stride > 0; stride >>= 1) {
        if (i < stride) red[i] += red[i + stride];
        __syncthreads();
    }
    // Gate activation is SIGMOID, not silu. The checkpoint's config.json sets
    // "output_gate_type": "sigmoid", and Qwen4ExpTextGatedDeltaNet builds its
    // Qwen4ExpTextRMSNormGated with activation=config.output_gate_type, so the
    // reference computes weight * rmsnorm(v) * sigmoid(g).
    // This kernel previously used g/(1+exp(-g)) = g*sigmoid(g) = silu(g),
    // multiplying every GDN output by an extra factor of g. That is 36 of the
    // 48 production layers.
    float g = z[h * GDN_VD + i];
    out[h * GDN_VD + i] = __float2bfloat16(__bfloat162float(w[i]) * (v * rsqrtf(red[0] / GDN_VD + eps)) * (1.f / (1.f + expf(-g))));
}
__global__ void k_scores(const float *q, const __nv_bfloat16 *K, float *scores, long pos, float scale, int kh) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pos) return;
    float acc = 0.f;
    for (int d = 0; d < HDIM; d++)
        acc += q[d] * __bfloat162float(K[(size_t)p * (NKV * HDIM) + kh * HDIM + d]);
    scores[p] = acc * scale;
}
__global__ void k_softmax(float *scores, long pos) {
    __shared__ float mx, sum;
    __shared__ float red[8];
    float local = -1e30f;
    for (int p = threadIdx.x; p < pos; p += blockDim.x) local = fmaxf(local, scores[p]);
    for (int off = 16; off; off >>= 1) local = fmaxf(local, __shfl_down_sync(~0u, local, off));
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m2 = -1e30f;
        int nw = (blockDim.x + 31) / 32;
        for (int w = 0; w < nw; w++) m2 = fmaxf(m2, red[w]);
        mx = m2; sum = 0.f;
    }
    __syncthreads();
    float e = 0.f;
    for (int p = threadIdx.x; p < pos; p += blockDim.x) {
        float ev = expf(scores[p] - mx);
        scores[p] = ev;
        e += ev;
    }
    for (int off = 16; off; off >>= 1) e += __shfl_down_sync(~0u, e, off);
    if (lane == 0) atomicAdd(&sum, e);
    __syncthreads();
    for (int p = threadIdx.x; p < pos; p += blockDim.x) scores[p] /= (sum + 1e-9f);
}
__global__ void k_attn_out(const float *scores, const __nv_bfloat16 *V, float *out, long pos, int kh) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HDIM) return;
    float acc = 0.f;
    for (long p = 0; p < pos; p++)
        acc += scores[p] * __bfloat162float(V[(size_t)p * (NKV * HDIM) + kh * HDIM + d]);
    out[d] = acc;
}
__global__ void k_apply_gate(float *out, const float *gate, int dim) {
    int h = blockIdx.x, i = threadIdx.x;
    out[h * dim + i] *= sigmoidf_(gate[h * 2 * dim + dim + i]);
}
__global__ void k_cache_write(__nv_bfloat16 *cache, const float *kv, long pos) {
    // kv [2,256]: cache[pos, 2, 256]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NKV * HDIM) cache[(size_t)pos * (NKV * HDIM) + i] = __float2bfloat16(kv[i]);
}
__global__ void k_nvfp4_gemv(const uint8_t *__restrict__ W, const uint8_t *__restrict__ scales,
                             const float *__restrict__ x, float *__restrict__ y,
                             int rows, int k, float scale2) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    int blocks = k / 64;
    const uint8_t *wr = W + (size_t)row * (size_t)(k / 2);
    const uint8_t *sr = scales + (size_t)row * (size_t)(k / 16);
    float acc = 0.f;
    for (int b = lane; b < blocks; b += 32) {
        const uint8_t *bd = wr + (size_t)b * 32;
        const uint8_t *bs = sr + (size_t)b * 4;
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            float sc = ue4m3f(bs[s]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t q = bd[s * 8 + j];
                int i0 = b * 64 + s * 16 + j, i1 = i0 + 8;
                acc += e2m1f(q & 15) * sc * x[i0];
                acc += e2m1f(q >> 4) * sc * x[i1];
            }
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    if (lane == 0) y[row] = acc * scale2;
}
__global__ void k_silu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}
__global__ void k_axpy(float *y, const float *x, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += w * x[i];
}
// axpy with the scale read from device memory (router weight stays on GPU)
__global__ void k_axpy_devw(float *y, const float *x, const float *w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (*w) * x[i];
}

// GPU-resident router: full NEXP-way softmax denominator + top-NEXPUSED selection.
// sel[k]  = expert id of k-th largest logit (ties: lowest index, matching the old host scan)
// wts[k]  = exp(logit - max) / denom, denom over ALL NEXP experts (norm_topk_prob = false:
//           selected weights are NOT renormalized to sum to 1)
// One block of 128 threads; no host round trip.
__global__ void k_router_topk(const float *__restrict__ logits,
                              int *__restrict__ sel, float *__restrict__ wts) {
    __shared__ float redv[4];
    __shared__ int   redi[4];
    __shared__ float mx, denom, wsum;
    __shared__ int   sel_sh[NEXPUSED];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x == 0) wsum = 0.f;
    __syncthreads();

    // max over all experts
    float bv = -1e30f; int bi = 0x7fffffff;
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
        float v = logits[e];
        if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
    }
    for (int off = 16; off; off >>= 1) {
        float ov = __shfl_down_sync(~0u, bv, off);
        int   oi = __shfl_down_sync(~0u, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { redv[wid] = bv; redi[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mv = -1e30f; int mi = 0x7fffffff;
        for (int w = 0; w < 4; w++)
            if (redv[w] > mv || (redv[w] == mv && redi[w] < mi)) { mv = redv[w]; mi = redi[w]; }
        mx = mv;
    }
    __syncthreads();

    // full-softmax denominator over ALL NEXP logits
    float acc = 0.f;
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) acc += expf(logits[e] - mx);
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    if (lane == 0) redv[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) denom = redv[0] + redv[1] + redv[2] + redv[3];
    __syncthreads();
    const float inv_denom = 1.f / denom;

    // top-NEXPUSED by repeated block argmax over not-yet-selected experts
    for (int k = 0; k < NEXPUSED; k++) {
        float cv = -1e30f; int ci = 0x7fffffff;
        for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (sel_sh[j] == e) { taken = 1; break; }
            float v = logits[e];
            if (!taken && (v > cv || (v == cv && e < ci))) { cv = v; ci = e; }
        }
        for (int off = 16; off; off >>= 1) {
            float ov = __shfl_down_sync(~0u, cv, off);
            int   oi = __shfl_down_sync(~0u, ci, off);
            if (ov > cv || (ov == cv && oi < ci)) { cv = ov; ci = oi; }
        }
        if (lane == 0) { redv[wid] = cv; redi[wid] = ci; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float mv = -1e30f; int mi = 0x7fffffff;
            for (int w = 0; w < 4; w++)
                if (redv[w] > mv || (redv[w] == mv && redi[w] < mi)) { mv = redv[w]; mi = redi[w]; }
            sel_sh[k] = mi;
            sel[k] = mi;
            wts[k] = expf(mv - mx) * inv_denom;
            wsum += wts[k];
        }
        __syncthreads();
    }

    // norm_topk_prob: renormalize the selected weights so they sum to 1.
    //
    // Qwen4ExpTextTopKRouter does
    //     if self.norm_topk_prob: router_top_value /= router_top_value.sum(-1)
    // and the checkpoint's config.json does NOT set norm_topk_prob, so the
    // Qwen4ExpTextConfig default (True) applies. Without this the routed
    // experts are scaled by the mass of the top-10 slice of a 512-way softmax
    // instead of by 1, attenuating the entire routed contribution relative to
    // the shared expert by a token-dependent factor.
    if (threadIdx.x == 0) {
        float inv = wsum > 0.f ? 1.f / wsum : 0.f;
        for (int k = 0; k < NEXPUSED; k++) wts[k] *= inv;
    }
}
// M-row clone of k_router_topk (identical arithmetic; one block per row).
__global__ void k_router_topk_M(const float *__restrict__ logits,
                              int *__restrict__ sel, float *__restrict__ wts) {
    // M-row form: blockIdx.x = request row over [M][NEXP] logits -> [M][NEXPUSED] sel/wts
    { const int mrow = blockIdx.x; logits += (size_t)mrow * NEXP;
      sel += (size_t)mrow * NEXPUSED; wts += (size_t)mrow * NEXPUSED; }
    __shared__ float redv[4];
    __shared__ int   redi[4];
    __shared__ float mx, denom, wsum;
    __shared__ int   sel_sh[NEXPUSED];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x == 0) wsum = 0.f;
    __syncthreads();

    // max over all experts
    float bv = -1e30f; int bi = 0x7fffffff;
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
        float v = logits[e];
        if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
    }
    for (int off = 16; off; off >>= 1) {
        float ov = __shfl_down_sync(~0u, bv, off);
        int   oi = __shfl_down_sync(~0u, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { redv[wid] = bv; redi[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mv = -1e30f; int mi = 0x7fffffff;
        for (int w = 0; w < 4; w++)
            if (redv[w] > mv || (redv[w] == mv && redi[w] < mi)) { mv = redv[w]; mi = redi[w]; }
        mx = mv;
    }
    __syncthreads();

    // full-softmax denominator over ALL NEXP logits
    float acc = 0.f;
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) acc += expf(logits[e] - mx);
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    if (lane == 0) redv[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) denom = redv[0] + redv[1] + redv[2] + redv[3];
    __syncthreads();
    const float inv_denom = 1.f / denom;

    // top-NEXPUSED by repeated block argmax over not-yet-selected experts
    for (int k = 0; k < NEXPUSED; k++) {
        float cv = -1e30f; int ci = 0x7fffffff;
        for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
            int taken = 0;
            for (int j = 0; j < k; j++) if (sel_sh[j] == e) { taken = 1; break; }
            float v = logits[e];
            if (!taken && (v > cv || (v == cv && e < ci))) { cv = v; ci = e; }
        }
        for (int off = 16; off; off >>= 1) {
            float ov = __shfl_down_sync(~0u, cv, off);
            int   oi = __shfl_down_sync(~0u, ci, off);
            if (ov > cv || (ov == cv && oi < ci)) { cv = ov; ci = oi; }
        }
        if (lane == 0) { redv[wid] = cv; redi[wid] = ci; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float mv = -1e30f; int mi = 0x7fffffff;
            for (int w = 0; w < 4; w++)
                if (redv[w] > mv || (redv[w] == mv && redi[w] < mi)) { mv = redv[w]; mi = redi[w]; }
            sel_sh[k] = mi;
            sel[k] = mi;
            wts[k] = expf(mv - mx) * inv_denom;
            wsum += wts[k];
        }
        __syncthreads();
    }

    // norm_topk_prob: renormalize the selected weights so they sum to 1.
    //
    // Qwen4ExpTextTopKRouter does
    //     if self.norm_topk_prob: router_top_value /= router_top_value.sum(-1)
    // and the checkpoint's config.json does NOT set norm_topk_prob, so the
    // Qwen4ExpTextConfig default (True) applies. Without this the routed
    // experts are scaled by the mass of the top-10 slice of a 512-way softmax
    // instead of by 1, attenuating the entire routed contribution relative to
    // the shared expert by a token-dependent factor.
    if (threadIdx.x == 0) {
        float inv = wsum > 0.f ? 1.f / wsum : 0.f;
        for (int k = 0; k < NEXPUSED; k++) wts[k] *= inv;
    }
}

// bf16 expert GEMV with device-resident expert id: row base = W + sel[k] * stride.
// Body identical to k_gemv_bf16 after the device-side expert lookup.
__global__ void k_gemv_bf16_exp(const __nv_bfloat16 *__restrict__ W, size_t stride,
                                const int *__restrict__ sel, int k,
                                const float *__restrict__ x, float *__restrict__ y, int out, int in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out) return;
    W += (size_t)sel[k] * stride;
    float acc = 0.f;
    for (int i = threadIdx.x; i < in; i += blockDim.x)
        acc += __bfloat162float(W[(size_t)row * in + i]) * x[i];
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    __shared__ float ws[4][4];   // [warp][row-in-block]
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid][threadIdx.y] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < 4; w++) t += ws[w][threadIdx.y];
        y[row] = t;
    }
}
static void gemv_bf16_exp(const void *W, size_t stride, const int *sel, int k,
                          const float *x, float *y, int out, int in, cudaStream_t s) {
    dim3 blk(128, 4);
    k_gemv_bf16_exp<<<(out + 3) / 4, blk, 0, s>>>((const __nv_bfloat16 *)W, stride, sel, k, x, y, out, in);
}

// ---------------- wave2: device-resident expert-cache dispatch ----------------

// Resolve a cache slot for every selected expert, entirely on device. On a hit
// the slot's LRU age is touched and the slot is handed to the expert GEMVs via
// route_slots. On a miss the coldest slot is picked as victim, the device maps
// are updated, the slot's doorbell is dropped to 0 (upload in flight), and the
// (expert, slot) pair is appended to the layer's pinned mailbox. The mailbox
// seq stamp is written last, after a system fence, so the host polls it with
// plain volatile reads -- no D2H copy, no router logits, no selected ids.
// One block of 128 threads; NEXPUSED and nslots are tiny.
__global__ void k_route_dispatch(const int *__restrict__ sel,
                                 int *__restrict__ slot_map,
                                 int *__restrict__ expert_in_slot,
                                 uint64_t *__restrict__ age,
                                 uint64_t *__restrict__ clock,
                                 int *__restrict__ route_slots,
                                 uint32_t *__restrict__ ready,
                                 volatile QfRouteMailbox *mb,
                                 const QfDecodeParams *__restrict__ params, int nslots) {
    __shared__ int miss_n_sh, cur_e, cur_slot;
    __shared__ uint64_t red_v[4];
    __shared__ int red_i[4];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x == 0) miss_n_sh = 0;
    __syncthreads();
    for (int k = 0; k < NEXPUSED; k++) {
        if (threadIdx.x == 0) {
            cur_e = sel[k];
            if (cur_e < 0 || cur_e >= NEXP) cur_e = 0;
            cur_slot = slot_map[cur_e];
        }
        __syncthreads();
        if (cur_slot >= 0) {
            // resident hit: touch the LRU clock, hand the slot to the GEMVs
            if (threadIdx.x == 0) {
                age[cur_slot] = ++(*clock);
                route_slots[k] = cur_slot;
            }
        } else {
            // miss: SERIAL coldest-slot victim search by thread 0 (nslots is
            // tiny, ~192; the shuffle-reduce form deadlocked on sm_121a/CUDA 13
            // -- its first GPU run was this LRU path). Hit slots were just
            // bumped to the latest clock, so min-age never evicts this round's
            // selection.
            if (threadIdx.x == 0) {
                uint64_t bv = ~uint64_t(0); int bi = 0;
                for (int i = 0; i < nslots; i++) {
                    uint64_t a2 = age[i];
                    if (a2 < bv || (a2 == bv && i < bi)) { bv = a2; bi = i; }
                }
                int ev = expert_in_slot[bi];
                if (ev >= 0) slot_map[ev] = -1;      // evict victim expert
                slot_map[cur_e] = bi;
                expert_in_slot[bi] = cur_e;
                age[bi] = ++(*clock);
                ready[bi] = 0;                        // doorbell down until host DMA lands
                route_slots[k] = bi;
                int n2 = miss_n_sh;
                mb->miss_expert[n2] = cur_e;
                mb->miss_slot[n2] = bi;
                miss_n_sh = n2 + 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mb->miss_n = miss_n_sh;
        __threadfence_system();
        mb->seq = params->seq;
    }
}

// ---------------- SM121 NVFP4 fast dequant (hand-ported kernel bodies) ------

// Branch-free, LUT-free dequant; bit-identical to e2m1f()/ue4m3f() above but
// without the dynamically indexed local-memory LUT and without ldexpf.
__device__ __forceinline__ float e2m1_bits(uint32_t nib) {
    // E2M1 nibble: s ee m -> 0,.5,1,1.5,2,3,4,6 with sign.
    uint32_t e = (nib >> 1) & 3u, m = nib & 1u;
    uint32_t mag = e ? (((e + 126u) << 23) | (m << 22))   // (1 + m/2) * 2^(e-1)
                     : (m ? 0x3F000000u : 0u);            // subnormal: m * 0.5
    return __uint_as_float(mag | ((nib & 8u) << 28));
}
__device__ __forceinline__ float ue4m3_bits(uint32_t b) {
    // e = b>>3, m = b&7: (8+m)*2^(e-10); e==0: m*2^-9; 0x7F -> INF.
    uint32_t e = b >> 3, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                : (float)m * 0x1p-9f;
    return (b == 0x7Fu) ? __int_as_float(0x7F800000) : v;
}
__device__ __forceinline__ float warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(~0u, v, off);
    return v;
}

// Per-lane partial dot product of one NVFP4 row against x staged in shared
// memory. Lane l processes 16-byte words u = l, l+32, ... so a warp issues
// fully coalesced 512-byte load transactions (vs. 1-byte strided loads).
// One uint4 = 16 packed bytes = 32 values = exactly two 16-value sub-blocks:
// word u covers block b = u>>1, sub-blocks s = 2*(u&1) and 2*(u&1)+1; their
// two scale bytes sit at scales[row*(K/16) + 2u .. 2u+1] (one uint16 load).
template<int K>
__device__ __forceinline__ float nvfp4_lane_dot(const uint8_t *__restrict__ W,
                                                const uint8_t *__restrict__ scales,
                                                const float *__restrict__ xs,
                                                int row, int lane) {
    static_assert(K % 64 == 0, "NVFP4 block layout requires K % 64 == 0");
    const int U4 = K / 32;   // 16-byte words per row: 80 (K=2560) / 20 (K=640)
    const uint4 *wr4 = (const uint4 *)(W + (size_t)row * (K / 2));
    const uint16_t *sr2 = (const uint16_t *)(scales + (size_t)row * (K / 16));
    float acc = 0.f;
    #pragma unroll 4
    for (int u = lane; u < U4; u += 32) {
        uint4 d = __ldg(wr4 + u);
        uint32_t sp = (uint32_t)__ldg(sr2 + u);   // lo byte = sub s0, hi byte = sub s1
        float sc[2] = { ue4m3_bits(sp & 0xFFu), ue4m3_bits(sp >> 8) };
        int base = (u >> 1) * 64 + (u & 1) * 32;  // first x index of this word
        const float4 *xv = (const float4 *)(xs + base);
        uint32_t wd[2][2] = { { d.x, d.y }, { d.z, d.w } };
        #pragma unroll
        // Nibble order: byte i holds value 2i in the LOW nibble and value 2i+1
        // in the HIGH nibble (adjacent pairs).
        //
        // This previously paired the low nibble of byte j with value j and the
        // high nibble with value j+8 - an interleaved layout the checkpoint
        // does not use. That is a permutation WITHIN each 16-value block, so it
        // leaves the output's magnitude distribution intact while destroying
        // its direction: the canary measured routed-expert cosine -0.03 against
        // the reference at a magnitude ratio of 1.01. A magnitude-only check
        // cannot see this, which is why it survived.
        //
        // Ground truth: dequantize layers.0.mlp.experts.0.gate_proj from the
        // NVFP4 checkpoint and compare against the same weights in the BF16
        // original (models/Qwen3.8-Flash-Next):
        //     adjacent pairs  {v[2i] lo, v[2i+1] hi} -> corr 0.9955, maxerr 0.0066
        //     interleaved     {v[j] lo,  v[j+8] hi}  -> corr 0.1420, maxerr 0.0809
        for (int h = 0; h < 2; h++) {             // two 16-value sub-blocks per word
            float4 va = xv[h * 4 + 0], vb = xv[h * 4 + 1];  // x: base+0..7
            float4 vc = xv[h * 4 + 2], vd = xv[h * 4 + 3];  // x: base+8..15
            float xa[8] = { va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w };
            float xb[8] = { vc.x, vc.y, vc.z, vc.w, vd.x, vd.y, vd.z, vd.w };
            float sub = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t q0 = (wd[h][0] >> (8 * j)) & 0xFFu;   // bytes 0..3 -> values 0..7
                uint32_t q1 = (wd[h][1] >> (8 * j)) & 0xFFu;   // bytes 4..7 -> values 8..15
                sub = fmaf(e2m1_bits(q0 & 15u), xa[2 * j],     sub);
                sub = fmaf(e2m1_bits(q0 >> 4),  xa[2 * j + 1], sub);
                sub = fmaf(e2m1_bits(q1 & 15u), xb[2 * j],     sub);
                sub = fmaf(e2m1_bits(q1 >> 4),  xb[2 * j + 1], sub);
            }
            acc = fmaf(sc[h], sub, acc);          // block scale once per 16 values
        }
    }
    return acc;
}

// NVFP4 expert GEMV with fully device-resident dispatch: expert id, cache slot
// and per-expert weight_scale_2 are all read on device, so the host launch loop
// needs no routing data. If the slot's doorbell is down (cold miss), thread 0
// waits for the host DMA to ring it; on the hit path the flag is already
// nonzero and there is no wait. The spin is bounded: on timeout the kernel
// latches route_err and zeroes its output instead of hanging the decode stream.
// SM121 body: x staged once per block in shared memory (coalesced float4
// loads), then the warp-wide nvfp4_lane_dot above. One warp per output row.
template<int K>
__global__ __launch_bounds__(256)
void k_nvfp4_gemv_slot(const uint8_t *__restrict__ W, size_t w_stride,
                       const uint8_t *__restrict__ scales, size_t s_stride,
                       const int *__restrict__ sel, const int *__restrict__ slots, int k,
                       const float *__restrict__ s2tab,
                       const uint32_t *__restrict__ ready,
                       int *__restrict__ route_err,
                       const float *__restrict__ x, float *__restrict__ y,
                       int rows) {
    __shared__ float xs[K];
    const float4 *x4 = (const float4 *)x;
    #pragma unroll 4
    for (int i = threadIdx.x; i < K / 4; i += 256)
        ((float4 *)xs)[i] = __ldg(x4 + i);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (threadIdx.x == 0) {
        const volatile uint32_t *flag = (const volatile uint32_t *)(ready + slots[k]);
        unsigned long long spins = 0;
        while (*flag == 0u) {
            if (++spins > (1ull << 26)) { *route_err = 1; break; }  // deadlock guard
            __nanosleep(256);
        }
        __threadfence();   // acquire: doorbell observed -> expert data visible
    }
    __syncthreads();
    if (*route_err) { if (lane == 0 && row < rows) y[row] = 0.f; return; }
    if (row >= rows) return;
    float scale2 = s2tab[sel[k]];
    int slot = slots[k];
    float acc = warp_sum32(nvfp4_lane_dot<K>(W + (size_t)slot * w_stride,
                                             scales + (size_t)slot * s_stride,
                                             xs, row, lane));
    if (lane == 0) y[row] = acc * scale2;
}

// Host launch wrapper: the decode shapes are exactly in=2560 (gate/up, 640
// rows) and in=640 (down, 2560 rows); both run one warp per row, block 256.
// Grouped NVFP4 expert GEMV: one launch covers ALL NEXPUSED selected experts.
//
// Identical arithmetic to k_nvfp4_gemv_slot - same nvfp4_lane_dot, same scales,
// same weight_scale_2 fold. The ONLY change is launch geometry: the selected
// expert comes from blockIdx.y instead of a kernel argument, so a 640-row
// projection goes from 80 blocks to 800 and actually fills the GPU.
//
// Measured 2026-08-28 (nsys, real checkpoint): the per-expert form spent
// 11.57 us moving 0.92 MB = 79 GB/s inside a single kernel, i.e. 30% of GB10's
// ~273 GB/s, and accounted for 16.40 ms of a 48.4 ms token. 80 blocks cannot
// fill this GPU; the stall is occupancy, not launch overhead.
//
// y_all is [NEXPUSED][rows]; the per-expert results are folded once at the end
// by k_moe_accum, which also applies the routing weights.
template<int K>
__global__ __launch_bounds__(256)
void k_nvfp4_gemv_grouped(const uint8_t *__restrict__ W, size_t w_stride,
                          const uint8_t *__restrict__ scales, size_t s_stride,
                          const int *__restrict__ sel, const int *__restrict__ slots,
                          const float *__restrict__ s2tab,
                          const uint32_t *__restrict__ ready,
                          int *__restrict__ route_err,
                          const float *__restrict__ x, float *__restrict__ y_all,
                          int rows, int x_per_expert) {
    const int kk = blockIdx.y;
    __shared__ float xs[K];
    // gate/up share one activation across all experts; the down projection
    // consumes each expert's OWN hidden vector, laid out [NEXPUSED][K].
    const float4 *x4 = (const float4 *)(x_per_expert ? x + (size_t)kk * K : x);
    #pragma unroll 4
    for (int i = threadIdx.x; i < K / 4; i += 256)
        ((float4 *)xs)[i] = __ldg(x4 + i);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (threadIdx.x == 0) {
        const volatile uint32_t *flag = (const volatile uint32_t *)(ready + slots[kk]);
        unsigned long long spins = 0;
        while (*flag == 0u) {
            if (++spins > (1ull << 26)) { *route_err = 1; break; }
            __nanosleep(256);
        }
        __threadfence();
    }
    __syncthreads();
    float *y = y_all + (size_t)kk * rows;
    if (*route_err) { if (lane == 0 && row < rows) y[row] = 0.f; return; }
    if (row >= rows) return;
    float scale2 = s2tab[sel[kk]];
    int slot = slots[kk];
    float acc = warp_sum32(nvfp4_lane_dot<K>(W + (size_t)slot * w_stride,
                                             scales + (size_t)slot * s_stride,
                                             xs, row, lane));
    if (lane == 0) y[row] = acc * scale2;
}
// Pair order for the M-row grouped GEMVs: the n = M*NEXPUSED (row, k) pairs
// sorted by expert id, ties by pair index (stable, deterministic). The grouped
// kernel reads an expert's weights once per PAIR; at M=16 roughly half of the
// 160 pairs repeat an expert another pair already read (measured 45% overlap
// at M=8). Blocks are dispatched in (x, y, z) order, so sorted pairs put the
// repeats within ~1 MB of traffic of the first read: L2 hits instead of DRAM.
// n <= 160 < blockDim, one block, O(n^2) ranking = trivial.
__global__ void k_moe_sort_pairs_M(const int *__restrict__ sel, int n, int *__restrict__ perm) {
    const int p = threadIdx.x;
    if (p >= n) return;
    const int e = sel[p];
    int rank = 0;
    for (int q = 0; q < n; q++) { const int eq = sel[q]; rank += (eq < e) || (eq == e && q < p); }
    perm[rank] = p;
}
// M-row clone of k_nvfp4_gemv_grouped: one launch covers every (row, expert)
// pair of the batch instead of one launch per row (6 launches/layer, not 48).
template<int K>
__global__ __launch_bounds__(256)
void k_nvfp4_gemv_grouped_M(const uint8_t *__restrict__ W, size_t w_stride,
                          const uint8_t *__restrict__ scales, size_t s_stride,
                          const int *__restrict__ sel, const int *__restrict__ slots,
                          const float *__restrict__ s2tab,
                          const uint32_t *__restrict__ ready,
                          int *__restrict__ route_err,
                          const float *__restrict__ x, float *__restrict__ y_all,
                          int rows, int x_per_expert, const int *__restrict__ perm) {
    // M-row form: blockIdx.z = request row. sel/slots are [M][NEXPUSED]; x is
    // [M][K] (gate/up) or [M][NEXPUSED][K] (down); y_all is [M][NEXPUSED][rows].
    // perm (optional): (blockIdx.z, blockIdx.y) index a pair list sorted by
    // expert (k_moe_sort_pairs_M) instead of (row, k) directly, so the blocks
    // of pairs sharing an expert are dispatched back-to-back and the repeat
    // weight reads hit L2. Each pair's arithmetic and output slot are unchanged.
    int mrow = blockIdx.z, kk = blockIdx.y;
    if (perm) { const int pp = perm[blockIdx.z * NEXPUSED + blockIdx.y]; mrow = pp / NEXPUSED; kk = pp - mrow * NEXPUSED; }
    { sel += (size_t)mrow * NEXPUSED; slots += (size_t)mrow * NEXPUSED;
      x += x_per_expert ? (size_t)mrow * NEXPUSED * K : (size_t)mrow * K;
      y_all += (size_t)mrow * NEXPUSED * rows; }
    __shared__ float xs[K];
    // gate/up share one activation across all experts; the down projection
    // consumes each expert's OWN hidden vector, laid out [NEXPUSED][K].
    const float4 *x4 = (const float4 *)(x_per_expert ? x + (size_t)kk * K : x);
    #pragma unroll 4
    for (int i = threadIdx.x; i < K / 4; i += 256)
        ((float4 *)xs)[i] = __ldg(x4 + i);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (threadIdx.x == 0) {
        const volatile uint32_t *flag = (const volatile uint32_t *)(ready + slots[kk]);
        unsigned long long spins = 0;
        while (*flag == 0u) {
            if (++spins > (1ull << 26)) { *route_err = 1; break; }
            __nanosleep(256);
        }
        __threadfence();
    }
    __syncthreads();
    float *y = y_all + (size_t)kk * rows;
    if (*route_err) { if (lane == 0 && row < rows) y[row] = 0.f; return; }
    if (row >= rows) return;
    float scale2 = s2tab[sel[kk]];
    int slot = slots[kk];
    float acc = warp_sum32(nvfp4_lane_dot<K>(W + (size_t)slot * w_stride,
                                             scales + (size_t)slot * s_stride,
                                             xs, row, lane));
    if (lane == 0) y[row] = acc * scale2;
}

// Fold the NEXPUSED per-expert outputs with their routing weights.
__global__ void k_moe_accum(float *__restrict__ y, const float *__restrict__ ed_all,
                            const float *__restrict__ wts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < NEXPUSED; k++)
        acc = fmaf(wts[k], ed_all[(size_t)k * n + i], acc);
    y[i] = acc;
}

static void nvfp4_gemv_grouped_launch(const void *W, size_t w_stride,
                                      const void *scales, size_t s_stride,
                                      const int *sel, const int *slots,
                                      const float *s2tab, const uint32_t *ready,
                                      int *route_err, const float *x, float *y_all,
                                      int rows, int in, int x_per_expert,
                                      cudaStream_t st) {
    dim3 g((rows + 7) / 8, NEXPUSED);
    if (in == 2560)
        k_nvfp4_gemv_grouped<2560><<<g, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, s2tab, ready, route_err, x, y_all, rows, x_per_expert);
    else if (in == 640)
        k_nvfp4_gemv_grouped<640><<<g, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, s2tab, ready, route_err, x, y_all, rows, x_per_expert);
}

static void nvfp4_gemv_slot_launch(const void *W, size_t w_stride,
                                   const void *scales, size_t s_stride,
                                   const int *sel, const int *slots, int k,
                                   const float *s2tab, const uint32_t *ready,
                                   int *route_err, const float *x, float *y,
                                   int rows, int in, cudaStream_t st) {
    if (in == 2560)
        k_nvfp4_gemv_slot<2560><<<(rows + 7) / 8, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, k, s2tab, ready, route_err, x, y, rows);
    else if (in == 640)
        k_nvfp4_gemv_slot<640><<<(rows + 7) / 8, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, k, s2tab, ready, route_err, x, y, rows);
}

__global__ void k_shexp_add(float *y, const float *sh, const float *g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NEMBD) y[i] += sigmoidf_(g[0]) * sh[i];
}

// ===========================================================================
// M-row launch geometry for the batched decode (W1 launch elimination).
// Each kernel is the per-row kernel above with the request row taken from the
// grid instead of from a host `for r` loop: identical arithmetic per (row,
// element), one launch instead of M. Operands are the [M][N]-contiguous batch
// buffers (bt.*, g_*M).
// ===========================================================================
__global__ void k_embed_dev_M(float *R, const __nv_bfloat16 *embd,
                              const QfDecodeParams *__restrict__ params) {
    const int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NEMBD) return;
    float value = __bfloat162float(embd[(size_t)params[r].token * NEMBD + i]);
    float *Rr = R + (size_t)r * HCC * NEMBD;
    #pragma unroll
    for (int c = 0; c < HCC; c++) Rr[c * NEMBD + i] = value;
}
__global__ void k_stream_inject_M(float *R, const float *y, const float *inj) {
    const int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCC * NEMBD)
        R[(size_t)r * HCC * NEMBD + i] += inj[(size_t)r * HCC + i / NEMBD] * y[(size_t)r * NEMBD + i % NEMBD];
}
__global__ void k_bf16_to_f32_M(float *out, const __nv_bfloat16 *in, int n) {
    const int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[(size_t)r * n + i] = __bfloat162float(in[(size_t)r * n + i]);
}
__global__ void k_shexp_add_M(float *y, const float *sh, const float *g) {
    const int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NEMBD) y[(size_t)r * NEMBD + i] += sigmoidf_(g[r]) * sh[(size_t)r * NEMBD + i];
}
// Fold the NEXPUSED per-expert outputs of every row: ed_all is [M][NEXPUSED][n].
__global__ void k_moe_accum_M(float *__restrict__ y, const float *__restrict__ ed_all,
                              const float *__restrict__ wts, int n) {
    const int r = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *ed = ed_all + (size_t)r * NEXPUSED * n;
    const float *w = wts + (size_t)r * NEXPUSED;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < NEXPUSED; k++)
        acc = fmaf(w[k], ed[(size_t)k * n + i], acc);
    y[(size_t)r * n + i] = acc;
}
// M-row slot resolution for the FULLY RESIDENT expert map (the only map the
// M=8 path runs: nothing services a miss mailbox there, so a miss would spin
// forever in the grouped GEMV). No LRU touch, no mailbox: slot = map[expert],
// and a missing expert latches route_err instead of hanging.
__global__ void k_route_dispatch_M(const int *__restrict__ sel, const int *__restrict__ slot_map,
                                   int *__restrict__ route_slots, int *__restrict__ route_err) {
    const int r = blockIdx.x, k = threadIdx.x;
    if (k >= NEXPUSED) return;
    int slot = slot_map[sel[r * NEXPUSED + k]];
    if (slot < 0) { *route_err = 1; slot = 0; }
    route_slots[r * NEXPUSED + k] = slot;
}
static void nvfp4_gemv_grouped_launch_M(const void *W, size_t w_stride,
                                        const void *scales, size_t s_stride,
                                        const int *sel, const int *slots,
                                        const float *s2tab, const uint32_t *ready,
                                        int *route_err, const float *x, float *y_all,
                                        int rows, int in, int x_per_expert, int M,
                                        const int *perm, cudaStream_t st) {
    dim3 g((rows + 7) / 8, NEXPUSED, M);
    if (in == 2560)
        k_nvfp4_gemv_grouped_M<2560><<<g, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, s2tab, ready, route_err, x, y_all, rows, x_per_expert, perm);
    else if (in == 640)
        k_nvfp4_gemv_grouped_M<640><<<g, 256, 0, st>>>(
            (const uint8_t *)W, w_stride, (const uint8_t *)scales, s_stride,
            sel, slots, s2tab, ready, route_err, x, y_all, rows, x_per_expert, perm);
}
// QF_MOE_SORT=0 restores the unsorted (row, k) block order for A/B.
static int moe_sort_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("QF_MOE_SORT");
        on = (e && !atoi(e)) ? 0 : 1;
        fprintf(stderr, "moe M: grouped expert pairs %s\n", on ? "SORTED by expert (repeat weight reads hit L2)" : "unsorted (row, k) order");
    }
    return on;
}


// ---------------- context ----------------
#ifndef PLE_HEADS
#define PLE_HEADS 16
#endif
void qf_ple_apply(QfModel *m, const int64_t *ids, float *R_dev, cudaStream_t s);  // ple.cu
int  qf_ple_stage_slot(QfModel *m, const int64_t *ids, int slot);
void qf_ple_apply_staged_slot(QfModel *m, float *R_dev, int slot, cudaStream_t s);
void qf_ple_stage_current_slot(QfModel *m, int slot);
int qf_ple_stage(QfModel *m, const int64_t *ids);
void qf_ple_apply_staged(QfModel *m, float *R_dev, cudaStream_t s);
void qf_ple_reset(cudaStream_t s);

struct QfCtx {
    float *x, *R, *normed, *hc_d, *mixed, *inj, *y2560, *qkv_raw, *z6144;
    float *q6144;      // [24,512]: q[256]+gate[256] per head
    float *k512, *v512, *a48, *b48;
    float *attn_out;   // [24,256]
    float *scores;     // [MAXPOS]
    float *router;     // [512]
    int   *sel_dev;    // [NEXPUSED] selected expert ids (device-resident)
    float *wts_dev;    // [NEXPUSED] full-softmax selected weights (device-resident)
    uint32_t route_seq;   // wave2: per-token mailbox stamp
    int   *route_err_dev; // wave2: latched device dispatch error flag
    float *eg, *eu, *ed, *sh;
    float *eg_all, *eu_all, *ed_all;   // [NEXPUSED][NFF/NFF/NEMBD], grouped MoE
    uint8_t *mma_xq, *mma_xs;          // activation in MMA B-operand form (8 cols)
    uint8_t *mma_dxq, *mma_dxs;        // per-expert hidden, same form
    float *sh_g, *sh_u;
    float *logits;     // [248320]
    float *out48;      // [48,128]
    __nv_bfloat16 *gdn_out_bf;
    float **gdnS, **convring;
    __nv_bfloat16 **kc, **vc;
    float *inv_freq;
    // QSA indexer state (per attention layer): pooled keys of 4-token blocks
    float *idx_pool_sum[NLAYER], *idx_pool_key[NLAYER]; int *idx_pool_cnt[NLAYER];
    float *idx640, *inv_freq_idx, *idx_blk_score; uint32_t *idx_mask; int idx_nb_max, idx_mw;
    int *idx_list, *idx_nlist;                    // decode: compact selected-block list [513], count [1]
    cudaStream_t s;
    QfDecodeParams *params_dev;
    QfDecodeParams *params_host;
    QfDecodeParams *paramsT_dev;      // [QF_SPEC_MAXT], one slot per position
    QfDecodeParams *paramsT_host;
};

static QfCtx ctx;
static float *g_logits = NULL;
// PLE n-gram token history. Qwen4ExpTextNGramEmbedding seeds its
// previous_context with context_len (= ngram_size - 1 = 2) EOS ids and then
// slides, so this window is ALWAYS full: three ids, EOS-padded at the start.
//
// It previously started with g_hist_n = 0 and filled g_hist[0..2] without
// shifting, while ple_hash reads g_hist[g_hist_n-1 .. g_hist_n-3]. For the
// first token that indexes g_hist[-1] and g_hist[-2] and for the second
// g_hist[-1] - out-of-bounds reads of whatever statics precede the array,
// where the reference uses EOS. Those two tokens feed the PLE conv ring, so
// the error propagated into every later token of the request.
#ifndef QF_EOS_ID
#define QF_EOS_ID 248044
#endif
static int g_hist[3] = {QF_EOS_ID, QF_EOS_ID, QF_EOS_ID};
static int g_hist_n = 3;
static uint64_t g_ple_mult[3];
static int64_t g_ple_vsizes[16], g_ple_offsets[16];
static int g_ple_ready = 0;
static float *g_ple_emb_dev = NULL;   // the PLE table on device if small (synth); real build uses host mmap

// Opt-in debug sync probes (QF_DEBUG_SYNC=1). Off by default; never active in
// production runs. cudaDeviceSynchronize is called only when the env var is
// set, so the decode hot path is untouched in normal operation.
// QF_ROUTE_DUMP=1: dump per-position expert selections (see the call site).
static int qf_route_dump(void) {
    static int en = -1;
    if (en < 0) en = getenv("QF_ROUTE_DUMP") ? atoi(getenv("QF_ROUTE_DUMP")) : 0;
    return en;
}

static void qf_dbg_probe(const char *tag, int il) {
    static int en = -1;
    if (en < 0) en = getenv("QF_DEBUG_SYNC") ? 1 : 0;
    if (!en) return;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
        fprintf(stderr, "DBGSYNC FAULT at %s il=%d: %s\n", tag, il, cudaGetErrorString(e));
    else if (il < 2)
        fprintf(stderr, "DBGSYNC ok %s il=%d\n", tag, il);
}

// Correctness-canary taps. Compiled in ONLY for the canary target
// (-DQF_CANARY_TAPS, see src/build_canary.sh); the production build does not
// contain the call sites at all, so no probe, sync, or readback can reach the
// serving path. Implemented in main_canary.cu.
#ifdef QF_CANARY_TAPS
extern int qf_canary_record;
int qf_canary_layer = -1;     // current decode layer, for taps in other TUs
void qf_canary_tap(const char *tag, int il, const float *dev, int n);
#define QF_TAP(tag, il, dev, n) do { \
    if (qf_canary_record) qf_canary_tap((tag), (il), (dev), (n)); \
} while (0)
#else
#define QF_TAP(tag, il, dev, n) ((void)0)
#endif

extern int qfd_fp8_is_on(void);
static int g_hc_fp8 = 0;   // QF_FP8_MODE gate: hc weights in E4M3

int qf_forward_init(QfModel *m) {
    // Refuse a context the fused attention cannot hold rather than write past
    // its shared score buffer. NEVER_AGAIN rule 1: no silent fallback path.
    // With the QSA indexer on (default) neither decode nor prefill runs the dense fused
    // kernel: decode walks qf_attn_qsa_layer_list and the chunk pass walks
    // k_attn_partial_list (dense rows carry a full list of at most 513 blocks), so the
    // compile-time score buffer is never indexed by the runtime token count. The KV and
    // indexer state are sized by qf_maxpos() at runtime. The limit stays for QF_QSA_INDEX=0.
    if (qf_maxpos() > (long)QF_ATTN_MAXPOS && !qsa_index_on()) {
        fprintf(stderr,
            "qf: QF_MAX_CONTEXT=%ld exceeds the fused attention limit of %d.\n"
            "    qf_attn_fused.cu sizes scores[] as QF_ATTN_MAXPOS/QF_ATTN_NSPLIT = %d\n"
            "    floats of shared memory per block, but each block's chunk comes from\n"
            "    the runtime token count, so a longer context overruns it. Raising\n"
            "    QF_ATTN_NSPLIT does not help - k_attn_combine keeps one float per\n"
            "    split in per-thread registers. Lower QF_MAX_CONTEXT, or rebuild with\n"
            "    a larger QF_ATTN_MAXPOS and check the shared-memory budget.\n",
            qf_maxpos(), QF_ATTN_MAXPOS, QF_ATTN_MAXPOS / QF_ATTN_NSPLIT);
        return -1;
    }
    // QSA's indexer selects at most indexer_budget positions (2048 in this
    // checkpoint). This engine never runs index_qk_proj and attends densely
    // instead. At or below the budget the two are the same computation, which
    // is why every check against the reference has agreed; past it they are
    // not, and this is the one place that says so.
    if (qf_maxpos() > 2048)
        fprintf(stderr, "qf: NOTE context %ld exceeds QSA indexer_budget 2048; "
                        "attention is DENSE, not the model's sparse top-2048 selection\n",
                qf_maxpos());

    if (g_hc_col < 0) g_hc_col = getenv("QF_HC_COL") ? atoi(getenv("QF_HC_COL")) : 1;
#ifndef SYNTH
    // Dense weights -> E4M3 with UE4M3 per-64-block scales (one-time, load).
    // Dense is ~80% of per-token bytes; BF16 storage left the GB10 bandwidth
    // on the table. The BF16 slabs are freed after conversion (~4.6 GiB back).
    {
        extern int qfd_quant_fp8(const void *src_bf16, int rows, int in,
                                 cudaStream_t s, void **W8);
        extern void qfd_set_fp8_weights(int on);
        int nconv = 0, nfail = 0;
        const char *first_err = NULL;
        // QF_FP8_MODE: all (default) | none | dense | hc  — A/B gate for the
        // FP8 storage rollout; 'none' keeps every weight BF16.
        int fp8_mode_all = 1, fp8_dense = 1, fp8_hc = 1;
        if (const char *e = getenv("QF_FP8_MODE")) {
            fp8_mode_all = strcmp(e, "none") != 0;
            fp8_dense = fp8_mode_all && strcmp(e, "hc") != 0;
            fp8_hc = fp8_mode_all && strcmp(e, "dense") != 0;
        }
        auto qconv = [&](void **pp, int rows, int in, const char *nm) {
            if (!pp || !*pp) return;
            void *w8 = NULL;
            if (qfd_quant_fp8(*pp, rows, in, 0, &w8) == 0) {
                {   // MXFP8 copy for the cuBLASLt prefill path, from the bf16 while it is still resident
                    extern int qfd_mx_register(const void *bf16, const void *key, int rows, int in);
                    qfd_mx_register(*pp, w8, rows, in);
                }
                cudaError_t qe = cudaDeviceSynchronize();   // init-time only
                if (qe != cudaSuccess && !first_err) {
                    fprintf(stderr, "fp8 quant FAULT at %s (%dx%d): %s\n",
                            nm, rows, in, cudaGetErrorString(qe));
                    first_err = nm;
                }
                cudaFree(*pp);
                *pp = w8;
                nconv++;
            } else nfail++;
        };
        for (int il = qf_layer_begin(); il < NLAYER; il++) {
            if (!fp8_dense) break;
            QfLayer *L = &m->layers[il];
            int rec = ((il + 1) % 4) != 0;
            if (rec) {
                qconv(&L->qkv, DINN, NEMBD, "qkv");
                qconv(&L->zgate, GDN_VDIM, NEMBD, "zgate");
                qconv(&L->beta, DTRANK, NEMBD, "beta");
                qconv(&L->alpha, DTRANK, NEMBD, "alpha");
                qconv(&L->gdn_out, NEMBD, GDN_VDIM, "gdn_out");
            } else {
                qconv(&L->wq, NHEAD * QGATE, NEMBD, "wq");
                qconv(&L->wk, KVDIM, NEMBD, "wk");
                qconv(&L->wv, KVDIM, NEMBD, "wv");
                qconv(&L->wo, NEMBD, NHEAD * HDIM, "wo");
                qconv(&L->idx_qk, 640, NEMBD, "idx_qk");
            }
            qconv(&L->router, NEXP, NEMBD, "router");
            qconv(&L->shexp_gate, NFF, NEMBD, "shexp_gate");
            qconv(&L->shexp_up, NFF, NEMBD, "shexp_up");
            qconv(&L->shexp_gate_inp, 1, NEMBD, "shexp_gate_inp");
            qconv(&L->shexp_down, NEMBD, NFF, "shexp_down");
            if (fp8_hc) {
                qconv(&L->hc_attn_down, HCL, HCC * NEMBD, "hc_attn_down");
                qconv(&L->hc_attn_up, HCC * NEMBD, HCL, "hc_attn_up");
                qconv(&L->hc_ffn_down, HCL, HCC * NEMBD, "hc_ffn_down");
                qconv(&L->hc_ffn_up, HCC * NEMBD, HCL, "hc_ffn_up");
            }
        }
        if (fp8_hc) {
            qconv(&m->output_hc_down, HCL, HCC * NEMBD, "output_hc_down");
            qconv(&m->output_hc_up, HCC * NEMBD, HCL, "output_hc_up");
        }
        g_hc_fp8 = fp8_hc && nfail == 0 && !first_err;
        if (fp8_dense) qconv(&m->lm_head, NVOCAB, NEMBD, "lm_head");
        fprintf(stderr, "fp8 dense: %d tensors quantized (%d failed)%s%s\n",
                nconv, nfail, first_err ? " FIRST FAULT: " : "",
                first_err ? first_err : "");
        qfd_set_fp8_weights(fp8_dense && nconv > 0 && nfail == 0 && !first_err);
        fprintf(stderr, "fp8 mode: dense=%d hc=%d\n", fp8_dense && nconv > 0, g_hc_fp8);
    }
#endif
    CHK(cudaMalloc(&ctx.x, NEMBD * 4));
    CHK(cudaMalloc(&ctx.R, HCC * NEMBD * 4));
    CHK(cudaMalloc(&ctx.normed, HCC * NEMBD * 4));
    CHK(cudaMalloc(&ctx.hc_d, HCL * 4));
    CHK(cudaMalloc(&ctx.mixed, NEMBD * 4));
    CHK(cudaMalloc(&ctx.inj, HCC * 4));
    CHK(cudaMalloc(&ctx.y2560, NEMBD * 4));
    CHK(cudaMalloc(&ctx.qkv_raw, DINN * 4));
    CHK(cudaMalloc(&ctx.z6144, GDN_VDIM * 4));
    CHK(cudaMalloc(&ctx.q6144, QKVDIM * 4));
    CHK(cudaMalloc(&ctx.k512, KVDIM * 4));
    CHK(cudaMalloc(&ctx.v512, KVDIM * 4));
    CHK(cudaMalloc(&ctx.a48, 48 * 4));
    CHK(cudaMalloc(&ctx.b48, 48 * 4));
    CHK(cudaMalloc(&ctx.attn_out, ATTN_OUT_DIM * 4));
    CHK(cudaMalloc(&ctx.scores, qf_maxpos() * 4));
    CHK(cudaMalloc(&ctx.router, NEXP * 4));
    CHK(cudaMalloc(&ctx.sel_dev, NEXPUSED * sizeof(int)));
    CHK(cudaMalloc(&ctx.wts_dev, NEXPUSED * 4));
    CHK(cudaMalloc(&ctx.route_err_dev, sizeof(int)));
    CHK(cudaMemset(ctx.route_err_dev, 0, sizeof(int)));
    ctx.route_seq = 0;
    CHK(cudaMalloc(&ctx.eg, NFF * 4));
    CHK(cudaMalloc(&ctx.eg_all, (size_t)NEXPUSED * NFF * 4));
    CHK(cudaMalloc(&ctx.mma_xq, (size_t)8 * (NEMBD / 2)));
    CHK(cudaMalloc(&ctx.mma_xs, (size_t)8 * (NEMBD / 16)));
    CHK(cudaMalloc(&ctx.mma_dxq, (size_t)NEXPUSED * 8 * (NFF / 2)));
    CHK(cudaMalloc(&ctx.mma_dxs, (size_t)NEXPUSED * 8 * (NFF / 16)));
    CHK(cudaMalloc(&ctx.eu_all, (size_t)NEXPUSED * NFF * 4));
    CHK(cudaMalloc(&ctx.ed_all, (size_t)NEXPUSED * NEMBD * 4));
    CHK(cudaMalloc(&ctx.eu, NFF * 4));
    CHK(cudaMalloc(&ctx.ed, NEMBD * 4));
    CHK(cudaMalloc(&ctx.sh, NFF * 4));
    CHK(cudaMalloc(&ctx.sh_g, NFF * 4));
    CHK(cudaMalloc(&ctx.sh_u, NFF * 4));
    CHK(cudaMalloc(&ctx.logits, NVOCAB * 4));
    CHK(cudaMalloc(&ctx.out48, GDN_VH * GDN_VD * 4));
    CHK(cudaMalloc(&ctx.gdn_out_bf, GDN_VH * GDN_VD * 2));
    {   // indexer tables/scratch for the decode step (T = 1)
        ctx.idx_nb_max = (int)((qf_maxpos() + 3) / 4); ctx.idx_mw = (ctx.idx_nb_max + 31) / 32;
        float invf_idx[64]; for (int i = 0; i < 64; i++) invf_idx[i] = powf(1e7f, -2.f * i / 128.f);
        CHK(cudaMalloc(&ctx.inv_freq_idx, 64 * 4)); CHK(cudaMemcpy(ctx.inv_freq_idx, invf_idx, 64 * 4, cudaMemcpyHostToDevice));
        CHK(cudaMalloc(&ctx.idx640, 640 * 4)); CHK(cudaMalloc(&ctx.idx_blk_score, (size_t)ctx.idx_nb_max * 4));
        CHK(cudaMalloc(&ctx.idx_mask, (size_t)ctx.idx_mw * 4)); CHK(cudaMemset(ctx.idx_mask, 0xFF, (size_t)ctx.idx_mw * 4));
        CHK(cudaMalloc(&ctx.idx_list, 513 * sizeof(int))); CHK(cudaMalloc(&ctx.idx_nlist, sizeof(int))); CHK(cudaMemset(ctx.idx_nlist, 0, sizeof(int))); }
    CHK(cudaMalloc(&ctx.inv_freq, 32 * 4));
    float invf[32];
    for (int i = 0; i < 32; i++) invf[i] = powf(1e7f, -2.f * i / 64.f);
    CHK(cudaMemcpy(ctx.inv_freq, invf, 32 * 4, cudaMemcpyHostToDevice));
    // calloc, not malloc: on a split node the loop below skips layers this
    // node does not own, and their entries must read as NULL, not heap
    // residue. On GB10's unified address space a garbage pointer passes the
    // CUDA API's range check and faults at stream execution — a sticky
    // illegal-memory-access that surfaces at the next synchronous call.
    ctx.gdnS = (float **)calloc(NLAYER, sizeof(void *));
    ctx.convring = (float **)calloc(NLAYER, sizeof(void *));
    ctx.kc = (__nv_bfloat16 **)calloc(NLAYER, sizeof(void *));
    ctx.vc = (__nv_bfloat16 **)calloc(NLAYER, sizeof(void *));
    // Recurrent state and KV are per-layer, so a node that does not own a
    // layer allocates nothing for it. On a 12/36 split this is where the
    // Spark's KV footprint drops from 12 QSA layers to 9.
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if ((il + 1) % 4 != 0) {
            CHK(cudaMalloc(&ctx.gdnS[il], (size_t)GDN_VH * GDN_KD * GDN_VD * 4));
            CHK(cudaMemset(ctx.gdnS[il], 0, (size_t)GDN_VH * GDN_KD * GDN_VD * 4));
            CHK(cudaMalloc(&ctx.convring[il], 3 * DINN * 4));
            CHK(cudaMemset(ctx.convring[il], 0, 3 * DINN * 4));
        } else {
            CHK(cudaMalloc(&ctx.kc[il], (size_t)qf_maxpos() * NKV * HDIM * 2));
            CHK(cudaMalloc(&ctx.vc[il], (size_t)qf_maxpos() * NKV * HDIM * 2));
            {   const size_t nb = (size_t)(qf_maxpos() + 3) / 4;
                CHK(cudaMalloc(&ctx.idx_pool_sum[il], nb * 128 * 4)); CHK(cudaMemset(ctx.idx_pool_sum[il], 0, nb * 128 * 4));
                CHK(cudaMalloc(&ctx.idx_pool_key[il], nb * 128 * 4)); CHK(cudaMemset(ctx.idx_pool_key[il], 0, nb * 128 * 4));
                CHK(cudaMalloc(&ctx.idx_pool_cnt[il], nb * sizeof(int))); CHK(cudaMemset(ctx.idx_pool_cnt[il], 0, nb * sizeof(int))); }
        }
    }
    CHK(cudaStreamCreate(&ctx.s));
    CHK(cudaMalloc(&ctx.params_dev, sizeof(QfDecodeParams)));
    CHK(cudaMemset(ctx.params_dev, 0, sizeof(QfDecodeParams)));
    CHK(cudaHostAlloc((void **)&ctx.params_host, sizeof(QfDecodeParams), cudaHostAllocDefault));
    memset(ctx.params_host, 0, sizeof(QfDecodeParams));
    CHK(cudaMalloc(&ctx.paramsT_dev, QF_SPEC_MAXT * sizeof(QfDecodeParams)));
    CHK(cudaMemset(ctx.paramsT_dev, 0, QF_SPEC_MAXT * sizeof(QfDecodeParams)));
    CHK(cudaHostAlloc((void **)&ctx.paramsT_host, QF_SPEC_MAXT * sizeof(QfDecodeParams), cudaHostAllocDefault));
    memset(ctx.paramsT_host, 0, QF_SPEC_MAXT * sizeof(QfDecodeParams));
    if (qf_attn_init() != 0) return -1;
    // cuBLASLt tensor-core state for the dense BF16 GEMVs. Non-fatal: if the
    // handle or scratch cannot be set up, gemv_bf16 falls back to k_gemv_bf16.
    g_lt.ok = 0;
    if (cublasLtCreate(&g_lt.h) == CUBLAS_STATUS_SUCCESS) {
        if (cudaMalloc(&g_lt.xbf, (size_t)QF_XBF_CAP * sizeof(__nv_bfloat16)) == cudaSuccess) {
            g_lt.xbf_cap = QF_XBF_CAP;
            g_lt.stream = ctx.s;    // plans bind to the decode stream
            g_lt.ok = 1;
            lt_plans_init();        // per-shape failure -> that shape falls back
        } else {
            cublasLtDestroy(g_lt.h);
            g_lt.h = NULL;
        }
    }
    if (!g_lt.ok)
        fprintf(stderr, "qf: cuBLASLt unavailable, dense BF16 GEMVs use fallback kernel\n");
    if (qf_fp4tc_init() != 0) return -1;
    qf_dbg_probe("end_init", -1);
    g_logits = ctx.logits;
    return 0;
}
static void qf_bodyT_graph_destroy(void);   // defined with the batched body below

void qf_forward_shutdown(void) {
    qf_graph_destroy();
    qf_bodyT_graph_destroy();
    qf_attn_shutdown();
    qfd_shutdown();
    qf_fp4tc_shutdown();
    for (int i = 0; i < g_lt.nplans; i++) {
        QfLtPlan *p = &g_lt.plans[i];
        if (p->layY) cublasLtMatrixLayoutDestroy(p->layY);
        if (p->layX) cublasLtMatrixLayoutDestroy(p->layX);
        if (p->layW) cublasLtMatrixLayoutDestroy(p->layW);
        if (p->desc) cublasLtMatmulDescDestroy(p->desc);
        if (p->ws) cudaFree(p->ws);
    }
    if (g_lt.h) cublasLtDestroy(g_lt.h);
    if (g_lt.xbf) cudaFree(g_lt.xbf);
    memset(&g_lt, 0, sizeof(g_lt));
}
float *qf_last_logits(void) { return g_logits; }

int qf_route_error(void) {
#ifdef SYNTH
    return 0;
#else
    int v = 0;
    if (ctx.route_err_dev)
        cudaMemcpy(&v, ctx.route_err_dev, sizeof(int), cudaMemcpyDeviceToHost);
    return v;
#endif
}

#ifndef SYNTH
// Opportunistic early service of recently queued layers whose dispatch kernel
// the GPU has already executed. On the hit path this is pure pinned-memory
// reads -- no CUDA API calls, no waits. Returns -1 if a miss upload failed;
// the caller must abort the token (the expert GEMV will spin out and latch
// route_err rather than hang).
static int route_service_window(QfModel *m, int il) {
    int lo = il - 3; if (lo < 0) lo = 0;
    for (int i = lo; i <= il; i++) {
        QfRouteMailbox *mb = m->layers[i].route_mb_host;
        if (mb) {
            // Wait for this token's dispatch kernel to stamp the mailbox;
            // the host races the GPU and a pre-stamp read silently skips
            // the service, so the expert GEMVs spin on the slot doorbells
            // and the next body sync deadlocks before the end-of-step
            // sweep can run.
            struct timespec tw0, tw1;
            clock_gettime(CLOCK_MONOTONIC, &tw0);
            for (;;) {
                if (mb->seq == ctx.route_seq) break;
                usleep(200);   // backoff: a tight spin on the pinned mailbox livelocks the GPU's own write on GB10 unified memory
                fprintf(stderr, "[winspin] L%d mb->seq=%u want=%u miss_n=%u\n", i, mb->seq, ctx.route_seq, mb->miss_n);
                cudaError_t qe = cudaStreamQuery(ctx.s);
                if (qe != cudaSuccess && qe != cudaErrorNotReady) {
                    fprintf(stderr, "[winspin] STREAM ERROR: %s\n", cudaGetErrorString(qe));
                    return -1;
                }
                clock_gettime(CLOCK_MONOTONIC, &tw1);
                double el = (double)(tw1.tv_sec - tw0.tv_sec) + 1e-9 * (double)(tw1.tv_nsec - tw0.tv_nsec);
                if (el > 2.0) {
                    fprintf(stderr, "route window: mailbox L%d not stamped in %.1fs\n", i, el);
                    return -1;
                }
            }
            if (mb->miss_n) {
                if (qf_route_service(m, i) != 0) {
                    fprintf(stderr, "route service failed at layer %d\n", i);
                    return -1;
                }
            }
        }
    }
    return 0;
}
// End-of-token sweep: every layer's mailbox must be stamped for this token and
// every miss serviced before returning, so the sampler's logits readback always
// observes completed expert work. The expert GEMVs were queued unconditionally
// (they spin on slot doorbells), so the GPU never idles waiting for the host;
// on an all-hit token this only waits for the dispatch kernels the host raced
// past while queueing. Bounded by a wall-clock deadline (QF_ROUTE_TIMEOUT_S,
// default 60 s): on timeout, or on any service/stream failure, returns -1 so
// decode aborts with a clean error instead of spinning forever.
#define QF_ROUTE_TIMEOUT_S 60.0
static int route_service_all(QfModel *m) {
    // FULL-RESIDENT: every routed expert of every layer is on the device before
    // the first generated token (exp_slot_for_expert[e] == e for all e), so
    // k_route_dispatch can never signal a miss and miss_n is always 0.
    //
    // The sweep below is a TIGHT HOST SPIN with no sleep and no yield: it polls
    // 48 pinned mailboxes until every layer's dispatch kernel has stamped the
    // current sequence, i.e. it burns a host core waiting for the GPU graph to
    // finish, once per token. On GB10 the host and GPU share one LPDDR5X pool,
    // so that spin also competes with the decode kernels for the very bandwidth
    // this engine is bound by.
    //
    // NEVER_AGAIN rule 3: "no mailbox polling on the all-hit path ... Host CPU
    // during decode must sit near 0%." In full-resident mode EVERY path is the
    // all-hit path.
    if (m->exp_mode == QF_SPARK_MODE_FULL) return 0;

    struct timespec ts0;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    unsigned long long sweeps = 0;
    for (;;) {
        int pending = 0;
        for (int il = 0; il < NLAYER; il++) {
            QfRouteMailbox *mb = m->layers[il].route_mb_host;
            if (!mb) continue;
            if (mb->seq != ctx.route_seq) { pending = 1; continue; }
            if (mb->miss_n && qf_route_service(m, il) != 0) {
                fprintf(stderr, "route service failed at layer %d\n", il);
                return -1;
            }
        }
        if (!pending) return 0;
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double el = (double)(ts.tv_sec - ts0.tv_sec) + 1e-9 * (double)(ts.tv_nsec - ts0.tv_nsec);
        if (el > QF_ROUTE_TIMEOUT_S) {
            fprintf(stderr, "route sweep: timed out after %.1fs waiting for expert uploads\n", el);
            return -1;
        }
        if (++sweeps % 64 == 0) {
            cudaError_t e = cudaStreamQuery(ctx.s);
            if (e != cudaSuccess && e != cudaErrorNotReady) {
                fprintf(stderr, "route sweep: stream error %s\n", cudaGetErrorString(e));
                return -1;
            }
        }
    }
}
#endif

static uint64_t splitmix64(uint64_t v) {
    v += 0x9E3779B97F4A7C15ULL;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
    return v ^ (v >> 31);
}
static uint64_t *g_mult_dev = NULL;
static int64_t *g_vs_dev = NULL, *g_of_dev = NULL;
// real-build constants (overridden in synth by checkpoint I64s via qf_ple_set_constants)
static void ple_init(void) {
#ifdef SYNTH
    (void)0; // constants come from checkpoint
#else
    uint64_t base = 10007ull * 0;
    for (int i = 0; i < 3; i++) {
        uint64_t v = base + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
        uint64_t half = ((1ULL << 63) - 1) / 248320;
        if (half < 1) half = 1;
        g_ple_mult[i] = 2 * (splitmix64(v) % half) + 1;
    }
    int64_t total = 0;
    for (int h = 0; h < 16; h++) {
        int64_t p = 19999999, cnt = h + 1;
        while (cnt > 0) {
            p++;
            int isp = p % 2 != 0;
            for (int64_t d = 3; d * d <= p; d += 2) if (p % d == 0) { isp = 0; break; }
            if (isp) cnt--;
        }
        g_ple_vsizes[h] = p;
        g_ple_offsets[h] = total;
        total += p;
    }
#endif
    g_ple_ready = 1;
}
void qf_ple_set_constants(const uint64_t *mult, const int64_t *vsizes, const int64_t *offsets, int heads) {
    for (int h = 0; h < heads && h < 16; h++) {
        g_ple_mult[h % 3] = mult[h % 3];
        g_ple_vsizes[h] = vsizes[h];
        g_ple_offsets[h] = offsets[h];
    }
    g_ple_ready = 1;
}
static void ple_hash(int64_t *ids) {
    int64_t t0 = g_hist[g_hist_n - 1], t1 = g_hist[g_hist_n - 2], t2 = g_hist[g_hist_n - 3];
    for (int h = 0; h < 8; h++)
        ids[h] = (int64_t)(((uint64_t)t0 * g_ple_mult[0]) ^ ((uint64_t)t1 * g_ple_mult[1])) % g_ple_vsizes[h] + g_ple_offsets[h];
    for (int h = 8; h < 16; h++)
        ids[h] = (int64_t)(((uint64_t)t0 * g_ple_mult[0]) ^ ((uint64_t)t1 * g_ple_mult[1]) ^ ((uint64_t)t2 * g_ple_mult[2])) % g_ple_vsizes[h] + g_ple_offsets[h];
}


// ---- split-deployment residual transfer ------------------------------------
// A node with qf_layer_begin() > 0 receives the residual produced by the node
// that owns the earlier layers, instead of embedding a token. A node that owns
// only a prefix pulls its residual out and ships it on. The residual is
// HCC*NEMBD floats = 40 KB, which is the entire per-token wire cost of a
// contiguous layer cut.
int qf_push_residual(QfModel *m, const float *hR) {
    (void)m;
    if (!ctx.R || !hR) return -1;
    return cudaMemcpy(ctx.R, hR, (size_t)HCC * NEMBD * 4,
                      cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

int qf_pull_residual(QfModel *m, float *hR) {
    (void)m;
    if (!ctx.R || !hR) return -1;
    if (cudaStreamSynchronize(ctx.s) != cudaSuccess) return -1;
    return cudaMemcpy(hR, ctx.R, (size_t)HCC * NEMBD * 4,
                      cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

extern "C" void qf_hc_set_vec(int on) {
    cudaMemcpyToSymbol(d_hc_vec, &on, sizeof(int));
}

// Alternating map: the AMD box owns the output HC + lm_head + argmax, so the
// Spark tail stops at layer 47 and ships ctx.R. The head here would be a wasted
// lm_head stream (NVOCAB x NEMBD) per token; the M=1 region driver switches it off.
static int g_skip_head = 0;
extern "C" void qf_set_skip_head(int on) { g_skip_head = on; }

int qf_decode_body(QfModel *m, int for_capture) {
    cudaStream_t s = ctx.s;
    const QfDecodeParams *params = ctx.params_dev;
    // Token boundary: clear the latched dispatch error so this token's routing
    // state is clean; it is inspected once at the end of the token.
    cudaMemsetAsync(ctx.route_err_dev, 0, sizeof(int), s);
    // Head node only: layers [0, lb) were computed elsewhere and their result
    // is already sitting in ctx.R, pushed in by qf_push_residual().
    const int lb = qf_layer_begin();
    if (lb == 0) {
        k_embed_dev<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.x, ctx.R,
            (const __nv_bfloat16 *)m->tok_embd, params);
        qf_dbg_probe("embed", -1);
        QF_TAP("embed", -1, ctx.R, HCC * NEMBD);
    }

    int hcd = HCC * NEMBD;
    // QF_RESID_DUMP receipt hook: dump ctx.R at the ENTRY of layer
    // QF_RESID_DUMP_LAYER (default 12) for pos<4 - the single-box authority
    // for the wire's per-position residual (compared by cosine, never
    // magnitude). Eager only: a synchronous D2H inside graph capture is
    // illegal, and the receipt positions are prefill (eager) anyway.
    static const char *resid_dump_dir = getenv("QF_RESID_DUMP");
    static const int resid_dump_layer =
        getenv("QF_RESID_DUMP_LAYER") ? atoi(getenv("QF_RESID_DUMP_LAYER")) : 12;
    for (int il = lb; il < NLAYER; il++) {
        if (resid_dump_dir && !for_capture && il == resid_dump_layer &&
            ctx.params_host->pos < 4) {
            static float hbuf[HCC * NEMBD];
            if (cudaStreamSynchronize(s) == cudaSuccess &&
                cudaMemcpy(hbuf, ctx.R, sizeof hbuf, cudaMemcpyDeviceToHost) == cudaSuccess) {
                char pth[512];
                snprintf(pth, sizeof pth, "%s/ref_resid_pos%d.bin",
                         resid_dump_dir, ctx.params_host->pos);
                FILE *df = fopen(pth, "wb");
                if (df) { fwrite(hbuf, 1, sizeof hbuf, df); fclose(df); }
            } else {
                fprintf(stderr, "resid dump: D2H failed at pos %d layer %d\n",
                        ctx.params_host->pos, il);
            }
        }
        QfLayer *L = &m->layers[il];
        int is_recr = ((il + 1) % 4) != 0;
        if (il == 1) qf_ple_apply_staged(m, ctx.R, s);
        qf_dbg_probe("ple_apply", il);
        if (il == 1) QF_TAP("ple", il, ctx.R, hcd);

        k_rmsnorm<<<HCC, 256, 0, s>>>(ctx.normed, ctx.R, (const __nv_bfloat16 *)L->hc_attn_norm, hcd, 1e-6f, NEMBD);
        if (g_hc_fp8) k_hc_down_fp8<<<HCL, 256, 0, s>>>(ctx.normed, (const uint8_t *)L->hc_attn_down, ctx.hc_d);
        else k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)L->hc_attn_down, ctx.hc_d);
        k_zero<<<10, 256, 0, s>>>(ctx.mixed, NEMBD);
        if (g_hc_fp8) hc_up_fp8_launch(ctx.normed, (const uint8_t *)L->hc_attn_up, ctx.hc_d, ctx.mixed, (const __nv_bfloat16 *)L->hc_attn_inject, ctx.inj, 1, s);
        else k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)L->hc_attn_up, ctx.hc_d, ctx.mixed, (const __nv_bfloat16 *)L->hc_attn_inject, ctx.inj, 1);

        QF_TAP("attn_in", il, ctx.mixed, NEMBD);
#ifdef QF_CANARY_TAPS
        qf_canary_layer = il;
#endif
        if (is_recr) {
            qfd_gdn_inproj(L->qkv, L->zgate, L->beta, L->alpha, ctx.mixed,
                           ctx.qkv_raw, ctx.z6144, ctx.a48, ctx.b48, s);
            k_conv_step<<<(DINN + 1023) / 1024, 1024, 0, s>>>(ctx.qkv_raw, ctx.convring[il], (const __nv_bfloat16 *)L->conv1d);
            k_gdn_decode<<<GDN_VH, GDN_VD, 0, s>>>(ctx.qkv_raw, ctx.a48, ctx.b48,
                                                   (const __nv_bfloat16 *)L->a, (const __nv_bfloat16 *)L->dt_bias, ctx.gdnS[il], ctx.out48);
            k_rmsnorm_gated<<<GDN_VH, GDN_VD, 0, s>>>(ctx.gdn_out_bf, ctx.out48, (const __nv_bfloat16 *)L->gdn_norm, ctx.z6144, 1e-6f);
            k_bf16_to_f32<<<(GDN_VDIM + 1023) / 1024, 1024, 0, s>>>(ctx.q6144, ctx.gdn_out_bf, GDN_VDIM);
            qfd_out_proj(L->gdn_out, ctx.q6144, ctx.y2560, NEMBD, GDN_VDIM, s);
            qf_dbg_probe("rec_out", il);
            QF_TAP("gdn_out", il, ctx.y2560, NEMBD);
        } else {
            qfd_qsa_inproj4(L->wq, L->wk, L->wv, L->idx_qk, ctx.mixed, ctx.q6144, ctx.k512, ctx.v512, ctx.idx640, s);
            // QSA indexer: pooled-key update + top-512 block mask for this position
            // (graph-safe: positions come from the device params; nfull <= 512 -> all-ones mask)
            qf_qsa_index_rows(ctx.idx640, 1, params, L->idx_qnorm, L->idx_knorm, ctx.inv_freq_idx, ctx.idx_pool_sum[il],
                              ctx.idx_pool_key[il], ctx.idx_pool_cnt[il], ctx.idx_blk_score, ctx.idx_nb_max, ctx.idx_mask, ctx.idx_mw,
                              ctx.idx_list, ctx.idx_nlist, (long)ctx.params_host->pos, s);
            if (qsa_index_on())
                qf_attn_qsa_layer_list(ctx.q6144, ctx.k512, ctx.v512, ctx.kc[il], ctx.vc[il],
                                       L->q_norm, L->k_norm, ctx.inv_freq, params, ctx.idx_list, ctx.idx_nlist, ctx.attn_out, s);
            else
            qf_attn_qsa_layer(ctx.q6144, ctx.k512, ctx.v512, ctx.kc[il], ctx.vc[il],
                              L->q_norm, L->k_norm, ctx.inv_freq, params, NULL,
                              ctx.attn_out, s);
            QF_TAP("qsa_core", il, ctx.attn_out, NHEAD * HDIM);
            qfd_out_proj(L->wo, ctx.attn_out, ctx.y2560, NEMBD, NHEAD * HDIM, s);
            qf_dbg_probe("attn_out", il);
            QF_TAP("attn_out", il, ctx.y2560, NEMBD);
        }
        k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(ctx.R, ctx.y2560, ctx.inj);

        k_rmsnorm<<<HCC, 256, 0, s>>>(ctx.normed, ctx.R, (const __nv_bfloat16 *)L->hc_ffn_norm, hcd, 1e-6f, NEMBD);
        if (g_hc_fp8) k_hc_down_fp8<<<HCL, 256, 0, s>>>(ctx.normed, (const uint8_t *)L->hc_ffn_down, ctx.hc_d);
        else k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)L->hc_ffn_down, ctx.hc_d);
        k_zero<<<10, 256, 0, s>>>(ctx.mixed, NEMBD);
        if (g_hc_fp8) hc_up_fp8_launch(ctx.normed, (const uint8_t *)L->hc_ffn_up, ctx.hc_d, ctx.mixed, (const __nv_bfloat16 *)L->hc_ffn_inject, ctx.inj, 1, s);
        else k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)L->hc_ffn_up, ctx.hc_d, ctx.mixed, (const __nv_bfloat16 *)L->hc_ffn_inject, ctx.inj, 1);

        // MoE: routing is fully GPU-resident. k_router_topk computes the full
        // 512-way softmax denominator and the top-NEXPUSED selection on device;
        // no 512-float logits copy, no cudaStreamSynchronize.
        qfd_moe_inproj(L->router, L->shexp_gate, L->shexp_up, L->shexp_gate_inp,
                       ctx.mixed, ctx.router, ctx.sh_g, ctx.sh_u, ctx.qkv_raw, s);
        qf_dbg_probe("moe_inproj", il);
        QF_TAP("moe_in", il, ctx.mixed, NEMBD);
        QF_TAP("router", il, ctx.router, NEXP);
        k_router_topk<<<1, 128, 0, s>>>(ctx.router, ctx.sel_dev, ctx.wts_dev);
        qf_dbg_probe("router", il);
        // QF_ROUTE_DUMP=1: print this position's selected expert ids, one line
        // per (layer, position). The factory's widened multi-row bodies decode
        // an expert's weights ONCE and dot them against M rows, so their whole
        // win is bounded by how many of the M rows route to the SAME expert -
        // and every row runs its own top-k. This dump is what makes that
        // overlap measurable instead of assumed. Sequential path only, which
        // during prefill runs eager (graph capture happens after the prompt),
        // so the copy cannot land inside a captured region.
        if (qf_route_dump()) {
            static int sel_h[NEXPUSED];
            static int seq[NLAYER];              // nth call of this layer == position
            if (cudaMemcpyAsync(sel_h, ctx.sel_dev, NEXPUSED * sizeof(int),
                                cudaMemcpyDeviceToHost, s) == cudaSuccess &&
                cudaStreamSynchronize(s) == cudaSuccess) {
                fprintf(stderr, "route L%d p%d:", il, seq[il]++);
                for (int k = 0; k < NEXPUSED; k++) fprintf(stderr, " %d", sel_h[k]);
                fprintf(stderr, "\n");
            }
        }
#ifdef SYNTH
        // Expert weights are all resident (bf16): dispatch reads sel_dev/wts_dev
        // on device, so the host never sees the routing decision.
        k_zero<<<10, 256, 0, s>>>(ctx.y2560, NEMBD);
        for (int k = 0; k < NEXPUSED; k++) {
            gemv_bf16_exp(L->exp_gate, (size_t)NFF * NEMBD, ctx.sel_dev, k, ctx.mixed, ctx.eg, NFF, NEMBD, s);
            gemv_bf16_exp(L->exp_up,   (size_t)NFF * NEMBD, ctx.sel_dev, k, ctx.mixed, ctx.eu, NFF, NEMBD, s);
            k_silu_mul<<<1, NFF, 0, s>>>(ctx.eg, ctx.eu, NFF);
            gemv_bf16_exp(L->exp_down, (size_t)NEMBD * NFF, ctx.sel_dev, k, ctx.eg, ctx.ed, NEMBD, NFF, s);
            k_axpy_devw<<<13, 256, 0, s>>>(ctx.y2560, ctx.ed, ctx.wts_dev + k, NEMBD);
        }
#else
        // Expert cache is device-resident: k_route_dispatch resolves slots,
        // maintains the LRU, and signals misses through the pinned mailbox
        // (zero-copy; the host never reads sel_dev or router logits). The
        // expert GEMVs below are queued unconditionally and read slot/scale2 on
        // device; on a cold miss they spin on the slot doorbell until the host
        // DMA lands. No D2H readback, no event wait on the hit path.
        fprintf(stderr, "[svc] dispatch L%d\n", il);
        k_route_dispatch<<<1, 128, 0, s>>>(ctx.sel_dev,
            L->exp_slot_dev, L->exp_expert_in_slot_dev, L->exp_slot_age_dev, L->exp_clock_dev,
            L->route_slot_dev, L->slot_ready_dev, L->route_mb_dev, params, L->exp_cache_slots);
        k_zero<<<10, 256, 0, s>>>(ctx.y2560, NEMBD);
        if (qf_fp4tc_enabled()) {
            qf_fp4tc_moe(L, ctx.sel_dev, ctx.wts_dev, ctx.mixed, ctx.y2560,
                         ctx.route_err_dev, s);
        } else {
        if (qf_fp4mma_on()) {
            // Native block-scaled FP4 MMA. Operand layout proven standalone by
            // tools/mma_probe.cu (cosine 1.000000 vs a scalar reference), with
            // adjacent-pair packing and no repack.
            qf_fp4mma_quant_launch(ctx.mixed, NEMBD, ctx.mma_xq, ctx.mma_xs, s);
            qf_fp4mma_gemv_launch(L->exp_gate, 640 * 1280, L->exp_scale, 640 * 160,
                ctx.sel_dev, L->route_slot_dev, L->s2_gate_dev,
                ctx.mma_xq, ctx.mma_xs, ctx.eg_all, NFF, NEMBD, 0, s);
            qf_fp4mma_gemv_launch(L->exp_up, 640 * 1280, L->exp_scale_up, 640 * 160,
                ctx.sel_dev, L->route_slot_dev, L->s2_up_dev,
                ctx.mma_xq, ctx.mma_xs, ctx.eu_all, NFF, NEMBD, 0, s);
            k_silu_mul<<<(NEXPUSED * NFF + 255) / 256, 256, 0, s>>>(ctx.eg_all, ctx.eu_all, NEXPUSED * NFF);
            for (int k = 0; k < NEXPUSED; k++)
                qf_fp4mma_quant_launch(ctx.eg_all + (size_t)k * NFF, NFF,
                    ctx.mma_dxq + (size_t)k * 8 * (NFF / 2),
                    ctx.mma_dxs + (size_t)k * 8 * (NFF / 16), s);
            qf_fp4mma_gemv_launch(L->exp_down, (size_t)NEMBD * NFF / 2,
                L->exp_scale_down, (size_t)2560 * 40,
                ctx.sel_dev, L->route_slot_dev, L->s2_down_dev,
                ctx.mma_dxq, ctx.mma_dxs, ctx.ed_all, NEMBD, NFF, 1, s);
            k_moe_accum<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.y2560, ctx.ed_all, ctx.wts_dev, NEMBD);
        } else {
        // Grouped: 3 launches for all NEXPUSED experts instead of 30, with
        // NEXPUSED x the blocks per launch. Identical arithmetic.
        nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_gate, 640 * 1280,
            (const uint8_t *)L->exp_scale, 640 * 160, ctx.sel_dev, L->route_slot_dev,
            L->s2_gate_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.mixed, ctx.eg_all, 640, 2560, 0, s);
        nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_up, 640 * 1280,
            (const uint8_t *)L->exp_scale_up, 640 * 160, ctx.sel_dev, L->route_slot_dev,
            L->s2_up_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.mixed, ctx.eu_all, 640, 2560, 0, s);
        k_silu_mul<<<(NEXPUSED * NFF + 255) / 256, 256, 0, s>>>(ctx.eg_all, ctx.eu_all, NEXPUSED * NFF);
        nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_down, (size_t)NEMBD * NFF / 2,
            (const uint8_t *)L->exp_scale_down, (size_t)2560 * 40, ctx.sel_dev, L->route_slot_dev,
            L->s2_down_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.eg_all, ctx.ed_all, NEMBD, 640, 1, s);
        k_moe_accum<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.y2560, ctx.ed_all, ctx.wts_dev, NEMBD);
        }
        }
        qf_dbg_probe("experts", il);
        // overlap: service any misses the GPU has already signaled
        fprintf(stderr, "[svc] window enter L%d\n", il);
        if (!for_capture && route_service_window(m, il) != 0) return -1;
        fprintf(stderr, "[svc] window done L%d\n", il);
#endif
        QF_TAP("routed", il, ctx.y2560, NEMBD);
        k_silu_mul<<<3, 256, 0, s>>>(ctx.sh_g, ctx.sh_u, NFF);
        qfd_out_proj(L->shexp_down, ctx.sh_g, ctx.ed, NEMBD, NFF, s);
        QF_TAP("shared", il, ctx.ed, NEMBD);
        k_shexp_add<<<13, 256, 0, s>>>(ctx.y2560, ctx.ed, ctx.qkv_raw);
        QF_TAP("moe_out", il, ctx.y2560, NEMBD);
        k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(ctx.R, ctx.y2560, ctx.inj);
        qf_dbg_probe("layer_end", il);
        QF_TAP("layer", il, ctx.R, hcd);
    }

    if (g_skip_head) return 0;                    // alternating map: AMD runs the head on ctx.R

    k_rmsnorm<<<HCC, 256, 0, s>>>(ctx.normed, ctx.R, (const __nv_bfloat16 *)m->output_hc_norm, hcd, 1e-6f, NEMBD);
    if (g_hc_fp8) k_hc_down_fp8<<<HCL, 256, 0, s>>>(ctx.normed, (const uint8_t *)m->output_hc_down, ctx.hc_d);
    else k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)m->output_hc_down, ctx.hc_d);
    k_zero<<<10, 256, 0, s>>>(ctx.mixed, NEMBD);
    if (g_hc_fp8) hc_up_fp8_launch(ctx.normed, (const uint8_t *)m->output_hc_up, ctx.hc_d, ctx.mixed, NULL, NULL, 0, s);
    else k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)m->output_hc_up, ctx.hc_d, ctx.mixed, NULL, NULL, 0);
    QF_TAP("prehead", -1, ctx.mixed, NEMBD);
    qfd_lm_head(m->lm_head, ctx.mixed, ctx.logits, s);
    qf_dbg_probe("lm_head", -1);

    return 0;
}


// ===========================================================================
// MTP speculative-decode draft head
// ===========================================================================
//
// Semantics read from the authority: sglang's
// python/sglang/srt/models/qwen4_exp_mtp.py inside the local docker image
// lmsysorg/sglang:qwen38flashnext. transformers 5.16 does NOT implement MTP
// (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so it is not a
// reference here.
//
// Per draft step (Qwen4ExpForCausalLMMTP.forward + _fuse_residual_linear_shared):
//   e   = fc_embedding( GemmaRMSNorm_2560( embed_tokens[token] ) )
//   hn  = GemmaRMSNorm_10240( hc_in )      <- ONE group over all 10240,
//                                             not grouped by 2560 like hc_norm
//   s_c = fc_hidden( hn[c] )               for each of the 4 hc streams
//   R_c = e + s_c                          (e broadcast to all streams)
//   R   = one Qwen4Exp FULL-ATTENTION decoder layer over R
//   mixed = hyper_connection_mixer(R)      (use_combine=False -> no inject)
//   logits = lm_head(mixed)                (shared head)
//   hc_out = R                             (pre-mixer streams -> next draft step)
//
// GemmaRMSNorm is zero-centered (gemma_weight = weight + 1), which is exactly
// what k_rmsnorm already applies, so the trunk's norm kernel is reused as-is.
//
// The MTP layer is built by sglang as Qwen4ExpModel(..., is_nextn=True) - the
// SAME decoder-layer code as the trunk. So its MoE renormalizes the top-k
// routing weights exactly like the trunk's does. (work/mtp_recon.md claims
// "no score normalization" citing docs/SEMANTICS.md; that claim is wrong and
// is the defect that made the trunk emit word salad until 2026-08-28.)
//
// MTP routed experts are BF16 and FUSED: gate_up_proj [512, 1280, 2560] with
// gate rows [0,640) and up rows [640,1280), and down_proj [512, 2560, 640].
// They are NOT NVFP4, so the bf16 expert GEMV is used, with the up-projection
// addressed by offsetting the base pointer by 640 rows.
// defined in model.cpp (C++ linkage, same as ple.cu's use of it)
void *qf_load_dev_global(const QfStore *st, const char *name);

// defined below in this file
uint32_t qf_decode_next_seq(void);
void qf_decode_set_params(int token, long pos, uint32_t seq);

struct QfMtpCtx {
    int   loaded;
    int   fp8;                         // dense weights converted to E4M3

    // fusion
    void *pre_norm_emb, *pre_norm_hid, *fc_emb, *fc_hid;
    // decoder layer
    QfLayer L;
    void *exp_gate_up, *exp_down;      // BF16 fused expert slabs
    // head mixer
    void *mix_norm, *mix_down, *mix_up;
    // own KV cache for the single full-attention layer
    __nv_bfloat16 *kc, *vc;
    // workspace
    float *emb, *e, *hn, *R, *normed, *hc_d, *mixed, *inj;
    float *q6144, *k512, *v512, *attn_out, *y2560;
    float *router, *sh_g, *sh_u, *shg1, *eg, *eu, *ed;
    int   *sel; float *wts;
    // Own token/pos params, as a RING. Sharing ctx.params_* with the trunk is a
    // trap: the pinned struct is copied with cudaMemcpyAsync, so a host caller
    // that queues a trunk step and a draft step back to back overwrites it
    // while the first copy is still queued behind a thousand kernels - the same
    // hazard qf_decode_set_params_T exists to avoid. Priming does exactly that
    // (decode_step then mtp_prime, once per prompt token) and the draft loop
    // does it again. A ring means the slot a queued copy reads is not touched
    // again until QF_MTP_PSLOTS later steps, which is far past any launch depth.
    QfDecodeParams *params_dev, *params_host;
    int params_slot;
};
#define QF_MTP_PSLOTS 32
static QfMtpCtx g_mtp;

// e broadcast over the 4 streams + per-stream fc_hidden result
__global__ void k_mtp_fuse(float *R, const float *e, const float *s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCC * NEMBD) R[i] = e[i % NEMBD] + s[i];
}

int qf_mtp_load(QfModel *m) {
    if (g_mtp.loaded) return 0;
    QfMtpCtx *g = &g_mtp;
    QfLayer *L = &g->L;
    const long mp = qf_maxpos();

    #define MTPW(dst, nm) do { \
        if (!(dst = qf_load_dev_global(&m->store, (nm)))) { \
            fprintf(stderr, "mtp: MISSING %s\n", (nm)); return -1; } } while (0)

    MTPW(g->pre_norm_emb, "mtp.pre_fc_norm_embedding.weight");
    MTPW(g->pre_norm_hid, "mtp.pre_fc_norm_hidden.weight");
    MTPW(g->fc_emb,       "mtp.fc_embedding.weight");
    MTPW(g->fc_hid,       "mtp.fc_hidden.weight");

    MTPW(L->hc_attn_norm,   "mtp.layers.0.attn_hyper_connection.hc_norm.weight");
    MTPW(L->hc_attn_down,   "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight");
    MTPW(L->hc_attn_up,     "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight");
    MTPW(L->hc_attn_inject, "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight");
    MTPW(L->hc_ffn_norm,    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight");
    MTPW(L->hc_ffn_down,    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight");
    MTPW(L->hc_ffn_up,      "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight");
    MTPW(L->hc_ffn_inject,  "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight");

    MTPW(L->wq,     "mtp.layers.0.self_attn.q_proj.weight");
    MTPW(L->wk,     "mtp.layers.0.self_attn.k_proj.weight");
    MTPW(L->wv,     "mtp.layers.0.self_attn.v_proj.weight");
    MTPW(L->wo,     "mtp.layers.0.self_attn.o_proj.weight");
    MTPW(L->q_norm, "mtp.layers.0.self_attn.q_norm.weight");
    MTPW(L->k_norm, "mtp.layers.0.self_attn.k_norm.weight");

    MTPW(L->router,         "mtp.layers.0.mlp.gate.weight");
    MTPW(L->shexp_gate,     "mtp.layers.0.mlp.shared_expert.gate_proj.weight");
    MTPW(L->shexp_up,       "mtp.layers.0.mlp.shared_expert.up_proj.weight");
    MTPW(L->shexp_down,     "mtp.layers.0.mlp.shared_expert.down_proj.weight");
    MTPW(L->shexp_gate_inp, "mtp.layers.0.mlp.shared_expert_gate.weight");

    MTPW(g->exp_gate_up, "mtp.layers.0.mlp.experts.gate_up_proj");
    MTPW(g->exp_down,    "mtp.layers.0.mlp.experts.down_proj");

    MTPW(g->mix_norm, "mtp.hyper_connection_mixer.hc_norm.weight");
    MTPW(g->mix_down, "mtp.hyper_connection_mixer.input_mix_weight_down.weight");
    MTPW(g->mix_up,   "mtp.hyper_connection_mixer.input_mix_weight_up.weight");
    #undef MTPW

    CHK(cudaMalloc(&g->kc, (size_t)mp * NKV * HDIM * 2));
    CHK(cudaMalloc(&g->vc, (size_t)mp * NKV * HDIM * 2));
    // The draft head never sees the prompt, so its cache has no entries for
    // positions 0..prompt_len while its attention still sums over them.
    // Uninitialized device memory there is unbounded garbage that never ages
    // out of the window. Zeroing does not make those positions RIGHT - priming
    // the head over the prompt would - but it makes them finite and identical
    // run to run, which is the difference between a bug and a coin flip.
    CHK(cudaMemset(g->kc, 0, (size_t)mp * NKV * HDIM * 2));
    CHK(cudaMemset(g->vc, 0, (size_t)mp * NKV * HDIM * 2));

    #define MTPB(p, n) CHK(cudaMalloc(&g->p, (size_t)(n) * sizeof(float)))
    MTPB(emb, NEMBD);      MTPB(e, NEMBD);        MTPB(hn, HCC * NEMBD);
    MTPB(R, HCC * NEMBD);  MTPB(normed, HCC * NEMBD);
    MTPB(hc_d, HCL);       MTPB(mixed, NEMBD);    MTPB(inj, HCC);
    MTPB(q6144, NHEAD * QGATE); MTPB(k512, NKV * HDIM); MTPB(v512, NKV * HDIM);
    MTPB(attn_out, NHEAD * HDIM); MTPB(y2560, NEMBD);
    MTPB(router, NEXP);    MTPB(sh_g, NFF);       MTPB(sh_u, NFF);
    MTPB(shg1, 1);         MTPB(eg, NFF);         MTPB(eu, NFF);  MTPB(ed, NEMBD);
    MTPB(wts, NEXPUSED);
    #undef MTPB
    CHK(cudaMalloc(&g->sel, NEXPUSED * sizeof(int)));
    CHK(cudaMalloc(&g->params_dev, QF_MTP_PSLOTS * sizeof(QfDecodeParams)));
    CHK(cudaMemset(g->params_dev, 0, QF_MTP_PSLOTS * sizeof(QfDecodeParams)));
    CHK(cudaHostAlloc((void **)&g->params_host, QF_MTP_PSLOTS * sizeof(QfDecodeParams),
                      cudaHostAllocDefault));
    memset(g->params_host, 0, QF_MTP_PSLOTS * sizeof(QfDecodeParams));
    g->params_slot = 0;

    // The draft head loads AFTER qf_forward_init's FP8 pass, so its dense
    // weights are still BF16 while g_qfd_fp8 is already on. qfd_gemv_group
    // keys nothing per tensor - the flag is global - so it would read these
    // BF16 buffers as E4M3 slabs. The slab layout puts the scales BEFORE the
    // returned base pointer, so that read runs off the front of the
    // allocation: an illegal access, not merely wrong numbers. Quantize every
    // MTP weight that reaches a qfd_* entry point, and only those - the hc
    // and expert weights go through kernels that take BF16 directly.
    // The draft head stays BF16 unless QF_MTP_FP8=1 asks otherwise. FP8 pays for
    // itself across the trunk's 48 layers, where decode is bandwidth-bound; the
    // head is ONE layer, so converting it saves almost no bytes per token while
    // spending exactly the precision its job needs - picking the right next
    // token. The checkpoint ships every MTP tensor as BF16.
    g->fp8 = getenv("QF_MTP_FP8") ? atoi(getenv("QF_MTP_FP8")) : 0;
    if (g->fp8 && qfd_fp8_is_on()) {
        extern int qfd_quant_fp8(const void *src_bf16, int rows, int in,
                                 cudaStream_t s, void **W8);
        int nq = 0, nf = 0;
        auto mconv = [&](void **pp, int rows, int in, const char *nm) {
            if (!pp || !*pp) return;
            void *w8 = NULL;
            if (qfd_quant_fp8(*pp, rows, in, 0, &w8) == 0) {
                cudaError_t qe = cudaDeviceSynchronize();     // load-time only
                if (qe != cudaSuccess) {
                    fprintf(stderr, "mtp fp8 FAULT at %s (%dx%d): %s\n",
                            nm, rows, in, cudaGetErrorString(qe));
                    nf++; return;
                }
                cudaFree(*pp); *pp = w8; nq++;
            } else nf++;
        };
        mconv(&g->fc_emb, NEMBD, NEMBD, "fc_embedding");
        mconv(&g->fc_hid, NEMBD, NEMBD, "fc_hidden");
        mconv(&L->wq, NHEAD * QGATE, NEMBD, "q_proj");
        mconv(&L->wk, KVDIM, NEMBD, "k_proj");
        mconv(&L->wv, KVDIM, NEMBD, "v_proj");
        mconv(&L->wo, NEMBD, NHEAD * HDIM, "o_proj");
        mconv(&L->router, NEXP, NEMBD, "router");
        mconv(&L->shexp_gate, NFF, NEMBD, "shexp_gate");
        mconv(&L->shexp_up, NFF, NEMBD, "shexp_up");
        mconv(&L->shexp_gate_inp, 1, NEMBD, "shexp_gate_inp");
        mconv(&L->shexp_down, NEMBD, NFF, "shexp_down");
        if (nf) { fprintf(stderr, "mtp: %d dense tensors failed FP8 conversion\n", nf); return -1; }
        fprintf(stderr, "mtp: %d dense tensors -> FP8\n", nq);
    } else {
        g->fp8 = 0;
        fprintf(stderr, "mtp: dense weights stay BF16\n");
    }

    g->loaded = 1;
    fprintf(stderr, "mtp: draft head loaded (31 tensors, own KV for %ld positions)\n", mp);
    return 0;
}

// Stage-by-stage fault localization for the draft head. Off unless QF_MTP_DEBUG=1
// so the syncs never cost anything in a real run.
static int g_mtp_dbg = -1;
static void mtp_ck(const char *stage) {
    if (g_mtp_dbg < 0) g_mtp_dbg = getenv("QF_MTP_DEBUG") ? atoi(getenv("QF_MTP_DEBUG")) : 0;
    if (!g_mtp_dbg) return;
    cudaError_t e = cudaStreamSynchronize(ctx.s);
    if (e != cudaSuccess) { fprintf(stderr, "mtp FAULT at stage: %s -> %s\n", stage, cudaGetErrorString(e)); exit(3); }
    fprintf(stderr, "mtp ok: %s\n", stage);
}

// One draft step. hc_in/hc_out are device [HCC*NEMBD]; they may alias.
// logits_out is device [NVOCAB]. Returns 0 on success.
// One projection in the draft head. qfd_* assumes an E4M3 slab whenever the
// GLOBAL g_qfd_fp8 flag is set, so a BF16 head has to take the BF16 kernel
// explicitly - the flag is not a per-tensor property.
static void mtp_proj(const void *w, const float *x, float *y, int out, int in, cudaStream_t s) {
    if (g_mtp.fp8) qfd_out_proj(w, x, y, out, in, s);
    else qf_gemv_bf16(w, x, y, out, in, s);
}

int qf_mtp_step(QfModel *m, int token, long pos,
                const float *hc_in, float *hc_out, float *logits_out) {
    if (!g_mtp.loaded) return -1;
    QfMtpCtx *g = &g_mtp;
    QfLayer *L = &g->L;
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD;

    // params for the attention layer: MTP advances its own absolute position
    const int ps = g->params_slot;
    g->params_slot = (ps + 1) % QF_MTP_PSLOTS;
    g->params_host[ps].token = token;
    g->params_host[ps].pos = (int)pos;
    g->params_host[ps].seq = qf_decode_next_seq();
    g->params_host[ps].flags = 0;
    cudaMemcpyAsync(g->params_dev + ps, g->params_host + ps, sizeof(QfDecodeParams),
                    cudaMemcpyHostToDevice, s);
    const QfDecodeParams *params = g->params_dev + ps;

    // 1. shared trunk embedding -> [NEMBD]
    k_embed_1<<<(NEMBD + 255) / 256, 256, 0, s>>>(g->emb,
        (const __nv_bfloat16 *)m->tok_embd, params);
    // 2. e = fc_embedding(norm(emb))
    k_rmsnorm<<<1, 256, 0, s>>>(g->normed, g->emb,
        (const __nv_bfloat16 *)g->pre_norm_emb, NEMBD, 1e-6f, NEMBD);
    mtp_proj(g->fc_emb, g->normed, g->e, NEMBD, NEMBD, s);
    mtp_ck("embed+fc_emb");
    // Control: QF_MTP_ZEROHC=1 feeds the head zeros instead of the trunk's
    // hidden streams. If the drafts do not change, the hidden path contributes
    // nothing and the head is running on the token embedding alone - which is
    // exactly what "generic but plausible" drafts look like.
    // hc_in == NULL means the same thing on purpose: position 0 has no
    // predecessor hidden state, and zeros are what pre_fc_norm_hidden maps to
    // an exactly-zero fc_hidden contribution (R = e broadcast), which is the
    // defined "no history" input rather than an uninitialized one.
    static int zerohc = -1;
    if (zerohc < 0) zerohc = getenv("QF_MTP_ZEROHC") ? 1 : 0;
    static float *zbuf = NULL;
    if (zerohc || !hc_in) {
        if (!zbuf) { cudaMalloc(&zbuf, (size_t)hcd * 4); cudaMemset(zbuf, 0, (size_t)hcd * 4); }
        hc_in = zbuf;
    }

    // 3. hn = norm(hc_in) over ONE group of 10240, then split into HCC streams.
    //    Per the reference (sglang qwen4_exp_mtp._init_pre_fc_norms):
    //        hidden_norm_size = hc_count * hidden_size if hc_count > 1 else hidden_size
    //        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden_norm_size)
    //    so the normalized dim really is 10240 - ONE statistic over all four
    //    streams - and _fuse_residual_linear_shared normalizes FIRST and only
    //    then does .view(..., hc_count, hidden_size) to apply fc_hidden per
    //    stream. This is deliberately NOT the per-stream convention the hc_norm
    //    weights use elsewhere in the model; it was checked and reverted.
    k_rmsnorm<<<1, 256, 0, s>>>(g->hn, hc_in,
        (const __nv_bfloat16 *)g->pre_norm_hid, hcd, 1e-6f, hcd);
    // 4. s_c = fc_hidden(hn_c) per stream, into g->normed reused as scratch
    for (int c = 0; c < HCC; c++)
        mtp_proj(g->fc_hid, g->hn + (size_t)c * NEMBD,
                 g->normed + (size_t)c * NEMBD, NEMBD, NEMBD, s);
    // 5. R = e (broadcast) + s
    k_mtp_fuse<<<(hcd + 255) / 256, 256, 0, s>>>(g->R, g->e, g->normed);
    mtp_ck("fuse");
    QF_TAP("mtp_hcin", 0, hc_in, hcd);
    QF_TAP("mtp_emb", 0, g->emb, NEMBD);
    QF_TAP("mtp_e", 0, g->e, NEMBD);
    QF_TAP("mtp_hn", 0, g->hn, hcd);
    QF_TAP("mtp_fused", 0, g->R, hcd);

    // 6. one full-attention decoder layer, identical structure to the trunk's
    k_rmsnorm<<<HCC, 256, 0, s>>>(g->normed, g->R,
        (const __nv_bfloat16 *)L->hc_attn_norm, hcd, 1e-6f, NEMBD);
    k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)L->hc_attn_down, g->hc_d);
    k_zero<<<10, 256, 0, s>>>(g->mixed, NEMBD);
    k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)L->hc_attn_up, g->hc_d, g->mixed,
        (const __nv_bfloat16 *)L->hc_attn_inject, g->inj, 1);

    mtp_proj(L->wq, g->mixed, g->q6144, NHEAD * QGATE, NEMBD, s);
    mtp_proj(L->wk, g->mixed, g->k512, NKV * HDIM, NEMBD, s);
    mtp_proj(L->wv, g->mixed, g->v512, NKV * HDIM, NEMBD, s);
    mtp_ck("attn_hc+inproj");
    qf_attn_qsa_layer(g->q6144, g->k512, g->v512, g->kc, g->vc,
                      L->q_norm, L->k_norm, ctx.inv_freq, params, NULL,
                      g->attn_out, s);
    mtp_ck("qsa_attn");
    QF_TAP("mtp_attn", 0, g->attn_out, NHEAD * HDIM);
    mtp_proj(L->wo, g->attn_out, g->y2560, NEMBD, NHEAD * HDIM, s);
    k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(g->R, g->y2560, g->inj);

    mtp_ck("attn_inject");
    k_rmsnorm<<<HCC, 256, 0, s>>>(g->normed, g->R,
        (const __nv_bfloat16 *)L->hc_ffn_norm, hcd, 1e-6f, NEMBD);
    k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)L->hc_ffn_down, g->hc_d);
    k_zero<<<10, 256, 0, s>>>(g->mixed, NEMBD);
    k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)L->hc_ffn_up, g->hc_d, g->mixed,
        (const __nv_bfloat16 *)L->hc_ffn_inject, g->inj, 1);

    mtp_proj(L->router, g->mixed, g->router, NEXP, NEMBD, s);
    mtp_proj(L->shexp_gate, g->mixed, g->sh_g, NFF, NEMBD, s);
    mtp_proj(L->shexp_up, g->mixed, g->sh_u, NFF, NEMBD, s);
    mtp_proj(L->shexp_gate_inp, g->mixed, g->shg1, 1, NEMBD, s);
    k_router_topk<<<1, 128, 0, s>>>(g->router, g->sel, g->wts);
    mtp_ck("moe_inproj+router");
    k_zero<<<10, 256, 0, s>>>(g->y2560, NEMBD);
    {
        // BF16 fused experts: gate rows [0,640), up rows [640,1280) of gate_up.
        const size_t gu_stride = (size_t)2 * NFF * NEMBD;
        const __nv_bfloat16 *gu = (const __nv_bfloat16 *)g->exp_gate_up;
        for (int k = 0; k < NEXPUSED; k++) {
            gemv_bf16_exp(gu, gu_stride, g->sel, k, g->mixed, g->eg, NFF, NEMBD, s);
            gemv_bf16_exp(gu + (size_t)NFF * NEMBD, gu_stride, g->sel, k,
                          g->mixed, g->eu, NFF, NEMBD, s);
            k_silu_mul<<<3, 256, 0, s>>>(g->eg, g->eu, NFF);
            gemv_bf16_exp((const __nv_bfloat16 *)g->exp_down, (size_t)NEMBD * NFF,
                          g->sel, k, g->eg, g->ed, NEMBD, NFF, s);
            k_axpy_devw<<<13, 256, 0, s>>>(g->y2560, g->ed, g->wts + k, NEMBD);
        }
    }
    k_silu_mul<<<3, 256, 0, s>>>(g->sh_g, g->sh_u, NFF);
    mtp_ck("experts");
    mtp_proj(L->shexp_down, g->sh_g, g->ed, NEMBD, NFF, s);
    k_shexp_add<<<13, 256, 0, s>>>(g->y2560, g->ed, g->shg1);
    mtp_ck("shexp");
    k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(g->R, g->y2560, g->inj);

    // 7. head mixer: no inject (use_combine=False)
    k_rmsnorm<<<HCC, 256, 0, s>>>(g->normed, g->R,
        (const __nv_bfloat16 *)g->mix_norm, hcd, 1e-6f, NEMBD);
    k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)g->mix_down, g->hc_d);
    k_zero<<<10, 256, 0, s>>>(g->mixed, NEMBD);
    k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(g->normed,
        (const __nv_bfloat16 *)g->mix_up, g->hc_d, g->mixed, NULL, NULL, 0);
    mtp_ck("mixer");

    QF_TAP("mtp_mixed", 0, g->mixed, NEMBD);
    // 8. shared lm_head
    qfd_lm_head(m->lm_head, g->mixed, logits_out, s);
    mtp_ck("lm_head");
    // 9. pre-mixer streams chain into the next draft step
    if (hc_out) cudaMemcpyAsync(hc_out, g->R, (size_t)hcd * sizeof(float),
                                cudaMemcpyDeviceToDevice, s);
    return 0;
}

float *qf_mtp_hc(void) { return g_mtp.loaded ? g_mtp.R : NULL; }

// Zero the draft head's KV cache. Its attention sums over every position up to
// the current one, so entries it never wrote contribute whatever was in device
// memory; a fresh sequence must not inherit the previous one's.
void qf_mtp_reset(void) {
    if (!g_mtp.loaded) return;
    const size_t n = (size_t)qf_maxpos() * NKV * HDIM * 2;
    cudaMemsetAsync(g_mtp.kc, 0, n, ctx.s);
    cudaMemsetAsync(g_mtp.vc, 0, n, ctx.s);
}

// Run the draft head over one prompt position so its KV cache covers the
// prompt. Called after the trunk has consumed the PREVIOUS token, so ctx.R
// holds the hidden streams the head expects; `token` is the id at `pos`.
// Without this the head attends over prompt positions it never wrote - a
// permanent corruption of its context, not one that ages out.
int qf_mtp_prime(QfModel *m, int token, long pos) {
    if (!g_mtp.loaded) return 0;
    static float *scratch = NULL;
    if (!scratch && cudaMalloc(&scratch, (size_t)NVOCAB * 4) != cudaSuccess) return -1;
    return qf_mtp_step(m, token, pos, ctx.R, NULL, scratch);
}

// Position 0 has no predecessor, so qf_mtp_prime cannot cover it - and an
// unwritten slot 0 is not harmless here: it is the attention sink every later
// query sees, and a zeroed K scores 0 against every q, which is a LARGE
// softmax weight next to typical logits, draining mass into a zero V. Prime it
// from the token alone (hc_in = NULL -> zeros), so slot 0 holds the head's own
// "no history" state instead of a hole.
int qf_mtp_prime_bos(QfModel *m, int token) {
    if (!g_mtp.loaded) return 0;
    static float *scratch = NULL;
    if (!scratch && cudaMalloc(&scratch, (size_t)NVOCAB * 4) != cudaSuccess) return -1;
    return qf_mtp_step(m, token, 0, NULL, NULL, scratch);
}

// The trunk's pre-output-mixer hyper-connection streams [HCC*NEMBD]. This is
// exactly sglang's spec_info.hidden_states handoff into the first draft step.
float *qf_decode_hc_streams(void) { return ctx.R; }

// qf_session_reset - clear per-sequence decode state between server requests.
// GDN SSM state and conv rings carry token history and must be zeroed; the
// full-attention KV caches are write-before-read per position and only
// positions [0, pos) are ever attended, so they need no clearing. PLE token
// history resets to the EOS-padded initial state. Weights and residency maps
// are never touched: the model stays loaded for the life of the process.
void qf_session_reset(QfModel *m) {
    (void)m;
    // From qf_layer_begin(), not 0: a split tail owns no recurrent state for
    // the head's layers and allocated none. This loop running from 0 with
    // lb=12 was the layer-run split's pos-0 illegal memory access.
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if ((il + 1) % 4 != 0) {
            cudaMemsetAsync(ctx.gdnS[il], 0, (size_t)GDN_VH * GDN_KD * GDN_VD * 4, ctx.s);
            cudaMemsetAsync(ctx.convring[il], 0, 3 * DINN * 4, ctx.s);
        } else if (ctx.idx_pool_sum[il]) {
            const size_t nb = (size_t)ctx.idx_nb_max;
            cudaMemsetAsync(ctx.idx_pool_sum[il], 0, nb * 128 * 4, ctx.s);
            cudaMemsetAsync(ctx.idx_pool_cnt[il], 0, nb * sizeof(int), ctx.s);
        }
    }
    qf_ple_reset(ctx.s);
    cudaStreamSynchronize(ctx.s);
    g_hist[0] = g_hist[1] = g_hist[2] = QF_EOS_ID;
    g_hist_n = 3;
}


// Device-side greedy argmax over the logits.
//
// Greedy sampling previously copied the whole logit vector to the host
// (NVOCAB * 4 = 993 KB per token at 248320 vocab) and scanned it there. Rule 4
// permits sampling on the host, but the full-vector D2H and the 248320-element
// host scan are avoidable: reduce on device and copy back one int.
//
// Two-stage: k_argmax_part reduces NVOCAB into QF_ARGMAX_BLOCKS partials, then
// k_argmax_fin picks the winner. Ties resolve to the LOWEST index, matching the
// host loop's `>` comparison exactly, so the emitted token stream is unchanged.
#define QF_ARGMAX_BLOCKS 256
__global__ void k_argmax_part(const float *__restrict__ x, int n,
                              float *__restrict__ bv, int *__restrict__ bi) {
    const int tid = threadIdx.x;
    float v = -INFINITY; int idx = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float t = x[i];
        if (t > v) { v = t; idx = i; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = v; si[tid] = idx;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (tid < off) {
            float o = sv[tid + off];
            if (o > sv[tid] || (o == sv[tid] && si[tid + off] < si[tid])) {
                sv[tid] = o; si[tid] = si[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { bv[blockIdx.x] = sv[0]; bi[blockIdx.x] = si[0]; }
}
// M-row clone: blockIdx.y = row; partials are [M][QF_ARGMAX_BLOCKS].
__global__ void k_argmax_part_M(const float *__restrict__ x, int n,
                              float *__restrict__ bv, int *__restrict__ bi) {
    { const int mrow = blockIdx.y; x += (size_t)mrow * n;
      bv += (size_t)mrow * QF_ARGMAX_BLOCKS; bi += (size_t)mrow * QF_ARGMAX_BLOCKS; }
    const int tid = threadIdx.x;
    float v = -INFINITY; int idx = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float t = x[i];
        if (t > v) { v = t; idx = i; }
    }
    __shared__ float sv[256];
    __shared__ int si[256];
    sv[tid] = v; si[tid] = idx;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if (tid < off) {
            float o = sv[tid + off];
            if (o > sv[tid] || (o == sv[tid] && si[tid + off] < si[tid])) {
                sv[tid] = o; si[tid] = si[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { bv[blockIdx.x] = sv[0]; bi[blockIdx.x] = si[0]; }
}
__global__ void k_argmax_fin(const float *__restrict__ bv, const int *__restrict__ bi,
                             int *__restrict__ out) {
    float v = -INFINITY; int idx = 0;
    for (int i = 0; i < QF_ARGMAX_BLOCKS; i++)
        if (bv[i] > v || (bv[i] == v && bi[i] < idx)) { v = bv[i]; idx = bi[i]; }
    *out = idx;
}
// M-row clone: one block per row, out is [M].
__global__ void k_argmax_fin_M(const float *__restrict__ bv, const int *__restrict__ bi,
                             int *__restrict__ out) {
    { const int mrow = blockIdx.x; bv += (size_t)mrow * QF_ARGMAX_BLOCKS;
      bi += (size_t)mrow * QF_ARGMAX_BLOCKS; out += mrow; }
    float v = -INFINITY; int idx = 0;
    for (int i = 0; i < QF_ARGMAX_BLOCKS; i++)
        if (bv[i] > v || (bv[i] == v && bi[i] < idx)) { v = bv[i]; idx = bi[i]; }
    *out = idx;
}
// Returns the greedy token for the last forward. One int crosses the bus.
int qf_argmax_token(void) {
    static float *bv = NULL; static int *bi = NULL, *dout = NULL, *hout = NULL;
    if (!bv) {
        if (cudaMalloc(&bv, QF_ARGMAX_BLOCKS * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMalloc(&bi, QF_ARGMAX_BLOCKS * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc(&dout, sizeof(int)) != cudaSuccess) return -1;
        if (cudaHostAlloc((void **)&hout, sizeof(int), cudaHostAllocDefault) != cudaSuccess) return -1;
    }
    k_argmax_part<<<QF_ARGMAX_BLOCKS, 256, 0, ctx.s>>>(ctx.logits, NVOCAB, bv, bi);
    k_argmax_fin<<<1, 1, 0, ctx.s>>>(bv, bi, dout);
    if (cudaMemcpyAsync(hout, dout, sizeof(int), cudaMemcpyDeviceToHost, ctx.s) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(ctx.s) != cudaSuccess) return -1;
    return *hout;
}

// Greedy argmax of ALL M rows of the batched logits: one launch pair, one
// 4*M-byte D2H, one sync - instead of M separate full-queue drains.
static float *g_amx_bv = NULL; static int *g_amx_bi = NULL, *g_amx_dout = NULL, *g_amx_hout = NULL;
extern "C" int qf_batch_argmax_M(const float *logitsM, int M, int *out) {
    if (M < 1 || M > QF_SPEC_MAXT) return -1;
    if (!g_amx_bv) {
        if (cudaMalloc(&g_amx_bv, (size_t)QF_SPEC_MAXT * QF_ARGMAX_BLOCKS * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMalloc(&g_amx_bi, (size_t)QF_SPEC_MAXT * QF_ARGMAX_BLOCKS * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc(&g_amx_dout, (size_t)QF_SPEC_MAXT * sizeof(int)) != cudaSuccess) return -1;
        if (cudaHostAlloc((void **)&g_amx_hout, (size_t)QF_SPEC_MAXT * sizeof(int), cudaHostAllocDefault) != cudaSuccess) return -1;
    }
    k_argmax_part_M<<<dim3(QF_ARGMAX_BLOCKS, M), 256, 0, ctx.s>>>(logitsM, NVOCAB, g_amx_bv, g_amx_bi);
    k_argmax_fin_M<<<M, 1, 0, ctx.s>>>(g_amx_bv, g_amx_bi, g_amx_dout);
    if (cudaMemcpyAsync(g_amx_hout, g_amx_dout, (size_t)M * sizeof(int), cudaMemcpyDeviceToHost, ctx.s) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(ctx.s) != cudaSuccess) return -1;
    for (int r = 0; r < M; r++) out[r] = g_amx_hout[r];
    return 0;
}

uint32_t qf_decode_next_seq(void) { return ++ctx.route_seq; }

cudaStream_t qf_decode_stream(void) { return ctx.s; }

// Per-position params for a batched forward. qf_decode_set_params writes ONE
// pinned struct and copies it with cudaMemcpyAsync; from pinned memory that
// copy is genuinely asynchronous, so calling it once per position in a loop
// lets the host overwrite the struct while earlier copies are still in flight
// and every position ends up reading the last token/pos. Sequential decode
// never hit this because a whole forward separates the calls.
void qf_decode_set_params_T(const int *tokens, long pos0, int T) {
    for (int t = 0; t < T; t++) {
        ctx.paramsT_host[t].token = tokens[t];
        ctx.paramsT_host[t].pos = (int)(pos0 + t);
        ctx.paramsT_host[t].seq = ++ctx.route_seq;
        ctx.paramsT_host[t].flags = 0;
    }
    cudaMemcpyAsync(ctx.paramsT_dev, ctx.paramsT_host,
                    (size_t)T * sizeof(QfDecodeParams), cudaMemcpyHostToDevice, ctx.s);
}

void qf_decode_set_params(int token, long pos, uint32_t seq) {
    ctx.params_host->token = token;
    ctx.params_host->pos = (int)pos;
    ctx.params_host->seq = seq;
    ctx.params_host->flags = 0;
    cudaMemcpyAsync(ctx.params_dev, ctx.params_host, sizeof(QfDecodeParams),
                    cudaMemcpyHostToDevice, ctx.s);
}

void qf_ple_stage_current(QfModel *m) { qf_ple_stage_current_slot(m, 0); }

void qf_ple_stage_current_slot(QfModel *m, int slot) {
    int64_t ids[PLE_HEADS];
    ple_hash(ids);
    qf_ple_stage_slot(m, ids, slot);
}

void qf_hist_push(int token) {
    g_hist[0] = g_hist[1];
    g_hist[1] = g_hist[2];
    g_hist[2] = token;
}

void qf_hist_reset(void) {
    g_hist[0] = g_hist[1] = g_hist[2] = QF_EOS_ID;
    g_hist_n = 3;
}

void qf_state_reset(QfModel *m) {
    (void)m;
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if ((il + 1) % 4 != 0) {
            cudaMemsetAsync(ctx.gdnS[il], 0, (size_t)GDN_VH * GDN_KD * GDN_VD * 4, ctx.s);
            cudaMemsetAsync(ctx.convring[il], 0, 3 * DINN * 4, ctx.s);
        }
    }
}

int qf_route_service_token(QfModel *m) {
#ifdef SYNTH
    (void)m;
    return 0;
#else
    if (route_service_all(m) != 0) return -1;
    if (qf_route_error()) {
        fprintf(stderr, "decode: device routing deadlock guard tripped\n");
        return -1;
    }
    return 0;
#endif
}

#ifndef SYNTH
// ---- Frankenpool expert-shard RPC entry ------------------------------------
// The AMD head routed the token and owns the hot experts; this node owns the
// rest. Evaluate n routed experts of layer il against activation x (2560 f32)
// and return their weighted sum (2560 f32, k_moe_accum semantics). The router
// is BYPASSED: sel/wts arrive from the wire, and full residency gives slot
// identity (slot e == expert e), so route_slot_dev is written directly and no
// dispatch/LRU/mailbox runs. Requires QF_EXPERT_MODE=full: on any other mode
// a shard answer could be a doorbell-timeout zero, so refuse loudly instead.
// Reuses the decode ctx buffers; callers must not overlap this with decode
// (the wire serve loop is sequential by construction).
// Padding: unused ids repeat ids[0] with weight 0 - fmaf(0,v,acc) contributes
// exactly nothing while keeping every ed_all row initialized.
extern "C" int qf_expert_eval(QfModel *m, int il, int n,
                              const int *ids, const float *w,
                              const float *x2560, float *y2560_out) {
    if (m->exp_mode != QF_SPARK_MODE_FULL) {
        fprintf(stderr, "expert_eval: refused, needs QF_EXPERT_MODE=full "
                        "(slot==expert identity)\n");
        return -1;
    }
    if (il < 0 || il >= NLAYER || n < 1 || n > NEXPUSED) return -1;
    QfLayer *L = &m->layers[il];
    cudaStream_t s = ctx.s;
    int sel_h[NEXPUSED]; float w_h[NEXPUSED];
    for (int k = 0; k < NEXPUSED; k++) {
        sel_h[k] = (k < n) ? ids[k] : ids[0];
        w_h[k]   = (k < n) ? w[k]   : 0.f;
        if (sel_h[k] < 0 || sel_h[k] >= NEXP) return -1;
    }
    if (cudaMemcpyAsync(ctx.mixed, x2560, NEMBD * sizeof(float),
                        cudaMemcpyHostToDevice, s) != cudaSuccess) return -1;
    cudaMemcpyAsync(ctx.sel_dev, sel_h, sizeof sel_h, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(ctx.wts_dev, w_h, sizeof w_h, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(L->route_slot_dev, sel_h, sizeof sel_h, cudaMemcpyHostToDevice, s);
    k_zero<<<10, 256, 0, s>>>(ctx.y2560, NEMBD);
    // Grouped NVFP4 path only (the certified baseline); the fp4tc/fp4mma fast
    // paths are decode-side options and cross-node comparison is by cosine and
    // argmax, never bitwise (M=1 vendor-noise law).
    nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_gate, 640 * 1280,
        (const uint8_t *)L->exp_scale, 640 * 160, ctx.sel_dev, L->route_slot_dev,
        L->s2_gate_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.mixed, ctx.eg_all, 640, 2560, 0, s);
    nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_up, 640 * 1280,
        (const uint8_t *)L->exp_scale_up, 640 * 160, ctx.sel_dev, L->route_slot_dev,
        L->s2_up_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.mixed, ctx.eu_all, 640, 2560, 0, s);
    k_silu_mul<<<(NEXPUSED * NFF + 255) / 256, 256, 0, s>>>(ctx.eg_all, ctx.eu_all, NEXPUSED * NFF);
    nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_down, (size_t)NEMBD * NFF / 2,
        (const uint8_t *)L->exp_scale_down, (size_t)2560 * 40, ctx.sel_dev, L->route_slot_dev,
        L->s2_down_dev, L->slot_ready_dev, ctx.route_err_dev, ctx.eg_all, ctx.ed_all, NEMBD, 640, 1, s);
    k_moe_accum<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.y2560, ctx.ed_all, ctx.wts_dev, NEMBD);
    if (cudaMemcpyAsync(y2560_out, ctx.y2560, NEMBD * sizeof(float),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaError_t ce = cudaGetLastError();
        fprintf(stderr, "expert_eval: L%d failed: %s\n", il, cudaGetErrorString(ce));
        return -1;
    }
    return qf_route_error() ? -1 : 0;
}
#endif

int qf_decode_step(QfModel *m, int token, long pos) {
    if (!g_ple_ready) ple_init();
    qf_dbg_probe("ple_init", -1);
    // The n-gram hash must include THIS token: Qwen4ExpTextNGramEmbedding uses
    // shifted_tokens[0] = the current input id (shift 0), with shifts 1 and 2
    // reaching back into previous_context. Pushing after staging made every
    // PLE gather one token stale, so the injected embedding belonged to the
    // previous position for the whole request.
    qf_hist_push(token);
    qf_ple_stage_current(m);
    qf_dbg_probe("ple_stage", -1);
    qf_decode_set_params(token, pos, qf_decode_next_seq());
    qf_dbg_probe("set_params", -1);
    if (qf_decode_body(m, 0) != 0) return -1;
    qf_dbg_probe("body", -1);
    if (qf_route_service_token(m) != 0) return -1;
    return 0;
}

// The TAIL form of the single-stream step for the alternating map (AMD prefix
// -> Spark layers lb..47 -> AMD head). The token's PLE gather and n-gram
// history belong to the AMD box (PLE sits in layer 1, which it owns), so the
// 16 mmap'd table gathers per token are skipped here; the residual arrives via
// qf_push_residual() and leaves via qf_pull_residual(). Same device body as
// qf_decode_step, minus the work this box does not own.
// QF_TAIL_GRAPH=1 (default): the tail body (layers lb..47, head skipped) is
// captured ONCE into a CUDA graph and replayed per token - the same
// capture/replay the single-Spark decode loop uses (qf_decode_graph.cu), so
// the per-token cost is the params write + one graph launch instead of ~900
// eager kernel launches. QF_TAIL_GRAPH=0 keeps the eager body.
static cudaGraphExec_t g_tail_exec = NULL;
static int g_tail_state = -1;                    // -1 unset, 0 eager, 1 graph
// Graph lifetime rule (refill, 09-06): the captured M=1 tail graph does not survive a chunk ingest of
// the resident sequence (its captured environment - pointer tables, workspaces, kv_len-dependent
// state - is mutated by the wide chunk path; 8K resident + 2K ingest faulted at the first replay while
// the eager tail passed). qf_ingest_region calls this; the next decode step recaptures (~35 ms once).
extern "C" void qf_tail_graph_invalidate(void) {
    if (g_tail_exec) { cudaGraphExecDestroy(g_tail_exec); g_tail_exec = NULL; fprintf(stderr, "tail graph: invalidated after ingest (recapture at the next decode step)\n"); }
    if (g_tail_state > 1) g_tail_state = 1;           // 1 = graph mode enabled, not captured
}
extern "C" int qf_decode_step_tail(QfModel *m, int token, long pos) {
    if (g_tail_state < 0) { const char *e = getenv("QF_TAIL_GRAPH"); g_tail_state = (e && e[0] == '0') ? 0 : 1; }
    qf_decode_set_params(token, pos, qf_decode_next_seq());
    if (g_tail_state == 1) {
        cudaStream_t s = ctx.s;
        if (!g_tail_exec) {
            if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
            cudaGraph_t graph = NULL;
            if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { g_tail_state = 0; }
            else {
                int rc = qf_decode_body(m, 1);
                cudaError_t e = cudaStreamEndCapture(s, &graph);
                if (rc != 0 || e != cudaSuccess || !graph ||
                    cudaGraphInstantiateWithFlags(&g_tail_exec, graph, 0) != cudaSuccess) {
                    fprintf(stderr, "tail graph: capture failed (%s); eager tail stays active\n", cudaGetErrorString(e));
                    if (graph) cudaGraphDestroy(graph);
                    g_tail_exec = NULL; g_tail_state = 0;
                } else {
                    cudaGraphDestroy(graph);
                    fprintf(stderr, "tail graph: captured layers %d..%d (head skipped=%d)\n", qf_layer_begin(), NLAYER - 1, g_skip_head);
                }
            }
        }
        if (g_tail_state == 1) {
            if (cudaGraphLaunch(g_tail_exec, s) != cudaSuccess) return -1;
            if (qf_route_service_token(m) != 0) return -1;
            return 0;
        }
    }
    if (qf_decode_body(m, 0) != 0) return -1;
    if (qf_route_service_token(m) != 0) return -1;
    return 0;
}

// ===========================================================================
// Speculative decode: batched T-token verify, state snapshot/rollback
// ===========================================================================
//
// Speculative decoding only pays if the k drafts are checked in ONE trunk
// forward. Running the trunk k times to check k drafts costs exactly what it
// saves. The whole win is that dense weights - 4.27 GB/token, 76% of traffic -
// are read once for the whole batch (qfd_gemv_group_T). The routed experts do
// NOT amortize (each token picks its own 10 of 512) and stay a per-token loop.
//
// Every batched primitive used here is proven against a reference:
//   qf_gdn_prefill_f32w / qf_attn_prefill / qf_conv_prefill_f32w / qf_hc_prefill_f32w
//     -> cuda/test_prefill.cu, tools/test_hc_T.cu
//   qfd_gemv_group_T -> tools/test_dense_T.cu (bit-identical to T sequential)
extern "C" {
int qf_prefill_init(void);
int qf_attn_prefill(const float *q, int qstride, const void *kc, const void *vc,
                    const int *mask, float *out, int T, cudaStream_t st);
extern "C" int qf_gdn_fla_prefill(const float *qkv, const float *a, const float *b, const float *A_log, const float *dt_bias, float *S, float *out48, int T, int astride, int unpack, cudaStream_t s);
extern "C" const void *qf_gdn_fla_out(void);
extern "C" int qfd_gdn_post_quant(const void *o_bf16, const float *z, const void *w, int T, cudaStream_t s);
extern "C" int qfd_hc_fused_rows(float *R, const void *w_norm, const void *w_down, const void *w_up, const void *w_inject, float *d, float *up, float *normed, float *mixed, float *inj, const float *yin, const float *injin, const void *w_sgi, float *shgi, int T, cudaStream_t s);
extern "C" int qfd_gemm_mx_prequant(const QfDenseProj *projs, int nproj, int in, int T, int pad_rows, cudaStream_t s);
extern "C" int qfd_shexp_act_quant(const float *g, const float *u, int T, cudaStream_t s);
static int gdn_fla_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_PF_GDN_FLA"); v = (e && e[0] == '0') ? 0 : 1; } return v; }
static int pf_hcf(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_PF_HCF"); v = (e && e[0] == '0') ? 0 : 1; } return v; }
int qf_gdn_prefill_f32w(const float *qkv, const float *a, const float *b, const float *A_log, const float *dt_bias,
                        float *S, float *out, int T, cudaStream_t st);
int qf_conv_prefill_f32w(float *raw, float *ring, const float *w, int T, cudaStream_t st);
}
size_t qf_ple_ring_bytes(void);
float *qf_ple_ring_ptr(void);


struct QfBatch {
    int cap;
    float *R, *normed, *mixed, *inj, *y2560, *qkv_raw, *z6144, *a48, *b48;
    float *q6144, *k512, *v512, *attn_out, *out48, *hc_d, *logits;
    float *hcW_normed, *hcW_d320, *hcW_up;      // qf_hc_prefill_f32w workspaces
    __nv_bfloat16 *gdn_out_bf;
    // snapshot of per-sequence recurrent state
    float *gdnS_bak, *ring_bak, *ple_ring_bak;
    int hist_bak[3];
    int have_snapshot;
};
static QfBatch bt;

static int qf_batch_init(int T) {
    if (bt.cap >= T) return 0;
    if (bt.cap) return -1;                    // one-shot sizing
    const int hcd = HCC * NEMBD;
    #define BA(f, n) if (cudaMalloc(&bt.f, (size_t)(n) * sizeof(float)) != cudaSuccess) return -1
    BA(R, T * hcd);        BA(normed, T * hcd);   BA(mixed, T * NEMBD);
    BA(inj, T * HCC);      BA(y2560, T * NEMBD);  BA(qkv_raw, T * DINN);
    BA(z6144, T * GDN_VDIM); BA(a48, T * DTRANK); BA(b48, T * DTRANK);
    BA(q6144, T * NHEAD * QGATE); BA(k512, T * NKV * HDIM); BA(v512, T * NKV * HDIM);
    BA(attn_out, T * NHEAD * HDIM); BA(out48, T * GDN_VH * GDN_VD);
    BA(hc_d, T * HCL);     BA(logits, (size_t)T * NVOCAB);
    BA(hcW_normed, T * hcd); BA(hcW_d320, T * HCL); BA(hcW_up, T * hcd);
    #undef BA
    if (cudaMalloc(&bt.gdn_out_bf, (size_t)T * GDN_VH * GDN_VD * 2) != cudaSuccess) return -1;
    // snapshots: GDN S and conv ring for every recurrent layer, plus the PLE ring
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD * 4, sR = (size_t)3 * DINN * 4;
    if (cudaMalloc(&bt.gdnS_bak, sS * NLAYER) != cudaSuccess) return -1;
    if (cudaMalloc(&bt.ring_bak, sR * NLAYER) != cudaSuccess) return -1;
    if (cudaMalloc(&bt.ple_ring_bak, qf_ple_ring_bytes()) != cudaSuccess) return -1;
    if (qf_prefill_init()) return -1;
    bt.cap = T;
    return 0;
}

// Snapshot every piece of per-sequence recurrent state a rejected draft must
// undo. KV caches are deliberately NOT saved: they are write-before-read per
// position and only [0, pos) is ever attended, so rewinding pos is enough.
int qf_spec_snapshot(QfModel *m) {
    if (!bt.cap) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD * 4, sR = (size_t)3 * DINN * 4;
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if (((il + 1) % 4) == 0) continue;         // full-attention layer: no GDN state
        if (cudaMemcpyAsync(bt.gdnS_bak + (size_t)il * (sS / 4), ctx.gdnS[il], sS,
                            cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
        if (cudaMemcpyAsync(bt.ring_bak + (size_t)il * (sR / 4), ctx.convring[il], sR,
                            cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
    }
    if (cudaMemcpyAsync(bt.ple_ring_bak, qf_ple_ring_ptr(), qf_ple_ring_bytes(),
                        cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
    for (int i = 0; i < 3; i++) bt.hist_bak[i] = g_hist[i];
    bt.have_snapshot = 1;
    return 0;
}

int qf_spec_restore(QfModel *m) {
    if (!bt.have_snapshot) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD * 4, sR = (size_t)3 * DINN * 4;
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if (((il + 1) % 4) == 0) continue;
        if (cudaMemcpyAsync(ctx.gdnS[il], bt.gdnS_bak + (size_t)il * (sS / 4), sS,
                            cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
        if (cudaMemcpyAsync(ctx.convring[il], bt.ring_bak + (size_t)il * (sR / 4), sR,
                            cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
    }
    if (cudaMemcpyAsync(qf_ple_ring_ptr(), bt.ple_ring_bak, qf_ple_ring_bytes(),
                        cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
    for (int i = 0; i < 3; i++) g_hist[i] = bt.hist_bak[i];
    return cudaStreamSynchronize(ctx.s) == cudaSuccess ? 0 : -1;
}

// Batched T-token trunk forward. Contract (docs/PHASE_SPEC_DECODE.md §2):
// the state left behind must equal T sequential qf_decode_step calls from the
// same start, and logits[t] must equal the logits of step t.
//
// tokens[0..T-1] are the ids to run at absolute positions pos0..pos0+T-1.
// logits_out is [T][NVOCAB] (bt.logits). Returns 0 on success.
// One token's hyper-connection, byte-identical to the trunk decode path.
// The batched qf_hc_prefill_f32w helper in prefill.cu takes f32 weight pointers;
// production HC weights are BF16, or FP8 slabs once g_hc_fp8 is set, so it
// cannot be used here. Batching the HC needs BF16/FP8 batched kernels - not
// written yet, so the T tokens loop. The HC is a small share of the layer
// (2x 320-wide down/up) next to the dense and MoE work, which IS batched.
static void qf_hc_one(const float *R_t, const void *w_norm, const void *w_down,
                      const void *w_up, const void *w_inject,
                      float *mixed_t, float *inj_t, cudaStream_t s) {
    const int hcd = HCC * NEMBD;
    k_rmsnorm<<<HCC, 256, 0, s>>>(ctx.normed, R_t, (const __nv_bfloat16 *)w_norm, hcd, 1e-6f, NEMBD);
    if (g_hc_fp8) k_hc_down_fp8<<<HCL, 256, 0, s>>>(ctx.normed, (const uint8_t *)w_down, ctx.hc_d);
    else k_hc_down<<<(HCL + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)w_down, ctx.hc_d);
    k_zero<<<10, 256, 0, s>>>(mixed_t, NEMBD);
    const int use_inj = (w_inject && inj_t) ? 1 : 0;
    if (g_hc_fp8)
        hc_up_fp8_launch(ctx.normed, (const uint8_t *)w_up, ctx.hc_d, mixed_t, (const __nv_bfloat16 *)w_inject, inj_t, use_inj, s);
    else
        k_hc_up<<<(hcd + 7) / 8, 256, 0, s>>>(ctx.normed, (const __nv_bfloat16 *)w_up, ctx.hc_d,
            mixed_t, (const __nv_bfloat16 *)w_inject, inj_t, use_inj);
}

// T tokens through one hyper-connection with ONE pass over its weights.
// Falls back to the per-token form whenever the batched kernels do not apply:
// they are written against the FP8 slab layout and the column-major up kernel,
// which is what production runs (QF_FP8_MODE default, QF_HC_COL default 1).
static void qf_hc_T(const float *R_T, const void *w_norm, const void *w_down,
                    const void *w_up, const void *w_inject,
                    float *mixedT, float *injT, int T, cudaStream_t s) {
    const int hcd = HCC * NEMBD;
    if (!g_hc_fp8 || !g_hc_col) {
        for (int t = 0; t < T; t++)
            qf_hc_one(R_T + (size_t)t * hcd, w_norm, w_down, w_up, w_inject,
                      mixedT + (size_t)t * NEMBD, injT ? injT + (size_t)t * HCC : NULL, s);
        return;
    }
        // The norm is per token and reads only 20 KB of weight; it is ONE launch over
    // (HCC groups x T rows) - the same per-row arithmetic, minus T-1 launches.
    k_rmsnorm_M<<<dim3(HCC, T), 256, 0, s>>>(bt.hcW_normed, R_T,
                                             (const __nv_bfloat16 *)w_norm, hcd, 1e-6f, NEMBD);

    const int use_inj = (w_inject && injT) ? 1 : 0;
    #define HC_T_CASE(N) case N: \
        k_hc_down_fp8_T<N><<<HCL, 256, 0, s>>>(bt.hcW_normed, (const uint8_t *)w_down, bt.hcW_d320); \
        k_hc_up_fp8_col_T<N><<<NEMBD / 8, 256, 0, s>>>(bt.hcW_normed, (const uint8_t *)w_up, \
            bt.hcW_d320, mixedT, (const __nv_bfloat16 *)w_inject, injT, use_inj); \
        break
    switch (T) {
                HC_T_CASE(1); HC_T_CASE(2); HC_T_CASE(3); HC_T_CASE(4);
        HC_T_CASE(5); HC_T_CASE(6); HC_T_CASE(7); HC_T_CASE(8);
        HC_T_CASE(9); HC_T_CASE(10); HC_T_CASE(11); HC_T_CASE(12);
        HC_T_CASE(13); HC_T_CASE(14); HC_T_CASE(15); HC_T_CASE(16);
        default: break;
    }
    #undef HC_T_CASE
}

// Host half of the batched forward: everything that depends on THIS call's
// token ids and start position, and nothing that touches the GPU except
// through buffers the device half reads at execution time.
//
// It is split out because the device half is CUDA-graph captured. A graph
// replays device work only; host code inside the captured region runs once, at
// capture. So the per-call token/pos params and the per-position PLE n-gram
// gather have to happen BEFORE the launch, writing the same pinned buffers the
// captured nodes read - exactly the contract qf_graph_step already uses for
// single-step decode (qf_decode_set_params + qf_ple_stage_current, then launch).
static void qf_body_T_host(QfModel *m, const int *tokens, int T) {
    if (qf_layer_begin() > 1) return;         // PLE (layer 1) belongs to the other box
    for (int t = 0; t < T; t++) {
        // The n-gram window must contain this position's token before its hash
        // is taken, and the window slides in order, so this loop is the same
        // sequence a run of T qf_decode_step calls would produce.
        qf_hist_push(tokens[t]);
        qf_ple_stage_current_slot(m, t);
    }
}

static int qf_body_T_device(QfModel *m, int T) {
    // On a layer-split tail (lb > 0) the T residuals were pushed into bt.R by
    // qf_push_residual_M (the AMD prefix produced them); the body then runs
    // layers lb..47 - the chunked prompt pass of the alternating map.
    const int lb = qf_layer_begin();
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD;
    qfd_gemv_group_T_rewind();

    // 1. embed each token into its own 4-stream residual (head node only)
    if (lb == 0)
    for (int t = 0; t < T; t++)
        k_embed_dev<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.x, bt.R + (size_t)t * hcd,
            (const __nv_bfloat16 *)m->tok_embd, ctx.paramsT_dev + t);

    for (int il = lb; il < NLAYER; il++) {
        QfLayer *L = &m->layers[il];
        const int is_recr = ((il + 1) % 4) != 0;

        if (il == 1) {
            // PLE injects into every stream at layer 1, per token, in order.
            // The gather for slot t was staged by qf_body_T_host; k_ple_decode
            // reads that pinned slot zero-copy at EXECUTION time, so a replay
            // picks up the current round's rows.
            for (int t = 0; t < T; t++)
                qf_ple_apply_staged_slot(m, bt.R + (size_t)t * hcd, t, s);
        }

        // ---- attention-side hyper-connection ----
        qf_hc_T(bt.R, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up,
                L->hc_attn_inject, bt.mixed, bt.inj, T, s);

        if (is_recr) {
            // GDN in-projections: ONE weight read for all T tokens
            { QfDenseProj p[4] = {{L->qkv, bt.qkv_raw, DINN}, {L->zgate, bt.z6144, GDN_VDIM},
                                  {L->beta, bt.a48, DTRANK}, {L->alpha, bt.b48, DTRANK}};
              if (qfd_gemv_group_T(p, 4, bt.mixed, NEMBD, T, s)) { fprintf(stderr, "body_T: fail#2\n"); return -1; } }
            // The conv ring and the GDN state S are recurrent, so the tokens
            // walk in order. These are the trunk's own kernels: qf_conv_prefill_f32w
            // and qf_gdn_prefill_f32w take f32 weight pointers, but conv1d, a and
            // dt_bias are BF16 in production - the same mismatch that made the
            // batched HC read past its allocation.
            for (int t = 0; t < T; t++) {
                k_conv_step<<<(DINN + 1023) / 1024, 1024, 0, s>>>(
                    bt.qkv_raw + (size_t)t * DINN, ctx.convring[il],
                    (const __nv_bfloat16 *)L->conv1d);
                k_gdn_decode<<<GDN_VH, GDN_VD, 0, s>>>(
                    bt.qkv_raw + (size_t)t * DINN,
                    bt.a48 + (size_t)t * DTRANK, bt.b48 + (size_t)t * DTRANK,
                    (const __nv_bfloat16 *)L->a, (const __nv_bfloat16 *)L->dt_bias,
                    ctx.gdnS[il], bt.out48 + (size_t)t * GDN_VH * GDN_VD);
            }
            for (int t = 0; t < T; t++) {
                k_rmsnorm_gated<<<GDN_VH, GDN_VD, 0, s>>>(
                    bt.gdn_out_bf + (size_t)t * GDN_VH * GDN_VD,
                    bt.out48 + (size_t)t * GDN_VH * GDN_VD,
                    (const __nv_bfloat16 *)L->gdn_norm, bt.z6144 + (size_t)t * GDN_VDIM, 1e-6f);
                k_bf16_to_f32<<<(GDN_VDIM + 1023) / 1024, 1024, 0, s>>>(
                    bt.q6144 + (size_t)t * GDN_VDIM,
                    bt.gdn_out_bf + (size_t)t * GDN_VH * GDN_VD, GDN_VDIM);
            }
            { QfDenseProj p[1] = {{L->gdn_out, bt.y2560, NEMBD}};
              if (qfd_gemv_group_T(p, 1, bt.q6144, GDN_VDIM, T, s)) { fprintf(stderr, "body_T: fail#5\n"); return -1; } }
        } else {
            { QfDenseProj p[3] = {{L->wq, bt.q6144, NHEAD * QGATE},
                                  {L->wk, bt.k512, NKV * HDIM}, {L->wv, bt.v512, NKV * HDIM}};
              if (qfd_gemv_group_T(p, 3, bt.mixed, NEMBD, T, s)) { fprintf(stderr, "body_T: fail#6\n"); return -1; } }
            // q/k norm + rope + KV-cache write, one position at a time so each
            // token's rope angle and cache slot are its own.
            for (int t = 0; t < T; t++) {
                qf_attn_qsa_layer(bt.q6144 + (size_t)t * NHEAD * QGATE,
                                  bt.k512 + (size_t)t * NKV * HDIM,
                                  bt.v512 + (size_t)t * NKV * HDIM,
                                  ctx.kc[il], ctx.vc[il], L->q_norm, L->k_norm,
                                  ctx.inv_freq, ctx.paramsT_dev + t, NULL,
                                  bt.attn_out + (size_t)t * NHEAD * HDIM, s);
            }
            { QfDenseProj p[1] = {{L->wo, bt.y2560, NEMBD}};
              if (qfd_gemv_group_T(p, 1, bt.attn_out, NHEAD * HDIM, T, s)) { fprintf(stderr, "body_T: fail#7\n"); return -1; } }
        }
        for (int t = 0; t < T; t++)
            k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(
                bt.R + (size_t)t * hcd, bt.y2560 + (size_t)t * NEMBD, bt.inj + (size_t)t * HCC);

        // ---- FFN-side hyper-connection ----
        qf_hc_T(bt.R, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up,
                L->hc_ffn_inject, bt.mixed, bt.inj, T, s);
        { QfDenseProj p[4] = {{L->router, ctx.router, NEXP}, {L->shexp_gate, ctx.sh_g, NFF},
                              {L->shexp_up, ctx.sh_u, NFF}, {L->shexp_gate_inp, ctx.qkv_raw, 1}};
          // MoE in-projections are per token: the router picks a different
          // expert set for each, so nothing downstream is shared.
          for (int t = 0; t < T; t++) {
              qfd_moe_inproj(L->router, L->shexp_gate, L->shexp_up, L->shexp_gate_inp,
                             bt.mixed + (size_t)t * NEMBD, ctx.router, ctx.sh_g, ctx.sh_u,
                             ctx.qkv_raw, s);
              k_router_topk<<<1, 128, 0, s>>>(ctx.router, ctx.sel_dev, ctx.wts_dev);
              fprintf(stderr, "[svc] dispatch L%d\n", il);
        k_route_dispatch<<<1, 128, 0, s>>>(ctx.sel_dev,
                  L->exp_slot_dev, L->exp_expert_in_slot_dev, L->exp_slot_age_dev,
                  L->exp_clock_dev, L->route_slot_dev, L->slot_ready_dev,
                  L->route_mb_dev, ctx.params_dev, L->exp_cache_slots);
              nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_gate, 640 * 1280,
                  (const uint8_t *)L->exp_scale, 640 * 160, ctx.sel_dev, L->route_slot_dev,
                  L->s2_gate_dev, L->slot_ready_dev, ctx.route_err_dev,
                  bt.mixed + (size_t)t * NEMBD, ctx.eg_all, 640, 2560, 0, s);
              nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_up, 640 * 1280,
                  (const uint8_t *)L->exp_scale_up, 640 * 160, ctx.sel_dev, L->route_slot_dev,
                  L->s2_up_dev, L->slot_ready_dev, ctx.route_err_dev,
                  bt.mixed + (size_t)t * NEMBD, ctx.eu_all, 640, 2560, 0, s);
              k_silu_mul<<<(NEXPUSED * NFF + 255) / 256, 256, 0, s>>>(ctx.eg_all, ctx.eu_all, NEXPUSED * NFF);
              nvfp4_gemv_grouped_launch((const uint8_t *)L->exp_down, (size_t)NEMBD * NFF / 2,
                  (const uint8_t *)L->exp_scale_down, (size_t)2560 * 40, ctx.sel_dev,
                  L->route_slot_dev, L->s2_down_dev, L->slot_ready_dev, ctx.route_err_dev,
                  ctx.eg_all, ctx.ed_all, NEMBD, 640, 1, s);
              k_moe_accum<<<(NEMBD + 255) / 256, 256, 0, s>>>(
                  bt.y2560 + (size_t)t * NEMBD, ctx.ed_all, ctx.wts_dev, NEMBD);
              k_silu_mul<<<3, 256, 0, s>>>(ctx.sh_g, ctx.sh_u, NFF);
              qfd_out_proj(L->shexp_down, ctx.sh_g, ctx.ed, NEMBD, NFF, s);
              k_shexp_add<<<13, 256, 0, s>>>(bt.y2560 + (size_t)t * NEMBD, ctx.ed, ctx.qkv_raw);
          }
          (void)p; }
        for (int t = 0; t < T; t++)
            k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(
                bt.R + (size_t)t * hcd, bt.y2560 + (size_t)t * NEMBD, bt.inj + (size_t)t * HCC);
    }

    if (g_skip_head) return 0;              // alternating map: the head runs on the AMD box
    // ---- output mixer (no inject) + lm_head ----
    qf_hc_T(bt.R, m->output_hc_norm, m->output_hc_down, m->output_hc_up, NULL,
            bt.mixed, NULL, T, s);
    // The output head is the single largest dense read in the model (248320 x
    // 2560, 636 MB as an E4M3 slab - as much as every other dense tensor in
    // twelve layers). Running it once per candidate read it T times for a
    // verify that exists precisely to read weights once, so at T=4 the head
    // alone cost 2.5 GB where 636 MB was needed. bt.mixed is [T][NEMBD] and
    // bt.logits is [T][NVOCAB], exactly the strided layout the batched kernel
    // wants, so this is the same arithmetic in one weight pass.
    { QfDenseProj p[1] = {{m->lm_head, bt.logits, NVOCAB}};
      if (qfd_gemv_group_T(p, 1, bt.mixed, NEMBD, T, s) != 0)
          for (int t = 0; t < T; t++)
              qfd_lm_head(m->lm_head, bt.mixed + (size_t)t * NEMBD,
                          bt.logits + (size_t)t * NVOCAB, s); }
    return 0;
}

// ---- CUDA-graph capture of the batched body -------------------------------
//
// Single-step decode already replays one captured graph; this gives the verify
// pass the same submission shape. Measurement showed that graph capture alone
// did not materially reduce round time, so launches were not the dominant
// cost. The routed experts and recurrent GDN work still scale with T.
//
// T only takes the values 1..QF_SPEC_MAXT, so one graph per T is enough. The
// first call at a given T runs EAGER on purpose: every lazily-created resource
// on this path (the dense pointer pool and PLE workspace) must
// already exist, because an allocation during capture is illegal. The second
// call captures and launches.
static cudaGraphExec_t g_bodyT_exec[QF_SPEC_MAXT + 1];
static int g_bodyT_warm[QF_SPEC_MAXT + 1];
static int g_bodyT_graph = -1;

static int bodyT_graph_enabled(void) {
    if (g_bodyT_graph < 0) {
        const char *e = getenv("QF_SPEC_GRAPH");
        g_bodyT_graph = (e && !atoi(e)) ? 0 : 1;
        // QF_TIMING records timing events on the decode stream inside the
        // dense wrappers, which cannot be captured. Measuring and capturing
        // are mutually exclusive here; say so instead of failing at capture.
        if (g_bodyT_graph && getenv("QF_TIMING") && getenv("QF_TIMING")[0] == '1') {
            fprintf(stderr, "spec: QF_TIMING=1 -> batched verify stays EAGER (timing events cannot be captured)\n");
            g_bodyT_graph = 0;
        }
    }
    return g_bodyT_graph;
}

static int qf_bodyT_capture(QfModel *m, int T) {
    cudaStream_t s = ctx.s;
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    cudaGraph_t graph = NULL;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return -1;
    int rc = qf_body_T_device(m, T);
    cudaError_t e = cudaStreamEndCapture(s, &graph);
    if (rc != 0 || e != cudaSuccess || !graph) {
        fprintf(stderr, "spec: verify capture failed at T=%d (rc=%d, %s); staying eager\n",
                T, rc, cudaGetErrorString(e));
        if (graph) cudaGraphDestroy(graph);
        return -1;
    }
    e = cudaGraphInstantiateWithFlags(&g_bodyT_exec[T], graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
        fprintf(stderr, "spec: verify instantiate failed at T=%d: %s; staying eager\n",
                T, cudaGetErrorString(e));
        g_bodyT_exec[T] = NULL;
        return -1;
    }
    fprintf(stderr, "spec: captured the batched verify body at T=%d\n", T);
    return 0;
}

// M=8 batched QSA (full-attention) layer. The QKV projection is already batched
// across the M rows (qfd_qsa_inproj on bt.q6144/k512/v512); this drives the
// per-request attention core, each row against its OWN KV cache slice and its
// OWN position (paramsM_dev[m]). The attention reads per-request KV state, not
// big weights, so per-row dispatch re-reads no weight matrix. A grid.y=M fusion
// of the score/softmax kernels is a later refinement, not a correctness need.
void qf_attn_qsa_layer_M_base(int il, const void *q_norm, const void *k_norm,
                              const QfDecodeParams *paramsM_dev, int r0, int M, cudaStream_t s);
void qf_attn_qsa_layer_M(int il, const void *q_norm, const void *k_norm,
                         const QfDecodeParams *paramsM_dev, int M, cudaStream_t s) {
    qf_attn_qsa_layer_M_base(il, q_norm, k_norm, paramsM_dev, 0, M, s);
}
void qf_attn_qsa_layer_M_base(int il, const void *q_norm, const void *k_norm,
                              const QfDecodeParams *paramsM_dev, int r0, int M, cudaStream_t s) {
        const size_t kvspan = (size_t)qf_maxpos() * KVDIM;
    // Rows [r0, r0+M) in ONE launch per kernel (row = grid dimension), each
    // against its own KV slice and position. Falls back to the per-row form only
    // if the M-row workspace cannot be allocated.
    if (qf_attn_qsa_layer_Mgrid(bt.q6144 + (size_t)r0 * NHEAD * QGATE,
                                bt.k512 + (size_t)r0 * KVDIM, bt.v512 + (size_t)r0 * KVDIM,
                                g_kc_M[il] + (size_t)r0 * kvspan, g_vc_M[il] + (size_t)r0 * kvspan, kvspan,
                                q_norm, k_norm, ctx.inv_freq, paramsM_dev + r0,
                                bt.attn_out + (size_t)r0 * NHEAD * HDIM, M, s) == 0)
        return;
    for (int mm = 0; mm < M; mm++) {
        int m = r0 + mm;
        qf_attn_qsa_layer(bt.q6144 + (size_t)m * NHEAD * QGATE,
                          bt.k512 + (size_t)m * KVDIM,
                          bt.v512 + (size_t)m * KVDIM,
                          g_kc_M[il] + (size_t)m * kvspan,
                          g_vc_M[il] + (size_t)m * kvspan,
                          q_norm, k_norm, ctx.inv_freq, paramsM_dev + m, NULL,
                          bt.attn_out + (size_t)m * NHEAD * HDIM, s);
    }
}


// Slot form: transient bt rows 0..M-1, persistent KV rows [state_r0, +M).
// The two-slot pipeline runs slot s's tail on the shared scratch against the
// KV caches of rows [8s, 8s+8).
void qf_attn_qsa_layer_M_state(int il, const void *q_norm, const void *k_norm,
                               const QfDecodeParams *paramsM_dev, int state_r0, int M, cudaStream_t s) {
    const size_t kvspan = (size_t)qf_maxpos() * KVDIM;
    if (qf_attn_qsa_layer_Mgrid(bt.q6144, bt.k512, bt.v512,
                                g_kc_M[il] + (size_t)state_r0 * kvspan, g_vc_M[il] + (size_t)state_r0 * kvspan, kvspan,
                                q_norm, k_norm, ctx.inv_freq, paramsM_dev, bt.attn_out, M, s) == 0)
        return;
    for (int mm = 0; mm < M; mm++)
        qf_attn_qsa_layer(bt.q6144 + (size_t)mm * NHEAD * QGATE,
                          bt.k512 + (size_t)mm * KVDIM, bt.v512 + (size_t)mm * KVDIM,
                          g_kc_M[il] + (size_t)(state_r0 + mm) * kvspan,
                          g_vc_M[il] + (size_t)(state_r0 + mm) * kvspan,
                          q_norm, k_norm, ctx.inv_freq, paramsM_dev + mm, NULL,
                          bt.attn_out + (size_t)mm * NHEAD * HDIM, s);
}

// ===========================================================================
// M=8 multi-request batched decode driver. Canonical workload (M=1 banned).
// One row per independent request; each keeps its OWN recurrent GDN state,
// conv ring, QSA KV cache, position and routing. Row-batchable ops (embed, HC,
// dense projections, lm_head) read each weight ONCE across the M rows; the
// per-request attention uses the M-fold state kernels (k_gdn_decode_M /
// qf_attn_qsa_layer_M). At the MoE fork the routed experts are dispatched to
// the four-MI50 tier (qf_routed_amd_layer_submit) and Spark computes the shared
// expert concurrently, joining at k_shexp_add. When the AMD offload is disabled
// the engine's local grouped routed path runs (the authority compute, not a
// slow fallback). tokensM/posM are [M]; logits land in bt.logits[M][NVOCAB].
// ===========================================================================
extern "C" int qf_routed_amd_layer_submit(int il, long long pos, int M, int K,
                                          const int *sel_dev, const float *wt_dev,
                                          const float *mixed_dev, cudaStream_t s);
extern "C" int qf_routed_amd_layer_wait(float *y2560_dev, int M, cudaStream_t s);
extern "C" int qf_routed_amd_enabled(void);
extern "C" int qf_region_amd_enabled(void);
extern "C" int qf_region_amd_reset(void);
extern "C" int qf_region_amd_submit_resid(long long pos, int M, const long *posM, const float *hR, int rsize);
extern "C" int qf_region_amd_prefix(int M, const int *tokens, const long *posM, float *rout, int rsize);
extern "C" int qf_region_amd_head(int M, const float *rin, int rsize, int *tokens_out);
extern "C" int qf_push_residual_M(const float *hR, int M);
extern "C" int qf_region_amd_collect(int M, int *tokens_out);
#include <time.h>
static double g_t_offload = 0, g_t_layer = 0; static long g_off_calls = 0;
static double m8_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
extern "C" void qf_m8_timing(double*off,double*lay,long*n){ *off=g_t_offload; *lay=g_t_layer; *n=g_off_calls; }
// Alternating-map wall-clock split (QF_M8_TIMING): host time inside the AMD
// prefix RPC, the Spark tail (incl. the residual H2D/D2H), and the AMD head
// RPC, summed over token-steps. The AMD ms/wave numbers do not exist yet; this
// is how the first receipt of the map produces them.
static double g_t_prefix = 0, g_t_tail = 0, g_t_head = 0; static long g_region_steps = 0;
extern "C" void qf_m8_region_timing(double *prefix, double *tail, double *head, long *steps) {
    *prefix = g_t_prefix; *tail = g_t_tail; *head = g_t_head; *steps = g_region_steps;
}

// Host half: fill the pinned per-row params. The device half (eager or the
// captured graph's first node) copies them; a graph replay re-reads this
// pinned buffer, so filling it is all a replay needs.
extern "C" void qf_decode_fill_params_M(const int *tokensM, const long *posM, int M) {
    for (int m = 0; m < M; m++) {
        ctx.paramsT_host[m].token = tokensM[m];
        ctx.paramsT_host[m].pos = (int)posM[m];
        ctx.paramsT_host[m].seq = ++ctx.route_seq;
        ctx.paramsT_host[m].flags = 0;
    }
}
extern "C" void qf_decode_set_params_M(const int *tokensM, const long *posM, int M) {
    qf_decode_fill_params_M(tokensM, posM, M);
    cudaMemcpyAsync(ctx.paramsT_dev, ctx.paramsT_host,
                    (size_t)M * sizeof(QfDecodeParams), cudaMemcpyHostToDevice, ctx.s);
}

// lend: run layers [0, lend). NLAYER = whole model (single-box). Under the
// established cut the Spark runs the head-side run and hands ONE residual to
// the AMD tail, which owns the remaining layers + output HC + lm_head + argmax.
extern "C" int qf_decode_step_M_upto(QfModel *m, const int *tokensM, const long *posM, int M, int lend);
int qf_decode_step_M(QfModel *m, const int *tokensM, const long *posM, int M) {
    return qf_decode_step_M_upto(m, tokensM, posM, M, NLAYER);
}
// Device half of the M step: launches only - no host sync, no allocation, no
// socket - so it runs eagerly or is captured once and replayed as ONE graph
// launch per token-step (W1 step 5). Params must already be in the pinned
// ctx.paramsT_host; the first node copies them, and a replay re-reads them.
// On a layer-split node (qf_layer_begin() > 0) the embed is skipped: the caller
// pushed the residual for the layers this box does not own (qf_push_residual_M)
// - that is the alternating map's Spark tail. want_logits=0 skips the output
// HC + lm_head (the head runs on the AMD box in that map).
// r0 = row base of the PERSISTENT per-row state (GDN S, conv ring, KV) this
// step advances; the transient batch scratch is always rows 0..M-1. The
// two-slot pipeline steps slot s with r0 = 8s.
static int qf_decode_step_M_device(QfModel *m, int M, int lend, int want_logits, int r0) {
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    qfd_gemv_group_T_rewind();
    if (cudaMemcpyAsync(ctx.paramsT_dev, ctx.paramsT_host, (size_t)M * sizeof(QfDecodeParams),
                        cudaMemcpyHostToDevice, s) != cudaSuccess) return -1;


        if (qf_layer_begin() == 0)
        k_embed_dev_M<<<dim3((NEMBD + 255) / 256, M), 256, 0, s>>>(bt.R,
            (const __nv_bfloat16 *)m->tok_embd, ctx.paramsT_dev);

    // The local routed path resolves slots as map[expert] with no miss service
    // (k_route_dispatch_M); it is only valid with every expert resident.
    static int local_full = -1;
    if (local_full < 0) {
        local_full = 1;
                for (int il = qf_layer_begin(); il < NLAYER; il++)
            if (m->layers[il].exp_cache_slots < NEXP) local_full = 0;
        if (!local_full)
            fprintf(stderr, "step_M: experts are NOT fully resident; the local routed M path "
                            "requires QF_SPARK_MODE_FULL (slot==expert)\n");
    }

    for (int il = qf_layer_begin(); il < lend; il++) {

        QfLayer *L = &m->layers[il];
        const int is_recr = ((il + 1) % 4) != 0;
        if (il == 1)
            for (int r = 0; r < M; r++) qf_ple_apply_staged_slot(m, bt.R + (size_t)r * hcd, r, s);

        qf_hc_T(bt.R, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up,
                L->hc_attn_inject, bt.mixed, bt.inj, M, s);

        if (is_recr) {
            { QfDenseProj p[4] = {{L->qkv, bt.qkv_raw, DINN}, {L->zgate, bt.z6144, GDN_VDIM},
                                  {L->beta, bt.a48, DTRANK}, {L->alpha, bt.b48, DTRANK}};
              if (qfd_gemv_group_T(p, 4, bt.mixed, NEMBD, M, s)) return -1; }
                        k_conv_step_M<<<dim3((DINN + 1023) / 1024, M), 1024, 0, s>>>(
                bt.qkv_raw, g_convring_M[il] + (size_t)r0 * 3 * DINN, (const __nv_bfloat16 *)L->conv1d);
            k_gdn_decode_M<<<dim3(GDN_VH, M), GDN_VD, 0, s>>>(
                bt.qkv_raw, bt.a48, bt.b48, (const __nv_bfloat16 *)L->a,
                (const __nv_bfloat16 *)L->dt_bias, g_gdnS_M[il] + (size_t)r0 * sS, bt.out48);
                        k_rmsnorm_gated_M<<<dim3(GDN_VH, M), GDN_VD, 0, s>>>(
                bt.gdn_out_bf, bt.out48, (const __nv_bfloat16 *)L->gdn_norm, bt.z6144, 1e-6f);
            k_bf16_to_f32_M<<<dim3((GDN_VDIM + 1023) / 1024, M), 1024, 0, s>>>(
                bt.q6144, bt.gdn_out_bf, GDN_VDIM);
            { QfDenseProj p[1] = {{L->gdn_out, bt.y2560, NEMBD}};
              if (qfd_gemv_group_T(p, 1, bt.q6144, GDN_VDIM, M, s)) return -1; }

        } else {
            { QfDenseProj p[3] = {{L->wq, bt.q6144, NHEAD * QGATE},
                                  {L->wk, bt.k512, NKV * HDIM}, {L->wv, bt.v512, NKV * HDIM}};
              if (qfd_gemv_group_T(p, 3, bt.mixed, NEMBD, M, s)) return -1; }
                        qf_attn_qsa_layer_M_state(il, L->q_norm, L->k_norm, ctx.paramsT_dev, r0, M, s);
            { QfDenseProj p[1] = {{L->wo, bt.y2560, NEMBD}};
              if (qfd_gemv_group_T(p, 1, bt.attn_out, NHEAD * HDIM, M, s)) return -1; }
        }
        k_stream_inject_M<<<dim3((hcd + 1023) / 1024, M), 1024, 0, s>>>(bt.R, bt.y2560, bt.inj);

        // ---- FFN-side HC (batched) then the MoE fork ----
        qf_hc_T(bt.R, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up,
                L->hc_ffn_inject, bt.mixed, bt.inj, M, s);
        // router + shared in-projection: ONE weight read for all M rows, then one
        // top-k launch (block per row). Falls back to the per-row form only if the
        // batched dense kernel is ineligible (non-FP8 dense weights).
        { QfDenseProj p[4] = {{L->router, g_routerM, NEXP}, {L->shexp_gate, g_shgM, NFF},
                              {L->shexp_up, g_shuM, NFF}, {L->shexp_gate_inp, g_ginpM, 1}};
          if (qfd_gemv_group_T(p, 4, bt.mixed, NEMBD, M, s) == 0) {
              k_router_topk_M<<<M, 128, 0, s>>>(g_routerM, g_selM, g_wtsM);
          } else {
              for (int r = 0; r < M; r++) {
                  qfd_moe_inproj(L->router, L->shexp_gate, L->shexp_up, L->shexp_gate_inp,
                                 bt.mixed + (size_t)r * NEMBD, ctx.router,
                                 g_shgM + (size_t)r * NFF, g_shuM + (size_t)r * NFF, g_ginpM + r, s);
                  k_router_topk<<<1, 128, 0, s>>>(ctx.router, g_selM + (size_t)r * NEXPUSED,
                                                  g_wtsM + (size_t)r * NEXPUSED);
              }
          } }

        // routed branch: AMD four-card tier, overlapping the Spark shared branch
        int offloaded = 0;
        int m8_time = getenv("QF_M8_TIMING") ? 1 : 0;
        double t_off0 = 0;
        if (m8_time) { cudaStreamSynchronize(s); t_off0 = m8_now(); }
        if (qf_routed_amd_enabled() &&
            qf_routed_amd_layer_submit(il, (long long)ctx.paramsT_host[0].pos, M, NEXPUSED, g_selM, g_wtsM, bt.mixed, s) == 0)
            offloaded = 1;
                // shared expert (Spark): silu over the [M][NFF] slab in one launch, then ONE
        // down-projection weight read for all M rows. Runs concurrently with the
        // AMD routed branch when offloaded.
        k_silu_mul<<<(M * NFF + 255) / 256, 256, 0, s>>>(g_shgM, g_shuM, M * NFF);
        { QfDenseProj p[1] = {{L->shexp_down, g_edM, NEMBD}};
          if (qfd_gemv_group_T(p, 1, g_shgM, NFF, M, s) != 0)
              for (int r = 0; r < M; r++)
                  qfd_out_proj(L->shexp_down, g_shgM + (size_t)r * NFF, g_edM + (size_t)r * NEMBD, NEMBD, NFF, s); }
        if (offloaded && qf_routed_amd_layer_wait(bt.y2560, M, s) != 0) offloaded = 0;
        if (m8_time && offloaded) { g_t_offload += m8_now() - t_off0; g_off_calls++; }
        if (!offloaded) {
            // Local grouped routed path (authority compute) -> bt.y2560: every
            // (row, expert) pair of the batch in ONE launch per projection. Rows
            // have their own slot/hidden buffers (g_rslotM, g_e*M_all), so they
            // no longer serialize through single-instance scratch.
            if (!local_full) return -1;
            k_route_dispatch_M<<<M, 128, 0, s>>>(g_selM, L->exp_slot_dev, g_rslotM, ctx.route_err_dev);
            qf_dbg_probe("m8_dispatch", il);
            const int *permM = NULL;
            if (moe_sort_on() && g_permM) { k_moe_sort_pairs_M<<<1, 256, 0, s>>>(g_selM, M * NEXPUSED, g_permM); permM = g_permM; }
            qf_dbg_probe("m8_gate", il);
            nvfp4_gemv_grouped_launch_M((const uint8_t *)L->exp_gate, 640 * 1280,
                (const uint8_t *)L->exp_scale, 640 * 160, g_selM, g_rslotM,
                L->s2_gate_dev, L->slot_ready_dev, ctx.route_err_dev,
                bt.mixed, g_egM_all, 640, 2560, 0, M, permM, s);
            nvfp4_gemv_grouped_launch_M((const uint8_t *)L->exp_up, 640 * 1280,
                (const uint8_t *)L->exp_scale_up, 640 * 160, g_selM, g_rslotM,
                L->s2_up_dev, L->slot_ready_dev, ctx.route_err_dev,
                bt.mixed, g_euM_all, 640, 2560, 0, M, permM, s);
            k_silu_mul<<<(M * NEXPUSED * NFF + 255) / 256, 256, 0, s>>>(g_egM_all, g_euM_all, M * NEXPUSED * NFF);
            qf_dbg_probe("m8_silu", il);
            nvfp4_gemv_grouped_launch_M((const uint8_t *)L->exp_down, (size_t)NEMBD * NFF / 2,
                (const uint8_t *)L->exp_scale_down, (size_t)2560 * 40, g_selM, g_rslotM,
                L->s2_down_dev, L->slot_ready_dev, ctx.route_err_dev,
                g_egM_all, g_edM_all, NEMBD, 640, 1, M, permM, s);
            k_moe_accum_M<<<dim3((NEMBD + 255) / 256, M), 256, 0, s>>>(bt.y2560, g_edM_all, g_wtsM, NEMBD);
            qf_dbg_probe("m8_accum", il);
        }
        // join the shared expert into the routed output, then inject: one launch each
        k_shexp_add_M<<<dim3((NEMBD + 255) / 256, M), 256, 0, s>>>(bt.y2560, g_edM, g_ginpM);
        k_stream_inject_M<<<dim3((hcd + 1023) / 1024, M), 1024, 0, s>>>(bt.R, bt.y2560, bt.inj);
    }


        if (want_logits) {
        qf_hc_T(bt.R, m->output_hc_norm, m->output_hc_down, m->output_hc_up, NULL, bt.mixed, NULL, M, s);
        { QfDenseProj p[1] = {{m->lm_head, bt.logits, NVOCAB}};
          if (qfd_gemv_group_T(p, 1, bt.mixed, NEMBD, M, s) != 0)
              for (int r = 0; r < M; r++)
                  qfd_lm_head(m->lm_head, bt.mixed + (size_t)r * NEMBD, bt.logits + (size_t)r * NVOCAB, s); }
    }
    return 0;
}

// ---- CUDA-graph replay of the M step (QF_M8_GRAPH, default on) --------------
// The M=8 body is ~7000 launches per token-step and the measured step is 205 ms
// against 32 ms of bytes: launch-bound. After the M-row kernels above (W1
// steps 1-4) the remaining launches are captured once per (M, layer range,
// logits) after one warm eager pass, and replayed as a single cudaGraphLaunch.
// Legality: no host sync, allocation, or socket inside the device half. The
// per-layer AMD routed offload (QF_ROUTED_AMD) syncs the stream and blocks on
// a socket per layer, so it stays eager; so do the timing modes, whose events
// cannot be captured.
// One exec per (slot, M): a slot's persistent-state pointers are baked into
// its graph, so the two-slot pipeline replays two graphs alternately.
#define QF_M8_NSLOTS  2
#define QF_M8_MAXROWS (QF_M8_NSLOTS * QF_SPEC_MAXT)
static cudaGraphExec_t g_stepM_exec[QF_M8_NSLOTS][QF_SPEC_MAXT + 1];
static int g_stepM_warm[QF_M8_NSLOTS][QF_SPEC_MAXT + 1];   // 0 never run, 1 warm (capturable), 2 eager for good
static int g_stepM_key[QF_M8_NSLOTS][QF_SPEC_MAXT + 1];
static int stepM_graph_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("QF_M8_GRAPH");
        on = (e && !atoi(e)) ? 0 : 1;
        if (on && qf_routed_amd_enabled()) {
            fprintf(stderr, "step_M: AMD routed offload is on -> eager (its submit/wait cannot be captured)\n"); on = 0; }
        if (on && getenv("QF_M8_TIMING")) {
            fprintf(stderr, "step_M: QF_M8_TIMING -> eager (per-layer syncs)\n"); on = 0; }
        if (on && getenv("QF_TIMING") && getenv("QF_TIMING")[0] == '1') {
            fprintf(stderr, "step_M: QF_TIMING=1 -> eager (timing events cannot be captured)\n"); on = 0; }
    }
    return on;
}
static int qf_stepM_capture(QfModel *m, int M, int lend, int want_logits, int r0, int key) {
    cudaStream_t s = ctx.s;
    const int si = r0 / QF_SPEC_MAXT;
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    if (g_stepM_exec[si][M]) { cudaGraphExecDestroy(g_stepM_exec[si][M]); g_stepM_exec[si][M] = NULL; }
    cudaGraph_t graph = NULL;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return -1;
    int rc = qf_decode_step_M_device(m, M, lend, want_logits, r0);
    cudaError_t e = cudaStreamEndCapture(s, &graph);
    if (rc != 0 || e != cudaSuccess || !graph) {
        fprintf(stderr, "step_M: capture failed at M=%d (rc=%d, %s); staying eager\n",
                M, rc, cudaGetErrorString(e));
        if (graph) cudaGraphDestroy(graph);
        return -1;
    }
    e = cudaGraphInstantiateWithFlags(&g_stepM_exec[si][M], graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
        fprintf(stderr, "step_M: instantiate failed at M=%d: %s; staying eager\n", M, cudaGetErrorString(e));
        g_stepM_exec[si][M] = NULL;
        return -1;
    }
    g_stepM_key[si][M] = key;
    fprintf(stderr, "step_M: captured the M=%d step (layers %d..%d%s, state rows %d..%d) as ONE graph launch\n",
            M, qf_layer_begin(), lend - 1, want_logits ? " + head" : "", r0, r0 + M - 1);
    return 0;
}
extern "C" int qf_decode_step_M_slot(QfModel *m, const int *tokensM, const long *posM,
                                     int M, int lend, int want_logits, int r0) {
    if (M < 1 || M > QF_SPEC_MAXT) { fprintf(stderr, "step_M: bad M=%d\n", M); return -1; }
    if (r0 < 0 || (r0 % QF_SPEC_MAXT) != 0 || r0 + M > QF_M8_MAXROWS) { fprintf(stderr, "step_M: bad row base %d\n", r0); return -1; }
    const int si = r0 / QF_SPEC_MAXT;
    if (lend < 1 || lend > NLAYER) { fprintf(stderr, "step_M: bad lend=%d\n", lend); return -1; }
    if (qf_batch_init(QF_SPEC_MAXT)) { fprintf(stderr, "step_M: batch init failed\n"); return -1; }
    if (qf_reqbatch_state_init(r0 + M)) { fprintf(stderr, "step_M: reqbatch state init failed\n"); return -1; }
    cudaStream_t s = ctx.s;
    qf_decode_fill_params_M(tokensM, posM, M);
    const int key = (lend << 8) | (r0 << 3) | ((want_logits ? 1 : 0) << 1) | (qf_layer_begin() > 0 ? 1 : 0);
    if (stepM_graph_enabled() && g_stepM_warm[si][M] == 1) {
        if (!g_stepM_exec[si][M] || g_stepM_key[si][M] != key) {
            if (qf_stepM_capture(m, M, lend, want_logits, r0, key) != 0) g_stepM_warm[si][M] = 2;
        }
        if (g_stepM_exec[si][M] && g_stepM_key[si][M] == key) {
            if (cudaGraphLaunch(g_stepM_exec[si][M], s) != cudaSuccess) return -1;
            return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
        }
    }
    int rc = qf_decode_step_M_device(m, M, lend, want_logits, r0);
    if (g_stepM_warm[si][M] == 0) g_stepM_warm[si][M] = 1;   // lazily-created resources now exist
    if (rc != 0) return -1;
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}
extern "C" int qf_decode_step_M_upto_ex(QfModel *m, const int *tokensM, const long *posM,
                                        int M, int lend, int want_logits) {
    return qf_decode_step_M_slot(m, tokensM, posM, M, lend, want_logits, 0);
}
extern "C" int qf_decode_step_M_upto(QfModel *m, const int *tokensM, const long *posM, int M, int lend) {
    return qf_decode_step_M_slot(m, tokensM, posM, M, lend, 1, 0);
}


// Hand the M residuals to the caller for the ONE per-token crossing.
extern "C" int qf_push_residual_M(const float *hR, int M) {
    const int hcd = HCC * NEMBD;
    // The batch buffers are allocated lazily by the first step; the alternating
    // map pushes a residual BEFORE its first step, so allocate here.
    if (!bt.cap && qf_batch_init(QF_SPEC_MAXT)) { fprintf(stderr, "push_residual_M: batch init failed\n"); return -1; }
    if (M < 1 || M > bt.cap) { fprintf(stderr, "push_residual_M: bad M=%d (cap %d)\n", M, bt.cap); return -1; }
    return cudaMemcpy(bt.R, hR, (size_t)M * hcd * sizeof(float), cudaMemcpyHostToDevice)
           == cudaSuccess ? 0 : -1;
}
extern "C" int qf_pull_residual_M_impl(float *hR, int M) {
    const int hcd = HCC * NEMBD;
    if (!bt.cap || M < 1 || M > bt.cap) { fprintf(stderr, "pull_residual_M: bad M=%d (cap %d)\n", M, bt.cap); return -1; }
    return cudaMemcpy(hR, bt.R, (size_t)M * hcd * sizeof(float), cudaMemcpyDeviceToHost)
           == cudaSuccess ? 0 : -1;
}
extern "C" int qf_pull_residual_M(float *hR, int M) { return qf_pull_residual_M_impl(hR, M); }

// Per-row PLE staging: swap this row's 3-token history into the hash window,
// stage slot `slot`, restore. Serial host prep before the batched device step.
extern int qf_ple_prefetch_ids(QfModel *m, const int64_t *ids);   // ple.cu: MADV_WILLNEED the 16 table rows
void qf_ple_prefetch_row(QfModel *m, const int *hist3) {
    int save0 = g_hist[0], save1 = g_hist[1], save2 = g_hist[2];
    g_hist[0] = hist3[0]; g_hist[1] = hist3[1]; g_hist[2] = hist3[2];
    int64_t ids[PLE_HEADS];
    ple_hash(ids);
    qf_ple_prefetch_ids(m, ids);
    g_hist[0] = save0; g_hist[1] = save1; g_hist[2] = save2;
}
void qf_ple_stage_row(QfModel *m, const int *hist3, int slot) {
    int save0 = g_hist[0], save1 = g_hist[1], save2 = g_hist[2];
    g_hist[0] = hist3[0]; g_hist[1] = hist3[1]; g_hist[2] = hist3[2];
    qf_ple_stage_current_slot(m, slot);
    g_hist[0] = save0; g_hist[1] = save1; g_hist[2] = save2;
}

// M=8 multi-request DECODE serving loop. Each row is an independent sequence
// primed to (init_tok[r], init_pos[r], init_hist[r]); the loop batches one
// step across all active rows via qf_decode_step_M, samples greedily per row,
// advances position/history, and retires a row on EOS (its recurrent state is
// zeroed for slot reuse). out_tokens is [M*max_new], out_len[r] the count.
// Canonical operation M=8. (Prompt prefill priming and admission of fresh
// requests into freed slots are the serving harness's job around this loop.)
int qf_batch_argmax(int t);   // fwd decl (defined below with the batched body)
extern "C" int qf_decode_step_M_ab(QfModel *m, const int *tokensM, const long *posM, int M);

extern "C" { volatile long qf_m8_steps_done = 0; }   // watchdog progress (main.cu)
void qf_ple_prefetch_row(QfModel *m, const int *hist3);
extern "C" int qf_decode_batch_run(QfModel *m, const int *init_tok, const long *init_pos,
                        const int (*init_hist)[3], int M, int max_new,
                        int *out_tokens, int *out_len) {
    if (M < 1 || M > QF_SPEC_MAXT) return -1;
    static int ple_pf = -1;
    if (ple_pf < 0) { const char *e = getenv("QF_PLE_PREFETCH"); ple_pf = (e && !atoi(e)) ? 0 : 1; }
    double t_ple = 0, t_step = 0, t_amx = 0; long nsteps = 0;
    int  active[QF_SPEC_MAXT], tok[QF_SPEC_MAXT], hist[QF_SPEC_MAXT][3];
    long pos[QF_SPEC_MAXT];
    for (int r = 0; r < M; r++) {
        active[r] = 1; tok[r] = init_tok[r]; pos[r] = init_pos[r];
        hist[r][0] = init_hist[r][0]; hist[r][1] = init_hist[r][1]; hist[r][2] = init_hist[r][2];
        out_len[r] = 0;
    }
    for (int step = 0; step < max_new; step++) {
        const double ta = m8_now();
        int any = 0;
        for (int r = 0; r < M; r++) if (active[r]) {
            // Reproduce qf_decode_step EXACTLY: push the INPUT token being decoded
            // this step, THEN stage PLE from the updated history (the intentional
            // one-token-stale gather), THEN the forward runs in qf_decode_step_M.
            hist[r][0] = hist[r][1]; hist[r][1] = hist[r][2]; hist[r][2] = tok[r];
            any = 1;
        }
        if (!any) break;
        // The 16 rows' n-gram gathers are demand-faulted from the host-mapped
        // table while the GPU sits idle between graph launches: start all of
        // this step's page reads first, then stage (the copies then overlap).
        if (ple_pf) for (int r = 0; r < M; r++) if (active[r]) qf_ple_prefetch_row(m, hist[r]);
        for (int r = 0; r < M; r++) if (active[r]) qf_ple_stage_row(m, hist[r], r);
        const double tb = m8_now();
        static int ab = -1;
        if (ab < 0) ab = (getenv("QF_M8_AB") && M == 8) ? 1 : 0;
                if (ab ? qf_decode_step_M_ab(m, tok, pos, M) : qf_decode_step_M(m, tok, pos, M)) return -1;
        const double tc = m8_now();
        int tM[QF_SPEC_MAXT];
        if (qf_batch_argmax_M(bt.logits, M, tM) != 0) return -1;   // all rows, one sync
        const double td = m8_now();
        t_ple += tb - ta; t_step += tc - tb; t_amx += td - tc; nsteps++; qf_m8_steps_done++;
        for (int r = 0; r < M; r++) {
            if (!active[r]) continue;
            int t = tM[r];
            if (t < 0) return -1;

            out_tokens[(size_t)r * max_new + out_len[r]] = t;
            out_len[r]++;
            pos[r]++; tok[r] = t;          // becomes next step's input; pushed at the top
            // Authority termination is a fixed token count (no EOS break). Slot
            // retirement is opt-in (QF_M8_RETIRE=1) so it never contaminates the
            // canonical fixed-length receipt. A chat turn ends on <|im_end|>
            // (248046 in this checkpoint), not only <|endoftext|> (248044):
            // retire on either, else a serving run burns its whole budget after
            // every row has already answered.
            static int m8_retire = -1;
            if (m8_retire < 0) m8_retire = getenv("QF_M8_RETIRE") ? 1 : 0;
            if (m8_retire && (t == QF_EOS_ID || t == 248046)) { active[r] = 0; qf_reqbatch_state_reset_row(r, ctx.s); }
        }
    }
    if (nsteps)
        fprintf(stderr, "m8 host split: %ld steps: PLE stage %.2f ms/step (host, GPU idle), step %.2f ms/step "
                        "(graph launch + sync), argmax+D2H %.2f ms/step\n",
                nsteps, t_ple / nsteps, t_step / nsteps, t_amx / nsteps);
    return 0;
}


// M=8 prompt-prefill priming. Each request's prompt is run through the proven
// single-request decode path (which advances GDN S, conv ring, KV cache and PLE
// history correctly), then its final per-request state is snapshotted into that
// request's M-fold slot. Priming is inherently per-request (distinct prompts);
// the snapshot copy happens once, not in the decode hot path. On return each row
// r is primed to run tok_out[r] (its last prompt token) at pos_out[r], with
// hist_out[r] its PLE history. ids[r] has len[r] tokens (>=1).
extern "C" int qf_decode_batch_prime(QfModel *m, const int *const *ids, const int *len,
                                     int M, int *tok_out, long *pos_out, int (*hist_out)[3]) {
    if (M < 1 || M > QF_SPEC_MAXT) return -1;
    if (qf_reqbatch_state_init(M)) return -1;
    cudaStream_t s = ctx.s;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    const size_t span = (size_t)qf_maxpos() * KVDIM;
    for (int r = 0; r < M; r++) {
        int L = len[r];
        if (L < 1) return -1;
        qf_session_reset(m);
        qf_hist_reset();
        for (int i = 0; i < L - 1; i++) {
            fprintf(stderr, "[prime] row %d tok %d/%d\n", r, i, L - 1);
            if (qf_decode_step(m, ids[r][i], i) != 0) { fprintf(stderr, "prime r%d tok%d failed\n", r, i); return -1; }
        }
        for (int il = 0; il < NLAYER; il++) {
            if (((il + 1) % 4) != 0) {
                cudaMemcpyAsync(g_gdnS_M[il] + (size_t)r * sS, ctx.gdnS[il], sS * sizeof(float), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_convring_M[il] + (size_t)r * 3 * DINN, ctx.convring[il], (size_t)3 * DINN * sizeof(float), cudaMemcpyDeviceToDevice, s);
            } else if (L - 1 > 0) {
                size_t nb = (size_t)(L - 1) * KVDIM * 2;
                cudaMemcpyAsync(g_kc_M[il] + (size_t)r * span, ctx.kc[il], nb, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_vc_M[il] + (size_t)r * span, ctx.vc[il], nb, cudaMemcpyDeviceToDevice, s);
            }
        }
        if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
        tok_out[r] = ids[r][L - 1];
        pos_out[r] = L - 1;
        hist_out[r][0] = g_hist[0]; hist_out[r][1] = g_hist[1]; hist_out[r][2] = g_hist[2];
    }
    return 0;
}


// ===========================================================================
// LARGE-CHUNK prompt pass on the tail: T <= QF_PF_MAXT consecutive positions of
// the one sequence, residuals from the AMD prefix. The templated T-row kernels
// (hyper-connection, dense projection groups) run over 16-row sub-tiles with
// their weights read once per sub-tile; the routed experts run ONCE per chunk
// through the CSR kernels (qf_prefill_moe.cu: every expert row read once per
// up-to-16 routed tokens); the recurrent parts (conv ring, GDN state, KV +
// positions) advance in order per token exactly as qf_decode_body_T does.
// ===========================================================================
#define QF_PF_MAXT 256
#define QF_PF_ST   16
extern "C" void qf_prefill_experts_csr(const void *Wg, const void *Sg, const void *Wu, const void *Su,
                            const void *Wd, const void *Sd, const float *s2g, const float *s2u, const float *s2d,
                            const int *sel, const float *wt, int T, const float *x, float *hidden, float *y,
                            int *exp_slot, int *exp_ptr, int *pair_tok, float *pair_wt, int *counts, cudaStream_t s);
extern "C" void qf_prefill_experts_mma(const void *Wg, const void *Sg, const void *Wu, const void *Su,
                            const void *Wd, const void *Sd, const float *s2g, const float *s2u, const float *s2d,
                            const int *sel, const float *wt, int T, const float *x, float *hidden, float *y,
                            int *exp_slot, int *exp_ptr, int *pair_tok, float *pair_wt, int *counts,
                            uint8_t *xq, uint8_t *xs, uint8_t *hq, uint8_t *hs, float *part, cudaStream_t s);
extern "C" int qf_fp4mma_available(void);
extern "C" size_t qf_gdn2_scratch_floats(int maxT);
extern "C" int qf_gdn2_prefill(const float *qkv, const float *a, const float *b, const float *A_log, const float *dt_bias,
                               float *S, float *out48, int T, float *scratch, int maxT, cudaStream_t s);
static int pf_gdn2(void) { static int v = -1; if (v < 0) { const char *e = getenv("QF_PF_GDN2"); v = (e && e[0] == '1') ? 1 : 0; } return v; }   // opt-in: the two-level form measured slower (224-410 vs 178 ms per 1024 tokens)
extern "C" int qfd_gemm_fp8_rows(const QfDenseProj *projs, int nproj, const float *xT, int in, int T,
                                 uint8_t *xq, uint8_t *xs, cudaStream_t s);
extern "C" int qfd_hc_fp8_rows(const float *normed, const void *w_down, const void *w_up, const void *w_inject,
                               float *d, float *up, float *mixed, float *inj, int T, int NE, int HC, int hcl,
                               uint8_t *xq, uint8_t *xs, cudaStream_t s);
// Spark chunk rows: QF_PF_MAXT env (default 256, max 2048); buffers are sized once at first use.
static int qf_pf_maxt(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QF_PF_MAXT"); v = e ? atoi(e) : 256; if (v < 16) v = 16; if (v > 2048) v = 2048; }
    return v;
}
// Routed experts of the chunk on the block-scaled FP4 tensor core (default on
// sm_121; QF_PF_MMA=0 falls back to the software-dequant CSR kernels).
static int pf_mma(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QF_PF_MMA"); v = (e && e[0] == '0') ? 0 : (qf_fp4mma_available() ? 1 : 0); }
    return v;
}
struct QfPf {
    int cap;
    float *R, *mixed, *inj, *y2560, *qkv_raw, *z6144, *a48, *b48, *q6144, *k512, *v512, *attn_out, *out48;
    __nv_bfloat16 *gdn_out_bf;
    float *router, *shg, *shu, *shgi, *shd, *hidden, *wts, *pair_wt;
    int *sel, *exp_slot, *exp_ptr, *pair_tok, *counts;
    QfDecodeParams *params_dev, *params_host;
    float *conv_f32[NLAYER], *alog_f32[NLAYER], *dtb_f32[NLAYER];   // fp32 twins of the bf16 GDN weights (the _f32w contract)
    uint8_t *xq, *xs, *hq, *hs;                    // NVFP4 activations for the tensor-core expert path
    float *part;                                   // owned down partials [T*10][NEMBD]
    uint8_t *xq8, *xs8;                            // fp8 activations for the chunk GEMMs [T][10240], [T][160]
    float *hc_normed, *hc_d, *hc_up;               // hyper-connection rows scratch [T][10240], [T][320], [T][10240]
    float *idx640, *idx_blk_score; uint32_t *idx_mask;   // QSA indexer rows scratch [T][640], [T][nb_max], [T][mw]
    int *idx_list, *idx_nlist;                     // [T][513] selected complete blocks, [T] counts
    float *gdn2;                                   // two-level GDN scratch (qf_gdn2_scratch_floats)
};
static QfPf pf;
// One-time fp32 copies of a layer's conv1d / A_log / dt_bias so the chunked
// prefill kernels (prefill.cu, proven by test_prefill.cu) can run on the
// production checkpoint. 160 KB + 384 B per layer.
static int qf_pf_gdn_w(QfModel *m, int il, cudaStream_t s) {
    if (pf.conv_f32[il]) return 0;
    QfLayer *L = &m->layers[il];
    if (cudaMalloc((void **)&pf.conv_f32[il], (size_t)DINN * 4 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.alog_f32[il], (size_t)GDN_VH * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.dtb_f32[il], (size_t)GDN_VH * sizeof(float)) != cudaSuccess) return -1;
    k_bf16_to_f32<<<(DINN * 4 + 1023) / 1024, 1024, 0, s>>>(pf.conv_f32[il], (const __nv_bfloat16 *)L->conv1d, DINN * 4);
    k_bf16_to_f32<<<1, GDN_VH, 0, s>>>(pf.alog_f32[il], (const __nv_bfloat16 *)L->a, GDN_VH);
    k_bf16_to_f32<<<1, GDN_VH, 0, s>>>(pf.dtb_f32[il], (const __nv_bfloat16 *)L->dt_bias, GDN_VH);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
static int qf_pf_init(int T) {
    if (pf.cap >= T) return 0;
    if (pf.cap) return -1;
    const int hcd = HCC * NEMBD;
    #define PA(f, n) if (cudaMalloc((void **)&pf.f, (size_t)(n) * sizeof(float)) != cudaSuccess) return -1
    PA(R, (size_t)T * hcd); PA(mixed, (size_t)T * NEMBD); PA(inj, (size_t)T * HCC); PA(y2560, (size_t)T * NEMBD);
    PA(qkv_raw, (size_t)T * DINN); PA(z6144, (size_t)T * GDN_VDIM); PA(a48, (size_t)T * 128); PA(b48, (size_t)T * 128);   /* 128: MX-padded gate outputs (fused glue path) */
    PA(q6144, (size_t)T * NHEAD * QGATE); PA(k512, (size_t)T * NKV * HDIM); PA(v512, (size_t)T * NKV * HDIM);
    PA(attn_out, (size_t)T * NHEAD * HDIM); PA(out48, (size_t)T * GDN_VH * GDN_VD);
    PA(router, (size_t)T * NEXP); PA(shg, (size_t)T * NFF); PA(shu, (size_t)T * NFF); PA(shgi, (size_t)T); PA(shd, (size_t)T * NEMBD);
    PA(hidden, (size_t)T * NEXPUSED * NFF); PA(wts, (size_t)T * NEXPUSED); PA(pair_wt, (size_t)T * NEXPUSED);
    #undef PA
    if (cudaMalloc((void **)&pf.gdn_out_bf, (size_t)T * GDN_VH * GDN_VD * 2) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.sel, (size_t)T * NEXPUSED * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.exp_slot, (size_t)T * NEXPUSED * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.exp_ptr, ((size_t)T * NEXPUSED + 1) * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.pair_tok, (size_t)T * NEXPUSED * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.counts, 2 * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.xq, (size_t)T * (NEMBD / 2)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.xs, (size_t)T * (NEMBD / 16)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.hq, (size_t)T * NEXPUSED * (NFF / 2)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.hs, (size_t)T * NEXPUSED * (NFF / 16)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.part, (size_t)T * NEXPUSED * NEMBD * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.xq8, (size_t)T * hcd) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.xs8, (size_t)T * (hcd / 64)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.hc_normed, (size_t)T * hcd * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.hc_d, (size_t)T * (HCL > 384 ? HCL : 384) * sizeof(float)) != cudaSuccess) return -1;   // MX hc path writes d with a 384 stride
    if (cudaMalloc((void **)&pf.hc_up, (size_t)T * hcd * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.idx640, (size_t)T * 640 * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.idx_blk_score, (size_t)T * ctx.idx_nb_max * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.idx_mask, (size_t)T * ctx.idx_mw * sizeof(uint32_t)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.idx_list, (size_t)T * 513 * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.idx_nlist, (size_t)T * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.gdn2, qf_gdn2_scratch_floats(T) * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&pf.params_dev, (size_t)T * sizeof(QfDecodeParams)) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&pf.params_host, (size_t)T * sizeof(QfDecodeParams), cudaHostAllocDefault) != cudaSuccess) return -1;
    pf.cap = T;
    return 0;
}
// Section timers for the chunk tail (QF_PF_TIMING=1): PFT_MARKS stream events
// per layer, folded once per chunk into one stderr line. Off by default.
#define PFT_MARKS 10
static cudaEvent_t pft_ev[NLAYER * PFT_MARKS];
static int pft_n = 0, pft_on = -1;
static void pft_mark(cudaStream_t s) {
    if (pft_on < 0) pft_on = (getenv("QF_PF_TIMING") && getenv("QF_PF_TIMING")[0] == '1') ? 1 : 0;
    if (!pft_on || pft_n >= NLAYER * PFT_MARKS) return;
    if (!pft_ev[pft_n]) cudaEventCreate(&pft_ev[pft_n]);
    cudaEventRecord(pft_ev[pft_n++], s);
}
static cudaEvent_t pft_aev[3]; static double pft_attn_ms = 0.0, pft_idx_ms = 0.0;   // attention layers' core: indexer vs attention
static void pft_fold(int T, int nl) {
    if (!pft_on || pft_n < 2) { pft_n = 0; return; }
    static const char *nm[PFT_MARKS] = { "gap", "hc", "proj", "core", "oproj", "inj", "hc2+rtr/shx-proj", "topk", "experts", "shx-down+inj" };
    float sec[PFT_MARKS] = {0}, tot = 0.f;
    cudaEventSynchronize(pft_ev[pft_n - 1]);
    for (int i = 0; i + 1 < pft_n; i++) { float ms = 0.f; cudaEventElapsedTime(&ms, pft_ev[i], pft_ev[i + 1]); sec[(i + 1) % PFT_MARKS] += ms; tot += ms; }
    fprintf(stderr, "prefill chunk T=%d (%d layers): total %.1f ms (%.2f ms/token):", T, nl, tot, tot / T);
    for (int k = 1; k < PFT_MARKS; k++) fprintf(stderr, " %s %.1f", nm[k], sec[k]);
    fprintf(stderr, " gap %.1f (core: indexer %.1f, attention %.1f, gdn %.1f)\n", sec[0], pft_idx_ms, pft_attn_ms, sec[3] - pft_attn_ms - pft_idx_ms);
    pft_attn_ms = 0.0; pft_idx_ms = 0.0;
    pft_n = 0;
}
// QF_PF_LEGACY bisection bits for the chunk tail: 1 = per-token GDN recurrence,
// 2 = per-token attention, 4 = per-token inject / top-k / shared-expert add.
static int pf_legacy(void) { static int v = -1; if (v < 0) v = getenv("QF_PF_LEGACY") ? atoi(getenv("QF_PF_LEGACY")) : 0; return v; }
extern "C" int qf_prefill_chunk_tail(QfModel *m, const int *tokens, int T, long pos0, const float *hR) {
    if (T < 1 || T > qf_pf_maxt()) return -1;
    if (qf_pf_init(qf_pf_maxt())) { fprintf(stderr, "prefill chunk: alloc failed\n"); return -1; }
    if (qf_batch_init(QF_SPEC_MAXT)) return -1;            // qf_hc_T workspaces
    const int lb = qf_layer_begin();
    if (lb <= 1) { fprintf(stderr, "prefill chunk: tail only (lb=%d)\n", lb); return -1; }
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD;
    if (cudaMemcpyAsync(pf.R, hR, (size_t)T * hcd * sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) return -1;
    static int nanchk = -1; if (nanchk < 0) nanchk = getenv("QF_PF_NANCHECK") ? 1 : 0;
    if (nanchk) {   // residual entering the Spark chunk (= the AMD prefix output): finite? magnitude?
        long bad = 0; double mx = 0.0;
        for (size_t i = 0; i < (size_t)T * hcd; i++) { const float v = hR[i]; if (!(v == v) || v > 3e38f || v < -3e38f) bad++; else if (fabsf(v) > mx) mx = fabsf(v); }
        fprintf(stderr, "nancheck chunk pos0=%ld T=%d: AMD->Spark residual nonfinite=%ld max|R|=%.3g\n", pos0, T, bad, mx);
    }
    for (int t = 0; t < T; t++) {
        pf.params_host[t].token = tokens[t]; pf.params_host[t].pos = (int)(pos0 + t);
        pf.params_host[t].seq = ++ctx.route_seq; pf.params_host[t].flags = 0;
    }
    if (cudaMemcpyAsync(pf.params_dev, pf.params_host, (size_t)T * sizeof(QfDecodeParams), cudaMemcpyHostToDevice, s) != cudaSuccess) return -1;
    cudaMemsetAsync(ctx.route_err_dev, 0, sizeof(int), s);
    // Eager pass: the grouped-projection pointer slots are staged through a
    // pinned mirror with ASYNC copies, so the pool is rewound ONCE per chunk
    // (the previous chunk ended in a stream sync) in its eager region. Rewinding
    // per layer let the run-ahead host re-stage slots the device had not copied
    // yet: recurrent and attention layers put different pointers in the same
    // slot -> garbage prefill, masked only by launch-queue throttling.
    qfd_gemv_group_T_rewind_eager();
    const float *pend_y = NULL, *pend_inj = NULL;      // lazy stream inject (R += inj (x) y) applied by the next norm kernel
    int ab_stride = DTRANK;                             // row stride of the GDN gate outputs a48/b48 (128 when MX-padded)
    int gdn_post = 0;                                   // fused GDN post path active for this layer
    for (int il = lb; il < NLAYER; il++) {
        QfLayer *L = &m->layers[il];
        const int is_recr = ((il + 1) % 4) != 0;
        pft_mark(s);                                    // 0 layer start
        // Hyper-connection over the whole chunk: norm (one launch over HCC x T),
        // down/up on the fp8 tensor-core GEMM, rows mix + inject. Same per-row
        // arithmetic as qf_hc_T, minus the 16-row sub-tiles.
        const int hcf = pf_hcf() && !(pf_legacy() & 4);
        int mixed_mx = 0;                               // MX(mixed) already in the activation scratch (fused path)
        if (hcf) {
            // fused: [pending FFN inject of the previous layer] + norm + MX + gate -> down -> silu -> up -> mix + MX
            if (qfd_hc_fused_rows(pf.R, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, pf.hc_d, pf.hc_up, pf.hc_normed, pf.mixed, pf.inj,
                                  pend_y, pend_inj, NULL, NULL, T, s)) { fprintf(stderr, "prefill chunk: fused hc failed\n"); return -1; }
            pend_y = NULL; pend_inj = NULL; mixed_mx = 1;
        } else {
        if (pend_y) { k_stream_inject_M<<<dim3((hcd + 1023) / 1024, T), 1024, 0, s>>>(pf.R, pend_y, pend_inj); pend_y = NULL; pend_inj = NULL; }
        k_rmsnorm_M<<<dim3(HCC, T), 256, 0, s>>>(pf.hc_normed, pf.R, (const __nv_bfloat16 *)L->hc_attn_norm, hcd, 1e-6f, NEMBD);
        if (qfd_hc_fp8_rows(pf.hc_normed, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, pf.hc_d, pf.hc_up, pf.mixed, pf.inj,
                            T, NEMBD, HCC, HCL, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: hc failed\n"); return -1; }
        }
        pft_mark(s);                                    // 1 hc
        if (is_recr) {
            {   // GDN in-projections: ONE fp8 tensor-core GEMM over the chunk (weights read once per 64 tokens)
                QfDenseProj p[4] = {{L->qkv, pf.qkv_raw, DINN}, {L->zgate, pf.z6144, GDN_VDIM}, {L->beta, pf.a48, DTRANK}, {L->alpha, pf.b48, DTRANK}};
                ab_stride = DTRANK;                     // gates stay on the existing path (operator: not in this pass)
                if (qfd_gemm_fp8_rows(p, 4, pf.mixed, NEMBD, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: gdn proj failed\n"); return -1; }
            }
            pft_mark(s);                                // 2 in-proj
            if (pf_legacy() & 1) {                      // bisection: the per-token recurrence
                for (int t = 0; t < T; t++) {
                    k_conv_step<<<(DINN + 1023) / 1024, 1024, 0, s>>>(pf.qkv_raw + (size_t)t * DINN, ctx.convring[il], (const __nv_bfloat16 *)L->conv1d);
                    k_gdn_decode<<<GDN_VH, GDN_VD, 0, s>>>(pf.qkv_raw + (size_t)t * DINN, pf.a48 + (size_t)t * DTRANK, pf.b48 + (size_t)t * DTRANK,
                        (const __nv_bfloat16 *)L->a, (const __nv_bfloat16 *)L->dt_bias, ctx.gdnS[il], pf.out48 + (size_t)t * GDN_VH * GDN_VD);
                    k_rmsnorm_gated<<<GDN_VH, GDN_VD, 0, s>>>(pf.gdn_out_bf + (size_t)t * GDN_VH * GDN_VD, pf.out48 + (size_t)t * GDN_VH * GDN_VD,
                        (const __nv_bfloat16 *)L->gdn_norm, pf.z6144 + (size_t)t * GDN_VDIM, 1e-6f);
                    k_bf16_to_f32<<<(GDN_VDIM + 1023) / 1024, 1024, 0, s>>>(pf.q6144 + (size_t)t * GDN_VDIM, pf.gdn_out_bf + (size_t)t * GDN_VH * GDN_VD, GDN_VDIM);
                }
            } else {
            // Chunked GDN: prefill.cu's own conv + parallel delta-rule kernels
            // (test_prefill.cu proves them against T decode steps). The conv ring
            // and the state S advance over all T positions in ONE launch each
            // instead of 4*T launches; the gated norm runs M-row.
            if (qf_pf_gdn_w(m, il, s)) { fprintf(stderr, "prefill chunk: gdn weight copy failed\n"); return -1; }
            for (int t0 = 0; t0 < T; t0 += 512) {           // conv: the prefill.cu kernel takes <= 512 positions; ring carries across pieces
                const int st = (T - t0) < 512 ? (T - t0) : 512;
                if (qf_conv_prefill_f32w(pf.qkv_raw + (size_t)t0 * DINN, ctx.convring[il], pf.conv_f32[il], st, s)) { fprintf(stderr, "prefill chunk: conv failed\n"); return -1; }
            }
            gdn_post = gdn_fla_on() && hcf;                // fused post: FLA bf16 out -> gated norm -> MX -> out-proj (no unpack/bf16/fp32 passes)
            if (gdn_fla_on()) {                             // FLA chunked gated delta rule (AOT Triton cubins, cuda/qf_gdn_fla.cu)
                if (qf_gdn_fla_prefill(pf.qkv_raw, pf.a48, pf.b48, pf.alog_f32[il], pf.dtb_f32[il], ctx.gdnS[il], pf.out48, T, ab_stride, !gdn_post, s)) { fprintf(stderr, "prefill chunk: gdn fla failed\n"); return -1; }
            } else
            if (pf_gdn2()) {                                // two-level chunked delta rule (parallel intra-chunk solves + per-head scan)
                if (qf_gdn2_prefill(pf.qkv_raw, pf.a48, pf.b48, pf.alog_f32[il], pf.dtb_f32[il], ctx.gdnS[il], pf.out48, T, pf.gdn2, pf.cap, s)) { fprintf(stderr, "prefill chunk: gdn2 failed\n"); return -1; }
            } else
            for (int t0 = 0; t0 < T; t0 += 512) {
                const int st = (T - t0) < 512 ? (T - t0) : 512;
                if (qf_gdn_prefill_f32w(pf.qkv_raw + (size_t)t0 * DINN, pf.a48 + (size_t)t0 * DTRANK, pf.b48 + (size_t)t0 * DTRANK, pf.alog_f32[il], pf.dtb_f32[il],
                                        ctx.gdnS[il], pf.out48 + (size_t)t0 * GDN_VH * GDN_VD, st, s)) { fprintf(stderr, "prefill chunk: gdn failed\n"); return -1; }
            }
            if (gdn_post) {
                if (qfd_gdn_post_quant(qf_gdn_fla_out(), pf.z6144, L->gdn_norm, T, s)) { fprintf(stderr, "prefill chunk: gdn post failed\n"); return -1; }
            } else {
            k_rmsnorm_gated_M<<<dim3(GDN_VH, T), GDN_VD, 0, s>>>(pf.gdn_out_bf, pf.out48, (const __nv_bfloat16 *)L->gdn_norm, pf.z6144, 1e-6f);
            k_bf16_to_f32_M<<<dim3((GDN_VDIM + 1023) / 1024, T), 1024, 0, s>>>(pf.q6144, pf.gdn_out_bf, GDN_VDIM);
            }
            }
            pft_mark(s);                                // 3 gdn core
            {   QfDenseProj p[1] = {{L->gdn_out, pf.y2560, NEMBD}};
                if (gdn_post) { if (qfd_gemm_mx_prequant(p, 1, GDN_VDIM, T, 0, s)) { fprintf(stderr, "prefill chunk: gdn out prequant failed\n"); return -1; } }
                else if (qfd_gemm_fp8_rows(p, 1, pf.q6144, GDN_VDIM, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: gdn out failed\n"); return -1; }
            }
            pft_mark(s);                                // 4 out-proj
        } else {
            {   QfDenseProj p[4] = {{L->wq, pf.q6144, NHEAD * QGATE}, {L->wk, pf.k512, NKV * HDIM}, {L->wv, pf.v512, NKV * HDIM}, {L->idx_qk, pf.idx640, 640}};
                if (!(mixed_mx && qfd_gemm_mx_prequant(p, 4, NEMBD, T, 0, s) == 0))
                if (qfd_gemm_fp8_rows(p, 4, pf.mixed, NEMBD, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: qkv failed\n"); return -1; }
            }
            pft_mark(s);                                // 2 qkv proj
            if (pft_on) { if (!pft_aev[0]) { cudaEventCreate(&pft_aev[0]); cudaEventCreate(&pft_aev[1]); cudaEventCreate(&pft_aev[2]); } cudaEventRecord(pft_aev[0], s); }
            // Chunk-causal attention (3 launches, not 3*T): all T keys/values are
            // written, then query t attends [0, pos0+t] of the one cache. Dense, as
            // the per-token form was (it passed no QSA mask either).
            if (pf_legacy() & 2) {                      // bisection: per-token attention
                for (int t = 0; t < T; t++)
                    qf_attn_qsa_layer(pf.q6144 + (size_t)t * NHEAD * QGATE, pf.k512 + (size_t)t * NKV * HDIM, pf.v512 + (size_t)t * NKV * HDIM,
                                      ctx.kc[il], ctx.vc[il], L->q_norm, L->k_norm, ctx.inv_freq, pf.params_dev + t, NULL,
                                      pf.attn_out + (size_t)t * NHEAD * HDIM, s);
            } else
            if (qf_qsa_index_rows(pf.idx640, T, pf.params_dev, L->idx_qnorm, L->idx_knorm, ctx.inv_freq_idx, ctx.idx_pool_sum[il],
                                  ctx.idx_pool_key[il], ctx.idx_pool_cnt[il], pf.idx_blk_score, ctx.idx_nb_max, pf.idx_mask, ctx.idx_mw,
                                  pf.idx_list, pf.idx_nlist, pos0, s)) { fprintf(stderr, "prefill chunk: indexer failed\n"); return -1; }
            if (pft_on) cudaEventRecord(pft_aev[2], s);
            if (qf_attn_qsa_chunk(pf.q6144, pf.k512, pf.v512, ctx.kc[il], ctx.vc[il], L->q_norm, L->k_norm, ctx.inv_freq,
                                  pf.params_dev, qsa_index_on() ? pf.idx_mask : NULL, ctx.idx_mw, qsa_index_on() ? pf.idx_list : NULL, pf.idx_nlist, 513,
                                  pf.attn_out, T, (long)pos0, s)) { fprintf(stderr, "prefill chunk: attention failed\n"); return -1; }
            if (pft_on) { cudaEventRecord(pft_aev[1], s); cudaEventSynchronize(pft_aev[1]); float ms = 0.f;
                          cudaEventElapsedTime(&ms, pft_aev[0], pft_aev[2]); pft_idx_ms += ms; cudaEventElapsedTime(&ms, pft_aev[2], pft_aev[1]); pft_attn_ms += ms; }
            pft_mark(s);                                // 3 attention core
            {   QfDenseProj p[1] = {{L->wo, pf.y2560, NEMBD}};
                if (qfd_gemm_fp8_rows(p, 1, pf.attn_out, NHEAD * HDIM, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: wo failed\n"); return -1; }
            }
            pft_mark(s);                                // 4 wo
        }
        if (pf_legacy() & 4) {                          // bisection: per-token inject / top-k / shexp add
            for (int t = 0; t < T; t++)
                k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(pf.R + (size_t)t * hcd, pf.y2560 + (size_t)t * NEMBD, pf.inj + (size_t)t * HCC);
        } else
        if (hcf) {
            pft_mark(s);                                // 5 inject (folded into the FFN-side fused norm)
            if (qfd_hc_fused_rows(pf.R, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, pf.hc_d, pf.hc_up, pf.hc_normed, pf.mixed, pf.inj,
                                  pf.y2560, pf.inj, L->shexp_gate_inp, pf.shgi, T, s)) { fprintf(stderr, "prefill chunk: fused hc2 failed\n"); return -1; }
            QfDenseProj p[3] = {{L->router, pf.router, NEXP}, {L->shexp_gate, pf.shg, NFF}, {L->shexp_up, pf.shu, NFF}};
            if (qfd_gemm_mx_prequant(p, 3, NEMBD, T, 0, s)) { fprintf(stderr, "prefill chunk: router/shexp prequant failed\n"); return -1; }
        } else {
        k_stream_inject_M<<<dim3((hcd + 1023) / 1024, T), 1024, 0, s>>>(pf.R, pf.y2560, pf.inj);
        pft_mark(s);                                    // 5 inject
        k_rmsnorm_M<<<dim3(HCC, T), 256, 0, s>>>(pf.hc_normed, pf.R, (const __nv_bfloat16 *)L->hc_ffn_norm, hcd, 1e-6f, NEMBD);
        if (qfd_hc_fp8_rows(pf.hc_normed, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, pf.hc_d, pf.hc_up, pf.mixed, pf.inj,
                            T, NEMBD, HCC, HCL, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: hc2 failed\n"); return -1; }
        {   QfDenseProj p[4] = {{L->router, pf.router, NEXP}, {L->shexp_gate, pf.shg, NFF}, {L->shexp_up, pf.shu, NFF}, {L->shexp_gate_inp, pf.shgi, 1}};
            if (qfd_gemm_fp8_rows(p, 4, pf.mixed, NEMBD, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: router/shexp failed\n"); return -1; }
        }
        }
        pft_mark(s);                                    // 6 hc2 + router/shexp proj
        if (pf_legacy() & 4) {
            for (int t = 0; t < T; t++)
                k_router_topk<<<1, 128, 0, s>>>(pf.router + (size_t)t * NEXP, pf.sel + (size_t)t * NEXPUSED, pf.wts + (size_t)t * NEXPUSED);
        } else
        k_router_topk_M<<<T, 128, 0, s>>>(pf.router, pf.sel, pf.wts);
        pft_mark(s);                                    // 7 topk
        cudaMemsetAsync(pf.y2560, 0, (size_t)T * NEMBD * sizeof(float), s);
        if (pf_mma())
            qf_prefill_experts_mma(L->exp_gate, L->exp_scale, L->exp_up, L->exp_scale_up, L->exp_down, L->exp_scale_down,
                                   L->s2_gate_dev, L->s2_up_dev, L->s2_down_dev, pf.sel, pf.wts, T, pf.mixed, pf.hidden, pf.y2560,
                                   pf.exp_slot, pf.exp_ptr, pf.pair_tok, pf.pair_wt, pf.counts, pf.xq, pf.xs, pf.hq, pf.hs, pf.part, s);
        else
        qf_prefill_experts_csr(L->exp_gate, L->exp_scale, L->exp_up, L->exp_scale_up, L->exp_down, L->exp_scale_down,
                               L->s2_gate_dev, L->s2_up_dev, L->s2_down_dev, pf.sel, pf.wts, T, pf.mixed, pf.hidden, pf.y2560,
                               pf.exp_slot, pf.exp_ptr, pf.pair_tok, pf.pair_wt, pf.counts, s);
        pft_mark(s);                                    // 8 routed experts (CSR)
        if (hcf) {
            QfDenseProj p[1] = {{L->shexp_down, pf.shd, NEMBD}};
            if (qfd_shexp_act_quant(pf.shg, pf.shu, T, s) || qfd_gemm_mx_prequant(p, 1, NFF, T, 0, s)) { fprintf(stderr, "prefill chunk: fused shexp down failed\n"); return -1; }
        } else {
        k_silu_mul<<<(T * NFF + 255) / 256, 256, 0, s>>>(pf.shg, pf.shu, T * NFF);
        {   QfDenseProj p[1] = {{L->shexp_down, pf.shd, NEMBD}};
            if (qfd_gemm_fp8_rows(p, 1, pf.shg, NFF, T, pf.xq8, pf.xs8, s)) { fprintf(stderr, "prefill chunk: shexp down failed\n"); return -1; }
        }
        }
        if (pf_legacy() & 4) {
            for (int t = 0; t < T; t++) {
                k_shexp_add<<<(NEMBD + 255) / 256, 256, 0, s>>>(pf.y2560 + (size_t)t * NEMBD, pf.shd + (size_t)t * NEMBD, pf.shgi + t);
                k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(pf.R + (size_t)t * hcd, pf.y2560 + (size_t)t * NEMBD, pf.inj + (size_t)t * HCC);
            }
        } else {
        k_shexp_add_M<<<dim3((NEMBD + 255) / 256, T), 256, 0, s>>>(pf.y2560, pf.shd, pf.shgi);
        if (hcf) { pend_y = pf.y2560; pend_inj = pf.inj; }   // applied by the next layer's fused norm (or below after the last layer)
        else k_stream_inject_M<<<dim3((hcd + 1023) / 1024, T), 1024, 0, s>>>(pf.R, pf.y2560, pf.inj);
        }
        pft_mark(s);                                    // 9 shexp down/add + inject
    }
    if (pend_y) k_stream_inject_M<<<dim3((hcd + 1023) / 1024, T), 1024, 0, s>>>(pf.R, pend_y, pend_inj);   // last layer's inject
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    if (nanchk) {   // residual leaving the Spark chunk
        static float *hchk = NULL; if (!hchk) cudaHostAlloc((void **)&hchk, (size_t)qf_pf_maxt() * hcd * sizeof(float), cudaHostAllocDefault);
        cudaMemcpy(hchk, pf.R, (size_t)T * hcd * sizeof(float), cudaMemcpyDeviceToHost);
        long bad = 0; double mx = 0.0;
        for (size_t i = 0; i < (size_t)T * hcd; i++) { const float v = hchk[i]; if (!(v == v) || v > 3e38f || v < -3e38f) bad++; else if (fabsf(v) > mx) mx = fabsf(v); }
        fprintf(stderr, "nancheck chunk pos0=%ld T=%d: Spark tail output nonfinite=%ld max|R|=%.3g\n", pos0, T, bad, mx);
    }
    pft_fold(T, NLAYER - lb);
    return 0;
}

// Prompt priming for the ALTERNATING MAP (AMD prefix + head, Spark tail). The
// Spark-only prime above runs each prompt through the single-request path and
// snapshots ctx into the row; on a split box that path needs the residual of
// the layers it does not own. So, per row and per prompt token: AMD runs its
// prefix for THAT ROW ONLY (QFW_PREFIX_M with row base r, advancing row r's
// GDN/conv/KV/PLE state on the AMD box), the residual is pushed here, and the
// single-request tail runs on ctx; then ctx is snapshotted into row r for the
// layers this box owns. Rows are primed independently because prompts differ
// in length. Both boxes end with every row at pos = len-1, ready for the
// batched decode loop (which processes the last prompt token first).
extern "C" int qf_region_amd_prefix_rows(int r0, int nr, const int *tokens, const long *posM, float *rout, int rsize);
extern "C" int qf_region_amd_reset(void);
extern "C" int qf_region_amd_prefix_submit(int slot, int r0, int nr, const int *tokens, const long *posM, float *rout, int rsize);
extern "C" int qf_region_amd_drain_one(int *op_out, int *slot_out);
extern "C" int qf_region_amd_headprefix(const float *rin, int rsize, long pos_next, int *tok_out, float *rout);
#define QF_REGION_PEND_MAX 16         // prefix RPCs per chunk (ring depth 8 for the per-token form, 16 for the chunk RPC)
extern "C" int qf_region_amd_prefix_chunk_submit(const int *tokens, int T, long pos0);
extern "C" int qf_region_amd_prefix_chunk_submit_to(const int *tokens, int T, long pos0, float *dst, int rsize);
extern "C" int qf_region_amd_prefix_chunk_wait_done(void);
extern "C" int qf_region_amd_prefix_chunk_wait(int T, float *rout, int rsize);
int qf_decode_body_T(QfModel *m, const int *tokens, int T, long pos0);   // defined below (spec/prefill body)
extern "C" int qf_decode_batch_prime_region(QfModel *m, const int *const *ids, const int *len,
                                            int M, int *tok_out, long *pos_out, int (*hist_out)[3]) {
        if (M < 1 || M > QF_M8_MAXROWS) return -1;        // up to two 8-row slots
    if (!qf_region_amd_enabled()) return -1;
    const int lb = qf_layer_begin();
    if (lb <= 0) { fprintf(stderr, "prime_region: QF_LAYER_BEGIN must be the AMD prefix depth\n"); return -1; }
    if (qf_reqbatch_state_init(M)) return -1;
    if (qf_region_amd_reset() != 0) { fprintf(stderr, "prime_region: AMD reset failed\n"); return -1; }
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD;
    static float *hR = NULL;
    if (!hR && cudaHostAlloc((void **)&hR, (size_t)hcd * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    const size_t span = (size_t)qf_maxpos() * KVDIM;
    qf_set_skip_head(1);                              // the head is AMD's in this map
    // M=1 prompt pass, PIPELINED across the two boxes: the AMD prefix of token
    // i+1 is submitted before the Spark tail of token i runs, so the boxes work
    // concurrently and the per-token cost is max(prefix, tail), not the sum.
    // (The FIFO ring in qf_region_amd.cpp delivers replies in order.)
    // Chunked, pipelined prompt pass (QF_PRIME_T tokens per chunk, default 8 =
    // the wire ring depth): the AMD prefix RPCs of chunk c+1 are in flight while
    // Spark runs chunk c through the T-token trunk body (dense projections and
    // hyper-connections read their weights ONCE per chunk). QF_PRIME_T=1 is the
    // token-serial pipelined form; QF_PRIME_SERIAL=1 the plain serial form.
    static float *hRc[3] = {NULL, NULL, NULL};   // chunk staging: the chunk Spark is on + two in flight
    static int   tkc[2][QF_REGION_PEND_MAX]; static long pcc[2][QF_REGION_PEND_MAX];
    if (M == 1 && !getenv("QF_PRIME_SERIAL")) {
        // QF_PRIME_CHUNK=1 (default): ONE prefix RPC per chunk (QFW_PREFIX_CHUNK, the AMD
        // chunk executor runs the T positions batched); QF_PRIME_CHUNK=0: T per-token RPCs.
        static int chunk_rpc = -1;
        if (chunk_rpc < 0) { const char *e = getenv("QF_PRIME_CHUNK"); chunk_rpc = (e && e[0] == '0') ? 0 : 1; }
        int TC = getenv("QF_PRIME_T") ? atoi(getenv("QF_PRIME_T")) : (chunk_rpc ? 16 : 8);
        if (TC < 1) TC = 1;
        if (!chunk_rpc && TC > 8) TC = 8;               // ring depth for the per-token form
        if (TC > qf_pf_maxt()) TC = qf_pf_maxt();       // Spark chunk (QF_PF_MAXT); AMD serves it in QF_AMD_CHUNK requests
        int TA = getenv("QF_AMD_CHUNK") ? atoi(getenv("QF_AMD_CHUNK")) : 256;   // AMD chunk request size (server QF_CHUNK_ROWS)
        if (TA < 1) TA = 1; if (TA > QF_PF_MAXT) TA = QF_PF_MAXT;
        for (int b = 0; b < 3; b++)
            if (!hRc[b] && cudaHostAlloc((void **)&hRc[b], (size_t)qf_pf_maxt() * hcd * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
        const int L = len[0];
        if (L < 1) { qf_set_skip_head(0); return -1; }
        qf_session_reset(m);
        qf_hist_reset();
        const int Lp = L - 1;                          // tokens 0..L-2 (the last one starts decode)
        // First-chunk ramp (QF_PRIME_T0, default = TC): the AMD pipeline must produce the
        // whole first logical chunk before Spark can start (1.33 s for 2048 rows), so the
        // first chunk is small and each following chunk doubles up to TC; from the second
        // chunk on the AMD rows for chunk c+1 are produced while Spark runs chunk c.
        int T0 = getenv("QF_PRIME_T0") ? atoi(getenv("QF_PRIME_T0")) : TC;   // 512 measured worse at the current AMD/Spark balance (AMD falls behind after small chunks)
        if (T0 < 1) T0 = 1; if (T0 > TC) T0 = TC;
        // Chunk plan (first-chunk ramp above) with LOOK chunks of AMD requests in flight ahead of
        // the Spark chunk pass. The AMD pipeline's THROUGHPUT hides behind Spark, but each chunk's
        // LAST 256-row request still traverses the three remaining MI50 stages (~130 ms each) plus
        // the wire after stage 0 finishes it: ~0.4 s exposed per chunk with one chunk in flight
        // (measured 308-386 ms). Two chunks in flight (three pinned staging buffers) hide that tail.
        const int LOOK = chunk_rpc ? 2 : 1, NBUF = LOOK + 1;
        static int cs_n = 0, cs_cap = 0; static int *cs_start = NULL, *cs_len = NULL;
        cs_n = 0;
        { int i0 = 0, cur = T0;
          while (i0 < Lp) {
              if (cs_n >= cs_cap) { cs_cap = cs_cap ? cs_cap * 2 : 1024; cs_start = (int *)realloc(cs_start, (size_t)cs_cap * sizeof(int)); cs_len = (int *)realloc(cs_len, (size_t)cs_cap * sizeof(int)); }
              const int t = (Lp - i0) < cur ? (Lp - i0) : cur;
              cs_start[cs_n] = i0; cs_len[cs_n] = t; cs_n++; i0 += t;
              int nx = cur * 2 < TC ? cur * 2 : TC; if (nx < T0) nx = T0; cur = nx; }
          // Drain: Spark's LAST chunk runs after AMD's last row, fully exposed (~0.95 s for 1893 rows).
          // Split the final logical chunk into halving pieces (QF_PRIME_TD, default 512, floor) so the
          // exposed tail is one small pass; the production 2048 chunks in the middle are untouched.
          int TD = getenv("QF_PRIME_TD") ? atoi(getenv("QF_PRIME_TD")) : 0;   // off: measured worse while Spark is the steady-state bound (5.64 vs 5.3 s)
          if (TD > 0 && cs_n >= 1 && cs_len[cs_n - 1] > TD) {
              int rem = cs_len[cs_n - 1], base = cs_start[cs_n - 1]; cs_n--;
              while (rem > TD) {
                  int pc = 1; while (pc * 2 <= rem / 2) pc *= 2;       // largest power of two <= rem/2
                  if (pc < TD) pc = TD; if (pc > rem) pc = rem;
                  if (cs_n >= cs_cap) { cs_cap *= 2; cs_start = (int *)realloc(cs_start, (size_t)cs_cap * sizeof(int)); cs_len = (int *)realloc(cs_len, (size_t)cs_cap * sizeof(int)); }
                  cs_start[cs_n] = base; cs_len[cs_n] = pc; cs_n++; base += pc; rem -= pc;
              }
              if (rem > 0) { if (cs_n >= cs_cap) { cs_cap *= 2; cs_start = (int *)realloc(cs_start, (size_t)cs_cap * sizeof(int)); cs_len = (int *)realloc(cs_len, (size_t)cs_cap * sizeof(int)); }
                             cs_start[cs_n] = base; cs_len[cs_n] = rem; cs_n++; }
          } }
        auto submit_chunk = [&](int c) -> int {
            if (c >= cs_n) return 0;
            const int base = cs_start[c], T = cs_len[c]; float *dst = hRc[c % NBUF]; const int bb = c % 2;
            if (chunk_rpc) { for (int k = 0; k < T; k += TA) { const int tk = (T - k) < TA ? (T - k) : TA;
                if (qf_region_amd_prefix_chunk_submit_to(ids[0] + base + k, tk, base + k, dst + (size_t)k * hcd, hcd) != 0) return -1; } }
            else for (int t = 0; t < T; t++) {
                tkc[bb][t] = ids[0][base + t]; pcc[bb][t] = base + t;
                if (qf_region_amd_prefix_submit(t, 0, 1, &tkc[bb][t], &pcc[bb][t], dst + (size_t)t * hcd, hcd) != 0) return -1; }
            return 0; };
        for (int c = 0; c < LOOK; c++) if (submit_chunk(c) != 0) { qf_set_skip_head(0); return -1; }
        static int ptim = -1; if (ptim < 0) ptim = getenv("QF_M8_TIMING") ? 1 : 0;
        for (int c = 0; c < cs_n; c++) {
            const int i = cs_start[c], T = cs_len[c]; float *hb = hRc[c % NBUF];
            struct timespec tw0, tw1, tw2; if (ptim) clock_gettime(CLOCK_MONOTONIC, &tw0);
            if (chunk_rpc) { for (int k = 0; k < T; k += TA) {
                if (qf_region_amd_prefix_chunk_wait_done() != 0) { fprintf(stderr, "prime_region chunk@%d: AMD prefix failed\n", i + k); qf_set_skip_head(0); return -1; } } }
            else for (int t = 0; t < T; t++) {          // collect chunk c (FIFO ring)
                int op = 0, sl = 0;
                if (qf_region_amd_drain_one(&op, &sl) != 0) { fprintf(stderr, "prime_region tok%d: AMD prefix failed\n", i + t); qf_set_skip_head(0); return -1; }
            }
            if (ptim) clock_gettime(CLOCK_MONOTONIC, &tw1);
            if (submit_chunk(c + LOOK) != 0) { qf_set_skip_head(0); return -1; }   // keep LOOK chunks in flight
            if (T == 1) {
                if (qf_push_residual(m, hb) != 0) { qf_set_skip_head(0); return -1; }
                if (qf_decode_step_tail(m, ids[0][i], i) != 0) { fprintf(stderr, "prime_region tok%d failed\n", i); qf_set_skip_head(0); return -1; }
            } else if (T <= QF_SPEC_MAXT) {
                if (qf_push_residual_M(hb, T) != 0) { qf_set_skip_head(0); return -1; }
                if (qf_decode_body_T(m, ids[0] + i, T, i) != 0) { fprintf(stderr, "prime_region chunk@%d (T=%d) failed\n", i, T); qf_set_skip_head(0); return -1; }
            } else {
                if (qf_prefill_chunk_tail(m, ids[0] + i, T, i, hb) != 0) { fprintf(stderr, "prime_region large chunk@%d (T=%d) failed\n", i, T); qf_set_skip_head(0); return -1; }
            }
            if (ptim) { clock_gettime(CLOCK_MONOTONIC, &tw2);
                        fprintf(stderr, "prime chunk@%d T=%d: waited AMD %.0f ms, Spark %.0f ms\n", i, T,
                                (tw1.tv_sec - tw0.tv_sec) * 1e3 + (tw1.tv_nsec - tw0.tv_nsec) / 1e6, (tw2.tv_sec - tw1.tv_sec) * 1e3 + (tw2.tv_nsec - tw1.tv_nsec) / 1e6); }
        }
        qf_set_skip_head(0);
        tok_out[0] = ids[0][L - 1];
        pos_out[0] = L - 1;
        hist_out[0][0] = g_hist[0]; hist_out[0][1] = g_hist[1]; hist_out[0][2] = g_hist[2];
        return 0;
    }
    for (int r = 0; r < M; r++) {
        int L = len[r];
        if (L < 1) { qf_set_skip_head(0); return -1; }
        qf_session_reset(m);
        qf_hist_reset();
        for (int i = 0; i < L - 1; i++) {
            const int tk = ids[r][i]; const long p = i;
            if (qf_region_amd_prefix_rows(r, 1, &tk, &p, hR, hcd) != 0) {
                fprintf(stderr, "prime_region r%d tok%d: AMD prefix failed\n", r, i); qf_set_skip_head(0); return -1; }
            if (qf_push_residual(m, hR) != 0) { qf_set_skip_head(0); return -1; }
            if (qf_decode_step_tail(m, tk, i) != 0) { fprintf(stderr, "prime_region r%d tok%d failed\n", r, i); qf_set_skip_head(0); return -1; }
        }
        for (int il = lb; il < NLAYER; il++) {           // only the layers this box owns
            if (((il + 1) % 4) != 0) {
                cudaMemcpyAsync(g_gdnS_M[il] + (size_t)r * sS, ctx.gdnS[il], sS * sizeof(float), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_convring_M[il] + (size_t)r * 3 * DINN, ctx.convring[il], (size_t)3 * DINN * sizeof(float), cudaMemcpyDeviceToDevice, s);
            } else if (L - 1 > 0) {
                size_t nb = (size_t)(L - 1) * KVDIM * 2;
                cudaMemcpyAsync(g_kc_M[il] + (size_t)r * span, ctx.kc[il], nb, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_vc_M[il] + (size_t)r * span, ctx.vc[il], nb, cudaMemcpyDeviceToDevice, s);
            }
        }
        if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
        tok_out[r] = ids[r][L - 1];
        pos_out[r] = L - 1;
        hist_out[r][0] = g_hist[0]; hist_out[r][1] = g_hist[1]; hist_out[r][2] = g_hist[2];
    }
    return 0;
}


// ===========================================================================
// WAVE=4 A/B scheduler. Global M=8 split A={0..3}, B={4..7}. Per layer, Spark
// computes wave A's frontend (HC+attention+dense+inject+FFN-HC+router), submits
// A's routed experts to AMD, then computes wave B's frontend WHILE AMD works on
// A (AMD(A) || Spark(B)); submits B; computes both shared experts while AMD
// works on B; then collects A and B (deferred completion) and joins. All state
// is row-indexed by r0 so the two waves keep independent GDN/conv/QSA/position/
// PLE/router state. M=8 amortization is preserved inside each dense/attn kernel;
// the M=4 split exists only to expose AMD||Spark overlap.
// ===========================================================================
static void qf_frontend_wave(QfModel *m, int il, int r0, int nr, cudaStream_t s) {
    QfLayer *L = &m->layers[il];
    const int hcd = HCC * NEMBD;
    const int is_recr = ((il + 1) % 4) != 0;
    if (il == 1)
        for (int r = r0; r < r0 + nr; r++) qf_ple_apply_staged_slot(m, bt.R + (size_t)r * hcd, r, s);
    qf_hc_T(bt.R + (size_t)r0 * hcd, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up,
            L->hc_attn_inject, bt.mixed + (size_t)r0 * NEMBD, bt.inj + (size_t)r0 * HCC, nr, s);
    if (is_recr) {
        { QfDenseProj p[4] = {{L->qkv, bt.qkv_raw + (size_t)r0 * DINN, DINN},
                              {L->zgate, bt.z6144 + (size_t)r0 * GDN_VDIM, GDN_VDIM},
                              {L->beta, bt.a48 + (size_t)r0 * DTRANK, DTRANK},
                              {L->alpha, bt.b48 + (size_t)r0 * DTRANK, DTRANK}};
          qfd_gemv_group_T(p, 4, bt.mixed + (size_t)r0 * NEMBD, NEMBD, nr, s); }
        const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
        k_conv_step_M<<<dim3((DINN + 1023) / 1024, nr), 1024, 0, s>>>(
            bt.qkv_raw + (size_t)r0 * DINN, g_convring_M[il] + (size_t)r0 * 3 * DINN, (const __nv_bfloat16 *)L->conv1d);
        k_gdn_decode_M<<<dim3(GDN_VH, nr), GDN_VD, 0, s>>>(
            bt.qkv_raw + (size_t)r0 * DINN, bt.a48 + (size_t)r0 * DTRANK, bt.b48 + (size_t)r0 * DTRANK,
            (const __nv_bfloat16 *)L->a, (const __nv_bfloat16 *)L->dt_bias,
            g_gdnS_M[il] + (size_t)r0 * sS, bt.out48 + (size_t)r0 * GDN_VDIM);
        for (int r = r0; r < r0 + nr; r++) {
            k_rmsnorm_gated<<<GDN_VH, GDN_VD, 0, s>>>(
                bt.gdn_out_bf + (size_t)r * GDN_VH * GDN_VD, bt.out48 + (size_t)r * GDN_VH * GDN_VD,
                (const __nv_bfloat16 *)L->gdn_norm, bt.z6144 + (size_t)r * GDN_VDIM, 1e-6f);
            k_bf16_to_f32<<<(GDN_VDIM + 1023) / 1024, 1024, 0, s>>>(
                bt.q6144 + (size_t)r * GDN_VDIM, bt.gdn_out_bf + (size_t)r * GDN_VH * GDN_VD, GDN_VDIM);
        }
        { QfDenseProj p[1] = {{L->gdn_out, bt.y2560 + (size_t)r0 * NEMBD, NEMBD}};
          qfd_gemv_group_T(p, 1, bt.q6144 + (size_t)r0 * GDN_VDIM, GDN_VDIM, nr, s); }
    } else {
        { QfDenseProj p[3] = {{L->wq, bt.q6144 + (size_t)r0 * NHEAD * QGATE, NHEAD * QGATE},
                              {L->wk, bt.k512 + (size_t)r0 * KVDIM, NKV * HDIM},
                              {L->wv, bt.v512 + (size_t)r0 * KVDIM, NKV * HDIM}};
          qfd_gemv_group_T(p, 3, bt.mixed + (size_t)r0 * NEMBD, NEMBD, nr, s); }
        qf_attn_qsa_layer_M_base(il, L->q_norm, L->k_norm, ctx.paramsT_dev, r0, nr, s);
        { QfDenseProj p[1] = {{L->wo, bt.y2560 + (size_t)r0 * NEMBD, NEMBD}};
          qfd_gemv_group_T(p, 1, bt.attn_out + (size_t)r0 * NHEAD * HDIM, NHEAD * HDIM, nr, s); }
    }
    for (int r = r0; r < r0 + nr; r++)
        k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(
            bt.R + (size_t)r * hcd, bt.y2560 + (size_t)r * NEMBD, bt.inj + (size_t)r * HCC);
    qf_hc_T(bt.R + (size_t)r0 * hcd, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up,
            L->hc_ffn_inject, bt.mixed + (size_t)r0 * NEMBD, bt.inj + (size_t)r0 * HCC, nr, s);
    for (int r = r0; r < r0 + nr; r++) {
        qfd_moe_inproj(L->router, L->shexp_gate, L->shexp_up, L->shexp_gate_inp,
                       bt.mixed + (size_t)r * NEMBD, ctx.router,
                       g_shgM + (size_t)r * NFF, g_shuM + (size_t)r * NFF, g_ginpM + r, s);
        k_router_topk<<<1, 128, 0, s>>>(ctx.router, g_selM + (size_t)r * NEXPUSED, g_wtsM + (size_t)r * NEXPUSED);
    }
}
// shared expert compute (Spark): silu+down -> g_edM for rows [r0,r0+nr). Needs
// only the frontend's shared in-projection, so it overlaps the AMD routed branch.
static void qf_shared_compute_wave(QfModel *m, int il, int r0, int nr, cudaStream_t s) {
    QfLayer *L = &m->layers[il];
    for (int r = r0; r < r0 + nr; r++) {
        k_silu_mul<<<3, 256, 0, s>>>(g_shgM + (size_t)r * NFF, g_shuM + (size_t)r * NFF, NFF);
        qfd_out_proj(L->shexp_down, g_shgM + (size_t)r * NFF, g_edM + (size_t)r * NEMBD, NEMBD, NFF, s);
    }
}
// join shared into routed y + stream inject, for rows [r0,r0+nr). Needs y2560[r0..].
static void qf_join_wave(QfModel *m, int il, int r0, int nr, cudaStream_t s) {
    (void)m; const int hcd = HCC * NEMBD;
    for (int r = r0; r < r0 + nr; r++) {
        k_shexp_add<<<13, 256, 0, s>>>(bt.y2560 + (size_t)r * NEMBD, g_edM + (size_t)r * NEMBD, g_ginpM + r);
        k_stream_inject<<<(hcd + 1023) / 1024, 1024, 0, s>>>(
            bt.R + (size_t)r * hcd, bt.y2560 + (size_t)r * NEMBD, bt.inj + (size_t)r * HCC);
    }
}

extern "C" int qf_decode_step_M_ab(QfModel *m, const int *tokensM, const long *posM, int M) {
    if (M != 8) { fprintf(stderr, "step_M_ab: requires M=8 (A/B waves)\n"); return -1; }
    if (!qf_routed_amd_enabled()) { fprintf(stderr, "step_M_ab: needs the AMD routed tier\n"); return -1; }
    if (qf_layer_begin() > 0) return -1;
    if (qf_batch_init(QF_SPEC_MAXT) || qf_reqbatch_state_init(M)) return -1;
    cudaStream_t s = ctx.s;
    const int hcd = HCC * NEMBD, K = NEXPUSED;
    qfd_gemv_group_T_rewind();
    qf_decode_set_params_M(tokensM, posM, M);
    for (int r = 0; r < M; r++)
        k_embed_dev<<<(NEMBD + 255) / 256, 256, 0, s>>>(ctx.x, bt.R + (size_t)r * hcd,
            (const __nv_bfloat16 *)m->tok_embd, ctx.paramsT_dev + r);
    for (int il = 0; il < NLAYER; il++) {
        qf_frontend_wave(m, il, 0, 4, s);                          // Spark: wave A frontend
        qf_routed_amd_layer_submit(il, (long long)posM[0], 4, K, g_selM, g_wtsM, bt.mixed, s);   // AMD: routed(A)
        qf_frontend_wave(m, il, 4, 4, s);                          // Spark(B) OVERLAPS AMD routed(A)
        qf_routed_amd_layer_submit(il, (long long)posM[4], 4, K, g_selM + 4 * K, g_wtsM + 4 * K, bt.mixed + 4 * NEMBD, s); // AMD: routed(B)
        qf_shared_compute_wave(m, il, 0, 8, s);                    // Spark shared(A,B) OVERLAPS AMD routed(B)
        if (qf_routed_amd_layer_wait(bt.y2560, 4, s) != 0) return -1;                 // collect A
        qf_join_wave(m, il, 0, 4, s);
        if (qf_routed_amd_layer_wait(bt.y2560 + 4 * NEMBD, 4, s) != 0) return -1;     // collect B
        qf_join_wave(m, il, 4, 4, s);
    }
    qf_hc_T(bt.R, m->output_hc_norm, m->output_hc_down, m->output_hc_up, NULL, bt.mixed, NULL, M, s);
    { QfDenseProj p[1] = {{m->lm_head, bt.logits, NVOCAB}};
      if (qfd_gemv_group_T(p, 1, bt.mixed, NEMBD, M, s) != 0)
          for (int r = 0; r < M; r++)
              qfd_lm_head(m->lm_head, bt.mixed + (size_t)r * NEMBD, bt.logits + (size_t)r * NVOCAB, s); }
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}


// ===========================================================================
// Heterogeneous REGION decode: both boxes are complete engines. AMD owns rows
// [0, ma) and runs them through all 48 layers on its four MI50s; Spark owns
// rows [ma, M) and runs them through its own engine. The AMD submit is issued
// FIRST and collected LAST, so the two domains execute concurrently and only
// token ids cross the wire (no per-layer traffic). QF_REGION_AMD_ROWS=ma
// (default 4 => A={0..3} on AMD, B={4..7} on Spark).
// ===========================================================================
// ---- resident-sequence continuation state (region1, fused head+prefix path) ----
static int   g_r1_exit_valid = 0, g_r1_exit_tok = -1;
static long  g_r1_exit_pos = -1;
static float *g_r1_exit_resid = NULL;                 // AMD prefix residual of the token at g_r1_exit_pos (Spark tail pending)
extern "C" int qf_region1_exit_state(int *tok, long *pos, const float **resid) {
    if (!g_r1_exit_valid) return -1;
    if (tok) *tok = g_r1_exit_tok; if (pos) *pos = g_r1_exit_pos; if (resid) *resid = g_r1_exit_resid;
    return 0;
}
// qf_ingest_region: add a suffix to the RESIDENT sequence without reset or replay.
//   resident state -> finish the pending Spark tail of the exit token (row 0, its AMD residual already in
//   hand) -> AMD prefix + Spark tail for suffix[0..N-2] only -> the last suffix token becomes the current
//   token (prefix + tail + head run in the first decode step, exactly as after a prime).
// Every state buffer advances by exactly N positions: AMD (KV, GDN, conv, PLE history) sees positions
// pos0+1..pos0+N-1 through the chunk requests and pos0+N in the next decode step; Spark (KV, GDN, conv,
// indexer pools) sees pos0..pos0+N-1 through the chunk tail. Same chunk machinery as the prime
// (qf_decode_batch_prime_region), same two-chunks-in-flight pipeline, positions offset by pos0.
extern "C" int qf_ingest_region(QfModel *m, const int *suffix, int N, int *tok_out, long *pos_out) {
    if (!qf_region_amd_enabled() || N < 1) return -1;
    if (!g_r1_exit_valid) { fprintf(stderr, "ingest: no resident exit state (decode with the fused region1 path first)\n"); return -1; }
    const int hcd = HCC * NEMBD;
    const int tok0 = g_r1_exit_tok; const long pos0 = g_r1_exit_pos;
    static int *rows = NULL; static int rows_cap = 0;
    if (N > rows_cap) { rows_cap = N; rows = (int *)realloc(rows, (size_t)rows_cap * sizeof(int)); }
    rows[0] = tok0; for (int i = 1; i < N; i++) rows[i] = suffix[i - 1];
    const int NR = N;                                 // rows through the chunk tail: tok0 + suffix[0..N-2]
    static float *hRc[3] = {NULL, NULL, NULL};
    for (int b = 0; b < 3; b++)
        if (!hRc[b] && cudaHostAlloc((void **)&hRc[b], (size_t)qf_pf_maxt() * hcd * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    int TC = getenv("QF_PRIME_T") ? atoi(getenv("QF_PRIME_T")) : 16;
    if (TC > 1024) TC = 1024;                        // ingest sub-chunks of 1024: AMD(n+1) overlaps Spark(n) (2,038 tok: 2.5 -> 2.19 s; 512 measured 2.24 s)
    if (getenv("QF_INGEST_T")) TC = atoi(getenv("QF_INGEST_T"));   // override
    if (TC < 1) TC = 1; if (TC > qf_pf_maxt()) TC = qf_pf_maxt();
    int TA = getenv("QF_AMD_CHUNK") ? atoi(getenv("QF_AMD_CHUNK")) : 256;
    if (TA < 1) TA = 1; if (TA > QF_PF_MAXT) TA = QF_PF_MAXT;
    const int nck = (NR + TC - 1) / TC;
    qf_set_skip_head(1);
    // AMD requests for chunk c: rows [max(c0,1), c0+T) in TA pieces; returns the number of pieces
    auto submit = [&](int c) -> int {
        if (c >= nck) return 0;
        const int c0 = c * TC, T = (NR - c0) < TC ? (NR - c0) : TC; float *dst = hRc[c % 3];
        if (c == 0) memcpy(dst, g_r1_exit_resid, (size_t)hcd * sizeof(float));   // pending tail's residual = row 0
        int np = 0;
        for (int r = (c0 < 1 ? 1 : c0); r < c0 + T; r += TA) {
            const int tk = (c0 + T - r) < TA ? (c0 + T - r) : TA;
            if (qf_region_amd_prefix_chunk_submit_to(rows + r, tk, pos0 + r, dst + (size_t)(r - c0) * hcd, hcd) != 0) return -1;
            np++;
        }
        return np;
    };
    static int npend[4096];                           // pieces per chunk (nck <= maxpos/TC)
    if (nck > 4096) { qf_set_skip_head(0); return -1; }
    for (int c = 0; c < 2 && c < nck; c++) { const int np = submit(c); if (np < 0) { qf_set_skip_head(0); return -1; } npend[c] = np; }
    static int ptim = -1; if (ptim < 0) ptim = getenv("QF_M8_TIMING") ? 1 : 0;
    for (int c = 0; c < nck; c++) {
        const int c0 = c * TC, T = (NR - c0) < TC ? (NR - c0) : TC; float *hb = hRc[c % 3];
        struct timespec tw0, tw1, tw2; if (ptim) clock_gettime(CLOCK_MONOTONIC, &tw0);
        for (int k = 0; k < npend[c]; k++)
            if (qf_region_amd_prefix_chunk_wait_done() != 0) { fprintf(stderr, "ingest chunk@%d: AMD prefix failed\n", c0); qf_set_skip_head(0); return -1; }
        if (ptim) clock_gettime(CLOCK_MONOTONIC, &tw1);
        if (c + 2 < nck) { const int np = submit(c + 2); if (np < 0) { qf_set_skip_head(0); return -1; } npend[c + 2] = np; }
        const long p0 = pos0 + c0;
        if (T == 1) {
            if (qf_push_residual(m, hb) != 0) { qf_set_skip_head(0); return -1; }
            if (qf_decode_step_tail(m, rows[c0], p0) != 0) { fprintf(stderr, "ingest tail@%ld failed\n", p0); qf_set_skip_head(0); return -1; }
        } else if (T <= QF_SPEC_MAXT) {
            if (qf_push_residual_M(hb, T) != 0) { qf_set_skip_head(0); return -1; }
            if (qf_decode_body_T(m, rows + c0, T, p0) != 0) { fprintf(stderr, "ingest chunk@%ld (T=%d) failed\n", p0, T); qf_set_skip_head(0); return -1; }
        } else {
            if (qf_prefill_chunk_tail(m, rows + c0, T, p0, hb) != 0) { fprintf(stderr, "ingest large chunk@%ld (T=%d) failed\n", p0, T); qf_set_skip_head(0); return -1; }
        }
        if (ptim) { clock_gettime(CLOCK_MONOTONIC, &tw2);
                    fprintf(stderr, "ingest chunk@%ld T=%d: waited AMD %.0f ms, Spark %.0f ms\n", p0, T,
                            (tw1.tv_sec - tw0.tv_sec) * 1e3 + (tw1.tv_nsec - tw0.tv_nsec) / 1e6, (tw2.tv_sec - tw1.tv_sec) * 1e3 + (tw2.tv_nsec - tw1.tv_nsec) / 1e6); }
    }
    qf_set_skip_head(0);
    { cudaError_t e = cudaDeviceSynchronize(); cudaError_t l = cudaGetLastError();
      if (e != cudaSuccess || l != cudaSuccess) fprintf(stderr, "ingest: CUDA state after the chunk pass: sync %s, last %s\n", cudaGetErrorString(e), cudaGetErrorString(l)); }
    qf_tail_graph_invalidate();                       // graph lifetime rule: recapture after a chunk ingest
    g_r1_exit_valid = 0;                              // the new current token has no prefix yet
    if (tok_out) *tok_out = suffix[N - 1];
    if (pos_out) *pos_out = pos0 + N;
    return 0;
}
// ---- FORK: one resident conditioned parent -> M sibling rows (Kolmogorov branch point) ----
// The parent is the M=1 resident state right after a prime: KV [0, P) for the QSA layers, GDN state and
// conv ring for the recurrent layers (the M=1 prime's ctx.* buffers). Each sibling row r gets its own copy
// in the M-row slots (g_kc_M/g_vc_M rows, g_gdnS_M/g_convring_M rows) - the same per-row state the M-row
// prime builds one row at a time from scratch. The parent's current token (the last prompt id, not yet
// prefixed) then starts every row at position P: step 0 of the M-row region decode prefixes it per row.
// One expensive conditioned state, M cheap continuations; the parent buffers stay untouched.
extern "C" int qf_fork_rows(int M, long P) {
    if (M < 1 || M > QF_SPEC_MAXT || P < 0) return -1;
    if (qf_reqbatch_state_init(M)) return -1;
    cudaStream_t s = ctx.s;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    const size_t span = (size_t)qf_maxpos() * KVDIM;
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        for (int r = 0; r < M; r++) {
            if (((il + 1) % 4) != 0) {
                cudaMemcpyAsync(g_gdnS_M[il] + (size_t)r * sS, ctx.gdnS[il], sS * sizeof(float), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_convring_M[il] + (size_t)r * 3 * DINN, ctx.convring[il], (size_t)3 * DINN * sizeof(float), cudaMemcpyDeviceToDevice, s);
            } else if (P > 0) {
                const size_t nb = (size_t)P * KVDIM * 2;
                cudaMemcpyAsync(g_kc_M[il] + (size_t)r * span, ctx.kc[il], nb, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(g_vc_M[il] + (size_t)r * span, ctx.vc[il], nb, cudaMemcpyDeviceToDevice, s);
            }
        }
    }
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}
// COMMIT (decision/commit loop): copy sibling row r's resident state back into the parent
// (ctx.*) buffers. Exact reverse of qf_fork_rows - same buffers, src/dst swapped.
extern "C" int qf_fork_commit_rows(int r, long P) {
    if (r < 0 || r >= QF_SPEC_MAXT || P < 0) return -1;
    cudaStream_t s = ctx.s;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    const size_t span = (size_t)qf_maxpos() * KVDIM;
    for (int il = qf_layer_begin(); il < NLAYER; il++) {
        if (((il + 1) % 4) != 0) {
            cudaMemcpyAsync(ctx.gdnS[il], g_gdnS_M[il] + (size_t)r * sS, sS * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ctx.convring[il], g_convring_M[il] + (size_t)r * 3 * DINN, (size_t)3 * DINN * sizeof(float), cudaMemcpyDeviceToDevice, s);
        } else if (P > 0) {
            const size_t nb = (size_t)P * KVDIM * 2;
            cudaMemcpyAsync(ctx.kc[il], g_kc_M[il] + (size_t)r * span, nb, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ctx.vc[il], g_vc_M[il] + (size_t)r * span, nb, cudaMemcpyDeviceToDevice, s);
        }
    }
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}
// Per-row state signature (fork independence proof): sum of the row's GDN states and conv rings over the
// Spark layers plus its KV entries at positions [P-4, P+4). Rows that received identical inputs must have
// identical signatures; a row that received a different token must differ; the others must not move.
extern "C" int qf_fork_state_sig(int M, long P, double *sig) { /* sig[2*r] = recurrent (GDN+conv), sig[2*r+1] = KV [P-4,P) */
    if (M < 1 || M > QF_SPEC_MAXT || !g_gdnS_M) return -1;
    const size_t sS = (size_t)GDN_VH * GDN_KD * GDN_VD;
    const size_t span = (size_t)qf_maxpos() * KVDIM;
    static float *hb = NULL; static __nv_bfloat16 *hk = NULL;
    if (!hb) { hb = (float *)malloc(sS * sizeof(float)); hk = (__nv_bfloat16 *)malloc((size_t)8 * KVDIM * 2 * 2); }
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    for (int r = 0; r < M; r++) {
        double acc = 0.0, acck = 0.0;
        for (int il = qf_layer_begin(); il < NLAYER; il++) {
            if (((il + 1) % 4) != 0) {
                if (cudaMemcpy(hb, g_gdnS_M[il] + (size_t)r * sS, sS * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
                for (size_t i = 0; i < sS; i += 7) acc += hb[i];
                if (cudaMemcpy(hb, g_convring_M[il] + (size_t)r * 3 * DINN, (size_t)3 * DINN * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
                for (size_t i = 0; i < (size_t)3 * DINN; i++) acc += hb[i];
            } else {
                long p0 = P - 4; if (p0 < 0) p0 = 0; long p1 = P; if (p1 > (long)qf_maxpos()) p1 = qf_maxpos();
                const size_t n = (size_t)(p1 - p0) * KVDIM;
                if (cudaMemcpy(hk, g_kc_M[il] + (size_t)r * span + (size_t)p0 * KVDIM, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
                for (size_t i = 0; i < n; i++) acck += __bfloat162float(hk[i]);
                if (cudaMemcpy(hk, g_vc_M[il] + (size_t)r * span + (size_t)p0 * KVDIM, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
                for (size_t i = 0; i < n; i++) acck += __bfloat162float(hk[i]);
            }
        }
        sig[2 * r] = acc; sig[2 * r + 1] = acck;
    }
    return 0;
}
extern "C" int qf_decode_batch_run_region(QfModel *m, const int *init_tok, const long *init_pos,
                                          const int (*init_hist)[3], int M, int max_new,
                                          int *out_tokens, int *out_len) {
    if (M < 1 || M > QF_SPEC_MAXT) return -1;
    if (!qf_region_amd_enabled()) return -1;
    // Receipted alternating map (handoff 2026-08-31 evening, this hardware):
    //   AMD prefix (layers 0..S-1, embed + PLE)  ->  Spark tail (S..47)  ->  AMD
    //   out HC + lm_head + argmax -> 4 B token id.
    // Two coarse crossings per token; no logits on the wire; the head runs where
    // it was measured fastest (1204 GB/s at 3.84x overlap across the four cards).
    const int hcd = HCC * NEMBD;
    static float *hR = NULL;
    if (!hR && cudaHostAlloc((void **)&hR, (size_t)QF_SPEC_MAXT * hcd * sizeof(float),
                             cudaHostAllocDefault) != cudaSuccess) return -1;
    int  tok[QF_SPEC_MAXT], hist[QF_SPEC_MAXT][3], amd_out[QF_SPEC_MAXT];
    long pos[QF_SPEC_MAXT];
    for (int r = 0; r < M; r++) {
        tok[r] = init_tok[r]; pos[r] = init_pos[r];
        hist[r][0] = init_hist[r][0]; hist[r][1] = init_hist[r][1]; hist[r][2] = init_hist[r][2];
        out_len[r] = 0;
    }
        // This box must own exactly the tail: QF_LAYER_BEGIN == the AMD prefix depth
    // (QF_AMD_PREFIX on the region server). With QF_LAYER_BEGIN unset the device
    // body would embed and overwrite the pushed residual.
    if (qf_layer_begin() <= 0) {
        fprintf(stderr, "region: QF_LAYER_BEGIN must be set to the AMD prefix depth (QF_AMD_PREFIX)\n");
        return -1;
    }
        // (No region reset here: qf_decode_batch_prime_region already reset the AMD
    // rows and then primed them; a reset now would erase the prompt state.)
    const int tm = getenv("QF_M8_TIMING") ? 1 : 0;
    if (M == 1) {
        // THE PRODUCT PATH: one request through the dedicated single-stream
        // executor (ctx state, eager qf_decode_body over layers lb..47, no PLE
        // staging, no Spark head), not the M-row batched executor at M=1.
        // Natural EOS: generation_config.json lists eos_token_id = [248046, 248044]
        // (<|im_end|>, <|endoftext|>); QF_STOP_IM_END=0 keeps only <|endoftext|>.
        static int stop_im_end = -1;
        if (stop_im_end < 0) { const char *e = getenv("QF_STOP_IM_END"); stop_im_end = e ? atoi(e) : 1; }
        qf_set_skip_head(1);
        int stop_tok = -1, steps = 0;
        // QF_HEADPREFIX=1 (default): ONE wire round trip per step - the AMD head
        // and the prefix of the token it produced run back-to-back on the AMD box
        // (QFW_HEADPREFIX). The first step still needs a plain prefix. The timing
        // print then reports the fused RPC under "AMD head" and ~0 under prefix.
        static int fused = -1;
        if (fused < 0) { const char *e = getenv("QF_HEADPREFIX"); fused = (e && e[0] == '0') ? 0 : 1; }
        static float *hR1 = NULL;
        if (fused && !hR1 && cudaHostAlloc((void **)&hR1, (size_t)hcd * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
        int have_prefix = 0;
        float *hRa = hR, *hRb = hR1;                  // local pair; the statics stay as allocated
        // CONTINUATION: if the caller resumes from the exact exit state of the previous region1 call
        // (or of qf_ingest_region), the current token's AMD prefix has already run and its residual is
        // here - reuse it instead of re-running the prefix (which would double-advance AMD's recurrent state).
        if (fused && g_r1_exit_valid && tok[0] == g_r1_exit_tok && pos[0] == g_r1_exit_pos) {
            memcpy(hRa, g_r1_exit_resid, (size_t)hcd * sizeof(float)); have_prefix = 1; }
        g_r1_exit_valid = 0;
        for (int step = 0; step < max_new; step++) {
            const double t0 = tm ? m8_now() : 0;
            if (!fused || !have_prefix) {
                if (qf_region_amd_prefix(1, tok, pos, hRa, hcd) != 0) { fprintf(stderr, "region1: AMD prefix RPC failed at step %d\n", step); qf_set_skip_head(0); return -1; }
            }
            const double t1 = tm ? m8_now() : 0;
            if (qf_push_residual(m, hRa) != 0) { fprintf(stderr, "region1: push residual failed at step %d\n", step); qf_set_skip_head(0); return -1; }
            if (qf_decode_step_tail(m, tok[0], pos[0]) != 0) { fprintf(stderr, "region1: Spark tail failed at step %d\n", step); qf_set_skip_head(0); return -1; }
            if (qf_pull_residual(m, hRa) != 0) { fprintf(stderr, "region1: pull residual failed at step %d: %s\n", step, cudaGetErrorString(cudaGetLastError())); qf_set_skip_head(0); return -1; }
            const double t2 = tm ? m8_now() : 0;
            if (fused) {
                // head(hRa) -> token t, then prefix(t at pos+1) -> hRb, in one RPC
                if (qf_region_amd_headprefix(hRa, hcd, pos[0] + 1, &amd_out[0], hRb) != 0) { fprintf(stderr, "region1: AMD head+prefix RPC failed at step %d\n", step); qf_set_skip_head(0); return -1; }
                float *sw = hRa; hRa = hRb; hRb = sw;   // hRa now holds the NEXT step's prefix residual
                have_prefix = 1;
            } else
            if (qf_region_amd_head(1, hRa, hcd, amd_out) != 0) { fprintf(stderr, "region1: AMD head RPC failed at step %d\n", step); qf_set_skip_head(0); return -1; }
            if (tm) { g_t_prefix += t1 - t0; g_t_tail += t2 - t1; g_t_head += m8_now() - t2; g_region_steps++; }
            const int t = amd_out[0];
            out_tokens[out_len[0]] = t;
            out_len[0]++; pos[0]++; tok[0] = t; steps++;
            qf_m8_steps_done++;                   // watchdog progress
            if (t == QF_EOS_ID || (stop_im_end && t == 248046)) { stop_tok = t; break; }
        }
        qf_set_skip_head(0);
        // EXIT INVARIANT (fused path): positions <= pos[0]-1 have AMD prefix + Spark tail; the token tok[0]
        // at pos[0] has its AMD prefix done (fused RPC) with the residual in hRa and its Spark tail PENDING.
        if (fused && have_prefix) {
            if (!g_r1_exit_resid) cudaHostAlloc((void **)&g_r1_exit_resid, (size_t)hcd * sizeof(float), cudaHostAllocDefault);
            if (g_r1_exit_resid) { memcpy(g_r1_exit_resid, hRa, (size_t)hcd * sizeof(float)); g_r1_exit_tok = tok[0]; g_r1_exit_pos = pos[0]; g_r1_exit_valid = 1; }
        }
        if (stop_tok >= 0) fprintf(stderr, "region1: NATURAL EOS (token %d) after %d generated tokens\n", stop_tok, steps);
        else fprintf(stderr, "region1: budget reached (%d tokens, QF_M8_MAX) without EOS\n", steps);
        return 0;
    }
    for (int step = 0; step < max_new; step++) {
        const double t0 = tm ? m8_now() : 0;
                if (qf_region_amd_prefix(M, tok, pos, hR, hcd) != 0) { fprintf(stderr, "region: AMD prefix RPC failed at step %d\n", step); return -1; }
        const double t1 = tm ? m8_now() : 0;
        if (qf_push_residual_M(hR, M) != 0) { fprintf(stderr, "region: push residual failed at step %d\n", step); return -1; }
        if (qf_decode_step_M_upto_ex(m, tok, pos, M, NLAYER, /*want_logits=*/0) != 0) { fprintf(stderr, "region: Spark tail failed at step %d\n", step); return -1; }
        if (qf_pull_residual_M(hR, M) != 0) { fprintf(stderr, "region: pull residual failed at step %d\n", step); return -1; }
        const double t2 = tm ? m8_now() : 0;
        if (qf_region_amd_head(M, hR, hcd, amd_out) != 0) { fprintf(stderr, "region: AMD head RPC failed at step %d\n", step); return -1; }
        if (tm) { g_t_prefix += t1 - t0; g_t_tail += t2 - t1; g_t_head += m8_now() - t2; g_region_steps++; }
        for (int r = 0; r < M; r++) {
            out_tokens[(size_t)r * max_new + out_len[r]] = amd_out[r];
            out_len[r]++; pos[r]++; tok[r] = amd_out[r];
        }
        qf_m8_steps_done++;                       // watchdog progress (region1)
    }
    return 0;
}

// ===========================================================================
// W2: TWO IN-FLIGHT BATCHES on the alternating map. M rows = nslots x 8; slot s
// owns rows [8s, 8s+8) of the persistent state on BOTH boxes (the AMD prefix
// request carries the row base, the Spark step carries it as r0). While the
// Spark runs slot s's tail, the AMD box runs the OTHER slot's prefix, which was
// queued before this tail started; the head and the two residual hops remain
// exposed. Replies are FIFO on the wire: a request registers its buffer at
// submit and qf_region_amd_drain_one() delivers the oldest reply, so a wait is
// "drain until my flag is set", never an assumption about ordering.
//
//   submit prefix(0) .. prefix(nslots-1)
//   for each tail t, slot s = t % nslots:
//     drain until prefix(s) landed          -> AMD did it during the previous tail
//     push R(s); tail(s) [that slot's graph]; pull R(s)
//     submit head(s); drain until head(s) landed   (exposed: head + 2 hops)
//     append tokens(s); submit prefix(s) with them  -> AMD busy during the next tail
// Per period this hides AMD's prefix behind Spark's tail; it does not raise
// Spark's per-step ceiling (8 tokens per tail). Wider steps (M per tail) are
// the next lever once receipt 1 shows where W1 put the Spark step.
// ===========================================================================
extern "C" int qf_region_amd_prefix_submit(int slot, int r0, int nr, const int *tokens, const long *posM,
                                           float *rout, int rsize);
extern "C" int qf_region_amd_head_submit(int slot, int nr, const float *rin, int rsize, int *ids_out);
extern "C" int qf_region_amd_drain_one(int *op_out, int *slot_out);
extern "C" int qf_region_amd_outstanding(void);
static double g_t2_wait_prefix = 0, g_t2_tail = 0, g_t2_wait_head = 0; static long g_t2_tails = 0;
extern "C" void qf_m8_region2_timing(double *wp, double *tail, double *wh, long *tails) {
    *wp = g_t2_wait_prefix; *tail = g_t2_tail; *wh = g_t2_wait_head; *tails = g_t2_tails;
}
extern "C" int qf_decode_batch_run_region2(QfModel *m, const int *init_tok, const long *init_pos,
                                           const int (*init_hist)[3], int M, int max_new,
                                           int *out_tokens, int *out_len) {
    const int ms = QF_SPEC_MAXT;                       // rows per slot (one Spark tail)
    if (M < 2 * ms || M > QF_M8_MAXROWS || (M % ms) != 0) {
        fprintf(stderr, "region2: M must be a multiple of %d in [%d, %d]\n", ms, 2 * ms, QF_M8_MAXROWS); return -1; }
    const int nslots = M / ms;
    if (!qf_region_amd_enabled()) return -1;
    if (qf_layer_begin() <= 0) {
        fprintf(stderr, "region2: QF_LAYER_BEGIN must be set to the AMD prefix depth (QF_AMD_PREFIX)\n"); return -1; }
    if (qf_batch_init(QF_SPEC_MAXT) || qf_reqbatch_state_init(M)) return -1;
    (void)init_hist;                                   // PLE history lives on the AMD box in this map
    const int hcd = HCC * NEMBD;
    static float *hR[QF_M8_NSLOTS] = {NULL, NULL};
    for (int sl = 0; sl < nslots; sl++)
        if (!hR[sl] && cudaHostAlloc((void **)&hR[sl], (size_t)ms * hcd * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
            return -1;
    int  tok[QF_M8_MAXROWS], ids[QF_M8_NSLOTS][QF_SPEC_MAXT];
    long pos[QF_M8_MAXROWS];
    int  ready_prefix[QF_M8_NSLOTS] = {0, 0}, ready_head[QF_M8_NSLOTS] = {0, 0};
    for (int r = 0; r < M; r++) { tok[r] = init_tok[r]; pos[r] = init_pos[r]; out_len[r] = 0; }
    const int tm = getenv("QF_M8_TIMING") ? 1 : 0;
    // (No region reset: the priming already built every row's AMD state.)
    for (int sl = 0; sl < nslots; sl++)
        if (qf_region_amd_prefix_submit(sl, sl * ms, ms, tok + sl * ms, pos + sl * ms, hR[sl], hcd)) return -1;
    const long total_tails = (long)max_new * nslots;
    for (long t = 0; t < total_tails; t++) {
        const int sl = (int)(t % nslots);
        const double t0 = tm ? m8_now() : 0;
        while (!ready_prefix[sl]) {
            int op, who;
            if (qf_region_amd_drain_one(&op, &who)) { fprintf(stderr, "region2: wire failed (prefix wait)\n"); return -1; }
            if (op == 12 /*QFW_PREFIX_M*/) ready_prefix[who] = 1; else ready_head[who] = 1;
        }
        ready_prefix[sl] = 0;
        const double t1 = tm ? m8_now() : 0;
        if (qf_push_residual_M(hR[sl], ms) != 0) return -1;
        if (qf_decode_step_M_slot(m, tok + sl * ms, pos + sl * ms, ms, NLAYER, /*want_logits=*/0, sl * ms) != 0) return -1;
        if (qf_pull_residual_M(hR[sl], ms) != 0) return -1;
        const double t2 = tm ? m8_now() : 0;
        if (qf_region_amd_head_submit(sl, ms, hR[sl], hcd, ids[sl])) return -1;
        while (!ready_head[sl]) {
            int op, who;
            if (qf_region_amd_drain_one(&op, &who)) { fprintf(stderr, "region2: wire failed (head wait)\n"); return -1; }
            if (op == 12 /*QFW_PREFIX_M*/) ready_prefix[who] = 1; else ready_head[who] = 1;
        }
        ready_head[sl] = 0;
        for (int i = 0; i < ms; i++) {
            const int r = sl * ms + i;
            out_tokens[(size_t)r * max_new + out_len[r]] = ids[sl][i];
            out_len[r]++; pos[r]++; tok[r] = ids[sl][i];
        }
        qf_m8_steps_done++;                       // watchdog progress (region2)
        if (t + nslots < total_tails)                  // this slot has another tail coming
            if (qf_region_amd_prefix_submit(sl, sl * ms, ms, tok + sl * ms, pos + sl * ms, hR[sl], hcd)) return -1;
        if (tm) { g_t2_wait_prefix += t1 - t0; g_t2_tail += t2 - t1; g_t2_wait_head += m8_now() - t2; g_t2_tails++; }
    }
    while (qf_region_amd_outstanding()) {              // nothing expected; never leave a reply unread
        int op, who;
        if (qf_region_amd_drain_one(&op, &who)) return -1;
    }
    return 0;
}

int qf_decode_body_T(QfModel *m, const int *tokens, int T, long pos0) {
    if (T < 1 || T > QF_SPEC_MAXT) { fprintf(stderr, "body_T: bad T=%d\n", T); return -1; }
    if (qf_batch_init(QF_SPEC_MAXT)) { fprintf(stderr, "body_T: init failed\n"); return -1; }
    cudaStream_t s = ctx.s;

    qf_decode_set_params_T(tokens, pos0, T);
    qf_body_T_host(m, tokens, T);

    if (bodyT_graph_enabled() && g_bodyT_warm[T]) {
        if (!g_bodyT_exec[T] && qf_bodyT_capture(m, T) != 0) g_bodyT_graph = 0;
        if (g_bodyT_exec[T]) {
            if (cudaGraphLaunch(g_bodyT_exec[T], s) != cudaSuccess) return -1;
            return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
        }
    }
    g_bodyT_warm[T] = 1;
    if (qf_body_T_device(m, T) != 0) return -1;
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}

static void qf_bodyT_graph_destroy(void) {
    for (int t = 0; t <= QF_SPEC_MAXT; t++)
        if (g_bodyT_exec[t]) { cudaGraphExecDestroy(g_bodyT_exec[t]); g_bodyT_exec[t] = NULL; }
}

float *qf_batch_logits(void) { return bt.logits; }

// Greedy argmax of one row of the batched logits, on device.
int qf_batch_argmax(int t) {
    static float *bv = NULL; static int *bi = NULL, *dout = NULL, *hout = NULL;
    if (!bv) {
        if (cudaMalloc(&bv, QF_ARGMAX_BLOCKS * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMalloc(&bi, QF_ARGMAX_BLOCKS * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc(&dout, sizeof(int)) != cudaSuccess) return -1;
        if (cudaHostAlloc((void **)&hout, sizeof(int), cudaHostAllocDefault) != cudaSuccess) return -1;
    }
    const float *row = bt.logits + (size_t)t * NVOCAB;
    k_argmax_part<<<QF_ARGMAX_BLOCKS, 256, 0, ctx.s>>>(row, NVOCAB, bv, bi);
    k_argmax_fin<<<1, 1, 0, ctx.s>>>(bv, bi, dout);
    if (cudaMemcpyAsync(hout, dout, sizeof(int), cudaMemcpyDeviceToHost, ctx.s) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(ctx.s) != cudaSuccess) return -1;
    return *hout;
}

// One speculative round, greedy.
//
//   draft : chain `nsteps` MTP steps from (last_token, trunk hc streams) to get
//           nsteps proposals
//   verify: ONE batched trunk forward over [last_token, d0, .. d_{n-1}], which
//           yields the trunk's own next token at each of those positions
//   accept: the trunk's output at slot j IS the correct token for position
//           pos+j. Take the longest prefix where the draft agreed, plus one -
//           the trunk's own token is always correct, so a round commits at
//           least one token and the emitted stream is EXACTLY the greedy
//           sequential stream (docs/PHASE_SPEC_DECODE.md §2 point 2).
//
// Returns the number of tokens committed (>=1), writes them to out[].
// `pos` is the absolute position of the first committed token. On entry the
// trunk state must already reflect everything before `pos`.
static long g_spec_step_n[QF_SPEC_MAXT], g_spec_step_hit[QF_SPEC_MAXT];

// Round-phase wall clock. A round is draft + verify (+ a commit re-run when
// the round was cut short), and which of those dominates decides what is worth
// optimizing. Both phases end in a stream sync, so host wall clock measures
// them honestly; the cost is two clock_gettime calls per phase.
static double g_spec_ms_draft, g_spec_ms_verify, g_spec_ms_commit;
static double spec_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}
void qf_spec_phase_ms(double *draft, double *verify, double *commit) {
    if (draft) *draft = g_spec_ms_draft;
    if (verify) *verify = g_spec_ms_verify;
    if (commit) *commit = g_spec_ms_commit;
}

// Per-draft-step hit counts for the acceptance gate. n[i] rounds ran step i,
// hit[i] of them produced the trunk's own token at that position.
void qf_spec_step_stats(long *n, long *hit, int maxn) {
    for (int i = 0; i < maxn && i < QF_SPEC_MAXT; i++) {
        if (n) n[i] = g_spec_step_n[i];
        if (hit) hit[i] = g_spec_step_hit[i];
    }
}
void qf_spec_step_stats_reset(void) {
    for (int i = 0; i < QF_SPEC_MAXT; i++) g_spec_step_n[i] = g_spec_step_hit[i] = 0;
    g_spec_ms_draft = g_spec_ms_verify = g_spec_ms_commit = 0.0;
}

int qf_spec_round(QfModel *m, int last_token, long pos, int nsteps, int *out, int *n_accepted) {
    // QF_SPEC_NODRAFT=1 forces T=1: the batched forward then has to reproduce
    // plain sequential decode exactly. It isolates "is the batched body right?"
    // from "does multi-position state advance right?".
    static int nodraft = -1;
    if (nodraft < 0) nodraft = getenv("QF_SPEC_NODRAFT") ? 1 : 0;
    if (m->exp_mode != QF_SPARK_MODE_FULL) {
        fprintf(stderr, "spec: needs QF_EXPERT_FULL_RESIDENT=1 "
                        "(the batched body does not service expert routing)\n");
        return -1;
    }
    if (nodraft) nsteps = 0;
    else if (nsteps < 1) nsteps = 1;
    if (nsteps > QF_SPEC_MAXT - 1) nsteps = QF_SPEC_MAXT - 1;
    if (qf_batch_init(QF_SPEC_MAXT)) { fprintf(stderr, "spec: batch init failed\n"); return -1; }

    // ---- draft ----
    const double t_draft0 = spec_now_ms();
    int cand[QF_SPEC_MAXT];
    cand[0] = last_token;
    int ndraft = 0;
    {
        float *hc = qf_decode_hc_streams();           // trunk pre-mixer streams
        static float *dlog = NULL;
        if (!dlog && cudaMalloc(&dlog, (size_t)NVOCAB * 4) != cudaSuccess) { fprintf(stderr, "spec: dlog alloc failed\n"); return -1; }
        int tok = last_token;
        for (int i = 0; i < nsteps; i++) {
            if (qf_mtp_step(m, tok, pos + i, hc, NULL, dlog) != 0) { fprintf(stderr, "spec: mtp_step %d failed\n", i); break; }
            // argmax of the draft head's logits
            static float *bv = NULL; static int *bi = NULL, *dout = NULL, *hout = NULL;
            if (!bv) {
                if (cudaMalloc(&bv, QF_ARGMAX_BLOCKS * sizeof(float)) != cudaSuccess) return -1;
                if (cudaMalloc(&bi, QF_ARGMAX_BLOCKS * sizeof(int)) != cudaSuccess) return -1;
                if (cudaMalloc(&dout, sizeof(int)) != cudaSuccess) return -1;
                if (cudaHostAlloc((void **)&hout, sizeof(int), cudaHostAllocDefault) != cudaSuccess) return -1;
            }
            k_argmax_part<<<QF_ARGMAX_BLOCKS, 256, 0, ctx.s>>>(dlog, NVOCAB, bv, bi);
            k_argmax_fin<<<1, 1, 0, ctx.s>>>(bv, bi, dout);
            cudaMemcpyAsync(hout, dout, sizeof(int), cudaMemcpyDeviceToHost, ctx.s);
            { cudaError_t e = cudaStreamSynchronize(ctx.s);
              if (e != cudaSuccess) { fprintf(stderr, "spec: draft sync %d: %s\n", i, cudaGetErrorString(e)); return -1; } }
            tok = *hout;
            cand[1 + ndraft++] = tok;
            hc = qf_mtp_hc();                          // chain the MTP streams
        }
    }

    // ---- verify: one batched trunk forward over the candidate prefix ----
    const double t_verify0 = spec_now_ms();
    g_spec_ms_draft += t_verify0 - t_draft0;
    const int T = 1 + ndraft;
    if (qf_spec_snapshot(m) != 0) { fprintf(stderr, "spec: snapshot failed\n"); return -1; }
    if (qf_decode_body_T(m, cand, T, pos) != 0) {
        fprintf(stderr, "spec: batched forward failed (T=%d)\n", T);
        qf_spec_restore(m); return -1;
    }

    // ---- accept ----
    int trunk[QF_SPEC_MAXT];
    for (int t = 0; t < T; t++) {
        trunk[t] = qf_batch_argmax(t);
        if (trunk[t] < 0) { fprintf(stderr, "spec: batch_argmax %d failed: %s\n", t, cudaGetErrorString(cudaGetLastError())); qf_spec_restore(m); return -1; }
    }
    int acc = 0;                                  // drafts confirmed by the trunk
    while (acc < ndraft && trunk[acc] == cand[1 + acc]) acc++;
    // Per-step draft QUALITY, counted unconditionally: trunk[i] is the correct
    // token at pos+i whether or not the prefix before it was accepted, so
    // draft[i] == trunk[i] measures step i's head in isolation. The prefix
    // acceptance `acc` conflates step 3's accuracy with steps 1 and 2 surviving.
    for (int i = 0; i < ndraft; i++) {
        g_spec_step_n[i]++;
        if (trunk[i] == cand[1 + i]) g_spec_step_hit[i]++;
    }
    static int spec_dbg = -1;
    if (spec_dbg < 0) spec_dbg = getenv("QF_SPEC_DEBUG") ? 1 : 0;
    if (spec_dbg) {
        fprintf(stderr, "spec pos=%ld draft=[", pos);
        for (int i = 0; i < ndraft; i++) fprintf(stderr, "%d%s", cand[1 + i], i + 1 < ndraft ? "," : "");
        fprintf(stderr, "] trunk=[");
        for (int t = 0; t < T; t++) fprintf(stderr, "%d%s", trunk[t], t + 1 < T ? "," : "");
        fprintf(stderr, "] acc=%d\n", acc);
    }
    const int commit = acc + 1;                   // cand[0] + the confirmed drafts
    // Emit the CANDIDATES, not the trunk's outputs. cand[0] is the trunk's own
    // argmax from the state before this round, so it is already the correct
    // greedy token; cand[1..acc] are drafts the trunk just confirmed equal to
    // its own. trunk[i] is the token that comes AFTER cand[i], so emitting it
    // here ran the stream one position ahead and dropped the first token.
    for (int i = 0; i < commit; i++) out[i] = cand[i];
    if (n_accepted) *n_accepted = acc;

    // The batched forward advanced state over all T positions; only `commit`
    // of them are real. Roll back and re-run the BATCHED body at T=commit.
    //
    // The first version replayed `commit` full SEQUENTIAL forwards here, which
    // made a round cost a batched forward PLUS the sequential forwards it was
    // supposed to replace - unprofitable even at 100% acceptance (measured:
    // 10.77 tok/s at T=1 with no drafts, against 26.12 sequential). When every
    // draft is accepted, commit == T and the state is already correct, so the
    // fast path costs nothing at all.
    const double t_commit0 = spec_now_ms();
    g_spec_ms_verify += t_commit0 - t_verify0;

    if (commit < T) {
        if (qf_spec_restore(m) != 0) { fprintf(stderr, "spec: restore failed\n"); return -1; }
        if (qf_decode_body_T(m, cand, commit, pos) != 0) {
            fprintf(stderr, "spec: commit forward failed (T=%d)\n", commit); return -1;
        }
    }
    g_spec_ms_commit += spec_now_ms() - t_commit0;
    // The caller's next qf_argmax_token() reads ctx.logits, which the batched
    // body does not write; hand it the distribution after cand[commit-1].
    if (cudaMemcpyAsync(ctx.logits, bt.logits + (size_t)(commit - 1) * NVOCAB,
                        (size_t)NVOCAB * sizeof(float), cudaMemcpyDeviceToDevice,
                        ctx.s) != cudaSuccess) return -1;
    // Same handoff for the HIDDEN side, and this one is not cosmetic. The next
    // round's first draft step reads qf_decode_hc_streams() == ctx.R, which
    // only qf_decode_body writes; the batched body writes bt.R. Without this
    // copy every round after the first fed the head the trunk streams from the
    // LAST SEQUENTIAL STEP - the end of the prompt - frozen for the whole
    // generation. Step 1 then drafts from (fresh token embedding + stale
    // context) and steps 2..n chain off that, which is why they collapsed to a
    // fixed echo of the prompt no matter what had been generated since.
    // Row commit-1 is the last committed position in both paths: the fast path
    // has commit == T rows, the rollback path re-ran the body with T = commit.
    if (cudaMemcpyAsync(ctx.R, bt.R + (size_t)(commit - 1) * HCC * NEMBD,
                        (size_t)HCC * NEMBD * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.s) != cudaSuccess) return -1;
    return commit;
}
