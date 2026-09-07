// qf_moe_m8_wave64.hip - batched M=8 routed-expert kernels (gfx906 / wave64).
// See qf_moe_m8_wave64.h. Host syntax check:
//   g++ -DQF_HOST_CHECK -I. -fsyntax-only -x c++ qf_moe_m8_wave64.hip
#include "qf_moe_m8_wave64.h"
#include <math.h>

// gfx906 wave64: __shfl_down_sync masks are wave-relative and the code groups
// are 32/8 lanes. On ROCm 5.7 the sync form is unavailable, so use the legacy
// __shfl_down with the WIDTH argument (segments the wave into width-lane
// groups; the code groups align to those segments). On HIP >= 6 use the sync
// form with per-group masks.
#if !defined(QF_HOST_CHECK) && defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define M8_SHFL32(v, off) __shfl_down((v), (off), 32)
#define M8_SHFL8(v, off)  __shfl_down((v), (off), 8)
#else
#define M8_WMASK 0xFFFFFFFFFFFFFFFFULL
#define M8_MASK32(t) ((unsigned long long)(0xffffffffULL << (((t) & 32) ? 32 : 0)))
#define M8_MASK8(t)  ((unsigned long long)(0xffULL << ((t) & 56)))
#define M8_SHFL32(v, off) __shfl_down_sync(M8_MASK32(threadIdx.x), (v), (off))
#define M8_SHFL8(v, off)  __shfl_down_sync(M8_MASK8(threadIdx.x), (v), (off))
#endif
#ifdef QF_HOST_CHECK
#define M8_LAUNCH(k, g, b, sh, s, ...) k(__VA_ARGS__)
#else
#define M8_LAUNCH(k, g, b, sh, s, ...) k<<<g, b, sh, s>>>(__VA_ARGS__)
#endif

// Exact NVFP4 dequant, copied verbatim from qf_nvfp4_wave64.hip so this TU is
// self-contained and byte-identical to the validated single-token path
// (adjacent-pair nibble order: corr 0.9955 vs the BF16 original).
__device__ __forceinline__ float m8_e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
    float v = e ? __uint_as_float(((e + 126u) << 23) | (m << 22))
                : (m ? 0.5f : 0.f);
    return (nib & 8u) ? -v : v;
}
__device__ __forceinline__ float m8_ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}
// One 16-value scale group. Activation read as float4 (16 B) rather than 16
// scalar loads: the record measured 90 -> 262 GB/s from exactly this change.
__device__ __forceinline__ float m8_group_dot(uint2 pk, const float *xs) {
    const float4 v0 = *(const float4 *)(xs + 0);
    const float4 v1 = *(const float4 *)(xs + 4);
    const float4 v2 = *(const float4 *)(xs + 8);
    const float4 v3 = *(const float4 *)(xs + 12);
    const float xa[16] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
                           v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w };
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t b0 = (pk.x >> (8 * j)) & 0xFFu;
        uint32_t b1 = (pk.y >> (8 * j)) & 0xFFu;
        a = fmaf(m8_e2m1(b0 & 15u), xa[2 * j],     a);
        a = fmaf(m8_e2m1(b0 >> 4),  xa[2 * j + 1], a);
        a = fmaf(m8_e2m1(b1 & 15u), xa[2 * j + 8], a);
        a = fmaf(m8_e2m1(b1 >> 4),  xa[2 * j + 9], a);
    }
    return a;
}
// Two adjacent scale groups from ONE uint4 (16 B) weight load.
__device__ __forceinline__ void m8_group_dot2(uint4 pk, const float *xs,
                                              float s0, float s1, float *acc) {
    uint2 lo = make_uint2(pk.x, pk.y), hi = make_uint2(pk.z, pk.w);
    *acc = fmaf(s0, m8_group_dot(lo, xs), *acc);
    *acc = fmaf(s1, m8_group_dot(hi, xs + 16), *acc);
}

// ---------------------------------------------------------------------------
// gate+up+silu-mul, batched over the tokens routed to each expert.
// grid = (QF5_NFF / rpb, n_exp); block = rpb*32 (one 32-lane half-wave/row).
// The (row,group) weight pair pg/pu and its scale are read ONCE and dotted
// against up to QF5_M8_TX staged tokens - the weight-traffic amortization the
// single-token kernel does not have. Scales come from the card-resident tables
// s2g_res/s2u_res, indexed by the expert's local slot.
// ---------------------------------------------------------------------------
__global__ void k_m8_gateup(const uint8_t *__restrict__ Wg, const uint8_t *__restrict__ Sg,
                            const uint8_t *__restrict__ Wu, const uint8_t *__restrict__ Su,
                            const float *__restrict__ s2g_res, const float *__restrict__ s2u_res,
                            const float *__restrict__ x, float *__restrict__ hidden,
                            const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
                            const int *__restrict__ pair_tok, const int *__restrict__ n_exp_dev) {
    if (blockIdx.y >= *n_exp_dev) return;      // device-decided extent: no host round-trip
    __shared__ int   sh_slot, sh_base, sh_nt;
    __shared__ float sh_s2g, sh_s2u;
    __shared__ int   sh_tok[QF5_M8_TX];
    const int e = blockIdx.y;
    if (threadIdx.x == 0) {
        int sl = exp_slot[e];
        sh_slot = sl; sh_base = exp_ptr[e]; sh_nt = exp_ptr[e + 1] - exp_ptr[e];
        sh_s2g = s2g_res[sl]; sh_s2u = s2u_res[sl];
    }
    if (threadIdx.x < QF5_M8_TX) {
        int idx = exp_ptr[e] + threadIdx.x;
        sh_tok[threadIdx.x] = (threadIdx.x < exp_ptr[e + 1] - exp_ptr[e]) ? pair_tok[idx] : -1;
    }
    __syncthreads();
    const int slot = sh_slot, nt = sh_nt;
    const int rpb = blockDim.x >> 5;
    const int l   = threadIdx.x & 31;
    const int row = blockIdx.x * rpb + (threadIdx.x >> 5);
    const size_t woff = (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NEMBD / 2);
    const size_t soff = (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NEMBD / 16);
    const uint8_t *wg = Wg + woff, *wu = Wu + woff;
    const uint8_t *sg = Sg + soff, *su = Su + soff;

    float ag[QF5_M8_TX], au[QF5_M8_TX];
    #pragma unroll
    for (int k = 0; k < QF5_M8_TX; k++) { ag[k] = 0.f; au[k] = 0.f; }
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 32 + l;
        uint2 pg = *(const uint2 *)(wg + (size_t)g * 8);   // weight group read ONCE
        uint2 pu = *(const uint2 *)(wu + (size_t)g * 8);
        float sgv = m8_ue4m3(sg[g]), suv = m8_ue4m3(su[g]);
        const int xg = g * 16;
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++) {
            if (k < nt) {
                const float *xt = x + (size_t)sh_tok[k] * QF5_NEMBD + xg;  // from global (L2)
                ag[k] = fmaf(sgv, m8_group_dot(pg, xt), ag[k]);
                au[k] = fmaf(suv, m8_group_dot(pu, xt), au[k]);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++) {
            ag[k] += M8_SHFL32(ag[k], off);
            au[k] += M8_SHFL32(au[k], off);
        }
    }
    if (l == 0) {
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++) {
            if (k < nt) {
                float gv = ag[k] * sh_s2g, uv = au[k] * sh_s2u;
                hidden[(size_t)(sh_base + k) * QF5_NFF + row] = gv / (1.f + expf(-gv)) * uv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// down projection + weighted accumulate. grid = (QF5_NEMBD/32, n_exp);
// block = 256 (32 rows * 8 lanes). Each expert's pairs land in y_partial with
// atomicAdd (a token may receive from several experts on this card).
// ---------------------------------------------------------------------------
__global__ void k_m8_downacc(const uint8_t *__restrict__ Wd, const uint8_t *__restrict__ Sd,
                             const float *__restrict__ s2d_res,
                             const float *__restrict__ hidden, float *__restrict__ y_partial,
                             const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
                             const int *__restrict__ pair_tok, const float *__restrict__ pair_wt,
                             const int *__restrict__ n_exp_dev) {
    if (blockIdx.y >= *n_exp_dev) return;      // device-decided extent
    __shared__ int   sh_slot, sh_base, sh_nt;
    __shared__ float sh_s2d;
    __shared__ int   sh_tok[QF5_M8_TX];
    __shared__ float sh_w[QF5_M8_TX];
    const int e = blockIdx.y;
    if (threadIdx.x == 0) {
        int sl = exp_slot[e];
        sh_slot = sl; sh_base = exp_ptr[e]; sh_nt = exp_ptr[e + 1] - exp_ptr[e]; sh_s2d = s2d_res[sl];
    }
    if (threadIdx.x < QF5_M8_TX) {
        int has = threadIdx.x < exp_ptr[e + 1] - exp_ptr[e];
        int idx = exp_ptr[e] + threadIdx.x;
        sh_tok[threadIdx.x] = has ? pair_tok[idx] : -1;
        sh_w[threadIdx.x]   = has ? pair_wt[idx] : 0.f;
    }
    __syncthreads();
    const int slot = sh_slot, nt = sh_nt;
    const int l   = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    const size_t woff = (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NFF / 2);
    const size_t soff = (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NFF / 16);
    const uint8_t *wr = Wd + woff;
    const uint8_t *sr = Sd + soff;

    float acc[QF5_M8_TX];
    #pragma unroll
    for (int k = 0; k < QF5_M8_TX; k++) acc[k] = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 8 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);   // weight group read ONCE
        float sdv = m8_ue4m3(sr[g]);
        const int hg = g * 16;
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++)
            if (k < nt) {
                const float *hk = hidden + (size_t)(sh_base + k) * QF5_NFF + hg;  // from global (L2)
                acc[k] = fmaf(sdv, m8_group_dot(pk, hk), acc[k]);
            }
    }
    #pragma unroll
    for (int off = 4; off; off >>= 1) {
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++) acc[k] += M8_SHFL8(acc[k], off);
    }
    if (l == 0) {
        #pragma unroll
        for (int k = 0; k < QF5_M8_TX; k++)
            if (k < nt)
                atomicAdd(&y_partial[(size_t)sh_tok[k] * QF5_NEMBD + row], sh_w[k] * acc[k] * sh_s2d);
    }
}

// Single-pass four-card combine: y_out = y0+y1+y2+y3. The three peer partials
// are copied into card0-local buffers concurrently (see qf_moe_4card.cpp), so
// this kernel just sums four local arrays - no card is a serialization funnel.
__global__ void k_m8_reduce4(float *__restrict__ y_out,
                             const float *__restrict__ y0, const float *__restrict__ y1,
                             const float *__restrict__ y2, const float *__restrict__ y3, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y_out[i] = y0[i] + y1[i] + y2[i] + y3[i];
}

extern "C" void qf5_m8_gateup(const uint8_t *Wg, const uint8_t *Sg,
                              const uint8_t *Wu, const uint8_t *Su,
                              const float *s2g_res, const float *s2u_res,
                              const float *x, float *hidden,
                              const Qf5M8Route *rt, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)Wg;(void)Sg;(void)Wu;(void)Su;(void)s2g_res;(void)s2u_res;(void)x;(void)hidden;(void)rt;(void)stream;
#else
    dim3 g((unsigned)(QF5_NFF / (QF5_GATEUP_THREADS / 32)), (unsigned)rt->n_exp_max);
    M8_LAUNCH(k_m8_gateup, g, QF5_GATEUP_THREADS, 0, stream,
              Wg, Sg, Wu, Su, s2g_res, s2u_res, x, hidden,
              rt->exp_slot, rt->exp_ptr, rt->pair_tok, rt->n_exp_dev);
#endif
}

extern "C" void qf5_m8_downacc(const uint8_t *Wd, const uint8_t *Sd, const float *s2d_res,
                               const float *hidden, float *y_partial,
                               const Qf5M8Route *rt, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)Wd;(void)Sd;(void)s2d_res;(void)hidden;(void)y_partial;(void)rt;(void)stream;
#else
    dim3 g((unsigned)(QF5_NEMBD / 32), (unsigned)rt->n_exp_max);
    M8_LAUNCH(k_m8_downacc, g, 256, 0, stream,
              Wd, Sd, s2d_res, hidden, y_partial,
              rt->exp_slot, rt->exp_ptr, rt->pair_tok, rt->pair_wt, rt->n_exp_dev);
#endif
}

extern "C" void qf5_m8_reduce4(float *y_out, const float *y0, const float *y1,
                               const float *y2, const float *y3, int n, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)y_out;(void)y0;(void)y1;(void)y2;(void)y3;(void)n;(void)stream;
#else
    int b = 256, g = (n + b - 1) / b;
    M8_LAUNCH(k_m8_reduce4, g, b, 0, stream, y_out, y0, y1, y2, y3, n);
#endif
}

// ---------------------------------------------------------------------------
// Device-side routing inversion (ported from GLM's k_moe_gather8_gfx906: the
// routing metadata is assembled ON DEVICE, the host never touches routing in
// the decode hot path). One block builds this card's expert->rows CSR from the
// batch's top-k selection:
//   exp_slot[j] = (il*EPC + local expert) of the j-th distinct owned expert
//   exp_ptr[j]  = CSR offset ; pair_tok/pair_wt = the rows selecting it
// Experts are emitted in ascending id so the order is a pure function of the
// routing (deterministic, like GLM's ascending-expert-id slot order).
// ---------------------------------------------------------------------------
__global__ void k_m8_build_csr(const int *__restrict__ sel, const float *__restrict__ wt,
                               int M, int K, int card, int epc, int il,
                               int *__restrict__ exp_slot, int *__restrict__ exp_ptr,
                               int *__restrict__ pair_tok, float *__restrict__ pair_wt,
                               int *__restrict__ n_exp_out, int *__restrict__ n_pair_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int lo = card * epc, hi = lo + epc;
    int nexp = 0, np = 0;
    exp_ptr[0] = 0;
    // ascending expert id; linear scan is exact for M*K <= 80 pairs
    int prev = -1;
    for (;;) {
        int best = 0x7FFFFFFF;
        for (int t = 0; t < M; t++)
            for (int k = 0; k < K; k++) {
                int e = sel[t * K + k];
                if (e < lo || e >= hi) continue;
                if (e > prev && e < best) best = e;
            }
        if (best == 0x7FFFFFFF) break;
        for (int t = 0; t < M; t++)
            for (int k = 0; k < K; k++)
                if (sel[t * K + k] == best) { pair_tok[np] = t; pair_wt[np] = wt[t * K + k]; np++; }
        exp_slot[nexp] = il * epc + (best - lo);
        nexp++; exp_ptr[nexp] = np;
        prev = best;
    }
    *n_exp_out = nexp; *n_pair_out = np;
}

extern "C" void qf5_m8_build_csr(const int *sel_dev, const float *wt_dev, int M, int K,
                                 int card, int epc, int il, int *exp_slot, int *exp_ptr,
                                 int *pair_tok, float *pair_wt, int *counts, hipStream_t s) {
#ifdef QF_HOST_CHECK
    (void)sel_dev;(void)wt_dev;(void)M;(void)K;(void)card;(void)epc;(void)il;(void)exp_slot;
    (void)exp_ptr;(void)pair_tok;(void)pair_wt;(void)counts;(void)s;
#else
    M8_LAUNCH(k_m8_build_csr, 1, 64, 0, s, sel_dev, wt_dev, M, K, card, epc, il,
              exp_slot, exp_ptr, pair_tok, pair_wt, counts, counts + 1);
#endif
}
