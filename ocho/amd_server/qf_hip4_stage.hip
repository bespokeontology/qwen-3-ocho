// qf_hip4_stage.hip - one 12-layer pipeline stage on one gfx906 (MI50) GPU.
//
// Wave-4 session 09 (persistent server engine core):
//   * Device-resident MoE router (qf_router_wave64.h): qf_rtr_gemv /
//     qf_rtr_topk10 / qf_rtr_dispatch replace the host router (logits D2H +
//     host top-10 + host LRU). Misses are signaled through the per-layer
//     pinned QfRouteMailbox (zero-copy, written on device) and serviced by
//     qf4_expcache_service; the expert loop uses the device-indexed fused
//     kernels qf5_nvfp4_gateup_640x2560_d / qf5_nvfp4_downacc_2560x640_d.
//   * Init-time HBM budget guard: the full per-GPU byte commitment (dense
//     weights + expert cache + GDN state + KV at QF_MAX_CONTEXT) is computed
//     BEFORE any hipMalloc and aborts init if over QF4_HBM_BUDGET_MIB.
//   * qf4_stage_run_nologits (prefill: skips output-HC + lm_head),
//     qf4_stage_reset (zeroes GDN S / conv rings, resets KV occupancy + PLE
//     history), and P2P handoff buffers/events (qf4_stage_emit_residual /
//     qf4_stage_recv_residual_dev / qf4_peer_available).
//
// Wave-2 session 05 merge (base): integrates the wave1/05_amd_wave64 concrete
// kernels and fixed-slot expert cache into the prior/09_amd_hip stage.
//   * The routed-expert hot path no longer touches whole-layer expert blobs
//     and has NO host-mapped zero-copy fallback: each layer owns a
//     Qf4ExpCache (fixed HBM slots, bounded LRU, QF_EXPERT_CACHE_SLOTS env,
//     default 192) and each GPU owns one Qf4ExpXfer (pinned double-buffer +
//     transfer stream + events). A miss gathers the six tensors out of the
//     mmap'd store into pinned staging and hands them to the transfer
//     stream; the compute stream only waits on an event.
//   * The per-expert loop calls the fused wave64 kernels
//     qf5_nvfp4_gateup_640x2560 (gate+up+silu-mul, one x staging, one
//     launch) and qf5_nvfp4_downacc_2560x640 (down + weighted accumulate,
//     one writer per row, no atomics). k_axpy and the routed-path k_silu_mul
//     / ed scratch are gone; the old generic k_nvfp4_gemv is deleted.
//   * All 12 layers of the stage use the same path (no per-layer special
//     cases).
//
// Corrected relative to the v1 CUDA/HIP reference (qf.cu / qf_hip.cu), as in
// prior/09_amd_hip:
//   * conv1d weights read as BF16 (checkpoint dtype), not f32.
//   * GDN state decay is exp(g): S *= __expf(g) (v1 multiplied by raw g).
//   * Router: softmax over 512 logits in fp32, top-10, weights are the
//     softmax probs and are NOT renormalized (norm_topk_prob = false).
//     v1 ranked and weighted by raw logits.
//   * Attention out-gate uses the [24][512] q_proj layout with the gate in
//     the upper 256 of each head row (v1 indexed it as packed [24][256]).
//   * q_norm / k_norm (zero-centered bf16, per 256-dim head) are applied
//     before rope (v1 dropped them).
//   * Every cross-lane reduction is wave64-safe: full 64-lane mask,
//     offsets 32,16,8,4,2,1; cross-wave combines via shared memory.
//
// Build: see Makefile.hip4 (hipcc, --offload-arch=gfx906, ROCm 5.7).
// Host syntax check without a GPU toolchain:
//   g++ -DQF_HOST_CHECK -I. -I$(ENGINESRC) -fsyntax-only -x c++ qf_hip4_stage.hip
#include "qf_hip4.h"
#include "qf_nvfp4_wave64.h"
#include "qf_router_wave64.h"
#include "qf8_dense.h"
#include "qf8_layer_m.h"
#include "qf_moe_m8_wave64.h"
#include "qf8_moe_fp8.h"
#include "qf8_moe_i8.h"
#include "qf8_dense_i8.h"

#ifdef QF_HOST_CHECK
#include "host_check_shim.h"
#else
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#define __shfl_sync(mask, value, lane)        __shfl((value), (lane), warpSize)
#endif
#endif

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------- dims (from SEMANTICS.md / qwenflash.h, baked for v1) -----
#define NEMBD   2560
#define NHEAD   24
#define NKV     2
#define HDIM    256
#define NLAYER  48
#define NEXP    512
#define NEXPUSED 10
#define NFF     640
#define HCC     4
#define HCL     320
#define GDN_KH  16
#define GDN_VH  48
#define GDN_KD  128
#define GDN_VD  128
#define DINN    10240
#define DTRANK  48
#define NVOCAB  248320
#define KVREP   (NHEAD / NKV)
#define QGATE   512     // q_proj row width per head: 256 q + 256 gate
#define RSIZE   QF4_RSIZE
#define SL      QF4_STAGE_NL
#define QF4_EOS 248044  // PLE history seed (SEMANTICS.md: eos 248044)

#define WAVE    64
#define WMASK   0xFFFFFFFFFFFFFFFFULL   // full 64-lane wave

#ifdef QF_HOST_CHECK
#define QF4_LAUNCH(k, g, b, sm, s, ...) k(__VA_ARGS__)
#else
#define QF4_LAUNCH(k, g, b, sm, s, ...) k<<<g, b, sm, s>>>(__VA_ARGS__)
#endif

// bf16 storage as raw u16; gfx906 has no bf16 hardware, convert by shifting
// into the high 16 bits of an f32 (exact) and round-to-nearest-even back.
__device__ __forceinline__ float bf2f(uint16_t v) {
    return __uint_as_float(((uint32_t)v) << 16);
}
__device__ __forceinline__ uint16_t f2bf(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }
// Full-wave sum reduction. Caller must be a full, non-divergent wave.
__device__ __forceinline__ float wave_sum(float v) {
    #pragma unroll
    for (int off = WAVE / 2; off; off >>= 1)
        v += __shfl_down_sync(WMASK, v, off);
    return v;   // valid in lane 0
}

// ---------------- kernels --------------------------------------------------
__global__ void k_cvt_bf16_f32(float *out, const uint16_t *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf2f(in[i]);
}
// Row-wise greedy argmax over [M][NVOCAB] logits: one block per row, ties
// resolve to the LOWEST index (the host loop's `>` comparison), so the emitted
// token stream is unchanged. Replaces an M x NVOCAB float D2H + host scan on
// the alternating map's exposed head path with a 4 B/row reply.
__global__ void k_argmax_rows(const float *__restrict__ logits, int *__restrict__ ids, int n) {
    const float *row = logits + (size_t)blockIdx.x * n;
    float best = -3.0e38f; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    __shared__ float sv[256];
    __shared__ int   si[256];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if ((int)threadIdx.x < off) {
            float ov = sv[threadIdx.x + off]; int oi = si[threadIdx.x + off];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov; si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[blockIdx.x] = si[0];
}
__global__ void k_repeat4(float *R, const float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < RSIZE) R[i] = x[i % NEMBD];
}
__global__ void k_zero(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}
__global__ void k_silu_mul(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}
__global__ void k_shexp_add(float *y, const float *sh, const float *g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NEMBD) y[i] += sigmoidf_(g[0]) * sh[i];
}
__global__ void k_stream_inject(float *R, const float *y, const float *inj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < RSIZE) R[i] += inj[i / NEMBD] * y[i % NEMBD];
}
// One row per wave: blockDim.x = 64, blockDim.y = 4 rows per block.
__global__ void k_gemv_bf16(const uint16_t *__restrict__ W, const float *__restrict__ x,
                            float *__restrict__ y, int out, int in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out) return;
    const uint16_t *wr = W + (size_t)row * in;
    float acc = 0.f;
    for (int i = threadIdx.x; i < in; i += WAVE)
        acc += bf2f(wr[i]) * x[i];
    acc = wave_sum(acc);
    if (threadIdx.x == 0) y[row] = acc;
}
// Zero-centered grouped RMSNorm: out[i] = x[i]*rsqrt(mean_g(x^2)+eps)*(1+w[i]).
// One block per group (grid.x = n/group), blockDim.x = 256 (4 waves).
__global__ void k_rmsnorm_group(float *out, const float *x, const uint16_t *w,
                                int group, float eps) {
    int g0 = blockIdx.x * group;
    float acc = 0.f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        float v = x[g0 + i]; acc += v * v;
    }
    acc = wave_sum(acc);
    __shared__ float red[16];
    __shared__ float rstd;
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < nw; i++) t += red[i];
        rstd = rsqrtf(t / group + eps);
    }
    __syncthreads();
    float r = rstd;
    for (int i = threadIdx.x; i < group; i += blockDim.x)
        out[g0 + i] = x[g0 + i] * r * (1.f + bf2f(w[g0 + i]));
}
// HC down-projection: d320[r] = silu(dot(w_down[r], n) / 4). One wave per row.
__global__ void k_hc_down(const float *n, const uint16_t *wd, float *d320) {
    int row = blockIdx.x, lane = threadIdx.x;
    const uint16_t *wr = wd + (size_t)row * RSIZE;
    float acc = 0.f;
    for (int i = lane; i < RSIZE; i += WAVE) acc += bf2f(wr[i]) * n[i];
    acc = wave_sum(acc);
    if (lane == 0) { float v = acc / 4.f; d320[row] = v / (1.f + expf(-v)); }
}
// HC up-projection + per-stream mean: mixed[c*2560+j] += m_i * n_i / 4.
__global__ void k_hc_up(const float *n, const uint16_t *wu, const float *d, float *mixed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RSIZE) return;
    const uint16_t *wr = wu + (size_t)i * HCL;
    float acc = 0.f;
    for (int l = 0; l < HCL; l++) acc += bf2f(wr[l]) * d[l];
    float m = sigmoidf_(acc);
    atomicAdd(&mixed[i % NEMBD], m * n[i] / 4.f);
}
// HC block inject: inj[c] = 2*sigmoid(dot(w_inject[c], n)/4).
// One block of 256 threads = 4 waves, wave c handles stream c.
__global__ void k_hc_inject(const float *n, const uint16_t *wi, float *inj) {
    int c = threadIdx.x / warpSize, lane = threadIdx.x % warpSize;
    const uint16_t *wr = wi + (size_t)c * RSIZE;
    float acc = 0.f;
    for (int i = lane; i < RSIZE; i += warpSize) acc += bf2f(wr[i]) * n[i];
    acc = wave_sum(acc);
    if (lane == 0) inj[c] = 2.f * sigmoidf_(acc / 4.f);
}
// GDN causal conv1d step (kernel 4) + silu. Weights are BF16 [10240,4]
// (checkpoint [10240,1,4] squeezed); tap 0 = oldest of the 3-deep ring.
__global__ void k_conv_step(float *raw, float *ring, const uint16_t *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DINN) return;
    float cur = raw[i];
    float o = bf2f(w[i * 4 + 0]) * ring[0 * DINN + i]
            + bf2f(w[i * 4 + 1]) * ring[1 * DINN + i]
            + bf2f(w[i * 4 + 2]) * ring[2 * DINN + i]
            + bf2f(w[i * 4 + 3]) * cur;
    ring[0 * DINN + i] = ring[1 * DINN + i];
    ring[1 * DINN + i] = ring[2 * DINN + i];
    ring[2 * DINN + i] = cur;
    raw[i] = o / (1.f + expf(-o));
}
// GDN decode recurrence, one 64-thread wave per v-head (48 blocks).
// Each lane covers dims tid and tid+64 so all reductions stay in-wave.
// Corrected decay: S *= exp(g) with g = -exp(A_log)*softplus(a+dt_bias).
__global__ void k_gdn_decode(const float *qkv, const float *a_in, const float *b_in,
                             const float *A_log, const float *dt_bias,
                             float *S, float *out48) {
    int h = blockIdx.x, kh = h / 3, tid = threadIdx.x;
    float g = -__expf(A_log[h]) * log1pf(expf(a_in[h] + dt_bias[h]));
    float decay = __expf(g);                    // <-- fix: v1 used raw g here
    float beta = sigmoidf_(b_in[h]);
    __shared__ float ks[GDN_KD], qs[GDN_KD];
    float q0 = qkv[kh * GDN_KD + tid],          q1 = qkv[kh * GDN_KD + tid + 64];
    float k0 = qkv[2048 + kh * GDN_KD + tid],   k1 = qkv[2048 + kh * GDN_KD + tid + 64];
    float v0 = qkv[4096 + h * GDN_VD + tid],    v1 = qkv[4096 + h * GDN_VD + tid + 64];
    float q2 = q0 * q0 + q1 * q1, k2 = k0 * k0 + k1 * k1;
    q2 = wave_sum(q2);
    k2 = wave_sum(k2);
    q2 = __shfl_sync(WMASK, q2, 0);
    k2 = __shfl_sync(WMASK, k2, 0);
    float rq = rsqrtf(q2 + 1e-6f), rk = rsqrtf(k2 + 1e-6f);
    q0 *= rq; q1 *= rq; k0 *= rk; k1 *= rk;
    ks[tid] = k0; ks[tid + 64] = k1;
    qs[tid]      = q0 * (1.f / sqrtf((float)GDN_KD));
    qs[tid + 64] = q1 * (1.f / sqrtf((float)GDN_KD));
    __syncthreads();
    float *Sh = S + (size_t)h * GDN_KD * GDN_VD;
    for (int r = tid; r < GDN_KD; r += WAVE) {
        float *row = Sh + (size_t)r * GDN_VD;
        for (int col = 0; col < GDN_VD; col++) row[col] *= decay;
    }
    __syncthreads();
    #pragma unroll
    for (int c = 0; c < GDN_VD / WAVE; c++) {
        int col = tid + c * WAVE;
        float v = c ? v1 : v0;
        float kv = 0.f;
        for (int r = 0; r < GDN_KD; r++) kv += Sh[r * GDN_VD + col] * ks[r];
        float d = (v - kv) * beta;
        for (int r = 0; r < GDN_KD; r++) Sh[r * GDN_VD + col] += ks[r] * d;
    }
    __syncthreads();
    #pragma unroll
    for (int c = 0; c < GDN_VD / WAVE; c++) {
        int col = tid + c * WAVE;
        float o = 0.f;
        for (int r = 0; r < GDN_KD; r++) o += Sh[r * GDN_VD + col] * qs[r];
        out48[h * GDN_VD + col] = o;
    }
}
// RMSNormGated (one-based weight), one wave per head: out = w * rms(x) * silu(z).
__global__ void k_rmsnorm_gated(uint16_t *out, const float *x, const uint16_t *w,
                                const float *z, float eps) {
    int h = blockIdx.x, i = threadIdx.x;
    float v0 = x[h * GDN_VD + i], v1 = x[h * GDN_VD + i + WAVE];
    float v2 = wave_sum(v0 * v0 + v1 * v1);
    __shared__ float var;
    if (i == 0) var = v2 / GDN_VD;
    __syncthreads();
    float rs = rsqrtf(var + eps);
    float g0 = z[h * GDN_VD + i], g1 = z[h * GDN_VD + i + WAVE];
    out[h * GDN_VD + i]        = f2bf(bf2f(w[i])        * (v0 * rs) * (g0 / (1.f + expf(-g0))));
    out[h * GDN_VD + i + WAVE] = f2bf(bf2f(w[i + WAVE]) * (v1 * rs) * (g1 / (1.f + expf(-g1))));
}
// Zero-centered per-head RMSNorm over HDIM=256 (q_norm/k_norm), one wave per
// head, 4 dims per lane. Applied in place before rope.
__global__ void k_qk_norm(float *v, const uint16_t *w, int stride, float eps) {
    int h = blockIdx.x, lane = threadIdx.x;
    float *base = v + (size_t)h * stride;
    float x0 = base[lane], x1 = base[lane + 64], x2 = base[lane + 128], x3 = base[lane + 192];
    float acc = wave_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
    acc = __shfl_sync(WMASK, acc, 0);
    float rs = rsqrtf(acc / HDIM + eps);
    base[lane]       = x0 * rs * (1.f + bf2f(w[lane]));
    base[lane + 64]  = x1 * rs * (1.f + bf2f(w[lane + 64]));
    base[lane + 128] = x2 * rs * (1.f + bf2f(w[lane + 128]));
    base[lane + 192] = x3 * rs * (1.f + bf2f(w[lane + 192]));
}
__global__ void k_rope(float *q, float *k, const float *inv_freq, long pos) {
    // grid (32, NHEAD+NKV): rotary on the first 64 dims (partial 0.25 of 256).
    int half = blockIdx.x, r = blockIdx.y;
    float *dst = (r < NHEAD) ? q + (size_t)r * QGATE : k + (size_t)(r - NHEAD) * HDIM;
    float a = dst[half], b = dst[half + 32];
    float f = pos * inv_freq[half];
    dst[half]      = a * cosf(f) - b * sinf(f);
    dst[half + 32] = b * cosf(f) + a * sinf(f);
}
__global__ void k_cache_write(uint16_t *cache, const float *kv, long pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NKV * HDIM) cache[(size_t)pos * (NKV * HDIM) + i] = f2bf(kv[i]);
}
__global__ void k_scores(const float *q, const uint16_t *K, float *scores,
                         long pos, float scale, int kh) {
    long p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pos) return;
    float acc = 0.f;
    for (int d = 0; d < HDIM; d++)
        acc += q[d] * bf2f(K[(size_t)p * (NKV * HDIM) + kh * HDIM + d]);
    scores[p] = acc * scale;
}
__global__ void k_softmax(float *scores, long pos) {
    __shared__ float mx, sum;
    __shared__ float red[16];           // up to 16 waves per block
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    float local = -1e30f;
    for (long p = threadIdx.x; p < pos; p += blockDim.x) local = fmaxf(local, scores[p]);
    #pragma unroll
    for (int off = WAVE / 2; off; off >>= 1)
        local = fmaxf(local, __shfl_down_sync(WMASK, local, off));
    if (lane == 0) red[wid] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m2 = -1e30f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; w++) m2 = fmaxf(m2, red[w]);
        mx = m2; sum = 0.f;
    }
    __syncthreads();
    float e = 0.f;
    for (long p = threadIdx.x; p < pos; p += blockDim.x) { e = expf(scores[p] - mx); scores[p] = e; }
    e = wave_sum(e);
    if (lane == 0) atomicAdd(&sum, e);
    __syncthreads();
    for (long p = threadIdx.x; p < pos; p += blockDim.x) scores[p] /= (sum + 1e-9f);
}
__global__ void k_attn_out(const float *scores, const uint16_t *V, float *out,
                           long pos, int kh) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HDIM) return;
    float acc = 0.f;
    for (long p = 0; p < pos; p++)
        acc += scores[p] * bf2f(V[(size_t)p * (NKV * HDIM) + kh * HDIM + d]);
    out[d] = acc;
}
// Corrected gate indexing: gate for head h sits at q_proj[h*QGATE + 256 .. +512).
__global__ void k_apply_gate(float *out, const float *qproj) {
    int h = blockIdx.x, i = threadIdx.x;
    out[h * HDIM + i] *= sigmoidf_(qproj[(size_t)h * QGATE + HDIM + i]);
}

// ---------------- host side ------------------------------------------------
#define SETDEV(st) do { if (hipSetDevice((st)->gpu) != hipSuccess) return -1; } while (0)

typedef struct {
    // hyper-connections (all bf16 on device)
    uint16_t *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    uint16_t *hc_ffn_norm,  *hc_ffn_down,  *hc_ffn_up,  *hc_ffn_inject;
    // GDN
    uint16_t *qkv, *zgate, *conv1d;      // conv1d is BF16 [10240,4] (fix)
    float    *dt_bias, *a_log;           // f32 [48]
    uint16_t *in_a, *in_b;               // [48,2560]
    uint16_t *gdn_norm;                  // [128] one-based
    uint16_t *gdn_out;                   // [2560,6144]
    // full attention
    uint16_t *wq, *wk, *wv, *wo;
    uint16_t *q_norm, *k_norm;           // [256] zero-centered
    uint16_t *idx_qk, *idx_qnorm, *idx_knorm;
    // MoE
    uint16_t *router;                    // [512,2560]
    // Routed experts: fixed-slot HBM cache (session 05). Replaces the six
    // whole-layer expert blobs + per-expert s2 tables; slot-indexed kernels
    // read straight out of cache slots, misses go through the GPU's pinned
    // staging double-buffer. No host-mapped zero-copy path exists anymore.
    Qf4ExpCache cache;
    uint16_t *shexp_gate, *shexp_up, *shexp_down, *shexp_gate_inp;
    // M=1 int8 dense copies (QF_M1_INT8=1, qf8_dense_i8.h), built at load.
    Qf8I8Mat i8_qkv, i8_zgate, i8_ina, i8_inb, i8_gdn_out, i8_router, i8_shg, i8_shu, i8_shgi, i8_shd;
    Qf8I8Mat i8_hca_down, i8_hca_inj, i8_hca_up, i8_hcf_down, i8_hcf_inj, i8_hcf_up;
    Qf8I8Mat i8_wq, i8_wk, i8_wv, i8_idxqk, i8_wo;
    int i8_dense;
} Qf4Layer;

struct Qf4Stage {
    int gpu, l0, nl;
    int is_first, is_last;
    int owns_embed;                     // this stage embeds the token (l0 == 0)
    // is_last  = last stage ON THIS BOX (it hands the residual onward)
    // owns_head = this box owns the model's FINAL layer, so it also runs the
    //             output HC mixer + lm_head. Under a split these differ: the
    //             AMD box's last stage is not the model's tail.
    int owns_head;
    const QfStore *st;                  // mmap'd store (expert misses read it)
    Qf4ExpXfer xfer;                    // per-GPU pinned staging + xfer stream
    Qf4Layer L[SL];
    // activations / scratch (device)
    float *x, *R, *normed, *hc_d, *mixed, *inj, *y2560, *qkv_raw, *z6144;
    float *q12288, *k512, *v512, *a48, *b48;
    float *attn_out, *scores, *router;
    float *eg, *eu, *ed;                // eg: fused gateup hidden / shexp gate
        float *out48, *logits;
    int   *ids_dev;                     // [MR] device argmax of logits (head stage)
    uint16_t *gdn_out_bf;
    float *inv_freq;
    float *inv_freq_idx, *idx640, *attn_out8;
    uint16_t *x_bf16;
    Qf8QsaIdx idx[SL * QF8_MROWS];   // one indexer per (layer, row)
    Qf8Ple *ple;
    int    ple_slot;                    // PLE chunk staging slot of the chunk being enqueued (stage 0)
    Qf8Prof prof;
    FILE *tapf;                         // correctness taps, per stage (see tap_vec)
    unsigned int *exp_hist;             // [NEXP] selection counts, all layers
    int  exp_slots;                     // expert cache slots per layer (auto-sized)
    // device-resident router outputs (session 09): selected expert ids and
    // their full-softmax weights for the current token, written by
    // qf_rtr_topk10 and consumed by qf_rtr_dispatch + the *_d expert kernels.
    int   *sel_ids_dev;                 // [NEXPUSED]
    float *wts_dev;                     // [NEXPUSED]
    // recurrent state (device), indexed by local layer; only one kind per layer
    float    *gdnS[SL], *convring[SL];
    qf8_kv_t *kc[SL], *vc[SL];
    int   *m8_exp_slot, *m8_exp_ptr, *m8_pair_tok, *m8_counts;   // device CSR
    float *m8_pair_wt;
    float *m8_part;                     // owned down partials [MR*10][NEMBD] (chunk experts)
    int8_t *c8q6; float *c8s6;          // X8 of [MR][6144] (gdn out-proj input, chunk)
    int8_t *cn8q; float *cn8s;          // X8 of normed [MR][RSIZE] (chunk hc)
    int8_t *cd8q; float *cd8s;          // X8 of hc_d [MR][HCL] (chunk hc)
    float *c_hcy;                       // hc up output [MR][RSIZE] (chunk hc)
    // gfx906 production FP8 routed path (QF8_MOE_FP8=1): device expert
    // descriptors, the per-stage control block (count + work cursors), and the
    // [desc][row][NFF] hidden buffer. All NULL on the NVFP4 path.
    Qf8MoeDesc *m8_desc; int *m8_ctl; float *m8_hidden;
    int8_t *x8q, *h8q; float *x8s, *h8s;   // M=1 int8 expert path scratch
    int8_t *n8q, *d8q; float *n8s, *d8s;   // M=1 int8 hyper-connection scratch
    int8_t *c8q, *ch8q; float *c8s, *ch8s;  // chunk (prefill) X8 scratch: T rows / pairs
    Qf8I8Mat i8_lm_head, i8_ohc_down, i8_ohc_up; int i8_head;   // M=1 int8 head
    float *attn_part;                                              // split flash partials
    // chunk2 QSA row scratch (prefill): [MR][nb_max] scores/masks, [MR][TOPK+1] lists,
    // [MR] counts, [MR][24][8][258] flash partials
    float *c_blk_score; uint8_t *c_blk_mask; int *c_blk_list, *c_nsel; float *c_part; int c_nb_max;
    int    exp_fp8;                     // 1 = expert caches hold the FP8 format
    // first/last stage extras (device)
    uint16_t *tok_embd;                 // NULL when the table stays on the host (below)
    const uint16_t *tok_embd_host;      // bf16 [NVOCAB][NEMBD] in the mmap'd store (page cache)
    uint16_t *emb_pin, *emb_dev;        // [MR][NEMBD] bf16 staging: host gather -> one H2D (M-row / single-token paths)
    // chunk intake: QF8_PLE_SLOTS pinned + device stagings and events on a transfer stream, so the
    // gather + H2D of chunk c+1 never waits for the compute stream (which holds chunk c's whole work)
    uint16_t *cemb_pin[QF8_PLE_SLOTS], *cemb_dev[QF8_PLE_SLOTS]; hipEvent_t cemb_ev[QF8_PLE_SLOTS]; hipStream_t cemb_xfer; int cemb_slot;
    uint16_t *lm_head, *ohc_norm, *ohc_down, *ohc_up;
    // P2P handoff (session 09): two fixed 40 KiB device buffers + events.
    // slot = pos % 2 pairs producer buffer/event with the consumer's wait.
    float     *handoff[2];
    hipEvent_t hev[2];
    // bookkeeping
    long   maxpos;                      // QF_MAX_CONTEXT in effect (clamped)
    long   kv_len;                      // KV occupancy bookkeeping (pos+1 last run)
    int    ple_hist[2];                 // PLE ngram history (ngram-1), EOS-seeded
    size_t hbm_bytes;                   // committed bytes (init budget guard)
    hipStream_t stream;
};

static void stage_err(const char *what, const char *name, hipError_t e) {
    fprintf(stderr, "qf4 stage: %s %s: %s\n", what, name ? name : "", hipGetErrorString(e));
}

// Load a tensor as bf16 on device. Accepts BF16 (2 B/elt) or F32 (4 B/elt,
// converted host-side, round-to-nearest-even) in the store.
static uint16_t *load_bf16(const QfStore *st, const char *name, size_t elems) {
    const QfEntry *e = qf_find(st, name);
    if (!e) { fprintf(stderr, "qf4 MISSING %s\n", name); return NULL; }
    const char *src = (const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off;
    uint16_t *dev = NULL;
    if (e->rec.nbytes == elems * 2) {
        if (hipMalloc((void **)&dev, e->rec.nbytes) != hipSuccess ||
            hipMemcpy(dev, src, e->rec.nbytes, hipMemcpyHostToDevice) != hipSuccess) {
            stage_err("upload failed for", name, hipErrorUnknown); return NULL;
        }
        return dev;
    }
    if (e->rec.nbytes == elems * 4) {
        uint16_t *h = (uint16_t *)malloc(elems * 2);
        if (!h) return NULL;
        const uint32_t *f = (const uint32_t *)src;
        for (size_t i = 0; i < elems; i++) {
            uint32_t u = f[i];
            u += 0x7FFFu + ((u >> 16) & 1u);
            h[i] = (uint16_t)(u >> 16);
        }
        if (hipMalloc((void **)&dev, elems * 2) != hipSuccess ||
            hipMemcpy(dev, h, elems * 2, hipMemcpyHostToDevice) != hipSuccess) {
            free(h); stage_err("upload failed for", name, hipErrorUnknown); return NULL;
        }
        free(h);
        return dev;
    }
    fprintf(stderr, "qf4 %s: unexpected size %llu (want %zu or %zu)\n", name,
            (unsigned long long)e->rec.nbytes, elems * 2, elems * 4);
    return NULL;
}

// Load a tensor as f32 on device (accepts F32 raw, or BF16 converted up).
static float *load_f32(const QfStore *st, const char *name, size_t elems) {
    const QfEntry *e = qf_find(st, name);
    if (!e) { fprintf(stderr, "qf4 MISSING %s\n", name); return NULL; }
    const char *src = (const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off;
    float *dev = NULL;
    if (e->rec.nbytes == elems * 4) {
        if (hipMalloc((void **)&dev, e->rec.nbytes) != hipSuccess ||
            hipMemcpy(dev, src, e->rec.nbytes, hipMemcpyHostToDevice) != hipSuccess) {
            stage_err("upload failed for", name, hipErrorUnknown); return NULL;
        }
        return dev;
    }
    if (e->rec.nbytes == elems * 2) {
        float *h = (float *)malloc(elems * 4);
        if (!h) return NULL;
        const uint16_t *b = (const uint16_t *)src;
        for (size_t i = 0; i < elems; i++) {
            uint32_t u = ((uint32_t)b[i]) << 16;
            float v; memcpy(&v, &u, 4);
            h[i] = v;
        }
        if (hipMalloc((void **)&dev, elems * 4) != hipSuccess ||
            hipMemcpy(dev, h, elems * 4, hipMemcpyHostToDevice) != hipSuccess) {
            free(h); stage_err("upload failed for", name, hipErrorUnknown); return NULL;
        }
        free(h);
        return dev;
    }
    fprintf(stderr, "qf4 %s: unexpected size %llu\n", name, (unsigned long long)e->rec.nbytes);
    return NULL;
}

#define LBF16(field, fmt, elems) do { \
    snprintf(nm, sizeof(nm), fmt, il); \
    L->field = load_bf16(st, nm, elems); \
    if (!L->field) return -1; } while (0)
#define LF32(field, fmt, elems) do { \
    snprintf(nm, sizeof(nm), fmt, il); \
    L->field = load_f32(st, nm, elems); \
    if (!L->field) return -1; } while (0)

static int load_layer_real(Qf4Stage *S, Qf4Layer *L, int il, const QfStore *st) {
    char nm[QF_MAX_NAME];
    LBF16(hc_attn_norm,   "model.language_model.layers.%d.attn_hyper_connection.hc_norm.weight", RSIZE);
    LBF16(hc_attn_down,   "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_down.weight", (size_t)HCL * RSIZE);
    LBF16(hc_attn_up,     "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_up.weight", (size_t)RSIZE * HCL);
    LBF16(hc_attn_inject, "model.language_model.layers.%d.attn_hyper_connection.block_inject_weight.weight", (size_t)HCC * RSIZE);
    LBF16(hc_ffn_norm,    "model.language_model.layers.%d.mlp_hyper_connection.hc_norm.weight", RSIZE);
    LBF16(hc_ffn_down,    "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight", (size_t)HCL * RSIZE);
    LBF16(hc_ffn_up,      "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight", (size_t)RSIZE * HCL);
    LBF16(hc_ffn_inject,  "model.language_model.layers.%d.mlp_hyper_connection.block_inject_weight.weight", (size_t)HCC * RSIZE);

    int is_recr = ((il + 1) % 4) != 0;
    if (is_recr) {
        LBF16(qkv,      "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", (size_t)DINN * NEMBD);
        LBF16(zgate,    "model.language_model.layers.%d.linear_attn.in_proj_z.weight", (size_t)6144 * NEMBD);
        LBF16(conv1d,   "model.language_model.layers.%d.linear_attn.conv1d.weight", (size_t)DINN * 4);
        LF32 (dt_bias,  "model.language_model.layers.%d.linear_attn.dt_bias", DTRANK);
        LF32 (a_log,    "model.language_model.layers.%d.linear_attn.A_log", DTRANK);
        LBF16(in_a,     "model.language_model.layers.%d.linear_attn.in_proj_a.weight", (size_t)DTRANK * NEMBD);
        LBF16(in_b,     "model.language_model.layers.%d.linear_attn.in_proj_b.weight", (size_t)DTRANK * NEMBD);
        LBF16(gdn_norm, "model.language_model.layers.%d.linear_attn.norm.weight", GDN_VD);
        LBF16(gdn_out,  "model.language_model.layers.%d.linear_attn.out_proj.weight", (size_t)NEMBD * 6144);
    } else {
        LBF16(wq,     "model.language_model.layers.%d.self_attn.q_proj.weight", (size_t)12288 * NEMBD);
        LBF16(wk,     "model.language_model.layers.%d.self_attn.k_proj.weight", (size_t)512 * NEMBD);
        LBF16(wv,     "model.language_model.layers.%d.self_attn.v_proj.weight", (size_t)512 * NEMBD);
        LBF16(wo,     "model.language_model.layers.%d.self_attn.o_proj.weight", (size_t)NEMBD * 6144);
        LBF16(q_norm, "model.language_model.layers.%d.self_attn.q_norm.weight", HDIM);
        LBF16(k_norm, "model.language_model.layers.%d.self_attn.k_norm.weight", HDIM);
        LBF16(idx_qk,    "model.language_model.layers.%d.self_attn.indexer.index_qk_proj.weight", (size_t)640 * NEMBD);
        LBF16(idx_qnorm, "model.language_model.layers.%d.self_attn.indexer.q_layernorm.weight", 128);
        LBF16(idx_knorm, "model.language_model.layers.%d.self_attn.indexer.k_layernorm.weight", 128);
    }

    LBF16(router, "model.language_model.layers.%d.mlp.gate.weight", (size_t)NEXP * NEMBD);
    // Routed experts: fixed-slot HBM cache instead of the six whole-layer
    // blobs. Slot blobs are hipMalloc'd HBM; per-expert scale_2 tables load
    // host-side here; packed weights/scales land in slots on first use.
    // slots <= 0 -> QF_EXPERT_CACHE_SLOTS env, default 192.
    if (qf4_expcache_init(&L->cache, st, il, S->exp_slots) != 0) {
        fprintf(stderr, "qf4: expert cache init failed for layer %d\n", il);
        return -1;
    }

    LBF16(shexp_gate,     "model.language_model.layers.%d.mlp.shared_expert.gate_proj.weight", (size_t)NFF * NEMBD);
    LBF16(shexp_up,       "model.language_model.layers.%d.mlp.shared_expert.up_proj.weight", (size_t)NFF * NEMBD);
    LBF16(shexp_down,     "model.language_model.layers.%d.mlp.shared_expert.down_proj.weight", (size_t)NEMBD * NFF);
    LBF16(shexp_gate_inp, "model.language_model.layers.%d.mlp.shared_expert_gate.weight", (size_t)NEMBD);
    L->i8_dense = 0;
    if (getenv("QF_M1_INT8") && getenv("QF_M1_INT8")[0] == '1') {
        // M=1 int8 dense copies: per-row absmax int8 of the resident bf16
        // tensors, built once here (init-time, null stream, then synced).
#define Q8(dst, src, rows, K) do { if (qf8_i8_quant_bf16(L->src, (rows), (K), &L->dst, 0) != 0) { \
            fprintf(stderr, "qf4: int8 quant failed L%d %s\n", il, #src); return -1; } } while (0)
        if (is_recr) {
            Q8(i8_qkv, qkv, DINN, NEMBD); Q8(i8_zgate, zgate, 6144, NEMBD);
            Q8(i8_ina, in_a, DTRANK, NEMBD); Q8(i8_inb, in_b, DTRANK, NEMBD);
            Q8(i8_gdn_out, gdn_out, NEMBD, 6144);
        } else {
            Q8(i8_wq, wq, 12288, NEMBD); Q8(i8_wk, wk, 512, NEMBD); Q8(i8_wv, wv, 512, NEMBD);
            Q8(i8_idxqk, idx_qk, 640, NEMBD); Q8(i8_wo, wo, NEMBD, 6144);
        }
        Q8(i8_router, router, NEXP, NEMBD);
        Q8(i8_hca_down, hc_attn_down, HCL, RSIZE); Q8(i8_hca_inj, hc_attn_inject, HCC, RSIZE); Q8(i8_hca_up, hc_attn_up, RSIZE, HCL);
        Q8(i8_hcf_down, hc_ffn_down, HCL, RSIZE);  Q8(i8_hcf_inj, hc_ffn_inject, HCC, RSIZE);  Q8(i8_hcf_up, hc_ffn_up, RSIZE, HCL);
        Q8(i8_shg, shexp_gate, NFF, NEMBD); Q8(i8_shu, shexp_up, NFF, NEMBD);
        Q8(i8_shgi, shexp_gate_inp, 1, NEMBD); Q8(i8_shd, shexp_down, NEMBD, NFF);
#undef Q8
        if (hipDeviceSynchronize() != hipSuccess) return -1;
        L->i8_dense = 1;
    }
    return 0;
}

// Sum the on-disk QfRec nbytes of every dense layer tensor this stage's
// loader will load (same name list as load_layer_real). Used only by the
// budget guard; missing tensors contribute 0 (the loader fails on them later).
static size_t stage_weight_bytes(const QfStore *st, int l0, int nl) {
    static const char *hc_fmt[8] = {
        "model.language_model.layers.%d.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_down.weight",
        "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.layers.%d.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.layers.%d.mlp_hyper_connection.hc_norm.weight",
        "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight",
        "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.layers.%d.mlp_hyper_connection.block_inject_weight.weight",
    };
    static const char *gdn_fmt[9] = {
        "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.%d.linear_attn.in_proj_z.weight",
        "model.language_model.layers.%d.linear_attn.conv1d.weight",
        "model.language_model.layers.%d.linear_attn.dt_bias",
        "model.language_model.layers.%d.linear_attn.A_log",
        "model.language_model.layers.%d.linear_attn.in_proj_a.weight",
        "model.language_model.layers.%d.linear_attn.in_proj_b.weight",
        "model.language_model.layers.%d.linear_attn.norm.weight",
        "model.language_model.layers.%d.linear_attn.out_proj.weight",
    };
    static const char *attn_fmt[6] = {
        "model.language_model.layers.%d.self_attn.q_proj.weight",
        "model.language_model.layers.%d.self_attn.k_proj.weight",
        "model.language_model.layers.%d.self_attn.v_proj.weight",
        "model.language_model.layers.%d.self_attn.o_proj.weight",
        "model.language_model.layers.%d.self_attn.q_norm.weight",
        "model.language_model.layers.%d.self_attn.k_norm.weight",
    };
    static const char *moe_fmt[5] = {
        "model.language_model.layers.%d.mlp.gate.weight",
        "model.language_model.layers.%d.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.%d.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.%d.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.%d.mlp.shared_expert_gate.weight",
    };
    size_t sum = 0;
    char nm[QF_MAX_NAME];
    for (int ll = 0; ll < nl; ll++) {
        int il = l0 + ll;
        int is_recr = ((il + 1) % 4) != 0;
        const char **lists[3] = { hc_fmt, is_recr ? gdn_fmt : attn_fmt, moe_fmt };
        int counts[3] = { 8, is_recr ? 9 : 6, 5 };
        for (int g = 0; g < 3; g++)
            for (int i = 0; i < counts[g]; i++) {
                snprintf(nm, sizeof(nm), lists[g][i], il);
                const QfEntry *e = qf_find(st, nm);
                if (e) sum += (size_t)e->rec.nbytes;
            }
    }
    return sum;
}

static long env_long(const char *name, long dflt) {
    const char *v = getenv(name);
    return v ? atol(v) : dflt;
}


// Layer count for a split deployment, from env OR from /tmp/qf_layers_total.
// The systemd EnvironmentFile is root-owned on this box, so the file sentinel
// is the only way to reconfigure a running unit. Default NLAYER = unchanged
// single-box behaviour.
// First global layer this NODE owns. 0 = this node is the head (it embeds the
// token). >0 = another node computed the earlier layers and hands the residual
// across, so no stage here embeds. Env QF_LAYER_BASE or /tmp/qf_layer_base.
static long qf4_layer_base(void) {
    static long v = -1;
    if (v >= 0) return v;
    v = env_long("QF_LAYER_BASE", -1);
    if (v < 0) {
        FILE *f = fopen("/tmp/qf_layer_base", "r");
        if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
    }
    if (v < 0 || v >= NLAYER) v = 0;
    v -= v % QF4_NGPU;          // keep stage boundaries aligned
    return v;
}

static long qf4_layers_total(void) {
    static long v = -1;
    if (v >= 0) return v;
    v = env_long("QF_LAYERS_TOTAL", 0);
    if (v <= 0) {
        FILE *f = fopen("/tmp/qf_layers_total", "r");
        if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
    }
    if (v < QF4_NGPU || v > NLAYER) v = NLAYER;
    v -= v % QF4_NGPU;
    return v;
}

int qf4_stage_init(Qf4Stage **out, int gpu, const QfStore *st, int l0) {
    if (hipSetDevice(gpu) != hipSuccess) { fprintf(stderr, "qf4: no GPU %d\n", gpu); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
    Qf4Stage *S = (Qf4Stage *)calloc(1, sizeof(Qf4Stage));
    if (!S) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    // Layers this NODE owns, for a split deployment. Default NLAYER keeps the
    // single-box behaviour identical. Must divide evenly across the 4 GPUs.
    const long ltot = qf4_layers_total();
    const int nl_per = (int)(ltot / QF4_NGPU);
    S->gpu = gpu; S->l0 = l0; S->nl = nl_per;
    S->st = st;
    // owns_embed: only the node holding layer 0 turns a token id into a
    // residual. A tail node's first stage receives the residual instead.
    S->owns_embed = (l0 == 0);
    S->is_first = (gpu == 0);
    S->is_last   = (gpu == QF4_NGPU - 1);          // last stage on this box
    S->owns_head = (l0 + nl_per >= NLAYER);        // ... and the model's tail
    // Alternating map: AMD owns the prefix AND the head. QF_AMD_OWNS_HEAD=1
    // makes the last AMD stage run the output HC + lm_head + argmax on the
    // residual the Spark tail hands back (lm_head measured 1204 GB/s at
    // 3.84x overlap here; the reply is a 4 B token id, never logits).
    if (env_long("QF_AMD_OWNS_HEAD", 0) && gpu == QF4_NGPU - 1) S->owns_head = 1;
    if (!S->owns_head && S->is_last)
        fprintf(stderr, "qf4: GPU %d is this node's tail but NOT the model's "
                        "(owns layers %d..%d of %d); emitting residual, no "
                        "output head\n", gpu, l0, l0 + nl_per - 1, NLAYER);
    S->ple_hist[0] = S->ple_hist[1] = QF4_EOS;

    // ---- budget guard: compute the per-GPU byte commitment BEFORE any
    // device allocation and bail out if it exceeds QF4_HBM_BUDGET_MIB ----
    // Context, from env or /tmp/qf_max_context (the systemd unit passes
    // --max-context and its EnvironmentFile is root-owned, so the sentinel is
    // the only way to retune a running deployment).
    long maxpos = env_long("QF_MAX_CONTEXT", 131072);
    {
        FILE *cf = fopen("/tmp/qf_max_context", "r");
        if (cf) { long v = 0; if (fscanf(cf, "%ld", &v) == 1 && v > 0) maxpos = v; fclose(cf); }
    }
    if (maxpos > 262144) maxpos = 262144;
    if (maxpos < 1) maxpos = 1;
    S->maxpos = maxpos;
    long budget_mib = env_long("QF4_HBM_BUDGET_MIB", 15872);
    // Expert slot bytes follow the resident FORMAT: the FP8 lane-chunk cache
    // (QF8_MOE_FP8=1) is 2.13x the NVFP4 slab, and the guard must price what
    // qf4_expcache_init will actually allocate or it under-counts by half.
    S->exp_fp8 = (getenv("QF8_MOE_FP8") && getenv("QF8_MOE_FP8")[0] == '1') ? 1 : 0;
    const int m1_i8 = (getenv("QF_M1_INT8") && getenv("QF_M1_INT8")[0] == '1') ? 1 : 0;
    // The M=1 int8 arenas live NEXT TO the NVFP4 slots they are built from
    // (3 x 1.64 MB + per-row scales per slot): price them per slot too.
    const size_t i8_slot_bytes = m1_i8 ? (size_t)3 * QF5_NFF * QF5_NEMBD + (size_t)(2 * QF5_NFF + QF5_NEMBD) * 4 : 0;
    const int m1_i8only = m1_i8 && getenv("QF_M1_INT8_ONLY") && getenv("QF_M1_INT8_ONLY")[0] == '1';
    const size_t slot_bytes = m1_i8only ? i8_slot_bytes
                            : (S->exp_fp8 ? QF8_FP8_SLOT_BYTES : (size_t)QF5_EXP_SLOT_BYTES) + i8_slot_bytes;

    size_t c_w    = stage_weight_bytes(st, l0, S->nl);
    if (m1_i8) c_w += c_w / 2;                     // M=1 int8 dense copies next to the bf16
    size_t c_gdn  = 0, c_kv = 0;
    for (int ll = 0; ll < S->nl; ll++) {
        int il = l0 + ll;
        if (((il + 1) % 4) != 0)
            // M-fold: per-row recurrent state
            c_gdn += (size_t)QF8_MROWS * ((size_t)48 * 128 * 128 * 4 + (size_t)3 * 10240 * 4);
        else
            // K and V, NKV=2 heads x HDIM=256, bf16 -> 2 bytes. This used a
            // stride of 4 and so double-counted KV; harmless at 8192 but it
            // would have wasted ~100 expert slots per layer at 262144.
            // M-fold: per-row KV at the KV element size (FP8 by policy)
            c_kv  += (size_t)QF8_MROWS * 2 * 2 * 256 * sizeof(qf8_kv_t) * (size_t)maxpos;
            // QSA indexer pools, sized by runtime context (see
            // qf8_qsa_idx_init): pool_sum + pool_key are [nb,128] f32, plus
            // blk_score f32 and blk_mask u8.
            {
                size_t nb = (size_t)maxpos / 4;
                c_kv += (size_t)QF8_MROWS * (nb * 128 * 4 * 2 + nb * 4 + nb);   // one indexer per row
            }
    }
    size_t c_extra = 2 * QF5_EXP_SLOT_BYTES;        // pinned staging
    if (S->owns_embed) c_extra += (size_t)0;                    // tok_embd stays on the host
    if (S->owns_head) c_extra += (m1_i8 ? (size_t)248320 * 2560 : (size_t)248320 * 2560 * 2) +   // lm_head: int8 only, or bf16
                                (size_t)NVOCAB * 4;             // logits buffers

    // ---- expert residency: spend ALL remaining HBM on it --------------------
    //
    // Measured on this box: at 192 of 512 slots per layer the cache misses
    // 31.4% of lookups and streams 397 MB/token H2D over PCIe. That traffic,
    // not HBM bandwidth (7.3% utilised) and not the inter-GPU handoff (0.4% of
    // the token), is what sets decode speed here. Every extra resident expert
    // removes 2.7 MB/token of PCIe transfer for a token that would have missed
    // it, so any HBM left unallocated is throughput thrown away.
    //
    // The deployed env pins QF_EXPERT_CACHE_SLOTS=192, but its own comment says
    // to raise it "until the per-GPU commitment approaches
    // QF4_HBM_BUDGET_MIB". Auto-sizing does exactly that, per GPU, accounting
    // for the fact that GPU0 and GPU3 also carry the 1.2 GiB embedding and
    // lm_head and so can hold fewer experts than GPU1/GPU2.
    long slots = env_long("QF_EXPERT_CACHE_SLOTS", 0);
    const int slots_auto = (slots <= 0) || (access("/tmp/qf_slots.auto", F_OK) == 0);
    if (slots_auto) {
        size_t fixed = c_w + c_gdn + c_kv + c_extra;
        size_t budget = (size_t)budget_mib * 1048576ull;
        size_t margin = 256ull * 1048576ull;            // headroom for scratch
        long fit = 0;
        if (budget > fixed + margin)
            fit = (long)((budget - fixed - margin) /
                         ((size_t)S->nl * slot_bytes));
        if (fit < 1) fit = 1;
        if (fit > NEXP) fit = NEXP;
        fprintf(stderr, "qf4: GPU %d expert slots auto %ld/%d (env asked %ld)\n",
                gpu, fit, NEXP, env_long("QF_EXPERT_CACHE_SLOTS", 0));
        slots = fit;
    }
    if (slots <= 0 || slots > NEXP) slots = NEXP;   // matches qf4_expcache_init
    S->exp_slots = (int)slots;

    size_t c_exp  = (size_t)S->nl * (size_t)slots * slot_bytes;
    size_t commit = c_w + c_exp + c_gdn + c_kv + c_extra;
    double gib = 1073741824.0;
    fprintf(stderr, "qf4: GPU %d commitment %.2f GiB / budget %.2f GiB (slots=%ld ctx=%ld experts=%s)\n",
            gpu, commit / gib, budget_mib / 1024.0, slots, maxpos, S->exp_fp8 ? "FP8" : (m1_i8 ? "NVFP4+INT8(M=1)" : "NVFP4"));
    if ((double)commit > (double)budget_mib * 1048576.0) {
        fprintf(stderr, "qf4: GPU %d OVER BUDGET: weights %.2f GiB, experts %.2f GiB, "
                "GDN %.2f GiB, KV %.2f GiB, other %.2f GiB\n",
                gpu, c_w / gib, c_exp / gib, c_gdn / gib, c_kv / gib, c_extra / gib);
        free(S);
        { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }      // before any device allocation
    }
    S->hbm_bytes = commit;

    // Per-GPU expert transfer context (fixed pinned double-buffer, transfer
    // stream, events), shared by this stage's 12 layer caches.
    if (qf4_expxfer_init(&S->xfer) != 0) { fprintf(stderr, "qf4: GPU %d expxfer init (pinned staging/stream/events) FAILED\n", gpu); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }

#define AL(f, bytes) do { \
    if (hipMalloc((void **)&S->f, bytes) != hipSuccess) { \
        stage_err("hipMalloc failed for", #f, hipErrorOutOfMemory); return -1; } } while (0)
    // M-fold (QF8_MROWS rows resident): every per-row activation buffer carries
    // a row stride so the executor can run M=8, or a row subset [r0,r0+nr) for
    // A/B M=4 waves. Row r of buffer f lives at S->f + r * <stride>.
    // Activation buffers carry max(QF8_MROWS, QF_CHUNK_ROWS) rows: the prefill
    // chunk executor uses rows as positions (row 0's recurrent state only).
    const int MR = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
    AL(x, (size_t)MR * NEMBD * 4);   AL(R, (size_t)MR * RSIZE * 4);  AL(normed, (size_t)MR * RSIZE * 4);
    AL(hc_d, (size_t)MR * HCL * 4);  AL(mixed, (size_t)MR * NEMBD * 4); AL(inj, (size_t)MR * HCC * 4);
    AL(y2560, (size_t)MR * NEMBD * 4); AL(qkv_raw, (size_t)MR * DINN * 4); AL(z6144, (size_t)MR * 6144 * 4);
    AL(q12288, (size_t)MR * 12288 * 4); AL(k512, (size_t)MR * NKV * HDIM * 4); AL(v512, (size_t)MR * NKV * HDIM * 4);
    AL(a48, (size_t)MR * DTRANK * 4); AL(b48, (size_t)MR * DTRANK * 4);
    AL(attn_out, (size_t)MR * NHEAD * HDIM * 4); AL(scores, (size_t)S->maxpos * 4);
    AL(router, (size_t)MR * NEXP * 4);
    // eg doubles as the fused MoE hidden buffer: NEXPUSED independent slices
    // of NFF so all selected experts can run in ONE launch. The shared expert
    // reuses slice 0 afterwards, which is free by then.
    AL(eg, (size_t)MR * NEXPUSED * NFF * 4); AL(eu, (size_t)MR * NFF * 4); AL(ed, (size_t)MR * NEMBD * 4);
    AL(out48, (size_t)MR * GDN_VH * GDN_VD * 4); AL(gdn_out_bf, (size_t)MR * GDN_VH * GDN_VD * 2);
    AL(exp_hist, (size_t)NEXP * 4);
    if (hipMemset(S->exp_hist, 0, (size_t)NEXP * 4) != hipSuccess) { fprintf(stderr, "qf4: GPU %d exp_hist memset FAILED\n", gpu); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
    AL(inv_freq, 32 * 4);
    AL(inv_freq_idx, 64 * 4); AL(idx640, (size_t)MR * 640 * 4); AL(attn_out8, (size_t)MR * 6144 * 4);
    AL(x_bf16, (size_t)MR * NEMBD * 2);
    AL(sel_ids_dev, (size_t)MR * NEXPUSED * sizeof(int)); AL(wts_dev, (size_t)MR * NEXPUSED * 4);
    { const size_t np = (size_t)MR * NEXPUSED;
      AL(m8_exp_slot, np * sizeof(int)); AL(m8_exp_ptr, (np + 1) * sizeof(int));
      AL(m8_pair_tok, np * sizeof(int)); AL(m8_pair_wt, np * 4); AL(m8_counts, 2 * sizeof(int));
      AL(m8_part, np * (size_t)NEMBD * 4); }
    AL(c8q6, (size_t)MR * 6144); AL(c8s6, (size_t)MR * (6144 / 32) * 4);
    AL(cn8q, (size_t)MR * RSIZE); AL(cn8s, (size_t)MR * (RSIZE / 32) * 4);
    AL(cd8q, (size_t)MR * HCL); AL(cd8s, (size_t)MR * (HCL / 32) * 4);
    AL(c_hcy, (size_t)MR * RSIZE * 4);
    // M=1 int8 expert path scratch (X8 activation + the 10 hidden rows)
    AL(x8q, (size_t)NEMBD); AL(x8s, (size_t)(NEMBD / 32) * 4);
    AL(h8q, (size_t)NEXPUSED * NFF); AL(h8s, (size_t)NEXPUSED * (NFF / 32) * 4);
    AL(n8q, (size_t)RSIZE); AL(n8s, (size_t)(RSIZE / 32) * 4); AL(d8q, (size_t)HCL); AL(d8s, (size_t)(HCL / 32) * 4);
    AL(attn_part, (size_t)QF8_FLASH_PART_FLOATS * 4);
    S->c_nb_max = (int)((S->maxpos + 3) / 4);
    AL(c_blk_score, (size_t)MR * S->c_nb_max * 4); AL(c_blk_mask, (size_t)MR * S->c_nb_max);
    AL(c_blk_list, (size_t)MR * (QF8_IDX_TOPK + 1) * sizeof(int)); AL(c_nsel, (size_t)MR * sizeof(int));
    AL(c_part, (size_t)MR * QF8_FLASH_PART_FLOATS * 4);
    AL(c8q, (size_t)MR * NEMBD); AL(c8s, (size_t)MR * (NEMBD / 32) * 4);
    AL(ch8q, (size_t)MR * NEXPUSED * NFF); AL(ch8s, (size_t)MR * NEXPUSED * (NFF / 32) * 4);
    S->m8_desc = NULL; S->m8_ctl = NULL; S->m8_hidden = NULL;
    if (S->exp_fp8) {
        // one descriptor per (row, expert) pair at most; hidden is [desc][row][NFF]
        AL(m8_desc, (size_t)MR * NEXPUSED * sizeof(Qf8MoeDesc));
        AL(m8_ctl, (size_t)QF8_FP8_CTL * sizeof(int));
        AL(m8_hidden, (size_t)MR * NEXPUSED * QF8_FP8_MAXROW * NFF * 4);
    }
    // P2P handoff buffers + events (also created in pinned mode; 80 KiB total)
    AL(handoff[0], RSIZE * 4); AL(handoff[1], RSIZE * 4);
        if (S->owns_head) { AL(logits, (size_t)QF8_MROWS * NVOCAB * 4); AL(ids_dev, (size_t)QF8_MROWS * sizeof(int)); }   // head rows, not chunk rows

    float invf[32];
    for (int i = 0; i < 32; i++) invf[i] = powf(1e7f, -2.f * i / 64.f);
    if (hipMemcpy(S->inv_freq, invf, sizeof(invf), hipMemcpyHostToDevice) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    float invf_idx[64];
    for (int i = 0; i < 64; i++) invf_idx[i] = powf(1e7f, -2.f * i / 128.f);
    if (hipMemcpy(S->inv_freq_idx, invf_idx, sizeof(invf_idx), hipMemcpyHostToDevice) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }

    // recurrent state for owned layers (local index ll, global il = l0 + ll)
    for (int ll = 0; ll < S->nl; ll++) {
        int il = l0 + ll;
        if (((il + 1) % 4) != 0) {
            // per-row recurrent state: row r at + r * QF8_GDNS_STRIDE / QF8_RING_STRIDE
            // Recurrent state is per REQUEST row (QF8_MROWS), never per chunk row: a
            // chunk advances row 0's state in order. Sizing these by MR put 256 GDN
            // states (2.4 GB) + 256 KV caches (4.3 GB) on the head card -> OOM.
            AL(gdnS[ll], (size_t)QF8_MROWS * QF8_GDNS_STRIDE * 4);
            AL(convring[ll], (size_t)QF8_MROWS * QF8_RING_STRIDE * 4);
            if (hipMemset(S->gdnS[ll], 0, (size_t)QF8_MROWS * QF8_GDNS_STRIDE * 4) != hipSuccess ||
                hipMemset(S->convring[ll], 0, (size_t)QF8_MROWS * QF8_RING_STRIDE * 4) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
        } else {
            // per-row KV cache + one indexer per row (independent sequences)
            AL(kc[ll], (size_t)QF8_MROWS * S->maxpos * NKV * HDIM * sizeof(qf8_kv_t));
            AL(vc[ll], (size_t)QF8_MROWS * S->maxpos * NKV * HDIM * sizeof(qf8_kv_t));
            for (int r = 0; r < MR; r++)
                if (qf8_qsa_idx_init(&S->idx[ll * QF8_MROWS + r], S->maxpos) != 0) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
        }
    }

    // weights
    for (int ll = 0; ll < S->nl; ll++)
        if (load_layer_real(S, &S->L[ll], l0 + ll, st) != 0) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }

    char nm[QF_MAX_NAME];
    if (S->owns_embed) {
        // The 1.27 GB bf16 embedding table stays on the HOST (the mmap'd store):
        // embedding is a 5 KB row gather per token, staged through emb_pin/emb_dev.
        {
            const QfEntry *e = qf_find(st, "model.language_model.embed_tokens.weight");
            if (!e || e->rec.nbytes != (size_t)NVOCAB * NEMBD * 2) { fprintf(stderr, "qf4: embed_tokens missing/odd size\n"); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
            S->tok_embd_host = (const uint16_t *)((const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off);
            S->tok_embd = NULL;
            // Pinned host copy of the table. Gathering rows straight from the
            // mmap'd store page-faults on nearly every new token: set_tokens_rows
            // went 0.29 -> 1.92 ms/token (32.4 -> 29.8 tok/s decode). A 1.27 GB
            // pinned copy makes the 5 KB row gather a memcpy from resident memory.
            if (!(getenv("QF_EMBED_PIN") && getenv("QF_EMBED_PIN")[0] == '0')) {
                uint16_t *pin = NULL;
                if (hipHostMalloc((void **)&pin, (size_t)NVOCAB * NEMBD * 2, hipHostMallocPortable) == hipSuccess) {
                    memcpy(pin, S->tok_embd_host, (size_t)NVOCAB * NEMBD * 2);
                    S->tok_embd_host = pin;
                    fprintf(stderr, "qf4: GPU %d embedding table pinned on the host (%.2f GB)\n", gpu, (double)NVOCAB * NEMBD * 2 / 1e9);
                } else fprintf(stderr, "qf4: GPU %d embedding pin FAILED, gathering from the mmap'd store\n", gpu);
            }
            const int MRe = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
            if (hipHostMalloc((void **)&S->emb_pin, (size_t)MRe * NEMBD * 2, hipHostMallocPortable) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
            if (hipMalloc((void **)&S->emb_dev, (size_t)MRe * NEMBD * 2) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
        }
    }
    // PLE n-gram injection at layer 1 is a first-class part of this
    // architecture, not an accessory. There is no no-PLE path: running a
    // forward pass without it is a silent quality cliff that looks exactly like
    // a working model (NEVER_AGAIN rule 1).
    //
    // It belongs to whichever stage OWNS layer 1 - stated as an explicit range
    // test rather than riding on "the first stage", which only coincides while
    // stage 0 happens to hold >= 2 layers and the node happens to be the head.
    if (S->l0 <= 1 && 1 < S->l0 + S->nl) {
        if (qf8_ple_init(&S->ple, st, 1) != 0 || !S->ple) {
            fprintf(stderr, "qf4 stage: FATAL - PLE block unavailable; refusing "
                            "to run a forward pass without PLE injection\n");
            { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
        }
    }
    // lm_head and the output HC mixer belong to whoever owns the model's FINAL
    // layer. Under a split this node may be a prefix and must not load them -
    // that is 1.2 GiB of HBM it would waste, and the residual goes on the wire
    // instead.
    if (S->owns_head) {
        const int i8head = getenv("QF_M1_INT8") && getenv("QF_M1_INT8")[0] == '1' &&
                           !(getenv("QF_M1_INT8_HEAD") && getenv("QF_M1_INT8_HEAD")[0] == '0');
        if (i8head) {
            // int8 head straight from the mmap'd bf16 table: the 1.27 GB bf16 copy
            // never lands on this card (it was the head card's OOM at 256 chunk rows).
            const QfEntry *e = qf_find(st, "lm_head.weight");
            if (!e || e->rec.nbytes != (size_t)NVOCAB * NEMBD * 2) { fprintf(stderr, "qf4: lm_head missing/odd size\n"); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
            const uint16_t *lh = (const uint16_t *)((const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off);
            if (qf8_i8_quant_bf16_host(lh, NVOCAB, NEMBD, &S->i8_lm_head, 0) != 0) { fprintf(stderr, "qf4: GPU %d int8 lm_head FAILED\n", gpu); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
            S->lm_head = NULL;
        } else
        S->lm_head = load_bf16(st, "lm_head.weight", (size_t)NVOCAB * NEMBD);
        if (!S->lm_head && !i8head) { fprintf(stderr, "qf4: GPU %d lm_head load FAILED\n", gpu); return -1; }
        snprintf(nm, sizeof(nm), "model.language_model.hyper_connection_mixer.hc_norm.weight");
        S->ohc_norm = load_bf16(st, nm, RSIZE);
        snprintf(nm, sizeof(nm), "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight");
        S->ohc_down = load_bf16(st, nm, (size_t)HCL * RSIZE);
        snprintf(nm, sizeof(nm), "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight");
        S->ohc_up = load_bf16(st, nm, (size_t)RSIZE * HCL);
        S->i8_head = 0;
        if (S->ohc_up && i8head) {
            if (qf8_i8_quant_bf16(S->ohc_down, HCL, RSIZE, &S->i8_ohc_down, 0) != 0 ||
                qf8_i8_quant_bf16(S->ohc_up, RSIZE, HCL, &S->i8_ohc_up, 0) != 0 ||
                hipDeviceSynchronize() != hipSuccess) {
                fprintf(stderr, "qf4: GPU %d int8 head quantization FAILED\n", gpu); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
            S->i8_head = 1;
            fprintf(stderr, "qf4: GPU %d M=1 INT8 head (lm_head from the host table + output HC), no bf16 lm_head on device\n", gpu);
        }
        if ((!S->lm_head && !S->i8_head) || !S->ohc_norm || !S->ohc_down || !S->ohc_up) { fprintf(stderr, "qf4: GPU %d head weights missing\n", gpu); return -1; }
    }

    for (int i = 0; i < 2; i++)
        if (hipEventCreateWithFlags(&S->hev[i], hipEventDisableTiming) != hipSuccess)
            { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    {
        hipError_t se = hipStreamCreate(&S->stream);
        if (se != hipSuccess) {
            size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot);
            fprintf(stderr, "qf4: GPU %d hipStreamCreate FAILED: %s (free %.2f / %.2f GiB)\n", gpu, hipGetErrorString(se),
                    fr / 1073741824.0, tot / 1073741824.0);
            return -1;
        }
    }

    // If every expert fits, give each a permanent address now. This retires the
    // entire per-token host handshake (mailbox stamp, host spin, LRU victim
    // search, miss upload) - measured at 29% of the stage - because a fully
    // resident map has exactly one possible answer: expert e lives in slot e.
    // Costs a one-off upload at init; buys zero host work in the decode path.
    for (int ll = 0; ll < S->nl; ll++) {
        int r = qf4_expcache_prefill(&S->L[ll].cache, st, &S->xfer, S->stream);
        if (r < 0) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
        if (r > 0 && ll == 0)
            fprintf(stderr, "qf4: GPU %d experts FULLY RESIDENT (%d/%d per layer)"
                            " - static slot map, no per-token host servicing\n",
                    gpu, S->exp_slots, NEXP);
    }
    if (hipStreamSynchronize(S->stream) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    // M=1 int8 representation: built once, on device, from the resident slots.
    for (int ll = 0; ll < S->nl; ll++) {
        Qf4ExpCache *C = &S->L[ll].cache;
        if (!C->i8_gate || C->nv_ring) continue;    // ring loader converted per expert
        if (!C->fully_resident) { fprintf(stderr, "qf4: GPU %d L%d int8 build needs full residency\n", gpu, S->l0 + ll); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
        if (qf8_i8_build(C, S->stream) != 0) { fprintf(stderr, "qf4: GPU %d L%d int8 build FAILED\n", gpu, S->l0 + ll); { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; } }
    }
    if (hipStreamSynchronize(S->stream) != hipSuccess) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    if (S->L[0].cache.i8_gate)
        fprintf(stderr, "qf4: GPU %d M=1 INT8 expert arenas built (%d layers x %d slots)\n", gpu, S->nl, S->exp_slots);
    if (qf8_prof_init(&S->prof) != 0) { fprintf(stderr, "qf4: GPU %d init FAILED at stage line %d\n", gpu, __LINE__); return -1; }
    *out = S;
    return 0;
}

void qf4_stage_free(Qf4Stage *S) {
    // Weight pointers are released implicitly by process exit in v1; the
    // expert caches and transfer context are released explicitly because the
    // store (mmap) may outlive the stage. (hipFree on each remaining weight
    // pointer is safe to add if the store outlives the engine.)
    if (!S) return;
    hipSetDevice(S->gpu);
    for (int ll = 0; ll < S->nl; ll++) {
        qf4_expcache_free(&S->L[ll].cache);
        if (((S->l0 + ll + 1) % 4) == 0)
            for (int r = 0; r < QF8_MROWS; r++) qf8_qsa_idx_free(&S->idx[ll * QF8_MROWS + r]);
    }
    qf8_ple_free(S->ple);
    qf8_prof_free(&S->prof);
    qf4_expxfer_free(&S->xfer);
    for (int i = 0; i < 2; i++) {
        if (S->handoff[i]) hipFree(S->handoff[i]);
        if (S->hev[i]) hipEventDestroy(S->hev[i]);
    }
    if (S->stream) hipStreamDestroy(S->stream);
    free(S);
}

// ---- launcher helpers -----------------------------------------------------
static void gemv(const uint16_t *W, const float *x, float *y, int out, int in, hipStream_t s) {
    dim3 blk(WAVE, 4);
    QF4_LAUNCH(k_gemv_bf16, (out + 3) / 4, blk, 0, s, W, x, y, out, in);
}
// HC pre-mixer: n = rmsnorm(R); d = silu(down(n)/4); mixed = mean_c(sigmoid(up(d)) * n);
// optionally inj[c] = 2*sigmoid(inject(n)/4).
static void hc_pre(Qf4Stage *S, const uint16_t *wn, const uint16_t *wd,
                   const uint16_t *wu, const uint16_t *wi, hipStream_t s) {
    QF4_LAUNCH(k_rmsnorm_group, RSIZE / NEMBD, 256, 0, s, S->normed, S->R, wn, NEMBD, 1e-6f);
    QF4_LAUNCH(k_hc_down, HCL, WAVE, 0, s, S->normed, wd, S->hc_d);
    QF4_LAUNCH(k_zero, (NEMBD + 255) / 256, 256, 0, s, S->mixed, NEMBD);
    QF4_LAUNCH(k_hc_up, RSIZE / 256, 256, 0, s, S->normed, wu, S->hc_d, S->mixed);
    if (wi) QF4_LAUNCH(k_hc_inject, 1, 256, 0, s, S->normed, wi, S->inj);
}

// Gather T embedding rows (bf16) from the host table into emb_dev rows [0, T).
static int stage_embed_rows(Qf4Stage *S, const int *tokens, int T, hipStream_t s) {
    for (int t = 0; t < T; t++)
        memcpy(S->emb_pin + (size_t)t * NEMBD, S->tok_embd_host + (size_t)tokens[t] * NEMBD, (size_t)NEMBD * 2);
    return hipMemcpyAsync(S->emb_dev, S->emb_pin, (size_t)T * NEMBD * 2, hipMemcpyHostToDevice, s) == hipSuccess ? 0 : -1;
}
int qf4_stage_set_token(Qf4Stage *S, int token) {
    SETDEV(S);
    // Only the stage owning layer 0 turns a token id into a residual. On a tail
    // node nothing embeds - qf4_stage_push_residual supplies R instead.
    if (!S->owns_embed) return -1;
    hipStream_t s = S->stream;
    if (stage_embed_rows(S, &token, 1, s) != 0) return -1;
    QF4_LAUNCH(k_cvt_bf16_f32, (NEMBD + 255) / 256, 256, 0, s,
               S->x, S->emb_dev, NEMBD);
    QF4_LAUNCH(k_repeat4, RSIZE / 256, 256, 0, s, S->R, S->x);
    if (S->ple && qf8_ple_step(S->ple, S->st, token) != 0) return -1;
    if (hipStreamSynchronize(s) != hipSuccess) return -1;
    return 0;
}

int qf4_stage_push_residual(Qf4Stage *S, const float *hR) {
    SETDEV(S);
    // Host-staged handoff (no P2P): plain synchronous H2D of the 40 KiB R.
    if (hipMemcpy(S->R, hR, RSIZE * 4, hipMemcpyHostToDevice) != hipSuccess) return -1;
    return 0;
}

int qf4_stage_pull_residual(Qf4Stage *S, float *hR) {
    SETDEV(S);
    if (hipMemcpy(hR, S->R, RSIZE * 4, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    return 0;
}


// ---- M-row variants of the stage boundary ops (row-strided) ----------------
int qf4_stage_set_tokens_M(Qf4Stage *S, const int *tokens, int M) {
    SETDEV(S);
    if (!S->owns_embed) return -1;
    hipStream_t s = S->stream;
    if (stage_embed_rows(S, tokens, M, s) != 0) return -1;
    for (int r = 0; r < M; r++) {
        QF4_LAUNCH(k_cvt_bf16_f32, (NEMBD + 255) / 256, 256, 0, s,
                   S->x + (size_t)r * NEMBD, S->emb_dev + (size_t)r * NEMBD, NEMBD);
        QF4_LAUNCH(k_repeat4, RSIZE / 256, 256, 0, s, S->R + (size_t)r * RSIZE, S->x + (size_t)r * NEMBD);
    }
    // per-row PLE: each row advances its OWN n-gram history and staging buffer
    if (S->ple)
        for (int r = 0; r < M; r++)
            if (qf8_ple_step_row(S->ple, S->st, tokens[r], r) != 0) return -1;
    return hipStreamSynchronize(s) == hipSuccess ? 0 : -1;
}
int qf4_stage_push_residual_M(Qf4Stage *S, const float *hR, int M) {
    SETDEV(S);
    return hipMemcpy(S->R, hR, (size_t)M * RSIZE * 4, hipMemcpyHostToDevice) == hipSuccess ? 0 : -1;
}
// Row-subset forms [r0, r0+nr): used to prime one request row at a time across
// the box split (each prompt has its own length, so rows are primed
// independently), and by any scheduler that advances a row subset.
int qf4_stage_set_tokens_rows(Qf4Stage *S, const int *tokens, int r0, int nr) {
    SETDEV(S);
    if (!S->owns_embed) return -1;
    if (r0 < 0 || nr < 1 || r0 + nr > QF8_MROWS) return -1;
    hipStream_t s = S->stream;
    if (stage_embed_rows(S, tokens, nr, s) != 0) return -1;
    for (int i = 0; i < nr; i++) {
        const int r = r0 + i;
        QF4_LAUNCH(k_cvt_bf16_f32, (NEMBD + 255) / 256, 256, 0, s,
                   S->x + (size_t)r * NEMBD, S->emb_dev + (size_t)i * NEMBD, NEMBD);
        QF4_LAUNCH(k_repeat4, RSIZE / 256, 256, 0, s, S->R + (size_t)r * RSIZE, S->x + (size_t)r * NEMBD);
    }
    if (S->ple)
        for (int i = 0; i < nr; i++)
            if (qf8_ple_step_row(S->ple, S->st, tokens[i], r0 + i) != 0) return -1;
    return hipStreamSynchronize(s) == hipSuccess ? 0 : -1;
}
int qf4_stage_push_residual_rows(Qf4Stage *S, const float *hR, int r0, int nr) {
    SETDEV(S);
    const int maxr = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
    if (r0 < 0 || nr < 1 || r0 + nr > maxr) return -1;
    return hipMemcpy(S->R + (size_t)r0 * RSIZE, hR, (size_t)nr * RSIZE * 4, hipMemcpyHostToDevice) == hipSuccess ? 0 : -1;
}
int qf4_stage_pull_residual_rows(Qf4Stage *S, float *hR, int r0, int nr) {
    SETDEV(S);
    const int maxr = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
    if (r0 < 0 || nr < 1 || r0 + nr > maxr) return -1;
    return hipMemcpy(hR, S->R + (size_t)r0 * RSIZE, (size_t)nr * RSIZE * 4, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
int qf4_stage_pull_residual_M(Qf4Stage *S, float *hR, int M) {
    SETDEV(S);
    return hipMemcpy(hR, S->R, (size_t)M * RSIZE * 4, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
int qf4_stage_pull_logits_M(Qf4Stage *S, float *h, int M) {
    SETDEV(S);
    if (!S->is_last) return -1;
    return hipMemcpy(h, S->logits, (size_t)M * NVOCAB * 4, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
// Greedy token ids of the M head rows, argmax on device, M ints to the host.
// Per-row Gumbel-max sampling (temperature > 0): argmax_v (logit_v / T + g_v), g_v = -log(-log(u_v)) with u_v
// from a deterministic 64-bit hash of (seed, row, v) - an exact sample of softmax(logits / T), reproducible
// by seed. Row 0 stays greedy (the control row of a fork receipt); ties to the lowest index like the argmax.
static __device__ __forceinline__ uint64_t k_splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
__global__ void k_sample_rows(const float *__restrict__ logits, int *__restrict__ ids, int n, float inv_temp, uint64_t seed) {
    const int r = blockIdx.x;
    const float *row = logits + (size_t)r * n;
    float best = -3.0e38f; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (r > 0) {
            const uint64_t h = k_splitmix(seed ^ ((uint64_t)r << 40) ^ (uint64_t)i);
            const float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
            v = v * inv_temp - logf(-logf(u));
        }
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    __shared__ float sv[256]; __shared__ int si[256];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int off = blockDim.x >> 1; off; off >>= 1) {
        if ((int)threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off]; const int oi = si[threadIdx.x + off];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[r] = si[0];
}
static float g_head_temp = -1.f; static uint64_t g_head_seed = 7, g_head_calls = 0;
int qf4_stage_head_ids_M(Qf4Stage *S, int *ids, int M) {
    SETDEV(S);
    if (!S->owns_head || !S->ids_dev || M < 1 || M > QF8_MROWS) return -1;
    if (g_head_temp < 0.f) { const char *e = getenv("QF_HEAD_TEMP"); g_head_temp = e ? (float)atof(e) : 0.f; const char *z = getenv("QF_HEAD_SEED"); if (z) g_head_seed = (uint64_t)atoll(z); }
    if (g_head_temp > 0.f && M > 1) {
        QF4_LAUNCH(k_sample_rows, M, 256, 0, S->stream, S->logits, S->ids_dev, NVOCAB, 1.f / g_head_temp, g_head_seed + 0x1000003ull * (g_head_calls++));
    } else
    QF4_LAUNCH(k_argmax_rows, M, 256, 0, S->stream, S->logits, S->ids_dev, NVOCAB);
    if (hipStreamSynchronize(S->stream) != hipSuccess) return -1;
    return hipMemcpy(ids, S->ids_dev, (size_t)M * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
// Reset the per-field sampling counter: every client fork must start its field from seed+0,
// independent of how many head-M calls this boot has already served (decision/commit loop law).
void qf4_stage_head_calls_reset(void) { g_head_calls = 0; }

// Device-to-device R handoff for a row subset between adjacent stages.
int qf4_stage_handoff_M(Qf4Stage *src, Qf4Stage *dst, int r0, int nr) {
    size_t off = (size_t)r0 * RSIZE, n = (size_t)nr * RSIZE * 4;
    if (hipSetDevice(src->gpu) != hipSuccess) return -1;
    if (hipStreamSynchronize(src->stream) != hipSuccess) return -1;
    return hipMemcpyPeer(dst->R + off, dst->gpu, src->R + off, src->gpu, n) == hipSuccess ? 0
         : -1;
}

int qf4_stage_pull_logits(Qf4Stage *S, float *h) {
    SETDEV(S);
    if (!S->is_last) return -1;
    if (hipMemcpy(h, S->logits, (size_t)NVOCAB * 4, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// wave4 session 09: device-resident router. The old host router (D2H of the
// 512 logits + hipStreamSynchronize + host top-10 + host LRU slot lookup) is
// gone; per MoE layer the stage now enqueues:
//   qf_rtr_gemv -> qf_rtr_topk10 -> qf_rtr_dispatch   (compute stream)
// then qf4_expcache_service spins on the layer's pinned mailbox (zero-copy,
// written by the dispatch kernel) and DMAs any missed experts on the transfer
// stream while the compute stream still executes earlier work, and finally
// the 10 device-indexed expert kernels are enqueued (they read expert id,
// slot, scale_2 and weight on device). No router logits leave the GPU.
// ---------------------------------------------------------------------------
// ---- correctness taps (inert unless armed) --------------------------------
// Appends <u32 taglen><tag><i32 layer><u32 n><n f32> per tap, the format
// tools/verify_real_stages.py reads. Entirely off the hot path: a stream sync
// plus a D2H, and only on the single decode step at QF_TAP_POS (default 4,
// the last id of the 5-token acceptance prompt).
//
// Armed by the EXISTENCE of /tmp/qf_tap.arm rather than an env var, so taps
// can be turned on for a running systemd unit without restarting it (a
// restart costs a full 4-GPU model load). The access() check only runs on the
// one step whose position matches, so the decode path pays nothing.
#define QF_TAP_ARM  "/tmp/qf_tap.arm"
static long g_tap_pos = 4;

// Returns 0 (off), 1 (tap this position only) or 2 (tap EVERY position, so a
// recurrent block's state can be reproduced from its own inputs). Mode 2 is
// selected by writing "all" into the arm file.
//
// The output file is PER STAGE (/tmp/qf_tap.g<gpu>.bin): each stage runs on
// its own host thread, so a single shared FILE* would interleave records from
// four writers and corrupt the stream.
static int tap_active(Qf4Stage *S, long pos) {
    if (access(QF_TAP_ARM, F_OK) != 0) return 0;   // cheap and the common case
    // arm file contents: "all" = every position; an integer = that position
    // only (needed for long context, where per-position tapping of the indexer
    // pools would be gigabytes); empty = the default position.
    int all = 0;
    FILE *a = fopen(QF_TAP_ARM, "r");
    if (a) {
        char b[32] = {0};
        if (fgets(b, sizeof(b), a)) {
            if (strncmp(b, "all", 3) == 0) all = 1;
            else { long v = atol(b); if (v > 0) g_tap_pos = v; }
        }
        fclose(a);
    }
    if (!all && pos != g_tap_pos) return 0;
    if (!S->tapf) {
        char path[64];
        snprintf(path, sizeof(path), "/tmp/qf_tap.g%d.bin", S->gpu);
        S->tapf = fopen(path, "wb");
    }
    if (!S->tapf) return 0;
    return all ? 2 : 1;
}

// uint8 tap (selection masks), written as floats so the reader stays uniform.
static void tap_u8(Qf4Stage *S, const char *tag, int il, const uint8_t *dev,
                   int n, hipStream_t s) {
    if (!S->tapf || !dev) return;
    if (hipStreamSynchronize(s) != hipSuccess) return;
    uint8_t *h = (uint8_t *)malloc((size_t)n);
    float *f = (float *)malloc((size_t)n * 4);
    if (h && f && hipMemcpy(h, dev, (size_t)n, hipMemcpyDeviceToHost) == hipSuccess) {
        for (int i = 0; i < n; i++) f[i] = (float)h[i];
        uint32_t tl = (uint32_t)strlen(tag);
        int32_t l32 = il; uint32_t n32 = (uint32_t)n;
        fwrite(&tl, 4, 1, S->tapf); fwrite(tag, 1, tl, S->tapf);
        fwrite(&l32, 4, 1, S->tapf); fwrite(&n32, 4, 1, S->tapf);
        fwrite(f, 4, (size_t)n, S->tapf);
        fflush(S->tapf);
    }
    free(h); free(f);
}

static void tap_vec(Qf4Stage *S, const char *tag, int il, const float *dev,
                    int n, hipStream_t s) {
    if (!S->tapf || !dev) return;
    if (hipStreamSynchronize(s) != hipSuccess) return;
    float *h = (float *)malloc((size_t)n * 4);
    if (!h) return;
    if (hipMemcpy(h, dev, (size_t)n * 4, hipMemcpyDeviceToHost) == hipSuccess) {
        uint32_t tl = (uint32_t)strlen(tag);
        int32_t  l32 = il;
        uint32_t n32 = (uint32_t)n;
        fwrite(&tl, 4, 1, S->tapf);
        fwrite(tag, 1, tl, S->tapf);
        fwrite(&l32, 4, 1, S->tapf);
        fwrite(&n32, 4, 1, S->tapf);
        fwrite(h, 4, (size_t)n, S->tapf);
        fflush(S->tapf);
    }
    free(h);
}


// ===========================================================================
// M=8 / M=4 row-subset execution over this same executor. No new arithmetic:
// every stage calls the kernels already used by stage_run_impl. Row-indexed
// mutable state (gdnS/convring/kc/vc/idx) is addressed by row stride, so rows
// are independent sequences; weight-dominated projections (router, lm_head)
// use the M-GEMV so ONE weight stream serves the whole row subset.
// r0/nr express A={0..3} / B={4..7} for the heterogeneous A/B scheduler.
// ===========================================================================
int qf4_stage_layer_M(Qf4Stage *S, int ll, const long *posM, int r0, int nr) {
    SETDEV(S);
    hipStream_t s = S->stream;
    const int il = S->l0 + ll;
    const int is_recr = ((il + 1) % 4) != 0;
    Qf4Layer *L = &S->L[ll];
    const size_t kvspan = (size_t)S->maxpos * NKV * HDIM;

    if (il == 1 && S->ple) {
        qf8_prof_mark(&S->prof, QF8_SEC_PLE, s);
        for (int r = r0; r < r0 + nr; r++)
            qf8_ple_apply_row(S->ple, S->R + (size_t)r * RSIZE, r, s);
    }

    qf8_prof_mark(&S->prof, QF8_SEC_HC_ATTN, s);
    if (nr == 1 && L->i8_dense)
        qf8_i8_hc_m1(S->normed + (size_t)r0 * RSIZE, S->hc_d + (size_t)r0 * HCL,
                     S->mixed + (size_t)r0 * NEMBD, S->inj + (size_t)r0 * HCC, S->R + (size_t)r0 * RSIZE,
                     L->hc_attn_norm, &L->i8_hca_down, &L->i8_hca_inj, &L->i8_hca_up,
                     S->n8q, S->n8s, S->d8q, S->d8s, S->q12288 + (size_t)r0 * 12288,
                     NULL, NULL, S->x8q, S->x8s, s);   // q12288 free here; emits X8(mixed)
    else
    qf8_hc_pre_M(S->normed + (size_t)r0 * RSIZE, S->hc_d + (size_t)r0 * HCL,
                 S->mixed + (size_t)r0 * NEMBD, S->inj + (size_t)r0 * HCC,
                 S->R + (size_t)r0 * RSIZE,
                 L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, nr, s);

    qf8_prof_mark(&S->prof, QF8_SEC_MIXER, s);
    if (is_recr && nr == 1) {
        // M=1 product path, section-marked: MIXER = the 4-matrix projection
        // GEMV, EXPSVC (printed "exp_svc") = the fused conv+recurrence+gated
        // norm, OUT_HEAD (printed "out_head") = the 6144->2560 out-projection.
        Qf8GdnW g = { L->qkv, L->zgate, L->in_a, L->in_b, L->conv1d,
                      L->a_log, L->dt_bias, L->gdn_norm, L->gdn_out };
        const int r = r0;
        if (L->i8_dense) {
            // X8(mixed) was emitted by the attention-side hc combine kernel
            Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 4;
            m.m[0] = &L->i8_qkv;   m.y[0] = S->qkv_raw + (size_t)r * DINN;
            m.m[1] = &L->i8_zgate; m.y[1] = S->z6144 + (size_t)r * 6144;
            m.m[2] = &L->i8_ina;   m.y[2] = S->a48 + (size_t)r * DTRANK;
            m.m[3] = &L->i8_inb;   m.y[3] = S->b48 + (size_t)r * DTRANK;
            qf8_i8_gemv_multi(&m, S->x8q, S->x8s, s);
        } else
        qf8_gdn_proj(&g, S->mixed + (size_t)r * NEMBD, S->qkv_raw + (size_t)r * DINN,
                     S->z6144 + (size_t)r * 6144, S->a48 + (size_t)r * DTRANK,
                     S->b48 + (size_t)r * DTRANK, s);
        qf8_prof_mark(&S->prof, QF8_SEC_GDN_RECUR, s);
        qf8_gdn_recur(&g, S->gdnS[ll] + (size_t)r * QF8_GDNS_STRIDE,
                      S->convring[ll] + (size_t)r * QF8_RING_STRIDE,
                      S->qkv_raw + (size_t)r * DINN, S->z6144 + (size_t)r * 6144,
                      S->a48 + (size_t)r * DTRANK, S->b48 + (size_t)r * DTRANK,
                      S->q12288 + (size_t)r * 12288, s);
        qf8_prof_mark(&S->prof, QF8_SEC_GDN_OUT, s);
        if (L->i8_dense) {
            qf8_i8_x8(S->q12288 + (size_t)r * 12288, S->h8q, S->h8s, 6144, s);   // h8q/h8s: 6400/200, free here
            Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 1; m.m[0] = &L->i8_gdn_out; m.y[0] = S->y2560 + (size_t)r * NEMBD;
            qf8_i8_gemv_multi(&m, S->h8q, S->h8s, s);
        } else
        qf8_gdn_outproj(&g, S->q12288 + (size_t)r * 12288, S->y2560 + (size_t)r * NEMBD, s);
    } else if (is_recr) {
        Qf8GdnW g = { L->qkv, L->zgate, L->in_a, L->in_b, L->conv1d,
                      L->a_log, L->dt_bias, L->gdn_norm, L->gdn_out };
        for (int r = r0; r < r0 + nr; r++)
            qf8_gdn_mixer(&g, S->mixed + (size_t)r * NEMBD,
                          S->gdnS[ll] + (size_t)r * QF8_GDNS_STRIDE,
                          S->convring[ll] + (size_t)r * QF8_RING_STRIDE,
                          S->qkv_raw + (size_t)r * DINN, S->z6144 + (size_t)r * 6144,
                          S->a48 + (size_t)r * DTRANK, S->b48 + (size_t)r * DTRANK,
                          S->q12288 + (size_t)r * 12288, S->y2560 + (size_t)r * NEMBD, s);
    } else if (nr == 1 && L->i8_dense) {
        Qf8AttnW a = { L->wq, L->wk, L->wv, L->wo, L->q_norm, L->k_norm,
                       L->idx_qk, L->idx_qnorm, L->idx_knorm };
        const int r = r0;
        qf8_qsa_mixer_i8(&a, &L->i8_wq, &L->i8_wk, &L->i8_wv, &L->i8_idxqk, &L->i8_wo,
                         S->mixed + (size_t)r * NEMBD, S->x8q, S->x8s, S->h8q, S->h8s,
                         S->kc[ll] + (size_t)r * kvspan, S->vc[ll] + (size_t)r * kvspan,
                         &S->idx[ll * QF8_MROWS + r], posM[r],
                         S->q12288 + (size_t)r * 12288,
                         S->k512 + (size_t)r * NKV * HDIM, S->v512 + (size_t)r * NKV * HDIM,
                         S->idx640 + (size_t)r * 640, S->attn_out8 + (size_t)r * 6144,
                         S->inv_freq, S->inv_freq_idx, S->y2560 + (size_t)r * NEMBD, /*quant_x=*/0,
                         S->attn_part, s);
    } else {
        Qf8AttnW a = { L->wq, L->wk, L->wv, L->wo, L->q_norm, L->k_norm,
                       L->idx_qk, L->idx_qnorm, L->idx_knorm };
        for (int r = r0; r < r0 + nr; r++)
            qf8_qsa_mixer(&a, S->mixed + (size_t)r * NEMBD,
                          S->kc[ll] + (size_t)r * kvspan, S->vc[ll] + (size_t)r * kvspan,
                          &S->idx[ll * QF8_MROWS + r], posM[r],
                          S->q12288 + (size_t)r * 12288,
                          S->k512 + (size_t)r * NKV * HDIM, S->v512 + (size_t)r * NKV * HDIM,
                          S->idx640 + (size_t)r * 640, S->attn_out8 + (size_t)r * 6144,
                          S->inv_freq, S->inv_freq_idx, S->y2560 + (size_t)r * NEMBD, s);
    }
    qf8_prof_mark(&S->prof, QF8_SEC_INJECT1, s);
    if (!(nr == 1 && L->i8_dense))                  // fused into the FFN-side norm kernel otherwise
    for (int r = r0; r < r0 + nr; r++)
        qf8_stream_inject(S->R + (size_t)r * RSIZE, S->y2560 + (size_t)r * NEMBD,
                          S->inj + (size_t)r * HCC, s);

    qf8_prof_mark(&S->prof, QF8_SEC_HC_FFN, s);
    if (nr == 1 && L->i8_dense)
        qf8_i8_hc_m1(S->normed + (size_t)r0 * RSIZE, S->hc_d + (size_t)r0 * HCL,
                     S->mixed + (size_t)r0 * NEMBD, S->inj + (size_t)r0 * HCC, S->R + (size_t)r0 * RSIZE,
                     L->hc_ffn_norm, &L->i8_hcf_down, &L->i8_hcf_inj, &L->i8_hcf_up,
                     S->n8q, S->n8s, S->d8q, S->d8s, S->q12288 + (size_t)r0 * 12288,
                     S->y2560 + (size_t)r0 * NEMBD, S->inj + (size_t)r0 * HCC,     // fused mixer inject
                     S->x8q, S->x8s, s);   // out6144 consumed; emits X8(mixed) for router/experts/shexp
    else
    qf8_hc_pre_M(S->normed + (size_t)r0 * RSIZE, S->hc_d + (size_t)r0 * HCL,
                 S->mixed + (size_t)r0 * NEMBD, S->inj + (size_t)r0 * HCC,
                 S->R + (size_t)r0 * RSIZE,
                 L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, nr, s);

    // router: ONE weight stream for the row subset
    qf8_prof_mark(&S->prof, QF8_SEC_ROUTER, s);
    if (nr == 1 && L->i8_dense) {
        // X8(mixed) was emitted by the FFN-side hc combine kernel (router, experts, shexp)
        Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 1; m.m[0] = &L->i8_router; m.y[0] = S->router + (size_t)r0 * NEXP;
        qf8_i8_gemv_multi(&m, S->x8q, S->x8s, s);
    } else
    qf8_gemv_bf16_M(L->router, S->mixed + (size_t)r0 * NEMBD,
                    S->router + (size_t)r0 * NEXP, NEXP, NEMBD, nr, 1.f, 0.f, s);
    for (int r = r0; r < r0 + nr; r++)
        qf_rtr_topk10_launch(S->router + (size_t)r * NEXP,
                             S->sel_ids_dev + (size_t)r * NEXPUSED,
                             S->wts_dev + (size_t)r * NEXPUSED, s);

    // routed experts, M-BATCHED: one resident weight traversal serves every row
    // routed to that expert (v100-skinny lesson: N dots against ONE activation
    // tile). CSR is built ON DEVICE from the batch top-k -- no host routing in
    // the hot path (GLM k_moe_gather8 mechanism).
    {
        Qf4ExpCache *C = &L->cache;
        qf8_prof_mark(&S->prof, QF8_SEC_EXPERTS, s);
        // Both routed bodies ACCUMULATE into y2560 with atomicAdd (a row takes
        // contributions from up to 10 experts), so the row subset is zeroed on
        // the stream first. y2560 still holds the attention mixer output here
        // (already injected into R above); without this the routed sum lands on
        // top of it.
        static int m1_moe_all = -1;
        if (m1_moe_all < 0) { const char *e = getenv("QF_M1_MOE_ALL"); m1_moe_all = (e && e[0] == '0') ? 0 : 1; }
        if (nr == 1 && !C->w8_gate && m1_moe_all) {
            // M=1 PRODUCT PATH: the fused all-expert NVFP4 kernels the single-row
            // executor runs (two launches for all 10 experts, per-expert hidden
            // slices, one writer per output row, y SEEDED so no zeroing pass).
            // The M-row CSR kernels above are shaped for TX-token batches and
            // are the wrong executor at one row. slot == expert (fully resident,
            // checked at init), so the sel ids double as slot ids.
            const float *xr = S->mixed + (size_t)r0 * NEMBD;
            float *hid = S->eg + (size_t)r0 * NEXPUSED * NFF, *yr = S->y2560 + (size_t)r0 * NEMBD;
            const int *sel = S->sel_ids_dev + (size_t)r0 * NEXPUSED;
            const float *wts = S->wts_dev + (size_t)r0 * NEXPUSED;
            if (C->i8_gate) {
                // gfx906-native int8 path (qf8_moe_i8.h): dot4, HBM-bound.
                qf8_i8_gateup_m1(C, xr, S->x8q, S->x8s, hid, sel, NEXPUSED, /*quant_x=*/!L->i8_dense, s);
                qf8_prof_mark(&S->prof, QF8_SEC_EXP_DOWN, s);
                qf8_i8_down_m1(C, hid, S->h8q, S->h8s, wts, sel, yr, NEXPUSED, s);
            } else {
            qf5_nvfp4_gateup_all_launch(C->w_gate, C->s_gate, C->w_up, C->s_up, xr, hid,
                                        C->s2g_dev, C->s2u_dev, sel, sel, NEXPUSED, s);
            qf8_prof_mark(&S->prof, QF8_SEC_EXP_DOWN, s);
            qf5_nvfp4_downacc_all_launch(C->w_down, C->s_down, hid, wts, yr,
                                         C->s2d_dev, sel, sel, NEXPUSED, s);
            }
            static int twice = -1;
            if (twice < 0) twice = getenv("QF_M1_MOE_TWICE") ? 1 : 0;
            if (twice) {   // diagnostic: same experts again, warm TLB/L2 -> cold-vs-warm split
                qf8_prof_mark(&S->prof, QF8_SEC_EXP_GU2, s);
                qf5_nvfp4_gateup_all_launch(C->w_gate, C->s_gate, C->w_up, C->s_up, xr, hid,
                                            C->s2g_dev, C->s2u_dev, sel, sel, NEXPUSED, s);
                qf8_prof_mark(&S->prof, QF8_SEC_EXP_DN2, s);
                qf5_nvfp4_downacc_all_launch(C->w_down, C->s_down, hid, wts, yr,
                                             C->s2d_dev, sel, sel, NEXPUSED, s);
            }
        } else {
        if (hipMemsetAsync(S->y2560 + (size_t)r0 * NEMBD, 0, (size_t)nr * NEMBD * 4, s) != hipSuccess)
            return -1;
        if (C->w8_gate) {
            // gfx906 production FP8 path: device routing inversion -> expert
            // descriptors -> persistent fused gate/up/SiLU -> fused down+accumulate.
            // slot == expert id: the region layout is fully resident (checked at init).
            qf8_fp8_build_desc(S->sel_ids_dev + (size_t)r0 * NEXPUSED,
                               S->wts_dev + (size_t)r0 * NEXPUSED, nr, NEXPUSED,
                               /*slot_base=*/0, S->m8_desc, S->m8_ctl, s);
            qf8_fp8_gateup(C->w8_gate, C->sc_gate, C->w8_up, C->sc_up,
                           S->mixed + (size_t)r0 * NEMBD, S->m8_hidden, S->m8_desc, S->m8_ctl, s);
            qf8_fp8_downacc(C->w8_down, C->sc_down, S->m8_hidden,
                            S->y2560 + (size_t)r0 * NEMBD, S->m8_desc, S->m8_ctl, s);
        } else {
            qf5_m8_build_csr(S->sel_ids_dev + (size_t)r0 * NEXPUSED,
                             S->wts_dev + (size_t)r0 * NEXPUSED, nr, NEXPUSED,
                             /*card=*/0, /*epc=*/NEXP, /*il=*/0,   // stage cache is slot==expert
                             S->m8_exp_slot, S->m8_exp_ptr, S->m8_pair_tok, S->m8_pair_wt,
                             S->m8_counts, s);
            Qf5M8Route rt;
            rt.n_exp = 0; rt.n_pairs = 0; rt.M = nr;
            rt.n_exp_max = nr * NEXPUSED;
            rt.n_exp_dev = S->m8_counts;
            rt.exp_slot = S->m8_exp_slot; rt.exp_ptr = S->m8_exp_ptr;
            rt.pair_tok = S->m8_pair_tok; rt.pair_wt = S->m8_pair_wt;
            qf5_m8_gateup(C->w_gate, C->s_gate, C->w_up, C->s_up,
                          C->s2g_dev, C->s2u_dev,
                          S->mixed + (size_t)r0 * NEMBD, S->eg + (size_t)r0 * NEXPUSED * NFF,
                          &rt, s);
            qf5_m8_downacc(C->w_down, C->s_down, C->s2d_dev,
                           S->eg + (size_t)r0 * NEXPUSED * NFF,
                           S->y2560 + (size_t)r0 * NEMBD, &rt, s);
        }
        }   // end M-row (nr > 1 or FP8) expert path
    }
    // shared expert, M-BATCHED: one weight stream for the row subset
    {
        qf8_prof_mark(&S->prof, QF8_SEC_SHEXP, s);
        Qf8ShexpW sw = { L->shexp_gate, L->shexp_up, L->shexp_down, L->shexp_gate_inp };
        if (nr == 1 && L->i8_dense) {
            // x8q/x8s still hold X8(mixed) from the router (the expert path
            // re-quantizes the same vector into them). a48 is free scratch here.
            qf8_i8_shexp_m1(&L->i8_shg, &L->i8_shu, &L->i8_shgi, &L->i8_shd, S->x8q, S->x8s,
                            S->eg + (size_t)r0 * NEXPUSED * NFF, S->eu + (size_t)r0 * NFF,
                            S->a48 + (size_t)r0 * DTRANK, S->ed + (size_t)r0 * NEMBD,
                            S->h8q, S->h8s, S->y2560 + (size_t)r0 * NEMBD, s);
        } else
        // g1 (the shared-expert gate scalar, one float per row) must NOT alias
        // ed: the down GEMV writes ed[0] before k8_shexp_add reads g1[0].
        // a48 (48 floats per row, consumed by the recurrence earlier) is free.
        qf8_shexp_M(&sw, S->mixed + (size_t)r0 * NEMBD, nr,
                    S->eg + (size_t)r0 * NEXPUSED * NFF, S->eu + (size_t)r0 * NFF,
                    S->a48 + (size_t)r0 * DTRANK, S->ed + (size_t)r0 * NEMBD,
                    S->y2560 + (size_t)r0 * NEMBD, NEXPUSED * NFF, NEMBD, s);
    }
    qf8_prof_mark(&S->prof, QF8_SEC_INJECT1, s);
    for (int r = r0; r < r0 + nr; r++)
        qf8_stream_inject(S->R + (size_t)r * RSIZE, S->y2560 + (size_t)r * NEMBD,
                          S->inj + (size_t)r * HCC, s);
    return 0;
}

// ===========================================================================
// CHUNK (prefill) executor: T consecutive POSITIONS of the ONE sequence in row 0.
// Rows 0..T-1 of the activation buffers hold the T tokens; every batched
// kernel (hyper-connections, projections, router, shared expert) runs over the T
// rows with its weights read ONCE, the routed experts run through the int8 CSR
// kernels (weights read once per expert, dotted against its routed tokens), and
// the genuinely recurrent parts (PLE history/ring, GDN state, QSA KV + indexer)
// advance in order on ROW 0's state. T <= QF8_I8_CSR_TX (16).
// ===========================================================================
// Bisection mode for the chunk executor, read per call from /tmp/qf_chunk_mode:
// "serial" = 1 (single-row path per position), "expm1" = 2 (CSR experts replaced by the
// per-token M=1 kernels), anything else / no file = 0 (full chunk path).
static int chunk_mode(void) {
    FILE *f = fopen("/tmp/qf_chunk_mode", "r");
    if (!f) return 0;
    char b[32] = {0}; size_t n = fread(b, 1, sizeof b - 1, f); fclose(f); b[n] = 0;
    if (!strncmp(b, "serial", 6)) return 1;
    if (!strncmp(b, "expm1", 5)) return 2;
    if (!strncmp(b, "gdnpos", 6)) return 3;
    return 0;
}
#define QF8_GEMV_MROWS 8   // == QF8_M_MAX in qf8_gemv_m.hip: accumulator rows of the bf16 M-row GEMVs
int qf4_chunk_rows(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QF_CHUNK_ROWS"); v = e ? atoi(e) : 0; if (v < 0) v = 0; if (v > 1024) v = 1024; }
    return v;
}
// Chunk-executor section timers (QF_REGION_TIMING): CT_MARKS stream events per
// layer, folded per stage after the chunk into one stderr line. Surgical.
#define CT_MARKS 14
static hipEvent_t g_ct_ev[8][CT_MARKS * 8];
static int g_ct_n[8], g_ct_on = -1;
static double g_ct_csr_ms = 0.0, g_ct_x8_ms = 0.0;
static double qf4_wall_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
extern "C" void qf8_i8_chunk_prof_print(FILE *out, const char *tag);
static void ct_mark(Qf4Stage *S, hipStream_t s) {
    if (g_ct_on < 0) g_ct_on = getenv("QF_REGION_TIMING") ? 1 : 0;
    if (!g_ct_on) return;
    const int g = S->gpu & 7;
    if (g_ct_n[g] >= CT_MARKS * 8) return;
    if (!g_ct_ev[g][g_ct_n[g]]) hipEventCreate(&g_ct_ev[g][g_ct_n[g]]);
    hipEventRecord(g_ct_ev[g][g_ct_n[g]++], s);
}
static void ct_fold(Qf4Stage *S, int T) {
    if (!g_ct_on) return;
    static const char *nm[CT_MARKS] = { "gap", "ple", "hc", "gproj", "gdn", "gout", "qsa", "inj", "hc2", "rtr", "topk", "exp", "shx", "inj2" };
    const int g = S->gpu & 7; float sec[CT_MARKS] = {0}, tot = 0.f;
    if (g_ct_n[g] < 2) { g_ct_n[g] = 0; return; }
    hipEventSynchronize(g_ct_ev[g][g_ct_n[g] - 1]);
    for (int i = 0; i + 1 < g_ct_n[g]; i++) { float ms = 0.f; hipEventElapsedTime(&ms, g_ct_ev[g][i], g_ct_ev[g][i + 1]); sec[(i + 1) % CT_MARKS] += ms; tot += ms; }
    fprintf(stderr, "[region] chunk g%d T=%d (%d layers) %.1f ms:", S->gpu, T, S->nl, tot);
    for (int k = 1; k < CT_MARKS; k++) fprintf(stderr, " %s %.1f", nm[k], sec[k]);
    fprintf(stderr, " gap %.1f\n", sec[0]);
    { char tag[16]; snprintf(tag, sizeof tag, "g%d", S->gpu); qf8_i8_chunk_prof_print(stderr, tag);
      fprintf(stderr, "[i8exp g%d] build_csr %.1f x8quant %.1f ms (host walls, synced)\n", S->gpu, g_ct_csr_ms, g_ct_x8_ms);
      g_ct_csr_ms = g_ct_x8_ms = 0.0; }
    g_ct_n[g] = 0;
}
static int qf4_stage_layer_chunk(Qf4Stage *S, int ll, const int *tokens, long pos0, int T) {
    SETDEV(S);
    hipStream_t s = S->stream;
    const int il = S->l0 + ll;
    const int is_recr = ((il + 1) % 4) != 0;
    Qf4Layer *L = &S->L[ll];
    const size_t kvspan = (size_t)S->maxpos * NKV * HDIM;
    if (!L->i8_dense || !L->cache.i8_gate) { fprintf(stderr, "chunk: needs the int8 layer copies (QF_M1_INT8=1)\n"); return -1; }
    ct_mark(S, s);                                           // 0 layer start

    if (il == 1 && S->ple) {
        // ROW 0's PLE over the T positions: the host gather + H2D ran at token intake
        // (qf4_stage_set_chunk_tokens, slot S->ple_slot) while the GPU was on the previous
        // chunk; here only the dequant, projections and the recurrent conv launch remain.
        qf8_ple_apply_chunk_slot(S->ple, S->R, T, S->ple_slot, s);
    }
    ct_mark(S, s);                                           // 1 ple
    // attention-side hyper-connection over the chunk (int8 rows form); emits X8(mixed) into c8q/c8s
    if (qf8_i8_hc_rows(S->normed, S->hc_d, S->mixed, S->inj, S->R, L->hc_attn_norm, &L->i8_hca_down, &L->i8_hca_inj, &L->i8_hca_up,
                       S->cn8q, S->cn8s, S->cd8q, S->cd8s, S->c_hcy, NULL, NULL, S->c8q, S->c8s, T, s)) { fprintf(stderr, "chunk: hc rows failed\n"); return -1; }
    ct_mark(S, s);                                           // 2 hc
    if (is_recr) {
        Qf8GdnW g = { L->qkv, L->zgate, L->in_a, L->in_b, L->conv1d,
                      L->a_log, L->dt_bias, L->gdn_norm, L->gdn_out };
        // the four in-projections over the chunk. The bf16 M-row GEMV kernels
        // carry QF8_GEMV_MROWS (8) rows of accumulators, so run 8-row groups.
        {   // the four in-projections over the chunk: int8 rows GEMV on X8(mixed), weights once per 16 positions
            Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 4;
            m.m[0] = &L->i8_qkv;   m.y[0] = S->qkv_raw;
            m.m[1] = &L->i8_zgate; m.y[1] = S->z6144;
            m.m[2] = &L->i8_ina;   m.y[2] = S->a48;
            m.m[3] = &L->i8_inb;   m.y[3] = S->b48;
            if (qf8_i8_gemv_rows(&m, S->c8q, S->c8s, T, s)) { fprintf(stderr, "chunk: gdn proj rows failed\n"); return -1; }
        }
        ct_mark(S, s);                                       // 3 gdn in-proj
        if (chunk_mode() == 3) {                // bisection: per-position recurrence
            for (int t = 0; t < T; t++)
                qf8_gdn_recur(&g, S->gdnS[ll], S->convring[ll],
                              S->qkv_raw + (size_t)t * DINN, S->z6144 + (size_t)t * 6144,
                              S->a48 + (size_t)t * DTRANK, S->b48 + (size_t)t * DTRANK,
                              S->attn_out8 + (size_t)t * 6144, s);
        } else {
            // conv + chunked delta rule + gated norm over the T positions (row 0's
            // ring and state), out rows contiguous [T][6144] in attn_out8.
            if (qf8_gdn_chunk(&g, S->gdnS[ll], S->convring[ll], S->qkv_raw, S->z6144, S->a48, S->b48,
                              S->attn_out8, T, s) != 0) { fprintf(stderr, "chunk: gdn chunk failed\n"); return -1; }
        }
        ct_mark(S, s);                                       // 4 gdn core
        {   // out-projection: X8 of the gated-norm output, int8 rows GEMV (K = 6144)
            qf8_i8_x8(S->attn_out8, S->c8q6, S->c8s6, T * 6144, s);
            Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 1; m.m[0] = &L->i8_gdn_out; m.y[0] = S->y2560;
            if (qf8_i8_gemv_rows(&m, S->c8q6, S->c8s6, T, s)) { fprintf(stderr, "chunk: gdn out rows failed\n"); return -1; }
        }
        ct_mark(S, s); ct_mark(S, s);                        // 5 gdn out-proj, 6 (no qsa)
    } else {
        ct_mark(S, s); ct_mark(S, s); ct_mark(S, s);         // 3,4,5 (no gdn)
        Qf8AttnW a = { L->wq, L->wk, L->wv, L->wo, L->q_norm, L->k_norm,
                       L->idx_qk, L->idx_qnorm, L->idx_knorm };
        static int qsa_rows = -1;
        if (qsa_rows < 0) { const char *e = getenv("QF_QSA_ROWS"); qsa_rows = e ? atoi(e) : 1; }
        if (qsa_rows) {
            // CHUNK2: QSA stage -> all positions (int8 rows projections, one launch per stage)
            qf8_i8_x8(S->mixed, S->c8q, S->c8s, T * NEMBD, s);
            if (qf8_qsa_chunk2(&a, &L->i8_wq, &L->i8_wk, &L->i8_wv, &L->i8_idxqk, &L->i8_wo,
                               S->c8q, S->c8s, S->c8q6, S->c8s6, T, S->kc[ll], S->vc[ll],
                               &S->idx[ll * QF8_MROWS], pos0, S->q12288, S->k512, S->v512, S->idx640,
                               S->attn_out8, S->inv_freq, S->inv_freq_idx, S->y2560,
                               S->c_blk_score, S->c_blk_mask, S->c_blk_list, S->c_nsel, S->c_nb_max,
                               S->c_part, s)) { fprintf(stderr, "chunk: qsa chunk2 failed\n"); return -1; }
        } else
        // projections batched over the chunk; KV, indexer and attention per position on row 0
        qf8_qsa_chunk(&a, S->mixed, T, S->kc[ll], S->vc[ll], &S->idx[ll * QF8_MROWS], pos0,
                      S->q12288, S->k512, S->v512, S->idx640, S->attn_out8,
                      S->inv_freq, S->inv_freq_idx, S->y2560, S->attn_part, s);
        ct_mark(S, s);                                       // 6 qsa (proj + per-position attention)
    }
    ct_mark(S, s);                                           // 7 (inject fused into the FFN-side norm below)
    // FFN-side hyper-connection: the mixer's stream inject (R += inj * y2560) is fused
    // into its norm kernel; emits the new inj and X8(mixed) for router / experts / shexp
    if (qf8_i8_hc_rows(S->normed, S->hc_d, S->mixed, S->inj, S->R, L->hc_ffn_norm, &L->i8_hcf_down, &L->i8_hcf_inj, &L->i8_hcf_up,
                       S->cn8q, S->cn8s, S->cd8q, S->cd8s, S->c_hcy, S->y2560, S->inj, S->c8q, S->c8s, T, s)) { fprintf(stderr, "chunk: hc2 rows failed\n"); return -1; }
    ct_mark(S, s);                                           // 8 hc2
    {   Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 1; m.m[0] = &L->i8_router; m.y[0] = S->router;
        if (qf8_i8_gemv_rows(&m, S->c8q, S->c8s, T, s)) { fprintf(stderr, "chunk: router rows failed\n"); return -1; }
    }
    ct_mark(S, s);                                           // 9 router
    qf_rtr_topk10_rows_launch(S->router, S->sel_ids_dev, S->wts_dev, T, s);   // T rows, one launch
    ct_mark(S, s);                                           // 10 top-k (rows form)
    {   // routed experts: device routing inversion + int8 CSR kernels
        Qf4ExpCache *C = &L->cache;
        const int exp_m1 = (chunk_mode() == 2);   // bisection: per-token M=1 kernels
        if (exp_m1) {
            for (int t = 0; t < T; t++) {
                qf8_i8_gateup_m1(C, S->mixed + (size_t)t * NEMBD, S->x8q, S->x8s, S->eg,
                                 S->sel_ids_dev + (size_t)t * NEXPUSED, NEXPUSED, /*quant_x=*/1, s);
                qf8_i8_down_m1(C, S->eg, S->h8q, S->h8s, S->wts_dev + (size_t)t * NEXPUSED,
                               S->sel_ids_dev + (size_t)t * NEXPUSED, S->y2560 + (size_t)t * NEMBD, NEXPUSED, s);
            }
        } else {
        if (hipMemsetAsync(S->y2560, 0, (size_t)T * NEMBD * 4, s) != hipSuccess) return -1;
        double tw0 = 0.0; if (g_ct_on == 1) { hipStreamSynchronize(s); tw0 = qf4_wall_ms(); }
        qf8_i8_csr_build(S->sel_ids_dev, S->wts_dev, T, S->m8_exp_slot, S->m8_exp_ptr, S->m8_pair_tok, S->m8_pair_wt, S->m8_counts, s);
        if (g_ct_on == 1) { hipStreamSynchronize(s); g_ct_csr_ms += qf4_wall_ms() - tw0; tw0 = qf4_wall_ms(); }
        // X8(mixed) already emitted by the FFN-side hc combine into c8q/c8s
        if (g_ct_on == 1) { hipStreamSynchronize(s); g_ct_x8_ms += qf4_wall_ms() - tw0; }
        qf8_i8_experts_chunk(C, S->c8q, S->c8s, S->eg, S->ch8q, S->ch8s,
                             S->m8_exp_slot, S->m8_exp_ptr, S->m8_pair_tok, S->m8_pair_wt,
                             S->m8_counts, (T * NEXPUSED) < NEXP ? (T * NEXPUSED) : NEXP, T * NEXPUSED,
                             S->sel_ids_dev, S->m8_part, T, S->y2560, s);
        }
    }
    ct_mark(S, s);                                           // 11 routed experts (CSR)
    {   // shared expert over the chunk (int8 rows form); g1 = a48 as a [T] scratch, ed never aliased
        if (qf8_i8_shexp_rows(&L->i8_shg, &L->i8_shu, &L->i8_shgi, &L->i8_shd, S->c8q, S->c8s, S->eg, S->eu, S->a48, S->ed,
                              S->ch8q, S->ch8s, S->y2560, T, s)) { fprintf(stderr, "chunk: shexp rows failed\n"); return -1; }
    }
    ct_mark(S, s);                                           // 12 shared expert
    for (int t = 0; t < T; t++)
        qf8_stream_inject(S->R + (size_t)t * RSIZE, S->y2560 + (size_t)t * NEMBD, S->inj + (size_t)t * HCC, s);
    ct_mark(S, s);                                           // 13 inject2 (per position)
    return 0;
}
int qf4_stage_run_chunk(Qf4Stage *S, const int *tokens, long pos0, int T) {
    const int maxT = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
    if (T < 1 || T > maxT) return -1;
    if (chunk_mode() == 1) {                       // bisection: the proven single-row path per position
        // rows 0..T-1 hold the T residuals; run each position through row 0's
        // single-row path (state of row 0), copying the residual in and out.
        SETDEV(S);
        for (int t = 0; t < T; t++) {
            if (t > 0 && hipMemcpyAsync(S->R, S->R + (size_t)t * RSIZE, (size_t)RSIZE * 4, hipMemcpyDeviceToDevice, S->stream) != hipSuccess) return -1;
            if (S->owns_embed && S->ple) {            // PLE history of row 0 for this token (stage 0 only)
                if (qf8_ple_step_row(S->ple, S->st, tokens[t], 0) != 0) return -1;
            }
            long posM[QF8_MROWS]; for (int i = 0; i < QF8_MROWS; i++) posM[i] = 0; posM[0] = pos0 + t;
            if (qf4_stage_run_M(S, posM, 0, 1) != 0) return -1;
            if (t > 0 && hipMemcpyAsync(S->R + (size_t)t * RSIZE, S->R, (size_t)RSIZE * 4, hipMemcpyDeviceToDevice, S->stream) != hipSuccess) return -1;
        }
        // row 0 now holds the LAST position's residual; restore position 0's from... (rows were
        // consumed in order and row 0 was overwritten by each position) -> keep row t copies above.
        return hipStreamSynchronize(S->stream) == hipSuccess ? 0 : -1;
    }
    for (int ll = 0; ll < S->nl; ll++)
        if (qf4_stage_layer_chunk(S, ll, tokens, pos0, T) != 0) return -1;
    ct_fold(S, T);
    return 0;
}
// Embed T positions into rows 0..T-1 WITHOUT touching the per-row PLE history
// (the chunk executor steps row 0's history itself, in order).
// ASYNC: the pinned embedding staging (emb_pin) is reused only after the previous
// chunk's H2D has completed (event), never on host call order. No stream sync:
// the caller chains the stage's work through events (pipelined chunk server).
int qf4_stage_set_chunk_tokens(Qf4Stage *S, const int *tokens, int T) {
    SETDEV(S);
    if (!S->owns_embed) return -1;
    const int maxT = qf4_chunk_rows() > QF8_MROWS ? qf4_chunk_rows() : QF8_MROWS;
    if (T < 1 || T > maxT) return -1;
    hipStream_t s = S->stream;
    if (!S->cemb_xfer) {                              // first chunk: transfer stream + per-slot stagings
        if (hipStreamCreateWithFlags(&S->cemb_xfer, hipStreamNonBlocking) != hipSuccess) return -1;
        for (int k = 0; k < QF8_PLE_SLOTS; k++) {
            if (hipHostMalloc((void **)&S->cemb_pin[k], (size_t)maxT * NEMBD * 2, hipHostMallocPortable) != hipSuccess) return -1;
            if (hipMalloc((void **)&S->cemb_dev[k], (size_t)maxT * NEMBD * 2) != hipSuccess) return -1;
            if (hipEventCreateWithFlags(&S->cemb_ev[k], hipEventDisableTiming) != hipSuccess) return -1;
            if (hipEventRecord(S->cemb_ev[k], S->cemb_xfer) != hipSuccess) return -1;   // "free" from the start
        }
        S->cemb_slot = 0;
    }
    S->cemb_slot = (S->cemb_slot + 1) % QF8_PLE_SLOTS;
    const int k = S->cemb_slot;
    if (hipEventSynchronize(S->cemb_ev[k]) != hipSuccess) return -1;   // this slot's H2D of QF8_PLE_SLOTS chunks ago landed
    for (int t = 0; t < T; t++)
        memcpy(S->cemb_pin[k] + (size_t)t * NEMBD, S->tok_embd_host + (size_t)tokens[t] * NEMBD, (size_t)NEMBD * 2);
    if (hipMemcpyAsync(S->cemb_dev[k], S->cemb_pin[k], (size_t)T * NEMBD * 2, hipMemcpyHostToDevice, S->cemb_xfer) != hipSuccess) return -1;
    if (hipEventRecord(S->cemb_ev[k], S->cemb_xfer) != hipSuccess) return -1;
    if (hipStreamWaitEvent(s, S->cemb_ev[k], 0) != hipSuccess) return -1;   // compute stream consumes the slot after its H2D
    if (S->ple && chunk_mode() != 1) {                // PLE gather + H2D for this chunk now (overlaps the GPU's previous chunk)
        S->ple_slot = (S->ple_slot + 1) % QF8_PLE_SLOTS;
        if (qf8_ple_prefetch_chunk(S->ple, S->st, tokens, T, S->ple_slot) != 0) { fprintf(stderr, "chunk: PLE prefetch failed\n"); return -1; }
    }
    for (int t = 0; t < T; t++) {
        QF4_LAUNCH(k_cvt_bf16_f32, (NEMBD + 255) / 256, 256, 0, s,
                   S->x + (size_t)t * NEMBD, S->cemb_dev[k] + (size_t)t * NEMBD, NEMBD);
        QF4_LAUNCH(k_repeat4, RSIZE / 256, 256, 0, s, S->R + (size_t)t * RSIZE, S->x + (size_t)t * NEMBD);
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
// Pipelined chunk plumbing: async residual rows in/out on the stage's stream,
// events recorded on / awaited by that stream (cross-device waits allowed).
int qf4_stage_push_rows_async(Qf4Stage *S, const float *hR, int nr) {
    SETDEV(S);
    return hipMemcpyAsync(S->R, hR, (size_t)nr * RSIZE * 4, hipMemcpyHostToDevice, S->stream) == hipSuccess ? 0 : -1;
}
int qf4_stage_pull_rows_async(Qf4Stage *S, float *hR, int nr) {
    SETDEV(S);
    return hipMemcpyAsync(hR, S->R, (size_t)nr * RSIZE * 4, hipMemcpyDeviceToHost, S->stream) == hipSuccess ? 0 : -1;
}
int qf4_stage_record(Qf4Stage *S, void *ev) { SETDEV(S); return hipEventRecord((hipEvent_t)ev, S->stream) == hipSuccess ? 0 : -1; }
int qf4_stage_wait(Qf4Stage *S, void *ev) { SETDEV(S); return hipStreamWaitEvent(S->stream, (hipEvent_t)ev, 0) == hipSuccess ? 0 : -1; }
int qf4_stage_event_create(Qf4Stage *S, void **ev) { SETDEV(S); return hipEventCreateWithFlags((hipEvent_t *)ev, hipEventDisableTiming) == hipSuccess ? 0 : -1; }

// All owned layers for a row subset (dependency-closed region on this card).
int qf4_stage_run_M(Qf4Stage *S, const long *posM, int r0, int nr) {
    for (int ll = 0; ll < S->nl; ll++)
        if (qf4_stage_layer_M(S, ll, posM, r0, nr) != 0) return -1;
    // Section timing (QF8_PROFILE=1 only; no-ops otherwise): close the last
    // interval and fold this call's marks. The fold synchronizes the stream,
    // which the caller's residual pull does right after anyway.
    qf8_prof_mark(&S->prof, QF8_SEC_N, S->stream);
    qf8_prof_report(&S->prof, S->stream, NULL, 0);
    return 0;
}

// Batched output head: per-row output HC mixer, then ONE lm_head stream for M rows.
int qf4_stage_head_M(Qf4Stage *S, int M) {
    SETDEV(S);
    hipStream_t s = S->stream;
    if (!S->lm_head && !(M == 1 && S->i8_head)) { fprintf(stderr, "head: bf16 lm_head freed; only the M=1 int8 head is available\n"); return -1; }
    if (M == 1 && S->i8_head) {
        // M=1 int8 head: output HC on the int8 copies, then the int8 lm_head
        // (248320 x 2560, 636 MB instead of 1.27 GB bf16) into the same logits.
        qf8_i8_hc_m1(S->normed, S->hc_d, S->mixed, NULL, S->R, S->ohc_norm,
                     &S->i8_ohc_down, NULL, &S->i8_ohc_up, S->n8q, S->n8s, S->d8q, S->d8s, S->q12288,
                     NULL, NULL, S->x8q, S->x8s, s);
        Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 1; m.m[0] = &S->i8_lm_head; m.y[0] = S->logits;
        qf8_i8_gemv_multi(&m, S->x8q, S->x8s, s);
        return 0;
    }
    qf8_hc_pre_M(S->normed, S->hc_d, S->mixed, NULL, S->R,
                 S->ohc_norm, S->ohc_down, S->ohc_up, NULL, M, s);
    qf8_gemv_bf16_M(S->lm_head, S->mixed, S->logits, NVOCAB, NEMBD, M, 1.f, 0.f, s);
    return 0;
}

static int stage_run_impl(Qf4Stage *S, long pos, int want_logits) {
    SETDEV(S);
    hipStream_t s = S->stream;
    S->kv_len = pos + 1;   // KV occupancy bookkeeping
    const int tapmode = tap_active(S, pos);
    const int tapall = tapmode == 2;       // per-position taps, keyed by pos
    // Layer-specific taps only make sense on the stage that owns those layers.
    const int tap = tapmode != 0 && S->is_first;
    const int tp = (int)pos;

    // Residual ENTERING this stage. On stage 0 that is exactly
    // repeat4(embed[token]); on every later stage it is whatever the previous
    // GPU handed across the boundary, so Rout(g) vs Rin(g+1) tests the
    // handoff itself.
    if (tapmode) {
        char tg[24];
        snprintf(tg, sizeof(tg), "Rin_g%d", S->gpu);
        tap_vec(S, tg, tp, S->R, RSIZE, s);
    }
    if (tap) tap_vec(S, "embed", -1, S->R, RSIZE, s);

    for (int ll = 0; ll < S->nl; ll++) {
        int il = S->l0 + ll;
        Qf4Layer *L = &S->L[ll];
        int is_recr = ((il + 1) % 4) != 0;

        // PLE belongs before layer 1's attention hyper-connection. Its row
        // gather was staged when GPU0 accepted this token.
        if (il == 1 && S->ple) {
            if (tapall) tap_vec(S, "ple_Rin", tp, S->R, RSIZE, s);
            qf8_prof_mark(&S->prof, QF8_SEC_PLE, s);
            qf8_ple_apply(S->ple, S->R, s);
            if (tapall) {
                tap_vec(S, "ple_Rout", tp, S->R, RSIZE, s);
                tap_vec(S, "ple_emb", tp, qf8_ple_dbg(S->ple, 0), NEMBD, s);
                tap_vec(S, "ple_key", tp, qf8_ple_dbg(S->ple, 1), RSIZE, s);
                tap_vec(S, "ple_val", tp, qf8_ple_dbg(S->ple, 2), NEMBD, s);
                tap_vec(S, "ple_kn",  tp, qf8_ple_dbg(S->ple, 3), RSIZE, s);
                tap_vec(S, "ple_qn",  tp, qf8_ple_dbg(S->ple, 4), RSIZE, s);
                tap_vec(S, "ple_gate", tp, qf8_ple_dbg(S->ple, 5), HCC, s);
                tap_vec(S, "ple_gv",  tp, qf8_ple_dbg(S->ple, 6), RSIZE, s);
                tap_vec(S, "ple_nc",  tp, qf8_ple_dbg(S->ple, 7), RSIZE, s);
            }
        }

        qf8_prof_mark(&S->prof, QF8_SEC_HC_ATTN, s);
        Qf8HcW hca = { L->hc_attn_norm, L->hc_attn_down,
                       L->hc_attn_up, L->hc_attn_inject };
        qf8_hc_pre(S->normed, S->hc_d, S->mixed, S->inj, S->R,
                   hca.norm, hca.down, hca.up, hca.inject, s);
        if (tap && il == 0) {
            tap_vec(S, "a_normed", tp, S->normed, RSIZE, s);
            tap_vec(S, "a_d", tp, S->hc_d, HCL, s);
            tap_vec(S, "a_mixed", tp, S->mixed, NEMBD, s);
            tap_vec(S, "a_inj", tp, S->inj, HCC, s);
        }

        qf8_prof_mark(&S->prof, QF8_SEC_MIXER, s);
        if (is_recr) {
            Qf8GdnW g = { L->qkv, L->zgate, L->in_a, L->in_b, L->conv1d,
                          L->a_log, L->dt_bias, L->gdn_norm, L->gdn_out };
            qf8_gdn_mixer(&g, S->mixed, S->gdnS[ll], S->convring[ll],
                           S->qkv_raw, S->z6144, S->a48, S->b48,
                           S->q12288, S->y2560, s);
        } else {
            Qf8AttnW a = { L->wq, L->wk, L->wv, L->wo, L->q_norm, L->k_norm,
                           L->idx_qk, L->idx_qnorm, L->idx_knorm };
            if (tap && il == 3) {
                tap_vec(S, "qsa_in", tp, S->mixed, NEMBD, s);
            }
            qf8_qsa_mixer(&a, S->mixed, S->kc[ll], S->vc[ll], &S->idx[ll], pos,
                           S->q12288, S->k512, S->v512, S->idx640, S->attn_out8,
                           S->inv_freq, S->inv_freq_idx, S->y2560, s);
            if (tap && il == 3) {
                // indexer state: q (normed+roped), the pooled block keys, the
                // scores and the resulting selection mask. Enough to rebuild
                // the whole sparse-attention decision from the checkpoint.
                int nfull = (int)((pos + 1) >> 2);
                tap_vec(S, "idx_q", tp, S->idx640, 640, s);
                if (nfull > 0) {
                    tap_vec(S, "idx_pool", tp, S->idx[ll].pool_key,
                            nfull * QF8_IDX_D, s);
                    if (nfull > QF8_IDX_TOPK) {
                        tap_vec(S, "idx_score", tp, S->idx[ll].blk_score, nfull, s);
                        tap_u8(S, "idx_mask", tp, S->idx[ll].blk_mask, nfull, s);
                    }
                }
                tap_vec(S, "qsa_q", tp, S->q12288, 12288, s);   // post norm+rope
                tap_vec(S, "qsa_k", tp, S->k512, 512, s);
                tap_vec(S, "qsa_v", tp, S->v512, 512, s);
                tap_vec(S, "qsa_att", tp, S->attn_out8, 6144, s);
                tap_vec(S, "qsa_out", tp, S->y2560, NEMBD, s);
            }
        }
        if (tap && il == 0) {
            tap_vec(S, "mix_out", tp, S->y2560, NEMBD, s);      // GDN/QSA output
            tap_vec(S, "qkv", tp, S->qkv_raw, QF8_DINN, s);     // post conv+silu
            tap_vec(S, "z", tp, S->z6144, 6144, s);
            tap_vec(S, "a48", tp, S->a48, 48, s);
            tap_vec(S, "b48", tp, S->b48, 48, s);
        }
        qf8_prof_mark(&S->prof, QF8_SEC_INJECT1, s);
        qf8_stream_inject(S->R, S->y2560, S->inj, s);

        qf8_prof_mark(&S->prof, QF8_SEC_HC_FFN, s);
        Qf8HcW hcf = { L->hc_ffn_norm, L->hc_ffn_down,
                       L->hc_ffn_up, L->hc_ffn_inject };
        qf8_hc_pre(S->normed, S->hc_d, S->mixed, S->inj, S->R,
                   hcf.norm, hcf.down, hcf.up, hcf.inject, s);

        // --- MoE: device-resident router + dispatch (session 09) ---
        Qf4ExpCache *C = &L->cache;
        qf8_prof_mark(&S->prof, QF8_SEC_ROUTER, s);
        uint32_t seq = ++C->expected_seq;   // monotone per-layer-per-token stamp
        char mtg[32];
        if (tapmode && ll == 0) {
            snprintf(mtg, sizeof(mtg), "moe_in_L%d", il);
            tap_vec(S, mtg, tp, S->mixed, NEMBD, s);
        }
        qf8_gemv_bf16(L->router, S->mixed, S->router, NEXP, NEMBD, 1.f, 0.f, s);
        if (tapmode && ll == 0) {
            snprintf(mtg, sizeof(mtg), "router_L%d", il);
            tap_vec(S, mtg, tp, S->router, NEXP, s);
        }
        qf_rtr_topk10_launch(S->router, S->sel_ids_dev, S->wts_dev, s);
        // routing skew census: one atomicAdd per selected expert per layer.
        // Order-independent counts, not an accumulate path.
        qf_rtr_hist_launch(S->sel_ids_dev, S->exp_hist, s);
        // Fully resident: slot == expert id, so route_slot IS sel_ids. No
        // dispatch kernel, no mailbox, no LRU victim search, and below, no host
        // service call. The CPU does nothing per layer.
        const int *slot_src = S->sel_ids_dev;
        if (!C->fully_resident) {
            qf_rtr_dispatch_launch(S->sel_ids_dev, C->slot_for_expert_dev,
                                   C->expert_in_slot_dev, C->age_dev, C->clock_dev,
                                   C->route_slot_dev, C->mb_dev, seq, C->slots, s);
            slot_src = C->route_slot_dev;
        }
        // No zeroing pass: k_nvfp4_downacc_all SEEDS y2560 (it is the sole
        // writer of every row), and the shared expert adds to it afterwards.
        // Host: spin on the pinned mailbox (zero-copy), gather + DMA missed
        // experts on the transfer stream; the compute stream waits the
        // per-buffer transfer events enqueued here.
        // Everything from here to the EXPERTS mark is the HOST round trip:
        // the spin waits for the GPU to reach the dispatch kernel, so the GPU
        // is idle inside this interval while the host reacts and enqueues.
        qf8_prof_mark(&S->prof, QF8_SEC_EXPSVC, s);
        if (!C->fully_resident &&
            qf4_expcache_service(C, S->st, &S->xfer, s) != 0) return -1;
        // Routed experts, fused wave64 device-indexed path (identical for all
        // 12 layers): expert id, slot, scale_2 and weight read on device.
        qf8_prof_mark(&S->prof, QF8_SEC_EXPERTS, s);
        // All NEXPUSED experts in TWO launches instead of 20. The old loop was
        // forced to serialize because every expert shared one hidden buffer;
        // per-expert slices remove that, and the down pass keeps exactly one
        // writer per output row (no atomics, fixed expert order, deterministic).
        qf5_nvfp4_moe_all(C->w_gate, C->s_gate, C->w_up, C->s_up,
                          C->w_down, C->s_down,
                          S->mixed, S->eg, S->y2560,
                          C->s2g_dev, C->s2u_dev, C->s2d_dev, S->wts_dev,
                          S->sel_ids_dev, slot_src, NEXPUSED, s);
        // y2560 currently holds ONLY the weighted routed-expert sum.
        if (tapmode && ll == 0) {
            snprintf(mtg, sizeof(mtg), "routed_L%d", il);
            tap_vec(S, mtg, tp, S->y2560, NEMBD, s);
        }
        qf8_prof_mark(&S->prof, QF8_SEC_SHEXP, s);
        Qf8ShexpW sh = { L->shexp_gate, L->shexp_up,
                         L->shexp_down, L->shexp_gate_inp };
        qf8_shexp(&sh, S->mixed, S->eg, S->eu, S->a48, S->ed, S->y2560, s);
        // ed is the shared expert BEFORE its sigmoid gate scalar.
        if (tapmode && ll == 0) {
            snprintf(mtg, sizeof(mtg), "shared_L%d", il);
            tap_vec(S, mtg, tp, S->ed, NEMBD, s);
        }
        qf8_stream_inject(S->R, S->y2560, S->inj, s);
    }

    // Residual LEAVING this stage, before the output head consumes it.
    if (tapmode) {
        char tg[24];
        snprintf(tg, sizeof(tg), "Rout_g%d", S->gpu);
        tap_vec(S, tg, tp, S->R, RSIZE, s);
    }

    qf8_prof_mark(&S->prof, QF8_SEC_OUT_HEAD, s);
    if (S->owns_head && want_logits) {
        // output HC mixer (no inject) + lm_head. No output norm per SEMANTICS.
        // Skipped for prefill tokens (qf4_stage_run_nologits): a 248320x2560
        // GEMV per prompt token is pure waste.
        qf8_out_head(S->ohc_norm, S->ohc_down, S->ohc_up, S->lm_head,
                     S->R, S->normed, S->hc_d, S->mixed, S->x_bf16,
                     S->logits, s);
    }

    qf8_prof_mark(&S->prof, QF8_SEC_N, s);   // closes the last interval
    if (hipStreamSynchronize(s) != hipSuccess) return -1;
    // Fold this token's intervals now: the stream is already drained
    // here, so the sync inside report costs nothing extra, and the
    // marks never span the inter-token wait.
    qf8_prof_report(&S->prof, s, NULL, 0);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) { stage_err("run", "stage", e); return -1; }
    return 0;
}

int qf4_stage_run(Qf4Stage *S, long pos) {
    return stage_run_impl(S, pos, 1);           // logits-producing variant
}

int qf4_stage_run_nologits(Qf4Stage *S, long pos) {
    return stage_run_impl(S, pos, 0);           // prefill: skip lm_head
}

// Zeroes all recurrent state carried between tokens: GDN S matrices and conv
// history rings (they carry token history), KV occupancy bookkeeping (the
// cache buffers' bytes are NOT touched -- positions beyond kv_len are never
// read), and the PLE history is reseeded to EOS. Called between requests only.
int qf4_stage_reset(Qf4Stage *S) {
    if (!S) return -1;
    SETDEV(S);
    for (int ll = 0; ll < S->nl; ll++) {
        int il = S->l0 + ll;
        if (((il + 1) % 4) != 0) {
            if (S->gdnS[ll] &&
                hipMemset(S->gdnS[ll], 0, (size_t)QF8_MROWS * QF8_GDNS_STRIDE * 4) != hipSuccess)
                return -1;
            if (S->convring[ll] &&
                hipMemset(S->convring[ll], 0, (size_t)QF8_MROWS * QF8_RING_STRIDE * 4) != hipSuccess)
                return -1;
        }
        // QSA layers: no buffer clearing -- kv_len below is the only occupancy
        // bookkeeping the attention kernels honor (scores run over [0, pos)).
    }
    S->kv_len = 0;
    S->ple_hist[0] = S->ple_hist[1] = QF4_EOS;
    if (qf8_ple_reset(S->ple) != 0) return -1;
    return 0;
}

// FORK: copy row 0's resident state into rows 1..M-1 on this stage: KV [0, P) of the QSA layers, GDN state and
// conv ring of the recurrent layers, the per-row indexer pools, and (stage 0) the PLE n-gram history + conv ring.
int qf4_stage_fork_rows(Qf4Stage *S, int M, long P) {
    if (!S || M < 1 || M > QF8_MROWS || P < 0 || P > S->maxpos) return -1;
    SETDEV(S);
    hipStream_t s = S->stream;
    const size_t kvspan = (size_t)S->maxpos * NKV * HDIM;
    const size_t nb = (size_t)S->maxpos / QF8_IDX_C;
    for (int ll = 0; ll < S->nl; ll++) {
        const int il = S->l0 + ll;
        for (int r = 1; r < M; r++) {
            if (((il + 1) % 4) != 0) {
                if (S->gdnS[ll] && hipMemcpyAsync(S->gdnS[ll] + (size_t)r * QF8_GDNS_STRIDE, S->gdnS[ll], (size_t)QF8_GDNS_STRIDE * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                if (S->convring[ll] && hipMemcpyAsync(S->convring[ll] + (size_t)r * QF8_RING_STRIDE, S->convring[ll], (size_t)QF8_RING_STRIDE * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            } else {
                if (P > 0) {
                    const size_t nbytes = (size_t)P * NKV * HDIM * sizeof(qf8_kv_t);
                    if (hipMemcpyAsync(S->kc[ll] + (size_t)r * kvspan, S->kc[ll], nbytes, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                    if (hipMemcpyAsync(S->vc[ll] + (size_t)r * kvspan, S->vc[ll], nbytes, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                }
                const Qf8QsaIdx *a = &S->idx[ll * QF8_MROWS]; Qf8QsaIdx *b = &S->idx[ll * QF8_MROWS + r];
                if (hipMemcpyAsync(b->pool_sum, a->pool_sum, nb * QF8_IDX_D * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                if (hipMemcpyAsync(b->pool_key, a->pool_key, nb * QF8_IDX_D * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                if (hipMemcpyAsync(b->pool_cnt, a->pool_cnt, nb * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            }
        }
    }
    if (S->owns_embed && S->ple) {
        if (qf8_ple_fork_rows(S->ple, M, s) != 0) return -1;
        for (int r = 1; r < M; r++) { /* per-row PLE history on the stage mirrors row 0 */ }
    }
    return hipStreamSynchronize(s) == hipSuccess ? 0 : -1;
}

// COMMIT (decision/commit loop): copy row r's resident state into row 0 on this stage.
// Exact reverse of qf4_stage_fork_rows (same buffers, src/dst swapped).
int qf4_stage_commit_rows(Qf4Stage *S, int r, long P) {
    if (!S || r < 0 || r >= QF8_MROWS || P < 0 || P > S->maxpos) return -1;
    SETDEV(S);
    hipStream_t s = S->stream;
    const size_t kvspan = (size_t)S->maxpos * NKV * HDIM;
    const size_t nb = (size_t)S->maxpos / QF8_IDX_C;
    for (int ll = 0; ll < S->nl; ll++) {
        const int il = S->l0 + ll;
        if (((il + 1) % 4) != 0) {
            if (S->gdnS[ll] && hipMemcpyAsync(S->gdnS[ll], S->gdnS[ll] + (size_t)r * QF8_GDNS_STRIDE, (size_t)QF8_GDNS_STRIDE * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            if (S->convring[ll] && hipMemcpyAsync(S->convring[ll], S->convring[ll] + (size_t)r * QF8_RING_STRIDE, (size_t)QF8_RING_STRIDE * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
        } else {
            if (P > 0) {
                const size_t nbytes = (size_t)P * NKV * HDIM * sizeof(qf8_kv_t);
                if (hipMemcpyAsync(S->kc[ll], S->kc[ll] + (size_t)r * kvspan, nbytes, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
                if (hipMemcpyAsync(S->vc[ll], S->vc[ll] + (size_t)r * kvspan, nbytes, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            }
            const Qf8QsaIdx *a = &S->idx[ll * QF8_MROWS + r]; Qf8QsaIdx *b = &S->idx[ll * QF8_MROWS];
            if (hipMemcpyAsync(b->pool_sum, a->pool_sum, nb * QF8_IDX_D * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            if (hipMemcpyAsync(b->pool_key, a->pool_key, nb * QF8_IDX_D * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
            if (hipMemcpyAsync(b->pool_cnt, a->pool_cnt, nb * 4, hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
        }
    }
    if (S->owns_embed && S->ple) {
        if (qf8_ple_commit_rows(S->ple, r, s) != 0) return -1;
    }
    return hipStreamSynchronize(s) == hipSuccess ? 0 : -1;
}
size_t qf4_stage_hbm_bytes(const Qf4Stage *S) { return S ? S->hbm_bytes : 0; }

// The M-row region path addresses expert slots by expert id (slot == expert):
// it is only valid when every owned layer's cache is fully resident. A paged
// (LRU) layer must fail at init, never silently read the wrong slot.
int qf4_stage_fully_resident(const Qf4Stage *S) {
    if (!S) return 0;
    for (int ll = 0; ll < S->nl; ll++)
        if (!S->L[ll].cache.fully_resident) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// P2P handoff (session 09). Two fixed 40 KiB device buffers + events per
// stage; the mailbox slot index (pos % 2) pairs producer buffer/event with
// the consumer's wait.
// ---------------------------------------------------------------------------
const void *qf4_stage_handoff_dev(Qf4Stage *S, int slot) {
    if (!S) return NULL;
    return (const void *)S->handoff[slot & 1];
}

int qf4_stage_emit_residual(Qf4Stage *S, int slot) {
    if (!S) return -1;
    SETDEV(S);
    slot &= 1;
    // Stream-ordered: runs after this stage's 12 layers on the stage stream.
    if (hipMemcpyAsync(S->handoff[slot], S->R, RSIZE * 4,
                       hipMemcpyDeviceToDevice, S->stream) != hipSuccess)
        return -1;
    if (hipEventRecord(S->hev[slot], S->stream) != hipSuccess) return -1;
    return 0;
}

void *qf4_stage_handoff_event(Qf4Stage *S, int slot) {
    if (!S) return NULL;
    return (void *)S->hev[slot & 1];
}

int qf4_stage_recv_residual_dev(Qf4Stage *S, int src_gpu, const void *src_dev,
                                int src_slot, void *ev) {
    if (!S || !src_dev || !ev) return -1;
    SETDEV(S);
    (void)src_slot;   // informational: producer buffer index (pos % 2)
    // Stream-ordered: the peer copy starts only after the producer's emit
    // event, and this stage's subsequent kernels wait the copy.
    if (hipStreamWaitEvent(S->stream, (hipEvent_t)ev, 0) != hipSuccess) return -1;
    if (hipMemcpyPeerAsync(S->R, S->gpu, src_dev, src_gpu, RSIZE * 4,
                           S->stream) != hipSuccess)
        return -1;
    return 0;
}

// Probe + enable P2P between two cards. ROCm 5.7 on MI50: P2P works only
// with large-BAR PCIe (and xGMI where present); the caller (pipeline init)
// probes every boundary and falls back to pinned host staging if any fails.
// Dump the selection histogram: how concentrated is routing? If a small hot
// set dominates, PINNING it beats LRU. If it is flat, only full residency
// helps. LRU cannot answer this; a census can.
void qf4_stage_hist_print(Qf4Stage *S, void *outv, const char *label) {
    FILE *out = (FILE *)outv;
    if (!S || !out || !S->exp_hist) return;
    unsigned int h[NEXP];
    if (hipSetDevice(S->gpu) != hipSuccess) return;   // void fn: no SETDEV
    if (hipMemcpy(h, S->exp_hist, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess) return;
    unsigned long long tot = 0;
    for (int e = 0; e < NEXP; e++) tot += h[e];
    if (!tot) return;
    unsigned int srt[NEXP];
    for (int e = 0; e < NEXP; e++) srt[e] = h[e];
    for (int i = 1; i < NEXP; i++) {          // insertion sort, descending
        unsigned int v = srt[i]; int j = i - 1;
        while (j >= 0 && srt[j] < v) { srt[j + 1] = srt[j]; j--; }
        srt[j + 1] = v;
    }
    fprintf(out, "  %s routing census: %llu selections\n", label, tot);
    int cuts[6] = {32, 64, 128, 192, 256, 384};
    for (int c = 0; c < 6; c++) {
        unsigned long long acc = 0;
        for (int i = 0; i < cuts[c] && i < NEXP; i++) acc += srt[i];
        fprintf(out, "    hottest %3d of %d experts cover %5.1f%% of selections\n",
                cuts[c], NEXP, 100.0 * (double)acc / (double)tot);
    }
    int nz = 0;
    for (int e = 0; e < NEXP; e++) if (h[e]) nz++;
    fprintf(out, "    experts ever selected: %d of %d\n", nz, NEXP);
}

void qf4_stage_prof_print(Qf4Stage *S, void *out, const char *label) {
    if (S) qf8_prof_print(&S->prof, (FILE *)out, label);
}

int qf4_peer_available(int src_gpu, int dst_gpu) {
    int can = 0;
    if (hipDeviceCanAccessPeer(&can, src_gpu, dst_gpu) != hipSuccess) return -1;
    if (!can) return 0;
    if (hipSetDevice(src_gpu) != hipSuccess) return -1;
    hipError_t e = hipDeviceEnablePeerAccess(dst_gpu, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        hipGetLastError();   // clear; treat as unavailable
        return 0;
    }
    if (hipSetDevice(dst_gpu) != hipSuccess) return -1;
    e = hipDeviceEnablePeerAccess(src_gpu, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        hipGetLastError();
        return 0;
    }

    // ---- FUNCTIONAL round-trip test; the capability bits are not enough ----
    //
    // Measured on this box (2026-08-29): hipDeviceCanAccessPeer reports true,
    // both hipDeviceEnablePeerAccess calls succeed, large-BAR is enabled, and
    // hipMemcpyPeerAsync still returns hipSuccess WITHOUT MOVING ANY DATA.
    // The engine then ran with every downstream GPU re-reading its own stale
    // residual instead of the one handed to it: R_in(g)[t] == R_out(g)[t-1] at
    // cosine 1.0, and R_out(g-1) never arriving at all. Stage 0 was bit-exact
    // the whole time, so this looked exactly like a kernel bug for two months
    // of engine work.
    //
    // A capability probe that does not move a byte cannot detect that, so this
    // one moves bytes and reads them back. If the copy does not round-trip we
    // return 0 and the pipeline uses pinned host staging - the guaranteed
    // path. This is NEVER_AGAIN rule 1 at the transport level: the broken
    // route is not something the engine can silently select.
    int ok = 0;
    uint32_t *s_dev = NULL, *d_dev = NULL;
    const uint32_t pat = 0xA5C3F00Du;
    uint32_t back = 0;
    if (hipSetDevice(src_gpu) == hipSuccess &&
        hipMalloc((void **)&s_dev, sizeof(uint32_t)) == hipSuccess &&
        hipMemcpy(s_dev, &pat, sizeof(pat), hipMemcpyHostToDevice) == hipSuccess &&
        hipSetDevice(dst_gpu) == hipSuccess &&
        hipMalloc((void **)&d_dev, sizeof(uint32_t)) == hipSuccess &&
        hipMemset(d_dev, 0, sizeof(uint32_t)) == hipSuccess &&
        hipMemcpyPeer(d_dev, dst_gpu, s_dev, src_gpu, sizeof(uint32_t)) == hipSuccess &&
        hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(&back, d_dev, sizeof(back), hipMemcpyDeviceToHost) == hipSuccess) {
        ok = (back == pat);
    }
    if (s_dev) { hipSetDevice(src_gpu); hipFree(s_dev); }
    if (d_dev) { hipSetDevice(dst_gpu); hipFree(d_dev); }
    hipGetLastError();
    if (!ok)
        fprintf(stderr, "qf4: P2P %d->%d advertises peer access but the copy "
                        "did not round-trip (got %08x, want %08x); using pinned "
                        "host staging\n", src_gpu, dst_gpu, back, pat);
    return ok;
}

// Pinned host memory for pipeline staging (portable: usable from any card).
void *qf4_host_pinned(size_t bytes) {
    void *p = NULL;
    if (hipHostMalloc(&p, bytes, hipHostMallocPortable) != hipSuccess) return NULL;
    return p;
}
void qf4_host_pinned_free(void *p) { if (p) hipHostFree(p); }
