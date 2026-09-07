// qf8_moe_fp8.hip - gfx906 production FP8 routed-expert MoE. See qf8_moe_fp8.h.
#include "qf8_moe_fp8.h"
#include <math.h>
#include <string.h>

#ifndef QF_HOST_CHECK
#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#define __shfl_sync(mask, value, lane)        __shfl((value), (lane), warpSize)
#endif
#define F8_LAUNCH(k, g, b, sh, s, ...) k<<<g, b, sh, s>>>(__VA_ARGS__)
#else
#define F8_LAUNCH(k, g, b, sh, s, ...) do { (void)(s); } while (0)
#endif

#define F8_WMASK 0xFFFFFFFFFFFFFFFFULL

// ---- E4M3 <-> float (no gfx906 hardware; exact integer bit manipulation) ----
__device__ __host__ __forceinline__ float f8_to_f(uint8_t b) {
    uint32_t x = b, sgn = (x & 0x80u) << 24, e = (x >> 3) & 0xFu, m = x & 0x7u;
    if (e == 0) { float v = (float)m * 0.001953125f;      // 2^-9 subnormal step
                  uint32_t u; __builtin_memcpy(&u, &v, 4); u |= sgn;
                  __builtin_memcpy(&v, &u, 4); return v; }
    uint32_t u = sgn | ((e + 120u) << 23) | (m << 20);
    float f; __builtin_memcpy(&f, &u, 4); return f;
}
__host__ __forceinline__ uint8_t f_to_f8(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    uint32_t sgn = (u >> 24) & 0x80u;
    int e = (int)((u >> 23) & 0xFFu) - 127;
    uint32_t m = u & 0x7FFFFFu;
    if (e < -9) return (uint8_t)sgn;
    if (e >= 8) return (uint8_t)(sgn | 0x7Eu);
    if (e < -6) { int sh = -6 - e; uint32_t mm = (m | 0x800000u) >> (sh + 20);
                  return (uint8_t)(sgn | (mm & 0x7u)); }
    uint32_t mm = m >> 20, rem = m & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (mm & 1u))) {
        mm++; if (mm == 8u) { mm = 0; e++; if (e >= 8) return (uint8_t)(sgn | 0x7Eu); } }
    return (uint8_t)(sgn | (uint32_t)((e + 7) << 3) | (mm & 0x7u));
}

// ---- device routing inversion: top-k -> expert descriptors -----------------
// One block. Ascending expert id => deterministic descriptor order, a pure
// function of the routing (the ordering law from our GLM dispatch).
__global__ void k8_fp8_build_desc(const int *__restrict__ sel, const float *__restrict__ wt,
                                  int M, int K, int slot_base,
                                  Qf8MoeDesc *__restrict__ desc, int *__restrict__ ctl) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int n = 0, prev = -1;
    for (;;) {
        int best = 0x7FFFFFFF;
        for (int t = 0; t < M; t++)
            for (int k = 0; k < K; k++) {
                int e = sel[t * K + k];
                if (e > prev && e < best) best = e;
            }
        if (best == 0x7FFFFFFF) break;
        Qf8MoeDesc d; d.slot = slot_base + best; d.nrow = 0;
        for (int t = 0; t < M && d.nrow < QF8_FP8_MAXROW; t++)
            for (int k = 0; k < K; k++)
                if (sel[t * K + k] == best) { d.row[d.nrow] = t; d.wt[d.nrow] = wt[t * K + k]; d.nrow++; break; }
        desc[n++] = d;
        prev = best;
    }
    ctl[0] = n;          // descriptor count
    ctl[1] = 0;          // gate/up work cursor (pulled by k8_fp8_gateup)
    ctl[2] = 0;          // down work cursor   (pulled by k8_fp8_downacc)
}

// ---- fused Gate+Up+SiLU*Up, persistent work queue ---------------------------
// Work item = (descriptor, output-row tile). A fixed grid pulls items with an
// atomic counter until the layer's routed work is gone: uneven routing cannot
// idle CUs, and a tiny expert never becomes its own launch.
// Lane L owns chunk L of every weight row (40 B at in=2560) and ONE scale, so
// the inner loop is pure vector loads with no address arithmetic.
// The work cursor lives in the caller's ctl block (ctl[1]), zeroed by
// k8_fp8_build_desc on the same stream: no in-kernel reset, no module global.
__global__ void k8_fp8_gateup(const uint8_t *__restrict__ Wg, const float *__restrict__ Sg,
                              const uint8_t *__restrict__ Wu, const float *__restrict__ Su,
                              const float *__restrict__ x, float *__restrict__ hidden,
                              const Qf8MoeDesc *__restrict__ desc, int *__restrict__ ctl,
                              int nemb, int nff) {
    const int CH = nemb / QF8_WAVE;                 // bytes per lane per row (40)
    const int tiles = nff / QF8_FP8_RPT;
    const int total = ctl[0] * tiles;
    const int wave = threadIdx.x / QF8_WAVE, lane = threadIdx.x % QF8_WAVE;
    const int nwave = blockDim.x / QF8_WAVE;
    for (;;) {
        __shared__ int base;
        if (threadIdx.x == 0) base = atomicAdd(&ctl[1], nwave);
        __syncthreads();
        int item = base + wave;
        if (item >= total) return;
        const Qf8MoeDesc d = desc[item / tiles];
        const int row = (item % tiles) * QF8_FP8_RPT;
        const size_t wo = (size_t)d.slot * nff * nemb, so = (size_t)d.slot * nff * QF8_WAVE;
        #pragma unroll
        for (int rr = 0; rr < QF8_FP8_RPT; rr++) {
            const int orow = row + rr;
            const uint8_t *wg = Wg + wo + (size_t)orow * nemb + (size_t)lane * CH;
            const uint8_t *wu = Wu + wo + (size_t)orow * nemb + (size_t)lane * CH;
            const float sgv = Sg[so + (size_t)orow * QF8_WAVE + lane];
            const float suv = Su[so + (size_t)orow * QF8_WAVE + lane];
            float ag[QF8_FP8_MAXROW], au[QF8_FP8_MAXROW];
            #pragma unroll
            for (int t = 0; t < QF8_FP8_MAXROW; t++) { ag[t] = 0.f; au[t] = 0.f; }
            for (int c = 0; c < CH; c += 4) {       // uint4 = 4 E4M3 x 4 lanes-worth
                uint32_t pg = *(const uint32_t *)(wg + c);
                uint32_t pu = *(const uint32_t *)(wu + c);
                float gv[4], uv[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    gv[j] = f8_to_f((uint8_t)(pg >> (8 * j)));
                    uv[j] = f8_to_f((uint8_t)(pu >> (8 * j)));
                }
                const int xb = lane * CH + c;
                #pragma unroll
                for (int t = 0; t < QF8_FP8_MAXROW; t++) {
                    if (t < d.nrow) {
                        const float4 xv = *(const float4 *)(x + (size_t)d.row[t] * nemb + xb);
                        ag[t] += gv[0] * xv.x + gv[1] * xv.y + gv[2] * xv.z + gv[3] * xv.w;
                        au[t] += uv[0] * xv.x + uv[1] * xv.y + uv[2] * xv.z + uv[3] * xv.w;
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < QF8_FP8_MAXROW; t++) {
                if (t < d.nrow) {
                    float g = ag[t] * sgv, u = au[t] * suv;
                    #pragma unroll
                    for (int off = 32; off; off >>= 1) {
                        g += __shfl_down_sync(F8_WMASK, g, off);
                        u += __shfl_down_sync(F8_WMASK, u, off);
                    }
                    if (lane == 0)                       // fused SiLU * Up, no round trip
                        hidden[((size_t)(item / tiles) * QF8_FP8_MAXROW + t) * nff + orow] =
                            (g / (1.f + __expf(-g))) * u;
                }
            }
        }
    }
}

// ---- fused Down + router-weighted accumulate --------------------------------
__global__ void k8_fp8_downacc(const uint8_t *__restrict__ Wd, const float *__restrict__ Sd,
                               const float *__restrict__ hidden, float *__restrict__ y,
                               const Qf8MoeDesc *__restrict__ desc, int *__restrict__ ctl,
                               int nemb, int nff) {
    const int CH = nff / QF8_WAVE;                   // 10 bytes/lane at nff=640
    const int tiles = nemb / QF8_FP8_RPT;
    const int total = ctl[0] * tiles;
    const int wave = threadIdx.x / QF8_WAVE, lane = threadIdx.x % QF8_WAVE;
    const int nwave = blockDim.x / QF8_WAVE;
    for (;;) {
        __shared__ int base;
        if (threadIdx.x == 0) base = atomicAdd(&ctl[2], nwave);
        __syncthreads();
        int item = base + wave;
        if (item >= total) return;
        const int di = item / tiles;
        const Qf8MoeDesc d = desc[di];
        const int row = (item % tiles) * QF8_FP8_RPT;
        const size_t wo = (size_t)d.slot * nemb * nff, so = (size_t)d.slot * nemb * QF8_WAVE;
        #pragma unroll
        for (int rr = 0; rr < QF8_FP8_RPT; rr++) {
            const int orow = row + rr;
            const uint8_t *wd = Wd + wo + (size_t)orow * nff + (size_t)lane * CH;
            const float sdv = Sd[so + (size_t)orow * QF8_WAVE + lane];
            float acc[QF8_FP8_MAXROW];
            #pragma unroll
            for (int t = 0; t < QF8_FP8_MAXROW; t++) acc[t] = 0.f;
            for (int c = 0; c < CH; c += 2) {
                uint16_t pk = *(const uint16_t *)(wd + c);
                float w0 = f8_to_f((uint8_t)pk), w1 = f8_to_f((uint8_t)(pk >> 8));
                const int hb = lane * CH + c;
                #pragma unroll
                for (int t = 0; t < QF8_FP8_MAXROW; t++)
                    if (t < d.nrow) {
                        const float *h = hidden + ((size_t)di * QF8_FP8_MAXROW + t) * nff + hb;
                        acc[t] += w0 * h[0] + w1 * h[1];
                    }
            }
            #pragma unroll
            for (int t = 0; t < QF8_FP8_MAXROW; t++) {
                if (t < d.nrow) {
                    float a = acc[t] * sdv;
                    #pragma unroll
                    for (int off = 32; off; off >>= 1) a += __shfl_down_sync(F8_WMASK, a, off);
                    if (lane == 0)                    // route weight applied here: no combine pass
                        atomicAdd(&y[(size_t)d.row[t] * nemb + orow], d.wt[t] * a);
                }
            }
        }
    }
}

// ---- launchers --------------------------------------------------------------
extern "C" void qf8_fp8_build_desc(const int *sel_dev, const float *wt_dev, int M, int K,
                                   int slot_base, Qf8MoeDesc *desc, int *ctl, hipStream_t s) {
    F8_LAUNCH(k8_fp8_build_desc, 1, 64, 0, s, sel_dev, wt_dev, M, K, slot_base, desc, ctl);
}
extern "C" void qf8_fp8_gateup(const uint8_t *Wg, const float *Sg, const uint8_t *Wu, const float *Su,
                               const float *x, float *hidden, const Qf8MoeDesc *desc,
                               int *ctl, hipStream_t s) {
    F8_LAUNCH(k8_fp8_gateup, QF8_FP8_WGS, 256, 0, s, Wg, Sg, Wu, Su, x, hidden, desc, ctl,
              QF8_NEMBD, QF8_NFF);
}
extern "C" void qf8_fp8_downacc(const uint8_t *Wd, const float *Sd, const float *hidden,
                                float *y, const Qf8MoeDesc *desc, int *ctl, hipStream_t s) {
    F8_LAUNCH(k8_fp8_downacc, QF8_FP8_WGS, 256, 0, s, Wd, Sd, hidden, y, desc, ctl,
              QF8_NEMBD, QF8_NFF);
}

// ---- offline packer: NVFP4 checkpoint form -> lane-chunk E4M3 + scales -------
// One scale per (row, lane) chosen from that chunk's max |w|, so the decode
// kernel loads exactly one scale and never does block arithmetic.
extern "C" int qf8_fp8_pack_expert(const uint8_t *nv_w, const uint8_t *nv_s, float s2,
                                   int rows, int in, uint8_t *w8, float *sc) {
    const int CH = in / QF8_WAVE;
    for (int r = 0; r < rows; r++) {
        for (int l = 0; l < QF8_WAVE; l++) {
            float mx = 0.f; float tmp[64];
            for (int c = 0; c < CH; c++) {
                int idx = l * CH + c;
                uint8_t byte = nv_w[(size_t)r * (in / 2) + idx / 2];
                uint32_t nib = (idx & 1) ? (byte >> 4) : (byte & 15u);
                uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
                float v = e ? __builtin_bit_cast(float, ((e + 126u) << 23) | (m << 22))
                            : (m ? 0.5f : 0.f);
                if (nib & 8u) v = -v;
                uint8_t sb = nv_s[(size_t)r * (in / 16) + idx / 16];
                uint32_t se = sb >> 3, sm = sb & 7u;
                float bs = se ? __builtin_bit_cast(float, ((se + 120u) << 23) | (sm << 20))
                              : (float)sm * 0.001953125f;
                v *= bs * s2;
                tmp[c] = v;
                float a = v < 0 ? -v : v; if (a > mx) mx = a;
            }
            float scale = mx > 0.f ? mx / 448.f : 1.f;   // E4M3 max magnitude
            sc[(size_t)r * QF8_WAVE + l] = scale;
            for (int c = 0; c < CH; c++)
                w8[(size_t)r * in + l * CH + c] = f_to_f8(tmp[c] / scale);
        }
    }
    return 0;
}
