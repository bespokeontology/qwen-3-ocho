// qf8_moe_i8.hip - see qf8_moe_i8.h. GLM W8 execution structure (int8 dot4,
// 16-byte nontemporal weight loads, ds_swizzle reductions, no LDS staging of
// the activation) specialized to the Qwen3.8 expert shapes at M=1.
#include "qf8_moe_i8.h"
#ifndef QF5_K
#define QF5_K 10
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef QF_HOST_CHECK

typedef int i8v4 __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ i8v4 i8_ld16(const int8_t *p) {
    return *reinterpret_cast<const i8v4 *>(p);
}
static __device__ __forceinline__ i8v4 i8_ld16nt(const int8_t *p) {   // weights: read once
    return __builtin_nontemporal_load(reinterpret_cast<const i8v4 *>(p));
}
// 16 int8 x 16 int8 -> fp32 (v_dot4_i32_i8 x4)
static __device__ __forceinline__ float i8_dot16(i8v4 w, i8v4 x) {
    int d = __builtin_amdgcn_sdot4(w.x, x.x, 0, false);
    d = __builtin_amdgcn_sdot4(w.y, x.y, d, false);
    d = __builtin_amdgcn_sdot4(w.z, x.z, d, false);
    d = __builtin_amdgcn_sdot4(w.w, x.w, d, false);
    return (float)d;
}
// sum within each 32-lane half (xor 1,2,4,8,16 via ds_swizzle; no LDS, no barrier)
static __device__ __forceinline__ float i8_half_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    return v;
}
// sum over 8-lane groups (down: 8 lanes per row)
static __device__ __forceinline__ float i8_g8_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    return v;
}
static __device__ __forceinline__ float i8_half_absmax(float v) {
    float m = fabsf(v);
    m = fmaxf(m, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(m), 0x041F)));
    m = fmaxf(m, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(m), 0x081F)));
    m = fmaxf(m, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(m), 0x101F)));
    m = fmaxf(m, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(m), 0x201F)));
    m = fmaxf(m, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(m), 0x401F)));
    return m;
}

// ---- X8 activation quantizer: one 32-lane half-wave per block of 32 ---------
// n % 256 == 0 (2560 for x, nexp*640 = 6400 for the hidden rows), so every
// half-wave is fully populated and the swizzles never see an inactive lane.
__global__ void __launch_bounds__(256) k_i8_x8quant(const float *__restrict__ x,
                                                    int8_t *__restrict__ q,
                                                    float *__restrict__ s, int n) {
    const int e = blockIdx.x * 256 + threadIdx.x;
    if (e >= n) return;
    const float v = x[e];
    const float mx = i8_half_absmax(v);
    const float sc = mx > 0.f ? mx * (1.f / 127.f) : 1.f;
    int qi = (int)rintf(v / sc);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    q[e] = (int8_t)qi;
    if ((threadIdx.x & 31) == 0) s[e >> 5] = sc;
}

// ---- gate + up + SiLU*up for the top-k experts of one row ------------------
// grid (640/8, nexp), block 256 = 8 rows, one 32-lane half-wave per row.
// Lane l of a half-wave owns bytes [st*512 + l*16, +16) of the 2560-byte row
// for st = 0..4: the half-wave's 32 lanes read 512 contiguous bytes per step,
// all 10 weight loads (5 gate + 5 up) are issued before the first dot.
__global__ void __launch_bounds__(256)
k_i8_gateup_all(const int8_t *__restrict__ Wg, const float *__restrict__ Sg,
                const int8_t *__restrict__ Wu, const float *__restrict__ Su,
                const int8_t *__restrict__ xq, const float *__restrict__ xs,
                float *__restrict__ hidden, const int *__restrict__ sel_ids) {
    const int k = blockIdx.y;
    const int slot = sel_ids[k];                       // slot == expert id
    const int l = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const size_t woff = (size_t)slot * ((size_t)QF5_NFF * QF5_NEMBD) + (size_t)row * QF5_NEMBD;
    const int8_t *wg = Wg + woff + l * 16, *wu = Wu + woff + l * 16;
    i8v4 vg[5], vu[5];
    #pragma unroll
    for (int st = 0; st < 5; st++) { vg[st] = i8_ld16nt(wg + st * 512); vu[st] = i8_ld16nt(wu + st * 512); }
    float ag = 0.f, au = 0.f;
    #pragma unroll
    for (int st = 0; st < 5; st++) {
        const int e0 = st * 512 + l * 16;
        const i8v4 xv = i8_ld16(xq + e0);
        const float sc = xs[e0 >> 5];
        ag = fmaf(i8_dot16(vg[st], xv), sc, ag);
        au = fmaf(i8_dot16(vu[st], xv), sc, au);
    }
    ag = i8_half_sum(ag);
    au = i8_half_sum(au);
    if (l == 0) {
        const float g = ag * Sg[(size_t)slot * QF5_NFF + row];
        const float u = au * Su[(size_t)slot * QF5_NFF + row];
        hidden[(size_t)k * QF5_NFF + row] = g / (1.f + expf(-g)) * u;   // same SiLU form as the NVFP4 path
    }
}

// ---- down + router-weighted accumulate, all experts, y SEEDED --------------
// grid 2560/32, block 256 = 32 rows x 8 lanes. Lane l of a row owns chunks
// c = l + 8*i (i = 0..4) of the 640-byte row: 8 lanes read 128 contiguous
// bytes per i. Experts are walked in fixed order by one thread group per row
// (deterministic; one writer per output row; no atomics, no zeroing pass).
__global__ void __launch_bounds__(256)
k_i8_downacc_all(const int8_t *__restrict__ Wd, const float *__restrict__ Sd,
                 const int8_t *__restrict__ hq, const float *__restrict__ hs,
                 const float *__restrict__ wts, const int *__restrict__ sel_ids,
                 float *__restrict__ y, int nexp) {
    __shared__ int   ssl[QF5_NEXPUSED];
    __shared__ float swt[QF5_NEXPUSED];
    if (threadIdx.x < nexp) { ssl[threadIdx.x] = sel_ids[threadIdx.x]; swt[threadIdx.x] = wts[threadIdx.x]; }
    __syncthreads();
    const int l = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    float total = 0.f;
    #pragma unroll 2
    for (int k = 0; k < nexp; k++) {
        const int slot = ssl[k];
        const int8_t *wr = Wd + (size_t)slot * ((size_t)QF5_NEMBD * QF5_NFF) + (size_t)row * QF5_NFF;
        i8v4 v[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) v[i] = i8_ld16nt(wr + (l + 8 * i) * 16);
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            const int c = l + 8 * i;                                   // 16-value chunk of the hidden row
            const i8v4 hv = i8_ld16(hq + (size_t)k * QF5_NFF + c * 16);
            acc = fmaf(i8_dot16(v[i], hv), hs[k * (QF5_NFF / 32) + (c >> 1)], acc);
        }
        acc = i8_g8_sum(acc);
        total = fmaf(swt[k] * Sd[(size_t)slot * QF5_NEMBD + row], acc, total);
    }
    if (l == 0) y[row] = total;
}

// ---- one-time conversion: NVFP4 slot rows -> int8 rows + per-row scale ------
// grid (rows, slots), block 256. Value k of a row: packed byte k/2 (value 2i in
// the LOW nibble, 2i+1 in the HIGH nibble - adjacent pairs, the checkpoint's
// layout), UE4M3 scale byte k/16, times the expert's weight_scale_2.
__device__ __forceinline__ float i8_e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
    float v = e ? __uint_as_float(((e + 126u) << 23) | (m << 22)) : (m ? 0.5f : 0.f);
    return (nib & 8u) ? -v : v;
}
__device__ __forceinline__ float i8_ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}
__global__ void __launch_bounds__(256)
k_nvfp4_to_i8(const uint8_t *__restrict__ W, const uint8_t *__restrict__ S,
              const float *__restrict__ s2, int rows, int K,
              int8_t *__restrict__ Wo, float *__restrict__ So) {
    const int slot = blockIdx.y, row = blockIdx.x;
    const uint8_t *wr = W + (size_t)slot * ((size_t)rows * K / 2) + (size_t)row * (K / 2);
    const uint8_t *sr = S + (size_t)slot * ((size_t)rows * K / 16) + (size_t)row * (K / 16);
    const float sc2 = s2[slot];
    float vals[10]; int nv = 0;
    float mx = 0.f;
    for (int k = threadIdx.x; k < K && nv < 10; k += 256) {
        const uint32_t byte = wr[k >> 1];
        const uint32_t nib = (k & 1) ? (byte >> 4) : (byte & 15u);
        const float v = i8_e2m1(nib) * i8_ue4m3(sr[k >> 4]) * sc2;
        vals[nv++] = v;
        mx = fmaxf(mx, fabsf(v));
    }
    // block absmax: wave max via swizzle/bpermute, then 4 waves via LDS
    mx = i8_half_absmax(mx);
    mx = fmaxf(mx, __int_as_float(__builtin_amdgcn_ds_bpermute(((__lane_id() ^ 32) & 63) << 2, __float_as_int(mx))));
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = mx;
    __syncthreads();
    const float bm = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float scale = bm > 0.f ? bm * (1.f / 127.f) : 1.f;
    const float inv = 1.f / scale;
    int8_t *wo = Wo + (size_t)slot * ((size_t)rows * K) + (size_t)row * K;
    nv = 0;
    for (int k = threadIdx.x; k < K && nv < 10; k += 256) {
        int qi = (int)rintf(vals[nv++] * inv);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        wo[k] = (int8_t)qi;
    }
    if (threadIdx.x == 0) So[(size_t)slot * rows + row] = scale;
}

// ---- CSR (chunk) forms: one expert per blockIdx.y. gfx906 realization: the
// row's int8 weights are read ONCE into registers (5 x 16 B per lane, half-wave
// per row), the token group's X8 activations are staged ONCE per block in LDS
// (every row of the block then reads them from LDS, not from L2/HBM per row),
// the down projection writes OWNED partials part[token][k] (each written exactly
// once) that k_i8_combine sums per token - no atomics on the output rows. ----
__global__ void __launch_bounds__(256)
k_i8_gateup_csr(const int8_t *__restrict__ Wg, const float *__restrict__ Sg,
                const int8_t *__restrict__ Wu, const float *__restrict__ Su,
                const int8_t *__restrict__ xq, const float *__restrict__ xs,
                float *__restrict__ hidden, const int *__restrict__ exp_slot,
                const int *__restrict__ exp_ptr, const int *__restrict__ pair_tok,
                const int *__restrict__ n_exp_dev) {
    const int e = blockIdx.y;
    if (e >= *n_exp_dev) return;
    __shared__ int stok[QF8_I8_CSR_TX];
    __shared__ __align__(16) int8_t sx[QF8_I8_CSR_TX * QF5_NEMBD];        // token tile: TX x 2560 int8
    __shared__ float ss[QF8_I8_CSR_TX * (QF5_NEMBD / 32)];                 // its block scales
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int l = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const size_t woff = (size_t)slot * ((size_t)QF5_NFF * QF5_NEMBD) + (size_t)row * QF5_NEMBD;
    const int8_t *wg = Wg + woff + l * 16, *wu = Wu + woff + l * 16;
    i8v4 vg[5], vu[5];                                     // the row's weights, read ONCE
    #pragma unroll
    for (int st = 0; st < 5; st++) { vg[st] = i8_ld16nt(wg + st * 512); vu[st] = i8_ld16nt(wu + st * 512); }
    const float sg = Sg[(size_t)slot * QF5_NFF + row], su = Su[(size_t)slot * QF5_NFF + row];
    for (int j0 = 0; j0 < n; j0 += QF8_I8_CSR_TX) {        // token groups of TX
        const int nt = (n - j0) < QF8_I8_CSR_TX ? (n - j0) : QF8_I8_CSR_TX;
        __syncthreads();
        if (threadIdx.x < nt) stok[threadIdx.x] = pair_tok[base + j0 + threadIdx.x];
        __syncthreads();
        for (int i = threadIdx.x; i < nt * (QF5_NEMBD / 16); i += 256) {     // stage the tile: 16 B per thread-step
            const int j = i / (QF5_NEMBD / 16), c = i - j * (QF5_NEMBD / 16);
            *(i8v4 *)(sx + j * QF5_NEMBD + c * 16) = i8_ld16(xq + (size_t)stok[j] * QF5_NEMBD + c * 16);
        }
        for (int i = threadIdx.x; i < nt * (QF5_NEMBD / 32); i += 256) {
            const int j = i / (QF5_NEMBD / 32), c = i - j * (QF5_NEMBD / 32);
            ss[i] = xs[stok[j] * (QF5_NEMBD / 32) + c];
        }
        __syncthreads();
        float ag[QF8_I8_CSR_TX], au[QF8_I8_CSR_TX];
        #pragma unroll
        for (int j = 0; j < QF8_I8_CSR_TX; j++) { ag[j] = 0.f; au[j] = 0.f; }
        #pragma unroll
        for (int st = 0; st < 5; st++) {
            const int e0 = st * 512 + l * 16;
            #pragma unroll
            for (int j = 0; j < QF8_I8_CSR_TX; j++) {
                if (j < nt) {
                    const i8v4 xv = *(const i8v4 *)(sx + j * QF5_NEMBD + e0);
                    const float sc = ss[j * (QF5_NEMBD / 32) + (e0 >> 5)];
                    ag[j] = fmaf(i8_dot16(vg[st], xv), sc, ag[j]);
                    au[j] = fmaf(i8_dot16(vu[st], xv), sc, au[j]);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < QF8_I8_CSR_TX; j++) {
            if (j < nt) {
                float g = i8_half_sum(ag[j]) * sg, u = i8_half_sum(au[j]) * su;
                if (l == 0) hidden[(size_t)(base + j0 + j) * QF5_NFF + row] = g / (1.f + expf(-g)) * u;
            }
        }
    }
}
#define QF8_I8_CSR_TXD 16                                  // pairs per staged hidden tile (10 KB)
__global__ void __launch_bounds__(256)
k_i8_downacc_csr(const int8_t *__restrict__ Wd, const float *__restrict__ Sd,
                 const int8_t *__restrict__ hq, const float *__restrict__ hs,
                 const float *__restrict__ pair_wt, const int *__restrict__ pair_tok,
                 const int *__restrict__ exp_slot, const int *__restrict__ exp_ptr,
                 const int *__restrict__ n_exp_dev, const int *__restrict__ sel,
                 float *__restrict__ part) {
    const int e = blockIdx.y;
    if (e >= *n_exp_dev) return;
    __shared__ __align__(16) int8_t sh[QF8_I8_CSR_TXD * QF5_NFF];
    __shared__ float shs[QF8_I8_CSR_TXD * (QF5_NFF / 32)];
    const int slot = exp_slot[e], base = exp_ptr[e], n = exp_ptr[e + 1] - base;
    const int l = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int8_t *wr = Wd + (size_t)slot * ((size_t)QF5_NEMBD * QF5_NFF) + (size_t)row * QF5_NFF;
    i8v4 v[5];                                             // the row's weights, read ONCE
    #pragma unroll
    for (int i = 0; i < 5; i++) v[i] = i8_ld16nt(wr + (l + 8 * i) * 16);
    const float sd = Sd[(size_t)slot * QF5_NEMBD + row];
    for (int j0 = 0; j0 < n; j0 += QF8_I8_CSR_TXD) {
        const int nt = (n - j0) < QF8_I8_CSR_TXD ? (n - j0) : QF8_I8_CSR_TXD;
        __syncthreads();
        for (int i = threadIdx.x; i < nt * (QF5_NFF / 16); i += 256) {
            const int j = i / (QF5_NFF / 16), c = i - j * (QF5_NFF / 16);
            *(i8v4 *)(sh + j * QF5_NFF + c * 16) = i8_ld16(hq + (size_t)(base + j0 + j) * QF5_NFF + c * 16);
        }
        for (int i = threadIdx.x; i < nt * (QF5_NFF / 32); i += 256) {
            const int j = i / (QF5_NFF / 32), c = i - j * (QF5_NFF / 32);
            shs[i] = hs[(base + j0 + j) * (QF5_NFF / 32) + c];
        }
        __syncthreads();
        for (int j = 0; j < nt; j++) {
            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                const int c = l + 8 * i;
                acc = fmaf(i8_dot16(v[i], *(const i8v4 *)(sh + j * QF5_NFF + c * 16)), shs[j * (QF5_NFF / 32) + (c >> 1)], acc);
            }
            acc = i8_g8_sum(acc);
            if (l == 0) {
                const int pair = base + j0 + j, tok = pair_tok[pair];
                int k = 0;
                for (int kk = 0; kk < QF5_K; kk++) if (sel[tok * QF5_K + kk] == slot) { k = kk; break; }
                part[((size_t)tok * QF5_K + k) * QF5_NEMBD + row] = pair_wt[pair] * sd * acc;
            }
        }
    }
}
// y[t] = sum over the token's 10 owned partials. grid T, block 256.
__global__ void k_i8_combine(const float *__restrict__ part, float *__restrict__ y) {
    const int t = blockIdx.x;
    const float *pt = part + (size_t)t * QF5_K * QF5_NEMBD;
    for (int i = threadIdx.x; i < QF5_NEMBD; i += 256) {
        float a = 0.f;
        #pragma unroll
        for (int k = 0; k < QF5_K; k++) a += pt[(size_t)k * QF5_NEMBD + i];
        y[(size_t)t * QF5_NEMBD + i] = a;
    }
}
// ---- routing inversion for the chunk: ONE block, thread = expert (512 = 8 waves).
// sel/wt [T][10] -> exp_slot[n] ascending, exp_ptr[n+1], pair_tok/pair_wt in
// (expert, token) order; counts[0] = n_exp, counts[1] = n_pairs. Deterministic.
// Replaces the M=8 tier's one-wave k_m8_build_csr, which cost 80-100 ms per
// layer at 2560 pairs (the whole "expert" section of the chunk was this kernel).
__global__ void __launch_bounds__(512)
k_i8_csr_build(const int *__restrict__ sel, const float *__restrict__ wt, int npairs,
               int *__restrict__ exp_slot, int *__restrict__ exp_ptr, int *__restrict__ pair_tok,
               float *__restrict__ pair_wt, int *__restrict__ counts) {
    __shared__ int cnt[512], red[512];
    const int e = threadIdx.x;
    int c = 0;
    for (int p = 0; p < npairs; p++) c += (sel[p] == e);
    cnt[e] = c; red[e] = c;
    __syncthreads();
    for (int st = 1; st < 512; st <<= 1) {                 // Hillis-Steele inclusive scan in LDS
        const int v = (e >= st) ? red[e - st] : 0;
        __syncthreads();
        red[e] += v;
        __syncthreads();
    }
    const int incl = red[e], off = incl - c;
    int nz = 0;
    for (int i = 0; i < e; i++) nz += (cnt[i] > 0);
    if (c > 0) {
        exp_slot[nz] = e;                                  // slot == expert (static resident map)
        exp_ptr[nz] = off;
        int w = off;
        for (int p = 0; p < npairs; p++)
            if (sel[p] == e) { pair_tok[w] = p / QF5_K; pair_wt[w] = wt[p]; w++; }
    }
    if (e == 511) {
        int n = 0; for (int i = 0; i < 512; i++) n += (cnt[i] > 0);
        exp_ptr[n] = incl; counts[0] = n; counts[1] = incl;
    }
}
extern "C" void qf8_i8_csr_build(const int *sel, const float *wt, int T, int *exp_slot, int *exp_ptr,
                                 int *pair_tok, float *pair_wt, int *counts, hipStream_t s) {
    k_i8_csr_build<<<1, 512, 0, s>>>(sel, wt, T * QF5_K, exp_slot, exp_ptr, pair_tok, pair_wt, counts);
}
// Surgical sub-split of the chunk expert section (QF_REGION_TIMING): events around
// the four launches, accumulated per process, printed/reset by qf8_i8_chunk_prof_print.
static int g_i8p_on = -1; static hipEvent_t g_i8p_ev[5]; static double g_i8p_ms[4]; static int g_i8p_n = 0;
extern "C" void qf8_i8_chunk_prof_print(FILE *out, const char *tag) {
    if (g_i8p_on != 1 || !g_i8p_n) return;
    fprintf(out, "[i8exp %s] %d calls: gateup %.1f hquant %.1f down %.1f combine %.1f ms\n", tag, g_i8p_n,
            g_i8p_ms[0], g_i8p_ms[1], g_i8p_ms[2], g_i8p_ms[3]);
    g_i8p_ms[0] = g_i8p_ms[1] = g_i8p_ms[2] = g_i8p_ms[3] = 0.0; g_i8p_n = 0;
}
extern "C" void qf8_i8_experts_chunk(const Qf4ExpCache *c, const int8_t *xq, const float *xs,
                                     float *hidden, int8_t *h8q, float *h8s,
                                     const int *exp_slot, const int *exp_ptr, const int *pair_tok,
                                     const float *pair_wt, const int *n_exp_dev, int n_exp_max,
                                     int n_pairs_max, const int *sel, float *part, int T, float *y, hipStream_t s) {
    if (g_i8p_on < 0) { g_i8p_on = getenv("QF_REGION_TIMING") ? 1 : 0; if (g_i8p_on) for (int i = 0; i < 5; i++) hipEventCreate(&g_i8p_ev[i]); }
    const int pr = g_i8p_on == 1;
    if (pr) hipEventRecord(g_i8p_ev[0], s);
    k_i8_gateup_csr<<<dim3(QF5_NFF / 8, n_exp_max), 256, 0, s>>>(c->i8_gate, c->i8s_gate, c->i8_up, c->i8s_up,
        xq, xs, hidden, exp_slot, exp_ptr, pair_tok, n_exp_dev);
    if (pr) hipEventRecord(g_i8p_ev[1], s);
    const int n = n_pairs_max * QF5_NFF;                          // multiple of 256 (640 = 2.5 x 256 -> pairs x 640)
    k_i8_x8quant<<<(n + 255) / 256, 256, 0, s>>>(hidden, h8q, h8s, n);
    if (pr) hipEventRecord(g_i8p_ev[2], s);
    k_i8_downacc_csr<<<dim3(QF5_NEMBD / 32, n_exp_max), 256, 0, s>>>(c->i8_down, c->i8s_down, h8q, h8s,
        pair_wt, pair_tok, exp_slot, exp_ptr, n_exp_dev, sel, part);
    if (pr) hipEventRecord(g_i8p_ev[3], s);
    k_i8_combine<<<T, 256, 0, s>>>(part, y);
    if (pr) {
        hipEventRecord(g_i8p_ev[4], s); hipEventSynchronize(g_i8p_ev[4]);
        for (int i = 0; i < 4; i++) { float ms = 0.f; hipEventElapsedTime(&ms, g_i8p_ev[i], g_i8p_ev[i + 1]); g_i8p_ms[i] += ms; }
        g_i8p_n++;
    }
}
extern "C" int qf8_i8_build_slot(Qf4ExpCache *c, int src_slot, int dst_slot, int expert, hipStream_t s) {
    if (!c || !c->i8_gate || !c->w_gate || !c->s2g_dev) return -1;
    const size_t gw = (size_t)QF5_NFF * QF5_NEMBD;            // element count, all three matrices
    const size_t so = gw / 16;
    k_nvfp4_to_i8<<<dim3(QF5_NFF, 1), 256, 0, s>>>(c->w_gate + (size_t)src_slot * (gw / 2), c->s_gate + (size_t)src_slot * so,
        c->s2g_dev + expert, QF5_NFF, QF5_NEMBD, c->i8_gate + (size_t)dst_slot * gw, c->i8s_gate + (size_t)dst_slot * QF5_NFF);
    k_nvfp4_to_i8<<<dim3(QF5_NFF, 1), 256, 0, s>>>(c->w_up + (size_t)src_slot * (gw / 2), c->s_up + (size_t)src_slot * so,
        c->s2u_dev + expert, QF5_NFF, QF5_NEMBD, c->i8_up + (size_t)dst_slot * gw, c->i8s_up + (size_t)dst_slot * QF5_NFF);
    k_nvfp4_to_i8<<<dim3(QF5_NEMBD, 1), 256, 0, s>>>(c->w_down + (size_t)src_slot * (gw / 2), c->s_down + (size_t)src_slot * so,
        c->s2d_dev + expert, QF5_NEMBD, QF5_NFF, c->i8_down + (size_t)dst_slot * gw, c->i8s_down + (size_t)dst_slot * QF5_NEMBD);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
extern "C" int qf8_i8_build(Qf4ExpCache *c, hipStream_t s) {
    if (!c || !c->i8_gate || !c->w_gate || !c->s2g_dev) return -1;
    const int ns = c->slots;
    k_nvfp4_to_i8<<<dim3(QF5_NFF, ns), 256, 0, s>>>(c->w_gate, c->s_gate, c->s2g_dev, QF5_NFF, QF5_NEMBD, c->i8_gate, c->i8s_gate);
    k_nvfp4_to_i8<<<dim3(QF5_NFF, ns), 256, 0, s>>>(c->w_up,   c->s_up,   c->s2u_dev, QF5_NFF, QF5_NEMBD, c->i8_up,   c->i8s_up);
    k_nvfp4_to_i8<<<dim3(QF5_NEMBD, ns), 256, 0, s>>>(c->w_down, c->s_down, c->s2d_dev, QF5_NEMBD, QF5_NFF, c->i8_down, c->i8s_down);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
extern "C" void qf8_i8_gateup_m1(const Qf4ExpCache *c, const float *x, int8_t *x8q, float *x8s,
                                 float *hidden, const int *sel_ids, int nexp, int quant_x, hipStream_t s) {
    if (quant_x) k_i8_x8quant<<<QF5_NEMBD / 256, 256, 0, s>>>(x, x8q, x8s, QF5_NEMBD);
    k_i8_gateup_all<<<dim3(QF5_NFF / 8, nexp), 256, 0, s>>>(c->i8_gate, c->i8s_gate, c->i8_up, c->i8s_up,
                                                             x8q, x8s, hidden, sel_ids);
}
extern "C" void qf8_i8_down_m1(const Qf4ExpCache *c, const float *hidden, int8_t *h8q, float *h8s,
                               const float *wts, const int *sel_ids, float *y, int nexp, hipStream_t s) {
    const int n = nexp * QF5_NFF;                                   // 6400 = 25 x 256
    k_i8_x8quant<<<(n + 255) / 256, 256, 0, s>>>(hidden, h8q, h8s, n);
    k_i8_downacc_all<<<QF5_NEMBD / 32, 256, 0, s>>>(c->i8_down, c->i8s_down, h8q, h8s, wts, sel_ids, y, nexp);
}

#else   // QF_HOST_CHECK: host-only type check
extern "C" int qf8_i8_build(Qf4ExpCache *c, hipStream_t s) { (void)c; (void)s; return 0; }
extern "C" int qf8_i8_build_slot(Qf4ExpCache *c, int a, int b, int e, hipStream_t s) { (void)c;(void)a;(void)b;(void)e;(void)s; return 0; }
extern "C" void qf8_i8_experts_chunk(const Qf4ExpCache *c, const int8_t *xq, const float *xs, float *hidden, int8_t *h8q, float *h8s,
                                     const int *a, const int *b, const int *d, const float *w, const int *n, int m, int p, float *y, hipStream_t s) {
    (void)c;(void)xq;(void)xs;(void)hidden;(void)h8q;(void)h8s;(void)a;(void)b;(void)d;(void)w;(void)n;(void)m;(void)p;(void)y;(void)s; }
extern "C" void qf8_i8_gateup_m1(const Qf4ExpCache *c, const float *x, int8_t *x8q, float *x8s,
                                 float *hidden, const int *sel_ids, int nexp, int quant_x, hipStream_t s) {
    (void)c;(void)x;(void)x8q;(void)x8s;(void)hidden;(void)sel_ids;(void)nexp;(void)quant_x;(void)s; }
extern "C" void qf8_i8_down_m1(const Qf4ExpCache *c, const float *hidden, int8_t *h8q, float *h8s,
                               const float *wts, const int *sel_ids, float *y, int nexp, hipStream_t s) {
    (void)c;(void)hidden;(void)h8q;(void)h8s;(void)wts;(void)sel_ids;(void)y;(void)nexp;(void)s; }
#endif
