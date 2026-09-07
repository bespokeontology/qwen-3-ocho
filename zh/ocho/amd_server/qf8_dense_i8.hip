// qf8_dense_i8.hip - see qf8_dense_i8.h.
#include "qf8_dense_i8.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef QF_HOST_CHECK

typedef int d8v4 __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ d8v4 d8_ld16(const int8_t *p) { return *reinterpret_cast<const d8v4 *>(p); }
static __device__ __forceinline__ d8v4 d8_ld16nt(const int8_t *p) {
    return __builtin_nontemporal_load(reinterpret_cast<const d8v4 *>(p));
}
static __device__ __forceinline__ float d8_dot16(d8v4 w, d8v4 x) {
    int d = __builtin_amdgcn_sdot4(w.x, x.x, 0, false);
    d = __builtin_amdgcn_sdot4(w.y, x.y, d, false);
    d = __builtin_amdgcn_sdot4(w.z, x.z, d, false);
    d = __builtin_amdgcn_sdot4(w.w, x.w, d, false);
    return (float)d;
}
template <int XM> static __device__ __forceinline__ float d8_swz(float v) {   // xor XM within a 32-lane half
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x001F | (XM << 10)));
}
template <int LPR> static __device__ __forceinline__ float d8_group_sum(float v) {
    if (LPR >= 2)  v += d8_swz<1>(v);
    if (LPR >= 4)  v += d8_swz<2>(v);
    if (LPR >= 8)  v += d8_swz<4>(v);
    if (LPR >= 16) v += d8_swz<8>(v);
    if (LPR >= 32) v += d8_swz<16>(v);
    if (LPR >= 64) v += __int_as_float(__builtin_amdgcn_ds_bpermute(((__lane_id() ^ 32) & 63) << 2, __float_as_int(v)));
    return v;
}
static __device__ __forceinline__ float d8_half_absmax(float v) {
    float m = fabsf(v);
    m = fmaxf(m, d8_swz<1>(m)); m = fmaxf(m, d8_swz<2>(m)); m = fmaxf(m, d8_swz<4>(m));
    m = fmaxf(m, d8_swz<8>(m)); m = fmaxf(m, d8_swz<16>(m));
    return m;
}

// ---- bf16 -> int8 per-row (GLM k_w8_quant) -----------------------------------
static __device__ __forceinline__ float d8_bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__global__ void __launch_bounds__(256) k_d8_quant(const uint16_t *__restrict__ W, int8_t *__restrict__ w8,
                                                  float *__restrict__ sc, int rows, int K) {
    __shared__ float red[256];
    const int r = blockIdx.x; if (r >= rows) return;
    const int t = threadIdx.x;
    const uint16_t *wr = W + (size_t)r * K;
    float mx = 0.f;
    for (int k = t; k < K; k += 256) mx = fmaxf(mx, fabsf(d8_bf(wr[k])));
    red[t] = mx; __syncthreads();
    for (int s = 128; s; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    const float scl = red[0] > 0.f ? red[0] / 127.f : 1.f;
    const float inv = 1.f / scl;
    if (t == 0) sc[r] = scl;
    int8_t *o = w8 + (size_t)r * K;
    for (int k = t; k < K; k += 256) {
        float q = rintf(d8_bf(wr[k]) * inv);
        o[k] = (int8_t)fminf(127.f, fmaxf(-127.f, q));
    }
}

// ---- X8 activation quantizer (one 32-lane half-wave per 32-block) ----------
__global__ void __launch_bounds__(256) k_d8_x8(const float *__restrict__ x, int8_t *__restrict__ q,
                                               float *__restrict__ s, int n) {
    const int e = blockIdx.x * 256 + threadIdx.x;
    if (e >= n) return;                       // n % 32 == 0: whole half-waves retire together
    const float v = x[e];
    const float mx = d8_half_absmax(v);
    const float sc = mx > 0.f ? mx * (1.f / 127.f) : 1.f;
    int qi = (int)rintf(v / sc);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    q[e] = (int8_t)qi;
    if ((threadIdx.x & 31) == 0) s[e >> 5] = sc;
}

// ---- the GEMV: LPR lanes per row, NST steps of 16 bytes per lane -------------
struct D8Desc { const int8_t *w; const float *s; float *y; int rows; int act; };
struct D8Batch { D8Desc d[4]; int n; int K; };
template <int LPR, int NST>
__global__ void __launch_bounds__(256)
k_d8_gemv(D8Batch b, const int8_t *__restrict__ xq, const float *__restrict__ xs) {
    constexpr int RPB = 256 / LPR;
    const D8Desc d = b.d[blockIdx.y];
    const int l = threadIdx.x % LPR;
    const int row = blockIdx.x * RPB + threadIdx.x / LPR;
    if (row >= d.rows) return;                // shorter descriptors early-out per block
    const int8_t *wr = d.w + (size_t)row * b.K + l * 16;
    d8v4 v[NST];
    #pragma unroll
    for (int st = 0; st < NST; st++) v[st] = d8_ld16nt(wr + st * (LPR * 16));
    float acc = 0.f;
    #pragma unroll
    for (int st = 0; st < NST; st++) {
        const int e0 = st * (LPR * 16) + l * 16;
        acc = fmaf(d8_dot16(v[st], d8_ld16(xq + e0)), xs[e0 >> 5], acc);
    }
    acc = d8_group_sum<LPR>(acc);
    if (l == 0) {
        float v = acc * d.s[row];
        if (d.act == 1)      v = (v * 0.25f) / (1.f + expf(-v * 0.25f));   // silu(v/4)
        else if (d.act == 2) v = 2.f / (1.f + expf(-v * 0.25f));          // 2*sigmoid(v/4)
        d.y[row] = v;
    }
}

// ---- hyper-connection pieces ------------------------------------------------
// zero-centered grouped RMSNorm over each 2560 stream: normed = R * rstd * (1 + w),
// fp32 out AND X8 out. Block = one stream (2560 = 10 x 256 elements per thread).
__global__ void __launch_bounds__(256) k_d8_hc_norm_x8(float *__restrict__ R,
        const uint16_t *__restrict__ w, float *__restrict__ normed,
        int8_t *__restrict__ q, float *__restrict__ sc,
        const float *__restrict__ y, const float *__restrict__ inj) {
    {   // rows form (chunk): position = blockIdx.y; the M=1 launch is y = 0
        const size_t tr = blockIdx.y;
        R += tr * QF8_RSIZE; normed += tr * QF8_RSIZE; q += tr * QF8_RSIZE; sc += tr * (QF8_RSIZE / 32);
        if (y) { y += tr * QF8_NEMBD; inj += tr * QF8_HCC; }
    }
    const int g0 = blockIdx.x * QF8_NEMBD;
    float acc = 0.f;
    if (y) {                                   // fused stream inject of the previous mixer
        const float sj = inj[blockIdx.x];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            const int e = g0 + i * 256 + threadIdx.x;
            float v = R[e] + sj * y[i * 256 + threadIdx.x];
            R[e] = v; acc += v * v;
        }
    } else {
    #pragma unroll
    for (int i = 0; i < 10; i++) { float v = R[g0 + i * 256 + threadIdx.x]; acc += v * v; }
    }
    acc = d8_group_sum<64>(acc);
    __shared__ float red[4];
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = acc;
    __syncthreads();
    const float rstd = rsqrtf((red[0] + red[1] + red[2] + red[3]) / QF8_NEMBD + 1e-6f);
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        const int e = g0 + i * 256 + threadIdx.x;          // 32 consecutive e per half-wave
        const float v = R[e] * rstd * (1.f + d8_bf(w[e]));
        normed[e] = v;
        const float mx = d8_half_absmax(v);
        const float s1 = mx > 0.f ? mx * (1.f / 127.f) : 1.f;
        int qi = (int)rintf(v / s1);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        q[e] = (int8_t)qi;
        if ((threadIdx.x & 31) == 0) sc[e >> 5] = s1;
    }
}
// mixed[j] = 1/4 sum_c sigmoid(hcy[c*2560+j]) * normed[c*2560+j]
__global__ void __launch_bounds__(256) k_d8_hc_combine(const float *__restrict__ hcy, const float *__restrict__ normed,
                                float *__restrict__ mixed, int8_t *__restrict__ xq, float *__restrict__ xs) {
    {   const size_t tr = blockIdx.y;                       // rows form (chunk); M=1 launch is y = 0
        hcy += tr * QF8_RSIZE; normed += tr * QF8_RSIZE; mixed += tr * QF8_NEMBD; xq += tr * QF8_NEMBD; xs += tr * (QF8_NEMBD / 32); }
    const int j = blockIdx.x * 256 + threadIdx.x;          // 2560 = 10 x 256: every half-wave full
    float sum = 0.f;
    #pragma unroll
    for (int c = 0; c < QF8_HCC; c++) {
        const float m = 1.f / (1.f + expf(-hcy[c * QF8_NEMBD + j]));
        sum += m * normed[c * QF8_NEMBD + j];
    }
    const float v = sum * 0.25f;
    mixed[j] = v;
    const float mx = d8_half_absmax(v);                    // X8(mixed) for the mixer / router
    const float s1 = mx > 0.f ? mx * (1.f / 127.f) : 1.f;
    int qi = (int)rintf(v / s1);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    xq[j] = (int8_t)qi;
    if ((threadIdx.x & 31) == 0) xs[j >> 5] = s1;
}

// shared expert elementwise pieces (M=1)
__global__ void k_d8_silu_mul(float *__restrict__ g, const float *__restrict__ u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = g[i]; g[i] = v / (1.f + expf(-v)) * u[i]; }
}
__global__ void k_d8_gated_add(float *__restrict__ y, const float *__restrict__ ed,
                               const float *__restrict__ g1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (1.f / (1.f + expf(-g1[0]))) * ed[i];
}

// ---- CHUNK (rows) form of the int8 GEMV: MR positions per pass ------------------
// The X8 tile of MR positions ([MR][K] int8 + block scales) is staged ONCE per block
// in LDS; every lane then reads its NST x 16 B of the weight row ONCE and dots it
// against all MR positions from LDS. RB row-groups per block amortize the tile
// staging. Weight bytes per position: 1/MR of the M=1 kernel; vs the bf16 8-row
// groups this is 2 (bytes) x MR/8 (rows) fewer HBM bytes. y is [T][rows].
template <int LPR, int NST, int MR, int RB>
__global__ void __launch_bounds__(256)
k_d8_gemv_rows(D8Batch b, const int8_t *__restrict__ xq, const float *__restrict__ xs, int T) {
    constexpr int K = LPR * NST * 16, RPB = 256 / LPR, KS = K / 32;
    __shared__ __align__(16) int8_t sx[MR * K];
    __shared__ float ss[MR * KS];
    const D8Desc d = b.d[blockIdx.y];
    if (blockIdx.x * (RPB * RB) >= d.rows) return;         // shorter descriptors: whole block out (before any sync)
    const int t0 = blockIdx.z * MR;
    const int nt = (T - t0) < MR ? (T - t0) : MR;
    for (int i = threadIdx.x; i < nt * (K / 16); i += 256) {
        const int j = i / (K / 16), c = i - j * (K / 16);
        *(d8v4 *)(sx + j * K + c * 16) = d8_ld16(xq + (size_t)(t0 + j) * K + c * 16);
    }
    for (int i = threadIdx.x; i < nt * KS; i += 256) { const int j = i / KS, c = i - j * KS; ss[i] = xs[(size_t)(t0 + j) * KS + c]; }
    __syncthreads();
    const int l = threadIdx.x % LPR;
    #pragma unroll 1
    for (int rb = 0; rb < RB; rb++) {
        const int row = (blockIdx.x * RB + rb) * RPB + threadIdx.x / LPR;
        if (row < d.rows) {                                 // whole LPR-lane group shares the row
            const int8_t *wr = d.w + (size_t)row * K + l * 16;
            d8v4 v[NST];
            #pragma unroll
            for (int st = 0; st < NST; st++) v[st] = d8_ld16nt(wr + st * (LPR * 16));
            float acc[MR];
            #pragma unroll
            for (int j = 0; j < MR; j++) acc[j] = 0.f;
            #pragma unroll
            for (int st = 0; st < NST; st++) {
                const int e0 = st * (LPR * 16) + l * 16;
                #pragma unroll
                for (int j = 0; j < MR; j++)
                    if (j < nt) acc[j] = fmaf(d8_dot16(v[st], *(const d8v4 *)(sx + j * K + e0)), ss[j * KS + (e0 >> 5)], acc[j]);
            }
            #pragma unroll
            for (int j = 0; j < MR; j++) acc[j] = d8_group_sum<LPR>(acc[j]);
            if (l == 0) {
                const float sr = d.s[row];
                for (int j = 0; j < nt; j++) {
                    float o = acc[j] * sr;
                    if (d.act == 1)      o = (o * 0.25f) / (1.f + expf(-o * 0.25f));
                    else if (d.act == 2) o = 2.f / (1.f + expf(-o * 0.25f));
                    d.y[(size_t)(t0 + j) * d.rows + row] = o;
                }
            }
        }
    }
}
// y[t] += sigmoid(g1[t]) * ed[t]; grid (NEMBD/256, T)
__global__ void k_d8_gated_add_rows(float *__restrict__ y, const float *__restrict__ ed, const float *__restrict__ g1) {
    const size_t t = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < QF8_NEMBD) y[t * QF8_NEMBD + i] += (1.f / (1.f + expf(-g1[t]))) * ed[t * QF8_NEMBD + i];
}
static int d8_fill_batch(const Qf8I8Multi *mm, D8Batch *b, int *maxr) {
    b->n = mm->n; b->K = mm->m[0]->K; *maxr = 0;
    for (int i = 0; i < mm->n; i++) {
        if (mm->m[i]->K != b->K) return -1;
        b->d[i].w = mm->m[i]->w; b->d[i].s = mm->m[i]->s; b->d[i].y = mm->y[i]; b->d[i].rows = mm->m[i]->rows; b->d[i].act = mm->act[i];
        if (mm->m[i]->rows > *maxr) *maxr = mm->m[i]->rows;
    }
    return 0;
}
extern "C" int qf8_i8_gemv_rows(const Qf8I8Multi *mm, const int8_t *xq, const float *xs, int T, hipStream_t s) {
    D8Batch b; int maxr;
    if (!mm || mm->n < 1 || mm->n > 4 || T < 1 || d8_fill_batch(mm, &b, &maxr)) return -1;
    switch (b.K) {
    case 2560:  k_d8_gemv_rows<32, 5, 16, 4><<<dim3((maxr + 31) / 32, b.n, (T + 15) / 16), 256, 0, s>>>(b, xq, xs, T); break;
    case 6144:  k_d8_gemv_rows<32, 12, 8, 4><<<dim3((maxr + 31) / 32, b.n, (T + 7) / 8), 256, 0, s>>>(b, xq, xs, T); break;
    case 640:   k_d8_gemv_rows<8, 5, 32, 2><<<dim3((maxr + 63) / 64, b.n, (T + 31) / 32), 256, 0, s>>>(b, xq, xs, T); break;
    case 320:   k_d8_gemv_rows<4, 5, 32, 1><<<dim3((maxr + 63) / 64, b.n, (T + 31) / 32), 256, 0, s>>>(b, xq, xs, T); break;
    case 10240: k_d8_gemv_rows<64, 10, 4, 4><<<dim3((maxr + 15) / 16, b.n, (T + 3) / 4), 256, 0, s>>>(b, xq, xs, T); break;
    default: return -1;
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
// Hyper-connection over T positions (the M=1 qf8_i8_hc_m1 sequence, rows form).
extern "C" int qf8_i8_hc_rows(float *normed, float *hc_d, float *mixed, float *inj, float *R,
                              const uint16_t *wn, const Qf8I8Mat *down, const Qf8I8Mat *inject, const Qf8I8Mat *up,
                              int8_t *n8q, float *n8s, int8_t *d8q, float *d8s, float *hcy,
                              const float *y_prev, const float *inj_prev, int8_t *xq_out, float *xs_out, int T, hipStream_t s) {
    k_d8_hc_norm_x8<<<dim3(QF8_HCC, T), 256, 0, s>>>(R, wn, normed, n8q, n8s, y_prev, inj_prev);
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = inject ? 2 : 1;
    m.m[0] = down; m.y[0] = hc_d; m.act[0] = 1;
    if (inject) { m.m[1] = inject; m.y[1] = inj; m.act[1] = 2; }
    if (qf8_i8_gemv_rows(&m, n8q, n8s, T, s)) return -1;     // K = 10240
    qf8_i8_x8(hc_d, d8q, d8s, T * QF8_HCL, s);
    Qf8I8Multi mu; memset(&mu, 0, sizeof mu); mu.n = 1; mu.m[0] = up; mu.y[0] = hcy;
    if (qf8_i8_gemv_rows(&mu, d8q, d8s, T, s)) return -1;    // K = 320, 10240 rows
    k_d8_hc_combine<<<dim3(QF8_NEMBD / 256, T), 256, 0, s>>>(hcy, normed, mixed, xq_out, xs_out);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
// Shared expert over T positions (the M=1 qf8_i8_shexp_m1 sequence, rows form). g1 is [T].
extern "C" int qf8_i8_shexp_rows(const Qf8I8Mat *gate, const Qf8I8Mat *up, const Qf8I8Mat *gate_inp, const Qf8I8Mat *down,
                                 const int8_t *xq, const float *xs, float *eg, float *eu, float *g1, float *ed,
                                 int8_t *e8q, float *e8s, float *y2560, int T, hipStream_t s) {
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 3;
    m.m[0] = gate; m.y[0] = eg; m.m[1] = up; m.y[1] = eu; m.m[2] = gate_inp; m.y[2] = g1;
    if (qf8_i8_gemv_rows(&m, xq, xs, T, s)) return -1;
    k_d8_silu_mul<<<(T * QF8_NFF + 255) / 256, 256, 0, s>>>(eg, eu, T * QF8_NFF);
    qf8_i8_x8(eg, e8q, e8s, T * QF8_NFF, s);
    Qf8I8Multi md; memset(&md, 0, sizeof md); md.n = 1; md.m[0] = down; md.y[0] = ed;
    if (qf8_i8_gemv_rows(&md, e8q, e8s, T, s)) return -1;
    k_d8_gated_add_rows<<<dim3((QF8_NEMBD + 255) / 256, T), 256, 0, s>>>(y2560, ed, g1);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

// ---- launchers ----------------------------------------------------------------
extern "C" int qf8_i8_quant_bf16(const uint16_t *W, int rows, int K, Qf8I8Mat *m, hipStream_t s) {
    if (!W || !m || rows < 1 || K < 16 || (K % 16)) return -1;
    memset(m, 0, sizeof(*m));
    if (hipMalloc((void **)&m->w, (size_t)rows * K) != hipSuccess) return -1;
    if (hipMalloc((void **)&m->s, (size_t)rows * sizeof(float)) != hipSuccess) return -1;
    m->rows = rows; m->K = K;
    k_d8_quant<<<rows, 256, 0, s>>>(W, m->w, m->s, rows, K);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
extern "C" int qf8_i8_quant_bf16_host(const uint16_t *Whost, int rows, int K, Qf8I8Mat *m, hipStream_t s) {
    if (!Whost || !m || rows < 1 || K < 16 || (K % 16)) return -1;
    memset(m, 0, sizeof(*m));
    if (hipMalloc((void **)&m->w, (size_t)rows * K) != hipSuccess) return -1;
    if (hipMalloc((void **)&m->s, (size_t)rows * sizeof(float)) != hipSuccess) return -1;
    m->rows = rows; m->K = K;
    const int RB = 4096;                                  // rows per staging block (21 MB at K=2560)
    uint16_t *stage = NULL;
    if (hipMalloc((void **)&stage, (size_t)RB * K * 2) != hipSuccess) return -1;
    for (int r0 = 0; r0 < rows; r0 += RB) {
        const int nr = (rows - r0) < RB ? (rows - r0) : RB;
        if (hipMemcpy(stage, Whost + (size_t)r0 * K, (size_t)nr * K * 2, hipMemcpyHostToDevice) != hipSuccess) { hipFree(stage); return -1; }
        k_d8_quant<<<nr, 256, 0, s>>>(stage, m->w + (size_t)r0 * K, m->s + r0, nr, K);
        if (hipStreamSynchronize(s) != hipSuccess) { hipFree(stage); return -1; }
    }
    hipFree(stage);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
extern "C" void qf8_i8_x8(const float *x, int8_t *q, float *sc, int n, hipStream_t s) {
    k_d8_x8<<<(n + 255) / 256, 256, 0, s>>>(x, q, sc, n);
}
extern "C" void qf8_i8_gemv_multi(const Qf8I8Multi *mm, const int8_t *xq, const float *xs, hipStream_t s) {
    D8Batch b; memset(&b, 0, sizeof b);
    int maxr = 0; b.n = mm->n; b.K = mm->m[0]->K;
    for (int i = 0; i < mm->n; i++) {
        b.d[i].w = mm->m[i]->w; b.d[i].s = mm->m[i]->s; b.d[i].y = mm->y[i]; b.d[i].rows = mm->m[i]->rows;
        b.d[i].act = mm->act[i];
        if (mm->m[i]->rows > maxr) maxr = mm->m[i]->rows;
        if (mm->m[i]->K != b.K) { fprintf(stderr, "qf8_i8_gemv_multi: mixed K\n"); return; }
    }
    switch (b.K) {
    case 2560:  k_d8_gemv<32, 5><<<dim3((maxr + 7) / 8, b.n), 256, 0, s>>>(b, xq, xs); break;
    case 6144:  k_d8_gemv<32, 12><<<dim3((maxr + 7) / 8, b.n), 256, 0, s>>>(b, xq, xs); break;
    case 640:   k_d8_gemv<8, 5><<<dim3((maxr + 31) / 32, b.n), 256, 0, s>>>(b, xq, xs); break;
    case 10240: k_d8_gemv<64, 10><<<dim3((maxr + 3) / 4, b.n), 256, 0, s>>>(b, xq, xs); break;
    case 320:   k_d8_gemv<4, 5><<<dim3((maxr + 63) / 64, b.n), 256, 0, s>>>(b, xq, xs); break;
    default:   fprintf(stderr, "qf8_i8_gemv_multi: unsupported K=%d\n", b.K); break;
    }
}
extern "C" void qf8_i8_shexp_m1(const Qf8I8Mat *gate, const Qf8I8Mat *up, const Qf8I8Mat *gate_inp,
                                const Qf8I8Mat *down, const int8_t *xq, const float *xs,
                                float *eg, float *eu, float *g1, float *ed, int8_t *e8q, float *e8s,
                                float *y2560, hipStream_t s) {
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = 3;
    m.m[0] = gate; m.y[0] = eg; m.m[1] = up; m.y[1] = eu; m.m[2] = gate_inp; m.y[2] = g1;
    qf8_i8_gemv_multi(&m, xq, xs, s);
    k_d8_silu_mul<<<(QF8_NFF + 255) / 256, 256, 0, s>>>(eg, eu, QF8_NFF);
    qf8_i8_x8(eg, e8q, e8s, QF8_NFF, s);
    Qf8I8Multi md; memset(&md, 0, sizeof md); md.n = 1; md.m[0] = down; md.y[0] = ed;
    qf8_i8_gemv_multi(&md, e8q, e8s, s);
    k_d8_gated_add<<<(QF8_NEMBD + 255) / 256, 256, 0, s>>>(y2560, ed, g1, QF8_NEMBD);
}

extern "C" void qf8_i8_hc_m1(float *normed, float *hc_d, float *mixed, float *inj, float *R,
                             const uint16_t *wn, const Qf8I8Mat *down, const Qf8I8Mat *inject,
                             const Qf8I8Mat *up, int8_t *n8q, float *n8s, int8_t *d8q, float *d8s,
                             float *hcy, const float *y_prev, const float *inj_prev,
                             int8_t *xq_out, float *xs_out, hipStream_t s) {
    k_d8_hc_norm_x8<<<QF8_HCC, 256, 0, s>>>(R, wn, normed, n8q, n8s, y_prev, inj_prev);
    Qf8I8Multi m; memset(&m, 0, sizeof m); m.n = inject ? 2 : 1;
    m.m[0] = down; m.y[0] = hc_d; m.act[0] = 1;
    if (inject) { m.m[1] = inject; m.y[1] = inj; m.act[1] = 2; }
    qf8_i8_gemv_multi(&m, n8q, n8s, s);                     // K = 10240: one launch, 320(+4) rows
    qf8_i8_x8(hc_d, d8q, d8s, QF8_HCL, s);                  // 320 = 10 blocks of 32
    Qf8I8Multi mu; memset(&mu, 0, sizeof mu); mu.n = 1; mu.m[0] = up; mu.y[0] = hcy;
    qf8_i8_gemv_multi(&mu, d8q, d8s, s);                    // K = 320: 10240 rows
    k_d8_hc_combine<<<QF8_NEMBD / 256, 256, 0, s>>>(hcy, normed, mixed, xq_out, xs_out);
}

#else
extern "C" void qf8_i8_hc_m1(float *normed, float *hc_d, float *mixed, float *inj, float *R,
                             const uint16_t *wn, const Qf8I8Mat *down, const Qf8I8Mat *inject,
                             const Qf8I8Mat *up, int8_t *n8q, float *n8s, int8_t *d8q, float *d8s,
                             float *hcy, const float *y_prev, const float *inj_prev,
                             int8_t *xq_out, float *xs_out, hipStream_t s) {
    (void)normed;(void)hc_d;(void)mixed;(void)inj;(void)R;(void)wn;(void)down;(void)inject;(void)up;
    (void)n8q;(void)n8s;(void)d8q;(void)d8s;(void)hcy;(void)y_prev;(void)inj_prev;(void)xq_out;(void)xs_out;(void)s; }
extern "C" int qf8_i8_quant_bf16(const uint16_t *W, int rows, int K, Qf8I8Mat *m, hipStream_t s) {
    (void)W;(void)rows;(void)K;(void)m;(void)s; return 0; }
extern "C" void qf8_i8_x8(const float *x, int8_t *q, float *sc, int n, hipStream_t s) { (void)x;(void)q;(void)sc;(void)n;(void)s; }
extern "C" int qf8_i8_quant_bf16_host(const uint16_t *W, int rows, int K, Qf8I8Mat *m, hipStream_t s) { (void)W;(void)rows;(void)K;(void)m;(void)s; return 0; }
extern "C" void qf8_i8_gemv_multi(const Qf8I8Multi *mm, const int8_t *xq, const float *xs, hipStream_t s) { (void)mm;(void)xq;(void)xs;(void)s; }
extern "C" void qf8_i8_shexp_m1(const Qf8I8Mat *gate, const Qf8I8Mat *up, const Qf8I8Mat *gate_inp,
                                const Qf8I8Mat *down, const int8_t *xq, const float *xs,
                                float *eg, float *eu, float *g1, float *ed, int8_t *e8q, float *e8s,
                                float *y2560, hipStream_t s) {
    (void)gate;(void)up;(void)gate_inp;(void)down;(void)xq;(void)xs;(void)eg;(void)eu;(void)g1;(void)ed;(void)e8q;(void)e8s;(void)y2560;(void)s; }
#endif
