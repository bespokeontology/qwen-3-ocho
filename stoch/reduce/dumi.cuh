// dumi.cuh — Dumitrescu reduction: online-stable weighted statistics over a branch field.
// One branch = one log-weight lw (double) + K observables f[K] (float); w = exp(lw).
// Convention: lw < -1e300 is the zero-weight sentinel (w = 0); such branches are skipped
// entirely (counted in neither N nor the sums). Dead branches should be compacted away by
// the caller, but the reduction is safe on them.
// Accumulation is fp64. Each block writes its partial to a partials array
// (stride 6+2K: [n, sw, sw2, maxlw, lse_m, lse_s, swf[K], swf2[K]]); the HOST merges the
// partials in double — deterministic merge, no cross-block atomics.
// Stats: N, sum w, sum w^2, sum w*f_k, sum w*f_k^2, max log-w, log-sum-exp(log w),
// ESS = (sum w)^2 / sum w^2.
#pragma once
#include <cstdint>
#include <cfloat>
#include <cstring>
#include <cmath>

struct DumiStats {
    uint64_t n;
    double sw, sw2, maxlw, lse_m, lse_s;
    double swf[8], swf2[8];
};

__host__ inline void dumi_zero(DumiStats* s) {
    memset(s, 0, sizeof *s);
    s->maxlw = -DBL_MAX; s->lse_m = -DBL_MAX; s->lse_s = 0.0;
}

template<int K>
struct DumiAccum {
    uint64_t n; double sw, sw2, maxlw, lse_m, lse_s; double swf[K], swf2[K];
    __device__ DumiAccum() : n(0), sw(0.0), sw2(0.0), maxlw(-DBL_MAX), lse_m(-DBL_MAX), lse_s(0.0) {
        for (int k = 0; k < K; k++) { swf[k] = 0.0; swf2[k] = 0.0; }
    }
    __device__ inline void add(double lw, const float* f) {
        if (lw <= -1e300) return;             // zero-weight sentinel
        n += 1;
        const double w = exp(lw);
        sw += w; sw2 += w * w;
        for (int k = 0; k < K; k++) { const double v = (double)f[k]; swf[k] += w * v; swf2[k] += w * v * v; }
        maxlw = fmax(maxlw, lw);
        const double nm = fmax(lse_m, lw);
        lse_s = lse_s * exp(lse_m - nm) + exp(lw - nm);
        lse_m = nm;
    }
};

__device__ inline uint64_t warp_reduce_ull(uint64_t v) { for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }
__device__ inline double warp_reduce_d(double v) { for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }
__device__ inline double warp_reduce_maxd(double v) { for (int o = 16; o; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o)); return v; }

// sm must be __shared__ double[8][3 + 2*K]; partial has room for 6 + 2*K doubles.
template<int K>
__device__ inline void dumi_block_commit(const DumiAccum<K>& a, double* sm, double* partial) {
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const uint64_t n = warp_reduce_ull(a.n);
    const double sw = warp_reduce_d(a.sw), sw2 = warp_reduce_d(a.sw2);
    double swf[K], swf2[K];
    for (int k = 0; k < K; k++) { swf[k] = warp_reduce_d(a.swf[k]); swf2[k] = warp_reduce_d(a.swf2[k]); }
    const double maxlw = warp_reduce_maxd(a.maxlw);
    double m = a.lse_m, s = a.lse_s;
    for (int o = 16; o; o >>= 1) {
        const double om = __shfl_down_sync(0xffffffffu, m, o);
        const double os = __shfl_down_sync(0xffffffffu, s, o);
        const double nm = fmax(m, om);
        s = s * exp(m - nm) + os * exp(om - nm);
        m = nm;
    }
    constexpr int SW = 6 + 2 * K;
    if (lane == 0) {
        sm[wid * SW + 0] = (double)n; sm[wid * SW + 1] = sw; sm[wid * SW + 2] = sw2;
        for (int k = 0; k < K; k++) { sm[wid * SW + 3 + k] = swf[k]; sm[wid * SW + 3 + K + k] = swf2[k]; }
        sm[wid * SW + 3 + 2 * K] = maxlw; sm[wid * SW + 3 + 2 * K + 1] = m; sm[wid * SW + 3 + 2 * K + 2] = s;
    }
    __syncthreads();
    if (wid == 0) {
        uint64_t n2 = 0; double swb = 0, sw2b = 0, swfb[K], swf2b[K], mxb = -DBL_MAX, mb = -DBL_MAX, sb = 0.0;
        for (int k = 0; k < K; k++) { swfb[k] = 0.0; swf2b[k] = 0.0; }
        for (int w = 0; w < 8; w++) {
            n2 += (uint64_t)sm[w * SW + 0]; swb += sm[w * SW + 1]; sw2b += sm[w * SW + 2];
            for (int k = 0; k < K; k++) { swfb[k] += sm[w * SW + 3 + k]; swf2b[k] += sm[w * SW + 3 + K + k]; }
            mxb = fmax(mxb, sm[w * SW + 3 + 2 * K]);
            const double om = sm[w * SW + 3 + 2 * K + 1], os = sm[w * SW + 3 + 2 * K + 2];
            const double nm = fmax(mb, om);
            sb = sb * exp(mb - nm) + os * exp(om - nm);
            mb = nm;
        }
        if (lane == 0) {
            double* p = partial;
            p[0] = (double)n2; p[1] = swb; p[2] = sw2b; p[3] = mxb; p[4] = mb; p[5] = sb;
            for (int k = 0; k < K; k++) { p[6 + k] = swfb[k]; p[6 + K + k] = swf2b[k]; }
        }
    }
}

template<int K>
__global__ void k_dumi_reduce(const double* __restrict__ lw, const float* __restrict__ f,
                              uint64_t M, double* __restrict__ partial) {
    DumiAccum<K> a;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M;
         i += (uint64_t)gridDim.x * blockDim.x) {
        a.add(lw[i], f + (size_t)i * K);
    }
    __shared__ double sm[8][6 + 2 * K];
    dumi_block_commit<K>(a, &sm[0][0], partial + (size_t)blockIdx.x * (6 + 2 * K));
}

__host__ inline void dumi_merge(const double* p, int blocks, int K, DumiStats* s) {
    dumi_zero(s);
    for (int b = 0; b < blocks; b++) {
        const double* q = p + (size_t)b * (6 + 2 * K);
        s->n += (uint64_t)q[0];
        s->sw += q[1]; s->sw2 += q[2];
        s->maxlw = fmax(s->maxlw, q[3]);
        const double nm = fmax(s->lse_m, q[4]);
        s->lse_s = s->lse_s * exp(s->lse_m - nm) + q[5] * exp(q[4] - nm);
        s->lse_m = nm;
        for (int k = 0; k < K; k++) { s->swf[k] += q[6 + k]; s->swf2[k] += q[6 + K + k]; }
    }
}

__host__ inline double dumi_ess(const DumiStats& s) { return s.sw2 > 0.0 ? s.sw * s.sw / s.sw2 : 0.0; }
__host__ inline double dumi_total_w(const DumiStats& s) { return s.lse_m > -1e300 ? exp(s.lse_m) * s.lse_s : 0.0; }
