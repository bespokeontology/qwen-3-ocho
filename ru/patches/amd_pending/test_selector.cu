// test_selector.cu — standalone unit test of the radix k_idx_select_rows
// (copied verbatim from src/cuda/qf_qsa_index.cu, B2) against a host reference
// of the original 48-step bisection semantics. Run under compute-sanitizer.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "cuda_runtime.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

#define IDX_H 4
#define IDX_D 128
#define IDX_TOPK 512
#define IDX_W 640

typedef struct { int token; int pos; uint32_t seq; uint32_t flags; } QfDecodeParams;

// ---------------- verbatim copy of idx_compact_list + k_idx_select_rows ----------------
__device__ __forceinline__ void idx_compact_list(const uint32_t *__restrict__ mrow, int nfull, int *__restrict__ list, int *__restrict__ nlist) {
    const int lane = threadIdx.x & 31;
    int count = 0;
    for (int base = 0; base < nfull; base += 32) {
        const int b = base + lane;
        const int sel = (b < nfull) ? (int)((mrow[b >> 5] >> (b & 31)) & 1u) : 0;
        const uint32_t bal = __ballot_sync(0xffffffffu, sel);
        if (sel) list[count + __popc(bal & ((1u << lane) - 1u))] = b;
        count += __popc(bal);
    }
    if (lane == 0) *nlist = count;
}
__global__ void __launch_bounds__(256) k_idx_select_rows(const float *__restrict__ idx640, const float *__restrict__ pool_key,
                                                         const QfDecodeParams *__restrict__ params, float *__restrict__ blk_score, int nb_max,
                                                         uint32_t *__restrict__ mask, int mw, int *__restrict__ list, int *__restrict__ nlist,
                                                         int pre) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const long pos = (long)params[t].pos;
    int nfull = (int)(pos >> 2);
    if (nfull > nb_max) nfull = nb_max;
    uint32_t *mrow = mask + (size_t)t * mw;
    int *lrow = list + (size_t)t * (IDX_TOPK + 1);
    if (nfull <= IDX_TOPK) {
        for (int w = tid; w < mw; w += 256) mrow[w] = 0xFFFFFFFFu;
        for (int b = tid; b < nfull; b += 256) lrow[b] = b;
        if (tid == 0) nlist[t] = nfull;
        return;
    }
    for (int w = tid; w < mw; w += 256) mrow[w] = 0u;
    __shared__ float q[IDX_H * IDX_D];
    __shared__ int sh_cnt;
    __shared__ int hist[256];
    __shared__ uint32_t sh_th;
    __shared__ int sh_above;
    for (int i = tid; i < IDX_H * IDX_D; i += 256) q[i] = idx640[(size_t)t * IDX_W + i];
    __syncthreads();
    float *sc = blk_score + (size_t)t * nb_max;
    const float iscale = rsqrtf((float)IDX_D);
    if (!pre)
    for (int b = tid; b < nfull; b += 256) {
        const float *kb = pool_key + (size_t)b * IDX_D;
        float s = 0.f;
        #pragma unroll
        for (int h = 0; h < IDX_H; h++) {
            float d = 0.f;
            #pragma unroll 8
            for (int i = 0; i < IDX_D; i++) d = fmaf(q[h * IDX_D + i], kb[i], d);
            s += fmaxf(d * iscale, 0.f);
        }
        sc[b] = s;
    }
    __syncthreads();
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int posc = 0;
    for (int b = tid; b < nfull; b += 256) posc += (sc[b] > 0.f);
    #pragma unroll
    for (int off = 16; off; off >>= 1) posc += __shfl_down_sync(0xffffffffu, posc, off);
    if (lane == 0) atomicAdd(&sh_cnt, posc);
    __syncthreads();
    if (sh_cnt < IDX_TOPK) {
        for (int b = tid; b < nfull; b += 256) if (sc[b] > 0.f) atomicOr(&mrow[b >> 5], 1u << (b & 31));
        __syncthreads();
        if (wid == 0) idx_compact_list(mrow, nfull, lrow, nlist + t);
        return;
    }
    if (tid == 0) { sh_th = 0u; sh_above = 0; }
    __syncthreads();
    for (int pass = 0; pass < 4; pass++) {
        const int shift = (3 - pass) * 8;
        const uint32_t pref_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        const uint32_t th = sh_th;
        const uint32_t pref = th & pref_mask;
        hist[tid] = 0;
        __syncthreads();
        for (int b = tid; b < nfull; b += 256) {
            const uint32_t u = __float_as_uint(sc[b]);
            if ((u & pref_mask) == pref) atomicAdd(&hist[(u >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = 0;
            for (int v = 255; v >= 0; v--) {
                if (cum + hist[v] >= IDX_TOPK - sh_above) { sh_th |= (uint32_t)v << shift; sh_above += cum; break; }
                cum += hist[v];
            }
        }
        __syncthreads();
    }
    const float thf = __uint_as_float(sh_th);
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int c = 0;
    for (int b = tid; b < nfull; b += 256) if (sc[b] > thf) { atomicOr(&mrow[b >> 5], 1u << (b & 31)); c++; }
    #pragma unroll
    for (int off = 16; off; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    if (lane == 0) atomicAdd(&sh_cnt, c);
    __syncthreads();
    if (tid == 0) {
        int r = IDX_TOPK - sh_cnt;
        for (int b = 0; b < nfull && r > 0; b++) if (sc[b] == thf) { mrow[b >> 5] |= 1u << (b & 31); r--; }
    }
    __syncthreads();
    if (wid == 0) idx_compact_list(mrow, nfull, lrow, nlist + t);
}
// ---------------- end verbatim copy ----------------

// host reference: the ORIGINAL 48-step bisection semantics, bit-for-bit
static void ref_select(const float* sc, int nfull, std::vector<uint32_t>& mask, int mw, std::vector<int>& list, int& nlist) {
    for (int w = 0; w < mw; w++) mask[w] = 0;
    if (nfull <= IDX_TOPK) {
        for (int w = 0; w < mw; w++) mask[w] = 0xFFFFFFFFu;
        list.assign(nfull, 0);
        for (int b = 0; b < nfull; b++) list[b] = b;
        nlist = nfull;
        return;
    }
    float mx = 0.f;
    for (int b = 0; b < nfull; b++) mx = fmaxf(mx, sc[b]);
    float lo = 0.f, hi = mx;
    for (int it = 0; it < 48; it++) {
        const float mid = 0.5f * (lo + hi);
        int c = 0;
        for (int b = 0; b < nfull; b++) c += (sc[b] > mid);
        if (c >= IDX_TOPK) lo = mid; else hi = mid;
    }
    int c = 0;
    for (int b = 0; b < nfull; b++) if (sc[b] > hi) { mask[b >> 5] |= 1u << (b & 31); c++; }
    int r = IDX_TOPK - c;
    for (int b = 0; b < nfull && r > 0; b++) { const float v = sc[b]; if (v > lo && v <= hi) { mask[b >> 5] |= 1u << (b & 31); r--; } }
    list.clear();
    for (int b = 0; b < nfull; b++) if ((mask[b >> 5] >> (b & 31)) & 1u) list.push_back(b);
    nlist = (int)list.size();
}

static uint32_t fu32(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float ffromu32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t host_radix(const float* sc, int nfull) {
    uint32_t th = 0;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = (3 - pass) * 8;
        const uint32_t pref_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        const uint32_t pref = th & pref_mask;
        int hist[256] = {0};
        for (int b = 0; b < nfull; b++) {
            const uint32_t u = fu32(sc[b]);
            if ((u & pref_mask) == pref) hist[(u >> shift) & 255u]++;
        }
        int cum = 0;
        for (int v = 255; v >= 0; v--) {
            if (cum + hist[v] >= IDX_TOPK) { th |= (uint32_t)v << shift; break; }
            cum += hist[v];
        }
    }
    return th;
}
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32() { lcg ^= lcg >> 12; lcg ^= lcg << 25; lcg ^= lcg >> 27; return (uint32_t)((lcg * 0x2545F4914F6CDD1Dull) >> 32); }
static float frand() { return (float)(rnd32() & 0xFFFFFF) * (1.0f / 16777216.0f); }

static int run_case(const char* name, int nfull, int mode, int pre) {
    const int nb_max = 4096, mw = nb_max / 32;
    std::vector<float> sc(nb_max, 0.f);
    for (int b = 0; b < nfull; b++) {
        switch (mode) {
            case 0: case 1: case 2: case 7: sc[b] = frand() * 10.f; break;
            case 3: sc[b] = roundf(frand() * 10.f * 100.f) / 100.f; break;
            case 4: sc[b] = 0.f; break;
            case 5: sc[b] = (b < 100) ? (1.0f + 0.001f * (float)(100 - b)) : 0.f; break;
            case 6: sc[b] = 1.0f; break;
            case 8: sc[b] = (b < 100) ? 5.0f : (frand() * 4.9f); break;
            case 9: sc[b] = (b < 600) ? 1.0f : 0.5f; break;
            default: sc[b] = frand(); break;
        }
    }
    // device side
    float *d_sc, *d_idx, *d_pool; uint32_t* d_mask; int *d_list, *d_nlist; QfDecodeParams* d_params;
    CK(cudaMalloc(&d_sc, nb_max * sizeof(float)));
    CK(cudaMalloc(&d_idx, IDX_W * sizeof(float)));
    CK(cudaMalloc(&d_pool, nb_max * IDX_D * sizeof(float)));
    CK(cudaMalloc(&d_mask, mw * sizeof(uint32_t)));
    CK(cudaMalloc(&d_list, (IDX_TOPK + 1) * sizeof(int)));
    CK(cudaMalloc(&d_nlist, sizeof(int)));
    CK(cudaMalloc(&d_params, sizeof(QfDecodeParams)));
    CK(cudaMemcpy(d_sc, sc.data(), nb_max * sizeof(float), cudaMemcpyHostToDevice));
    if (pre == 0) {
        std::vector<float> qv(IDX_W, 0.25f), pv(nb_max * IDX_D, 0.25f);
        CK(cudaMemcpy(d_idx, qv.data(), IDX_W * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pool, pv.data(), nb_max * IDX_D * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CK(cudaMemset(d_idx, 0, IDX_W * sizeof(float)));
        CK(cudaMemset(d_pool, 0, nb_max * IDX_D * sizeof(float)));
    }
    CK(cudaMemset(d_mask, 0xCD, mw * sizeof(uint32_t)));
    CK(cudaMemset(d_list, 0xCD, (IDX_TOPK + 1) * sizeof(int)));
    QfDecodeParams p; p.token = 0; p.pos = nfull * 4; p.seq = 0; p.flags = 0;
    CK(cudaMemcpy(d_params, &p, sizeof(p), cudaMemcpyHostToDevice));
    k_idx_select_rows<<<1, 256>>>(d_idx, d_pool, d_params, d_sc, nb_max, d_mask, mw, d_list, d_nlist, pre);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<uint32_t> gmask(mw); int glist[IDX_TOPK + 1]; int gnlist;
    CK(cudaMemcpy(gmask.data(), d_mask, mw * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(glist, d_list, (IDX_TOPK + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&gnlist, d_nlist, sizeof(int), cudaMemcpyDeviceToHost));
    // host reference
    std::vector<uint32_t> hmask(mw, 0u); std::vector<int> hlist; int hnlist;
    if (pre == 0) {
        // pre=0: q = pool_key = 0.25 -> d = 128 fmaf(0.0625) = 8.0 exact, all scores bit-identical
        for (int b = 0; b < nfull; b++) sc[b] = 1.0f;   // any constant: all-equal selection = first blocks
        ref_select(sc.data(), nfull, hmask, mw, hlist, hnlist);
    } else {
        ref_select(sc.data(), nfull, hmask, mw, hlist, hnlist);
    }
    bool ok = (gnlist == hnlist) && (memcmp(gmask.data(), hmask.data(), mw * sizeof(uint32_t)) == 0);
    for (int i = 0; ok && i < hnlist; i++) if (glist[i] != hlist[i]) ok = false;
    if (!ok && pre == 1) {
        const uint32_t hr = host_radix(sc.data(), nfull);
        std::printf("[host] radix th=%08x (%.9f)\n", hr, ffromu32(hr));
    }
    std::printf("%-28s nfull=%-5d mode=%d pre=%d nlist(gpu=%d ref=%d): %s\n", name, nfull, mode, pre, gnlist, hnlist, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  gpu list head: ");
        for (int i = 0; i < 8 && i < gnlist; i++) std::printf("%d ", glist[i]);
        std::printf("\n  ref list head: ");
        for (int i = 0; i < 8 && i < hnlist; i++) std::printf("%d ", hlist[i]);
        std::printf("\n");
    }
    CK(cudaFree(d_sc)); CK(cudaFree(d_idx)); CK(cudaFree(d_pool)); CK(cudaFree(d_mask));
    CK(cudaFree(d_list)); CK(cudaFree(d_nlist)); CK(cudaFree(d_params));
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += run_case("random 513", 513, 0, 1);
    fails += run_case("random 1024", 1024, 1, 1);
    fails += run_case("random 4096", 4096, 2, 1);
    fails += run_case("random 2009 (prod)", 2009, 7, 1);
    fails += run_case("quantized ties", 1024, 3, 1);
    fails += run_case("all zero", 1024, 4, 1);
    fails += run_case("100 positive", 1024, 5, 1);
    fails += run_case("all equal 1.0", 4096, 6, 1);
    fails += run_case("max ties boundary", 1024, 8, 1);
    fails += run_case("two-level ties", 2048, 9, 1);
    fails += run_case("dense 512", 512, 0, 1);
    fails += run_case("dense 300", 300, 0, 1);
    fails += run_case("pre=0 all-equal", 2048, 0, 0);
    std::printf(fails ? "SELECTOR_TEST: %d FAILURE(S)\n" : "SELECTOR_TEST: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
