// test_topk.hip — AMD radix k8_idx_topk_rows (B3 port) vs host reference of the
// original 48-step bisection semantics. Run on the AMD box: hipcc -O2 -I. test_topk.hip
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "hip/hip_runtime.h"
#include "qf8_dense.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP fail %s:%d %d\n", __FILE__, __LINE__, (int)e_); std::exit(1); } } while (0)

// ---------------- verbatim copy of the B3 radix rows kernel ----------------
__global__ void k8_idx_topk_rows(const float *__restrict__ blk_score_rows,
                                 uint8_t *__restrict__ mask_rows, int nb_max, long pos0,
                                 int *__restrict__ list_rows, int *__restrict__ nsel_rows) {
    __shared__ int   sh_cnt;
    __shared__ int   hist[256];
    __shared__ uint32_t sh_th;
    __shared__ int   sh_above;
    const int t = blockIdx.x;
    const int nfull = (int)((pos0 + t + 1) >> 2);
    const int tid = threadIdx.x, nthr = blockDim.x;
    if (nfull <= QF8_IDX_TOPK) { if (tid == 0) nsel_rows[t] = -1; return; }
    const float *blk_score = blk_score_rows + (size_t)t * nb_max;
    uint8_t *mask = mask_rows + (size_t)t * nb_max;
    int *blk_list = list_rows + (size_t)t * (QF8_IDX_TOPK + 1);
    const int lane = tid % QF8_WAVE, wid = tid / QF8_WAVE;

    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int posc = 0;
    for (int i = tid; i < nfull; i += nthr) posc += (blk_score[i] > 0.f);
    posc = (int)qf8_wave_sum((float)posc);
    if (lane == 0) atomicAdd(&sh_cnt, posc);
    __syncthreads();
    if (sh_cnt < QF8_IDX_TOPK) {
        for (int i = tid; i < nfull; i += nthr) mask[i] = (uint8_t)(blk_score[i] > 0.f);
        __syncthreads();
        if (wid == 0) {
            int count = 0;
            for (int base = 0; base < nfull; base += QF8_WAVE) {
                const int b = base + lane;
                const int sel = (b < nfull) ? (int)mask[b] : 0;
                const unsigned long long bal = __ballot(sel);
                if (sel) blk_list[count + __popcll(bal & ((1ull << lane) - 1ull))] = b;
                count += __popcll(bal);
            }
            if (lane == 0) nsel_rows[t] = count;
        }
        return;
    }

    if (tid == 0) { sh_th = 0u; sh_above = 0; }
    __syncthreads();
    for (int pass = 0; pass < 4; pass++) {
        const int shift = (3 - pass) * 8;
        const uint32_t pref_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        const uint32_t th = sh_th;
        const uint32_t pref = th & pref_mask;
        for (int i = tid; i < 256; i += nthr) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < nfull; i += nthr) {
            const uint32_t u = __float_as_uint(blk_score[i]);
            if ((u & pref_mask) == pref) atomicAdd(&hist[(u >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = 0;
            for (int v = 255; v >= 0; v--) {
                if (cum + hist[v] >= QF8_IDX_TOPK - sh_above) { sh_th |= (uint32_t)v << shift; sh_above += cum; break; }
                cum += hist[v];
            }
        }
        __syncthreads();
    }
    const float thf = __uint_as_float(sh_th);
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    int c = 0;
    for (int i = tid; i < nfull; i += nthr) {
        const uint8_t sel = (blk_score[i] > thf);
        mask[i] = sel;
        c += sel;
    }
    c = (int)qf8_wave_sum((float)c);
    if (lane == 0) atomicAdd(&sh_cnt, c);
    __syncthreads();
    if (tid == 0) {
        int r = QF8_IDX_TOPK - sh_cnt;
        for (int i = 0; i < nfull && r > 0; i++) {
            const float v = blk_score[i];
            if (v == thf) { mask[i] = 1; r--; }
        }
    }
    __syncthreads();
    if (wid == 0) {
        int count = 0;
        for (int base = 0; base < nfull; base += QF8_WAVE) {
            const int b = base + lane;
            const int sel = (b < nfull) ? (int)mask[b] : 0;
            const unsigned long long bal = __ballot(sel);
            if (sel) blk_list[count + __popcll(bal & ((1ull << lane) - 1ull))] = b;
            count += __popcll(bal);
        }
        if (lane == 0) nsel_rows[t] = count;
    }
}
// ---------------- end verbatim copy ----------------

static void ref_topk(const float* sc, int nfull, std::vector<uint8_t>& mask, std::vector<int>& list, int& nsel) {
    mask.assign(nfull, 0);
    if (nfull <= QF8_IDX_TOPK) {
        list.assign(nfull, 0);
        for (int b = 0; b < nfull; b++) list[b] = b;
        nsel = nfull;
        return;
    }
    float mx = 0.f;
    for (int i = 0; i < nfull; i++) mx = fmaxf(mx, sc[i]);
    float lo = 0.f, hi = mx;
    for (int it = 0; it < 48; it++) {
        const float mid = 0.5f * (lo + hi);
        int c = 0;
        for (int i = 0; i < nfull; i++) c += (sc[i] > mid);
        if (c >= QF8_IDX_TOPK) lo = mid; else hi = mid;
    }
    int c = 0;
    for (int i = 0; i < nfull; i++) { uint8_t sel = (sc[i] > hi); mask[i] = sel; c += sel; }
    int r = QF8_IDX_TOPK - c;
    for (int i = 0; i < nfull && r > 0; i++) { const float v = sc[i]; if (v > lo && v <= hi) { mask[i] = 1; r--; } }
    list.clear();
    for (int b = 0; b < nfull; b++) if (mask[b]) list.push_back(b);
    nsel = (int)list.size();
}

static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32() { lcg ^= lcg >> 12; lcg ^= lcg << 25; lcg ^= lcg >> 27; return (uint32_t)((lcg * 0x2545F4914F6CDD1Dull) >> 32); }
static float frand() { return (float)(rnd32() & 0xFFFFFF) * (1.0f / 16777216.0f); }

static int run_case(const char* name, int nfull, int mode) {
    const int nb_max = 4096;
    std::vector<float> sc(nb_max, 0.f);
    for (int b = 0; b < nfull; b++) {
        switch (mode) {
            case 0: case 2: case 7: sc[b] = frand() * 10.f; break;
            case 3: sc[b] = roundf(frand() * 10.f * 100.f) / 100.f; break;
            case 4: sc[b] = 0.f; break;
            case 5: sc[b] = (b < 100) ? (1.0f + 0.001f * (float)(100 - b)) : 0.f; break;
            case 6: sc[b] = 1.0f; break;
            case 8: sc[b] = (b < 100) ? 5.0f : frand() * 4.9f; break;
            case 9: sc[b] = (b < 600) ? 1.0f : 0.5f; break;
            default: sc[b] = frand(); break;
        }
    }
    float *d_sc; uint8_t* d_mask; int *d_list, *d_nsel;
    CK(hipMalloc(&d_sc, nb_max * sizeof(float)));
    CK(hipMalloc(&d_mask, nb_max));
    CK(hipMalloc(&d_list, (QF8_IDX_TOPK + 1) * sizeof(int)));
    CK(hipMalloc(&d_nsel, sizeof(int)));
    CK(hipMemcpy(d_sc, sc.data(), nb_max * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemset(d_mask, 0xCD, nb_max));
    CK(hipMemset(d_list, 0xCD, (QF8_IDX_TOPK + 1) * sizeof(int)));
    k8_idx_topk_rows<<<1, 256>>>(d_sc, d_mask, nb_max, (long)(nfull * 4 - 1), d_list, d_nsel);
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    std::vector<uint8_t> gmask(nb_max); std::vector<int> glist(QF8_IDX_TOPK + 1); int gnsel;
    CK(hipMemcpy(gmask.data(), d_mask, nb_max, hipMemcpyDeviceToHost));
    CK(hipMemcpy(glist.data(), d_list, (QF8_IDX_TOPK + 1) * sizeof(int), hipMemcpyDeviceToHost));
    CK(hipMemcpy(&gnsel, d_nsel, sizeof(int), hipMemcpyDeviceToHost));
    std::vector<uint8_t> hmask; std::vector<int> hlist; int hnsel;
    ref_topk(sc.data(), nfull, hmask, hlist, hnsel);
    bool ok = (gnsel == hnsel);
    for (int i = 0; ok && i < nfull; i++) if (gmask[i] != hmask[i]) ok = false;
    for (int i = 0; ok && i < hnsel; i++) if (glist[i] != hlist[i]) ok = false;
    std::printf("%-24s nfull=%-5d mode=%d nsel(gpu=%d ref=%d): %s\n", name, nfull, mode, gnsel, hnsel, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  gpu list head: ");
        for (int i = 0; i < 8 && i < gnsel; i++) std::printf("%d ", glist[i]);
        std::printf("\n  ref list head: ");
        for (int i = 0; i < 8 && i < hnsel; i++) std::printf("%d ", hlist[i]);
        std::printf("\n");
    }
    CK(hipFree(d_sc)); CK(hipFree(d_mask)); CK(hipFree(d_list)); CK(hipFree(d_nsel));
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += run_case("random 513", 513, 0);
    fails += run_case("random 1024", 1024, 1);
    fails += run_case("random 4096", 4096, 2);
    fails += run_case("random 2009 (prod)", 2009, 7);
    fails += run_case("quantized ties", 1024, 3);
    fails += run_case("all zero", 1024, 4);
    fails += run_case("100 positive", 1024, 5);
    fails += run_case("all equal 1.0", 4096, 6);
    fails += run_case("max ties boundary", 1024, 8);
    fails += run_case("two-level ties", 2048, 9);
    fails += run_case("dense 512", 512, 0);
    fails += run_case("dense 300", 300, 0);
    std::printf(fails ? "TOPK_TEST: %d FAILURE(S)\n" : "TOPK_TEST: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
