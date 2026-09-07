// dumi_test.cu — Dumitrescu reduction vs CPU fp64 reference. Exit 0 = all pass.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "dumi.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

template<int K>
static void cpu_ref(const double* lw, const float* f, size_t M, DumiStats* st) {
    dumi_zero(st);
    for (size_t i = 0; i < M; i++) {
        if (lw[i] <= -1e300) continue;
        st->n++;
        const double w = exp(lw[i]);
        st->sw += w; st->sw2 += w * w;
        for (int k = 0; k < K; k++) { const double v = (double)f[i * K + k]; st->swf[k] += w * v; st->swf2[k] += w * v * v; }
        st->maxlw = fmax(st->maxlw, lw[i]);
        const double nm = fmax(st->lse_m, lw[i]);
        st->lse_s = st->lse_s * exp(st->lse_m - nm) + exp(lw[i] - nm);
        st->lse_m = nm;
    }
}

static bool close_rel(double a, double b, double tol) {
    if (a == b) return true;
    const double scale = fmax(fabs(a), fabs(b));
    if (scale == 0.0) return true;
    return fabs(a - b) <= tol * fmax(1.0, scale);
}

template<int K>
static bool run_case(const char* name, size_t M, const std::vector<double>& lw, const std::vector<float>& f) {
    double *d_lw; float *d_f; double *d_part;
    const size_t grid = std::min<size_t>((M + 255) / 256, 4096);
    const int stride = 6 + 2 * K;
    CK(cudaMalloc(&d_lw, M * sizeof(double)));
    CK(cudaMalloc(&d_f, M * K * sizeof(float)));
    CK(cudaMalloc(&d_part, grid * stride * sizeof(double)));
    CK(cudaMemcpy(d_lw, lw.data(), M * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_f, f.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_part, 0, grid * stride * sizeof(double)));
    k_dumi_reduce<K><<<(unsigned)grid, 256>>>(d_lw, d_f, (uint64_t)M, d_part);
    CK(cudaDeviceSynchronize());
    std::vector<double> part(grid * stride);
    CK(cudaMemcpy(part.data(), d_part, grid * stride * sizeof(double), cudaMemcpyDeviceToHost));
    DumiStats g, c;
    dumi_merge(part.data(), (int)grid, K, &g);
    cpu_ref<K>(lw.data(), f.data(), M, &c);
    const double ge = dumi_ess(g), ce = dumi_ess(c);
    const double gt = dumi_total_w(g), ct = dumi_total_w(c);
    bool ok = g.n == c.n
        && close_rel(g.sw, c.sw, 1e-12) && close_rel(g.sw2, c.sw2, 1e-12)
        && close_rel(g.maxlw, c.maxlw, 1e-12)
        && close_rel(g.lse_m, c.lse_m, 1e-12) && close_rel(g.lse_s, c.lse_s, 1e-12)
        && close_rel(gt, ct, 1e-12) && close_rel(ge, ce, 1e-9);
    for (int k = 0; k < K && ok; k++)
        ok = close_rel(g.swf[k], c.swf[k], 1e-9) && close_rel(g.swf2[k], c.swf2[k], 1e-9);
    std::printf("%-30s K=%d M=%-9zu n=%llu sw=%.6e ess=%.6e total_w=%.6e -> %s\n",
                name, K, M, (unsigned long long)g.n, g.sw, ge, gt, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("   gpu: n=%llu sw=%.17g sw2=%.17g maxlw=%.17g m=%.17g s=%.17g\n",
                    (unsigned long long)g.n, g.sw, g.sw2, g.maxlw, g.lse_m, g.lse_s);
        std::printf("   cpu: n=%llu sw=%.17g sw2=%.17g maxlw=%.17g m=%.17g s=%.17g\n",
                    (unsigned long long)c.n, c.sw, c.sw2, c.maxlw, c.lse_m, c.lse_s);
        for (int k = 0; k < K; k++)
            std::printf("   k=%d swf gpu=%.17g cpu=%.17g | swf2 gpu=%.17g cpu=%.17g\n",
                        k, g.swf[k], c.swf[k], g.swf2[k], c.swf2[k]);
    }
    CK(cudaFree(d_lw)); CK(cudaFree(d_f)); CK(cudaFree(d_part));
    return ok;
}

template<int K>
static void gen(int cid, size_t M, std::vector<double>& lw, std::vector<float>& f) {
    lw.assign(M, 0.0); f.assign(M * K, 0.0f);
    for (size_t i = 0; i < M; i++) {
        double w = 1.0;
        switch (cid) {
            case 0: w = 1.0; break;
            case 1: w = (double)((i % 7) + 1); break;
            case 2: w = (i % 10 < 3) ? 0.0 : (double)((i % 5) + 1); break;
            case 3: w = 1e-30; break;
            case 4: w = 1e30; break;
            case 5: w = (double)(i + 1); break;
            case 6: w = 0.0; break;
            default: w = 1.0; break;
        }
        lw[i] = (w == 0.0) ? -1e300 : log(w);
        for (int k = 0; k < K; k++) {
            const double x = sin((double)i * 0.001 + (double)k) + 0.25 * cos((double)i * 0.007 + 2.0 * k);
            f[i * K + k] = (float)((cid == 5 && k == 0) ? -x : x);
        }
    }
}

int main() {
    int fails = 0;
    { std::vector<double> lw; std::vector<float> f;
      const size_t sizes[] = {1, 3, 1024, 1000003};
      for (size_t M : sizes) { gen<1>(0, M, lw, f); if (!run_case<1>("uniform w=1", M, lw, f)) fails++; } }
    { std::vector<double> lw; std::vector<float> f;
      gen<1>(1, 4096, lw, f); if (!run_case<1>("w in 1..7", 4096, lw, f)) fails++;
      gen<1>(2, 4096, lw, f); if (!run_case<1>("30% zero-weight", 4096, lw, f)) fails++;
      gen<1>(3, 1024, lw, f); if (!run_case<1>("tiny w=1e-30", 1024, lw, f)) fails++;
      gen<1>(4, 1024, lw, f); if (!run_case<1>("huge w=1e30", 1024, lw, f)) fails++;
      gen<1>(5, 1000003, lw, f); if (!run_case<1>("w=i+1, signed f", 1000003, lw, f)) fails++;
      gen<1>(6, 1024, lw, f); if (!run_case<1>("all zero-weight", 1024, lw, f)) fails++;
      gen<1>(0, 17, lw, f); if (!run_case<1>("N=17 odd", 17, lw, f)) fails++; }
    { std::vector<double> lw; std::vector<float> f;
      gen<2>(2, 4096, lw, f); if (!run_case<2>("K=2 obs, 30% dead", 4096, lw, f)) fails++;
      gen<2>(1, 65536, lw, f); if (!run_case<2>("K=2 obs, w 1..7", 65536, lw, f)) fails++; }
    { std::vector<double> lw; std::vector<float> f;
      gen<4>(1, 65536, lw, f); if (!run_case<4>("K=4 obs, w 1..7", 65536, lw, f)) fails++; }
    std::printf(fails ? "DUMI_TEST: %d FAILURE(S)\n" : "DUMI_TEST: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
