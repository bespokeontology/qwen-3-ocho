// kolmo_field.cu — synthetic Kolmogorov field (Lane A prototype of the stochastic-field
// advance): one conditioned parent x0 forked into M GPU-native stochastic children, advanced
// as ONE field of OU dynamics with per-step kill-coin pruning (Russian roulette with weight
// compensation => unbiased), cub-based dynamic branch compaction, fused observable
// accumulation, Dumitrescu reduction, and a stochastic rendering artifact (2D density
// heatmap, PPM). All sampling/integrand/reduction on GPU; CPU sees only final stats.
//
// Observables (analytic gates, all unbiased under the kill weights):
//   f0 = x_T          E = a^T * x0
//   f1 = x_T^2        E = (a^T x0)^2 + sig^2 (1 - a^{2T})/(1 - a^2)
//   f2 = (1/T) sum_t x_t^2   E = (1/T) sum_t [ a^{2t} x0^2 + sig^2 (1-a^{2t})/(1-a^2) ]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cuda_runtime.h"
#include "philox.cuh"
#include "../reduce/dumi.cuh"
#include <cub/cub.cuh>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

struct Br { float x; float acc; double lw; };   // 16 B per branch

__global__ void k_init(Br* b, uint64_t M, float x0) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x) {
        b[i].x = x0; b[i].acc = 0.f; b[i].lw = 0.0;
    }
}

// advance one step: OU x <- a*x + sig*xi (Box-Muller from GEN1-extractor uniforms),
// kill coin u3 < pkill => weight zero (removed by compaction), else lw -= log(1-p).
__global__ void k_advance(const Br* __restrict__ b, Br* __restrict__ bn, uint32_t* __restrict__ dead,
                          uint32_t seed_lo, uint32_t seed_hi, uint32_t step, uint64_t M,
                          float a, float sig, float pkill) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t w0 = stoch_rand4(seed_lo, seed_hi, step, (uint32_t)i, 1u, 0u).v[0];
        const uint32_t w1 = stoch_rand4(seed_lo, seed_hi, step, (uint32_t)i, 1u, 1u).v[0];
        const uint32_t w2 = stoch_rand4(seed_lo, seed_hi, step, (uint32_t)i, 1u, 2u).v[0];
        const double u1 = ((double)(w0 >> 8) + 0.5) * (1.0 / 16777216.0);
        const double u2 = ((double)(w1 >> 8) + 0.5) * (1.0 / 16777216.0);
        const double u3 = ((double)(w2 >> 8) + 0.5) * (1.0 / 16777216.0);
        const double xi = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
        const float nx = (float)((double)a * (double)b[i].x + (double)sig * xi);
        bn[i].x = nx; bn[i].acc = b[i].acc + nx * nx;
        if (u3 < (double)pkill) { dead[i] = 1u; bn[i].lw = -1e300; }
        else                    { dead[i] = 0u; bn[i].lw = b[i].lw - log(1.0 - (double)pkill); }
    }
}

__global__ void k_not(const uint32_t* __restrict__ dead, uint32_t* __restrict__ alive, uint64_t M) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x)
        alive[i] = 1u - dead[i];
}

__global__ void k_getlw(const Br* __restrict__ b, double* __restrict__ lw, uint64_t M) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x)
        lw[i] = b[i].lw;
}

__global__ void k_compact(const Br* __restrict__ in, Br* __restrict__ out,
                          const uint32_t* __restrict__ off, const uint32_t* __restrict__ alive, uint64_t M) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x) {
        if (alive[i]) out[off[i]] = in[i];
    }
}

// final observables f0=x, f1=x^2, f2=acc/T  (dead branches carry lw=-1e300 and are skipped)
__global__ void k_observe(const Br* __restrict__ b, float* __restrict__ f, uint64_t M, float invT) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x) {
        f[(size_t)i * 3 + 0] = b[i].x;
        f[(size_t)i * 3 + 1] = b[i].x * b[i].x;
        f[(size_t)i * 3 + 2] = b[i].acc * invT;
    }
}

// 2D density heatmap of (x_T, x_{T-1}) over surviving branches
__global__ void k_hist2d(const Br* __restrict__ b, const Br* __restrict__ bp, uint64_t M,
                         float* __restrict__ H, int W, int HH, float lo, float hi) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < M; i += (uint64_t)gridDim.x * blockDim.x) {
        if (b[i].lw < -1e300) continue;
        const float scale = (float)W / (hi - lo);
        int ix = (int)((b[i].x - lo) * scale); ix = max(0, min(W - 1, ix));
        int iy = (int)((bp[i].x - lo) * scale); iy = max(0, min(HH - 1, iy));
        atomicAdd(&H[(size_t)iy * W + ix], 1.0f);
    }
}

int main(int argc, char** argv) {
    const uint64_t M = (argc > 1) ? strtoull(argv[1], 0, 10) : (1u << 20);
    const int T = (argc > 2) ? atoi(argv[2]) : 64;
    const float pkill = (argc > 3) ? (float)atof(argv[3]) : 0.01f;
    const float a = 0.5f, sig = 1.0f, x0 = 1.0f;
    const uint32_t seed_lo = 42, seed_hi = 0;
    const unsigned grid = (unsigned)std::min<uint64_t>((M + 255) / 256, 4096);

    Br *d_cur, *d_nxt; uint32_t *d_dead, *d_alive, *d_off;
    CK(cudaMalloc(&d_cur, M * sizeof(Br))); CK(cudaMalloc(&d_nxt, M * sizeof(Br)));
    CK(cudaMalloc(&d_dead, M * sizeof(uint32_t))); CK(cudaMalloc(&d_alive, M * sizeof(uint32_t)));
    CK(cudaMalloc(&d_off, M * sizeof(uint32_t)));
    void* d_tmp = NULL; size_t tmpBytes = 0;
    CK(cub::DeviceScan::ExclusiveSum(d_tmp, tmpBytes, d_alive, d_off, (int)M));
    CK(cudaMalloc(&d_tmp, tmpBytes));

    cudaEvent_t t0, t1, t2, t3; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);
    cudaEventRecord(t0, 0);
    k_init<<<grid, 256>>>(d_cur, M, x0);
    CK(cudaDeviceSynchronize()); cudaEventRecord(t1, 0);

    uint64_t n = M;
    Br* d_prev = NULL; CK(cudaMalloc(&d_prev, M * sizeof(Br)));
    for (int t = 1; t <= T; t++) {
        CK(cudaMemcpy(d_prev, d_cur, n * sizeof(Br), cudaMemcpyDeviceToDevice));
        k_advance<<<grid, 256>>>(d_cur, d_nxt, d_dead, seed_lo, seed_hi, (uint32_t)t, n, a, sig, pkill);
        CK(cudaDeviceSynchronize());
        if (t < T) {
            k_not<<<grid, 256>>>(d_dead, d_alive, n);
            CK(cub::DeviceScan::ExclusiveSum(d_tmp, tmpBytes, d_alive, d_off, (int)n));
            uint32_t last; CK(cudaMemcpy(&last, d_off + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t lastAlive; CK(cudaMemcpy(&lastAlive, d_alive + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            const uint64_t nn = (uint64_t)last + lastAlive;
            k_compact<<<grid, 256>>>(d_nxt, d_cur, d_off, d_alive, n);
            CK(cudaDeviceSynchronize());
            n = nn;
        } else {
            std::swap((void*&)d_cur, (void*&)d_nxt);   // final state stays in d_cur (uncompacted)
        }
    }
    cudaEventRecord(t2, 0); cudaEventSynchronize(t2);
    float init_ms = 0, adv_ms = 0; cudaEventElapsedTime(&init_ms, t0, t1); cudaEventElapsedTime(&adv_ms, t1, t2);

    // Dumitrescu reduction over the final field (K=3), fused observables
    float* d_f; CK(cudaMalloc(&d_f, M * 3 * sizeof(float)));
    k_observe<<<grid, 256>>>(d_cur, d_f, n, 1.0f / (float)T);
    CK(cudaDeviceSynchronize());
    double* d_lw; CK(cudaMalloc(&d_lw, M * sizeof(double)));
    k_getlw<<<grid, 256>>>(d_cur, d_lw, n);
    const unsigned rgrid = std::min<unsigned>(4096, (unsigned)((n + 255) / 256));
    double* d_part; CK(cudaMalloc(&d_part, (size_t)rgrid * 12 * sizeof(double)));
    CK(cudaMemset(d_part, 0, (size_t)rgrid * 12 * sizeof(double)));
    k_dumi_reduce<3><<<rgrid, 256>>>(d_lw, d_f, n, d_part);
    CK(cudaDeviceSynchronize());
    std::vector<double> part((size_t)rgrid * 12);
    CK(cudaMemcpy(part.data(), d_part, part.size() * sizeof(double), cudaMemcpyDeviceToHost));
    DumiStats st; dumi_merge(part.data(), (int)rgrid, 3, &st);
    cudaEventRecord(t3, 0); cudaEventSynchronize(t3);
    float red_ms = 0; cudaEventElapsedTime(&red_ms, t2, t3);

    const double est0 = st.swf[0] / st.sw, est1 = st.swf[1] / st.sw, est2 = st.swf[2] / st.sw;
    const double at = pow(a, T);
    const double ana0 = at * x0;
    const double ana1 = at * at * x0 * x0 + sig * sig * (1.0 - pow(a, 2 * T)) / (1.0 - a * a);
    double sum = 0.0;
    for (int t = 1; t <= T; t++) {
        const double a2t = pow(a, 2 * t);
        sum += a2t * x0 * x0 + sig * sig * (1.0 - a2t) / (1.0 - a * a);
    }
    const double ana2 = sum / (double)T;

    // rendering artifact: (x_T, x_{T-1}) density
    const int W = 512, HH = 512; const float lo = -3.0f, hi = 3.0f;
    float* d_H; CK(cudaMalloc(&d_H, (size_t)W * HH * sizeof(float)));
    CK(cudaMemset(d_H, 0, (size_t)W * HH * sizeof(float)));
    k_hist2d<<<grid, 256>>>(d_cur, d_prev, n, d_H, W, HH, lo, hi);
    CK(cudaDeviceSynchronize());
    std::vector<float> H((size_t)W * HH);
    CK(cudaMemcpy(H.data(), d_H, H.size() * sizeof(float), cudaMemcpyDeviceToHost));
    float hmax = 0.f; for (float v : H) hmax = std::max(hmax, v);
    FILE* ppm = fopen("kolmo_field_density.ppm", "wb");
    fprintf(ppm, "P3\n%d %d\n255\n", W, HH);
    for (int iy = HH - 1; iy >= 0; iy--) for (int ix = 0; ix < W; ix++) {
        float t = H[(size_t)iy * W + ix] / hmax;
        int r, g, b;
        if (t < 0.33f) { float s = t / 0.33f; r = 0; g = (int)(255 * s * 0.6); b = (int)(255 * (0.4 + 0.6 * s)); }
        else if (t < 0.66f) { float s = (t - 0.33f) / 0.33f; r = (int)(255 * s); g = (int)(255 * (0.6 + 0.4 * s)); b = (int)(255 * (1 - s)); }
        else { float s = std::min(1.f, (t - 0.66f) / 0.34f); r = 255; g = (int)(255 * (1 - 0.6 * s)); b = 0; }
        fprintf(ppm, "%d %d %d\n", r, g, b);
    }
    fclose(ppm);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    std::printf("KOLMO FIELD device=%s M=%llu T=%d a=%.2f sig=%.2f x0=%.2f pkill=%.2f seed=%u\n",
                prop.name, (unsigned long long)M, T, a, sig, x0, pkill, seed_lo);
    std::printf("  fork/init wall=%.3f ms | advance+compact wall=%.3f ms (%d steps) | reduce wall=%.3f ms\n",
                init_ms, adv_ms, T, red_ms);
    std::printf("  trajectories/s=%.0f  n_alive_end=%llu (%.1f%%)  ESS=%.3f  sum_w=%.6e\n",
                (double)M * T / (adv_ms * 1e-3), (unsigned long long)st.n, 100.0 * st.n / M, dumi_ess(st), st.sw);
    std::printf("  est E[X_T]    = %.6e  analytic = %.6e  |err| = %.3e\n", est0, ana0, fabs(est0 - ana0));
    std::printf("  est E[X_T^2]  = %.6f  analytic = %.6f  |err| = %.3e\n", est1, ana1, fabs(est1 - ana1));
    std::printf("  est E[avg x^2]= %.6f  analytic = %.6f  |err| = %.3e\n", est2, ana2, fabs(est2 - ana2));
    std::printf("  render: kolmo_field_density.ppm (%dx%d, max density=%.0f)\n", W, HH, hmax);
    return 0;
}
