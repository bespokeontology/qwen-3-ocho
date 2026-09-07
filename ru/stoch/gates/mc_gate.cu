// mc_gate.cu — analytic Monte Carlo gates. GPU RNG + GPU integrand + GPU reduction,
// no CPU in the sample loop.
//   mode 0 (MC-A):  E[X^2], X ~ U(0,1) = 1/3        (GEN1 §7 extractor, unweighted)
//   mode 1 (MC-A2): importance-sampled E[X^2] = 1/3 via q(x)=2x on [0,1]:
//                   x = sqrt(u_mid), w = 1/(2x)     (midpoint cells; u in [2^-25, 1-2^-25])
//   mode 2 (MC-B):  d=16, E[sum_j X_j^2] = 16/3     (GEN1 §7 extractor, 4 words per sample)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "philox.cuh"
#include "../reduce/dumi.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

template<int MODE>
__global__ void k_mc_gate(uint32_t seed_lo, uint32_t seed_hi, uint64_t N, double* partial, int stride) {
    DumiAccum<1> a;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < N;
         i += (uint64_t)gridDim.x * blockDim.x) {
        float fv; double lw = 0.0;
        if (MODE == 0) {
            const uint32_t w = stoch_rand4(seed_lo, seed_hi, (uint32_t)(i >> 2), 0u, 1u, (uint32_t)(i & 3)).v[0];
            const double x = (double)(w >> 8) * (1.0 / 16777216.0);
            fv = (float)(x * x);
        } else if (MODE == 1) {
            const uint32_t w = stoch_rand4(seed_lo, seed_hi, (uint32_t)(i >> 2), 0u, 1u, (uint32_t)(i & 3)).v[0];
            const double u = ((double)(w >> 8) + 0.5) * (1.0 / 16777216.0);
            const double x = sqrt(u);
            fv = (float)(x * x);
            lw = -log(2.0) - log(x);
        } else if (MODE == 3) {
            // stratified (jittered): S=64 strata, u = (stratum + uniform)/S
            const uint32_t s = (uint32_t)(i % 64u);
            const uint32_t w = stoch_rand4(seed_lo, seed_hi, (uint32_t)(i >> 2), 0u, 1u, (uint32_t)(i & 3)).v[0];
            const double u = ((double)s + (double)(w >> 8) * (1.0 / 16777216.0)) * (1.0 / 64.0);
            fv = (float)(u * u);
        } else {
            double acc = 0.0;
            for (int j = 0; j < 16; j++) {
                const uint32_t w = stoch_rand4(seed_lo, seed_hi, (uint32_t)((i >> 2) + 4 * (j >> 2)), 0u, 1u, (uint32_t)(j & 3)).v[0];
                const double x = (double)(w >> 8) * (1.0 / 16777216.0);
                acc += x * x;
            }
            fv = (float)acc;
        }
        a.add(lw, &fv);
    }
    __shared__ double sm[8][8];   // K=1 -> SW = 6 + 2*K = 8
    dumi_block_commit<1>(a, &sm[0][0], partial + (size_t)blockIdx.x * stride);
}

int main(int argc, char** argv) {
    const int mode = (argc > 1) ? atoi(argv[1]) : 0;
    uint64_t N = (argc > 2) ? strtoull(argv[2], 0, 10) : (1u << 20);
    const uint32_t seed_lo = 42, seed_hi = 0;
    const unsigned grid = (unsigned)std::min<uint64_t>((N + 255) / 256, 4096);
    const int stride = 8;  // K=1 -> 6 + 2
    double* d_part; CK(cudaMalloc(&d_part, (size_t)grid * stride * sizeof(double)));
    CK(cudaMemset(d_part, 0, (size_t)grid * stride * sizeof(double)));
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    if (mode == 0)      k_mc_gate<0><<<grid, 256>>>(seed_lo, seed_hi, N, d_part, stride);
    else if (mode == 1) k_mc_gate<1><<<grid, 256>>>(seed_lo, seed_hi, N, d_part, stride);
    else if (mode == 3) k_mc_gate<3><<<grid, 256>>>(seed_lo, seed_hi, N, d_part, stride);
    else                k_mc_gate<2><<<grid, 256>>>(seed_lo, seed_hi, N, d_part, stride);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);
    float wall_ms = 0.f; cudaEventElapsedTime(&wall_ms, t0, t1);
    std::vector<double> part((size_t)grid * stride);
    CK(cudaMemcpy(part.data(), d_part, part.size() * sizeof(double), cudaMemcpyDeviceToHost));
    DumiStats st; dumi_merge(part.data(), (int)grid, 1, &st);
    const double est = st.swf[0] / st.sw;
    const double analytic = (mode == 2) ? 16.0 / 3.0 : 1.0 / 3.0;
    const double err = fabs(est - analytic);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    const char* names[4] = {"MC-A  E[X^2] U(0,1)=1/3 (unweighted)",
                            "MC-A2 E[X^2]=1/3 via q(x)=2x (weighted)",
                            "MC-B  E[sum X_j^2], d=16 = 16/3 (unweighted)",
                            "MC-A3 E[X^2]=1/3 stratified S=64 jittered (unweighted)"};
    std::printf("GATE %s\n", names[mode]);
    std::printf("  device=%s N=%llu seed=%u\n", prop.name, (unsigned long long)N, seed_lo);
    std::printf("  estimate=%.9f analytic=%.9f |err|=%.3e rel=%.3e\n", est, analytic, err, err / analytic);
    std::printf("  wall=%.3f ms samples/s=%.0f ess=%.3f sum_w=%.6e n=%llu\n",
                wall_ms, (double)N / (wall_ms * 1e-3), dumi_ess(st), st.sw, (unsigned long long)st.n);
    CK(cudaFree(d_part));
    return 0;
}
