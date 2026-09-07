// rng_test.cu — Philox4x32-10 acceptance: KAT, determinism, stream independence,
// counter snapshot/restore, distribution sanity. Exit 0 = all pass.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <set>
#include <vector>
#include "cuda_runtime.h"
#include "philox.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void k_fill(uint32_t seed_lo, uint32_t seed_hi, uint32_t s0, uint32_t S, uint32_t B, uint32_t L, uint4* out) {
    const uint64_t cell = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)S * B * L;
    if (cell >= total) return;
    const uint32_t l = (uint32_t)(cell % L);
    const uint32_t b = (uint32_t)((cell / L) % B);
    const uint32_t s = s0 + (uint32_t)(cell / ((uint64_t)L * B));
    ph4x32_ctr c = stoch_rand4(seed_lo, seed_hi, s, b, 1u, l);
    uint4 o; o.x = c.v[0]; o.y = c.v[1]; o.z = c.v[2]; o.w = c.v[3];
    out[cell] = o;
}

__global__ void k_unif_stats(uint32_t seed_lo, uint32_t seed_hi, uint64_t N,
                             double* p_sum, double* p_sumsq, float* p_min, float* p_max) {
    double s = 0.0, sq = 0.0; float mn = 1e30f, mx = -1e30f;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < N; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t w = stoch_rand4(seed_lo, seed_hi, (uint32_t)(i >> 2), 0u, 1u, (uint32_t)(i & 3)).v[0];
        const float u = stoch_uniform_u32(w);
        s += u; sq += (double)u * u; mn = fminf(mn, u); mx = fmaxf(mx, u);
    }
    __shared__ double sh_s[256], sh_q[256]; __shared__ float sh_mn[256], sh_mx[256];
    const int tid = threadIdx.x;
    sh_s[tid] = s; sh_q[tid] = sq; sh_mn[tid] = mn; sh_mx[tid] = mx;
    __syncthreads();
    for (int o = 128; o; o >>= 1) {
        if (tid < o) {
            sh_s[tid] += sh_s[tid + o]; sh_q[tid] += sh_q[tid + o];
            sh_mn[tid] = fminf(sh_mn[tid], sh_mn[tid + o]);
            sh_mx[tid] = fmaxf(sh_mx[tid], sh_mx[tid + o]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        p_sum[blockIdx.x] = sh_s[0]; p_sumsq[blockIdx.x] = sh_q[0];
        p_min[blockIdx.x] = sh_mn[0]; p_max[blockIdx.x] = sh_mx[0];
    }
}

__global__ void k_kat(uint4* out) {
    ph4x32_ctr c{{0, 0, 0, 0}}; ph4x32_key k{{0, 0}};
    ph4x32_ctr r = philox4x32_10(c, k);
    uint4 o; o.x = r.v[0]; o.y = r.v[1]; o.z = r.v[2]; o.w = r.v[3];
    *out = o;
}

struct Cell { uint32_t x, y, z, w; bool operator<(const Cell& o) const {
    if (x != o.x) return x < o.x; if (y != o.y) return y < o.y;
    if (z != o.z) return z < o.z; return w < o.w; } };

int main() {
    int fails = 0;

    // T1: KAT, host and device
    {
        ph4x32_ctr c{{0, 0, 0, 0}}; ph4x32_key k{{0, 0}};
        ph4x32_ctr r = philox4x32_10(c, k);
        const uint32_t exp[4] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u};
        bool ok = r.v[0] == exp[0] && r.v[1] == exp[1] && r.v[2] == exp[2] && r.v[3] == exp[3];
        uint4* d; CK(cudaMalloc(&d, sizeof(uint4)));
        k_kat<<<1, 1>>>(d);
        CK(cudaDeviceSynchronize());
        uint4 h; CK(cudaMemcpy(&h, d, sizeof(uint4), cudaMemcpyDeviceToHost));
        bool okd = h.x == exp[0] && h.y == exp[1] && h.z == exp[2] && h.w == exp[3];
        std::printf("T1 KAT (ctr=key=0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8): host=%s device=%s\n",
                    ok ? "PASS" : "FAIL", okd ? "PASS" : "FAIL");
        if (!(ok && okd)) fails++;
        CK(cudaFree(d));
    }

    // T2: determinism — identical launches give identical bytes
    {
        const uint32_t S = 32, B = 4, L = 4; const size_t n = S * B * L;
        uint4 *d1, *d2; CK(cudaMalloc(&d1, n * sizeof(uint4))); CK(cudaMalloc(&d2, n * sizeof(uint4)));
        k_fill<<<((unsigned)n + 255) / 256, 256>>>(0xC0FFEEu, 0xDEADu, 0u, S, B, L, d1);
        k_fill<<<((unsigned)n + 255) / 256, 256>>>(0xC0FFEEu, 0xDEADu, 0u, S, B, L, d2);
        CK(cudaDeviceSynchronize());
        std::vector<uint4> h1(n), h2(n);
        CK(cudaMemcpy(h1.data(), d1, n * sizeof(uint4), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), d2, n * sizeof(uint4), cudaMemcpyDeviceToHost));
        bool ok = memcmp(h1.data(), h2.data(), n * sizeof(uint4)) == 0;
        std::printf("T2 determinism (2 launches, %zu cells): %s\n", n, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        CK(cudaFree(d1)); CK(cudaFree(d2));
    }

    // T3: stream independence — all cells unique; distinct branches disagree at every step
    {
        const uint32_t S = 64, B = 8, L = 4; const size_t n = S * B * L;
        uint4* d; CK(cudaMalloc(&d, n * sizeof(uint4)));
        k_fill<<<((unsigned)n + 255) / 256, 256>>>(0x12345678u, 0x9ABCDEF0u, 0u, S, B, L, d);
        CK(cudaDeviceSynchronize());
        std::vector<uint4> h(n);
        CK(cudaMemcpy(h.data(), d, n * sizeof(uint4), cudaMemcpyDeviceToHost));
        std::set<Cell> seen; bool uniq = true;
        for (const auto& o : h) { Cell c{ o.x, o.y, o.z, o.w }; if (!seen.insert(c).second) { uniq = false; break; } }
        uint32_t diff = 0;
        for (uint32_t s = 0; s < S; s++) {
            const uint4 a = h[(size_t)s * B * L], b = h[(size_t)s * B * L + L];
            if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) diff++;
        }
        const bool ok = uniq && diff == S;
        std::printf("T3 independence (%zu cells unique=%s, branch0-vs-1 step disagreements=%u/%u): %s\n",
                    n, uniq ? "yes" : "NO", diff, S, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        CK(cudaFree(d));
    }

    // T4: counter snapshot/restore — mid-stream cells recomputed from (seed, branch, step, lane)
    {
        const uint32_t S = 100, B = 1, L = 4;
        uint4* d; CK(cudaMalloc(&d, (size_t)S * B * L * sizeof(uint4)));
        k_fill<<<((unsigned)(S * B * L) + 255) / 256, 256>>>(7u, 0u, 0u, S, B, L, d);
        CK(cudaDeviceSynchronize());
        std::vector<uint4> full(S * B * L);
        CK(cudaMemcpy(full.data(), d, full.size() * sizeof(uint4), cudaMemcpyDeviceToHost));
        const uint32_t s0 = 50, SN = 10;
        k_fill<<<((unsigned)(SN * B * L) + 255) / 256, 256>>>(7u, 0u, s0, SN, B, L, d);
        CK(cudaDeviceSynchronize());
        std::vector<uint4> slice(SN * B * L);
        CK(cudaMemcpy(slice.data(), d, slice.size() * sizeof(uint4), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (uint32_t i = 0; i < SN * B * L; i++)
            if (memcmp(&slice[i], &full[s0 * B * L + i], sizeof(uint4)) != 0) { ok = false; break; }
        std::printf("T4 counter snapshot/restore (steps %u..%u): %s\n", s0, s0 + SN - 1, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        CK(cudaFree(d));
    }

    // T5: distribution sanity — 1M uniforms, mean/var/min/max
    {
        const uint64_t N = 1u << 20; const unsigned grid = 4096;
        double *d_s, *d_q; float *d_mn, *d_mx;
        CK(cudaMalloc(&d_s, grid * sizeof(double))); CK(cudaMalloc(&d_q, grid * sizeof(double)));
        CK(cudaMalloc(&d_mn, grid * sizeof(float))); CK(cudaMalloc(&d_mx, grid * sizeof(float)));
        k_unif_stats<<<grid, 256>>>(42u, 0u, N, d_s, d_q, d_mn, d_mx);
        CK(cudaDeviceSynchronize());
        std::vector<double> ps(grid), pq(grid); std::vector<float> pmn(grid), pmx(grid);
        CK(cudaMemcpy(ps.data(), d_s, grid * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pq.data(), d_q, grid * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pmn.data(), d_mn, grid * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pmx.data(), d_mx, grid * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0, sumsq = 0; float mn = 1e30f, mx = -1e30f;
        for (unsigned b = 0; b < grid; b++) { sum += ps[b]; sumsq += pq[b]; mn = fminf(mn, pmn[b]); mx = fmaxf(mx, pmx[b]); }
        const double mean = sum / (double)N, var = sumsq / (double)N - mean * mean;
        const bool ok = fabs(mean - 0.5) < 2e-3 && fabs(var - 1.0 / 12.0) < 2e-3 && mn >= 0.0f && mx < 1.0f;
        std::printf("T5 distribution (N=%llu): mean=%.6f var=%.6f min=%.6f max=%.6f -> %s\n",
                    (unsigned long long)N, mean, var, mn, mx, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        CK(cudaFree(d_s)); CK(cudaFree(d_q)); CK(cudaFree(d_mn)); CK(cudaFree(d_mx));
    }

    std::printf(fails ? "RNG_TEST: %d FAILURE(S)\n" : "RNG_TEST: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
