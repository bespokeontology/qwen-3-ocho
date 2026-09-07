// qf_routed_amd_layer.cu - Spark-side dispatch of the routed-expert branch to
// the four-MI50 tier. Compiled by nvcc; deliberately does NOT include the HIP
// header chain (only the pure-host gate qf_routed_amd.h). The MoE fork calls
// _submit before the Spark shared branch and _wait after it, so the AMD routed
// branch overlaps shared:  submit -> (shared runs) -> wait -> combine.
//
// Gate off (QF_ROUTED_AMD unset) -> both return "use local path" and the
// authority routed launches run unchanged.
#include <cuda_runtime.h>
#include <stdio.h>
#include "qf_routed_amd.h"

#define QFR_NEMBD 2560            // matches QF5_NEMBD; kept local to avoid the HIP chain
#define QFR_MAXM  64

static int   *g_sel = nullptr;    // pinned host scratch, sized once for MAXM
static float *g_wt  = nullptr, *g_x = nullptr;
static float *g_y[2] = {nullptr, nullptr};   // A/B: two waves outstanding
static int    g_yphase = 0;
static int    g_K   = 0;

static int ensure_scratch(int K) {
    if (g_sel) return 0;
    g_K = K;
    if (cudaHostAlloc((void **)&g_sel, (size_t)QFR_MAXM * K * sizeof(int),  cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&g_wt,  (size_t)QFR_MAXM * K * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&g_x,   (size_t)QFR_MAXM * QFR_NEMBD * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&g_y[0], (size_t)QFR_MAXM * QFR_NEMBD * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    if (cudaHostAlloc((void **)&g_y[1], (size_t)QFR_MAXM * QFR_NEMBD * sizeof(float), cudaHostAllocDefault) != cudaSuccess) return -1;
    return 0;
}

// Returns 0 if the batch was submitted to AMD (caller must call _wait later and
// must NOT run the local routed path); 1 if the offload is disabled (caller
// runs the local routed path); <0 on a wire/copy error (caller falls back).
extern "C" int qf_routed_amd_layer_submit(int il, long long pos, int M, int K,
                                          const int *sel_dev, const float *wt_dev,
                                          const float *mixed_dev, cudaStream_t s) {
    if (!qf_routed_amd_enabled()) return 1;
    if (M < 1 || M > QFR_MAXM) return 1;
    if (ensure_scratch(K)) { fprintf(stderr, "[routed-amd] pinned scratch alloc failed\n"); return -1; }
    cudaMemcpyAsync(g_sel, sel_dev, (size_t)M * K * sizeof(int),   cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(g_wt,  wt_dev,  (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(g_x,   mixed_dev, (size_t)M * QFR_NEMBD * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);   // router outputs + activation must be host-side before the send
    return qf_routed_amd_submit(il, pos, M, K, g_sel, g_wt, g_x);  // 0 ok, <0 disables offload
}

// Fills y2560_dev [M*NEMBD] with the routed output from AMD. Returns 0 on
// success, <0 on error (caller falls back to local path for this step).
extern "C" int qf_routed_amd_layer_wait(float *y2560_dev, int M, cudaStream_t s) {
    float *yb = g_y[g_yphase]; g_yphase ^= 1;   // A's H2D may still be in flight
    if (qf_routed_amd_wait(M, yb)) return -1;
    // H2D on stream s; the caller's k_shexp_add runs on s and orders after this
    // copy automatically -> no host-side stream barrier needed here. g_y is
    // pinned and not reused until the next layer's wait, well after this 80 KB
    // copy drains.
    cudaMemcpyAsync(y2560_dev, yb, (size_t)M * QFR_NEMBD * sizeof(float), cudaMemcpyHostToDevice, s);
    return 0;
}
