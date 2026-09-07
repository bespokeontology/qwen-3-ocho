// FLA (flash-linear-attention) chunked gated delta rule for the Spark chunk pass, STOLEN as
// ahead-of-time Triton cubins (operator 09-06: Spark = NVIDIA-native bulk kernels).
// The five kernels of fla.ops.gated_delta_rule.chunk_gated_delta_rule (cumsum, kkt+solve,
// recompute_w_u, fwd_h, fwd_o) were harvested from a real run on this GB10 (Triton 3.8, FLA
// current; offline recipe fla_aot/harvest_recipe.py, run once) into src/fla_aot/*.cubin. One cubin
// per kernel serves every T (no integer specialization differs across our chunk lengths).
// Metadata is fixed in native code below; the cubins are the only runtime artifact. Triton 3.8 entries take the runtime params in signature order followed by two scratch
// pointers (global, profile), both unused (size 0) here.
// Semantics match k_gdn_prefill_chunked (checked against a sequential reference: cos 0.99999):
//   g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), q/k l2-normalized with eps 1e-6,
//   scale 1/sqrt(128) inside fwd_o, decay-first update, state [H][K][V] fp32 in/out.
// Only the delta-rule core is replaced: conv, gated norm and projections stay in our kernels.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define F_H 48
#define F_D 128
#define F_DINN 10240
#define F_BT 64

struct FlaKernel { CUmodule mod; CUfunction fn; int num_warps, shared; const char *name; const char *cubin; };
// Fixed metadata of the five harvested kernels (see fla_aot/README.md): entry name, cubin file, warps, dynamic smem.
static FlaKernel g_fk[5] = {
    { 0, 0, 1, 0,     "chunk_local_cumsum_scalar_kernel",               "chunk_local_cumsum_scalar_kernel_015981fd043a.cubin" },
    { 0, 0, 1, 9216,  "chunk_gated_delta_rule_fwd_kkt_solve_kernel",    "chunk_gated_delta_rule_fwd_kkt_solve_kernel_a97d0e50caa1.cubin" },
    { 0, 0, 8, 28672, "recompute_w_u_fwd_kernel",                       "recompute_w_u_fwd_kernel_6ebd0c04cb2e.cubin" },
    { 0, 0, 2, 16384, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_a8224ca22210.cubin" },
    { 0, 0, 8, 65536, "chunk_fwd_kernel_o",                             "chunk_fwd_kernel_o_fd65cef187c8.cubin" },
};
static int g_fla_state = 0;   // 0 = not loaded, 1 = ok, -1 = failed
static int fla_load(void) {
    if (g_fla_state) return g_fla_state;
    const char *dir = getenv("QF_FLA_DIR") ? getenv("QF_FLA_DIR") : "./fla_aot";
    char path[512];
    for (int i = 0; i < 5; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, g_fk[i].cubin);
        FILE *c = fopen(path, "rb");
        if (!c) { fprintf(stderr, "fla: cannot open %s\n", path); return g_fla_state = -1; }
        fseek(c, 0, SEEK_END); long n = ftell(c); fseek(c, 0, SEEK_SET);
        void *bytes = malloc(n); if (fread(bytes, 1, n, c) != (size_t)n) { fclose(c); free(bytes); return g_fla_state = -1; }
        fclose(c);
        CUresult r = cuModuleLoadData(&g_fk[i].mod, bytes); free(bytes);
        if (r != CUDA_SUCCESS) { fprintf(stderr, "fla: cuModuleLoadData(%s) = %d\n", g_fk[i].cubin, (int)r); return g_fla_state = -1; }
        r = cuModuleGetFunction(&g_fk[i].fn, g_fk[i].mod, g_fk[i].name);
        if (r != CUDA_SUCCESS) { fprintf(stderr, "fla: cuModuleGetFunction(%s) = %d\n", g_fk[i].name, (int)r); return g_fla_state = -1; }
        if (g_fk[i].shared > 48 * 1024) cuFuncSetAttribute(g_fk[i].fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, g_fk[i].shared);
    }
    fprintf(stderr, "fla: 5 Triton cubins loaded from %s (native Driver API, no Python)\n", dir);
    return g_fla_state = 1;
}
static int fla_launch(const FlaKernel &k, unsigned gx, unsigned gy, unsigned gz, void **params, cudaStream_t s) {
    CUresult r = cuLaunchKernel(k.fn, gx, gy, gz, 32u * k.num_warps, 1, 1, k.shared, (CUstream)s, params, NULL);
    if (r != CUDA_SUCCESS) { const char *e = NULL; cuGetErrorString(r, &e); fprintf(stderr, "fla: launch %s failed: %s\n", k.name, e ? e : "?"); return -1; }
    return 0;
}
// pack: q/k l2-normed, v raw -> bf16 [T][H][D]; g_raw, beta -> fp32 [T][H]. grid (T, H), block D.
__global__ void __launch_bounds__(128) k_fla_pack(const float *__restrict__ qkv, const float *__restrict__ a, const float *__restrict__ b,
                                                  const float *__restrict__ A_log, const float *__restrict__ dt_bias,
                                                  __nv_bfloat16 *__restrict__ q, __nv_bfloat16 *__restrict__ k, __nv_bfloat16 *__restrict__ v,
                                                  float *__restrict__ g, float *__restrict__ beta, int T, int astride) {
    const int t = blockIdx.x, h = blockIdx.y, kh = h / 3, d = threadIdx.x;
    const float *row = qkv + (size_t)t * F_DINN;
    const float qv = row[kh * F_D + d], kv = row[2048 + kh * F_D + d], vv = row[4096 + h * F_D + d];
    __shared__ float red[2][4];
    float sq = qv * qv, sk = kv * kv;
    #pragma unroll
    for (int off = 16; off; off >>= 1) { sq += __shfl_down_sync(0xffffffffu, sq, off); sk += __shfl_down_sync(0xffffffffu, sk, off); }
    if ((d & 31) == 0) { red[0][d >> 5] = sq; red[1][d >> 5] = sk; }
    __syncthreads();
    const float q2 = red[0][0] + red[0][1] + red[0][2] + red[0][3], k2 = red[1][0] + red[1][1] + red[1][2] + red[1][3];
    const float rq = rsqrtf(q2 + 1e-6f), rk = rsqrtf(k2 + 1e-6f);
    const size_t o = ((size_t)t * F_H + h) * F_D + d;
    q[o] = __float2bfloat16(qv * rq); k[o] = __float2bfloat16(kv * rk); v[o] = __float2bfloat16(vv);
    if (d == 0) {
        const float dt = a[(size_t)t * astride + h] + dt_bias[h];
        const float sp = dt > 20.f ? dt : log1pf(expf(dt));
        g[(size_t)t * F_H + h] = -__expf(A_log[h]) * sp;
        beta[(size_t)t * F_H + h] = 1.f / (1.f + expf(-b[(size_t)t * astride + h]));
    }
}
__global__ void k_fla_unpack(const __nv_bfloat16 *__restrict__ o, float *__restrict__ out48, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out48[i] = __bfloat162float(o[i]);
}
static __nv_bfloat16 *g_q, *g_k, *g_v, *g_w, *g_u, *g_vn, *g_o, *g_A, *g_h; static float *g_g, *g_gcs, *g_beta, *g_ht; static int g_T = 0;
static int fla_buffers(int T) {
    if (T <= g_T) return 0;
    const int NT = (T + F_BT - 1) / F_BT;
    if (g_T) { cudaFree(g_q); cudaFree(g_k); cudaFree(g_v); cudaFree(g_w); cudaFree(g_u); cudaFree(g_vn); cudaFree(g_o); cudaFree(g_A); cudaFree(g_h); cudaFree(g_g); cudaFree(g_gcs); cudaFree(g_beta); cudaFree(g_ht); }
    const size_t thd = (size_t)T * F_H * F_D;
    if (cudaMalloc(&g_q, thd * 2) || cudaMalloc(&g_k, thd * 2) || cudaMalloc(&g_v, thd * 2) || cudaMalloc(&g_w, thd * 2) || cudaMalloc(&g_u, thd * 2) ||
        cudaMalloc(&g_vn, thd * 2) || cudaMalloc(&g_o, thd * 2) || cudaMalloc(&g_A, (size_t)T * F_H * F_BT * 2) ||
        cudaMalloc(&g_h, (size_t)NT * F_H * F_D * F_D * 2) || cudaMalloc(&g_g, (size_t)T * F_H * 4) || cudaMalloc(&g_gcs, (size_t)T * F_H * 4) ||
        cudaMalloc(&g_beta, (size_t)T * F_H * 4) || cudaMalloc(&g_ht, (size_t)F_H * F_D * F_D * 4)) { fprintf(stderr, "fla: buffer alloc failed\n"); g_T = 0; return -1; }
    g_T = T; return 0;
}
extern "C" int qf_gdn_fla_prefill(const float *qkv, const float *a, const float *b, const float *A_log, const float *dt_bias,
                                  float *S, float *out48, int T, int astride, int unpack, cudaStream_t s) {
    if (fla_load() < 0 || fla_buffers(T)) return -1;
    k_fla_pack<<<dim3(T, F_H), F_D, 0, s>>>(qkv, a, b, A_log, dt_bias, g_q, g_k, g_v, g_g, g_beta, T, astride);
    const int NT = (T + F_BT - 1) / F_BT;
    int32_t Ti = T; float rcp_ln2 = 1.4426950216f, scale = 0.08838834764831843f; uint64_t z0 = 0, z1 = 0;
    CUdeviceptr pq = (CUdeviceptr)g_q, pk = (CUdeviceptr)g_k, pv = (CUdeviceptr)g_v, pw = (CUdeviceptr)g_w, pu = (CUdeviceptr)g_u, pvn = (CUdeviceptr)g_vn,
                po = (CUdeviceptr)g_o, pA = (CUdeviceptr)g_A, ph = (CUdeviceptr)g_h, pg = (CUdeviceptr)g_g, pgc = (CUdeviceptr)g_gcs, pb = (CUdeviceptr)g_beta,
                ph0 = (CUdeviceptr)S, pht = (CUdeviceptr)g_ht;
    {   void *p[] = {&pg, &pgc, &rcp_ln2, &Ti, &z0, &z1};                              // cumsum(s=g_raw, o=g, scale, T)
        if (fla_launch(g_fk[0], NT, F_H, 1, p, s)) return -1; }
    {   void *p[] = {&pk, &pgc, &pb, &pA, &Ti, &z0, &z1};                              // kkt+solve(k, g, beta, A, T)
        if (fla_launch(g_fk[1], NT, F_H, 1, p, s)) return -1; }
    {   void *p[] = {&pk, &pv, &pb, &pw, &pu, &pA, &pgc, &Ti, &z0, &z1};               // recompute_w_u(k, v, beta, w, u, A, g, T)
        if (fla_launch(g_fk[2], NT, F_H, 1, p, s)) return -1; }
    {   void *p[] = {&pk, &pu, &pw, &pvn, &pgc, &ph, &ph0, &pht, &Ti, &z0, &z1};       // fwd_h(k, v=u, w, v_new, g, h, h0, ht, T)
        if (fla_launch(g_fk[3], (F_D + 31) / 32, F_H, 1, p, s)) return -1; }
    {   void *p[] = {&pq, &pk, &pvn, &ph, &pgc, &po, &scale, &Ti, &z0, &z1};           // fwd_o(q, k, v=v_new, h, g, o, scale, T)
        if (fla_launch(g_fk[4], 1, NT, F_H, p, s)) return -1; }
    if (unpack) { const size_t n = (size_t)T * F_H * F_D; k_fla_unpack<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(g_o, out48, n); }
    if (cudaMemcpyAsync(S, g_ht, (size_t)F_H * F_D * F_D * 4, cudaMemcpyDeviceToDevice, s) != cudaSuccess) return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
extern "C" const void *qf_gdn_fla_out(void) { return g_o; }   // bf16 [T][48][128] output of the last call (fused post path)
