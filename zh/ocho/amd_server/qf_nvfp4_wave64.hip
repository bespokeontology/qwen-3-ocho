// qf_nvfp4_wave64.hip - fused NVFP4 unpack/dequantize/accumulate GEMV kernels
// for gfx906 (MI50), wave64-safe, no native FP4/FP8 instructions.
//
// Session 05 deliverable. Drop-in replacement for k_nvfp4_gemv in
// prior/09_amd_hip/qf_hip4_stage.hip, specialized for the two expert shapes:
//   640x2560  (gate_proj / up_proj)  - 160 scale groups per row
//   2560x640  (down_proj)            -  40 scale groups per row
// plus fused epilogue variants (gate+up+silu-mul, down+weighted-accumulate).
//
// Checkpoint packing (identical to source/src/cuda/qf.cu k_nvfp4_gemv):
//   scale group g (16 values) = 8 packed bytes at row + g*8, scale byte at
//   srow + g. Byte i: LOW nibble -> value 2i, HIGH nibble -> value 2i+1
//   (adjacent pairs). See qf5_group_dot for the measured ground truth.
//
// Dequant is pure integer bit construction (exact, no libm, no FP4/FP8/bf16
// hardware ops):
//   E2M1  (n&7): e=n>>1, m=n&1 -> e ? flt((e+126)<<23 | m<<22) : m*0.5
//   UE4M3 (b)  : e=b>>3, m=b&7 -> e ? flt((e+120)<<23 | m<<20) : m*2^-9
// Both identities are exact: E2M1 e>0 is 1.m x 2^(e-1); UE4M3 e>0 is
// 1.m x 2^(e-7). Sign of E2M1 applied by negation.
//
// Accumulation: per 16-value group, fp32 partial dot is formed first and the
// UE4M3 scale is applied once per group (acc += sc * dot), which halves the
// multiplies vs the reference (sc folded into every product) and preserves
// val = sign*E2M1*UE4M3*scale2 semantics up to fp32 sum reordering (allowed:
// TARGET.md does not require bit-identical parity).
//
// Wave64 safety:
//   - blockDim = 256 = 4 full wave64s; every __shfl_down_sync uses the full
//     64-lane mask.
//   - 640x2560: one ROW PER HALF-WAVE (32 lanes). 160 groups / 32 lanes = 5
//     exactly -> no idle lanes, no divergence. Reduction offsets 16..1: each
//     half's lanes 0/32 hold their own half's sum (cross-half reads land only
//     in lanes whose values are discarded).
//   - 2560x640: one ROW PER EIGHTH-WAVE (8 lanes). 40 groups / 8 lanes = 5
//     exactly. Reduction offsets 4,2,1, same argument.
//   - x / hidden staged in LDS once per block (reused by 8 / 32 resident
//     rows), killing redundant global traffic for the activation vector.
//
// Build: see Makefile.wave64 (hipcc --offload-arch=gfx906). Host syntax check:
//   g++ -DQF_HOST_CHECK -I. -I../../source/src -fsyntax-only qf_nvfp4_wave64.hip
#include "qf_nvfp4_wave64.h"

#include <math.h>

#if !defined(QF_HOST_CHECK) && defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 60000000
#define __shfl_down_sync(mask, value, delta) __shfl_down((value), (delta), warpSize)
#endif

#define WAVE  64
#define WMASK 0xFFFFFFFFFFFFFFFFULL   // full 64-lane wave

#ifdef QF_HOST_CHECK
#define QF5_LAUNCH(k, g, b, s, ...) k(__VA_ARGS__)
#else
#define QF5_LAUNCH(k, g, b, s, ...) k<<<g, b, 0, s>>>(__VA_ARGS__)
#endif

// ---------------------------------------------------------------------------
// Exact integer dequant helpers (no FP4/FP8 instructions exist on gfx906).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float qf5_e2m1(uint32_t nib) {
    uint32_t mag = nib & 7u, e = mag >> 1, m = mag & 1u;
    float v = e ? __uint_as_float(((e + 126u) << 23) | (m << 22))
                : (m ? 0.5f : 0.f);
    return (nib & 8u) ? -v : v;
}
__device__ __forceinline__ float qf5_ue4m3(uint32_t b) {
    uint32_t e = b >> 3, m = b & 7u;
    if (e == 0) return (float)m * 0x1p-9f;          // subnormal-scale values
    if (e == 15 && m == 7) return INFINITY;         // UE4M3 saturation guard
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}

// fp32 dot of one 16-value NVFP4 scale group against xs[0..15] (LDS-resident).
// pk.x = bytes 0..3, pk.y = bytes 4..7 of the group.
//
// Nibble order: byte i holds value 2i in the LOW nibble and value 2i+1 in the
// HIGH nibble (adjacent pairs).
//
// This previously paired the low nibble of byte j with value j and the high
// nibble with value j+8 - an interleaved layout the checkpoint does not use.
// That is a permutation WITHIN each 16-value block, so it leaves the output's
// magnitude distribution intact while destroying its direction; a
// magnitude-only check cannot see it. Ground truth, dequantizing
// layers.0.mlp.experts.0.gate_proj against the BF16 original:
//     adjacent pairs  {v[2i] lo, v[2i+1] hi} -> corr 0.9955, maxerr 0.0066
//     interleaved     {v[j] lo,  v[j+8] hi}  -> corr 0.1420, maxerr 0.0809
__device__ __forceinline__ float qf5_group_dot(uint2 pk, const float *xs) {
    float a = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t b0 = (pk.x >> (8 * j)) & 0xFFu;    // byte j     -> v[2j],   v[2j+1]
        uint32_t b1 = (pk.y >> (8 * j)) & 0xFFu;    // byte j + 4 -> v[2j+8], v[2j+9]
        a = fmaf(qf5_e2m1(b0 & 15u), xs[2 * j],     a);
        a = fmaf(qf5_e2m1(b0 >> 4),  xs[2 * j + 1], a);
        a = fmaf(qf5_e2m1(b1 & 15u), xs[2 * j + 8], a);
        a = fmaf(qf5_e2m1(b1 >> 4),  xs[2 * j + 9], a);
    }
    return a;
}

// ---------------------------------------------------------------------------
// 640x2560 (gate / up): row per half-wave, 8 rows per 256-thread block.
// Per row: 160 groups; lane l (0..31) takes groups i*32 + l, i = 0..4.
// Coalescing: the 32 lanes of a half-wave read 32 consecutive uint2 = 256
// contiguous bytes per iteration; scales likewise (32 consecutive bytes).
// ---------------------------------------------------------------------------
__global__ void k_nvfp4_gemv_640x2560(const uint8_t *__restrict__ W,
                                      const uint8_t *__restrict__ S,
                                      const float *__restrict__ x,
                                      float *__restrict__ y,
                                      float scale2, int slot) {
    __shared__ float xs[QF5_NEMBD];
    for (int i = threadIdx.x; i < QF5_NEMBD; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int l   = threadIdx.x & 31;
    int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint8_t *wr = W + (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NEMBD / 2);
    const uint8_t *sr = S + (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NEMBD / 16);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 32 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
        acc = fmaf(qf5_ue4m3(sr[g]), qf5_group_dot(pk, xs + g * 16), acc);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(WMASK, acc, off);   // halves stay independent
    if (l == 0) y[row] = acc * scale2;
}

// ---------------------------------------------------------------------------
// 2560x640 (down): row per eighth-wave, 32 rows per 256-thread block.
// Per row: 40 groups; lane l (0..7) takes groups i*8 + l, i = 0..4.
// Coalescing: the 8 lanes of a row-group read 8 consecutive uint2 = 64
// contiguous bytes (one HBM2 atom) per iteration; a wave covers 8 rows.
// ---------------------------------------------------------------------------
__global__ void k_nvfp4_gemv_2560x640(const uint8_t *__restrict__ W,
                                      const uint8_t *__restrict__ S,
                                      const float *__restrict__ x,
                                      float *__restrict__ y,
                                      float scale2, int slot) {
    __shared__ float xs[QF5_NFF];
    for (int i = threadIdx.x; i < QF5_NFF; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int l   = threadIdx.x & 7;
    int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    const uint8_t *wr = W + (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NFF / 2);
    const uint8_t *sr = S + (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NFF / 16);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 8 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
        acc = fmaf(qf5_ue4m3(sr[g]), qf5_group_dot(pk, xs + g * 16), acc);
    }
    #pragma unroll
    for (int off = 4; off; off >>= 1)
        acc += __shfl_down_sync(WMASK, acc, off);   // eighths stay independent
    if (l == 0) y[row] = acc * scale2;
}

// ---------------------------------------------------------------------------
// Fused gate+up+silu-mul for one expert slot:
//   hidden[r] = silu(dot(g_r, x) * s2g) * (dot(u_r, x) * s2u),  r < 640
// Same half-wave geometry as k_nvfp4_gemv_640x2560; each lane walks its 5
// groups in BOTH matrices (10 group dots, 2 accumulators). Replaces two gemv
// launches + k_silu_mul and stages x once instead of twice.
// silu order matches k_silu_mul exactly: g / (1 + exp(-g)) * u.
// The body is shared by the value-based kernel (slot/s2 passed by value) and
// the wave4 device-indexed kernel (slot/s2 resolved on device below).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void qf5_gateup_body(const uint8_t *__restrict__ Wg,
                                                const uint8_t *__restrict__ Sg,
                                                const uint8_t *__restrict__ Wu,
                                                const uint8_t *__restrict__ Su,
                                                const float *__restrict__ x,
                                                float *__restrict__ hidden,
                                                float s2g, float s2u, int slot) {
    __shared__ float xs[QF5_NEMBD];
    for (int i = threadIdx.x; i < QF5_NEMBD; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    // One row per HALF-WAVE, so rows-per-block == blockDim/32. Widening the
    // block cuts the REDUNDANT activation staging: every block re-reads the
    // same 10 KB x vector into LDS, so at 8 rows/block that is 8.2 MB of the
    // 26.6 MB an expert layer moves - 31%. This is v100-skinny's "activation
    // traffic per weight byte" lever, which we did not have.
    int l   = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    size_t woff = (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NEMBD / 2);
    size_t soff = (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NEMBD / 16);
    const uint8_t *wg = Wg + woff, *wu = Wu + woff;
    const uint8_t *sg = Sg + soff, *su = Su + soff;
    float ag = 0.f, au = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 32 + l;
        uint2 pg = *(const uint2 *)(wg + (size_t)g * 8);
        uint2 pu = *(const uint2 *)(wu + (size_t)g * 8);
        float dg = qf5_group_dot(pg, xs + g * 16);
        float du = qf5_group_dot(pu, xs + g * 16);
        ag = fmaf(qf5_ue4m3(sg[g]), dg, ag);
        au = fmaf(qf5_ue4m3(su[g]), du, au);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        ag += __shfl_down_sync(WMASK, ag, off);
        au += __shfl_down_sync(WMASK, au, off);
    }
    if (l == 0) {
        float gv = ag * s2g, uv = au * s2u;
        hidden[row] = gv / (1.f + expf(-gv)) * uv;
    }
}
__global__ void k_nvfp4_gateup_640x2560(const uint8_t *__restrict__ Wg,
                                        const uint8_t *__restrict__ Sg,
                                        const uint8_t *__restrict__ Wu,
                                        const uint8_t *__restrict__ Su,
                                        const float *__restrict__ x,
                                        float *__restrict__ hidden,
                                        float s2g, float s2u, int slot) {
    qf5_gateup_body(Wg, Sg, Wu, Su, x, hidden, s2g, s2u, slot);
}
// Device-indexed variant (wave4 session 09): expert id, slot and scale_2 are
// read on device from the router outputs. Thread 0 resolves and broadcasts
// via shared memory; slot < 0 (unserviced miss -- impossible when the compute
// stream waits the transfer event) fails safe by zeroing this block's rows.
__global__ void k_nvfp4_gateup_640x2560_d(const uint8_t *__restrict__ Wg,
                                          const uint8_t *__restrict__ Sg,
                                          const uint8_t *__restrict__ Wu,
                                          const uint8_t *__restrict__ Su,
                                          const float *__restrict__ x,
                                          float *__restrict__ hidden,
                                          const float *__restrict__ s2g_dev,
                                          const float *__restrict__ s2u_dev,
                                          const int *__restrict__ sel_ids,
                                          const int *__restrict__ slot_ids,
                                          int k) {
    __shared__ int   sh_slot;
    __shared__ float sh_s2g, sh_s2u;
    if (threadIdx.x == 0) {
        int e  = sel_ids[k];
        int sl = slot_ids[k];
        sh_slot = sl;
        sh_s2g = (sl >= 0) ? s2g_dev[e] : 0.f;
        sh_s2u = (sl >= 0) ? s2u_dev[e] : 0.f;
    }
    __syncthreads();
    if (sh_slot < 0) {                      // uniform fail-safe branch
        int row = blockIdx.x * 8 + (threadIdx.x >> 5);
        if ((threadIdx.x & 31) == 0) hidden[row] = 0.f;
        return;
    }
    qf5_gateup_body(Wg, Sg, Wu, Su, x, hidden, sh_s2g, sh_s2u, sh_slot);
}

// ---------------------------------------------------------------------------
// Fused down + weighted accumulate for one expert slot:
//   y[r] += w * (dot(d_r, hidden) * s2d),  r < 2560
// Same eighth-wave geometry as k_nvfp4_gemv_2560x640. Replaces the gemv into
// scratch + k_axpy pair; each row has exactly one writer so the read-add-
// write needs no atomics. y must be zeroed once per token before the top-10
// expert loop (k_zero on the stage stream, as today).
// Body shared by the value-based and device-indexed kernels, as above.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void qf5_downacc_body(const uint8_t *__restrict__ W,
                                                 const uint8_t *__restrict__ S,
                                                 const float *__restrict__ h,
                                                 float w,
                                                 float *__restrict__ y,
                                                 float scale2, int slot) {
    __shared__ float xs[QF5_NFF];
    for (int i = threadIdx.x; i < QF5_NFF; i += blockDim.x) xs[i] = h[i];
    __syncthreads();
    int l   = threadIdx.x & 7;
    int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    const uint8_t *wr = W + (size_t)slot * QF5_EXP_W_BYTES + (size_t)row * (QF5_NFF / 2);
    const uint8_t *sr = S + (size_t)slot * QF5_EXP_S_BYTES + (size_t)row * (QF5_NFF / 16);
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int g = i * 8 + l;
        uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
        acc = fmaf(qf5_ue4m3(sr[g]), qf5_group_dot(pk, xs + g * 16), acc);
    }
    #pragma unroll
    for (int off = 4; off; off >>= 1)
        acc += __shfl_down_sync(WMASK, acc, off);
    if (l == 0) y[row] = fmaf(w, acc * scale2, y[row]);
}
__global__ void k_nvfp4_downacc_2560x640(const uint8_t *__restrict__ W,
                                         const uint8_t *__restrict__ S,
                                         const float *__restrict__ h,
                                         float w,
                                         float *__restrict__ y,
                                         float scale2, int slot) {
    qf5_downacc_body(W, S, h, w, y, scale2, slot);
}
// Device-indexed variant (wave4 session 09): slot, scale_2 and the mixing
// weight w = wts[k] are read on device. slot < 0 fails safe: y untouched.
__global__ void k_nvfp4_downacc_2560x640_d(const uint8_t *__restrict__ W,
                                           const uint8_t *__restrict__ S,
                                           const float *__restrict__ h,
                                           const float *__restrict__ wts,
                                           float *__restrict__ y,
                                           const float *__restrict__ s2d_dev,
                                           const int *__restrict__ sel_ids,
                                           const int *__restrict__ slot_ids,
                                           int k) {
    __shared__ int   sh_slot;
    __shared__ float sh_s2d, sh_w;
    if (threadIdx.x == 0) {
        int e  = sel_ids[k];
        int sl = slot_ids[k];
        sh_slot = sl;
        sh_s2d = (sl >= 0) ? s2d_dev[e] : 0.f;
        sh_w   = (sl >= 0) ? wts[k] : 0.f;
    }
    __syncthreads();
    if (sh_slot < 0) return;                // uniform fail-safe: y untouched
    qf5_downacc_body(W, S, h, sh_w, y, sh_s2d, sh_slot);
}

// ---------------------------------------------------------------------------
// Launchers. Grid is exact for both shapes (80 blocks x 256 threads); the
// row-count argument is gone because the shapes are baked.
// ---------------------------------------------------------------------------
void qf5_nvfp4_gemv_640x2560(const uint8_t *W, const uint8_t *S,
                             const float *x, float *y, float scale2,
                             int slot, hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_gemv_640x2560, 80, 256, stream, W, S, x, y, scale2, slot);
}
void qf5_nvfp4_gemv_2560x640(const uint8_t *W, const uint8_t *S,
                             const float *x, float *y, float scale2,
                             int slot, hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_gemv_2560x640, 80, 256, stream, W, S, x, y, scale2, slot);
}
void qf5_nvfp4_gateup_640x2560(const uint8_t *Wg, const uint8_t *Sg,
                               const uint8_t *Wu, const uint8_t *Su,
                               const float *x, float *hidden,
                               float s2g, float s2u, int slot,
                               hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_gateup_640x2560, 80, 256, stream,
               Wg, Sg, Wu, Su, x, hidden, s2g, s2u, slot);
}
void qf5_nvfp4_downacc_2560x640(const uint8_t *W, const uint8_t *S,
                                const float *h, float w, float *y,
                                float s2d, int slot, hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_downacc_2560x640, 80, 256, stream, W, S, h, w, y, s2d, slot);
}
// Device-indexed launchers (wave4 session 09), same 80x256 geometry.
// ---------------------------------------------------------------------------
// FUSED ALL-EXPERT PATH: one launch for all NEXPUSED experts instead of one
// launch each.
//
// Why the old path was serial: `hidden` was a single 640-float scratch, so
// expert k's downacc had to finish before expert k+1's gateup overwrote it.
// That forced gateup(0), downacc(0), gateup(1), downacc(1), ... = 20 launches
// per MoE layer, 240 per stage per token, each only 80 blocks on a 60-CU card
// (~1.3 blocks/CU). Measured: the expert section ran 1926 us/layer against a
// byte count that justifies ~30 us. It was launch- and occupancy-bound, not
// bandwidth-bound.
//
// Giving each expert its own hidden slice (hidden[k * QF5_NFF]) removes the
// dependency:
//   gateup  grid (80, NEXPUSED) - expert k = blockIdx.y, own slice, no race
//   downacc grid  80            - each block owns 32 rows and loops all
//                                 experts in registers, so there is still
//                                 exactly ONE writer per row and no atomics.
//                                 It also seeds y instead of accumulating,
//                                 which retires the separate zeroing pass.
// Determinism is preserved: the k-loop runs in fixed order inside one thread.
// ---------------------------------------------------------------------------
__global__ void k_nvfp4_gateup_all(const uint8_t *__restrict__ Wg,
                                   const uint8_t *__restrict__ Sg,
                                   const uint8_t *__restrict__ Wu,
                                   const uint8_t *__restrict__ Su,
                                   const float *__restrict__ x,
                                   float *__restrict__ hidden,
                                   const float *__restrict__ s2g_dev,
                                   const float *__restrict__ s2u_dev,
                                   const int *__restrict__ sel_ids,
                                   const int *__restrict__ slot_ids) {
    const int k = blockIdx.y;
    __shared__ int   sh_slot;
    __shared__ float sh_s2g, sh_s2u;
    if (threadIdx.x == 0) {
        int e  = sel_ids[k];
        int sl = slot_ids[k];
        sh_slot = sl;
        sh_s2g = (sl >= 0) ? s2g_dev[e] : 0.f;
        sh_s2u = (sl >= 0) ? s2u_dev[e] : 0.f;
    }
    __syncthreads();
    float *hk = hidden + (size_t)k * QF5_NFF;
    if (sh_slot < 0) {
        int row = blockIdx.x * 8 + (threadIdx.x >> 5);
        if ((threadIdx.x & 31) == 0) hk[row] = 0.f;
        return;
    }
    qf5_gateup_body(Wg, Sg, Wu, Su, x, hk, sh_s2g, sh_s2u, sh_slot);
}

// One block owns 32 output rows and walks every expert for them.
__global__ void k_nvfp4_downacc_all(const uint8_t *__restrict__ W,
                                    const uint8_t *__restrict__ S,
                                    const float *__restrict__ hidden,
                                    const float *__restrict__ wts,
                                    float *__restrict__ y,
                                    const float *__restrict__ s2d_dev,
                                    const int *__restrict__ sel_ids,
                                    const int *__restrict__ slot_ids,
                                    int nexp) {
    __shared__ float xs[QF5_NEXPUSED][QF5_NFF];
    __shared__ int   sh_slot[QF5_NEXPUSED];
    __shared__ float sh_s2d[QF5_NEXPUSED], sh_w[QF5_NEXPUSED];
    for (int k = 0; k < nexp; k++) {
        for (int i = threadIdx.x; i < QF5_NFF; i += blockDim.x)
            xs[k][i] = hidden[(size_t)k * QF5_NFF + i];
        if (threadIdx.x == 0) {
            int e  = sel_ids[k];
            int sl = slot_ids[k];
            sh_slot[k] = sl;
            sh_s2d[k]  = (sl >= 0) ? s2d_dev[e] : 0.f;
            sh_w[k]    = (sl >= 0) ? wts[k] : 0.f;
        }
    }
    __syncthreads();

    const int l   = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    float total = 0.f;
    for (int k = 0; k < nexp; k++) {
        if (sh_slot[k] < 0) continue;
        const uint8_t *wr = W + (size_t)sh_slot[k] * QF5_EXP_W_BYTES +
                            (size_t)row * (QF5_NFF / 2);
        const uint8_t *sr = S + (size_t)sh_slot[k] * QF5_EXP_S_BYTES +
                            (size_t)row * (QF5_NFF / 16);
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            int g = i * 8 + l;
            uint2 pk = *(const uint2 *)(wr + (size_t)g * 8);
            acc = fmaf(qf5_ue4m3(sr[g]), qf5_group_dot(pk, xs[k] + g * 16), acc);
        }
        #pragma unroll
        for (int off = 4; off; off >>= 1)
            acc += __shfl_down_sync(WMASK, acc, off);
        total = fmaf(sh_w[k], acc * sh_s2d[k], total);   // fixed k order
    }
    if (l == 0) y[row] = total;      // sole writer; seeds y, no zeroing pass
}

void qf5_nvfp4_moe_all(const uint8_t *Wg, const uint8_t *Sg,
                       const uint8_t *Wu, const uint8_t *Su,
                       const uint8_t *Wd, const uint8_t *Sd,
                       const float *x, float *hidden, float *y,
                       const float *s2g_dev, const float *s2u_dev,
                       const float *s2d_dev, const float *wts,
                       const int *sel_ids, const int *slot_ids,
                       int nexp, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)Wg;(void)Sg;(void)Wu;(void)Su;(void)Wd;(void)Sd;(void)x;(void)hidden;
    (void)y;(void)s2g_dev;(void)s2u_dev;(void)s2d_dev;(void)wts;(void)sel_ids;
    (void)slot_ids;(void)nexp;(void)stream;
#else
    // QF5_GATEUP_THREADS sets rows/block (threads/32). 256 -> 8 rows, 80
    // blocks, 31% of traffic redundant activation. 512 -> 16 rows, 40 blocks,
    // 18%. Measured, not assumed - see the gate.
    const int rpb = QF5_GATEUP_THREADS / 32;
    dim3 g((unsigned)(QF5_NFF / rpb), (unsigned)nexp);
    k_nvfp4_gateup_all<<<g, QF5_GATEUP_THREADS, 0, stream>>>(
        Wg, Sg, Wu, Su, x, hidden, s2g_dev, s2u_dev, sel_ids, slot_ids);
    k_nvfp4_downacc_all<<<80, 256, 0, stream>>>(Wd, Sd, hidden, wts, y, s2d_dev,
                                                sel_ids, slot_ids, nexp);
#endif
}

void qf5_nvfp4_gateup_640x2560_d(const uint8_t *Wg, const uint8_t *Sg,
                                 const uint8_t *Wu, const uint8_t *Su,
                                 const float *x, float *hidden,
                                 const float *s2g_dev, const float *s2u_dev,
                                 const int *sel_ids, const int *slot_ids,
                                 int k, hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_gateup_640x2560_d, 80, 256, stream,
               Wg, Sg, Wu, Su, x, hidden, s2g_dev, s2u_dev, sel_ids, slot_ids, k);
}
void qf5_nvfp4_downacc_2560x640_d(const uint8_t *W, const uint8_t *S,
                                  const float *h, const float *wts, float *y,
                                  const float *s2d_dev,
                                  const int *sel_ids, const int *slot_ids,
                                  int k, hipStream_t stream) {
    QF5_LAUNCH(k_nvfp4_downacc_2560x640_d, 80, 256, stream,
               W, S, h, wts, y, s2d_dev, sel_ids, slot_ids, k);
}

// ---- the two halves of qf5_nvfp4_moe_all as separate launches (M=1 product
// path: the stage executor puts a section mark between them). Same kernels,
// same geometry, same buffers.
void qf5_nvfp4_gateup_all_launch(const uint8_t *Wg, const uint8_t *Sg,
                                 const uint8_t *Wu, const uint8_t *Su,
                                 const float *x, float *hidden,
                                 const float *s2g_dev, const float *s2u_dev,
                                 const int *sel_ids, const int *slot_ids,
                                 int nexp, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)Wg;(void)Sg;(void)Wu;(void)Su;(void)x;(void)hidden;(void)s2g_dev;(void)s2u_dev;
    (void)sel_ids;(void)slot_ids;(void)nexp;(void)stream;
#else
    const int rpb = QF5_GATEUP_THREADS / 32;
    dim3 g((unsigned)(QF5_NFF / rpb), (unsigned)nexp);
    k_nvfp4_gateup_all<<<g, QF5_GATEUP_THREADS, 0, stream>>>(
        Wg, Sg, Wu, Su, x, hidden, s2g_dev, s2u_dev, sel_ids, slot_ids);
#endif
}
void qf5_nvfp4_downacc_all_launch(const uint8_t *Wd, const uint8_t *Sd,
                                  const float *hidden, const float *wts, float *y,
                                  const float *s2d_dev, const int *sel_ids,
                                  const int *slot_ids, int nexp, hipStream_t stream) {
#ifdef QF_HOST_CHECK
    (void)Wd;(void)Sd;(void)hidden;(void)wts;(void)y;(void)s2d_dev;(void)sel_ids;(void)slot_ids;(void)nexp;(void)stream;
#else
    k_nvfp4_downacc_all<<<80, 256, 0, stream>>>(Wd, Sd, hidden, wts, y, s2d_dev,
                                                sel_ids, slot_ids, nexp);
#endif
}
