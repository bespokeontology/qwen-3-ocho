// ple.cu - PLE n-gram injection: host hash + host gather from mmap, device projections
#include "../qwenflash.h"
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef NEMBD
#define NEMBD 2560
#endif
#ifndef HCC
#define HCC 4
#endif
#ifndef PLEH
#define PLEH 16
#endif
#ifndef PLEDIM
#define PLEDIM 2560
#endif
#define PLE_HEADS PLEH
#define PLE_ROW (PLEDIM / PLEH)   // per-head row width
#define PLE_ROWS_PER_SHARD 2500012

// fwd decl (model.cpp, C++ mangled)
void *qf_load_dev_global(const QfStore *st, const char *name);

#ifdef QF_CANARY_TAPS
extern int qf_canary_record;
void qf_canary_tap(const char *tag, int il, const float *dev, int n);
#define QF_TAP(tag, il, dev, n) do { \
    if (qf_canary_record) qf_canary_tap((tag), (il), (dev), (n)); \
} while (0)
#else
#define QF_TAP(tag, il, dev, n) ((void)0)
#endif

static void *ple_key_dev, *ple_value_dev, *ple_normk_dev, *ple_normq_dev, *ple_normc_dev, *ple_conv_dev;
static int ple_weights_loaded = 0;


// OCP E4M3 (torch float8_e4m3fn): sign(1) | exponent(4) | mantissa(3), bias 7,
// no infinities; 0x7F/0xFF are NaN.
//
// The PLE n-gram embedding table is SIGNED and roughly half its bytes have the
// sign bit set (measured on the real checkpoint: 50.19%). The previous decode
// took `e = b >> 3`, which folds the sign bit into the exponent and drops the
// sign entirely: a true -1.5 (0xBC) decoded to +98304, and the table's true
// [-240, 208] range read as [0, 15728640]. Those values are projected through
// key_proj/value_proj and added to the hyper-connection residual at layer 1,
// so every downstream layer saw a poisoned state while the logits stayed
// finite - the exact "runs fine, emits word salad" signature.
static float host_e4m3(uint8_t b) {
    uint32_t s = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0) v = ldexpf((float)m, -9);
    else if (e == 15 && m == 7) v = NAN;
    else v = ldexpf((float)(8 + m), (int)e - 10);
    return s ? -v : v;
}

// Per-tensor scale for the n-gram table.
//
// The table ships as F8_E4M3 with ONE BF16 scalar beside it:
//   ...ple_embedding.ngram_embedding.weight_scale = 1.9932e-4
// The raw bytes span [-240, 208]; scaled they span [-0.0478, 0.0415], which is
// an embedding-sized quantity. Reading the table unscaled made the PLE
// injection ~5000x too large and it swamped the hyper-connection residual
// (measured: R absum 57 before the PLE layer, 21457 after), so every one of
// the 48 layers computed on a state that was almost entirely PLE.
//
// This is the LongCat weight_scale_inv trap. A missing scale is FATAL here -
// never fall back to unscaled.
static float g_ple_table_scale = 0.f;

static int ple_load_table_scale(const QfStore *st) {
    static const char *names[2] = {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
    };
    for (int i = 0; i < 2; i++) {
        const QfEntry *e = qf_find(st, names[i]);
        if (!e || e->rec.nbytes != 2) continue;
        uint16_t bits;
        memcpy(&bits, (const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off, 2);
        uint32_t f = (uint32_t)bits << 16;
        float v;
        memcpy(&v, &f, 4);
        if (!(v > 0.f)) break;
        g_ple_table_scale = v;
        fprintf(stderr, "ple: n-gram table scale %.8g (from %s)\n", v, names[i]);
        return 0;
    }
    fprintf(stderr, "ple: FATAL - n-gram table weight_scale missing or invalid; "
                    "refusing to gather an unscaled table\n");
    return -1;
}

// Gathers the 16 n-gram rows as RAW E4M3 BYTES. The decode to float happens on
// the device (k_ple_decode).
//
// NEVER_AGAIN rule 4: the host does HTTP, tokenization, scheduling, sampling
// and DMA orchestration - not tensor math. Converting 2560 E4M3 values per
// token on the host was tensor math on the decode critical path. Staging bytes
// also shrinks the transfer 4x (2560 B instead of 2560 floats).
//
// The gather itself must stay on the host: the n-gram table is a 51 GB file
// mapping the GPU cannot address, so pulling the rows is a memcpy - DMA
// orchestration, which rule 4 permits.
static int64_t g_sh_file[128], g_sh_off[128];
static int g_sh_built = 0;
static void ple_shard_table(const QfStore *st) {
    if (g_sh_built) return;
    for (int ei = 0; ei < st->n_entries; ei++) {
        const char *nm = st->entries[ei].name;
        const char *p = strstr(nm, ".ple_embedding.ngram_embedding.shard_");
        if (!p) continue;
        int N = atoi(p + strlen(".ple_embedding.ngram_embedding.shard_"));
        if (N >= 0 && N < 128) {
            g_sh_file[N] = st->entries[ei].rec.file_idx;
            g_sh_off[N] = st->entries[ei].rec.data_off;
        }
    }
    g_sh_built = 1;
}
// Host address of table row r (NULL if out of range).
static const uint8_t *ple_row_ptr(const QfStore *st, int64_t r) {
    int shard = (int)(r / PLE_ROWS_PER_SHARD);
    int64_t local = r - (int64_t)shard * PLE_ROWS_PER_SHARD;
    if (shard < 0 || shard >= 128) return NULL;
    return (const uint8_t *)st->shard_maps[g_sh_file[shard]] + g_sh_off[shard] + local * PLE_ROW;
}
static int ple_gather_host(const QfStore *st, const int64_t *ids, uint8_t *out) {
    if (g_ple_table_scale == 0.f && ple_load_table_scale(st) != 0) return -1;
    ple_shard_table(st);
    for (int h = 0; h < PLE_HEADS; h++) {
        const uint8_t *src = ple_row_ptr(st, ids[h]);
        if (!src) return -1;
        memcpy(out + h * PLE_ROW, src, PLE_ROW);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// QF_PLE_BF16_DIR : gather from the UPSTREAM BF16 n-gram table.
//
// The shipped table is F8_E4M3 (measured 2.659% per-row relative error against
// the BF16 it was made from, cosine 0.999651). That is the rung BELOW the NVFP4
// question, and it had never been tested: showing NVFP4 is worse than FP8 does
// not show FP8 is free. Pointing this at the upstream shards runs the whole
// precision ladder - BF16 native, FP8 shipped, NVFP4 simulated on top of either
// - through one binary, so the three points are directly comparable.
//
// Rows are laid out identically to ours (128 shards x [2500012, 160]); only the
// element width differs, 2 bytes instead of 1, and there is no per-tensor scale
// upstream (BF16 is unscaled - the 1.9932e-4 came from the FP8 step).
//
// Offsets come from tools/make_bf16_ple_index.py. Per NEVER_AGAIN rule 1 there
// is NO fallback: if the index or a needed shard is missing, this aborts. A
// silent slide back to the FP8 table would make the two arms of the experiment
// identical while claiming to compare them - the vacuous receipt again.
static void ple_fatal(const char *what);

#define PLE_BF16_MAXF 64
static int      g_bf16_on = -1;                 // -1 untried, 0 off, 1 on
static void    *g_bf16_map[PLE_BF16_MAXF];
static size_t   g_bf16_len[PLE_BF16_MAXF];
static int      g_bf16_file[128];
static uint64_t g_bf16_off[128];
static uint8_t  g_bf16_have[128];
static uint32_t g_bf16_rowb = 0;

static int ple_bf16_init(void) {
    const char *dir = getenv("QF_PLE_BF16_DIR");
    if (!dir || !*dir) return (g_bf16_on = 0);

    char path[4096];
    snprintf(path, sizeof path, "%s/ple_bf16.idx", dir);
    FILE *f = fopen(path, "rb");
    if (!f) ple_fatal("QF_PLE_BF16_DIR set but ple_bf16.idx is missing "
                      "(run tools/make_bf16_ple_index.py)");
    uint32_t nf = 0, ns = 0, rowb = 0, rows = 0;
    if (fread(&nf, 4, 1, f) != 1 || fread(&ns, 4, 1, f) != 1 ||
        fread(&rowb, 4, 1, f) != 1 || fread(&rows, 4, 1, f) != 1)
        ple_fatal("ple_bf16.idx truncated header");
    if (nf > PLE_BF16_MAXF || ns != 128 || rowb != PLE_ROW * 2 ||
        rows != PLE_ROWS_PER_SHARD)
        ple_fatal("ple_bf16.idx geometry does not match the engine");
    g_bf16_rowb = rowb;

    for (uint32_t i = 0; i < nf; i++) {
        uint16_t len = 0;
        if (fread(&len, 2, 1, f) != 1 || len == 0 || len > 512)
            ple_fatal("ple_bf16.idx bad path record");
        char name[513];
        if (fread(name, 1, len, f) != len) ple_fatal("ple_bf16.idx truncated path");
        name[len] = 0;
        snprintf(path, sizeof path, "%s/%s", dir, name);
        int fd = open(path, O_RDONLY);
        if (fd < 0) ple_fatal("a BF16 n-gram shard file named by the index is missing");
        off_t sz = lseek(fd, 0, SEEK_END);
        void *m = mmap(NULL, (size_t)sz, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) ple_fatal("mmap of a BF16 n-gram shard failed");
        g_bf16_map[i] = m;
        g_bf16_len[i] = (size_t)sz;
    }
    int present = 0;
    for (uint32_t n = 0; n < ns; n++) {
        uint32_t fi; uint64_t off; uint8_t ok;
        if (fread(&fi, 4, 1, f) != 1 || fread(&off, 8, 1, f) != 1 ||
            fread(&ok, 1, 1, f) != 1)
            ple_fatal("ple_bf16.idx truncated shard table");
        g_bf16_file[n] = (int)fi; g_bf16_off[n] = off; g_bf16_have[n] = ok;
        present += ok;
    }
    fclose(f);
    fprintf(stderr, "ple: GATHERING FROM THE BF16 TABLE (%s), %d/128 shards present\n",
            dir, present);
    if (present < 128)
        fprintf(stderr, "ple: WARNING - %d shards absent; a gather touching one "
                        "will abort rather than substitute FP8\n", 128 - present);
    return (g_bf16_on = 1);
}

static int ple_bf16_active(void) {
    if (g_bf16_on < 0) ple_bf16_init();
    return g_bf16_on;
}

static int ple_gather_host_bf16(const int64_t *ids, uint8_t *out) {
    for (int h = 0; h < PLE_HEADS; h++) {
        int64_t r = ids[h];
        int shard = (int)(r / PLE_ROWS_PER_SHARD);
        int64_t local = r - (int64_t)shard * PLE_ROWS_PER_SHARD;
        if (shard < 0 || shard >= 128) return -1;
        if (!g_bf16_have[shard])
            ple_fatal("a token's n-gram row lives in a BF16 shard that is not "
                      "downloaded; refusing to substitute the FP8 table");
        const uint8_t *base = (const uint8_t *)g_bf16_map[g_bf16_file[shard]]
                            + g_bf16_off[shard];
        memcpy(out + (size_t)h * g_bf16_rowb,
               base + (size_t)local * g_bf16_rowb, g_bf16_rowb);
    }
    return 0;
}

__global__ void k_ple_decode_bf16(float *__restrict__ out,
                                  const uint8_t *__restrict__ raw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PLE_HEADS * PLE_ROW) return;
    uint16_t b = ((const uint16_t *)raw)[i];
    out[i] = __int_as_float((uint32_t)b << 16);   // BF16 -> F32, no scale
}

// Device-side E4M3 decode of the staged n-gram bytes, scaled by the table's
// per-tensor weight_scale. Mirrors host_e4m3 exactly, including the sign bit.
__global__ void k_ple_decode(float *__restrict__ out, const uint8_t *__restrict__ raw,
                             float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PLE_HEADS * PLE_ROW) return;
    uint32_t b = raw[i];
    uint32_t sg = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0) v = ldexpf((float)m, -9);
    else if (e == 15 && m == 7) v = NAN;
    else v = ldexpf((float)(8 + m), (int)e - 10);
    out[i] = (sg ? -v : v) * scale;
}

// ---------------------------------------------------------------------------
// QF_PLE_NVFP4=1 : store-the-table-at-NVFP4, simulated.
//
// The field's proposed win is BF16 -> NVFP4 on this table (-68.5 GiB). We hold
// it at F8_E4M3 already, so this rounds the GATHERED row through the NVFP4 grid
// - E2M1 elements, per-16 UE4M3 group scale, one F32 per-tensor scale - and
// changes nothing else. A run then answers the only question worth asking:
// does the model still SAY the same thing?
//
// The per-tensor scale is FIXED from the table's measured amax (256 raw units
// x 1.9931793e-4 = 0.05102539), never recomputed per token. A stored tensor has
// one global scale; letting it float per token would quietly flatter NVFP4.
#define PLE_NVFP4_S2 1.898266e-05f

__device__ __forceinline__ float ple_e2m1(float a) {
    if (a < 0.25f) return 0.f;
    if (a < 0.75f) return 0.5f;
    if (a < 1.25f) return 1.f;
    if (a < 1.75f) return 1.5f;
    if (a < 2.5f)  return 2.f;
    if (a < 3.5f)  return 3.f;
    if (a < 5.f)   return 4.f;
    return 6.f;
}

__device__ __forceinline__ float ple_e4m3(float a) {
    if (a <= 0.f) return 0.f;
    if (a > 448.f) a = 448.f;
    int e; frexpf(a, &e);              // a = m * 2^e, m in [0.5,1)
    float step = ldexpf(1.f, e - 4);   // 3 mantissa bits -> 8 steps per binade
    float q = rintf(a / step) * step;
    return q > 448.f ? 448.f : q;
}

__global__ void k_ple_nvfp4(float *__restrict__ v) {
    __shared__ float sh[16];           // one block per 16-element scaling group
    int g = blockIdx.x, t = threadIdx.x;
    float x = v[g * 16 + t];
    sh[t] = fabsf(x);
    __syncthreads();
    for (int s = 8; s; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    float eff = ple_e4m3(sh[0] / 6.f / PLE_NVFP4_S2) * PLE_NVFP4_S2;
    v[g * 16 + t] = eff > 0.f ? copysignf(ple_e2m1(fabsf(x) / eff) * eff, x) : 0.f;
}

static int ple_nvfp4_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QF_PLE_NVFP4");
        v = (e && atoi(e)) ? 1 : 0;
        if (v) fprintf(stderr, "ple: SIMULATING NVFP4 STORAGE of the n-gram table "
                               "(s2=%.6g)\n", (double)PLE_NVFP4_S2);
    }
    return v;
}

static void ple_nvfp4_maybe(float *emb, cudaStream_t s) {
    if (ple_nvfp4_on())
        k_ple_nvfp4<<<PLE_HEADS * PLE_ROW / 16, 16, 0, s>>>(emb);
}

__global__ void k_ple_gemv(const __nv_bfloat16 *W, const float *x, float *y, int rows, int in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    float acc = 0.f;
    for (int i = threadIdx.x; i < in; i += blockDim.x)
        acc += __bfloat162float(W[(size_t)row * in + i]) * x[i];
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    __shared__ float ws[8][4];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid][threadIdx.y] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < 4; w++) t += ws[w][threadIdx.y];
        y[row] = t;
    }
}

__device__ __forceinline__ float sigmoidf_p(float v) { return 1.f / (1.f + expf(-v)); }
// rmsnorm group 2560 over 10240, zero-centered weights
__global__ void k_ple_norm(float *out, const float *x, const __nv_bfloat16 *w) {
    const int g = blockIdx.x * NEMBD;
    float acc = 0.f;
    for (int j = threadIdx.x; j < NEMBD; j += blockDim.x) {
        float v = x[g + j];
        acc += v * v;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp_sum[8];
    __shared__ float inv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int wi = 0; wi < blockDim.x / 32; wi++) total += warp_sum[wi];
        inv = rsqrtf(total / NEMBD + 1e-6f);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < NEMBD; j += blockDim.x) {
        const int i = g + j;
        out[i] = x[i] * inv * (1.f + __bfloat162float(w[i]));
    }
}
// gate per stream: 4 threads
__global__ void k_ple_gate4(const float *keyn, const float *queryn, float *gate) {
    int c = threadIdx.x;
    if (c >= HCC) return;
    float g = 0.f;
    for (int j = 0; j < NEMBD; j++) g += keyn[c * NEMBD + j] * queryn[c * NEMBD + j];
    g /= sqrtf((float)NEMBD);
    float ag = fmaxf(fabsf(g), 1e-6f);
    gate[c] = sqrtf(ag) * ((g < 0) ? -1.f : 1.f);
}
// apply: gated[c,d] = sigmoid(gate[c]) * value[d]; then norm_conv
__global__ void k_ple_apply(const float *value, const float *gate, float *gated) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HCC * NEMBD) return;
    gated[i] = sigmoidf_p(gate[i / NEMBD]) * value[i % NEMBD];
}
__global__ void k_ple_normconv(float *out, const float *x, const __nv_bfloat16 *w) {
    const int g = blockIdx.x * NEMBD;
    float acc = 0.f;
    for (int j = threadIdx.x; j < NEMBD; j += blockDim.x) {
        float v = x[g + j];
        acc += v * v;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float warp_sum[8];
    __shared__ float inv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int wi = 0; wi < blockDim.x / 32; wi++) total += warp_sum[wi];
        inv = rsqrtf(total / NEMBD + 1e-6f);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < NEMBD; j += blockDim.x) {
        const int i = g + j;
        out[i] = x[i] * inv * (1.f + __bfloat162float(w[i]));
    }
}
// dilated (3) depthwise conv kernel 4 + silu; ring [9,10240]
__global__ void k_ple_conv_step(float *x, float *ring, const __nv_bfloat16 *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HCC * NEMBD) return;
    float cur = x[i];
    float o = __bfloat162float(w[i * 4 + 0]) * ring[0 * (HCC * NEMBD) + i]
            + __bfloat162float(w[i * 4 + 1]) * ring[3 * (HCC * NEMBD) + i]
            + __bfloat162float(w[i * 4 + 2]) * ring[6 * (HCC * NEMBD) + i]
            + __bfloat162float(w[i * 4 + 3]) * cur;
    for (int s2 = 0; s2 < 8; s2++) ring[s2 * (HCC * NEMBD) + i] = ring[(s2 + 1) * (HCC * NEMBD) + i];
    ring[8 * (HCC * NEMBD) + i] = cur;
    x[i] = o / (1.f + expf(-o));
}
__global__ void k_ple_add(float *R, const float *gated, const float *conv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HCC * NEMBD) R[i] += gated[i] + conv[i];
}

static float *ring_dev = NULL;
static float *emb_dev = NULL, *key_dev = NULL, *val_dev = NULL;
static float *kn_dev = NULL, *qn_dev = NULL, *gate_dev = NULL, *gv_dev = NULL, *nc_dev = NULL;
static uint8_t *ple_stage_host = NULL;

static int ple_ensure(QfModel *m) {
    if (!ple_weights_loaded) {
        char n[256];
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.key_proj.weight");
        ple_key_dev = qf_load_dev_global(&m->store, n);
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.value_proj.weight");
        ple_value_dev = qf_load_dev_global(&m->store, n);
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.norm_key.weight");
        ple_normk_dev = qf_load_dev_global(&m->store, n);
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.norm_query.weight");
        ple_normq_dev = qf_load_dev_global(&m->store, n);
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.norm_conv.weight");
        ple_normc_dev = qf_load_dev_global(&m->store, n);
        snprintf(n, sizeof(n), "model.language_model.layers.1.ple.conv1d.weight");
        ple_conv_dev = qf_load_dev_global(&m->store, n);
        if (!ple_key_dev || !ple_value_dev || !ple_normk_dev || !ple_normq_dev || !ple_normc_dev || !ple_conv_dev) {
            fprintf(stderr, "ple: FATAL - a PLE weight failed to load "
                            "(key=%p value=%p nk=%p nq=%p nc=%p conv=%p)\n",
                    ple_key_dev, ple_value_dev, ple_normk_dev, ple_normq_dev,
                    ple_normc_dev, ple_conv_dev);
            return -1;
        }
        if (!ring_dev) { cudaMalloc(&ring_dev, 9 * HCC * NEMBD * 4); cudaMemset(ring_dev, 0, 9 * HCC * NEMBD * 4); }
    }
    int hcd = HCC * NEMBD;
    if (!emb_dev) {
        cudaMalloc(&emb_dev, NEMBD * 4);
        cudaMalloc(&key_dev, hcd * 4);
        cudaMalloc(&val_dev, NEMBD * 4);
        cudaMalloc(&kn_dev, hcd * 4);
        cudaMalloc(&qn_dev, hcd * 4);
        cudaMalloc(&gate_dev, HCC * 4);
        cudaMalloc(&gv_dev, hcd * 4);
        cudaMalloc(&nc_dev, hcd * 4);
    }
    if (!emb_dev || !key_dev || !val_dev || !kn_dev || !qn_dev || !gate_dev || !gv_dev || !nc_dev) {
        fprintf(stderr, "ple: FATAL - PLE workspace allocation failed\n");
        return -1;
    }
    ple_weights_loaded = 1;
    return 1;   // >0 = ready; <=0 is fatal at the call sites
}

static void ple_launch(float *R_dev, cudaStream_t s) {
    int hcd = HCC * NEMBD;
    k_ple_gemv<<<(hcd + 3) / 4, dim3(128, 4), 0, s>>>((const __nv_bfloat16 *)ple_key_dev, emb_dev, key_dev, hcd, NEMBD);
    k_ple_gemv<<<(NEMBD + 3) / 4, dim3(128, 4), 0, s>>>((const __nv_bfloat16 *)ple_value_dev, emb_dev, val_dev, NEMBD, NEMBD);
    k_ple_norm<<<HCC, 256, 0, s>>>(kn_dev, key_dev, (const __nv_bfloat16 *)ple_normk_dev);
    k_ple_norm<<<HCC, 256, 0, s>>>(qn_dev, R_dev, (const __nv_bfloat16 *)ple_normq_dev);
    QF_TAP("ple_emb", 1, emb_dev, NEMBD);
    QF_TAP("ple_key", 1, key_dev, hcd);
    QF_TAP("ple_val", 1, val_dev, NEMBD);
    QF_TAP("ple_kn", 1, kn_dev, hcd);
    QF_TAP("ple_qn", 1, qn_dev, hcd);
    k_ple_gate4<<<1, HCC, 0, s>>>(kn_dev, qn_dev, gate_dev);
    k_ple_apply<<<(hcd + 1023) / 1024, 1024, 0, s>>>(val_dev, gate_dev, gv_dev);
    QF_TAP("ple_gv", 1, gv_dev, hcd);
    k_ple_normconv<<<HCC, 256, 0, s>>>(nc_dev, gv_dev, (const __nv_bfloat16 *)ple_normc_dev);
    k_ple_conv_step<<<(hcd + 1023) / 1024, 1024, 0, s>>>(nc_dev, ring_dev, (const __nv_bfloat16 *)ple_conv_dev);
    QF_TAP("ple_conv", 1, nc_dev, hcd);
    k_ple_add<<<(hcd + 1023) / 1024, 1024, 0, s>>>(R_dev, gv_dev, nc_dev);
}

// One staging slot per position in a batched forward. The buffer is
// cudaHostAllocMapped, so k_ple_decode reads it ZERO-COPY at kernel EXECUTION
// time - not at enqueue. Staging T tokens into one slot in a host loop lets the
// host overwrite it while earlier decodes are still pending, and every position
// silently gets the last token's n-gram gather.
#define PLE_SLOTS 16
// Stride is sized for the WIDEST element the gather can stage (BF16, 2 bytes).
// The FP8 path uses the first half of each slot; keeping one stride means the
// two arms of a precision comparison share identical slot addressing.
#define PLE_STAGE_STRIDE ((size_t)PLE_HEADS * PLE_ROW * 2)
static uint8_t *qf_ple_stage_ptr_slot(int slot) {
    if (!ple_stage_host)
        cudaHostAlloc((void **)&ple_stage_host, PLE_STAGE_STRIDE * PLE_SLOTS,
                      cudaHostAllocMapped);
    if (!ple_stage_host) return NULL;
    if (slot < 0 || slot >= PLE_SLOTS) return NULL;
    return ple_stage_host + (size_t)slot * PLE_STAGE_STRIDE;
}
static uint8_t *qf_ple_stage_ptr(void) { return qf_ple_stage_ptr_slot(0); }

int qf_ple_stage_slot(QfModel *m, const int64_t *ids, int slot);
void qf_ple_apply_staged_slot(QfModel *m, float *R_dev, int slot, cudaStream_t s);

int qf_ple_stage(QfModel *m, const int64_t *ids) { return qf_ple_stage_slot(m, ids, 0); }

// ---- QF_PLE_WIRE: rows delivered by the control plane ----------------------
// The head node owns the token stream, computes the same 16 hashes, gathers
// from ITS copy of the table, and ships the raw F8 rows on the step message.
// This node then does ZERO table work: no mmap fault, no host gather. There
// is NO fallback: wire mode with no fresh rows for the step is fatal
// (NEVER_AGAIN rule 1 - a silent slide back to the local table would make
// the two configurations indistinguishable).
static int g_ple_wire = -1;
static uint8_t g_ple_wire_rows[PLE_HEADS * PLE_ROW];
static int g_ple_wire_fresh = 0;
static int ple_wire_on(void) {
    if (g_ple_wire < 0) {
        g_ple_wire = getenv("QF_PLE_WIRE") ? 1 : 0;
        if (g_ple_wire)
            fprintf(stderr, "ple: ROWS FROM WIRE - the local table will not be read\n");
    }
    return g_ple_wire;
}
int qf_ple_stage_wire_bytes(const void *bytes, int nbytes) {
    if (!ple_wire_on()) return -1;
    if (nbytes != (int)(PLE_HEADS * PLE_ROW)) return -1;
    memcpy(g_ple_wire_rows, bytes, (size_t)nbytes);
    g_ple_wire_fresh = 1;
    return 0;
}

// Prefetch for the batched decode loop: every row's 16 n-gram rows are
// demand-faulted from the 51 GB host-mapped table inside the host staging
// loop, i.e. with the GPU idle between graph launches, and the full-resident
// load dropped the page cache, so a new row is a synchronous NVMe read.
// MADV_WILLNEED starts all of a step's reads at once (16 rows x 16 heads);
// the memcpys that follow then overlap instead of serializing. No-op in wire
// mode and on the BF16 table path (neither reads the mmap'd FP8 shards).
int qf_ple_prefetch_ids(QfModel *m, const int64_t *ids) {
    if (ple_wire_on() || ple_bf16_active()) return 0;
    const QfStore *st = &m->store;
    ple_shard_table(st);
    static long pg = 0;
    if (!pg) pg = sysconf(_SC_PAGESIZE);
    for (int h = 0; h < PLE_HEADS; h++) {
        const uint8_t *src = ple_row_ptr(st, ids[h]);
        if (!src) return -1;
        const uintptr_t a0 = (uintptr_t)src & ~(uintptr_t)(pg - 1);
        const uintptr_t a1 = ((uintptr_t)src + PLE_ROW + (uintptr_t)pg - 1) & ~(uintptr_t)(pg - 1);
        madvise((void *)a0, (size_t)(a1 - a0), MADV_WILLNEED);
    }
    return 0;
}

int qf_ple_stage_slot(QfModel *m, const int64_t *ids, int slot) {
    uint8_t *stage = qf_ple_stage_ptr_slot(slot);
    if (!stage) return -1;
    if (ple_wire_on()) {
        // Scale still comes from this node's checkpoint (identical file).
        if (g_ple_table_scale == 0.f && ple_load_table_scale(&m->store) != 0)
            ple_fatal("wire mode: n-gram weight_scale unavailable");
        if (!g_ple_wire_fresh)
            ple_fatal("QF_PLE_WIRE set but no wire rows staged for this step");
        memcpy(stage, g_ple_wire_rows, PLE_HEADS * PLE_ROW);
        g_ple_wire_fresh = 0;
        return 0;
    }
    if (ple_bf16_active()) return ple_gather_host_bf16(ids, stage);
    return ple_gather_host(&m->store, ids, stage);
}

__global__ void k_ple_stage_in(float *dst, const float *src) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NEMBD) dst[i] = src[i];
}

// NEVER_AGAIN rule 1: there is no slow/degraded path behind the fast one. If
// the PLE block cannot run, the engine must fail loudly - not quietly emit a
// forward pass with no PLE injection, which is a silent quality cliff that
// looks exactly like a working model.
static void ple_fatal(const char *what) {
    fprintf(stderr, "ple: FATAL - %s; refusing to run a forward pass without "
                    "PLE injection\n", what);
    abort();
}

void qf_ple_apply_staged(QfModel *m, float *R_dev, cudaStream_t s) {
    qf_ple_apply_staged_slot(m, R_dev, 0, s);
}

void qf_ple_apply_staged_slot(QfModel *m, float *R_dev, int slot, cudaStream_t s) {
    if (ple_ensure(m) <= 0) ple_fatal("PLE block unavailable");
    uint8_t *stage = qf_ple_stage_ptr_slot(slot);
    if (!stage) ple_fatal("PLE staging buffer unavailable");
    if (ple_bf16_active())
        k_ple_decode_bf16<<<(PLE_HEADS * PLE_ROW + 255) / 256, 256, 0, s>>>(
            emb_dev, stage);
    else
        k_ple_decode<<<(PLE_HEADS * PLE_ROW + 255) / 256, 256, 0, s>>>(
            emb_dev, stage, g_ple_table_scale);
    ple_nvfp4_maybe(emb_dev, s);
    ple_launch(R_dev, s);
}

// The PLE short-conv ring is per-sequence recurrent state, so a speculative
// round must be able to snapshot and restore it alongside the GDN state.
size_t qf_ple_ring_bytes(void) { return (size_t)9 * HCC * NEMBD * sizeof(float); }
float *qf_ple_ring_ptr(void) { return ring_dev; }

void qf_ple_reset(cudaStream_t s) {
    if (ring_dev)
        cudaMemsetAsync(ring_dev, 0, 9 * HCC * NEMBD * sizeof(float), s);
}

void qf_ple_apply(QfModel *m, const int64_t *ids, float *R_dev, cudaStream_t s) {
    if (ple_ensure(m) <= 0) ple_fatal("PLE block unavailable");
    uint8_t *stage = qf_ple_stage_ptr();
    if (!stage) ple_fatal("PLE staging buffer unavailable");
    if ((ple_bf16_active() ? ple_gather_host_bf16(ids, stage)
                           : ple_gather_host(&m->store, ids, stage)) != 0)
        ple_fatal("PLE gather failed");
    if (ple_bf16_active())
        k_ple_decode_bf16<<<(PLE_HEADS * PLE_ROW + 255) / 256, 256, 0, s>>>(
            emb_dev, stage);
    else
        k_ple_decode<<<(PLE_HEADS * PLE_ROW + 255) / 256, 256, 0, s>>>(
            emb_dev, stage, g_ple_table_scale);
    ple_nvfp4_maybe(emb_dev, s);
    ple_launch(R_dev, s);
}
