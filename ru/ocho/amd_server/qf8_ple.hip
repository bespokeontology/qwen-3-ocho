// qf8_ple.hip - PLE (layer-1 n-gram injection) for the gfx906 pipeline, GPU math.
//
// Split of work (mirrors the expert-cache contract: host gathers bytes out of
// the mmap'd store, GPU does ALL arithmetic):
//   host (qf8_ple_step, called once per token before the layer loop):
//     - history ring push (last 3 token ids, eos-padded)
//     - 16 integer hashes (splitmix multipliers read from the checkpoint I64
//       tensors; 2-gram for heads 0..7, 3-gram for heads 8..15 -- identical
//       to the CUDA path in source/src/cuda/qf.cu)
//     - gather 16 rows x 160 B of fp8 table out of the sharded mmap
//       (PLE_ROWS_PER_SHARD = 2500012) into a pinned double buffer
//     - async H2D on a dedicated transfer stream + event (2.5 KiB, overlaps
//       in-flight compute; the compute stream only waits on an event)
//   device (qf8_ple_apply, stream-ordered, no syncs):
//     fp8 dequant -> key/value GEMVs -> grouped norms -> per-stream gate ->
//     gated value + norm_conv -> dilated (3) depthwise conv (kernel 4) + silu
//     -> R += gated + conv, in 6 launches.
//
// The fp8 table is far too large for 16 GiB HBM (128 shards x 2500012 rows x
// 160 B ~= 51 GB); it stays in the mmap'd page cache with drop-behind, same
// as the expert store.
#include "qf8_dense.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "qf8_layer_m.h"   // QF8_MROWS: per-row PLE state

#define PLE_SHARD_ROWS 2500012
#define PLE_MAX_SHARD  128
#define PLE_EOS        248044

// ---- device kernels ---------------------------------------------------------
// fp8 E4M3 -> f32, scaled by the n-gram table's per-tensor weight_scale.
// Exact integer construction, matching host_e4m3 in the CUDA path (subnormal:
// m * 2^-9; NaN at e15m7). The sign bit is masked out of the exponent and
// applied at the end -- E4M3 has a sign bit, and roughly half the table's
// bytes set it (measured 50.19% on the real checkpoint).
//
// THE SCALE IS NOT OPTIONAL. ...ngram_embedding.weight_scale is a single BF16
// scalar = 1.9932e-4. The raw bytes span [-240, 208]; scaled they span
// [-0.0478, 0.0415], which is an embedding-sized quantity. Gathering the table
// unscaled makes the PLE injection ~5000x too large and it swamps the
// hyper-connection residual, so all 48 layers compute on a state that is
// almost entirely PLE -- word salad with finite logits. hf_quant_config.json
// EXCLUDES *.ple.*, so no generic loader applies it for you.
__global__ void k8_ple_dequant(float *__restrict__ out, const uint8_t *__restrict__ in,
                               float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t b = in[i];
    uint32_t e = (b >> 3) & 15, mm = b & 7, s = b >> 7;
    float v;
    if (e == 0) v = (float)mm * 0x1p-9f;                          // subnormal: m * 2^-9
    else if (e == 15 && mm == 7) v = __uint_as_float(0x7fc00000u);  // NaN
    else v = (float)(8 + mm) * __uint_as_float((uint32_t)(e - 10 + 127) << 23);
    // normal: (1+m/8) * 2^(e-7) == (8+m) * 2^(e-10); (8+m) is exact in f32
    out[i] = (s ? -v : v) * scale;
}

// zero-centered grouped RMSNorm (group 2560) for key (blocks 0..3) and query
// (blocks 4..7, source is R) in one launch.
__global__ void k8_ple_norm2(float *__restrict__ kn, float *__restrict__ qn,
                             const float *__restrict__ key,
                             const float *__restrict__ R,
                             const uint16_t *__restrict__ wk,
                             const uint16_t *__restrict__ wq) {
    { const size_t tr = (size_t)blockIdx.y * QF8_RSIZE; kn += tr; qn += tr; key += tr; R += tr; }   // chunk: position = blockIdx.y
    int isq = blockIdx.x >= QF8_HCC;
    int g0 = (blockIdx.x % QF8_HCC) * QF8_NEMBD;
    const float *src = (isq ? R : key) + g0;
    const uint16_t *w = (isq ? wq : wk) + g0;
    float *dst = (isq ? qn : kn) + g0;
    float acc = 0.f;
    for (int i = threadIdx.x; i < QF8_NEMBD; i += blockDim.x) {
        float v = src[i];
        acc += v * v;
    }
    acc = qf8_wave_sum(acc);
    __shared__ float red[4];
    __shared__ float rstd;
    int lane = threadIdx.x % QF8_WAVE, wid = threadIdx.x / QF8_WAVE;
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        rstd = rsqrtf((red[0] + red[1] + red[2] + red[3]) / QF8_NEMBD + 1e-6f);
    __syncthreads();
    float r = rstd;
    for (int i = threadIdx.x; i < QF8_NEMBD; i += blockDim.x)
        dst[i] = src[i] * r * (1.f + qf8_bf2f(w[i]));
}

// gate per stream: g = |dot(kn_c, qn_c)/sqrt(2560)| clamped >= 1e-6, sqrt,
// sign-restored. One 256-thread block, one wave per stream.
__global__ void k8_ple_gate(const float *__restrict__ kn, const float *__restrict__ qn,
                            float *__restrict__ gate) {
    { const size_t tr = (size_t)blockIdx.x; kn += tr * QF8_RSIZE; qn += tr * QF8_RSIZE; gate += tr * QF8_HCC; }   // chunk: position = blockIdx.x
    int c = threadIdx.x / QF8_WAVE, lane = threadIdx.x % QF8_WAVE;
    const float *kp = kn + c * QF8_NEMBD, *qp = qn + c * QF8_NEMBD;
    float acc = 0.f;
    for (int i = lane; i < QF8_NEMBD; i += QF8_WAVE) acc += kp[i] * qp[i];
    acc = qf8_wave_sum(acc);
    if (lane == 0) {
        float g = acc / sqrtf((float)QF8_NEMBD);
        float ag = fmaxf(fabsf(g), 1e-6f);
        gate[c] = sqrtf(ag) * ((g < 0) ? -1.f : 1.f);
    }
}

// gated value + norm_conv, fused: one 256-thread block per stream.
// gv[c,i] = sigmoid(gate[c]) * value[i]; nc = rmsnorm_group(gv) (1+w).
__global__ void k8_ple_apply_nc(float *__restrict__ gv, float *__restrict__ nc,
                                const float *__restrict__ value,
                                const float *__restrict__ gate,
                                const uint16_t *__restrict__ wc) {
    { const size_t tr = (size_t)blockIdx.y; gv += tr * QF8_RSIZE; nc += tr * QF8_RSIZE; value += tr * QF8_NEMBD; gate += tr * QF8_HCC; }   // chunk: position = blockIdx.y
    int c = blockIdx.x, g0 = c * QF8_NEMBD;
    float sg = qf8_sig(gate[c]);
    float v[QF8_NEMBD / 256];                   // 10 per thread
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < QF8_NEMBD / 256; j++) {
        int i = threadIdx.x + j * 256;
        v[j] = sg * value[i];
        acc += v[j] * v[j];
    }
    acc = qf8_wave_sum(acc);
    __shared__ float red[4];
    __shared__ float rstd;
    int lane = threadIdx.x % QF8_WAVE, wid = threadIdx.x / QF8_WAVE;
    if (lane == 0) red[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        rstd = rsqrtf((red[0] + red[1] + red[2] + red[3]) / QF8_NEMBD + 1e-6f);
    __syncthreads();
    float r = rstd;
    #pragma unroll
    for (int j = 0; j < QF8_NEMBD / 256; j++) {
        int i = threadIdx.x + j * 256;
        gv[g0 + i] = v[j];
        nc[g0 + i] = v[j] * r * (1.f + qf8_bf2f(wc[g0 + i]));
    }
}

// dilated (3) depthwise conv, kernel 4, over 10240 + silu, ring [9,10240],
// fused residual add: R += gv + silu(convout). Taps: ring 0, 3, 6, current.
__global__ void k8_ple_conv_add(float *__restrict__ R,
                                float *__restrict__ nc,
                                const float *__restrict__ gv,
                                float *__restrict__ ring,
                                const uint16_t *__restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= QF8_RSIZE) return;
    float cur = nc[i];
    float o = qf8_bf2f(w[i * 4 + 0]) * ring[0 * QF8_RSIZE + i]
            + qf8_bf2f(w[i * 4 + 1]) * ring[3 * QF8_RSIZE + i]
            + qf8_bf2f(w[i * 4 + 2]) * ring[6 * QF8_RSIZE + i]
            + qf8_bf2f(w[i * 4 + 3]) * cur;
    #pragma unroll
    for (int t = 0; t < 8; t++) ring[t * QF8_RSIZE + i] = ring[(t + 1) * QF8_RSIZE + i];
    ring[8 * QF8_RSIZE + i] = cur;
    R[i] += gv[i] + qf8_silu(o);
}

// Chunk form of the conv ring: ONE launch walks the T positions in order with
// the 9-deep ring of this element in registers (written back once).
__global__ void k8_ple_conv_add_chunk(float *__restrict__ R, const float *__restrict__ nc,
                                      const float *__restrict__ gv, float *__restrict__ ring,
                                      const uint16_t *__restrict__ w, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= QF8_RSIZE) return;
    float rg[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) rg[k] = ring[k * QF8_RSIZE + i];
    const float w0 = qf8_bf2f(w[i * 4 + 0]), w1 = qf8_bf2f(w[i * 4 + 1]), w2 = qf8_bf2f(w[i * 4 + 2]), w3 = qf8_bf2f(w[i * 4 + 3]);
    for (int t = 0; t < T; t++) {
        const size_t o_ = (size_t)t * QF8_RSIZE + i;
        const float cur = nc[o_];
        const float o = w0 * rg[0] + w1 * rg[3] + w2 * rg[6] + w3 * cur;
        #pragma unroll
        for (int k = 0; k < 8; k++) rg[k] = rg[k + 1];
        rg[8] = cur;
        R[o_] += gv[o_] + qf8_silu(o);
    }
    #pragma unroll
    for (int k = 0; k < 9; k++) ring[k * QF8_RSIZE + i] = rg[k];
}

// ---- host context -----------------------------------------------------------
struct Qf8Ple {
    // weights (device)
    uint16_t *key_proj, *value_proj, *norm_key, *norm_query, *norm_conv, *conv1d;
    // state / scratch (device)
    float *ring;                    // [QF8_MROWS][9, 10240]  (per-row conv ring)
    float *emb, *key, *val, *kn, *qn, *gate, *gv, *nc;   // per-row scratch (row-strided)
    uint8_t *emb_fp8[QF8_MROWS][2]; // device double buffer per row, 2560 B each
    // host
    uint8_t *emb_pin[QF8_MROWS][2]; // pinned staging per row
    float table_scale;              // ngram_embedding.weight_scale (FATAL if absent)
    int cur[QF8_MROWS];             // per-row double-buffer phase
    int64_t hist[QF8_MROWS][3];     // per-row n-gram history, newest first
    uint64_t mult[3];
    int64_t vsizes[QF8_PLE_H], offsets[QF8_PLE_H];
    int64_t shard_file[PLE_MAX_SHARD];
    uint64_t shard_off[PLE_MAX_SHARD];
    int n_shards;
    hipStream_t xfer;
    hipEvent_t ev[QF8_MROWS][2];
    // chunk form (row 0, T positions): one pinned staging, one H2D, T-row scratch
    // chunk form: QF8_PLE_SLOTS pinned stagings + device fp8 buffers + events (one per pipeline
    // slot) so the host gather + H2D of chunk c+1 runs at token intake while the GPU computes
    // chunk c; the fp32 scratch below is used inside apply only (stream order).
    int ccap; uint8_t *cemb_pin[QF8_PLE_SLOTS], *cemb_fp8[QF8_PLE_SLOTS]; hipEvent_t cev[QF8_PLE_SLOTS];
    float *cemb, *ckey, *cval, *ckn, *cqn, *cgate, *cgv, *cnc;
};

// Per-tensor scale for the n-gram table: ONE BF16 scalar beside the F8_E4M3
// rows. This is the LongCat weight_scale_inv trap; a missing scale is FATAL
// here, never fall back to unscaled. Returns 0 on success, -1 if absent.
static int ple_load_table_scale(Qf8Ple *p, const QfStore *st, int layer) {
    char nm[QF_MAX_NAME];
    const char *pats[2] = {
        "model.language_model.layers.%d.ple.ple_embedding.ngram_embedding.weight_scale",
        "model.layers.%d.ple.ple_embedding.ngram_embedding.weight_scale",
    };
    for (int i = 0; i < 2; i++) {
        snprintf(nm, sizeof(nm), pats[i], layer);
        const QfEntry *e = qf_find(st, nm);
        if (!e || e->rec.nbytes != 2) continue;
        uint16_t bits;
        memcpy(&bits, (const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off, 2);
        uint32_t f = (uint32_t)bits << 16;
        float v;
        memcpy(&v, &f, 4);
        if (!(v > 0.f)) break;
        p->table_scale = v;
        fprintf(stderr, "qf8 ple: n-gram table scale %.8g (from %s)\n", v, nm);
        return 0;
    }
    fprintf(stderr, "qf8 ple: FATAL - n-gram table weight_scale missing or invalid; "
                    "refusing to gather an unscaled table\n");
    return -1;
}

static uint64_t qf8_splitmix64(uint64_t v) {
    v += 0x9E3779B97F4A7C15ULL;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
    return v ^ (v >> 31);
}

// bf16 (or f32, converted) tensor load, same acceptance rules as the stage's
// load_bf16.
static uint16_t *ple_load_bf16(const QfStore *st, const char *name, size_t elems) {
    const QfEntry *e = qf_find(st, name);
    if (!e) return NULL;
    const char *src = (const char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off;
    uint16_t *dev = NULL;
    if (e->rec.nbytes == elems * 2) {
        if (hipMalloc((void **)&dev, e->rec.nbytes) != hipSuccess ||
            hipMemcpy(dev, src, e->rec.nbytes, hipMemcpyHostToDevice) != hipSuccess)
            return NULL;
        return dev;
    }
    if (e->rec.nbytes == elems * 4) {
        uint16_t *hbuf = (uint16_t *)malloc(elems * 2);
        if (!hbuf) return NULL;
        const uint32_t *f = (const uint32_t *)src;
        for (size_t i = 0; i < elems; i++) {
            uint32_t u = f[i];
            u += 0x7FFFu + ((u >> 16) & 1u);
            hbuf[i] = (uint16_t)(u >> 16);
        }
        int ok = hipMalloc((void **)&dev, elems * 2) == hipSuccess &&
                 hipMemcpy(dev, hbuf, elems * 2, hipMemcpyHostToDevice) == hipSuccess;
        free(hbuf);
        return ok ? dev : NULL;
    }
    return NULL;
}

int qf8_ple_init(Qf8Ple **out, const QfStore *st, int layer) {
    char nm[QF_MAX_NAME];
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.key_proj.weight", layer);
    if (!qf_find(st, nm)) { *out = NULL; return 1; }   // no PLE in this model
    Qf8Ple *p = (Qf8Ple *)calloc(1, sizeof(Qf8Ple));
    if (!p) return -1;
    for (int r = 0; r < QF8_MROWS; r++) { p->hist[r][0] = p->hist[r][1] = p->hist[r][2] = PLE_EOS; p->cur[r] = 0; }

    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.key_proj.weight", layer);
    p->key_proj = ple_load_bf16(st, nm, (size_t)QF8_RSIZE * QF8_NEMBD);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.value_proj.weight", layer);
    p->value_proj = ple_load_bf16(st, nm, (size_t)QF8_NEMBD * QF8_NEMBD);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.norm_key.weight", layer);
    p->norm_key = ple_load_bf16(st, nm, QF8_RSIZE);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.norm_query.weight", layer);
    p->norm_query = ple_load_bf16(st, nm, QF8_RSIZE);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.norm_conv.weight", layer);
    p->norm_conv = ple_load_bf16(st, nm, QF8_RSIZE);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.conv1d.weight", layer);
    p->conv1d = ple_load_bf16(st, nm, (size_t)QF8_RSIZE * 4);
    if (!p->key_proj || !p->value_proj || !p->norm_key || !p->norm_query ||
        !p->norm_conv || !p->conv1d) {
        fprintf(stderr, "qf8 ple: weight load failed (layer %d)\n", layer);
        qf8_ple_free(p);
        return -1;
    }
    if (ple_load_table_scale(p, st, layer) != 0) { qf8_ple_free(p); return -1; }

    // hash constants: checkpoint I64 tensors first, splitmix/prime fallback
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.ple_embedding.layer_multipliers", layer);
    const QfEntry *em = qf_find(st, nm);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.ple_embedding.ngram_heads_vocab_sizes", layer);
    const QfEntry *ev = qf_find(st, nm);
    snprintf(nm, sizeof(nm), "model.language_model.layers.%d.ple.ple_embedding.ngram_heads_offsets", layer);
    const QfEntry *eo = qf_find(st, nm);
    if (em && ev && eo) {
        memcpy(p->mult, (const char *)st->shard_maps[em->rec.file_idx] + em->rec.data_off, 3 * 8);
        memcpy(p->vsizes, (const char *)st->shard_maps[ev->rec.file_idx] + ev->rec.data_off, QF8_PLE_H * 8);
        memcpy(p->offsets, (const char *)st->shard_maps[eo->rec.file_idx] + eo->rec.data_off, QF8_PLE_H * 8);
    } else {
        // fallback: the derivation in source/src/cuda/qf.cu ple_init
        for (int i = 0; i < 3; i++) {
            uint64_t v = 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
            uint64_t half = ((1ULL << 63) - 1) / 248320;
            p->mult[i] = 2 * (qf8_splitmix64(v) % half) + 1;
        }
        int64_t total = 0;
        for (int hh = 0; hh < QF8_PLE_H; hh++) {
            int64_t pr = 19999999, cnt = hh + 1;
            while (cnt > 0) {
                pr++;
                int isp = pr % 2 != 0;
                for (int64_t d = 3; d * d <= pr; d += 2)
                    if (pr % d == 0) { isp = 0; break; }
                if (isp) cnt--;
            }
            p->vsizes[hh] = pr;
            p->offsets[hh] = total;
            total += pr;
        }
    }

    // fp8 table shard map (host gathers rows straight out of the mmap)
    p->n_shards = 0;
    for (int ei = 0; ei < st->n_entries; ei++) {
        const char *tn = st->entries[ei].name;
        const char *pp = strstr(tn, ".ple_embedding.ngram_embedding.shard_");
        if (!pp) continue;
        int N = atoi(pp + strlen(".ple_embedding.ngram_embedding.shard_"));
        if (N >= 0 && N < PLE_MAX_SHARD) {
            p->shard_file[N] = st->entries[ei].rec.file_idx;
            p->shard_off[N] = st->entries[ei].rec.data_off;
            if (N >= p->n_shards) p->n_shards = N + 1;
        }
    }
    if (!p->n_shards) { qf8_ple_free(p); return -1; }

#define PLE_AL(f, bytes) do { \
    if (hipMalloc((void **)&p->f, bytes) != hipSuccess) { qf8_ple_free(p); return -1; } } while (0)
    PLE_AL(ring, (size_t)QF8_MROWS * 9 * QF8_RSIZE * 4);
    PLE_AL(emb, (size_t)QF8_MROWS * QF8_NEMBD * 4);
    PLE_AL(key, (size_t)QF8_MROWS * QF8_RSIZE * 4);
    PLE_AL(val, (size_t)QF8_MROWS * QF8_NEMBD * 4);
    PLE_AL(kn, (size_t)QF8_MROWS * QF8_RSIZE * 4);
    PLE_AL(qn, (size_t)QF8_MROWS * QF8_RSIZE * 4);
    PLE_AL(gate, (size_t)QF8_MROWS * QF8_HCC * 4);
    PLE_AL(gv, (size_t)QF8_MROWS * QF8_RSIZE * 4);
    PLE_AL(nc, (size_t)QF8_MROWS * QF8_RSIZE * 4);
    for (int r = 0; r < QF8_MROWS; r++) {
        PLE_AL(emb_fp8[r][0], QF8_NEMBD);
        PLE_AL(emb_fp8[r][1], QF8_NEMBD);
    }
    if (hipMemset(p->ring, 0, (size_t)QF8_MROWS * 9 * QF8_RSIZE * 4) != hipSuccess) { qf8_ple_free(p); return -1; }
    int pin_ok = 1;
    for (int r = 0; r < QF8_MROWS; r++)
        if (hipHostMalloc((void **)&p->emb_pin[r][0], QF8_NEMBD, 0) != hipSuccess ||
            hipHostMalloc((void **)&p->emb_pin[r][1], QF8_NEMBD, 0) != hipSuccess ||
            hipEventCreateWithFlags(&p->ev[r][0], hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&p->ev[r][1], hipEventDisableTiming) != hipSuccess) pin_ok = 0;
    if (!pin_ok || hipStreamCreate(&p->xfer) != hipSuccess) {
        qf8_ple_free(p);
        return -1;
    }
    *out = p;
    return 0;
}

void qf8_ple_free(Qf8Ple *p) {
    if (!p) return;
    if (p->ring) hipFree(p->ring);
    if (p->emb) hipFree(p->emb);
    if (p->key) hipFree(p->key);
    if (p->val) hipFree(p->val);
    if (p->kn) hipFree(p->kn);
    if (p->qn) hipFree(p->qn);
    if (p->gate) hipFree(p->gate);
    if (p->gv) hipFree(p->gv);
    if (p->nc) hipFree(p->nc);
    for (int r = 0; r < QF8_MROWS; r++) {
        if (p->emb_fp8[r][0]) hipFree(p->emb_fp8[r][0]);
        if (p->emb_fp8[r][1]) hipFree(p->emb_fp8[r][1]);
        if (p->emb_pin[r][0]) hipHostFree(p->emb_pin[r][0]);
        if (p->emb_pin[r][1]) hipHostFree(p->emb_pin[r][1]);
        if (p->ev[r][0]) hipEventDestroy(p->ev[r][0]);
        if (p->ev[r][1]) hipEventDestroy(p->ev[r][1]);
    }
    if (p->xfer) hipStreamDestroy(p->xfer);
    free(p);
}

// The 51 GB n-gram table is mmap'd; a row not in the page cache costs one
// synchronous NVMe read (~100 us). Advising all of a token's 16 rows first puts
// the 16 reads in flight together, so the memcpy loop pays ~one latency
// instead of sixteen (measured 2.2 ms/token cold vs 0.3 ms warm).
static inline void ple_advise_row(const uint8_t *addr) {
    const uintptr_t a = (uintptr_t)addr & ~(uintptr_t)4095;
    const uintptr_t e = ((uintptr_t)addr + QF8_PLE_ROW + 4095) & ~(uintptr_t)4095;
    madvise((void *)a, e - a, MADV_WILLNEED);
}
int qf8_ple_step_row(Qf8Ple *p, const QfStore *st, int token, int row) {
    if (!p) return 0;
    if (row < 0 || row >= QF8_MROWS) return -1;
    // per-row history ring (newest first): rows are independent requests
    p->hist[row][2] = p->hist[row][1];
    p->hist[row][1] = p->hist[row][0];
    p->hist[row][0] = token;
    uint64_t t0 = (uint64_t)p->hist[row][0], t1 = (uint64_t)p->hist[row][1], t2 = (uint64_t)p->hist[row][2];
    uint64_t m2 = (t0 * p->mult[0]) ^ (t1 * p->mult[1]);
    uint64_t m3 = m2 ^ (t2 * p->mult[2]);
    int cur = p->cur[row] ^ 1;
    p->cur[row] = cur;
    uint8_t *dst = p->emb_pin[row][cur];
    const uint8_t *src[QF8_PLE_H];
    for (int hh = 0; hh < QF8_PLE_H; hh++) {
        uint64_t mixed = (hh < 8) ? m2 : m3;
        int64_t r = (int64_t)(mixed % (uint64_t)p->vsizes[hh]) + p->offsets[hh];
        int shard = (int)(r / PLE_SHARD_ROWS);
        int64_t local = r - (int64_t)shard * PLE_SHARD_ROWS;
        if (shard < 0 || shard >= p->n_shards) return -1;
        src[hh] = (const uint8_t *)st->shard_maps[p->shard_file[shard]] + p->shard_off[shard] + local * QF8_PLE_ROW;
        ple_advise_row(src[hh]);
    }
    for (int hh = 0; hh < QF8_PLE_H; hh++) memcpy(dst + hh * QF8_PLE_ROW, src[hh], QF8_PLE_ROW);
    // async H2D on the transfer stream; the compute stream waits on the event
    if (hipMemcpyAsync(p->emb_fp8[row][cur], dst, QF8_NEMBD, hipMemcpyHostToDevice,
                       p->xfer) != hipSuccess ||
        hipEventRecord(p->ev[row][cur], p->xfer) != hipSuccess) return -1;
    return 0;
}
// Back-compat single-row entry (row 0).
int qf8_ple_step(Qf8Ple *p, const QfStore *st, int token) { return qf8_ple_step_row(p, st, token, 0); }

// ---- CHUNK form: T consecutive positions of ROW 0 ----------------------------
static int ple_chunk_alloc(Qf8Ple *p, int T) {
    if (T <= p->ccap) return 0;
    const int cap = T < 256 ? 256 : T;
    if (p->ccap) {
        for (int k = 0; k < QF8_PLE_SLOTS; k++) { hipEventSynchronize(p->cev[k]); hipHostFree(p->cemb_pin[k]); hipFree(p->cemb_fp8[k]); }
        hipFree(p->cemb); hipFree(p->ckey); hipFree(p->cval);
        hipFree(p->ckn); hipFree(p->cqn); hipFree(p->cgate); hipFree(p->cgv); hipFree(p->cnc);
    } else for (int k = 0; k < QF8_PLE_SLOTS; k++) if (hipEventCreateWithFlags(&p->cev[k], hipEventDisableTiming) != hipSuccess) return -1;
    p->ccap = 0;
    for (int k = 0; k < QF8_PLE_SLOTS; k++) {
        if (hipHostMalloc((void **)&p->cemb_pin[k], (size_t)cap * QF8_NEMBD, 0) != hipSuccess) return -1;
        if (hipMalloc((void **)&p->cemb_fp8[k], (size_t)cap * QF8_NEMBD) != hipSuccess) return -1;
    }
    if (hipMalloc((void **)&p->cemb, (size_t)cap * QF8_NEMBD * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->ckey, (size_t)cap * QF8_RSIZE * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->cval, (size_t)cap * QF8_NEMBD * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->ckn, (size_t)cap * QF8_RSIZE * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->cqn, (size_t)cap * QF8_RSIZE * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->cgate, (size_t)cap * QF8_HCC * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->cgv, (size_t)cap * QF8_RSIZE * 4) != hipSuccess) return -1;
    if (hipMalloc((void **)&p->cnc, (size_t)cap * QF8_RSIZE * 4) != hipSuccess) return -1;
    p->ccap = cap;
    return 0;
}
// Host: row 0's n-gram history advances over the T tokens IN ORDER (same update
// as qf8_ple_step_row), the 16 head rows of every position are gathered into
// one pinned staging, one async H2D. The staging is reused only by the next
// chunk, after this chunk's apply has waited on the copy and the stage synced.
int qf8_ple_prefetch_chunk(Qf8Ple *p, const QfStore *st, const int *tokens, int T, int slot) {
    if (!p) return 0;
    if (T < 1 || slot < 0 || slot >= QF8_PLE_SLOTS) return -1;
    const int had = p->ccap;
    if (ple_chunk_alloc(p, T) != 0) return -1;
    // this slot's pinned staging is reused only after its previous H2D completed (QF8_PLE_SLOTS chunks ago)
    if (had && hipEventSynchronize(p->cev[slot]) != hipSuccess) return -1;
    uint8_t *pin = p->cemb_pin[slot];
    for (int t = 0; t < T; t++) {
        p->hist[0][2] = p->hist[0][1];
        p->hist[0][1] = p->hist[0][0];
        p->hist[0][0] = tokens[t];
        uint64_t t0 = (uint64_t)p->hist[0][0], t1 = (uint64_t)p->hist[0][1], t2 = (uint64_t)p->hist[0][2];
        uint64_t m2 = (t0 * p->mult[0]) ^ (t1 * p->mult[1]);
        uint64_t m3 = m2 ^ (t2 * p->mult[2]);
        uint8_t *dst = pin + (size_t)t * QF8_NEMBD;
        const uint8_t *src[QF8_PLE_H];
        for (int hh = 0; hh < QF8_PLE_H; hh++) {
            uint64_t mixed = (hh < 8) ? m2 : m3;
            int64_t r = (int64_t)(mixed % (uint64_t)p->vsizes[hh]) + p->offsets[hh];
            int shard = (int)(r / PLE_SHARD_ROWS);
            int64_t local = r - (int64_t)shard * PLE_SHARD_ROWS;
            if (shard < 0 || shard >= p->n_shards) return -1;
            src[hh] = (const uint8_t *)st->shard_maps[p->shard_file[shard]] + p->shard_off[shard] + local * QF8_PLE_ROW;
            ple_advise_row(src[hh]);
        }
        for (int hh = 0; hh < QF8_PLE_H; hh++) memcpy(dst + hh * QF8_PLE_ROW, src[hh], QF8_PLE_ROW);
    }
    if (hipMemcpyAsync(p->cemb_fp8[slot], pin, (size_t)T * QF8_NEMBD, hipMemcpyHostToDevice, p->xfer) != hipSuccess ||
        hipEventRecord(p->cev[slot], p->xfer) != hipSuccess) return -1;
    return 0;
}
int qf8_ple_step_chunk(Qf8Ple *p, const QfStore *st, const int *tokens, int T) { return qf8_ple_prefetch_chunk(p, st, tokens, T, 0); }
// Device: R[t] += PLE(R[t]) for the T positions. Dequant over all rows, key /
// value projections in 8-row groups (weights read once per 8 positions instead
// of once per position), norms / gate / gated-value over the position grid, and
// the recurrent conv ring in ONE launch (k8_ple_conv_add_chunk).
void qf8_ple_apply_chunk_slot(Qf8Ple *p, float *R, int T, int slot, hipStream_t s) {
    if (!p || T < 1 || T > p->ccap || slot < 0 || slot >= QF8_PLE_SLOTS) return;
    hipStreamWaitEvent(s, p->cev[slot], 0);
    k8_ple_dequant<<<(T * QF8_NEMBD + 255) / 256, 256, 0, s>>>(p->cemb, p->cemb_fp8[slot], p->table_scale, T * QF8_NEMBD);
    for (int r0 = 0; r0 < T; r0 += 8) {
        const int nr = (T - r0) < 8 ? (T - r0) : 8;
        qf8_gemv_bf16_M(p->key_proj, p->cemb + (size_t)r0 * QF8_NEMBD, p->ckey + (size_t)r0 * QF8_RSIZE, QF8_RSIZE, QF8_NEMBD, nr, 1.f, 0.f, s);
        qf8_gemv_bf16_M(p->value_proj, p->cemb + (size_t)r0 * QF8_NEMBD, p->cval + (size_t)r0 * QF8_NEMBD, QF8_NEMBD, QF8_NEMBD, nr, 1.f, 0.f, s);
    }
    k8_ple_norm2<<<dim3(2 * QF8_HCC, T), 256, 0, s>>>(p->ckn, p->cqn, p->ckey, R, p->norm_key, p->norm_query);
    k8_ple_gate<<<T, 256, 0, s>>>(p->ckn, p->cqn, p->cgate);
    k8_ple_apply_nc<<<dim3(QF8_HCC, T), 256, 0, s>>>(p->cgv, p->cnc, p->cval, p->cgate, p->norm_conv);
    k8_ple_conv_add_chunk<<<QF8_RSIZE / 256, 256, 0, s>>>(R, p->cnc, p->cgv, p->ring, p->conv1d, T);
}
void qf8_ple_apply_chunk(Qf8Ple *p, float *R, int T, hipStream_t s) { qf8_ple_apply_chunk_slot(p, R, T, 0, s); }

int qf8_ple_fork_rows(Qf8Ple *p, int M, hipStream_t s) {   // rows 1..M-1 <- row 0: n-gram history, ring, phase
    if (!p) return 0;
    for (int r = 1; r < M && r < QF8_MROWS; r++) {
        p->hist[r][0] = p->hist[0][0]; p->hist[r][1] = p->hist[0][1]; p->hist[r][2] = p->hist[0][2];
        p->cur[r] = p->cur[0];
        if (hipMemcpyAsync(p->ring + (size_t)r * 9 * QF8_RSIZE, p->ring, (size_t)9 * QF8_RSIZE * sizeof(float), hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
    }
    return 0;
}
int qf8_ple_commit_rows(Qf8Ple *p, int r, hipStream_t s) {   // row 0 <- row r: n-gram history, ring, phase (decision/commit loop)
    if (!p) return 0;
    if (r < 0 || r >= QF8_MROWS) return -1;
    p->hist[0][0] = p->hist[r][0]; p->hist[0][1] = p->hist[r][1]; p->hist[0][2] = p->hist[r][2];
    p->cur[0] = p->cur[r];
    if (hipMemcpyAsync(p->ring, p->ring + (size_t)r * 9 * QF8_RSIZE, (size_t)9 * QF8_RSIZE * sizeof(float), hipMemcpyDeviceToDevice, s) != hipSuccess) return -1;
    return 0;
}
int qf8_ple_reset(Qf8Ple *p) {
    if (!p) return 0;
    for (int r = 0; r < QF8_MROWS; r++) {
        p->hist[r][0] = p->hist[r][1] = p->hist[r][2] = PLE_EOS;
        p->cur[r] = 0;
    }
    return hipMemset(p->ring, 0, (size_t)QF8_MROWS * 9 * QF8_RSIZE * sizeof(float)) == hipSuccess ? 0 : -1;
}

const float *qf8_ple_dbg(const Qf8Ple *p, int which) {
    if (!p) return NULL;
    switch (which) {
        case 0: return p->emb;  case 1: return p->key; case 2: return p->val;
        case 3: return p->kn;   case 4: return p->qn;  case 5: return p->gate;
        case 6: return p->gv;   case 7: return p->nc;
        default: return NULL;
    }
}

void qf8_ple_apply_row(Qf8Ple *p, float *R, int row, hipStream_t s) {
    if (!p) return;
    if (row < 0 || row >= QF8_MROWS) return;
    const size_t oN = (size_t)row * QF8_NEMBD, oR = (size_t)row * QF8_RSIZE;
    float *emb = p->emb + oN, *key = p->key + oR, *val = p->val + oN;
    float *kn = p->kn + oR, *qn = p->qn + oR, *gate = p->gate + (size_t)row * QF8_HCC;
    float *gv = p->gv + oR, *nc = p->nc + oR;
    float *ring = p->ring + (size_t)row * 9 * QF8_RSIZE;
    hipStreamWaitEvent(s, p->ev[row][p->cur[row]], 0);
    QF8_LAUNCH(k8_ple_dequant, QF8_NEMBD / 256, 256, 0, s, emb, p->emb_fp8[row][p->cur[row]],
               p->table_scale, QF8_NEMBD);
    qf8_gemv_bf16(p->key_proj, emb, key, QF8_RSIZE, QF8_NEMBD, 1.f, 0.f, s);
    qf8_gemv_bf16(p->value_proj, emb, val, QF8_NEMBD, QF8_NEMBD, 1.f, 0.f, s);
    QF8_LAUNCH(k8_ple_norm2, 2 * QF8_HCC, 256, 0, s, kn, qn, key, R,
               p->norm_key, p->norm_query);
    QF8_LAUNCH(k8_ple_gate, 1, 256, 0, s, kn, qn, gate);
    QF8_LAUNCH(k8_ple_apply_nc, QF8_HCC, 256, 0, s, gv, nc, val, gate, p->norm_conv);
    QF8_LAUNCH(k8_ple_conv_add, QF8_RSIZE / 256, 256, 0, s, R, nc, gv, ring, p->conv1d);
}
void qf8_ple_apply(Qf8Ple *p, float *R, hipStream_t s) { qf8_ple_apply_row(p, R, 0, s); }
