// model.c - load Qwen3.8-Flash-Next from RadixArk NVFP4 safetensors into device memory
#define _GNU_SOURCE
#include "qwenflash.h"

#include <stdlib.h>

// Layer range for a split deployment. Read once; 0 keeps the single-node
// behaviour byte for byte.
int qf_layer_begin(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QF_LAYER_BEGIN");
        v = e ? atoi(e) : 0;
        if (v < 0) v = 0;
    }
    return v;
}
#ifndef SYNTH
#include "planner.h"
#include "qf_fp4tc.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cuda_bf16.h>

static int g_trace = -1;
static void *load_dev(const QfStore *st, const char *name, uint64_t *nbytes_out);

void *qf_load_dev_global(const QfStore *st, const char *name) { return load_dev(st, name, NULL); }

static const QfEntry *qf_find_shim(const QfStore *st, const char *name) {
    const QfEntry *e = qf_find(st, name);
    if (e) return e;
    static char alt[QF_MAX_NAME];
    // strip ".language_model" for text-only checkpoints
    const char *lm = strstr(name, ".language_model");
    if (lm) {
        snprintf(alt, sizeof(alt), "%.*s%s", (int)(lm - name), name, lm + strlen(".language_model"));
        e = qf_find(st, alt);
    }
    return e;
}
static void *load_dev(const QfStore *st, const char *name, uint64_t *nbytes_out) {
    const QfEntry *e = qf_find_shim(st, name);

    if (!e) { fprintf(stderr, "MISSING %s\n", name); return NULL; }
    if (g_trace < 0) g_trace = getenv("QF_TRACE") ? 1 : 0;
    if (g_trace) fprintf(stderr, "LOAD %s (%llu B)\n", name, (unsigned long long)e->rec.nbytes);
    void *dev = NULL;
    if (cudaMalloc(&dev, e->rec.nbytes) != cudaSuccess) { fprintf(stderr, "ALLOC FAIL %s (%llu B): %s\n", name, (unsigned long long)e->rec.nbytes, cudaGetErrorString(cudaGetLastError())); return NULL; }
    if (cudaMemcpy(dev, (char *)st->shard_maps[e->rec.file_idx] + e->rec.data_off,
                   e->rec.nbytes, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    if (nbytes_out) *nbytes_out = e->rec.nbytes;
    return dev;
}

static void *load_dev_scaled(const QfStore *st, const char *name, float **scale2_host) {
    // loads tensor + its weight_scale_2 (returned as host float, folded later)
    void *dev = load_dev(st, name, NULL);
    if (!dev) return NULL;
    char base[QF_MAX_NAME];
    snprintf(base, sizeof(base), "%s", name);
    char *dot = strstr(base, ".weight");
    if (dot && scale2_host) {
        char s2name[QF_MAX_NAME];
        snprintf(s2name, sizeof(s2name), "%.*s.weight_scale_2", (int)(dot - base), base);
        const QfEntry *e2 = qf_find(st, s2name);
        if (e2) {
            *scale2_host = (float *)malloc(4);
            memcpy(*scale2_host, (char *)st->shard_maps[e2->rec.file_idx] + e2->rec.data_off, 4);
        } else *scale2_host = NULL;
    }
    return dev;
}

#ifndef NHEAD
#ifndef NHEAD
#define NHEAD 24
#endif
#endif
#ifndef NKV
#ifndef NKV
#define NKV 2
#endif
#endif
#ifndef HDIM
#define HDIM 256
#endif
#ifndef NEXPUSED
#ifndef NEXPUSED
#define NEXPUSED 10
#endif
#endif
#ifndef NVOCAB
#ifndef NVOCAB
#define NVOCAB 248320
#endif
#endif
#ifndef DINN
#ifndef DINN
#define DINN 10240
#endif
#endif
#ifndef GDN_KD
#ifndef GDN_KD
#define GDN_KD 128
#endif
#endif
#ifndef GDN_VD
#ifndef GDN_VD
#define GDN_VD 128
#endif
#endif
#ifndef DTRANK
#ifndef DTRANK
#define DTRANK 48
#endif
#endif
#ifndef GDN_KH
#ifndef GDN_KH
#define GDN_KH 16
#endif
#endif
#ifndef NEXP
#ifndef NEXP
#define NEXP 512
#endif
#endif
#ifndef NFF
#ifndef NFF
#define NFF 640
#endif
#endif
#ifndef NEMBD
#ifndef NEMBD
#define NEMBD 2560
#endif
#endif
#ifndef NLAYER
#define NLAYER 48
#endif
#ifndef HCC
#ifndef HCC
#define HCC 4
#endif
#endif
#ifndef HCL
#ifndef HCL
#define HCL 320
#endif
#endif
#ifndef PLEH
#ifndef PLEH
#define PLEH 16
#endif
#endif
#ifndef PLEDIM
#ifndef PLEDIM
#define PLEDIM 160
#endif
#endif
#define CHK(x) do { if (!(x)) return -1; } while (0)

static float s2_gate_tab[NLAYER][512], s2_up_tab[NLAYER][512], s2_down_tab[NLAYER][512];
float *qf_layer_s2_gate(int il) { return s2_gate_tab[il]; }
float *qf_layer_s2_up(int il) { return s2_up_tab[il]; }
float *qf_layer_s2_down(int il) { return s2_down_tab[il]; }

#ifndef SYNTH
// ---- full-resident expert staging (QF_EXPERT_MODE=full) -------------------
// All 512 experts of every layer are staged to the device once at load time
// through a FIXED pinned host buffer (QF_SPARK_STAGING_BYTES, two 4 MiB halves
// ping-ponged) instead of pageable mmap->device copies. After each tensor is
// copied out of the mmap store, its source range is dropped from the page
// cache (madvise + posix_fadvise DONTNEED): GB10 is unified memory, so without
// drop-behind the ~63 GiB of expert bytes would sit in the page cache as a
// second copy on top of the device residency.

#define QF_STAGING_HALF (QF_SPARK_STAGING_BYTES / 2)

static int g_staging_half = 0;   // ping-pong half for the current load (single-threaded)

// Copy one tensor from the mmap store to the device through a pinned half,
// then drop the source pages. Expert tensors are at most 800 KiB, far below
// STAGING_HALF; anything larger is a format error, not a reason to grow the
// staging (fixed means fixed).
static int stage_upload(QfModel *m, int *half, void *dev_dst, const QfEntry *e) {
    if (e->rec.nbytes > QF_STAGING_HALF) {
        fprintf(stderr, "staging: tensor %s too large (%llu B > %u B half)\n",
                e->name, (unsigned long long)e->rec.nbytes, (unsigned)QF_STAGING_HALF);
        return -1;
    }
    int h = *half;
    // Load-time only. Reusing a half requires the previous H2D FROM THAT HALF
    // to have drained - not every H2D. Waiting on the whole stream before each
    // upload serialized the host memcpy behind the previous device copy and
    // defeated the double buffer entirely: measured 147,458 cudaStreamSynchronize
    // calls for a full-resident load (512 experts x 48 layers x 6 tensors),
    // 2.46 s of pure sync. One event per half lets the memcpy into half h
    // overlap the in-flight H2D out of half h^1.
    static cudaEvent_t ev[2];
    static int ev_ready = 0;
    if (!ev_ready) {
        if (cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) != cudaSuccess) {
            fprintf(stderr, "staging: event create failed\n");
            return -1;
        }
        ev_ready = 1;
    } else if (cudaEventSynchronize(ev[h]) != cudaSuccess) {
        fprintf(stderr, "staging: event sync failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    char *pin = (char *)m->exp_staging + h * QF_STAGING_HALF;
    const char *src = (const char *)m->store.shard_maps[e->rec.file_idx] + e->rec.data_off;
    memcpy(pin, src, e->rec.nbytes);
    // Drop-behind immediately after the read: the pages were just faulted in
    // and must not linger as a duplicate of the device copy.
    qf_store_drop_range(&m->store, e->rec.file_idx, e->rec.data_off, e->rec.nbytes);
    if (cudaMemcpyAsync(dev_dst, pin, e->rec.nbytes, cudaMemcpyHostToDevice, 0) != cudaSuccess) {
        fprintf(stderr, "staging: H2D failed for %s: %s\n", e->name,
                cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    if (cudaEventRecord(ev[h], 0) != cudaSuccess) {
        fprintf(stderr, "staging: event record failed\n");
        return -1;
    }
    *half = h ^ 1;
    return 0;
}

// Locate the 6 store entries (packed weight + block scales for gate/up/down)
// of one routed expert. Shared by qf_prepare_expert, the full-resident eager
// load, and the wave2 miss-service path.
static int find_expert_entries(const QfStore *st, int layer, int expert,
                               const QfEntry *weights[3], const QfEntry *scales[3]) {
    const char *proj[3] = {"gate_proj", "up_proj", "down_proj"};
    const size_t weight_bytes = (size_t)NFF * NEMBD / 2;
    const size_t scale_bytes = (size_t)NFF * NEMBD / 16;
    for (int p = 0; p < 3; p++) {
        char name[QF_MAX_NAME];
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.experts.%d.%s.weight", layer, expert, proj[p]);
        weights[p] = qf_find_shim(st, name);
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.experts.%d.%s.weight_scale", layer, expert, proj[p]);
        scales[p] = qf_find_shim(st, name);
        if (!weights[p] || !scales[p] || weights[p]->rec.nbytes != weight_bytes || scales[p]->rec.nbytes != scale_bytes) {
            fprintf(stderr, "invalid routed expert tensor L%d E%d %s\n", layer, expert, proj[p]);
            return -1;
        }
    }
    return 0;
}
#endif

static int load_layer(QfModel *m, QfLayer *L, int il, int is_recr, int is_ple) {
    fprintf(stderr, "load_layer L%d rec=%d\n", il, is_recr);
    char n[QF_MAX_NAME];
    const QfStore *st = &m->store;
    snprintf(n, sizeof(n), "model.language_model.layers.%d.attn_hyper_connection.hc_norm.weight", il);
    CHK(L->hc_attn_norm = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_down.weight", il);
    CHK(L->hc_attn_down = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.attn_hyper_connection.input_mix_weight_up.weight", il);
    CHK(L->hc_attn_up = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.attn_hyper_connection.block_inject_weight.weight", il);
    CHK(L->hc_attn_inject = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp_hyper_connection.hc_norm.weight", il);
    CHK(L->hc_ffn_norm = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight", il);
    CHK(L->hc_ffn_down = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight", il);
    CHK(L->hc_ffn_up = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp_hyper_connection.block_inject_weight.weight", il);
    CHK(L->hc_ffn_inject = load_dev(st, n, NULL));

    if (is_recr) {
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", il);
        CHK(L->qkv = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", il);
        CHK(L->zgate = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.conv1d.weight", il);
        CHK(L->conv1d = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.dt_bias", il);
        CHK(L->dt_bias = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.A_log", il);
        CHK(L->a = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", il);
        CHK(L->beta = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", il);
        CHK(L->alpha = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.norm.weight", il);
        CHK(L->gdn_norm = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.linear_attn.out_proj.weight", il);
        CHK(L->gdn_out = load_dev(st, n, NULL));
    } else {
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.q_proj.weight", il);
        CHK(L->wq = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.k_proj.weight", il);
        CHK(L->wk = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.v_proj.weight", il);
        CHK(L->wv = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.o_proj.weight", il);
        CHK(L->wo = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.q_norm.weight", il);
        CHK(L->q_norm = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.k_norm.weight", il);
        CHK(L->k_norm = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.indexer.index_qk_proj.weight", il);
        CHK(L->idx_qk = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.indexer.q_layernorm.weight", il);
        CHK(L->idx_qnorm = load_dev(st, n, NULL));
        snprintf(n, sizeof(n), "model.language_model.layers.%d.self_attn.indexer.k_layernorm.weight", il);
        CHK(L->idx_knorm = load_dev(st, n, NULL));
    }

    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.gate.weight", il);
    CHK(L->router = load_dev(st, n, NULL));
    // routed experts: 512 per layer, gathered from layer-NNNN-experts-XXXX-YYYY shards
    // v1: load contiguous [512][640][1280] u8 + [512][640][160] f8 + per-expert scale2
#ifdef SYNTH
    {
        // synth: HF fused 3D BF16 experts: gate_up_proj [NEXP, 2*NFF, NEMBD], down [NEXP, NEMBD, NFF]
        size_t gusz = (size_t)NEXP * 2 * NFF * NEMBD;
        size_t dsz2 = (size_t)NEXP * NEMBD * NFF;
        cudaMalloc(&L->exp_gate, (size_t)NEXP * NFF * NEMBD * 2);  // gate rows
        cudaMalloc(&L->exp_up, (size_t)NEXP * NFF * NEMBD * 2);    // up rows
        cudaMalloc(&L->exp_down, dsz2);
        cudaMalloc(&L->exp_scale, 4); cudaMalloc(&L->exp_scale_down, 4);
        char w[QF_MAX_NAME];
        std::string base1 = "model.language_model.layers." + std::to_string(il);
        std::string base2 = "model.layers." + std::to_string(il);
        const QfEntry *we = qf_find_shim(st, ("model.layers." + std::to_string(il) + ".mlp.experts.gate_up_proj.weight").c_str());
        if (!we) we = qf_find_shim(st, ("model.layers." + std::to_string(il) + ".mlp.experts.gate_up_proj").c_str());
        if (!we) we = qf_find_shim(st, (base1 + ".mlp.experts.gate_up_proj.weight").c_str());
        const QfEntry *de = qf_find_shim(st, ("model.layers." + std::to_string(il) + ".mlp.experts.down_proj.weight").c_str());
        if (!de) de = qf_find_shim(st, ("model.layers." + std::to_string(il) + ".mlp.experts.down_proj").c_str());
        if (!de) de = qf_find_shim(st, (base1 + ".mlp.experts.down_proj.weight").c_str());
        if (!we || !de) { fprintf(stderr, "MISSING fused experts L%d\n", il); return -1; }
        // split: for each expert e: gate rows = we[e*2*NFF .. +NFF], up rows = we[(e*2+1)*NFF .. +NFF]
        const uint8_t *src = (const uint8_t *)((char *)st->shard_maps[we->rec.file_idx] + we->rec.data_off);
        uint8_t *hg = (uint8_t *)malloc(gusz), *hu = (uint8_t *)malloc(gusz);
        if (!hg || !hu) return -1;
        for (int e = 0; e < NEXP; e++) {
            memcpy(hg + (size_t)e * NFF * NEMBD * 2, src + (size_t)e * 2 * NFF * NEMBD * 2, (size_t)NFF * NEMBD * 2);
            memcpy(hu + (size_t)e * NFF * NEMBD * 2, src + (size_t)(e * 2 + 1) * NFF * NEMBD * 2, (size_t)NFF * NEMBD * 2);
        }
        if (cudaMemcpy(L->exp_gate, hg, gusz, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaMemcpy(L->exp_up, hu, gusz, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaMemcpy(L->exp_down, (char *)st->shard_maps[de->rec.file_idx] + de->rec.data_off, dsz2, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        free(hg); free(hu);
        fprintf(stderr, "synth experts L%d copied\n", il);
    }
#else
    {
        int slots = 192;
        if (const char *v = getenv("QF_EXPERT_CACHE_SLOTS")) slots = atoi(v);
        if (m->exp_mode == QF_SPARK_MODE_FULL) slots = NEXP;  // full-resident mode
        if (slots < NEXPUSED) slots = NEXPUSED;
        if (slots > NEXP) slots = NEXP;
        L->exp_cache_slots = slots;
        L->exp_slot_for_expert = (int *)malloc((size_t)NEXP * sizeof(int));
        L->exp_expert_in_slot = (int *)malloc((size_t)slots * sizeof(int));
        L->exp_slot_age = (uint64_t *)calloc((size_t)slots, sizeof(uint64_t));
        if (!L->exp_slot_for_expert || !L->exp_expert_in_slot || !L->exp_slot_age) return -1;
        for (int e = 0; e < NEXP; e++) L->exp_slot_for_expert[e] = -1;
        for (int slot = 0; slot < slots; slot++) L->exp_expert_in_slot[slot] = -1;

        const size_t weight_bytes = (size_t)NFF * NEMBD / 2;
        const size_t scale_bytes = (size_t)NFF * NEMBD / 16;
        if (cudaMalloc(&L->exp_gate, (size_t)slots * weight_bytes) != cudaSuccess) return -1;
        if (cudaMalloc(&L->exp_up, (size_t)slots * weight_bytes) != cudaSuccess) return -1;
        if (cudaMalloc(&L->exp_down, (size_t)slots * weight_bytes) != cudaSuccess) return -1;
        if (cudaMalloc(&L->exp_scale, (size_t)slots * scale_bytes) != cudaSuccess) return -1;
        if (cudaMalloc(&L->exp_scale_up, (size_t)slots * scale_bytes) != cudaSuccess) return -1;
        if (cudaMalloc(&L->exp_scale_down, (size_t)slots * scale_bytes) != cudaSuccess) return -1;

        // Per-expert scalar scales are tiny; keep all of them on the host while
        // the packed weights and block scales are loaded into the LRU on demand.
        for (int e = 0; e < NEXP; e++) {
            const char *proj[3] = {"gate_proj", "up_proj", "down_proj"};
            float *dst[3] = {s2_gate_tab[il], s2_up_tab[il], s2_down_tab[il]};
            for (int p = 0; p < 3; p++) {
                char s2[QF_MAX_NAME];
                snprintf(s2, sizeof(s2), "model.language_model.layers.%d.mlp.experts.%d.%s.weight_scale_2", il, e, proj[p]);
                const QfEntry *s2e = qf_find_shim(st, s2);
                if (!s2e || s2e->rec.nbytes != sizeof(float)) {
                    fprintf(stderr, "MISSING/INVALID %s\n", s2);
                    return -1;
                }
                memcpy(dst[p] + e, (char *)st->shard_maps[s2e->rec.file_idx] + s2e->rec.data_off, sizeof(float));
            }
        }

        // wave2: device-resident routing state. During decode the slot map, LRU
        // clocks and miss bookkeeping live on device; the host maps above only
        // seed them here. The mailbox is pinned+mapped so the dispatch kernel
        // signals misses with zero D2H copies.
        if (cudaMalloc((void **)&L->exp_slot_dev, (size_t)NEXP * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->exp_expert_in_slot_dev, (size_t)slots * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->exp_slot_age_dev, (size_t)slots * sizeof(uint64_t)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->exp_clock_dev, sizeof(uint64_t)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->route_slot_dev, (size_t)NEXPUSED * sizeof(int)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->slot_ready_dev, (size_t)slots * sizeof(uint32_t)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->s2_gate_dev, (size_t)NEXP * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->s2_up_dev, (size_t)NEXP * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMalloc((void **)&L->s2_down_dev, (size_t)NEXP * sizeof(float)) != cudaSuccess) return -1;
        if (cudaMemset(L->exp_slot_age_dev, 0, (size_t)slots * sizeof(uint64_t)) != cudaSuccess) return -1;
        if (cudaMemset(L->exp_clock_dev, 0, sizeof(uint64_t)) != cudaSuccess) return -1;
        // doorbells: nonzero = resident/ready, 0 = upload in flight (set by dispatch)
        if (cudaMemset(L->slot_ready_dev, 0x01, (size_t)slots * sizeof(uint32_t)) != cudaSuccess) return -1;
        if (cudaMemcpy(L->s2_gate_dev, s2_gate_tab[il], (size_t)NEXP * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaMemcpy(L->s2_up_dev, s2_up_tab[il], (size_t)NEXP * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaMemcpy(L->s2_down_dev, s2_down_tab[il], (size_t)NEXP * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaHostAlloc((void **)&L->route_mb_host, sizeof(QfRouteMailbox), cudaHostAllocMapped) != cudaSuccess) return -1;
        memset(L->route_mb_host, 0, sizeof(QfRouteMailbox));
        if (cudaHostGetDevicePointer((void **)&L->route_mb_dev, L->route_mb_host, 0) != cudaSuccess) return -1;

        if (slots == NEXP) {
            // Full-resident (Spark) mode: every routed expert occupies a fixed
            // slot (slot == expert), uploaded once here through the fixed
            // pinned staging buffer with per-tensor page-cache drop-behind.
            // Device dispatch then never misses, the mailbox stays empty, and
            // the host never participates in routing during decode.
            if (!m->exp_staging) {
                if (cudaHostAlloc(&m->exp_staging, QF_SPARK_STAGING_BYTES,
                                  cudaHostAllocDefault) != cudaSuccess) {
                    fprintf(stderr, "staging: cudaHostAlloc(%u) failed: %s\n",
                            (unsigned)QF_SPARK_STAGING_BYTES,
                            cudaGetErrorString(cudaGetLastError()));
                    return -1;
                }
                m->exp_staging_bytes = QF_SPARK_STAGING_BYTES;
            }
            void *weight_dst[3] = {L->exp_gate, L->exp_up, L->exp_down};
            void *scale_dst[3] = {L->exp_scale, L->exp_scale_up, L->exp_scale_down};
            for (int e = 0; e < NEXP; e++) {
                const QfEntry *weights[3], *scales[3];
                if (find_expert_entries(st, il, e, weights, scales) != 0) return -1;
                for (int p = 0; p < 3; p++) {
                    if (stage_upload(m, &g_staging_half, (char *)weight_dst[p] + (size_t)e * weight_bytes, weights[p]) != 0) return -1;
                    if (stage_upload(m, &g_staging_half, (char *)scale_dst[p] + (size_t)e * scale_bytes, scales[p]) != 0) return -1;
                }
                L->exp_slot_for_expert[e] = e;
                L->exp_expert_in_slot[e] = e;
            }
            fprintf(stderr, "layer %d: full-resident experts (%d slots)\n", il, slots);
            if (qf_fp4tc_supported()) {
                qf_fp4tc_repack_slab(L->exp_gate, weight_bytes, NEXP, 0);
                qf_fp4tc_repack_slab(L->exp_up, weight_bytes, NEXP, 0);
                qf_fp4tc_repack_slab(L->exp_down, weight_bytes, NEXP, 0);
            }
        }
        // seed the device maps (identity in full-resident mode, all-empty otherwise)
        if (cudaMemcpy(L->exp_slot_dev, L->exp_slot_for_expert, (size_t)NEXP * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        if (cudaMemcpy(L->exp_expert_in_slot_dev, L->exp_expert_in_slot, (size_t)slots * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    }
#endif
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.shared_expert.gate_proj.weight", il);
    CHK(L->shexp_gate = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.shared_expert.up_proj.weight", il);
    CHK(L->shexp_up = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.shared_expert.down_proj.weight", il);
    CHK(L->shexp_down = load_dev(st, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.layers.%d.mlp.shared_expert_gate.weight", il);
    CHK(L->shexp_gate_inp = load_dev(st, n, NULL));
    return 0;
}

int qf_prepare_expert(QfModel *m, int layer, int expert, void *stream) {
#ifdef SYNTH
    (void)m; (void)layer; (void)stream;
    return expert;
#else
    // NOTE: decode no longer calls this (wave2 dispatch is device-resident);
    // it remains as the host-side cache-fill entry point for tooling.
    if (!m || layer < 0 || layer >= NLAYER || expert < 0 || expert >= NEXP) return -1;
    QfLayer *L = &m->layers[layer];
    if (!L->exp_slot_for_expert || L->exp_cache_slots <= 0) return -1;

    int slot = L->exp_slot_for_expert[expert];
    if (slot >= 0) {
        L->exp_slot_age[slot] = ++L->exp_cache_clock;
        return slot;
    }

    const QfEntry *weights[3], *scales[3];
    const size_t weight_bytes = (size_t)NFF * NEMBD / 2;
    const size_t scale_bytes = (size_t)NFF * NEMBD / 16;
    if (find_expert_entries(&m->store, layer, expert, weights, scales) != 0) return -1;

    slot = 0;
    for (int i = 0; i < L->exp_cache_slots; i++) {
        if (L->exp_expert_in_slot[i] < 0) { slot = i; break; }
        if (L->exp_slot_age[i] < L->exp_slot_age[slot]) slot = i;
    }

    void *weight_dst[3] = {L->exp_gate, L->exp_up, L->exp_down};
    void *scale_dst[3] = {L->exp_scale, L->exp_scale_up, L->exp_scale_down};
    cudaStream_t s = (cudaStream_t)stream;
    for (int p = 0; p < 3; p++) {
        const void *wsrc = (const char *)m->store.shard_maps[weights[p]->rec.file_idx] + weights[p]->rec.data_off;
        const void *ssrc = (const char *)m->store.shard_maps[scales[p]->rec.file_idx] + scales[p]->rec.data_off;
        cudaError_t e = cudaMemcpyAsync((char *)weight_dst[p] + (size_t)slot * weight_bytes,
                                        wsrc, weight_bytes, cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) {
            fprintf(stderr, "expert weight upload failed L%d E%d: %s\n", layer, expert, cudaGetErrorString(e));
            return -1;
        }
        e = cudaMemcpyAsync((char *)scale_dst[p] + (size_t)slot * scale_bytes,
                            ssrc, scale_bytes, cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) {
            fprintf(stderr, "expert scale upload failed L%d E%d: %s\n", layer, expert, cudaGetErrorString(e));
            return -1;
        }
    }

    if (qf_fp4tc_supported())
        qf_fp4tc_repack_slot3(L->exp_gate, L->exp_up, L->exp_down, weight_bytes, slot, s);

    int evicted = L->exp_expert_in_slot[slot];
    if (evicted >= 0) L->exp_slot_for_expert[evicted] = -1;
    L->exp_expert_in_slot[slot] = expert;
    L->exp_slot_for_expert[expert] = slot;
    L->exp_slot_age[slot] = ++L->exp_cache_clock;
    return slot;
#endif
}

// wave2 ---------------------------------------------------------------------
// Host side of the bounded-cache miss protocol. During decode the device
// dispatch kernel (k_route_dispatch in cuda/qf.cu) owns all cache metadata:
// on a miss it picks the coldest slot on device, updates the device maps,
// drops the slot's doorbell, and records (expert, slot) in the layer's pinned
// mailbox. The host's only job here is the DMA: stage the expert through fixed
// pinned buffers, upload on a dedicated transfer stream, then ring the slot
// doorbell (the 4-byte doorbell copy is stream-ordered after the data). The
// expert GEMV spins on that doorbell on device, so the all-hit path involves
// no host wait and no D2H readback at all.
#ifndef SYNTH
#define QF_ROUTE_STAGE_BYTES (3 * ((size_t)NFF * NEMBD / 2 + (size_t)NFF * NEMBD / 16))
static cudaStream_t g_route_stream = NULL;
static uint8_t *g_route_stage[2] = {NULL, NULL};  // ping-pong pinned staging, one expert each
static cudaEvent_t g_route_ev[2];
static int g_route_armed[2] = {0, 0};
static int g_route_cur = 0;
static uint32_t *g_route_one = NULL;              // pinned doorbell value

int qf_route_service(QfModel *m, int layer) {
    if (!m || layer < 0 || layer >= NLAYER) return -1;
    QfLayer *L = &m->layers[layer];
    QfRouteMailbox *mb = L->route_mb_host;
    if (!mb) return -1;
    uint32_t nmiss = mb->miss_n;
    if (nmiss > NEXPUSED) nmiss = NEXPUSED;   // corrupt-mailbox guard
    if (!nmiss) return 0;

    if (!g_route_stream) {
        if (cudaStreamCreate(&g_route_stream) != cudaSuccess) return -1;
        for (int b = 0; b < 2; b++) {
            if (cudaMallocHost((void **)&g_route_stage[b], QF_ROUTE_STAGE_BYTES) != cudaSuccess) return -1;
            if (cudaEventCreateWithFlags(&g_route_ev[b], cudaEventDisableTiming) != cudaSuccess) return -1;
        }
        if (cudaMallocHost((void **)&g_route_one, sizeof(uint32_t)) != cudaSuccess) return -1;
        *g_route_one = 1u;
    }

    const size_t weight_bytes = (size_t)NFF * NEMBD / 2;
    const size_t scale_bytes = (size_t)NFF * NEMBD / 16;
    void *weight_dst[3] = {L->exp_gate, L->exp_up, L->exp_down};
    void *scale_dst[3] = {L->exp_scale, L->exp_scale_up, L->exp_scale_down};

    for (uint32_t i = 0; i < nmiss; i++) {
        int expert = mb->miss_expert[i];
        int slot = mb->miss_slot[i];
        if (expert < 0 || expert >= NEXP || slot < 0 || slot >= L->exp_cache_slots) {
            fprintf(stderr, "route mailbox corrupt L%d: expert %d slot %d\n", layer, expert, slot);
            mb->miss_n = 0;
            return -1;
        }
        const QfEntry *weights[3], *scales[3];
        if (find_expert_entries(&m->store, layer, expert, weights, scales) != 0) { mb->miss_n = 0; return -1; }

        int b = g_route_cur;
        if (g_route_armed[b]) cudaEventSynchronize(g_route_ev[b]);  // staging b reusable
        uint8_t *stg = g_route_stage[b];
        size_t off = 0;
        for (int p = 0; p < 3; p++) {
            memcpy(stg + off, (const char *)m->store.shard_maps[weights[p]->rec.file_idx] + weights[p]->rec.data_off, weight_bytes);
            off += weight_bytes;
            memcpy(stg + off, (const char *)m->store.shard_maps[scales[p]->rec.file_idx] + scales[p]->rec.data_off, scale_bytes);
            off += scale_bytes;
        }
        off = 0;
        for (int p = 0; p < 3; p++) {
            if (cudaMemcpyAsync((char *)weight_dst[p] + (size_t)slot * weight_bytes, stg + off,
                                weight_bytes, cudaMemcpyHostToDevice, g_route_stream) != cudaSuccess) { mb->miss_n = 0; return -1; }
            off += weight_bytes;
            if (cudaMemcpyAsync((char *)scale_dst[p] + (size_t)slot * scale_bytes, stg + off,
                                scale_bytes, cudaMemcpyHostToDevice, g_route_stream) != cudaSuccess) { mb->miss_n = 0; return -1; }
            off += scale_bytes;
        }
        // doorbell last: stream order guarantees expert data lands before the flag
        if (qf_fp4tc_supported())
            qf_fp4tc_repack_slot3(L->exp_gate, L->exp_up, L->exp_down,
                                  weight_bytes, slot, g_route_stream);
        if (cudaMemcpyAsync(L->slot_ready_dev + slot, g_route_one, sizeof(uint32_t),
                            cudaMemcpyHostToDevice, g_route_stream) != cudaSuccess) { mb->miss_n = 0; return -1; }
        cudaEventRecord(g_route_ev[b], g_route_stream);
        g_route_armed[b] = 1;
        g_route_cur ^= 1;
    }
    mb->miss_n = 0;
    return 0;
}
#else
int qf_route_service(QfModel *m, int layer) { (void)m; (void)layer; return 0; }
#endif

int qf_model_load(QfModel *m, const char *dir) {
    memset(m, 0, sizeof(*m));
    QfConfig *c = &m->cfg;
    c->n_vocab = NVOCAB; c->n_embd = NEMBD; c->n_layer = NLAYER;
    c->n_head = NHEAD; c->n_head_kv = NKV; c->head_dim = HDIM;
    c->n_experts = NEXP; c->n_expert_used = NEXPUSED; c->n_ff_exp = NFF;
    c->hc_count = HCC; c->hc_lowrank = HCL; c->full_attn_interval = 4;
    c->ssm_d_conv = 4; c->ssm_d_inner = DINN; c->ssm_d_state = GDN_KD;
    c->ssm_dt_rank = DTRANK; c->ssm_n_group = GDN_KH;
    c->indexer_n_head = 4; c->indexer_head_dim = 128; c->indexer_top_k = 2048;
    c->indexer_compress = 4; c->rms_eps = 1e-6f;
    c->ple_ngram = 3; c->ple_heads_per_ngram = PLEH; c->ple_conv_kernel = 4;
    c->ple_embd_per_layer = PLEDIM;

#ifndef SYNTH
    // Spark expert-residency mode + fail-before-allocation budget guard.
    // Runs before qf_store_open and before the first cudaMalloc: an unknown
    // mode or an over-budget plan refuses to touch anything. Default budget
    // 96 GiB (QF_BUDGET_GIB to override). In full mode the plan forces 512
    // slots/layer and includes the fixed pinned staging; in lru mode it
    // follows QF_EXPERT_CACHE_SLOTS (default 192).
    m->exp_mode = qf_spark_mode_env(stderr);
    if (m->exp_mode < 0) return -1;
    if (getenv("QF_EXPERT_FULL_RESIDENT")) m->exp_mode = QF_SPARK_MODE_FULL;  // legacy alias
    {
        QfPlanCfg pc;
        qf_plan_defaults_spark(&pc);
        pc.mode = m->exp_mode;
        if (m->exp_mode == QF_SPARK_MODE_FULL) pc.cache_slots = pc.n_experts;
        else if (const char *v = getenv("QF_EXPERT_CACHE_SLOTS")) pc.cache_slots = atoi(v);
        if (qf_budget_check(&pc, qf_budget_env("QF_BUDGET_GIB",
                                               QF_SPARK_DEFAULT_BUDGET_GIB), stderr) != 0) {
            fprintf(stderr, "budget guard: refusing to load\n");
            return -1;
        }
    }
#endif

    CHK(qf_store_open(&m->store, dir) == 0);
#ifndef SYNTH
    // PLE hash constants come FROM THE CHECKPOINT (I64 tensors next to the
    // n-gram table). Inventing them (splitmix64 seed games) makes every PLE
    // gather return wrong rows - garbage embeddings injected at the PLE layer
    // poison every downstream layer. The synth path already does this.
    {
        const QfEntry *em = qf_find_shim(&m->store, "model.language_model.layers.1.ple.ple_embedding.layer_multipliers");
        const QfEntry *eo = qf_find_shim(&m->store, "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets");
        const QfEntry *ev = qf_find_shim(&m->store, "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes");
        if (em && eo && ev && em->rec.nbytes == 24 && eo->rec.nbytes == 128 && ev->rec.nbytes == 128) {
            uint64_t mult[3]; int64_t offs[16], vs[16];
            memcpy(mult, (char *)m->store.shard_maps[em->rec.file_idx] + em->rec.data_off, 24);
            memcpy(offs, (char *)m->store.shard_maps[eo->rec.file_idx] + eo->rec.data_off, 128);
            memcpy(vs, (char *)m->store.shard_maps[ev->rec.file_idx] + ev->rec.data_off, 128);
            qf_ple_set_constants(mult, vs, offs, 16);
            fprintf(stderr, "ple constants from checkpoint: mult0=%llu vs0=%lld off0=%lld\n",
                    (unsigned long long)mult[0], (long long)vs[0], (long long)offs[0]);
        } else {
            fprintf(stderr, "WARNING: ple constant tensors missing/invalid (em=%p eo=%p ev=%p);"
                    " PLE gathers will use built-in constants\n", (void *)em, (void *)eo, (void *)ev);
        }
    }
#endif
    char n[QF_MAX_NAME];
    CHK(m->tok_embd = load_dev(&m->store, "model.language_model.embed_tokens.weight", NULL));
    CHK(m->lm_head = load_dev(&m->store, "lm_head.weight", NULL));
    snprintf(n, sizeof(n), "model.language_model.hyper_connection_mixer.hc_norm.weight");
    CHK(m->output_hc_norm = load_dev(&m->store, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight");
    CHK(m->output_hc_down = load_dev(&m->store, n, NULL));
    snprintf(n, sizeof(n), "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight");
    CHK(m->output_hc_up = load_dev(&m->store, n, NULL));

    m->layers = (QfLayer *)calloc(c->n_layer, sizeof(QfLayer));
    const int lb = qf_layer_begin();
    if (lb > 0)
        fprintf(stderr, "qf: LAYER SPLIT - this node owns layers %d..%d; "
                        "layers 0..%d are another node's and are not loaded\n",
                lb, c->n_layer - 1, lb - 1);
    for (int il = lb; il < c->n_layer; il++) {
        int is_recr = ((il + 1) % c->full_attn_interval) != 0;
        if (load_layer(m, &m->layers[il], il, is_recr, il == 1) != 0) {
            fprintf(stderr, "layer %d load failed\n", il);
            return -1;
        }
    }

    // PLE table stays host-mapped (51 GB fp8, gathered per token)
#ifndef SYNTH
    if (m->exp_mode == QF_SPARK_MODE_FULL) {
        // Belt-and-braces on top of the per-tensor drop-behind: nothing read
        // during the full-resident load (dense weights included) may linger
        // in the page cache as a duplicate of device memory.
        qf_store_drop_all(&m->store);
        fprintf(stderr, "full-resident: all %d experts/layer staged via pinned"
                " buffer, page cache dropped\n", NEXP);
    }
#endif
    return 0;
}

void qf_model_free(QfModel *m) {
    if (!m) return;
#ifndef SYNTH
    if (m->exp_staging) {
        cudaFreeHost(m->exp_staging);
        m->exp_staging = NULL;
        m->exp_staging_bytes = 0;
    }
#endif
    if (m->layers) {
        for (int il = 0; il < m->cfg.n_layer; il++) {
            QfLayer *L = &m->layers[il];
            free(L->exp_slot_for_expert);
            free(L->exp_expert_in_slot);
            free(L->exp_slot_age);
        }
        free(m->layers);
        m->layers = NULL;
    }
    // Device weight tensors are process-lifetime in v1; the store mapping is not.
    qf_store_close(&m->store);
}
