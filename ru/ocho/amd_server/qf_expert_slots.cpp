// qf_expert_slots.cpp - fixed-slot HBM expert residency for the gfx906
// backend. Mirrors the CUDA-side contract in source/src/model.cpp
// (qf_prepare_expert): bounded per-layer slots (default 192, env
// QF_EXPERT_CACHE_SLOTS), LRU eviction, host maps, tiny scale_2 tables kept
// on the host, packed weights + block scales uploaded only on miss.
//
// Difference from the current CUDA miss path (which cudaMemcpyAsyncs straight
// out of pageable mmap on the decode stream): misses here are gathered into a
// FIXED pinned staging buffer (double-buffered per GPU), transferred on a
// dedicated transfer stream, and the compute stream only waits on an event.
// That satisfies TARGET.md: "Expert misses use fixed pinned staging and
// overlap with compute" and "No unbounded full-expert load and no host-mapped
// zero-copy expert GEMV hot path".
//
// One Qf4ExpCache per layer (12 per GPU on the 4-card split). One Qf4ExpXfer
// per GPU, shared by that GPU's layer caches so pinned memory stays fixed at
// 2 x 2.64 MiB per card.
//
// Host syntax check (no ROCm needed):
//   g++ -DQF_HOST_CHECK -I. -I../../source/src -fsyntax-only qf_expert_slots.cpp
#include "qf_nvfp4_wave64.h"
#include "qf8_moe_i8.h"
#include "qf8_moe_fp8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kProj[3] = {"gate_proj", "up_proj", "down_proj"};

// ---------------------------------------------------------------------------
// Transfer context (per GPU)
// ---------------------------------------------------------------------------
int qf4_expxfer_init(Qf4ExpXfer *xf) {
    memset(xf, 0, sizeof(*xf));
    for (int b = 0; b < 2; b++) {
        if (hipHostMalloc(&xf->buf[b], QF5_EXP_SLOT_BYTES, 0) != hipSuccess) {
            fprintf(stderr, "qf5: pinned staging alloc failed (%zu bytes)\n",
                    QF5_EXP_SLOT_BYTES);
            return -1;
        }
        if (hipEventCreateWithFlags(&xf->ev[b], hipEventDisableTiming) != hipSuccess)
            return -1;
        // Buffer starts life "ready": record a satisfied event so the first
        // hipEventSynchronize below is a no-op.
        if (hipEventRecord(xf->ev[b], 0) != hipSuccess) return -1;
    }
    if (hipStreamCreate(&xf->xfer) != hipSuccess) return -1;
    if (hipEventRecord(xf->ev[0], xf->xfer) != hipSuccess ||
        hipEventRecord(xf->ev[1], xf->xfer) != hipSuccess) return -1;
    return 0;
}

void qf4_expxfer_free(Qf4ExpXfer *xf) {
    for (int b = 0; b < 2; b++) {
        if (xf->buf[b]) hipHostFree(xf->buf[b]);
        if (xf->ev[b]) hipEventDestroy(xf->ev[b]);
    }
    if (xf->xfer) hipStreamDestroy(xf->xfer);
    memset(xf, 0, sizeof(*xf));
}

// ---------------------------------------------------------------------------
// Per-layer fixed-slot cache
// ---------------------------------------------------------------------------
static void *q5_malloc(size_t n, const char *what) {
    void *p = NULL;
    if (hipMalloc(&p, n) != hipSuccess) {
        fprintf(stderr, "qf5: hipMalloc failed for %s (%.1f MiB)\n", what, n / 1048576.0);
        return NULL;
    }
    return p;
}

static int load_s2(const QfStore *st, int layer, int proj, float *dst) {
    char nm[QF_MAX_NAME];
    for (int e = 0; e < QF5_NEXP; e++) {
        snprintf(nm, sizeof(nm),
                 "model.language_model.layers.%d.mlp.experts.%d.%s.weight_scale_2",
                 layer, e, kProj[proj]);
        const QfEntry *en = qf_find(st, nm);
        if (!en || en->rec.nbytes != 4) {
            fprintf(stderr, "qf5 MISSING/BADSIZE %s\n", nm);
            return -1;
        }
        memcpy(dst + e, (const char *)st->shard_maps[en->rec.file_idx] + en->rec.data_off, 4);
    }
    return 0;
}

int qf4_expcache_init(Qf4ExpCache *c, const QfStore *st, int layer, int slots) {
    memset(c, 0, sizeof(*c));
    c->layer = layer;
    if (slots <= 0) {
        const char *env = getenv("QF_EXPERT_CACHE_SLOTS");
        slots = env ? atoi(env) : 192;
    }
    if (slots <= 0 || slots > QF5_NEXP) slots = QF5_NEXP;
    c->slots = slots;

    // FP8 production path: allocate the lane-chunk E4M3 cache INSTEAD of the
    // NVFP4 one (E4M3 is 2x the weight bytes but 1 byte -> 1 half on gfx906,
    // and the record's per-node rule is FP8 on the box with bandwidth headroom).
    const int fp8 = getenv("QF8_MOE_FP8") && getenv("QF8_MOE_FP8")[0] == '1';
    if (fp8) {
        const size_t w8 = (size_t)slots * (size_t)QF5_NFF * QF5_NEMBD;     // gate/up
        const size_t w8d = (size_t)slots * (size_t)QF5_NEMBD * QF5_NFF;    // down
        const size_t sg = (size_t)slots * (size_t)QF5_NFF * 64 * sizeof(float);
        const size_t sd = (size_t)slots * (size_t)QF5_NEMBD * 64 * sizeof(float);
        c->w8_gate = (uint8_t *)q5_malloc(w8,  "slot w8_gate");
        c->w8_up   = (uint8_t *)q5_malloc(w8,  "slot w8_up");
        c->w8_down = (uint8_t *)q5_malloc(w8d, "slot w8_down");
        c->sc_gate = (float *)q5_malloc(sg, "slot sc_gate");
        c->sc_up   = (float *)q5_malloc(sg, "slot sc_up");
        c->sc_down = (float *)q5_malloc(sd, "slot sc_down");
        if (!c->w8_gate || !c->w8_up || !c->w8_down || !c->sc_gate || !c->sc_up || !c->sc_down)
            return -1;
    }
    // M=1 int8-only residency: the NVFP4 slot arrays shrink to a staging ring;
    // every expert is converted into the int8 arena as it is loaded.
    const int i8only = getenv("QF_M1_INT8_ONLY") && getenv("QF_M1_INT8_ONLY")[0] == '1' &&
                       getenv("QF_M1_INT8") && getenv("QF_M1_INT8")[0] == '1';
    c->nv_ring = i8only ? 2 : 0;
    const size_t nvslots = i8only ? 2 : (size_t)slots;
    size_t wb = fp8 ? 0 : nvslots * QF5_EXP_W_BYTES;
    size_t sb = fp8 ? 0 : nvslots * QF5_EXP_S_BYTES;
    if (!fp8) {
    c->w_gate = (uint8_t *)q5_malloc(wb, "slot w_gate");
    c->w_up   = (uint8_t *)q5_malloc(wb, "slot w_up");
    c->w_down = (uint8_t *)q5_malloc(wb, "slot w_down");
    c->s_gate = (uint8_t *)q5_malloc(sb, "slot s_gate");
    c->s_up   = (uint8_t *)q5_malloc(sb, "slot s_up");
    c->s_down = (uint8_t *)q5_malloc(sb, "slot s_down");
    if (!c->w_gate || !c->w_up || !c->w_down || !c->s_gate || !c->s_up || !c->s_down)
        return -1;
    }

    // M=1 int8 arenas (in ADDITION to the NVFP4 slots they are built from).
    if (getenv("QF_M1_INT8") && getenv("QF_M1_INT8")[0] == '1') {
        const size_t wi = (size_t)slots * (size_t)QF5_NFF * QF5_NEMBD;     // same count for gate/up/down
        c->i8_gate  = (int8_t *)q5_malloc(wi, "slot i8_gate");
        c->i8_up    = (int8_t *)q5_malloc(wi, "slot i8_up");
        c->i8_down  = (int8_t *)q5_malloc(wi, "slot i8_down");
        c->i8s_gate = (float *)q5_malloc((size_t)slots * QF5_NFF * 4, "slot i8s_gate");
        c->i8s_up   = (float *)q5_malloc((size_t)slots * QF5_NFF * 4, "slot i8s_up");
        c->i8s_down = (float *)q5_malloc((size_t)slots * QF5_NEMBD * 4, "slot i8s_down");
        if (!c->i8_gate || !c->i8_up || !c->i8_down || !c->i8s_gate || !c->i8s_up || !c->i8s_down)
            return -1;
    }
    c->slot_for_expert = (int *)malloc(QF5_NEXP * sizeof(int));
    c->expert_in_slot  = (int *)malloc((size_t)slots * sizeof(int));
    c->age             = (uint64_t *)calloc((size_t)slots, sizeof(uint64_t));
    c->s2g = (float *)malloc(QF5_NEXP * sizeof(float));
    c->s2u = (float *)malloc(QF5_NEXP * sizeof(float));
    c->s2d = (float *)malloc(QF5_NEXP * sizeof(float));
    if (!c->slot_for_expert || !c->expert_in_slot || !c->age ||
        !c->s2g || !c->s2u || !c->s2d) return -1;
    for (int e = 0; e < QF5_NEXP; e++) c->slot_for_expert[e] = -1;
    for (int i = 0; i < slots; i++) c->expert_in_slot[i] = -1;
    c->clock = 0;

    if (load_s2(st, layer, 0, c->s2g) || load_s2(st, layer, 1, c->s2u) ||
        load_s2(st, layer, 2, c->s2d))
        return -1;

    // wave4 session 09: device-resident routing state. During decode the slot
    // map, reverse map, LRU clocks and miss mailbox live on device and are
    // owned by qf_rtr_dispatch; the host maps above only seed them here
    // (the cache starts empty: all slots -1). The mailbox is pinned+mapped so
    // the dispatch kernel signals misses with zero D2H copies.
    c->slot_for_expert_dev = (int *)q5_malloc(QF5_NEXP * sizeof(int), "slot map dev");
    c->expert_in_slot_dev  = (int *)q5_malloc((size_t)slots * sizeof(int), "rev map dev");
    c->age_dev             = (uint64_t *)q5_malloc((size_t)slots * sizeof(uint64_t), "age dev");
    c->clock_dev           = (uint64_t *)q5_malloc(sizeof(uint64_t), "clock dev");
    c->route_slot_dev      = (int *)q5_malloc(QF5_NEXPUSED * sizeof(int), "route slots dev");
    c->s2g_dev = (float *)q5_malloc(QF5_NEXP * sizeof(float), "s2g dev");
    c->s2u_dev = (float *)q5_malloc(QF5_NEXP * sizeof(float), "s2u dev");
    c->s2d_dev = (float *)q5_malloc(QF5_NEXP * sizeof(float), "s2d dev");
    if (!c->slot_for_expert_dev || !c->expert_in_slot_dev || !c->age_dev ||
        !c->clock_dev || !c->route_slot_dev ||
        !c->s2g_dev || !c->s2u_dev || !c->s2d_dev) return -1;
    if (hipMemcpy(c->slot_for_expert_dev, c->slot_for_expert,
                  QF5_NEXP * sizeof(int), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(c->expert_in_slot_dev, c->expert_in_slot,
                  (size_t)slots * sizeof(int), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(c->age_dev, 0, (size_t)slots * sizeof(uint64_t)) != hipSuccess ||
        hipMemset(c->clock_dev, 0, sizeof(uint64_t)) != hipSuccess ||
        hipMemcpy(c->s2g_dev, c->s2g, QF5_NEXP * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(c->s2u_dev, c->s2u, QF5_NEXP * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(c->s2d_dev, c->s2d, QF5_NEXP * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "qf5: device routing state upload failed L%d\n", layer);
        return -1;
    }
    if (hipHostMalloc((void **)&c->mb_host, sizeof(QfRouteMailbox),
                      hipHostMallocMapped) != hipSuccess) {
        fprintf(stderr, "qf5: route mailbox alloc failed L%d\n", layer);
        return -1;
    }
    memset((void *)c->mb_host, 0, sizeof(QfRouteMailbox));
    if (hipHostGetDevicePointer((void **)&c->mb_dev, c->mb_host, 0) != hipSuccess)
        return -1;
    c->expected_seq = 0;
    return 0;
}

void qf4_expcache_free(Qf4ExpCache *c) {
    if (!c) return;
    if (c->w_gate) hipFree(c->w_gate);
    if (c->w_up)   hipFree(c->w_up);
    if (c->w_down) hipFree(c->w_down);
    if (c->s_gate) hipFree(c->s_gate);
    if (c->s_up)   hipFree(c->s_up);
    if (c->s_down) hipFree(c->s_down);
    // wave4 session 09: device-resident routing state + mapped mailbox
    if (c->slot_for_expert_dev) hipFree(c->slot_for_expert_dev);
    if (c->expert_in_slot_dev)  hipFree(c->expert_in_slot_dev);
    if (c->age_dev)             hipFree(c->age_dev);
    if (c->clock_dev)           hipFree(c->clock_dev);
    if (c->route_slot_dev)      hipFree(c->route_slot_dev);
    if (c->s2g_dev) hipFree(c->s2g_dev);
    if (c->s2u_dev) hipFree(c->s2u_dev);
    if (c->s2d_dev) hipFree(c->s2d_dev);
    if (c->mb_host) hipHostFree(c->mb_host);
    free(c->slot_for_expert); free(c->expert_in_slot); free(c->age);
    free(c->s2g); free(c->s2u); free(c->s2d);
    memset(c, 0, sizeof(*c));
}

// Locate the six store entries (packed weight + block scales for
// gate/up/down) of one routed expert. Shared by qf4_expcache_prepare and the
// wave4 device-dispatch miss service.
static int find_expert_entries(const QfStore *st, int layer, int expert,
                               const QfEntry *we[3], const QfEntry *se[3]) {
    for (int p = 0; p < 3; p++) {
        char nm[QF_MAX_NAME];
        snprintf(nm, sizeof(nm),
                 "model.language_model.layers.%d.mlp.experts.%d.%s.weight",
                 layer, expert, kProj[p]);
        we[p] = qf_find(st, nm);
        snprintf(nm, sizeof(nm),
                 "model.language_model.layers.%d.mlp.experts.%d.%s.weight_scale",
                 layer, expert, kProj[p]);
        se[p] = qf_find(st, nm);
        if (!we[p] || !se[p] || we[p]->rec.nbytes != QF5_EXP_W_BYTES ||
            se[p]->rec.nbytes != QF5_EXP_S_BYTES) {
            fprintf(stderr, "qf5: invalid routed expert tensor L%d E%d %s\n",
                    layer, expert, kProj[p]);
            return -1;
        }
    }
    return 0;
}

// Gather expert `expert` out of the mmap'd store into the current pinned
// staging buffer (packed [w_gate|w_up|w_down|s_gate|s_up|s_down]; waits on
// that buffer's previous transfer first), hand it to the transfer stream,
// record the buffer event, and make the compute stream wait on it. The host
// gather overlaps whatever the compute stream is still running.
// FP8 production path: pack the NVFP4 shard into the lane-chunk E4M3 layout on
// the host and upload once. Load-time cost, paid back on every decode token.
static int xfer_expert_fp8(Qf4ExpCache *c, const QfStore *st, int expert, int slot) {
    const QfEntry *we[3], *se[3];
    if (find_expert_entries(st, c->layer, expert, we, se) != 0) return -1;
    static uint8_t *w8 = NULL; static float *sc = NULL;
    const size_t wmax = (size_t)QF5_NEMBD * QF5_NFF, smax = (size_t)QF5_NEMBD * 64;
    if (!w8) { w8 = (uint8_t *)malloc(wmax); sc = (float *)malloc(smax * sizeof(float)); }
    if (!w8 || !sc) return -1;
    const float s2[3] = { c->s2g[expert], c->s2u[expert], c->s2d[expert] };
    uint8_t *dstw[3] = { c->w8_gate, c->w8_up, c->w8_down };
    float   *dsts[3] = { c->sc_gate, c->sc_up, c->sc_down };
    for (int p = 0; p < 3; p++) {
        const int rows = (p == 2) ? QF5_NEMBD : QF5_NFF;
        const int in   = (p == 2) ? QF5_NFF   : QF5_NEMBD;
        const uint8_t *nw = (const uint8_t *)st->shard_maps[we[p]->rec.file_idx] + we[p]->rec.data_off;
        const uint8_t *ns = (const uint8_t *)st->shard_maps[se[p]->rec.file_idx] + se[p]->rec.data_off;
        if (qf8_fp8_pack_expert(nw, ns, s2[p], rows, in, w8, sc) != 0) return -1;
        if (hipMemcpy(dstw[p] + (size_t)slot * rows * in, w8, (size_t)rows * in,
                      hipMemcpyHostToDevice) != hipSuccess) return -1;
        if (hipMemcpy(dsts[p] + (size_t)slot * rows * 64, sc, (size_t)rows * 64 * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) return -1;
    }
    return 0;
}

static int xfer_expert(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                       int expert, int slot, hipStream_t compute) {
    if (c->w8_gate) { (void)xf; (void)compute; return xfer_expert_fp8(c, st, expert, slot); }
    const QfEntry *we[3], *se[3];
    if (find_expert_entries(st, c->layer, expert, we, se) != 0) return -1;

    int b = xf->cur;
    xf->cur ^= 1;
    if (hipEventSynchronize(xf->ev[b]) != hipSuccess) return -1;
    char *dst = (char *)xf->buf[b];
    for (int p = 0; p < 3; p++) {
        memcpy(dst + (size_t)p * QF5_EXP_W_BYTES,
               (const char *)st->shard_maps[we[p]->rec.file_idx] + we[p]->rec.data_off,
               QF5_EXP_W_BYTES);
        memcpy(dst + 3 * QF5_EXP_W_BYTES + (size_t)p * QF5_EXP_S_BYTES,
               (const char *)st->shard_maps[se[p]->rec.file_idx] + se[p]->rec.data_off,
               QF5_EXP_S_BYTES);
    }

    uint8_t *wdst[3] = {c->w_gate, c->w_up, c->w_down};
    uint8_t *sdst[3] = {c->s_gate, c->s_up, c->s_down};
    const int dst_slot = slot;
    if (c->nv_ring) slot = expert % c->nv_ring;           // NVFP4 lands in the ring slot
    for (int p = 0; p < 3; p++) {
        if (hipMemcpyAsync(wdst[p] + (size_t)slot * QF5_EXP_W_BYTES,
                           dst + (size_t)p * QF5_EXP_W_BYTES,
                           QF5_EXP_W_BYTES, hipMemcpyHostToDevice, xf->xfer) != hipSuccess ||
            hipMemcpyAsync(sdst[p] + (size_t)slot * QF5_EXP_S_BYTES,
                           dst + 3 * QF5_EXP_W_BYTES + (size_t)p * QF5_EXP_S_BYTES,
                           QF5_EXP_S_BYTES, hipMemcpyHostToDevice, xf->xfer) != hipSuccess) {
            fprintf(stderr, "qf5: expert upload failed L%d E%d\n", c->layer, expert);
            return -1;
        }
    }
    if (hipEventRecord(xf->ev[b], xf->xfer) != hipSuccess) return -1;
    // The compute stream is ordered after the slot fill without any host-side
    // synchronization; the kernels launched later this token read the slot.
    if (hipStreamWaitEvent(compute, xf->ev[b], 0) != hipSuccess) return -1;
    if (c->nv_ring && c->i8_gate) {
        // int8-only residency: convert this expert out of the ring slot now, and
        // drain so the ring slot can be refilled (init-time only).
        if (qf8_i8_build_slot(c, slot, dst_slot, expert, compute) != 0) return -1;
        if (hipStreamSynchronize(compute) != hipSuccess) return -1;
    }
    return 0;
}

// ---- static residency: give every expert a permanent address -------------
//
// When the cache has a slot for every expert, the mapping is the IDENTITY and
// is known at load time: expert e lives in slot e, forever. Nothing can evict
// it, so there is no LRU to maintain, no miss to service, no mailbox to stamp
// and no host spin per layer per token. `route_slot` is just `sel_ids`.
//
// This is what the measured 29% `exp_svc` cost actually was: the CPU deciding,
// every layer of every token, which expert goes where - a decision that has
// only one possible answer once everything is resident. Prefilling here trades
// a one-off upload at init for zero host work in the decode path.
//
// Returns 1 if the cache is now fully resident, 0 if it is not eligible.
int qf4_expcache_prefill(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                         hipStream_t compute) {
    if (!c || c->slots < QF5_NEXP) return 0;
    for (int e = 0; e < QF5_NEXP; e++) {
        if (xfer_expert(c, st, xf, e, e, compute) != 0) {
            fprintf(stderr, "qf5: prefill failed L%d E%d\n", c->layer, e);
            return -1;
        }
        c->slot_for_expert[e] = e;
        c->expert_in_slot[e]  = e;
        c->age[e] = 1;
    }
    if (hipMemcpy(c->slot_for_expert_dev, c->slot_for_expert,
                  QF5_NEXP * sizeof(int), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(c->expert_in_slot_dev, c->expert_in_slot,
                  (size_t)c->slots * sizeof(int), hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    c->fully_resident = 1;
    return 1;
}

int qf4_expcache_prepare(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                         int expert, hipStream_t compute) {
    if (!c || expert < 0 || expert >= QF5_NEXP || c->slots <= 0) return -1;

    int slot = c->slot_for_expert[expert];
    if (slot >= 0) {
        c->age[slot] = ++c->clock;
        return slot;
    }

    // Free slot, else LRU victim.
    slot = 0;
    for (int i = 0; i < c->slots; i++) {
        if (c->expert_in_slot[i] < 0) { slot = i; break; }
        if (c->age[i] < c->age[slot]) slot = i;
    }

    if (xfer_expert(c, st, xf, expert, slot, compute) != 0) return -1;

    int evicted = c->expert_in_slot[slot];
    if (evicted >= 0) c->slot_for_expert[evicted] = -1;
    c->expert_in_slot[slot]  = expert;
    c->slot_for_expert[expert] = slot;
    c->age[slot] = ++c->clock;
    return slot;
}

// ---------------------------------------------------------------------------
// wave4 session 09: host side of the device-resident dispatch protocol.
// qf_rtr_dispatch (qf_router_wave64.hip) owns all cache metadata on device:
// on a miss it picks the victim slot, updates the device maps/ages, and
// records (expert, slot) in the pinned QfRouteMailbox, writing miss_n and
// then seq (system-fenced, last). This function spins on the mailbox seq
// (volatile read of zero-copy mapped memory -- the one sanctioned
// host<->device interaction per layer, NOT a memcpy readback), then acts as
// a pure DMA engine: gather -> pinned double buffer -> H2D on the transfer
// stream -> compute stream waits the buffer event. The host slot maps are
// NOT updated here; during decode the device maps are the ground truth.
// ---------------------------------------------------------------------------
// ---- miss accounting (always on; a counter costs nothing) -----------------
// The per-layer expert cache holds QF_EXPERT_CACHE_SLOTS of 512 experts. Every
// miss is a host-side gather out of the mmap'd store plus an H2D of the whole
// expert (gate+up+down, NVFP4 + scales). That traffic crosses PCIe on EVERY
// token it is not resident for, which is a completely different bottleneck
// from HBM bandwidth and is invisible to a bytes-from-HBM roofline.
static unsigned long long g_exp_calls = 0, g_exp_miss = 0, g_exp_bytes = 0;

void qf4_expcache_stats(unsigned long long *calls, unsigned long long *miss,
                        unsigned long long *bytes) {
    if (calls) *calls = g_exp_calls;
    if (miss)  *miss  = g_exp_miss;
    if (bytes) *bytes = g_exp_bytes;
}

int qf4_expcache_service(Qf4ExpCache *c, const QfStore *st, Qf4ExpXfer *xf,
                         hipStream_t compute) {
    if (!c || !c->mb_host) return -1;
    QfRouteMailbox *mb = c->mb_host;
    uint32_t want = c->expected_seq;

    // Zero-copy spin: volatile read of pinned memory while the compute stream
    // keeps executing the work already enqueued for this token. Bounded as a
    // dead-GPU guard instead of hanging forever.
    uint64_t spins = 0;
    while (mb->seq != want) {
        if (++spins > (1ull << 34)) {
            fprintf(stderr, "qf5: mailbox timeout L%d (want seq %u, got %u)\n",
                    c->layer, want, (unsigned)mb->seq);
            return -1;
        }
    }

    uint32_t nmiss = mb->miss_n;
    if (nmiss > QF5_NEXPUSED) nmiss = QF5_NEXPUSED;   // corrupt-mailbox guard
    g_exp_calls += QF5_NEXPUSED;
    g_exp_miss  += nmiss;
    g_exp_bytes += (unsigned long long)nmiss * QF5_EXP_SLOT_BYTES;
    for (uint32_t i = 0; i < nmiss; i++) {
        int expert = mb->miss_expert[i];
        int slot   = mb->miss_slot[i];
        if (expert < 0 || expert >= QF5_NEXP || slot < 0 || slot >= c->slots) {
            fprintf(stderr, "qf5: route mailbox corrupt L%d: expert %d slot %d\n",
                    c->layer, expert, slot);
            mb->miss_n = 0;
            return -1;
        }
        if (xfer_expert(c, st, xf, expert, slot, compute) != 0) {
            mb->miss_n = 0;
            return -1;
        }
    }
    mb->miss_n = 0;   // consumed; the next dispatch round may stamp the mailbox
    return 0;
}
