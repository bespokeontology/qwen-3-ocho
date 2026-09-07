// qf_moe_4card_boot.cpp - AMD-side boot + serve wiring for the M=8 routed tier.
// Host check: g++ -DQF_HOST_CHECK -I. -fsyntax-only qf_moe_4card_boot.cpp
//
// Assembles the four-card pool from per-card resident NVFP4 bases and exposes
// one call to drop into a QfWireHdr serve loop's op switch:
//
//   else if (h.op == QFW_EXPERT_M8) {
//       if (qf_m8_serve(fd, &h)) break;      // dead-arm rule
//   }
//
// Residency contract (the split loader fills this): for card c the six base
// pointers address that card's 128 resident experts (local slot 0..127 ==
// global expert 128c..128c+127), allocated ON device c. s2{g,u,d}_all are the
// host [512] weight_scale_2 folds; the pool uploads each card's 128-slice.
#include "qf_moe_handoff.h"     // pool + serve handler + wire
#include "qf_m8_boot.h"
#include <stdio.h>

static Qf5MoePool *g_m8_pool = 0;
static int         g_m8_wave = 0;

// Per-card resident NVFP4 bases (device pointers on card c). The existing
// full-resident loader, run per device with an expert-range, produces these.

// Build the pool once at startup. Returns 0 on success.
extern "C" int qf_m8_boot(const Qf5CardBases bases[QF5_NCARD]) {
    Qf5Card cards[QF5_NCARD];
    for (int c = 0; c < QF5_NCARD; c++) {
        cards[c].dev = bases[c].dev;
        cards[c].comp = 0;               // pool creates per-wave streams
        cards[c].Wg = bases[c].Wg; cards[c].Sg = bases[c].Sg;
        cards[c].Wu = bases[c].Wu; cards[c].Su = bases[c].Su;
        cards[c].Wd = bases[c].Wd; cards[c].Sd = bases[c].Sd;
        cards[c].s2g = bases[c].s2g; cards[c].s2u = bases[c].s2u; cards[c].s2d = bases[c].s2d;
    }
    g_m8_pool = qf5_moe_pool_init(cards, /*M=*/8, QF5_HANDOFF_WAVES);
    if (!g_m8_pool) { fprintf(stderr, "qf_m8_boot: pool init failed\n"); return -1; }
    fprintf(stderr, "qf_m8_boot: four-card M=8 routed tier ready (experts %d/card)\n", QF5_EPC);
    return 0;
}

// Serve-loop hook: run one M=8 batch, rotate the wave ring.
extern "C" int qf_m8_serve(int fd, const QfWireHdr *h) {
    if (!g_m8_pool) { fprintf(stderr, "qf_m8_serve: not booted\n"); return -1; }
    int rc = qf_amd_routed_serve(fd, g_m8_pool, g_m8_wave, h);
    g_m8_wave = (g_m8_wave + 1) % QF5_HANDOFF_WAVES;
    return rc;
}

// Deferred receive: a small ring of admitted-but-not-drained waves. Capacity is
// exactly QF5_HANDOFF_WAVES because each slot serializes on its own streams.
static int g_admit_wave = 0, g_drain_wave = 0;
extern "C" int qf_m8_admit(int fd, const QfWireHdr *h, int *M_out) {
    if (!g_m8_pool) return -1;
    int il = 0;
    int rc = qf_amd_routed_admit(fd, g_m8_pool, g_admit_wave, h, M_out, &il);
    if (rc == 0) g_admit_wave = (g_admit_wave + 1) % QF5_HANDOFF_WAVES;
    return rc;
}
extern "C" int qf_m8_drain(int fd, int M, const QfWireHdr *h) {
    if (!g_m8_pool) return -1;
    int rc = qf_amd_routed_drain(fd, g_m8_pool, g_drain_wave, M, h);
    g_drain_wave = (g_drain_wave + 1) % QF5_HANDOFF_WAVES;
    return rc;
}

extern "C" void qf_m8_shutdown(void) { if (g_m8_pool) { qf5_moe_pool_free(g_m8_pool); g_m8_pool = 0; } }
