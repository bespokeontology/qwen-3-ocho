// qf_moe_4card.cpp - host orchestration of the four-MI50 routed-expert tier.
// Host syntax check:  g++ -DQF_HOST_CHECK -I. -fsyntax-only qf_moe_4card.cpp
//
// route once -> dispatch four cards independently -> each card produces its
// contribution -> concurrent completion/reduction -> handoff. Card0 hosts only
// the final tiny combine (M*NEMBD = 80 KB), never the per-card compute or an
// intermediate byte funnel: the three peer partials land in card0-local
// buffers concurrently on their SOURCE cards' streams, then one kernel sums.
#include "qf_moe_4card.h"
#ifndef QF5_MROWS_MAX
#define QF5_MROWS_MAX 8      // canonical M=8
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Sized to the ACTUAL execution shape: M rows x top-k. The old 64-token cap
// over-provisioned hidden/CSR by 8x on a card that is 99% full of experts.
#define NPAIRS_MAX (QF5_MROWS_MAX * QF5_NEXPUSED)

// Per (wave,card) device scratch + the CSR staged for the current batch.
typedef struct {
    hipStream_t s;
    hipEvent_t  done;             // compute (downacc) complete on this card
    hipEvent_t  copied;           // peer partial landed on card0 (cards 1..3)
    float *x_dev;                 // [M*NEMBD]
    float *hidden;                // [NPAIRS_MAX*NFF]
    float *y_partial;             // [M*NEMBD]
    int   *d_exp_slot, *d_exp_ptr, *d_pair_tok;
    int   *d_n_exp;                // device extent the kernels early-exit on
    float *d_pair_wt;
    Qf5M8Route rt;
} CardWave;

typedef struct {
    float *y_routed;              // [M*NEMBD] on card0 (final combine target)
    float *stage[QF5_NCARD - 1];  // [M*NEMBD] on card0, one per peer card
} WaveHead;

struct Qf5MoePool {
    int ncard, epc, M, nwaves;
    Qf5Card card[QF5_NCARD];
    CardWave *cw;                 // [nwaves*ncard], row-major [wave][card]
    WaveHead *head;               // [nwaves]
};

static inline CardWave *CW(Qf5MoePool *p, int w, int c) { return &p->cw[w * p->ncard + c]; }

static void dalloc(int dev, void **pp, size_t bytes) { hipSetDevice(dev); hipMalloc(pp, bytes); }

Qf5MoePool *qf5_moe_pool_init(const Qf5Card cards[QF5_NCARD], int M, int nwaves) {
    Qf5MoePool *p = (Qf5MoePool *)calloc(1, sizeof(Qf5MoePool));
    p->ncard = QF5_NCARD; p->epc = QF5_EPC; p->M = M; p->nwaves = nwaves;
    for (int c = 0; c < QF5_NCARD; c++) p->card[c] = cards[c];
    // Enable P2P from every peer card to card0 so partials can be peer-copied.
    for (int c = 1; c < QF5_NCARD; c++) {
        int can = 0;
        hipDeviceCanAccessPeer(&can, cards[0].dev, cards[c].dev);
        if (can) { hipSetDevice(cards[0].dev); hipDeviceEnablePeerAccess(cards[c].dev, 0); }
    }
    p->cw   = (CardWave *)calloc((size_t)nwaves * QF5_NCARD, sizeof(CardWave));
    p->head = (WaveHead *)calloc(nwaves, sizeof(WaveHead));
    for (int w = 0; w < nwaves; w++) {
        for (int c = 0; c < QF5_NCARD; c++) {
            CardWave *q = CW(p, w, c);
            hipSetDevice(cards[c].dev);
            hipStreamCreate(&q->s);
            hipEventCreateWithFlags(&q->done, hipEventDisableTiming);
            hipEventCreateWithFlags(&q->copied, hipEventDisableTiming);
            dalloc(cards[c].dev, (void **)&q->x_dev,      (size_t)M * QF5_NEMBD * sizeof(float));
            dalloc(cards[c].dev, (void **)&q->hidden,     (size_t)NPAIRS_MAX * QF5_NFF * sizeof(float));
            dalloc(cards[c].dev, (void **)&q->y_partial,  (size_t)M * QF5_NEMBD * sizeof(float));
            dalloc(cards[c].dev, (void **)&q->d_exp_slot, (size_t)NPAIRS_MAX * sizeof(int));
            dalloc(cards[c].dev, (void **)&q->d_exp_ptr,  (size_t)(NPAIRS_MAX + 1) * sizeof(int));
            dalloc(cards[c].dev, (void **)&q->d_pair_tok, (size_t)NPAIRS_MAX * sizeof(int));
            dalloc(cards[c].dev, (void **)&q->d_pair_wt,  (size_t)NPAIRS_MAX * sizeof(float));
            dalloc(cards[c].dev, (void **)&q->d_n_exp,    sizeof(int));
            q->rt.n_exp_dev = q->d_n_exp;
            q->rt.n_exp_max = NPAIRS_MAX;
        }
        hipSetDevice(cards[0].dev);
        dalloc(cards[0].dev, (void **)&p->head[w].y_routed, (size_t)M * QF5_NEMBD * sizeof(float));
        for (int c = 0; c < QF5_NCARD - 1; c++)
            dalloc(cards[0].dev, (void **)&p->head[w].stage[c], (size_t)M * QF5_NEMBD * sizeof(float));
    }
    return p;
}

// Invert the batch's top-k routing into card c's CSR (group-by-expert).
static int build_card_csr(const Qf5MoePool *p, int c, int il, const int *sel, const float *wt,
                          int M, int K, int *exp_slot, int *exp_ptr, int *pair_tok,
                          float *pair_wt, int *n_exp_out) {
    int te[NPAIRS_MAX], tt[NPAIRS_MAX]; float tw[NPAIRS_MAX];
    int np = 0;
    for (int t = 0; t < M; t++)
        for (int k = 0; k < K; k++) {
            int e = sel[t * K + k];
            if (e < 0 || e / p->epc != c) continue;
            te[np] = e; tt[np] = t; tw[np] = wt[t * K + k]; np++;
        }
    int exp_ids[NPAIRS_MAX], nexp = 0;
    for (int i = 0; i < np; i++) {
        int found = 0;
        for (int j = 0; j < nexp; j++) if (exp_ids[j] == te[i]) { found = 1; break; }
        if (!found) exp_ids[nexp++] = te[i];
    }
    exp_ptr[0] = 0;
    int pt = 0;
    for (int j = 0; j < nexp; j++) {
        int e = exp_ids[j];
        for (int i = 0; i < np; i++) if (te[i] == e) { pair_tok[pt] = tt[i]; pair_wt[pt] = tw[i]; pt++; }
        exp_ptr[j + 1] = pt;
        exp_slot[j] = il * QF5_EPC + (e - c * p->epc);   // (layer, local expert) slot
    }
    *n_exp_out = nexp;
    return np;
}

void qf5_moe_m8_submit(Qf5MoePool *p, int wave, int il, const float *x_host,
                       const int *sel, const float *wt, int K) {
    const int M = p->M;
    int   h_slot[NPAIRS_MAX], h_ptr[NPAIRS_MAX + 1], h_tok[NPAIRS_MAX];
    float h_wt[NPAIRS_MAX];
    // Dispatch: each card runs independently on its own device + stream.
    for (int c = 0; c < p->ncard; c++) {
        CardWave *q = CW(p, wave, c);
        int nexp = 0;
        int np = build_card_csr(p, c, il, sel, wt, M, K, h_slot, h_ptr, h_tok, h_wt, &nexp);
        q->rt.n_exp = nexp; q->rt.n_pairs = np; q->rt.M = M;
        hipMemcpyAsync(q->d_n_exp, &nexp, sizeof(int), hipMemcpyHostToDevice, q->s);
        hipSetDevice(p->card[c].dev);
        hipMemcpyAsync(q->x_dev, x_host, (size_t)M * QF5_NEMBD * sizeof(float), hipMemcpyHostToDevice, q->s);
        hipMemcpyAsync(q->d_exp_slot, h_slot, (size_t)(nexp > 0 ? nexp : 1) * sizeof(int), hipMemcpyHostToDevice, q->s);
        hipMemcpyAsync(q->d_exp_ptr,  h_ptr,  (size_t)(nexp + 1) * sizeof(int), hipMemcpyHostToDevice, q->s);
        hipMemcpyAsync(q->d_pair_tok, h_tok,  (size_t)(np > 0 ? np : 1) * sizeof(int), hipMemcpyHostToDevice, q->s);
        hipMemcpyAsync(q->d_pair_wt,  h_wt,   (size_t)(np > 0 ? np : 1) * sizeof(float), hipMemcpyHostToDevice, q->s);
        q->rt.exp_slot = q->d_exp_slot; q->rt.exp_ptr = q->d_exp_ptr;
        q->rt.pair_tok = q->d_pair_tok; q->rt.pair_wt = q->d_pair_wt;
        hipMemsetAsync(q->y_partial, 0, (size_t)M * QF5_NEMBD * sizeof(float), q->s);
        if (getenv("QF_DEBUG")) {
            fprintf(stderr, "[qfd] card %d il=%d nexp=%d np=%d slots=", c, il, nexp, np);
            for (int j = 0; j < nexp && j < 3; j++) fprintf(stderr, " %d", h_slot[j]);
            fprintf(stderr, "\n");
        }
        if (nexp > 0) {
            qf5_m8_gateup(p->card[c].Wg, p->card[c].Sg, p->card[c].Wu, p->card[c].Su,
                          p->card[c].s2g, p->card[c].s2u, q->x_dev, q->hidden, &q->rt, q->s);
            qf5_m8_downacc(p->card[c].Wd, p->card[c].Sd, p->card[c].s2d,
                           q->hidden, q->y_partial, &q->rt, q->s);
        }
        hipEventRecord(q->done, q->s);
        // gfx906 P2P silently moves no data on this hardware -> no on-device peer
        // reduce. Each card's y_partial stays resident; qf5_moe_m8_wait D2H-copies
        // all four and sums them on the host (partials are tiny: M*NEMBD).
    }
}

void qf5_moe_m8_wait(Qf5MoePool *p, int wave, float *y_routed_host) {
    int n = p->M * QF5_NEMBD;
    if (getenv("QF_DEBUG")) {
        for (int c = 0; c < p->ncard; c++) {
            float pr[8] = {0};
            hipMemcpy(pr, CW(p, wave, c)->y_partial, 8 * sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "[qfd] card %d partial[0..3]: %.4f %.4f %.4f %.4f\n", c, pr[0], pr[1], pr[2], pr[3]);
        }
    }
    // Four cards own disjoint expert ranges and compute concurrently. Issue all
    // four D2H copies of the tiny partials AT ONCE (pinned host buffers), then a
    // single barrier, then sum on host. No serial card-by-card synchronize.
    static float *hbuf[QF5_NCARD] = {0}; static float *acc = 0;
    if (!acc) {
        acc = (float *)malloc((size_t)n * sizeof(float));
        for (int c = 0; c < QF5_NCARD; c++) hipHostMalloc((void **)&hbuf[c], (size_t)n * sizeof(float), 0);
    }
    for (int c = 0; c < p->ncard; c++) {
        CardWave *q = CW(p, wave, c);
        hipSetDevice(p->card[c].dev);
        hipMemcpyAsync(hbuf[c], q->y_partial, (size_t)n * sizeof(float), hipMemcpyDeviceToHost, q->s);
    }
    for (int c = 0; c < p->ncard; c++) { hipSetDevice(p->card[c].dev); hipStreamSynchronize(CW(p, wave, c)->s); }
    for (int i = 0; i < n; i++) { float a = 0.f; for (int c = 0; c < p->ncard; c++) a += hbuf[c][i]; y_routed_host[i] = a; }
    if (getenv("QF_M8_DEBUG")) {
        double nn = 0; for (int i = 0; i < n; i++) nn += (double)y_routed_host[i] * y_routed_host[i];
        fprintf(stderr, "[amd] routed y L2=%.4f n=%d\n", sqrt(nn), n);
    }
}

void qf5_moe_pool_free(Qf5MoePool *p) {
    if (!p) return;
    for (int w = 0; w < p->nwaves; w++) {
        for (int c = 0; c < p->ncard; c++) {
            CardWave *q = CW(p, w, c);
            hipSetDevice(p->card[c].dev);
            hipFree(q->x_dev); hipFree(q->hidden); hipFree(q->y_partial);
            hipFree(q->d_exp_slot); hipFree(q->d_exp_ptr); hipFree(q->d_pair_tok); hipFree(q->d_pair_wt);
            hipStreamDestroy(q->s); hipEventDestroy(q->done); hipEventDestroy(q->copied);
        }
        hipSetDevice(p->card[0].dev);
        hipFree(p->head[w].y_routed);
        for (int c = 0; c < QF5_NCARD - 1; c++) hipFree(p->head[w].stage[c]);
    }
    free(p->cw); free(p->head); free(p);
}
