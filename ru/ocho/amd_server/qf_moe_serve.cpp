// qf_moe_serve.cpp - AMD-side serve handler for QFW_EXPERT_M8. Drives the pool.
// Host check: g++ -DQF_HOST_CHECK -I. -fsyntax-only qf_moe_serve.cpp
#include "qf_moe_handoff.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>

static_assert(QF5_NEMBD == QF_ROUTED_NEMBD, "wire NEMBD must match kernel NEMBD");
#define QF5_HANDOFF_MAXM 64
#define QF5_HANDOFF_MAXK QF5_NEXPUSED

static int io_all(int fd, void *p, size_t n, int wr) {
    char *c = (char *)p;
    while (n) {
        long k = wr ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, MSG_WAITALL);
        if (k <= 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}
// ---- deferred receive (GLM mechanism) --------------------------------------
// admit: read the payload, LAUNCH the wave asynchronously on this slot's
// per-card streams, and return immediately so the serve loop can read the next
// header. No synchronization here.
extern "C" int qf_amd_routed_admit(int fd, Qf5MoePool *pool, int wave, const QfWireHdr *h,
                                   int *M_out, int *il_out) {
    static int   sel[QF5_HANDOFF_MAXM * QF5_HANDOFF_MAXK];
    static float wt [QF5_HANDOFF_MAXM * QF5_HANDOFF_MAXK];
    static float x  [QF5_HANDOFF_MAXM * QF5_NEMBD];
    unsigned uM = 0, uK = 0;
    if (h->nbytes < (int)(2 * sizeof(unsigned))) return -1;
    if (io_all(fd, &uM, sizeof uM, 0) || io_all(fd, &uK, sizeof uK, 0)) return -1;
    int M = (int)uM, K = (int)uK;
    if (M < 1 || M > QF5_HANDOFF_MAXM || K < 1 || K > QF5_HANDOFF_MAXK) return -1;
    size_t sel_b = (size_t)M * K * sizeof(int), wt_b = (size_t)M * K * sizeof(float);
    size_t x_b = (size_t)M * QF5_NEMBD * sizeof(float);
    if (h->nbytes != (int)(2 * sizeof(unsigned) + sel_b + wt_b + x_b)) return -1;
    if (io_all(fd, sel, sel_b, 0) || io_all(fd, wt, wt_b, 0) || io_all(fd, x, x_b, 0)) return -1;
    qf5_moe_m8_submit(pool, wave, h->token, x, sel, wt, K);   // async: no wait
    *M_out = M; *il_out = h->token;
    return 0;
}
// drain: synchronize this slot, reduce, and reply. Called in ARRIVAL ORDER.
extern "C" int qf_amd_routed_drain(int fd, Qf5MoePool *pool, int wave, int M, const QfWireHdr *h) {
    static float y[QF5_HANDOFF_MAXM * QF5_NEMBD];
    qf5_moe_m8_wait(pool, wave, y);
    QfWireHdr r = *h;
    r.nbytes = (int)((size_t)M * QF5_NEMBD * sizeof(float));
    if (io_all(fd, &r, sizeof r, 1)) return -1;
    return io_all(fd, y, (size_t)M * QF5_NEMBD * sizeof(float), 1);
}

extern "C" int qf_amd_routed_serve(int fd, Qf5MoePool *pool, int wave, const QfWireHdr *h) {
    // Pinned handoff buffers: pageable H2D of the activation to four cards was
    // slow/serialized. Pin once so the per-card broadcasts are truly async.
    static int   *sel = 0; static float *wt = 0, *x = 0, *y = 0;
    if (!sel) {
        hipHostMalloc((void **)&sel, (size_t)QF5_HANDOFF_MAXM * QF5_HANDOFF_MAXK * sizeof(int), 0);
        hipHostMalloc((void **)&wt,  (size_t)QF5_HANDOFF_MAXM * QF5_HANDOFF_MAXK * sizeof(float), 0);
        hipHostMalloc((void **)&x,   (size_t)QF5_HANDOFF_MAXM * QF5_NEMBD * sizeof(float), 0);
        hipHostMalloc((void **)&y,   (size_t)QF5_HANDOFF_MAXM * QF5_NEMBD * sizeof(float), 0);
    }
    unsigned uM = 0, uK = 0;
    if (h->nbytes < (int)(2 * sizeof(unsigned))) { fprintf(stderr, "serve: M8 short payload\n"); return -1; }
    if (io_all(fd, &uM, sizeof uM, 0) || io_all(fd, &uK, sizeof uK, 0)) return -1;
    int M = (int)uM, K = (int)uK;
    if (M < 1 || M > QF5_HANDOFF_MAXM || K < 1 || K > QF5_HANDOFF_MAXK) {
        fprintf(stderr, "serve: M8 bad M=%d K=%d\n", M, K); return -1;
    }
    size_t sel_b = (size_t)M * K * sizeof(int), wt_b = (size_t)M * K * sizeof(float);
    size_t x_b = (size_t)M * QF5_NEMBD * sizeof(float);
    if (h->nbytes != (int)(2 * sizeof(unsigned) + sel_b + wt_b + x_b)) {
        fprintf(stderr, "serve: M8 payload mismatch %d\n", h->nbytes); return -1;
    }
    if (io_all(fd, sel, sel_b, 0) || io_all(fd, wt, wt_b, 0) || io_all(fd, x, x_b, 0)) return -1;
    qf5_moe_m8_submit(pool, wave, h->token, x, sel, wt, K);
    qf5_moe_m8_wait(pool, wave, y);
    QfWireHdr r = *h; r.nbytes = (int)x_b;
    if (io_all(fd, &r, sizeof r, 1)) return -1;
    return io_all(fd, y, (size_t)M * QF5_NEMBD * sizeof(float), 1);
}
