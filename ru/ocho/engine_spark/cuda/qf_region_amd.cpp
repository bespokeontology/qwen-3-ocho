#include "qf_region_amd.h"
#include "qf_moe_wire.h"
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
static int g_state = -1, g_fd = -1;
static void ensure(void) {
    if (g_state >= 0) return;
    const char *on = getenv("QF_REGION_AMD");
    if (!on || on[0] != '1') { g_state = 0; return; }
    const char *host = getenv("QF_REGION_AMD_HOST"); if (!host) host = "127.0.0.1";
    int port = 5578; if (const char *p = getenv("QF_REGION_AMD_PORT")) port = atoi(p);
    g_fd = qf_amd_client_connect(host, port);
    if (g_fd < 0) { fprintf(stderr, "[region-amd] connect %s:%d failed\n", host, port); g_state = 0; return; }
    fprintf(stderr, "[region-amd] AMD owns a row region via %s:%d\n", host, port);
    g_state = 1;
}
extern "C" int qf_region_amd_enabled(void) { ensure(); return g_state == 1; }
extern "C" int qf_amd_region_fork_fd(int fd, int M, long long P);
extern "C" int qf_region_amd_fork(int M, long P) {           // copy AMD row 0's state into rows 1..M-1 (all stages)
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_region_fork_fd(g_fd, M, (long long)P)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_region_commit_fd(int fd, int r, long long P);
extern "C" int qf_region_amd_commit(int r, long P) {           // copy AMD row r's state into row 0 (decision/commit loop)
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_region_commit_fd(g_fd, r, (long long)P)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_reset(void) {
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_region_reset(g_fd)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_region_submit_resid(long long pos, int M, const long *posM,
                                          const float *hR, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    extern int qf_amd_region_submit_resid_fd(int, long long, int, const long *, const float *, int);
    if (qf_amd_region_submit_resid_fd(g_fd, pos, M, posM, hR, rsize)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_region_submit_resid_fd(int fd, long long pos, int M, const long *posM,
                                             const float *hR, int rsize);
extern "C" int qf_region_amd_submit_resid(long long pos, int M, const long *posM,
                                          const float *hR, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_region_submit_resid_fd(g_fd, pos, M, posM, hR, rsize)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_prefix_fd(int fd, int M, const int *tokens, const long *posM, float *rout, int rsize);
extern "C" int qf_amd_head_fd(int fd, int M, const float *rin, int rsize, int *tokens_out);
extern "C" int qf_region_amd_prefix(int M, const int *tokens, const long *posM, float *rout, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_prefix_fd(g_fd, M, tokens, posM, rout, rsize)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_prefix_rows_fd(int fd, int r0, int nr, const int *tokens, const long *posM, float *rout, int rsize);
extern "C" int qf_region_amd_prefix_rows(int r0, int nr, const int *tokens, const long *posM, float *rout, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_prefix_rows_fd(g_fd, r0, nr, tokens, posM, rout, rsize)) { g_state = 0; return -1; }
    return 0;
}
// ---- W2: asynchronous prefix/head with an ORDERED ring of outstanding requests.
// The socket delivers replies FIFO, so a request registers its destination
// buffer at submit and qf_region_amd_drain_one() delivers the OLDEST reply into
// it; the caller keeps per-slot ready flags and drains until the one it needs
// has landed. (g10_overlap contract: one outstanding request per slot, ordered
// ring, slot id echoed and validated on receive.)
extern "C" int qf_amd_prefix_submit_fd(int fd, int r0, int nr, const int *tokens, const long *posM);
extern "C" int qf_amd_prefix_wait_fd(int fd, int r0, int nr, float *rout, int rsize);
extern "C" int qf_amd_head_submit_fd(int fd, int slot, int M, const float *rin, int rsize);
extern "C" int qf_amd_head_wait_fd(int fd, int slot, int M, int *tokens_out);
#define QF_REGION_PEND 8
typedef struct { int op, slot, r0, nr, rsize; float *rout; int *ids; } QfRegionPend;
static QfRegionPend g_pend[QF_REGION_PEND];
static int g_pend_head = 0, g_pend_n = 0;
static int pend_push(int op, int slot, int r0, int nr, int rsize, float *rout, int *ids) {
    if (g_pend_n >= QF_REGION_PEND) return -1;
    QfRegionPend *p = &g_pend[(g_pend_head + g_pend_n) % QF_REGION_PEND];
    p->op = op; p->slot = slot; p->r0 = r0; p->nr = nr; p->rsize = rsize; p->rout = rout; p->ids = ids;
    g_pend_n++;
    return 0;
}
extern "C" int qf_region_amd_outstanding(void) { return g_pend_n; }
extern "C" int qf_region_amd_prefix_submit(int slot, int r0, int nr, const int *tokens, const long *posM,
                                           float *rout, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    if (pend_push(QFW_PREFIX_M, slot, r0, nr, rsize, rout, NULL)) return -1;
    if (qf_amd_prefix_submit_fd(g_fd, r0, nr, tokens, posM)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_head_submit(int slot, int nr, const float *rin, int rsize, int *ids_out) {
    if (!qf_region_amd_enabled()) return -1;
    if (pend_push(QFW_HEAD_M, slot, slot, nr, rsize, NULL, ids_out)) return -1;
    if (qf_amd_head_submit_fd(g_fd, slot, nr, rin, rsize)) { g_state = 0; return -1; }
    return 0;
}
// Blocks for the oldest outstanding reply and delivers it. Returns its op and
// slot so the caller can flip the matching ready flag; -1 on wire error.
extern "C" int qf_region_amd_drain_one(int *op_out, int *slot_out) {
    if (g_state != 1 || g_pend_n < 1) return -1;
    QfRegionPend p = g_pend[g_pend_head];
    g_pend_head = (g_pend_head + 1) % QF_REGION_PEND; g_pend_n--;
    int rc = (p.op == QFW_PREFIX_M) ? qf_amd_prefix_wait_fd(g_fd, p.r0, p.nr, p.rout, p.rsize)
                          : qf_amd_head_wait_fd(g_fd, p.slot, p.nr, p.ids);
    if (rc) { g_state = 0; return -1; }
    if (op_out) *op_out = p.op;
    if (slot_out) *slot_out = p.slot;
    return 0;
}
extern "C" int qf_amd_headprefix_fd(int fd, const float *rin, int rsize, long long pos_next, int *tok_out, float *rout);
extern "C" int qf_region_amd_headprefix(const float *rin, int rsize, long pos_next, int *tok_out, float *rout) {
    if (g_state != 1) return -1;
    if (qf_amd_headprefix_fd(g_fd, rin, rsize, (long long)pos_next, tok_out, rout)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_amd_prefix_chunk_submit_fd(int fd, const int *tokens, int T, long long pos0);
extern "C" int qf_amd_prefix_chunk_wait_fd(int fd, int T, float *rout, int rsize);
// ---- asynchronous chunk replies: a receiver thread drains QFW_PREFIX_CHUNK replies
// into their destination buffers while the caller computes on Spark. Without it
// each 10 MB reply sat in the socket until the caller returned from its chunk,
// which stalled the AMD server's reply thread and therefore its 4-stage pipeline
// (8K prompt: 18 s = AMD ~9 s + Spark ~9 s, serialized). Replies are in request order.
#define QF_CP_RING 64
struct ChunkPend { int T; float *dst; int rsize; int err; };
static ChunkPend g_cp[QF_CP_RING];
static int g_cp_head = 0, g_cp_tail = 0, g_cp_cons = 0, g_rx_started = 0;
static pthread_mutex_t g_cm = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cc = PTHREAD_COND_INITIALIZER;
static void *qf_rx_thread(void *) {
    for (;;) {
        pthread_mutex_lock(&g_cm);
        while (g_cp_tail == g_cp_head) pthread_cond_wait(&g_cc, &g_cm);
        ChunkPend *cp = &g_cp[g_cp_tail % QF_CP_RING];
        pthread_mutex_unlock(&g_cm);
        const int err = qf_amd_prefix_chunk_wait_fd(g_fd, cp->T, cp->dst, cp->rsize);
        pthread_mutex_lock(&g_cm);
        cp->err = err; g_cp_tail++;
        pthread_cond_broadcast(&g_cc);
        pthread_mutex_unlock(&g_cm);
    }
    return NULL;
}
extern "C" int qf_region_amd_prefix_chunk_submit_to(const int *tokens, int T, long pos0, float *dst, int rsize) {
    if (!qf_region_amd_enabled()) return -1;
    pthread_mutex_lock(&g_cm);
    if (!g_rx_started) { pthread_t th; if (pthread_create(&th, NULL, qf_rx_thread, NULL) != 0) { pthread_mutex_unlock(&g_cm); return -1; } pthread_detach(th); g_rx_started = 1; }
    while (g_cp_head - g_cp_cons >= QF_CP_RING) pthread_cond_wait(&g_cc, &g_cm);
    ChunkPend *cp = &g_cp[g_cp_head % QF_CP_RING];
    cp->T = T; cp->dst = dst; cp->rsize = rsize; cp->err = 0;
    if (qf_amd_prefix_chunk_submit_fd(g_fd, tokens, T, (long long)pos0)) { pthread_mutex_unlock(&g_cm); g_state = 0; return -1; }
    g_cp_head++;
    pthread_cond_broadcast(&g_cc);
    pthread_mutex_unlock(&g_cm);
    return 0;
}
// Wait for the oldest submitted chunk reply (in submission order); its data is in the dst given at submit.
extern "C" int qf_region_amd_prefix_chunk_wait_done(void) {
    if (g_state != 1) return -1;
    pthread_mutex_lock(&g_cm);
    while (g_cp_tail <= g_cp_cons) pthread_cond_wait(&g_cc, &g_cm);
    const int err = g_cp[g_cp_cons % QF_CP_RING].err;
    g_cp_cons++;
    pthread_cond_broadcast(&g_cc);
    pthread_mutex_unlock(&g_cm);
    if (err) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_prefix_chunk_submit(const int *tokens, int T, long pos0) {
    if (!qf_region_amd_enabled()) return -1;
    if (qf_amd_prefix_chunk_submit_fd(g_fd, tokens, T, (long long)pos0)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_prefix_chunk_wait(int T, float *rout, int rsize) {
    if (g_state != 1) return -1;
    if (qf_amd_prefix_chunk_wait_fd(g_fd, T, rout, rsize)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_head(int M, const float *rin, int rsize, int *tokens_out) {
    if (g_state != 1) return -1;
    if (qf_amd_head_fd(g_fd, M, rin, rsize, tokens_out)) { g_state = 0; return -1; }
    return 0;
}
extern "C" int qf_region_amd_collect(int M, int *tokens_out) {
    if (g_state != 1) return -1;
    if (qf_amd_region_wait(g_fd, M, tokens_out)) { g_state = 0; return -1; }
    return 0;
}
