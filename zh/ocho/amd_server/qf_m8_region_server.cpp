// qf_m8_region_server.cpp - AMD dependency-closed region server.
// The four MI50s hold the WHOLE model (4 stages x 12 layers, existing pipeline
// residency). A region step runs the AMD-owned rows through ALL 48 layers on
// this box and returns sampled token ids: only tokens cross the wire, no
// per-layer traffic. This is the AMD half of  AMD region(A) || Spark region(B).
#include "qf_hip4.h"
#include "qf_moe_wire.h"
#ifdef QF_HOST_CHECK
#include "host_check_shim.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <stdio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#define NSTAGE 4
// Alternating map (5836a6d): AMD owns the PREFIX (embed + PLE + layers 0..S-1,
// S/4 layers per card) AND the head (output HC + lm_head + argmax on card 3).
// Spark runs the tail S..47. Two coarse crossings per token: the prefix
// residual out, the tail residual back in; the reply is M token ids, never
// logits. region_step_tail (QFW_REGION_M8) is the older whole-model-on-AMD
// form and is kept for the A||B region experiment.
static int g_S = 12;                     // AMD prefix depth: layers 0..S-1 (env QF_AMD_PREFIX)
static int g_lps = 3;                    // layers per stage = S/4
#define MAXM   32      // resident rows on the box (two 16-row slots, == QF8_MROWS)
#define MAXCHUNK 256   // prefill chunk positions per request (QF_CHUNK_ROWS)
#define MAXREQ 16      // rows per request (kernel row cap: QF8_FP8_MAXROW / QF5_M8_TX)
#define RS     QF4_RSIZE

static Qf4Stage *g_st[NSTAGE];
static QfStore   g_store;
static float    *g_hR;                   // [MAXM][RS] host handoff staging (pinned)

static float    *g_rin;                  // [MAXM][RS] wire residual in (pinned)

static inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16); uint32_t lsb = (u >> 16) & 1u; return (uint16_t)((u + 0x7fffu + lsb) >> 16); }
static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t g_wbuf[MAXCHUNK * RS];    // bf16 wire staging
static int io_all(int fd, void *p, size_t n, int wr);
// ---- pipelined prefill chunks: up to PIPE_SLOTS chunks in flight, one per stage.
// Stage g's stream waits (cross-device event) for stage g-1's D2H of the same
// chunk, so the four MI50s run DIFFERENT chunks concurrently; the reply thread
// answers in order once the last stage's D2H event completes. Slot buffers are
// reused only after their reply was sent (never on host call order).
#define PIPE_SLOTS 4
struct ChunkSlot { float *hop[NSTAGE]; hipEvent_t ev[NSTAGE]; int tk[MAXCHUNK]; int T; long pos0; QfWireHdr h; int bf; double t_submit; };
static ChunkSlot g_slot[PIPE_SLOTS];
static pthread_mutex_t g_pm = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_pc = PTHREAD_COND_INITIALIZER;
static int g_pipe_head = 0, g_pipe_tail = 0, g_pipe_stop = 0, g_reply_fail = 0, g_reply_fd = -1;
static int g_pipe_sub = 0;                        // chunks submitted to the pipeline (sequence numbers)
static int send_resid(int fd, const float *r, size_t n, int bf);
static double region_ms(void);
static void *pipe_reply_thread(void *) {
    for (;;) {
        pthread_mutex_lock(&g_pm);
        while (g_pipe_tail == g_pipe_head && !g_pipe_stop) pthread_cond_wait(&g_pc, &g_pm);
        if (g_pipe_tail == g_pipe_head && g_pipe_stop) { pthread_mutex_unlock(&g_pm); return NULL; }
        const int c = g_pipe_tail;
        pthread_mutex_unlock(&g_pm);
        ChunkSlot *sl = &g_slot[c % PIPE_SLOTS];
        if (hipEventSynchronize(sl->ev[NSTAGE - 1]) != hipSuccess) g_reply_fail = 1;
        if (!g_reply_fail) {
            QfWireHdr r = sl->h; r.op |= sl->bf ? QFW_BF16 : 0; r.nbytes = (int)((size_t)sl->T * RS * (sl->bf ? 2 : 4));
            if (io_all(g_reply_fd, &r, sizeof r, 1) || send_resid(g_reply_fd, sl->hop[NSTAGE - 1], (size_t)sl->T * RS, sl->bf)) g_reply_fail = 1;
            if (getenv("QF_REGION_TIMING")) fprintf(stderr, "[region] chunk #%d T=%d pos0=%ld done %.1f ms after submit\n", c, sl->T, sl->pos0, region_ms() - sl->t_submit);
        }
        pthread_mutex_lock(&g_pm); g_pipe_tail = c + 1; pthread_cond_broadcast(&g_pc); pthread_mutex_unlock(&g_pm);
    }
}
// One host thread per stage g >= 1: it blocks (hipEventSynchronize) only on stage
// g-1's D2H event of the same chunk, then enqueues stage g's H2D + compute + D2H on
// its own card. ROCm's cross-device hipStreamWaitEvent blocked the enqueueing host
// thread instead, which serialized the four stages of every chunk (AMD ran at
// ~1.9 ms/token on an 8K prompt while its stages measured ~0.45 ms/token).
static int g_sq[NSTAGE][PIPE_SLOTS * 2], g_sq_head[NSTAGE], g_sq_tail[NSTAGE];
static pthread_mutex_t g_sm[NSTAGE]; static pthread_cond_t g_sc[NSTAGE];
static int g_stage_fail = 0;
static void stage_push(int g, int c) {
    pthread_mutex_lock(&g_sm[g]);
    g_sq[g][g_sq_head[g] % (PIPE_SLOTS * 2)] = c; g_sq_head[g]++;
    pthread_cond_broadcast(&g_sc[g]);
    pthread_mutex_unlock(&g_sm[g]);
}
static int stage_pop(int g) {
    pthread_mutex_lock(&g_sm[g]);
    while (g_sq_tail[g] == g_sq_head[g]) pthread_cond_wait(&g_sc[g], &g_sm[g]);
    const int c = g_sq[g][g_sq_tail[g] % (PIPE_SLOTS * 2)]; g_sq_tail[g]++;
    pthread_mutex_unlock(&g_sm[g]);
    return c;
}
static void *stage_worker(void *arg) {
    const int g = (int)(intptr_t)arg;
    for (;;) {
        const int c = stage_pop(g);
        ChunkSlot *sl = &g_slot[c % PIPE_SLOTS];
        int ok = hipEventSynchronize(sl->ev[g - 1]) == hipSuccess;     // stage g-1 finished this chunk (its D2H landed in hop[g-1])
        if (ok && qf4_stage_push_rows_async(g_st[g], sl->hop[g - 1], sl->T) != 0) ok = 0;
        if (ok && qf4_stage_run_chunk(g_st[g], sl->tk, sl->pos0, sl->T) != 0) ok = 0;
        if (ok && (qf4_stage_pull_rows_async(g_st[g], sl->hop[g], sl->T) != 0 || qf4_stage_record(g_st[g], (void *)sl->ev[g]) != 0)) ok = 0;
        if (!ok) { g_stage_fail = 1; fprintf(stderr, "pipe: stage %d failed on chunk #%d\n", g, c); }
        if (g + 1 < NSTAGE) stage_push(g + 1, c);
        else { pthread_mutex_lock(&g_pm); g_pipe_head = c + 1; pthread_cond_broadcast(&g_pc); pthread_mutex_unlock(&g_pm); }
    }
    return NULL;
}
static int pipe_init(void) {
    for (int g = 0; g < NSTAGE; g++) { pthread_mutex_init(&g_sm[g], NULL); pthread_cond_init(&g_sc[g], NULL); g_sq_head[g] = g_sq_tail[g] = 0; }
    for (int g = 1; g < NSTAGE; g++) { pthread_t th; if (pthread_create(&th, NULL, stage_worker, (void *)(intptr_t)g) != 0) return -1; pthread_detach(th); }
    for (int k = 0; k < PIPE_SLOTS; k++) {
        for (int g = 0; g < NSTAGE; g++) {
            if (hipHostMalloc((void **)&g_slot[k].hop[g], (size_t)MAXCHUNK * RS * 4, hipHostMallocPortable) != hipSuccess) return -1;
            if (qf4_stage_event_create(g_st[g], (void **)&g_slot[k].ev[g]) != 0) return -1;
        }
    }
    return 0;
}
// Drain: wait until every submitted chunk has been replied (before decode ops / reset).
static void pipe_drain(void) {
    pthread_mutex_lock(&g_pm);
    while (g_pipe_tail != g_pipe_sub) pthread_cond_wait(&g_pc, &g_pm);
    pthread_mutex_unlock(&g_pm);
}
static int send_resid(int fd, const float *r, size_t n, int bf) {
    if (!bf) return io_all(fd, (void *)r, n * sizeof(float), 1);
    for (size_t i = 0; i < n; i++) g_wbuf[i] = f2bf(r[i]);
    return io_all(fd, g_wbuf, n * sizeof(uint16_t), 1);
}
static int recv_resid(int fd, float *r, size_t n, int bf) {
    if (!bf) return io_all(fd, r, n * sizeof(float), 0);
    if (io_all(fd, g_wbuf, n * sizeof(uint16_t), 0)) return -1;
    for (size_t i = 0; i < n; i++) r[i] = bf2f(g_wbuf[i]);
    return 0;
}
static int io_all(int fd, void *p, size_t n, int wr) {
    char *c = (char *)p;
    while (n) { long k = wr ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, MSG_WAITALL);
        if (k <= 0) return -1; c += k; n -= (size_t)k; }
    return 0;
}


// One decode step for M rows through all four stages.
// Tail region step: the residual arrives from Spark (one crossing), runs the
// AMD-owned layer run across the four cards, then the head + argmax here.
// AMD prefix: embed + layers 0..S-1 (PLE at layer 1) -> residual for the Spark tail.
// Rows [r0, r0+nr): tokens/rout are indexed 0..nr-1, posM by ABSOLUTE row.
// r0 > 0 is how one request row is primed through its own prompt while the
// other rows keep their state (each prompt has its own length).
static double region_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
static long g_nprefix = 0;
static void region_prof_print(const char *why) {
    for (int g = 0; g < NSTAGE; g++) {
        char lb[32]; snprintf(lb, sizeof lb, "stage g%d", g);
        qf4_stage_prof_print(g_st[g], stderr, lb);
    }
    fprintf(stderr, "[region] section profile above: %s (%ld prefix calls)\n", why, g_nprefix);
}
static int region_prefix(const int *tokens, const long *posM, int r0, int nr, float *rout) {
    static int time1 = -1;
    if (time1 < 0) time1 = getenv("QF_REGION_TIMING") ? 1 : 0;   // M=1 per-stage wall clock
    const int dbg = (nr > 1) || time1;
    g_nprefix++;
    const double t_enter = region_ms();
    if (dbg) fprintf(stderr, "[region] prefix enter r0=%d nr=%d posM0=%ld\n", r0, nr, posM ? posM[0] : -1);
    if (qf4_stage_set_tokens_rows(g_st[0], tokens, r0, nr) != 0) { fprintf(stderr, "[region] set_tokens_rows FAILED\n"); return -1; }
    if (dbg) fprintf(stderr, "[region] set_tokens_rows done %.1f ms\n", region_ms() - t_enter);
    for (int g = 0; g < NSTAGE; g++) {
        const double tg = region_ms();
        if (g > 0 && qf4_stage_push_residual_rows(g_st[g], g_hR, r0, nr) != 0) { fprintf(stderr, "[region] push rows g%d FAILED\n", g); return -1; }
        if (qf4_stage_run_M(g_st[g], posM, r0, nr) != 0) { fprintf(stderr, "[region] run_M g%d FAILED\n", g); return -1; }
        const double trun = region_ms();
        if (qf4_stage_pull_residual_rows(g_st[g], g + 1 < NSTAGE ? g_hR : rout, r0, nr) != 0) { fprintf(stderr, "[region] pull rows g%d FAILED\n", g); return -1; }
        if (dbg) fprintf(stderr, "[region] stage g%d run %.1f ms pull %.1f ms\n", g, trun - tg, region_ms() - trun);
    }
    if (dbg) fprintf(stderr, "[region] prefix total %.1f ms\n", region_ms() - t_enter);
    if (getenv("QF8_PROFILE") && (g_nprefix % 100) == 0) region_prof_print("periodic");
    return 0;
}
// AMD head: the Spark tail's residual -> output HC + lm_head + argmax -> token ids.
// The head is stateless, so it always uses transient rows 0..M-1 whichever
// slot the residual belongs to; the request header's pos (slot) is echoed.
static int region_head(const float *residual, int M, int *out) {
    Qf4Stage *H = g_st[NSTAGE - 1];
    if (qf4_stage_push_residual_M(H, residual, M) != 0) return -1;
    if (qf4_stage_head_M(H, M) != 0) return -1;
    return qf4_stage_head_ids_M(H, out, M);          // argmax on device, M ints back
}
static int region_step_tail(const float *residual, const long *posM, int M, int *out) {
    if (qf4_stage_push_residual_M(g_st[0], residual, M) != 0) return -1;
    for (int g = 0; g < NSTAGE; g++) {
        if (g > 0 && qf4_stage_push_residual_M(g_st[g], g_hR, M) != 0) return -1;
        if (qf4_stage_run_M(g_st[g], posM, 0, M) != 0) return -1;
        if (g + 1 < NSTAGE && qf4_stage_pull_residual_M(g_st[g], g_hR, M) != 0) return -1;
    }
        if (qf4_stage_head_M(g_st[NSTAGE - 1], M) != 0) return -1;
    return qf4_stage_head_ids_M(g_st[NSTAGE - 1], out, M);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : (getenv("QF_MODEL_DIR") ? getenv("QF_MODEL_DIR") : "./model");
    int port = argc > 2 ? atoi(argv[2]) : 5578;
    if (const char *e = getenv("QF_AMD_PREFIX")) { int v = atoi(e); if (v >= NSTAGE && v < 48) g_S = v; }
    g_S -= g_S % NSTAGE;                         // keep stage boundaries aligned
    g_lps = g_S / NSTAGE;
    setenv("QF_AMD_OWNS_HEAD", "1", 1);          // AMD owns the head as well
    setenv("QF_M1_INT8", "1", 0);                // M=1 int8 expert arenas (QF_M1_INT8=0 to disable)
    setenv("QF_CHUNK_ROWS", "256", 0);           // prefill chunk positions per request (QF_CHUNK_ROWS=0 off)
    // The stage layer count comes from QF_LAYERS_TOTAL/QF4_NGPU; for the tail cut
    // that is (48 - L0), and stage g owns [L0 + g*lps, +lps). owns_head fires on
    // the last stage because L0 + 4*lps == 48.
    char lt[32]; snprintf(lt, sizeof lt, "%d", g_S);
    setenv("QF_LAYERS_TOTAL", lt, 1);
    fprintf(stderr, "qf_m8_region_server: AMD owns PREFIX layers 0..%d (%d/stage) + PLE, AND the head (out HC + lm_head + argmax)\n",
            g_S - 1, g_lps);
    if (qf_store_open(&g_store, dir) != 0) { fprintf(stderr, "store open failed\n"); return 1; }
    for (int g = 0; g < NSTAGE; g++)
        if (qf4_stage_init(&g_st[g], g, &g_store, g * g_lps) != 0) {
            fprintf(stderr, "stage %d init failed\n", g); return 1; }
    // The M-row path addresses expert slots as slot == expert. Refuse a paged
    // layout at init instead of reading the wrong expert at decode.
    for (int g = 0; g < NSTAGE; g++)
        if (!qf4_stage_fully_resident(g_st[g])) {
            fprintf(stderr, "qf_m8_region_server: GPU %d experts are NOT fully resident; "
                            "the M-row region path requires slot==expert. Lower QF_AMD_PREFIX "
                            "or raise the HBM budget.\n", g);
            return 1;
        }
    // Pinned host staging for the inter-card bounce, the wire residual and the
    // logits: pageable buffers make every hipMemcpy a staged copy.
        if (hipHostMalloc((void **)&g_hR, (size_t)MAXCHUNK * RS * 4, hipHostMallocPortable) != hipSuccess ||
        hipHostMalloc((void **)&g_rin, (size_t)MAXCHUNK * RS * 4, hipHostMallocPortable) != hipSuccess) {
        fprintf(stderr, "qf_m8_region_server: pinned staging alloc failed\n"); return 1;
    }
    if (pipe_init() != 0) { fprintf(stderr, "pipe: init failed\n"); return 1; }
    { pthread_t th; if (pthread_create(&th, NULL, pipe_reply_thread, NULL) != 0) { fprintf(stderr, "pipe: thread failed\n"); return 1; } pthread_detach(th); }
    float *rin = g_rin;
    fprintf(stderr, "qf_m8_region_server: READY on %d (QFW_REGION_M8: residual in, token ids out)\n", port);

    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons((unsigned short)port);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 4)) { fprintf(stderr, "bind failed\n"); return 1; }
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fprintf(stderr, "region: peer connected\n");
        g_reply_fd = fd; g_reply_fail = 0;
        for (;;) {
            QfWireHdr h;
            if (io_all(fd, &h, sizeof h, 0)) break;
            if ((h.op & ~QFW_BF16) != QFW_PREFIX_CHUNK) pipe_drain();   // chunks in flight finish before any other op
            if (h.magic != QFW_MAGIC) break;
            const int bf = (h.op & QFW_BF16) ? 1 : 0; h.op &= ~QFW_BF16;   // bf16 wire residuals
            fprintf(stderr, "[region] op %u token=%d pos=%lld\n", h.op, h.token, (long long)h.pos);
            if (h.op == QFW_REGION_RESET) {
                for (int g = 0; g < NSTAGE; g++) qf4_stage_reset(g_st[g]); qf4_stage_head_calls_reset();
                if (io_all(fd, &h, sizeof h, 1)) break;
                continue;
            }
            if (h.op == QFW_REGION_FORK) {               // token = M, pos = P: row 0 -> rows 1..M-1 on every stage
                pipe_drain();
                int ok = 1;
                for (int g = 0; g < NSTAGE; g++) if (qf4_stage_fork_rows(g_st[g], h.token, (long)h.pos) != 0) { fprintf(stderr, "fork: stage %d failed\n", g); ok = 0; } qf4_stage_head_calls_reset();
                if (!ok) break;
                if (io_all(fd, &h, sizeof h, 1)) break;
                continue;
            }
            if (h.op == QFW_REGION_COMMIT) {              // token = source row r, pos = P: row r -> row 0 (decision/commit loop)
                pipe_drain();
                int ok = 1;
                for (int g = 0; g < NSTAGE; g++) if (qf4_stage_commit_rows(g_st[g], h.token, (long)h.pos) != 0) { fprintf(stderr, "commit: stage %d failed\n", g); ok = 0; }
                if (!ok) break;
                if (io_all(fd, &h, sizeof h, 1)) break;
                continue;
            }
                        if (h.op == QFW_PREFIX_M) {
                // h.token = nr rows, h.pos = row base r0 (0 for the decode step;
                // r0 = the row being primed during prompt priming).
                                int nr = h.token, r0 = (int)h.pos;
                if (nr < 1 || nr > MAXREQ || r0 < 0 || r0 + nr > MAXM) { fprintf(stderr, "prefix: bad rows r0=%d nr=%d\n", r0, nr); break; }
                int tk[MAXM]; long long p64[MAXM]; long posM[MAXM];
                if (io_all(fd, tk, (size_t)nr * sizeof(int), 0)) break;
                if (io_all(fd, p64, (size_t)nr * sizeof(long long), 0)) break;
                for (int i = 0; i < MAXM; i++) posM[i] = 0;
                for (int i = 0; i < nr; i++) posM[r0 + i] = (long)p64[i];
                if (region_prefix(tk, posM, r0, nr, rin) != 0) { fprintf(stderr, "prefix failed\n"); break; }
                QfWireHdr r = h; r.op |= bf ? QFW_BF16 : 0; r.nbytes = (int)((size_t)nr * RS * (bf ? 2 : 4));
                if (io_all(fd, &r, sizeof r, 1)) break;
                if (send_resid(fd, rin, (size_t)nr * RS, bf)) break;
                continue;
            }
                        if (h.op == QFW_HEAD_M) {
                int M = h.token; if (M < 1 || M > MAXREQ) break;   // h.pos = slot, echoed in the reply
                int out[MAXM];
                if (recv_resid(fd, rin, (size_t)M * RS, bf)) break;
                if (region_head(rin, M, out) != 0) { fprintf(stderr, "head failed\n"); break; }
                QfWireHdr r = h; r.nbytes = (int)((size_t)M * sizeof(int));
                if (io_all(fd, &r, sizeof r, 1)) break;
                if (io_all(fd, out, (size_t)M * sizeof(int), 1)) break;
                continue;
            }
            if (h.op == QFW_PREFIX_CHUNK) {
                // T consecutive prompt positions of row 0: tokens in, T residuals out.
                // PIPELINED: enqueue all four stages now (event-chained), reply later
                // from the reply thread; the next request is read immediately.
                const int T = h.token; const long pos0 = (long)h.pos;
                if (T < 1 || T > MAXCHUNK || T > qf4_chunk_rows()) { fprintf(stderr, "chunk: bad T=%d (QF_CHUNK_ROWS=%d)\n", T, qf4_chunk_rows()); break; }
                int tk[MAXCHUNK];
                if (io_all(fd, tk, (size_t)T * sizeof(int), 0)) break;
                if (g_reply_fail) { fprintf(stderr, "chunk: reply thread failed\n"); break; }
                pthread_mutex_lock(&g_pm);
                while (g_pipe_sub - g_pipe_tail >= PIPE_SLOTS) pthread_cond_wait(&g_pc, &g_pm);   // slot of chunk c-PIPE_SLOTS replied
                const int c = g_pipe_sub;
                pthread_mutex_unlock(&g_pm);
                ChunkSlot *sl = &g_slot[c % PIPE_SLOTS];
                memcpy(sl->tk, tk, (size_t)T * sizeof(int)); sl->T = T; sl->pos0 = pos0; sl->h = h; sl->bf = bf; sl->t_submit = region_ms();
                if (g_stage_fail) { fprintf(stderr, "chunk: a stage worker failed\n"); break; }
                int ok = 1;
                if (qf4_stage_set_chunk_tokens(g_st[0], sl->tk, T) != 0) { fprintf(stderr, "chunk: embed failed\n"); ok = 0; }
                if (ok && qf4_stage_run_chunk(g_st[0], sl->tk, pos0, T) != 0) ok = 0;
                if (ok && (qf4_stage_pull_rows_async(g_st[0], sl->hop[0], T) != 0 || qf4_stage_record(g_st[0], (void *)sl->ev[0]) != 0)) ok = 0;
                if (!ok) { fprintf(stderr, "chunk: stage 0 enqueue failed\n"); break; }
                g_pipe_sub = c + 1;                              // submitted; the last stage worker publishes completion
                stage_push(1, c);
                continue;
            }
            if (h.op == QFW_HEADPREFIX) {
                // M=1 fused step: head (token) + prefix of that token at h.pos, one round trip.
                int out[1]; long posM[MAXM]; int tk[1];
                if (recv_resid(fd, rin, (size_t)RS, bf)) break;
                if (region_head(rin, 1, out) != 0) { fprintf(stderr, "headprefix: head failed\n"); break; }
                tk[0] = out[0];
                for (int i = 0; i < MAXM; i++) posM[i] = 0;
                posM[0] = (long)h.pos;
                if (region_prefix(tk, posM, 0, 1, rin) != 0) { fprintf(stderr, "headprefix: prefix failed\n"); break; }
                QfWireHdr r = h; r.op |= bf ? QFW_BF16 : 0; r.token = out[0]; r.nbytes = (int)((size_t)RS * (bf ? 2 : 4));
                if (io_all(fd, &r, sizeof r, 1)) break;
                if (send_resid(fd, rin, (size_t)RS, bf)) break;
                continue;
            }
            if (h.op != QFW_REGION_M8) { fprintf(stderr, "region: bad op %u\n", h.op); break; }
                        int M = h.token;
            if (M < 1 || M > MAXREQ) break;
            long long p64[MAXM]; long posM[MAXM]; int out[MAXM];
            if (io_all(fd, p64, (size_t)M * sizeof(long long), 0)) break;
            if (io_all(fd, rin, (size_t)M * RS * sizeof(float), 0)) break;   // ONE crossing
            for (int i = 0; i < M; i++) posM[i] = (long)p64[i];
            if (region_step_tail(rin, posM, M, out) != 0) { fprintf(stderr, "region: step failed\n"); break; }
            QfWireHdr r = h; r.nbytes = (int)((size_t)M * sizeof(int));
            if (io_all(fd, &r, sizeof r, 1)) break;
            if (io_all(fd, out, (size_t)M * sizeof(int), 1)) break;          // 4 B/row back
        }
        close(fd);
        fprintf(stderr, "region: peer disconnected\n");
        if (getenv("QF8_PROFILE")) region_prof_print("peer disconnected");
    }
}
