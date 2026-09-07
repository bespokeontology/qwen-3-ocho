// main.cu - qwenflash engine entry
#include "qwenflash.h"
#include "planner.h"
#include "cuda/qf_dense.h"
#include "cuda/qf_decode_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int qf_forward_init(QfModel *m);
extern int qf_decode_step(QfModel *m, int token, long pos);
extern int qf_decode_body_T(QfModel *m, const int *tokens, int T, long pos0);
extern void qf_mtp_reset(void);
extern int qf_mtp_prime(QfModel *m, int token, long pos);
extern int qf_mtp_prime_bos(QfModel *m, int token);
extern float *qf_last_logits(void);
extern int qf_argmax_token(void);   // device-side greedy argmax; one int crosses the bus
extern "C" void qf_m8_timing(double*,double*,long*);
extern "C" void qf_m8_region_timing(double*,double*,double*,long*);
extern "C" void qf_m8_region2_timing(double*,double*,double*,long*);
extern "C" int qf_decode_batch_run_region2(QfModel *m, const int *init_tok, const long *init_pos,
                                           const int (*init_hist)[3], int M, int max_new,
                                           int *out_tokens, int *out_len);
extern "C" int qf_push_residual_M(const float *hR, int M);
extern "C" int qf_pull_residual_M(float *hR, int M);
extern "C" int qf_decode_step_M_upto(QfModel *m, const int *tokensM, const long *posM, int M, int lend);
extern "C" int qf_decode_step_M_upto_ex(QfModel *m, const int *tokensM, const long *posM, int M, int lend, int want_logits);
extern "C" void qf_decode_set_params_M(const int *tokensM, const long *posM, int M);
extern "C" int qf_decode_batch_run_region(QfModel *m, const int *init_tok, const long *init_pos,
                                          const int (*init_hist)[3], int M, int max_new,
                                          int *out_tokens, int *out_len);
extern "C" int qf_region_amd_enabled(void);
extern "C" int qf_decode_batch_prime_region(QfModel *m, const int *const *ids, const int *len,
                                            int M, int *tok_out, long *pos_out, int (*hist_out)[3]);
extern "C" int qf_decode_batch_prime(QfModel *m, const int *const *ids, const int *len,
                                     int M, int *tok_out, long *pos_out, int (*hist_out)[3]);
extern "C" int qf_decode_batch_run(QfModel *m, const int *init_tok, const long *init_pos,
                                   const int (*init_hist)[3], int M, int max_new,
                                   int *out_tokens, int *out_len);
extern int qf_spec_round(QfModel *m, int last_token, long pos, int nsteps,
                         int *out, int *n_accepted);
extern int qf_mtp_load(QfModel *m);
extern void qf_spec_step_stats(long *n, long *hit, int maxn);
extern void qf_spec_step_stats_reset(void);
extern void qf_spec_phase_ms(double *draft, double *verify, double *commit);
extern void qf_forward_shutdown(void);
extern "C" void qf_hc_set_vec(int on);

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---- m8 watchdog: fail fast instead of wedging the GPU ---------------------
// A run whose prime or step exceeds its bound is a defect, not a slow run
// (2026-09-01: a mis-configured prime burned the GPU for 10+ minutes at 95%
// while the operator waited). Bounds (seconds): QF_M8_LOAD_MAX_S (600),
// QF_M8_PRIME_MAX_S (120), QF_M8_STEP_MAX_S (15, no completed step within it).
// On breach: one stderr line with the phase, elapsed and steps done, then
// _exit(3) - the process dies, the driver releases device memory, logs are
// already on disk (stderr is unbuffered).
#include <pthread.h>
#include <unistd.h>
extern "C" volatile long qf_m8_steps_done;
extern "C" long qf_context_cap(void);      /* runtime context cap (cuda/qf.cu) */
extern "C" int qf_ingest_region(QfModel *m, const int *suffix, int N, int *tok_out, long *pos_out);   /* resident-sequence refill */
extern "C" int qf_region1_exit_state(int *tok, long *pos, const float **resid);
extern "C" int qf_fork_rows(int M, long P);
extern "C" int qf_reqbatch_state_init(int M);   /* one-shot sizing: fork width must be set before the prime */
extern "C" int qf_fork_state_sig(int M, long P, double *sig);
extern "C" int qf_region_amd_fork(int M, long P);
extern "C" int qf_region_amd_commit(int r, long P);
extern "C" int qf_fork_commit_rows(int r, long P);
static volatile int g_wd_phase = 0;       // 0 load, 1 prime, 2 decode, 3 done
static volatile double g_wd_t0 = 0;
static double wd_env(const char *k, double d) { const char *e = getenv(k); return (e && atof(e) > 0) ? atof(e) : d; }
static void m8_wd_phase(int ph) { g_wd_t0 = now_seconds(); g_wd_phase = ph; }
static void *m8_watchdog(void *arg) {
    (void)arg;
    const double load_max = wd_env("QF_M8_LOAD_MAX_S", 600), prime_max = wd_env("QF_M8_PRIME_MAX_S", 120),
                 step_max = wd_env("QF_M8_STEP_MAX_S", 15);
    long last = -1; double last_t = now_seconds();
    for (;;) {
        usleep(250000);
        const int ph = g_wd_phase; const double now = now_seconds();
        if (ph == 3) return NULL;
        const char *what = NULL; double lim = 0, el = 0;
        if (ph == 0 && now - g_wd_t0 > load_max) { what = "load"; lim = load_max; el = now - g_wd_t0; }
        if (ph == 1 && now - g_wd_t0 > prime_max) { what = "prime"; lim = prime_max; el = now - g_wd_t0; }
        if (ph == 2) {
            const long sd = qf_m8_steps_done;
            if (sd != last) { last = sd; last_t = now; }
            else if (now - last_t > step_max) { what = "step"; lim = step_max; el = now - last_t; }
        }
        if (what) {
            fprintf(stderr, "\nm8 WATCHDOG: %s phase exceeded %.0f s (%.1f s elapsed, %ld steps done); "
                            "aborting to release the GPU\n", what, lim, el, (long)qf_m8_steps_done);
            fflush(stderr); fflush(stdout);
            _exit(3);
        }
    }
}
static void m8_watchdog_start(void) {
    if (getenv("QF_M8_WATCHDOG") && !atoi(getenv("QF_M8_WATCHDOG"))) return;
    pthread_t th; m8_wd_phase(0);
    if (pthread_create(&th, NULL, m8_watchdog, NULL) == 0) pthread_detach(th);
}


// Stop set. The engine has always stopped only on <|endoftext|> (248044), which
// is what generation_config.json calls bos/pad; the CHAT terminator is
// <|im_end|>, and in THIS checkpoint's 248320-entry vocab that is 248046, not
// the 151645 of the Qwen2/Qwen3 tokenizers (tokenizer_config.json
// added_tokens_decoder: 248045 <|im_start|>, 248046 <|im_end|>). 151645 is an
// ordinary token here and stopping on it would truncate real text.
// generation_config.json lists eos_token_id = [248046, 248044]; QF_STOP_IM_END=1
// makes the engine honour both, so a chat-formatted canary ends where the model
// says it ends instead of running into whatever follows.
#define QF_TOK_ENDOFTEXT 248044
#define QF_TOK_IM_END    248046
static int qf_is_stop(int tok) {
    static int im_end = -1;
    if (im_end < 0) im_end = getenv("QF_STOP_IM_END") ? atoi(getenv("QF_STOP_IM_END")) : 0;
    return tok == QF_TOK_ENDOFTEXT || (im_end && tok == QF_TOK_IM_END);
}

// ---- Frankenpool 2 tail service (QF_SERVE_PORT) ----------------------------
// The AMD head node owns layers 0..QF_LAYER_BEGIN-1 (embed + PLE) and ships
// the residual here; this node runs its layers + output HC + lm_head + argmax
// and returns the token id. Protocol per FRANKENPOOL2_SPLIT_DESIGN.md
// ("Return the TOKEN, not the logits"): residual 40 KB in, token id 4 B out.
// The 40 KB is HCC*NEMBD floats = 10240 f32 (qf.cu residual-transfer note).
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <string.h>

#define QFW_MAGIC 0x30574651u   /* "QFW0" */
#define QFW_RESET 1
#define QFW_STEP  2
#define QFW_TOK   3   /* token-only step: 4 B in, 4 B out, no payload */
#define QFW_TOKPLE 4  /* token + 16 F8 n-gram rows (2560 B) from the head */
#define QFW_STEP_M 11   /* middle region: M residuals in, M residuals out */
#define QFW_EXPERT 6  /* expert-shard RPC: n routed experts evaluated here
                         (5 is reserved: QFW_KCHUNK per the adjudicated T25
                         wire format draft) */
#define QFW_PLE_BYTES 2560
#define QFW_EXP_MAX 10          /* NEXPUSED; payload cap for op 6 */
extern "C" int qf_expert_eval(QfModel *m, int il, int n, const int *ids,
                              const float *w, const float *x2560, float *y2560);
extern int qf_ple_stage_wire_bytes(const void *bytes, int nbytes);
#define QFW_RSIZE 10240         /* floats; must equal HCC*NEMBD */

typedef struct { unsigned magic, op; long long pos; int token, nbytes; } QfWireHdr;

static int io_all(int fd, void *p, size_t n, int wr) {
    char *c = (char *)p;
    while (n) {
        ssize_t k = wr ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, MSG_WAITALL);
        if (k <= 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}

static QfModel *m_ptr;
static int qf_serve(QfModel *m, int port) {
    m_ptr = m;
    static float resid[QFW_RSIZE];
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons((unsigned short)port);
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 1)) {
        fprintf(stderr, "serve: bind/listen on %d failed\n", port); return 1;
    }
    fprintf(stderr, "serve: TAIL ready on port %d (layers from QF_LAYER_BEGIN, eager)\n", port);
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fprintf(stderr, "serve: head node connected\n");
        for (;;) {
            QfWireHdr h;
            if (io_all(fd, &h, sizeof h, 0)) break;
            if (h.magic != QFW_MAGIC) { fprintf(stderr, "serve: bad magic, dropping conn\n"); break; }
            if (h.op == QFW_RESET) {
                qf_session_reset(m);
                h.token = -1;
                if (io_all(fd, &h, sizeof h, 1)) break;
            } else if (h.op == QFW_STEP) {
                if (h.nbytes != (int)sizeof resid) { fprintf(stderr, "serve: bad payload %d\n", h.nbytes); break; }
                if (io_all(fd, resid, sizeof resid, 0)) break;
                // Receipt hook: dump the first few received residuals so the
                // cut can be compared against a reference by cosine.
                if (const char *dd = getenv("QF_SERVE_DUMP")) {
                    if (h.pos < 4) {
                        char pth[512];
                        snprintf(pth, sizeof pth, "%s/wire_resid_pos%lld.bin", dd, h.pos);
                        FILE *df = fopen(pth, "wb");
                        if (df) { fwrite(resid, 1, sizeof resid, df); fclose(df); }
                        // Token ids alongside the residuals: the reference run
                        // must replay the IDENTICAL sequence (frozen condition
                        // object), so the receipt records what actually arrived.
                        snprintf(pth, sizeof pth, "%s/wire_tokens.txt", dd);
                        df = fopen(pth, h.pos == 0 ? "wb" : "ab");
                        if (df) { fprintf(df, "%d\n", h.token); fclose(df); }
                    }
                }
                // Split, not ||-chained: these fail for completely different
                // reasons (a bad residual handoff vs a bad forward pass) and a
                // combined message cannot tell a debugger which. The layer-run
                // split is an exercised-once path; it deserves a precise error.
                if (qf_push_residual(m, resid) != 0) {
                    // Name the CUDA error rather than just the call: a sticky
                    // error from an earlier async launch surfaces here and
                    // would otherwise be blamed on the residual handoff.
                    cudaError_t ce = cudaGetLastError();
                    fprintf(stderr, "serve: PUSH_RESIDUAL failed at pos %lld (lb=%d): %s — refusing to answer\n",
                            h.pos, qf_layer_begin(), cudaGetErrorString(ce));
                    break;
                }
                if (qf_decode_step(m, h.token, (long)h.pos) != 0) {
                    fprintf(stderr, "serve: DECODE_STEP failed at pos %lld (lb=%d) — refusing to answer\n",
                            h.pos, qf_layer_begin());
                    break;   /* no fallback: a dead arm must look dead */
                }
                h.token = qf_argmax_token();
                h.nbytes = 0;
                if (h.token < 0) { fprintf(stderr, "serve: argmax failed\n"); break; }
                if (io_all(fd, &h, sizeof h, 1)) break;
            } else if (h.op == QFW_TOK) {
                // Control-plane form: the head node sends only the token id;
                // this node embeds and runs everything. No payload either way.
                // A split tail (lb>0) cannot embed: without a pushed residual
                // it would decode over stale ctx.R and emit plausible garbage.
                // Refuse loudly instead (the dead-arm rule).
                if (qf_layer_begin() > 0) {
                    fprintf(stderr, "serve: TOK refused on a layer-split tail (lb=%d);"
                            " use STEP with a residual\n", qf_layer_begin());
                    break;
                }
                if (h.nbytes != 0) { fprintf(stderr, "serve: TOK with payload\n"); break; }
                if (qf_decode_step(m, h.token, (long)h.pos) != 0) {
                    fprintf(stderr, "serve: decode failed at pos %lld — refusing to answer\n", h.pos);
                    break;
                }
                h.token = qf_argmax_token();
                if (h.token < 0) { fprintf(stderr, "serve: argmax failed\n"); break; }
                if (io_all(fd, &h, sizeof h, 1)) break;
            } else if (h.op == QFW_STEP_M) {
                // Middle region of the established map: AMD owns 0..lb-1 + the
                // head; this box runs [lb, 48) and hands the residual back. No
                // lm_head here, no logits on the wire.
                static float rbuf[8 * QFW_RSIZE];
                static long  pm[8];
                int M = h.token;
                if (M < 1 || M > 8) { fprintf(stderr, "serve: STEP_M bad M=%d\n", M); break; }
                long long p64[8];
                if (io_all(fd, p64, (size_t)M * sizeof(long long), 0)) break;
                if (io_all(fd, rbuf, (size_t)M * QFW_RSIZE * 4, 0)) break;
                for (int i = 0; i < M; i++) pm[i] = (long)p64[i];
                static int dummy_tok[8];
                for (int i = 0; i < M; i++) dummy_tok[i] = 0;   // embed skipped (lb>0)
                if (qf_push_residual_M(rbuf, M) != 0) { fprintf(stderr, "serve: STEP_M push failed\n"); break; }
                if (qf_decode_step_M_upto_ex(&*m_ptr, dummy_tok, pm, M, 48, /*want_logits=*/0) != 0) {
                    fprintf(stderr, "serve: STEP_M decode failed\n"); break; }
                if (qf_pull_residual_M(rbuf, M) != 0) break;
                QfWireHdr r = h; r.nbytes = (int)((size_t)M * QFW_RSIZE * 4);
                if (io_all(fd, &r, sizeof r, 1)) break;
                if (io_all(fd, rbuf, (size_t)M * QFW_RSIZE * 4, 1)) break;
            } else if (h.op == QFW_EXPERT) {
                // Frankenpool expert-shard RPC: the head routed the token and
                // ships (layer in h.token, position in h.pos advisory) plus
                // n routed (id, w) pairs THIS node owns and the activation x.
                // Reply: the weighted expert contribution, 2560 f32. Works on
                // any full-resident node regardless of lb: expert weights are
                // per-layer tensors, not layer-run state.
                static struct { unsigned n; struct { unsigned id; float w; } e[QFW_EXP_MAX]; } req;
                static float xact[2560], ypart[2560];
                if (h.nbytes < (int)sizeof(unsigned) ||
                    h.nbytes > (int)(sizeof req + sizeof xact)) {
                    fprintf(stderr, "serve: EXPERT bad payload %d\n", h.nbytes); break;
                }
                if (io_all(fd, &req.n, sizeof(unsigned), 0)) break;
                if (req.n < 1 || req.n > QFW_EXP_MAX ||
                    h.nbytes != (int)(sizeof(unsigned) + req.n * 8 + sizeof xact)) {
                    fprintf(stderr, "serve: EXPERT bad n=%u nbytes=%d\n", req.n, h.nbytes); break;
                }
                if (io_all(fd, req.e, req.n * 8, 0)) break;
                if (io_all(fd, xact, sizeof xact, 0)) break;
                int ids[QFW_EXP_MAX]; float ws[QFW_EXP_MAX];
                for (unsigned k = 0; k < req.n; k++) { ids[k] = (int)req.e[k].id; ws[k] = req.e[k].w; }
                if (qf_expert_eval(m, h.token, (int)req.n, ids, ws, xact, ypart) != 0) {
                    fprintf(stderr, "serve: EXPERT_EVAL failed L%d pos %lld — refusing to answer\n",
                            h.token, h.pos);
                    break;   /* dead-arm rule: no zero-filled reply */
                }
                h.nbytes = (int)sizeof ypart;
                if (io_all(fd, &h, sizeof h, 1)) break;
                if (io_all(fd, ypart, sizeof ypart, 1)) break;
            } else if (h.op == QFW_TOKPLE) {
                // Token + the 16 n-gram rows gathered by the control plane.
                // Requires QF_PLE_WIRE=1 on this node; the local table is
                // then never read (staging a step without rows is fatal).
                static unsigned char plerows[QFW_PLE_BYTES];
                if (h.nbytes != QFW_PLE_BYTES) { fprintf(stderr, "serve: TOKPLE bad payload %d\n", h.nbytes); break; }
                if (io_all(fd, plerows, sizeof plerows, 0)) break;
                if (qf_ple_stage_wire_bytes(plerows, QFW_PLE_BYTES) != 0) {
                    fprintf(stderr, "serve: TOKPLE without QF_PLE_WIRE=1 — refusing\n");
                    break;
                }
                if (qf_decode_step(m, h.token, (long)h.pos) != 0) {
                    fprintf(stderr, "serve: decode failed at pos %lld — refusing to answer\n", h.pos);
                    break;
                }
                h.token = qf_argmax_token();
                h.nbytes = 0;
                if (h.token < 0) { fprintf(stderr, "serve: argmax failed\n"); break; }
                if (io_all(fd, &h, sizeof h, 1)) break;
            } else break;
        }
        close(fd);
        fprintf(stderr, "serve: head node disconnected\n");
    }
}


// ---- M=8 multi-request bring-up harness (QF_M8_PROMPTS=f0,f1,...,f7) --------
// Primes N<=8 independent prompts, runs the true M=8 batched decode driver
// (gate off -> local Spark routed path; gate on -> AMD four-card offload), and
// prints each row's complete output + aggregate throughput. Not a microbenchmark
// - an integrated functional M=8 E2E proving independent state + the scheduler.
static int qf_run_m8(QfModel *model, const char *csv, int max_new) {
        char buf[8192]; strncpy(buf, csv, sizeof buf - 1); buf[sizeof buf - 1] = 0;
            int  *ids[32]; int len[32]; int M = 0;                 // up to two 16-row slots
    for (char *tok = strtok(buf, ","); tok && M < 32; tok = strtok(NULL, ",")) {
        FILE *f = fopen(tok, "r");
        if (!f) { fprintf(stderr, "m8: cannot open %s\n", tok); return 1; }
        int *a = (int *)malloc(sizeof(int) * 262144), n = 0, v;   /* prompt ids cap raised from 4096 (it silently truncated 5K+ prompts) */
        while (n < 262144 && fscanf(f, "%d", &v) == 1) a[n++] = v;
        fclose(f);
        if (!n) { fprintf(stderr, "m8: empty %s\n", tok); return 1; }
        ids[M] = a; len[M] = n; M++;
    }
    if (M < 1) { fprintf(stderr, "m8: no prompts\n"); return 1; }
    fprintf(stderr, "m8: %d requests; priming...\n", M);
    m8_wd_phase(1);
    if (getenv("QF_M8_FORK")) { const int Mf = atoi(getenv("QF_M8_FORK")); if (Mf > 1 && qf_reqbatch_state_init(Mf)) { fprintf(stderr, "fork: state sizing for %d rows failed\n", Mf); return 1; } }
            int tok[32]; long pos[32]; int hist[32][3];
    double t_prime = now_seconds();
        const int region = qf_region_amd_enabled();
        if (region) fprintf(stderr, "m8: HETEROGENEOUS ALTERNATING MAP - priming each row across AMD prefix + Spark tail\n");
        if (!region && M > 16) { fprintf(stderr, "m8: the Spark-only path takes at most 16 rows\n"); return 1; }
    if ((region ? qf_decode_batch_prime_region(model, (const int *const *)ids, len, M, tok, pos, hist)
                : qf_decode_batch_prime(model, (const int *const *)ids, len, M, tok, pos, hist)) != 0) {
        fprintf(stderr, "m8: prime failed\n"); return 1;
    }
    fprintf(stderr, "m8: primed in %.2fs; decoding M=%d...\n", now_seconds() - t_prime, M);
    m8_wd_phase(2);
    {   // never generate past the context cap: KV, conv ring and indexer pools on BOTH boxes are
        // sized by the context; a step past it indexes outside them (AMD server hang at pos 8196
        // with an 8192 context on the 8K prompt, 09-06).
        long pmax = 0; for (int r = 0; r < M; r++) if (pos[r] > pmax) pmax = pos[r];
        const long room = qf_context_cap() - 1 - pmax;
        if (room < 1) { fprintf(stderr, "m8: prompt (%ld tokens) fills the context (%ld); nothing to generate\n", pmax, qf_context_cap()); return 1; }
        if (max_new > room) { fprintf(stderr, "m8: max_new %d capped to %ld by the context cap %ld\n", max_new, room, qf_context_cap()); max_new = (int)room; }
    }
    // FORK (Kolmogorov branch point): QF_M8_FORK=Mf siblings from the ONE primed parent (M == 1, region path).
    // Both boxes copy the parent's resident state into Mf row slots; the M-row region decode then runs
    // QF_M8_FORK_STEPS steps for all siblings at once (AMD prefix Mf rows -> Spark tail Mf -> AMD head Mf).
    // Row 0 stays greedy (control: must equal the M=1 continuation); rows >= 1 sample at the server's
    // QF_HEAD_TEMP. Receipt: fork wall, steps wall, ms/step, sibling-tokens/s, the Mf continuations.
    if (M == 1 && region && getenv("QF_M8_FORK")) {
        const int Mf = atoi(getenv("QF_M8_FORK")); const int K = getenv("QF_M8_FORK_STEPS") ? atoi(getenv("QF_M8_FORK_STEPS")) : 32;
        if (Mf < 1 || Mf > 16) { fprintf(stderr, "fork: QF_M8_FORK must be 1..16\n"); return 1; }
        if (getenv("QF_M8_FORK_LOOP")) {
            // DECISION/COMMIT LOOP: iterate { render Mf alternatives for K steps; evaluate; choose; commit }.
            // The chosen row's full state (AMD prefix + Spark tail) becomes the new parent (row 0) on BOTH
            // boxes; the next iteration re-renders FROM the committed state. The conditional-probability tree
            // is preserved in the log: every branch's tokens, signatures and scores, and every choice.
            // Chooser (v1, transparent): exclude early-EOS rows; score = |sigR - median(sigR of survivors)|;
            // commit the closest-to-median survivor. Row 0 (greedy) is a candidate like any other.
            const int NIT = getenv("QF_M8_FORK_LOOP_ITERS") ? atoi(getenv("QF_M8_FORK_LOOP_ITERS")) : 4;
            long P = pos[0]; int last_tok = tok[0];
            int tokM[16]; long posM[16]; int histM[16][3];
            for (int r = 0; r < Mf; r++) { histM[r][0] = hist[0][0]; histM[r][1] = hist[0][1]; histM[r][2] = hist[0][2]; }
            long made = 0;
            for (int it = 0; it < NIT; it++) {
                const double tl0 = now_seconds();
                if (qf_region_amd_fork(Mf, P)) { fprintf(stderr, "loop: AMD fork failed\n"); return 1; }
                if (qf_fork_rows(Mf, P)) { fprintf(stderr, "loop: Spark fork failed\n"); return 1; }
                for (int r = 0; r < Mf; r++) { tokM[r] = last_tok; posM[r] = P; }
                int *outM = (int *)malloc(sizeof(int) * (size_t)Mf * K), outlenM[16];
                // Chunked render (observation boundaries only): 16-token chunks with a live per-row
                // preview (ids + artifact count), accumulated into the SAME outM the one-shot call
                // would produce; the chooser still sees all K tokens. Chunking does not change
                // semantics - the decoder state advances identically across chunk boundaries.
                const int CH = (K % 16 == 0) ? 16 : K; const int nch = (K + CH - 1) / CH;
                int *tmp = (int *)malloc(sizeof(int) * (size_t)Mf * CH);
                for (int r = 0; r < Mf; r++) outlenM[r] = 0;
                for (int c = 0; c < nch; c++) {
                    int ol[16];
                    if (qf_decode_batch_run_region(model, tokM, posM, histM, Mf, CH, tmp, ol)) { fprintf(stderr, "loop: render chunk %d failed\n", c); return 1; }
                    // the region decode lays rows out with stride max_new (=CH): copy each row's
                    // chunk into the K-stride outM layout the chooser and the SIBLING dump read
                    for (int r = 0; r < Mf; r++)
                        memcpy(outM + (size_t)r * K + (size_t)c * CH, tmp + (size_t)r * CH, (size_t)ol[r] * sizeof(int));
                    printf("OCHO it%02d render %d/%d\n", it, (c + 1) * CH, K);
                    for (int r = 0; r < Mf; r++) {
                        outlenM[r] += ol[r];
                        int art = 0;
                        for (int i = 0; i < ol[r]; i++) if (outM[(size_t)r * K + (size_t)c * CH + i] >= 110000) art++;
                        printf("  r%d art=%d ids:", r, art);
                        for (int i = 0; i < ol[r]; i++) printf(" %d", outM[(size_t)r * K + (size_t)c * CH + i]);
                        printf("\n");
                    }
                    fflush(stdout);
                    for (int r = 0; r < Mf; r++) {
                        if (ol[r] > 0) tokM[r] = outM[(size_t)r * K + (size_t)c * CH + ol[r] - 1];
                        posM[r] += ol[r];
                    }
                }
                free(tmp);
                double s[32];
                if (qf_fork_state_sig(Mf, P + K, s)) { fprintf(stderr, "loop: sig failed\n"); return 1; }
                double sigR[16]; int nart[16] = {0}; int eos[16] = {0};
                for (int r = 0; r < Mf; r++) {
                    sigR[r] = s[2 * r];
                    for (int i = 0; i < outlenM[r]; i++) {
                        int t = outM[(size_t)r * K + i];
                        if (t == 248046 && i < K / 2) eos[r] = 1;   // early EOS = nearly dead
                        if (t >= 110000) nart[r]++;                  // mojibake/tail-vocab artifact (beam measurement)
                    }
                }
                double sr[16]; for (int r = 0; r < Mf; r++) sr[r] = sigR[r];
                for (int a = 0; a < Mf; a++) for (int b = a + 1; b < Mf; b++) if (sr[b] < sr[a]) { double t = sr[a]; sr[a] = sr[b]; sr[b] = t; }
                double med = sr[Mf / 2];
                int chosen = 0; long bestscore = 1L << 62;
                for (int r = 0; r < Mf; r++) {
                    double d = sigR[r] - med; if (d < 0) d = -d;
                    // lexicographic: fewest artifacts first (early-EOS = +100), consensus distance second
                    long sc = ((long)(nart[r] + (eos[r] ? 100 : 0)) << 40) + (long)(d > 1.0e6 ? 1.0e6 : d);
                    fprintf(stderr, "LOOP it%d row %d: sigR %.1f | dist %.1f | art=%d eos=%d\n", it, r, sigR[r], d, nart[r], eos[r]);
                    if (sc < bestscore) { bestscore = sc; chosen = r; }
                }
                if (qf_fork_commit_rows(chosen, P + K)) { fprintf(stderr, "loop: Spark commit failed\n"); return 1; }
                if (qf_region_amd_commit(chosen, P + K)) { fprintf(stderr, "loop: AMD commit failed\n"); return 1; }
                const double tl1 = now_seconds();
                printf("OCHO it%02d DECISION: selected r%d (art=%d eos=%d); COMMIT\n", it, chosen, nart[chosen], eos[chosen]);
                fflush(stdout);
                fprintf(stderr, "LOOP it%d COMMIT row %d (sigR %.1f); render+commit wall %.2f s; parent now pos %ld\n", it, chosen, sigR[chosen], tl1 - tl0, P + K);
                for (int r = 0; r < Mf; r++) {
                    printf("=== LOOP it%d SIBLING %d (%d out tok) ===\n", it, r, outlenM[r]);
                    for (int i = 0; i < outlenM[r]; i++) printf("%d ", outM[(size_t)r * K + i]);
                    printf("\n");
                }
                last_tok = outM[(size_t)chosen * K + outlenM[chosen] - 1];
                for (int r = 0; r < Mf; r++) { histM[r][0] = histM[chosen][0]; histM[r][1] = histM[chosen][1]; histM[r][2] = histM[chosen][2]; }
                P += K; made += K;
                free(outM);
            }
            fprintf(stderr, "LOOP END: %d iterations, %ld committed tokens, final parent pos %ld, last token %d\n", NIT, made, P, last_tok);
            return 0;
        }
        const double tf0 = now_seconds();
        if (qf_region_amd_fork(Mf, pos[0])) { fprintf(stderr, "fork: AMD fork failed\n"); return 1; }
        const double tf1 = now_seconds();
        if (qf_fork_rows(Mf, pos[0])) { fprintf(stderr, "fork: Spark fork failed\n"); return 1; }
        const double tf2 = now_seconds();
        int tokM[16]; long posM[16]; int histM[16][3];
        for (int r = 0; r < Mf; r++) { tokM[r] = tok[0]; posM[r] = pos[0]; histM[r][0] = hist[0][0]; histM[r][1] = hist[0][1]; histM[r][2] = hist[0][2]; }
        int *outM = (int *)malloc(sizeof(int) * Mf * K), outlenM[16];
        const double ts0 = now_seconds();
        if (getenv("QF_M8_FORK_ALT")) {
            // INDEPENDENCE PROOF (no sampler): step 1 all rows greedy from the same token; step 2 rows 0..Mf-2
            // continue from their own outputs, the LAST row is fed a different known token. Tokens and state
            // signatures per row after each step: untouched rows stay mutually identical, the altered row differs.
            const int alt = atoi(getenv("QF_M8_FORK_ALT"));
            int o1[16], l1[16], o2[16], l2[16]; double s1[32], s2[32];
            if (qf_decode_batch_run_region(model, tokM, posM, histM, Mf, 1, o1, l1)) { fprintf(stderr, "fork: step 1 failed\n"); return 1; }
            if (qf_fork_state_sig(Mf, posM[0] + 1, s1)) { fprintf(stderr, "fork: sig 1 failed\n"); return 1; }   /* written positions [0, pos+1) */
            int tok2[16]; long pos2[16];
            for (int r = 0; r < Mf; r++) { tok2[r] = o1[r]; pos2[r] = posM[r] + 1; }
            tok2[Mf - 1] = alt;
            if (qf_decode_batch_run_region(model, tok2, pos2, histM, Mf, 1, o2, l2)) { fprintf(stderr, "fork: step 2 failed\n"); return 1; }
            if (qf_fork_state_sig(Mf, posM[0] + 2, s2)) { fprintf(stderr, "fork: sig 2 failed\n"); return 1; }
            const double ts1 = now_seconds();
            fprintf(stderr, "FORK RECEIPT: parent %ld positions (token %d at pos %ld); fork AMD %.1f ms + Spark %.1f ms; %d siblings x 2 steps (row %d fed token %d at step 2): %.2f s\n",
                    pos[0] + 1, tok[0], pos[0], (tf1 - tf0) * 1e3, (tf2 - tf1) * 1e3, Mf, Mf - 1, alt, ts1 - ts0);
            for (int r = 0; r < Mf; r++)
                fprintf(stderr, "FORK ROW %d: step1 tok %d sigR %.9e sigK %.9e | step2 in %d out %d sigR %.9e sigK %.9e\n", r, o1[r], s1[2*r], s1[2*r+1], tok2[r], o2[r], s2[2*r], s2[2*r+1]);
            for (int r = 0; r < Mf; r++) { printf("=== SIBLING %d (2 out tok) ===\n%d %d \n", r, o1[r], o2[r]); }
            return 0;
        }
        if (qf_decode_batch_run_region(model, tokM, posM, histM, Mf, K, outM, outlenM)) { fprintf(stderr, "fork: sibling decode failed\n"); return 1; }
        const double ts1 = now_seconds();
        long tot = 0; for (int r = 0; r < Mf; r++) tot += outlenM[r];
        fprintf(stderr, "FORK RECEIPT: parent %ld positions (token %d at pos %ld); fork AMD %.1f ms + Spark %.1f ms; %d siblings x %d steps: %.2f s = %.1f ms/step, %.1f sibling-tok/s (%.2f tok/s per sibling)\n",
                pos[0] + 1, tok[0], pos[0], (tf1 - tf0) * 1e3, (tf2 - tf1) * 1e3, Mf, K, ts1 - ts0, (ts1 - ts0) * 1e3 / K, tot / (ts1 - ts0), tot / (ts1 - ts0) / Mf);
        for (int r = 0; r < Mf; r++) {
            printf("=== SIBLING %d (%d out tok) ===\n", r, outlenM[r]);
            for (int i = 0; i < outlenM[r]; i++) printf("%d ", outM[(size_t)r * K + i]);
            printf("\n");
        }
        if (getenv("QF_M8_FORK_SIG")) {
            // MC-C: per-row recurrent/KV state signatures after the K-step stochastic field
            double s[32];
            if (qf_fork_state_sig(Mf, posM[0] + K, s)) { fprintf(stderr, "fork: final sig failed\n"); return 1; }
            for (int r = 0; r < Mf; r++)
                fprintf(stderr, "FORK SIG %d: sigR %.9e sigK %.9e\n", r, s[2 * r], s[2 * r + 1]);
        }
        return 0;
    }
            int *out = (int *)malloc(sizeof(int) * M * max_new), outlen[32];
    double t0 = now_seconds();
    int rc_run;
    const int slots = getenv("QF_M8_SLOTS") ? atoi(getenv("QF_M8_SLOTS")) : 1;
    if (region && slots >= 2) {
        fprintf(stderr, "m8: ALTERNATING MAP, %d slots in flight (AMD prefix(other) || Spark tail(this))\n", slots);
                if (M != slots * 16) { fprintf(stderr, "m8: QF_M8_SLOTS=%d needs exactly %d prompts (got %d)\n", slots, slots * 16, M); return 1; }
        rc_run = qf_decode_batch_run_region2(model, tok, pos, hist, M, max_new, out, outlen);
    } else if (region) {
        fprintf(stderr, "m8: ALTERNATING MAP, one batch in flight (AMD prefix -> Spark tail -> AMD head, serial)\n");
                if (M > 16) { fprintf(stderr, "m8: one slot takes at most 16 rows; set QF_M8_SLOTS=2 for 32\n"); return 1; }
        rc_run = qf_decode_batch_run_region(model, tok, pos, hist, M, max_new, out, outlen);
    } else {
        rc_run = qf_decode_batch_run(model, tok, pos, hist, M, max_new, out, outlen);
    }
    m8_wd_phase(3);
    if (rc_run != 0) { fprintf(stderr, "m8: decode failed\n"); return 1; }
    double dt = now_seconds() - t0;
    if (getenv("QF_M8_TIMING")) {
        double off,lay; long nc; qf_m8_timing(&off,&lay,&nc);
        fprintf(stderr, "m8 timing: offload %.0f ms over %ld layer-calls (%.2f ms/call) = %.0f%% of %.0f ms decode\n",
                off, nc, nc? off/nc:0.0, 100.0*off/(dt*1000), dt*1000);
                double tp, tt, th; long ns; qf_m8_region_timing(&tp, &tt, &th, &ns);
        if (ns) fprintf(stderr, "m8 timing: alternating map, %ld steps: AMD prefix %.1f ms/step, "
                        "Spark tail %.1f ms/step (incl. residual H2D/D2H), AMD head %.1f ms/step "
                        "(host wall clock around each RPC)\n", ns, tp/ns, tt/ns, th/ns);
        double wp, t2, wh; long nt; qf_m8_region2_timing(&wp, &t2, &wh, &nt);
        if (nt) fprintf(stderr, "m8 timing: two slots, %ld tails: exposed prefix wait %.1f ms/tail, "
                        "Spark tail %.1f ms/tail (incl. residual H2D/D2H), exposed head wait %.1f ms/tail\n",
                        nt, wp/nt, t2/nt, wh/nt); }
    long total = 0; for (int r = 0; r < M; r++) total += outlen[r];
    for (int r = 0; r < M; r++) {
        printf("=== ROW %d (%d out tok) ===\n", r, outlen[r]);
        for (int i = 0; i < outlen[r]; i++) printf("%d ", out[(size_t)r * max_new + i]);
        printf("\n");
    }
    // REFILL / CONTEXT INJECTION (M=1, region1 fused path): QF_M8_REFILL=ids1.txt[,ids2.txt...] appends each
    // file's tokens to the RESIDENT sequence (no reset, no replay), then decodes QF_M8_MAX2 tokens.
    // Reported per injection: suffix size, ingest wall (suffix tok/s), first new token latency measured
    // from context arrival, continuation decode tok/s.
    if (M == 1 && region && slots < 2 && getenv("QF_M8_REFILL")) {
        const int max2 = getenv("QF_M8_MAX2") ? atoi(getenv("QF_M8_MAX2")) : max_new;
        char rb[8192]; strncpy(rb, getenv("QF_M8_REFILL"), sizeof rb - 1); rb[sizeof rb - 1] = 0;
        int inj = 0;
        for (char *fn = strtok(rb, ","); fn; fn = strtok(NULL, ","), inj++) {
            FILE *f = fopen(fn, "r"); if (!f) { fprintf(stderr, "refill: cannot open %s\n", fn); return 1; }
            int *sfx = (int *)malloc(sizeof(int) * 262144), n = 0, v;
            while (n < 262144 && fscanf(f, "%d", &v) == 1) sfx[n++] = v;
            fclose(f);
            if (!n) { fprintf(stderr, "refill: empty %s\n", fn); return 1; }
            int etok; long epos;
            if (qf_region1_exit_state(&etok, &epos, NULL)) { fprintf(stderr, "refill: no exit state\n"); return 1; }
            const long room = qf_context_cap() - 1 - (epos + n);
            if (room < 2) { fprintf(stderr, "refill: %d tokens do not fit the context (resident %ld, cap %ld)\n", n, epos, qf_context_cap()); return 1; }
            fprintf(stderr, "refill #%d: resident %ld positions (exit token %d at pos %ld), injecting %d tokens from %s\n", inj, epos + 1, etok, epos, n, fn);
            const double ta = now_seconds();
            int ntok; long npos;
            if (qf_ingest_region(model, sfx, n, &ntok, &npos)) { fprintf(stderr, "refill: ingest failed\n"); return 1; }
            const double tb = now_seconds();
            int tk2[1] = {ntok}; long ps2[1] = {npos}; int hs2[1][3] = {{0, 0, 0}};
            int out1[1], ol1[1];
            if (qf_decode_batch_run_region(model, tk2, ps2, hs2, 1, 1, out1, ol1)) { fprintf(stderr, "refill: first-token decode failed\n"); return 1; }
            const double tc = now_seconds();
            int *out2 = (int *)malloc(sizeof(int) * (max2 + 1)), ol2[1] = {0};
            int rem = max2 - 1;
            if (rem > 0 && ol1[0] == 1) {
                int tk3[1] = {out1[0]}; long ps3[1] = {npos + 1};
                if (qf_decode_batch_run_region(model, tk3, ps3, hs2, 1, rem, out2, ol2)) { fprintf(stderr, "refill: continuation decode failed\n"); return 1; }
            }
            const double td = now_seconds();
            const long cap2 = (long)max2 - 1 - room; (void)cap2;
            fprintf(stderr, "refill #%d RECEIPT: suffix %d tok ingested in %.3f s (%.0f tok/s); first new token %.3f s after context arrival (ingest %.3f + first step %.3f); continuation %d tok in %.2f s = %.2f tok/s; resident now %ld positions\n",
                    inj, n, tb - ta, n / (tb - ta), tc - ta, tb - ta, tc - tb, ol2[0], td - tc, ol2[0] > 0 ? ol2[0] / (td - tc) : 0.0, npos + 1 + ol1[0] + ol2[0]);
            printf("=== REFILL %d (%d suffix tok, %d out tok) ===\n", inj, n, ol1[0] + ol2[0]);
            for (int i = 0; i < ol1[0]; i++) printf("%d ", out1[i]);
            for (int i = 0; i < ol2[0]; i++) printf("%d ", out2[i]);
            printf("\n");
            free(sfx); free(out2);
        }
    }
    fprintf(stderr, "\nm8 E2E: M=%d, %ld tokens in %.2fs = %.2f tok/s aggregate (%.1f tok/min), %.2f tok/s/req\n",
            M, total, dt, total / dt, total / dt * 60.0, (total / dt) / M);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = getenv("QF_MODEL_DIR") ? getenv("QF_MODEL_DIR") : "./model";
    QfModel model;
    if (getenv("QF_M8_PROMPTS")) m8_watchdog_start();
    fprintf(stderr, "loading model...\n");
    if (qf_model_load(&model, dir) != 0) { fprintf(stderr, "load FAILED\n"); return 1; }
    fprintf(stderr, "model loaded\n");
    if (qf_forward_init(&model) != 0) { fprintf(stderr, "forward init failed\n"); return 1; }

    if (getenv("QF_M8_PROMPTS")) {
        int mx = getenv("QF_M8_MAX") ? atoi(getenv("QF_M8_MAX")) : 128;
        return qf_run_m8(&model, getenv("QF_M8_PROMPTS"), mx);
    }

    // Frankenpool tail mode: serve residual->token over TCP, no prompt file.
    if (getenv("QF_SERVE_PORT"))
        return qf_serve(&model, atoi(getenv("QF_SERVE_PORT")));

    int ids[512], n_ids = 0;
    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "no prompt file\n"); return 1; }
        while (n_ids < 512 && fscanf(f, "%d", &ids[n_ids]) == 1) n_ids++;
        fclose(f);
    }
    if (!n_ids) { fprintf(stderr, "empty prompt\n"); return 1; }
    fprintf(stderr, "prompt %d tokens\n", n_ids);

    static float host_logits[248320];
    static int prevb = 0;
    // QF_SPEC=<nsteps> runs MTP speculative decode. The draft head loads BEFORE
    // the prompt so it can be primed over it: its KV cache has to cover every
    // position its attention will sum over.
    const int spec = getenv("QF_SPEC") ? atoi(getenv("QF_SPEC")) : 0;
    if (spec && qf_mtp_load(&model) != 0) { fprintf(stderr, "mtp load failed\n"); return 1; }

    long pos = 0;
    double prompt_start = now_seconds();
    if (spec) qf_mtp_reset();
    // Slot 0 of the draft head's KV has no predecessor hidden state, so the
    // per-token prime below (which pairs token p with the trunk hidden from
    // p-1) can never write it. Leaving it zero leaves a zero-K attention sink
    // in every later draft query. Prime it from the token alone.
    if (spec && qf_mtp_prime_bos(&model, ids[0]) != 0) {
        fprintf(stderr, "mtp bos prime failed\n"); return 1;
    }
    // QF_PREFILL_CHUNK: run all-but-last prompt tokens through the batched
    // trunk body in chunks (dense weights read once per chunk instead of
    // once per token - the same amortization the verify pass exists for),
    // then the final token via the sequential step so last-logits and state
    // land exactly where the stock path leaves them. Sequential under spec
    // (per-position MTP priming) and when not fully resident (the batched
    // body does not service expert routing).
    //
    // The value is the chunk WIDTH, clamped to [1, QF_SPEC_MAXT]; =1 keeps
    // its original published meaning of "on at the default width 8" so the
    // Addendum 7 receipt still reproduces from its own command line. Width is
    // a free parameter and 8 was chosen only because it is the maximum the
    // batched body supports - which is exactly the reasoning that produced
    // the AMD side's M=8 pessimum (Addendum 7a), so it is swept, not assumed.
    const char *pfe = getenv("QF_PREFILL_CHUNK");
    int pfw = pfe ? atoi(pfe) : 0;
    if (pfw == 1) pfw = 8;                        // published default
    if (pfw > 8) pfw = 8;
    const int pfchunk = pfw > 0 && !spec && model.exp_mode == QF_SPARK_MODE_FULL;
    if (pfchunk) {
        fprintf(stderr, "prefill: chunked, width %d\n", pfw);
        int i = 0;
        while (i < n_ids - 1) {
            int T = n_ids - 1 - i; if (T > pfw) T = pfw;
            if (qf_decode_body_T(&model, ids + i, T, pos) != 0) {
                fprintf(stderr, "chunked prefill failed at %d\n", i); return 1;
            }
            i += T; pos += T;
        }
        if (qf_decode_step(&model, ids[n_ids - 1], pos++) != 0) {
            fprintf(stderr, "decode failed at last prompt token\n"); return 1;
        }
    } else
    for (int i = 0; i < n_ids; i++) {
        if (qf_decode_step(&model, ids[i], pos++) != 0) { fprintf(stderr, "decode failed at prompt token %d\n", i); return 1; }
        if (spec && i + 1 < n_ids && qf_mtp_prime(&model, ids[i + 1], pos) != 0) {
            fprintf(stderr, "mtp prime failed at %d\n", i); return 1;
        }
    }
    double prompt_elapsed = now_seconds() - prompt_start;
    fprintf(stderr, "prompt throughput: %.2f tok/s (%d tokens, %.3fs)\n",
            n_ids / prompt_elapsed, n_ids, prompt_elapsed);
    if (qf_graph_capture(&model) != 0) {
        fprintf(stderr, "decode graph capture failed\n");
        return 1;
    }
    float *logits = qf_last_logits();
    cudaMemcpy(host_logits, logits, model.cfg.n_vocab * 4, cudaMemcpyDeviceToHost);
    logits = host_logits;
    { float mx=-1e30f, mn=1e30f; int nan=0;
      for (int v=0; v<model.cfg.n_vocab; v++){ if(isnan(logits[v]))nan++; if(logits[v]>mx)mx=logits[v]; if(logits[v]<mn)mn=logits[v]; }
      fprintf(stderr, "logits: max=%.4f min=%.4f nan=%d\n", mx, mn, nan);
      for (int t=0;t<5;t++){ int b=0; for(int v=0;v<model.cfg.n_vocab;v++) if(logits[v]>logits[b] && (t==0 || logits[v]<logits[prevb])) b=v; if(t==0) prevb=b; fprintf(stderr,"top%d=%d (%.3f)\n",t,b,logits[b]); }
    }
    // Repeat the generation from the same resident load so a run yields a
    // VARIANCE BAND, not a single coordinate. One sample cannot separate a
    // real change from run-to-run spread (measured ~3.4% on this engine), and
    // a verdict typed from one sample is the 87-rows error.
    //   QF_GEN_TOKENS  tokens per generation (default 24)
    //   QF_GEN_REPEAT  generations per load   (default 1)
    const int gen_n = getenv("QF_GEN_TOKENS") ? atoi(getenv("QF_GEN_TOKENS")) : 24;
    const int gen_r = getenv("QF_GEN_REPEAT") ? atoi(getenv("QF_GEN_REPEAT")) : 1;
    // QF_HC_AB=1: alternate the hyper-connection arm per generation, so one
    // resident load yields interleaved samples of both. Cross-load comparison
    // cannot resolve a change smaller than the inter-load spread.
    const int hc_ab = getenv("QF_HC_AB") ? atoi(getenv("QF_HC_AB")) : 0;
    for (int rep = 0; rep < gen_r; rep++) {
    if (hc_ab) {
        const int arm = rep & 1;              // 0 = scalar, 1 = vectorized
        qf_hc_set_vec(arm);
        fprintf(stderr, "hc arm: %s\n", arm ? "VECTORIZED" : "scalar");
    }
    if (rep) {
        // fresh sequence: same prompt, same state, so each sample is drawn
        // under the identical condition object
        qf_session_reset(&model);
        pos = 0;
        if (spec) qf_mtp_reset();
        if (spec && qf_mtp_prime_bos(&model, ids[0]) != 0) return 1;
        // Each repeat's prefill is timed too: the first prefill of a load
        // carries graph-capture and cache warmup, so a prefill BAND needs the
        // per-repeat samples, same as generation.
        double rp_start = now_seconds();
        if (pfchunk) {
            int i = 0;
            while (i < n_ids - 1) {
                int T = n_ids - 1 - i; if (T > pfw) T = pfw;
                if (qf_decode_body_T(&model, ids + i, T, pos) != 0) return 1;
                i += T; pos += T;
            }
            if (qf_decode_step(&model, ids[n_ids - 1], pos++) != 0) return 1;
        } else
        for (int i = 0; i < n_ids; i++) {
            if (qf_decode_step(&model, ids[i], pos++) != 0) return 1;
            if (spec && i + 1 < n_ids && qf_mtp_prime(&model, ids[i + 1], pos) != 0) return 1;
        }
        double rp_el = now_seconds() - rp_start;
        fprintf(stderr, "prompt throughput: %.2f tok/s (%d tokens, %.3fs)\n",
                n_ids / rp_el, n_ids, rp_el);
        cudaMemcpy(host_logits, qf_last_logits(), model.cfg.n_vocab * 4, cudaMemcpyDeviceToHost);
        logits = host_logits;
    }
    // Greedy speculative output MUST match greedy sequential decode token for
    // token; the round commits only candidates the trunk has confirmed, so
    // drafts decide how many tokens arrive per round, never which.
    long spec_drafted = 0, spec_accepted = 0, spec_rounds = 0;
    if (spec) qf_spec_step_stats_reset();   // per-generation, so a band reports N samples

    int generated = 0;
    double gen_start = now_seconds();
    for (int g = 0; g < gen_n; g++) {
        int best = qf_argmax_token();
        if (best < 0) { fprintf(stderr, "argmax failed\n"); return 1; }
        if (spec) {
            int out[8], nacc = 0;
            int nc = qf_spec_round(&model, best, pos, spec, out, &nacc);
            if (nc < 1) { fprintf(stderr, "spec round failed\n"); return 1; }
            spec_rounds++; spec_drafted += spec; spec_accepted += nacc;
            pos += nc;
            for (int i = 0; i < nc && generated < gen_n; i++) {
                fprintf(stderr, "%d ", out[i]);
                generated++;
                if (qf_is_stop(out[i])) { g = gen_n; break; }
            }
            g += nc - 1;
            continue;
        }
        fprintf(stderr, "%d ", best);
        fflush(stderr);
        if (qf_is_stop(best)) break;
        if (qf_graph_step(&model, best, pos++) != 0) { fprintf(stderr, "decode failed at gen step %d\n", g); return 1; }
        generated++;
    }
    double gen_elapsed = now_seconds() - gen_start;
    fprintf(stderr, "\ngeneration throughput: %.2f tok/s (%d tokens, %.3fs)\n",
            generated ? generated / gen_elapsed : 0.0, generated, gen_elapsed);
    if (spec && spec_rounds) {
        fprintf(stderr, "spec: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "
                        "%.2f tokens/round\n", spec_rounds, spec_drafted, spec_accepted,
                100.0 * (double)spec_accepted / (double)spec_drafted,
                (double)generated / (double)spec_rounds);
        long sn[8] = {0}, sh[8] = {0};
        qf_spec_step_stats(sn, sh, 8);
        for (int i = 0; i < spec && i < 8; i++)
            if (sn[i]) fprintf(stderr, "spec step %d: %ld/%ld correct (%.1f%%)\n",
                               i + 1, sh[i], sn[i], 100.0 * (double)sh[i] / (double)sn[i]);
        double dms = 0, vms = 0, cms = 0;
        qf_spec_phase_ms(&dms, &vms, &cms);
        fprintf(stderr, "spec phases: draft %.1f ms (%.1f/round), verify %.1f ms (%.1f/round), "
                        "commit %.1f ms (%.1f/round)\n",
                dms, dms / spec_rounds, vms, vms / spec_rounds, cms, cms / spec_rounds);
    }
    }
    fprintf(stderr, "done\n");
    qfd_timing_report(stderr);
    qf_forward_shutdown();
    return 0;
}
