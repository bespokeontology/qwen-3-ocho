// qf_hip4_pipeline.cpp - host orchestration for the 4-GPU pipeline.
//
// Pure host C++ (no HIP headers): every HIP interaction happens inside
// qf_hip4_stage.hip behind the qf4_stage_* API. This file owns:
//   - the shared read-only QfStore (mmap'd weight shards),
//   - one worker thread per GPU (stage threads),
//   - the inter-GPU handoff mailboxes (2-deep ring each, mutex/cond),
//   - the handoff mode decision (QF4_HANDOFF=auto|pinned|p2p, default auto),
//   - the frozen public API (qf_hip4.h): submit / submit_prefill / wait /
//     decode_step / reset / stats.
//
// Token flow per decode step (token t at position p):
//   driver -> submit queue -> GPU0 (embed, layers 0-11)  -> mb0
//          -> GPU1 (layers 12-23) -> mb1 -> GPU2 (24-35) -> mb2
//          -> GPU3 (36-47, output HC, lm_head) -> logits mailbox -> driver
// Because each stage is an independent thread and the mailboxes are 2-deep,
// GPU0 can start token t+1 while GPU1 still works on token t.
//
// Handoff modes (wave4 session 09):
//   pinned: producer D2H R into a pinned staging buffer, mailbox carries the
//     payload, consumer H2D. Works on any topology (pure-PCIe MI50).
//   p2p:    producer copies R into one of its TWO fixed 40 KiB device handoff
//     buffers and records the slot event (qf4_stage_emit_residual); the
//     mailbox carries only (token, pos, flags) with slot = pos % 2; the
//     consumer waits the producer event and hipMemcpyPeerAsyncs 40 KiB
//     (qf4_stage_recv_residual_dev). With MB_DEPTH=2 this double buffering is
//     race-free: the producer cannot start token t+2's emit (and thus cannot
//     reuse slot t%2) before the consumer has popped token t -- mailbox
//     backpressure guarantees it -- and the consumer issues the slot-t peer
//     copy immediately after popping t, a full stage iteration before t+2's
//     emit can be queued.
//   auto (default): probe all three boundaries with qf4_peer_available; P2P
//     only if ALL probe true, else pinned everywhere. ROCm 5.7 / MI50 P2P
//     needs large-BAR PCIe (or xGMI), so the runtime probe is the decision.
// The logits mailbox stays host-staged in both modes (1 MB, once per token).
//
// Reset: qf_hip4_reset submits an in-band sentinel (token = -1). Each stage
// thread runs qf4_stage_reset and forwards it; stage 3 pushes a logits_mb
// entry with pos = -1 and no payload as the ack, which qf_hip4_reset pops.
// Only legal when the pipeline is idle (server contract).
#include "qf_hip4.h"

// Defined in qf_expert_slots.cpp (declared in qf_nvfp4_wave64.h, which pulls
// in HIP headers this translation unit deliberately avoids).
extern "C" void qf4_expcache_stats(unsigned long long *calls,
                                   unsigned long long *miss,
                                   unsigned long long *bytes);

#include <pthread.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MB_DEPTH 2

// Mailbox flags
#define QF4F_WANT_LOGITS 1    // run lm_head and push to the logits mailbox

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv_space, cv_data;
    float  *buf[MB_DEPTH];      // payload slots (NULL for token-only queues)
    long    pos[MB_DEPTH];
    int     token[MB_DEPTH];
    int     flags[MB_DEPTH];
    long    prod, cons;         // monotone sequence counters
    int     stop;
    size_t  payload;            // bytes per slot (0 = token-only)
} Qf4Mb;

static void mb_init(Qf4Mb *mb, size_t payload) {
    memset(mb, 0, sizeof(*mb));
    pthread_mutex_init(&mb->mu, NULL);
    pthread_cond_init(&mb->cv_space, NULL);
    pthread_cond_init(&mb->cv_data, NULL);
    mb->payload = payload;
    for (int i = 0; i < MB_DEPTH; i++)
        if (payload) mb->buf[i] = (float *)malloc(payload);
}

static void mb_destroy(Qf4Mb *mb) {
    for (int i = 0; i < MB_DEPTH; i++) free(mb->buf[i]);
    pthread_mutex_destroy(&mb->mu);
    pthread_cond_destroy(&mb->cv_space);
    pthread_cond_destroy(&mb->cv_data);
}

static void mb_stop(Qf4Mb *mb) {
    pthread_mutex_lock(&mb->mu);
    mb->stop = 1;
    pthread_cond_broadcast(&mb->cv_space);
    pthread_cond_broadcast(&mb->cv_data);
    pthread_mutex_unlock(&mb->mu);
}

// Blocks until a slot is free, copies payload in (NULL payload = none),
// publishes.
static void mb_push(Qf4Mb *mb, int token, long pos, int flags, const float *payload) {
    pthread_mutex_lock(&mb->mu);
    while (!mb->stop && mb->prod - mb->cons >= MB_DEPTH)
        pthread_cond_wait(&mb->cv_space, &mb->mu);
    if (!mb->stop) {
        int s = (int)(mb->prod % MB_DEPTH);
        if (mb->payload && payload) memcpy(mb->buf[s], payload, mb->payload);
        mb->token[s] = token;
        mb->pos[s] = pos;
        mb->flags[s] = flags;
        mb->prod++;
        pthread_cond_signal(&mb->cv_data);
    }
    pthread_mutex_unlock(&mb->mu);
}

// Blocks until data is available, copies payload out (NULL payload = skip),
// releases the slot. Returns 0 on success, -1 when stopped.
static int mb_pop(Qf4Mb *mb, int *token, long *pos, int *flags, float *payload) {
    pthread_mutex_lock(&mb->mu);
    while (!mb->stop && mb->prod == mb->cons)
        pthread_cond_wait(&mb->cv_data, &mb->mu);
    if (mb->stop) { pthread_mutex_unlock(&mb->mu); return -1; }
    int s = (int)(mb->cons % MB_DEPTH);
    if (mb->payload && payload) memcpy(payload, mb->buf[s], mb->payload);
    *token = mb->token[s];
    *pos = mb->pos[s];
    *flags = mb->flags[s];
    mb->cons++;
    pthread_cond_signal(&mb->cv_space);
    pthread_mutex_unlock(&mb->mu);
    return 0;
}

struct Qf4Targ { QfHip4 *p; int g; };

struct QfHip4 {
    QfStore   store;
    Qf4Stage *st[QF4_NGPU];
    pthread_t th[QF4_NGPU];
    int       th_up[QF4_NGPU];
    Qf4Targ   targ[QF4_NGPU];
    Qf4Mb     submit;                       // token-only queue into GPU0
    Qf4Mb     mb[QF4_NGPU - 1];             // residual boundaries 0|1|2
    Qf4Mb     logits_mb;                    // GPU3 -> driver
    float    *h_res[QF4_NGPU];              // pinned residual staging (pinned mode)
    float    *h_logits;                     // pinned GPU3 logits staging
    int       handoff;                      // QF4_HANDOFF_* in use
    long      max_context;                  // QF_MAX_CONTEXT in effect
    int       err;
    // stats
    double    load_seconds;
    uint64_t  tokens_decoded;
    double    last_step_ms;
};

static void stage_fail(QfHip4 *p, int g) {
    fprintf(stderr, "qf4: stage %d failed, stopping pipeline\n", g);
    p->err = 1;
    mb_stop(&p->submit);
    for (int i = 0; i < QF4_NGPU - 1; i++) mb_stop(&p->mb[i]);
    mb_stop(&p->logits_mb);
}

// ---- per-stage wall-clock accounting (inert unless /tmp/qf_prof.on) -------
//
// Gate 2 asks how much of the ~125 ms/token is staging versus compute. That is
// a host-side question - who is waiting on whom - so it is answered with host
// timers around the four things a stage thread actually does, not with GPU
// events. `wait` is time blocked in mb_pop, i.e. idle because an upstream
// stage has not produced yet; on a serial pipeline every stage but one is
// waiting, so `wait` is the pipeline bubble and `run` is the real work.
struct Qf4Prof {
    double wait, recv, run, send;
    long   n;
};
static Qf4Prof g_prof[QF4_NGPU];
static QfHip4 *p_global = NULL;   // profiling only: reach the stages from the dump
static int prof_on(void) {
    static int checked = 0, on = 0;
    if (!checked) { checked = 1; on = (access("/tmp/qf_prof.on", F_OK) == 0); }
    return on;
}

static double now_s(void);

static void *stage_main(void *arg) {
    QfHip4 *p = ((Qf4Targ *)arg)->p;
    int g     = ((Qf4Targ *)arg)->g;
    Qf4Stage *st = p->st[g];
    const int p2p = (p->handoff == QF4_HANDOFF_P2P);
    const int prof = prof_on();
    double tA, tB, tC, tD, tE;

    for (;;) {
        int token, flags; long pos;
        tA = prof ? now_s() : 0.0;
        if (g == 0) {
            if (mb_pop(&p->submit, &token, &pos, &flags, NULL) != 0) break;
            tB = prof ? now_s() : 0.0;
            if (token >= 0 && qf4_stage_set_token(st, token) != 0) {
                stage_fail(p, g); break;
            }
        } else {
            if (mb_pop(&p->mb[g - 1], &token, &pos, &flags,
                       p2p ? NULL : p->h_res[g]) != 0) break;
            tB = prof ? now_s() : 0.0;
            if (token >= 0) {
                int rc;
                if (p2p) {
                    int slot = (int)(pos % 2);   // buffer/event pairing
                    rc = qf4_stage_recv_residual_dev(st, g - 1,
                            qf4_stage_handoff_dev(p->st[g - 1], slot), slot,
                            qf4_stage_handoff_event(p->st[g - 1], slot));
                } else {
                    rc = qf4_stage_push_residual(st, p->h_res[g]);
                }
                if (rc != 0) { stage_fail(p, g); break; }
            }
        }
        tC = prof ? now_s() : 0.0;

        if (token < 0) {
            // Reset sentinel: clear recurrent state, forward downstream.
            if (qf4_stage_reset(st) != 0) { stage_fail(p, g); break; }
        } else {
            int rc = (flags & QF4F_WANT_LOGITS)
                   ? qf4_stage_run(st, pos)
                   : qf4_stage_run_nologits(st, pos);
            if (rc != 0) { stage_fail(p, g); break; }
        }
        tD = prof ? now_s() : 0.0;

        if (g < QF4_NGPU - 1) {
            if (token >= 0) {
                if (p2p) {
                    if (qf4_stage_emit_residual(st, (int)(pos % 2)) != 0) {
                        stage_fail(p, g); break;
                    }
                } else {
                    if (qf4_stage_pull_residual(st, p->h_res[g]) != 0) {
                        stage_fail(p, g); break;
                    }
                }
            }
            mb_push(&p->mb[g], token, pos, flags,
                    p2p ? NULL : (token >= 0 ? p->h_res[g] : NULL));
        } else {
            if (token < 0) {
                mb_push(&p->logits_mb, token, -1, 0, NULL);   // reset ack
            } else if (flags & QF4F_WANT_LOGITS) {
                if (qf4_stage_pull_logits(st, p->h_logits) != 0) {
                    stage_fail(p, g); break;
                }
                mb_push(&p->logits_mb, token, pos, flags, p->h_logits);
            }
            // prefill steps (want_logits=0) produce no logits entry
        }
        if (prof && token >= 0) {
            tE = now_s();
            g_prof[g].wait += tB - tA;      // blocked in mb_pop
            g_prof[g].recv += tC - tB;      // residual into this GPU
            g_prof[g].run  += tD - tC;      // the 12 layers (+ head on g3)
            g_prof[g].send += tE - tD;      // residual/logits out + mb_push
            g_prof[g].n++;
        }
    }
    return NULL;
}

void qf_hip4_prof_dump(void *outv) {
    FILE *out = (FILE *)outv;
    if (!prof_on()) return;
    fprintf(out, "qf4 per-stage wall clock (ms/token, decode steps only):\n");
    fprintf(out, "  gpu   wait    recv     run    send   total    n\n");
    double crit = 0.0;
    for (int g = 0; g < QF4_NGPU; g++) {
        Qf4Prof &q = g_prof[g];
        if (!q.n) continue;
        double k = 1000.0 / (double)q.n;
        double tot = (q.wait + q.recv + q.run + q.send) * k;
        fprintf(out, "  %3d %6.2f  %6.2f  %6.2f  %6.2f  %6.2f  %4ld\n",
                g, q.wait * k, q.recv * k, q.run * k, q.send * k, tot, q.n);
        crit += (q.recv + q.run + q.send) * k;
    }
    fprintf(out, "  sum of non-wait time across stages: %.2f ms/token\n", crit);
    unsigned long long ec, em, eb;
    qf4_expcache_stats(&ec, &em, &eb);
    if (ec) {
        long steps = 0;
        for (int g = 0; g < QF4_NGPU; g++) steps = g_prof[g].n > steps ? g_prof[g].n : steps;
        double tok = steps ? (double)steps : 1.0;
        fprintf(out, "qf4 expert cache: %llu lookups, %llu misses (%.1f%%), "
                     "%.2f GB streamed H2D = %.1f MB/token\n",
                ec, em, 100.0 * (double)em / (double)ec,
                (double)eb / (1024.0 * 1024.0 * 1024.0),
                (double)eb / (1024.0 * 1024.0) / tok);
    }
    fprintf(out, "qf4 per-section GPU time inside `run`:\n");
    for (int g = 0; g < QF4_NGPU; g++) {
        char lbl[16];
        snprintf(lbl, sizeof(lbl), "gpu%d", g);
        qf4_stage_prof_print(p_global ? p_global->st[g] : NULL, out, lbl);
        qf4_stage_hist_print(p_global ? p_global->st[g] : NULL, out, lbl);
    }
}

static double now_s(void) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int qf_hip4_init(QfHip4 **out, const char *model_dir) {
    double t0 = now_s();
    QfHip4 *p = (QfHip4 *)calloc(1, sizeof(QfHip4));
    if (!p) return -1;

    if (qf_store_open(&p->store, model_dir) != 0) {
        fprintf(stderr, "qf4: cannot open store %s\n", model_dir);
        free(p);
        return -1;
    }

    p->max_context = 131072;
    if (const char *v = getenv("QF_MAX_CONTEXT")) p->max_context = atol(v);
    if (p->max_context > 262144) p->max_context = 262144;
    if (p->max_context < 1) p->max_context = 1;

    // Load stages serially from the main thread (weight upload is the slow
    // part; stage functions hipSetDevice() internally). Each GPU gets its
    // 12-layer shard: GPU g owns global layers [12g, 12g+12). Stage init runs
    // the HBM budget guard and prints the per-GPU commitment line.
    for (int g = 0; g < QF4_NGPU; g++) {
        long ltot_env = getenv("QF_LAYERS_TOTAL") ? atol(getenv("QF_LAYERS_TOTAL")) : 0;
        if (ltot_env <= 0) {
            FILE *lf = fopen("/tmp/qf_layers_total", "r");
            if (lf) { if (fscanf(lf, "%ld", &ltot_env) != 1) ltot_env = 0; fclose(lf); }
        }
        if (ltot_env < QF4_NGPU || ltot_env > 48) ltot_env = 48;
        ltot_env -= ltot_env % QF4_NGPU;
        const int nlp = (int)(ltot_env / QF4_NGPU);
        long lbase = getenv("QF_LAYER_BASE") ? atol(getenv("QF_LAYER_BASE")) : -1;
        if (lbase < 0) {
            FILE *bf = fopen("/tmp/qf_layer_base", "r");
            if (bf) { if (fscanf(bf, "%ld", &lbase) != 1) lbase = 0; fclose(bf); }
        }
        if (lbase < 0 || lbase >= 48) lbase = 0;
        lbase -= lbase % QF4_NGPU;
        if (qf4_stage_init(&p->st[g], g, &p->store, (int)lbase + g * nlp) != 0) {
            fprintf(stderr, "qf4: stage %d init failed\n", g);
            qf_hip4_free(p);
            return -1;
        }
        fprintf(stderr, "qf4: GPU %d owns layers %d..%d\n", g,
                g * QF4_STAGE_NL, g * QF4_STAGE_NL + QF4_STAGE_NL - 1);
    }

    // Handoff mode: env QF4_HANDOFF=auto|pinned|p2p (default auto). auto uses
    // P2P only if ALL three boundaries probe true; otherwise pinned
    // everywhere. gfx906/MI50 note: ROCm 5.7 P2P works only with large-BAR
    // PCIe (and xGMI where present), so the auto-probe is the correct runtime
    // decision and pinned host staging is the guaranteed fallback.
    int want = QF4_HANDOFF_AUTO;
    if (const char *v = getenv("QF4_HANDOFF")) {
        if (!strcmp(v, "pinned")) want = QF4_HANDOFF_PINNED;
        else if (!strcmp(v, "p2p")) want = QF4_HANDOFF_P2P;
    }
    // Sentinel override: the systemd EnvironmentFile is root-owned, so
    // /tmp/qf_handoff.pinned forces host staging for an A/B of the P2P path
    // without editing /etc or restarting under a different environment.
    if (access("/tmp/qf_handoff.pinned", F_OK) == 0) {
        want = QF4_HANDOFF_PINNED;
        fprintf(stderr, "qf4: /tmp/qf_handoff.pinned present - forcing pinned handoff\n");
    }
    if (want == QF4_HANDOFF_PINNED) {
        p->handoff = QF4_HANDOFF_PINNED;
    } else {
        int all = 1;
        for (int g = 0; g < QF4_NGPU - 1; g++) {
            int r = qf4_peer_available(g, g + 1);
            if (r < 0) {
                fprintf(stderr, "qf4: peer probe %d->%d failed\n", g, g + 1);
                qf_hip4_free(p);
                return -1;
            }
            if (!r) all = 0;
        }
        p->handoff = all ? QF4_HANDOFF_P2P : QF4_HANDOFF_PINNED;
        if (want == QF4_HANDOFF_P2P && !all)
            fprintf(stderr, "qf4: QF4_HANDOFF=p2p requested but a boundary "
                    "failed the probe; falling back to pinned\n");
    }
    p_global = p;
    fprintf(stderr, "qf4: handoff mode: %s\n",
            p->handoff == QF4_HANDOFF_P2P ? "p2p (peer-to-peer)"
                                          : "pinned (host-staged)");

    const int p2p = (p->handoff == QF4_HANDOFF_P2P);
    mb_init(&p->submit, 0);
    // P2P boundaries are token-only queues (device buffers carry the payload);
    // pinned boundaries carry the 40 KiB residual in the mailbox.
    for (int i = 0; i < QF4_NGPU - 1; i++)
        mb_init(&p->mb[i], p2p ? 0 : QF4_RSIZE * sizeof(float));
    mb_init(&p->logits_mb, (size_t)QF4_NVOCAB * sizeof(float));
    if (!p2p)
        for (int g = 0; g < QF4_NGPU; g++) {
            p->h_res[g] = (float *)qf4_host_pinned(QF4_RSIZE * sizeof(float));
            if (!p->h_res[g]) { fprintf(stderr, "qf4: pinned h_res alloc failed\n"); qf_hip4_free(p); return -1; }
        }
    // Logits stay host-staged in both modes (1 MB, once per token).
    p->h_logits = (float *)qf4_host_pinned((size_t)QF4_NVOCAB * sizeof(float));
    if (!p->h_logits) { fprintf(stderr, "qf4: pinned h_logits alloc failed\n"); qf_hip4_free(p); return -1; }

    for (int g = 0; g < QF4_NGPU; g++) {
        p->targ[g].p = p;
        p->targ[g].g = g;
        if (pthread_create(&p->th[g], NULL, stage_main, &p->targ[g]) != 0) {
            fprintf(stderr, "qf4: pthread_create failed for stage %d\n", g);
            qf_hip4_free(p);
            return -1;
        }
        p->th_up[g] = 1;
    }
    p->load_seconds = now_s() - t0;
    *out = p;
    return 0;
}

int qf_hip4_submit(QfHip4 *p, int token, long pos) {
    if (!p || p->err) return -1;
    mb_push(&p->submit, token, pos, QF4F_WANT_LOGITS, NULL);
    return p->err ? -1 : 0;
}

int qf_hip4_submit_prefill(QfHip4 *p, int token, long pos) {
    if (!p || p->err) return -1;
    // want_logits=0: the flag flows through every boundary mailbox; stage 3
    // runs qf4_stage_run_nologits and does NOT push to the logits mailbox.
    // Steps are still strictly ordered by pos.
    mb_push(&p->submit, token, pos, 0, NULL);
    return p->err ? -1 : 0;
}

int qf_hip4_wait(QfHip4 *p, long pos, float *logits_out) {
    if (!p) return -1;
    // Logits-producing steps are processed in submission order; the next
    // logits mailbox entry must be the requested position.
    int token, flags; long got;
    if (mb_pop(&p->logits_mb, &token, &got, &flags, logits_out) != 0) return -1;
    if (got != pos) {
        fprintf(stderr, "qf4: pipeline order violation: want pos %ld, got %ld\n", pos, got);
        return -1;
    }
    p->tokens_decoded++;
    (void)token; (void)flags;
    return 0;
}

int qf_hip4_decode_step(QfHip4 *p, int token, long pos, float *logits_out) {
    double t0 = now_s();
    if (qf_hip4_submit(p, token, pos) != 0) return -1;
    int rc = qf_hip4_wait(p, pos, logits_out);
    if (rc == 0) p->last_step_ms = (now_s() - t0) * 1e3;
    return rc;
}

void qf_hip4_reset(QfHip4 *p) {
    if (!p || p->err) return;
    // In-band sentinel: token = -1 flows through every stage (each runs
    // qf4_stage_reset and forwards it); stage 3 acks on the logits mailbox
    // with pos = -1 and no payload. Legal only when the pipeline is idle.
    mb_push(&p->submit, -1, -1, 0, NULL);
    int token, flags; long pos;
    // Block popping the logits mailbox until the ack (pos == -1) arrives.
    // On an idle pipeline the ack is the next entry; anything else means the
    // caller violated the idle contract (drain it and keep waiting).
    for (;;) {
        if (mb_pop(&p->logits_mb, &token, &pos, &flags, NULL) != 0) return;
        if (pos == -1) break;
        fprintf(stderr, "qf4: reset discarding stray logits entry (pos %ld); "
                "reset is only legal on an idle pipeline\n", pos);
    }
}

int qf_hip4_stats(QfHip4 *p, QfHip4Stats *out) {
    if (!p || !out) return -1;
    out->load_seconds   = p->load_seconds;
    out->tokens_decoded = p->tokens_decoded;
    out->last_step_ms   = p->last_step_ms;
    for (int g = 0; g < QF4_NGPU; g++)
        out->hbm_bytes[g] = p->st[g] ? qf4_stage_hbm_bytes(p->st[g]) : 0;
    out->handoff_mode = p->handoff;
    out->max_context  = p->max_context;
    return 0;
}

void qf_hip4_free(QfHip4 *p) {
    if (!p) return;
    mb_stop(&p->submit);
    for (int i = 0; i < QF4_NGPU - 1; i++) mb_stop(&p->mb[i]);
    mb_stop(&p->logits_mb);
    for (int g = 0; g < QF4_NGPU; g++)
        if (p->th_up[g]) pthread_join(p->th[g], NULL);
    for (int g = 0; g < QF4_NGPU; g++) {
        if (p->st[g]) qf4_stage_free(p->st[g]);
        if (p->h_res[g]) qf4_host_pinned_free(p->h_res[g]);
    }
    if (p->h_logits) qf4_host_pinned_free(p->h_logits);
    mb_destroy(&p->submit);
    for (int i = 0; i < QF4_NGPU - 1; i++) mb_destroy(&p->mb[i]);
    mb_destroy(&p->logits_mb);
    qf_store_close(&p->store);
    free(p);
}
