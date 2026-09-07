// engine_stub.cpp - host-build stub of the frozen QfHip4 API. No HIP, no
// GPU: lets the full server compile, link and serve synthetic requests on
// any machine for integration testing. Selected at link time (the
// server-target build links the real engine objects instead).
//
// Logits are fabricated deterministically from (index, pos) so runs are
// reproducible: logits[i] = ((i*2654435761u ^ pos) % 1000) / 1000.
#include "qf_hip4.h"

#include <chrono>
#include <cstdio>
#include <cstring>

struct QfHip4 {
    uint64_t tokens_decoded = 0;
    double   load_seconds = 0;
    double   last_step_ms = 0;
    long     max_context = 262144;
    std::chrono::steady_clock::time_point t0;
};

int qf_hip4_init(QfHip4 **out, const char *model_dir) {
    if (!model_dir || !*model_dir) {
        fprintf(stderr, "engine_stub: null model_dir\n");
        return -1;
    }
    auto t0 = std::chrono::steady_clock::now();
    QfHip4 *p = new QfHip4();
    p->t0 = t0;
    p->load_seconds = 0.001;  // fake: no weights loaded
    fprintf(stderr, "engine_stub: init model_dir=%s (4 fake gfx906 agents, no weights)\n",
            model_dir);
    *out = p;
    return 0;
}

static void stub_logits(QfHip4 *p, long pos, float *logits_out) {
    for (size_t i = 0; i < QF4_NVOCAB; i++)
        logits_out[i] = (float)((((uint32_t)i * 2654435761u) ^ (uint32_t)pos) % 1000) / 1000.0f;
    // Dominant peak on a low id so sampled/argmax tokens land inside small
    // test fixture vocabs (integration runs produce real text, not "").
    logits_out[(size_t)((pos * 31 + 7) % 16)] += 10.0f;
    p->tokens_decoded++;
    p->last_step_ms = 1.0;
}

int qf_hip4_submit(QfHip4 *p, int token, long pos) {
    (void)p; (void)token; (void)pos;
    return 0;
}

int qf_hip4_submit_prefill(QfHip4 *p, int token, long pos) {
    (void)p; (void)token; (void)pos;
    return 0;
}

int qf_hip4_wait(QfHip4 *p, long pos, float *logits_out) {
    if (!p || !logits_out) return -1;
    stub_logits(p, pos, logits_out);
    return 0;
}

int qf_hip4_decode_step(QfHip4 *p, int token, long pos, float *logits_out) {
    if (qf_hip4_submit(p, token, pos) != 0) return -1;
    return qf_hip4_wait(p, pos, logits_out);
}

void qf_hip4_reset(QfHip4 *p) {
    if (p) p->tokens_decoded = 0;
}

int qf_hip4_stats(QfHip4 *p, QfHip4Stats *out) {
    if (!p || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->load_seconds = p->load_seconds;
    out->tokens_decoded = p->tokens_decoded;
    out->last_step_ms = p->last_step_ms;
    for (int i = 0; i < QF4_NGPU; i++)
        out->hbm_bytes[i] = (size_t)15000 * 1024 * 1024;  // fake ~15 GiB/agent
    out->handoff_mode = 1;  // pretend pinned staging
    out->max_context = p->max_context;
    return 0;
}

void qf_hip4_free(QfHip4 *p) {
    delete p;
    fprintf(stderr, "engine_stub: freed\n");
}
