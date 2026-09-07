// engine_stub.cpp - host-only link-check stub for the QF engine ABI.
//
// Exists ONLY so `make server-check` can compile and link the full server on
// hosts without a CUDA toolchain. It is never linked into the production
// server (qf_server links the real engine) and must never be run against a
// real model dir. No GPU, no model files, no sampling correctness here.
#include "qwenflash.h"

#include <stdlib.h>
#include <string.h>

int qf_model_load(QfModel *m, const char *) {
    memset(m, 0, sizeof(*m));
    m->cfg.n_vocab = 248320;
    m->cfg.n_embd = 2560;
    m->cfg.n_layer = 48;
    return 0;
}

void qf_model_free(QfModel *) {}

int qf_route_error(void) { return 0; }

int qf_generate_init(QfModel *) { return 0; }

int qf_generate(QfModel *m, const int *, int n_prompt, int n_gen,
                float, int, float,
                int (*cb)(int, void *), void *ud) {
    (void)m;
    // Emit a few fixed token ids then "stop"; honors cancellation via cb.
    static const int kIds[] = {9707, 11, 1879, 330, 13339};
    int n = (int)(sizeof(kIds) / sizeof(kIds[0]));
    if (n_gen < n) n = n_gen;
    (void)n_prompt;
    for (int i = 0; i < n; i++)
        if (cb && cb(kIds[i], ud)) return 1;
    return 0;
}
