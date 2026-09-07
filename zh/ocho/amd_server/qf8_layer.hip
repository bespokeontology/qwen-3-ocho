// qf8_layer.hip - shared-expert fast path, output head, and opt-in section
// timing hooks for the session-08 dense path (gfx906).
#include "qf8_dense.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// silu(eg) * eu, in place on eg. 640 elems.
__global__ void k8_silu_mul(float *__restrict__ g, const float *__restrict__ u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < QF8_NFF) g[i] = qf8_silu(g[i]) * u[i];
}
// y[i] += sigmoid(g1[0]) * ed[i]
__global__ void k8_shexp_add(float *__restrict__ y, const float *__restrict__ ed,
                             const float *__restrict__ g1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < QF8_NEMBD) y[i] += qf8_sig(g1[0]) * ed[i];
}

// M-row shared expert: ONE weight stream for the whole row subset (gate/up/
// gate_inp fused, then down), instead of a per-row call. Ported shape from the
// M-GEMV work; the elementwise kernels stay per row (they are tiny).
void qf8_shexp_M(const Qf8ShexpW *w, const float *mixed, int M,
                 float *eg, float *eu, float *g1, float *ed, float *y2560,
                 int eg_stride, int y_stride, hipStream_t s) {
    Qf8GemvMulti m;
    m.W[0] = w->gate;     m.y[0] = eg; m.rows[0] = QF8_NFF;
    m.W[1] = w->up;       m.y[1] = eu; m.rows[1] = QF8_NFF;
    m.W[2] = w->gate_inp; m.y[2] = g1; m.rows[2] = 1;
    m.n = 3;
    qf8_gemv_multi_M(&m, mixed, QF8_NEMBD, M, s);          // one weight read, M rows
    for (int r = 0; r < M; r++)
        QF8_LAUNCH(k8_silu_mul, (QF8_NFF + 255) / 256, 256, 0, s,
                   eg + (size_t)r * QF8_NFF, eu + (size_t)r * QF8_NFF);
    qf8_gemv_bf16_M(w->down, eg, ed, QF8_NEMBD, QF8_NFF, M, 1.f, 0.f, s);
    for (int r = 0; r < M; r++)
        QF8_LAUNCH(k8_shexp_add, (QF8_NEMBD + 255) / 256, 256, 0, s,
                   y2560 + (size_t)r * y_stride, ed + (size_t)r * QF8_NEMBD, g1 + r);
    (void)eg_stride;
}

void qf8_shexp(const Qf8ShexpW *w, const float *mixed,
               float *eg, float *eu, float *g1, float *ed, float *y2560,
               hipStream_t s) {
    Qf8GemvMulti m;
    m.W[0] = w->gate;     m.y[0] = eg; m.rows[0] = QF8_NFF;
    m.W[1] = w->up;       m.y[1] = eu; m.rows[1] = QF8_NFF;
    m.W[2] = w->gate_inp; m.y[2] = g1; m.rows[2] = 1;
    m.n = 3;
    qf8_gemv_multi(&m, mixed, QF8_NEMBD, s);
    QF8_LAUNCH(k8_silu_mul, (QF8_NFF + 255) / 256, 256, 0, s, eg, eu);
    qf8_gemv_bf16(w->down, eg, ed, QF8_NEMBD, QF8_NFF, 1.f, 0.f, s);
    QF8_LAUNCH(k8_shexp_add, (QF8_NEMBD + 255) / 256, 256, 0, s, y2560, ed, g1);
}

void qf8_out_head(const uint16_t *ohc_norm, const uint16_t *ohc_down,
                  const uint16_t *ohc_up, const uint16_t *lm_head,
                  const float *R, float *normed, float *hc_d, float *mixed,
                  uint16_t *x_bf16_scratch, float *logits, hipStream_t s) {
    // output HC mixer (GatedResidual without inject) -> mixed; NO output norm.
    qf8_hc_pre(normed, hc_d, mixed, 0, R, ohc_norm, ohc_down, ohc_up, 0, s);
    qf8_lm_head(lm_head, mixed, logits, x_bf16_scratch, s);
}

// ---- timing hooks -----------------------------------------------------------
int qf8_prof_init(Qf8Prof *p) {
    memset(p, 0, sizeof(*p));
    // Sentinel as well as env: the systemd EnvironmentFile is root-owned on
    // the target box, so /tmp/qf_prof.on is the only way to turn this on for
    // the running unit.
    const char *e = getenv("QF8_PROFILE");
    p->on = (e && e[0] == '1') || (access("/tmp/qf_prof.on", F_OK) == 0);
    if (!p->on) return 0;
    for (int i = 0; i < QF8_PROF_SLOTS; i++)
        if (hipEventCreate(&p->ev[i]) != hipSuccess) return -1;
    return 0;
}
void qf8_prof_free(Qf8Prof *p) {
    if (!p || !p->on) return;
    for (int i = 0; i < QF8_PROF_SLOTS; i++) hipEventDestroy(p->ev[i]);
    p->on = 0;
}
void qf8_prof_mark(Qf8Prof *p, int tag, hipStream_t s) {
    if (!p->on || p->n >= QF8_PROF_SLOTS) return;
    p->tag[p->n] = tag;
    hipEventRecord(p->ev[p->n], s);
    p->n++;
}
// Print the accumulated section totals without touching the mark ring, so the
// folding can happen once per token on the stage stream and the printing can
// happen later, off the hot path, from the server thread.
void qf8_prof_print(Qf8Prof *p, FILE *out, const char *label) {
    static const char *names[QF8_SEC_N] = {
        "hc_attn", "mixer", "inject1", "hc_ffn", "router", "exp_svc",
        "experts", "shexp", "out_head", "ple",
        "exp_down", "gdn_recur", "gdn_out", "gateup2", "down2"
    };
    if (!p->on || !out) return;
    long steps = 0;
    for (int t = 0; t < QF8_SEC_N; t++) steps = p->acc_n[t] > steps ? p->acc_n[t] : steps;
    double tot = 0.0;
    for (int t = 0; t < QF8_SEC_N; t++) tot += p->acc_ms[t];
    fprintf(out, "  %s  total %.2f ms over %ld intervals\n", label, tot, steps);
    for (int t = 0; t < QF8_SEC_N; t++)
        if (p->acc_n[t])
            fprintf(out, "    %-9s %8.2f ms  %5.1f%%  (%ld calls, %.1f us avg)\n",
                    names[t], p->acc_ms[t], tot > 0 ? 100.0 * p->acc_ms[t] / tot : 0.0,
                    p->acc_n[t], 1000.0 * p->acc_ms[t] / (double)p->acc_n[t]);
}

void qf8_prof_report(Qf8Prof *p, hipStream_t s, FILE *out, int print) {
    static const char *names[QF8_SEC_N] = {
        "hc_attn", "mixer", "inject1", "hc_ffn", "router", "exp_svc",
        "experts", "shexp", "out_head", "ple",
        "exp_down", "gdn_recur", "gdn_out", "gateup2", "down2"
    };
    if (!p->on || p->n < 2) return;
    hipStreamSynchronize(s);                    // OFF hot path only
    for (int i = 0; i + 1 < p->n; i++) {
        float ms = 0.f;
        if (hipEventElapsedTime(&ms, p->ev[i], p->ev[i + 1]) == hipSuccess) {
            int t = p->tag[i];
            if (t >= 0 && t < QF8_SEC_N) { p->acc_ms[t] += ms; p->acc_n[t]++; }
        }
    }
    if (print && out) {
        fprintf(out, "qf8 profile (cumulative):\n");
        for (int t = 0; t < QF8_SEC_N; t++)
            if (p->acc_n[t])
                fprintf(out, "  %-8s %8.3f ms  (%ld sections, %.3f us avg)\n",
                        names[t], p->acc_ms[t], p->acc_n[t],
                        1000.0 * p->acc_ms[t] / (double)p->acc_n[t]);
    }
    p->n = 0;
}
