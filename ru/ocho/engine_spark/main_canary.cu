// main_canary.cu - production-geometry correctness canary runner.
//
// Loads synth_canary/ (production widths, reduced counts: 4 layers, 16
// experts, 4096 vocab) through the SAME production path as main.cu, consumes
// the fixed prompt, and emits JSON that tools/canary_check.py diffs against
// the HF reference in synth_canary/ref.json.
//
// This is a correctness instrument, not a benchmark: it never reports tok/s
// and it never loads the real checkpoint. Seconds per run, no 78 GiB reload.
//
// Per-layer taps are compiled in only for this target and are additionally
// env-gated (QF_TAP=1), so no probe can reach a production build.
#include "qwenflash.h"
#include "cuda/qf_decode_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

extern int qf_route_error(void);
extern "C" int qf_fp4tc_enabled(void);
extern "C" int qf_fp4tc_supported(void);
extern int qf_forward_init(QfModel *m);
extern int qf_decode_step(QfModel *m, int token, long pos);
extern float *qf_last_logits(void);
extern void qf_forward_shutdown(void);

#ifndef CANARY_DIR
#define CANARY_DIR "./synth_canary"
#endif

// Must match TOKENS in tools/build_canary.py
static int g_prompt[] = {11, 47, 210, 1999, 3, 777, 1234, 88};
static int *g_prompt_p = g_prompt;
static int g_n_prompt = (int)(sizeof(g_prompt) / sizeof(g_prompt[0]));

// Set to 1 only for the final prompt token, so the taps record the same
// quantity the HF reference hook records: the last token's layer output.
int qf_canary_record = 0;

static FILE *g_tapf = NULL;
static FILE *g_vecf = NULL;

// Called from qf.cu's QF_TAP sites (canary target only). Copies one device
// buffer to the host and records its shape-independent statistics.
void qf_canary_tap(const char *tag, int il, const float *dev, int n) {
    static float *h = NULL;
    static int cap = 0;
    if (!g_tapf) return;
    if (n > cap) { free(h); h = (float *)malloc((size_t)n * 4); cap = n; }
    if (!h) return;
    if (cudaMemcpy(h, dev, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    double absum = 0.0, mean = 0.0;
    int nonfinite = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(h[i])) { nonfinite++; continue; }
        absum += fabs(h[i]);
        mean += h[i];
    }
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++)
        if (isfinite(h[i])) var += (h[i] - mean) * (h[i] - mean);
    var /= n;
    fprintf(g_tapf, "{\"tag\": \"%s\", \"il\": %d, \"n\": %d, \"nonfinite\": %d, "
                    "\"absum\": %.4f, \"std\": %.6f, \"first8\": [",
            tag, il, n, nonfinite, absum, sqrt(var));
    for (int i = 0; i < 8 && i < n; i++)
        fprintf(g_tapf, "%s%.6f", i ? ", " : "", h[i]);
    fprintf(g_tapf, "]}\n");

    // Full-vector dump. mean|x| alone cannot see a sign flip, a permutation or
    // a wrong dot product - it only sees magnitude. tools/canary_check.py uses
    // these to report cosine similarity and max abs difference against the
    // reference vectors saved by tools/build_canary.py.
    if (g_vecf) {
        uint32_t tl = (uint32_t)strlen(tag), un = (uint32_t)n;
        int32_t li = il;
        fwrite(&tl, 4, 1, g_vecf);
        fwrite(tag, 1, tl, g_vecf);
        fwrite(&li, 4, 1, g_vecf);
        fwrite(&un, 4, 1, g_vecf);
        fwrite(h, 4, (size_t)n, g_vecf);
    }
}

int main(void) {
    QfModel model;
    // Model dir and prompt are overridable so the SAME tapped binary can be
    // built at production dims and pointed at the real checkpoint. The stages
    // that are pure linear algebra or table gathers (embed, router, ple_emb,
    // routed experts) can then be checked against numpy computed from the real
    // weights, with no reference model needed.
    const char *dir = getenv("QF_CANARY_MODEL_DIR");
    if (!dir) dir = CANARY_DIR;
    if (const char *pr = getenv("QF_CANARY_PROMPT")) {
        static int toks[512];
        int n = 0;
        const char *q = pr;
        while (*q && n < 512) {
            while (*q == ' ' || *q == ',') q++;
            if (!*q) break;
            toks[n++] = (int)strtol(q, (char **)&q, 10);
        }
        if (n > 0) { g_prompt_p = toks; g_n_prompt = n; }
    }
    fprintf(stderr, "canary: loading %s\n", dir);
    if (qf_model_load(&model, dir) != 0) {
        fprintf(stderr, "canary: load FAILED\n");
        return 1;
    }
    if (qf_forward_init(&model) != 0) {
        fprintf(stderr, "canary: forward init FAILED\n");
        return 1;
    }
    fprintf(stderr, "canary: loaded, %d layers, vocab %d, prompt %d tokens\n",
            model.cfg.n_layer, model.cfg.n_vocab, g_n_prompt);

    if (const char *tp = getenv("QF_CANARY_TAPFILE")) {
        g_tapf = fopen(tp, "w");
        if (!g_tapf) fprintf(stderr, "canary: cannot open tap file %s\n", tp);
    }
    if (const char *vp = getenv("QF_CANARY_VECFILE")) {
        g_vecf = fopen(vp, "wb");
        if (!g_vecf) fprintf(stderr, "canary: cannot open vec file %s\n", vp);
    }

    long pos = 0;
    for (int i = 0; i < g_n_prompt; i++) {
        qf_canary_record = (i == g_n_prompt - 1);
        if (qf_decode_step(&model, g_prompt_p[i], pos++) != 0) {
            fprintf(stderr, "canary: decode failed at prompt token %d\n", i);
            return 1;
        }
    }
    qf_canary_record = 0;

    if (getenv("QF_MTP")) {
        extern int qf_mtp_load(QfModel *m);
        extern int qf_mtp_step(QfModel *m, int token, long pos,
                               const float *hc_in, float *hc_out, float *logits_out);
        extern float *qf_decode_hc_streams(void);
        if (qf_mtp_load(&model) != 0) { fprintf(stderr, "canary: mtp load failed\n"); return 1; }
        // Hand the trunk's pre-mixer streams to one draft step, exactly as
        // sglang hands over spec_info.hidden_states.
        float *hc = qf_decode_hc_streams();
        float *dlog = NULL;
        if (cudaMalloc(&dlog, (size_t)model.cfg.n_vocab * 4) != cudaSuccess) return 1;
        // A draft step consumes embed(the token just committed) together with
        // the streams from the forward that produced it. The token just
        // committed is the trunk's own argmax, not the last prompt token.
        float *tl = (float *)malloc((size_t)model.cfg.n_vocab * 4);
        cudaMemcpy(tl, qf_last_logits(), (size_t)model.cfg.n_vocab * 4, cudaMemcpyDeviceToHost);
        int t_next = 0;
        for (int v = 1; v < model.cfg.n_vocab; v++) if (tl[v] > tl[t_next]) t_next = v;
        fprintf(stderr, "mtp: trunk committed token %d (%.4f)\n", t_next, tl[t_next]);
        qf_canary_record = 1;
        if (qf_mtp_step(&model, t_next, (long)g_n_prompt, hc, NULL, dlog) != 0) return 1;
        qf_canary_record = 0;
        float *hl = (float *)malloc((size_t)model.cfg.n_vocab * 4);
        cudaMemcpy(hl, dlog, (size_t)model.cfg.n_vocab * 4, cudaMemcpyDeviceToHost);
        int b = 0, b2 = 0;
        for (int v = 1; v < model.cfg.n_vocab; v++) if (hl[v] > hl[b]) b = v;
        for (int v = 1; v < model.cfg.n_vocab; v++)
            if (v != b && hl[v] > hl[b2]) b2 = v;
        fprintf(stderr, "mtp: draft top1=%d (%.4f)  top2=%d (%.4f)\n",
                b, hl[b], b2, hl[b2]);
        free(hl); free(tl);
    }

    {
        fprintf(stderr, "canary: fp4tc supported=%d enabled=%d route_error=%d\n",
                qf_fp4tc_supported(), qf_fp4tc_enabled(), qf_route_error());
    }

    if (g_tapf) { fclose(g_tapf); g_tapf = NULL; }
    if (g_vecf) { fclose(g_vecf); g_vecf = NULL; }

    int nv = model.cfg.n_vocab;
    float *h = (float *)malloc((size_t)nv * sizeof(float));
    if (cudaMemcpy(h, qf_last_logits(), (size_t)nv * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "canary: logits copy failed\n");
        return 1;
    }

    int nan = 0;
    double absum = 0.0, mean = 0.0;
    for (int v = 0; v < nv; v++) {
        if (isnan(h[v]) || isinf(h[v])) nan++;
        else { absum += fabs(h[v]); mean += h[v]; }
    }
    mean /= nv;
    double var = 0.0;
    for (int v = 0; v < nv; v++)
        if (!isnan(h[v]) && !isinf(h[v])) var += (h[v] - mean) * (h[v] - mean);
    var /= nv;

    printf("{\n");
    printf(" \"nonfinite\": %d,\n", nan);
    printf(" \"final_logit_absum\": %.4f,\n", absum);
    printf(" \"final_logit_std\": %.6f,\n", sqrt(var));
    printf(" \"final_top\": [");
    {
        int used[8];
        for (int k = 0; k < 8; k++) {
            int best = -1;
            for (int v = 0; v < nv; v++) {
                int skip = 0;
                for (int j = 0; j < k; j++) if (used[j] == v) skip = 1;
                if (skip || isnan(h[v])) continue;
                if (best < 0 || h[v] > h[best]) best = v;
            }
            used[k] = best;
            printf("%s[%d, %.4f]", k ? ", " : "", best, best >= 0 ? h[best] : 0.0f);
        }
    }
    printf("]\n}\n");

    free(h);
    qf_forward_shutdown();
    return 0;
}
