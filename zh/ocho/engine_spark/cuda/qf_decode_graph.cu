// qf_decode_graph.cu - capture/replay for the fixed decode schedule
#include "../qwenflash.h"
#include "qf_attn_fused.h"
#include "qf_decode_graph.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

extern int qf_decode_body(QfModel *m, int for_capture);
extern uint32_t qf_decode_next_seq(void);
extern void qf_decode_set_params(int token, long pos, uint32_t seq);
extern cudaStream_t qf_decode_stream(void);
extern void qf_ple_stage_current(QfModel *m);
extern void qf_hist_push(int token);
extern void qf_hist_reset(void);
extern void qf_state_reset(QfModel *m);
extern int qf_route_service_token(QfModel *m);
extern int qf_decode_step(QfModel *m, int token, long pos);
extern void qf_ple_reset(cudaStream_t s);

static cudaGraphExec_t g_exec = NULL;
static cudaEvent_t g_ev0 = NULL, g_ev1 = NULL;
static int g_ready = 0;
static int g_ev_pending = 0;
static long g_steps = 0;
static float g_last_ms = 0.f;
static double g_ms_total = 0.0;

int qf_graph_ready(void) { return g_ready; }

static void timing_collect(int wait) {
    if (!g_ev_pending) return;
    if (!wait && cudaEventQuery(g_ev1) != cudaSuccess) return;
    if (wait) cudaEventSynchronize(g_ev1);
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, g_ev0, g_ev1) == cudaSuccess) {
        g_last_ms = ms;
        g_ms_total += ms;
    }
    g_ev_pending = 0;
}

int qf_graph_capture(QfModel *m) {
    if (g_ready) return 0;
    cudaStream_t s = qf_decode_stream();
    if (!s) return -1;
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    if (!g_ev0) {
        if (cudaEventCreate(&g_ev0) != cudaSuccess || cudaEventCreate(&g_ev1) != cudaSuccess)
            return -1;
    }
    cudaGraph_t graph = NULL;
    cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        fprintf(stderr, "qf_graph: begin capture failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    int rc = qf_decode_body(m, 1);
    e = cudaStreamEndCapture(s, &graph);
    if (rc != 0 || e != cudaSuccess || !graph) {
        fprintf(stderr, "qf_graph: capture failed (rc=%d, %s); eager path stays active\n",
                rc, cudaGetErrorString(e));
        return -1;
    }
    e = cudaGraphInstantiateWithFlags(&g_exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
        fprintf(stderr, "qf_graph: instantiate failed: %s; eager path stays active\n",
                cudaGetErrorString(e));
        g_exec = NULL;
        return -1;
    }
    g_ready = 1;
    fprintf(stderr, "qf_graph: captured the full decode step\n");
    return 0;
}

int qf_graph_step(QfModel *m, int token, long pos) {
    if (!g_ready) return qf_decode_step(m, token, pos);
    cudaStream_t s = qf_decode_stream();
    timing_collect(0);
    // Same ordering rule as qf_decode_step: the current token must be in the
    // history window before the n-gram hash is computed.
    qf_hist_push(token);
    qf_ple_stage_current(m);
    qf_decode_set_params(token, pos, qf_decode_next_seq());
    cudaEventRecord(g_ev0, s);
    cudaGraphLaunch(g_exec, s);
    cudaEventRecord(g_ev1, s);
    g_ev_pending = 1;
    int rc = qf_route_service_token(m);
    g_steps++;
    return rc;
}

int qf_kv_reset(QfModel *m) {
    cudaStream_t s = qf_decode_stream();
    qf_state_reset(m);
    qf_ple_reset(s);
    qf_hist_reset();
    qf_decode_set_params(0, 0, qf_decode_next_seq());
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -1;
}

float qf_graph_last_ms(void) {
    timing_collect(1);
    return g_last_ms;
}

void qf_graph_stats(long *steps, double *avg_ms) {
    timing_collect(1);
    if (steps) *steps = g_steps;
    if (avg_ms) *avg_ms = g_steps ? g_ms_total / (double)g_steps : 0.0;
}

void qf_graph_destroy(void) {
    if (g_ready) cudaStreamSynchronize(qf_decode_stream());
    if (g_exec) cudaGraphExecDestroy(g_exec);
    if (g_ev0) cudaEventDestroy(g_ev0);
    if (g_ev1) cudaEventDestroy(g_ev1);
    g_exec = NULL;
    g_ev0 = g_ev1 = NULL;
    g_ready = 0;
    g_ev_pending = 0;
    g_steps = 0;
    g_last_ms = 0.f;
    g_ms_total = 0.0;
}
