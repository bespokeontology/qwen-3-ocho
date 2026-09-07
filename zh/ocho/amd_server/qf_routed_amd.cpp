// qf_routed_amd.cpp - see qf_routed_amd.h. Spark-side; host-only.
// Host syntax check:  g++ -DQF_HOST_CHECK -I. -fsyntax-only qf_routed_amd.cpp
#include "qf_routed_amd.h"
#include "qf_moe_wire.h"
#include <stdio.h>
#include <stdlib.h>

static int g_state = -1;   // -1 unknown, 0 disabled, 1 enabled
static int g_fd    = -1;

static void ensure(void) {
    if (g_state >= 0) return;
    const char *on = getenv("QF_ROUTED_AMD");
    if (!on || on[0] != '1') { g_state = 0; return; }
    const char *host = getenv("QF_ROUTED_AMD_HOST"); if (!host) host = "127.0.0.1";
    int port = 5577; if (const char *p = getenv("QF_ROUTED_AMD_PORT")) port = atoi(p);
    g_fd = qf_amd_client_connect(host, port);
    if (g_fd < 0) {
        fprintf(stderr, "[routed-amd] connect %s:%d failed; using local routed path\n", host, port);
        g_state = 0; return;
    }
    fprintf(stderr, "[routed-amd] offloading routed experts to %s:%d\n", host, port);
    g_state = 1;
}

int qf_routed_amd_enabled(void) { ensure(); return g_state == 1; }

int qf_routed_amd_submit(int il, long long pos, int M, int K,
                         const int *sel, const float *wt, const float *x) {
    if (!qf_routed_amd_enabled()) return -1;
    if (qf_amd_routed_submit(g_fd, il, pos, M, K, sel, wt, x)) {
        fprintf(stderr, "[routed-amd] submit failed L%d; disabling offload\n", il);
        g_state = 0; return -1;
    }
    return 0;
}

int qf_routed_amd_wait(int M, float *y) {
    if (g_state != 1) return -1;
    if (qf_amd_routed_wait(g_fd, M, y)) {
        fprintf(stderr, "[routed-amd] wait failed; disabling offload\n");
        g_state = 0; return -1;
    }
    return 0;
}
