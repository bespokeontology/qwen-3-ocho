// test_qf_client.cpp — wire test for the AMD M=8 routed-expert tier.
#include "qf_moe_wire.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned lcg = 123u;
static unsigned rnd(void){ lcg = lcg*1664525u + 1013904223u; return lcg>>1; }

int main(int argc, char **argv) {
    const char *host = argc > 1 ? argv[1] : "127.0.0.1";
    int port = argc > 2 ? atoi(argv[2]) : 5577;
    int M = 8, K = 10;
    int fd = qf_amd_client_connect(host, port);
    if (fd < 0) { printf("connect failed\n"); return 1; }
    printf("connected %s:%d\n", host, port);
    int sel[64*10]; float wt[64*10]; float x[64*2560]; float y[64*2560];
    for (int i = 0; i < M*K; i++) { sel[i] = (int)(rnd() % 512); wt[i] = (float)(rnd()%1000)/1000.f; }
    for (int i = 0; i < M*2560; i++) x[i] = (float)((int)(rnd()%2000) - 1000) / 1000.f;
    int total_nz = 0; float total_sum = 0;
    for (int il = 0; il < 48; il += 12) {
        int rc = qf_amd_routed_submit(fd, il, 2000 + il, M, K, sel, wt, x);
        if (rc) { printf("layer %d submit rc=%d\n", il, rc); continue; }
        rc = qf_amd_routed_wait(fd, M, y);
        if (rc) { printf("layer %d wait rc=%d\n", il, rc); continue; }
        float ss = 0; int nn = 0;
        for (int i = 0; i < M*2560; i++) { ss += y[i]; if (y[i] != 0) nn++; }
        printf("layer %d: nonzero=%d sum=%.3f\n", il, nn, ss);
        total_nz += nn; total_sum += ss;
    }
    printf("total: nonzero=%d sum=%.3f\n", total_nz, total_sum);
    if (total_nz == 0) { printf("CLIENT TEST FAIL (all-zero reply)\n"); return 4; }
    printf("CLIENT TEST PASS\n");
    return 0;
}
