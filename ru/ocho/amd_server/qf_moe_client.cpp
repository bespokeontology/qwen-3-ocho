// qf_moe_client.cpp - Spark-side wire client for the M=8 routed-expert RPC.
// Pure sockets; no HIP. Host check: g++ -DQF_HOST_CHECK -I. -fsyntax-only qf_moe_client.cpp
#include "qf_moe_wire.h"
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define QF5_HANDOFF_MAXM 64
#define QF5_HANDOFF_MAXK 10

static int io_all(int fd, void *p, size_t n, int wr) {
    char *c = (char *)p;
    while (n) {
        long k = wr ? send(fd, c, n, MSG_NOSIGNAL) : recv(fd, c, n, MSG_WAITALL);
        if (k <= 0) return -1;
        c += k; n -= (size_t)k;
    }
    return 0;
}
extern "C" int qf_amd_client_connect(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    if (fd < 0) return -1;
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) { close(fd); return -1; }
    if (connect(fd, (struct sockaddr *)&a, sizeof a)) { close(fd); return -1; }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}
extern "C" int qf_amd_routed_submit(int fd, int il, long long pos, int M, int K,
                                    const int *sel, const float *wt, const float *x) {
    if (M < 1 || M > QF5_HANDOFF_MAXM || K < 1 || K > QF5_HANDOFF_MAXK) return -1;
    unsigned uM = (unsigned)M, uK = (unsigned)K;
    size_t sel_b = (size_t)M * K * sizeof(int), wt_b = (size_t)M * K * sizeof(float);
    size_t x_b = (size_t)M * QF_ROUTED_NEMBD * sizeof(float);
    QfWireHdr h; h.magic = QFW_MAGIC; h.op = QFW_EXPERT_M8; h.pos = pos; h.token = il;
    h.nbytes = (int)(2 * sizeof(unsigned) + sel_b + wt_b + x_b);
    if (io_all(fd, &h, sizeof h, 1) || io_all(fd, &uM, sizeof uM, 1) || io_all(fd, &uK, sizeof uK, 1)) return -1;
    if (io_all(fd, (void *)sel, sel_b, 1) || io_all(fd, (void *)wt, wt_b, 1) || io_all(fd, (void *)x, x_b, 1)) return -1;
    return 0;
}
extern "C" int qf_amd_routed_wait(int fd, int M, float *y) {
    QfWireHdr h;
    if (io_all(fd, &h, sizeof h, 0)) return -1;
    if (h.magic != QFW_MAGIC || h.op != QFW_EXPERT_M8) return -1;
    if (h.nbytes != (int)((size_t)M * QF_ROUTED_NEMBD * sizeof(float))) return -1;
    return io_all(fd, y, (size_t)M * QF_ROUTED_NEMBD * sizeof(float), 0);
}

// ONE crossing per token: M residuals out, M token ids back.
extern "C" int qf_amd_region_submit_resid_fd(int fd, long long pos, int M, const long *posM,
                                             const float *hR, int rsize) {
    if (M < 1 || M > QF5_HANDOFF_MAXM) return -1;
    QfWireHdr h; h.magic = QFW_MAGIC; h.op = QFW_REGION_M8; h.pos = pos; h.token = M;
    h.nbytes = (int)((size_t)M * (sizeof(long long) + (size_t)rsize * sizeof(float)));
    if (io_all(fd, &h, sizeof h, 1)) return -1;
    long long p64[QF5_HANDOFF_MAXM];
    for (int i = 0; i < M; i++) p64[i] = (long long)posM[i];
    if (io_all(fd, p64, (size_t)M * sizeof(long long), 1)) return -1;
    return io_all(fd, (void *)hR, (size_t)M * rsize * sizeof(float), 1);
}
// (The token-form QFW_REGION_M8 submit was retired: it shared op 8 with the
// residual form above but the server reads pos64 + a full residual for op 8,
// so the two were wire-incompatible. Tokens travel via QFW_PREFIX_M.)
extern "C" int qf_amd_region_wait(int fd, int M, int *tokens_out) {
    QfWireHdr h;
    if (io_all(fd, &h, sizeof h, 0)) return -1;
    if (h.magic != QFW_MAGIC || h.op != QFW_REGION_M8) return -1;
    if (h.nbytes != (int)((size_t)M * sizeof(int))) return -1;
    return io_all(fd, tokens_out, (size_t)M * sizeof(int), 0);
}
extern "C" int qf_amd_region_reset(int fd) {
    QfWireHdr h; h.magic = QFW_MAGIC; h.op = QFW_REGION_RESET; h.pos = 0; h.token = 0; h.nbytes = 0;
    if (io_all(fd, &h, sizeof h, 1)) return -1;
    if (io_all(fd, &h, sizeof h, 0)) return -1;
    return (h.magic == QFW_MAGIC && h.op == QFW_REGION_RESET) ? 0 : -1;
}

// Alternating map: AMD prefix (token ids -> residuals) and AMD head
// (residuals -> token ids). Two coarse crossings per token; the reply from the
// head is 4 B/row, never logits.
// Split submit/wait forms (W2 pipeline): a request is sent, the caller does
// other work, then reads the reply. Replies come back FIFO on the one socket
// and the server echoes the request header, so every wait checks (op, pos =
// row base or slot, row count) against what it expects: a mis-ordered wait is
// an error, never silently the wrong batch's data.
// Rows [r0, r0+nr) of the AMD prefix: h.pos = row base, h.token = row count.
// r0 > 0 primes one request row through its own prompt, or selects a slot.
extern "C" int qf_amd_prefix_submit_fd(int fd, int r0, int nr, const int *tokens, const long *posM) {
    if (r0 < 0 || nr < 1 || r0 + nr > QF5_HANDOFF_MAXM) return -1;
    QfWireHdr h; h.magic = QFW_MAGIC; h.op = QFW_PREFIX_M; h.pos = r0; h.token = nr;
    h.nbytes = (int)((size_t)nr * (sizeof(int) + sizeof(long long)));
    if (io_all(fd, &h, sizeof h, 1)) return -1;
    if (io_all(fd, (void *)tokens, (size_t)nr * sizeof(int), 1)) return -1;
    long long p64[QF5_HANDOFF_MAXM];
    for (int i = 0; i < nr; i++) p64[i] = (long long)posM[i];
    return io_all(fd, p64, (size_t)nr * sizeof(long long), 1);
}
extern "C" int qf_amd_prefix_wait_fd(int fd, int r0, int nr, float *rout, int rsize) {
    QfWireHdr r;
    if (io_all(fd, &r, sizeof r, 0)) return -1;
    if (r.magic != QFW_MAGIC || r.op != QFW_PREFIX_M || r.pos != r0 || r.token != nr) return -1;
    return io_all(fd, rout, (size_t)nr * rsize * sizeof(float), 0);
}
extern "C" int qf_amd_prefix_rows_fd(int fd, int r0, int nr, const int *tokens, const long *posM,
                                     float *rout, int rsize) {
    if (qf_amd_prefix_submit_fd(fd, r0, nr, tokens, posM)) return -1;
    return qf_amd_prefix_wait_fd(fd, r0, nr, rout, rsize);
}
extern "C" int qf_amd_prefix_fd(int fd, int M, const int *tokens, const long *posM,
                                float *rout, int rsize) {
    return qf_amd_prefix_rows_fd(fd, 0, M, tokens, posM, rout, rsize);
}
// AMD head: h.pos = slot (echoed), h.token = rows; reply = M token ids.
extern "C" int qf_amd_head_submit_fd(int fd, int slot, int M, const float *rin, int rsize) {
    if (M < 1 || M > QF5_HANDOFF_MAXM) return -1;
    QfWireHdr h; h.magic = QFW_MAGIC; h.op = QFW_HEAD_M; h.pos = slot; h.token = M;
    h.nbytes = (int)((size_t)M * rsize * sizeof(float));
    if (io_all(fd, &h, sizeof h, 1)) return -1;
    return io_all(fd, (void *)rin, (size_t)M * rsize * sizeof(float), 1);
}
extern "C" int qf_amd_head_wait_fd(int fd, int slot, int M, int *tokens_out) {
    QfWireHdr r;
    if (io_all(fd, &r, sizeof r, 0)) return -1;
    if (r.magic != QFW_MAGIC || r.op != QFW_HEAD_M || r.pos != slot || r.token != M) return -1;
    return io_all(fd, tokens_out, (size_t)M * sizeof(int), 0);
}
extern "C" int qf_amd_head_fd(int fd, int M, const float *rin, int rsize, int *tokens_out) {
    if (qf_amd_head_submit_fd(fd, 0, M, rin, rsize)) return -1;
    return qf_amd_head_wait_fd(fd, 0, M, tokens_out);
}
