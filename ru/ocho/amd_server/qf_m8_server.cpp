// qf_m8_server.cpp - AMD four-MI50 M=8 routed-expert server.
// Loads the RadixArk-Qwen3.8-Flash-Next-NVFP4 experts EXPERT-SHARDED across the
// four cards (experts 128c..128c+127 on card c, all NLAYER layers, resident),
// boots the four-card pool, and serves QFW_EXPERT_M8 on a TCP port.
//   build: hipcc ... qf_m8_server.cpp <m8 objects> -o qf_m8_server
//   run:   ./qf_m8_server [ckpt_dir] [port]
#include "qf_m8_boot.h"
#include "qf_moe_wire.h"
#ifdef QF_HOST_CHECK
#include "host_check_shim.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

static int g_nl = QF5_NLAYER;   // AMD-owned MoE layers (env QF_AMD_MOE_LAYERS)
#define NL   g_nl
#define EPC  QF5_EPC
#define WB   QF5_EXP_W_BYTES
#define SB   QF5_EXP_S_BYTES

struct STFile { const char *map; size_t maplen; const char *data; std::string hdr; };
static bool st_open(const char *path, STFile &f) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "st_open: %s: %s\n", path, strerror(errno)); return false; }
    struct stat sb; if (fstat(fd, &sb)) { close(fd); return false; }
    f.maplen = sb.st_size;
    f.map = (const char *)mmap(NULL, f.maplen, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (f.map == MAP_FAILED) { fprintf(stderr, "st_open: mmap %s failed\n", path); return false; }
    uint64_t hl; memcpy(&hl, f.map, 8);
    f.hdr.assign(f.map + 8, hl);
    f.data = f.map + 8 + hl;
    return true;
}
static bool st_span(const STFile &f, const std::string &name, size_t &off, size_t &len) {
    std::string key = "\"" + name + "\"";
    size_t p = f.hdr.find(key); if (p == std::string::npos) return false;
    size_t d = f.hdr.find("\"data_offsets\"", p); if (d == std::string::npos) return false;
    size_t lb = f.hdr.find('[', d), comma = f.hdr.find(',', lb), rb = f.hdr.find(']', comma);
    if (lb == std::string::npos || comma == std::string::npos || rb == std::string::npos) return false;
    long s = atol(f.hdr.substr(lb + 1, comma - lb - 1).c_str());
    long e = atol(f.hdr.substr(comma + 1, rb - comma - 1).c_str());
    off = (size_t)s; len = (size_t)(e - s); return true;
}

#define HCK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "hip err %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); return 1; } } while (0)

static int load_one(const STFile &f, const std::string &pfx, uint8_t *Wdst, uint8_t *Sdst,
                    float *s2dst, size_t slot) {
    size_t off, len;
    if (!st_span(f, pfx + ".weight", off, len) || len != WB) { fprintf(stderr, "miss %s.weight len\n", pfx.c_str()); return 1; }
    if (hipMemcpy(Wdst + slot * WB, f.data + off, WB, hipMemcpyHostToDevice) != hipSuccess) return 1;
    if (!st_span(f, pfx + ".weight_scale", off, len) || len != SB) { fprintf(stderr, "miss %s.weight_scale\n", pfx.c_str()); return 1; }
    if (hipMemcpy(Sdst + slot * SB, f.data + off, SB, hipMemcpyHostToDevice) != hipSuccess) return 1;
    if (!st_span(f, pfx + ".weight_scale_2", off, len) || len != 4) { fprintf(stderr, "miss %s.weight_scale_2\n", pfx.c_str()); return 1; }
    if (hipMemcpy(s2dst + slot, f.data + off, 4, hipMemcpyHostToDevice) != hipSuccess) return 1;
    return 0;
}

static int load_card(const char *ckpt, int c, Qf5CardBases *b) {
    HCK(hipSetDevice(c));
    uint8_t *Wg, *Sg, *Wu, *Su, *Wd, *Sd; float *s2g, *s2u, *s2d;
    HCK(hipMalloc(&Wg, (size_t)NL * EPC * WB)); HCK(hipMalloc(&Sg, (size_t)NL * EPC * SB));
    HCK(hipMalloc(&Wu, (size_t)NL * EPC * WB)); HCK(hipMalloc(&Su, (size_t)NL * EPC * SB));
    HCK(hipMalloc(&Wd, (size_t)NL * EPC * WB)); HCK(hipMalloc(&Sd, (size_t)NL * EPC * SB));
    HCK(hipMalloc(&s2g, (size_t)NL * EPC * 4)); HCK(hipMalloc(&s2u, (size_t)NL * EPC * 4));
    HCK(hipMalloc(&s2d, (size_t)NL * EPC * 4));
    for (int L = 0; L < NL; L++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/layer-%05d-experts-%04d-%04d.safetensors", ckpt, L, 128 * c, 128 * c + 127);
        STFile f;
        if (!st_open(path, f)) return 1;
        for (int le = 0; le < EPC; le++) {
            int e = 128 * c + le; size_t slot = (size_t)L * EPC + le;
            char pre[256];
            snprintf(pre, sizeof pre, "model.language_model.layers.%d.mlp.experts.%d", L, e);
            std::string P(pre);
            if (load_one(f, P + ".gate_proj", Wg, Sg, s2g, slot)) return 1;
            if (load_one(f, P + ".up_proj",   Wu, Su, s2u, slot)) return 1;
            if (load_one(f, P + ".down_proj", Wd, Sd, s2d, slot)) return 1;
        }
        munmap((void *)f.map, f.maplen);
        if ((L % 12) == 11) fprintf(stderr, "card %d: layers 0..%d loaded\n", c, L);
    }
    b->dev = c; b->Wg = Wg; b->Sg = Sg; b->Wu = Wu; b->Su = Su; b->Wd = Wd; b->Sd = Sd;
    b->s2g = s2g; b->s2u = s2u; b->s2d = s2d;
    fprintf(stderr, "card %d resident: experts %d..%d, %d layers (~%.1f GiB weights)\n",
            c, 128 * c, 128 * c + 127, NL, (double)NL * EPC * WB * 3 / (1 << 30));
    return 0;
}

static int io_all(int fd, void *p, size_t n, int wr) {
    char *ch = (char *)p;
    while (n) { long k = wr ? send(fd, ch, n, MSG_NOSIGNAL) : recv(fd, ch, n, MSG_WAITALL);
        if (k <= 0) return -1; ch += k; n -= (size_t)k; }
    return 0;
}

int qf_amd_moe_layers(void) { return g_nl; }

int main(int argc, char **argv) {
    const char *ckpt = argc > 1 ? argv[1] : (getenv("QF_MODEL_DIR") ? getenv("QF_MODEL_DIR") : "./model");
    if (const char *e = getenv("QF_AMD_MOE_LAYERS")) { int v = atoi(e); if (v > 0 && v <= QF5_NLAYER) g_nl = v; }
    fprintf(stderr, "qf_m8_server: AMD owns MoE layers 0..%d (%d of %d); layers %d..%d stay on Spark\n",
            g_nl - 1, g_nl, QF5_NLAYER, g_nl, QF5_NLAYER - 1);
    int port = argc > 2 ? atoi(argv[2]) : 5577;
    fprintf(stderr, "qf_m8_server: ckpt=%s port=%d\n", ckpt, port);
    Qf5CardBases bases[QF5_NCARD];
    for (int c = 0; c < QF5_NCARD; c++) if (load_card(ckpt, c, &bases[c])) { fprintf(stderr, "load card %d failed\n", c); return 1; }
    if (qf_m8_boot(bases)) { fprintf(stderr, "boot failed\n"); return 1; }

    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons((unsigned short)port);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 4)) { fprintf(stderr, "bind/listen %d failed\n", port); return 1; }
    fprintf(stderr, "qf_m8_server: READY on %d (QFW_EXPERT_M8)\n", port);
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        fprintf(stderr, "qf_m8_server: Spark connected\n");
        for (;;) {
            QfWireHdr h;
            if (io_all(fd, &h, sizeof h, 0)) break;
            if (h.magic != QFW_MAGIC) { fprintf(stderr, "bad magic\n"); break; }
            if (h.op != QFW_EXPERT_M8) { fprintf(stderr, "unexpected op %u\n", h.op); break; }
            // Deferred receive (GLM mechanism): admit this wave (async launch on
            // its own slot), then admit any header ALREADY on the wire before
            // draining. Both waves are then in flight on the four cards at once.
            QfWireHdr hq[QF5_HANDOFF_WAVES]; int Mq[QF5_HANDOFF_WAVES], nq = 0;
            int Mv = 0;
            if (qf_m8_admit(fd, &h, &Mv)) { fprintf(stderr, "admit failed\n"); break; }
            hq[nq] = h; Mq[nq] = Mv; nq++;
            while (nq < QF5_HANDOFF_WAVES) {
                QfWireHdr h2;
                ssize_t pk = recv(fd, &h2, sizeof h2, MSG_PEEK | MSG_DONTWAIT);
                if (pk != (ssize_t)sizeof h2) break;                 // nothing queued yet
                if (io_all(fd, &h2, sizeof h2, 0)) { nq = -1; break; }
                if (h2.magic != QFW_MAGIC || h2.op != QFW_EXPERT_M8) { nq = -1; break; }
                if (qf_m8_admit(fd, &h2, &Mv)) { nq = -1; break; }
                hq[nq] = h2; Mq[nq] = Mv; nq++;
            }
            if (nq < 0) break;
            int bad = 0;
            for (int i = 0; i < nq; i++)
                if (qf_m8_drain(fd, Mq[i], &hq[i])) { bad = 1; break; }   // arrival order
            if (bad) break;
        }
        close(fd);
        fprintf(stderr, "qf_m8_server: Spark disconnected\n");
    }
}
