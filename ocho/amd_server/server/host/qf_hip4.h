// qf_hip4.h - HOST-BUILD STUB of the frozen 4x gfx906 pipeline engine API.
//
// This header exists so the server TU compiles and links on any host without
// ROCm. It reproduces the frozen public API from src/qf_hip4.h verbatim
// (pure C ABI; QF4_NVOCAB is #defined here instead of coming from
// qwenflash.h). TARGET builds must NOT see this file: the server-target
// Makefile rule uses -I$(ENGINESRC)/src and never adds this directory to the
// include path, so the real engine header wins there.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QfHip4 QfHip4;
#define QF4_NGPU 4
#define QF4_NVOCAB 248320

typedef struct {
    double   load_seconds;
    uint64_t tokens_decoded;
    double   last_step_ms;
    size_t   hbm_bytes[QF4_NGPU];
    int      handoff_mode;      // 0=auto 1=pinned 2=p2p (mode actually in use)
    long     max_context;
} QfHip4Stats;

int  qf_hip4_init(QfHip4 **out, const char *model_dir);
int  qf_hip4_submit(QfHip4 *p, int token, long pos);                 // decode step, logits produced
int  qf_hip4_submit_prefill(QfHip4 *p, int token, long pos);         // prompt token, no logits
int  qf_hip4_wait(QfHip4 *p, long pos, float *logits_out);           // logits_out: QF4_NVOCAB fp32
int  qf_hip4_decode_step(QfHip4 *p, int token, long pos, float *logits_out);
void qf_hip4_reset(QfHip4 *p);   // between requests only; clears KV/GDN/conv/PLE state
int  qf_hip4_stats(QfHip4 *p, QfHip4Stats *out);
void qf_hip4_free(QfHip4 *p);

#ifdef __cplusplus
}
#endif
