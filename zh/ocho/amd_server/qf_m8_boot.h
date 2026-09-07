// qf_m8_boot.h - AMD M=8 four-card boot contract (shared by boot + server).
#pragma once
#include "qf_moe_4card.h"
#include "qf_moe_wire.h"
typedef struct {
    int dev;
    const unsigned char *Wg, *Sg, *Wu, *Su, *Wd, *Sd;   // device [NLAYER*EPC][EXP_*_BYTES]
    const float *s2g, *s2u, *s2d;                        // device [NLAYER*EPC]
} Qf5CardBases;
#ifdef __cplusplus
extern "C" {
#endif
int  qf_m8_boot(const Qf5CardBases bases[QF5_NCARD]);
int  qf_m8_serve(int fd, const QfWireHdr *h);
int  qf_m8_admit(int fd, const QfWireHdr *h, int *M_out);   // launch async, return
int  qf_m8_drain(int fd, int M, const QfWireHdr *h);        // sync + reply (arrival order)
void qf_m8_shutdown(void);
#ifdef __cplusplus
}
#endif
