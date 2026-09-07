// qf_moe_handoff.h - AMD-side serve handler for the M=8 routed-expert RPC.
// Includes the pool; AMD-only. The pure client contract is in qf_moe_wire.h.
#pragma once
#include "qf_moe_wire.h"
#include "qf_moe_4card.h"

#ifdef __cplusplus
extern "C" {
#endif
// Handle one already-received QFW_EXPERT_M8 header on fd: read the batch, run it
// on ring slot `wave`, send the reply. Drop into the serve loop's op switch.
// Returns 0 on success, non-zero to drop the connection (dead-arm rule).
int qf_amd_routed_serve(int fd, Qf5MoePool *pool, int wave, const QfWireHdr *h);
// Deferred receive (GLM mechanism): admit launches the wave async on its slot
// and returns so the loop can read the next header; drain syncs + replies, in
// ARRIVAL ORDER. Two slots => A and B genuinely in flight on the AMD side.
int qf_amd_routed_admit(int fd, Qf5MoePool *pool, int wave, const QfWireHdr *h, int *M_out, int *il_out);
int qf_amd_routed_drain(int fd, Qf5MoePool *pool, int wave, int M, const QfWireHdr *h);
#ifdef __cplusplus
}
#endif
