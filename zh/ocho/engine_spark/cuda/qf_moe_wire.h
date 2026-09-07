// qf_moe_wire.h - pure Spark<->AMD wire contract for the M=8 routed-expert RPC.
// No HIP, no pool: both the Spark client and the AMD server include this. The
// AMD serve handler additionally includes qf_moe_handoff.h for the pool type.
#pragma once
#define QF_ROUTED_NEMBD 2560         // == QF5_NEMBD; kept HIP-free for the Spark build
#define QFW_EXPERT_M8   7            // routed-expert RPC (per-layer)
#define QFW_REGION_M8   8            // dependency-closed region: AMD runs its rows
                                     // through ALL its owned layers and returns
                                     // sampled token ids. Only tokens cross the
                                     // wire; no per-layer traffic.
#define QFW_REGION_RESET 9
#define QFW_REGION_FORK  16   // token = M rows, pos = P: copy row 0's resident state into rows 1..M-1 (all stages)
#define QFW_REGION_COMMIT 17 // token = source row r, pos = P: copy row r's resident state into row 0 (all stages) - decision/commit loop
// Alternating map (newest receipted): AMD prefix -> Spark tail -> AMD head.
#define QFW_PREFIX_M   12   // token ids in  -> M residuals out (AMD layers 0..S-1)
#define QFW_BF16       0x100   // op flag: residual payloads on the wire are bf16 (half the bytes); the reply mirrors it
#define QFW_PREFIX_CHUNK 15   // T consecutive prompt positions (tokens in) -> T residuals out (prefill chunk, row 0 state)
#define QFW_CHUNK_MAXT   256   // max positions per QFW_PREFIX_CHUNK request (server MAXCHUNK)
#define QFW_HEADPREFIX 14   // residual in -> head+argmax -> token, then the prefix of that token at h.pos -> token + residual out (ONE round trip per step)
#define QFW_HEAD_M     13   // M residuals in -> M token ids out (AMD out HC+lm_head+argmax)           // clear per-row recurrent/KV state
#define QF5_HANDOFF_WAVES 2   // deferred receive: A and B genuinely in flight          // serve-side ring depth (prefill overlap; decode uses 1)

#ifndef QFW_HDR_DEFINED
#define QFW_HDR_DEFINED
#define QFW_MAGIC 0x30574651u        // "QFW0", identical to main.cu
typedef struct { unsigned magic, op; long long pos; int token, nbytes; } QfWireHdr;
#endif

#ifdef __cplusplus
extern "C" {
#endif
// ---- Spark (head) side: pure sockets, no HIP ----
int  qf_amd_client_connect(const char *host, int port);
int  qf_amd_routed_submit(int fd, int il, long long pos, int M, int K,
                          const int *sel, const float *wt, const float *x);
int  qf_amd_routed_wait(int fd, int M, float *y);

// Region ops (QFW_REGION_M8): ship M positions + residuals, get M token ids back.
int  qf_amd_region_submit_resid_fd(int fd, long long pos, int M, const long *posM,
                                   const float *hR, int rsize);
int  qf_amd_region_wait(int fd, int M, int *tokens_out);
int  qf_amd_region_reset(int fd);
// Alternating map: AMD prefix (tokens -> residuals) and AMD head (residuals -> ids).
int  qf_amd_prefix_fd(int fd, int M, const int *tokens, const long *posM, float *rout, int rsize);
int  qf_amd_prefix_rows_fd(int fd, int r0, int nr, const int *tokens, const long *posM, float *rout, int rsize);
// Split forms (W2): submit, do other work, wait. Replies are FIFO; waits verify (op, pos, rows).
int  qf_amd_prefix_submit_fd(int fd, int r0, int nr, const int *tokens, const long *posM);
int  qf_amd_prefix_wait_fd(int fd, int r0, int nr, float *rout, int rsize);
int  qf_amd_head_submit_fd(int fd, int slot, int M, const float *rin, int rsize);
int  qf_amd_head_wait_fd(int fd, int slot, int M, int *tokens_out);
int  qf_amd_head_fd(int fd, int M, const float *rin, int rsize, int *tokens_out);
#ifdef __cplusplus
}
#endif
