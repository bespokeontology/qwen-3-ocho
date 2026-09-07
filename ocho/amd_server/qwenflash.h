// qwenflash.h - Qwen3.8-Flash-Next native engine
// Reads RadixArk NVFP4 safetensors directly. GPU-native (CUDA/HIP backends).
#pragma once
#include <stdint.h>
#include <stddef.h>

#define QF_MAX_TENSORS 300000
#define QF_MAX_NAME 256
#define QF_REC_SIZE 512
#define QF_MANIFEST_MAGIC 0x51464D31u  // "QFM1"

typedef struct {
    uint32_t file_idx;
    uint32_t dtype;          // 1=BF16 2=F32 3=U8(fp4 packed) 4=F8_E4M3 5=I64
    uint64_t ne[4];          // dims, fastest first (ggml convention)
    uint64_t data_off;       // byte offset into shard file
    uint64_t nbytes;
    uint64_t scale2_off;     // weight_scale_2 (F32) offset, 0 if absent
    uint64_t input_off;      // input_scale (F32) offset, 0 if absent
    uint32_t scale2_file;
    uint32_t input_file;
    uint32_t flags;
} QfRec;

typedef struct {
    char  name[QF_MAX_NAME];
    QfRec rec;
    uint8_t _pad[168];       // total record = 512 bytes on-disk stride
} QfEntry;

typedef struct {
    int      n_shards;
    char   (*shard_paths)[1024];
    int    *shard_fds;           // open handles
    void  **shard_maps;          // mmap base per shard
    size_t *shard_sizes;
    QfEntry *entries;            // sorted by name
    int      n_entries;
} QfStore;

// model config (from config.json, baked at build time for v1)
typedef struct {
    int n_vocab;          // 248320
    int n_embd;           // 2560
    int n_layer;          // 48
    int n_head;           // 24
    int n_head_kv;        // 2
    int head_dim;         // 256
    int n_experts;        // 512
    int n_expert_used;    // 10
    int n_ff_exp;         // 640
    int hc_count;         // 4   (hyper-connections)
    int hc_lowrank;       // 320
    int full_attn_interval; // 4
    int ssm_d_conv;       // 4
    int ssm_d_inner;      // 10240
    int ssm_d_state;      // 128
    int ssm_dt_rank;      // 48
    int ssm_n_group;      // 16
    int indexer_n_head;   // 4
    int indexer_head_dim; // 128
    int indexer_top_k;    // 2048
    int indexer_compress; // 4
    int ple_ngram;        // 3
    int ple_heads_per_ngram; // 8
    int ple_conv_kernel;  // 4
    int ple_embd_per_layer;  // 160
    int ple_row_dim;
    int ple_total_rows;
    float rms_eps;        // 1e-6
} QfConfig;

// wave2: bounded-cache miss mailbox. Written on device by the routing dispatch
// kernel (zero-copy pinned memory, no D2H memcpy), read and cleared by the host.
// Single device writer / single host reader per decode step; seq is written last
// (after __threadfence_system) and stamps "this layer's dispatch ran for token seq".
#define QF_ROUTE_MB_CAP 16    // >= n_expert_used (10)
typedef struct {
    volatile uint32_t seq;                          // token sequence stamp (written last)
    volatile uint32_t miss_n;                       // 0..n_expert_used
    volatile int32_t  miss_expert[QF_ROUTE_MB_CAP]; // experts to upload
    volatile int32_t  miss_slot[QF_ROUTE_MB_CAP];   // victim slot assigned on device
} QfRouteMailbox;

// loaded layer weights (device pointers)
typedef struct {
    // hyper-connections (attn side)
    void *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    // hyper-connections (ffn side)
    void *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
    // linear attention (GDN) layers
    void *qkv, *zgate;         // attn_qkv [10240,2560], attn_gate [6144,2560]
    void *conv1d;              // [4,10240] f32
    void *dt_bias, *a;         // [48] each
    void *beta, *alpha;        // [2560,48] (ssm_beta/ssm_alpha)
    void *gdn_norm;            // [128]
    void *gdn_out;             // [6144,2560]
    // full attention (QSA) layers
    void *wq, *wk, *wv, *wo;   // q [6144,2560](q+gate interleaved), k/v [512,2560], out [2560,6144]
    void *q_norm, *k_norm;     // [256]
    void *idx_qk;              // [640,2560] split 512/128 at load
    void *idx_qnorm, *idx_knorm; // [128]
    // MoE
    void *router;              // [512,2560] bf16
    void *exp_gate;            // NVFP4 [512, 640, 2560] + scales
    void *exp_up;
    void *exp_down;
    void *exp_scale;           // f8 gate scales [512,640,160]
    void *exp_scale_up;        // f8 up scales [512,640,160]
    void *exp_scale_down;      // f8 scales down [512,2560,40]
    int exp_cache_slots;       // resident routed-expert slots for this layer
    int *exp_slot_for_expert;  // host map [n_experts], -1 when not resident (load-time / qf_prepare_expert only)
    int *exp_expert_in_slot;   // host map [exp_cache_slots], -1 when empty
    uint64_t *exp_slot_age;    // host LRU timestamps
    uint64_t exp_cache_clock;
    // wave2: device-resident routing/cache metadata (real build; NULL in SYNTH).
    // Ground truth for expert residency lives on device during decode; the host
    // maps above are only used to seed these at load.
    int      *exp_slot_dev;          // device slot map [n_experts], -1 when not resident
    int      *exp_expert_in_slot_dev;// device reverse map [exp_cache_slots]
    uint64_t *exp_slot_age_dev;      // device LRU clock per slot [exp_cache_slots]
    uint64_t *exp_clock_dev;         // device per-layer LRU clock [1]
    float    *s2_gate_dev, *s2_up_dev, *s2_down_dev; // device weight_scale_2 tables [n_experts]
    int      *route_slot_dev;        // device slots for the current token's top-k [n_expert_used]
    uint32_t *slot_ready_dev;        // device doorbell per slot: 0 = upload in flight
    QfRouteMailbox *route_mb_host;   // pinned miss mailbox (host view)
    QfRouteMailbox *route_mb_dev;    // device alias of the same mailbox
    void *shexp_gate, *shexp_up, *shexp_down; // [640,2560] x2, [2560,640]
    void *shexp_gate_inp;      // [2560]
} QfLayer;

typedef struct {
    QfConfig cfg;
    QfStore store;
    void *tok_embd;            // bf16 [248320, 2560]
    void *output_hc_norm, *output_hc_down, *output_hc_up; // hc mixer at head
    void *lm_head;             // bf16 [248320, 2560] (output.weight)
    QfLayer *layers;
    // expert residency (Spark): exp_mode is a QF_SPARK_MODE_* value (planner.h);
    // exp_staging is the fixed pinned buffer of the full-resident loader (NULL
    // in lru mode). GB10 is unified memory, so pinned bytes draw from the same
    // pool as device allocations and are charged to the budget in full mode.
    int    exp_mode;
    void  *exp_staging;
    size_t exp_staging_bytes;
    // PLE
    void *ple_table;           // fp8 mmap'd [total_rows, 160]
    void *ple_key, *ple_value; // per PLE layer
    void *ple_norm_k, *ple_norm_q, *ple_norm_c, *ple_conv;
    int   ple_layer;           // which layer carries PLE
} QfModel;

// store.c
int  qf_store_open(QfStore *st, const char *dir);
void qf_store_close(QfStore *st);
const QfEntry *qf_find(const QfStore *st, const char *name);
// page-cache drop-behind: used by the full-resident loader so the mmap store
// never duplicates expert bytes already staged to the device
void qf_store_drop_range(QfStore *st, int file_idx, uint64_t off, uint64_t len);
void qf_store_drop_all(QfStore *st);

// model.c
int  qf_model_load(QfModel *m, const char *dir);
void qf_model_free(QfModel *m);

// generate.c
int  qf_generate(QfModel *m, const int *tokens, int n_prompt, int n_gen,
                 float temp, int top_k, float top_p,
                 int (*cb)(int token, void *ud), void *ud);

#ifdef __cplusplus
static_assert(sizeof(QfRec) == 88, "QfRec layout");
static_assert(sizeof(QfEntry) == 512, "QfEntry layout");
#endif

// exposed for decode (filled by model.c)
float *qf_layer_s2_gate(int il);
float *qf_layer_s2_up(int il);
float *qf_layer_s2_down(int il);
int qf_prepare_expert(QfModel *m, int layer, int expert, void *stream);
// wave2: service the device-signaled expert-cache misses for one layer
// (fixed pinned staging + transfer stream + device doorbell). No-op in SYNTH.
int qf_route_service(QfModel *m, int layer);
// wave2: nonzero if a device routing kernel latched a deadlock-guard error.
int qf_route_error(void);
