// planner.h - fail-before-allocation memory planner / budget guard.
//
// Pure host arithmetic: no CUDA/HIP headers, no device calls, no heap
// allocation. Given a backend configuration it computes the full device
// memory commitment (dense weights / expert residency / recurrent state /
// workspace / pinned staging), prints the breakdown, and refuses (returns
// -1 from qf_budget_check) BEFORE any device allocation happens when the
// plan does not fit the budget.
//
// Call sites:
//   - Spark (CUDA): qf_model_load() in model.cpp, before the first
//     cudaMalloc (before qf_store_open, in fact). Default budget
//     QF_SPARK_DEFAULT_BUDGET_GIB (96 GiB); the selected configuration must
//     plan strictly below it in BOTH expert-residency modes.
//   - AMD (HIP):    qf_hip4_init() in hip/qf_hip4_pipeline.cpp, once per
//     gfx906 stage, before any hipMalloc. Default per-card budget
//     QF_AMD_DEFAULT_BUDGET_GIB (14 of the 16 GB HBM).
//
// Spark expert-residency modes (model.cpp, QF_EXPERT_MODE):
//   lru  - bounded per-layer LRU, QF_EXPERT_CACHE_SLOTS experts resident
//          (clamped to [n_expert_used, n_experts]; default 192).
//   full - all 512 experts resident per layer (~63.3 GiB device). Loaded
//          once through a fixed pinned staging buffer
//          (QF_SPARK_STAGING_BYTES) with fadvise/madvise drop-behind on the
//          mmap store, so the host page cache never duplicates the expert
//          bytes. GB10 is unified memory: device allocations, pinned host
//          buffers, and page cache all draw from the same pool, which is
//          why the guard accounts for staging and the loader drops cache.
//
// Knobs (read by the call sites / qf_spark_mode_env, not by qf_plan_compute):
//   QF_EXPERT_MODE           lru (default) | full   (Spark)
//   QF_EXPERT_CACHE_SLOTS    resident routed experts per layer (Spark lru)
//   QF4_EXPERT_CACHE_SLOTS   same, per AMD stage (512 = eager full residency)
//   QF_BUDGET_GIB            override the budget (per card on AMD)
#pragma once
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// DGX Spark: the whole model on one GB10. "Default below 96 GiB total
// commitment": the default cap is 96 GiB and every supported configuration
// must plan strictly below it before a single byte is allocated.
#define QF_SPARK_DEFAULT_BUDGET_GIB 96.0
// gfx906 (MI50): 16 GB HBM per card; reserve ~2 GB for driver/runtime and
// host-staged handoff buffers.
#define QF_AMD_DEFAULT_BUDGET_GIB 14.0

// Spark expert-residency modes (QF_EXPERT_MODE).
#define QF_SPARK_MODE_LRU  0   // bounded LRU, cache_slots per layer
#define QF_SPARK_MODE_FULL 1   // all n_experts resident per layer

// Fixed pinned staging used by the Spark full-resident loader: two 4 MiB
// halves, ping-ponged (memcpy from mmap into one half while the stream
// drains the other). One expert tensor is at most 800 KiB, so a half is
// never subdivided and never exceeded. Host bytes, but on GB10 host and
// device share one memory pool, so the budget guard accounts for them.
#define QF_SPARK_STAGING_BYTES (2u * 4u * 1024u * 1024u)

typedef struct {
    // model shape; defaults match Qwen3.8-Flash-Next (SEMANTICS.md)
    int n_vocab;        // 248320
    int n_embd;         // 2560
    int n_layer;        // layers committed on this device (48 Spark, 12/stage AMD)
    int n_gdn;          // linear-attention layers in scope (36 Spark, 9/stage)
    int n_attn;         // full-attention layers in scope (12 Spark, 3/stage)
    int n_head;         // 24
    int n_head_kv;      // 2
    int head_dim;       // 256
    int n_experts;      // 512
    int n_expert_used;  // 10 (lower clamp for cache_slots)
    int n_ff_exp;       // 640
    int hc_count;       // 4
    int hc_lowrank;     // 320
    int ssm_d_inner;    // 10240
    int ssm_v_heads;    // 48
    int ssm_d_state;    // 128
    int max_pos;        // KV cache length (MAXPOS in cuda/qf.cu / hip stage)
    int cache_slots;    // resident routed experts per layer on this device
    int mode;           // QF_SPARK_MODE_* (FULL adds the fixed staging term)
    int has_embed;      // device holds tok_embd  (Spark: 1, AMD stage 0)
    int has_head;       // device holds lm_head + output HC (Spark: 1, AMD last stage)
    int load_indexer;   // QSA indexer tensors resident (Spark: 1, AMD v1: 0)
    const char *tag;    // label for log lines
} QfPlanCfg;

typedef struct {
    uint64_t dense_bytes;      // non-expert weights resident on device
    uint64_t cache_bytes;      // routed-expert residency (slots per layer)
    uint64_t state_bytes;      // GDN S matrices, conv rings, KV cache
    uint64_t workspace_bytes;  // decode activations + logits
    uint64_t staging_bytes;    // fixed pinned staging (Spark full mode only)
    uint64_t total_bytes;
    uint64_t budget_bytes;
    int      ok;               // total strictly below budget
} QfPlan;

// Whole-model single-GPU configuration (DGX Spark / GB10), 192 slots/layer,
// LRU mode. Caller overrides mode/cache_slots from the environment knobs.
void qf_plan_defaults_spark(QfPlanCfg *cfg);
// One 12-layer stage of the 4x gfx906 pipeline. cache_slots defaults to 512
// (eager full residency, matching the current qf_hip4_stage.hip loader);
// lower it once the bounded per-layer cache is ported to HIP.
void qf_plan_defaults_amd_stage(QfPlanCfg *cfg, int stage, int n_stages);

// Compute the plan. Pure arithmetic; never fails.
void qf_plan_compute(const QfPlanCfg *cfg, QfPlan *out);

// Print the dense/cache/state/workspace/staging breakdown and the verdict.
void qf_plan_print(const QfPlanCfg *cfg, const QfPlan *p, FILE *f);

// The guard: compute + print; return 0 when the plan fits strictly below the
// budget, -1 (after printing the shortfall and the remediation) otherwise.
// budget_gib <= 0 is rejected by the caller passing a default instead.
int qf_budget_check(const QfPlanCfg *cfg, double budget_gib, FILE *f);

// getenv("name") parsed as a GiB double, or dflt when unset/invalid.
double qf_budget_env(const char *name, double dflt);

// Parse QF_EXPERT_MODE: unset/empty/"lru" -> QF_SPARK_MODE_LRU, "full" ->
// QF_SPARK_MODE_FULL, anything else -> -1 after printing to err (callers
// must refuse to load: an unknown mode is a configuration error, and the
// guard cannot budget a mode it does not know).
int qf_spark_mode_env(FILE *err);

#ifdef __cplusplus
}
#endif
