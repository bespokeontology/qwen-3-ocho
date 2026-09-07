// planner.cpp - fail-before-allocation memory planner / budget guard.
// Host-only; see planner.h for the contract. All sizes are computed from the
// model constants in SEMANTICS.md and mirror the allocation sites:
//   - dense:    load_dev() calls in model.cpp / qf_hip4_stage.hip
//   - cache:    the expert LRU / full residency in model.cpp
//               (slots * per-expert packed bytes)
//   - state:    qf_forward_init() in cuda/qf.cu / qf4_stage_init() in HIP
//   - workspace: the decode activation buffers of the same functions
//   - staging:  the fixed pinned buffer of the Spark full-resident loader
#include "planner.h"
#include <stdlib.h>
#include <string.h>

#define BF16 2

void qf_plan_defaults_spark(QfPlanCfg *c) {
    memset(c, 0, sizeof(*c));
    c->n_vocab = 248320; c->n_embd = 2560;
    c->n_layer = 48; c->n_gdn = 36; c->n_attn = 12;
    c->n_head = 24; c->n_head_kv = 2; c->head_dim = 256;
    c->n_experts = 512; c->n_expert_used = 10; c->n_ff_exp = 640;
    c->hc_count = 4; c->hc_lowrank = 320;
    c->ssm_d_inner = 10240; c->ssm_v_heads = 48; c->ssm_d_state = 128;
    // matches cuda/qf.cu: MAXPOS default 262144, QF_MAX_CONTEXT env override
    // (the server exports --max-context into QF_MAX_CONTEXT before load, so
    // the guard budgets the same KV footprint the engine allocates)
    c->max_pos = 262144;
    if (const char *e = getenv("QF_MAX_CONTEXT"))
        if (atol(e) > 0) c->max_pos = (int)atol(e);
    c->cache_slots = 192;           // model.cpp default, ~23.74 GiB total
    c->mode = QF_SPARK_MODE_LRU;
    c->has_embed = 1; c->has_head = 1; c->load_indexer = 1;
    c->tag = "spark";
}

void qf_plan_defaults_amd_stage(QfPlanCfg *c, int stage, int n_stages) {
    qf_plan_defaults_spark(c);
    c->n_layer = 48 / n_stages;     // 12 layers per stage
    c->n_gdn = c->n_layer * 3 / 4;  // 9 GDN + 3 full-attn in every 12-layer block
    c->n_attn = c->n_layer - c->n_gdn;
    c->cache_slots = 512;           // eager full residency (current HIP loader)
    c->mode = QF_SPARK_MODE_LRU;    // HIP loader does not use the Spark staging
    c->has_embed = (stage == 0);
    c->has_head = (stage == n_stages - 1);
    c->load_indexer = 0;            // qf_hip4_stage.hip: indexer not loaded in v1
    c->tag = "amd-stage";
}

// One hyper-connection block (attn or ffn side): norm + down + up (+ inject).
static uint64_t hc_bytes(const QfPlanCfg *c, int with_inject) {
    uint64_t rsize = (uint64_t)c->hc_count * c->n_embd;
    uint64_t e = rsize + (uint64_t)c->hc_lowrank * rsize + rsize * c->hc_lowrank;
    if (with_inject) e += (uint64_t)c->hc_count * rsize;
    return e * BF16;
}

void qf_plan_compute(const QfPlanCfg *c, QfPlan *p) {
    const uint64_t embd = (uint64_t)c->n_embd;
    const int gdn_vdim = c->ssm_v_heads * c->ssm_d_state;   // 6144

    // ---- dense per-layer weights (bf16) ----
    uint64_t hc2 = 2 * hc_bytes(c, 1);                      // attn + ffn side
    uint64_t moe_common = ((uint64_t)c->n_experts * embd    // router
                         + 3 * (uint64_t)c->n_ff_exp * embd // shared expert
                         + embd) * BF16;                    // shared_expert_gate
    uint64_t gdn = hc2 + moe_common
        + ((uint64_t)c->ssm_d_inner * embd                  // in_proj_qkv
         + (uint64_t)gdn_vdim * embd                        // in_proj_z
         + 4 * (uint64_t)c->ssm_d_inner                     // conv1d
         + 2 * 48 * embd                                    // in_proj_a/b (dt_rank 48)
         + (uint64_t)c->ssm_d_state                         // gated norm
         + embd * gdn_vdim) * BF16                          // out_proj
         + 2 * 48 * 4;                                      // dt_bias, A_log (f32)
    uint64_t att = hc2 + moe_common
        + ((uint64_t)c->n_head * 2 * c->head_dim * embd     // q_proj (q+gate)
         + 2 * (uint64_t)c->n_head_kv * c->head_dim * embd  // k/v_proj
         + embd * c->n_head * c->head_dim                   // o_proj
         + 2 * (uint64_t)c->head_dim) * BF16;               // q/k_norm
    if (c->load_indexer)
        att += (640 * embd + 2 * 128) * BF16;               // index_qk_proj + norms

    p->dense_bytes = (uint64_t)c->n_gdn * gdn + (uint64_t)c->n_attn * att;
    if (c->has_embed) p->dense_bytes += (uint64_t)c->n_vocab * embd * BF16;
    if (c->has_head)  p->dense_bytes += (uint64_t)c->n_vocab * embd * BF16
                                      + hc_bytes(c, 0);     // output HC (no inject)

    // ---- routed-expert residency ----
    // Per slot: gate/up/down packed NVFP4 (n_ff*n_embd/2 each) + f8 block
    // scales (n_ff*n_embd/16 each). weight_scale_2 stays on the host.
    uint64_t slot = 3 * ((uint64_t)c->n_ff_exp * embd / 2
                       + (uint64_t)c->n_ff_exp * embd / 16);
    p->cache_bytes = (uint64_t)c->n_layer * (uint64_t)c->cache_slots * slot;

    // ---- recurrent state ----
    p->state_bytes = (uint64_t)c->n_gdn *
                       ((uint64_t)c->ssm_v_heads * c->ssm_d_state * c->ssm_d_state * 4
                      + 3 * (uint64_t)c->ssm_d_inner * 4)   // conv ring
                   + (uint64_t)c->n_attn * 2 *
                       ((uint64_t)c->max_pos * c->n_head_kv * c->head_dim * BF16);

    // ---- decode workspace (activations; mirrors qf_forward_init) ----
    uint64_t ws = 0;
    ws += embd * 4 * 4;                    // x, mixed, y2560, ed
    ws += (uint64_t)c->hc_count * embd * 4 * 2;   // R, normed
    ws += (uint64_t)c->hc_lowrank * 4 + c->hc_count * 4;    // hc_d, inj
    ws += (uint64_t)c->ssm_d_inner * 4;    // qkv_raw
    ws += (uint64_t)gdn_vdim * 4 * 2;      // z6144, out48
    ws += (uint64_t)gdn_vdim * BF16;       // gdn_out_bf
    ws += (uint64_t)c->n_head * 2 * c->head_dim * 4;        // q6144/q12288
    ws += 2 * (uint64_t)c->n_head_kv * c->head_dim * 4;     // k512, v512
    ws += 2 * 48 * 4;                      // a48, b48
    ws += (uint64_t)c->n_head * c->head_dim * 4;            // attn_out
    ws += (uint64_t)c->max_pos * 4;        // scores
    ws += (uint64_t)c->n_experts * 4;      // router logits
    ws += (uint64_t)c->n_ff_exp * 4 * 3;   // eg, eu, sh
    ws += embd / 2 + embd / 16;            // FP4 activation + UE4M3 scales
    ws += (uint64_t)10 * c->n_ff_exp * 4 * 2; // grouped gate/up expert outputs
    ws += 32 * 4;                          // inv_freq
    if (c->has_head) ws += (uint64_t)c->n_vocab * 4;        // logits
    p->workspace_bytes = ws;

    // ---- fixed pinned staging (Spark full-resident loader only) ----
    p->staging_bytes = (c->mode == QF_SPARK_MODE_FULL) ? QF_SPARK_STAGING_BYTES : 0;

    p->total_bytes = p->dense_bytes + p->cache_bytes
                   + p->state_bytes + p->workspace_bytes + p->staging_bytes;
}

static double gib(uint64_t b) { return (double)b / 1073741824.0; }

void qf_plan_print(const QfPlanCfg *c, const QfPlan *p, FILE *f) {
    fprintf(f, "[budget] %s: mode %s | dense %.2f GiB | cache %.2f GiB"
            " (%d slots x %d layers) | state %.2f GiB | workspace %.2f GiB"
            " | staging %.2f GiB | total %.2f GiB / budget %.2f GiB -> %s\n",
            c->tag ? c->tag : "plan",
            c->mode == QF_SPARK_MODE_FULL ? "full" : "lru",
            gib(p->dense_bytes), gib(p->cache_bytes), c->cache_slots, c->n_layer,
            gib(p->state_bytes), gib(p->workspace_bytes), gib(p->staging_bytes),
            gib(p->total_bytes), gib(p->budget_bytes),
            p->ok ? "OK" : "OVER BUDGET");
}

int qf_budget_check(const QfPlanCfg *c, double budget_gib, FILE *f) {
    QfPlanCfg cfg = *c;
    // mirror the slot clamp in model.cpp: [n_expert_used, n_experts]
    if (cfg.cache_slots < cfg.n_expert_used) cfg.cache_slots = cfg.n_expert_used;
    if (cfg.cache_slots > cfg.n_experts)     cfg.cache_slots = cfg.n_experts;
    if (cfg.mode == QF_SPARK_MODE_FULL)      cfg.cache_slots = cfg.n_experts;
    if (budget_gib <= 0.0) {
        fprintf(f, "[budget] invalid budget %.2f GiB; refusing to allocate\n", budget_gib);
        return -1;
    }
    QfPlan p;
    qf_plan_compute(&cfg, &p);
    p.budget_bytes = (uint64_t)(budget_gib * 1073741824.0);
    p.ok = p.total_bytes < p.budget_bytes;   // strictly below
    qf_plan_print(&cfg, &p, f);
    if (!p.ok) {
        if (cfg.mode == QF_SPARK_MODE_FULL)
            fprintf(f, "[budget] OVER BUDGET by %.2f GiB; refusing to allocate. "
                    "Use QF_EXPERT_MODE=lru, or raise QF_BUDGET_GIB.\n",
                    gib(p.total_bytes - p.budget_bytes));
        else
            fprintf(f, "[budget] OVER BUDGET by %.2f GiB; refusing to allocate. "
                    "Lower %s or raise QF_BUDGET_GIB.\n",
                    gib(p.total_bytes - p.budget_bytes),
                    cfg.n_layer == 48 ? "QF_EXPERT_CACHE_SLOTS"
                                      : "QF4_EXPERT_CACHE_SLOTS");
        return -1;
    }
    return 0;
}

double qf_budget_env(const char *name, double dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    double d = atof(v);
    return d > 0.0 ? d : dflt;
}

int qf_spark_mode_env(FILE *err) {
    const char *v = getenv("QF_EXPERT_MODE");
    if (!v || !*v || !strcmp(v, "lru")) return QF_SPARK_MODE_LRU;
    if (!strcmp(v, "full")) return QF_SPARK_MODE_FULL;
    if (err)
        fprintf(err, "[budget] unknown QF_EXPERT_MODE='%s' (want 'lru' or 'full');"
                " refusing to load\n", v);
    return -1;
}
