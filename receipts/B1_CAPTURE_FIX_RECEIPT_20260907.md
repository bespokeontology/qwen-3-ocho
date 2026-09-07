# Lane B (m1-30tok-0907) — capture-frozen scoring universe fix receipt (2026-09-07 night)

**Defect (QSA_DECODE_ROUND1_RECEIPT_20260907.md):** qf_qsa_index_rows sized the cuBLAS scoring GEMM,
the planes buffer and the fold by `nfull_max = (pos0_host + T) >> 2` from the HOST position. Inside the
captured tail graph (T=1) that value freezes at capture (the first decode step after the prime), so blocks
created during generation were never scored: the top-2048 selection silently ignored the newest context
after an 8K prompt until a recapture.

**Fix (commit 12652f0, binary src/qwenflash_m8.b1fix):** for T == 1 size the GEMM/fold by nb_max
(allocation max; the fold and the selector still read only each row's device-side complete-block count).
g_idx_planes is already sized >= IDX_H*T*nb_max by the prefill path before capture (64 KB at T=1), so no
cudaMalloc inside the graph. Also deleted the stale "attention is DENSE" note (cuda/qf.cu).

**Receipt design:** eager (QF_TAIL_GRAPH=0) recomputes nfull_max from the current host position each step
= the correct universe. Current production binary: eager vs graph must DIVERGE (defect demonstrated).
Fixed binary: graph vs eager must be TOKEN-IDENTICAL (fix proven). Tokens vs the pre-fix 300-id reference
legitimately change once newly-created blocks become selectable.

**Runs (exact 8K production recipe, engrep8k, QF_M8_MAX=300, QF_LAYER_BEGIN=16, frozen AMD server 5578):**
R1 prod-graph, R2 prod-eager, R3 fix-graph, R4 fix-eager. Logs /tmp/qf_b1_r{1..4}_*.log.

```
qf_b1_r1_prod_graph vs qf_b1_r2_prod_eager: tokens=300 identical=97 first_diff_idx=92
qf_b1_r3_fix_graph vs qf_b1_r4_fix_eager: tokens=300 identical=300 first_diff_idx=-1
qf_b1_r1_prod_graph vs qf_b1_r3_fix_graph: tokens=300 identical=97 first_diff_idx=92
--- E2E lines:
qf_b1_r1_prod_graph.log:m8 E2E: M=1, 300 tokens in 10.10s = 29.70 tok/s aggregate (1781.9 tok/min), 29.70 tok/s/req
qf_b1_r2_prod_eager.log:m8 E2E: M=1, 300 tokens in 10.77s = 27.85 tok/s aggregate (1671.1 tok/min), 27.85 tok/s/req
qf_b1_r3_fix_graph.log:m8 E2E: M=1, 300 tokens in 10.24s = 29.30 tok/s aggregate (1758.3 tok/min), 29.30 tok/s/req
qf_b1_r4_fix_eager.log:m8 E2E: M=1, 300 tokens in 11.51s = 26.06 tok/s aggregate (1563.8 tok/min), 26.06 tok/s/req
```

**Verdict: KEEP — identity_fix=false drift_matches=? change_matches=?.** Eager is ~4 ms/step slower than graph on the fixed binary (diagnostic
arm only; production stays in graph mode). The doubled GEMM M-dim (2048 -> 4096 blocks) is ~0.09 ms/step;
no speedup claimed and none expected — this is the correctness fix the top-512 selector (B2) will optimize over.

Binaries: prod src/qwenflash_m8 sha 2495e5ad17bdc50f (archived qwenflash_m8.prod_420bca8_0907);
fix src/qwenflash_m8.b1fix sha fdb9e05a3acc7f460854a6ab.
