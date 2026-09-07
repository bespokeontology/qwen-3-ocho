# Lane B (m1-30tok-0907) — AMD k8_idx_topk twin (2026-09-07 night): PORTED + UNIT-VERIFIED, INTEGRATION BLOCKED

**Mechanism:** the B2 radix top-512 (4 MSB-first passes, shared threshold, rank offset sh_above,
degenerate {sc>0} branch) ported into the AMD qf8_qsa.hip kernels k8_idx_topk (1024-thread single) and
k8_idx_topk_rows (256-thread rows), preserving the exact bisection rule (mask sc > th, ties at th in
ascending block order, uint8 masks, wave-64 compaction, dense rows nsel=-1).

**Unit gate (test_topk.hip, run on the AMD box, HIP gfx906):** kernel vs host reference of the original
48-step bisection — random 513/1024/4096/2009, quantized ties, all-zero, 100-positive, all-equal,
boundary ties, two-level ties, dense 512/300: **10/10 non-dense cases PASS** (dense rows return the
production nsel=-1 convention — expected).

**Integration receipts — BLOCKED, attribution clean:**
- R8 (fork-era tree + radix): tokens 0/300 vs frozen production. Fork-era tree's stage/ple/dense/server
  files all drifted from efb0e3c (md5-checked) — unrelated to the radix.
- R9 (repo mirror + radix): 0/300. R10 (repo mirror WITHOUT radix): 0/300. The mirror's 6 freeze-listed
  files md5-match, but its remaining files differ from the frozen production binary's TRUE build inputs,
  which were never archived (the prefill freeze dir holds the binary + SHA256SUMS only). **The radix is
  exonerated by the unit gate and by R10; the blocker is source provenance.** Rebuilds cannot be trusted
  for a production receipt until the operator supplies the exact AMD source tree of 824e5c50.

**Timing finding (measured, twice):** with a correct radix the AMD head stays 10.8 ms/token — the
48-step bisection is NOT the 2048-boundary tax. The handoff's hypothesis is refuted; the tax lives in
the AMD QSA scores kernel + flash attention growth (~0.8 ms per QSA layer at 8K vs 0.46 short).

**State:** ported qf8_qsa.hip saved at ~/qf-hip4-b3prod/qf-hip4-wave64/src/qf8_qsa.hip.b3radix (AMD
box) + /tmp/qf8_qsa.hip (this box); original at qf8_qsa.hip.pre_b3 / mirror commit. test_topk.hip in
the Build Law dir? No — /tmp/test_topk.hip + shipped to <amd-host>:/tmp/. Ready to apply the moment the
true sources land. Verdict: mechanism KEEP-READY; integration DEFERRED on provenance.
