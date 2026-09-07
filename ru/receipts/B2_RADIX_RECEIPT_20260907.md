# Lane B (m1-30tok-0907) — exact single-pass top-512 selection receipt (2026-09-07 night)

**Mechanism:** replaced the 48-step threshold bisection in k_idx_select_rows (0.59 ms/token, 8 QSA
layers) with a 4-pass radix select (MSB first) over the order-preserving uint32 float keys, preserving
the exact selection rule: mask sc > th, fill ties at th in ascending block order; degenerate case
(< 512 strictly positive scores) selects { sc > 0 } with no fill, bit-identical to the bisection's
lo == hi == 0 terminal state. One 256-thread block per row, shared hist[256] + rank offset sh_above
(the 512th of the whole row is the (512 - count_above_prefix)-th of the prefix population).

**Unit gate (b2test/test_selector.cu, committed):** kernel copy vs a host reference of the original
bisection — random 513/1024/4096/2009, quantized ties, all-zero, 100-positive, all-equal, boundary
ties, two-level ties, dense 512/300, pre=0 in-kernel scoring: **13/13 PASS**; compute-sanitizer memcheck
**0 errors**.

**Defects caught by the gate (all three fixed before acceptance):**
1. pass-0 prefix mask `0xFFFFFFFFu << 32` (UB; hardware masks the shift) — first binary selected
   garbage, overflowed the 513-slot list, decode crashed with an illegal memory access at step 0.
2. threshold `th` held in a per-thread register while only thread 0 advanced it — lower bytes stayed 0
   for 255/256 threads (same crash class).
3. rank offset lost in a cleanup edit — each pass searched for the 512th item of the PREFIX population
   instead of rank 512 - count_above (threshold too low, list overflow).
The token-identity receipt would have caught every one of these; the unit gate caught them faster.

**8K production receipt (engrep8k, QF_M8_MAX=300, frozen AMD server 5578):**
R3 (b1fix, bisection): tail 23.5 ms/token, E2E 29.30 tok/s. R7 (b2fix, radix): tail 23.1, head 10.8,
E2E 29.44 tok/s; **tokens 300/300 identical, first_diff = none.**

**Verdict: KEEP — exact, -0.4 ms/token Spark tail, no output change.** vs the authoritative production
binary (R1, 29.70 tok/s): B1's correctness fix (+0.5 ms GEMM) minus B2 (−0.4 ms) nets ≈ +0.1 ms/token,
inside run variance. The ≥30 tok/s target is dominated by the AMD 2048-boundary tax (head 10.8 ms) —
the AMD twin (B3) is the same construct in k8_idx_topk_rows.

Binary: src/qwenflash_m8.b2fix sha 597eba85e85bc5e065ea5e7c. Logs /tmp/qf_b2_r7_fix_graph.log (+ R5/R6 crash logs).
