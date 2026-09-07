# Prefill and resident refill

These mechanisms were frozen before the overnight stochastic work and are documented here because
they establish what "resident state" means in this appliance. All numbers come from the earlier
canonical freezes (FREEZE_QWEN_PREFILL_1543_20260906 and the refill receipt); no changes were made
in this window.

## Frozen prefill

Production heterogeneous prefill, 8,038-token prompt, T=2048 logical chunks, AMD prefix +
Spark tail: **5.21 s ≈ 1,543 tok/s** (same-binary band 5.21-5.42 s). The retained progression,
one 8K receipt per accepted step:

| step | prime |
|---|---|
| two AMD chunks in flight | 6.02 s |
| fused HC/router glue | 5.78 s |
| fused GDN post | 5.70 s |
| AMD union-tile QSA | 5.44 s |
| bulk CSR + 64-token expert pass | 5.26 s |
| final production receipt | 5.21 s |

Earlier receipts on the same binaries: 382 tokens in 0.85 s; 2,038 tokens in 2.18 s; 5,038 tokens
in 4.35 s. Known-worse variants (512-token first chunk, halving tail split, 8-warp expert forms,
sequential GDN, hand fp8 GEMM, list-kernel attention, per-position AMD QSA) remain OFF by default
and are documented in the freeze.

## Resident refill / context injection

Refill ingests additional context into a resident state without replay: the ingest region is
processed chunked through the prefill path; exit-state continuation resumes decode. A graph
lifetime rule applies: the captured tail graph is invalidated at ingest and recaptured
(~4 ms). Measured: 136 tokens in 0.60 s (TTFT 0.65 s); 2,038 tokens in 2.19 s with 1024
sub-chunks; 5,038 tokens in 4.06 s (≈1,240 tok/s); chaining to a 15.8K resident context;
continuation at 26-28 tok/s; A/B against a cold run: 41 identical tokens, then phrasing
divergence (accepted). A tiny-suffix admission floor of ~0.6 s was recorded.

## Why this matters for the stochastic lane

Resident state is the precondition for everything the stochastic engine does: it can be
continued, extended without replay, forked into branch fields, and conditioned further. The
fork that starts every field experiment is a copy of a state this machinery created; the fork
receipt (earlier freeze) established eight physically independent causal children from one
parent without replaying it.
