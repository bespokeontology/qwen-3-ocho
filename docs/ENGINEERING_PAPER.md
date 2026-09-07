# The Qwen3.8-Flash-Next Heterogeneous Engine: Native Inference and Resident-State Stochastic Computation

**Version:** 0.1.0-rc1 — release candidate, not peer-reviewed, not pushed publicly.
**Date:** 2026-09-07. **Authority:** the canonical freeze package FREEZE_QWEN_OVERNIGHT_20260907
and its receipts; see freeze/PROVENANCE.md.

## Abstract

This report documents a two-accelerator native-inference appliance for a Qwen3.8-Flash-Next-class
model — four MI50 (gfx906) devices owning the model prefix, PLE, and output head, and an NVIDIA
GB10 owning the tail — and a GPU-native stochastic computation engine built on the appliance's
resident state. The stochastic engine treats one conditioned resident model state as the initial
condition of a field of independent branches: the state is forked, branches advance under a
deterministic counter-based RNG, dead branches are pruned and compacted, observables are
accumulated, and the field is reduced on the GPU to statistical quantities. The reduction
primitive was validated against CPU references; the integration path was validated against
integrals with closed-form answers; stratified sampling reduced estimator error by approximately
188x at equal wall time; a synthetic conditioned field matched three analytic expectations while
sustaining approximately 1.6e9 trajectory-steps/s; and a model-backed field (eight branches of a
resident parent, seven stochastic, one greedy control) produced field-level statistics without
exporting trajectories to the CPU. On the inference side, an authoritative baseline of 29.70
native tok/s at an 8,038-token context was measured; a correctness fix and an exact selector
replacement were retained; the 30 tok/s target was not reached (29.44 tok/s retained state), and
the remaining measured lever — the AMD long-context tax — is gated on recovering the exact source
tree of the frozen production AMD binary, which has not been recovered. Negative results and
provenance limits are reported explicitly.

## 1. Motivation

Two questions drove this work. First: can an autoregressive model with resident causal state be
used as an expensive conditioned transition operator inside a larger stochastic computation —
branching, evaluating, and reducing trajectories without exporting each trajectory to the CPU?
Second: can hardware that is obsolete in conventional single-device inference comparisons remain
materially useful when the architecture is decomposed around it? The appliance and the stochastic
engine are two answers to those questions, built on the same resident state.

## 2. Hardware and constraints

Table 1. Hardware.

| device | role | memory | notes |
|---|---|---|---|
| 4x AMD MI50 (gfx906) | prefix 0..15, PLE, head (production); prefix 0..7 (fork topology) | 16 GiB HBM2 each | custom HIP kernels, pinned clocks |
| NVIDIA GB10 (sm_121a) | tail 16..47 (production); 8..47 (fork topology) | unified | CUDA 13.0, locked 3003 MHz |

Precision is fixed by earlier freezes: FP8 dense weights (E4M3 per-64-block scales), NVFP4 routed
experts on Spark, INT8 expert arenas on the production AMD tier (NVFP4+INT8 in the fork topology),
higher-precision PLE and head. Precision changes were out of scope.

## 3. Heterogeneous decomposition

The model is split at a layer boundary (QF_LAYER_BEGIN). Each decode step: AMD prefix advance,
residual over the network, Spark tail (dense FP8 projections, routed experts, hyper-connections,
GDN recurrence, QSA sparse attention), residual back, AMD fused head plus next-prefix preparation.
The AMD tier is on the critical path twice per token. Figure 1
(figures/heterogeneous_pipeline.svg). Assignment follows physical strengths: HBM-resident layers
and state on the AMD tier, tail compute on the GB10. The split is not a benchmark artifact; it is
the serving configuration.

## 4. Resident causal state

Both tiers keep everything resident: weights, KV, GDN recurrent state, convolution rings, QSA
indexer pools, expert slot maps. Refill ingests new context without replay; continuation resumes
any resident state; fork copies a primed parent into M row slots on both tiers. Resident state is
what makes the stochastic overlay possible: the fork is a state copy (milliseconds), not a
recompute.

## 5. Production inference engine

The frozen prefill receipt: 8,038 tokens in 5.21 s (≈1,543 tok/s) with the production
heterogeneous path. Refill receipts: 136 tokens in 0.60 s; 2,038 tokens in 2.19 s; 5,038 tokens
in 4.06 s (≈1,240 tok/s); continuation 26-28 tok/s. Details in docs/PREFILL_REFILL.md.

Native M=1 decode at the 8,038-token production recipe, 300 generated tokens, locked clocks:

Table 2. Native decode (receipts B1/B2).

| configuration | Spark tail | AMD fused head | tok/s |
|---|---|---|---|
| frozen production binary (baseline) | 23.0 ms/token | 10.7 ms/token | 29.70 |
| B1+B2 retained | 23.1 ms/token | 10.8 ms/token | 29.44 |

The ≥30 native tok/s target was **not reached**. The retained changes are a correctness fix and an
exact selector replacement (sections 16-17); their net end-to-end effect is approximately
+0.1 ms/token, inside run-to-run variance. The remaining distance to the 33.33 ms/token budget is
approximately 0.34 ms/token. No speculative decoding, no MTP, no precision change is involved in
any of these numbers; all are native single-request execution.

## 6. The GPU-native stochastic field abstraction

The central architectural observation: an autoregressive model with resident causal state can be
treated as an expensive conditioned transition operator inside a larger stochastic computation.
The pipeline (Figure 2, figures/stochastic_pipeline.svg):

    resident conditioned state -> GPU fork -> deterministic counter RNG -> field advance
    -> survival/compaction -> observable evaluation -> reduction -> small result to host

The distinguishing property is not that multiple continuations can be generated — it is that one
expensive resident state becomes the initial condition of a GPU-resident field whose trajectories
are evaluated and reduced without exporting every trajectory to the CPU. Two primitives are kept
conceptually separate: the Kolmogorov advance (evolve the conditioned field) and the Dumitrescu
reduction (evaluate and reduce it). Separation survives kernel fusion; generation and estimation
are different concerns.

Branch state is small and classified: per-branch mutable append state (KV rows, indexer pools),
must-copy state (GDN, convolution rings, PLE), and cheap metadata (position, token, log-weight,
observable accumulator, alive flag). Shared immutable history is never rewritten by children
(demonstrated by the fork-independence receipt). RNG state is not stored: streams are addressed by
(seed, branch, step, lane) counters. At 16 B per branch descriptor, 1M branches cost 0.08 ms to
initialize; copy-on-write is unnecessary at this scale. See stoch/BRANCH_DESCRIPTOR.md.

A branch's identity is therefore fully determined by (parent state, seed, branch slot, step):
forking copies the small mutable state, and the trajectory is reproducible from the counters
alone. Compaction changes slot numbering, but because the step word is part of every RNG address,
a slot reused in a later step never collides with a stream already drawn; determinism survives
compaction by construction. The log-weight field doubles as the termination marker (lw <= -1e300
means dead), so pruning, weighting, and reduction share one representation.

## 7. Deterministic RNG

Philox4x32-10 implemented from the project's frozen RNG specification (key=(slot,round);
counter=(position,branch,purpose,index); uniform=(word0>>8)*2^-24). Known-answer test
(counter=key=0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8) passes on host and device; determinism,
stream independence (2,048 cells, no aliasing), counter snapshot/restore, and 1M-uniform
mean/variance checks all pass. Counter-based addressing is what makes fork, replay, and branch
identity cheap: no RNG state to copy or restore.

## 8. Kolmogorov field advance

The synthetic field exercises every mechanism on an analytically checkable process. A parent
x0=1.0 forks into M branches of Ornstein-Uhlenbeck dynamics (a=0.50, sigma=1.00) with Box-Muller
noise from the counter RNG. Each step: advance, kill coin (p=0.01) with weight compensation
1/(1-p) (unbiased Russian roulette), exclusive-scan compaction, fused observable accumulation.
Three closed-form gates: E[X_T] = a^T x0; E[X_T^2] = (a^T x0)^2 + sigma^2 (1-a^{2T})/(1-a^2);
E[(1/T) sum x_t^2]. All three matched within expected Monte Carlo error; survival (52.5%)
matched the kill-coin prediction (0.99^64 = 52.6%); ESS equaled the survivor count (equal
weights). Throughput scales from 2.6e7 trajectory-steps/s at M=1,024 to 1.60e9 at M=1,048,576,
saturating above ~262K branches (launch-bound below ~16K).

The pruning rule is the field's first weighted mechanism and deserves a precise statement. A
branch that survives step t has its log-weight decreased by log(1-p); a branch killed at step t
is removed. For an observable accumulated at step t, E[w_t * f_t * 1_alive(t)] = E[f_t] *
(1-p)^t * (1-p)^(-t) = E[f_t], because the kill coins are independent of the process: the
weights exactly compensate the deletion, so the weighted mean over survivors estimates the
unpruned expectation. This is the same structure weighted model-backed Monte Carlo will use,
with the branch's conditional log-probability replacing the kill-coin compensation.

## 9. Dumitrescu statistical reduction

The reduction primitive computes N, sum w, sum w^2, sum w f_k, sum w f_k^2, max log-w,
log-sum-exp(log w), and ESS for K observables per branch in one pass: fp64 accumulation, per-block
partials, deterministic host merge (no cross-block atomics), zero-weight sentinel. Validated
against a CPU fp64 reference on 14 cases (odd sizes, 30% dead, weights 1e-30..1e30, all-zero,
K=1/2/4, signed observables) within 1e-12 (sums) and 1e-9 (moments). The primitive is the
statistical boundary of the field: what crosses back to the CPU is a handful of merged scalars,
not branch values.

## 10. Monte Carlo integral validation

Table 3 (receipt MC_GATES_RECEIPT_20260907.md; GB10; seed 42; all sampling, integrand, and
reduction on GPU).

| gate | analytic | N | estimate | |err| | wall | samples/s | notes |
|---|---|---|---|---|---|---|---|---|
| MC-A: E[X^2], X~U(0,1) | 1/3 | 1,048,576 | 0.333674980 | 3.4e-4 | 1.9 ms | 5.4e8 | unweighted; 1-sigma ~2.9e-4 |
| MC-A2: E[X^2] via q(x)=2x | 1/3 | 1,048,576 | 0.333907284 | 5.7e-4 | 2.1 ms | 5.0e8 | weighted; ESS ~3.1e5 |
| MC-B: E[sum X_j^2], d=16 | 16/3 | 262,144 | 5.334712595 | 1.4e-3 | 0.65 ms | 4.1e8 | vector RNG |

These establish that the fused sample-evaluate-reduce path is numerically correct end to end,
weighted and unweighted, scalar and multidimensional. Structurally, each gate is one kernel: a
grid-stride loop draws the sample from the counter RNG, evaluates the integrand in registers,
accumulates fp64 partial sums per thread, reduces within the block, and writes a block partial;
the host merges the partials in double. No intermediate sample or branch vector is ever
materialized. The same accumulation structure is the one used by the full field engine, so these
gates validate the reduction boundary the field depends on.

## 11. Variance reduction

Stratified (jittered) sampling, S=64 strata: u = (stratum + uniform)/64. Identical N, identical
wall (1.8 vs 1.9 ms): plain MC-A error 3.4e-4; stratified error 1.8e-6 — approximately 188x
lower absolute error at equal wall time (Figure 6, figures/variance_reduction.svg). The lesson is
metric, not mechanism: for Monte Carlo systems the optimization target is statistical work per
second — estimator error at fixed wall, ESS, trajectories/s — rather than samples/s alone. A
sampling rule change costs nothing here and buys two orders of magnitude of accuracy.

## 12. Stochastic rendering

Rendering is defined as visualizing statistical structure generated by a conditioned stochastic
process, not raster graphics. The synthetic field's terminal pairs (X_T, X_{T-1}) are binned on
the GPU into a 512x512 density grid and written as a PPM (figures/kolmo_field_density.ppm, and a
lossless PNG conversion). Parameters: M=1,048,576; T=64; a=0.50, sigma=1.00; x0=1.00; Philox seed
42; p_kill=0.01 with weight compensation; 44.7 ms advance wall; ~1.5e9 trajectory-steps/s;
ESS 550,847. The artifact demonstrates the general chain — stochastic samples -> field evolution
-> observable -> reduction/render — as first-class machinery rather than a graphics demo.

## 13. Model-backed conditioned stochastic evaluation (MC-C)

The bridge experiment: one resident parent (382 tokens, 8/40 split, frozen fork binaries) is
forked into M=8 rows. Row 0 is the greedy control; rows 1-7 sample temperature-1 Gumbel-max
transitions (server-side sampler, deterministic splitmix64 keyed by (seed,row,v)). After 16 steps
the per-row state signatures (sigR: sum of GDN+convolution state over Spark layers; sigK: sum of
K/V of the last 4 positions) are read back and reduced unweighted over the 7 sampled rows:
field mean sigR +228 vs greedy-row sigR -704 (sample std 1076, ESS 7); terminal-token
distribution 318 x3, 264, 19, 8260, 53235, 156566; execution 367.5 ms/step (21.8 sibling-tokens/s;
the full-vocab Gumbel loop costs ~110 ms/step). A repeat run reproduced the token trajectories
bitwise (the signature values vary within the documented M-row arithmetic band — the rows start
from bit-identical forked state and the same token, yet float arithmetic in the M-row body is not
bit-stable across runs; token identity is the project's gate, and it holds). The significance is
architectural: the same field machinery operates on an actual resident model state, not only a
synthetic process. The reduction here is deliberately trivial — an unweighted mean over seven
sampled branches — because the region wire carries token ids, not per-row log-probabilities; the
moment the wire carries the sampled token's conditional probability, the same field reduces to a
properly weighted Monte Carlo estimate of a model-conditioned expectation, with ESS reporting for
free. No claim is made beyond that: this does not
demonstrate general Bayesian inference or world-model simulation. sigR is a deterministic
functional of branch recurrent state; its interpretation is limited to a trajectory observable.
M>8 is not validated (section 18).

## 14. Why old AMD hardware mattered

Section 9 of the repository README and docs/AMD_HETEROGENEOUS.md carry the full case; the
substance: the four gfx906 devices own the prefix, PLE, and head of every production token, and
their remaining long-context tax is the largest measured lever between the appliance and the
30 tok/s target. Accelerator value here comes from decomposition — assigning work to match HBM
capacity/bandwidth, state ownership, and custom HIP kernels — not from isolated device
benchmarks. Two hypotheses about the AMD tax (HIP graphs; selector bisection) were tested and
rejected by controlled comparison, which bounds where the next milliseconds can come from. The
claim is deliberately narrow: heterogeneous decomposition made hardware that is obsolete in
conventional single-device comparisons materially useful in a modern serving and
stochastic-compute system.

## 15. Performance characterization

Table 4. Measured walls (receipts; units as stated).

| quantity | value | context |
|---|---|---|
| prefill | 5.21 s / 1,543 tok/s | 8,038 tokens, frozen binary |
| refill | 0.60 / 2.19 / 4.06 s | 136 / 2,038 / 5,038 tokens |
| decode step | 33.7 ms | 8K production recipe (baseline) |
| AMD fused head | 10.7 ms (baseline) | 8K; 8.2-8.6 ms short context |
| Spark tail | 23.0 ms (baseline) | 8K; 23.8-23.9 ms short context |
| synthetic field | 1.60e9 trajectory-steps/s | M=1,048,576, T=64, GB10 |
| model-backed field | 21.8 sibling-tokens/s | M=8, 16 steps, 8/40 split |
| standalone MC | 4.1-5.8e8 samples/s | GB10, fused kernels |

Every number maps to a receipt (docs/REPRODUCIBILITY.md lists the mapping). Units are not
interchangeable across rows.

## 16. Correctness methodology

Bit identity is not universally required; each mechanism has its own gate: token identity where
semantics must be exact (B1, B2); analytic expectation for Monte Carlo kernels; known-answer for
RNG; CPU-reference for reductions; deterministic seeded replay; branch independence; state
signatures; causal-equivalence continuation; compute-sanitizer memcheck; hash provenance. The
working discipline is one mechanism at a time — strict build, smallest physical gate, end-to-end
receipt, then KEEP / ROLLBACK / PROMOTE — with the operator as the sole authority on output
quality. Concrete cases: B1 was gated by eager-vs-graph token identity (drift at token 92 before
the fix, 300/300 identical after); B2 by a 13-case unit harness plus memcheck plus a 300/300
token-identical receipt; the RNG by its known-answer test; the reduction by CPU reference. See
docs/CORRECTNESS.md and docs/BUILD_LAW.md.

## 17. The production-lane mechanisms

**B1 — capture-frozen scoring universe (correctness).** The QSA indexer sized its scoring GEMM,
planes buffer, and fold by nfull_max = (pos0_host + T) >> 2 from the host position. Inside the
captured decode graph (T=1) that value freezes at capture: blocks created during generation were
never scored, and the top-2048 selection silently ignored the newest context. Discriminator:
eager decode (recomputes the bound each step) vs captured graph. The pre-fix binary drifts at
token 92 (97/300 identical); the fixed binary (GEMM sized by the allocation maximum at T=1) is
300/300 token-identical between eager and graph. Cost: the doubled GEMM dimension, ~+0.5 ms/token.

**B2 — exact single-pass top-512 selection.** The 48-step threshold bisection was replaced by a
4-pass radix select over order-preserving uint32 float keys with identical tie semantics (mask
sc > t, fill ties at t in ascending block order, degenerate all-non-positive case selects
{sc > 0}). Validation: 13-case unit harness against a host implementation of the original
bisection plus memcheck (0 errors); then a 300/300 token-identical production receipt and tail
23.5 -> 23.1 ms/token. No major end-to-end speedup is claimed; the gain is ~0.4 ms/token of tail
and the removal of a 48-step dependency.

**B3 — AMD twin (port verified, integration blocked).** The same radix construct was ported to
the AMD top-k kernels and unit-verified on gfx906 (10/10 non-dense cases exact). Integration was
attempted and refused: every rebuild diverges at token 0 from the frozen production binary, with
and without the radix, because the frozen binary's true source tree has not been recovered and
the available mirror is only partially hash-verified. The port is retained as experimental,
unit-verified, pending integration.

## 18. Negative results

- **30 tok/s not reached.** 29.70 baseline; 29.44 retained. The target was not met and is not
  rounded to it. The retained changes were correctness and exactness work, not a speedup
  mechanism; their net effect is inside variance.
- **AMD selector hypothesis rejected.** The radix and the bisection produce the same measured
  head time (10.8 ms/token) in controlled rebuilds; the 2048-boundary tax is located in the QSA
  scores kernel and flash attention growth instead.
- **M>8 not validated.** At M=16 the second slot's state signatures are out of scale and slot 0
  drifts relative to the M=8 run. M=8 remains the validated stochastic width; M=16 numbers are
  not published as valid stochastic scaling.
- **AMD provenance unresolved.** The exact source of the frozen production AMD binary is
  missing; no mirror is authoritative (section 19).

## 19. Reproducibility and source provenance

The standalone stochastic kernels build and run from stoch/ on any CUDA GPU >= sm_70
(scripts/reproduce_*.sh; the retained builds target sm_121a and the scripts honor QF_ARCH).
The production-lane changes are patches against m8-amd-routed @ 420bca8, which is not part of
this tree. Model-backed experiments require the two-box appliance and the frozen binaries.
Provenance labels per component are in freeze/PROVENANCE.md; the AMD production source is
PENDING-PROVENANCE, and no source reproducibility is claimed for it.

## 20. Limitations

- AMD production source tree not recovered; AMD work frozen (above).
- Native 30 tok/s not reached; the next mechanism (AMD QSA layer to its byte floor) is gated on
  provenance.
- M>8 stochastic width unvalidated.
- The region wire carries token ids, not per-row log-probabilities; weighted model-backed Monte
  Carlo requires a wire extension.
- The model-backed sampler scans the full vocabulary (~110 ms/step at M=8).
- MC-C is single-seed at present (the seed is read at server boot); two-seed coherence requires
  server restarts.
- sigR has no broader statistical interpretation; it is a trajectory observable.

## 21. Future engineering work

In order of gate-readiness: (1) recover and hash-match the exact AMD source tree of 824e5c50,
then integrate the verified radix and attack the QSA scores/attention tax; (2) Spark routed
experts (~2.2 ms above roofline floor) and hyper-connections (~2.1 ms), whose provenance is
intact; (3) restricted-candidate Gumbel sampling and per-row log-prob on the region wire for
weighted MC-C; (4) an M>8 correctness round for the second slot; (5) a GPU-resident field loop
(persistent orchestration) as the long-horizon target.

## 22. Conclusion

One resident model state was made the initial condition of a GPU-resident stochastic field:
forked, advanced under deterministic per-branch randomness, pruned, compacted, observed, and
reduced — with the integration path validated against closed-form integrals and the reduction
path against CPU references, and with a measured variance-reduction mechanism that buys two
orders of magnitude of accuracy at equal wall. The same appliance measures 29.70 native tok/s at
8K context, retains one correctness fix and one exact selector replacement, and does not reach
30; the distance and the blocker are measured and stated. The engineering record, including its
failures, is in the receipts.

## Appendix A. Receipt index

B1_CAPTURE_FIX_RECEIPT_20260907.md; B2_RADIX_RECEIPT_20260907.md; B3_AMD_TWIN_RECEIPT_20260907.md;
DUMI_RECEIPT_20260907.md; KOLMO_FIELD_RECEIPT_20260907.md; MCC_MODEL_FIELD_RECEIPT_20260907.md;
MC_GATES_RECEIPT_20260907.md; RNG_RECEIPT_20260907.md; THROUGHPUT_LAW_RECEIPT_20260907.md;
VARIANCE_REDUCTION_RECEIPT_20260907.md. Prefill/refill/fork receipts: the earlier canonical
freezes referenced in docs/PREFILL_REFILL.md.
