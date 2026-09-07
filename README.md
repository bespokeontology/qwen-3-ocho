# Qwen 3.ocho (C++) — Stochastic Trajectory Rendering with a Decision/Commit Loop on a Native CUDA/HIP/ROCm Inference Engine (NVIDIA GB10 + 4x AMD MI50)

Native M=1 decode on this appliance: 27–29 tok/s on the 8K production recipe with no speculative
decoding (frozen-binary receipts, section 4). Qwen 3.ocho's stochastic modes run alongside that
path; their per-row rendering cost is separate and is not a substitute for it.

A two-accelerator, resident-state inference and stochastic-computation appliance built around a
Qwen3.8-Flash-Next-class model. One box carries a 4x MI50 (gfx906) AMD tier; the other carries an
NVIDIA GB10 tier. The AMD tier owns the model prefix, the PLE layer, and the output head. The NVIDIA
tier owns the model tail. Neither accelerator holds the whole model.

This repository is the engineering record of the 2026-09-07 overnight window: the stochastic
computation lane (new) and the production decode lane (correctness fix and one retained selector
change). It is a release candidate derived from the canonical freeze package
(`FREEZE_QWEN_OVERNIGHT_20260907/`); the package and its SHA256SUMS are the authority. See
`freeze/PROVENANCE.md` for the authority hierarchy and per-component provenance labels.

## 1. What this repository is

A native (non-speculative) model execution appliance built on CUDA (NVIDIA GB10, sm_121a) and HIP/ROCm (4x AMD MI50, gfx906), plus a GPU-native stochastic
evaluation engine that treats the model's resident causal state as a conditioned transition operator.
The engine forks one resident parent state into a field of independent stochastic branches, advances
them on the GPU, prunes and compacts them, evaluates observables, and reduces the field to statistical
quantities without exporting every trajectory to the CPU.

The two subsystems are connected: the resident-state machinery that serves production requests
(prefill, refill, continuation) is the same machinery the stochastic lane forks.

## 2. Hardware

| device | role | memory | notes |
|---|---|---|---|
| 4x AMD MI50 (gfx906) | prefix layers 0..15, PLE, lm_head (production); prefix layers 0..7 (fork topology) | 16 GiB HBM2/card | custom HIP kernels |
| NVIDIA GB10 (sm_121a) | tail layers 16..47 (production); layers 8..47 (fork topology) | unified memory | CUDA 13.0 |

The split is configurable (`QF_LAYER_BEGIN`), the AMD prefix width is configurable
(`QF_AMD_PREFIX`), and expert arenas may be INT8-only (production) or NVFP4+INT8 (fork topology).

## 3. Architecture

```
                +-----------------------------------------+
request  ---->  |  AMD: prefix layers, PLE               |
                +-------------------+---------------------+
                                    | residual (network)
                +-------------------+---------------------+
                |  Spark/NVIDIA: tail layers, QSA index  |
                +-------------------+---------------------+
                                    | residual (network)
                +-------------------+---------------------+
                |  AMD: fused head + next-prefix window  |
                +-------------------+---------------------+
                                    | token ids
                                    v
```

See `docs/ARCHITECTURE.md` and `figures/heterogeneous_pipeline.svg`. The AMD tier is compute on
both ends of the critical path, not storage.

## 4. What was actually measured

Native M=1 decode, 8,038-token resident prompt, 300 generated tokens, locked clocks, frozen binaries:

| configuration | Spark tail | AMD fused head | tok/s |
|---|---|---|---|
| frozen production binary (authoritative baseline) | 23.0 ms/token | 10.7 ms/token | 29.70 |
| B1+B2 retained changes | 23.1 ms/token | 10.8 ms/token | 29.44 |

The retained changes are a correctness fix (B1) and
an exact selector replacement (B2); their net end-to-end effect is approximately +0.1 ms/token, inside
run-to-run variance.
See `receipts/B1_CAPTURE_FIX_RECEIPT_20260907.md`, `receipts/B2_RADIX_RECEIPT_20260907.md`, and
`docs/LIMITATIONS.md`.

Frozen prefill (separate freeze, unchanged by this work): 8,038 tokens in 5.21 s ≈ 1,543 tok/s.
Resident refill/context injection continues existing state without replay (receipts in
`docs/PREFILL_REFILL.md`).

## 5. Qwen 3.ocho — stochastic trajectory rendering with a decision/commit loop

The model-backed product layer of the stochastic lane. One resident parent state is forked into
eight per-row continuations on both accelerators; row 0 stays greedy, rows 1–7 sample by
deterministic per-row Gumbel-max noise (seed-parametrized). The eight rows advance together
through the M-row region path (AMD prefix -> Spark tail -> AMD head).

- **Ocho Beam**: the eight futures are the output. K=16 field: 5.16 s, 24.8 sibling-tok/s.
  K=1500 beams (C++/Rust tasks): ~850 s each, ~14 sibling-tok/s; temperature-1.0 rows accumulate
  tail-vocabulary artifacts over long horizons while the greedy row stays clean.
- **Ocho Loop**: render -> evaluate -> choose -> commit -> continue. Evaluation is simple and
  fully logged (early-EOS and artifact-token penalties, then consensus distance on the
  recurrent-state signature). The chosen row's full state is committed back to the parent slot on
  both boxes. Capability run: K=48 x 16 iterations = 768 committed tokens of continuous design
  reasoning, ~24 s per cycle, clean exit; same-boot field-identity gate bit-identical.

Full mechanism, measured behavior, repairs, and limitations: docs/QWEN_3_OCHO.md. Frozen
capability artifacts: FREEZE_OCHO_20260907 and FREEZE_OCHO_CAPABILITY_20260907 (in the canonical
freeze package tree; SHA256-verified).

## 6. Stochastic compute engine

```
  resident conditioned state
        |
  GPU-native fork
        |
  deterministic counter-based RNG (seed, branch, step, lane)
        |
  Kolmogorov field advance (branch-local state, shared immutable history)
        |
  branch survival / pruning / compaction
        |
  observable evaluation
        |
  Dumitrescu reduction (N, Σw, Σw², Σw f, Σw f², log-sum-exp, ESS)
        |
  expectation / integral / distribution / rendered field
        |
  small result to the host
```

The distinction that matters: one expensive resident causal state becomes the initial condition of a
GPU-resident stochastic field, whose trajectories are evaluated and reduced on the GPU. The CPU never
iterates over branch values.

Components (all receipt-checked; see `receipts/` and `docs/STOCHASTIC_ENGINE.md`):

- **RNG** — Philox4x32-10, counter-based, implemented from the project's frozen RNG specification
  (key=(slot,round); counter=(position,branch,purpose,index)). Known-answer-tested on host and device.
- **Dumitrescu reduction** — the project's name for its weighted field-reduction primitive. fp64
  accumulation, per-block partials, deterministic host merge, multiple observables per branch,
  zero-weight sentinel. Validated against a CPU fp64 reference over 14 cases.
- **Kolmogorov field advance** — conditioned parent forked into M branches; branch-local mutable state
  plus shared immutable history; per-step kill-coin pruning with weight compensation (unbiased);
  exclusive-scan compaction; fused observable accumulation. Three analytically known observables
  matched their closed forms in the synthetic experiment.
- **Dynamic branch survival** — pruning + compaction are exercised at every step; dead branches
  contribute zero weight and are removed.

Throughput: the synthetic field sustains approximately 1.6e9 trajectory-steps/s at M ≥ 262,144
branches on the GB10 (launch-bound below ~16K branches). See `receipts/THROUGHPUT_LAW_RECEIPT_20260907.md`.

## 7. Monte Carlo integration

The engine was validated against integrals with closed-form answers, with sampling, integrand
evaluation, and reduction all on the GPU:

| gate | analytic | N | estimate | |err| | wall | samples/s | notes |
|---|---|---|---|---|---|---|---|---|
| MC-A: E[X²], X~U(0,1) | 1/3 | 1,048,576 | 0.333675 | 3.4e-4 | 1.9 ms | 5.4e8 | unweighted |
| MC-A2: E[X²] via q(x)=2x | 1/3 | 1,048,576 | 0.333907 | 5.7e-4 | 2.1 ms | 5.0e8 | weighted, ESS 3.1e5 |
| MC-B: E[Σ X_j²], d=16 | 16/3 | 262,144 | 5.334713 | 1.4e-3 | 0.65 ms | 4.1e8 | vector RNG |
| MC-A3: stratified S=64 | 1/3 | 1,048,576 | 0.333335 | 1.8e-6 | 1.8 ms | 5.8e8 | 188x lower error at equal wall |

The stratified result is the retained variance-reduction mechanism: approximately 188x lower
absolute error at the compared wall time, not "188x faster Monte Carlo". For this class of system
the useful metric is statistical work per second (error at fixed wall, ESS, trajectories/s), not
samples/s alone. See `docs/MONTE_CARLO.md` and `figures/variance_reduction.svg`.

## 8. Stochastic rendering

Rendering here means visualizing statistical structure generated by a conditioned stochastic process,
not conventional raster graphics. The synthetic field's terminal state pairs (X_T, X_{T-1}) are binned
on the GPU into a 512x512 density heatmap — samples, evolution, observables, and reduction all GPU-side.
The retained artifact: `figures/kolmo_field_density.ppm` (and a lossless PNG conversion).

Caption: M = 1,048,576 branches; T = 64 steps; OU dynamics a=0.50, σ=1.00 from parent x0=1.00;
Philox4x32-10 RNG (seed 42); per-step kill-coin pruning p=0.01 with weight compensation 1/(1-p);
observable = terminal position pair; 512x512 bins over [-3,3]²; 44.7 ms advance wall; ~1.5e9
trajectory-steps/s; ESS = 550,847. See `docs/STOCHASTIC_ENGINE.md` section "Rendering".

## 9. The AMD tier

The gfx906 tier is a first-class compute participant. It owns the prefix, the PLE, and the fused head
of every production token, and its remaining milliseconds are the largest measured lever in the decode path. Its value comes from the decomposition, not from isolated device benchmarks:
a device that loses conventional single-device comparisons can still be materially useful when the
architecture assigns it work that matches its physical characteristics (HBM bandwidth/capacity for
resident layers, custom HIP kernels, state ownership). The long-context tax measured on the AMD side
(≈2.4-2.8 ms/token at 8K) is the single largest measured lever. Two hypotheses about that tax were
tested and rejected by controlled comparison (HIP graphs: earlier freeze; selector bisection: this
window). The remaining measured location is the QSA scores + flash attention growth. See
`docs/AMD_HETEROGENEOUS.md` and `figures/latency_breakdown.svg`.

## 10. Correctness methodology

Bit identity is not universally required; each mechanism has its own gate. Used in this work:
token identity where semantics must be exact; analytic expectation for Monte Carlo kernels;
known-answer tests for RNG; CPU-reference reductions; deterministic seeded replay; branch
independence; state-signature comparison; causal-equivalence continuations; compute-sanitizer
memcheck; hash provenance. Changes proceed one mechanism at a time: strict build, smallest physical
gate, end-to-end receipt, then KEEP / ROLLBACK / PROMOTE. See `docs/CORRECTNESS.md` and
`docs/BUILD_LAW.md`.

## 11. Reproducing the results

```
./scripts/verify_freeze.sh                 # checks public artifacts against SHA256SUMS
./scripts/reproduce_spark_tests.sh         # RNG + Dumitrescu unit tests (CUDA)
./scripts/reproduce_mc.sh                  # MC-A / MC-A2 / MC-A3 / MC-B gates
./scripts/reproduce_stochastic_field.sh    # synthetic Kolmogorov field + rendering artifact
```

Requirements: CUDA ≥ 12 with nvcc; a GPU ≥ sm_70 for the standalone kernels (the retained builds
target sm_121a; the scripts honor `CUDA_HOME` and `QF_ARCH`). Model-backed experiments (MC-C,
decode receipts) require the two-box appliance and are documented in `docs/REPRODUCIBILITY.md` with
exact recipes; they cannot run from this repository alone.

## 12. Known limitations

- The exact AMD source tree that produced the frozen production binary (824e5c50…) has not been
  recovered. Rebuilds from the available mirror diverge at token 0 with and without the radix port;
  the port itself is unit-verified and retained as ready-to-apply, but it cannot receive a production
  integration receipt until the true source tree is recovered. See `freeze/PROVENANCE.md` and
  `docs/LIMITATIONS.md`.
- M>8 stochastic width is not validated: at M=16 the second slot's state signatures are out of scale
  and slot 0 drifts relative to the M=8 run. M=8 remains the validated width.
- The region wire carries token ids, not per-row log-probabilities; weighted model-backed Monte Carlo
  requires a wire extension.
- The model-backed sampler scans the full vocabulary (≈110 ms/step at M=8); a restricted candidate-set
  sampler is the next mechanism.
- Qwen 3.ocho's artifact filter is a host-side id-threshold discriminator with false negatives below
  id 110000; see `docs/QWEN_3_OCHO.md`.

## 13. Repository map

`docs/` — architecture, engineering paper, stochastic engine, Monte Carlo, AMD tier, prefill/refill,
correctness, reproducibility, limitations, build law, Qwen 3.ocho. `figures/` — engineering diagrams and the
retained density artifact. `stoch/` — the stochastic engine sources and unit tests. `src/` —
patch-derived production sources (provenance labels in `freeze/PROVENANCE.md`). `patches/` —
lane A and lane B diffs vs the production base, and the AMD pending work. `receipts/` — sanitized
copies of the canonical receipts (private originals are in the canonical freeze package).
`freeze/` — manifest, provenance, package hashes. `scripts/` — verification and reproduction.
`ocho/` — Qwen 3.ocho capability sources: the Spark engine tree (CUDA) and the AMD region server
tree (HIP/ROCm), sanitized, plus the loop client patch.

## 14. Citation and license

See `CITATION.cff`. License text in `LICENSE`. Published: https://github.com/bespokeontology/qwen-3-ocho
