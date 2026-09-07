# The stochastic compute engine

## Concept

An autoregressive model with resident causal state is an expensive conditioned transition
operator. The engine treats one resident conditioned state as the initial condition of a
GPU-resident stochastic field: M independent branches are forked from it, advanced stepwise,
pruned, compacted, observed, and reduced — without exporting every trajectory to the CPU. The
pipeline:

    resident conditioned state -> GPU fork -> per-branch counter RNG -> field advance
    -> survival/compaction -> observable evaluation -> reduction -> small result to host

"Kolmogorov" and "Dumitrescu" are project names for the two complementary primitives: Kolmogorov
advances the conditioned stochastic field; Dumitrescu evaluates and reduces it. Keeping generation
and reduction separable is deliberate even where kernels are fused.

## Branch state model

Per branch: KV append rows, GDN recurrent state, convolution rings, indexer pools, position,
current token, log-weight, observable accumulator, alive flag. Shared: the parent's immutable
history (children append; parent rows are never rewritten — demonstrated by the fork-independence
receipt). RNG state is not stored at all: streams are counter-based, addressed by
(seed, branch, step, lane). Branch descriptor size is 16 B; at 1M branches state creation is
0.08 ms, so copy-on-write is not required at current scale.

## Components

### RNG (stoch/include/philox.cuh)

Philox4x32-10, implemented from the project's frozen RNG specification: key=(slot,round),
counter=(position,branch,purpose,index); uniform=(word0>>8)*2^-24. Ten rounds, Weyl key bump after
every round except the last. Known-answer test: counter=key=0 -> 6627e8d5 e169c58d bc57ac4c
9b00dbd8 (host and device). Also tested: determinism, stream independence across 2,048 cells,
counter snapshot/restore, and 1M-uniform mean/variance. Receipt:
receipts/RNG_RECEIPT_20260907.md.

### Dumitrescu reduction (stoch/reduce/dumi.cuh)

Single-pass weighted statistics over a branch field: N, sum w, sum w^2, sum w f_k, sum w f_k^2,
max log-w, log-sum-exp(log w), ESS = (sum w)^2 / sum w^2, for K observables per branch. fp64
accumulation; each block writes a partial; the host merges partials in double (deterministic, no
cross-block atomics). Zero-weight sentinel lw <= -1e300. Validated against a CPU fp64 reference on
14 cases (odd sizes, tiny/huge/zero weights, multiple observables). Receipt:
receipts/DUMI_RECEIPT_20260907.md.

### Kolmogorov field advance (stoch/field/kolmo_field.cu)

The synthetic field: a conditioned parent x0 is forked into M branches of Ornstein-Uhlenbeck
dynamics x <- a*x + sigma*xi with Box-Muller noise from the counter RNG. Each step: advance, then
a kill coin (p=0.01) prunes the branch with weight compensation 1/(1-p) (unbiased Russian
roulette), then an exclusive-scan compacts survivors. Observables are accumulated per step and
reduced once at the end. Three closed-form gates: E[X_T] = a^T x0, E[X_T^2] = (a^T x0)^2 +
sigma^2 (1-a^{2T})/(1-a^2), E[(1/T) sum x_t^2]. All three matched within expected Monte Carlo
error; survival 52.5% matched the kill-coin prediction 0.99^64 = 52.6%. Receipt:
receipts/KOLMO_FIELD_RECEIPT_20260907.md.

### Model-backed field (MC-C)

The same machinery applied to the resident model: a primed 382-token parent (8/40 split) is forked
into M=8 rows. Row 0 is the greedy control; rows 1-7 sample temperature-1 Gumbel-max transitions
(server-side k_sample_rows, deterministic splitmix64 keyed by (seed,row,v)). After 16 steps, each
row's state signature (sigR = sum of GDN+convolution state over Spark layers; sigK = sum of K/V of
the last 4 written positions) is read back via the additive QF_M8_FORK_SIG switch and reduced
unweighted over the 7 sampled rows: field mean sigR +228 vs greedy-row sigR -704, std 1076, ESS 7;
terminal-token distribution 318 x3, 264, 19, 8260, 53235, 156566. The field executes at
21.8 sibling-tokens/s (367.5 ms/step); the full-vocab Gumbel loop costs ~110 ms/step and is the
first optimization target. A repeat run reproduced the token trajectories bitwise (signature
values vary within the documented M-row arithmetic band). Receipt:
receipts/MCC_MODEL_FIELD_RECEIPT_20260907.md.

sigR's interpretation is limited: it is a deterministic functional of branch recurrent state used
as a trajectory observable; no broader statistical interpretation is claimed. M>8 is not validated
(see docs/LIMITATIONS.md).

## Throughput

Synthetic field, T=64 steps, GB10, seed 42: 2.6e7 trajectory-steps/s at M=1,024; 3.8e8 at
16,384; 1.52e9 at 262,144; 1.60e9 at 1,048,576. Launch-bound below ~16K branches; saturated above
~262K. Units: trajectory-steps/s (M branches advanced one step each); this is not completed
trajectories/s and not token throughput. Receipt: receipts/THROUGHPUT_LAW_RECEIPT_20260907.md.

## Rendering

Rendering means visualizing statistical structure of a conditioned stochastic process. The
terminal field state pairs (X_T, X_{T-1}) are binned on the GPU into a 512x512 density grid
(-3..3 per axis) and written as a PPM. Parameters: M=1,048,576; T=64; a=0.50, sigma=1.00; x0=1.00;
Philox seed 42; p_kill=0.01 with weight compensation; 44.7 ms advance wall; ~1.5e9
trajectory-steps/s; ESS 550,847. Artifact: figures/kolmo_field_density.ppm (+ lossless PNG). This
demonstrates the general chain — stochastic samples -> field evolution -> observable ->
reduction/render — not a graphics pipeline.
