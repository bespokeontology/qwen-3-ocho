# Monte Carlo integration

The engine's numerical core was validated against integrals with closed-form answers. All gates
fuse sampling, integrand evaluation, and reduction in one kernel; the CPU never iterates over
samples. RNG: Philox4x32-10 per the frozen specification. Reduction: the Dumitrescu primitive
(fp64). Hardware: NVIDIA GB10, locked clocks. Seed 42 throughout. Receipt (authority):
receipts/MC_GATES_RECEIPT_20260907.md.

## MC-A — scalar sanity

Problem: E[X^2] for X ~ U(0,1). Analytic: 1/3. Integrand: f(x) = x^2. Unweighted.
N = 1,048,576. Estimate 0.333674980. |err| = 3.4e-4 (expected 1-sigma ~2.9e-4). Wall 1.9 ms;
~5.4e8 samples/s.

## MC-A2 — weighted (importance sampling)

Same integral, importance-sampled from q(x) = 2x on [0,1]: x = sqrt(u), w = 1/(2x).
Estimate 0.333907284, |err| = 5.7e-4, ESS ~3.1e5. Exercises the weighted path end to end
(log-weights, weighted moments, ESS reporting).

## MC-B — multidimensional

Problem: E[sum_j X_j^2] over [0,1)^16. Analytic: 16/3. 16 uniforms per sample (4 Philox words per
call). N = 262,144. Estimate 5.334712595, |err| = 1.4e-3. Wall 0.65 ms; ~4.1e8 samples/s.
Exercises vectorized RNG and the multi-observable reduction.

## MC-A3 — stratified (variance reduction)

Stratified/jittered sampling with S=64 strata: u = (stratum + uniform)/64. Same integral as MC-A.
N = 1,048,576. Estimate 0.333335152, |err| = 1.8e-6 — approximately 188x lower absolute error
than plain MC-A at the compared wall time (both ~1.9 ms). This is the retained variance-reduction
mechanism. It is not "188x faster Monte Carlo"; it is 188x lower error at equal wall. Receipt:
receipts/VARIANCE_REDUCTION_RECEIPT_20260907.md.

## Why this metric matters

For this class of system, samples/s alone is incomplete. The quantities that matter are samples/s,
effective samples/s, estimator error, estimator error at fixed wall time, ESS, and trajectories/s.
The optimization target is statistical work per second, not merely tokens or samples per second.
The stratified result is the concrete demonstration: identical wall, two orders of magnitude less
error, from changing only the sampling rule.

## Model-backed (MC-C)

The bridge from synthetic to resident-model integration is MC-C, documented in
docs/STOCHASTIC_ENGINE.md and receipts/MCC_MODEL_FIELD_RECEIPT_20260907.md: one resident parent,
seven temperature-1 stochastic children, one greedy control, sixteen steps, state-signature
observables, reduced to field statistics. No closed form exists for a model-conditioned
observable; the gate there is deterministic seeded replay plus the greedy control, and the M>8
anomaly is recorded as a limitation rather than a result.
