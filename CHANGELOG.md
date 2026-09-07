# Changelog

## 0.1.0-rc1 — 2026-09-07

Release candidate derived from the canonical overnight freeze package
FREEZE_QWEN_OVERNIGHT_20260907 (internal SHA256SUMS; see freeze/package_SHA256SUMS).

Added (stochastic lane, branch stoch-engine-0907 off 420bca8):

- Philox4x32-10 counter-based GPU RNG implementing the project's frozen RNG specification,
  known-answer tested on host and device.
- Weighted statistical reduction primitive (N, sum w, sum w^2, sum w f, sum w f^2, log-sum-exp,
  ESS; fp64; multi-observable) with CPU-reference validation.
- Fused Monte Carlo gates MC-A, MC-A2 (weighted), MC-B, and stratified MC-A3 (188x lower error at
  equal wall vs MC-A).
- Synthetic conditioned Kolmogorov field: fork, kill-coin pruning with weight compensation,
  exclusive-scan compaction, fused observables, three analytic gates, density rendering artifact.
- Model-backed conditioned stochastic field (MC-C): M=8 fork of a resident parent, greedy control
  row, temperature-1 sampled rows, 16 steps, per-row state-signature observables (additive switch
  QF_M8_FORK_SIG), unweighted field statistics.
- Throughput characterization of the synthetic field (1.6e9 trajectory-steps/s at M>=262144).

Changed (production lane, branch m1-30tok-0907 off 420bca8):

- B1: QSA indexer scoring universe sized by allocation maximum at T=1 (graph capture froze the
  host-derived candidate count; blocks created during generation were never scored). Gate: fixed
  graph vs eager decode, 300/300 token-identical.
- B2: exact single-pass radix top-512 selection replacing the 48-step threshold bisection, with
  identical tie semantics. Gate: 13-case unit harness + memcheck + 300/300 token-identical receipt;
  tail 23.5 -> 23.1 ms/token.

Pending (AMD, not integrated):

- B3 radix port of the same selector into the AMD top-k kernels: HIP unit-verified (10/10
  non-dense cases exact). Integration deferred: the exact source tree of the frozen production
  AMD binary (824e5c50) has not been recovered, and rebuilds from the available mirror diverge at
  token 0 with and without the radix.

Known limitations: see docs/LIMITATIONS.md.
