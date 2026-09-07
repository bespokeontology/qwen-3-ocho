# Lane A — stochastic-field throughput law (2026-09-07 night, stoch-engine-0907)

Synthetic Kolmogorov field (OU advance + kill-coin + cub compaction + fused observables), T=64 steps,
width sweep on the GB10, seed 42, pkill=0.01:

| M (branches) | trajectories/s | n_alive_end | ESS |
|---|---|---|---|
| 1,024 | 2.6e7 | 559 (54.6%) | 559 |
| 16,384 | 3.8e8 | 8,760 (53.5%) | 8,760 |
| 262,144 | 1.52e9 | 137,861 (52.6%) | 137,861 |
| 1,048,576 | 1.60e9 | 550,847 (52.5%) | 550,847 |

Launch-bound below ~16K branches; saturating at **~1.6e9 trajectories/s** above ~262K branches
(field-major kernels, per-step cub compaction included). Survival matches the kill-coin model
(0.99^64 = 52.6%); ESS = n_alive (equal weights). Fork/init wall is 0.08 ms at M=1M — state
creation is negligible next to advance work; at this scale a shared-prefix/COW representation
is NOT yet needed (goal 11: the scaling law says the crossing point is far above 1M branches
at 16 B/branch).

Binary: stoch/build/kolmo_field (sha in the field receipt). Logs: none (stdout above).
