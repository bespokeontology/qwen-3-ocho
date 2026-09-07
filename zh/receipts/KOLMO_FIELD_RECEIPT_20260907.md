# Lane A — synthetic Kolmogorov field receipt (2026-09-07 night, stoch-engine-0907)

**Mechanism:** conditioned parent x0=1.0 forked into M GPU-native stochastic children advanced as
ONE field (OU dynamics x<-a·x+σ·ξ, ξ via Box-Muller from GEN1 §7 Philox uniforms keyed
(seed, step, branch, purpose=DRAFT, lane)); per-step kill-coin pruning p_kill with weight
compensation w *= 1/(1-p) => unbiased (Russian roulette); cub exclusive-scan compaction of dead
branches (dynamic branch survival); fused per-step observable accumulation; Dumitrescu reduction
at the end; 2D (x_T, x_{T-1}) density heatmap rendered on GPU as the stochastic-rendering artifact.
CPU never sees a branch value — only merged stats.

**Analytic gates:** E[X_T] = a^T x0; E[X_T²] = (a^T x0)² + σ²(1-a^{2T})/(1-a²);
E[(1/T)Σ x_t²] = (1/T)Σ_t [a^{2t} x0² + σ²(1-a^{2t})/(1-a²)]. All three unbiased under the
kill weights (weighted mean = swf/sw).

```
KOLMO FIELD device=NVIDIA GB10 M=1048576 T=64 a=0.50 sig=1.00 x0=1.00 pkill=0.01 seed=42
  fork/init wall=0.079 ms | advance+compact wall=44.720 ms (64 steps) | reduce wall=2.246 ms
  trajectories/s=1500629269  n_alive_end=550847 (52.5%)  ESS=550847.000  sum_w=1.048042e+06
  est E[X_T]    = 3.063804e-03  analytic = 5.421011e-20  |err| = 3.064e-03
  est E[X_T^2]  = 1.337359  analytic = 1.333333  |err| = 4.025e-03
  est E[avg x^2]= 1.331219  analytic = 1.331597  |err| = 3.785e-04
  render: kolmo_field_density.ppm (512x512, max density=282)
```

Files: stoch/field/kolmo_field.cu. Binary: stoch/build/kolmo_field.

```
8c2ab5e3b3681685faeb560a3b582d64265994a95eb17cc43e9560cdd9e30627  rng_test
3c0b84827dd50f80f468191e0496780a3f78ef650bd3326cfa8b467e3542072d  dumi_test
bfdc7145ae7d88a18ee6ab0b5603d23c56ec2b5afc4ed70a1865e30e638beb9b  mc_gate
d224936f121803b45ffaba12a7f2405b6d08cc1404e29234c758b4e1db0fe0b8  kolmo_field
```
