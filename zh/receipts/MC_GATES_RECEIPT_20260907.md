# Lane A — Monte Carlo gates receipt (2026-09-07 night, stoch-engine-0907)

**Mechanism:** fused GPU sample + GPU integrand + GPU reduction (no CPU in the loop; no intermediate
branch vectors materialized). MC-A: E[X^2], X~U(0,1) = 1/3 (GEN1 §7 extractor, unweighted).
MC-A2: same integral importance-sampled from q(x)=2x on [0,1], x=sqrt(u), w=1/(2x) (weighted path;
midpoint cells u=(word>>8 + 0.5)·2^-24 in [2^-25, 1-2^-25]; expected MC error dominates the 2^-48
discretization bias). MC-B: d=16, E[sum_j X_j^2] = 16/3 (16 uniforms/sample, 4 Philox words each).
Analytic answers known exactly. Weighted moments validated by the dumi CPU-reference receipt.

```
GATE MC-A  E[X^2] U(0,1)=1/3 (unweighted)
  device=NVIDIA GB10 N=1048576 seed=42
  estimate=0.333674980 analytic=0.333333333 |err|=3.416e-04 rel=1.025e-03
  wall=1.884 ms samples/s=556701394 ess=1048576.000 sum_w=1.048576e+06 n=1048576
GATE MC-A2 E[X^2]=1/3 via q(x)=2x (weighted)
  device=NVIDIA GB10 N=1048576 seed=42
  estimate=0.333907284 analytic=0.333333333 |err|=5.740e-04 rel=1.722e-03
  wall=2.085 ms samples/s=502869747 ess=309829.812 sum_w=1.047004e+06 n=1048576
GATE MC-B  E[sum X_j^2], d=16 = 16/3 (unweighted)
  device=NVIDIA GB10 N=262144 seed=42
  estimate=5.334712595 analytic=5.333333333 |err|=1.379e-03 rel=2.586e-04
  wall=0.646 ms samples/s=405584699 ess=262144.000 sum_w=2.621440e+05 n=262144
```

Files: stoch/gates/mc_gate.cu. Binary: stoch/build/mc_gate.

```
85928328f6e73622cdb8c01a7975b59eea4a7f1618356ccce4e4770fd6dd265f  rng_test
e3d34fd865da43cb960f9eb8ee524ee8a597de5e4b6ecf3d09b39a81738061ff  dumi_test
ef6ac25e98fb221c04aa7ce5cd6726e1ef61c4f6a4caa782a5d4ea297de7a704  mc_gate
```
