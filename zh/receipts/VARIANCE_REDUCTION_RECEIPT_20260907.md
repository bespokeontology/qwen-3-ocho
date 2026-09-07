# Lane A — variance reduction receipt (2026-09-07 night, stoch-engine-0907)

**Mechanism:** stratified (jittered) sampling, S=64 equi-probable strata: u = (stratum + uniform)/64.
Unbiased; each stratum gets N/64 samples; the within-stratum variance is O(1/S²) of the raw one.
Same fused GPU sample+integrand+reduction path as MC-A (mode 3 of stoch/gates/mc_gate.cu).

**Gate:** error at identical N (and effectively identical wall, both ~2 ms on GB10) vs plain MC-A.

```
GATE MC-A  E[X^2] U(0,1)=1/3 (unweighted)
  device=NVIDIA GB10 N=1048576 seed=42
  estimate=0.333674980 analytic=0.333333333 |err|=3.416e-04 rel=1.025e-03
  wall=1.937 ms samples/s=541244108 ess=1048576.000 sum_w=1.048576e+06 n=1048576
GATE MC-A3 E[X^2]=1/3 stratified S=64 jittered (unweighted)
  device=NVIDIA GB10 N=1048576 seed=42
  estimate=0.333335152 analytic=0.333333333 |err|=1.818e-06 rel=5.455e-06
  wall=1.817 ms samples/s=577145290 ess=1048576.000 sum_w=1.048576e+06 n=1048576
```

**Judgement: error-per-fixed-wall — keep as the retained variance-reduction mechanism** (goal 8 asks for
exactly this measurable win). Antithetic/common-random-numbers deferred: stratification is the cheap,
analytic, deterministic win.
