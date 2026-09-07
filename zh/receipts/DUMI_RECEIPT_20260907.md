# Lane A — Dumitrescu reduction receipt (2026-09-07 night, stoch-engine-0907)

**Mechanism ([ENG], no prior art in the record):** online-stable weighted statistics over a branch
field. One branch = log-weight lw + K observables f[K]; w=exp(lw); lw <= -1e300 = zero-weight sentinel
(skipped entirely: counted in neither N nor the sums). Stats: N, sum w, sum w^2, sum w*f_k, sum w*f_k^2,
max log-w, log-sum-exp(log w), ESS=(sum w)^2/sum w^2. fp64 accumulation; per-block partials;
deterministic host merge (no cross-block atomics). Multi-observable in one traversal (K template).

**Correctness gate:** vs CPU fp64 reference — N=1, N=3, N=17 odd, 1024, 1000003; weights uniform /
1..7 / 30% dead / tiny 1e-30 / huge 1e30 / i+1 / all-zero; signed observables; K=2 and K=4.
Tolerances: sums 1e-12 rel, weighted moments 1e-9 rel, ESS 1e-9 rel.

**Result: ALL PASS.**

```
uniform w=1                    K=1 M=1         n=1 sw=1.000000e+00 ess=1.000000e+00 total_w=1.000000e+00 -> PASS
uniform w=1                    K=1 M=3         n=3 sw=3.000000e+00 ess=3.000000e+00 total_w=3.000000e+00 -> PASS
uniform w=1                    K=1 M=1024      n=1024 sw=1.024000e+03 ess=1.024000e+03 total_w=1.024000e+03 -> PASS
uniform w=1                    K=1 M=1000003   n=1000003 sw=1.000003e+06 ess=1.000003e+06 total_w=1.000003e+06 -> PASS
w in 1..7                      K=1 M=4096      n=4096 sw=1.638100e+04 ess=3.276360e+03 total_w=1.638100e+04 -> PASS
30% zero-weight                K=1 M=4096      n=2866 sw=9.826000e+03 ess=2.456375e+03 total_w=9.826000e+03 -> PASS
tiny w=1e-30                   K=1 M=1024      n=1024 sw=1.024000e-27 ess=1.024000e+03 total_w=1.024000e-27 -> PASS
huge w=1e30                    K=1 M=1024      n=1024 sw=1.024000e+33 ess=1.024000e+03 total_w=1.024000e+33 -> PASS
w=i+1, signed f                K=1 M=1000003   n=1000003 sw=5.000035e+11 ess=7.500026e+05 total_w=5.000035e+11 -> PASS
all zero-weight                K=1 M=1024      n=0 sw=0.000000e+00 ess=0.000000e+00 total_w=0.000000e+00 -> PASS
N=17 odd                       K=1 M=17        n=17 sw=1.700000e+01 ess=1.700000e+01 total_w=1.700000e+01 -> PASS
K=2 obs, 30% dead              K=2 M=4096      n=2866 sw=9.826000e+03 ess=2.456375e+03 total_w=9.826000e+03 -> PASS
K=2 obs, w 1..7                K=2 M=65536     n=65536 sw=2.621390e+05 ess=5.242820e+04 total_w=2.621390e+05 -> PASS
K=4 obs, w 1..7                K=4 M=65536     n=65536 sw=2.621390e+05 ess=5.242820e+04 total_w=2.621390e+05 -> PASS
DUMI_TEST: ALL PASS
```

Files: stoch/reduce/dumi.cuh, stoch/reduce/dumi_test.cu. Binary: stoch/build/dumi_test.

```
8c2ab5e3b3681685faeb560a3b582d64265994a95eb17cc43e9560cdd9e30627  rng_test
3c0b84827dd50f80f468191e0496780a3f78ef650bd3326cfa8b467e3542072d  dumi_test
bfdc7145ae7d88a18ee6ab0b5603d23c56ec2b5afc4ed70a1865e30e638beb9b  mc_gate
d224936f121803b45ffaba12a7f2405b6d08cc1404e29234c758b4e1db0fe0b8  kolmo_field
```
