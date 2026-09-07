# Lane A — GPU RNG receipt (2026-09-07 night, stoch-engine-0907)

**Mechanism:** Philox4x32-10 counter-based RNG, ported VERBATIM from the frozen KD GEN1 §7
law (`sampled_mode/mcsd_reference.py` philox4x32_10): key=(slot,round) -> (seed_lo,seed_hi);
counter=(position,branch,purpose,index) -> (step,branch,purpose,lane); purposes DRAFT=1/ACCEPT=2/
RESIDUAL=3/BONUS=4; uniform=(word0>>8)*2^-24. Round: each of the 10 rounds multiplies c0*M0 and
c2*M1 (64-bit), Weyl key bump after every round except the last. Deterministic (seed,branch,step,lane)
mapping, no global state, snapshot = counters only.

**Correctness gate:** KAT (GEN1 §7 / Random123) ctr=key=0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8
(host AND device); determinism across launches; stream independence (all cells unique, distinct
branches disagree at every step); counter snapshot/restore mid-stream; 1M-uniform mean/var sanity.

**Result: ALL PASS.**

```
T1 KAT (ctr=key=0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8): host=PASS device=PASS
T2 determinism (2 launches, 512 cells): PASS
T3 independence (2048 cells unique=yes, branch0-vs-1 step disagreements=64/64): PASS
T4 counter snapshot/restore (steps 50..59): PASS
T5 distribution (N=1048576): mean=0.500249 var=0.083426 min=0.000001 max=0.999999 -> PASS
RNG_TEST: ALL PASS
```

Files: stoch/include/philox.cuh, stoch/rng/rng_test.cu. Binary: stoch/build/rng_test.

```
3eb272353a8b286cbcda22062ad07e1053d451863fd69e9369fedcced5430ff6  rng_test
f8ebb0b46e7d667961fb9196e9658103e0305da430da966c14d140126a89d16d  dumi_test
f4e4b49888249a15aa2bea958656a10874ce5cba40a14c7238952246f12c7ac5  mc_gate
```
