# stoch — GPU-native stochastic evaluation engine (experimental, Lane A)

Branch stoch-engine-0907 (off 420bca8). Lane-separated from the m1-30tok-0907
production lane: separate branch, separate receipts, separate commits. Frozen
production lanes (prefill/refill/fork/M=8) are untouched by anything in here.

- include/philox.cuh — Philox4x32-10 counter-based RNG implementing the FROZEN
  KD GEN1 §7 contract: key=(slot,round) -> (seed_lo,seed_hi); counter=
  (position,branch,purpose,index) -> (step,branch,purpose,lane); purposes
  DRAFT=1/ACCEPT=2/RESIDUAL=3/BONUS=4; uniform=(word0>>8)*2^-24; KAT included.
- rng/rng_test.cu — KAT, determinism, stream independence, counter snapshot/
  restore, distribution sanity.
- reduce/dumi.cuh — Dumitrescu reduction ([ENG], no prior art in the record):
  N, sum w, sum w^2, sum w*f_k, sum w*f_k^2, max log-w, log-sum-exp(log w),
  ESS=(sum w)^2/sum w^2. fp64 accumulation, per-block partials, deterministic
  host merge. Zero-weight sentinel: lw < -1e300.
- reduce/dumi_test.cu — vs CPU fp64 reference (odd sizes, tiny/huge weights,
  zero weights, multiple observables).
- gates/mc_gate.cu — MC-A E[X^2]=1/3; MC-A2 weighted (q(x)=2x) E[X^2]=1/3;
  MC-B d=16 E[sum X_j^2]=16/3. Samples + integrand + reduction all on GPU.

Build: ./build.sh   (binaries in stoch/build/). No Python at runtime; the only
tooling here is C++/CUDA. Receipts: Kolmogorov Build Law/receipts_stoch_0907/.
