# Build law

The project's working discipline, applied to every mechanism in this tree.

## One mechanism per round

Freeze the predecessor. Change one physical mechanism. Strict build. Smallest correctness gate.
End-to-end receipt. Then dispose:

- materially worse -> ROLLBACK;
- equivalent but structurally superior -> KEEP/PROMOTE (no speedup claim);
- materially better -> KEEP with a performance receipt.

Ties no longer destroy structurally better work; equivalences are decided on structure and
non-regression, and speedups are only claimed from measurements that establish them.

## Receipt rules

- Search existing receipts before measuring anything.
- One measured discriminator on the exact frozen recipe before building.
- Numeric gates are set before the experiment, not after.
- Never subtract estimates from estimates.
- Receipts paste the full prompt and answer where outputs are involved.

## Frozen lanes

Prefill (1,543 tok/s), refill, and the fork proof are frozen; they are changed only for a
correctness regression. M=8 is the validated stochastic width; M>8 has its own correctness round.
The AMD production binary is frozen pending source recovery.

## Environment discipline

- No Python at runtime (offline tooling only). Serving paths are CUDA/HIP C++.
- Clock locks are re-applied after any profiler run before trusting a receipt.
- Kill by PID; never pattern-kill server processes (documented failure mode).
- Freeze archives must carry full source trees, not binaries alone (the AMD production source
  was lost this way).

## Output authority

The operator is the sole authority on output quality. Agents measure, gate, and report; they do
not judge coherence or quality on their own.
