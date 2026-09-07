# Branch descriptor / stochastic state model (goal 1)

Synthesis of FORK_MACHINERY_INVENTORY_420bca8.md (Spark M-row) + KOLMOGOROV_FORK_RECEIPT_20260906.md
(AMD fork copy) + the KD branch descriptors (KdFieldHdr/KdFieldRow + PAIR-1 C2 trailer).

## Causal state components of one trajectory
| component | where | class |
|---|---|---|
| KV (kc/vc) | Spark g_kc_M/g_vc_M (qf.cu:971); AMD Qf4Stage.kc/vc (:472) | per-branch mutable append |
| GDN recurrent state (gdnS) | Spark g_gdnS_M; AMD Qf4Stage.gdnS (:471) | must-copy (small) |
| convolution rings | Spark g_convring_M; AMD Qf4Stage.convring | must-copy (small) |
| QSA indexer pools | AMD Qf8QsaIdx (:458) | per-branch mutable append |
| PLE hist/ring | AMD Qf8Ple (:189) | must-copy (small) |
| position / frontier | QfDecodeParams.pos | cheap metadata |
| current token | QfDecodeParams.token | cheap metadata |
| RNG state | NONE persisted: counter-based (seed, branch, step, lane) | cheap metadata (counters) |
| branch weight / log-prob | NEW: lw field (dumi sentinel <= -1e300 = dead) | cheap metadata |
| observable accumulator | NEW: acc[K] | cheap metadata (K floats) |
| termination | NEW: alive flag / compaction slot | cheap metadata |

## Representation law
- parent_state + branch_descriptor[M] rather than M independent copies: the fork copies the small mutable
  state (gdnS, convring, PLE) and SHARES the append-only history by construction (children append rows,
  parent rows are never rewritten — the fork proof's row-7 independence receipt demonstrates no aliasing).
- COW is unnecessary at current scale: 16 B/branch state, M=1M = 16 MB (throughput receipt).
- Branch identity = (seed, branch slot, step): RNG streams need no stored state (Philox counters).
- Dead branches: lw <= -1e300 sentinel; compaction = cub exclusive scan (kolmo_field proves the pattern;
  the KD GEN1 §5 commit/compaction semantics are the model-side counterpart).
- The KD trailer fields (TOPO_HASH, per-row cand_id/rng_key/logp_proposal/logp_conditional/cost) are the
  wire form when a branch field crosses boxes; logp_conditional feeds lw for weighted MC (goal 7).

## What remains CPU-owned (current fork driver)
Step orchestration, token ids readback, sigR/sigK signature readback (main.cu:394/399). The GPU-side
field loop (persistent kernel) is the exceptional-success target, not required for the MC gates.
