# Qwen 3.ocho (C++) — Stochastic Trajectory Rendering with a Decision/Commit Loop on a Native CUDA/HIP/ROCm Inference Engine (NVIDIA GB10 + 4x AMD MI50)

Status: capability boundary 2026-09-07. This document records a working mechanism, not a
benchmark result. All numbers are measurements from the hardware available that night; the
node's thermal/power limits are stated where they matter.

Native throughput reference: the engine's M=1 decode path on this hardware — one DGX Spark
(GB10) tail plus a 4x AMD MI50 prefix — runs at 27–29 tok/s on the 8K production recipe with no
speculative decoding (measured across the receipt runs of the canonical overnight freeze).
Qwen 3.ocho's stochastic modes run alongside that path; their per-row rendering cost is stated
separately below and is not a substitute for the native path.

## 1. What it is

Qwen 3.Ocho is a mode of the native engine (4x AMD MI50 prefix + NVIDIA GB10 tail,
the architecture described in ARCHITECTURE.md) that renders several stochastic continuations of
one resident model state and either returns them as parallel futures or repeatedly evaluates and
commits one of them into the resident state.

Two first-class modes:

- **Ocho Beam**: one resident parent state forked into eight alternate futures, each rendered
  independently for a chosen number of steps. The futures are the output.
- **Ocho Loop**: render eight futures -> evaluate -> choose -> commit the chosen consequence into
  the resident state -> continue from that causal state -> render the next eight futures.

## 2. Mechanism

The resident parent is the M=1 state after prompt priming: KV [0,P) on both accelerators, the
recurrent (GDN) states, convolution rings, indexer pools, and the PLE history. A fork copies this
state into eight per-row slots on both boxes (AMD prefix layers and Spark tail layers). Row 0
remains greedy argmax; rows 1-7 sample from the head logits by per-row Gumbel-max noise,
deterministic for a given seed:

    sample_v = argmax_v ( logit_v / T - log(-log(u_v)) ),  u_v = splitmix64(seed ^ (r<<40) ^ v)

where v is the vocabulary index. The eight rows advance together through the M-row region path
(AMD prefix -> Spark tail -> AMD head) one step at a time; only token ids cross the wire.

The loop's evaluation stage is deliberately simple and fully logged: a row is penalized for
early end-of-sequence and for tail-vocabulary artifact tokens (measured mojibake cluster at
token id >= 110000); the remaining score is the distance of the row's recurrent-state signature
(sigR: summed GDN and convolution state over the Spark layers) from the field median. The row
with the best lexicographic score is committed: its full state (KV, recurrent state, indexer
pools, PLE history) is copied back to the parent slot on both accelerators, and the next
iteration re-forks from that state. The conditional-probability tree is preserved in the run
log: every branch's tokens, signatures, scores, and every choice.

Discarded-branch context inheritance (proof-of-concept): after each commit, losing branches are
decoded host-side and classified into constraint / failure / unresolved-question propositions
(generic restatements are rejected); ONE proposition per round, capped at 64 tokens, is injected
into the committed champion as a [OCHO BRANCH MEMORY] block through the proven resident refill
path. Losing rows' KV/state is not merged. Freeze: FREEZE_OCHO_BRANCHMEMORY_POC_20260907.

Field identity invariant: the same parent state, seed, and fork request produce the same
eight-way field regardless of what the server served previously. This required resetting the
head sampler's per-boot call counter at the fork/reset boundary (see section 4).

## 3. Measured behavior

| configuration | measurement |
|---|---|
| K=16 field, 8 rows, coding-task prompt | 5.16 s; 322.7 ms/step; 24.8 sibling-tok/s (3.10 per sibling) |
| K=1500 beam, C++ task | 851.7 s; 567.8 ms/step; 14.1 sibling-tok/s |
| K=1500 beam, Rust task | 845.6 s; 563.7 ms/step; 14.2 sibling-tok/s |
| Ocho Loop, K=48 x 16 iterations, C++ task | 768 committed tokens; ~24 s per render+commit cycle; clean exit |
| Ocho Loop, K=48 x 16 iterations, Rust task | 768 committed tokens; ~20-25 s per cycle; clean exit; chooser switched among six rows |
| Restricted-candidate sampler (server 7c677e80), K=16 field | 311.9 ms/step (was 324.6 full-vocab); zero non-ASCII mojibake in a K=128 beam |
| Same-boot field-identity gate (K=16 run twice) | bit-identical 8-row fields |

At K=128 the eight futures are readable engineering reasoning about the task (all converge on
mutex + two condition variables for the MPMC queue; one branch skips the reasoning block and
starts coding). At K=1500 every temperature-1.0 sampled row accumulates mojibake artifacts
(Cyrillic/Arabic/CJK fragments from the vocabulary tail) while the greedy row stays clean. The
loop's committed trajectory over 768 tokens is continuous design reasoning: ring buffer with
two condition variables, the close()-drain-versus-fail ambiguity revisited across iterations,
and the lost-wakeup predicate analysis. Three artifact tokens slipped the current filter in
768 committed tokens (~0.4%). The Rust-task trace (FREEZE_OCHO_RUST_LOOP_20260907) shows the
same structure: 16 decision/commit cycles, the chooser switching among six rows, and a continuous
DAG-scheduler design trajectory (ownership/borrowing, mutex+condvar worker loop, cycle detection
by DFS, AtomicBool cancellation).

## 4. Repairs made to reach this capability

- **Field identity (server side)**: the AMD head's Gumbel seed was offset by a per-boot call
  counter, so a reused server produced different fields for identical requests. Fixed by
  resetting the counter at REGION_FORK/REGION_RESET; validated by the same-boot gate above.
- **Loop accumulation (client side)**: the chunked render writes rows with stride equal to the
  chunk size; the chooser reads stride K. Fixed with a per-chunk temporary buffer copied into
  the K-stride trajectory. The chunk boundaries are observation points only; the chooser sees
  the same K tokens the one-shot call would produce.
- **Chooser deadlock**: when every row carried an artifact, the first chooser excluded all rows
  and committed none. The current chooser always selects: fewest artifacts first, early-EOS
  heavily penalized, consensus distance second.

## 5. Known limitations

- The region wire carries token ids, not logits: there is no per-row log-probability, so
  weighted Monte Carlo integration needs a logits extension.
- The artifact filter is a host-side id-threshold discriminator, not a tokenizer-aware
  classifier; it has false negatives below id 110000.
- Restricted-candidate sampling is now RETAINED (server 7c677e80, receipt
  SAMPLER_TOPK_RECEIPT_20260907): per-thread top-8 candidates, Gumbel-max over the 2048 candidate
  set, row 0's full-vocabulary argmax bit-identical, and the observed tail-vocabulary text
  corruption eliminated at the sampler level. The full scan stays available via
  QF_HEAD_FULLSCAN=1. The loop chooser's id-threshold filter remains as a backstop.
- M=8 is the validated fork width; the M=16 state path needs its own correctness round.
- QF_HEAD_TEMP and QF_HEAD_SEED are read once at server boot: one seed per boot.
- Measured on a Spark node at a ~220 W power ceiling with a known instability history
  (drivetrain-deficit failure mode). Timings should be read as node-specific.

## 6. Frozen artifacts

FREEZE_OCHO_20260907 (behavioral evidence: K=128 and K=1500 beams, parity gate, field-identity
leak reproduction), FREEZE_OCHO_CAPABILITY_20260907 (the working Loop: binaries, sources,
gate logs, the 16-iteration run, and the committed trajectory), FREEZE_OCHO_RUST_LOOP_20260907
(the second capability trace, Rust task), and FREEZE_OCHO_TOPKSAMPLER_20260907 (the retained
restricted-candidate sampler: binary, source, gates, K=128 beam comparison), and
FREEZE_OCHO_BRANCHMEMORY_POC_20260907 (the discarded-branch context inheritance proof-of-concept). All are
SHA256-verified and mirrored. The loop client lives on branch stoch-engine-0907 (commit
257a2a1); the commit-enabled AMD server is built from the archived pre-B3 fork tree plus the
QFW_REGION_COMMIT op, the field-identity reset, and the restricted-candidate sampler.
