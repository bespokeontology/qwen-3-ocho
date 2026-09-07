# Lane A — Gate MC-C: model-conditioned stochastic field reduced to an observable (2026-09-07 night)

**Mechanism:** one resident conditioned parent (kolmo87, 382 tokens, 8/40 split, frozen fork binary
5179fff3) forked into M=8 rows; row 0 = greedy control, rows 1-7 = temperature-1 Gumbel-max sampling
(k_sample_rows, deterministic splitmix64(seed ^ (r<<40) ^ i), QF_HEAD_TEMP=1.0 QF_HEAD_SEED=7);
16-step M-row field decode (region path, QF_EXPERT_MODE=full); per-row state signatures (sigR =
Σ GDN+conv over Spark layers, sigK = Σ K/V of the last 4 positions) read back via the new additive
QF_M8_FORK_SIG=1 switch (stoch-engine-0907, main.cu); reduction = unweighted mean/variance/ESS over
the 7 sampled branches.

**Run:** fork AMD 2.8 ms + Spark 7.1 ms; 8 × 16 steps in 5.88 s = 367.5 ms/step, 21.8 sibling-tok/s
(sampling costs ~110 ms/step vs 253.6 ms greedy — naive full-vocab Gumbel loop, the first
optimization target). Trajectories: row 4 branches at step 1, rows 2/6/7 at step 4, row 5 coincides
with greedy for all 16 steps (a legitimate sample).

**Observable:** sigR over rows 1-7: 1565.6, 1201.3, -281.6, -1232.8, -698.3, 794.3, -751.9 →
field mean +228.1, sample std 1076, ESS 7. Greedy row sigR -704.4 (field mean is +932 above the
greedy trajectory ≈ 2.3 SE of the field mean). Terminal-token distribution over the 8-row field:
318 ×3, 264, 19, 8260, 53235, 156566. Row 5's sigR (-698.3) tracks its greedy-identical tokens —
the signature follows the trajectory.

**Gate:** single-seed field + greedy control + the documented M-row arithmetic band (the fork receipt's
non-bit-exactness). Determinism of the whole field verified (repeat run: tokens bitwise identical).
Two-seed coherence requires a server reboot per seed (QF_HEAD_SEED is read at boot) — recorded, not run.

**Findings carried forward:** (1) M=16 second-slot signatures are on a different scale and slot 0 drifts
vs the M=8 run — the M>8 state path needs its own correctness round; M=8 stays the validated width.
(2) The fork recipe REQUIRES QF_EXPERT_MODE=full: the default expert mode hits a pre-existing OOB in
k_pfm_gateup2 (qf_prefill_mma.cu:353, T=256 chunks) — frozen-binary hazard, not fixed. (3) No per-row
log-prob on the region wire — weighted MC needs a logits/prob extension (the next mechanism).

Client: src/qwenflash_m8.mcc (420bca8 + QF_M8_FORK_SIG). Logs /tmp/mcc4_m8_k16_temp.log,
/tmp/mcc2_m{8,16}_k16.log, /tmp/mcc3_m8_k16_rep.log.
