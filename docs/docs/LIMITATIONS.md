# Limitations

## AMD source provenance (blocking)

The exact source tree that produced the frozen production AMD binary (824e5c50…) has not been
recovered. The available mirror is only partially hash-verified against the freeze record.
Controlled rebuilds diverge at token 0 from the frozen binary both with and without the radix
port, so the port is exonerated and the rebuild itself lacks provenance. AMD production work
stays frozen until source archaeology recovers and hash-matches the true tree. The radix port is
retained as experimental, unit-verified, pending integration (patches/amd_pending/).

## Native 30 tok/s not reached

Authoritative overnight baseline: 29.70 tok/s at the 8,038-token production recipe. Retained
state after B1+B2: 29.44 tok/s (net +0.1 ms/token, inside variance). The remaining distance is
~0.34 ms/token. The next mechanism (AMD QSA layer scores + attention to the byte floor) is gated
on the provenance recovery above.

## M>8 stochastic width unvalidated

At M=16 the second slot's state signatures are out of scale and slot 0 drifts relative to the
M=8 run. M=8 is the validated width; M=16 is not published as valid stochastic scaling and is
its own correctness round.

## Model-backed Monte Carlo gaps

- The region wire carries token ids only; per-row log-probabilities are not available, so
  weighted model-backed Monte Carlo requires a wire extension.
- The model-backed sampler scans the full vocabulary (~110 ms/step at M=8); a restricted
  candidate-set sampler is the next mechanism.
- MC-C is single-seed: the sampling seed is read at server boot, so two-seed coherence requires
  server restarts.
- sigR/sigK are deterministic functionals of branch state used as trajectory observables; no
  broader statistical interpretation is claimed.

## Frozen-binary hazard (recorded, not fixed)

The fork recipe requires QF_EXPERT_MODE=full. With the default expert mode the frozen binary's
prefill path performs an out-of-bounds read (compute-sanitizer evidence in the B3 and MC-C
receipts) on T=256 chunks. The recipe as actually used historically is unaffected; the hazard is
documented because the fork receipt elided the expert-mode variable.

## Scope exclusions

Speculative decoding/MTP, quantization changes, HIP graphs (rejected by earlier measurement), and
any M=8 performance rework are out of scope. This tree makes no comparison claims against other
serving systems; external numbers would require labeling along many axes (native vs speculative,
precision, context, hardware, concurrency) and are not included.
