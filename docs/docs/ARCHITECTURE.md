# Architecture

## Decomposition

The model is split across two accelerators. In the production configuration the AMD tier owns
layers 0..15 (prefix), the PLE constants layer, and the lm_head; the NVIDIA tier owns layers 16..47
(tail) and the QSA indexer for the tail layers. Each decode step is:

1. AMD prefix: the resident prefix state advances for the new token; residual is sent over the
   network.
2. Spark tail: the tail layers run (dense FP8 projections, routed NVFP4 experts, hyper-connections,
   GDN recurrence, QSA sparse attention where applicable); residual is returned.
3. AMD fused head: output HC, lm_head, argmax; the next prefix window is prepared.

The AMD tier therefore appears twice per token on the critical path. The split points are
configurable (QF_LAYER_BEGIN, QF_AMD_PREFIX). See figures/heterogeneous_pipeline.svg.

## Resident state

Both tiers keep the model resident: weights, KV caches, GDN recurrent state, convolution rings,
QSA indexer pools, and expert slot maps. A request is not a cold load. This is what makes refill
(ingest without replay), continuation, and forking meaningful; see docs/PREFILL_REFILL.md.

## The stochastic overlay

The same resident state supports the stochastic lane: one primed parent state is copied into M row
slots on both tiers (fork), then advanced as a field. Rows share the parent's immutable history and
diverge only through their branch-local mutable state and their independent RNG streams. See
docs/STOCHASTIC_ENGINE.md.

## Precision

Dense weights are FP8 (E4M3, per-64-block scales); routed experts are NVFP4 in the production
Spark path and INT8 on the AMD production tier (both arenas in the fork topology); PLE constants
and lm_head are higher precision. Precision changes were closed by earlier freezes and were not
part of this window's work.
