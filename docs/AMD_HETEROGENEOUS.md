# The role of gfx906 in the heterogeneous pipeline

## What the AMD tier actually owns

In the production configuration the four MI50 cards own model layers 0..15, the PLE layer, and the
lm_head. Every decode step crosses the AMD tier twice: the prefix advance before the Spark tail,
and the fused head (output HC + lm_head + argmax) plus the next-prefix window after it. The AMD
tier is compute on both ends of the critical path, with resident expert arenas (INT8 in
production; NVFP4+INT8 in the fork topology) and its own QSA indexer path. It is not storage and
it is not a device hanging off the side of the pipeline. See figures/heterogeneous_pipeline.svg.

## Measured contribution

At the 8,038-token production recipe the AMD fused head measures 10.7-10.8 ms/token against a
Spark tail of 23.0-23.5 ms/token. At short context the head is 8.2-8.6 ms; the difference
(2.4-2.8 ms/token) is the AMD long-context tax. Because the total step is 33.7 ms against a
33.33 ms budget for 30 tok/s, the AMD tax is currently the largest single measured lever —
approximately the entire remaining distance plus margin.

## Where the tax is not

Two hypotheses about the AMD long-context tax were tested by controlled comparison and rejected:

1. HIP graphs / launch overhead. An earlier freeze measured the exposed host residual at
   0.63 ms/token and closed that road (AMD_M1_LAUNCH_OVERHEAD_DISCRIMINATOR receipt).
2. The indexer's 48-step threshold bisection. This window ported the exact radix selector to the
   AMD top-k kernels, unit-verified it against the bisection reference (10/10 non-dense cases
   exact on gfx906), and rebuilt the server with and without the radix. Both rebuilds measure the
   same head time (10.8 ms/token). The selector is therefore not the measured source of the
   2048-boundary increase. The remaining measured location is the QSA scores kernel and the flash
   attention growth over the selected list (~0.8 ms per QSA layer at 8K vs ~0.46 ms short context).

## The provenance constraint

The exact source tree that produced the frozen production AMD binary (824e5c50) has not been
recovered. Rebuilds from the available mirror diverge at token 0 from the frozen binary both with
and without the radix port — the port is exonerated by its unit gate and by the no-radix control,
while the rebuild itself lacks production source provenance. AMD production work stays frozen
until the true source tree is recovered and hash-matched. The radix port is retained as
experimental, unit-verified, pending integration. See freeze/PROVENANCE.md and
docs/LIMITATIONS.md.

## Why the AMD side remained valuable

Accelerator value in this system comes from decomposition, not from winning isolated
single-device benchmarks. A device that loses conventional comparisons can still be materially
useful when the architecture assigns it work that matches its physical characteristics: the MI50
tier contributes HBM capacity and bandwidth for resident layers, runs custom HIP kernels matched
to gfx906, owns long-lived state, and removes both the prefix and the head from the other tier's
latency budget. Its measured weaknesses — the long-context QSA tax, the rejected HIP-graph idea,
the rejected selector idea — are documented precisely because they bound where the next
milliseconds can come from. The claim here is narrow: heterogeneous decomposition made hardware
that is obsolete in conventional single-device inference comparisons materially useful in a
modern serving and stochastic-compute system. No claim is made that the MI50 is faster than any
other device in isolation.
