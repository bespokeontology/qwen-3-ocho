# Correctness model

Bit identity is not a universal requirement in this project. Each mechanism has its own gate,
chosen to be the smallest physical check that would catch a wrong implementation. The gates used
in the retained work:

| gate | mechanisms | what it establishes |
|---|---|---|
| known-answer test | RNG | the generator matches the frozen specification (KAT on host and device) |
| CPU reference | Dumitrescu reduction | merged statistics match an fp64 reference (14 cases, tolerances stated) |
| analytic expectation | MC-A / MC-A2 / MC-B / synthetic field | estimator matches a closed-form answer within expected Monte Carlo error |
| token identity | B1, B2 | exact semantic equivalence where the rule demands it (300/300) |
| eager-vs-graph discriminator | B1 | the captured graph's frozen universe equals the eager universe |
| unit harness vs original algorithm | B2, B3 | replacement reproduces the bisection exactly (13 and 10 cases) |
| compute-sanitizer memcheck | B2 development, fork diagnostics | no out-of-bounds accesses |
| deterministic seeded replay | MC-C, RNG | identical inputs reproduce identical trajectories |
| branch independence | fork proof (earlier freeze) | forced-token perturbation moves only the perturbed row |
| state signatures | MC-C | trajectory observables track the branch (sigR follows tokens) |
| causal-equivalence continuation | refill (earlier freeze) | resumed state produces a coherent continuation |
| hash provenance | all retained binaries | every artifact is addressed by SHA256 and a freeze record |

Ordering matters: the cheap unit gate runs first (minutes), the end-to-end receipt second. During
this window the unit gates caught three defects in the B2 selector before any receipt ran
(an undefined shift in the pass-0 prefix mask, a threshold held in per-thread registers while
only thread 0 advanced it, and a lost rank offset), one defect in the weighted gate (uniform
extractor applied to the raw word instead of word>>8), and one in the reduction (a shared-memory
row width omitting three columns). The end-to-end token gate exists to catch exactly the class
of bug a unit test misses; both are retained.

The operator is the sole authority on output quality; agents measure and report.
