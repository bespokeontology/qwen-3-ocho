# Reproducibility

## Authority

The canonical freeze package FREEZE_QWEN_OVERNIGHT_20260907 (internal SHA256SUMS, mirrored on
three machines) is the authority for every number in this tree. The public receipts are
sanitized copies (hosts/paths replaced); the raw receipts are in the package. See
freeze/PROVENANCE.md and freeze/MANIFEST.md.

## Verification

    ./scripts/verify_freeze.sh

checks the public artifacts against the public-tree SHA256SUMS, and prints the provenance
class of each component. The private package manifest is kept verbatim at
freeze/package_SHA256SUMS for cross-checking the canonical package itself.

## Standalone reproduction (any CUDA GPU >= sm_70)

    ./scripts/reproduce_spark_tests.sh         # RNG + Dumitrescu unit tests
    ./scripts/reproduce_mc.sh                  # MC-A / MC-A2 / MC-A3 / MC-B
    ./scripts/reproduce_stochastic_field.sh    # synthetic field + density artifact

Requirements: CUDA >= 12 (nvcc), no external dependencies beyond the CUDA toolkit for these
targets. The retained builds target sm_121a; the scripts honor CUDA_HOME and QF_ARCH. Expected
outputs are stated in each script header and must fall within the receipt tolerances.

## Appliance reproduction (two boxes, frozen binaries)

The model-backed experiments and the decode receipts require the two-box appliance with the
frozen binaries, which are not part of this tree. The exact environment and recipes are quoted in
the receipts (sanitized) and in the canonical freeze package. Key configuration: production split
QF_LAYER_BEGIN=16, QF_AMD_PREFIX=16, QF_MAX_CONTEXT=16384, QF_EXPERT_MODE=full, locked clocks;
fork topology QF_LAYER_BEGIN=8, QF_AMD_PREFIX=8, QF_M1_INT8_HEAD=0. Model checkpoint identity:
RadixArk-Qwen3.8-Flash-Next-NVFP4 (directory layout per the freeze package). The frozen AMD
production binary is 824e5c50…; its exact source tree has not been recovered (PENDING-PROVENANCE),
so a from-source rebuild of the AMD side is not reproducible at this time.

## Claim-to-receipt mapping (numerical audit)

| claim | receipt |
|---|---|
| RNG KAT + determinism | RNG_RECEIPT_20260907.md |
| reduction CPU-reference validation | DUMI_RECEIPT_20260907.md |
| MC-A / MC-A2 / MC-B numbers | MC_GATES_RECEIPT_20260907.md |
| stratified 188x error reduction | VARIANCE_REDUCTION_RECEIPT_20260907.md |
| field analytic gates + survival | KOLMO_FIELD_RECEIPT_20260907.md |
| throughput law numbers | THROUGHPUT_LAW_RECEIPT_20260907.md |
| MC-C field statistics | MCC_MODEL_FIELD_RECEIPT_20260907.md |
| B1 drift/identity gates | B1_CAPTURE_FIX_RECEIPT_20260907.md |
| B2 unit + 300/300 + tail timing | B2_RADIX_RECEIPT_20260907.md |
| B3 unit gate + provenance blocker | B3_AMD_TWIN_RECEIPT_20260907.md |
| prefill 5.21 s / 1,543 tok/s | FREEZE_QWEN_PREFILL_1543 (earlier freeze) |
| refill rates | QWEN_REFILL_RECEIPT (earlier freeze) |

## Units

samples/s, trajectory-steps/s, completed trajectories/s, and sibling-tokens/s are different
quantities and are labeled as such everywhere. ms/token and tok/s are reciprocals of the same
measurement only within one configuration; mixing boots or configurations in one comparison is
disallowed without labels.
