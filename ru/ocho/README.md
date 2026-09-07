# ocho/ — Qwen 3.ocho capability sources

The complete code that implements the Qwen 3.ocho capability boundary (2026-09-07): the Spark
engine tree (CUDA) and the AMD region server tree (HIP/ROCm), both with the Ocho Loop and the
commit wire op in place. Mechanism, measurements, and limitations: ../docs/QWEN_3_OCHO.md.

## Layout

- `engine_spark/` — the complete NVIDIA-side engine sources (CUDA 13.0, sm_121a) from
  stoch-engine-0907 @ 257a2a1: the M=8 request harness with the Ocho Loop
  (fork -> chunked render with live previews -> artifact/EOS filter -> median-sigR chooser ->
  commit -> continue), the Spark-side commit copy, and the QFW_REGION_COMMIT client.
  Build: `OUT=qwenflash_m8.loop ./build_m8.sh` from this directory.
- `amd_server/` — the complete AMD-side region server sources (ROCm 5.7, HIP, gfx906) from the
  archived pre-B3 fork tree plus the QFW_REGION_COMMIT op, the stage/PLE commit copies, and the
  field-identity reset (qf4_stage_head_calls_reset on REGION_FORK/REGION_RESET).
  Build: `make -f Makefile.hip4 qf_m8_region_server`.
- `ocho_loop_client.patch` — the same client changes as a diff (stoch-engine-0907
  b4bb1d2..257a2a1), for applying to a private tree instead of this copy.

## Validated binaries

Client: qwenflash_m8.loop.v3 sha256 b74089d8014b105d3578063f9b1c162de5966db563f50ad5d2cc87c5d31528e8.
Server: sha256 4c9155680a604f4aa6cee2eb6ca29035f52d45b9e464404e61e22b4e2ce40090.

## Validation record

- Same-boot field-identity gate: two identical K=16 fork requests on one server boot produce
  bit-identical 8-row fields.
- Capability run: QF_M8_FORK=8 QF_M8_FORK_STEPS=48 QF_M8_FORK_LOOP=1 QF_M8_FORK_LOOP_ITERS=16
  on the C++ task prompt — 16 decision/commit cycles, 768 committed tokens, clean exit.
  Full logs: FREEZE_OCHO_CAPABILITY_20260907.

## Sanitization and provenance

The FLA AOT kernels (engine_spark/fla_aot/*.cubin) are not shipped: the compiled
artifacts embed build-machine paths. Rebuild them with fla_aot/harvest_recipe.py.

This directory is a SANITIZED COPY of the working trees: machine-specific paths, addresses, and
directory names were replaced with environment-variable defaults (QF_MODEL_DIR, QF_FLA_DIR) or
relative paths. The working trees themselves are unchanged and remain the authoritative sources.
The AMD files descend from the archived fork-era tree whose built binary matched the frozen fork
server 5179fff3 byte-for-byte; the production server 824e5c50 and its freeze are untouched by
anything in this directory.
