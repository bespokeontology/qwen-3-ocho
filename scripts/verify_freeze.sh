#!/usr/bin/env bash
# verify_freeze.sh — check public-tree artifacts against the public SHA256SUMS,
# and print the provenance class of each component (freeze/PROVENANCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== public-tree SHA256SUMS =="
sha256sum -c freeze/SHA256SUMS

echo
echo "== provenance classes =="
printf "%-46s %s\n" "path" "label"
printf "%-46s %s\n" "stoch/ (engine sources)" "AUTHORITATIVE"
printf "%-46s %s\n" "figures/kolmo_field_density.ppm" "HASH-VERIFIED (retained artifact)"
printf "%-46s %s\n" "figures/kolmo_field_density.png" "HASH-VERIFIED (lossless conversion)"
printf "%-46s %s\n" "src/cuda/*.cu, src/main.cu" "PATCH-DERIVED"
printf "%-46s %s\n" "patches/lane_a, patches/lane_b" "HASH-VERIFIED"
printf "%-46s %s\n" "patches/amd_pending" "EXPERIMENTAL / UNIT-VERIFIED / PENDING-INTEGRATION"
printf "%-46s %s\n" "receipts/" "SANITIZED COPY (raw receipts: canonical package)"
printf "%-46s %s\n" "freeze/package_SHA256SUMS" "HASH-VERIFIED (verbatim package manifest)"
printf "%-46s %s\n" "AMD production source (824e5c50…)" "PENDING-PROVENANCE (not present)"
