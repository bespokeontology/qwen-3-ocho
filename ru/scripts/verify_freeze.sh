#!/usr/bin/env bash
# verify_freeze.sh — проверка публичных артефактов по публичным SHA256SUMS и печать
# класса происхождения каждого компонента (freeze/PROVENANCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== SHA256SUMS публичного дерева =="
sha256sum -c freeze/SHA256SUMS

echo
echo "== классы происхождения =="
printf "%-46s %s\n" "путь" "метка"
printf "%-46s %s\n" "stoch/ (исходники движка)" "AUTHORITATIVE"
printf "%-46s %s\n" "figures/kolmo_field_density.ppm" "HASH-VERIFIED (сохранённый артефакт)"
printf "%-46s %s\n" "figures/kolmo_field_density.png" "HASH-VERIFIED (lossless-конверсия)"
printf "%-46s %s\n" "src/cuda/*.cu, src/main.cu" "PATCH-DERIVED"
printf "%-46s %s\n" "patches/lane_a, patches/lane_b" "HASH-VERIFIED"
printf "%-46s %s\n" "patches/amd_pending" "EXPERIMENTAL / UNIT-VERIFIED / PENDING-INTEGRATION"
printf "%-46s %s\n" "receipts/" "SANITIZED COPY (оригиналы: канонический пакет)"
printf "%-46s %s\n" "freeze/package_SHA256SUMS" "HASH-VERIFIED (дословный манифест пакета)"
printf "%-46s %s\n" "продуктовые исходники AMD (824e5c50…)" "PENDING-PROVENANCE (отсутствуют)"
