#!/usr/bin/env bash
# verify_freeze.sh — 按公开 SHA256SUMS 校验公开树工件，并打印各组件的来源类别
# （freeze/PROVENANCE.md）。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 公开树 SHA256SUMS =="
sha256sum -c freeze/SHA256SUMS

echo
echo "== 来源类别 =="
printf "%-46s %s\n" "路径" "标签"
printf "%-46s %s\n" "stoch/（引擎源码）" "AUTHORITATIVE（权威）"
printf "%-46s %s\n" "figures/kolmo_field_density.ppm" "HASH-VERIFIED（保留工件）"
printf "%-46s %s\n" "figures/kolmo_field_density.png" "HASH-VERIFIED（无损转换）"
printf "%-46s %s\n" "src/cuda/*.cu、src/main.cu" "PATCH-DERIVED（补丁派生）"
printf "%-46s %s\n" "patches/lane_a、patches/lane_b" "HASH-VERIFIED"
printf "%-46s %s\n" "patches/amd_pending" "EXPERIMENTAL（实验性，已单元验证，待集成）"
printf "%-46s %s\n" "receipts/" "SANITIZED COPY（脱敏副本；原文在规范包）"
printf "%-46s %s\n" "freeze/package_SHA256SUMS" "HASH-VERIFIED（包清单逐字复制）"
printf "%-46s %s\n" "AMD 生产源码（824e5c50…）" "PENDING-PROVENANCE（不在本树）"
