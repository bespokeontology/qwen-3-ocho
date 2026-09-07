#!/usr/bin/env bash
# reproduce_mc.sh — сборка и запуск слитных гейтов Монте-Карло.
# Ожидается (чек MC_GATES_RECEIPT_20260907.md; GB10, seed 42):
#   MC-A   оценка ~0.333675  ошибка ~3.4e-4
#   MC-A2  оценка ~0.333907  ошибка ~5.7e-4  ESS ~3.1e5
#   MC-A3  оценка ~0.333335  ошибка ~1.8e-6  (стратификация S=64)
#   MC-B   оценка ~5.334713  ошибка ~1.4e-3  (d=16, N=262144)
# На других GPU последние цифры могут меняться в пределах шума Монте-Карло; гейты —
# аналитические цели 1/3, 1/3, 1/3, 16/3.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "nvcc не найден: $NVCC; задайте CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -o stoch/build/mc_gate stoch/gates/mc_gate.cu

./stoch/build/mc_gate 0 1048576
./stoch/build/mc_gate 1 1048576
./stoch/build/mc_gate 3 1048576
./stoch/build/mc_gate 2 262144
