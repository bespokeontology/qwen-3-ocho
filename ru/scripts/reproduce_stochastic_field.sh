#!/usr/bin/env bash
# reproduce_stochastic_field.sh — сборка и запуск синтетического поля Колмогорова с
# параметрами чека и вывод артефакта рендеринга плотности (PPM).
# Ожидается (чек KOLMO_FIELD_RECEIPT_20260907.md): все три аналитических гейта в пределах
# ошибки Монте-Карло; выживаемость ~52.5%; ESS == числу выживших; записан kolmo_field_density.ppm.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "nvcc не найден: $NVCC; задайте CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -I stoch/reduce         -o stoch/build/kolmo_field stoch/field/kolmo_field.cu

./stoch/build/kolmo_field 1048576 64 0.01
echo
echo "артефакт плотности записан в ./kolmo_field_density.ppm"
