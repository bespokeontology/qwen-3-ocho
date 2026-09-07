#!/usr/bin/env bash
# reproduce_stochastic_field.sh — 按回执参数构建并运行合成 Kolmogorov 场，并输出密度渲染工件（PPM）。
# 期望（回执 KOLMO_FIELD_RECEIPT_20260907.md）：三个解析门全部在蒙特卡洛误差内；
# 生存率约 52.5%；ESS == 幸存者数；写出 kolmo_field_density.ppm。
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "未找到 nvcc：$NVCC；请设置 CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -I stoch/reduce         -o stoch/build/kolmo_field stoch/field/kolmo_field.cu

./stoch/build/kolmo_field 1048576 64 0.01
echo
echo "密度工件已写入 ./kolmo_field_density.ppm"
