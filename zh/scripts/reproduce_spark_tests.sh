#!/usr/bin/env bash
# reproduce_spark_tests.sh — 从公开 stoch/ 树构建并运行 RNG 与 Dumitrescu 单元测试。
# 期望输出：RNG_TEST: ALL PASS；DUMI_TEST: ALL PASS。
# 要求：CUDA >= 12（nvcc），GPU >= sm_70。
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "未找到 nvcc：$NVCC；请设置 CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -o stoch/build/rng_test   stoch/rng/rng_test.cu
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -o stoch/build/dumi_test stoch/reduce/dumi_test.cu

echo "== rng_test =="
./stoch/build/rng_test
echo "== dumi_test =="
./stoch/build/dumi_test
