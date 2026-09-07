#!/bin/bash
# Lane A stochastic engine — standalone CUDA build (sm_121a, CUDA 13.0).
set -eu
cd "$(dirname "$0")"
NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
ARCH="${QF_ARCH:-sm_121a}"
mkdir -p build
$NVCC -O3 -std=c++17 -lineinfo -arch=$ARCH -I include -o build/rng_test   rng/rng_test.cu
$NVCC -O3 -std=c++17 -lineinfo -arch=$ARCH -I include -o build/dumi_test reduce/dumi_test.cu
$NVCC -O3 -std=c++17 -lineinfo -arch=$ARCH -I include -o build/mc_gate   gates/mc_gate.cu
$NVCC -O3 -std=c++17 -lineinfo -arch=$ARCH -I include -I reduce -o build/kolmo_field field/kolmo_field.cu
echo "built: rng_test dumi_test mc_gate kolmo_field"
