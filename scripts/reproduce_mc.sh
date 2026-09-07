#!/usr/bin/env bash
# reproduce_mc.sh — build and run the fused Monte Carlo gates.
# Expected (receipt MC_GATES_RECEIPT_20260907.md; GB10, seed 42):
#   MC-A   est ~0.333675  |err| ~3.4e-4
#   MC-A2  est ~0.333907  |err| ~5.7e-4  ESS ~3.1e5
#   MC-A3  est ~0.333335  |err| ~1.8e-6  (stratified S=64)
#   MC-B   est ~5.334713  |err| ~1.4e-3  (d=16, N=262144)
# Different GPUs may shift the last digits within Monte Carlo noise; the gates are the
# analytic targets 1/3, 1/3, 1/3, 16/3.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "nvcc not found at $NVCC; set CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -o stoch/build/mc_gate stoch/gates/mc_gate.cu

./stoch/build/mc_gate 0 1048576
./stoch/build/mc_gate 1 1048576
./stoch/build/mc_gate 3 1048576
./stoch/build/mc_gate 2 262144
