#!/usr/bin/env bash
# reproduce_stochastic_field.sh — build and run the synthetic Kolmogorov field with the
# receipt parameters, and emit the density rendering artifact (PPM).
# Expected (receipt KOLMO_FIELD_RECEIPT_20260907.md): all three analytic gates within
# Monte Carlo error; survival ~52.5%; ESS == survivors; kolmo_field_density.ppm written.
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "nvcc not found at $NVCC; set CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -I stoch/reduce         -o stoch/build/kolmo_field stoch/field/kolmo_field.cu

./stoch/build/kolmo_field 1048576 64 0.01
echo
echo "density artifact written to ./kolmo_field_density.ppm"
