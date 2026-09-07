#!/usr/bin/env bash
# reproduce_mc.sh — 构建并运行融合蒙特卡洛门。
# 期望（回执 MC_GATES_RECEIPT_20260907.md；GB10，seed 42）：
#   MC-A   估计约 0.333675  误差约 3.4e-4
#   MC-A2  估计约 0.333907  误差约 5.7e-4  ESS 约 3.1e5
#   MC-A3  估计约 0.333335  误差约 1.8e-6  （分层 S=64）
#   MC-B   估计约 5.334713  误差约 1.4e-3  （d=16，N=262144）
# 不同 GPU 的最后几位可能在蒙特卡洛噪声内变化；门是解析目标 1/3、1/3、1/3、16/3。
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
ARCH="${QF_ARCH:-sm_121a}"
NVCC="${CUDA_HOME}/bin/nvcc"
[ -x "$NVCC" ] || { echo "未找到 nvcc：$NVCC；请设置 CUDA_HOME"; exit 1; }

mkdir -p stoch/build
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I stoch/include -o stoch/build/mc_gate stoch/gates/mc_gate.cu

./stoch/build/mc_gate 0 1048576
./stoch/build/mc_gate 1 1048576
./stoch/build/mc_gate 3 1048576
./stoch/build/mc_gate 2 262144
