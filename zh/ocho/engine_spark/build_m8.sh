#!/bin/bash
# build_m8.sh - DGX Spark (GB10, sm_121a) build of the M=8 multi-request engine
# (qwenflash_m8: QF_M8_PROMPTS harness, Spark-only authority path, per-layer
# routed offload client, and the alternating-map region client). Same link set
# as build_cuda.sh plus cuda/qf_region_amd.cpp, which the M=8 driver needs.
#   ./build_m8.sh            -> ./qwenflash_m8
#   OUT=/path ./build_m8.sh  -> /path
set -eu
cd "$(dirname "$0")"
ARCH="${QF_ARCH:-sm_121a}"
OUT="${OUT:-qwenflash_m8}"
FI="${QF_FLASHINFER:-$(pwd)/../third_party/flashinfer}"
FLAGS="-O3 -std=c++17 -lineinfo -arch=$ARCH -I$FI/include --expt-relaxed-constexpr"
nvcc $FLAGS -o "$OUT" \
    main.cu model.cpp reader.cpp planner.cpp cuda/qf.cu cuda/qf_dense.cu cuda/ple.cu cuda/qf_fp4tc.cu cuda/qf_fp4mma.cu cuda/prefill.cu \
    cuda/qf_attn_fused.cu cuda/qf_decode_graph.cu cuda/qf_routed_amd_layer.cu cuda/qf_routed_amd.cpp cuda/qf_moe_client.cpp \
    cuda/qf_region_amd.cpp cuda/qf_prefill_moe.cu cuda/qf_prefill_mma.cu cuda/qf_prefill_gemm.cu cuda/qf_qsa_index.cu cuda/qf_gdn_pf2.cu cuda/qf_fi_attn.cu cuda/qf_gdn_fla.cu \
    -lcublas -lcublasLt -lcuda -lpthread
echo "built: $OUT"
