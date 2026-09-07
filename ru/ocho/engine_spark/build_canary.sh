#!/bin/bash
# build_canary.sh - production-geometry correctness canary.
#
# Same sources and same kernels as the production build (src/build_cuda.sh);
# only the COUNTS are overridden, so every width, tile shape and kernel the
# canary exercises is the one production runs. -DQF_CANARY_TAPS compiles the
# per-layer taps in for this target only.
set -eu
cd "$(dirname "$0")"
ARCH="${QF_ARCH:-sm_121a}"
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    -DNLAYER=4 -DNEXP=16 -DNVOCAB=4096 -DQF_EOS_ID=3000 -DQF_CANARY_TAPS \
    -o "${1:-/tmp/qf_canary}" \
    main_canary.cu model.cpp reader.cpp planner.cpp \
    cuda/qf.cu cuda/qf_dense.cu cuda/ple.cu cuda/qf_fp4tc.cu cuda/qf_fp4mma.cu cuda/prefill.cu \
    cuda/qf_attn_fused.cu cuda/qf_decode_graph.cu \
    -lcublas -lcublasLt
