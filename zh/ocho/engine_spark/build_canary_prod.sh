#!/bin/bash
# build_canary_prod.sh - tapped binary at PRODUCTION dims.
# Same sources and same counts as src/build_cuda.sh, plus -DQF_CANARY_TAPS, so
# the real checkpoint can be instrumented stage by stage. Point it at a model
# with QF_CANARY_MODEL_DIR and give a prompt with QF_CANARY_PROMPT.
set -eu
cd "$(dirname "$0")"
ARCH="${QF_ARCH:-sm_121a}"
nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -DQF_CANARY_TAPS \
    -o "${1:-/tmp/qf_canary_prod}" \
    main_canary.cu model.cpp reader.cpp planner.cpp \
    cuda/qf.cu cuda/qf_dense.cu cuda/ple.cu cuda/qf_fp4tc.cu cuda/qf_fp4mma.cu cuda/prefill.cu \
    cuda/qf_attn_fused.cu cuda/qf_decode_graph.cu \
    -lcublas -lcublasLt
