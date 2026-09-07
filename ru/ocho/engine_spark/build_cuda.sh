#!/bin/bash
# build_cuda.sh - DGX Spark (GB10, sm_121a) engine build.
# qf.cu uses cuBLASLt with persistent per-shape plans for the dense BF16
# decode GEMVs, so the link needs -lcublasLt in addition to -lcublas.
# planner.cpp provides the fail-before-allocation budget guard (planner.h).
# qf_fp4tc.cu is the grouped native NVFP4 tensor-core MoE backend.
# qf_dense.cu fuses dense BF16 projection groups that share an activation.
set -eu
cd "$(dirname "$0")"

ARCH="${QF_ARCH:-sm_121a}"
FLAGS="-O3 -std=c++17 -lineinfo -arch=$ARCH"

# decode engine (production build line)
nvcc $FLAGS -o qwenflash \
    main.cu model.cpp reader.cpp planner.cpp cuda/qf.cu cuda/qf_dense.cu cuda/ple.cu cuda/qf_fp4tc.cu cuda/qf_fp4mma.cu cuda/prefill.cu \
    cuda/qf_attn_fused.cu cuda/qf_decode_graph.cu cuda/qf_routed_amd_layer.cu cuda/qf_routed_amd.cpp cuda/qf_moe_client.cpp \
    -lcublas -lcublasLt

# synth engine (tiny dims, SYNTH expert path; no planner)
nvcc $FLAGS -DSYNTH -o qwenflash_synth \
    main_synth.cu model.cpp reader.cpp cuda/qf.cu cuda/qf_dense.cu cuda/ple.cu cuda/qf_fp4tc.cu cuda/qf_fp4mma.cu cuda/prefill.cu \
    cuda/qf_attn_fused.cu cuda/qf_decode_graph.cu \
    -lcublas -lcublasLt

# prefill unit test (unchanged, kept working with the same link set)
nvcc $FLAGS cuda/test_prefill.cu cuda/prefill.cu -o test_prefill \
    -lcublas -lcublasLt
