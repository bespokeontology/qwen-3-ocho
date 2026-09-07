# QF server - plain Makefile build (no CMake).
#
# Host-only checks (no CUDA toolchain, no model, no GPU):
#   make server-check      # full server compile+link against a stub engine
#   make tokenizer-check   # tokenizer.json loader unit test
#
# Production engine build (requires nvcc on the DGX Spark):
#   make                   # ENGINE_SRC auto-detected, or pass ENGINE_SRC=<path>
#
# ENGINE_SRC must point at the engine source dir containing qwenflash.h
# (i.e. the patched <engine>/src). Auto-detected when this Makefile lives
# inside the engine tree at src/server/, or in the wave-4 out/ drop.
ENGINE_SRC ?= $(patsubst %/,%,$(dir $(firstword $(wildcard ../qwenflash.h ../../../../source/src/qwenflash.h))))
CXX       ?= c++
NVCC      ?= nvcc
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra -pthread
# Production flags per STATUS_2026-08-28: GB10 = sm_121a, lineinfo for profiling.
# qf.cu sizes KV caches from QF_MAX_CONTEXT at runtime (default 262144);
# override the compile-time ceiling with -DMAXPOS=N if ever needed.
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_121a -lineinfo
LDFLAGS   ?= -pthread -ldl

# Server sources live flat alongside this Makefile after the patch is applied
# to the engine tree (src/server/), and that is also how they ship in the
# out/ drop.
SRVDIR := $(if $(wildcard json.cpp),.,src)
SERVER_SRCS := $(SRVDIR)/json.cpp $(SRVDIR)/http_server.cpp $(SRVDIR)/tokenizer.cpp \
    $(SRVDIR)/tokenizer_json.cpp $(SRVDIR)/scheduler.cpp $(SRVDIR)/qf_server.cpp
ENGINE_SRCS := $(ENGINE_SRC)/model.cpp $(ENGINE_SRC)/reader.cpp $(ENGINE_SRC)/planner.cpp
ENGINE_CU   := $(ENGINE_SRC)/generate.cu $(ENGINE_SRC)/cuda/qf.cu \
    $(ENGINE_SRC)/cuda/qf_dense.cu $(ENGINE_SRC)/cuda/prefill.cu \
    $(ENGINE_SRC)/cuda/ple.cu $(ENGINE_SRC)/cuda/qf_fp4tc.cu \
    $(ENGINE_SRC)/cuda/qf_attn_fused.cu $(ENGINE_SRC)/cuda/qf_decode_graph.cu

BUILD := build

# Host-only build with a stub engine: verifies the full server compiles and
# links without a CUDA toolchain. Not for serving.
server-check: $(BUILD)/qf_server_stub
	@echo "server-check OK"

$(BUILD)/qf_server_stub: $(SERVER_SRCS) test/engine_stub.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -I $(SRVDIR) -I $(ENGINE_SRC) $^ -o $@ $(LDFLAGS)

# Host-only tokenizer.json loader unit test (no engine involved).
tokenizer-check: $(BUILD)/tokenizer_json_test
	$(BUILD)/tokenizer_json_test
	@echo "tokenizer-check OK"

$(BUILD)/tokenizer_json_test: $(SRVDIR)/json.cpp $(SRVDIR)/tokenizer.cpp \
    $(SRVDIR)/tokenizer_json.cpp test/tokenizer_json_test.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -I $(SRVDIR) -I $(ENGINE_SRC) $^ -o $@ $(LDFLAGS)

# Full production build against the real engine (DGX Spark, sm_121a).
qf_server: $(SERVER_SRCS) $(ENGINE_SRCS) $(ENGINE_CU) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -x cu $(filter %.cpp,$(SERVER_SRCS) $(ENGINE_SRCS)) \
	    $(ENGINE_CU) -I $(SRVDIR) -I $(ENGINE_SRC) -o $(BUILD)/$@ \
	    -Xcompiler=-pthread -ldl \
	    -lcudart -lcublas -lcublasLt

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: server-check tokenizer-check clean
