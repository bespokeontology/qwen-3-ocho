# Makefile - qf_hip4_server (4x gfx906 MI50 pipeline engine server).
#
#   make            / make server   host check: stub engine, links anywhere
#   make server-target            target build: real engine objects (ROCm host)
#   make check                    stub server build + tokenizer smoke test
#
# Target builds are driven by deploy/build_ubuntu.sh. ENGINESRC points at the
# deliverable tree root (provides src/qf_hip4.h + src/qwenflash.h);
# ENGINE_OBJS lists the real engine objects built by src/Makefile.hip4.

CXX        ?= g++
CXXFLAGS   ?= -O2 -std=c++17 -Wall -Wextra
ENGINESRC  ?= ../..
ROCM_PATH  ?= /opt/rocm
ENGINE_OBJS ?= $(ENGINESRC)/src/qf_hip4_stage.o $(ENGINESRC)/src/qf_nvfp4_wave64.o \
               $(ENGINESRC)/src/qf_router_wave64.o \
               $(ENGINESRC)/src/qf_expert_slots.o $(ENGINESRC)/src/qf_hip4_pipeline.o \
               $(ENGINESRC)/src/qf8_gemv.o $(ENGINESRC)/src/qf8_hc.o \
               $(ENGINESRC)/src/qf8_gdn.o $(ENGINESRC)/src/qf8_qsa.o \
               $(ENGINESRC)/src/qf8_ple.o $(ENGINESRC)/src/qf8_layer.o

# Host builds: -Ihost shadows the real engine header with the frozen-API stub.
HOSTINC := -Ihost -I..

SERVER_SRCS := qf_hip4_server.cpp sampler.cpp tokenizer_json.cpp json.cpp

server: qf_hip4_server_stub

qf_hip4_server_stub: $(SERVER_SRCS) engine_stub.cpp sampler.h host/qf_hip4.h
	$(CXX) $(CXXFLAGS) $(HOSTINC) $(SERVER_SRCS) engine_stub.cpp -o $@ -lpthread

server-target: $(SERVER_SRCS) sampler.h
	mkdir -p $(ENGINESRC)/bin
	mkdir -p $(ENGINESRC)/build/server
	$(CXX) $(CXXFLAGS) -I$(ENGINESRC)/src -c qf_hip4_server.cpp -o $(ENGINESRC)/build/server/qf_hip4_server.o
	$(CXX) $(CXXFLAGS) -I$(ENGINESRC)/src -c sampler.cpp -o $(ENGINESRC)/build/server/sampler.o
	$(CXX) $(CXXFLAGS) -I$(ENGINESRC)/src -c tokenizer_json.cpp -o $(ENGINESRC)/build/server/tokenizer_json.o
	$(CXX) $(CXXFLAGS) -I$(ENGINESRC)/src -c json.cpp -o $(ENGINESRC)/build/server/json.o
	$(CXX) $(CXXFLAGS) -I$(ENGINESRC)/src -c $(ENGINESRC)/src/reader.cpp -o $(ENGINESRC)/build/server/reader.o
	$(ROCM_PATH)/bin/hipcc -O3 --offload-arch=gfx906 -fno-gpu-rdc \
	    $(ENGINESRC)/build/server/qf_hip4_server.o $(ENGINESRC)/build/server/sampler.o \
	    $(ENGINESRC)/build/server/tokenizer_json.o $(ENGINESRC)/build/server/json.o \
	    $(ENGINESRC)/build/server/reader.o $(ENGINE_OBJS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib \
	    -lamdhip64 -lrocblas -lpthread -o $(ENGINESRC)/bin/qf_hip4_server

check: qf_hip4_server_stub tokenizer-check
	@echo "stub server link OK: ./qf_hip4_server_stub"

tokenizer-check: test/tokenizer_json_test.cpp tokenizer_json.cpp json.cpp
	$(CXX) $(CXXFLAGS) -I. test/tokenizer_json_test.cpp tokenizer_json.cpp json.cpp \
	    -o /tmp/qf_tokenizer_json_test
	/tmp/qf_tokenizer_json_test

clean:
	rm -f qf_hip4_server_stub /tmp/qf_tokenizer_json_test

.PHONY: server server-target check tokenizer-check clean
