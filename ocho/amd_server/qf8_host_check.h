// qf8_host_check.h - host-only syntax-check shim for the qf8_* dense TUs.
// One shim for the whole gfx906 tree: this is a thin wrapper over
// host_check_shim.h so a TU that includes both (qf_hip4_stage.hip) sees a
// single set of definitions. NOTHING here is numerically meaningful.
//   g++ -DQF_HOST_CHECK -I. -std=c++17 -fsyntax-only -x c++ qf8_gemv.hip
#pragma once
#include "host_check_shim.h"
