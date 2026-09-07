// host_check_shim.h - host-only syntax-check shim for the merged wave64 stage
// tree (session 05 wave-2 merge). Single shim covering both the prior
// qf_hip4_stage.hip needs and the session-05 NVFP4 kernels / expert-slot
// manager (uint2, events).
//
// Compiles the device translation units with a plain C++ compiler so CI or a
// GPU-less dev box can catch syntax/type errors:
//   g++ -DQF_HOST_CHECK -I. -I$(ENGINESRC) -fsyntax-only -x c++ qf_hip4_stage.hip
//   g++ -DQF_HOST_CHECK -I. -I$(ENGINESRC) -fsyntax-only -x c++ qf_nvfp4_wave64.hip
//   g++ -DQF_HOST_CHECK -I. -I$(ENGINESRC) -fsyntax-only qf_expert_slots.cpp
// Kernels become plain functions (launches collapse via QF4_LAUNCH /
// QF5_LAUNCH), device intrinsics become trivial stubs. NOTHING here is
// numerically meaningful; this exists only to make the TUs parse and
// type-check.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#define warpSize 64

struct Qf4Dim3 {
    unsigned x, y, z;
    Qf4Dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
typedef Qf4Dim3 dim3;
// One definition per check TU; only the .hip check includes this header.
static dim3 threadIdx(0, 0, 0), blockIdx(0, 0, 0), blockDim(0, 0, 0), gridDim(0, 0, 0);

// launch syntax is already hidden behind QF4_LAUNCH / QF5_LAUNCH under
// QF_HOST_CHECK.

typedef int hipError_t;
typedef void *hipStream_t;
#define hipSuccess 0
#define hipErrorOutOfMemory 2
#define hipErrorUnknown 999
#define hipErrorPeerAccessAlreadyEnabled 704
#define hipHostMallocMapped 1
#define hipHostMallocPortable 2
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
#define hipMemcpyDeviceToDevice 3

static inline hipError_t hipSetDevice(int) { return 0; }
// HIP's hipMalloc/hipHostMalloc are templates over T** (any pointer type);
// mirror that so callers need not cast to void**.
template <typename T> static inline hipError_t hipMalloc(T **p, size_t n) { *p = (T *)__builtin_malloc(n); return 0; }
static inline hipError_t hipFree(void *) { return 0; }
static inline hipError_t hipMemset(void *, int, size_t) { return 0; }
static inline hipError_t hipMemcpy(void *, const void *, size_t, int) { return 0; }
static inline hipError_t hipMemcpyAsync(void *, const void *, size_t, int, hipStream_t) { return 0; }
static inline hipError_t hipMemsetAsync(void *, int, size_t, hipStream_t) { return 0; }
static inline hipError_t hipDeviceSynchronize(void) { return 0; }
template <typename T> static inline hipError_t hipHostMalloc(T **p, size_t n, int) { *p = (T *)__builtin_malloc(n); return 0; }
static inline hipError_t hipHostFree(void *) { return 0; }
static inline hipError_t hipHostGetDevicePointer(void **d, void *h, int) { *d = h; return 0; }
static inline hipError_t hipStreamCreate(hipStream_t *) { return 0; }
static inline hipError_t hipStreamDestroy(hipStream_t) { return 0; }
static inline hipError_t hipStreamSynchronize(hipStream_t) { return 0; }
static inline hipError_t hipGetLastError(void) { return 0; }
static inline const char *hipGetErrorString(hipError_t) { return "host-check"; }

// HIP vector type used by the packed-weight loads (8 bytes = one NVFP4
// scale group of 16 values).
struct uint2 {
    uint32_t x, y;
};
struct uint4 { uint32_t x, y, z, w; };
struct float4 { float x, y, z, w; };
static inline uint2 make_uint2(uint32_t x, uint32_t y) { uint2 v; v.x = x; v.y = y; return v; }
static inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { uint4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline float4 make_float4(float x, float y, float z, float w) { float4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
static inline int atomicAdd(int *a, int v) { int o = *a; *a += v; return o; }
static inline unsigned atomicAdd(unsigned *a, unsigned v) { unsigned o = *a; *a += v; return o; }
#define hipStreamNonBlocking 1
static inline hipError_t hipStreamCreateWithFlags(hipStream_t *, unsigned) { return 0; }
static inline hipError_t hipMemcpyPeer(void *, int, const void *, int, size_t) { return 0; }
static inline hipError_t hipMemGetInfo(size_t *f, size_t *t) { *f = 0; *t = 0; return 0; }
static inline hipError_t hipGetDeviceCount(int *n) { *n = 4; return 0; }
static inline hipError_t hipDeviceReset(void) { return 0; }

// Events for the pinned-staging -> slot H2D overlap.
typedef void *hipEvent_t;
#define hipEventDisableTiming 2
static inline hipError_t hipEventCreateWithFlags(hipEvent_t *, int) { return 0; }
static inline hipError_t hipEventCreate(hipEvent_t *) { return 0; }
static inline hipError_t hipEventElapsedTime(float *ms, hipEvent_t, hipEvent_t) { *ms = 0.f; return 0; }
static inline hipError_t hipEventQuery(hipEvent_t) { return 0; }
static inline hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return 0; }
static inline hipError_t hipEventSynchronize(hipEvent_t) { return 0; }
static inline hipError_t hipEventDestroy(hipEvent_t) { return 0; }
static inline hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned) { return 0; }

// Peer-to-peer handoff (wave4 session 09): capability probe + peer copies.
static inline hipError_t hipDeviceCanAccessPeer(int *c, int, int) { *c = 0; return 0; }
static inline hipError_t hipDeviceEnablePeerAccess(int, unsigned) { return 0; }
static inline hipError_t hipMemcpyPeerAsync(void *, int, const void *, int, size_t, hipStream_t) { return 0; }

static inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
template <typename T> static inline T __shfl_down_sync(unsigned long long, T v, int) { return v; }
template <typename T> static inline T __shfl_sync(unsigned long long, T v, int) { return v; }
static inline void __syncthreads(void) {}
static inline void __threadfence_system(void) {}
static inline void __threadfence(void) {}
static inline float atomicAdd(float *a, float v) { float o = *a; *a += v; return o; }
#define __expf expf
static inline float rsqrtf(float v) { return 1.f / sqrtf(v); }   // device intrinsic on HIP
