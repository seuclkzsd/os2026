// GPU module (P10 "gpu"; docs/DESIGN.md §3.9, docs/BLAS.md §10): the CUDA runtime and cuBLAS,
// loaded at run time with dlopen. hsolve needs no CUDA at build time and runs unchanged where no
// NVIDIA driver, CUDA library or GPU exists: load() then reports why and every caller stays on the
// CPU. Scalar set: none (function table); the GPU work itself is FP64 (cublasDgemm).
//
// Only the handful of entry points the GEMM engine (gemm_engine.hpp) uses are declared, with the
// ABI of the CUDA 11 / 12 headers (enum values checked against CUDA 12.2's driver_types.h and
// cublas_api.h); handles and streams are opaque pointers. The libraries are searched in this order
// (the first directory that has both wins, so cudart and cuBLAS always come from one toolkit):
//   1. $HS_CUDA_LIBDIR (a directory) when set;
//   2. the dynamic loader's own search (LD_LIBRARY_PATH, RUNPATH, ld.so.cache) for the sonames
//      libcudart.so.12 / libcublas.so.12, then .so.11.0 / .so.11;
//   3. /usr/local/cuda/lib64, then /usr/local/cuda-12*/lib64 and /usr/local/cuda-11*/lib64 (newest
//      first).
// libcublasLt is opened from cuBLAS' own directory first (cuBLAS needs it; its RUNPATH is $ORIGIN
// in the NVIDIA packages, but not in every distribution's).
//
// Thread safety: load() may be called from any thread, any number of times; the first call loads
// (std::call_once), later calls return the same table. The table is immutable afterwards. The
// libraries stay loaded until the process exits (never dlclose'd: the CUDA runtime registers exit
// handlers).
#pragma once

#include <cstddef>
#include <string>

namespace hs::gpu {

// CUDA runtime / cuBLAS ABI (subset). Values from CUDA 12.2's headers; stable since CUDA 10.
using cudaError_t = int;
using cudaStream_t = struct CUstream_st*;
using cudaEvent_t = struct CUevent_st*;
using cublasHandle_t = struct cublasContext*;
using cublasStatus_t = int;

constexpr cudaError_t kCudaSuccess = 0;
constexpr cudaError_t kCudaErrorNotReady = 600;
constexpr int kCudaMemcpyHostToDevice = 1;
constexpr int kCudaMemcpyDeviceToHost = 2;
constexpr unsigned kCudaHostAllocDefault = 0x00;
constexpr unsigned kCudaStreamNonBlocking = 0x01;
constexpr unsigned kCudaEventBlockingSync = 0x01;
constexpr unsigned kCudaEventDisableTiming = 0x02;
constexpr unsigned kCudaDeviceScheduleBlockingSync = 0x04;
constexpr unsigned kCudaHostRegisterDefault = 0x00;
constexpr int kCudaDevAttrMultiProcessorCount = 16;
constexpr int kCudaDevAttrComputeCapabilityMajor = 75;
constexpr int kCudaDevAttrComputeCapabilityMinor = 76;
constexpr int kCublasStatusSuccess = 0;
constexpr int kCublasOpN = 0;
constexpr int kCublasOpT = 1;
constexpr int kCublasDefaultMath = 0;

struct CudaLib {
  // CUDA runtime
  cudaError_t (*GetDeviceCount)(int*) = nullptr;
  cudaError_t (*SetDevice)(int) = nullptr;
  cudaError_t (*SetDeviceFlags)(unsigned) = nullptr;
  cudaError_t (*DeviceGetAttribute)(int*, int, int) = nullptr;
  cudaError_t (*DeviceGetPCIBusId)(char*, int, int) = nullptr;
  cudaError_t (*MemGetInfo)(std::size_t*, std::size_t*) = nullptr;
  cudaError_t (*Malloc)(void**, std::size_t) = nullptr;
  cudaError_t (*Free)(void*) = nullptr;
  cudaError_t (*HostAlloc)(void**, std::size_t, unsigned) = nullptr;
  cudaError_t (*FreeHost)(void*) = nullptr;
  cudaError_t (*HostRegister)(void*, std::size_t, unsigned) = nullptr;
  cudaError_t (*HostUnregister)(void*) = nullptr;
  cudaError_t (*MemcpyAsync)(void*, const void*, std::size_t, int, cudaStream_t) = nullptr;
  cudaError_t (*Memcpy2DAsync)(void*, std::size_t, const void*, std::size_t, std::size_t,
                               std::size_t, int, cudaStream_t) = nullptr;
  cudaError_t (*StreamCreateWithFlags)(cudaStream_t*, unsigned) = nullptr;
  cudaError_t (*StreamDestroy)(cudaStream_t) = nullptr;
  cudaError_t (*StreamSynchronize)(cudaStream_t) = nullptr;
  cudaError_t (*StreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned) = nullptr;
  cudaError_t (*EventCreateWithFlags)(cudaEvent_t*, unsigned) = nullptr;
  cudaError_t (*EventDestroy)(cudaEvent_t) = nullptr;
  cudaError_t (*EventRecord)(cudaEvent_t, cudaStream_t) = nullptr;
  cudaError_t (*EventSynchronize)(cudaEvent_t) = nullptr;
  cudaError_t (*EventQuery)(cudaEvent_t) = nullptr;
  cudaError_t (*DeviceSynchronize)() = nullptr;
  cudaError_t (*DriverGetVersion)(int*) = nullptr;
  cudaError_t (*RuntimeGetVersion)(int*) = nullptr;
  cudaError_t (*GetLastError)() = nullptr;
  const char* (*GetErrorString)(cudaError_t) = nullptr;
  // cuBLAS (the _v2 entry points that cublas_v2.h maps cublasCreate & co. to)
  cublasStatus_t (*BlasCreate)(cublasHandle_t*) = nullptr;
  cublasStatus_t (*BlasDestroy)(cublasHandle_t) = nullptr;
  cublasStatus_t (*BlasSetStream)(cublasHandle_t, cudaStream_t) = nullptr;
  cublasStatus_t (*BlasSetMathMode)(cublasHandle_t, int) = nullptr;
  cublasStatus_t (*BlasGetVersion)(cublasHandle_t, int*) = nullptr;
  cublasStatus_t (*Dgemm)(cublasHandle_t, int, int, int, int, int, const double*, const double*,
                          int, const double*, int, const double*, double*, int) = nullptr;

  std::string cudart_path;  ///< file the runtime was loaded from (dladdr)
  std::string cublas_path;  ///< file cuBLAS was loaded from
};

/// The loaded libraries, or nullptr with `*why` (when given) saying what is missing (no library
/// found, a missing entry point). Loading does not touch a GPU: no CUDA call is made.
const CudaLib* load(std::string* why = nullptr);

/// "cuda error <code> (<string>)" for a runtime error code (the string only when loaded).
std::string error_text(const CudaLib* lib, cudaError_t e);

}  // namespace hs::gpu
