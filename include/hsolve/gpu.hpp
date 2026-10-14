// GPU offload control for applications (P10 "gpu"; docs/DESIGN.md §3.9, docs/BLAS.md §10).
// Scalar set: none.
//
// The GPU offload lives in the BLAS hook (src/blas + src/gpu), which is linked into the hsolve
// executable by default or preloaded (libhsolve_blashook.so) - not in libhsolve. These inline
// helpers find the hook's C entry points in the running process with dlsym(RTLD_DEFAULT), so an
// application works the same with and without the hook (link ${CMAKE_DL_LIBS}).
//
// Modes: "off" (default: exactly the CPU path, nothing CUDA is loaded), "auto" (factorizations
// whose MUMPS flop estimate is large enough offload their large dgemm calls to the first visible
// GPU, if there is one), "on" (every factorization offloads); without a usable GPU or CUDA library
// both auto and on run the CPU path and say why in the report. The environment variables HS_GPU
// (mode) and HS_GPU_OPTS (tuning, src/gpu/offload.hpp) configure the same without code.
//
// Thread safety: as the hook's functions (any thread; the mode applies from the next
// factorization).
#pragma once

#include <dlfcn.h>

#include <string>

namespace hs {

/// Result of gpu_offload_set_mode.
enum class GpuControl { Ok = 0, InvalidMode = -1, NotBuilt = -2, NoHook = -3 };

/// Sets the GPU mode ("off" | "auto" | "on") of the BLAS hook in this process. NotBuilt: the hook
/// was built without the GPU module; NoHook: no BLAS hook in this process (the CPU path runs).
inline GpuControl gpu_offload_set_mode(const std::string& mode) {
  using Fn = int (*)(const char*);
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hsolve_gpu_set_mode"));
  if (fn == nullptr) return GpuControl::NoHook;
  return static_cast<GpuControl>(fn(mode.c_str()));
}

/// The offload's state and totals as a JSON object ("" without a BLAS hook): mode, available
/// (null until a factorization needed the GPU), reason, device, init_s, factorizations and
/// factorizations_offloaded, calls, gpu_gflop, gpu_share (of the offloaded factorizations' MUMPS
/// estimates), call_s, predicted_cpu_s, h2d_gb, d2h_gb, errors (docs/BLAS.md §10).
inline std::string gpu_offload_report_json() {
  using Fn = const char* (*)();
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hsolve_gpu_report_json"));
  if (fn == nullptr) return {};
  const char* s = fn();
  return s != nullptr ? std::string(s) : std::string();
}

}  // namespace hs
