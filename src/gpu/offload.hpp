// GPU module: the process-level GPU offload of the BLAS hook (P10 "gpu"; docs/DESIGN.md §3.10,
// docs/BLAS.md §10). Scalar set: double.
//
// Mode (process-wide; read from the environment variable HS_GPU at the first factorization unless
// set_mode() was called before; unset: the build's HSOLVE_GPU_DEFAULT, "off" unless configured
// otherwise; an invalid value aborts the process with a message, like an unknown HS_BLAS_ROUTE
// token):
//   off  - nothing CUDA is ever loaded: exactly the CPU path.
//   auto - a MUMPS factorization (JOB 2, including the certificate's inertia-only counts) whose
//          analysis estimate RINFOG(1) is at least auto_min_flops offloads its large dgemm calls
//          to the GPU, if one is usable; smaller problems stay on the CPU without touching CUDA.
//   on   - every factorization offloads (the per-call model still keeps unprofitable calls on the
//          CPU).
// The GPU is device 0 of the process' CUDA_VISIBLE_DEVICES. The engine (gemm_engine.hpp) is
// created at the first factorization that offloads; when that fails (no CUDA library, no device,
// too little free memory) the reason is kept, the process stays on the CPU for good and the run's
// results are those of the CPU path ("degrade mode"). A CUDA error later also returns the process
// to the CPU (the call in flight is completed on the CPU).
// Options: HS_GPU_OPTS, comma-separated key=value (parse_offload_options): device, mem_mb, slot_mb,
// chunk_kb (capped at slot_mb: normalized()), nt, pack, slots, hybrid, cpu_threads, min_gflop,
// cpu_gflops, host_gbs, min_speedup, auto_min_tflop, fault_after (test hook);
// unknown keys abort. HS_GPU_REPORT=<file> ("%p" = the process id) writes report_json() to the
// file at exit (for programs that cannot ask, e.g. a benchmark with the hook preloaded).
//
// Thread safety: every function may be called from any thread. The mode and the window flag are
// atomics; the engine is created once under a mutex; offload_dgemm defers to the engine's
// try-lock (one call at a time). MUMPS calls are serialized by hsolve's MUMPS lock, so at most one
// factorization window is open at a time.
#pragma once

#include <string>

#include "gpu/gemm_engine.hpp"

namespace hs::gpu {

enum class Mode { Off = 0, Auto = 1, On = 2 };
const char* mode_name(Mode m) noexcept;
/// "off" | "auto" | "on" -> true and *out; anything else -> false.
bool parse_mode(const std::string& s, Mode* out) noexcept;

struct OffloadOptions {
  EngineOptions engine;
  /// auto: smallest MUMPS estimate RINFOG(1) that offloads (gpu02: the offload pays from ~5e12, the
  /// 1e6 3-D cases; it is neutral at 1.75e12, syn-mem27-80, and loses below: results/p10_gpu.md)
  double auto_min_flops = 3e12;
};

/// Parses HS_GPU_OPTS-style text (see above) on top of the defaults; unknown or malformed tokens
/// are appended (comma-separated) to `*unknown`.
OffloadOptions parse_offload_options(const std::string& spec, std::string* unknown);

/// The current mode (resolving HS_GPU / the build default on first use).
Mode mode() noexcept;
/// Sets the mode (overrides HS_GPU); false for an invalid name. Takes effect at the next
/// factorization.
bool set_mode(const std::string& name) noexcept;

/// Called by the MUMPS entry-point wrapper around a factorization: decides whether it offloads
/// (mode, estimate `est_flops` = RINFOG(1), or < 0 when unknown) and creates the engine when
/// needed. end must follow begin.
void factorization_begin(double est_flops);
void factorization_end();

/// True inside a factorization that offloads (cheap: one atomic load).
bool window_open() noexcept;

/// Offloads one dgemm when the window is open, the engine takes the shape and the caller is not
/// inside an OpenMP parallel region; true when C was computed. `real` computes CPU work.
bool offload_dgemm(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta, double* c, int ldc, CpuDgemm real);

/// The offload's state and totals as one JSON object (mode, available, reason, device, init_s,
/// factorizations seen / offloaded, engine totals); see docs/BLAS.md §10 for the fields.
std::string report_json();

}  // namespace hs::gpu
