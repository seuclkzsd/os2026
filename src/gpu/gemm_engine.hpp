// GPU module: dgemm offload for the large dense updates of MUMPS' factorizations (P10 "gpu";
// docs/DESIGN.md §3.9, docs/BLAS.md §10, results/p10_gpu.md). Scalar set: double (FP64 on both
// the CPU and the GPU: cublasDgemm is IEEE double arithmetic like any BLAS).
//
// One call C = alpha op(A) op(B) + beta C is split along one dimension of C - its columns when
// m <= n (every GPU tile then needs all of op(A), the smaller operand: the "shared" operand,
// uploaded once per call, and a column tile of op(B)), else its rows:
//   * GPU (the engine's pipeline threads): `pack_threads` issuing threads (one cuBLAS handle each)
//     claim tiles; per tile one packs the tile's operand into a pinned staging buffer
//     (non-temporal stores: the copy engine reads it, the CPU never does), copies it to the
//     device, computes T = alpha op(A) op(B) for the tile with cublasDgemm (beta = 0, so C is never
//     uploaded) and copies T back into a pinned buffer; the completing thread adds it into C on the
//     host: C_tile = beta C_tile + T (beta = 0 never reads C). `slots` pipeline slots (staging and
//     device buffers, a stream each) keep packing, transfers, GEMMs and additions of different
//     tiles overlapping.
//   * CPU: in the default GPU-only mode the calling thread waits (MUMPS has nothing else to do
//     above its L0 layer: it updates one front at a time). In hybrid mode (measurement option) it
//     computes chunks from the other end of the same dimension with the real BLAS over
//     `cpu_threads` threads, and both sides claim guided portions (a share of what is left) so that
//     they finish together.
// Measured on gpu02 (results/p10_gpu.md §2): the pipeline is bound by the host's memory bandwidth,
// not by PCIe - per staged byte the host reads the source, writes the staging buffer and the copy
// engine reads it again, and the additions read T and C and write C - so GPU-only beats hybrid on
// every large shape (the CPU share competes for the same bandwidth), and the time of a call is
// close to  t_gpu = (24 (m + n) k + 32 m n) bytes / host_gbs + latency_s.
//
// Results: every element of C is computed by one cuBLAS call over the full K (then added to
// beta C in double on the host); it differs from the CPU BLAS by rounding only. GPU-only calls
// are reproducible run to run (the tile sizes depend on the shape only); hybrid calls are not
// (which columns the GPU takes depends on timing).
//
// Eligibility (dgemm returns false and leaves C untouched otherwise; the caller computes the call
// itself): the engine is usable (not broken by an earlier CUDA error); 2 m n k >= min_flops; the
// model predicts t_cpu = 2 m n k / cpu_rate >= min_speedup * t_gpu, with cpu_rate = cpu_gflops
// (default: cpu_gflops_per_thread * omp_get_max_threads()); the shared operand fits the device
// buffer and a tile of kMinTile units fits a slot; no other thread is in a call (try-lock: a
// concurrent caller gets false and stays on the CPU).
// A CUDA error during a call marks the engine broken for the rest of the process; the tiles whose
// result had not been added yet are recomputed on the CPU, so the call still completes correctly.
//
// Thread safety: dgemm may be called from any thread (one call at a time; others get false);
// eligible(), stats() and the accessors from any thread. dgemm must not be called from inside an
// OpenMP parallel region (the caller checks; inside MUMPS' L0 subtrees every thread works on its
// own small fronts and the calls stay on the CPU).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace hs::gpu {

/// Fortran dgemm_ (LP64, gfortran hidden string lengths) - the real BLAS for the CPU work.
using CpuDgemm = void (*)(const char*, const char*, const int*, const int*, const int*,
                          const double*, const double*, const int*, const double*, const int*,
                          const double*, double*, const int*, std::size_t, std::size_t);

/// Smallest GPU tile (columns or rows of C) a call accepts.
constexpr int kMinTile = 32;

struct EngineOptions {
  int device = 0;                                ///< ordinal among the visible devices
  std::size_t mem_bytes = std::size_t{1} << 30;  ///< device memory allocated at init (all of it)
  std::size_t slot_bytes = std::size_t{16} << 20;  ///< staging per slot and direction
  int pack_threads = 2;  ///< issuing (packing) threads, 1..4
  int slots = 6;         ///< pipeline slots, pack_threads + 1 .. 8
  bool stream_stores = true;  ///< non-temporal stores when packing (x86; else memcpy)
  bool hybrid = false;        ///< the CPU computes a share of each call (measurement option)
  /// BLAS threads of the CPU share in hybrid mode; 0 = omp_get_max_threads() minus the pipeline
  /// threads (pack_threads + 1), at least 1.
  int cpu_threads = 0;
  // Offload model (defaults measured on gpu02: 16 cores of a Xeon Silver 4314 socket, one V100 on
  // PCIe 3 x16; results/p10_gpu.md §2).
  double min_flops = 1e9;               ///< 2 m n k of the smallest call taken
  double cpu_gflops = 0.0;              ///< CPU dgemm rate; 0 = per_thread * omp_get_max_threads()
  double cpu_gflops_per_thread = 40.0;  ///< large dgemm, 16 busy cores
  double host_gbs = 30.0;               ///< host memory traffic rate of the staged pipeline
  double latency_s = 1e-3;              ///< pipeline fill per call
  double min_speedup = 1.25;            ///< predicted t_cpu / t_gpu needed to offload a call
  /// Test hook: after this many tiles of the engine's lifetime, the next tile fails as if a CUDA
  /// call had failed (-1 = never). Exercises the error path: recomputation on the CPU, broken().
  std::int64_t inject_fault_after_tiles = -1;
};

/// Process-lifetime totals of the calls the engine took.
struct EngineStats {
  std::int64_t calls = 0;          ///< calls computed
  std::int64_t tiles = 0;          ///< GPU tiles completed
  double gpu_flops = 0.0;          ///< flops computed on the GPU
  double cpu_flops = 0.0;          ///< flops of the same calls computed on the CPU (hybrid / errors)
  double h2d_bytes = 0.0;
  double d2h_bytes = 0.0;
  double seconds = 0.0;            ///< wall time inside the calls taken
  double predicted_cpu_s = 0.0;    ///< the model's CPU time of the calls taken
  std::int64_t declined_busy = 0;  ///< calls refused because another thread was in a call
  std::int64_t errors = 0;         ///< CUDA errors (the first one breaks the engine)
  // Where the time goes (seconds summed over the calls; thread time of the pipeline threads):
  double shared_s = 0.0;     ///< packing and issuing the shared operand
  double pack_s = 0.0;       ///< packing tiles (all issuing threads)
  double slot_wait_s = 0.0;  ///< issuing threads waiting for a free slot
  double sync_s = 0.0;       ///< completing thread waiting for tiles
  double add_s = 0.0;        ///< host additions C = beta C + T
  double cpu_s = 0.0;        ///< the CPU share (hybrid mode, calling thread)
  double tail_s = 0.0;       ///< calling thread waiting for the GPU
};

/// Device and set-up description.
struct EngineInfo {
  std::string device_name;  ///< e.g. "Tesla V100-PCIE-16GB" (cuDeviceGetName; "" if unavailable)
  std::string pci_bus_id;
  int compute_major = 0, compute_minor = 0, multiprocessors = 0;
  int driver_version = 0, runtime_version = 0, cublas_version = 0;
  std::size_t device_total_bytes = 0, device_free_bytes = 0;  ///< at init, before allocating
  std::size_t pinned_bytes = 0;                              ///< page-locked host staging
  std::string cudart_path, cublas_path;
  double init_seconds = 0.0;  ///< library load + context + cuBLAS + buffers + threads
  EngineOptions options;      ///< as used (clamped)
};

class GemmEngine {
 public:
  /// Initializes the device (context, cuBLAS handles, streams, events, device and pinned buffers,
  /// pipeline threads). nullptr with `*why` when anything is missing or fails (no library, no
  /// visible device, too little free device memory, a CUDA error); nothing stays allocated then.
  static std::unique_ptr<GemmEngine> create(const EngineOptions& opt, std::string* why);
  ~GemmEngine();
  GemmEngine(const GemmEngine&) = delete;
  GemmEngine& operator=(const GemmEngine&) = delete;

  /// Whether the engine would take the shape (the eligibility rules above, except the busy and
  /// broken states). Arguments as dgemm's; no CUDA call.
  bool eligible(char ta, char tb, int m, int n, int k) const noexcept;

  /// The model's t_cpu / t_gpu for the shape (see above).
  double predicted_speedup(int m, int n, int k) const noexcept;

  /// C = alpha op(A) op(B) + beta C (Fortran BLAS semantics, column-major, 'N' / 'T' / 'C' in
  /// either case; valid arguments only - the caller checks them). True when computed (C final);
  /// false when not taken (C untouched). `cpu_dgemm` computes the CPU work (hybrid share, tiles
  /// lost to an error).
  bool dgemm(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda,
             const double* b, int ldb, double beta, double* c, int ldc, CpuDgemm cpu_dgemm);

  EngineStats stats() const;
  const EngineInfo& info() const noexcept;
  /// True after a CUDA error (no further call is taken).
  bool broken() const noexcept;
  /// The first CUDA error's text ("" if none).
  std::string error() const;

  struct Impl;

 private:
  explicit GemmEngine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace hs::gpu
