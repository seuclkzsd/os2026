// GPU module: CPU affinity of the threads the engine starts (P10 "gpu"; docs/DESIGN.md §3.9,
// docs/BLAS.md §10). Scalar set: none.
//
// A thread starts with the CPU affinity of the thread that creates it. With OMP_PROC_BIND (or
// OMP_PLACES) set, the OpenMP runtime binds its initial thread to one place - one core with
// OMP_PLACES=cores - and that is the thread that creates the GPU engine at the first factorization:
// the engine's pipeline threads (and the CUDA runtime's own threads, started with the context)
// would all share that one core (measured on gpu02: 1.3-1.4x longer offloaded calls,
// results/p10_gpu.md §2). The engine is therefore set up under a ScopedPlacesAffinity, which gives
// the creating thread the union of the OpenMP places - every CPU the run's OpenMP threads may use -
// and restores the thread's own binding afterwards. Without OpenMP binding (no places) nothing
// changes: the threads inherit the creating thread's (unbound) affinity as before.
//
// Thread safety: any thread; ScopedPlacesAffinity changes only the calling thread.
#pragma once

#include <sched.h>

namespace hs::gpu {

/// The union of the OpenMP places' processors in *out; false when the OpenMP runtime has no places
/// (no OMP_PROC_BIND / OMP_PLACES binding: its threads are not bound).
bool openmp_places_cpuset(cpu_set_t* out) noexcept;

/// Number of CPUs in the calling thread's affinity mask (0 if it cannot be read).
int thread_cpu_count() noexcept;

/// While alive, the calling thread may run on every CPU of the OpenMP places (threads it starts
/// inherit that); its previous affinity is restored at destruction. It only widens: no effect
/// without places, unless the thread's mask is a proper subset of the places' union (a thread bound
/// to its place), or when the affinity cannot be read or set.
class ScopedPlacesAffinity {
 public:
  ScopedPlacesAffinity() noexcept;
  ~ScopedPlacesAffinity();
  ScopedPlacesAffinity(const ScopedPlacesAffinity&) = delete;
  ScopedPlacesAffinity& operator=(const ScopedPlacesAffinity&) = delete;
  /// True when the thread's affinity was changed (and will be restored).
  bool widened() const noexcept { return active_; }

 private:
  cpu_set_t saved_;
  bool active_ = false;
};

}  // namespace hs::gpu
