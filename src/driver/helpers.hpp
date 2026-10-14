// Helper processes of a multi-process solve as the driver sees them (DESIGN §3.9; driver-internal).
// src/parallel implements this interface over hs::Transport (include/hsolve/parallel.hpp) and runs
// the helpers' side; the driver never sees messages or MPI. Scalar set of the templates: T in
// {double, cplx}.
//
// The driver hands out two kinds of independent work:
//   * CountTask: the second Sylvester count of a nearest target's exact certificate, made by the
//     helper while the leader makes the first (definite.cpp, concurrent_counts);
//   * SliceTask: a spectrum slice of a window target (WindowSolver::run_slice on the helper's own
//     main object, analysed on the leader's pivot order).
// A task that fails on the helper (any exception there, incl. injected faults) comes back as a
// reply with ok == false; the driver then does the work itself and abandons that helper for the
// rest of the solve (ParallelReport::fallbacks).
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "driver/projection.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

struct Ctx;
struct SpectrumEstimate;

/// What a helper must know about the solve besides the matrices (the leader's working
/// configuration with its resolved memory budget, norms and scale, the main analysis' pivot order
/// and factor options, the spectrum estimate of one-sided targets).
struct ProblemInfo {
  double n2a = 0.0, n2b = 1.0, scale = 1.0;
  double sigma0 = 0.0;          ///< the main analysis' shift (representative values of the helpers')
  bool prepare_count = false;   ///< analyse the count instance at once (a nearest certificate will count)
  bool spectrum_valid = false;
  double lo1 = 0.0, lo2 = 0.0, hi1 = 0.0, hi2 = 0.0;
};

/// A helper's failure: the code and message of what it threw (Internal for non-hs exceptions).
struct HelperError {
  bool ok = true;
  Error::Code code = Error::Code::Internal;
  bool memory = false;  ///< OutOfMemoryBudget or std::bad_alloc
  std::string what;
};

/// count_below(s, outward, step, s_is_gap) (steps.cpp) on the helper's InertiaCounter (built from
/// its main object, which is analysed on the leader's pivot order: the same elimination as the
/// leader's counter up to MUMPS' rebuilt tree).
struct CountTask {
  double s = 0.0;
  double outward = 1.0;
  double step = 0.0;
  bool s_is_gap = true;
  double cap_mb = 0.0;     ///< memory the leader granted the count (MB; the helper's counter is capped
                           ///< at the leader's factorization cap, fixed when it was built)
  double budget_gb = 0.0;  ///< the helper's memory budget for the task (its SolverConfig's)
};

struct CountReply {
  HelperError error;
  index_t nu = 0;               ///< eigenvalues below the final s
  double s = 0.0;               ///< the final s (moved outward while K(s) was singular)
  index_t endpoint_moves = 0;
  bool detection_off = false;   ///< counted without null-pivot detection (dense cluster)
  double seconds = 0.0;         ///< the helper's time on the task
  double rss_bytes = 0.0;       ///< its resident memory after the task
  double count_mb = 0.0;        ///< its count's working memory (MUMPS INFOG(21))
};

/// A slice of a window target, as WindowSolver plans it (definite.cpp): its two count points,
/// a partial cluster's `want` and `focus`, and the zero band the run must avoid.
struct SliceTask {
  index_t id = 0;
  double lo_s = 0.0, hi_s = 0.0;
  index_t lo_nu = 0, hi_nu = 0;
  bool lo_off = false, lo_near = false, hi_off = false, hi_near = false;
  index_t want = -1;
  double focus = std::numeric_limits<double>::quiet_NaN();
  double band = -1.0, band_lo = 0.0, band_hi = 0.0;
  double cap_mb = 0.0;     ///< the helper's factorization cap (ICNTL(23)); 0: none
  double budget_gb = 0.0;  ///< the helper's memory budget for the task
};

template <class T>
struct SliceReply {
  HelperError error;
  /// false: a partial cluster that is not numerically equal (run_slice returned false): the
  /// leader solves the slice in full itself, as the single-process path does.
  bool completed = true;
  SliceReport report;
  std::vector<double> lambda;     ///< the slice's pairs (incl. those just outside its ends)
  DenseMatrix<T> x;               ///< their vectors (n x lambda.size())
  std::vector<PairMetrics> metrics;
  index_t restarts = 0, applies = 0;
  index_t block_size = 0;         ///< the Krylov width of its first run
  std::vector<std::string> warnings;
  double seconds = 0.0;
  double rss_bytes = 0.0;
};

/// The helpers of one solve. Not thread-safe (the leader's driver thread only).
class Helpers {
 public:
  virtual ~Helpers() = default;
  /// Helper processes of the session (ranks 1..size()).
  virtual int size() const = 0;
  /// Helper h (1..size()) has not failed in this solve.
  virtual bool usable(int h) const = 0;
  /// Stops using helper h for the rest of the solve (why: for the report's notes).
  virtual void abandon(int h, const std::string& why) = 0;
  /// Whether the problem was shared with the helpers (share()); before it no task can be posted.
  virtual bool shared() const = 0;
  /// Before the first task: sends the shared problem to the helpers if not done yet and waits
  /// until they are ready (copied it, built the pencil, analysed their count instance when the
  /// problem asked for one). Done here, when the leader is about to wait anyway, so that helpers
  /// never compute while the leader does (results/p10_mpi.md: preparing during the leader's main
  /// factorization slowed it more than it saved). A helper that fails to prepare is not usable.
  virtual void prepare(Ctx& c) = 0;
  /// Memory the helpers hold or may hold now, bytes: per helper its last reported resident
  /// memory, or the budget handed out with a task still running if that is larger. The driver
  /// adds it to the leader's own (memory_in_use), so the solve's budget covers the process tree.
  virtual double memory_bytes() const = 0;
  /// Estimated working memory of one count on a helper's inertia-only instance (MB, INFOG(16) of
  /// its analysis, reported when it prepared after share(); 0: not known yet).
  virtual double count_estimate_mb() const = 0;
  /// Threads of the helpers' factorizations (as reported; 0: not known).
  virtual int helper_threads() const = 0;

  /// Posts a count to idle helper h; the reply must be collected with wait_count(h).
  virtual void post_count(Ctx& c, int h, const CountTask& t) = 0;
  virtual CountReply wait_count(Ctx& c, int h) = 0;
};

template <class T>
class HelpersOf : public Helpers {
 public:
  /// Makes the problem available to the helpers: the matrices the solve factorizes (the pencil's A
  /// and B), the working configuration, `info` and the main analysis' pivot order and options. The
  /// objects must outlive the solve's helper work; they are sent at prepare() (the first task).
  /// Called again (a fallback from inexact factors: another main object) it replaces them.
  virtual void share(Ctx& c, const ShiftedPencil<T>& pencil, const SparseFactorization<T>& main,
                     const ProblemInfo& info) = 0;
  /// Posts slice t to idle helper h; collect with wait_slice().
  virtual void post_slice(Ctx& c, int h, const SliceTask& t) = 0;
  /// Waits for the reply of a busy helper (any); returns its index and the reply in `out`.
  virtual int wait_slice(Ctx& c, SliceReply<T>& out) = 0;
  /// Helpers currently working on a posted task.
  virtual int busy() const = 0;
};

// ------------------------------------------------------------------------------ leader side

/// hs::solve with helper processes (nullptr: none - hs::solve itself). solver.cpp; called by
/// hs::solve_parallel (src/parallel), whose HelpersOf talks to the helpers.
template <class T>
Solution<T> solve_with_helpers(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const SolverConfig& cfg, Logger* log,
                               HelpersOf<T>* helpers);

template <class T>
class CountSource;

/// The nearest certificate's two exact counts, nu(a) by the leader (count_below on `counts`) and
/// nu(b) by a helper process at the same time (CountTask; helpers.cpp). Returns false - nothing
/// counted, the caller counts both - when no helper can take it: none usable, the problem not
/// shared, no count estimate, or two counts at once beyond the memory the budget leaves for counts.
/// A helper whose count fails is abandoned and the leader counts nu(b) itself (true is returned).
/// a, b: moved outward like count_below's s. Throws what count_below throws (after collecting the
/// helper's reply, so that the session stays in step).
template <class T>
bool concurrent_counts(Ctx& c, CountSource<T>& counts, double& a, double& b, double step, index_t& nu_a,
                       index_t& nu_b);

// ------------------------------------------------------------------------------ helper side

/// The helper's side of a CountTask: count_below on `counter` (an InertiaCounter on the helper's
/// main object, analysed on the leader's pivot order) with the same rules as the leader's
/// (steps.cpp). Throws as count_below.
template <class T>
CountReply count_on_helper(Ctx& c, InertiaCounter<T>& counter, const CountTask& t);

/// The helper's side of a SliceTask: WindowSolver::run_slice (definite.cpp) on `fac` (analysed on
/// the leader's pivot order; the helper's main object, which keeps the factorization of its last
/// shift between tasks: Ctx::held). Throws what run_slice throws (incl. CountFailure).
template <class T>
SliceReply<T> slice_on_helper(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac,
                              const ProblemInfo& info, const SliceTask& t);

}  // namespace hs::driver
