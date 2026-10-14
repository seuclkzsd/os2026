// Thin RAII layer over the MUMPS C interface (module-internal to src/factor).
// hsolve drives only the real arithmetics: dmumps (U = double) and smumps (U = float). Complex
// Hermitian matrices are factorized through their real symmetric embedding (see factor.hpp),
// because complex MUMPS with SYM=2 is complex-symmetric, not Hermitian.
#pragma once

#include <dmumps_c.h>
#include <pthread.h>
#include <smumps_c.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hsolve/error.hpp"

namespace hs::mumps {

/// Serializes every MUMPS call of the process (all arithmetics, all instances).
/// Sequential MUMPS 5.8.2 is NOT thread-safe across distinct instances: every factorization runs
/// MUMPS_LOAD_INIT/END (process-global Fortran module arrays of mumps_common, allocated and freed
/// per factorization; gdb: once per JOB=2, plus ~800 MUMPS_LOAD_UPDATE calls on a 100x100 grid),
/// allocates the module communication buffers BUF_CB/BUF_SMALL and sets the module pointer
/// DMUMPS_TMP_PTR. Two instances factorizing at the same time from two std::threads aborted with
/// "PB allocation in MUMPS_LOAD_INIT" + SIGSEGV in the first concurrent round, 7 of 7 runs
/// (bench/certpar_inertia.cpp --exp concurrent --mumps-lock 0). With this lock, objects used
/// from different threads are safe; their MUMPS phases simply run one after the other.
/// The lock is held for the duration of one MUMPS call (JOB = -1, -2, -4, 1, 2, 3).
///
/// Why not std::mutex: libstdc++ (glibc < 2.34) makes std::mutex::lock() a no-op unless
/// __gthread_active_p() sees a weak reference to __pthread_key_create resolved in the
/// EXECUTABLE. With the conda toolchain's -Wl,--as-needed, a link that gets pthread_create from
/// elsewhere (the ASan debug preset: libasan intercepts it) drops libpthread, the weak reference
/// resolves to 0 at link time, and the lock silently does nothing even though libpthread is loaded
/// at run time (libgomp and libasan need it): build/debug/test_factor_counter then crashed in
/// MUMPS_LOAD_INIT in 3 of 3 runs. pthread_mutex_lock() called directly has no such check: it binds
/// to libc's forwarder (or libpthread itself), which uses libpthread whenever it is loaded - and
/// it is, in any process with a second thread. bench/certpar_fix_probes.cpp --exp lock checks it.
class CallMutex {
 public:
  constexpr CallMutex() noexcept = default;
  CallMutex(const CallMutex&) = delete;
  CallMutex& operator=(const CallMutex&) = delete;

  void lock() {
    const int rc = pthread_mutex_lock(&m_);
    HS_ASSERT(rc == 0, "pthread_mutex_lock on the MUMPS call lock failed: " << rc);
  }
  void unlock() noexcept { pthread_mutex_unlock(&m_); }

 private:
  pthread_mutex_t m_ = PTHREAD_MUTEX_INITIALIZER;
};

/// The process-wide MUMPS call lock (BasicLockable; constant-initialised, never destroyed early).
inline CallMutex& call_mutex() {
  static CallMutex m;
  return m;
}

/// Block sizes a MumpsFactorization gives MUMPS (P9 "factspeed", results/p9_factspeed.md):
///   Mumps - MUMPS' own;
///   P9    - kFullRankPanelWidth, kCbUpdateBlock and, for a MUMPS patched with
///           patches/mumps-5.8.2-l0-blocking.patch, kL0PanelWidth / kL0Block below. Every
///           factorization of the 1e6 3-D cases 1.04-1.06x faster, end to end 1.03-1.04x with the
///           environment's MUMPS (below the 1.05x a default needs); with the patched MUMPS a count
///           1.07-1.08x faster on a fixed pivot order, end to end syn-mem27-100 ~1.09-1.12x but
///           yee-pec-box-1e6 only at the 1.05x bar (results/p9_factspeed.md §4.4-4.5).
/// The build's default (default_blocking()): P9 when hsolve links a MUMPS of tools/dev/build_mumps.sh
/// (CMake HSOLVE_MUMPS_ROOT; HSOLVE_MUMPS_BLOCKING overrides), else Mumps. Each object reads the
/// environment variable HS_MUMPS_BLOCKING at construction: unset or "" -> the build's default,
/// "mumps" -> Mumps, "p9" -> P9, anything else -> InvalidInput. Only the blocking changes: the
/// same threshold-pivoted LDL^T, the same counts (test_factor_panel).
enum class Blocking { Mumps, P9 };
Blocking default_blocking();
Blocking blocking_from_environment();

/// MUMPS' full-rank outer panel width of LDL^T / LU fronts: KEEP(420), the block NBLR_ORIG of
/// [sd]fac_front_LDLT_type1.F (MUMPS sets 4 * KEEP(6) = 128 at JOB=-1; the analysis sizes its
/// buffers with it, so it is set before the analysis). A front's fully summed columns are
/// factorized in panels of this width, each followed by a rank-KEEP(420) dgemm update of the rest
/// of the front; at 128 that update is memory-bound on 16 cores (~350 GFlop/s against 585-620 at
/// larger ranks, ~500 at 256, gpu01 slot).
inline constexpr int kFullRankPanelWidth = 256;

/// Row block of the contribution-block update at the end of a front's LDL^T (KEEP(58), used only
/// by [sd]fac_front_aux.F DMUMPS_FAC_T_LDLT: blocks of KEEP(58) rows once the contribution block
/// has more than KEEP(57) = 2 * KEEP(58) rows, else two halves; MUMPS: 1000 / 2000). Fewer, larger
/// dgemm calls on the big fronts; the same arithmetic (P9 "factspeed").
inline constexpr int kCbUpdateBlock = 2000;

/// Block sizes under MUMPS' L0-OpenMP layer, read only by a MUMPS built with
/// patches/mumps-5.8.2-l0-blocking.patch (tools/dev/build_mumps.sh --patch; CMake
/// HSOLVE_MUMPS_ROOT): KEEP(498) the full-rank panel width there (its small fronts prefer MUMPS'
/// 128 to kFullRankPanelWidth), KEEP(499) the triangle stripes / contribution-block diagonal blocks
/// there and hence the band above the diagonal zeroed at assembly (64 instead of 150 / 120 / 250:
/// less memset on small fronts). MUMPS 5.8.2 itself does not use these two entries (0 after JOB=-1),
/// so with the environment's library they change nothing.
inline constexpr int kL0PanelWidth = 128;
inline constexpr int kL0Block = 64;

/// Fortran communicator understood by the sequential (libmpiseq) MUMPS build.
inline constexpr int kUseCommWorld = -987654;

/// MUMPS JOB codes.
/// kFreeFactors (JOB=-4) frees the factorization but keeps the analysis (verified on 5.8.2: RSS
/// drops by the factor storage, a later JOB=2 and JOB=3 work).
enum Job : int {
  kInit = -1,
  kTerminate = -2,
  kFreeFactors = -4,
  kAnalyze = 1,
  kFactorize = 2,
  kSolve = 3
};

template <class U>
struct Api;
template <>
struct Api<double> {
  using Struc = DMUMPS_STRUC_C;
  static void call(Struc* s) { dmumps_c(s); }
};
template <>
struct Api<float> {
  using Struc = SMUMPS_STRUC_C;
  static void call(Struc* s) { smumps_c(s); }
};

/// MUMPS reports integer statistics that overflow 32 bits as negative numbers in millions
/// ("-x means x * 10^6"); this returns the decoded value.
inline std::int64_t decode_count(int v) {
  return v >= 0 ? static_cast<std::int64_t>(v) : -static_cast<std::int64_t>(v) * 1000000;
}

/// INFO(1) codes caused by an under-estimated internal workspace; a larger ICNTL(14) cures
/// them without re-analysis (-8/-9 main integer/real arrays, -14/-15 integer arrays,
/// -17/-20 send/receive buffers).
inline bool is_workspace_error(int info1) {
  switch (info1) {
    case -8:
    case -9:
    case -14:
    case -15:
    case -17:
    case -20:
      return true;
    default:
      return false;
  }
}

/// INFO(1) codes meaning "not enough memory": allocation failures during analysis (-5 real,
/// -7 integer workspace), allocation failure during factorization/solve (-13) and the
/// ICNTL(23) working-memory cap being too small (-19).
inline bool is_memory_error(int info1) {
  return info1 == -5 || info1 == -7 || info1 == -13 || info1 == -19;
}

/// Extra MUMPS controls for measurements (bench/factspeed_*.cpp): (1-based index, value) pairs
/// written after hsolve's own settings, before every analysis (JOB=1) and factorization (JOB=2)
/// of a MumpsFactorization. Not used by the library itself.
struct MumpsControlOverrides {
  std::vector<std::pair<int, int>> icntl;
  std::vector<std::pair<int, int>> keep;
  std::vector<std::pair<int, double>> cntl;
  bool empty() const { return icntl.empty() && keep.empty() && cntl.empty(); }
};

/// One MUMPS instance with SYM=2 (symmetric indefinite LDL^T) and PAR=1 (host works).
/// JOB=-1 runs at construction, JOB=-2 at destruction (which also deletes out-of-core files);
/// every job holds call_mutex().
/// The control structure is heap-allocated and never moves; the wrapper is neither copyable
/// nor movable. Accessors take the 1-based indices of the MUMPS user guide: icntl(14) is
/// ICNTL(14).
template <class U>
class Instance {
 public:
  using Struc = typename Api<U>::Struc;

  Instance() : s_(std::make_unique<Struc>()) {  // value-initialised: all fields zero
    s_->sym = 2;
    s_->par = 1;
    s_->comm_fortran = kUseCommWorld;
    if (run(kInit) < 0)
      HS_THROW(FactorizationFailed, "MUMPS initialization (JOB=-1) failed: INFOG(1)="
                                        << infog(1) << ", INFOG(2)=" << infog(2));
  }
  ~Instance() {
    std::lock_guard<CallMutex> lock(call_mutex());
    s_->job = kTerminate;
    Api<U>::call(s_.get());
  }
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&&) = delete;
  Instance& operator=(Instance&&) = delete;

  int& icntl(int i) { return s_->icntl[i - 1]; }
  U& cntl(int i) { return s_->cntl[i - 1]; }
  /// MUMPS' internal KEEP array (exposed by [sd]mumps_c.h; set by JOB=-1, read at analysis and
  /// factorization). Written only for the block sizes above (Blocking) and by measurement code
  /// (MumpsControlOverrides).
  int& keep(int i) { return s_->keep[i - 1]; }
  int keep(int i) const { return s_->keep[i - 1]; }
  int infog(int i) const { return s_->infog[i - 1]; }
  double rinfog(int i) const { return static_cast<double>(s_->rinfog[i - 1]); }
  /// MUMPS' internal real statistics (read only; e.g. DKEEP(16): elimination flops of the last
  /// factorization under the L0-OpenMP layer, 0 without one - dfac_b.F).
  double dkeep(int i) const { return static_cast<double>(s_->dkeep[i - 1]); }
  Struc& raw() { return *s_; }

  /// Run one job under call_mutex(); returns INFOG(1) (negative: error, positive: warning).
  int run(int job) {
    std::lock_guard<CallMutex> lock(call_mutex());
    s_->job = job;
    Api<U>::call(s_.get());
    return s_->infog[0];
  }

 private:
  std::unique_ptr<Struc> s_;
};

}  // namespace hs::mumps
