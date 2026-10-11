// Thin RAII layer over the MUMPS C interface (module-internal to src/factor).
// hsolve drives only the real arithmetics: dmumps (U = double) and smumps (U = float). Complex
// Hermitian matrices are factorized through their real symmetric embedding (see factor.hpp),
// because complex MUMPS with SYM=2 is complex-symmetric, not Hermitian.
#pragma once

#include <dmumps_c.h>
#include <smumps_c.h>

#include <cstdint>
#include <memory>

#include "hsolve/error.hpp"

namespace hs::mumps {

/// Fortran communicator understood by the sequential (libmpiseq) MUMPS build.
inline constexpr int kUseCommWorld = -987654;

/// MUMPS JOB codes.
enum Job : int { kInit = -1, kTerminate = -2, kAnalyze = 1, kFactorize = 2, kSolve = 3 };

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

/// One MUMPS instance with SYM=2 (symmetric indefinite LDL^T) and PAR=1 (host works).
/// JOB=-1 runs at construction, JOB=-2 at destruction (which also deletes out-of-core files).
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
    s_->job = kTerminate;
    Api<U>::call(s_.get());
  }
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&&) = delete;
  Instance& operator=(Instance&&) = delete;

  int& icntl(int i) { return s_->icntl[i - 1]; }
  U& cntl(int i) { return s_->cntl[i - 1]; }
  int infog(int i) const { return s_->infog[i - 1]; }
  double rinfog(int i) const { return static_cast<double>(s_->rinfog[i - 1]); }
  Struc& raw() { return *s_; }

  /// Run one job; returns INFOG(1) (negative: error, positive: warning).
  int run(int job) {
    s_->job = job;
    Api<U>::call(s_.get());
    return s_->infog[0];
  }

 private:
  std::unique_ptr<Struc> s_;
};

}  // namespace hs::mumps
