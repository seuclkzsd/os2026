// Module-internal entry point of the thick-restart Lanczos solver that also reports where the
// wall-clock time went (benchmarks: tests/cpp/test_lanczos_perf.cpp). Not part of the public API:
// hs::lanczos_krylov_schur is exactly this function with the timings discarded.
// Scalar set: T in {double, cplx} (instantiated in src/eigen/lanczos_krylov_schur.cpp).
#pragma once

#include "hsolve/eigen.hpp"

namespace hs::eigen_detail {

/// Seconds spent in each phase of one lanczos_krylov_schur call (phases do not overlap).
struct LanczosTimings {
  double op = 0.0;                 ///< operator applications Op * v_j
  double orthogonalization = 0.0;  ///< CGS2 against [D V] incl. M applications, normalization,
                                   ///< random replacement vectors after a breakdown
  double rayleigh_ritz = 0.0;      ///< heev of the projected matrix, ranking, residual estimates
  double restart = 0.0;            ///< basis rotation V <- V S and the new arrowhead matrix
  double total = 0.0;              ///< whole call (adds setup and the final Ritz vectors)
};

/// Same contract as hs::lanczos_krylov_schur; `timings` is overwritten. Not thread-safe with
/// respect to `op` / `b_inner` (they are applied), otherwise reentrant.
template <class T>
HermitianEigResult<T> lanczos_krylov_schur_timed(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                                 Which which, const KrylovSchurOptions<T>& options,
                                                 LanczosTimings& timings);

}  // namespace hs::eigen_detail
