// Instrumented entry point of the general Krylov-Schur solver (module-internal, src/eigen).
// hs::krylov_schur_general is this function with times == nullptr; benchmarks and the driver's
// profiling use it to split the run time by phase. Scalar set: T in {double, cplx}.
#pragma once

#include "hsolve/eigen.hpp"
#include "hsolve/linear_operator.hpp"

namespace hs::eigen_detail {

/// Wall-clock seconds per phase, accumulated over the whole call.
struct GeneralPhaseTimes {
  double apply_s = 0.0;          ///< operator applications
  double orthogonalize_s = 0.0;  ///< output scaling, CGS2 + DGKS checks, breakdown vectors,
                                 ///< normalization
  double projected_s = 0.0;      ///< dense m x m work: Schur form, ranking, Ritz vectors, trsen
  double update_s = 0.0;         ///< n x m products: basis compression and final vectors
  double total_s = 0.0;
};

/// Same contract as hs::krylov_schur_general; when times != nullptr the phase timings of this
/// call are ADDED to *times (a handful of clock reads per Arnoldi step).
template <class T>
GeneralEigResult<T> krylov_schur_general_timed(LinearOperator<T>& op, Which which,
                                               const KrylovSchurOptions<T>& options,
                                               GeneralPhaseTimes* times);

}  // namespace hs::eigen_detail
