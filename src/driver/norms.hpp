// Spectral-norm and spectrum-end estimation for Hermitian sparse matrices and pencils
// (driver-internal).
#pragma once

#include <cstdint>
#include <functional>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

struct NormEstimate {
  double norm2 = 0.0;      ///< max |Ritz value|: a LOWER bound of ||A||_2 (conservative eta)
  double ritz_min = 0.0;   ///< smallest Ritz value (estimate of lambda_min, from inside)
  double ritz_max = 0.0;   ///< largest Ritz value (estimate of lambda_max, from inside)
  double ritz_min2 = 0.0;  ///< second smallest Ritz value (== ritz_min after 1 step)
  double ritz_max2 = 0.0;  ///< second largest Ritz value (== ritz_max after 1 step)
  int steps = 0;           ///< Lanczos steps actually taken (fewer on breakdown)
};

/// Plain three-term Lanczos (no reorthogonalization, O(n) memory) on a Hermitian matrix.
/// Without reorthogonalization the Ritz values may contain ghost copies, but they still lie in
/// [lambda_min, lambda_max] up to O(eps ||A||) (Paige), so max |Ritz| is a lower bound of
/// ||A||_2 that is typically within a few percent after 30 steps. Cost: steps SpMVs.
template <class T>
NormEstimate estimate_norm2(const CsrMatrix<T>& a, int steps = 30, std::uint64_t seed = 2026);

/// In-place solve with B: x <- B^{-1} x for one vector.
template <class T>
using SolveFn = std::function<void(T* x)>;

/// The same Lanczos for the pencil (A, B), B Hermitian positive definite: Op = B^{-1} A is
/// self-adjoint in the B inner product, its Ritz values lie in [lambda_min, lambda_max] of the
/// pencil. Each step costs one A product, one B product and one solve_b. norm2 is max |Ritz|.
template <class T>
NormEstimate estimate_pencil_spectrum(const CsrMatrix<T>& a, const CsrMatrix<T>& b, const SolveFn<T>& solve_b,
                                      int steps = 30, std::uint64_t seed = 2026);

}  // namespace hs::driver
