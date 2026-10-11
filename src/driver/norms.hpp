// Spectral-norm estimation for Hermitian sparse matrices (driver-internal).
#pragma once

#include <cstdint>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

struct NormEstimate {
  double norm2 = 0.0;      ///< max |Ritz value|: a LOWER bound of ||A||_2 (conservative eta)
  double ritz_min = 0.0;   ///< smallest Ritz value (estimate of lambda_min, from inside)
  double ritz_max = 0.0;   ///< largest Ritz value (estimate of lambda_max, from inside)
  int steps = 0;           ///< Lanczos steps actually taken (fewer on breakdown)
};

/// Plain three-term Lanczos (no reorthogonalization, O(n) memory) on a Hermitian matrix.
/// Without reorthogonalization the Ritz values may contain ghost copies, but they still lie in
/// [lambda_min, lambda_max] up to O(eps ||A||) (Paige), so max |Ritz| is a lower bound of
/// ||A||_2 that is typically within a few percent after 30 steps. Cost: steps SpMVs.
template <class T>
NormEstimate estimate_norm2(const CsrMatrix<T>& a, int steps = 30, std::uint64_t seed = 2026);

}  // namespace hs::driver
