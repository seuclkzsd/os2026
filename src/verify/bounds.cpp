// Failure-probability bound of the randomized exclusion check (include/hsolve/verify.hpp, "(1)"),
// evaluated in logarithms: T_k(x) overflows double for k acosh(x) > 709, which happens at the
// step counts used for delta = 1e-12 with wide gaps.
#include <algorithm>
#include <cmath>
#include <limits>

#include "hsolve/error.hpp"
#include "hsolve/verify.hpp"

namespace hs {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// log T_k(x) for x >= 1: T_k(x) = cosh(k a), a = acosh(x).
double log_chebyshev(index_t k, double x) {
  const double ka = static_cast<double>(k) * std::acosh(x);
  return ka + std::log1p(std::exp(-2.0 * ka)) - std::log(2.0);
}

/// log(c / (e T^2 + c)) for log T = lt, e > 0, c > 0, without overflow.
double log_ratio(double c, double e, double lt) {
  // log(e T^2 + c) as a log-sum-exp of log(e T^2) and log(c).
  const double a = std::log(e) + 2.0 * lt, b = std::log(c);
  const double hi = std::max(a, b), lo = std::min(a, b);
  return b - (hi + std::log1p(std::exp(lo - hi)));
}

/// log t*(eps, s): the smaller of the direct and squared Chebyshev thresholds (verify.hpp).
double log_threshold(double eps, index_t s) {
  // Direct: Chebyshev of degree s - 1 on [-Theta, (1 - eps) Theta], evaluated at Theta.
  const double x1 = (2.0 + eps) / (2.0 - eps);
  double lt = log_ratio(2.0 - eps, eps, log_chebyshev(s - 1, x1));
  if (s >= 2) {
    // Squared: Chebyshev of degree k = floor((s - 2) / 2) in theta^2 on [0, (1 - eps)^2 Theta^2].
    const double q = (1.0 - eps) * (1.0 - eps);
    if (!(q > 0.0)) return -kInf;  // eps == 1: tau = 0, nothing can pass anyway
    const double e2 = eps * (2.0 - eps);
    const double x2 = (2.0 - q) / q;  // (1 + e2) / (1 - e2)
    lt = std::min(lt, log_ratio(q, e2, log_chebyshev((s - 2) / 2, x2)));
  }
  return lt;
}

}  // namespace

double exclusion_failure_bound(double epsilon, index_t block_steps, index_t dimension,
                               index_t block_size, double kappa, bool complex_scalars) {
  if (block_steps < 1 || block_size < 1 || dimension < 0 || !(kappa >= 1.0) || !std::isfinite(kappa))
    HS_THROW(InvalidInput, "exclusion_failure_bound: need block_steps >= 1, block_size >= 1, dimension >= 0, "
                           "finite kappa >= 1 (got "
                               << block_steps << ", " << block_size << ", " << dimension << ", " << kappa << ")");
  if (dimension == 0) return 0.0;  // nothing outside span X
  if (!(epsilon > 0.0)) return 1.0;
  const double eps = std::min(epsilon, 1.0);
  const double lt = log_threshold(eps, block_steps) + std::log(kappa);
  // One column: P(beta^2 < kappa t*) <= sqrt(N kappa t*) (real) or (N - 1) kappa t* (complex).
  double lf = 0.0;
  if (complex_scalars) {
    if (dimension == 1) return 0.0;  // |z1|^2 == 1 on the unit sphere of C^1
    lf = std::log(static_cast<double>(dimension - 1)) + lt;
  } else {
    lf = 0.5 * (std::log(static_cast<double>(dimension)) + lt);
  }
  lf = std::min(lf, 0.0);
  return std::exp(static_cast<double>(block_size) * lf);
}

index_t exclusion_block_steps(double epsilon, double delta, index_t dimension, index_t block_size,
                              double kappa, bool complex_scalars) {
  if (!(epsilon > 0.0) || !(delta > 0.0) || !(delta < 1.0)) return -1;
  if (block_size < 1) HS_THROW(InvalidInput, "exclusion_block_steps: block_size must be >= 1");
  // A Krylov space of ceil(N / p) blocks covers the complement (exact, delta = 0); one more block
  // keeps the solver's layout feasible.
  const index_t cover = (dimension + block_size - 1) / block_size + 1;
  for (index_t s = 1; s < cover; ++s)
    if (exclusion_failure_bound(epsilon, s, dimension, block_size, kappa, complex_scalars) <= delta) return s;
  return std::max<index_t>(cover, 1);
}

}  // namespace hs
