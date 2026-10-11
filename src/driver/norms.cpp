#include "driver/norms.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

template <class T>
NormEstimate estimate_norm2(const CsrMatrix<T>& a, int steps, std::uint64_t seed) {
  NormEstimate est;
  const index_t n = a.nrows;
  if (n == 0 || a.nnz() == 0) return est;
  steps = static_cast<int>(std::min<index_t>(steps, n));
  std::vector<T> v(static_cast<std::size_t>(n)), v_prev(static_cast<std::size_t>(n), T(0)),
      w(static_cast<std::size_t>(n));
  linalg::fill_random(v.data(), n, seed, /*stream=*/0x6e6f726dULL);
  const double nv = linalg::norm2(n, v.data());
  linalg::scal(n, T(1.0 / nv), v.data());

  std::vector<double> alpha, beta;  // tridiagonal: diag alpha, off-diag beta
  double beta_prev = 0.0;
  for (int j = 0; j < steps; ++j) {
    spmv(a, v.data(), w.data());
    // w -= beta_{j-1} v_{j-1};  alpha_j = Re(v^H w) (Hermitian => real);  w -= alpha_j v
    if (j > 0) linalg::axpy(n, T(-beta_prev), v_prev.data(), w.data());
    const double aj = std::real(linalg::dot(n, v.data(), w.data()));
    linalg::axpy(n, T(-aj), v.data(), w.data());
    alpha.push_back(aj);
    const double bj = linalg::norm2(n, w.data());
    // Breakdown: an invariant subspace was found; its Ritz values are exact eigenvalues.
    const double scale = std::max(std::abs(aj), beta_prev);
    if (bj <= 1e-13 * std::max(scale, 1e-300) || j + 1 == steps) break;
    beta.push_back(bj);
    std::swap(v_prev, v);
    for (index_t i = 0; i < n; ++i) v[i] = w[i] / bj;
    beta_prev = bj;
  }
  // Eigenvalues of the m x m symmetric tridiagonal matrix (dense heev; m <= 30).
  const auto m = static_cast<lapack::fint>(alpha.size());
  std::vector<double> t(static_cast<std::size_t>(m) * m, 0.0), theta(static_cast<std::size_t>(m));
  for (lapack::fint i = 0; i < m; ++i) {
    t[i + i * m] = alpha[i];
    if (i + 1 < m) t[(i + 1) + i * m] = t[i + (i + 1) * m] = beta[i];
  }
  lapack::heev('N', 'U', m, t.data(), m, theta.data());
  est.steps = static_cast<int>(m);
  est.ritz_min = theta.front();
  est.ritz_max = theta.back();
  est.norm2 = std::max(std::abs(theta.front()), std::abs(theta.back()));
  return est;
}

#define HS_INSTANTIATE_NORMS(T) template NormEstimate estimate_norm2<T>(const CsrMatrix<T>&, int, std::uint64_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_NORMS)
#undef HS_INSTANTIATE_NORMS

}  // namespace hs::driver
