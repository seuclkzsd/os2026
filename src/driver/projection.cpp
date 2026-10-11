#include "driver/projection.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

namespace {

using lapack::fint;

// In-place Hermitian part of a small square matrix: (C + C^H)/2 (diagonal made real).
template <class T>
void make_hermitian(DenseMatrix<T>& c) {
  const index_t m = c.rows();
  for (index_t j = 0; j < m; ++j) {
    c(j, j) = T(std::real(c(j, j)));
    for (index_t i = j + 1; i < m; ++i) {
      const T h = 0.5 * (c(i, j) + conj_if(c(j, i)));
      c(i, j) = h;
      c(j, i) = conj_if(h);
    }
  }
}

template <class T>
void replace_by_product(DenseMatrix<T>& x, const DenseMatrix<T>& y) {
  DenseMatrix<T> tmp(x.rows(), y.cols());
  linalg::multiply(x.rows(), x.cols(), y.cols(), x.data(), x.ld(), y.data(), y.ld(), tmp.data(), tmp.ld());
  x = std::move(tmp);
}

}  // namespace

template <class T>
void apply_block(const CsrMatrix<T>* m, const DenseMatrix<T>& x, DenseMatrix<T>& y) {
  if (y.rows() != x.rows() || y.cols() != x.cols()) y.resize(x.rows(), x.cols());
  if (!m) {
    std::copy(x.data(), x.data() + x.rows() * x.cols(), y.data());
    return;
  }
  if (x.cols() == 0) return;
  spmm(*m, x.data(), x.ld(), y.data(), y.ld(), x.cols());
}

template <class T, class L>
PairMetrics pair_metrics(const T* ax, const T* bx, const T* x, index_t n, L lambda, double norm2_a,
                         double norm2_b) {
  double r2 = 0.0, b2 = 0.0, x2 = 0.0;
  for (index_t i = 0; i < n; ++i) {
    r2 += std::norm(ax[i] - lambda * bx[i]);
    b2 += std::norm(bx[i]);
    x2 += std::norm(x[i]);
  }
  PairMetrics pm;
  pm.abs_residual = std::sqrt(r2);
  const double lam = std::abs(lambda), bnorm = std::sqrt(b2), xnorm = std::sqrt(x2);
  pm.rel_residual = pm.abs_residual / std::max((lam > 0 ? lam : 1.0) * bnorm, 1e-300);
  pm.backward_error = pm.abs_residual / std::max((norm2_a + lam * norm2_b) * xnorm, 1e-300);
  return pm;
}

template <class T>
index_t orthonormalize_columns(DenseMatrix<T>& w, double drop_tol) {
  const index_t n = w.rows();
  index_t kept = 0;
  for (index_t j = 0; j < w.cols(); ++j) {
    T* col = w.col(j);
    const double before = linalg::norm2(n, col);
    if (before == 0.0) continue;
    const double after =
        linalg::orthogonalize_cgs2<T>(n, kept, w.data(), w.ld(), nullptr, 0, col, nullptr, nullptr);
    if (after <= drop_tol * before) continue;
    linalg::scal(n, T(1.0 / after), col);
    if (kept != j) std::copy(col, col + n, w.col(kept));
    ++kept;
  }
  w.resize_cols(kept);
  return kept;
}

template <class T>
index_t rayleigh_ritz_definite(DenseMatrix<T>& x, DenseMatrix<T>& ax, DenseMatrix<T>& bx,
                                  std::vector<double>& lambda, double drop_tol) {
  const index_t n = x.rows(), m = x.cols();
  if (m == 0) {
    lambda.clear();
    return 0;
  }
  DenseMatrix<T> g(m, m), h(m, m);
  linalg::gram(n, m, x.data(), x.ld(), ax.data(), ax.ld(), g.data(), g.ld());
  linalg::gram(n, m, x.data(), x.ld(), bx.data(), bx.ld(), h.data(), h.ld());
  make_hermitian(g);
  make_hermitian(h);
  const auto fm = static_cast<fint>(m);
  std::vector<double> mu(static_cast<std::size_t>(m));
  lapack::heev('V', 'U', fm, h.data(), fm, mu.data());  // h <- V
  std::vector<index_t> keep;
  for (index_t q = 0; q < m; ++q)
    if (mu[q] > drop_tol * mu.back() && mu[q] > 0.0) keep.push_back(q);
  const auto r = static_cast<index_t>(keep.size());
  // Y = V_r diag(mu_r)^{-1/2}: X Y is B-orthonormal.
  DenseMatrix<T> y(m, r);
  for (index_t q = 0; q < r; ++q)
    for (index_t p = 0; p < m; ++p) y(p, q) = h(p, keep[q]) / std::sqrt(mu[keep[q]]);
  // Projected A on that basis: Y^H G Y (r x r), then its eigenvectors Z; coefficients C = Y Z.
  DenseMatrix<T> gy(m, r), gr(r, r);
  linalg::multiply(m, m, r, g.data(), g.ld(), y.data(), y.ld(), gy.data(), gy.ld());
  linalg::gram(m, r, y.data(), y.ld(), gy.data(), gy.ld(), gr.data(), gr.ld());
  make_hermitian(gr);
  std::vector<double> theta(static_cast<std::size_t>(r));
  lapack::heev('V', 'U', static_cast<fint>(r), gr.data(), static_cast<fint>(r), theta.data());
  DenseMatrix<T> coef(m, r);
  linalg::multiply(m, r, r, y.data(), y.ld(), gr.data(), gr.ld(), coef.data(), coef.ld());
  replace_by_product(x, coef);
  replace_by_product(ax, coef);
  replace_by_product(bx, coef);
  lambda = std::move(theta);
  return r;
}

template <class T>
bool rayleigh_ritz_indefinite(const DenseMatrix<T>& w, const DenseMatrix<T>& aw, const DenseMatrix<T>& bw,
                              std::vector<cplx>& lambda, DenseMatrix<cplx>& x) {
  const index_t n = w.rows(), m = w.cols();
  lambda.clear();
  x.resize(n, m);
  if (m == 0) return true;
  DenseMatrix<T> g(m, m), h(m, m);
  linalg::gram(n, m, w.data(), w.ld(), aw.data(), aw.ld(), g.data(), g.ld());
  linalg::gram(n, m, w.data(), w.ld(), bw.data(), bw.ld(), h.data(), h.ld());
  make_hermitian(g);
  make_hermitian(h);
  // M = H^{-1} G (H Hermitian indefinite: LU with partial pivoting).
  const auto fm = static_cast<fint>(m);
  std::vector<fint> ipiv(static_cast<std::size_t>(m));
  if (lapack::getrf(fm, h.data(), fm, ipiv.data()) != 0) return false;
  double umax = 0.0, umin = INFINITY;
  for (index_t i = 0; i < m; ++i) {
    umax = std::max(umax, std::abs(h(i, i)));
    umin = std::min(umin, std::abs(h(i, i)));
  }
  if (!(umin > 1e-12 * umax)) return false;
  lapack::getrs(fm, fm, h.data(), fm, ipiv.data(), g.data(), fm);
  // Complex Schur form of M (small, so complex arithmetic avoids 2x2 real blocks) and its
  // eigenvectors Y.
  DenseMatrix<cplx> mc(m, m), q(m, m);
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < m; ++i) mc(i, j) = cplx(g(i, j));
  std::vector<cplx> mu(static_cast<std::size_t>(m));
  lapack::gees(fm, mc.data(), fm, mu.data(), q.data(), fm);
  lapack::trevc_backtransform(fm, mc.data(), fm, q.data(), fm);  // q <- eigenvectors of M
  // X = W * Y (complex), columns normalised to unit 2-norm.
  if constexpr (is_complex_v<T>) {
    linalg::multiply(n, m, m, w.data(), w.ld(), q.data(), q.ld(), x.data(), x.ld());
  } else {
    DenseMatrix<double> yr(m, m), yi(m, m), xr(n, m), xi(n, m);
    for (index_t j = 0; j < m; ++j)
      for (index_t i = 0; i < m; ++i) {
        yr(i, j) = q(i, j).real();
        yi(i, j) = q(i, j).imag();
      }
    linalg::multiply(n, m, m, w.data(), w.ld(), yr.data(), yr.ld(), xr.data(), xr.ld());
    linalg::multiply(n, m, m, w.data(), w.ld(), yi.data(), yi.ld(), xi.data(), xi.ld());
    for (index_t j = 0; j < m; ++j)
      for (index_t i = 0; i < n; ++i) x(i, j) = cplx(xr(i, j), xi(i, j));
  }
  for (index_t j = 0; j < m; ++j) {
    const double nrm = linalg::norm2(n, x.col(j));
    if (nrm > 0) linalg::scal(n, cplx(1.0 / nrm), x.col(j));
  }
  lambda = std::move(mu);
  return true;
}

#define HS_INSTANTIATE_PROJECTION(T)                                                                   \
  template void apply_block<T>(const CsrMatrix<T>*, const DenseMatrix<T>&, DenseMatrix<T>&);           \
  template index_t orthonormalize_columns<T>(DenseMatrix<T>&, double);                                 \
  template index_t rayleigh_ritz_definite<T>(DenseMatrix<T>&, DenseMatrix<T>&, DenseMatrix<T>&,     \
                                                std::vector<double>&, double);                         \
  template bool rayleigh_ritz_indefinite<T>(const DenseMatrix<T>&, const DenseMatrix<T>&,              \
                                            const DenseMatrix<T>&, std::vector<cplx>&, DenseMatrix<cplx>&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_PROJECTION)
#undef HS_INSTANTIATE_PROJECTION

template PairMetrics pair_metrics<double, double>(const double*, const double*, const double*, index_t, double,
                                                  double, double);
template PairMetrics pair_metrics<cplx, double>(const cplx*, const cplx*, const cplx*, index_t, double, double,
                                                double);
template PairMetrics pair_metrics<cplx, cplx>(const cplx*, const cplx*, const cplx*, index_t, cplx, double, double);

}  // namespace hs::driver
