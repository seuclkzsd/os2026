// Helpers shared by lobpcg and jacobi_davidson (see src/eigen/lobpcg_jd_common.hpp).
#include "eigen/lobpcg_jd_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

/// Rows per block of the in-place rotation (temporary of kRowBlock x k scalars).
constexpr index_t kRowBlock = 4096;
/// SVQB drops Gram eigen-directions below this fraction of the largest eigenvalue: the
/// normalized combination would be determined to no better than eps / sqrt(1e-12) ~ 2e-10.
constexpr double kSvqbDrop = 1e-12;
/// One orthonormalization round suffices when the projection removed at most this multiple of
/// what remains (CGS2 then leaves ~eps * ratio of span(against) in each column) ...
constexpr double kRoundRatio = 1e4;
/// ... and the scaled Gram matrix was this well conditioned (SVQB leaves ~eps * cond).
constexpr double kRoundCond = 1e4;
constexpr int kMaxRounds = 3;
/// Lanczos steps of the norm estimate (the driver uses 30 as well, DESIGN §3.6-1).
constexpr index_t kNormSteps = 30;

inline fint fi(index_t x) { return static_cast<fint>(x); }

template <class T>
bool finite(const T& x) {
  if constexpr (is_complex_v<T>) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
  } else {
    return std::isfinite(x);
  }
}

}  // namespace

template <class T>
void hermitize(DenseMatrix<T>& m) {
  const index_t k = m.rows();
  for (index_t j = 0; j < k; ++j) {
    m(j, j) = T(std::real(m(j, j)));
    for (index_t i = j + 1; i < k; ++i) {
      const T h = 0.5 * (m(i, j) + conj_if(m(j, i)));
      m(i, j) = h;
      m(j, i) = conj_if(h);
    }
  }
}

template <class T>
void apply_or_copy(LinearOperator<T>* op, index_t n, const T* x, T* y, index_t p) {
  if (op) {
    op->apply(x, n, y, n, p);
  } else {
    std::copy(x, x + n * p, y);
  }
}

template <class T>
double estimate_norm2(LinearOperator<T>& op, std::uint64_t seed) {
  const index_t n = op.size();
  if (n == 0) return 0.0;
  KrylovSchurOptions<T> o;
  o.nev = 1;
  o.ncv = std::min(n, kNormSteps);
  o.max_restarts = 0;
  o.tol = 1e-2;
  o.seed = seed;
  const auto r = lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, o);
  double m = 0.0;
  for (double th : r.ritz_all) m = std::max(m, std::abs(th));
  return m;
}

template <class T>
void rotate_in_place(index_t n, T* v, index_t m, const T* s, index_t lds, index_t k) {
  if (n == 0 || k == 0) return;
  const index_t rb = std::min(n, kRowBlock);
  DenseMatrix<T> tmp(rb, k);
  for (index_t r0 = 0; r0 < n; r0 += rb) {
    const index_t nr = std::min(rb, n - r0);
    lapack::gemm('N', 'N', fi(nr), fi(k), fi(m), T(1), v + r0, fi(n), s, fi(lds), T(0), tmp.data(),
                 fi(rb));
    for (index_t l = 0; l < k; ++l) std::copy(tmp.col(l), tmp.col(l) + nr, v + r0 + l * n);
  }
}

template <class T>
void require_finite(index_t n, const T* x, index_t p, const char* what) {
  const index_t len = n * p;
  bool ok = true;
#pragma omp parallel for reduction(&& : ok) schedule(static)
  for (index_t i = 0; i < len; ++i) ok = ok && finite(x[i]);
  if (!ok) HS_THROW(InvalidInput, what << ": non-finite values (an operator returned Inf/NaN)");
}

template <class T>
void require_b_orthonormal(index_t n, index_t d, const T* y, const T* by, const char* who) {
  DenseMatrix<T> g(d, d);
  linalg::gram(n, d, y, n, by, n, g.data(), d);
  double dev = 0.0;
  for (index_t j = 0; j < d; ++j)
    for (index_t i = 0; i < d; ++i) {
      const double e = std::abs(g(i, j) - T(i == j ? 1.0 : 0.0));
      if (std::isnan(e) || e > dev) dev = e;
    }
  if (!(dev <= kDeflationOrthoTol))
    HS_THROW(InvalidInput, who << ": deflation block is not B-orthonormal or not finite (max |Y^H B Y - I| = " << dev
                               << " > " << kDeflationOrthoTol << ")");
}

template <class T>
index_t svqb(const DenseMatrix<T>& g, DenseMatrix<T>& m, double* cond) {
  const index_t k = g.rows();
  std::vector<double> d(static_cast<std::size_t>(k));
  for (index_t i = 0; i < k; ++i) {
    const double gii = std::real(g(i, i));
    HS_ASSERT(gii > 0.0, "svqb: non-positive diagonal " << gii);
    d[i] = 1.0 / std::sqrt(gii);
  }
  DenseMatrix<T> s(k, k);
  for (index_t j = 0; j < k; ++j)
    for (index_t i = 0; i < k; ++i) s(i, j) = g(i, j) * (d[i] * d[j]);
  hermitize(s);
  std::vector<double> mu(static_cast<std::size_t>(k));
  lapack::heev('V', 'L', fi(k), s.data(), fi(k), mu.data());
  const double mu_max = k > 0 ? mu.back() : 0.0;
  if (!(mu_max > 0.0)) {
    m.resize(k, 0);
    return 0;
  }
  index_t f = 0;
  while (f < k && !(mu[f] > kSvqbDrop * mu_max)) ++f;
  const index_t q = k - f;
  m.resize(k, q);
  for (index_t l = 0; l < q; ++l) {
    const index_t e = k - 1 - l;  // descending mu: most independent direction first
    const double sc = 1.0 / std::sqrt(mu[e]);
    for (index_t i = 0; i < k; ++i) m(i, l) = s(i, e) * (sc * d[i]);
  }
  if (cond) *cond = mu_max / mu[f];
  return q;
}

template <class T>
index_t b_orthonormalize(index_t n, const std::vector<OrthoBlock<T>>& against, T* w, T* bw, index_t p,
                         LinearOperator<T>* b, double drop_rel) {
  if (!b) bw = w;
  HS_ASSERT(bw != nullptr, "b_orthonormalize: B given without a B*W buffer");
  index_t cur = p;
  for (int round = 0; round < kMaxRounds && cur > 0; ++round) {
    // Two block CGS passes; removed2[j] accumulates |coefficients|^2, so that by Pythagoras
    // (the blocks are B-orthonormal) ||w_j||_B^2 before = removed2[j] + ||w_j||_B^2 after.
    std::vector<double> removed2(static_cast<std::size_t>(cur), 0.0);
    for (int pass = 0; pass < 2; ++pass) {
      for (const OrthoBlock<T>& blk : against) {
        if (blk.cols == 0) continue;
        DenseMatrix<T> c(blk.cols, cur);
        lapack::gemm('C', 'N', fi(blk.cols), fi(cur), fi(n), T(1), blk.bv, fi(n), w, fi(n), T(0),
                     c.data(), fi(blk.cols));
        lapack::gemm('N', 'N', fi(n), fi(cur), fi(blk.cols), T(-1), blk.v, fi(n), c.data(),
                     fi(blk.cols), T(1), w, fi(n));
        for (index_t j = 0; j < cur; ++j)
          for (index_t i = 0; i < blk.cols; ++i) removed2[j] += abs2(c(i, j));
      }
    }
    if (b) b->apply(w, n, bw, n, cur);
    DenseMatrix<T> g(cur, cur);
    linalg::gram(n, cur, w, n, bw, n, g.data(), cur);
    for (index_t j = 0; j < cur; ++j)
      for (index_t i = 0; i < cur; ++i)
        if (!finite(g(i, j)))
          HS_THROW(InvalidInput, "B-orthonormalization: non-finite block (an operator returned Inf/NaN)");
    hermitize(g);

    // Drop columns that were (numerically) inside span(against).
    std::vector<index_t> keep;
    double worst_ratio = 0.0;
    for (index_t j = 0; j < cur; ++j) {
      const double a2 = std::real(g(j, j));
      const double a = a2 > 0.0 ? std::sqrt(a2) : 0.0;
      const double before = std::sqrt(removed2[j] + std::max(a2, 0.0));
      if (a > 0.0 && a > drop_rel * before) {
        keep.push_back(j);
        worst_ratio = std::max(worst_ratio, std::sqrt(removed2[j]) / a);
      } else if (!(a2 >= -1e-8 * before * before)) {
        HS_THROW(InvalidInput, "B-orthonormalization: w^H B w = " << a2 << " < 0, B is not positive definite");
      }
    }
    const index_t q0 = static_cast<index_t>(keep.size());
    for (index_t i = 0; i < q0; ++i) {
      if (keep[i] == i) continue;
      std::copy(w + keep[i] * n, w + (keep[i] + 1) * n, w + i * n);
      if (b) std::copy(bw + keep[i] * n, bw + (keep[i] + 1) * n, bw + i * n);
    }
    cur = q0;
    if (cur == 0) return 0;

    // SVQB with dropping of numerically dependent directions.
    DenseMatrix<T> gk(cur, cur), m;
    for (index_t j = 0; j < cur; ++j)
      for (index_t i = 0; i < cur; ++i) gk(i, j) = g(keep[i], keep[j]);
    double cond = 0.0;
    const index_t q = svqb(gk, m, &cond);
    if (q == 0) return 0;
    rotate_in_place(n, w, cur, m.data(), cur, q);
    if (b) rotate_in_place(n, bw, cur, m.data(), cur, q);
    cur = q;
    if (worst_ratio <= kRoundRatio && cond <= kRoundCond) break;
  }
  return cur;
}

template <class T>
bool rayleigh_ritz(const DenseMatrix<T>& ga, const DenseMatrix<T>& gb, const std::vector<index_t>& idx,
                   double ill_cond, bool force, std::vector<double>& theta, DenseMatrix<T>& c,
                   double* deviation) {
  const index_t q = static_cast<index_t>(idx.size()), ks = ga.rows();
  std::vector<double> d(static_cast<std::size_t>(q));
  double dev = 0.0;
  for (index_t i = 0; i < q; ++i) {
    const double gii = std::real(gb(idx[i], idx[i]));
    if (!(gii > 0.0) || !std::isfinite(gii))
      HS_THROW(InvalidInput, "Rayleigh-Ritz: basis vector with B-norm^2 " << gii
                                                                        << " (B not positive definite?)");
    d[i] = 1.0 / std::sqrt(gii);
  }
  DenseMatrix<T> s(q, q), gas(q, q);
  for (index_t j = 0; j < q; ++j)
    for (index_t i = 0; i < q; ++i) {
      const T gbij = gb(idx[i], idx[j]);
      dev = std::max(dev, std::abs(gbij - T(i == j ? 1.0 : 0.0)));
      s(i, j) = gbij * (d[i] * d[j]);
      gas(i, j) = ga(idx[i], idx[j]);
    }
  if (deviation) *deviation = dev;
  hermitize(s);
  hermitize(gas);
  std::vector<double> mu(static_cast<std::size_t>(q));
  lapack::heev('V', 'L', fi(q), s.data(), fi(q), mu.data());
  const double mu_max = mu.back();
  if (!(mu_max > 0.0)) HS_THROW(InvalidInput, "Rayleigh-Ritz: Gram matrix is not positive definite");
  if (!force && mu.front() < ill_cond * mu_max) return false;
  index_t f = 0;
  while (f < q && !(mu[f] > kSvqbDrop * mu_max)) ++f;
  const index_t r = q - f;
  DenseMatrix<T> m(q, r);  // D U(:, f:) diag(mu)^{-1/2}: the whitening transformation
  for (index_t l = 0; l < r; ++l) {
    const double sc = 1.0 / std::sqrt(mu[f + l]);
    for (index_t i = 0; i < q; ++i) m(i, l) = s(i, f + l) * (d[i] * sc);
  }
  DenseMatrix<T> tmp(q, r), red(r, r);
  lapack::gemm('N', 'N', fi(q), fi(r), fi(q), T(1), gas.data(), fi(q), m.data(), fi(q), T(0), tmp.data(),
               fi(q));
  lapack::gemm('C', 'N', fi(r), fi(r), fi(q), T(1), m.data(), fi(q), tmp.data(), fi(q), T(0), red.data(),
               fi(r));
  hermitize(red);
  theta.assign(static_cast<std::size_t>(r), 0.0);
  lapack::heev('V', 'L', fi(r), red.data(), fi(r), theta.data());
  DenseMatrix<T> csub(q, r);
  lapack::gemm('N', 'N', fi(q), fi(r), fi(r), T(1), m.data(), fi(q), red.data(), fi(r), T(0), csub.data(),
               fi(q));
  c.resize(ks, r);
  for (index_t l = 0; l < r; ++l)
    for (index_t i = 0; i < q; ++i) c(idx[i], l) = csub(i, l);
  return true;
}

double backward_error(double r, double theta, double xnorm, double norm_a, double norm_b) {
  if (r == 0.0) return 0.0;
  const double denom = (norm_a + std::abs(theta) * norm_b) * xnorm;
  return denom > 0.0 ? r / denom : std::numeric_limits<double>::infinity();
}

template <class T>
void fresh_residuals(LinearOperator<T>& a, LinearOperator<T>* b, index_t n, const T* x, index_t p,
                     const std::vector<double>& theta, double norm_a, double norm_b,
                     std::vector<double>& residual, std::vector<double>& eta) {
  residual.assign(static_cast<std::size_t>(p), 0.0);
  eta.assign(static_cast<std::size_t>(p), 0.0);
  if (p == 0) return;
  DenseMatrix<T> ax(n, p), bx(n, p);
  a.apply(x, n, ax.data(), n, p);
  apply_or_copy(b, n, x, bx.data(), p);
  require_finite(n, ax.data(), p, "residual check");
  require_finite(n, bx.data(), p, "residual check");
  for (index_t j = 0; j < p; ++j) {
    linalg::axpy(n, T(-theta[j]), bx.col(j), ax.col(j));
    residual[j] = linalg::norm2(n, ax.col(j));
    eta[j] = backward_error(residual[j], theta[j], linalg::norm2(n, x + j * n), norm_a, norm_b);
  }
}

#define HS_INSTANTIATE_LOBPCG_JD_COMMON(T)                                                            \
  template void apply_or_copy<T>(LinearOperator<T>*, index_t, const T*, T*, index_t);                 \
  template double estimate_norm2<T>(LinearOperator<T>&, std::uint64_t);                               \
  template void rotate_in_place<T>(index_t, T*, index_t, const T*, index_t, index_t);                 \
  template void hermitize<T>(DenseMatrix<T>&);                                                         \
  template void require_finite<T>(index_t, const T*, index_t, const char*);                           \
  template void require_b_orthonormal<T>(index_t, index_t, const T*, const T*, const char*);          \
  template index_t svqb<T>(const DenseMatrix<T>&, DenseMatrix<T>&, double*);                            \
  template index_t b_orthonormalize<T>(index_t, const std::vector<OrthoBlock<T>>&, T*, T*, index_t,   \
                                       LinearOperator<T>*, double);                                   \
  template bool rayleigh_ritz<T>(const DenseMatrix<T>&, const DenseMatrix<T>&,                        \
                                 const std::vector<index_t>&, double, bool, std::vector<double>&,     \
                                 DenseMatrix<T>&, double*);                                           \
  template void fresh_residuals<T>(LinearOperator<T>&, LinearOperator<T>*, index_t, const T*, index_t, \
                                   const std::vector<double>&, double, double, std::vector<double>&,  \
                                   std::vector<double>&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_LOBPCG_JD_COMMON)
#undef HS_INSTANTIATE_LOBPCG_JD_COMMON

}  // namespace hs::eigen_detail
