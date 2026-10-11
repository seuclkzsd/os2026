// Tests of the general (non-Hermitian) Krylov-Schur solver hs::krylov_schur_general and its
// projected-problem helpers (src/eigen/general_*). Operators are small dense nonsymmetric
// matrices with known spectra and generalized pencils with indefinite B (via shift-invert), so
// every check is against an analytic or dense LAPACK reference.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "eigen/general_krylov_schur.hpp"
#include "eigen/general_schur.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::index_t;
using hs::Which;
using hs::lapack::fint;

constexpr double kEps = std::numeric_limits<double>::epsilon();

// ------------------------------------------------------------------------------- dense helpers

template <class T>
DenseMatrix<T> random_matrix(index_t rows, index_t cols, std::uint64_t seed) {
  DenseMatrix<T> r(rows, cols);
  for (index_t j = 0; j < cols; ++j) hs::linalg::fill_random(r.col(j), rows, seed, j);
  return r;
}

template <class T>
DenseMatrix<T> product(const DenseMatrix<T>& a, const DenseMatrix<T>& b) {
  DenseMatrix<T> c(a.rows(), b.cols());
  hs::linalg::multiply(a.rows(), a.cols(), b.cols(), a.data(), a.rows(), b.data(), b.rows(),
                       c.data(), c.rows());
  return c;
}

/// A^{-1} B by LU (A square, nonsingular).
template <class T>
DenseMatrix<T> solve(DenseMatrix<T> a, DenseMatrix<T> b) {
  const auto n = static_cast<fint>(a.rows());
  std::vector<fint> ipiv(static_cast<std::size_t>(n));
  EXPECT_EQ(hs::lapack::getrf(n, a.data(), n, ipiv.data()), 0);
  hs::lapack::getrs(n, static_cast<fint>(b.cols()), a.data(), n, ipiv.data(), b.data(), n);
  return b;
}

/// All eigenvalues of a dense square matrix (LAPACK gees on a copy).
template <class T>
std::vector<cplx> dense_eigenvalues(DenseMatrix<T> a) {
  const auto n = static_cast<fint>(a.rows());
  DenseMatrix<T> q(n, n);
  std::vector<cplx> w(static_cast<std::size_t>(n));
  hs::lapack::gees(n, a.data(), n, w.data(), q.data(), n);
  return w;
}

/// A = X D X^{-1} with X = I + 0.5 R / sqrt(n) (well conditioned, far from normal).
template <class T>
DenseMatrix<T> similar_to(const DenseMatrix<T>& d, std::uint64_t seed) {
  const index_t n = d.rows();
  DenseMatrix<T> x = random_matrix<T>(n, n, seed);
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) x(i, j) *= 0.5 / std::sqrt(static_cast<double>(n));
    x(j, j) += T(1);
  }
  DenseMatrix<T> eye(n, n);
  for (index_t i = 0; i < n; ++i) eye(i, i) = T(1);
  return product(product(x, d), solve(x, eye));
}

/// Block diagonal matrix with the given spectrum. Real T: each value with nonzero imaginary part
/// stands for the conjugate pair a +- ib, stored as the rotation block [[a, b], [-b, a]].
/// Complex T: diagonal. Returns the matrix; *all receives every eigenvalue (pairs expanded).
template <class T>
DenseMatrix<T> block_diagonal(const std::vector<cplx>& values, std::vector<cplx>* all) {
  index_t n = 0;
  for (const cplx& v : values) n += (!hs::is_complex_v<T> && v.imag() != 0.0) ? 2 : 1;
  DenseMatrix<T> d(n, n);
  all->clear();
  index_t i = 0;
  for (const cplx& v : values) {
    if constexpr (hs::is_complex_v<T>) {
      d(i, i) = v;
      all->push_back(v);
      ++i;
    } else if (v.imag() == 0.0) {
      d(i, i) = v.real();
      all->push_back(v);
      ++i;
    } else {
      d(i, i) = d(i + 1, i + 1) = v.real();
      d(i, i + 1) = v.imag();
      d(i + 1, i) = -v.imag();
      all->push_back(v);
      all->push_back(std::conj(v));
      i += 2;
    }
  }
  return d;
}

double rank_key(const cplx& v, Which which) {
  switch (which) {
    case Which::LargestMagnitude:
      return -std::abs(v);
    case Which::LargestAlgebraic:
      return -v.real();
    case Which::SmallestAlgebraic:
      return v.real();
  }
  return 0.0;
}

/// The first nev values of `all` by `which` ranking, completed by the conjugate partner when a
/// real-matrix pair straddles position nev (the contract's m = nev + 1 case).
std::vector<cplx> expected_wanted(std::vector<cplx> all, Which which, index_t nev, bool pairs) {
  std::stable_sort(all.begin(), all.end(), [which](const cplx& a, const cplx& b) {
    const double ka = rank_key(a, which), kb = rank_key(b, which);
    return ka != kb ? ka < kb : a.imag() > b.imag();
  });
  index_t m = nev;
  if (pairs && all[nev - 1].imag() > 0.0) m = nev + 1;
  all.resize(static_cast<std::size_t>(m));
  return all;
}

/// Each value of `got` matches a distinct value of `want` within tol (greedy nearest match).
void expect_same_values(const std::vector<cplx>& got, std::vector<cplx> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (const cplx& g : got) {
    auto best = std::min_element(want.begin(), want.end(), [&g](const cplx& a, const cplx& b) {
      return std::abs(a - g) < std::abs(b - g);
    });
    EXPECT_LE(std::abs(*best - g), tol) << "computed " << g << " nearest expected " << *best;
    want.erase(best);
  }
}

/// y = Op x for a complex vector (real operators are applied to the real and imaginary parts).
template <class T>
std::vector<cplx> apply_complex(hs::LinearOperator<T>& op, const cplx* x) {
  const index_t n = op.size();
  std::vector<cplx> y(static_cast<std::size_t>(n));
  if constexpr (hs::is_complex_v<T>) {
    op.apply(x, n, y.data(), n, 1);
  } else {
    DenseMatrix<double> parts(n, 2), out(n, 2);
    for (index_t i = 0; i < n; ++i) parts(i, 0) = x[i].real(), parts(i, 1) = x[i].imag();
    op.apply(parts.data(), n, out.data(), n, 2);
    for (index_t i = 0; i < n; ++i) y[i] = cplx(out(i, 0), out(i, 1));
  }
  return y;
}

/// Checks every structural promise of the GeneralEigResult contract and the accuracy of the
/// estimates against true residuals. op_norm is an upper bound of ||Op||_2.
template <class T>
void check_contract(hs::LinearOperator<T>& op, const hs::GeneralEigResult<T>& r, Which which,
                    index_t nev, double tol, double op_norm) {
  const index_t n = op.size();
  const auto m = static_cast<index_t>(r.theta.size());
  ASSERT_TRUE(m == nev || (!hs::is_complex_v<T> && m == nev + 1)) << "m=" << m << " nev=" << nev;
  ASSERT_EQ(static_cast<index_t>(r.estimate.size()), m);
  ASSERT_EQ(r.vectors.rows(), n);
  ASSERT_EQ(r.vectors.cols(), m);
  ASSERT_EQ(r.schur_basis.rows(), n);
  ASSERT_EQ(r.schur_basis.cols(), m);
  ASSERT_EQ(r.schur_form.rows(), m);
  ASSERT_EQ(r.schur_form.cols(), m);
  ASSERT_GE(r.nconv, 0);
  ASSERT_LE(r.nconv, m);
  EXPECT_EQ(r.converged, r.nconv >= nev);
  ASSERT_FALSE(r.history.empty());
  EXPECT_EQ(static_cast<index_t>(r.history.size()), r.restarts + 1);
  EXPECT_EQ(r.history.back().op_applies, r.op_applies);

  double largest = 0.0;
  for (const cplx& t : r.ritz_all) largest = std::max(largest, std::abs(t));
  for (index_t i = 0; i < m; ++i) {
    // [0, nconv) converged, both partitions sorted by the ranking.
    if (i < r.nconv) {
      EXPECT_LE(r.estimate[i], tol * std::abs(r.theta[i])) << i;
    }
    if (i + 1 < m && i + 1 != r.nconv) {
      EXPECT_LE(rank_key(r.theta[i], which), rank_key(r.theta[i + 1], which)) << i;
    }
    // Conjugate pairs adjacent, positive imaginary part first, conjugate vectors; nconv and
    // the Schur cut never split them.
    if (!hs::is_complex_v<T> && r.theta[i].imag() != 0.0) {
      ASSERT_GT(r.theta[i].imag(), 0.0) << i;
      ASSERT_LT(i + 1, m);
      EXPECT_EQ(r.theta[i + 1], std::conj(r.theta[i]));
      EXPECT_EQ(r.estimate[i + 1], r.estimate[i]);
      EXPECT_NE(r.nconv, i + 1);
      for (index_t k = 0; k < n; ++k) ASSERT_EQ(r.vectors(k, i + 1), std::conj(r.vectors(k, i)));
      ++i;
    }
  }
  // Unit vectors; true residuals agree with the estimates up to rounding.
  for (index_t i = 0; i < m; ++i) {
    const cplx* x = r.vectors.col(i);
    EXPECT_NEAR(hs::linalg::norm2(n, x), 1.0, 1e-13);
    std::vector<cplx> y = apply_complex(op, x);
    for (index_t k = 0; k < n; ++k) y[k] -= r.theta[i] * x[k];
    const double res = hs::linalg::norm2(n, y.data());
    EXPECT_NEAR(res, r.estimate[i], 1e3 * kEps * op_norm) << "pair " << i;
  }
  // Orthonormal Schur basis, quasi-triangular S without a cut 2x2 block.
  DenseMatrix<T> gram(m, m);
  hs::linalg::gram(n, m, r.schur_basis.data(), n, r.schur_basis.data(), n, gram.data(), m);
  for (index_t j = 0; j < m; ++j) {
    for (index_t i = 0; i < m; ++i) {
      EXPECT_NEAR(std::abs(gram(i, j) - T(i == j ? 1.0 : 0.0)), 0.0, 1e-12);
    }
  }
  const DenseMatrix<T>& s = r.schur_form;
  for (index_t j = 0; j < m; ++j) {
    for (index_t i = j + (hs::is_complex_v<T> ? 1 : 2); i < m; ++i) EXPECT_EQ(s(i, j), T(0));
    if (j + 2 < m && s(j + 1, j) != T(0)) {
      EXPECT_EQ(s(j + 2, j + 1), T(0)) << "chained 2x2 blocks";
    }
  }
  if (r.nconv > 0 && r.nconv < m) {
    EXPECT_EQ(s(r.nconv, r.nconv - 1), T(0)) << "2x2 block cut at nconv";
  }
  // spec(S) = theta, spec(S(0:nconv, 0:nconv)) = theta[0:nconv).
  expect_same_values(dense_eigenvalues(s), r.theta, 1e-10 * largest);
  if (r.nconv > 0) {
    DenseMatrix<T> lead(r.nconv, r.nconv);
    for (index_t j = 0; j < r.nconv; ++j)
      for (index_t i = 0; i < r.nconv; ++i) lead(i, j) = s(i, j);
    expect_same_values(dense_eigenvalues(lead),
                       std::vector<cplx>(r.theta.begin(), r.theta.begin() + r.nconv),
                       1e-10 * largest);
    // Leading nconv Schur vectors span an approximately invariant subspace.
    DenseMatrix<T> oq(n, r.nconv);
    op.apply(r.schur_basis.data(), n, oq.data(), n, r.nconv);
    double res2 = 0.0;
    for (index_t j = 0; j < r.nconv; ++j) {
      for (index_t i = 0; i < n; ++i) {
        T qs = T(0);
        for (index_t l = 0; l <= std::min(j + 1, r.nconv - 1); ++l)
          qs += r.schur_basis(i, l) * s(l, j);
        res2 += hs::abs2(oq(i, j) - qs);
      }
    }
    EXPECT_LE(std::sqrt(res2), 1e2 * std::sqrt(static_cast<double>(r.nconv)) * tol * op_norm);
  }
}

template <class T>
double frobenius(const DenseMatrix<T>& a) {
  double s = 0.0;
  for (index_t j = 0; j < a.cols(); ++j)
    for (index_t i = 0; i < a.rows(); ++i) s += hs::abs2(a(i, j));
  return std::sqrt(s);
}

// --------------------------------------------------------------------------------- spectra

/// Distinct magnitudes 1..~0.05 with a few conjugate pairs (real T) or genuinely complex values
/// (complex T); nothing is near a ranking tie at the cuts used below.
template <class T>
std::vector<cplx> mixed_spectrum(index_t count) {
  std::vector<cplx> v;
  for (index_t i = 0; i < count; ++i) {
    const double mag = 1.0 / (1.0 + 0.37 * static_cast<double>(i));
    const double phase = 0.9 * std::sin(1.3 * static_cast<double>(i) + 0.4);
    if constexpr (hs::is_complex_v<T>) {
      v.push_back(std::polar(mag, 2.0 * phase));
    } else if (i % 4 == 1) {
      v.push_back(std::polar(mag, 0.6 + 0.3 * phase));  // conjugate pair
    } else {
      v.push_back(i % 3 == 0 ? -mag : mag);
    }
  }
  return v;
}

template <class T>
class KrylovSchurGeneralTyped : public ::testing::Test {};
using ScalarTypes = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(KrylovSchurGeneralTyped, ScalarTypes);

TYPED_TEST(KrylovSchurGeneralTyped, KnownSpectrumLargestMagnitude) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(70), &all), 11);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 9;
  opt.tol = 1e-12;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  EXPECT_EQ(r.nconv, static_cast<index_t>(r.theta.size()));
  EXPECT_EQ(r.op_applies, op.applications());
  expect_same_values(
      r.theta, expected_wanted(all, Which::LargestMagnitude, opt.nev, !hs::is_complex_v<T>), 1e-10);
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, KnownSpectrumAlgebraicEnds) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(60), &all), 12);
  for (const Which which : {Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
    hs::test::DenseOperator<T> op(a);
    hs::KrylovSchurOptions<T> opt;
    opt.nev = 5;
    opt.ncv = 24;
    const auto r = hs::krylov_schur_general(op, which, opt);
    ASSERT_TRUE(r.converged);
    expect_same_values(r.theta, expected_wanted(all, which, opt.nev, !hs::is_complex_v<T>), 1e-10);
    check_contract(op, r, which, opt.nev, opt.tol, frobenius(a));
  }
}

TEST(KrylovSchurGeneral, ConjugatePairAtNevBoundaryGivesNevPlusOne) {
  // |theta| ranking: 10, -9, 8, then the pair 7 +- 2i straddles nev = 4.
  std::vector<cplx> values{{10, 0}, {-9, 0}, {8, 0}, {7, 2}, {5, 0}, {-4.5, 0}, {3, 1}};
  for (index_t i = 0; i < 40; ++i)
    values.emplace_back((i % 2 ? -1.0 : 1.0) * (2.0 - 0.04 * i), 0.0);
  std::vector<cplx> all;
  const DenseMatrix<double> a = similar_to(block_diagonal<double>(values, &all), 13);
  hs::test::DenseOperator<double> op(a);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 4;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  ASSERT_EQ(r.theta.size(), 5u);
  EXPECT_EQ(r.nconv, 5);
  EXPECT_NEAR(r.theta[3].real(), 7.0, 1e-10);
  EXPECT_NEAR(r.theta[3].imag(), 2.0, 1e-10);
  EXPECT_EQ(r.theta[4], std::conj(r.theta[3]));
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));

  // nev = 3 cuts before the pair: m == nev.
  opt.nev = 3;
  hs::test::DenseOperator<double> op3(a);
  const auto r3 = hs::krylov_schur_general(op3, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  EXPECT_EQ(r3.theta.size(), 3u);
  check_contract(op3, r3, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

// ------------------------------------------------------------- generalized indefinite pencils

/// Random dense Hermitian matrix.
template <class T>
DenseMatrix<T> random_hermitian(index_t n, std::uint64_t seed) {
  DenseMatrix<T> r = random_matrix<T>(n, n, seed), h(n, n);
  for (index_t j = 0; j < n; ++j)
    for (index_t i = 0; i < n; ++i) h(i, j) = 0.5 * (r(i, j) + hs::conj_if(r(j, i)));
  for (index_t i = 0; i < n; ++i) h(i, i) = T(std::real(h(i, i)));
  return h;
}

/// ||Op||_F for Op = (A - sigma B)^{-1} B, an upper bound of ||Op||_2.
template <class T>
double shift_invert_norm(const DenseMatrix<T>& a, const DenseMatrix<T>& b, double sigma) {
  DenseMatrix<T> k = a;
  for (index_t j = 0; j < a.cols(); ++j)
    for (index_t i = 0; i < a.rows(); ++i) k(i, j) -= T(sigma) * b(i, j);
  return frobenius(solve(k, b));
}

/// lambda = sigma + 1 / theta for every Ritz value.
std::vector<cplx> to_lambda(const std::vector<cplx>& theta, double sigma) {
  std::vector<cplx> l;
  for (const cplx& t : theta) l.push_back(sigma + 1.0 / t);
  return l;
}

/// Computed eigenvalues must be the ones nearest to sigma: each matches a reference eigenvalue
/// and none is farther than the nev-th nearest reference (ties between conjugates allowed).
void expect_nearest(const std::vector<cplx>& lambda, std::vector<cplx> ref, double sigma,
                    index_t nev, double tol) {
  std::sort(ref.begin(), ref.end(), [sigma](const cplx& a, const cplx& b) {
    return std::abs(a - sigma) < std::abs(b - sigma);
  });
  const double cut = std::abs(ref[nev - 1] - sigma);
  for (const cplx& l : lambda) EXPECT_LE(std::abs(l - sigma), cut * (1.0 + 1e-8) + tol) << l;
  std::vector<cplx> near(ref.begin(),
                         ref.begin() + std::min<std::size_t>(ref.size(), lambda.size() + 2));
  std::vector<cplx> pool = near;
  for (const cplx& l : lambda) {
    auto best = std::min_element(pool.begin(), pool.end(), [&l](const cplx& a, const cplx& b) {
      return std::abs(a - l) < std::abs(b - l);
    });
    EXPECT_LE(std::abs(*best - l), tol) << "lambda " << l << " reference " << *best;
    pool.erase(best);
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, GhiepShiftInvertMatchesDenseReference) {
  using T = TypeParam;
  const index_t n = 90;
  const DenseMatrix<T> a = random_hermitian<T>(n, 21);
  // B Hermitian and indefinite: diagonal of both signs plus a Hermitian perturbation.
  DenseMatrix<T> b = random_hermitian<T>(n, 22);
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) b(i, j) *= 0.15;
    b(j, j) += T((j % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.01 * j));
  }
  const double sigma = 0.137;
  const std::vector<cplx> ref = dense_eigenvalues(solve(b, a));  // eig(B^{-1} A)
  index_t ncomplex = 0;
  for (const cplx& l : ref) ncomplex += std::abs(l.imag()) > 1e-8 * std::abs(l) ? 1 : 0;
  EXPECT_GT(ncomplex, 0) << "pencil should have complex eigenvalues";

  hs::test::DenseShiftInvertOperator<T> op(a, &b, sigma);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 10;
  opt.tol = 1e-12;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  double scale = 0.0;
  for (const cplx& l : ref) scale = std::max(scale, std::abs(l));
  expect_nearest(to_lambda(r.theta, sigma), ref, sigma, opt.nev, 1e-9 * scale);
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, shift_invert_norm(a, b, sigma));
}

/// Waveguide-like pencil (Lee-Sun-Cendes structure) on an nx x ny node grid with the grid
/// edges as edge unknowns: A = blkdiag(A_tt, 0), B = [[T, T D], [D^T T, D^T T D - k2 M]] with
/// SPD A_tt, T, M and the edge-node incidence D. B = C^T blkdiag(T, -k2 M) C with C = [[I, D],
/// [0, I]] is indefinite; A is PSD with null space = the node block, so lambda = 0 is a
/// semisimple eigenvalue of multiplicity #nodes and every nonzero eigenvalue is real
/// (x^H A x > 0 for Ax != 0 and lambda = x^H A x / x^H B x).
struct Waveguide {
  DenseMatrix<double> a, b;
  index_t nodes = 0, edges = 0;
};

Waveguide waveguide_pencil(index_t nx, index_t ny, double k2) {
  Waveguide w;
  w.nodes = nx * ny;
  std::vector<std::pair<index_t, index_t>> edge;
  for (index_t j = 0; j < ny; ++j)
    for (index_t i = 0; i < nx; ++i) {
      if (i + 1 < nx) edge.emplace_back(i + j * nx, i + 1 + j * nx);
      if (j + 1 < ny) edge.emplace_back(i + j * nx, i + (j + 1) * nx);
    }
  w.edges = static_cast<index_t>(edge.size());
  const index_t ne = w.edges, nn = w.nodes, n = ne + nn;
  DenseMatrix<double> d(ne, nn), t(ne, ne), m(nn, nn), att(ne, ne);
  for (index_t e = 0; e < ne; ++e) {
    d(e, edge[e].first) = -1.0;
    d(e, edge[e].second) = 1.0;
  }
  // Diagonally dominant tridiagonal SPD blocks with mildly varying coefficients.
  for (index_t e = 0; e < ne; ++e) {
    t(e, e) = 2.0 / 3.0 + 0.05 * std::sin(0.7 * e);
    att(e, e) = 4.0 + 0.5 * std::cos(1.1 * e);
    if (e + 1 < ne) t(e, e + 1) = t(e + 1, e) = 1.0 / 6.0, att(e, e + 1) = att(e + 1, e) = -1.0;
  }
  for (index_t i = 0; i < nn; ++i) {
    m(i, i) = 2.0 / 3.0 + 0.05 * std::cos(0.3 * i);
    if (i + 1 < nn) m(i, i + 1) = m(i + 1, i) = 1.0 / 6.0;
  }
  const DenseMatrix<double> td = product(t, d);
  w.a.resize(n, n);
  w.b.resize(n, n);
  for (index_t j = 0; j < ne; ++j)
    for (index_t i = 0; i < ne; ++i) w.a(i, j) = att(i, j), w.b(i, j) = t(i, j);
  for (index_t j = 0; j < nn; ++j)
    for (index_t i = 0; i < ne; ++i) w.b(i, ne + j) = w.b(ne + j, i) = td(i, j);
  for (index_t j = 0; j < nn; ++j)
    for (index_t i = 0; i < nn; ++i) {
      double dtd = 0.0;
      for (index_t e = 0; e < ne; ++e) dtd += d(e, i) * td(e, j);
      w.b(ne + i, ne + j) = dtd - k2 * m(i, j);
    }
  return w;
}

/// Dense reference of the waveguide pencil, split into the zero cluster and the rest.
struct WaveguideReference {
  std::vector<cplx> nonzero;  ///< eigenvalues with |lambda| > 1e-10 scale
  double scale = 0.0;         ///< max |lambda|
};

WaveguideReference waveguide_reference(const Waveguide& w) {
  WaveguideReference ref;
  const std::vector<cplx> all = dense_eigenvalues(solve(w.b, w.a));
  for (const cplx& l : all) ref.scale = std::max(ref.scale, std::abs(l));
  for (const cplx& l : all) {
    if (std::abs(l) > 1e-10 * ref.scale) ref.nonzero.push_back(l);
  }
  return ref;
}

std::vector<cplx> sorted_by_distance(std::vector<cplx> v, double sigma) {
  std::sort(v.begin(), v.end(), [sigma](const cplx& a, const cplx& b) {
    return std::abs(a - sigma) < std::abs(b - sigma);
  });
  return v;
}

TEST(KrylovSchurGeneral, WaveguideReferenceHasExactZeroClusterAndRealSpectrum) {
  const Waveguide w = waveguide_pencil(8, 6, 2.0);
  const WaveguideReference ref = waveguide_reference(w);
  EXPECT_EQ(static_cast<index_t>(ref.nonzero.size()), w.edges);  // #nodes exact zeros
  for (const cplx& l : ref.nonzero) EXPECT_LT(std::abs(l.imag()), 1e-10 * ref.scale) << l;
}

TEST(KrylovSchurGeneral, WaveguideZeroClusterWantedComesOutReal) {
  // sigma next to 0: theta = -1/sigma (multiplicity #nodes) dominates and is wanted together with
  // the nearest nonzero eigenvalues. A single-vector Krylov space holds one copy of a semisimple
  // multiple eigenvalue in exact arithmetic; further copies only emerge from rounding, so the run
  // may converge with part of the cluster (completeness of exact multiplicities is outside the
  // solver contract, DESIGN §3.6-8/9). Whatever is returned must be exact: zero copies real to
  // roundoff with eigenvectors in null(A), and the nonzero eigenvalues the nearest ones in order.
  const Waveguide w = waveguide_pencil(8, 6, 2.0);
  const WaveguideReference ref = waveguide_reference(w);
  double smallest_nonzero = std::numeric_limits<double>::infinity();
  for (const cplx& l : ref.nonzero) smallest_nonzero = std::min(smallest_nonzero, std::abs(l));
  const double sigma = -0.05 * smallest_nonzero;
  hs::test::DenseShiftInvertOperator<double> op(w.a, &w.b, sigma);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = w.nodes + 5;
  opt.ncv = opt.nev + 30;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  const std::vector<cplx> lambda = to_lambda(r.theta, sigma);
  index_t zeros = 0;
  std::vector<cplx> found_nonzero;
  for (index_t i = 0; i < static_cast<index_t>(lambda.size()); ++i) {
    if (std::abs(lambda[i]) > 1e-10 * ref.scale) {
      found_nonzero.push_back(lambda[i]);
      continue;
    }
    ++zeros;
    EXPECT_LE(std::abs(lambda[i].imag()), 1e-15 * ref.scale) << "zero cluster must be real";
    double edge_part = 0.0;  // null(A) is the node block
    for (index_t e = 0; e < w.edges; ++e) edge_part += std::norm(r.vectors(e, i));
    EXPECT_LE(std::sqrt(edge_part), 1e-10);
  }
  EXPECT_GE(zeros, 1);
  EXPECT_LE(zeros, w.nodes);
  std::vector<cplx> nearest = sorted_by_distance(ref.nonzero, sigma);
  nearest.resize(found_nonzero.size());
  expect_same_values(found_nonzero, nearest, 1e-10 * ref.scale);
  for (const cplx& l : found_nonzero) EXPECT_LE(std::abs(l.imag()), 1e-12 * ref.scale) << l;
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol,
                 shift_invert_norm(w.a, w.b, sigma));
}

TEST(KrylovSchurGeneral, WaveguideShiftAwayFromZeroClusterFindsNearestModes) {
  // The driver's placement (DESIGN §3.6-9): sigma inside the nonzero spectrum, the zero cluster
  // present but unwanted. The nev nearest eigenvalues must all be found, real, none spurious.
  const Waveguide w = waveguide_pencil(8, 6, 2.0);
  const WaveguideReference ref = waveguide_reference(w);
  std::vector<double> positive;
  for (const cplx& l : ref.nonzero) {
    if (l.real() > 0.0) positive.push_back(l.real());
  }
  std::sort(positive.begin(), positive.end());
  ASSERT_GE(positive.size(), 12u);
  const double sigma = 0.5 * (positive[9] + positive[10]);
  hs::test::DenseShiftInvertOperator<double> op(w.a, &w.b, sigma);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 6;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  ASSERT_EQ(r.theta.size(), 6u);
  const std::vector<cplx> lambda = to_lambda(r.theta, sigma);
  std::vector<cplx> nearest = sorted_by_distance(ref.nonzero, sigma);
  nearest.resize(lambda.size());
  expect_same_values(lambda, nearest, 1e-10 * ref.scale);
  for (const cplx& l : lambda) EXPECT_LE(std::abs(l.imag()), 1e-12 * ref.scale) << l;
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol,
                 shift_invert_norm(w.a, w.b, sigma));
}

// ----------------------------------------------------------------------- behaviour/contract

TYPED_TEST(KrylovSchurGeneralTyped, DeterministicForFixedSeed) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(50), &all), 31);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 16;
  hs::test::DenseOperator<T> op1(a), op2(a), op3(a);
  const auto r1 = hs::krylov_schur_general(op1, Which::LargestMagnitude, opt);
  const auto r2 = hs::krylov_schur_general(op2, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r1.converged);
  EXPECT_EQ(r1.theta, r2.theta);
  EXPECT_EQ(r1.estimate, r2.estimate);
  EXPECT_EQ(r1.restarts, r2.restarts);
  EXPECT_EQ(r1.op_applies, r2.op_applies);
  for (index_t j = 0; j < r1.vectors.cols(); ++j)
    for (index_t i = 0; i < r1.vectors.rows(); ++i) ASSERT_EQ(r1.vectors(i, j), r2.vectors(i, j));
  // Another seed takes another path to the same eigenvalues.
  opt.seed = 7;
  const auto r3 = hs::krylov_schur_general(op3, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  expect_same_values(r3.theta, r1.theta, 1e-10);
}

TYPED_TEST(KrylovSchurGeneralTyped, InvalidOptionsThrow) {
  using T = TypeParam;
  hs::test::DiagonalOperator<T> op(std::vector<T>(20, T(1)));
  auto expect_invalid = [&op](const hs::KrylovSchurOptions<T>& o) {
    try {
      hs::krylov_schur_general(op, Which::LargestMagnitude, o);
      ADD_FAILURE() << "expected InvalidInput";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  };
  hs::KrylovSchurOptions<T> o;
  o.nev = 3;
  DenseMatrix<T> d(20, 1);
  d(0, 0) = T(1);
  auto bad = o;
  bad.deflation = &d;
  expect_invalid(bad);
  bad = o, bad.nev = 0;
  expect_invalid(bad);
  bad = o, bad.nev = 21;
  expect_invalid(bad);
  bad = o, bad.ncv = 3;  // < nev + 2 (real) / nev + 1 (complex)
  expect_invalid(bad);
  if (!hs::is_complex_v<T>) {
    bad = o, bad.ncv = 4;
    expect_invalid(bad);
  }
  bad = o, bad.max_restarts = -1;
  expect_invalid(bad);
  bad = o, bad.tol = -1e-3;
  expect_invalid(bad);
  DenseMatrix<T> wrong(19, 1), zero(20, 1);
  wrong(0, 0) = T(1);
  bad = o, bad.initial = &wrong;
  expect_invalid(bad);
  bad = o, bad.initial = &zero;
  expect_invalid(bad);
  EXPECT_EQ(op.applications(), 0);
  // ncv larger than n is clamped, not rejected.
  hs::KrylovSchurOptions<T> big = o;
  big.ncv = 100;
  EXPECT_TRUE(hs::krylov_schur_general(op, Which::LargestMagnitude, big).converged);
}

TYPED_TEST(KrylovSchurGeneralTyped, NonConvergenceIsReportedNotThrown) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(80), &all), 41);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 8;
  opt.ncv = 11;
  opt.max_restarts = 0;
  opt.tol = 1e-14;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  EXPECT_FALSE(r.converged);
  EXPECT_LT(r.nconv, opt.nev);
  EXPECT_EQ(r.restarts, 0);
  EXPECT_EQ(r.history.size(), 1u);
  EXPECT_GT(r.history[0].max_estimate, opt.tol);
  EXPECT_EQ(r.op_applies, opt.ncv);
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, PartialConvergenceKeepsConvergedSchurBlockOnTop) {
  using T = TypeParam;
  // Three dominant, well separated values (a conjugate pair among them for real T) converge
  // within one restart; the clustered rest cannot, so the result is a partial one whose
  // converged Schur block must lead and be invariant.
  std::vector<cplx> values{{100, 0}, {50, 20}, {-25, 0}};
  for (index_t i = 0; i < 60; ++i)
    values.emplace_back(1.0 - 0.002 * i, hs::is_complex_v<T> ? 0.001 * i : 0.0);
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(values, &all), 43);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 12;
  opt.max_restarts = 1;
  opt.tol = 1e-10;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  EXPECT_FALSE(r.converged);
  const index_t dominant = hs::is_complex_v<T> ? 3 : 4;
  ASSERT_EQ(r.nconv, dominant);
  EXPECT_NEAR(std::abs(r.theta[0] - cplx(100)), 0.0, 1e-8);
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

TEST(KrylovSchurGeneral, MinimalBasisWithManyPairsConverges) {
  // ncv = nev + 2 is the smallest real basis: pairs straddling the kept count must sometimes be
  // dropped from the retained set instead of being kept whole.
  std::vector<cplx> values;
  for (index_t i = 0; i < 50; ++i) {
    const double mag = 1.0 / (1.0 + 0.3 * i);
    values.push_back(i % 3 == 2 ? cplx(mag, 0.0) : std::polar(mag, 0.3 + 0.02 * i));
  }
  std::vector<cplx> all;
  const DenseMatrix<double> a = similar_to(block_diagonal<double>(values, &all), 44);
  hs::test::DenseOperator<double> op(a);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 5;
  opt.ncv = 7;
  opt.tol = 1e-11;
  opt.max_restarts = 5000;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  expect_same_values(r.theta, expected_wanted(all, Which::LargestMagnitude, opt.nev, true), 1e-9);
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, BreakdownOnInvariantStartVectorRecovers) {
  using T = TypeParam;
  // Diagonal operator and a start vector inside a 3-dimensional invariant subspace: the Krylov
  // space is exhausted after 3 steps and must be continued with fresh directions.
  const index_t n = 60;
  std::vector<T> diag(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) diag[i] = T(1.0 + 0.5 * i);
  hs::test::DiagonalOperator<T> op(diag);
  DenseMatrix<T> start(n, 1);
  start(2, 0) = T(1), start(10, 0) = T(-2), start(30, 0) = T(0.5);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 5;
  opt.ncv = 12;
  opt.initial = &start;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  for (index_t i = 0; i < opt.nev; ++i) {
    EXPECT_NEAR(r.theta[i].real(), 1.0 + 0.5 * (n - 1 - i), 1e-10);
    EXPECT_NEAR(r.theta[i].imag(), 0.0, hs::is_complex_v<T> ? 1e-12 : 0.0);
  }
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, 1.0 + 0.5 * n);
}

TYPED_TEST(KrylovSchurGeneralTyped, WholeSpaceBasisIsExactAtOnce) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(5), &all), 51);
  const index_t n = a.rows();
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = n;
  const auto r = hs::krylov_schur_general(op, Which::SmallestAlgebraic, opt);
  ASSERT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  EXPECT_EQ(r.op_applies, n);
  expect_same_values(r.theta, all, 1e-12);
  check_contract(op, r, Which::SmallestAlgebraic, opt.nev, opt.tol, frobenius(a));

  // n = 1.
  hs::test::DiagonalOperator<T> one(std::vector<T>{T(-3)});
  opt.nev = 1;
  const auto r1 = hs::krylov_schur_general(one, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r1.converged);
  ASSERT_EQ(r1.theta.size(), 1u);
  EXPECT_NEAR(std::abs(r1.theta[0] - cplx(-3)), 0.0, 1e-15);
}

TEST(KrylovSchurGeneral, RealRotationNevOneReturnsWholePair) {
  DenseMatrix<double> a(2, 2);
  a(0, 0) = a(1, 1) = 0.5;
  a(0, 1) = 2.0;
  a(1, 0) = -2.0;
  hs::test::DenseOperator<double> op(a);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 1;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  ASSERT_EQ(r.theta.size(), 2u);
  EXPECT_NEAR(std::abs(r.theta[0] - cplx(0.5, 2.0)), 0.0, 1e-14);
  EXPECT_EQ(r.theta[1], std::conj(r.theta[0]));
  check_contract(op, r, Which::LargestMagnitude, opt.nev, opt.tol, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, PhaseTimesAccountForTheRun) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(60), &all), 61);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.ncv = 12;
  hs::eigen_detail::GeneralPhaseTimes times;
  const auto r =
      hs::eigen_detail::krylov_schur_general_timed(op, Which::LargestMagnitude, opt, &times);
  ASSERT_TRUE(r.converged);
  const auto plain = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  EXPECT_EQ(r.theta, plain.theta);
  for (double t : {times.apply_s, times.orthogonalize_s, times.projected_s, times.update_s})
    EXPECT_GE(t, 0.0);
  EXPECT_GT(times.total_s, 0.0);
  EXPECT_LE(times.apply_s + times.orthogonalize_s + times.projected_s + times.update_s,
            times.total_s * (1.0 + 1e-9));
}

// ------------------------------------------------------------- projected-problem helpers

TYPED_TEST(KrylovSchurGeneralTyped, ProjectedSchurRanksAndReordersWholeBlocks) {
  using T = TypeParam;
  std::vector<cplx> all;
  // Real T: values 1 and 3 +- 4i (|.| = 5) and -2; complex T: four distinct complex values.
  const std::vector<cplx> values = hs::is_complex_v<T>
                                       ? std::vector<cplx>{{1, 0}, {3, 4}, {-2, 0}, {0, -6}}
                                       : std::vector<cplx>{{1, 0}, {3, 4}, {-2, 0}, {0.5, 6}};
  const DenseMatrix<T> h = similar_to(block_diagonal<T>(values, &all), 71);
  const index_t m = h.rows();
  hs::eigen_detail::ProjectedSchur<T> ps(m, h.data(), m);
  const auto ranked = ps.ranking(Which::LargestMagnitude);
  const auto& blocks = ps.blocks();
  for (std::size_t i = 0; i + 1 < ranked.size(); ++i) {
    EXPECT_GE(std::abs(blocks[ranked[i]].value), std::abs(blocks[ranked[i + 1]].value));
  }
  index_t held = 0;
  const index_t nb = hs::eigen_detail::blocks_for_count(blocks, ranked, 1, &held);
  EXPECT_EQ(nb, 1);
  EXPECT_EQ(held, hs::is_complex_v<T> ? 1 : 2);  // the top value is a pair for real T
  // Move the smallest-magnitude block to the front: Q S Q^H must still equal H.
  std::vector<char> sel(blocks.size(), 0);
  sel[ranked.back()] = 1;
  const cplx moved = blocks[ranked.back()].value;
  ASSERT_TRUE(ps.reorder_to_front(sel));
  EXPECT_NEAR(std::abs(ps.blocks()[0].value - moved), 0.0, 1e-12);
  EXPECT_TRUE(ps.is_boundary(ps.blocks()[0].size));
  const DenseMatrix<T> qs = product(ps.vectors(), ps.form());
  DenseMatrix<T> qsqh(m, m);
  hs::lapack::gemm('N', 'C', static_cast<fint>(m), static_cast<fint>(m), static_cast<fint>(m), T(1),
                   qs.data(), static_cast<fint>(m), ps.vectors().data(), static_cast<fint>(m), T(0),
                   qsqh.data(), static_cast<fint>(m));
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < m; ++i) EXPECT_NEAR(std::abs(qsqh(i, j) - h(i, j)), 0.0, 1e-12);
  // Ritz vectors with coupling b: H y = theta y and estimate = |b^T y|.
  std::vector<T> b(static_cast<std::size_t>(m));
  for (index_t i = 0; i < m; ++i) b[i] = T(0.1 * (i + 1));
  const auto ritz = ps.ritz_vectors(b.data());
  for (const auto& blk : ps.blocks()) {
    std::vector<cplx> y(static_cast<std::size_t>(m));
    for (index_t i = 0; i < m; ++i) {
      if constexpr (hs::is_complex_v<T>) {
        y[i] = ritz.y(i, blk.start);
      } else {
        y[i] = blk.size == 2 ? cplx(ritz.y(i, blk.start), ritz.y(i, blk.start + 1))
                             : cplx(ritz.y(i, blk.start));
      }
    }
    cplx by = 0;
    double nrm = 0;
    for (index_t i = 0; i < m; ++i) {
      cplx hy = 0;
      for (index_t l = 0; l < m; ++l) hy += cplx(h(i, l)) * y[l];
      EXPECT_NEAR(std::abs(hy - blk.value * y[i]), 0.0, 1e-12);
      by += cplx(b[i]) * y[i];
      nrm += std::norm(y[i]);
    }
    EXPECT_NEAR(nrm, 1.0, 1e-14);
    EXPECT_NEAR(ritz.estimate[blk.start], std::abs(by), 1e-14);
  }
}

}  // namespace
