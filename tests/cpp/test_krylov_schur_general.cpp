// Tests of the general (non-Hermitian) Krylov-Schur solver hs::krylov_schur_general and its
// projected-problem helpers (src/eigen/general_*). Operators are small dense nonsymmetric
// matrices with known spectra and generalized pencils with indefinite B (via shift-invert), so
// every check is against an analytic or dense LAPACK reference.
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <vector>

#include "eigen/block_general.hpp"
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

/// All eigenvalues of a dense square (general, non-Hermitian) matrix: LAPACK gees on a copy.
template <class T>
std::vector<cplx> general_eigenvalues(DenseMatrix<T> a) {
  const auto n = static_cast<fint>(a.rows());
  DenseMatrix<T> q(n, n);
  std::vector<cplx> w(static_cast<std::size_t>(n));
  hs::lapack::gees(n, a.data(), n, w.data(), q.data(), n);
  return w;
}

/// A = X D X^{-1} with X = I + strength R / sqrt(n) (well conditioned for strength < 1, far from
/// normal).
template <class T>
DenseMatrix<T> similar_to(const DenseMatrix<T>& d, std::uint64_t seed, double strength = 0.5) {
  const index_t n = d.rows();
  DenseMatrix<T> x = random_matrix<T>(n, n, seed);
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) x(i, j) *= strength / std::sqrt(static_cast<double>(n));
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

template <class T>
double frobenius(const DenseMatrix<T>& a) {
  double s = 0.0;
  for (index_t j = 0; j < a.cols(); ++j)
    for (index_t i = 0; i < a.rows(); ++i) s += hs::abs2(a(i, j));
  return std::sqrt(s);
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

/// ||Op x - theta x||_2.
template <class T>
double true_residual(hs::LinearOperator<T>& op, const cplx& theta, const cplx* x) {
  const index_t n = op.size();
  std::vector<cplx> y = apply_complex(op, x);
  for (index_t k = 0; k < n; ++k) y[k] -= theta * x[k];
  return hs::linalg::norm2(n, y.data());
}

/// Scale of the convergence test for theta: |theta|, or the largest |Ritz value| for theta == 0.
/// The solver's zero level is ncv * eps * ||H||_F; in these tests every Ritz value above
/// 1e-10 * largest is far above that level and every exact zero far below it.
double test_scale(const cplx& theta, double largest) {
  return std::abs(theta) > 1e-10 * largest ? std::abs(theta) : largest;
}

/// Runs the enclosing scope with one OpenMP thread. linalg::dot/norm2 use an OpenMP reduction
/// whose partial sums libgomp combines in thread-arrival order, which is bitwise reproducible
/// only for up to two threads (interface request to the linalg owner). Tests comparing whole
/// solver runs bit for bit therefore run single-threaded.
class SingleThreadedOpenMP {
 public:
  SingleThreadedOpenMP() : saved_(omp_get_max_threads()) { omp_set_num_threads(1); }
  ~SingleThreadedOpenMP() { omp_set_num_threads(saved_); }
  SingleThreadedOpenMP(const SingleThreadedOpenMP&) = delete;
  SingleThreadedOpenMP& operator=(const SingleThreadedOpenMP&) = delete;

 private:
  int saved_;
};

/// Checks every structural promise of the GeneralEigResult contract, the converged prefix and
/// its invariant subspace, and the estimates against true residuals. op_norm is an upper bound
/// of ||Op||_2; the estimates may drift from the true residuals by O(eps ||Op||) per restart.
template <class T>
void check_contract(hs::LinearOperator<T>& op, const hs::GeneralEigResult<T>& r, Which which,
                    const hs::KrylovSchurOptions<T>& opt, double op_norm) {
  const index_t n = op.size();
  const index_t nev = opt.nev;
  const double tol = opt.tol;
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
  EXPECT_EQ(r.history.back().nconv, r.nconv);
  // A partial result names the reason: some wanted value or Schur block above the tolerance.
  EXPECT_TRUE(r.nconv == m || r.history.back().max_estimate > tol);

  // ritz_all: all ncv Ritz values of the final basis (auto ncv rule of eigen.hpp), in ranking
  // order, conjugate partners adjacent; theta is taken from them. Block mode: the final basis is
  // between nev + 1 and the layout's ncv vectors (whole blocks, early stop), or the whole space.
  const index_t basis = std::min(n, opt.ncv == 0 ? std::max(2 * nev + 1, nev + 16) : opt.ncv);
  const auto layout = opt.block_size > 1
                          ? hs::eigen_detail::block_general_layout(n, nev, basis, opt.ncv == 0,
                                                                   opt.block_size,
                                                                   hs::is_complex_v<T>)
                          : hs::eigen_detail::BlockGeneralLayout{};
  if (!layout.feasible) {
    ASSERT_EQ(static_cast<index_t>(r.ritz_all.size()), opt.block_size > 1 ? n : basis);
  } else {
    ASSERT_GT(static_cast<index_t>(r.ritz_all.size()), nev);
    ASSERT_LE(static_cast<index_t>(r.ritz_all.size()), layout.ncv);
  }
  for (std::size_t i = 0; i < r.ritz_all.size(); ++i) {
    if (i + 1 < r.ritz_all.size()) {
      EXPECT_LE(rank_key(r.ritz_all[i], which), rank_key(r.ritz_all[i + 1], which)) << i;
    }
    if (!hs::is_complex_v<T> && r.ritz_all[i].imag() != 0.0) {
      ASSERT_GT(r.ritz_all[i].imag(), 0.0) << i;
      ASSERT_LT(i + 1, r.ritz_all.size());
      EXPECT_EQ(r.ritz_all[i + 1], std::conj(r.ritz_all[i]));
      ++i;
    }
  }
  for (const cplx& t : r.theta) {
    EXPECT_NE(std::find(r.ritz_all.begin(), r.ritz_all.end(), t), r.ritz_all.end()) << t;
  }
  double largest = 0.0;
  for (const cplx& t : r.ritz_all) largest = std::max(largest, std::abs(t));
  const double rounding = (100.0 + static_cast<double>(r.restarts)) * kEps * op_norm;

  double scale2 = 0.0;  // sum of test_scale^2 over [0, nconv)
  for (index_t i = 0; i < m; ++i) {
    // [0, nconv) converged; theta as a whole in ranking order ([0, nconv) is a ranking prefix).
    if (i < r.nconv) {
      const double s = test_scale(r.theta[i], largest);
      EXPECT_LE(r.estimate[i], tol * s) << i;
      scale2 += s * s;
    }
    if (i + 1 < m) {
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
      if (i + 1 < r.nconv) scale2 += std::pow(test_scale(r.theta[i + 1], largest), 2);
      ++i;
    }
  }
  // Unit vectors; true residuals agree with the estimates up to the rounding drift.
  for (index_t i = 0; i < m; ++i) {
    const cplx* x = r.vectors.col(i);
    EXPECT_NEAR(hs::linalg::norm2(n, x), 1.0, 1e-13);
    EXPECT_NEAR(true_residual(op, r.theta[i], x), r.estimate[i], rounding) << "pair " << i;
  }
  // Orthonormal Schur basis (each compression V <- V Q costs O(eps) of orthogonality), and a
  // quasi-triangular S without a cut 2x2 block.
  const double orthogonality = 1e-12 + 10.0 * static_cast<double>(r.restarts) * kEps;
  DenseMatrix<T> gram(m, m);
  hs::linalg::gram(n, m, r.schur_basis.data(), n, r.schur_basis.data(), n, gram.data(), m);
  for (index_t j = 0; j < m; ++j) {
    for (index_t i = 0; i < m; ++i) {
      EXPECT_NEAR(std::abs(gram(i, j) - T(i == j ? 1.0 : 0.0)), 0.0, orthogonality);
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
  expect_same_values(general_eigenvalues(s), r.theta, 1e-10 * largest);
  if (r.nconv > 0) {
    DenseMatrix<T> lead(r.nconv, r.nconv);
    for (index_t j = 0; j < r.nconv; ++j)
      for (index_t i = 0; i < r.nconv; ++i) lead(i, j) = s(i, j);
    expect_same_values(general_eigenvalues(lead),
                       std::vector<cplx>(r.theta.begin(), r.theta.begin() + r.nconv),
                       1e-10 * largest);
    // The leading nconv Schur vectors span an invariant subspace of Op + E with
    // ||E||_2 = ||Op Q1 - Q1 S11||_2 <= tol * ||scale(theta[0:nconv))||_2.
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
    EXPECT_LE(std::sqrt(res2), tol * std::sqrt(scale2) + rounding);
  }
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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
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
    check_contract(op, r, which, opt, frobenius(a));
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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));

  // nev = 3 cuts before the pair: m == nev.
  opt.nev = 3;
  hs::test::DenseOperator<double> op3(a);
  const auto r3 = hs::krylov_schur_general(op3, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  EXPECT_EQ(r3.theta.size(), 3u);
  check_contract(op3, r3, Which::LargestMagnitude, opt, frobenius(a));
}

TEST(KrylovSchurGeneral, ConvergedPairBehindUnconvergedValueIsNotCounted) {
  // Wanted for nev = 3 by |theta|: 1, 0.9997, then the pair 0.5 +- 0.6i straddling the cut
  // (m = 4). The isolated pair converges restarts before the close values 1 and 0.9997 do, so
  // stopping early leaves converged values ranked behind unconverged ones. They must not count:
  // otherwise the pair plus one of the two reaches nconv = nev and reports success while the
  // best value is still unconverged.
  std::vector<cplx> values{{1.0, 0}, {0.9997, 0}, {0.5, 0.6}};
  for (index_t i = 0; i < 60; ++i)
    values.emplace_back((i % 2 ? -1.0 : 1.0) * (0.05 + 0.65 * static_cast<double>(i) / 59.0), 0.0);
  std::vector<cplx> all;
  const DenseMatrix<double> a = similar_to(block_diagonal<double>(values, &all), 60);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 3;
  opt.ncv = 10;
  opt.tol = 1e-12;
  opt.seed = 2;
  hs::test::DenseOperator<double> full_op(a);
  const auto full = hs::krylov_schur_general(full_op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(full.converged);
  ASSERT_EQ(full.theta.size(), 4u);
  index_t behind = 0;  // stops with a converged value ranked behind an unconverged one
  for (index_t stop = 0; stop <= full.restarts; ++stop) {
    hs::test::DenseOperator<double> op(a);
    opt.max_restarts = stop;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
    const auto m = static_cast<index_t>(r.theta.size());
    bool all_pass = true;
    index_t first_fail = m;
    for (index_t i = 0; i < m; ++i) {
      const bool pass = r.estimate[i] <= opt.tol * std::abs(r.theta[i]);
      all_pass = all_pass && pass;
      if (!pass && first_fail == m) first_fail = i;
      if (pass && first_fail < i) ++behind;
    }
    EXPECT_LE(r.nconv, first_fail) << "stop " << stop;
    if (r.converged) {
      EXPECT_TRUE(all_pass) << "stop " << stop << ": success with an open value";
    }
  }
  EXPECT_GT(behind, 0) << "the run never stopped in the critical state";
}

// ------------------------------------------------------------- generalized indefinite pencils

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
  const DenseMatrix<T> a = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 21));
  // B Hermitian and indefinite: diagonal of both signs plus a Hermitian perturbation.
  DenseMatrix<T> b = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 22));
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) b(i, j) *= 0.15;
    b(j, j) += T((j % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.01 * j));
  }
  const double sigma = 0.137;
  const std::vector<cplx> ref = general_eigenvalues(solve(b, a));  // eig(B^{-1} A)
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
  check_contract(op, r, Which::LargestMagnitude, opt, shift_invert_norm(a, b, sigma));
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
  const std::vector<cplx> all = general_eigenvalues(solve(w.b, w.a));
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
  check_contract(op, r, Which::LargestMagnitude, opt, shift_invert_norm(w.a, w.b, sigma));
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
  check_contract(op, r, Which::LargestMagnitude, opt, shift_invert_norm(w.a, w.b, sigma));
}

// ----------------------------------------------------------------------- behaviour/contract

TYPED_TEST(KrylovSchurGeneralTyped, DeterministicForFixedSeed) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(50), &all), 31);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 16;
  hs::test::DenseOperator<T> op1(a), op2(a), op3(a), op4(a);
  std::vector<cplx> theta;
  {
    // Bit for bit only where linalg's reductions are (see SingleThreadedOpenMP).
    SingleThreadedOpenMP serial;
    const auto r1 = hs::krylov_schur_general(op1, Which::LargestMagnitude, opt);
    const auto r2 = hs::krylov_schur_general(op2, Which::LargestMagnitude, opt);
    ASSERT_TRUE(r1.converged);
    EXPECT_EQ(r1.theta, r2.theta);
    EXPECT_EQ(r1.estimate, r2.estimate);
    EXPECT_EQ(r1.restarts, r2.restarts);
    EXPECT_EQ(r1.op_applies, r2.op_applies);
    for (index_t j = 0; j < r1.vectors.cols(); ++j)
      for (index_t i = 0; i < r1.vectors.rows(); ++i) ASSERT_EQ(r1.vectors(i, j), r2.vectors(i, j));
    theta = r1.theta;
  }
  // With the ambient thread count (any order of the partial sums) the rounding differs, and the
  // run agrees within the accuracy of the converged values.
  const auto r3 = hs::krylov_schur_general(op3, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  expect_same_values(r3.theta, theta, 1e-10);
  // Another seed takes another path to the same eigenvalues.
  opt.seed = 7;
  const auto r4 = hs::krylov_schur_general(op4, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r4.converged);
  expect_same_values(r4.theta, theta, 1e-10);
}

TYPED_TEST(KrylovSchurGeneralTyped, InvalidOptionsThrow) {
  using T = TypeParam;
  hs::test::DiagonalOperator<T> op(std::vector<T>(20, T(1)));
  auto expect_invalid = [](hs::LinearOperator<T>& target, const hs::KrylovSchurOptions<T>& o) {
    try {
      hs::krylov_schur_general(target, Which::LargestMagnitude, o);
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
  expect_invalid(op, bad);
  bad = o, bad.nev = 0;
  expect_invalid(op, bad);
  bad = o, bad.nev = 21;
  expect_invalid(op, bad);
  bad = o, bad.ncv = 3;  // < nev + 2 (real) / nev + 1 (complex)
  expect_invalid(op, bad);
  if (!hs::is_complex_v<T>) {
    bad = o, bad.ncv = 4;
    expect_invalid(op, bad);
  }
  bad = o, bad.max_restarts = -1;
  expect_invalid(op, bad);
  for (const double tol : {-1e-3, std::nan(""), std::numeric_limits<double>::infinity()}) {
    bad = o, bad.tol = tol;
    expect_invalid(op, bad);
  }
  DenseMatrix<T> wrong(19, 1), zero(20, 1), no_column(20, 0), nan(20, 1), inf(20, 1);
  wrong(0, 0) = T(1);
  nan(0, 0) = T(1), nan(5, 0) = T(std::nan(""));
  inf(0, 0) = T(1), inf(7, 0) = T(std::numeric_limits<double>::infinity());
  for (const DenseMatrix<T>* initial : {&wrong, &zero, &no_column, &nan, &inf}) {
    bad = o, bad.initial = initial;
    expect_invalid(op, bad);
  }
  EXPECT_EQ(op.applications(), 0);
  hs::test::DiagonalOperator<T> empty(std::vector<T>{});
  expect_invalid(empty, o);
  // ncv larger than n is clamped, not rejected.
  hs::KrylovSchurOptions<T> big = o;
  big.ncv = 100;
  EXPECT_TRUE(hs::krylov_schur_general(op, Which::LargestMagnitude, big).converged);
}

/// diag(1 / (1 + i)) whose output entry 3 is replaced by `value` on the bad_at-th application.
template <class T>
class PoisonedOperator final : public hs::LinearOperator<T> {
 public:
  PoisonedOperator(index_t n, double value, index_t bad_at)
      : n_(n), value_(value), bad_at_(bad_at) {}
  index_t size() const override { return n_; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
    for (index_t k = 0; k < p; ++k) {
      for (index_t i = 0; i < n_; ++i)
        y[i + k * ldy] = x[i + k * ldx] / (1.0 + static_cast<double>(i));
      if (++count_ == bad_at_) y[3 + k * ldy] = T(value_);
    }
  }

 private:
  index_t n_;
  double value_;
  index_t bad_at_;
  index_t count_ = 0;
};

TYPED_TEST(KrylovSchurGeneralTyped, NonFiniteOperatorOutputThrows) {
  using T = TypeParam;
  // NaN/Inf would otherwise be mistaken for a breakdown (first application) or reach gees.
  for (const double value : {std::nan(""), std::numeric_limits<double>::infinity()}) {
    for (const index_t bad_at : {index_t{1}, index_t{5}}) {
      PoisonedOperator<T> op(60, value, bad_at);
      hs::KrylovSchurOptions<T> opt;
      opt.nev = 4;
      opt.ncv = 12;
      try {
        hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
        ADD_FAILURE() << "value " << value << " at application " << bad_at << " not detected";
      } catch (const hs::Error& e) {
        EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
      }
      EXPECT_EQ(op.applications(), bad_at);
    }
  }
}

/// The result for Op / s from the result r for Op: every eigenvalue-sized quantity divided by s.
template <class T>
hs::GeneralEigResult<T> divided_by(hs::GeneralEigResult<T> r, double s) {
  for (cplx& t : r.theta) t /= s;
  for (cplx& t : r.ritz_all) t /= s;
  for (double& e : r.estimate) e /= s;
  for (index_t j = 0; j < r.schur_form.cols(); ++j)
    for (index_t i = 0; i < r.schur_form.rows(); ++i) r.schur_form(i, j) /= s;
  return r;
}

/// s * A for a dense A.
template <class T>
class ScaledOperator final : public hs::LinearOperator<T> {
 public:
  ScaledOperator(const DenseMatrix<T>& a, double s) : a_(a), s_(s) {}
  index_t size() const override { return a_.size(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
    a_.apply(x, ldx, y, ldy, p);
    for (index_t k = 0; k < p; ++k) hs::linalg::scal(size(), T(s_), y + k * ldy);
  }

 private:
  hs::test::DenseOperator<T> a_;
  double s_;
};

TYPED_TEST(KrylovSchurGeneralTyped, OperatorMagnitudeDoesNotMatter) {
  // linalg::norm2 is a plain sum of squares: without the solver's power-of-two scaling, operator
  // entries near 1e-170 underflow and near 1e+160 overflow, which looked like breakdowns.
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(50), &all), 81);
  const index_t n = a.rows();
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.ncv = 14;
  SingleThreadedOpenMP serial;  // bitwise comparisons below
  hs::test::DenseOperator<T> ref_op(a);
  const auto ref = hs::krylov_schur_general(ref_op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(ref.converged);
  for (const int e : {-560, 520}) {
    // A power-of-two multiple of A is the same iteration bit for bit, scaled back exactly.
    const double s = std::ldexp(1.0, e);
    ScaledOperator<T> op(a, s);
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_EQ(r.theta.size(), ref.theta.size()) << e;
    EXPECT_EQ(r.nconv, ref.nconv);
    EXPECT_EQ(r.restarts, ref.restarts);
    for (std::size_t i = 0; i < r.theta.size(); ++i) {
      EXPECT_EQ(r.theta[i], s * ref.theta[i]) << e << " " << i;
      EXPECT_EQ(r.estimate[i], s * ref.estimate[i]) << e << " " << i;
    }
    for (index_t j = 0; j < r.vectors.cols(); ++j)
      for (index_t i = 0; i < n; ++i) ASSERT_EQ(r.vectors(i, j), ref.vectors(i, j));
  }
  for (const double s : {1e-170, 1e+160}) {
    // Inexact multiples: the same eigenpairs up to rounding. The contract is checked on A,
    // where this test's own residual norms neither underflow nor overflow.
    ScaledOperator<T> op(a, s);
    const auto r = divided_by(hs::krylov_schur_general(op, Which::LargestMagnitude, opt), s);
    ASSERT_TRUE(r.converged) << s;
    expect_same_values(r.theta, ref.theta, 1e-10);
    hs::test::DenseOperator<T> unscaled(a);
    check_contract(unscaled, r, Which::LargestMagnitude, opt, frobenius(a));
  }
  // A start vector whose norm alone would overflow is accepted.
  DenseMatrix<T> huge = random_matrix<T>(n, 1, 5);
  hs::linalg::scal(n, T(1e200), huge.data());
  opt.initial = &huge;
  hs::test::DenseOperator<T> op(a);
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  expect_same_values(r.theta, ref.theta, 1e-10);
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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, EveryStopSatisfiesTheContract) {
  using T = TypeParam;
  // Stopping after each restart count gives partial results in which the passing ranking prefix
  // often does not lead the Schur form (an unconverged wanted value sits above a converged
  // better-ranked one). The prefix must then be moved to the top of S, and the Ritz data
  // recomputed, before its invariant-subspace test: [0, nconv) is the whole passing prefix, not
  // merely some valid one (certifying nothing would be safe but useless). The eigenbasis is well
  // conditioned, so the subspace test cannot reject a prefix whose pairs pass by a factor of 10.
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(60), &all), 12);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 14;
  hs::test::DenseOperator<T> full_op(a);
  const auto full = hs::krylov_schur_general(full_op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(full.converged);
  index_t partial = 0;  // stops with 0 < nconv < m
  for (index_t stop = 0; stop <= full.restarts; ++stop) {
    hs::test::DenseOperator<T> op(a);
    opt.max_restarts = stop;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    EXPECT_EQ(r.converged, stop == full.restarts) << "stop " << stop;
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
    // Leading pairs passing estimate <= factor * tol * |theta|; pairs share estimate and |theta|,
    // so the count never ends inside a pair.
    const auto m = static_cast<index_t>(r.theta.size());
    auto passing_prefix = [&](double factor) {
      index_t p = 0;
      while (p < m && r.estimate[p] <= factor * opt.tol * std::abs(r.theta[p])) ++p;
      return p;
    };
    EXPECT_LE(passing_prefix(0.1), r.nconv) << "stop " << stop;
    EXPECT_LE(r.nconv, passing_prefix(1.0)) << "stop " << stop;
    partial += r.nconv > 0 && r.nconv < m ? 1 : 0;
  }
  EXPECT_GT(partial, 0) << "no partial result: the sweep does not test the prefix";
}

TYPED_TEST(KrylovSchurGeneralTyped, IllConditionedEigenbasisNeedsInvariantSubspace) {
  using T = TypeParam;
  // Leading block [[1, 100], [0, 0.98]]: the eigenvectors of 1 and 0.98 are nearly parallel
  // (condition ~ 100 / 0.02), so both eigenvector estimates pass the tolerance while the Schur
  // vectors they define are still ~1e3 times further from invariant. Counting the pairs as
  // converged on their estimates alone left ||Op Q1 - Q1 S11|| near 6e-10 for tol = 1e-12.
  const index_t n = 52;
  DenseMatrix<T> d(n, n);
  d(0, 0) = T(1.0), d(0, 1) = T(100.0), d(1, 1) = T(0.98);
  for (index_t i = 2; i < n; ++i) d(i, i) = T(0.01 + 0.49 * static_cast<double>(i - 2) / 49.0);
  const DenseMatrix<T> a = similar_to(d, 66);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 2;
  opt.ncv = 12;
  opt.tol = 1e-12;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  expect_same_values(r.theta, {cplx(1.0), cplx(0.98)}, 1e-9);
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, ZeroEigenvalueUsesLargestRitzScale) {
  using T = TypeParam;
  // Wanted by LargestAlgebraic: 0, -0.1, -0.2. The computed Ritz value of the exact 0 is an error
  // of the size of its residual estimate or below, so a test against its own magnitude could
  // never pass: it must be recognized as zero (|theta| <= ncv eps ||H||_F) and tested against the
  // largest |Ritz value|. (tol = 1e-8 stops the run while that estimate is still far above
  // tol * |theta|, where the two tests differ.)
  std::vector<cplx> values;
  for (index_t i = 0; i < 60; ++i) values.emplace_back(-0.1 * static_cast<double>(i), 0.0);
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(values, &all), 72);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 3;
  opt.tol = 1e-8;
  const auto r = hs::krylov_schur_general(op, Which::LargestAlgebraic, opt);
  ASSERT_TRUE(r.converged);
  double largest = 0.0;
  for (const cplx& t : r.ritz_all) largest = std::max(largest, std::abs(t));
  EXPECT_LE(std::abs(r.theta[0]), 1e-13 * largest);
  EXPECT_GT(r.estimate[0], opt.tol * std::abs(r.theta[0])) << "the zero rule was not needed";
  expect_same_values(r.theta, {cplx(0.0), cplx(-0.1), cplx(-0.2)}, 1e-6);
  check_contract(op, r, Which::LargestAlgebraic, opt, frobenius(a));
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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
}

TEST(KrylovSchurGeneral, ManyRestartsKeepEstimatesNearTrueResiduals) {
  // A clustered, strongly non-normal spectrum with the smallest basis: thousands of restarts.
  // Each compression adds O(eps ||Op||) of rounding to the Krylov-Schur relation, so the
  // estimates drift from the true residuals by that much per restart, and no faster
  // (check_contract allows (100 + restarts) eps ||Op||_F).
  std::vector<cplx> values;
  for (index_t i = 0; i < 6; ++i) values.emplace_back(1.0 - 2e-4 * static_cast<double>(i), 0.0);
  values.emplace_back(0.99, 0.02);
  for (index_t i = 0; i < 200; ++i)
    values.emplace_back((i % 2 ? -1.0 : 1.0) * (0.01 + 0.975 * static_cast<double>(i) / 199.0),
                        0.0);
  std::vector<cplx> all;
  const DenseMatrix<double> a = similar_to(block_diagonal<double>(values, &all), 67, 0.9);
  hs::test::DenseOperator<double> op(a);
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 6;
  opt.ncv = 8;
  opt.max_restarts = 3000;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  EXPECT_EQ(r.restarts, opt.max_restarts);
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
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
  check_contract(op, r, Which::LargestMagnitude, opt, 1.0 + 0.5 * n);
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
  check_contract(op, r, Which::SmallestAlgebraic, opt, frobenius(a));

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
  check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
}

TYPED_TEST(KrylovSchurGeneralTyped, PhaseTimesAccountForTheRun) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(60), &all), 61);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.ncv = 12;
  SingleThreadedOpenMP serial;  // the timed and plain runs are compared bit for bit
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

// ------------------------------------------------------------------------------ block mode

/// Block sizes exercised by the block-mode tests (p = 1 is the single-vector solver above).
const std::vector<index_t> kBlockSizes{2, 4, 8};

TYPED_TEST(KrylovSchurGeneralTyped, BlockKnownSpectrumMatchesDenseReference) {
  // Every block size finds the same wanted set as the dense reference (real T: conjugate pairs
  // never split, m = nev + 1 when one straddles), in whole blocks of operator applications.
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(150), &all), 11);
  for (const Which which :
       {Which::LargestMagnitude, Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
    const index_t nev = which == Which::LargestMagnitude ? 9 : 5;
    for (const index_t p : kBlockSizes) {
      hs::test::DenseOperator<T> op(a);
      hs::KrylovSchurOptions<T> opt;
      opt.nev = nev;
      opt.tol = 1e-12;
      opt.block_size = p;
      const auto r = hs::krylov_schur_general(op, which, opt);
      ASSERT_TRUE(r.converged) << "p=" << p;
      EXPECT_EQ(r.op_applies, op.applications());
      EXPECT_EQ(r.op_applies % p, 0) << "p=" << p;
      expect_same_values(r.theta, expected_wanted(all, which, nev, !hs::is_complex_v<T>), 1e-10);
      check_contract(op, r, which, opt, frobenius(a));
    }
  }
}

TEST(KrylovSchurGeneral, BlockConjugatePairAtNevBoundaryGivesNevPlusOne) {
  std::vector<cplx> values{{10, 0}, {-9, 0}, {8, 0}, {7, 2}, {5, 0}, {-4.5, 0}, {3, 1}};
  for (index_t i = 0; i < 80; ++i)
    values.emplace_back((i % 2 ? -1.0 : 1.0) * (2.0 - 0.02 * i), 0.0);
  std::vector<cplx> all;
  const DenseMatrix<double> a = similar_to(block_diagonal<double>(values, &all), 13);
  for (const index_t p : kBlockSizes) {
    hs::test::DenseOperator<double> op(a);
    hs::KrylovSchurOptions<double> opt;
    opt.nev = 4;
    opt.block_size = p;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_TRUE(r.converged) << "p=" << p;
    ASSERT_EQ(r.theta.size(), 5u);
    EXPECT_NEAR(r.theta[3].real(), 7.0, 1e-10);
    EXPECT_NEAR(r.theta[3].imag(), 2.0, 1e-10);
    EXPECT_EQ(r.theta[4], std::conj(r.theta[3]));
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockFindsEveryCopyOfAMultipleEigenvalue) {
  // The dominant eigenvalue 1 has multiplicity 3 (non-defective: A = X D X^{-1}). A single-vector
  // Krylov space holds one copy in exact arithmetic (further ones only through rounding); a start
  // block of p >= 3 random columns has generic components in all three eigenvectors, so the
  // block solver returns three copies whose Schur vectors span the 3-dimensional eigenspace
  // (check_contract: orthonormal Schur basis, invariant leading block with spectrum theta).
  using T = TypeParam;
  std::vector<cplx> values{{1, 0}, {1, 0}, {1, 0}, {0.9, 0}, {-0.8, 0}};
  for (index_t i = 0; i < 120; ++i)
    values.emplace_back((i % 2 ? -1.0 : 1.0) * (0.6 - 0.004 * static_cast<double>(i)), 0.0);
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(values, &all), 17);
  for (const index_t p : {index_t{4}, index_t{8}}) {
    hs::test::DenseOperator<T> op(a);
    hs::KrylovSchurOptions<T> opt;
    opt.nev = 5;
    opt.tol = 1e-12;
    opt.block_size = p;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_TRUE(r.converged) << "p=" << p;
    expect_same_values(r.theta, {cplx(1), cplx(1), cplx(1), cplx(0.9), cplx(-0.8)}, 1e-10);
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockGhiepShiftInvertMatchesDenseReference) {
  using T = TypeParam;
  const index_t n = 240;
  const DenseMatrix<T> a = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 21));
  DenseMatrix<T> b = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 22));
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) b(i, j) *= 0.15;
    b(j, j) += T((j % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.01 * j));
  }
  const double sigma = 0.137;
  const std::vector<cplx> ref = general_eigenvalues(solve(b, a));
  double scale = 0.0;
  for (const cplx& l : ref) scale = std::max(scale, std::abs(l));
  for (const index_t p : kBlockSizes) {
    hs::test::DenseShiftInvertOperator<T> op(a, &b, sigma);
    hs::KrylovSchurOptions<T> opt;
    opt.nev = 10;
    opt.tol = 1e-12;
    opt.block_size = p;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_TRUE(r.converged) << "p=" << p;
    expect_nearest(to_lambda(r.theta, sigma), ref, sigma, opt.nev, 1e-9 * scale);
    check_contract(op, r, Which::LargestMagnitude, opt, shift_invert_norm(a, b, sigma));
  }
}

TEST(KrylovSchurGeneral, BlockWaveguideNeverConvergesClusterCopies) {
  // The driver's zero-cluster placements (DESIGN §3.6-9) with blocks: a shift inside the nonzero
  // spectrum (LargestMagnitude) and one-sided runs at sigma+ > 0 (LargestAlgebraic) and sigma- < 0
  // (SmallestAlgebraic), where the cluster's theta = -1/sigma sits at the unwanted end. A block of
  // p vectors carries p independent components of the 48-dimensional null space, but no cluster
  // copy may appear among the returned (wanted) values, which must be the reference ones.
  const Waveguide w = waveguide_pencil(8, 6, 2.0);
  const WaveguideReference ref = waveguide_reference(w);
  std::vector<double> positive, negative;
  for (const cplx& l : ref.nonzero) (l.real() > 0.0 ? positive : negative).push_back(l.real());
  std::sort(positive.begin(), positive.end());
  std::sort(negative.begin(), negative.end(), std::greater<>());
  ASSERT_GE(positive.size(), 12u);
  ASSERT_GE(negative.size(), 4u);
  struct Run {
    double sigma;
    Which which;
    index_t nev;
    std::vector<double> want;  // the nev wanted eigenvalues
  };
  const double mid = 0.5 * (positive[9] + positive[10]);
  const std::vector<cplx> near = sorted_by_distance(ref.nonzero, mid);
  std::vector<double> near6;
  for (index_t i = 0; i < 6; ++i) near6.push_back(near[i].real());
  const std::vector<Run> runs{{mid, Which::LargestMagnitude, 6, near6},
                              {0.5 * positive[0], Which::LargestAlgebraic, 4,
                               std::vector<double>(positive.begin(), positive.begin() + 4)},
                              {0.5 * negative[0], Which::SmallestAlgebraic, 3,
                               std::vector<double>(negative.begin(), negative.begin() + 3)}};
  for (const Run& run : runs) {
    for (const index_t p : {index_t{1}, index_t{4}, index_t{8}}) {
      hs::test::DenseShiftInvertOperator<double> op(w.a, &w.b, run.sigma);
      hs::KrylovSchurOptions<double> opt;
      opt.nev = run.nev;
      opt.block_size = p;
      const auto r = hs::krylov_schur_general(op, run.which, opt);
      ASSERT_TRUE(r.converged) << "sigma=" << run.sigma << " p=" << p;
      const std::vector<cplx> lambda = to_lambda(r.theta, run.sigma);
      for (const cplx& l : lambda) {
        EXPECT_GT(std::abs(l), 1e-6 * ref.scale) << "cluster copy, sigma=" << run.sigma
                                                 << " p=" << p;
        EXPECT_LE(std::abs(l.imag()), 1e-12 * ref.scale) << l;
      }
      std::vector<cplx> want;
      for (const double x : run.want) want.emplace_back(x);
      expect_same_values(lambda, want, 1e-10 * ref.scale);
      check_contract(op, r, run.which, opt, shift_invert_norm(w.a, w.b, run.sigma));
    }
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockBreakdownInInvariantStartBlockRecovers) {
  // Diagonal operator and a start block of 4 columns inside the 3-dimensional invariant subspace
  // span(e2, e10, e30): the start block itself is rank deficient and its first block step is
  // dependent on the basis; both are continued with random replacements.
  using T = TypeParam;
  const index_t n = 80;
  std::vector<T> diag(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) diag[i] = T(1.0 + 0.5 * i);
  hs::test::DiagonalOperator<T> op(diag);
  DenseMatrix<T> start(n, 4);
  for (index_t c = 0; c < 4; ++c) {
    const double x = static_cast<double>(c);
    start(2, c) = T(1.0 + x);
    start(10, c) = T(-2.0 + 0.5 * x);
    start(30, c) = T(0.5 - 0.25 * x * x);
  }
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 5;
  opt.ncv = 12;
  opt.initial = &start;
  opt.block_size = 4;
  const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r.converged);
  for (index_t i = 0; i < opt.nev; ++i) {
    EXPECT_NEAR(r.theta[i].real(), 1.0 + 0.5 * static_cast<double>(n - 1 - i), 1e-10);
    EXPECT_NEAR(r.theta[i].imag(), 0.0, hs::is_complex_v<T> ? 1e-12 : 0.0);
  }
  check_contract(op, r, Which::LargestMagnitude, opt, 1.0 + 0.5 * static_cast<double>(n));
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockSpaceTooSmallIsOneExactCycle) {
  // n < 2 (ncv + p) (and p > n / 8): one single-vector cycle over the whole space.
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(32), &all), 52);
  const index_t n = a.rows();
  for (const index_t p : {index_t{8}, std::numeric_limits<index_t>::max()}) {
    hs::test::DenseOperator<T> op(a);
    hs::KrylovSchurOptions<T> opt;
    opt.nev = 4;
    opt.block_size = p;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_TRUE(r.converged);
    EXPECT_EQ(r.restarts, 0);
    EXPECT_EQ(r.op_applies, n);
    expect_same_values(
        r.theta,
        expected_wanted(all, Which::LargestMagnitude, opt.nev, !hs::is_complex_v<T>), 1e-10);
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockInvalidOptionsThrow) {
  using T = TypeParam;
  hs::test::DiagonalOperator<T> op(std::vector<T>(64, T(1)));
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
  for (const index_t p : {index_t{0}, index_t{-3}}) {
    auto bad = o;
    bad.block_size = p;
    expect_invalid(bad);
  }
  // Block mode uses the first min(p, cols) start columns: a NaN in column 1 is rejected with
  // p = 2 and ignored with p = 1 (only column 0 is used).
  DenseMatrix<T> start(64, 2);
  start(0, 0) = T(1);
  start(5, 1) = T(std::nan(""));
  auto bad = o;
  bad.initial = &start;
  bad.block_size = 2;
  expect_invalid(bad);
  EXPECT_EQ(op.applications(), 0);
  bad.block_size = 1;
  EXPECT_NO_THROW(hs::krylov_schur_general(op, Which::LargestMagnitude, bad));
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockNonFiniteOperatorOutputThrows) {
  using T = TypeParam;
  for (const double value : {std::nan(""), std::numeric_limits<double>::infinity()}) {
    for (const index_t bad_at : {index_t{1}, index_t{5}}) {
      PoisonedOperator<T> op(60, value, bad_at);
      hs::KrylovSchurOptions<T> opt;
      opt.nev = 4;
      opt.ncv = 12;
      opt.block_size = 2;
      try {
        hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
        ADD_FAILURE() << "value " << value << " at application " << bad_at << " not detected";
      } catch (const hs::Error& e) {
        EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
      }
      EXPECT_EQ(op.applications(), (bad_at + 1) / 2 * 2);  // the whole block holding it
    }
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockDeterministicAndMagnitudeIndependent) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(90), &all), 31);
  const index_t n = a.rows();
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.block_size = 4;
  SingleThreadedOpenMP serial;  // bitwise comparisons
  hs::test::DenseOperator<T> op1(a), op2(a);
  const auto r1 = hs::krylov_schur_general(op1, Which::LargestMagnitude, opt);
  const auto r2 = hs::krylov_schur_general(op2, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r1.converged);
  EXPECT_EQ(r1.theta, r2.theta);
  EXPECT_EQ(r1.estimate, r2.estimate);
  EXPECT_EQ(r1.op_applies, r2.op_applies);
  for (index_t j = 0; j < r1.vectors.cols(); ++j)
    for (index_t i = 0; i < n; ++i) ASSERT_EQ(r1.vectors(i, j), r2.vectors(i, j));
  // 2^e Op is the same iteration bit for bit, scaled back exactly.
  for (const int e : {-560, 520}) {
    const double s = std::ldexp(1.0, e);
    ScaledOperator<T> op(a, s);
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    ASSERT_EQ(r.theta.size(), r1.theta.size());
    EXPECT_EQ(r.op_applies, r1.op_applies);
    for (std::size_t i = 0; i < r.theta.size(); ++i) {
      EXPECT_EQ(r.theta[i], s * r1.theta[i]) << e << " " << i;
      EXPECT_EQ(r.estimate[i], s * r1.estimate[i]) << e << " " << i;
    }
    for (index_t j = 0; j < r.vectors.cols(); ++j)
      for (index_t i = 0; i < n; ++i) ASSERT_EQ(r.vectors(i, j), r1.vectors(i, j));
  }
  // Another seed, or a user start column (the rest of the block random), reach the same values.
  opt.seed = 7;
  hs::test::DenseOperator<T> op3(a);
  const auto r3 = hs::krylov_schur_general(op3, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  expect_same_values(r3.theta, r1.theta, 1e-10);
  DenseMatrix<T> start = random_matrix<T>(n, 1, 5);
  hs::linalg::scal(n, T(1e200), start.data());  // its norm alone would overflow
  opt.initial = &start;
  hs::test::DenseOperator<T> op4(a);
  const auto r4 = hs::krylov_schur_general(op4, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r4.converged);
  expect_same_values(r4.theta, r1.theta, 1e-10);
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockEveryStopSatisfiesTheContract) {
  // As EveryStopSatisfiesTheContract, in block mode: partial results after each restart count
  // (probes end cycles early in between) keep the certified prefix on top of S.
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(120), &all), 12);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 16;
  opt.block_size = 4;
  hs::test::DenseOperator<T> full_op(a);
  const auto full = hs::krylov_schur_general(full_op, Which::LargestMagnitude, opt);
  ASSERT_TRUE(full.converged);
  ASSERT_GT(full.restarts, 0) << "the sweep needs restarts";
  for (index_t stop = 0; stop <= full.restarts; ++stop) {
    hs::test::DenseOperator<T> op(a);
    opt.max_restarts = stop;
    const auto r = hs::krylov_schur_general(op, Which::LargestMagnitude, opt);
    EXPECT_EQ(r.converged, stop == full.restarts) << "stop " << stop;
    check_contract(op, r, Which::LargestMagnitude, opt, frobenius(a));
  }
}

TYPED_TEST(KrylovSchurGeneralTyped, BlockPhaseTimesAccountForTheRun) {
  using T = TypeParam;
  std::vector<cplx> all;
  const DenseMatrix<T> a = similar_to(block_diagonal<T>(mixed_spectrum<T>(100), &all), 61);
  hs::test::DenseOperator<T> op(a);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.block_size = 4;
  hs::eigen_detail::GeneralPhaseTimes times;
  const auto r =
      hs::eigen_detail::krylov_schur_general_timed(op, Which::LargestMagnitude, opt, &times);
  ASSERT_TRUE(r.converged);
  for (double t : {times.apply_s, times.orthogonalize_s, times.projected_s, times.update_s})
    EXPECT_GE(t, 0.0);
  EXPECT_GT(times.total_s, 0.0);
  EXPECT_LE(times.apply_s + times.orthogonalize_s + times.projected_s + times.update_s,
            times.total_s * (1.0 + 1e-9));
}

TEST(KrylovSchurGeneral, BlockLayoutRules) {
  using hs::eigen_detail::block_general_layout;
  // Real T, nev = 18, auto ncv (single-vector 37), p = 8: spare = 20 -> 24, base = 40, + 4 blocks.
  auto lay = block_general_layout(100000, 18, 37, true, 8, false);
  EXPECT_TRUE(lay.feasible);
  EXPECT_EQ(lay.ncv, 72);
  EXPECT_EQ(lay.first_probe, 40);
  EXPECT_EQ(lay.keep, 40);  // 18 + 54 / 2 = 45 -> 40
  // Requested ncv: rounded up to whole blocks, at least spare + 2p.
  lay = block_general_layout(100000, 18, 20, false, 4, true);
  EXPECT_EQ(lay.ncv, 28);  // complex: spare = 19 -> 20, + 2 blocks
  lay = block_general_layout(100000, 4, 30, false, 4, false);
  EXPECT_EQ(lay.ncv, 32);
  EXPECT_EQ(lay.keep, 16);  // 4 + 28 / 2 = 18 -> 16
  // Small spaces: the auto ncv keeps 2 (ncv + p) <= n; infeasible below.
  lay = block_general_layout(100, 4, 20, true, 4, false);
  EXPECT_TRUE(lay.feasible);
  EXPECT_LE(2 * (lay.ncv + 4), 100);
  EXPECT_FALSE(block_general_layout(40, 4, 20, true, 8, false).feasible);
  EXPECT_FALSE(block_general_layout(60, 4, 60, false, 2, false).feasible);  // ncv == n: exact
  EXPECT_FALSE(
      block_general_layout(1000, 4, 20, true, std::numeric_limits<index_t>::max(), false).feasible);
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
