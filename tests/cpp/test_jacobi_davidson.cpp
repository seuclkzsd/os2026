// Tests of hs::jacobi_davidson (include/hsolve/eigen_iterative.hpp): interior targets with exact,
// perturbed ("single-precision-like") and real single-precision MUMPS preconditioners, harmonic
// and standard extraction, locking of multiple eigenvalues, restarts, both inner solvers and the
// result contract, for both scalar types.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::index_t;
using hs::IterativeEigResult;
using hs::JdExtraction;
using hs::JdInnerSolver;
using hs::JdOptions;
using hs::LinearOperator;
using hs::test::DenseOperator;

// ---------------------------------------------------------------------------------- helpers

/// Dense inverse (n <= 2000) by LU.
template <class T>
DenseMatrix<T> dense_inverse(DenseMatrix<T> k) {
  const auto n = static_cast<hs::lapack::fint>(k.rows());
  std::vector<hs::lapack::fint> piv(static_cast<std::size_t>(n));
  EXPECT_EQ(hs::lapack::getrf(n, k.data(), n, piv.data()), 0);
  DenseMatrix<T> inv(n, n);
  for (index_t i = 0; i < n; ++i) inv(i, i) = T(1);
  hs::lapack::getrs(n, n, k.data(), n, piv.data(), inv.data(), n);
  return inv;
}

/// Dense A - sigma B (B = I when b == nullptr).
template <class T>
DenseMatrix<T> shifted(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, double sigma) {
  DenseMatrix<T> k = hs::test::to_dense(a);
  if (b) {
    const auto bd = hs::test::to_dense(*b);
    for (index_t j = 0; j < k.cols(); ++j)
      for (index_t i = 0; i < k.rows(); ++i) k(i, j) -= T(sigma) * bd(i, j);
  } else {
    for (index_t i = 0; i < k.rows(); ++i) k(i, i) -= T(sigma);
  }
  return k;
}

/// Every entry rounded to single precision (real and imaginary parts): the matrix a
/// single-precision factorization actually factorizes.
template <class T>
DenseMatrix<T> round_to_float(DenseMatrix<T> k) {
  for (index_t i = 0; i < k.rows() * k.cols(); ++i) {
    if constexpr (hs::is_complex_v<T>) {
      k.data()[i] = T(static_cast<float>(k.data()[i].real()), static_cast<float>(k.data()[i].imag()));
    } else {
      k.data()[i] = static_cast<float>(k.data()[i]);
    }
  }
  return k;
}

/// The values of `spectrum` nearest tau, nearest first.
std::vector<double> nearest(std::vector<double> spectrum, double tau, index_t k) {
  std::stable_sort(spectrum.begin(), spectrum.end(),
                   [tau](double x, double y) { return std::abs(x - tau) < std::abs(y - tau); });
  spectrum.resize(static_cast<std::size_t>(k));
  return spectrum;
}

/// Compares as multisets (the order of equidistant values may differ).
void expect_same_values(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
}

/// max |X^H B Y - delta|.
template <class T>
double gram_deviation(const DenseMatrix<T>& x, LinearOperator<T>* b, const DenseMatrix<T>& y, bool identity) {
  const index_t n = x.rows();
  DenseMatrix<T> by(n, y.cols());
  if (b) {
    b->apply(y.data(), n, by.data(), n, y.cols());
  } else {
    by = y;
  }
  double dev = 0.0;
  for (index_t j = 0; j < y.cols(); ++j)
    for (index_t i = 0; i < x.cols(); ++i)
      dev = std::max(dev, std::abs(hs::linalg::dot(n, x.col(i), by.col(j)) - T(identity && i == j ? 1.0 : 0.0)));
  return dev;
}

/// Result contract: sizes, nearest-first order per group, B-orthonormality, estimates equal to
/// recomputed backward errors, bookkeeping.
template <class T>
void check_contract(LinearOperator<T>& a, LinearOperator<T>* b, const IterativeEigResult<T>& r, double tau,
                    index_t nev, double tol) {
  const index_t m = static_cast<index_t>(r.theta.size());
  ASSERT_LE(m, nev);
  if (r.converged) {
    ASSERT_EQ(m, nev);
  }
  ASSERT_EQ(static_cast<index_t>(r.estimate.size()), m);
  ASSERT_EQ(static_cast<index_t>(r.residual.size()), m);
  ASSERT_EQ(r.vectors.rows(), a.size());
  ASSERT_EQ(r.vectors.cols(), m);
  EXPECT_EQ(r.converged, r.nconv >= nev);
  for (index_t i = 0; i < m; ++i) {
    if (i < r.nconv) {
      EXPECT_LE(r.estimate[i], tol) << "pair " << i;
    } else {
      EXPECT_GT(r.estimate[i], tol) << "pair " << i;
    }
    if (i != 0 && i != r.nconv) {
      EXPECT_LE(std::abs(r.theta[i - 1] - tau), std::abs(r.theta[i] - tau)) << i;
    }
  }
  EXPECT_LT(gram_deviation<T>(r.vectors, b, r.vectors, true), 1e-11);
  const index_t n = a.size();
  DenseMatrix<T> ax(n, m), bx(n, m);
  a.apply(r.vectors.data(), n, ax.data(), n, m);
  if (b) {
    b->apply(r.vectors.data(), n, bx.data(), n, m);
  } else {
    bx = r.vectors;
  }
  for (index_t j = 0; j < m; ++j) {
    hs::linalg::axpy(n, T(-r.theta[j]), bx.col(j), ax.col(j));
    const double res = hs::linalg::norm2(n, ax.col(j));
    const double eta =
        res / ((r.norm_a + std::abs(r.theta[j]) * r.norm_b) * hs::linalg::norm2(n, r.vectors.col(j)));
    EXPECT_NEAR(eta, r.estimate[j], 1e-12 + 1e-6 * eta) << j;
    EXPECT_NEAR(res, r.residual[j], 1e-12 * (r.norm_a + std::abs(r.theta[j]) * r.norm_b) + 1e-6 * res) << j;
  }
  ASSERT_EQ(static_cast<index_t>(r.history.size()), r.iterations + 1);
  for (index_t c = 0; c <= r.iterations; ++c) EXPECT_EQ(r.history[c].restart, c);
  for (std::size_t c = 1; c < r.history.size(); ++c) EXPECT_GE(r.history[c].nconv, r.history[c - 1].nconv);
  // history counts locked pairs; the final check may also accept a not yet locked approximation.
  EXPECT_LE(r.history.back().nconv, r.nconv);
  if (r.converged) {
    EXPECT_EQ(r.history.back().max_estimate, 0.0);
  }
}

/// Interior test spectrum: 400 values in [-4, 6] with a cluster near tau = 0.37.
std::vector<double> interior_spectrum() {
  std::vector<double> w;
  for (index_t i = 0; i < 392; ++i) w.push_back(-4.0 + 10.0 * static_cast<double>(i) / 391.0 + 1e-3);
  for (double x : {0.30, 0.33, 0.36, 0.38, 0.41, 0.45, 0.452, 0.5}) w.push_back(x);
  std::sort(w.begin(), w.end());
  return w;
}

/// y = K^{-1} x through a sparse factorization (no B): the JD preconditioner of the memory-limited
/// use case.
template <class T>
class FactorSolveOperator final : public LinearOperator<T> {
 public:
  explicit FactorSolveOperator(hs::SparseFactorization<T>& f) : f_(f) {}
  index_t size() const override { return f_.order(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
    const index_t n = size();
    for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, y + k * ldy);
    f_.solve(y, p, ldy);
  }

 private:
  hs::SparseFactorization<T>& f_;
};

/// Number of values within tol of x.
index_t copies(const std::vector<double>& values, double x, double tol) {
  return static_cast<index_t>(
      std::count_if(values.begin(), values.end(), [&](double v) { return std::abs(v - x) <= tol; }));
}

template <class T>
class JacobiDavidsonTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(JacobiDavidsonTyped, Scalars);

// -------------------------------------------------------------------------------------- tests

TYPED_TEST(JacobiDavidsonTyped, InteriorTargetWithExactPreconditioner) {
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 11);
  hs::CsrOperator<T> op(a);
  const double tau = 0.371;
  DenseOperator<T> prec(dense_inverse(shifted<T>(a, nullptr, tau)));
  JdOptions<T> opt;
  opt.nev = 6;
  opt.tol = 1e-12;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
  expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-10);
  EXPECT_GT(r.precond_applies, 0);
  EXPECT_GT(r.inner_iterations, 0);
}

TYPED_TEST(JacobiDavidsonTyped, SinglePrecisionLikePreconditionerReachesDoubleAccuracy) {
  // The preconditioner inverts A - tau I rounded to float: only the speed suffers, the eigenpairs
  // reach tol = 1e-10 because the outer residual is exact.
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 12);
  hs::CsrOperator<T> op(a);
  const double tau = 0.371;
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau))));
  for (JdInnerSolver solver : {JdInnerSolver::Gmres, JdInnerSolver::Auto}) {
    JdOptions<T> opt;
    opt.nev = 8;
    opt.tol = 1e-10;
    opt.inner_solver = solver;
    const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
    expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-9);
  }
}

TYPED_TEST(JacobiDavidsonTyped, HarmonicAndRitzExtractionAgree) {
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 13);
  hs::CsrOperator<T> op(a);
  const double tau = 0.43;
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau))));
  for (JdExtraction ex : {JdExtraction::Harmonic, JdExtraction::Ritz}) {
    SCOPED_TRACE(ex == JdExtraction::Harmonic ? "harmonic" : "ritz");
    JdOptions<T> opt;
    opt.nev = 5;
    opt.tol = 1e-11;
    opt.extraction = ex;
    const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
    expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-9);
  }
}

TYPED_TEST(JacobiDavidsonTyped, GeneralizedPencilWithHpdB) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.04, 21);
  const auto b = hs::test::random_hpd<T>(n, 0.04, 22);
  const auto lambda = hs::test::dense_eigenvalues(a, &b);
  hs::CsrOperator<T> aop(a), bop(b);
  const double tau = 0.5 * (lambda[140] + lambda[141]);
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, &b, tau))));
  JdOptions<T> opt;
  opt.nev = 6;
  opt.tol = 1e-11;
  const auto r = hs::jacobi_davidson<T>(aop, &bop, &prec, tau, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(aop, &bop, r, tau, opt.nev, opt.tol);
  expect_same_values(r.theta, nearest(lambda, tau, opt.nev), 1e-9);
  EXPECT_GT(r.b_applies, 0);
  EXPECT_NE(r.norm_b, 1.0);
}

TYPED_TEST(JacobiDavidsonTyped, LockingFindsEveryCopyOfAMultipleEigenvalue) {
  // A triple eigenvalue next to the target and a double one behind it; the rest lies outside
  // [0.6, 1.5].
  using T = TypeParam;
  std::vector<double> spectrum = {1.0, 1.0, 1.0, 1.2, 1.2, 0.7};
  for (index_t i = 0; i < 147; ++i) spectrum.push_back(-3.0 + 0.0245 * static_cast<double>(i));
  for (index_t i = 0; i < 147; ++i) spectrum.push_back(1.5 + 0.0245 * static_cast<double>(i));
  std::sort(spectrum.begin(), spectrum.end());
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 6, 14);
  hs::CsrOperator<T> op(a);
  const double tau = 1.05;
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau))));
  JdOptions<T> opt;
  opt.nev = 5;
  opt.tol = 1e-11;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
  expect_same_values(r.theta, {1.0, 1.0, 1.0, 1.2, 1.2}, 1e-10);
}

TYPED_TEST(JacobiDavidsonTyped, DeflationFindsTheNextPairs) {
  using T = TypeParam;
  const index_t n = 240;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 23);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 24);
  const auto lambda = hs::test::dense_eigenvalues(a, &b);
  hs::CsrOperator<T> aop(a), bop(b);
  const double tau = 0.5 * (lambda[100] + lambda[101]);
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, &b, tau))));
  JdOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-12;
  const auto first = hs::jacobi_davidson<T>(aop, &bop, &prec, tau, opt);
  ASSERT_TRUE(first.converged);
  opt.deflation = &first.vectors;
  const auto second = hs::jacobi_davidson<T>(aop, &bop, &prec, tau, opt);
  EXPECT_TRUE(second.converged);
  check_contract<T>(aop, &bop, second, tau, opt.nev, opt.tol);
  const auto near8 = nearest(lambda, tau, 8);
  expect_same_values(second.theta, std::vector<double>(near8.begin() + 4, near8.end()), 1e-9);
  EXPECT_LT(gram_deviation<T>(first.vectors, &bop, second.vectors, false), 1e-10);
}

TYPED_TEST(JacobiDavidsonTyped, DeflatedRerunRecoversAFurtherCopy) {
  // The driver's remedy for a missed copy of a multiple eigenvalue (inertia certificate count >
  // found): rerun with the found vectors as deflation block.
  using T = TypeParam;
  std::vector<double> spectrum = {2.0, 2.0, 2.0, 2.0};
  for (index_t i = 0; i < 196; ++i) spectrum.push_back(i < 98 ? -1.0 + 0.01 * i : 3.0 + 0.01 * (i - 98));
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 5, 25);
  hs::CsrOperator<T> op(a);
  const double tau = 2.01;
  DenseOperator<T> prec(dense_inverse(shifted<T>(a, nullptr, tau)));
  JdOptions<T> opt;
  opt.nev = 3;
  opt.tol = 1e-12;
  const auto first = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  ASSERT_TRUE(first.converged);
  expect_same_values(first.theta, {2.0, 2.0, 2.0}, 1e-10);
  opt.deflation = &first.vectors;
  opt.nev = 1;
  const auto second = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(second.converged);
  check_contract<T>(op, nullptr, second, tau, opt.nev, opt.tol);
  expect_same_values(second.theta, {2.0}, 1e-10);
  EXPECT_LT(gram_deviation<T>(first.vectors, nullptr, second.vectors, false), 1e-10);
}

TYPED_TEST(JacobiDavidsonTyped, MinresWithAndWithoutHpdPreconditioner) {
  // Target below the spectrum: A - tau I is positive definite, so its inverse is a valid MINRES
  // preconditioner; without one MINRES solves the correction equation unpreconditioned.
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(20, 21);
  hs::CsrOperator<T> op(a);
  const auto spectrum = hs::test::laplacian_2d_eigenvalues(20, 21);
  const double tau = 0.0;
  DenseOperator<T> prec(dense_inverse(shifted<T>(a, nullptr, -0.2)));
  for (const bool with_prec : {false, true}) {
    SCOPED_TRACE(with_prec ? "preconditioned" : "plain");
    JdOptions<T> opt;
    opt.nev = 4;
    opt.tol = 1e-10;
    opt.inner_solver = JdInnerSolver::Minres;
    opt.inner_iterations = 30;
    const auto r = hs::jacobi_davidson<T>(op, nullptr, with_prec ? &prec : nullptr, tau, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
    expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-9);
    EXPECT_EQ(r.precond_applies > 0, with_prec);
  }
}

TYPED_TEST(JacobiDavidsonTyped, DavidsonStepAndSmallRestartingBasis) {
  // inner_iterations = 0 (Olsen step with the projected preconditioner) and a basis of at most 6
  // vectors kept at 3: many thick restarts, still converged.
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 15);
  hs::CsrOperator<T> op(a);
  const double tau = 0.371;
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau))));
  JdOptions<T> opt;
  opt.nev = 6;
  opt.tol = 1e-10;
  opt.min_basis = 3;
  opt.max_basis = 6;
  opt.inner_iterations = 0;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
  expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-9);
  EXPECT_GT(r.restarts, 0);
  EXPECT_EQ(r.inner_iterations, 0);
}

TYPED_TEST(JacobiDavidsonTyped, SinglePrecisionMumpsFactorAsPreconditioner) {
  // The memory-limited use case: JD preconditioned by a single-precision MUMPS factor of
  // A - tau I still converges to tol = 1e-10.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 3000; ++i) spectrum.push_back(-1.0 + 2.0 * std::pow(static_cast<double>(i) / 2999.0, 1.5));
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 6, 16);
  const double tau = 0.123;
  hs::ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.precision = hs::FactorPrecision::Single;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -tau);
  ASSERT_TRUE(fac->factorize(1.0, -tau));
  FactorSolveOperator<T> prec(*fac);
  hs::CsrOperator<T> op(a);
  JdOptions<T> opt;
  opt.nev = 8;
  opt.tol = 1e-10;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
  expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-10);
}

TYPED_TEST(JacobiDavidsonTyped, InitialSearchSpace) {
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 17);
  hs::CsrOperator<T> op(a);
  const double tau = 0.371;
  DenseOperator<T> prec(dense_inverse(shifted<T>(a, nullptr, tau)));
  JdOptions<T> opt;
  opt.nev = 3;
  opt.tol = 1e-12;
  const auto first = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  ASSERT_TRUE(first.converged);
  opt.initial = &first.vectors;
  const auto again = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
  EXPECT_TRUE(again.converged);
  EXPECT_EQ(again.iterations, 0);
  check_contract<T>(op, nullptr, again, tau, opt.nev, opt.tol);
  expect_same_values(again.theta, first.theta, 1e-12);
}

TYPED_TEST(JacobiDavidsonTyped, NonConvergenceIsReportedNotThrown) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(30, 30);
  hs::CsrOperator<T> op(a);
  JdOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-12;
  opt.max_iterations = 4;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, nullptr, 3.3, opt);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.iterations, 4);
  check_contract<T>(op, nullptr, r, 3.3, opt.nev, opt.tol);
  EXPECT_EQ(static_cast<index_t>(r.theta.size()), opt.nev);  // the basis holds >= nev vectors
}

TYPED_TEST(JacobiDavidsonTyped, ReproducibleForFixedSeed) {
  using T = TypeParam;
  const auto spectrum = interior_spectrum();
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 18);
  hs::CsrOperator<T> op(a);
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, 0.4))));
  JdOptions<T> opt;
  opt.nev = 4;
  const auto r1 = hs::jacobi_davidson<T>(op, nullptr, &prec, 0.4, opt);
  const auto r2 = hs::jacobi_davidson<T>(op, nullptr, &prec, 0.4, opt);
  ASSERT_EQ(r1.theta.size(), r2.theta.size());
  EXPECT_EQ(r1.iterations, r2.iterations);
  EXPECT_EQ(r1.inner_iterations, r2.inner_iterations);
  for (std::size_t i = 0; i < r1.theta.size(); ++i) EXPECT_EQ(r1.theta[i], r2.theta[i]);
  for (index_t i = 0; i < r1.vectors.rows() * r1.vectors.cols(); ++i)
    ASSERT_EQ(r1.vectors.data()[i], r2.vectors.data()[i]);
}

TYPED_TEST(JacobiDavidsonTyped, TinyProblemSpansTheWholeSpace) {
  using T = TypeParam;
  const auto a = hs::test::random_hermitian<T>(7, 0.5, 19);
  const auto lambda = hs::test::dense_eigenvalues(a, static_cast<const hs::CsrMatrix<T>*>(nullptr));
  hs::CsrOperator<T> op(a);
  JdOptions<T> opt;
  opt.nev = 7;
  opt.tol = 1e-12;
  const auto r = hs::jacobi_davidson<T>(op, nullptr, nullptr, 0.1, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, 0.1, opt.nev, opt.tol);
  expect_same_values(r.theta, lambda, 1e-12);
}

TYPED_TEST(JacobiDavidsonTyped, TargetExactlyOnASimpleEigenvalue) {
  // Regression: harmonic extraction (the default with B == nullptr) cannot rank an eigenvalue at
  // tau - the pencil (V^H K V, (K V)^H K V) is singular in that direction - and skipped it,
  // returning a farther set as converged. The certified Ritz override selects it.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 300; ++i) spectrum.push_back(-3.0 + 0.02 * static_cast<double>(i));
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 6, 31);
  hs::CsrOperator<T> op(a);
  const double tau = spectrum[150];  // an eigenvalue of A (to rounding)
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau + 1e-3))));
  for (JdExtraction ex : {JdExtraction::Auto, JdExtraction::Harmonic, JdExtraction::Ritz}) {
    for (const bool with_prec : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "extraction " << static_cast<int>(ex) << ", preconditioner " << with_prec);
      JdOptions<T> opt;
      opt.nev = 5;
      opt.tol = 1e-11;
      opt.extraction = ex;
      opt.max_iterations = 3000;
      const auto r = hs::jacobi_davidson<T>(op, nullptr, with_prec ? &prec : nullptr, tau, opt);
      EXPECT_TRUE(r.converged);
      check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
      expect_same_values(r.theta, nearest(spectrum, tau, opt.nev), 1e-9);
      EXPECT_EQ(copies(r.theta, tau, 1e-9), 1);
    }
  }
}

TYPED_TEST(JacobiDavidsonTyped, TargetExactlyOnAMultipleEigenvalue) {
  // tau = 4 is an eigenvalue of multiplicity 25 of the 25 x 25 grid Laplacian; the
  // preconditioner is a float-rounded inverse at a slightly moved shift (the double factor at
  // tau itself is singular).
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(25, 25);
  hs::CsrOperator<T> op(a);
  const double tau = 4.0;
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau + 1e-3))));
  for (JdExtraction ex : {JdExtraction::Harmonic, JdExtraction::Ritz}) {
    SCOPED_TRACE(ex == JdExtraction::Harmonic ? "harmonic" : "ritz");
    JdOptions<T> opt;
    opt.nev = 6;
    opt.tol = 1e-11;
    opt.extraction = ex;
    const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
    EXPECT_EQ(copies(r.theta, tau, 1e-9), opt.nev);
  }
}

TYPED_TEST(JacobiDavidsonTyped, NullSpaceAtTargetZero) {
  // Ten exact zero eigenvalues (an EM-like null space) and tau = 0: all of them first, then the
  // nearest nonzero ones.
  using T = TypeParam;
  std::vector<double> spectrum(10, 0.0);
  for (index_t i = 0; i < 290; ++i) spectrum.push_back(0.5 + 0.02 * static_cast<double>(i));
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 6, 32);
  hs::CsrOperator<T> op(a);
  DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, -1e-3))));
  for (JdExtraction ex : {JdExtraction::Harmonic, JdExtraction::Ritz}) {
    for (const index_t nev : {6, 12}) {
      SCOPED_TRACE(::testing::Message() << (ex == JdExtraction::Harmonic ? "harmonic" : "ritz") << " nev " << nev);
      JdOptions<T> opt;
      opt.nev = nev;
      opt.tol = 1e-11;
      opt.extraction = ex;
      const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, 0.0, opt);
      EXPECT_TRUE(r.converged);
      check_contract<T>(op, nullptr, r, 0.0, opt.nev, opt.tol);
      expect_same_values(r.theta, nearest(spectrum, 0.0, nev), 1e-9);
    }
  }
}

/// {0.5 (mult times), 1.0, 1.01, ...}: n = 300.
std::vector<double> multiple_spectrum(index_t mult) {
  std::vector<double> w(static_cast<std::size_t>(mult), 0.5);
  for (index_t i = 0; i < 300 - mult; ++i) w.push_back(1.0 + 0.01 * static_cast<double>(i));
  return w;
}

TYPED_TEST(JacobiDavidsonTyped, MultiplicityAtLeastNevWithPreconditioner) {
  // nev = 6 copies of a 6- or 8-fold eigenvalue (EM: 6-fold eigenvalues and k = 12).
  // Regression: each expansion amplifies only the copy inside the current approximation, so the
  // other copies of the random start block stayed diluted and an approximation of 1.0 built
  // along the way was preferred: 4 or 5 copies + 1.0 were returned as converged. The start block
  // iteration with the preconditioner (start_iterations) represents every copy.
  using T = TypeParam;
  for (const index_t mult : {6, 8}) {
    const auto spectrum = multiple_spectrum(mult);
    const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 3);
    hs::CsrOperator<T> op(a);
    const double tau = 0.45;
    DenseOperator<T> prec(dense_inverse(round_to_float(shifted<T>(a, nullptr, tau))));
    for (JdExtraction ex : {JdExtraction::Harmonic, JdExtraction::Ritz}) {
      for (const std::uint64_t seed : {2026ULL, 7ULL}) {
        SCOPED_TRACE(::testing::Message() << "mult " << mult << (ex == JdExtraction::Harmonic ? " harmonic" : " ritz")
                                          << " seed " << seed);
        JdOptions<T> opt;
        opt.nev = 6;
        opt.tol = 1e-11;
        opt.extraction = ex;
        opt.seed = seed;
        const auto r = hs::jacobi_davidson<T>(op, nullptr, &prec, tau, opt);
        EXPECT_TRUE(r.converged);
        check_contract<T>(op, nullptr, r, tau, opt.nev, opt.tol);
        EXPECT_EQ(copies(r.theta, 0.5, 1e-9), opt.nev);
      }
    }
  }
}

TYPED_TEST(JacobiDavidsonTyped, MultiplicityWithoutPreconditionerIsCompletedByDeflatedReruns) {
  // Documented limitation: without a preconditioner (no start block iteration) a copy of an
  // eigenvalue of multiplicity >= about nev - 1 can be missed, and that is not detected here.
  // The driver's remedy - rerun with the copies found as deflation block, as long as the inertia
  // count says copies are missing - completes the set.
  using T = TypeParam;
  const auto spectrum = multiple_spectrum(6);
  const auto a = hs::test::prescribed_spectrum<T>(spectrum, 8, 3);
  hs::CsrOperator<T> op(a);
  const index_t n = a.nrows;
  const double tau = 0.45;
  JdOptions<T> opt;
  opt.nev = 6;
  opt.tol = 1e-11;
  DenseMatrix<T> found(n, 0);
  for (int run = 0; run < 4 && found.cols() < opt.nev; ++run) {
    JdOptions<T> o = opt;
    o.nev = opt.nev - found.cols();
    if (found.cols() > 0) o.deflation = &found;
    const auto r = hs::jacobi_davidson<T>(op, nullptr, nullptr, tau, o);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, tau, o.nev, o.tol);
    if (found.cols() > 0) {
      EXPECT_LT(gram_deviation<T>(found, nullptr, r.vectors, false), 1e-10);
    }
    DenseMatrix<T> next(n, found.cols() + copies(r.theta, 0.5, 1e-9));
    std::copy(found.data(), found.data() + n * found.cols(), next.data());
    index_t c = found.cols();
    for (index_t j = 0; j < static_cast<index_t>(r.theta.size()); ++j)
      if (std::abs(r.theta[j] - 0.5) <= 1e-9) std::copy(r.vectors.col(j), r.vectors.col(j) + n, next.col(c++));
    found = next;
  }
  EXPECT_EQ(found.cols(), opt.nev);
}

TYPED_TEST(JacobiDavidsonTyped, InvalidOptionsThrowInvalidInput) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(20);
  const auto a5 = hs::test::laplacian_1d<T>(5);
  hs::CsrOperator<T> op(a), small(a5), bop(a);
  auto expect_invalid = [&](const JdOptions<T>& o, LinearOperator<T>* b = nullptr, LinearOperator<T>* m = nullptr,
                            double tau = 0.5) {
    try {
      hs::jacobi_davidson<T>(op, b, m, tau, o);
      ADD_FAILURE() << "no exception";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  };
  JdOptions<T> o;
  o.nev = 0;
  expect_invalid(o);
  o.nev = 21;
  expect_invalid(o);
  o = {};
  o.tol = -1.0;
  expect_invalid(o);
  o = {};
  o.max_iterations = -1;
  expect_invalid(o);
  o = {};
  o.inner_iterations = -1;
  expect_invalid(o);
  o = {};
  o.min_basis = 8;
  o.max_basis = 8;
  expect_invalid(o);
  o = {};
  o.start_iterations = -1;
  expect_invalid(o);
  o = {};
  expect_invalid(o, &small);
  expect_invalid(o, nullptr, &small);
  expect_invalid(o, nullptr, nullptr, std::numeric_limits<double>::quiet_NaN());
  o.extraction = JdExtraction::Harmonic;
  expect_invalid(o, &bop);
  o = {};
  DenseMatrix<T> wrong(20, 0);
  o.initial = &wrong;
  expect_invalid(o);
  o = {};
  DenseMatrix<T> defl_rows(19, 2), defl_zero(20, 2), defl_ok(20, 18);
  for (index_t j = 0; j < 18; ++j) defl_ok(j, j) = T(1);
  o.deflation = &defl_rows;
  expect_invalid(o);
  o.deflation = &defl_zero;  // not orthonormal
  expect_invalid(o);
  o.deflation = &defl_ok;  // nev = 6 > n - d = 2
  expect_invalid(o);
}

}  // namespace
