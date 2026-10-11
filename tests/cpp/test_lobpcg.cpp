// Tests of hs::lobpcg (include/hsolve/eigen_iterative.hpp) on operators with known spectra and
// against dense references, for both scalar types.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
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
using hs::LinearOperator;
using hs::LobpcgOptions;
using hs::Which;
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

/// Operator that returns NaN after `good` single-vector applications (malformed input test).
template <class T>
class PoisonedOperator final : public LinearOperator<T> {
 public:
  PoisonedOperator(LinearOperator<T>& inner, index_t good) : inner_(inner), good_(good) {}
  index_t size() const override { return inner_.size(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
    inner_.apply(x, ldx, y, ldy, p);
    seen_ += p;
    if (seen_ > good_) y[0] = T(std::numeric_limits<double>::quiet_NaN());
  }

 private:
  LinearOperator<T>& inner_;
  index_t good_;
  index_t seen_ = 0;
};

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

/// Result contract of include/hsolve/eigen_iterative.hpp: sizes, ordering, B-orthonormality,
/// estimates equal to recomputed backward errors, bookkeeping.
template <class T>
void check_contract(LinearOperator<T>& a, LinearOperator<T>* b, const IterativeEigResult<T>& r, Which which,
                    index_t nev, double tol) {
  const index_t m = static_cast<index_t>(r.theta.size());
  ASSERT_EQ(m, nev);
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
      if (which == Which::SmallestAlgebraic) {
        EXPECT_LE(r.theta[i - 1], r.theta[i]) << "pair " << i;
      } else {
        EXPECT_GE(r.theta[i - 1], r.theta[i]) << "pair " << i;
      }
    }
  }
  EXPECT_LT(gram_deviation<T>(r.vectors, b, r.vectors, true), 1e-11);
  // Recompute the backward errors.
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
    EXPECT_NEAR(res, r.residual[j], 1e-12 * (r.norm_a + std::abs(r.theta[j]) * r.norm_b) + 1e-6 * res) << j;
    EXPECT_NEAR(eta, r.estimate[j], 1e-12 + 1e-6 * eta) << j;
  }
  ASSERT_EQ(static_cast<index_t>(r.history.size()), r.iterations + 1);
  for (index_t c = 0; c <= r.iterations; ++c) EXPECT_EQ(r.history[c].restart, c);
  EXPECT_EQ(r.history.back().nconv, r.nconv);
  EXPECT_GE(r.op_applies, r.history.back().op_applies);
  EXPECT_GT(r.norm_a, 0.0);
}

/// The nev smallest (or largest, descending) values of an ascending spectrum.
std::vector<double> ends(const std::vector<double>& ascending, Which which, index_t nev) {
  std::vector<double> out;
  for (index_t i = 0; i < nev; ++i)
    out.push_back(which == Which::SmallestAlgebraic ? ascending[i] : ascending[ascending.size() - 1 - i]);
  return out;
}

void expect_values(const std::vector<double>& got, const std::vector<double>& want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
}

template <class T>
class LobpcgTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LobpcgTyped, Scalars);

// -------------------------------------------------------------------------------------- tests

TYPED_TEST(LobpcgTyped, LaplacianEndsWithoutPreconditioner) {
  // 2-D Laplacian on a square grid: exact double eigenvalues (lambda_ij = lambda_ji) at both ends.
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(24, 24);
  hs::CsrOperator<T> op(a);
  const auto spectrum = hs::test::laplacian_2d_eigenvalues(24, 24);
  for (Which which : {Which::SmallestAlgebraic, Which::LargestAlgebraic}) {
    SCOPED_TRACE(which == Which::SmallestAlgebraic ? "smallest" : "largest");
    LobpcgOptions<T> opt;
    opt.nev = 6;
    opt.tol = 1e-10;
    const auto r = hs::lobpcg<T>(op, nullptr, nullptr, which, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, which, opt.nev, opt.tol);
    expect_values(r.theta, ends(spectrum, which, opt.nev), 1e-8);
    EXPECT_EQ(r.precond_applies, 0);
  }
}

TYPED_TEST(LobpcgTyped, ShiftInvertPreconditionerAcceleratesConvergence) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(24, 24);
  hs::CsrOperator<T> op(a);
  const auto spectrum = hs::test::laplacian_2d_eigenvalues(24, 24);
  LobpcgOptions<T> opt;
  opt.nev = 5;
  opt.tol = 1e-10;
  const auto plain = hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
  // Smallest: T ~ (A - s I)^{-1}, s below the spectrum (HPD). Largest: s above it (negative
  // definite; the sign of the preconditioner does not matter).
  for (Which which : {Which::SmallestAlgebraic, Which::LargestAlgebraic}) {
    SCOPED_TRACE(which == Which::SmallestAlgebraic ? "smallest" : "largest");
    const double s = which == Which::SmallestAlgebraic ? -0.05 : 8.05;
    DenseOperator<T> prec(dense_inverse(shifted<T>(a, nullptr, s)));
    const auto r = hs::lobpcg<T>(op, nullptr, &prec, which, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, which, opt.nev, opt.tol);
    expect_values(r.theta, ends(spectrum, which, opt.nev), 1e-9);
    EXPECT_GT(r.precond_applies, 0);
    if (which == Which::SmallestAlgebraic) {
      EXPECT_LT(4 * r.iterations, plain.iterations);
    }
  }
}

TYPED_TEST(LobpcgTyped, GeneralizedPencilWithHpdB) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.04, 41);
  const auto b = hs::test::random_hpd<T>(n, 0.04, 42);
  const auto lambda = hs::test::dense_eigenvalues(a, &b);
  hs::CsrOperator<T> aop(a), bop(b);
  for (Which which : {Which::SmallestAlgebraic, Which::LargestAlgebraic}) {
    SCOPED_TRACE(which == Which::SmallestAlgebraic ? "smallest" : "largest");
    const double s = which == Which::SmallestAlgebraic ? lambda.front() - 0.1 : lambda.back() + 0.1;
    DenseOperator<T> prec(dense_inverse(shifted<T>(a, &b, s)));
    LobpcgOptions<T> opt;
    opt.nev = 5;
    opt.tol = 1e-11;
    for (const bool with_prec : {false, true}) {
      SCOPED_TRACE(with_prec ? "preconditioned" : "plain");
      const auto r = hs::lobpcg<T>(aop, &bop, with_prec ? &prec : nullptr, which, opt);
      EXPECT_TRUE(r.converged);
      check_contract<T>(aop, &bop, r, which, opt.nev, opt.tol);
      expect_values(r.theta, ends(lambda, which, opt.nev), 1e-9);
      EXPECT_GT(r.b_applies, 0);
    }
  }
}

TYPED_TEST(LobpcgTyped, MultipleEigenvaluesWithinTheBlock) {
  // A triple and a double eigenvalue at the lower end: the block finds every copy.
  using T = TypeParam;
  std::vector<double> spectrum = {0.5, 0.5, 0.5, 0.9, 0.9, 1.3};
  for (index_t i = 0; i < 394; ++i) spectrum.push_back(2.0 + 0.01 * static_cast<double>(i));
  DenseOperator<T> op(hs::test::to_dense(hs::test::prescribed_spectrum<T>(spectrum, 8, 7)));
  DenseOperator<T> prec(dense_inverse(shifted<T>(hs::test::prescribed_spectrum<T>(spectrum, 8, 7), nullptr, 0.0)));
  LobpcgOptions<T> opt;
  opt.nev = 6;
  opt.tol = 1e-11;
  const auto r = hs::lobpcg<T>(op, nullptr, &prec, Which::SmallestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, Which::SmallestAlgebraic, opt.nev, opt.tol);
  expect_values(r.theta, {0.5, 0.5, 0.5, 0.9, 0.9, 1.3}, 1e-10);
}

TYPED_TEST(LobpcgTyped, DeflationFindsTheNextPairs) {
  using T = TypeParam;
  const index_t n = 240;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 51);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 52);
  const auto lambda = hs::test::dense_eigenvalues(a, &b);
  hs::CsrOperator<T> aop(a), bop(b);
  DenseOperator<T> prec(dense_inverse(shifted<T>(a, &b, lambda.front() - 0.1)));
  LobpcgOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-12;
  const auto first = hs::lobpcg<T>(aop, &bop, &prec, Which::SmallestAlgebraic, opt);
  ASSERT_TRUE(first.converged);
  opt.deflation = &first.vectors;
  opt.seed = 99;
  const auto second = hs::lobpcg<T>(aop, &bop, &prec, Which::SmallestAlgebraic, opt);
  EXPECT_TRUE(second.converged);
  check_contract<T>(aop, &bop, second, Which::SmallestAlgebraic, opt.nev, opt.tol);
  expect_values(second.theta, {lambda[4], lambda[5], lambda[6], lambda[7]}, 1e-9);
  EXPECT_LT(gram_deviation<T>(first.vectors, &bop, second.vectors, false), 1e-10);
}

TYPED_TEST(LobpcgTyped, ExactInitialBlockConvergesWithoutIterating) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(16, 17);
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 3;
  opt.tol = 1e-10;
  const auto first = hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
  ASSERT_TRUE(first.converged);
  opt.initial = &first.vectors;
  opt.block_size = 3;
  const auto again = hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_TRUE(again.converged);
  EXPECT_EQ(again.iterations, 0);
  check_contract<T>(op, nullptr, again, Which::SmallestAlgebraic, opt.nev, opt.tol);
  expect_values(again.theta, first.theta, 1e-12);
}

TYPED_TEST(LobpcgTyped, SmallProblemWhereTheBasisExceedsTheSpace) {
  // 3 m > n: W and P directions become dependent and are dropped; the result is still exact.
  using T = TypeParam;
  const auto a = hs::test::random_hermitian<T>(12, 0.3, 61);
  const auto lambda = hs::test::dense_eigenvalues(a, static_cast<const hs::CsrMatrix<T>*>(nullptr));
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 3;
  opt.block_size = 5;
  opt.tol = 1e-12;
  const auto r = hs::lobpcg<T>(op, nullptr, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt.nev, opt.tol);
  expect_values(r.theta, ends(lambda, Which::LargestAlgebraic, 3), 1e-11);
}

TYPED_TEST(LobpcgTyped, IndefiniteMatrixSmallestAlgebraic) {
  using T = TypeParam;
  const auto a = hs::test::random_hermitian<T>(400, 0.02, 71);
  const auto lambda = hs::test::dense_eigenvalues(a, static_cast<const hs::CsrMatrix<T>*>(nullptr));
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-10;
  opt.max_iterations = 3000;
  const auto r = hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  check_contract<T>(op, nullptr, r, Which::SmallestAlgebraic, opt.nev, opt.tol);
  expect_values(r.theta, ends(lambda, Which::SmallestAlgebraic, 4), 1e-8);
  EXPECT_LT(r.theta[0], 0.0);
}

TYPED_TEST(LobpcgTyped, NonConvergenceIsReportedNotThrown) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(30, 30);
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-12;
  opt.max_iterations = 3;
  const auto r = hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.iterations, 3);
  check_contract<T>(op, nullptr, r, Which::SmallestAlgebraic, opt.nev, opt.tol);
  for (const auto& rec : r.history) EXPECT_GT(rec.max_estimate, opt.tol);
}

TYPED_TEST(LobpcgTyped, ReproducibleForFixedSeed) {
  using T = TypeParam;
  const auto a = hs::test::random_hermitian<T>(200, 0.05, 81);
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 3;
  const auto r1 = hs::lobpcg<T>(op, nullptr, nullptr, Which::LargestAlgebraic, opt);
  const auto r2 = hs::lobpcg<T>(op, nullptr, nullptr, Which::LargestAlgebraic, opt);
  ASSERT_EQ(r1.theta.size(), r2.theta.size());
  EXPECT_EQ(r1.iterations, r2.iterations);
  for (std::size_t i = 0; i < r1.theta.size(); ++i) EXPECT_EQ(r1.theta[i], r2.theta[i]);
  for (index_t i = 0; i < r1.vectors.rows() * r1.vectors.cols(); ++i)
    ASSERT_EQ(r1.vectors.data()[i], r2.vectors.data()[i]);
}

TYPED_TEST(LobpcgTyped, GivenNormsAreUsedInTheBackwardError) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(200);
  hs::CsrOperator<T> op(a);
  LobpcgOptions<T> opt;
  opt.nev = 2;
  opt.norm_a = 10.0;
  const auto r = hs::lobpcg<T>(op, nullptr, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_EQ(r.norm_a, 10.0);
  EXPECT_EQ(r.norm_b, 1.0);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt.nev, opt.tol);
  // The estimate needs 30 applications only when the norm is not given.
  opt.norm_a = 0.0;
  const auto e = hs::lobpcg<T>(op, nullptr, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_NEAR(e.norm_a, 4.0, 0.05);
}

TYPED_TEST(LobpcgTyped, InvalidOptionsThrowInvalidInput) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(20);
  const auto b5 = hs::test::laplacian_1d<T>(5);
  hs::CsrOperator<T> op(a), small(b5);
  auto expect_invalid = [&](const LobpcgOptions<T>& o, Which which = Which::SmallestAlgebraic,
                            LinearOperator<T>* b = nullptr, LinearOperator<T>* t = nullptr) {
    try {
      hs::lobpcg<T>(op, b, t, which, o);
      ADD_FAILURE() << "no exception";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  };
  LobpcgOptions<T> o;
  o.nev = 0;
  expect_invalid(o);
  o = {};
  o.nev = 4;
  o.block_size = 3;
  expect_invalid(o);
  o.block_size = 21;
  expect_invalid(o);
  o = {};
  expect_invalid(o, Which::LargestMagnitude);
  o.tol = 0.0;
  expect_invalid(o);
  o = {};
  o.max_iterations = -1;
  expect_invalid(o);
  o = {};
  o.norm_a = -1.0;
  expect_invalid(o);
  o = {};
  expect_invalid(o, Which::SmallestAlgebraic, &small);
  expect_invalid(o, Which::SmallestAlgebraic, nullptr, &small);
  DenseMatrix<T> wrong(19, 2);
  o.initial = &wrong;
  expect_invalid(o);
  DenseMatrix<T> bad_defl(20, 2);  // zero columns: not orthonormal
  o = {};
  o.deflation = &bad_defl;
  expect_invalid(o);
}

TYPED_TEST(LobpcgTyped, NonFiniteOperatorOutputThrows) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(60);
  hs::CsrOperator<T> inner(a);
  LobpcgOptions<T> opt;
  opt.nev = 2;
  opt.norm_a = 4.0;
  for (index_t good : {0, 20}) {
    PoisonedOperator<T> op(inner, good);
    try {
      hs::lobpcg<T>(op, nullptr, nullptr, Which::SmallestAlgebraic, opt);
      ADD_FAILURE() << "no exception";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  }
}

}  // namespace
