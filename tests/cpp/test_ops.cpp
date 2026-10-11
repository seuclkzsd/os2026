// Unit tests for src/ops: CsrOperator and ShiftInvertOperator (with B, with B = I, with
// double-precision iterative refinement of a single-precision factor, including shifts too close
// to an eigenvalue for refinement to converge, and the guard against refactorizing the factor
// behind the operator's back). Dense LU references from the fixtures.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::FactorOptions;
using hs::FactorPrecision;
using hs::index_t;
using hs::ShiftedPencil;

// Column-major n x p block of deterministic random entries with leading dimension ld.
template <class T>
std::vector<T> random_block(index_t n, index_t p, index_t ld, std::uint64_t seed) {
  std::vector<T> x(static_cast<std::size_t>(ld * p), T(0));
  for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.data() + j * ld, n, seed, j);
  return x;
}

// max_j ||y_j - z_j||_2 / ||z_j||_2.
template <class T>
double max_rel_diff(index_t n, index_t p, const T* y, index_t ldy, const T* z, index_t ldz) {
  double worst = 0.0;
  for (index_t j = 0; j < p; ++j) {
    double d = 0.0;
    for (index_t i = 0; i < n; ++i) d += hs::abs2(y[i + j * ldy] - z[i + j * ldz]);
    worst = std::max(worst, std::sqrt(d) / hs::linalg::norm2(n, z + j * ldz));
  }
  return worst;
}

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

// Indefinite test pencil of order n: random Hermitian A, random HPD B (or B = I), a shift in a
// spectral gap away from the eigenvalues, and the dense LU reference operator for it.
template <class T>
struct ShiftInvertCase {
  // Members are initialized in this order (each from the previous ones).
  index_t n;
  bool has_b;
  CsrMatrix<T> a;
  CsrMatrix<T> b;
  hs::DenseMatrix<T> dense_b;
  double sigma;
  hs::test::DenseShiftInvertOperator<T> reference;

  ShiftInvertCase(index_t order, bool with_b)
      : n(order),
        has_b(with_b),
        a(hs::test::random_hermitian<T>(order, 0.03, 71)),
        b(hs::test::random_hpd<T>(order, 0.02, 72)),
        dense_b(hs::test::to_dense(b)),
        sigma(gap_shift(a, b_ptr())),
        reference(hs::test::to_dense(a), has_b ? &dense_b : nullptr, sigma) {}

  const CsrMatrix<T>* b_ptr() const { return has_b ? &b : nullptr; }

  // Midpoint of the widest gap among the eigenvalues near the middle of the spectrum.
  static double gap_shift(const CsrMatrix<T>& a, const CsrMatrix<T>* b) {
    const auto w = hs::test::dense_eigenvalues(a, b);
    const std::size_t mid = w.size() / 2;
    std::size_t best = mid;
    for (std::size_t k = mid - 10; k < mid + 10; ++k)
      if (w[k + 1] - w[k] > w[best + 1] - w[best]) best = k;
    return 0.5 * (w[best] + w[best + 1]);
  }
};

template <class T>
class OpsCsr : public ::testing::Test {};
template <class T>
class OpsShiftInvert : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(OpsCsr, Scalars);
TYPED_TEST_SUITE(OpsShiftInvert, Scalars);

// ------------------------------------------------------------------------------- CsrOperator

TYPED_TEST(OpsCsr, AppliesMatrixToBlocks) {
  using T = TypeParam;
  const index_t n = 200, p = 5, ldx = n + 2, ldy = n + 4;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 1);
  hs::CsrOperator<T> op(a);
  EXPECT_EQ(op.size(), n);
  const auto x = random_block<T>(n, p, ldx, 2);
  std::vector<T> y(static_cast<std::size_t>(ldy * p), T(0));
  op.apply(x.data(), ldx, y.data(), ldy, p);
  EXPECT_EQ(op.applications(), p);

  hs::test::DenseOperator<T> reference(hs::test::to_dense(a));
  std::vector<T> z(static_cast<std::size_t>(n * p));
  reference.apply(x.data(), ldx, z.data(), n, p);
  EXPECT_LE(max_rel_diff(n, p, y.data(), ldy, z.data(), n), 1e-14);
}

TYPED_TEST(OpsCsr, RejectsNonSquareMatrix) {
  using T = TypeParam;
  CsrMatrix<T> rect;
  rect.nrows = 3;
  rect.ncols = 4;
  rect.rowptr.assign(4, 0);
  EXPECT_EQ(error_code_of([&] { hs::CsrOperator<T> op(rect); }), hs::Error::Code::InvalidInput);
}

// -------------------------------------------------------------------------------
// ShiftInvertOperator

// Op = (A - sigma B)^{-1} B and, with B = I, (A - sigma I)^{-1}: compared with dense LU.
TYPED_TEST(OpsShiftInvert, MatchesDenseReferenceWithAndWithoutB) {
  using T = TypeParam;
  for (bool with_b : {true, false}) {
    ShiftInvertCase<T> c(400, with_b);
    ShiftedPencil<T> pencil(c.a, c.b_ptr());
    auto f = hs::make_mumps_factorization<T>();
    f->analyze(pencil, 1.0, -c.sigma);
    ASSERT_TRUE(f->factorize(1.0, -c.sigma));
    hs::ShiftInvertOperator<T> op(*f, pencil, c.sigma);
    EXPECT_EQ(op.size(), c.n);
    EXPECT_EQ(op.sigma(), c.sigma);

    const index_t p = 4, ldx = c.n + 1, ldy = c.n + 3;
    const auto x = random_block<T>(c.n, p, ldx, 3);
    std::vector<T> y(static_cast<std::size_t>(ldy * p), T(0)), z(static_cast<std::size_t>(c.n * p));
    op.apply(x.data(), ldx, y.data(), ldy, p);
    c.reference.apply(x.data(), ldx, z.data(), c.n, p);
    EXPECT_LE(max_rel_diff(c.n, p, y.data(), ldy, z.data(), c.n), 1e-10) << "with_b=" << with_b;
    EXPECT_EQ(op.applications(), p);
    EXPECT_EQ(op.refinement_steps(), 0);
    EXPECT_EQ(op.last_relative_residual(), 0.0);
  }
}

// A single-precision factor alone gives Op only to ~eps_single * cond(K); double-precision
// refinement against the exact A and B restores ~1e-12, one extra block solve per correction.
// Blocks use distinct leading dimensions (ldx = n+1, ldy = n+3, against the n-strided internal
// workspace) and growing widths p = 1, 3, 5 on one operator (the workspace grows on demand).
TYPED_TEST(OpsShiftInvert, RefinementOfSinglePrecisionFactorReachesDoubleAccuracy) {
  using T = TypeParam;
  for (bool with_b : {true, false}) {
    ShiftInvertCase<T> c(400, with_b);
    ShiftedPencil<T> pencil(c.a, c.b_ptr());
    FactorOptions opt;
    opt.precision = FactorPrecision::Single;
    auto f = hs::make_mumps_factorization<T>(opt);
    f->analyze(pencil, 1.0, -c.sigma);
    ASSERT_TRUE(f->factorize(1.0, -c.sigma));

    const index_t n = c.n, ldx = n + 1, ldy = n + 3, pmax = 5;
    const auto x = random_block<T>(n, pmax, ldx, 4);
    std::vector<T> z(static_cast<std::size_t>(n * pmax)), y(static_cast<std::size_t>(ldy * pmax));
    c.reference.apply(x.data(), ldx, z.data(), n, pmax);

    hs::ShiftInvertOperator<T> plain(*f, pencil, c.sigma);
    plain.apply(x.data(), ldx, y.data(), ldy, pmax);
    const double plain_err = max_rel_diff(n, pmax, y.data(), ldy, z.data(), n);
    EXPECT_GT(plain_err, 1e-9) << "single precision alone cannot be double-accurate";
    EXPECT_LT(plain_err, 1e-2);

    hs::SolveRefinement refine;
    refine.max_steps = 10;
    refine.rel_tol = 1e-14;
    hs::ShiftInvertOperator<T> refined(*f, pencil, c.sigma, refine);
    index_t steps_before = 0;
    for (index_t p : {1, 3, 5}) {
      std::fill(y.begin(), y.end(), T(0));
      refined.apply(x.data(), ldx, y.data(), ldy, p);
      EXPECT_LE(max_rel_diff(n, p, y.data(), ldy, z.data(), n), 1e-12)
          << "with_b=" << with_b << " p=" << p;
      EXPECT_LE(refined.last_relative_residual(), 1e-13);
      const index_t steps = refined.refinement_steps() - steps_before;
      EXPECT_GE(steps, 1);
      EXPECT_LE(steps, refine.max_steps);
      steps_before = refined.refinement_steps();
      // Padding rows between the columns of Y are never written.
      for (index_t j = 0; j < p; ++j)
        for (index_t i = n; i < ldy; ++i) EXPECT_EQ(y[i + j * ldy], T(0));
    }
    EXPECT_EQ(refined.applications(), 1 + 3 + 5);

    // A loose tolerance stops early: fewer corrections, residual below that tolerance.
    hs::SolveRefinement loose;
    loose.max_steps = 10;
    loose.rel_tol = 1e-8;
    hs::ShiftInvertOperator<T> early(*f, pencil, c.sigma, loose);
    early.apply(x.data(), ldx, y.data(), ldy, pmax);
    EXPECT_LE(early.last_relative_residual(), 1e-8);
    hs::ShiftInvertOperator<T> tight(*f, pencil, c.sigma, refine);
    tight.apply(x.data(), ldx, y.data(), ldy, pmax);
    EXPECT_LT(early.refinement_steps(), tight.refinement_steps());

    // rel_tol = 0 is unattainable: refinement stops once corrections no longer reduce the
    // residuals (double-precision floor), long before max_steps.
    hs::SolveRefinement exhaustive;
    exhaustive.max_steps = 50;
    hs::ShiftInvertOperator<T> stagnating(*f, pencil, c.sigma, exhaustive);
    stagnating.apply(x.data(), ldx, y.data(), ldy, pmax);
    EXPECT_LT(stagnating.refinement_steps(), 15);
    EXPECT_LE(max_rel_diff(n, pmax, y.data(), ldy, z.data(), n), 1e-12);
  }
}

// Per-column relative residuals ||B x_j - K y_j|| / ||B x_j||, computed independently of the
// operator (assembled K, spmv).
template <class T>
std::vector<double> column_residuals(const ShiftedPencil<T>& pencil, double sigma, const T* x,
                                     const T* y, index_t n, index_t p) {
  const auto k = pencil.assemble_full(1.0, -sigma);
  std::vector<T> bx(static_cast<std::size_t>(n)), r(static_cast<std::size_t>(n));
  std::vector<double> rel;
  for (index_t j = 0; j < p; ++j) {
    if (pencil.b()) {
      hs::spmv(*pencil.b(), x + j * n, bx.data());
    } else {
      std::copy(x + j * n, x + (j + 1) * n, bx.begin());
    }
    hs::spmv(k, y + j * n, r.data());
    for (index_t i = 0; i < n; ++i) r[i] = bx[i] - r[i];
    rel.push_back(hs::linalg::norm2(n, r.data()) / hs::linalg::norm2(n, bx.data()));
  }
  return rel;
}

// With sigma within 1e-8 * ||K|| of an eigenvalue a single-precision factor is too inaccurate for
// refinement to converge (eps_single * cond(K) > 1): corrections stagnate or diverge. Each column
// must still return its best iterate, so allowing more steps never makes the result worse, and
// the reported residual is that of the returned block (up to rounding of the residual itself).
// Shifts the Single factorization flags as singular are skipped.
TYPED_TEST(OpsShiftInvert, RefinementKeepsTheBestIterateWhenItCannotConverge) {
  using T = TypeParam;
  const index_t nx = 50, ny = 51, n = nx * ny, p = 3;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions opt;
  opt.precision = FactorPrecision::Single;
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(pencil, 1.0, -3.0);
  const auto x = random_block<T>(n, p, n, 7);
  std::vector<T> y(static_cast<std::size_t>(n * p));
  int stalled = 0;
  for (int t = 1; t < 12; ++t) {
    const double sigma = w[(n - 1) * t / 12] + 1e-8 * w.back();
    if (!f->factorize(1.0, -sigma)) continue;
    hs::ShiftInvertOperator<T> plain(*f, pencil, sigma);
    plain.apply(x.data(), n, y.data(), n, p);
    const auto plain_rel = column_residuals(pencil, sigma, x.data(), y.data(), n, p);

    double previous = std::numeric_limits<double>::infinity();
    for (int max_steps = 1; max_steps <= 8; ++max_steps) {
      hs::SolveRefinement refine;
      refine.max_steps = max_steps;
      refine.rel_tol = 1e-14;
      hs::ShiftInvertOperator<T> op(*f, pencil, sigma, refine);
      op.apply(x.data(), n, y.data(), n, p);
      const auto rel = column_residuals(pencil, sigma, x.data(), y.data(), n, p);
      const double worst = *std::max_element(rel.begin(), rel.end());
      EXPECT_NEAR(op.last_relative_residual(), worst, 1e-6 * worst + 1e-10)
          << "sigma=" << sigma << " max_steps=" << max_steps;
      for (index_t j = 0; j < p; ++j)
        EXPECT_LE(rel[j], plain_rel[j] * (1 + 1e-9) + 1e-10)
            << "sigma=" << sigma << " column " << j << " max_steps=" << max_steps;
      EXPECT_LE(worst, previous * (1 + 1e-9) + 1e-10)
          << "more steps gave a worse block: sigma=" << sigma << " max_steps=" << max_steps;
      previous = worst;
    }
    if (previous > 1e-6) ++stalled;
  }
  EXPECT_GT(stalled, 0) << "refinement was expected to stall at some of these shifts";
}

// A Single factor cannot flag every exactly singular shift (delocalized eigenvectors, see
// factor.hpp); when factorize() returns true there, refinement cannot converge and the reported
// residual exposes the unusable factor. (Measured: 8 of these 24 shifts escape detection.)
TYPED_TEST(OpsShiftInvert, UndetectedSinglePrecisionSingularityShowsInTheResidual) {
  using T = TypeParam;
  const index_t nx = 50, ny = 51, n = nx * ny, p = 2;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions opt;
  opt.precision = FactorPrecision::Single;
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(pencil, 1.0, -3.0);
  const auto x = random_block<T>(n, p, n, 8);
  std::vector<T> y(static_cast<std::size_t>(n * p));
  hs::SolveRefinement refine;
  refine.max_steps = 10;
  refine.rel_tol = 1e-12;
  int undetected = 0;
  for (int t = 0; t < 24; ++t) {
    const double lambda = w[(n - 1) * t / 23];
    if (!f->factorize(1.0, -lambda)) continue;
    ++undetected;
    hs::ShiftInvertOperator<T> op(*f, pencil, lambda, refine);
    op.apply(x.data(), n, y.data(), n, p);
    EXPECT_GT(op.last_relative_residual(), 1e-6) << "lambda=" << lambda;
  }
  EXPECT_GT(undetected, 0) << "every singular shift was flagged: update the notes in factor.hpp";
}

TYPED_TEST(OpsShiftInvert, RefactorizingTheFactorInvalidatesTheOperator) {
  using T = TypeParam;
  ShiftInvertCase<T> c(200, true);
  ShiftedPencil<T> pencil(c.a, c.b_ptr());
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -c.sigma);
  ASSERT_TRUE(f->factorize(1.0, -c.sigma));
  hs::ShiftInvertOperator<T> op(*f, pencil, c.sigma);
  const auto x = random_block<T>(c.n, 1, c.n, 5);
  std::vector<T> y(static_cast<std::size_t>(c.n));
  op.apply(x.data(), c.n, y.data(), c.n, 1);  // fine before the refactorization

  ASSERT_TRUE(f->factorize(1.0, -(c.sigma + 1e-3)));
  EXPECT_EQ(error_code_of([&] { op.apply(x.data(), c.n, y.data(), c.n, 1); }),
            hs::Error::Code::Internal);
}

TYPED_TEST(OpsShiftInvert, ConstructionRequiresAUsableMatchingFactor) {
  using T = TypeParam;
  using Code = hs::Error::Code;
  const index_t nx = 50, ny = 50;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, 1.0); }),
            Code::InvalidInput);  // not analysed
  f->analyze(pencil, 1.0, -3.0);
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, 1.0); }),
            Code::InvalidInput);  // not factorized
  const double lambda = w[w.size() / 2];
  ASSERT_FALSE(f->factorize(1.0, -lambda));  // singular shift
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, lambda); }),
            Code::InvalidInput);

  ASSERT_TRUE(f->factorize(1.0, -3.0));
  const auto other = hs::test::laplacian_1d<T>(10);
  ShiftedPencil<T> wrong_order(other, nullptr);
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, wrong_order, 3.0); }),
            Code::InvalidInput);
  hs::SolveRefinement negative;
  negative.max_steps = -1;
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, 3.0, negative); }),
            Code::InvalidInput);
  hs::SolveRefinement nan_tol;
  nan_tol.max_steps = 2;
  nan_tol.rel_tol = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, 3.0, nan_tol); }),
            Code::InvalidInput);
  for (double bad_sigma :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, bad_sigma); }),
              Code::InvalidInput);
}

}  // namespace
