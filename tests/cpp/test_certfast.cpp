// Tests of the randomized exclusion check (src/verify, include/hsolve/verify.hpp "(1)"): the
// failure-probability bound, the check on pencils with known spectra (complete X passes; a removed
// copy of a 4-fold eigenvalue and a removed edge eigenvalue are detected; standard, diagonal-B and
// general-B pencils; dense and MUMPS shift-invert), determinism, and the start scaling.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/verify.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::DenseMatrix;
using hs::ExclusionOptions;
using hs::ExclusionStatus;
using hs::index_t;

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

// ------------------------------------------------------------------------------- the bound

// Independent evaluation of the bound's formula (verify.hpp) in plain double arithmetic, for
// parameters where nothing overflows.
double reference_bound(double eps, index_t s, index_t big_n, index_t p, double kappa, bool cplx_scalars) {
  const double x1 = (2.0 + eps) / (2.0 - eps);
  const double t1 = std::cosh(static_cast<double>(s - 1) * std::acosh(x1));
  double t = (2.0 - eps) / (eps * t1 * t1 + 2.0 - eps);
  if (s >= 2) {
    const double q = (1.0 - eps) * (1.0 - eps), e2 = 1.0 - q;
    const double t2 = std::cosh(static_cast<double>((s - 2) / 2) * std::acosh((1.0 + e2) / (1.0 - e2)));
    t = std::min(t, q / (e2 * t2 * t2 + q));
  }
  const double f = cplx_scalars ? static_cast<double>(big_n - 1) * kappa * t : std::sqrt(big_n * kappa * t);
  return std::pow(std::min(f, 1.0), static_cast<double>(p));
}

TEST(VerifyBound, MatchesTheFormula) {
  for (bool c : {false, true})
    for (double eps : {0.05, 0.3, 0.7})
      for (index_t s : {1, 2, 3, 6, 11})
        for (index_t p : {1, 3}) {
          const double got = hs::exclusion_failure_bound(eps, s, 5000, p, 4.0, c);
          const double want = reference_bound(eps, s, 5000, p, 4.0, c);
          EXPECT_NEAR(got, want, 1e-12 * std::max(want, 1e-300) + 1e-300)
              << "eps=" << eps << " s=" << s << " p=" << p << " complex=" << c;
        }
}

TEST(VerifyBound, StepsReachDeltaAndAreMinimal) {
  for (bool c : {false, true})
    for (double eps : {0.02, 0.25, 0.6, 0.99})
      for (index_t p : {1, 4})
        for (double delta : {1e-6, 1e-12}) {
          const index_t big_n = 100000;
          const index_t s = hs::exclusion_block_steps(eps, delta, big_n, p, 1.0, c);
          ASSERT_GE(s, 1);
          EXPECT_LE(hs::exclusion_failure_bound(eps, s, big_n, p, 1.0, c), delta);
          if (s > 1) {
            EXPECT_GT(hs::exclusion_failure_bound(eps, s - 1, big_n, p, 1.0, c), delta);
          }
          // Monotone: more steps never hurt.
          for (index_t t = 1; t < s + 5; ++t)
            EXPECT_GE(hs::exclusion_failure_bound(eps, t, big_n, p, 1.0, c),
                      hs::exclusion_failure_bound(eps, t + 1, big_n, p, 1.0, c));
        }
}

// The worked example quoted in verify.hpp (real, N = 1e5, kappa = 1, p = 1).
TEST(VerifyBound, HeaderExampleStepCounts) {
  EXPECT_EQ(hs::exclusion_block_steps(0.99, 1e-12, 100000, 1, 1.0, false), 8);
  EXPECT_EQ(hs::exclusion_block_steps(0.25, 1e-6, 100000, 1, 1.0, false), 28);
  EXPECT_EQ(hs::exclusion_block_steps(0.25, 1e-12, 100000, 1, 1.0, false), 46);
}

TEST(VerifyBound, DependsOnGapDimensionKappaAndScalarsAsExpected) {
  const auto steps = [](double eps, index_t n, index_t p, double kappa, bool c) {
    return hs::exclusion_block_steps(eps, 1e-10, n, p, kappa, c);
  };
  EXPECT_GT(steps(0.1, 100000, 1, 1.0, false), steps(0.5, 100000, 1, 1.0, false));   // gap
  EXPECT_GE(steps(0.3, 1000000, 1, 1.0, false), steps(0.3, 1000, 1, 1.0, false));    // log N
  EXPECT_GE(steps(0.3, 100000, 1, 1e4, false), steps(0.3, 100000, 1, 1.0, false));   // log kappa
  EXPECT_LE(steps(0.3, 100000, 1, 1.0, true), steps(0.3, 100000, 1, 1.0, false));    // complex
  EXPECT_LT(steps(0.3, 100000, 4, 1.0, false), steps(0.3, 100000, 1, 1.0, false));   // blocks
  // No statement without a gap; a covering space is exact.
  EXPECT_EQ(hs::exclusion_failure_bound(0.0, 10, 1000, 1, 1.0, false), 1.0);
  EXPECT_EQ(hs::exclusion_block_steps(-0.1, 1e-6, 1000, 1, 1.0, false), -1);
  EXPECT_EQ(hs::exclusion_block_steps(1e-9, 1e-12, 40, 4, 1.0, false), 11);
  EXPECT_EQ(hs::exclusion_failure_bound(0.5, 3, 0, 1, 1.0, false), 0.0);
  EXPECT_EQ(error_code_of([] { hs::exclusion_failure_bound(0.5, 0, 10, 1, 1.0, false); }),
            hs::Error::Code::InvalidInput);
  EXPECT_EQ(error_code_of([] { hs::exclusion_failure_bound(0.5, 1, 10, 1, 0.5, false); }),
            hs::Error::Code::InvalidInput);
}

// The deterministic core of the bound: a start vector whose component along the top eigenvector
// is just above the threshold t*(eps, s) yields max |Ritz| >= (1 - eps) Theta after s steps, even
// with the rest of its mass placed adversarially (just below (1 - eps) Theta and at -Theta
// excluded, the worst points of the two Chebyshev arguments).
TEST(VerifyBound, ChebyshevLemmaHoldsForAdversarialStart) {
  const index_t n = 400;
  const double theta = 1.0;
  for (double eps : {0.1, 0.4, 0.8}) {
    for (index_t s : {4, 7, 10}) {
      // t* of the formula (both Chebyshev variants), from the reference implementation.
      const double t_star = reference_bound(eps, s, 1, 1, 1.0, false);  // sqrt(1 * t*) with N = 1
      const double beta2 = std::min(1.0, 1.0001 * t_star * t_star);
      std::vector<double> d(static_cast<std::size_t>(n));
      d[0] = theta;
      for (index_t i = 1; i < n; ++i)  // bad spectrum: [-(1-eps) Theta, (1-eps) Theta) dense
        d[i] = (1.0 - eps) * theta * (-1.0 + 2.0 * static_cast<double>(i - 1) / static_cast<double>(n - 1));
      hs::test::DiagonalOperator<double> op(d);
      DenseMatrix<double> b(n, 1);
      b(0, 0) = std::sqrt(beta2);
      // Mass just below (1 - eps) Theta and at the bottom of the bad interval.
      b(n - 1, 0) = std::sqrt(0.5 * (1.0 - beta2));
      b(1, 0) = std::sqrt(0.5 * (1.0 - beta2));
      hs::KrylovSchurOptions<double> ko;
      ko.nev = 1;
      ko.ncv = s;
      ko.max_restarts = 0;
      ko.tol = 1e-300;
      ko.initial = &b;
      const auto res = hs::lanczos_krylov_schur<double>(op, nullptr, hs::Which::LargestMagnitude, ko);
      double rho = 0.0;
      for (double t : res.ritz_all) rho = std::max(rho, std::abs(t));
      EXPECT_GE(rho, (1.0 - eps) * theta * (1.0 - 1e-12)) << "eps=" << eps << " s=" << s << " beta2=" << beta2;
    }
  }
}

// ------------------------------------------------------------------------------- pencils

enum class BKind { Identity, Diagonal, General };

/// A x = lambda B x with exactly the prescribed spectrum (A = C^H Ahat C, B = C^H C), its
/// B-orthonormal eigenvectors (dense, n <= 800) and helpers to build the operators.
template <class T>
struct Pencil {
  CsrMatrix<T> a, b;
  bool has_b = false;
  std::vector<double> lambda;  ///< ascending
  DenseMatrix<T> v;            ///< B-orthonormal eigenvectors, column j <-> lambda[j]
  const CsrMatrix<T>* b_ptr() const { return has_b ? &b : nullptr; }
};

template <class T>
Pencil<T> make_pencil(const std::vector<double>& spectrum, BKind kind, std::uint64_t seed) {
  const index_t n = static_cast<index_t>(spectrum.size());
  Pencil<T> pc;
  const CsrMatrix<T> ahat = hs::test::prescribed_spectrum<T>(spectrum, 4, seed);
  if (kind == BKind::Identity) {
    pc.a = ahat;
  } else {
    // C = D^{1/2} (diagonal) or D^{1/2} + upper bidiagonal coupling (general).
    std::vector<double> u(static_cast<std::size_t>(n));
    hs::linalg::fill_random(u.data(), n, seed + 1);
    DenseMatrix<T> c(n, n);
    for (index_t i = 0; i < n; ++i) {
      c(i, i) = T(std::sqrt(1.0 + 1.5 * (u[i] + 1.0)));
      if (kind == BKind::General && i + 1 < n) c(i, i + 1) = T(0.35 * u[(i * 7 + 3) % n]);
    }
    const DenseMatrix<T> ad = hs::test::to_dense(ahat);
    DenseMatrix<T> ac(n, n), cac(n, n), ctc(n, n);
    hs::linalg::multiply(n, n, n, ad.data(), n, c.data(), n, ac.data(), n);
    hs::lapack::gemm('C', 'N', static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), T(1), c.data(),
                     static_cast<int>(n), ac.data(), static_cast<int>(n), T(0), cac.data(), static_cast<int>(n));
    hs::lapack::gemm('C', 'N', static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), T(1), c.data(),
                     static_cast<int>(n), c.data(), static_cast<int>(n), T(0), ctc.data(), static_cast<int>(n));
    for (DenseMatrix<T>* m : {&cac, &ctc})  // exactly Hermitian
      for (index_t j = 0; j < n; ++j)
        for (index_t i = 0; i <= j; ++i) {
          const T h = T(0.5) * ((*m)(i, j) + hs::conj_if((*m)(j, i)));
          (*m)(i, j) = i == j ? T(std::real(h)) : h;
          (*m)(j, i) = hs::conj_if((*m)(i, j));
        }
    pc.a = hs::test::from_dense(cac, 1e-300);
    pc.b = hs::test::from_dense(ctc, 1e-300);
    pc.has_b = true;
  }
  DenseMatrix<T> ad = hs::test::to_dense(pc.a);
  pc.lambda.resize(static_cast<std::size_t>(n));
  if (pc.has_b) {
    DenseMatrix<T> bd = hs::test::to_dense(pc.b);
    EXPECT_EQ(hs::lapack::hegv('V', 'U', static_cast<int>(n), ad.data(), static_cast<int>(n), bd.data(),
                               static_cast<int>(n), pc.lambda.data()),
              0);
  } else {
    hs::lapack::heev('V', 'U', static_cast<int>(n), ad.data(), static_cast<int>(n), pc.lambda.data());
  }
  pc.v = std::move(ad);
  return pc;
}

/// Columns of the eigenvectors with the given indices.
template <class T>
DenseMatrix<T> columns(const DenseMatrix<T>& v, const std::vector<index_t>& idx) {
  DenseMatrix<T> x(v.rows(), static_cast<index_t>(idx.size()));
  for (std::size_t c = 0; c < idx.size(); ++c)
    std::copy(v.col(idx[c]), v.col(idx[c]) + v.rows(), x.col(static_cast<index_t>(c)));
  return x;
}

/// Indices of the eigenvalues within distance r of sigma (inclusive, relative slack 1e-9).
std::vector<index_t> within(const std::vector<double>& lambda, double sigma, double r) {
  std::vector<index_t> idx;
  for (index_t i = 0; i < static_cast<index_t>(lambda.size()); ++i)
    if (std::abs(lambda[i] - sigma) <= r * (1.0 + 1e-9)) idx.push_back(i);
  return idx;
}

/// The test spectrum (n values): a wanted group around sigma = 0.3 with a 4-fold eigenvalue
/// (0.25, 0.27, 0.31, 0.33 x4, 0.36: radius 0.06), the nearest others at distance 0.12 and 0.13,
/// the rest spread over [-8, -0.5] and [1, 8].
constexpr double kSigma = 0.3;
constexpr double kRadius = 0.06;
std::vector<double> test_spectrum(index_t n) {
  std::vector<double> s = {0.25, 0.27, 0.31, 0.33, 0.33, 0.33, 0.33, 0.36, 0.42, 0.17};
  const index_t rest = n - static_cast<index_t>(s.size());
  for (index_t i = 0; i < rest; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(rest);
    s.push_back(i % 2 == 0 ? -8.0 + 7.5 * t : 1.0 + 7.0 * t);
  }
  return s;
}

template <class T>
struct Setup {
  Pencil<T> pc;
  DenseMatrix<T> ad, bd;
  std::unique_ptr<hs::test::DenseShiftInvertOperator<T>> op;
  std::unique_ptr<hs::CsrOperator<T>> b_op;
  explicit Setup(BKind kind, index_t n = 400, std::uint64_t seed = 11)
      : pc(make_pencil<T>(test_spectrum(n), kind, seed)) {
    ad = hs::test::to_dense(pc.a);
    if (pc.has_b) {
      bd = hs::test::to_dense(pc.b);
      b_op = std::make_unique<hs::CsrOperator<T>>(pc.b);
    }
    op = std::make_unique<hs::test::DenseShiftInvertOperator<T>>(ad, pc.has_b ? &bd : nullptr, kSigma);
  }
  hs::LinearOperator<T>* b_inner() { return b_op.get(); }
};

template <class T>
class VerifyExclusion : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(VerifyExclusion, Scalars);

ExclusionOptions default_options(double pass_radius = 0.1) {
  ExclusionOptions o;
  o.pass_radius = pass_radius;
  o.delta = 1e-12;
  o.block_size = 4;
  o.seed = 7;
  return o;
}

TYPED_TEST(VerifyExclusion, CompleteSetPassesStandardProblem) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const auto idx = within(st.pc.lambda, kSigma, kRadius);
  ASSERT_EQ(idx.size(), 8u);
  const DenseMatrix<T> x = columns(st.pc.v, idx);
  const auto opt = default_options();
  const auto r = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, opt);
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  EXPECT_FALSE(r.exact);
  EXPECT_LE(r.failure_probability, opt.delta);
  EXPECT_EQ(r.found_inside, 8);
  // 0.36 sits exactly at distance r: within e of the edge, so Kahan's theorem cannot place it
  // inside; the statement still bounds the count by found_inside = 8.
  EXPECT_EQ(r.certified_inside, 7);
  EXPECT_EQ(r.deflated, 8);
  EXPECT_EQ(r.dimension, 392);
  EXPECT_LT(r.coupling, 1e-8);
  EXPECT_NEAR(r.epsilon, 0.4, 1e-5);  // 1 - (1/0.1) / (1/0.06)
  EXPECT_EQ(r.new_count, 0);
  EXPECT_EQ(r.new_vectors.cols(), 0);
  // The remaining spectrum's largest |theta| is 1/0.12: rho approaches it from below.
  EXPECT_LT(r.max_ritz, 1.0 / 0.12 * (1 + 1e-10));
  EXPECT_GT(r.max_ritz, 0.5 / 0.12);
  // Cost: m applications for the coupling + s blocks of p.
  EXPECT_EQ(r.block_steps, r.block_steps_planned);
  EXPECT_EQ(r.op_applications, 8 + r.block_steps * 4);
  EXPECT_EQ(r.b_applications, 0);
  EXPECT_EQ(r.deflated_theta.size(), 8u);
  EXPECT_NEAR(std::abs(r.deflated_theta.front()), 1.0 / 0.01, 1e-6);  // 0.31 is nearest
  EXPECT_FALSE(r.statement.empty());
}

TYPED_TEST(VerifyExclusion, DetectsARemovedCopyOfAFourFoldEigenvalue) {
  using T = TypeParam;
  for (BKind kind : {BKind::Identity, BKind::Diagonal, BKind::General}) {
    Setup<T> st(kind);
    auto idx = within(st.pc.lambda, kSigma, kRadius);
    ASSERT_EQ(idx.size(), 8u);
    // Remove one copy of 0.33.
    const auto it = std::find_if(idx.begin(), idx.end(), [&](index_t i) { return std::abs(st.pc.lambda[i] - 0.33) < 1e-9; });
    ASSERT_NE(it, idx.end());
    idx.erase(it);
    auto opt = default_options();
    hs::StartScaling sc = hs::start_scaling_for<T>(st.pc.b_ptr());
    if (!sc.scaling.empty()) opt.start_scaling = &sc.scaling;
    opt.b_condition = sc.kappa;
    const auto r = hs::exclusion_check<T>(*st.op, st.b_inner(), columns(st.pc.v, idx), kSigma, kRadius, opt);
    ASSERT_EQ(r.status, ExclusionStatus::Detected) << r.statement;
    EXPECT_EQ(r.found_inside, 7);
    EXPECT_EQ(r.new_count, 1);
    EXPECT_EQ(r.new_inside, 1);
    ASSERT_GE(r.new_lambda.size(), 1u);
    EXPECT_NEAR(r.new_lambda[0], 0.33, 1e-8);
    EXPECT_TRUE(r.new_converged[0]);
    EXPECT_EQ(r.new_vectors.cols(), static_cast<index_t>(r.new_lambda.size()));
  }
}

TYPED_TEST(VerifyExclusion, DetectsARemovedEdgeEigenvalue) {
  using T = TypeParam;
  Setup<T> st(BKind::General);
  auto idx = within(st.pc.lambda, kSigma, kRadius);
  ASSERT_EQ(idx.size(), 8u);
  // 0.36 is the found eigenvalue nearest the window edge (distance r).
  const auto it = std::find_if(idx.begin(), idx.end(), [&](index_t i) { return std::abs(st.pc.lambda[i] - 0.36) < 1e-9; });
  ASSERT_NE(it, idx.end());
  idx.erase(it);
  auto opt = default_options();
  hs::StartScaling sc = hs::start_scaling_for<T>(st.pc.b_ptr());
  opt.start_scaling = &sc.scaling;
  opt.b_condition = sc.kappa;
  const auto r = hs::exclusion_check<T>(*st.op, st.b_inner(), columns(st.pc.v, idx), kSigma, kRadius, opt);
  ASSERT_EQ(r.status, ExclusionStatus::Detected) << r.statement;
  EXPECT_EQ(r.new_count, 1);
  ASSERT_GE(r.new_lambda.size(), 1u);
  EXPECT_NEAR(r.new_lambda[0], 0.36, 1e-8);
}

TYPED_TEST(VerifyExclusion, CompleteSetPassesGeneralizedProblems) {
  using T = TypeParam;
  for (BKind kind : {BKind::Diagonal, BKind::General}) {
    Setup<T> st(kind);
    const auto idx = within(st.pc.lambda, kSigma, kRadius);
    ASSERT_EQ(idx.size(), 8u);
    auto opt = default_options();
    hs::StartScaling sc = hs::start_scaling_for<T>(st.pc.b_ptr());
    if (kind == BKind::Diagonal) {
      EXPECT_TRUE(sc.rigorous) << sc.note;
    }
    opt.start_scaling = &sc.scaling;
    opt.b_condition = sc.kappa;
    const auto r = hs::exclusion_check<T>(*st.op, st.b_inner(), columns(st.pc.v, idx), kSigma, kRadius, opt);
    EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
    EXPECT_EQ(r.found_inside, 8);
    EXPECT_GT(r.b_applications, 0);
    EXPECT_EQ(r.kappa, sc.kappa);
    if (kind == BKind::Diagonal) {
      EXPECT_EQ(r.kappa, 1.0);
    }
  }
}

// A legitimate eigenvalue inside the pass radius (0.42 at distance 0.12 < r' = 0.125) is
// detected, but outside the claim radius.
TYPED_TEST(VerifyExclusion, NeighbourInsidePassRadiusIsReportedOutsideTheClaim) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const auto idx = within(st.pc.lambda, kSigma, kRadius);
  const auto r = hs::exclusion_check<T>(*st.op, nullptr, columns(st.pc.v, idx), kSigma, kRadius, default_options(0.125));
  ASSERT_EQ(r.status, ExclusionStatus::Detected) << r.statement;
  EXPECT_EQ(r.new_count, 1);
  EXPECT_EQ(r.new_inside, 0);
  ASSERT_GE(r.new_lambda.size(), 1u);
  EXPECT_NEAR(r.new_lambda[0], 0.42, 1e-6);
}

TYPED_TEST(VerifyExclusion, DeterministicForAFixedSeedAndPassingForOthers) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const DenseMatrix<T> x = columns(st.pc.v, within(st.pc.lambda, kSigma, kRadius));
  auto opt = default_options();
  const auto r1 = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, opt);
  const auto r2 = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, opt);
  ASSERT_EQ(r1.ritz_all.size(), r2.ritz_all.size());
  for (std::size_t i = 0; i < r1.ritz_all.size(); ++i) EXPECT_EQ(r1.ritz_all[i], r2.ritz_all[i]);
  for (std::uint64_t seed : {1u, 2u, 3u, 1000u}) {
    opt.seed = seed;
    for (index_t p : {1, 2, 4}) {
      opt.block_size = p;
      const auto r = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, opt);
      EXPECT_EQ(r.status, ExclusionStatus::Passed) << "seed " << seed << " p " << p << ": " << r.statement;
      EXPECT_EQ(r.block_size, p);
    }
  }
}

// Without found vectors the check asks whether any eigenvalue lies within r.
TYPED_TEST(VerifyExclusion, EmptySetChecksTheWholeBall) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const DenseMatrix<T> none(st.pc.a.nrows, 0);
  auto opt = default_options(0.009);
  const auto clear = hs::exclusion_check<T>(*st.op, nullptr, none, kSigma, 0.005, opt);
  EXPECT_EQ(clear.status, ExclusionStatus::Passed) << clear.statement;
  EXPECT_EQ(clear.found_inside, 0);
  EXPECT_EQ(clear.op_applications, clear.block_steps * opt.block_size);
  opt.pass_radius = 0.05;
  const auto hit = hs::exclusion_check<T>(*st.op, nullptr, none, kSigma, 0.02, opt);
  EXPECT_EQ(hit.status, ExclusionStatus::Detected);
  ASSERT_GE(hit.new_lambda.size(), 1u);
  EXPECT_NEAR(hit.new_lambda[0], 0.31, 1e-8);
}

// A complement smaller than the planned Krylov space is searched exactly.
TYPED_TEST(VerifyExclusion, SmallComplementIsSearchedExactly) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity, 40);
  const DenseMatrix<T> x = columns(st.pc.v, within(st.pc.lambda, kSigma, kRadius));
  const auto r = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, default_options());
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  EXPECT_TRUE(r.exact);
  EXPECT_EQ(r.failure_probability, 0.0);
  EXPECT_EQ(static_cast<index_t>(r.ritz_all.size()), 32);
  EXPECT_NEAR(r.max_ritz, 1.0 / 0.12, 1e-8);
}

TYPED_TEST(VerifyExclusion, NonOrthonormalBasisOfTheSameSpanGivesTheSameVerdict) {
  using T = TypeParam;
  Setup<T> st(BKind::General);
  const DenseMatrix<T> x = columns(st.pc.v, within(st.pc.lambda, kSigma, kRadius));
  DenseMatrix<T> mixed(x.rows(), x.cols());
  DenseMatrix<T> rmat(x.cols(), x.cols());
  for (index_t j = 0; j < x.cols(); ++j)
    for (index_t i = 0; i <= j; ++i) rmat(i, j) = T(i == j ? 2.0 + j : 0.3 * (i + 1));
  hs::linalg::multiply(x.rows(), x.cols(), x.cols(), x.data(), x.rows(), rmat.data(), x.cols(), mixed.data(),
                       x.rows());
  auto opt = default_options();
  hs::StartScaling sc = hs::start_scaling_for<T>(st.pc.b_ptr());
  opt.start_scaling = &sc.scaling;
  opt.b_condition = sc.kappa;
  const auto r = hs::exclusion_check<T>(*st.op, st.b_inner(), mixed, kSigma, kRadius, opt);
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  EXPECT_EQ(r.found_inside, 8);
  EXPECT_LT(r.coupling, 1e-7);
}

TYPED_TEST(VerifyExclusion, SuppliedCouplingSavesTheBlockApplication) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const auto idx = within(st.pc.lambda, kSigma, kRadius);
  std::vector<double> theta;
  for (index_t i : idx) theta.push_back(1.0 / (st.pc.lambda[i] - kSigma));
  auto opt = default_options();
  opt.coupling_bound = 1e-12;
  opt.deflated_theta = &theta;
  const auto r = hs::exclusion_check<T>(*st.op, nullptr, columns(st.pc.v, idx), kSigma, kRadius, opt);
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  EXPECT_EQ(r.op_applications, r.block_steps * 4);
  EXPECT_EQ(r.coupling, 1e-12);
  EXPECT_EQ(r.found_inside, 8);
}

TYPED_TEST(VerifyExclusion, NoStatementWithoutGapOrWithinTheStepCap) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity);
  const auto idx = within(st.pc.lambda, kSigma, kRadius);
  const DenseMatrix<T> x = columns(st.pc.v, idx);
  // A coupling as large as the gap between 1/r and 1/r' leaves no eps.
  std::vector<double> theta(idx.size(), 50.0);
  auto opt = default_options();
  opt.coupling_bound = 1.0 / kRadius - 1.0 / opt.pass_radius;
  opt.deflated_theta = &theta;
  const auto r = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, opt);
  EXPECT_EQ(r.status, ExclusionStatus::Inconclusive);
  EXPECT_LE(r.epsilon, 0.0);
  EXPECT_EQ(r.op_applications, 0);
  EXPECT_EQ(r.failure_probability, 1.0);
  // One block step cannot reach delta = 1e-12.
  auto capped = default_options();
  capped.max_block_steps = 1;
  const auto c = hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, kRadius, capped);
  EXPECT_EQ(c.status, ExclusionStatus::Inconclusive) << c.statement;
  EXPECT_GT(c.failure_probability, 1e-12);
}

TYPED_TEST(VerifyExclusion, InvalidInputsThrow) {
  using T = TypeParam;
  Setup<T> st(BKind::Identity, 60);
  const index_t n = st.pc.a.nrows;
  const DenseMatrix<T> x = columns(st.pc.v, within(st.pc.lambda, kSigma, kRadius));
  const auto bad = [&](const std::function<void(ExclusionOptions&)>& edit, double radius = kRadius) {
    auto o = default_options();
    edit(o);
    return error_code_of([&] { hs::exclusion_check<T>(*st.op, nullptr, x, kSigma, radius, o); });
  };
  const auto invalid = hs::Error::Code::InvalidInput;
  EXPECT_EQ(bad([](ExclusionOptions& o) { o.pass_radius = kRadius; }), invalid);
  EXPECT_EQ(bad([](ExclusionOptions&) {}, -1.0), invalid);
  EXPECT_EQ(bad([](ExclusionOptions& o) { o.delta = 1.0; }), invalid);
  EXPECT_EQ(bad([](ExclusionOptions& o) { o.block_size = 0; }), invalid);
  EXPECT_EQ(bad([](ExclusionOptions& o) { o.rounding_margin = 0.5; }), invalid);
  EXPECT_EQ(bad([](ExclusionOptions& o) { o.coupling_bound = 1e-9; }), invalid);  // no deflated_theta
  std::vector<double> short_scaling(static_cast<std::size_t>(n - 1), 1.0);
  EXPECT_EQ(bad([&](ExclusionOptions& o) { o.start_scaling = &short_scaling; }), invalid);
  // Wrong row count, rank-deficient x.
  DenseMatrix<T> wrong(n + 1, 2);
  EXPECT_EQ(error_code_of([&] { hs::exclusion_check<T>(*st.op, nullptr, wrong, kSigma, kRadius, default_options()); }),
            invalid);
  DenseMatrix<T> dup = x;
  std::copy(dup.col(0), dup.col(0) + n, dup.col(1));
  EXPECT_EQ(error_code_of([&] { hs::exclusion_check<T>(*st.op, nullptr, dup, kSigma, kRadius, default_options()); }),
            invalid);
}

// The check on a MUMPS shift-invert operator: a 2-D Laplacian on a square grid has exact
// 2-fold eigenvalues; remove one copy of the nearest pair.
TYPED_TEST(VerifyExclusion, MumpsShiftInvertDetectsAMissingCopy) {
  using T = TypeParam;
  const index_t nx = 24;
  const CsrMatrix<T> a = hs::test::laplacian_2d<T>(nx, nx);
  const index_t n = nx * nx;
  DenseMatrix<T> v = hs::test::to_dense(a);
  std::vector<double> lambda(static_cast<std::size_t>(n));
  hs::lapack::heev('V', 'U', static_cast<int>(n), v.data(), static_cast<int>(n), lambda.data());
  // sigma just above a double eigenvalue in the lower part of the spectrum.
  index_t j = 40;
  while (!(std::abs(lambda[j + 1] - lambda[j]) < 1e-10 && lambda[j + 2] - lambda[j + 1] > 1e-3)) ++j;
  const double sigma = lambda[j + 1] + 0.1 * (lambda[j + 2] - lambda[j + 1]);
  std::vector<index_t> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](index_t p, index_t q) { return std::abs(lambda[p] - sigma) < std::abs(lambda[q] - sigma); });
  const index_t k = 6;
  // r = the k-th distance; r' half-way to the next distinct distance.
  double r = std::abs(lambda[order[k - 1]] - sigma);
  index_t next = k;
  while (std::abs(lambda[order[next]] - sigma) <= r * (1 + 1e-9)) ++next;  // keep whole multiplets
  r = std::abs(lambda[order[next - 1]] - sigma);
  const double d_next = std::abs(lambda[order[next]] - sigma);
  std::vector<index_t> found(order.begin(), order.begin() + next);

  hs::ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -sigma);
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  hs::ShiftInvertOperator<T> op(*f, pencil, sigma);
  auto opt = default_options(0.5 * (r + d_next));
  const auto pass = hs::exclusion_check<T>(op, nullptr, columns(v, found), sigma, r, opt);
  EXPECT_EQ(pass.status, ExclusionStatus::Passed) << pass.statement;
  EXPECT_EQ(pass.found_inside, next);
  // Drop one copy of the double eigenvalue nearest sigma (lambda[j] or lambda[j+1]).
  std::vector<index_t> missing;
  bool dropped = false;
  for (index_t i : found) {
    if (!dropped && (i == j || i == j + 1)) {
      dropped = true;
      continue;
    }
    missing.push_back(i);
  }
  ASSERT_TRUE(dropped);
  const auto det = hs::exclusion_check<T>(op, nullptr, columns(v, missing), sigma, r, opt);
  ASSERT_EQ(det.status, ExclusionStatus::Detected) << det.statement;
  EXPECT_EQ(det.new_inside, 1);
  ASSERT_GE(det.new_lambda.size(), 1u);
  EXPECT_NEAR(det.new_lambda[0], lambda[j], 1e-9);
}

// ------------------------------------------------------------------------------- start scaling

TYPED_TEST(VerifyExclusion, StartScalingKappaBoundsTheScaledCondition) {
  using T = TypeParam;
  hs::StartScaling none = hs::start_scaling_for<T>(nullptr);
  EXPECT_TRUE(none.scaling.empty());
  EXPECT_EQ(none.kappa, 1.0);
  EXPECT_TRUE(none.rigorous);
  for (BKind kind : {BKind::Diagonal, BKind::General}) {
    const Pencil<T> pc = make_pencil<T>(test_spectrum(200), kind, 5);
    const hs::StartScaling sc = hs::start_scaling_for<T>(&pc.b);
    ASSERT_EQ(sc.scaling.size(), 200u);
    // Dense cond(S B S).
    DenseMatrix<T> sbs = hs::test::to_dense(pc.b);
    for (index_t j = 0; j < 200; ++j)
      for (index_t i = 0; i < 200; ++i) sbs(i, j) *= T(sc.scaling[i] * sc.scaling[j]);
    std::vector<double> w(200);
    hs::lapack::heev('N', 'U', 200, sbs.data(), 200, w.data());
    const double cond = w.back() / w.front();
    EXPECT_GE(sc.kappa * (1 + 1e-12), cond) << sc.note;
    if (kind == BKind::Diagonal) {
      EXPECT_NEAR(cond, 1.0, 1e-12);
    }
  }
  // Diagonally dominant B: rigorous Gershgorin bound.
  const CsrMatrix<T> dd = hs::test::random_hpd<T>(150, 0.05, 3);
  const hs::StartScaling sd = hs::start_scaling_for<T>(&dd);
  EXPECT_TRUE(sd.rigorous) << sd.note;
  EXPECT_GE(sd.kappa, 1.0);
  // Not diagonally dominant: B = C^H C, C = I + N + 0.5 N^2 (N the shift matrix) is pentadiagonal
  // with off-diagonal row sums 4 against a diagonal of 2.25 - only an estimate is possible.
  {
    const index_t n = 200;
    DenseMatrix<T> c(n, n);
    for (index_t i = 0; i < n; ++i) {
      c(i, i) = T(1.0);
      if (i + 1 < n) c(i, i + 1) = T(1.0);
      if (i + 2 < n) c(i, i + 2) = T(0.5);
    }
    DenseMatrix<T> ctc(n, n);
    hs::lapack::gemm('C', 'N', static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), T(1), c.data(),
                     static_cast<int>(n), c.data(), static_cast<int>(n), T(0), ctc.data(), static_cast<int>(n));
    const CsrMatrix<T> b = hs::test::from_dense(ctc, 1e-300);
    const hs::StartScaling se = hs::start_scaling_for<T>(&b);
    EXPECT_FALSE(se.rigorous) << se.note;
    DenseMatrix<T> sbs = hs::test::to_dense(b);
    for (index_t j = 0; j < n; ++j)
      for (index_t i = 0; i < n; ++i) sbs(i, j) *= T(se.scaling[i] * se.scaling[j]);
    std::vector<double> w(static_cast<std::size_t>(n));
    hs::lapack::heev('N', 'U', static_cast<int>(n), sbs.data(), static_cast<int>(n), w.data());
    EXPECT_GE(se.kappa, w.back() / w.front()) << se.note;
  }
  CsrMatrix<T> bad = dd;
  bad.values[0] = T(-1.0);
  EXPECT_EQ(error_code_of([&] { hs::start_scaling_for<T>(&bad); }), hs::Error::Code::InvalidInput);
}

}  // namespace
