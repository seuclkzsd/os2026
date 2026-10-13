// Proven condition bounds of a general HPD B (P8 "certcost", src/driver/norms.* prove_condition_bound,
// src/driver/certificate.* b_bounds, src/verify/single_count.cpp SingleCountRule::b_min_proven): the
// incomplete-Cholesky proof of lambda_min(W B W) plus a Gershgorin bound of lambda_max give a
// rigorous kappa for the randomized check and a lambda_min(B) for single counts, so that both
// certify on a B that is neither diagonal nor diagonally dominant. Real and complex.
//
// Test pencil: the bilinear (Q1) finite-element pair on an m1 x m2 grid,
//   A = K(m1) x M(m2) + M(m1) x K(m2) - c B,   B = M(m1) x M(m2),
// K = tridiag(-1, 2, -1), M = tridiag(1, 4, 1) / 6 (x: Kronecker product). K and M share their
// eigenvectors, so the pencil's eigenvalues are mu_i(m1) + mu_j(m2) - c, mu_i(m) = (2 - 2 cos t) /
// ((4 + 2 cos t) / 6), t = i pi / (m + 1). B is a 2-D mass matrix: diagonal 16/36 against 20/36 off
// it per interior row - not diagonally dominant - with lambda_min(W B W) about 1/4. Complex T: the
// gauge transform D^H (A, B) D, D = diag(exp(i phi_v)), keeps the spectrum.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "driver/certificate.hpp"
#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/verify.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

constexpr index_t kM1 = 23, kM2 = 21;
constexpr double kShift = 20.0;  // A = ... - kShift B: eigenvalues in (-kShift, ...)
constexpr double kSigma = 0.3;
constexpr index_t kK = 8;

double mu(index_t i, index_t m) {
  const double t = static_cast<double>(i) * M_PI / static_cast<double>(m + 1);
  return (2.0 - 2.0 * std::cos(t)) / ((4.0 + 2.0 * std::cos(t)) / 6.0);
}

std::vector<double> spectrum() {
  std::vector<double> w;
  for (index_t i = 1; i <= kM1; ++i)
    for (index_t j = 1; j <= kM2; ++j) w.push_back(mu(i, kM1) + mu(j, kM2) - kShift);
  return w;
}

std::vector<double> nearest(std::vector<double> w, double sigma, index_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
  w.resize(static_cast<std::size_t>(k));
  return w;
}

void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

/// 1-D tridiagonal (lo, d, lo) of order m, entry (i, j).
double tri(index_t i, index_t j, double d, double lo) { return i == j ? d : (std::abs(i - j) == 1 ? lo : 0.0); }

/// The Q1 pencil (header comment).
template <class T>
std::pair<hs::CsrMatrix<T>, hs::CsrMatrix<T>> q1_pencil() {
  const index_t n = kM1 * kM2;
  const auto phase = [](index_t v) { return 0.37 * static_cast<double>(v % 53); };
  std::vector<index_t> rows, cols;
  std::vector<T> va, vb;
  for (index_t i1 = 0; i1 < kM1; ++i1)
    for (index_t j1 = std::max<index_t>(0, i1 - 1); j1 <= std::min(kM1 - 1, i1 + 1); ++j1)
      for (index_t i2 = 0; i2 < kM2; ++i2)
        for (index_t j2 = std::max<index_t>(0, i2 - 1); j2 <= std::min(kM2 - 1, i2 + 1); ++j2) {
          const index_t r = i1 * kM2 + i2, c = j1 * kM2 + j2;
          const double m1 = tri(i1, j1, 4.0 / 6.0, 1.0 / 6.0), m2 = tri(i2, j2, 4.0 / 6.0, 1.0 / 6.0);
          const double k1 = tri(i1, j1, 2.0, -1.0), k2 = tri(i2, j2, 2.0, -1.0);
          const double b = m1 * m2, a = k1 * m2 + m1 * k2 - kShift * b;
          T ga(a), gb(b);
          if constexpr (hs::is_complex_v<T>) {
            const cplx g = std::polar(1.0, phase(c) - phase(r));
            ga *= g;
            gb *= g;
          }
          rows.push_back(r);
          cols.push_back(c);
          va.push_back(ga);
          vb.push_back(gb);
        }
  return {hs::csr_from_coo<T>(n, n, rows, cols, va), hs::csr_from_coo<T>(n, n, rows, cols, vb)};
}

hs::SolverConfig nearest_cfg(hs::CertificateMode mode) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = kSigma;
  cfg.target.k = kK;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.certificate = mode;
  return cfg;
}

template <class T>
class DriverConditionBoundTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(DriverConditionBoundTyped, Scalars);

// The bounds hold against dense eigenvalues: lambda_min(W B W) >= lower, lambda_max <= upper, so
// cond(W B W) <= kappa, and lambda_min(B) >= b_min; W are the Jacobi weights 1 / sqrt(b_ii).
TYPED_TEST(DriverConditionBoundTyped, BoundsHoldForAMassMatrix) {
  using T = TypeParam;
  const auto pencil = q1_pencil<T>();
  const hs::CsrMatrix<T>& b = pencil.second;
  // Not diagonally dominant: the O(nnz) Gershgorin route has no bound.
  EXPECT_FALSE(hs::start_scaling_for<T>(&b, 100.0, false).rigorous);
  const hs::driver::ConditionBound cb = hs::driver::prove_condition_bound(b);
  ASSERT_TRUE(cb.proven) << cb.proof.outcome;
  ASSERT_EQ(cb.weights.size(), static_cast<std::size_t>(b.nrows));
  // W B W explicitly, and its spectrum (n = 483, dense reference).
  hs::CsrMatrix<T> s = b;
  for (index_t i = 0; i < s.nrows; ++i)
    for (index_t q = s.rowptr[i]; q < s.rowptr[i + 1]; ++q) s.values[q] *= cb.weights[i] * cb.weights[s.colind[q]];
  const std::vector<double> ws = hs::test::dense_eigenvalues<T>(s, nullptr);
  const std::vector<double> wb = hs::test::dense_eigenvalues<T>(b, nullptr);
  EXPECT_GT(cb.lower, 0.0);
  EXPECT_LE(cb.lower, ws.front());
  EXPECT_GE(cb.upper, ws.back());
  EXPECT_GE(cb.kappa, ws.back() / ws.front());
  EXPECT_GT(cb.b_min, 0.0);
  EXPECT_LE(cb.b_min, wb.front());
  // Not loose beyond what the proof's shift costs: lower is within a factor 10 of lambda_min.
  EXPECT_GT(cb.lower, 0.1 * ws.front());
  EXPECT_LT(cb.kappa, 100.0 * ws.back() / ws.front());
}

// A B whose scaled lambda_min is below the proof's threshold: nothing is proven.
TYPED_TEST(DriverConditionBoundTyped, NearlySingularBIsNotProven) {
  using T = TypeParam;
  const auto l = hs::test::laplacian_2d<T>(12, 11);  // lambda_min(L / 4) ~ 0.035
  const hs::driver::ConditionBound cb = hs::driver::prove_condition_bound(l);
  EXPECT_FALSE(cb.proven);
  EXPECT_NE(cb.proof.outcome, "proven");
  EXPECT_FALSE(std::isfinite(cb.kappa));
  EXPECT_EQ(cb.b_min, 0.0);
}

// The single-count rule applies to a general B only with a proven lambda_min(B).
TYPED_TEST(DriverConditionBoundTyped, SingleRuleNeedsAProvenLambdaMin) {
  using T = TypeParam;
  const auto [a, b] = q1_pencil<T>();
  const hs::ShiftedPencil<T> pencil(a, &b);
  hs::SingleWindow w;
  w.lower = -0.5;
  w.upper = 1.5;
  w.wanted_lower = 0.0;
  w.wanted_upper = 1.0;
  hs::SingleCountRule rule;
  const auto none = hs::single_count_validity(pencil, w, rule);
  EXPECT_FALSE(none.applicable);
  EXPECT_NE(none.reason.find("no lower bound on lambda_min(B) is proven"), std::string::npos) << none.reason;
  rule.b_min_proven = hs::driver::prove_condition_bound(b).b_min;
  ASSERT_GT(rule.b_min_proven, 0.0);
  const auto proven = hs::single_count_validity(pencil, w, rule);
  EXPECT_TRUE(proven.applicable);
  EXPECT_EQ(proven.b_min, rule.b_min_proven);
  EXPECT_GT(proven.eta_lower, 0.0);
  rule.b_min_proven = -1.0;
  EXPECT_THROW(hs::single_count_validity(pencil, w, rule), hs::Error);
}

// End to end: the randomized check now certifies on this general B (it used to be refused: kappa
// only estimable), single counts certify by the rule (the certifier directly), and auto states a
// rigorous verdict.
TYPED_TEST(DriverConditionBoundTyped, RandomizedAndSingleCountsCertifyOnAGeneralB) {
  using T = TypeParam;
  const auto [a, b] = q1_pencil<T>();
  const std::vector<double> want = nearest(spectrum(), kSigma, kK);
  const auto rnd = hs::solve(a, &b, nearest_cfg(hs::CertificateMode::Randomized));
  EXPECT_EQ(rnd.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(rnd.report.success);
  const auto& c = rnd.report.certificate;
  EXPECT_EQ(c.method, "randomized") << c.statement;
  EXPECT_TRUE(c.rigorous);
  EXPECT_LE(c.failure_probability, 1e-12);
  EXPECT_GT(c.solves, 0);
  EXPECT_FALSE(has_warning(rnd.report, "SCREENING"));
  EXPECT_FALSE(has_warning(rnd.report, "kappa of B is not proven"));
  expect_same_set(rnd.eigenvalues, want, 1e-9);

  const auto aut = hs::solve(a, &b, nearest_cfg(hs::CertificateMode::Auto));
  EXPECT_TRUE(aut.report.success);
  EXPECT_TRUE(aut.report.certificate.rigorous);
  EXPECT_EQ(aut.report.certificate.statement.rfind("auto (round ", 0), 0u) << aut.report.certificate.statement;
  expect_same_set(aut.eigenvalues, want, 1e-9);

  // The certifier directly: single counts on the exact certificate's window.
  const auto ex = hs::solve(a, &b, nearest_cfg(hs::CertificateMode::Exact));
  ASSERT_TRUE(ex.report.certificate.passed);
  const hs::ShiftedPencil<T> pencil(a, &b);
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -kSigma);
  ASSERT_TRUE(fac->factorize(1.0, -kSigma));
  hs::driver::Block<T> all;
  all.lambda = ex.eigenvalues;
  all.x = ex.eigenvectors;
  hs::driver::recompute_products(a, &b, all);
  double r = 0.0;
  for (double l : ex.eigenvalues) r = std::max(r, std::abs(l - kSigma));
  const hs::driver::NearestWindow win{kSigma, r, ex.report.certificate.lower, ex.report.certificate.upper};
  hs::SolverConfig cfg = nearest_cfg(hs::CertificateMode::Auto);
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx ctx{cfg, nullptr, rep, clock};
  ctx.held = hs::driver::FactorHeld{kSigma, false};
  hs::driver::NearestCertifier<T> certifier(ctx, pencil, *fac, kSigma, false, ex.report.scale, ex.report.norm2_a,
                                            ex.report.norm2_b, {});
  const auto out = certifier.run(hs::CertificateMethod::ExactSingle, win, {all}, 0);
  EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Passed) << rep.certificate.statement;
  EXPECT_EQ(rep.certificate.method, "exact_single");
  EXPECT_TRUE(rep.certificate.rigorous);
  EXPECT_NE(rep.certificate.statement.find("general B with lambda_min(B) >= "), std::string::npos)
      << rep.certificate.statement;
  // ... and the randomized check through the certifier is rigorous with a finite proven kappa.
  const auto ch = certifier.choose(win, {all});
  EXPECT_TRUE(ch.kappa_rigorous);
}

}  // namespace
