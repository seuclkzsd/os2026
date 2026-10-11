// End-to-end tests of hs::solve (driver + factor + ops + eigen). Every case has an analytic or a
// dense reference (n <= 2000), and checks the report contract (residuals, certificate, warnings).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

// The k reference eigenvalues nearest sigma, ordered by distance.
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

hs::SolverConfig nearest_cfg(double sigma, index_t k) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  return cfg;
}

// The phase times are exclusive: non-negative and summing to the total (report contract).
void expect_times_add_up(const hs::SolveReport& r) {
  EXPECT_GE(r.time_setup_s, 0.0);
  EXPECT_GE(r.time_factor_s, 0.0);
  EXPECT_GE(r.time_iterate_s, 0.0);
  EXPECT_GE(r.time_verify_s, 0.0);
  EXPECT_GE(r.time_certificate_s, 0.0);
  EXPECT_GT(r.time_total_s, 0.0);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s + r.time_certificate_s,
              r.time_total_s, 1e-12 * r.time_total_s);
}

template <class T>
void expect_report_ok(const hs::Solution<T>& sol, bool expect_certificate) {
  expect_times_add_up(sol.report);
  EXPECT_TRUE(sol.report.success);
  for (const auto& p : sol.report.pairs) {
    EXPECT_TRUE(p.accepted);
    EXPECT_LE(p.backward_error, 1e-10);
  }
  EXPECT_EQ(sol.report.certificate.performed, expect_certificate);
  if (expect_certificate) {
    EXPECT_TRUE(sol.report.certificate.passed);
  }
}

TEST(DriverSolve, StandardIndefiniteLaplacianNearest) {
  // 2-D Laplacian shifted by 4: indefinite, eigenvalues known; nx != ny avoids multiplicities.
  auto a = hs::test::laplacian_2d<double>(40, 37);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] == i) a.values[p] -= 4.0;
  auto w = hs::test::laplacian_2d_eigenvalues(40, 37);
  for (auto& x : w) x -= 4.0;
  const double sigma = 0.3;
  auto sol = hs::solve(a, nullptr, nearest_cfg(sigma, 8));
  expect_report_ok(sol, true);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 8), 1e-10);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::Standard);
  // Output order: nearest first.
  for (std::size_t i = 1; i < sol.eigenvalues.size(); ++i)
    EXPECT_LE(std::abs(sol.eigenvalues[i - 1] - sigma), std::abs(sol.eigenvalues[i] - sigma) + 1e-12);
}

TEST(DriverSolve, GeneralizedDefiniteMatchesDense) {
  auto a = hs::test::random_hermitian<double>(500, 0.02, 1);
  auto b = hs::test::random_hpd<double>(500, 0.01, 2);
  auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = 0.5 * (w[200] + w[201]);
  auto sol = hs::solve(a, &b, nearest_cfg(sigma, 10));
  expect_report_ok(sol, true);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 10), 1e-9);
}

TEST(DriverSolve, ComplexHermitianGeneralizedDefinite) {
  auto a = hs::test::random_hermitian<cplx>(300, 0.03, 3);
  auto b = hs::test::random_hpd<cplx>(300, 0.02, 4);
  auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = w[120] + 1e-3;
  auto sol = hs::solve(a, &b, nearest_cfg(sigma, 6));
  expect_report_ok(sol, true);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 6), 1e-9);
}

TEST(DriverSolve, MultiplicitiesAreAllFound) {
  // Square grid: many double eigenvalues; the certificate must force every copy to be found.
  auto a = hs::test::laplacian_2d<double>(30, 30);
  auto w = hs::test::laplacian_2d_eigenvalues(30, 30);
  const double sigma = w[300] + 1e-4;
  auto sol = hs::solve(a, nullptr, nearest_cfg(sigma, 12));
  expect_report_ok(sol, true);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 12), 1e-10);
}

TEST(DriverSolve, ShiftExactlyOnAnEigenvalue) {
  auto a = hs::test::laplacian_2d<double>(50, 51);
  auto w = hs::test::laplacian_2d_eigenvalues(50, 51);
  const double sigma = w[777];  // exactly singular A - sigma I (up to rounding of the formula)
  auto sol = hs::solve(a, nullptr, nearest_cfg(sigma, 5));
  expect_report_ok(sol, true);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 5), 1e-10);
}

// sigma inside a cluster whose spacing (2e-8) is below the null-pivot threshold (1e-8 * ||K|| ~ 1e-7):
// every nearby shift is flagged "singular". The driver must factorize AT sigma without detection
// and still certify the k nearest (endpoint counts at gap midpoints without detection).
TEST(DriverSolve, ShiftInsideDenseCluster) {
  std::vector<double> w;
  for (int i = -150; i < 150; ++i) w.push_back(2e-8 * i);
  for (int i = 0; i < 150; ++i) w.push_back(1.0 + 0.06 * i);
  for (int i = 0; i < 150; ++i) w.push_back(-1.0 - 0.06 * i);
  auto a = hs::test::prescribed_spectrum<double>(w, 4, 31);
  auto sol = hs::solve(a, nullptr, nearest_cfg(0.0, 5));  // {0, +-2e-8, +-4e-8}: unique at tol
  expect_report_ok(sol, true);
  ASSERT_EQ(sol.eigenvalues.size(), 5u);
  auto got = sol.eigenvalues;
  std::sort(got.begin(), got.end());
  const std::vector<double> want{-4e-8, -2e-8, 0.0, 2e-8, 4e-8};
  for (int i = 0; i < 5; ++i) EXPECT_NEAR(got[i], want[i], 1e-12);
}

TEST(DriverSolve, IntervalCountsWithInertia) {
  auto a = hs::test::laplacian_2d<double>(25, 24);
  auto w = hs::test::laplacian_2d_eigenvalues(25, 24);
  hs::SolverConfig cfg = nearest_cfg(0, 1);
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 1.0;
  cfg.target.upper = 1.6;
  std::vector<double> want;
  for (double x : w)
    if (x >= 1.0 && x <= 1.6) want.push_back(x);
  auto sol = hs::solve(a, nullptr, cfg);
  expect_report_ok(sol, true);
  expect_same_set(sol.eigenvalues, want, 1e-10);
  EXPECT_EQ(sol.report.certificate.count_inertia, static_cast<index_t>(want.size()));
}

TEST(DriverSolve, SmallestAndLargestStandard) {
  auto a = hs::test::laplacian_1d<double>(800);
  auto w = hs::test::laplacian_1d_eigenvalues(800);
  hs::SolverConfig cfg = nearest_cfg(0, 4);
  cfg.target.mode = hs::TargetMode::Smallest;
  auto s = hs::solve(a, nullptr, cfg);
  expect_report_ok(s, true);
  expect_same_set(s.eigenvalues, {w[0], w[1], w[2], w[3]}, 1e-9);
  cfg.target.mode = hs::TargetMode::Largest;
  auto l = hs::solve(a, nullptr, cfg);
  expect_report_ok(l, true);
  expect_same_set(l.eigenvalues, {w[799], w[798], w[797], w[796]}, 1e-9);
}

// Indefinite-B pencil with a known REAL spectrum and a zero cluster, like the waveguide pencils:
// A = Y^T diag(lambda_i j_i) Y, B = Y^T J Y with J = diag(+-1) (so B is indefinite) and Y
// nonsingular; then A (Y^{-1} e_i) = lambda_i B (Y^{-1} e_i). 20 of the lambda_i are exactly 0.
TEST(DriverSolve, IndefiniteBRealSpectrumWithZeroCluster) {
  const index_t n = 80, nzero = 20;
  hs::DenseMatrix<double> y(n, n);
  std::vector<double> lam(n), sgn(n);
  for (index_t i = 0; i < n; ++i) {
    for (index_t j = 0; j < n; ++j) y(i, j) = (i == j ? 1.0 : 0.0) + 0.3 * std::sin(1.7 * i + 0.9 * j * j + 0.1);
    lam[i] = i < nzero ? 0.0 : -5.0 + 10.0 * static_cast<double>(i - nzero) / static_cast<double>(n - nzero - 1);
    sgn[i] = (i % 3 == 0) ? -1.0 : 1.0;
  }
  hs::DenseMatrix<double> ad(n, n), bd(n, n);
  for (index_t r = 0; r < n; ++r)
    for (index_t c = 0; c < n; ++c) {
      double sa = 0.0, sb = 0.0;
      for (index_t i = 0; i < n; ++i) {
        sa += y(i, r) * lam[i] * sgn[i] * y(i, c);
        sb += y(i, r) * sgn[i] * y(i, c);
      }
      ad(r, c) = sa;
      bd(r, c) = sb;
    }
  for (index_t r = 0; r < n; ++r)  // exact symmetry despite rounding of the products
    for (index_t c = 0; c < r; ++c) {
      ad(r, c) = ad(c, r);
      bd(r, c) = bd(c, r);
    }
  auto a = hs::test::from_dense(ad), b = hs::test::from_dense(bd);
  // Target band next to the zero cluster - the realistic waveguide query. The 5 nearest sigma = 0.3
  // are 4 eigenvalues and then (distance 0.3) the 20-fold zero. The cluster (null space of A) is
  // never wanted (DESIGN §3.6-9): the driver sees the selection disk reach it, counts dim null(A)
  // = 20 and takes the 5th pair from one-sided runs on both sides of the cluster instead
  // (test_driver_cluster.cpp covers this in detail). Before P3 one copy of the zero was returned.
  const double sigma = 0.3;
  auto sol = hs::solve(a, &b, nearest_cfg(sigma, 5));
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(sol.complex_eigenvalues.empty());
  expect_report_ok(sol, false);
  std::vector<double> nonzero;
  for (double x : lam)
    if (x != 0.0) nonzero.push_back(x);
  expect_same_set(sol.eigenvalues, nearest(nonzero, sigma, 5), 1e-8);
  EXPECT_EQ(sol.report.zero_cluster_dim, nzero);
}

TEST(DriverSolve, PhaseTimesCoverTheWallTime) {
  // time_total_s is the sum of the phases by construction; what must hold is that it accounts
  // for the wall time of hs::solve (a phase clock started late, or work outside it, would not).
  // Nearest, interval (sliced), smallest (inertia search) and GHIEP paths. The margin covers the
  // return and the factorization's destructor, and scheduling noise when tests run in parallel.
  auto a = hs::test::laplacian_2d<double>(60, 59);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] == i) a.values[p] -= 4.0;
  std::vector<hs::SolverConfig> cfgs(3, nearest_cfg(0.3, 20));
  cfgs[1].target.mode = hs::TargetMode::Interval;
  cfgs[1].target.lower = -0.15;
  cfgs[1].target.upper = 0.15;
  cfgs[1].slice_size = 40;
  cfgs[2].target.mode = hs::TargetMode::Smallest;
  for (const auto& cfg : cfgs) {
    hs::Timer wall;
    const auto sol = hs::solve(a, nullptr, cfg);
    const double w = wall.seconds();
    expect_times_add_up(sol.report);
    EXPECT_LE(sol.report.time_total_s, w);
    EXPECT_GE(sol.report.time_total_s, 0.8 * w - 5e-3) << static_cast<int>(cfg.target.mode);
  }
  // GHIEP: B = diag(+-1) congruent to an indefinite pencil.
  const index_t n = 400;
  auto l = hs::test::laplacian_1d<double>(n);
  hs::DenseMatrix<double> bd(n, n);
  for (index_t i = 0; i < n; ++i) bd(i, i) = (i % 4 == 0) ? -1.0 : 1.0;
  const auto b = hs::test::from_dense(bd);
  hs::Timer wall;
  const auto sol = hs::solve(l, &b, nearest_cfg(0.5, 6));
  const double w = wall.seconds();
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  expect_times_add_up(sol.report);
  EXPECT_LE(sol.report.time_total_s, w);
  EXPECT_GE(sol.report.time_total_s, 0.8 * w - 5e-3);
}

TEST(DriverSolve, RejectsNonHermitianAndSymmetrizesNearHermitian) {
  auto a = hs::test::laplacian_1d<double>(50);
  a.values[1] += 1e-3;  // clearly not symmetric
  EXPECT_THROW((void)hs::solve(a, nullptr, nearest_cfg(1.0, 2)), hs::Error);
  auto c = hs::test::laplacian_1d<double>(50);
  c.values[1] += 1e-13;  // within hermitian_tol: symmetrized and solved
  auto sol = hs::solve(c, nullptr, nearest_cfg(1.0, 2));
  EXPECT_TRUE(sol.report.success);
  EXPECT_GT(sol.report.hermitian_defect_a, 0.0);
}

TEST(DriverSolve, MemoryBudgetIsEnforced) {
  auto a = hs::test::laplacian_2d<double>(60, 60);
  hs::SolverConfig cfg = nearest_cfg(1.0, 4);
  cfg.memory_budget_gb = 1e-6;  // nothing fits
  try {
    (void)hs::solve(a, nullptr, cfg);
    FAIL() << "expected OutOfMemoryBudget";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::OutOfMemoryBudget);
  }
}

}  // namespace
