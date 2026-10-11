// End-to-end tests of hs::solve (driver + factor + ops + eigen). Every case has an analytic or a
// dense reference (n <= 2000), and checks the report contract (residuals, certificate, warnings).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
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

template <class T>
void expect_report_ok(const hs::Solution<T>& sol, bool expect_certificate) {
  EXPECT_TRUE(sol.report.success);
  for (const auto& p : sol.report.pairs) {
    EXPECT_TRUE(p.accepted);
    EXPECT_LE(p.backward_error, 1e-10);
  }
  EXPECT_EQ(sol.report.certificate.performed, expect_certificate);
  if (expect_certificate) EXPECT_TRUE(sol.report.certificate.passed);
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

// Waveguide-like pencil with a null block: A = blkdiag(Att, 0), B = [[T, T D], [D^T T, D^T T D - k M]].
// Exactly #nodes zero eigenvalues, B indefinite (Schur complement -k M), all eigenvalues real.
TEST(DriverSolve, IndefiniteBWaveguideLike) {
  const index_t ne = 60, nn = 20, n = ne + nn;
  auto t_tt = hs::test::to_dense(hs::test::random_hpd<double>(ne, 0.05, 5));
  auto att = hs::test::to_dense(hs::test::random_hermitian<double>(ne, 0.1, 6));
  auto mz = hs::test::to_dense(hs::test::random_hpd<double>(nn, 0.1, 7));
  hs::DenseMatrix<double> d(ne, nn);
  for (index_t e = 0; e < ne; ++e) {  // incidence-like: +1 / -1 at two nodes
    d(e, e % nn) = 1.0;
    d(e, (e * 7 + 3) % nn) -= 1.0;
  }
  hs::DenseMatrix<double> ad(n, n), bd(n, n);
  for (index_t i = 0; i < ne; ++i)
    for (index_t j = 0; j < ne; ++j) {
      ad(i, j) = att(i, j);
      bd(i, j) = t_tt(i, j);
    }
  // T D, D^T T D
  hs::DenseMatrix<double> td(ne, nn);
  for (index_t i = 0; i < ne; ++i)
    for (index_t j = 0; j < nn; ++j)
      for (index_t q = 0; q < ne; ++q) td(i, j) += t_tt(i, q) * d(q, j);
  for (index_t i = 0; i < ne; ++i)
    for (index_t j = 0; j < nn; ++j) bd(i, ne + j) = bd(ne + j, i) = td(i, j);
  const double kk = 3.0;
  for (index_t i = 0; i < nn; ++i)
    for (index_t j = 0; j < nn; ++j) {
      double s = 0;
      for (index_t q = 0; q < ne; ++q) s += d(q, i) * td(q, j);
      bd(ne + i, ne + j) = s - kk * mz(i, j);
    }
  auto a = hs::test::from_dense(ad), b = hs::test::from_dense(bd);
  // Dense reference: eigenvalues of B^{-1} A (B nonsingular).
  hs::DenseMatrix<double> lu = bd, m = ad;
  std::vector<int> ipiv(n);
  ASSERT_EQ(hs::lapack::getrf(n, lu.data(), n, ipiv.data()), 0);
  hs::lapack::getrs(n, n, lu.data(), n, ipiv.data(), m.data(), n);
  hs::DenseMatrix<double> q(n, n);
  std::vector<cplx> mu(n);
  hs::lapack::gees(n, m.data(), n, mu.data(), q.data(), n);
  std::vector<double> w;
  for (auto x : mu) {
    ASSERT_LT(std::abs(x.imag()), 1e-8 * std::max(1.0, std::abs(x)));
    w.push_back(x.real());
  }
  // Target between the zero cluster and the nonzero spectrum.
  std::sort(w.begin(), w.end());
  double sigma = 0.0;
  for (double x : w)
    if (x > 1e-6) {
      sigma = 0.9 * x;
      break;
    }
  auto sol = hs::solve(a, &b, nearest_cfg(sigma, 5));
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(sol.complex_eigenvalues.empty());
  expect_report_ok(sol, false);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 5), 1e-8);
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
