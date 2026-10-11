// Tests of the P4 driver integration (src/driver): the O(nnz) definiteness screen of B, block
// Krylov runs in the definite path (block_size, auto rule, deflated reruns), certificate counts on
// an InertiaCounter, the deterministic run-time models and the exclusive phase times.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

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

void expect_times_add_up(const hs::SolveReport& r) {
  for (double t : {r.time_setup_s, r.time_factor_s, r.time_iterate_s, r.time_verify_s, r.time_certificate_s})
    EXPECT_GE(t, 0.0);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s + r.time_certificate_s,
              r.time_total_s, 1e-12 * r.time_total_s);
}

// ------------------------------------------------------------------------- definiteness of B

// random_hpd is strictly diagonally dominant with a positive diagonal: HPD by Gershgorin, so the
// driver needs no factorization of B (main + two certificate counts only).
TEST(DriverDefiniteness, DiagonallyDominantBNeedsNoFactorization) {
  auto a = hs::test::random_hermitian<double>(400, 0.02, 11);
  auto b = hs::test::random_hpd<double>(400, 0.01, 12);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = 0.5 * (w[150] + w[151]);
  auto cfg = nearest_cfg(sigma, 6);
  cfg.inertia_counter = hs::Toggle::Off;  // count the certificate's factorizations on the main object
  const auto sol = hs::solve(a, &b, cfg);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(sol.report.success);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 6), 1e-9);
  EXPECT_EQ(sol.report.factor_stats.factorizations, 3);  // K(sigma), K(a), K(b); none of B
}

// A diagonal entry <= 0 proves B indefinite (e_i^H B e_i <= 0): GHIEP without factorizing B.
TEST(DriverDefiniteness, NonPositiveDiagonalEntryMeansIndefinite) {
  const index_t n = 300;
  auto a = hs::test::laplacian_1d<double>(n);
  auto b = hs::test::random_hpd<double>(n, 0.01, 5);
  for (index_t p = b.rowptr[7]; p < b.rowptr[8]; ++p)
    if (b.colind[p] == 7) b.values[p] = -0.5;
  const auto sol = hs::solve(a, &b, nearest_cfg(0.37, 4));
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  // Same answer as when the inertia of B decides (problem given explicitly).
  auto cfg = nearest_cfg(0.37, 4);
  cfg.problem = hs::ProblemType::GeneralizedIndefinite;
  const auto ref = hs::solve(a, &b, cfg);
  expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-9);
}

// HPD but not strictly diagonally dominant (Dirichlet Laplacian: interior rows have diagonal 4 and
// four -1 neighbours): the inertia of B decides, one factorization of B.
TEST(DriverDefiniteness, NonDominantHpdBIsFactorized) {
  auto a = hs::test::random_hermitian<cplx>(200, 0.05, 21);
  const auto b = hs::test::laplacian_2d<cplx>(20, 10);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = 0.5 * (w[60] + w[61]);
  auto cfg = nearest_cfg(sigma, 5);
  cfg.inertia_counter = hs::Toggle::Off;
  const auto sol = hs::solve(a, &b, cfg);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(sol.report.success);
  expect_same_set(sol.eigenvalues, nearest(w, sigma, 5), 1e-9);
  expect_times_add_up(sol.report);
  EXPECT_EQ(sol.report.factor_stats.factorizations, 4);  // B, K(sigma), K(a), K(b)
}

}  // namespace
