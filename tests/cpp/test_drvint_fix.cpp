// Regression tests of the drvint review fixes (src/driver): the deterministic nested-dissection
// ordering.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// 3-D 7-point Dirichlet Laplacian on an nx x ny x nz grid (3-D fill: nested dissection pays).
template <class T>
hs::CsrMatrix<T> laplacian_3d(index_t nx, index_t ny, index_t nz) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [&](index_t i, index_t j, index_t k) { return i + nx * (j + ny * k); };
  for (index_t k = 0; k < nz; ++k)
    for (index_t j = 0; j < ny; ++j)
      for (index_t i = 0; i < nx; ++i) {
        const index_t p = id(i, j, k);
        r.push_back(p), c.push_back(p), v.push_back(T(6));
        if (i > 0) r.push_back(p), c.push_back(id(i - 1, j, k)), v.push_back(T(-1));
        if (i + 1 < nx) r.push_back(p), c.push_back(id(i + 1, j, k)), v.push_back(T(-1));
        if (j > 0) r.push_back(p), c.push_back(id(i, j - 1, k)), v.push_back(T(-1));
        if (j + 1 < ny) r.push_back(p), c.push_back(id(i, j + 1, k)), v.push_back(T(-1));
        if (k > 0) r.push_back(p), c.push_back(id(i, j, k - 1)), v.push_back(T(-1));
        if (k + 1 < nz) r.push_back(p), c.push_back(id(i, j, k + 1)), v.push_back(T(-1));
      }
  const index_t n = nx * ny * nz;
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

std::vector<double> laplacian_3d_eigenvalues(index_t nx, index_t ny, index_t nz) {
  const auto wx = hs::test::laplacian_1d_eigenvalues(nx), wy = hs::test::laplacian_1d_eigenvalues(ny),
             wz = hs::test::laplacian_1d_eigenvalues(nz);
  std::vector<double> w;
  for (double a : wx)
    for (double b : wy)
      for (double c : wz) w.push_back(a + b + c);
  std::sort(w.begin(), w.end());
  return w;
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

std::vector<double> nearest(std::vector<double> w, double sigma, index_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
  w.resize(static_cast<std::size_t>(k));
  return w;
}

void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

// ------------------------------------------------------------------------------ ordering

// deterministic = true promises bitwise-reproducible runs. The adaptive ordering's nested
// dissection must then be METIS: SCOTCH's orderings differ from analysis to analysis (in one
// process as across processes), and with them the factors and every result.
// sigma: the 8 nearest eigenvalues do not split a multiple one.
template <class T>
void deterministic_runs_are_bitwise_equal(index_t m, double sigma) {
  const auto a = laplacian_3d<T>(m, m, m - 1);
  const auto w = laplacian_3d_eigenvalues(m, m, m - 1);
  auto cfg = nearest_cfg(sigma, 8);
  cfg.deterministic = true;
  ASSERT_EQ(cfg.ordering, -1);  // adaptive
  const auto s1 = hs::solve(a, nullptr, cfg);
  const auto s2 = hs::solve(a, nullptr, cfg);
  ASSERT_EQ(s1.report.factor_options.ordering, 5) << "the AMD probe should pick nested dissection (METIS)";
  EXPECT_TRUE(s1.report.success);
  expect_same_set(s1.eigenvalues, nearest(w, cfg.target.sigma, 8), 1e-9);
  EXPECT_EQ(s1.report.factor_stats.factor_entries, s2.report.factor_stats.factor_entries);
  ASSERT_EQ(s1.eigenvalues.size(), s2.eigenvalues.size());
  EXPECT_EQ(std::memcmp(s1.eigenvalues.data(), s2.eigenvalues.data(), s1.eigenvalues.size() * sizeof(double)), 0);
  ASSERT_EQ(s1.eigenvectors.cols(), s2.eigenvectors.cols());
  EXPECT_EQ(std::memcmp(s1.eigenvectors.data(), s2.eigenvectors.data(),
                        static_cast<std::size_t>(s1.eigenvectors.rows() * s1.eigenvectors.cols()) * sizeof(T)),
            0);
}

TEST(DriverOrdering, DeterministicRunsAreBitwiseEqualReal) { deterministic_runs_are_bitwise_equal<double>(18, 1.003); }
TEST(DriverOrdering, DeterministicRunsAreBitwiseEqualComplex) { deterministic_runs_are_bitwise_equal<cplx>(18, 1.003); }

TEST(DriverOrdering, NestedDissectionIsScotchUnlessDeterministic) {
  hs::SolverConfig cfg;
  EXPECT_EQ(hs::driver::nd_ordering(cfg), 3);
  cfg.deterministic = true;
  EXPECT_EQ(hs::driver::nd_ordering(cfg), 5);
}

}  // namespace
