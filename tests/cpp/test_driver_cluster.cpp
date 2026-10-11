// GHIEP zero-cluster handling of hs::solve (DESIGN §3.6-9): pencils with indefinite B whose A has
// a large null space, like the Lee-Sun-Cendes waveguide pencils (A = blkdiag(A_tt, 0)). The
// cluster lambda = 0 is never wanted: one-sided shifts keep it at the unwanted end.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::index_t;

// A = Y^H diag(lambda_i s_i) Y, B = Y^H diag(s_i) Y with s_i = +-1 (B indefinite) and Y
// nonsingular (complex for T = cplx): A (Y^{-1} e_i) = lambda_i B (Y^{-1} e_i). Every
// lambda_i == 0 contributes a null vector of A, so the zero cluster has dimension #zeros, all real.
template <class T>
struct Pencil {
  hs::CsrMatrix<T> a, b;
};

// complex_pairs: x +- iy appended as 2x2 blocks A = [[x, y], [y, -x]], B = diag(1, -1)
// (det(A - l B) = -((l - x)^2 + y^2)).
template <class T = double>
Pencil<T> make_pencil(const std::vector<double>& lam, double y_coupling,
                      const std::vector<hs::cplx>& complex_pairs = {}) {
  const auto nr = static_cast<index_t>(lam.size());
  const index_t n = nr + 2 * static_cast<index_t>(complex_pairs.size());
  hs::DenseMatrix<T> y(n, n), ad(n, n), bd(n, n), da(n, n), db(n, n);
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j) {
      const double re = std::sin(1.7 * i + 0.9 * j * j + 0.1), im = std::cos(0.7 * i * j + 0.3);
      if constexpr (hs::is_complex_v<T>) {
        y(i, j) = T(i == j ? 1.0 : 0.0) + y_coupling * T(re, im) / std::sqrt(2.0 * n);
      } else {
        y(i, j) = (i == j ? 1.0 : 0.0) + y_coupling * re / std::sqrt(double(n));
      }
    }
  for (index_t i = 0; i < nr; ++i) {
    const double sgn = (i % 3 == 0) ? -1.0 : 1.0;
    da(i, i) = T(lam[i] * sgn);
    db(i, i) = T(sgn);
  }
  for (std::size_t p = 0; p < complex_pairs.size(); ++p) {
    const index_t i = nr + 2 * static_cast<index_t>(p);
    da(i, i) = T(complex_pairs[p].real());
    da(i, i + 1) = da(i + 1, i) = T(complex_pairs[p].imag());
    da(i + 1, i + 1) = T(-complex_pairs[p].real());
    db(i, i) = T(1.0);
    db(i + 1, i + 1) = T(-1.0);
  }
  // A = Y^H D_A Y, B = Y^H D_B Y (D_A, D_B have at most 2 entries per row).
  for (index_t r = 0; r < n; ++r)
    for (index_t c = r; c < n; ++c) {
      T sa = T(0), sb = T(0);
      for (index_t i = 0; i < n; ++i)
        for (index_t q = std::max<index_t>(0, i - 1); q <= std::min<index_t>(n - 1, i + 1); ++q) {
          sa += hs::conj_if(y(i, r)) * da(i, q) * y(q, c);
          sb += hs::conj_if(y(i, r)) * db(i, q) * y(q, c);
        }
      ad(r, c) = sa;
      ad(c, r) = hs::conj_if(sa);
      bd(r, c) = sb;
      bd(c, r) = hs::conj_if(sb);
    }
  for (index_t r = 0; r < n; ++r) {  // exactly real diagonals
    ad(r, r) = T(std::real(ad(r, r)));
    bd(r, r) = T(std::real(bd(r, r)));
  }
  return {hs::test::from_dense(ad), hs::test::from_dense(bd)};
}

std::vector<double> nearest_nonzero(std::vector<double> w, double sigma, index_t k) {
  w.erase(std::remove(w.begin(), w.end(), 0.0), w.end());
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
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  return cfg;
}

template <class T>
void expect_ok(const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  EXPECT_EQ(r.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(r.success);
  for (const auto& p : r.pairs) EXPECT_LE(p.backward_error, 1e-10);
  EXPECT_TRUE(sol.complex_eigenvalues.empty());
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s, r.time_total_s,
              1e-12 * r.time_total_s);
  for (double x : sol.eigenvalues) EXPECT_GT(std::abs(x), 1e-6);  // never a cluster member
}

// 30 zeros, 20 negative (propagating-like), 50 positive eigenvalues.
std::vector<double> waveguide_like() {
  std::vector<double> lam(30, 0.0);
  for (int i = 0; i < 20; ++i) lam.push_back(-0.15 - 0.05 * i);
  for (int i = 0; i < 50; ++i) lam.push_back(0.2 + 0.07 * i);
  return lam;
}

TEST(DriverZeroCluster, SigmaAtZeroReturnsNearestNonzeroModes) {
  const auto lam = waveguide_like();
  const auto p = make_pencil(lam, 0.3);
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(0.0, 8));
  expect_ok(sol);
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, 0.0, 8), 1e-8);
  EXPECT_EQ(sol.report.zero_cluster_dim, 30);
  EXPECT_EQ(sol.report.slices.size(), 2u);  // one run on each side of the cluster
}

TEST(DriverZeroCluster, ComplexHermitianPencil) {
  const auto lam = waveguide_like();
  const auto p = make_pencil<hs::cplx>(lam, 0.3);
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(0.0, 8));
  expect_ok(sol);
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, 0.0, 8), 1e-8);
  EXPECT_EQ(sol.report.zero_cluster_dim, 30);
}

TEST(DriverZeroCluster, WindowReachingTheClusterSwitchesToOneSidedRuns) {
  // sigma = -0.5 with k = 24: the 20 negatives are nearest, the cluster at distance 0.5 comes next
  // and must not take the remaining slots - the 4 smallest positives do.
  const auto lam = waveguide_like();
  const auto p = make_pencil(lam, 0.3);
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(-0.5, 24));
  expect_ok(sol);
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, -0.5, 24), 1e-8);
  EXPECT_EQ(sol.report.zero_cluster_dim, 30);
}

TEST(DriverZeroCluster, ClusterFarFromTheWindowCostsNothing) {
  // The k nearest stay clear of 0: one plain run, A is never factorized for the cluster.
  const auto lam = waveguide_like();
  const auto p = make_pencil(lam, 0.3);
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(1.5, 5));
  expect_ok(sol);
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, 1.5, 5), 1e-8);
  EXPECT_EQ(sol.report.zero_cluster_dim, -1);
  EXPECT_TRUE(sol.report.slices.empty());
}

TEST(DriverZeroCluster, ZeroFilterTolBandAndExhaustedSide) {
  // Only positive eigenvalues besides the cluster: the negative side holds nothing, its one-sided
  // run must stop early (exhausted) instead of chasing nonexistent eigenvalues. A small positive
  // eigenvalue inside the user's band is filtered as well.
  std::vector<double> lam(25, 0.0);
  lam.push_back(1e-7);  // inside the band zero_filter_tol * scale
  for (int i = 0; i < 60; ++i) lam.push_back(0.3 + 0.05 * i);
  const auto p = make_pencil(lam, 0.3);
  auto cfg = nearest_cfg(0.01, 6);
  cfg.zero_filter_tol = 1e-6;
  auto sol = hs::solve(p.a, &p.b, cfg);
  expect_ok(sol);
  expect_same_set(sol.eigenvalues, {0.3, 0.35, 0.4, 0.45, 0.5, 0.55}, 1e-8);
  ASSERT_EQ(sol.report.slices.size(), 2u);
  bool exhausted_side = false;
  for (const auto& s : sol.report.slices) exhausted_side = exhausted_side || (s.passed && s.count_found < s.nev);
  EXPECT_TRUE(exhausted_side);
}

TEST(DriverZeroCluster, ComplexPairBetweenTheSideShifts) {
  // A complex pair 0.02 +- 0.1i whose real part lies between the two side shifts is unwanted by
  // both one-sided runs (Re theta has the wrong sign for each); the side runs' Ritz values reveal
  // it and a recovery run adds it. Nearest 4 of sigma = 1e-3: the pair, -0.15 and 0.2.
  const auto p = make_pencil(waveguide_like(), 0.3, {hs::cplx(0.02, 0.1)});
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(1e-3, 4));
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  EXPECT_EQ(r.zero_cluster_dim, 30);
  expect_same_set(sol.eigenvalues, {-0.15, 0.2}, 1e-8);
  ASSERT_EQ(sol.complex_eigenvalues.size(), 2u);
  for (const auto& z : sol.complex_eigenvalues) {
    EXPECT_NEAR(z.real(), 0.02, 1e-8);
    EXPECT_NEAR(std::abs(z.imag()), 0.1, 1e-8);
  }
  for (const auto& pr : r.pairs) EXPECT_LE(pr.backward_error, 1e-10);
}

TEST(DriverZeroCluster, SideShiftsBeyondAPlainRunsEdge) {
  // sigma = 0.01: the plain run converges cluster copies and real pairs out to a disk whose edge
  // is a double eigenvalue. The side shifts go mid-way between that edge pair and the next
  // eigenvalue (a shift just inside the edge pair, 1e-6 r from it, left eta ~1e-12 near tol).
  std::vector<double> lam = waveguide_like();
  for (int i = 0; i < 6; ++i) lam.push_back(0.2 + 0.07 * i);  // doubles
  const auto p = make_pencil(lam, 0.3);
  for (index_t k : {12, 24}) {
    auto cfg = nearest_cfg(0.01, k);
    cfg.zero_filter_tol = 1e-9;
    auto sol = hs::solve(p.a, &p.b, cfg);
    expect_ok(sol);
    expect_same_set(sol.eigenvalues, nearest_nonzero(lam, 0.01, k), 1e-8);
    for (const auto& pr : sol.report.pairs) EXPECT_LE(pr.backward_error, 1e-13) << k;
    ASSERT_FALSE(sol.report.slices.empty());
    for (const auto& sl : sol.report.slices)
      for (double x : lam)
        if (x != 0.0) EXPECT_GT(std::abs(sl.sigma - x), 1e-3 * std::abs(x)) << "shift " << sl.sigma << " on " << x;
  }
}

}  // namespace
