// Interval / smallest / largest targets of hs::solve (B HPD): spectrum slicing, inertia search for
// the window ends, zero-band exclusion and the output memory budget. Every case has an analytic or
// a dense reference (n <= 2000 for dense ones).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

hs::SolverConfig base_cfg() {
  hs::SolverConfig cfg;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  return cfg;
}

hs::SolverConfig interval_cfg(double lo, double hi, index_t slice_size = 0) {
  hs::SolverConfig cfg = base_cfg();
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = lo;
  cfg.target.upper = hi;
  cfg.slice_size = slice_size;
  return cfg;
}

hs::SolverConfig end_cfg(hs::TargetMode mode, index_t k, index_t slice_size = 0) {
  hs::SolverConfig cfg = base_cfg();
  cfg.target.mode = mode;
  cfg.target.k = k;
  cfg.slice_size = slice_size;
  return cfg;
}

std::vector<double> in_interval(const std::vector<double>& w, double lo, double hi) {
  std::vector<double> out;
  for (double x : w)
    if (x >= lo && x <= hi) out.push_back(x);
  return out;
}

void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

void expect_times_add_up(const hs::SolveReport& r) {
  EXPECT_GE(r.time_setup_s, 0.0);
  EXPECT_GE(r.time_factor_s, 0.0);
  EXPECT_GE(r.time_iterate_s, 0.0);
  EXPECT_GE(r.time_verify_s, 0.0);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s, r.time_total_s,
              1e-12 * r.time_total_s);
}

// Report contract of a successful windowed solve: accepted pairs, certificate over all slices.
template <class T>
void expect_window_ok(const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  expect_times_add_up(r);
  EXPECT_TRUE(r.success);
  for (const auto& p : r.pairs) {
    EXPECT_TRUE(p.accepted);
    EXPECT_LE(p.backward_error, 1e-10);
  }
  EXPECT_TRUE(r.certificate.performed);
  EXPECT_TRUE(r.certificate.passed) << r.certificate.count_inertia << " vs " << r.certificate.count_found;
  for (const auto& s : r.slices) EXPECT_TRUE(s.passed) << "[" << s.lower << ", " << s.upper << "]";
  EXPECT_EQ(sol.eigenvectors.cols(), static_cast<index_t>(sol.eigenvalues.size()));
}

// Dense X^H B X == I (B == nullptr: identity), n small.
template <class T>
void expect_b_orthonormal(const hs::Solution<T>& sol, const hs::CsrMatrix<T>* b, double tol) {
  const auto& x = sol.eigenvectors;
  const index_t n = x.rows(), m = x.cols();
  std::vector<T> bx(static_cast<std::size_t>(n));
  double worst = 0.0;
  for (index_t j = 0; j < m; ++j) {
    if (b) {
      hs::spmv(*b, x.col(j), bx.data());
    } else {
      std::copy(x.col(j), x.col(j) + n, bx.begin());
    }
    for (index_t i = 0; i < m; ++i) {
      T s = T(0);
      for (index_t r = 0; r < n; ++r) s += hs::conj_if(x(r, i)) * bx[r];
      worst = std::max(worst, std::abs(s - T(i == j ? 1.0 : 0.0)));
    }
  }
  EXPECT_LE(worst, tol);
}

template <class T>
hs::CsrMatrix<T> shifted(hs::CsrMatrix<T> a, double shift) {
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] == i) a.values[p] -= shift;
  return a;
}

// ---------------------------------------------------------------------------------- slicing

TEST(DriverSlicing, IntervalSplitIntoSlicesMatchesAnalytic) {
  // Indefinite 2-D Laplacian; 30 x 29 avoids multiplicities. ~200 eigenvalues in the interval,
  // at most 25 per shift.
  auto a = shifted(hs::test::laplacian_2d<double>(30, 29), 4.0);
  auto w = hs::test::laplacian_2d_eigenvalues(30, 29);
  for (auto& x : w) x -= 4.0;
  const double lo = -1.3, hi = 1.1;
  const auto want = in_interval(w, lo, hi);
  ASSERT_GT(want.size(), 180u);
  auto sol = hs::solve(a, nullptr, interval_cfg(lo, hi, 25));
  expect_window_ok(sol);
  expect_same_set(sol.eigenvalues, want, 1e-10);
  EXPECT_TRUE(std::is_sorted(sol.eigenvalues.begin(), sol.eigenvalues.end()));
  EXPECT_GE(sol.report.slices.size(), want.size() / 25);
  index_t counted = 0;
  for (const auto& s : sol.report.slices) {
    EXPECT_LE(s.count_inertia, 25);
    counted += s.count_inertia;
  }
  EXPECT_EQ(counted, static_cast<index_t>(want.size()));
  EXPECT_EQ(sol.report.certificate.count_inertia, static_cast<index_t>(want.size()));
  expect_b_orthonormal(sol, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
}

TEST(DriverSlicing, MultipleEigenvaluesAcrossSmallSlices) {
  // Square grid: most eigenvalues are double. Slices of at most 9 pairs put many cuts next to
  // multiple eigenvalues; every copy must be found exactly once.
  auto a = hs::test::laplacian_2d<double>(30, 30);
  auto w = hs::test::laplacian_2d_eigenvalues(30, 30);
  const double lo = 2.0, hi = 2.6;
  const auto want = in_interval(w, lo, hi);
  ASSERT_GT(want.size(), 60u);
  auto sol = hs::solve(a, nullptr, interval_cfg(lo, hi, 9));
  expect_window_ok(sol);
  expect_same_set(sol.eigenvalues, want, 1e-10);
  EXPECT_GT(sol.report.slices.size(), 6u);
  expect_b_orthonormal(sol, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
}

TEST(DriverSlicing, AutomaticSliceSizeOnLargeInterval) {
  // 1-D Laplacian: 500 eigenvalues in the interval; the cost model chooses the slices.
  const index_t n = 1500;
  auto a = hs::test::laplacian_1d<double>(n);
  auto w = hs::test::laplacian_1d_eigenvalues(n);
  const double lo = w[300] - 1e-7, hi = w[799] + 1e-7;
  const auto want = in_interval(w, lo, hi);
  ASSERT_EQ(want.size(), 500u);
  auto sol = hs::solve(a, nullptr, interval_cfg(lo, hi));
  expect_window_ok(sol);
  expect_same_set(sol.eigenvalues, want, 1e-10);
  EXPECT_GT(sol.report.slices.size(), 2u);
}

TEST(DriverSlicing, GeneralizedRealAndComplexIntervals) {
  {
    auto a = hs::test::random_hermitian<double>(400, 0.02, 11);
    auto b = hs::test::random_hpd<double>(400, 0.01, 12);
    auto w = hs::test::dense_eigenvalues(a, &b);
    const double lo = 0.5 * (w[100] + w[101]), hi = 0.5 * (w[219] + w[220]);
    auto sol = hs::solve(a, &b, interval_cfg(lo, hi, 25));
    expect_window_ok(sol);
    EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
    expect_same_set(sol.eigenvalues, in_interval(w, lo, hi), 1e-9);
    expect_b_orthonormal(sol, &b, 1e-9);
  }
  {
    auto a = hs::test::random_hermitian<cplx>(300, 0.03, 13);
    auto b = hs::test::random_hpd<cplx>(300, 0.02, 14);
    auto w = hs::test::dense_eigenvalues(a, &b);
    const double lo = 0.5 * (w[50] + w[51]), hi = 0.5 * (w[149] + w[150]);
    auto sol = hs::solve(a, &b, interval_cfg(lo, hi, 30));
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, in_interval(w, lo, hi), 1e-9);
    expect_b_orthonormal(sol, &b, 1e-9);
  }
}

TEST(DriverSlicing, EigenvaluesOnTheIntervalEndsAreIncluded) {
  // The interval is closed: eigenvalues exactly at lower and upper belong to it. Cuts land
  // inside a 3-fold eigenvalue and next to close pairs.
  std::vector<double> lam;
  for (int i = 0; i < 300; ++i) lam.push_back(-3.0 + 0.02 * i);  // -3 .. 2.98
  lam.push_back(1.0);
  lam.push_back(1.0);
  lam.push_back(-1.0 + 1e-7);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 21);
  const double lo = -1.0, hi = 1.0;  // both are eigenvalues (i = 100 and i = 200, plus copies)
  auto sol = hs::solve(a, nullptr, interval_cfg(lo, hi, 13));
  expect_window_ok(sol);
  std::vector<double> want;
  for (double x : lam)
    if (x >= lo - 1e-12 && x <= hi + 1e-12) want.push_back(x);
  expect_same_set(sol.eigenvalues, want, 1e-10);
}

TEST(DriverSlicing, ZeroBandIsCutOutOfTheInterval) {
  // 40 exact zero eigenvalues (a null space) inside the interval: with zero_filter_tol the band
  // |lambda| <= tol*scale is excluded by inertia counts at its edges, never computed.
  std::vector<double> lam(40, 0.0);
  for (int i = 1; i <= 150; ++i) {
    lam.push_back(0.01 * i);
    lam.push_back(-0.013 * i);
  }
  auto a = hs::test::prescribed_spectrum<double>(lam, 5, 22);
  auto cfg = interval_cfg(-0.5, 0.5, 30);
  cfg.zero_filter_tol = 1e-6;
  auto sol = hs::solve(a, nullptr, cfg);
  expect_window_ok(sol);
  std::vector<double> want;
  for (double x : lam)
    if (x != 0.0 && x >= -0.5 && x <= 0.5) want.push_back(x);
  expect_same_set(sol.eigenvalues, want, 1e-10);
  EXPECT_EQ(sol.report.zero_modes_filtered, 40);
  for (double x : sol.eigenvalues) EXPECT_GT(std::abs(x), 1e-3);
}

TEST(DriverSlicing, EmptyIntervalSucceedsWithNoPairs) {
  auto a = hs::test::laplacian_1d<double>(200);
  auto sol = hs::solve(a, nullptr, interval_cfg(10.0, 11.0));
  EXPECT_TRUE(sol.report.success);
  EXPECT_TRUE(sol.eigenvalues.empty());
  EXPECT_EQ(sol.eigenvectors.cols(), 0);
  EXPECT_EQ(sol.report.certificate.count_inertia, 0);
}

TEST(DriverSlicing, OutputLargerThanTheBudgetIsRefused) {
  // 160 x 160 grid, the whole spectrum: 25600 eigenvectors need 4.9 GiB, the budget is 1 GiB.
  // The refusal comes after the two inertia counts, before any Krylov work.
  auto a = hs::test::laplacian_2d<double>(160, 160);
  auto cfg = interval_cfg(-1.0, 9.0);
  cfg.memory_budget_gb = 1.0;
  try {
    (void)hs::solve(a, nullptr, cfg);
    FAIL() << "expected OutOfMemoryBudget";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::OutOfMemoryBudget);
  }
}

// ---------------------------------------------------------------------- smallest / largest

TEST(DriverTargets, SmallestAndLargestGeneralizedMatchDense) {
  auto a = hs::test::random_hermitian<double>(500, 0.02, 31);
  auto b = hs::test::random_hpd<double>(500, 0.01, 32);
  auto w = hs::test::dense_eigenvalues(a, &b);
  auto s = hs::solve(a, &b, end_cfg(hs::TargetMode::Smallest, 7));
  expect_window_ok(s);
  EXPECT_EQ(s.report.problem, hs::ProblemType::GeneralizedDefinite);
  expect_same_set(s.eigenvalues, {w.begin(), w.begin() + 7}, 1e-9);
  EXPECT_TRUE(std::is_sorted(s.eigenvalues.begin(), s.eigenvalues.end()));
  EXPECT_TRUE(std::isinf(s.report.certificate.lower));
  auto l = hs::solve(a, &b, end_cfg(hs::TargetMode::Largest, 7));
  expect_window_ok(l);
  expect_same_set(l.eigenvalues, {w.end() - 7, w.end()}, 1e-9);
  EXPECT_TRUE(std::is_sorted(l.eigenvalues.rbegin(), l.eigenvalues.rend()));  // descending
  EXPECT_TRUE(std::isinf(l.report.certificate.upper));
  expect_b_orthonormal(l, &b, 1e-9);
}

TEST(DriverTargets, SmallestComplexHermitianGeneralized) {
  auto a = hs::test::random_hermitian<cplx>(300, 0.03, 33);
  auto b = hs::test::random_hpd<cplx>(300, 0.02, 34);
  auto w = hs::test::dense_eigenvalues(a, &b);
  auto s = hs::solve(a, &b, end_cfg(hs::TargetMode::Smallest, 5));
  expect_window_ok(s);
  expect_same_set(s.eigenvalues, {w.begin(), w.begin() + 5}, 1e-9);
}

TEST(DriverTargets, SmallestOfAClusteredLowEndStandard) {
  // 2-D Laplacian: the 10 smallest span 1% of the spectrum; the window end comes from the
  // inertia search and the run is one-sided at it.
  auto a = hs::test::laplacian_2d<double>(70, 71);
  auto w = hs::test::laplacian_2d_eigenvalues(70, 71);
  auto s = hs::solve(a, nullptr, end_cfg(hs::TargetMode::Smallest, 10));
  expect_window_ok(s);
  expect_same_set(s.eigenvalues, {w.begin(), w.begin() + 10}, 1e-10);
  ASSERT_FALSE(s.report.slices.empty());
  EXPECT_EQ(s.report.slices.front().which, "smallest_algebraic");
}

TEST(DriverTargets, LargestOverManySlices) {
  const index_t n = 2000;
  auto a = hs::test::laplacian_1d<double>(n);
  auto w = hs::test::laplacian_1d_eigenvalues(n);
  auto l = hs::solve(a, nullptr, end_cfg(hs::TargetMode::Largest, 150, 40));
  expect_window_ok(l);
  expect_same_set(l.eigenvalues, {w.end() - 150, w.end()}, 1e-10);
  EXPECT_GE(l.report.slices.size(), 4u);
  EXPECT_EQ(l.report.slices.back().which, "largest_algebraic");
}

TEST(DriverTargets, SmallestSkipsTheZeroBand) {
  // Positive semidefinite with a 60-dimensional null space (like a curl-curl stiffness): the 5
  // smallest outside the zero band are the 5 smallest positive eigenvalues.
  std::vector<double> lam(60, 0.0);
  for (int i = 1; i <= 400; ++i) lam.push_back(0.5 + 0.01 * i * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 35);
  auto cfg = end_cfg(hs::TargetMode::Smallest, 5);
  cfg.zero_filter_tol = 1e-8;
  auto s = hs::solve(a, nullptr, cfg);
  expect_window_ok(s);
  expect_same_set(s.eigenvalues, {0.51, 0.54, 0.59, 0.66, 0.75}, 1e-10);
  EXPECT_EQ(s.report.zero_modes_filtered, 60);
}

TEST(DriverTargets, SmallestAcrossTheZeroBandFromBothSides) {
  // 3 negative eigenvalues, 50 zeros, positives: the 6 smallest outside the band are the 3
  // negatives and the 3 smallest positives. Largest with the band: mirror image.
  std::vector<double> lam(50, 0.0);
  lam.insert(lam.end(), {-2.0, -1.5, -0.25});
  for (int i = 1; i <= 300; ++i) lam.push_back(0.1 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 36);
  auto cfg = end_cfg(hs::TargetMode::Smallest, 6);
  cfg.zero_filter_tol = 1e-8;
  auto s = hs::solve(a, nullptr, cfg);
  expect_window_ok(s);
  expect_same_set(s.eigenvalues, {-2.0, -1.5, -0.25, 0.1, 0.2, 0.3}, 1e-10);
  EXPECT_EQ(s.report.zero_modes_filtered, 50);
  std::vector<double> neg(lam);
  for (auto& x : neg) x = -x;
  auto an = hs::test::prescribed_spectrum<double>(neg, 4, 37);
  cfg.target.mode = hs::TargetMode::Largest;
  auto l = hs::solve(an, nullptr, cfg);
  expect_window_ok(l);
  expect_same_set(l.eigenvalues, {2.0, 1.5, 0.25, -0.1, -0.2, -0.3}, 1e-10);
}

TEST(DriverTargets, NearestNextToAFilteredNullSpace) {
  // Positive semidefinite with an 80-dimensional null space; sigma just above (or inside) the zero
  // band. A single shift there converges copies of the null space; the driver must switch to
  // windows sigma +- r minus the band, certified by inertia, and return the 5 nearest nonzero.
  std::vector<double> lam(80, 0.0);
  for (int i = 1; i <= 300; ++i) lam.push_back(0.3 + 0.01 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 39);
  for (double sigma : {0.01, 0.0}) {
    auto cfg = base_cfg();
    cfg.target.sigma = sigma;
    cfg.target.k = 5;
    cfg.zero_filter_tol = 1e-6;
    auto sol = hs::solve(a, nullptr, cfg);
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, {0.31, 0.32, 0.33, 0.34, 0.35}, 1e-10);
    EXPECT_EQ(sol.report.zero_modes_filtered, 80) << sigma;
    EXPECT_TRUE(std::is_sorted(sol.eigenvalues.begin(), sol.eigenvalues.end()));  // nearest first
  }
}

// ------------------------------------------------- clusters, multiplets, inexact factors

// Largest backward error of the returned pairs.
double worst_eta(const hs::SolveReport& r) {
  double w = 0.0;
  for (const auto& p : r.pairs) w = std::max(w, p.backward_error);
  return w;
}

TEST(DriverTargets, LargestWhoseTopEigenvalueIsMultiple) {
  // The top eigenvalue is 10-fold and the 4 largest are 4 of its copies: the inertia search for
  // the window end must fall back BELOW the multiplet (a window above it holds nothing).
  std::vector<double> lam(10, 3.0);
  for (int i = 1; i <= 300; ++i) lam.push_back(-0.01 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 41);
  auto l = hs::solve(a, nullptr, end_cfg(hs::TargetMode::Largest, 4));
  expect_window_ok(l);
  expect_same_set(l.eigenvalues, {3.0, 3.0, 3.0, 3.0}, 1e-10);
  expect_b_orthonormal(l, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
}

TEST(DriverTargets, KthEigenvalueInsideAMultiplet) {
  // 3 + 2 low eigenvalues, then a 6-fold 2.5: smallest k = 6, 7 end inside the multiplet, which
  // counts cannot split. The window takes the whole multiplet, its end lies mid-way in the gap
  // beyond it (a one-sided shift hugging the multiplet left far pairs at eta ~1e-10 after ~25
  // counts), and largest k (mirror image) falls back below it.
  std::vector<double> lam = {1.0, 1.0, 1.0, 1.5, 1.5};
  for (int i = 0; i < 6; ++i) lam.push_back(2.5);
  for (int i = 1; i <= 400; ++i) lam.push_back(2.5 + 0.37 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 48);
  std::vector<double> neg(lam);
  for (auto& x : neg) x = -x;
  auto an = hs::test::prescribed_spectrum<double>(neg, 4, 49);
  for (index_t k : {6, 7}) {
    auto s = hs::solve(a, nullptr, end_cfg(hs::TargetMode::Smallest, k));
    expect_window_ok(s);
    expect_same_set(s.eigenvalues, {lam.begin(), lam.begin() + k}, 1e-10);
    EXPECT_LE(worst_eta(s.report), 1e-12) << k;
    EXPECT_LE(s.report.factor_stats.factorizations, 18) << k;
    ASSERT_EQ(s.report.slices.size(), 1u);
    EXPECT_GT(s.report.slices[0].sigma - 2.5, 0.01) << "one-sided shift hugs the multiplet";
    EXPECT_LT(s.report.slices[0].sigma, 2.87);
    auto l = hs::solve(an, nullptr, end_cfg(hs::TargetMode::Largest, k));
    expect_window_ok(l);
    expect_same_set(l.eigenvalues, {neg.begin(), neg.begin() + k}, 1e-10);
    EXPECT_LE(worst_eta(l.report), 1e-12) << k;
  }
}

TEST(DriverTargets, SmallestInsideALargeNullSpaceReturnsCopies) {
  // PSD with a 600-dimensional null space and no zero filter: the 5 smallest are 5 null vectors.
  // The cluster is numerically one eigenvalue (verified by two counts), so 5 of its copies are a
  // valid answer - computing all 600 (the cluster's count) is not needed.
  std::vector<double> lam(600, 0.0);
  for (int i = 1; i <= 1400; ++i) lam.push_back(0.3 + 0.01 * i);
  std::vector<double> neg(lam);
  for (auto& x : neg) x = -x;
  const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 40);
  const auto an = hs::test::prescribed_spectrum<double>(neg, 4, 42);  // largest 5: mirror image
  for (auto mode : {hs::TargetMode::Smallest, hs::TargetMode::Largest}) {
    auto sol = hs::solve(mode == hs::TargetMode::Smallest ? a : an, nullptr, end_cfg(mode, 5));
    expect_window_ok(sol);
    ASSERT_EQ(sol.eigenvalues.size(), 5u);
    for (double x : sol.eigenvalues) EXPECT_LE(std::abs(x), 1e-9);
    EXPECT_EQ(sol.report.certificate.count_inertia, 5);
    EXPECT_LT(sol.report.op_applies, 300) << "the whole null space was computed";
    expect_b_orthonormal(sol, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
    bool warned = false;
    for (const auto& w : sol.report.warnings) warned = warned || w.find("numerically equal") != std::string::npos;
    EXPECT_TRUE(warned);
  }
}

TEST(DriverTargets, SmallestInsideCloseButDistinctEigenvaluesComputesThemAll) {
  // The 5 smallest end inside 40 close eigenvalues 1 + 1e-6 i: counts do not split them, but they
  // are not numerically equal, so the partial-cluster shortcut must not pick arbitrary members -
  // the two verification counts fail and all 40 are computed; the 5 smallest are exact.
  std::vector<double> lam = {0.2, 0.3, 0.4};
  for (int i = 0; i < 40; ++i) lam.push_back(1.0 + 1e-6 * i);
  for (int i = 1; i <= 300; ++i) lam.push_back(2.0 + 0.01 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 52);
  auto s = hs::solve(a, nullptr, end_cfg(hs::TargetMode::Smallest, 5));
  expect_window_ok(s);
  expect_same_set(s.eigenvalues, {lam.begin(), lam.begin() + 5}, 1e-12);
  EXPECT_EQ(s.report.certificate.count_inertia, 43);
}

TEST(DriverSlicing, IntervalsAroundA30FoldEigenvalue) {
  // 30 x 30 Laplacian: eigenvalue 4 has multiplicity 30. Cuts cannot split it, so one slice holds
  // all copies, at its edge; the slice's reruns get a shift on the multiplet, where single-vector
  // Krylov recovers the copies (from the slice centre two deflated reruns found only 1-2 each).
  auto a = hs::test::laplacian_2d<double>(30, 30);
  auto w = hs::test::laplacian_2d_eigenvalues(30, 30);
  for (double h : {0.1, 0.2}) {
    auto sol = hs::solve(a, nullptr, interval_cfg(4.0 - h, 4.0 + h));
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, in_interval(w, 4.0 - h - 1e-12, 4.0 + h + 1e-12), 1e-10);
    expect_b_orthonormal(sol, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
  }
}

TEST(DriverSlicing, MultipletsInsideSlicesGetAllCopies) {
  {  // 40-fold 0.5 inside [0.2, 0.8] with slices of at most 10
    std::vector<double> lam(40, 0.5);
    for (int i = 0; i < 300; ++i) lam.push_back(-1.0 + 0.01 * i);
    auto a = hs::test::prescribed_spectrum<double>(lam, 4, 46);
    auto sol = hs::solve(a, nullptr, interval_cfg(0.2, 0.8, 10));
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, in_interval(lam, 0.2, 0.8), 1e-10);
  }
  {  // 12 copies of 1.0, automatic slices
    std::vector<double> lam(12, 1.0);
    for (int i = 1; i <= 200; ++i) lam.push_back(0.5 + 0.0071 * i);
    auto a = hs::test::prescribed_spectrum<double>(lam, 4, 23);
    auto sol = hs::solve(a, nullptr, interval_cfg(0.9, 1.15));
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, in_interval(lam, 0.9, 1.15), 1e-10);
  }
}

TEST(DriverSlicing, IntervalEndOnAnEigenvalueWithInexactFactors) {
  // [-1, w_i] with w_i an exact eigenvalue and nothing below -1: the count at w_i sits ~1e-11 from
  // that eigenvalue. As a one-sided shift it broke single / BLR factors and residual correction
  // collapsed columns onto one eigenvector (BLR: 10 equal values accepted by the certificate). The
  // shift moves away, and counts are counts of distinct vectors.
  const index_t n = 400;
  auto a = hs::test::laplacian_1d<double>(n);
  auto w = hs::test::laplacian_1d_eigenvalues(n);
  struct Case {
    int idx;
    hs::FactorPrecision precision;
    hs::Toggle blr;
  };
  for (const Case cs : {Case{9, hs::FactorPrecision::Double, hs::Toggle::On},
                        Case{19, hs::FactorPrecision::Double, hs::Toggle::On},
                        Case{9, hs::FactorPrecision::Single, hs::Toggle::Off},
                        Case{4, hs::FactorPrecision::Single, hs::Toggle::Off}}) {
    auto cfg = interval_cfg(-1.0, w[cs.idx]);
    cfg.factor_precision = cs.precision;
    cfg.blr = cs.blr;
    auto sol = hs::solve(a, nullptr, cfg);
    expect_window_ok(sol);
    expect_same_set(sol.eigenvalues, {w.begin(), w.begin() + cs.idx + 1}, 1e-10);
    expect_b_orthonormal(sol, static_cast<const hs::CsrMatrix<double>*>(nullptr), 1e-9);
  }
}

TEST(DriverTargets, InertiaSearchExpandsFromTheOvershootingEnd) {
  // Largest 10: the start point (top Ritz value) overshoots; the search expands downward from it
  // instead of counting below the whole spectrum (nu = 0 is known a priori). Smallest: mirror.
  auto a = hs::test::laplacian_2d<double>(60, 61);
  for (auto mode : {hs::TargetMode::Largest, hs::TargetMode::Smallest}) {
    const std::string path = ::testing::TempDir() + "/driver_inertia_search.jsonl";
    {
      hs::Logger log(0, path);
      auto sol = hs::solve(a, nullptr, end_cfg(mode, 10), &log);
      expect_window_ok(sol);
    }
    std::ifstream in(path);
    std::string line;
    index_t counts = 0;
    while (std::getline(in, line)) {
      const auto j = nlohmann::json::parse(line);
      if (j.value("event", "") != "count") continue;
      ++counts;
      if (mode == hs::TargetMode::Largest) {
        EXPECT_GT(j["nu"].get<index_t>(), 0) << "count below the whole spectrum";
      } else {
        EXPECT_LT(j["nu"].get<index_t>(), a.nrows) << "count above the whole spectrum";
      }
    }
    EXPECT_GT(counts, 0);
  }
}

TEST(DriverTargets, NearestWithFewerEigenvaluesOutsideTheBandExplainsTheShortfall) {
  // 80 zeros and 20 nonzero eigenvalues, nearest 30 outside the band: 20 exist. The run fails
  // (k not reached) and says why.
  std::vector<double> lam(80, 0.0);
  for (int i = 1; i <= 20; ++i) lam.push_back(0.3 + 0.01 * i);
  auto a = hs::test::prescribed_spectrum<double>(lam, 4, 39);
  auto cfg = base_cfg();
  cfg.target.sigma = 0.01;
  cfg.target.k = 30;
  cfg.zero_filter_tol = 1e-6;
  auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_FALSE(sol.report.success);
  expect_same_set(sol.eigenvalues, {lam.begin() + 80, lam.end()}, 1e-10);
  bool explained = false;
  for (const auto& w : sol.report.warnings)
    explained = explained || w.find("outside the zero band") != std::string::npos;
  EXPECT_TRUE(explained);
}

TEST(DriverTargets, IndefiniteBSupportsNearestOnly) {
  const index_t n = 50;
  auto a = hs::test::laplacian_1d<double>(n);
  std::vector<double> d(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) d[i] = (i % 2 == 0) ? 1.0 : -1.0;
  auto b = hs::test::prescribed_spectrum<double>(d, 1, 38);  // diagonal-like indefinite B
  for (auto mode : {hs::TargetMode::Smallest, hs::TargetMode::Largest, hs::TargetMode::Interval}) {
    auto cfg = end_cfg(mode, 3);
    cfg.target.lower = 0.0;
    cfg.target.upper = 1.0;
    try {
      (void)hs::solve(a, &b, cfg);
      FAIL() << "expected Unsupported";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::Unsupported);
    }
  }
}

}  // namespace
