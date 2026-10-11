// Driver hygiene (module anaint, P5; src/driver/solver.cpp): the in-memory Hermitian check by the
// loader's rules (scaled defect, exact-value test) at extreme magnitudes, non-finite and all-zero
// inputs, the OpenMP thread count restored on exit.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
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

template <class T>
hs::CsrMatrix<T> scaled(hs::CsrMatrix<T> a, double s) {
  for (auto& v : a.values) v *= s;
  return a;
}

/// A Hermitian matrix with one off-diagonal pair made asymmetric by `rel` (relative to the entry).
template <class T>
hs::CsrMatrix<T> nearly_hermitian(index_t n, double rel, std::uint64_t seed) {
  auto a = hs::test::random_hermitian<T>(n, 0.05, seed);
  for (index_t i = 0; i < n; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] > i) {
        a.values[p] *= 1.0 + rel;
        return a;
      }
  return a;
}

template <class T>
void expect_bitwise_equal(const hs::Solution<T>& x, const hs::Solution<T>& y) {
  ASSERT_EQ(x.eigenvalues.size(), y.eigenvalues.size());
  EXPECT_EQ(std::memcmp(x.eigenvalues.data(), y.eigenvalues.data(), x.eigenvalues.size() * sizeof(double)), 0);
  ASSERT_EQ(x.eigenvectors.cols(), y.eigenvectors.cols());
  EXPECT_EQ(std::memcmp(x.eigenvectors.data(), y.eigenvectors.data(),
                        static_cast<std::size_t>(x.eigenvectors.rows() * x.eigenvectors.cols()) * sizeof(T)),
            0);
}

// ------------------------------------------------------------------------ Hermitian input

// x 1e155: the unscaled sums of squares of hs::hermitian_defect overflow (inf / inf = NaN), which
// passed the tolerance test and failed later inside LAPACK. The scaled defect rejects it.
template <class T>
void huge_non_hermitian_is_rejected() {
  const auto a = scaled(nearly_hermitian<T>(40, 0.5, 3), 1e155);
  try {
    (void)hs::solve(a, nullptr, nearest_cfg(0.0, 2));
    FAIL() << "accepted a non-Hermitian matrix";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::NotHermitian) << e.what();
  }
  // As B as well (A fine).
  const auto b = scaled(nearly_hermitian<T>(40, 0.5, 4), 1e155);
  const auto good = hs::test::random_hermitian<T>(40, 0.05, 5);
  try {
    (void)hs::solve(good, &b, nearest_cfg(0.0, 2));
    FAIL() << "accepted a non-Hermitian B";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::NotHermitian) << e.what();
  }
}

TEST(DriverHermitianInput, HugeNonHermitianIsRejectedReal) { huge_non_hermitian_is_rejected<double>(); }
TEST(DriverHermitianInput, HugeNonHermitianIsRejectedComplex) { huge_non_hermitian_is_rejected<cplx>(); }

// x 1e-150: the defect of a nearly Hermitian matrix (1e-13 relative) underflowed to 0 in the
// unscaled sums, so its Hermitian part was not taken - the factorization (one triangle) and the
// operators (full matrix) then saw different matrices. Now the defect is reported at its true
// relative size and the solve is that of the Hermitian part, bitwise.
template <class T>
void tiny_nearly_hermitian_takes_the_hermitian_part() {
  const index_t n = 120;
  const auto unit = nearly_hermitian<T>(n, 1e-13, 9);
  const auto a = scaled(unit, 1e-150);
  const double rel = hs::hermitian_defect(unit);  // scale-invariant; fine at magnitude 1
  ASSERT_GT(rel, 0.0);
  ASSERT_LT(rel, 1e-12);
  auto cfg = nearest_cfg(0.0, 4);
  cfg.deterministic = true;
  const auto w = hs::test::dense_eigenvalues(hs::hermitian_part(unit), static_cast<const hs::CsrMatrix<T>*>(nullptr));
  cfg.target.sigma = 1e-150 * 0.5 * (w[60] + w[61]);
  const auto sol = hs::solve(a, nullptr, cfg);
  // The asymmetry (1e-13 relative) carries ~eps / 1e-13 relative rounding, differently at each scale.
  EXPECT_NEAR(sol.report.hermitian_defect_a, rel, 1e-2 * rel);
  EXPECT_TRUE(sol.report.success);
  std::vector<double> want = nearest(w, 0.5 * (w[60] + w[61]), 4);
  for (auto& x : want) x *= 1e-150;
  std::vector<double> got = sol.eigenvalues;
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i] / want[i], 1.0, 1e-9) << i;
  const auto sym = hs::solve(hs::hermitian_part(a), nullptr, cfg);
  EXPECT_EQ(sym.report.hermitian_defect_a, 0.0);
  expect_bitwise_equal(sol, sym);
}

TEST(DriverHermitianInput, TinyNearlyHermitianTakesTheHermitianPartReal) {
  tiny_nearly_hermitian_takes_the_hermitian_part<double>();
}
TEST(DriverHermitianInput, TinyNearlyHermitianTakesTheHermitianPartComplex) {
  tiny_nearly_hermitian_takes_the_hermitian_part<cplx>();
}

// The decision to symmetrize is the exact value comparison: an exactly Hermitian input reports
// defect 0 at any magnitude, a defect within hermitian_tol is reported and removed.
TEST(DriverHermitianInput, DefectMatchesTheLoadersRule) {
  const auto a = hs::test::random_hermitian<cplx>(60, 0.1, 13);
  for (const double s : {1e-150, 1.0, 1e150}) {
    const auto sol = hs::solve(scaled(a, s), nullptr, nearest_cfg(0.0, 2));
    EXPECT_EQ(sol.report.hermitian_defect_a, 0.0) << s;
    EXPECT_TRUE(sol.report.success) << s;
  }
  // A defect above hermitian_tol is rejected at small magnitudes too (was 0 below ~1e-154).
  const auto bad = scaled(nearly_hermitian<double>(50, 1e-6, 17), 1e-160);
  EXPECT_THROW((void)hs::solve(bad, nullptr, nearest_cfg(0.0, 2)), hs::Error);
}

// ------------------------------------------------------------------------ degenerate input

TEST(DriverDegenerateInput, ZeroMatricesAreRejectedClearly) {
  const index_t n = 30;
  auto zero = hs::test::laplacian_1d<double>(n);
  std::fill(zero.values.begin(), zero.values.end(), 0.0);  // stored entries, all zero
  hs::CsrMatrix<double> empty_pattern;
  empty_pattern.nrows = empty_pattern.ncols = n;
  empty_pattern.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  const auto lap = hs::test::laplacian_1d<double>(n);
  auto expect_invalid = [](auto&& f, const std::string& what) {
    try {
      f();
      FAIL() << "no error for " << what;
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
      EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    }
  };
  expect_invalid([&] { (void)hs::solve(zero, nullptr, nearest_cfg(0.0, 2)); }, "A is the zero matrix");
  expect_invalid([&] { (void)hs::solve(empty_pattern, nullptr, nearest_cfg(1.0, 2)); }, "A is the zero matrix");
  expect_invalid([&] { (void)hs::solve(lap, &zero, nearest_cfg(1.0, 2)); }, "B is the zero matrix");
  auto nan = lap;
  nan.values[5] = std::numeric_limits<double>::quiet_NaN();
  expect_invalid([&] { (void)hs::solve(nan, nullptr, nearest_cfg(1.0, 2)); }, "non-finite");
  auto inf = hs::to_complex(lap);
  inf.values[7] = cplx(0.0, std::numeric_limits<double>::infinity());
  expect_invalid([&] { (void)hs::solve(hs::to_complex(lap), &inf, nearest_cfg(1.0, 2)); }, "B has a non-finite");
}

// ------------------------------------------------------------------------ OpenMP threads

TEST(DriverThreads, ThreadCountIsRestoredOnReturnAndOnError) {
#ifdef _OPENMP
  const int before = omp_get_max_threads();
  omp_set_num_threads(3);
  const auto a = hs::test::laplacian_1d<double>(200);
  auto cfg = nearest_cfg(1.0, 3);
  cfg.threads = 2;
  const auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(omp_get_max_threads(), 3);
  const auto bad = scaled(nearly_hermitian<double>(40, 0.5, 3), 1.0);
  EXPECT_THROW((void)hs::solve(bad, nullptr, cfg), hs::Error);
  EXPECT_EQ(omp_get_max_threads(), 3);
  cfg.threads = 0;  // follow the caller's setting, nothing to restore
  (void)hs::solve(a, nullptr, cfg);
  EXPECT_EQ(omp_get_max_threads(), 3);
  omp_set_num_threads(before);
#else
  GTEST_SKIP() << "built without OpenMP";
#endif
}

}  // namespace
