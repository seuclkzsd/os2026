// Driver hygiene and GHIEP planning (module anaint, P5; src/driver/solver.cpp, indefinite.cpp):
// the in-memory Hermitian check by the loader's rules (scaled defect, exact-value test) at extreme
// magnitudes, non-finite and all-zero inputs, the OpenMP thread count restored on exit, B proven
// indefinite by its norm-estimate Lanczos (no factorization of B), and the GHIEP memory plan.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
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

void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
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

/// Events named `name` of a JSON-lines log.
std::vector<nlohmann::json> events(const std::string& path, const std::string& name) {
  std::vector<nlohmann::json> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto j = nlohmann::json::parse(line);
    if (j.value("event", "") == name) out.push_back(j);
  }
  return out;
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

// ------------------------------------------------------------------------ definiteness of B

TEST(DriverBDefiniteness, CheapTestsInOrder) {
  using hs::driver::Definiteness;
  using hs::driver::test_b_definiteness;
  const hs::driver::NormEstimate none;
  // Diagonal: by the signs.
  std::vector<index_t> idx{0, 1, 2, 3};
  auto diag = hs::csr_from_coo<double>(4, 4, idx, idx, std::vector<double>{1.0, 2.0, 3.0, 4.0});
  auto r = test_b_definiteness(diag, none);
  EXPECT_EQ(r.verdict, Definiteness::Positive);
  EXPECT_EQ(r.how, "diagonal");
  EXPECT_TRUE(r.inertia.valid);
  diag.values[2] = -3.0;
  r = test_b_definiteness(diag, none);
  EXPECT_EQ(r.verdict, Definiteness::NotPositive);
  EXPECT_EQ(r.inertia.neg, 1);
  // Strictly dominant with a positive diagonal: HPD.
  r = test_b_definiteness(hs::test::random_hpd<cplx>(50, 0.1, 3), none);
  EXPECT_EQ(r.verdict, Definiteness::Positive);
  EXPECT_EQ(r.how, "dominant");
  // A diagonal entry <= 0.
  auto lap = hs::test::laplacian_2d<double>(10, 9);  // diagonal 4, not strictly dominant
  r = test_b_definiteness(lap, none);
  EXPECT_EQ(r.verdict, Definiteness::Unknown);  // HPD, but only its inertia tells
  auto neg_diag = lap;
  for (index_t p = neg_diag.rowptr[5]; p < neg_diag.rowptr[6]; ++p)
    if (neg_diag.colind[p] == 5) neg_diag.values[p] = 0.0;
  r = test_b_definiteness(neg_diag, none);
  EXPECT_EQ(r.verdict, Definiteness::NotPositive);
  EXPECT_EQ(r.how, "diagonal_entry");
  // The norm-estimate Lanczos: a Ritz value below -kIndefiniteRitz * ||B||_2 proves indefiniteness.
  hs::driver::NormEstimate est;
  est.steps = 30;
  est.norm2 = 8.0;
  est.ritz_max = 8.0;
  est.ritz_min = -2.0 * hs::driver::kIndefiniteRitz * 8.0;
  r = test_b_definiteness(lap, est);
  EXPECT_EQ(r.verdict, Definiteness::NotPositive);
  EXPECT_EQ(r.how, "lanczos");
  est.ritz_min = -0.5 * hs::driver::kIndefiniteRitz * 8.0;  // within rounding of a PSD matrix
  EXPECT_EQ(test_b_definiteness(lap, est).verdict, Definiteness::Unknown);
  // The real estimate: L - 2I is indefinite with eigenvalues down to ~ -2.
  auto shifted = lap;
  for (index_t i = 0; i < shifted.nrows; ++i)
    for (index_t p = shifted.rowptr[i]; p < shifted.rowptr[i + 1]; ++p)
      if (shifted.colind[p] == i) shifted.values[p] -= 2.0;
  r = test_b_definiteness(shifted, hs::driver::estimate_norm2(shifted, 30, 2026));
  EXPECT_EQ(r.verdict, Definiteness::NotPositive);
  EXPECT_EQ(r.how, "lanczos");
  EXPECT_EQ(test_b_definiteness(lap, hs::driver::estimate_norm2(lap, 30, 2026)).verdict, Definiteness::Unknown);
}

/// A = L (2-D Laplacian, HPD), B = L - 2I (indefinite, positive diagonal, not dominant): the
/// eigenvalues l / (l - 2) are real (A definite) and known from those of L.
template <class T>
void ghiep_without_factorizing_b() {
  const index_t nx = 30, ny = 20;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  auto b = a;
  for (index_t i = 0; i < b.nrows; ++i)
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] == i) b.values[p] -= 2.0;
  std::vector<double> w;
  for (double l : hs::test::laplacian_2d_eigenvalues(nx, ny)) w.push_back(l / (l - 2.0));
  auto cfg = nearest_cfg(3.3, 6);
  cfg.deterministic = true;
  // One file per scalar type: ctest runs the typed variants in parallel.
  const std::string path =
      ::testing::TempDir() + "/anaint_ghiep_lanczos_" + (hs::is_complex_v<T> ? "complex" : "real") + ".jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, &b, cfg, &log);
  }
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(sol.report.success);
  expect_same_set(sol.eigenvalues, nearest(w, 3.3, 6), 1e-9);
  const auto bi = events(path, "b_inertia");
  ASSERT_EQ(bi.size(), 1u);
  EXPECT_EQ(bi[0].at("how").get<std::string>(), "lanczos");
  // No factorization of B: the same factorizations as when the type is given.
  auto given = cfg;
  given.problem = hs::ProblemType::GeneralizedIndefinite;
  const auto ref = hs::solve(a, &b, given);
  EXPECT_EQ(sol.report.factor_stats.factorizations, ref.report.factor_stats.factorizations);
  expect_bitwise_equal(sol, ref);
}

TEST(DriverBDefiniteness, LanczosProvesIndefiniteBReal) { ghiep_without_factorizing_b<double>(); }
TEST(DriverBDefiniteness, LanczosProvesIndefiniteBComplex) { ghiep_without_factorizing_b<cplx>(); }

// SolveReport::block_size of a GHIEP solve is the width its run used (it said 1 while the runs used
// the auto p = 2 of ghiep_block_size): n = 21000 >= 2e4 and k = 6 >= 4 give p = 2, a configured 1
// gives 1.
TEST(DriverGhiepReport, BlockSizeIsTheRunsWidth) {
  const index_t nx = 150, ny = 140;
  const auto a = hs::test::laplacian_2d<double>(nx, ny);
  auto b = a;
  for (index_t i = 0; i < b.nrows; ++i)
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] == i) b.values[p] -= 2.0;
  std::vector<double> w;
  for (double l : hs::test::laplacian_2d_eigenvalues(nx, ny)) w.push_back(l / (l - 2.0));
  auto cfg = nearest_cfg(3.3, 6);
  ASSERT_EQ(hs::driver::ghiep_block_size(cfg, a.nrows, 6), 2);
  const std::string path = ::testing::TempDir() + "/anaint_ghiep_block_size.jsonl";
  hs::Solution<double> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, &b, cfg, &log);
  }
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(sol.report.success);
  expect_same_set(sol.eigenvalues, nearest(w, 3.3, 6), 1e-9);
  const auto runs = events(path, "krylov_schur");
  ASSERT_FALSE(runs.empty());
  EXPECT_EQ(runs[0].at("block_size").get<index_t>(), 2);
  EXPECT_EQ(sol.report.block_size, 2);
  cfg.block_size = 1;
  const auto single = hs::solve(a, &b, cfg);
  EXPECT_TRUE(single.report.success);
  EXPECT_EQ(single.report.block_size, 1);
}

// ------------------------------------------------------------------------ GHIEP memory plan

template <class T>
void ghiep_reserve_is_the_general_peak() {
  hs::SolverConfig cfg;
  cfg.target.k = 18;
  for (const index_t n : {index_t{5000}, index_t{100000}, index_t{1000000}}) {
    hs::KrylovSchurOptions<T> ko;
    ko.nev = 18;
    ko.block_size = hs::driver::ghiep_block_size(cfg, n, 18);
    const index_t m = 18 + (hs::is_complex_v<T> ? 0 : 1);
    EXPECT_EQ(hs::driver::ghiep_reserve_columns<T>(cfg, n), hs::krylov_schur_general_peak_columns<T>(n, ko) + 2 * m)
        << n;
  }
  EXPECT_EQ(hs::driver::ghiep_block_size(cfg, 100000, 18), 2);
  EXPECT_EQ(hs::driver::ghiep_block_size(cfg, 5000, 18), 1);
  // Explicit block width (capped at nev), single precision (refinement workspace 2p).
  cfg.block_size = 8;
  cfg.factor_precision = hs::FactorPrecision::Single;
  hs::KrylovSchurOptions<T> ko;
  ko.nev = 18;
  ko.block_size = 8;
  const index_t m = 18 + (hs::is_complex_v<T> ? 0 : 1);
  EXPECT_EQ(hs::driver::ghiep_reserve_columns<T>(cfg, 100000),
            hs::krylov_schur_general_peak_columns<T>(100000, ko) + 2 * 8 + 2 * m);
  // k beyond n and an ncv the run would reject do not throw here (the run reports them).
  cfg.target.k = 50;
  cfg.ncv = 3;
  EXPECT_GT(hs::driver::ghiep_reserve_columns<T>(cfg, 40), 0);
}

TEST(DriverGhiepMemory, ReserveIsTheGeneralKrylovSchurPeakReal) { ghiep_reserve_is_the_general_peak<double>(); }
TEST(DriverGhiepMemory, ReserveIsTheGeneralKrylovSchurPeakComplex) { ghiep_reserve_is_the_general_peak<cplx>(); }

// The waveguide shape (k = 18, n = 1e6, real): the general solver's plan holds far fewer columns
// than the definite path's (ncv guess + 6k = 196), so the factorization cap grows by ~0.7 GB.
TEST(DriverGhiepMemory, WaveguideReserveIsBelowTheDefinitePlan) {
  hs::SolverConfig cfg;
  cfg.target.k = 18;
  const index_t cols = hs::driver::ghiep_reserve_columns<double>(cfg, 1000000);
  EXPECT_LT(cols, 130);
  EXPECT_GT(cols, 18 * 3);
}

}  // namespace
