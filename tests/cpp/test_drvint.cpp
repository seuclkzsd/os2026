// Tests of the P4 driver integration (src/driver): the O(nnz) definiteness screen of B, block
// Krylov runs in the definite path (block_size, auto rule, deflated reruns), certificate counts on
// an InertiaCounter, the deterministic run-time models and the exclusive phase times.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
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

// ------------------------------------------------------------------------------ block Krylov

// 2-D Laplacian shifted to be indefinite (nx != ny: simple eigenvalues), nearest 8.
hs::CsrMatrix<double> shifted_laplacian(std::vector<double>& w) {
  auto a = hs::test::laplacian_2d<double>(40, 37);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] == i) a.values[p] -= 4.0;
  w = hs::test::laplacian_2d_eigenvalues(40, 37);
  for (auto& x : w) x -= 4.0;
  return a;
}

TEST(DriverBlocks, EveryBlockSizeGivesTheSameCertifiedSet) {
  std::vector<double> w;
  const auto a = shifted_laplacian(w);
  const double sigma = 0.3;
  for (const index_t p : {1, 2, 4, 8}) {
    auto cfg = nearest_cfg(sigma, 8);
    cfg.block_size = p;
    const auto sol = hs::solve(a, nullptr, cfg);
    EXPECT_TRUE(sol.report.success) << p;
    EXPECT_TRUE(sol.report.certificate.passed) << p;
    EXPECT_EQ(sol.report.block_size, p);
    for (const auto& pr : sol.report.pairs) EXPECT_LE(pr.backward_error, 1e-10) << p;
    expect_same_set(sol.eigenvalues, nearest(w, sigma, 8), 1e-10);
    expect_times_add_up(sol.report);
  }
}

// Block runs in the slices of a window target (interval, smallest), complex Hermitian.
TEST(DriverBlocks, SlicedTargetsWithBlocks) {
  const auto a = hs::test::random_hermitian<cplx>(600, 0.02, 41);
  const auto w = hs::test::dense_eigenvalues(a, static_cast<const hs::CsrMatrix<cplx>*>(nullptr));
  auto cfg = nearest_cfg(0.0, 7);
  cfg.block_size = 4;
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 0.5 * (w[200] + w[201]);
  cfg.target.upper = 0.5 * (w[260] + w[261]);
  cfg.slice_size = 24;  // three slices
  const auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.block_size, 4);
  EXPECT_GE(sol.report.slices.size(), 2u);
  expect_same_set(sol.eigenvalues, std::vector<double>(w.begin() + 201, w.begin() + 261), 1e-9);
  cfg.target.mode = hs::TargetMode::Smallest;
  cfg.slice_size = 0;
  const auto s = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(s.report.success);
  expect_same_set(s.eigenvalues, std::vector<double>(w.begin(), w.begin() + 7), 1e-9);
}

TEST(DriverBlocks, RerunWidthAndBasisColumns) {
  using hs::driver::lanczos_basis_columns;
  using hs::driver::rerun_block_size;
  EXPECT_EQ(rerun_block_size(1, 1), 2);  // missing copies rounded up to even
  EXPECT_EQ(rerun_block_size(1, 3), 4);
  EXPECT_EQ(rerun_block_size(4, 3), 4);  // never below the run's own width
  EXPECT_EQ(rerun_block_size(4, 5), 6);
  EXPECT_EQ(rerun_block_size(8, 2), 8);
  EXPECT_EQ(rerun_block_size(0, 0), 1);
  // p = 1: [D V v] + M*w with the auto ncv max(2 nev + 1, nev + 16).
  EXPECT_EQ(lanczos_basis_columns(100000, 0, 12, 0, 1), 28 + 2);
  EXPECT_EQ(lanczos_basis_columns(100000, 5, 12, 40, 1), 5 + 40 + 2);
  // p = 8, nev = 12 (eigen.hpp example): ncv = 64, basis + residual block + M*W workspace.
  EXPECT_EQ(lanczos_basis_columns(100000, 0, 12, 0, 8), 64 + 16);
  // Block layout within the free space: n - d = 90 >= 2 (ncv + p) with ncv = 32 (no room for the
  // 4 extra auto blocks).
  EXPECT_EQ(lanczos_basis_columns(100, 10, 12, 0, 8), 10 + 32 + 16);
  // A free space too small for the block layout: one single-vector cycle over it.
  EXPECT_EQ(lanczos_basis_columns(60, 10, 12, 0, 8), 10 + 50 + 2);
}

// Memory: a budget that holds the p = 1 basis but not the p = 8 one halves the block width (with
// a warning) instead of failing.
TEST(DriverBlocks, BlockWidthShrinksToTheMemoryBudget) {
  hs::SolverConfig cfg;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx c{cfg, nullptr, rep, clock};
  const index_t n = 2000000, nev = 12;
  const double rss = static_cast<double>(std::max<std::int64_t>(hs::current_rss_bytes(), 0));
  // Room for ~60 n-vectors of doubles beyond the current RSS: p = 8 needs 80 + 48 extra columns.
  cfg.memory_budget_gb = (rss + 60.0 * 8.0 * static_cast<double>(n) / 0.9) / 1073741824.0;
  const index_t p = hs::driver::fit_block_size<double>(c, n, nev, 0, 4 * nev, 8);
  EXPECT_LT(p, 8);
  EXPECT_GE(p, 1);
  ASSERT_FALSE(rep.warnings.empty());
  EXPECT_NE(rep.warnings.back().find("block size reduced"), std::string::npos);
  cfg.memory_budget_gb = 1000.0;
  EXPECT_EQ(hs::driver::fit_block_size<double>(c, n, nev, 0, 4 * nev, 8), 8);
}

// A six-fold eigenvalue at the edge of the wanted set, single-vector first run: rounding cannot
// bring its other copies within the run (they dominate the unwanted part of the operator only by
// a factor 1.4), so the certificate finds copies missing, and one deflated rerun of width >= the
// missing copies brings them all (a random start block has components in every copy), instead of
// one copy per rerun.
TEST(DriverBlocks, DeflatedRerunFindsAllMissingCopiesAtOnce) {
  std::vector<double> w;
  for (int i = 0; i < 6; ++i) w.push_back(1.0);
  for (int i = 1; i <= 40; ++i) {
    w.push_back(1.0 + 0.05 * i);
    w.push_back(1.0 - 0.05 * i);
  }
  for (int i = 0; i < 400; ++i) w.push_back(4.0 + 0.01 * i);
  const auto a = hs::test::prescribed_spectrum<double>(w, 3, 77);
  // Nearest 10 of 1.103: 1.1, 1.15, 1.05, 1.2 and the six copies of 1.0 (distance 0.103).
  auto cfg = nearest_cfg(1.103, 10);
  cfg.block_size = 1;
  cfg.deterministic = true;  // bitwise reproducible solves: the first run's misses do not vary
  const auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(sol.report.success);
  EXPECT_TRUE(sol.report.certificate.passed);
  expect_same_set(sol.eigenvalues, nearest(w, 1.103, 10), 1e-10);
  index_t reruns = 0;
  for (const auto& m : sol.report.warnings) reruns += m.find("deflated rerun") != std::string::npos;
  EXPECT_EQ(reruns, 1);  // the first run (p = 1) misses copies; the rerun (p = 6) has them all
  // Interval target around the multiplet, one slice with p = 1: its reruns as well.
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 0.93;
  cfg.target.upper = 1.12;  // centre shift 1.025: the multiplet does not dominate the operator
  const auto iv = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(iv.report.success);
  expect_same_set(iv.eigenvalues, {0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.1}, 1e-10);
  ASSERT_EQ(iv.report.slices.size(), 1u);
  EXPECT_LE(iv.report.slices[0].reruns, 1);  // here rounding usually supplies the copies at once
}

// ---------------------------------------------------------------------------- inertia counter

template <class T>
void expect_same_certificate(const hs::Solution<T>& x, const hs::Solution<T>& y) {
  EXPECT_TRUE(x.report.success);
  EXPECT_TRUE(y.report.success);
  EXPECT_EQ(x.report.certificate.count_inertia, y.report.certificate.count_inertia);
  EXPECT_EQ(x.report.certificate.count_found, y.report.certificate.count_found);
  // Window ends come from Ritz values, equal up to the last bits of L0-OpenMP solves.
  const double tol = 1e-10 * std::max(1.0, std::abs(x.report.certificate.upper));
  EXPECT_NEAR(x.report.certificate.lower, y.report.certificate.lower, tol);
  EXPECT_NEAR(x.report.certificate.upper, y.report.certificate.upper, tol);
  expect_same_set(x.eigenvalues, y.eigenvalues, 1e-10);
}

template <class T>
void counter_matches_main_object(std::uint64_t seed) {
  const auto a = hs::test::random_hermitian<T>(500, 0.02, seed);
  const auto b = hs::test::random_hpd<T>(500, 0.01, seed + 1);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  auto cfg = nearest_cfg(0.5 * (w[230] + w[231]), 9);
  cfg.block_size = 1;  // same Krylov runs: only the source of the counts differs
  cfg.inertia_counter = hs::Toggle::Off;
  const auto off = hs::solve(a, &b, cfg);
  cfg.inertia_counter = hs::Toggle::On;
  const auto on = hs::solve(a, &b, cfg);
  expect_same_certificate(off, on);
  expect_same_set(on.eigenvalues, nearest(w, cfg.target.sigma, 9), 1e-9);
  // The counter keeps the main factors: K(sigma) once, nothing refactorized for the counts.
  EXPECT_EQ(on.report.factor_stats.factorizations, off.report.factor_stats.factorizations - 2);
  EXPECT_GT(on.report.time_certificate_s, 0.0);
  // Window target, counts on the counter for every cut and end.
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 0.5 * (w[100] + w[101]);
  cfg.target.upper = 0.5 * (w[140] + w[141]);
  cfg.slice_size = 16;
  const auto won = hs::solve(a, &b, cfg);
  cfg.inertia_counter = hs::Toggle::Off;
  const auto woff = hs::solve(a, &b, cfg);
  EXPECT_EQ(won.report.certificate.count_inertia, 40);
  expect_same_certificate(woff, won);
  ASSERT_EQ(won.report.slices.size(), woff.report.slices.size());
  for (std::size_t i = 0; i < won.report.slices.size(); ++i) {
    EXPECT_NEAR(won.report.slices[i].lower, woff.report.slices[i].lower, 1e-10);
    EXPECT_EQ(won.report.slices[i].count_inertia, woff.report.slices[i].count_inertia);
  }
}

TEST(DriverCounter, CountsEqualMainObjectCountsReal) { counter_matches_main_object<double>(61); }
TEST(DriverCounter, CountsEqualMainObjectCountsComplex) { counter_matches_main_object<cplx>(62); }

// A single-precision main factor always counts on the (double) counter: the same certificate as an
// all-double run, without an analysis per count.
TEST(DriverCounter, SinglePrecisionFactorCountsInDouble) {
  std::vector<double> w;
  const auto a = shifted_laplacian(w);
  auto cfg = nearest_cfg(0.3, 6);
  cfg.block_size = 1;
  const auto dbl = hs::solve(a, nullptr, cfg);
  cfg.factor_precision = hs::FactorPrecision::Single;
  const auto sgl = hs::solve(a, nullptr, cfg);
  EXPECT_EQ(sgl.report.factor_stats.analyses, dbl.report.factor_stats.analyses);
  expect_same_certificate(dbl, sgl);
}

// ------------------------------------------------------------------------------ models, phases

TEST(DriverModels, OrderingDecisionIsAPureFunctionOfTheEstimate) {
  hs::FactorEstimate e;
  e.factor_entries = 50000000;
  e.flops = 7e10;
  const hs::driver::MumpsInputSize in{97336.0, 1306396.0};
  const auto d1 = hs::driver::decide_ordering(e, in, 1306396.0, 4, 3, 40);
  const auto d2 = hs::driver::decide_ordering(e, in, 1306396.0, 4, 3, 40);
  EXPECT_EQ(d1.use_nd, d2.use_nd);
  EXPECT_EQ(d1.saving_s, d2.saving_s);
  EXPECT_EQ(d1.nd_cost_s, hs::driver::kNdAnalysisOverAmd * d1.amd_analysis_s);
  EXPECT_TRUE(d1.use_nd);  // 3-D fill: nested dissection pays
  e.factor_entries = 3000000;  // 2-D-like fill, cheap factorization
  e.flops = 1.7e8;
  const auto d3 = hs::driver::decide_ordering(e, {99459.0, 297605.0}, 297605.0, 4, 4, 40);
  EXPECT_FALSE(d3.use_nd);
  // The analysis model grows with each of its inputs.
  using hs::driver::model_analysis_s;
  EXPECT_LT(model_analysis_s(1e4, 1e5, 1e6), model_analysis_s(2e4, 1e5, 1e6));
  EXPECT_LT(model_analysis_s(1e4, 1e5, 1e6), model_analysis_s(1e4, 2e5, 1e6));
  EXPECT_LT(model_analysis_s(1e4, 1e5, 1e6), model_analysis_s(1e4, 1e5, 2e6));
}

TEST(DriverModels, MumpsInputSizeOfTheEmbedding) {
  const auto a = hs::test::laplacian_1d<double>(10);  // 10 diagonal + 9 upper entries
  const hs::ShiftedPencil<double> pr(a, nullptr);
  EXPECT_EQ(hs::driver::mumps_input_size(pr).order, 10.0);
  EXPECT_EQ(hs::driver::mumps_input_size(pr).entries, static_cast<double>(pr.nnz()));
  const auto ac = hs::test::laplacian_1d<cplx>(10);
  const hs::ShiftedPencil<cplx> pc(ac, nullptr);
  EXPECT_EQ(hs::driver::mumps_input_size(pc).order, 20.0);
  EXPECT_EQ(hs::driver::mumps_input_size(pc).entries, 4.0 * static_cast<double>(pc.nnz()) - 10.0);
}

TEST(DriverPhases, CertificateTimeIsItsOwnPhase) {
  std::vector<double> w;
  const auto a = shifted_laplacian(w);
  auto cfg = nearest_cfg(0.3, 6);
  const auto with = hs::solve(a, nullptr, cfg);
  expect_times_add_up(with.report);
  EXPECT_TRUE(with.report.certificate.performed);
  EXPECT_GT(with.report.time_certificate_s, 0.0);
  cfg.inertia_check = hs::Toggle::Off;
  const auto without = hs::solve(a, nullptr, cfg);
  expect_times_add_up(without.report);
  EXPECT_FALSE(without.report.certificate.performed);
  EXPECT_EQ(without.report.time_certificate_s, 0.0);
}

}  // namespace
