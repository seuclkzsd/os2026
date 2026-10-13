// Driver hook-up of the blocktune block-size rule (P5c integration, src/driver): nearest-target
// runs take eigen_detail::recommended_block_size on the shape of their shift's factorization
// (block_size_for(cfg, ShiftInvertShape)), spectrum slices keep the P4 width
// (block_size_for(cfg, nev)), and the memory plan reserves the basis of the P4 width only
// (krylov_reservation; the factorization has priority, p = 8 is taken where the factors leave room).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "eigen/block_rule.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;
using hs::eigen_detail::ShiftInvertShape;

ShiftInvertShape shape(index_t n, index_t nev, double entries, double nnz_b, bool cx) {
  ShiftInvertShape s;
  s.n = n;
  s.nev = nev;
  s.factor_entries = entries;
  s.nnz_b = nnz_b;
  s.complex_scalars = cx;
  return s;
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

std::vector<double> nearest(std::vector<double> w, double sigma, index_t k) {
  std::stable_sort(w.begin(), w.end(),
                   [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
  w.resize(static_cast<std::size_t>(k));
  std::sort(w.begin(), w.end());
  return w;
}

void expect_same_set(std::vector<double> got, const std::vector<double>& want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

/// 2-D Laplacian 40 x 37 shifted by -4 (eigenvalues in (-4, 4), low fill: AMD).
template <class T>
hs::CsrMatrix<T> shifted_laplacian(std::vector<double>& w) {
  auto a = hs::test::laplacian_2d<T>(40, 37);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] == i) a.values[p] -= T(4.0);
  w = hs::test::laplacian_2d_eigenvalues(40, 37);
  for (auto& x : w) x -= 4.0;
  return a;
}

// ------------------------------------------------------------------------------ the widths

// Auto: the rule's pick for the shape (calibration shapes of block_rule.cpp where it differs
// from the P4 width 4 for nev >= 8, else 1); an explicit width wins.
TEST(BlocktuneDriver, NearestAutoWidthIsTheRulesPick) {
  hs::SolverConfig cfg;
  ASSERT_EQ(cfg.block_size, 0);
  const ShiftInvertShape box = shape(109744, 10, 5.376e7, 109744, false);   // yee-pec-box-1e5
  const ShiftInvertShape cav = shape(99459, 6, 3.871e6, 495751, false);     // cav2d-rect-pert-1e5
  const ShiftInvertShape qa8 = shape(66127, 10, 2.795e7, 1660579, false);   // ss-qa8fk-qa8fm
  ASSERT_EQ(hs::eigen_detail::recommended_block_size(box), 8);
  ASSERT_EQ(hs::eigen_detail::recommended_block_size(cav), 2);
  ASSERT_EQ(hs::eigen_detail::recommended_block_size(qa8), 4);
  EXPECT_EQ(hs::driver::block_size_for(cfg, box), 8);
  EXPECT_EQ(hs::driver::block_size_for(cfg, cav), 2);
  EXPECT_EQ(hs::driver::block_size_for(cfg, qa8), 4);
  // the P4 widths of the same nev (slices, plan_work)
  EXPECT_EQ(hs::driver::block_size_for(cfg, box.nev), 4);
  EXPECT_EQ(hs::driver::block_size_for(cfg, cav.nev), 1);
  cfg.block_size = 3;
  EXPECT_EQ(hs::driver::block_size_for(cfg, box), 3);
  EXPECT_EQ(hs::driver::block_size_for(cfg, cav), 3);
  cfg.block_size = 1;
  EXPECT_EQ(hs::driver::block_size_for(cfg, box), 1);
}

// The memory plan reserves the P4 width for every target (P5c integration review: the p = 8
// basis shrank the factorization cap); an explicit width up to that.
TEST(BlocktuneDriver, ReservedWidth) {
  hs::SolverConfig cfg;
  for (const auto mode : {hs::TargetMode::Nearest, hs::TargetMode::Interval, hs::TargetMode::Smallest,
                          hs::TargetMode::Largest}) {
    cfg.target.mode = mode;
    EXPECT_EQ(hs::driver::reserved_block_size(cfg), hs::driver::kMaxAutoBlockSize);
  }
  ASSERT_EQ(hs::driver::kMaxAutoBlockSize, 4);
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.block_size = 2;
  EXPECT_EQ(hs::driver::reserved_block_size(cfg), 2);
  cfg.block_size = 16;  // not reserved beyond the P4 width (the factorization has priority)
  EXPECT_EQ(hs::driver::reserved_block_size(cfg), hs::driver::kMaxAutoBlockSize);
}

TEST(BlocktuneDriver, KrylovReservationIsTheP4Plan) {
  using hs::driver::lanczos_basis_columns;
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.k = 12;
  const index_t n = 100000;
  const auto definite = [&](index_t p) {
    return std::max<index_t>(2 * (12 + 16) + 1, lanczos_basis_columns(n, 0, 12, 0, p)) + 6 * 12;
  };
  ASSERT_GT(definite(8), definite(4));  // the wider basis is larger: not reserved
  auto r = hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Standard, n);
  EXPECT_EQ(r.block_size, 4);
  EXPECT_EQ(r.columns, definite(4));
  EXPECT_EQ(hs::driver::krylov_reservation<cplx>(cfg, hs::ProblemType::Standard, n).columns, definite(4));
  // GHIEP: its own plan; type open: the larger of both.
  const index_t g = hs::driver::ghiep_reserve_columns<double>(cfg, n);
  EXPECT_EQ(hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::GeneralizedIndefinite, n).columns, g);
  EXPECT_EQ(hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Auto, n).columns,
            std::max(g, definite(4)));
  // Window targets reserve the P4 width too; an explicit width at most that.
  cfg.target.mode = hs::TargetMode::Smallest;
  EXPECT_EQ(hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Standard, n).block_size, 4);
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.block_size = 2;
  r = hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Standard, n);
  EXPECT_EQ(r.block_size, 2);
  EXPECT_EQ(r.columns, definite(2));
}

// ------------------------------------------------------------------------------ end to end

// A nearest solve runs the rule's width for the shape of its shift's factorization (entries from
// the "factorization" event, as bench/blocktune_fix_e2e.cpp reads them); the memory plan
// reserved the P4 basis. k = 6 on this low-fill pattern: the rule picks 2 where the P4 width
// was 1 - the test fails if the driver went back to block_size_for(cfg, nev).
template <class T>
void nearest_runs_the_rules_width() {
  std::vector<double> w;
  const auto a = shifted_laplacian<T>(w);
  for (const index_t k : {6, 10}) {
    const auto cfg = nearest_cfg(0.3, k);
    const std::string path = ::testing::TempDir() + "/blocktune_driver_" +
                             (hs::is_complex_v<T> ? "complex" : "real") + std::to_string(k) + ".jsonl";
    hs::Solution<T> sol;
    {
      hs::Logger log(0, path);
      sol = hs::solve(a, nullptr, cfg, &log);
    }
    EXPECT_TRUE(sol.report.success) << k;
    EXPECT_TRUE(sol.report.certificate.passed) << k;
    expect_same_set(sol.eigenvalues, nearest(w, 0.3, k), 1e-9);
    const auto fac = events(path, "factorization");
    ASSERT_FALSE(fac.empty()) << k;
    const ShiftInvertShape s = shape(a.nrows, k, fac[0].at("entries").get<double>(), 0.0, hs::is_complex_v<T>);
    const index_t rule = hs::eigen_detail::recommended_block_size(s);
    EXPECT_EQ(sol.report.block_size, rule) << k;
    if (k == 6) {
      ASSERT_EQ(hs::driver::block_size_for(cfg, k), 1);
      EXPECT_EQ(rule, 2) << "entries " << s.factor_entries;
    } else {
      EXPECT_GE(rule, hs::eigen_detail::kRuleFloorWidth);  // the floor from 8 wanted pairs on
    }
    const auto plan = events(path, "memory_plan");
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].at("reserved_block_size").get<index_t>(), hs::driver::kMaxAutoBlockSize) << k;
    EXPECT_GT(plan[0].at("cap_mb").get<double>(), 0.0);
    // An explicit width wins and is reserved up to the P4 width.
    auto fixed = cfg;
    fixed.block_size = 4;
    const std::string path4 = path + ".p4";
    hs::Solution<T> sol4;
    {
      hs::Logger log(0, path4);
      sol4 = hs::solve(a, nullptr, fixed, &log);
    }
    EXPECT_TRUE(sol4.report.success);
    EXPECT_EQ(sol4.report.block_size, 4);
    const auto plan4 = events(path4, "memory_plan");
    ASSERT_EQ(plan4.size(), 1u);
    EXPECT_EQ(plan4[0].at("reserved_block_size").get<index_t>(), 4);
    EXPECT_EQ(plan4[0].at("reserved_columns").get<index_t>(), plan[0].at("reserved_columns").get<index_t>());
  }
}

TEST(BlocktuneDriver, NearestRunsTheRulesWidthReal) { nearest_runs_the_rules_width<double>(); }
TEST(BlocktuneDriver, NearestRunsTheRulesWidthComplex) { nearest_runs_the_rules_width<cplx>(); }

// GHEP: nnz(B) enters the shape (diagonal B: the report's nnz_b).
TEST(BlocktuneDriver, GhepShapeCountsB) {
  std::vector<double> w;
  const auto a = shifted_laplacian<double>(w);
  const index_t n = a.nrows;
  std::vector<index_t> idx(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
  const std::vector<double> ones(static_cast<std::size_t>(n), 2.0);
  const auto b = hs::csr_from_coo<double>(n, n, idx, idx, ones);  // B = 2 I: lambda / 2
  const auto cfg = nearest_cfg(0.15, 6);
  const std::string path = ::testing::TempDir() + "/blocktune_driver_ghep.jsonl";
  hs::Solution<double> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, &b, cfg, &log);
  }
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  for (auto& x : w) x *= 0.5;
  expect_same_set(sol.eigenvalues, nearest(w, 0.15, 6), 1e-9);
  const auto fac = events(path, "factorization");
  ASSERT_FALSE(fac.empty());
  ASSERT_EQ(sol.report.nnz_b, n);
  const ShiftInvertShape s = shape(n, 6, fac[0].at("entries").get<double>(), static_cast<double>(n), false);
  EXPECT_EQ(sol.report.block_size, hs::eigen_detail::recommended_block_size(s));
}

// Spectrum slices keep the P4 width of the slice's nev (the rule was calibrated on nearest
// targets only), with the P4 reservation.
TEST(BlocktuneDriver, SlicesKeepTheP4Width) {
  std::vector<double> w;
  const auto a = shifted_laplacian<double>(w);
  for (const index_t k : {3, 12}) {
    hs::SolverConfig cfg = nearest_cfg(0.0, k);
    cfg.target.mode = hs::TargetMode::Smallest;
    const std::string path = ::testing::TempDir() + "/blocktune_driver_slices" + std::to_string(k) + ".jsonl";
    hs::Solution<double> sol;
    {
      hs::Logger log(0, path);
      sol = hs::solve(a, nullptr, cfg, &log);
    }
    EXPECT_TRUE(sol.report.success) << k;
    std::vector<double> want(w);
    std::sort(want.begin(), want.end());
    want.resize(static_cast<std::size_t>(k));
    expect_same_set(sol.eigenvalues, want, 1e-9);
    ASSERT_FALSE(sol.report.slices.empty());
    EXPECT_EQ(sol.report.block_size, hs::driver::block_size_for(cfg, sol.report.slices[0].nev)) << k;
    const auto plan = events(path, "memory_plan");
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].at("reserved_block_size").get<index_t>(), hs::driver::kMaxAutoBlockSize);
  }
}

}  // namespace
