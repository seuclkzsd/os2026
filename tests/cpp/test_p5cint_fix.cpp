// Regression tests of the P5c integration review fixes (src/eigen, src/driver):
//   * KrylovSchurOptions::lead_nev: a run whose extra wanted pairs lie in a cluster wider than
//     tol (they never converge) ends once its leading pairs have converged, and the driver's
//     deflated certificate reruns use it - they ask for missing + 2 pairs but end with the
//     missing ones (yee-pec-box-1e5: the third pair of a rerun, a curl-curl null-space vector,
//     held the rerun until max_restarts);
//   * a deflated rerun after an auto width takes at most the P4 width (rerun_block_size(cfg, ...));
//   * the memory plan reserves the P4 basis for nearest targets, so factors that fit beside it
//     stay in core (the p = 8 reservation sent them out of core or over the budget), and an auto
//     width beyond the reserved one is halved back without a warning.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
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

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
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

/// `count` values c (1 + width u), u evenly spread in [-1, 1]: a cluster of relative width
/// `width`, far below any gap a Krylov polynomial of a few hundred degrees resolves.
std::vector<double> cluster(double c, double width, int count) {
  std::vector<double> v;
  for (int i = 0; i < count; ++i) v.push_back(c * (1.0 + width * (2.0 * i / (count - 1) - 1.0)));
  return v;
}

// ------------------------------------------------------------------- lead_nev (src/eigen)

// theta = 10 (the leading pair), then 200 eigenvalues within 1e-8 relative of 3, then the rest
// in [-2, 2]. Wanted: the 3 largest |theta|, i.e. 10 and two members of the cluster. Ritz vectors
// inside the cluster are mixtures whose estimates stay near its width (~1e-8 |theta| >> tol), so
// without lead_nev the run holds on until max_restarts; with lead_nev = 1 it ends as soon as 10
// has converged and returns all three pairs (the cluster ones unconverged).
template <class T>
void extra_pairs_in_a_wide_cluster(index_t p) {
  std::vector<double> d{10.0};
  for (double x : cluster(3.0, 1e-8, 200)) d.push_back(x);
  for (int i = 0; i < 400; ++i) d.push_back(-2.0 + 4.0 * i / 399.0);
  hs::test::DiagonalOperator<T> op(std::vector<T>(d.begin(), d.end()));
  hs::KrylovSchurOptions<T> ko;
  ko.nev = 3;
  ko.tol = 1e-11;
  ko.max_restarts = 30;
  ko.block_size = p;
  const auto held = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_EQ(held.restarts, 30) << "p = " << p << ": the cluster pairs no longer stall?";
  EXPECT_FALSE(held.converged) << p;
  ASSERT_GE(held.nconv, 1) << p;
  EXPECT_NEAR(held.theta[0], 10.0, 1e-9) << p;

  ko.lead_nev = 1;
  const auto led = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_LE(led.restarts, 3) << p;
  EXPECT_LT(led.op_applies, held.op_applies / 5) << p;
  ASSERT_EQ(led.theta.size(), 3u) << p;  // all nev pairs returned, converged first
  ASSERT_GE(led.nconv, 1) << p;
  EXPECT_FALSE(led.converged) << p;  // contract unchanged: nconv < nev
  EXPECT_NEAR(led.theta[0], 10.0, 1e-9) << p;
  EXPECT_LE(led.estimate[0], 1e-11 * 10.0) << p;
  ASSERT_EQ(led.estimate.size(), 3u) << p;  // the other two as far as they got (no claim)
  // lead_nev = nev is the default behaviour.
  ko.lead_nev = 3;
  const auto all = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_EQ(all.restarts, held.restarts) << p;
  EXPECT_EQ(all.op_applies, held.op_applies) << p;
  EXPECT_EQ(all.theta, held.theta) << p;
}

TEST(P5cIntFixLead, ExtraPairsInAWideClusterDoNotHoldTheRunReal) {
  for (const index_t p : {1, 2, 4, 8}) extra_pairs_in_a_wide_cluster<double>(p);
}
TEST(P5cIntFixLead, ExtraPairsInAWideClusterDoNotHoldTheRunComplex) {
  for (const index_t p : {1, 4}) extra_pairs_in_a_wide_cluster<cplx>(p);
}

TEST(P5cIntFixLead, LeadNevIsValidated) {
  std::vector<double> d;
  for (int i = 0; i < 100; ++i) d.push_back(1.0 + i);
  hs::test::DiagonalOperator<double> op(d);
  hs::KrylovSchurOptions<double> ko;
  ko.nev = 4;
  for (const index_t lead : {index_t{-1}, index_t{5}}) {
    ko.lead_nev = lead;
    try {
      hs::lanczos_krylov_schur<double>(op, nullptr, hs::Which::LargestAlgebraic, ko);
      ADD_FAILURE() << "lead_nev " << lead << " accepted";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  }
  ko.lead_nev = 1;
  try {
    hs::krylov_schur_general<double>(op, hs::Which::LargestMagnitude, ko);
    ADD_FAILURE() << "krylov_schur_general accepted lead_nev";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
  }
}

// ------------------------------------------------- the driver's deflated reruns (definite.cpp)

// k = 4 nearest 0, a cluster of 150 eigenvalues within 1e-8 relative of 0.5 right beyond them.
// The fault injection removes the farthest found pair (0.25); the count finds it missing and the
// rerun asks for 3 pairs: 0.25 and two members of the cluster, which never converge to the inner
// tolerance. Before lead_nev the rerun ran into max_restarts (40 here) at every width.
template <class T>
void rerun_ends_with_the_missing_pairs(index_t p) {
  std::vector<double> w{-0.1, 0.12, -0.2, 0.25};
  for (double x : cluster(0.5, 1e-8, 150)) w.push_back(x);
  for (int i = 0; i < 300; ++i) w.push_back(1.0 + 3.0 * i / 299.0);
  for (int i = 0; i < 100; ++i) w.push_back(-1.0 - 3.0 * i / 99.0);
  const auto a = hs::test::prescribed_spectrum<T>(w, 3, 31);
  auto cfg = nearest_cfg(0.0, 4);
  cfg.block_size = p;
  cfg.max_restarts = 40;
  cfg.certificate_fault_drop = 1;
  const std::string path = ::testing::TempDir() + "/p5cint_fix_rerun_" + (hs::is_complex_v<T> ? "c" : "r") +
                           std::to_string(p) + ".jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, nullptr, cfg, &log);
  }
  const auto& r = sol.report;
  EXPECT_TRUE(r.success) << p << ": " << r.certificate.statement;
  EXPECT_TRUE(r.certificate.passed) << p;
  EXPECT_TRUE(has_warning(r, "certificate_fault_drop: 1 found pair(s) removed")) << p;
  EXPECT_TRUE(has_warning(r, "deflated rerun")) << p;
  expect_same_set(sol.eigenvalues, nearest(w, 0.0, 4), 1e-10);
  const auto runs = events(path, "lanczos");
  ASSERT_EQ(runs.size(), 2u) << p;  // the main run and one rerun
  const auto& rerun = runs[1];
  EXPECT_EQ(rerun.at("nev").get<index_t>(), 3) << p;
  EXPECT_EQ(rerun.at("lead").get<index_t>(), 1) << p;
  EXPECT_EQ(rerun.at("deflated").get<index_t>(), 3) << p;
  EXPECT_LT(rerun.at("restarts").get<index_t>(), 10) << p << ": " << rerun.dump();
  EXPECT_GE(rerun.at("nconv").get<index_t>(), 1) << p;
  EXPECT_EQ(runs[0].at("lead").get<index_t>(), 0) << p;  // the main run converges all k
}

TEST(P5cIntFixRerun, EndsWithTheMissingPairsReal) {
  for (const index_t p : {1, 4, 8}) rerun_ends_with_the_missing_pairs<double>(p);
}
TEST(P5cIntFixRerun, EndsWithTheMissingPairsComplex) { rerun_ends_with_the_missing_pairs<cplx>(4); }

// Width of a deflated rerun: an auto width enters capped at the P4 width, an explicit one as is.
TEST(P5cIntFixRerun, RerunWidthOfAnAutoRun) {
  using hs::driver::rerun_block_size;
  hs::SolverConfig cfg;
  ASSERT_EQ(cfg.block_size, 0);
  EXPECT_EQ(rerun_block_size(cfg, 8, 1), 4);
  EXPECT_EQ(rerun_block_size(cfg, 8, 2), 4);
  EXPECT_EQ(rerun_block_size(cfg, 8, 9), 4);
  EXPECT_EQ(rerun_block_size(cfg, 4, 1), 4);
  EXPECT_EQ(rerun_block_size(cfg, 2, 1), 2);
  EXPECT_EQ(rerun_block_size(cfg, 1, 3), 4);
  EXPECT_EQ(rerun_block_size(cfg, 1, 1), 2);
  cfg.block_size = 8;
  EXPECT_EQ(rerun_block_size(cfg, 8, 2), 8);
  EXPECT_EQ(rerun_block_size(cfg, 4, 2), 4);  // an explicit 8 halved by the memory fit
  EXPECT_EQ(rerun_block_size(8, 2), 8);       // the P4 function itself is unchanged
}

/// Dense real symmetric matrix H diag(w) H, H = I - 2 u u^T (u a fixed unit vector): exactly the
/// spectrum w (up to rounding), with the factor fill of a dense matrix - the blocktune rule picks
/// p = 8 for k = 12 on it.
hs::CsrMatrix<double> dense_with_spectrum(const std::vector<double>& w) {
  const auto n = static_cast<index_t>(w.size());
  std::vector<double> u(static_cast<std::size_t>(n));
  double norm = 0.0;
  for (index_t i = 0; i < n; ++i) {
    u[i] = std::sin(0.7 * static_cast<double>(i) + 0.3) + 0.1;
    norm += u[i] * u[i];
  }
  for (auto& x : u) x /= std::sqrt(norm);
  double s = 0.0;
  for (index_t i = 0; i < n; ++i) s += w[i] * u[i] * u[i];
  std::vector<index_t> rows, cols;
  std::vector<double> vals;
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j) {
      const double v = (i == j ? w[i] : 0.0) - 2.0 * u[i] * u[j] * (w[i] + w[j]) + 4.0 * u[i] * u[j] * s;
      rows.push_back(i);
      cols.push_back(j);
      vals.push_back(v);
    }
  return hs::csr_from_coo<double>(n, n, rows, cols, vals);
}

// End to end: the rule picks 8 for the main run; the rerun after the fault injection runs at the
// P4 width 4 (before: 8, inherited through max(p, ...)), an explicit 8 keeps 8.
TEST(P5cIntFixRerun, AutoWidthRerunTakesTheP4Width) {
  std::vector<double> w;
  for (int j = 0; j < 12; ++j) w.push_back((j % 2 ? 1.0 : -1.0) * (0.05 + 0.02 * j));
  for (int i = 0; i < 700; ++i) w.push_back(1.0 + 4.0 * i / 699.0);
  for (int i = 0; i < 300; ++i) w.push_back(-1.0 - 4.0 * i / 299.0);
  const auto a = dense_with_spectrum(w);
  for (const index_t configured : {0, 8}) {
    auto cfg = nearest_cfg(0.0, 12);
    cfg.block_size = configured;
    cfg.certificate_fault_drop = 1;
    const std::string path = ::testing::TempDir() + "/p5cint_fix_width" + std::to_string(configured) + ".jsonl";
    hs::Solution<double> sol;
    {
      hs::Logger log(0, path);
      sol = hs::solve(a, nullptr, cfg, &log);
    }
    const auto& r = sol.report;
    EXPECT_TRUE(r.success) << configured << ": " << r.certificate.statement;
    ASSERT_EQ(r.block_size, 8) << configured << ": the rule no longer picks 8 here; the test needs a shape where it does";
    expect_same_set(sol.eigenvalues, nearest(w, 0.0, 12), 1e-9);
    const auto runs = events(path, "lanczos");
    ASSERT_EQ(runs.size(), 2u) << configured;
    EXPECT_EQ(runs[0].at("block_size").get<index_t>(), 8) << configured;
    EXPECT_EQ(runs[1].at("block_size").get<index_t>(), configured == 0 ? 4 : 8) << configured;
  }
}

// --------------------------------------------------------------- the memory plan (solver.cpp)

// A budget whose factorization cap holds the factors beside the P4 basis (by half the 27
// n-vectors the p = 8 basis adds at k = 10) with out_of_core = off: the run stays in core and
// succeeds. Reserving the p = 8 basis for the auto width (P5c integration) put the cap that far
// below the factors' estimate: OutOfMemoryBudget (with out_of_core = auto: out-of-core factors).
// The factors (2-D, AMD) are larger than the 27 vectors, so the p = 8 basis alone still fit the
// budget - the case the integration's fallback to the P4 plan did not catch. The resident memory
// before the plan moves by a few MB from solve to solve in one process, so the budget follows the
// previous run's (glibc keeps freed factor memory resident after the first solves) and the
// window is checked on the run's own plan (memory_plan event).
TEST(P5cIntFixMemory, FactorsThatFitBesideTheP4BasisStayInCore) {
  const index_t nx = 300, ny = 200, n = nx * ny;
  auto a = hs::test::laplacian_2d<double>(nx, ny);
  for (index_t i = 0; i < n; ++i)
    for (index_t q = a.rowptr[i]; q < a.rowptr[i + 1]; ++q)
      if (a.colind[q] == i) a.values[q] -= 4.0;
  auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  for (auto& x : w) x -= 4.0;
  // Just below the spectrum (-3.99965): K(sigma) is definite (no delayed pivots, so the
  // factorization needs no more than its estimate plus MUMPS' usual margin, factor.hpp), and the
  // 10 nearest are well separated relative to their distance.
  const double sigma = -4.0;
  auto cfg = nearest_cfg(sigma, 10);
  cfg.out_of_core = hs::Toggle::Off;
  cfg.ordering = 0;  // AMD: one analysis, the same estimate in every run
  const double gib = 1073741824.0;
  struct Plan {
    double rss = 0.0, krylov_mb = 0.0, cap_mb = 0.0, incore_mb = 0.0;
    index_t reserved = 0;
    bool success = false;
    std::string error;
  };
  hs::Solution<double> sol;
  auto probe = [&](double budget_gb, const std::string& tag) {
    const std::string path = ::testing::TempDir() + "/p5cint_fix_mem_" + tag + ".jsonl";
    auto c = cfg;
    c.memory_budget_gb = budget_gb;
    Plan p;
    try {
      hs::Logger log(0, path);
      sol = hs::solve(a, nullptr, c, &log);
      p.success = sol.report.success;
    } catch (const hs::Error& e) {
      p.error = e.what();
    }
    const auto plan = events(path, "memory_plan");
    const auto analysis = events(path, "analysis");
    if (plan.empty()) return p;
    p.krylov_mb = plan[0].at("krylov_mb").get<double>();
    p.cap_mb = plan[0].at("cap_mb").get<double>();
    p.reserved = plan[0].at("reserved_block_size").get<index_t>();
    p.rss = budget_gb * gib - p.cap_mb * 1e6 / 0.95 - p.krylov_mb * 1e6;
    if (!analysis.empty()) p.incore_mb = analysis[0].at("incore_mb").get<double>();
    return p;
  };
  Plan last = probe(4.0, "ref");
  ASSERT_TRUE(last.success) << last.error;
  ASSERT_EQ(last.reserved, hs::driver::kMaxAutoBlockSize);
  const double incore = last.incore_mb;
  // What the p = 8 basis would add to the reservation at k = 10 (krylov_reservation's definite
  // columns: 27 n-vectors).
  const auto definite = [&](index_t p) {
    return std::max<index_t>(2 * (10 + 16) + 1, hs::driver::lanczos_basis_columns(n, 0, 10, 0, p)) + 6 * 10;
  };
  ASSERT_EQ(definite(8) - definite(4), 27);
  const double extra_mb = static_cast<double>(definite(8) - definite(4)) * static_cast<double>(n) * 8.0 / 1e6;
  const double headroom_mb = 0.5 * 0.95 * extra_mb;  // 6.2 MB both ways
  ASSERT_GT(incore, extra_mb) << "the p = 8 basis alone must fit the budget below";
  bool in_window = false;
  Plan tight;
  for (int attempt = 0; attempt < 6 && !in_window; ++attempt) {
    const double budget_gb = (last.rss + last.krylov_mb * 1e6 + (incore + headroom_mb) * 1e6 / 0.95) / gib;
    tight = probe(budget_gb, "tight" + std::to_string(attempt));
    ASSERT_EQ(tight.reserved, hs::driver::kMaxAutoBlockSize) << tight.error;
    // This run's cap holds the factors with a margin; the p = 8 reservation would not have.
    in_window = tight.cap_mb >= incore + 0.25 * headroom_mb && tight.cap_mb - 0.95 * extra_mb < incore;
    last = tight;
  }
  ASSERT_TRUE(in_window) << "cap " << tight.cap_mb << " MB, factors " << incore << " MB";
  EXPECT_TRUE(tight.success) << tight.error << " (factors " << incore << " MB, cap " << tight.cap_mb << " MB)";
  if (tight.success) {
    EXPECT_FALSE(has_warning(sol.report, "out-of-core"));
    expect_same_set(sol.eigenvalues, nearest(w, sigma, 10), 1e-9);
  }
}

// An auto width beyond the reserved one is halved back to it without a warning (event
// block_size_fit); below the reserved width, or for an explicit width, the reduction warns.
TEST(P5cIntFixMemory, AutoWidthFitsBackToTheReservedOneQuietly) {
  hs::SolverConfig cfg;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  const std::string path = ::testing::TempDir() + "/p5cint_fix_fit.jsonl";
  hs::Logger log(0, path);  // flushed per event
  hs::driver::Ctx c{cfg, &log, rep, clock};
  const index_t n = 2000000, nev = 12;
  const auto cols = [&](index_t p) {
    return static_cast<double>(hs::driver::lanczos_basis_columns(n, 0, nev, 0, p) + 4 * nev);
  };
  const double vec = 8.0 * static_cast<double>(n);
  const double rss = static_cast<double>(std::max<std::int64_t>(hs::current_rss_bytes(), 0));
  // Room for the p = 4 basis but not the p = 8 one.
  cfg.memory_budget_gb = (rss + 0.5 * (cols(4) + cols(8)) * vec / 0.9) / 1073741824.0;
  EXPECT_EQ(hs::driver::fit_block_size<double>(c, n, nev, 0, 4 * nev, 8, 4), 4);
  EXPECT_TRUE(rep.warnings.empty());
  EXPECT_EQ(hs::driver::fit_block_size<double>(c, n, nev, 0, 4 * nev, 8), 4);  // explicit: warns
  ASSERT_EQ(rep.warnings.size(), 1u);
  EXPECT_NE(rep.warnings[0].find("block size reduced from 8 to 4"), std::string::npos);
  // Room for p = 2 only: below the reserved width the auto width warns too.
  cfg.memory_budget_gb = (rss + 0.5 * (cols(2) + cols(4)) * vec / 0.9) / 1073741824.0;
  EXPECT_EQ(hs::driver::fit_block_size<double>(c, n, nev, 0, 4 * nev, 8, 4), 2);
  ASSERT_EQ(rep.warnings.size(), 2u);
  EXPECT_NE(rep.warnings[1].find("block size reduced from 8 to 2"), std::string::npos);
  EXPECT_EQ(events(path, "block_size_fit").size(), 1u);
}

}  // namespace
