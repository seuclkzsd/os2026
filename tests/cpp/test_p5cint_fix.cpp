// Regression tests of the P5c integration review fixes (src/eigen, src/driver):
//   * KrylovSchurOptions::lead_nev: a run whose extra wanted pairs lie in a cluster wider than
//     tol (they never converge) ends once its leading pairs have converged, and the driver's
//     deflated certificate reruns use it - they ask for missing + 2 pairs but end with the
//     missing ones (yee-pec-box-1e5: the third pair of a rerun, a curl-curl null-space vector,
//     held the rerun until max_restarts).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
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

}  // namespace
