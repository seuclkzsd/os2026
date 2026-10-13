// The B proof's memory and time limits hold before anything is allocated or built (module anaint,
// P5c fix; src/driver/norms.cpp prove_positive_definite). Regression: a B with one dense column (an
// arrowhead) has a fill-level-1 pattern of n^2 / 2 entries, which used to be built before the
// limits were consulted - 4.3 s and 4.5 GiB at n = 2e4 against a 0.005 s budget and a 1 GiB
// memory budget in hs::solve; and S and the level-1 pattern were allocated whatever max_bytes said.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// b_ii = 1, b_0j = b_j0 = 0.632 / sqrt(n - 1): eigenvalues 1 +- 0.632 and 1 (HPD), not
/// diagonally dominant in row 0, lambda_min of the scaled matrix 0.368 (above the theta gate).
template <class T>
hs::CsrMatrix<T> arrowhead(index_t n) {
  const double c = 0.632 / std::sqrt(static_cast<double>(n - 1));
  std::vector<index_t> r, cc;
  std::vector<T> v;
  for (index_t i = 0; i < n; ++i) r.push_back(i), cc.push_back(i), v.push_back(T(1.0));
  for (index_t j = 1; j < n; ++j) {
    r.push_back(0), cc.push_back(j), v.push_back(T(c));
    r.push_back(j), cc.push_back(0), v.push_back(T(c));
  }
  return hs::csr_from_coo<T>(n, n, r, cc, v);
}

/// Consistent P1 mass matrix of an nx x ny node grid (as in test_anaint_bproof.cpp): HPD, not
/// diagonally dominant; level 0 and level 1 both prove it.
template <class T>
hs::CsrMatrix<T> p1_mass(index_t nx, index_t ny) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [&](index_t i, index_t j) { return i + nx * j; };
  auto element = [&](index_t a, index_t b, index_t d) {
    const index_t nodes[3] = {a, b, d};
    for (const index_t p : nodes)
      for (const index_t q : nodes) {
        r.push_back(p);
        c.push_back(q);
        v.push_back(T((p == q ? 2.0 : 1.0) / 24.0));
      }
  };
  for (index_t j = 0; j + 1 < ny; ++j)
    for (index_t i = 0; i + 1 < nx; ++i) {
      element(id(i, j), id(i + 1, j), id(i + 1, j + 1));
      element(id(i, j), id(i + 1, j + 1), id(i, j + 1));
    }
  const index_t n = nx * ny;
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

/// Strict lower entries of b.
template <class T>
double strict_lower(const hs::CsrMatrix<T>& b) {
  double s = 0;
  for (index_t i = 0; i < b.nrows; ++i)
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] < i) ++s;
  return s;
}

// The O(n) bounds: exact for an arrowhead (its level-1 pattern is the full lower triangle), above
// the built pattern of a mass matrix.
TEST(AnaintFixBProof, LevelOneBounds) {
  const index_t n = 400;
  const auto a = hs::driver::level1_pattern_size(arrowhead<double>(n), true);
  const double full = 0.5 * static_cast<double>(n) * static_cast<double>(n + 1);
  EXPECT_EQ(a.entries_bound, full);
  EXPECT_EQ(static_cast<double>(a.entries), full);
  EXPECT_EQ(a.visits, 0.5 * static_cast<double>(n - 1) * static_cast<double>(n));  // column 0: c = n - 1
  const auto m = hs::driver::level1_pattern_size(p1_mass<cplx>(30, 20), true);
  EXPECT_GT(m.entries, 0);
  EXPECT_LE(static_cast<double>(m.entries), m.entries_bound);
  const auto unbuilt = hs::driver::level1_pattern_size(p1_mass<cplx>(30, 20), false);
  EXPECT_EQ(unbuilt.entries, -1);
  EXPECT_EQ(unbuilt.entries_bound, m.entries_bound);
}

// The reviewer's case: n = 2e4, a 10 ms budget. Neither the level-1 pattern (2e8 entries) nor a
// level-0 attempt (its check scans n^2 pairs) fits: no attempt, and the pattern is not even
// counted.
TEST(AnaintFixBProof, ArrowheadPatternIsPricedBeforeItIsBuilt) {
  const auto b = arrowhead<double>(20000);
  hs::driver::HpdProofOptions o;
  o.max_bytes = 1e9;
  o.max_seconds = 0.01;
  const auto p = hs::driver::prove_positive_definite(b, o);
  EXPECT_FALSE(p.proven);
  EXPECT_EQ(p.outcome, "too_expensive");
  EXPECT_EQ(p.attempts, 0);
  EXPECT_EQ(p.level1_entries, -1);  // not even counted
  EXPECT_EQ(p.pattern_s, 0.0);
  EXPECT_GT(p.pattern_predicted_s, o.max_seconds);
  EXPECT_EQ(p.level1_bound, 0.5 * 20000.0 * 20001.0);
}

// max_bytes: S is not stored when it does not fit with the smallest attempt; the level-1 pattern
// is counted, but not allocated, when it does not fit - level 0 then proves it.
template <class T>
void memory_limits() {
  const auto b = p1_mass<T>(40, 30);
  const index_t n = b.nrows;
  const double s_entries = strict_lower(b) + static_cast<double>(n);
  const double stored = hs::driver::hpd_stored_bytes<T>(n, s_entries);
  const double level0 = hs::driver::hpd_attempt_bytes<T>(n, 0, s_entries);
  hs::driver::HpdProofOptions o;
  o.max_bytes = stored + 0.5 * level0;
  auto p = hs::driver::prove_positive_definite(b, o);
  EXPECT_EQ(p.outcome, "too_large");
  EXPECT_EQ(p.attempts, 0);
  EXPECT_EQ(p.stored_bytes, 0.0);  // S not stored
  EXPECT_EQ(p.setup_s, 0.0);
  // Between level 0 and level 1: level 1 counted (exactly) but not built, level 0 proves it.
  const auto size = hs::driver::level1_pattern_size(b, true);
  const double level1 = hs::driver::hpd_attempt_bytes<T>(n, 1, static_cast<double>(size.entries));
  ASSERT_GT(level1, level0);
  o.max_bytes = stored + 0.5 * (level0 + level1);
  p = hs::driver::prove_positive_definite(b, o);
  EXPECT_TRUE(p.proven) << p.outcome;
  EXPECT_EQ(p.level, 0);
  EXPECT_EQ(p.level1_entries, size.entries);
  EXPECT_EQ(p.stored_bytes, stored);
  // Enough for level 1.
  o.max_bytes = stored + level1;
  p = hs::driver::prove_positive_definite(b, o);
  EXPECT_TRUE(p.proven) << p.outcome;
  EXPECT_EQ(p.level, 1);
}

TEST(AnaintFixBProof, MemoryLimitsHoldBeforeAllocationReal) { memory_limits<double>(); }
TEST(AnaintFixBProof, MemoryLimitsHoldBeforeAllocationComplex) { memory_limits<cplx>(); }

// The time budget pays for the level-1 pattern too: a budget below the pattern's predicted
// construction leaves level 1 unbuilt (level 0 is then tried with what is left).
TEST(AnaintFixBProof, PatternIsChargedToTheBudget) {
  const auto b = p1_mass<double>(60, 50);
  const auto full = hs::driver::prove_positive_definite(b);
  ASSERT_TRUE(full.proven);
  ASSERT_EQ(full.level, 1);
  ASSERT_GT(full.pattern_predicted_s, 0.0);
  ASSERT_GT(full.level1_entries, 0);
  hs::driver::HpdProofOptions o;
  o.max_seconds = 0.5 * full.pattern_predicted_s;
  const auto p = hs::driver::prove_positive_definite(b, o);
  EXPECT_EQ(p.level1_entries, -1);
  EXPECT_EQ(p.pattern_s, 0.0);
  EXPECT_NE(p.level, 1);
}

/// A = diag(1 + i/n) plus a weak tridiagonal coupling, B the arrowhead: GHEP, Auto type.
hs::CsrMatrix<double> weak_tridiagonal(index_t n) {
  std::vector<index_t> r, c;
  std::vector<double> v;
  for (index_t i = 0; i < n; ++i) r.push_back(i), c.push_back(i), v.push_back(1.0 + static_cast<double>(i) / n);
  for (index_t i = 0; i + 1 < n; ++i) {
    r.push_back(i), c.push_back(i + 1), v.push_back(-0.01);
    r.push_back(i + 1), c.push_back(i), v.push_back(-0.01);
  }
  return hs::csr_from_coo<double>(n, n, r, c, v);
}

// hs::solve with a 1 GiB budget: the proof gives up without building the pattern, B is factorized
// for its inertia, the solve succeeds.
TEST(AnaintFixBProof, SolveWithArrowheadBDoesNotBuildThePattern) {
  const index_t n = 20000;
  const auto a = weak_tridiagonal(n);
  const auto b = arrowhead<double>(n);
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = 1.5;
  cfg.target.k = 6;
  cfg.threads = 2;
  cfg.memory_budget_gb = 1.0;
  const std::string path = ::testing::TempDir() + "/anaint_fix_bproof_arrow.jsonl";
  hs::Solution<double> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, &b, cfg, &log);
  }
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  std::ifstream in(path);
  std::string line;
  int proofs = 0;
  while (std::getline(in, line)) {
    const auto j = nlohmann::json::parse(line);
    if (j.value("event", "") == "b_proof") {
      ++proofs;
      EXPECT_FALSE(j.at("proven").get<bool>());
      EXPECT_EQ(j.at("level1_entries").get<std::int64_t>(), -1);
      EXPECT_EQ(j.at("pattern_s").get<double>(), 0.0);
      EXPECT_LE(j.at("max_bytes").get<double>(), 1073741824.0);
    }
    if (j.value("event", "") == "b_inertia") {
      EXPECT_EQ(j.at("how").get<std::string>(), "factorization");
    }
  }
  EXPECT_EQ(proofs, 1);
}

}  // namespace
