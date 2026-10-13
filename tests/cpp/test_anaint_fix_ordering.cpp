// The analysis loop of the adaptive ordering never chooses a candidate whose analysis it released
// (module anaint, P5c fix; src/driver/steps.cpp run_ordering_choice). Regression: the loop kept the
// best of ALL analysed candidates by choose_ordering, which is not monotone - AMD 100, METIS 90,
// PORD 88 keeps METIS and releases PORD, and once ND joins with 94 PORD wins again - and
// adaptive_analysis then asserted on the released analysis (Internal error from hs::solve).
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/factor.hpp"

namespace {

using hs::kOrderingNestedDissection;
using hs::driver::OrderingCandidate;

constexpr int kScotch = hs::driver::kOrderingScotch;

/// A scripted loop: actual scores by candidate index, predicted scores per anchor (the prices the
/// candidates not analysed yet get from predict(anchor)).
struct Script {
  std::vector<double> actual;
  std::map<std::size_t, std::vector<double>> predicted;  ///< anchor -> scores of every candidate
};

struct Trace {
  std::size_t pick = 0;
  std::vector<std::size_t> analysed, released;
};

Trace play(const Script& s) {
  std::vector<OrderingCandidate> cand(5);
  const std::vector<int> orderings = hs::driver::candidate_orderings(hs::SolverConfig{});
  for (std::size_t i = 0; i < cand.size(); ++i) cand[i].ordering = orderings[i];
  Trace t;
  auto analyse = [&](std::size_t i) {
    EXPECT_FALSE(cand[i].analysed) << i;
    cand[i].analysed = true;
    cand[i].score_s = s.actual[i];
    t.analysed.push_back(i);
  };
  auto predict = [&](std::size_t anchor) {
    const auto it = s.predicted.find(anchor);
    ASSERT_NE(it, s.predicted.end()) << anchor;
    for (std::size_t i = 0; i < cand.size(); ++i)
      if (!cand[i].analysed) cand[i].score_s = it->second[i];
  };
  auto release = [&](std::size_t i) { t.released.push_back(i); };
  t.pick = hs::driver::run_ordering_choice(cand, analyse, predict, release);
  return t;
}

bool contains(const std::vector<std::size_t>& v, std::size_t x) {
  for (const std::size_t y : v)
    if (y == x) return true;
  return false;
}

// Candidates in order: AMD (0), SCOTCH (1), hsolve's ND (2), METIS (3), PORD (4). The reviewer's
// sequence: AMD 100 -> METIS 90 -> PORD 88 -> ND 94.
TEST(AnaintFixOrdering, ReleasedCandidateIsNeverChosen) {
  Script s;
  s.actual = {100.0, 0.0, 94.0, 90.0, 88.0};
  s.predicted[0] = {0, 1000.0, 200.0, 80.0, 120.0};  // METIS next
  s.predicted[3] = {0, 1000.0, 150.0, 0, 70.0};      // then PORD
  s.predicted[4] = {0, 1000.0, 80.0, 0, 0};          // then ND
  s.predicted[2] = {0, 1000.0, 0, 0, 0};
  const Trace t = play(s);
  ASSERT_EQ(t.analysed, (std::vector<std::size_t>{0, 3, 4, 2}));
  // The old rule would take PORD (choose_ordering over {AMD 100, ND 94, METIS 90, PORD 88}),
  // released after it lost to METIS. Pairwise: METIS beats AMD, PORD does not beat METIS by the
  // margin, ND (earlier) keeps its place against METIS (90 > 0.95 * 94).
  EXPECT_EQ(t.pick, 2u);
  EXPECT_FALSE(contains(t.released, t.pick));
  EXPECT_EQ(t.released, (std::vector<std::size_t>{0, 4, 3}));  // every loser, once
}

// SCOTCH is taken on its prediction once analysed: the kept candidate is released.
TEST(AnaintFixOrdering, ScotchEndsTheLoop) {
  Script s;
  s.actual = {100.0, 70.0, 0, 0, 0};
  s.predicted[0] = {0, 60.0, 90.0, 95.0, 99.0};
  const Trace t = play(s);
  EXPECT_EQ(t.pick, 1u);
  EXPECT_EQ(t.analysed, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(t.released, (std::vector<std::size_t>{0}));
}

// Random scores: whatever the sequence, the choice is analysed, never released, every other
// analysed candidate is released exactly once, and the same script gives the same choice.
TEST(AnaintFixOrdering, RandomScriptsKeepTheChoiceAlive) {
  std::uint64_t state = 2026;
  auto uniform = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0;
  };
  for (int trial = 0; trial < 20000; ++trial) {
    Script s;
    for (int i = 0; i < 5; ++i) s.actual.push_back(80.0 + 40.0 * uniform());
    for (std::size_t a = 0; a < 5; ++a) {
      std::vector<double> p;
      for (int i = 0; i < 5; ++i) p.push_back(60.0 + 60.0 * uniform());
      s.predicted[a] = p;
    }
    const Trace t = play(s);
    ASSERT_TRUE(contains(t.analysed, t.pick)) << trial;
    ASSERT_FALSE(contains(t.released, t.pick)) << trial;
    const std::set<std::size_t> rel(t.released.begin(), t.released.end());
    ASSERT_EQ(rel.size(), t.released.size()) << trial;           // each once
    ASSERT_EQ(t.released.size() + 1, t.analysed.size()) << trial;  // all others
    EXPECT_EQ(play(s).pick, t.pick) << trial;
  }
}

}  // namespace
