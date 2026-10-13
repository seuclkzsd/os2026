// Tests of the block-size rule of shift-invert block Lanczos (src/eigen/block_rule): the measured
// step table, the choice from given step costs (margin, layout feasibility), the modelled step
// costs, and the rule's picks on the shapes of the calibration cases (pinned: a change of the
// model must be re-measured, bench/blocktune_sweep.cpp).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "eigen/block_lanczos.hpp"
#include "eigen/block_rule.hpp"
#include "hsolve/error.hpp"

namespace {

using hs::index_t;
using hs::eigen_detail::kRuleMargin;
using hs::eigen_detail::kRuleWidthCount;
using hs::eigen_detail::kRuleWidths;
using hs::eigen_detail::modelled_step_seconds;
using hs::eigen_detail::recommended_block_size;
using hs::eigen_detail::relative_block_steps;
using hs::eigen_detail::ShiftInvertShape;
using hs::eigen_detail::WidthTable;

/// Runs f and expects hs::Error with code InvalidInput.
template <class F>
void expect_invalid(F&& f, const std::string& what) {
  try {
    f();
    ADD_FAILURE() << what << ": no exception";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << what << ": " << e.what();
  }
}

TEST(LanczosBlockRule, RelativeStepsStartAtOneAndNeverIncreaseWithTheWidth) {
  for (index_t nev : {1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 15, 21, 22, 23, 40, 1000}) {
    SCOPED_TRACE("nev " + std::to_string(nev));
    EXPECT_EQ(relative_block_steps(1, nev), 1.0);
    double prev = 1.0;
    for (index_t p = 2; p <= 40; ++p) {
      const double r = relative_block_steps(p, nev);
      EXPECT_GT(r, 0.0);
      EXPECT_LE(r, prev + 1e-15) << "p " << p;
      prev = r;
    }
  }
  // Measured down to nev = 1: blocks need fewer steps even for a single wanted pair.
  for (index_t p : {2, 4, 8, 16, 100}) EXPECT_LT(relative_block_steps(p, 1), 0.75);
  // Fewer wanted pairs gain no more from a given width (the measured rows are ordered so).
  for (index_t p : {2, 4, 8}) {
    double prev = 1.0;
    for (index_t nev = 1; nev <= 30; ++nev) {
      const double r = relative_block_steps(p, nev);
      EXPECT_LE(r, prev + 1e-15) << "p " << p << " nev " << nev;
      prev = r;
    }
  }
}

TEST(LanczosBlockRule, RelativeStepsReproduceTheMeasuredRows) {
  EXPECT_NEAR(relative_block_steps(2, 1), 0.69, 1e-12);  // nev = 1..5 pooled
  EXPECT_NEAR(relative_block_steps(4, 3), 0.56, 1e-12);
  EXPECT_NEAR(relative_block_steps(8, 5), 0.49, 1e-12);
  EXPECT_NEAR(relative_block_steps(4, 7), 0.51, 1e-12);
  EXPECT_NEAR(relative_block_steps(4, 11), 0.50, 1e-12);
  EXPECT_NEAR(relative_block_steps(8, 11), 0.38, 1e-12);
  EXPECT_NEAR(relative_block_steps(16, 22), 0.23, 1e-12);
  EXPECT_NEAR(relative_block_steps(8, 500), 0.29, 1e-12);  // clamped to the last row
  // Between rows: linear in nev; between widths: geometric in p; beyond 16: the p = 16 value.
  EXPECT_NEAR(relative_block_steps(2, 6), 0.5 * (0.69 + 0.65), 1e-12);
  EXPECT_NEAR(relative_block_steps(8, 9), 0.5 * (0.44 + 0.38), 1e-12);
  EXPECT_NEAR(relative_block_steps(2, 16), 0.65 - (5.0 / 11.0) * 0.07, 1e-12);
  const double l3 = std::log2(3.0);
  EXPECT_NEAR(relative_block_steps(3, 11),
              std::exp((2.0 - l3) * std::log(0.65) + (l3 - 1.0) * std::log(0.50)), 1e-12);
  EXPECT_NEAR(relative_block_steps(64, 11), 0.31, 1e-12);
}

TEST(LanczosBlockRule, ChoiceFollowsTheCostTimesSteps) {
  // nev below kRuleFloorMinNev: the plain minimization (the floor: test_blocktune_fix_rule.cpp).
  const index_t n = 100000, nev = 7;
  // Equal step costs: the widest width (fewest steps).
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{1, 1, 1, 1}), 8);
  // Cost proportional to the width: blocks never pay.
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{1, 2, 4, 8}), 1);
  // p = 2 costs 0.65 c(2) against c(1): it wins only with the margin.
  const double edge = (1.0 - kRuleMargin) / relative_block_steps(2, nev);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{1, edge * 1.001, 100, 100}), 1);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{1, edge * 0.999, 100, 100}), 2);
  // A wider width must beat the best narrower one (not only p = 1) by the margin.
  const double c2 = 1.0;  // relative cost 0.65
  const double c4_tie =
      relative_block_steps(2, nev) / relative_block_steps(4, nev) * (1.0 - kRuleMargin);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{2, c2, c4_tie * 1.001, 100}), 2);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{2, c2, c4_tie * 0.999, 100}), 4);
  // Beyond the floor width the step needs the wider margin: 8 against 4.
  const double c8_tie =
      relative_block_steps(4, nev) / relative_block_steps(8, nev) * (1.0 - hs::eigen_detail::kRuleWideMargin);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{100, 100, 1.0, c8_tie * 1.001}), 4);
  EXPECT_EQ(recommended_block_size(n, nev, WidthTable{100, 100, 1.0, c8_tie * 0.999}), 8);
  // ... which kRuleMargin alone would have passed.
  const double c8_narrow = relative_block_steps(4, nev) / relative_block_steps(8, nev) * (1.0 - kRuleMargin);
  ASSERT_LT(c8_tie * 1.001, c8_narrow);
  // Any unit: scaling all costs changes nothing.
  for (double s : {1e-9, 1.0, 1e9}) {
    EXPECT_EQ(recommended_block_size(n, nev, WidthTable{1.2 * s, 1.0 * s, 1.05 * s, 1.4 * s}), 4);
  }
}

TEST(LanczosBlockRule, WidthsWhoseLayoutDoesNotFitAreSkipped) {
  // The block layout needs kFreeSpaceFactor * (ncv + p) <= n (block_dimensions, auto ncv).
  using hs::eigen_detail::block_dimensions;
  using hs::eigen_detail::LanczosSetup;
  const WidthTable equal{1, 1, 1, 1};
  for (index_t n : {10, 40, 60, 80, 100, 150, 200, 400}) {
    for (index_t nev : {1, 4, 10}) {
      if (nev > n) continue;
      SCOPED_TRACE("n " + std::to_string(n) + " nev " + std::to_string(nev));
      const index_t p = recommended_block_size(n, nev, equal);
      LanczosSetup s;
      s.n = s.n_free = n;
      s.nev = nev;
      s.ncv = std::min(n, std::max(2 * nev + 1, nev + 16));
      if (p > 1) {
        EXPECT_TRUE(block_dimensions(s, p, true).feasible);
      }
      for (std::size_t i = 0; i < kRuleWidthCount; ++i) {
        if (kRuleWidths[i] > p && relative_block_steps(kRuleWidths[i], nev) < 1.0) {
          EXPECT_FALSE(block_dimensions(s, kRuleWidths[i], true).feasible) << kRuleWidths[i];
        }
      }
    }
  }
  EXPECT_EQ(recommended_block_size(10, 4, equal), 1);  // no block layout fits
}

TEST(LanczosBlockRule, InvalidInputsThrow) {
  expect_invalid([&] { relative_block_steps(0, 5); }, "relative_block_steps(0, 5)");
  expect_invalid([&] { relative_block_steps(2, 0); }, "relative_block_steps(2, 0)");
  const WidthTable ok{1, 1, 1, 1};
  expect_invalid([&] { recommended_block_size(5, 10, ok); },
                 "recommended_block_size(5, 10, ok)");  // n < nev
  expect_invalid([&] { recommended_block_size(100, 0, ok); }, "recommended_block_size(100, 0, ok)");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double bad : {0.0, -1.0, nan, inf}) {
    WidthTable c = ok;
    c[2] = bad;
    expect_invalid([&] { recommended_block_size(100, 10, c); },
                   "recommended_block_size(100, 10, c)");
  }
  ShiftInvertShape s;
  s.n = 1000;
  s.nev = 10;
  s.factor_entries = 1e5;
  EXPECT_NO_THROW(modelled_step_seconds(s));
  for (auto mutate :
       {+[](ShiftInvertShape& x) { x.n = 0; }, +[](ShiftInvertShape& x) { x.nev = 0; },
        +[](ShiftInvertShape& x) { x.factor_entries = -1.0; },
        +[](ShiftInvertShape& x) { x.factor_entries = std::nan(""); },
        +[](ShiftInvertShape& x) { x.nnz_b = -1.0; },
        +[](ShiftInvertShape& x) { x.nnz_b = HUGE_VAL; }}) {
    ShiftInvertShape t = s;
    mutate(t);
    expect_invalid([&] { modelled_step_seconds(t); }, "modelled_step_seconds(t)");
    expect_invalid([&] { recommended_block_size(t); }, "recommended_block_size(t)");
  }
}

TEST(LanczosBlockRule, ModelledStepCostsArePositiveAndGrowWithTheWidth) {
  for (double fill : {1.0, 3.0, 24.0, 40.0, 200.0, 500.0, 1500.0}) {
    for (bool cx : {false, true}) {
      for (double nnz_b : {0.0, 5e5}) {
        ShiftInvertShape s;
        s.n = 100000;
        s.nev = 10;
        s.factor_entries = fill * static_cast<double>(s.n) * (cx ? 2.0 : 1.0);
        s.nnz_b = nnz_b;
        s.complex_scalars = cx;
        const WidthTable t = modelled_step_seconds(s);
        for (std::size_t i = 0; i < kRuleWidthCount; ++i) {
          EXPECT_GT(t[i], 0.0);
          EXPECT_TRUE(std::isfinite(t[i]));
          if (i >= 2) {
            EXPECT_GT(t[i], t[i - 1]);  // p >= 2: a + b p + o(p) grows
          }
        }
        // Per column, wider blocks are cheaper to apply.
        EXPECT_LT(t[3] / 8.0, t[0]);
      }
    }
  }
  // Zero factor entries (degenerate input) stay finite: the factor holds at least its diagonal.
  ShiftInvertShape z;
  z.n = 50;
  z.nev = 2;
  for (double c : modelled_step_seconds(z)) EXPECT_TRUE(std::isfinite(c) && c > 0.0);
}

/// Shape of a calibration case (bench/blocktune_sweep.cpp "case" records: n, k, factor entries,
/// nnz(B), scalar type) and the width the rule picks for it.
struct Pinned {
  const char* name;
  index_t n, nev;
  double entries, nnz_b;
  bool cx;
  index_t p;
};

TEST(LanczosBlockRule, PicksOnTheCalibrationShapesArePinned) {
  // Iteration time of the pick vs p = 1 and vs the best width, 16 threads (block_rule.cpp):
  // 3-D: p = 4 / 8 (8 only for a modelled gain of kRuleWideMargin over 4: ss-2cubes_sphere,
  // syn-lap3d-aniso-1e5 and yee-pec-box-1e5 took 8 before the BLAS hook measurements); little fill
  // per row (2-D meshes, random sparse patterns, complex): p = 2 for nev < 8, the floor p = 4 from
  // nev = 8 on.
  const Pinned cases[] = {
      {"cav2d-rect-pert-1e5", 99459, 6, 3.871e6, 495751, false, 2},
      {"cav2d-square-1e4", 9976, 7, 3.418e5, 34984, false, 2},
      {"ss-2cubes_sphere", 101492, 10, 4.422e7, 0, false, 4},
      {"ss-qa8fk-qa8fm", 66127, 10, 2.795e7, 1660579, false, 4},
      {"ss-tem152078", 152078, 10, 6.786e7, 6459330, false, 4},
      {"ss-tem27623", 27623, 10, 5.751e6, 439469, false, 4},
      {"syn-ghep-near-singular-real-1e5", 100000, 10, 2.611e6, 469406, false, 4},
      {"syn-lap3d-aniso-1e5", 97200, 12, 3.651e7, 0, false, 4},
      {"syn-mem27-46", 97336, 12, 5.402e7, 0, false, 8},
      {"syn-multiplicity-real-1e4", 10000, 14, 2.399e5, 0, false, 4},
      {"syn-near-singular-complex-1e5", 100000, 10, 6.401e6, 0, true, 4},
      {"syn-zero-cluster-real-1e5", 100000, 21, 2.401e6, 0, false, 4},
      {"yee-bloch-1e4", 10080, 10, 1.408e7, 0, true, 8},
      {"yee-pec-box-1e5", 109744, 10, 5.376e7, 109744, false, 4},
      {"yee-pec-cube-32", 92256, 12, 4.584e7, 0, false, 8},
  };
  for (const Pinned& c : cases) {
    ShiftInvertShape s;
    s.n = c.n;
    s.nev = c.nev;
    s.factor_entries = c.entries;
    s.nnz_b = c.nnz_b;
    s.complex_scalars = c.cx;
    EXPECT_EQ(recommended_block_size(s), c.p) << c.name;
  }
}

TEST(LanczosBlockRule, TinyOperatorsKeepSingleVectors) {
  // No block layout fits (few wanted pairs get blocks otherwise: test_blocktune_fix_rule.cpp).
  ShiftInvertShape s;
  s.n = 30;
  s.nev = 3;
  s.factor_entries = 400;
  EXPECT_EQ(recommended_block_size(s), 1);
  s.nev = 10;
  EXPECT_EQ(recommended_block_size(s), 1);  // the floor needs a layout that fits, too
}

}  // namespace
