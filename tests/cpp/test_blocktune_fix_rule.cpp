// Regression tests of the block-size rule (src/eigen/block_rule) for the P5c review of "blocktune":
// the floor at the P4 auto width from kRuleFloorMinNev wanted pairs on (low-fill multiplicity
// cases were slower at the model's p = 2), the measured picks below nev = 8 (blocks for a few
// wanted pairs, which the P5b documentation denied), and the picks' robustness against the
// run-to-run spread of the factor entries under the SCOTCH ordering.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "eigen/block_lanczos.hpp"
#include "eigen/block_rule.hpp"

namespace {

using hs::index_t;
using hs::eigen_detail::block_dimensions;
using hs::eigen_detail::kRuleFloorMinNev;
using hs::eigen_detail::kRuleFloorWidth;
using hs::eigen_detail::LanczosSetup;
using hs::eigen_detail::recommended_block_size;
using hs::eigen_detail::ShiftInvertShape;
using hs::eigen_detail::WidthTable;

ShiftInvertShape shape(index_t n, index_t nev, double entries, double nnz_b, bool cx) {
  ShiftInvertShape s;
  s.n = n;
  s.nev = nev;
  s.factor_entries = entries;
  s.nnz_b = nnz_b;
  s.complex_scalars = cx;
  return s;
}

bool fits(index_t n, index_t nev, index_t p) {
  LanczosSetup s;
  s.n = s.n_free = n;
  s.nev = nev;
  s.ncv = std::min(n, std::max(2 * nev + 1, nev + 16));
  return block_dimensions(s, p, true).feasible;
}

TEST(LanczosBlockRuleFix, FloorKeepsTheP4WidthFromEightWantedPairs) {
  EXPECT_EQ(kRuleFloorMinNev, 8);
  EXPECT_EQ(kRuleFloorWidth, 4);
  // Low-fill shapes (2-D meshes, random sparse patterns, real and complex), where the model alone
  // picks p = 2: from nev = 8 on never below 4, below that p = 2 stays.
  struct LowFill {
    const char* name;
    index_t n;
    double entries, nnz_b;
    bool cx;
  };
  const LowFill shapes[] = {{"cav2d-rect-pert-1e5", 99459, 3.871e6, 495751, false},
                            {"syn-ghep-multiplicity-real-1e4", 10000, 2.608e5, 47096, false},
                            {"syn-multiplicity-complex-1e4", 10000, 6.402e5, 0, true},
                            {"syn-zero-cluster-real-1e5", 100000, 2.401e6, 0, false}};
  for (const LowFill& l : shapes) {
    SCOPED_TRACE(l.name);
    EXPECT_EQ(recommended_block_size(shape(l.n, 7, l.entries, l.nnz_b, l.cx)), 2);
    for (index_t nev : {8, 9, 10, 14, 22, 40}) {
      const index_t p = recommended_block_size(shape(l.n, nev, l.entries, l.nnz_b, l.cx));
      EXPECT_GE(p, 4) << "nev " << nev;
    }
  }
  // Any step costs: the floor applies even where blocks are modelled not to pay (P4 ran p = 4
  // there too), from nev = 8 on only.
  const WidthTable linear{1, 2, 4, 8};
  EXPECT_EQ(recommended_block_size(100000, 7, linear), 1);
  EXPECT_EQ(recommended_block_size(100000, 8, linear), 4);
  EXPECT_EQ(recommended_block_size(100000, 30, linear), 4);
  // Wider picks stay.
  EXPECT_EQ(recommended_block_size(100000, 12, WidthTable{1, 1, 1, 1}), 8);
  // The floor needs the p = 4 layout to fit: nev = 8, n = 53 fits p = 2 (2 (24 + 2) <= 53) but
  // not p = 4 (2 (24 + 4) > 53).
  ASSERT_TRUE(fits(53, 8, 2));
  ASSERT_FALSE(fits(53, 8, 4));
  EXPECT_EQ(recommended_block_size(53, 8, WidthTable{1, 1, 1, 1}), 2);
  EXPECT_EQ(recommended_block_size(53, 8, linear), 1);
  ASSERT_TRUE(fits(56, 8, 4));
  EXPECT_EQ(recommended_block_size(56, 8, linear), 4);
}

TEST(LanczosBlockRuleFix, FewWantedPairsGetTheMeasuredWidths) {
  // nev = 1..7 on the six cases measured with the target k replaced (block_rule.cpp): blocks pay
  // for a single wanted pair already; 3-D factors p = 4 (iteration time 0.40..0.77 of p = 1), little
  // fill p = 2 (0.57..0.99). Shapes: n, the median factor entries of the runs, nnz(B), scalar type.
  struct Measured {
    const char* name;
    index_t n;
    double entries, nnz_b;
    bool cx;
    index_t max_nev, p;
  };
  const Measured cases[] = {
      {"yee-pec-box-1e5", 109744, 5.43e7, 109744, false, 7, 4},
      {"ss-qa8fk-qa8fm", 66127, 2.75e7, 1660579, false, 7, 4},
      {"yee-bloch-1e4", 10080, 1.41e7, 0, true, 7, 4},
      {"cav2d-rect-pert-1e5", 99459, 3.871e6, 495751, false, 7, 2},
      {"ss-tem27623", 27623, 5.71e6, 439469, false, 6, 2},  // nev = 7 sits at the 2 / 4 switch
      {"syn-near-singular-complex-1e4", 10000, 6.401e5, 0, true, 7, 2},
  };
  for (const Measured& c : cases) {
    for (index_t nev = 1; nev <= c.max_nev; ++nev) {
      EXPECT_EQ(recommended_block_size(shape(c.n, nev, c.entries, c.nnz_b, c.cx)), c.p)
          << c.name << " nev " << nev;
    }
  }
}

TEST(LanczosBlockRuleFix, PinnedPicksSurviveTheOrderingSpread) {
  // SCOTCH orders differently from analysis to analysis: the factor entries of identical runs
  // varied by up to 8.8 % (yee-bloch-1e4), 2.2..6.9 % on the other 3-D cases. Every calibration
  // shape keeps its pick for entries within +-10 % of the pinned value.
  struct Pinned {
    const char* name;
    index_t n, nev;
    double entries, nnz_b;
    bool cx;
  };
  const Pinned cases[] = {
      {"cav2d-rect-pert-1e5", 99459, 6, 3.871e6, 495751, false},
      {"cav2d-square-1e4", 9976, 7, 3.418e5, 34984, false},
      {"ss-2cubes_sphere", 101492, 10, 4.422e7, 0, false},
      {"ss-qa8fk-qa8fm", 66127, 10, 2.795e7, 1660579, false},
      {"ss-tem152078", 152078, 10, 6.786e7, 6459330, false},
      {"ss-tem27623", 27623, 10, 5.751e6, 439469, false},
      {"syn-ghep-near-singular-real-1e5", 100000, 10, 2.611e6, 469406, false},
      {"syn-lap3d-aniso-1e5", 97200, 12, 3.651e7, 0, false},
      {"syn-mem27-46", 97336, 12, 5.402e7, 0, false},
      {"syn-multiplicity-real-1e4", 10000, 14, 2.399e5, 0, false},
      {"syn-near-singular-complex-1e5", 100000, 10, 6.401e6, 0, true},
      {"syn-zero-cluster-real-1e5", 100000, 21, 2.401e6, 0, false},
      {"yee-bloch-1e4", 10080, 10, 1.408e7, 0, true},
      {"yee-pec-box-1e5", 109744, 10, 5.376e7, 109744, false},
      {"yee-pec-cube-32", 92256, 12, 4.584e7, 0, false},
  };
  for (const Pinned& c : cases) {
    const index_t p = recommended_block_size(shape(c.n, c.nev, c.entries, c.nnz_b, c.cx));
    for (int i = -10; i <= 10; ++i) {
      const double f = 1.0 + 0.01 * i;
      EXPECT_EQ(recommended_block_size(shape(c.n, c.nev, f * c.entries, c.nnz_b, c.cx)), p)
          << c.name << " entries x" << f;
    }
  }
}

}  // namespace
