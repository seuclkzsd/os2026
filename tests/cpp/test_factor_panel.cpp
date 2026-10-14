// Unit tests of the MUMPS block sizes of P9 "factspeed" (src/factor/mumps_api.hpp Blocking,
// mumps_factorization.cpp, docs/DESIGN.md §3.4): with HS_MUMPS_BLOCKING unset a MumpsFactorization
// takes the build's default (CMake: MUMPS' own block sizes, P9 ones with HSOLVE_MUMPS_ROOT);
// "mumps" keeps MUMPS' own; "p9" gives every instance (double and single, main objects and
// InertiaCounter's) the full-rank panel width KEEP(420), the contribution-block row blocks
// KEEP(57/58) and the L0-layer blocks KEEP(498/499) that only a MUMPS patched with
// patches/mumps-5.8.2-l0-blocking.patch reads (a build with HSOLVE_MUMPS_ROOT runs these tests
// against it). Changing them changes how a front is blocked, not the factorization's meaning: the
// Sylvester counts with the P9 blocks equal those with MUMPS' defaults (128, 2000 / 1000, unset)
// and the exact counts - real and through the complex embedding, main factorizations and
// inertia-only counts alike. Also the measurement hooks (MumpsControlOverrides, instance()).
// Which test reaches which blocking path (MUMPS 5.8.2):
//   * CountsEqualMumpsDefaultAndAnalytic, SingularShiftAgrees - 2-D grids of 200 x 210 (real) and
//     140 x 150 (complex embedding), mid-gap shifts and shifts next to eigenvalues (1e-6 away with
//     null-pivot detection, 1e-9 without), and an exact eigenvalue: fronts wider than 256 fully
//     summed columns, so KEEP(420) = 256 changes the panels above the L0 layer. Their contribution
//     blocks have at most ~600 rows, which KEEP(57/58) leave alone (both blockings take one block
//     of NFRONT - NASS rows, dfac_front_aux.F DMUMPS_FAC_T_LDLT), and which of their fronts lie
//     under the L0 layer is not known.
//   * CountsEqualOnLargeContributionBlocksAndWideL0Fronts - a matrix whose assembly tree is known
//     (asserted): two leaf fronts of 200 fully summed columns with 1100-row contribution blocks
//     under one root front of 2100, the two leaves forming the L0 layer of 2 threads (asserted
//     from MUMPS' flops under L0, DKEEP(16)). So every path differs between the two blockings: the
//     contribution-block update (MUMPS' own 1000 / 2000: two blocks of 550 rows; P9: one of 1100),
//     the root's panels (128 vs 256), and on the leaves under L0 the patched MUMPS' KEEP(498) =
//     128 panels (unpatched: 256) and KEEP(499) = 64 stripes, diagonal blocks and zeroed band
//     (MUMPS: 150 / 120, 250, 249).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "factor/mumps_api.hpp"
#include "factor/mumps_factorization.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

#ifndef HSOLVE_MUMPS_BLOCKING_DEFAULT
#define HSOLVE_MUMPS_BLOCKING_DEFAULT "mumps"
#endif

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::FactorOptions;
using hs::index_t;
using hs::ShiftedPencil;
template <class T, class U>
using Mumps = hs::mumps::MumpsFactorization<T, U>;

index_t count_below(const std::vector<double>& w, double s) {
  return std::lower_bound(w.begin(), w.end(), s) - w.begin();
}

// D K D^H with D = diag(exp(i theta_j)), random theta: the same spectrum as K, genuinely complex
// off-diagonal entries (the embedding does not decouple). Real T: K itself.
template <class T>
CsrMatrix<T> phased(const CsrMatrix<T>& k) {
  if constexpr (std::is_same_v<T, double>) {
    return k;
  } else {
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> u(0.0, 6.283185307179586);
    std::vector<cplx> d(static_cast<std::size_t>(k.nrows));
    for (auto& x : d) x = std::polar(1.0, u(rng));
    CsrMatrix<T> out = k;
    for (index_t i = 0; i < k.nrows; ++i)
      for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p)
        out.values[p] = d[i] * k.values[p] * std::conj(d[k.colind[p]]);
    return out;
  }
}

// Simple eigenvalues w[j] with gaps >= 1e-6 on both sides, spread over the spectrum.
std::vector<index_t> isolated(const std::vector<double>& w) {
  std::vector<index_t> out;
  const index_t n = static_cast<index_t>(w.size());
  for (index_t k : {n / 7, n / 3, n / 2, (2 * n) / 3, n - n / 9}) {
    index_t j = std::max<index_t>(k, 1);
    while (j + 1 < n && (w[j + 1] - w[j] < 1e-6 || w[j] - w[j - 1] < 1e-6)) ++j;
    if (j + 1 < n) out.push_back(j);
  }
  return out;
}

// Mid-gap shifts and shifts `d` away from isolated eigenvalues (both sides). With null-pivot
// detection d must exceed its threshold (CNTL(3) = 1e-8 relative to ||K|| ~ 8): 1e-6; without
// detection 1e-9 is still far above the rounding of K (~1e-15 ||K||): exact counts either way.
std::vector<double> test_shifts(const std::vector<double>& w, double d) {
  std::vector<double> s;
  for (index_t j : isolated(w)) {
    s.push_back(0.5 * (w[j] + w[j + 1]));
    s.push_back(w[j] - d);
    s.push_back(w[j] + d);
  }
  return s;
}

// Two dense "leaf" cliques of p vertices, each coupled densely to the first cb vertices of a dense
// "root" clique of r >= cb vertices, the leaves not coupled to each other; random entries in
// [-1, 1] (and i [-1, 1]). Vertices 0..p-1 and p..2p-1 are the leaves, 2p..2p+r-1 the root.
template <class T>
CsrMatrix<T> two_leaves_and_root(index_t p, index_t cb, index_t r, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto draw = [&]() -> T {
    if constexpr (std::is_same_v<T, double>) {
      return u(rng);
    } else {
      const double re = u(rng);
      return T(re, u(rng));
    }
  };
  const index_t n = 2 * p + r;
  auto part = [p](index_t i) { return i < p ? 0 : (i < 2 * p ? 1 : 2); };
  std::vector<index_t> rows, cols;
  std::vector<T> vals;
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j <= i; ++j) {
      if (part(i) != 2 && part(j) != 2 && part(i) != part(j)) continue;  // leaf 1 - leaf 2
      if (part(i) == 2 && part(j) != 2 && i - 2 * p >= cb) continue;     // root beyond cb - leaf
      if (i == j) {
        rows.push_back(i), cols.push_back(i), vals.push_back(T(u(rng)));
        continue;
      }
      const T v = draw();
      rows.push_back(i), cols.push_back(j), vals.push_back(v);
      rows.push_back(j), cols.push_back(i), vals.push_back(hs::conj_if(v));
    }
  return hs::csr_from_coo<T>(n, n, rows, cols, vals);
}

// Pivot order (MumpsFactorization::set_pivot_order: 1-based pivot position of every variable of
// the matrix MUMPS factorizes) that eliminates the vertices in their natural order - the leaves
// before the root; for complex T the embedding's variables v (Re) and n + v (Im) one after the
// other.
std::vector<int> natural_pivot_order(index_t n, bool embedding) {
  std::vector<int> o(static_cast<std::size_t>(embedding ? 2 * n : n));
  for (index_t v = 0; v < n; ++v) {
    if (embedding) {
      o[static_cast<std::size_t>(v)] = static_cast<int>(2 * v + 1);
      o[static_cast<std::size_t>(n + v)] = static_cast<int>(2 * v + 2);
    } else {
      o[static_cast<std::size_t>(v)] = static_cast<int>(v + 1);
    }
  }
  return o;
}

// The P9 block sizes and MUMPS' own, as explicit overrides (independent of the environment).
hs::mumps::MumpsControlOverrides p9_blocks() {
  hs::mumps::MumpsControlOverrides o;
  o.keep = {{420, hs::mumps::kFullRankPanelWidth},
            {57, 2 * hs::mumps::kCbUpdateBlock},
            {58, hs::mumps::kCbUpdateBlock},
            {498, hs::mumps::kL0PanelWidth},
            {499, hs::mumps::kL0Block}};
  return o;
}
hs::mumps::MumpsControlOverrides mumps_blocks() {
  hs::mumps::MumpsControlOverrides o;
  o.keep = {{420, 128}, {57, 2000}, {58, 1000}, {498, 0}, {499, 0}};
  return o;
}

// HS_MUMPS_BLOCKING for the objects constructed in its scope (the previous value is restored).
class ScopedBlocking {
 public:
  explicit ScopedBlocking(const char* value) {
    if (const char* old = std::getenv("HS_MUMPS_BLOCKING")) {
      old_ = old;
      had_ = true;
    }
    if (value != nullptr) {
      setenv("HS_MUMPS_BLOCKING", value, 1);
    } else {
      unsetenv("HS_MUMPS_BLOCKING");
    }
  }
  ~ScopedBlocking() {
    if (had_) {
      setenv("HS_MUMPS_BLOCKING", old_.c_str(), 1);
    } else {
      unsetenv("HS_MUMPS_BLOCKING");
    }
  }
  ScopedBlocking(const ScopedBlocking&) = delete;
  ScopedBlocking& operator=(const ScopedBlocking&) = delete;

 private:
  std::string old_;
  bool had_ = false;
};

template <class T>
class FactorPanelWidth : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(FactorPanelWidth, Scalars);

TYPED_TEST(FactorPanelWidth, UnsetMeansTheBuildsDefault) {
  // CMake's HSOLVE_MUMPS_BLOCKING as resolved for this build (p9 with HSOLVE_MUMPS_ROOT, else mumps).
  const auto expected = std::string(HSOLVE_MUMPS_BLOCKING_DEFAULT) == "p9" ? hs::mumps::Blocking::P9
                                                                         : hs::mumps::Blocking::Mumps;
  for (const char* v : {static_cast<const char*>(nullptr), ""}) {
    ScopedBlocking env(v);
    EXPECT_EQ(hs::mumps::default_blocking(), expected);
    EXPECT_EQ(hs::mumps::blocking_from_environment(), expected);
  }
}

TYPED_TEST(FactorPanelWidth, MumpsKeepsMumpsOwnBlocking) {
  using T = TypeParam;
  {
    ScopedBlocking env("mumps");
    EXPECT_EQ(hs::mumps::blocking_from_environment(), hs::mumps::Blocking::Mumps);
    Mumps<T, double> d(FactorOptions{});
    Mumps<T, float> s(FactorOptions{});
    EXPECT_EQ(d.instance().keep(420), 128);  // MUMPS: 4 * KEEP(6)
    EXPECT_EQ(s.instance().keep(420), 128);
    EXPECT_EQ(d.instance().keep(58), 1000);
    EXPECT_EQ(d.instance().keep(57), 2000);
    EXPECT_EQ(d.instance().keep(498), 0);
    EXPECT_EQ(d.instance().keep(499), 0);
  }
}

TYPED_TEST(FactorPanelWidth, OptInGivesEveryInstanceTheP9Blocking) {
  using T = TypeParam;
  ScopedBlocking env("p9");
  EXPECT_EQ(hs::mumps::blocking_from_environment(), hs::mumps::Blocking::P9);
  Mumps<T, double> d(FactorOptions{});
  Mumps<T, float> s(FactorOptions{});
  EXPECT_EQ(d.instance().keep(420), hs::mumps::kFullRankPanelWidth);
  EXPECT_EQ(s.instance().keep(420), hs::mumps::kFullRankPanelWidth);
  EXPECT_EQ(d.instance().keep(58), hs::mumps::kCbUpdateBlock);
  EXPECT_EQ(s.instance().keep(58), hs::mumps::kCbUpdateBlock);
  EXPECT_EQ(d.instance().keep(57), 2 * hs::mumps::kCbUpdateBlock);
  EXPECT_EQ(d.instance().keep(498), hs::mumps::kL0PanelWidth);  // only a patched MUMPS reads these
  EXPECT_EQ(s.instance().keep(499), hs::mumps::kL0Block);
  // InertiaCounter's instances are MumpsFactorizations too: exact counts with the setting.
  const auto a = phased(hs::test::laplacian_2d<T>(30, 31));
  const auto w = hs::test::laplacian_2d_eigenvalues(30, 31);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -1.0);
  hs::InertiaCounter<T> counter(*f, {});
  for (index_t j : isolated(w)) {
    const double sigma = 0.5 * (w[j] + w[j + 1]);
    EXPECT_EQ(counter.count(1.0, -sigma).neg, count_below(w, sigma)) << "sigma=" << sigma;
  }
}

TYPED_TEST(FactorPanelWidth, BadValueIsRejected) {
  using T = TypeParam;
  ScopedBlocking env("wide");
  EXPECT_THROW((Mumps<T, double>(FactorOptions{})), hs::Error);
}

// The override hook writes after hsolve's settings, before analysis and factorization; instance()
// shows MUMPS' statistics.
TYPED_TEST(FactorPanelWidth, OverridesWinAndStatisticsAreReadable) {
  using T = TypeParam;
  ScopedBlocking env("p9");
  Mumps<T, double> d(FactorOptions{});
  const auto a = hs::test::laplacian_2d<T>(12, 13);
  ShiftedPencil<T> pencil(a, nullptr);
  hs::mumps::MumpsControlOverrides ov;
  ov.keep.push_back({420, 128});
  ov.icntl.push_back({14, 50});
  d.set_control_overrides(ov);
  d.analyze(pencil, 1.0, -1.0);
  EXPECT_EQ(d.instance().keep(420), 128);
  EXPECT_TRUE(d.factorize(1.0, -1.0));
  EXPECT_GT(d.instance().rinfog(3), 0.0);  // elimination flops of the factorization
  EXPECT_GE(d.instance().infog(11), 1);    // largest front
}

// Main factorizations (factors kept) and inertia-only instances on the main pivot order, with the
// P9 blocks and with MUMPS' own: identical, exact counts.
TYPED_TEST(FactorPanelWidth, CountsEqualMumpsDefaultAndAnalytic) {
  using T = TypeParam;
  // Real: the top fronts' fully summed parts are ~400 wide; the embedding doubles them.
  const bool real = std::is_same_v<T, double>;
  const index_t nx = real ? 200 : 140, ny = real ? 210 : 150;
  const auto a = phased(hs::test::laplacian_2d<T>(nx, ny));
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.ordering = 5;  // METIS (fixed seed): the same pivot order in both objects
  o.threads = 2;
  o.deterministic = false;  // L0-OpenMP: the patched build's L0 blocks are exercised
  for (bool discard : {false, true}) {
    Mumps<T, double> ours(o), theirs(o);
    ours.set_control_overrides(p9_blocks());
    theirs.set_control_overrides(mumps_blocks());
    ours.set_discard_factors(discard);
    theirs.set_discard_factors(discard);
    ours.analyze(pencil, 1.0, -4.0);
    theirs.set_pivot_order(ours.pivot_order());
    theirs.analyze(pencil, 1.0, -4.0);
    ASSERT_GT(ours.instance().infog(5), hs::mumps::kFullRankPanelWidth);  // fronts wide enough
    for (bool detection : {true, false}) {
      ours.set_null_pivot_detection(detection);
      theirs.set_null_pivot_detection(detection);
      for (double s : test_shifts(w, detection ? 1e-6 : 1e-9)) {
        const bool ok1 = ours.factorize(1.0, -s);
        const bool ok2 = theirs.factorize(1.0, -s);
        ASSERT_TRUE(ok1) << "s=" << s << " detection=" << detection;
        ASSERT_TRUE(ok2) << "s=" << s << " detection=" << detection;
        EXPECT_EQ(ours.inertia().neg, count_below(w, s)) << "s=" << s << " discard=" << discard;
        EXPECT_EQ(theirs.inertia().neg, ours.inertia().neg) << "s=" << s << " discard=" << discard;
        EXPECT_EQ(ours.inertia().zero, 0);
      }
    }
  }
}

// Every path the P9 setting changes, on a matrix whose assembly tree is known (header comment).
// MUMPS sizes (variables of the matrix it factorizes; the complex embedding from half as many
// vertices): leaves of p = 200 fully summed columns with a contribution block of cb = 1100 rows,
// a root of r = 2100. A father of more than 2000 pivots keeps MUMPS from amalgamating children
// with more than KEEP(1) pivots into it (dana_aux.F: MAXNODE = max(2000, ...)); the leaves hold
// ~10% of the factor, so the L0 layer of 2 threads is the two leaves (ana_omp_m.F: at most 50% of
// the factor under it, the threads' loads within 90%). The preconditions are asserted, so a MUMPS
// whose tree, L0 layer or contribution-block rule differs fails here instead of passing without
// reaching those paths.
TYPED_TEST(FactorPanelWidth, CountsEqualOnLargeContributionBlocksAndWideL0Fronts) {
  using T = TypeParam;
  const bool real = std::is_same_v<T, double>;
  const index_t m = real ? 1 : 2;  // MUMPS variables per vertex
  const index_t p = 200 / m, cb = 1100 / m, r = 2100 / m;
  const auto a = two_leaves_and_root<T>(p, cb, r, 2026);
  const auto w = hs::test::dense_eigenvalues<T>(a, nullptr);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.threads = 2;
  o.deterministic = false;  // the L0-OpenMP layer
  constexpr int kMumpsCbBlock = 1000;  // MUMPS' KEEP(58)
  // Contribution blocks in (1000, 2000] rows: MUMPS' own KEEP(57/58) update them in two row
  // blocks, the P9 ones in one (DMUMPS_FAC_T_LDLT). Panels: the root wider than the P9 panel, the
  // leaves wider than the L0 panel and than MUMPS' triangle-stripe threshold KEEP(7) = 150.
  ASSERT_GT(m * cb, kMumpsCbBlock);
  ASSERT_LE(m * cb, hs::mumps::kCbUpdateBlock);
  ASSERT_GT(m * r, hs::mumps::kFullRankPanelWidth);
  ASSERT_GT(m * p, hs::mumps::kL0PanelWidth);
  ASSERT_GT(m * p, 150);
  for (bool discard : {false, true}) {
    Mumps<T, double> ours(o), theirs(o);
    ours.set_control_overrides(p9_blocks());
    theirs.set_control_overrides(mumps_blocks());
    for (auto* f : {&ours, &theirs}) {
      f->set_discard_factors(discard);
      f->set_pivot_order(natural_pivot_order(a.nrows, !real));
      f->analyze(pencil, 1.0, -4.0);
      // Three fronts (INFOG(6)): the two leaves (NASS = m p) under the root (NASS = m r); the
      // largest contribution block (KEEP(2), rows) is a leaf's.
      ASSERT_EQ(f->instance().infog(6), 3);
      ASSERT_EQ(f->instance().keep(2), m * cb);
    }
    EXPECT_EQ(theirs.pivot_order(), ours.pivot_order());
    // Two isolated eigenvalues: the gap's middle and 1e-3 to either side (gaps here ~0.05; the
    // dense fronts make each factorization ~4 GFlop, so fewer shifts than the grid tests).
    auto shifts = test_shifts(w, 1e-3);
    ASSERT_GE(shifts.size(), 6u);
    shifts.resize(6);
    for (double s : shifts) {
      ASSERT_TRUE(ours.factorize(1.0, -s)) << "s=" << s;
      ASSERT_TRUE(theirs.factorize(1.0, -s)) << "s=" << s;
      EXPECT_EQ(ours.inertia().neg, count_below(w, s)) << "s=" << s << " discard=" << discard;
      EXPECT_EQ(theirs.inertia().neg, ours.inertia().neg) << "s=" << s << " discard=" << discard;
      EXPECT_EQ(ours.inertia().zero, 0);
      // Work under the L0 layer and above it: with three fronts that layer is the two leaves.
      for (auto* f : {&ours, &theirs}) {
        EXPECT_GT(f->instance().dkeep(16), 0.0) << "s=" << s;
        EXPECT_LT(f->instance().dkeep(16), f->instance().rinfog(3)) << "s=" << s;
      }
    }
  }
}

// At an exact (simple) eigenvalue K is singular: both blockings detect it (valid == false) or, if
// the heuristic missed it, count within [below, below + 1].
TYPED_TEST(FactorPanelWidth, SingularShiftAgrees) {
  using T = TypeParam;
  const index_t nx = 200, ny = 211;
  const auto a = phased(hs::test::laplacian_2d<T>(nx, ny));
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.ordering = 5;
  o.threads = 2;
  o.deterministic = false;
  const index_t j = isolated(w).at(2);  // a simple eigenvalue in the middle of the spectrum
  const double s = w[j];
  for (bool p9 : {true, false}) {
    Mumps<T, double> f(o);
    f.set_control_overrides(p9 ? p9_blocks() : mumps_blocks());
    f.analyze(pencil, 1.0, -s);
    const bool ok = f.factorize(1.0, -s);
    if (ok) {
      EXPECT_GE(f.inertia().neg, j) << "p9 " << p9;
      EXPECT_LE(f.inertia().neg, j + 1) << "p9 " << p9;
    } else {
      EXPECT_FALSE(f.inertia().valid);
    }
  }
}

}  // namespace
