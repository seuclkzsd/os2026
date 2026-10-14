// Unit tests of the MUMPS block sizes of P9 "factspeed" (src/factor/mumps_api.hpp Blocking,
// mumps_factorization.cpp, docs/DESIGN.md §3.4): by default every MumpsFactorization keeps MUMPS'
// own block sizes; HS_MUMPS_BLOCKING=p9 gives every instance (double and single, main objects and
// InertiaCounter's) the full-rank panel width KEEP(420), the contribution-block row blocks
// KEEP(57/58) and the L0-layer blocks KEEP(498/499) that only a MUMPS patched with
// patches/mumps-5.8.2-l0-blocking.patch reads (a build with HSOLVE_MUMPS_ROOT runs these tests
// against it). Changing them changes how a front is blocked, not the factorization's meaning: the
// Sylvester counts with the P9 blocks equal those with MUMPS' defaults (128, 2000 / 1000, unset)
// and the analytic counts - real and through the complex embedding, at mid-gap shifts and next to
// eigenvalues (1e-6 away with null-pivot detection, 1e-9 without), main factorizations and
// inertia-only counts alike; at an exact eigenvalue both report a singular K. Also the measurement
// hooks (MumpsControlOverrides, instance()). The fronts must be wider than the panels for the
// setting to matter: 2-D grids of 200 x 210 (real) and 140 x 150 (complex embedding), whose largest
// fronts are wider than 256.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
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

TYPED_TEST(FactorPanelWidth, DefaultKeepsMumpsOwnBlocking) {
  using T = TypeParam;
  for (const char* v : {static_cast<const char*>(nullptr), "", "mumps"}) {
    ScopedBlocking env(v);
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
