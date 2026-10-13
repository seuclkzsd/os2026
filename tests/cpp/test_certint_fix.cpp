// Regression tests of the certint review fixes (src/driver/certificate.*, definite.cpp):
//   * fewer than k selected pairs never pass a completeness check: the deflated rerun looks for
//     the missing ones (fault injection removing the FARTHEST pair, all modes);
//   * a randomized check whose kappa of B is only estimated is never reported as a certificate;
//   * single-precision counts above the found pairs, or contradicting the exact count at sigma,
//     are checked by double counts on the same InertiaCounter (a miscount no longer reruns, nor
//     certifies), the statement names each step once;
//   * the driver's exact_single path end to end (auto on a 3-D-fill pencil: deficit, confirmed,
//     rerun with a refactorization, certified again), the exact path's P4 invariants, and the
//     proven-only start scaling.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/certificate.hpp"
#include "driver/internal.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/verify.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

hs::SolverConfig nearest_cfg(double sigma, index_t k, hs::CertificateMode mode) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.certificate = mode;
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
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

std::size_t occurrences(const std::string& text, const std::string& what) {
  std::size_t n = 0;
  for (auto p = text.find(what); p != std::string::npos; p = text.find(what, p + 1)) ++n;
  return n;
}

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

const char* mode_name(hs::CertificateMode m) {
  switch (m) {
    case hs::CertificateMode::Exact: return "exact";
    case hs::CertificateMode::Randomized: return "randomized";
    case hs::CertificateMode::Auto: return "auto";
    case hs::CertificateMode::Off: return "off";
  }
  return "?";
}

constexpr hs::CertificateMode kCheckingModes[] = {hs::CertificateMode::Exact, hs::CertificateMode::Randomized,
                                                  hs::CertificateMode::Auto};

template <class T>
class CertintFixTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(CertintFixTyped, Scalars);

// --------------------------------------------- fewer than k selected pairs: rerun, never pass

/// Fault injection on a spectrum without multiple eigenvalues removes the selected eigenvalue
/// farthest from sigma (and its Ritz hints). Then fewer than k pairs are selected, and no check
/// may pass that: the randomized claim disc (nearer half-width of the window) and even the count
/// window may end before the removed eigenvalue. Every mode reruns and finds all k.
template <class T>
void expect_farthest_drop_recovered(const std::vector<double>& w, double sigma, index_t k) {
  const auto a = hs::test::prescribed_spectrum<T>(w, 3, 29);
  for (const auto mode : kCheckingModes) {
    auto cfg = nearest_cfg(sigma, k, mode);
    cfg.certificate_fault_drop = 1;
    const auto sol = hs::solve(a, nullptr, cfg);
    const auto& r = sol.report;
    EXPECT_TRUE(r.success) << mode_name(mode) << ": " << r.certificate.statement;
    EXPECT_TRUE(r.certificate.passed) << mode_name(mode);
    EXPECT_TRUE(r.certificate.rigorous) << mode_name(mode);
    EXPECT_TRUE(has_warning(r, "certificate_fault_drop: 1 found pair(s) removed")) << mode_name(mode);
    EXPECT_TRUE(has_warning(r, "deflated rerun")) << mode_name(mode);
    EXPECT_FALSE(has_warning(r, "pairs found")) << mode_name(mode);  // "only k-1 of k pairs found"
    expect_same_set(sol.eigenvalues, nearest(w, sigma, k), 1e-10);
  }
}

// Review finding: the randomized pass radius (0.225, the window's nearer half-width) ends before
// the removed 0.3; the check passed and the solve returned 2 of 3 pairs.
TYPED_TEST(CertintFixTyped, FarthestDroppedOutsideTheClaimDiscIsRecovered) {
  std::vector<double> w{-0.1, 0.1, 0.3, -0.35};
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 0.0, 3);
}

// Review finding: an asymmetric window - the 10th nearest (1.13) lies inside the count window but
// beyond the symmetric pass radius (0.12, set by 0.85 below).
TYPED_TEST(CertintFixTyped, FarthestDroppedInAnAsymmetricWindowIsRecovered) {
  std::vector<double> w;
  for (int i = 1; i <= 9; ++i) w.push_back(1.0 + (i % 2 ? 1.0 : -1.0) * 0.01 * i);
  w.push_back(1.13);
  w.push_back(0.85);
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 1.0, 10);
}

// Review finding: the next eigenvalue (1.14) right beyond the removed 1.13 puts the count window's
// end mid-way between 1.09 and 1.14, before 1.13: the counts equal the found pairs, and even the
// exact mode returned k-1 pairs.
TYPED_TEST(CertintFixTyped, FarthestDroppedBeyondTheCountWindowIsRecovered) {
  std::vector<double> w;
  for (int i = 1; i <= 9; ++i) w.push_back(1.0 + (i % 2 ? 1.0 : -1.0) * 0.01 * i);
  w.push_back(1.13);
  w.push_back(1.14);
  w.push_back(0.84);
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 1.0, 10);
}

// ------------------------------------------------- a screening result is never a certificate

/// blockdiag(1000 L, A0) x = lambda blockdiag(L, I) x with L a 2-D Dirichlet Laplacian: B is HPD
/// but not diagonally dominant, so start_scaling_for only estimates its kappa (as in test_certint).
template <class T>
std::pair<hs::CsrMatrix<T>, hs::CsrMatrix<T>> general_b_pencil(const std::vector<double>& w) {
  const hs::CsrMatrix<T> a0 = hs::test::prescribed_spectrum<T>(w, 3, 91);
  const hs::CsrMatrix<T> l = hs::test::laplacian_2d<T>(6, 6);
  const index_t nl = l.nrows, n0 = a0.nrows, n = nl + n0;
  auto build = [&](bool for_a) {
    std::vector<index_t> rows, cols;
    std::vector<T> vals;
    for (index_t i = 0; i < nl; ++i)
      for (index_t q = l.rowptr[i]; q < l.rowptr[i + 1]; ++q) {
        rows.push_back(i);
        cols.push_back(l.colind[q]);
        vals.push_back(for_a ? T(1000.0) * l.values[q] : l.values[q]);
      }
    for (index_t i = 0; i < n0; ++i) {
      if (for_a) {
        for (index_t q = a0.rowptr[i]; q < a0.rowptr[i + 1]; ++q) {
          rows.push_back(nl + i);
          cols.push_back(nl + a0.colind[q]);
          vals.push_back(a0.values[q]);
        }
      } else {
        rows.push_back(nl + i);
        cols.push_back(nl + i);
        vals.push_back(T(1.0));
      }
    }
    return hs::csr_from_coo<T>(n, n, rows, cols, vals);
  };
  return {build(true), build(false)};
}

std::vector<double> gapped_spectrum() {
  std::vector<double> w{1.0, 1.0, 1.0, 1.0, 0.97, 1.05, 1.08, 0.93, 1.11, 0.9};
  for (int i = 0; i < 400; ++i) w.push_back(2.0 + 3.0 * i / 400);
  for (int i = 0; i < 100; ++i) w.push_back(-1.0 - 2.0 * i / 100);
  return w;
}

// Review finding: certificate = randomized with an estimated kappa reported passed = true,
// success = true (a screening result read as a certificate downstream). The report's passed
// verdict must be a proof; the randomized check, which could only screen here, is not run and the
// exact counts certify (the statement says why).
TYPED_TEST(CertintFixTyped, EstimatedKappaNeverReportsACertificate) {
  using T = TypeParam;
  const std::vector<double> w = gapped_spectrum();
  const auto [a, b] = general_b_pencil<T>(w);
  for (const index_t drop : {0, 1}) {
    auto cfg = nearest_cfg(1.02, 10, hs::CertificateMode::Randomized);
    cfg.certificate_fault_drop = drop;
    const auto sol = hs::solve(a, &b, cfg);
    const auto& c = sol.report.certificate;
    EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
    EXPECT_TRUE(sol.report.success) << drop;
    EXPECT_TRUE(c.passed) << drop;
    EXPECT_TRUE(c.rigorous) << drop << ": " << c.statement;
    EXPECT_EQ(c.method, "exact") << drop;
    EXPECT_EQ(c.failure_probability, 0.0) << drop;
    EXPECT_EQ(c.solves, 0) << drop;  // no randomized check ran
    EXPECT_EQ(c.statement.rfind("randomized check", 0), 0u) << c.statement;
    EXPECT_NE(c.statement.find("kappa of B"), std::string::npos) << c.statement;
    EXPECT_FALSE(has_warning(sol.report, "SCREENING")) << drop;
    EXPECT_TRUE(has_warning(sol.report, "certificate=randomized")) << drop;
    expect_same_set(sol.eigenvalues, nearest(w, 1.02, 10), 1e-10);
  }
}

// ------------------------------------------------------------- the certifier with its counts

/// A solved nearest problem and what the certifier works on (as in test_certint): the found pairs
/// as a Block, the exact certificate's window, the main factorization at sigma (AMD).
template <class T>
struct Solved {
  hs::CsrMatrix<T> a;
  double sigma = 0.0;
  hs::SolverConfig cfg;
  hs::Solution<T> sol;
  std::unique_ptr<hs::ShiftedPencil<T>> pencil;
  std::unique_ptr<hs::SparseFactorization<T>> fac;
  hs::driver::Block<T> all;
  hs::driver::NearestWindow window;
};

template <class T>
std::unique_ptr<Solved<T>> solved(hs::CsrMatrix<T> a, double sigma, index_t k, hs::CertificateMode mode) {
  auto s = std::make_unique<Solved<T>>();
  s->a = std::move(a);
  s->sigma = sigma;
  s->cfg = nearest_cfg(sigma, k, hs::CertificateMode::Exact);
  s->sol = hs::solve(s->a, nullptr, s->cfg);
  EXPECT_TRUE(s->sol.report.certificate.passed);
  s->cfg.certificate = mode;
  s->pencil = std::make_unique<hs::ShiftedPencil<T>>(s->a, nullptr);
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;  // AMD
  s->fac = hs::make_mumps_factorization<T>(fo);
  s->fac->analyze(*s->pencil, 1.0, -sigma);
  EXPECT_TRUE(s->fac->factorize(1.0, -sigma));
  s->all.lambda = s->sol.eigenvalues;
  s->all.x = s->sol.eigenvectors;
  hs::driver::recompute_products(s->a, static_cast<const hs::CsrMatrix<T>*>(nullptr), s->all);
  double r = 0.0;
  for (double l : s->sol.eigenvalues) r = std::max(r, std::abs(l - sigma));
  s->window = {sigma, r, s->sol.report.certificate.lower, s->sol.report.certificate.upper};
  return s;
}

/// 2-D Laplacian shifted to be indefinite (simple eigenvalues), nearest 8 to 0.3: low fill; the
/// single-count rule holds on its exact window with the default factor (margin ~5x).
template <class T>
std::unique_ptr<Solved<T>> solved_laplacian(hs::CertificateMode mode) {
  auto a = hs::test::laplacian_2d<T>(40, 37);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t q = a.rowptr[i]; q < a.rowptr[i + 1]; ++q)
      if (a.colind[q] == i) a.values[q] -= T(4.0);
  return solved<T>(std::move(a), 0.3, 8, mode);
}

/// 3-D-fill pencil (B = I) whose wanted eigenvalues sit in a window with margins the single-count
/// rule holds even at 10x its backward factor: blockdiag(7-point 3-D Laplacian (20 x 21 x 22,
/// eigenvalues in (0.06, 12): fill), prescribed block). Nearest 8 to -5: -5.1 (twice), -4.9,
/// -5.5, -4.5, -5.8, -6.0, -4.0; then -6.08, -6.16, ... and -3.92, -3.84, ... (spacing 0.08), so
/// the window ends sit 0.04 beyond the wanted disc against eta(1e4 u_s ||K||_1) ~ 0.012, while the
/// randomized check's relative gap is only ~0.04 (many block steps: auto prefers single counts).
constexpr double kFillSigma = -5.0;
constexpr index_t kFillK = 8;
std::vector<double> fill_block_spectrum() {
  std::vector<double> w{-5.1, -5.1, -4.9, -5.5, -4.5, -5.8, -6.0, -4.0};
  for (int i = 0; i < 25; ++i) {
    w.push_back(-6.08 - 0.08 * i);
    w.push_back(-3.92 + 0.08 * i);
  }
  return w;
}

template <class T>
hs::CsrMatrix<T> fill_pencil() {
  const index_t nx = 20, ny = 21, nz = 22, nl = nx * ny * nz;
  const hs::CsrMatrix<T> p = hs::test::prescribed_spectrum<T>(fill_block_spectrum(), 3, 41);
  std::vector<index_t> rows, cols;
  std::vector<T> vals;
  auto id = [&](index_t i, index_t j, index_t l) { return i + nx * (j + ny * l); };
  for (index_t l = 0; l < nz; ++l)
    for (index_t j = 0; j < ny; ++j)
      for (index_t i = 0; i < nx; ++i) {
        const index_t r = id(i, j, l);
        rows.push_back(r);
        cols.push_back(r);
        vals.push_back(T(6.0));
        auto off = [&](index_t q) {
          rows.push_back(r);
          cols.push_back(q);
          vals.push_back(T(-1.0));
        };
        if (i > 0) off(id(i - 1, j, l));
        if (i + 1 < nx) off(id(i + 1, j, l));
        if (j > 0) off(id(i, j - 1, l));
        if (j + 1 < ny) off(id(i, j + 1, l));
        if (l > 0) off(id(i, j, l - 1));
        if (l + 1 < nz) off(id(i, j, l + 1));
      }
  for (index_t i = 0; i < p.nrows; ++i)
    for (index_t q = p.rowptr[i]; q < p.rowptr[i + 1]; ++q) {
      rows.push_back(nl + i);
      cols.push_back(nl + p.colind[q]);
      vals.push_back(p.values[q]);
    }
  return hs::csr_from_coo<T>(nl + p.nrows, nl + p.nrows, rows, cols, vals);
}

template <class T>
struct Rig {
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  std::unique_ptr<hs::Logger> log;
  hs::driver::Ctx ctx;
  hs::driver::NearestCertifier<T> certifier;
  Rig(Solved<T>& s, const hs::driver::CertificateTimings& t, const std::string& log_path)
      : log(std::make_unique<hs::Logger>(0, log_path)),
        ctx{s.cfg, log.get(), rep, clock},
        certifier(ctx, *s.pencil, *s.fac, s.sigma, false, s.sol.report.scale, s.sol.report.norm2_a,
                  s.sol.report.norm2_b, t) {
    ctx.held = hs::driver::FactorHeld{s.sigma, false};
  }
  /// What definite.cpp does after an undecided check: the exact counts (on the certifier's
  /// counter when it has one), the report fields, record_exact.
  bool count_exactly(const hs::driver::NearestWindow& w, index_t found) {
    index_t nu_a = 0, nu_b = 0;
    if (!certifier.exact_counts(w.lower, w.upper, nu_a, nu_b)) return false;
    auto& c = rep.certificate;
    c.performed = true;
    c.lower = w.lower;
    c.upper = w.upper;
    c.count_inertia = nu_b - nu_a;
    c.count_found = found;
    c.passed = c.count_inertia == c.count_found;
    certifier.record_exact(true);
    return true;
  }
};

// Review finding: single counts above the found pairs triggered deflated reruns without a double
// count (yee-pec-box-1e5: 12 vs 10, two needless reruns, 11 s instead of 3). An injected
// overcount at b is now checked in double on the same counter: the counts disagree, the single
// counts are not used again, the (kept) double counts certify, and the statement says so once.
TYPED_TEST(CertintFixTyped, SingleOvercountIsCheckedByDoubleCounts) {
  using T = TypeParam;
  auto s = solved_laplacian<T>(hs::CertificateMode::Auto);
  const std::string path = ::testing::TempDir() + "/certint_fix_overcount.jsonl";
  {
    Rig<T> rig(*s, {}, path);
    rig.certifier.inject_single_miscount(0, 2);
    const auto out = rig.certifier.run(hs::CertificateMethod::ExactSingle, s->window, {s->all}, 0);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Undecided);
    EXPECT_FALSE(rig.rep.certificate.performed);  // no verdict of the single counts
    EXPECT_TRUE(has_warning(rig.rep, "disagree with the double counts"));
    ASSERT_TRUE(rig.count_exactly(s->window, 8));
    const auto& c = rig.rep.certificate;
    EXPECT_TRUE(c.passed);
    EXPECT_TRUE(c.rigorous);
    EXPECT_EQ(c.method, "exact");
    EXPECT_EQ(c.count_inertia, 8);
    EXPECT_EQ(occurrences(c.statement, "disagree with the double counts"), 1u) << c.statement;
    EXPECT_NE(c.statement.find("; therefore double-precision Sylvester counts (InertiaCounter)"), std::string::npos)
        << c.statement;
    EXPECT_EQ(c.statement.find("is not found"), std::string::npos) << c.statement;
    // single counts are out for the rest of the solve
    EXPECT_EQ(rig.certifier.choose(s->window, {s->all}).single_reason,
              "single counts could not decide in an earlier round");
  }
  // one double count pair (the check), kept for the exact counts: no second pair, no analysis
  EXPECT_EQ(events(path, "certifier_double_counts").size(), 1u);
  EXPECT_EQ(events(path, "inertia_counter").size(), 1u);
  const auto single = events(path, "single_counts");
  ASSERT_EQ(single.size(), 1u);
  EXPECT_EQ(single[0]["count"], 10);
}

// Per side of sigma the single counts must match the exact count nu(sigma) of the main
// factorization: a miscount by one at BOTH ends keeps nu(b) - nu(a) equal to the found pairs
// (the old code certified), but contradicts nu(sigma) - nu(a) - and the double counts decide.
TYPED_TEST(CertintFixTyped, SingleCountsPerSideMustMatchTheCountAtSigma) {
  using T = TypeParam;
  auto s = solved_laplacian<T>(hs::CertificateMode::Auto);
  const std::string path = ::testing::TempDir() + "/certint_fix_sides.jsonl";
  {
    Rig<T> rig(*s, {}, path);
    rig.certifier.inject_single_miscount(1, 1);
    const auto out = rig.certifier.run(hs::CertificateMethod::ExactSingle, s->window, {s->all}, 0);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Undecided);
    EXPECT_FALSE(rig.rep.certificate.performed);
    ASSERT_TRUE(rig.count_exactly(s->window, 8));
    EXPECT_TRUE(rig.rep.certificate.passed);
    EXPECT_EQ(rig.rep.certificate.method, "exact");
  }
  const auto single = events(path, "single_counts");
  ASSERT_EQ(single.size(), 1u);
  EXPECT_EQ(single[0]["count"], 8);
  EXPECT_EQ(single[0]["sides_ok"], false);
  EXPECT_GE(single[0]["nu_sigma"], 0);
  // Without the miscount the sides agree and the single counts certify.
  ASSERT_TRUE(s->fac->factorize(1.0, -s->sigma));
  Rig<T> rig(*s, {}, path + ".2");
  EXPECT_EQ(rig.certifier.run(hs::CertificateMethod::ExactSingle, s->window, {s->all}, 0).verdict,
            hs::driver::CheckOutcome::Verdict::Passed);
  EXPECT_EQ(rig.rep.certificate.method, "exact_single");
  EXPECT_NE(rig.rep.certificate.statement.find("on either side of sigma"), std::string::npos);
  EXPECT_EQ(events(path + ".2", "single_counts").at(0)["sides_ok"], true);
}

// auto on a 3-D-fill pencil (timings that make single counts the cheapest): a real deficit (one
// copy of the double eigenvalue removed) is confirmed by the double counts - verdict Missing, the
// statement names the auto prediction once (it was repeated when the counts decided after a
// deficit) - and an injected overcount defers to the double counts.
TEST(CertintFixCertifier, AutoSingleDeficitConfirmedAndStatedOnce) {
  auto s = solved<double>(fill_pencil<double>(), kFillSigma, kFillK, hs::CertificateMode::Auto);
  expect_same_set(s->sol.eigenvalues, nearest(fill_block_spectrum(), kFillSigma, kFillK), 1e-10);
  const hs::driver::CertificateTimings slow{10.0, 10.0, 4.0};
  auto dropped = s->all;
  std::vector<double> hints;
  const auto removed = hs::driver::drop_found_pairs(dropped, hints, s->sol.eigenvalues, s->sigma, 1);
  ASSERT_EQ(removed.size(), 1u);
  EXPECT_NEAR(removed[0], -5.1, 1e-10);
  {
    Rig<double> rig(*s, slow, ::testing::TempDir() + "/certint_fix_auto_deficit.jsonl");
    const auto ch = rig.certifier.choose(s->window, {dropped});
    ASSERT_EQ(ch.method, hs::CertificateMethod::ExactSingle) << ch.single_reason << " / " << ch.randomized_reason;
    const auto out = rig.certifier.check(s->window, {dropped}, 2);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Missing);
    EXPECT_EQ(out.missing, 1);
    const auto& c = rig.rep.certificate;
    EXPECT_EQ(c.method, "exact_single");
    EXPECT_FALSE(c.passed);
    EXPECT_EQ(c.statement.rfind("auto (round 2)", 0), 0u) << c.statement;
    EXPECT_NE(c.statement.find("the double counts on the same InertiaCounter agree"), std::string::npos);
    // last round: the exact counts (kept) give the final verdict
    ASSERT_TRUE(rig.count_exactly(s->window, kFillK - 1));
    EXPECT_FALSE(rig.rep.certificate.passed);
    EXPECT_EQ(occurrences(rig.rep.certificate.statement, "auto (round"), 1u) << rig.rep.certificate.statement;
    EXPECT_EQ(occurrences(rig.rep.certificate.statement, "exceed the 7 found pairs"), 1u);
  }
  ASSERT_TRUE(s->fac->factorize(1.0, -s->sigma));
  Rig<double> rig(*s, slow, ::testing::TempDir() + "/certint_fix_auto_overcount.jsonl");
  rig.certifier.inject_single_miscount(-1, 0);  // nu_s(a) one low: one more counted
  EXPECT_EQ(rig.certifier.check(s->window, {s->all}, 0).verdict, hs::driver::CheckOutcome::Verdict::Undecided);
  ASSERT_TRUE(rig.count_exactly(s->window, kFillK));
  EXPECT_TRUE(rig.rep.certificate.passed);
  EXPECT_EQ(occurrences(rig.rep.certificate.statement, "auto (round"), 1u) << rig.rep.certificate.statement;
  EXPECT_EQ(rig.rep.certificate.statement.find("is not found"), std::string::npos);
}

// The driver's exact_single path end to end (review: never executed by ctest): auto picks single
// counts on the 3-D-fill pencil, finds the removed copy missing (confirmed in double), reruns -
// which refactorizes K(sigma) after the counts released it - and certifies with single counts.
TEST(CertintFixDriver, AutoSingleCountsDetectRerunAndCertify) {
  const auto a = fill_pencil<double>();
  auto cfg = nearest_cfg(kFillSigma, kFillK, hs::CertificateMode::Auto);
  cfg.certificate_fault_drop = 1;
  cfg.block_size = 4;  // the first run finds both copies of -5.1: the fault removes one of them
  const std::string path = ::testing::TempDir() + "/certint_fix_driver_single.jsonl";
  hs::Solution<double> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, nullptr, cfg, &log);
  }
  const auto choices = events(path, "certificate_choice");
  ASSERT_FALSE(choices.empty());
  ASSERT_EQ(choices[0]["method"], "exact_single") << choices[0].dump();
  const auto& r = sol.report;
  EXPECT_TRUE(r.success) << r.certificate.statement;
  EXPECT_EQ(r.certificate.method, "exact_single");
  EXPECT_TRUE(r.certificate.passed);
  EXPECT_TRUE(r.certificate.rigorous);
  EXPECT_TRUE(has_warning(r, "certificate_fault_drop: 1 found pair(s) removed before the completeness check (-5.1)"));
  EXPECT_TRUE(has_warning(r, "the exact_single check finds"));
  EXPECT_TRUE(has_warning(r, "eigenvalue(s) missing near sigma; deflated rerun"));
  EXPECT_EQ(occurrences(r.certificate.statement, "auto (round"), 1u) << r.certificate.statement;
  EXPECT_EQ(r.factor_stats.factorizations, 2);  // K(sigma), and again for the rerun
  EXPECT_EQ(events(path, "single_counts").size(), 2u);
  EXPECT_EQ(events(path, "certifier_double_counts").size(), 1u);  // the deficit, checked once
  EXPECT_TRUE(events(path, "certificate").empty());               // no exact-count round
  expect_same_set(sol.eigenvalues, nearest(fill_block_spectrum(), kFillSigma, kFillK), 1e-10);
}

// ------------------------------------------------------------------- the exact path, as in P4

// Review finding: test_certint's "exact unchanged" compared exact with exact. The P4 invariants
// instead: counts at gap midpoints of the hints on the main object (inertia_counter = off: K(sigma)
// plus two factorizations per round) or on an InertiaCounter (K(sigma) only), the same window and
// counts either way, a rerun on a deficit, and none of the fast checks' work.
TEST(CertintFixExact, ExactPathKeepsItsP4Invariants) {
  const std::vector<double> w = gapped_spectrum();
  const auto a = hs::test::prescribed_spectrum<double>(w, 3, 91);
  for (const index_t drop : {0, 1}) {
    auto cfg = nearest_cfg(1.02, 10, hs::CertificateMode::Exact);
    cfg.deterministic = true;
    cfg.certificate_fault_drop = drop;
    hs::Solution<double> sol[2];
    for (int on = 0; on < 2; ++on) {
      cfg.inertia_counter = on ? hs::Toggle::On : hs::Toggle::Off;
      const std::string path = ::testing::TempDir() + "/certint_fix_exact_" + std::to_string(drop) +
                               std::to_string(on) + ".jsonl";
      {
        hs::Logger log(0, path);
        sol[on] = hs::solve(a, nullptr, cfg, &log);
      }
      const auto& r = sol[on].report;
      const auto& c = r.certificate;
      EXPECT_TRUE(r.success) << drop << on;
      EXPECT_TRUE(c.passed);
      EXPECT_EQ(c.method, "exact");
      EXPECT_TRUE(c.rigorous);
      EXPECT_EQ(c.failure_probability, 0.0);
      EXPECT_EQ(c.solves, 0);
      EXPECT_EQ(c.count_inertia, 10);
      EXPECT_EQ(c.count_found, 10);
      // window ends mid-way into the gaps around the wanted [0.9, 1.11] (nearest hints beyond)
      EXPECT_GT(c.lower, -1.0);
      EXPECT_LT(c.lower, 0.9);
      EXPECT_GT(c.upper, 1.14);
      EXPECT_LT(c.upper, 2.0);
      EXPECT_EQ(c.statement.rfind(on ? "double-precision Sylvester counts (InertiaCounter): nu(b) - nu(a) = 10"
                                     : "double-precision Sylvester counts (main factorization): nu(b) - nu(a) = 10",
                                  0),
                0u)
          << c.statement;
      // Every round but the last finds pairs missing (an unconverged Ritz hint far out can widen
      // the first window over the bulk in [-3, -1]: P4 then reruns, as here); the removed copy
      // is among them.
      const auto rounds = events(path, "certificate");
      ASSERT_GE(rounds.size(), drop ? 2u : 1u) << drop << on;
      ASSERT_LE(rounds.size(), 3u);
      for (std::size_t i = 0; i + 1 < rounds.size(); ++i)
        EXPECT_GT(rounds[i]["count_inertia"].get<int>(), rounds[i]["count_found"].get<int>()) << i;
      EXPECT_TRUE(rounds.back()["passed"].get<bool>());
      // K(sigma), K(a), K(b) per round on the main object, K(sigma) again per rerun: 3 per round;
      // on the counter only K(sigma) and its refactorization per rerun: 1 per round.
      EXPECT_EQ(r.factor_stats.factorizations, (on ? 1 : 3) * static_cast<index_t>(rounds.size())) << drop << on;
      for (const char* fast : {"certificate_choice", "randomized_check", "single_counts", "certifier_double_counts"})
        EXPECT_TRUE(events(path, fast).empty()) << fast;
      expect_same_set(sol[on].eigenvalues, nearest(w, 1.02, 10), 1e-10);
    }
    EXPECT_EQ(sol[0].report.certificate.lower, sol[1].report.certificate.lower);
    EXPECT_EQ(sol[0].report.certificate.upper, sol[1].report.certificate.upper);
    EXPECT_EQ(sol[0].eigenvalues, sol[1].eigenvalues);
  }
}

// ------------------------------------------------------- proven-only start scaling (verify.hpp)

// start_scaling_for(b, safety, estimate = false): what the driver's auto choice asks - a proven
// kappa or none, without the Lanczos estimate.
TEST(VerifyStartScaling, ProvenOnlyWithoutEstimate) {
  using T = double;
  const auto none = hs::start_scaling_for<T>(nullptr, 100.0, false);
  EXPECT_TRUE(none.rigorous);
  EXPECT_EQ(none.kappa, 1.0);
  hs::CsrMatrix<T> diag = hs::identity<T>(5);
  for (index_t i = 0; i < 5; ++i) diag.values[i] = 1.0 + i;
  const auto d = hs::start_scaling_for<T>(&diag, 100.0, false);
  EXPECT_TRUE(d.rigorous);
  EXPECT_EQ(d.kappa, 1.0);
  ASSERT_EQ(d.scaling.size(), 5u);
  EXPECT_DOUBLE_EQ(d.scaling[3], 0.5);
  // strictly diagonally dominant: the Gershgorin bound, the same with or without the estimate
  const auto hpd = hs::test::random_hpd<T>(60, 0.1, 7);
  const auto dd = hs::start_scaling_for<T>(&hpd, 100.0, false);
  EXPECT_TRUE(dd.rigorous);
  EXPECT_GE(dd.kappa, 1.0);
  EXPECT_TRUE(std::isfinite(dd.kappa));
  EXPECT_EQ(dd.kappa, hs::start_scaling_for<T>(&hpd, 100.0, true).kappa);
  // a Laplacian B (interior rows 4 = 4 x 1: not strictly dominant): no proven kappa
  const auto lap = hs::test::laplacian_2d<T>(6, 6);
  const auto g = hs::start_scaling_for<T>(&lap, 100.0, false);
  EXPECT_FALSE(g.rigorous);
  EXPECT_EQ(g.kappa, std::numeric_limits<double>::infinity());
  EXPECT_NE(g.note.find("not estimated"), std::string::npos);
  const auto est = hs::start_scaling_for<T>(&lap, 100.0, true);
  EXPECT_FALSE(est.rigorous);
  EXPECT_TRUE(std::isfinite(est.kappa));
  EXPECT_GE(est.kappa, 1.0);
}

}  // namespace
