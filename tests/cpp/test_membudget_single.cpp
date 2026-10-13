// P7 safe single precision: a single-precision (or BLR) factor never yields a wrong answer. Its
// refined applications are checked (ShiftInvertOperator::last_apply_converged, the driver's
// DriverShiftInvert), an inexact factor flagged singular is not used, and an attempt that fails
// either way - or whose result is not accepted - is solved again with double factors. Small
// analogues of the P6 failures (results/p6_mixprec.md section 6.1: syn-mem27-46 - wanted pairs
// far nearer sigma than eps_single allows, syn-ghep-near-singular-real-1e5, the near-singular
// complex case flagged "exactly singular") must return valid results. factor_precision = auto
// takes single factors only when the double ones do not fit in core within the budget.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

hs::SolverConfig nearest(double sigma, index_t k, hs::FactorPrecision precision) {
  hs::SolverConfig cfg;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.factor_precision = precision;
  return cfg;
}

std::vector<double> nearest_of(std::vector<double> w, double sigma, std::size_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double x, double y) { return std::abs(x - sigma) < std::abs(y - sigma); });
  w.resize(std::min(k, w.size()));
  std::sort(w.begin(), w.end());
  return w;
}

void expect_same(std::vector<double> got, const std::vector<double>& want, double tol) {
  std::sort(got.begin(), got.end());
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol) << i;
}

template <class T>
void expect_valid(const hs::Solution<T>& sol) {
  EXPECT_TRUE(sol.report.success);
  for (const auto& p : sol.report.pairs) EXPECT_LE(p.backward_error, 1e-10);
  if (sol.report.certificate.performed) {
    EXPECT_TRUE(sol.report.certificate.passed);
  }
}

bool warned(const hs::SolveReport& r, const std::string& what) {
  for (const auto& m : r.warnings)
    if (m.find(what) != std::string::npos) return true;
  return false;
}

/// 2-D Laplacian shifted by -c (indefinite): delocalized eigenvectors, which single precision's
/// null-pivot test mostly misses (factor.hpp).
struct Grid {
  hs::CsrMatrix<double> a;
  std::vector<double> w;
};
Grid shifted_grid(index_t nx, index_t ny, double c) {
  Grid g{hs::test::laplacian_2d<double>(nx, ny), hs::test::laplacian_2d_eigenvalues(nx, ny)};
  for (index_t i = 0; i < g.a.nrows; ++i)
    for (index_t p = g.a.rowptr[i]; p < g.a.rowptr[i + 1]; ++p)
      if (g.a.colind[p] == i) g.a.values[p] -= c;
  for (auto& x : g.w) x -= c;
  return g;
}

/// Index i in [lo, hi) of the widest gap w[i] .. w[i + 1] (a shift there is well separated).
std::size_t widest_gap(const std::vector<double>& w, std::size_t lo, std::size_t hi) {
  std::size_t best = lo;
  for (std::size_t i = lo; i < hi; ++i)
    if (w[i + 1] - w[i] > w[best + 1] - w[best]) best = i;
  return best;
}

// ------------------------------------------------------------------------------ the operator check

// A single factor at a shift 1e-9 ||K|| from an eigenvalue (eps_single * cond(K) ~ 60): its
// refinement cannot converge, the operator says so, and the driver's operator throws instead of
// returning the operator of a perturbed matrix. At a shift in a wide gap it converges.
TEST(OpsShiftInvertConvergence, RefinementThatCannotConvergeIsReported) {
  auto g = shifted_grid(60, 61, 2.0);
  const double scale = 8.0;
  hs::ShiftedPencil<double> pencil(g.a, nullptr);
  hs::FactorOptions o;
  o.precision = hs::FactorPrecision::Single;
  o.threads = 2;
  hs::SolverConfig cfg;
  const index_t n = g.a.nrows;
  std::vector<double> x(static_cast<std::size_t>(2 * n)), y(x.size());
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd;
  for (auto& v : x) v = nd(rng);
  struct Shift {
    double sigma;
    bool converges;
  };
  const std::size_t j = widest_gap(g.w, 1000, 2500);
  for (const Shift s : {Shift{0.5 * (g.w[j] + g.w[j + 1]), true}, Shift{g.w[j] + 1e-9 * scale, false}}) {
    auto f = hs::make_mumps_factorization<double>(o);
    f->analyze(pencil, 1.0, -s.sigma);
    f->set_null_pivot_detection(false);  // the operator check must not rely on the pivot test
    ASSERT_TRUE(f->factorize(1.0, -s.sigma));
    hs::ShiftInvertOperator<double> op(*f, pencil, s.sigma, hs::driver::shift_invert_refinement(cfg, f->options()));
    op.apply(x.data(), n, y.data(), n, 2);
    EXPECT_EQ(op.last_apply_converged(), s.converges) << s.sigma << " rel " << op.last_relative_residual();
    hs::driver::DriverShiftInvert<double> checked(cfg, *f, pencil, s.sigma);
    if (s.converges) {
      EXPECT_NO_THROW(checked.apply(x.data(), n, y.data(), n, 2));
    } else {
      EXPECT_THROW(checked.apply(x.data(), n, y.data(), n, 2), hs::driver::InexactFactorFailure);
    }
  }
  // a double factor is never refined, and never fails the check
  o.precision = hs::FactorPrecision::Double;
  auto f = hs::make_mumps_factorization<double>(o);
  f->analyze(pencil, 1.0, -g.w[j]);
  f->set_null_pivot_detection(false);
  ASSERT_TRUE(f->factorize(1.0, -(g.w[j] + 1e-9 * scale)));
  EXPECT_EQ(hs::driver::shift_invert_refinement(cfg, f->options()).max_steps, 0);
  hs::driver::DriverShiftInvert<double> checked(cfg, *f, pencil, g.w[j] + 1e-9 * scale);
  EXPECT_NO_THROW(checked.apply(x.data(), n, y.data(), n, 2));
}

// ------------------------------------------------------------------------------ the driver

// Well-separated shift: single factors serve the whole run (no fallback) and give the double result.
TEST(DriverInexactFactors, SingleAtAWellSeparatedShiftStaysSingle) {
  auto g = shifted_grid(60, 61, 2.0);
  const std::size_t j = widest_gap(g.w, 1000, 2500);
  const double sigma = g.w[j] + 0.3 * (g.w[j + 1] - g.w[j]);
  const auto sol = hs::solve(g.a, nullptr, nearest(sigma, 10, hs::FactorPrecision::Single));
  expect_valid(sol);
  EXPECT_EQ(sol.report.precision.requested, hs::FactorPrecision::Single);
  EXPECT_EQ(sol.report.precision.used, hs::FactorPrecision::Single);
  EXPECT_FALSE(sol.report.precision.fallback);
  expect_same(sol.eigenvalues, nearest_of(g.w, sigma, 10), 1e-9);
}

// The syn-mem27-46 analogue: the wanted pairs lie 1e-9 ||K|| from sigma (delocalized: single's
// pivot test misses it, its refinement diverges). Before P7: pairs with eta up to 2.8e-5, the
// certificate failed. Now: the single attempt is abandoned and the double result returned.
TEST(DriverInexactFactors, SingleNearAnEigenvalueFallsBackToDouble) {
  auto g = shifted_grid(80, 81, 2.0);
  const double sigma = g.w[3000] + 1e-9 * 8.0;
  const auto sol = hs::solve(g.a, nullptr, nearest(sigma, 8, hs::FactorPrecision::Single));
  expect_valid(sol);
  EXPECT_TRUE(sol.report.precision.fallback);
  EXPECT_FALSE(sol.report.precision.fallback_reason.empty());
  EXPECT_GT(sol.report.precision.attempt_s, 0.0);
  EXPECT_EQ(sol.report.precision.used, hs::FactorPrecision::Double);
  EXPECT_TRUE(warned(sol.report, "abandoned"));
  expect_same(sol.eigenvalues, nearest_of(g.w, sigma, 8), 1e-9);
  // the phases still add up, the abandoned attempt included
  const auto& r = sol.report;
  EXPECT_NEAR(r.time_total_s, r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s +
                                  r.time_certificate_s, 1e-9);
  EXPECT_GE(r.time_total_s, r.precision.attempt_s);
}

// The near-singular complex analogue (prescribed spectrum, localized eigenvectors): the single
// factor - of the real embedding - is flagged singular at sigma. Before P7: "exactly singular"
// error (rc = 2). Now: double.
TEST(DriverInexactFactors, ComplexSingleFlaggedSingularFallsBackToDouble) {
  const index_t n = 3000;
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = -1.0 + 2.0 * static_cast<double>(i) / (n - 1);
  const double lambda = w[1700];
  const auto a = hs::test::prescribed_spectrum<cplx>(w, 8, 11);
  const double sigma = lambda + 1e-11;
  const auto sol = hs::solve(a, nullptr, nearest(sigma, 6, hs::FactorPrecision::Single));
  expect_valid(sol);
  EXPECT_TRUE(sol.report.precision.fallback);
  EXPECT_EQ(sol.report.precision.used, hs::FactorPrecision::Double);
  expect_same(sol.eigenvalues, nearest_of(w, sigma, 6), 1e-9);
}

// The syn-ghep-near-singular analogue: GHEP with an eigenvalue of the pencil 1e-9 from sigma.
TEST(DriverInexactFactors, GhepSingleNearAnEigenvalueFallsBackToDouble) {
  const index_t nx = 36, ny = 37, n = nx * ny;
  const auto a = hs::test::laplacian_2d<double>(nx, ny);
  std::vector<index_t> idx(static_cast<std::size_t>(n));
  std::vector<double> d(static_cast<std::size_t>(n));
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> u(1.0, 3.0);
  for (index_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i, d[static_cast<std::size_t>(i)] = u(rng);
  const auto b = hs::csr_from_coo<double>(n, n, idx, idx, d);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = w[600] + 1e-9 * (w.back() - w.front());
  const auto sol = hs::solve(a, &b, nearest(sigma, 6, hs::FactorPrecision::Single));
  expect_valid(sol);
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(sol.report.precision.fallback);
  expect_same(sol.eigenvalues, nearest_of(w, sigma, 6), 1e-9 * (1.0 + std::abs(sigma)));
  // the same in double never falls back
  const auto dbl = hs::solve(a, &b, nearest(sigma, 6, hs::FactorPrecision::Double));
  expect_valid(dbl);
  EXPECT_FALSE(dbl.report.precision.fallback);
}

// GHIEP (B indefinite) with single factors: either they serve (refined, checked) or the run falls
// back; the result is the double one.
TEST(DriverInexactFactors, GhiepSingleGivesTheDoubleResult) {
  const index_t nx = 50, ny = 41, n = nx * ny;
  auto a = hs::test::laplacian_2d<double>(nx, ny);
  std::vector<index_t> idx(static_cast<std::size_t>(n));
  std::vector<double> diag(static_cast<std::size_t>(n), 1.0);
  for (index_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
  diag[77] = -1.0;
  const auto b = hs::csr_from_coo<double>(n, n, idx, idx, diag);
  const auto dbl = hs::solve(a, &b, nearest(0.3, 8, hs::FactorPrecision::Double));
  const auto sgl = hs::solve(a, &b, nearest(0.3, 8, hs::FactorPrecision::Single));
  ASSERT_EQ(dbl.report.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(dbl.report.success);
  EXPECT_TRUE(sgl.report.success);
  ASSERT_EQ(sgl.eigenvalues.size(), dbl.eigenvalues.size());
  expect_same(sgl.eigenvalues, [&] {
    auto v = dbl.eigenvalues;
    std::sort(v.begin(), v.end());
    return v;
  }(), 1e-9);
}

// ------------------------------------------------------------------------------ precision auto

/// 3-D 7-point Dirichlet Laplacian on an m^3 grid.
hs::CsrMatrix<double> laplacian_3d(index_t m) {
  std::vector<index_t> r, c;
  std::vector<double> v;
  auto id = [&](index_t i, index_t j, index_t k) { return i + m * (j + m * k); };
  for (index_t k = 0; k < m; ++k)
    for (index_t j = 0; j < m; ++j)
      for (index_t i = 0; i < m; ++i) {
        const index_t p = id(i, j, k);
        r.push_back(p), c.push_back(p), v.push_back(6.0);
        if (i > 0) r.push_back(p), c.push_back(id(i - 1, j, k)), v.push_back(-1.0);
        if (i + 1 < m) r.push_back(p), c.push_back(id(i + 1, j, k)), v.push_back(-1.0);
        if (j > 0) r.push_back(p), c.push_back(id(i, j - 1, k)), v.push_back(-1.0);
        if (j + 1 < m) r.push_back(p), c.push_back(id(i, j + 1, k)), v.push_back(-1.0);
        if (k > 0) r.push_back(p), c.push_back(id(i, j, k - 1)), v.push_back(-1.0);
        if (k + 1 < m) r.push_back(p), c.push_back(id(i, j, k + 1)), v.push_back(-1.0);
      }
  const index_t n = m * m * m;
  return hs::csr_from_coo<double>(n, n, r, c, v);
}

std::vector<double> laplacian_3d_eigenvalues(index_t m) {
  const auto w1 = hs::test::laplacian_1d_eigenvalues(m);
  std::vector<double> w;
  for (double x : w1)
    for (double y : w1)
      for (double z : w1) w.push_back(x + y + z);
  std::sort(w.begin(), w.end());
  return w;
}

struct Estimates {
  hs::FactorEstimate dbl, sgl;
};
Estimates estimates(const hs::ShiftedPencil<double>& pencil, double sigma) {
  hs::FactorOptions o;
  o.threads = 2;
  o.ordering = 0;
  auto d = hs::make_mumps_factorization<double>(o);
  d->analyze(pencil, 1.0, -sigma);
  o.precision = hs::FactorPrecision::Single;
  auto s = hs::make_mumps_factorization<double>(o);
  s->set_pivot_order(d->pivot_order());
  s->analyze(pencil, 1.0, -sigma);
  return {d->estimate(), s->estimate()};
}

// choose_precision on the main analysis: double while it fits the cap (or out of core is forced),
// single when only single fits, double (out of core) when neither does.
TEST(DriverAutoPrecision, SingleOnlyWhenDoubleDoesNotFitInCoreButSingleDoes) {
  const auto a = laplacian_3d(24);
  hs::ShiftedPencil<double> pencil(a, nullptr);
  const double sigma = 0.77;
  const Estimates e = estimates(pencil, sigma);
  ASSERT_LT(e.sgl.incore_mb, 0.8 * e.dbl.incore_mb);
  struct Case {
    double cap_mb;
    hs::Toggle ooc;
    hs::FactorPrecision want;
    const char* decision;
  };
  const double between = 0.5 * (e.sgl.incore_mb + e.dbl.incore_mb);
  for (const Case cs : {Case{2.0 * e.dbl.incore_mb, hs::Toggle::Auto, hs::FactorPrecision::Double, "double: in-core"},
                        Case{between, hs::Toggle::Auto, hs::FactorPrecision::Single, "single: in-core estimates"},
                        Case{between, hs::Toggle::On, hs::FactorPrecision::Double, "double: out_of_core = on"},
                        Case{0.5 * e.sgl.incore_mb, hs::Toggle::Auto, hs::FactorPrecision::Double,
                             "double: in-core estimates"}}) {
    hs::SolverConfig cfg;
    cfg.factor_precision = hs::FactorPrecision::Auto;
    cfg.out_of_core = cs.ooc;
    cfg.threads = 2;
    hs::SolveReport rep;
    hs::driver::PhaseClock clock;
    hs::driver::Ctx c{cfg, nullptr, rep, clock};
    hs::FactorOptions o = hs::driver::factor_options_from(cfg, cs.cap_mb);
    EXPECT_EQ(o.precision, hs::FactorPrecision::Double);
    o.ordering = 0;
    auto fac = hs::make_mumps_factorization<double>(o);
    fac->analyze(pencil, 1.0, -sigma);
    hs::driver::choose_precision(c, fac, pencil, sigma);
    EXPECT_EQ(fac->options().precision, cs.want) << cs.cap_mb;
    EXPECT_EQ(rep.precision.decision.rfind(cs.decision, 0), 0u) << rep.precision.decision;
    if (cs.want == hs::FactorPrecision::Single) {
      EXPECT_EQ(fac->pivot_order().size(), static_cast<std::size_t>(a.nrows));
      EXPECT_TRUE(fac->factorize(1.0, -sigma));
    }
  }
}

// End to end: a budget whose factorization cap lies between the single and the double in-core
// estimates gives single factors (in core, valid result); a large budget gives double ones.
TEST(DriverAutoPrecision, AutoTakesSingleFactorsWhenOnlyTheyFitInCore) {
  const index_t m = 30;
  const auto a = laplacian_3d(m);
  const auto w = laplacian_3d_eigenvalues(m);
  const std::size_t j = widest_gap(w, 40, 100);  // a cube: many multiple eigenvalues
  const double sigma = w[j] + 0.3 * (w[j + 1] - w[j]);  // not mid-way: equidistant multiplets would tie
  hs::ShiftedPencil<double> pencil(a, nullptr);
  const Estimates e = estimates(pencil, sigma);
  auto cfg = nearest(sigma, 8, hs::FactorPrecision::Auto);
  cfg.ordering = 0;
  cfg.memory_budget_gb = 8.0;
  const auto big = hs::solve(a, nullptr, cfg);
  expect_valid(big);
  EXPECT_EQ(big.report.precision.used, hs::FactorPrecision::Double);
  EXPECT_FALSE(big.report.factor_options.out_of_core);
  // nearer the double estimate: what the solve allocates before its plan (a few MB) lowers the cap
  const double cap_mb = e.sgl.incore_mb + 0.7 * (e.dbl.incore_mb - e.sgl.incore_mb);
  ASSERT_GT(cap_mb - e.sgl.incore_mb, 15.0) << "the window between the estimates is too narrow";
  const auto reserve = hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Standard, a.nrows);
  const double krylov = static_cast<double>(a.nrows) * 8.0 * static_cast<double>(reserve.columns);
  const double rss = static_cast<double>(std::max<std::int64_t>(hs::current_rss_bytes(), 0));
  cfg.memory_budget_gb = (rss + krylov + cap_mb * 1e6 / 0.95) / 1073741824.0;
  const auto small = hs::solve(a, nullptr, cfg);
  expect_valid(small);
  EXPECT_GT(small.report.factor_options.max_memory_mb, e.sgl.incore_mb)
      << "estimates " << e.dbl.incore_mb << " / " << e.sgl.incore_mb << " MB, wanted cap " << cap_mb;
  EXPECT_LT(small.report.factor_options.max_memory_mb, e.dbl.incore_mb);
  EXPECT_EQ(small.report.precision.requested, hs::FactorPrecision::Auto);
  EXPECT_EQ(small.report.precision.used, hs::FactorPrecision::Single) << small.report.precision.decision;
  EXPECT_FALSE(small.report.factor_options.out_of_core);
  EXPECT_FALSE(small.report.precision.fallback) << small.report.precision.fallback_reason;
  expect_same(small.eigenvalues, nearest_of(w, sigma, 8), 1e-9);
}

}  // namespace
