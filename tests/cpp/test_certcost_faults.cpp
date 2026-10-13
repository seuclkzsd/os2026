// Robustness of the certificate's factorizations (P8 "certcost", src/driver/steps.cpp CountSource,
// src/driver/certificate.*, src/driver/definite.cpp): a count that fails - an InertiaCounter whose
// MUMPS factorization runs out of memory (INFO -19 under its ICNTL(23) cap), a count on the main
// factorization object, a counter predicted not to fit the memory left, the refactorization of a
// deflated rerun - never ends the solve. Each path is forced by SolverConfig::certificate_fault_count
// (its bits: internal.hpp kFault*), real and complex: counts move to the other source; with no
// source left the converged pairs are returned with the certificate performed but not completed.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/certificate.hpp"
#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

// A 7-point Dirichlet Laplacian on a 17 x 19 x 21 grid (n = 6783): large enough that an
// InertiaCounter needs more than the 1 MB working-memory cap of kFaultCounterCounts (a genuine
// MUMPS INFO -19), analytic eigenvalues (sums of 1-D ones; the three grid sizes keep the wanted ones
// simple). Complex T: the gauge transform D^H L D, D = diag(exp(i phi_v)), makes every coupling
// complex and keeps the spectrum.
constexpr index_t kNx = 17, kNy = 19, kNz = 21;
constexpr double kSigma = 0.9;
constexpr index_t kK = 10;
std::vector<double> spectrum() {
  std::vector<double> w;
  const auto wx = hs::test::laplacian_1d_eigenvalues(kNx), wy = hs::test::laplacian_1d_eigenvalues(kNy),
             wz = hs::test::laplacian_1d_eigenvalues(kNz);
  for (double a : wx)
    for (double b : wy)
      for (double c : wz) w.push_back(a + b + c);
  return w;
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
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

std::string warnings(const hs::SolveReport& r) {
  std::string s;
  for (const auto& w : r.warnings) s += w + "\n";
  return s;
}

hs::SolverConfig nearest_cfg(hs::Toggle counter, index_t faults) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = kSigma;
  cfg.target.k = kK;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.certificate = hs::CertificateMode::Exact;
  cfg.inertia_counter = counter;
  cfg.certificate_fault_count = faults;
  return cfg;
}

template <class T>
hs::CsrMatrix<T> matrix() {
  const auto id = [](index_t i, index_t j, index_t k) { return i + kNx * (j + kNy * k); };
  const index_t n = kNx * kNy * kNz;
  const auto phase = [](index_t v) { return 0.7 * static_cast<double>(v % 97); };
  std::vector<index_t> rows, cols;
  std::vector<T> vals;
  auto add = [&](index_t r, index_t c, double x) {
    rows.push_back(r);
    cols.push_back(c);
    if constexpr (hs::is_complex_v<T>)
      vals.push_back(T(x) * std::polar(1.0, phase(c) - phase(r)));
    else
      vals.push_back(T(x));
  };
  for (index_t k = 0; k < kNz; ++k)
    for (index_t j = 0; j < kNy; ++j)
      for (index_t i = 0; i < kNx; ++i) {
        const index_t v = id(i, j, k);
        add(v, v, 6.0);
        if (i > 0) add(v, id(i - 1, j, k), -1.0);
        if (i + 1 < kNx) add(v, id(i + 1, j, k), -1.0);
        if (j > 0) add(v, id(i, j - 1, k), -1.0);
        if (j + 1 < kNy) add(v, id(i, j + 1, k), -1.0);
        if (k > 0) add(v, id(i, j, k - 1), -1.0);
        if (k + 1 < kNz) add(v, id(i, j, k + 1), -1.0);
      }
  return hs::csr_from_coo<T>(n, n, rows, cols, vals);
}

template <class T>
void expect_certified(const hs::Solution<T>& sol, const std::string& source) {
  const auto& r = sol.report;
  const auto& c = r.certificate;
  EXPECT_TRUE(r.success) << warnings(r);
  EXPECT_TRUE(c.performed);
  EXPECT_TRUE(c.passed) << c.statement;
  EXPECT_TRUE(c.rigorous);
  EXPECT_EQ(c.method, "exact");
  EXPECT_EQ(c.failure_probability, 0.0);
  EXPECT_EQ(c.count_inertia, kK);
  EXPECT_EQ(c.count_found, kK);
  EXPECT_NE(c.statement.find("double-precision Sylvester counts (" + source + ")"), std::string::npos) << c.statement;
  expect_same_set(sol.eigenvalues, nearest(spectrum(), kSigma, kK), 1e-10);
}

// The pairs returned without a completed certificate are the converged nearest ones; the report
// says that nothing was proven.
template <class T>
void expect_not_completed(const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  const auto& c = r.certificate;
  EXPECT_FALSE(r.success);
  EXPECT_TRUE(c.performed);
  EXPECT_FALSE(c.passed);
  EXPECT_FALSE(c.rigorous);
  EXPECT_EQ(c.failure_probability, 1.0);
  EXPECT_EQ(c.method, "exact");
  EXPECT_EQ(c.count_inertia, -1);
  EXPECT_EQ(c.count_found, kK);
  EXPECT_LT(c.lower, kSigma);
  EXPECT_GT(c.upper, kSigma);
  EXPECT_EQ(c.note.rfind("not completed", 0), 0u) << c.note;
  EXPECT_NE(c.statement.find("NOT completed"), std::string::npos) << c.statement;
  EXPECT_NE(c.statement.find("returned uncertified"), std::string::npos) << c.statement;
  EXPECT_TRUE(has_warning(r, "certificate not completed: no inertia count possible")) << warnings(r);
  ASSERT_EQ(r.pairs.size(), static_cast<std::size_t>(kK));
  for (const auto& p : r.pairs) EXPECT_TRUE(p.accepted);
  expect_same_set(sol.eigenvalues, nearest(spectrum(), kSigma, kK), 1e-10);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s + r.time_certificate_s,
              r.time_total_s, 1e-12 * r.time_total_s);
}

// ---------------------------------------------------------------------------- configuration

TEST(DriverCountFailureConfig, ParseValidateRoundTrip) {
  EXPECT_EQ(hs::SolverConfig{}.certificate_fault_count, 0);
  EXPECT_EQ(hs::parse_config(R"({"certificate_fault_count": 13})").certificate_fault_count, 13);
  const auto j = nlohmann::json::parse(hs::config_to_json(hs::parse_config(R"({"certificate_fault_count": 5})")));
  EXPECT_EQ(j["certificate_fault_count"], 5);
  for (const char* bad : {R"({"certificate_fault_count": -1})", R"({"certificate_fault_count": 16})",
                          R"({"certificate_fault_count": 1.5})"})
    EXPECT_THROW(hs::parse_config(bad), hs::Error) << bad;
  EXPECT_EQ(hs::driver::kFaultCountMask,
            hs::driver::kFaultCounterCounts | hs::driver::kFaultMainCounts | hs::driver::kFaultCountAllowance |
                hs::driver::kFaultRefactorization);
}

// ---------------------------------------------------------------------- end to end, typed

template <class T>
class DriverCountFailureTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(DriverCountFailureTyped, Scalars);

// The counter's MUMPS factorization fails (INFO -19 under a 1 MB cap): the counts move to the
// main factorization object and certify.
TYPED_TEST(DriverCountFailureTyped, CounterFailureCountsOnTheMainObject) {
  using T = TypeParam;
  const auto a = matrix<T>();
  const auto sol = hs::solve(a, nullptr, nearest_cfg(hs::Toggle::On, hs::driver::kFaultCounterCounts));
  expect_certified(sol, "main factorization");
  EXPECT_TRUE(has_warning(sol.report, "the InertiaCounter could not count")) << warnings(sol.report);
  EXPECT_TRUE(has_warning(sol.report, "INFO(1)=-19")) << warnings(sol.report);
}

// A count on the main object fails: the counter counts.
TYPED_TEST(DriverCountFailureTyped, MainObjectFailureCountsOnTheCounter) {
  using T = TypeParam;
  const auto a = matrix<T>();
  const auto sol = hs::solve(a, nullptr, nearest_cfg(hs::Toggle::Off, hs::driver::kFaultMainCounts));
  expect_certified(sol, "InertiaCounter");
  EXPECT_TRUE(has_warning(sol.report, "the main factorization could not count")) << warnings(sol.report);
}

// A counter predicted not to fit the memory the budget leaves is not even tried.
TYPED_TEST(DriverCountFailureTyped, CounterOverTheAllowanceIsNotTried) {
  using T = TypeParam;
  const auto a = matrix<T>();
  const auto sol = hs::solve(a, nullptr, nearest_cfg(hs::Toggle::On, hs::driver::kFaultCountAllowance));
  expect_certified(sol, "main factorization");
  EXPECT_TRUE(has_warning(sol.report, "MB the memory budget leaves for a count")) << warnings(sol.report);
}

// No source can count: the converged pairs come back, uncertified, and the report says so.
TYPED_TEST(DriverCountFailureTyped, NoCountReturnsTheConvergedPairsUncertified) {
  using T = TypeParam;
  const auto a = matrix<T>();
  for (const auto& [counter, faults] :
       {std::pair{hs::Toggle::On, hs::driver::kFaultCounterCounts | hs::driver::kFaultMainCounts},
        std::pair{hs::Toggle::Off, hs::driver::kFaultMainCounts | hs::driver::kFaultCountAllowance}}) {
    const auto sol = hs::solve(a, nullptr, nearest_cfg(counter, faults));
    expect_not_completed(sol);
  }
  // A check that needs no count still certifies (B = I: the randomized check on the
  // factorization held at sigma).
  auto cfg = nearest_cfg(hs::Toggle::On, hs::driver::kFaultCounterCounts | hs::driver::kFaultMainCounts);
  cfg.certificate = hs::CertificateMode::Randomized;
  const auto rnd = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(rnd.report.success) << warnings(rnd.report);
  EXPECT_EQ(rnd.report.certificate.method, "randomized");
  EXPECT_TRUE(rnd.report.certificate.rigorous);
  expect_same_set(rnd.eigenvalues, nearest(spectrum(), kSigma, kK), 1e-10);
}

// The deflated rerun for a missing copy cannot refactorize at sigma: the pairs found are returned,
// the certificate says what is missing.
TYPED_TEST(DriverCountFailureTyped, RerunRefactorizationFailureReturnsThePairsFound) {
  using T = TypeParam;
  const auto a = matrix<T>();
  auto cfg = nearest_cfg(hs::Toggle::Auto, hs::driver::kFaultRefactorization);
  cfg.certificate_fault_drop = 1;
  const auto sol = hs::solve(a, nullptr, cfg);
  const auto& r = sol.report;
  const auto& c = r.certificate;
  EXPECT_FALSE(r.success);
  EXPECT_TRUE(c.performed);
  EXPECT_TRUE(has_warning(r, "is not possible (the refactorization at sigma failed")) << warnings(r);
  EXPECT_NE(c.statement.find("the deflated rerun for the missing pair(s) failed"), std::string::npos) << c.statement;
  // Either the counts find the removed pair missing in the window, or the window ends before it
  // and the solve is short of k pairs: never a claim of completeness for k pairs.
  if (c.passed) {
    EXPECT_TRUE(has_warning(r, "only 9 of 10 pairs found")) << warnings(r);
  } else {
    EXPECT_FALSE(c.rigorous);
    EXPECT_EQ(c.count_inertia, c.count_found + 1);
  }
  EXPECT_EQ(sol.eigenvalues.size(), static_cast<std::size_t>(kK - 1));
  for (const auto& p : r.pairs) EXPECT_TRUE(p.accepted);
  const auto want = nearest(spectrum(), kSigma, kK);
  for (double l : sol.eigenvalues) {
    double d = 1e300;
    for (double w : want) d = std::min(d, std::abs(l - w));
    EXPECT_LT(d, 1e-10) << l;
  }
}

// Window targets are built on counts: they move to the other source as well, and end with the
// error only when no source can count.
TYPED_TEST(DriverCountFailureTyped, WindowCountsFallBackOrFail) {
  using T = TypeParam;
  const auto a = matrix<T>();
  auto cfg = nearest_cfg(hs::Toggle::On, hs::driver::kFaultCounterCounts);
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 0.85;
  cfg.target.upper = 0.95;
  const auto iv = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(iv.report.success) << warnings(iv.report);
  EXPECT_TRUE(has_warning(iv.report, "the InertiaCounter could not count")) << warnings(iv.report);
  std::vector<double> inside;
  for (double l : spectrum())
    if (l >= cfg.target.lower && l <= cfg.target.upper) inside.push_back(l);
  ASSERT_GT(inside.size(), 3u);
  expect_same_set(iv.eigenvalues, inside, 1e-10);
  cfg.certificate_fault_count = hs::driver::kFaultCounterCounts | hs::driver::kFaultMainCounts;
  try {
    (void)hs::solve(a, nullptr, cfg);
    ADD_FAILURE() << "expected the count failure";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::OutOfMemoryBudget);
    EXPECT_NE(std::string(e.what()).find("no inertia count possible"), std::string::npos) << e.what();
  }
}

// ------------------------------------------------------------------- the certifier directly

// The certifier's own counter (single counts) fails in MUMPS: the check leaves the verdict to the
// exact counts, which do not try that counter again, and the failure is one of memory (so
// definite.cpp keeps CountSource from building another counter on the same pivot order).
TYPED_TEST(DriverCountFailureTyped, CertifierCounterFailureLeavesTheVerdictToTheCounts) {
  using T = TypeParam;
  const auto a = matrix<T>();
  hs::SolverConfig cfg = nearest_cfg(hs::Toggle::Auto, 0);
  const auto sol = hs::solve(a, nullptr, cfg);
  ASSERT_TRUE(sol.report.certificate.passed);
  const hs::ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -kSigma);
  ASSERT_TRUE(fac->factorize(1.0, -kSigma));
  hs::driver::Block<T> all;
  all.lambda = sol.eigenvalues;
  all.x = sol.eigenvectors;
  hs::driver::recompute_products(a, static_cast<const hs::CsrMatrix<T>*>(nullptr), all);
  double r = 0.0;
  for (double l : sol.eigenvalues) r = std::max(r, std::abs(l - kSigma));
  const hs::driver::NearestWindow w{kSigma, r, sol.report.certificate.lower, sol.report.certificate.upper};

  cfg.certificate_fault_count = hs::driver::kFaultCounterCounts;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx ctx{cfg, nullptr, rep, clock};
  ctx.held = hs::driver::FactorHeld{kSigma, false};
  hs::driver::NearestCertifier<T> certifier(ctx, pencil, *fac, kSigma, false, sol.report.scale, sol.report.norm2_a,
                                            sol.report.norm2_b, {});
  const auto out = certifier.run(hs::CertificateMethod::ExactSingle, w, {all}, 0);
  EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Undecided);
  EXPECT_FALSE(rep.certificate.performed);
  EXPECT_NE(certifier.counter_failure().find("single counts: out_of_memory_budget"), std::string::npos)
      << certifier.counter_failure();
  EXPECT_NE(certifier.counter_failure().find("INFO(1)=-19"), std::string::npos) << certifier.counter_failure();
  EXPECT_TRUE(certifier.counter_memory_failure());
  index_t nu_a = 0, nu_b = 0;
  EXPECT_FALSE(certifier.exact_counts(w.lower, w.upper, nu_a, nu_b));
  // The exact counts (here: by hand on the main object) give the verdict; the statement says why
  // the single counts did not.
  ASSERT_TRUE(fac->factorize(1.0, -w.lower));
  nu_a = fac->inertia().neg;
  ASSERT_TRUE(fac->factorize(1.0, -w.upper));
  nu_b = fac->inertia().neg;
  rep.certificate.performed = true;
  rep.certificate.count_inertia = nu_b - nu_a;
  rep.certificate.count_found = kK;
  rep.certificate.passed = nu_b - nu_a == kK;
  certifier.record_exact(false);
  EXPECT_TRUE(rep.certificate.passed);
  EXPECT_NE(rep.certificate.statement.find("single counts failed"), std::string::npos) << rep.certificate.statement;
  // With no count possible the report says the certificate was not completed.
  certifier.record_not_completed(w.lower, w.upper, kK, "test");
  EXPECT_FALSE(rep.certificate.passed);
  EXPECT_EQ(rep.certificate.count_inertia, -1);
  EXPECT_EQ(rep.certificate.count_found, kK);
}

// The genuine MUMPS failure behind kFaultCounterCounts: an InertiaCounter whose working memory is
// above its ICNTL(23) cap throws OutOfMemoryBudget with INFO(1) = -19 (the error the driver
// recovers from), and its estimate says so beforehand.
TYPED_TEST(DriverCountFailureTyped, CounterAboveItsCapFailsInMumps) {
  using T = TypeParam;
  const auto a = matrix<T>();
  const hs::ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -kSigma);
  fo.max_memory_mb = 1.0;
  hs::InertiaCounter<T> counter(*fac, fo);
  EXPECT_GT(counter.estimate().incore_mb, 1.0);
  try {
    (void)counter.count(1.0, -kSigma);
    ADD_FAILURE() << "expected INFO -19";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::OutOfMemoryBudget);
    EXPECT_NE(std::string(e.what()).find("INFO(1)=-19"), std::string::npos) << e.what();
    EXPECT_TRUE(hs::driver::count_failure(e));
  }
}

// --------------------------------------------------------------------------------- factor

template <class T>
class FactorInertiaCounterEstimateTyped : public ::testing::Test {};
TYPED_TEST_SUITE(FactorInertiaCounterEstimateTyped, Scalars);

// estimate(Single) analyses the single-precision instance (a smaller working memory than the double
// one's); counts afterwards are unchanged.
TYPED_TEST(FactorInertiaCounterEstimateTyped, SingleEstimateBeforeTheFirstCount) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(30, 29);
  const hs::ShiftedPencil<T> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -1.0);
  hs::InertiaCounter<T> counter(*fac, fo, /*defer_double_analysis=*/true);
  const hs::FactorEstimate es = counter.estimate(hs::FactorPrecision::Single);
  EXPECT_FALSE(counter.double_analysed());
  EXPECT_EQ(counter.stats().analyses, 1);
  const hs::FactorEstimate ed = counter.estimate(hs::FactorPrecision::Double);
  EXPECT_TRUE(counter.double_analysed());
  EXPECT_GT(es.incore_mb, 0.0);
  EXPECT_LE(es.incore_mb, ed.incore_mb);
  EXPECT_EQ(es.factor_entries, ed.factor_entries);  // the same pivot order
  for (const double s : {0.37, 2.9, 5.1}) {
    const hs::Inertia single = counter.count(1.0, -s, true, hs::FactorPrecision::Single);
    const hs::Inertia exact = hs::compute_inertia(pencil, 1.0, -s);
    ASSERT_TRUE(single.valid && exact.valid) << s;
    EXPECT_EQ(single.neg, exact.neg) << s;
  }
  EXPECT_EQ(counter.stats().analyses, 2);  // no third analysis for the counts
}

}  // namespace
