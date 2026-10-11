// Tests of the completeness-check integration (module certint, src/driver/certificate.*): the
// certificate = exact | randomized | auto | off configuration and its report, the randomized check
// and single-precision counts in the nearest path, detection of an artificially removed copy of a
// multiple eigenvalue (SolverConfig::certificate_fault_drop) with recovery by the deflated rerun,
// propagation of the rigorous flag (proven vs estimated kappa of B), and the exact path unchanged.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/certificate.hpp"
#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
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

void expect_times_add_up(const hs::SolveReport& r) {
  for (double t : {r.time_setup_s, r.time_factor_s, r.time_iterate_s, r.time_verify_s, r.time_certificate_s})
    EXPECT_GE(t, 0.0);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s + r.time_certificate_s,
              r.time_total_s, 1e-12 * r.time_total_s);
  // The checks' own wall time is the exclusive certificate phase of the definite path.
  EXPECT_NEAR(r.certificate.seconds, r.time_certificate_s, 1e-12 + 1e-12 * r.time_certificate_s);
}

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

// Spectrum with a wide gap around the wanted set: the 10 eigenvalues nearest sigma = 1.02 are a
// 4-fold 1.0 and six simple ones within 0.12 of sigma; nothing else lies within 0.98 of sigma
// (the randomized check then plans a few block steps), a bulk in [2, 5] and [-3, -1].
constexpr double kSigma = 1.02;
constexpr index_t kK = 10;
std::vector<double> gapped_spectrum() {
  std::vector<double> w{1.0, 1.0, 1.0, 1.0, 0.97, 1.05, 1.08, 0.93, 1.11, 0.9};
  const int bulk = 400;
  for (int i = 0; i < bulk; ++i) w.push_back(2.0 + 3.0 * i / bulk);
  for (int i = 0; i < bulk / 4; ++i) w.push_back(-1.0 - 2.0 * i / (bulk / 4));
  return w;
}

template <class T>
hs::CsrMatrix<T> gapped_matrix() {
  return hs::test::prescribed_spectrum<T>(gapped_spectrum(), 3, 91);
}

/// Same spectrum as a GHEP: A = D^{1/2} A0 D^{1/2}, B = D (positive diagonal), so that
/// (A, B) has the eigenvalues of A0.
template <class T>
std::pair<hs::CsrMatrix<T>, hs::CsrMatrix<T>> diagonal_pencil() {
  hs::CsrMatrix<T> a = gapped_matrix<T>();
  const index_t n = a.nrows;
  std::vector<double> d(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) d[i] = 1.0 + 2.0 * static_cast<double>((i * 37) % 101) / 100.0;
  for (index_t i = 0; i < n; ++i)
    for (index_t q = a.rowptr[i]; q < a.rowptr[i + 1]; ++q) a.values[q] *= T(std::sqrt(d[i] * d[a.colind[q]]));
  hs::CsrMatrix<T> b = hs::identity<T>(n);
  for (index_t i = 0; i < n; ++i) b.values[i] = T(d[i]);
  return {std::move(a), std::move(b)};
}

/// A GHEP whose B is HPD but not diagonally dominant (start_scaling_for can only estimate its
/// kappa): blockdiag(1000 L, A0) x = lambda blockdiag(L, I) x with L a 2-D Dirichlet Laplacian
/// (interior rows: diagonal 4, four -1): eigenvalues 1000 (|L| copies, far away) and those of A0.
template <class T>
std::pair<hs::CsrMatrix<T>, hs::CsrMatrix<T>> general_b_pencil() {
  const hs::CsrMatrix<T> a0 = gapped_matrix<T>();
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

// ------------------------------------------------------------------------- configuration

TEST(DriverCertificateConfig, ModesParseValidateAndRoundTrip) {
  const hs::SolverConfig def;
  EXPECT_EQ(def.certificate, hs::CertificateMode::Exact);  // today's behaviour by default
  EXPECT_EQ(def.certificate_delta, 1e-12);
  EXPECT_EQ(def.certificate_fault_drop, 0);
  const auto j = nlohmann::json::parse(hs::config_to_json(def));
  EXPECT_EQ(j["certificate"], "exact");
  EXPECT_EQ(j["certificate_delta"], 1e-12);
  EXPECT_EQ(j["certificate_fault_drop"], 0);
  const std::pair<const char*, hs::CertificateMode> modes[] = {{"exact", hs::CertificateMode::Exact},
                                                               {"randomized", hs::CertificateMode::Randomized},
                                                               {"auto", hs::CertificateMode::Auto},
                                                               {"off", hs::CertificateMode::Off}};
  for (const auto& [name, mode] : modes) {
    const auto c = hs::parse_config(std::string(R"({"certificate": ")") + name + R"(", "certificate_delta": 1e-6})");
    EXPECT_EQ(c.certificate, mode) << name;
    EXPECT_EQ(c.certificate_delta, 1e-6);
    const auto again = hs::parse_config(hs::config_to_json(c));
    EXPECT_EQ(again.certificate, mode) << name;
    EXPECT_EQ(nlohmann::json::parse(hs::config_to_json(again))["certificate"], name);
  }
  EXPECT_EQ(hs::parse_config(R"({"certificate_fault_drop": 2})").certificate_fault_drop, 2);
  auto bad = [](const char* doc) {
    try {
      (void)hs::parse_config(doc);
      return false;
    } catch (const hs::Error& e) {
      return e.code() == hs::Error::Code::InvalidInput;
    }
  };
  EXPECT_TRUE(bad(R"({"certificate": "fast"})"));
  EXPECT_TRUE(bad(R"({"certificate": true})"));
  EXPECT_TRUE(bad(R"({"certificate_delta": 0})"));
  EXPECT_TRUE(bad(R"({"certificate_delta": 1})"));
  EXPECT_TRUE(bad(R"({"certificate_delta": "1e-6"})"));
  EXPECT_TRUE(bad(R"({"certificate_fault_drop": -1})"));
  EXPECT_TRUE(bad(R"({"certificate_fault_drop": 1.5})"));
}

TEST(DriverCertificateConfig, ReportStatesWhatWasProven) {
  hs::SolveReport r;
  r.certificate.performed = true;
  r.certificate.method = "randomized";
  r.certificate.rigorous = false;
  r.certificate.failure_probability = 3e-13;
  r.certificate.solves = 48;
  r.certificate.seconds = 0.25;
  r.certificate.statement = "P(...) <= 3e-13; NOT RIGOROUS";
  const auto j = nlohmann::json::parse(r.to_json());
  const auto& c = j["certificate"];
  EXPECT_EQ(c["method"], "randomized");
  EXPECT_EQ(c["rigorous"], false);
  EXPECT_EQ(c["failure_probability"], 3e-13);
  EXPECT_EQ(c["solves"], 48);
  EXPECT_EQ(c["seconds"], 0.25);
  EXPECT_EQ(c["statement"], "P(...) <= 3e-13; NOT RIGOROUS");
  // A default report claims nothing.
  const auto d = nlohmann::json::parse(hs::SolveReport{}.to_json())["certificate"];
  EXPECT_EQ(d["method"], "none");
  EXPECT_EQ(d["rigorous"], false);
  EXPECT_EQ(d["failure_probability"], 1.0);
}

// The single-count path pays its own analyses (a fresh InertiaCounter: double instance at
// construction, single one at the first count); < 0 keeps the counter's analysis (P4 behaviour).
TEST(VerifyCertificateCost, SingleAnalysesAreChargedToTheSinglePath) {
  hs::CertificateCostInputs in;
  in.n = 100000;
  in.deflated = 10;
  in.factorize_s = 1.0;
  in.counter_analysis_s = 0.1;
  in.single_valid = true;
  in.apply1_s = 0.1;
  const double counts = 2.0 * in.single_ratio * in.count_ratio * in.factorize_s;
  EXPECT_DOUBLE_EQ(hs::certificate_cost(in).exact_single_s, 0.1 + counts);
  in.single_analysis_s = 0.2;
  const auto est = hs::certificate_cost(in);
  EXPECT_DOUBLE_EQ(est.exact_single_s, 0.2 + counts);
  EXPECT_DOUBLE_EQ(est.exact_s, std::min(2.0, 0.1 + 2.0 * in.count_ratio));  // unchanged
  in.single_analysis_s = 0.0;
  EXPECT_DOUBLE_EQ(hs::certificate_cost(in).exact_single_s, counts);
  in.single_analysis_s = std::nan("");
  EXPECT_THROW((void)hs::certificate_cost(in), hs::Error);
}

// ------------------------------------------------------------------------------ end to end

// The exact mode is the default and the P4 path: same factorizations, counts and pairs as a run
// with the default configuration (bitwise: deterministic factorizations), and the new report
// fields say what the counts proved.
TEST(DriverCertificate, ExactIsTheUnchangedDefault) {
  const auto a = gapped_matrix<double>();
  auto cfg = nearest_cfg(kSigma, kK, hs::CertificateMode::Exact);
  cfg.deterministic = true;
  cfg.inertia_counter = hs::Toggle::Off;  // counts on the main object: 3 factorizations
  hs::SolverConfig def = cfg;
  def.certificate = hs::SolverConfig{}.certificate;
  const auto ex = hs::solve(a, nullptr, cfg);
  const auto df = hs::solve(a, nullptr, def);
  ASSERT_TRUE(ex.report.success);
  EXPECT_EQ(ex.eigenvalues, df.eigenvalues);
  EXPECT_EQ(ex.report.factor_stats.factorizations, df.report.factor_stats.factorizations);
  EXPECT_GE(ex.report.factor_stats.factorizations, 3);  // K(sigma), K(a), K(b) (+ 3 per rerun)
  EXPECT_EQ(ex.report.op_applies, df.report.op_applies);
  EXPECT_EQ(ex.report.warnings, df.report.warnings);
  for (const auto& w : ex.report.warnings) std::printf("exact run warning: %s\n", w.c_str());
  const auto& c = ex.report.certificate;
  EXPECT_EQ(c.count_inertia, df.report.certificate.count_inertia);
  EXPECT_EQ(c.lower, df.report.certificate.lower);
  EXPECT_EQ(c.upper, df.report.certificate.upper);
  EXPECT_TRUE(c.passed);
  EXPECT_EQ(c.method, "exact");
  EXPECT_TRUE(c.rigorous);
  EXPECT_EQ(c.failure_probability, 0.0);
  EXPECT_EQ(c.solves, 0);
  EXPECT_NE(c.statement.find("double-precision Sylvester counts"), std::string::npos);
  expect_times_add_up(ex.report);
  expect_same_set(ex.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
}

template <class T>
class DriverCertificateTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(DriverCertificateTyped, Scalars);

// B = I: the randomized check certifies on the factorization held at sigma - one factorization in
// the whole solve - with a rigorous failure bound.
TYPED_TEST(DriverCertificateTyped, RandomizedCertifiesWithoutCountFactorizations) {
  using T = TypeParam;
  const auto a = gapped_matrix<T>();
  const auto cfg = nearest_cfg(kSigma, kK, hs::CertificateMode::Randomized);
  const auto sol = hs::solve(a, nullptr, cfg);
  const auto& c = sol.report.certificate;
  EXPECT_TRUE(sol.report.success);
  EXPECT_TRUE(c.performed);
  EXPECT_TRUE(c.passed);
  EXPECT_EQ(c.method, "randomized");
  EXPECT_TRUE(c.rigorous);
  EXPECT_LE(c.failure_probability, cfg.certificate_delta);
  EXPECT_GT(c.solves, 0);
  EXPECT_EQ(c.count_inertia, -1);
  EXPECT_EQ(c.count_found, kK);  // the claim disc holds exactly the wanted pairs
  EXPECT_LT(c.lower, kSigma - 0.12);
  EXPECT_GT(c.upper, kSigma + 0.12);
  EXPECT_NE(c.statement.find("P(pass and more than"), std::string::npos);
  EXPECT_EQ(sol.report.factor_stats.factorizations, 1);  // K(sigma) only
  expect_times_add_up(sol.report);
  EXPECT_GT(sol.report.time_certificate_s, 0.0);
  expect_same_set(sol.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
}

// One copy of the 4-fold eigenvalue removed from the found set before the check: every mode
// detects it, the deflated rerun recovers it, and the final check passes.
TYPED_TEST(DriverCertificateTyped, RemovedCopyIsDetectedAndRecovered) {
  using T = TypeParam;
  const auto a = gapped_matrix<T>();
  for (const auto mode : {hs::CertificateMode::Exact, hs::CertificateMode::Randomized, hs::CertificateMode::Auto}) {
    auto cfg = nearest_cfg(kSigma, kK, mode);
    cfg.certificate_fault_drop = 1;
    const auto sol = hs::solve(a, nullptr, cfg);
    const auto& r = sol.report;
    EXPECT_TRUE(r.success) << static_cast<int>(mode);
    EXPECT_TRUE(r.certificate.passed) << static_cast<int>(mode);
    EXPECT_TRUE(r.certificate.rigorous) << static_cast<int>(mode);
    EXPECT_TRUE(has_warning(r, "certificate_fault_drop: 1 found pair(s) removed")) << static_cast<int>(mode);
    EXPECT_TRUE(has_warning(r, "deflated rerun")) << static_cast<int>(mode);
    if (mode == hs::CertificateMode::Randomized) {
      EXPECT_EQ(r.certificate.method, "randomized");
      EXPECT_TRUE(has_warning(r, "the randomized check finds"));
      EXPECT_EQ(r.factor_stats.factorizations, 1);  // the rerun reuses the factorization at sigma
    }
    expect_same_set(sol.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
    expect_times_add_up(r);
  }
}

// auto never claims less than a certificate: its pick (timing-dependent) is rigorous and the
// statement gives the predictions behind it.
TYPED_TEST(DriverCertificateTyped, AutoStatesItsChoiceAndProvesIt) {
  using T = TypeParam;
  const auto a = gapped_matrix<T>();
  const auto sol = hs::solve(a, nullptr, nearest_cfg(kSigma, kK, hs::CertificateMode::Auto));
  const auto& c = sol.report.certificate;
  EXPECT_TRUE(sol.report.success);
  EXPECT_TRUE(c.passed);
  EXPECT_TRUE(c.rigorous);
  // "auto (round <r>): predicted ...", r > 0 when a first run missed a copy (rerun)
  EXPECT_EQ(c.statement.rfind("auto (round ", 0), 0u) << c.statement;
  EXPECT_NE(c.statement.find("): predicted exact counts "), std::string::npos) << c.statement;
  EXPECT_TRUE(c.method == "exact" || c.method == "exact_single" || c.method == "randomized") << c.method;
  expect_same_set(sol.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
  expect_times_add_up(sol.report);
}

// Diagonal B: S = B^{-1/2} makes kappa = 1, proven - the randomized result is a certificate.
TYPED_TEST(DriverCertificateTyped, DiagonalBIsRigorous) {
  using T = TypeParam;
  const auto [a, b] = diagonal_pencil<T>();
  const auto sol = hs::solve(a, &b, nearest_cfg(kSigma, kK, hs::CertificateMode::Randomized));
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.certificate.method, "randomized");
  EXPECT_TRUE(sol.report.certificate.rigorous);
  EXPECT_FALSE(has_warning(sol.report, "SCREENING"));
  expect_same_set(sol.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
}

TEST(DriverCertificate, OffAndWindowTargets) {
  const auto a = gapped_matrix<double>();
  const auto off = hs::solve(a, nullptr, nearest_cfg(kSigma, kK, hs::CertificateMode::Off));
  EXPECT_FALSE(off.report.certificate.performed);
  EXPECT_EQ(off.report.certificate.method, "none");
  EXPECT_EQ(off.report.certificate.note, "disabled by configuration");
  EXPECT_EQ(off.report.time_certificate_s, 0.0);
  expect_same_set(off.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
  // Windows count exactly whatever the mode asks (their count points are cuts and shifts).
  auto cfg = nearest_cfg(kSigma, kK, hs::CertificateMode::Randomized);
  cfg.target.mode = hs::TargetMode::Interval;
  cfg.target.lower = 0.8;
  cfg.target.upper = 1.5;
  const auto iv = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(iv.report.success);
  EXPECT_EQ(iv.report.certificate.method, "exact");
  EXPECT_TRUE(iv.report.certificate.rigorous);
  EXPECT_EQ(iv.report.certificate.failure_probability, 0.0);
  EXPECT_TRUE(has_warning(iv.report, "certificate=randomized applies to nearest targets"));
  expect_same_set(iv.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
  expect_times_add_up(iv.report);
}

// ------------------------------------------------------------------- the certifier directly

/// A solved nearest problem and the pieces the certifier works on: the found pairs as a Block,
/// the exact certificate's window, and the main factorization at sigma.
template <class T>
struct Solved {
  hs::CsrMatrix<T> a, b;
  bool has_b = false;
  double sigma = 0.0;
  hs::SolverConfig cfg;
  hs::Solution<T> sol;
  std::unique_ptr<hs::ShiftedPencil<T>> pencil;
  std::unique_ptr<hs::SparseFactorization<T>> fac;
  hs::driver::Block<T> all;
  hs::driver::NearestWindow window;
  const hs::CsrMatrix<T>* bp() const { return has_b ? &b : nullptr; }
};

/// Solves (a, b) nearest k to sigma with exact counts, then sets up the pieces with `mode`.
template <class T>
std::unique_ptr<Solved<T>> solved(hs::CsrMatrix<T> a, const hs::CsrMatrix<T>* b, double sigma, index_t k,
                                  hs::CertificateMode mode) {
  auto s = std::make_unique<Solved<T>>();
  s->a = std::move(a);
  if (b) {
    s->b = *b;
    s->has_b = true;
  }
  s->sigma = sigma;
  s->cfg = nearest_cfg(sigma, k, hs::CertificateMode::Exact);
  s->sol = hs::solve(s->a, s->bp(), s->cfg);
  EXPECT_TRUE(s->sol.report.certificate.passed);
  s->cfg.certificate = mode;
  s->pencil = std::make_unique<hs::ShiftedPencil<T>>(s->a, s->bp());
  hs::FactorOptions fo;
  fo.threads = 2;
  fo.ordering = 0;  // AMD
  s->fac = hs::make_mumps_factorization<T>(fo);
  s->fac->analyze(*s->pencil, 1.0, -sigma);
  EXPECT_TRUE(s->fac->factorize(1.0, -sigma));
  s->all.lambda = s->sol.eigenvalues;
  s->all.x = s->sol.eigenvectors;
  hs::driver::recompute_products(s->a, s->bp(), s->all);
  double r = 0.0;
  for (double l : s->sol.eigenvalues) r = std::max(r, std::abs(l - sigma));
  s->window = {sigma, r, s->sol.report.certificate.lower, s->sol.report.certificate.upper};
  return s;
}

template <class T>
std::unique_ptr<Solved<T>> solved_laplacian(hs::CertificateMode mode) {
  // 2-D Laplacian shifted to be indefinite (nx != ny: simple eigenvalues), nearest 8 to 0.3.
  auto a = hs::test::laplacian_2d<T>(40, 37);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t q = a.rowptr[i]; q < a.rowptr[i + 1]; ++q)
      if (a.colind[q] == i) a.values[q] -= T(4.0);
  return solved<T>(std::move(a), nullptr, 0.3, 8, mode);
}

template <class T>
struct CertifierRig {
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx ctx;
  hs::driver::NearestCertifier<T> certifier;
  CertifierRig(Solved<T>& s, const hs::driver::CertificateTimings& t)
      : ctx{s.cfg, nullptr, rep, clock},
        certifier(ctx, *s.pencil, *s.fac, s.sigma, false, s.sol.report.scale, s.sol.report.norm2_a,
                  s.sol.report.norm2_b, t) {
    ctx.held = hs::driver::FactorHeld{s.sigma, false};
  }
};

// A general B without diagonal dominance: kappa is only estimated. The randomized check then
// gives a SCREENING result (rigorous = false, warning, statement) - here on a window with the
// known gap of the spectrum (nothing within 0.98 of sigma beyond the 10 wanted): end to end, an
// unconverged Ritz hint can narrow the driver's window so that no plan fits and the counts decide
// (then the statement says so). auto never takes the randomized check with such a kappa.
TYPED_TEST(DriverCertificateTyped, EstimatedKappaIsScreeningOnly) {
  using T = TypeParam;
  auto [a, b] = general_b_pencil<T>();
  auto s = solved<T>(a, &b, kSigma, kK, hs::CertificateMode::Randomized);
  expect_same_set(s->sol.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
  {
    CertifierRig<T> rig(*s, {});
    hs::driver::NearestWindow w = s->window;
    w.lower = kSigma - 0.5;
    w.upper = kSigma + 0.5;
    const auto out = rig.certifier.run(hs::CertificateMethod::Randomized, w, s->all, 0);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Passed) << rig.rep.certificate.statement;
    const auto& c = rig.rep.certificate;
    EXPECT_EQ(c.method, "randomized");
    EXPECT_TRUE(c.passed);
    EXPECT_FALSE(c.rigorous);
    EXPECT_LE(c.failure_probability, 1e-12);  // the bound under the assumed kappa
    EXPECT_NE(c.statement.find("NOT RIGOROUS"), std::string::npos) << c.statement;
    EXPECT_NE(c.statement.find("SCREENING only"), std::string::npos) << c.statement;
    EXPECT_TRUE(has_warning(rig.rep, "SCREENING"));
  }
  // End to end: whatever decided, a randomized verdict is never rigorous here.
  const auto rnd = hs::solve(a, &b, nearest_cfg(kSigma, kK, hs::CertificateMode::Randomized));
  EXPECT_EQ(rnd.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(rnd.report.certificate.passed);
  if (rnd.report.certificate.method == "randomized") {
    EXPECT_FALSE(rnd.report.certificate.rigorous);
    EXPECT_TRUE(has_warning(rnd.report, "SCREENING"));
  } else {
    EXPECT_EQ(rnd.report.certificate.method, "exact");
    EXPECT_TRUE(rnd.report.certificate.rigorous);
    EXPECT_EQ(rnd.report.certificate.statement.rfind("randomized check", 0), 0u) << rnd.report.certificate.statement;
  }
  expect_same_set(rnd.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
  const auto aut = hs::solve(a, &b, nearest_cfg(kSigma, kK, hs::CertificateMode::Auto));
  EXPECT_TRUE(aut.report.success);
  EXPECT_EQ(aut.report.certificate.method, "exact");  // single counts need B = I or diagonal B
  EXPECT_TRUE(aut.report.certificate.rigorous);
  EXPECT_FALSE(has_warning(aut.report, "SCREENING"));
  expect_same_set(aut.eigenvalues, nearest(gapped_spectrum(), kSigma, kK), 1e-10);
}

// Single-precision counts on the exact certificate's window: the rule holds (B = I, wide margins),
// the counts equal the found pairs - certified; one pair left out of the found set - not.
TYPED_TEST(DriverCertificateTyped, SingleCountsCertifyTheWindowOrDeferToExactCounts) {
  using T = TypeParam;
  auto s = solved_laplacian<T>(hs::CertificateMode::Auto);
  ASSERT_TRUE(s->sol.report.certificate.passed);
  {
    CertifierRig<T> rig(*s, {});
    const auto out = rig.certifier.run(hs::CertificateMethod::ExactSingle, s->window, s->all, 0);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Passed);
    const auto& c = rig.rep.certificate;
    EXPECT_EQ(c.method, "exact_single");
    EXPECT_TRUE(c.passed);
    EXPECT_TRUE(c.rigorous);
    EXPECT_EQ(c.count_inertia, 8);
    EXPECT_EQ(c.count_found, 8);
    EXPECT_EQ(c.failure_probability, 0.0);
    EXPECT_NE(c.statement.find("single-precision Sylvester counts"), std::string::npos);
    EXPECT_GT(rig.clock.seconds(hs::driver::Phase::Certificate), 0.0);
    EXPECT_FALSE(s->fac->inertia().valid);  // the main factors were released for the counts
  }
  // One pair fewer: the counts (8) exceed the found pairs (7), no certificate; the exact counts
  // that follow say why.
  ASSERT_TRUE(s->fac->factorize(1.0, -s->sigma));
  auto dropped = s->all;
  std::vector<double> hints;
  const auto removed = hs::driver::drop_found_pairs(dropped, hints, s->sol.eigenvalues, s->sigma, 1);
  ASSERT_EQ(removed.size(), 1u);
  CertifierRig<T> rig(*s, {});
  const auto out = rig.certifier.run(hs::CertificateMethod::ExactSingle, s->window, dropped, 0);
  EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Undecided);
  rig.rep.certificate.passed = false;
  rig.certifier.record_exact(true);
  EXPECT_NE(rig.rep.certificate.statement.find("single counts did not certify (valid 1, count 8 vs found 7)"),
            std::string::npos)
      << rig.rep.certificate.statement;
}

// The auto choice follows the measured times: expensive factorizations and cheap solves pick the
// randomized check (B = I: kappa proven); cheap factorizations and expensive solves the exact
// counts. The randomized check then runs on the held factorization and certifies.
TYPED_TEST(DriverCertificateTyped, AutoChoiceFollowsTheMeasuredTimes) {
  using T = TypeParam;
  auto s = solved_laplacian<T>(hs::CertificateMode::Auto);
  {
    hs::driver::CertificateTimings slow_factor{10.0, 1e-4, 4.0};
    CertifierRig<T> rig(*s, slow_factor);
    const auto ch = rig.certifier.choose(s->window, s->all);
    EXPECT_EQ(ch.method, hs::CertificateMethod::Randomized);
    EXPECT_TRUE(ch.kappa_rigorous);
    EXPECT_GT(ch.estimate.block_size, 0);
    EXPECT_GT(ch.epsilon, 0.0);
    const auto out = rig.certifier.check(s->window, s->all, 0);
    EXPECT_EQ(out.verdict, hs::driver::CheckOutcome::Verdict::Passed);
    EXPECT_EQ(rig.rep.certificate.method, "randomized");
    EXPECT_TRUE(rig.rep.certificate.rigorous);
    EXPECT_LE(rig.rep.certificate.failure_probability, s->cfg.certificate_delta);
    EXPECT_GT(rig.rep.certificate.solves, 0);
    EXPECT_EQ(rig.rep.certificate.statement.rfind("auto (round 0)", 0), 0u);
    EXPECT_TRUE(s->fac->inertia().valid);  // the factorization at sigma stays
  }
  {
    hs::driver::CertificateTimings slow_solves{1e-4, 10.0, 4.0};
    CertifierRig<T> rig(*s, slow_solves);
    EXPECT_EQ(rig.certifier.choose(s->window, s->all).method, hs::CertificateMethod::Exact);
    EXPECT_EQ(rig.certifier.check(s->window, s->all, 0).verdict, hs::driver::CheckOutcome::Verdict::Undecided);
    EXPECT_FALSE(rig.rep.certificate.performed);  // left to the exact counts
  }
}

// The fault injection removes copies of the largest multiplet first (one always stays), then the
// farthest selected eigenvalue together with its Ritz hints.
TYPED_TEST(DriverCertificateTyped, FaultDropOrder) {
  using T = TypeParam;
  const index_t n = 6;
  hs::driver::Block<T> all;
  all.lambda = {1.0, 1.0, 1.0, 2.0, 0.5, 3.0};
  all.x.resize(n, n);
  for (index_t j = 0; j < n; ++j) all.x(j, j) = T(1.0);
  hs::driver::recompute_products(hs::identity<T>(n), static_cast<const hs::CsrMatrix<T>*>(nullptr), all);
  std::vector<double> hints{3.0, 3.0 + 1e-12, 4.0};
  const std::vector<double> selected{1.0, 1.0, 1.0, 2.0, 0.5, 3.0};
  const auto removed = hs::driver::drop_found_pairs(all, hints, selected, 1.1, 3);
  EXPECT_EQ(removed, (std::vector<double>{1.0, 1.0, 3.0}));
  EXPECT_EQ(all.lambda, (std::vector<double>{1.0, 2.0, 0.5}));
  EXPECT_EQ(hints, std::vector<double>{4.0});
  EXPECT_EQ(all.x.cols(), 3);
  EXPECT_EQ(all.ax.cols(), 3);
}

}  // namespace
