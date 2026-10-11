// Regression tests of the drvint review fixes (src/driver): the deterministic nested-dissection
// ordering, when certificate counts on an InertiaCounter release the main factors, and the
// counter_pays rule.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// 3-D 7-point Dirichlet Laplacian on an nx x ny x nz grid (3-D fill: nested dissection pays).
template <class T>
hs::CsrMatrix<T> laplacian_3d(index_t nx, index_t ny, index_t nz) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [&](index_t i, index_t j, index_t k) { return i + nx * (j + ny * k); };
  for (index_t k = 0; k < nz; ++k)
    for (index_t j = 0; j < ny; ++j)
      for (index_t i = 0; i < nx; ++i) {
        const index_t p = id(i, j, k);
        r.push_back(p), c.push_back(p), v.push_back(T(6));
        if (i > 0) r.push_back(p), c.push_back(id(i - 1, j, k)), v.push_back(T(-1));
        if (i + 1 < nx) r.push_back(p), c.push_back(id(i + 1, j, k)), v.push_back(T(-1));
        if (j > 0) r.push_back(p), c.push_back(id(i, j - 1, k)), v.push_back(T(-1));
        if (j + 1 < ny) r.push_back(p), c.push_back(id(i, j + 1, k)), v.push_back(T(-1));
        if (k > 0) r.push_back(p), c.push_back(id(i, j, k - 1)), v.push_back(T(-1));
        if (k + 1 < nz) r.push_back(p), c.push_back(id(i, j, k + 1)), v.push_back(T(-1));
      }
  const index_t n = nx * ny * nz;
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

std::vector<double> laplacian_3d_eigenvalues(index_t nx, index_t ny, index_t nz) {
  const auto wx = hs::test::laplacian_1d_eigenvalues(nx), wy = hs::test::laplacian_1d_eigenvalues(ny),
             wz = hs::test::laplacian_1d_eigenvalues(nz);
  std::vector<double> w;
  for (double a : wx)
    for (double b : wy)
      for (double c : wz) w.push_back(a + b + c);
  std::sort(w.begin(), w.end());
  return w;
}

index_t count_below(const std::vector<double>& w, double s) {
  return static_cast<index_t>(std::lower_bound(w.begin(), w.end(), s) - w.begin());
}

/// Mid-point of the first gap after w[i] (w ascending; skips multiple eigenvalues).
double gap_after(const std::vector<double>& w, std::size_t i) {
  while (w[i + 1] - w[i] < 1e-6) ++i;
  return 0.5 * (w[i] + w[i + 1]);
}

hs::SolverConfig nearest_cfg(double sigma, index_t k) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
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

/// Events named `name` of a JSON-lines log.
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

// ------------------------------------------------------------------------------ ordering

// deterministic = true promises bitwise-reproducible runs. The adaptive ordering's nested
// dissection must then be METIS: SCOTCH's orderings differ from analysis to analysis (in one
// process as across processes), and with them the factors and every result.
// sigma: the 8 nearest eigenvalues do not split a multiple one.
template <class T>
void deterministic_runs_are_bitwise_equal(index_t m, double sigma) {
  const auto a = laplacian_3d<T>(m, m, m - 1);
  const auto w = laplacian_3d_eigenvalues(m, m, m - 1);
  auto cfg = nearest_cfg(sigma, 8);
  cfg.deterministic = true;
  ASSERT_EQ(cfg.ordering, -1);  // adaptive
  const auto s1 = hs::solve(a, nullptr, cfg);
  const auto s2 = hs::solve(a, nullptr, cfg);
  ASSERT_EQ(s1.report.factor_options.ordering, 5) << "the AMD probe should pick nested dissection (METIS)";
  EXPECT_TRUE(s1.report.success);
  expect_same_set(s1.eigenvalues, nearest(w, cfg.target.sigma, 8), 1e-9);
  EXPECT_EQ(s1.report.factor_stats.factor_entries, s2.report.factor_stats.factor_entries);
  ASSERT_EQ(s1.eigenvalues.size(), s2.eigenvalues.size());
  EXPECT_EQ(std::memcmp(s1.eigenvalues.data(), s2.eigenvalues.data(), s1.eigenvalues.size() * sizeof(double)), 0);
  ASSERT_EQ(s1.eigenvectors.cols(), s2.eigenvectors.cols());
  EXPECT_EQ(std::memcmp(s1.eigenvectors.data(), s2.eigenvectors.data(),
                        static_cast<std::size_t>(s1.eigenvectors.rows() * s1.eigenvectors.cols()) * sizeof(T)),
            0);
}

TEST(DriverOrdering, DeterministicRunsAreBitwiseEqualReal) { deterministic_runs_are_bitwise_equal<double>(18, 1.003); }
TEST(DriverOrdering, DeterministicRunsAreBitwiseEqualComplex) { deterministic_runs_are_bitwise_equal<cplx>(18, 1.003); }

TEST(DriverOrdering, NestedDissectionIsScotchUnlessDeterministic) {
  hs::SolverConfig cfg;
  EXPECT_EQ(hs::driver::nd_ordering(cfg), 3);
  cfg.deterministic = true;
  EXPECT_EQ(hs::driver::nd_ordering(cfg), 5);
}

// ---------------------------------------------------------------------- counts and main factors

// Counts on the counter with the main factors resident add the counter's working memory to them.
// With MainFactors::KeepWithinCap the main factors must be released whenever both together exceed
// the factorization cap - before every count, not only when the counter is built: here the main
// object is refactorized between two counts (as for a rerun or a slice shift).
TEST(DriverCounter, CountsReleaseTheMainFactorsWheneverBothExceedTheCap) {
  const index_t m = 26;
  const auto a = laplacian_3d<double>(m, m, m);
  const auto w = laplacian_3d_eigenvalues(m, m, m);
  const hs::ShiftedPencil<double> pencil(a, nullptr);
  hs::FactorOptions o;
  o.ordering = 5;  // METIS: the same estimates in every run
  o.threads = 2;
  const double sigma = 0.0;  // below the spectrum: K(sigma) positive definite, no delayed pivots
  // Memory of the main factors and of a counter, measured without a cap.
  double main_mb = 0.0, main_est = 0.0, counter_mb = 0.0;
  {
    auto probe = hs::make_mumps_factorization<double>(o);
    probe->analyze(pencil, 1.0, -sigma);
    ASSERT_TRUE(probe->factorize(1.0, -sigma));
    main_mb = probe->stats().factor_mb_used;
    main_est = probe->estimate().incore_mb;
    counter_mb = hs::InertiaCounter<double>(*probe, o).estimate().incore_mb;
  }
  // A cap that each fits (MUMPS needs ~1.01x its estimate) but not both together.
  const double lo = 1.1 * std::max(main_est, counter_mb) + 2.0, hi = main_mb + counter_mb;
  ASSERT_LT(lo, hi) << "main " << main_mb << " (estimate " << main_est << "), counter " << counter_mb;
  o.max_memory_mb = 0.5 * (lo + hi);
  auto fac = hs::make_mumps_factorization<double>(o);
  fac->analyze(pencil, 1.0, -sigma);

  hs::SolverConfig cfg;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx c{cfg, nullptr, rep, clock};
  bool off = false;
  hs::driver::factorize_near(c, *fac, sigma, 12.0, off);
  ASSERT_TRUE(fac->inertia().valid);
  hs::driver::CountSource<double> src(*fac, pencil, /*prefer_counter=*/true);
  ASSERT_TRUE(src.uses_counter());
  const double s1 = gap_after(w, 40), s2 = gap_after(w, 90);
  auto in = src.count(c, s1, true);
  ASSERT_TRUE(in.valid);
  EXPECT_EQ(in.neg, count_below(w, s1));
  EXPECT_FALSE(fac->inertia().valid) << "main factors resident next to the counter beyond the cap";
  EXPECT_TRUE(std::isnan(c.held.shift));
  hs::driver::refactorize_at(c, *fac, sigma, off);  // a rerun at the working shift
  ASSERT_TRUE(fac->inertia().valid);
  in = src.count(c, s2, true);
  ASSERT_TRUE(in.valid);
  EXPECT_EQ(in.neg, count_below(w, s2));
  EXPECT_FALSE(fac->inertia().valid) << "second count with the main factors resident beyond the cap";

  // Within the cap the main factors stay (window paths reuse them between counts) ...
  hs::FactorOptions roomy = o;
  roomy.max_memory_mb = 10.0 * hi;
  auto fac2 = hs::make_mumps_factorization<double>(roomy);
  fac2->analyze(pencil, 1.0, -sigma);
  hs::driver::factorize_near(c, *fac2, sigma, 12.0, off);
  hs::driver::CountSource<double> keep(*fac2, pencil, true, hs::driver::MainFactors::KeepWithinCap);
  EXPECT_EQ(keep.count(c, s1, true).neg, count_below(w, s1));
  EXPECT_TRUE(fac2->inertia().valid);
  // ... unless the source releases them for every count (nearest certificates).
  hs::driver::CountSource<double> rel(*fac2, pencil, true, hs::driver::MainFactors::Release);
  EXPECT_EQ(rel.count(c, s2, true).neg, count_below(w, s2));
  EXPECT_FALSE(fac2->inertia().valid);
  hs::driver::refactorize_at(c, *fac2, sigma, off);
  EXPECT_EQ(rel.count(c, s1, true).neg, count_below(w, s1));
  EXPECT_FALSE(fac2->inertia().valid);
}

// Nearest certificate on the counter: the main factors are released before the counts (never
// resident next to the counter) and refactorized only for a rerun - once per rerun, and released
// again before the next round's counts. Same certified set as main-object counts.
TEST(DriverCounter, NearestCertificateReleasesTheMainFactors) {
  std::vector<double> w;
  for (int i = 0; i < 6; ++i) w.push_back(1.0);  // six-fold eigenvalue at the edge of the set
  for (int i = 1; i <= 40; ++i) {
    w.push_back(1.0 + 0.05 * i);
    w.push_back(1.0 - 0.05 * i);
  }
  for (int i = 0; i < 400; ++i) w.push_back(4.0 + 0.01 * i);
  const auto a = hs::test::prescribed_spectrum<double>(w, 3, 77);
  auto cfg = nearest_cfg(1.103, 10);
  cfg.block_size = 1;        // the first run misses copies of the multiplet: one rerun
  cfg.deterministic = true;  // bitwise reproducible: the same misses in every run
  cfg.inertia_counter = hs::Toggle::On;
  const std::string path = ::testing::TempDir() + "/drvint_fix_nearest_release.jsonl";
  hs::Solution<double> on;
  {
    hs::Logger log(0, path);
    on = hs::solve(a, nullptr, cfg, &log);
  }
  EXPECT_TRUE(on.report.success);
  EXPECT_TRUE(on.report.certificate.passed);
  expect_same_set(on.eigenvalues, nearest(w, 1.103, 10), 1e-10);
  const auto certs = events(path, "certificate");
  const auto released = events(path, "main_factors_released");
  ASSERT_EQ(certs.size(), 2u);     // round 0 fails (copies missing), round 1 passes
  EXPECT_EQ(released.size(), 2u);  // before the counts of each round
  // K(sigma), refactorized once for the rerun; no count on the main object.
  EXPECT_EQ(on.report.factor_stats.factorizations, 2);
  cfg.inertia_counter = hs::Toggle::Off;
  const auto off = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(off.report.success);
  EXPECT_EQ(off.report.certificate.count_inertia, on.report.certificate.count_inertia);
  EXPECT_EQ(off.report.factor_stats.factorizations, 2 + 4);  // + two counts per round
}

// counter_pays (nearest certificate, two counts): the analysis estimates of three data cases at 4
// threads (bench/drvint_fix_ordering.cpp). Factorizations dominate the 3-D case (pays); the 2-D
// cavity and the low-fill complex embedding cost more to analyse than two counts save.
TEST(DriverCounter, CounterPaysRule) {
  using hs::driver::counter_pays;
  hs::SolverConfig cfg;
  ASSERT_EQ(cfg.inertia_counter, hs::Toggle::Auto);
  hs::FactorEstimate mem27;  // syn-mem27-46, SCOTCH
  mem27.factor_entries = 50885020;
  mem27.flops = 6.98e10;
  const hs::driver::MumpsInputSize mem27_in{97336.0, 1306396.0};
  hs::FactorEstimate cav;  // cav2d-rect-pert-1e5, AMD
  cav.factor_entries = 3014511;
  cav.flops = 1.67e8;
  const hs::driver::MumpsInputSize cav_in{99459.0, 297605.0};
  hs::FactorEstimate emb;  // syn-near-singular-complex-1e5 (real embedding of order 2n), AMD
  emb.factor_entries = 4899808;
  emb.flops = 1.24e8;
  const hs::driver::MumpsInputSize emb_in{200000.0, 4899680.0};
  EXPECT_TRUE(counter_pays(cfg, mem27, mem27_in, 4, 2));
  EXPECT_FALSE(counter_pays(cfg, cav, cav_in, 4, 2));
  EXPECT_FALSE(counter_pays(cfg, emb, emb_in, 4, 2));
  // More counts amortize the analysis; more threads make each factorization cheaper.
  EXPECT_TRUE(counter_pays(cfg, cav, cav_in, 4, 100));
  EXPECT_FALSE(counter_pays(cfg, mem27, mem27_in, 4, 0));
  // The configuration overrides the model.
  cfg.inertia_counter = hs::Toggle::On;
  EXPECT_TRUE(counter_pays(cfg, cav, cav_in, 4, 2));
  cfg.inertia_counter = hs::Toggle::Off;
  EXPECT_FALSE(counter_pays(cfg, mem27, mem27_in, 4, 2));
}

}  // namespace
