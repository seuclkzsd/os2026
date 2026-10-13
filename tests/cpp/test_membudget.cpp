// P7 memory budget: the memory limits of the process (detect_memory_limits: cgroup v1/v2, the
// caller's HSOLVE_MEMORY_LIMIT_GB, rlimits, MemTotal), the automatic budget derived from them
// (resolve_memory_budget), its configuration ("auto" / 0 / GiB) and report, and that a budget too
// small for in-core factors gives a valid out-of-core run - also when the in-core attempt itself
// fails for memory (FactorOptions::ooc_on_memory_error).
#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

namespace fs = std::filesystem;
using hs::index_t;

constexpr double kGiB = 1073741824.0;

/// A fake /proc + /sys tree for detect_memory_limits.
class FakeRoot {
 public:
  FakeRoot() {
    static int serial = 0;
    root_ = fs::path(::testing::TempDir()) / ("membudget_root_" + std::to_string(::getpid()) + "_" +
                                              std::to_string(serial++));
    fs::remove_all(root_);
    fs::create_directories(root_);
    write("proc/meminfo", "MemTotal:       263000000 kB\nMemFree:         1000 kB\n");
  }
  ~FakeRoot() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }
  void write(const std::string& rel, const std::string& text) const {
    const fs::path p = root_ / rel;
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
  }
  hs::MemoryProbe probe(const std::string& env = "") const {
    hs::MemoryProbe p;
    p.root = root_.string();
    p.use_environment = false;
    p.env_value = env;
    p.use_rlimits = false;
    return p;
  }

 private:
  fs::path root_;
};

const hs::MemoryLimitSource& source(const hs::MemoryLimits& m, const std::string& name) {
  for (const auto& s : m.sources)
    if (s.name == name) return s;
  static const hs::MemoryLimitSource none{};
  ADD_FAILURE() << "no source " << name;
  return none;
}

// ------------------------------------------------------------------------------ memory limits

TEST(UtilMemoryLimits, CgroupV2TakesTheSmallestLimitAlongThePath) {
  FakeRoot r;
  r.write("proc/self/cgroup", "0::/a/b\n");
  r.write("sys/fs/cgroup/a/b/memory.max", "max\n");
  r.write("sys/fs/cgroup/a/memory.max", "17179869184\n");  // 16 GiB
  const auto m = hs::detect_memory_limits(r.probe());
  EXPECT_DOUBLE_EQ(source(m, "cgroup").gib, 16.0);
  EXPECT_NE(source(m, "cgroup").detail.find("a/memory.max"), std::string::npos);
  EXPECT_NEAR(source(m, "mem_total").gib, 263000000.0 / 1048576.0, 1e-9);
  EXPECT_DOUBLE_EQ(m.gib, 16.0);
  EXPECT_EQ(m.source, "cgroup");
  ASSERT_EQ(m.sources.size(), 5u);
  EXPECT_EQ(m.sources[0].name, "cgroup");
  EXPECT_EQ(m.sources[4].name, "mem_total");
}

TEST(UtilMemoryLimits, CgroupV1ReadsTheLimitAndTheHierarchicalLimit) {
  FakeRoot r;
  r.write("proc/self/cgroup", "11:cpu,cpuacct:/\n4:memory:/user.slice\n");
  r.write("sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes", "9223372036854771712\n");  // unlimited
  r.write("sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n");
  auto m = hs::detect_memory_limits(r.probe());
  EXPECT_LT(source(m, "cgroup").gib, 0.0) << "unlimited";
  EXPECT_EQ(source(m, "cgroup").detail, "unlimited");
  EXPECT_EQ(m.source, "mem_total");
  // the effective limit of an ancestor shows in memory.stat
  r.write("sys/fs/cgroup/memory/user.slice/memory.stat", "cache 0\nhierarchical_memory_limit 34359738368\n");
  m = hs::detect_memory_limits(r.probe());
  EXPECT_DOUBLE_EQ(source(m, "cgroup").gib, 32.0);
  EXPECT_EQ(m.source, "cgroup");
  // and a limit of the cgroup itself, below it
  r.write("sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes", "8589934592\n");
  m = hs::detect_memory_limits(r.probe());
  EXPECT_DOUBLE_EQ(m.gib, 8.0);
}

TEST(UtilMemoryLimits, ContainerCgroupIsReadAtTheMountRoot) {
  // A cgroup namespace shows the host's path, which does not exist below the container's mount.
  FakeRoot r;
  r.write("proc/self/cgroup", "0::/kubepods/pod42/c1\n");
  r.write("sys/fs/cgroup/memory.max", "8589934592\n");
  const auto m = hs::detect_memory_limits(r.probe());
  EXPECT_DOUBLE_EQ(m.gib, 8.0);
  EXPECT_EQ(m.source, "cgroup");
}

TEST(UtilMemoryLimits, EnvironmentAndResourceLimits) {
  FakeRoot r;
  r.write("proc/self/cgroup", "0::/\n");
  auto p = r.probe("12.5");
  auto m = hs::detect_memory_limits(p);
  EXPECT_DOUBLE_EQ(source(m, "env").gib, 12.5);
  EXPECT_EQ(m.source, "env");
  p.rlimit_as = static_cast<std::uint64_t>(8 * kGiB);
  p.rlimit_data = static_cast<std::uint64_t>(10 * kGiB);
  m = hs::detect_memory_limits(p);
  EXPECT_DOUBLE_EQ(source(m, "rlimit_as").gib, 8.0);
  EXPECT_DOUBLE_EQ(source(m, "rlimit_data").gib, 10.0);
  EXPECT_EQ(m.source, "rlimit_as");
  EXPECT_DOUBLE_EQ(m.gib, 8.0);
  // malformed or non-positive declarations are ignored, and say so
  for (const std::string bad : {"12x", "-3", "0", "inf", "nan", "abc"}) {
    m = hs::detect_memory_limits(r.probe(bad));
    EXPECT_LT(source(m, "env").gib, 0.0) << bad;
    EXPECT_NE(source(m, "env").detail.find("ignored"), std::string::npos) << bad;
    EXPECT_EQ(m.source, "mem_total") << bad;
  }
}

TEST(UtilMemoryLimits, NothingReadableSetsNoLimit) {
  FakeRoot r;
  fs::remove(fs::path(r.probe().root) / "proc/meminfo");
  const auto m = hs::detect_memory_limits(r.probe());
  EXPECT_LT(m.gib, 0.0);
  EXPECT_EQ(m.source, "");
  for (const auto& s : m.sources) EXPECT_LT(s.gib, 0.0) << s.name;
}

TEST(UtilMemoryLimits, ThisProcessHasAtLeastItsMachine) {
  const auto m = hs::detect_memory_limits();
  EXPECT_GT(source(m, "mem_total").gib, 0.0);
  EXPECT_GT(m.gib, 0.0);
  EXPECT_LE(m.gib, source(m, "mem_total").gib);
}

// ------------------------------------------------------------------------------ the budget

hs::MemoryLimits limits_of(double gib, const std::string& src = "env") {
  hs::MemoryLimits m;
  m.gib = gib;
  m.source = src;
  m.sources.push_back({src, gib, "test"});
  return m;
}

TEST(DriverMemoryBudget, AutoIsAShareOfTheSmallestLimit) {
  // budget = max(floor, min(fraction * L, L - headroom))
  auto b = hs::resolve_memory_budget(hs::kMemoryBudgetAuto, limits_of(32.0));
  EXPECT_DOUBLE_EQ(b.gb, hs::kAutoBudgetFraction * 32.0);
  EXPECT_EQ(b.source, "env");
  EXPECT_DOUBLE_EQ(b.limit_gb, 32.0);
  EXPECT_EQ(b.limit_source, "env");
  ASSERT_EQ(b.limits.size(), 1u);
  b = hs::resolve_memory_budget(0.0, limits_of(16.0, "cgroup"));
  EXPECT_DOUBLE_EQ(b.gb, std::min(hs::kAutoBudgetFraction * 16.0, 16.0 - hs::kAutoBudgetHeadroomGb));
  EXPECT_EQ(b.source, "cgroup");
  b = hs::resolve_memory_budget(0.0, limits_of(4.0));  // small machines keep the absolute headroom
  EXPECT_DOUBLE_EQ(b.gb, std::min(hs::kAutoBudgetFraction * 4.0, 4.0 - hs::kAutoBudgetHeadroomGb));
  EXPECT_LE(b.gb, 4.0 - hs::kAutoBudgetHeadroomGb);
  b = hs::resolve_memory_budget(0.0, limits_of(0.8));  // tiny: the floor
  EXPECT_DOUBLE_EQ(b.gb, hs::kAutoBudgetFloorGb);
  b = hs::resolve_memory_budget(0.0, hs::MemoryLimits{});  // nothing readable
  EXPECT_DOUBLE_EQ(b.gb, hs::kAutoBudgetFallbackGb);
  EXPECT_EQ(b.source, "default");
  EXPECT_LT(b.limit_gb, 0.0);
}

TEST(DriverMemoryBudget, AConfiguredBudgetWinsAndTheLimitIsStillRecorded) {
  const auto b = hs::resolve_memory_budget(20.0, limits_of(16.0, "cgroup"));
  EXPECT_DOUBLE_EQ(b.gb, 20.0);
  EXPECT_EQ(b.source, "config");
  EXPECT_DOUBLE_EQ(b.limit_gb, 16.0);
  EXPECT_EQ(b.limit_source, "cgroup");
  for (const double bad : {-1.0, std::nan(""), HUGE_VAL})
    EXPECT_THROW(hs::resolve_memory_budget(bad, limits_of(16.0)), hs::Error) << bad;
}

TEST(DriverMemoryBudget, ConfigAcceptsAutoZeroAndGiB) {
  EXPECT_EQ(hs::SolverConfig{}.memory_budget_gb, hs::kMemoryBudgetAuto);
  EXPECT_EQ(hs::parse_config(R"({"memory_budget_gb": "auto"})").memory_budget_gb, hs::kMemoryBudgetAuto);
  EXPECT_EQ(hs::parse_config(R"({"memory_budget_gb": 0})").memory_budget_gb, hs::kMemoryBudgetAuto);
  EXPECT_EQ(hs::parse_config(R"({"memory_budget_gb": 14.0})").memory_budget_gb, 14.0);  // pre-P7 configs
  for (const char* bad : {R"({"memory_budget_gb": -2})", R"({"memory_budget_gb": "big"})",
                          R"({"memory_budget_gb": true})"})
    EXPECT_THROW(hs::parse_config(bad), hs::Error) << bad;
  // auto is written as "auto" and reads back; a number stays a number
  const auto j = nlohmann::json::parse(hs::config_to_json(hs::SolverConfig{}));
  EXPECT_EQ(j["memory_budget_gb"], "auto");
  EXPECT_EQ(hs::parse_config(j.dump()).memory_budget_gb, hs::kMemoryBudgetAuto);
  hs::SolverConfig c;
  c.memory_budget_gb = 7.5;
  EXPECT_EQ(nlohmann::json::parse(hs::config_to_json(c))["memory_budget_gb"], 7.5);
  EXPECT_EQ(hs::parse_config(R"({"factor_precision": "auto"})").factor_precision, hs::FactorPrecision::Auto);
}

/// Sets an environment variable for the scope.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old, had_ = true;
    ::setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_)
      ::setenv(name_.c_str(), old_.c_str(), 1);
    else
      ::unsetenv(name_.c_str());
  }

 private:
  std::string name_, old_;
  bool had_ = false;
};

hs::SolverConfig nearest(double sigma, index_t k) {
  hs::SolverConfig cfg;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.threads = 2;
  return cfg;
}

TEST(DriverMemoryBudget, TheSolveReportsItsBudgetAndWhereItCameFrom) {
  const auto a = hs::test::laplacian_2d<double>(30, 31);
  {
    ScopedEnv env(hs::kMemoryLimitEnv, "3");
    const auto sol = hs::solve(a, nullptr, nearest(1.0, 4));
    EXPECT_TRUE(sol.report.success);
    const auto& m = sol.report.memory;
    EXPECT_GT(m.limit_gb, 0.0);
    EXPECT_LE(m.limit_gb, 3.0);  // the declaration, or an even smaller limit of the test machine
    const auto expected = hs::resolve_memory_budget(0.0, limits_of(m.limit_gb, m.limit_source));
    EXPECT_DOUBLE_EQ(m.budget_gb, expected.gb);
    EXPECT_EQ(m.source, m.limit_source);
    EXPECT_EQ(m.limits.size(), 5u);
    const auto j = nlohmann::json::parse(sol.report.to_json());
    EXPECT_EQ(j["memory"]["budget_gb"], m.budget_gb);
    EXPECT_EQ(j["memory"]["source"], m.source);
    EXPECT_EQ(j["memory"]["limits"].size(), 5u);
    EXPECT_EQ(j["precision"]["requested"], "double");
    EXPECT_EQ(j["precision"]["used"], "double");
    EXPECT_EQ(j["precision"]["fallback"], false);
  }
  auto cfg = nearest(1.0, 4);
  cfg.memory_budget_gb = 2.5;
  const auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_EQ(sol.report.memory.source, "config");
  EXPECT_DOUBLE_EQ(sol.report.memory.budget_gb, 2.5);
}

// ------------------------------------------------------------------------------ out of core

/// 3-D 7-point Dirichlet Laplacian on an n^3 grid (3-D fill: in-core factors well above the
/// out-of-core working memory).
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

std::vector<double> nearest_of(std::vector<double> w, double sigma, std::size_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double x, double y) { return std::abs(x - sigma) < std::abs(y - sigma); });
  w.resize(k);
  std::sort(w.begin(), w.end());
  return w;
}

void expect_same(std::vector<double> got, const std::vector<double>& want, double tol) {
  std::sort(got.begin(), got.end());
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol) << i;
}

// An in-core factorization that fails for memory (here: a cap below the in-core estimate, set on
// purpose; in the driver: delayed pivots that grew the workspace beyond the estimate the in-core
// decision was made on) is retried out of core with ooc_on_memory_error, else it throws.
TEST(FactorOutOfCore, AMemoryErrorInCoreIsRetriedOutOfCore) {
  const auto a = laplacian_3d(26);
  const index_t n = a.nrows;
  hs::ShiftedPencil<double> pencil(a, nullptr);
  hs::FactorOptions o;
  o.threads = 2;
  o.ordering = 0;
  auto probe = hs::make_mumps_factorization<double>(o);
  probe->analyze(pencil, 1.0, -0.5);
  const hs::FactorEstimate e = probe->estimate();
  probe.reset();
  ASSERT_GT(e.incore_mb, 1.5 * e.ooc_mb) << "the case must need out of core below its in-core estimate";
  // MUMPS sizes its workspace from the cap and often needs less than INFOG(16) in core: a cap just
  // above the out-of-core working memory leaves no room for the in-core factors.
  o.max_memory_mb = e.ooc_mb + 0.1 * (e.incore_mb - e.ooc_mb) + 2.0;
  for (const bool retry : {false, true}) {
    o.ooc_on_memory_error = retry;
    auto f = hs::make_mumps_factorization<double>(o);
    f->analyze(pencil, 1.0, -0.5);
    if (!retry) {
      try {
        static_cast<void>(f->factorize(1.0, -0.5));
        FAIL() << "an in-core factorization beyond the cap must fail (estimates " << e.incore_mb << " / "
               << e.ooc_mb << " MB, cap " << o.max_memory_mb << " MB)";
      } catch (const hs::Error& err) {
        // -19 (above the cap), or a workspace that its retries could not grow under the cap
        EXPECT_TRUE(err.code() == hs::Error::Code::OutOfMemoryBudget ||
                    err.code() == hs::Error::Code::FactorizationFailed)
            << err.what();
      }
      continue;
    }
    ASSERT_TRUE(f->factorize(1.0, -0.5));
    EXPECT_TRUE(f->options().out_of_core);
    EXPECT_EQ(f->stats().ooc_switches, 1);
    // the out-of-core factors solve
    std::vector<double> x(static_cast<std::size_t>(n), 1.0), b(x.size());
    hs::spmv(a, x.data(), b.data());
    for (std::size_t i = 0; i < b.size(); ++i) b[i] -= 0.5 * x[i];
    f->solve(b.data(), 1, n);
    double err = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) err = std::max(err, std::abs(b[i] - 1.0));
    EXPECT_LT(err, 1e-8);
    // and later factorizations stay out of core
    ASSERT_TRUE(f->factorize(1.0, -0.7));
    EXPECT_EQ(f->stats().ooc_switches, 1);
  }
}

// A budget too small for the in-core factors but enough for the out-of-core working memory (and
// the certificate's inertia-only counts, which need about as much): a valid out-of-core run.
TEST(DriverMemoryBudget, ABudgetBelowTheInCoreFactorsGivesAValidOutOfCoreRun) {
  const index_t m = 26;
  const auto a = laplacian_3d(m);
  const auto w = laplacian_3d_eigenvalues(m);
  std::size_t j = 40;  // the widest gap among w[40..100] (a cube: many multiple eigenvalues)
  for (std::size_t i = 40; i < 100; ++i)
    if (w[i + 1] - w[i] > w[j + 1] - w[j]) j = i;
  const double sigma = 0.5 * (w[j] + w[j + 1]);
  auto cfg = nearest(sigma, 8);
  cfg.ordering = 0;  // AMD: one analysis, the same estimate in both runs
  cfg.memory_budget_gb = 8.0;
  const auto big = hs::solve(a, nullptr, cfg);
  ASSERT_TRUE(big.report.success);
  ASSERT_FALSE(big.report.factor_options.out_of_core);
  const auto& e = big.report.factor_stats.estimate;
  ASSERT_GT(e.incore_mb, 1.5 * e.ooc_mb);
  // cap = 0.95 (budget - rss - krylov): put it between the out-of-core and the in-core estimates
  const double cap_mb = 0.5 * (e.incore_mb + e.ooc_mb);
  const auto reserve = hs::driver::krylov_reservation<double>(cfg, hs::ProblemType::Standard, a.nrows);
  const double krylov = static_cast<double>(a.nrows) * 8.0 * static_cast<double>(reserve.columns);
  const double rss = static_cast<double>(std::max<std::int64_t>(hs::current_rss_bytes(), 0));
  cfg.memory_budget_gb = (rss + krylov + cap_mb * 1e6 / 0.95) / kGiB;
  const auto small = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(small.report.success);
  EXPECT_TRUE(small.report.factor_options.out_of_core);
  EXPECT_TRUE(small.report.certificate.passed);
  expect_same(small.eigenvalues, nearest_of(w, sigma, 8), 1e-9);
  bool warned = false;
  for (const auto& msg : small.report.warnings) warned = warned || msg.find("out-of-core") != std::string::npos;
  EXPECT_TRUE(warned);
}

}  // namespace
