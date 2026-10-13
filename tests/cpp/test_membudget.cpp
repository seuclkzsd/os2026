// P7 memory budget: the memory limits of the process (detect_memory_limits: cgroup v1/v2, the
// caller's HSOLVE_MEMORY_LIMIT_GB, rlimits, MemTotal), the automatic budget derived from them
// (resolve_memory_budget) in the measure of its limit (resident memory, or the address space under
// RLIMIT_AS), its configuration ("auto" / 0 / GiB) and report, that a solve under a real RLIMIT_AS
// succeeds, and that a budget too small for in-core factors gives a valid out-of-core run - also
// when the in-core attempt itself fails for memory (FactorOptions::ooc_on_memory_error).
#include <gtest/gtest.h>

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// The measures a limit applies to, read for this process.
TEST(UtilMemoryLimits, MeasuresOfThisProcess) {
  const auto rss = hs::memory_in_use_bytes(hs::MemoryMeasure::Resident);
  const auto vm = hs::memory_in_use_bytes(hs::MemoryMeasure::AddressSpace);
  const auto data = hs::memory_in_use_bytes(hs::MemoryMeasure::Data);
  EXPECT_EQ(rss > 0, hs::current_rss_bytes() > 0);
  EXPECT_GT(rss, 0);
  EXPECT_GT(vm, rss);  // code, libraries and stacks are mapped, not all resident
  EXPECT_GT(data, 0);
  EXPECT_LE(data, vm);
  EXPECT_GE(hs::peak_address_space_bytes(), vm);
  EXPECT_EQ(hs::measure_of("rlimit_as"), hs::MemoryMeasure::AddressSpace);
  EXPECT_EQ(hs::measure_of("rlimit_data"), hs::MemoryMeasure::Data);
  for (const char* s : {"cgroup", "env", "mem_total", "config", "default", ""})
    EXPECT_EQ(hs::measure_of(s), hs::MemoryMeasure::Resident) << s;
  EXPECT_STREQ(hs::to_string(hs::MemoryMeasure::Resident), "resident");
  EXPECT_STREQ(hs::to_string(hs::MemoryMeasure::AddressSpace), "address_space");
  EXPECT_STREQ(hs::to_string(hs::MemoryMeasure::Data), "data");
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

// RLIMIT_AS / RLIMIT_DATA limit what the process maps, also what it reserves without touching:
// the budget is charged in that measure, and what is reserved unused when it is resolved (BLAS
// buffers, malloc arenas, stacks: 3.4 GB at 16 threads) is set aside before the formula's share.
TEST(DriverMemoryBudget, AnAddressSpaceLimitIsPlannedInItsOwnMeasure) {
  auto b = hs::resolve_memory_budget(0.0, limits_of(16.0, "rlimit_as"), 3.0);
  EXPECT_EQ(b.measure, hs::MemoryMeasure::AddressSpace);
  EXPECT_DOUBLE_EQ(b.reserved_gb, 3.0);
  EXPECT_DOUBLE_EQ(b.gb, 3.0 + std::min(hs::kAutoBudgetFraction * 13.0, 13.0 - hs::kAutoBudgetHeadroomGb));
  EXPECT_EQ(b.source, "rlimit_as");
  b = hs::resolve_memory_budget(0.0, limits_of(16.0, "rlimit_data"), 2.0);
  EXPECT_EQ(b.measure, hs::MemoryMeasure::Data);
  EXPECT_DOUBLE_EQ(b.gb, 2.0 + hs::kAutoBudgetFraction * 14.0);
  b = hs::resolve_memory_budget(0.0, limits_of(2.0, "rlimit_as"), 3.0);  // nothing left: the floor
  EXPECT_DOUBLE_EQ(b.gb, 3.0 + hs::kAutoBudgetFloorGb);
  // resident limits (and a configured budget, which states resident memory) set nothing aside
  for (const char* src : {"env", "cgroup", "mem_total"}) {
    b = hs::resolve_memory_budget(0.0, limits_of(16.0, src), 3.0);
    EXPECT_EQ(b.measure, hs::MemoryMeasure::Resident) << src;
    EXPECT_EQ(b.reserved_gb, 0.0) << src;
    EXPECT_DOUBLE_EQ(b.gb, hs::kAutoBudgetFraction * 16.0) << src;
  }
  b = hs::resolve_memory_budget(5.0, limits_of(16.0, "rlimit_as"), 3.0);
  EXPECT_EQ(b.measure, hs::MemoryMeasure::Resident);
  EXPECT_DOUBLE_EQ(b.gb, 5.0);
  EXPECT_EQ(b.reserved_gb, 0.0);
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
    const auto expected = hs::resolve_memory_budget(0.0, limits_of(m.limit_gb, m.limit_source), m.reserved_gb);
    EXPECT_DOUBLE_EQ(m.budget_gb, expected.gb);
    EXPECT_EQ(m.source, m.limit_source);
    EXPECT_EQ(m.measure, hs::measure_of(m.source));
    EXPECT_EQ(m.limits.size(), 5u);
    const auto j = nlohmann::json::parse(sol.report.to_json());
    EXPECT_EQ(j["memory"]["budget_gb"], m.budget_gb);
    EXPECT_EQ(j["memory"]["source"], m.source);
    EXPECT_EQ(j["memory"]["measure"], hs::to_string(m.measure));
    EXPECT_EQ(j["memory"]["reserved_gb"], m.reserved_gb);
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

// The cap is planned before the analysis; what the analysis keeps resident is taken off it
// before the in-core / out-of-core decision (event memory_replan), and the factors get that cap.
TEST(DriverMemoryBudget, TheCapShrinksByWhatTheAnalysisKeepsResident) {
  const auto a = laplacian_3d(30);
  auto cfg = nearest(1.0, 6);
  cfg.memory_budget_gb = 3.0;
  const std::string path = ::testing::TempDir() + "/membudget_replan.jsonl";
  hs::Solution<double> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, nullptr, cfg, &log);
  }
  EXPECT_TRUE(sol.report.success);
  double plan = -1, replan = -1, grown = -1;
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);) {
    const auto j = nlohmann::json::parse(line);
    EXPECT_TRUE(j.contains("rss_mb") && j.contains("peak_rss_mb")) << line;  // the memory timeline
    EXPECT_TRUE(j.contains("vm_mb") && j.contains("peak_vm_mb")) << line;    // and the address space
    if (j["event"] == "memory_plan") plan = j["cap_mb"];
    if (j["event"] == "memory_replan") replan = j["cap_mb"], grown = j["grown_mb"];
  }
  ASSERT_GT(plan, 0.0);
  ASSERT_GT(grown, 0.0) << "the analysis allocates its structures";
  EXPECT_NEAR(replan, plan - 0.95 * grown, 1e-6 * plan);
  EXPECT_DOUBLE_EQ(sol.report.factor_options.max_memory_mb, replan);
  // the factor layer's setter
  hs::ShiftedPencil<double> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<double>(hs::FactorOptions{});
  f->set_max_memory_mb(123.0);
  EXPECT_EQ(f->options().max_memory_mb, 123.0);
  EXPECT_THROW(f->set_max_memory_mb(-1.0), hs::Error);
  EXPECT_THROW(f->set_max_memory_mb(HUGE_VAL), hs::Error);
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
  const double sigma = w[j] + 0.3 * (w[j + 1] - w[j]);  // not mid-way: equidistant multiplets would tie
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

// ------------------------------------------------------------------------------ RLIMIT_AS

// The child of AnAddressSpaceLimitIsHonoured: a 3-D solve (syn-mem27-46 under `ulimit -v 8`, scaled
// down) under RLIMIT_AS = its address space + 1.25 GiB. Before the fix the budget derived from that
// limit was planned against RSS and MUMPS reserved the whole factorization cap: with the BLAS
// buffers and malloc arenas already mapped (~0.15 GB per thread) the address space ran out
// (INFO -13, or OpenBLAS failing to allocate and the process crashing). Exits 0 on a valid result
// planned in the address space, 1 on a wrong one, 2 when the solve throws.
[[noreturn]] void solve_under_an_address_space_limit() {
  ::unsetenv(hs::kMemoryLimitEnv);  // RLIMIT_AS must be the smallest limit (the guard declares one)
  const index_t m = 30;
  const auto a = laplacian_3d(m);
  const auto w = laplacian_3d_eigenvalues(m);
  std::size_t j = 40;
  for (std::size_t i = 40; i < 100; ++i)
    if (w[i + 1] - w[i] > w[j + 1] - w[j]) j = i;
  const double sigma = w[j] + 0.3 * (w[j + 1] - w[j]);
  auto cfg = nearest(sigma, 8);
  cfg.threads = 4;
  const auto vm = hs::memory_in_use_bytes(hs::MemoryMeasure::AddressSpace);
  struct rlimit rl {};
  ::getrlimit(RLIMIT_AS, &rl);
  rl.rlim_cur = static_cast<rlim_t>(vm + static_cast<std::int64_t>(1.25 * kGiB));
  if (vm <= 0 || ::setrlimit(RLIMIT_AS, &rl) != 0) {
    std::fprintf(stderr, "cannot set RLIMIT_AS\n");
    std::_Exit(3);
  }
  hs::Solution<double> sol;
  try {
    sol = hs::solve(a, nullptr, cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "solve under RLIMIT_AS failed: %s\n", e.what());
    std::_Exit(2);
  }
  const auto& r = sol.report;
  auto want = nearest_of(w, sigma, 8);
  auto got = sol.eigenvalues;
  std::sort(got.begin(), got.end());
  bool same = got.size() == want.size();
  for (std::size_t i = 0; same && i < got.size(); ++i) same = std::abs(got[i] - want[i]) <= 1e-9;
  std::fprintf(stderr, "measure %s, limit %.2f GiB (%s), reserved %.2f GiB, budget %.2f GiB, cap %.0f MB, "
               "reserved by MUMPS %.0f MB, peak address space %.2f GiB\n",
               hs::to_string(r.memory.measure), r.memory.limit_gb, r.memory.limit_source.c_str(),
               r.memory.reserved_gb, r.memory.budget_gb, r.factor_options.max_memory_mb,
               r.factor_stats.workspace_reserved_mb, static_cast<double>(hs::peak_address_space_bytes()) / kGiB);
  if (!r.success || !same || r.memory.measure != hs::MemoryMeasure::AddressSpace ||
      r.memory.limit_source != "rlimit_as" || !(r.memory.reserved_gb > 0)) {
    std::fprintf(stderr, "wrong result under RLIMIT_AS (success %d, eigenvalues %s)\n", r.success ? 1 : 0,
                 same ? "ok" : "wrong");
    std::_Exit(1);
  }
  std::fprintf(stderr, "address-space solve ok\n");
  std::_Exit(0);
}

// A solve under a real RLIMIT_AS, in a child process (threadsafe death-test style: the test binary
// is re-executed, so the child starts without this process's threads and memory).
TEST(DriverMemoryBudgetDeathTest, AnAddressSpaceLimitIsHonoured) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(solve_under_an_address_space_limit(), ::testing::ExitedWithCode(0), "address-space solve ok");
}

}  // namespace
