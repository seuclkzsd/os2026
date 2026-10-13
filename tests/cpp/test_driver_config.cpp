// Tests for config.json parsing/serialization and report.json serialization (src/driver).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

TEST(DriverConfig, ParsesDocumentedExample) {
  const char* doc = R"({
    "target": {"mode": "nearest", "sigma": -2.5e5, "k": 10},
    "tol": 1e-11, "problem": "generalized_indefinite", "method": "shift_invert_krylov_schur",
    "ncv": 40, "seed": 7, "threads": 8, "memory_budget_gb": 28.5, "factor_precision": "single",
    "blr": "on", "out_of_core": "off", "inertia_check": "auto", "zero_filter_tol": 1e-6,
    "complex_tol": 1e-9, "hermitian_tol": 1e-12, "max_shift_retries": 3, "inner_tol_factor": 0.5,
    "ordering": 5, "max_restarts": 77, "block_size": 4, "inertia_counter": "off"})";
  const hs::SolverConfig c = hs::parse_config(doc);
  EXPECT_EQ(c.target.mode, hs::TargetMode::Nearest);
  EXPECT_EQ(c.target.sigma, -2.5e5);
  EXPECT_EQ(c.target.k, 10);
  EXPECT_EQ(c.tol, 1e-11);
  EXPECT_EQ(c.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_EQ(c.method, hs::Method::ShiftInvertKrylovSchur);
  EXPECT_EQ(c.ncv, 40);
  EXPECT_EQ(c.seed, 7u);
  EXPECT_EQ(c.threads, 8);
  EXPECT_EQ(c.memory_budget_gb, 28.5);
  EXPECT_EQ(c.factor_precision, hs::FactorPrecision::Single);
  EXPECT_EQ(c.blr, hs::Toggle::On);
  EXPECT_EQ(c.out_of_core, hs::Toggle::Off);
  EXPECT_EQ(c.zero_filter_tol, 1e-6);
  EXPECT_EQ(c.max_shift_retries, 3);
  EXPECT_EQ(c.ordering, 5);
  EXPECT_EQ(c.max_restarts, 77);
  EXPECT_EQ(c.block_size, 4);
  EXPECT_EQ(c.inertia_counter, hs::Toggle::Off);
}

TEST(DriverConfig, BlockSizeAndInertiaCounterRoundTrip) {
  const hs::SolverConfig def;
  EXPECT_EQ(def.block_size, 0);  // auto
  EXPECT_EQ(def.inertia_counter, hs::Toggle::Auto);
  const auto j = nlohmann::json::parse(hs::config_to_json(def));
  EXPECT_EQ(j["block_size"], 0);
  EXPECT_EQ(j["inertia_counter"], "auto");
  hs::SolverConfig c;
  c.block_size = 8;
  c.inertia_counter = hs::Toggle::On;
  const auto again = hs::parse_config(hs::config_to_json(c));
  EXPECT_EQ(again.block_size, 8);
  EXPECT_EQ(again.inertia_counter, hs::Toggle::On);
  EXPECT_EQ(hs::parse_config(R"({"block_size": 1})").block_size, 1);
  auto bad = [](const char* doc) {
    try {
      (void)hs::parse_config(doc);
      return false;
    } catch (const hs::Error& e) {
      return e.code() == hs::Error::Code::InvalidInput;
    }
  };
  EXPECT_TRUE(bad(R"({"block_size": -1})"));
  EXPECT_TRUE(bad(R"({"block_size": 2.5})"));
  EXPECT_TRUE(bad(R"({"block_size": "4"})"));
  EXPECT_TRUE(bad(R"({"inertia_counter": "yes"})"));
}

// ordering: -1 adaptive, MUMPS ICNTL(7) 0 / 2..7, or 10 = hsolve's nested dissection
// (kOrderingNestedDissection); 1 (a caller-given pivot order) and anything else are rejected.
TEST(DriverConfig, OrderingAcceptsMumpsCodesAndOwnNestedDissection) {
  for (int o : {-1, 0, 2, 3, 4, 5, 6, 7, hs::kOrderingNestedDissection}) {
    const auto c = hs::parse_config(R"({"ordering": )" + std::to_string(o) + "}");
    EXPECT_EQ(c.ordering, o);
    EXPECT_EQ(hs::parse_config(hs::config_to_json(c)).ordering, o);
  }
  EXPECT_EQ(hs::kOrderingNestedDissection, 10);
  for (int o : {-2, 1, 8, 9, 11}) {
    try {
      (void)hs::parse_config(R"({"ordering": )" + std::to_string(o) + "}");
      ADD_FAILURE() << "ordering " << o << " accepted";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << o;
      EXPECT_NE(std::string(e.what()).find("hsolve's nested dissection"), std::string::npos) << e.what();
    }
  }
}

// ordering = 10 reaches the factorization: hsolve's nested dissection orders the main analysis
// (and the certificate's counts) and the solve finds the nearest eigenvalues.
TEST(DriverConfig, OrderingTenSolvesWithOwnNestedDissection) {
  const auto a = hs::test::laplacian_2d<double>(30, 31);
  auto cfg = hs::parse_config(R"({"target": {"mode": "nearest", "sigma": 0.5, "k": 5}, "ordering": 10,
                                   "threads": 2, "tol": 1e-10})");
  const auto sol = hs::solve(a, nullptr, cfg);
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.factor_options.ordering, hs::kOrderingNestedDissection);
  EXPECT_TRUE(sol.report.certificate.passed);
  std::vector<double> want = hs::test::laplacian_2d_eigenvalues(30, 31), got = sol.eigenvalues;
  std::stable_sort(want.begin(), want.end(),
                   [](double x, double y) { return std::abs(x - 0.5) < std::abs(y - 0.5); });
  want.resize(5);
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], 1e-10) << i;
}

TEST(DriverConfig, MissingKeysKeepBaseAndRoundTrip) {
  hs::SolverConfig base;
  base.tol = 3e-9;
  base.target.k = 4;
  const auto c = hs::parse_config(R"({"target": {"mode": "interval", "lower": -1, "upper": 2}})", base);
  EXPECT_EQ(c.tol, 3e-9);
  EXPECT_EQ(c.target.mode, hs::TargetMode::Interval);
  EXPECT_EQ(c.target.lower, -1.0);
  const auto again = hs::parse_config(hs::config_to_json(c));
  EXPECT_EQ(hs::config_to_json(again), hs::config_to_json(c));
}

TEST(DriverConfig, RejectsUnknownKeysBadTypesAndValues) {
  auto bad = [](const char* doc) {
    try {
      (void)hs::parse_config(doc);
      return false;
    } catch (const hs::Error& e) {
      return e.code() == hs::Error::Code::InvalidInput;
    }
  };
  EXPECT_TRUE(bad(R"({"tolerance": 1e-8})"));
  EXPECT_TRUE(bad(R"({"target": {"sigma": 0, "kk": 3}})"));
  EXPECT_TRUE(bad(R"({"tol": "1e-8"})"));
  EXPECT_TRUE(bad(R"({"blr": "maybe"})"));
  EXPECT_TRUE(bad(R"({"ncv": 2.5})"));
  EXPECT_TRUE(bad(R"({"tol": 0})"));
  EXPECT_TRUE(bad(R"({"target": {"mode": "interval", "lower": 2, "upper": 1}})"));
  EXPECT_TRUE(bad(R"([1, 2])"));
  EXPECT_TRUE(bad(R"({not json)"));
}

TEST(DriverReport, JsonHasDocumentedKeysAndNullForInfiniteBounds) {
  hs::SolveReport r;
  r.n = 5;
  r.problem = hs::ProblemType::GeneralizedDefinite;
  r.method = hs::Method::ShiftInvertLanczos;
  r.pairs.push_back({1.5, 0.0, 1e-14, 1e-15, 1e-16, true});
  r.certificate.performed = true;
  r.certificate.lower = -std::numeric_limits<double>::infinity();
  r.certificate.upper = 2.0;
  r.warnings.push_back("w");
  hs::SliceReport sl;
  sl.lower = -std::numeric_limits<double>::infinity();
  sl.upper = 0.75;
  sl.sigma = 0.75;
  sl.which = "smallest_algebraic";
  sl.count_inertia = 4;
  sl.count_found = 4;
  sl.nev = 4;
  sl.passed = true;
  r.slices.push_back(sl);
  r.zero_cluster_dim = 30;
  r.zero_modes_filtered = 12;
  const auto j = nlohmann::json::parse(r.to_json());
  for (const char* key : {"schema_version", "n", "nnz_a", "nnz_b", "complex_scalars", "problem", "norm2_a", "norm2_b",
                          "scale", "method", "sigma_requested", "sigma_used", "shift_retries", "factor_options",
                          "factor_stats", "restarts", "op_applies", "time_total_s", "time_setup_s", "time_factor_s",
                          "time_iterate_s", "time_verify_s", "time_certificate_s", "peak_rss_bytes", "pairs",
                          "certificate", "slices", "zero_cluster_dim", "zero_modes_filtered", "success", "warnings",
                          "block_size", "solver", "version"})
    EXPECT_TRUE(j.contains(key)) << key;
  EXPECT_EQ(j["problem"], "generalized_definite");
  EXPECT_EQ(j["solver"], "hsolve");
  EXPECT_EQ(j["version"], std::string(hs::version()));
  EXPECT_EQ(j["block_size"], 1);
  EXPECT_EQ(j["method"], "shift_invert_lanczos");
  EXPECT_TRUE(j["certificate"]["lower"].is_null());
  EXPECT_EQ(j["certificate"]["upper"], 2.0);
  ASSERT_EQ(j["slices"].size(), 1u);
  for (const char* key :
       {"lower", "upper", "sigma", "which", "count_inertia", "count_found", "nev", "reruns", "restarts", "op_applies",
        "passed"})
    EXPECT_TRUE(j["slices"][0].contains(key)) << key;
  EXPECT_TRUE(j["slices"][0]["lower"].is_null());  // open (-inf) slice end
  EXPECT_EQ(j["slices"][0]["upper"], 0.75);
  EXPECT_EQ(j["slices"][0]["which"], "smallest_algebraic");
  EXPECT_EQ(j["zero_cluster_dim"], 30);
  EXPECT_EQ(j["zero_modes_filtered"], 12);
  EXPECT_EQ(j["pairs"][0]["lambda_re"], 1.5);
  EXPECT_TRUE(j["factor_stats"].contains("estimate"));
}

}  // namespace
