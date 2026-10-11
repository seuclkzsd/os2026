// The adaptive ordering of P5 (module anaint; src/driver/steps.cpp): the planned work of a solve,
// the ordering cost model, the choice among AMD / hsolve's nested dissection / METIS / PORD and the
// analysis loop of adaptive_analysis (determinism, second analyses only where they pay).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;
using hs::kOrderingNestedDissection;

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

// ------------------------------------------------------------------------------ planned work

TEST(DriverPlannedWork, FollowsTheProblemTypeAndTheBlockRules) {
  using hs::ProblemType;
  using hs::driver::plan_work;
  auto cfg = nearest_cfg(0.5, 12);
  // HEP nearest, k >= 8: block width 4, 40 + 4k applications, two certificate counts.
  auto w = plan_work(cfg, ProblemType::Standard, 100000, 1.0);
  EXPECT_EQ(w.block, 4);
  EXPECT_DOUBLE_EQ(w.solve_calls, (40.0 + 48.0) / 4.0);
  EXPECT_DOUBLE_EQ(w.factorizations, 1.0);
  EXPECT_DOUBLE_EQ(w.counts, 2.0);
  // B's inertia still open: one factorization more.
  EXPECT_DOUBLE_EQ(plan_work(cfg, ProblemType::Auto, 100000, 1.0).factorizations, 2.0);
  // No certificate, no counts; k < 8: single vector.
  cfg.inertia_check = hs::Toggle::Off;
  EXPECT_DOUBLE_EQ(plan_work(cfg, ProblemType::GeneralizedDefinite, 100000, 1.0).counts, 0.0);
  cfg.target.k = 6;
  w = plan_work(cfg, ProblemType::Standard, 100000, 1.0);
  EXPECT_EQ(w.block, 1);
  EXPECT_DOUBLE_EQ(w.solve_calls, 24.0);
  // GHIEP: no counts, the GHIEP block rule (p = 2 from n = 2e4), one-sided runs for a shift in the
  // zero band.
  cfg = nearest_cfg(0.5, 18);
  w = plan_work(cfg, ProblemType::GeneralizedIndefinite, 100000, 1.0);
  EXPECT_EQ(w.block, 2);
  EXPECT_DOUBLE_EQ(w.counts, 0.0);
  EXPECT_DOUBLE_EQ(w.factorizations, 1.0);
  EXPECT_EQ(plan_work(cfg, ProblemType::GeneralizedIndefinite, 10000, 1.0).block, 1);
  cfg.target.sigma = 0.0;
  w = plan_work(cfg, ProblemType::GeneralizedIndefinite, 100000, 1.0);
  EXPECT_DOUBLE_EQ(w.factorizations, 5.0);
  EXPECT_GT(w.solve_calls, 100.0);
}

// ------------------------------------------------------------------------------ cost model

TEST(DriverOrderingModel, AnalysisTimes) {
  using hs::driver::ordering_analysis_s;
  const hs::driver::MumpsInputSize m{97336.0, 1306396.0};  // syn-mem27-46
  const hs::driver::OrderingInput in{m, m};
  const double fe = 5e7;
  // MUMPS orderings are serial; hsolve's nested dissection gets faster with threads, down to its
  // serial part.
  EXPECT_EQ(ordering_analysis_s(5, in, fe, 1), ordering_analysis_s(5, in, fe, 16));
  EXPECT_GT(ordering_analysis_s(kOrderingNestedDissection, in, fe, 1),
            ordering_analysis_s(kOrderingNestedDissection, in, fe, 4));
  EXPECT_GT(ordering_analysis_s(kOrderingNestedDissection, in, fe, 4),
            ordering_analysis_s(kOrderingNestedDissection, in, fe, 64));
  // AMD is the cheapest; at 4 threads hsolve's nested dissection undercuts the MUMPS ones.
  for (const int o : {kOrderingNestedDissection, 5, 4, 3})
    EXPECT_LT(ordering_analysis_s(0, in, fe, 4), ordering_analysis_s(o, in, fe, 4)) << o;
  for (const int o : {5, 4, 3})
    EXPECT_LT(ordering_analysis_s(kOrderingNestedDissection, in, fe, 4), ordering_analysis_s(o, in, fe, 4)) << o;
  // Larger inputs cost more; the compressed graph of a mostly complex pencil costs hsolve's nested
  // dissection less (MUMPS orders the full embedding).
  const hs::driver::MumpsInputSize m2{2 * m.order, 2 * m.entries};
  EXPECT_LT(ordering_analysis_s(0, in, fe, 4), ordering_analysis_s(0, {m2, m2}, fe, 4));
  const hs::driver::OrderingInput compressed{m2, m};
  EXPECT_LT(ordering_analysis_s(kOrderingNestedDissection, compressed, fe, 4),
            ordering_analysis_s(kOrderingNestedDissection, {m2, m2}, fe, 4));
  EXPECT_EQ(ordering_analysis_s(5, compressed, fe, 4), ordering_analysis_s(5, {m2, m2}, fe, 4));
}

TEST(DriverOrderingModel, CompressedGraphOfMostlyComplexPencils) {
  const auto real = hs::test::laplacian_2d<double>(20, 15);
  const hs::ShiftedPencil<double> pr(real, nullptr);
  const auto r = hs::driver::ordering_input(pr);
  EXPECT_EQ(r.nd.order, r.mumps.order);
  // Every coupling complex: hsolve orders the n-vertex graph of K.
  auto c = hs::to_complex(real);
  for (index_t i = 0; i < c.nrows; ++i)
    for (index_t p = c.rowptr[i]; p < c.rowptr[i + 1]; ++p)
      if (c.colind[p] != i) c.values[p] = cplx(c.values[p].real(), c.colind[p] > i ? 0.25 : -0.25);
  const hs::ShiftedPencil<cplx> pc(c, nullptr);
  const auto x = hs::driver::ordering_input(pc);
  EXPECT_EQ(x.mumps.order, 2.0 * c.nrows);
  EXPECT_EQ(x.nd.order, static_cast<double>(c.nrows));
  EXPECT_EQ(x.nd.entries, static_cast<double>(pc.nnz()));
  // Real values in complex storage: the embedding decouples, the full graph is ordered.
  const auto rc = hs::to_complex(real);
  const hs::ShiftedPencil<cplx> prc(rc, nullptr);
  EXPECT_EQ(hs::driver::ordering_input(prc).nd.order, 2.0 * rc.nrows);
}

TEST(DriverOrderingModel, NumericWorkGrowsWithTheFactorsAndTheWork) {
  using hs::driver::planned_numeric_s;
  const auto cfg = nearest_cfg(0.0, 12);
  const hs::driver::MumpsInputSize in{97336.0, 1306396.0};
  const auto w = hs::driver::plan_work(cfg, hs::ProblemType::Standard, 97336, 1.0);
  const double base = planned_numeric_s(cfg, w, in, 7e10, 5e7, 4);
  EXPECT_GT(base, 0.0);
  EXPECT_LT(base, planned_numeric_s(cfg, w, in, 2e11, 5e7, 4));
  EXPECT_LT(base, planned_numeric_s(cfg, w, in, 7e10, 1e8, 4));
  EXPECT_GT(base, planned_numeric_s(cfg, w, in, 7e10, 5e7, 16));  // more threads, less time
  auto more = w;
  more.factorizations += 3;
  EXPECT_LT(base, planned_numeric_s(cfg, more, in, 7e10, 5e7, 4));
  more = w;
  more.solve_calls *= 2;
  EXPECT_LT(base, planned_numeric_s(cfg, more, in, 7e10, 5e7, 4));
  // A wider block costs more per call, less per column.
  more = w;
  more.block = 8;
  EXPECT_LT(base, planned_numeric_s(cfg, more, in, 7e10, 5e7, 4));
  more.solve_calls = w.solve_calls * 4.0 / 8.0;
  EXPECT_GT(base, planned_numeric_s(cfg, more, in, 7e10, 5e7, 4));
}

TEST(DriverOrderingModel, ShapePredictions) {
  using hs::driver::predicted_shape;
  hs::FactorEstimate amd;
  amd.flops = 2.6e11;
  amd.factor_entries = 88600000;
  const double nnz = 874000.0;  // fill ~ 100: 3-D-like
  const auto self = predicted_shape(0, amd, nnz, 0, amd);
  EXPECT_DOUBLE_EQ(self.first, amd.flops);
  EXPECT_DOUBLE_EQ(self.second, static_cast<double>(amd.factor_entries));
  for (const int o : {kOrderingNestedDissection, 5, 4}) {
    const auto s = predicted_shape(o, amd, nnz, 0, amd);
    EXPECT_LT(s.first, 0.6 * amd.flops) << o;  // nested dissection: far less fill on 3-D graphs
    EXPECT_LT(s.second, static_cast<double>(amd.factor_entries)) << o;
  }
  // Predicted from an analysed nested dissection: its own prediction is its estimate.
  hs::FactorEstimate nd;
  nd.flops = 5.6e10;
  nd.factor_entries = 42500000;
  const auto from_nd = predicted_shape(kOrderingNestedDissection, amd, nnz, kOrderingNestedDissection, nd);
  EXPECT_DOUBLE_EQ(from_nd.first, nd.flops);
  // 2-D-like fill: nested dissection keeps most of AMD's flops.
  const auto flat = predicted_shape(kOrderingNestedDissection, amd, amd.factor_entries / 9.0, 0, amd);
  EXPECT_GT(flat.first, 0.5 * amd.flops);
}

TEST(DriverOrderingModel, ChoiceWithMargin) {
  using hs::driver::choose_ordering;
  using hs::driver::kOrderingMargin;
  std::vector<hs::driver::OrderingCandidate> c(3);
  c[0].score_s = 1.0;
  c[1].score_s = 1.0 - 0.5 * kOrderingMargin;  // within the margin: the earlier candidate stays
  c[2].score_s = 2.0;
  EXPECT_EQ(choose_ordering(c), 0u);
  c[1].score_s = 1.0 - 2.0 * kOrderingMargin;
  EXPECT_EQ(choose_ordering(c), 1u);
  c[2].score_s = 0.5;
  EXPECT_EQ(choose_ordering(c), 2u);
  EXPECT_EQ(choose_ordering(std::vector<hs::driver::OrderingCandidate>(1)), 0u);
}

TEST(DriverOrderingModel, CandidatesExcludeScotch) {
  const auto o = hs::driver::candidate_orderings(hs::SolverConfig{});
  ASSERT_FALSE(o.empty());
  EXPECT_EQ(o.front(), 0);  // the AMD probe comes first
  EXPECT_EQ(std::count(o.begin(), o.end(), 3), 0);
  EXPECT_EQ(std::count(o.begin(), o.end(), kOrderingNestedDissection), 1);
}

// ------------------------------------------------------------------------------ adaptive analysis

struct Analysed {
  int choice = -1;
  std::string candidates;
};

template <class T>
Analysed run_adaptive(const hs::CsrMatrix<T>& a, const hs::SolverConfig& cfg, const hs::driver::PlannedWork& work,
                      const std::string& tag, double cap_mb = 0.0) {
  const hs::ShiftedPencil<T> pencil(a, nullptr);
  const std::string path = ::testing::TempDir() + "/anaint_ordering_" + tag + ".jsonl";
  Analysed out;
  {
    hs::Logger log(0, path);
    hs::SolveReport rep;
    hs::driver::PhaseClock clock;
    hs::driver::Ctx c{cfg, &log, rep, clock};
    const hs::FactorOptions fo = hs::driver::factor_options_from(cfg, cap_mb);
    auto fac = hs::driver::adaptive_analysis<T>(c, fo, pencil, cfg.target.sigma, work);
    out.choice = fac->options().ordering;
  }
  const auto ev = events(path, "ordering");
  EXPECT_EQ(ev.size(), 1u);
  if (!ev.empty()) {
    EXPECT_EQ(ev[0].at("choice").get<int>(), out.choice);
    out.candidates = ev[0].at("candidates").get<std::string>();
  }
  return out;
}

int analysed_count(const std::string& s) {
  int n = 0;
  for (std::size_t p = s.find(":a"); p != std::string::npos; p = s.find(":a", p + 1)) ++n;
  return n;
}

// 2-D fill: AMD; 3-D fill: hsolve's nested dissection (cheapest analysis, fill of METIS), each
// with a single analysis beyond the probe at most; the same choice in every run.
TEST(DriverAdaptiveOrdering, AmdOnTwoDimensionalAndNestedDissectionOnThreeDimensionalFill) {
  auto cfg = nearest_cfg(0.0, 12);
  cfg.threads = 4;
  const auto flat = hs::test::laplacian_2d<double>(300, 200);
  const auto wf = hs::driver::plan_work(cfg, hs::ProblemType::Standard, flat.nrows, 8.0);
  const auto r2 = run_adaptive(flat, cfg, wf, "2d");
  EXPECT_EQ(r2.choice, 0) << r2.candidates;
  EXPECT_EQ(analysed_count(r2.candidates), 1) << r2.candidates;
  const auto cube = laplacian_3d<double>(30, 30, 30);
  const auto wc = hs::driver::plan_work(cfg, hs::ProblemType::Standard, cube.nrows, 12.0);
  const auto r3 = run_adaptive(cube, cfg, wc, "3d");
  EXPECT_EQ(r3.choice, kOrderingNestedDissection) << r3.candidates;
  EXPECT_EQ(analysed_count(r3.candidates), 2) << r3.candidates;
  const auto again = run_adaptive(cube, cfg, wc, "3d_again");
  EXPECT_EQ(again.choice, r3.choice);
  EXPECT_EQ(again.candidates.substr(0, again.candidates.find('/')), r3.candidates.substr(0, r3.candidates.find('/')));
}

// When the numeric work dwarfs an analysis, the expected gain of an actual estimate pays for more
// analyses than one: here 2000 planned factorizations of a 3-D Laplacian.
TEST(DriverAdaptiveOrdering, HeavyWorkAnalysesMoreCandidates) {
  auto cfg = nearest_cfg(0.0, 12);
  cfg.threads = 2;
  const auto cube = laplacian_3d<cplx>(16, 16, 16);
  auto w = hs::driver::plan_work(cfg, hs::ProblemType::Standard, cube.nrows, 12.0);
  const auto light = run_adaptive(cube, cfg, w, "light");
  w.factorizations = 2000.0;
  const auto heavy = run_adaptive(cube, cfg, w, "heavy");
  EXPECT_GT(analysed_count(heavy.candidates), analysed_count(light.candidates)) << light.candidates << " | "
                                                                                 << heavy.candidates;
  EXPECT_NE(heavy.choice, 0);
}

// A candidate whose factors would not fit the in-core cap loses to any that fits: a 2-D Laplacian
// takes AMD, unless the cap lies between hsolve's nested dissection's estimate and AMD's.
TEST(DriverAdaptiveOrdering, FactorsBeyondTheCapLoseToOnesThatFit) {
  auto cfg = nearest_cfg(0.0, 12);
  const auto flat = hs::test::laplacian_2d<double>(400, 300);
  const hs::ShiftedPencil<double> pencil(flat, nullptr);
  auto estimate = [&](int ordering) {
    hs::FactorOptions o = hs::driver::factor_options_from(cfg, 0.0);
    o.ordering = ordering;
    auto f = hs::make_mumps_factorization<double>(o);
    f->analyze(pencil, 1.0, 0.0);
    return f->estimate().incore_mb;
  };
  const double amd = estimate(0), nd = estimate(kOrderingNestedDissection);
  ASSERT_LT(nd, 0.95 * amd);
  const auto w = hs::driver::plan_work(cfg, hs::ProblemType::Standard, flat.nrows, 8.0);
  EXPECT_EQ(run_adaptive(flat, cfg, w, "cap_none").choice, 0);
  const auto capped = run_adaptive(flat, cfg, w, "cap_between", 0.5 * (amd + nd));
  EXPECT_NE(capped.choice, 0) << capped.candidates;
  // Nothing fits: the smallest estimate is taken (the budget check then decides about out-of-core).
  const auto none = run_adaptive(flat, cfg, w, "cap_tiny", 0.01 * nd);
  EXPECT_NE(none.choice, 0) << none.candidates;
}

// End to end: the solve takes the adaptive choice and succeeds; deterministic runs keep the P4 rule.
TEST(DriverAdaptiveOrdering, SolveUsesTheChoice) {
  const auto cube = laplacian_3d<double>(31, 30, 29);
  auto cfg = nearest_cfg(0.9, 10);
  const auto sol = hs::solve(cube, nullptr, cfg);
  EXPECT_TRUE(sol.report.success);
  EXPECT_EQ(sol.report.factor_options.ordering, kOrderingNestedDissection);
  EXPECT_NE(sol.report.factor_options.ordering, 3);
}

}  // namespace
