// Multi-process solves without MPI (module src/parallel; DESIGN §3.9): the session protocol between
// a leader and its helpers over thread transports (make_thread_transports - the helpers' MUMPS calls
// are serialized by the process-wide MUMPS lock, so this checks the logic, not the speed), the
// message encoding, the driver hooks (two certificate counts on two helpers, spectrum slices on the
// helpers), their limits (fewer than two helpers, the memory budget) and failures (a helper task
// that fails is done by the leader: fault bit 128), real and complex. The same checks over MPI with
// 1, 2 and 3 ranks: tests/cpp/mpi/test_mpi_session.cpp.
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "driver/helpers.hpp"
#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/parallel.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "parallel/protocol.hpp"
#include "parallel/remote_helpers.hpp"
#include "parallel/wire.hpp"
#include "support/parallel_problems.hpp"

namespace {

using hs::cplx;
using hs::index_t;
using namespace hs::partest;

// ------------------------------------------------------------------------------- encoding

TEST(ParallelWire, RoundTripOfEveryKind) {
  hs::parallel::Writer w(hs::parallel::Msg::Count);
  const auto a = hs::partest::laplacian3d<cplx>();
  w.pod(std::uint32_t{77});
  w.str("helper");
  w.strs({"a", "", "bc"});
  w.vec(std::vector<int>{3, 1, 2});
  w.csr(a);
  const std::vector<double> x{1.0, -2.5, 3.25};
  w.raw(x.data(), x.size());
  hs::FactorOptions fo;
  fo.threads = 5;
  fo.ooc_dir = "/tmp/x";
  fo.precision = hs::FactorPrecision::Single;
  hs::parallel::put(w, fo);
  hs::parallel::Reader r(w.bytes());
  EXPECT_EQ(r.type(), hs::parallel::Msg::Count);
  EXPECT_EQ(r.pod<std::uint32_t>(), 77u);
  EXPECT_EQ(r.str(), "helper");
  EXPECT_EQ(r.strs(), (std::vector<std::string>{"a", "", "bc"}));
  EXPECT_EQ(r.vec<int>(), (std::vector<int>{3, 1, 2}));
  const auto b = r.csr<cplx>();
  EXPECT_EQ(b.nrows, a.nrows);
  EXPECT_EQ(b.rowptr, a.rowptr);
  EXPECT_EQ(b.colind, a.colind);
  EXPECT_EQ(b.values, a.values);
  std::vector<double> y(3);
  r.raw(y.data(), y.size());
  EXPECT_EQ(y, x);
  const hs::FactorOptions g = hs::parallel::get_factor_options(r);
  EXPECT_EQ(g.threads, 5);
  EXPECT_EQ(g.ooc_dir, "/tmp/x");
  EXPECT_EQ(g.precision, hs::FactorPrecision::Single);
  EXPECT_TRUE(r.done());
}

TEST(ParallelWire, TruncatedMessagesThrowInternal) {
  hs::parallel::Writer w(hs::parallel::Msg::Setup);
  w.vec(std::vector<double>(100, 1.0));
  std::vector<std::uint8_t> cut(w.bytes().begin(), w.bytes().begin() + 50);
  hs::parallel::Reader r(cut);
  r.type();
  try {
    r.vec<double>();
    FAIL() << "a truncated array was read";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::Internal);
  }
  std::vector<double> y(3);
  hs::parallel::Writer w2(hs::parallel::Msg::Count);
  w2.raw(y.data(), 2);
  hs::parallel::Reader r2(w2.bytes());
  r2.type();
  EXPECT_THROW(r2.raw(y.data(), 3), hs::Error);  // a size other than expected
}

TEST(ParallelWire, SliceReplyKeepsVectorsAndReport) {
  hs::driver::SliceReply<cplx> s;
  s.report.which = "largest_magnitude";
  s.report.count_found = 2;
  s.lambda = {1.0, 2.0};
  s.x.resize(4, 2);
  for (index_t j = 0; j < 2; ++j)
    for (index_t i = 0; i < 4; ++i) s.x(i, j) = cplx(static_cast<double>(i), static_cast<double>(j));
  s.metrics.resize(2);
  s.metrics[1].backward_error = 1e-12;
  s.warnings = {"w"};
  s.block_size = 4;
  hs::parallel::Writer w(hs::parallel::Msg::SliceDone);
  hs::parallel::put(w, s);
  hs::parallel::Reader r(w.bytes());
  r.type();
  const auto t = hs::parallel::get_slice_reply<cplx>(r);
  EXPECT_EQ(t.report.which, "largest_magnitude");
  EXPECT_EQ(t.report.count_found, 2);
  EXPECT_EQ(t.lambda, s.lambda);
  ASSERT_EQ(t.x.rows(), 4);
  ASSERT_EQ(t.x.cols(), 2);
  EXPECT_EQ(t.x(3, 1), cplx(3.0, 1.0));
  EXPECT_EQ(t.metrics[1].backward_error, 1e-12);
  EXPECT_EQ(t.warnings, s.warnings);
  EXPECT_EQ(t.block_size, 4);
  EXPECT_TRUE(r.done());
}

// ------------------------------------------------------------------------------- transport

TEST(ParallelThreadTransport, OrderPerSenderAndAnySource) {
  auto t = hs::make_thread_transports(3);
  ASSERT_EQ(t.size(), 3u);
  EXPECT_EQ(t[2]->rank(), 2);
  EXPECT_EQ(t[0]->size(), 3);
  t[1]->send(0, {1});
  t[2]->send(0, {2});
  t[1]->send(0, {3});
  EXPECT_TRUE(t[0]->ready(1));
  EXPECT_FALSE(t[1]->ready(hs::Transport::kAnySource));
  int from = -1;
  EXPECT_EQ(t[0]->receive(2, &from), std::vector<std::uint8_t>{2});
  EXPECT_EQ(from, 2);
  EXPECT_EQ(t[0]->receive(hs::Transport::kAnySource, &from), std::vector<std::uint8_t>{1});
  EXPECT_EQ(from, 1);
  EXPECT_EQ(t[0]->receive(1), std::vector<std::uint8_t>{3});
  EXPECT_THROW(t[0]->send(0, {1}), hs::Error);
  EXPECT_THROW(hs::make_thread_transports(0), hs::Error);
}

// ------------------------------------------------------------------------------- sessions

/// A session of `ranks` thread-ranks: helpers serve() on their threads while `leader` runs on this
/// one; the helpers are released however the leader ends.
template <class Fn>
void with_session(int ranks, Fn leader) {
  auto t = hs::make_thread_transports(ranks);
  std::vector<std::thread> helpers;
  for (int r = 1; r < ranks; ++r) helpers.emplace_back([&t, r]() { hs::serve(*t[static_cast<std::size_t>(r)]); });
  try {
    leader(*t[0]);
  } catch (...) {
    hs::release_helpers(*t[0]);
    for (auto& h : helpers) h.join();
    throw;
  }
  hs::release_helpers(*t[0]);
  for (auto& h : helpers) h.join();
}

template <class T>
class ParallelSessionTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(ParallelSessionTyped, Scalars);

// One rank is hs::solve itself.
TYPED_TEST(ParallelSessionTyped, OneRankIsTheSingleProcessSolve) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  with_session(1, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers, kForce);
    expect_nearest_certified(sol);
    EXPECT_EQ(sol.report.parallel.processes, 1);
    EXPECT_EQ(sol.report.parallel.helper_counts, 0);
  });
}

// Three ranks: the certificate's two counts are made by the two helpers, and the result equals the
// single-process one (the counts are exact integers; the pairs the same up to rounding).
TYPED_TEST(ParallelSessionTyped, TwoHelpersMakeTheCertificateCounts) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  const auto ref = hs::solve(a, nullptr, nearest_cfg());
  with_session(3, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers, kForce);
    expect_nearest_certified(sol);
    const auto& p = sol.report.parallel;
    EXPECT_EQ(p.processes, 3);
    EXPECT_EQ(p.helper_counts, 2) << warnings(sol.report);
    EXPECT_EQ(p.fallbacks, 0);
    EXPECT_GT(p.helper_threads, 0);
    EXPECT_GT(p.share_s, 0.0);
    EXPECT_GT(p.helper_busy_s, 0.0);
    EXPECT_EQ(sol.report.certificate.count_inertia, ref.report.certificate.count_inertia);
    EXPECT_NE(sol.report.certificate.statement.find("2 of the 2 counts made by helper processes"), std::string::npos)
        << sol.report.certificate.statement;
    expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-12);
    // A second solve in the same session (the helpers dropped the first problem).
    const auto again = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers, kForce);
    expect_nearest_certified(again);
    EXPECT_EQ(again.report.parallel.helper_counts, 2);
  });
}

// Two ranks: one helper cannot count concurrently with the leader without sharing its cores (the
// leader's threads span every socket), so the leader counts as hs::solve does.
TYPED_TEST(ParallelSessionTyped, OneHelperLeavesTheCountsToTheLeader) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  with_session(2, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers, kForce);
    expect_nearest_certified(sol);
    EXPECT_EQ(sol.report.parallel.processes, 2);
    EXPECT_EQ(sol.report.parallel.helper_counts, 0);
    EXPECT_TRUE(has_note(sol.report, "fewer than two helpers")) << warnings(sol.report);
  });
}

// The helpers solve the slices of an interval target; the merged result is the single-process one.
TYPED_TEST(ParallelSessionTyped, HelpersSolveTheSlices) {
  using T = TypeParam;
  const auto a = laplacian3d<T>();
  const auto ref = hs::solve(a, nullptr, interval_cfg());
  expect_interval_certified(ref);
  for (int ranks : {2, 3}) {
    with_session(ranks, [&](hs::Transport& peers) {
      const auto sol = hs::solve_parallel(a, nullptr, interval_cfg(), nullptr, peers, kForce);
      expect_interval_certified(sol);
      const auto& p = sol.report.parallel;
      if (ranks == 3) {
        EXPECT_EQ(p.helper_slices, static_cast<index_t>(sol.report.slices.size())) << warnings(sol.report);
      } else {
        EXPECT_EQ(p.helper_slices, 0);  // one helper alone would be slower than the leader
      }
      EXPECT_EQ(sol.report.slices.size(), ref.report.slices.size());
      for (std::size_t i = 0; i < ref.report.slices.size(); ++i) {
        EXPECT_EQ(sol.report.slices[i].count_inertia, ref.report.slices[i].count_inertia) << i;
        EXPECT_EQ(sol.report.slices[i].count_found, ref.report.slices[i].count_found) << i;
      }
      expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-12);
    });
  }
}

// The automatic slice size: with two helpers a window of at least two minimum slices (16) gets at
// least two slices, all solved by the helpers, whatever the sequential cost model would choose.
TEST(ParallelSessionSlices, AutoSliceSizeKeepsEveryHelperBusy) {
  const auto a = laplacian3d<double>();
  hs::SolverConfig cfg = interval_cfg();
  cfg.slice_size = 0;
  const auto want = inside(spectrum(), kLower, kUpper);
  ASSERT_GE(want.size(), 32u);
  with_session(3, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, cfg, nullptr, peers, kForce);
    EXPECT_TRUE(sol.report.success) << warnings(sol.report);
    expect_same_set(sol.eigenvalues, want, 1e-9);
    EXPECT_GE(sol.report.slices.size(), 2u);
    EXPECT_EQ(sol.report.parallel.helper_slices, static_cast<index_t>(sol.report.slices.size())) << warnings(sol.report);
  });
}

// Every helper task fails (fault bit 128, each kind): the leader does the work itself, the result
// is certified, the report counts the fallbacks and the helpers are not used again.
TYPED_TEST(ParallelSessionTyped, FailingHelpersAreReplacedByTheLeader) {
  using T = TypeParam;
  namespace d = hs::driver;
  const auto a = laplacian3d<T>();
  for (index_t kind : {index_t{0}, d::kFaultOtherError, d::kFaultBadAlloc}) {
    with_session(3, [&](hs::Transport& peers) {
      const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(d::kFaultHelperTasks | kind), nullptr, peers, kForce);
      expect_nearest_certified(sol);
      const auto& p = sol.report.parallel;
      EXPECT_EQ(p.helper_counts, 0);
      EXPECT_EQ(p.fallbacks, 2) << warnings(sol.report);
      EXPECT_TRUE(has_note(sol.report, "could not count")) << warnings(sol.report);
      EXPECT_TRUE(has_note(sol.report, kind == d::kFaultBadAlloc ? "std::bad_alloc" : "fault injection"))
          << warnings(sol.report);
      const auto iv = hs::solve_parallel(a, nullptr, interval_cfg(d::kFaultHelperTasks | kind), nullptr, peers, kForce);
      expect_interval_certified(iv);
      EXPECT_EQ(iv.report.parallel.helper_slices, 0);
      EXPECT_EQ(iv.report.parallel.fallbacks, 2) << warnings(iv.report);  // the first slice of each helper
      EXPECT_TRUE(has_note(iv.report, "could not solve slice")) << warnings(iv.report);
    });
  }
}

// ------------------------------------------------------------------------------- driver hooks

// The driver's rules for the helpers' work (driver::concurrent_counts, the window's slices) against
// helpers whose replies and memory the test scripts: who counts, the memory rule, a failing helper.
// Memory: `memory` always, plus `prepared_bytes` from prepare() until drop(); or, with leave_bytes
// >= 0, whatever leaves the leader exactly leave_bytes of budget_bytes beside its own RSS.
class FakeHelpers final : public hs::driver::HelpersOf<double> {
 public:
  FakeHelpers(int n, std::vector<double> spectrum) : usable_(static_cast<std::size_t>(n) + 1, true), w_(std::move(spectrum)) {}
  int size() const override { return static_cast<int>(usable_.size()) - 1; }
  const hs::ParallelOptions& options() const override { return opts; }
  bool usable(int h) const override { return usable_.at(static_cast<std::size_t>(h)); }
  void abandon(int h, const std::string&) override { usable_.at(static_cast<std::size_t>(h)) = false; }
  bool shared() const override { return true; }
  void prepare(hs::driver::Ctx&) override {
    ++prepared;
    holds = true;
  }
  void drop(hs::driver::Ctx&, const std::string&) override {
    EXPECT_EQ(busy(), 0) << "drop() with a task in flight";
    ++drops;
    holds = false;
  }
  double memory_bytes() const override {
    if (leave_bytes >= 0)
      return std::max(0.0, budget_bytes - static_cast<double>(hs::current_rss_bytes()) - leave_bytes);
    return memory + (holds ? prepared_bytes : 0.0);
  }
  double count_estimate_mb() const override { return estimate_mb; }
  int helper_threads() const override { return 1; }
  void post_count(hs::driver::Ctx&, int h, const hs::driver::CountTask& t) override { posted.push_back({h, t}); }
  hs::driver::CountReply wait_count(hs::driver::Ctx&, int h) override {
    hs::driver::CountReply r;
    for (const auto& [ph, t] : posted)
      if (ph == h) r.s = t.s;
    if (h == failing) {
      r.error.ok = false;
      r.error.code = hs::Error::Code::FactorizationFailed;
      r.error.what = "scripted failure";
      return r;
    }
    for (double x : w_) r.nu += x < r.s ? 1 : 0;
    r.seconds = 0.5;
    return r;
  }
  void share(hs::driver::Ctx&, const hs::ShiftedPencil<double>&, const hs::SparseFactorization<double>&,
             const hs::driver::ProblemInfo&) override {}
  // Every slice fails on the helper (scripted): the leader solves it.
  void post_slice(hs::driver::Ctx&, int h, const hs::driver::SliceTask& t) override { slices.push_back({h, t}); }
  int wait_slice(hs::driver::Ctx&, hs::driver::SliceReply<double>& out) override {
    const auto& [h, t] = slices.at(static_cast<std::size_t>(slices_done++));
    out = hs::driver::SliceReply<double>{};
    out.error.ok = false;
    out.error.code = hs::Error::Code::FactorizationFailed;
    out.error.what = "scripted failure";
    return h;
  }
  int busy() const override { return static_cast<int>(slices.size()) - slices_done; }

  double memory = 0.0, estimate_mb = 1.0, prepared_bytes = 0.0;
  double leave_bytes = -1.0, budget_bytes = 0.0;
  bool holds = false;
  hs::ParallelOptions opts;
  int failing = 0, prepared = 0, drops = 0, slices_done = 0;
  std::vector<std::pair<int, hs::driver::CountTask>> posted;
  std::vector<std::pair<int, hs::driver::SliceTask>> slices;

 private:
  std::vector<bool> usable_;
  std::vector<double> w_;
};

struct HookFixture {
  static constexpr index_t kN = 300;
  hs::CsrMatrix<double> a = hs::test::laplacian_1d<double>(kN);
  hs::ShiftedPencil<double> pencil{a, nullptr};
  std::unique_ptr<hs::SparseFactorization<double>> fac;
  hs::SolverConfig cfg;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx c{cfg, nullptr, rep, clock};
  std::vector<double> w = hs::test::laplacian_1d_eigenvalues(kN);
  HookFixture() {
    cfg.memory_budget_gb = 4.0;
    hs::FactorOptions fo;
    fo.threads = 1;
    fac = hs::make_mumps_factorization<double>(fo);
    fac->analyze(pencil, 1.0, -1.0);
    EXPECT_TRUE(fac->factorize(1.0, -1.0));
  }
  index_t below(double s) const {
    index_t k = 0;
    for (double x : w) k += x < s ? 1 : 0;
    return k;
  }
  // mid-way between two eigenvalues: a certificate endpoint
  double gap(std::size_t i) const { return 0.5 * (w[i] + w[i + 1]); }
};

TEST(ParallelDriverHooks, TwoHelpersCountBothEnds) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  ASSERT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_EQ(nu_a, 41);
  EXPECT_EQ(nu_b, 61);
  ASSERT_EQ(h.posted.size(), 2u);
  EXPECT_EQ(h.posted[0].second.outward, -1.0);
  EXPECT_EQ(h.posted[1].second.outward, +1.0);
  EXPECT_TRUE(h.posted[0].second.s_is_gap);
  EXPECT_EQ(f.rep.parallel.helper_counts, 2);
  EXPECT_EQ(f.rep.parallel.fallbacks, 0);
  EXPECT_EQ(h.prepared, 1);              // the problem goes to the helpers at their first task
  EXPECT_EQ(h.drops, 1);                 // ... and is dropped once their counts are in
  EXPECT_FALSE(h.holds);
  EXPECT_FALSE(f.fac->inertia().valid);  // the main factors were released before the counts
}

TEST(ParallelDriverHooks, FewerThanTwoHelpersOrNoEstimateCountOnTheLeader) {
  HookFixture f;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  FakeHelpers one(1, f.w);
  f.c.helpers = &one;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  FakeHelpers two(2, f.w);
  two.abandon(1, "test");
  f.c.helpers = &two;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  FakeHelpers none(2, f.w);
  none.estimate_mb = 0.0;
  f.c.helpers = &none;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_TRUE(one.posted.empty() && two.posted.empty() && none.posted.empty());
  EXPECT_TRUE(has_note(f.rep, "fewer than two helpers"));
  EXPECT_TRUE(has_note(f.rep, "could not prepare"));
}

// The helpers' memory counts against the solve's budget: two counts that do not fit beside it stay
// with the leader, which counts one at a time as hs::solve does - decided from the prediction
// before the helpers get the problem when it does not fit there already, else with what the helpers
// report once they have it (then they drop it again).
TEST(ParallelDriverHooks, TwoCountsBeyondTheBudgetStayOnTheLeader) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  h.memory = f.cfg.memory_budget_gb * 1073741824.0;  // all of the 4 GiB budget: nothing fits
  h.estimate_mb = 200.0;
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_TRUE(h.posted.empty());
  EXPECT_EQ(h.prepared, 0);  // nothing was sent
  EXPECT_EQ(h.drops, 0);
  EXPECT_TRUE(has_note(f.rep, "predicted")) << warnings(f.rep);
  // Room for the prediction (tiny here), not for two counts of 200 MB.
  h.memory = f.cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(hs::current_rss_bytes()) - 300e6;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_TRUE(h.posted.empty());
  EXPECT_EQ(h.prepared, 1);
  EXPECT_EQ(h.drops, 1);
  EXPECT_TRUE(has_note(f.rep, "two counts at once (400 MB)")) << warnings(f.rep);
  h.memory = 0.0;
  EXPECT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_EQ(h.drops, 2);
}

// Regression (review p10mpi, finding 1): counts declined once the helpers hold the problem - their
// copies and analyses leave no room for two counts at once - must not leave the helpers holding it:
// they drop it, and the leader's own counts get the budget hs::solve would give them. Before, the
// helpers kept it to the end of the solve, still charged to the budget, and the leader's counter
// and main-object counts were refused for memory: the certificate was not completed where
// hs::solve certifies (syn-mem27-46, --memory-budget-gb 1.0).
TEST(ParallelDriverHooks, DeclinedCountsReleaseTheHelpers) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  h.prepared_bytes = f.cfg.memory_budget_gb * 1073741824.0;  // once they have the problem: all of it
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_EQ(h.prepared, 1);
  EXPECT_EQ(h.drops, 1);
  EXPECT_TRUE(h.posted.empty());
  EXPECT_EQ(h.memory_bytes(), 0.0);
  EXPECT_TRUE(has_note(f.rep, "two counts at once (")) << warnings(f.rep);
  // The leader counts both ends itself, as hs::solve does: within the budget again.
  EXPECT_EQ(hs::driver::count_below(f.c, counts, a, -1.0, 1e-9, nullptr, true), 41);
  EXPECT_EQ(hs::driver::count_below(f.c, counts, b, +1.0, 1e-9, nullptr, true), 61);
  EXPECT_TRUE(counts.uses_counter());
}

// Regression (review p10mpi, finding 2): a window is cut into one slice per helper only when the
// helpers will take the slices. With memory for the leader's slices but not for two helpers'
// slices at once, the window keeps the slices it gets without helper slices (the same cuts and
// factorizations) and nothing is sent; with ample memory it is cut for the helpers (here they fail
// and the leader solves the slices: fallbacks). Before, the window was cut for the helpers and the
// problem sent to them before the memory rule declined them.
TEST(ParallelDriverHooks, WindowCutForHelpersOnlyWhenTheyTakeTheSlices) {
  const auto a = laplacian3d<double>();
  hs::SolverConfig cfg = interval_cfg();
  cfg.slice_size = 0;  // the automatic slice size: the case the helpers' cut applies to
  const double budget = cfg.memory_budget_gb * 1073741824.0;
  const auto want = inside(spectrum(), kLower, kUpper);
  const index_t total = static_cast<index_t>(want.size());
  ASSERT_GE(total, 32);
  auto run = [&](hs::Toggle slices, double leave, FakeHelpers& h) {
    h.opts = {hs::Toggle::Off, slices};
    h.leave_bytes = leave;
    h.budget_bytes = budget;
    return hs::driver::solve_with_helpers<double>(a, nullptr, cfg, nullptr, &h);
  };
  auto certified = [&](const hs::Solution<double>& sol) {
    EXPECT_TRUE(sol.report.success) << warnings(sol.report);
    EXPECT_TRUE(sol.report.certificate.passed) << sol.report.certificate.statement;
    expect_same_set(sol.eigenvalues, want, 1e-9);
  };
  // Ample memory: cut for the two helpers (they fail; the leader solves their slices).
  FakeHelpers ample(2, spectrum());
  const auto cut = run(hs::Toggle::On, 2.0 * 1073741824.0, ample);
  certified(cut);
  EXPECT_EQ(ample.prepared, 1);
  EXPECT_EQ(cut.report.parallel.fallbacks, 2) << warnings(cut.report);
  // A helper's slice: the main analysis' in-core estimate x 1.15 and the Krylov work of its pairs
  // (definite.cpp helper_slice_bytes); `leave` holds the leader's slices but not two of these.
  const index_t per = std::max<index_t>(16, (total + 1) / 2);
  const double incore = cut.report.factor_stats.estimate.incore_mb * 1e6;
  const double need = 1.15 * incore + 7.0 * static_cast<double>(a.nrows) * sizeof(double) * static_cast<double>(per + per / 4);
  // The leader's own slices need its factorization and Krylov basis (~need); two helpers' slices at
  // once 2 x need: leave room between.
  const double leave = 1.4 * need;
  FakeHelpers tight(2, spectrum()), off(2, spectrum());
  const auto kept = run(hs::Toggle::On, leave, tight);
  const auto seq = run(hs::Toggle::Off, leave, off);
  certified(kept);
  certified(seq);
  EXPECT_GE(cut.report.slices.size(), 2u);
  EXPECT_TRUE(has_note(kept.report, "as for one process"))
      << warnings(kept.report) << "need " << need / 1e6 << " MB, leave " << leave / 1e6 << " MB, incore "
      << incore / 1e6 << " MB; slices " << kept.report.slices.size() << " / cut " << cut.report.slices.size();
  ASSERT_EQ(kept.report.slices.size(), seq.report.slices.size());
  for (std::size_t i = 0; i < seq.report.slices.size(); ++i) {
    EXPECT_EQ(kept.report.slices[i].lower, seq.report.slices[i].lower) << i;
    EXPECT_EQ(kept.report.slices[i].count_inertia, seq.report.slices[i].count_inertia) << i;
  }
  EXPECT_EQ(kept.report.factor_stats.factorizations, seq.report.factor_stats.factorizations);
  EXPECT_EQ(tight.prepared, 0);  // nothing was sent
  EXPECT_TRUE(tight.slices.empty());
  EXPECT_EQ(kept.report.parallel.helper_slices, 0);
  EXPECT_EQ(kept.report.parallel.fallbacks, 0);
}

// The cost rule (Auto): counts after a main factorization too short for the helpers' preparation
// stay with the leader; On ignores the rule, Off never hands counts out.
TEST(ParallelDriverHooks, CountsTooCheapForTheHelpersStayOnTheLeader) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 1e-6));
  EXPECT_TRUE(has_note(f.rep, "too cheap for the helpers")) << warnings(f.rep);
  EXPECT_EQ(h.prepared, 0);  // nothing was sent
  EXPECT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 1e3));  // a long one
  h.opts.helper_counts = hs::Toggle::On;
  EXPECT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 1e-6));
  h.opts.helper_counts = hs::Toggle::Off;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 1e3));
  EXPECT_EQ(nu_a, 41);
  EXPECT_EQ(nu_b, 61);
}

// A helper's failed count is made by the leader (same rules: the same count), the helper abandoned.
TEST(ParallelDriverHooks, AFailedHelperCountIsMadeByTheLeader) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  h.failing = 2;
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  ASSERT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b, 0.0));
  EXPECT_EQ(nu_a, 41);
  EXPECT_EQ(nu_b, f.below(b));
  EXPECT_EQ(nu_b, 61);
  EXPECT_EQ(f.rep.parallel.helper_counts, 1);
  EXPECT_EQ(f.rep.parallel.fallbacks, 1);
  EXPECT_FALSE(h.usable(2));
  EXPECT_TRUE(has_note(f.rep, "scripted failure"));
}

// The leader's side of the protocol: what the helpers hold once they have the problem is charged
// (above their idle level) until drop(), which they confirm; a later task sends the problem again.
TEST(ParallelSessionMemory, DropFreesWhatTheHelpersHold) {
  const auto a = laplacian3d<double>();
  hs::ShiftedPencil<double> pencil(a, nullptr);
  hs::FactorOptions fo;
  fo.threads = 1;
  auto fac = hs::make_mumps_factorization<double>(fo);
  fac->analyze(pencil, 1.0, -kSigma);
  hs::SolverConfig cfg = nearest_cfg();
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx c{cfg, nullptr, rep, clock};
  hs::driver::ProblemInfo info;
  info.sigma0 = kSigma;
  info.prepare_count = true;
  with_session(3, [&](hs::Transport& peers) {
    hs::parallel::RemoteHelpers<double> hp(peers, 7u, kForce);
    c.helpers = &hp;
    hp.share(c, pencil, *fac, info);
    EXPECT_EQ(hp.memory_bytes(), 0.0);  // nothing sent yet
    hp.prepare(c);
    EXPECT_GT(hp.count_estimate_mb(), 0.0);
    EXPECT_GE(hp.memory_bytes(), 0.0);
    hp.drop(c, "test");
    EXPECT_EQ(hp.memory_bytes(), 0.0);  // confirmed: back at the idle level
    hp.drop(c, "again");                 // nothing held: no message
    hp.prepare(c);                       // sent again
    EXPECT_GT(hp.count_estimate_mb(), 0.0);
    hs::driver::CountTask t;
    t.s = 0.5 * (spectrum()[3] + spectrum()[4]);
    t.cap_mb = 0.0;
    t.budget_gb = 1.0;
    hp.post_count(c, 1, t);
    const auto r = hp.wait_count(c, 1);
    EXPECT_TRUE(r.error.ok) << r.error.what;
    EXPECT_EQ(r.nu, 4);
    hp.drop(c, "done");
    EXPECT_EQ(hp.memory_bytes(), 0.0);
    c.helpers = nullptr;
  });
}

// Helpers are fed only what the solve path uses: GHIEP and nearest targets without a certificate
// share nothing.
TEST(ParallelSessionScope, NothingSharedWithoutHelperWork) {
  const auto a = laplacian3d<double>();
  hs::SolverConfig cfg = nearest_cfg();
  cfg.certificate = hs::CertificateMode::Off;
  with_session(3, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, cfg, nullptr, peers, kForce);
    EXPECT_TRUE(sol.report.success);
    EXPECT_EQ(sol.report.parallel.share_s, 0.0);
    EXPECT_EQ(sol.report.parallel.helper_counts, 0);
  });
}

// With the default options (Auto) the cost rules keep this small problem's counts and slices on the
// leader; Off does the same unconditionally. Results are the single-process ones either way.
TEST(ParallelSessionScope, AutoAndOffKeepSmallWorkOnTheLeader) {
  const auto a = laplacian3d<double>();
  with_session(3, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers);
    expect_nearest_certified(sol);
    EXPECT_EQ(sol.report.parallel.helper_counts, 0);
    EXPECT_TRUE(has_note(sol.report, "too cheap for the helpers")) << warnings(sol.report);
    EXPECT_EQ(sol.report.parallel.share_s, 0.0);  // nothing was sent
    const auto iv = hs::solve_parallel(a, nullptr, interval_cfg(), nullptr, peers);
    expect_interval_certified(iv);  // the slicing cost model decides; when it keeps them, it says so
    if (iv.report.parallel.helper_slices == 0) {
      EXPECT_TRUE(has_note(iv.report, "too small for the helpers")) << warnings(iv.report);
    }
    const hs::ParallelOptions off{hs::Toggle::Off, hs::Toggle::Off};
    const auto o = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers, off);
    expect_nearest_certified(o);
    EXPECT_EQ(o.report.parallel.helper_counts, 0);
    const auto oi = hs::solve_parallel(a, nullptr, interval_cfg(), nullptr, peers, off);
    expect_interval_certified(oi);
    EXPECT_EQ(oi.report.parallel.helper_slices, 0);
  });
}

TEST(ParallelSessionScope, SolveParallelNeedsRankZero) {
  auto t = hs::make_thread_transports(2);
  const auto a = laplacian3d<double>();
  EXPECT_THROW(hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, *t[1]), hs::Error);
}

}  // namespace
