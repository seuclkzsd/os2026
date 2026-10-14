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
#include "parallel/protocol.hpp"
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
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers);
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
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers);
    expect_nearest_certified(sol);
    const auto& p = sol.report.parallel;
    EXPECT_EQ(p.processes, 3);
    EXPECT_EQ(p.helper_counts, 2) << warnings(sol.report);
    EXPECT_EQ(p.fallbacks, 0);
    EXPECT_GT(p.share_s, 0.0);
    EXPECT_GT(p.helper_busy_s, 0.0);
    EXPECT_EQ(sol.report.certificate.count_inertia, ref.report.certificate.count_inertia);
    EXPECT_NE(sol.report.certificate.statement.find("2 of the 2 counts made by helper processes"), std::string::npos)
        << sol.report.certificate.statement;
    expect_same_set(sol.eigenvalues, ref.eigenvalues, 1e-12);
    // A second solve in the same session (the helpers dropped the first problem).
    const auto again = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers);
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
    const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, peers);
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
      const auto sol = hs::solve_parallel(a, nullptr, interval_cfg(), nullptr, peers);
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
    const auto sol = hs::solve_parallel(a, nullptr, cfg, nullptr, peers);
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
      const auto sol = hs::solve_parallel(a, nullptr, nearest_cfg(d::kFaultHelperTasks | kind), nullptr, peers);
      expect_nearest_certified(sol);
      const auto& p = sol.report.parallel;
      EXPECT_EQ(p.helper_counts, 0);
      EXPECT_EQ(p.fallbacks, 2) << warnings(sol.report);
      EXPECT_TRUE(has_note(sol.report, "could not count")) << warnings(sol.report);
      EXPECT_TRUE(has_note(sol.report, kind == d::kFaultBadAlloc ? "std::bad_alloc" : "fault injection"))
          << warnings(sol.report);
      const auto iv = hs::solve_parallel(a, nullptr, interval_cfg(d::kFaultHelperTasks | kind), nullptr, peers);
      expect_interval_certified(iv);
      EXPECT_EQ(iv.report.parallel.helper_slices, 0);
      EXPECT_EQ(iv.report.parallel.fallbacks, 2) << warnings(iv.report);  // the first slice of each helper
      EXPECT_TRUE(has_note(iv.report, "could not solve slice")) << warnings(iv.report);
    });
  }
}

// ------------------------------------------------------------------------------- driver hooks

// The driver's rule for the certificate's counts (driver::concurrent_counts) against helpers whose
// replies the test scripts: who counts, the memory rule, a failing helper.
class FakeHelpers final : public hs::driver::HelpersOf<double> {
 public:
  FakeHelpers(int n, std::vector<double> spectrum) : usable_(static_cast<std::size_t>(n) + 1, true), w_(std::move(spectrum)) {}
  int size() const override { return static_cast<int>(usable_.size()) - 1; }
  bool usable(int h) const override { return usable_.at(static_cast<std::size_t>(h)); }
  void abandon(int h, const std::string&) override { usable_.at(static_cast<std::size_t>(h)) = false; }
  bool shared() const override { return true; }
  void prepare(hs::driver::Ctx&) override { ++prepared; }
  double memory_bytes() const override { return memory; }
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
  void post_slice(hs::driver::Ctx&, int, const hs::driver::SliceTask&) override { FAIL() << "no slices here"; }
  int wait_slice(hs::driver::Ctx&, hs::driver::SliceReply<double>&) override { return 0; }
  int busy() const override { return 0; }

  double memory = 0.0, estimate_mb = 1.0;
  int failing = 0, prepared = 0;
  std::vector<std::pair<int, hs::driver::CountTask>> posted;

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
  ASSERT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  EXPECT_EQ(nu_a, 41);
  EXPECT_EQ(nu_b, 61);
  ASSERT_EQ(h.posted.size(), 2u);
  EXPECT_EQ(h.posted[0].second.outward, -1.0);
  EXPECT_EQ(h.posted[1].second.outward, +1.0);
  EXPECT_TRUE(h.posted[0].second.s_is_gap);
  EXPECT_EQ(f.rep.parallel.helper_counts, 2);
  EXPECT_EQ(f.rep.parallel.fallbacks, 0);
  EXPECT_EQ(h.prepared, 1);              // the problem goes to the helpers at their first task
  EXPECT_FALSE(f.fac->inertia().valid);  // the main factors were released before the counts
}

TEST(ParallelDriverHooks, FewerThanTwoHelpersOrNoEstimateCountOnTheLeader) {
  HookFixture f;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  FakeHelpers one(1, f.w);
  f.c.helpers = &one;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  FakeHelpers two(2, f.w);
  two.abandon(1, "test");
  f.c.helpers = &two;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  FakeHelpers none(2, f.w);
  none.estimate_mb = 0.0;
  f.c.helpers = &none;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  EXPECT_TRUE(one.posted.empty() && two.posted.empty() && none.posted.empty());
  EXPECT_TRUE(has_note(f.rep, "fewer than two helpers"));
  EXPECT_TRUE(has_note(f.rep, "could not prepare"));
}

// The helpers' memory counts against the solve's budget: two counts that do not fit beside it stay
// with the leader, which counts one at a time as hs::solve does.
TEST(ParallelDriverHooks, TwoCountsBeyondTheBudgetStayOnTheLeader) {
  HookFixture f;
  FakeHelpers h(2, f.w);
  h.memory = 3.9 * 1073741824.0;  // what the helpers hold of the 4 GiB budget
  h.estimate_mb = 200.0;
  f.c.helpers = &h;
  hs::driver::CountSource<double> counts(*f.fac, f.pencil, true, hs::driver::MainFactors::Release);
  double a = f.gap(40), b = f.gap(60);
  index_t nu_a = -1, nu_b = -1;
  EXPECT_FALSE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  EXPECT_TRUE(h.posted.empty());
  EXPECT_TRUE(has_note(f.rep, "two counts at once")) << warnings(f.rep);
  h.memory = 0.0;
  EXPECT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
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
  ASSERT_TRUE(hs::driver::concurrent_counts(f.c, counts, a, b, 1e-9, nu_a, nu_b));
  EXPECT_EQ(nu_a, 41);
  EXPECT_EQ(nu_b, f.below(b));
  EXPECT_EQ(nu_b, 61);
  EXPECT_EQ(f.rep.parallel.helper_counts, 1);
  EXPECT_EQ(f.rep.parallel.fallbacks, 1);
  EXPECT_FALSE(h.usable(2));
  EXPECT_TRUE(has_note(f.rep, "scripted failure"));
}

// Helpers are fed only what the solve path uses: GHIEP and nearest targets without a certificate
// share nothing.
TEST(ParallelSessionScope, NothingSharedWithoutHelperWork) {
  const auto a = laplacian3d<double>();
  hs::SolverConfig cfg = nearest_cfg();
  cfg.certificate = hs::CertificateMode::Off;
  with_session(3, [&](hs::Transport& peers) {
    const auto sol = hs::solve_parallel(a, nullptr, cfg, nullptr, peers);
    EXPECT_TRUE(sol.report.success);
    EXPECT_EQ(sol.report.parallel.share_s, 0.0);
    EXPECT_EQ(sol.report.parallel.helper_counts, 0);
  });
}

TEST(ParallelSessionScope, SolveParallelNeedsRankZero) {
  auto t = hs::make_thread_transports(2);
  const auto a = laplacian3d<double>();
  EXPECT_THROW(hs::solve_parallel(a, nullptr, nearest_cfg(), nullptr, *t[1]), hs::Error);
}

}  // namespace
