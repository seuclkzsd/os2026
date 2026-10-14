// The driver's side of multi-process solves (DESIGN §3.9, driver/helpers.hpp): the nearest
// certificate's two counts on two helper processes at once, and the helper's side of a count. The
// slice tasks are in definite.cpp (WindowSolver).
#include <algorithm>
#include <cmath>
#include <new>
#include <string>

#include "driver/helpers.hpp"
#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/util.hpp"

namespace hs::driver {

namespace {

void note(Ctx& c, const std::string& what) {
  auto& notes = c.rep.parallel.notes;
  if (std::find(notes.begin(), notes.end(), what) == notes.end()) notes.push_back(what);
}

}  // namespace

// The two counts of a nearest certificate are independent inertia-only factorizations on one pivot
// order (P8). In a multi-process solve the helpers - one per socket, each with its socket's cores
// and memory bandwidth - count nu(a) and nu(b) at the same time while the leader, whose threads
// span all sockets, waits (it never computes while helpers do: no core is oversubscribed;
// results/p10_mpi.md). Both use count_below's rules, so the counts - exact integers - are those of
// the sequential path. The leader's main factors are released first, as before counter counts
// (MainFactors::Release; a rerun refactorizes at sigma), and the two counts at once must fit what
// the budget then leaves for counts (count_allowance_mb, charged with the helpers' memory).
template <class T>
bool concurrent_counts(Ctx& c, CountSource<T>& counts, double& a, double& b, double step, index_t& nu_a,
                       index_t& nu_b) {
  if (!c.helpers || !c.helpers->shared()) return false;
  Helpers& hp = *c.helpers;
  int h[2] = {0, 0};
  for (int i = 1, k = 0; i <= hp.size() && k < 2; ++i)
    if (hp.usable(i)) h[k++] = i;
  if (h[1] == 0) {
    note(c, "certificate counts on the leader: fewer than two helpers usable");
    return false;
  }
  SparseFactorization<T>& fac = counts.fac();
  {
    PhaseScope phase(c.clock, Phase::Certificate);
    hp.prepare(c);  // the helpers get the problem now and analyse their counters (the leader waits)
  }
  const double est = hp.count_estimate_mb();
  if (!(est > 0) || !hp.usable(h[0]) || !hp.usable(h[1])) {
    note(c, "certificate counts on the leader: the helpers could not prepare their counters");
    return false;
  }
  if (fac.inertia().valid) {
    const double main_mb = fac.stats().factor_mb_used;
    if (fac.release_factors()) {
      event(c, "main_factors_released", {{"main_mb", main_mb}, {"counter_mb", est},
                                         {"cap_mb", fac.options().max_memory_mb}, {"over_cap", false},
                                         {"why", std::string("helper counts")}});
      c.held = FactorHeld{};
    }
  }
  const double cap = fac.options().max_memory_mb;
  const double allowance = count_allowance_mb(c, 0.0);
  if (2.0 * est > allowance) {
    note(c, "certificate counts on the leader: two counts at once (" + fmt(2.0 * est) + " MB) exceed the " +
                fmt(allowance) + " MB the budget leaves");
    event(c, "helper_count_skipped", {{"need_mb", 2.0 * est}, {"allowance_mb", allowance}});
    return false;
  }
  CountTask task[2];
  double* at[2] = {&a, &b};
  for (int k = 0; k < 2; ++k) {
    task[k].s = *at[k];
    task[k].outward = k == 0 ? -1.0 : +1.0;
    task[k].step = step;
    task[k].s_is_gap = true;
    task[k].cap_mb = cap > 0 ? std::min(cap, 0.5 * allowance) : 0.5 * allowance;
    task[k].budget_gb = 0.5 * allowance / 1073.741824;
  }
  hp.post_count(c, h[0], task[0]);
  hp.post_count(c, h[1], task[1]);
  CountReply reply[2];
  {
    PhaseScope phase(c.clock, Phase::Certificate);
    reply[0] = hp.wait_count(c, h[0]);
    reply[1] = hp.wait_count(c, h[1]);
  }
  index_t* nu[2] = {&nu_a, &nu_b};
  for (int k = 0; k < 2; ++k) {
    if (reply[k].error.ok) {
      *nu[k] = reply[k].nu;
      *at[k] = reply[k].s;
      c.rep.certificate.endpoint_moves += reply[k].endpoint_moves;
      if (reply[k].detection_off && c.rep.certificate.note.empty())
        c.rep.certificate.note = "endpoint inertia counted without null-pivot detection (dense cluster)";
      ++c.rep.parallel.helper_counts;
      event(c, "helper_count",
            {{"helper", static_cast<std::int64_t>(h[k])}, {"s", *at[k]}, {"nu", static_cast<std::int64_t>(*nu[k])},
             {"seconds", reply[k].seconds}, {"count_mb", reply[k].count_mb},
             {"helper_rss_mb", reply[k].rss_bytes / 1e6}});
      continue;
    }
    // The helper could not count (memory, a MUMPS error, an exception): the leader counts as the
    // sequential path does (after both replies: no oversubscribed cores), and does not use that
    // helper again in this solve.
    const std::string why = std::string(to_string(reply[k].error.code)) + ": " + reply[k].error.what;
    hp.abandon(h[k], "count failed: " + why);
    ++c.rep.parallel.fallbacks;
    warn(c, "multi-process solve: helper " + std::to_string(h[k]) + " could not count (" + why +
                "); the leader counts instead");
    *nu[k] = count_below(c, counts, *at[k], task[k].outward, step, nullptr, /*s_is_gap=*/true);
  }
  return true;
}

template <class T>
CountReply count_on_helper(Ctx& c, InertiaCounter<T>& counter, const CountTask& t) {
  Timer timer;
  CountReply r;
  r.s = t.s;
  if (fault(c.cfg, kFaultHelperTasks)) throw_injected_fault(c.cfg, kFaultHelperTasks, "count on a helper process");
  const index_t moves = c.rep.certificate.endpoint_moves;
  bool off = false;
  r.nu = count_below(
      c,
      [&](double s, bool detection) {
        PhaseScope phase(c.clock, Phase::Certificate);
        return counter.count(1.0, -s, detection);
      },
      r.s, t.outward, t.step, &off, t.s_is_gap);
  r.detection_off = off;
  r.endpoint_moves = c.rep.certificate.endpoint_moves - moves;
  r.count_mb = counter.stats().max_mb_used;
  r.seconds = timer.seconds();
  r.rss_bytes = static_cast<double>(current_rss_bytes());
  return r;
}

#define HS_INSTANTIATE_HELPERS(T)                                                                         \
  template bool concurrent_counts<T>(Ctx&, CountSource<T>&, double&, double&, double, index_t&, index_t&); \
  template CountReply count_on_helper<T>(Ctx&, InertiaCounter<T>&, const CountTask&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_HELPERS)
#undef HS_INSTANTIATE_HELPERS

}  // namespace hs::driver
