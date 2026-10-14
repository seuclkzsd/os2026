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
double helper_preparation_s(const MumpsInputSize& in, const FactorEstimate& e, int analyses) {
  return analyses * model_analysis_s(in.order, in.entries, static_cast<double>(e.factor_entries)) + kHelperMessageS;
}

namespace {
template <class T>
double csr_bytes(const CsrMatrix<T>& m) {
  return static_cast<double>(m.rowptr.size() * sizeof(m.rowptr[0]) + m.colind.size() * sizeof(m.colind[0]) +
                             m.values.size() * sizeof(T));
}
}  // namespace

template <class T>
double helper_problem_bytes(const ShiftedPencil<T>& pencil) {
  return csr_bytes(pencil.a()) + (pencil.b() ? csr_bytes(*pencil.b()) : 0.0);
}

template <class T>
double helper_footprint_bytes(const ShiftedPencil<T>& pencil) {
  return kHelperProblemCopies * helper_problem_bytes(pencil);
}

double helper_count_estimate_mb(const FactorEstimate& main) { return kHelperCountShare * main.incore_mb; }

int slice_helpers_fitting(double avail, double need, int usable) {
  const int fit = need > 0 ? static_cast<int>(std::min(std::max(avail, 0.0) / need, 1e6)) : usable;
  const int take = std::min(usable, fit);
  return take >= 2 ? take : 0;
}

// Order of the decisions (DESIGN §3.9): the cost rule; the leader's main factors released (as its
// own counter counts would); the memory rule PREDICTED - two counts at once plus the helpers' copies
// of the problem must fit what the budget leaves - before anything is sent; the helpers get the
// problem (prepare) and report their count estimate and memory; the memory rule again with those.
// Whenever the counts stay with the leader after the helpers got the problem, and after their two
// counts, the helpers drop it (Helpers::drop): the leader's own counts (all of them, or the one a
// failed helper could not make) then see the budget hs::solve would see.
template <class T>
bool concurrent_counts(Ctx& c, CountSource<T>& counts, double& a, double& b, double step, index_t& nu_a,
                       index_t& nu_b, double factorize_s) {
  if (!c.helpers || !c.helpers->shared() || c.helpers->options().helper_counts == Toggle::Off) return false;
  Helpers& hp = *c.helpers;
  int h[2] = {0, 0};
  for (int i = 1, k = 0; i <= hp.size() && k < 2; ++i)
    if (hp.usable(i)) h[k++] = i;
  if (h[1] == 0) {
    note(c, "certificate counts on the leader: fewer than two helpers usable");
    return false;
  }
  SparseFactorization<T>& fac = counts.fac();
  // Worth it? The leader's pair takes ~2 counts of kCounterCountShare x its main factorization;
  // on the helpers both at once take one count at kHelperCountSlowdown of the leader's speed after
  // their preparation (helper_preparation_s): they pay when (2 - kHelperCountSlowdown) counts
  // outweigh it (cav2d-rect-pert-1e5: a 0.035 s factorization against ~0.18 s of preparation).
  const double prep = helper_preparation_s(mumps_input_size(counts.pencil()), fac.estimate(), 2);
  if (hp.options().helper_counts == Toggle::Auto && factorize_s > 0 &&
      (2.0 - kHelperCountSlowdown) * kCounterCountShare * factorize_s < prep) {
    note(c, "certificate counts on the leader: too cheap for the helpers (main factorization " + fmt(factorize_s) +
                " s, their preparation ~" + fmt(prep) + " s)");
    return false;
  }
  if (fac.inertia().valid) {
    const double main_mb = fac.stats().factor_mb_used;
    if (fac.release_factors()) {
      event(c, "main_factors_released",
            {{"main_mb", main_mb}, {"counter_mb", helper_count_estimate_mb(fac.estimate())},
             {"cap_mb", fac.options().max_memory_mb}, {"over_cap", false}, {"why", std::string("helper counts")}});
      c.held = FactorHeld{};
    }
  }
  // Predicted before anything is sent: two counts at once and what the helpers will hold.
  {
    const double need =
        2.0 * (helper_count_estimate_mb(fac.estimate()) + helper_footprint_bytes(counts.pencil()) / 1e6);
    const double allowance = count_allowance_mb(c, 0.0);
    if (need > allowance) {
      note(c, "certificate counts on the leader: two counts at once and the helpers' copies of the problem (~" +
                  fmt(need) + " MB predicted) exceed the " + fmt(allowance) + " MB the budget leaves");
      event(c, "helper_count_skipped", {{"need_mb", need}, {"allowance_mb", allowance}, {"predicted", true}});
      return false;
    }
  }
  {
    PhaseScope phase(c.clock, Phase::Certificate);
    hp.prepare(c);  // the helpers get the problem now and analyse their counters (the leader waits)
  }
  const double est = hp.count_estimate_mb();
  if (!(est > 0) || !hp.usable(h[0]) || !hp.usable(h[1])) {
    note(c, "certificate counts on the leader: the helpers could not prepare their counters");
    PhaseScope phase(c.clock, Phase::Certificate);
    hp.drop(c, "counts on the leader: helpers not prepared");
    return false;
  }
  const double cap = fac.options().max_memory_mb;
  const double allowance = count_allowance_mb(c, 0.0);  // charged with what the helpers now hold
  if (2.0 * est > allowance) {
    note(c, "certificate counts on the leader: two counts at once (" + fmt(2.0 * est) + " MB) exceed the " +
                fmt(allowance) + " MB the budget leaves");
    event(c, "helper_count_skipped", {{"need_mb", 2.0 * est}, {"allowance_mb", allowance}, {"predicted", false}});
    PhaseScope phase(c.clock, Phase::Certificate);
    hp.drop(c, "counts on the leader: memory");
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
    hp.drop(c, "helper counts done");  // before a count of the leader's below, and before a rerun
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
    // sequential path does (after both replies and the helpers' drop: no oversubscribed cores, the
    // whole budget), and does not use that helper again in this solve.
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
  template double helper_problem_bytes<T>(const ShiftedPencil<T>&);                                       \
  template double helper_footprint_bytes<T>(const ShiftedPencil<T>&);                                     \
  template bool concurrent_counts<T>(Ctx&, CountSource<T>&, double&, double&, double, index_t&, index_t&,    \
                                    double);                                                              \
  template CountReply count_on_helper<T>(Ctx&, InertiaCounter<T>&, const CountTask&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_HELPERS)
#undef HS_INSTANTIATE_HELPERS

}  // namespace hs::driver
