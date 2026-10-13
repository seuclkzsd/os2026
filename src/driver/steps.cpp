// Driver building blocks shared by the solve paths: context helpers, pair blocks and the
// factorization steps (shift retries, inertia counts, analysis and memory budget, adaptive
// ordering). See internal.hpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "driver/internal.hpp"
#include "driver/projection.hpp"
#include "eigen/block_lanczos.hpp"
#include "factor/ordering_detail.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/util.hpp"

namespace hs::driver {

Phase PhaseClock::switch_to(Phase p) {
  const double now = timer_.seconds();
  acc_[static_cast<int>(current_)] += now - last_;
  last_ = now;
  const Phase prev = current_;
  current_ = p;
  return prev;
}

double PhaseClock::seconds(Phase p) const {
  double s = acc_[static_cast<int>(p)];
  if (p == current_) s += timer_.seconds() - last_;
  return s;
}

double PhaseClock::total() const {
  double t = 0.0;
  for (int p = 0; p < kPhases; ++p) t += seconds(static_cast<Phase>(p));
  return t;
}

void warn(Ctx& c, const std::string& msg) {
  c.rep.warnings.push_back(msg);
  if (c.log) c.log->warn(msg);
}

void event(Ctx& c, const std::string& name, std::initializer_list<LogField> fields) {
  if (c.log) c.log->event(name, fields);
}

std::string fmt(double x) {
  std::ostringstream os;
  os.precision(6);
  os << x;
  return os.str();
}

template <class T>
void recompute_products(const CsrMatrix<T>& a, const CsrMatrix<T>* b, Block<T>& blk) {
  driver::apply_block(&a, blk.x, blk.ax);
  driver::apply_block(b, blk.x, blk.bx);
}

template <class T>
Block<T> take_columns(const Block<T>& src, const std::vector<index_t>& cols) {
  Block<T> out;
  const index_t n = src.x.rows(), m = static_cast<index_t>(cols.size());
  out.x.resize(n, m);
  out.ax.resize(n, m);
  out.bx.resize(n, m);
  for (index_t j = 0; j < m; ++j) out.lambda.push_back(src.lambda[cols[j]]);
#pragma omp parallel for schedule(static)
  for (index_t j = 0; j < m; ++j) {
    const index_t c = cols[j];
    std::copy(src.x.col(c), src.x.col(c) + n, out.x.col(j));
    std::copy(src.ax.col(c), src.ax.col(c) + n, out.ax.col(j));
    std::copy(src.bx.col(c), src.bx.col(c) + n, out.bx.col(j));
  }
  return out;
}

template <class T>
void append_columns(DenseMatrix<T>& dst, const DenseMatrix<T>& src) {
  const index_t n = dst.rows() ? dst.rows() : src.rows(), c0 = dst.cols();
  if (dst.rows() == 0) dst.resize(n, 0);
  dst.resize_cols(c0 + src.cols());
  std::copy(src.data(), src.data() + n * src.cols(), dst.col(c0));
}

FactorOptions factor_options_from(const SolverConfig& cfg, double cap_mb) {
  FactorOptions o;
  o.precision = cfg.factor_precision;
  o.ordering = cfg.ordering < 0 ? 7 : cfg.ordering;  // adaptive (-1) is resolved by adaptive_analysis
  o.deterministic = cfg.deterministic;
  o.threads = cfg.threads > 0 ? cfg.threads : max_threads();
  o.blr = cfg.blr == Toggle::On;
  o.out_of_core = cfg.out_of_core == Toggle::On;
  o.max_memory_mb = cap_mb > 0 ? cap_mb : 0.0;
  return o;
}

// ------------------------------------------------------------------------------ run-time models

// Factorization / solve model of DESIGN §3.6-3b (calibrated on the 2x Xeon 6336Y node).
double model_factor_s(const FactorEstimate& e, double order, int threads) {
  return 6e-7 * order + e.flops / (12e9 * std::max(1, threads));
}
double model_solve_s(const FactorEstimate& e) { return 16.0 * static_cast<double>(e.factor_entries) / 12e9; }

namespace {
// AMD analysis model, least squares on the relative error over the 42 data cases (n = 289 ..
// 584016 MUMPS order, 1.6e3 .. 2.6e7 input entries, 5.7e3 .. 5.4e9 AMD factor entries; median of 3
// analyses each, bench/drvint_profile --exp analysis, 1 thread of MUMPS analysis on the 6336Y):
//   t = 3.4e-7 s * order + 2.5e-8 s * entries + 3.6e-10 s * factor_entries
// predicts every case with n >= 1e4 within 0.41x .. 1.44x (median 0.95; order and entries alone:
// 0.18x .. 1.50x, median 0.84). The third term is the symbolic factorization: 3-D cases with
// AMD fill of 1e8..1e9 entries take 2..8x longer than 2-D ones with the same input size. The
// measured wall time it replaces varied 3x between runs of the same input (0.086 vs 0.258 s,
// syn-ghep-near-singular-real-1e5, first MUMPS call of the process), which made the adaptive
// ordering run-dependent (wg-square-1e6: saving / SCOTCH cost = 0.78 measured, 1.02 modelled).
constexpr double kAnalysisPerOrder = 3.4e-7;
constexpr double kAnalysisPerEntry = 2.5e-8;
constexpr double kAnalysisPerFactorEntry = 3.6e-10;
}  // namespace

double model_analysis_s(double order, double entries, double factor_entries) {
  return kAnalysisPerOrder * order + kAnalysisPerEntry * entries + kAnalysisPerFactorEntry * factor_entries;
}

template <class T>
MumpsInputSize mumps_input_size(const ShiftedPencil<T>& pencil) {
  const double n = static_cast<double>(pencil.order()), nnz = static_cast<double>(pencil.nnz());
  // Complex T: the real embedding of order 2n holds up to 4 real entries per off-diagonal complex
  // one (3 per diagonal entry) in its upper triangle.
  if constexpr (is_complex_v<T>) return {2.0 * n, 4.0 * nnz - n};
  return {n, nnz};
}

// ------------------------------------------------------------------------- factorization steps

index_t near_null_eigenvalues(index_t null_pivots, bool complex_scalars) {
  return complex_scalars ? (null_pivots + 1) / 2 : null_pivots;
}

// Factorize K(sigma0) (DESIGN §3.6-4). A null pivot means sigma0 is (numerically) ON an eigenvalue:
// perturb the shift once, by delta = 1e-8 * max(scale, |sigma0|) - an isolated eigenvalue at
// sigma0 is then ~1e3 times beyond the detection threshold (factor.hpp). Inside a cluster denser
// than the null-pivot threshold, moving away (a second, 10x larger perturbation used to be tried)
// changes which eigenvalues are nearest - on syn-zero-cluster-real-1e5 it landed outside the
// cluster at 5x the wanted radius and the certificate then counted 166 eigenvalues for 21 found -
// so there K is factorized AT sigma0 with null-pivot detection off (tiny pivots are kept;
// shift-invert then amplifies exactly the wanted near-null directions). `detection_off` tells
// later refactorizations at this shift to do the same.
// Which case it is: more than one eigenvalue's worth of null pivots at sigma0 (near_null_eigenvalues:
// the complex embedding doubles them) means several eigenvalues within the threshold - the
// perturbation, of the threshold's size, would reorder them - and the perturbation is skipped; a
// perturbed shift that still looks singular means the same. The null-pivot count decides first
// because whether the perturbed shift is flagged depends on the pivot order: on
// syn-zero-cluster-real-1e5 (21 wanted of 10^4 eigenvalues spaced 2e-9 around sigma0 = 0) AMD's
// factors flagged it, METIS', SCOTCH's and hsolve's nested dissection's did not, and their runs at
// sigma0 + 1e-8 found 21 pairs out to 2.85e-8 while 34 lie that close to 0 (the 21 nearest reach
// 1.86e-8): a certificate rerun on every seed, 10..11 factorizations instead of 6 (P5c fix).
// Exactly zero pivots at sigma0 with detection off (factorize() false): the perturbed shift after
// all.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off,
                      bool on_eigenvalue) {
  PhaseScope phase(c.clock, Phase::Factor);
  detection_off = false;
  if (c.held.shift == sigma0 && !c.held.detection_off) return sigma0;  // e.g. right after counting there
  c.held = FactorHeld{};
  fac.set_null_pivot_detection(true);
  if (fac.factorize(1.0, -sigma0)) {
    c.held = FactorHeld{sigma0, false};
    return sigma0;
  }
  const index_t nulls = fac.stats().null_pivots;
  const bool cluster = !on_eigenvalue && near_null_eigenvalues(nulls, is_complex_v<T>) >= 2;
  const bool may_perturb = c.cfg.max_shift_retries > 0;
  const double perturbed = sigma0 + 1e-8 * std::max(scale, std::abs(sigma0));
  event(c, "singular_shift", {{"sigma", sigma0}, {"null_pivots", static_cast<std::int64_t>(nulls)},
                              {"cluster", cluster}, {"on_eigenvalue", on_eigenvalue}});
  auto at_perturbed = [&]() {
    if (!fac.factorize(1.0, -perturbed)) return false;
    if (!on_eigenvalue) {
      c.rep.shift_retries = std::max<index_t>(c.rep.shift_retries, 1);
      warn(c, "shift perturbed to " + fmt(perturbed) + " (A - sigma B singular at the requested shift)");
    }
    c.held = FactorHeld{perturbed, false};
    return true;
  };
  if (!cluster && may_perturb && at_perturbed()) return perturbed;
  fac.set_null_pivot_detection(false);
  if (fac.factorize(1.0, -sigma0)) {
    detection_off = true;
    if (!on_eigenvalue) {
      c.rep.shift_retries = std::max<index_t>(c.rep.shift_retries, may_perturb ? 2 : 1);
      warn(c, "shift " + fmt(sigma0) + " lies inside an eigenvalue cluster denser than the null-pivot threshold; "
              "factorized there without null-pivot detection");
    }
    c.held = FactorHeld{sigma0, true};
    return sigma0;
  }
  fac.set_null_pivot_detection(true);
  if (cluster && may_perturb && at_perturbed()) return perturbed;
  HS_THROW(Singular, "A - sigma B is exactly singular at and near sigma = " << sigma0);
}

// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
template <class T>
bool try_refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off) {
  if (c.held.shift == sigma && c.held.detection_off == detection_off) return true;
  PhaseScope phase(c.clock, Phase::Factor);
  c.held = FactorHeld{};
  fac.set_null_pivot_detection(!detection_off);
  const bool ok = fac.factorize(1.0, -sigma);
  fac.set_null_pivot_detection(true);
  if (ok) c.held = FactorHeld{sigma, detection_off};
  return ok;
}

template <class T>
void refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off) {
  if (!try_refactorize_at(c, fac, sigma, detection_off))
    HS_THROW(Internal, "refactorization at the working shift " << sigma << " failed");
}

// ------------------------------------------------------------------------------ inertia counts

double count_allowance_mb(const Ctx& c, double cap_mb) {
  const double free_bytes = c.cfg.memory_budget_gb * 1073741824.0 -
                            static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
  double mb = kCountMemoryShare * std::max(0.0, free_bytes) / 1e6;
  if (cap_mb > 0) mb = std::min(mb, cap_mb);
  return mb;
}

double counter_allowance_mb(const Ctx& c, double cap_mb) {
  return fault(c.cfg, kFaultCountAllowance) ? 0.0 : count_allowance_mb(c, cap_mb);
}

void throw_injected_fault(const SolverConfig& cfg, index_t bit, const std::string& what) {
  if (fault(cfg, kFaultBadAlloc)) throw std::bad_alloc();
  const std::string msg = "fault injection (certificate_fault_count & " + std::to_string(bit) + "): " + what;
  if (fault(cfg, kFaultOtherError)) HS_THROW(FactorizationFailed, msg);
  HS_THROW(OutOfMemoryBudget, msg);
}

template <class T>
FactorOptions counter_options(const Ctx& c, const SparseFactorization<T>& fac, bool faults) {
  FactorOptions o = fac.options();
  // MUMPS INFO -19 at every count
  if (faults && fault(c.cfg, kFaultCounterCounts) && !counter_fault_throws(c.cfg)) o.max_memory_mb = 1.0;
  return o;
}

template <class T>
CountSource<T>::CountSource(SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, bool prefer_counter,
                            MainFactors main)
    : fac_(fac), pencil_(pencil), main_(main) {
  main_exact_ = fac.options().precision == FactorPrecision::Double && !fac.options().blr;
  use_counter_ = prefer_counter || !main_exact_;
  last_on_counter_ = use_counter_;
}

template <class T>
CountSource<T>::~CountSource() = default;

// The preferred source first; a source that failed is not tried again in this solve (whatever
// stopped it - the memory its working set needs, a MUMPS error - would stop it again: an
// InertiaCounter's analysis on a given pivot order is deterministic, and so is the main object's).
template <class T>
Inertia CountSource<T>::count(Ctx& c, double s, bool detection) {
  PhaseScope phase(c.clock, Phase::Certificate);
  for (;;) {
    const bool counter_ok = !counter_failed_;
    const bool main_ok = main_exact_ && !main_failed_;
    if (!counter_ok && !main_ok) break;
    const bool on_counter = use_counter_ ? counter_ok : !main_ok;
    try {
      const Inertia in = on_counter ? count_on_counter(c, s, detection) : count_on_main(c, s, detection);
      last_on_counter_ = on_counter;
      return in;
    } catch (const Error& e) {
      if (!count_failure(e)) throw;
      abandon(c, on_counter, std::string(to_string(e.code())) + ": " + e.what(),
              e.code() == Error::Code::OutOfMemoryBudget);
    } catch (const std::bad_alloc&) {
      abandon(c, on_counter, "allocation failure (std::bad_alloc)", true);
    }
  }
  // Out of memory only when every source that failed failed for memory (CountFailure comment).
  const bool memory = (!counter_failed_ || counter_memory_) && (!main_failed_ || main_memory_);
  throw CountFailure(memory ? Error::Code::OutOfMemoryBudget : Error::Code::FactorizationFailed,
                     "no inertia count possible at s = " + fmt(s) + " (" + failures() + ")");
}

template <class T>
Inertia CountSource<T>::count_on_main(Ctx& c, double s, bool detection) {
  if (injected(c, kFaultMainCounts)) throw_injected_fault(c.cfg, kFaultMainCounts, "count on the main factorization");
  if (use_counter_) {
    // Standing in for the preferred counter, which failed: memory is short. The main object's own
    // ICNTL(23) cap was fixed at the analysis, before the Krylov basis and the pairs, so it does
    // not keep this count within the budget (CountSource comment): its factors go first, then the
    // working memory of its last factorization (INFOG(21); before any, the analysis estimate) must
    // fit what the budget leaves.
    if (fac_.inertia().valid && fac_.release_factors()) c.held = FactorHeld{};
    const FactorEstimate est = fac_.estimate();
    const double need = fac_.stats().factor_mb_used > 0 ? fac_.stats().factor_mb_used
                                                        : (fac_.options().out_of_core ? est.ooc_mb : est.incore_mb);
    const double allowance = count_allowance_mb(c, fac_.options().max_memory_mb);
    event(c, "main_count_check", {{"s", s}, {"need_mb", need}, {"allowance_mb", allowance},
                                  {"rss_mb", static_cast<double>(current_rss_bytes()) / 1e6}});
    if (need > allowance)
      HS_THROW(OutOfMemoryBudget, "main factorization: the working memory of its last factorization, "
                                      << need << " MB, exceeds the " << allowance
                                      << " MB the memory budget leaves for a count");
  }
  c.held = FactorHeld{};
  fac_.set_null_pivot_detection(detection);
  Inertia in;
  try {
    if (fac_.factorize(1.0, -s)) in = fac_.inertia();
  } catch (...) {
    fac_.set_null_pivot_detection(true);
    throw;
  }
  fac_.set_null_pivot_detection(true);
  if (in.valid) c.held = FactorHeld{s, !detection};
  return in;
}

template <class T>
Inertia CountSource<T>::count_on_counter(Ctx& c, double s, bool detection) {
  const double cap = fac_.options().max_memory_mb;
  if (!counter_) {
    // Inertia-only instance on the main analysis' pivot order.
    counter_ = std::make_unique<InertiaCounter<T>>(fac_, counter_options(c, fac_, injected(c, kFaultCounterCounts)));
    event(c, "inertia_counter", {{"analyze_s", counter_->stats().analyze_s},
                                 {"incore_mb", counter_->estimate().incore_mb}, {"cap_mb", cap}});
  }
  const double counter_mb = counter_->estimate().incore_mb;
  // The counter's working memory comes on top of resident main factors. Decided before every
  // count: the main object may hold factors again since the last one (refactorized for a rerun,
  // a rescue, a slice shift).
  auto release_main = [&](bool over_cap, const char* why) {
    const double main_mb = fac_.stats().factor_mb_used;
    if (!(fac_.inertia().valid && fac_.release_factors())) return false;
    c.held = FactorHeld{};
    event(c, "main_factors_released", {{"main_mb", main_mb}, {"counter_mb", counter_mb}, {"cap_mb", cap},
                                       {"over_cap", over_cap}, {"why", std::string(why)}});
    return true;
  };
  if (fac_.inertia().valid) {
    const bool over_cap = cap > 0 && fac_.stats().factor_mb_used + counter_mb > cap;
    if (main_ == MainFactors::Release || over_cap) release_main(over_cap, over_cap ? "cap" : "policy");
  }
  // The counter must fit what the budget leaves now (CountSource comment); resident main factors
  // go first.
  double allowance = counter_allowance_mb(c, cap);
  if (counter_mb > allowance && release_main(false, "allowance")) allowance = counter_allowance_mb(c, cap);
  if (counter_mb > allowance)
    HS_THROW(OutOfMemoryBudget, "InertiaCounter: its in-core estimate " << counter_mb << " MB exceeds the "
                                                                        << allowance
                                                                        << " MB the memory budget leaves for a count");
  if (injected(c, kFaultCounterCounts) && counter_fault_throws(c.cfg))
    throw_injected_fault(c.cfg, kFaultCounterCounts, "InertiaCounter count");
  return counter_->count(1.0, -s, detection);
}

template <class T>
void CountSource<T>::abandon(Ctx& c, bool counter, const std::string& why, bool memory) {
  (counter ? counter_failed_ : main_failed_) = true;
  (counter ? counter_why_ : main_why_) = why;
  (counter ? counter_memory_ : main_memory_) = memory;
  if (counter) counter_.reset();  // its analysis and whatever workspace MUMPS kept
  const bool other = counter ? main_exact_ && !main_failed_ : !counter_failed_;
  event(c, "count_source",
        {{"failed", std::string(counter ? "inertia_counter" : "main_factorization")},
         {"next", std::string(!other ? "none" : counter ? "main_factorization" : "inertia_counter")},
         {"reason", why}, {"memory", memory}, {"rss_mb", static_cast<double>(current_rss_bytes()) / 1e6}});
  if (other)
    warn(c, std::string("certificate counts: the ") + (counter ? "InertiaCounter" : "main factorization") +
                " could not count (" + why + "); counting on the " +
                (counter ? "main factorization" : "InertiaCounter") + " instead");
}

template <class T>
void CountSource<T>::avoid_counter(Ctx& c, const std::string& why) {
  if (!counter_failed_) abandon(c, true, why, true);  // another counter on the same order failed for memory
}

template <class T>
std::string CountSource<T>::failures() const {
  std::string s;
  if (counter_failed_) s += "InertiaCounter: " + counter_why_;
  if (main_failed_) s += (s.empty() ? "" : "; ") + std::string("main factorization: ") + main_why_;
  if (!main_exact_) s += (s.empty() ? "" : "; ") + std::string("the main factors are inexact (single / BLR)");
  return s;
}

#define HS_INSTANTIATE_COUNTER_OPTIONS(T) \
  template FactorOptions counter_options<T>(const Ctx&, const SparseFactorization<T>&, bool);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_COUNTER_OPTIONS)
#undef HS_INSTANTIATE_COUNTER_OPTIONS

bool counter_pays(const SolverConfig& cfg, const FactorEstimate& e, const MumpsInputSize& in, int threads,
                  index_t counts) {
  if (cfg.inertia_counter != Toggle::Auto) return cfg.inertia_counter == Toggle::On;
  const double saving = static_cast<double>(counts) * kCounterSaving * model_factor_s(e, in.order, threads);
  return saving > model_analysis_s(in.order, in.entries, static_cast<double>(e.factor_entries));
}

// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2). The count must come from an exact
// (double, full-rank) factorization: the inertia of a single-precision or BLR factor is that of the
// rounded/compressed matrix, so then the counts come from an InertiaCounter (CountSource). Each
// position is tried with null-pivot detection first (certified). A null pivot means an eigenvalue
// at s, or a cluster denser than the detection threshold around s; then the count is taken
// WITHOUT detection: a backward-stable LDL^T gives the exact inertia of K + E with
// ||E|| ~ eps*growth*||K||, i.e. it counts correctly every eigenvalue farther than that from s.
//   * s_is_gap (certificate endpoints placed at gap midpoints of computed eigenvalues): that count
//     is taken at s itself - in a dense cluster any move would cross eigenvalues;
//   * otherwise (user bounds, interpolated cuts, band edges: arbitrary points that may coincide
//     with a - possibly multiple - eigenvalue, whose copies a count right at s would put on
//     arbitrary sides) it is taken at s + outward * step / 10, which resolves an eigenvalue at s
//     (1e-11 relative for the window counts: far above eps*growth, far below spectral gaps).
// Positions where even that fails (exactly singular K) move outward by step * 4^i. Returns the
// final s via `s`.
template <class T>
index_t count_below(Ctx& c, CountSource<T>& src, double& s, double outward, double step, bool* detection_off,
                    bool s_is_gap, bool* near_eigenvalue) {
  if (near_eigenvalue) *near_eigenvalue = false;
  for (int tries = 0; tries < 8; ++tries) {
    for (const bool detection : {true, false}) {
      if (!detection && tries == 0 && !s_is_gap) s += outward * 0.1 * step;
      const Inertia in = src.count(c, s, detection);
      if (in.valid) {
        if (!detection && c.rep.certificate.note.empty())
          c.rep.certificate.note = "endpoint inertia counted without null-pivot detection (dense cluster)";
        if (detection_off) *detection_off = !detection;
        return in.neg;
      }
      if (near_eigenvalue) *near_eigenvalue = true;
    }
    s += outward * step * std::pow(4.0, tries);
    ++c.rep.certificate.endpoint_moves;
  }
  HS_THROW(Singular, "cannot find a nonsingular certificate endpoint near " << s);
}

// Symbolic analysis at sigma0 and the memory-budget decision (DESIGN §3.6-3): in-core if the
// estimate fits the cap, else out-of-core if allowed and fitting, else OutOfMemoryBudget.
// An ordering that differs from analysis to analysis (SCOTCH, taken_on_prediction: multithreaded,
// not reproducible even with its random state reset, factor.hpp) whose estimate does not fit is
// analysed again, up to kMaxReanalyses times, before giving up (P8): its estimates scatter by up to
// ~30% - ss-3dspectralwave2 at the default 14 GiB budget had out-of-core estimates of 10.6..11.9 GB
// in most analyses and 13.4..14.4 GB in 6 of 42 (P5c and P8; 5 of them above the cap of ~13.49 GB;
// results/p8_certcost.md §2.3); each re-analysis costs one ordering (16 s there).
constexpr int kMaxReanalyses = 2;
template <class T>
void analyze_within_budget(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma0,
                           bool already_analyzed) {
  if (!already_analyzed) {
    PhaseScope phase(c.clock, Phase::Factor);
    fac.analyze(pencil, 1.0, -sigma0);
  }
  for (int attempt = 0;; ++attempt) {
    const FactorEstimate est = fac.estimate();
    event(c, "analysis", {{"factor_entries", static_cast<std::int64_t>(est.factor_entries)},
                          {"incore_mb", est.incore_mb}, {"ooc_mb", est.ooc_mb}, {"flops", est.flops}});
    const double cap_mb = fac.options().max_memory_mb;
    if (!(cap_mb > 0 && est.incore_mb > cap_mb)) return;
    if (c.cfg.out_of_core != Toggle::Off && est.ooc_mb <= cap_mb) {
      fac.set_out_of_core(true);
      warn(c, "factor estimate " + fmt(est.incore_mb) + " MB exceeds the budget; using out-of-core factors");
      return;
    }
    if (attempt < kMaxReanalyses && taken_on_prediction(fac.options().ordering)) {
      warn(c, "factor estimate " + fmt(est.incore_mb) + " MB (out-of-core " + fmt(est.ooc_mb) +
                  " MB) exceeds the budget of " + fmt(cap_mb) + " MB; analysing again (the ordering differs from run "
                  "to run)");
      PhaseScope phase(c.clock, Phase::Factor);
      fac.analyze(pencil, 1.0, -sigma0);
      continue;
    }
    HS_THROW(OutOfMemoryBudget, "predicted factorization memory " << est.incore_mb << " MB (out-of-core " << est.ooc_mb
                                                                  << " MB) exceeds the budget of " << cap_mb << " MB");
  }
}

// ------------------------------------------------------------------------------ cost model

namespace {

// Shift-invert Lanczos converging c pairs in one run (thick restart, ncv = 2c + 1), measured with
// bench/slicing_sweep (yee-pec-box 1e4 and 1e5, slices of 12..1075 pairs, 2 threads):
//   solves per pair      ~ kSolvesPerPair + kSolvesPerRun / c  (2.0 + 33/c fits 2.0 .. 5.0)
//   basis work per pair  ~ (kVecPassesPerPair + kVecPassesPerPairPerC * c) n-vector passes
//                          (CGS2 against a basis growing with c, B products, restarts): about 200
//                          passes plus a slope that grows with c, 4.5..9 per unit c over c =
//                          25..170 (n = 1e4 and 1e5) - 7 is used, which put the 1e5 optimum (~100
//                          pairs per slice, flat within ~10% for 64..128) closer than 5 did (155)
//   factorizations       ~ kFactorizationsPerSlice per slice (its shift + ~1.2 inertia counts to
//                          place the next cut)
// at kStreamBwPerThread bytes/s per thread for the basis passes (saturating at kStreamBwMax).
// Timings on the shared machine varied by 10..30% between identical runs (solves 69..98 ms on
// the same 1e5 factor), so the model only needs to land in the flat part of the optimum.
constexpr double kSolvesPerPair = 2.1;
constexpr double kSolvesPerRun = 33.0;
constexpr double kVecPassesPerPair = 200.0;
constexpr double kVecPassesPerPairPerC = 7.0;
constexpr double kFactorizationsPerSlice = 2.2;
constexpr double kStreamBwPerThread = 6e9;
constexpr double kStreamBwMax = 60e9;
constexpr index_t kMinSlice = 16;
constexpr index_t kMaxSlice = 1000;

}  // namespace

double CostModel::per_pair(index_t c) const {
  const double cc = static_cast<double>(std::max<index_t>(c, 1));
  return t_solve * (kSolvesPerPair + kSolvesPerRun / cc) +
         vec_bytes / stream_bw * (kVecPassesPerPair + kVecPassesPerPairPerC * cc);
}

double CostModel::marginal_pair(index_t c) const {
  // d/dc of c * per_pair(c): one more pair in a run that converges c.
  const double cc = static_cast<double>(std::max<index_t>(c, 1));
  return t_solve * kSolvesPerPair + vec_bytes / stream_bw * (kVecPassesPerPair + 2.0 * kVecPassesPerPairPerC * cc);
}

index_t CostModel::best_slice() const {
  // Per pair: (F t_factor + kSolvesPerRun t_solve) / c + const + kVecPassesPerPairPerC (vec/bw) c,
  // minimal at c* = sqrt((F t_factor + kSolvesPerRun t_solve) / (kVecPassesPerPairPerC vec/bw)).
  const double fixed = kFactorizationsPerSlice * t_factor + kSolvesPerRun * t_solve;
  const double slope = kVecPassesPerPairPerC * vec_bytes / stream_bw;
  const double c = slope > 0 ? std::sqrt(fixed / slope) : static_cast<double>(kMaxSlice);
  return std::clamp<index_t>(static_cast<index_t>(std::llround(c)), kMinSlice, kMaxSlice);
}

CostModel make_cost_model(const FactorEstimate& est, index_t n, int threads, bool complex_scalars) {
  CostModel m;
  const double order = static_cast<double>(n) * (complex_scalars ? 2.0 : 1.0);
  m.t_factor = model_factor_s(est, order, threads);
  m.t_solve = model_solve_s(est);
  m.vec_bytes = static_cast<double>(n) * (complex_scalars ? 16.0 : 8.0);
  m.stream_bw = std::min(kStreamBwMax, kStreamBwPerThread * std::max(1, threads));
  return m;
}

// The P4 rule of the adaptive ordering (DESIGN §3.6-3b until P5; internal.hpp). Nested dissection
// (SCOTCH) minimises fill but its analysis costs ~kNdAnalysisOverAmd AMD analyses; on 2-D-like
// graphs AMD's extra fill is cheap. Probe with AMD (fast), predict what nested dissection could
// save on the run's factorizations and solves, and pay for it only when the predicted saving
// exceeds its predicted analysis cost.
// Cost model (calibrated on the 2x Xeon 6336Y node, P3 measurements; only ratios matter):
//   t_factorize ~ 6e-7 n + flops / (12 GF/s * threads),  t_solve ~ 16 B * entries / 12 GB/s,
//   nested dissection keeps (flops, entries) >= (0.35, 0.6) of AMD's on 3-D-like graphs (AMD fill
//   ratio entries/nnz(K) > 20) and >= (0.6, 0.85) otherwise.
OrderingDecision decide_ordering(const FactorEstimate& amd, const MumpsInputSize& in, double pencil_nnz,
                                 int threads, index_t expected_factorizations, index_t expected_solves) {
  OrderingDecision d;
  const double t_f = model_factor_s(amd, in.order, std::max(1, threads));
  const double t_s = model_solve_s(amd);
  d.fill_ratio = static_cast<double>(amd.factor_entries) / std::max(1.0, pencil_nnz);
  const bool three_d_like = d.fill_ratio > 20.0;
  const double keep_f = three_d_like ? 0.35 : 0.6, keep_e = three_d_like ? 0.6 : 0.85;
  d.saving_s = static_cast<double>(expected_factorizations) * t_f * (1.0 - keep_f) +
               static_cast<double>(expected_solves) * t_s * (1.0 - keep_e);
  d.amd_analysis_s = model_analysis_s(in.order, in.entries, static_cast<double>(amd.factor_entries));
  d.nd_cost_s = kNdAnalysisOverAmd * d.amd_analysis_s;
  d.use_nd = d.saving_s > d.nd_cost_s;
  return d;
}

// Nested-dissection ordering of the adaptive analysis. SCOTCH 7 (ICNTL(7)=3) orders differently
// from analysis to analysis, in one process as across processes (its default context draws a
// random seed and orders with threads): syn-lap3d-22, three identical CLI runs with
// deterministic = true, 4 threads: 1.73 / 1.77 / 1.83e6 factor entries and different
// eigenvectors (SCOTCH_DETERMINISTIC=1 in the environment fixes that, but a library must not
// impose it on its host process). METIS (5) has a fixed seed: identical estimates in every
// analysis and bitwise equal results. Measured on the 42 data cases (bench/drvint_fix_ordering.cpp,
// 4 threads, 3 interleaved analyses each): the SCOTCH estimates varied between the analyses of one
// process on every case, the METIS ones on none. On the 20 cases where the AMD probe picks nested
// dissection, METIS analyses take 0.52x .. 1.98x SCOTCH's time (median 0.84x): faster on the 1e4
// and the largest 3-D cases (ss-ge87h76 4.87 -> 2.53 s), slower on the 1e5 3-D ones this solver
// targets (syn-mem27-46 0.82 -> 1.06 s, yee-pec-cube-32 0.64 -> 0.98 s, yee-pec-box-1e5 0.81 ->
// 1.19 s, ss-tem152078 1.07 -> 1.77 s); factorizations take 0.84x .. 1.06x. So the default keeps
// SCOTCH (its runs are reproducible to tolerance, like the L0-OpenMP factorizations) and only runs
// that promise bitwise reproducibility take METIS. One cost constant serves both: the median
// nested-dissection / AMD analysis ratio is 9.3 (SCOTCH) and 9.8 (METIS) on those cases.
int nd_ordering(const SolverConfig& cfg) { return cfg.deterministic ? 5 : 3; }

// ------------------------------------------------------------------------------ ordering choice

// What a solve does with its factors (PlannedWork), from the end-to-end runs of the data cases
// (hsolve CLI at 6a3d7c1, 4 threads, report op_applies / factor_stats; all nearest targets):
//   * HEP / GHEP nearest, block width p = block_size_for(cfg, k), the P4 width (the run itself
//     takes the blocktune rule's width, which needs the factorization this plan prices; the
//     constants were measured with the P4 widths): column applications ~ 40 + 4k for
//     p = 4 (32 cases, k = 10..26: measured / model median 1.04, range 0.50..1.48), 2k + 12 for
//     p = 1 (k = 6, 7: 0.92, 1.19); one factorization at sigma, B's inertia one more while
//     undecided, two certificate counts;
//   * GHIEP nearest (ghiep_block_size: p = 2 from n = 2e4, k >= 4): ~3.4k column solves (7 cases,
//     k = 18..28: 0.75..1.05) and one factorization; a shift in the zero band runs A's null-space
//     count, two probes and two one-sided runs (5 factorizations, ~20k applications: wg-square-1e5
//     at sigma = 0, ghiep_block_size measurements);
//   * windows (interval / smallest / largest): one slice of k (64 for intervals) pairs with ~2
//     factorizations and 4 counts - a rough guess; the data cases are all nearest targets.
// Only the relative cost of the orderings matters: rescue solves, reruns and window details move
// every candidate's price alike.
PlannedWork plan_work(const SolverConfig& cfg, ProblemType type, index_t n, double scale) {
  PlannedWork w;
  const Target& t = cfg.target;
  const index_t k = std::max<index_t>(1, std::min(n, t.mode == TargetMode::Interval ? index_t{64} : t.k));
  const double kk = static_cast<double>(k);
  if (type == ProblemType::GeneralizedIndefinite) {
    const double band = std::max(cfg.zero_filter_tol, kGhiepZeroBand) * scale;
    const bool one_sided = std::abs(t.sigma) <= band;
    w.block = std::max<index_t>(1, ghiep_block_size(cfg, n, k));
    w.factorizations = one_sided ? 5.0 : 1.0;
    w.solve_calls = (one_sided ? 20.0 * kk : 3.4 * kk) / static_cast<double>(w.block);
    return w;
  }
  w.block = std::max<index_t>(1, block_size_for(cfg, k));
  const double applies = w.block > 1 ? 40.0 + 4.0 * kk : 2.0 * kk + 12.0;
  w.solve_calls = applies / static_cast<double>(w.block);
  w.factorizations = 1.0 + (type == ProblemType::Auto ? 1.0 : 0.0);
  const bool window = t.mode != TargetMode::Nearest;
  if (window) w.factorizations += 1.0;
  w.counts = window ? 4.0 : (cfg.inertia_check != Toggle::Off ? 2.0 : 0.0);
  return w;
}

// The ordering cost model (OrderingModel, internal.hpp), fitted on the 40 data cases of MUMPS order
// >= 5000 at the judging configuration: bench/anaint_orderings.cpp in the gpu02 (2x Xeon Silver
// 4314) measurement slot, cores 16-31, 16 threads, one one-process run per case and ordering (P5c);
// the 1-thread work of hsolve's nested dissection and the 16 / 4-thread ratios come from the P5
// campaign (4 threads, medians of 3 interleaved runs, --nd-serial 1; run on cores 28-31 of the
// slot before the slot lock existed, so other runs may have shared that socket - only the
// thread exponents and W rest on it, and W only through the 16-thread median kNdSerial, fitted
// against it). bench/anaint_fix_fit.cpp reproduces every constant below from those raw outputs
// (nonnegative least squares on the relative error, medians, the shape regression) and replays the
// driver's choice on the measured components, in-sample and leave-one-case-out (P5c fix).
// Inputs: the MUMPS input size (mumps_input_size: order, upper entries), the graph hsolve's nested
// dissection orders (OrderingInput::nd) and the factor entries fe / flops of an analysis estimate.
// In brackets: the range of predicted / measured over the fitted cases.
//   * Given-order analysis (MUMPS analysis with PERM_IN: hsolve's nested dissection hands its
//     order over, the InertiaCounter reuses the main one), the floor of every analysis
//     [0.58, 1.54].
//   * hsolve's nested dissection on one thread: W = w_v * vertices + w_e * edges + w_fe * fe of
//     its graph [0.47, 1.38]; on p threads W (nd_serial + (1 - nd_serial) / p), nd_serial the
//     median over the 40 cases with W > 0.05 s of the measured 16-thread ordering times
//     (quartiles 0.26, 0.47, range 0.13..0.86; whole analysis [0.57, 2.04]). It does not scale
//     like Amdahl's law with one serial fraction: 4 threads gave 0.19 (P5), 16 threads 0.38 - the
//     16-thread analysis is only 1.14x faster than at 4 threads (median over the 29 cases measured
//     at both). The worst misses are W's own: ss-tem152078's 1-thread ordering took 2.1x and
//     syn-mem125-46's 1.9x the fitted W (no input statistic tried - degree, fill, separator sizes
//     - accounts for it), which makes ss-tem152078 take hsolve's nested dissection where SCOTCH's
//     analysis is twice as fast (P5c review: 1.15x main's time). Pricing ND's analysis risk against
//     SCOTCH (0.5 of its predicted analysis added while not analysed) fixed that case but, replayed
//     with bench/anaint_fix_fit.cpp, sent yee-bloch-1e4, yee-photonic-*, syn-lap3d-aniso-1e5,
//     syn-mem27-46 and ss-qa8fk-qa8fm to SCOTCH as well: mean excess over the best ordering 4.3% ->
//     5.3% at equal total time - not adopted.
//   * MUMPS orderings are serial, on the full input graph: METIS, PORD and SCOTCH cost rho * W (W
//     of the MUMPS input with their own fe; medians; whole analysis [0.45, 1.0], except the
//     low-fill block graphs of the syn-* cases, which they order 3-8x faster); AMD's ordering
//     a_order * order + a_entry * entries [0.25, 1.31].
//   * Factorization flops and entries of an ordering relative to AMD's on the same input, as
//     functions of AMD's fill ratio f = fe(AMD) / nnz(pencil): ln r piecewise linear in ln f
//     between the knots kShapeKnotFill (constant beyond), least squares with a ridge of 0.05. Low
//     fill (block or banded structure, f < 5): nested dissections need 2-3x AMD's flops; 2-D
//     (f ~ 10): 0.4-0.9x; 3-D (f >= 40): 0.1-0.35x. Flops [0.44, 1.86], entries [0.64, 1.36].
//   * Numeric phases at 16 threads: factorization t_f = f_order * order + f_entry * entries +
//     f_fe * fe + f_flops * flops [0.23, 1.79] (input assembly and scaling, front assembly, dense
//     kernels); a refactorization of the same object `refactor` t_f, an InertiaCounter count
//     `count` t_f (no factors kept); a 1-RHS solve t_s = s_order * order + s_fe * fe [0.30, 1.44],
//     p right-hand sides in one call (1 + solve_per_rhs (p - 1)) t_s (medians: 0.97 t_s for p = 2,
//     1.19 t_s for p = 4).
//   * Other thread counts: (16 / threads)^gamma, from the medians of t(16) / t(4) over the same
//     cases and orderings: factorizations 0.529 (150 pairs, gamma 0.46), solves 0.735 (57 pairs,
//     gamma 0.22). Below 4 threads an extrapolation (2 against 4 threads measured 0.68 and 0.58 in
//     P5).
//   * AMD at low fill (fe / nnz(pencil) <= kAmdLowFill: the block / banded syn-* cases): its
//     numeric phases take far longer than their size predicts at 16 threads - deep, narrow
//     elimination trees without separators leave the threads of the tree-parallel L0 layer idle.
//     AMD's measured / predicted time over hsolve's nested dissection's on the same case, medians
//     over the 13 data cases with fill <= 4: factorization 1.86 (1.37..2.26), 4-RHS solve 3.15
//     (2.50..3.81, 10 cases); at fill 5.1..10.8 (2-D grids) 1.05..1.25, so the factor falls to 1 at
//     kAmdFillEnd (ln-linear between). Measured at 16 threads only: scaled by ln(threads) / ln(16),
//     none on one thread.
const OrderingModel& ordering_model() {
  static const OrderingModel m = [] {
    OrderingModel x{};
    x.floor_order = 2.25e-7, x.floor_entry = 3.44e-8, x.floor_fe = 0.0;
    x.nd_work_vertex = 7.03e-6, x.nd_work_edge = 6.54e-8, x.nd_work_fe = 3.85e-9, x.nd_serial = 0.383;
    x.rho_metis = 1.033, x.rho_pord = 1.126, x.rho_scotch = 0.626;
    x.amd_order = 3.26e-8, x.amd_entry = 5.32e-9;
    x.shapes[0] = {kOrderingNestedDissection,
                   {0.81, 1.00, -0.52, -1.03, -1.32, -1.90},
                   {0.41, 0.51, -0.18, -0.43, -0.63, -1.00}};
    x.shapes[1] = {5, {0.89, 0.93, -0.58, -1.04, -1.34, -2.00}, {0.45, 0.48, -0.24, -0.45, -0.64, -1.04}};
    x.shapes[2] = {4, {0.65, 0.83, -0.64, -0.82, -1.07, -1.14}, {0.34, 0.41, -0.24, -0.35, -0.53, -0.65}};
    x.shapes[3] = {3, {0.88, 1.01, -0.49, -1.03, -1.43, -1.87}, {0.44, 0.52, -0.12, -0.39, -0.66, -0.98}};
    x.fac_order = 1.91e-7, x.fac_entry = 3.21e-8, x.fac_fe = 5.95e-9, x.fac_flops = 2.74e-12;
    x.refactor = 0.962, x.count = 0.886;
    x.solve_order = 5.16e-8, x.solve_fe = 7.10e-10, x.solve_per_rhs = 0.062;
    x.fac_thread_gamma = 0.46, x.solve_thread_gamma = 0.22;
    x.amd_low_fill_factorization = 1.86, x.amd_low_fill_solve = 3.15;
    return x;
  }();
  return m;
}

namespace {

constexpr double kFitThreads = 16.0;
constexpr double kAmdLowFill = 4.0, kAmdFillEnd = 5.0;

// Expected gain of analysing a candidate rather than trusting its prediction, as a fraction of its
// predicted numeric time: from AMD's estimate the nested dissections' fill is known to within
// 0.5..1.9x; from another nested dissection's actual estimate within ~10% (METIS and SCOTCH had
// 0.82..1.10x hsolve's flops on the 3-D cases).
constexpr double kGainFromAmd = 0.05, kGainFromNd = 0.03;
/// Score of an analysed candidate whose in-core estimate exceeds the factorization cap, per unit of
/// estimate / cap: above every feasible score, ordered by memory.
constexpr double kOverCapScore = 1e30;

const OrderingModel::Shape* shape_fit(const OrderingModel& m, int ordering) {
  for (const auto& f : m.shapes)
    if (f.ordering == ordering) return &f;
  return nullptr;
}

double thread_factor(int threads, double gamma) {
  return std::pow(kFitThreads / static_cast<double>(std::max(1, threads)), gamma);
}

/// AMD's low-fill factor (see kAmdLowFill) at fill ratio `fill` on `threads` threads.
double amd_low_fill_factor(double factor, double fill, int threads) {
  const double share = std::clamp(std::log(static_cast<double>(std::max(1, threads))) / std::log(kFitThreads), 0.0, 1.0);
  const double lf = std::log(std::max(fill, 1.0)), lo = std::log(kAmdLowFill), hi = std::log(kAmdFillEnd);
  const double weight = lf <= lo ? 1.0 : lf >= hi ? 0.0 : (hi - lf) / (hi - lo);
  return 1.0 + (factor - 1.0) * weight * share;
}

double given_order_analysis_s(const OrderingModel& m, const MumpsInputSize& in, double fe) {
  return m.floor_order * in.order + m.floor_entry * in.entries + m.floor_fe * fe;
}

double nd_work_s(const OrderingModel& m, const MumpsInputSize& g, double fe) {
  return m.nd_work_vertex * g.order + m.nd_work_edge * g.entries + m.nd_work_fe * fe;
}

/// (flops, entries) of `ordering` relative to AMD's at AMD fill ratio f.
std::pair<double, double> shape_ratio(const OrderingModel& m, int ordering, double fill) {
  const OrderingModel::Shape* s = shape_fit(m, ordering);
  if (!s) return {1.0, 1.0};  // AMD itself (and orderings without a fit)
  return {std::exp(shape_log_ratio(s->flops, fill)), std::exp(shape_log_ratio(s->entries, fill))};
}

bool nested_dissection(int ordering) { return ordering != 0; }

}  // namespace

double shape_log_ratio(const double (&knots)[OrderingModel::kShapeKnots], double fill) {
  constexpr int k = OrderingModel::kShapeKnots;
  const double lf = std::log(std::clamp(fill, kShapeKnotFill[0], kShapeKnotFill[k - 1]));
  for (int i = 0; i + 1 < k; ++i) {
    const double a = std::log(kShapeKnotFill[i]), b = std::log(kShapeKnotFill[i + 1]);
    if (lf <= b) return knots[i] + (knots[i + 1] - knots[i]) * (lf - a) / (b - a);
  }
  return knots[k - 1];
}

template <class T>
OrderingInput ordering_input(const ShiftedPencil<T>& pencil) {
  OrderingInput in;
  in.mumps = mumps_input_size(pencil);
  in.nd = in.mumps;
  if constexpr (is_complex_v<T>) {
    // compute_ordering's Embedding::Auto: the n-vertex graph of K when most couplings are complex.
    if (ordering::complex_coupling_fraction(pencil) >= ordering::kCompressedFraction)
      in.nd = {static_cast<double>(pencil.order()), static_cast<double>(pencil.nnz())};
  }
  return in;
}

// Candidates of the adaptive ordering, in order of preference on near-equal scores: AMD, SCOTCH,
// hsolve's nested dissection, METIS, PORD. SCOTCH's orderings differ from analysis to analysis
// (nd_ordering), so it is taken on its prediction (taken_on_prediction): once it is the
// best-scoring candidate it is analysed and used, its estimate never compared with another
// candidate's - the choice stays a function of deterministic estimates. Why SCOTCH and why before
// hsolve's nested dissection: at 16 threads its factors need 0.80..0.95x the flops of hsolve's on
// the high-fill 3-D cases (Yee grids, ss-ge87h76, ss-tem152078; bench/anaint_orderings.cpp), which
// the fill-based shape fits see only beyond fill ~150; end to end (16 threads, P5c) hsolve's
// nested dissection in its place cost yee-pec-box-1e5 13% and ss-ge87h76 17% (69 vs 59 s).
std::vector<int> candidate_orderings(const SolverConfig& /*cfg*/) {
  return {0, kOrderingScotch, kOrderingNestedDissection, 5, 4};
}

bool taken_on_prediction(int ordering) { return ordering == kOrderingScotch; }

double ordering_analysis_s(const OrderingModel& m, int ordering, const OrderingInput& in, double factor_entries,
                           int threads) {
  const double floor = given_order_analysis_s(m, in.mumps, factor_entries);
  switch (ordering) {
    case 0: return floor + m.amd_order * in.mumps.order + m.amd_entry * in.mumps.entries;
    case kOrderingNestedDissection: {
      const double p = static_cast<double>(std::max(1, threads));
      return floor + nd_work_s(m, in.nd, factor_entries) * (m.nd_serial + (1.0 - m.nd_serial) / p);
    }
    case 4: return floor + m.rho_pord * nd_work_s(m, in.mumps, factor_entries);
    case 3: return floor + m.rho_scotch * nd_work_s(m, in.mumps, factor_entries);
    default: return floor + m.rho_metis * nd_work_s(m, in.mumps, factor_entries);  // METIS and the rest
  }
}

double ordering_analysis_s(int ordering, const OrderingInput& in, double factor_entries, int threads) {
  return ordering_analysis_s(ordering_model(), ordering, in, factor_entries, threads);
}

double factorization_s(const OrderingModel& m, const MumpsInputSize& in, double flops, double entries, int threads) {
  return (m.fac_order * in.order + m.fac_entry * in.entries + m.fac_fe * entries + m.fac_flops * flops) *
         thread_factor(threads, m.fac_thread_gamma);
}

double factorization_s(const MumpsInputSize& in, double flops, double entries, int threads) {
  return factorization_s(ordering_model(), in, flops, entries, threads);
}

double planned_numeric_s(const OrderingModel& m, const SolverConfig& cfg, const PlannedWork& w,
                         const MumpsInputSize& in, double flops, double entries, int threads, double amd_fill) {
  double t_f = factorization_s(m, in, flops, entries, threads);
  double t_s = (m.solve_order * in.order + m.solve_fe * entries) * thread_factor(threads, m.solve_thread_gamma);
  if (amd_fill > 0.0) {
    t_f *= amd_low_fill_factor(m.amd_low_fill_factorization, amd_fill, threads);
    t_s *= amd_low_fill_factor(m.amd_low_fill_solve, amd_fill, threads);
  }
  double s = t_f + std::max(0.0, w.factorizations - 1.0) * m.refactor * t_f;
  if (w.counts > 0) {
    // As the driver will count: on an InertiaCounter when counter_pays (its analysis on the given
    // order, then inertia-only factorizations), else refactorizations of the main object.
    FactorEstimate e;
    e.flops = flops;
    e.factor_entries = static_cast<std::int64_t>(entries);
    const auto counts = static_cast<index_t>(std::llround(w.counts));
    s += counter_pays(cfg, e, in, threads, counts) ? given_order_analysis_s(m, in, entries) + w.counts * m.count * t_f
                                                   : w.counts * m.refactor * t_f;
  }
  s += w.solve_calls * t_s * (1.0 + m.solve_per_rhs * static_cast<double>(std::max<index_t>(w.block, 1) - 1));
  return s;
}

double planned_numeric_s(const SolverConfig& cfg, const PlannedWork& w, const MumpsInputSize& in, double flops,
                         double entries, int threads, double amd_fill) {
  return planned_numeric_s(ordering_model(), cfg, w, in, flops, entries, threads, amd_fill);
}

std::pair<double, double> predicted_shape(const OrderingModel& m, int ordering, const FactorEstimate& amd,
                                          double pencil_nnz, int anchor_ordering, const FactorEstimate& anchor) {
  const double fill = static_cast<double>(amd.factor_entries) / std::max(1.0, pencil_nnz);
  const auto r = shape_ratio(m, ordering, fill), ra = shape_ratio(m, anchor_ordering, fill);
  return {anchor.flops * r.first / ra.first, static_cast<double>(anchor.factor_entries) * r.second / ra.second};
}

std::pair<double, double> predicted_shape(int ordering, const FactorEstimate& amd, double pencil_nnz,
                                          int anchor_ordering, const FactorEstimate& anchor) {
  return predicted_shape(ordering_model(), ordering, amd, pencil_nnz, anchor_ordering, anchor);
}

void OrderingPricer::analysed(std::vector<OrderingCandidate>& cand, std::size_t i, const FactorEstimate& e) {
  if (i == 0) amd = e;  // the probe
  OrderingCandidate& c = cand[i];
  c.analysed = true;
  c.analysis_s = 0.0;
  c.flops = e.flops;
  c.entries = static_cast<double>(e.factor_entries);
  c.numeric_s = planned_numeric_s(model, cfg, work, in.mumps, c.flops, c.entries, threads,
                                  c.ordering == 0 ? c.entries / std::max(1.0, pencil_nnz) : 0.0);
  c.incore_mb = e.incore_mb;
  c.score_s = c.numeric_s;
  // Factors beyond the in-core cap (DESIGN §3.6-3) only run out of core, or not at all: any
  // candidate that fits is preferred, and among those that do not, the smallest.
  if (cap_mb > 0 && e.incore_mb > cap_mb) c.score_s = kOverCapScore * (e.incore_mb / cap_mb);
}

void OrderingPricer::predict(std::vector<OrderingCandidate>& cand, std::size_t anchor) const {
  FactorEstimate a;
  a.flops = cand[anchor].flops;
  a.factor_entries = static_cast<std::int64_t>(cand[anchor].entries);
  for (auto& c : cand) {
    if (c.analysed) continue;
    const auto [flops, entries] = predicted_shape(model, c.ordering, amd, pencil_nnz, cand[anchor].ordering, a);
    c.flops = flops;
    c.entries = entries;
    c.analysis_s = ordering_analysis_s(model, c.ordering, in, entries, threads);
    c.numeric_s = planned_numeric_s(model, cfg, work, in.mumps, flops, entries, threads,
                                    c.ordering == 0 ? entries / std::max(1.0, pencil_nnz) : 0.0);
    const double gain = nested_dissection(cand[anchor].ordering) ? kGainFromNd : kGainFromAmd;
    c.score_s = c.analysis_s + (1.0 - gain) * c.numeric_s;
    // In-core size in proportion to the factor entries, from the anchor's estimate: a candidate
    // predicted beyond the cap is scored like an analysed one beyond it, so that, when nothing
    // fits (out-of-core runs), candidates are not analysed one after the other for nothing
    // (ss-3dspectralwave2: all four were, 50 s of analyses).
    c.incore_mb = cand[anchor].incore_mb * entries / std::max(1.0, cand[anchor].entries);
    if (cap_mb > 0 && c.incore_mb > cap_mb) c.score_s = kOverCapScore * (c.incore_mb / cap_mb);
  }
}

std::size_t choose_ordering(const std::vector<OrderingCandidate>& candidates) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i)
    if (candidates[i].score_s < (1.0 - kOrderingMargin) * candidates[best].score_s) best = i;
  return best;
}

// The analysis loop (adaptive_analysis). The analysed candidate kept is an incumbent: each newly
// analysed one is compared with it alone, by choose_ordering's rule for the two (the earlier
// candidate stays unless the later one beats it by the margin), and the loser is released at once.
// choose_ordering over all analysed candidates instead is not monotone - a candidate released
// because it lost may win again once another one joins (AMD 100, METIS 90, PORD 88: METIS, PORD
// released; with ND 94 added: PORD) - and the driver then asked for a released analysis (P5c
// review: 1 of 400k random replays; the incumbent rule cannot).
std::size_t run_ordering_choice(std::vector<OrderingCandidate>& cand, const std::function<void(std::size_t)>& analyse,
                                const std::function<void(std::size_t)>& predict,
                                const std::function<void(std::size_t)>& release) {
  HS_ASSERT(!cand.empty() && cand[0].ordering == 0, "run_ordering_choice: AMD must be the first candidate");
  analyse(0);  // the AMD probe
  predict(0);
  std::size_t incumbent = 0;
  auto duel = [&](std::size_t i, std::size_t j) {  // choose_ordering on the pair, in candidate order
    const std::size_t lo = std::min(i, j), hi = std::max(i, j);
    return cand[hi].score_s < (1.0 - kOrderingMargin) * cand[lo].score_s ? hi : lo;
  };
  // Analyse the best-scoring candidate while it is not analysed yet: then no further analysis is
  // expected to pay. A candidate taken on its prediction (SCOTCH) is the choice once analysed -
  // unless its factors exceed the in-core cap; it never anchors the predictions of the others.
  for (std::size_t next = choose_ordering(cand); !cand[next].analysed; next = choose_ordering(cand)) {
    analyse(next);
    const bool on_prediction = taken_on_prediction(cand[next].ordering);
    if (on_prediction && cand[next].score_s < kOverCapScore) {
      release(incumbent);
      return next;
    }
    const std::size_t keep = duel(incumbent, next);
    release(keep == next ? incumbent : next);
    incumbent = keep;
    if (!on_prediction) predict(next);
  }
  return incumbent;
}

// Adaptive fill-reducing ordering (DESIGN §3.6-3b). The AMD probe (fast; its estimate tells 2-D-like
// from 3-D-like fill) prices every candidate on the planned work: AMD by its own estimate, the
// others by the shape fits. The best-scoring candidate is analysed and its actual estimate replaces
// the prediction (the remaining nested dissections are then predicted from it, more tightly); the
// loop ends when the best-scoring candidate is an analysed one, or SCOTCH (taken on its
// prediction, candidate_orderings). A second analysis beyond the probe therefore runs only where
// the numeric work dwarfs it (the expected gain of an actual estimate, kGainFrom*, is a fraction
// of the numeric time). The loop is run_ordering_choice: the analysed candidate kept is compared
// with each newly analysed one, and the loser's analysis is freed at once (at most two live).
// Every input of the decisions is deterministic (estimates of deterministic orderings, models,
// thread count; SCOTCH's estimate only for the in-core cap): identical runs choose the same
// ordering. Wall times are logged, never used.
template <class T>
std::unique_ptr<SparseFactorization<T>> adaptive_analysis(Ctx& c, const FactorOptions& fopts,
                                                          const ShiftedPencil<T>& pencil, double sigma0,
                                                          const PlannedWork& work) {
  PhaseScope phase(c.clock, Phase::Factor);
  const OrderingInput oin = ordering_input(pencil);
  const MumpsInputSize& in = oin.mumps;
  const double nnz = static_cast<double>(pencil.nnz());
  const int threads = std::max(1, fopts.threads);
  if (c.cfg.deterministic) {
    // The P4 rule and its inputs, unchanged: AMD or METIS. deterministic_runs_are_bitwise_equal in
    // tests/cpp/test_drvint_fix.cpp pins METIS for its 3-D Laplacian; hsolve's nested dissection
    // and PORD are deterministic as well, so the P5 choice could serve these runs once that test
    // accepts any deterministic ordering (interface request of module anaint).
    FactorOptions amd = fopts;
    amd.ordering = 0;
    auto fac = make_mumps_factorization<T>(amd);
    Timer t;
    fac->analyze(pencil, 1.0, -sigma0);
    const double t_amd = t.seconds();
    const FactorEstimate e = fac->estimate();
    const Target& tg = c.cfg.target;
    const index_t kk = tg.mode == TargetMode::Interval ? 32 : tg.k;
    const index_t f4 = 1 + (c.cfg.inertia_check != Toggle::Off ? 2 : 0) + (pencil.b() ? 1 : 0);
    const OrderingDecision d = decide_ordering(e, in, nnz, threads, f4, std::max<index_t>(40, 3 * kk));
    event(c, "ordering", {{"choice", static_cast<std::int64_t>(d.use_nd ? nd_ordering(c.cfg) : 0)},
                          {"rule", std::string("p4")}, {"amd_analyze_s", t_amd}, {"amd_flops", e.flops},
                          {"amd_entries", static_cast<std::int64_t>(e.factor_entries)}, {"fill_ratio", d.fill_ratio},
                          {"predicted_saving_s", d.saving_s}, {"predicted_nd_cost_s", d.nd_cost_s}});
    if (!d.use_nd) return fac;
    fac.reset();
    FactorOptions nd = fopts;
    nd.ordering = nd_ordering(c.cfg);
    auto fac_nd = make_mumps_factorization<T>(nd);
    fac_nd->analyze(pencil, 1.0, -sigma0);
    return fac_nd;
  }
  const std::vector<int> orderings = candidate_orderings(c.cfg);
  std::vector<OrderingCandidate> cand(orderings.size());
  for (std::size_t i = 0; i < orderings.size(); ++i) cand[i].ordering = orderings[i];
  std::vector<std::unique_ptr<SparseFactorization<T>>> obj(orderings.size());
  std::vector<double> seconds(orderings.size(), 0.0);
  OrderingPricer pricer{ordering_model(), c.cfg, work, oin, nnz, threads, fopts.max_memory_mb, {}};
  auto analyse = [&](std::size_t i) {
    FactorOptions o = fopts;
    o.ordering = orderings[i];
    obj[i] = make_mumps_factorization<T>(o);
    Timer t;
    obj[i]->analyze(pencil, 1.0, -sigma0);
    seconds[i] = t.seconds();
    pricer.analysed(cand, i, obj[i]->estimate());
  };
  auto predict = [&](std::size_t anchor) { pricer.predict(cand, anchor); };
  const std::size_t pick = run_ordering_choice(cand, analyse, predict, [&](std::size_t i) { obj[i].reset(); });
  for (std::size_t i = 0; i < cand.size(); ++i)
    if (i != pick) obj[i].reset();
  const FactorEstimate& amd_estimate = pricer.amd;
  const double fill = cand[0].entries / std::max(1.0, nnz);
  // The log: every candidate's price, which were analysed (with their wall time), the choice.
  std::string summary;
  for (std::size_t i = 0; i < cand.size(); ++i) {
    summary += (i ? " " : "") + std::to_string(orderings[i]) + (cand[i].analysed ? ":a" : ":p") + "/" +
               fmt(cand[i].score_s) + "s";
    if (cand[i].analysed) summary += "/" + fmt(seconds[i]) + "s";
  }
  event(c, "ordering", {{"choice", static_cast<std::int64_t>(orderings[pick])},
                        {"amd_flops", amd_estimate.flops},
                        {"amd_entries", static_cast<std::int64_t>(amd_estimate.factor_entries)},
                        {"fill_ratio", fill}, {"amd_analyze_s", seconds[0]},
                        {"work_factorizations", work.factorizations}, {"work_counts", work.counts},
                        {"work_solve_calls", work.solve_calls}, {"work_block", static_cast<std::int64_t>(work.block)},
                        {"candidates", summary}});
  HS_ASSERT(obj[pick], "adaptive_analysis: the chosen ordering's analysis was freed");
  return std::move(obj[pick]);
}

#define HS_INSTANTIATE_DRIVER_STEPS(T)                                                                        \
  template void recompute_products<T>(const CsrMatrix<T>&, const CsrMatrix<T>*, Block<T>&);                  \
  template Block<T> take_columns<T>(const Block<T>&, const std::vector<index_t>&);                          \
  template void append_columns<T>(DenseMatrix<T>&, const DenseMatrix<T>&);                                  \
  template double factorize_near<T>(Ctx&, SparseFactorization<T>&, double, double, bool&, bool);            \
  template void refactorize_at<T>(Ctx&, SparseFactorization<T>&, double, bool);                             \
  template bool try_refactorize_at<T>(Ctx&, SparseFactorization<T>&, double, bool);                         \
  template class CountSource<T>;                                                                            \
  template index_t count_below<T>(Ctx&, CountSource<T>&, double&, double, double, bool*, bool, bool*);      \
  template void analyze_within_budget<T>(Ctx&, SparseFactorization<T>&, const ShiftedPencil<T>&, double, bool); \
  template MumpsInputSize mumps_input_size<T>(const ShiftedPencil<T>&);                                     \
  template OrderingInput ordering_input<T>(const ShiftedPencil<T>&);                                       \
  template std::unique_ptr<SparseFactorization<T>> adaptive_analysis<T>(                                    \
      Ctx&, const FactorOptions&, const ShiftedPencil<T>&, double, const PlannedWork&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DRIVER_STEPS)
#undef HS_INSTANTIATE_DRIVER_STEPS

// ------------------------------------------------------------------------------ block Krylov

index_t lanczos_basis_columns(index_t n, index_t d, index_t nev, index_t ncv, index_t p) {
  eigen_detail::LanczosSetup s;
  s.n = n;
  s.d = d;
  s.nev = nev;
  s.n_free = n - d;
  s.ncv = ncv > 0 ? std::min(ncv, s.n_free) : std::min(s.n_free, std::max(2 * nev + 1, nev + 16));
  if (p <= 1) return d + s.ncv + 2;
  const eigen_detail::BlockDimensions dims = eigen_detail::block_dimensions(s, p, ncv == 0);
  if (!dims.feasible) return d + s.n_free + 2;
  return d + dims.ncv + 2 * p;
}

// Deflated reruns (missing copies found by the certificate). Without the cap, p = missing rounded
// up to even cost more than the single-vector rerun it replaced (5 interleaved runs, 4 threads,
// block_size = 1 so that reruns happen): syn-ghep-multiplicity-real-1e4 (9 missing): single
// vector 59 applications in the rerun / iterate 0.221 s, p = 10: 200 / 0.314 s, cap 4: 0.226 s;
// syn-multiplicity-complex-1e4 0.444 / 0.403 / 0.416 s (noise). After deflation the missing copies
// dominate the operator, so a single-vector rerun usually gets them through rounding too; a block
// of width >= the multiplicity gets them by construction. With the auto width (4) no data case
// needed a rerun at all.
index_t rerun_block_size(index_t p, index_t missing) {
  return std::max<index_t>(std::max<index_t>(p, 1), std::min<index_t>(kMaxAutoBlockSize, (missing + 1) / 2 * 2));
}

// An auto width enters capped at the P4 width: a rerun wants missing + 2 pairs, the regime in
// which the P4 measurements below found 8 slower than 4, whereas the blocktune rule picks 8 for
// the main run's k = 10..12 pairs. Measured (gpu01 slot, 16 threads, 5 interleaved runs,
// certificate_fault_drop = 1, rule width 8 in the main run; rerun phase = refactorization at
// sigma + rerun iterations, both ending with the missing pair): rerun at 8 -> 4 on
// ss-2cubes_sphere 1.301 -> 1.197 s (168 -> 100 applications), yee-pec-cube-32 0.788 -> 0.735 s,
// syn-mem27-46 (certificate auto) 0.235 -> 0.185 s, yee-bloch-1e4 (auto) 0.165 -> 0.131 s; no
// overlap of the ranges. Against the integration tree (rerun at 8, no lead_nev) the whole solve
// is 1.05x, 1.09x, 1.33x and 1.14x faster there.
index_t rerun_block_size(const SolverConfig& cfg, index_t p, index_t missing) {
  return rerun_block_size(cfg.block_size > 0 ? p : std::min(p, kMaxAutoBlockSize), missing);
}

// P4 auto block width (spectrum slices and plan_work; nearest-target runs take the blocktune rule
// below, whose measurements and calibration are in src/eigen/block_rule.cpp), from end-to-end
// runs of p = 1, 4, 8 (ncv auto) on the 30 HEP/GHEP data cases
// (bench reproduction: tools/run/guard.py + hsolve --block-size, medians of 3 interleaved runs,
// 4 threads; the full table is in the drvint report). Iterate-phase time p = 1 -> 4 -> 8:
//   3-D, 1e5 (syn-mem27-46, yee-pec-cube-32, yee-pec-box-1e5, ss-2cubes_sphere, ss-qa8fk-qa8fm,
//     syn-lap3d-aniso-1e5; k = 10..12): 1.4x .. 2.5x faster with p = 4; p = 8 better than 4 only
//     where multiple eigenvalues dominate (syn-mem27-46 3.6x, yee-pec-cube-32 3.0x), worse on the
//     others (distinct eigenvalues: p blocks need ~2x (p = 4) / ~3x (p = 8) the applications of
//     p = 1, and a p-RHS solve costs 0.3..0.55 of p single ones per vector);
//   low fill, 1e5 (syn-zero-cluster-real, syn-ghep-near-singular-real, syn-near-singular-complex;
//     k = 10..21): 1.10x .. 1.16x with p = 4, p = 8 no gain or slower;
//   3-D complex 1e4 (yee-bloch, yee-photonic-*): 2.0x; multiple eigenvalues (syn-multiplicity-*,
//     syn-ghep-multiplicity; no certificate rerun needed with p = 4): 1.4x .. 2.2x in total;
//   small low-fill 1e4 cases lose up to 0.1 s (syn-ghep-zero-cluster-complex-1e4 0.58 -> 0.80 s,
//     syn-near-singular-complex-1e4 0.33 -> 0.46 s), the sum over the 11 syn-*-1e4 cases still
//     drops (3.91 -> 3.73 s);
//   k = 6, 7 (cav2d-rect-pert-1e5, cav2d-square-1e4): p = 4 is slower (0.37 -> 0.55 s, 22 -> 60
//     applications: the single-vector first cycle of max(2k + 1, k + 16) vectors converges all k
//     at once, a block run needs more applications for the same polynomial degree).
// Hence p = 4 for nev >= 8, else 1. A timed 1-RHS vs p-RHS probe solve was not used: its outcome
// would depend on the machine load, and the eigenvectors with it (reproducibility).
index_t block_size_for(const SolverConfig& cfg, index_t nev) {
  if (cfg.block_size > 0) return cfg.block_size;
  return nev >= kAutoBlockMinNev ? kMaxAutoBlockSize : 1;
}

// Nearest targets (P5c blocktune): the rule prices each width's block step from the shift's
// factor entries, n, nnz(B) and the scalar type at 16 threads and keeps p >= 4 from nev >= 8 on
// (block_rule.hpp; calibrated without the BLAS hook: 0.98x .. 1.19x against the P4 width there).
// In the P5c integration tree (BLAS hook, route all) the wired default against the P4 width, one
// binary, bench/blocktune_fix_e2e.cpp --p 0,4 / --p 0,1 on its 12 cases, 16 threads in the
// measurement slot, three campaigns of 3 - 5 interleaved runs, pooled medians, every run valid by
// tools/eval: gains where it moves 4 -> 8 on multiple eigenvalues (yee-pec-cube-32 1.08x,
// syn-mem27-46 1.05x) or 1 -> 2 at k = 6, 7 (cav2d-square-1e4 1.08x, cav2d-rect-pert-1e5 1.02x);
// losses where it moves 4 -> 8 on distinct ones (ss-2cubes_sphere 0.96x, yee-pec-box-1e5 0.97x,
// yee-bloch-1e4 0.98x; with HS_BLAS_ROUTE=off 1.00x, 0.98x, 1.03x: the hook's narrow kernels made
// the p <= 4 solves cheaper than the rule's step model assumes); the six cases where both widths
// are 4 within +-1 %; geometric mean 1.01x. Never slower than p = 1 (blocktune: 1.03x .. 2.45x).
// Since then the step from 4 to 8 needs a modelled gain of kRuleWideMargin = 10 % (review of the
// integration, 20 definite data cases in the slot, block_rule.cpp "BLAS hook"): ss-2cubes_sphere,
// yee-pec-box-1e5 and syn-lap3d-aniso-1e5 run at 4, the multiplicity cases keep 8.
index_t block_size_for(const SolverConfig& cfg, const eigen_detail::ShiftInvertShape& shape) {
  if (cfg.block_size > 0) return cfg.block_size;
  return eigen_detail::recommended_block_size(shape);
}

// The memory plan reserves the P4 width (internal.hpp). The p = 8 basis reserved for nearest
// targets in the P5c integration shrank the factorization cap by 27 n-vectors at k = 10..12: on
// syn-near-singular-complex-1e5 (AMD, 4 threads; factors in core 142 MB, out of core 74 MB)
// budgets of 0.3692 and 0.3706 GiB held the factors in core beside the P4 basis (cap 143..145 MB)
// but sent them out of core beside the p = 8 one (cap 102 / 105 MB; out_of_core = off:
// OutOfMemoryBudget), where the rule then picked 4 anyway. With the P4 plan the auto width runs
// in core there like --block-size 4.
index_t reserved_block_size(const SolverConfig& cfg) {
  return std::min(cfg.block_size > 0 ? cfg.block_size : kMaxAutoBlockSize, kMaxAutoBlockSize);
}

template <class T>
index_t fit_block_size(Ctx& c, index_t n, index_t nev, index_t d, index_t extra_columns, index_t p,
                       index_t warn_below) {
  const double vec = static_cast<double>(n) * sizeof(T);
  const double avail =
      c.cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
  const auto bytes = [&](index_t q) {
    return vec * static_cast<double>(lanczos_basis_columns(n, d, nev, c.cfg.ncv, q) + extra_columns);
  };
  const index_t wanted = std::max<index_t>(p, 1);
  p = wanted;
  while (p > 1 && bytes(p) > 0.9 * avail) p /= 2;
  if (p < wanted) {
    if (p < (warn_below > 0 ? warn_below : wanted))
      warn(c, "Krylov block size reduced from " + std::to_string(wanted) + " to " + std::to_string(p) +
                  " by the memory budget");
    else
      event(c, "block_size_fit", {{"wanted", static_cast<std::int64_t>(wanted)},
                                  {"used", static_cast<std::int64_t>(p)}, {"avail_mb", avail / 1e6}});
  }
  return p;
}

template <class T>
KrylovReservation krylov_reservation(const SolverConfig& cfg, ProblemType type, index_t n) {
  const index_t k =
      cfg.target.mode == TargetMode::Interval ? std::min<index_t>(n, 64) : cfg.target.k;
  const index_t indefinite = ghiep_reserve_columns<T>(cfg, n);
  const auto columns = [&](index_t p) {
    const index_t ncv = std::max(
        cfg.ncv > 0 ? cfg.ncv : std::max<index_t>(2 * (k + 16) + 1, k + 32),
        lanczos_basis_columns(n, 0, std::min(n, k), cfg.ncv, p));
    const index_t definite = ncv + 6 * k;
    return type == ProblemType::GeneralizedIndefinite ? indefinite
           : type == ProblemType::Auto                ? std::max(definite, indefinite)
                                                      : definite;
  };
  KrylovReservation r;
  r.block_size = reserved_block_size(cfg);
  r.columns = columns(r.block_size);
  return r;
}

#define HS_INSTANTIATE_DRIVER_BLOCKS(T)                                                                     \
  template index_t fit_block_size<T>(Ctx&, index_t, index_t, index_t, index_t, index_t, index_t);         \
  template KrylovReservation krylov_reservation<T>(const SolverConfig&, ProblemType, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DRIVER_BLOCKS)
#undef HS_INSTANTIATE_DRIVER_BLOCKS

}  // namespace hs::driver
