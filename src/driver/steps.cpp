// Driver building blocks shared by the solve paths: context helpers, pair blocks and the
// factorization steps (shift retries, inertia counts, analysis and memory budget, adaptive
// ordering). See internal.hpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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

// Factorize K(sigma0) (DESIGN §3.6-4). A null pivot means sigma0 is (numerically) ON an eigenvalue:
// perturb the shift once, by delta = 1e-8 * max(scale, |sigma0|) - an isolated eigenvalue at
// sigma0 is then ~1e3 times beyond the detection threshold (factor.hpp). If that shift still looks
// singular, sigma0 lies inside a cluster denser than the null-pivot threshold: moving away (a
// second, 10x larger perturbation used to be tried) would change which eigenvalues are nearest -
// on syn-zero-cluster-real-1e5 it landed outside the cluster at 5x the wanted radius and the
// certificate then counted 166 eigenvalues for 21 found - so factorize AT sigma0 with null-pivot
// detection off (tiny pivots are kept; shift-invert then amplifies exactly the wanted near-null
// directions). `detection_off` tells later refactorizations at this shift to do the same.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off,
                      bool on_eigenvalue) {
  PhaseScope phase(c.clock, Phase::Factor);
  detection_off = false;
  if (c.held.shift == sigma0 && !c.held.detection_off) return sigma0;  // e.g. right after counting there
  c.held = FactorHeld{};
  fac.set_null_pivot_detection(true);
  double s = sigma0;
  const index_t perturbations = std::min<index_t>(1, c.cfg.max_shift_retries);
  for (index_t r = 0;; ++r) {
    if (fac.factorize(1.0, -s)) {
      if (!on_eigenvalue) {
        c.rep.shift_retries = std::max(c.rep.shift_retries, r);
        if (r > 0) warn(c, "shift perturbed to " + fmt(s) + " (A - sigma B singular at the requested shift)");
      }
      c.held = FactorHeld{s, false};
      return s;
    }
    if (r >= perturbations) break;
    const double mag = 1e-8 * std::max(scale, std::abs(sigma0)) * std::pow(10.0, static_cast<double>(r));
    s = sigma0 + ((r % 2 == 0) ? mag : -mag);
  }
  fac.set_null_pivot_detection(false);
  if (fac.factorize(1.0, -sigma0)) {
    detection_off = true;
    if (!on_eigenvalue) {
      c.rep.shift_retries = std::max(c.rep.shift_retries, perturbations + 1);
      warn(c, "shift " + fmt(sigma0) + " lies inside an eigenvalue cluster denser than the null-pivot threshold; "
              "factorized there without null-pivot detection");
    }
    c.held = FactorHeld{sigma0, true};
    return sigma0;
  }
  fac.set_null_pivot_detection(true);
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

template <class T>
CountSource<T>::CountSource(SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, bool prefer_counter,
                            MainFactors main)
    : fac_(fac), pencil_(pencil), main_(main) {
  const bool exact = fac.options().precision == FactorPrecision::Double && !fac.options().blr;
  use_counter_ = prefer_counter || !exact;
}

template <class T>
CountSource<T>::~CountSource() = default;

template <class T>
Inertia CountSource<T>::count(Ctx& c, double s, bool detection) {
  PhaseScope phase(c.clock, Phase::Certificate);
  if (!use_counter_) {
    c.held = FactorHeld{};
    fac_.set_null_pivot_detection(detection);
    Inertia in;
    if (fac_.factorize(1.0, -s)) in = fac_.inertia();
    fac_.set_null_pivot_detection(true);
    if (in.valid) c.held = FactorHeld{s, !detection};
    return in;
  }
  const double cap = fac_.options().max_memory_mb;
  if (!counter_) {
    // Inertia-only instance on the main analysis' pivot order.
    counter_ = std::make_unique<InertiaCounter<T>>(fac_, fac_.options());
    event(c, "inertia_counter", {{"analyze_s", counter_->stats().analyze_s},
                                 {"incore_mb", counter_->estimate().incore_mb}, {"cap_mb", cap}});
  }
  // The counter's working memory comes on top of resident main factors. Decided before every
  // count: the main object may hold factors again since the last one (refactorized for a rerun,
  // a rescue, a slice shift).
  if (fac_.inertia().valid) {
    const double main_mb = fac_.stats().factor_mb_used;
    const bool over_cap = cap > 0 && main_mb + counter_->estimate().incore_mb > cap;
    if ((main_ == MainFactors::Release || over_cap) && fac_.release_factors()) {
      c.held = FactorHeld{};
      event(c, "main_factors_released", {{"main_mb", main_mb}, {"counter_mb", counter_->estimate().incore_mb},
                                         {"cap_mb", cap}, {"over_cap", over_cap}});
    }
  }
  return counter_->count(1.0, -s, detection);
}

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
template <class T>
void analyze_within_budget(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma0,
                           bool already_analyzed) {
  if (!already_analyzed) {
    PhaseScope phase(c.clock, Phase::Factor);
    fac.analyze(pencil, 1.0, -sigma0);
  }
  const FactorEstimate est = fac.estimate();
  event(c, "analysis", {{"factor_entries", static_cast<std::int64_t>(est.factor_entries)},
                        {"incore_mb", est.incore_mb}, {"ooc_mb", est.ooc_mb}, {"flops", est.flops}});
  const double cap_mb = fac.options().max_memory_mb;
  if (cap_mb > 0 && est.incore_mb > cap_mb) {
    if (c.cfg.out_of_core != Toggle::Off && est.ooc_mb <= cap_mb) {
      fac.set_out_of_core(true);
      warn(c, "factor estimate " + fmt(est.incore_mb) + " MB exceeds the budget; using out-of-core factors");
    } else {
      HS_THROW(OutOfMemoryBudget, "predicted factorization memory " << est.incore_mb << " MB (out-of-core "
                                                                    << est.ooc_mb << " MB) exceeds the budget of "
                                                                    << cap_mb << " MB");
    }
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
//   * HEP / GHEP nearest, block width p = block_size_for(k): column applications ~ 40 + 4k for
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

namespace {

// The ordering cost model, fitted on bench/anaint_orderings.cpp over the 40 data cases of MUMPS
// order >= 5000 (gpu02: 2x Xeon Silver 4314, cores 28-31, 4 threads; medians of 3 interleaved
// one-process runs per case and ordering; one run for the 1-thread time of hsolve ND, for AMD and
// PORD on ss-ge87h76 / ss-si34h36 and for ss-3dspectralwave2; nonnegative least squares on the
// relative error; the commands are in bench/anaint_orderings.cpp, the fit script accompanies the
// module report). Inputs: the MUMPS input size (mumps_input_size: order, upper entries), the graph
// hsolve's nested dissection orders (OrderingInput::nd) and the factor entries fe / flops of an
// analysis estimate. In brackets: the range of predicted / measured over the fitted cases.
//
// Given-order analysis (MUMPS analysis with PERM_IN: hsolve's nested dissection hands its order
// over, the InertiaCounter reuses the main one), the floor of every analysis [0.38, 1.55]:
constexpr double kFloorOrder = 2.1e-7, kFloorEntry = 4.14e-8, kFloorFe = 0.0;
// hsolve's nested dissection on one thread: W = w_v * vertices + w_e * edges + w_fe * fe of its
// graph [0.47, 1.38]; on p threads W (kNdSerial + (1 - kNdSerial) / p), kNdSerial the median over
// the measured 1- and 4-thread times [0.06, 0.35]. 2 threads scaled less than this predicts
// (median t2 / t4 = 1.28 against 1.52); beyond 4 threads it is an extrapolation.
constexpr double kNdWorkVertex = 7.03e-6, kNdWorkEdge = 6.54e-8, kNdWorkFe = 3.85e-9, kNdSerial = 0.186;
// MUMPS orderings are serial, on the full input graph: METIS, PORD and SCOTCH cost rho * W (W of
// the MUMPS input with their own fe; medians; whole analysis [0.5, 1.0], except the disconnected
// block graphs of the syn-*-real low-fill cases, which they order 3-8x faster - those take AMD
// anyway); AMD's ordering a_order * order + a_entry * entries [0.28, 1.40].
constexpr double kRhoMetis = 0.993, kRhoPord = 0.869, kRhoScotch = 1.017;
constexpr double kAmdOrder = 5.07e-8, kAmdEntry = 4.95e-9;

// Factorization flops and entries of an ordering relative to AMD's on the same input, as functions
// of AMD's fill ratio f = fe(AMD) / nnz(pencil): ln r piecewise linear in ln f between the knots
// f = 1, 4, 10, 40, 150, 1000 (constant beyond), least squares with a small ridge. Low fill (block
// or banded structure, f < 5): nested dissections need 2-3x AMD's flops; 2-D (f ~ 10): 0.4-0.9x;
// 3-D (f >= 40): 0.1-0.35x. Flops [0.44, 1.86], entries [0.68, 1.36].
constexpr double kShapeKnots[] = {1.0, 4.0, 10.0, 40.0, 150.0, 1000.0};
constexpr int kShapeKnotCount = 6;
struct ShapeFit {
  int ordering;
  double flops[kShapeKnotCount], entries[kShapeKnotCount];  ///< ln r at the knots
};
constexpr ShapeFit kShapeFits[] = {
    {kOrderingNestedDissection, {0.81, 1.00, -0.52, -1.03, -1.32, -1.90}, {0.41, 0.51, -0.18, -0.43, -0.63, -1.00}},
    {5, {0.89, 0.93, -0.58, -1.04, -1.34, -2.00}, {0.45, 0.48, -0.24, -0.45, -0.64, -1.04}},
    {4, {0.65, 0.83, -0.64, -0.82, -1.07, -1.14}, {0.34, 0.41, -0.24, -0.35, -0.53, -0.65}},
    {3, {0.88, 1.01, -0.49, -1.03, -1.43, -1.87}, {0.44, 0.52, -0.12, -0.39, -0.66, -0.98}},
};

// Numeric phases at 4 threads: factorization t_f = f_order * order + f_entry * entries + f_fe * fe +
// f_flops * flops [0.36, 1.40] (input assembly and scaling, front assembly, dense kernels); a
// refactorization of the same object kRefactor t_f, an InertiaCounter count kCount t_f (no factors
// kept); a 1-RHS solve t_s = s_order * order + s_fe * fe [0.36, 1.47], p right-hand sides in one
// call (1 + kSolvePerRhs (p - 1)) t_s (medians: 1.05 t_s for p = 2, 1.36 t_s for p = 4).
constexpr double kFacOrder = 4.1e-7, kFacEntry = 5.54e-8, kFacFe = 9.56e-9, kFacFlops = 6.22e-12;
constexpr double kRefactor = 0.906, kCount = 0.793;
constexpr double kSolveOrder = 1.08e-7, kSolveFe = 8.12e-10, kSolvePerRhs = 0.121;
// Other thread counts: (4 / threads)^gamma, the medians of log2(t(2) / t(4)) over 13 cases with AMD
// and hsolve ND at 2 threads (cores 28-29): factorizations 0.68 [0.40, 1.20], solves 0.58
// [-0.25, 1.36]. Extrapolated beyond 4 threads.
constexpr double kFacThreadGamma = 0.68, kSolveThreadGamma = 0.58;
constexpr double kFitThreads = 4.0;

// Expected gain of analysing a candidate rather than trusting its prediction, as a fraction of its
// predicted numeric time: from AMD's estimate the nested dissections' fill is known to within
// 0.5..1.9x; from another nested dissection's actual estimate within ~10% (METIS and SCOTCH had
// 0.82..1.10x hsolve's flops on the 3-D cases).
constexpr double kGainFromAmd = 0.05, kGainFromNd = 0.03;
/// Score of an analysed candidate whose in-core estimate exceeds the factorization cap, per unit of
/// estimate / cap: above every feasible score, ordered by memory.
constexpr double kOverCapScore = 1e30;

const ShapeFit* shape_fit(int ordering) {
  for (const auto& f : kShapeFits)
    if (f.ordering == ordering) return &f;
  return nullptr;
}

double thread_factor(int threads, double gamma) {
  return std::pow(kFitThreads / static_cast<double>(std::max(1, threads)), gamma);
}

double given_order_analysis_s(const MumpsInputSize& in, double fe) {
  return kFloorOrder * in.order + kFloorEntry * in.entries + kFloorFe * fe;
}

double nd_work_s(const MumpsInputSize& g, double fe) {
  return kNdWorkVertex * g.order + kNdWorkEdge * g.entries + kNdWorkFe * fe;
}

/// ln r of `ordering` at ln fill (piecewise linear between the knots, constant beyond them).
double shape_log_ratio(const double (&knots)[kShapeKnotCount], double fill) {
  const double lf = std::log(std::clamp(fill, kShapeKnots[0], kShapeKnots[kShapeKnotCount - 1]));
  for (int i = 0; i + 1 < kShapeKnotCount; ++i) {
    const double a = std::log(kShapeKnots[i]), b = std::log(kShapeKnots[i + 1]);
    if (lf <= b) return knots[i] + (knots[i + 1] - knots[i]) * (lf - a) / (b - a);
  }
  return knots[kShapeKnotCount - 1];
}

/// (flops, entries) of `ordering` relative to AMD's at AMD fill ratio f.
std::pair<double, double> shape_ratio(int ordering, double fill) {
  const ShapeFit* s = shape_fit(ordering);
  if (!s) return {1.0, 1.0};  // AMD itself (and orderings without a fit)
  return {std::exp(shape_log_ratio(s->flops, fill)), std::exp(shape_log_ratio(s->entries, fill))};
}

bool nested_dissection(int ordering) { return ordering != 0; }

}  // namespace

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

// Candidates of the adaptive ordering. SCOTCH is not among them: its orderings differ from
// analysis to analysis (nd_ordering), so a choice that compared its estimate with another
// candidate's could differ between identical runs.
std::vector<int> candidate_orderings(const SolverConfig& /*cfg*/) { return {0, kOrderingNestedDissection, 5, 4}; }

double ordering_analysis_s(int ordering, const OrderingInput& in, double factor_entries, int threads) {
  const double floor = given_order_analysis_s(in.mumps, factor_entries);
  switch (ordering) {
    case 0: return floor + kAmdOrder * in.mumps.order + kAmdEntry * in.mumps.entries;
    case kOrderingNestedDissection: {
      const double p = static_cast<double>(std::max(1, threads));
      return floor + nd_work_s(in.nd, factor_entries) * (kNdSerial + (1.0 - kNdSerial) / p);
    }
    case 4: return floor + kRhoPord * nd_work_s(in.mumps, factor_entries);
    case 3: return floor + kRhoScotch * nd_work_s(in.mumps, factor_entries);
    default: return floor + kRhoMetis * nd_work_s(in.mumps, factor_entries);  // METIS and the rest
  }
}

double factorization_s(const MumpsInputSize& in, double flops, double entries, int threads) {
  return (kFacOrder * in.order + kFacEntry * in.entries + kFacFe * entries + kFacFlops * flops) *
         thread_factor(threads, kFacThreadGamma);
}

double planned_numeric_s(const SolverConfig& cfg, const PlannedWork& w, const MumpsInputSize& in, double flops,
                         double entries, int threads) {
  const double t_f = factorization_s(in, flops, entries, threads);
  const double t_s = (kSolveOrder * in.order + kSolveFe * entries) * thread_factor(threads, kSolveThreadGamma);
  double s = t_f + std::max(0.0, w.factorizations - 1.0) * kRefactor * t_f;
  if (w.counts > 0) {
    // As the driver will count: on an InertiaCounter when counter_pays (its analysis on the given
    // order, then inertia-only factorizations), else refactorizations of the main object.
    FactorEstimate e;
    e.flops = flops;
    e.factor_entries = static_cast<std::int64_t>(entries);
    const auto counts = static_cast<index_t>(std::llround(w.counts));
    s += counter_pays(cfg, e, in, threads, counts) ? given_order_analysis_s(in, entries) + w.counts * kCount * t_f
                                                   : w.counts * kRefactor * t_f;
  }
  s += w.solve_calls * t_s * (1.0 + kSolvePerRhs * static_cast<double>(std::max<index_t>(w.block, 1) - 1));
  return s;
}

std::pair<double, double> predicted_shape(int ordering, const FactorEstimate& amd, double pencil_nnz,
                                          int anchor_ordering, const FactorEstimate& anchor) {
  const double fill = static_cast<double>(amd.factor_entries) / std::max(1.0, pencil_nnz);
  const auto r = shape_ratio(ordering, fill), ra = shape_ratio(anchor_ordering, fill);
  return {anchor.flops * r.first / ra.first, static_cast<double>(anchor.factor_entries) * r.second / ra.second};
}

std::size_t choose_ordering(const std::vector<OrderingCandidate>& candidates) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i)
    if (candidates[i].score_s < (1.0 - kOrderingMargin) * candidates[best].score_s) best = i;
  return best;
}

// Adaptive fill-reducing ordering (DESIGN §3.6-3b). The AMD probe (fast; its estimate tells 2-D-like
// from 3-D-like fill) prices every candidate on the planned work: AMD by its own estimate, the
// others by the shape fits. The best-scoring candidate is analysed and its actual estimate replaces
// the prediction (the remaining nested dissections are then predicted from it, more tightly); the
// loop ends when the best-scoring candidate is an analysed one. A second analysis therefore runs
// only where the numeric work dwarfs it (the expected gain of an actual estimate, kGainFrom*, is a
// fraction of the numeric time): on the largest 3-D cases, never on the 1e5 ones. Analysis objects
// that can no longer be chosen are freed at once (at most two live). Every input of the decisions
// is deterministic (estimates of deterministic orderings, models, thread count): identical runs
// choose the same ordering. Wall times are logged, never used.
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
  std::vector<std::unique_ptr<SparseFactorization<T>>> obj(orderings.size());
  std::vector<double> seconds(orderings.size(), 0.0);
  auto analyse = [&](std::size_t i) {
    FactorOptions o = fopts;
    o.ordering = orderings[i];
    obj[i] = make_mumps_factorization<T>(o);
    Timer t;
    obj[i]->analyze(pencil, 1.0, -sigma0);
    seconds[i] = t.seconds();
    const FactorEstimate e = obj[i]->estimate();
    cand[i].analysed = true;
    cand[i].analysis_s = 0.0;
    cand[i].flops = e.flops;
    cand[i].entries = static_cast<double>(e.factor_entries);
    cand[i].numeric_s = planned_numeric_s(c.cfg, work, in, cand[i].flops, cand[i].entries, threads);
    cand[i].score_s = cand[i].numeric_s;
    // Factors beyond the in-core cap (DESIGN §3.6-3) only run out of core, or not at all: any
    // candidate that fits is preferred, and among those that do not, the smallest.
    const double cap = fopts.max_memory_mb;
    if (cap > 0 && e.incore_mb > cap) cand[i].score_s = kOverCapScore * (e.incore_mb / cap);
  };
  for (std::size_t i = 0; i < orderings.size(); ++i) cand[i].ordering = orderings[i];
  HS_ASSERT(!orderings.empty() && orderings[0] == 0, "adaptive_analysis: AMD must be the first candidate");
  analyse(0);  // the AMD probe
  const FactorEstimate amd_estimate = obj[0]->estimate();
  const double fill = cand[0].entries / std::max(1.0, nnz);
  // Re-price the candidates not analysed yet from the analysed candidate `anchor`.
  auto predict = [&](std::size_t anchor) {
    FactorEstimate a;
    a.flops = cand[anchor].flops;
    a.factor_entries = static_cast<std::int64_t>(cand[anchor].entries);
    for (std::size_t i = 0; i < cand.size(); ++i) {
      if (cand[i].analysed) continue;
      const auto [flops, entries] = predicted_shape(orderings[i], amd_estimate, nnz, orderings[anchor], a);
      cand[i].flops = flops;
      cand[i].entries = entries;
      cand[i].analysis_s = ordering_analysis_s(orderings[i], oin, entries, threads);
      cand[i].numeric_s = planned_numeric_s(c.cfg, work, in, flops, entries, threads);
      const double gain = nested_dissection(orderings[anchor]) ? kGainFromNd : kGainFromAmd;
      cand[i].score_s = cand[i].analysis_s + (1.0 - gain) * cand[i].numeric_s;
    }
  };
  // The choice among the analysed candidates; their other analyses are freed.
  auto keep_best_analysed = [&]() {
    std::vector<OrderingCandidate> done;
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < cand.size(); ++i)
      if (cand[i].analysed) done.push_back(cand[i]), idx.push_back(i);
    const std::size_t keep = idx[choose_ordering(done)];
    for (std::size_t i : idx)
      if (i != keep) obj[i].reset();
    return keep;
  };
  predict(0);
  // Analyse the best-scoring candidate while it is not analysed yet: then no further analysis is
  // expected to pay, and the choice is among the analysed ones.
  for (std::size_t next = choose_ordering(cand); !cand[next].analysed; next = choose_ordering(cand)) {
    analyse(next);
    keep_best_analysed();
    predict(next);
  }
  const std::size_t pick = keep_best_analysed();
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

// Auto block width, from end-to-end runs of p = 1, 4, 8 (ncv auto) on the 30 HEP/GHEP data cases
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

template <class T>
index_t fit_block_size(Ctx& c, index_t n, index_t nev, index_t d, index_t extra_columns, index_t p) {
  const double vec = static_cast<double>(n) * sizeof(T);
  const double avail =
      c.cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
  const auto bytes = [&](index_t q) {
    return vec * static_cast<double>(lanczos_basis_columns(n, d, nev, c.cfg.ncv, q) + extra_columns);
  };
  const index_t wanted = std::max<index_t>(p, 1);
  p = wanted;
  while (p > 1 && bytes(p) > 0.9 * avail) p /= 2;
  if (p < wanted)
    warn(c, "Krylov block size reduced from " + std::to_string(wanted) + " to " + std::to_string(p) +
                " by the memory budget");
  return p;
}

#define HS_INSTANTIATE_DRIVER_BLOCKS(T)                                                                     \
  template index_t fit_block_size<T>(Ctx&, index_t, index_t, index_t, index_t, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DRIVER_BLOCKS)
#undef HS_INSTANTIATE_DRIVER_BLOCKS

}  // namespace hs::driver
