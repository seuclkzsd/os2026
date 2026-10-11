// Driver building blocks shared by the solve paths: context helpers, pair blocks and the
// factorization steps (shift retries, inertia counts, analysis and memory budget, adaptive
// ordering). See internal.hpp.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "driver/projection.hpp"
#include "eigen/block_lanczos.hpp"
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

// Adaptive fill-reducing ordering (DESIGN §3.6-3b). Nested dissection (SCOTCH) minimises fill but
// its analysis costs ~kNdAnalysisOverAmd AMD analyses; on 2-D-like graphs AMD's extra fill is
// cheap. Probe with AMD (fast), predict what nested dissection could save on the run's
// factorizations and solves, and pay for it only when the predicted saving exceeds its predicted
// analysis cost. Every input of the decision is deterministic (the analysis estimate and
// model_analysis_s, not the probe's wall time), so identical runs choose the same ordering even
// under load.
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

template <class T>
std::unique_ptr<SparseFactorization<T>> adaptive_analysis(Ctx& c, const FactorOptions& fopts,
                                                          const ShiftedPencil<T>& pencil, double sigma0,
                                                          index_t expected_factorizations, index_t expected_solves) {
  PhaseScope phase(c.clock, Phase::Factor);
  FactorOptions amd = fopts;
  amd.ordering = 0;
  auto fac = make_mumps_factorization<T>(amd);
  Timer t;
  fac->analyze(pencil, 1.0, -sigma0);
  const double t_amd = t.seconds();  // logged only: decisions must not depend on the load
  const FactorEstimate e = fac->estimate();
  const OrderingDecision d = decide_ordering(e, mumps_input_size(pencil), static_cast<double>(pencil.nnz()),
                                             fopts.threads, expected_factorizations, expected_solves);
  event(c, "ordering", {{"amd_analyze_s", t_amd}, {"amd_analyze_model_s", d.amd_analysis_s},
                        {"amd_flops", e.flops}, {"amd_entries", static_cast<std::int64_t>(e.factor_entries)},
                        {"fill_ratio", d.fill_ratio}, {"predicted_saving_s", d.saving_s},
                        {"predicted_nd_cost_s", d.nd_cost_s},
                        {"choice", std::string(!d.use_nd ? "amd" : nd_ordering(c.cfg) == 5 ? "metis" : "scotch")}});
  if (!d.use_nd) return fac;
  fac.reset();  // free the AMD analysis before the nested-dissection one: lower peak of the phase
  FactorOptions nd = fopts;
  nd.ordering = nd_ordering(c.cfg);
  auto fac_nd = make_mumps_factorization<T>(nd);
  fac_nd->analyze(pencil, 1.0, -sigma0);
  return fac_nd;
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
  template std::unique_ptr<SparseFactorization<T>> adaptive_analysis<T>(                                    \
      Ctx&, const FactorOptions&, const ShiftedPencil<T>&, double, index_t, index_t);
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
