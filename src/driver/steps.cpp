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
  return seconds(Phase::Setup) + seconds(Phase::Factor) + seconds(Phase::Iterate) + seconds(Phase::Verify);
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
  for (index_t j = 0; j < m; ++j) {
    const index_t c = cols[j];
    out.lambda.push_back(src.lambda[c]);
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

// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2). The count must come from an exact
// (double, full-rank) factorization: the inertia of a single-precision or BLR factor is that of the
// rounded/compressed matrix, so then compute_inertia builds a separate double factorization. Each
// position is tried with null-pivot detection first (certified). A null pivot means an eigenvalue
// at s, or a cluster denser than the detection threshold around s; then (exact factor only) the
// count is taken WITHOUT detection: a backward-stable LDL^T gives the exact inertia of K + E with
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
index_t count_below(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double& s, double outward,
                    double step, bool* detection_off, bool s_is_gap, bool* near_eigenvalue) {
  PhaseScope phase(c.clock, Phase::Factor);
  const bool exact = fac.options().precision == FactorPrecision::Double && !fac.options().blr;
  if (exact) c.held = FactorHeld{};
  if (near_eigenvalue) *near_eigenvalue = false;
  for (int tries = 0; tries < 8; ++tries) {
    for (const bool detection : {true, false}) {
      if (!detection && !exact) break;
      if (!detection && tries == 0 && !s_is_gap) s += outward * 0.1 * step;
      Inertia in;
      if (exact) {
        fac.set_null_pivot_detection(detection);
        if (fac.factorize(1.0, -s)) in = fac.inertia();
      } else {
        FactorOptions o = fac.options();
        o.precision = FactorPrecision::Double;
        o.blr = false;
        in = compute_inertia(pencil, 1.0, -s, o);
      }
      if (in.valid) {
        if (!detection && c.rep.certificate.note.empty())
          c.rep.certificate.note = "endpoint inertia counted without null-pivot detection (dense cluster)";
        fac.set_null_pivot_detection(true);
        if (exact) c.held = FactorHeld{s, !detection};
        if (detection_off) *detection_off = !detection;
        return in.neg;
      }
      if (near_eigenvalue) *near_eigenvalue = true;
    }
    s += outward * step * std::pow(4.0, tries);
    ++c.rep.certificate.endpoint_moves;
  }
  fac.set_null_pivot_detection(true);
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

// Factorization / solve model of DESIGN §3.6-3b (calibrated on the 2x Xeon 6336Y node).
double model_factor_s(const FactorEstimate& e, double order, int threads) {
  return 6e-7 * order + e.flops / (12e9 * std::max(1, threads));
}
double model_solve_s(const FactorEstimate& e) { return 16.0 * static_cast<double>(e.factor_entries) / 12e9; }

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
// its analysis costs ~7x an AMD analysis; on 2-D-like graphs AMD's extra fill is cheap. Probe with
// AMD (fast), predict what nested dissection could save on the run's factorizations and solves,
// and pay for it only when the predicted saving exceeds its predicted analysis cost.
// Cost model (calibrated on the 2x Xeon 6336Y node, P3 measurements; only ratios matter):
//   t_factorize ~ 6e-7 n + flops / (12 GF/s * threads),  t_solve ~ 16 B * entries / 12 GB/s,
//   nested dissection keeps (flops, entries) >= (0.35, 0.6) of AMD's on 3-D-like graphs (AMD fill
//   ratio entries/nnz(K) > 20) and >= (0.6, 0.85) otherwise.
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
  const double t_amd = t.seconds();
  const FactorEstimate e = fac->estimate();
  const double threads = std::max(1, fopts.threads);
  const double n = static_cast<double>(pencil.order()) * (is_complex_v<T> ? 2.0 : 1.0);
  const double t_f = model_factor_s(e, n, static_cast<int>(threads));
  const double t_s = model_solve_s(e);
  const double fill_ratio = static_cast<double>(e.factor_entries) / std::max<double>(1.0, pencil.nnz());
  const bool three_d_like = fill_ratio > 20.0;
  const double keep_f = three_d_like ? 0.35 : 0.6, keep_e = three_d_like ? 0.6 : 0.85;
  const double saving = expected_factorizations * t_f * (1.0 - keep_f) + expected_solves * t_s * (1.0 - keep_e);
  const double nd_cost = 7.0 * t_amd;
  const bool use_nd = saving > nd_cost;
  event(c, "ordering", {{"amd_analyze_s", t_amd}, {"amd_flops", e.flops},
                        {"amd_entries", static_cast<std::int64_t>(e.factor_entries)}, {"fill_ratio", fill_ratio},
                        {"predicted_saving_s", saving}, {"predicted_nd_cost_s", nd_cost},
                        {"choice", std::string(use_nd ? "scotch" : "amd")}});
  if (!use_nd) return fac;
  FactorOptions nd = fopts;
  nd.ordering = 3;  // SCOTCH: measured faster analysis than METIS at equal or better total time
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
  template index_t count_below<T>(Ctx&, SparseFactorization<T>&, const ShiftedPencil<T>&, double&, double,  \
                                  double, bool*, bool, bool*);                                              \
  template void analyze_within_budget<T>(Ctx&, SparseFactorization<T>&, const ShiftedPencil<T>&, double, bool); \
  template std::unique_ptr<SparseFactorization<T>> adaptive_analysis<T>(                                    \
      Ctx&, const FactorOptions&, const ShiftedPencil<T>&, double, index_t, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DRIVER_STEPS)
#undef HS_INSTANTIATE_DRIVER_STEPS

}  // namespace hs::driver
