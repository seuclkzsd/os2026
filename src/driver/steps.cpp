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
// perturb the shift, delta_r = (+/-) 1e-8 * max(scale, |sigma0|) * 10^r, at most twice. If every
// nearby shift still looks singular, sigma0 lies inside a cluster denser than the null-pivot
// threshold: moving away would change which eigenvalues are nearest, so factorize AT sigma0 with
// null-pivot detection off (tiny pivots are kept; shift-invert then amplifies exactly the wanted
// near-null directions). `detection_off` tells later refactorizations at this shift to do the same.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off) {
  detection_off = false;
  fac.set_null_pivot_detection(true);
  double s = sigma0;
  const index_t perturbations = std::min<index_t>(2, c.cfg.max_shift_retries);
  for (index_t r = 0;; ++r) {
    if (fac.factorize(1.0, -s)) {
      c.rep.shift_retries = r;
      if (r > 0) warn(c, "shift perturbed to " + fmt(s) + " (A - sigma B singular at the requested shift)");
      return s;
    }
    if (r >= perturbations) break;
    const double mag = 1e-8 * std::max(scale, std::abs(sigma0)) * std::pow(10.0, static_cast<double>(r));
    s = sigma0 + ((r % 2 == 0) ? mag : -mag);
  }
  fac.set_null_pivot_detection(false);
  if (fac.factorize(1.0, -sigma0)) {
    detection_off = true;
    c.rep.shift_retries = perturbations + 1;
    warn(c, "shift lies inside an eigenvalue cluster denser than the null-pivot threshold; factorized at the "
            "requested shift without null-pivot detection");
    return sigma0;
  }
  fac.set_null_pivot_detection(true);
  HS_THROW(Singular, "A - sigma B is exactly singular at and near sigma = " << sigma0);
}

// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
template <class T>
void refactorize_at(SparseFactorization<T>& fac, double sigma, bool detection_off) {
  fac.set_null_pivot_detection(!detection_off);
  if (!fac.factorize(1.0, -sigma)) HS_THROW(Internal, "refactorization at the working shift " << sigma << " failed");
}

// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2). Endpoints are placed at spectral-gap
// midpoints. The count must come from an exact (double, full-rank) factorization: the inertia of a
// single-precision or BLR factor is that of the rounded/compressed matrix, so then compute_inertia
// builds a separate double factorization. The first two attempts use null-pivot detection (a null
// pivot means s sits on an eigenvalue: move it outward). Inside a cluster denser than the detection
// threshold every s is flagged, so later attempts count WITHOUT detection: a backward-stable LDL^T
// gives the exact inertia of K + E with ||E|| ~ eps*growth*||K||, i.e. it counts correctly every
// eigenvalue farther than that from s - and s is a gap midpoint. Returns the final s via `s`.
template <class T>
index_t count_below(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double& s, double outward,
                    double step) {
  const bool exact = fac.options().precision == FactorPrecision::Double && !fac.options().blr;
  for (int tries = 0; tries < 8; ++tries) {
    // At each position: detection on (certified), then - exact factor only - detection off at
    // the SAME position; move outward only if both fail (exactly singular / no exact factor).
    for (const bool detection : {true, false}) {
      if (!detection && !exact) break;
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
        return in.neg;
      }
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
  if (!already_analyzed) fac.analyze(pencil, 1.0, -sigma0);
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
  FactorOptions amd = fopts;
  amd.ordering = 0;
  auto fac = make_mumps_factorization<T>(amd);
  Timer t;
  fac->analyze(pencil, 1.0, -sigma0);
  const double t_amd = t.seconds();
  const FactorEstimate e = fac->estimate();
  const double threads = std::max(1, fopts.threads);
  const double n = static_cast<double>(pencil.order()) * (is_complex_v<T> ? 2.0 : 1.0);
  const double t_f = 6e-7 * n + e.flops / (12e9 * threads);
  const double t_s = 16.0 * static_cast<double>(e.factor_entries) / 12e9;
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
  template double factorize_near<T>(Ctx&, SparseFactorization<T>&, double, double, bool&);                  \
  template void refactorize_at<T>(SparseFactorization<T>&, double, bool);                                   \
  template index_t count_below<T>(Ctx&, SparseFactorization<T>&, const ShiftedPencil<T>&, double&, double,  \
                                  double);                                                                  \
  template void analyze_within_budget<T>(Ctx&, SparseFactorization<T>&, const ShiftedPencil<T>&, double, bool); \
  template std::unique_ptr<SparseFactorization<T>> adaptive_analysis<T>(                                    \
      Ctx&, const FactorOptions&, const ShiftedPencil<T>&, double, index_t, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DRIVER_STEPS)
#undef HS_INSTANTIATE_DRIVER_STEPS

}  // namespace hs::driver
