// The HEP / GHEP path (B Hermitian positive definite): shift-invert thick-restart Lanczos in the
// B inner product, Rayleigh-Ritz on the original pencil, residual rescue and the inertia
// certificate with deflated reruns (docs/DESIGN.md §3.6).
//
// Two drivers share the building blocks:
//   * nearest k (one shift at sigma, LargestMagnitude; certificate window pushed into gaps);
//   * windows - interval [lower, upper], smallest k (-inf, U], largest k [L, +inf) - solved by
//     spectrum slicing. Every slice end is an exact Sylvester count on the ONE analysis, so each
//     slice knows how many eigenvalues it holds; a run converges exactly that many, is certified on
//     its own (deflated reruns on a deficit), and the slices are merged with a rank-revealing
//     de-duplication at the cuts. Slice shifts: the centre (LargestMagnitude) for interior slices;
//     one-sided at a slice end (SmallestAlgebraic: the eigenvalues below the shift, or
//     LargestAlgebraic: above) when nothing lies beyond the other end (spectrum end) or the other
//     end borders the excluded zero band. One-sided runs at a count point reuse its factorization.
//     Smallest / largest locate the window end U (L) by inertia search: counts are exact for B HPD.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "driver/projection.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

namespace {

/// Pairs with a backward error above this are not counted as found (ghost / unconverged Ritz
/// pairs); between tol and this they count but are reported as not accepted.
constexpr double kTrustworthy = 1e-6;

// Residual correction x <- x - K^{-1}(A x - rho B x), rho = Rayleigh quotient, for the columns whose
// backward error exceeds tol (DESIGN §3.6-7). With an exact K this is inverse iteration; for any
// approximate K its fixed points are exact eigenpairs of (A, B).
template <class T>
index_t rescue_pairs(SparseFactorization<T>& fac, const CsrMatrix<T>& a, const CsrMatrix<T>* b, Block<T>& blk,
                     const std::vector<index_t>& cols, double n2a, double n2b, double tol) {
  const index_t n = blk.x.rows();
  index_t rescued = 0;
  std::vector<T> r(static_cast<std::size_t>(n));
  for (index_t j : cols) {
    for (int it = 0; it < 3; ++it) {
      T* x = blk.x.col(j);
      const T* ax = blk.ax.col(j);
      const T* bx = blk.bx.col(j);
      const double rho = std::real(linalg::dot(n, x, ax)) / std::real(linalg::dot(n, x, bx));
      for (index_t i = 0; i < n; ++i) r[i] = ax[i] - T(rho) * bx[i];
      fac.solve(r.data(), 1, n);
      for (index_t i = 0; i < n; ++i) x[i] -= r[i];
      // B-normalize and refresh the products of this column.
      DenseMatrix<T> xc(n, 1), axc, bxc;
      std::copy(x, x + n, xc.col(0));
      driver::apply_block(&a, xc, axc);
      driver::apply_block(b, xc, bxc);
      const double bn = std::sqrt(std::max(std::real(linalg::dot(n, xc.col(0), bxc.col(0))), 1e-300));
      for (index_t i = 0; i < n; ++i) {
        blk.x(i, j) = xc(i, 0) / bn;
        blk.ax(i, j) = axc(i, 0) / bn;
        blk.bx(i, j) = bxc(i, 0) / bn;
      }
      blk.lambda[j] = std::real(linalg::dot(n, blk.x.col(j), blk.ax.col(j)));
      const auto pm = driver::pair_metrics(blk.ax.col(j), blk.bx.col(j), blk.x.col(j), n, blk.lambda[j], n2a, n2b);
      if (pm.backward_error <= tol) break;
    }
    ++rescued;
  }
  return rescued;
}

template <class T>
struct DefiniteRun {
  Block<T> all;                    ///< every Ritz pair returned (after Rayleigh-Ritz), lambda ascending
  std::vector<double> ritz_hints;  ///< all Ritz values of the final Krylov bases mapped to lambda
  index_t restarts = 0;
  index_t applies = 0;
  bool converged = false;
};

const char* which_name(Which w) {
  switch (w) {
    case Which::LargestMagnitude: return "largest_magnitude";
    case Which::LargestAlgebraic: return "largest_algebraic";
    case Which::SmallestAlgebraic: return "smallest_algebraic";
  }
  return "unknown";
}

// One shift-invert Lanczos run at the shift the factorization currently holds, optionally deflated.
// theta = 1/(lambda - sigma): LargestMagnitude = nearest sigma, SmallestAlgebraic = the eigenvalues
// just below sigma, LargestAlgebraic = just above.
template <class T>
void lanczos_run(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma, Which which,
                 index_t nev, const DenseMatrix<T>* deflation, DefiniteRun<T>& out) {
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  SolveRefinement refine;
  if (fac.options().precision == FactorPrecision::Single || fac.options().blr) {
    refine.max_steps = 10;
    refine.rel_tol = c.cfg.tol * c.cfg.inner_tol_factor * 0.1;
  }
  ShiftInvertOperator<T> op(fac, pencil, sigma, refine);
  std::unique_ptr<CsrOperator<T>> b_op;
  if (b) b_op = std::make_unique<CsrOperator<T>>(*b);
  KrylovSchurOptions<T> ko;
  ko.nev = nev;
  ko.ncv = c.cfg.ncv;
  ko.max_restarts = c.cfg.max_restarts;
  ko.tol = c.cfg.tol * c.cfg.inner_tol_factor;
  ko.seed = c.cfg.seed + static_cast<std::uint64_t>(out.restarts);  // fresh start for reruns
  ko.deflation = deflation;
  HermitianEigResult<T> res;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    res = lanczos_krylov_schur<T>(op, b_op.get(), which, ko);
  }
  out.restarts += res.restarts;
  out.applies += res.op_applies;
  out.converged = res.converged;
  event(c, "lanczos", {{"sigma", sigma}, {"which", std::string(which_name(which))},
                       {"nev", static_cast<std::int64_t>(nev)}, {"nconv", static_cast<std::int64_t>(res.nconv)},
                       {"restarts", static_cast<std::int64_t>(res.restarts)},
                       {"op_applies", static_cast<std::int64_t>(res.op_applies)},
                       {"deflated", static_cast<std::int64_t>(deflation ? deflation->cols() : 0)}});
  for (double th : res.ritz_all)
    if (th != 0.0) out.ritz_hints.push_back(sigma + 1.0 / th);
  // Append the new vectors (all nev, converged first) to the candidate block.
  Block<T> blk;
  blk.x = std::move(res.vectors);
  for (double th : res.theta) blk.lambda.push_back(th != 0.0 ? sigma + 1.0 / th : kInf);
  recompute_products(a, b, blk);
  if (out.all.size() == 0) {
    out.all = std::move(blk);
  } else {
    out.all.lambda.insert(out.all.lambda.end(), blk.lambda.begin(), blk.lambda.end());
    append_columns(out.all.x, blk.x);
    append_columns(out.all.ax, blk.ax);
    append_columns(out.all.bx, blk.bx);
  }
  if (!driver::rayleigh_ritz_definite(out.all.x, out.all.ax, out.all.bx, out.all.lambda))
    warn(c, "Rayleigh-Ritz skipped: X^H B X not positive definite (degenerate subspace)");
}

/// Keeps the converged pairs of run.all (backward error <= 10 tol, finite): the deflation block of
/// a rerun. Unconverged Ritz vectors are not invariant and would cut away part of a missing
/// eigenvector; after Rayleigh-Ritz the kept ones are B-orthonormal, as deflation requires.
template <class T>
void keep_converged(DefiniteRun<T>& run, double n2a, double n2b, double tol) {
  const index_t n = run.all.x.rows();
  std::vector<index_t> conv;
  for (index_t j = 0; j < run.all.size(); ++j) {
    const auto pm =
        driver::pair_metrics(run.all.ax.col(j), run.all.bx.col(j), run.all.x.col(j), n, run.all.lambda[j], n2a, n2b);
    if (std::isfinite(run.all.lambda[j]) && pm.backward_error <= 10.0 * tol) conv.push_back(j);
  }
  run.all = take_columns(run.all, conv);
}

template <class T>
std::vector<PairMetrics> block_metrics(const Block<T>& blk, double n2a, double n2b) {
  std::vector<PairMetrics> m;
  const index_t n = blk.x.rows();
  for (index_t j = 0; j < blk.size(); ++j)
    m.push_back(driver::pair_metrics(blk.ax.col(j), blk.bx.col(j), blk.x.col(j), n, blk.lambda[j], n2a, n2b));
  return m;
}

/// Metrics of sel; pairs above tol get residual correction with the factorization the object
/// holds (that of the run that produced them), then a Rayleigh-Ritz over sel (lambda ascending
/// afterwards) and fresh metrics.
template <class T>
void verify_and_rescue(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, Block<T>& sel,
                       std::vector<PairMetrics>& metrics, double n2a, double n2b) {
  metrics = block_metrics(sel, n2a, n2b);
  std::vector<index_t> failing;
  for (index_t j = 0; j < sel.size(); ++j)
    if (metrics[j].backward_error > c.cfg.tol) failing.push_back(j);
  if (failing.empty() || !fac.inertia().valid) return;
  const index_t r = rescue_pairs(fac, pencil.a(), pencil.b(), sel, failing, n2a, n2b, c.cfg.tol);
  event(c, "rescue", {{"pairs", static_cast<std::int64_t>(r)}});
  if (!driver::rayleigh_ritz_definite(sel.x, sel.ax, sel.bx, sel.lambda))
    warn(c, "Rayleigh-Ritz after rescue skipped: degenerate subspace");
  metrics = block_metrics(sel, n2a, n2b);
}

// Indices of the pairs selected for the nearest target, in output order (nearest first).
std::vector<index_t> select_nearest(const std::vector<double>& lambda, double sigma, index_t k, double zero_tol,
                                    index_t& zero_dropped) {
  std::vector<index_t> idx;
  zero_dropped = 0;
  for (index_t i = 0; i < static_cast<index_t>(lambda.size()); ++i) {
    if (!std::isfinite(lambda[i])) continue;
    if (zero_tol >= 0.0 && std::abs(lambda[i]) <= zero_tol) {
      ++zero_dropped;
      continue;
    }
    idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(),
                   [&](index_t p, index_t q) { return std::abs(lambda[p] - sigma) < std::abs(lambda[q] - sigma); });
  if (static_cast<index_t>(idx.size()) > k) idx.resize(static_cast<std::size_t>(k));
  return idx;
}

// Closest hint strictly beyond `edge` in direction dir (+1 above, -1 below); NaN if none.
double nearest_beyond(const std::vector<double>& hints, double edge, double dir, double eps) {
  double best = std::numeric_limits<double>::quiet_NaN();
  for (double h : hints) {
    if (!std::isfinite(h) || dir * (h - edge) <= eps) continue;
    if (std::isnan(best) || dir * (h - best) < 0) best = h;
  }
  return best;
}

template <class T>
void add_pair_reports(SolveReport& rep, const std::vector<double>& lambda, const std::vector<PairMetrics>& metrics,
                      double tol, bool& all_ok) {
  for (std::size_t j = 0; j < metrics.size(); ++j) {
    PairReport p;
    p.lambda_re = lambda[j];
    p.abs_residual = metrics[j].abs_residual;
    p.rel_residual = metrics[j].rel_residual;
    p.backward_error = metrics[j].backward_error;
    p.accepted = metrics[j].backward_error <= tol;
    all_ok = all_ok && p.accepted;
    rep.pairs.push_back(p);
  }
}

// =============================================================================== nearest k

// First radius of the windowed nearest solve: distance of the k-th nearest hint (Ritz values of
// the plain run, converged or not) outside the band, or twice the farthest if there are fewer.
double radius_from_hints(std::vector<double> hints, double sigma, index_t k, double band) {
  std::vector<double> d;
  for (double h : hints)
    if (std::isfinite(h) && std::abs(h) > band) d.push_back(std::abs(h - sigma));
  std::sort(d.begin(), d.end());
  const double floor_r = std::max(std::abs(sigma) + 2.0 * band, 4.0 * band);
  if (static_cast<index_t>(d.size()) >= k) return std::max(d[static_cast<std::size_t>(k - 1)] * (1.0 + 1e-6), floor_r);
  return std::max(d.empty() ? 0.0 : 2.0 * d.back(), floor_r);
}

// Nearest k with one shift at sigma. With a zero band (zero_filter_tol) that reaches the selection
// or the certificate window, the band's cluster competes for the run's pairs (a null space of
// dimension d supplies copy after copy) and inertia counts in the window include it: then
// *band_radius receives a first radius for the windowed solve and the result is discarded.
template <class T>
Solution<T> solve_nearest(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a, double n2b,
                          double scale, double* band_radius) {
  const SolverConfig& cfg = c.cfg;
  SolveReport& rep = c.rep;
  const index_t n = pencil.order();
  const Target& t = cfg.target;
  const index_t k = t.k;
  if (k > n) HS_THROW(InvalidInput, "target.k = " << k << " exceeds the matrix order " << n);

  // Factorization at the shift (near-singular handling in factorize_near).
  bool detection_off = false;
  const double sigma = factorize_near(c, fac, t.sigma, scale, detection_off);
  rep.sigma_used = sigma;
  const FactorStats& fs = fac.stats();
  event(c, "factorization", {{"sigma", sigma}, {"factor_s", fs.factorize_s}, {"entries", static_cast<std::int64_t>(fs.factor_entries)},
                             {"mb", fs.factor_mb_used}, {"delayed_pivots", static_cast<std::int64_t>(fs.delayed_pivots)}});

  // Converge exactly the k wanted pairs (most wanted by |theta| = nearest to sigma). Asking for
  // extra pairs only costs iterations: completeness is the certificate's job, not a safety margin.
  const index_t nev = std::min<index_t>(n, k);
  DefiniteRun<T> run;
  lanczos_run(c, fac, pencil, sigma, Which::LargestMagnitude, nev, static_cast<const DenseMatrix<T>*>(nullptr), run);
  if (!run.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

  const double zero_tol = cfg.zero_filter_tol >= 0 ? cfg.zero_filter_tol * scale : -1.0;
  Block<T> sel;
  std::vector<PairMetrics> metrics;
  auto select_and_verify = [&]() {
    index_t zero_dropped = 0;
    sel = take_columns(run.all, select_nearest(run.all.lambda, t.sigma, k, zero_tol, zero_dropped));
    rep.zero_modes_filtered = zero_dropped;
    refactorize_at(c, fac, sigma, detection_off);  // rescue solves with K(sigma)
    verify_and_rescue(c, fac, pencil, sel, metrics, n2a, n2b);
    index_t zd = 0;  // restore the target order (Rayleigh-Ritz sorts ascending)
    const auto order = select_nearest(sel.lambda, t.sigma, sel.size(), -1.0, zd);
    std::vector<PairMetrics> m2;
    for (index_t j : order) m2.push_back(metrics[j]);
    sel = take_columns(sel, order);
    metrics = std::move(m2);
  };
  select_and_verify();
  auto hand_over = [&]() {
    std::vector<double> hints = run.ritz_hints;
    hints.insert(hints.end(), run.all.lambda.begin(), run.all.lambda.end());
    *band_radius = radius_from_hints(hints, t.sigma, k, zero_tol);
    rep.restarts = run.restarts;
    rep.op_applies = run.applies;
    return Solution<T>{};
  };
  if (zero_tol >= 0) {
    double r = 0.0;
    for (double l : sel.lambda) r = std::max(r, std::abs(l - t.sigma));
    if (rep.zero_modes_filtered > 0 || sel.size() < k || std::abs(t.sigma) - r <= zero_tol) return hand_over();
  }

  // Inertia certificate (DESIGN §3.6-8), with up to two deflated reruns on a count mismatch.
  const bool want_cert = cfg.inertia_check != Toggle::Off;
  InertiaCertificate& cert = rep.certificate;
  for (int round = 0; want_cert && round < 3; ++round) {
    if (sel.size() == 0) break;
    const double tiny = 1e-12 * std::max(scale, std::abs(t.sigma));
    std::vector<double> hints = run.ritz_hints;
    hints.insert(hints.end(), run.all.lambda.begin(), run.all.lambda.end());
    auto push_out = [&](double edge, double dir) {
      const double nb = nearest_beyond(hints, edge, dir, tiny);
      const double gap_default = std::max(1e-3 * std::abs(edge - t.sigma), 1e-8 * std::max(scale, 1e-300));
      return std::isnan(nb) ? edge + dir * gap_default : 0.5 * (edge + nb);
    };
    double r = 0.0;
    for (double l : sel.lambda) r = std::max(r, std::abs(l - t.sigma));
    double a_end = push_out(t.sigma - r, -1.0);
    double b_end = push_out(t.sigma + r, +1.0);
    if (zero_tol >= 0 && a_end <= zero_tol && b_end >= -zero_tol) return hand_over();  // window meets the band
    const double step = 1e-9 * std::max(scale, std::abs(t.sigma));
    const index_t nu_a = count_below(c, fac, pencil, a_end, -1.0, step);
    const index_t nu_b = count_below(c, fac, pencil, b_end, +1.0, step);
    cert.performed = true;
    cert.lower = a_end;
    cert.upper = b_end;
    cert.count_inertia = nu_b - nu_a;
    index_t found = 0;
    for (double l : run.all.lambda)
      if (std::isfinite(l) && l >= a_end && l <= b_end && !(zero_tol >= 0 && std::abs(l) <= zero_tol)) ++found;
    cert.count_found = found;
    cert.passed = cert.count_inertia == cert.count_found;  // the window stays clear of the zero band
    event(c, "certificate", {{"round", static_cast<std::int64_t>(round)}, {"lower", a_end}, {"upper", b_end},
                             {"count_inertia", static_cast<std::int64_t>(cert.count_inertia)},
                             {"count_found", static_cast<std::int64_t>(found)}, {"passed", cert.passed}});
    if (cert.passed || cert.count_inertia < cert.count_found || round == 2) break;
    const index_t missing = cert.count_inertia - cert.count_found;
    warn(c, "certificate: " + std::to_string(missing) + " eigenvalue(s) missing in the window; deflated rerun");
    keep_converged(run, n2a, n2b, cfg.tol);
    const index_t nev_rerun = std::min<index_t>(n - run.all.size(), missing + 2);
    if (nev_rerun < 1) break;
    refactorize_at(c, fac, sigma, detection_off);
    DenseMatrix<T> defl = run.all.x;
    lanczos_run(c, fac, pencil, sigma, Which::LargestMagnitude, nev_rerun, &defl, run);
    select_and_verify();
  }
  if (cert.performed && !cert.passed) {
    if (cert.count_inertia < cert.count_found)
      warn(c, "certificate: more computed eigenvalues than inertia allows (duplicates/ghosts) in [" + fmt(cert.lower) +
                  ", " + fmt(cert.upper) + "]");
    else
      warn(c, "certificate failed: inertia " + std::to_string(cert.count_inertia) + " vs found " +
                  std::to_string(cert.count_found));
  }
  if (!want_cert) cert.note = "disabled by configuration";

  Solution<T> sol;
  sol.eigenvalues = sel.lambda;
  sol.eigenvectors = std::move(sel.x);
  bool all_ok = static_cast<index_t>(sol.eigenvalues.size()) == k;
  if (!all_ok) warn(c, "only " + std::to_string(sol.eigenvalues.size()) + " of " + std::to_string(k) + " pairs found");
  add_pair_reports<T>(rep, sol.eigenvalues, metrics, cfg.tol, all_ok);
  rep.success = all_ok && (!cert.performed || cert.passed);
  rep.restarts = run.restarts;
  rep.op_applies = run.applies;
  return sol;
}

// ======================================================================== windows and slices

/// A point with its exact Sylvester count nu(s) = #{eigenvalues < s}; s = -inf (nu = 0) and
/// s = +inf (nu = n) stand for open window ends.
struct CountPoint {
  double s = 0.0;
  index_t nu = 0;
  bool detection_off = false;  ///< counted without null-pivot detection (dense cluster)
};

/// Part of a target window with exact counts at both ends.
struct Slice {
  CountPoint lo, hi;
  index_t count() const { return hi.nu - lo.nu; }
};

enum class Anchor { Centre, Lower, Upper };

/// Pairs accepted from all slices, eigenvalue order maintained per slice.
template <class T>
struct Pool {
  DenseMatrix<T> x;  ///< n x capacity, the first `used` columns hold the pairs
  index_t used = 0;
  std::vector<double> lambda;
  std::vector<PairMetrics> metrics;
  std::vector<index_t> slice;  ///< producing slice of each pair
};

template <class T>
class WindowSolver {
 public:
  WindowSolver(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a, double n2b,
               double scale, const SpectrumEstimate& est)
      : c_(c), pencil_(pencil), fac_(fac), n2a_(n2a), n2b_(n2b), scale_(scale), est_(est), n_(pencil.order()) {
    cost_ = make_cost_model(fac.estimate(), n_, fac.options().threads, is_complex_v<T>);
  }

  Solution<T> solve();
  /// Nearest target with a zero band: the k nearest sigma outside the band, from windows
  /// [sigma - r, sigma + r] minus the band; r0 (> 0) is the first radius tried.
  void set_nearest_radius(double r0) { nearest_r0_ = r0; }

 private:
  // ---- counts and cuts
  CountPoint count(double s, double outward);
  CountPoint minus_inf() const { return {-kInf, 0, false}; }
  CountPoint plus_inf() const { return {kInf, n_, false}; }
  double width_min(double x, double y) const {
    return 1e-8 * std::max({scale_, std::abs(x), std::abs(y)});
  }
  double expansion_step() const;
  CountPoint find_count(const CountPoint& a, const CountPoint& b, index_t g_lo, index_t g, index_t g_hi,
                        double start, double guess = std::numeric_limits<double>::quiet_NaN());
  std::vector<Slice> plan(const CountPoint& a, const CountPoint& b);
  Anchor anchor_of(const Slice& s) const;
  // ---- runs
  void run_slice(const Slice& s, index_t id);
  void reserve_pool(index_t pairs);
  void append_to_pool(const Block<T>& blk, const std::vector<PairMetrics>& m, index_t id);
  index_t deduplicate();
  bool in_band(double l) const { return band_ >= 0.0 && l > band_lo_ && l < band_hi_; }
  double factorize_anchor(const CountPoint& p, bool& detection_off);

  Ctx& c_;
  const ShiftedPencil<T>& pencil_;
  SparseFactorization<T>& fac_;
  double n2a_, n2b_, scale_;
  SpectrumEstimate est_;
  index_t n_;
  CostModel cost_;
  index_t c_target_ = 0, c_max_ = 0;
  double band_ = -1.0;                ///< zero band half-width (zero_filter_tol * scale), < 0: none
  double band_lo_ = 0.0, band_hi_ = 0.0;  ///< the band as counted: (band_lo_, band_hi_) is excluded
  std::vector<CountPoint> known_;     ///< every count evaluated, ascending in s
  std::vector<Slice> slices_;
  Pool<T> pool_;
  index_t restarts_ = 0, applies_ = 0;
  double nearest_r0_ = -1.0;
};

template <class T>
CountPoint WindowSolver<T>::count(double s, double outward) {
  for (const auto& p : known_)
    if (p.s == s) return p;
  const double step = 1e-10 * std::max(scale_, std::abs(s));
  bool off = false;
  const index_t nu = count_below(c_, fac_, pencil_, s, outward, step, &off);
  const CountPoint p{s, nu, off};
  auto it = std::lower_bound(known_.begin(), known_.end(), s, [](const CountPoint& q, double v) { return q.s < v; });
  if (it != known_.end() && it->s == s) {
    *it = p;
  } else {
    known_.insert(it, p);
  }
  event(c_, "count", {{"s", s}, {"nu", static_cast<std::int64_t>(nu)}});
  return p;
}

template <class T>
double WindowSolver<T>::expansion_step() const {
  double d = 1e-6 * std::max(scale_, 1e-300);
  if (est_.valid) {
    d = std::max({d, est_.lo2 - est_.lo1, est_.hi1 - est_.hi2, 1e-3 * (est_.hi1 - est_.lo1)});
  }
  return d;
}

// A point p in (a, b) with g_lo <= nu(p) <= g_hi, aiming at g (a.nu < g_lo <= g <= g_hi < b.nu).
// Brackets from the known counts are narrowed by count interpolation (regula falsi with the
// Illinois modification: the end retained twice in a row gets half the weight; steps kept 5% inside
// the bracket); open ends are found by geometric expansion from the spectrum estimates. If a cluster narrower than width_min() straddles the target count, the
// smallest known point above the cluster is returned (its count exceeds g_hi).
template <class T>
CountPoint WindowSolver<T>::find_count(const CountPoint& a, const CountPoint& b, index_t g_lo, index_t g,
                                       index_t g_hi, double start, double guess) {
  int same_side = 0, last_side = 0;
  double lft_weight = 1.0, rgt_weight = 1.0;  // Illinois weights of the retained ends
  index_t expansions = 0;
  for (int it = 0; it < 40; ++it) {
    CountPoint lft = a, rgt = b;
    const CountPoint* best = nullptr;
    for (const auto& p : known_) {
      if (!(p.s > a.s && p.s < b.s)) continue;
      if (p.nu >= g_lo && p.nu <= g_hi) {
        if (!best || std::abs(p.nu - g) < std::abs(best->nu - g)) best = &p;
      } else if (p.nu < g_lo) {
        lft = p;  // ascending: the last one is the largest
      } else if (rgt.s == b.s) {
        rgt = p;  // the first one above g_hi
      }
    }
    if (best) return *best;
    double s = 0.0;
    const bool lf = std::isfinite(lft.s), rf = std::isfinite(rgt.s);
    const double d = expansion_step() * std::pow(4.0, static_cast<double>(expansions));
    if (std::isfinite(guess) && guess > lft.s && guess < rgt.s) {
      s = guess;  // caller's estimate (local eigenvalue density), tried once
      guess = std::numeric_limits<double>::quiet_NaN();
    } else if (!lf && !rf) {
      s = start;
    } else if (!lf) {
      s = (est_.valid ? std::min(rgt.s, est_.lo1) : rgt.s) - d;
      ++expansions;
    } else if (!rf) {
      s = (est_.valid ? std::max(lft.s, est_.hi1) : lft.s) + d;
      ++expansions;
    } else {
      if (rgt.s - lft.s <= width_min(lft.s, rgt.s)) return rgt;  // unsplittable cluster
      const double fl = lft_weight * (static_cast<double>(lft.nu) - static_cast<double>(g));  // < 0
      const double fr = rgt_weight * (static_cast<double>(rgt.nu) - static_cast<double>(g));  // > 0
      const double frac = std::clamp(-fl / (fr - fl), 0.05, 0.95);
      s = lft.s + frac * (rgt.s - lft.s);
    }
    const CountPoint p = count(s, +1.0);
    const int side = p.nu < g_lo ? -1 : (p.nu > g_hi ? +1 : 0);
    same_side = (side != 0 && side == last_side) ? same_side + 1 : 1;
    last_side = side;
    // Illinois: a new left end twice in a row halves the weight of the retained right end, and
    // vice versa; a switch of sides resets both.
    if (same_side >= 2 && side < 0) rgt_weight *= 0.5;
    if (same_side >= 2 && side > 0) lft_weight *= 0.5;
    if (same_side == 1) lft_weight = rgt_weight = 1.0;
  }
  // Give up refining: the closest known point above the target.
  CountPoint rgt = b;
  for (const auto& p : known_)
    if (p.s > a.s && p.s < b.s && p.nu >= g_lo) {
      rgt = p;
      break;
    }
  if (!std::isfinite(rgt.s)) HS_THROW(NotConverged, "inertia search found no point with " << g << " eigenvalues below it");
  return rgt;
}

// Cuts window (a, b) into slices of at most c_max_ eigenvalues (aiming at c_target_), marching
// upward from the finite lower end (an open lower end first gets its own leftmost slice).
template <class T>
std::vector<Slice> WindowSolver<T>::plan(const CountPoint& a, const CountPoint& b) {
  std::vector<CountPoint> cuts{a};
  CountPoint cur = a;
  auto next_cut = [&](index_t remaining) {
    const index_t step = remaining <= 2 * c_target_ ? (remaining + 1) / 2 : c_target_;
    const index_t g_lo = cur.nu + std::max<index_t>(1, step / 2);
    const index_t g_hi = cur.nu + std::min(c_max_, remaining - 1);
    const double start = std::isfinite(cur.s) ? cur.s : (est_.valid ? est_.lo1 : b.s);
    // First guess from the local eigenvalue density: that of the previous slice, else the average
    // over what is left of the window (open ends: up to the estimated end of the spectrum).
    double guess = std::numeric_limits<double>::quiet_NaN();
    const double from = std::isfinite(cur.s) ? cur.s : (est_.valid ? est_.lo1 : kInf);
    const double to = std::isfinite(b.s) ? b.s : (est_.valid ? est_.hi1 : kInf);
    if (cuts.size() >= 2 && std::isfinite(cuts[cuts.size() - 2].s) && std::isfinite(cur.s) &&
        cur.nu > cuts[cuts.size() - 2].nu) {
      const CountPoint& prev = cuts[cuts.size() - 2];
      guess = cur.s + (cur.s - prev.s) * static_cast<double>(step) / static_cast<double>(cur.nu - prev.nu);
    } else if (std::isfinite(from) && std::isfinite(to) && to > from) {
      guess = from + (to - from) * static_cast<double>(step) / static_cast<double>(b.nu - cur.nu);
    }
    return find_count(cur, b, g_lo, cur.nu + step, std::max(g_lo, g_hi), start, guess);
  };
  while (b.nu - cur.nu > c_max_) {
    const CountPoint p = next_cut(b.nu - cur.nu);
    if (p.nu <= cur.nu || !(p.s < b.s)) break;
    cuts.push_back(p);
    cur = p;
  }
  cuts.push_back(b);
  std::vector<Slice> out;
  for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
    if (cuts[i + 1].nu > cuts[i].nu) out.push_back({cuts[i], cuts[i + 1]});
  return out;
}

// Shift placement (see the file comment): one-sided toward the side that holds nothing or borders
// the zero band, else the centre.
template <class T>
Anchor WindowSolver<T>::anchor_of(const Slice& s) const {
  if (!std::isfinite(s.hi.s)) return Anchor::Lower;  // windows have at least one finite end
  if (!std::isfinite(s.lo.s)) return Anchor::Upper;
  if (band_ >= 0 && s.hi.s == band_lo_) return Anchor::Lower;  // keep the shift away from the band
  if (band_ >= 0 && s.lo.s == band_hi_) return Anchor::Upper;
  if (s.lo.nu == 0) return Anchor::Upper;  // nothing below: a centre shift could be far from all
  if (s.hi.nu == n_) return Anchor::Lower;
  return Anchor::Centre;
}

template <class T>
void WindowSolver<T>::reserve_pool(index_t pairs) {
  pool_.x.resize(n_, std::max<index_t>(pairs, 1));
  pool_.used = 0;
}

template <class T>
void WindowSolver<T>::append_to_pool(const Block<T>& blk, const std::vector<PairMetrics>& m, index_t id) {
  for (index_t j = 0; j < blk.size(); ++j) {
    if (pool_.used == pool_.x.cols()) pool_.x.resize_cols(std::max<index_t>(2 * pool_.used, pool_.used + 16));
    std::copy(blk.x.col(j), blk.x.col(j) + n_, pool_.x.col(pool_.used));
    ++pool_.used;
    pool_.lambda.push_back(blk.lambda[j]);
    pool_.metrics.push_back(m[j]);
    pool_.slice.push_back(id);
  }
}

// One slice: shift, Krylov run for exactly its count, verification, certificate with deflated
// reruns. Pairs within an ambiguity margin delta outside the slice are kept too (an eigenvalue
// that close to a cut may be computed on the other side of it); deduplicate() resolves them.
template <class T>
void WindowSolver<T>::run_slice(const Slice& s, index_t id) {
  const SolverConfig& cfg = c_.cfg;
  const index_t count = s.count();
  const Anchor anchor = anchor_of(s);
  double sigma = 0.0;
  bool detection_off = false;
  Which which = Which::LargestMagnitude;
  switch (anchor) {
    case Anchor::Centre:
      sigma = factorize_near(c_, fac_, 0.5 * (s.lo.s + s.hi.s), scale_, detection_off);
      break;
    case Anchor::Upper:  // eigenvalues below the shift: theta < 0, most negative = nearest below
      sigma = factorize_anchor(s.hi, detection_off);
      which = Which::SmallestAlgebraic;
      break;
    case Anchor::Lower:  // eigenvalues above the shift: theta > 0, largest = nearest above
      sigma = factorize_anchor(s.lo, detection_off);
      which = Which::LargestAlgebraic;
      break;
  }
  SliceReport sr;
  sr.lower = s.lo.s;
  sr.upper = s.hi.s;
  sr.sigma = sigma;
  sr.which = which_name(which);
  sr.count_inertia = count;
  sr.nev = std::min(n_, count);
  DefiniteRun<T> run;
  lanczos_run(c_, fac_, pencil_, sigma, which, sr.nev, static_cast<const DenseMatrix<T>*>(nullptr), run);

  const double d_lo = std::isfinite(s.lo.s) ? 1e-9 * std::max(scale_, std::abs(s.lo.s)) : 0.0;
  const double d_hi = std::isfinite(s.hi.s) ? 1e-9 * std::max(scale_, std::abs(s.hi.s)) : 0.0;
  Block<T> sel;
  std::vector<PairMetrics> metrics;
  index_t inside = 0, near_in = 0, near_out = 0;
  auto select = [&]() {
    std::vector<index_t> idx;
    for (index_t j = 0; j < run.all.size(); ++j) {
      const double l = run.all.lambda[j];
      if (std::isfinite(l) && l >= s.lo.s - d_lo && l <= s.hi.s + d_hi && !in_band(l)) idx.push_back(j);
    }
    sel = take_columns(run.all, idx);
    refactorize_at(c_, fac_, sigma, detection_off);
    verify_and_rescue(c_, fac_, pencil_, sel, metrics, n2a_, n2b_);
    // Drop ghosts (Ritz pairs that are no eigenpairs at all) and count.
    std::vector<index_t> keep;
    inside = near_in = near_out = 0;
    for (index_t j = 0; j < sel.size(); ++j) {
      if (!(metrics[j].backward_error <= std::max(kTrustworthy, cfg.tol))) continue;  // ghost / NaN
      const double l = sel.lambda[j];
      if (l < s.lo.s - d_lo || l > s.hi.s + d_hi || in_band(l)) continue;  // moved away by the rescue
      keep.push_back(j);
      if (l >= s.lo.s && l <= s.hi.s) {
        ++inside;
        if (l < s.lo.s + d_lo || l > s.hi.s - d_hi) ++near_in;
      } else {
        ++near_out;
      }
    }
    std::vector<PairMetrics> mk;
    for (index_t j : keep) mk.push_back(metrics[j]);
    sel = take_columns(sel, keep);
    metrics = std::move(mk);
  };
  select();
  // Certificate of the slice: count in [lo, hi] up to the pairs within delta of its ends.
  for (int round = 0; round < 2; ++round) {
    if (count <= inside + near_out) break;
    const index_t missing = count - inside - near_out;
    keep_converged(run, n2a_, n2b_, cfg.tol);
    const index_t nev_rerun = std::min<index_t>(n_ - run.all.size(), missing + 2);
    if (nev_rerun < 1) break;
    event(c_, "slice_rerun", {{"slice", static_cast<std::int64_t>(id)}, {"missing", static_cast<std::int64_t>(missing)}});
    refactorize_at(c_, fac_, sigma, detection_off);
    DenseMatrix<T> defl = run.all.x;
    lanczos_run(c_, fac_, pencil_, sigma, which, nev_rerun, &defl, run);
    ++sr.reruns;
    select();
  }
  sr.count_found = inside;
  sr.passed = inside - near_in <= count && count <= inside + near_out;
  sr.restarts = run.restarts;
  sr.op_applies = run.applies;
  restarts_ += run.restarts;
  applies_ += run.applies;
  event(c_, "slice", {{"slice", static_cast<std::int64_t>(id)}, {"lower", s.lo.s}, {"upper", s.hi.s}, {"sigma", sigma},
                      {"which", std::string(sr.which)}, {"count", static_cast<std::int64_t>(count)},
                      {"found", static_cast<std::int64_t>(inside)}, {"passed", sr.passed},
                      {"restarts", static_cast<std::int64_t>(run.restarts)},
                      {"op_applies", static_cast<std::int64_t>(run.applies)}});
  if (!sr.passed)
    warn(c_, "slice [" + fmt(s.lo.s) + ", " + fmt(s.hi.s) + "]: inertia " + std::to_string(count) + " vs found " +
                 std::to_string(inside));
  c_.rep.slices.push_back(sr);
  append_to_pool(sel, metrics, id);
}

// Factorization at a counted slice end used as a one-sided shift: the count's own factorization
// when the object still holds it, else a refactorization with the same null-pivot setting; if that
// is flagged singular now (last-bit differences of the parallel factorization), the shift moves a
// little like any near-singular shift.
template <class T>
double WindowSolver<T>::factorize_anchor(const CountPoint& p, bool& detection_off) {
  detection_off = p.detection_off;
  if (try_refactorize_at(c_, fac_, p.s, p.detection_off)) return p.s;
  warn(c_, "one-sided shift " + fmt(p.s) + " refactorized as a near-singular shift");
  return factorize_near(c_, fac_, p.s, scale_, detection_off);
}

// Pairs from different slices with eigenvalues within 2 delta of each other may be the same
// eigenvector found twice (near a cut). Such a group is replaced by a B-orthonormal basis of its
// span (eigenvalues of the B-Gram matrix below 1e-4 of the largest: dependent copies) followed
// by Rayleigh-Ritz. Returns the number of pairs removed.
template <class T>
index_t WindowSolver<T>::deduplicate() {
  const index_t m = pool_.used;
  std::vector<index_t> order(static_cast<std::size_t>(m));
  std::iota(order.begin(), order.end(), index_t{0});
  std::stable_sort(order.begin(), order.end(), [&](index_t p, index_t q) { return pool_.lambda[p] < pool_.lambda[q]; });
  std::vector<char> drop(static_cast<std::size_t>(m), 0);
  index_t removed = 0;
  const CsrMatrix<T>& a = pencil_.a();
  const CsrMatrix<T>* b = pencil_.b();
  for (index_t i = 0; i < m;) {
    index_t j = i + 1;
    bool multi = false;  // members from more than one slice
    while (j < m) {
      const double l0 = pool_.lambda[order[j - 1]], l1 = pool_.lambda[order[j]];
      if (l1 - l0 > 2e-9 * std::max({scale_, std::abs(l0), std::abs(l1)})) break;
      for (index_t q = i; q < j && !multi; ++q) multi = pool_.slice[order[q]] != pool_.slice[order[j]];
      ++j;
    }
    if (multi) {
      const index_t g = j - i;
      Block<T> grp;
      grp.x.resize(n_, g);
      for (index_t q = 0; q < g; ++q) {
        std::copy(pool_.x.col(order[i + q]), pool_.x.col(order[i + q]) + n_, grp.x.col(q));
        grp.lambda.push_back(pool_.lambda[order[i + q]]);
      }
      recompute_products(a, b, grp);
      // B-Gram matrix and its eigen-decomposition (g is tiny).
      DenseMatrix<T> gm(g, g);
      linalg::gram(n_, g, grp.x.data(), n_, grp.bx.data(), n_, gm.data(), g);
      for (index_t q = 0; q < g; ++q)
        for (index_t r = q + 1; r < g; ++r) {
          const T h = 0.5 * (gm(r, q) + conj_if(gm(q, r)));
          gm(r, q) = h;
          gm(q, r) = conj_if(h);
        }
      std::vector<double> mu(static_cast<std::size_t>(g));
      lapack::heev('V', 'U', static_cast<lapack::fint>(g), gm.data(), static_cast<lapack::fint>(g), mu.data());
      std::vector<index_t> keep;
      for (index_t q = 0; q < g; ++q)
        if (mu[q] > 1e-4 * mu.back()) keep.push_back(q);
      const index_t r = static_cast<index_t>(keep.size());
      if (r < g) {
        // Y = V_keep diag(1/sqrt(mu)): X Y is B-orthonormal; Rayleigh-Ritz then sorts it out.
        DenseMatrix<T> y(g, r);
        for (index_t q = 0; q < r; ++q)
          for (index_t p = 0; p < g; ++p) y(p, q) = gm(p, keep[q]) / std::sqrt(mu[keep[q]]);
        Block<T> red;
        red.x.resize(n_, r);
        linalg::multiply(n_, g, r, grp.x.data(), n_, y.data(), g, red.x.data(), n_);
        recompute_products(a, b, red);
        red.lambda.assign(static_cast<std::size_t>(r), 0.0);
        if (!driver::rayleigh_ritz_definite(red.x, red.ax, red.bx, red.lambda))
          HS_THROW(Internal, "de-duplication: degenerate Gram matrix after rank reduction");
        const auto mr = block_metrics(red, n2a_, n2b_);
        for (index_t q = 0; q < g; ++q) {
          const index_t col = order[i + q];
          if (q < r) {
            std::copy(red.x.col(q), red.x.col(q) + n_, pool_.x.col(col));
            pool_.lambda[col] = red.lambda[q];
            pool_.metrics[col] = mr[q];
          } else {
            drop[col] = 1;
          }
        }
        removed += g - r;
      }
    }
    i = j;
  }
  if (removed > 0) {
    // Compact the pool (keeps the column order of the survivors).
    index_t w = 0;
    for (index_t q = 0; q < m; ++q) {
      if (drop[q]) continue;
      if (w != q) {
        std::copy(pool_.x.col(q), pool_.x.col(q) + n_, pool_.x.col(w));
        pool_.lambda[w] = pool_.lambda[q];
        pool_.metrics[w] = pool_.metrics[q];
        pool_.slice[w] = pool_.slice[q];
      }
      ++w;
    }
    pool_.used = w;
    pool_.lambda.resize(static_cast<std::size_t>(w));
    pool_.metrics.resize(static_cast<std::size_t>(w));
    pool_.slice.resize(static_cast<std::size_t>(w));
  }
  return removed;
}

/// Columns of x reordered in place so that new column j = old column perm[j] (perm a permutation
/// of 0..m-1), using one n-vector of workspace (cycle following).
template <class T>
void permute_columns(DenseMatrix<T>& x, const std::vector<index_t>& perm) {
  const index_t n = x.rows(), m = static_cast<index_t>(perm.size());
  std::vector<char> done(static_cast<std::size_t>(m), 0);
  std::vector<T> tmp(static_cast<std::size_t>(n));
  for (index_t start = 0; start < m; ++start) {
    if (done[start] || perm[start] == start) {
      done[start] = 1;
      continue;
    }
    std::copy(x.col(start), x.col(start) + n, tmp.data());
    index_t j = start;
    while (true) {
      done[j] = 1;
      const index_t src = perm[j];
      if (src == start) {
        std::copy(tmp.begin(), tmp.end(), x.col(j));
        break;
      }
      std::copy(x.col(src), x.col(src) + n, x.col(j));
      j = src;
    }
  }
}

template <class T>
Solution<T> WindowSolver<T>::solve() {
  const SolverConfig& cfg = c_.cfg;
  SolveReport& rep = c_.rep;
  const Target& t = cfg.target;
  const TargetMode mode = t.mode;
  band_ = cfg.zero_filter_tol >= 0 ? cfg.zero_filter_tol * scale_ : -1.0;
  band_lo_ = -band_;
  band_hi_ = band_;
  if (mode != TargetMode::Interval && t.k > n_)
    HS_THROW(InvalidInput, "target.k = " << t.k << " exceeds the matrix order " << n_);

  // Slice size: the cost model's optimum (or the user's), capped below by the memory budget.
  c_target_ = cfg.slice_size > 0 ? cfg.slice_size : cost_.best_slice();
  c_max_ = cfg.slice_size > 0 ? cfg.slice_size : c_target_ + c_target_ / 4;
  event(c_, "slicing_model", {{"t_factor_s", cost_.t_factor}, {"t_solve_s", cost_.t_solve},
                              {"slice_target", static_cast<std::int64_t>(c_target_)},
                              {"slice_max", static_cast<std::int64_t>(c_max_)}});

  // ---- 1. Target windows with exact counts at their ends. The zero band (|lambda| <= band) is
  //         cut out between two counted points, so filtering and counting agree exactly.
  std::vector<std::pair<CountPoint, CountPoint>> windows;
  const index_t k = t.k;
  auto band_edges = [&]() {
    const CountPoint neg = count(-band_, +1.0);  // moved into the band if singular
    const CountPoint pos = count(band_, -1.0);
    band_lo_ = neg.s;
    band_hi_ = pos.s;
    rep.zero_modes_filtered = pos.nu - neg.nu;
    return std::make_pair(neg, pos);
  };
  if (mode == TargetMode::Interval) {
    const CountPoint lo = count(t.lower, -1.0);
    const CountPoint hi = count(t.upper, +1.0);
    if (band_ >= 0 && lo.s < band_ && hi.s > -band_) {
      const auto [neg, pos] = band_edges();
      if (lo.s < neg.s) windows.push_back({lo, neg});
      if (hi.s > pos.s) windows.push_back({pos, hi});
      rep.zero_modes_filtered = std::min(pos.nu, hi.nu) - std::max(neg.nu, lo.nu);
    } else {
      windows.push_back({lo, hi});
    }
    rep.certificate.lower = lo.s;
    rep.certificate.upper = hi.s;
  } else if (mode == TargetMode::Nearest) {
    // The k nearest sigma outside the band lie in [sigma - r, sigma + r] once that disk holds k
    // eigenvalues outside the band: grow r from the hint until it does (two counts per radius).
    if (!(band_ >= 0) || !(nearest_r0_ > 0)) HS_THROW(Internal, "windowed nearest needs a zero band and a radius");
    const auto [neg, pos] = band_edges();
    auto outside = [&](const CountPoint& lo, const CountPoint& hi) {
      index_t tot = hi.nu - lo.nu;
      const CountPoint& bl = lo.s > neg.s ? lo : neg;
      const CountPoint& bh = hi.s < pos.s ? hi : pos;
      if (bh.s > bl.s) tot -= bh.nu - bl.nu;
      return tot;
    };
    double r = nearest_r0_;
    CountPoint lo{}, hi{};
    index_t tot = 0;
    for (int it = 0; it < 40; ++it) {
      lo = count(t.sigma - r, -1.0);
      hi = count(t.sigma + r, +1.0);
      tot = outside(lo, hi);
      if (tot >= k || (lo.nu == 0 && hi.nu == n_)) break;
      r *= 2.0;
    }
    if (lo.s < neg.s) windows.push_back({lo, hi.s < neg.s ? hi : neg});
    if (hi.s > pos.s) windows.push_back({lo.s > pos.s ? lo : pos, hi});
    if (windows.empty()) HS_THROW(Internal, "windowed nearest: the disk lies inside the zero band");
    rep.zero_modes_filtered = tot >= 0 ? (hi.nu - lo.nu) - tot : 0;
    rep.certificate.lower = lo.s;
    rep.certificate.upper = hi.s;
  } else {
    // Smallest: window (-inf, U] with nu(U) >= k; largest: [L, +inf) with n - nu(L) >= k. The
    // search stops refining once the extra pairs cost less than another count (slack).
    const bool smallest = mode == TargetMode::Smallest;
    const index_t slack = std::clamp<index_t>(
        static_cast<index_t>(cost_.t_factor / std::max(cost_.marginal_pair(std::min(k, c_target_)), 1e-300)), 1,
        std::max<index_t>(16, k));
    CountPoint a = minus_inf(), b = plus_inf();
    index_t need = k;
    if (band_ >= 0) {
      const auto [neg, pos] = band_edges();
      if (smallest) {
        if (neg.nu >= k) {
          b = neg;
        } else {  // all eigenvalues below the band, the rest from above it
          if (neg.nu > 0) windows.push_back({minus_inf(), neg});
          a = pos;
          need = k - neg.nu;
        }
      } else {
        const index_t above = n_ - pos.nu;
        if (above >= k) {
          a = pos;
        } else {
          if (above > 0) windows.push_back({pos, plus_inf()});
          b = neg;
          need = k - above;
        }
      }
      if (need > b.nu - a.nu) {
        warn(c_, "only " + std::to_string(b.nu - a.nu + k - need) + " eigenvalues lie outside the zero band");
        need = b.nu - a.nu;
      }
    }
    if (need > 0 && smallest) {
      const index_t g_lo = a.nu + need;
      const index_t g_hi = std::min(g_lo + slack, std::max(g_lo, b.nu - 1));
      CountPoint u = b;
      if (!(std::isfinite(b.s) && b.nu <= g_lo + slack))
        u = find_count(a, b, g_lo, std::min(g_lo + slack / 2, g_hi), g_hi,
                       std::isfinite(a.s) ? a.s : (est_.valid ? est_.lo1 : 0.0));
      windows.push_back({a, u});
    } else if (need > 0) {
      const index_t g_hi = b.nu - need;  // nu(L) <= n - need
      const index_t g_lo = std::max(g_hi - slack, std::min(g_hi, a.nu + 1));
      CountPoint l = a;
      if (!(std::isfinite(a.s) && a.nu >= g_hi - slack))
        l = find_count(a, b, g_lo, std::max(g_hi - slack / 2, g_lo), g_hi,
                       std::isfinite(b.s) ? b.s : (est_.valid ? est_.hi1 : 0.0));
      windows.push_back({l, b});
    }
    // A window from -inf up to the band (or from the band to +inf) would put a one-sided shift
    // next to the zero cluster; close it at a point beyond the end of the spectrum instead.
    for (auto& w : windows) {
      if (!std::isfinite(w.first.s) && w.second.s == band_lo_)
        w.first = find_count(minus_inf(), w.second, 0, 0, 0, est_.valid ? est_.lo1 : w.second.s);
      if (!std::isfinite(w.second.s) && w.first.s == band_hi_)
        w.second = find_count(w.first, plus_inf(), n_, n_, n_, est_.valid ? est_.hi1 : w.first.s);
    }
  }
  std::sort(windows.begin(), windows.end(), [](const auto& p, const auto& q) { return p.first.s < q.first.s; });
  index_t total = 0;
  for (const auto& w : windows) total += w.second.nu - w.first.nu;
  event(c_, "windows", {{"windows", static_cast<std::int64_t>(windows.size())}, {"count", static_cast<std::int64_t>(total)}});

  InertiaCertificate& cert = rep.certificate;
  cert.performed = true;
  if (mode == TargetMode::Smallest) {
    cert.lower = -kInf;
    cert.upper = windows.empty() ? -kInf : windows.back().second.s;
  } else if (mode == TargetMode::Largest) {
    cert.lower = windows.empty() ? kInf : windows.front().first.s;
    cert.upper = kInf;
  }
  cert.count_inertia = total;
  if (band_ >= 0)
    cert.note = "zero band (" + fmt(band_lo_) + ", " + fmt(band_hi_) + ") excluded: " +
                std::to_string(rep.zero_modes_filtered) + " eigenvalue(s) by inertia";
  if (rep.zero_modes_filtered > 0)
    warn(c_, std::to_string(rep.zero_modes_filtered) + " zero modes filtered (|lambda| <= zero_filter_tol*scale)");

  Solution<T> sol;
  const index_t expected = mode == TargetMode::Interval ? total : std::min(k, total);
  if (total == 0) {
    sol.eigenvectors.resize(n_, 0);
    cert.count_found = 0;
    cert.passed = true;
    if (cert.note.empty()) cert.note = "window is empty";
    rep.success = mode == TargetMode::Interval;
    return sol;
  }

  // ---- 2. Memory: the n x total output plus one slice's working set must fit the budget.
  {
    const double vec = static_cast<double>(n_) * sizeof(T);
    const double avail =
        cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
    const double out_bytes = vec * static_cast<double>(total + 16);
    if (out_bytes > 0.9 * avail)
      HS_THROW(OutOfMemoryBudget, "the target holds " << total << " eigenvalues: their eigenvectors need "
                                                      << out_bytes / 1073741824.0 << " GiB, "
                                                      << std::max(avail, 0.0) / 1073741824.0
                                                      << " GiB of the memory budget remain");
    // Working set of one slice of c pairs: Lanczos basis (~2c) + Rayleigh-Ritz blocks X, AX, BX
    // and a product temporary (4c) vectors, plus the selected block (c).
    const double per_pair = 7.0 * vec;
    const auto fit = static_cast<index_t>((0.9 * avail - out_bytes) / per_pair);
    if (fit < c_max_) {
      c_max_ = std::max<index_t>(1, fit);
      c_target_ = std::max<index_t>(1, std::min(c_target_, c_max_));
      warn(c_, "slice size capped at " + std::to_string(c_max_) + " pairs by the memory budget");
    }
    reserve_pool(total + 8);
  }

  // ---- 3. Slices, runs, merge.
  for (const auto& w : windows) {
    const auto s = plan(w.first, w.second);
    slices_.insert(slices_.end(), s.begin(), s.end());
  }
  event(c_, "slices", {{"slices", static_cast<std::int64_t>(slices_.size())}});
  for (std::size_t i = 0; i < slices_.size(); ++i) run_slice(slices_[i], static_cast<index_t>(i));
  const index_t removed = deduplicate();
  if (removed > 0) event(c_, "deduplicate", {{"removed", static_cast<std::int64_t>(removed)}});

  // ---- 4. Global certificate: distinct pairs inside the windows vs the inertia counts.
  auto inside_windows = [&](double l) {
    for (const auto& w : windows)
      if (l >= w.first.s && l <= w.second.s) return true;
    return false;
  };
  std::vector<index_t> in;
  for (index_t j = 0; j < pool_.used; ++j)
    if (inside_windows(pool_.lambda[j]) && !in_band(pool_.lambda[j])) in.push_back(j);
  cert.count_found = static_cast<index_t>(in.size());
  cert.passed = cert.count_found == cert.count_inertia;
  event(c_, "certificate", {{"lower", cert.lower}, {"upper", cert.upper},
                            {"count_inertia", static_cast<std::int64_t>(cert.count_inertia)},
                            {"count_found", static_cast<std::int64_t>(cert.count_found)}, {"passed", cert.passed}});
  if (!cert.passed)
    warn(c_, "certificate failed: inertia " + std::to_string(cert.count_inertia) + " vs found " +
                 std::to_string(cert.count_found) + " over " + std::to_string(slices_.size()) + " slice(s)");
  if (slices_.size() > 1) {
    const std::string sl = std::to_string(slices_.size()) + " slices";
    cert.note = cert.note.empty() ? sl : sl + "; " + cert.note;
  }

  // ---- 5. Output: target order and selection, applied to the pool in place.
  std::stable_sort(in.begin(), in.end(), [&](index_t p, index_t q) { return pool_.lambda[p] < pool_.lambda[q]; });
  if (mode == TargetMode::Largest) std::reverse(in.begin(), in.end());
  if (mode == TargetMode::Nearest)
    std::stable_sort(in.begin(), in.end(), [&](index_t p, index_t q) {
      return std::abs(pool_.lambda[p] - t.sigma) < std::abs(pool_.lambda[q] - t.sigma);
    });
  if (mode != TargetMode::Interval && static_cast<index_t>(in.size()) > k) in.resize(static_cast<std::size_t>(k));
  std::vector<index_t> perm = in;
  {
    std::vector<char> taken(static_cast<std::size_t>(pool_.used), 0);
    for (index_t j : in) taken[j] = 1;
    for (index_t j = 0; j < pool_.used; ++j)
      if (!taken[j]) perm.push_back(j);
  }
  pool_.x.resize_cols(pool_.used);
  permute_columns(pool_.x, perm);
  pool_.x.resize_cols(static_cast<index_t>(in.size()));
  sol.eigenvectors = std::move(pool_.x);
  std::vector<PairMetrics> metrics;
  for (index_t j : in) {
    sol.eigenvalues.push_back(pool_.lambda[j]);
    metrics.push_back(pool_.metrics[j]);
  }
  bool all_ok = static_cast<index_t>(sol.eigenvalues.size()) == expected && expected == (mode == TargetMode::Interval ? total : k);
  if (static_cast<index_t>(sol.eigenvalues.size()) != expected)
    warn(c_, "only " + std::to_string(sol.eigenvalues.size()) + " of " + std::to_string(expected) + " pairs found");
  add_pair_reports<T>(rep, sol.eigenvalues, metrics, cfg.tol, all_ok);
  rep.success = all_ok && cert.passed;
  rep.restarts = restarts_;
  rep.op_applies = applies_;
  if (!rep.slices.empty()) rep.sigma_used = rep.slices.front().sigma;
  return sol;
}

}  // namespace

template <class T>
Solution<T> solve_definite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                           double n2b, double scale, double sigma0, const SpectrumEstimate& spectrum) {
  c.rep.sigma_requested = sigma0;
  // Work not inside a Factor / Iterate scope (refinement, residuals, rescue, certificate
  // bookkeeping) is verification.
  PhaseScope phase(c.clock, Phase::Verify);
  if (c.cfg.target.mode == TargetMode::Nearest) {
    double r0 = -1.0;
    Solution<T> sol = solve_nearest(c, pencil, fac, n2a, n2b, scale, &r0);
    if (!(r0 > 0)) return sol;
    // The zero band interferes: k nearest outside it from windows [sigma - r, sigma + r] minus the
    // band (slices with shifts kept away from the band), r from inertia counts.
    warn(c, "zero band |lambda| <= " + fmt(c.cfg.zero_filter_tol * scale) +
                " reaches the nearest window: windowed solve with radius from inertia counts");
    const index_t restarts = c.rep.restarts, applies = c.rep.op_applies;
    c.rep.pairs.clear();
    c.rep.certificate = InertiaCertificate{};
    c.rep.zero_modes_filtered = 0;
    WindowSolver<T> ws(c, pencil, fac, n2a, n2b, scale, spectrum);
    ws.set_nearest_radius(r0);
    sol = ws.solve();
    c.rep.restarts += restarts;
    c.rep.op_applies += applies;
    return sol;
  }
  if (c.cfg.inertia_check == Toggle::Off)
    warn(c, "inertia_check=off ignored: interval / smallest / largest targets are built on inertia counts");
  WindowSolver<T> ws(c, pencil, fac, n2a, n2b, scale, spectrum);
  return ws.solve();
}

#define HS_INSTANTIATE_SOLVE_DEFINITE(T)                                                                  \
  template Solution<T> solve_definite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                         double, double, double, const SpectrumEstimate&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_DEFINITE)
#undef HS_INSTANTIATE_SOLVE_DEFINITE

}  // namespace hs::driver
