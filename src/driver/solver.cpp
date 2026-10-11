// hs::solve - the driver (docs/DESIGN.md §3.6). Orchestrates diagnosis, factorization with shift
// retries, the Krylov solvers, refinement, residual verification, the inertia certificate and the
// report. All numerics live in the lower layers; this file only decides what to call and checks.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "driver/norms.hpp"
#include "driver/projection.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ------------------------------------------------------------------------------ small helpers

struct Ctx {
  const SolverConfig& cfg;
  Logger* log;
  SolveReport& rep;
};

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

// Exactly Hermitian working copy of an input matrix (DESIGN §3.6-1).
template <class T>
const CsrMatrix<T>* hermitian_input(const CsrMatrix<T>& m, double tol, const char* name, double& defect,
                                    CsrMatrix<T>& store) {
  m.validate();
  if (!m.square()) HS_THROW(InvalidInput, name << " is not square (" << m.nrows << "x" << m.ncols << ")");
  defect = hermitian_defect(m);
  if (defect > tol)
    HS_THROW(NotHermitian, name << " is not Hermitian: ||M-M^H||_F/||M||_F = " << defect << " > hermitian_tol " << tol);
  if (defect > 0.0) {
    store = hermitian_part(m);
    return &store;
  }
  return &m;
}

// Pairs being refined: eigenvalues with their vectors and A*X, B*X kept in sync.
template <class T>
struct Block {
  std::vector<double> lambda;
  DenseMatrix<T> x, ax, bx;

  index_t size() const { return static_cast<index_t>(lambda.size()); }
};

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
  o.ordering = cfg.ordering;
  o.threads = cfg.threads > 0 ? cfg.threads : max_threads();
  o.blr = cfg.blr == Toggle::On;
  o.out_of_core = cfg.out_of_core == Toggle::On;
  o.max_memory_mb = cap_mb > 0 ? cap_mb : 0.0;
  return o;
}

// ------------------------------------------------------------------------- factorization steps

// Factorize K(sigma0), perturbing the shift when K is numerically singular (DESIGN §3.6-4):
// delta_r = (+/-) 1e-8 * max(scale, |sigma0|) * 10^(r/2). Returns the shift actually used.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale) {
  double s = sigma0;
  for (index_t r = 0;; ++r) {
    if (fac.factorize(1.0, -s)) {
      c.rep.shift_retries = r;
      if (r > 0) warn(c, "shift perturbed to " + fmt(s) + " (A - sigma B singular at the requested shift)");
      return s;
    }
    if (r >= c.cfg.max_shift_retries)
      HS_THROW(Singular, "A - sigma B is numerically singular near sigma = " << sigma0 << " after " << r
                                                                             << " perturbations");
    const double mag = 1e-8 * std::max(scale, std::abs(sigma0)) * std::pow(10.0, static_cast<double>(r / 2));
    s = sigma0 + ((r % 2 == 0) ? mag : -mag);
  }
}

// nu_-(A - s B): number of eigenvalues below s. If K(s) is singular the endpoint is moved in
// direction `outward` (+1 / -1) by a growing step; moves are counted. Returns the final s.
template <class T>
index_t count_below(Ctx& c, SparseFactorization<T>& fac, double& s, double outward, double step) {
  for (int tries = 0; tries < 8; ++tries) {
    if (fac.factorize(1.0, -s)) {
      const Inertia in = fac.inertia();
      if (in.valid) return in.neg;
    }
    s += outward * step * std::pow(4.0, tries);
    ++c.rep.certificate.endpoint_moves;
  }
  HS_THROW(Singular, "cannot find a nonsingular certificate endpoint near " << s);
}

// --------------------------------------------------------------------------- definite path

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

// One shift-invert Lanczos run at the shift the factorization currently holds, optionally deflated.
template <class T>
void lanczos_run(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma, index_t nev,
                 const DenseMatrix<T>* deflation, DefiniteRun<T>& out) {
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
  auto res = lanczos_krylov_schur<T>(op, b_op.get(), Which::LargestMagnitude, ko);
  out.restarts += res.restarts;
  out.applies += res.op_applies;
  out.converged = res.converged;
  event(c, "lanczos", {{"nev", static_cast<std::int64_t>(nev)}, {"nconv", static_cast<std::int64_t>(res.nconv)},
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

// Indices of the pairs selected for the target, in output order.
std::vector<index_t> select_pairs(const std::vector<double>& lambda, const Target& t, index_t k, double zero_tol,
                                  index_t& zero_dropped) {
  std::vector<index_t> idx;
  zero_dropped = 0;
  for (index_t i = 0; i < static_cast<index_t>(lambda.size()); ++i) {
    if (!std::isfinite(lambda[i])) continue;
    if (zero_tol >= 0.0 && std::abs(lambda[i]) <= zero_tol) {
      ++zero_dropped;
      continue;
    }
    if (t.mode == TargetMode::Interval && (lambda[i] < t.lower || lambda[i] > t.upper)) continue;
    idx.push_back(i);
  }
  switch (t.mode) {
    case TargetMode::Nearest:
      std::stable_sort(idx.begin(), idx.end(), [&](index_t p, index_t q) {
        return std::abs(lambda[p] - t.sigma) < std::abs(lambda[q] - t.sigma);
      });
      break;
    case TargetMode::Largest:
      std::stable_sort(idx.begin(), idx.end(), [&](index_t p, index_t q) { return lambda[p] > lambda[q]; });
      break;
    case TargetMode::Smallest:
    case TargetMode::Interval:
      std::stable_sort(idx.begin(), idx.end(), [&](index_t p, index_t q) { return lambda[p] < lambda[q]; });
      break;
  }
  if (t.mode != TargetMode::Interval && static_cast<index_t>(idx.size()) > k) idx.resize(static_cast<std::size_t>(k));
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
Solution<T> solve_definite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                           double n2b, double scale, const driver::NormEstimate& na) {
  const SolverConfig& cfg = c.cfg;
  SolveReport& rep = c.rep;
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  const index_t n = pencil.order();
  Target t = cfg.target;
  Timer timer;

  // Shift for the target (Smallest/Largest: beyond the estimated end of the spectrum, HEP only).
  double sigma0 = t.sigma;
  if (t.mode == TargetMode::Interval) sigma0 = 0.5 * (t.lower + t.upper);
  if (t.mode == TargetMode::Smallest || t.mode == TargetMode::Largest) {
    if (b) HS_THROW(Unsupported, "smallest/largest targets are implemented for standard problems only (roadmap P3)");
    const double spread = std::max(na.ritz_max - na.ritz_min, 1e-12 * std::max(scale, 1.0));
    sigma0 = t.mode == TargetMode::Smallest ? na.ritz_min - 0.05 * spread : na.ritz_max + 0.05 * spread;
  }
  rep.sigma_requested = sigma0;

  fac.analyze(pencil, 1.0, -sigma0);
  const FactorEstimate est = fac.estimate();
  event(c, "analysis", {{"factor_entries", static_cast<std::int64_t>(est.factor_entries)},
                        {"incore_mb", est.incore_mb}, {"ooc_mb", est.ooc_mb}, {"flops", est.flops}});
  const double cap_mb = fac.options().max_memory_mb;
  if (cap_mb > 0 && est.incore_mb > cap_mb) {
    if (cfg.out_of_core != Toggle::Off && est.ooc_mb <= cap_mb) {
      fac.set_out_of_core(true);
      warn(c, "factor estimate " + fmt(est.incore_mb) + " MB exceeds the budget; using out-of-core factors");
    } else {
      HS_THROW(OutOfMemoryBudget, "predicted factorization memory " << est.incore_mb << " MB (out-of-core "
                                                                    << est.ooc_mb << " MB) exceeds the budget of "
                                                                    << cap_mb << " MB");
    }
  }

  // Interval: count the eigenvalues first (Sylvester inertia) to size the Krylov run.
  index_t k = t.k;
  index_t interval_count = -1;
  double lo = t.lower, hi = t.upper;
  if (t.mode == TargetMode::Interval) {
    const double step = 1e-10 * std::max(scale, std::max(std::abs(lo), std::abs(hi)));
    const index_t nu_lo = count_below(c, fac, lo, -1.0, step);
    const index_t nu_hi = count_below(c, fac, hi, +1.0, step);
    interval_count = nu_hi - nu_lo;
    event(c, "interval_count", {{"lower", lo}, {"upper", hi}, {"count", static_cast<std::int64_t>(interval_count)}});
    if (interval_count == 0) {
      Solution<T> sol;
      sol.eigenvectors.resize(n, 0);
      rep.certificate = {true, lo, hi, 0, 0, rep.certificate.endpoint_moves, true, "interval is empty"};
      rep.success = true;
      return sol;
    }
    if (interval_count > std::max<index_t>(2000, n / 2))
      HS_THROW(Unsupported, "interval holds " << interval_count << " eigenvalues; spectrum slicing is roadmap P3");
    k = interval_count;
  }
  if (k > n) HS_THROW(InvalidInput, "target.k = " << k << " exceeds the matrix order " << n);

  // Factorization at the shift (with near-singular perturbation).
  const double sigma = factorize_near(c, fac, sigma0, scale);
  rep.sigma_used = sigma;
  const FactorStats& fs = fac.stats();
  event(c, "factorization", {{"sigma", sigma}, {"factor_s", fs.factorize_s}, {"entries", static_cast<std::int64_t>(fs.factor_entries)},
                             {"mb", fs.factor_mb_used}, {"delayed_pivots", static_cast<std::int64_t>(fs.delayed_pivots)}});

  // Krylov run (+ a margin of extra pairs so that the k-th nearest is reliably among them).
  const index_t extra = std::min<index_t>(16, std::max<index_t>(2, k / 4));
  const index_t nev = std::min<index_t>(n, k + extra);
  DefiniteRun<T> run;
  Timer it_timer;
  lanczos_run(c, fac, pencil, sigma, nev, static_cast<const DenseMatrix<T>*>(nullptr), run);
  double t_iter = it_timer.seconds();
  if (!run.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

  Timer verify_timer;
  const double zero_tol = cfg.zero_filter_tol >= 0 ? cfg.zero_filter_tol * scale : -1.0;
  Target sel_target = t;
  sel_target.sigma = t.mode == TargetMode::Interval ? sigma0 : t.sigma;
  if (t.mode == TargetMode::Smallest || t.mode == TargetMode::Largest) sel_target.sigma = sigma0;
  if (t.mode == TargetMode::Interval) {
    sel_target.lower = lo;
    sel_target.upper = hi;
  }

  Block<T> sel;
  std::vector<driver::PairMetrics> metrics;
  auto select_and_verify = [&]() {
    index_t zero_dropped = 0;
    const auto idx = select_pairs(run.all.lambda, sel_target, k, zero_tol, zero_dropped);
    if (zero_dropped > 0) warn(c, std::to_string(zero_dropped) + " zero modes filtered (|lambda| <= zero_filter_tol*scale)");
    sel = take_columns(run.all, idx);
    metrics.clear();
    std::vector<index_t> failing;
    for (index_t j = 0; j < sel.size(); ++j) {
      metrics.push_back(driver::pair_metrics(sel.ax.col(j), sel.bx.col(j), sel.x.col(j), n, sel.lambda[j], n2a, n2b));
      if (metrics.back().backward_error > cfg.tol) failing.push_back(j);
    }
    if (!failing.empty() && fac.inertia().valid) {
      const index_t r = rescue_pairs(fac, a, b, sel, failing, n2a, n2b, cfg.tol);
      event(c, "rescue", {{"pairs", static_cast<std::int64_t>(r)}});
      if (!driver::rayleigh_ritz_definite(sel.x, sel.ax, sel.bx, sel.lambda))
        warn(c, "Rayleigh-Ritz after rescue skipped: degenerate subspace");
      // Restore the target order after Rayleigh-Ritz (it sorts ascending).
      index_t zd = 0;
      const auto order = select_pairs(sel.lambda, sel_target, sel.size(), -1.0, zd);
      sel = take_columns(sel, order);
      metrics.clear();
      for (index_t j = 0; j < sel.size(); ++j)
        metrics.push_back(driver::pair_metrics(sel.ax.col(j), sel.bx.col(j), sel.x.col(j), n, sel.lambda[j], n2a, n2b));
    }
  };
  select_and_verify();

  // Inertia certificate (DESIGN §3.6-8), with up to two deflated reruns on a count mismatch.
  const bool want_cert = cfg.inertia_check != Toggle::Off;
  InertiaCertificate& cert = rep.certificate;
  for (int round = 0; want_cert && round < 3; ++round) {
    if (sel.size() == 0) break;
    const double tiny = 1e-12 * std::max(scale, std::abs(sigma0));
    double a_end = -kInf, b_end = kInf;
    std::vector<double> hints = run.ritz_hints;
    hints.insert(hints.end(), run.all.lambda.begin(), run.all.lambda.end());
    auto push_out = [&](double edge, double dir) {
      const double nb = nearest_beyond(hints, edge, dir, tiny);
      const double gap_default = std::max(1e-3 * std::abs(edge - sel_target.sigma), 1e-8 * std::max(scale, 1e-300));
      return std::isnan(nb) ? edge + dir * gap_default : 0.5 * (edge + nb);
    };
    double lo_edge = *std::min_element(sel.lambda.begin(), sel.lambda.end());
    double hi_edge = *std::max_element(sel.lambda.begin(), sel.lambda.end());
    switch (t.mode) {
      case TargetMode::Nearest: {
        double r = 0.0;
        for (double l : sel.lambda) r = std::max(r, std::abs(l - t.sigma));
        a_end = push_out(t.sigma - r, -1.0);
        b_end = push_out(t.sigma + r, +1.0);
        break;
      }
      case TargetMode::Smallest: b_end = push_out(hi_edge, +1.0); break;
      case TargetMode::Largest: a_end = push_out(lo_edge, -1.0); break;
      case TargetMode::Interval: a_end = lo; b_end = hi; break;
    }
    const double step = 1e-9 * std::max(scale, std::abs(sigma0));
    index_t nu_a = 0, nu_b = n;
    if (t.mode == TargetMode::Interval) {
      nu_a = 0;
      nu_b = interval_count;  // counted before the Krylov run
    } else {
      if (std::isfinite(a_end)) nu_a = count_below(c, fac, a_end, -1.0, step);
      if (std::isfinite(b_end)) nu_b = count_below(c, fac, b_end, +1.0, step);
    }
    cert.performed = true;
    cert.lower = a_end;
    cert.upper = b_end;
    cert.count_inertia = nu_b - nu_a;
    index_t found = 0;
    for (double l : run.all.lambda)
      if (std::isfinite(l) && l >= a_end && l <= b_end && !(zero_tol >= 0 && std::abs(l) <= zero_tol)) ++found;
    cert.count_found = found;
    cert.passed = cert.count_inertia == cert.count_found ||
                  (zero_tol >= 0 && cert.count_inertia >= cert.count_found);  // filtered zeros are counted by inertia
    event(c, "certificate", {{"round", static_cast<std::int64_t>(round)}, {"lower", a_end}, {"upper", b_end},
                             {"count_inertia", static_cast<std::int64_t>(cert.count_inertia)},
                             {"count_found", static_cast<std::int64_t>(found)}, {"passed", cert.passed}});
    if (cert.passed || cert.count_inertia < cert.count_found || round == 2) break;
    // Missing eigenvalues: deflated rerun. Only CONVERGED pairs may be deflated (an unconverged Ritz
    // vector is not invariant and would cut away part of a missing eigenvector). After the
    // Rayleigh-Ritz step they are B-orthonormal, as the deflation contract requires.
    const index_t missing = cert.count_inertia - cert.count_found;
    warn(c, "certificate: " + std::to_string(missing) + " eigenvalue(s) missing in the window; deflated rerun");
    std::vector<index_t> conv;
    for (index_t j = 0; j < run.all.size(); ++j) {
      const auto pm = driver::pair_metrics(run.all.ax.col(j), run.all.bx.col(j), run.all.x.col(j), n,
                                           run.all.lambda[j], n2a, n2b);
      if (std::isfinite(run.all.lambda[j]) && pm.backward_error <= 10.0 * cfg.tol) conv.push_back(j);
    }
    run.all = take_columns(run.all, conv);
    const index_t nev_rerun = std::min<index_t>(n - run.all.size(), missing + 2);
    if (nev_rerun < 1) break;
    if (!fac.factorize(1.0, -sigma)) HS_THROW(Internal, "refactorization at the working shift failed");
    DenseMatrix<T> defl = run.all.x;
    it_timer.reset();
    lanczos_run(c, fac, pencil, sigma, nev_rerun, &defl, run);
    t_iter += it_timer.seconds();
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

  // Assemble the solution.
  Solution<T> sol;
  sol.eigenvalues = sel.lambda;
  sol.eigenvectors = std::move(sel.x);
  bool all_ok = static_cast<index_t>(sol.eigenvalues.size()) == k;
  if (!all_ok) warn(c, "only " + std::to_string(sol.eigenvalues.size()) + " of " + std::to_string(k) + " pairs found");
  for (std::size_t j = 0; j < metrics.size(); ++j) {
    PairReport p;
    p.lambda_re = sol.eigenvalues[j];
    p.abs_residual = metrics[j].abs_residual;
    p.rel_residual = metrics[j].rel_residual;
    p.backward_error = metrics[j].backward_error;
    p.accepted = metrics[j].backward_error <= cfg.tol;
    all_ok = all_ok && p.accepted;
    rep.pairs.push_back(p);
  }
  rep.success = all_ok && (!cert.performed || cert.passed);
  rep.restarts = run.restarts;
  rep.op_applies = run.applies;
  rep.time_iterate_s = t_iter;
  rep.time_verify_s = verify_timer.seconds() - t_iter;
  return sol;
}

// ------------------------------------------------------------------------- indefinite path

template <class T>
Solution<T> solve_indefinite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                             double n2b, double scale) {
  const SolverConfig& cfg = c.cfg;
  SolveReport& rep = c.rep;
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  const index_t n = pencil.order();
  const Target& t = cfg.target;
  if (t.mode != TargetMode::Nearest)
    HS_THROW(Unsupported, "indefinite-B problems support target mode 'nearest' only (roadmap P3)");
  if (t.k > n) HS_THROW(InvalidInput, "target.k = " << t.k << " exceeds the matrix order " << n);
  rep.sigma_requested = t.sigma;
  rep.certificate.note = "not available for indefinite B (Sylvester inertia does not count eigenvalues)";

  fac.analyze(pencil, 1.0, -t.sigma);
  const FactorEstimate est = fac.estimate();
  const double cap_mb = fac.options().max_memory_mb;
  if (cap_mb > 0 && est.incore_mb > cap_mb) {
    if (cfg.out_of_core != Toggle::Off && est.ooc_mb <= cap_mb) {
      fac.set_out_of_core(true);
      warn(c, "factor estimate exceeds the budget; using out-of-core factors");
    } else {
      HS_THROW(OutOfMemoryBudget, "predicted factorization memory " << est.incore_mb << " MB exceeds the budget of "
                                                                    << cap_mb << " MB");
    }
  }
  const double sigma = factorize_near(c, fac, t.sigma, scale);
  rep.sigma_used = sigma;

  SolveRefinement refine;
  if (fac.options().precision == FactorPrecision::Single || fac.options().blr) {
    refine.max_steps = 10;
    refine.rel_tol = cfg.tol * cfg.inner_tol_factor * 0.1;
  }
  ShiftInvertOperator<T> op(fac, pencil, sigma, refine);
  const index_t extra = std::min<index_t>(16, std::max<index_t>(2, t.k / 4));
  KrylovSchurOptions<T> ko;
  ko.nev = std::min<index_t>(n, t.k + extra);
  ko.ncv = cfg.ncv;
  ko.max_restarts = cfg.max_restarts;
  ko.tol = cfg.tol * cfg.inner_tol_factor;
  ko.seed = cfg.seed;
  Timer it_timer;
  auto res = krylov_schur_general<T>(op, Which::LargestMagnitude, ko);
  rep.time_iterate_s = it_timer.seconds();
  rep.restarts = res.restarts;
  rep.op_applies = res.op_applies;
  event(c, "krylov_schur", {{"nev", static_cast<std::int64_t>(ko.nev)}, {"nconv", static_cast<std::int64_t>(res.nconv)},
                            {"restarts", static_cast<std::int64_t>(res.restarts)}});
  if (!res.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

  Timer verify_timer;
  // Real basis of the computed invariant subspace: Re/Im parts of every vector (for real T a real
  // eigenvalue's vector is real up to a phase; a near-real conjugate pair - e.g. the zero cluster -
  // spans the same space as its real and imaginary parts). Dependent columns are dropped.
  const index_t m = static_cast<index_t>(res.theta.size());
  DenseMatrix<T> w;
  if constexpr (is_complex_v<T>) {
    w = res.vectors;
  } else {
    w.resize(n, 2 * m);
    for (index_t j = 0; j < m; ++j)
      for (index_t i = 0; i < n; ++i) {
        w(i, 2 * j) = res.vectors(i, j).real();
        w(i, 2 * j + 1) = res.vectors(i, j).imag();
      }
  }
  driver::orthonormalize_columns(w, 1e-8);
  DenseMatrix<T> aw, bw;
  driver::apply_block(&a, w, aw);
  driver::apply_block(b, w, bw);
  std::vector<cplx> mu;
  DenseMatrix<cplx> xc;
  if (!driver::rayleigh_ritz_indefinite(w, aw, bw, mu, xc)) {
    warn(c, "projected pencil singular; using the Krylov-Schur pairs without refinement");
    mu.clear();
    for (cplx th : res.theta) mu.push_back(cplx(sigma) + 1.0 / th);
    xc = res.vectors;
  }

  // Classify each pair real/complex (DESIGN §3.6-10) and compute its metrics.
  struct Cand {
    cplx lambda;
    bool real;
    driver::PairMetrics pm;
    index_t col;
  };
  std::vector<Cand> cands;
  DenseMatrix<T> xr(n, static_cast<index_t>(mu.size()));
  DenseMatrix<cplx> col(n, 1), acol, bcol;
  for (index_t j = 0; j < static_cast<index_t>(mu.size()); ++j) {
    if (!std::isfinite(std::abs(mu[j]))) continue;
    std::copy(xc.col(j), xc.col(j) + n, col.col(0));
    // Complex metrics need complex products; for real T apply A, B to real and imaginary parts.
    DenseMatrix<T> re(n, 1), im(n, 1), are, aim, bre, bim;
    for (index_t i = 0; i < n; ++i) {
      if constexpr (is_complex_v<T>) {
        re(i, 0) = col(i, 0);
      } else {
        re(i, 0) = col(i, 0).real();
        im(i, 0) = col(i, 0).imag();
      }
    }
    driver::apply_block(&a, re, are);
    driver::apply_block(b, re, bre);
    acol.resize(n, 1);
    bcol.resize(n, 1);
    if constexpr (is_complex_v<T>) {
      for (index_t i = 0; i < n; ++i) {
        acol(i, 0) = are(i, 0);
        bcol(i, 0) = bre(i, 0);
      }
    } else {
      driver::apply_block(&a, im, aim);
      driver::apply_block(b, im, bim);
      for (index_t i = 0; i < n; ++i) {
        acol(i, 0) = cplx(are(i, 0), aim(i, 0));
        bcol(i, 0) = cplx(bre(i, 0), bim(i, 0));
      }
    }
    const auto pm_c = driver::pair_metrics(acol.col(0), bcol.col(0), col.col(0), n, mu[j], n2a, n2b);
    const bool small_imag = std::abs(mu[j].imag()) <= cfg.complex_tol * std::max(std::abs(mu[j]), scale);
    Cand cd{mu[j], false, pm_c, j};
    // Real candidate: phase-normalised vector, its real part (real T) and lambda = Re(mu).
    index_t imax = 0;
    for (index_t i = 1; i < n; ++i)
      if (std::abs(col(i, 0)) > std::abs(col(imax, 0))) imax = i;
    const cplx phase = std::abs(col(imax, 0)) > 0 ? std::conj(col(imax, 0)) / std::abs(col(imax, 0)) : cplx(1);
    DenseMatrix<T> xreal(n, 1), axr, bxr;
    for (index_t i = 0; i < n; ++i) {
      if constexpr (is_complex_v<T>) {
        xreal(i, 0) = col(i, 0) * phase;
      } else {
        xreal(i, 0) = (col(i, 0) * phase).real();
      }
    }
    driver::apply_block(&a, xreal, axr);
    driver::apply_block(b, xreal, bxr);
    const double lam_r = mu[j].real();
    const auto pm_r = driver::pair_metrics(axr.col(0), bxr.col(0), xreal.col(0), n, lam_r, n2a, n2b);
    // Real unless the imaginary part is significant AND the real projection fails while the complex pair passes.
    const bool complex_genuine = !small_imag && pm_r.backward_error > cfg.tol && pm_c.backward_error <= pm_r.backward_error;
    if (!complex_genuine) {
      cd.real = true;
      cd.lambda = cplx(lam_r, 0.0);
      cd.pm = pm_r;
      const double nrm = linalg::norm2(n, xreal.col(0));
      for (index_t i = 0; i < n; ++i) xr(i, j) = xreal(i, 0) / nrm;
    }
    cands.push_back(cd);
  }
  // Nearest k to sigma (complex distance). The projected pencil also yields pairs built from the
  // unconverged Krylov directions; they have large residuals and must not crowd out true pairs
  // near sigma, so pairs with backward error <= kTrustworthy are preferred and the others only fill
  // up a shortfall (reported as not accepted).
  constexpr double kTrustworthy = 1e-6;
  auto nearer = [&](const Cand& p, const Cand& q) {
    return std::abs(p.lambda - cplx(t.sigma)) < std::abs(q.lambda - cplx(t.sigma));
  };
  std::vector<Cand> good, poor;
  for (const auto& cd : cands) (cd.pm.backward_error <= std::max(kTrustworthy, cfg.tol) ? good : poor).push_back(cd);
  std::stable_sort(good.begin(), good.end(), nearer);
  std::stable_sort(poor.begin(), poor.end(), nearer);
  cands = good;
  for (const auto& cd : poor) {
    if (cands.size() >= static_cast<std::size_t>(t.k)) break;
    cands.push_back(cd);
  }
  if (cands.size() > static_cast<std::size_t>(t.k)) cands.resize(static_cast<std::size_t>(t.k));
  std::stable_sort(cands.begin(), cands.end(), nearer);

  Solution<T> sol;
  index_t nreal = 0, ncomplex = 0;
  for (const auto& cd : cands) (cd.real ? nreal : ncomplex)++;
  sol.eigenvectors.resize(n, nreal);
  sol.complex_eigenvectors.resize(n, ncomplex);
  index_t jr = 0, jc = 0;
  bool all_ok = static_cast<index_t>(cands.size()) == t.k;
  std::vector<PairReport> complex_reports;
  for (const auto& cd : cands) {
    PairReport p;
    p.lambda_re = cd.lambda.real();
    p.lambda_im = cd.real ? 0.0 : cd.lambda.imag();
    p.abs_residual = cd.pm.abs_residual;
    p.rel_residual = cd.pm.rel_residual;
    p.backward_error = cd.pm.backward_error;
    p.accepted = cd.pm.backward_error <= cfg.tol;
    all_ok = all_ok && p.accepted;
    if (cd.real) {
      sol.eigenvalues.push_back(cd.lambda.real());
      std::copy(xr.col(cd.col), xr.col(cd.col) + n, sol.eigenvectors.col(jr++));
      rep.pairs.push_back(p);
    } else {
      sol.complex_eigenvalues.push_back(cd.lambda);
      std::copy(xc.col(cd.col), xc.col(cd.col) + n, sol.complex_eigenvectors.col(jc++));
      complex_reports.push_back(p);
    }
  }
  rep.pairs.insert(rep.pairs.end(), complex_reports.begin(), complex_reports.end());
  if (ncomplex > 0) warn(c, std::to_string(ncomplex) + " genuinely complex eigenvalue(s) returned separately");
  if (!all_ok && static_cast<index_t>(cands.size()) < t.k)
    warn(c, "only " + std::to_string(cands.size()) + " of " + std::to_string(t.k) + " pairs found");
  rep.success = all_ok;
  rep.time_verify_s = verify_timer.seconds();
  return sol;
}

}  // namespace

// ------------------------------------------------------------------------------------- solve

template <class T>
Solution<T> solve(const CsrMatrix<T>& a_in, const CsrMatrix<T>* b_in, const SolverConfig& cfg, Logger* log) {
  Timer total;
  SolveReport rep;
  Ctx c{cfg, log, rep};
#ifdef _OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#endif

  // 1. Diagnosis: exact Hermitian inputs, norms, scale.
  CsrMatrix<T> a_store, b_store;
  const CsrMatrix<T>* a = hermitian_input(a_in, cfg.hermitian_tol, "A", rep.hermitian_defect_a, a_store);
  const CsrMatrix<T>* b = nullptr;
  if (b_in) {
    if (b_in->nrows != a->nrows) HS_THROW(InvalidInput, "B has order " << b_in->nrows << ", A has " << a->nrows);
    b = hermitian_input(*b_in, cfg.hermitian_tol, "B", rep.hermitian_defect_b, b_store);
  }
  const index_t n = a->nrows;
  if (n == 0) HS_THROW(InvalidInput, "empty matrix");
  rep.n = n;
  rep.nnz_a = a->nnz();
  rep.nnz_b = b ? b->nnz() : 0;
  rep.complex_scalars = is_complex_v<T>;
  const auto na = driver::estimate_norm2(*a, 30, cfg.seed);
  rep.norm2_a = na.norm2;
  rep.norm2_b = b ? driver::estimate_norm2(*b, 30, cfg.seed).norm2 : 1.0;
  rep.scale = rep.norm2_b > 0 ? rep.norm2_a / rep.norm2_b : rep.norm2_a;
  const double scale = std::max(rep.scale, 1e-300);
  event(c, "problem", {{"n", static_cast<std::int64_t>(n)}, {"nnz_a", static_cast<std::int64_t>(rep.nnz_a)},
                       {"nnz_b", static_cast<std::int64_t>(rep.nnz_b)}, {"complex", rep.complex_scalars},
                       {"norm2_a", rep.norm2_a}, {"norm2_b", rep.norm2_b}});

  ShiftedPencil<T> pencil(*a, b);

  // 2. Problem type (B definiteness by inertia) and method.
  ProblemType type = cfg.problem;
  if (!b) {
    if (type == ProblemType::GeneralizedDefinite || type == ProblemType::GeneralizedIndefinite)
      HS_THROW(InvalidInput, "a generalized problem type was requested but no B was given");
    type = ProblemType::Standard;
  } else if (type == ProblemType::Standard) {
    HS_THROW(InvalidInput, "problem 'standard' requested but B was given");
  } else if (type == ProblemType::Auto) {
    Timer tb;
    const Inertia ib = compute_inertia(pencil, 0.0, 1.0, factor_options_from(cfg, 0.0));
    type = (ib.valid && ib.neg == 0 && ib.zero == 0) ? ProblemType::GeneralizedDefinite
                                                     : ProblemType::GeneralizedIndefinite;
    event(c, "b_inertia", {{"neg", static_cast<std::int64_t>(ib.neg)}, {"zero", static_cast<std::int64_t>(ib.zero)},
                           {"pos", static_cast<std::int64_t>(ib.pos)}, {"valid", ib.valid}, {"seconds", tb.seconds()}});
    rep.time_factor_s += tb.seconds();
  }
  rep.problem = type;
  Method method = cfg.method;
  if (method == Method::Auto)
    method = type == ProblemType::GeneralizedIndefinite ? Method::ShiftInvertKrylovSchur : Method::ShiftInvertLanczos;
  if (method == Method::ShiftInvertLanczos && type == ProblemType::GeneralizedIndefinite)
    HS_THROW(InvalidInput, "shift_invert_lanczos needs B positive definite; B is indefinite");
  if (method == Method::Lanczos) HS_THROW(Unsupported, "factorization-free Lanczos is roadmap P3");
  rep.method = method;

  // 3. Memory budget for the factorization (DESIGN §3.6-3).
  const double budget = cfg.memory_budget_gb * 1073741824.0;
  const index_t k_guess = cfg.target.mode == TargetMode::Interval ? std::min<index_t>(n, 64) : cfg.target.k;
  const index_t ncv_guess = cfg.ncv > 0 ? cfg.ncv : std::max<index_t>(2 * (k_guess + 16) + 1, k_guess + 32);
  const double krylov_bytes = static_cast<double>(n) * sizeof(T) * static_cast<double>(ncv_guess + 6 * k_guess);
  const double cap_mb = (budget - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0)) - krylov_bytes) * 0.95 / 1e6;
  if (cap_mb <= 0)
    HS_THROW(OutOfMemoryBudget, "matrices and Krylov basis alone exceed memory_budget_gb = " << cfg.memory_budget_gb);
  const FactorOptions fopts = factor_options_from(cfg, cap_mb);
  rep.factor_options = fopts;
  auto fac = make_mumps_factorization<T>(fopts);

  // 4-10. Solve along the chosen path.
  Solution<T> sol = (method == Method::ShiftInvertKrylovSchur)
                        ? solve_indefinite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale)
                        : solve_definite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale, na);
  rep.factor_stats = fac->stats();
  rep.time_factor_s += rep.factor_stats.analyze_s + rep.factor_stats.factorize_s;
  rep.time_total_s = total.seconds();
  rep.peak_rss_bytes = peak_rss_bytes();
  event(c, "done", {{"success", rep.success}, {"pairs", static_cast<std::int64_t>(rep.pairs.size())},
                    {"seconds", rep.time_total_s}});
  sol.report = std::move(rep);
  return sol;
}

#define HS_INSTANTIATE_SOLVE(T) \
  template Solution<T> solve<T>(const CsrMatrix<T>&, const CsrMatrix<T>*, const SolverConfig&, Logger*);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE)
#undef HS_INSTANTIATE_SOLVE

}  // namespace hs
