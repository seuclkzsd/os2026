// The HEP / GHEP path (B Hermitian positive definite): shift-invert thick-restart Lanczos in the
// B inner product, Rayleigh-Ritz on the original pencil, residual rescue and the inertia
// certificate with deflated reruns (docs/DESIGN.md §3.6).
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "driver/projection.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

namespace {

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
  HermitianEigResult<T> res;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    res = lanczos_krylov_schur<T>(op, b_op.get(), Which::LargestMagnitude, ko);
  }
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

}  // namespace

template <class T>
Solution<T> solve_definite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                           double n2b, double scale, double sigma0) {
  const SolverConfig& cfg = c.cfg;
  SolveReport& rep = c.rep;
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  const index_t n = pencil.order();
  Target t = cfg.target;
  rep.sigma_requested = sigma0;
  // Work not inside a Factor / Iterate scope (refinement, residuals, rescue, certificate
  // bookkeeping) is verification.
  PhaseScope phase(c.clock, Phase::Verify);

  // Interval: count the eigenvalues first (Sylvester inertia) to size the Krylov run.
  index_t k = t.k;
  index_t interval_count = -1;
  double lo = t.lower, hi = t.upper;
  if (t.mode == TargetMode::Interval) {
    const double step = 1e-10 * std::max(scale, std::max(std::abs(lo), std::abs(hi)));
    const index_t nu_lo = count_below(c, fac, pencil, lo, -1.0, step);
    const index_t nu_hi = count_below(c, fac, pencil, hi, +1.0, step);
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

  // Factorization at the shift (near-singular handling in factorize_near).
  bool detection_off = false;
  const double sigma = factorize_near(c, fac, sigma0, scale, detection_off);
  rep.sigma_used = sigma;
  const FactorStats& fs = fac.stats();
  event(c, "factorization", {{"sigma", sigma}, {"factor_s", fs.factorize_s}, {"entries", static_cast<std::int64_t>(fs.factor_entries)},
                             {"mb", fs.factor_mb_used}, {"delayed_pivots", static_cast<std::int64_t>(fs.delayed_pivots)}});

  // Krylov run (+ a margin of extra pairs so that the k-th nearest is reliably among them).
  // Converge exactly the k wanted pairs (most wanted by |theta| = nearest to sigma). Asking for
  // extra pairs only costs iterations: completeness is the certificate's job, not a safety margin.
  const index_t nev = std::min<index_t>(n, k);
  DefiniteRun<T> run;
  lanczos_run(c, fac, pencil, sigma, nev, static_cast<const DenseMatrix<T>*>(nullptr), run);
  if (!run.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

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
      if (std::isfinite(a_end)) nu_a = count_below(c, fac, pencil, a_end, -1.0, step);
      if (std::isfinite(b_end)) nu_b = count_below(c, fac, pencil, b_end, +1.0, step);
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
    refactorize_at(c, fac, sigma, detection_off);
    DenseMatrix<T> defl = run.all.x;
    lanczos_run(c, fac, pencil, sigma, nev_rerun, &defl, run);
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
  return sol;
}

#define HS_INSTANTIATE_SOLVE_DEFINITE(T)                                                                  \
  template Solution<T> solve_definite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                         double, double, double);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_DEFINITE)
#undef HS_INSTANTIATE_SOLVE_DEFINITE

}  // namespace hs::driver
