// The GHIEP path (B Hermitian indefinite): shift-invert general Krylov-Schur, refinement on the
// projected pencil and real/complex classification (docs/DESIGN.md §3.6-10).
#include <algorithm>
#include <cmath>
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
  PhaseScope phase(c.clock, Phase::Verify);  // default for work outside Factor / Iterate scopes
  rep.certificate.note = "not available for indefinite B (Sylvester inertia does not count eigenvalues)";

  bool detection_off = false;
  const double sigma = factorize_near(c, fac, t.sigma, scale, detection_off);
  rep.sigma_used = sigma;

  SolveRefinement refine;
  if (fac.options().precision == FactorPrecision::Single || fac.options().blr) {
    refine.max_steps = 10;
    refine.rel_tol = cfg.tol * cfg.inner_tol_factor * 0.1;
  }
  ShiftInvertOperator<T> op(fac, pencil, sigma, refine);
  KrylovSchurOptions<T> ko;
  ko.nev = std::min<index_t>(n, t.k);
  ko.ncv = cfg.ncv;
  ko.max_restarts = cfg.max_restarts;
  ko.tol = cfg.tol * cfg.inner_tol_factor;
  ko.seed = cfg.seed;
  GeneralEigResult<T> res;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    res = krylov_schur_general<T>(op, Which::LargestMagnitude, ko);
  }
  rep.restarts = res.restarts;
  rep.op_applies = res.op_applies;
  event(c, "krylov_schur", {{"nev", static_cast<std::int64_t>(ko.nev)}, {"nconv", static_cast<std::int64_t>(res.nconv)},
                            {"restarts", static_cast<std::int64_t>(res.restarts)}});
  if (!res.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

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
  return sol;
}

#define HS_INSTANTIATE_SOLVE_INDEFINITE(T)                                                                  \
  template Solution<T> solve_indefinite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                           double, double);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_INDEFINITE)
#undef HS_INSTANTIATE_SOLVE_INDEFINITE

}  // namespace hs::driver
