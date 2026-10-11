// The GHIEP path (B Hermitian indefinite): shift-invert general Krylov-Schur, refinement on the
// projected pencil and real/complex classification (docs/DESIGN.md §3.6-10), and the zero-cluster
// handling of §3.6-9.
//
// Zero cluster. In EM pencils A often has a large null space (waveguides: A = blkdiag(A_tt, 0),
// dim = #interior nodes); every null vector is an eigenvector with lambda = 0. Shift-invert near 0
// maps the whole cluster to one dominant theta = -1/sigma, and a single-vector Krylov space then
// converges copy after copy of it (rounding supplies new directions), crowding out the wanted
// modes. Such a cluster is never wanted (non-physical; zero_filter_tol widens the band):
//   1. a plain run at sigma (LargestMagnitude) is kept when its k nearest pairs stay clear of the
//      band |lambda| <= band;
//   2. otherwise (or when sigma lies in the band) A's zero-pivot count gives the cluster dimension
//      (one factorization of A on the existing analysis), and if there is a cluster (or the user
//      asked for a band) the eigenvalues are taken from one-sided runs at sigma+ > band
//      (LargestAlgebraic: theta > 0 <=> lambda > sigma+) and sigma- < -band (SmallestAlgebraic),
//      where the cluster sits at the unwanted end (theta = -1/sigma+, -1/sigma-) - shift placement,
//      no explicit deflation of the null space. Each side shift goes mid-way between the cluster
//      and the first eigenvalue on that side, located by a probe run (one pair, loose tolerance)
//      at the band edge: a shift right at the band edge leaves the cluster's |theta| = 1/band
//      dominating the wanted ones by lambda_first/band (measured: residuals stall at 1e-10..1e-9
//      for band = 1e-8 scale), mid-gap the two are balanced. An empty side shows up in the probe.
//      The union of all Ritz vectors is refined once on the projected pencil and the k nearest
//      sigma outside the band are selected. A side is rerun with more pairs until its converged
//      range covers the final radius, or it runs out of eigenvalues (exhausted: the unconverged
//      wanted Ritz values have the wrong sign).
// GHIEP supports target mode 'nearest' only: interval / smallest / largest need eigenvalue counts,
// which Sylvester inertia does not give for indefinite B.
#include <algorithm>
#include <cmath>
#include <limits>
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

/// Pairs with a backward error above this are not trusted as eigenpairs (selection fills a
/// shortfall with them, reported as not accepted).
constexpr double kTrustworthy = 1e-6;
/// |lambda| <= kZeroBand * scale is "numerically zero" (zero_filter_tol off): the cluster band.
constexpr double kZeroBand = 1e-8;
/// Restart budget of the first attempt of a one-sided run: enough for well-separated runs (2..5
/// restarts measured), small enough that a side with fewer eigenvalues than requested (its extra
/// wanted Ritz values chase the dense theta ~ 0 end and never converge) costs little. Well-placed
/// runs took 2..6 restarts in the tests and on the waveguide cases.
constexpr index_t kSideFirstRestarts = 16;
/// Probe locating the first eigenvalue beyond the band on one side: one pair to this relative
/// Ritz tolerance (placement only needs a few digits), within this many restarts.
constexpr double kProbeTol = 1e-6;
constexpr index_t kProbeRestarts = 4;

template <class T>
struct KsRun {
  double sigma = 0.0;
  Which which = Which::LargestMagnitude;
  index_t nev = 0;
  GeneralEigResult<T> res;
  bool exhausted = false;  ///< one-sided: fewer eigenvalues on the side than requested
};

/// Factorizes K(sigma) (near-singular handling) and runs the general Krylov-Schur solver.
template <class T>
KsRun<T> ks_run(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double scale, double sigma0,
                Which which, index_t nev, index_t max_restarts, double tol = -1.0) {
  const SolverConfig& cfg = c.cfg;
  KsRun<T> run;
  bool detection_off = false;
  run.sigma = factorize_near(c, fac, sigma0, scale, detection_off);
  run.which = which;
  run.nev = std::min<index_t>(pencil.order(), nev);
  SolveRefinement refine;
  if (fac.options().precision == FactorPrecision::Single || fac.options().blr) {
    refine.max_steps = 10;
    refine.rel_tol = cfg.tol * cfg.inner_tol_factor * 0.1;
  }
  ShiftInvertOperator<T> op(fac, pencil, run.sigma, refine);
  KrylovSchurOptions<T> ko;
  ko.nev = run.nev;
  ko.ncv = cfg.ncv;
  ko.max_restarts = max_restarts;
  ko.tol = tol > 0 ? tol : cfg.tol * cfg.inner_tol_factor;
  ko.seed = cfg.seed;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    run.res = krylov_schur_general<T>(op, which, ko);
  }
  event(c, "krylov_schur", {{"sigma", run.sigma}, {"nev", static_cast<std::int64_t>(ko.nev)},
                            {"nconv", static_cast<std::int64_t>(run.res.nconv)},
                            {"restarts", static_cast<std::int64_t>(run.res.restarts)},
                            {"op_applies", static_cast<std::int64_t>(run.res.op_applies)}});
  return run;
}

/// One-sided run: are the wanted-but-unconverged Ritz values on the wrong side of the shift
/// (theta of the wrong sign, or negligible next to the converged ones)? Then the side holds no
/// further eigenvalues: its extra wanted values are the dense theta ~ 0 end of far eigenvalues.
template <class T>
bool side_exhausted(const KsRun<T>& run) {
  const auto& r = run.res;
  const double sgn = run.which == Which::LargestAlgebraic ? 1.0 : -1.0;
  double big = 0.0;  // largest |theta| on the wanted side (the cluster sits on the other one)
  for (const cplx& th : r.ritz_all)
    if (sgn * th.real() > 0) big = std::max(big, std::abs(th));
  for (index_t j = r.nconv; j < static_cast<index_t>(r.theta.size()) && j < run.nev; ++j)
    if (sgn * r.theta[j].real() > 1e-3 * big) return false;  // a genuine one-sided value still converging
  return r.nconv < run.nev;
}

struct Cand {
  cplx lambda;
  bool real = false;
  PairMetrics pm;
  index_t col = 0;
};

template <class T>
struct Refined {
  std::vector<Cand> cands;
  DenseMatrix<T> xr;     ///< real-classified vectors, column = Cand::col
  DenseMatrix<cplx> xc;  ///< (complex) refined vectors, column = Cand::col
};

// Refinement on the projected pencil over the union of the runs' Ritz vectors, then real/complex
// classification (DESIGN §3.6-10) with metrics for every candidate.
template <class T>
Refined<T> refine_union(Ctx& c, const ShiftedPencil<T>& pencil, const std::vector<KsRun<T>>& runs, double n2a,
                        double n2b, double scale) {
  const SolverConfig& cfg = c.cfg;
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  const index_t n = pencil.order();
  // Real basis of the computed invariant subspaces: Re/Im parts of every vector (for real T a real
  // eigenvalue's vector is real up to a phase; a near-real conjugate pair - e.g. the zero cluster -
  // spans the same space as its real and imaginary parts). Dependent columns are dropped.
  index_t m = 0;
  for (const auto& r : runs) m += static_cast<index_t>(r.res.theta.size());
  DenseMatrix<T> w(n, is_complex_v<T> ? m : 2 * m);
  index_t col0 = 0;
  for (const auto& r : runs) {
    const index_t mr = static_cast<index_t>(r.res.theta.size());
    for (index_t j = 0; j < mr; ++j)
      for (index_t i = 0; i < n; ++i) {
        if constexpr (is_complex_v<T>) {
          w(i, col0 + j) = r.res.vectors(i, j);
        } else {
          w(i, 2 * (col0 + j)) = r.res.vectors(i, j).real();
          w(i, 2 * (col0 + j) + 1) = r.res.vectors(i, j).imag();
        }
      }
    col0 += mr;
  }
  driver::orthonormalize_columns(w, 1e-8);
  DenseMatrix<T> aw, bw;
  driver::apply_block(&a, w, aw);
  driver::apply_block(b, w, bw);
  std::vector<cplx> mu;
  Refined<T> out;
  if (!driver::rayleigh_ritz_indefinite(w, aw, bw, mu, out.xc)) {
    warn(c, "projected pencil singular; using the Krylov-Schur pairs without refinement");
    mu.clear();
    out.xc.resize(n, m);
    index_t q = 0;
    for (const auto& r : runs)
      for (index_t j = 0; j < static_cast<index_t>(r.res.theta.size()); ++j, ++q) {
        mu.push_back(cplx(r.sigma) + 1.0 / r.res.theta[j]);
        std::copy(r.res.vectors.col(j), r.res.vectors.col(j) + n, out.xc.col(q));
      }
  }

  out.xr.resize(n, static_cast<index_t>(mu.size()));
  DenseMatrix<cplx> col(n, 1), acol, bcol;
  for (index_t j = 0; j < static_cast<index_t>(mu.size()); ++j) {
    if (!std::isfinite(std::abs(mu[j]))) continue;
    std::copy(out.xc.col(j), out.xc.col(j) + n, col.col(0));
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
      for (index_t i = 0; i < n; ++i) out.xr(i, j) = xreal(i, 0) / nrm;
    }
    out.cands.push_back(cd);
  }
  return out;
}

// The k candidates nearest sigma outside the band (complex distance). The projected pencil also
// yields pairs built from unconverged Krylov directions; they have large residuals and must not
// crowd out true pairs near sigma, so pairs with backward error <= kTrustworthy are preferred and
// the others only fill up a shortfall (reported as not accepted). `good` receives the number of
// trustworthy candidates selected.
std::vector<Cand> select_nearest(const std::vector<Cand>& all, double sigma, index_t k, double band, double tol,
                                 index_t& good_count, index_t& in_band) {
  auto nearer = [&](const Cand& p, const Cand& q) { return std::abs(p.lambda - cplx(sigma)) < std::abs(q.lambda - cplx(sigma)); };
  std::vector<Cand> good, poor;
  in_band = 0;
  for (const auto& cd : all) {
    if (band >= 0 && std::abs(cd.lambda) <= band) {
      if (cd.pm.backward_error <= std::max(kTrustworthy, tol)) ++in_band;
      continue;
    }
    (cd.pm.backward_error <= std::max(kTrustworthy, tol) ? good : poor).push_back(cd);
  }
  std::stable_sort(good.begin(), good.end(), nearer);
  std::stable_sort(poor.begin(), poor.end(), nearer);
  std::vector<Cand> out = good;
  if (out.size() > static_cast<std::size_t>(k)) out.resize(static_cast<std::size_t>(k));
  good_count = static_cast<index_t>(out.size());
  for (const auto& cd : poor) {
    if (out.size() >= static_cast<std::size_t>(k)) break;
    out.push_back(cd);
  }
  std::stable_sort(out.begin(), out.end(), nearer);
  return out;
}

/// dim null(A): zero pivots of an exact LDL^T of A on the main analysis (null-pivot detection).
template <class T>
index_t null_dimension(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil) {
  PhaseScope phase(c.clock, Phase::Factor);
  c.held = FactorHeld{};
  Inertia in;
  if (fac.options().precision == FactorPrecision::Double && !fac.options().blr) {
    fac.set_null_pivot_detection(true);
    if (fac.factorize(1.0, 0.0)) return 0;
    in = fac.inertia();
  } else {
    FactorOptions o = fac.options();
    o.precision = FactorPrecision::Double;
    o.blr = false;
    in = compute_inertia(pencil, 1.0, 0.0, o);
    if (in.valid) return 0;
  }
  return std::max<index_t>(in.zero, 0);
}

}  // namespace

template <class T>
Solution<T> solve_indefinite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                             double n2b, double scale) {
  const SolverConfig& cfg = c.cfg;
  SolveReport& rep = c.rep;
  const index_t n = pencil.order();
  const Target& t = cfg.target;
  if (t.mode != TargetMode::Nearest)
    HS_THROW(Unsupported, "indefinite-B problems support target mode 'nearest' only: interval / smallest / largest "
                          "need eigenvalue counts, which Sylvester inertia does not give for indefinite B");
  if (t.k > n) HS_THROW(InvalidInput, "target.k = " << t.k << " exceeds the matrix order " << n);
  const index_t k = t.k;
  rep.sigma_requested = t.sigma;
  PhaseScope phase(c.clock, Phase::Verify);  // default for work outside Factor / Iterate scopes
  rep.certificate.note = "not available for indefinite B (Sylvester inertia does not count eigenvalues)";

  const double band_user = cfg.zero_filter_tol >= 0 ? cfg.zero_filter_tol * scale : -1.0;
  const double band_zero = std::max(band_user, kZeroBand * scale);  // what counts as the zero cluster
  std::vector<KsRun<T>> runs;
  Refined<T> ref;
  std::vector<Cand> sel;
  index_t good = 0, in_band = 0;
  bool cluster_mode = false;
  auto cluster_dimension = [&]() {
    rep.zero_cluster_dim = null_dimension(c, fac, pencil);
    event(c, "zero_cluster", {{"dimension", static_cast<std::int64_t>(rep.zero_cluster_dim)}});
    return rep.zero_cluster_dim;
  };

  if (std::abs(t.sigma) <= band_zero) {
    // sigma inside the band: a plain run would be dominated by the cluster (if there is one).
    cluster_mode = band_user >= 0 || cluster_dimension() > 0;
  }
  if (!cluster_mode) {
    runs.push_back(ks_run(c, fac, pencil, scale, t.sigma, Which::LargestMagnitude, k, cfg.max_restarts));
    rep.sigma_used = runs.back().sigma;
    if (!runs.back().res.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");
    ref = refine_union(c, pencil, runs, n2a, n2b, scale);
    sel = select_nearest(ref.cands, t.sigma, k, band_user, cfg.tol, good, in_band);
    // Does the band interfere: a selected pair in it, or the selection disk reaching it?
    double r = 0.0;
    bool touches = false;
    for (const auto& cd : sel) {
      r = std::max(r, std::abs(cd.lambda - cplx(t.sigma)));
      touches = touches || std::abs(cd.lambda) <= band_zero;
    }
    touches = touches || static_cast<index_t>(sel.size()) < k || r >= std::abs(t.sigma) - band_zero;
    if (touches) cluster_mode = band_user >= 0 || cluster_dimension() > 0;
  }

  if (cluster_mode) {
    const double band = band_user >= 0 ? band_user : band_zero;
    const index_t plain_runs = static_cast<index_t>(runs.size());
    if (rep.zero_cluster_dim > 0)
      warn(c, "zero cluster of dimension " + std::to_string(rep.zero_cluster_dim) +
                  " (null space of A) excluded by one-sided shifts");
    // The cluster is exactly zero up to rounding: shifts never closer to it than kZeroBand*scale,
    // whatever the user's band.
    const double band_place = std::max(band, kZeroBand * scale);
    // Side s (0: lambda > band, 1: lambda < -band) is searched outward from its inner edge.
    // Without a plain run that is the band, and each side may hold all k. A plain run already
    // holds every genuine eigenvalue in its disk |lambda - sigma| < r_plain (its converged pairs,
    // cluster copies included): a side then only needs what lies beyond the disk - the shortfall -
    // searched from the disk edge (or from the band, if the disk reaches it). A probe at the inner
    // edge finds the first eigenvalue beyond it; the shift goes mid-way between the cluster and
    // that eigenvalue when this is farther out than the inner edge.
    index_t need_side[2] = {k, k};
    double inner[2] = {band_place, -band_place};
    if (plain_runs > 0) {
      const auto& pr = runs.front().res;
      double r_plain = 0.0;
      for (index_t j = 0; j < pr.nconv; ++j) r_plain = std::max(r_plain, 1.0 / std::abs(pr.theta[j]));
      index_t good_out = 0;
      for (const auto& cd : sel)
        if (cd.pm.backward_error <= std::max(kTrustworthy, cfg.tol) && std::abs(cd.lambda) > band) ++good_out;
      const index_t shortfall = k - good_out;
      const double edge_hi = t.sigma + r_plain * (1.0 - 1e-6), edge_lo = t.sigma - r_plain * (1.0 - 1e-6);
      if (edge_hi > band_place) inner[0] = edge_hi;
      if (edge_lo < -band_place) inner[1] = edge_lo;
      for (int s = 0; s < 2; ++s) need_side[s] = std::min(k, shortfall + 2);
      event(c, "cluster_plain_disk", {{"radius", r_plain}, {"shortfall", static_cast<std::int64_t>(shortfall)}});
    }
    bool exhausted[2] = {false, false};
    index_t side_run[2] = {-1, -1};
    double place[2] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    index_t probe_applies = 0;
    const int first = t.sigma >= 0 ? 0 : 1;
    for (int attempt = 0; attempt < 3; ++attempt) {
      bool ran = false;
      for (int q = 0; q < 2; ++q) {
        const int s = q == 0 ? first : 1 - first;
        if (exhausted[s] || need_side[s] < 1) continue;
        if (side_run[s] >= 0 && runs[side_run[s]].nev >= need_side[s]) continue;  // already covered
        const Which w = s == 0 ? Which::LargestAlgebraic : Which::SmallestAlgebraic;
        const double sgn = s == 0 ? 1.0 : -1.0;
        if (std::isnan(place[s])) {
          // Probe at the inner edge for the first eigenvalue on this side.
          const KsRun<T> probe = ks_run(c, fac, pencil, scale, inner[s], w, 1, kProbeRestarts, kProbeTol);
          probe_applies += probe.res.op_applies;
          place[s] = probe.sigma;
          // An eigenvalue beyond the band on this side has theta of the wanted sign, and the nearest
          // one is an extreme of the spectrum of Op: the first Krylov cycle shows it. No Ritz value
          // of that sign means an empty side (a probe would never converge there: its wanted value
          // is buried in the dense theta ~ 0 end, next to the dominant cluster).
          bool any = false;
          for (const cplx& th : probe.res.ritz_all) any = any || sgn * th.real() > 0;
          if (!any) {
            exhausted[s] = true;
            event(c, "side_empty", {{"side", static_cast<std::int64_t>(s)}});
            SliceReport sr;
            sr.lower = s == 0 ? probe.sigma : -kInf;
            sr.upper = s == 0 ? kInf : probe.sigma;
            sr.sigma = probe.sigma;
            sr.which = s == 0 ? "largest_algebraic" : "smallest_algebraic";
            sr.nev = 1;  // the probe: no Ritz value on this side of the shift
            sr.count_found = 0;
            sr.restarts = probe.res.restarts;
            sr.op_applies = probe.res.op_applies;
            sr.passed = true;
            rep.slices.push_back(sr);
            continue;
          }
          if (probe.res.nconv >= 1 && sgn * probe.res.theta[0].real() > 0) {
            // Mid-gap between the cluster and the first eigenvalue beyond the inner edge, unless
            // the inner edge (a plain disk's edge) is already farther out: no eigenvalue is skipped.
            const double first = probe.sigma + (1.0 / probe.res.theta[0]).real();
            if (std::abs(first) > 4.0 * band_place && sgn * (0.5 * first - probe.sigma) > 0) place[s] = 0.5 * first;
            event(c, "side_probe", {{"side", static_cast<std::int64_t>(s)}, {"first", first}, {"shift", place[s]}});
          }
        }
        const double sigma_s = place[s];
        KsRun<T> run = ks_run(c, fac, pencil, scale, sigma_s, w, need_side[s], std::min(cfg.max_restarts, kSideFirstRestarts));
        run.exhausted = !run.res.converged && side_exhausted(run);
        if (!run.res.converged && !run.exhausted && cfg.max_restarts > kSideFirstRestarts) {
          event(c, "side_rerun", {{"side", static_cast<std::int64_t>(s)}});
          run = ks_run(c, fac, pencil, scale, sigma_s, w, need_side[s], cfg.max_restarts);
          run.exhausted = !run.res.converged && side_exhausted(run);
        }
        if (!run.res.converged && !run.exhausted)
          warn(c, "one-sided Krylov-Schur at " + fmt(run.sigma) + " did not converge within max_restarts");
        exhausted[s] = run.exhausted;
        SliceReport sr;
        sr.lower = s == 0 ? run.sigma : -kInf;
        sr.upper = s == 0 ? kInf : run.sigma;
        sr.sigma = run.sigma;
        sr.which = s == 0 ? "largest_algebraic" : "smallest_algebraic";
        sr.nev = run.nev;
        sr.count_found = run.res.nconv;
        sr.reruns = attempt;
        sr.restarts = run.res.restarts;
        sr.op_applies = run.res.op_applies;
        sr.passed = run.res.converged || run.exhausted;
        rep.slices.push_back(sr);
        if (side_run[s] >= 0) {
          runs[side_run[s]] = std::move(run);
        } else {
          side_run[s] = static_cast<index_t>(runs.size());
          runs.push_back(std::move(run));
        }
        ran = true;
      }
      if (!ran && attempt > 0) break;
      ref = refine_union(c, pencil, runs, n2a, n2b, scale);
      sel = select_nearest(ref.cands, t.sigma, k, band, cfg.tol, good, in_band);
      // Coverage: a side's converged range (from its shift outward) must reach the final radius,
      // unless the side ran out of eigenvalues.
      double r = 0.0;
      for (const auto& cd : sel) r = std::max(r, std::abs(cd.lambda - cplx(t.sigma)));
      bool covered = true;
      for (int s = 0; s < 2; ++s) {
        if (exhausted[s] || side_run[s] < 0) continue;
        const auto& run = runs[side_run[s]];
        const double edge = s == 0 ? t.sigma + r : t.sigma - r;  // farthest point that matters
        if ((s == 0 && edge <= band) || (s == 1 && edge >= -band)) continue;  // side not reached
        double reach = run.sigma;  // farthest converged eigenvalue from the shift
        for (index_t j = 0; j < run.res.nconv; ++j) {
          const double l = run.sigma + (1.0 / run.res.theta[j]).real();
          reach = s == 0 ? std::max(reach, l) : std::min(reach, l);
        }
        // The k-th pair may be the side's last converged one itself: compare up to rounding.
        const double slack = 1e-9 * std::max({std::abs(edge), std::abs(t.sigma), band});
        const bool ok = run.res.nconv >= run.nev && (s == 0 ? reach >= edge - slack : reach <= edge + slack);
        if (!ok && run.res.nconv >= run.nev) {
          need_side[s] = std::min(n, 2 * need_side[s]);
          covered = false;
        }
      }
      if (covered && static_cast<index_t>(sel.size()) >= k && good >= k) break;
      if (covered) {  // too few trustworthy pairs: widen both sides
        for (int s = 0; s < 2; ++s)
          if (!exhausted[s]) need_side[s] = std::min(n, need_side[s] + (k - good) + 2);
      }
    }
    rep.zero_modes_filtered = in_band;
    rep.sigma_used = runs.empty() ? t.sigma : runs.front().sigma;
    rep.op_applies += probe_applies;
    rep.certificate.note += "; zero band |lambda| <= " + fmt(band) + " excluded by one-sided shifts";
  } else if (band_user >= 0) {
    rep.zero_modes_filtered = in_band;
  }
  for (const auto& r : runs) {
    rep.restarts += r.res.restarts;
    rep.op_applies += r.res.op_applies;
  }

  Solution<T> sol;
  index_t nreal = 0, ncomplex = 0;
  for (const auto& cd : sel) (cd.real ? nreal : ncomplex)++;
  sol.eigenvectors.resize(n, nreal);
  sol.complex_eigenvectors.resize(n, ncomplex);
  index_t jr = 0, jc = 0;
  bool all_ok = static_cast<index_t>(sel.size()) == k;
  std::vector<PairReport> complex_reports;
  for (const auto& cd : sel) {
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
      std::copy(ref.xr.col(cd.col), ref.xr.col(cd.col) + n, sol.eigenvectors.col(jr++));
      rep.pairs.push_back(p);
    } else {
      sol.complex_eigenvalues.push_back(cd.lambda);
      std::copy(ref.xc.col(cd.col), ref.xc.col(cd.col) + n, sol.complex_eigenvectors.col(jc++));
      complex_reports.push_back(p);
    }
  }
  rep.pairs.insert(rep.pairs.end(), complex_reports.begin(), complex_reports.end());
  if (ncomplex > 0) warn(c, std::to_string(ncomplex) + " genuinely complex eigenvalue(s) returned separately");
  if (!all_ok && static_cast<index_t>(sel.size()) < k)
    warn(c, "only " + std::to_string(sel.size()) + " of " + std::to_string(k) + " pairs found");
  if (rep.zero_modes_filtered > 0)
    warn(c, std::to_string(rep.zero_modes_filtered) + " computed zero mode(s) filtered (|lambda| <= band)");
  rep.success = all_ok;
  return sol;
}

#define HS_INSTANTIATE_SOLVE_INDEFINITE(T)                                                                  \
  template Solution<T> solve_indefinite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                           double, double);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_INDEFINITE)
#undef HS_INSTANTIATE_SOLVE_INDEFINITE

}  // namespace hs::driver
