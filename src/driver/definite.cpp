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
//     end borders the excluded zero band. One-sided runs at a count point reuse its factorization;
//     a count point on an eigenvalue (a user bound, say) is no shift: such a slice takes the centre,
//     an open one a shift mid-way in the certified gap beside that point.
//     Smallest / largest locate the window end U (L) by inertia search: counts are exact for B HPD.
//     The zero band (zero_filter_tol) is cut out of every window between two counted points.
//   * Clusters that counts cannot split (a multiple eigenvalue straddling the wanted count): the
//     search stops after a few bisections and returns a point mid-way in the certified gap beside
//     the cluster, on the side that keeps it in the window (largest k: below it). A smallest /
//     largest window whose k-th eigenvalue lies in a large cluster of numerically equal
//     eigenvalues (a null space) asks only for the copies still needed, verified by two counts.
//   * Every computed block passes a rank-revealing Rayleigh-Ritz, so counts of found pairs are
//     counts of distinct B-orthonormal vectors. A slice missing pairs reruns deflated while the
//     reruns find new ones, with its own shift on a multiple eigenvalue among the found pairs
//     (single-vector Krylov sees further copies only where the multiplet dominates).
// A nearest target whose selection or certificate window reaches the zero band is handed to the
// windows as [sigma - r, sigma + r] minus the band, r grown from the plain run's Ritz hints until
// the counts hold k eigenvalues outside the band (a single shift there converges null-space copies).
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "driver/certificate.hpp"
#include "driver/internal.hpp"
#include "driver/projection.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

namespace {

/// Pairs with a backward error above this are not counted as found (ghost / unconverged Ritz
/// pairs); between tol and this they count but are reported as not accepted.
constexpr double kTrustworthy = 1e-6;
/// Rayleigh-Ritz drops directions whose B-Gram eigenvalue is below this fraction of the largest:
/// columns that are (numerically) the same vector - an eigenvector computed twice, or columns that
/// residual correction drove onto the same eigenvector - so that a count of computed pairs is a
/// count of distinct B-orthonormal vectors (DESIGN §3.6-8).
constexpr double kDependentGram = 1e-8;

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

// One shift-invert Lanczos run at the shift the factorization currently holds, optionally deflated,
// with Krylov blocks of p vectors (KrylovSchurOptions::block_size; p = 1: single vector).
// theta = 1/(lambda - sigma): LargestMagnitude = nearest sigma, SmallestAlgebraic = the eigenvalues
// just below sigma, LargestAlgebraic = just above. lead > 0: the run may end once the lead best
// ranked of the nev pairs have converged (KrylovSchurOptions::lead_nev).
template <class T>
void lanczos_run(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma, Which which,
                 index_t nev, index_t p, const DenseMatrix<T>* deflation, DefiniteRun<T>& out, index_t lead = 0) {
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
  ko.block_size = std::max<index_t>(p, 1);
  ko.lead_nev = lead < nev ? lead : 0;
  HermitianEigResult<T> res;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    res = lanczos_krylov_schur<T>(op, b_op.get(), which, ko);
  }
  out.restarts += res.restarts;
  out.applies += res.op_applies;
  out.converged = res.converged;
  event(c, "lanczos", {{"sigma", sigma}, {"which", std::string(which_name(which))},
                       {"nev", static_cast<std::int64_t>(nev)},
                       {"lead", static_cast<std::int64_t>(ko.lead_nev)},
                       {"block_size", static_cast<std::int64_t>(ko.block_size)},
                       {"nconv", static_cast<std::int64_t>(res.nconv)},
                       {"restarts", static_cast<std::int64_t>(res.restarts)},
                       {"op_applies", static_cast<std::int64_t>(res.op_applies)},
                       {"deflated", static_cast<std::int64_t>(deflation ? deflation->cols() : 0)},
                       {"max_estimate", res.history.empty() ? 0.0 : res.history.back().max_estimate}});
  if (!res.converged) {
    // Where the restarts stalled: the first restarts, then every 4^j-th, and the last one.
    const std::size_t len = res.history.size();
    for (std::size_t i = 0; i < len; i = (i < 4 || i + 1 == len) ? i + 1 : std::min(4 * i, len - 1)) {
      const RestartRecord& h = res.history[i];
      event(c, "lanczos_history", {{"restart", static_cast<std::int64_t>(h.restart)},
                                   {"nconv", static_cast<std::int64_t>(h.nconv)}, {"max_estimate", h.max_estimate},
                                   {"op_applies", static_cast<std::int64_t>(h.op_applies)}});
    }
  }
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
  const index_t m0 = out.all.size();
  const index_t r = driver::rayleigh_ritz_definite(out.all.x, out.all.ax, out.all.bx, out.all.lambda, kDependentGram);
  if (r < m0) event(c, "dependent_dropped", {{"columns", static_cast<std::int64_t>(m0 - r)}});
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

// Residual metrics of every column (DESIGN §4.5). Columns in parallel; each column's sums run in
// one thread in index order, so the result does not depend on the thread count.
template <class T>
std::vector<PairMetrics> block_metrics(const Block<T>& blk, double n2a, double n2b) {
  const index_t n = blk.x.rows(), m = blk.size();
  std::vector<PairMetrics> out(static_cast<std::size_t>(m));
#pragma omp parallel for schedule(static)
  for (index_t j = 0; j < m; ++j)
    out[static_cast<std::size_t>(j)] =
        driver::pair_metrics(blk.ax.col(j), blk.bx.col(j), blk.x.col(j), n, blk.lambda[j], n2a, n2b);
  return out;
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

/// Metrics of sel; pairs above tol get residual correction with the factorization that
/// prepare() makes the object hold (that of the run's shift; called only when a pair needs it),
/// then a rank-revealing Rayleigh-Ritz over sel (lambda ascending afterwards; columns that the
/// correction drove onto the same eigenvector are merged) and fresh metrics.
template <class T, class Prepare>
void verify_and_rescue(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, Block<T>& sel,
                       std::vector<PairMetrics>& metrics, double n2a, double n2b, Prepare&& prepare) {
  metrics = block_metrics(sel, n2a, n2b);
  std::vector<index_t> failing;
  for (index_t j = 0; j < sel.size(); ++j)
    if (metrics[j].backward_error > c.cfg.tol) failing.push_back(j);
  if (failing.empty()) return;
  prepare();
  if (!fac.inertia().valid) return;
  const index_t r = rescue_pairs(fac, pencil.a(), pencil.b(), sel, failing, n2a, n2b, c.cfg.tol);
  const index_t m0 = sel.size();
  const index_t kept = driver::rayleigh_ritz_definite(sel.x, sel.ax, sel.bx, sel.lambda, kDependentGram);
  event(c, "rescue",
        {{"pairs", static_cast<std::int64_t>(r)}, {"dependent_dropped", static_cast<std::int64_t>(m0 - kept)}});
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

// First radius of the windowed nearest solve, from the distances of the hints (Ritz values of the
// plain run, converged or not) outside the band: mid-way between the k-th nearest and the next
// distinct one, so that the window ends - which may become one-sided shifts - keep half that gap
// from the hinted eigenvalues (a radius just beyond the k-th left the shift 1e-6 relative from
// it: far pairs at eta ~7e-11 on yee-pec-box-1e4, sigma = 1, k = 6); 1.25 times the k-th without
// a further hint, twice the farthest if there are fewer than k.
double radius_from_hints(std::vector<double> hints, double sigma, index_t k, double band) {
  std::vector<double> d;
  for (double h : hints)
    if (std::isfinite(h) && std::abs(h) > band) d.push_back(std::abs(h - sigma));
  std::sort(d.begin(), d.end());
  const double floor_r = std::max(std::abs(sigma) + 2.0 * band, 4.0 * band);
  if (static_cast<index_t>(d.size()) < k) return std::max(d.empty() ? 0.0 : 2.0 * d.back(), floor_r);
  const double dk = d[static_cast<std::size_t>(k - 1)];
  double r = 1.25 * dk;
  for (std::size_t j = static_cast<std::size_t>(k); j < d.size(); ++j)
    if (d[j] > dk * (1.0 + 1e-6)) {
      r = 0.5 * (dk + d[j]);
      break;
    }
  return std::max(r, floor_r);
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

  // Factorization at the shift (near-singular handling in factorize_near). Its time and the solve
  // calls of the Krylov run are measured for the auto choice of the completeness check.
  bool detection_off = false;
  CertificateTimings timings;
  const FactorStats before_factor = fac.stats();
  const double sigma = factorize_near(c, fac, t.sigma, scale, detection_off);
  rep.sigma_used = sigma;
  const FactorStats& fs = fac.stats();
  if (fs.factorizations > before_factor.factorizations)
    timings.factorize_s = (fs.factorize_s - before_factor.factorize_s) /
                          static_cast<double>(fs.factorizations - before_factor.factorizations);
  event(c, "factorization", {{"sigma", sigma}, {"factor_s", fs.factorize_s}, {"entries", static_cast<std::int64_t>(fs.factor_entries)},
                             {"mb", fs.factor_mb_used}, {"delayed_pivots", static_cast<std::int64_t>(fs.delayed_pivots)}});

  // Converge exactly the k wanted pairs (most wanted by |theta| = nearest to sigma). Asking for
  // extra pairs only costs iterations: completeness is the certificate's job, not a safety margin.
  const index_t nev = std::min<index_t>(n, k);
  // Block width: the blocktune rule on the run's shape, with the entries of the factorization the
  // run's solves use (block_size_for; reserved up to its widest width before the analysis).
  eigen_detail::ShiftInvertShape shape;
  shape.n = n;
  shape.nev = nev;
  shape.factor_entries = static_cast<double>(fs.factor_entries);
  shape.nnz_b = pencil.b() ? static_cast<double>(pencil.b()->nnz()) : 0.0;
  shape.complex_scalars = is_complex_v<T>;
  // Candidate block X, AX, BX (3 nev) and the returned vectors (nev) next to the basis. The memory
  // plan reserved the P4 width: an auto width beyond it is halved back to it without a warning.
  const index_t p_rule = block_size_for(c.cfg, shape);
  const index_t p = fit_block_size<T>(c, n, nev, 0, 4 * nev, p_rule,
                                      c.cfg.block_size > 0 ? p_rule : std::min(p_rule, reserved_block_size(c.cfg)));
  rep.block_size = p;
  DefiniteRun<T> run;
  const FactorStats before_run = fac.stats();
  lanczos_run(c, fac, pencil, sigma, Which::LargestMagnitude, nev, p, static_cast<const DenseMatrix<T>*>(nullptr),
              run);
  if (fac.stats().solves > before_run.solves) {
    const auto calls = static_cast<double>(fac.stats().solves - before_run.solves);
    timings.solve_call_s = (fac.stats().solve_s - before_run.solve_s) / calls;
    timings.rhs_per_call = static_cast<double>(fac.stats().rhs_solved - before_run.rhs_solved) / calls;
  }
  if (!run.converged) warn(c, "Krylov-Schur did not converge all requested pairs within max_restarts");

  const double zero_tol = cfg.zero_filter_tol >= 0 ? cfg.zero_filter_tol * scale : -1.0;
  Block<T> sel;
  std::vector<PairMetrics> metrics;
  auto select_and_verify = [&]() {
    index_t zero_dropped = 0;
    sel = take_columns(run.all, select_nearest(run.all.lambda, t.sigma, k, zero_tol, zero_dropped));
    rep.zero_modes_filtered = zero_dropped;
    // rescue solves with K(sigma)
    verify_and_rescue(c, fac, pencil, sel, metrics, n2a, n2b, [&]() { refactorize_at(c, fac, sigma, detection_off); });
    index_t zd = 0;  // restore the target order (Rayleigh-Ritz sorts ascending)
    const auto order = select_nearest(sel.lambda, t.sigma, sel.size(), -1.0, zd);
    std::vector<PairMetrics> m2;
    std::vector<double> l2;
    for (index_t j : order) {
      m2.push_back(metrics[j]);
      l2.push_back(sel.lambda[j]);
    }
    if (static_cast<index_t>(order.size()) == sel.size()) {  // a permutation: reorder in place
      permute_columns(sel.x, order);
      permute_columns(sel.ax, order);
      permute_columns(sel.bx, order);
      sel.lambda = std::move(l2);
    } else {
      sel = take_columns(sel, order);
    }
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

  // Completeness check (DESIGN §3.6-8; method by SolverConfig::certificate, certificate.hpp),
  // with up to two deflated reruns when pairs are missing. The exact certificate's two counts per
  // round come from an InertiaCounter when that pays (counter_pays). The main factors at sigma are
  // released before the counter counts (MainFactors::Release): after the rescue above they are
  // needed only for a rerun, which refactorizes once - as main-object counts would - and the
  // counts never hold both in memory. The randomized check runs on the factors held at sigma.
  const bool want_cert = cfg.inertia_check != Toggle::Off && cfg.certificate != CertificateMode::Off;
  if (want_cert && cfg.certificate_fault_drop > 0) {
    // Fault injection (tests and measurements): the check must find these pairs missing.
    const std::vector<double> removed =
        drop_found_pairs(run.all, run.ritz_hints, sel.lambda, t.sigma, cfg.certificate_fault_drop);
    std::string list;
    for (double l : removed) list += (list.empty() ? "" : ", ") + fmt(l);
    warn(c, "certificate_fault_drop: " + std::to_string(removed.size()) +
                " found pair(s) removed before the completeness check (" + list + ")");
    select_and_verify();
  }
  InertiaCertificate& cert = rep.certificate;
  const bool on_counter = want_cert && counter_pays(cfg, fac.estimate(), mumps_input_size(pencil),
                                                    fac.options().threads, 2);
  CountSource<T> counts(fac, pencil, on_counter, MainFactors::Release);
  NearestCertifier<T> fast(c, pencil, fac, sigma, detection_off, scale, n2a, n2b, timings);
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
    index_t missing = 0;
    index_t zd = 0;
    // Fewer than k selected pairs (fault injection, pairs left unconverged, a rescue merging
    // columns): every check covers only the found pairs' window / disc, which may end before the
    // k-th nearest eigenvalue - a passed check then still reruns for the pairs short.
    const index_t short_by = std::min<index_t>(k, n) - sel.size();
    auto rerun_short = [&]() {
      warn(c, "certificate: only " + std::to_string(sel.size()) + " of " + std::to_string(k) +
                  " pairs selected; deflated rerun for the nearest ones beyond");
      return short_by;
    };
    const RoundPairs<T> pairs{run.all, &sel, select_nearest(run.all.lambda, t.sigma, k, zero_tol, zd)};
    const CheckOutcome quick = fast.check({t.sigma, r, a_end, b_end}, pairs, round);
    if (quick.verdict == CheckOutcome::Verdict::Passed) {
      if (short_by == 0 || round == 2) break;
      missing = rerun_short();
    } else if (quick.verdict == CheckOutcome::Verdict::Missing && round < 2) {
      // Still missing after the last rerun: the exact counts give the final verdict (below).
      missing = std::max(quick.missing, short_by);
      warn(c, "certificate: the " + cert.method + " check finds " + std::to_string(quick.missing) +
                  " eigenvalue(s) missing near sigma; deflated rerun");
    } else {
      // Exact counts: on the certifier's InertiaCounter when its single counts built one (no
      // second analysis; a window it counted to check single counts is not counted again).
      index_t nu_a = 0, nu_b = 0;
      const bool on_certifier = fast.exact_counts(a_end, b_end, nu_a, nu_b);
      if (!on_certifier) {
        const double step = 1e-9 * std::max(scale, std::abs(t.sigma));
        nu_a = count_below(c, counts, a_end, -1.0, step, nullptr, /*s_is_gap=*/true);
        nu_b = count_below(c, counts, b_end, +1.0, step, nullptr, /*s_is_gap=*/true);
      }
      cert.performed = true;
      cert.lower = a_end;
      cert.upper = b_end;
      cert.count_inertia = nu_b - nu_a;
      index_t found = 0;
      for (double l : run.all.lambda)
        if (std::isfinite(l) && l >= a_end && l <= b_end && !(zero_tol >= 0 && std::abs(l) <= zero_tol)) ++found;
      cert.count_found = found;
      cert.passed = cert.count_inertia == cert.count_found;  // the window stays clear of the zero band
      fast.record_exact(on_certifier || counts.uses_counter());
      event(c, "certificate", {{"round", static_cast<std::int64_t>(round)}, {"lower", a_end}, {"upper", b_end},
                               {"count_inertia", static_cast<std::int64_t>(cert.count_inertia)},
                               {"count_found", static_cast<std::int64_t>(found)}, {"passed", cert.passed},
                               {"short_by", static_cast<std::int64_t>(short_by)}});
      if (cert.count_inertia < cert.count_found || round == 2) break;
      if (cert.passed) {
        if (short_by == 0) break;
        missing = rerun_short();
      } else {
        missing = std::max(cert.count_inertia - cert.count_found, short_by);
        warn(c, "certificate: " + std::to_string(cert.count_inertia - cert.count_found) +
                    " eigenvalue(s) missing in the window; deflated rerun");
      }
    }
    keep_converged(run, n2a, n2b, cfg.tol);
    // The missing pairs are the nearest sigma beyond the deflated ones: the rerun asks for two
    // more, which shape its basis and are kept when they converge, but it ends once the missing
    // ones have (lead; the next round checks the window again). A pair beyond them may lie in a
    // cluster wider than the inner tolerance and never converge: yee-pec-box-1e5, k = 10, one
    // copy of the double eigenvalue 59.986 removed - the rerun's third pair is a vector of the
    // 35475-fold curl-curl null space (|theta| = 1/sigma, its Ritz values at lambda ~ 4e-9),
    // whose estimate stalled at 3e-10 |theta| against the inner tolerance 1e-11 at widths 4 and
    // 8 (deterministic run, 4 threads, max_restarts 60: 1236 / 1976 applications; with lead 40 /
    // 72); reruns of width 2 got under it after 154 and 218 applications (with lead: 20). In the
    // measurement slot (16 threads, 5 interleaved runs per width) the rerun now ends after one
    // restart at widths 4 and 8, whole solve 4.65 .. 4.81 s (the review: 1000 restarts, ~200 s).
    const index_t nev_rerun = std::min<index_t>(n - run.all.size(), missing + 2);
    if (nev_rerun < 1) break;
    refactorize_at(c, fac, sigma, detection_off);
    DenseMatrix<T> defl = run.all.x;
    const index_t d = defl.cols();
    const index_t p_rerun =
        fit_block_size<T>(c, n, nev_rerun, d, 3 * d + 4 * nev_rerun, rerun_block_size(cfg, p, missing));
    lanczos_run(c, fac, pencil, sigma, Which::LargestMagnitude, nev_rerun, p_rerun, &defl, run, missing);
    select_and_verify();
  }
  if (cert.performed && cert.passed && sel.size() < std::min<index_t>(k, n))
    cert.statement += "; but only " + std::to_string(sel.size()) + " of " + std::to_string(k) +
                      " pairs were found: the window covers the found pairs, not the k-th nearest eigenvalue";
  if (cert.performed && !cert.passed) {
    if (cert.count_inertia < 0)
      warn(c, "certificate failed: the " + cert.method + " check still finds eigenvalues missing near sigma");
    else if (cert.count_inertia < cert.count_found)
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
/// s = +inf (nu = n) stand for open window ends. A point inside a certified gap (between two
/// counted points with equal counts) carries that count without a factorization of its own.
struct CountPoint {
  double s = 0.0;
  index_t nu = 0;
  bool detection_off = false;  ///< counted without null-pivot detection (dense cluster)
  bool near = false;           ///< K(s) was singular before s moved off an eigenvalue (or s hugs an
                               ///< unsplittable cluster): a valid count, but no place for a shift
};

/// Part of a target window with exact counts at both ends. want >= 0: the slice is a cluster of
/// numerically equal eigenvalues (smallest / largest k ending inside it) of which any `want`
/// are a valid answer; focus is a point of the cluster's tight bracket.
struct Slice {
  CountPoint lo, hi;
  index_t want = -1;
  double focus = std::numeric_limits<double>::quiet_NaN();
  index_t count() const { return hi.nu - lo.nu; }
  index_t required() const { return want >= 0 ? want : count(); }
};

/// Result of an inertia search. cluster: eigenvalues too close to be split by counts straddle the
/// requested count range; (cl_lo, cl_hi) is their tight bracket (cl_lo.nu < g_lo, cl_hi.nu > g_hi,
/// every eigenvalue counted between them lies in (cl_lo.s, cl_hi.s)) and p lies in the gap beside
/// the cluster on the requested side.
struct Search {
  CountPoint p;
  bool cluster = false;
  CountPoint cl_lo, cl_hi;
  static Search at(const CountPoint& q) {
    Search r;
    r.p = q;
    return r;
  }
};

/// Side of an unsplittable cluster an inertia search returns: Above (the window must contain the
/// cluster: smallest k, slice cuts) or Below (largest k).
enum class Side { Above, Below };

enum class Anchor { Centre, Lower, Upper };

/// Bracket of a cluster that counts cannot split: narrower than this fraction of the bracket in
/// which the same two counts were first seen (the eigenvalues between them are then >= 32x
/// denser than that bracket suggested; five bisections), or than width_min().
constexpr double kClusterShrink = 1.0 / 32.0;
/// Smallest / largest: a cluster at the k-th position with more than max(this, k) copies beyond
/// the k needed is solved partially (any `want` copies of numerically equal eigenvalues).
constexpr index_t kPartialClusterMin = 16;
/// Numerically equal: all eigenvalues of a partial cluster within this relative width.
constexpr double kEqualWidth = 1e-8;
/// Deflated reruns of a slice: at most this many beyond the initial deficit, and stop after
/// kStallRounds rounds without a new pair.
constexpr int kExtraRerunRounds = 4;
constexpr int kMaxRerunRounds = 32;
constexpr int kStallRounds = 2;
/// Rerun shift of a multiple eigenvalue: this fraction of the gap to its nearest neighbour away.
constexpr double kMultipletOffset = 0.01;

/// Pairs accepted from all slices, eigenvalue order maintained per slice.
template <class T>
struct Pool {
  DenseMatrix<T> x;  ///< n x capacity, the first `used` columns hold the pairs
  index_t used = 0;
  std::vector<double> lambda;
  std::vector<PairMetrics> metrics;
  std::vector<index_t> slice;  ///< producing slice of each pair
};

/// A target window: [lo, hi] with exact counts; want >= 0 marks a partial cluster (see Slice).
using Window = Slice;

template <class T>
class WindowSolver {
 public:
  WindowSolver(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a, double n2b,
               double scale, const SpectrumEstimate& est)
      : c_(c),
        pencil_(pencil),
        fac_(fac),
        counts_(fac, pencil, c.cfg.inertia_counter == Toggle::On),
        n2a_(n2a),
        n2b_(n2b),
        scale_(scale),
        est_(est),
        n_(pencil.order()) {
    cost_ = make_cost_model(fac.estimate(), n_, fac.options().threads, is_complex_v<T>);
  }

  Solution<T> solve();
  /// Nearest target with a zero band: the k nearest sigma outside the band, from windows
  /// [sigma - r, sigma + r] minus the band; r0 (> 0) is the first radius tried.
  void set_nearest_radius(double r0) { nearest_r0_ = r0; }

 private:

  // ---- counts and cuts
  CountPoint count(double s, double outward);
  void remember(const CountPoint& p);
  CountPoint minus_inf() const { return {-kInf, 0, false, false}; }
  CountPoint plus_inf() const { return {kInf, n_, false, false}; }
  double width_min(double x, double y) const {
    return 1e-8 * std::max({scale_, std::abs(x), std::abs(y)});
  }
  double expansion_step() const;
  Search find_count(const CountPoint& a, const CountPoint& b, index_t g_lo, index_t g, index_t g_hi, double start,
                    Side side, double guess = std::numeric_limits<double>::quiet_NaN());
  CountPoint gap_point(const CountPoint& p, double dir, const CountPoint& limit);
  std::vector<Slice> plan(const Window& w);
  Anchor anchor_of(const Slice& s) const;
  // ---- runs
  bool run_slice(const Slice& s, index_t id);
  double multiplet_shift(const Block<T>& sel, const Slice& s, std::vector<double>& tried) const;
  double multiplet_rerun_shift(double mean, const Block<T>& sel, const Slice& s) const;
  bool cluster_verified(const Slice& s, const std::vector<double>& lambda, double& mean, double& delta);
  void reserve_pool(index_t pairs);
  void append_to_pool(const Block<T>& blk, const std::vector<PairMetrics>& m, index_t id);
  void compact_pool(const std::vector<char>& drop);
  index_t deduplicate();
  bool in_band(double l) const { return band_ >= 0.0 && l > band_lo_ && l < band_hi_; }
  double factorize_anchor(const CountPoint& p, bool& detection_off);

  Ctx& c_;
  const ShiftedPencil<T>& pencil_;
  SparseFactorization<T>& fac_;
  /// Counts on the main object by default: a count point can then serve as a one-sided shift
  /// without refactorization (anchors), which an InertiaCounter count cannot.
  CountSource<T> counts_;
  double n2a_, n2b_, scale_;
  SpectrumEstimate est_;
  index_t n_;
  CostModel cost_;
  index_t c_target_ = 0, c_max_ = 0;
  double band_ = -1.0;                ///< zero band half-width (zero_filter_tol * scale), < 0: none
  double band_lo_ = 0.0, band_hi_ = 0.0;  ///< the band as counted: (band_lo_, band_hi_) is excluded
  std::vector<CountPoint> known_;     ///< every count evaluated (or certified), ascending in s
  std::vector<Slice> slices_;
  std::vector<SliceReport> reports_;  ///< per slice
  Pool<T> pool_;
  index_t restarts_ = 0, applies_ = 0;
  double nearest_r0_ = -1.0;
};

template <class T>
void WindowSolver<T>::remember(const CountPoint& p) {
  auto it = std::lower_bound(known_.begin(), known_.end(), p.s, [](const CountPoint& q, double v) { return q.s < v; });
  if (it != known_.end() && it->s == p.s) {
    *it = p;
  } else {
    known_.insert(it, p);
  }
}

template <class T>
CountPoint WindowSolver<T>::count(double s, double outward) {
  for (const auto& p : known_)
    if (p.s == s) return p;
  const double step = 1e-10 * std::max(scale_, std::abs(s));
  bool off = false, near = false;
  const index_t nu = count_below(c_, counts_, s, outward, step, &off, false, &near);
  const CountPoint p{s, nu, off, near};
  remember(p);
  event(c_, "count", {{"s", s}, {"nu", static_cast<std::int64_t>(nu)}, {"near_eigenvalue", near}});
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
// Illinois modification: the end retained twice in a row gets half the weight; steps kept 5%
// inside the bracket) - by bisection after a count that equals one of the bracket's end counts,
// a sign that the eigenvalues between are not spread as interpolation assumes; an open end is
// found by geometric expansion from the finite point (start, or the bracket end that overshot).
// Eigenvalues too close together to be split by counts (the bracket of the same two counts shrank
// by kClusterShrink, or to width_min) end the search: the result is then a point in the gap
// beside that cluster on the requested side (gap_point), never one hugging it - such a point
// would be a poor shift, and reaching it took ~20 more counts.
template <class T>
Search WindowSolver<T>::find_count(const CountPoint& a, const CountPoint& b, index_t g_lo, index_t g, index_t g_hi,
                                   double start, Side side, double guess) {
  int same_side = 0, last_side = 0;
  double lft_weight = 1.0, rgt_weight = 1.0;  // Illinois weights of the retained ends
  index_t expansions = 0;
  index_t pair_lo = -1, pair_hi = -1;
  double pair_width = kInf;  // bracket width when its two end counts were first seen
  bool bisect = false;       // the last count repeated an end count of the bracket
  CountPoint lft = a, rgt = b;
  auto bracket = [&]() -> const CountPoint* {
    lft = a;
    rgt = b;
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
    return best;
  };
  auto beside_cluster = [&]() {
    Search r;
    r.cluster = true;
    r.cl_lo = lft;
    r.cl_hi = rgt;
    r.p = side == Side::Above ? gap_point(rgt, +1.0, b) : gap_point(lft, -1.0, a);
    event(c_, "count_cluster", {{"lower", lft.s}, {"upper", rgt.s},
                                {"eigenvalues", static_cast<std::int64_t>(rgt.nu - lft.nu)}, {"point", r.p.s}});
    return r;
  };
  for (int it = 0; it < 40; ++it) {
    if (const CountPoint* best = bracket()) return Search::at(*best);
    const bool lf = std::isfinite(lft.s), rf = std::isfinite(rgt.s);
    if (lf && rf) {
      if (lft.nu != pair_lo || rgt.nu != pair_hi) {
        pair_lo = lft.nu;
        pair_hi = rgt.nu;
        pair_width = rgt.s - lft.s;
      }
      if (rgt.s - lft.s <= std::max(width_min(lft.s, rgt.s), kClusterShrink * pair_width)) return beside_cluster();
    }
    double s = 0.0;
    const double d = expansion_step() * std::pow(4.0, static_cast<double>(expansions));
    if (std::isfinite(guess) && guess > lft.s && guess < rgt.s) {
      s = guess;  // caller's estimate (local eigenvalue density), tried once
      guess = std::numeric_limits<double>::quiet_NaN();
    } else if (!lf && !rf) {
      s = start;
    } else if (!lf) {  // expand downward from the point that overshot
      s = rgt.s - d;
      ++expansions;
    } else if (!rf) {
      s = lft.s + d;
      ++expansions;
    } else if (bisect) {
      s = 0.5 * (lft.s + rgt.s);
    } else {
      const double fl = lft_weight * (static_cast<double>(lft.nu) - static_cast<double>(g));  // < 0
      const double fr = rgt_weight * (static_cast<double>(rgt.nu) - static_cast<double>(g));  // > 0
      const double frac = std::clamp(-fl / (fr - fl), 0.05, 0.95);
      s = lft.s + frac * (rgt.s - lft.s);
    }
    const CountPoint p = count(s, +1.0);
    bisect = lf && rf && (p.nu == lft.nu || p.nu == rgt.nu);
    const int sd = p.nu < g_lo ? -1 : (p.nu > g_hi ? +1 : 0);
    same_side = (sd != 0 && sd == last_side) ? same_side + 1 : 1;
    last_side = sd;
    // Illinois: a new left end twice in a row halves the weight of the retained right end, and
    // vice versa; a switch of sides resets both.
    if (same_side >= 2 && sd < 0) rgt_weight *= 0.5;
    if (same_side >= 2 && sd > 0) lft_weight *= 0.5;
    if (same_side == 1) lft_weight = rgt_weight = 1.0;
  }
  if (const CountPoint* best = bracket()) return Search::at(*best);
  if (!std::isfinite(lft.s) || !std::isfinite(rgt.s))
    HS_THROW(NotConverged, "inertia search found no point with " << g << " eigenvalues below it");
  return beside_cluster();  // out of iterations: treat the bracket as unsplittable
}

// A point beside p on side dir (+1: above p, -1: below), in the same spectral gap (count p.nu),
// strictly between p and limit: the middle of the part of the gap that known counts certify (they
// share p's count, so no eigenvalue lies between them). Without such a point, up to three counts
// at p + dir * delta (delta: half the distance to the next known count, else the expansion step,
// quartered while the count differs) certify one. Each half of a certified gap keeps the point
// away from both the eigenvalues at p's side and the next ones beyond. If no gap is found, p is
// returned flagged near.
template <class T>
CountPoint WindowSolver<T>::gap_point(const CountPoint& p, double dir, const CountPoint& limit) {
  CountPoint far = p, next = limit;
  auto consider = [&](const CountPoint& q) {  // q beyond p, in order of distance; false: stop
    if (q.nu != p.nu) {
      next = q;
      return false;
    }
    far = q;
    return true;
  };
  if (dir > 0) {
    for (const auto& q : known_)
      if (q.s > p.s && q.s < limit.s && !consider(q)) break;
  } else {
    for (auto it = known_.rbegin(); it != known_.rend(); ++it)
      if (it->s < p.s && it->s > limit.s && !consider(*it)) break;
  }
  if (next.s == limit.s && std::isfinite(limit.s) && limit.nu == p.nu) far = limit;  // the gap reaches limit
  auto middle = [&](double x) {
    const CountPoint m{0.5 * (p.s + x), p.nu, false, false};
    remember(m);
    return m;
  };
  if (far.s != p.s) return middle(far.s);
  double delta = std::isfinite(next.s) ? 0.5 * std::abs(next.s - p.s) : expansion_step();
  for (int t = 0; t < 3; ++t) {
    const CountPoint q = count(p.s + dir * delta, dir);
    if (q.nu == p.nu && dir * (q.s - p.s) > 0) return middle(q.s);
    delta *= 0.25;
  }
  CountPoint out = p;
  out.near = true;
  return out;
}

// Cuts window w into slices of at most c_max_ eigenvalues (aiming at c_target_), marching upward
// from the finite lower end (an open lower end first gets its own leftmost slice). A partial
// cluster window is one slice.
template <class T>
std::vector<Slice> WindowSolver<T>::plan(const Window& w) {
  const CountPoint& a = w.lo;
  const CountPoint& b = w.hi;
  if (w.want >= 0) return {Slice{a, b, w.want, w.focus}};
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
    return find_count(cur, b, g_lo, cur.nu + step, std::max(g_lo, g_hi), start, Side::Above, guess).p;
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
// the zero band, else the centre. A finite end flagged near (an eigenvalue on it, e.g. a user
// bound) is no anchor: such a slice takes the centre shift; an open slice moves its shift into the
// gap beside that end (run_slice).
template <class T>
Anchor WindowSolver<T>::anchor_of(const Slice& s) const {
  if (!std::isfinite(s.hi.s)) return Anchor::Lower;  // windows have at least one finite end
  if (!std::isfinite(s.lo.s)) return Anchor::Upper;
  // Next to the band: the shift at the far end, away from the band's cluster.
  if (band_ >= 0 && s.hi.s == band_lo_) return s.lo.near ? Anchor::Centre : Anchor::Lower;
  if (band_ >= 0 && s.lo.s == band_hi_) return s.hi.near ? Anchor::Centre : Anchor::Upper;
  // Nothing below (above): a centre shift could be far from all of the slice's eigenvalues.
  if (s.lo.nu == 0) return s.hi.near ? Anchor::Centre : Anchor::Upper;
  if (s.hi.nu == n_) return s.lo.near ? Anchor::Centre : Anchor::Lower;
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

// Removes the flagged pool columns (keeps the column order of the survivors).
template <class T>
void WindowSolver<T>::compact_pool(const std::vector<char>& drop) {
  index_t w = 0;
  for (index_t q = 0; q < pool_.used; ++q) {
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

// A multiple eigenvalue among the pairs found inside the slice (two or more within kEqualWidth,
// not tried before): its copies sit at one theta, and a single-vector Krylov run at a shift next to
// it - where it dominates the operator - recovers the missing copies through rounding, while a run
// at a distant shift does not (multiplet_rerun_shift). Returns its mean, or NaN.
template <class T>
double WindowSolver<T>::multiplet_shift(const Block<T>& sel, const Slice& s, std::vector<double>& tried) const {
  std::vector<double> l;
  for (double x : sel.lambda)
    if (x >= s.lo.s && x <= s.hi.s) l.push_back(x);
  std::sort(l.begin(), l.end());
  double best = std::numeric_limits<double>::quiet_NaN();
  std::size_t best_size = 1;
  for (std::size_t i = 0; i < l.size();) {
    std::size_t j = i + 1;
    while (j < l.size() && l[j] - l[j - 1] <= kEqualWidth * std::max({scale_, std::abs(l[j])})) ++j;
    const double mean = std::accumulate(l.begin() + static_cast<std::ptrdiff_t>(i),
                                        l.begin() + static_cast<std::ptrdiff_t>(j), 0.0) /
                        static_cast<double>(j - i);
    bool seen = false;
    for (double t : tried) seen = seen || std::abs(t - mean) <= kEqualWidth * std::max(scale_, std::abs(mean));
    if (j - i > best_size && !seen) {
      best_size = j - i;
      best = mean;
    }
    i = j;
  }
  return best;
}

// The shift of a multiplet's rerun: kMultipletOffset of the way from the multiplet (mean) to its
// nearest other found eigenvalue (or the slice end) on the side with the larger gap. There the
// multiplet dominates the operator ~1/kMultipletOffset times, so rounding supplies its missing
// copies; a shift right on it (1e-8 relative) also resolves the spread of a numerically multiple
// cluster (a null space: eigenvalues ~1e-14 scale apart) into distinct theta that must converge
// one by one - measured on yee-pec-box-1e4 smallest 10: 215 solves on the multiplet, 87 at 1%;
// exact multiplets (bench_slicing_fix_probe lap4, cluster_mid, mult) took +3..8% solves.
template <class T>
double WindowSolver<T>::multiplet_rerun_shift(double mean, const Block<T>& sel, const Slice& s) const {
  double gl = kInf, gr = kInf;
  for (double x : sel.lambda) {
    const double d = x - mean;
    if (std::abs(d) <= kEqualWidth * std::max(scale_, std::abs(mean))) continue;  // a copy
    if (d > 0) {
      gr = std::min(gr, d);
    } else {
      gl = std::min(gl, -d);
    }
  }
  if (!std::isfinite(gl) && std::isfinite(s.lo.s)) gl = mean - s.lo.s;
  if (!std::isfinite(gr) && std::isfinite(s.hi.s)) gr = s.hi.s - mean;
  const double gap = std::isfinite(gr) && (gr >= gl || !std::isfinite(gl)) ? gr : -gl;
  return std::isfinite(gap) ? mean + kMultipletOffset * gap : mean;
}

// A partial cluster slice's copies (lambda, inside the slice) prove the whole cluster numerically
// equal when two counts at their mean -+ Delta (Delta = 16x their spread, >= 1e-9 relative, at
// most kEqualWidth relative) hold all of its eigenvalues: then any `want` of them are a valid answer.
template <class T>
bool WindowSolver<T>::cluster_verified(const Slice& s, const std::vector<double>& lambda, double& mean,
                                       double& delta) {
  if (lambda.empty()) return false;
  const auto [mn, mx] = std::minmax_element(lambda.begin(), lambda.end());
  mean = std::accumulate(lambda.begin(), lambda.end(), 0.0) / static_cast<double>(lambda.size());
  const double rel = std::max(scale_, std::abs(mean));
  delta = std::max(16.0 * (*mx - *mn), 1e-9 * rel);
  if (delta > kEqualWidth * rel) return false;
  const CountPoint lo = count(mean - delta, -1.0);
  const CountPoint hi = count(mean + delta, +1.0);
  return lo.nu == s.lo.nu && hi.nu == s.hi.nu;
}

// One slice: shift, Krylov run for exactly its count (or the copies wanted of a partial cluster),
// verification, certificate with deflated reruns. Pairs within an ambiguity margin delta outside
// the slice are kept too (an eigenvalue that close to a cut may be computed on the other side of
// it); deduplicate() resolves them. Returns false only for a partial cluster that turned out not
// to be numerically equal (the caller then solves it in full).
template <class T>
bool WindowSolver<T>::run_slice(const Slice& s, index_t id) {
  const SolverConfig& cfg = c_.cfg;
  const index_t count = s.count();
  const bool partial = s.want >= 0;
  double sigma = 0.0;
  bool detection_off = false;
  Which which = Which::LargestMagnitude;
  if (partial) {  // a shift at the cluster's bracket: the cluster dominates, rounding brings out copies
    sigma = factorize_near(c_, fac_, s.focus, scale_, detection_off, /*on_eigenvalue=*/true);
  } else {
    switch (anchor_of(s)) {
      case Anchor::Centre:
        sigma = factorize_near(c_, fac_, 0.5 * (s.lo.s + s.hi.s), scale_, detection_off);
        break;
      case Anchor::Upper: {  // eigenvalues below the shift: theta < 0, most negative = nearest below
        const CountPoint p = s.hi.near ? gap_point(s.hi, +1.0, plus_inf()) : s.hi;
        sigma = factorize_anchor(p, detection_off);
        which = Which::SmallestAlgebraic;
        break;
      }
      case Anchor::Lower: {  // eigenvalues above the shift: theta > 0, largest = nearest above
        const CountPoint p = s.lo.near ? gap_point(s.lo, -1.0, minus_inf()) : s.lo;
        sigma = factorize_anchor(p, detection_off);
        which = Which::LargestAlgebraic;
        break;
      }
    }
  }
  if (static_cast<index_t>(reports_.size()) <= id) reports_.resize(static_cast<std::size_t>(id + 1));
  SliceReport& sr = reports_[static_cast<std::size_t>(id)];
  sr = SliceReport{};
  sr.lower = s.lo.s;
  sr.upper = s.hi.s;
  sr.sigma = sigma;
  sr.which = which_name(which);
  sr.count_inertia = count;

  const double d_lo = std::isfinite(s.lo.s) ? 1e-9 * std::max(scale_, std::abs(s.lo.s)) : 0.0;
  const double d_hi = std::isfinite(s.hi.s) ? 1e-9 * std::max(scale_, std::abs(s.hi.s)) : 0.0;
  DefiniteRun<T> run;
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
    verify_and_rescue(c_, fac_, pencil_, sel, metrics, n2a_, n2b_,
                      [&]() { refactorize_at(c_, fac_, sigma, detection_off); });
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
  // Pairs still missing by the slice's count (partial cluster: by its want), up to the pairs
  // within delta of its ends.
  auto missing = [&]() {
    return partial ? std::max<index_t>(0, s.want - inside) : count - inside - near_out;
  };
  sr.nev = std::min(n_, s.required());
  // The P4 width: the blocktune rule was calibrated on nearest-target runs, not per-slice nev.
  const index_t p = fit_block_size<T>(c_, n_, sr.nev, 0, 4 * sr.nev, block_size_for(c_.cfg, sr.nev));
  if (id == 0) c_.rep.block_size = p;
  lanczos_run(c_, fac_, pencil_, sigma, which, sr.nev, p, static_cast<const DenseMatrix<T>*>(nullptr), run);
  select();
  // Certificate of the slice: deflated reruns while pairs are missing and the reruns find new
  // ones. A multiple eigenvalue among the found pairs gets its own shift (multiplet_shift).
  std::vector<double> tried;
  int stalls = 0;
  const int max_rounds = static_cast<int>(std::min<index_t>(kMaxRerunRounds, missing() + kExtraRerunRounds));
  for (int round = 0; missing() > 0 && round < max_rounds && stalls < kStallRounds; ++round) {
    const index_t miss = missing();
    keep_converged(run, n2a_, n2b_, cfg.tol);
    const index_t nev_rerun = std::min<index_t>(n_ - run.all.size(), miss + 2);
    if (nev_rerun < 1) break;
    const double m_sigma = multiplet_shift(sel, s, tried);
    event(c_, "slice_rerun", {{"slice", static_cast<std::int64_t>(id)}, {"missing", static_cast<std::int64_t>(miss)},
                              {"multiplet", m_sigma}});
    DenseMatrix<T> defl = run.all.x;
    const index_t d = defl.cols();
    const index_t p_rerun =
        fit_block_size<T>(c_, n_, nev_rerun, d, 3 * d + 4 * nev_rerun, rerun_block_size(cfg, p, miss));
    if (std::isfinite(m_sigma)) {
      tried.push_back(m_sigma);
      bool off = false;
      const double ms = factorize_near(c_, fac_, multiplet_rerun_shift(m_sigma, sel, s), scale_, off,
                                       /*on_eigenvalue=*/true);
      lanczos_run(c_, fac_, pencil_, ms, Which::LargestMagnitude, nev_rerun, p_rerun, &defl, run);
    } else {
      refactorize_at(c_, fac_, sigma, detection_off);
      lanczos_run(c_, fac_, pencil_, sigma, which, nev_rerun, p_rerun, &defl, run);
    }
    ++sr.reruns;
    select();
    stalls = missing() < miss ? 0 : stalls + 1;
  }
  if (partial) {
    // Any `want` copies of a numerically equal cluster: verified by two counts, then trimmed.
    std::vector<double> l;
    std::vector<index_t> in_cl;
    for (index_t j = 0; j < sel.size(); ++j)
      if (sel.lambda[j] >= s.lo.s && sel.lambda[j] <= s.hi.s) {
        l.push_back(sel.lambda[j]);
        in_cl.push_back(j);
      }
    double mean = 0.0, delta = 0.0;
    if (inside < s.want || !cluster_verified(s, l, mean, delta)) {
      restarts_ += run.restarts;
      applies_ += run.applies;
      return false;
    }
    warn(c_, "the target ends inside a cluster of " + std::to_string(count) + " numerically equal eigenvalues " +
                 fmt(mean) + " +- " + fmt(delta) + ": " + std::to_string(s.want) +
                 " of its copies are returned (a null space? see zero_filter_tol)");
    in_cl.resize(static_cast<std::size_t>(s.want));
    std::vector<PairMetrics> mk;
    for (index_t j : in_cl) mk.push_back(metrics[j]);
    sel = take_columns(sel, in_cl);
    metrics = std::move(mk);
    inside = s.want;
    near_in = near_out = 0;
  }
  sr.count_found = inside;
  sr.passed = partial ? inside == s.want : (inside - near_in <= count && count <= inside + near_out);
  sr.restarts = run.restarts;
  sr.op_applies = run.applies;
  restarts_ += run.restarts;
  applies_ += run.applies;
  event(c_, "slice", {{"slice", static_cast<std::int64_t>(id)}, {"lower", s.lo.s}, {"upper", s.hi.s}, {"sigma", sigma},
                      {"which", std::string(sr.which)}, {"count", static_cast<std::int64_t>(count)},
                      {"want", static_cast<std::int64_t>(s.want)}, {"found", static_cast<std::int64_t>(inside)},
                      {"passed", sr.passed}, {"reruns", static_cast<std::int64_t>(sr.reruns)},
                      {"restarts", static_cast<std::int64_t>(run.restarts)},
                      {"op_applies", static_cast<std::int64_t>(run.applies)}});
  if (!sr.passed)
    warn(c_, "slice [" + fmt(s.lo.s) + ", " + fmt(s.hi.s) + "]: inertia " + std::to_string(s.required()) +
                 " vs found " + std::to_string(inside));
  append_to_pool(sel, metrics, id);
  return true;
}

// Factorization at a slice end used as a one-sided shift: the count's own factorization when the
// object still holds it, else a refactorization with the same null-pivot setting; if that is
// flagged singular now (last-bit differences of the parallel factorization), the shift moves a
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
      recompute_products(pencil_.a(), pencil_.b(), grp);
      const index_t r = driver::rayleigh_ritz_definite(grp.x, grp.ax, grp.bx, grp.lambda, 1e-4);
      if (r < g) {
        const auto mr = block_metrics(grp, n2a_, n2b_);
        for (index_t q = 0; q < g; ++q) {
          const index_t col = order[i + q];
          if (q < r) {
            std::copy(grp.x.col(q), grp.x.col(q) + n_, pool_.x.col(col));
            pool_.lambda[col] = grp.lambda[q];
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
  if (removed > 0) compact_pool(drop);
  return removed;
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
  std::vector<Window> windows;
  const index_t k = t.k;
  bool short_warned = false;  // "fewer eigenvalues than k" already explained
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
    if (tot < k) {
      warn(c_, "only " + std::to_string(tot) + " eigenvalues lie outside the zero band (k = " + std::to_string(k) +
                   ")");
      short_warned = true;
    }
  } else {
    // Smallest: window (-inf, U] with nu(U) >= k; largest: [L, +inf) with n - nu(L) >= k. The
    // search stops refining once the extra pairs cost less than another count (slack). When the
    // k-th eigenvalue belongs to a cluster that counts cannot split, the window takes the whole
    // cluster - or, for a large cluster of numerically equal eigenvalues, a partial cluster
    // window that asks only for the copies still needed (any of them is a valid answer).
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
        short_warned = true;
        need = b.nu - a.nu;
      }
    }
    if (need > 0 && smallest) {
      const index_t g_lo = a.nu + need;
      const index_t g_hi = std::min(g_lo + slack, std::max(g_lo, b.nu - 1));
      Search r = Search::at(b);
      if (!(std::isfinite(b.s) && b.nu <= g_lo + slack))
        r = find_count(a, b, g_lo, std::min(g_lo + slack / 2, g_hi), g_hi,
                       std::isfinite(a.s) ? a.s : (est_.valid ? est_.lo1 : 0.0), Side::Above);
      if (r.cluster && r.cl_hi.nu - g_lo > std::max(kPartialClusterMin, need)) {
        const CountPoint below = gap_point(r.cl_lo, -1.0, a);
        if (below.nu > a.nu) windows.push_back({a, below});
        windows.push_back({below, r.p, g_lo - below.nu, 0.5 * (r.cl_lo.s + r.cl_hi.s)});
      } else {
        windows.push_back({a, r.p});
      }
    } else if (need > 0) {
      const index_t g_hi = b.nu - need;  // nu(L) <= n - need
      const index_t g_lo = std::max(g_hi - slack, std::min(g_hi, a.nu + 1));
      Search r = Search::at(a);
      if (!(std::isfinite(a.s) && a.nu >= g_hi - slack))
        r = find_count(a, b, g_lo, std::max(g_hi - slack / 2, g_lo), g_hi,
                       std::isfinite(b.s) ? b.s : (est_.valid ? est_.hi1 : 0.0), Side::Below);
      if (r.cluster && g_hi - r.cl_lo.nu > std::max(kPartialClusterMin, need)) {
        const CountPoint above = gap_point(r.cl_hi, +1.0, b);
        if (b.nu > above.nu) windows.push_back({above, b});
        windows.push_back({r.p, above, above.nu - g_hi, 0.5 * (r.cl_lo.s + r.cl_hi.s)});
      } else {
        windows.push_back({r.p, b});
      }
    }
    // A window from -inf up to the band (or from the band to +inf) would put a one-sided shift
    // next to the zero cluster; close it at a point beyond the end of the spectrum instead.
    for (auto& w : windows) {
      if (!std::isfinite(w.lo.s) && w.hi.s == band_lo_)
        w.lo = find_count(minus_inf(), w.hi, 0, 0, 0, est_.valid ? est_.lo1 : w.hi.s, Side::Below).p;
      if (!std::isfinite(w.hi.s) && w.lo.s == band_hi_)
        w.hi = find_count(w.lo, plus_inf(), n_, n_, n_, est_.valid ? est_.hi1 : w.lo.s, Side::Above).p;
    }
  }
  std::sort(windows.begin(), windows.end(), [](const Window& p, const Window& q) { return p.lo.s < q.lo.s; });
  index_t total = 0, required = 0;
  for (const auto& w : windows) {
    total += w.count();
    required += w.required();
  }
  event(c_, "windows",
        {{"windows", static_cast<std::int64_t>(windows.size())}, {"count", static_cast<std::int64_t>(total)},
         {"required", static_cast<std::int64_t>(required)}});

  InertiaCertificate& cert = rep.certificate;
  cert.performed = true;
  if (mode == TargetMode::Smallest) {
    cert.lower = -kInf;
    cert.upper = windows.empty() ? -kInf : windows.back().hi.s;
  } else if (mode == TargetMode::Largest) {
    cert.lower = windows.empty() ? kInf : windows.front().lo.s;
    cert.upper = kInf;
  }
  cert.count_inertia = required;
  if (band_ >= 0)
    cert.note = "zero band (" + fmt(band_lo_) + ", " + fmt(band_hi_) + ") excluded: " +
                std::to_string(rep.zero_modes_filtered) + " eigenvalue(s) by inertia";
  for (const auto& w : windows)
    if (w.want >= 0) {
      const std::string cl = "cluster [" + fmt(w.lo.s) + ", " + fmt(w.hi.s) + "] of " + std::to_string(w.count()) +
                             " numerically equal eigenvalues: " + std::to_string(w.want) + " required";
      cert.note = cert.note.empty() ? cl : cert.note + "; " + cl;
    }
  if (rep.zero_modes_filtered > 0)
    warn(c_, std::to_string(rep.zero_modes_filtered) + " zero modes filtered (|lambda| <= zero_filter_tol*scale)");
  if (mode != TargetMode::Interval && required < k && !short_warned)
    warn(c_, "the target window holds only " + std::to_string(required) + " eigenvalue(s), fewer than k = " +
                 std::to_string(k));

  Solution<T> sol;
  if (total == 0) {
    sol.eigenvectors.resize(n_, 0);
    cert.count_found = 0;
    cert.passed = true;
    if (cert.note.empty()) cert.note = "window is empty";
    rep.success = mode == TargetMode::Interval;
    return sol;
  }

  // ---- 2. Memory: the n x required output plus one slice's working set must fit the budget.
  {
    const double vec = static_cast<double>(n_) * sizeof(T);
    const double avail =
        cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
    // + up to 4 pairs per slice found just outside it (see step 3)
    const double out_bytes =
        vec * static_cast<double>(required + 4 * (total / std::max<index_t>(c_target_, 1) + 2) + 8);
    if (out_bytes > 0.9 * avail)
      HS_THROW(OutOfMemoryBudget, "the target holds " << required << " eigenvalues: their eigenvectors need "
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
  }

  // ---- 3. Slices, runs, merge. The pool holds the output plus the few pairs per slice found just
  //         outside it (resolved by deduplicate()), reserved once: growing it would copy it.
  for (const auto& w : windows) {
    const auto s = plan(w);
    slices_.insert(slices_.end(), s.begin(), s.end());
  }
  reserve_pool(required + 4 * static_cast<index_t>(slices_.size()) + 8);
  event(c_, "slices", {{"slices", static_cast<std::int64_t>(slices_.size())}});
  for (std::size_t i = 0; i < slices_.size(); ++i) {
    if (run_slice(slices_[i], static_cast<index_t>(i))) continue;
    // A partial cluster that is not numerically equal: all of its eigenvalues are needed - one
    // slice (counts cannot split it), whose output and working set must fit the budget.
    Slice& s = slices_[i];
    const double need_bytes = 8.0 * static_cast<double>(n_) * sizeof(T) * static_cast<double>(s.count());
    const double avail =
        cfg.memory_budget_gb * 1073741824.0 - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
    if (need_bytes > 0.9 * avail)
      HS_THROW(OutOfMemoryBudget, "the target ends inside a cluster of " << s.count() << " eigenvalues in [" << s.lo.s
                                                                         << ", " << s.hi.s
                                                                         << "] that are not numerically equal: "
                                                                         << need_bytes / 1073741824.0
                                                                         << " GiB needed (zero_filter_tol?)");
    warn(c_, "cluster [" + fmt(s.lo.s) + ", " + fmt(s.hi.s) + "] is not numerically multiple: all " +
                 std::to_string(s.count()) + " of its eigenvalues computed");
    for (auto& w : windows)
      if (w.lo.s == s.lo.s && w.hi.s == s.hi.s) w.want = -1;
    required += s.count() - s.want;
    cert.count_inertia = required;
    s.want = -1;
    run_slice(s, static_cast<index_t>(i));
  }
  const index_t removed = deduplicate();
  if (removed > 0) event(c_, "deduplicate", {{"removed", static_cast<std::int64_t>(removed)}});
  rep.slices = reports_;

  // ---- 4. Global certificate: distinct pairs per window vs its inertia count (partial cluster:
  //         at least the copies required, of which exactly that many are kept).
  auto window_of = [&](double l) -> index_t {
    if (in_band(l)) return -1;
    for (std::size_t w = 0; w < windows.size(); ++w)
      if (l >= windows[w].lo.s && l <= windows[w].hi.s) return static_cast<index_t>(w);
    return -1;
  };
  std::vector<index_t> in, found(windows.size(), 0);
  for (index_t j = 0; j < pool_.used; ++j) {
    const index_t w = window_of(pool_.lambda[j]);
    if (w < 0) continue;
    const Window& win = windows[static_cast<std::size_t>(w)];
    if (win.want >= 0 && found[static_cast<std::size_t>(w)] >= win.want) continue;
    ++found[static_cast<std::size_t>(w)];
    in.push_back(j);
  }
  cert.count_found = static_cast<index_t>(in.size());
  cert.passed = true;
  for (std::size_t w = 0; w < windows.size(); ++w) cert.passed = cert.passed && found[w] == windows[w].required();
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
  const index_t expected = mode == TargetMode::Interval ? total : std::min(k, required);
  const index_t want_out = mode == TargetMode::Interval ? total : k;
  bool all_ok = static_cast<index_t>(sol.eigenvalues.size()) == want_out;
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
  // Report: the checks' wall time (all rounds; the Certificate phase of this path), and the
  // exact counts of window targets, which set no method of their own.
  const auto finish = [&](Solution<T>&& sol) {
    InertiaCertificate& cert = c.rep.certificate;
    cert.seconds = c.clock.seconds(Phase::Certificate);
    if (cert.performed && cert.method == "none") {
      cert.method = certificate_method_name(CertificateMethod::Exact);
      cert.rigorous = cert.passed;
      cert.failure_probability = cert.passed ? 0.0 : 1.0;
      cert.statement = "double-precision Sylvester counts at every window end and slice cut: " +
                       std::to_string(cert.count_inertia) + " eigenvalue(s) required, " +
                       std::to_string(cert.count_found) + " distinct found pair(s)" +
                       (cert.passed ? std::string(": every required eigenvalue is found") : std::string());
    }
    return std::move(sol);
  };
  if (c.cfg.target.mode == TargetMode::Nearest) {
    double r0 = -1.0;
    Solution<T> sol = solve_nearest(c, pencil, fac, n2a, n2b, scale, &r0);
    if (!(r0 > 0)) return finish(std::move(sol));
    // The zero band interferes: k nearest outside it from windows [sigma - r, sigma + r] minus the
    // band (slices with shifts kept away from the band), r from inertia counts.
    warn(c, "zero band |lambda| <= " + fmt(c.cfg.zero_filter_tol * scale) +
                " reaches the nearest window: windowed solve with radius from inertia counts");
    if (c.cfg.certificate == CertificateMode::Randomized)
      warn(c, "certificate=randomized not used: the windowed solve counts exactly (its count points are shifts)");
    const index_t restarts = c.rep.restarts, applies = c.rep.op_applies;
    const index_t solves = c.rep.certificate.solves;
    c.rep.pairs.clear();
    c.rep.certificate = InertiaCertificate{};
    c.rep.certificate.solves = solves;
    c.rep.zero_modes_filtered = 0;
    WindowSolver<T> ws(c, pencil, fac, n2a, n2b, scale, spectrum);
    ws.set_nearest_radius(r0);
    sol = ws.solve();
    c.rep.restarts += restarts;
    c.rep.op_applies += applies;
    return finish(std::move(sol));
  }
  if (c.cfg.inertia_check == Toggle::Off)
    warn(c, "inertia_check=off ignored: interval / smallest / largest targets are built on inertia counts");
  else if (c.cfg.certificate == CertificateMode::Off)
    warn(c, "certificate=off ignored: interval / smallest / largest targets are built on inertia counts");
  else if (c.cfg.certificate == CertificateMode::Randomized)
    warn(c, "certificate=randomized applies to nearest targets: interval / smallest / largest targets count "
            "exactly (their count points are slice cuts and shifts)");
  WindowSolver<T> ws(c, pencil, fac, n2a, n2b, scale, spectrum);
  return finish(ws.solve());
}

#define HS_INSTANTIATE_SOLVE_DEFINITE(T)                                                                  \
  template Solution<T> solve_definite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                         double, double, double, const SpectrumEstimate&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_DEFINITE)
#undef HS_INSTANTIATE_SOLVE_DEFINITE

}  // namespace hs::driver
