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
//      no explicit deflation of the null space. Each side shift goes mid-way between the last
//      eigenvalue already held on that side (the cluster, or a plain run's outermost pair) and the
//      next one, located by a probe run (two pairs, loose tolerance) at the inner edge: a shift
//      right at the band edge leaves the cluster's |theta| = 1/band dominating the wanted ones by
//      lambda_first/band (measured: residuals stall at 1e-10..1e-9 for band = 1e-8 scale), and one
//      just inside a plain run's outermost pair (an eigenvalue) does the same; mid-gap the two
//      neighbours are balanced. An empty side shows up in the probe. Complex eigenvalues whose
//      real part lies between the side shifts are unwanted by both runs: recover_strip finds them
//      from the runs' Ritz values and adds a recovery run.
//      The union of all Ritz vectors is refined once on the projected pencil and the k nearest
//      sigma outside the band are selected. A side is rerun with more pairs until its converged
//      range covers the final radius, or it runs out of eigenvalues (exhausted: the unconverged
//      wanted Ritz values have the wrong sign).
// GHIEP supports target mode 'nearest' only: interval / smallest / largest need eigenvalue counts,
// which Sylvester inertia does not give for indefinite B.
//
// Block runs. Every Krylov-Schur run uses krylov_schur_general's block mode with the block size of
// ghiep_block_size (multi-RHS solves are much cheaper per vector than single ones); the one-sided
// runs keep the cluster at their unwanted end whatever the block size (p cluster directions enter
// a block Krylov space, none is wanted).
//
// Memory (DESIGN §3.6-3). A run keeps only its Schur basis (its Ritz vectors span the same space
// and are dropped at once); the factorization is released before every refinement (idle there; a
// later run sits at another shift, or is a rerun whose own solves outweigh one refactorization);
// the projected pencil is formed in column blocks, without n x m images A W, B W; candidates are
// refined in batches, kept as coefficients in the refinement basis, and vectors are formed only
// for the selected pairs, after the runs' bases are freed.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
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

/// Pairs with a backward error above this are not trusted as eigenpairs (selection fills a
/// shortfall with them, reported as not accepted).
constexpr double kTrustworthy = 1e-6;
/// |lambda| <= kZeroBand * scale is "numerically zero" (zero_filter_tol off): the cluster band.
constexpr double kZeroBand = kGhiepZeroBand;
/// Restart budget of the first attempt of a one-sided run: enough for well-separated runs (2..5
/// restarts measured), small enough that a side with fewer eigenvalues than requested (its extra
/// wanted Ritz values chase the dense theta ~ 0 end and never converge) costs little. Well-placed
/// runs took 2..6 restarts in the tests and on the waveguide cases.
constexpr index_t kSideFirstRestarts = 16;
/// Probe locating the first eigenvalue beyond the band on one side: one pair to this relative
/// Ritz tolerance (placement only needs a few digits), within this many restarts.
constexpr double kProbeTol = 1e-6;
constexpr index_t kProbeRestarts = 4;
/// Ritz value of a one-sided run counted as a complex eigenvalue estimate (recover_strip): the
/// imaginary part of lambda exceeds this fraction of |lambda| (unconverged estimates of real
/// eigenvalues stay far below it); estimates within kStripMatch relative are the same eigenvalue.
constexpr double kStripImag = 1e-4;
constexpr double kStripMatch = 1e-3;
/// Estimates of the same eigenvalue from two Krylov spaces agree to this relative distance.
constexpr double kStripConfirm = 1e-2;

/// Candidates refined per batch (refine_union): their vectors and A, B images are formed together,
/// so spmm streams A and B once per batch (real T: 2 * kBatch real columns, the width spmm
/// streams at once). Batch memory: 6 kBatch n-vectors of T.
constexpr index_t kBatch = 4;
/// Basis columns per block when the projected pencil W^H A W, W^H B W is formed (the width spmm
/// streams A and B for at once); memory: n x kProjectBatch of T.
constexpr index_t kProjectBatch = 8;
/// Block-size rule of the GHIEP runs (ghiep_block_size): p = kAutoBlock for orders from
/// kBlockMinOrder on (below, the solves are cheap next to the basis work: totals of ~0.1 s at
/// n = 1e4, where p = 2 saves at most a few 10 ms) and runs with at least kBlockMinNev wanted pairs
/// (a block needs pairs to fill it; the probes, nev = 2, stay single-vector).
constexpr index_t kAutoBlock = 2;
constexpr index_t kBlockMinOrder = 20000;
constexpr index_t kBlockMinNev = 4;

/// Largest configured block size used: wider blocks only add basis work, and a block too wide for
/// the space makes krylov_schur_general fall back to one exact cycle over all of it.
constexpr index_t kMaxConfiguredBlock = 64;
/// Runs with at most this many wanted pairs (the side probes, nev = 2) stay single-vector whatever
/// is configured: they only locate a shift, and a block would multiply their basis for nothing.
constexpr index_t kMaxProbeNev = 2;
/// Share of the memory left in the budget (memory_budget_gb - current RSS) a run's block basis may
/// take (fit_block_size); the rest is headroom for the solves' own workspace.
constexpr double kBlockMemoryShare = 0.9;

}  // namespace

// Krylov block size of a GHIEP run (internal.hpp). Measurements (2 threads, cores 28-29, medians of interleaved repetitions). One multi-RHS solve
/// of p columns costs much less than p single solves (bench/ghiep_block.cpp --mode ks:
/// wg-square-1e5 20.8 / 11.5 / 7.9 / 5.5 ms per vector for p = 1 / 2 / 4 / 8, wg-square-1e6
/// 277 / 150 / 105 / 65 ms), while a block Krylov space needs more vectors for distinct
/// eigenvalues (wg-square-1e5, k = 18: 57 / 62 / 84 / 128 applications) - and many more in the
/// one-sided runs next to the zero cluster, where every block brings p cluster directions back
/// (wg-square-1e5 at sigma = 0, whole solve: 284 / 400 / 620 / 855). End to end, seconds for
/// p = 1 / 2 / 4 / 8 (3 repetitions; bench/ghiep_fix_e2e.cpp on the data cases, with --sigma 0 for
/// the one-sided rows, bench/ghiep_block.cpp --mode e2e on the synthetic ones; p forced through
/// SolverConfig::block_size):
///   wg-square-1e5     2.09 / 1.56 / 1.57 / 1.69    wg-square-1e5, sigma 0    9.9 / 8.6 / 14.6 / 10.1
///   wg-halfslab-1e5   3.89 / 2.46 / 3.10 / 3.40    wg-halfslab-1e5, sigma 0  14.5 / 11.4 / 10.6 / 12.7
/// and on the five synthetic pencils (syn-ghiep-{wg,wg0}-{20480,102400}, syn-ghiep-cplx-102400:
/// banded, cheap solves, zero cluster, complex pairs) p = 2 was 1.06 - 1.14x, p = 4 0.97 - 1.12x
/// and p = 8 0.79 - 1.12x as fast as p = 1. p = 2 never lost against p = 1 (1.06 - 1.58x over the
/// nine cases); p = 4 lost up to 1.5x (0.68x). Every solve succeeded (eta <= 3e-12), with the same
/// eigenvalues for every p (relative differences <= 3e-10, as between repetitions of p = 1: the
/// L0-OpenMP solves are not bitwise reproducible). n ~ 1e4 (wg-square-pert-1e4, wg-halfslab-1e4):
/// p = 2 1.03 - 1.19x, but on totals of 0.11 - 0.16 s (10 - 20 ms, the noise level of these
/// runs), so the rule keeps p = 1 below kBlockMinOrder.
index_t ghiep_block_size(const SolverConfig& cfg, index_t n, index_t nev) {
  const index_t configured = cfg.block_size;
  if (configured > 0) {
    if (nev <= kMaxProbeNev) return 1;
    return std::max<index_t>(1, std::min({configured, nev, kMaxConfiguredBlock, n / 16}));
  }
  return n >= kBlockMinOrder && nev >= kBlockMinNev ? kAutoBlock : 1;
}

// Memory plan of the GHIEP path (internal.hpp). The first run is the plain run at sigma (or, with
// sigma in the zero band, a one-sided run with as many pairs): its peak - Krylov basis plus Schur
// basis, or Schur basis plus vectors - comes from krylov_schur_general_peak_columns with the block
// size of ghiep_block_size, plus the operator's refinement workspace (2p) for single / BLR factors.
// While later runs (probes, one-sided runs, reruns) hold the factorization, the solve keeps the
// plain run's Schur basis and the union refinement basis (refine_union: m columns each, m = the
// run's materialized Schur vectors); bases of further runs are not reserved - fit_block_size halves
// their block size to what the budget leaves. The outputs are formed after the factorization is
// released (release_factorization), so they need no reservation beside the factors.
template <class T>
index_t ghiep_reserve_columns(const SolverConfig& cfg, index_t n) {
  if (n < 1) return 0;
  KrylovSchurOptions<T> ko;
  ko.nev = std::clamp<index_t>(cfg.target.k, 1, n);
  // An ncv the run would reject is reported by the run itself; plan with the automatic one.
  const index_t min_ncv = std::min(n, ko.nev + (is_complex_v<T> ? 1 : 2));
  ko.ncv = cfg.ncv >= min_ncv ? cfg.ncv : 0;
  ko.block_size = ghiep_block_size(cfg, n, ko.nev);
  const bool refinement = cfg.factor_precision == FactorPrecision::Single || cfg.blr == Toggle::On;
  const index_t m = std::min(n, ko.nev + (is_complex_v<T> ? 0 : 1));
  return krylov_schur_general_peak_columns<T>(n, ko) + (refinement ? 2 * ko.block_size : 0) + 2 * m;
}

namespace {

/// ko.block_size, halved while the run's peak - krylov_schur_general_peak_columns n-vectors, plus
/// the operator's refinement workspace (2 n p, with refinement only) - exceeds kBlockMemoryShare of
/// the memory the budget leaves beside everything the solve holds now (current RSS: the live
/// factorization, earlier runs' Schur bases), DESIGN §3.6-3. A configured p = 64 at n = 1e6 would
/// otherwise hold a 512-column basis (4.1 GB, real) that the solver's memory plan never reserved.
/// Warns (once per distinct reduction) when p is reduced.
template <class T>
index_t fit_block_size(Ctx& c, index_t n, KrylovSchurOptions<T> ko, bool refinement) {
  const index_t wanted = ko.block_size;
  if (wanted <= 1) return wanted;
  const double vec = static_cast<double>(n) * static_cast<double>(sizeof(T));
  const double avail = c.cfg.memory_budget_gb * 1073741824.0 -
                       static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
  auto bytes = [&](index_t q) {
    ko.block_size = q;
    return vec * static_cast<double>(krylov_schur_general_peak_columns<T>(n, ko) + (refinement ? 2 * q : 0));
  };
  index_t p = wanted;
  while (p > 1 && bytes(p) > kBlockMemoryShare * avail) p /= 2;
  if (p < wanted) {
    const std::string msg = "GHIEP Krylov block size reduced from " + std::to_string(wanted) + " to " +
                            std::to_string(p) + " by the memory budget";
    if (std::find(c.rep.warnings.begin(), c.rep.warnings.end(), msg) == c.rep.warnings.end()) warn(c, msg);
  }
  return p;
}

template <class T>
struct KsRun {
  double sigma = 0.0;
  Which which = Which::LargestMagnitude;
  index_t nev = 0;
  index_t block_size = 1;  ///< Krylov block width p of the run (after the memory fit)
  /// The Krylov-Schur result without its Ritz vectors: the refinement works on the Schur basis
  /// (same span, orthonormal), so ks_run drops the n x m complex vectors right away.
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
  ko.block_size = ghiep_block_size(cfg, pencil.order(), ko.nev);
  ko.block_size = fit_block_size(c, pencil.order(), ko, refine.max_steps > 0);
  run.block_size = ko.block_size;
  {
    PhaseScope phase(c.clock, Phase::Iterate);
    run.res = krylov_schur_general<T>(op, which, ko);
  }
  run.res.vectors = DenseMatrix<cplx>();
  event(c, "krylov_schur", {{"sigma", run.sigma}, {"nev", static_cast<std::int64_t>(ko.nev)},
                            {"block_size", static_cast<std::int64_t>(ko.block_size)},
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
  index_t col = 0;  ///< column of Refined::coef
};

/// Refined candidate pairs: the vector of candidate c is basis * coef(:, c.col), unit 2-norm (real
/// for a real-classified candidate of real T).
template <class T>
struct Refined {
  std::vector<Cand> cands;
  DenseMatrix<T> basis;    ///< n x mb
  DenseMatrix<cplx> coef;  ///< mb x (candidates)
};

/// (c + c^H) / 2 in place for a small square matrix (diagonal made real).
template <class T>
void make_hermitian_small(DenseMatrix<T>& c) {
  const index_t m = c.rows();
  for (index_t j = 0; j < m; ++j) {
    c(j, j) = T(std::real(c(j, j)));
    for (index_t i = j + 1; i < m; ++i) {
      const T h = 0.5 * (c(i, j) + conj_if(c(j, i)));
      c(i, j) = h;
      c(j, i) = conj_if(h);
    }
  }
}

/// Eigenvalues w and eigenvectors y of a small square matrix: complex Schur form (zgees) and the
/// back-transformed triangular eigenvectors (ztrevc).
template <class T>
void small_eig(const DenseMatrix<T>& a, std::vector<cplx>& w, DenseMatrix<cplx>& y) {
  const index_t m = a.rows();
  const auto fm = static_cast<lapack::fint>(m);
  DenseMatrix<cplx> t(m, m);
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < m; ++i) t(i, j) = cplx(a(i, j));
  y.resize(m, m);
  w.assign(static_cast<std::size_t>(m), cplx(0.0));
  lapack::gees(fm, t.data(), fm, w.data(), y.data(), fm);
  lapack::trevc_backtransform(fm, t.data(), fm, y.data(), fm);
}

/// Rayleigh-Ritz for the indefinite pencil on span(W), W (n x m) Euclidean orthonormal: the
/// eigenpairs (mu, y) of the small pencil (W^H A W, W^H B W), computed as eig(H^{-1} G) via LU and
/// a complex Schur form, with G and H formed in column
/// blocks of kProjectBatch: no n x m image A W or B W is materialized (DESIGN §3.6-3). Ritz vectors
/// are W y. Returns false when H is numerically singular (pivot ratio < 1e-12).
template <class T>
bool projected_pairs(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const DenseMatrix<T>& w,
                     std::vector<cplx>& mu, DenseMatrix<cplx>& y) {
  const index_t n = w.rows(), m = w.cols();
  mu.clear();
  y.resize(m, m);
  if (m == 0) return true;
  const auto fn = static_cast<lapack::fint>(n), fm = static_cast<lapack::fint>(m);
  DenseMatrix<T> g(m, m), h(m, m), img(n, std::min(kProjectBatch, m));
  for (index_t c0 = 0; c0 < m; c0 += kProjectBatch) {
    const index_t nb = std::min(kProjectBatch, m - c0);
    const auto fb = static_cast<lapack::fint>(nb);
    spmm(a, w.col(c0), n, img.data(), n, nb);
    lapack::gemm('C', 'N', fm, fb, fn, T(1), w.data(), fn, img.data(), fn, T(0), g.col(c0), fm);
    if (b) {
      spmm(*b, w.col(c0), n, img.data(), n, nb);
      lapack::gemm('C', 'N', fm, fb, fn, T(1), w.data(), fn, img.data(), fn, T(0), h.col(c0), fm);
    } else {
      lapack::gemm('C', 'N', fm, fb, fn, T(1), w.data(), fn, w.col(c0), fn, T(0), h.col(c0), fm);
    }
  }
  make_hermitian_small(g);
  make_hermitian_small(h);
  std::vector<lapack::fint> ipiv(static_cast<std::size_t>(m));
  if (lapack::getrf(fm, h.data(), fm, ipiv.data()) != 0) return false;
  double umax = 0.0, umin = kInf;
  for (index_t i = 0; i < m; ++i) {
    umax = std::max(umax, std::abs(h(i, i)));
    umin = std::min(umin, std::abs(h(i, i)));
  }
  if (!(umin > 1e-12 * umax)) return false;
  lapack::getrs(fm, fm, h.data(), fm, ipiv.data(), g.data(), fm);
  small_eig(g, mu, y);
  return true;
}

/// The factorization is idle while the union is refined (and usually for good: a later run sits at
/// another shift, or a refactorization at a rerun's shift is cheap next to that rerun's solves):
/// every caller of refine_union frees its factors first, so that the refinement's n-sized buffers
/// do not stack on top of them. The next ks_run refactorizes (factorize_near).
template <class T>
void release_factorization(Ctx& c, SparseFactorization<T>& fac) {
  if (fac.release_factors()) c.held = FactorHeld{};
}

// Refinement on the projected pencil over the union of the runs' computed invariant subspaces,
// then real/complex classification (DESIGN §3.6-10) with metrics for every candidate. The basis is
// the union of the runs' Schur bases (each spans the same space as its run's Ritz vectors, Re/Im
// parts included for real T), orthonormalized with dependent columns dropped. Memory: the basis
// (sum of the runs' m columns), the batch buffers (6 kBatch n-vectors) and 9 n-vectors of
// per-candidate work - no n x m images A W, B W or vector sets (DESIGN §3.6-3).
template <class T>
Refined<T> refine_union(Ctx& c, const ShiftedPencil<T>& pencil, const std::vector<KsRun<T>>& runs, double n2a,
                        double n2b, double scale) {
  const SolverConfig& cfg = c.cfg;
  const CsrMatrix<T>& a = pencil.a();
  const CsrMatrix<T>* b = pencil.b();
  const index_t n = pencil.order();
  index_t mr = 0;
  for (const auto& r : runs) mr += r.res.schur_basis.cols();
  auto concatenate = [&](DenseMatrix<T>& out) {
    out.resize(n, mr);
    index_t col0 = 0;
    for (const auto& r : runs) {
      const DenseMatrix<T>& q = r.res.schur_basis;
      std::copy(q.data(), q.data() + n * q.cols(), out.col(col0));
      col0 += q.cols();
    }
  };
  Refined<T> out;
  concatenate(out.basis);
  driver::orthonormalize_columns(out.basis, 1e-8);
  std::vector<cplx> mu;
  DenseMatrix<cplx> y;
  if (!projected_pairs(a, b, out.basis, mu, y)) {
    warn(c, "projected pencil singular; using the Krylov-Schur pairs without refinement");
    // The Krylov-Schur pairs: eigenpairs (theta, z) of each run's Schur form S, x = Q1 z.
    concatenate(out.basis);
    mu.clear();
    y.resize(mr, mr);
    index_t col0 = 0;
    for (const auto& r : runs) {
      std::vector<cplx> th;
      DenseMatrix<cplx> z;
      small_eig(r.res.schur_form, th, z);
      for (index_t i = 0; i < static_cast<index_t>(th.size()); ++i) {
        mu.push_back(cplx(r.sigma) + 1.0 / th[i]);
        std::copy(z.col(i), z.col(i) + z.rows(), y.col(col0 + i) + col0);
      }
      col0 += static_cast<index_t>(th.size());
    }
  }

  const index_t mb = out.basis.cols(), mc = static_cast<index_t>(mu.size());
  out.coef.resize(mb, mc);
  // Batch buffers: real T holds the real and imaginary parts of kBatch vectors side by side.
  constexpr index_t kParts = is_complex_v<T> ? 1 : 2;
  DenseMatrix<T> ys(mb, kParts * kBatch), xs(n, kParts * kBatch), axs(n, kParts * kBatch),
      bxs(n, kParts * kBatch);
  std::vector<cplx> x(static_cast<std::size_t>(n)), ax(x.size()), bx(x.size());
  std::vector<T> xr(x.size()), axr(x.size()), bxr(x.size());
  for (index_t j0 = 0; j0 < mc; j0 += kBatch) {
    const index_t nb = std::min(kBatch, mc - j0), cols = kParts * nb;
    for (index_t q = 0; q < nb; ++q)
      for (index_t i = 0; i < mb; ++i) {
        if constexpr (is_complex_v<T>) {
          ys(i, q) = y(i, j0 + q);
        } else {
          ys(i, q) = y(i, j0 + q).real();
          ys(i, nb + q) = y(i, j0 + q).imag();
        }
      }
    linalg::multiply(n, mb, cols, out.basis.data(), n, ys.data(), mb, xs.data(), n);
    spmm(a, xs.data(), n, axs.data(), n, cols);
    if (b) {
      spmm(*b, xs.data(), n, bxs.data(), n, cols);
    } else {
      std::copy(xs.data(), xs.data() + n * cols, bxs.data());
    }
    for (index_t q = 0; q < nb; ++q) {
      const index_t j = j0 + q;
      if (!std::isfinite(std::abs(mu[j]))) continue;
      // The candidate vector x = basis y_j (and its images), scaled to unit 2-norm. (The loops
      // over n are element-wise, so OpenMP keeps the results independent of the thread count;
      // the norms are linalg's deterministic reductions.)
      double xn = linalg::norm2(n, xs.col(q));
      if constexpr (!is_complex_v<T>) xn = std::hypot(xn, linalg::norm2(n, xs.col(nb + q)));
      if (!(xn > 0.0)) continue;
      const double inv_xn = 1.0 / xn;
#pragma omp parallel for schedule(static)
      for (index_t i = 0; i < n; ++i) {
        if constexpr (is_complex_v<T>) {
          x[i] = xs(i, q) * inv_xn, ax[i] = axs(i, q) * inv_xn, bx[i] = bxs(i, q) * inv_xn;
        } else {
          x[i] = cplx(xs(i, q), xs(i, nb + q)) * inv_xn;
          ax[i] = cplx(axs(i, q), axs(i, nb + q)) * inv_xn;
          bx[i] = cplx(bxs(i, q), bxs(i, nb + q)) * inv_xn;
        }
      }
      const auto pm_c = driver::pair_metrics(ax.data(), bx.data(), x.data(), n, mu[j], n2a, n2b);
      const bool small_imag = std::abs(mu[j].imag()) <= cfg.complex_tol * std::max(std::abs(mu[j]), scale);
      Cand cd{mu[j], false, pm_c, j};
      // Real candidate: the phase-normalised vector's real part (real T; complex T: the vector
      // itself) and lambda = Re(mu). A and B are applied linearly: A Re(phase x) = Re(phase A x).
      // The phase comes from the largest entry (first one of equal magnitude).
      index_t imax = 0;
      double vmax = std::norm(x[0]);
      for (index_t i = 1; i < n; ++i) {
        const double v = std::norm(x[i]);
        if (v > vmax) vmax = v, imax = i;
      }
      const cplx phase = vmax > 0 ? std::conj(x[imax]) / std::abs(x[imax]) : cplx(1);
#pragma omp parallel for schedule(static)
      for (index_t i = 0; i < n; ++i) {
        if constexpr (is_complex_v<T>) {
          xr[i] = phase * x[i], axr[i] = phase * ax[i], bxr[i] = phase * bx[i];
        } else {
          xr[i] = (phase * x[i]).real(), axr[i] = (phase * ax[i]).real(), bxr[i] = (phase * bx[i]).real();
        }
      }
      const double rn = linalg::norm2(n, xr.data());
      const double lam_r = mu[j].real();
      PairMetrics pm_r;
      pm_r.backward_error = kInf;
      if (rn > 0.0) {
        const double inv_rn = 1.0 / rn;
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < n; ++i) xr[i] *= inv_rn, axr[i] *= inv_rn, bxr[i] *= inv_rn;
        pm_r = driver::pair_metrics(axr.data(), bxr.data(), xr.data(), n, lam_r, n2a, n2b);
      }
      // Real unless the imaginary part is significant AND the real projection fails while the complex pair passes.
      const bool complex_genuine = !small_imag && pm_r.backward_error > cfg.tol && pm_c.backward_error <= pm_r.backward_error;
      cplx* cj = out.coef.col(j);
      if (!complex_genuine) {
        cd.real = true;
        cd.lambda = cplx(lam_r, 0.0);
        cd.pm = pm_r;
        // x_real = basis Re(phase y_j) / (xn rn) (real T; complex T: phase y_j / (xn rn)).
        for (index_t i = 0; i < mb; ++i) {
          const cplx v = phase * y(i, j) / (xn * rn);
          cj[i] = is_complex_v<T> ? v : cplx(v.real());
        }
      } else {
        for (index_t i = 0; i < mb; ++i) cj[i] = y(i, j) / xn;
      }
      out.cands.push_back(cd);
    }
  }
  return out;
}

/// The vectors basis * coef(:, cols) of refined candidates, n x cols.size() (n = the pencil order,
/// also for an empty selection): of T for real candidates (the coefficients' imaginary parts are
/// zero for real T), or complex.
template <class T, class V>
DenseMatrix<V> candidate_vectors(const Refined<T>& ref, const std::vector<index_t>& cols, index_t n) {
  const index_t mb = ref.basis.cols(), k = static_cast<index_t>(cols.size());
  DenseMatrix<V> out(n, k);
  if (k == 0 || mb == 0) return out;
  constexpr index_t kParts = is_complex_v<V> && !is_complex_v<T> ? 2 : 1;
  DenseMatrix<T> c(mb, kParts * k), x(kParts == 2 ? n : 0, kParts == 2 ? 2 * k : 0);
  for (index_t q = 0; q < k; ++q)
    for (index_t i = 0; i < mb; ++i) {
      const cplx v = ref.coef(i, cols[q]);
      if constexpr (is_complex_v<T>) {
        c(i, q) = v;
      } else {
        c(i, q) = v.real();
        if (kParts == 2) c(i, k + q) = v.imag();
      }
    }
  if constexpr (kParts == 2) {
    linalg::multiply(n, mb, 2 * k, ref.basis.data(), n, c.data(), mb, x.data(), n);
    for (index_t q = 0; q < k; ++q)
      for (index_t i = 0; i < n; ++i) out(i, q) = V(x(i, q), x(i, k + q));
  } else {
    if constexpr (std::is_same_v<V, T>) {
      linalg::multiply(n, mb, k, ref.basis.data(), n, c.data(), mb, out.data(), n);
    }
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

/// Ritz values of a Krylov space (shift sigma) kept as evidence for recover_strip.
struct RitzSet {
  double sigma = 0.0;
  std::vector<cplx> theta;
};

// Complex eigenvalues x +- iy whose real part lies between the two side shifts are on the unwanted
// side of both one-sided runs (Re theta has the wrong sign for each), and a run near the cluster
// sees only the cluster: without care they are dropped while the run reports success. Being
// extremes of the operator's spectrum in modulus, they show up among the Ritz values of every
// Krylov space built near the cluster (probes and side runs); an estimate seen in one space only
// is a Ritz value of a mixture (unconverged ones of a non-normal operator are complex in general).
// Each estimate confirmed by two spaces, nearer sigma than the selection's radius and not yet
// found gets a recovery run, one-sided at a quarter of its real part: there Re theta of the pair
// has the wanted sign and the cluster's theta the other one; nev covers the real eigenvalues
// ranked before the pair. The union is then refined and selected again. An estimate is spurious
// when the recovery run converges every eigenvalue ranked before it without finding it. Returns
// false when an estimate was neither found nor excluded (or its real part lies inside the band) -
// reported as a warning.
template <class T>
bool recover_strip(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double scale, double n2a,
                   double n2b, double band, double band_place, std::vector<KsRun<T>>& runs,
                   const std::vector<RitzSet>& probes, Refined<T>& ref, std::vector<Cand>& sel, index_t& good,
                   index_t& in_band) {
  const SolverConfig& cfg = c.cfg;
  const Target& t = cfg.target;
  const index_t k = t.k;
  double r_sel = kInf;
  if (static_cast<index_t>(sel.size()) >= k && good >= k) {
    r_sel = 0.0;
    for (const auto& cd : sel) r_sel = std::max(r_sel, std::abs(cd.lambda - cplx(t.sigma)));
  }
  auto found = [&](const cplx& l) {
    for (const auto& cd : ref.cands)
      if (!cd.real && cd.pm.backward_error <= cfg.tol && std::abs(cd.lambda - l) <= kStripMatch * std::abs(l))
        return true;
    return false;
  };
  // Estimates near the cluster (relative to the side shifts) are mixtures with its copies.
  double near_cluster = kInf;
  for (const auto& run : runs)
    if (run.which != Which::LargestMagnitude) near_cluster = std::min(near_cluster, kStripMatch * std::abs(run.sigma));
  if (!std::isfinite(near_cluster)) return true;  // no side runs
  std::vector<RitzSet> spaces = probes;
  for (const auto& run : runs) spaces.push_back({run.sigma, run.res.ritz_all});
  std::vector<cplx> est;
  std::vector<std::size_t> origin;  // space of each estimate
  for (std::size_t i = 0; i < spaces.size(); ++i)
    for (const cplx& th : spaces[i].theta) {
      if (th == cplx(0.0)) continue;
      const cplx l = cplx(spaces[i].sigma) + 1.0 / th;
      if (std::abs(l.imag()) > kStripImag * std::abs(l) && std::abs(l) > std::max(band_place, near_cluster)) {
        est.push_back(l);
        origin.push_back(i);
      }
    }
  auto seen = [&](const cplx& v, std::size_t other_than) {  // in a space other than `other_than`
    for (std::size_t q = 0; q < est.size(); ++q)
      if (origin[q] != other_than && std::abs(est[q] - v) <= kStripConfirm * std::abs(v)) return true;
    return false;
  };
  std::vector<cplx> strip;
  for (std::size_t e = 0; e < est.size(); ++e) {
    const cplx l = est[e];
    if (l.imag() < 0 || std::abs(l - cplx(t.sigma)) >= r_sel || found(l)) continue;
    if (!seen(l, origin[e])) continue;
    // Complex eigenvalues of a Hermitian pencil come in conjugate pairs; for complex arithmetic the
    // Ritz values do not, so the partner must show too.
    if (is_complex_v<T> && !seen(std::conj(l), std::numeric_limits<std::size_t>::max())) continue;
    bool dup = false;
    for (const cplx& q : strip) dup = dup || std::abs(q - l) <= kStripMatch * std::abs(l);
    if (!dup) strip.push_back(l);
  }
  if (strip.empty()) return true;
  bool resolved = true, ran = false;
  for (const cplx& l : strip) {
    const double x = l.real(), sgn = x > 0 ? 1.0 : -1.0;
    double sc = 0.25 * x;
    if (std::abs(sc) <= 2.0 * band_place) sc = 0.5 * x;
    if (std::abs(sc) <= 2.0 * band_place) {
      warn(c, "complex eigenvalue estimate " + fmt(x) + (l.imag() >= 0 ? "+" : "") + fmt(l.imag()) +
                  "i has its real part inside the zero band: not resolved");
      resolved = false;
      continue;
    }
    const Which w = x > 0 ? Which::LargestAlgebraic : Which::SmallestAlgebraic;
    // Real eigenvalues ranked before the pair at sc: sgn * theta_r > sgn * Re theta_pair.
    const double re_th = (x - sc) / std::norm(l - cplx(sc));
    index_t ahead = 0;
    std::vector<double> reals;
    for (const auto& cd : ref.cands)
      if (cd.real && cd.pm.backward_error <= kTrustworthy) reals.push_back(cd.lambda.real());
    for (const auto& run : runs)
      for (const cplx& th : run.res.ritz_all)
        if (th != cplx(0.0) && std::abs(th.imag()) <= kStripImag * std::abs(th))
          reals.push_back(run.sigma + 1.0 / th.real());
    std::sort(reals.begin(), reals.end());
    for (std::size_t i = 0; i < reals.size(); ++i) {
      if (i > 0 && std::abs(reals[i] - reals[i - 1]) <= kStripMatch * std::abs(reals[i])) continue;  // same value twice
      const double d = reals[i] - sc;
      if (sgn * d > 0 && sgn / d > sgn * re_th) ++ahead;
    }
    index_t nev = ahead + 4;
    bool got = false, absent = false;
    for (int attempt = 0; attempt < 2 && !got && !absent && nev <= std::max<index_t>(4 * k, 64); ++attempt, nev *= 2) {
      KsRun<T> run = ks_run(c, fac, pencil, scale, sc, w, nev, cfg.max_restarts);
      // Every eigenvalue ranked before the estimate converged without it: there is none there (a
      // Ritz value of a mixture, e.g. an unconverged one of complex arithmetic).
      absent = run.res.converged && run.res.nconv > 0 && sgn * run.res.theta[run.res.nconv - 1].real() < sgn * re_th;
      event(c, "strip_recovery",
            {{"re", x}, {"im", l.imag()}, {"sigma", run.sigma}, {"nev", static_cast<std::int64_t>(nev)},
             {"nconv", static_cast<std::int64_t>(run.res.nconv)}});
      SliceReport sr;
      sr.lower = x > 0 ? run.sigma : -kInf;
      sr.upper = x > 0 ? kInf : run.sigma;
      sr.sigma = run.sigma;
      sr.which = x > 0 ? "largest_algebraic" : "smallest_algebraic";
      sr.nev = run.nev;
      sr.count_found = run.res.nconv;
      sr.reruns = attempt;
      sr.restarts = run.res.restarts;
      sr.op_applies = run.res.op_applies;
      sr.passed = run.res.converged;
      c.rep.slices.push_back(sr);
      runs.push_back(std::move(run));
      ran = true;
      release_factorization(c, fac);
      ref = refine_union(c, pencil, runs, n2a, n2b, scale);
      got = found(l);
      absent = absent && !got;
    }
    if (!got && !absent) {
      warn(c, "complex eigenvalue estimate " + fmt(x) + "+" + fmt(l.imag()) +
                  "i between the one-sided shifts: recovery runs neither found nor excluded it");
      resolved = false;
    }
  }
  if (ran) sel = select_nearest(ref.cands, t.sigma, k, band, cfg.tol, good, in_band);
  return resolved;
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
  bool strip_unresolved = false;
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
    release_factorization(c, fac);
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
    // edge finds the next eigenvalues; the shift goes mid-way between the last eigenvalue held on
    // that side (the cluster, or the plain run's outermost pair) and the next distinct one.
    index_t need_side[2] = {k, k};
    double inner[2] = {band_place, -band_place};
    // Outermost eigenvalue a plain run already holds on each side beyond the band (NaN: none):
    // the side shift goes beyond it, mid-way to the next eigenvalue.
    double plain_edge[2] = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    double r_plain = 0.0;
    if (plain_runs > 0) {
      const auto& pr = runs.front().res;
      for (index_t j = 0; j < pr.nconv; ++j) {
        r_plain = std::max(r_plain, 1.0 / std::abs(pr.theta[j]));
        const double l = (runs.front().sigma + 1.0 / pr.theta[j]).real();
        const int side = l > 0 ? 0 : 1;
        if (std::abs(l) > band && (std::isnan(plain_edge[side]) || std::abs(l) > std::abs(plain_edge[side])))
          plain_edge[side] = l;
      }
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
    std::vector<RitzSet> probes;
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
          // Probe at the inner edge for the first two eigenvalues on this side.
          const KsRun<T> probe = ks_run(c, fac, pencil, scale, inner[s], w, 2, kProbeRestarts, kProbeTol);
          probe_applies += probe.res.op_applies;
          probes.push_back({probe.sigma, probe.res.ritz_all});
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
          // The probe's distinct eigenvalues beyond the inner edge, nearest first: converged ones,
          // then the Ritz estimates on the wanted side (placement needs only a few digits). Copies
          // of a multiple eigenvalue count once.
          std::vector<double> beyond;
          const double same = 1e-6 * std::max(std::abs(inner[s]), r_plain);
          auto add = [&](double l) {
            for (double b : beyond)
              if (std::abs(l - b) <= same) return;
            beyond.push_back(l);
          };
          for (index_t j = 0; j < probe.res.nconv; ++j)
            if (sgn * probe.res.theta[j].real() > 0) add(probe.sigma + (1.0 / probe.res.theta[j]).real());
          for (const cplx& th : probe.res.ritz_all)
            if (sgn * th.real() > 0 && std::abs(th.imag()) <= 1e-3 * std::abs(th)) add(probe.sigma + (1.0 / th).real());
          std::sort(beyond.begin(), beyond.end(), [&](double x, double y) { return sgn * x < sgn * y; });
          // Mid-gap between the last eigenvalue already held on this side (the cluster, or the
          // plain run's outermost pair) and the next one: the inner edge lies just inside a plain
          // run's outermost pair, which the probe then finds first - it is held already.
          double held = std::isnan(plain_edge[s]) ? 0.0 : plain_edge[s];
          std::size_t nxt = 0;
          if (!beyond.empty() && !std::isnan(plain_edge[s]) && std::abs(beyond[0] - plain_edge[s]) <= same) {
            held = beyond[0];
            nxt = 1;
          }
          if (nxt < beyond.size()) {
            const double mid = 0.5 * (held + beyond[nxt]);
            if (std::abs(mid) > 2.0 * band_place) place[s] = mid;
            event(c, "side_probe", {{"side", static_cast<std::int64_t>(s)}, {"held", held}, {"next", beyond[nxt]},
                                    {"shift", place[s]}});
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
      release_factorization(c, fac);
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
    strip_unresolved =
        !recover_strip(c, pencil, fac, scale, n2a, n2b, band, band_place, runs, probes, ref, sel, good, in_band);
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
  // SolveReport::block_size: the width of the first run (the plain run, or the first one-sided run
  // - the probes before it are single-vector by rule).
  if (!runs.empty()) rep.block_size = runs.front().block_size;

  Solution<T> sol;
  // The runs' Schur bases are no longer needed: free them before the outputs are formed from the
  // refinement basis (the factorization was released before the last refinement).
  for (auto& r : runs) r.res.schur_basis = DenseMatrix<T>();
  std::vector<index_t> real_cols, complex_cols;
  for (const auto& cd : sel) (cd.real ? real_cols : complex_cols).push_back(cd.col);
  sol.eigenvectors = candidate_vectors<T, T>(ref, real_cols, n);
  sol.complex_eigenvectors = candidate_vectors<T, cplx>(ref, complex_cols, n);
  ref.basis = DenseMatrix<T>();
  const auto ncomplex = static_cast<index_t>(complex_cols.size());
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
      rep.pairs.push_back(p);
    } else {
      sol.complex_eigenvalues.push_back(cd.lambda);
      complex_reports.push_back(p);
    }
  }
  rep.pairs.insert(rep.pairs.end(), complex_reports.begin(), complex_reports.end());
  if (ncomplex > 0) warn(c, std::to_string(ncomplex) + " genuinely complex eigenvalue(s) returned separately");
  if (!all_ok && static_cast<index_t>(sel.size()) < k)
    warn(c, "only " + std::to_string(sel.size()) + " of " + std::to_string(k) + " pairs found");
  if (rep.zero_modes_filtered > 0)
    warn(c, std::to_string(rep.zero_modes_filtered) + " computed zero mode(s) filtered (|lambda| <= band)");
  rep.success = all_ok && !strip_unresolved;
  return sol;
}

#define HS_INSTANTIATE_SOLVE_INDEFINITE(T)                                                                  \
  template Solution<T> solve_indefinite<T>(Ctx&, const ShiftedPencil<T>&, SparseFactorization<T>&, double, \
                                           double, double);                                                \
  template index_t ghiep_reserve_columns<T>(const SolverConfig&, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE_INDEFINITE)
#undef HS_INSTANTIATE_SOLVE_INDEFINITE

}  // namespace hs::driver
