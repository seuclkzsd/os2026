// Hermitian Jacobi-Davidson (G. L. G. Sleijpen & H. A. van der Vorst, "A Jacobi-Davidson
// iteration method for linear eigenvalue problems", SIMAX 17(2), 1996) with JDQR-style locking,
// deflated correction equations and thick restart (D. R. Fokkema, G. L. G. Sleijpen & H. A. van
// der Vorst, "Jacobi-Davidson style QR and QZ algorithms...", SISC 20(1), 1998), for the pairs
// of A x = lambda B x (B HPD or I) nearest a target tau. Contract:
// include/hsolve/eigen_iterative.hpp.
//
// Storage. [Q | V] in one n x (nev + max_basis) block: Q (nconv locked eigenvectors) in front, the
// search space V (k columns) behind it; [Q V] is B-orthonormal. Next to it B [Q V] (B only) and
// K V with K = A - tau B (column-aligned; the K Q part is stale and unused). Small matrices:
// HK = V^H K V and, for harmonic extraction, G = (K V)^H (K V), both updated column by column.
// Locking a vector u = V c rotates V by a Householder matrix H with H e1 ~ c, so that u becomes
// the first column of V, and moves the Q/V boundary by one: no copies, V stays B-orthonormal and
// B-orthogonal to Q.
//
// Start space (start_block_iteration). p random vectors; with a preconditioner M ~ (A - tau B)^-1
// they first get start_iterations steps of subspace iteration S <- orth(M B S) (B-orthogonal to
// a deflation block). Every later expansion is derived from the residual of the one current
// approximation u, so it amplifies only the copy of a multiple eigenvalue contained in u; the
// other copies of a plain random block stay diluted by the far spectrum, and the extraction
// prefers a farther eigenvalue whose approximation was built along the way (measured: 4-5 of 6
// copies + 1.0 returned as converged). The block iteration amplifies all p directions at once.
// start_iterations 1 / 2 / 3 (bench/lobpcg_jd_fix_options.cpp, single factor, Ritz, 5 runs,
// solves min/median/max): syn-mem27-46 125/133/142, 129/135/136, 135/143/153; yee-pec-cube-32
// 184/196/199, 196/200/206, 200/204/230 (0: 168/187/211 and 202/207/216). One step still left 2
// harmonic failures in bench/lobpcg_jd_fix_synthetic.cpp, two none: default 2.
//
// Extraction. Ritz (the default): eigenpairs of HK, ranked by |nu| (theta = tau + nu). Measured
// against harmonic (bench/lobpcg_jd_fix_options.cpp, single factor, 5 runs, solves
// min/median/max, with / without the start block iteration): syn-mem27-46 Ritz 129/135/136 and
// 168/187/211, harmonic 147/165/288 and 432/916/1882; yee-pec-cube-32 Ritz 196/200/206 and
// 202/207/216, harmonic 194/201/218 and 203/208/262 (one run without the block iteration missed
// a copy). The harmonic variance is explained below: tau = 0 lies within 1e-6 ||A|| of
// eigenvalues of syn-mem27-46.
// Harmonic (B = I only, see eigen_iterative.hpp): HK c = mu G c, i.e. Ritz values of
// (A - tau I)^{-1} on the test space K V, ranked by |mu| (largest first = nearest tau). G is
// whitened through its eigen-decomposition with eigenvalues floored at 1e-14 max (G is singular
// when V holds an exact eigenvector at tau).
// Harmonic extraction cannot rank an eigenvalue at (or within the approximation error of) tau: for
// K x = 0 and u = x cos(phi) + e sin(phi) the harmonic value ||K u||^2 / (u^H K u) =
// ||K e||^2 / (e^H K e) depends only on the error direction e (for lambda = tau + delta it
// approaches delta only once the error angle is below ~delta / ||K||), and for x itself the pencil
// (HK, G) is singular (0/0; the floored whitening gives it a rounding-level |mu|, i.e. ranks it
// LAST). Measured before the fix: tau on an eigenvalue was skipped and a farther set was returned
// as converged. Hence the "certified Ritz override": with the Rayleigh quotient nu and residual
// norm rho of every Ritz vector (rho^2 = y^H G y - nu^2, from the small matrices), a Ritz pair
// whose inclusion interval (it contains an eigenvalue) satisfies |nu| + rho < |nu_sel| - rho_sel
// proves an eigenvalue nearer tau than anything the harmonic selection can approximate, and is
// selected instead. Spurious Ritz values (large rho) never pass this test. Solving through the
// normal matrix G limits the accuracy of a harmonic vector to about
// eps ||K||^2 / (nu_2^2 - nu_1^2) (nu_2: next nearest harmonic value), which can exceed tol when
// other eigenvalues lie very close to tau. Hence also "Ritz polishing": once the best harmonic
// vector is within ~8 degrees (overlap >= 0.99) of a Ritz vector - near convergence, where
// spurious Ritz values no longer occur - that Ritz vector (accurate to eps ||K|| / gap) is used.
//
// Correction equation. With U = [Q u] (u^H B u = 1, U B-orthonormal) and the shift s,
//     (I - B U U^H)(A - s B)(I - U U^H B) t = -r,   U^H B t = 0,
// where P_L = I - B U U^H and P_R = P_L^H = I - U U^H B, so the operator is Hermitian. Deflating
// Q keeps the operator well conditioned once eigenvalues near tau are locked (theta ~ tau).
//   * GMRES (flexible, right-preconditioned, one cycle) with the oblique projected
//     preconditioner z = M y - Y^ H^-1 (B U)^H M y, Y^ = M B U, H^ = (B U)^H Y^ (Fokkema et al.),
//     valid for any M (e.g. an indefinite single-precision factor of A - tau B). M B q is computed
//     once per locked vector when a solve first needs it; M B u costs one application per solve.
//   * MINRES (Paige-Saunders, preconditioned): Hermitian operator on range(P_L) with the
//     preconditioner N = P_R M, Hermitian positive definite on range(P_L) iff M is HPD.
// Target tracking (choose_shift): s = tau while u is rough (relative inner tolerance 0.25, at most
// 5 steps with a preconditioner), s = theta once u is accurate (relative inner tolerance
// 2^-(j+1), j = theta-shifted outer iterations on the current pair, at most inner_iterations
// steps).
//
// Convergence and locking. r = K u - nu B u comes from the stored K V, B V (one column update per
// expansion, rotations at restarts). When it passes the backward-error test, u is checked again
// with fresh products A u, B u and the Rayleigh quotient; only then it is locked (else the stored
// products are recomputed and the iteration continues with the fresh residual). The final
// residuals of all returned vectors are recomputed once more from scratch.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "eigen/lobpcg_jd_common.hpp"
#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {

namespace {

using eigen_detail::hermitize;
using eigen_detail::OrthoBlock;
using lapack::fint;

/// An expansion vector is dependent on [Q V] when less than this fraction of its B-norm survives.
constexpr double kExpandDrop = 1e-10;
/// Relative floor of the eigenvalues of G = (K V)^H K V in harmonic extraction.
constexpr double kHarmonicFloor = 1e-14;
/// Ritz polishing threshold: |<Ritz vector, best harmonic vector>| (unit vectors).
constexpr double kPolishOverlap = 0.99;
/// Coefficient vectors are dependent in a restart when less than this fraction survives.
constexpr double kRestartDrop = 1e-8;
/// Attempts at drawing a random vector that is not inside span[Q V].
constexpr int kRandomAttempts = 4;
/// The correction equation is shifted by theta (instead of tau) once ||r|| / ||B u|| is below
/// this fraction of the distance from theta to the nearest other Ritz value.
constexpr double kShiftSwitch = 0.1;
/// Relative inner tolerance of the tau-shifted correction equations (target tracking).
constexpr double kTrackingInnerTol = 0.25;
/// At most this many inner steps for a tau-shifted correction equation: it only has to supply a
/// shift-and-invert-like direction, and with a preconditioner that is inaccurate near tau more
/// steps stall without progress (measured: 20 steps ending at relative residual 0.5 - 0.8).
constexpr index_t kTrackingMaxInner = 5;

inline fint fi(index_t x) { return static_cast<fint>(x); }

/// Leading k x k block of m as its own matrix.
template <class T>
DenseMatrix<T> leading(const DenseMatrix<T>& m, index_t k) {
  DenseMatrix<T> out(k, k);
  for (index_t j = 0; j < k; ++j) std::copy(m.col(j), m.col(j) + k, out.col(j));
  return out;
}

/// Writes the k x k matrix s into the leading block of m.
template <class T>
void set_leading(DenseMatrix<T>& m, const DenseMatrix<T>& s) {
  for (index_t j = 0; j < s.cols(); ++j) std::copy(s.col(j), s.col(j) + s.rows(), m.col(j));
}

/// C^H M C for M (k x k) and C (k x q).
template <class T>
DenseMatrix<T> congruence(const DenseMatrix<T>& m, const DenseMatrix<T>& c) {
  const index_t k = c.rows(), q = c.cols();
  DenseMatrix<T> tmp(k, q), out(q, q);
  lapack::gemm('N', 'N', fi(k), fi(q), fi(k), T(1), m.data(), fi(m.rows()), c.data(), fi(k), T(0), tmp.data(),
               fi(k));
  lapack::gemm('C', 'N', fi(q), fi(q), fi(k), T(1), c.data(), fi(k), tmp.data(), fi(k), T(0), out.data(), fi(q));
  hermitize(out);
  return out;
}

/// Small-vector Euclidean helpers (length k).
template <class T>
T sdot(index_t k, const T* x, const T* y) {
  T s = T(0);
  for (index_t i = 0; i < k; ++i) s += conj_if(x[i]) * y[i];
  return s;
}

/// Complex/real Givens rotation [c s; -conj(s) c] with c real, mapping (a, b) to (rho, 0).
template <class T>
void givens(const T& a, const T& b, double& c, T& s) {
  const double aa = std::abs(a), bb = std::abs(b);
  if (bb == 0.0) {
    c = 1.0;
    s = T(0);
    return;
  }
  if (aa == 0.0) {
    c = 0.0;
    s = T(1);
    return;
  }
  const double r = std::hypot(aa, bb);
  c = aa / r;
  s = (a / aa) * conj_if(b) / r;
}

template <class T>
class JacobiDavidson {
 public:
  JacobiDavidson(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* m, double tau,
                 const JdOptions<T>& options);
  IterativeEigResult<T> run();

 private:
  T* qcol(index_t j) { return basis_.col(j); }
  T* bqcol(index_t j) { return b_ ? bbasis_.col(j) : basis_.col(j); }
  T* vcol(index_t j) { return basis_.col(nconv_ + j); }
  T* bvcol(index_t j) { return bqcol(nconv_ + j); }
  T* kvcol(index_t j) { return kv_.col(nconv_ + j); }

  void load_deflation();
  bool append(const T* t);
  bool append_random();
  void start_block_iteration(index_t p);
  void extract();
  DenseMatrix<T> orthonormal_candidates(index_t q);
  void restart();
  void current_pair();
  void choose_shift();
  bool verify_and_lock();
  void lock(double theta);
  void refresh_kv();
  void correction(index_t j);
  void prepare_projected_preconditioner();
  void project_left(T* y);
  void project_right(T* t);
  void apply_operator(const T* x, T* y);
  void apply_gmres_preconditioner(const T* y, T* z);
  void apply_minres_preconditioner(const T* y, T* z);
  void gmres(double rtol, index_t max_steps);
  void minres(double rtol, index_t max_steps);
  void record();
  IterativeEigResult<T> finish();

  LinearOperator<T>& a_;
  LinearOperator<T>* b_;
  LinearOperator<T>* m_;
  const double tau_;
  const JdOptions<T>& opt_;

  index_t n_ = 0, nev_ = 0, kmin_ = 0, kmax_ = 0, inner_max_ = 0;
  index_t d_ = 0;  ///< deflation columns (pre-locked, never returned) at the front of Q
  bool harmonic_ = false, use_gmres_ = false;
  double norm_a_ = 0.0, norm_b_ = 1.0;

  DenseMatrix<T> basis_, bbasis_, kv_, mbq_;
  index_t mbq_ready_ = 0;  ///< leading columns of mbq_ holding M B q (computed lazily)
  index_t fallbacks_ = 0;  ///< expansions by r or a random vector (dependent correction)
  index_t tau_steps_ = 0;  ///< outer iterations whose correction equation was shifted by tau
  DenseMatrix<T> hk_, g_;  ///< kmax x kmax, leading k x k in use
  index_t nconv_ = 0, k_ = 0;
  std::vector<double> locked_theta_;

  // Extraction results (coefficient vectors in the V basis, unit 2-norm).
  DenseMatrix<T> coef_;       ///< k x k, ranking order (harmonic or Ritz)
  std::vector<T> sel_;        ///< coefficients of the current approximation u
  double sel_theta_ = 0.0;

  // Current approximation.
  std::vector<T> u_, bu_, ku_, r_, t_, w1_, w2_;
  double theta_ = 0.0, eta_ = 0.0;
  double shift_ = 0.0;              ///< shift of the correction equation: tau or theta_
  std::vector<double> ritz_theta_;  ///< Ritz values of the current V (gap estimate)

  // Oblique projected preconditioner for GMRES.
  std::vector<T> mbu_;
  DenseMatrix<T> hhat_;
  std::vector<fint> hhat_piv_;
  bool hhat_ok_ = false;

  // Inner workspace.
  DenseMatrix<T> kry_, zkry_;
  std::vector<T> mr_[7];

  index_t iterations_ = 0, restarts_ = 0, inner_total_ = 0;
  std::uint64_t next_stream_ = 0;
  index_t a0_ = 0, b0_ = 0, m0_ = 0;
  std::vector<RestartRecord> history_;
};

template <class T>
JacobiDavidson<T>::JacobiDavidson(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* m,
                                  double tau, const JdOptions<T>& options)
    : a_(a), b_(b), m_(m), tau_(tau), opt_(options) {
  n_ = a.size();
  nev_ = options.nev;
  if (b && b->size() != n_) HS_THROW(InvalidInput, "jacobi_davidson: B has order " << b->size() << ", A " << n_);
  if (m && m->size() != n_)
    HS_THROW(InvalidInput, "jacobi_davidson: preconditioner has order " << m->size() << ", A " << n_);
  if (!std::isfinite(tau)) HS_THROW(InvalidInput, "jacobi_davidson: non-finite target " << tau);
  if (options.deflation && options.deflation->cols() > 0) {
    if (options.deflation->rows() != n_)
      HS_THROW(InvalidInput, "jacobi_davidson: deflation block has " << options.deflation->rows()
                                                                     << " rows, A has order " << n_);
    d_ = options.deflation->cols();
  }
  if (nev_ < 1 || nev_ > n_ - d_)
    HS_THROW(InvalidInput, "jacobi_davidson: nev must lie in [1, n - d = " << n_ - d_ << "], got " << nev_);
  if (!(options.tol > 0.0) || !std::isfinite(options.tol))
    HS_THROW(InvalidInput, "jacobi_davidson: tol must be positive and finite, got " << options.tol);
  if (options.max_iterations < 0) HS_THROW(InvalidInput, "jacobi_davidson: max_iterations must be >= 0");
  if (options.inner_iterations < 0) HS_THROW(InvalidInput, "jacobi_davidson: inner_iterations must be >= 0");
  if (options.start_iterations < 0) HS_THROW(InvalidInput, "jacobi_davidson: start_iterations must be >= 0");
  if (!(options.norm_a >= 0.0) || !(options.norm_b >= 0.0) || !std::isfinite(options.norm_a) ||
      !std::isfinite(options.norm_b))
    HS_THROW(InvalidInput, "jacobi_davidson: norm_a / norm_b must be finite and >= 0");
  if (options.min_basis < 0 || options.max_basis < 0)
    HS_THROW(InvalidInput, "jacobi_davidson: min_basis / max_basis must be >= 0");
  const index_t kmin_auto = std::max<index_t>(nev_, 10);
  kmin_ = options.min_basis > 0 ? options.min_basis : kmin_auto;
  kmax_ = options.max_basis > 0 ? options.max_basis : std::max(2 * kmin_, kmin_ + 10);
  if (options.min_basis > 0 && options.max_basis > 0 && options.max_basis <= options.min_basis)
    HS_THROW(InvalidInput, "jacobi_davidson: max_basis " << options.max_basis << " must exceed min_basis "
                                                         << options.min_basis);
  kmax_ = std::min(kmax_, n_ - d_);
  kmin_ = kmax_ > 1 ? std::min(kmin_, kmax_ - 1) : 1;
  if (options.initial) {
    const auto* ini = options.initial;
    if (ini->rows() != n_ || ini->cols() < 1 || ini->cols() > kmax_)
      HS_THROW(InvalidInput, "jacobi_davidson: initial block is " << ini->rows() << " x " << ini->cols() << ", need "
                                                                  << n_ << " x [1, " << kmax_ << "]");
  }
  switch (options.extraction) {
    case JdExtraction::Auto:  // Ritz: measured faster and more robust, see eigen_iterative.hpp
      harmonic_ = false;
      break;
    case JdExtraction::Harmonic:
      if (b)
        HS_THROW(InvalidInput,
                 "jacobi_davidson: harmonic extraction is implemented for B == nullptr only (with B it needs "
                 "B^-1); use JdExtraction::Ritz");
      harmonic_ = true;
      break;
    case JdExtraction::Ritz:
      harmonic_ = false;
      break;
  }
  switch (options.inner_solver) {
    case JdInnerSolver::Auto:
      use_gmres_ = m != nullptr;
      break;
    case JdInnerSolver::Gmres:
      use_gmres_ = true;
      break;
    case JdInnerSolver::Minres:
      use_gmres_ = false;
      break;
  }
  inner_max_ = options.inner_iterations;
  a0_ = a.applications();
  b0_ = b ? b->applications() : 0;
  m0_ = m ? m->applications() : 0;

  const index_t cols = d_ + nev_ + kmax_;
  basis_.resize(n_, cols);
  if (b_) bbasis_.resize(n_, cols);
  kv_.resize(n_, cols);
  if (m_ && use_gmres_) {
    mbq_.resize(n_, d_ + nev_);
    mbu_.resize(static_cast<std::size_t>(n_));
  }
  hk_.resize(kmax_, kmax_);
  if (harmonic_) g_.resize(kmax_, kmax_);
  for (auto* v : {&u_, &bu_, &ku_, &r_, &t_, &w1_, &w2_}) v->resize(static_cast<std::size_t>(n_));
  if (inner_max_ > 0) {
    if (use_gmres_) {
      kry_.resize(n_, inner_max_ + 1);
      zkry_.resize(n_, inner_max_);
    } else {
      for (auto& v : mr_) v.resize(static_cast<std::size_t>(n_));
    }
  }
}

/// Copies the deflation block Y in front of Q (pre-locked) and checks Y^H B Y = I.
template <class T>
void JacobiDavidson<T>::load_deflation() {
  if (d_ == 0) return;
  const DenseMatrix<T>& y = *opt_.deflation;
  std::copy(y.data(), y.data() + n_ * d_, basis_.data());
  eigen_detail::require_finite(n_, basis_.data(), d_, "jacobi_davidson: deflation block");
  if (b_) {
    b_->apply(basis_.data(), n_, bbasis_.data(), n_, d_);
    eigen_detail::require_finite(n_, bbasis_.data(), d_, "jacobi_davidson: B * deflation");
  }
  eigen_detail::require_b_orthonormal(n_, d_, basis_.data(), bqcol(0), "jacobi_davidson");
  nconv_ = d_;
}

/// B-orthonormalizes t against [Q V] and appends it to V (K v, HK, G updated). False when t lies
/// in span[Q V] to rounding.
template <class T>
bool JacobiDavidson<T>::append(const T* t) {
  HS_ASSERT(k_ < kmax_ && nconv_ + k_ < n_, "jacobi_davidson: no room to expand");
  T* slot = vcol(k_);
  if (slot != t) std::copy(t, t + n_, slot);
  std::vector<OrthoBlock<T>> against;
  if (nconv_ + k_ > 0) against.push_back({basis_.data(), bqcol(0), nconv_ + k_});
  const index_t kept =
      eigen_detail::b_orthonormalize<T>(n_, against, slot, b_ ? bvcol(k_) : nullptr, 1, b_, kExpandDrop);
  if (kept == 0) return false;
  T* kv = kvcol(k_);
  a_.apply(slot, n_, kv, n_, 1);
  eigen_detail::require_finite(n_, kv, 1, "jacobi_davidson: A * v");
  linalg::axpy(n_, T(-tau_), bvcol(k_), kv);
  const index_t k1 = k_ + 1;
  std::vector<T> h(static_cast<std::size_t>(k1));
  lapack::gemv('C', fi(n_), fi(k1), T(1), vcol(0), fi(n_), kv, T(0), h.data());
  for (index_t i = 0; i < k_; ++i) {
    hk_(i, k_) = h[i];
    hk_(k_, i) = conj_if(h[i]);
  }
  hk_(k_, k_) = T(std::real(h[k_]));
  if (harmonic_) {
    lapack::gemv('C', fi(n_), fi(k1), T(1), kvcol(0), fi(n_), kv, T(0), h.data());
    for (index_t i = 0; i < k_; ++i) {
      g_(i, k_) = h[i];
      g_(k_, i) = conj_if(h[i]);
    }
    g_(k_, k_) = T(std::real(h[k_]));
  }
  ++k_;
  return true;
}

template <class T>
bool JacobiDavidson<T>::append_random() {
  for (int attempt = 0; attempt < kRandomAttempts; ++attempt) {
    linalg::fill_random(vcol(k_), n_, opt_.seed, next_stream_++);
    if (append(vcol(k_))) return true;
  }
  return false;
}

/// Start space by subspace iteration: p random vectors, then start_iterations steps
/// S <- M B S, each B-orthonormalized against Q (deflation block) and within S; appended to V.
template <class T>
void JacobiDavidson<T>::start_block_iteration(index_t p) {
  DenseMatrix<T> s(n_, p), bs, ms(n_, p);
  if (b_) bs.resize(n_, p);
  T* bsd = b_ ? bs.data() : nullptr;
  std::vector<OrthoBlock<T>> against;
  if (nconv_ > 0) against.push_back({basis_.data(), bqcol(0), nconv_});
  for (index_t j = 0; j < p; ++j) linalg::fill_random(s.col(j), n_, opt_.seed, next_stream_++);
  index_t q = eigen_detail::b_orthonormalize<T>(n_, against, s.data(), bsd, p, b_, kExpandDrop);
  for (index_t step = 0; step < opt_.start_iterations && q > 0; ++step) {
    m_->apply(b_ ? bsd : s.data(), n_, ms.data(), n_, q);
    eigen_detail::require_finite(n_, ms.data(), q, "jacobi_davidson: preconditioner output");
    std::swap(s, ms);
    q = eigen_detail::b_orthonormalize<T>(n_, against, s.data(), bsd, q, b_, kExpandDrop);
  }
  for (index_t j = 0; j < q && k_ < kmax_; ++j) append(s.col(j));
}

template <class T>
void JacobiDavidson<T>::extract() {
  const index_t k = k_;
  DenseMatrix<T> hk = leading(hk_, k);
  DenseMatrix<T> ritz = hk;
  std::vector<double> nu(static_cast<std::size_t>(k));
  lapack::heev('V', 'L', fi(k), ritz.data(), fi(k), nu.data());
  std::vector<index_t> ritz_order(static_cast<std::size_t>(k));
  std::iota(ritz_order.begin(), ritz_order.end(), index_t{0});
  std::stable_sort(ritz_order.begin(), ritz_order.end(),
                   [&](index_t x, index_t y) { return std::abs(nu[x]) < std::abs(nu[y]); });
  coef_.resize(k, k);
  sel_.assign(static_cast<std::size_t>(k), T(0));
  ritz_theta_.resize(static_cast<std::size_t>(k));
  for (index_t i = 0; i < k; ++i) ritz_theta_[i] = tau_ + nu[i];
  if (!harmonic_) {
    for (index_t l = 0; l < k; ++l) std::copy(ritz.col(ritz_order[l]), ritz.col(ritz_order[l]) + k, coef_.col(l));
    std::copy(coef_.col(0), coef_.col(0) + k, sel_.begin());
    sel_theta_ = tau_ + nu[ritz_order[0]];
    return;
  }
  // Harmonic: HK c = mu G c through the whitening of G (eigenvalues floored).
  DenseMatrix<T> w = leading(g_, k);
  std::vector<double> sigma(static_cast<std::size_t>(k));
  lapack::heev('V', 'L', fi(k), w.data(), fi(k), sigma.data());
  const double floor = std::max(sigma.back() * kHarmonicFloor, std::numeric_limits<double>::min());
  for (index_t j = 0; j < k; ++j) {
    const double sc = 1.0 / std::sqrt(std::max(sigma[j], floor));
    for (index_t i = 0; i < k; ++i) w(i, j) *= sc;
  }
  DenseMatrix<T> red = congruence(hk, w);
  std::vector<double> mu(static_cast<std::size_t>(k));
  lapack::heev('V', 'L', fi(k), red.data(), fi(k), mu.data());
  DenseMatrix<T> c(k, k);
  lapack::gemm('N', 'N', fi(k), fi(k), fi(k), T(1), w.data(), fi(k), red.data(), fi(k), T(0), c.data(), fi(k));
  std::vector<index_t> order(static_cast<std::size_t>(k));
  std::iota(order.begin(), order.end(), index_t{0});
  std::stable_sort(order.begin(), order.end(), [&](index_t x, index_t y) { return std::abs(mu[x]) > std::abs(mu[y]); });
  for (index_t l = 0; l < k; ++l) {
    const T* src = c.col(order[l]);
    const double nrm = std::sqrt(std::real(sdot(k, src, src)));
    for (index_t i = 0; i < k; ++i) coef_(i, l) = src[i] / nrm;
  }
  // Rayleigh quotient nu(y) = y^H HK y and residual norm rho(y) = ||K V y - nu V y|| of a unit
  // coefficient vector y (V orthonormal, B = I): rho^2 = y^H G y - nu^2.
  const DenseMatrix<T> gk = leading(g_, k);
  std::vector<T> tmp(static_cast<std::size_t>(k));
  auto quotient = [&](const T* y, double& nu_y, double& rho_y) {
    lapack::gemv('N', fi(k), fi(k), T(1), hk.data(), fi(k), y, T(0), tmp.data());
    nu_y = std::real(sdot(k, y, tmp.data()));
    lapack::gemv('N', fi(k), fi(k), T(1), gk.data(), fi(k), y, T(0), tmp.data());
    rho_y = std::sqrt(std::max(std::real(sdot(k, y, tmp.data())) - nu_y * nu_y, 0.0));
  };
  // Ritz polishing (see the file comment).
  index_t best = 0;
  double overlap = -1.0;
  for (index_t l = 0; l < k; ++l) {
    const double o = std::abs(sdot(k, ritz.col(l), coef_.col(0)));
    if (o > overlap) {
      overlap = o;
      best = l;
    }
  }
  double sel_nu = 0.0, sel_rho = 0.0;
  if (overlap >= kPolishOverlap) {
    std::copy(ritz.col(best), ritz.col(best) + k, sel_.begin());
  } else {
    std::copy(coef_.col(0), coef_.col(0) + k, sel_.begin());
  }
  quotient(sel_.data(), sel_nu, sel_rho);
  // Certified Ritz override (see the file comment): a Ritz pair whose inclusion interval
  // [nu - rho, nu + rho] lies strictly nearer tau than every value the selected pair can
  // approximate proves a nearer eigenvalue; it replaces the harmonic selection.
  index_t cert = -1;
  double cert_bound = std::numeric_limits<double>::infinity();
  for (index_t l = 0; l < k; ++l) {
    double nu_l = 0.0, rho_l = 0.0;
    quotient(ritz.col(l), nu_l, rho_l);
    const double bound = std::abs(nu_l) + rho_l;
    if (bound < cert_bound) {
      cert_bound = bound;
      cert = l;
    }
  }
  if (cert >= 0 && cert_bound < std::abs(sel_nu) - sel_rho) {
    std::copy(ritz.col(cert), ritz.col(cert) + k, sel_.begin());
    quotient(sel_.data(), sel_nu, sel_rho);
  }
  sel_theta_ = tau_ + sel_nu;
}

/// The current approximation followed by the ranked extraction vectors, orthonormalized (CGS2)
/// in the coefficient space; the first q independent ones (k x q).
template <class T>
DenseMatrix<T> JacobiDavidson<T>::orthonormal_candidates(index_t q) {
  const index_t k = k_;
  DenseMatrix<T> c(k, q);
  index_t got = 0;
  auto try_add = [&](const T* src) {
    T* dst = c.col(got);
    std::copy(src, src + k, dst);
    const double before = std::sqrt(std::real(sdot(k, dst, dst)));
    for (int pass = 0; pass < 2; ++pass)
      for (index_t l = 0; l < got; ++l) {
        const T p = sdot(k, c.col(l), dst);
        for (index_t i = 0; i < k; ++i) dst[i] -= p * c(i, l);
      }
    const double after = std::sqrt(std::real(sdot(k, dst, dst)));
    if (!(after > kRestartDrop * before)) return;
    for (index_t i = 0; i < k; ++i) dst[i] /= after;
    ++got;
  };
  if (q > 0) try_add(sel_.data());
  for (index_t l = 0; l < k && got < q; ++l) try_add(coef_.col(l));
  HS_ASSERT(got == q, "jacobi_davidson: candidate block of rank " << got << " < " << q);
  return c;
}

/// Thick restart: keep kmin vectors - the current approximation first, then the best ranked
/// extraction vectors - orthonormalized in the coefficient space.
template <class T>
void JacobiDavidson<T>::restart() {
  const index_t k = k_, keep = kmin_;
  const DenseMatrix<T> c = orthonormal_candidates(keep);
  eigen_detail::rotate_in_place(n_, vcol(0), k, c.data(), k, keep);
  if (b_) eigen_detail::rotate_in_place(n_, bvcol(0), k, c.data(), k, keep);
  eigen_detail::rotate_in_place(n_, kvcol(0), k, c.data(), k, keep);
  const DenseMatrix<T> hk_new = congruence(leading(hk_, k), c);
  hk_.set_zero();
  set_leading(hk_, hk_new);
  if (harmonic_) {
    const DenseMatrix<T> g_new = congruence(leading(g_, k), c);
    g_.set_zero();
    set_leading(g_, g_new);
  }
  k_ = keep;
  ++restarts_;
}

template <class T>
void JacobiDavidson<T>::current_pair() {
  const index_t k = k_;
  lapack::gemv('N', fi(n_), fi(k), T(1), vcol(0), fi(n_), sel_.data(), T(0), u_.data());
  if (b_) lapack::gemv('N', fi(n_), fi(k), T(1), bvcol(0), fi(n_), sel_.data(), T(0), bu_.data());
  else std::copy(u_.begin(), u_.end(), bu_.begin());
  lapack::gemv('N', fi(n_), fi(k), T(1), kvcol(0), fi(n_), sel_.data(), T(0), ku_.data());
  theta_ = sel_theta_;
  const T nu = T(theta_ - tau_);
  const T* ku = ku_.data();
  const T* bu = bu_.data();
  T* r = r_.data();
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n_; ++i) r[i] = ku[i] - nu * bu[i];
  eta_ = eigen_detail::backward_error(linalg::norm2(n_, r), theta_, linalg::norm2(n_, u_.data()), norm_a_, norm_b_);
  choose_shift();
}

/// Target tracking (Sleijpen & van der Vorst; Fokkema et al.). The correction equation is shifted
/// by tau while u is rough: with a preconditioner ~ (A - tau B)^{-1} the expansion then acts like
/// shift-and-invert at tau and fills V with the eigenvectors nearest tau in order. Shifting by
/// theta from the start turns the iteration into Rayleigh quotient iteration from a random start,
/// which converges to whichever eigenvalue theta happens to approach (measured: nearer eigenvalues
/// skipped). theta is used once
///   ||r|| / ||B u|| <= 0.1 * (distance from theta to the nearest Ritz value of another
///   eigenvalue), i.e. u is within an angle ~0.1 of the eigenvector nearest theta.
/// Ritz values within 2 ||r|| / ||B u|| of theta are copies of a multiple eigenvalue unresolved at
/// this accuracy and do not count as "another eigenvalue".
/// Variants measured and rejected (bench/lobpcg_jd_memory.cpp): expanding by M B u or by the
/// Davidson step M r in tau mode (no convergence when M is inaccurate near tau), a sticky theta
/// shift after stalled tau steps, theta only for "resolved" (polished) approximations, capping the
/// inner steps of unresolved theta steps, a deflated two-level preconditioner on span(V) (none
/// reliably faster), and JDQR's fixed switch at eta <= 1e-4 (theta then took over before the
/// tau steps had amplified further copies of a multiple eigenvalue: a copy at distance 0.05 was
/// skipped for a simple eigenvalue at 0.35).
template <class T>
void JacobiDavidson<T>::choose_shift() {
  shift_ = tau_;
  if (k_ < 2) return;
  const double rho = linalg::norm2(n_, r_.data()) / linalg::norm2(n_, bu_.data());
  // Distance to the nearest Ritz value of another eigenvalue: skip the one representing u and
  // those within 2 rho of theta (copies of a multiple eigenvalue, unresolved at this accuracy).
  std::vector<double> d;
  for (double rt : ritz_theta_) d.push_back(std::abs(rt - theta_));
  std::sort(d.begin(), d.end());
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < d.size(); ++i)
    if (d[i] > 2.0 * rho) {
      gap = d[i];
      break;
    }
  if (rho <= kShiftSwitch * gap) shift_ = theta_;
}

/// Checks the current approximation with fresh products A u, B u and its Rayleigh quotient; locks
/// it when it passes. Otherwise the stored products are recomputed (they had drifted) and the
/// fresh residual replaces r for the correction equation.
template <class T>
bool JacobiDavidson<T>::verify_and_lock() {
  T* au = w1_.data();
  T* bu = w2_.data();
  a_.apply(u_.data(), n_, au, n_, 1);
  eigen_detail::require_finite(n_, au, 1, "jacobi_davidson: A * u");
  eigen_detail::apply_or_copy(b_, n_, u_.data(), bu, 1);
  if (b_) eigen_detail::require_finite(n_, bu, 1, "jacobi_davidson: B * u");
  const double ubu = std::real(linalg::dot(n_, u_.data(), bu));
  if (!(ubu > 0.0)) HS_THROW(InvalidInput, "jacobi_davidson: u^H B u = " << ubu << ", B is not positive definite");
  const double theta = std::real(linalg::dot(n_, u_.data(), au)) / ubu;
  linalg::axpy(n_, T(-theta), bu, au);
  const double eta = eigen_detail::backward_error(linalg::norm2(n_, au), theta, linalg::norm2(n_, u_.data()),
                                                  norm_a_, norm_b_);
  if (eta <= opt_.tol) {
    lock(theta);
    return true;
  }
  theta_ = theta;
  eta_ = eta;
  std::copy(w1_.begin(), w1_.end(), r_.begin());
  std::copy(w2_.begin(), w2_.end(), bu_.begin());
  choose_shift();
  refresh_kv();
  return false;
}

/// Moves the current approximation u = V sel from V to Q (see the file comment).
template <class T>
void JacobiDavidson<T>::lock(double theta) {
  const index_t k = k_;
  // Householder H = I - 2 v v^H / (v^H v), v = sel - alpha e1, alpha = -phase(sel_0): H sel =
  // alpha e1 (v^H sel = 1 + |sel_0| is real), hence H e1 = sel / alpha.
  std::vector<T> v(sel_);
  const double a0 = std::abs(v[0]);
  const T phase = a0 > 0.0 ? v[0] / a0 : T(1);
  v[0] += phase;  // sel_0 - alpha
  const double vv = std::real(sdot(k, v.data(), v.data()));
  DenseMatrix<T> h(k, k);
  for (index_t j = 0; j < k; ++j)
    for (index_t i = 0; i < k; ++i) h(i, j) = T(i == j ? 1.0 : 0.0) - (2.0 / vv) * v[i] * conj_if(v[j]);
  eigen_detail::rotate_in_place(n_, vcol(0), k, h.data(), k, k);
  if (b_) eigen_detail::rotate_in_place(n_, bvcol(0), k, h.data(), k, k);
  eigen_detail::rotate_in_place(n_, kvcol(0), k, h.data(), k, k);
  const DenseMatrix<T> hk_new = congruence(leading(hk_, k), h);
  hk_.set_zero();
  for (index_t j = 1; j < k; ++j)
    for (index_t i = 1; i < k; ++i) hk_(i - 1, j - 1) = hk_new(i, j);
  if (harmonic_) {
    const DenseMatrix<T> g_new = congruence(leading(g_, k), h);
    g_.set_zero();
    for (index_t j = 1; j < k; ++j)
      for (index_t i = 1; i < k; ++i) g_(i - 1, j - 1) = g_new(i, j);
  }
  locked_theta_.push_back(theta);
  ++nconv_;
  --k_;
}

/// Recomputes B V, K V, HK and G from scratch.
template <class T>
void JacobiDavidson<T>::refresh_kv() {
  const index_t k = k_;
  if (k == 0) return;
  if (b_) {
    b_->apply(vcol(0), n_, bvcol(0), n_, k);
    eigen_detail::require_finite(n_, bvcol(0), k, "jacobi_davidson: B * V");
  }
  a_.apply(vcol(0), n_, kvcol(0), n_, k);
  eigen_detail::require_finite(n_, kvcol(0), k, "jacobi_davidson: A * V");
  for (index_t j = 0; j < k; ++j) linalg::axpy(n_, T(-tau_), bvcol(j), kvcol(j));
  DenseMatrix<T> hk(k, k);
  linalg::gram(n_, k, vcol(0), n_, kvcol(0), n_, hk.data(), k);
  hermitize(hk);
  set_leading(hk_, hk);
  if (harmonic_) {
    DenseMatrix<T> g(k, k);
    linalg::gram(n_, k, kvcol(0), n_, kvcol(0), n_, g.data(), k);
    hermitize(g);
    set_leading(g_, g);
  }
}

/// y <- P_L y = y - B U (U^H y), U = [Q u].
template <class T>
void JacobiDavidson<T>::project_left(T* y) {
  std::vector<T> c(static_cast<std::size_t>(nconv_));
  if (nconv_ > 0) lapack::gemv('C', fi(n_), fi(nconv_), T(1), qcol(0), fi(n_), y, T(0), c.data());
  const T cu = linalg::dot(n_, u_.data(), y);
  if (nconv_ > 0) lapack::gemv('N', fi(n_), fi(nconv_), T(-1), bqcol(0), fi(n_), c.data(), T(1), y);
  linalg::axpy(n_, -cu, bu_.data(), y);
}

/// t <- P_R t = t - U ((B U)^H t).
template <class T>
void JacobiDavidson<T>::project_right(T* t) {
  std::vector<T> c(static_cast<std::size_t>(nconv_));
  if (nconv_ > 0) lapack::gemv('C', fi(n_), fi(nconv_), T(1), bqcol(0), fi(n_), t, T(0), c.data());
  const T cu = linalg::dot(n_, bu_.data(), t);
  if (nconv_ > 0) lapack::gemv('N', fi(n_), fi(nconv_), T(-1), qcol(0), fi(n_), c.data(), T(1), t);
  linalg::axpy(n_, -cu, u_.data(), t);
}

/// y = P_L (A - s B) P_R x, s = shift_ (tau or theta, see choose_shift).
template <class T>
void JacobiDavidson<T>::apply_operator(const T* x, T* y) {
  std::copy(x, x + n_, w1_.data());
  project_right(w1_.data());
  a_.apply(w1_.data(), n_, y, n_, 1);
  if (b_) {
    b_->apply(w1_.data(), n_, w2_.data(), n_, 1);
    linalg::axpy(n_, T(-shift_), w2_.data(), y);
  } else {
    linalg::axpy(n_, T(-shift_), w1_.data(), y);
  }
  project_left(y);
}

/// Y^ = M B U and the LU factors of H^ = (B U)^H Y^ for the oblique projected preconditioner.
template <class T>
void JacobiDavidson<T>::prepare_projected_preconditioner() {
  if (mbq_ready_ < nconv_) {
    // M B q for the pairs locked since the last theta-shifted solve (Q only grows).
    m_->apply(bqcol(mbq_ready_), n_, mbq_.col(mbq_ready_), n_, nconv_ - mbq_ready_);
    eigen_detail::require_finite(n_, mbq_.col(mbq_ready_), nconv_ - mbq_ready_,
                                 "jacobi_davidson: preconditioner output");
    mbq_ready_ = nconv_;
  }
  m_->apply(bu_.data(), n_, mbu_.data(), n_, 1);
  eigen_detail::require_finite(n_, mbu_.data(), 1, "jacobi_davidson: preconditioner output");
  const index_t p = nconv_ + 1;
  hhat_.resize(p, p);
  std::vector<T> col(static_cast<std::size_t>(p));
  for (index_t j = 0; j < p; ++j) {
    const T* y = j < nconv_ ? mbq_.col(j) : mbu_.data();
    if (nconv_ > 0) lapack::gemv('C', fi(n_), fi(nconv_), T(1), bqcol(0), fi(n_), y, T(0), col.data());
    col[nconv_] = linalg::dot(n_, bu_.data(), y);
    std::copy(col.begin(), col.end(), hhat_.col(j));
  }
  hhat_piv_.assign(static_cast<std::size_t>(p), 0);
  hhat_ok_ = lapack::getrf(fi(p), hhat_.data(), fi(p), hhat_piv_.data()) == 0;
  if (hhat_ok_) {
    double dmax = 0.0, dmin = std::numeric_limits<double>::infinity();
    for (index_t i = 0; i < p; ++i) {
      dmax = std::max(dmax, std::abs(hhat_(i, i)));
      dmin = std::min(dmin, std::abs(hhat_(i, i)));
    }
    hhat_ok_ = std::isfinite(dmax) && dmin > 1e-14 * dmax;
  }
}

/// GMRES preconditioner z = M y - Y^ H^-1 (B U)^H M y (so (B U)^H z = 0); without M, or with a
/// numerically singular H^, the orthogonal projection P_R (M y).
template <class T>
void JacobiDavidson<T>::apply_gmres_preconditioner(const T* y, T* z) {
  if (!m_) {
    std::copy(y, y + n_, z);
    project_right(z);
    return;
  }
  m_->apply(y, n_, z, n_, 1);
  if (!hhat_ok_) {
    project_right(z);
    return;
  }
  const index_t p = nconv_ + 1;
  std::vector<T> s(static_cast<std::size_t>(p));
  if (nconv_ > 0) lapack::gemv('C', fi(n_), fi(nconv_), T(1), bqcol(0), fi(n_), z, T(0), s.data());
  s[nconv_] = linalg::dot(n_, bu_.data(), z);
  lapack::getrs(fi(p), 1, hhat_.data(), fi(p), hhat_piv_.data(), s.data(), fi(p));
  if (nconv_ > 0) lapack::gemv('N', fi(n_), fi(nconv_), T(-1), mbq_.data(), fi(n_), s.data(), T(1), z);
  linalg::axpy(n_, -s[nconv_], mbu_.data(), z);
}

/// MINRES preconditioner N y = P_R (M y) (identity without M); Hermitian positive definite on
/// range(P_L) when M is.
template <class T>
void JacobiDavidson<T>::apply_minres_preconditioner(const T* y, T* z) {
  if (!m_) {
    std::copy(y, y + n_, z);
    return;
  }
  m_->apply(y, n_, z, n_, 1);
  project_right(z);
}

/// Flexible GMRES, one cycle of at most inner_max_ steps, x0 = 0, for the correction equation.
template <class T>
void JacobiDavidson<T>::gmres(double rtol, index_t max_steps) {
  const index_t mmax = max_steps;
  T* v0 = kry_.col(0);
  for (index_t i = 0; i < n_; ++i) v0[i] = -r_[i];
  project_left(v0);
  const double beta = linalg::norm2(n_, v0);
  std::fill(t_.begin(), t_.end(), T(0));
  if (!(beta > 0.0)) return;
  linalg::scal(n_, T(1.0 / beta), v0);
  DenseMatrix<T> h(mmax + 1, mmax);
  std::vector<double> cs(static_cast<std::size_t>(mmax));
  std::vector<T> sn(static_cast<std::size_t>(mmax)), g(static_cast<std::size_t>(mmax + 1), T(0));
  g[0] = T(beta);
  index_t steps = 0;
  for (index_t j = 0; j < mmax; ++j) {
    apply_gmres_preconditioner(kry_.col(j), zkry_.col(j));
    apply_operator(zkry_.col(j), kry_.col(j + 1));
    const double hn =
        linalg::orthogonalize_cgs2<T>(n_, j + 1, kry_.data(), n_, nullptr, n_, kry_.col(j + 1), nullptr, h.col(j));
    if (!std::isfinite(hn))
      HS_THROW(InvalidInput, "jacobi_davidson: non-finite inner iterate (operator or preconditioner returned Inf/NaN)");
    h(j + 1, j) = T(hn);
    if (hn > 0.0) linalg::scal(n_, T(1.0 / hn), kry_.col(j + 1));
    for (index_t i = 0; i < j; ++i) {
      const T x = h(i, j), y = h(i + 1, j);
      h(i, j) = cs[i] * x + sn[i] * y;
      h(i + 1, j) = -conj_if(sn[i]) * x + cs[i] * y;
    }
    givens(h(j, j), h(j + 1, j), cs[j], sn[j]);
    h(j, j) = cs[j] * h(j, j) + sn[j] * h(j + 1, j);
    h(j + 1, j) = T(0);
    g[j + 1] = -conj_if(sn[j]) * g[j];
    g[j] = cs[j] * g[j];
    ++inner_total_;
    steps = j + 1;
    if (std::abs(g[j + 1]) <= rtol * beta || hn == 0.0) break;
  }
  // Back substitution H(0:steps, 0:steps) y = g(0:steps), then t = Z y.
  std::vector<T> y(static_cast<std::size_t>(steps), T(0));
  for (index_t i = steps - 1; i >= 0; --i) {
    T acc = g[i];
    for (index_t l = i + 1; l < steps; ++l) acc -= h(i, l) * y[l];
    y[i] = std::abs(h(i, i)) > 0.0 ? acc / h(i, i) : T(0);
  }
  if (steps > 0) lapack::gemv('N', fi(n_), fi(steps), T(1), zkry_.data(), fi(n_), y.data(), T(0), t_.data());
  project_right(t_.data());
}

/// Preconditioned MINRES (Paige & Saunders 1975, in the formulation of scipy's minres), x0 = 0.
template <class T>
void JacobiDavidson<T>::minres(double rtol, index_t max_steps) {
  T* r1 = mr_[0].data();
  T* r2 = mr_[1].data();
  T* y = mr_[2].data();
  T* v = mr_[3].data();
  T* w = mr_[4].data();
  T* w1 = mr_[5].data();
  T* w2 = mr_[6].data();
  T* x = t_.data();
  std::fill(t_.begin(), t_.end(), T(0));
  for (index_t i = 0; i < n_; ++i) r1[i] = -r_[i];
  project_left(r1);
  apply_minres_preconditioner(r1, y);
  const double b1sq = std::real(linalg::dot(n_, r1, y));
  if (!(b1sq > 0.0)) {
    // Zero right-hand side, or a preconditioner that is not positive definite: fall back to the
    // preconditioned residual (Davidson step).
    if (b1sq < 0.0) std::copy(y, y + n_, x);
    project_right(x);
    return;
  }
  const double beta1 = std::sqrt(b1sq);
  std::copy(r1, r1 + n_, r2);
  std::fill(w, w + n_, T(0));
  std::fill(w2, w2 + n_, T(0));
  double oldb = 0.0, beta = beta1, dbar = 0.0, epsln = 0.0, phibar = beta1, cs = -1.0, sn = 0.0;
  for (index_t itn = 1; itn <= max_steps; ++itn) {
    const double s = 1.0 / beta;
    for (index_t i = 0; i < n_; ++i) v[i] = s * y[i];
    apply_operator(v, y);
    if (itn >= 2) linalg::axpy(n_, T(-beta / oldb), r1, y);
    const double alfa = std::real(linalg::dot(n_, v, y));
    linalg::axpy(n_, T(-alfa / beta), r2, y);
    std::swap(r1, r2);  // r1 <- old r2
    std::swap(r2, y);   // r2 <- new y; y <- free buffer
    apply_minres_preconditioner(r2, y);
    oldb = beta;
    const double b2 = std::real(linalg::dot(n_, r2, y));
    if (!std::isfinite(b2))
      HS_THROW(InvalidInput, "jacobi_davidson: non-finite inner iterate (operator or preconditioner returned Inf/NaN)");
    if (b2 < 0.0) break;  // preconditioner not positive definite: keep the current iterate
    beta = std::sqrt(b2);
    const double oldeps = epsln;
    const double delta = cs * dbar + sn * alfa;
    const double gbar = sn * dbar - cs * alfa;
    epsln = sn * beta;
    dbar = -cs * beta;
    double gamma = std::hypot(gbar, beta);
    if (gamma == 0.0) gamma = std::numeric_limits<double>::epsilon();
    cs = gbar / gamma;
    sn = beta / gamma;
    const double phi = cs * phibar;
    phibar = sn * phibar;
    std::swap(w1, w2);  // w1 <- old w2
    std::swap(w2, w);   // w2 <- old w; w <- free buffer
    const double inv = 1.0 / gamma;
    for (index_t i = 0; i < n_; ++i) w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * inv;
    linalg::axpy(n_, T(phi), w, x);
    ++inner_total_;
    if (std::abs(phibar) <= rtol * beta1 || beta == 0.0) break;
  }
  project_right(x);
}

/// Approximate solution t_ of the deflated correction equation (see the file comment).
template <class T>
void JacobiDavidson<T>::correction(index_t j) {
  // Inner tolerance: a fixed loose one while tracking the target (the tau-shifted equation only
  // supplies a shift-and-invert-like direction; one projected GMRES step when M is accurate, more
  // when it is not), 2^-(j+1) over the theta-shifted steps j = 0, 1, ... of the current pair.
  // With a preconditioner the tau-shifted solves are also capped at kTrackingMaxInner steps.
  double rtol = kTrackingInnerTol;
  index_t max_steps = m_ ? std::min(inner_max_, kTrackingMaxInner) : inner_max_;
  if (shift_ == tau_) {
    ++tau_steps_;
  } else {
    rtol = std::pow(0.5, static_cast<double>(j + 1));
    max_steps = inner_max_;
  }

  if (m_ && use_gmres_) prepare_projected_preconditioner();
  if (inner_max_ == 0) {
    // Olsen / preconditioned Davidson step t = -M~ r.
    for (index_t i = 0; i < n_; ++i) w2_[i] = -r_[i];
    project_left(w2_.data());
    if (use_gmres_) {
      apply_gmres_preconditioner(w2_.data(), t_.data());
    } else {
      apply_minres_preconditioner(w2_.data(), t_.data());
    }
    project_right(t_.data());
    return;
  }
  if (use_gmres_) {
    gmres(rtol, max_steps);
  } else {
    minres(rtol, max_steps);
  }
}

template <class T>
void JacobiDavidson<T>::record() {
  RestartRecord rec;
  rec.restart = iterations_;
  rec.nconv = nconv_ - d_;
  rec.max_estimate = nconv_ - d_ >= nev_ ? 0.0 : eta_;
  rec.op_applies = a_.applications() - a0_;
  history_.push_back(rec);
}

template <class T>
IterativeEigResult<T> JacobiDavidson<T>::run() {
  norm_a_ = opt_.norm_a > 0.0 ? opt_.norm_a : eigen_detail::estimate_norm2(a_, opt_.seed ^ 0x6a645f6e6f726dULL);
  norm_b_ = !b_ ? 1.0 : (opt_.norm_b > 0.0 ? opt_.norm_b : eigen_detail::estimate_norm2(*b_, opt_.seed ^ 0x62ULL));
  load_deflation();
  if (opt_.initial) {
    eigen_detail::require_finite(n_, opt_.initial->data(), opt_.initial->cols(), "jacobi_davidson: initial block");
    for (index_t j = 0; j < opt_.initial->cols() && k_ < kmax_; ++j) append(opt_.initial->col(j));
  }
  if (k_ == 0) {
    // A block of random start vectors (not one): from a single vector further copies of a
    // multiple eigenvalue appear only at rounding level (measured: a triple eigenvalue at
    // distance 0.05 lost to a simple one at 0.35). With a preconditioner the block is improved
    // by subspace iteration (file comment); a plain random block still loses copies once the
    // multiplicity reaches about nev - 1.
    const index_t p = std::max<index_t>(std::min<index_t>(std::min(nev_, kmax_ - 1), n_ - d_), 1);
    if (m_ && opt_.start_iterations > 0) start_block_iteration(p);
    for (index_t j = k_; j < p; ++j)
      if (!append_random() && k_ == 0)
        HS_THROW(InvalidInput, "jacobi_davidson: cannot draw a start vector with positive B-norm (B not HPD?)");
  }
  index_t j_cur = 0;  // theta-shifted outer iterations spent on the current eigenpair
  for (;;) {
    extract();
    if (k_ == kmax_) {
      restart();
      extract();
    }
    current_pair();
    if (eta_ <= opt_.tol && verify_and_lock()) {
      j_cur = 0;
      if (nconv_ - d_ == nev_) {
        record();
        break;
      }
      if (k_ == 0 && (nconv_ >= n_ || !append_random())) {
        record();
        break;
      }
      continue;
    }
    record();
    if (iterations_ == opt_.max_iterations || nconv_ + k_ >= n_) break;
    correction(j_cur);
    if (shift_ != tau_) ++j_cur;
    if (!append(t_.data())) {
      // The correction lies in span[Q V] to rounding: expand with the residual, else randomly.
      ++fallbacks_;
      if (!append(r_.data()) && !append_random()) break;
    }
    ++iterations_;
  }
  return finish();
}

template <class T>
IterativeEigResult<T> JacobiDavidson<T>::finish() {
  const index_t found = nconv_ - d_;
  const index_t extra = std::min(nev_ - found, k_);
  const index_t mtot = found + extra;
  DenseMatrix<T> x(n_, mtot);
  std::vector<double> th(locked_theta_);
  std::copy(qcol(d_), qcol(d_) + n_ * found, x.data());
  if (extra > 0) {
    // Best current approximations: [sel, ranked extraction vectors] orthonormalized, then a
    // Rayleigh-Ritz step inside their span, so the vectors are B-orthonormal Ritz vectors.
    const index_t k = k_;
    DenseMatrix<T> c = orthonormal_candidates(extra);
    DenseMatrix<T> red = congruence(leading(hk_, k), c);
    std::vector<double> nu(static_cast<std::size_t>(extra));
    lapack::heev('V', 'L', fi(extra), red.data(), fi(extra), nu.data());
    DenseMatrix<T> cr(k, extra);
    lapack::gemm('N', 'N', fi(k), fi(extra), fi(extra), T(1), c.data(), fi(k), red.data(), fi(extra), T(0),
                 cr.data(), fi(k));
    lapack::gemm('N', 'N', fi(n_), fi(extra), fi(k), T(1), vcol(0), fi(n_), cr.data(), fi(k), T(0),
                 x.col(found), fi(n_));
    for (double v : nu) th.push_back(tau_ + v);
  }
  std::vector<double> res, eta;
  eigen_detail::fresh_residuals(a_, b_, n_, x.data(), mtot, th, norm_a_, norm_b_, res, eta);
  std::vector<index_t> order(static_cast<std::size_t>(mtot));
  std::iota(order.begin(), order.end(), index_t{0});
  std::stable_sort(order.begin(), order.end(), [&](index_t p, index_t q) {
    const bool cp = eta[p] <= opt_.tol, cq = eta[q] <= opt_.tol;
    if (cp != cq) return cp;
    return std::abs(th[p] - tau_) < std::abs(th[q] - tau_);
  });
  IterativeEigResult<T> out;
  out.vectors.resize(n_, mtot);
  for (index_t l = 0; l < mtot; ++l) {
    const index_t j = order[l];
    std::copy(x.col(j), x.col(j) + n_, out.vectors.col(l));
    out.theta.push_back(th[j]);
    out.estimate.push_back(eta[j]);
    out.residual.push_back(res[j]);
    if (eta[j] <= opt_.tol) ++out.nconv;
  }
  out.iterations = iterations_;
  out.restarts = restarts_;
  out.inner_iterations = inner_total_;
  out.tracking_steps = tau_steps_;
  out.fallback_expansions = fallbacks_;
  out.op_applies = a_.applications() - a0_;
  out.b_applies = b_ ? b_->applications() - b0_ : 0;
  out.precond_applies = m_ ? m_->applications() - m0_ : 0;
  out.norm_a = norm_a_;
  out.norm_b = norm_b_;
  out.converged = out.nconv >= nev_;
  out.history = std::move(history_);
  return out;
}

}  // namespace

template <class T>
IterativeEigResult<T> jacobi_davidson(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* precond,
                                      double target, const JdOptions<T>& options) {
  JacobiDavidson<T> solver(a, b, precond, target, options);
  return solver.run();
}

#define HS_INSTANTIATE_JD(T)                                                                          \
  template IterativeEigResult<T> jacobi_davidson<T>(LinearOperator<T>&, LinearOperator<T>*,             \
                                                    LinearOperator<T>*, double, const JdOptions<T>&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_JD)
#undef HS_INSTANTIATE_JD

}  // namespace hs
