// LOBPCG (A. V. Knyazev, "Toward the optimal preconditioned eigensolver: locally optimal block
// preconditioned conjugate gradient method", SISC 23(2), 2001) with soft locking and the robust
// basis of U. Hetmaniuk & R. Lehoucq ("Basis selection in LOBPCG", JCP 218, 2006) and J. A.
// Duersch, M. Shao, C. Yang & M. Gu ("A robust and efficient implementation of LOBPCG", SISC
// 40(5), 2018). Contract: include/hsolve/eigen_iterative.hpp.
//
// State. S = [X | P | W] (n x (m + np + nw)) with A S and B S stored next to it (B S only with B);
// X holds the m current Ritz vectors in ranking order, P the "conjugate" directions, W the
// preconditioned residuals of the active columns. X and P are B-orthonormal and mutually
// B-orthogonal up to rounding; W is explicitly B-orthonormalized against [Y X P] and itself.
//
// One iteration:
//   1. R = A X - B X Theta (stored products); eta_j; active = unconverged columns (soft locking:
//      converged columns stay in X and in the Rayleigh-Ritz problem but get no new direction).
//   2. W = T R(:, active) (or R), B-orthonormalized against [Y X P] with rank dropping
//      (eigen_detail::b_orthonormalize); A W.
//   3. Rayleigh-Ritz on span(S) with the explicitly computed G_A = S^H A S and G_B = S^H B S.
//      Using the computed G_B (not I) keeps the step exact for the computed basis, so the slow
//      drift of [X P] from B-orthonormality is corrected by every step (C^H G_B C = I makes the
//      new X B-orthonormal to the accuracy of G_B). If the scaled G_B is ill conditioned
//      (mu_min < 1e-6 mu_max), P is dropped for this step (Hetmaniuk-Lehoucq restart) and the
//      problem is solved on [X W], dropping any remaining near-null directions.
//   4. X <- S C_sel. The new P spans the part of the new X outside the old one (Duersch et al.,
//      Alg. 4): Z0 = rows (P, W) of C_sel(:, active), made G_B-orthogonal to C_sel and
//      G_B-orthonormal (SVQB with dropping) inside the small space, P <- S Z. Then [X P] is
//      B-orthonormal by construction and no n-sized orthogonalization of P is ever needed.
//      [X P] <- S [C_sel Z] is computed in place, one block of rows at a time.
// The products A X, B X, A P, B P follow from the same small-space combinations (no operator
// applications); before declaring convergence A X and B X are recomputed from scratch and the
// residuals re-checked, so the returned estimates are true backward errors.
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

using eigen_detail::OrthoBlock;
using lapack::fint;

/// A new direction is kept when more than this fraction of its B-norm survives the projection
/// against [Y X P]; below it the direction is rounding noise inside that span.
constexpr double kDropRel = 1e-10;
/// The Rayleigh-Ritz basis [X P W] counts as ill conditioned (P is dropped) below this ratio of
/// the extreme eigenvalues of its scaled Gram matrix.
constexpr double kIllCond = 1e-6;
/// Attempts at completing a start block with fresh random vectors.
constexpr int kStartAttempts = 4;

inline fint fi(index_t x) { return static_cast<fint>(x); }

template <class T>
class Lobpcg {
 public:
  Lobpcg(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* t, Which which,
         const LobpcgOptions<T>& options);
  IterativeEigResult<T> run();

 private:
  T* bs_data() { return b_ ? bs_.data() : s_.data(); }
  T* bs_col(index_t j) { return bs_data() + j * n_; }

  void load_deflation();
  void init_x();
  void residuals();
  index_t wanted_converged() const;
  double wanted_max_eta() const;
  void refresh_products();
  bool iterate();
  std::vector<OrthoBlock<T>> constraint_blocks(index_t xp_cols);
  void fill_random_columns(T* w, index_t count);

  LinearOperator<T>& a_;
  LinearOperator<T>* b_;
  LinearOperator<T>* t_;
  const Which which_;
  const LobpcgOptions<T>& opt_;

  index_t n_ = 0, nev_ = 0, m_ = 0, d_ = 0, np_ = 0;
  double norm_a_ = 0.0, norm_b_ = 1.0;
  const T* y_ = nullptr;  ///< deflation block (options-owned)
  DenseMatrix<T> by_;     ///< B Y (B only)
  DenseMatrix<T> s_, as_, bs_;
  DenseMatrix<T> r_;      ///< residual block R = A X - B X Theta
  std::vector<double> theta_, res_, eta_;
  std::vector<char> conv_;
  bool fresh_ = false;    ///< A X, B X were recomputed from scratch for the current X
  std::uint64_t next_stream_ = 0;
  index_t iterations_ = 0, restarts_ = 0;
  index_t a0_ = 0, b0_ = 0, t0_ = 0;
  std::vector<RestartRecord> history_;
};

template <class T>
Lobpcg<T>::Lobpcg(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* t, Which which,
                  const LobpcgOptions<T>& options)
    : a_(a), b_(b), t_(t), which_(which), opt_(options) {
  n_ = a.size();
  nev_ = options.nev;
  if (b && b->size() != n_) HS_THROW(InvalidInput, "lobpcg: B has order " << b->size() << ", A " << n_);
  if (t && t->size() != n_)
    HS_THROW(InvalidInput, "lobpcg: preconditioner has order " << t->size() << ", A " << n_);
  if (which != Which::SmallestAlgebraic && which != Which::LargestAlgebraic)
    HS_THROW(InvalidInput, "lobpcg: which must be SmallestAlgebraic or LargestAlgebraic");
  if (nev_ < 1) HS_THROW(InvalidInput, "lobpcg: nev must be >= 1, got " << nev_);
  if (!(options.tol > 0.0) || !std::isfinite(options.tol))
    HS_THROW(InvalidInput, "lobpcg: tol must be positive and finite, got " << options.tol);
  if (options.max_iterations < 0) HS_THROW(InvalidInput, "lobpcg: max_iterations must be >= 0");
  if (!(options.norm_a >= 0.0) || !(options.norm_b >= 0.0) || !std::isfinite(options.norm_a) ||
      !std::isfinite(options.norm_b))
    HS_THROW(InvalidInput, "lobpcg: norm_a / norm_b must be finite and >= 0");
  if (options.deflation && options.deflation->cols() > 0) {
    if (options.deflation->rows() != n_)
      HS_THROW(InvalidInput, "lobpcg: deflation block has " << options.deflation->rows() << " rows, A has order "
                                                            << n_);
    d_ = options.deflation->cols();
    y_ = options.deflation->data();
  }
  const index_t n_free = n_ - d_;
  if (options.block_size < 0) HS_THROW(InvalidInput, "lobpcg: block_size must be >= 0");
  m_ = options.block_size == 0 ? std::min(n_free, nev_ + std::max<index_t>(2, std::min<index_t>(nev_, 8)))
                               : options.block_size;
  if (m_ < nev_) HS_THROW(InvalidInput, "lobpcg: block_size " << m_ << " < nev " << nev_);
  if (m_ > n_free)
    HS_THROW(InvalidInput, "lobpcg: block_size " << m_ << " exceeds the available dimension n - d = " << n_free);
  if (options.initial) {
    const auto* ini = options.initial;
    if (ini->rows() != n_ || ini->cols() < 1 || ini->cols() > m_)
      HS_THROW(InvalidInput, "lobpcg: initial block is " << ini->rows() << " x " << ini->cols() << ", need " << n_
                                                         << " x [1, " << m_ << "]");
  }
  a0_ = a.applications();
  b0_ = b ? b->applications() : 0;
  t0_ = t ? t->applications() : 0;
  s_.resize(n_, 3 * m_);
  as_.resize(n_, 3 * m_);
  if (b_) bs_.resize(n_, 3 * m_);
  r_.resize(n_, m_);
}

template <class T>
void Lobpcg<T>::load_deflation() {
  if (d_ == 0) return;
  const T* by = y_;
  if (b_) {
    by_.resize(n_, d_);
    b_->apply(y_, n_, by_.data(), n_, d_);
    eigen_detail::require_finite(n_, by_.data(), d_, "lobpcg: B * deflation");
    by = by_.data();
  }
  eigen_detail::require_b_orthonormal(n_, d_, y_, by, "lobpcg");
}

template <class T>
std::vector<OrthoBlock<T>> Lobpcg<T>::constraint_blocks(index_t xp_cols) {
  std::vector<OrthoBlock<T>> blocks;
  if (d_ > 0) blocks.push_back({y_, b_ ? by_.data() : y_, d_});
  if (xp_cols > 0) blocks.push_back({s_.data(), bs_data(), xp_cols});
  return blocks;
}

template <class T>
void Lobpcg<T>::fill_random_columns(T* w, index_t count) {
  for (index_t j = 0; j < count; ++j) linalg::fill_random(w + j * n_, n_, opt_.seed, next_stream_++);
}

template <class T>
void Lobpcg<T>::init_x() {
  index_t p0 = 0;
  if (opt_.initial) {
    p0 = opt_.initial->cols();
    std::copy(opt_.initial->data(), opt_.initial->data() + n_ * p0, s_.data());
    eigen_detail::require_finite(n_, s_.data(), p0, "lobpcg: initial block");
  }
  fill_random_columns(s_.col(p0), m_ - p0);
  index_t kept = eigen_detail::b_orthonormalize<T>(n_, constraint_blocks(0), s_.data(), bs_data(), m_, b_, kDropRel);
  for (int attempt = 0; kept < m_ && attempt < kStartAttempts; ++attempt) {
    fill_random_columns(s_.col(kept), m_ - kept);
    kept += eigen_detail::b_orthonormalize<T>(n_, constraint_blocks(kept), s_.col(kept), bs_col(kept), m_ - kept, b_,
                                              kDropRel);
  }
  if (kept < m_)
    HS_THROW(InvalidInput, "lobpcg: cannot build " << m_ << " B-orthonormal start vectors outside span(Y) (got "
                                                   << kept << "): B not positive definite?");
  a_.apply(s_.data(), n_, as_.data(), n_, m_);
  eigen_detail::require_finite(n_, as_.data(), m_, "lobpcg: A * X");

  // Rayleigh-Ritz on X alone.
  DenseMatrix<T> ga(m_, m_), gb(m_, m_);
  linalg::gram(n_, m_, s_.data(), n_, as_.data(), n_, ga.data(), m_);
  linalg::gram(n_, m_, s_.data(), n_, bs_data(), n_, gb.data(), m_);
  std::vector<index_t> idx(static_cast<std::size_t>(m_));
  std::iota(idx.begin(), idx.end(), index_t{0});
  std::vector<double> th;
  DenseMatrix<T> c;
  eigen_detail::rayleigh_ritz(ga, gb, idx, kIllCond, /*force=*/true, th, c);
  HS_ASSERT(static_cast<index_t>(th.size()) == m_, "lobpcg: start block lost rank in Rayleigh-Ritz");
  if (which_ == Which::LargestAlgebraic) {
    std::reverse(th.begin(), th.end());
    DenseMatrix<T> cr(m_, m_);
    for (index_t l = 0; l < m_; ++l) std::copy(c.col(m_ - 1 - l), c.col(m_ - l), cr.col(l));
    c = std::move(cr);
  }
  eigen_detail::rotate_in_place(n_, s_.data(), m_, c.data(), m_, m_);
  eigen_detail::rotate_in_place(n_, as_.data(), m_, c.data(), m_, m_);
  if (b_) eigen_detail::rotate_in_place(n_, bs_.data(), m_, c.data(), m_, m_);
  theta_ = th;
  np_ = 0;
  residuals();
}

template <class T>
void Lobpcg<T>::residuals() {
  res_.assign(static_cast<std::size_t>(m_), 0.0);
  eta_.assign(static_cast<std::size_t>(m_), 0.0);
  conv_.assign(static_cast<std::size_t>(m_), 0);
  for (index_t j = 0; j < m_; ++j) {
    T* r = r_.col(j);
    const T* ax = as_.col(j);
    const T* bx = bs_col(j);
    const T th = T(theta_[j]);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n_; ++i) r[i] = ax[i] - th * bx[i];
    res_[j] = linalg::norm2(n_, r);
    eta_[j] = eigen_detail::backward_error(res_[j], theta_[j], linalg::norm2(n_, s_.col(j)), norm_a_, norm_b_);
    conv_[j] = eta_[j] <= opt_.tol ? 1 : 0;
  }
}

template <class T>
index_t Lobpcg<T>::wanted_converged() const {
  index_t c = 0;
  for (index_t j = 0; j < nev_; ++j) c += conv_[j];
  return c;
}

template <class T>
double Lobpcg<T>::wanted_max_eta() const {
  double e = 0.0;
  for (index_t j = 0; j < nev_; ++j)
    if (!conv_[j]) e = std::max(e, std::isnan(eta_[j]) ? std::numeric_limits<double>::infinity() : eta_[j]);
  return e;
}

template <class T>
void Lobpcg<T>::refresh_products() {
  a_.apply(s_.data(), n_, as_.data(), n_, m_);
  eigen_detail::require_finite(n_, as_.data(), m_, "lobpcg: A * X");
  if (b_) {
    b_->apply(s_.data(), n_, bs_.data(), n_, m_);
    eigen_detail::require_finite(n_, bs_.data(), m_, "lobpcg: B * X");
  }
  residuals();
  fresh_ = true;
}

/// One LOBPCG step; false when no new direction could be generated (stagnation).
template <class T>
bool Lobpcg<T>::iterate() {
  std::vector<index_t> active;
  for (index_t j = 0; j < m_; ++j)
    if (!conv_[j]) active.push_back(j);
  const index_t nact = static_cast<index_t>(active.size());
  HS_ASSERT(nact > 0, "lobpcg: iterate() without active columns");

  // W = T R(:, active), B-orthonormalized against [Y X P].
  const index_t wc = m_ + np_;
  for (index_t l = 0; l < nact; ++l)
    if (active[l] != l) std::copy(r_.col(active[l]), r_.col(active[l]) + n_, r_.col(l));
  if (t_) {
    t_->apply(r_.data(), n_, s_.col(wc), n_, nact);
  } else {
    std::copy(r_.data(), r_.data() + n_ * nact, s_.col(wc));
  }
  index_t nw =
      eigen_detail::b_orthonormalize<T>(n_, constraint_blocks(m_ + np_), s_.col(wc), bs_col(wc), nact, b_, kDropRel);
  if (nw == 0) {
    // The preconditioned residuals lie in span[Y X P] (to rounding) although some pairs are not
    // converged: continue with random directions rather than stopping.
    fill_random_columns(s_.col(wc), nact);
    nw = eigen_detail::b_orthonormalize<T>(n_, constraint_blocks(m_ + np_), s_.col(wc), bs_col(wc), nact, b_,
                                           kDropRel);
    if (nw == 0) return false;
  }
  a_.apply(s_.col(wc), n_, as_.col(wc), n_, nw);
  eigen_detail::require_finite(n_, as_.col(wc), nw, "lobpcg: A * W");

  // Rayleigh-Ritz on [X P W].
  const index_t ks = m_ + np_ + nw;
  DenseMatrix<T> ga(ks, ks), gb(ks, ks);
  linalg::gram(n_, ks, s_.data(), n_, as_.data(), n_, ga.data(), ks);
  linalg::gram(n_, ks, s_.data(), n_, bs_data(), n_, gb.data(), ks);
  eigen_detail::hermitize(gb);
  std::vector<index_t> idx(static_cast<std::size_t>(ks));
  std::iota(idx.begin(), idx.end(), index_t{0});
  std::vector<double> th;
  DenseMatrix<T> c;
  if (!eigen_detail::rayleigh_ritz(ga, gb, idx, kIllCond, /*force=*/np_ == 0, th, c)) {
    // Ill-conditioned [X P W]: drop P for this step (Hetmaniuk-Lehoucq restart).
    idx.clear();
    for (index_t i = 0; i < ks; ++i)
      if (i < m_ || i >= m_ + np_) idx.push_back(i);
    ++restarts_;
    eigen_detail::rayleigh_ritz(ga, gb, idx, kIllCond, /*force=*/true, th, c);
  }
  const index_t r = static_cast<index_t>(th.size());
  HS_ASSERT(r >= m_, "lobpcg: Rayleigh-Ritz basis of rank " << r << " < block size " << m_);

  // Selected Ritz vectors (ranking order) and the coefficients of the new P.
  DenseMatrix<T> cz(ks, m_ + nact);  // [C_sel Z]
  std::vector<double> th_sel(static_cast<std::size_t>(m_));
  for (index_t l = 0; l < m_; ++l) {
    const index_t src = which_ == Which::SmallestAlgebraic ? l : r - 1 - l;
    th_sel[l] = th[src];
    std::copy(c.col(src), c.col(src) + ks, cz.col(l));
  }
  // Z0 = rows (P, W) of C_sel(:, active).
  DenseMatrix<T> z(ks, nact);
  for (index_t l = 0; l < nact; ++l)
    for (index_t i = m_; i < ks; ++i) z(i, l) = cz(i, active[l]);
  // G_B-norms before the projection, then two passes of z -= C_sel (C_sel^H G_B z).
  DenseMatrix<T> gbz(ks, nact), coef(m_, nact);
  lapack::gemm('N', 'N', fi(ks), fi(nact), fi(ks), T(1), gb.data(), fi(ks), z.data(), fi(ks), T(0), gbz.data(),
               fi(ks));
  std::vector<double> before(static_cast<std::size_t>(nact));
  for (index_t l = 0; l < nact; ++l) {
    T s = T(0);
    for (index_t i = 0; i < ks; ++i) s += conj_if(z(i, l)) * gbz(i, l);
    before[l] = std::sqrt(std::max(std::real(s), 0.0));
  }
  for (int pass = 0; pass < 2; ++pass) {
    if (pass > 0)
      lapack::gemm('N', 'N', fi(ks), fi(nact), fi(ks), T(1), gb.data(), fi(ks), z.data(), fi(ks), T(0),
                   gbz.data(), fi(ks));
    lapack::gemm('C', 'N', fi(m_), fi(nact), fi(ks), T(1), cz.data(), fi(ks), gbz.data(), fi(ks), T(0),
                 coef.data(), fi(m_));
    lapack::gemm('N', 'N', fi(ks), fi(nact), fi(m_), T(-1), cz.data(), fi(ks), coef.data(), fi(m_), T(1), z.data(),
                 fi(ks));
  }
  lapack::gemm('N', 'N', fi(ks), fi(nact), fi(ks), T(1), gb.data(), fi(ks), z.data(), fi(ks), T(0), gbz.data(),
               fi(ks));
  DenseMatrix<T> gz(nact, nact);
  lapack::gemm('C', 'N', fi(nact), fi(nact), fi(ks), T(1), z.data(), fi(ks), gbz.data(), fi(ks), T(0), gz.data(),
               fi(nact));
  std::vector<index_t> zkeep;
  for (index_t l = 0; l < nact; ++l) {
    const double after = std::sqrt(std::max(std::real(gz(l, l)), 0.0));
    if (after > kDropRel * before[l] && after > 0.0) zkeep.push_back(l);
  }
  index_t np_new = 0;
  if (!zkeep.empty()) {
    // G_B-orthonormalize the kept columns of Z (SVQB with dropping) and append them to cz. In
    // exact arithmetic rank(Z) <= ks - m (Z is G_B-orthogonal to the m columns of C_sel); keep
    // the most independent directions if rounding left more.
    const auto nk = static_cast<index_t>(zkeep.size());
    DenseMatrix<T> gzk(nk, nk), zk(ks, nk), mz;
    for (index_t j = 0; j < nk; ++j) {
      std::copy(z.col(zkeep[j]), z.col(zkeep[j]) + ks, zk.col(j));
      for (index_t i = 0; i < nk; ++i) gzk(i, j) = gz(zkeep[i], zkeep[j]);
    }
    np_new = std::min(eigen_detail::svqb(gzk, mz), ks - m_);
    lapack::gemm('N', 'N', fi(ks), fi(np_new), fi(nk), T(1), zk.data(), fi(ks), mz.data(), fi(nk), T(0),
                 cz.col(m_), fi(ks));
  }
  HS_ASSERT(m_ + np_new <= ks, "lobpcg: new [X P] wider than the basis");

  // [X P] <- S [C_sel Z] (and the A, B products), in place.
  const index_t kx = m_ + np_new;
  eigen_detail::rotate_in_place(n_, s_.data(), ks, cz.data(), ks, kx);
  eigen_detail::rotate_in_place(n_, as_.data(), ks, cz.data(), ks, kx);
  if (b_) eigen_detail::rotate_in_place(n_, bs_.data(), ks, cz.data(), ks, kx);
  np_ = np_new;
  theta_ = th_sel;
  residuals();
  fresh_ = false;
  return true;
}

template <class T>
IterativeEigResult<T> Lobpcg<T>::run() {
  load_deflation();
  norm_a_ = opt_.norm_a > 0.0 ? opt_.norm_a : eigen_detail::estimate_norm2(a_, opt_.seed ^ 0x6c6f62706367ULL);
  norm_b_ = !b_ ? 1.0 : (opt_.norm_b > 0.0 ? opt_.norm_b : eigen_detail::estimate_norm2(*b_, opt_.seed ^ 0x62ULL));
  init_x();
  for (;;) {
    index_t nconv = wanted_converged();
    if (nconv == nev_ && !fresh_) {
      // Converged according to the recurrence products: confirm with fresh ones.
      refresh_products();
      nconv = wanted_converged();
    }
    RestartRecord rec;
    rec.restart = iterations_;
    rec.nconv = nconv;
    rec.max_estimate = wanted_max_eta();
    rec.op_applies = a_.applications() - a0_;
    history_.push_back(rec);
    if (nconv == nev_ || iterations_ == opt_.max_iterations) break;
    if (!iterate()) break;
    ++iterations_;
  }

  // The nev best ranked columns of X, converged ones first.
  IterativeEigResult<T> out;
  std::vector<double> res(res_.begin(), res_.begin() + nev_), eta(eta_.begin(), eta_.begin() + nev_);
  std::vector<double> th(theta_.begin(), theta_.begin() + nev_);
  if (!fresh_) eigen_detail::fresh_residuals(a_, b_, n_, s_.data(), nev_, th, norm_a_, norm_b_, res, eta);
  std::vector<index_t> order(static_cast<std::size_t>(nev_));
  std::iota(order.begin(), order.end(), index_t{0});
  std::stable_partition(order.begin(), order.end(), [&](index_t j) { return eta[j] <= opt_.tol; });
  out.vectors.resize(n_, nev_);
  for (index_t l = 0; l < nev_; ++l) {
    const index_t j = order[l];
    std::copy(s_.col(j), s_.col(j) + n_, out.vectors.col(l));
    out.theta.push_back(th[j]);
    out.estimate.push_back(eta[j]);
    out.residual.push_back(res[j]);
    if (eta[j] <= opt_.tol) ++out.nconv;
  }
  out.iterations = iterations_;
  out.restarts = restarts_;
  out.op_applies = a_.applications() - a0_;
  out.b_applies = b_ ? b_->applications() - b0_ : 0;
  out.precond_applies = t_ ? t_->applications() - t0_ : 0;
  out.norm_a = norm_a_;
  out.norm_b = norm_b_;
  out.converged = out.nconv >= nev_;
  out.history = std::move(history_);
  return out;
}

}  // namespace

template <class T>
IterativeEigResult<T> lobpcg(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* precond,
                             Which which, const LobpcgOptions<T>& options) {
  Lobpcg<T> solver(a, b, precond, which, options);
  return solver.run();
}

#define HS_INSTANTIATE_LOBPCG(T)                                                                    \
  template IterativeEigResult<T> lobpcg<T>(LinearOperator<T>&, LinearOperator<T>*, LinearOperator<T>*, \
                                           Which, const LobpcgOptions<T>&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_LOBPCG)
#undef HS_INSTANTIATE_LOBPCG

}  // namespace hs
