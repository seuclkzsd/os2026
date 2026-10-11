// Thick-restart Lanczos == Hermitian Krylov-Schur (Stewart 2001; Wu & Simon 2000) with full CGS2
// reorthogonalization, locking by zeroing couplings, and an optional deflation block D.
//
// Notation. M = b_inner (HPD) or I, <x, y>_M = y^H M x; every basis below is M-orthonormal.
//   D   n x d deflation block, copied into the leading d columns of the basis storage so that a
//       new vector is orthogonalized against [D V] by ONE CGS2 call (3 M applications per step in
//       recompute mode instead of 6 for two separate calls; same projector since D is
//       M-orthogonal to V).
//   V   n x m Krylov basis v_0..v_{m-1}; column m holds the residual vector v_m.
//   H   m x m Hermitian projected matrix, H(i, j) = <Op v_j, v_i>_M.
// Every cycle maintains the Krylov-Schur relation
//   Op V = V H + beta v_m e_{m-1}^T,
// so a Ritz pair (theta_i, V s_i) of H has the residual estimate |beta s_i(m-1)|. The first cycle
// is plain Lanczos (H tridiagonal). A restart keeps the k best Ritz pairs:
//   V <- V S(:, 0:k), H <- diag(theta_0..theta_{k-1}), v_k <- v_m, b = beta S(m-1, 0:k),
// and H gets the arrow H(k, 0:k) = b (H(0:k, k) = conj(b)) followed again by tridiagonal Lanczos
// steps from v_k. Only the theoretical entries of H are stored (diagonal, arrow, tridiagonal):
// the remaining CGS2 coefficients vanish in exact arithmetic and only purify the new vector.
//
// Dropped couplings. Setting a coupling to zero (locking, breakdown) makes the stored relation the
// one of Op - E, with E a Hermitian rank-2 term of norm |dropped coupling| (e.g. locking b_l:
// E = b_l v_k v_l^H M + conj(b_l) v_l v_k^H M, v_k the residual vector). The estimates do not see
// E, so the true residual of a Ritz vector can exceed its estimate by up to |dropped coupling|.
// A coupling is therefore only dropped at the rounding level of the quantities it touches; this
// keeps "estimate <= tol |theta| => converged" honest for pairs of any magnitude, including a
// wide dynamic range of |theta| (shift-invert near an eigenvalue).
//
// Locking. A kept converged pair gets b_l := 0, which decouples it exactly from later
// Rayleigh-Ritz steps (its row and column of H are diagonal), once |b_l| <= kLockTol * (smallest
// |theta| reference among the wanted pairs): E is then a few roundoffs relative to every wanted
// pair. Zeroing a tol-sized coupling as soon as the pair converges would hide conj(b_l) <v_k, y>_M
// in the residual of every other Ritz vector y; relative to a smaller |theta_y| that breaks the
// convergence test (measured: true residuals 28x the estimate at tol = 1e-6). Scaling the
// threshold by ||Op|| instead fails the same way when ||Op|| >> |theta_y|. Pairs converged only
// to tol therefore stay coupled and keep improving; the estimates remain exact for the computed
// relation.
//
// Breakdown. beta <= kBreakdownTol * ||Op v_j||_M means that Op v_j lies in span[D V] up to the
// rounding error of computing it: the coupling is set to 0 and the next basis vector is a fresh
// deterministic random vector orthogonalized against [D V]; H becomes block diagonal and Lanczos
// continues there. The test is local to the step on purpose: compared with a global ||Op|| it
// would discard genuine couplings that are small only relative to the largest eigenvalue, and
// the wanted pairs with small |theta| would be declared converged (estimate 0) with large true
// residuals or be skipped altogether. Any beta above rounding level is kept: CGS2 still yields a
// new direction orthogonal to working precision, since the orthogonality loss of the second
// pass is about eps * (rounding noise of the first pass) / beta.
//
// store_bv. The stored M*V columns are never updated by the CGS2 recurrence bw -= (M V) c: the
// error of the stored columns would enter the new column amplified by ||Op v_j||_M / beta and
// compound over the restarts (measured: M-orthonormality lost to 5e-5 after 1000 restarts). Each
// new column instead gets M applied to the orthogonalized vector, still one application per step.
//
// Deflation semantics. The basis stays M-orthogonal to span(D), so the solver computes eigenpairs
// of the deflated operator P Op P (P = I - D D^H M). The estimates measure ||P (Op x - theta
// x)||_M; the components of Op x along span(D), bounded by the residuals of D's own columns, are
// not included (exact when span(D) is invariant; a Rayleigh-Ritz step over [D X] removes them).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "eigen/lanczos_detail.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {

namespace {

using lapack::fint;

constexpr double kEps = std::numeric_limits<double>::epsilon();
/// beta <= kBreakdownTol * ||Op v_j||_M is an invariant subspace to rounding: a few dozen
/// roundoffs of the step's own magnitude, the accuracy to which Op v_j and its CGS2 projection
/// are computed anyway (see "Breakdown" above).
constexpr double kBreakdownTol = 64 * kEps;
/// A converged pair is locked (coupling zeroed) once |b_l| <= kLockTol * (smallest reference
/// |theta| of the wanted pairs): a few roundoffs relative to every wanted pair.
constexpr double kLockTol = 4 * kEps;
/// Largest accepted |(D^H M D - I)_ij|: D is used as an orthonormal projector basis, so a
/// noticeably non-orthonormal block would silently corrupt every orthogonalization.
constexpr double kDeflationOrthoTol = 1e-8;
/// A user start vector must keep at least this fraction of its M-norm after projecting out D.
constexpr double kStartMinRatio = 1e-12;
/// Random vectors are drawn only while the free space is not exhausted. With an M-orthonormal
/// basis a random vector keeps most of its M-norm, so losing more than this fraction to [D V]
/// means that M (b_inner) is not positive definite.
constexpr double kRandomMinRatio = 1e-8;
/// Rows per block of the in-place basis rotation (temporary of kRowBlock x k scalars instead of
/// a second n x k basis).
constexpr index_t kRowBlock = 4096;
/// Columns of M*D formed at a time when checking a deflation block in recompute mode.
constexpr index_t kGramChunk = 64;

/// Adds the lifetime of the scope to a counter.
class ScopedTimer {
 public:
  explicit ScopedTimer(double& acc) : acc_(acc) {}
  ~ScopedTimer() { acc_ += timer_.seconds(); }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  double& acc_;
  Timer timer_;
};

template <class T>
class ThickRestartLanczos {
 public:
  ThickRestartLanczos(LinearOperator<T>& op, LinearOperator<T>* b_inner, Which which,
                      const KrylovSchurOptions<T>& options, eigen_detail::LanczosTimings& timings);

  HermitianEigResult<T> run();

 private:
  T* vcol(index_t j) { return basis_.col(d_ + j); }
  T* bvcol(index_t j) { return store_bv_ ? bbasis_.col(d_ + j) : nullptr; }

  void load_deflation();
  void set_start_vector();
  std::pair<double, double> orthogonalize(index_t j);
  void normalize(index_t j, double norm);
  void set_random_vector(index_t j);
  void expand(index_t first);
  void rayleigh_ritz();
  index_t record_history();
  void restart(index_t keep);
  void rotate_in_place(T* v, const DenseMatrix<T>& s);
  HermitianEigResult<T> finish();

  LinearOperator<T>& op_;
  LinearOperator<T>* b_inner_;
  const Which which_;
  const KrylovSchurOptions<T>& opt_;
  eigen_detail::LanczosTimings& time_;

  index_t n_ = 0;
  index_t d_ = 0;  ///< deflation columns
  index_t nev_ = 0;
  index_t ncv_ = 0;          ///< basis size m of every cycle
  bool full_space_ = false;  ///< ncv == n - d: one cycle spans the whole free space (exact)
  bool store_bv_ = false;
  linalg::ApplyFn<T> apply_m_;  ///< recompute mode only

  DenseMatrix<T> basis_;           ///< [D | v_0 .. v_ncv], n x (d + ncv + 1)
  DenseMatrix<T> bbasis_;          ///< M * basis_ (store_bv only)
  std::vector<T> mw_;              ///< M*w workspace (recompute mode)
  std::vector<T> coef_;            ///< CGS2 coefficients of the current vector against [D V]
  DenseMatrix<T> h_;               ///< ncv x ncv projected matrix, both triangles stored
  double beta_ = 0.0;              ///< coupling of v_{ncv-1} to the residual vector v_ncv
  std::uint64_t next_stream_ = 0;  ///< fill_random stream: 0 = start vector, then breakdowns

  // Rayleigh-Ritz results of the current cycle.
  std::vector<double> theta_;   ///< ascending (heev order)
  DenseMatrix<T> s_;            ///< eigenvectors of H, column i <-> theta_[i]
  std::vector<double> est_;     ///< residual estimates
  std::vector<char> conv_;      ///< est_ <= tol * reference
  std::vector<double> ref_;     ///< |theta|, or max |theta| when theta == 0
  std::vector<index_t> order_;  ///< indices into theta_ by decreasing preference (which_)

  index_t restarts_ = 0;
  index_t op_applies0_ = 0;
  std::vector<RestartRecord> history_;
};

template <class T>
ThickRestartLanczos<T>::ThickRestartLanczos(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                            Which which, const KrylovSchurOptions<T>& options,
                                            eigen_detail::LanczosTimings& timings)
    : op_(op), b_inner_(b_inner), which_(which), opt_(options), time_(timings) {
  n_ = op.size();
  nev_ = options.nev;
  if (b_inner && b_inner->size() != n_)
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: b_inner has order " << b_inner->size() << ", operator " << n_);
  if (nev_ < 1) HS_THROW(InvalidInput, "lanczos_krylov_schur: nev must be >= 1, got " << nev_);
  if (!(options.tol > 0.0) || !std::isfinite(options.tol))
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: tol must be positive and finite, got " << options.tol);
  if (options.max_restarts < 0)
    HS_THROW(InvalidInput, "lanczos_krylov_schur: max_restarts must be >= 0");
  if (options.ncv < 0) HS_THROW(InvalidInput, "lanczos_krylov_schur: ncv must be >= 0");
  // An empty deflation block (0 columns) means "no deflation" whatever its row count.
  if (options.deflation && options.deflation->cols() > 0) {
    if (options.deflation->rows() != n_)
      HS_THROW(InvalidInput, "lanczos_krylov_schur: deflation block has "
                                 << options.deflation->rows() << " rows, operator order " << n_);
    d_ = options.deflation->cols();
  }
  const index_t n_free = n_ - d_;
  if (n_free < nev_)
    HS_THROW(InvalidInput, "lanczos_krylov_schur: nev = "
                               << nev_ << " exceeds the available dimension n - d = " << n_free);
  if (options.ncv == 0) {
    ncv_ = std::min(n_free, std::max(2 * nev_ + 1, nev_ + 16));
  } else if (options.ncv >= n_free) {
    ncv_ = n_free;
  } else if (options.ncv <= nev_) {
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: ncv = " << options.ncv << " must exceed nev = " << nev_
                                            << " (or reach n - d = " << n_free << ")");
  } else {
    ncv_ = options.ncv;
  }
  full_space_ = ncv_ == n_free;
  if (options.initial) {
    if (options.initial->rows() != n_ || options.initial->cols() < 1)
      HS_THROW(InvalidInput, "lanczos_krylov_schur: initial block is "
                                 << options.initial->rows() << " x " << options.initial->cols()
                                 << ", need " << n_ << " x >=1");
  }
  // store_bv only matters for a non-identity inner product; with M = I there is nothing to store.
  store_bv_ = options.store_bv && b_inner != nullptr;

  basis_.resize(n_, d_ + ncv_ + 1);
  if (store_bv_) bbasis_.resize(n_, d_ + ncv_ + 1);
  if (b_inner && !store_bv_) {
    mw_.resize(static_cast<std::size_t>(n_));
    apply_m_ = [this](const T* x, T* y) { b_inner_->apply(x, n_, y, n_, 1); };
  }
  coef_.resize(static_cast<std::size_t>(d_ + ncv_ + 1));
  h_.resize(ncv_, ncv_);
  op_applies0_ = op.applications();
}

template <class T>
void ThickRestartLanczos<T>::load_deflation() {
  if (d_ == 0) return;
  const DenseMatrix<T>& dm = *opt_.deflation;
  std::copy(dm.data(), dm.data() + n_ * d_, basis_.data());
  // Gram matrix G = D^H M D, formed with the stored M*D or chunk by chunk.
  DenseMatrix<T> gram(d_, d_);
  const auto nn = static_cast<fint>(n_), dd = static_cast<fint>(d_);
  if (!b_inner_) {
    linalg::gram(n_, d_, basis_.data(), n_, basis_.data(), n_, gram.data(), d_);
  } else if (store_bv_) {
    b_inner_->apply(basis_.data(), n_, bbasis_.data(), n_, d_);
    linalg::gram(n_, d_, basis_.data(), n_, bbasis_.data(), n_, gram.data(), d_);
  } else {
    DenseMatrix<T> md(n_, std::min(d_, kGramChunk));
    for (index_t c0 = 0; c0 < d_; c0 += kGramChunk) {
      const index_t nc = std::min(kGramChunk, d_ - c0);
      b_inner_->apply(basis_.col(c0), n_, md.data(), n_, nc);
      lapack::gemm('C', 'N', dd, static_cast<fint>(nc), nn, T(1), basis_.data(), nn, md.data(), nn,
                   T(0), gram.col(c0), dd);
    }
  }
  // NaN-propagating maximum (once NaN, `e > dev` is never true again): std::max(dev, NaN) would
  // keep dev and accept a non-finite block.
  double dev = 0.0;
  for (index_t j = 0; j < d_; ++j)
    for (index_t i = 0; i < d_; ++i) {
      const double e = std::abs(gram(i, j) - T(i == j ? 1.0 : 0.0));
      if (std::isnan(e) || e > dev) dev = e;
    }
  if (!(dev <= kDeflationOrthoTol))
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: deflation block is not orthonormal in the solver inner "
             "product, or not finite (max |D^H M D - I| = "
                 << dev << ")");
}

/// Orthogonalizes basis vector j (stored in place) against [D v_0 .. v_{j-1}] and returns
/// {M-norm before, M-norm after}. The norm before follows from Pythagoras, ||w||^2 =
/// ||coef||^2 + ||w_perp||^2 (the columns are M-orthonormal), so it costs no M application.
/// Throws InvalidInput when the vector is not finite (Inf/NaN from the operator or b_inner).
template <class T>
std::pair<double, double> ThickRestartLanczos<T>::orthogonalize(index_t j) {
  T* w = vcol(j);
  const index_t k = d_ + j;
  double after = 0.0;
  if (store_bv_) {
    // The coefficients (M V)^H w need only the stored columns and w. orthogonalize_cgs2 also
    // updates bw -= (M V) c and derives the norm from it; both are discarded (see "store_bv"
    // above): the column slot of M*w only serves as scratch until M w_perp is stored in it.
    T* bw = bvcol(j);
    linalg::orthogonalize_cgs2(n_, k, basis_.data(), n_, bbasis_.data(), n_, w, bw, coef_.data());
    b_inner_->apply(w, n_, bw, n_, 1);
    // std::max keeps a NaN (first argument) so that the finiteness check below sees it.
    after = std::sqrt(std::max(std::real(linalg::dot(n_, w, bw)), 0.0));
  } else {
    after = linalg::orthogonalize_cgs2<T>(n_, k, basis_.data(), n_, nullptr, n_, w,
                                          mw_.empty() ? nullptr : mw_.data(), coef_.data(),
                                          apply_m_ ? &apply_m_ : nullptr);
  }
  double c2 = after * after;
  for (index_t i = 0; i < k; ++i) c2 += abs2(coef_[i]);
  const double before = std::sqrt(c2);
  if (!std::isfinite(before))
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: non-finite basis vector (the operator or b_inner returned "
             "Inf/NaN)");
  return {before, after};
}

template <class T>
void ThickRestartLanczos<T>::normalize(index_t j, double norm) {
  linalg::scal(n_, T(1.0 / norm), vcol(j));
  if (store_bv_) linalg::scal(n_, T(1.0 / norm), bvcol(j));
}

template <class T>
void ThickRestartLanczos<T>::set_start_vector() {
  T* v0 = vcol(0);
  const bool user = opt_.initial != nullptr;
  if (user) {
    std::copy(opt_.initial->data(), opt_.initial->data() + n_, v0);
    const double norm = linalg::norm2(n_, v0);
    if (!(norm > 0.0) || !std::isfinite(norm))
      HS_THROW(InvalidInput,
               "lanczos_krylov_schur: initial vector must be nonzero and finite, 2-norm " << norm);
  } else {
    linalg::fill_random(v0, n_, opt_.seed, next_stream_++);
  }
  const auto [before, after] = orthogonalize(0);
  if (!(after > kStartMinRatio * before)) {
    // n - d >= nev >= 1 leaves room, so only a user vector can lie in span(D); any other loss
    // of the M-norm (a nonzero vector with w^H M w <= 0) means that M is not positive definite.
    if (user && d_ > 0)
      HS_THROW(InvalidInput,
               "lanczos_krylov_schur: initial vector has no component outside span(deflation) in "
               "the solver inner product (M-norm "
                   << after << " of " << before << "), or b_inner is not positive definite");
    HS_THROW(InvalidInput, "lanczos_krylov_schur: start vector has M-norm "
                               << after << " (of " << before
                               << "): b_inner is not Hermitian positive definite");
  }
  normalize(0, after);
}

template <class T>
void ThickRestartLanczos<T>::set_random_vector(index_t j) {
  linalg::fill_random(vcol(j), n_, opt_.seed, next_stream_++);
  const auto [before, after] = orthogonalize(j);
  if (!(after > kRandomMinRatio * before))
    HS_THROW(InvalidInput, "lanczos_krylov_schur: a random vector lost its M-norm against [D V] ("
                               << after << " of " << before
                               << "): b_inner is not Hermitian positive definite");
  normalize(j, after);
}

/// Lanczos steps j = first .. ncv-1: v_{j+1} = (Op v_j - V H(:, j)) / beta_j with full CGS2
/// reorthogonalization; fills H(j, j) and the tridiagonal couplings (the arrow H(0:first, first)
/// was set by restart()).
template <class T>
void ThickRestartLanczos<T>::expand(index_t first) {
  for (index_t j = first; j < ncv_; ++j) {
    {
      ScopedTimer t(time_.op);
      op_.apply(vcol(j), n_, vcol(j + 1), n_, 1);
    }
    ScopedTimer t(time_.orthogonalization);
    const auto [op_norm, beta] = orthogonalize(j + 1);  // op_norm = ||Op v_j||_M
    // Op is M-self-adjoint, so the diagonal coefficient is real up to rounding.
    h_(j, j) = T(std::real(coef_[d_ + j]));
    const bool last = j + 1 == ncv_;
    double coupling = beta;
    if (beta <= kBreakdownTol * op_norm) {  // local scale, see "Breakdown" above
      coupling = 0.0;
      // A full-space cycle needs no vector beyond v_{n-d-1}; otherwise the continuation (or the
      // next restart) needs a residual vector, so start an independent Krylov sequence.
      if (!(last && full_space_)) set_random_vector(j + 1);
    } else {
      normalize(j + 1, beta);
    }
    if (last) {
      beta_ = coupling;
    } else {
      h_(j + 1, j) = T(coupling);
      h_(j, j + 1) = T(coupling);
    }
  }
}

template <class T>
void ThickRestartLanczos<T>::rayleigh_ritz() {
  ScopedTimer t(time_.rayleigh_ritz);
  const index_t m = ncv_;
  const auto mm = static_cast<fint>(m);
  s_ = h_;  // H is stored exactly Hermitian (both triangles), so it is its own Hermitian part
  theta_.assign(static_cast<std::size_t>(m), 0.0);
  lapack::heev('V', 'L', mm, s_.data(), mm, theta_.data());

  double max_abs = 0.0;
  for (double th : theta_) max_abs = std::max(max_abs, std::abs(th));

  est_.assign(static_cast<std::size_t>(m), 0.0);
  conv_.assign(static_cast<std::size_t>(m), 0);
  ref_.assign(static_cast<std::size_t>(m), 0.0);
  for (index_t i = 0; i < m; ++i) {
    est_[i] = beta_ * std::abs(s_(m - 1, i));
    ref_[i] = theta_[i] != 0.0 ? std::abs(theta_[i]) : max_abs;
    conv_[i] = est_[i] <= opt_.tol * ref_[i] ? 1 : 0;
  }

  order_.resize(static_cast<std::size_t>(m));
  std::iota(order_.begin(), order_.end(), index_t{0});
  switch (which_) {
    case Which::SmallestAlgebraic:
      break;  // heev returns ascending eigenvalues
    case Which::LargestAlgebraic:
      std::reverse(order_.begin(), order_.end());
      break;
    case Which::LargestMagnitude:
      // Ties in |theta| prefer the positive value, so the ranking is a strict total order.
      std::stable_sort(order_.begin(), order_.end(), [this](index_t a, index_t b) {
        const double fa = std::abs(theta_[a]), fb = std::abs(theta_[b]);
        return fa > fb || (fa == fb && theta_[a] > theta_[b]);
      });
      break;
  }
}

/// Appends the history record of the current Rayleigh-Ritz step; returns the number of converged
/// wanted pairs.
template <class T>
index_t ThickRestartLanczos<T>::record_history() {
  RestartRecord rec;
  rec.restart = restarts_;
  for (index_t l = 0; l < nev_; ++l) {
    const index_t i = order_[l];
    if (conv_[i]) {
      ++rec.nconv;
    } else {
      const double rel =
          ref_[i] > 0.0 ? est_[i] / ref_[i] : std::numeric_limits<double>::infinity();
      rec.max_estimate = std::max(rec.max_estimate, rel);
    }
  }
  rec.op_applies = op_.applications() - op_applies0_;
  history_.push_back(rec);
  return rec.nconv;
}

/// V(:, 0:k) <- V(:, 0:m) * S for S of size m x k, in place, one block of rows at a time
/// (each row of the result depends only on the same row of V).
template <class T>
void ThickRestartLanczos<T>::rotate_in_place(T* v, const DenseMatrix<T>& s) {
  const index_t m = s.rows(), k = s.cols();
  const index_t rb = std::min(n_, kRowBlock);
  DenseMatrix<T> tmp(rb, k);
  for (index_t r0 = 0; r0 < n_; r0 += rb) {
    const index_t nr = std::min(rb, n_ - r0);
    lapack::gemm('N', 'N', static_cast<fint>(nr), static_cast<fint>(k), static_cast<fint>(m), T(1),
                 v + r0, static_cast<fint>(n_), s.data(), static_cast<fint>(m), T(0), tmp.data(),
                 static_cast<fint>(rb));
    for (index_t l = 0; l < k; ++l) std::copy(tmp.col(l), tmp.col(l) + nr, v + r0 + l * n_);
  }
}

/// Thick restart keeping the `keep` best Ritz pairs (ranking order) and locking the converged ones
/// whose coupling is at working precision relative to every wanted pair.
template <class T>
void ThickRestartLanczos<T>::restart(index_t keep) {
  ScopedTimer t(time_.restart);
  const index_t m = ncv_;
  double min_wanted_ref = std::numeric_limits<double>::infinity();
  for (index_t l = 0; l < nev_; ++l) min_wanted_ref = std::min(min_wanted_ref, ref_[order_[l]]);
  const double lock_tol = kLockTol * min_wanted_ref;
  DenseMatrix<T> sk(m, keep);
  std::vector<T> b(static_cast<std::size_t>(keep));
  for (index_t l = 0; l < keep; ++l) {
    const index_t i = order_[l];
    std::copy(s_.col(i), s_.col(i) + m, sk.col(l));
    b[l] = T(beta_) * s_(m - 1, i);  // coupling of Ritz vector l to v_m
    if (conv_[i] && std::abs(b[l]) <= lock_tol) b[l] = T(0);
  }

  rotate_in_place(vcol(0), sk);
  std::copy(vcol(m), vcol(m) + n_, vcol(keep));  // the residual vector continues the basis
  if (store_bv_) {
    rotate_in_place(bvcol(0), sk);
    std::copy(bvcol(m), bvcol(m) + n_, bvcol(keep));
  }

  // Arrowhead: diag(theta) with the couplings in row/column `keep`.
  h_.set_zero();
  for (index_t l = 0; l < keep; ++l) {
    h_(l, l) = T(theta_[order_[l]]);
    h_(keep, l) = b[l];
    h_(l, keep) = conj_if(b[l]);
  }
}

template <class T>
HermitianEigResult<T> ThickRestartLanczos<T>::finish() {
  HermitianEigResult<T> r;
  const index_t m = ncv_;
  // The wanted pairs are the nev best ranked; converged ones first, both groups in ranking order.
  std::vector<index_t> sel(order_.begin(), order_.begin() + nev_);
  std::stable_partition(sel.begin(), sel.end(), [this](index_t i) { return conv_[i] != 0; });
  DenseMatrix<T> ssel(m, nev_);
  for (index_t l = 0; l < nev_; ++l) {
    const index_t i = sel[l];
    std::copy(s_.col(i), s_.col(i) + m, ssel.col(l));
    r.theta.push_back(theta_[i]);
    r.estimate.push_back(est_[i]);
    if (conv_[i]) ++r.nconv;
  }
  r.vectors.resize(n_, nev_);
  linalg::multiply(n_, m, nev_, vcol(0), n_, ssel.data(), m, r.vectors.data(), n_);
  r.ritz_all = theta_;
  r.restarts = restarts_;
  r.op_applies = op_.applications() - op_applies0_;
  r.converged = r.nconv >= nev_;
  r.history = std::move(history_);
  return r;
}

template <class T>
HermitianEigResult<T> ThickRestartLanczos<T>::run() {
  load_deflation();
  {
    ScopedTimer t(time_.orthogonalization);
    set_start_vector();
  }
  // Keep k = nev + (ncv - nev) / 2 Ritz pairs per restart; k is in [nev, ncv - 1] because
  // ncv > nev whenever restarts happen (ncv == nev only in full-space mode), so every converged
  // wanted pair is kept and every cycle adds at least one new vector.
  const index_t keep = nev_ + (ncv_ - nev_) / 2;
  index_t first = 0;
  for (;;) {
    expand(first);
    rayleigh_ritz();
    const index_t nconv = record_history();
    if (nconv >= nev_ || full_space_ || restarts_ == opt_.max_restarts) break;
    restart(keep);
    first = keep;
    ++restarts_;
  }
  return finish();
}

}  // namespace

namespace eigen_detail {

template <class T>
HermitianEigResult<T> lanczos_krylov_schur_timed(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                                 Which which, const KrylovSchurOptions<T>& options,
                                                 LanczosTimings& timings) {
  timings = LanczosTimings{};
  Timer total;
  ThickRestartLanczos<T> solver(op, b_inner, which, options, timings);
  HermitianEigResult<T> result = solver.run();
  timings.total = total.seconds();
  return result;
}

}  // namespace eigen_detail

template <class T>
HermitianEigResult<T> lanczos_krylov_schur(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                           Which which, const KrylovSchurOptions<T>& options) {
  eigen_detail::LanczosTimings unused;
  return eigen_detail::lanczos_krylov_schur_timed(op, b_inner, which, options, unused);
}

#define HS_INSTANTIATE_LANCZOS(T)                                                                \
  template HermitianEigResult<T> lanczos_krylov_schur<T>(LinearOperator<T>&, LinearOperator<T>*, \
                                                         Which, const KrylovSchurOptions<T>&);   \
  template HermitianEigResult<T> eigen_detail::lanczos_krylov_schur_timed<T>(                    \
      LinearOperator<T>&, LinearOperator<T>*, Which, const KrylovSchurOptions<T>&,               \
      eigen_detail::LanczosTimings&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_LANCZOS)
#undef HS_INSTANTIATE_LANCZOS

}  // namespace hs
