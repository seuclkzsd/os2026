// Block Krylov-Schur basis of hs::krylov_schur_general (block mode); see block_general.hpp.
//
// Block steps. W = Op U for the residual block U (one op.apply of p columns) is orthonormalized
// against the whole basis [V U] (Euclidean inner product), W = [V U] C + U' R with R upper
// triangular. For a general operator every coefficient of C is a genuine entry of the block upper
// Hessenberg matrix H (unlike block Lanczos, nothing vanishes by symmetry), so C and R are stored
// as they are.
//   * Fast path: linalg::orthogonalize_block_cgs2 (block classical Gram-Schmidt with the DGKS
//     test - a second pass only after cancellation - in the fused, thread-count independent row
//     block kernel) followed by Cholesky-QR twice inside the block (CholQR2). Accepted when no
//     column lost more than its breakdown share (kBreakdownRatio), the projected block W1 has a
//     column-scaled 2-norm condition number <= kMaxFastCond and the second Gram matrix is close
//     to I. The projection leaves each column orthogonal to V up to O(eps / eta) of its norm
//     (DGKS), and R1^{-1} amplifies that by at most the condition number (whatever the block
//     width: the test uses the 2-norm, not a Frobenius estimate, which is >= p for every p x p
//     factor); no second projection against V follows. Measured with bench/ghiep_fix_ortho.cpp
//     (see there).
//   * Otherwise the block is restored (W = W1 + V C1) and handed to BlockOrthonormalizer (two
//     passes of block CGS, each followed by Cholesky-QR, column-wise fallback; block_ortho.hpp).
//     A column of W that is dependent on [V U] and the block's earlier columns to working
//     precision (the block analogue of an Arnoldi breakdown: span(V) holds an invariant subspace)
//     gets R(i, i) = 0 there and a deterministic random replacement orthogonal to everything
//     before; the relation stays exact up to rounding and the search covers the rest of the
//     space, as the single-vector solver does after a breakdown.
//
// Thick restart. The kept Schur vectors V Q(:, 0:k) are formed in place (row blocks, no second
// n x k buffer) and the residual block moves to V(:, k:k+p); H restarts as the quasi-triangular
// S(0:k, 0:k) with the coupling rows C Q(:, 0:k) below it, followed by block Arnoldi columns.
#include "eigen/block_general.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "eigen/lobpcg_jd_common.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

/// A start-block column keeping less than this fraction of its norm against the earlier columns
/// is dependent and replaced (as the block Hermitian solver's kStartMinRatio).
constexpr double kStartRatio = 1e-12;
/// Rows per block of the in-place basis compression V(:, 0:k) = V(:, 0:m) Q(:, 0:k).
constexpr index_t kRowBlock = 2048;
/// Fast path (file comment): largest 2-norm condition number of the column-scaled projected block
/// W1 D^{-1} (= that of R1 D^{-1}, R1 its Cholesky factor), i.e. the largest amplification of its
/// rounding-level components in span(V) by R1^{-1}.
constexpr double kMaxFastCond = 64.0;
/// Fast path: largest ||G2 - I||_F of the second Cholesky-QR (BlockOrthonormalizer's bound).
constexpr double kMaxFastDeviation = 0.5;

/// 2-norm condition number of the column-scaled block W1 D^{-1}, D = diag(||W1(:, i)||), from its
/// Gram matrix g = W1^H W1 (Hermitian): sqrt(mu_max / mu_min) with mu the eigenvalues of
/// D^{-1} g D^{-1} (unit diagonal). A test at kappa <= kMaxFastCond needs mu_min / mu_max >=
/// kMaxFastCond^-2 = 2.4e-4, far above the absolute rounding of the eigenvalues (~p eps mu_max), so
/// the estimate is accurate wherever it decides. +infinity when that matrix is not numerically
/// positive definite or not finite. O(p^3).
template <class T>
double scaled_condition(const DenseMatrix<T>& g) {
  const index_t p = g.rows();
  std::vector<double> inv_d(static_cast<std::size_t>(p));
  for (index_t j = 0; j < p; ++j) inv_d[j] = 1.0 / std::sqrt(std::real(g(j, j)));
  DenseMatrix<T> gs(p, p);
  for (index_t j = 0; j < p; ++j) {
    for (index_t i = 0; i < p; ++i) {
      gs(i, j) = g(i, j) * T(inv_d[i] * inv_d[j]);
      if (!std::isfinite(std::abs(gs(i, j)))) return std::numeric_limits<double>::infinity();
    }
  }
  std::vector<double> mu(static_cast<std::size_t>(p));
  const auto pp = static_cast<fint>(p);
  lapack::heev('N', 'U', pp, gs.data(), pp, mu.data());  // ascending
  if (!(mu.front() > 0.0)) return std::numeric_limits<double>::infinity();
  return std::sqrt(mu.back() / mu.front());
}

/// In place: g <- its upper Cholesky factor (Hermitian part of g, strictly lower part zeroed).
/// False when g is not numerically positive definite.
template <class T>
bool cholesky(DenseMatrix<T>& g) {
  const index_t p = g.rows();
  hermitize(g);
  if (lapack::potrf('U', static_cast<fint>(p), g.data(), static_cast<fint>(p)) != 0) return false;
  for (index_t j = 0; j < p; ++j)
    for (index_t i = j + 1; i < p; ++i) g(i, j) = T(0);
  return true;
}

/// X <- X U in place for X (n x p, ld n) and an upper triangular U (p x p): column j becomes
/// sum_{i <= j} U(i, j) X(:, i), so the columns are rewritten last to first.
template <class T>
void multiply_upper_right(T* x, index_t n, const DenseMatrix<T>& u) {
  for (index_t j = u.cols() - 1; j >= 0; --j) {
    linalg::scal(n, u(j, j), x + j * n);
    for (index_t i = 0; i < j; ++i) linalg::axpy(n, u(i, j), x + i * n, x + j * n);
  }
}

}  // namespace

BlockGeneralLayout block_general_layout(index_t n, index_t nev, index_t ncv1, bool auto_ncv,
                                        index_t p, bool complex_scalars) {
  HS_ASSERT(p >= 2 && nev >= 1 && ncv1 >= nev && ncv1 <= n,
            "block layout: n=" << n << " nev=" << nev << " ncv=" << ncv1 << " p=" << p);
  BlockGeneralLayout lay;
  // Every layout has ncv >= spare + 2p >= 3p, so 2 (ncv + p) <= n needs p <= n / 8; larger p
  // (up to INT64_MAX) is infeasible before the rounding below could overflow.
  if (p > n / 8) return lay;
  const auto round_up = [p](index_t x) { return (x + p - 1) / p * p; };
  const index_t spare = round_up(nev + (complex_scalars ? 1 : 2));
  const index_t base = std::max(round_up(ncv1), spare + 2 * p);
  lay.ncv = base;
  if (auto_ncv) {
    const index_t room = (n / 2 - p) / p * p;  // largest multiple of p with 2 (ncv + p) <= n
    lay.ncv = std::max(base, std::min(base + kAutoExtraBlocks * p, room));
  }
  lay.keep = std::clamp((nev + (lay.ncv - nev) / 2) / p * p, spare, lay.ncv - p);
  lay.first_probe = base;
  lay.feasible = ncv1 < n && lay.ncv + p <= n / 2;
  return lay;
}

template <class T>
BlockKrylovSchurBasis<T>::BlockKrylovSchurBasis(index_t n, index_t capacity, index_t p,
                                                std::uint64_t seed)
    : n_(n),
      capacity_(capacity),
      p_(p),
      seed_(seed),
      v_(n, capacity + p),
      h_(capacity + p, capacity),
      ortho_(n, p, nullptr, false) {
  HS_ASSERT(p >= 1 && capacity >= p && capacity + p <= n,
            "block basis n=" << n << " capacity=" << capacity << " p=" << p);
}

template <class T>
void BlockKrylovSchurBasis<T>::start(const DenseMatrix<T>* initial) {
  const index_t user = initial ? std::min(p_, initial->cols()) : 0;
  for (index_t c = 0; c < p_; ++c) {
    T* v = v_.col(c);
    if (c < user) {
      std::copy(initial->col(c), initial->col(c) + n_, v);
      // Largest entry to [1, 2) (exact): the Gram matrix of the block is a plain sum of squares.
      const double amax = max_abs_component(n_, v);
      if (amax > 0.0) linalg::scal(n_, T(std::ldexp(1.0, -binary_exponent(amax))), v);
    } else {
      linalg::fill_random(v, n_, seed_, static_cast<std::uint64_t>(c));
    }
  }
  next_stream_ = static_cast<std::uint64_t>(p_);
  ortho_.run(v_.data(), nullptr, 0, kStartRatio, seed_, next_stream_, coef_, rblk_);
  // Column 0 is nonzero (validated) and comes first in the QR: it cannot be dependent.
  if (user > 0 && ortho_.deficient()[0])
    HS_THROW(InvalidInput, "krylov_schur_general: initial vector is numerically zero");
  replacements_ += static_cast<index_t>(
      std::count(ortho_.deficient().begin(), ortho_.deficient().end(), char{1}));
  m_ = 0;
}

template <class T>
void BlockKrylovSchurBasis<T>::scale_output(index_t j) {
  T* w = v_.col(j);
  const index_t len = n_ * p_;  // the p output columns are contiguous (ld n)
  if (!scale_fixed_) {
    const double amax = max_abs_component(len, w);
    // An exactly zero block (breakdown, handled by the orthonormalizer) or a non-finite one (the
    // check below throws) fixes nothing.
    if (amax > 0.0 && std::isfinite(amax)) {
      scale_exponent_ = binary_exponent(amax);
      scale_fixed_ = true;
    }
  }
  if (scale_exponent_ != 0) linalg::scal(len, T(std::ldexp(1.0, -scale_exponent_)), w);
  for (index_t c = 0; c < p_; ++c) {
    if (!std::isfinite(linalg::norm2(n_, w + c * n_))) {
      // Not a breakdown: NaN/Inf would end up in H, where gees either fails or deflates them into
      // "converged" garbage.
      HS_THROW(InvalidInput, "krylov_schur_general: operator output column "
                                 << c << " of the block step at basis size " << m_
                                 << " is not finite (NaN/Inf entries, or a norm beyond the double "
                                 << "range relative to the first output)");
    }
  }
}

template <class T>
void BlockKrylovSchurBasis<T>::add_block(LinearOperator<T>& op, GeneralPhaseTimes* times) {
  HS_ASSERT(m_ + p_ <= capacity_, "block step beyond the capacity: m=" << m_ << " p=" << p_
                                                                       << " capacity=" << capacity_);
  const index_t j = m_ + p_;  // first column of the new block
  {
    PhaseClock clock(phase(times, &GeneralPhaseTimes::apply_s));
    op.apply(v_.col(m_), n_, v_.col(j), n_, p_);
  }
  PhaseClock clock(phase(times, &GeneralPhaseTimes::orthogonalize_s));
  scale_output(j);
  if (fast_orthonormalize(j)) {
    ++fast_steps_;
  } else {
    // Back to W = W1 + V C1 and the two-pass kernel with its column-wise fallback.
    const auto nn = static_cast<fint>(n_), jj = static_cast<fint>(j), pp = static_cast<fint>(p_);
    lapack::gemm('N', 'N', nn, pp, jj, T(1), v_.data(), nn, coef_.data(), jj, T(1), v_.col(j), nn);
    ortho_.run(v_.data(), nullptr, j, kBreakdownRatio, seed_, next_stream_, coef_, rblk_);
    replacements_ += static_cast<index_t>(
        std::count(ortho_.deficient().begin(), ortho_.deficient().end(), char{1}));
    ++fallback_steps_;
  }
  for (index_t c = 0; c < p_; ++c) {
    T* hc = h_.col(m_ + c);
    std::copy(coef_.col(c), coef_.col(c) + j, hc);
    for (index_t r = 0; r < p_; ++r) hc[j + r] = rblk_(r, c);
  }
  m_ = j;
}

template <class T>
bool BlockKrylovSchurBasis<T>::fast_orthonormalize(index_t j) {
  const index_t n = n_, p = p_;
  T* w = v_.col(j);
  coef_.resize(j, p);
  rblk_.resize(p, p);
  std::vector<double> before(static_cast<std::size_t>(p)), after(static_cast<std::size_t>(p));
  linalg::orthogonalize_block_cgs2<T>(n, j, p, v_.data(), n, nullptr, 0, w, n, nullptr, 0,
                                      coef_.data(), j, before.data(), after.data());
  for (index_t c = 0; c < p; ++c)
    if (!(after[c] > kBreakdownRatio * before[c])) return false;  // breakdown (or NaN)
  const auto nn = static_cast<fint>(n), pp = static_cast<fint>(p);
  DenseMatrix<T> g1(p, p);
  linalg::gram(n, p, w, n, w, n, g1.data(), p);
  hermitize(g1);
  DenseMatrix<T> r1 = g1;
  if (!cholesky(r1)) return false;
  if (!(scaled_condition(g1) <= kMaxFastCond)) return false;
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), r1.data(), pp, w, nn);  // Q1 = W1 R1^{-1}
  DenseMatrix<T> r2(p, p);
  linalg::gram(n, p, w, n, w, n, r2.data(), p);
  double dev2 = 0.0;
  for (index_t c = 0; c < p; ++c)
    for (index_t r = 0; r < p; ++r) dev2 += abs2(r2(r, c) - T(r == c ? 1.0 : 0.0));
  if (!(std::sqrt(dev2) <= kMaxFastDeviation) || !cholesky(r2)) {
    multiply_upper_right(w, n, r1);  // back to W1 = Q1 R1 for the caller's fallback
    return false;
  }
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), r2.data(), pp, w, nn);
  // R = R2 R1 (upper triangular: the strictly lower products are exact zeros), and the part of
  // each column orthogonal to V and the block's earlier columns, R(i, i), must stay above the
  // breakdown share as in the two-pass kernel.
  lapack::gemm('N', 'N', pp, pp, pp, T(1), r2.data(), pp, r1.data(), pp, T(0), rblk_.data(), pp);
  for (index_t c = 0; c < p; ++c) {
    if (!(std::abs(rblk_(c, c)) > kBreakdownRatio * before[c])) {
      multiply_upper_right(w, n, rblk_);
      return false;
    }
  }
  return true;
}

template <class T>
DenseMatrix<T> BlockKrylovSchurBasis<T>::coupling() const {
  DenseMatrix<T> c(p_, m_);
  for (index_t j = 0; j < m_; ++j)
    for (index_t r = 0; r < p_; ++r) c(r, j) = h_(m_ + r, j);
  return c;
}

template <class T>
void BlockKrylovSchurBasis<T>::compress(const ProjectedSchur<T>& ps, const DenseMatrix<T>& c,
                                        index_t k, GeneralPhaseTimes* times) {
  const index_t m = m_;
  HS_ASSERT(ps.order() == m && c.rows() == p_ && c.cols() == m && k >= 1 && k + p_ <= m &&
                ps.is_boundary(k),
            "block restart keeps k=" << k << " of m=" << m << " (p=" << p_ << ")");
  const DenseMatrix<T>& q = ps.vectors();
  const DenseMatrix<T>& s = ps.form();
  {
    PhaseClock clock(phase(times, &GeneralPhaseTimes::update_s));
    // In place by row blocks: each block of V's rows is copied out once and overwritten by the
    // product, so no second n x k buffer is needed.
    aligned_vector<T> rows(static_cast<std::size_t>(std::min(kRowBlock, n_) * m));
    for (index_t r0 = 0; r0 < n_; r0 += kRowBlock) {
      const index_t rb = std::min(kRowBlock, n_ - r0);
      for (index_t j = 0; j < m; ++j) std::copy(v_.col(j) + r0, v_.col(j) + r0 + rb, rows.data() + j * rb);
      linalg::multiply(rb, m, k, rows.data(), rb, q.data(), m, v_.data() + r0, n_);
    }
    // The residual block continues the basis (k + p <= m: the destination starts first).
    std::copy(v_.col(m), v_.col(m) + n_ * p_, v_.col(k));
  }
  h_.set_zero();
  for (index_t j = 0; j < k; ++j) std::copy(s.col(j), s.col(j) + k, h_.col(j));
  const auto pp = static_cast<fint>(p_), kk = static_cast<fint>(k), mm = static_cast<fint>(m);
  lapack::gemm('N', 'N', pp, kk, mm, T(1), c.data(), pp, q.data(), mm, T(0), h_.data() + k,
               static_cast<fint>(h_.rows()));
  m_ = k;
}

template <class T>
void BlockKrylovSchurBasis<T>::combine(const T* c, index_t ldc, index_t cols, T* y,
                                       index_t ldy) const {
  linalg::multiply(n_, m_, cols, v_.data(), n_, c, ldc, y, ldy);
}

template <class T>
void BlockKrylovSchurBasis<T>::release() {
  v_ = DenseMatrix<T>();
  h_ = DenseMatrix<T>();
}

#define HS_INSTANTIATE_BLOCK_GENERAL(T) template class BlockKrylovSchurBasis<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_BLOCK_GENERAL)
#undef HS_INSTANTIATE_BLOCK_GENERAL

}  // namespace hs::eigen_detail
