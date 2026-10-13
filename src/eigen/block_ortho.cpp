// Block orthonormalization kernel (BCGS2 with Cholesky-QR, column-wise fallback); see
// block_ortho.hpp.
#include "eigen/block_ortho.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "eigen/block_lanczos.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

// A random replacement losing more than kRandomMinRatio of its M-norm (block_lanczos.hpp, shared
// with the single-vector solver) means that M is not positive definite.

/// Largest accepted ||G2 - I||_F in the second Cholesky-QR pass (G2 = W2^H M W2, W2 = Q1 with V
/// projected out). The column-scaled condition test bounds ||Q1^H M Q1 - I|| by ~eps *
/// kMaxCholQrCond^2, so a larger deviation means that the second projection removed a large part
/// of a column (it was dependent on a V that is M-orthonormal only to more than rounding), or
/// non-finite values, or an M that is not positive definite: the column-wise path then decides.
constexpr double kMaxSecondPassDeviation = 0.5;

/// a <- (a + a^H) / 2 for a square matrix (real diagonal for complex T).
template <class T>
void hermitize(DenseMatrix<T>& a) {
  const index_t p = a.rows();
  for (index_t j = 0; j < p; ++j) {
    a(j, j) = T(std::real(a(j, j)));
    for (index_t i = 0; i < j; ++i) {
      const T v = T(0.5) * (a(i, j) + conj_if(a(j, i)));
      a(i, j) = v;
      a(j, i) = conj_if(v);
    }
  }
}

/// Column-scaled Frobenius condition estimate ||R D^-1||_F ||D R^-1||_F of an upper triangular
/// Cholesky factor R of G, D = diag(sqrt(G(i, i))); >= the 2-norm condition number of the
/// column-scaled block. NaN for non-finite input.
template <class T>
double scaled_condition(const DenseMatrix<T>& r, const DenseMatrix<T>& g) {
  const index_t p = r.rows();
  DenseMatrix<T> rs(p, p), inv(p, p);
  double fro = 0.0;
  for (index_t j = 0; j < p; ++j) {
    const double s = 1.0 / std::sqrt(std::real(g(j, j)));
    for (index_t i = 0; i <= j; ++i) {
      rs(i, j) = r(i, j) * T(s);
      fro += abs2(rs(i, j));
    }
    inv(j, j) = T(1);
  }
  const auto pp = static_cast<fint>(p);
  lapack::trsm('L', 'U', 'N', 'N', pp, pp, T(1), rs.data(), pp, inv.data(), pp);
  double fro_inv = 0.0;
  for (index_t j = 0; j < p; ++j)
    for (index_t i = 0; i <= j; ++i) fro_inv += abs2(inv(i, j));
  return std::sqrt(fro) * std::sqrt(fro_inv);
}

/// 2-norm condition number of the column-scaled block W1 D^{-1}, D = diag(||W1(:, i)||_M), from
/// its Gram matrix g = W1^H M W1 (Hermitian): sqrt(mu_max / mu_min) with mu the eigenvalues of
/// D^{-1} g D^{-1} (unit diagonal). The test at kMaxFastCond needs mu_min / mu_max >= 2.4e-4, far
/// above their rounding (~p eps), so the estimate is accurate where it decides (the Frobenius
/// estimate scaled_condition above is >= p for every p x p factor, too coarse for wide blocks).
/// +infinity when that matrix is not numerically positive definite or not finite. O(p^3).
template <class T>
double scaled_condition_2norm(const DenseMatrix<T>& g) {
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

template <class T>
BlockOrthonormalizer<T>::BlockOrthonormalizer(index_t n, index_t p, LinearOperator<T>* m,
                                              bool store_m)
    : n_(n), p_(p), m_(m), store_m_(store_m && m != nullptr) {
  HS_ASSERT(n >= 0 && p >= 1, "BlockOrthonormalizer: n=" << n << ", p=" << p);
  if (m_ && !store_m_) mw_.resize(n_, p_);
}

template <class T>
const T* BlockOrthonormalizer<T>::apply_m(const T* x, T* y) {
  if (!m_) return x;
  m_->apply(x, n_, y, n_, p_);
  return y;
}

template <class T>
std::pair<double, double> BlockOrthonormalizer<T>::orthogonalize_column(T* basis, T* mbasis,
                                                                        index_t kk, T* h) {
  T* w = basis + kk * n_;
  double after = 0.0;
  // Always two passes: this fallback must also clean columns against a V that is M-orthonormal
  // only to delta (a deflation block accepted at kDeflationOrthoTol), where one DGKS pass would
  // leave V^H M w = O(delta) (see linalg::Reorth). The path is rare, so the extra pass is free.
  const linalg::OrthoOptions two_pass{linalg::Reorth::Always};
  if (store_m_) {
    // Coefficients from the stored M*V; the recurrence-updated M*w is discarded and M is applied
    // to the orthogonalized vector instead (see "store_bv" in lanczos_krylov_schur.cpp).
    T* bw = mbasis + kk * n_;
    linalg::orthogonalize_cgs2<T>(n_, kk, basis, n_, mbasis, n_, w, bw, h, nullptr, two_pass);
    m_->apply(w, n_, bw, n_, 1);
    // std::max keeps a NaN (first argument) for the finiteness check below.
    after = std::sqrt(std::max(std::real(linalg::dot(n_, w, bw)), 0.0));
  } else if (m_) {
    const linalg::ApplyFn<T> apply = [this](const T* x, T* y) { m_->apply(x, n_, y, n_, 1); };
    after = linalg::orthogonalize_cgs2<T>(n_, kk, basis, n_, nullptr, n_, w, mw_.data(), h, &apply,
                                          two_pass);
  } else {
    after = linalg::orthogonalize_cgs2<T>(n_, kk, basis, n_, nullptr, n_, w, nullptr, h, nullptr,
                                          two_pass);
  }
  double c2 = after * after;
  for (index_t i = 0; i < kk; ++i) c2 += abs2(h[i]);
  const double before = std::sqrt(c2);
  if (!std::isfinite(before))
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: non-finite basis vector (the operator or b_inner returned "
             "Inf/NaN)");
  return {before, after};
}

template <class T>
void BlockOrthonormalizer<T>::scale_column(T* basis, T* mbasis, index_t kk, double norm) {
  linalg::scal(n_, T(1.0 / norm), basis + kk * n_);
  if (store_m_) linalg::scal(n_, T(1.0 / norm), mbasis + kk * n_);
}

template <class T>
void BlockOrthonormalizer<T>::run(T* basis, T* mbasis, index_t k, double ratio, std::uint64_t seed,
                                  std::uint64_t& next_stream, DenseMatrix<T>& coef,
                                  DenseMatrix<T>& r) {
  HS_ASSERT(store_m_ == (mbasis != nullptr),
            "BlockOrthonormalizer: mbasis must be given exactly in store mode");
  const index_t n = n_, p = p_;
  const auto nn = static_cast<fint>(n), kk = static_cast<fint>(k), pp = static_cast<fint>(p);
  T* w = basis + k * n;
  // M times the block: the stored slot, the workspace, or unused (Euclidean).
  T* slot = store_m_ ? mbasis + k * n : (m_ ? mw_.data() : nullptr);
  coef.resize(k, p);
  r.resize(p, p);
  fallback_ = false;
  deficient_.assign(static_cast<std::size_t>(p), 0);

  // Pass 1 against V: C1 = (M V)^H W (stored) or V^H (M W).
  if (k > 0) {
    const T* rhs = w;
    if (m_ && !store_m_) {
      m_->apply(w, n, slot, n, p);
      rhs = slot;
    }
    lapack::gemm('C', 'N', kk, pp, nn, T(1), store_m_ ? mbasis : basis, nn, rhs, nn, T(0),
                 coef.data(), kk);
    lapack::gemm('N', 'N', nn, pp, kk, T(-1), basis, nn, coef.data(), kk, T(1), w, nn);
  }
  // G1 = W1^H M W1 and the M-norms of the input columns (Pythagoras: V is M-orthonormal).
  DenseMatrix<T> g1(p, p);
  lapack::gemm('C', 'N', pp, pp, nn, T(1), w, nn, apply_m(w, slot), nn, T(0), g1.data(), pp);
  hermitize(g1);
  before_.assign(static_cast<std::size_t>(p), 0.0);
  bool fast = true;
  for (index_t i = 0; i < p; ++i) {
    const double gii = std::real(g1(i, i));
    double c2 = 0.0;
    for (index_t row = 0; row < k; ++row) c2 += abs2(coef(row, i));
    if (!std::isfinite(gii) || !std::isfinite(c2))
      HS_THROW(InvalidInput,
               "lanczos_krylov_schur: non-finite basis vector (the operator or b_inner returned "
               "Inf/NaN)");
    before_[i] = std::sqrt(c2 + std::max(gii, 0.0));
    const double floor = ratio * before_[i];
    if (!(gii > floor * floor)) fast = false;  // (nearly) dependent column, or M not definite
  }
  DenseMatrix<T> r1;
  if (fast) {
    r1 = g1;
    fast = lapack::potrf('U', pp, r1.data(), pp) == 0;
  }
  if (fast) {
    for (index_t j = 0; j < p; ++j)
      for (index_t i = j + 1; i < p; ++i) r1(i, j) = T(0);
    fast = scaled_condition(r1, g1) <= kMaxCholQrCond;
  }
  if (!fast) {
    column_fallback(basis, mbasis, k, ratio, seed, next_stream, coef, r);
    return;
  }

  // Q1 = W1 R1^{-1}; pass 2: [C2; G] = [V Q1]^H M Q1 in one product.
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), r1.data(), pp, w, nn);
  DenseMatrix<T> cg(k + p, p);
  const auto kp = static_cast<fint>(k + p);
  lapack::gemm('C', 'N', kp, pp, nn, T(1), basis, nn, apply_m(w, slot), nn, T(0), cg.data(), kp);
  DenseMatrix<T> g2(p, p);
  for (index_t j = 0; j < p; ++j)
    for (index_t i = 0; i < p; ++i) g2(i, j) = cg(k + i, j);
  if (k > 0) {
    lapack::gemm('N', 'N', nn, pp, kk, T(-1), basis, nn, cg.data(), kp, T(1), w, nn);
    lapack::gemm('C', 'N', pp, pp, kk, T(-1), cg.data(), kp, cg.data(), kp, T(1), g2.data(), pp);
    if (store_m_)
      lapack::gemm('N', 'N', nn, pp, kk, T(-1), mbasis, nn, cg.data(), kp, T(1), slot, nn);
  }
  hermitize(g2);
  double dev2 = 0.0;
  for (index_t j = 0; j < p; ++j)
    for (index_t i = 0; i < p; ++i) dev2 += abs2(g2(i, j) - T(i == j ? 1.0 : 0.0));
  bool second = std::sqrt(dev2) <= kMaxSecondPassDeviation &&
                lapack::potrf('U', pp, g2.data(), pp) == 0;
  if (second) {
    for (index_t j = 0; j < p; ++j)
      for (index_t i = j + 1; i < p; ++i) g2(i, j) = T(0);
    // R = R2 R1 (upper triangular: the strictly lower products are exact zeros).
    lapack::gemm('N', 'N', pp, pp, pp, T(1), g2.data(), pp, r1.data(), pp, T(0), r.data(), pp);
    // Deficiency after both passes, as the single-vector solver tests beta after CGS2: R(i, i) is
    // the M-norm of the part of column i orthogonal to V and the block's earlier columns.
    for (index_t i = 0; i < p; ++i)
      if (!(std::abs(r(i, i)) > ratio * before_[i])) second = false;
  }
  if (k > 0)  // C = C1 + C2 R1 (both outcomes: W = V C + W2 R1 holds exactly in exact arithmetic)
    lapack::gemm('N', 'N', kk, pp, pp, T(1), cg.data(), kp, r1.data(), pp, T(1), coef.data(), kk);
  if (!second) {
    // A column dependent on V to working precision whose first-pass remainder stayed above the
    // threshold (V M-orthonormal only to more than ~ratio, e.g. a deflation block accepted at
    // 1e-8): the second pass removes it and G2 is singular or R(i, i) tiny. Rebuild the block as
    // W2 R1 = W - V C and let the column-wise CGS2 path decide which columns are dependent.
    multiply_upper_right(w, n, r1);
    column_fallback(basis, mbasis, k, ratio, seed, next_stream, coef, r);
    return;
  }
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), g2.data(), pp, w, nn);
  if (store_m_) lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), g2.data(), pp, slot, nn);
}

template <class T>
bool BlockOrthonormalizer<T>::try_fast(T* basis, T* mbasis, index_t k, double ratio,
                                       DenseMatrix<T>& coef, DenseMatrix<T>& r) {
  HS_ASSERT(store_m_ == (mbasis != nullptr),
            "BlockOrthonormalizer: mbasis must be given exactly in store mode");
  if (store_m_) {  // stored M*Q must come from a fresh M application: run()
    ++fast_refusals_;
    return false;
  }
  const index_t n = n_, p = p_;
  const auto nn = static_cast<fint>(n), kk = static_cast<fint>(k), pp = static_cast<fint>(p);
  T* w = basis + k * n;
  T* mw = m_ ? mw_.data() : nullptr;  // M W, updated by orthogonalize_block_cgs2
  coef.resize(k, p);
  r.resize(p, p);
  fallback_ = false;
  deficient_.assign(static_cast<std::size_t>(p), 0);
  before_.assign(static_cast<std::size_t>(p), 0.0);
  std::vector<double> after(static_cast<std::size_t>(p));
  const linalg::ApplyBlockFn<T> apply = [this](const T* x, index_t ldx, T* y, index_t ldy,
                                               index_t q) { m_->apply(x, ldx, y, ldy, q); };
  // Reorth::Auto: the DGKS test (V is M-orthonormal to working precision, see block_ortho.hpp).
  linalg::orthogonalize_block_cgs2<T>(n, k, p, basis, n, nullptr, 0, w, n, mw, n, coef.data(), k,
                                      before_.data(), after.data(), m_ ? &apply : nullptr);
  auto refuse = [&]() {
    if (k > 0)  // W = W1 + V C: the input block (up to rounding) for run()
      lapack::gemm('N', 'N', nn, pp, kk, T(1), basis, nn, coef.data(), kk, T(1), w, nn);
    ++fast_refusals_;
    return false;
  };
  // A column that lost more than its share (or non-finite input: NaN fails every test here, and
  // run() then reports it).
  for (index_t c = 0; c < p; ++c)
    if (!(after[c] > ratio * before_[c]) || !std::isfinite(before_[c])) return refuse();
  DenseMatrix<T> g1(p, p);
  linalg::gram(n, p, w, n, mw ? mw : w, n, g1.data(), p);
  hermitize(g1);
  if (!(scaled_condition_2norm(g1) <= kMaxFastCond)) return refuse();
  DenseMatrix<T> r1 = g1;
  if (lapack::potrf('U', pp, r1.data(), pp) != 0) return refuse();
  for (index_t j = 0; j < p; ++j)
    for (index_t i = j + 1; i < p; ++i) r1(i, j) = T(0);
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), r1.data(), pp, w, nn);  // Q1 = W1 R1^{-1}
  if (mw) lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), r1.data(), pp, mw, nn);  // M Q1
  DenseMatrix<T> g2(p, p);
  linalg::gram(n, p, w, n, mw ? mw : w, n, g2.data(), p);
  hermitize(g2);
  double dev2 = 0.0;
  for (index_t j = 0; j < p; ++j)
    for (index_t i = 0; i < p; ++i) dev2 += abs2(g2(i, j) - T(i == j ? 1.0 : 0.0));
  if (!(std::sqrt(dev2) <= kMaxSecondPassDeviation) || lapack::potrf('U', pp, g2.data(), pp) != 0) {
    multiply_upper_right(w, n, r1);  // back to W1
    return refuse();
  }
  for (index_t j = 0; j < p; ++j)
    for (index_t i = j + 1; i < p; ++i) g2(i, j) = T(0);
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), g2.data(), pp, w, nn);  // Q = Q1 R2^{-1}
  // R = R2 R1 (upper triangular: the strictly lower products are exact zeros); R(i, i) is the
  // M-norm of the part of column i orthogonal to V and the block's earlier columns.
  lapack::gemm('N', 'N', pp, pp, pp, T(1), g2.data(), pp, r1.data(), pp, T(0), r.data(), pp);
  for (index_t c = 0; c < p; ++c) {
    if (!(std::abs(r(c, c)) > ratio * before_[c])) {
      multiply_upper_right(w, n, r);  // Q R = W1
      return refuse();
    }
  }
  ++fast_runs_;
  return true;
}

template <class T>
void BlockOrthonormalizer<T>::column_fallback(T* basis, T* mbasis, index_t k, double ratio,
                                              std::uint64_t seed, std::uint64_t& next_stream,
                                              DenseMatrix<T>& coef, DenseMatrix<T>& r) {
  fallback_ = true;
  ++fallbacks_;
  const index_t p = p_;
  std::vector<T> h(static_cast<std::size_t>(k + p));
  for (index_t i = 0; i < p; ++i) {
    const index_t kk = k + i;
    // W1 is already orthogonal to V up to the first pass; CGS2 against [V Q_0 .. Q_{i-1}].
    const double after = orthogonalize_column(basis, mbasis, kk, h.data()).second;
    for (index_t row = 0; row < k; ++row) coef(row, i) += h[row];
    for (index_t l = 0; l < i; ++l) r(l, i) = h[k + l];
    if (after > ratio * before_[i]) {
      r(i, i) = T(after);
      scale_column(basis, mbasis, kk, after);
      continue;
    }
    // Dependent column: drop the rounding-level remainder and continue the block with a fresh
    // direction (R(i, i) = 0 decouples it from W).
    deficient_[i] = 1;
    r(i, i) = T(0);
    linalg::fill_random(basis + kk * n_, n_, seed, next_stream++);
    const auto [rb, ra] = orthogonalize_column(basis, mbasis, kk, h.data());
    if (!(ra > kRandomMinRatio * rb))
      HS_THROW(InvalidInput, "lanczos_krylov_schur: a random vector lost its M-norm against [D V] ("
                                 << ra << " of " << rb
                                 << "): b_inner is not Hermitian positive definite");
    scale_column(basis, mbasis, kk, ra);
  }
}

#define HS_INSTANTIATE_BLOCK_ORTHO(T) template class BlockOrthonormalizer<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_BLOCK_ORTHO)
#undef HS_INSTANTIATE_BLOCK_ORTHO

}  // namespace hs::eigen_detail
