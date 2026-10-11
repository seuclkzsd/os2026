// Block orthonormalization kernel (BCGS2 with Cholesky-QR, column-wise fallback); see
// block_ortho.hpp.
#include "eigen/block_ortho.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

/// A random replacement keeps most of its M-norm against an M-orthonormal basis that leaves room;
/// losing more than this fraction means M is not positive definite (as in the single-vector code).
constexpr double kRandomMinRatio = 1e-8;
/// Largest accepted ||Q1^H M Q1 - I||_F before the second Cholesky-QR pass. The column-scaled
/// condition test bounds it by ~eps * kMaxCholQrCond^2 for any Hermitian M, so failing it means
/// non-finite or inconsistent arithmetic.
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
  if (store_m_) {
    // Coefficients from the stored M*V; the recurrence-updated M*w is discarded and M is applied
    // to the orthogonalized vector instead (see "store_bv" in lanczos_krylov_schur.cpp).
    T* bw = mbasis + kk * n_;
    linalg::orthogonalize_cgs2(n_, kk, basis, n_, mbasis, n_, w, bw, h);
    m_->apply(w, n_, bw, n_, 1);
    // std::max keeps a NaN (first argument) for the finiteness check below.
    after = std::sqrt(std::max(std::real(linalg::dot(n_, w, bw)), 0.0));
  } else if (m_) {
    const linalg::ApplyFn<T> apply = [this](const T* x, T* y) { m_->apply(x, n_, y, n_, 1); };
    after = linalg::orthogonalize_cgs2<T>(n_, kk, basis, n_, nullptr, n_, w, mw_.data(), h, &apply);
  } else {
    after = linalg::orthogonalize_cgs2<T>(n_, kk, basis, n_, nullptr, n_, w, nullptr, h);
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
  if (!(std::sqrt(dev2) <= kMaxSecondPassDeviation) || lapack::potrf('U', pp, g2.data(), pp) != 0)
    HS_THROW(InvalidInput,
             "lanczos_krylov_schur: block orthonormalization failed (||Q^H M Q - I||_F = "
                 << std::sqrt(dev2)
                 << " after the first Cholesky-QR pass): non-finite values, or "
                    "b_inner is not Hermitian positive definite");
  for (index_t j = 0; j < p; ++j)
    for (index_t i = j + 1; i < p; ++i) g2(i, j) = T(0);
  lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), g2.data(), pp, w, nn);
  if (store_m_) lapack::trsm('R', 'U', 'N', 'N', nn, pp, T(1), g2.data(), pp, slot, nn);
  // R = R2 R1 (upper triangular: the strictly lower products are exact zeros), C = C1 + C2 R1.
  lapack::gemm('N', 'N', pp, pp, pp, T(1), g2.data(), pp, r1.data(), pp, T(0), r.data(), pp);
  if (k > 0)
    lapack::gemm('N', 'N', kk, pp, pp, T(1), cg.data(), kp, r1.data(), pp, T(1), coef.data(), kk);
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
