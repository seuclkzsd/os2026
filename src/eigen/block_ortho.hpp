// Block orthonormalization kernel of the block Krylov-Schur solver (module-internal, src/eigen).
// Scalar set: T in {double, cplx} (instantiated in src/eigen/block_ortho.cpp).
//
// Owner note: dense_ops (src/linalg) owns the single-vector CGS2; this is its block (BLAS3)
// counterpart, built on top of it for the rank-deficient fallback.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"

namespace hs::eigen_detail {

/// Orthonormalizes a block of p new columns against an M-orthonormal basis and within itself, in
/// <x, y>_M = y^H M x with M = `m` (Hermitian positive definite) or the identity (m == nullptr):
///   W = V C + Q R,  V = basis(:, 0:k) (M-orthonormal on entry), W = basis(:, k:k+p) on entry,
///   Q = basis(:, k:k+p) on exit with [V Q] M-orthonormal, R (p x p) upper triangular.
///
/// Fast path: two passes of block classical Gram-Schmidt against V, each followed by Cholesky-QR
/// of the block (BCGS2 with CholQR intra-orthogonalization; all products BLAS3):
///   C1 = V^H M W, W1 = W - V C1;        G1 = W1^H M W1 = R1^H R1,         Q1 = W1 R1^{-1};
///   [C2; G] = [V Q1]^H M Q1, W2 = Q1 - V C2;  G2 = G - C2^H C2 = R2^H R2,  Q = W2 R2^{-1};
///   R = R2 R1, C = C1 + C2 R1.
/// The second pass removes both the loss of orthogonality to V that R1^{-1} amplifies (by up to
/// cond(W1)) and the O(eps cond(W1)^2) loss inside the block. G2 subtracts C2^H C2 (Pythagoras)
/// instead of forming M W2: C2 is O(eps cond(W1)), so nothing cancels.
/// Fallback, column by column (BLAS2, linalg::orthogonalize_cgs2): taken when the block is
/// (nearly) rank deficient, i.e. some column of W1 keeps an M-norm <= ratio * (M-norm of the input
/// column) (dependent on V to working precision: the block analogue of a Lanczos breakdown), G1 is
/// not numerically positive definite, or the column-scaled condition number of R1 exceeds
/// kMaxCholQrCond (Cholesky-QR could no longer restore orthogonality in the second pass); and
/// after the second pass when G2 is far from I or not positive definite, or some |R(i, i)| <=
/// ratio * (M-norm of input column i). The test after both passes is the one of the single-vector
/// solver (beta after CGS2): a column dependent on a V that is M-orthonormal only to delta >>
/// ratio (e.g. a deflation block accepted at 1e-8) keeps a first-pass remainder of size ~delta
/// that the second pass removes; the block W2 R1 = W - V C is then handed to this path. Each
/// column of W1 is orthogonalized by CGS2 against V and the block's earlier columns; a column left
/// with an M-norm <= ratio * (its input M-norm) gets R(i, i) = 0 and is replaced by a deterministic
/// random vector (linalg::fill_random(seed, stream), stream = next_stream++) orthogonalized the
/// same way. The dropped part has an M-norm below ratio * ||w_i||_M, as for a single-vector
/// breakdown.
///
/// Modes (as in linalg::orthogonalize_cgs2):
///   * Euclidean (m == nullptr): no M application.
///   * Stored M*basis (store_m): run() gets mbasis with M*V in mbasis(:, 0:k) and stores M*Q in
///     mbasis(:, k:k+p) (the slot is scratch before). M*Q comes from fresh M applications updated
///     only by O(eps) terms (M Q = (M Q1 - (M V) C2) R2^{-1}, R2 = I + O(eps cond^2)), never by
///     a recurrence that cancels. Cost: 2 block applications of M.
///   * Recompute: M*W is formed in an n x p workspace held by the object. Cost: 3 block
///     applications of M (before each pass and for G1).
/// Cost (fast path): about 8 n k p + 6 n p^2 flops in gemm/trsm, O(p^3) dense work.
/// Not thread-safe (workspace); m is applied, never retained beyond the object's lifetime.
template <class T>
class BlockOrthonormalizer {
 public:
  /// Largest accepted column-scaled condition number of R1 (Frobenius estimate): the second pass
  /// then starts from Q1 with ||Q1^H M Q1 - I|| ~ eps * cond^2 <= 1e-5.
  static constexpr double kMaxCholQrCond = 1e5;

  BlockOrthonormalizer() = default;
  /// n rows, block width p >= 1. m: inner-product operator (nullptr = Euclidean), not owned.
  /// store_m: the caller keeps M*basis (run's mbasis); ignored (false) when m == nullptr.
  BlockOrthonormalizer(index_t n, index_t p, LinearOperator<T>* m, bool store_m);

  /// Runs the kernel on basis(:, k:k+p) (ld n, the k leading columns M-orthonormal). mbasis: M *
  /// basis with the same layout in store mode, nullptr otherwise. ratio: deficiency threshold
  /// (see above), in (0, 1). On exit coef (k x p) = C accumulated over all passes and r (p x p) =
  /// R. Throws InvalidInput when a column or its M-image is not finite, and when a random
  /// replacement loses its M-norm against the basis (M not Hermitian positive definite).
  void run(T* basis, T* mbasis, index_t k, double ratio, std::uint64_t seed,
           std::uint64_t& next_stream, DenseMatrix<T>& coef, DenseMatrix<T>& r);

  /// One-projection fast path for Krylov blocks (Euclidean and recompute mode; store mode always
  /// refuses): linalg::orthogonalize_block_cgs2 against V (row-blocked, thread-count independent
  /// sweeps; a second pass only where the DGKS test asks for it) followed by Cholesky-QR twice
  /// inside the block, CholQR2, with M*Q1 = (M W1) R1^{-1} instead of a further M application:
  ///   W1 = W - V C;  G1 = W1^H M W1 = R1^H R1,  Q1 = W1 R1^{-1};  G2 = Q1^H M Q1 = R2^H R2,
  ///   Q = Q1 R2^{-1},  R = R2 R1.
  /// No projection against V follows the Cholesky steps, so R1^{-1} amplifies the rounding-level
  /// components of W1 in span(V) (O(eps / eta) of each column after DGKS) by up to the 2-norm
  /// condition number of the column-scaled W1; the path accepts only blocks with that number <=
  /// kMaxFastCond, no column that lost more than its share (after <= ratio * before, as run()
  /// tests after both passes), ||G2 - I||_F <= 0.5 and |R(i, i)| > ratio * before(i). Cost: 2 or
  /// 3 block applications of M (one per projection pass plus the first), against 3 for run(), and
  /// 2 or 4 row-blocked sweeps over V instead of 4 BLAS calls on the tall n x k operand.
  /// Returns false, with basis(:, k:k+p) restored to the input W (= W1 + V C, up to rounding),
  /// when it refuses (then call run()). On success coef, r, before() and deficient() (all 0) are
  /// set as by run(), and used_fallback() is false.
  /// Precondition beyond run()'s: V M-orthonormal to working precision (the DGKS test's, see
  /// linalg::Reorth). For a V orthonormal only to delta >> eps (a deflation block accepted at
  /// 1e-8) even two projections leave V^H M W1 = O(delta^2 ||W||), which the normalization
  /// amplifies by ||W|| / ||W1||: near an invariant subspace (W1 ~ delta W) the new columns lose
  /// orthogonality to V at the delta level (measured: 2.6e-8 for delta = 1e-10, p = 4). run()
  /// projects the normalized block again and keeps it at rounding level; use run() there.
  bool try_fast(T* basis, T* mbasis, index_t k, double ratio, DenseMatrix<T>& coef,
                DenseMatrix<T>& r);
  /// Largest accepted 2-norm condition number of the column-scaled W1 in try_fast.
  static constexpr double kMaxFastCond = 64.0;

  /// M-norms of the input columns of the last run (sqrt(||C1(:, i)||^2 + G1(i, i)), Pythagoras).
  const std::vector<double>& before() const noexcept { return before_; }
  /// Column i of the last run was replaced by a random vector (R(i, i) = 0).
  const std::vector<char>& deficient() const noexcept { return deficient_; }
  /// Whether the last run took the column-by-column fallback.
  bool used_fallback() const noexcept { return fallback_; }
  /// Runs that took the fallback, over the object's lifetime.
  index_t fallbacks() const noexcept { return fallbacks_; }
  /// try_fast calls that succeeded / refused, over the object's lifetime.
  index_t fast_runs() const noexcept { return fast_runs_; }
  index_t fast_refusals() const noexcept { return fast_refusals_; }

 private:
  /// "M X" for the p columns of x: x itself (Euclidean) or M x written to y.
  const T* apply_m(const T* x, T* y);
  /// CGS2 of column kk against basis(:, 0:kk) (single vector); h (length kk) receives the
  /// coefficients. In store mode mbasis(:, kk) receives M * (the orthogonalized column).
  /// Returns {M-norm before, M-norm after}.
  std::pair<double, double> orthogonalize_column(T* basis, T* mbasis, index_t kk, T* h);
  void scale_column(T* basis, T* mbasis, index_t kk, double norm);
  void column_fallback(T* basis, T* mbasis, index_t k, double ratio, std::uint64_t seed,
                       std::uint64_t& next_stream, DenseMatrix<T>& coef, DenseMatrix<T>& r);

  index_t n_ = 0;
  index_t p_ = 0;
  LinearOperator<T>* m_ = nullptr;
  bool store_m_ = false;
  DenseMatrix<T> mw_;  ///< n x p workspace for M*W (recompute mode only)
  std::vector<double> before_;
  std::vector<char> deficient_;
  bool fallback_ = false;
  index_t fallbacks_ = 0;
  index_t fast_runs_ = 0;
  index_t fast_refusals_ = 0;
};

}  // namespace hs::eigen_detail
