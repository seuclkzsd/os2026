// Module-internal helpers shared by lobpcg and jacobi_davidson (src/eigen): operator norm
// estimates, B-orthonormalization of blocks with rank dropping, in-place basis rotations, the
// robust Rayleigh-Ritz step for a nearly B-orthonormal basis, and the final residual check.
// Scalar set: T in {double, cplx} (instantiated in src/eigen/lobpcg_jd_common.cpp).
// All blocks are column-major with leading dimension n (the operator order).
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/eigen_iterative.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"

namespace hs::eigen_detail {

/// Y = Op X for p columns, or Y = X when op == nullptr (identity). ld = n for both.
template <class T>
void apply_or_copy(LinearOperator<T>* op, index_t n, const T* x, T* y, index_t p);

/// Largest |Ritz value| of 30 Lanczos steps on the Hermitian op (Euclidean inner product): a lower
/// bound of ||op||_2, typically within a few percent (the driver's definition, DESIGN §4.5).
template <class T>
double estimate_norm2(LinearOperator<T>& op, std::uint64_t seed);

/// V(:, 0:k) <- V(:, 0:m) * S for S (m x k, ld lds), in place, one block of rows at a time
/// (each row of the result depends only on the same row of V). V has ld n and at least
/// max(m, k) columns.
template <class T>
void rotate_in_place(index_t n, T* v, index_t m, const T* s, index_t lds, index_t k);

/// SVQB whitening (Stathopoulos & Wu 2002) with dropping: for a Hermitian positive semidefinite
/// k x k Gram matrix g with positive diagonal, D = diag(g)^{-1/2} and D g D = U diag(mu) U^H,
/// returns q and M = D U(:, kept) diag(mu_kept)^{-1/2} (k x q, M^H g M = I), keeping the
/// directions with mu > 1e-12 mu_max, most independent (largest mu) first. cond (optional)
/// receives mu_max / mu_min over the kept directions. q == 0 when g has no positive eigenvalue.
template <class T>
index_t svqb(const DenseMatrix<T>& g, DenseMatrix<T>& m, double* cond = nullptr);

/// A block that is already B-orthonormal: columns v (n x cols) and bv = B v (== v when B = I).
template <class T>
struct OrthoBlock {
  const T* v = nullptr;
  const T* bv = nullptr;
  index_t cols = 0;
};

/// B-orthonormalizes the p columns of W against every block of `against` (each B-orthonormal and
/// mutually B-orthogonal) and among themselves: rounds of {two block CGS passes, BW = B W, SVQB
/// with dropping}. A column is dropped when less than drop_rel of its B-norm survives the
/// projection (it lies in span(against) up to rounding), and SVQB drops directions whose scaled
/// Gram eigenvalue is below 1e-12 of the largest (numerically dependent columns). Rounds repeat
/// (at most 3) until the projection removed at most 1e4 times what remains and the scaled Gram
/// matrix had condition <= 1e4, which leaves W B-orthonormal and B-orthogonal to `against` to
/// ~1e-12. The kept columns (generally linear combinations of the input columns) are compacted
/// to the front of W, and of BW (= B W on exit; bw may be nullptr only when b == nullptr).
/// Returns the number of kept columns (0..p). Throws InvalidInput when W or B W is not finite.
template <class T>
index_t b_orthonormalize(index_t n, const std::vector<OrthoBlock<T>>& against, T* w, T* bw, index_t p,
                         LinearOperator<T>* b, double drop_rel);

/// Rayleigh-Ritz for the pencil (G_A, G_B) restricted to the index set idx of a k_s x k_s problem
/// (G_B ~ I, the Gram matrix of a nearly B-orthonormal basis). With D = diag(G_B)^{-1/2} and
/// D G_B D = U diag(mu) U^H, returns false (nothing computed) when mu_min < ill_cond * mu_max
/// (ill-conditioned basis) unless `force`; otherwise drops the directions with mu < 1e-12 mu_max,
/// solves the remaining standard problem and returns the Ritz values (ascending) in theta and the
/// coefficient vectors in c (k_s x q, zero rows outside idx), normalized so that
/// c^H G_B c = I. `deviation` receives max |G_B(idx, idx) - I|.
template <class T>
bool rayleigh_ritz(const DenseMatrix<T>& ga, const DenseMatrix<T>& gb, const std::vector<index_t>& idx,
                   double ill_cond, bool force, std::vector<double>& theta, DenseMatrix<T>& c,
                   double* deviation = nullptr);

/// Backward error eta = r / ((norm_a + |theta| norm_b) * xnorm), 0 when r == 0.
double backward_error(double r, double theta, double xnorm, double norm_a, double norm_b);

/// Fresh residuals of the p columns of X: residual[j] = ||A x_j - theta_j B x_j||_2 and eta[j]
/// (backward error). Applies A and B once to the block (2 n p workspace).
template <class T>
void fresh_residuals(LinearOperator<T>& a, LinearOperator<T>* b, index_t n, const T* x, index_t p,
                     const std::vector<double>& theta, double norm_a, double norm_b,
                     std::vector<double>& residual, std::vector<double>& eta);

/// Replaces the square matrix m by its Hermitian part (both triangles; real diagonal).
template <class T>
void hermitize(DenseMatrix<T>& m);

/// Throws InvalidInput naming `what` if any of the p columns (ld n) holds Inf/NaN.
template <class T>
void require_finite(index_t n, const T* x, index_t p, const char* what);

/// Largest accepted max |Y^H B Y - I| of a deflation block (LobpcgOptions / JdOptions::deflation).
inline constexpr double kDeflationOrthoTol = 1e-8;

/// Throws InvalidInput (message prefixed by `who`) unless the d columns y (ld n), with by = B y
/// (== y when B = I), are B-orthonormal: max |Y^H B Y - I| <= kDeflationOrthoTol (NaN fails).
template <class T>
void require_b_orthonormal(index_t n, index_t d, const T* y, const T* by, const char* who);

}  // namespace hs::eigen_detail
