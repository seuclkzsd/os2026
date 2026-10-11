// Projected eigenproblem of the general Krylov-Schur solver (module-internal, src/eigen).
//
// The Rayleigh quotient H (m x m, m = basis size, at most a few hundred) is reduced to Schur
// form H = Q S Q^H. Everything here is O(m^3) dense work on small matrices.
// Real T: S is quasi-upper-triangular (LAPACK standardized form); every complex-conjugate pair
// of Ritz values occupies one 2x2 diagonal block and is handled as one indivisible unit, which is
// what guarantees that pairs are never split by selection, truncation or reordering.
// Complex T: S is upper triangular and every block is 1x1.
// Scalar set: T in {double, cplx}.
#pragma once

#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/types.hpp"

namespace hs::eigen_detail {

/// A diagonal block of the Schur form: one Ritz value, or (real T) a conjugate pair.
struct SchurBlock {
  index_t start = 0;  ///< first row/column of the block in S
  index_t size = 1;   ///< 1, or 2 for a conjugate pair (real T only)
  cplx value;         ///< the Ritz value; for a pair the member with positive imaginary part
};

/// Eigenvectors of the projected matrix and their residual estimates.
template <class T>
struct RitzVectors {
  /// m x m eigenvectors y of H (coordinates in the Krylov basis, x = V y), unit 2-norm.
  /// Complex T: column i belongs to the Ritz value at position i.
  /// Real T (LAPACK layout): a real Ritz value at position i has eigenvector column i; a pair at
  /// positions (i, i+1) stores (Re y, Im y) of the eigenvector of the positive-imaginary member
  /// (the other member's eigenvector is the conjugate), scaled so ||Re y||^2 + ||Im y||^2 = 1.
  DenseMatrix<T> y;
  /// Per position: |b^T y|. With the Krylov-Schur relation Op V = V H + v b^T and orthonormal
  /// [V v], this equals ||Op x - theta x||_2 for x = V y (exactly in exact arithmetic; both
  /// members of a pair share it).
  std::vector<double> estimate;
};

/// Schur decomposition H = Q S Q^H of a small dense projected matrix, with block bookkeeping.
template <class T>
class ProjectedSchur {
 public:
  /// Schur decomposition of the m x m matrix h (column-major, leading dimension ldh >= m).
  ProjectedSchur(index_t m, const T* h, index_t ldh);

  index_t order() const noexcept { return s_.rows(); }
  const DenseMatrix<T>& form() const noexcept { return s_; }     ///< S
  const DenseMatrix<T>& vectors() const noexcept { return q_; }  ///< Q (orthonormal / unitary)
  /// Diagonal blocks of S in positional order.
  const std::vector<SchurBlock>& blocks() const noexcept { return blocks_; }

  /// Whether S(0:p, 0:p) is a closed leading block (p == 0, p == m, or no 2x2 block across p).
  bool is_boundary(index_t p) const;

  /// Block indices ordered best first: largest |theta| (LargestMagnitude), largest real part
  /// (LargestAlgebraic) or smallest real part (SmallestAlgebraic). Equal keys keep positional
  /// order, so the ranking is deterministic.
  std::vector<index_t> ranking(Which which) const;

  /// Move the flagged blocks (one flag per entry of blocks()) to the leading positions, keeping
  /// their relative order, by an orthogonal (unitary) reordering of S and Q (LAPACK trsen).
  /// Returns false when LAPACK refused to swap two adjacent blocks with nearly equal eigenvalues
  /// (possible for real T only; the swap would not be backward stable). S, Q then remain an exact
  /// Schur decomposition that is only partially reordered. blocks() is refreshed either way.
  bool reorder_to_front(const std::vector<char>& selected_blocks);

  /// Ritz vectors of H and residual estimates for the coupling row b (length m) of the
  /// Krylov-Schur relation. O(m^3).
  RitzVectors<T> ritz_vectors(const T* b) const;

 private:
  void refresh_blocks();

  DenseMatrix<T> s_;
  DenseMatrix<T> q_;
  std::vector<cplx> w_;  ///< eigenvalues by position (LAPACK output order)
  std::vector<SchurBlock> blocks_;
};

/// sum_i b_i y_i with a plain (unconjugated) transpose: the coupling row b^T of the Krylov-Schur
/// relation Op V = V H + v b^T multiplies coordinates directly.
template <class T>
inline T row_times_column(index_t m, const T* b, const T* y) {
  T acc = T(0);
  for (index_t i = 0; i < m; ++i) acc += b[i] * y[i];
  return acc;
}

/// Number of leading entries of `ranked` (block indices) whose blocks hold at least `count`
/// eigenvalues; *held receives the eigenvalue count, which is count + 1 when a pair straddles.
index_t blocks_for_count(const std::vector<SchurBlock>& blocks, const std::vector<index_t>& ranked,
                         index_t count, index_t* held);

}  // namespace hs::eigen_detail
