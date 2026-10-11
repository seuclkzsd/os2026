// Compressed sparse row matrices and the sparse kernels the solver needs.
// Instantiated for T in {double, cplx} unless noted otherwise.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/types.hpp"

namespace hs {

/// CSR matrix. Invariants (checked by validate()):
///   rowptr.size() == nrows + 1, rowptr[0] == 0, rowptr non-decreasing, rowptr[nrows] == nnz;
///   within each row column indices are strictly increasing and lie in [0, ncols);
///   colind.size() == values.size() == nnz.
template <class T>
struct CsrMatrix {
  index_t nrows = 0;
  index_t ncols = 0;
  std::vector<index_t> rowptr{0};
  std::vector<std::int32_t> colind;
  std::vector<T> values;

  index_t nnz() const noexcept { return static_cast<index_t>(colind.size()); }
  bool square() const noexcept { return nrows == ncols; }
  /// Throws Error::Code::InvalidInput describing the first violated invariant. O(nnz).
  void validate() const;
};

/// Build CSR from coordinate triplets (0-based). Duplicates are summed; entries whose value is
/// exactly zero after summation are dropped when drop_zeros is true. O(nnz log(row length)).
template <class T>
CsrMatrix<T> csr_from_coo(index_t nrows, index_t ncols, const std::vector<index_t>& rows,
                          const std::vector<index_t>& cols, const std::vector<T>& vals,
                          bool drop_zeros = true);

/// Conjugate transpose (plain transpose for real T). O(nnz).
template <class T>
CsrMatrix<T> conj_transpose(const CsrMatrix<T>& a);

/// C = alpha * A + beta * B on the union pattern (shapes must match). O(nnz(A) + nnz(B)).
template <class T>
CsrMatrix<T> add(T alpha, const CsrMatrix<T>& a, T beta, const CsrMatrix<T>& b);

/// Identity of order n.
template <class T>
CsrMatrix<T> identity(index_t n);

/// Keep only entries with col >= row (upper=true) or col <= row (upper=false). O(nnz).
template <class T>
CsrMatrix<T> triangle(const CsrMatrix<T>& a, bool upper);

/// Frobenius norm. O(nnz).
template <class T>
double norm_fro(const CsrMatrix<T>& a);

/// Max column sum of |a_ij| (matrix 1-norm). O(nnz).
template <class T>
double norm_one(const CsrMatrix<T>& a);

/// ||A - A^H||_F / ||A||_F (0 for the zero matrix). Square A only. O(nnz).
template <class T>
double hermitian_defect(const CsrMatrix<T>& a);

/// y = alpha * A * x + beta * y. x has ncols entries, y has nrows entries; y must not alias x.
/// OpenMP-parallel over rows; thread-safe w.r.t. distinct outputs.
template <class T>
void spmv(const CsrMatrix<T>& a, const T* x, T* y, T alpha = T(1), T beta = T(0));

/// Y = alpha * A * X + beta * Y for p column-major vectors (X: ncols x p, ld ldx;
/// Y: nrows x p, ld ldy). Equivalent to p spmv calls but streams A once.
template <class T>
void spmm(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p,
          T alpha = T(1), T beta = T(0));

/// Shifted pencil K(sigma) = A - sigma * B with B = I when b == nullptr.
/// The union pattern and the scatter maps are computed once in the constructor, so that
/// assemble() costs O(nnz(A) + nnz(B)) and every K(sigma) shares one sparsity pattern
/// (lets the factorization reuse its symbolic analysis across shifts).
template <class T>
class ShiftedPencil {
 public:
  /// a (and b, if given) must be square and of equal order; they must outlive this object.
  ShiftedPencil(const CsrMatrix<T>& a, const CsrMatrix<T>* b);

  /// The union pattern of A and B (B = I contributes the diagonal). Values are zero.
  const CsrMatrix<T>& pattern() const noexcept { return pattern_; }

  /// Fill k (resized to the union pattern on first use) with A - sigma * B.
  void assemble(double sigma, CsrMatrix<T>& k) const;

  index_t order() const noexcept { return pattern_.nrows; }

 private:
  const CsrMatrix<T>* a_;
  const CsrMatrix<T>* b_;
  CsrMatrix<T> pattern_;
  std::vector<index_t> a_pos_;  ///< position in pattern_ of each entry of A
  std::vector<index_t> b_pos_;  ///< position in pattern_ of each entry of B (or of each diagonal)
};

/// Real symmetric embedding of a complex Hermitian matrix K (order n):
///   E = [[Re K, -Im K], [Im K, Re K]]  (order 2n).
/// Each eigenvalue of K appears twice in E, so inertia(E) = 2 * inertia(K).
CsrMatrix<double> real_embedding(const CsrMatrix<cplx>& k);

}  // namespace hs
