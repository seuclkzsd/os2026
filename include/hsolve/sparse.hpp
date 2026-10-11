// Compressed sparse row matrices and the sparse kernels the solver needs.
// Scalar set: T in {double, cplx} unless noted otherwise (instantiated in src/sparse/).
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/types.hpp"

namespace hs {

/// CSR matrix. Invariants (checked by validate()):
///   rowptr.size() == nrows + 1, rowptr[0] == 0, rowptr non-decreasing, rowptr[nrows] == nnz;
///   within each row column indices are strictly increasing and lie in [0, ncols);
///   colind.size() == values.size() == nnz; ncols <= 2^31.
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

/// C = alpha * A + beta * B on the union pattern (shapes must match; no zero dropping).
template <class T>
CsrMatrix<T> add(T alpha, const CsrMatrix<T>& a, T beta, const CsrMatrix<T>& b);

/// Identity of order n.
template <class T>
CsrMatrix<T> identity(index_t n);

/// Keep only entries with col >= row (upper=true) or col <= row (upper=false). O(nnz).
template <class T>
CsrMatrix<T> triangle(const CsrMatrix<T>& a, bool upper);

/// Restore full Hermitian storage from one stored triangle: U + U^H - diag(U).
/// Precondition: `tri` holds only one triangle (plus diagonal). O(nnz).
template <class T>
CsrMatrix<T> symmetrize_from_triangle(const CsrMatrix<T>& tri);

/// Exact Hermitian part (A + A^H) / 2 on the union pattern of A and A^H (diagonal of a complex
/// matrix becomes real). Used to make "Hermitian within tolerance" input exactly Hermitian so
/// that factorization (one triangle), operators and residuals all see the same matrix.
template <class T>
CsrMatrix<T> hermitian_part(const CsrMatrix<T>& a);

/// Promote a real matrix to complex (needed when A is complex and B is real, or vice versa).
CsrMatrix<cplx> to_complex(const CsrMatrix<double>& a);

/// Frobenius norm. O(nnz).
template <class T>
double norm_fro(const CsrMatrix<T>& a);

/// Max column sum of |a_ij| (matrix 1-norm). O(nnz).
template <class T>
double norm_one(const CsrMatrix<T>& a);

/// ||A - A^H||_F / ||A||_F (0 for the zero matrix). Square A only. O(nnz log).
template <class T>
double hermitian_defect(const CsrMatrix<T>& a);

/// y = alpha * A * x + beta * y. x has ncols entries, y has nrows entries; y must not alias x.
/// beta == 0 never reads y. OpenMP-parallel over rows.
template <class T>
void spmv(const CsrMatrix<T>& a, const T* x, T* y, T alpha = T(1), T beta = T(0));

/// Y = alpha * A * X + beta * Y for p column-major vectors (X: ncols x p, ld ldx;
/// Y: nrows x p, ld ldy). Equivalent to p spmv calls but streams A once per 8 vectors.
template <class T>
void spmm(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p,
          T alpha = T(1), T beta = T(0));

/// The pencil K(alpha, beta) = alpha*A + beta*B (B = I when b == nullptr) for REAL alpha, beta,
/// stored as the index-only UPPER-triangle pattern of the union of A and B (the only part a
/// Hermitian LDL^T backend reads), plus scatter maps from A and B into it.
///  * One pattern serves every shift: K(sigma) = 1*A + (-sigma)*B, B alone = 0*A + 1*B, so a
///    factorization analyses once and refactorizes for each (alpha, beta).
///  * assemble() writes values straight into the caller's (backend's) array: the full K is
///    never materialised. Memory: rowptr (n+1) + colind (nnz) + maps (~nnz(A)/2 + nnz(B)/2).
/// A and B must be square, of equal order, exactly Hermitian (see hermitian_part) and outlive
/// the pencil.
template <class T>
class ShiftedPencil {
 public:
  ShiftedPencil(const CsrMatrix<T>& a, const CsrMatrix<T>* b);
  ShiftedPencil(CsrMatrix<T>&& a, const CsrMatrix<T>* b) = delete;  // would dangle

  index_t order() const noexcept { return n_; }
  /// Number of entries in the upper-triangle union pattern (diagonal included).
  index_t nnz() const noexcept { return static_cast<index_t>(colind_.size()); }
  /// Row pointers / sorted column indices of the upper pattern (row i holds columns >= i;
  /// the diagonal is always present).
  const std::vector<index_t>& rowptr() const noexcept { return rowptr_; }
  const std::vector<std::int32_t>& colind() const noexcept { return colind_; }
  const CsrMatrix<T>& a() const noexcept { return *a_; }
  const CsrMatrix<T>* b() const noexcept { return b_; }

  /// vals[p] = (alpha*A + beta*B)[upper entry p], for p in [0, nnz()). U is T or the
  /// single-precision counterpart rebind_t<T, float> (values are rounded). OpenMP, O(nnz).
  template <class U>
  void assemble(double alpha, double beta, U* vals) const;

  /// Full (both triangles) CSR of alpha*A + beta*B on the union pattern. Allocates; for tests,
  /// small problems and verification only.
  CsrMatrix<T> assemble_full(double alpha, double beta) const;

 private:
  index_t n_ = 0;
  const CsrMatrix<T>* a_;
  const CsrMatrix<T>* b_;
  std::vector<index_t> rowptr_;
  std::vector<std::int32_t> colind_;
  std::vector<index_t> a_first_upper_;  ///< per row: first entry of A with col >= row
  std::vector<index_t> a_dst_;          ///< pattern position of each upper entry of A (row order)
  bool a_identity_ = false;             ///< upper(A) pattern == union pattern: a_dst_ unused
  std::vector<index_t> b_first_upper_;  ///< same for B (unused when b_ == nullptr)
  std::vector<index_t> b_dst_;          ///< pattern position of each upper entry of B / of each diagonal
};

/// Real symmetric embedding of a complex Hermitian matrix K (order n):
///   E = [[Re K, -Im K], [Im K, Re K]]  (order 2n).
/// Each eigenvalue of K appears twice in E, so inertia(E) = 2 * inertia(K), and
/// K x = b  <=>  E [Re x; Im x] = [Re b; Im b]. (MUMPS has no Hermitian LDL^H.)
CsrMatrix<double> real_embedding(const CsrMatrix<cplx>& k);

}  // namespace hs
