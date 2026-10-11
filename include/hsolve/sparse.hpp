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

/// Restore full Hermitian storage from one stored triangle: U + U^H - diag(U) (mirrored values
/// conjugated). Throws InvalidInput if `tri` is not square or stores entries in both strict
/// triangles; a diagonal-only matrix is returned unchanged. One counting pass, then the result is
/// allocated once and filled: memory = input + result. O(nnz).
template <class T>
CsrMatrix<T> symmetrize_from_triangle(const CsrMatrix<T>& tri);
/// Same, consuming `tri` (left as an empty 0 x 0 matrix): its arrays are released as soon as they
/// are no longer needed, so the peak is ~1.2x (real) / ~1.3x (complex) the result instead of 1.5x.
template <class T>
CsrMatrix<T> symmetrize_from_triangle(CsrMatrix<T>&& tri);

/// Exact Hermitian part (A + A^H) / 2 on the union pattern of A and A^H (diagonal of a complex
/// matrix becomes real). Used to make "Hermitian within tolerance" input exactly Hermitian so
/// that factorization (one triangle), operators and residuals all see the same matrix.
/// Computed as add(0.5, A, 0.5, A^H) would (bitwise), without forming A^H: entries whose mirror
/// is not stored are found by binary search. Memory: input + result (+ the list of missing
/// mirrors). O(nnz log(row length)); OpenMP over rows. Square A only (InvalidInput otherwise).
template <class T>
CsrMatrix<T> hermitian_part(const CsrMatrix<T>& a);
/// Same values, consuming `a` (left as an empty 0 x 0 matrix). A structurally symmetric A (the
/// usual case: rounding-level asymmetry) is updated in place, with no allocation; otherwise the
/// peak is ~1.7x (real) / ~1.8x (complex) the result instead of ~3x.
template <class T>
CsrMatrix<T> hermitian_part(CsrMatrix<T>&& a);

/// Promote a real matrix to complex (needed when A is complex and B is real, or vice versa).
CsrMatrix<cplx> to_complex(const CsrMatrix<double>& a);
/// Same, consuming `a` (left as an empty 0 x 0 matrix): the row pointers and column indices move
/// over instead of being copied, and the real values are released on return.
CsrMatrix<cplx> to_complex(CsrMatrix<double>&& a);

/// Frobenius norm. O(nnz).
template <class T>
double norm_fro(const CsrMatrix<T>& a);

/// Max column sum of |a_ij| (matrix 1-norm). O(nnz).
template <class T>
double norm_one(const CsrMatrix<T>& a);

/// ||A - A^H||_F / ||A||_F (0 for the zero matrix). Square A only. O(nnz log).
template <class T>
double hermitian_defect(const CsrMatrix<T>& a);

// ---- sparse matrix times dense vectors / blocks (src/sparse/kernels.cpp) ----------------------
//
// Every kernel is OpenMP-parallel over rows and computes each output row on one thread in a fixed
// order, so for a given KernelPath its results are bitwise independent of the thread count and
// reproducible from run to run. Different paths round differently (summation order, FMA): they
// agree to ~1e-15 relative per row sum, not bitwise.
// Multi-vector kernels (p >= 2) first copy X into an interleaved (row-major) block, so that one
// pass over A serves 16 real or 16 complex vectors with contiguous loads instead of p strided
// gathers per nonzero; A is streamed once per 16 vectors. That copy is n*min(p, 16) scalars of
// scratch (KernelOptions::workspace, else allocated per call). spmm keeps the Reference loop when
// rows of X are reused too rarely for the copy to pay off (fewer than 2 entries per column of A on
// average for p <= 4, fewer than 8 for wider blocks: e.g. a diagonal B), and every path computes
// a single vector with the Reference row loop.

/// Instruction-set path of the sparse kernels.
enum class KernelPath {
  Auto,       ///< Avx512 when the CPU and OS support AVX-512F (checked at run time), else Portable
  Reference,  ///< the P1 loops (strided spmm, 8 vectors per pass): for tests and benchmarks
  Portable,   ///< interleaved kernels in plain C++ (compiler-vectorized; any x86-64 / other ISA)
  Avx512,     ///< AVX-512F intrinsics; Unsupported (hs::Error) where AVX-512F is unavailable
};

/// Options of the sparse kernels. The workspace pointer is used during the call only.
struct KernelOptions {
  KernelPath path = KernelPath::Auto;
  /// Scratch for the interleaved copies (grown on demand, contents irrelevant): pass the same
  /// vector to repeated calls to avoid an allocation (and its page faults) per call. nullptr:
  /// allocate per call.
  std::vector<double>* workspace = nullptr;
};

/// Whether `path` can run on this machine (Auto, Reference and Portable always can).
bool kernel_path_available(KernelPath path) noexcept;
/// The path `path` runs as: Auto resolves to Avx512 or Portable; an unavailable Avx512 throws
/// Unsupported. Cheap (the CPU check is done once).
KernelPath resolve_kernel_path(KernelPath path);

/// y = alpha * A * x + beta * y. x has ncols entries, y has nrows entries; y must not alias x.
/// beta == 0 never reads y. OpenMP-parallel over rows.
template <class T>
void spmv(const CsrMatrix<T>& a, const T* x, T* y, T alpha = T(1), T beta = T(0),
          const KernelOptions& options = {});

/// Y = alpha * A * X + beta * Y for p column-major vectors (X: ncols x p, ld ldx >= ncols;
/// Y: nrows x p, ld ldy >= nrows); Y must not overlap X. Equivalent to p spmv calls (p = 1 is
/// one). beta == 0 never reads Y. Rows of Y beyond nrows (ld padding) are never touched.
template <class T>
void spmm(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p,
          T alpha = T(1), T beta = T(0), const KernelOptions& options = {});

/// Refinement residual of a shift-invert solve: R = B X - (A Y - sigma B Y) for p columns, with
/// B = I when b == nullptr (A, B square of equal order n; X, Y, R column-major with leading
/// dimensions >= n; R must not overlap X or Y). The fast paths compute it as B (X + sigma Y) - A Y
/// in one sweep over the rows of A and B (Reference: three spmm passes, as before P4): A and B
/// are read once per 16 vectors and R written once, with 2*n*min(p, 16) scalars of scratch for
/// p >= 2 (a single column is computed straight from x and y).
/// backward_error (optional, p entries): the componentwise backward error of every column,
///   omega_k = max_i |r_ik| / s_ik,  s = |B| |w| + |A| |y|,  w = x + sigma y  (B = I: |w| + |A| |y|),
/// with |z| = |Re z| + |Im z| for complex entries, 0/0 = 0 and r/0 = inf. omega_k at the level of
/// a few eps means r_k is rounding noise of its own evaluation: no refinement step can reduce it.
/// Computed in the same sweep (max is exact: deterministic for any thread count).
template <class T>
void shifted_residual(const CsrMatrix<T>& a, const CsrMatrix<T>* b, double sigma, const T* x,
                      index_t ldx, const T* y, index_t ldy, T* r, index_t ldr, index_t p,
                      const KernelOptions& options = {}, double* backward_error = nullptr);

/// The pencil K(alpha, beta) = alpha*A + beta*B (B = I when b == nullptr) for REAL alpha, beta,
/// stored as the index-only UPPER-triangle pattern of the union of A and B (the only part a
/// Hermitian LDL^T backend reads).
///  * One pattern serves every shift: K(sigma) = 1*A + (-sigma)*B, B alone = 0*A + 1*B, so a
///    factorization analyses once and refactorizes for each (alpha, beta).
///  * assemble() writes values straight into the caller's (backend's) array: the full K is
///    never materialised. Memory: rowptr (8 (n+1) bytes) + colind (4 nnz bytes) + 1 byte per row;
///    assemble() merges the upper rows of A and B into the pattern rows on the fly (no scatter
///    maps), with the same arithmetic as before, i.e. bitwise the same values.
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
  /// Per row: bit 0 (1) if the upper part of A's row is the whole pattern row, bit 1 (2) the same
  /// for B. Such rows are assembled by a plain copy; the others by merging the sorted rows. (The
  /// former int64 scatter maps took 8 bytes per upper entry of A and of B: with them the pencil
  /// of the n = 1e6 waveguide case held 112 MB instead of 33 MB; bench/memio_pencil.cpp.)
  std::vector<std::uint8_t> whole_rows_;
};

/// Real symmetric embedding of a complex Hermitian matrix K (order n):
///   E = [[Re K, -Im K], [Im K, Re K]]  (order 2n).
/// Each eigenvalue of K appears twice in E, so inertia(E) = 2 * inertia(K), and
/// K x = b  <=>  E [Re x; Im x] = [Re b; Im b]. (MUMPS has no Hermitian LDL^H.)
CsrMatrix<double> real_embedding(const CsrMatrix<cplx>& k);

}  // namespace hs
