// Internal to src/io: canonical CSR from the index/value arrays stored in matrix files, without
// widening them. Inputs reach ~1e8 entries: hs::csr_from_coo takes int64 triplets and a
// (column, value) workspace (~40 bytes per complex entry on top of the input), whereas these
// routines work on the stored int32 indices and assemble in place, keeping the peak near the
// size of the result. Also the Hermitian check of the loader (scale-safe defect, Hermitian part in
// place). T in {double, cplx}.
#pragma once

#include <cstddef>
#include <cstdint>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::io {

struct AssemblyStats {
  index_t duplicates_summed = 0;  ///< entries merged into an earlier entry at the same position
  index_t zeros_dropped = 0;      ///< exact zeros removed (after summation)
};

/// Brings raw CSR arrays (nrows/ncols set; rowptr/colind/values as read from a file) to the
/// canonical form CsrMatrix::validate() demands: checks rowptr and column ranges (InvalidInput),
/// drops unused storage after rowptr[nrows] (scipy permits it), sorts each row (stable, so
/// duplicates are summed in file order), sums duplicates and, if drop_zeros, removes exact zeros.
/// In place apart from a per-thread row buffer; OpenMP over rows. O(nnz log(row length)).
template <class T>
AssemblyStats canonicalize_csr(CsrMatrix<T>& a, bool drop_zeros);

/// Canonical CSR from 0-based int32 triplets (scipy coo, MatrixMarket); duplicates are summed in
/// input order. Throws InvalidInput for indices outside nrows x ncols. Memory: input + result.
template <class T>
CsrMatrix<T> csr_from_triplets(index_t nrows, index_t ncols, const std::int32_t* rows,
                               const std::int32_t* cols, const T* vals, std::size_t nnz,
                               bool drop_zeros, AssemblyStats& stats);

/// Plain transpose A^T (no conjugation): the CSC arrays of A are the CSR arrays of A^T.
template <class T>
CsrMatrix<T> transpose(const CsrMatrix<T>& a);

struct HermitianDefect {
  /// a_ij == conj(a_ji) for every i, j (a structurally absent entry counts as 0): decided by
  /// comparing values, never by the computed norms.
  bool exact = true;
  /// ||A - A^H||_F / ||A||_F; > 0 exactly when !exact.
  double relative = 0.0;
};

/// Hermitian defect of a canonical square A with finite values, valid at any magnitude of the
/// entries. hs::hermitian_defect squares the entries unscaled: below ~1e-154 the squares underflow
/// and a non-Hermitian matrix reports 0; above ~1e154 they overflow and the ratio is inf/inf =
/// NaN. Here both norms are scaled sums of squares (LAPACK's lassq), and A - A^H is formed from
/// halved entries so that the difference of two finite values cannot overflow. A positive defect
/// below the double range (largest asymmetry under ~1e-308 relative to the largest entry) is
/// reported as the smallest normal double, keeping "relative > 0" equivalent to "!exact".
/// O(nnz log(row length)); OpenMP over fixed row blocks merged in order, so the result does not
/// depend on the number of threads.
template <class T>
HermitianDefect hermitian_defect_scaled(const CsrMatrix<T>& a);

/// Replaces a canonical square A by (A + A^H)/2 in place when its pattern is structurally
/// symmetric (the usual case: assembled matrices with rounding-level asymmetry), using exactly the
/// arithmetic of hs::hermitian_part, so the result is bitwise identical. Returns false and leaves
/// A untouched otherwise (the union pattern then needs new storage: use hermitian_part).
/// O(nnz log(row length)); OpenMP over rows.
template <class T>
bool hermitian_part_in_place(CsrMatrix<T>& a);

}  // namespace hs::io
