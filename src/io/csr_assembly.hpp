// Internal to src/io: canonical CSR from the index/value arrays stored in matrix files, without
// widening them. Inputs reach ~1e8 entries: hs::csr_from_coo takes int64 triplets and a
// (column, value) workspace (~40 bytes per complex entry on top of the input), whereas these
// routines work on the stored int32 indices and place entries straight into the result, keeping
// the peak near the size of the result. Also the scale-safe Hermitian defect of the loader.
// T in {double, cplx}.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::io {

struct AssemblyStats {
  index_t duplicates_summed = 0;  ///< entries merged into an earlier entry at the same position
  index_t zeros_dropped = 0;      ///< exact zeros removed (after summation)
};

/// Checks the index arrays of a compressed layout as read from a file (CSR: ptr = row pointers,
/// idx = column indices; csc: column pointers and row indices): ptr has outer + 1 entries, starts
/// at 0 and never decreases; idx.size() == nvals; ptr[outer] <= idx.size(); idx[0, ptr[outer]) lies
/// in [0, inner). Throws InvalidInput naming the layout. Returns ptr[outer]: entries behind it are
/// unused storage (scipy permits it). OpenMP for the index range check.
index_t check_compressed(const std::vector<index_t>& ptr, index_t outer,
                         const std::vector<std::int32_t>& idx, std::size_t nvals, index_t inner,
                         bool csc);

/// Throws InvalidInput naming the first 0-based coordinate entry outside nrows x ncols.
void check_coordinates(index_t nrows, index_t ncols, const std::int32_t* rows,
                       const std::int32_t* cols, std::size_t nnz);

/// Counting-sort placement of coordinate entries into CSR order. Built from the (range-checked)
/// row index of every entry; next(r) then hands out the positions of row r in turn, so entries
/// placed in input order keep that order within their row and canonicalize_csr sums duplicates in
/// input order. rewind() restarts the hand-out for another array placed in the same order (e.g.
/// the values after the column indices). Memory: 16 bytes per row.
class RowSlots {
 public:
  RowSlots(index_t nrows, const std::int32_t* rows, std::size_t count);
  index_t next(std::int32_t row) { return rowptr_[row] + filled_[row]++; }
  void rewind() { std::fill(filled_.begin(), filled_.end(), 0); }
  index_t total() const { return rowptr_.back(); }
  /// The row pointers of the placed entries (leaves the object unusable).
  std::vector<index_t> take_rowptr() { return std::move(rowptr_); }

 private:
  std::vector<index_t> rowptr_;
  std::vector<index_t> filled_;
};

/// Brings raw CSR arrays (nrows/ncols set; rowptr/colind/values as read from a file) to the
/// canonical form CsrMatrix::validate() demands: checks rowptr and column ranges (InvalidInput),
/// drops unused storage after rowptr[nrows] (scipy permits it), sorts each row (stable, so
/// duplicates are summed in file order), sums duplicates and, if drop_zeros, removes exact zeros.
/// In place apart from a per-thread row buffer; OpenMP over rows. O(nnz log(row length)).
template <class T>
AssemblyStats canonicalize_csr(CsrMatrix<T>& a, bool drop_zeros);

/// Canonical CSR from 0-based int32 triplets (MatrixMarket array format, tests); duplicates are
/// summed in input order. Throws InvalidInput for indices outside nrows x ncols. Memory: input +
/// result.
template <class T>
CsrMatrix<T> csr_from_triplets(index_t nrows, index_t ncols, const std::int32_t* rows,
                               const std::int32_t* cols, const T* vals, std::size_t nnz,
                               bool drop_zeros, AssemblyStats& stats);

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

}  // namespace hs::io
