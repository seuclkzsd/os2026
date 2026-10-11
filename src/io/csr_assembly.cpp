// CSR assembly from file arrays (see csr_assembly.hpp).
#include "io/csr_assembly.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "hsolve/error.hpp"

namespace hs::io {
namespace {

// Position of column j in (sorted) row i, or -1 if absent.
template <class T>
index_t find_entry(const CsrMatrix<T>& a, index_t i, std::int32_t j) {
  const std::int32_t* b = a.colind.data() + a.rowptr[i];
  const std::int32_t* e = a.colind.data() + a.rowptr[i + 1];
  const std::int32_t* it = std::lower_bound(b, e, j);
  return it != e && *it == j ? a.rowptr[i] + (it - b) : -1;
}

// Sum of squares kept as scale^2 * sumsq with scale = the largest magnitude added (LAPACK's
// dlassq): squares are only ever taken of ratios <= 1, so nothing overflows, and a term is lost to
// underflow only when it is below ~1e-308 relative to the largest one, where it cannot change
// the sum.
class ScaledSumSquares {
 public:
  void add(double x) {
    const double ax = std::fabs(x);
    if (ax == 0.0) return;
    if (ax <= scale_) {
      const double r = ax / scale_;
      sumsq_ += r * r;
    } else {
      const double r = scale_ / ax;
      sumsq_ = 1.0 + sumsq_ * r * r;
      scale_ = ax;
    }
  }
  template <class T>
  void add_scalar(const T& v) {  // |v|^2 = re^2 + im^2
    if constexpr (is_complex_v<T>) {
      add(v.real());
      add(v.imag());
    } else {
      add(v);
    }
  }
  void merge(const ScaledSumSquares& o) {
    if (o.scale_ == 0.0) return;
    if (o.scale_ <= scale_) {
      const double r = o.scale_ / scale_;
      sumsq_ += o.sumsq_ * r * r;
    } else {
      const double r = scale_ / o.scale_;
      sumsq_ = o.sumsq_ + sumsq_ * r * r;
      scale_ = o.scale_;
    }
  }
  double scale() const { return scale_; }
  double sumsq() const { return sumsq_; }

 private:
  double scale_ = 0.0;
  double sumsq_ = 0.0;
};

}  // namespace

template <class T>
AssemblyStats canonicalize_csr(CsrMatrix<T>& a, bool drop_zeros) {
  const index_t n = a.nrows;
  std::vector<index_t>& rp = a.rowptr;
  const index_t nz = check_compressed(rp, n, a.colind, a.values.size(), a.ncols, /*csc=*/false);
  a.colind.resize(static_cast<std::size_t>(nz));
  a.values.resize(static_cast<std::size_t>(nz));

  // A row that shrinks records its new length in its last slot, which is then free: column -1
  // marks it, the value holds the length (exact in a double up to 2^53). The compaction below
  // reads it from there, so no per-row length array (8 bytes per row) is needed.
  index_t dups = 0, zeros = 0;
#pragma omp parallel reduction(+ : dups, zeros)
  {
    std::vector<std::pair<std::int32_t, T>> row;  // only for rows that are not strictly sorted
#pragma omp for schedule(dynamic, 256)
    for (index_t i = 0; i < n; ++i) {
      std::int32_t* col = a.colind.data() + rp[i];
      T* val = a.values.data() + rp[i];
      index_t len = rp[i + 1] - rp[i];
      const bool strictly_sorted =
          std::adjacent_find(col, col + len, [](std::int32_t x, std::int32_t y) { return x >= y; }) ==
          col + len;
      if (!strictly_sorted) {
        row.resize(static_cast<std::size_t>(len));
        for (index_t k = 0; k < len; ++k) row[k] = {col[k], val[k]};
        std::stable_sort(row.begin(), row.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        index_t out = 0;
        for (index_t k = 0; k < len;) {
          const std::int32_t c = row[k].first;
          T sum = row[k].second;
          for (++k; k < len && row[k].first == c; ++k) sum += row[k].second;
          col[out] = c;
          val[out] = sum;
          ++out;
        }
        dups += len - out;
        len = out;
      }
      if (drop_zeros) {
        index_t out = 0;
        for (index_t k = 0; k < len; ++k) {
          if (val[k] == T(0)) continue;
          col[out] = col[k];
          val[out] = val[k];
          ++out;
        }
        zeros += len - out;
        len = out;
      }
      const index_t stored = rp[i + 1] - rp[i];
      if (len < stored) {
        col[stored - 1] = -1;
        val[stored - 1] = T(static_cast<double>(len));
      }
    }
  }

  if (dups + zeros > 0) {
    // Close the gaps; rows only shrank, so moving each row left in a forward sweep is safe (it
    // never reaches the marker of the row being moved, nor anything of a later row).
    index_t out = 0;
    for (index_t i = 0; i < n; ++i) {
      const index_t src = rp[i], stored = rp[i + 1] - src;  // rp[i + 1]: not yet rewritten
      const index_t len = stored > 0 && a.colind[src + stored - 1] < 0
                              ? static_cast<index_t>(std::real(a.values[src + stored - 1]))
                              : stored;
      rp[i] = out;
      if (out != src) {
        std::copy(a.colind.begin() + src, a.colind.begin() + src + len, a.colind.begin() + out);
        std::copy(a.values.begin() + src, a.values.begin() + src + len, a.values.begin() + out);
      }
      out += len;
    }
    rp[n] = out;
    a.colind.resize(static_cast<std::size_t>(out));
    a.values.resize(static_cast<std::size_t>(out));
    // Give the slack back when it matters (heavily duplicated coordinate input); shrinking
    // reallocates, so a few merged entries are not worth a transient copy.
    if (out < nz - nz / 8) {
      a.colind.shrink_to_fit();
      a.values.shrink_to_fit();
    }
  }
  return {dups, zeros};
}

template <class T>
CsrMatrix<T> csr_from_triplets(index_t nrows, index_t ncols, const std::int32_t* rows,
                               const std::int32_t* cols, const T* vals, std::size_t nnz,
                               bool drop_zeros, AssemblyStats& stats) {
  check_coordinates(nrows, ncols, rows, cols, nnz);
  RowSlots slots(nrows, rows, nnz);
  CsrMatrix<T> m;
  m.nrows = nrows;
  m.ncols = ncols;
  m.colind.resize(nnz);
  m.values.resize(nnz);
  for (std::size_t k = 0; k < nnz; ++k) {
    const index_t dst = slots.next(rows[k]);
    m.colind[dst] = cols[k];
    m.values[dst] = vals[k];
  }
  m.rowptr = slots.take_rowptr();
  stats = canonicalize_csr(m, drop_zeros);
  return m;
}

index_t check_compressed(const std::vector<index_t>& ptr, index_t outer,
                         const std::vector<std::int32_t>& idx, std::size_t nvals, index_t inner,
                         bool csc) {
  const char* layout = csc ? "CSC" : "CSR";
  const char* outer_name = csc ? "column" : "row";
  const char* index_name = csc ? "row" : "column";
  if (static_cast<index_t>(ptr.size()) != outer + 1)
    HS_THROW(InvalidInput, layout << " index pointer has " << ptr.size() << " entries, expected "
                                  << outer + 1);
  if (ptr[0] != 0) HS_THROW(InvalidInput, layout << " index pointer starts at " << ptr[0] << ", not 0");
  for (index_t i = 0; i < outer; ++i)
    if (ptr[i + 1] < ptr[i])
      HS_THROW(InvalidInput, layout << " index pointer decreases at " << outer_name << " " << i);
  if (idx.size() != nvals)
    HS_THROW(InvalidInput, layout << " has " << idx.size() << " indices but " << nvals << " values");
  const index_t nz = ptr[outer];
  if (static_cast<std::size_t>(nz) > idx.size())
    HS_THROW(InvalidInput, layout << " index pointer ends at " << nz << " but only " << idx.size()
                                  << " entries are stored");
  // Exceptions must not leave an OpenMP region: find the first bad index, throw afterwards.
  index_t bad = nz;
#pragma omp parallel for reduction(min : bad) schedule(static)
  for (index_t p = 0; p < nz; ++p)
    if (idx[p] < 0 || idx[p] >= inner) bad = std::min(bad, p);
  if (bad < nz)
    HS_THROW(InvalidInput, layout << " " << index_name << " index " << idx[bad] << " at position "
                                  << bad << " is outside [0, " << inner << ")");
  return nz;
}

void check_coordinates(index_t nrows, index_t ncols, const std::int32_t* rows,
                       const std::int32_t* cols, std::size_t nnz) {
  const auto count = static_cast<index_t>(nnz);
  index_t bad = count;
#pragma omp parallel for reduction(min : bad) schedule(static)
  for (index_t k = 0; k < count; ++k)
    if (rows[k] < 0 || rows[k] >= nrows || cols[k] < 0 || cols[k] >= ncols) bad = std::min(bad, k);
  if (bad < count)
    HS_THROW(InvalidInput, "entry " << bad << " at (" << rows[bad] << ", " << cols[bad]
                                    << ") lies outside the " << nrows << " x " << ncols << " matrix");
}

RowSlots::RowSlots(index_t nrows, const std::int32_t* rows, std::size_t count)
    : cursor_(static_cast<std::size_t>(nrows) + 1, 0) {
  for (std::size_t k = 0; k < count; ++k) ++cursor_[rows[k] + 1];
  std::partial_sum(cursor_.begin(), cursor_.end(), cursor_.begin());
}

void RowSlots::rewind() {
  // After a complete pass the cursor of row r stands at the first slot of row r + 1: shifting
  // by one restores the row starts. (A last row that is not complete means an incomplete pass.)
  const std::size_t n = cursor_.size() - 1;
  HS_ASSERT(n == 0 || cursor_[n - 1] == cursor_[n], "RowSlots: rewind after an incomplete pass");
  std::copy_backward(cursor_.begin(), cursor_.end() - 1, cursor_.end());
  cursor_[0] = 0;
}

template <class T>
HermitianDefect hermitian_defect_scaled(const CsrMatrix<T>& a) {
  HS_ASSERT(a.square(), "hermitian_defect_scaled: " << a.nrows << " x " << a.ncols);
  struct Partial {
    ScaledSumSquares norm;  ///< ||A||_F^2 over the block's rows
    ScaledSumSquares asym;  ///< ||(A - A^H)/2||_F^2 over the block's rows
    bool exact = true;
  };
  constexpr index_t kBlockRows = 4096;
  const index_t n = a.nrows;
  const index_t nblocks = (n + kBlockRows - 1) / kBlockRows;
  std::vector<Partial> part(static_cast<std::size_t>(nblocks));
#pragma omp parallel for schedule(dynamic)
  for (index_t blk = 0; blk < nblocks; ++blk) {
    Partial& s = part[static_cast<std::size_t>(blk)];
    const index_t last = std::min(n, (blk + 1) * kBlockRows);
    for (index_t i = blk * kBlockRows; i < last; ++i) {
      for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
        const T aij = a.values[p];
        const index_t q = find_entry(a, a.colind[p], static_cast<std::int32_t>(i));
        const T aji_conj = q < 0 ? T(0) : conj_if(a.values[q]);
        s.norm.add_scalar(aij);
        if (aij == aji_conj) continue;
        s.exact = false;
        // Entry (i, j) of (A - A^H)/2; entry (j, i) is visited from row j (or, if not stored, has
        // the same modulus and is never visited: hs::hermitian_defect counts it twice, so do
        // we). Halving first keeps the difference of two finite values finite.
        const T half_diff = 0.5 * aij - 0.5 * aji_conj;
        s.asym.add_scalar(half_diff);
        if (q < 0) s.asym.add_scalar(half_diff);
      }
    }
  }
  Partial total;
  for (const Partial& s : part) {
    total.norm.merge(s.norm);
    total.asym.merge(s.asym);
    total.exact = total.exact && s.exact;
  }
  if (total.exact) return {true, 0.0};
  // ||A - A^H|| / ||A|| = 2 ||(A - A^H)/2|| / ||A||. Every half difference is bounded by the
  // largest entry, so the ratio of the scales is <= 1 and the result is finite; it can only
  // underflow, and then the true value lies below the smallest normal double.
  const double rel = 2.0 * (total.asym.scale() / total.norm.scale()) *
                     std::sqrt(total.asym.sumsq() / total.norm.sumsq());
  return {false, std::max(rel, std::numeric_limits<double>::min())};
}

#define HS_IO_INSTANTIATE_ASSEMBLY(T)                                                             \
  template AssemblyStats canonicalize_csr<T>(CsrMatrix<T>&, bool);                                \
  template CsrMatrix<T> csr_from_triplets<T>(index_t, index_t, const std::int32_t*,               \
                                             const std::int32_t*, const T*, std::size_t, bool,    \
                                             AssemblyStats&);                                     \
  template HermitianDefect hermitian_defect_scaled<T>(const CsrMatrix<T>&);

HS_FOR_EACH_SCALAR(HS_IO_INSTANTIATE_ASSEMBLY)
#undef HS_IO_INSTANTIATE_ASSEMBLY

}  // namespace hs::io
