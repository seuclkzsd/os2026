// Conversion of borrowed numpy / scipy.sparse buffers into hs::CsrMatrix for the Python module
// (python/src/module.cpp) and its measurement program (bench/pybind_convert.cpp).
// Header-only and free of pybind11, so the bench program can compile it on its own.
// Scalar set: T in {double, cplx} (result); source values float, double, complex<float>,
// complex<double>; source indices int32_t, int64_t.
//
// Copy policy ("fewest copies", docs/PYTHON.md): every input layout is turned into the owning
// CsrMatrix<T> by passes that read the caller's buffers and write each kept entry exactly once;
// dtype conversions (float32 -> double, int64 -> int32 column indices, real -> complex) happen
// inside that single write. No layout is densified.
//  * csr:   rows that are already strictly sorted are copied as they are; other rows are
//           sorted/merged individually (stable, so duplicates are summed in input order).
//  * csc:   the arrays of a CSC matrix ARE the CSR arrays of A^T. Converting them with conjugated
//           values gives A^H, which is as good as A for hs::solve: it has the same Hermitian
//           part (A + A^H)/2 and the same Hermitian defect, the only things the driver uses (an
//           exactly Hermitian A equals A^H bit for bit). So CSC input needs no transpose.
//  * coo:   counting sort by row straight into the CSR arrays, then per-row sort/merge.
//  * dense: one counting pass and one copy of the nonzeros (any strides, C or Fortran order).
// Canonicalisation follows hs::load_matrix: duplicates are summed, exact zeros (after summation)
// are dropped, non-finite values are rejected with InvalidInput.
//
// Thread safety: all functions are reentrant; the buffers must not be modified concurrently.
// OpenMP: the per-row passes are parallel; results do not depend on the thread count.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::pyconv {

enum class ValueType { F32, F64, C64, C128 };
enum class IndexType { I32, I64 };
enum class Layout { Csr, Csc, Coo, Dense };

inline bool is_complex_type(ValueType t) noexcept { return t == ValueType::C64 || t == ValueType::C128; }

/// Borrowed description of one matrix. Every pointer must stay valid during a call.
///  Csr/Csc: index0 = indptr (major + 1 entries), index1 = indices; both of type index_type.
///  Coo:     index0 = row, index1 = col (nnz entries each, type index_type).
///  Dense:   values = element (0, 0), addressed with byte strides row_stride / col_stride.
/// For Csr/Csc/Coo `values` holds `nvalues` contiguous entries of value_type.
struct MatrixView {
  Layout layout = Layout::Csr;
  index_t nrows = 0;
  index_t ncols = 0;
  IndexType index_type = IndexType::I32;
  const void* index0 = nullptr;
  std::size_t nindex0 = 0;
  const void* index1 = nullptr;
  std::size_t nindex1 = 0;
  ValueType value_type = ValueType::F64;
  const void* values = nullptr;
  std::size_t nvalues = 0;
  std::ptrdiff_t row_stride = 0;  ///< Dense only, bytes
  std::ptrdiff_t col_stride = 0;  ///< Dense only, bytes
  std::string name = "A";         ///< used in error messages and notes
};

/// What canonicalisation did (reported as notes, like hs::load_matrix).
struct ConvertStats {
  index_t duplicates_summed = 0;
  index_t zeros_dropped = 0;
  index_t unsorted_rows = 0;
};

namespace detail {

constexpr index_t kMaxOrder = index_t{1} << 31;  // CSR column indices are int32

template <class S>
inline bool finite(const S& s) {
  if constexpr (is_complex_v<S>) {
    return std::isfinite(s.real()) && std::isfinite(s.imag());
  } else {
    return std::isfinite(s);
  }
}

template <class S>
inline bool nonzero_imag(const S& s) {
  if constexpr (is_complex_v<S>) {
    return s.imag() != 0;
  } else {
    return false;
  }
}

/// Source value -> T. conj applies to complex sources only; a complex source converted to a real
/// T must have a zero imaginary part (checked by the scans before this is called).
template <class T, class S>
inline T convert(const S& s, bool conj) {
  if constexpr (is_complex_v<S>) {
    if constexpr (is_complex_v<T>) {
      return T(static_cast<double>(s.real()), conj ? -static_cast<double>(s.imag()) : static_cast<double>(s.imag()));
    } else {
      return static_cast<T>(s.real());
    }
  } else {
    return T(static_cast<double>(s));
  }
}

enum class Fault : int { None = 0, Range, NonFinite, ImagPart };

/// First failing row over an OpenMP loop (rows are independent; the smallest row wins so the
/// message does not depend on the thread count).
struct FirstFault {
  index_t row = std::numeric_limits<index_t>::max();
  Fault fault = Fault::None;
  index_t position = 0;  ///< entry or column where it happened

  void record(index_t r, Fault f, index_t pos) {
#pragma omp critical(hs_pyconv_fault)
    {
      if (r < row) {
        row = r;
        fault = f;
        position = pos;
      }
    }
  }
  bool any() const { return fault != Fault::None; }
};

[[noreturn]] inline void throw_fault(const std::string& name, const FirstFault& f, const char* what_pos,
                                     index_t minor_dim) {
  switch (f.fault) {
    case Fault::Range:
      HS_THROW(InvalidInput, name << ": index out of range in row/column " << f.row << " (" << what_pos << " "
                                  << f.position << "; the other dimension is " << minor_dim << ")");
    case Fault::NonFinite:
      HS_THROW(InvalidInput, name << ": non-finite value (inf or nan) in row/column " << f.row << " (" << what_pos
                                  << " " << f.position << ")");
    case Fault::ImagPart:
      HS_THROW(Internal, name << ": complex value with a nonzero imaginary part converted to a real matrix");
    case Fault::None:
      break;
  }
  HS_THROW(Internal, name << ": conversion fault without a cause");
}

/// Stable sort by column. Rows are short in practice: insertion sort needs no allocation, whereas
/// std::stable_sort allocates a buffer per call (bench_pybind_convert, 1e6 reversed rows of 7
/// entries, 2 threads: 0.118 s -> 0.105 s).
template <class T>
void stable_sort_by_column(std::vector<std::pair<std::int32_t, T>>& buf) {
  constexpr std::size_t kShortRow = 32;
  if (buf.size() > kShortRow) {
    std::stable_sort(buf.begin(), buf.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    return;
  }
  for (std::size_t p = 1; p < buf.size(); ++p) {
    const auto x = buf[p];
    std::size_t q = p;
    for (; q > 0 && buf[q - 1].first > x.first; --q) buf[q] = buf[q - 1];
    buf[q] = x;
  }
}

/// Stable sort of (column, value) pairs by column, then duplicates summed in input order and
/// exact zeros (after summation) dropped, in place. Returns the number of kept entries.
template <class T>
index_t merge_sorted_row(std::vector<std::pair<std::int32_t, T>>& buf, index_t& duplicates, index_t& zeros) {
  stable_sort_by_column(buf);
  index_t w = 0;
  const index_t len = static_cast<index_t>(buf.size());
  for (index_t p = 0; p < len;) {
    const std::int32_t c = buf[p].first;
    T sum = buf[p].second;
    index_t q = p + 1;
    for (; q < len && buf[q].first == c; ++q) sum += buf[q].second;
    duplicates += q - p - 1;
    if (sum == T(0)) {
      ++zeros;
    } else {
      buf[w++] = {c, sum};
    }
    p = q;
  }
  buf.resize(static_cast<std::size_t>(w));
  return w;
}

inline void prefix_sum(std::vector<index_t>& rowptr) {
  for (std::size_t i = 1; i < rowptr.size(); ++i) rowptr[i] += rowptr[i - 1];
}

// ------------------------------------------------------------------------ csr / csc

template <class T, class I, class S>
CsrMatrix<T> compressed_to_csr(const MatrixView& v, ConvertStats& st) {
  const bool csc = v.layout == Layout::Csc;
  const index_t n = csc ? v.ncols : v.nrows;  // rows of the converted matrix (major dimension)
  const index_t m = csc ? v.nrows : v.ncols;  // its columns
  const bool conj = csc && is_complex_v<T>;
  const I* ptr = static_cast<const I*>(v.index0);
  const I* idx = static_cast<const I*>(v.index1);
  const S* val = static_cast<const S*>(v.values);
  if (v.nindex0 != static_cast<std::size_t>(n) + 1)
    HS_THROW(InvalidInput, v.name << ": indptr has " << v.nindex0 << " entries, expected " << n + 1);
  if (ptr[0] != 0) HS_THROW(InvalidInput, v.name << ": indptr[0] = " << ptr[0] << ", expected 0");
  for (index_t i = 0; i < n; ++i)
    if (ptr[i + 1] < ptr[i]) HS_THROW(InvalidInput, v.name << ": indptr decreases at " << i);
  const index_t nnz_in = static_cast<index_t>(ptr[n]);
  if (static_cast<std::size_t>(nnz_in) > v.nindex1 || static_cast<std::size_t>(nnz_in) > v.nvalues)
    HS_THROW(InvalidInput, v.name << ": indptr[-1] = " << nnz_in << " exceeds the length of indices ("
                                  << v.nindex1 << ") or data (" << v.nvalues << ")");

  CsrMatrix<T> out;
  out.nrows = n;
  out.ncols = m;
  out.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  std::vector<std::uint8_t> sorted_row(static_cast<std::size_t>(n), 1);
  FirstFault fault;
  index_t dups = 0, zeros = 0, unsorted = 0;

  // Pass 1: validate, detect unsorted rows, count kept entries per row.
#pragma omp parallel reduction(+ : dups, zeros, unsorted)
  {
    std::vector<std::pair<std::int32_t, T>> buf;
#pragma omp for schedule(dynamic, 512)
    for (index_t i = 0; i < n; ++i) {
      const index_t b = static_cast<index_t>(ptr[i]), e = static_cast<index_t>(ptr[i + 1]);
      bool sorted = true, ok = true;
      index_t prev = -1, kept = 0;
      for (index_t p = b; p < e; ++p) {
        const index_t j = static_cast<index_t>(idx[p]);
        if (j < 0 || j >= m) {
          fault.record(i, Fault::Range, j);
          ok = false;
          break;
        }
        if (!finite(val[p])) {
          fault.record(i, Fault::NonFinite, j);
          ok = false;
          break;
        }
        if constexpr (!is_complex_v<T>) {
          if (nonzero_imag(val[p])) {
            fault.record(i, Fault::ImagPart, j);
            ok = false;
            break;
          }
        }
        if (j <= prev) sorted = false;
        prev = j;
        if (!(val[p] == S(0))) ++kept;
      }
      if (!ok) continue;
      if (!sorted) {
        ++unsorted;
        sorted_row[static_cast<std::size_t>(i)] = 0;
        buf.clear();
        for (index_t p = b; p < e; ++p) buf.emplace_back(static_cast<std::int32_t>(idx[p]), convert<T>(val[p], conj));
        kept = merge_sorted_row(buf, dups, zeros);
      } else {
        zeros += (e - b) - kept;
      }
      out.rowptr[static_cast<std::size_t>(i) + 1] = kept;
    }
  }
  if (fault.any()) throw_fault(v.name, fault, "index", m);
  prefix_sum(out.rowptr);
  out.colind.resize(static_cast<std::size_t>(out.rowptr.back()));
  out.values.resize(static_cast<std::size_t>(out.rowptr.back()));

  // Pass 2: write each kept entry once.
  index_t unused_dups = 0, unused_zeros = 0;
#pragma omp parallel reduction(+ : unused_dups, unused_zeros)
  {
    std::vector<std::pair<std::int32_t, T>> buf;
#pragma omp for schedule(dynamic, 512)
    for (index_t i = 0; i < n; ++i) {
      const index_t b = static_cast<index_t>(ptr[i]), e = static_cast<index_t>(ptr[i + 1]);
      index_t q = out.rowptr[static_cast<std::size_t>(i)];
      if (sorted_row[static_cast<std::size_t>(i)]) {
        for (index_t p = b; p < e; ++p) {
          if (val[p] == S(0)) continue;
          out.colind[static_cast<std::size_t>(q)] = static_cast<std::int32_t>(idx[p]);
          out.values[static_cast<std::size_t>(q)] = convert<T>(val[p], conj);
          ++q;
        }
      } else {
        buf.clear();
        for (index_t p = b; p < e; ++p) buf.emplace_back(static_cast<std::int32_t>(idx[p]), convert<T>(val[p], conj));
        merge_sorted_row(buf, unused_dups, unused_zeros);
        for (const auto& [c, x] : buf) {
          out.colind[static_cast<std::size_t>(q)] = c;
          out.values[static_cast<std::size_t>(q)] = x;
          ++q;
        }
      }
    }
  }
  st.duplicates_summed += dups;
  st.zeros_dropped += zeros;
  st.unsorted_rows += unsorted;
  return out;
}

// ------------------------------------------------------------------------ coo

template <class T, class I, class S>
CsrMatrix<T> coo_to_csr(const MatrixView& v, ConvertStats& st) {
  const index_t n = v.nrows, m = v.ncols;
  const I* row = static_cast<const I*>(v.index0);
  const I* col = static_cast<const I*>(v.index1);
  const S* val = static_cast<const S*>(v.values);
  const std::size_t nnz = v.nvalues;
  if (v.nindex0 != nnz || v.nindex1 != nnz)
    HS_THROW(InvalidInput, v.name << ": coo arrays differ in length (row " << v.nindex0 << ", col " << v.nindex1
                                  << ", data " << nnz << ")");
  CsrMatrix<T> out;
  out.nrows = n;
  out.ncols = m;
  out.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  // Validate and count entries per row (serial: a histogram).
  for (std::size_t p = 0; p < nnz; ++p) {
    const index_t r = static_cast<index_t>(row[p]), c = static_cast<index_t>(col[p]);
    if (r < 0 || r >= n || c < 0 || c >= m)
      HS_THROW(InvalidInput, v.name << ": coo entry " << p << " at (" << r << ", " << c << ") is outside the "
                                    << n << " x " << m << " matrix");
    if (!finite(val[p]))
      HS_THROW(InvalidInput, v.name << ": non-finite value (inf or nan) at (" << r << ", " << c << ")");
    if constexpr (!is_complex_v<T>) {
      if (nonzero_imag(val[p])) HS_THROW(Internal, v.name << ": complex value converted to a real matrix");
    }
    ++out.rowptr[static_cast<std::size_t>(r) + 1];
  }
  prefix_sum(out.rowptr);
  // Counting sort by row, straight into the CSR arrays (input order kept within each row).
  out.colind.resize(nnz);
  out.values.resize(nnz);
  {
    std::vector<index_t> next(out.rowptr.begin(), out.rowptr.end() - 1);
    for (std::size_t p = 0; p < nnz; ++p) {
      const index_t q = next[static_cast<std::size_t>(row[p])]++;
      out.colind[static_cast<std::size_t>(q)] = static_cast<std::int32_t>(col[p]);
      out.values[static_cast<std::size_t>(q)] = convert<T>(val[p], false);
    }
  }
  // Canonicalise each row in place; kept[i] = entries left at the start of row i's segment.
  std::vector<index_t> kept(static_cast<std::size_t>(n), 0);
  index_t dups = 0, zeros = 0, unsorted = 0;
#pragma omp parallel reduction(+ : dups, zeros, unsorted)
  {
    std::vector<std::pair<std::int32_t, T>> buf;
#pragma omp for schedule(dynamic, 512)
    for (index_t i = 0; i < n; ++i) {
      const index_t b = out.rowptr[static_cast<std::size_t>(i)], e = out.rowptr[static_cast<std::size_t>(i) + 1];
      bool sorted = true, has_zero = false;
      for (index_t p = b; p < e; ++p) {
        if (p > b && out.colind[static_cast<std::size_t>(p)] <= out.colind[static_cast<std::size_t>(p) - 1]) sorted = false;
        if (out.values[static_cast<std::size_t>(p)] == T(0)) has_zero = true;
      }
      if (sorted && !has_zero) {
        kept[static_cast<std::size_t>(i)] = e - b;
        continue;
      }
      if (!sorted) ++unsorted;
      buf.clear();
      for (index_t p = b; p < e; ++p)
        buf.emplace_back(out.colind[static_cast<std::size_t>(p)], out.values[static_cast<std::size_t>(p)]);
      const index_t k = merge_sorted_row(buf, dups, zeros);
      for (index_t t = 0; t < k; ++t) {
        out.colind[static_cast<std::size_t>(b + t)] = buf[static_cast<std::size_t>(t)].first;
        out.values[static_cast<std::size_t>(b + t)] = buf[static_cast<std::size_t>(t)].second;
      }
      kept[static_cast<std::size_t>(i)] = k;
    }
  }
  // Close the gaps left by merged duplicates / dropped zeros (moves only towards the front).
  if (dups + zeros > 0) {
    index_t w = 0;
    for (index_t i = 0; i < n; ++i) {
      const index_t b = out.rowptr[static_cast<std::size_t>(i)], k = kept[static_cast<std::size_t>(i)];
      if (w != b) {
        std::copy(out.colind.begin() + b, out.colind.begin() + b + k, out.colind.begin() + w);
        std::copy(out.values.begin() + b, out.values.begin() + b + k, out.values.begin() + w);
      }
      out.rowptr[static_cast<std::size_t>(i)] = w;
      w += k;
    }
    out.rowptr[static_cast<std::size_t>(n)] = w;
    out.colind.resize(static_cast<std::size_t>(w));
    out.values.resize(static_cast<std::size_t>(w));
    if (4 * w < 3 * static_cast<index_t>(nnz)) {  // more than a quarter unused: give it back
      out.colind.shrink_to_fit();
      out.values.shrink_to_fit();
    }
  }
  st.duplicates_summed += dups;
  st.zeros_dropped += zeros;
  st.unsorted_rows += unsorted;
  return out;
}

// ------------------------------------------------------------------------ dense

template <class T, class S>
CsrMatrix<T> dense_to_csr(const MatrixView& v, ConvertStats& st) {
  const index_t n = v.nrows, m = v.ncols;
  const char* base = static_cast<const char*>(v.values);
  const std::ptrdiff_t rs = v.row_stride, cs = v.col_stride;
  auto at = [&](index_t i, index_t j) -> const S& {
    return *reinterpret_cast<const S*>(base + static_cast<std::ptrdiff_t>(i) * rs + static_cast<std::ptrdiff_t>(j) * cs);
  };
  CsrMatrix<T> out;
  out.nrows = n;
  out.ncols = m;
  out.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  FirstFault fault;
#pragma omp parallel for schedule(dynamic, 64)
  for (index_t i = 0; i < n; ++i) {
    index_t kept = 0;
    for (index_t j = 0; j < m; ++j) {
      const S& x = at(i, j);
      if (!finite(x)) {
        fault.record(i, Fault::NonFinite, j);
        break;
      }
      if constexpr (!is_complex_v<T>) {
        if (nonzero_imag(x)) {
          fault.record(i, Fault::ImagPart, j);
          break;
        }
      }
      if (!(x == S(0))) ++kept;
    }
    out.rowptr[static_cast<std::size_t>(i) + 1] = kept;
  }
  if (fault.any()) throw_fault(v.name, fault, "column", m);
  prefix_sum(out.rowptr);
  out.colind.resize(static_cast<std::size_t>(out.rowptr.back()));
  out.values.resize(static_cast<std::size_t>(out.rowptr.back()));
#pragma omp parallel for schedule(dynamic, 64)
  for (index_t i = 0; i < n; ++i) {
    index_t q = out.rowptr[static_cast<std::size_t>(i)];
    for (index_t j = 0; j < m; ++j) {
      const S& x = at(i, j);
      if (x == S(0)) continue;
      out.colind[static_cast<std::size_t>(q)] = static_cast<std::int32_t>(j);
      out.values[static_cast<std::size_t>(q)] = convert<T>(x, false);
      ++q;
    }
  }
  (void)st;  // dense input: zeros mark structural absence (not reported, as in hs::load_matrix)
  return out;
}

template <class T, class S>
CsrMatrix<T> dispatch_index(const MatrixView& v, ConvertStats& st) {
  switch (v.layout) {
    case Layout::Dense:
      return dense_to_csr<T, S>(v, st);
    case Layout::Csr:
    case Layout::Csc:
      return v.index_type == IndexType::I32 ? compressed_to_csr<T, std::int32_t, S>(v, st)
                                            : compressed_to_csr<T, std::int64_t, S>(v, st);
    case Layout::Coo:
      return v.index_type == IndexType::I32 ? coo_to_csr<T, std::int32_t, S>(v, st)
                                            : coo_to_csr<T, std::int64_t, S>(v, st);
  }
  HS_THROW(Internal, "unknown layout");
}

template <class S>
bool any_nonzero_imag(const MatrixView& v) {
  if constexpr (!is_complex_v<S>) {
    return false;
  } else {
    if (v.layout == Layout::Dense) {
      const char* base = static_cast<const char*>(v.values);
      for (index_t i = 0; i < v.nrows; ++i)
        for (index_t j = 0; j < v.ncols; ++j) {
          const S& x = *reinterpret_cast<const S*>(base + static_cast<std::ptrdiff_t>(i) * v.row_stride +
                                                   static_cast<std::ptrdiff_t>(j) * v.col_stride);
          if (x.imag() != 0) return true;
        }
      return false;
    }
    const S* val = static_cast<const S*>(v.values);
    for (std::size_t p = 0; p < v.nvalues; ++p)
      if (val[p].imag() != 0) return true;
    return false;
  }
}

}  // namespace detail

/// True if some stored value has a nonzero imaginary part (always false for real dtypes). For
/// csr/csc the whole data array is scanned (scipy allows unused trailing entries; scanning them
/// can only make a real problem complex, never the reverse). Early exit; O(nnz) worst case.
inline bool has_nonzero_imag(const MatrixView& v) {
  switch (v.value_type) {
    case ValueType::C64:
      return detail::any_nonzero_imag<std::complex<float>>(v);
    case ValueType::C128:
      return detail::any_nonzero_imag<std::complex<double>>(v);
    default:
      return false;
  }
}

/// Convert `v` to a canonical CsrMatrix<T> (validate() holds): duplicates summed, exact zeros
/// dropped, non-finite values rejected (InvalidInput), shape checked (order < 2^31).
/// T = double requires every value to be real (has_nonzero_imag(v) == false). For Csc the result
/// is A^H (see the file comment), whose shape is ncols x nrows. Stats are accumulated into `st`.
template <class T>
CsrMatrix<T> to_csr(const MatrixView& v, ConvertStats& st) {
  if (v.nrows < 0 || v.ncols < 0 || v.nrows > detail::kMaxOrder || v.ncols > detail::kMaxOrder)
    HS_THROW(InvalidInput, v.name << ": shape " << v.nrows << " x " << v.ncols << " is outside the int32 index range");
  if (v.values == nullptr && (v.nvalues > 0 || (v.layout == Layout::Dense && v.nrows * v.ncols > 0)))
    HS_THROW(InvalidInput, v.name << ": missing data buffer");
  switch (v.value_type) {
    case ValueType::F32:
      return detail::dispatch_index<T, float>(v, st);
    case ValueType::F64:
      return detail::dispatch_index<T, double>(v, st);
    case ValueType::C64:
      return detail::dispatch_index<T, std::complex<float>>(v, st);
    case ValueType::C128:
      return detail::dispatch_index<T, std::complex<double>>(v, st);
  }
  HS_THROW(Internal, "unknown value type");
}

/// Restore full Hermitian storage from one stored triangle (hs::load_matrix semantics):
///   Auto  - strict lower triangle empty and strict upper non-empty -> upper storage, and vice
///           versa; anything else is full storage (no change);
///   Upper / Lower - declared; entries in the other strict triangle throw InvalidInput;
///   Full  - no change.
/// `transposed` = m holds A^H of the user's matrix (csc input): the declared triangle is mirrored
/// accordingly and notes name the user's triangle. Returns a note ("storage: ...") or "".
template <class T>
std::string restore_storage(CsrMatrix<T>& m, Storage declared, bool transposed, const std::string& name) {
  if (declared == Storage::Full) return {};
  bool has_lower = false, has_upper = false;
#pragma omp parallel for reduction(|| : has_lower, has_upper) schedule(static)
  for (index_t i = 0; i < m.nrows; ++i)
    for (index_t p = m.rowptr[static_cast<std::size_t>(i)]; p < m.rowptr[static_cast<std::size_t>(i) + 1]; ++p) {
      const index_t j = m.colind[static_cast<std::size_t>(p)];
      has_lower = has_lower || j < i;
      has_upper = has_upper || j > i;
    }
  auto flip = [&](Storage s) {
    if (!transposed) return s;
    return s == Storage::Upper ? Storage::Lower : s == Storage::Lower ? Storage::Upper : s;
  };
  Storage s = flip(declared);  // in terms of m
  std::string why = "declared by storage='";
  why += declared == Storage::Upper ? "upper'" : declared == Storage::Lower ? "lower'" : "auto'";
  if (s == Storage::Auto) {
    if (!has_lower && has_upper) {
      s = Storage::Upper;
    } else if (has_lower && !has_upper) {
      s = Storage::Lower;
    } else {
      return {};
    }
    why = "auto-detected: the other strict triangle is empty";
  }
  if (s == Storage::Upper && has_lower)
    HS_THROW(InvalidInput, name << ": storage='" << (transposed ? "lower" : "upper")
                                << "' but the matrix has entries in the other strict triangle");
  if (s == Storage::Lower && has_upper)
    HS_THROW(InvalidInput, name << ": storage='" << (transposed ? "upper" : "lower")
                                << "' but the matrix has entries in the other strict triangle");
  m = symmetrize_from_triangle(m);
  const Storage user = flip(s);
  return std::string("storage: ") + (user == Storage::Upper ? "upper" : "lower") +
         " triangle mirrored to full Hermitian storage (" + why + ")";
}

/// Notes in hs::load_matrix wording for the canonicalisation stats.
inline std::vector<std::string> stats_notes(const ConvertStats& st) {
  std::vector<std::string> notes;
  if (st.duplicates_summed > 0)
    notes.push_back("duplicates: summed " + std::to_string(st.duplicates_summed) + " duplicate entries");
  if (st.zeros_dropped > 0)
    notes.push_back("zeros: dropped " + std::to_string(st.zeros_dropped) + " explicit zero entries");
  return notes;
}

}  // namespace hs::pyconv
