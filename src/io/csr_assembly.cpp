// CSR assembly from file arrays (see csr_assembly.hpp).
#include "io/csr_assembly.hpp"

#include <algorithm>
#include <complex>
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

}  // namespace

template <class T>
AssemblyStats canonicalize_csr(CsrMatrix<T>& a, bool drop_zeros) {
  const index_t n = a.nrows;
  std::vector<index_t>& rp = a.rowptr;
  if (static_cast<index_t>(rp.size()) != n + 1)
    HS_THROW(InvalidInput, "CSR index pointer has " << rp.size() << " entries, expected " << n + 1);
  if (rp[0] != 0) HS_THROW(InvalidInput, "CSR index pointer starts at " << rp[0] << ", not 0");
  for (index_t i = 0; i < n; ++i)
    if (rp[i + 1] < rp[i]) HS_THROW(InvalidInput, "CSR index pointer decreases at row " << i);
  if (a.colind.size() != a.values.size())
    HS_THROW(InvalidInput, "CSR has " << a.colind.size() << " indices but " << a.values.size()
                                      << " values");
  const index_t nz = rp[n];
  if (static_cast<std::size_t>(nz) > a.colind.size())
    HS_THROW(InvalidInput, "CSR index pointer ends at " << nz << " but only " << a.colind.size()
                                                        << " entries are stored");
  a.colind.resize(static_cast<std::size_t>(nz));
  a.values.resize(static_cast<std::size_t>(nz));

  // Exceptions must not leave an OpenMP region: find the first bad index, throw afterwards.
  index_t bad = nz;
  const index_t ncols = a.ncols;
#pragma omp parallel for reduction(min : bad) schedule(static)
  for (index_t p = 0; p < nz; ++p)
    if (a.colind[p] < 0 || a.colind[p] >= ncols) bad = std::min(bad, p);
  if (bad < nz)
    HS_THROW(InvalidInput, "CSR column index " << a.colind[bad] << " at position " << bad
                                               << " is outside [0, " << ncols << ")");

  std::vector<index_t> kept(static_cast<std::size_t>(n));
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
      kept[i] = len;
    }
  }

  if (dups + zeros > 0) {
    // Close the gaps; rows only shrank, so moving each row left in a forward sweep is safe.
    index_t out = 0;
    for (index_t i = 0; i < n; ++i) {
      const index_t src = rp[i];
      rp[i] = out;
      if (out != src) {
        std::copy(a.colind.begin() + src, a.colind.begin() + src + kept[i], a.colind.begin() + out);
        std::copy(a.values.begin() + src, a.values.begin() + src + kept[i], a.values.begin() + out);
      }
      out += kept[i];
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
  CsrMatrix<T> m;
  m.nrows = nrows;
  m.ncols = ncols;
  m.rowptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
  for (std::size_t k = 0; k < nnz; ++k) {
    const index_t r = rows[k], c = cols[k];
    if (r < 0 || r >= nrows || c < 0 || c >= ncols)
      HS_THROW(InvalidInput, "entry " << k << " at (" << r << ", " << c << ") lies outside the "
                                      << nrows << " x " << ncols << " matrix");
    ++m.rowptr[r + 1];
  }
  std::partial_sum(m.rowptr.begin(), m.rowptr.end(), m.rowptr.begin());
  m.colind.resize(nnz);
  m.values.resize(nnz);
  // Counting sort by row; within a row the input order is kept, so the stable per-row sort in
  // canonicalize_csr sums duplicates in input order (deterministic rounding).
  std::vector<index_t> next(m.rowptr.begin(), m.rowptr.end() - 1);
  for (std::size_t k = 0; k < nnz; ++k) {
    const index_t dst = next[rows[k]]++;
    m.colind[dst] = cols[k];
    m.values[dst] = vals[k];
  }
  stats = canonicalize_csr(m, drop_zeros);
  return m;
}

template <class T>
CsrMatrix<T> transpose(const CsrMatrix<T>& a) {
  CsrMatrix<T> t = conj_transpose(a);
  if constexpr (is_complex_v<T>) {
#pragma omp parallel for schedule(static)
    for (index_t p = 0; p < t.nnz(); ++p) t.values[p] = std::conj(t.values[p]);
  }
  return t;
}

template <class T>
bool hermitian_part_in_place(CsrMatrix<T>& a) {
  HS_ASSERT(a.square(), "hermitian_part_in_place: " << a.nrows << " x " << a.ncols);
  const index_t n = a.nrows;
  bool symmetric = true;
#pragma omp parallel for reduction(&& : symmetric) schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
      if (a.colind[p] != i && find_entry(a, a.colind[p], static_cast<std::int32_t>(i)) < 0)
        symmetric = false;
  if (!symmetric) return false;

  // hermitian_part is add(0.5, A, 0.5, A^H): c_ij = 0.5*a_ij + 0.5*conj(a_ji). The same
  // expressions are evaluated here. Each off-diagonal pair is updated by the thread that owns
  // the row of its upper entry, so no value is touched by two threads.
  const T half(0.5);
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      const index_t j = a.colind[p];
      if (j < i) continue;
      if (j == i) {
        a.values[p] = half * a.values[p] + half * conj_if(a.values[p]);
        continue;
      }
      const index_t q = find_entry(a, j, static_cast<std::int32_t>(i));
      const T aij = a.values[p], aji = a.values[q];
      a.values[p] = half * aij + half * conj_if(aji);
      a.values[q] = half * aji + half * conj_if(aij);
    }
  }
  return true;
}

#define HS_IO_INSTANTIATE_ASSEMBLY(T)                                                             \
  template AssemblyStats canonicalize_csr<T>(CsrMatrix<T>&, bool);                                \
  template CsrMatrix<T> csr_from_triplets<T>(index_t, index_t, const std::int32_t*,               \
                                             const std::int32_t*, const T*, std::size_t, bool,    \
                                             AssemblyStats&);                                     \
  template CsrMatrix<T> transpose<T>(const CsrMatrix<T>&);                                        \
  template bool hermitian_part_in_place<T>(CsrMatrix<T>&);

HS_FOR_EACH_SCALAR(HS_IO_INSTANTIATE_ASSEMBLY)
#undef HS_IO_INSTANTIATE_ASSEMBLY

}  // namespace hs::io
