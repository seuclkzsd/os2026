// CSR construction, structural operations and norms (include/hsolve/sparse.hpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"

namespace hs {

namespace {

constexpr index_t kMaxCol = std::numeric_limits<std::int32_t>::max();

template <class T>
bool is_zero(const T& v) {
  return v == T(0);
}

template <class V>
void release(std::vector<V>& v) {
  std::vector<V>().swap(v);
}

// Position of column j in (sorted) row i of a, or -1 if absent. O(log(row length)).
template <class T>
index_t find_entry(const CsrMatrix<T>& a, index_t i, index_t j) {
  const std::int32_t* b = a.colind.data() + a.rowptr[i];
  const std::int32_t* e = a.colind.data() + a.rowptr[i + 1];
  const std::int32_t* it = std::lower_bound(b, e, static_cast<std::int32_t>(j));
  return it != e && *it == j ? a.rowptr[i] + (it - b) : -1;
}

// ---- symmetrize_from_triangle: U + U^H - diag(U) after one counting pass.
// Row i of the result holds its own stored entries plus the mirror images of the strict entries
// stored in column i. For upper storage the mirrored entries have columns < i and come first, for
// lower storage they come last; visiting the source rows in increasing order appends them sorted.

// Which strict triangle holds entries: +1 upper, -1 lower, 0 neither (diagonal only). Rows are
// sorted, so the first and last entries of a row decide. Throws InvalidInput when both do.
template <class T>
int stored_triangle(const CsrMatrix<T>& tri) {
  bool lower = false, upper = false;
#pragma omp parallel for reduction(|| : lower, upper) schedule(static)
  for (index_t i = 0; i < tri.nrows; ++i) {
    const index_t b = tri.rowptr[i], e = tri.rowptr[i + 1];
    if (b == e) continue;
    lower = lower || tri.colind[b] < i;
    upper = upper || tri.colind[e - 1] > i;
  }
  if (lower && upper)
    HS_THROW(InvalidInput, "symmetrize_from_triangle: input holds entries in both triangles");
  return upper ? 1 : (lower ? -1 : 0);
}

// Row pointers of the full matrix: own entries plus mirrored strict entries per row. O(nnz).
template <class T>
std::vector<index_t> mirrored_rowptr(const CsrMatrix<T>& tri) {
  const index_t n = tri.nrows;
  std::vector<index_t> rp(static_cast<std::size_t>(n) + 1, 0);
  for (index_t i = 0; i < n; ++i) {
    rp[i + 1] += tri.rowptr[i + 1] - tri.rowptr[i];
    for (index_t p = tri.rowptr[i]; p < tri.rowptr[i + 1]; ++p)
      if (tri.colind[p] != i) ++rp[tri.colind[p] + 1];
  }
  std::partial_sum(rp.begin(), rp.end(), rp.begin());
  return rp;
}

// dst[own position of entry p] = src[p] for every stored entry (OpenMP over rows).
template <class T, class V>
void copy_own(const CsrMatrix<T>& tri, const std::vector<index_t>& rp, bool upper, const V* src,
              V* dst) {
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < tri.nrows; ++i) {
    const index_t b = tri.rowptr[i], len = tri.rowptr[i + 1] - b;
    std::copy(src + b, src + b + len, dst + (upper ? rp[i + 1] - len : rp[i]));
  }
}

// put(position, p, j) for the mirror image of every strict entry p = (j, c), which lands in
// row c. Sequential in j, which keeps the mirrored block of every row sorted.
template <class T, class Put>
void scatter_mirrored(const CsrMatrix<T>& tri, const std::vector<index_t>& rp, bool upper, Put&& put) {
  const index_t n = tri.nrows;
  // Offset of the next mirrored entry within its row (rows hold fewer than 2^31 entries).
  std::vector<std::int32_t> next(static_cast<std::size_t>(n));
  for (index_t c = 0; c < n; ++c)
    next[c] = upper ? 0 : static_cast<std::int32_t>(tri.rowptr[c + 1] - tri.rowptr[c]);
  for (index_t j = 0; j < n; ++j)
    for (index_t p = tri.rowptr[j]; p < tri.rowptr[j + 1]; ++p) {
      const index_t c = tri.colind[p];
      if (c != j) put(rp[c] + next[c]++, p, j);
    }
}

// ---- hermitian_part: (A + A^H)/2 on the union pattern of A and A^H without forming A^H.
// An entry a_ij (i != j) whose mirror a_ji is not stored (an "orphan") adds the entry (j, i).
// Values use exactly the arithmetic of add(0.5, A, 0.5, A^H), so every variant agrees bitwise.

struct Insertions {
  index_t total = 0;
  std::vector<index_t> ptr;       ///< n + 1 row pointers (empty when total == 0)
  std::vector<std::int32_t> col;  ///< inserted columns of each row, ascending
};

template <class T>
Insertions orphan_mirrors(const CsrMatrix<T>& a) {
  const index_t n = a.nrows;
  const auto orphan = [&a](index_t i, index_t p) {
    const index_t j = a.colind[p];
    return j != i && find_entry(a, j, i) < 0;
  };
  std::vector<std::int32_t> per_row(static_cast<std::size_t>(n));  // orphans stored in row i
  index_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    std::int32_t cnt = 0;
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) cnt += orphan(i, p);
    per_row[i] = cnt;
    total += cnt;
  }
  Insertions ins;
  if (total == 0) return ins;
  ins.total = total;
  ins.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (index_t i = 0; i < n; ++i)
    if (per_row[i] > 0)
      for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
        if (orphan(i, p)) ++ins.ptr[a.colind[p] + 1];
  std::partial_sum(ins.ptr.begin(), ins.ptr.end(), ins.ptr.begin());
  ins.col.resize(static_cast<std::size_t>(total));
  // ptr[j] serves as the fill cursor of row j and is shifted back afterwards; increasing source
  // rows i keep every row of insertions sorted.
  for (index_t i = 0; i < n; ++i)
    if (per_row[i] > 0)
      for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
        if (orphan(i, p)) ins.col[ins.ptr[a.colind[p]]++] = static_cast<std::int32_t>(i);
  for (index_t j = n; j > 0; --j) ins.ptr[j] = ins.ptr[j - 1];
  ins.ptr[0] = 0;
  return ins;
}

template <class T>
std::vector<index_t> union_rowptr(const CsrMatrix<T>& a, const Insertions& ins) {
  const index_t n = a.nrows;
  std::vector<index_t> rp(static_cast<std::size_t>(n) + 1, 0);
  for (index_t i = 0; i < n; ++i)
    rp[i + 1] = rp[i] + (a.rowptr[i + 1] - a.rowptr[i]) + (ins.ptr[i + 1] - ins.ptr[i]);
  return rp;
}

// Walks row i of the union pattern, starting at output position o: visit(position, column,
// value) in column order, value = c_ij computed as add(0.5, A, 0.5, A^H) does: 0.5*a_ij +
// 0.5*conj(a_ji) when both are stored, 0.5*a_ij or 0.5*conj(a_ji) when only one is.
template <class T, class Visit>
void union_row(const CsrMatrix<T>& a, const Insertions& ins, index_t i, index_t o, Visit&& visit) {
  const T half(0.5);
  index_t p = a.rowptr[i], q = 0, qe = 0;
  const index_t pe = a.rowptr[i + 1];
  if (ins.total > 0) {
    q = ins.ptr[i];
    qe = ins.ptr[i + 1];
  }
  while (p < pe || q < qe) {
    if (q >= qe || (p < pe && a.colind[p] < ins.col[q])) {
      const index_t j = a.colind[p];
      const index_t r = j == i ? p : find_entry(a, j, i);
      visit(o++, a.colind[p],
            r >= 0 ? half * a.values[p] + half * conj_if(a.values[r]) : half * a.values[p]);
      ++p;
    } else {
      visit(o++, ins.col[q], half * conj_if(a.values[find_entry(a, ins.col[q], i)]));
      ++q;
    }
  }
}

}  // namespace

template <class T>
void CsrMatrix<T>::validate() const {
  if (nrows < 0 || ncols < 0) HS_THROW(InvalidInput, "CSR: negative shape " << nrows << "x" << ncols);
  if (ncols > kMaxCol + 1) HS_THROW(InvalidInput, "CSR: ncols " << ncols << " exceeds int32 column index range");
  if (static_cast<index_t>(rowptr.size()) != nrows + 1)
    HS_THROW(InvalidInput, "CSR: rowptr size " << rowptr.size() << " != nrows+1 = " << nrows + 1);
  if (rowptr[0] != 0) HS_THROW(InvalidInput, "CSR: rowptr[0] = " << rowptr[0] << " != 0");
  if (colind.size() != values.size())
    HS_THROW(InvalidInput, "CSR: colind size " << colind.size() << " != values size " << values.size());
  if (rowptr[nrows] != nnz()) HS_THROW(InvalidInput, "CSR: rowptr[n] = " << rowptr[nrows] << " != nnz = " << nnz());
  for (index_t i = 0; i < nrows; ++i) {
    if (rowptr[i + 1] < rowptr[i]) HS_THROW(InvalidInput, "CSR: rowptr decreases at row " << i);
    for (index_t p = rowptr[i]; p < rowptr[i + 1]; ++p) {
      const index_t c = colind[p];
      if (c < 0 || c >= ncols) HS_THROW(InvalidInput, "CSR: column " << c << " out of range in row " << i);
      if (p > rowptr[i] && c <= colind[p - 1])
        HS_THROW(InvalidInput, "CSR: columns not strictly increasing in row " << i);
    }
  }
}

template <class T>
CsrMatrix<T> csr_from_coo(index_t nrows, index_t ncols, const std::vector<index_t>& rows,
                          const std::vector<index_t>& cols, const std::vector<T>& vals, bool drop_zeros) {
  if (nrows < 0 || ncols < 0) HS_THROW(InvalidInput, "csr_from_coo: negative shape");
  if (ncols > kMaxCol + 1) HS_THROW(InvalidInput, "csr_from_coo: ncols exceeds int32 range");
  if (rows.size() != cols.size() || rows.size() != vals.size())
    HS_THROW(InvalidInput, "csr_from_coo: triplet arrays have different lengths");
  const index_t nz = static_cast<index_t>(rows.size());

  // Counting sort by row.
  std::vector<index_t> rowptr(static_cast<std::size_t>(nrows) + 1, 0);
  for (index_t p = 0; p < nz; ++p) {
    const index_t r = rows[p], c = cols[p];
    if (r < 0 || r >= nrows || c < 0 || c >= ncols)
      HS_THROW(InvalidInput, "csr_from_coo: entry (" << r << "," << c << ") outside " << nrows << "x" << ncols);
    ++rowptr[r + 1];
  }
  std::partial_sum(rowptr.begin(), rowptr.end(), rowptr.begin());
  std::vector<std::pair<std::int32_t, T>> entries(static_cast<std::size_t>(nz));
  {
    std::vector<index_t> next(rowptr.begin(), rowptr.end() - 1);
    for (index_t p = 0; p < nz; ++p) {
      entries[next[rows[p]]++] = {static_cast<std::int32_t>(cols[p]), vals[p]};
    }
  }

  // Sort each row by column, merge duplicates, optionally drop zeros (in place, per row).
  std::vector<index_t> kept(static_cast<std::size_t>(nrows), 0);
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < nrows; ++i) {
    auto* b = entries.data() + rowptr[i];
    auto* e = entries.data() + rowptr[i + 1];
    std::sort(b, e, [](const auto& x, const auto& y) { return x.first < y.first; });
    index_t out = 0;
    for (auto* it = b; it != e;) {
      std::int32_t c = it->first;
      T sum = it->second;
      for (++it; it != e && it->first == c; ++it) sum += it->second;
      if (drop_zeros && is_zero(sum)) continue;
      b[out++] = {c, sum};
    }
    kept[i] = out;
  }

  CsrMatrix<T> m;
  m.nrows = nrows;
  m.ncols = ncols;
  m.rowptr.assign(static_cast<std::size_t>(nrows) + 1, 0);
  for (index_t i = 0; i < nrows; ++i) m.rowptr[i + 1] = m.rowptr[i] + kept[i];
  m.colind.resize(static_cast<std::size_t>(m.rowptr[nrows]));
  m.values.resize(static_cast<std::size_t>(m.rowptr[nrows]));
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < nrows; ++i) {
    const auto* src = entries.data() + rowptr[i];
    for (index_t k = 0; k < kept[i]; ++k) {
      m.colind[m.rowptr[i] + k] = src[k].first;
      m.values[m.rowptr[i] + k] = src[k].second;
    }
  }
  return m;
}

template <class T>
CsrMatrix<T> conj_transpose(const CsrMatrix<T>& a) {
  CsrMatrix<T> t;
  t.nrows = a.ncols;
  t.ncols = a.nrows;
  if (t.ncols > kMaxCol + 1) HS_THROW(InvalidInput, "conj_transpose: nrows exceeds int32 range");
  t.rowptr.assign(static_cast<std::size_t>(t.nrows) + 1, 0);
  for (index_t p = 0; p < a.nnz(); ++p) ++t.rowptr[a.colind[p] + 1];
  std::partial_sum(t.rowptr.begin(), t.rowptr.end(), t.rowptr.begin());
  t.colind.resize(a.colind.size());
  t.values.resize(a.values.size());
  std::vector<index_t> next(t.rowptr.begin(), t.rowptr.end() - 1);
  // Visiting rows of A in increasing order keeps the columns of A^H sorted.
  for (index_t i = 0; i < a.nrows; ++i) {
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      const index_t dst = next[a.colind[p]]++;
      t.colind[dst] = static_cast<std::int32_t>(i);
      t.values[dst] = conj_if(a.values[p]);
    }
  }
  return t;
}

template <class T>
CsrMatrix<T> add(T alpha, const CsrMatrix<T>& a, T beta, const CsrMatrix<T>& b) {
  if (a.nrows != b.nrows || a.ncols != b.ncols)
    HS_THROW(InvalidInput, "add: shape mismatch " << a.nrows << "x" << a.ncols << " vs " << b.nrows << "x" << b.ncols);
  const index_t n = a.nrows;
  CsrMatrix<T> c;
  c.nrows = n;
  c.ncols = a.ncols;
  c.rowptr.assign(static_cast<std::size_t>(n) + 1, 0);
  // Pass 1: union size per row.
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    index_t p = a.rowptr[i], q = b.rowptr[i], cnt = 0;
    const index_t pe = a.rowptr[i + 1], qe = b.rowptr[i + 1];
    while (p < pe || q < qe) {
      if (q >= qe || (p < pe && a.colind[p] < b.colind[q])) {
        ++p;
      } else if (p >= pe || b.colind[q] < a.colind[p]) {
        ++q;
      } else {
        ++p;
        ++q;
      }
      ++cnt;
    }
    c.rowptr[i + 1] = cnt;
  }
  std::partial_sum(c.rowptr.begin(), c.rowptr.end(), c.rowptr.begin());
  c.colind.resize(static_cast<std::size_t>(c.rowptr[n]));
  c.values.resize(static_cast<std::size_t>(c.rowptr[n]));
  // Pass 2: fill.
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    index_t p = a.rowptr[i], q = b.rowptr[i], o = c.rowptr[i];
    const index_t pe = a.rowptr[i + 1], qe = b.rowptr[i + 1];
    while (p < pe || q < qe) {
      if (q >= qe || (p < pe && a.colind[p] < b.colind[q])) {
        c.colind[o] = a.colind[p];
        c.values[o] = alpha * a.values[p];
        ++p;
      } else if (p >= pe || b.colind[q] < a.colind[p]) {
        c.colind[o] = b.colind[q];
        c.values[o] = beta * b.values[q];
        ++q;
      } else {
        c.colind[o] = a.colind[p];
        c.values[o] = alpha * a.values[p] + beta * b.values[q];
        ++p;
        ++q;
      }
      ++o;
    }
  }
  return c;
}

template <class T>
CsrMatrix<T> identity(index_t n) {
  if (n < 0 || n > kMaxCol + 1) HS_THROW(InvalidInput, "identity: invalid order " << n);
  CsrMatrix<T> m;
  m.nrows = m.ncols = n;
  m.rowptr.resize(static_cast<std::size_t>(n) + 1);
  m.colind.resize(static_cast<std::size_t>(n));
  m.values.assign(static_cast<std::size_t>(n), T(1));
  for (index_t i = 0; i <= n; ++i) m.rowptr[i] = i;
  for (index_t i = 0; i < n; ++i) m.colind[i] = static_cast<std::int32_t>(i);
  return m;
}

template <class T>
CsrMatrix<T> triangle(const CsrMatrix<T>& a, bool upper) {
  CsrMatrix<T> t;
  t.nrows = a.nrows;
  t.ncols = a.ncols;
  t.rowptr.assign(static_cast<std::size_t>(a.nrows) + 1, 0);
  for (index_t i = 0; i < a.nrows; ++i) {
    index_t cnt = 0;
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      const index_t c = a.colind[p];
      if (upper ? c >= i : c <= i) ++cnt;
    }
    t.rowptr[i + 1] = t.rowptr[i] + cnt;
  }
  t.colind.resize(static_cast<std::size_t>(t.rowptr[a.nrows]));
  t.values.resize(static_cast<std::size_t>(t.rowptr[a.nrows]));
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < a.nrows; ++i) {
    index_t o = t.rowptr[i];
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      const index_t c = a.colind[p];
      if (upper ? c >= i : c <= i) {
        t.colind[o] = a.colind[p];
        t.values[o] = a.values[p];
        ++o;
      }
    }
  }
  return t;
}

template <class T>
CsrMatrix<T> symmetrize_from_triangle(const CsrMatrix<T>& tri) {
  if (!tri.square()) HS_THROW(InvalidInput, "symmetrize_from_triangle: matrix is not square");
  const int side = stored_triangle(tri);
  if (side == 0) return tri;
  const bool upper = side > 0;
  CsrMatrix<T> full;
  full.nrows = full.ncols = tri.nrows;
  full.rowptr = mirrored_rowptr(tri);
  const auto nnz = static_cast<std::size_t>(full.rowptr[tri.nrows]);
  full.colind.resize(nnz);
  full.values.resize(nnz);
  copy_own(tri, full.rowptr, upper, tri.colind.data(), full.colind.data());
  copy_own(tri, full.rowptr, upper, tri.values.data(), full.values.data());
  scatter_mirrored(tri, full.rowptr, upper, [&](index_t dst, index_t p, index_t j) {
    full.colind[dst] = static_cast<std::int32_t>(j);
    full.values[dst] = conj_if(tri.values[p]);
  });
  return full;
}

template <class T>
CsrMatrix<T> symmetrize_from_triangle(CsrMatrix<T>&& tri) {
  if (!tri.square()) HS_THROW(InvalidInput, "symmetrize_from_triangle: matrix is not square");
  const int side = stored_triangle(tri);
  CsrMatrix<T> full;
  if (side == 0) {
    full = std::move(tri);
    tri = CsrMatrix<T>();
    return full;
  }
  const bool upper = side > 0;
  full.nrows = full.ncols = tri.nrows;
  full.rowptr = mirrored_rowptr(tri);
  const auto nnz = static_cast<std::size_t>(full.rowptr[tri.nrows]);
  // Values first; the stored values are released before the full pattern is allocated. This
  // order never holds both value arrays together with the full pattern: the peak is
  // colind(tri) + values(tri) + values(full), about 1.17x (real) / 1.3x (complex) the result.
  full.values.resize(nnz);
  copy_own(tri, full.rowptr, upper, tri.values.data(), full.values.data());
  scatter_mirrored(tri, full.rowptr, upper,
                   [&](index_t dst, index_t p, index_t) { full.values[dst] = conj_if(tri.values[p]); });
  release(tri.values);
  full.colind.resize(nnz);
  copy_own(tri, full.rowptr, upper, tri.colind.data(), full.colind.data());
  scatter_mirrored(tri, full.rowptr, upper, [&](index_t dst, index_t, index_t j) {
    full.colind[dst] = static_cast<std::int32_t>(j);
  });
  tri = CsrMatrix<T>();
  return full;
}

template <class T>
CsrMatrix<T> hermitian_part(const CsrMatrix<T>& a) {
  if (!a.square()) HS_THROW(InvalidInput, "hermitian_part: matrix is not square");
  const Insertions ins = orphan_mirrors(a);
  CsrMatrix<T> c;
  c.nrows = c.ncols = a.nrows;
  c.rowptr = ins.total > 0 ? union_rowptr(a, ins) : a.rowptr;
  const auto nnz = static_cast<std::size_t>(c.rowptr[a.nrows]);
  c.colind.resize(nnz);
  c.values.resize(nnz);
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < a.nrows; ++i)
    union_row(a, ins, i, c.rowptr[i], [&](index_t o, std::int32_t j, const T& v) {
      c.colind[o] = j;
      c.values[o] = v;
    });
  return c;
}

template <class T>
CsrMatrix<T> hermitian_part(CsrMatrix<T>&& a) {
  if (!a.square()) HS_THROW(InvalidInput, "hermitian_part: matrix is not square");
  const index_t n = a.nrows;
  const Insertions ins = orphan_mirrors(a);
  CsrMatrix<T> c;
  if (ins.total == 0) {
    // Structurally symmetric: update each off-diagonal pair in place, by the thread that owns
    // the row of its upper entry (no value is touched by two threads).
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
        const index_t q = find_entry(a, j, i);
        const T aij = a.values[p], aji = a.values[q];
        a.values[p] = half * aij + half * conj_if(aji);
        a.values[q] = half * aji + half * conj_if(aij);
      }
    }
    c = std::move(a);
    a = CsrMatrix<T>();
    return c;
  }
  // New entries are needed. Values first (they read the stored pattern and values), then the
  // stored values are released and the union pattern is built: peak = A + values(result),
  // about 1.7x (real) / 1.8x (complex) the result.
  c.nrows = c.ncols = n;
  c.rowptr = union_rowptr(a, ins);
  const auto nnz = static_cast<std::size_t>(c.rowptr[n]);
  c.values.resize(nnz);
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i)
    union_row(a, ins, i, c.rowptr[i], [&](index_t o, std::int32_t, const T& v) { c.values[o] = v; });
  release(a.values);
  c.colind.resize(nnz);
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    index_t p = a.rowptr[i], q = ins.ptr[i], o = c.rowptr[i];
    const index_t pe = a.rowptr[i + 1], qe = ins.ptr[i + 1];
    while (p < pe || q < qe)
      c.colind[o++] = (q >= qe || (p < pe && a.colind[p] < ins.col[q])) ? a.colind[p++] : ins.col[q++];
  }
  a = CsrMatrix<T>();
  return c;
}

CsrMatrix<cplx> to_complex(const CsrMatrix<double>& a) {
  CsrMatrix<cplx> c;
  c.nrows = a.nrows;
  c.ncols = a.ncols;
  c.rowptr = a.rowptr;
  c.colind = a.colind;
  c.values.assign(a.values.begin(), a.values.end());
  return c;
}

CsrMatrix<cplx> to_complex(CsrMatrix<double>&& a) {
  CsrMatrix<cplx> c;
  c.nrows = a.nrows;
  c.ncols = a.ncols;
  c.rowptr = std::move(a.rowptr);
  c.colind = std::move(a.colind);
  c.values.assign(a.values.begin(), a.values.end());
  a = CsrMatrix<double>();
  return c;
}

template <class T>
double norm_fro(const CsrMatrix<T>& a) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (index_t p = 0; p < a.nnz(); ++p) s += static_cast<double>(abs2(a.values[p]));
  return std::sqrt(s);
}

template <class T>
double norm_one(const CsrMatrix<T>& a) {
  std::vector<double> colsum(static_cast<std::size_t>(a.ncols), 0.0);
  for (index_t p = 0; p < a.nnz(); ++p) colsum[a.colind[p]] += std::abs(a.values[p]);
  return colsum.empty() ? 0.0 : *std::max_element(colsum.begin(), colsum.end());
}

template <class T>
double hermitian_defect(const CsrMatrix<T>& a) {
  if (!a.square()) HS_THROW(InvalidInput, "hermitian_defect: matrix is not square");
  const double nrm = norm_fro(a);
  if (nrm == 0.0) return 0.0;
  // ||A - A^H||_F^2 without forming A^H: for each stored (i,j), look up (j,i) by binary search.
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(dynamic, 512)
  for (index_t i = 0; i < a.nrows; ++i) {
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      const index_t j = a.colind[p];
      const auto* b = a.colind.data() + a.rowptr[j];
      const auto* e = a.colind.data() + a.rowptr[j + 1];
      const auto* it = std::lower_bound(b, e, static_cast<std::int32_t>(i));
      if (it != e && *it == i) {
        // Visited from both (i,j) and (j,i): each contributes its own entry of A - A^H.
        s += static_cast<double>(abs2(a.values[p] - conj_if(a.values[a.rowptr[j] + (it - b)])));
      } else {
        // (j,i) is structurally zero: entries (i,j) and (j,i) of A - A^H both have modulus |a_ij|.
        s += 2.0 * static_cast<double>(abs2(a.values[p]));
      }
    }
  }
  return std::sqrt(s) / nrm;
}

CsrMatrix<double> real_embedding(const CsrMatrix<cplx>& k) {
  if (!k.square()) HS_THROW(InvalidInput, "real_embedding: matrix is not square");
  const index_t n = k.nrows;
  if (2 * n > kMaxCol + 1) HS_THROW(InvalidInput, "real_embedding: order 2n exceeds int32 range");
  CsrMatrix<double> e;
  e.nrows = e.ncols = 2 * n;
  e.rowptr.assign(static_cast<std::size_t>(2 * n) + 1, 0);
  // Row i: [Re K_i, -Im K_i]; row n+i: [Im K_i, Re K_i]. Exact zeros are dropped symmetrically.
  std::vector<index_t> cnt_re(static_cast<std::size_t>(n)), cnt_im(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) {
    index_t r = 0, m = 0;
    for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p) {
      r += k.values[p].real() != 0.0;
      m += k.values[p].imag() != 0.0;
    }
    cnt_re[i] = r;
    cnt_im[i] = m;
  }
  for (index_t i = 0; i < n; ++i) e.rowptr[i + 1] = e.rowptr[i] + cnt_re[i] + cnt_im[i];
  for (index_t i = 0; i < n; ++i) e.rowptr[n + i + 1] = e.rowptr[n + i] + cnt_im[i] + cnt_re[i];
  e.colind.resize(static_cast<std::size_t>(e.rowptr[2 * n]));
  e.values.resize(static_cast<std::size_t>(e.rowptr[2 * n]));
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    index_t o = e.rowptr[i];
    for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p) {
      if (k.values[p].real() != 0.0) {
        e.colind[o] = k.colind[p];
        e.values[o++] = k.values[p].real();
      }
    }
    for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p) {
      if (k.values[p].imag() != 0.0) {
        e.colind[o] = static_cast<std::int32_t>(n + k.colind[p]);
        e.values[o++] = -k.values[p].imag();
      }
    }
    o = e.rowptr[n + i];
    for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p) {
      if (k.values[p].imag() != 0.0) {
        e.colind[o] = k.colind[p];
        e.values[o++] = k.values[p].imag();
      }
    }
    for (index_t p = k.rowptr[i]; p < k.rowptr[i + 1]; ++p) {
      if (k.values[p].real() != 0.0) {
        e.colind[o] = static_cast<std::int32_t>(n + k.colind[p]);
        e.values[o++] = k.values[p].real();
      }
    }
  }
  return e;
}

#define HS_INSTANTIATE_CSR(T)                                                                      \
  template struct CsrMatrix<T>;                                                                    \
  template CsrMatrix<T> csr_from_coo<T>(index_t, index_t, const std::vector<index_t>&,             \
                                        const std::vector<index_t>&, const std::vector<T>&, bool); \
  template CsrMatrix<T> conj_transpose<T>(const CsrMatrix<T>&);                                    \
  template CsrMatrix<T> add<T>(T, const CsrMatrix<T>&, T, const CsrMatrix<T>&);                    \
  template CsrMatrix<T> identity<T>(index_t);                                                      \
  template CsrMatrix<T> triangle<T>(const CsrMatrix<T>&, bool);                                    \
  template CsrMatrix<T> symmetrize_from_triangle<T>(const CsrMatrix<T>&);                          \
  template CsrMatrix<T> symmetrize_from_triangle<T>(CsrMatrix<T>&&);                               \
  template CsrMatrix<T> hermitian_part<T>(const CsrMatrix<T>&);                                    \
  template CsrMatrix<T> hermitian_part<T>(CsrMatrix<T>&&);                                         \
  template double norm_fro<T>(const CsrMatrix<T>&);                                                \
  template double norm_one<T>(const CsrMatrix<T>&);                                                \
  template double hermitian_defect<T>(const CsrMatrix<T>&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_CSR)
#undef HS_INSTANTIATE_CSR

}  // namespace hs
