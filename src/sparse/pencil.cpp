// ShiftedPencil: index-only upper-triangle union pattern of A and B plus one flag byte per row;
// assemble() merges the sorted upper rows of A and B into the pattern rows (no scatter maps).
#include <algorithm>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"

namespace hs {

namespace {

template <class U, class T>
U convert_scalar(const T& x) {
  if constexpr (is_complex_v<T>) {
    return U(static_cast<real_t<U>>(x.real()), static_cast<real_t<U>>(x.imag()));
  } else {
    return static_cast<U>(x);
  }
}

// Index of the first entry of row i of m with column >= i.
template <class T>
index_t first_upper(const CsrMatrix<T>& m, index_t i) {
  const auto* b = m.colind.data() + m.rowptr[i];
  const auto* e = m.colind.data() + m.rowptr[i + 1];
  return m.rowptr[i] + (std::lower_bound(b, e, static_cast<std::int32_t>(i)) - b);
}

// True when the upper part of row i of m is the whole pattern row (length len): the pattern row
// starts with the diagonal and contains m's upper row, so it suffices that the entry len places
// before the end of m's row is the diagonal. O(1).
template <class T>
bool upper_row_is_pattern(const CsrMatrix<T>& m, index_t i, index_t len) {
  const index_t p0 = m.rowptr[i + 1] - len;
  return p0 >= m.rowptr[i] && m.colind[p0] == i;
}

// Adds scale * m(i, j) for the upper entries of row i of m into acc, where acc[k] belongs to
// pattern column pcol[k], k < len. `whole`: upper_row_is_pattern (the common case for A, and for
// B when B's pattern contains A's), a plain copy. Otherwise both rows are sorted and the pattern
// row contains every upper column of m, so one forward merge finds every slot (no stored scatter
// map); it is branch-free, as rows of m that are much sparser than the pattern are its use case.
template <class T>
void add_upper_row(const CsrMatrix<T>& m, index_t i, const T& scale, const std::int32_t* pcol, index_t len,
                   bool whole, T* acc) {
  const index_t last = m.rowptr[i + 1];
  const T* v = m.values.data();
  if (whole) {
    for (index_t k = 0, p0 = last - len; k < len; ++k) acc[k] += scale * v[p0 + k];
    return;
  }
  const std::int32_t* col = m.colind.data();
  for (index_t p = first_upper(m, i), o = 0; p < last; ++o) {
    const bool hit = pcol[o] == col[p];
    const T sum = acc[o] + scale * v[p];
    acc[o] = hit ? sum : acc[o];
    p += hit;
  }
}

}  // namespace

template <class T>
ShiftedPencil<T>::ShiftedPencil(const CsrMatrix<T>& a, const CsrMatrix<T>* b) : a_(&a), b_(b) {
  if (!a.square()) HS_THROW(InvalidInput, "ShiftedPencil: A is not square");
  if (b && (b->nrows != a.nrows || b->ncols != a.ncols))
    HS_THROW(InvalidInput, "ShiftedPencil: B is " << b->nrows << "x" << b->ncols << ", A is " << a.nrows << "x"
                                                  << a.ncols);
  n_ = a.nrows;
  rowptr_.assign(static_cast<std::size_t>(n_) + 1, 0);

  // Pass 1: size of each upper union row (A upper ∪ B upper ∪ {diagonal}).
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n_; ++i) {
    const index_t pa0 = first_upper(a, i), pa1 = a.rowptr[i + 1];
    index_t pb0 = 0, pb1 = 0;
    if (b) {
      pb0 = first_upper(*b, i);
      pb1 = b->rowptr[i + 1];
    }
    index_t cnt = 0, p = pa0, q = pb0;
    std::int32_t last = -1;
    bool diag = false;
    while (p < pa1 || q < pb1) {
      std::int32_t c;
      if (q >= pb1 || (p < pa1 && a.colind[p] <= b->colind[q])) {
        c = a.colind[p++];
      } else {
        c = b->colind[q++];
      }
      if (c != last) {
        ++cnt;
        last = c;
        diag |= (c == i);
      }
    }
    if (!diag) ++cnt;
    rowptr_[i + 1] = cnt;
  }
  for (index_t i = 0; i < n_; ++i) rowptr_[i + 1] += rowptr_[i];
  colind_.resize(static_cast<std::size_t>(rowptr_[n_]));

  // Pass 2: fill the pattern (diagonal first, since columns >= i are sorted and include i).
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n_; ++i) {
    index_t o = rowptr_[i];
    colind_[o++] = static_cast<std::int32_t>(i);
    index_t p = first_upper(a, i), q = b ? first_upper(*b, i) : 0;
    const index_t pa1 = a.rowptr[i + 1], pb1 = b ? b->rowptr[i + 1] : 0;
    std::int32_t last = static_cast<std::int32_t>(i);
    while (p < pa1 || q < pb1) {
      std::int32_t c;
      if (q >= pb1 || (p < pa1 && a.colind[p] <= b->colind[q])) {
        c = a.colind[p++];
      } else {
        c = b->colind[q++];
      }
      if (c != last) {
        colind_[o++] = c;
        last = c;
      }
    }
    HS_ASSERT(o == rowptr_[i + 1], "pattern fill mismatch in row " << i);
  }

  // Rows whose upper part of A (bit 0) or B (bit 1) is the whole pattern row: assembled by a
  // plain copy, without reading the column indices of A or B.
  whole_rows_.resize(static_cast<std::size_t>(n_));
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n_; ++i) {
    const index_t len = rowptr_[i + 1] - rowptr_[i];
    whole_rows_[i] = static_cast<std::uint8_t>((upper_row_is_pattern(a, i, len) ? 1 : 0) |
                                               (b && upper_row_is_pattern(*b, i, len) ? 2 : 0));
  }
}

template <class T>
template <class U>
void ShiftedPencil<T>::assemble(double alpha, double beta, U* vals) const {
  const T ta = T(alpha), tb = T(beta);
#pragma omp parallel
  {
    std::vector<T> acc;  // row accumulator in full precision; rounded once to U
#pragma omp for schedule(dynamic, 512)
    for (index_t i = 0; i < n_; ++i) {
      const index_t r0 = rowptr_[i], len = rowptr_[i + 1] - r0;
      const std::int32_t* pcol = colind_.data() + r0;
      acc.assign(static_cast<std::size_t>(len), T(0));
      const std::uint8_t whole = whole_rows_[i];
      if (alpha != 0.0) add_upper_row(*a_, i, ta, pcol, len, (whole & 1) != 0, acc.data());
      if (beta != 0.0) {
        if (b_) {
          add_upper_row(*b_, i, tb, pcol, len, (whole & 2) != 0, acc.data());
        } else {
          acc[0] += tb;  // identity: the diagonal is the first upper entry
        }
      }
      for (index_t k = 0; k < len; ++k) vals[r0 + k] = convert_scalar<U>(acc[k]);
    }
  }
}

template <class T>
CsrMatrix<T> ShiftedPencil<T>::assemble_full(double alpha, double beta) const {
  if (b_) return add(T(alpha), *a_, T(beta), *b_);
  return add(T(alpha), *a_, T(beta), identity<T>(n_));
}

#define HS_INSTANTIATE_PENCIL(T)                                                           \
  template class ShiftedPencil<T>;                                                         \
  template void ShiftedPencil<T>::assemble<T>(double, double, T*) const;                   \
  template void ShiftedPencil<T>::assemble<rebind_t<T, float>>(double, double, rebind_t<T, float>*) const;

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_PENCIL)
#undef HS_INSTANTIATE_PENCIL

}  // namespace hs
