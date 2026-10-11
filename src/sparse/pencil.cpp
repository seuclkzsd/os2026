// ShiftedPencil: index-only upper-triangle union pattern of A and B with scatter maps.
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

}  // namespace

template <class T>
ShiftedPencil<T>::ShiftedPencil(const CsrMatrix<T>& a, const CsrMatrix<T>* b) : a_(&a), b_(b) {
  if (!a.square()) HS_THROW(InvalidInput, "ShiftedPencil: A is not square");
  if (b && (b->nrows != a.nrows || b->ncols != a.ncols))
    HS_THROW(InvalidInput, "ShiftedPencil: B is " << b->nrows << "x" << b->ncols << ", A is " << a.nrows << "x"
                                                  << a.ncols);
  n_ = a.nrows;
  rowptr_.assign(static_cast<std::size_t>(n_) + 1, 0);
  a_first_upper_.resize(static_cast<std::size_t>(n_));
  if (b) b_first_upper_.resize(static_cast<std::size_t>(n_));

  // Pass 1: size of each upper union row (A upper ∪ B upper ∪ {diagonal}).
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n_; ++i) {
    const index_t pa0 = first_upper(a, i), pa1 = a.rowptr[i + 1];
    a_first_upper_[i] = pa0;
    index_t pb0 = 0, pb1 = 0;
    if (b) {
      pb0 = first_upper(*b, i);
      pb1 = b->rowptr[i + 1];
      b_first_upper_[i] = pb0;
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
    index_t p = a_first_upper_[i], q = b ? b_first_upper_[i] : 0;
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

  // Scatter maps. A: identity when upper(A) already equals the union pattern row by row.
  bool identity = true;
  for (index_t i = 0; i < n_ && identity; ++i)
    identity = (a.rowptr[i + 1] - a_first_upper_[i]) == (rowptr_[i + 1] - rowptr_[i]);
  a_identity_ = identity;
  auto build_map = [this](const CsrMatrix<T>& m, const std::vector<index_t>& first,
                          std::vector<index_t>& dst) {
    std::vector<index_t> ptr(static_cast<std::size_t>(n_) + 1, 0);
    for (index_t i = 0; i < n_; ++i) ptr[i + 1] = ptr[i] + (m.rowptr[i + 1] - first[i]);
    dst.resize(static_cast<std::size_t>(ptr[n_]));
#pragma omp parallel for schedule(dynamic, 512)
    for (index_t i = 0; i < n_; ++i) {
      index_t o = rowptr_[i];
      for (index_t p = first[i]; p < m.rowptr[i + 1]; ++p) {
        while (colind_[o] != m.colind[p]) ++o;
        dst[ptr[i] + (p - first[i])] = o;
      }
    }
  };
  if (!a_identity_) build_map(a, a_first_upper_, a_dst_);
  if (b) build_map(*b, b_first_upper_, b_dst_);
}

template <class T>
template <class U>
void ShiftedPencil<T>::assemble(double alpha, double beta, U* vals) const {
  const CsrMatrix<T>& a = *a_;
  // Prefix offsets into a_dst_/b_dst_ (cheap: n additions).
  std::vector<index_t> aptr, bptr;
  if (!a_identity_) {
    aptr.assign(static_cast<std::size_t>(n_) + 1, 0);
    for (index_t i = 0; i < n_; ++i) aptr[i + 1] = aptr[i] + (a.rowptr[i + 1] - a_first_upper_[i]);
  }
  if (b_) {
    bptr.assign(static_cast<std::size_t>(n_) + 1, 0);
    for (index_t i = 0; i < n_; ++i) bptr[i + 1] = bptr[i] + (b_->rowptr[i + 1] - b_first_upper_[i]);
  }
  const T ta = T(alpha), tb = T(beta);
#pragma omp parallel
  {
    std::vector<T> acc;  // row accumulator in full precision; rounded once to U
#pragma omp for schedule(dynamic, 512)
    for (index_t i = 0; i < n_; ++i) {
      const index_t r0 = rowptr_[i], len = rowptr_[i + 1] - r0;
      acc.assign(static_cast<std::size_t>(len), T(0));
      if (alpha != 0.0) {
        for (index_t p = a_first_upper_[i]; p < a.rowptr[i + 1]; ++p) {
          const index_t dst = a_identity_ ? r0 + (p - a_first_upper_[i]) : a_dst_[aptr[i] + (p - a_first_upper_[i])];
          acc[dst - r0] += ta * a.values[p];
        }
      }
      if (beta != 0.0) {
        if (b_) {
          for (index_t p = b_first_upper_[i]; p < b_->rowptr[i + 1]; ++p)
            acc[b_dst_[bptr[i] + (p - b_first_upper_[i])] - r0] += tb * b_->values[p];
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
