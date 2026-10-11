// Sparse matrix-vector kernels and the shifted pencil K(sigma) = A - sigma*B.
// Reference (portable, OpenMP) implementations; SIMD-specialised versions come in P3 and must
// match these to 1e-14 relative accuracy.
#include <algorithm>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"

namespace hs {

template <class T>
void spmv(const CsrMatrix<T>& a, const T* x, T* y, T alpha, T beta) {
  const index_t n = a.nrows;
  const index_t* rp = a.rowptr.data();
  const std::int32_t* ci = a.colind.data();
  const T* v = a.values.data();
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    T sum = T(0);
    for (index_t p = rp[i]; p < rp[i + 1]; ++p) sum += v[p] * x[ci[p]];
    // beta == 0 must not read y (it may hold garbage/NaN).
    y[i] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i];
  }
}

template <class T>
void spmm(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p, T alpha,
          T beta) {
  if (p == 1) {
    spmv(a, x, y, alpha, beta);
    return;
  }
  const index_t n = a.nrows;
  const index_t* rp = a.rowptr.data();
  const std::int32_t* ci = a.colind.data();
  const T* v = a.values.data();
  constexpr index_t kBlock = 8;  // vectors processed together per row pass
#pragma omp parallel
  {
    T acc[kBlock];
#pragma omp for schedule(static)
    for (index_t i = 0; i < n; ++i) {
      for (index_t k0 = 0; k0 < p; k0 += kBlock) {
        const index_t kb = std::min(kBlock, p - k0);
        for (index_t k = 0; k < kb; ++k) acc[k] = T(0);
        for (index_t q = rp[i]; q < rp[i + 1]; ++q) {
          const T aq = v[q];
          const T* xc = x + ci[q] + k0 * ldx;
          for (index_t k = 0; k < kb; ++k) acc[k] += aq * xc[k * ldx];
        }
        for (index_t k = 0; k < kb; ++k) {
          T& yk = y[i + (k0 + k) * ldy];
          yk = (beta == T(0)) ? alpha * acc[k] : alpha * acc[k] + beta * yk;
        }
      }
    }
  }
}

template <class T>
ShiftedPencil<T>::ShiftedPencil(const CsrMatrix<T>& a, const CsrMatrix<T>* b) : a_(&a), b_(b) {
  if (!a.square()) HS_THROW(InvalidInput, "ShiftedPencil: A is not square");
  if (b && (b->nrows != a.nrows || b->ncols != a.ncols))
    HS_THROW(InvalidInput, "ShiftedPencil: B has a different shape than A");
  const CsrMatrix<T> eye = b ? CsrMatrix<T>{} : identity<T>(a.nrows);
  const CsrMatrix<T>& bb = b ? *b : eye;
  pattern_ = add(T(1), a, T(1), bb);
  std::fill(pattern_.values.begin(), pattern_.values.end(), T(0));

  // Scatter maps: for each stored entry of A (resp. B or I), its position in the union pattern.
  a_pos_.resize(static_cast<std::size_t>(a.nnz()));
  b_pos_.resize(static_cast<std::size_t>(bb.nnz()));
  const index_t n = a.nrows;
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    const index_t start = pattern_.rowptr[i], end = pattern_.rowptr[i + 1];
    index_t o = start;
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) {
      while (pattern_.colind[o] != a.colind[p]) ++o;
      a_pos_[p] = o;
    }
    o = start;
    for (index_t p = bb.rowptr[i]; p < bb.rowptr[i + 1]; ++p) {
      while (pattern_.colind[o] != bb.colind[p]) ++o;
      b_pos_[p] = o;
    }
    HS_ASSERT(o <= end, "scatter map overflow in row " << i);
  }
}

template <class T>
void ShiftedPencil<T>::assemble(double sigma, CsrMatrix<T>& k) const {
  if (k.nrows != pattern_.nrows || k.nnz() != pattern_.nnz()) {
    k = pattern_;
  } else {
    std::fill(k.values.begin(), k.values.end(), T(0));
  }
  const index_t na = a_->nnz();
  const T* av = a_->values.data();
  T* kv = k.values.data();
  // a_pos_ is injective, so the scatter has no write conflicts; same for b_pos_.
#pragma omp parallel for schedule(static)
  for (index_t p = 0; p < na; ++p) kv[a_pos_[p]] += av[p];
  const index_t nb = static_cast<index_t>(b_pos_.size());
  if (b_) {
    const T* bv = b_->values.data();
#pragma omp parallel for schedule(static)
    for (index_t p = 0; p < nb; ++p) kv[b_pos_[p]] -= sigma * bv[p];
  } else {
#pragma omp parallel for schedule(static)
    for (index_t p = 0; p < nb; ++p) kv[b_pos_[p]] -= T(sigma);
  }
}

#define HS_INSTANTIATE_KERNELS(T)                                                         \
  template void spmv<T>(const CsrMatrix<T>&, const T*, T*, T, T);                         \
  template void spmm<T>(const CsrMatrix<T>&, const T*, index_t, T*, index_t, index_t, T, T); \
  template class ShiftedPencil<T>;

HS_INSTANTIATE_KERNELS(double)
HS_INSTANTIATE_KERNELS(cplx)
#undef HS_INSTANTIATE_KERNELS

}  // namespace hs
