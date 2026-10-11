// Sparse matrix-vector kernels.
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

#define HS_INSTANTIATE_KERNELS(T)                         \
  template void spmv<T>(const CsrMatrix<T>&, const T*, T*, T, T); \
  template void spmm<T>(const CsrMatrix<T>&, const T*, index_t, T*, index_t, index_t, T, T);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_KERNELS)
#undef HS_INSTANTIATE_KERNELS

}  // namespace hs
