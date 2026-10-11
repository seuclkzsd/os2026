#include "linalg/dense_ops.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"

namespace hs::linalg {

namespace {

// SplitMix64: counter-based, so entry i is a pure function of (seed, stream, i) and the fill
// can be parallelised without changing the sequence.
inline std::uint64_t splitmix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline double uniform_pm1(std::uint64_t key) {
  // 53 random bits -> [0, 1) -> [-1, 1)
  return static_cast<double>(splitmix64(key) >> 11) * 0x1.0p-53 * 2.0 - 1.0;
}

}  // namespace

template <class T>
void fill_random(T* x, index_t n, std::uint64_t seed, std::uint64_t stream) {
  const std::uint64_t base = splitmix64(seed ^ splitmix64(stream + 0x51ED270B27A1F3ULL));
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    const std::uint64_t k = base + 2 * static_cast<std::uint64_t>(i);
    if constexpr (is_complex_v<T>) {
      x[i] = T(uniform_pm1(k), uniform_pm1(k + 1));
    } else {
      x[i] = uniform_pm1(k);
    }
  }
}

template <class T>
T dot(index_t n, const T* x, const T* y) {
  if constexpr (is_complex_v<T>) {
    double re = 0.0, im = 0.0;
#pragma omp parallel for reduction(+ : re, im) schedule(static)
    for (index_t i = 0; i < n; ++i) {
      const T p = std::conj(x[i]) * y[i];
      re += p.real();
      im += p.imag();
    }
    return T(re, im);
  } else {
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (index_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
  }
}

template <class T>
double norm2(index_t n, const T* x) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (index_t i = 0; i < n; ++i) s += abs2(x[i]);
  return std::sqrt(s);
}

template <class T>
void axpy(index_t n, T alpha, const T* x, T* y) {
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

template <class T>
void scal(index_t n, T alpha, T* x) {
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) x[i] *= alpha;
}

template <class T>
double orthogonalize_cgs2(index_t n, index_t k, const T* v, index_t ldv, const T* bv, index_t ldbv,
                          T* w, T* bw, T* h) {
  HS_ASSERT((bv == nullptr) == (bw == nullptr), "bv and bw must be given together");
  const auto nn = static_cast<lapack::fint>(n);
  const auto kk = static_cast<lapack::fint>(k);
  std::vector<T> c(static_cast<std::size_t>(k));
  if (h) {
    for (index_t j = 0; j < k; ++j) h[j] = T(0);
  }
  for (int pass = 0; pass < 2 && k > 0; ++pass) {
    // c = (M V)^H w  ==  V^H M w   (M Hermitian)
    const T* proj = bv ? bv : v;
    const index_t ldp = bv ? ldbv : ldv;
    lapack::gemv('C', nn, kk, T(1), proj, static_cast<lapack::fint>(ldp), w, T(0), c.data());
    // w -= V c ; bw -= (M V) c
    lapack::gemv('N', nn, kk, T(-1), v, static_cast<lapack::fint>(ldv), c.data(), T(1), w);
    if (bv) lapack::gemv('N', nn, kk, T(-1), bv, static_cast<lapack::fint>(ldbv), c.data(), T(1), bw);
    if (h) {
      for (index_t j = 0; j < k; ++j) h[j] += c[j];
    }
  }
  if (bw) {
    const double m2 = std::real(dot(n, w, bw));
    return m2 > 0.0 ? std::sqrt(m2) : 0.0;
  }
  return norm2(n, w);
}

template <class T>
void gram(index_t n, index_t k, const T* v, index_t ldv, const T* w, index_t ldw, T* c, index_t ldc) {
  lapack::gemm('C', 'N', static_cast<lapack::fint>(k), static_cast<lapack::fint>(k),
               static_cast<lapack::fint>(n), T(1), v, static_cast<lapack::fint>(ldv), w,
               static_cast<lapack::fint>(ldw), T(0), c, static_cast<lapack::fint>(ldc));
}

template <class T>
void multiply(index_t n, index_t k, index_t m, const T* v, index_t ldv, const T* s, index_t lds, T* y,
              index_t ldy) {
  lapack::gemm('N', 'N', static_cast<lapack::fint>(n), static_cast<lapack::fint>(m),
               static_cast<lapack::fint>(k), T(1), v, static_cast<lapack::fint>(ldv), s,
               static_cast<lapack::fint>(lds), T(0), y, static_cast<lapack::fint>(ldy));
}

#define HS_INSTANTIATE_DENSE_OPS(T)                                                                 \
  template void fill_random<T>(T*, index_t, std::uint64_t, std::uint64_t);                          \
  template T dot<T>(index_t, const T*, const T*);                                                   \
  template double norm2<T>(index_t, const T*);                                                      \
  template void axpy<T>(index_t, T, const T*, T*);                                                  \
  template void scal<T>(index_t, T, T*);                                                            \
  template double orthogonalize_cgs2<T>(index_t, index_t, const T*, index_t, const T*, index_t, T*, \
                                        T*, T*);                                                    \
  template void gram<T>(index_t, index_t, const T*, index_t, const T*, index_t, T*, index_t);       \
  template void multiply<T>(index_t, index_t, index_t, const T*, index_t, const T*, index_t, T*,    \
                            index_t);

HS_INSTANTIATE_DENSE_OPS(double)
HS_INSTANTIATE_DENSE_OPS(cplx)
#undef HS_INSTANTIATE_DENSE_OPS

}  // namespace hs::linalg
