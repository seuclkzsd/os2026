#include "linalg/dense_ops.hpp"

#include <algorithm>
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

// ---- deterministic reductions -----------------------------------------------------------------
// Rows are cut into blocks of a fixed size, consecutive blocks are grouped into at most kMaxSlots
// slots, and each slot is reduced by one thread in block order into its own accumulator. The slot
// accumulators are then added serially in slot order. Every boundary depends only on the sizes,
// never on the thread count, so the result is bitwise reproducible for any number of threads.
// Within a block the sums use kLanes explicit accumulators (vectorizable without reassociation:
// the compiler may not reorder floating-point additions on its own).

constexpr int kLanes = 8;
constexpr index_t kMaxSlots = 256;
constexpr index_t kVecBlock = 1024;  ///< rows per block of dot / norm2
/// Below this many scalars of reduction work a call runs on one thread (results are unchanged).
constexpr index_t kParallelMinWork = 32768;

inline double lane_sum(const double* a) {
  return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

/// sum_t x[t] * y[t] over len doubles: Re(x^H y) for interleaved complex data as well.
inline double ddot(index_t len, const double* x, const double* y) {
  double a[kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int l = 0; l < kLanes; ++l) a[l] += x[t + l] * y[t + l];
  for (int l = 0; t + l < len; ++l) a[l] += x[t + l] * y[t + l];
  return lane_sum(a);
}

/// x^H y over len complex entries stored as interleaved doubles: re = sum_t x[t] y[t] and
/// im = sum_i (x_re y_im - x_im y_re), the latter from lane products x[t] * y[t ^ 1] with the odd
/// lanes subtracted (len * 2 is even, so the remainder holds whole pairs).
inline void zdotc(index_t len, const double* x, const double* y, double& re, double& im) {
  const index_t m = 2 * len;
  double a[kLanes] = {}, b[kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= m; t += kLanes)
    for (int l = 0; l < kLanes; ++l) {
      a[l] += x[t + l] * y[t + l];
      b[l] += x[t + l] * y[t + (l ^ 1)];
    }
  for (int l = 0; t + l < m; ++l) {
    a[l] += x[t + l] * y[t + l];
    b[l] += x[t + l] * y[t + (l ^ 1)];
  }
  re = lane_sum(a);
  im = ((b[0] - b[1]) + (b[4] - b[5])) + ((b[2] - b[3]) + (b[6] - b[7]));
}

/// Fixed partition of [0, n) into blocks of `block` rows grouped into slots.
struct Partition {
  index_t n = 0, block = 1, nblocks = 0, nslots = 0;
  Partition(index_t rows, index_t block_rows, index_t max_slots)
      : n(rows), block(std::max<index_t>(block_rows, 1)) {
    nblocks = (n + block - 1) / block;
    nslots = std::min(nblocks, std::max<index_t>(max_slots, 1));
  }
  index_t first_block(index_t s) const { return s * nblocks / nslots; }
  index_t row(index_t b) const { return std::min(b * block, n); }
};

/// out[0, width) = sum over slots of the per-slot accumulators filled by
/// block_fn(row0, rows, acc) (acc: the slot's `width` doubles, zero-initialised; blocks of one
/// slot arrive in order on one thread). `work` (scalars touched) only decides whether to go
/// parallel. Deterministic for any thread count.
template <class BlockFn>
void reduce_blocks(const Partition& part, index_t width, index_t work, double* out,
                   BlockFn&& block_fn) {
  std::fill(out, out + width, 0.0);
  if (part.nblocks == 0) return;
  if (part.nslots == 1) {
    for (index_t b = 0; b < part.nblocks; ++b)
      block_fn(part.row(b), part.row(b + 1) - part.row(b), out);
    return;
  }
  std::vector<double> acc(static_cast<std::size_t>(part.nslots * width), 0.0);
#pragma omp parallel for schedule(static) if (work >= kParallelMinWork)
  for (index_t s = 0; s < part.nslots; ++s) {
    double* slot = acc.data() + s * width;
    for (index_t b = part.first_block(s); b < part.first_block(s + 1); ++b)
      block_fn(part.row(b), part.row(b + 1) - part.row(b), slot);
  }
  for (index_t s = 0; s < part.nslots; ++s)
    for (index_t j = 0; j < width; ++j) out[j] += acc[static_cast<std::size_t>(s * width + j)];
}

/// Scalars viewed as doubles (std::complex<double> is layout-compatible with double[2]).
template <class T>
const double* dbl(const T* p) {
  return reinterpret_cast<const double*>(p);
}
template <class T>
double* dbl(T* p) {
  return reinterpret_cast<double*>(p);
}
template <class T>
constexpr index_t kDoubles = is_complex_v<T> ? 2 : 1;

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
  const Partition part(n, kVecBlock, kMaxSlots);
  const double* xd = dbl(x);
  const double* yd = dbl(y);
  if constexpr (is_complex_v<T>) {
    double s[2];
    reduce_blocks(part, 2, 2 * n, s, [&](index_t r0, index_t rows, double* acc) {
      double re, im;
      zdotc(rows, xd + 2 * r0, yd + 2 * r0, re, im);
      acc[0] += re;
      acc[1] += im;
    });
    return T(s[0], s[1]);
  } else {
    double s;
    reduce_blocks(part, 1, n, &s,
                  [&](index_t r0, index_t rows, double* acc) { acc[0] += ddot(rows, xd + r0, yd + r0); });
    return s;
  }
}

template <class T>
double norm2(index_t n, const T* x) {
  const Partition part(n, kVecBlock, kMaxSlots);
  const double* xd = dbl(x);
  constexpr index_t d = kDoubles<T>;
  double s;
  reduce_blocks(part, 1, d * n, &s, [&](index_t r0, index_t rows, double* acc) {
    acc[0] += ddot(d * rows, xd + d * r0, xd + d * r0);
  });
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
                          T* w, T* bw, T* h, const ApplyFn<T>* apply_m) {
  const bool stored = bv != nullptr;
  const bool recompute = !stored && apply_m != nullptr;
  HS_ASSERT(!stored || bw != nullptr, "stored mode needs bw = M*w");
  HS_ASSERT(!recompute || bw != nullptr, "recompute mode needs a bw workspace");
  HS_ASSERT(stored || recompute || bw == nullptr, "bw given without bv or apply_m");
  const auto nn = static_cast<lapack::fint>(n);
  const auto kk = static_cast<lapack::fint>(k);
  std::vector<T> c(static_cast<std::size_t>(k));
  if (h) {
    for (index_t j = 0; j < k; ++j) h[j] = T(0);
  }
  for (int pass = 0; pass < 2 && k > 0; ++pass) {
    if (recompute) (*apply_m)(w, bw);
    // c = V^H M w: (M V)^H w in stored mode, V^H (M w) in recompute mode, V^H w otherwise.
    if (stored) {
      lapack::gemv('C', nn, kk, T(1), bv, static_cast<lapack::fint>(ldbv), w, T(0), c.data());
    } else {
      lapack::gemv('C', nn, kk, T(1), v, static_cast<lapack::fint>(ldv), recompute ? bw : w, T(0), c.data());
    }
    lapack::gemv('N', nn, kk, T(-1), v, static_cast<lapack::fint>(ldv), c.data(), T(1), w);
    if (stored) lapack::gemv('N', nn, kk, T(-1), bv, static_cast<lapack::fint>(ldbv), c.data(), T(1), bw);
    if (h) {
      for (index_t j = 0; j < k; ++j) h[j] += c[j];
    }
  }
  if (recompute) (*apply_m)(w, bw);
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
                                        T*, T*, const ApplyFn<T>*);                                 \
  template void gram<T>(index_t, index_t, const T*, index_t, const T*, index_t, T*, index_t);       \
  template void multiply<T>(index_t, index_t, index_t, const T*, index_t, const T*, index_t, T*,    \
                            index_t);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DENSE_OPS)
#undef HS_INSTANTIATE_DENSE_OPS

}  // namespace hs::linalg
