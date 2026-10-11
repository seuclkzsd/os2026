// Follow-up measurements of src/linalg/dense_ops (review of the ortho module). Every section runs
// its methods interleaved, one call each per round with the method order rotated from round to
// round (so that neither load drift on the shared machine nor cache state left by the previous
// method favours one of them), and reports medians over the rounds after a warm-up round.
// Prints one JSON line per measurement. Run under the guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 4 --timeout 900 --
//       build/release/bench_ortho_fix_kernels cgs 10000 30 301
// Sections:
//   reduce <n> [rounds]          dot / norm2 against the OpenMP-reduction versions they replaced
//   kernels <n> <k> [rounds]     one thread: the CGS row-block kernels in the array form of
//                                4373b71 against the vector-extension form (renamed copies), and
//                                BLAS gemv over the same rows; all OpenMP-free
//   cgs <n> <k> [rounds]         orthogonalize_cgs2 (Always / IfNeeded) against the legacy CGS2
//                                (four multithreaded BLAS gemv sweeps), all modes, real/complex,
//                                ||w_perp|| / ||w|| = 1 (one DGKS pass) and 1e-6 (two passes)
// The cgs section uses the library API only, so the same source built against two library
// versions gives a before/after comparison.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::index_t;
namespace la = hs::linalg;
namespace lp = hs::lapack;

// ---- timing ------------------------------------------------------------------------------------

/// Runs the methods `rounds` + 1 times each, one call per method and round, the order rotated
/// by one every round; returns the median seconds per method (round 0 is a warm-up). before(m)
/// runs untimed before each call of method m (input reset).
std::vector<double> interleaved_medians(int rounds, const std::vector<std::function<void()>>& methods,
                                        const std::function<void(int)>& before = nullptr) {
  const int m = static_cast<int>(methods.size());
  std::vector<std::vector<double>> t(static_cast<std::size_t>(m));
  for (int r = 0; r <= rounds; ++r)
    for (int i = 0; i < m; ++i) {
      const int which = (i + r) % m;
      if (before) before(which);
      hs::Timer timer;
      methods[static_cast<std::size_t>(which)]();
      const double sec = timer.seconds();
      if (r > 0) t[static_cast<std::size_t>(which)].push_back(sec);
    }
  std::vector<double> med(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    auto& v = t[static_cast<std::size_t>(i)];
    std::sort(v.begin(), v.end());
    med[static_cast<std::size_t>(i)] = v[v.size() / 2];
  }
  return med;
}

const char* scalar_name(bool complex) { return complex ? "complex" : "real"; }

// ---- reduce: dot / norm2 -----------------------------------------------------------------------

/// The reductions before 78795e6 (renamed copies).
template <class T>
T legacy_dot(index_t n, const T* x, const T* y) {
  if constexpr (hs::is_complex_v<T>) {
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
double legacy_norm2(index_t n, const T* x) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (index_t i = 0; i < n; ++i) s += hs::abs2(x[i]);
  return std::sqrt(s);
}

template <class T>
void bench_reduce(index_t n, int rounds) {
  std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
  la::fill_random(x.data(), n, 1, 0);
  la::fill_random(y.data(), n, 1, 1);
  volatile double sink = 0.0;
  const auto t = interleaved_medians(rounds, {
                                                 [&] { sink = sink + std::real(la::dot(n, x.data(), y.data())); },
                                                 [&] { sink = sink + std::real(legacy_dot(n, x.data(), y.data())); },
                                                 [&] { sink = sink + la::norm2(n, x.data()); },
                                                 [&] { sink = sink + legacy_norm2(n, x.data()); },
                                             });
  std::printf(
      "{\"section\":\"reduce\",\"scalar\":\"%s\",\"n\":%lld,\"threads\":%d,\"rounds\":%d,"
      "\"dot_new_ms\":%.4f,\"dot_old_ms\":%.4f,\"dot_speedup\":%.2f,"
      "\"norm2_new_ms\":%.4f,\"norm2_old_ms\":%.4f,\"norm2_speedup\":%.2f}\n",
      scalar_name(hs::is_complex_v<T>), static_cast<long long>(n), omp_get_max_threads(), rounds, 1e3 * t[0],
      1e3 * t[1], t[1] / t[0], 1e3 * t[2], 1e3 * t[3], t[3] / t[2]);
  std::fflush(stdout);
}

// ---- kernels: array form (4373b71) vs vector-extension form ------------------------------------

constexpr int kLanes = 8;

inline double lane_sum(const double* a) { return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7])); }
inline double cross_sum(const double* b) { return ((b[0] - b[1]) + (b[4] - b[5])) + ((b[2] - b[3]) + (b[6] - b[7])); }

namespace array_form {  // the kernels of 4373b71

template <int C>
inline void proj_real_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  double a[C][kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < kLanes; ++l) a[c][l] += p[c * ldp + t + l] * x[t + l];
  for (int c = 0; c < C; ++c)
    for (int l = 0; t + l < len; ++l) a[c][l] += p[c * ldp + t + l] * x[t + l];
  for (int c = 0; c < C; ++c) acc[c] += lane_sum(a[c]);
}

template <int C>
inline void proj_cplx_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  double a[C][kLanes] = {}, b[C][kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < kLanes; ++l) {
        a[c][l] += p[c * ldp + t + l] * x[t + l];
        b[c][l] += p[c * ldp + t + l] * x[t + (l ^ 1)];
      }
  for (int c = 0; c < C; ++c)
    for (int l = 0; t + l < len; ++l) {
      a[c][l] += p[c * ldp + t + l] * x[t + l];
      b[c][l] += p[c * ldp + t + l] * x[t + (l ^ 1)];
    }
  for (int c = 0; c < C; ++c) {
    acc[2 * c] += lane_sum(a[c]);
    acc[2 * c + 1] += cross_sum(b[c]);
  }
}

template <int C>
inline void update_real_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  double cc[C];
  for (int j = 0; j < C; ++j) cc[j] = c[j];
#pragma omp simd
  for (index_t t = 0; t < len; ++t) {
    double s = 0.0;
    for (int j = 0; j < C; ++j) s += v[j * ldv + t] * cc[j];
    x[t] -= s;
  }
}

template <int C>
inline void update_cplx_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  double cr[C], ci[C];
  for (int j = 0; j < C; ++j) {
    cr[j] = c[2 * j];
    ci[j] = c[2 * j + 1];
  }
  const index_t m = len / 2;
#pragma omp simd
  for (index_t i = 0; i < m; ++i) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < C; ++j) {
      const double a = v[j * ldv + 2 * i], b = v[j * ldv + 2 * i + 1];
      sr += a * cr[j] - b * ci[j];
      si += b * cr[j] + a * ci[j];
    }
    x[2 * i] -= sr;
    x[2 * i + 1] -= si;
  }
}

inline void proj_real(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) proj_real_cols<8>(len, p + j * ldp, ldp, x, acc + j);
  if (j + 4 <= k) {
    proj_real_cols<4>(len, p + j * ldp, ldp, x, acc + j);
    j += 4;
  }
  for (; j < k; ++j) proj_real_cols<1>(len, p + j * ldp, ldp, x, acc + j);
}
inline void proj_cplx(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 4 <= k; j += 4) proj_cplx_cols<4>(len, p + j * ldp, ldp, x, acc + 2 * j);
  for (; j < k; ++j) proj_cplx_cols<1>(len, p + j * ldp, ldp, x, acc + 2 * j);
}
inline void update_real(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) update_real_cols<8>(len, v + j * ldv, ldv, c + j, x);
  if (j + 4 <= k) {
    update_real_cols<4>(len, v + j * ldv, ldv, c + j, x);
    j += 4;
  }
  for (; j < k; ++j) update_real_cols<1>(len, v + j * ldv, ldv, c + j, x);
}
inline void update_cplx(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + 2 <= k; j += 2) update_cplx_cols<2>(len, v + j * ldv, ldv, c + 2 * j, x);
  for (; j < k; ++j) update_cplx_cols<1>(len, v + j * ldv, ldv, c + 2 * j, x);
}

}  // namespace array_form

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace vector_form {  // the kernels of src/linalg/dense_ops.cpp since the review fixes

typedef double Vec4 __attribute__((vector_size(32)));
inline Vec4 load4(const double* p) {
  Vec4 r;
  std::memcpy(&r, p, sizeof r);
  return r;
}
inline void store4(double* p, const Vec4& v) { std::memcpy(p, &v, sizeof v); }
inline Vec4 splat4(double s) { return Vec4{s, s, s, s}; }
inline Vec4 swap_pairs(const Vec4& v) { return __builtin_shufflevector(v, v, 1, 0, 3, 2); }

template <int C>
inline void proj_real_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  Vec4 lo[C], hi[C];
  for (int c = 0; c < C; ++c) lo[c] = hi[c] = splat4(0.0);
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes) {
    const Vec4 xlo = load4(x + t), xhi = load4(x + t + 4);
    for (int c = 0; c < C; ++c) {
      lo[c] += load4(p + c * ldp + t) * xlo;
      hi[c] += load4(p + c * ldp + t + 4) * xhi;
    }
  }
  for (int c = 0; c < C; ++c) {
    double a[kLanes];
    store4(a, lo[c]);
    store4(a + 4, hi[c]);
    for (int l = 0; t + l < len; ++l) a[l] += p[c * ldp + t + l] * x[t + l];
    acc[c] += lane_sum(a);
  }
}

template <int C>
inline void proj_cplx_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  Vec4 alo[C], ahi[C], blo[C], bhi[C];
  for (int c = 0; c < C; ++c) alo[c] = ahi[c] = blo[c] = bhi[c] = splat4(0.0);
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes) {
    const Vec4 xlo = load4(x + t), xhi = load4(x + t + 4);
    const Vec4 slo = swap_pairs(xlo), shi = swap_pairs(xhi);
    for (int c = 0; c < C; ++c) {
      const Vec4 plo = load4(p + c * ldp + t), phi = load4(p + c * ldp + t + 4);
      alo[c] += plo * xlo;
      ahi[c] += phi * xhi;
      blo[c] += plo * slo;
      bhi[c] += phi * shi;
    }
  }
  for (int c = 0; c < C; ++c) {
    double a[kLanes], b[kLanes];
    store4(a, alo[c]);
    store4(a + 4, ahi[c]);
    store4(b, blo[c]);
    store4(b + 4, bhi[c]);
    for (int l = 0; t + l < len; ++l) {
      a[l] += p[c * ldp + t + l] * x[t + l];
      b[l] += p[c * ldp + t + l] * x[t + (l ^ 1)];
    }
    acc[2 * c] += lane_sum(a);
    acc[2 * c + 1] += cross_sum(b);
  }
}

template <int C>
inline void update_real_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  Vec4 cc[C];
  for (int j = 0; j < C; ++j) cc[j] = splat4(c[j]);
  index_t t = 0;
  for (; t + 4 <= len; t += 4) {
    Vec4 s = splat4(0.0);
    for (int j = 0; j < C; ++j) s += load4(v + j * ldv + t) * cc[j];
    store4(x + t, load4(x + t) - s);
  }
  for (; t < len; ++t) {
    double s = 0.0;
    for (int j = 0; j < C; ++j) s += v[j * ldv + t] * c[j];
    x[t] -= s;
  }
}

template <int C>
inline void update_cplx_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  Vec4 cr[C], ci[C];
  for (int j = 0; j < C; ++j) {
    cr[j] = splat4(c[2 * j]);
    ci[j] = splat4(c[2 * j + 1]);
  }
  const Vec4 sign = {-1.0, 1.0, -1.0, 1.0};
  index_t t = 0;
  for (; t + 4 <= len; t += 4) {
    Vec4 u = splat4(0.0), s = splat4(0.0);
    for (int j = 0; j < C; ++j) {
      const Vec4 a = load4(v + j * ldv + t);
      u += a * cr[j];
      s += a * ci[j];
    }
    store4(x + t, load4(x + t) - (u + sign * swap_pairs(s)));
  }
  if (t < len) {
    double ur = 0.0, ui = 0.0, sr = 0.0, si = 0.0;
    for (int j = 0; j < C; ++j) {
      const double a = v[j * ldv + t], b = v[j * ldv + t + 1];
      ur += a * c[2 * j];
      ui += b * c[2 * j];
      sr += a * c[2 * j + 1];
      si += b * c[2 * j + 1];
    }
    x[t] -= ur - si;
    x[t + 1] -= ui + sr;
  }
}

inline void proj_real(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) proj_real_cols<8>(len, p + j * ldp, ldp, x, acc + j);
  if (j + 4 <= k) {
    proj_real_cols<4>(len, p + j * ldp, ldp, x, acc + j);
    j += 4;
  }
  for (; j < k; ++j) proj_real_cols<1>(len, p + j * ldp, ldp, x, acc + j);
}
inline void proj_cplx(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 4 <= k; j += 4) proj_cplx_cols<4>(len, p + j * ldp, ldp, x, acc + 2 * j);
  for (; j < k; ++j) proj_cplx_cols<1>(len, p + j * ldp, ldp, x, acc + 2 * j);
}
inline void update_real(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) update_real_cols<8>(len, v + j * ldv, ldv, c + j, x);
  if (j + 4 <= k) {
    update_real_cols<4>(len, v + j * ldv, ldv, c + j, x);
    j += 4;
  }
  for (; j < k; ++j) update_real_cols<1>(len, v + j * ldv, ldv, c + j, x);
}
template <int G>
inline void update_cplx(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + G <= k; j += G) update_cplx_cols<G>(len, v + j * ldv, ldv, c + 2 * j, x);
  for (; j + 2 <= k; j += 2) update_cplx_cols<2>(len, v + j * ldv, ldv, c + 2 * j, x);
  for (; j < k; ++j) update_cplx_cols<1>(len, v + j * ldv, ldv, c + 2 * j, x);
}

}  // namespace vector_form

/// One thread, no OpenMP: a sweep over the rows of V (n x k) in blocks of `rows` rows (the
/// library's fused-sweep tile, 128 KB of V per block): projection acc = V^H x or update y -= V c.
template <class T>
void bench_kernels(index_t n, index_t k, int rounds) {
  constexpr bool cx = hs::is_complex_v<T>;
  constexpr index_t d = cx ? 2 : 1;
  std::vector<T> v(static_cast<std::size_t>(n * k)), x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
  std::vector<T> c(static_cast<std::size_t>(k)), cg(static_cast<std::size_t>(k));
  la::fill_random(v.data(), n * k, 1, 0);
  la::fill_random(x.data(), n, 1, 1);
  la::fill_random(y.data(), n, 1, 2);
  la::fill_random(c.data(), k, 1, 3);
  for (T& z : c) z *= 1e-9;  // y stays O(1) over the rounds
  const index_t rows = std::clamp<index_t>((128 * 1024) / (k * 8 * d) / 32 * 32, 128, 4096);
  const double* vd = reinterpret_cast<const double*>(v.data());
  const double* xd = reinterpret_cast<const double*>(x.data());
  const double* cd = reinterpret_cast<const double*>(c.data());
  double* yd = reinterpret_cast<double*>(y.data());
  std::vector<double> acc(static_cast<std::size_t>(k * d));
  auto proj_sweep = [&](auto&& kern) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (index_t r0 = 0; r0 < n; r0 += rows) {
      const index_t len = std::min(rows, n - r0) * d;
      kern(len, k, vd + r0 * d, n * d, xd + r0 * d, acc.data());
    }
  };
  auto update_sweep = [&](auto&& kern) {
    for (index_t r0 = 0; r0 < n; r0 += rows) {
      const index_t len = std::min(rows, n - r0) * d;
      kern(len, k, vd + r0 * d, n * d, cd, yd + r0 * d);
    }
  };
  std::vector<std::function<void()>> methods;
  std::vector<std::string> names;
  if constexpr (cx) {
    methods = {[&] { proj_sweep(array_form::proj_cplx); },
               [&] { proj_sweep(vector_form::proj_cplx); },
               [&] {
                 for (index_t r0 = 0; r0 < n; r0 += rows)
                   lp::gemv('C', static_cast<lp::fint>(std::min(rows, n - r0)), static_cast<lp::fint>(k), T(1),
                            v.data() + r0, static_cast<lp::fint>(n), x.data() + r0, T(0), cg.data());
               },
               [&] { update_sweep(array_form::update_cplx); },
               [&] { update_sweep(vector_form::update_cplx<2>); },
               [&] { update_sweep(vector_form::update_cplx<4>); },
               [&] { update_sweep(vector_form::update_cplx<8>); },
               [&] {
                 for (index_t r0 = 0; r0 < n; r0 += rows)
                   lp::gemv('N', static_cast<lp::fint>(std::min(rows, n - r0)), static_cast<lp::fint>(k), T(-1),
                            v.data() + r0, static_cast<lp::fint>(n), c.data(), T(1), y.data() + r0);
               }};
    names = {"proj_array", "proj_vector", "proj_gemv", "update_array", "update_vector_g2", "update_vector_g4",
             "update_vector_g8", "update_gemv"};
  } else {
    methods = {[&] { proj_sweep(array_form::proj_real); },
               [&] { proj_sweep(vector_form::proj_real); },
               [&] {
                 for (index_t r0 = 0; r0 < n; r0 += rows)
                   lp::gemv('T', static_cast<lp::fint>(std::min(rows, n - r0)), static_cast<lp::fint>(k), T(1),
                            v.data() + r0, static_cast<lp::fint>(n), x.data() + r0, T(0), cg.data());
               },
               [&] { update_sweep(array_form::update_real); },
               [&] { update_sweep(vector_form::update_real); },
               [&] {
                 for (index_t r0 = 0; r0 < n; r0 += rows)
                   lp::gemv('N', static_cast<lp::fint>(std::min(rows, n - r0)), static_cast<lp::fint>(k), T(-1),
                            v.data() + r0, static_cast<lp::fint>(n), c.data(), T(1), y.data() + r0);
               }};
    names = {"proj_array", "proj_vector", "proj_gemv", "update_array", "update_vector", "update_gemv"};
  }
  // Projections: the two forms must agree bitwise (same lanes).
  if constexpr (cx) {
    proj_sweep(array_form::proj_cplx);
  } else {
    proj_sweep(array_form::proj_real);
  }
  const std::vector<double> ref = acc;
  if constexpr (cx) {
    proj_sweep(vector_form::proj_cplx);
  } else {
    proj_sweep(vector_form::proj_real);
  }
  const bool same = std::memcmp(ref.data(), acc.data(), ref.size() * sizeof(double)) == 0;
  const auto t = interleaved_medians(rounds, methods);
  const double gb = static_cast<double>(n) * static_cast<double>(k) * sizeof(T) * 1e-9;
  std::printf("{\"section\":\"kernels\",\"scalar\":\"%s\",\"n\":%lld,\"k\":%lld,\"rows_per_block\":%lld,"
              "\"basis_MB\":%.2f,\"rounds\":%d,\"projections_bitwise_equal\":%s",
              scalar_name(cx), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(rows),
              gb * 1e3, rounds, same ? "true" : "false");
  for (std::size_t i = 0; i < t.size(); ++i)
    std::printf(",\"%s_GBs\":%.1f", names[i].c_str(), gb / t[i]);
  std::printf("}\n");
  std::fflush(stdout);
}

// ---- cgs: orthogonalize_cgs2 vs the legacy gemv CGS2 ---------------------------------------------

/// CGS2 as before 28fcce0: two passes of multithreaded BLAS gemv ('C' then 'N', plus 'N' on BV in
/// stored mode), M*w recomputed before each pass and at the end in recompute mode.
template <class T>
double legacy_cgs2(index_t n, index_t k, const T* v, const T* bv, T* w, T* bw, T* h, const la::ApplyFn<T>* apply_m) {
  const bool stored = bv != nullptr;
  const bool recompute = !stored && apply_m != nullptr;
  const auto nn = static_cast<lp::fint>(n);
  const auto kk = static_cast<lp::fint>(k);
  std::vector<T> c(static_cast<std::size_t>(k));
  for (index_t j = 0; j < k; ++j) h[j] = T(0);
  for (int pass = 0; pass < 2 && k > 0; ++pass) {
    if (recompute) (*apply_m)(w, bw);
    lp::gemv('C', nn, kk, T(1), stored ? bv : v, nn, recompute ? bw : w, T(0), c.data());
    lp::gemv('N', nn, kk, T(-1), v, nn, c.data(), T(1), w);
    if (stored) lp::gemv('N', nn, kk, T(-1), bv, nn, c.data(), T(1), bw);
    for (index_t j = 0; j < k; ++j) h[j] += c[j];
  }
  if (recompute) (*apply_m)(w, bw);
  if (bw) {
    const double m2 = std::real(legacy_dot(n, w, bw));
    return m2 > 0.0 ? std::sqrt(m2) : 0.0;
  }
  return legacy_norm2(n, w);
}

enum class Mode { Euclidean, Stored, Recompute };
const char* mode_name(Mode m) { return m == Mode::Euclidean ? "euclidean" : m == Mode::Stored ? "stored" : "recompute"; }

template <class T>
void bench_cgs(index_t n, index_t k, Mode mode, double ratio, int rounds) {
  // V (n x k) M-orthonormal, M = diag(1 + (i mod 7) / 7) (identity in Euclidean mode), BV = M V.
  std::vector<double> m(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) m[i] = mode == Mode::Euclidean ? 1.0 : 1.0 + static_cast<double>(i % 7) / 7.0;
  const la::ApplyFn<T> apply = [&](const T* x, T* y) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) y[i] = m[i] * x[i];
  };
  hs::DenseMatrix<T> v(n, k), bv(n, k);
  la::OrthoOptions two;
  two.reorth = la::Reorth::Always;
  for (index_t j = 0; j < k; ++j) {
    la::fill_random(v.col(j), n, 3, static_cast<std::uint64_t>(j));
    apply(v.col(j), bv.col(j));
    la::orthogonalize_cgs2<T>(n, j, v.data(), n, bv.data(), n, v.col(j), bv.col(j), nullptr, nullptr, two);
    apply(v.col(j), bv.col(j));
    const double nrm = std::sqrt(std::real(la::dot(n, v.col(j), bv.col(j))));
    la::scal(n, T(1.0 / nrm), v.col(j));
    la::scal(n, T(1.0 / nrm), bv.col(j));
  }
  // w0 = V y + s r with ||s r_perp||_M / ||w0||_M ~ ratio.
  std::vector<T> y(static_cast<std::size_t>(k)), r(static_cast<std::size_t>(n)), br(static_cast<std::size_t>(n));
  la::fill_random(y.data(), k, 4, 0);
  la::fill_random(r.data(), n, 4, 1);
  apply(r.data(), br.data());
  const double rn = la::orthogonalize_cgs2<T>(n, k, v.data(), n, bv.data(), n, r.data(), br.data(), nullptr, nullptr, two);
  std::vector<T> w0(static_cast<std::size_t>(n), T(0));
  if (ratio < 1.0) la::multiply(n, k, 1, v.data(), n, y.data(), k, w0.data(), n);
  const double yn = la::norm2(k, y.data());
  la::axpy(n, T(ratio < 1.0 ? ratio * yn / std::sqrt(1.0 - ratio * ratio) / rn : 1.0 / rn), r.data(), w0.data());

  std::vector<T> w(static_cast<std::size_t>(n)), bw(static_cast<std::size_t>(n)), h(static_cast<std::size_t>(k));
  const T* bvp = mode == Mode::Stored ? bv.data() : nullptr;
  T* bwp = mode == Mode::Euclidean ? nullptr : bw.data();
  const la::ApplyFn<T>* ap = mode == Mode::Recompute ? &apply : nullptr;
  la::OrthoOptions always, dgks;
  always.reorth = la::Reorth::Always;
  dgks.reorth = la::Reorth::IfNeeded;  // stored mode: valid bw on entry (reset below)
  int dgks_passes = 0;
  const auto t = interleaved_medians(
      rounds,
      {[&] { legacy_cgs2<T>(n, k, v.data(), bvp, w.data(), bwp, h.data(), ap); },
       [&] { la::orthogonalize_cgs2<T>(n, k, v.data(), n, bvp, n, w.data(), bwp, h.data(), ap, always); },
       [&] {
         la::OrthoInfo info;
         la::orthogonalize_cgs2<T>(n, k, v.data(), n, bvp, n, w.data(), bwp, h.data(), ap, dgks, &info);
         dgks_passes = info.passes;
       }},
      [&](int) {
        std::copy(w0.begin(), w0.end(), w.begin());
        if (mode == Mode::Stored) apply(w.data(), bw.data());
      });
  std::printf(
      "{\"section\":\"cgs\",\"scalar\":\"%s\",\"mode\":\"%s\",\"n\":%lld,\"k\":%lld,\"ratio\":%.0e,\"threads\":%d,"
      "\"rounds\":%d,\"legacy_ms\":%.4f,\"always_ms\":%.4f,\"dgks_ms\":%.4f,\"dgks_passes\":%d,"
      "\"speedup_always\":%.2f,\"speedup_dgks\":%.2f}\n",
      scalar_name(hs::is_complex_v<T>), mode_name(mode), static_cast<long long>(n), static_cast<long long>(k), ratio,
      omp_get_max_threads(), rounds, 1e3 * t[0], 1e3 * t[1], 1e3 * t[2], dgks_passes, t[0] / t[1], t[0] / t[2]);
  std::fflush(stdout);
}

int usage() {
  std::fprintf(stderr,
               "usage: bench_ortho_fix_kernels reduce <n> [rounds]\n"
               "       bench_ortho_fix_kernels kernels <n> <k> [rounds]\n"
               "       bench_ortho_fix_kernels cgs <n> <k> [rounds]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string section = argv[1];
  const index_t n = std::atoll(argv[2]);
  if (section == "reduce") {
    const int rounds = argc >= 4 ? std::atoi(argv[3]) : 51;
    bench_reduce<double>(n, rounds);
    bench_reduce<cplx>(n, rounds);
    return 0;
  }
  if (argc < 4) return usage();
  const index_t k = std::atoll(argv[3]);
  if (section == "kernels") {
    const int rounds = argc >= 5 ? std::atoi(argv[4]) : 101;
    bench_kernels<double>(n, k, rounds);
    bench_kernels<cplx>(n, k, rounds);
    return 0;
  }
  if (section == "cgs") {
    const int rounds = argc >= 5 ? std::atoi(argv[4]) : 101;
    for (Mode mode : {Mode::Euclidean, Mode::Stored, Mode::Recompute})
      for (double ratio : {1.0, 1e-6}) {
        bench_cgs<double>(n, k, mode, ratio, rounds);
        bench_cgs<cplx>(n, k, mode, ratio, rounds);
      }
    return 0;
  }
  return usage();
}
