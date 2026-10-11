// Kernel-level measurements of src/linalg/dense_ops: deterministic reductions and CGS
// orthogonalization, against copies of the previous implementations (OpenMP `reduction` clauses,
// CGS2 as four multithreaded BLAS gemv sweeps). Prints one JSON line per measurement.
// Run under the guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 4 --timeout 600 --
//       build/release/bench_ortho_kernels reduce 2000000
// Sections: reduce <n>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// ---- previous implementations (renamed copies, for comparison only) ---------------------------

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

// ---- timing ----------------------------------------------------------------------------------

/// Median wall time of `reps` calls of f (after one warm-up call).
template <class F>
double median_seconds(int reps, F&& f) {
  f();
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    hs::Timer timer;
    f();
    t.push_back(timer.seconds());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

const char* scalar_name(bool complex) { return complex ? "complex" : "real"; }

// ---- sections --------------------------------------------------------------------------------

template <class T>
void bench_reduce(index_t n) {
  std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
  la::fill_random(x.data(), n, 1, 0);
  la::fill_random(y.data(), n, 1, 1);
  const double bytes_dot = 2.0 * static_cast<double>(n) * sizeof(T);
  const int reps = 21;
  volatile double sink = 0.0;
  const double t_dot_new = median_seconds(reps, [&] { sink = sink + std::real(la::dot(n, x.data(), y.data())); });
  const double t_dot_old = median_seconds(reps, [&] { sink = sink + std::real(legacy_dot(n, x.data(), y.data())); });
  const double t_nrm_new = median_seconds(reps, [&] { sink = sink + la::norm2(n, x.data()); });
  const double t_nrm_old = median_seconds(reps, [&] { sink = sink + legacy_norm2(n, x.data()); });
  std::printf(
      "{\"section\":\"reduce\",\"scalar\":\"%s\",\"n\":%lld,\"threads\":%d,"
      "\"dot_new_ms\":%.4f,\"dot_old_ms\":%.4f,\"norm2_new_ms\":%.4f,\"norm2_old_ms\":%.4f,"
      "\"dot_new_GBs\":%.1f,\"dot_old_GBs\":%.1f}\n",
      scalar_name(hs::is_complex_v<T>), static_cast<long long>(n), omp_get_max_threads(), 1e3 * t_dot_new,
      1e3 * t_dot_old, 1e3 * t_nrm_new, 1e3 * t_nrm_old, bytes_dot / t_dot_new * 1e-9,
      bytes_dot / t_dot_old * 1e-9);
}

int usage() {
  std::fprintf(stderr, "usage: bench_ortho_kernels reduce <n>\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string section = argv[1];
  if (section == "reduce" && argc >= 3) {
    const index_t n = std::atoll(argv[2]);
    bench_reduce<double>(n);
    bench_reduce<cplx>(n);
    return 0;
  }
  return usage();
}
