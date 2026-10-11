// Kernel-level measurements of src/linalg/dense_ops: deterministic reductions and CGS
// orthogonalization, against copies of the previous implementations (OpenMP `reduction` clauses,
// CGS2 as four multithreaded BLAS gemv sweeps). Prints one JSON line per measurement.
// Run under the guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 4 --timeout 600 --
//       build/release/bench_ortho_kernels reduce 2000000
// Sections: reduce <n> | cgs <n> <k> [reps] | block <n> <k> <p> [reps]
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

/// CGS2 as before the fused kernels: two passes of BLAS gemv ('C' then 'N', plus 'N' on BV in
/// stored mode), M*w recomputed before each pass and at the end in recompute mode.
template <class T>
double legacy_cgs2(index_t n, index_t k, const T* v, index_t ldv, const T* bv, index_t ldbv, T* w, T* bw,
                   T* h, const la::ApplyFn<T>* apply_m) {
  namespace lp = hs::lapack;
  const bool stored = bv != nullptr;
  const bool recompute = !stored && apply_m != nullptr;
  const auto nn = static_cast<lp::fint>(n);
  const auto kk = static_cast<lp::fint>(k);
  std::vector<T> c(static_cast<std::size_t>(k));
  if (h) {
    for (index_t j = 0; j < k; ++j) h[j] = T(0);
  }
  for (int pass = 0; pass < 2 && k > 0; ++pass) {
    if (recompute) (*apply_m)(w, bw);
    if (stored) {
      lp::gemv('C', nn, kk, T(1), bv, static_cast<lp::fint>(ldbv), w, T(0), c.data());
    } else {
      lp::gemv('C', nn, kk, T(1), v, static_cast<lp::fint>(ldv), recompute ? bw : w, T(0), c.data());
    }
    lp::gemv('N', nn, kk, T(-1), v, static_cast<lp::fint>(ldv), c.data(), T(1), w);
    if (stored) lp::gemv('N', nn, kk, T(-1), bv, static_cast<lp::fint>(ldbv), c.data(), T(1), bw);
    if (h) {
      for (index_t j = 0; j < k; ++j) h[j] += c[j];
    }
  }
  if (recompute) (*apply_m)(w, bw);
  if (bw) {
    const double m2 = std::real(legacy_dot(n, w, bw));
    return m2 > 0.0 ? std::sqrt(m2) : 0.0;
  }
  return legacy_norm2(n, w);
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

enum class Mode { Euclidean, Stored, Recompute };
const char* mode_name(Mode m) {
  return m == Mode::Euclidean ? "euclidean" : m == Mode::Stored ? "stored" : "recompute";
}

/// Problem of one CGS measurement: V (n x k) M-orthonormal with M = diag(m), m_i in [1, 2), and
/// BV = M V; w0 = V y + r scaled so that ||w0 after projection|| / ||w0|| ~ `ratio`.
template <class T>
struct CgsProblem {
  index_t n, k;
  std::vector<double> m;
  hs::DenseMatrix<T> v, bv;
  std::vector<T> w0;
  la::ApplyFn<T> apply;

  CgsProblem(index_t n_, index_t k_, double ratio, bool weighted) : n(n_), k(k_), v(n_, k_), bv(n_, k_) {
    m.resize(static_cast<std::size_t>(n));
    for (index_t i = 0; i < n; ++i) m[i] = weighted ? 1.0 + static_cast<double>(i % 7) / 7.0 : 1.0;
    apply = [this](const T* x, T* y) {
#pragma omp parallel for schedule(static)
      for (index_t i = 0; i < n; ++i) y[i] = m[i] * x[i];
    };
    for (index_t j = 0; j < k; ++j) {
      la::fill_random(v.col(j), n, 3, j);
      apply(v.col(j), bv.col(j));
      la::OrthoOptions two;
      two.reorth = la::Reorth::Always;
      const double nrm = la::orthogonalize_cgs2<T>(n, j, v.data(), n, bv.data(), n, v.col(j), bv.col(j),
                                                   nullptr, nullptr, two);
      la::scal(n, T(1.0 / nrm), v.col(j));
      la::scal(n, T(1.0 / nrm), bv.col(j));
    }
    // w0 = V y + s r, r projected out of span(V) once (so the ratio is close to the target).
    std::vector<T> y(static_cast<std::size_t>(k)), r(static_cast<std::size_t>(n)), br(static_cast<std::size_t>(n));
    la::fill_random(y.data(), k, 4, 0);
    la::fill_random(r.data(), n, 4, 1);
    apply(r.data(), br.data());
    const double rn = la::orthogonalize_cgs2<T>(n, k, v.data(), n, bv.data(), n, r.data(), br.data(), nullptr);
    const double yn = la::norm2(k, y.data());
    w0.assign(static_cast<std::size_t>(n), T(0));
    if (ratio < 1.0) la::multiply(n, k, 1, v.data(), n, y.data(), k, w0.data(), n);
    // ||V y||_M = ||y||; target ||s r|| / sqrt(||y||^2 + ||s r||^2) = ratio.
    const double s = ratio < 1.0 ? ratio * yn / std::sqrt(1.0 - ratio * ratio) / rn : 1.0 / rn;
    la::axpy(n, T(s), r.data(), w0.data());
  }

  /// max_j |v_j^H M q| for q = w / ||w||_M.
  double loss(const std::vector<T>& w, double nrm) const {
    std::vector<T> c(static_cast<std::size_t>(k));
    hs::lapack::gemv('C', static_cast<hs::lapack::fint>(n), static_cast<hs::lapack::fint>(k), T(1), bv.data(),
                     static_cast<hs::lapack::fint>(n), w.data(), T(0), c.data());
    double worst = 0.0;
    for (const T& x : c) worst = std::max(worst, std::abs(x) / nrm);
    return worst;
  }
};

template <class T>
void bench_cgs(index_t n, index_t k, Mode mode, double ratio, int reps) {
  CgsProblem<T> pb(n, k, ratio, mode != Mode::Euclidean);
  std::vector<T> w(static_cast<std::size_t>(n)), bw(static_cast<std::size_t>(n)), h(static_cast<std::size_t>(k));
  const T* bv = mode == Mode::Stored ? pb.bv.data() : nullptr;
  T* bwp = mode == Mode::Euclidean ? nullptr : bw.data();
  const la::ApplyFn<T>* apply = mode == Mode::Recompute ? &pb.apply : nullptr;
  auto reset = [&] {
    std::copy(pb.w0.begin(), pb.w0.end(), w.begin());
    if (mode == Mode::Stored) pb.apply(w.data(), bw.data());
  };
  struct Result {
    double ms = 0, loss = 0;
    int passes = 0;
  };
  // The three variants run interleaved (one call each per round), so that bandwidth taken by
  // other jobs on the machine affects them alike; medians over the rounds after a warm-up round.
  auto run = [&](int method, int& passes) {
    if (method == 0) {
      passes = 2;
      return legacy_cgs2<T>(n, k, pb.v.data(), n, bv, n, w.data(), bwp, h.data(), apply);
    }
    la::OrthoOptions opt;
    opt.reorth = method == 1 ? la::Reorth::Always : la::Reorth::IfNeeded;
    la::OrthoInfo info;
    const double nrm = la::orthogonalize_cgs2<T>(n, k, pb.v.data(), n, bv, n, w.data(), bwp, h.data(), apply, opt, &info);
    passes = info.passes;
    return nrm;
  };
  Result res[3];
  std::vector<double> times[3];
  for (int rep = 0; rep <= reps; ++rep)
    for (int method = 0; method < 3; ++method) {
      reset();
      hs::Timer timer;
      const double nrm = run(method, res[method].passes);
      const double sec = timer.seconds();
      if (rep > 0) times[method].push_back(sec);
      if (rep == reps) res[method].loss = pb.loss(w, nrm);
    }
  for (int method = 0; method < 3; ++method) {
    std::sort(times[method].begin(), times[method].end());
    res[method].ms = 1e3 * times[method][times[method].size() / 2];
  }
  const Result& legacy = res[0];
  const Result& always = res[1];
  const Result& dgks = res[2];
  const double mats = mode == Mode::Stored ? 2.0 : 1.0;
  const double sweep_gb = mats * static_cast<double>(n) * static_cast<double>(k) * sizeof(T) * 1e-9;
  std::printf(
      "{\"section\":\"cgs\",\"scalar\":\"%s\",\"mode\":\"%s\",\"n\":%lld,\"k\":%lld,\"ratio\":%.0e,"
      "\"threads\":%d,\"basis_GB\":%.3f,\"legacy_ms\":%.3f,\"always_ms\":%.3f,\"dgks_ms\":%.3f,"
      "\"dgks_passes\":%d,\"speedup_always\":%.2f,\"speedup_dgks\":%.2f,"
      "\"loss_legacy\":%.1e,\"loss_always\":%.1e,\"loss_dgks\":%.1e}\n",
      scalar_name(hs::is_complex_v<T>), mode_name(mode), static_cast<long long>(n), static_cast<long long>(k), ratio,
      omp_get_max_threads(), sweep_gb, legacy.ms, always.ms, dgks.ms, dgks.passes, legacy.ms / always.ms,
      legacy.ms / dgks.ms, legacy.loss, always.loss, dgks.loss);
  std::fflush(stdout);
}

/// Block routine vs p single-vector calls: W (n x p) with column ratios cycling over
/// {1, 0.5, 1e-6} (so the block needs two passes) against the same V as bench_cgs.
template <class T>
void bench_block(index_t n, index_t k, index_t p, Mode mode, int reps) {
  CgsProblem<T> pb(n, k, 1.0, mode != Mode::Euclidean);
  hs::DenseMatrix<T> w0(n, p), w(n, p), bw(n, p), h(k, p);
  for (index_t j = 0; j < p; ++j) {
    const double ratio = j % 3 == 0 ? 1.0 : j % 3 == 1 ? 0.5 : 1e-6;
    std::vector<T> y(static_cast<std::size_t>(k)), r(static_cast<std::size_t>(n)), br(static_cast<std::size_t>(n));
    la::fill_random(y.data(), k, 40 + static_cast<std::uint64_t>(j), 0);
    la::fill_random(r.data(), n, 40 + static_cast<std::uint64_t>(j), 1);
    pb.apply(r.data(), br.data());
    const double rn = la::orthogonalize_cgs2<T>(n, k, pb.v.data(), n, pb.bv.data(), n, r.data(), br.data(), nullptr);
    if (ratio < 1.0) la::multiply(n, k, 1, pb.v.data(), n, y.data(), k, w0.col(j), n);
    const double yn = la::norm2(k, y.data());
    const double sc = ratio < 1.0 ? ratio * yn / std::sqrt(1.0 - ratio * ratio) / rn : 1.0 / rn;
    la::axpy(n, T(sc), r.data(), w0.col(j));
  }
  const T* bv = mode == Mode::Stored ? pb.bv.data() : nullptr;
  T* bwp = mode == Mode::Euclidean ? nullptr : bw.data();
  const la::ApplyBlockFn<T> apply_block = [&](const T* x, index_t ldx, T* y, index_t ldy, index_t cols) {
    for (index_t j = 0; j < cols; ++j) pb.apply(x + j * ldx, y + j * ldy);
  };
  auto reset = [&] {
    std::copy(w0.data(), w0.data() + n * p, w.data());
    if (mode == Mode::Stored)
      for (index_t j = 0; j < p; ++j) pb.apply(w.col(j), bw.col(j));
  };
  // 0: p legacy calls, 1: p orthogonalize_cgs2 calls (DGKS), 2: block routine (DGKS per block).
  std::vector<double> times[3];
  int block_passes = 0;
  double loss[3] = {0, 0, 0};
  for (int rep = 0; rep <= reps; ++rep)
    for (int method = 0; method < 3; ++method) {
      reset();
      std::vector<double> nrm(static_cast<std::size_t>(p));
      hs::Timer timer;
      if (method == 2) {
        block_passes = la::orthogonalize_block_cgs2<T>(n, k, p, pb.v.data(), n, bv, n, w.data(), n, bwp, n, h.data(),
                                                       k, nullptr, nrm.data(),
                                                       mode == Mode::Recompute ? &apply_block : nullptr);
      } else {
        for (index_t j = 0; j < p; ++j) {
          T* bwj = bwp ? bw.col(j) : nullptr;
          const la::ApplyFn<T>* apply = mode == Mode::Recompute ? &pb.apply : nullptr;
          nrm[j] = method == 0 ? legacy_cgs2<T>(n, k, pb.v.data(), n, bv, n, w.col(j), bwj, h.col(j), apply)
                               : la::orthogonalize_cgs2<T>(n, k, pb.v.data(), n, bv, n, w.col(j), bwj, h.col(j), apply);
        }
      }
      const double sec = timer.seconds();
      if (rep > 0) times[method].push_back(sec);
      if (rep == reps) {
        for (index_t j = 0; j < p; ++j) {
          std::vector<T> col(w.col(j), w.col(j) + n);
          loss[method] = std::max(loss[method], pb.loss(col, nrm[j]));
        }
      }
    }
  double ms[3];
  for (int m = 0; m < 3; ++m) {
    std::sort(times[m].begin(), times[m].end());
    ms[m] = 1e3 * times[m][times[m].size() / 2];
  }
  std::printf(
      "{\"section\":\"block\",\"scalar\":\"%s\",\"mode\":\"%s\",\"n\":%lld,\"k\":%lld,\"p\":%lld,\"threads\":%d,"
      "\"legacy_columns_ms\":%.3f,\"cgs2_columns_ms\":%.3f,\"block_ms\":%.3f,\"block_passes\":%d,"
      "\"speedup_block_vs_legacy\":%.2f,\"speedup_block_vs_cgs2\":%.2f,"
      "\"loss_legacy\":%.1e,\"loss_cgs2\":%.1e,\"loss_block\":%.1e}\n",
      scalar_name(hs::is_complex_v<T>), mode_name(mode), static_cast<long long>(n), static_cast<long long>(k),
      static_cast<long long>(p), omp_get_max_threads(), ms[0], ms[1], ms[2], block_passes, ms[0] / ms[2],
      ms[1] / ms[2], loss[0], loss[1], loss[2]);
  std::fflush(stdout);
}

int usage() {
  std::fprintf(stderr,
               "usage: bench_ortho_kernels reduce <n>\n"
               "       bench_ortho_kernels cgs <n> <k> [reps]\n"
               "       bench_ortho_kernels block <n> <k> <p> [reps]\n");
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
  if (section == "cgs" && argc >= 4) {
    const index_t n = std::atoll(argv[2]);
    const index_t k = std::atoll(argv[3]);
    const int reps = argc >= 5 ? std::atoi(argv[4]) : 9;
    // ratio 1: w independent of span(V) (one DGKS pass); 0.5: Lanczos-like cancellation; 1e-6:
    // nearly dependent w (two passes needed).
    for (Mode mode : {Mode::Euclidean, Mode::Stored, Mode::Recompute})
      for (double ratio : {1.0, 0.5, 1e-6}) {
        bench_cgs<double>(n, k, mode, ratio, reps);
        bench_cgs<cplx>(n, k, mode, ratio, reps);
      }
    return 0;
  }
  if (section == "block" && argc >= 5) {
    const index_t n = std::atoll(argv[2]);
    const index_t k = std::atoll(argv[3]);
    const index_t p = std::atoll(argv[4]);
    const int reps = argc >= 6 ? std::atoi(argv[5]) : 5;
    for (Mode mode : {Mode::Euclidean, Mode::Stored, Mode::Recompute}) {
      bench_block<double>(n, k, p, mode, reps);
      bench_block<cplx>(n, k, p, mode, reps);
    }
    return 0;
  }
  return usage();
}
