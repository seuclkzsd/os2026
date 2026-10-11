// Benchmarks of the thick-restart Lanczos solver. Disabled in the unit-test run; execute explicitly
// under the resource guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --mem-gb 4 --timeout 1800 --
//       ./build/release/test_lanczos_perf --gtest_also_run_disabled_tests
// Each benchmark prints the time per restart and the split between operator applications,
// orthogonalization (CGS2), Rayleigh-Ritz and the basis rotation of the restart.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "eigen/lanczos_detail.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;
using hs::Which;

/// The nev entries of `spectrum` ranked first by `which`, sorted ascending.
std::vector<double> wanted_sorted(std::vector<double> spectrum, Which which, index_t nev) {
  auto key = [which](double x) {
    switch (which) {
      case Which::SmallestAlgebraic:
        return x;
      case Which::LargestAlgebraic:
        return -x;
      case Which::LargestMagnitude:
        return -std::abs(x);
    }
    return x;
  };
  std::stable_sort(spectrum.begin(), spectrum.end(),
                   [&](double a, double b) { return key(a) < key(b); });
  spectrum.resize(static_cast<std::size_t>(nev));
  std::sort(spectrum.begin(), spectrum.end());
  return spectrum;
}

template <class T>
void run_diagonal_benchmark(const char* label, const std::vector<double>& spectrum, Which which,
                            index_t nev, index_t ncv, double tol) {
  std::vector<T> diag(spectrum.begin(), spectrum.end());
  hs::test::DiagonalOperator<T> op(std::move(diag));
  hs::KrylovSchurOptions<T> opt;
  opt.nev = nev;
  opt.ncv = ncv;
  opt.tol = tol;
  hs::eigen_detail::LanczosTimings t;
  const auto r = hs::eigen_detail::lanczos_krylov_schur_timed<T>(op, nullptr, which, opt, t);

  EXPECT_TRUE(r.converged);
  auto got = r.theta;
  std::sort(got.begin(), got.end());
  const auto want = wanted_sorted(spectrum, which, nev);
  double scale = 0.0;
  for (double x : spectrum) scale = std::max(scale, std::abs(x));
  double max_err = 0.0;
  for (index_t i = 0; i < nev; ++i) max_err = std::max(max_err, std::abs(got[i] - want[i]) / scale);
  EXPECT_LT(max_err, 1e-9);

  const double cycles = static_cast<double>(r.restarts + 1);
  const double accounted = t.op + t.orthogonalization + t.rayleigh_ritz + t.restart;
  auto pct = [&](double x) { return 100.0 * x / t.total; };
  std::printf(
      "[ bench    ] %s (%s): n=%lld nev=%lld ncv=%lld tol=%.0e threads=%d\n"
      "[ bench    ]   converged=%d nconv=%lld restarts=%lld op_applies=%lld max_rel_eig_err=%.1e\n"
      "[ bench    ]   total %.3f s, per cycle (restart) %.4f s, per Lanczos step %.3f ms\n"
      "[ bench    ]   operator %.3f s (%.1f%%), orthogonalization %.3f s (%.1f%%), Rayleigh-Ritz "
      "%.4f s "
      "(%.2f%%), basis rotation %.3f s (%.1f%%), other %.3f s (%.1f%%)\n",
      label, hs::is_complex_v<T> ? "complex" : "real", static_cast<long long>(op.size()),
      static_cast<long long>(nev), static_cast<long long>(ncv), tol, hs::max_threads(),
      r.converged ? 1 : 0, static_cast<long long>(r.nconv), static_cast<long long>(r.restarts),
      static_cast<long long>(r.op_applies), max_err, t.total, t.total / cycles,
      1e3 * (t.op + t.orthogonalization) / static_cast<double>(r.op_applies), t.op, pct(t.op),
      t.orthogonalization, pct(t.orthogonalization), t.rayleigh_ritz, pct(t.rayleigh_ritz),
      t.restart, pct(t.restart), t.total - accounted, pct(t.total - accounted));
}

/// Spectrum of a shift-invert operator (A - sigma)^{-1} for A with eigenvalues on a uniform grid
/// in (0, 1] and sigma between two of them: theta = 1 / (lambda - sigma), wanted = largest |theta|.
std::vector<double> shift_invert_spectrum(index_t n) {
  const double sigma = 0.3 + 0.37 / static_cast<double>(n);
  std::vector<double> theta(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i)
    theta[i] = 1.0 / (static_cast<double>(i + 1) / static_cast<double>(n) - sigma);
  return theta;
}

/// Uniform grid in (0, 1]: equally spaced extreme eigenvalues, the hard case for Lanczos.
std::vector<double> uniform_spectrum(index_t n) {
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[i] = static_cast<double>(i + 1) / static_cast<double>(n);
  return w;
}

constexpr index_t kN = 200000;

TEST(LanczosPerf, DISABLED_DiagonalShiftInvertSpectrumReal) {
  run_diagonal_benchmark<double>("shift-invert-like spectrum, largest magnitude",
                                 shift_invert_spectrum(kN), Which::LargestMagnitude, 50, 101,
                                 1e-12);
}

TEST(LanczosPerf, DISABLED_DiagonalShiftInvertSpectrumComplex) {
  run_diagonal_benchmark<cplx>("shift-invert-like spectrum, largest magnitude",
                               shift_invert_spectrum(kN), Which::LargestMagnitude, 50, 101, 1e-12);
}

TEST(LanczosPerf, DISABLED_DiagonalUniformSpectrumReal) {
  run_diagonal_benchmark<double>("uniform spectrum, largest algebraic", uniform_spectrum(kN),
                                 Which::LargestAlgebraic, 50, 101, 1e-10);
}

}  // namespace
