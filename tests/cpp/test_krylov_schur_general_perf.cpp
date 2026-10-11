// Performance probe of hs::krylov_schur_general on a cheap matrix-free operator, with the run time
// split by phase (src/eigen/general_krylov_schur.hpp). The default size keeps it a fast unit
// test; the measurement run sets the size explicitly, e.g.
//   HS_KSG_PERF_N=100000 python3 tools/run/guard.py --mem-gb 4 --timeout 600 -- <test binary>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "eigen/general_krylov_schur.hpp"
#include "hsolve/eigen.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// Op = D + U W^T: D block diagonal with a shift-invert-like spectrum theta_i ~ 1/(1+i) (for real
/// T every fourth leading value is a 2x2 rotation block holding a conjugate pair; complex T gets
/// complex diagonal values), U, W n x r random with ||U W^T|| ~ 0.05 making Op non-normal.
/// Each application costs O(n r) flops: the solver's own work dominates the profile.
template <class T>
class DiagonalPlusLowRank final : public hs::LinearOperator<T> {
 public:
  DiagonalPlusLowRank(index_t n, index_t rank, std::uint64_t seed)
      : n_(n),
        r_(rank),
        diag_(static_cast<std::size_t>(n)),
        off_(static_cast<std::size_t>(n)),
        u_(n, rank),
        w_(n, rank) {
    for (index_t i = 0; i < n; ++i) {
      const double mag = 1.0 / (1.0 + static_cast<double>(i));
      if constexpr (hs::is_complex_v<T>) {
        diag_[i] = std::polar(mag, 0.5 * std::sin(static_cast<double>(i)));
      } else {
        diag_[i] = mag;
      }
    }
    if constexpr (!hs::is_complex_v<T>) {
      // Rotation blocks [[a, b], [-b, a]] at (i, i+1) for i = 0, 8, 16, ... (< 64).
      for (index_t i = 0; i + 1 < std::min<index_t>(n, 64); i += 8) {
        diag_[i + 1] = diag_[i];
        off_[i] = 0.3 * diag_[i];
      }
    }
    const double scale = 0.05 / std::sqrt(static_cast<double>(n));
    for (index_t j = 0; j < rank; ++j) {
      hs::linalg::fill_random(u_.col(j), n, seed, 2 * j);
      hs::linalg::fill_random(w_.col(j), n, seed, 2 * j + 1);
      hs::linalg::scal(n, T(scale), u_.col(j));
    }
  }
  index_t size() const override { return n_; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override {
    using hs::lapack::fint;
    std::vector<T> c(static_cast<std::size_t>(r_ * p));
    hs::lapack::gemm('T', 'N', static_cast<fint>(r_), static_cast<fint>(p), static_cast<fint>(n_),
                     T(1), w_.data(), static_cast<fint>(n_), x, static_cast<fint>(ldx), T(0),
                     c.data(), static_cast<fint>(r_));
    for (index_t k = 0; k < p; ++k) {
      const T* xk = x + k * ldx;
      T* yk = y + k * ldy;
#pragma omp parallel for schedule(static)
      for (index_t i = 0; i < n_; ++i) {
        T v = diag_[i] * xk[i];
        if (off_[i] != 0.0) v += off_[i] * xk[i + 1];  // upper entry of a rotation block
        if (i > 0 && off_[i - 1] != 0.0) v -= off_[i - 1] * xk[i - 1];  // lower entry
        yk[i] = v;
      }
    }
    hs::lapack::gemm('N', 'N', static_cast<fint>(n_), static_cast<fint>(p), static_cast<fint>(r_),
                     T(1), u_.data(), static_cast<fint>(n_), c.data(), static_cast<fint>(r_), T(1),
                     y, static_cast<fint>(ldy));
  }

 private:
  index_t n_, r_;
  std::vector<T> diag_;
  std::vector<double> off_;
  hs::DenseMatrix<T> u_, w_;
};

index_t perf_size() {
  const char* env = std::getenv("HS_KSG_PERF_N");
  return env ? std::stoll(env) : 20000;
}

template <class T>
class KrylovSchurGeneralPerf : public ::testing::Test {};
using ScalarTypes = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(KrylovSchurGeneralPerf, ScalarTypes);

TYPED_TEST(KrylovSchurGeneralPerf, DiagonalPlusLowRankTimingSplit) {
  using T = TypeParam;
  const index_t n = perf_size();
  DiagonalPlusLowRank<T> op(n, 4, 2026);
  hs::KrylovSchurOptions<T> opt;
  opt.nev = 20;
  opt.tol = 1e-10;
  hs::eigen_detail::GeneralPhaseTimes t;
  const auto r =
      hs::eigen_detail::krylov_schur_general_timed(op, hs::Which::LargestMagnitude, opt, &t);
  ASSERT_TRUE(r.converged);
  const index_t m = static_cast<index_t>(r.theta.size());
  const auto ncv = static_cast<index_t>(r.ritz_all.size());  // one Ritz value per basis vector
  // The leading values are the ones of D moved by at most ||U W^T||.
  EXPECT_NEAR(std::abs(r.theta[0]), 1.0, 0.1);
  // True residuals of the returned vectors agree with the estimates.
  for (index_t i = 0; i < m; ++i) {
    std::vector<cplx> y(static_cast<std::size_t>(n));
    if constexpr (hs::is_complex_v<T>) {
      op.apply(r.vectors.col(i), n, y.data(), n, 1);
    } else {
      hs::DenseMatrix<double> parts(n, 2), out(n, 2);
      for (index_t k = 0; k < n; ++k)
        parts(k, 0) = r.vectors(k, i).real(), parts(k, 1) = r.vectors(k, i).imag();
      op.apply(parts.data(), n, out.data(), n, 2);
      for (index_t k = 0; k < n; ++k) y[k] = cplx(out(k, 0), out(k, 1));
    }
    for (index_t k = 0; k < n; ++k) y[k] -= r.theta[i] * r.vectors(k, i);
    EXPECT_NEAR(hs::linalg::norm2(n, y.data()), r.estimate[i], 1e-12) << i;
    EXPECT_LE(r.estimate[i], opt.tol * std::abs(r.theta[i])) << i;
  }
  const double cycles = static_cast<double>(r.restarts + 1);
  std::printf(
      "[perf] T=%s n=%ld nev=%ld ncv=%ld m=%ld restarts=%ld op_applies=%ld total=%.3fs "
      "per_cycle=%.4fs | apply=%.3fs orth=%.3fs projected=%.3fs update=%.3fs other=%.3fs\n",
      hs::is_complex_v<T> ? "cplx" : "double", static_cast<long>(n), static_cast<long>(opt.nev),
      static_cast<long>(ncv), static_cast<long>(m), static_cast<long>(r.restarts),
      static_cast<long>(r.op_applies), t.total_s, t.total_s / cycles, t.apply_s, t.orthogonalize_s,
      t.projected_s, t.update_s,
      t.total_s - t.apply_s - t.orthogonalize_s - t.projected_s - t.update_s);
}

}  // namespace
