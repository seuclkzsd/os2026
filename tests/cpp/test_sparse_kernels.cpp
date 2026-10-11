// Unit tests for the sparse kernel paths of src/sparse/kernels.cpp (P4 "solvephase"): spmv,
// spmm and shifted_residual on every KernelPath against the Reference path (the P1 loops), for
// real and complex scalars, rectangular matrices with empty rows, orders that are not multiples
// of the SIMD width, p = 1 .. 33 vectors (several 16-vector chunks), padded leading dimensions,
// general alpha / beta; bitwise independence of the thread count; workspace reuse; runtime
// dispatch (Auto resolves to the AVX-512 path exactly when it is available, else Portable).
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::KernelOptions;
using hs::KernelPath;

template <class T>
T rand_scalar(std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  if constexpr (hs::is_complex_v<T>) {
    return T(u(g), u(g));
  } else {
    return u(g);
  }
}

/// re + i im (re for real T).
template <class T>
T scalar(double re, double im) {
  if constexpr (hs::is_complex_v<T>) {
    return T(re, im);
  } else {
    static_cast<void>(im);
    return re;
  }
}

/// Random m x n CSR with about `per_row` entries per row; every `empty_every`-th row is empty.
template <class T>
CsrMatrix<T> random_csr(index_t m, index_t n, index_t per_row, index_t empty_every, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_int_distribution<index_t> col(0, n - 1), len(0, 2 * per_row);
  std::vector<index_t> rows, cols;
  std::vector<T> vals;
  for (index_t i = 0; i < m; ++i) {
    if (empty_every > 0 && i % empty_every == 0) continue;
    const index_t l = len(g);
    for (index_t t = 0; t < l; ++t) {
      rows.push_back(i);
      cols.push_back(col(g));
      vals.push_back(rand_scalar<T>(g));
    }
  }
  return hs::csr_from_coo(m, n, rows, cols, vals);
}

template <class T>
std::vector<T> random_block(index_t rows, index_t p, index_t ld, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::vector<T> x(static_cast<std::size_t>(ld * std::max<index_t>(p, 1)));
  for (auto& e : x) e = rand_scalar<T>(g);
  return x;
}

/// |A| |x| row sums (the scale of the rounding error of a row sum), column-major like Y.
template <class T>
std::vector<double> abs_product(const CsrMatrix<T>& a, const T* x, index_t ldx, index_t p) {
  std::vector<double> s(static_cast<std::size_t>(a.nrows * p), 0.0);
  for (index_t k = 0; k < p; ++k)
    for (index_t i = 0; i < a.nrows; ++i)
      for (index_t t = a.rowptr[i]; t < a.rowptr[i + 1]; ++t)
        s[i + k * a.nrows] += std::abs(a.values[t]) * std::abs(x[a.colind[t] + k * ldx]);
  return s;
}

std::vector<KernelPath> fast_paths() {
  std::vector<KernelPath> paths{KernelPath::Portable, KernelPath::Auto};
  if (hs::kernel_path_available(KernelPath::Avx512)) paths.push_back(KernelPath::Avx512);
  return paths;
}

const char* name(KernelPath p) {
  switch (p) {
    case KernelPath::Auto: return "Auto";
    case KernelPath::Reference: return "Reference";
    case KernelPath::Portable: return "Portable";
    case KernelPath::Avx512: return "Avx512";
  }
  return "?";
}

template <class T>
class SparseKernelPaths : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(SparseKernelPaths, Scalars);

// spmm (and spmv through p = 1) of every fast path agrees with the Reference path entry by entry
// within 1e-14 of |alpha| |A| |X| + |beta| |Y|; padding rows of Y are never written.
TYPED_TEST(SparseKernelPaths, SpmmMatchesReference) {
  using T = TypeParam;
  const T alpha = scalar<T>(0.5, 1.25), beta = scalar<T>(-0.5, 0.25);
  for (index_t m : {1, 7, 37, 300}) {
    const index_t n = m + 5;  // rectangular
    const auto a = random_csr<T>(m, n, 6, 5, 10 + static_cast<std::uint64_t>(m));
    for (index_t p : {1, 2, 3, 5, 8, 9, 16, 17, 33}) {
      const index_t ldx = n + 3, ldy = m + 2;
      const auto x = random_block<T>(n, p, ldx, 20 + static_cast<std::uint64_t>(p));
      const auto y0 = random_block<T>(m, p, ldy, 30 + static_cast<std::uint64_t>(p));
      const auto scale = abs_product(a, x.data(), ldx, p);
      for (bool beta_zero : {false, true}) {
        const T b = beta_zero ? T(0) : beta;
        auto ref = y0;
        hs::spmm(a, x.data(), ldx, ref.data(), ldy, p, alpha, b, {KernelPath::Reference});
        for (KernelPath path : fast_paths()) {
          auto y = y0;
          if (beta_zero)  // beta == 0 must not read Y (NaN would propagate)
            for (index_t k = 0; k < p; ++k)
              for (index_t i = 0; i < m; ++i) y[i + k * ldy] = T(std::numeric_limits<double>::quiet_NaN());
          hs::spmm(a, x.data(), ldx, y.data(), ldy, p, alpha, b, {path});
          for (index_t k = 0; k < p; ++k) {
            for (index_t i = 0; i < m; ++i) {
              const double bound = 1e-14 * (std::abs(alpha) * scale[i + k * m] +
                                            std::abs(b) * std::abs(y0[i + k * ldy])) + 1e-300;
              ASSERT_LE(std::abs(y[i + k * ldy] - ref[i + k * ldy]), bound)
                  << name(path) << " m=" << m << " p=" << p << " i=" << i << " k=" << k << " beta0=" << beta_zero;
            }
            for (index_t i = m; i < ldy; ++i) ASSERT_EQ(y[i + k * ldy], y0[i + k * ldy]) << "padding written";
          }
        }
      }
    }
  }
}

TYPED_TEST(SparseKernelPaths, SpmvMatchesReferenceOnHermitianMatrices) {
  using T = TypeParam;
  for (index_t n : {1, 15, 64, 999}) {
    const auto a = hs::test::random_hermitian<T>(n, std::min(1.0, 12.0 / static_cast<double>(n)), 3);
    const auto x = random_block<T>(n, 1, n, 4);
    const auto scale = abs_product(a, x.data(), n, 1);
    std::vector<T> ref(static_cast<std::size_t>(n)), y(ref.size());
    hs::spmv(a, x.data(), ref.data(), T(1), T(0), {KernelPath::Reference});
    for (KernelPath path : fast_paths()) {
      hs::spmv(a, x.data(), y.data(), T(1), T(0), {path});
      for (index_t i = 0; i < n; ++i)
        ASSERT_LE(std::abs(y[i] - ref[i]), 1e-14 * scale[i] + 1e-300) << name(path) << " n=" << n << " i=" << i;
    }
  }
}

// Each path is bitwise reproducible and independent of the OpenMP thread count; the workspace
// option changes nothing but the allocation.
TYPED_TEST(SparseKernelPaths, ThreadCountAndWorkspaceDoNotChangeBits) {
  using T = TypeParam;
  const index_t m = 5000, n = 5000, p = 13;
  const auto a = random_csr<T>(m, n, 9, 7, 41);
  const auto x = random_block<T>(n, p, n, 42);
  const int threads = omp_get_max_threads();
  for (KernelPath path : fast_paths()) {
    for (index_t q : {index_t(1), p}) {
      std::vector<T> y1(static_cast<std::size_t>(m * q)), y2(y1.size()), y3(y1.size());
      omp_set_num_threads(1);
      hs::spmm(a, x.data(), n, y1.data(), m, q, T(1), T(0), {path});
      omp_set_num_threads(std::max(3, threads));
      hs::spmm(a, x.data(), n, y2.data(), m, q, T(1), T(0), {path});
      std::vector<double> ws;
      KernelOptions with_ws{path, &ws};
      hs::spmm(a, x.data(), n, y3.data(), m, q, T(1), T(0), with_ws);
      hs::spmm(a, x.data(), n, y3.data(), m, q, T(1), T(0), with_ws);  // reused workspace
      omp_set_num_threads(threads);
      EXPECT_TRUE(y1 == y2) << name(path) << " q=" << q << ": results depend on the thread count";
      EXPECT_TRUE(y1 == y3) << name(path) << " q=" << q << ": the workspace changed the results";
      if (q > 1) {
        EXPECT_GE(ws.size(), static_cast<std::size_t>(n * q * (hs::is_complex_v<T> ? 2 : 1)));
      }
    }
  }
}

TYPED_TEST(SparseKernelPaths, EmptyMatricesAndBlocks) {
  using T = TypeParam;
  CsrMatrix<T> empty;  // 0 x 0
  std::vector<T> x(4, T(1)), y(4, T(2));
  for (KernelPath path : fast_paths()) {
    hs::spmm(empty, x.data(), 1, y.data(), 1, 3, T(1), T(0), {path});
    // 3 x 2 with no entries: Y = beta * Y.
    CsrMatrix<T> zero;
    zero.nrows = 3;
    zero.ncols = 2;
    zero.rowptr.assign(4, 0);
    std::vector<T> xs(2 * 5, T(1)), ys(3 * 5, T(2));
    hs::spmm(zero, xs.data(), 2, ys.data(), 3, 5, T(1), T(0.5), {path});
    for (const T& v : ys) EXPECT_EQ(v, T(1));
    hs::spmm(zero, xs.data(), 2, ys.data(), 3, 0, T(1), T(0), {path});  // p = 0: no-op
    for (const T& v : ys) EXPECT_EQ(v, T(1));
  }
  EXPECT_EQ(y, std::vector<T>(4, T(2)));
}

// R = B X - (A Y - sigma B Y) (and B = I) on the fast paths against the Reference path (three
// spmm passes), within 1e-14 of the scale |B| (|X| + |sigma| |Y|) + |A| |Y|.
TYPED_TEST(SparseKernelPaths, ShiftedResidualMatchesReference) {
  using T = TypeParam;
  const double sigma = -0.75;
  for (index_t n : {9, 100, 777}) {
    const auto a = hs::test::random_hermitian<T>(n, std::min(1.0, 10.0 / static_cast<double>(n)), 5);
    const auto b = hs::test::random_hpd<T>(n, std::min(1.0, 6.0 / static_cast<double>(n)), 6);
    for (bool with_b : {true, false}) {
      const CsrMatrix<T>* bp = with_b ? &b : nullptr;
      for (index_t p : {1, 4, 8, 16, 17, 21}) {
        const index_t ldx = n + 1, ldy = n + 2, ldr = n + 3;
        const auto x = random_block<T>(n, p, ldx, 50 + static_cast<std::uint64_t>(p));
        const auto y = random_block<T>(n, p, ldy, 60 + static_cast<std::uint64_t>(p));
        std::vector<T> ref(static_cast<std::size_t>(ldr * p), T(7));
        hs::shifted_residual(a, bp, sigma, x.data(), ldx, y.data(), ldy, ref.data(), ldr, p, {KernelPath::Reference});
        // Scale of the rounding errors.
        std::vector<double> scale = abs_product(a, y.data(), ldy, p);
        if (with_b) {
          const auto bx = abs_product(b, x.data(), ldx, p);
          const auto by = abs_product(b, y.data(), ldy, p);
          for (std::size_t i = 0; i < scale.size(); ++i) scale[i] += bx[i] + std::abs(sigma) * by[i];
        } else {
          for (index_t k = 0; k < p; ++k)
            for (index_t i = 0; i < n; ++i)
              scale[i + k * n] += std::abs(x[i + k * ldx]) + std::abs(sigma) * std::abs(y[i + k * ldy]);
        }
        for (KernelPath path : fast_paths()) {
          std::vector<T> r(static_cast<std::size_t>(ldr * p), T(7));
          std::vector<double> ws;
          hs::shifted_residual(a, bp, sigma, x.data(), ldx, y.data(), ldy, r.data(), ldr, p, {path, &ws});
          for (index_t k = 0; k < p; ++k) {
            for (index_t i = 0; i < n; ++i)
              ASSERT_LE(std::abs(r[i + k * ldr] - ref[i + k * ldr]), 1e-14 * scale[i + k * n])
                  << name(path) << " n=" << n << " p=" << p << " with_b=" << with_b << " i=" << i;
            for (index_t i = n; i < ldr; ++i) ASSERT_EQ(r[i + k * ldr], T(7)) << "padding written";
          }
        }
      }
    }
  }
}

TYPED_TEST(SparseKernelPaths, ShiftedResidualRejectsMismatchedOrders) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(5);
  const auto b = hs::test::laplacian_1d<T>(6);
  std::vector<T> x(6, T(1)), y(6, T(1)), r(6);
  try {
    hs::shifted_residual(a, &b, 1.0, x.data(), 6, y.data(), 6, r.data(), 6, 1);
    ADD_FAILURE() << "expected InvalidInput";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput);
  }
}

// Runtime dispatch: Auto is the AVX-512 path exactly when it can run, and gives its bits; an
// unavailable AVX-512 path is refused instead of crashing; the portable fallback always runs.
TEST(SparseKernelDispatch, AutoResolvesToTheBestAvailablePath) {
  EXPECT_TRUE(hs::kernel_path_available(KernelPath::Auto));
  EXPECT_TRUE(hs::kernel_path_available(KernelPath::Reference));
  EXPECT_TRUE(hs::kernel_path_available(KernelPath::Portable));
  const bool avx = hs::kernel_path_available(KernelPath::Avx512);
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Auto), avx ? KernelPath::Avx512 : KernelPath::Portable);
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Portable), KernelPath::Portable);
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Reference), KernelPath::Reference);
  if (avx) {
    EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Avx512), KernelPath::Avx512);
  } else {
    try {
      hs::resolve_kernel_path(KernelPath::Avx512);
      ADD_FAILURE() << "expected Unsupported";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::Unsupported);
    }
  }
  const auto a = hs::test::random_hermitian<double>(300, 0.03, 9);
  const auto x = random_block<double>(300, 6, 300, 10);
  std::vector<double> y_auto(300 * 6), y_best(300 * 6);
  hs::spmm(a, x.data(), 300, y_auto.data(), 300, 6);
  hs::spmm(a, x.data(), 300, y_best.data(), 300, 6, 1.0, 0.0, {hs::resolve_kernel_path(KernelPath::Auto)});
  EXPECT_TRUE(y_auto == y_best);
}

}  // namespace
