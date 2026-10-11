// Unit tests for the sparse kernel paths of src/sparse/kernels.cpp (P4 "solvephase"): spmv,
// spmm and shifted_residual on every KernelPath against the Reference path (the P1 loops), for
// real and complex scalars, rectangular matrices with empty rows, orders that are not multiples
// of the SIMD width, p = 1 .. 33 vectors (several 16-vector chunks), padded leading dimensions,
// general alpha / beta; bitwise independence of the thread count; workspace reuse; the strided
// fallback for matrices with few entries per column; the componentwise backward error of
// shifted_residual (paths agree; rounding level for an exact solve); runtime dispatch (Auto
// resolves to the AVX-512 path exactly when it is available, else Portable).
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
CsrMatrix<T> random_csr(index_t m, index_t n, index_t per_row, index_t empty_every,
                        std::uint64_t seed) {
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
    case KernelPath::Auto:
      return "Auto";
    case KernelPath::Reference:
      return "Reference";
    case KernelPath::Portable:
      return "Portable";
    case KernelPath::Avx512:
      return "Avx512";
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
  for (index_t m : {1, 7, 37, 300, 1000}) {
    const index_t n = m + 5;  // rectangular
    const auto a = random_csr<T>(m, n, 12, 5, 10 + static_cast<std::uint64_t>(m));
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
              for (index_t i = 0; i < m; ++i)
                y[i + k * ldy] = T(std::numeric_limits<double>::quiet_NaN());
          hs::spmm(a, x.data(), ldx, y.data(), ldy, p, alpha, b, {path});
          for (index_t k = 0; k < p; ++k) {
            for (index_t i = 0; i < m; ++i) {
              const double bound = 1e-14 * (std::abs(alpha) * scale[i + k * m] +
                                            std::abs(b) * std::abs(y0[i + k * ldy])) +
                                   1e-300;
              ASSERT_LE(std::abs(y[i + k * ldy] - ref[i + k * ldy]), bound)
                  << name(path) << " m=" << m << " p=" << p << " i=" << i << " k=" << k
                  << " beta0=" << beta_zero;
            }
            for (index_t i = m; i < ldy; ++i)
              ASSERT_EQ(y[i + k * ldy], y0[i + k * ldy]) << "padding written";
          }
        }
      }
    }
  }
}

TYPED_TEST(SparseKernelPaths, SpmvMatchesReferenceOnHermitianMatrices) {
  using T = TypeParam;
  for (index_t n : {1, 15, 64, 999}) {
    const auto a =
        hs::test::random_hermitian<T>(n, std::min(1.0, 12.0 / static_cast<double>(n)), 3);
    const auto x = random_block<T>(n, 1, n, 4);
    const auto scale = abs_product(a, x.data(), n, 1);
    std::vector<T> ref(static_cast<std::size_t>(n)), y(ref.size());
    hs::spmv(a, x.data(), ref.data(), T(1), T(0), {KernelPath::Reference});
    for (KernelPath path : fast_paths()) {
      hs::spmv(a, x.data(), y.data(), T(1), T(0), {path});
      for (index_t i = 0; i < n; ++i)
        ASSERT_LE(std::abs(y[i] - ref[i]), 1e-14 * scale[i] + 1e-300)
            << name(path) << " n=" << n << " i=" << i;
    }
  }
}

// Each path is bitwise reproducible and independent of the OpenMP thread count; the workspace
// option changes nothing but the allocation.
TYPED_TEST(SparseKernelPaths, ThreadCountAndWorkspaceDoNotChangeBits) {
  using T = TypeParam;
  const index_t m = 5000, n = 5000, p = 13;
  const auto a = random_csr<T>(m, n, 14, 7, 41);  // >= 8 entries per column: interleaved path
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

// Matrices whose X rows are reused too rarely keep the strided loop on every path (bitwise the
// Reference result): a diagonal matrix for any p, 4 entries per column for p > 4.
TYPED_TEST(SparseKernelPaths, SparseColumnsKeepTheStridedLoop) {
  using T = TypeParam;
  const index_t n = 123;
  for (index_t band : {0, 1}) {  // diagonal; 4 entries per column (offsets -1, 0, 1, 2)
    std::vector<index_t> rows, cols;
    std::vector<T> vals;
    for (index_t i = 0; i < n; ++i)
      for (index_t off = -band; off <= 2 * band; ++off)
        if (i + off >= 0 && i + off < n) {
          rows.push_back(i);
          cols.push_back(i + off);
          vals.push_back(scalar<T>(1.0 + 0.01 * static_cast<double>(i + off), 0.5));
        }
    const auto d = hs::csr_from_coo(n, n, rows, cols, vals);
    for (index_t p : {2, 9}) {
      if (band == 1 && p <= 4) continue;  // 4 entries per column suffice for p <= 4
      const auto x = random_block<T>(n, p, n, 77);
      std::vector<T> ref(static_cast<std::size_t>(n * p)), y(ref.size());
      hs::spmm(d, x.data(), n, ref.data(), n, p, T(1), T(0), {KernelPath::Reference});
      for (KernelPath path : fast_paths()) {
        hs::spmm(d, x.data(), n, y.data(), n, p, T(1), T(0), {path});
        EXPECT_TRUE(y == ref) << name(path) << " band=" << band << " p=" << p;
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
    const auto a =
        hs::test::random_hermitian<T>(n, std::min(1.0, 10.0 / static_cast<double>(n)), 5);
    const auto b = hs::test::random_hpd<T>(n, std::min(1.0, 6.0 / static_cast<double>(n)), 6);
    for (bool with_b : {true, false}) {
      const CsrMatrix<T>* bp = with_b ? &b : nullptr;
      for (index_t p : {1, 4, 8, 16, 17, 21}) {
        const index_t ldx = n + 1, ldy = n + 2, ldr = n + 3;
        const auto x = random_block<T>(n, p, ldx, 50 + static_cast<std::uint64_t>(p));
        const auto y = random_block<T>(n, p, ldy, 60 + static_cast<std::uint64_t>(p));
        std::vector<T> ref(static_cast<std::size_t>(ldr * p), T(7));
        std::vector<double> omega_ref(static_cast<std::size_t>(p), -1.0);
        hs::shifted_residual(a, bp, sigma, x.data(), ldx, y.data(), ldy, ref.data(), ldr, p,
                             {KernelPath::Reference}, omega_ref.data());
        // Scale of the rounding errors.
        std::vector<double> scale = abs_product(a, y.data(), ldy, p);
        if (with_b) {
          const auto bx = abs_product(b, x.data(), ldx, p);
          const auto by = abs_product(b, y.data(), ldy, p);
          for (std::size_t i = 0; i < scale.size(); ++i)
            scale[i] += bx[i] + std::abs(sigma) * by[i];
        } else {
          for (index_t k = 0; k < p; ++k)
            for (index_t i = 0; i < n; ++i)
              scale[i + k * n] +=
                  std::abs(x[i + k * ldx]) + std::abs(sigma) * std::abs(y[i + k * ldy]);
        }
        for (KernelPath path : fast_paths()) {
          std::vector<T> r(static_cast<std::size_t>(ldr * p), T(7)), r2(r);
          std::vector<double> ws, omega(static_cast<std::size_t>(p), -1.0);
          hs::shifted_residual(a, bp, sigma, x.data(), ldx, y.data(), ldy, r.data(), ldr, p,
                               {path, &ws}, omega.data());
          // The backward-error output changes nothing in R.
          hs::shifted_residual(a, bp, sigma, x.data(), ldx, y.data(), ldy, r2.data(), ldr, p,
                               {path, &ws});
          EXPECT_TRUE(r == r2) << name(path);
          for (index_t k = 0; k < p; ++k) {
            // Random X, Y: R is far from rounding level, so omega agrees closely across paths.
            EXPECT_GT(omega_ref[k], 1e-3);
            EXPECT_NEAR(omega[k], omega_ref[k], 1e-12 * omega_ref[k]) << name(path) << " k=" << k;
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

// For Y = K^{-1} B X computed accurately (dense LU of a small pencil) the residual is rounding
// noise: the componentwise backward error is a few eps on every path, for both B and B = I, while
// a perturbed Y gives a large one; an exactly zero residual row counts as 0.
TYPED_TEST(SparseKernelPaths, BackwardErrorIsAtRoundingLevelForAnAccurateSolve) {
  using T = TypeParam;
  const index_t n = 120, p = 6;
  const double sigma = 0.37;
  const auto a = hs::test::random_hermitian<T>(n, 0.08, 15);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 16);
  const auto db = hs::test::to_dense(b);
  for (bool with_b : {true, false}) {
    const CsrMatrix<T>* bp = with_b ? &b : nullptr;
    hs::test::DenseShiftInvertOperator<T> exact(hs::test::to_dense(a), with_b ? &db : nullptr,
                                                sigma);
    const auto x = random_block<T>(n, p, n, 17);
    std::vector<T> y(static_cast<std::size_t>(n * p)), r(y.size());
    exact.apply(x.data(), n, y.data(), n, p);
    std::vector<KernelPath> all = fast_paths();
    all.push_back(KernelPath::Reference);
    for (KernelPath path : all) {
      for (index_t q : {index_t(1), p}) {
        std::vector<double> omega(static_cast<std::size_t>(q));
        hs::shifted_residual(a, bp, sigma, x.data(), n, y.data(), n, r.data(), n, q, {path},
                             omega.data());
        for (index_t k = 0; k < q; ++k)
          EXPECT_LT(omega[k], 16 * std::numeric_limits<double>::epsilon())
              << name(path) << " with_b=" << with_b << " q=" << q << " k=" << k;
      }
      auto y2 = y;
      y2[3] += T(1e-6);
      std::vector<double> omega(1);
      hs::shifted_residual(a, bp, sigma, x.data(), n, y2.data(), n, r.data(), n, 1, {path},
                           omega.data());
      EXPECT_GT(omega[0], 1e-9) << name(path);
    }
  }
  // A zero right-hand side and solution: every row is 0 / 0.
  std::vector<T> zero(static_cast<std::size_t>(n * 2), T(0)), r(zero.size());
  std::vector<double> omega(2, -1.0);
  hs::shifted_residual(a, &b, sigma, zero.data(), n, zero.data(), n, r.data(), n, 2, {},
                       omega.data());
  EXPECT_EQ(omega[0], 0.0);
  EXPECT_EQ(omega[1], 0.0);
}

// A residual entry that is +-Inf or NaN makes the backward error of its column infinite on every
// path, for B and B = I, so that such a column never counts as "at the rounding level": whether
// the Inf sits in Y itself or comes from an overflow of A Y (r = -inf with scale s = inf: inf / inf
// must not become a NaN that the max reduction drops). Finite columns keep a finite value.
TYPED_TEST(SparseKernelPaths, NonFiniteResidualHasInfiniteBackwardError) {
  using T = TypeParam;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<KernelPath> all = fast_paths();
  all.push_back(KernelPath::Reference);
  {
    // Y holds +Inf in row 7 of the even columns.
    const index_t n = 50;
    const auto a = hs::test::laplacian_1d<T>(n);
    const auto b = hs::test::random_hpd<T>(n, 0.1, 3);
    for (bool with_b : {false, true}) {
      for (index_t p : {1, 2, 5}) {
        const auto x = random_block<T>(n, p, n, 5);
        auto y = random_block<T>(n, p, n, 6);
        for (index_t k = 0; k < p; k += 2) y[7 + k * n] = T(inf);
        for (KernelPath path : all) {
          std::vector<T> r(static_cast<std::size_t>(n * p));
          std::vector<double> omega(static_cast<std::size_t>(p), -1.0);
          hs::shifted_residual(a, with_b ? &b : nullptr, -0.5, x.data(), n, y.data(), n, r.data(),
                               n, p, {path}, omega.data());
          for (index_t k = 0; k < p; ++k) {
            if (k % 2 == 0) {
              EXPECT_EQ(omega[k], inf) << name(path) << " with_b=" << with_b << " p=" << p
                                       << " k=" << k << " r[7]=" << r[7 + k * n];
            } else {
              EXPECT_TRUE(std::isfinite(omega[k]) && omega[k] > 0.0) << name(path) << " k=" << k;
            }
          }
        }
      }
    }
  }
  {
    // Overflow: A = 4 I, B = I (implicit and explicit), sigma = 0.5, x = 0, y = 1e308 e_5:
    // (A y)_5 = 4e308 = inf, so r_5 = 0.5e308 - inf = -inf and s_5 = inf.
    const index_t n = 40;
    auto a = hs::identity<T>(n);
    for (auto& v : a.values) v = T(4);
    const auto eye = hs::identity<T>(n);
    for (bool explicit_b : {false, true}) {
      for (index_t p : {1, 2, 5}) {
        const std::vector<T> x(static_cast<std::size_t>(n * p), T(0));
        std::vector<T> y(x.size(), T(0));
        for (index_t k = 0; k < p; ++k) y[5 + k * n] = T(1e308);
        for (KernelPath path : all) {
          std::vector<T> r(x.size());
          std::vector<double> omega(static_cast<std::size_t>(p), -1.0);
          hs::shifted_residual(a, explicit_b ? &eye : nullptr, 0.5, x.data(), n, y.data(), n,
                               r.data(), n, p, {path}, omega.data());
          for (index_t k = 0; k < p; ++k) {
            EXPECT_TRUE(std::isinf(std::abs(r[5 + k * n]))) << name(path) << " k=" << k;
            EXPECT_EQ(omega[k], inf) << name(path) << " explicit_b=" << explicit_b << " p=" << p
                                     << " k=" << k;
          }
        }
      }
    }
  }
}

// "B = I when b == nullptr": a literal nullptr is accepted without naming the scalar type, and
// gives the same bits as a typed null pointer.
TYPED_TEST(SparseKernelPaths, ShiftedResidualTakesALiteralNullptrForTheIdentity) {
  using T = TypeParam;
  const index_t n = 64, p = 3;
  const auto a = hs::test::laplacian_1d<T>(n);
  const auto x = random_block<T>(n, p, n, 7);
  const auto y = random_block<T>(n, p, n, 8);
  std::vector<T> r1(static_cast<std::size_t>(n * p)), r2(r1.size());
  std::vector<double> o1(static_cast<std::size_t>(p)), o2(o1.size());
  hs::shifted_residual(a, nullptr, 0.25, x.data(), n, y.data(), n, r1.data(), n, p, {},
                       o1.data());
  const CsrMatrix<T>* none = nullptr;
  hs::shifted_residual(a, none, 0.25, x.data(), n, y.data(), n, r2.data(), n, p, {}, o2.data());
  EXPECT_TRUE(r1 == r2);
  EXPECT_TRUE(o1 == o2);
}

// shifted_residual and its backward error are bitwise independent of the thread count and of the
// workspace on every path (the per-column max of the backward error included), for B and B = I,
// one column, several and more than one 16-column chunk.
TYPED_TEST(SparseKernelPaths, ShiftedResidualBitsDoNotDependOnThreadsOrWorkspace) {
  using T = TypeParam;
  const index_t n = 5000, pmax = 19;
  const auto a = random_csr<T>(n, n, 14, 7, 43);
  const auto b = random_csr<T>(n, n, 9, 0, 44);
  const auto x = random_block<T>(n, pmax, n, 45);
  const auto y = random_block<T>(n, pmax, n, 46);
  const int threads = omp_get_max_threads();
  std::vector<KernelPath> all = fast_paths();
  all.push_back(KernelPath::Reference);
  for (KernelPath path : all) {
    for (bool with_b : {true, false}) {
      const CsrMatrix<T>* bp = with_b ? &b : nullptr;
      for (index_t p : {1, 3, 16, 19}) {
        const std::size_t len = static_cast<std::size_t>(n * p);
        std::vector<T> r1(len), r2(len), r3(len);
        std::vector<double> o1(static_cast<std::size_t>(p)), o2(o1.size()), o3(o1.size());
        omp_set_num_threads(1);
        hs::shifted_residual(a, bp, -0.3, x.data(), n, y.data(), n, r1.data(), n, p, {path},
                             o1.data());
        omp_set_num_threads(std::max(3, threads));
        hs::shifted_residual(a, bp, -0.3, x.data(), n, y.data(), n, r2.data(), n, p, {path},
                             o2.data());
        std::vector<double> ws;
        const KernelOptions with_ws{path, &ws};
        for (int rep = 0; rep < 2; ++rep)  // the second call reuses the workspace
          hs::shifted_residual(a, bp, -0.3, x.data(), n, y.data(), n, r3.data(), n, p, with_ws,
                               o3.data());
        omp_set_num_threads(threads);
        EXPECT_TRUE(r1 == r2) << name(path) << " with_b=" << with_b << " p=" << p
                              << ": R depends on the thread count";
        EXPECT_TRUE(o1 == o2) << name(path) << " with_b=" << with_b << " p=" << p
                              << ": the backward error depends on the thread count";
        EXPECT_TRUE(r1 == r3 && o1 == o3)
            << name(path) << " with_b=" << with_b << " p=" << p << ": the workspace changed bits";
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
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Auto),
            avx ? KernelPath::Avx512 : KernelPath::Portable);
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
  hs::spmm(a, x.data(), 300, y_best.data(), 300, 6, 1.0, 0.0,
           {hs::resolve_kernel_path(KernelPath::Auto)});
  EXPECT_TRUE(y_auto == y_best);
}

}  // namespace
