// Tests of src/linalg/dense_ops: deterministic reductions (bitwise identical for any thread count)
// and the CGS orthogonalization routines.
#include <gtest/gtest.h>
#include <omp.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::index_t;
namespace la = hs::linalg;

/// Sets the OpenMP thread count for its lifetime and restores the previous one.
class ThreadCount {
 public:
  explicit ThreadCount(int n) : saved_(omp_get_max_threads()) { omp_set_num_threads(n); }
  ~ThreadCount() { omp_set_num_threads(saved_); }
  ThreadCount(const ThreadCount&) = delete;
  ThreadCount& operator=(const ThreadCount&) = delete;

 private:
  int saved_;
};

template <class T>
bool bitwise_equal(const T& a, const T& b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
bool bitwise_equal(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

constexpr int kThreadCounts[] = {1, 3, 4};

template <class T>
class LinalgDenseOpsReduce : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LinalgDenseOpsReduce, Scalars);

TYPED_TEST(LinalgDenseOpsReduce, DotAndNorm2MatchExtendedPrecisionReference) {
  using T = TypeParam;
  // Sizes around the lane width (8), the block (1024) and beyond 256 blocks (several blocks per
  // reduction slot).
  for (index_t n : {0, 1, 7, 8, 9, 1023, 1024, 1025, 5000, 300001}) {
    std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    la::fill_random(x.data(), n, 11, 0);
    la::fill_random(y.data(), n, 11, 1);
    long double re = 0, im = 0, nn = 0, mag = 0;
    for (index_t i = 0; i < n; ++i) {
      const std::complex<long double> xi(std::real(x[i]), std::imag(x[i]));
      const std::complex<long double> yi(std::real(y[i]), std::imag(y[i]));
      const auto p = std::conj(xi) * yi;
      re += p.real();
      im += p.imag();
      nn += std::norm(xi);
      mag += std::abs(xi) * std::abs(yi);
    }
    const T d = la::dot(n, x.data(), y.data());
    const double tol = 1e-15 * static_cast<double>(mag) + 1e-300;
    EXPECT_NEAR(std::real(d), static_cast<double>(re), tol) << "n=" << n;
    EXPECT_NEAR(std::imag(d), static_cast<double>(im), tol) << "n=" << n;
    const double nrm = la::norm2(n, x.data());
    EXPECT_NEAR(nrm, std::sqrt(static_cast<double>(nn)), 1e-15 * std::sqrt(static_cast<double>(nn)) + 1e-300)
        << "n=" << n;
    // x^H x is real and equals norm2^2 up to rounding.
    EXPECT_EQ(std::imag(la::dot(n, x.data(), x.data())), 0.0);
  }
}

TYPED_TEST(LinalgDenseOpsReduce, BitwiseIdenticalForAnyThreadCountAndRepetition) {
  using T = TypeParam;
  for (index_t n : {20000, 300001}) {
    std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    la::fill_random(x.data(), n, 5, 0);
    la::fill_random(y.data(), n, 5, 1);
    // Scale entries over many orders of magnitude so that any change of summation order would
    // show in the last bits.
    for (index_t i = 0; i < n; ++i) x[i] *= std::ldexp(1.0, static_cast<int>(i % 41) - 20);
    T d_ref;
    double n_ref;
    {
      ThreadCount tc(1);
      d_ref = la::dot(n, x.data(), y.data());
      n_ref = la::norm2(n, x.data());
    }
    for (int threads : kThreadCounts) {
      ThreadCount tc(threads);
      for (int rep = 0; rep < 3; ++rep) {
        EXPECT_TRUE(bitwise_equal(la::dot(n, x.data(), y.data()), d_ref)) << "n=" << n << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(la::norm2(n, x.data()), n_ref)) << "n=" << n << " threads=" << threads;
      }
    }
  }
}

TYPED_TEST(LinalgDenseOpsReduce, NonFiniteEntriesPropagate) {
  using T = TypeParam;
  const index_t n = 5000;
  std::vector<T> x(static_cast<std::size_t>(n), T(1.0));
  x[3333] = T(std::nan(""));
  EXPECT_TRUE(std::isnan(la::norm2(n, x.data())));
  EXPECT_TRUE(std::isnan(std::real(la::dot(n, x.data(), x.data()))));
  x[3333] = T(std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isinf(la::norm2(n, x.data())));
}

}  // namespace
