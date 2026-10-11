// Runtime-dispatch fallback of the sparse kernels and operators (P4 "solvephase"), exercised on
// any host. This executable sets HS_KERNELS_NO_AVX512=1 before main(), so the library behaves as
// on a CPU without AVX-512F (src/sparse/kernels.cpp reads the variable once, at the first kernel
// call, and ctest runs every test case in its own process): Avx512 is unavailable and refused
// with Unsupported (kernels and operator constructors), and Auto runs the Portable path, whose
// results agree with the Reference path. On a CPU without AVX-512F the same holds without the
// variable (tests/cpp/test_sparse_kernels.cpp, SparseKernelDispatch).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

// Dynamic initialisation of this translation unit runs before main(), hence before any kernel
// call can read the variable.
const bool kAvx512Masked = ::setenv("HS_KERNELS_NO_AVX512", "1", 1) == 0;

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::KernelOptions;
using hs::KernelPath;

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

template <class T>
std::vector<T> random_block(index_t n, index_t p, std::uint64_t seed) {
  std::vector<T> x(static_cast<std::size_t>(n * p));
  for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.data() + j * n, n, seed, j);
  return x;
}

// max over entries of |y - z| / max |z|.
template <class T>
double rel_dev(const std::vector<T>& y, const std::vector<T>& z) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    num = std::max(num, std::abs(y[i] - z[i]));
    den = std::max(den, std::abs(z[i]));
  }
  return num / den;
}

TEST(SparseKernelFallback, Avx512IsUnavailableAndAutoResolvesToPortable) {
  ASSERT_TRUE(kAvx512Masked);
  EXPECT_FALSE(hs::kernel_path_available(KernelPath::Avx512));
  EXPECT_TRUE(hs::kernel_path_available(KernelPath::Auto));
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Auto), KernelPath::Portable);
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Portable), KernelPath::Portable);
  EXPECT_EQ(hs::resolve_kernel_path(KernelPath::Reference), KernelPath::Reference);
  EXPECT_EQ(error_code_of([] { hs::resolve_kernel_path(KernelPath::Avx512); }),
            hs::Error::Code::Unsupported);
}

template <class T>
class SparseKernelFallbackTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(SparseKernelFallbackTyped, Scalars);

TYPED_TEST(SparseKernelFallbackTyped, KernelsRefuseTheAvx512Path) {
  using T = TypeParam;
  const index_t n = 30;
  const auto a = hs::test::laplacian_1d<T>(n);
  const auto x = random_block<T>(n, 3, 1);
  std::vector<T> y(x.size());
  const KernelOptions avx{KernelPath::Avx512};
  EXPECT_EQ(error_code_of([&] { hs::spmv(a, x.data(), y.data(), T(1), T(0), avx); }),
            hs::Error::Code::Unsupported);
  EXPECT_EQ(error_code_of([&] { hs::spmm(a, x.data(), n, y.data(), n, 3, T(1), T(0), avx); }),
            hs::Error::Code::Unsupported);
  EXPECT_EQ(error_code_of([&] {
              hs::shifted_residual(a, nullptr, 0.5, x.data(), n, x.data(), n, y.data(), n, 3, avx);
            }),
            hs::Error::Code::Unsupported);
}

// Auto gives the Portable bits (spmm, shifted_residual with its backward error) for one column,
// several, a full 16-column chunk and more, and agrees with the Reference path.
TYPED_TEST(SparseKernelFallbackTyped, AutoRunsThePortableKernels) {
  using T = TypeParam;
  const index_t n = 700;
  const auto a = hs::test::random_hermitian<T>(n, 0.03, 12);  // ~21 entries per column
  const auto b = hs::test::random_hpd<T>(n, 0.015, 13);
  for (index_t p : {1, 2, 5, 16, 19}) {
    const auto x = random_block<T>(n, p, 14);
    const auto y = random_block<T>(n, p, 15);
    std::vector<T> auto_y(x.size()), port_y(x.size()), ref_y(x.size());
    hs::spmm(a, x.data(), n, auto_y.data(), n, p);
    hs::spmm(a, x.data(), n, port_y.data(), n, p, T(1), T(0), {KernelPath::Portable});
    hs::spmm(a, x.data(), n, ref_y.data(), n, p, T(1), T(0), {KernelPath::Reference});
    EXPECT_TRUE(auto_y == port_y) << "spmm p=" << p;
    EXPECT_LE(rel_dev(auto_y, ref_y), 1e-14) << "spmm p=" << p;
    for (const CsrMatrix<T>* bp : {&b, static_cast<const CsrMatrix<T>*>(nullptr)}) {
      std::vector<double> o_auto(static_cast<std::size_t>(p)), o_port(o_auto.size()),
          o_ref(o_auto.size());
      hs::shifted_residual(a, bp, -0.4, x.data(), n, y.data(), n, auto_y.data(), n, p, {},
                           o_auto.data());
      hs::shifted_residual(a, bp, -0.4, x.data(), n, y.data(), n, port_y.data(), n, p,
                           {KernelPath::Portable}, o_port.data());
      hs::shifted_residual(a, bp, -0.4, x.data(), n, y.data(), n, ref_y.data(), n, p,
                           {KernelPath::Reference}, o_ref.data());
      EXPECT_TRUE(auto_y == port_y && o_auto == o_port) << "residual p=" << p;
      EXPECT_LE(rel_dev(auto_y, ref_y), 1e-13) << "residual p=" << p;
      for (index_t k = 0; k < p; ++k) EXPECT_NEAR(o_auto[k], o_ref[k], 1e-12 * o_ref[k]);
    }
  }
}

template <class T>
class OpsKernelFallback : public ::testing::Test {};
TYPED_TEST_SUITE(OpsKernelFallback, Scalars);

// The operators check the kernel path when they are built: an unavailable Avx512 fails there
// (Unsupported), not at the first apply; Auto builds and matches the Reference operators.
TYPED_TEST(OpsKernelFallback, OperatorsRefuseAvx512AndRunAuto) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.03, 21);
  const auto b = hs::test::random_hpd<T>(n, 0.02, 22);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  const double sigma = 0.5 * (w[n / 2] + w[n / 2 + 1]);
  hs::ShiftedPencil<T> pencil(a, &b);
  hs::FactorOptions single;
  single.precision = hs::FactorPrecision::Single;
  auto f = hs::make_mumps_factorization<T>(single);
  f->analyze(pencil, 1.0, -sigma);
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  hs::SolveRefinement refine;
  refine.max_steps = 10;
  refine.rel_tol = 1e-12;

  EXPECT_EQ(error_code_of([&] { hs::CsrOperator<T> op(a, KernelPath::Avx512); }),
            hs::Error::Code::Unsupported);
  EXPECT_EQ(error_code_of(
                [&] { hs::ShiftInvertOperator<T> op(*f, pencil, sigma, {}, KernelPath::Avx512); }),
            hs::Error::Code::Unsupported);
  EXPECT_EQ(error_code_of([&] {
              hs::ShiftInvertOperator<T> op(*f, pencil, sigma, refine, KernelPath::Avx512);
            }),
            hs::Error::Code::Unsupported);

  const index_t p = 6;
  const auto x = random_block<T>(n, p, 23);
  std::vector<T> y_auto(x.size()), y_ref(x.size());
  hs::CsrOperator<T> c_auto(a), c_port(a, KernelPath::Portable);
  c_auto.apply(x.data(), n, y_auto.data(), n, p);
  c_port.apply(x.data(), n, y_ref.data(), n, p);
  EXPECT_TRUE(y_auto == y_ref);
  hs::ShiftInvertOperator<T> s_auto(*f, pencil, sigma, refine),
      s_ref(*f, pencil, sigma, refine, KernelPath::Reference);
  s_auto.apply(x.data(), n, y_auto.data(), n, p);
  s_ref.apply(x.data(), n, y_ref.data(), n, p);
  EXPECT_LE(s_auto.last_relative_residual(), refine.rel_tol);
  EXPECT_LE(rel_dev(y_auto, y_ref), 1e-10);
}

}  // namespace
