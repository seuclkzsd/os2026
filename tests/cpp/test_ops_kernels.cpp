// Unit tests for the P4 "solvephase" changes of src/ops: the operators on every sparse KernelPath
// (results unchanged within rounding / within the refinement tolerance against the P1 Reference
// kernels), the fused refinement residual, and FactorSolveOperator (Op = K^{-1}).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::FactorOptions;
using hs::FactorPrecision;
using hs::index_t;
using hs::KernelPath;
using hs::ShiftedPencil;

template <class T>
std::vector<T> random_block(index_t n, index_t p, index_t ld, std::uint64_t seed) {
  std::vector<T> x(static_cast<std::size_t>(ld * p), T(0));
  for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.data() + j * ld, n, seed, j);
  return x;
}

// max_j ||y_j - z_j||_2 / ||z_j||_2.
template <class T>
double max_rel_diff(index_t n, index_t p, const T* y, index_t ldy, const T* z, index_t ldz) {
  double worst = 0.0;
  for (index_t j = 0; j < p; ++j) {
    double d = 0.0;
    for (index_t i = 0; i < n; ++i) d += hs::abs2(y[i + j * ldy] - z[i + j * ldz]);
    worst = std::max(worst, std::sqrt(d) / hs::linalg::norm2(n, z + j * ldz));
  }
  return worst;
}

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

std::vector<KernelPath> paths() {
  std::vector<KernelPath> p{KernelPath::Reference, KernelPath::Portable, KernelPath::Auto};
  if (hs::kernel_path_available(KernelPath::Avx512)) p.push_back(KernelPath::Avx512);
  return p;
}

/// Indefinite pencil with a shift in a spectral gap (dense eigenvalues of a small problem).
template <class T>
struct Pencil {
  index_t n;
  CsrMatrix<T> a, b;
  bool has_b;
  double sigma;
  Pencil(index_t order, bool with_b)
      : n(order),
        a(hs::test::random_hermitian<T>(order, 0.03, 81)),
        b(hs::test::random_hpd<T>(order, 0.02, 82)),
        has_b(with_b) {
    const auto w = hs::test::dense_eigenvalues(a, with_b ? &b : nullptr);
    std::size_t best = w.size() / 2;
    for (std::size_t k = best - 8; k < best + 8; ++k)
      if (w[k + 1] - w[k] > w[best + 1] - w[best]) best = k;
    sigma = 0.5 * (w[best] + w[best + 1]);
  }
  const CsrMatrix<T>* b_ptr() const { return has_b ? &b : nullptr; }
};

template <class T>
class OpsKernelPaths : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(OpsKernelPaths, Scalars);

TYPED_TEST(OpsKernelPaths, CsrOperatorAgreesOnEveryPath) {
  using T = TypeParam;
  const index_t n = 500, ld = n + 3;
  const auto a = hs::test::random_hermitian<T>(n, 0.02, 3);
  const auto x = random_block<T>(n, 12, ld, 4);
  std::vector<T> ref(static_cast<std::size_t>(ld * 12)), y(ref.size());
  hs::CsrOperator<T> reference(a, KernelPath::Reference);
  for (index_t p : {1, 3, 12}) {
    reference.apply(x.data(), ld, ref.data(), ld, p);
    for (KernelPath path : paths()) {
      hs::CsrOperator<T> op(a, path);
      op.apply(x.data(), ld, y.data(), ld, p);
      op.apply(x.data(), ld, y.data(), ld, p);  // scratch reused
      EXPECT_LE(max_rel_diff(n, p, y.data(), ld, ref.data(), ld), 1e-14) << "p=" << p;
      EXPECT_EQ(op.applications(), 2 * p);
    }
  }
}

// Without refinement only B X changes with the path (rounding); with refinement of a single
// factor every path converges to the same Op x within the refinement tolerance, in a similar
// number of corrections, and the reported residual is that of the returned block.
TYPED_TEST(OpsKernelPaths, ShiftInvertResultsUnchangedAcrossPaths) {
  using T = TypeParam;
  for (bool with_b : {true, false}) {
    Pencil<T> c(300, with_b);
    ShiftedPencil<T> pencil(c.a, c.b_ptr());
    const index_t n = c.n, p = 5, ld = n + 2;
    const auto x = random_block<T>(n, p, ld, 11);

    auto fd = hs::make_mumps_factorization<T>();
    fd->analyze(pencil, 1.0, -c.sigma);
    ASSERT_TRUE(fd->factorize(1.0, -c.sigma));
    std::vector<T> ref(static_cast<std::size_t>(ld * p)), y(ref.size());
    hs::ShiftInvertOperator<T>(*fd, pencil, c.sigma, {}, KernelPath::Reference)
        .apply(x.data(), ld, ref.data(), ld, p);
    for (KernelPath path : paths()) {
      hs::ShiftInvertOperator<T> op(*fd, pencil, c.sigma, {}, path);
      op.apply(x.data(), ld, y.data(), ld, p);
      EXPECT_LE(max_rel_diff(n, p, y.data(), ld, ref.data(), ld), 1e-12) << "with_b=" << with_b;
    }

    FactorOptions single;
    single.precision = FactorPrecision::Single;
    auto fs = hs::make_mumps_factorization<T>(single);
    fs->analyze(pencil, 1.0, -c.sigma);
    ASSERT_TRUE(fs->factorize(1.0, -c.sigma));
    hs::SolveRefinement refine;
    refine.max_steps = 10;
    refine.rel_tol = 1e-12;
    index_t ref_steps = -1;
    for (KernelPath path : paths()) {
      hs::ShiftInvertOperator<T> op(*fs, pencil, c.sigma, refine, path);
      op.apply(x.data(), ld, y.data(), ld, p);
      EXPECT_LE(max_rel_diff(n, p, y.data(), ld, ref.data(), ld), 1e-10) << "with_b=" << with_b;
      EXPECT_LE(op.last_relative_residual(), 1e-12);
      EXPECT_GE(op.refinement_steps(), 1);
      if (ref_steps < 0) ref_steps = op.refinement_steps();
      EXPECT_LE(std::abs(op.refinement_steps() - ref_steps), 1) << "with_b=" << with_b;
    }
  }
}

TYPED_TEST(OpsKernelPaths, UnavailablePathIsRefusedAtConstruction) {
  using T = TypeParam;
  if (hs::kernel_path_available(KernelPath::Avx512)) GTEST_SKIP() << "AVX-512 available here";
  const auto a = hs::test::laplacian_1d<T>(10);
  EXPECT_EQ(error_code_of([&] { hs::CsrOperator<T> op(a, KernelPath::Avx512); }),
            hs::Error::Code::Unsupported);
}

// --------------------------------------------------------------------------- FactorSolveOperator

template <class T>
class OpsFactorSolve : public ::testing::Test {};
TYPED_TEST_SUITE(OpsFactorSolve, Scalars);

TYPED_TEST(OpsFactorSolve, AppliesTheInverseOfTheFactorizedMatrix) {
  using T = TypeParam;
  for (bool with_b : {true, false}) {
    Pencil<T> c(250, with_b);
    ShiftedPencil<T> pencil(c.a, c.b_ptr());
    auto f = hs::make_mumps_factorization<T>();
    f->analyze(pencil, 1.0, -c.sigma);
    ASSERT_TRUE(f->factorize(1.0, -c.sigma));
    hs::FactorSolveOperator<T> op(*f);
    EXPECT_EQ(op.size(), c.n);
    // Checked directly: K (Op x) = x with K = A - sigma B assembled from the pencil.
    const index_t n = c.n, p = 4, ldx = n + 1, ldy = n + 5;
    const auto x = random_block<T>(n, p, ldx, 21);
    std::vector<T> y(static_cast<std::size_t>(ldy * p), T(3));
    op.apply(x.data(), ldx, y.data(), ldy, p);
    EXPECT_EQ(op.applications(), p);
    const auto k = pencil.assemble_full(1.0, -c.sigma);
    std::vector<T> ky(static_cast<std::size_t>(n * p));
    hs::spmm(k, y.data(), ldy, ky.data(), n, p);
    EXPECT_LE(max_rel_diff(n, p, ky.data(), n, x.data(), ldx), 1e-10) << "with_b=" << with_b;
    for (index_t j = 0; j < p; ++j)
      for (index_t i = n; i < ldy; ++i) EXPECT_EQ(y[i + j * ldy], T(3)) << "padding written";
    // Same as the dense reference without B.
    if (!with_b) {
      hs::test::DenseShiftInvertOperator<T> dense(hs::test::to_dense(c.a), nullptr, c.sigma);
      std::vector<T> z(static_cast<std::size_t>(n * p));
      dense.apply(x.data(), ldx, z.data(), n, p);
      EXPECT_LE(max_rel_diff(n, p, y.data(), ldy, z.data(), n), 1e-10);
    }
  }
}

TYPED_TEST(OpsFactorSolve, RequiresAUsableFactorAndDetectsRefactorization) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(20, 21);
  const auto w = hs::test::laplacian_2d_eigenvalues(20, 21);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  EXPECT_EQ(error_code_of([&] { hs::FactorSolveOperator<T> op(*f); }), hs::Error::Code::InvalidInput);
  const double s1 = 0.5 * (w[100] + w[101]), s2 = 0.5 * (w[200] + w[201]);  // in spectral gaps
  f->analyze(pencil, 1.0, -s1);
  const double lambda = w[w.size() / 2];
  ASSERT_FALSE(f->factorize(1.0, -lambda));  // singular shift
  EXPECT_EQ(error_code_of([&] { hs::FactorSolveOperator<T> op(*f); }), hs::Error::Code::InvalidInput);
  ASSERT_TRUE(f->factorize(1.0, -s1));
  hs::FactorSolveOperator<T> op(*f);
  const auto x = random_block<T>(a.nrows, 1, a.nrows, 5);
  std::vector<T> y(static_cast<std::size_t>(a.nrows));
  op.apply(x.data(), a.nrows, y.data(), a.nrows, 1);
  ASSERT_TRUE(f->factorize(1.0, -s2));
  EXPECT_EQ(error_code_of([&] { op.apply(x.data(), a.nrows, y.data(), a.nrows, 1); }), hs::Error::Code::Internal);
}

// A single-precision factor gives K^{-1} to single accuracy only (no refinement here).
TYPED_TEST(OpsFactorSolve, SinglePrecisionFactorIsAnApproximateInverse) {
  using T = TypeParam;
  Pencil<T> c(200, true);
  ShiftedPencil<T> pencil(c.a, c.b_ptr());
  FactorOptions single;
  single.precision = FactorPrecision::Single;
  auto f = hs::make_mumps_factorization<T>(single);
  f->analyze(pencil, 1.0, -c.sigma);
  ASSERT_TRUE(f->factorize(1.0, -c.sigma));
  hs::FactorSolveOperator<T> op(*f);
  const index_t n = c.n;
  const auto x = random_block<T>(n, 2, n, 6);
  std::vector<T> y(static_cast<std::size_t>(n * 2)), ky(y.size());
  op.apply(x.data(), n, y.data(), n, 2);
  hs::spmm(pencil.assemble_full(1.0, -c.sigma), y.data(), n, ky.data(), n, 2);
  const double err = max_rel_diff(n, 2, ky.data(), n, x.data(), n);
  EXPECT_LT(err, 1e-3);
  EXPECT_GT(err, 1e-12);
}

}  // namespace
