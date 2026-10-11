// Regression tests of src/linalg/dense_ops through its consumers: configurations where a change of
// the CGS defaults showed only at solver level.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/types.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;
namespace la = hs::linalg;

template <class T>
class LinalgDenseOpsConsumers : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LinalgDenseOpsConsumers, Scalars);

// Thick-restart Lanczos with store_bv passes a scratch bw (zeros, or a stale M*v of an earlier
// restart cycle) to the stored-mode CGS: a DGKS test reading it skipped needed second passes, and
// the solver returned "converged" Ritz vectors with |X^H M X - I| = 12 and residuals of 3.6 (real
// variant 5 below). Diagonal operator with a dominant cluster (heavy cancellation in the Lanczos
// steps) in a strongly weighted M inner product; store_bv and the recompute mode must agree.
TYPED_TEST(LinalgDenseOpsConsumers, LanczosStoreBvKeepsRitzVectorsOrthonormal) {
  using T = TypeParam;
  const index_t n = 3000, nev = 10;
  std::vector<T> m(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) m[i] = T(1.0 + 50.0 * std::fmod(0.6180339887 * static_cast<double>(i), 1.0));
  hs::test::DiagonalOperator<T> mop(m);
  for (int variant = 0; variant < 24; ++variant) {
    std::vector<T> a(static_cast<std::size_t>(n));
    for (index_t i = 0; i < n; ++i) a[i] = T(std::fmod(0.7548776662 * static_cast<double>(i), 1.0));
    // 30 dominant values 10^(1 + variant % 6) (1 + spread k), spread 0.004 .. 0.016.
    const double top = std::pow(10.0, 1 + variant % 6), spread = 0.004 * (variant / 6 + 1);
    for (index_t k = 0; k < 30; ++k) a[n - 1 - k * 97] = T(top * (1.0 + spread * static_cast<double>(k)));
    hs::test::DiagonalOperator<T> op(a);
    for (bool store_bv : {true, false}) {
      hs::KrylovSchurOptions<T> opt;
      opt.nev = nev;
      opt.ncv = 16 + 2 * (variant % 3);
      opt.tol = 1e-13;
      opt.max_restarts = 400;
      opt.store_bv = store_bv;
      opt.seed = 5 + static_cast<std::uint64_t>(variant);
      const auto r = hs::lanczos_krylov_schur<T>(op, &mop, hs::Which::LargestMagnitude, opt);
      ASSERT_TRUE(r.converged) << "variant " << variant << " store_bv " << store_bv;
      const index_t q = r.vectors.cols();
      hs::DenseMatrix<T> mx(n, q), g(q, q);
      for (index_t j = 0; j < q; ++j)
        for (index_t i = 0; i < n; ++i) mx(i, j) = m[i] * r.vectors(i, j);
      la::gram(n, q, r.vectors.data(), n, mx.data(), n, g.data(), q);
      double dev = 0.0, worst = 0.0;
      for (index_t j = 0; j < q; ++j) {
        for (index_t i = 0; i < q; ++i) dev = std::max(dev, std::abs(g(i, j) - T(i == j ? 1.0 : 0.0)));
        // True residual ||Op x - theta x||_M / |theta| (the solver's own relative measure).
        double s2 = 0.0;
        for (index_t i = 0; i < n; ++i) s2 += std::real(m[i]) * hs::abs2(a[i] * r.vectors(i, j) - T(r.theta[j]) * r.vectors(i, j));
        worst = std::max(worst, std::sqrt(s2) / std::abs(r.theta[j]));
      }
      EXPECT_LE(dev, 1e-13) << "variant " << variant << " store_bv " << store_bv;
      EXPECT_LE(worst, 1e-12) << "variant " << variant << " store_bv " << store_bv;
    }
  }
}

}  // namespace
