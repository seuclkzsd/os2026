// Tests of the shared test fixtures themselves (a broken fixture would invalidate other tests).
#include <gtest/gtest.h>

#include <cmath>

#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

template <class T>
void expect_spectrum(const hs::CsrMatrix<T>& a, const std::vector<double>& want, double tol) {
  auto got = hs::test::dense_eigenvalues(a, static_cast<const hs::CsrMatrix<T>*>(nullptr));
  ASSERT_EQ(got.size(), want.size());
  auto w = want;
  std::sort(w.begin(), w.end());
  for (std::size_t i = 0; i < w.size(); ++i) EXPECT_NEAR(got[i], w[i], tol) << i;
}

TEST(SupportFixtures, LaplaciansHaveAnalyticSpectra) {
  expect_spectrum(hs::test::laplacian_1d<double>(37), hs::test::laplacian_1d_eigenvalues(37), 1e-12);
  expect_spectrum(hs::test::laplacian_2d<cplx>(6, 6), hs::test::laplacian_2d_eigenvalues(6, 6), 1e-12);
}

TEST(SupportFixtures, PrescribedSpectrumIsExactAndHermitian) {
  std::vector<double> w{-3, -1e-3, 0, 0, 0, 1e-9, 2, 2, 2, 5, 7.5, 100, -50, 3, 3, 1};
  auto a = hs::test::prescribed_spectrum<cplx>(w, 4, 11);
  EXPECT_LT(hs::hermitian_defect(a), 1e-15);
  expect_spectrum(a, w, 1e-12);
  auto b = hs::test::prescribed_spectrum<double>(w, 3, 12);
  expect_spectrum(b, w, 1e-12);
}

TEST(SupportFixtures, RandomHpdIsPositiveDefinite) {
  auto b = hs::test::random_hpd<cplx>(40, 0.2, 13);
  auto w = hs::test::dense_eigenvalues(b, static_cast<const hs::CsrMatrix<cplx>*>(nullptr));
  EXPECT_GT(w.front(), 0.0);
}

TEST(SupportFixtures, DenseShiftInvertMatchesDefinition) {
  auto a = hs::test::to_dense(hs::test::random_hermitian<double>(30, 0.3, 14));
  auto b = hs::test::to_dense(hs::test::random_hpd<double>(30, 0.2, 15));
  hs::test::DenseShiftInvertOperator<double> op(a, &b, 0.3);
  std::vector<double> x(30, 1.0), y(30);
  op.apply(x.data(), 30, y.data(), 30, 1);
  // Check (A - 0.3 B) y == B x.
  for (index_t i = 0; i < 30; ++i) {
    double lhs = 0, rhs = 0;
    for (index_t j = 0; j < 30; ++j) {
      lhs += (a(i, j) - 0.3 * b(i, j)) * y[j];
      rhs += b(i, j) * x[j];
    }
    EXPECT_NEAR(lhs, rhs, 1e-10);
  }
  EXPECT_EQ(op.applications(), 1);
}

}  // namespace
