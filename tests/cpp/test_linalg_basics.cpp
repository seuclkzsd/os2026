// Smoke tests for the P0 foundation: DenseMatrix, error helpers and the LAPACK wrappers.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

TEST(DenseMatrix, ShapeAndAlignment) {
  hs::DenseMatrix<double> m(5, 3);
  EXPECT_EQ(m.rows(), 5);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.data()) % 64, 0u);
  m(4, 2) = 7.0;
  EXPECT_EQ(m.col(2)[4], 7.0);
  m.resize_cols(5);
  EXPECT_EQ(m(4, 2), 7.0);
  EXPECT_EQ(m(0, 4), 0.0);
}

TEST(Error, ThrowMacroCarriesCode) {
  try {
    HS_THROW(InvalidInput, "n=" << 3);
    FAIL();
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput);
    EXPECT_STREQ(e.what(), "n=3");
  }
}

TEST(Lapack, HeevTridiagonalLaplacian) {
  // 1-D Laplacian tridiag(-1, 2, -1): eigenvalues 2 - 2 cos(k pi / (n + 1)).
  const int n = 8;
  hs::DenseMatrix<double> a(n, n);
  for (int i = 0; i < n; ++i) {
    a(i, i) = 2.0;
    if (i + 1 < n) a(i, i + 1) = a(i + 1, i) = -1.0;
  }
  std::vector<double> w(n);
  hs::lapack::heev('V', 'U', n, a.data(), n, w.data());
  for (int k = 1; k <= n; ++k) {
    EXPECT_NEAR(w[k - 1], 2.0 - 2.0 * std::cos(k * M_PI / (n + 1)), 1e-13);
  }
}

TEST(Lapack, ComplexHermitianAndGeneralized) {
  const int n = 3;
  using z = std::complex<double>;
  hs::DenseMatrix<z> a(n, n), b(n, n);
  // A = [[2, i, 0], [-i, 2, 0], [0, 0, 5]] has eigenvalues 1, 3, 5.
  a(0, 0) = 2.0; a(1, 1) = 2.0; a(2, 2) = 5.0;
  a(0, 1) = z(0, 1); a(1, 0) = z(0, -1);
  for (int i = 0; i < n; ++i) b(i, i) = 2.0;  // B = 2I halves the eigenvalues
  std::vector<double> w(n);
  EXPECT_EQ(hs::lapack::hegv('V', 'U', n, a.data(), n, b.data(), n, w.data()), 0);
  EXPECT_NEAR(w[0], 0.5, 1e-14);
  EXPECT_NEAR(w[1], 1.5, 1e-14);
  EXPECT_NEAR(w[2], 2.5, 1e-14);
}

TEST(Lapack, RealSchurAndReorder) {
  // Block diag(rotation(1 +- 2i), 3, -4): reorder 3 to the front.
  const int n = 4;
  hs::DenseMatrix<double> t(n, n), q(n, n);
  t(0, 0) = 1; t(0, 1) = 2; t(1, 0) = -2; t(1, 1) = 1;
  t(2, 2) = 3; t(3, 3) = -4;
  std::vector<std::complex<double>> w(n);
  hs::lapack::gees(n, t.data(), n, w.data(), q.data(), n);
  std::vector<int> select(n, 0);
  for (int i = 0; i < n; ++i) select[i] = std::abs(w[i].real() - 3.0) < 1e-12 ? 1 : 0;
  const int m = hs::lapack::trsen(n, t.data(), n, q.data(), n, select, w.data());
  EXPECT_EQ(m, 1);
  EXPECT_NEAR(w[0].real(), 3.0, 1e-12);
  EXPECT_NEAR(w[0].imag(), 0.0, 1e-12);
}

}  // namespace
