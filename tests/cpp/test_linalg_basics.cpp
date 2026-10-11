// Smoke tests for the P0 foundation: DenseMatrix, error helpers and the LAPACK wrappers.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace {

TEST(LinalgDenseMatrix, ShapeAndAlignment) {
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

TEST(UtilError, ThrowMacroCarriesCode) {
  try {
    HS_THROW(InvalidInput, "n=" << 3);
    FAIL();
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput);
    EXPECT_STREQ(e.what(), "n=3");
  }
}

TEST(LinalgLapack, HeevTridiagonalLaplacian) {
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

TEST(LinalgLapack, ComplexHermitianAndGeneralized) {
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

TEST(LinalgLapack, RealSchurAndReorder) {
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

TEST(LinalgDenseOps, FillRandomDeterministicAndStreamsDiffer) {
  std::vector<double> a(1000), b(1000), c(1000);
  hs::linalg::fill_random(a.data(), 1000, 42, 0);
  hs::linalg::fill_random(b.data(), 1000, 42, 0);
  hs::linalg::fill_random(c.data(), 1000, 42, 1);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  for (double x : a) {
    EXPECT_GE(x, -1.0);
    EXPECT_LT(x, 1.0);
  }
}

TEST(LinalgDenseOps, Cgs2EuclideanAndMInnerProduct) {
  using z = std::complex<double>;
  const hs::index_t n = 50, k = 6;
  // Orthonormal V via Gram-Schmidt of random vectors (Euclidean).
  hs::DenseMatrix<z> v(n, k);
  for (hs::index_t j = 0; j < k; ++j) {
    hs::linalg::fill_random(v.col(j), n, 7, j);
    double nrm = hs::linalg::orthogonalize_cgs2<z>(n, j, v.data(), n, nullptr, 0, v.col(j), nullptr, nullptr);
    hs::linalg::scal(n, z(1.0 / nrm), v.col(j));
  }
  std::vector<z> w(n), h(k);
  hs::linalg::fill_random(w.data(), n, 7, 99);
  hs::linalg::orthogonalize_cgs2<z>(n, k, v.data(), n, nullptr, 0, w.data(), nullptr, h.data());
  for (hs::index_t j = 0; j < k; ++j) EXPECT_LT(std::abs(hs::linalg::dot(n, v.col(j), w.data())), 1e-14);

  // M-inner product with M = diag(1..n): build M-orthonormal basis, check M-orthogonality.
  std::vector<double> m(n);
  for (hs::index_t i = 0; i < n; ++i) m[i] = 1.0 + i;
  hs::DenseMatrix<z> u(n, k), mu(n, k);
  for (hs::index_t j = 0; j < k; ++j) {
    hs::linalg::fill_random(u.col(j), n, 8, j);
    for (hs::index_t i = 0; i < n; ++i) mu(i, j) = m[i] * u(i, j);
    double nrm = hs::linalg::orthogonalize_cgs2<z>(n, j, u.data(), n, mu.data(), n, u.col(j), mu.col(j), nullptr);
    hs::linalg::scal(n, z(1.0 / nrm), u.col(j));
    hs::linalg::scal(n, z(1.0 / nrm), mu.col(j));
  }
  for (hs::index_t a = 0; a < k; ++a)
    for (hs::index_t b = 0; b < k; ++b) {
      z ip = 0;
      for (hs::index_t i = 0; i < n; ++i) ip += std::conj(u(i, b)) * m[i] * u(i, a);
      EXPECT_NEAR(std::abs(ip - (a == b ? z(1) : z(0))), 0.0, 1e-13);
    }

  // Recompute mode (no stored M*V) must give the same M-orthogonality and bw == M*w on exit.
  hs::linalg::ApplyFn<z> apply_m = [&](const z* x, z* y) {
    for (hs::index_t i = 0; i < n; ++i) y[i] = m[i] * x[i];
  };
  std::vector<z> w2(n), bw2(n), h2(k);
  hs::linalg::fill_random(w2.data(), n, 9, 0);
  const double nrm2 = hs::linalg::orthogonalize_cgs2<z>(n, k, u.data(), n, nullptr, 0, w2.data(), bw2.data(),
                                                        h2.data(), &apply_m);
  for (hs::index_t a = 0; a < k; ++a) {
    z ip = 0;
    for (hs::index_t i = 0; i < n; ++i) ip += std::conj(u(i, a)) * m[i] * w2[i];
    EXPECT_LT(std::abs(ip), 1e-12 * nrm2);
  }
  for (hs::index_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(bw2[i] - m[i] * w2[i]), 0.0, 1e-14);
}

}  // namespace
