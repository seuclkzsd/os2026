// Unit tests for src/sparse (CSR construction, structure ops, norms, kernels, pencil, embedding).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;

// Dense copy for verification only (tests may densify small matrices; see DESIGN §5).
template <class T>
hs::DenseMatrix<T> to_dense(const CsrMatrix<T>& a) {
  hs::DenseMatrix<T> d(a.nrows, a.ncols);
  for (index_t i = 0; i < a.nrows; ++i)
    for (index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) d(i, a.colind[p]) += a.values[p];
  return d;
}

template <class T>
T rand_scalar(std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  if constexpr (hs::is_complex_v<T>) {
    return T(u(g), u(g));
  } else {
    return u(g);
  }
}

// Random sparse matrix (optionally Hermitian) of order n with ~density*n*n entries.
template <class T>
CsrMatrix<T> random_sparse(index_t n, double density, bool hermitian, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < n; ++i) {
    for (index_t j = hermitian ? i : 0; j < n; ++j) {
      if (i != j && u(g) > density) continue;
      T x = rand_scalar<T>(g);
      if (hermitian && i == j) x = T(std::real(x) + 4.0);
      r.push_back(i);
      c.push_back(j);
      v.push_back(x);
      if (hermitian && i != j) {
        r.push_back(j);
        c.push_back(i);
        v.push_back(hs::conj_if(x));
      }
    }
  }
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

TEST(Csr, FromCooSumsDuplicatesSortsAndDropsZeros) {
  // entries: (0,2)=1, (0,0)=2, (0,2)=3, (1,1)=5, (1,1)=-5 (cancels), (2,0)=7
  std::vector<index_t> r{0, 0, 0, 1, 1, 2}, c{2, 0, 2, 1, 1, 0};
  std::vector<double> v{1, 2, 3, 5, -5, 7};
  auto m = hs::csr_from_coo<double>(3, 3, r, c, v);
  m.validate();
  EXPECT_EQ(m.nnz(), 3);
  EXPECT_EQ(m.rowptr, (std::vector<index_t>{0, 2, 2, 3}));
  EXPECT_EQ(m.colind, (std::vector<std::int32_t>{0, 2, 0}));
  EXPECT_EQ(m.values, (std::vector<double>{2, 4, 7}));
  auto keep = hs::csr_from_coo<double>(3, 3, r, c, v, /*drop_zeros=*/false);
  EXPECT_EQ(keep.nnz(), 4);
}

TEST(Csr, FromCooRejectsOutOfRange) {
  std::vector<index_t> r{0, 3}, c{0, 0};
  std::vector<double> v{1, 1};
  EXPECT_THROW(hs::csr_from_coo<double>(3, 3, r, c, v), hs::Error);
}

TEST(Csr, ValidateDetectsUnsortedColumns) {
  CsrMatrix<double> m;
  m.nrows = m.ncols = 2;
  m.rowptr = {0, 2, 2};
  m.colind = {1, 0};
  m.values = {1, 1};
  EXPECT_THROW(m.validate(), hs::Error);
}

TEST(Csr, ConjTransposeComplex) {
  auto a = random_sparse<cplx>(17, 0.3, false, 1);
  auto t = hs::conj_transpose(a);
  t.validate();
  auto da = to_dense(a), dt = to_dense(t);
  for (index_t i = 0; i < 17; ++i)
    for (index_t j = 0; j < 17; ++j) EXPECT_EQ(dt(j, i), std::conj(da(i, j)));
}

TEST(Csr, AddUnionPattern) {
  auto a = random_sparse<double>(20, 0.2, false, 2);
  auto b = random_sparse<double>(20, 0.2, false, 3);
  auto c = hs::add(2.0, a, -0.5, b);
  c.validate();
  auto da = to_dense(a), db = to_dense(b), dc = to_dense(c);
  for (index_t i = 0; i < 20; ++i)
    for (index_t j = 0; j < 20; ++j) EXPECT_NEAR(dc(i, j), 2.0 * da(i, j) - 0.5 * db(i, j), 1e-15);
}

TEST(Csr, TriangleAndNorms) {
  auto a = random_sparse<double>(15, 0.3, true, 4);
  auto u = hs::triangle(a, true), l = hs::triangle(a, false);
  u.validate();
  l.validate();
  for (index_t i = 0; i < 15; ++i) {
    for (index_t p = u.rowptr[i]; p < u.rowptr[i + 1]; ++p) EXPECT_GE(u.colind[p], i);
    for (index_t p = l.rowptr[i]; p < l.rowptr[i + 1]; ++p) EXPECT_LE(l.colind[p], i);
  }
  auto d = to_dense(a);
  double fro = 0, one = 0;
  for (index_t j = 0; j < 15; ++j) {
    double cs = 0;
    for (index_t i = 0; i < 15; ++i) {
      fro += d(i, j) * d(i, j);
      cs += std::abs(d(i, j));
    }
    one = std::max(one, cs);
  }
  EXPECT_NEAR(hs::norm_fro(a), std::sqrt(fro), 1e-12);
  EXPECT_NEAR(hs::norm_one(a), one, 1e-12);
}

TEST(Csr, HermitianDefect) {
  auto h = random_sparse<cplx>(30, 0.2, true, 5);
  EXPECT_LT(hs::hermitian_defect(h), 1e-16);
  // Remove one off-diagonal entry: defect becomes sqrt(2)|a|/||A||... verify against dense.
  auto g = random_sparse<cplx>(30, 0.2, false, 6);
  auto d = to_dense(g);
  double num = 0, den = 0;
  for (index_t i = 0; i < 30; ++i)
    for (index_t j = 0; j < 30; ++j) {
      num += std::norm(d(i, j) - std::conj(d(j, i)));
      den += std::norm(d(i, j));
    }
  EXPECT_NEAR(hs::hermitian_defect(g), std::sqrt(num / den), 1e-13);
}

template <class T>
void check_spmm(index_t n, index_t p) {
  auto a = random_sparse<T>(n, 0.1, false, 7);
  std::mt19937_64 g(8);
  std::vector<T> x(static_cast<std::size_t>(n * p)), y(static_cast<std::size_t>(n * p));
  for (auto& e : x) e = rand_scalar<T>(g);
  for (auto& e : y) e = rand_scalar<T>(g);
  auto y0 = y;
  const T alpha = T(1.5), beta = T(-0.25);
  hs::spmm(a, x.data(), n, y.data(), n, p, alpha, beta);
  auto d = to_dense(a);
  for (index_t k = 0; k < p; ++k)
    for (index_t i = 0; i < n; ++i) {
      T ref = T(0);
      for (index_t j = 0; j < n; ++j) ref += d(i, j) * x[j + k * n];
      ref = alpha * ref + beta * y0[i + k * n];
      EXPECT_NEAR(std::abs(y[i + k * n] - ref), 0.0, 1e-13);
    }
}

TEST(Kernels, SpmvSpmmReal) {
  check_spmm<double>(40, 1);
  check_spmm<double>(40, 11);
}
TEST(Kernels, SpmvSpmmComplex) {
  check_spmm<cplx>(33, 1);
  check_spmm<cplx>(33, 9);
}

TEST(Kernels, BetaZeroIgnoresGarbage) {
  auto a = hs::identity<double>(4);
  std::vector<double> x{1, 2, 3, 4}, y(4, std::nan(""));
  hs::spmv(a, x.data(), y.data(), 1.0, 0.0);
  EXPECT_EQ(y, x);
}

TEST(Pencil, AssembleMatchesAdd) {
  auto a = random_sparse<cplx>(25, 0.15, true, 9);
  auto b = random_sparse<cplx>(25, 0.15, true, 10);
  hs::ShiftedPencil<cplx> pencil(a, &b);
  CsrMatrix<cplx> k;
  for (double sigma : {0.0, 1.25, -3.5}) {
    pencil.assemble(sigma, k);
    k.validate();
    auto ref = to_dense(hs::add(cplx(1), a, cplx(-sigma), b));
    auto dk = to_dense(k);
    for (index_t i = 0; i < 25; ++i)
      for (index_t j = 0; j < 25; ++j) EXPECT_NEAR(std::abs(dk(i, j) - ref(i, j)), 0.0, 1e-14);
    EXPECT_EQ(k.colind, pencil.pattern().colind);  // pattern is shared across shifts
  }
}

TEST(Pencil, IdentityB) {
  auto a = random_sparse<double>(12, 0.2, true, 11);
  hs::ShiftedPencil<double> pencil(a, nullptr);
  CsrMatrix<double> k;
  pencil.assemble(2.0, k);
  auto dk = to_dense(k), da = to_dense(a);
  for (index_t i = 0; i < 12; ++i)
    for (index_t j = 0; j < 12; ++j) EXPECT_NEAR(dk(i, j), da(i, j) - (i == j ? 2.0 : 0.0), 1e-15);
}

TEST(Embedding, InertiaDoublesAndSpectrumDuplicates) {
  const index_t n = 20;
  auto k = random_sparse<cplx>(n, 0.25, true, 12);
  // Make it indefinite: subtract 4.5 I.
  k = hs::add(cplx(1), k, cplx(-4.5), hs::identity<cplx>(n));
  auto e = hs::real_embedding(k);
  e.validate();
  EXPECT_LT(hs::hermitian_defect(e), 1e-16);
  auto dk = to_dense(k);
  auto de = to_dense(e);
  std::vector<double> wk(n), we(2 * n);
  hs::lapack::heev('N', 'U', static_cast<int>(n), dk.data(), static_cast<int>(n), wk.data());
  hs::lapack::heev('N', 'U', static_cast<int>(2 * n), de.data(), static_cast<int>(2 * n), we.data());
  for (index_t i = 0; i < n; ++i) {
    EXPECT_NEAR(we[2 * i], wk[i], 1e-12);
    EXPECT_NEAR(we[2 * i + 1], wk[i], 1e-12);
  }
}

}  // namespace
