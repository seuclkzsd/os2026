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

TEST(SparseCsr, FromCooSumsDuplicatesSortsAndDropsZeros) {
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

TEST(SparseCsr, FromCooRejectsOutOfRange) {
  std::vector<index_t> r{0, 3}, c{0, 0};
  std::vector<double> v{1, 1};
  EXPECT_THROW(hs::csr_from_coo<double>(3, 3, r, c, v), hs::Error);
}

TEST(SparseCsr, ValidateDetectsUnsortedColumns) {
  CsrMatrix<double> m;
  m.nrows = m.ncols = 2;
  m.rowptr = {0, 2, 2};
  m.colind = {1, 0};
  m.values = {1, 1};
  EXPECT_THROW(m.validate(), hs::Error);
}

TEST(SparseCsr, ConjTransposeComplex) {
  auto a = random_sparse<cplx>(17, 0.3, false, 1);
  auto t = hs::conj_transpose(a);
  t.validate();
  auto da = to_dense(a), dt = to_dense(t);
  for (index_t i = 0; i < 17; ++i)
    for (index_t j = 0; j < 17; ++j) EXPECT_EQ(dt(j, i), std::conj(da(i, j)));
}

TEST(SparseCsr, AddUnionPattern) {
  auto a = random_sparse<double>(20, 0.2, false, 2);
  auto b = random_sparse<double>(20, 0.2, false, 3);
  auto c = hs::add(2.0, a, -0.5, b);
  c.validate();
  auto da = to_dense(a), db = to_dense(b), dc = to_dense(c);
  for (index_t i = 0; i < 20; ++i)
    for (index_t j = 0; j < 20; ++j) EXPECT_NEAR(dc(i, j), 2.0 * da(i, j) - 0.5 * db(i, j), 1e-15);
}

TEST(SparseCsr, TriangleAndNorms) {
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

TEST(SparseCsr, HermitianDefect) {
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

TEST(SparseKernels, SpmvSpmmReal) {
  check_spmm<double>(40, 1);
  check_spmm<double>(40, 11);
}
TEST(SparseKernels, SpmvSpmmComplex) {
  check_spmm<cplx>(33, 1);
  check_spmm<cplx>(33, 9);
}

TEST(SparseKernels, BetaZeroIgnoresGarbage) {
  auto a = hs::identity<double>(4);
  std::vector<double> x{1, 2, 3, 4}, y(4, std::nan(""));
  hs::spmv(a, x.data(), y.data(), 1.0, 0.0);
  EXPECT_EQ(y, x);
}

// Check upper-pattern values of the pencil against the dense alpha*A + beta*B.
template <class T, class U>
void check_pencil(const hs::ShiftedPencil<T>& pencil, double alpha, double beta, double tol) {
  const index_t n = pencil.order();
  std::vector<U> vals(static_cast<std::size_t>(pencil.nnz()));
  pencil.assemble(alpha, beta, vals.data());
  auto ref = to_dense(pencil.assemble_full(alpha, beta));
  for (index_t i = 0; i < n; ++i) {
    ASSERT_EQ(pencil.colind()[pencil.rowptr()[i]], i) << "diagonal must lead row " << i;
    for (index_t p = pencil.rowptr()[i]; p < pencil.rowptr()[i + 1]; ++p) {
      const index_t j = pencil.colind()[p];
      ASSERT_GE(j, i);
      if (p > pencil.rowptr()[i]) {
        ASSERT_GT(j, pencil.colind()[p - 1]);
      }
      const T got = T(vals[p]);
      EXPECT_LE(std::abs(got - ref(i, j)), tol * (1.0 + std::abs(ref(i, j)))) << i << "," << j;
    }
  }
  // Every nonzero of the upper part of the reference appears in the pattern.
  index_t ref_upper = 0;
  for (index_t i = 0; i < n; ++i)
    for (index_t j = i; j < n; ++j) ref_upper += (ref(i, j) != T(0));
  EXPECT_LE(ref_upper, pencil.nnz());
}

TEST(SparsePencil, AssembleMatchesDenseForSeveralShifts) {
  auto a = random_sparse<cplx>(25, 0.15, true, 9);
  auto b = random_sparse<cplx>(25, 0.15, true, 10);
  hs::ShiftedPencil<cplx> pencil(a, &b);
  for (double sigma : {0.0, 1.25, -3.5}) check_pencil<cplx, cplx>(pencil, 1.0, -sigma, 1e-15);
  check_pencil<cplx, cplx>(pencil, 0.0, 1.0, 1e-15);  // B alone (definiteness test)
}

TEST(SparsePencil, IdentityBAndIdentityMapPath) {
  auto a = random_sparse<double>(30, 0.2, true, 11);  // diagonal always present
  hs::ShiftedPencil<double> pencil(a, nullptr);       // union pattern == upper(A): identity map
  check_pencil<double, double>(pencil, 1.0, -2.0, 1e-15);
  // B diagonal (subset of A's pattern) also takes the identity path for A.
  auto d = hs::identity<double>(30);
  for (auto& v : d.values) v = 3.0;
  hs::ShiftedPencil<double> pencil2(a, &d);
  check_pencil<double, double>(pencil2, 1.0, -0.5, 1e-15);
}

TEST(SparsePencil, MissingDiagonalIsAdded) {
  // A with an empty diagonal and B = I: the pattern must still contain every diagonal entry.
  std::vector<index_t> r{0, 1, 1, 2}, c{1, 0, 2, 1};
  std::vector<double> v{1, 1, 2, 2};
  auto a = hs::csr_from_coo<double>(3, 3, r, c, v);
  hs::ShiftedPencil<double> pencil(a, nullptr);
  EXPECT_EQ(pencil.nnz(), 5);  // (0,0) (0,1) (1,1) (1,2) (2,2)
  check_pencil<double, double>(pencil, 1.0, -1.0, 1e-15);
}

TEST(SparsePencil, SinglePrecisionAssemble) {
  auto a = random_sparse<cplx>(20, 0.2, true, 12);
  auto b = random_sparse<cplx>(20, 0.2, true, 13);
  hs::ShiftedPencil<cplx> pencil(a, &b);
  check_pencil<cplx, std::complex<float>>(pencil, 1.0, -0.75, 2e-7);
}

TEST(SparseCsr, SymmetrizeFromTriangleAndHermitianPart) {
  auto h = random_sparse<cplx>(18, 0.3, true, 14);
  auto up = hs::triangle(h, true), lo = hs::triangle(h, false);
  auto f1 = hs::symmetrize_from_triangle(up), f2 = hs::symmetrize_from_triangle(lo);
  auto dh = to_dense(h), d1 = to_dense(f1), d2 = to_dense(f2);
  for (index_t i = 0; i < 18; ++i)
    for (index_t j = 0; j < 18; ++j) {
      EXPECT_EQ(d1(i, j), dh(i, j));
      EXPECT_EQ(d2(i, j), dh(i, j));
    }
  EXPECT_THROW(hs::symmetrize_from_triangle(h), hs::Error);
  auto g = random_sparse<cplx>(18, 0.3, false, 15);
  auto hp = hs::hermitian_part(g);
  EXPECT_LT(hs::hermitian_defect(hp), 1e-16);
  auto dg = to_dense(g), dp = to_dense(hp);
  for (index_t i = 0; i < 18; ++i)
    for (index_t j = 0; j < 18; ++j)
      EXPECT_NEAR(std::abs(dp(i, j) - 0.5 * (dg(i, j) + std::conj(dg(j, i)))), 0.0, 1e-15);
}

TEST(SparseCsr, ToComplex) {
  auto a = random_sparse<double>(10, 0.3, false, 16);
  auto c = hs::to_complex(a);
  c.validate();
  ASSERT_EQ(c.nnz(), a.nnz());
  for (index_t p = 0; p < a.nnz(); ++p) EXPECT_EQ(c.values[p], cplx(a.values[p], 0.0));
}

TEST(SparseEmbedding, InertiaDoublesAndSpectrumDuplicates) {
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
