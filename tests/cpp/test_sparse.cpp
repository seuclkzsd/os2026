// Unit tests for src/sparse (CSR construction, structure ops, norms, kernels, pencil, embedding).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <utility>
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

using Scalars = ::testing::Types<double, cplx>;

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

// Reference assembly of one pattern row: each upper entry of A, then of B, is added at its column
// (found by binary search) in a T accumulator, rounded once to U: the arithmetic the pencil
// promises, so the comparison is bitwise.
template <class T, class U>
std::vector<U> reference_assemble(const hs::ShiftedPencil<T>& pencil, double alpha, double beta) {
  std::vector<U> out(static_cast<std::size_t>(pencil.nnz()));
  const T ta = T(alpha), tb = T(beta);
  for (index_t i = 0; i < pencil.order(); ++i) {
    const index_t r0 = pencil.rowptr()[i], r1 = pencil.rowptr()[i + 1];
    std::vector<T> acc(static_cast<std::size_t>(r1 - r0), T(0));
    const auto add = [&](const CsrMatrix<T>& m, const T& s) {
      for (index_t p = m.rowptr[i]; p < m.rowptr[i + 1]; ++p) {
        if (m.colind[p] < i) continue;
        const auto* b = pencil.colind().data() + r0;
        const auto* it = std::lower_bound(b, pencil.colind().data() + r1, m.colind[p]);
        acc[it - b] += s * m.values[p];
      }
    };
    if (alpha != 0.0) add(pencil.a(), ta);
    if (beta != 0.0) {
      if (pencil.b()) {
        add(*pencil.b(), tb);
      } else {
        acc[0] += tb;
      }
    }
    for (std::size_t k = 0; k < acc.size(); ++k) {
      if constexpr (hs::is_complex_v<T>) {
        using R = hs::real_t<U>;
        out[r0 + k] = U(static_cast<R>(acc[k].real()), static_cast<R>(acc[k].imag()));
      } else {
        out[r0 + k] = static_cast<U>(acc[k]);
      }
    }
  }
  return out;
}

template <class T, class U>
void expect_assembly_bitwise(const hs::ShiftedPencil<T>& pencil, double alpha, double beta) {
  std::vector<U> got(static_cast<std::size_t>(pencil.nnz()));
  pencil.assemble(alpha, beta, got.data());
  const std::vector<U> want = reference_assemble<T, U>(pencil, alpha, beta);
  ASSERT_EQ(got.size(), want.size());
  EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * sizeof(U)), 0);
}

// Keeps a random subset of the entries of a Hermitian matrix, symmetrically, dropping some
// diagonal entries and emptying some rows entirely.
template <class T>
CsrMatrix<T> thinned(const CsrMatrix<T>& h, double keep, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < h.nrows; ++i)
    for (index_t p = h.rowptr[i]; p < h.rowptr[i + 1]; ++p) {
      const index_t j = h.colind[p];
      if (j < i || i % 9 == 4 || j % 9 == 4) continue;  // rows/columns 4, 13, ... stay empty
      if (j == i ? i % 6 == 0 : u(g) > keep) continue;
      r.push_back(i);
      c.push_back(j);
      v.push_back(h.values[p]);
      if (j != i) {
        r.push_back(j);
        c.push_back(i);
        v.push_back(hs::conj_if(h.values[p]));
      }
    }
  return hs::csr_from_coo<T>(h.nrows, h.ncols, r, c, v);
}

template <class T>
class SparsePencilAssembly : public ::testing::Test {};
TYPED_TEST_SUITE(SparsePencilAssembly, Scalars);

TYPED_TEST(SparsePencilAssembly, IsBitwiseTheRowMergeOfAAndB) {
  using T = TypeParam;
  using F = hs::rebind_t<T, float>;
  const CsrMatrix<T> full = random_sparse<T>(70, 0.2, true, 51);
  const CsrMatrix<T> sparse = thinned(full, 0.3, 52);  // pattern subset, missing diagonals
  const CsrMatrix<T> other = random_sparse<T>(70, 0.1, true, 53);
  // A's rows are whole pattern rows and B is sparser (copy + merge), the reverse (waveguide-like
  // B containing A's pattern), unrelated patterns, B = I, and A without some diagonals alone.
  const std::vector<std::pair<const CsrMatrix<T>*, const CsrMatrix<T>*>> pairs{
      {&full, &sparse}, {&sparse, &full}, {&full, &other}, {&sparse, nullptr}, {&full, nullptr}};
  for (const auto& [a, b] : pairs) {
    const hs::ShiftedPencil<T> pencil(*a, b);
    for (const auto& [alpha, beta] : std::vector<std::pair<double, double>>{{1.0, -0.37}, {0.0, 1.0}, {1.0, 0.0}}) {
      SCOPED_TRACE(std::to_string(alpha) + ", " + std::to_string(beta) + (b ? "" : ", B = I"));
      expect_assembly_bitwise<T, T>(pencil, alpha, beta);
      expect_assembly_bitwise<T, F>(pencil, alpha, beta);
    }
  }
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
  // The consuming overload gives the same matrix and leaves an empty, valid source.
  auto moved_from = a;
  auto d = hs::to_complex(std::move(moved_from));
  EXPECT_EQ(d.rowptr, c.rowptr);
  EXPECT_EQ(d.colind, c.colind);
  EXPECT_EQ(d.values, c.values);
  EXPECT_EQ(moved_from.nrows, 0);
  EXPECT_NO_THROW(moved_from.validate());
}

// ------------------------------------------- symmetrize_from_triangle / hermitian_part

// U + U^H - diag(U) entry by entry: every stored entry, plus conj of each strict one at the
// mirrored position. No arithmetic touches a value, so this is exact, signed zeros included, and
// symmetrize_from_triangle must agree with it bitwise.
template <class T>
CsrMatrix<T> exact_symmetrize(const CsrMatrix<T>& tri) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < tri.nrows; ++i)
    for (index_t p = tri.rowptr[i]; p < tri.rowptr[i + 1]; ++p) {
      const index_t j = tri.colind[p];
      r.push_back(i);
      c.push_back(j);
      v.push_back(tri.values[p]);
      if (j != i) {
        r.push_back(j);
        c.push_back(i);
        v.push_back(hs::conj_if(tri.values[p]));
      }
    }
  return hs::csr_from_coo<T>(tri.nrows, tri.ncols, r, c, v, false);
}

// The formulations these functions had before the single-pass rewrite (via an explicit
// transpose and add), kept as references. hermitian_part must agree with its reference bitwise.
// The symmetrize reference adds 1 * conj(u_ij) in complex arithmetic, which turns the imaginary
// part -0 of the conjugate of a real-valued u_ij into +0 (for real parts >= 0): it agrees with
// symmetrize_from_triangle by value only.
template <class T>
CsrMatrix<T> reference_symmetrize(const CsrMatrix<T>& tri, bool upper) {
  CsrMatrix<T> strict = hs::triangle(tri, upper);
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < strict.nrows; ++i)
    for (index_t p = strict.rowptr[i]; p < strict.rowptr[i + 1]; ++p)
      if (strict.colind[p] != i) {
        r.push_back(i);
        c.push_back(strict.colind[p]);
        v.push_back(strict.values[p]);
      }
  const CsrMatrix<T> s = hs::csr_from_coo<T>(tri.nrows, tri.ncols, r, c, v, false);
  return hs::add(T(1), tri, T(1), hs::conj_transpose(s));
}

template <class T>
CsrMatrix<T> reference_hermitian_part(const CsrMatrix<T>& a) {
  return hs::add(T(0.5), a, T(0.5), hs::conj_transpose(a));
}

template <class T>
void expect_bitwise_equal(const CsrMatrix<T>& got, const CsrMatrix<T>& want) {
  EXPECT_NO_THROW(got.validate());
  EXPECT_EQ(got.nrows, want.nrows);
  EXPECT_EQ(got.ncols, want.ncols);
  EXPECT_EQ(got.rowptr, want.rowptr);
  EXPECT_EQ(got.colind, want.colind);
  ASSERT_EQ(got.values.size(), want.values.size());
  if (!got.values.empty()) {
    EXPECT_EQ(std::memcmp(got.values.data(), want.values.data(), got.values.size() * sizeof(T)), 0);
  }
}

// Random one-triangle matrix with some empty rows and missing diagonal entries. For complex T,
// every 4th strict entry has imaginary part +0 and every 4th (shifted) -0: their conjugates carry
// the other signed zero, which the mirrored entries must keep.
template <class T>
CsrMatrix<T> random_triangle(index_t n, double density, bool upper, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<index_t> r, c;
  std::vector<T> v;
  std::size_t strict = 0;
  for (index_t i = 0; i < n; ++i) {
    if (i % 7 == 3) continue;  // empty row
    for (index_t j = upper ? i : 0; j < (upper ? n : i + 1); ++j) {
      if (j == i ? (i % 5 == 1) : u(g) > density) continue;  // every 5th diagonal missing
      T x = rand_scalar<T>(g);
      if constexpr (hs::is_complex_v<T>) {
        if (j != i && strict % 4 == 0) x = T(x.real(), 0.0);
        if (j != i && strict % 4 == 2) x = T(x.real(), -0.0);
      }
      strict += j != i;
      r.push_back(i);
      c.push_back(j);
      v.push_back(x);
    }
  }
  return hs::csr_from_coo<T>(n, n, r, c, v, false);
}

template <class T>
class SparseSymmetrize : public ::testing::Test {};
TYPED_TEST_SUITE(SparseSymmetrize, Scalars);

TYPED_TEST(SparseSymmetrize, MatchesReferenceForBothTrianglesAndOverloads) {
  using T = TypeParam;
  for (const bool upper : {true, false}) {
    for (const double density : {0.02, 0.2, 1.0}) {
      SCOPED_TRACE(std::string(upper ? "upper" : "lower") + " density " + std::to_string(density));
      const CsrMatrix<T> tri = random_triangle<T>(61, density, upper, 100 + static_cast<int>(density * 10));
      const CsrMatrix<T> want = exact_symmetrize(tri);
      const CsrMatrix<T> got = hs::symmetrize_from_triangle(tri);
      expect_bitwise_equal(got, want);
      CsrMatrix<T> consumed = tri;
      expect_bitwise_equal(hs::symmetrize_from_triangle(std::move(consumed)), want);
      EXPECT_EQ(consumed.nrows, 0);
      EXPECT_NO_THROW(consumed.validate());
      // The previous formulation gives the same matrix by value (signed zeros may differ).
      const CsrMatrix<T> old = reference_symmetrize(tri, upper);
      EXPECT_EQ(got.rowptr, old.rowptr);
      EXPECT_EQ(got.colind, old.colind);
      EXPECT_EQ(got.values, old.values);  // operator==: -0 == +0
    }
  }
}

TYPED_TEST(SparseSymmetrize, RestoresAHermitianMatrixFromEitherTriangle) {
  using T = TypeParam;
  const CsrMatrix<T> h = random_sparse<T>(40, 0.15, true, 21);
  for (const bool upper : {true, false}) {
    const CsrMatrix<T> tri = hs::triangle(h, upper);
    expect_bitwise_equal(hs::symmetrize_from_triangle(tri), h);
    expect_bitwise_equal(hs::symmetrize_from_triangle(CsrMatrix<T>(tri)), h);
  }
}

TYPED_TEST(SparseSymmetrize, EdgeCases) {
  using T = TypeParam;
  // Empty 0 x 0, 1 x 1 and diagonal-only matrices come back unchanged (both overloads).
  CsrMatrix<T> empty;
  expect_bitwise_equal(hs::symmetrize_from_triangle(empty), empty);
  expect_bitwise_equal(hs::symmetrize_from_triangle(CsrMatrix<T>(empty)), empty);
  CsrMatrix<T> diag = hs::identity<T>(5);
  diag.values[2] = T(-3);
  expect_bitwise_equal(hs::symmetrize_from_triangle(diag), diag);
  expect_bitwise_equal(hs::symmetrize_from_triangle(CsrMatrix<T>(diag)), diag);
  CsrMatrix<T> zero_rows;  // 4 x 4 without entries
  zero_rows.nrows = zero_rows.ncols = 4;
  zero_rows.rowptr.assign(5, 0);
  expect_bitwise_equal(hs::symmetrize_from_triangle(zero_rows), zero_rows);
  // A single strict entry, no diagonal at all: (0, 3) = v mirrors to (3, 0) = conj(v).
  const std::vector<index_t> r{0}, c{3};
  std::mt19937_64 g(5);
  const T v = rand_scalar<T>(g);
  const CsrMatrix<T> one = hs::csr_from_coo<T>(4, 4, r, c, {v});
  for (const CsrMatrix<T>& full : {hs::symmetrize_from_triangle(one), hs::symmetrize_from_triangle(CsrMatrix<T>(one))}) {
    EXPECT_EQ(full.rowptr, (std::vector<index_t>{0, 1, 1, 1, 2}));
    EXPECT_EQ(full.colind, (std::vector<std::int32_t>{3, 0}));
    EXPECT_EQ(full.values, (std::vector<T>{v, hs::conj_if(v)}));
  }
  // Both triangles, or a non-square matrix: InvalidInput, also for the consuming overload.
  const CsrMatrix<T> both = random_sparse<T>(9, 0.4, true, 22);
  EXPECT_THROW(hs::symmetrize_from_triangle(both), hs::Error);
  EXPECT_THROW(hs::symmetrize_from_triangle(CsrMatrix<T>(both)), hs::Error);
  CsrMatrix<T> wide;
  wide.nrows = 2;
  wide.ncols = 3;
  wide.rowptr.assign(3, 0);
  EXPECT_THROW(hs::symmetrize_from_triangle(wide), hs::Error);
  EXPECT_THROW(hs::symmetrize_from_triangle(CsrMatrix<T>(wide)), hs::Error);
}

template <class T>
class SparseHermitianPart : public ::testing::Test {};
TYPED_TEST_SUITE(SparseHermitianPart, Scalars);

TYPED_TEST(SparseHermitianPart, MatchesReferenceOnAsymmetricPatterns) {
  using T = TypeParam;
  for (const double density : {0.01, 0.1, 0.5}) {
    SCOPED_TRACE(density);
    // General square matrices: many entries without a stored mirror, empty rows and rows whose
    // Hermitian-part row consists of inserted entries only.
    const CsrMatrix<T> a = random_sparse<T>(53, density, false, 300 + static_cast<int>(density * 100));
    const CsrMatrix<T> want = reference_hermitian_part(a);
    expect_bitwise_equal(hs::hermitian_part(a), want);
    CsrMatrix<T> consumed = a;
    expect_bitwise_equal(hs::hermitian_part(std::move(consumed)), want);
    EXPECT_EQ(consumed.nrows, 0);
    EXPECT_EQ(hs::hermitian_defect(want), 0.0);
  }
  // One triangle only: every off-diagonal entry is missing its mirror.
  const CsrMatrix<T> up = random_triangle<T>(30, 0.3, true, 31);
  expect_bitwise_equal(hs::hermitian_part(up), reference_hermitian_part(up));
  expect_bitwise_equal(hs::hermitian_part(CsrMatrix<T>(up)), reference_hermitian_part(up));
}

TYPED_TEST(SparseHermitianPart, SymmetricPatternIsUpdatedInPlace) {
  using T = TypeParam;
  CsrMatrix<T> a = random_sparse<T>(47, 0.2, true, 41);
  // Rounding-level asymmetry and (complex) non-real diagonal on an unchanged pattern.
  for (index_t p = 0; p < a.nnz(); ++p) a.values[p] *= 1.0 + 1e-13 * static_cast<double>(p % 7 - 3);
  if constexpr (hs::is_complex_v<T>) a.values[a.rowptr[5]] += T(0.0, 1e-9);
  const CsrMatrix<T> want = reference_hermitian_part(a);
  expect_bitwise_equal(hs::hermitian_part(a), want);
  CsrMatrix<T> consumed = a;
  const T* storage = consumed.values.data();
  const CsrMatrix<T> got = hs::hermitian_part(std::move(consumed));
  expect_bitwise_equal(got, want);
  EXPECT_EQ(got.values.data(), storage);  // no new value storage
  EXPECT_EQ(hs::hermitian_defect(got), 0.0);
}

TYPED_TEST(SparseHermitianPart, EdgeCases) {
  using T = TypeParam;
  CsrMatrix<T> empty;
  expect_bitwise_equal(hs::hermitian_part(empty), empty);
  expect_bitwise_equal(hs::hermitian_part(CsrMatrix<T>(empty)), empty);
  CsrMatrix<T> zero_rows;
  zero_rows.nrows = zero_rows.ncols = 3;
  zero_rows.rowptr.assign(4, 0);
  expect_bitwise_equal(hs::hermitian_part(zero_rows), zero_rows);
  expect_bitwise_equal(hs::hermitian_part(CsrMatrix<T>(zero_rows)), zero_rows);
  // Single orphan (2, 0) in an otherwise empty matrix.
  const std::vector<index_t> r{2}, c{0};
  const CsrMatrix<T> one = hs::csr_from_coo<T>(3, 3, r, c, {T(4)});
  for (const CsrMatrix<T>& h : {hs::hermitian_part(one), hs::hermitian_part(CsrMatrix<T>(one))}) {
    EXPECT_EQ(h.rowptr, (std::vector<index_t>{0, 1, 1, 2}));
    EXPECT_EQ(h.colind, (std::vector<std::int32_t>{2, 0}));
    EXPECT_EQ(h.values, (std::vector<T>{T(2), T(2)}));
  }
  CsrMatrix<T> wide;
  wide.nrows = 2;
  wide.ncols = 3;
  wide.rowptr.assign(3, 0);
  EXPECT_THROW(hs::hermitian_part(wide), hs::Error);
  EXPECT_THROW(hs::hermitian_part(CsrMatrix<T>(wide)), hs::Error);
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
