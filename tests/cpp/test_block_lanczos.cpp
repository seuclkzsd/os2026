// Tests of the block orthonormalization kernel (src/eigen/block_ortho) of the block Krylov-Schur
// solver.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "eigen/block_ortho.hpp"
#include "hsolve/error.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::index_t;
using hs::LinearOperator;
using hs::eigen_detail::BlockOrthonormalizer;
using hs::test::DenseOperator;

constexpr double kEps = std::numeric_limits<double>::epsilon();

template <class T>
DenseMatrix<T> mat_mul(const DenseMatrix<T>& a, const DenseMatrix<T>& b) {
  DenseMatrix<T> c(a.rows(), b.cols());
  hs::linalg::multiply(a.rows(), a.cols(), b.cols(), a.data(), a.rows(), b.data(), b.rows(),
                       c.data(), c.rows());
  return c;
}

/// M x (copy when m == nullptr).
template <class T>
DenseMatrix<T> times_m(LinearOperator<T>* m, const DenseMatrix<T>& x) {
  if (!m) return x;
  DenseMatrix<T> y(x.rows(), x.cols());
  m->apply(x.data(), x.rows(), y.data(), y.rows(), x.cols());
  return y;
}

/// max |X^H M X - I|.
template <class T>
double orthonormality_error(LinearOperator<T>* m, const DenseMatrix<T>& x) {
  const DenseMatrix<T> mx = times_m(m, x);
  double dev = 0.0;
  for (index_t j = 0; j < x.cols(); ++j)
    for (index_t i = 0; i < x.cols(); ++i) {
      const T g = hs::linalg::dot(x.rows(), x.col(i), mx.col(j));
      dev = std::max(dev, std::abs(g - T(i == j ? 1.0 : 0.0)));
    }
  return dev;
}

/// The k + p columns [V W] of a basis: V M-orthonormal (from a random block, Gram-Schmidt in
/// the M inner product) and W = `w` (n x p).
template <class T>
DenseMatrix<T> basis_with_block(LinearOperator<T>* m, index_t n, index_t k,
                                const DenseMatrix<T>& w) {
  DenseMatrix<T> b(n, k + w.cols());
  std::uint64_t stream = 0;
  for (index_t j = 0; j < k; ++j) {
    hs::linalg::fill_random(b.col(j), n, 99, stream++);
    std::vector<T> h(static_cast<std::size_t>(j) + 1), mw(static_cast<std::size_t>(n));
    hs::linalg::ApplyFn<T> fn = [m, n](const T* x, T* y) { m->apply(x, n, y, n, 1); };
    const double nrm =
        hs::linalg::orthogonalize_cgs2<T>(n, j, b.data(), n, nullptr, n, b.col(j),
                                          m ? mw.data() : nullptr, h.data(), m ? &fn : nullptr);
    hs::linalg::scal(n, T(1.0 / nrm), b.col(j));
  }
  std::copy(w.data(), w.data() + n * w.cols(), b.col(k));
  return b;
}

template <class T>
DenseMatrix<T> random_block(index_t n, index_t p, std::uint64_t seed) {
  DenseMatrix<T> w(n, p);
  for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(w.col(j), n, seed, j);
  return w;
}

enum class Mode { Euclidean, Recompute, Stored };

struct OrthoOutcome {
  double ortho = 0.0;     ///< max |[V Q]^H M [V Q] - I|
  double relation = 0.0;  ///< ||W - V C - Q R||_max / ||W||_max
  double lower = 0.0;     ///< max |R(i, j)|, i > j
  double stored = 0.0;    ///< store mode: max |stored M Q - M Q| / max |M Q|
  bool fallback = false;
  std::vector<char> deficient;
  DenseMatrix<double> rdiag;  ///< |R(i, i)|
};

/// Runs the kernel on [V W] and checks its output against its contract.
template <class T>
OrthoOutcome run_kernel(Mode mode, LinearOperator<T>* m_op, index_t k, const DenseMatrix<T>& w,
                        double ratio = 64 * kEps) {
  const index_t n = w.rows(), p = w.cols();
  LinearOperator<T>* m = mode == Mode::Euclidean ? nullptr : m_op;
  DenseMatrix<T> basis = basis_with_block<T>(m, n, k, w);
  DenseMatrix<T> mbasis;
  if (mode == Mode::Stored) mbasis = times_m(m, basis);  // slot columns are scratch
  BlockOrthonormalizer<T> ortho(n, p, m, mode == Mode::Stored);
  std::uint64_t stream = 1000;
  DenseMatrix<T> coef, r;
  ortho.run(basis.data(), mode == Mode::Stored ? mbasis.data() : nullptr, k, ratio, 7, stream, coef,
            r);
  OrthoOutcome out;
  out.fallback = ortho.used_fallback();
  out.deficient = ortho.deficient();
  out.ortho = orthonormality_error(m, basis);
  // W - V C - Q R
  DenseMatrix<T> v(n, k), q(n, p);
  std::copy(basis.data(), basis.data() + n * k, v.data());
  std::copy(basis.col(k), basis.col(k) + n * p, q.data());
  DenseMatrix<T> vc = mat_mul(v, coef), qr = mat_mul(q, r);
  double wmax = 0.0, res = 0.0;
  for (index_t j = 0; j < p; ++j)
    for (index_t i = 0; i < n; ++i) {
      wmax = std::max(wmax, std::abs(w(i, j)));
      res = std::max(res, std::abs(w(i, j) - (k > 0 ? vc(i, j) : T(0)) - qr(i, j)));
    }
  out.relation = res / wmax;
  out.rdiag.resize(p, 1);
  for (index_t j = 0; j < p; ++j) {
    out.rdiag(j, 0) = std::abs(r(j, j));
    for (index_t i = j + 1; i < p; ++i) out.lower = std::max(out.lower, std::abs(r(i, j)));
  }
  if (mode == Mode::Stored) {
    const DenseMatrix<T> mq = times_m(m, q);
    double dmax = 0.0, mmax = 0.0;
    for (index_t j = 0; j < p; ++j)
      for (index_t i = 0; i < n; ++i) {
        dmax = std::max(dmax, std::abs(mbasis(i, k + j) - mq(i, j)));
        mmax = std::max(mmax, std::abs(mq(i, j)));
      }
    out.stored = dmax / mmax;
  }
  return out;
}

template <class T>
class LanczosBlockOrthoTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LanczosBlockOrthoTyped, Scalars);

constexpr Mode kModes[] = {Mode::Euclidean, Mode::Recompute, Mode::Stored};

TYPED_TEST(LanczosBlockOrthoTyped, GenericBlockTakesTheBlas3Path) {
  using T = TypeParam;
  const index_t n = 300, k = 17, p = 5;
  DenseOperator<T> m(hs::test::to_dense(hs::test::random_hpd<T>(n, 0.04, 3)));
  for (Mode mode : kModes) {
    SCOPED_TRACE(static_cast<int>(mode));
    const auto o = run_kernel<T>(mode, &m, k, random_block<T>(n, p, 11));
    EXPECT_FALSE(o.fallback);
    EXPECT_LT(o.ortho, 1e-14);
    EXPECT_LT(o.relation, 1e-14);
    EXPECT_EQ(o.lower, 0.0);
    EXPECT_LT(o.stored, 1e-14);
    for (char c : o.deficient) EXPECT_EQ(c, 0);
  }
  // No basis at all (start block): pure Cholesky-QR2.
  const auto o = run_kernel<T>(Mode::Recompute, &m, 0, random_block<T>(n, p, 12));
  EXPECT_FALSE(o.fallback);
  EXPECT_LT(o.ortho, 1e-14);
  EXPECT_LT(o.relation, 1e-14);
}

TYPED_TEST(LanczosBlockOrthoTyped, BlockNearlyInsideTheBasisStaysOrthogonalToIt) {
  // W = V X + 1e-9 * (random): W1 is a billion times smaller than W, the regime of a converging
  // Lanczos block. Cholesky-QR amplifies the first pass's loss of orthogonality to V; the second
  // pass must remove it.
  using T = TypeParam;
  const index_t n = 400, k = 24, p = 4;
  DenseOperator<T> m(hs::test::to_dense(hs::test::random_hpd<T>(n, 0.03, 5)));
  for (Mode mode : kModes) {
    SCOPED_TRACE(static_cast<int>(mode));
    // The same V as run_kernel builds (deterministic).
    LinearOperator<T>* mm = mode == Mode::Euclidean ? nullptr : &m;
    const DenseMatrix<T> v = basis_with_block<T>(mm, n, k, DenseMatrix<T>(n, 0));
    DenseMatrix<T> w = mat_mul(v, random_block<T>(k, p, 21));
    const DenseMatrix<T> noise = random_block<T>(n, p, 22);
    for (index_t j = 0; j < p; ++j) hs::linalg::axpy(n, T(1e-9), noise.col(j), w.col(j));
    const auto o = run_kernel<T>(mode, &m, k, w);
    EXPECT_FALSE(o.fallback);
    EXPECT_LT(o.ortho, 1e-14);
    EXPECT_LT(o.relation, 1e-14);
  }
}

TYPED_TEST(LanczosBlockOrthoTyped, IllConditionedBlockFallsBackColumnByColumn) {
  // Two columns 1e-8 apart (scaled condition ~1e8 > kMaxCholQrCond): Cholesky-QR could not
  // restore orthogonality, so the column-wise path runs; nothing is dependent at rounding level.
  using T = TypeParam;
  const index_t n = 250, k = 9, p = 4;
  DenseOperator<T> m(hs::test::to_dense(hs::test::random_hpd<T>(n, 0.05, 6)));
  for (Mode mode : kModes) {
    SCOPED_TRACE(static_cast<int>(mode));
    DenseMatrix<T> w = random_block<T>(n, p, 31);
    std::copy(w.col(0), w.col(0) + n, w.col(2));
    w(5, 2) += T(1e-8);
    const auto o = run_kernel<T>(mode, &m, k, w);
    EXPECT_TRUE(o.fallback);
    EXPECT_LT(o.ortho, 1e-14);
    EXPECT_LT(o.relation, 1e-14);
    EXPECT_EQ(o.lower, 0.0);
    EXPECT_LT(o.stored, 1e-14);
    for (char c : o.deficient) EXPECT_EQ(c, 0);
    EXPECT_GT(o.rdiag(2, 0), 0.0);
  }
}

TYPED_TEST(LanczosBlockOrthoTyped, DependentColumnsAreReplacedByRandomVectors) {
  // Column 1 lies in span(V), column 3 = column 0 + column 2 exactly: both are dependent at
  // rounding level, get R(i, i) = 0 and fresh random directions; the relation W = V C + Q R still
  // holds and [V Q] is M-orthonormal.
  using T = TypeParam;
  const index_t n = 200, k = 12, p = 4;
  DenseOperator<T> m(hs::test::to_dense(hs::test::random_hpd<T>(n, 0.05, 8)));
  for (Mode mode : kModes) {
    SCOPED_TRACE(static_cast<int>(mode));
    LinearOperator<T>* mm = mode == Mode::Euclidean ? nullptr : &m;
    const DenseMatrix<T> v = basis_with_block<T>(mm, n, k, DenseMatrix<T>(n, 0));
    DenseMatrix<T> w = random_block<T>(n, p, 41);
    const DenseMatrix<T> in_span = mat_mul(v, random_block<T>(k, 1, 42));
    std::copy(in_span.col(0), in_span.col(0) + n, w.col(1));
    for (index_t i = 0; i < n; ++i) w(i, 3) = w(i, 0) + w(i, 2);
    const auto o = run_kernel<T>(mode, &m, k, w);
    EXPECT_TRUE(o.fallback);
    ASSERT_EQ(o.deficient.size(), 4u);
    EXPECT_EQ(o.deficient[0], 0);
    EXPECT_EQ(o.deficient[1], 1);
    EXPECT_EQ(o.deficient[2], 0);
    EXPECT_EQ(o.deficient[3], 1);
    EXPECT_EQ(o.rdiag(1, 0), 0.0);
    EXPECT_EQ(o.rdiag(3, 0), 0.0);
    EXPECT_LT(o.ortho, 1e-14);
    EXPECT_LT(o.relation, 1e-13);
    EXPECT_LT(o.stored, 1e-14);
  }
}

TYPED_TEST(LanczosBlockOrthoTyped, NonFiniteOrIndefiniteInputThrowsInvalidInput) {
  using T = TypeParam;
  const index_t n = 60, k = 0, p = 3;  // start-block situation: no basis yet
  auto expect_invalid = [&](LinearOperator<T>* m, bool store, const DenseMatrix<T>& w,
                            const char* topic) {
    DenseMatrix<T> basis = basis_with_block<T>(nullptr, n, k, w);
    DenseMatrix<T> mbasis;
    if (store) mbasis = times_m(m, basis);
    BlockOrthonormalizer<T> ortho(n, p, m, store);
    std::uint64_t stream = 0;
    DenseMatrix<T> coef, r;
    try {
      ortho.run(basis.data(), store ? mbasis.data() : nullptr, k, 64 * kEps, 1, stream, coef, r);
      ADD_FAILURE() << "no exception (" << topic << ")";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
      EXPECT_NE(std::string(e.what()).find(topic), std::string::npos) << e.what();
    }
  };
  DenseMatrix<T> w = random_block<T>(n, p, 51);
  w(7, 1) = T(std::numeric_limits<double>::quiet_NaN());
  expect_invalid(nullptr, false, w, "non-finite");
  // M = -I: every M-norm is imaginary, random replacements cannot help.
  DenseMatrix<T> neg(n, n);
  for (index_t i = 0; i < n; ++i) neg(i, i) = T(-1.0);
  DenseOperator<T> mneg(neg);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store);
    expect_invalid(&mneg, store, random_block<T>(n, p, 52), "positive definite");
  }
}

}  // namespace
