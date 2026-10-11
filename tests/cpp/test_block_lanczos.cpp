// Tests of the building blocks of the block Krylov-Schur mode of hs::lanczos_krylov_schur: the
// block orthonormalization kernel (src/eigen/block_ortho), the block basis layout, and the
// rejection of block_size by krylov_schur_general. The solver itself is tested next to the
// single-vector solver in test_lanczos.cpp (suite LanczosBlockTyped).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "eigen/block_lanczos.hpp"
#include "eigen/block_ortho.hpp"
#include "hsolve/eigen.hpp"
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

TYPED_TEST(LanczosBlockOrthoTyped, BlockInsideANearlyOrthonormalBasisIsReplaced) {
  // Review finding (P3 blockks): V M-orthonormal only to `loss` >> eps (as a deflation block
  // accepted by the 1e-8 check), W = V X entirely inside span(V). The first pass leaves a
  // remainder of size ~loss made of V components, far above the rounding threshold, so the
  // Cholesky-QR path normalizes it and the second pass removes it again. Every column must come
  // out dependent and replaced by a random direction (the first-pass-only test used to throw
  // InvalidInput "b_inner is not Hermitian positive definite" here).
  using T = TypeParam;
  const index_t n = 200, k = 40, p = 4;
  DenseOperator<T> m(hs::test::to_dense(hs::test::random_hpd<T>(n, 0.05, 9)));
  for (Mode mode : kModes) {
    for (double loss : {3e-14, 1e-12, 1e-10}) {
      SCOPED_TRACE("mode " + std::to_string(static_cast<int>(mode)) + ", loss " +
                   std::to_string(loss));
      LinearOperator<T>* mm = mode == Mode::Euclidean ? nullptr : &m;
      DenseMatrix<T> basis = basis_with_block<T>(mm, n, k, DenseMatrix<T>(n, p));
      const DenseMatrix<T> noise = random_block<T>(n, k, 62);
      for (index_t j = 0; j < k; ++j)
        hs::linalg::axpy(n, T(loss / std::sqrt(static_cast<double>(n))), noise.col(j),
                         basis.col(j));
      DenseMatrix<T> v(n, k);
      std::copy(basis.data(), basis.data() + n * k, v.data());
      const DenseMatrix<T> w = mat_mul(v, random_block<T>(k, p, 63));
      std::copy(w.data(), w.data() + n * p, basis.col(k));
      DenseMatrix<T> mbasis;
      if (mode == Mode::Stored) mbasis = times_m(mm, basis);
      BlockOrthonormalizer<T> ortho(n, p, mm, mode == Mode::Stored);
      std::uint64_t stream = 100;
      DenseMatrix<T> coef, r;
      ortho.run(basis.data(), mode == Mode::Stored ? mbasis.data() : nullptr, k, 64 * kEps, 7,
                stream, coef, r);
      EXPECT_TRUE(ortho.used_fallback());
      for (char c : ortho.deficient()) EXPECT_EQ(c, 1);
      EXPECT_EQ(stream, 100u + static_cast<std::uint64_t>(p));
      DenseMatrix<T> q(n, p);
      std::copy(basis.col(k), basis.col(k) + n * p, q.data());
      EXPECT_LT(orthonormality_error(mm, q), 1e-14);
      // Q is orthogonal to V to second order in the loss.
      const DenseMatrix<T> mq = times_m(mm, q);
      double cross = 0.0;
      for (index_t j = 0; j < p; ++j)
        for (index_t i = 0; i < k; ++i)
          cross = std::max(cross, std::abs(hs::linalg::dot(n, v.col(i), mq.col(j))));
      EXPECT_LT(cross, 1e-13);
      // W = V C + Q R with R(i, i) = 0 (only rounding-level couplings).
      const DenseMatrix<T> vc = mat_mul(v, coef), qr = mat_mul(q, r);
      double res = 0.0, wmax = 0.0;
      for (index_t j = 0; j < p; ++j) {
        EXPECT_EQ(r(j, j), T(0));
        for (index_t i = 0; i < n; ++i) {
          wmax = std::max(wmax, std::abs(w(i, j)));
          res = std::max(res, std::abs(w(i, j) - vc(i, j) - qr(i, j)));
        }
      }
      EXPECT_LT(res / wmax, 1e-13);
      if (mode == Mode::Stored) {
        double dmax = 0.0;
        for (index_t j = 0; j < p; ++j)
          for (index_t i = 0; i < n; ++i)
            dmax = std::max(dmax, std::abs(mbasis(i, k + j) - mq(i, j)));
        EXPECT_LT(dmax, 1e-13);
      }
    }
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

TEST(LanczosBlockLayout, BasisSizesAreWholeBlocks) {
  using hs::eigen_detail::block_dimensions;
  using hs::eigen_detail::LanczosSetup;
  auto setup = [](index_t n_free, index_t nev, index_t ncv) {
    LanczosSetup s;
    s.n = s.n_free = n_free;
    s.nev = nev;
    s.ncv = ncv;
    return s;
  };
  // Driver-like sizes: single-vector auto ncv = max(2 nev + 1, nev + 16) = 28 for nev = 12.
  // Requested ncv: whole blocks only, at least round_up(nev + 1, p) + 2p. Auto: 4 more blocks.
  struct Case {
    index_t nev, ncv1, p;
    bool automatic;
    index_t ncv, keep, first_probe;
  };
  const Case cases[] = {
      {12, 28, 2, false, 28, 20, 28},  // 12 + 16 / 2 = 20
      {12, 28, 2, true, 36, 24, 28},   // 28 + 4 * 2; 12 + 24 / 2 = 24
      {12, 28, 4, false, 28, 20, 28},  //
      {12, 28, 4, true, 44, 28, 28},   // 28 + 4 * 4; 12 + 32 / 2 = 28
      {12, 28, 8, false, 32, 16, 32},  // 28 -> 32; 12 + 20 / 2 = 22 -> 16
      {12, 28, 8, true, 64, 32, 32},   // 32 + 4 * 8; 12 + 52 / 2 = 38 -> 32
      {10, 26, 4, false, 28, 16, 28},  // 26 -> 28; 10 + 18 / 2 = 19 -> 16
      {10, 26, 4, true, 44, 24, 28},   // 28 + 16; 10 + 34 / 2 = 27 -> 24
      // Small requested ncv: grows to spare + 2p, spare = round_up(nev + 1, p); keep >= spare.
      {5, 7, 4, false, 16, 8, 16},     // spare 8; 5 + 11 / 2 = 10 -> 8
      {7, 10, 8, false, 24, 8, 24},    // spare 8; 7 + 17 / 2 = 15 -> 8
      {12, 15, 3, false, 21, 15, 21},  // spare 15; 12 + 9 / 2 = 16 -> 15
      {12, 15, 4, false, 24, 16, 24},  // spare 16 (keep == nev = 12 before); 12 + 6 -> 16
      {8, 24, 8, true, 64, 32, 32},    // spare 16: base 32 (> 24), + 4 * 8
  };
  for (const Case& c : cases) {
    SCOPED_TRACE("nev " + std::to_string(c.nev) + " ncv " + std::to_string(c.ncv1) + " p " +
                 std::to_string(c.p) + (c.automatic ? " auto" : " requested"));
    const auto d = block_dimensions(setup(100000, c.nev, c.ncv1), c.p, c.automatic);
    EXPECT_EQ(d.ncv, c.ncv);
    EXPECT_EQ(d.keep, c.keep);
    EXPECT_EQ(d.first_probe, c.first_probe);
    EXPECT_TRUE(d.feasible);
  }
  for (bool automatic : {false, true}) {
    for (index_t p : {2, 3, 4, 8}) {
      for (index_t nev : {1, 3, 7, 8, 12, 30}) {
        for (index_t ncv1 : {nev + 1, nev + 3, std::max(2 * nev + 1, nev + 16)}) {
          const auto d = block_dimensions(setup(5000, nev, ncv1), p, automatic);
          EXPECT_TRUE(d.feasible);
          EXPECT_EQ(d.ncv % p, 0);
          EXPECT_EQ(d.keep % p, 0);
          EXPECT_GE(d.keep, nev + 1);           // a spare Ritz pair
          EXPECT_LE(d.keep + p, d.ncv);         // a new block per cycle
          EXPECT_GE(d.ncv, nev + 1 + 2 * p);    // room for two
          EXPECT_LE(d.first_probe, d.ncv);
          EXPECT_GE(d.first_probe, ncv1);       // never less evidence than p = 1
        }
      }
    }
  }
  // The free space must hold twice the basis plus its residual block (else the dispatcher takes
  // the exact single-vector cycle); the auto extension shrinks to fit; a full-space ncv never
  // takes blocks.
  auto d = block_dimensions(setup(48, 4, 20), 4, false);
  EXPECT_TRUE(d.feasible);  // 2 (20 + 4) <= 48
  EXPECT_EQ(d.ncv, 20);
  EXPECT_FALSE(block_dimensions(setup(47, 4, 20), 4, false).feasible);
  d = block_dimensions(setup(64, 4, 20), 4, true);
  EXPECT_TRUE(d.feasible);  // 20 + 16 shrinks to 28: 2 (28 + 4) <= 64
  EXPECT_EQ(d.ncv, 28);
  EXPECT_FALSE(block_dimensions(setup(47, 4, 20), 4, true).feasible);   // 2 (20 + 4) > 47
  EXPECT_FALSE(block_dimensions(setup(20, 4, 20), 2, false).feasible);  // ncv == n_free
  // Large block sizes: infeasible without overflowing (p up to INT64_MAX); the largest feasible
  // p for n_free = 100, nev = 4, ncv = 20 is 12 (base 36, 2 (36 + 12) <= 100).
  d = block_dimensions(setup(100, 4, 20), 12, false);
  EXPECT_TRUE(d.feasible);
  EXPECT_EQ(d.ncv, 36);
  const index_t big = std::numeric_limits<index_t>::max();
  for (index_t p : {index_t{13}, index_t{1000}, index_t{1} << 40, index_t{1} << 62, big / 2 - 1,
                    big / 2, big - 1, big}) {
    for (bool automatic : {false, true}) {
      SCOPED_TRACE("p " + std::to_string(p));
      EXPECT_FALSE(block_dimensions(setup(100, 4, 20), p, automatic).feasible);
      // A free space of INT64_MAX: feasible up to p = n_free / 8, no arithmetic overflows.
      const auto huge = block_dimensions(setup(big, 4, 20), p, automatic);
      EXPECT_EQ(huge.feasible, p <= big / 8);
      if (huge.feasible) EXPECT_EQ(huge.ncv % p, 0);
    }
  }
}

TEST(KrylovSchurGeneralBlock, BlockSizeOtherThanOneIsRejected) {
  hs::test::DiagonalOperator<double> op(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  hs::KrylovSchurOptions<double> opt;
  opt.nev = 2;
  for (index_t p : {0, 2, 4}) {
    opt.block_size = p;
    try {
      hs::krylov_schur_general<double>(op, hs::Which::LargestMagnitude, opt);
      ADD_FAILURE() << "block_size " << p << " accepted";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  }
  opt.block_size = 1;
  EXPECT_TRUE(hs::krylov_schur_general<double>(op, hs::Which::LargestMagnitude, opt).converged);
}

}  // namespace
