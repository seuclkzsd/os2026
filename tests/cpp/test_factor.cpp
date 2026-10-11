// Unit tests for src/factor (MUMPS backend): exact inertia, null-pivot (singular) detection,
// multi-RHS solve accuracy, refactorization with one analysis, error paths, estimates,
// out-of-core storage and compute_inertia. Analytic spectra wherever possible.
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "factor/mumps_api.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::FactorOptions;
using hs::FactorPrecision;
using hs::index_t;
using hs::ShiftedPencil;

// ------------------------------------------------------------------------------- helpers

// Number of entries of the ascending vector w that are < sigma.
index_t count_below(const std::vector<double>& w, double sigma) {
  return std::lower_bound(w.begin(), w.end(), sigma) - w.begin();
}

// `count` shifts placed in the middle of spectral gaps of the ascending spectrum w, spread over
// the whole spectrum; gaps narrower than min_gap (multiplicities, near-degeneracies) are skipped.
std::vector<double> mid_gap_shifts(const std::vector<double>& w, int count, double min_gap) {
  std::vector<double> s;
  const index_t n = static_cast<index_t>(w.size());
  for (int t = 1; t <= count; ++t) {
    for (index_t k = n * t / (count + 1); k < n; ++k) {
      if (w[k] - w[k - 1] > min_gap) {
        s.push_back(0.5 * (w[k - 1] + w[k]));
        break;
      }
    }
  }
  return s;
}

// Normwise backward error max_j ||K x_j - b_j||_2 / (||K|| ||x_j||_2) of a block solve, with the
// upper bound ||K||_1 >= ||K||_2 (Hermitian K) for ||K||.
template <class T>
double backward_error(const CsrMatrix<T>& k, const T* x, const T* b, index_t nrhs, index_t ld) {
  const index_t n = k.nrows;
  const double knorm = hs::norm_one(k);
  std::vector<T> r(static_cast<std::size_t>(n));
  double worst = 0.0;
  for (index_t j = 0; j < nrhs; ++j) {
    hs::spmv(k, x + j * ld, r.data());
    for (index_t i = 0; i < n; ++i) r[i] -= b[i + j * ld];
    worst = std::max(worst,
                     hs::linalg::norm2(n, r.data()) / (knorm * hs::linalg::norm2(n, x + j * ld)));
  }
  return worst;
}

// Solves K X = B for nrhs random right-hand sides stored with leading dimension n + 3 and
// returns the backward error.
template <class T>
double solve_backward_error(hs::SparseFactorization<T>& f, const ShiftedPencil<T>& pencil,
                            double alpha, double beta, index_t nrhs, std::uint64_t seed) {
  const index_t n = pencil.order(), ld = n + 3;
  std::vector<T> b(static_cast<std::size_t>(ld * nrhs), T(0));
  for (index_t j = 0; j < nrhs; ++j) hs::linalg::fill_random(b.data() + j * ld, n, seed, j);
  std::vector<T> x = b;
  f.solve(x.data(), nrhs, ld);
  return backward_error(pencil.assemble_full(alpha, beta), x.data(), b.data(), nrhs, ld);
}

// Saddle-point matrix [[M, C], [C^H, 0]] with M the SPD 2-D Laplacian on an m x m grid and C a
// random sparse m^2 x n2 block (full column rank with probability 1). Its inertia is exactly
// (n2 negative, 0, m^2 positive), and the zero block forces 2x2 / delayed pivots.
template <class T>
CsrMatrix<T> saddle_point(index_t m, index_t n2, int per_column, std::uint64_t seed) {
  const auto lap = hs::test::laplacian_2d<T>(m, m);
  const index_t n1 = lap.nrows;
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < n1; ++i)
    for (index_t p = lap.rowptr[i]; p < lap.rowptr[i + 1]; ++p) {
      r.push_back(i);
      c.push_back(lap.colind[p]);
      v.push_back(lap.values[p]);
    }
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<index_t> row(0, n1 - 1);
  for (index_t j = 0; j < n2; ++j)
    for (int t = 0; t < per_column; ++t) {
      const index_t i = row(g);
      T x;
      if constexpr (hs::is_complex_v<T>) {
        x = T(u(g), u(g));
      } else {
        x = u(g);
      }
      r.push_back(i), c.push_back(n1 + j), v.push_back(x);
      r.push_back(n1 + j), c.push_back(i), v.push_back(hs::conj_if(x));
    }
  return hs::csr_from_coo<T>(n1 + n2, n1 + n2, r, c, v);
}

// 2-D Laplacian whose couplings carry a phase on every fifth grid row only (a "magnetic"
// Laplacian): Hermitian, with a mix of purely real and genuinely complex off-diagonals.
CsrMatrix<cplx> partially_complex_laplacian(index_t nx, index_t ny) {
  std::vector<index_t> r, c;
  std::vector<cplx> v;
  auto add_edge = [&](index_t p, index_t q, cplx w) {
    r.push_back(p), c.push_back(q), v.push_back(w);
    r.push_back(q), c.push_back(p), v.push_back(std::conj(w));
  };
  for (index_t j = 0; j < ny; ++j)
    for (index_t i = 0; i < nx; ++i) {
      const index_t p = i + j * nx;
      r.push_back(p), c.push_back(p), v.push_back(cplx(4.0));
      const cplx w = (j % 5 == 0) ? -std::polar(1.0, 0.3 * static_cast<double>(i + 1)) : cplx(-1.0);
      if (i + 1 < nx) add_edge(p, p + 1, w);
      if (j + 1 < ny) add_edge(p, p + nx, cplx(-1.0));
    }
  return hs::csr_from_coo<cplx>(nx * ny, nx * ny, r, c, v);
}

// Spectrum with distinct, irregularly spaced eigenvalues in about [-2, 3].
std::vector<double> irregular_spectrum(index_t n) {
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[i] = -2.0 + 5.0 * (i + 0.5) / n + 0.4 * std::sin(1.3 * i) / n;
  std::sort(w.begin(), w.end());
  return w;
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

// Every typed suite runs for both public scalar types.
using Scalars = ::testing::Types<double, cplx>;
template <class T>
class FactorInertia : public ::testing::Test {};
TYPED_TEST_SUITE(FactorInertia, Scalars);
template <class T>
class FactorSingular : public ::testing::Test {};
TYPED_TEST_SUITE(FactorSingular, Scalars);
template <class T>
class FactorSolve : public ::testing::Test {};
TYPED_TEST_SUITE(FactorSolve, Scalars);
template <class T>
class FactorRefactorization : public ::testing::Test {};
TYPED_TEST_SUITE(FactorRefactorization, Scalars);
template <class T>
class FactorErrors : public ::testing::Test {};
TYPED_TEST_SUITE(FactorErrors, Scalars);
template <class T>
class FactorEstimate : public ::testing::Test {};
TYPED_TEST_SUITE(FactorEstimate, Scalars);
template <class T>
class FactorOutOfCore : public ::testing::Test {};
TYPED_TEST_SUITE(FactorOutOfCore, Scalars);
template <class T>
class FactorComputeInertia : public ::testing::Test {};
TYPED_TEST_SUITE(FactorComputeInertia, Scalars);

// ------------------------------------------------------------------------------- inertia

TYPED_TEST(FactorInertia, ShiftedLaplacian1DIsExact) {
  using T = TypeParam;
  const index_t n = 600;
  const auto a = hs::test::laplacian_1d<T>(n);
  const auto w = hs::test::laplacian_1d_eigenvalues(n);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -2.0);
  for (double sigma : mid_gap_shifts(w, 9, 1e-9)) {
    ASSERT_TRUE(f->factorize(1.0, -sigma)) << "sigma=" << sigma;
    const auto in = f->inertia();
    EXPECT_TRUE(in.valid);
    EXPECT_EQ(in.neg, count_below(w, sigma)) << "sigma=" << sigma;
    EXPECT_EQ(in.zero, 0);
    EXPECT_EQ(in.pos, n - count_below(w, sigma));
  }
}

TYPED_TEST(FactorInertia, ShiftedLaplacian2DIsExact) {
  using T = TypeParam;
  const index_t nx = 60, ny = 45;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -4.0);
  for (double sigma : mid_gap_shifts(w, 12, 1e-7)) {
    ASSERT_TRUE(f->factorize(1.0, -sigma)) << "sigma=" << sigma;
    EXPECT_TRUE(f->inertia().valid);
    EXPECT_EQ(f->inertia().neg, count_below(w, sigma)) << "sigma=" << sigma;
    EXPECT_EQ(f->inertia().zero, 0);
  }
}

// prescribed_spectrum<cplx> is genuinely complex Hermitian, so this exercises the embedding with
// a full -Im K block; the real instantiation checks the same counts for real symmetric input.
TYPED_TEST(FactorInertia, PrescribedSpectrumIsExact) {
  using T = TypeParam;
  const index_t n = 1500;
  const auto w = irregular_spectrum(n);
  const auto a = hs::test::prescribed_spectrum<T>(w, 12, 2026);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, 0.0);
  for (double sigma : mid_gap_shifts(w, 10, 1e-6)) {
    ASSERT_TRUE(f->factorize(1.0, -sigma)) << "sigma=" << sigma;
    EXPECT_TRUE(f->inertia().valid);
    EXPECT_EQ(f->inertia().neg, count_below(w, sigma)) << "sigma=" << sigma;
    EXPECT_EQ(f->inertia().pos, n - count_below(w, sigma));
  }
}

// With B HPD, nu_-(A - sigma B) is the number of generalized eigenvalues below sigma (Sylvester).
TYPED_TEST(FactorInertia, GeneralizedPencilMatchesDenseCount) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.03, 11);
  const auto b = hs::test::random_hpd<T>(n, 0.02, 12);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  ShiftedPencil<T> pencil(a, &b);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, 0.0);
  for (double sigma : mid_gap_shifts(w, 8, 1e-6)) {
    ASSERT_TRUE(f->factorize(1.0, -sigma));
    EXPECT_EQ(f->inertia().neg, count_below(w, sigma)) << "sigma=" << sigma;
  }
}

// B-definiteness test as the driver does it: inertia of 0*A + 1*B on the pencil's pattern.
TYPED_TEST(FactorInertia, OfBDetectsPositiveDefiniteness) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.03, 21);
  const auto hpd = hs::test::random_hpd<T>(n, 0.02, 22);
  const auto indefinite = hs::test::random_hermitian<T>(n, 0.02, 23);

  const auto in_hpd = hs::compute_inertia(ShiftedPencil<T>(a, &hpd), 0.0, 1.0);
  EXPECT_TRUE(in_hpd.valid && in_hpd.neg == 0 && in_hpd.zero == 0 && in_hpd.pos == n);

  const auto in_identity = hs::compute_inertia(ShiftedPencil<T>(a, nullptr), 0.0, 1.0);
  EXPECT_TRUE(in_identity.valid && in_identity.neg == 0 && in_identity.pos == n);

  const auto wb =
      hs::test::dense_eigenvalues(indefinite, static_cast<const CsrMatrix<T>*>(nullptr));
  const auto in_indef = hs::compute_inertia(ShiftedPencil<T>(a, &indefinite), 0.0, 1.0);
  EXPECT_TRUE(in_indef.valid);
  EXPECT_EQ(in_indef.neg, count_below(wb, 0.0));
  EXPECT_GT(in_indef.neg, 0);
  EXPECT_EQ(in_indef.zero, 0);
}

// The embedding leaves out real / imaginary parts that are zero in both A and B. Pencils where
// only A or only B carries imaginary parts exercise both halves of that rule.
TEST(FactorEmbedding, PartiallyComplexPencilsHaveExactInertiaAndSolves) {
  const auto a_mixed = partially_complex_laplacian(15, 20);
  const auto b_real = hs::to_complex(hs::test::random_hpd<double>(300, 0.02, 81));
  const auto a_real = hs::to_complex(hs::test::random_hermitian<double>(300, 0.03, 82));
  const auto b_cplx = hs::test::random_hpd<cplx>(300, 0.02, 83);
  for (auto [a, b] : {std::pair{&a_mixed, &b_real}, std::pair{&a_real, &b_cplx}}) {
    const auto w = hs::test::dense_eigenvalues(*a, b);
    ShiftedPencil<cplx> pencil(*a, b);
    auto f = hs::make_mumps_factorization<cplx>();
    f->analyze(pencil, 1.0, 0.0);
    for (double sigma : mid_gap_shifts(w, 6, 1e-6)) {
      ASSERT_TRUE(f->factorize(1.0, -sigma));
      EXPECT_EQ(f->inertia().neg, count_below(w, sigma)) << "sigma=" << sigma;
      EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -sigma, 2, 11), 1e-12);
    }
  }
}

// ------------------------------------------------------------------------------- singularity

// sigma exactly on a (double-rounded) eigenvalue of a 2-D Laplacian of order >= 2500: the null
// pivot is detected, factorize() returns false, the inertia is invalid, solve() refuses, and the
// object stays analysed so that a perturbed shift factorizes with the exact inertia.
// Detection depends on the elimination order (see the null-pivot note in factor.hpp); the
// analysis uses a generic first shift, as the driver does. Measured margin: every sample below is
// still detected with CNTL(3) = 1e-11, a tenth of the default.
TYPED_TEST(FactorSingular, ShiftOnLaplacianEigenvalueIsSingular) {
  using T = TypeParam;
  for (auto [nx, ny] : {std::pair<index_t, index_t>{50, 50}, {50, 51}, {60, 60}}) {
    const auto a = hs::test::laplacian_2d<T>(nx, ny);
    const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
    const index_t n = a.nrows;
    ShiftedPencil<T> pencil(a, nullptr);
    auto f = hs::make_mumps_factorization<T>();
    f->analyze(pencil, 1.0, -3.0);
    const int samples = 24;
    for (int t = 0; t < samples; ++t) {
      const double lambda = w[(n - 1) * t / (samples - 1)];
      EXPECT_FALSE(f->factorize(1.0, -lambda)) << nx << "x" << ny << " lambda=" << lambda;
      EXPECT_FALSE(f->inertia().valid);
      std::vector<T> b(static_cast<std::size_t>(n), T(1));
      EXPECT_EQ(error_code_of([&] { f->solve(b.data(), 1, n); }), hs::Error::Code::Internal);
      const double sigma = lambda + 1e-8 * 8.0;
      ASSERT_TRUE(f->factorize(1.0, -sigma)) << "perturbed shift " << sigma;
      EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
    }
    EXPECT_EQ(f->stats().analyses, 1);
  }
}

// Shifts at relative distance 1e-10 .. 1e-8 from an eigenvalue (relative to |lambda| and to
// ||K||_2 ~ 8) are NOT singular: factorization succeeds with the exact inertia.
TYPED_TEST(FactorSingular, NearSingularShiftHasExactInertia) {
  using T = TypeParam;
  const index_t nx = 50, ny = 51, n = nx * ny;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -4.0);
  const int samples = 12;
  for (int t = 1; t < samples; ++t) {
    const double lambda = w[(n - 1) * t / samples];
    for (double delta : {1e-10, 1e-9, 1e-8})
      for (double sign : {-1.0, 1.0})
        for (double sigma : {lambda * (1.0 + sign * delta), lambda + sign * delta * 8.0}) {
          ASSERT_TRUE(f->factorize(1.0, -sigma)) << "lambda=" << lambda << " sigma=" << sigma;
          EXPECT_EQ(f->inertia().neg, count_below(w, sigma))
              << "lambda=" << lambda << " sigma=" << sigma;
          EXPECT_EQ(f->stats().null_pivots, 0);
        }
  }
}

// A genuinely complex Hermitian matrix shifted onto one of its prescribed eigenvalues.
TEST(FactorSingularComplex, ShiftOnEigenvalueIsSingular) {
  const index_t n = 1200;
  const auto w = irregular_spectrum(n);
  const auto a = hs::test::prescribed_spectrum<cplx>(w, 8, 99);
  ShiftedPencil<cplx> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<cplx>();
  f->analyze(pencil, 1.0, 0.0);
  for (index_t k : {index_t(0), n / 4, n / 2, 3 * n / 4, n - 1}) {
    EXPECT_FALSE(f->factorize(1.0, -w[k])) << "lambda=" << w[k];
    EXPECT_FALSE(f->inertia().valid);
    const double sigma = w[k] + 1e-9;
    ASSERT_TRUE(f->factorize(1.0, -sigma));
    EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
  }
}

// ------------------------------------------------------------------------------- solves

// Measured backward errors: 1e-14 .. 2.3e-13 in double (up to ~1000 eps: element growth with
// MUMPS' threshold pivoting on random indefinite matrices).
TYPED_TEST(FactorSolve, MultiRhsIsBackwardStable) {
  using T = TypeParam;
  // Indefinite shifted Laplacian (standard problem) ...
  const auto lap = hs::test::laplacian_2d<T>(50, 50);
  ShiftedPencil<T> std_pencil(lap, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(std_pencil, 1.0, -3.3);
  ASSERT_TRUE(f->factorize(1.0, -3.3));
  EXPECT_LE(solve_backward_error(*f, std_pencil, 1.0, -3.3, 5, 1), 1e-12);

  // ... and an indefinite generalized pencil with HPD B.
  const auto a = hs::test::random_hermitian<T>(800, 0.01, 31);
  const auto b = hs::test::random_hpd<T>(800, 0.005, 32);
  ShiftedPencil<T> gen_pencil(a, &b);
  auto g = hs::make_mumps_factorization<T>();
  g->analyze(gen_pencil, 1.0, -0.2);
  ASSERT_TRUE(g->factorize(1.0, -0.2));
  EXPECT_LE(solve_backward_error(*g, gen_pencil, 1.0, -0.2, 7, 2), 1e-12);
  EXPECT_EQ(g->stats().solves, 1);
  EXPECT_EQ(g->stats().rhs_solved, 7);
}

// Measured: ~4e-6 for the shifted Laplacian, up to ~6e-5 for random indefinite pencils (more
// element growth), i.e. 70 .. 1000 eps_single.
TYPED_TEST(FactorSolve, SinglePrecisionHasSinglePrecisionAccuracy) {
  using T = TypeParam;
  FactorOptions opt;
  opt.precision = FactorPrecision::Single;

  const auto lap = hs::test::laplacian_2d<T>(50, 50);
  ShiftedPencil<T> std_pencil(lap, nullptr);
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(std_pencil, 1.0, -3.3);
  ASSERT_TRUE(f->factorize(1.0, -3.3));
  const double eta_lap = solve_backward_error(*f, std_pencil, 1.0, -3.3, 3, 3);
  EXPECT_LE(eta_lap, 2e-5);
  EXPECT_GE(eta_lap, 1e-9) << "a single-precision factor cannot be double-accurate";
  EXPECT_EQ(f->options().precision, FactorPrecision::Single);

  const auto a = hs::test::random_hermitian<T>(800, 0.01, 31);
  const auto b = hs::test::random_hpd<T>(800, 0.005, 32);
  ShiftedPencil<T> gen_pencil(a, &b);
  auto g = hs::make_mumps_factorization<T>(opt);
  g->analyze(gen_pencil, 1.0, 0.2);
  ASSERT_TRUE(g->factorize(1.0, 0.2));
  const double eta_gen = solve_backward_error(*g, gen_pencil, 1.0, 0.2, 4, 4);
  EXPECT_LE(eta_gen, 2e-4);
  EXPECT_GE(eta_gen, 1e-9);
}

TYPED_TEST(FactorSolve, BlockLowRankFactorizationSolvesAccurately) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(90, 80);
  const auto w = hs::test::laplacian_2d_eigenvalues(90, 80);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions opt;
  opt.blr = true;
  opt.blr_tol = 1e-10;
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(pencil, 1.0, -2.5);
  EXPECT_GT(f->estimate().blr_incore_mb, 0.0);
  const double sigma = mid_gap_shifts(w, 1, 1e-6)[0];
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
  EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -sigma, 2, 4), 1e-8);
}

TYPED_TEST(FactorSolve, ValidatesArguments) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_1d<T>(50);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -1.0);
  std::vector<T> b(100, T(1));
  // No factorization yet.
  EXPECT_EQ(error_code_of([&] { f->solve(b.data(), 1, 50); }), hs::Error::Code::Internal);
  ASSERT_TRUE(f->factorize(1.0, -1.01));
  EXPECT_EQ(error_code_of([&] { f->solve(b.data(), 1, 49); }), hs::Error::Code::InvalidInput);
  f->solve(b.data(), 0, 50);  // no right-hand side: no-op
  EXPECT_EQ(b[0], T(1));
  EXPECT_EQ(f->order(), 50);
}

// ------------------------------------------------------------------------------- refactorization

TYPED_TEST(FactorRefactorization, ManyShiftsReuseOneAnalysis) {
  using T = TypeParam;
  const index_t n = 400;
  const auto a = hs::test::random_hermitian<T>(n, 0.02, 51);
  const auto b = hs::test::random_hpd<T>(n, 0.01, 52);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  ShiftedPencil<T> pencil(a, &b);
  auto f = hs::make_mumps_factorization<T>();
  const auto shifts = mid_gap_shifts(w, 5, 1e-6);
  f->analyze(pencil, 1.0, -shifts[0]);
  for (double sigma : shifts) {
    ASSERT_TRUE(f->factorize(1.0, -sigma));
    EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
    EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -sigma, 3, 5), 1e-12);
  }
  const auto& s = f->stats();
  EXPECT_EQ(s.analyses, 1);
  EXPECT_EQ(s.factorizations, static_cast<index_t>(shifts.size()));
  EXPECT_EQ(s.solves, static_cast<index_t>(shifts.size()));
  EXPECT_EQ(s.rhs_solved, 3 * static_cast<index_t>(shifts.size()));
  EXPECT_GT(s.analyze_s, 0.0);
  EXPECT_GT(s.factorize_s, 0.0);
  EXPECT_GT(s.solve_s, 0.0);
  EXPECT_EQ(s.last_info1, 0);
}

// ------------------------------------------------------------------------------- error paths

// mem_relax_pct = 0 under-estimates the workspace of a saddle-point matrix whose zero block
// forces delayed pivots: MUMPS reports a workspace error, factorize() retries with a larger
// ICNTL(14) and then succeeds with the exact inertia.
TYPED_TEST(FactorErrors, WorkspaceErrorIsRetriedWithMoreRelaxation) {
  using T = TypeParam;
  const index_t m = 40, n2 = 400;
  const auto k = saddle_point<T>(m, n2, 3, 61);
  ShiftedPencil<T> pencil(k, nullptr);
  FactorOptions opt;
  opt.mem_relax_pct = 0;
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(pencil, 1.0, 0.0);
  ASSERT_TRUE(f->factorize(1.0, 0.0));
  EXPECT_GE(f->stats().workspace_retries, 1);
  EXPECT_GT(f->stats().delayed_pivots, 0);
  EXPECT_EQ(f->inertia().neg, n2);
  EXPECT_EQ(f->inertia().pos, m * m);
  EXPECT_LE(solve_backward_error(*f, pencil, 1.0, 0.0, 2, 6), 1e-12);
}

TYPED_TEST(FactorErrors, MemoryCapTooSmallThrowsOutOfMemoryBudget) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(100, 100);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions opt;
  opt.max_memory_mb = 1.0;  // ICNTL(23) = 1 MB, far below the estimate
  auto f = hs::make_mumps_factorization<T>(opt);
  f->analyze(pencil, 1.0, -3.3);
  ASSERT_GT(f->estimate().incore_mb, 1.0);
  EXPECT_EQ(error_code_of([&] { (void)f->factorize(1.0, -3.3); }),
            hs::Error::Code::OutOfMemoryBudget);
  EXPECT_FALSE(f->inertia().valid);
  EXPECT_EQ(f->stats().last_info1, -19);
}

TYPED_TEST(FactorErrors, InvalidOptionsAndMisuseAreRejected) {
  using T = TypeParam;
  auto bad = [](auto edit) {
    FactorOptions o;
    edit(o);
    return error_code_of([&] { hs::make_mumps_factorization<T>(o); });
  };
  using Code = hs::Error::Code;
  EXPECT_EQ(bad([](FactorOptions& o) { o.null_pivot_tol = 0.0; }), Code::InvalidInput);
  EXPECT_EQ(bad([](FactorOptions& o) { o.ordering = 1; }), Code::InvalidInput);
  EXPECT_EQ(bad([](FactorOptions& o) { o.scaling = -1; }), Code::InvalidInput);
  EXPECT_EQ(bad([](FactorOptions& o) { o.mem_relax_pct = -5; }), Code::InvalidInput);
  EXPECT_EQ(bad([](FactorOptions& o) { o.max_memory_mb = -1.0; }), Code::InvalidInput);
  EXPECT_EQ(bad([](FactorOptions& o) { o.ooc_dir = std::string(2000, 'x'); }), Code::InvalidInput);

  auto f = hs::make_mumps_factorization<T>();
  EXPECT_EQ(error_code_of([&] { (void)f->factorize(1.0, 0.0); }), Code::Internal);
  EXPECT_EQ(error_code_of([&] { (void)f->estimate(); }), Code::Internal);
  EXPECT_EQ(f->order(), 0);
  EXPECT_FALSE(f->inertia().valid);
}

// ------------------------------------------------------------------------------- estimates

TEST(FactorInfogDecoding, NegativeInfogCountsAreMillions) {
  EXPECT_EQ(hs::mumps::decode_count(123456), 123456);
  EXPECT_EQ(hs::mumps::decode_count(-2500), std::int64_t(2500) * 1000000);
  EXPECT_EQ(hs::mumps::decode_count(0), 0);
}

TYPED_TEST(FactorEstimate, FieldsArePlausible) {
  using T = TypeParam;
  const index_t nx = 120, ny = 100, n = nx * ny;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  const index_t mumps_order = hs::is_complex_v<T> ? 2 * n : n;

  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -3.3);
  const auto e = f->estimate();
  EXPECT_GT(e.factor_entries, pencil.nnz());
  EXPECT_LT(e.factor_entries, mumps_order * mumps_order / 10);
  EXPECT_GT(e.incore_mb, 0.0);
  EXPECT_GT(e.ooc_mb, 0.0);
  EXPECT_LE(e.ooc_mb, e.incore_mb);
  EXPECT_GT(e.flops, static_cast<double>(e.factor_entries));
  EXPECT_EQ(e.blr_incore_mb, -1.0);
  EXPECT_EQ(e.blr_ooc_mb, -1.0);
  EXPECT_EQ(f->stats().estimate.factor_entries, e.factor_entries);

  ASSERT_TRUE(f->factorize(1.0, -3.3));
  const auto& s = f->stats();
  EXPECT_GE(s.factor_entries, e.factor_entries / 2);
  EXPECT_LE(s.factor_entries, 2 * e.factor_entries);
  EXPECT_GT(s.factor_mb_used, 0.0);
  EXPECT_LE(s.factor_mb_used, 2.0 * e.incore_mb);

  // BLR at analysis only: estimates INFOG(36), INFOG(38) become available.
  FactorOptions opt;
  opt.blr_estimates = true;
  auto g = hs::make_mumps_factorization<T>(opt);
  g->analyze(pencil, 1.0, -3.3);
  // (BLR bookkeeping can exceed the full-rank estimate on problems this small.)
  EXPECT_GT(g->estimate().blr_incore_mb, 0.0);
  EXPECT_GT(g->estimate().blr_ooc_mb, 0.0);
  ASSERT_TRUE(g->factorize(1.0, -3.3));  // full-rank factorization after a BLR analysis
  EXPECT_LE(solve_backward_error(*g, pencil, 1.0, -3.3, 1, 7), 1e-12);
}

// ------------------------------------------------------------------------------- out-of-core

TYPED_TEST(FactorOutOfCore, FactorsSolveAndAreCleanedUp) {
  using T = TypeParam;
  namespace fs = std::filesystem;
  // ctest runs the tests in the build directory, so the factor files stay under build/.
  const fs::path dir = fs::current_path() / ("factor_ooc_" + std::to_string(::getpid()) + "_" +
                                             (hs::is_complex_v<T> ? "z" : "d"));
  fs::create_directories(dir);
  const auto a = hs::test::laplacian_2d<T>(80, 70);
  ShiftedPencil<T> pencil(a, nullptr);
  {
    FactorOptions opt;
    opt.out_of_core = true;
    opt.ooc_dir = dir.string();
    auto f = hs::make_mumps_factorization<T>(opt);
    f->analyze(pencil, 1.0, -3.3);
    ASSERT_TRUE(f->factorize(1.0, -3.3));
    EXPECT_FALSE(fs::is_empty(dir)) << "factors were not written out of core";
    EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -3.3, 3, 8), 1e-12);
    f->set_out_of_core(false);
    EXPECT_FALSE(f->options().out_of_core);
    ASSERT_TRUE(f->factorize(1.0, -3.4));
    EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -3.4, 3, 9), 1e-12);
    f->set_out_of_core(true);
    ASSERT_TRUE(f->factorize(1.0, -3.5));
    EXPECT_LE(solve_backward_error(*f, pencil, 1.0, -3.5, 3, 10), 1e-12);
  }
  EXPECT_TRUE(fs::is_empty(dir)) << "out-of-core files left behind in " << dir;
  fs::remove_all(dir);
}

// ------------------------------------------------------------------------------- compute_inertia

// compute_inertia must give the exact inertia even when the options ask for a single-precision
// BLR factor, and must report (not throw) singularity.
TYPED_TEST(FactorComputeInertia, IgnoresSingleAndBlrAndNeverThrowsOnSingularity) {
  using T = TypeParam;
  const index_t nx = 50, ny = 51;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions lossy;
  lossy.precision = FactorPrecision::Single;
  lossy.blr = true;
  lossy.blr_tol = 1e-2;
  const double lambda = w[w.size() / 3];
  for (double sigma : {lambda * (1 + 1e-9), lambda * (1 - 1e-9)}) {
    const auto in = hs::compute_inertia(pencil, 1.0, -sigma, lossy);
    EXPECT_TRUE(in.valid);
    EXPECT_EQ(in.neg, count_below(w, sigma));
  }
  hs::Inertia singular;
  EXPECT_NO_THROW(singular = hs::compute_inertia(pencil, 1.0, -lambda, lossy));
  EXPECT_FALSE(singular.valid);
}

}  // namespace
