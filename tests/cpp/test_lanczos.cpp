// Tests of hs::lanczos_krylov_schur (thick-restart Lanczos / Hermitian Krylov-Schur) on operators
// with known spectra, for both scalar types.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "eigen/block_lanczos.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::HermitianEigResult;
using hs::index_t;
using hs::KrylovSchurOptions;
using hs::LinearOperator;
using hs::Which;
using hs::test::DenseOperator;

constexpr Which kAllWhich[] = {Which::LargestMagnitude, Which::LargestAlgebraic,
                               Which::SmallestAlgebraic};

// ---------------------------------------------------------------------------------- helpers

/// a is ranked no later than b under `which`.
bool ranked_before(double a, double b, Which which) {
  switch (which) {
    case Which::SmallestAlgebraic:
      return a <= b;
    case Which::LargestAlgebraic:
      return a >= b;
    case Which::LargestMagnitude:
      return std::abs(a) >= std::abs(b);
  }
  return false;
}

/// The nev values of `spectrum` the solver should return, in ranking order.
std::vector<double> wanted(std::vector<double> spectrum, Which which, index_t nev) {
  std::stable_sort(spectrum.begin(), spectrum.end(), [which](double a, double b) {
    return ranked_before(a, b, which) && !ranked_before(b, a, which);
  });
  spectrum.resize(static_cast<std::size_t>(nev));
  return spectrum;
}

/// Dense operator with exactly the given eigenvalues (random unitary blocks, permuted).
template <class T>
DenseOperator<T> operator_with_spectrum(const std::vector<double>& eigenvalues,
                                        std::uint64_t seed) {
  return DenseOperator<T>(
      hs::test::to_dense(hs::test::prescribed_spectrum<T>(eigenvalues, 8, seed)));
}

/// ||P (Op x_i - theta_i x_i)||_M for every returned pair (M = I when m == nullptr), where
/// P = I - D D^H M projects out the deflation block (P = I without one): with deflation the solver
/// works on the deflated operator, whose residuals exclude the components along span(D).
template <class T>
std::vector<double> true_residuals(LinearOperator<T>& op, LinearOperator<T>* m,
                                   const HermitianEigResult<T>& r, const DenseMatrix<T>* d) {
  const index_t n = op.size(), k = r.vectors.cols();
  DenseMatrix<T> res(n, k), mres(n, k);
  op.apply(r.vectors.data(), n, res.data(), n, k);
  for (index_t j = 0; j < k; ++j) hs::linalg::axpy(n, T(-r.theta[j]), r.vectors.col(j), res.col(j));
  auto apply_m = [&] {
    if (m) {
      m->apply(res.data(), n, mres.data(), n, k);
    } else {
      mres = res;
    }
  };
  apply_m();
  if (d && d->cols() > 0) {
    for (index_t j = 0; j < k; ++j)
      for (index_t c = 0; c < d->cols(); ++c)
        hs::linalg::axpy(n, -hs::linalg::dot(n, d->col(c), mres.col(j)), d->col(c), res.col(j));
    apply_m();
  }
  std::vector<double> out(static_cast<std::size_t>(k));
  for (index_t j = 0; j < k; ++j)
    out[j] = std::sqrt(std::max(0.0, std::real(hs::linalg::dot(n, res.col(j), mres.col(j)))));
  return out;
}

/// max |X^H M Y - delta| (delta = I when `identity`, else 0).
template <class T>
double gram_deviation(const DenseMatrix<T>& x, LinearOperator<T>* m, const DenseMatrix<T>& y,
                      bool identity) {
  const index_t n = x.rows();
  DenseMatrix<T> my(n, y.cols()), g(x.cols(), y.cols());
  if (m) {
    m->apply(y.data(), n, my.data(), n, y.cols());
  } else {
    my = y;
  }
  for (index_t j = 0; j < y.cols(); ++j)
    for (index_t i = 0; i < x.cols(); ++i) g(i, j) = hs::linalg::dot(n, x.col(i), my.col(j));
  double dev = 0.0;
  for (index_t j = 0; j < g.cols(); ++j)
    for (index_t i = 0; i < g.rows(); ++i)
      dev = std::max(dev, std::abs(g(i, j) - T(identity && i == j ? 1.0 : 0.0)));
  return dev;
}

/// Result contract of include/hsolve/eigen.hpp plus the solver-specific guarantees (M-orthonormal
/// vectors, estimates consistent with the true residuals, consistent bookkeeping).
/// The (deflated, see true_residuals) residual is recomputed with the operator itself, so it
/// carries the rounding error of one application, about eps * app_cond * ||Op|| (app_cond =
/// cond(A - sigma B) for a dense-LU shift-invert operator, 1 for an explicit matrix); the
/// estimates must agree up to that floor:
///   ||Op x - theta x||_M <= 10 * estimate + 1e-14 * app_cond * max|theta|.
template <class T>
void check_contract(LinearOperator<T>& op, LinearOperator<T>* m, const HermitianEigResult<T>& r,
                    Which which, const KrylovSchurOptions<T>& opt, double app_cond = 1.0) {
  const index_t nev = opt.nev;
  ASSERT_EQ(static_cast<index_t>(r.theta.size()), nev);
  ASSERT_EQ(static_cast<index_t>(r.estimate.size()), nev);
  ASSERT_EQ(r.vectors.rows(), op.size());
  ASSERT_EQ(r.vectors.cols(), nev);
  EXPECT_EQ(r.converged, r.nconv >= nev);
  EXPECT_TRUE(std::is_sorted(r.ritz_all.begin(), r.ritz_all.end()));
  double max_ritz = 0.0;
  for (double th : r.ritz_all) max_ritz = std::max(max_ritz, std::abs(th));
  for (index_t i = 0; i < nev; ++i) {
    const double ref = r.theta[i] != 0.0 ? std::abs(r.theta[i]) : max_ritz;
    if (i < r.nconv) {
      EXPECT_LE(r.estimate[i], opt.tol * ref) << "pair " << i;
    } else {
      EXPECT_GT(r.estimate[i], opt.tol * ref) << "pair " << i;
    }
    const bool group_start = i == 0 || i == r.nconv;
    if (!group_start) {
      EXPECT_TRUE(ranked_before(r.theta[i - 1], r.theta[i], which)) << "pair " << i;
    }
  }
  ASSERT_EQ(static_cast<index_t>(r.history.size()), r.restarts + 1);
  for (index_t c = 0; c <= r.restarts; ++c) EXPECT_EQ(r.history[c].restart, c);
  EXPECT_EQ(r.history.back().nconv, r.nconv);
  EXPECT_EQ(r.history.back().op_applies, r.op_applies);
  EXPECT_EQ(r.history.back().max_estimate == 0.0, r.converged);

  EXPECT_LT(gram_deviation<T>(r.vectors, m, r.vectors, true), 1e-12);
  const auto res = true_residuals(op, m, r, opt.deflation);
  for (index_t i = 0; i < nev; ++i)
    EXPECT_LE(res[i], 10.0 * r.estimate[i] + 1e-14 * app_cond * max_ritz)
        << "pair " << i << " theta " << r.theta[i];
}

/// 2-norm condition number of a Hermitian matrix (dense reference, n <= 4000).
template <class T>
double condition_number(const hs::CsrMatrix<T>& k) {
  const auto mu = hs::test::dense_eigenvalues(k, static_cast<const hs::CsrMatrix<T>*>(nullptr));
  double lo = std::abs(mu.front()), hi = 0.0;
  for (double x : mu) {
    lo = std::min(lo, std::abs(x));
    hi = std::max(hi, std::abs(x));
  }
  return hi / lo;
}

void expect_values(const std::vector<double>& got, const std::vector<double>& want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol) << "index " << i;
}

/// |got[i] - want[i]| <= rel * |want[i]| for every i.
void expect_relative(const std::vector<double>& got, const std::vector<double>& want, double rel) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], rel * std::abs(want[i])) << "index " << i;
}

/// Op = diag(spectrum). Its application has only relative rounding errors (|fl(d_i x_i) - d_i x_i|
/// <= eps |d_i x_i|), so eigenvalues far below ||Op|| stay accurately determined.
template <class T>
hs::test::DiagonalOperator<T> diagonal_operator(const std::vector<double>& spectrum) {
  return hs::test::DiagonalOperator<T>(std::vector<T>(spectrum.begin(), spectrum.end()));
}

/// Spectrum with spread-out, asymmetric ends (fast convergence for every `which`): 10 t^3 + t +
/// 0.05 on a uniform grid t in [-1, 1].
std::vector<double> cubic_spectrum(index_t n) {
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) {
    const double t = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    w[i] = 10.0 * t * t * t + t + 0.05;
  }
  return w;
}

template <class T>
class LanczosTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LanczosTyped, Scalars);

// ------------------------------------------------------------------------------------ tests

TYPED_TEST(LanczosTyped, AllWhichModesOnKnownSpectrum) {
  using T = TypeParam;
  const auto spectrum = cubic_spectrum(300);
  auto op = operator_with_spectrum<T>(spectrum, 1);
  for (Which which : kAllWhich) {
    KrylovSchurOptions<T> opt;
    opt.nev = 8;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
    SCOPED_TRACE(static_cast<int>(which));
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, which, opt);
    expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-10);
  }
}

TYPED_TEST(LanczosTyped, GeneralizedShiftInvertWithBInnerProduct) {
  // Op = (A - sigma B)^{-1} B is self-adjoint in the B inner product; theta -> lambda = sigma +
  // 1/theta.
  using T = TypeParam;
  const index_t n = 200;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 21);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 22);
  const double sigma = 0.1;
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, sigma);
  DenseOperator<T> bop(bd);
  const auto lambda = hs::test::dense_eigenvalues(a, &b);
  std::vector<double> theta_exact;
  for (double l : lambda) theta_exact.push_back(1.0 / (l - sigma));
  const double kappa = condition_number(hs::add(T(1), a, T(-sigma), b));

  KrylovSchurOptions<T> opt;
  opt.nev = 6;
  std::vector<HermitianEigResult<T>> runs;
  for (bool store : {false, true}) {
    opt.store_bv = store;
    runs.push_back(hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt));
    const auto& r = runs.back();
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, &bop, r, Which::LargestMagnitude, opt, kappa);
    expect_values(r.theta, wanted(theta_exact, Which::LargestMagnitude, opt.nev),
                  1e-10 * std::abs(r.theta[0]));
    // Back-transformed eigenvalues are the ones nearest sigma.
    std::vector<double> near = lambda;
    std::sort(near.begin(), near.end(),
              [sigma](double x, double y) { return std::abs(x - sigma) < std::abs(y - sigma); });
    for (index_t i = 0; i < opt.nev; ++i) EXPECT_NEAR(sigma + 1.0 / r.theta[i], near[i], 1e-9);
  }
  // Stored and recomputed B*V agree to rounding (same Krylov space, different arithmetic).
  for (index_t i = 0; i < opt.nev; ++i) {
    EXPECT_NEAR(runs[0].theta[i], runs[1].theta[i], 1e-11 * std::abs(runs[0].theta[0]));
    // Same B-unit vector up to a phase: |x0^H B x1| == 1.
    DenseMatrix<T> x0(n, 1), x1(n, 1);
    std::copy(runs[0].vectors.col(i), runs[0].vectors.col(i) + n, x0.data());
    std::copy(runs[1].vectors.col(i), runs[1].vectors.col(i) + n, x1.data());
    EXPECT_NEAR(gram_deviation<T>(x0, &bop, x1, false), 1.0, 1e-9) << "pair " << i;
  }
}

TYPED_TEST(LanczosTyped, DeflationInBInnerProductFindsTheNextPairs) {
  using T = TypeParam;
  const index_t n = 160;
  const auto a = hs::test::random_hermitian<T>(n, 0.06, 31);
  const auto b = hs::test::random_hpd<T>(n, 0.06, 32);
  const double sigma = -0.2;
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, sigma);
  DenseOperator<T> bop(bd);
  std::vector<double> theta_exact;
  for (double l : hs::test::dense_eigenvalues(a, &b)) theta_exact.push_back(1.0 / (l - sigma));
  const auto expect = wanted(theta_exact, Which::LargestMagnitude, 8);
  const double kappa = condition_number(hs::add(T(1), a, T(-sigma), b));

  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  const auto first = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(first.converged);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    opt.deflation = &first.vectors;
    opt.store_bv = store;
    const auto second = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
    EXPECT_TRUE(second.converged);
    check_contract<T>(op, &bop, second, Which::LargestMagnitude, opt, kappa);
    expect_values(second.theta, std::vector<double>(expect.begin() + 4, expect.end()),
                  1e-10 * std::abs(expect[0]));
    EXPECT_LT(gram_deviation<T>(first.vectors, &bop, second.vectors, false), 1e-12);
  }
}

TYPED_TEST(LanczosTyped, DeflatedEstimatesExcludeTheSpanDComponent) {
  // Documented contract (eigen.hpp): with deflation the estimates measure the residual of the
  // deflated operator. The span(D) component of the full residual is R_D^H M x (R_D: residuals of
  // D's columns) and bounded by ||R_D||, here clearly above the rerun's estimates because D comes
  // from a loose first run.
  using T = TypeParam;
  const index_t n = 400;
  DenseOperator<T> op(hs::test::to_dense(hs::test::laplacian_1d<T>(n)));
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.tol = 1e-6;
  const auto first = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  ASSERT_TRUE(first.converged);
  double rd_fro2 = 0.0;  // ||R_D||_F^2 >= ||R_D||_2^2
  for (double x : true_residuals<T>(op, nullptr, first, nullptr)) rd_fro2 += x * x;

  opt.tol = 1e-12;
  opt.deflation = &first.vectors;
  const auto rerun = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(rerun.converged);
  check_contract<T>(op, nullptr, rerun, Which::LargestAlgebraic, opt);  // deflated residuals
  const auto projected = true_residuals<T>(op, nullptr, rerun, &first.vectors);
  const auto full = true_residuals<T>(op, nullptr, rerun, nullptr);
  double max_span_d = 0.0, max_est = 0.0;
  for (index_t i = 0; i < opt.nev; ++i) {
    // The two parts are orthogonal: full^2 = projected^2 + |span(D) part|^2.
    const double span_d = std::sqrt(std::max(0.0, full[i] * full[i] - projected[i] * projected[i]));
    EXPECT_LE(span_d, std::sqrt(rd_fro2) * (1.0 + 1e-6) + 1e-14) << "pair " << i;
    max_span_d = std::max(max_span_d, span_d);
    max_est = std::max(max_est, rerun.estimate[i]);
  }
  EXPECT_GT(max_span_d, 100.0 * max_est);
}

TYPED_TEST(LanczosTyped, ClusteredSpectrum) {
  // Four eigenvalues 1e-3 apart, well separated from a bulk in [-1, 1], plus a tight pair 1e-6
  // apart at the other end: every member is resolved to the tolerance.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 194; ++i) spectrum.push_back(-1.0 + 2.0 * static_cast<double>(i) / 193.0);
  for (double c : {5.0, 5.001, 5.002, 5.003}) spectrum.push_back(c);
  spectrum.push_back(-3.0);
  spectrum.push_back(-3.0 + 1e-6);
  auto op = operator_with_spectrum<T>(spectrum, 41);
  for (Which which : {Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
    SCOPED_TRACE(static_cast<int>(which));
    KrylovSchurOptions<T> opt;
    opt.nev = which == Which::LargestAlgebraic ? 4 : 2;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, which, opt);
    expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-10);
  }
}

/// Runs `which` for nev pairs, then reruns deflated by the vectors found. Every returned value must
/// be an exact eigenvalue (`spectrum`, with multiplicity) and the best nev of both runs together
/// must be exactly the nev wanted values. Returns how many of those the plain run found.
template <class T>
index_t plain_then_deflated(LinearOperator<T>& op, Which which, const std::vector<double>& spectrum,
                            index_t nev, double app_cond, double tol) {
  const auto expect = wanted(spectrum, which, nev);
  double scale = 0.0;
  for (double x : spectrum) scale = std::max(scale, std::abs(x));
  KrylovSchurOptions<T> opt;
  opt.nev = nev;
  opt.tol = tol;
  const auto plain = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
  EXPECT_TRUE(plain.converged);
  check_contract<T>(op, nullptr, plain, which, opt, app_cond);
  std::vector<double> missing = expect;
  for (double th : plain.theta) {
    const auto nearest =
        std::min_element(spectrum.begin(), spectrum.end(),
                         [th](double x, double y) { return std::abs(x - th) < std::abs(y - th); });
    EXPECT_NEAR(*nearest, th, 1e-10 * scale);
    const auto hit = std::find_if(missing.begin(), missing.end(), [th, scale](double x) {
      return std::abs(x - th) < 1e-9 * scale;
    });
    if (hit != missing.end()) missing.erase(hit);
  }

  opt.deflation = &plain.vectors;
  const auto rerun = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
  EXPECT_TRUE(rerun.converged);
  check_contract<T>(op, nullptr, rerun, which, opt, app_cond);
  EXPECT_LT(gram_deviation<T>(plain.vectors, nullptr, rerun.vectors, false), 1e-12);
  std::vector<double> merged = plain.theta;
  merged.insert(merged.end(), rerun.theta.begin(), rerun.theta.end());
  expect_values(wanted(merged, which, nev), expect, 1e-10 * scale);
  return nev - static_cast<index_t>(missing.size());
}

TYPED_TEST(LanczosTyped, MultiplicitiesRecoveredByDeflatedRerun) {
  // 2-D Laplacian with nx == ny: lambda_ij = lambda_ji, so most eigenvalues are double. A single
  // Krylov sequence sees one vector per eigenspace in exact arithmetic; second copies only emerge
  // from rounding, amplified over the iterations. With few iterations (loose tolerance) copies are
  // missed (here 4 of 6 for Op = L at tol 1e-6); with many they usually appear. Either way a rerun
  // deflated by the found vectors must supply the missing copies.
  using T = TypeParam;
  const index_t nx = 12, nev = 6;
  const auto lap = hs::test::laplacian_2d<T>(nx, nx);
  const auto lambda = hs::test::laplacian_2d_eigenvalues(nx, nx);  // ascending, with multiplicity
  DenseOperator<T> direct(hs::test::to_dense(lap));
  // Same eigenvalues through shift-invert Op = L^{-1}: theta = 1/lambda converge in few restarts.
  hs::test::DenseShiftInvertOperator<T> inverse(hs::test::to_dense(lap), nullptr, 0.0);
  std::vector<double> theta;
  for (double l : lambda) theta.push_back(1.0 / l);
  const double kappa = condition_number(lap);

  const index_t direct_loose =
      plain_then_deflated<T>(direct, Which::SmallestAlgebraic, lambda, nev, 1.0, 1e-6);
  const index_t direct_tight =
      plain_then_deflated<T>(direct, Which::SmallestAlgebraic, lambda, nev, 1.0, 1e-11);
  const index_t inverse_tight =
      plain_then_deflated<T>(inverse, Which::LargestMagnitude, theta, nev, kappa, 1e-11);
  std::cout << "[ info     ] copies of the " << nev
            << " smallest eigenvalues (with multiplicity) found by a "
            << "plain run: Op = L tol 1e-6: " << direct_loose
            << ", Op = L tol 1e-11: " << direct_tight << ", Op = L^-1 tol 1e-11: " << inverse_tight
            << "; the deflated reruns supplied the rest\n";
  this->RecordProperty("plain_copies_direct_tol1e-6", static_cast<int>(direct_loose));
  this->RecordProperty("plain_copies_direct_tol1e-11", static_cast<int>(direct_tight));
  this->RecordProperty("plain_copies_inverse_tol1e-11", static_cast<int>(inverse_tight));
}

TYPED_TEST(LanczosTyped, InvariantSubspaceBreakdown) {
  // Three distinct eigenvalues: every Krylov sequence breaks down after 3 steps and continues with
  // a fresh random vector, so one cycle yields several exact copies of each eigenvalue.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 100; ++i) spectrum.push_back(i < 30 ? 1.0 : (i < 60 ? 2.0 : 3.0));
  auto op = operator_with_spectrum<T>(spectrum, 51);
  for (Which which : kAllWhich) {
    SCOPED_TRACE(static_cast<int>(which));
    KrylovSchurOptions<T> opt;
    opt.nev = 5;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.restarts, 0);
    check_contract<T>(op, nullptr, r, which, opt);
    expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-12);
  }
}

TYPED_TEST(LanczosTyped, SmallCouplingsAreKeptOverAWideDynamicRange) {
  // A Lanczos coupling far below ||Op|| can still be large compared with a wanted |theta|. A
  // breakdown test scaled by ||Op|| (instead of the step's own ||Op v_j||) dropped such couplings:
  // wrong eigenvalues or large true residuals came back marked converged with estimate 0.
  // Diagonal operators keep the contract's residual floor (1e-14 max |theta|) honest here.
  using T = TypeParam;
  {
    SCOPED_TRACE("one eigenvalue 1e13 above a bulk in (0.5, 1.5]");
    std::vector<double> spectrum = {1e13};
    for (index_t i = 1; i < 200; ++i) spectrum.push_back(0.5 + static_cast<double>(i) / 199.0);
    auto op = diagonal_operator<T>(spectrum);
    KrylovSchurOptions<T> opt;
    opt.nev = 3;
    opt.tol = 1e-8;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
    // A bulk Ritz vector keeps a rounding-size component along e_0, amplified by 1e13: true
    // residuals ~1e-4, hence eigenvalue errors ~residual^2 / gap ~ 1e-6 (the bug gave 0.3).
    expect_relative(r.theta, wanted(spectrum, Which::LargestMagnitude, opt.nev), 1e-4);
  }
  {
    SCOPED_TRACE("shift-invert spectrum, sigma 1e-12 (relative) below lambda_1");
    // theta = 1 / (lambda - sigma) of the 1-D Laplacian: ||Op|| = 4e17, wanted theta 1.4e5..1.2e4.
    const auto lambda = hs::test::laplacian_1d_eigenvalues(2000);
    const double sigma = lambda[0] * (1.0 - 1e-12);
    std::vector<double> theta;
    for (double l : lambda) theta.push_back(1.0 / (l - sigma));
    auto op = diagonal_operator<T>(theta);
    KrylovSchurOptions<T> opt;
    opt.nev = 6;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
    expect_relative(r.theta, wanted(theta, Which::LargestMagnitude, opt.nev), 1e-5);
  }
  {
    SCOPED_TRACE("start vector within 1e-9 of the invariant subspace span{e_0, e_1}");
    // The couplings out of span{e_0, e_1} are far above rounding but below 1e-12 ||Op||; dropping
    // them hid a true residual of 5e-8 behind the estimate 0 of the pair theta = 3.
    const index_t n = 2000;
    std::vector<double> spectrum = {1e6, 3.0};
    for (index_t i = 2; i < n; ++i)
      spectrum.push_back(0.5 + 1.5 * static_cast<double>(i - 2) / static_cast<double>(n - 3));
    auto op = diagonal_operator<T>(spectrum);
    DenseMatrix<T> x0(n, 1);
    hs::linalg::fill_random(x0.data(), n, 5);
    hs::linalg::scal(n, T(1e-9), x0.data());
    x0(0, 0) += T(1.0);
    x0(1, 0) += T(1.0);
    KrylovSchurOptions<T> opt;
    opt.nev = 2;
    opt.initial = &x0;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
    expect_relative(r.theta, {1e6, 3.0}, 1e-10);
  }
}

TYPED_TEST(LanczosTyped, InitialVectorSpanningTheWantedSpace) {
  // A start vector inside the wanted invariant subspace: the first three Lanczos steps reproduce
  // it exactly, so the first cycle already returns the exact pairs.
  using T = TypeParam;
  const index_t n = 500;
  std::vector<T> diag(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) diag[i] = T(static_cast<double>(i + 1));
  hs::test::DiagonalOperator<T> op(diag);
  DenseMatrix<T> x0(n, 1);
  x0(n - 1, 0) = T(1.0);
  x0(n - 2, 0) = T(2.0);
  x0(n - 3, 0) = T(3.0);
  KrylovSchurOptions<T> opt;
  opt.nev = 3;
  opt.initial = &x0;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
  expect_values(r.theta, {500.0, 499.0, 498.0}, 1e-11);
  // The same problem from a random start needs restarts (unit gaps over a range of 500).
  opt.initial = nullptr;
  const auto random_start = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_GT(random_start.restarts, 0);
}

TYPED_TEST(LanczosTyped, StoredAndRecomputedBvAgreeThroughRestarts) {
  // store_bv must not change the mathematics: B*V updated through many basis rotations agrees with
  // B*w recomputed on demand (the deflated variant is covered by the deflation test above).
  using T = TypeParam;
  const index_t n = 120;
  const auto a = hs::test::random_hermitian<T>(n, 0.08, 61);
  const auto b = hs::test::random_hpd<T>(n, 0.08, 62);
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, 0.05);
  DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 5;
  opt.ncv = 11;  // forces several restarts
  const auto ref = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(ref.converged);
  ASSERT_GT(ref.restarts, 0);
  opt.store_bv = true;
  const auto stored = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(stored.converged);
  for (index_t i = 0; i < opt.nev; ++i)
    EXPECT_NEAR(ref.theta[i], stored.theta[i], 1e-11 * std::abs(ref.theta[0])) << i;
}

TYPED_TEST(LanczosTyped, StoredBvStaysOrthonormalOverManyRestarts) {
  // store_bv: the stored M*w of a new basis vector must come from M applied to the orthogonalized
  // vector. Updating it by the CGS2 recurrence passes the error of the stored columns on, amplified
  // by ||Op v|| / beta every step: here the returned vectors had lost M-orthonormality to 1.8e-9
  // after 300 restarts (check_contract requires 1e-12). Slow problem: Op = L^{-1} with B ~ I,
  // smallest theta wanted, a small basis and an unreachable tolerance.
  using T = TypeParam;
  const index_t n = 400;
  const auto lap = hs::test::laplacian_1d<T>(n);
  const auto bd = hs::test::to_dense(hs::test::prescribed_spectrum<T>(
      std::vector<double>(static_cast<std::size_t>(n), 1.0), 8, 3));
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(lap), &bd, 0.0);
  DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 8;
  opt.ncv = 18;
  opt.tol = 1e-14;
  opt.max_restarts = 300;
  opt.store_bv = true;
  const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::SmallestAlgebraic, opt);
  EXPECT_EQ(r.restarts, opt.max_restarts);
  check_contract<T>(op, &bop, r, Which::SmallestAlgebraic, opt, condition_number(lap));
}

TYPED_TEST(LanczosTyped, RowBlockedRotationAndChunkedDeflationCheck) {
  // n > 2 * 4096 rotates the basis (and the stored M*V) in place in three row blocks, and d = 70
  // deflation columns are checked chunk by chunk (in recompute mode the chunks of M*D are limited
  // by the free Krylov columns, here ncv + 1 = 13). Op = diag(a) is self-adjoint in the inner
  // product of M = diag(m) (they commute); its eigenvectors are e_i / sqrt(m_i). The largest 80
  // values 1 + 1.1^k (k = 1..80, at the end) sit above a bulk in [0, 1): relative gaps of ~10%
  // converge in a few restarts, so rounding accumulated by the rotations stays far below the
  // contract's residual floor.
  using T = TypeParam;
  const index_t n = 9001, nev = 5, d = 70, top = 80;
  std::vector<double> a(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n - top; ++i)
    a[i] = static_cast<double>(i) / static_cast<double>(n - top);
  for (index_t k = 0; k < top; ++k)
    a[n - top + k] = 1.0 + std::pow(1.1, static_cast<double>(k + 1));
  std::vector<double> m(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i)
    m[i] = 1.0 + 3.0 * std::fmod(0.6180339887 * static_cast<double>(i), 1.0);
  auto op = diagonal_operator<T>(a);
  auto mop = diagonal_operator<T>(m);
  KrylovSchurOptions<T> opt;
  opt.nev = nev;
  opt.ncv = 12;
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    opt.store_bv = store;
    const auto r = hs::lanczos_krylov_schur<T>(op, &mop, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    EXPECT_GT(r.restarts, 2);
    check_contract<T>(op, &mop, r, Which::LargestAlgebraic, opt);
    expect_values(r.theta, wanted(a, Which::LargestAlgebraic, nev), 1e-10);
  }

  // Deflate the exact eigenvectors of the d largest values: the rerun returns the next nev.
  DenseMatrix<T> defl(n, d);
  for (index_t c = 0; c < d; ++c) defl(n - 1 - c, c) = T(1.0 / std::sqrt(m[n - 1 - c]));
  opt.deflation = &defl;
  const auto all = wanted(a, Which::LargestAlgebraic, d + nev);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "deflated, store_bv" : "deflated, recompute");
    opt.store_bv = store;
    const auto r = hs::lanczos_krylov_schur<T>(op, &mop, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, &mop, r, Which::LargestAlgebraic, opt);
    expect_values(r.theta, std::vector<double>(all.begin() + d, all.end()), 1e-10);
    EXPECT_LT(gram_deviation<T>(defl, &mop, r.vectors, false), 1e-13);
  }
  // A defect in a column beyond the first chunks is found as well.
  DenseMatrix<T> bad = defl;
  bad(n - 1 - 66, 66) *= T(1.001);
  opt.deflation = &bad;
  for (bool store : {false, true}) {
    opt.store_bv = store;
    try {
      hs::lanczos_krylov_schur<T>(op, &mop, Which::LargestAlgebraic, opt);
      ADD_FAILURE() << "non-orthonormal column 66 accepted (store_bv " << store << ")";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  }
}

TYPED_TEST(LanczosTyped, ReproducibleForFixedSeed) {
  using T = TypeParam;
  auto op = operator_with_spectrum<T>(cubic_spectrum(250), 71);
  KrylovSchurOptions<T> opt;
  opt.nev = 7;
  opt.ncv = 18;
  const auto r1 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  const auto r2 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r1.converged);
  ASSERT_EQ(r1.theta.size(), r2.theta.size());
  EXPECT_EQ(0, std::memcmp(r1.theta.data(), r2.theta.data(), r1.theta.size() * sizeof(double)));
  EXPECT_EQ(
      0, std::memcmp(r1.estimate.data(), r2.estimate.data(), r1.estimate.size() * sizeof(double)));
  EXPECT_EQ(
      0, std::memcmp(r1.vectors.data(), r2.vectors.data(),
                     static_cast<std::size_t>(r1.vectors.rows() * r1.vectors.cols()) * sizeof(T)));
  EXPECT_EQ(r1.restarts, r2.restarts);
  EXPECT_EQ(r1.op_applies, r2.op_applies);
  // Another seed takes another path to the same eigenvalues.
  opt.seed = 7;
  const auto r3 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r3.converged);
  expect_values(r3.theta, r1.theta, 1e-10);
}

TYPED_TEST(LanczosTyped, FullKrylovSpaceIsExact) {
  // ncv >= n - d: a single cycle spans the free space; with deflation the result stays orthogonal
  // to D and returns the remaining spectrum.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 20; ++i) spectrum.push_back(std::pow(-1.0, i) * (1.0 + i));
  auto op = operator_with_spectrum<T>(spectrum, 81);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;  // auto ncv = min(20, 20) = n
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  EXPECT_EQ(r.op_applies, 20);
  EXPECT_EQ(r.ritz_all.size(), 20u);
  check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
  expect_values(r.theta, wanted(spectrum, Which::LargestMagnitude, 4), 1e-12);

  opt.deflation = &r.vectors;
  opt.nev = 16;  // nev == n - d
  const auto rest = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  EXPECT_TRUE(rest.converged);
  EXPECT_EQ(rest.op_applies, 16);
  check_contract<T>(op, nullptr, rest, Which::LargestMagnitude, opt);
  const auto all = wanted(spectrum, Which::LargestMagnitude, 20);
  expect_values(rest.theta, std::vector<double>(all.begin() + 4, all.end()), 1e-12);
  EXPECT_LT(gram_deviation<T>(r.vectors, nullptr, rest.vectors, false), 1e-13);
}

TYPED_TEST(LanczosTyped, EstimatesStayHonestOverManyRestarts) {
  // A slowly converging problem with a small basis: pairs converge at different restarts (and are
  // locked once their coupling reaches working precision). The estimates must bound the true
  // residuals also for a loose tolerance, where decoupling pairs at the tolerance level would hide
  // a residual of order tol * |theta_locked| in the other pairs.
  using T = TypeParam;
  const index_t n = 400;
  auto op = DenseOperator<T>(hs::test::to_dense(hs::test::laplacian_1d<T>(n)));
  const auto exact = hs::test::laplacian_1d_eigenvalues(n);
  for (double tol : {1e-6, 1e-10}) {
    SCOPED_TRACE(tol);
    KrylovSchurOptions<T> opt;
    opt.nev = 10;
    opt.ncv = 24;
    opt.tol = tol;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    EXPECT_GT(r.restarts, 3);
    check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
    // A Hermitian Ritz value lies within its residual of an eigenvalue: <= tol * |theta| <= 4 tol.
    expect_values(r.theta, wanted(exact, Which::LargestAlgebraic, opt.nev), 4.0 * tol);
  }
}

TYPED_TEST(LanczosTyped, ZeroEstimatesOfEarlyConvergedPairsAreHonest) {
  // The isolated top values 10, 6, 3 converge long before 2 (relative gap 1/11 to the bulk in
  // [-1, 1]) and keep improving until their coupling is at rounding level relative to every wanted
  // pair. They are then decoupled exactly, by locking or equally by heev's deflation of negligible
  // off-diagonals, and report estimate 0. That must stay honest: the true residual is at rounding
  // level, which bounds the locking threshold from above (a tol-level threshold fails here).
  using T = TypeParam;
  std::vector<double> spectrum = {10.0, 6.0, 3.0, 2.0};
  for (index_t i = 0; i < 396; ++i) spectrum.push_back(-1.0 + 2.0 * static_cast<double>(i) / 395.0);
  auto op = diagonal_operator<T>(spectrum);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.ncv = 9;
  opt.tol = 1e-12;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_GT(r.restarts, 2);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
  expect_values(r.theta, {10.0, 6.0, 3.0, 2.0}, 1e-11);
  const auto res = true_residuals<T>(op, nullptr, r, nullptr);
  index_t decoupled = 0;
  for (index_t i = 0; i < opt.nev; ++i) {
    if (r.estimate[i] != 0.0) continue;
    ++decoupled;
    EXPECT_LE(res[i], 1e-14 * std::abs(r.theta[i])) << "pair " << i;
  }
  EXPECT_GE(decoupled, 1);
}

TYPED_TEST(LanczosTyped, LargestMagnitudeTiesRankThePositiveValueFirst) {
  // Op = diag(1, -1) started from (1, 1): both Lanczos diagonal entries are exactly 0, so
  // H = [[0, b], [b, 0]] has the Ritz values -b and b, an exact tie in |theta|.
  using T = TypeParam;
  hs::test::DiagonalOperator<T> op(std::vector<T>{T(1.0), T(-1.0)});
  DenseMatrix<T> x0(2, 1);
  x0(0, 0) = T(1.0);
  x0(1, 0) = T(1.0);
  KrylovSchurOptions<T> opt;
  opt.initial = &x0;
  for (index_t nev : {1, 2}) {
    opt.nev = nev;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
    ASSERT_EQ(r.ritz_all.size(), 2u);
    ASSERT_EQ(r.ritz_all[0], -r.ritz_all[1]);  // the tie is exact
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
    EXPECT_EQ(r.theta[0], r.ritz_all[1]) << "nev " << nev;
    EXPECT_NEAR(r.theta[0], 1.0, 1e-15);
  }
}

TYPED_TEST(LanczosTyped, ConvergedPairsPrecedeBetterRankedUnconvergedOnes) {
  // The tight top pair {1, 1 - 1e-7} converges much more slowly than the isolated 0.7 below it;
  // after a few restarts only 0.7 has converged and must come first although it ranks third.
  using T = TypeParam;
  std::vector<double> spectrum = {1.0, 1.0 - 1e-7, 0.7};
  for (index_t i = 3; i < 200; ++i)
    spectrum.push_back(-1.0 + 1.4 * static_cast<double>(i - 3) / 196.0);
  auto op = diagonal_operator<T>(spectrum);
  KrylovSchurOptions<T> opt;
  opt.nev = 3;
  opt.ncv = 8;
  opt.tol = 1e-7;
  opt.max_restarts = 8;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  ASSERT_EQ(r.nconv, 1);
  EXPECT_NEAR(r.theta[0], 0.7, 1e-12);
  EXPECT_GT(r.theta[1], 0.99);
  EXPECT_GT(r.theta[2], 0.99);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
}

TYPED_TEST(LanczosTyped, NonConvergenceIsReportedNotThrown) {
  using T = TypeParam;
  const index_t n = 400;
  auto op = DenseOperator<T>(hs::test::to_dense(hs::test::laplacian_1d<T>(n)));
  KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 13;
  opt.max_restarts = 2;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.restarts, 2);
  EXPECT_LT(r.nconv, opt.nev);
  EXPECT_GT(r.history.back().max_estimate, opt.tol);
  check_contract<T>(op, nullptr, r, Which::SmallestAlgebraic, opt);
  EXPECT_EQ(r.op_applies, 13 + 2 * (13 - (6 + (13 - 6) / 2)));
  EXPECT_EQ(op.applications(), r.op_applies + opt.nev);  // + the residual check in check_contract

  opt.max_restarts = 0;
  const auto once = hs::lanczos_krylov_schur<T>(op, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_EQ(once.restarts, 0);
  EXPECT_EQ(once.history.size(), 1u);
  EXPECT_EQ(once.op_applies, 13);
}

TYPED_TEST(LanczosTyped, InvalidOptionsThrowInvalidInput) {
  using T = TypeParam;
  const index_t n = 30;
  auto op = operator_with_spectrum<T>(cubic_spectrum(n), 91);
  DenseOperator<T> wrong_size_b(DenseMatrix<T>(n + 1, n + 1));
  auto expect_invalid = [&](const KrylovSchurOptions<T>& opt, LinearOperator<T>* b,
                            const char* what) {
    try {
      hs::lanczos_krylov_schur<T>(op, b, Which::LargestAlgebraic, opt);
      ADD_FAILURE() << "no exception for " << what;
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << what << ": " << e.what();
    }
  };
  KrylovSchurOptions<T> base;
  base.nev = 3;
  auto opt = base;
  opt.nev = 0;
  expect_invalid(opt, nullptr, "nev = 0");
  opt = base;
  opt.nev = n + 1;
  expect_invalid(opt, nullptr, "nev > n");
  opt = base;
  opt.ncv = 3;
  expect_invalid(opt, nullptr, "ncv == nev < n");
  opt = base;
  opt.ncv = -1;
  expect_invalid(opt, nullptr, "ncv < 0");
  opt = base;
  opt.tol = 0.0;
  expect_invalid(opt, nullptr, "tol = 0");
  opt = base;
  opt.max_restarts = -1;
  expect_invalid(opt, nullptr, "max_restarts < 0");
  expect_invalid(base, &wrong_size_b, "b_inner of wrong order");

  DenseMatrix<T> zero(n, 1), short_block(n - 1, 1);
  opt = base;
  opt.initial = &zero;
  expect_invalid(opt, nullptr, "zero initial vector");
  opt.initial = &short_block;
  expect_invalid(opt, nullptr, "initial vector of wrong length");

  // Deflation: wrong shape, not orthonormal, leaves fewer than nev free dimensions, initial in
  // span(D).
  DenseMatrix<T> d(n, 2);
  d(0, 0) = T(1.0);
  d(1, 1) = T(1.0);
  opt = base;
  opt.deflation = &short_block;
  expect_invalid(opt, nullptr, "deflation of wrong length");
  DenseMatrix<T> not_orthonormal = d;
  not_orthonormal(1, 0) = T(0.5);
  opt.deflation = &not_orthonormal;
  expect_invalid(opt, nullptr, "non-orthonormal deflation");
  opt.deflation = &d;
  opt.nev = n - 1;
  expect_invalid(opt, nullptr, "nev > n - d");
  opt.nev = 3;
  DenseMatrix<T> in_span(n, 1);
  in_span(1, 0) = T(2.0);
  opt.initial = &in_span;
  expect_invalid(opt, nullptr, "initial vector inside span(D)");
  // The valid deflated problem itself runs.
  opt.initial = nullptr;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_LT(gram_deviation<T>(d, nullptr, r.vectors, false), 1e-13);
}

TYPED_TEST(LanczosTyped, NumericallyMalformedInputsThrowInvalidInput) {
  // Input errors that only show up in the arithmetic (non-finite blocks or operator values, a
  // b_inner that is not positive definite) are InvalidInput, never Internal.
  using T = TypeParam;
  const index_t n = 40;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> spectrum(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) spectrum[i] = 1.0 + static_cast<double>(i);
  auto op = diagonal_operator<T>(spectrum);
  // `topic`: a word the message must contain, so that it names the actual problem.
  auto expect_invalid = [](LinearOperator<T>& o, LinearOperator<T>* b,
                           const KrylovSchurOptions<T>& opt, const char* what, const char* topic) {
    try {
      hs::lanczos_krylov_schur<T>(o, b, Which::LargestAlgebraic, opt);
      ADD_FAILURE() << "no exception for " << what;
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << what << ": " << e.what();
      EXPECT_NE(std::string(e.what()).find(topic), std::string::npos) << what << ": " << e.what();
    }
  };
  KrylovSchurOptions<T> base;
  base.nev = 3;

  // std::max(dev, NaN) == dev: a NaN entry must not pass the orthonormality check of D.
  DenseMatrix<T> d(n, 2);
  d(0, 0) = T(1.0);
  d(1, 1) = T(nan);
  auto opt = base;
  opt.deflation = &d;
  expect_invalid(op, nullptr, opt, "deflation block with a NaN entry", "deflation");
  DenseMatrix<T> x0(n, 1);
  x0(0, 0) = T(1.0);
  x0(3, 0) = T(nan);
  opt = base;
  opt.initial = &x0;
  expect_invalid(op, nullptr, opt, "initial vector with a NaN entry", "initial");
  auto nan_op = diagonal_operator<T>([&] {
    auto s = spectrum;
    s[7] = nan;
    return s;
  }());
  expect_invalid(nan_op, nullptr, base, "operator returning NaN", "non-finite");

  // Not positive definite b_inner: M = -I fails on the start vector; M = diag(+-1) fails on the
  // start vector or on a later random vector (whichever first has w^H M w <= 0; indefiniteness
  // is not detectable in general, but must never surface as an internal error).
  auto negative = diagonal_operator<T>(std::vector<double>(static_cast<std::size_t>(n), -1.0));
  std::vector<double> signs(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) signs[i] = i % 2 ? -1.0 : 1.0;
  auto indefinite = diagonal_operator<T>(signs);
  DenseMatrix<T> e0(n, 1);
  e0(0, 0) = T(1.0);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    opt = base;
    opt.store_bv = store;
    expect_invalid(op, &negative, opt, "negative definite b_inner", "positive definite");
    expect_invalid(op, &indefinite, opt, "indefinite b_inner", "positive definite");
    opt.initial = &e0;
    expect_invalid(op, &negative, opt, "negative definite b_inner, user start vector",
                   "positive definite");
  }
}

// --------------------------------------------------------- block Krylov-Schur (block_size > 1)

template <class T>
class LanczosBlockTyped : public ::testing::Test {};
TYPED_TEST_SUITE(LanczosBlockTyped, Scalars);

/// Copies of `value` among `theta` (within rel * |value|).
index_t copies(const std::vector<double>& theta, double value, double rel) {
  return static_cast<index_t>(std::count_if(theta.begin(), theta.end(), [&](double t) {
    return std::abs(t - value) <= rel * std::abs(value);
  }));
}

TYPED_TEST(LanczosBlockTyped, BlockSizesAgreeWithSingleVectorInBInnerProduct) {
  // Shift-invert of a random pencil, B inner product: every block size returns the same pairs as
  // p = 1 (values to tolerance, the same B-unit vectors up to a phase), in both store modes.
  using T = TypeParam;
  const index_t n = 240;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 121);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 122);
  const double sigma = 0.15;
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, sigma);
  DenseOperator<T> bop(bd);
  std::vector<double> theta_exact;
  for (double l : hs::test::dense_eigenvalues(a, &b)) theta_exact.push_back(1.0 / (l - sigma));
  const double kappa = condition_number(hs::add(T(1), a, T(-sigma), b));
  const auto expect = wanted(theta_exact, Which::LargestMagnitude, 7);

  KrylovSchurOptions<T> opt;
  opt.nev = 7;
  opt.tol = 1e-11;
  const auto ref = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(ref.converged);
  for (index_t p : {1, 2, 4, 8}) {
    for (bool store : {false, true}) {
      SCOPED_TRACE("p = " + std::to_string(p) + (store ? ", store_bv" : ", recompute"));
      opt.block_size = p;
      opt.store_bv = store;
      const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
      EXPECT_TRUE(r.converged);
      check_contract<T>(op, &bop, r, Which::LargestMagnitude, opt, kappa);
      expect_values(r.theta, expect, 1e-10 * std::abs(expect[0]));
      for (index_t i = 0; i < opt.nev; ++i) {
        EXPECT_NEAR(r.theta[i], ref.theta[i], 1e-10 * std::abs(ref.theta[0])) << "pair " << i;
        DenseMatrix<T> x0(n, 1), x1(n, 1);
        std::copy(ref.vectors.col(i), ref.vectors.col(i) + n, x0.data());
        std::copy(r.vectors.col(i), r.vectors.col(i) + n, x1.data());
        EXPECT_NEAR(gram_deviation<T>(x0, &bop, x1, false), 1.0, 1e-8) << "pair " << i;
      }
      if (p > 1) {
        EXPECT_EQ(r.op_applies % p, 0);
      }
    }
  }
}

TYPED_TEST(LanczosBlockTyped, AllWhichModesWithAnOddBlockSize) {
  using T = TypeParam;
  const auto spectrum = cubic_spectrum(300);
  auto op = operator_with_spectrum<T>(spectrum, 1);
  for (Which which : kAllWhich) {
    SCOPED_TRACE(static_cast<int>(which));
    KrylovSchurOptions<T> opt;
    opt.nev = 8;
    opt.block_size = 3;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, which, opt);
    expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-10);
  }
}

TYPED_TEST(LanczosBlockTyped, FourFoldMultiplicityFoundInOneRun) {
  // An eigenvalue of multiplicity 4 nearest to the shift: a single Krylov sequence sees one vector
  // of its eigenspace in exact arithmetic (further copies only through rounding), while a block of
  // p >= 4 random vectors has generic components in all of it, so one run returns all four
  // copies by construction. (Measured: the single-vector runs here also recover all four through
  // rounding, at tol 1e-6 and 1e-10; their count is reported, not asserted.) Shift-invert in the B
  // inner product as in the driver: A = B^{1/2} Q diag(lambda) Q^H B^{1/2} with B diagonal, so the
  // pencil's eigenvalues are lambda.
  using T = TypeParam;
  const index_t n = 300;
  std::vector<double> lambda;
  for (index_t i = 0; i < 4; ++i) lambda.push_back(0.5);
  lambda.push_back(0.8);
  lambda.push_back(-0.9);
  for (index_t i = 6; i < n; ++i) lambda.push_back(1.5 + static_cast<double>(i) / 50.0);
  const auto core = hs::test::to_dense(hs::test::prescribed_spectrum<T>(lambda, 8, 131));
  std::vector<double> bdiag(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) bdiag[i] = 1.0 + 0.5 * std::sin(0.37 * static_cast<double>(i));
  DenseMatrix<T> ad(n, n), bd(n, n);
  for (index_t j = 0; j < n; ++j) {
    bd(j, j) = T(bdiag[j]);
    for (index_t i = 0; i < n; ++i) ad(i, j) = T(std::sqrt(bdiag[i] * bdiag[j])) * core(i, j);
  }
  const double sigma = 0.3;
  hs::test::DenseShiftInvertOperator<T> op(ad, &bd, sigma);
  DenseOperator<T> bop(bd);
  std::vector<double> theta_exact;
  for (double l : lambda) theta_exact.push_back(1.0 / (l - sigma));
  const double theta4 = 1.0 / (0.5 - sigma);
  for (double tol : {1e-6, 1e-10}) {
    for (index_t p : {1, 4, 8}) {
      SCOPED_TRACE("p = " + std::to_string(p) + ", tol " + std::to_string(tol));
      KrylovSchurOptions<T> opt;
      opt.nev = 6;
      opt.tol = tol;
      opt.block_size = p;
      const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
      EXPECT_TRUE(r.converged);
      check_contract<T>(op, &bop, r, Which::LargestMagnitude, opt, 1e3);
      const index_t found = copies(r.theta, theta4, 1e-9);
      if (p >= 4) {
        EXPECT_EQ(found, 4);
        expect_relative(r.theta, wanted(theta_exact, Which::LargestMagnitude, opt.nev), 1e-9);
      } else {
        std::cout << "[ info     ] tol " << tol
                  << ": copies of the 4-fold eigenvalue found by a single-vector run: " << found
                  << " of 4\n";
        this->RecordProperty(tol > 1e-8 ? "p1_copies_tol1e-6" : "p1_copies_tol1e-10",
                             static_cast<int>(found));
      }
    }
  }
}

TYPED_TEST(LanczosBlockTyped, ExactFourFoldEigenvalueOfADiagonalOperator) {
  // Op = diag(...) keeps the multiplicity exact (its application has only relative rounding):
  // blocks of p >= 4 span the 4-dimensional eigenspace from the start block (an exactly rank-4
  // share of the first block), the multiplicity case most prone to dependent block columns.
  // A single Krylov sequence holds one direction of it in exact arithmetic; measured, rounding
  // still supplies all four copies here (reported, not asserted).
  using T = TypeParam;
  std::vector<double> spectrum = {10.0, 10.0, 10.0, 10.0, 9.0};
  for (index_t i = 0; i < 395; ++i) spectrum.push_back(static_cast<double>(i) / 394.0);
  auto op = diagonal_operator<T>(spectrum);
  for (index_t p : {1, 4, 8}) {
    SCOPED_TRACE("p = " + std::to_string(p));
    KrylovSchurOptions<T> opt;
    opt.nev = 6;
    opt.tol = 1e-10;
    opt.block_size = p;
    const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
    const index_t found = copies(r.theta, 10.0, 1e-12);
    if (p >= 4) {
      EXPECT_EQ(found, 4);
      expect_values(r.theta, wanted(spectrum, Which::LargestAlgebraic, opt.nev), 1e-10);
    } else {
      std::cout << "[ info     ] diagonal operator: copies of the exact 4-fold eigenvalue found by "
                << "a single-vector run: " << found << " of 4\n";
      this->RecordProperty("p1_copies_diagonal", static_cast<int>(found));
    }
  }
}

TYPED_TEST(LanczosBlockTyped, LaplacianDoubleEigenvaluesFoundInOneRun) {
  // 2-D Laplacian with nx == ny (eigenvalues of multiplicity 2), Op = L^{-1}: p = 2 and p = 4
  // return the 8 smallest eigenvalues with multiplicity from one run each.
  using T = TypeParam;
  const index_t nx = 14, nev = 8;
  const auto lap = hs::test::laplacian_2d<T>(nx, nx);
  const auto lambda = hs::test::laplacian_2d_eigenvalues(nx, nx);
  hs::test::DenseShiftInvertOperator<T> inverse(hs::test::to_dense(lap), nullptr, 0.0);
  std::vector<double> theta;
  for (double l : lambda) theta.push_back(1.0 / l);
  const double kappa = condition_number(lap);
  for (index_t p : {2, 4}) {
    SCOPED_TRACE("p = " + std::to_string(p));
    KrylovSchurOptions<T> opt;
    opt.nev = nev;
    opt.tol = 1e-11;
    opt.block_size = p;
    const auto r = hs::lanczos_krylov_schur<T>(inverse, nullptr, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(inverse, nullptr, r, Which::LargestMagnitude, opt, kappa);
    expect_relative(r.theta, wanted(theta, Which::LargestMagnitude, nev), 1e-10);
  }
}

TYPED_TEST(LanczosBlockTyped, InvariantSubspaceBreakdownOfWholeBlocks) {
  // Three distinct eigenvalues: span{X, Op X, Op^2 X} is invariant, so the fourth block is
  // dependent to rounding in every column; all p columns are replaced by random vectors and the
  // cycle continues in a decoupled block. One cycle yields exact copies of each value.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 100; ++i) spectrum.push_back(i < 30 ? 1.0 : (i < 60 ? 2.0 : 3.0));
  auto op = operator_with_spectrum<T>(spectrum, 51);
  for (index_t p : {2, 4}) {
    for (Which which : kAllWhich) {
      SCOPED_TRACE("p = " + std::to_string(p) + ", which " +
                   std::to_string(static_cast<int>(which)));
      KrylovSchurOptions<T> opt;
      opt.nev = 5;
      opt.block_size = p;
      const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
      EXPECT_TRUE(r.converged);
      EXPECT_EQ(r.restarts, 0);
      check_contract<T>(op, nullptr, r, which, opt);
      expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-12);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, BreakdownAgainstADeflationBlockOrthonormalOnlyToDelta) {
  // Review finding (P3 blockks): D orthonormal only to delta >> eps (the deflation check accepts
  // up to 1e-8; driver reruns pass Ritz vectors that are M-orthonormal to rounding times
  // cond(B)). Three distinct eigenvalues: the deflated block Krylov space becomes invariant after
  // a few blocks, the next block lies in span([D V]) up to delta, its first-pass remainder (~delta)
  // passed the breakdown test and the second Cholesky-QR pass threw InvalidInput. Every block size
  // must converge like p = 1. The deflated operator keeps >= 37 copies of 1 and of 3 (D has 3
  // columns), so the 5 wanted values are copies of the extreme eigenvalue.
  using T = TypeParam;
  const index_t n = 100;
  std::vector<double> spectrum;
  for (index_t i = 0; i < n; ++i) spectrum.push_back(i < 30 ? 1.0 : (i < 60 ? 2.0 : 3.0));
  auto op = operator_with_spectrum<T>(spectrum, 51);
  for (double delta : {1e-12, 1e-10}) {
    DenseMatrix<T> d(n, 3);
    for (index_t j = 0; j < 3; ++j) {
      hs::linalg::fill_random(d.col(j), n, 4242, static_cast<std::uint64_t>(j));
      std::vector<T> h(static_cast<std::size_t>(j) + 1);
      const double nrm =
          hs::linalg::orthogonalize_cgs2<T>(n, j, d.data(), n, nullptr, n, d.col(j), nullptr,
                                            h.data());
      hs::linalg::scal(n, T(1.0 / nrm), d.col(j));
    }
    std::vector<T> noise(static_cast<std::size_t>(n));
    for (index_t j = 0; j < 3; ++j) {
      hs::linalg::fill_random(noise.data(), n, 77, static_cast<std::uint64_t>(j));
      hs::linalg::axpy(n, T(delta), noise.data(), d.col(j));
    }
    for (index_t p : {1, 2, 4}) {
      for (Which which : {Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
        SCOPED_TRACE("delta " + std::to_string(delta) + ", p = " + std::to_string(p) +
                     ", which " + std::to_string(static_cast<int>(which)));
        KrylovSchurOptions<T> opt;
        opt.nev = 5;
        opt.block_size = p;
        opt.deflation = &d;
        const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
        EXPECT_TRUE(r.converged);
        const double value = which == Which::LargestAlgebraic ? 3.0 : 1.0;
        expect_values(r.theta, std::vector<double>(5, value), 1e-8);
        EXPECT_LT(gram_deviation<T>(r.vectors, nullptr, r.vectors, true), 1e-12);
        EXPECT_LT(gram_deviation<T>(d, nullptr, r.vectors, false), 1e-9);
      }
    }
  }
}

TYPED_TEST(LanczosBlockTyped, DeflationInBInnerProduct) {
  using T = TypeParam;
  const index_t n = 160;
  const auto a = hs::test::random_hermitian<T>(n, 0.06, 31);
  const auto b = hs::test::random_hpd<T>(n, 0.06, 32);
  const double sigma = -0.2;
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, sigma);
  DenseOperator<T> bop(bd);
  std::vector<double> theta_exact;
  for (double l : hs::test::dense_eigenvalues(a, &b)) theta_exact.push_back(1.0 / (l - sigma));
  const auto expect = wanted(theta_exact, Which::LargestMagnitude, 8);
  const double kappa = condition_number(hs::add(T(1), a, T(-sigma), b));

  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.block_size = 4;
  const auto first = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(first.converged);
  expect_values(first.theta, std::vector<double>(expect.begin(), expect.begin() + 4),
                1e-10 * std::abs(expect[0]));
  for (index_t p : {2, 3}) {
    for (bool store : {false, true}) {
      SCOPED_TRACE("p = " + std::to_string(p) + (store ? ", store_bv" : ", recompute"));
      opt.deflation = &first.vectors;
      opt.store_bv = store;
      opt.block_size = p;
      const auto second = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
      EXPECT_TRUE(second.converged);
      check_contract<T>(op, &bop, second, Which::LargestMagnitude, opt, kappa);
      expect_values(second.theta, std::vector<double>(expect.begin() + 4, expect.end()),
                    1e-10 * std::abs(expect[0]));
      EXPECT_LT(gram_deviation<T>(first.vectors, &bop, second.vectors, false), 1e-12);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, FastAndTwoPassBlockOrthogonalizationAgree) {
  // The block steps' default kernel (BlockOrthonormalizer::try_fast first) and the two-pass
  // kernel alone (BlockTuning::fast_ortho = false) return the same pairs, both satisfying the
  // contract, in the Euclidean and the B inner product and over restarts (nev 6 of a basis of a
  // few blocks); next to a deflation block both take the two-pass kernel (try_fast needs a basis
  // orthonormal to working precision, see BreakdownAgainstADeflationBlockOrthonormalOnlyToDelta).
  using T = TypeParam;
  const index_t n = 200;
  const auto a = hs::test::random_hermitian<T>(n, 0.05, 141);
  const auto b = hs::test::random_hpd<T>(n, 0.05, 142);
  const double sigma = 0.1;
  const auto bd = hs::test::to_dense(b);
  for (bool with_b : {false, true}) {
    hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), with_b ? &bd : nullptr, sigma);
    DenseOperator<T> bop(bd);
    LinearOperator<T>* m = with_b ? &bop : nullptr;
    std::vector<double> theta_exact;
    for (double l : hs::test::dense_eigenvalues(a, with_b ? &b : nullptr))
      theta_exact.push_back(1.0 / (l - sigma));
    const auto expect = wanted(theta_exact, Which::LargestMagnitude, 12);
    const double kappa = condition_number(with_b ? hs::add(T(1), a, T(-sigma), b)
                                                 : hs::add(T(1), a, T(-sigma), hs::identity<T>(n)));
    KrylovSchurOptions<T> first_opt;
    first_opt.nev = 6;
    first_opt.block_size = 2;
    const auto first = hs::lanczos_krylov_schur<T>(op, m, Which::LargestMagnitude, first_opt);
    ASSERT_TRUE(first.converged);
    for (bool deflate : {false, true}) {
      for (index_t p : {2, 3, 4}) {
        SCOPED_TRACE(std::string(with_b ? "B" : "Euclidean") + (deflate ? ", deflation" : "") +
                     ", p = " + std::to_string(p));
        KrylovSchurOptions<T> opt;
        opt.nev = 6;
        opt.ncv = 3 * p + 10;  // a few blocks: several restarts
        opt.tol = 1e-11;
        opt.block_size = p;
        if (deflate) opt.deflation = &first.vectors;
        const std::vector<double> want(expect.begin() + (deflate ? 6 : 0),
                                       expect.begin() + (deflate ? 12 : 6));
        HermitianEigResult<T> r[2];
        for (int fast = 0; fast < 2; ++fast) {
          hs::eigen_detail::BlockTuning tuning;
          tuning.fast_ortho = fast == 1;
          hs::eigen_detail::LanczosTimings tm;
          r[fast] = hs::eigen_detail::block_lanczos_krylov_schur_timed<T>(
              op, m, Which::LargestMagnitude, opt, tm, tuning);
          EXPECT_TRUE(r[fast].converged) << "fast " << fast;
          EXPECT_GT(r[fast].restarts, 0) << "fast " << fast;
          check_contract<T>(op, m, r[fast], Which::LargestMagnitude, opt, kappa);
          expect_values(r[fast].theta, want, 1e-10 * std::abs(expect[0]));
          if (deflate) {
            EXPECT_LT(gram_deviation<T>(first.vectors, m, r[fast].vectors, false), 1e-12);
          }
        }
        for (index_t i = 0; i < opt.nev; ++i)
          EXPECT_NEAR(r[1].theta[i], r[0].theta[i], 1e-10 * std::abs(r[0].theta[0])) << i;
      }
    }
  }
}

TYPED_TEST(LanczosBlockTyped, RowBlockedRotationLargeDeflationAndStoredBv) {
  // As RowBlockedRotationAndChunkedDeflationCheck, with blocks: n > 2 * 4096 (row-blocked
  // rotation of the basis and of M*V), d = 70 deflation columns checked in chunks.
  using T = TypeParam;
  const index_t n = 9001, nev = 5, d = 70, top = 80;
  std::vector<double> a(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n - top; ++i)
    a[i] = static_cast<double>(i) / static_cast<double>(n - top);
  for (index_t k = 0; k < top; ++k)
    a[n - top + k] = 1.0 + std::pow(1.1, static_cast<double>(k + 1));
  std::vector<double> m(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i)
    m[i] = 1.0 + 3.0 * std::fmod(0.6180339887 * static_cast<double>(i), 1.0);
  auto op = diagonal_operator<T>(a);
  auto mop = diagonal_operator<T>(m);
  KrylovSchurOptions<T> opt;
  opt.nev = nev;
  opt.ncv = 12;
  opt.block_size = 4;
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    opt.store_bv = store;
    const auto r = hs::lanczos_krylov_schur<T>(op, &mop, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    EXPECT_GT(r.restarts, 0);
    check_contract<T>(op, &mop, r, Which::LargestAlgebraic, opt);
    expect_values(r.theta, wanted(a, Which::LargestAlgebraic, nev), 1e-10);
  }
  DenseMatrix<T> defl(n, d);
  for (index_t c = 0; c < d; ++c) defl(n - 1 - c, c) = T(1.0 / std::sqrt(m[n - 1 - c]));
  opt.deflation = &defl;
  const auto all = wanted(a, Which::LargestAlgebraic, d + nev);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "deflated, store_bv" : "deflated, recompute");
    opt.store_bv = store;
    const auto r = hs::lanczos_krylov_schur<T>(op, &mop, Which::LargestAlgebraic, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, &mop, r, Which::LargestAlgebraic, opt);
    expect_values(r.theta, std::vector<double>(all.begin() + d, all.end()), 1e-10);
    EXPECT_LT(gram_deviation<T>(defl, &mop, r.vectors, false), 1e-13);
  }
}

TYPED_TEST(LanczosBlockTyped, EstimatesStayHonestOverManyRestarts) {
  // Blocks raise the restart count of this slow problem (ncv = 24: 130 restarts for p = 1, 950
  // for p = 4, measured), and every restart adds rounding drift of O(eps ||Op||) to the residuals
  // of locked pairs; ncv = 32 keeps the count (<= 300) where the contract's floor still applies.
  using T = TypeParam;
  const index_t n = 400;
  auto op = DenseOperator<T>(hs::test::to_dense(hs::test::laplacian_1d<T>(n)));
  const auto exact = hs::test::laplacian_1d_eigenvalues(n);
  for (index_t p : {2, 4}) {
    for (double tol : {1e-6, 1e-10}) {
      SCOPED_TRACE("p = " + std::to_string(p) + ", tol " + std::to_string(tol));
      KrylovSchurOptions<T> opt;
      opt.nev = 10;
      opt.ncv = 32;
      opt.tol = tol;
      opt.block_size = p;
      const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
      EXPECT_TRUE(r.converged);
      EXPECT_GT(r.restarts, 3);
      check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
      expect_values(r.theta, wanted(exact, Which::LargestAlgebraic, opt.nev), 4.0 * tol);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, LockedPairsReportHonestZeroEstimates) {
  // As ZeroEstimatesOfEarlyConvergedPairsAreHonest, with blocks of 2: decoupled pairs (estimate 0)
  // have true residuals at rounding level.
  using T = TypeParam;
  std::vector<double> spectrum = {10.0, 6.0, 3.0, 2.0};
  for (index_t i = 0; i < 396; ++i) spectrum.push_back(-1.0 + 2.0 * static_cast<double>(i) / 395.0);
  auto op = diagonal_operator<T>(spectrum);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.ncv = 10;
  opt.tol = 1e-12;
  opt.block_size = 2;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_GT(r.restarts, 2);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
  expect_values(r.theta, {10.0, 6.0, 3.0, 2.0}, 1e-11);
  const auto res = true_residuals<T>(op, nullptr, r, nullptr);
  for (index_t i = 0; i < opt.nev; ++i) {
    if (r.estimate[i] == 0.0) {
      EXPECT_LE(res[i], 1e-14 * std::abs(r.theta[i])) << "pair " << i;
    }
  }
}

TYPED_TEST(LanczosBlockTyped, StoredBvStaysOrthonormalOverManyRestarts) {
  using T = TypeParam;
  const index_t n = 400;
  const auto lap = hs::test::laplacian_1d<T>(n);
  const auto bd = hs::test::to_dense(hs::test::prescribed_spectrum<T>(
      std::vector<double>(static_cast<std::size_t>(n), 1.0), 8, 3));
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(lap), &bd, 0.0);
  DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 8;
  opt.ncv = 18;
  opt.tol = 1e-14;
  opt.max_restarts = 300;
  opt.store_bv = true;
  opt.block_size = 3;
  const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::SmallestAlgebraic, opt);
  EXPECT_EQ(r.restarts, opt.max_restarts);
  check_contract<T>(op, &bop, r, Which::SmallestAlgebraic, opt, condition_number(lap));
}

TYPED_TEST(LanczosBlockTyped, BookkeepingAndNonConvergence) {
  // ncv = 13 -> 14 (blocks of 2), keep = 6 + (14 - 6) / 2 = 10: 14 applications in the first cycle
  // and 4 per restart; non-convergence is reported, not thrown.
  using T = TypeParam;
  const index_t n = 400;
  auto op = DenseOperator<T>(hs::test::to_dense(hs::test::laplacian_1d<T>(n)));
  KrylovSchurOptions<T> opt;
  opt.nev = 6;
  opt.ncv = 13;
  opt.max_restarts = 2;
  opt.block_size = 2;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::SmallestAlgebraic, opt);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.restarts, 2);
  EXPECT_EQ(r.op_applies, 14 + 2 * 4);
  EXPECT_EQ(r.ritz_all.size(), 14u);
  check_contract<T>(op, nullptr, r, Which::SmallestAlgebraic, opt);
  ASSERT_EQ(r.history.size(), 3u);
  EXPECT_EQ(r.history[0].op_applies, 14);
  EXPECT_EQ(r.history[1].op_applies, 18);
}

TYPED_TEST(LanczosBlockTyped, CycleStopsEarlyAtTheFirstConvergedBlockBoundary) {
  // Four isolated top values converge within a few blocks. nev = 4, p = 2: single-vector auto
  // ncv = 20, block auto ncv = 20 + 4 * 2 = 28. The first cycle is probed from the single-vector
  // size 20 on and stops there: 20 applications instead of 28, a basis (and ritz_all) of 20, the
  // full result contract. With ncv = 28 requested the first cycle is not probed before its end.
  using T = TypeParam;
  std::vector<double> spectrum = {100.0, 90.0, 80.0, 70.0};
  for (index_t i = 0; i < 496; ++i) spectrum.push_back(static_cast<double>(i) / 495.0);
  auto op = operator_with_spectrum<T>(spectrum, 17);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.block_size = 2;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  EXPECT_EQ(r.op_applies, 20);
  EXPECT_EQ(r.ritz_all.size(), 20u);
  ASSERT_EQ(r.history.size(), 1u);
  EXPECT_EQ(r.history[0].op_applies, 20);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
  expect_values(r.theta, {100.0, 90.0, 80.0, 70.0}, 1e-10);
  opt.ncv = 28;
  const auto full = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(full.converged);
  EXPECT_EQ(full.op_applies, 28);
  EXPECT_EQ(full.ritz_all.size(), 28u);
  check_contract<T>(op, nullptr, full, Which::LargestAlgebraic, opt);
  expect_values(full.theta, {100.0, 90.0, 80.0, 70.0}, 1e-10);
}

TYPED_TEST(LanczosBlockTyped, FirstCycleIsNotProbedBeforeTheSingleVectorSize) {
  // Three distinct eigenvalues of multiplicity >= 30: the block Krylov space breaks down every
  // three blocks and each round adds p exact (converged) copies of every value. With p = 2,
  // nev = 5, ncv = 16 (keep = 10) a first-cycle probe at keep + p = 12 sees 4 copies of the
  // extreme value plus a converged copy of the next one and would accept that wrong set (measured
  // with the probe rule mutated to m >= keep + p in the first cycle: wrong in all four
  // configurations below, at 12 applications). The first cycle is probed only from the
  // single-vector basis size (here ncv itself): the run continues and returns 5 copies.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 100; ++i) spectrum.push_back(i < 30 ? 1.0 : (i < 60 ? 2.0 : 3.0));
  auto op = operator_with_spectrum<T>(spectrum, 51);
  for (auto [p, ncv] : {std::pair<index_t, index_t>{2, 16}, {4, 12}}) {
    for (Which which : {Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
      SCOPED_TRACE("p = " + std::to_string(p) + ", ncv = " + std::to_string(ncv) + ", which " +
                   std::to_string(static_cast<int>(which)));
      KrylovSchurOptions<T> opt;
      opt.nev = 5;
      opt.ncv = ncv;
      opt.block_size = p;
      const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, which, opt);
      EXPECT_TRUE(r.converged);
      EXPECT_GT(r.op_applies, 12);
      check_contract<T>(op, nullptr, r, which, opt);
      expect_values(r.theta, wanted(spectrum, which, opt.nev), 1e-12);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, SmallRequestedNcvKeepsASpareRitzPairAndTwoNewBlocks) {
  // Review finding (P3 blockks): a requested ncv near nev gave keep == nev (no spare Ritz pair)
  // and one new block per cycle, and block runs converged to a wrong wanted set far more often
  // than p = 1 (ncv = nev + 3: 13% of the runs for p = 3, 4). The basis now holds at least
  // round_up(nev + 1, p) + 2p vectors. Configurations below: wrong sets before for the listed p
  // (random shift-invert pencils, B inner product, LargestMagnitude, from the review sweep).
  using T = TypeParam;
  const index_t n = 150;
  const bool cplx_t = hs::is_complex_v<T>;
  const int s = 1;
  const double sigma = cplx_t ? -0.27 : 0.13;
  const index_t nev = cplx_t ? 12 : 7;
  const auto a = hs::test::random_hermitian<T>(n, 0.1, 1000 + 7 * s + n);
  const auto b = hs::test::random_hpd<T>(n, 0.1, 1000 + 7 * s + n + 1);
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, sigma);
  DenseOperator<T> bop(bd);
  std::vector<double> theta_exact;
  for (double l : hs::test::dense_eigenvalues(a, &b)) theta_exact.push_back(1.0 / (l - sigma));
  const auto expect = wanted(theta_exact, Which::LargestMagnitude, nev);
  std::vector<double> expect_sorted = expect;
  std::sort(expect_sorted.begin(), expect_sorted.end());
  const double kappa = condition_number(hs::add(T(1), a, T(-sigma), b));
  for (index_t p : {1, 2, 3, 4, 8}) {
    SCOPED_TRACE("p = " + std::to_string(p));
    KrylovSchurOptions<T> opt;
    opt.nev = nev;
    opt.ncv = nev + 3;
    opt.tol = 1e-11;
    opt.block_size = p;
    opt.seed = 2026 + s;
    const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    check_contract<T>(op, &bop, r, Which::LargestMagnitude, opt, kappa);
    std::vector<double> got = r.theta;
    std::sort(got.begin(), got.end());
    expect_values(got, expect_sorted, 1e-7 * std::abs(expect[0]));
    if (p > 1) {
      // The first cycle (not probed before its end for a requested ncv) spans the whole basis.
      const index_t spare = (nev + 1 + p - 1) / p * p;
      EXPECT_EQ(r.history.front().op_applies, spare + 2 * p);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, HugeBlockSizesTakeTheExactSingleVectorCycle) {
  // Review finding (P3 blockks): block sizes near INT64_MAX overflowed the layout arithmetic
  // (undefined behaviour; an unrelated DenseMatrix error or an internal assertion). Any p with
  // no feasible block layout solves the free space with one exact single-vector cycle.
  using T = TypeParam;
  const index_t n = 100;
  std::vector<double> spectrum;
  for (index_t i = 0; i < n; ++i) spectrum.push_back(1.0 + static_cast<double>(i));
  auto op = diagonal_operator<T>(spectrum);
  const index_t big = std::numeric_limits<index_t>::max();
  for (index_t p : {index_t{13}, index_t{101}, index_t{1} << 40, index_t{1} << 62, big / 2 - 1,
                    big / 2, big - 1, big}) {
    for (index_t ncv : {0, 10}) {
      SCOPED_TRACE("p = " + std::to_string(p) + ", ncv = " + std::to_string(ncv));
      KrylovSchurOptions<T> opt;
      opt.nev = 4;
      opt.ncv = ncv;
      opt.block_size = p;
      const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
      EXPECT_TRUE(r.converged);
      EXPECT_EQ(r.restarts, 0);
      EXPECT_EQ(r.op_applies, n);
      expect_values(r.theta, {100.0, 99.0, 98.0, 97.0}, 1e-10);
    }
  }
}

TYPED_TEST(LanczosBlockTyped, FreeSpaceBelowTwiceTheBlockBasisIsSolvedExactly) {
  // Review finding (P3 blockks): just above the old feasibility boundary (ncv + p <= n - d) a
  // restarted block run took several times n - d applications (nev = 4, p = 8: 104-200 against
  // 32-41). Free spaces below kFreeSpaceFactor (ncv + p) = 2 (24 + 8) now take the exact
  // single-vector cycle (n - d applications); from 64 on the block solver runs.
  using T = TypeParam;
  for (index_t n : {32, 41, 63, 64}) {
    SCOPED_TRACE("n = " + std::to_string(n));
    const auto a = hs::test::random_hermitian<T>(n, 0.1, 5);
    const auto b = hs::test::random_hpd<T>(n, 0.1, 6);
    const auto bd = hs::test::to_dense(b);
    hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, 0.0);
    DenseOperator<T> bop(bd);
    std::vector<double> theta_exact;
    for (double l : hs::test::dense_eigenvalues(a, &b)) theta_exact.push_back(1.0 / l);
    KrylovSchurOptions<T> opt;
    opt.nev = 4;
    opt.tol = 1e-11;
    opt.block_size = 8;
    const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
    EXPECT_TRUE(r.converged);
    const auto expect = wanted(theta_exact, Which::LargestMagnitude, 4);
    expect_values(r.theta, expect, 1e-9 * std::abs(expect[0]));
    if (n < 64) {
      EXPECT_EQ(r.op_applies, n);
      EXPECT_EQ(r.restarts, 0);
    } else {
      EXPECT_EQ(r.op_applies % 8, 0);
      EXPECT_EQ(r.ritz_all.size() % 8, 0u);
    }
  }
  // Deflated: n = 61, d = 20 leaves 41 < 64.
  const index_t n = 61;
  const auto a = hs::test::random_hermitian<T>(n, 0.1, 9);
  const auto b = hs::test::random_hpd<T>(n, 0.1, 10);
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, 0.0);
  DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 20;
  opt.tol = 1e-11;
  const auto first = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_TRUE(first.converged);
  opt.nev = 4;
  opt.block_size = 8;
  opt.deflation = &first.vectors;
  const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.op_applies, n - 20);
  EXPECT_LT(gram_deviation<T>(first.vectors, &bop, r.vectors, false), 1e-10);
}

TYPED_TEST(LanczosBlockTyped, SmallFreeSpaceIsSolvedExactly) {
  // n - d too small for a block basis plus its residual block: one exact single-vector cycle over
  // the free space (as ncv >= n - d does for p = 1), also with deflation.
  using T = TypeParam;
  std::vector<double> spectrum;
  for (index_t i = 0; i < 20; ++i) spectrum.push_back(std::pow(-1.0, i) * (1.0 + i));
  auto op = operator_with_spectrum<T>(spectrum, 81);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.block_size = 4;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  EXPECT_EQ(r.op_applies, 20);
  check_contract<T>(op, nullptr, r, Which::LargestMagnitude, opt);
  expect_values(r.theta, wanted(spectrum, Which::LargestMagnitude, 4), 1e-12);
  opt.deflation = &r.vectors;
  opt.nev = 16;
  const auto rest = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  EXPECT_TRUE(rest.converged);
  EXPECT_EQ(rest.op_applies, 16);
  check_contract<T>(op, nullptr, rest, Which::LargestMagnitude, opt);
  const auto all = wanted(spectrum, Which::LargestMagnitude, 20);
  expect_values(rest.theta, std::vector<double>(all.begin() + 4, all.end()), 1e-12);
}

TYPED_TEST(LanczosBlockTyped, InitialBlock) {
  using T = TypeParam;
  const index_t n = 500;
  std::vector<T> diag(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) diag[i] = T(static_cast<double>(i + 1));
  hs::test::DiagonalOperator<T> op(diag);
  // Two user columns inside the top-4 eigenspace: span{x0, x1, Op x0, Op x1} is that invariant
  // subspace, so the first cycle is exact.
  DenseMatrix<T> x0(n, 2);
  x0(n - 1, 0) = T(1.0);
  x0(n - 2, 0) = T(2.0);
  x0(n - 3, 1) = T(3.0);
  x0(n - 4, 1) = T(1.0);
  KrylovSchurOptions<T> opt;
  opt.nev = 4;
  opt.initial = &x0;
  opt.block_size = 2;
  const auto r = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r.converged);
  EXPECT_EQ(r.restarts, 0);
  check_contract<T>(op, nullptr, r, Which::LargestAlgebraic, opt);
  expect_values(r.theta, {500.0, 499.0, 498.0, 497.0}, 1e-11);
  // A user column dependent on the first is replaced by a random vector; the run still converges.
  DenseMatrix<T> dup(n, 2);
  for (index_t i = 0; i < n; ++i) dup(i, 0) = dup(i, 1) = T(1.0 + static_cast<double>(i % 7));
  opt.initial = &dup;
  opt.block_size = 3;
  const auto r2 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestAlgebraic, opt);
  EXPECT_TRUE(r2.converged);
  check_contract<T>(op, nullptr, r2, Which::LargestAlgebraic, opt);
  expect_values(r2.theta, {500.0, 499.0, 498.0, 497.0}, 1e-10);
}

TYPED_TEST(LanczosBlockTyped, ReproducibleForFixedSeed) {
  using T = TypeParam;
  auto op = operator_with_spectrum<T>(cubic_spectrum(250), 71);
  KrylovSchurOptions<T> opt;
  opt.nev = 7;
  opt.ncv = 18;
  opt.block_size = 4;
  const auto r1 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  const auto r2 = hs::lanczos_krylov_schur<T>(op, nullptr, Which::LargestMagnitude, opt);
  ASSERT_TRUE(r1.converged);
  EXPECT_EQ(0, std::memcmp(r1.theta.data(), r2.theta.data(), r1.theta.size() * sizeof(double)));
  EXPECT_EQ(
      0, std::memcmp(r1.vectors.data(), r2.vectors.data(),
                     static_cast<std::size_t>(r1.vectors.rows() * r1.vectors.cols()) * sizeof(T)));
  EXPECT_EQ(r1.restarts, r2.restarts);
  EXPECT_EQ(r1.op_applies, r2.op_applies);
}

TYPED_TEST(LanczosBlockTyped, BlockSizeOneKeepsTheSingleVectorBasisLayout) {
  // block_size = 1 (the default) is the single-vector solver: a basis of exactly ncv = 11 vectors
  // (no whole blocks), keep = 5 + (11 - 5) / 2 = 8, so 11 applications and 3 per restart. The
  // bitwise identity with the base commit's solver is pinned in test_lanczos_bitwise.cpp.
  using T = TypeParam;
  const index_t n = 120;
  const auto a = hs::test::random_hermitian<T>(n, 0.08, 61);
  const auto b = hs::test::random_hpd<T>(n, 0.08, 62);
  const auto bd = hs::test::to_dense(b);
  hs::test::DenseShiftInvertOperator<T> op(hs::test::to_dense(a), &bd, 0.05);
  DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 5;
  opt.ncv = 11;
  opt.block_size = 1;
  const auto r = hs::lanczos_krylov_schur<T>(op, &bop, Which::LargestMagnitude, opt);
  ASSERT_GT(r.restarts, 0);
  EXPECT_EQ(r.ritz_all.size(), 11u);
  EXPECT_EQ(r.op_applies, 11 + 3 * r.restarts);
  for (index_t c = 0; c <= r.restarts; ++c) EXPECT_EQ(r.history[c].op_applies, 11 + 3 * c);
}

TYPED_TEST(LanczosBlockTyped, InvalidAndMalformedInputsThrowInvalidInput) {
  using T = TypeParam;
  // n = 60 >= 2 (ncv + p) = 44 for nev = 3, p = 2: the block solver runs (not the exact cycle).
  const index_t n = 60;
  std::vector<double> spectrum(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) spectrum[i] = 1.0 + static_cast<double>(i);
  auto op = diagonal_operator<T>(spectrum);
  auto expect_invalid = [](LinearOperator<T>& o, LinearOperator<T>* b,
                           const KrylovSchurOptions<T>& opt, const char* what, const char* topic) {
    try {
      hs::lanczos_krylov_schur<T>(o, b, Which::LargestAlgebraic, opt);
      ADD_FAILURE() << "no exception for " << what;
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << what << ": " << e.what();
      EXPECT_NE(std::string(e.what()).find(topic), std::string::npos) << what << ": " << e.what();
    }
  };
  KrylovSchurOptions<T> base;
  base.nev = 3;
  base.block_size = 2;
  auto opt = base;
  for (index_t p : {0, -3}) {
    opt.block_size = p;
    expect_invalid(op, nullptr, opt, "block_size < 1", "block_size");
  }
  opt = base;
  opt.ncv = 3;
  expect_invalid(op, nullptr, opt, "ncv == nev < n", "ncv");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto nan_op = diagonal_operator<T>([&] {
    auto s = spectrum;
    s[7] = nan;
    return s;
  }());
  expect_invalid(nan_op, nullptr, base, "operator returning NaN", "non-finite");
  DenseMatrix<T> x0(n, 2);
  x0(3, 0) = T(nan);
  opt = base;
  opt.initial = &x0;
  expect_invalid(op, nullptr, opt, "initial block with a NaN in column 0", "initial");
  DenseMatrix<T> zero_first(n, 2);
  zero_first(0, 1) = T(1.0);
  opt.initial = &zero_first;
  expect_invalid(op, nullptr, opt, "zero first initial column", "initial");
  DenseMatrix<T> nan_second(n, 2);
  nan_second(0, 0) = T(1.0);
  nan_second(1, 1) = T(nan);
  opt.initial = &nan_second;
  expect_invalid(op, nullptr, opt, "NaN in initial column 1", "initial");
  // Initial vector inside span(D).
  DenseMatrix<T> d(n, 2);
  d(0, 0) = T(1.0);
  d(1, 1) = T(1.0);
  DenseMatrix<T> in_span(n, 1);
  in_span(1, 0) = T(2.0);
  opt = base;
  opt.deflation = &d;
  opt.initial = &in_span;
  expect_invalid(op, nullptr, opt, "initial vector inside span(D)", "span(deflation)");
  // Not positive definite b_inner.
  auto negative = diagonal_operator<T>(std::vector<double>(static_cast<std::size_t>(n), -1.0));
  std::vector<double> signs(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) signs[i] = i % 2 ? -1.0 : 1.0;
  auto indefinite = diagonal_operator<T>(signs);
  for (bool store : {false, true}) {
    SCOPED_TRACE(store ? "store_bv" : "recompute");
    opt = base;
    opt.store_bv = store;
    expect_invalid(op, &negative, opt, "negative definite b_inner", "positive definite");
    expect_invalid(op, &indefinite, opt, "indefinite b_inner", "positive definite");
  }
}

}  // namespace
