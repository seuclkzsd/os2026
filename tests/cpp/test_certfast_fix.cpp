// Regression tests of the P4 review fixes of src/verify (include/hsolve/verify.hpp "(1)", "(3)"),
// on diagonal operators with known spectra:
//   * a start scaling without b_inner (B = I) is charged in kappa = cond(S^2); before the fix
//     kappa was 1 and a hidden eigenvalue under a tiny s_i passed with "P <= 1e-13";
//   * the rank test of the B-orthonormalization ignores the column scaling of X;
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/verify.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::ExclusionOptions;
using hs::ExclusionStatus;
using hs::index_t;

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

/// theta_0 = 1.25 (distance 0.8 from sigma = 0, inside r = 1), every other |theta| <= 0.6.
template <class T>
std::vector<T> hidden_spectrum(index_t n) {
  std::vector<T> d(static_cast<std::size_t>(n));
  d[0] = T(1.25);
  for (index_t i = 1; i < n; ++i)
    d[i] = T((i % 2 ? 0.6 : -0.6) * (1.0 - 0.3 * static_cast<double>(i) / static_cast<double>(n)));
  return d;
}

ExclusionOptions hidden_options(index_t p, std::uint64_t seed) {
  ExclusionOptions o;
  o.pass_radius = 1.0 / 0.7;  // eps = 0.3
  o.delta = 1e-12;
  o.block_size = p;
  o.seed = seed;
  return o;
}

// ---------------------------------------------------- start scaling without b_inner (findings 1, 6)

template <class T>
class VerifyFixScaling : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(VerifyFixScaling, Scalars);

// A tiny s_0 on the hidden eigenvector suppresses its start component; kappa = cond(S^2) pays for
// it, so the planned Krylov space still finds the eigenvalue. Before the fix every case here
// returned Passed with kappa = 1 (s_0 = 1e-40 and 1e-300: 20/20 seeds).
TYPED_TEST(VerifyFixScaling, TinyScalingOnTheMissedVectorIsChargedInKappa) {
  using T = TypeParam;
  const index_t n = 1000;
  hs::test::DiagonalOperator<T> op(hidden_spectrum<T>(n));
  const DenseMatrix<T> none(n, 0);
  for (double s0 : {1e-8, 1e-40}) {
    std::vector<double> scaling(static_cast<std::size_t>(n), 1.0);
    scaling[0] = s0;
    for (index_t p : {1, 4})
      for (std::uint64_t seed : {1u, 2u}) {
        ExclusionOptions o = hidden_options(p, seed);
        o.start_scaling = &scaling;
        const auto r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
        EXPECT_NEAR(r.kappa, 1.0 / (s0 * s0), 1e-12 / (s0 * s0)) << "s0=" << s0;
        EXPECT_TRUE(r.rigorous);
        EXPECT_EQ(r.status, ExclusionStatus::Detected)
            << "s0=" << s0 << " p=" << p << " seed=" << seed << ": " << r.statement;
        EXPECT_EQ(r.new_inside, 1);
        ASSERT_GE(r.new_lambda.size(), 1u);
        EXPECT_NEAR(r.new_lambda[0], 0.8, 1e-8);
      }
  }
  // (max s / min s)^2 overflows: no statement, nothing run.
  std::vector<double> scaling(static_cast<std::size_t>(n), 1.0);
  scaling[0] = 1e-300;
  ExclusionOptions o = hidden_options(4, 1);
  o.start_scaling = &scaling;
  const auto r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_EQ(r.status, ExclusionStatus::Inconclusive) << r.statement;
  EXPECT_TRUE(std::isinf(r.kappa));
  EXPECT_EQ(r.op_applications, 0);
  EXPECT_EQ(r.failure_probability, 1.0);
}

TYPED_TEST(VerifyFixScaling, KappaWithoutBInnerIsExactAndCombinesWithBCondition) {
  using T = TypeParam;
  const index_t n = 300;
  std::vector<T> d(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) d[i] = T(0.5 * (1.0 - static_cast<double>(i) / static_cast<double>(n)));
  hs::test::DiagonalOperator<T> op(d);  // nothing within r = 1
  const DenseMatrix<T> none(n, 0);
  // A uniform scaling changes nothing: kappa = 1.
  std::vector<double> uniform(static_cast<std::size_t>(n), 3.0);
  ExclusionOptions o = hidden_options(1, 5);
  o.start_scaling = &uniform;
  auto r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_EQ(r.kappa, 1.0);
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  // A graded scaling: kappa = (max / min)^2 = 1e4, and a larger b_condition wins.
  std::vector<double> graded(static_cast<std::size_t>(n), 1.0);
  graded[7] = 100.0;
  o.start_scaling = &graded;
  r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_NEAR(r.kappa, 1e4, 1e-8);
  EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
  o.b_condition = 1e6;
  r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_EQ(r.kappa, 1e6);
  o.b_condition = 2.0;  // smaller than cond(S^2): ignored
  r = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_NEAR(r.kappa, 1e4, 1e-8);
  // A larger kappa needs at least as many steps.
  o.b_condition = 0.0;
  o.start_scaling = nullptr;
  const auto plain = hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o);
  EXPECT_EQ(plain.kappa, 1.0);
  EXPECT_GE(r.block_steps_planned, plain.block_steps_planned);
  // b_condition without b_inner: 0 (unset) or a finite bound >= 1.
  for (double bad : {0.5, -1.0, std::numeric_limits<double>::infinity(), std::nan("")}) {
    o.b_condition = bad;
    EXPECT_EQ(error_code_of([&] { hs::exclusion_check<T>(op, nullptr, none, 0.0, 1.0, o); }),
              hs::Error::Code::InvalidInput)
        << bad;
  }
}

// ------------------------------------------------ column scaling of X (findings 2, 8)

TYPED_TEST(VerifyFixScaling, ColumnScaledBasisIsAccepted) {
  using T = TypeParam;
  const index_t n = 200;
  std::vector<T> d(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i)
    d[i] = T(i < 4 ? 10.0 + static_cast<double>(i) : 0.5 * (1.0 - static_cast<double>(i) / static_cast<double>(n)));
  hs::test::DiagonalOperator<T> op(d);
  const auto check = [&](const std::vector<double>& col_scale) {
    DenseMatrix<T> x(n, static_cast<index_t>(col_scale.size()));
    for (index_t j = 0; j < x.cols(); ++j) x(j, j) = T(col_scale[static_cast<std::size_t>(j)]);
    ExclusionOptions o;
    o.pass_radius = 1.0 / 0.7;
    return hs::exclusion_check<T>(op, nullptr, x, 0.0, 1.0 / 9.0, o);
  };
  // Exact eigenvectors normalized very differently (before the fix: "numerically rank deficient,
  // diagonal ratio 1e-09" / "1e-08").
  for (const std::vector<double>& scale :
       {std::vector<double>{1.0, 1e-3, 1e-6, 1e-9}, std::vector<double>{1e4, 1.0, 1e-4, 1.0}}) {
    const auto r = check(scale);
    EXPECT_EQ(r.status, ExclusionStatus::Passed) << r.statement;
    EXPECT_EQ(r.found_inside, 4);
    EXPECT_LT(r.coupling, 1e-12);
    ASSERT_EQ(r.deflated_theta.size(), 4u);
    EXPECT_NEAR(r.deflated_theta.front(), 13.0, 1e-12);
  }
  // Still rejected: a zero column, a scaled duplicate, a nearly dependent pair (cond ~1e9).
  const auto invalid = hs::Error::Code::InvalidInput;
  EXPECT_EQ(error_code_of([&] { check({1.0, 0.0, 1.0}); }), invalid);
  DenseMatrix<T> dup(n, 2);
  dup(0, 0) = T(1.0);
  dup(0, 1) = T(1e-6);
  ExclusionOptions o;
  o.pass_radius = 1.0 / 0.7;
  EXPECT_EQ(error_code_of([&] { hs::exclusion_check<T>(op, nullptr, dup, 0.0, 1.0 / 9.0, o); }), invalid);
  DenseMatrix<T> near(n, 2);
  near(0, 0) = T(1.0);
  near(0, 1) = T(1e3);
  near(1, 1) = T(1e-6);  // column 1 = 1e3 (e_0 + 1e-9 e_1)
  EXPECT_EQ(error_code_of([&] { hs::exclusion_check<T>(op, nullptr, near, 0.0, 1.0 / 9.0, o); }), invalid);
}

}  // namespace
