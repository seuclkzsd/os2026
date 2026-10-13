// InertiaCounter with the double-precision instance deferred (include/hsolve/factor.hpp): a
// counter used for single counts only pays one analysis, the double instance is analysed at its
// first use (double count or estimate()) and then counts exactly as an eager counter does.
#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <vector>

#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::index_t;

template <class T>
class CounterDeferred : public ::testing::Test {};
using Scalars = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(CounterDeferred, Scalars);

TYPED_TEST(CounterDeferred, DoubleInstanceIsAnalysedAtItsFirstUse) {
  using T = TypeParam;
  const index_t nx = 30, ny = 31;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  std::sort(w.begin(), w.end());
  const hs::ShiftedPencil<T> pencil(a, nullptr);
  ASSERT_GT(w[201] - w[200], 1e-6);
  ASSERT_GT(w[101] - w[100], 1e-6);
  const double sigma = 0.5 * (w[200] + w[201]);  // 201 eigenvalues below, in a gap
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -sigma);

  hs::InertiaCounter<T> eager(*f, f->options());
  EXPECT_TRUE(eager.double_analysed());
  EXPECT_EQ(eager.stats().analyses, 1);

  hs::InertiaCounter<T> deferred(*f, f->options(), /*defer_double_analysis=*/true);
  EXPECT_FALSE(deferred.double_analysed());
  EXPECT_EQ(deferred.stats().analyses, 0);
  EXPECT_EQ(deferred.order(), a.nrows);
  // Single counts: one analysis (the single instance), the double one still deferred.
  const hs::Inertia s = deferred.count(1.0, -sigma, true, hs::FactorPrecision::Single);
  EXPECT_TRUE(s.valid);
  EXPECT_EQ(s.neg, 201);
  EXPECT_EQ(deferred.stats().analyses, 1);
  EXPECT_EQ(deferred.stats().single_counts, 1);
  EXPECT_FALSE(deferred.double_analysed());
  // The first double count analyses the double instance and counts as the eager counter.
  const hs::Inertia d = deferred.count(1.0, -sigma);
  EXPECT_TRUE(deferred.double_analysed());
  EXPECT_EQ(deferred.stats().analyses, 2);
  const hs::Inertia e = eager.count(1.0, -sigma);
  EXPECT_TRUE(d.valid && e.valid);
  EXPECT_EQ(d.neg, 201);
  EXPECT_EQ(d.neg, e.neg);
  EXPECT_EQ(d.pos, e.pos);
  EXPECT_EQ(deferred.stats().counts, 2);
  // No further analysis for later counts of either precision.
  const auto pair = deferred.count_pair(1.0, -0.5 * (w[100] + w[101]), 1.0, -sigma);
  EXPECT_EQ(pair.second.neg - pair.first.neg, 100);
  static_cast<void>(deferred.count(1.0, -sigma, true, hs::FactorPrecision::Single));
  EXPECT_EQ(deferred.stats().analyses, 2);

  // estimate() of a deferred counter analyses the double instance (the eager counter's estimate).
  hs::InertiaCounter<T> lazy(*f, f->options(), true);
  const hs::FactorEstimate est = lazy.estimate();
  EXPECT_TRUE(lazy.double_analysed());
  EXPECT_EQ(lazy.stats().analyses, 1);
  EXPECT_EQ(est.factor_entries, eager.estimate().factor_entries);
  EXPECT_EQ(est.incore_mb, eager.estimate().incore_mb);
}

}  // namespace
