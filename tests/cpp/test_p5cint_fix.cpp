// Regression tests of the P5c integration review fixes (src/eigen, src/driver):
//   * KrylovSchurOptions::lead_nev: a run whose extra wanted pairs lie in a cluster wider than
//     tol (they never converge) ends once its leading pairs have converged.
#include <gtest/gtest.h>

#include <vector>

#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// `count` values c (1 + width u), u evenly spread in [-1, 1]: a cluster of relative width
/// `width`, far below any gap a Krylov polynomial of a few hundred degrees resolves.
std::vector<double> cluster(double c, double width, int count) {
  std::vector<double> v;
  for (int i = 0; i < count; ++i) v.push_back(c * (1.0 + width * (2.0 * i / (count - 1) - 1.0)));
  return v;
}

// ------------------------------------------------------------------- lead_nev (src/eigen)

// theta = 10 (the leading pair), then 200 eigenvalues within 1e-8 relative of 3, then the rest
// in [-2, 2]. Wanted: the 3 largest |theta|, i.e. 10 and two members of the cluster. Ritz vectors
// inside the cluster are mixtures whose estimates stay near its width (~1e-8 |theta| >> tol), so
// without lead_nev the run holds on until max_restarts; with lead_nev = 1 it ends as soon as 10
// has converged and returns all three pairs (the cluster ones unconverged).
template <class T>
void extra_pairs_in_a_wide_cluster(index_t p) {
  std::vector<double> d{10.0};
  for (double x : cluster(3.0, 1e-8, 200)) d.push_back(x);
  for (int i = 0; i < 400; ++i) d.push_back(-2.0 + 4.0 * i / 399.0);
  hs::test::DiagonalOperator<T> op(std::vector<T>(d.begin(), d.end()));
  hs::KrylovSchurOptions<T> ko;
  ko.nev = 3;
  ko.tol = 1e-11;
  ko.max_restarts = 30;
  ko.block_size = p;
  const auto held = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_EQ(held.restarts, 30) << "p = " << p << ": the cluster pairs no longer stall?";
  EXPECT_FALSE(held.converged) << p;
  ASSERT_GE(held.nconv, 1) << p;
  EXPECT_NEAR(held.theta[0], 10.0, 1e-9) << p;

  ko.lead_nev = 1;
  const auto led = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_LE(led.restarts, 3) << p;
  EXPECT_LT(led.op_applies, held.op_applies / 5) << p;
  ASSERT_EQ(led.theta.size(), 3u) << p;  // all nev pairs returned, converged first
  ASSERT_GE(led.nconv, 1) << p;
  EXPECT_FALSE(led.converged) << p;  // contract unchanged: nconv < nev
  EXPECT_NEAR(led.theta[0], 10.0, 1e-9) << p;
  EXPECT_LE(led.estimate[0], 1e-11 * 10.0) << p;
  ASSERT_EQ(led.estimate.size(), 3u) << p;  // the other two as far as they got (no claim)
  // lead_nev = nev is the default behaviour.
  ko.lead_nev = 3;
  const auto all = hs::lanczos_krylov_schur<T>(op, nullptr, hs::Which::LargestMagnitude, ko);
  EXPECT_EQ(all.restarts, held.restarts) << p;
  EXPECT_EQ(all.op_applies, held.op_applies) << p;
  EXPECT_EQ(all.theta, held.theta) << p;
}

TEST(P5cIntFixLead, ExtraPairsInAWideClusterDoNotHoldTheRunReal) {
  for (const index_t p : {1, 2, 4, 8}) extra_pairs_in_a_wide_cluster<double>(p);
}
TEST(P5cIntFixLead, ExtraPairsInAWideClusterDoNotHoldTheRunComplex) {
  for (const index_t p : {1, 4}) extra_pairs_in_a_wide_cluster<cplx>(p);
}

TEST(P5cIntFixLead, LeadNevIsValidated) {
  std::vector<double> d;
  for (int i = 0; i < 100; ++i) d.push_back(1.0 + i);
  hs::test::DiagonalOperator<double> op(d);
  hs::KrylovSchurOptions<double> ko;
  ko.nev = 4;
  for (const index_t lead : {index_t{-1}, index_t{5}}) {
    ko.lead_nev = lead;
    try {
      hs::lanczos_krylov_schur<double>(op, nullptr, hs::Which::LargestAlgebraic, ko);
      ADD_FAILURE() << "lead_nev " << lead << " accepted";
    } catch (const hs::Error& e) {
      EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
    }
  }
  ko.lead_nev = 1;
  try {
    hs::krylov_schur_general<double>(op, hs::Which::LargestMagnitude, ko);
    ADD_FAILURE() << "krylov_schur_general accepted lead_nev";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::InvalidInput) << e.what();
  }
}

}  // namespace
