// Tests of the single-precision certificate rule and the certificate cost model (src/verify,
// include/hsolve/verify.hpp "(2)" and "(3)").
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/verify.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::ShiftedPencil;
using hs::SingleWindow;

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

constexpr double kUnitSingle = 5.9604644775390625e-8;

/// diag(d) as CSR.
template <class T>
CsrMatrix<T> diagonal(const std::vector<double>& d) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  for (index_t i = 0; i < static_cast<index_t>(d.size()); ++i) r.push_back(i), c.push_back(i), v.push_back(T(d[i]));
  return hs::csr_from_coo<T>(static_cast<index_t>(d.size()), static_cast<index_t>(d.size()), r, c, v);
}

/// D^{1/2} A D^{1/2}: the pencil (D^{1/2} A D^{1/2}, D) has the spectrum of A.
template <class T>
CsrMatrix<T> congruence(const CsrMatrix<T>& a, const std::vector<double>& d) {
  CsrMatrix<T> s = a;
  for (index_t i = 0; i < s.nrows; ++i)
    for (index_t q = s.rowptr[i]; q < s.rowptr[i + 1]; ++q) s.values[q] *= T(std::sqrt(d[i] * d[s.colind[q]]));
  return s;
}

std::vector<double> diag_values(index_t n) {
  std::vector<double> d(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) d[i] = 0.5 + static_cast<double>((i * 37) % 11) / 4.0;  // [0.5, 3]
  return d;
}

/// A window around sigma: ends mid-way in the gaps around the k eigenvalues nearest sigma.
struct Window {
  SingleWindow w;
  std::vector<double> found;
};
Window window_around(const std::vector<double>& eig, double sigma, index_t k, double min_gap = 1e-9) {
  std::vector<double> e = eig;
  std::sort(e.begin(), e.end());
  index_t lo = std::lower_bound(e.begin(), e.end(), sigma) - e.begin(), hi = lo;  // [lo, hi)
  while (hi - lo < k) {
    if (lo == 0 || (hi < static_cast<index_t>(e.size()) && e[hi] - sigma < sigma - e[lo - 1])) {
      ++hi;
    } else {
      --lo;
    }
  }
  // Whole multiplets, and ends in gaps of at least min_gap.
  while (lo > 0 && e[lo] - e[lo - 1] < min_gap) --lo;
  while (hi < static_cast<index_t>(e.size()) && e[hi] - e[hi - 1] < min_gap) ++hi;
  Window win;
  win.w.lower = 0.5 * (e[lo - 1] + e[lo]);
  win.w.upper = 0.5 * (e[hi - 1] + e[hi]);
  win.w.wanted_lower = e[lo];
  win.w.wanted_upper = e[hi - 1];
  win.found.assign(e.begin() + lo, e.begin() + hi);
  win.w.found = win.found;
  return win;
}

template <class T>
class VerifySingle : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(VerifySingle, Scalars);

TYPED_TEST(VerifySingle, RuleAppliesOnlyToIdentityOrDiagonalB) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::laplacian_2d<T>(12, 13);
  const auto eig = hs::test::laplacian_2d_eigenvalues(12, 13);
  const Window win = window_around(eig, 3.1, 6);
  // B = I.
  {
    ShiftedPencil<T> pencil(a, nullptr);
    const auto v = hs::single_count_validity(pencil, win.w);
    EXPECT_TRUE(v.applicable);
    EXPECT_TRUE(v.valid()) << v.reason;
    EXPECT_EQ(v.b_min, 1.0);
    const double emb = hs::is_complex_v<T> ? std::sqrt(2.0) : 1.0;
    const double norm_a = hs::norm_one(a);
    EXPECT_NEAR(v.eta_lower, 1e3 * kUnitSingle * emb * (norm_a + std::abs(win.w.lower)), 1e-15);
    EXPECT_NEAR(v.eta_upper, 1e3 * kUnitSingle * emb * (norm_a + std::abs(win.w.upper)), 1e-15);
  }
  // Diagonal B: eta divided by min b_ii.
  {
    const auto d = diag_values(a.nrows);
    const CsrMatrix<T> ad = congruence(a, d), bd = diagonal<T>(d);
    ShiftedPencil<T> pencil(ad, &bd);
    const auto v = hs::single_count_validity(pencil, win.w);
    EXPECT_TRUE(v.valid()) << v.reason;
    EXPECT_EQ(v.b_min, 0.5);
  }
  // A general HPD B: no certificate.
  {
    const CsrMatrix<T> b = hs::test::random_hpd<T>(a.nrows, 0.02, 4);
    ShiftedPencil<T> pencil(a, &b);
    const auto v = hs::single_count_validity(pencil, win.w);
    EXPECT_FALSE(v.applicable);
    EXPECT_FALSE(v.valid());
    EXPECT_FALSE(v.reason.empty());
  }
}

TYPED_TEST(VerifySingle, MarginsMustClearTheWantedRegionAndTheFoundEigenvalues) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::laplacian_2d<T>(12, 13);
  ShiftedPencil<T> pencil(a, nullptr);
  const auto eig = hs::test::laplacian_2d_eigenvalues(12, 13);
  Window win = window_around(eig, 3.1, 6);
  // A larger assumed backward error swallows the gaps.
  hs::SingleCountRule loose;
  loose.backward_factor = 1e7;
  const auto v1 = hs::single_count_validity(pencil, win.w, loose);
  EXPECT_TRUE(v1.applicable);
  EXPECT_FALSE(v1.margins_ok);
  EXPECT_FALSE(v1.reason.empty());
  // A found eigenvalue (with its error bar) reaching an end.
  win.w.found_error.assign(win.w.found.size(), 0.0);
  win.w.found_error.back() = win.w.upper - win.w.found.back();
  EXPECT_FALSE(hs::single_count_validity(pencil, win.w).margins_ok);
  // Malformed windows.
  SingleWindow bad = win.w;
  bad.found_error.clear();
  bad.wanted_upper = bad.upper;
  EXPECT_EQ(error_code_of([&] { hs::single_count_validity(pencil, bad); }), hs::Error::Code::InvalidInput);
  bad = win.w;
  bad.found_error = {1.0};
  EXPECT_EQ(error_code_of([&] { hs::single_count_validity(pencil, bad); }), hs::Error::Code::InvalidInput);
  bad = win.w;
  bad.lower = bad.upper;
  EXPECT_EQ(error_code_of([&] { hs::single_count_validity(pencil, bad); }), hs::Error::Code::InvalidInput);
}

// Single counts on the main analysis certify a complete window and expose a missing pair; both
// with B = I and with a diagonal B (same spectrum by congruence).
TYPED_TEST(VerifySingle, CertifiesCompleteWindowsAndExposesAMissingPair) {
  using T = TypeParam;
  // Coarse enough that the window ends clear eta (diagonal B: 1e3 u_s ||K||_1 / min b_ii ~ 4e-3)
  // on both sides; on a 30 x 31 grid one gap is only 2e-3 and the rule rightly refuses.
  const index_t nx = 16, ny = 17;
  const CsrMatrix<T> lap = hs::test::laplacian_2d<T>(nx, ny);
  const auto eig = hs::test::laplacian_2d_eigenvalues(nx, ny);
  const auto d = diag_values(nx * ny);
  const CsrMatrix<T> ad = congruence(lap, d), bd = diagonal<T>(d);
  for (bool with_b : {false, true}) {
    ShiftedPencil<T> pencil(with_b ? ad : lap, with_b ? &bd : nullptr);
    const double sigma = 2.7;
    auto f = hs::make_mumps_factorization<T>();
    f->analyze(pencil, 1.0, -sigma);
    hs::InertiaCounter<T> counter(*f, f->options());
    Window win = window_around(eig, sigma, 8, 1e-2);
    const auto cert = hs::single_precision_certificate(counter, pencil, win.w);
    EXPECT_TRUE(cert.validity.valid()) << cert.validity.reason;
    EXPECT_TRUE(cert.certified) << "count " << cert.count << " found " << win.w.found.size();
    EXPECT_EQ(cert.count, static_cast<index_t>(win.found.size()));
    EXPECT_TRUE(cert.lower.valid && cert.upper.valid);
    EXPECT_EQ(counter.stats().single_counts, 2);
    // The exact counts agree.
    const auto [lo, hi] = counter.count_pair(1.0, -win.w.lower, 1.0, -win.w.upper);
    EXPECT_EQ(hi.neg - lo.neg, cert.count);
    // One found pair fewer: the counts expose it.
    win.w.found.erase(win.w.found.begin() + 1);
    const auto miss = hs::single_precision_certificate(counter, pencil, win.w);
    EXPECT_TRUE(miss.validity.valid());
    EXPECT_FALSE(miss.certified);
    EXPECT_EQ(miss.count, static_cast<index_t>(win.w.found.size()) + 1);
  }
}

TYPED_TEST(VerifySingle, NoCountsWhereTheRuleFails) {
  using T = TypeParam;
  const CsrMatrix<T> a = hs::test::laplacian_2d<T>(15, 16);
  const CsrMatrix<T> b = hs::test::random_hpd<T>(a.nrows, 0.02, 9);
  ShiftedPencil<T> pencil(a, &b);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -0.5);
  hs::InertiaCounter<T> counter(*f, f->options());
  SingleWindow w;
  w.lower = 0.1;
  w.upper = 0.9;
  w.wanted_lower = 0.2;
  w.wanted_upper = 0.8;
  const auto cert = hs::single_precision_certificate(counter, pencil, w);
  EXPECT_FALSE(cert.validity.applicable);
  EXPECT_FALSE(cert.certified);
  EXPECT_EQ(cert.count, -1);
  EXPECT_EQ(counter.stats().counts, 0);
}

// ------------------------------------------------------------------------------- cost model

hs::CertificateCostInputs inputs_3d() {
  // syn-mem27-46-like at 2 threads: factorization 1.9 s, one solve 0.1 s, huge gap.
  hs::CertificateCostInputs in;
  in.n = 97336;
  in.deflated = 12;
  in.factorize_s = 1.9;
  in.counter_analysis_s = 0.06;
  in.apply1_s = 0.1;
  in.epsilon = 0.99;
  in.delta = 1e-12;
  return in;
}

TEST(VerifyCost, RandomizedWinsWithExpensiveFactorizationsAndWideGaps) {
  const auto est = hs::certificate_cost(inputs_3d());
  EXPECT_EQ(est.recommended, hs::CertificateMethod::Randomized);
  EXPECT_LT(est.randomized_s, est.exact_s);
  EXPECT_NEAR(est.exact_s, std::min(3.8, 0.06 + 2 * 0.85 * 1.9), 1e-12);
  EXPECT_TRUE(std::isinf(est.exact_single_s));
  EXPECT_LE(est.failure_bound, 1e-12);
  EXPECT_EQ(est.block_steps,
            hs::exclusion_block_steps(0.99, 1e-12, 97336 - 12, est.block_size, 1.0, false));
}

TEST(VerifyCost, ExactWinsWithCheapFactorizationsOrNarrowGaps) {
  auto in = inputs_3d();
  in.factorize_s = 0.05;  // low-fill 2-D case
  EXPECT_EQ(hs::certificate_cost(in).recommended, hs::CertificateMethod::Exact);
  in = inputs_3d();
  in.epsilon = 0.01;
  EXPECT_EQ(hs::certificate_cost(in).recommended, hs::CertificateMethod::Exact);
  in.epsilon = 0.0;
  const auto none = hs::certificate_cost(in);
  EXPECT_TRUE(std::isinf(none.randomized_s));
  EXPECT_EQ(none.recommended, hs::CertificateMethod::Exact);
}

TEST(VerifyCost, SingleCountsWhenTheRuleHolds) {
  auto in = inputs_3d();
  in.epsilon = 0.05;  // randomized expensive
  in.single_valid = true;
  const auto est = hs::certificate_cost(in);
  EXPECT_EQ(est.recommended, hs::CertificateMethod::ExactSingle);
  EXPECT_NEAR(est.exact_single_s, 0.06 + 2 * 0.55 * 0.85 * 1.9, 1e-12);
}

TEST(VerifyCost, BlockSizeFollowsTheMultipleRightHandSideCost) {
  auto in = inputs_3d();
  in.epsilon = 0.25;
  in.rhs_ratio = 1.0;  // no amortization: fewest vectors wins
  const auto p1 = hs::certificate_cost(in);
  in.rhs_ratio = 0.1;  // cheap extra right-hand sides: blocks win
  const auto p8 = hs::certificate_cost(in);
  EXPECT_LT(p1.block_size, p8.block_size);
  EXPECT_EQ(p1.block_size, 1);
  // A known coupling saves the block application to X.
  in.coupling_known = true;
  EXPECT_LT(hs::certificate_cost(in).randomized_s, p8.randomized_s);
}

TEST(VerifyCost, InvalidInputsThrow) {
  auto in = inputs_3d();
  in.factorize_s = -1.0;
  EXPECT_EQ(error_code_of([&] { hs::certificate_cost(in); }), hs::Error::Code::InvalidInput);
  in = inputs_3d();
  in.deflated = in.n;
  EXPECT_EQ(error_code_of([&] { hs::certificate_cost(in); }), hs::Error::Code::InvalidInput);
  in = inputs_3d();
  in.preference = 0.5;
  EXPECT_EQ(error_code_of([&] { hs::certificate_cost(in); }), hs::Error::Code::InvalidInput);
  EXPECT_STREQ(hs::certificate_method_name(hs::CertificateMethod::Randomized), "randomized");
  EXPECT_STREQ(hs::exclusion_status_name(hs::ExclusionStatus::Detected), "detected");
}

}  // namespace
