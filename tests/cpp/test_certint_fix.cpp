// Regression tests of the certint review fixes (src/driver/certificate.*, definite.cpp):
//   * fewer than k selected pairs never pass a completeness check: the deflated rerun looks for
//     the missing ones (fault injection removing the FARTHEST pair, all modes);
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "hsolve/solver.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

hs::SolverConfig nearest_cfg(double sigma, index_t k, hs::CertificateMode mode) {
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.certificate = mode;
  return cfg;
}

std::vector<double> nearest(std::vector<double> w, double sigma, index_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
  w.resize(static_cast<std::size_t>(k));
  return w;
}

void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

bool has_warning(const hs::SolveReport& r, const std::string& what) {
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

const char* mode_name(hs::CertificateMode m) {
  switch (m) {
    case hs::CertificateMode::Exact: return "exact";
    case hs::CertificateMode::Randomized: return "randomized";
    case hs::CertificateMode::Auto: return "auto";
    case hs::CertificateMode::Off: return "off";
  }
  return "?";
}

constexpr hs::CertificateMode kCheckingModes[] = {hs::CertificateMode::Exact, hs::CertificateMode::Randomized,
                                                  hs::CertificateMode::Auto};

template <class T>
class CertintFixTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(CertintFixTyped, Scalars);

// --------------------------------------------- fewer than k selected pairs: rerun, never pass

/// Fault injection on a spectrum without multiple eigenvalues removes the selected eigenvalue
/// farthest from sigma (and its Ritz hints). Then fewer than k pairs are selected, and no check
/// may pass that: the randomized claim disc (nearer half-width of the window) and even the count
/// window may end before the removed eigenvalue. Every mode reruns and finds all k.
template <class T>
void expect_farthest_drop_recovered(const std::vector<double>& w, double sigma, index_t k) {
  const auto a = hs::test::prescribed_spectrum<T>(w, 3, 29);
  for (const auto mode : kCheckingModes) {
    auto cfg = nearest_cfg(sigma, k, mode);
    cfg.certificate_fault_drop = 1;
    const auto sol = hs::solve(a, nullptr, cfg);
    const auto& r = sol.report;
    EXPECT_TRUE(r.success) << mode_name(mode) << ": " << r.certificate.statement;
    EXPECT_TRUE(r.certificate.passed) << mode_name(mode);
    EXPECT_TRUE(r.certificate.rigorous) << mode_name(mode);
    EXPECT_TRUE(has_warning(r, "certificate_fault_drop: 1 found pair(s) removed")) << mode_name(mode);
    EXPECT_TRUE(has_warning(r, "deflated rerun")) << mode_name(mode);
    EXPECT_FALSE(has_warning(r, "pairs found")) << mode_name(mode);  // "only k-1 of k pairs found"
    expect_same_set(sol.eigenvalues, nearest(w, sigma, k), 1e-10);
  }
}

// Review finding: the randomized pass radius (0.225, the window's nearer half-width) ends before
// the removed 0.3; the check passed and the solve returned 2 of 3 pairs.
TYPED_TEST(CertintFixTyped, FarthestDroppedOutsideTheClaimDiscIsRecovered) {
  std::vector<double> w{-0.1, 0.1, 0.3, -0.35};
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 0.0, 3);
}

// Review finding: an asymmetric window - the 10th nearest (1.13) lies inside the count window but
// beyond the symmetric pass radius (0.12, set by 0.85 below).
TYPED_TEST(CertintFixTyped, FarthestDroppedInAnAsymmetricWindowIsRecovered) {
  std::vector<double> w;
  for (int i = 1; i <= 9; ++i) w.push_back(1.0 + (i % 2 ? 1.0 : -1.0) * 0.01 * i);
  w.push_back(1.13);
  w.push_back(0.85);
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 1.0, 10);
}

// Review finding: the next eigenvalue (1.14) right beyond the removed 1.13 puts the count window's
// end mid-way between 1.09 and 1.14, before 1.13: the counts equal the found pairs, and even the
// exact mode returned k-1 pairs.
TYPED_TEST(CertintFixTyped, FarthestDroppedBeyondTheCountWindowIsRecovered) {
  std::vector<double> w;
  for (int i = 1; i <= 9; ++i) w.push_back(1.0 + (i % 2 ? 1.0 : -1.0) * 0.01 * i);
  w.push_back(1.13);
  w.push_back(1.14);
  w.push_back(0.84);
  for (int i = 0; i < 300; ++i) w.push_back(2.0 + 3.0 * i / 300.0);
  for (int i = 0; i < 100; ++i) w.push_back(-2.0 - 3.0 * i / 100.0);
  expect_farthest_drop_recovered<TypeParam>(w, 1.0, 10);
}

}  // namespace
