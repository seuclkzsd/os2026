// Problems and checks shared by the multi-process tests (test_parallel_session.cpp over thread
// transports, mpi/test_mpi_session.cpp over MPI): a 3-D Dirichlet Laplacian with analytic spectrum,
// real or (gauge-transformed, every coupling complex) complex, and the targets the helpers serve.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "hsolve/parallel.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "support/fixtures.hpp"

namespace hs::partest {

/// The test problems are far too small for the cost rules (Auto): force the helpers' work.
inline const ParallelOptions kForce{Toggle::On, Toggle::On};

constexpr index_t kNx = 13, kNy = 15, kNz = 17;  // n = 3315, simple wanted eigenvalues
constexpr double kSigma = 0.9;
constexpr index_t kK = 10;
constexpr double kLower = 0.6, kUpper = 1.4;  // an interval of a few dozen eigenvalues

inline std::vector<double> spectrum() {
  std::vector<double> w;
  const auto wx = test::laplacian_1d_eigenvalues(kNx), wy = test::laplacian_1d_eigenvalues(kNy),
             wz = test::laplacian_1d_eigenvalues(kNz);
  for (double a : wx)
    for (double b : wy)
      for (double c : wz) w.push_back(a + b + c);
  std::sort(w.begin(), w.end());
  return w;
}

inline std::vector<double> nearest(std::vector<double> w, double sigma, index_t k) {
  std::stable_sort(w.begin(), w.end(), [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
  w.resize(static_cast<std::size_t>(k));
  return w;
}

inline std::vector<double> inside(const std::vector<double>& w, double lo, double hi) {
  std::vector<double> out;
  for (double x : w)
    if (x >= lo && x <= hi) out.push_back(x);
  return out;
}

inline void expect_same_set(std::vector<double> got, std::vector<double> want, double tol) {
  ASSERT_EQ(got.size(), want.size());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], tol * std::max(1.0, std::abs(want[i]))) << i;
}

inline std::string warnings(const SolveReport& r) {
  std::string s;
  for (const auto& w : r.warnings) s += w + "\n";
  for (const auto& w : r.parallel.notes) s += "note: " + w + "\n";
  return s;
}

inline bool has_note(const SolveReport& r, const std::string& what) {
  for (const auto& w : r.parallel.notes)
    if (w.find(what) != std::string::npos) return true;
  for (const auto& w : r.warnings)
    if (w.find(what) != std::string::npos) return true;
  return false;
}

template <class T>
CsrMatrix<T> laplacian3d() {
  const auto id = [](index_t i, index_t j, index_t k) { return i + kNx * (j + kNy * k); };
  const index_t n = kNx * kNy * kNz;
  std::vector<double> phi(static_cast<std::size_t>(n));
  for (index_t v = 0; v < n; ++v) phi[static_cast<std::size_t>(v)] = 0.7 * static_cast<double>(v % 97);
  std::vector<index_t> rows, cols;
  std::vector<T> vals;
  auto add = [&](index_t r, index_t c, double x) {
    rows.push_back(r);
    cols.push_back(c);
    if constexpr (is_complex_v<T>)
      vals.push_back(r == c ? T(x)
                            : T(x) * std::polar(1.0, phi[static_cast<std::size_t>(c)] - phi[static_cast<std::size_t>(r)]));
    else
      vals.push_back(T(x));
  };
  for (index_t k = 0; k < kNz; ++k)
    for (index_t j = 0; j < kNy; ++j)
      for (index_t i = 0; i < kNx; ++i) {
        const index_t v = id(i, j, k);
        add(v, v, 6.0);
        if (i > 0) add(v, id(i - 1, j, k), -1.0);
        if (i + 1 < kNx) add(v, id(i + 1, j, k), -1.0);
        if (j > 0) add(v, id(i, j - 1, k), -1.0);
        if (j + 1 < kNy) add(v, id(i, j + 1, k), -1.0);
        if (k > 0) add(v, id(i, j, k - 1), -1.0);
        if (k + 1 < kNz) add(v, id(i, j, k + 1), -1.0);
      }
  return csr_from_coo<T>(n, n, rows, cols, vals);
}

inline SolverConfig nearest_cfg(index_t faults = 0) {
  SolverConfig cfg;
  cfg.target.mode = TargetMode::Nearest;
  cfg.target.sigma = kSigma;
  cfg.target.k = kK;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.certificate = CertificateMode::Exact;
  cfg.certificate_fault_count = faults;
  return cfg;
}

inline SolverConfig interval_cfg(index_t faults = 0) {
  SolverConfig cfg = nearest_cfg(faults);
  cfg.target.mode = TargetMode::Interval;
  cfg.target.lower = kLower;
  cfg.target.upper = kUpper;
  cfg.slice_size = 8;  // several slices
  return cfg;
}

/// The checks every multi-process nearest solve must pass whatever the helpers did: the k nearest
/// eigenvalues, certified by exact counts, phases adding up.
template <class T>
void expect_nearest_certified(const Solution<T>& sol) {
  const auto& r = sol.report;
  EXPECT_TRUE(r.success) << warnings(r);
  EXPECT_TRUE(r.certificate.passed) << r.certificate.statement;
  EXPECT_EQ(r.certificate.method, "exact");
  EXPECT_EQ(r.certificate.count_inertia, kK);
  EXPECT_EQ(r.certificate.count_found, kK);
  expect_same_set(sol.eigenvalues, nearest(spectrum(), kSigma, kK), 1e-9);
  EXPECT_NEAR(r.time_setup_s + r.time_factor_s + r.time_iterate_s + r.time_verify_s + r.time_certificate_s,
              r.time_total_s, 1e-9 * std::max(1.0, r.time_total_s));
}

template <class T>
void expect_interval_certified(const Solution<T>& sol) {
  const auto& r = sol.report;
  const auto want = inside(spectrum(), kLower, kUpper);
  EXPECT_TRUE(r.success) << warnings(r);
  EXPECT_TRUE(r.certificate.passed) << r.certificate.statement;
  EXPECT_EQ(r.certificate.count_inertia, static_cast<index_t>(want.size()));
  EXPECT_GE(r.slices.size(), 3u);
  for (const auto& s : r.slices) EXPECT_TRUE(s.passed);
  expect_same_set(sol.eigenvalues, want, 1e-9);
  // eigenvectors: B-orthonormal (B = I) and matching their eigenvalues
  ASSERT_EQ(sol.eigenvectors.cols(), static_cast<index_t>(sol.eigenvalues.size()));
  for (const auto& p : r.pairs) EXPECT_TRUE(p.accepted);
}

}  // namespace hs::partest
