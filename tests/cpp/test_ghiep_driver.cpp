// GHIEP path of hs::solve with block Krylov-Schur runs (module ghiep, P4): pencils of order
// >= 20000, where the driver's automatic rule picks block sizes > 1, with known spectra (sparse,
// block diagonal), so the computed sets are checked against analytic eigenvalues. Covers the
// plain nearest run, the zero-cluster one-sided runs (the cluster is never returned), complex
// pairs and complex Hermitian pencils, and the refinement's outputs (unit vectors whose residuals
// match the report).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::index_t;

template <class T>
struct Pencil {
  hs::CsrMatrix<T> a, b;
};

/// 2 x 2 Householder reflector I - 2 v v^H / (v^H v) (unitary and Hermitian) from a deterministic
/// v (complex for T = cplx), seeded by `index`.
template <class T>
std::array<std::array<T, 2>, 2> reflector(index_t index) {
  T v[2];
  const double phi = 0.7 + 1.3 * std::sin(0.37 * static_cast<double>(index));
  if constexpr (hs::is_complex_v<T>) {
    v[0] = T(1.0, 0.0);
    v[1] = T(std::cos(phi), std::sin(0.5 * phi));
  } else {
    v[0] = 1.0;
    v[1] = std::cos(phi);
  }
  const double vv = std::norm(v[0]) + std::norm(v[1]);
  std::array<std::array<T, 2>, 2> q{};
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c)
      q[r][c] = T(r == c ? 1.0 : 0.0) - T(2.0 / vv) * v[r] * hs::conj_if(v[c]);
  return q;
}

/// Banded pencil with a prescribed spectrum: A = Q^H D_A Q, B = Q^H D_B Q with Q = G2 G1 unitary,
/// G1 the 2 x 2 reflectors on the index pairs (0, 1), (2, 3), ... and G2 those on (1, 2), (3, 4),
/// ... (so A and B are connected band matrices, a chain for the factorization), and block
/// diagonal (D_A, D_B):
///   * real eigenvalues: D_B = diag(s) with signs s = +-1 (B indefinite), D_A = diag(lambda s);
///   * a complex pair x +- iy per 2 x 2 block at the end: D_A = [[x, y], [y, -x]],
///     D_B = diag(1, -1) (det(D_A - l D_B) = -((l - x)^2 + y^2)).
/// A x = lambda B x has exactly these eigenvalues (y = Q x). lam.size() must be even. Zero
/// eigenvalues form null(A) (the zero cluster).
template <class T>
Pencil<T> block_pencil(const std::vector<double>& lam, const std::vector<cplx>& pairs) {
  const auto nreal = static_cast<index_t>(lam.size());
  const index_t n = nreal + 2 * static_cast<index_t>(pairs.size());
  // Rows of Q = G2 G1: row k of G2 (pair (k, k +- 1) or identity) times G1, at most 4 entries.
  std::vector<std::vector<std::pair<index_t, T>>> q(static_cast<std::size_t>(n));
  auto g1_row = [&](index_t k) {  // row k of G1: entries (col, value)
    const auto r = reflector<T>(k / 2 + 7919);
    const index_t base = k - k % 2;
    return std::vector<std::pair<index_t, T>>{{base, r[k % 2][0]}, {base + 1, r[k % 2][1]}};
  };
  for (index_t k = 0; k < n; ++k) {
    std::vector<std::pair<index_t, T>> g2;  // row k of G2
    if (k == 0 || (k == n - 1 && k % 2 == 1)) {
      g2 = {{k, T(1)}};
    } else {
      const index_t base = k - (k + 1) % 2;  // pairs (1, 2), (3, 4), ...
      const auto r = reflector<T>(base);
      g2 = {{base, r[k - base][0]}, {base + 1, r[k - base][1]}};
    }
    for (const auto& [l, gv] : g2)
      for (const auto& [c, v] : g1_row(l)) q[k].push_back({c, gv * v});
  }
  // D: entries (k, l, a, b) of the block diagonal (D_A, D_B).
  struct DEntry {
    index_t k, l;
    T a, b;
  };
  std::vector<DEntry> d;
  for (index_t i = 0; i < nreal; ++i) {
    const double s = (i % 3 == 0) ? -1.0 : 1.0;
    d.push_back({i, i, T(lam[i] * s), T(s)});
  }
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const index_t i = nreal + 2 * static_cast<index_t>(p);
    d.push_back({i, i, T(pairs[p].real()), T(1.0)});
    d.push_back({i + 1, i + 1, T(-pairs[p].real()), T(-1.0)});
    d.push_back({i, i + 1, T(pairs[p].imag()), T(0.0)});
    d.push_back({i + 1, i, T(pairs[p].imag()), T(0.0)});
  }
  // (Q^H D Q)(i, j) = sum conj(Q(k, i)) D(k, l) Q(l, j), accumulated as duplicates.
  std::vector<index_t> rows, cols;
  std::vector<T> av, bv;
  for (const DEntry& e : d)
    for (const auto& [i, qki] : q[e.k])
      for (const auto& [j, qlj] : q[e.l]) {
        rows.push_back(i);
        cols.push_back(j);
        av.push_back(hs::conj_if(qki) * e.a * qlj);
        bv.push_back(hs::conj_if(qki) * e.b * qlj);
      }
  Pencil<T> p;
  p.a = hs::csr_from_coo(n, n, rows, cols, av);
  p.b = hs::csr_from_coo(n, n, rows, cols, bv);
  return p;
}

/// 2048 zeros (the cluster), 64 negative eigenvalues in [-1, -0.1] (propagating-like), and
/// positive ones from 0.05 to 40 (spacing ~2e-3, so that the one-sided runs converge in a few
/// restarts): n = 20480.
std::vector<double> waveguide_like() {
  std::vector<double> lam(2048, 0.0);
  for (int i = 0; i < 64; ++i) lam.push_back(-0.1 - 0.9 * i / 63.0);
  const auto rest = static_cast<index_t>(20480 - lam.size());
  for (index_t i = 0; i < rest; ++i)
    lam.push_back(0.05 + 40.0 * static_cast<double>(i) / static_cast<double>(rest));
  return lam;
}

std::vector<double> nearest_nonzero(std::vector<double> w, double sigma, index_t k) {
  w.erase(std::remove(w.begin(), w.end(), 0.0), w.end());
  std::stable_sort(w.begin(), w.end(),
                   [&](double a, double b) { return std::abs(a - sigma) < std::abs(b - sigma); });
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

hs::SolverConfig nearest_cfg(double sigma, index_t k) {
  hs::SolverConfig cfg;
  cfg.target.sigma = sigma;
  cfg.target.k = k;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 2.0;
  return cfg;
}

/// Block sizes of the Krylov-Schur runs in a JSON-lines event log.
std::vector<index_t> run_block_sizes(const std::string& path) {
  std::vector<index_t> sizes;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto j = nlohmann::json::parse(line);
    if (j.value("event", "") == "krylov_schur") sizes.push_back(j.at("block_size").get<index_t>());
  }
  return sizes;
}

/// The returned real pairs: unit vectors whose backward errors (recomputed) match the report.
template <class T>
void expect_vectors_match_report(const Pencil<T>& p, const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  const index_t n = p.a.nrows, m = static_cast<index_t>(sol.eigenvalues.size());
  ASSERT_EQ(sol.eigenvectors.rows(), n);
  ASSERT_EQ(sol.eigenvectors.cols(), m);
  std::vector<T> ax(static_cast<std::size_t>(n)), bx(ax.size());
  for (index_t j = 0; j < m; ++j) {
    const T* x = sol.eigenvectors.col(j);
    EXPECT_NEAR(hs::linalg::norm2(n, x), 1.0, 1e-12) << j;
    hs::spmv(p.a, x, ax.data());
    hs::spmv(p.b, x, bx.data());
    for (index_t i = 0; i < n; ++i) ax[i] -= sol.eigenvalues[j] * bx[i];
    const double eta = hs::linalg::norm2(n, ax.data()) /
                       (r.norm2_a + std::abs(sol.eigenvalues[j]) * r.norm2_b);
    EXPECT_NEAR(eta, r.pairs[j].backward_error, 1e-3 * r.pairs[j].backward_error + 1e-16) << j;
    EXPECT_LE(eta, 1e-10) << j;
  }
}

template <class T>
class DriverGhiepBlock : public ::testing::Test {};
using ScalarTypes = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(DriverGhiepBlock, ScalarTypes);

TYPED_TEST(DriverGhiepBlock, PlainRunUsesBlocksAndMatchesTheSpectrum) {
  // sigma between two negative eigenvalues, k = 12: the nearest pairs stay clear of the cluster, so
  // one plain run (nev = 12: p = 4) does it.
  using T = TypeParam;
  const auto lam = waveguide_like();
  const auto p = block_pencil<T>(lam, {});
  const std::string path =
      ::testing::TempDir() + "/ghiep_block_plain_" + (hs::is_complex_v<T> ? "z" : "d") + ".jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(p.a, &p.b, nearest_cfg(-0.507, 12), &log);
  }
  const auto& r = sol.report;
  EXPECT_EQ(r.problem, hs::ProblemType::GeneralizedIndefinite);
  EXPECT_TRUE(r.success);
  EXPECT_EQ(run_block_sizes(path), std::vector<index_t>{4});
  EXPECT_EQ(r.op_applies % 4, 0);
  EXPECT_TRUE(sol.complex_eigenvalues.empty());
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, -0.507, 12), 1e-9);
  expect_vectors_match_report(p, sol);
}

TEST(DriverGhiepBlock, ZeroClusterSidesUseBlocksAndNeverReturnTheCluster) {
  // sigma = 0 sits in the 2048-dimensional cluster: one-sided runs at sigma+ and sigma- (block
  // runs: nev = k), the cluster at their unwanted ends, and none of its copies in the result.
  // (Real T: the complex Hermitian path is the same template, covered in test_driver_cluster.)
  using T = double;
  const auto lam = waveguide_like();
  const auto p = block_pencil<T>(lam, {});
  const std::string path = ::testing::TempDir() + "/ghiep_block_cluster.jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(p.a, &p.b, nearest_cfg(0.0, 10), &log);
  }
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  EXPECT_EQ(r.zero_cluster_dim, 2048);
  const auto sizes = run_block_sizes(path);
  EXPECT_NE(std::find(sizes.begin(), sizes.end(), index_t{4}), sizes.end());
  for (const double x : sol.eigenvalues) EXPECT_GT(std::abs(x), 1e-6) << "cluster copy returned";
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam, 0.0, 10), 1e-9);
  expect_vectors_match_report(p, sol);
}

TEST(DriverGhiepBlock, ComplexPairNearTheShiftIsReturnedSeparately) {
  // A genuine complex pair -0.5 +- 0.004i among the nearest: the refinement of the block run
  // classifies it complex (returned separately, unit vectors), the real pairs stay real.
  const auto lam = waveguide_like();
  std::vector<double> lam_even(lam.begin(), lam.end() - 2);
  const auto p = block_pencil<double>(lam_even, {cplx(-0.5, 0.004)});
  auto sol = hs::solve(p.a, &p.b, nearest_cfg(-0.505, 10));
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  ASSERT_EQ(sol.complex_eigenvalues.size(), 2u);
  for (const cplx& z : sol.complex_eigenvalues) {
    EXPECT_NEAR(z.real(), -0.5, 1e-9);
    EXPECT_NEAR(std::abs(z.imag()), 0.004, 1e-9);
  }
  for (index_t j = 0; j < 2; ++j)
    EXPECT_NEAR(hs::linalg::norm2(p.a.nrows, sol.complex_eigenvectors.col(j)), 1.0, 1e-12);
  expect_same_set(sol.eigenvalues, nearest_nonzero(lam_even, -0.505, 8), 1e-9);
  for (const auto& pr : r.pairs) EXPECT_LE(pr.backward_error, 1e-10);
}

}  // namespace
