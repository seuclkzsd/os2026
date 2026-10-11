// GHIEP path of hs::solve, review fixes of module ghiep (P4): outputs that the earlier tests
// (test_ghiep_driver.cpp) do not check.
//   * Complex eigenvectors of the block runs (real and complex Hermitian pencils): every returned
//     complex pair satisfies A x = lambda B x to the tolerance, with the residual the report gives.
//   * The fallback of refine_union when the projected pencil (W^H A W, W^H B W) is singular: the
//     Krylov-Schur pairs of each run, placed by coefficients in the union basis (one run, and two
//     one-sided runs, i.e. a second run's block at an offset). It is reached by pencils with
//     infinite eigenvalues (B singular) when nev exceeds the finite eigenvalues on offer: the Schur
//     bases then hold null(B) directions, and W^H B W is exactly singular.
//   * A complex Hermitian pencil whose k nearest eigenvalues end inside a conjugate pair (complex
//     arithmetic keeps one member): the Schur basis holds x but not its partner, x^H B x = 0.
// Pencils with known spectra: A = Q^H D_A Q, B = Q^H D_B Q with Q a product of 2 x 2 reflectors
// (banded, as in test_ghiep_driver.cpp) and (D_A, D_B) block diagonal.
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

/// Spectrum of a test pencil: real eigenvalues (0 = null(A), the zero cluster), complex pairs
/// x +- iy, and eigenvalues at infinity (null(B)). The order must be even.
struct Spectrum {
  std::vector<double> real;
  std::vector<cplx> pairs;
  index_t infinite = 0;
  index_t order() const { return static_cast<index_t>(real.size() + 2 * pairs.size()) + infinite; }
};

/// 2 x 2 Householder reflector I - 2 v v^H / (v^H v) from a deterministic v seeded by `index`.
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
    for (int c = 0; c < 2; ++c) q[r][c] = T(r == c ? 1.0 : 0.0) - T(2.0 / vv) * v[r] * hs::conj_if(v[c]);
  return q;
}

/// A = Q^H D_A Q, B = Q^H D_B Q, Q = G2 G1 with G1 the reflectors on (0, 1), (2, 3), ... and G2 those
/// on (1, 2), (3, 4), ...; (D_A, D_B): (lambda s, s) with s = +-1 per real eigenvalue,
/// ([[x, r y], [r y, -2x]], diag(1, -2)), r = sqrt(2), per complex pair: det(D_A - l D_B) =
/// -2 ((l - x)^2 + y^2), eigenvalues x +- iy with eigenvectors (r, +-i), which are not orthogonal
/// (a projected eigenvector is then a genuinely complex combination of the Schur vectors, unlike for
/// diag(1, -1)); (a, 0) with a != 0 per infinite eigenvalue.
template <class T>
Pencil<T> make_pencil(const Spectrum& sp) {
  const index_t n = sp.order();
  EXPECT_EQ(n % 2, 0);
  std::vector<std::vector<std::pair<index_t, T>>> q(static_cast<std::size_t>(n));
  auto g1_row = [&](index_t k) {
    const auto r = reflector<T>(k / 2 + 7919);
    const index_t base = k - k % 2;
    return std::vector<std::pair<index_t, T>>{{base, r[k % 2][0]}, {base + 1, r[k % 2][1]}};
  };
  for (index_t k = 0; k < n; ++k) {
    std::vector<std::pair<index_t, T>> g2;
    if (k == 0 || (k == n - 1 && k % 2 == 1)) {
      g2 = {{k, T(1)}};
    } else {
      const index_t base = k - (k + 1) % 2;
      const auto r = reflector<T>(base);
      g2 = {{base, r[k - base][0]}, {base + 1, r[k - base][1]}};
    }
    for (const auto& [l, gv] : g2)
      for (const auto& [c, v] : g1_row(l)) q[k].push_back({c, gv * v});
  }
  struct DEntry {
    index_t k, l;
    T a, b;
  };
  std::vector<DEntry> d;
  index_t i = 0;
  for (std::size_t r = 0; r < sp.real.size(); ++r, ++i) {
    const double s = (r % 3 == 0) ? -1.0 : 1.0;
    d.push_back({i, i, T(sp.real[r] * s), T(s)});
  }
  for (const cplx& z : sp.pairs) {
    const double ry = std::sqrt(2.0) * z.imag();
    d.push_back({i, i, T(z.real()), T(1.0)});
    d.push_back({i + 1, i + 1, T(-2.0 * z.real()), T(-2.0)});
    d.push_back({i, i + 1, T(ry), T(0.0)});
    d.push_back({i + 1, i, T(ry), T(0.0)});
    i += 2;
  }
  for (index_t r = 0; r < sp.infinite; ++r, ++i) d.push_back({i, i, T(1.0 + 0.1 * static_cast<double>(r % 5)), T(0.0)});
  std::vector<index_t> rows, cols;
  std::vector<T> av, bv;
  for (const DEntry& e : d)
    for (const auto& [ii, qki] : q[e.k])
      for (const auto& [jj, qlj] : q[e.l]) {
        rows.push_back(ii);
        cols.push_back(jj);
        av.push_back(hs::conj_if(qki) * e.a * qlj);
        bv.push_back(hs::conj_if(qki) * e.b * qlj);
      }
  Pencil<T> p;
  p.a = hs::csr_from_coo(n, n, rows, cols, av);
  p.b = hs::csr_from_coo(n, n, rows, cols, bv);
  return p;
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

/// (A x, B x) for a complex vector x (real A, B: applied to the real and imaginary parts).
template <class T>
void apply_pencil(const Pencil<T>& p, const cplx* x, std::vector<cplx>& ax, std::vector<cplx>& bx) {
  const index_t n = p.a.nrows;
  ax.assign(static_cast<std::size_t>(n), cplx(0.0));
  bx.assign(ax.size(), cplx(0.0));
  if constexpr (hs::is_complex_v<T>) {
    hs::spmv(p.a, x, ax.data());
    hs::spmv(p.b, x, bx.data());
  } else {
    std::vector<double> part(static_cast<std::size_t>(n)), y(part.size());
    for (int im = 0; im < 2; ++im) {
      for (index_t i = 0; i < n; ++i) part[i] = im ? x[i].imag() : x[i].real();
      hs::spmv(p.a, part.data(), y.data());
      for (index_t i = 0; i < n; ++i) ax[i] += im ? cplx(0.0, y[i]) : cplx(y[i], 0.0);
      hs::spmv(p.b, part.data(), y.data());
      for (index_t i = 0; i < n; ++i) bx[i] += im ? cplx(0.0, y[i]) : cplx(y[i], 0.0);
    }
  }
}

/// Backward error ||A x - lambda B x|| / ((||A|| + |lambda| ||B||) ||x||) with the report's norms.
template <class T>
double backward_error(const Pencil<T>& p, const hs::SolveReport& r, const cplx* x, cplx lambda) {
  const index_t n = p.a.nrows;
  std::vector<cplx> ax, bx;
  apply_pencil(p, x, ax, bx);
  for (index_t i = 0; i < n; ++i) ax[i] -= lambda * bx[i];
  return hs::linalg::norm2(n, ax.data()) / ((r.norm2_a + std::abs(lambda) * r.norm2_b) * hs::linalg::norm2(n, x));
}

/// Every returned pair, real and complex: a unit vector whose backward error, recomputed from the
/// returned vector, matches the report's (pairs: the real ones first, then the complex ones) and,
/// for the accepted ones, meets tol.
template <class T>
void expect_outputs_match_report(const Pencil<T>& p, const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  const index_t n = p.a.nrows;
  const auto nr = static_cast<index_t>(sol.eigenvalues.size());
  const auto nc = static_cast<index_t>(sol.complex_eigenvalues.size());
  ASSERT_EQ(static_cast<index_t>(r.pairs.size()), nr + nc);
  ASSERT_EQ(sol.eigenvectors.rows(), n);
  ASSERT_EQ(sol.eigenvectors.cols(), nr);
  ASSERT_EQ(sol.complex_eigenvectors.rows(), n);
  ASSERT_EQ(sol.complex_eigenvectors.cols(), nc);
  std::vector<cplx> x(static_cast<std::size_t>(n));
  for (index_t j = 0; j < nr + nc; ++j) {
    cplx lambda;
    if (j < nr) {
      for (index_t i = 0; i < n; ++i) x[i] = cplx(sol.eigenvectors(i, j));
      lambda = cplx(sol.eigenvalues[j]);
    } else {
      std::copy(sol.complex_eigenvectors.col(j - nr), sol.complex_eigenvectors.col(j - nr) + n, x.begin());
      lambda = sol.complex_eigenvalues[j - nr];
      EXPECT_NEAR(r.pairs[j].lambda_im, lambda.imag(), 0.0) << j;
    }
    EXPECT_NEAR(r.pairs[j].lambda_re, lambda.real(), 0.0) << j;
    EXPECT_NEAR(hs::linalg::norm2(n, x.data()), 1.0, 1e-12) << j;
    const double eta = backward_error(p, r, x.data(), lambda);
    EXPECT_NEAR(eta, r.pairs[j].backward_error, 1e-3 * r.pairs[j].backward_error + 1e-16) << j << " " << lambda;
    if (r.pairs[j].accepted) {
      EXPECT_LE(eta, 1e-10) << j << " " << lambda;
    }
  }
}

/// Some element of `got` within tol of z (complex values; conjugates are distinct).
bool contains(const std::vector<cplx>& got, cplx z, double tol) {
  return std::any_of(got.begin(), got.end(), [&](const cplx& g) { return std::abs(g - z) <= tol; });
}

bool has_warning(const hs::SolveReport& r, const std::string& text) {
  return std::any_of(r.warnings.begin(), r.warnings.end(),
                     [&](const std::string& w) { return w.find(text) != std::string::npos; });
}

const char* const kSingularFallback = "projected pencil singular";

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

template <class T>
class DriverGhiepFix : public ::testing::Test {};
using ScalarTypes = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(DriverGhiepFix, ScalarTypes);

TYPED_TEST(DriverGhiepFix, ComplexPairsOfBlockRunsSatisfyThePencil) {
  // Two complex pairs among the 10 eigenvalues nearest sigma = -0.505 of an n = 20480 pencil (the
  // driver's block runs, p = 2): -0.51 +- 0.002i, -0.5 +- 0.004i, and the reals -0.5, -0.5143,
  // -0.4857, -0.5286, -0.4714, -0.5429 (64 negatives in [-1, -0.1], 2048 zeros, positives).
  using T = TypeParam;
  Spectrum sp;
  sp.real.assign(2048, 0.0);
  for (int i = 0; i < 64; ++i) sp.real.push_back(-0.1 - 0.9 * i / 63.0);
  const auto pos = static_cast<index_t>(20480 - 4 - sp.real.size());
  for (index_t i = 0; i < pos; ++i) sp.real.push_back(0.05 + 40.0 * static_cast<double>(i) / static_cast<double>(pos));
  sp.pairs = {cplx(-0.51, 0.002), cplx(-0.5, 0.004)};
  const auto p = make_pencil<T>(sp);
  const std::string path =
      ::testing::TempDir() + "/ghiep_fix_complex_" + (hs::is_complex_v<T> ? "z" : "d") + ".jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(p.a, &p.b, nearest_cfg(-0.505, 10), &log);
  }
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  EXPECT_EQ(run_block_sizes(path), std::vector<index_t>{2});
  ASSERT_EQ(sol.complex_eigenvalues.size(), 4u);
  for (const cplx z : {cplx(-0.51, 0.002), cplx(-0.51, -0.002), cplx(-0.5, 0.004), cplx(-0.5, -0.004)})
    EXPECT_TRUE(contains(sol.complex_eigenvalues, z, 1e-9)) << z;
  std::vector<double> reals = sol.eigenvalues;
  std::sort(reals.begin(), reals.end());
  const std::vector<double> want = {-0.1 - 0.9 * 31 / 63.0, -0.1 - 0.9 * 30 / 63.0, -0.1 - 0.9 * 29 / 63.0,
                                    -0.5,                   -0.1 - 0.9 * 27 / 63.0, -0.1 - 0.9 * 26 / 63.0};
  ASSERT_EQ(reals.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(reals[i], want[i], 1e-9) << i;
  expect_outputs_match_report(p, sol);
}

TYPED_TEST(DriverGhiepFix, SingularProjectedPencilFallsBackToTheRunsPairs) {
  // Six finite eigenvalues and 34 infinite ones (B of rank 6), k = 8 > 6: the plain run's Schur
  // basis holds null(B) directions, W^H B W is singular, and the result comes from the Krylov-Schur
  // pairs. The finite ones are found with their vectors; the null(B) directions give theta ~ 0, i.e.
  // |lambda| ~ 1e16 or no candidate at all (rounding decides), never a finite value.
  using T = TypeParam;
  Spectrum sp;
  sp.real = {-0.3, -0.5, 0.4, 0.8, -1.2, 1.5};
  sp.infinite = 34;
  const auto p = make_pencil<T>(sp);
  const auto sol = hs::solve(p.a, &p.b, nearest_cfg(0.1, 8));
  const auto& r = sol.report;
  EXPECT_TRUE(has_warning(r, kSingularFallback));
  std::vector<double> finite;
  for (const double x : sol.eigenvalues)
    if (std::abs(x) < 1e3) finite.push_back(x);
  std::sort(finite.begin(), finite.end());
  std::vector<double> want = sp.real;
  std::sort(want.begin(), want.end());
  ASSERT_EQ(finite.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(finite[i], want[i], 1e-10) << i;
  EXPECT_TRUE(sol.complex_eigenvalues.empty());
  expect_outputs_match_report(p, sol);
}

TYPED_TEST(DriverGhiepFix, SingularProjectedPencilAcrossOneSidedRuns) {
  // sigma = 0 inside a 4-dimensional zero cluster: one-sided runs on both sides, each asked for
  // k = 4 pairs with only two finite eigenvalues on its side (the rest of its wanted set are
  // null(B) directions): the union's W^H B W is singular, and the fallback places each run's pairs
  // at that run's offset in the union basis.
  using T = TypeParam;
  Spectrum sp;
  sp.real = {0.0, 0.0, 0.0, 0.0, 0.5, 0.9, -0.6, -1.1};
  sp.infinite = 32;
  const auto p = make_pencil<T>(sp);
  const auto sol = hs::solve(p.a, &p.b, nearest_cfg(0.0, 4));
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  EXPECT_TRUE(has_warning(r, kSingularFallback));
  EXPECT_EQ(r.zero_cluster_dim, 4);
  index_t above = 0, below = 0;
  for (const auto& s : r.slices) {
    if (s.which == "largest_algebraic" && s.count_found > 0) ++above;
    if (s.which == "smallest_algebraic" && s.count_found > 0) ++below;
  }
  EXPECT_GE(above, 1);
  EXPECT_GE(below, 1);
  std::vector<double> got = sol.eigenvalues;
  std::sort(got.begin(), got.end());
  const std::vector<double> want = {-1.1, -0.6, 0.5, 0.9};
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(got[i], want[i], 1e-10) << i;
  expect_outputs_match_report(p, sol);
}

TYPED_TEST(DriverGhiepFix, NearestSetEndingInsideAConjugatePair) {
  // Nearest sigma = -0.505: -0.5 (distance 0.005), then the pair -0.51 +- 0.004i (0.0064 each).
  // k = 2 ends inside the pair: real T keeps the pair whole in its Schur basis, complex T holds one
  // member (its partner's direction is missing, so x^H B x = 0 makes the projected B singular).
  // Either way: -0.5 and one member of the pair, both satisfying the pencil.
  using T = TypeParam;
  Spectrum sp;
  for (int i = 0; i < 60; ++i) sp.real.push_back(-0.1 - 0.9 * i / 59.0);
  sp.real.push_back(-0.5);
  for (int i = 0; i < 137; ++i) sp.real.push_back(0.05 + 3.0 * i / 137.0);
  sp.pairs = {cplx(-0.51, 0.004)};
  const auto p = make_pencil<T>(sp);
  const auto sol = hs::solve(p.a, &p.b, nearest_cfg(-0.505, 2));
  const auto& r = sol.report;
  EXPECT_TRUE(r.success);
  ASSERT_EQ(sol.eigenvalues.size(), 1u);
  EXPECT_NEAR(sol.eigenvalues[0], -0.5, 1e-10);
  ASSERT_EQ(sol.complex_eigenvalues.size(), 1u);
  EXPECT_NEAR(sol.complex_eigenvalues[0].real(), -0.51, 1e-10);
  EXPECT_NEAR(std::abs(sol.complex_eigenvalues[0].imag()), 0.004, 1e-10);
  expect_outputs_match_report(p, sol);
}

}  // namespace
