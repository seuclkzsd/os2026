// The rigorous positive-definiteness proof of B (module anaint, P5; src/driver/norms.hpp
// prove_positive_definite) and its use by hs::solve: the bound never exceeds lambda_min of the
// scaled matrix (dense reference), indefinite matrices are never proven definite - down to a
// negative eigenvalue 1e-12 below zero -, finite-element mass matrices are proven, the budget and
// memory limits stop the attempts, and a GHEP whose B is proven runs without factorizing B.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// Consistent P1 mass matrix of the unit-spacing triangulation of an nx x ny node grid (each
/// square split along its diagonal): element matrices (1/24) [[2,1,1],[1,2,1],[1,1,2]]. HPD, not
/// diagonally dominant (row sums of |off-diagonal| equal the diagonal inside the grid).
template <class T>
hs::CsrMatrix<T> p1_mass(index_t nx, index_t ny) {
  std::vector<index_t> r, c;
  std::vector<T> v;
  auto id = [&](index_t i, index_t j) { return i + nx * j; };
  auto element = [&](index_t a, index_t b, index_t d) {
    const index_t nodes[3] = {a, b, d};
    for (const index_t p : nodes)
      for (const index_t q : nodes) {
        r.push_back(p);
        c.push_back(q);
        v.push_back(T((p == q ? 2.0 : 1.0) / 24.0));
      }
  };
  for (index_t j = 0; j + 1 < ny; ++j)
    for (index_t i = 0; i + 1 < nx; ++i) {
      element(id(i, j), id(i + 1, j), id(i + 1, j + 1));
      element(id(i, j), id(i + 1, j + 1), id(i, j + 1));
    }
  const index_t n = nx * ny;
  return hs::csr_from_coo<T>(n, n, r, c, v);
}

/// lambda_min of D^{-1/2} B D^{-1/2}, D = diag(B), from the dense pencil (B, D) (n <= 4000).
template <class T>
double scaled_lambda_min(const hs::CsrMatrix<T>& b) {
  std::vector<index_t> idx;
  std::vector<T> d;
  for (index_t i = 0; i < b.nrows; ++i)
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] == i) {
        idx.push_back(i);
        d.push_back(T(std::real(b.values[p])));
      }
  const auto dm = hs::csr_from_coo<T>(b.nrows, b.nrows, idx, idx, d);
  // eig(B, D) = eig(D^{-1/2} B D^{-1/2}); dense_eigenvalues needs the second matrix HPD (D is).
  return hs::test::dense_eigenvalues(b, &dm).front();
}

/// B - mu diag(B): lambda_min of the scaled matrix moves down by exactly mu.
template <class T>
hs::CsrMatrix<T> minus_scaled_identity(hs::CsrMatrix<T> b, double mu) {
  for (index_t i = 0; i < b.nrows; ++i)
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] == i) b.values[p] *= (1.0 - mu);
  return b;
}

std::vector<nlohmann::json> events(const std::string& path, const std::string& name) {
  std::vector<nlohmann::json> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto j = nlohmann::json::parse(line);
    if (j.value("event", "") == name) out.push_back(j);
  }
  return out;
}

// ------------------------------------------------------------------------------ the bound

/// Block-diagonal matrices with prescribed spectra (dense random unitary blocks of 12): level 0 is
/// the blocks' full lower triangle, so the incomplete factor is a complete one and the bound is
/// tight up to the shift. Over spectra whose smallest eigenvalue ranges from 1e-1 down to -1e-12
/// (after the diagonal scaling: checked densely) and shifts from 0 to 0.9 lambda_min, the bound
/// never exceeds lambda_min of the scaled matrix, and is positive only for positive definite ones.
/// (The rounding allowance itself is tested by exactly singular matrices below: these spectra
/// keep a margin of ~1e-12 that rounding alone does not cross.)
template <class T>
void bound_never_exceeds_lambda_min() {
  int proven = 0, cases = 0;
  for (std::uint64_t seed = 1; seed <= 6; ++seed)
    for (const double low : {1e-1, 1e-4, 1e-8, 1e-12, -1e-12, -1e-8, -1e-3}) {
      std::vector<double> eig;
      for (index_t i = 0; i < 240; ++i) eig.push_back(1.0 + 0.5 * std::sin(0.37 * static_cast<double>(i + seed)));
      eig[static_cast<std::size_t>(seed * 7)] = low;
      const auto b = hs::test::prescribed_spectrum<T>(eig, 12, seed);
      const double lmin = scaled_lambda_min(b);
      for (const double frac : {0.0, 0.5, 0.9}) {
        const double shift = lmin > 0 ? frac * lmin : 0.0;
        const auto ib = hs::driver::incomplete_cholesky_bound(b, 0, shift);
        ++cases;
        if (ib.breakdown) {
          EXPECT_LT(lmin, 1e-6) << "seed " << seed << " low " << low;  // only near-singular ones break down
          continue;
        }
        // Dense eigenvalues carry ~1e-15 of rounding themselves.
        EXPECT_LE(ib.lower_bound, lmin + 1e-14) << "seed " << seed << " low " << low << " frac " << frac;
        if (lmin < 0) {
          EXPECT_LT(ib.lower_bound, 0.0) << "indefinite proven: seed " << seed << " low " << low;
        }
        if (ib.lower_bound > 0) ++proven;
      }
    }
  EXPECT_GE(proven, 30);  // lambda_min >= 1e-8 with shifts 0.5 / 0.9 lambda_min: 36 cases
}

TEST(DriverHpdProof, BoundNeverExceedsLambdaMinReal) { bound_never_exceeds_lambda_min<double>(); }
TEST(DriverHpdProof, BoundNeverExceedsLambdaMinComplex) { bound_never_exceeds_lambda_min<cplx>(); }

/// v v^H for an integer (Gaussian integer) vector v: exactly singular, stored exactly, positive
/// diagonal. Level 0 is the full lower triangle, so the computed M = S - L L^H is pure rounding
/// around lambda_min = 0, and any positive lower bound is a false proof. Without the rounding
/// allowance e_ij of the check (e = 0) half of the attempts below are falsely proven (P5c:
/// 379 of 758 real, 2570 of 5170 complex; the other tests of this file all passed that mutant);
/// with it, none may. The allowance is an a-priori bound with much slack on such small problems:
/// 8x smaller still passed here, 200x smaller (complex) let 2 of 7488 through.
template <class T>
hs::CsrMatrix<T> rank_one(const std::vector<T>& v) {
  const auto n = static_cast<index_t>(v.size());
  std::vector<index_t> r, c;
  std::vector<T> val;
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j) {
      r.push_back(i);
      c.push_back(j);
      val.push_back(v[i] * hs::conj_if(v[j]));
    }
  return hs::csr_from_coo<T>(n, n, r, c, val);
}

template <class T>
T gaussian(int re, int im) {
  if constexpr (hs::is_complex_v<T>)
    return T(re, im);
  else
    return T(re + 0 * im);
}

/// G^H G for an integer (Gaussian integer) (m - 1) x m matrix G (column-major): exactly singular
/// (rank <= m - 1), stored exactly; the factorization runs through m - 1 regular pivots, so the
/// last row's M entries carry the rounding of m-term dot products.
template <class T>
hs::CsrMatrix<T> gram(index_t m, const std::vector<T>& g) {
  std::vector<index_t> r, c;
  std::vector<T> val;
  for (index_t i = 0; i < m; ++i)
    for (index_t j = 0; j < m; ++j) {
      T s(0);
      for (index_t k = 0; k + 1 < m; ++k) s += hs::conj_if(g[k + (m - 1) * i]) * g[k + (m - 1) * j];
      r.push_back(i);
      c.push_back(j);
      val.push_back(s);
    }
  return hs::csr_from_coo<T>(m, m, r, c, val, /*drop_zeros=*/false);
}

template <class T>
void exactly_singular_never_proven() {
  int attempts = 0, false_proofs = 0;
  auto probe_matrix = [&](const hs::CsrMatrix<T>& b) {
    for (index_t i = 0; i < b.nrows; ++i)
      for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
        if (b.colind[p] == i && !(std::real(b.values[p]) > 0.0)) return;  // a zero column of G
    for (const double shift : {0.0, 1e-17}) {
      const auto ib = hs::driver::incomplete_cholesky_bound(b, 0, shift);
      if (ib.breakdown) continue;
      ++attempts;
      if (ib.lower_bound > 0.0) ++false_proofs;
    }
  };
  std::uint64_t state = 7;
  auto small = [&](int range) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<int>((state >> 33) % static_cast<std::uint64_t>(2 * range + 1)) - range;
  };
  const int im0 = hs::is_complex_v<T> ? 1 : 0;
  for (index_t m = 3; m <= 12; ++m)
    for (int seed = 0; seed < 300; ++seed) {
      std::vector<T> g;
      for (index_t k = 0; k < (m - 1) * m; ++k) {
        const int re = small(4), im = small(4) * im0;
        g.push_back(gaussian<T>(re, im));
      }
      probe_matrix(gram<T>(m, g));
    }
  auto probe = [&](const std::vector<T>& v) {
    const auto b = rank_one(v);
    for (const double shift : {0.0, 1e-17}) {
      const auto ib = hs::driver::incomplete_cholesky_bound(b, 0, shift);
      if (ib.breakdown) continue;
      ++attempts;
      if (ib.lower_bound > 0.0) ++false_proofs;
    }
  };
  const int im = hs::is_complex_v<T> ? 1 : 0;
  for (int p = 1; p <= 120; ++p)
    for (int q = 1; q <= 120; ++q) probe({gaussian<T>(p, im * (q % 7)), gaussian<T>(q, -im * (p % 5))});
  for (int p = 1; p <= 14; ++p)
    for (int q = 1; q <= 14; ++q)
      for (int s = 1; s <= 14; ++s)
        probe({gaussian<T>(p, im * s), gaussian<T>(q, im * (p % 3)), gaussian<T>(s, -im * q)});
  EXPECT_GT(attempts, 500);  // 758 (real) do not break down: the test exercises the check
  EXPECT_EQ(false_proofs, 0) << "of " << attempts;
}

TEST(DriverHpdProof, ExactlySingularIsNeverProvenReal) { exactly_singular_never_proven<double>(); }
TEST(DriverHpdProof, ExactlySingularIsNeverProvenComplex) { exactly_singular_never_proven<cplx>(); }

// ------------------------------------------------------------------------------ the proof

/// P1 mass matrices are proven (level 0 or 1) with a bound below lambda_min of the scaled matrix;
/// shifted down beyond lambda_min (one or more negative eigenvalues) they never are.
template <class T>
void mass_matrices() {
  const auto m = p1_mass<T>(24, 17);
  const double lmin = scaled_lambda_min(m);
  ASSERT_GT(lmin, 0.05);
  ASSERT_LT(lmin, 0.6);
  const auto proof = hs::driver::prove_positive_definite(m);
  EXPECT_TRUE(proof.proven) << proof.outcome;
  EXPECT_EQ(proof.outcome, "proven");
  EXPECT_GT(proof.lower_bound, 0.0);
  EXPECT_LE(proof.lower_bound, lmin);
  EXPECT_GE(proof.theta_min, lmin - 1e-12);  // a Ritz value lies inside the spectrum
  EXPECT_EQ(proof.attempts, 1);
  EXPECT_EQ(proof.level, 1);  // fill level 1 first
  EXPECT_GE(proof.seconds + 1e-9, proof.setup_s + proof.lanczos_s + proof.attempts_s);
  EXPECT_GE(proof.pattern_s, 0.0);
  // Level 0 alone proves it as well.
  hs::driver::HpdProofOptions only0;
  only0.levels = {0};
  const auto p0 = hs::driver::prove_positive_definite(m, only0);
  EXPECT_TRUE(p0.proven) << p0.outcome;
  EXPECT_EQ(p0.level, 0);
  EXPECT_EQ(p0.pattern_s, 0.0);  // the level-1 pattern was not built
  // lambda_min of the scaled B - mu D is (lambda_min - mu) / (1 - mu).
  for (const double excess : {1e-12, 1e-9, 1e-4}) {
    const auto shifted = minus_scaled_identity(m, lmin + excess);
    const auto p = hs::driver::prove_positive_definite(shifted);
    EXPECT_FALSE(p.proven) << excess;
    EXPECT_NE(p.outcome, "proven");
  }
  // Clearly indefinite after scaling: the Lanczos run on S shows a negative Ritz value.
  const auto p = hs::driver::prove_positive_definite(minus_scaled_identity(m, lmin + 0.3));
  EXPECT_TRUE(p.indefinite) << p.outcome;
  EXPECT_FALSE(p.proven);
}

TEST(DriverHpdProof, MassMatricesReal) { mass_matrices<double>(); }
TEST(DriverHpdProof, MassMatricesComplex) { mass_matrices<cplx>(); }

TEST(DriverHpdProof, NonPositiveDiagonalAndNearlySingular) {
  auto m = p1_mass<double>(10, 9);
  auto z = m;
  for (index_t p = z.rowptr[4]; p < z.rowptr[5]; ++p)
    if (z.colind[p] == 4) z.values[p] = 0.0;
  EXPECT_EQ(hs::driver::prove_positive_definite(z).outcome, "nonpositive_diagonal");
  EXPECT_THROW(hs::driver::incomplete_cholesky_bound(z, 0, 0.0), hs::Error);
  // lambda_min of the scaled matrix ~2e-9: positive definite, but without the margin a proof needs.
  const auto near = minus_scaled_identity(m, scaled_lambda_min(m) - 1e-9);
  EXPECT_FALSE(hs::driver::prove_positive_definite(near).proven);
  // The Lanczos estimate below min_theta: no attempt at all.
  hs::driver::HpdProofOptions o;
  o.min_theta = 10.0;
  const auto p = hs::driver::prove_positive_definite(m, o);
  EXPECT_EQ(p.outcome, "nearly_singular");
  EXPECT_EQ(p.attempts, 0);
}

TEST(DriverHpdProof, BudgetAndMemoryLimitsStopTheAttempts) {
  const auto m = p1_mass<double>(40, 30);
  hs::driver::HpdProofOptions o;
  o.max_seconds = 0.0;
  auto p = hs::driver::prove_positive_definite(m, o);
  EXPECT_FALSE(p.proven);
  EXPECT_EQ(p.outcome, "too_expensive");
  EXPECT_EQ(p.attempts, 0);
  o = hs::driver::HpdProofOptions{};
  o.max_bytes = 1000.0;
  p = hs::driver::prove_positive_definite(m, o);
  EXPECT_EQ(p.outcome, "too_large");
  EXPECT_EQ(p.attempts, 0);
  EXPECT_TRUE(hs::driver::prove_positive_definite(m).proven);  // without limits
}

// The factorization is computed level by level in parallel: the bound does not depend on the
// thread count (bitwise).
// (80 x 80 grid: 159 levels of 40 rows on average, above the parallel threshold.)
TEST(DriverHpdProof, IndependentOfTheThreadCount) {
  const auto m = p1_mass<cplx>(80, 80);
  const int saved = hs::max_threads();
  for (const int level : {0, 1}) {
    double bound[2] = {0.0, 0.0};
    for (int t = 0; t < 2; ++t) {
#ifdef _OPENMP
      omp_set_num_threads(t == 0 ? 1 : 4);
#endif
      const auto ib = hs::driver::incomplete_cholesky_bound(m, level, 0.2);
      EXPECT_FALSE(ib.breakdown);
      bound[t] = ib.lower_bound;
    }
    EXPECT_EQ(bound[0], bound[1]) << level;
    EXPECT_GT(bound[0], 0.0) << level;
  }
#ifdef _OPENMP
  omp_set_num_threads(saved);
#endif
}

// ------------------------------------------------------------------------------ hs::solve

/// A = 2-D Laplacian, B = P1 mass matrix of the same grid: hs::solve proves B HPD without
/// factorizing it - the same factorizations and bitwise the same result as with the type given.
template <class T>
void ghep_without_factorizing_b() {
  const index_t nx = 120, ny = 100;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto b = p1_mass<T>(nx, ny);
  hs::SolverConfig cfg;
  cfg.target.mode = hs::TargetMode::Nearest;
  cfg.target.sigma = 5.0;
  cfg.target.k = 6;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.deterministic = true;
  // One file per scalar type: ctest runs the typed variants in parallel.
  const std::string path =
      ::testing::TempDir() + "/anaint_bproof_ghep_" + (hs::is_complex_v<T> ? "complex" : "real") + ".jsonl";
  hs::Solution<T> sol;
  {
    hs::Logger log(0, path);
    sol = hs::solve(a, &b, cfg, &log);
  }
  EXPECT_EQ(sol.report.problem, hs::ProblemType::GeneralizedDefinite);
  EXPECT_TRUE(sol.report.success);
  const auto bi = events(path, "b_inertia");
  ASSERT_EQ(bi.size(), 1u);
  EXPECT_EQ(bi[0].at("how").get<std::string>(), "incomplete_cholesky");
  EXPECT_EQ(bi[0].at("pos").get<index_t>(), nx * ny);
  const auto bp = events(path, "b_proof");
  ASSERT_EQ(bp.size(), 1u);
  EXPECT_TRUE(bp[0].at("proven").get<bool>());
  auto given = cfg;
  given.problem = hs::ProblemType::GeneralizedDefinite;
  const auto ref = hs::solve(a, &b, given);
  EXPECT_EQ(sol.report.factor_stats.factorizations, ref.report.factor_stats.factorizations);
  ASSERT_EQ(sol.eigenvalues.size(), ref.eigenvalues.size());
  for (std::size_t i = 0; i < sol.eigenvalues.size(); ++i) EXPECT_EQ(sol.eigenvalues[i], ref.eigenvalues[i]);
  // B - 0.8 diag(B) is indefinite (lambda_min of the scaled B is ~0.5) with a positive diagonal:
  // never taken for HPD, the solve runs the GHIEP path.
  const auto bad = minus_scaled_identity(b, 0.8);
  const auto ind = hs::solve(a, &bad, cfg);
  EXPECT_EQ(ind.report.problem, hs::ProblemType::GeneralizedIndefinite);
}

TEST(DriverHpdProof, GhepWithoutFactorizingBReal) { ghep_without_factorizing_b<double>(); }
TEST(DriverHpdProof, GhepWithoutFactorizingBComplex) { ghep_without_factorizing_b<cplx>(); }

}  // namespace
