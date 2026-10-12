// Block-size rule of shift-invert block Lanczos; see block_rule.hpp. Calibration below.
#include "eigen/block_rule.hpp"

#include <algorithm>
#include <cmath>

#include "eigen/block_lanczos.hpp"
#include "hsolve/error.hpp"

namespace hs::eigen_detail {

namespace {

// ---------------------------------------------------------------------------------------------
// Calibration (bench/blocktune_sweep.cpp on the HEP/GHEP data cases, 8 threads on cores 24-31 of
// a 2x Xeon Silver 4314 node, MUMPS L0-OpenMP, the driver's adaptive ordering; constants are
// least-squares fits on relative errors, see the commit message of this file for the tables).
//
// Operator application to a block of p columns, t(p) = a + b p for p >= 2 and t(1) = kSingleRhs
// (a + b) (MUMPS solves one right-hand side by a different, BLAS2 path):
//   a = kApplyPerEntry * E + kApplyPerRow * n_eff      (reading the factors: entries E)
//   b = kApplyPerRowRhs * n_eff                          (per right-hand side: O(n) work)
// with n_eff = n (real) or 2n (complex: MUMPS factorizes the real embedding of order 2n).
constexpr double kApplyPerEntry = 0.58e-9;
constexpr double kApplyPerRow = 61e-9;
constexpr double kApplyPerRowRhs = 22e-9;
constexpr double kSingleRhs = 1.1;
// Orthogonalization of one step (block kernel: BlockOrthonormalizer::try_fast; p = 1: CGS2 of
// lanczos_krylov_schur.cpp), against a basis of m vectors of n scalars:
//   o(p) = (kOrthoPerBasisEntry * n_s * m + kOrthoPerB * (nnz_b + n_s)) * p
// with n_s = n (real) or 2n (complex: twice the bytes per scalar), m the average basis size of a
// step (three quarters of the cycle's ncv).
constexpr double kOrthoPerBasisEntry = 0.5e-9;
constexpr double kOrthoPerB = 3e-9;
constexpr double kOrthoSingle = 0.5;  // p = 1 CGS2 per vector relative to the block formula

/// Average basis size of a step of a width-p run with the automatic ncv (block_dimensions).
double average_basis(index_t n, index_t nev, index_t p) {
  LanczosSetup s;
  s.n = s.n_free = n;
  s.nev = nev;
  s.ncv = std::min(n, std::max(2 * nev + 1, nev + 16));
  if (p == 1) return 0.75 * static_cast<double>(s.ncv);
  const BlockDimensions d = block_dimensions(s, p, true);
  return 0.75 * static_cast<double>(d.feasible ? d.ncv : s.ncv);
}

}  // namespace

WidthTable modelled_step_seconds(const ShiftInvertShape& shape) {
  if (shape.n < 1 || shape.nev < 1 || !(shape.factor_entries >= 0.0) || !(shape.nnz_b >= 0.0))
    HS_THROW(InvalidInput, "modelled_step_seconds: n = " << shape.n << ", nev = " << shape.nev
                                                         << ", factor entries "
                                                         << shape.factor_entries << ", nnz(B) "
                                                         << shape.nnz_b);
  const double n = static_cast<double>(shape.n);
  const double n_eff = shape.complex_scalars ? 2.0 * n : n;
  const double a = kApplyPerEntry * shape.factor_entries + kApplyPerRow * n_eff;
  const double b = kApplyPerRowRhs * n_eff;
  WidthTable t{};
  for (std::size_t i = 0; i < kRuleWidthCount; ++i) {
    const index_t p = kRuleWidths[i];
    const double pd = static_cast<double>(p);
    const double apply = p == 1 ? kSingleRhs * (a + b) : a + b * pd;
    const double m = average_basis(shape.n, shape.nev, p);
    double ortho = (kOrthoPerBasisEntry * n_eff * m + kOrthoPerB * (shape.nnz_b + n_eff)) * pd;
    if (p == 1) ortho *= kOrthoSingle;
    t[i] = apply + ortho;
  }
  return t;
}

// Block steps relative to p = 1 (operator applications / p of the width-p run over those of the
// p = 1 run, automatic ncv, tol 1e-11), medians over the data cases whose wanted eigenvalues are
// distinct: nev = 6 (cav2d-rect-pert-1e5), nev = 10..12 (16 cases), nev = 21..26 (2 cases).
// Multiple eigenvalues make wider blocks much better (nev >= 8, 12 cases: p = 4 0.34, p = 8 0.2),
// which the rule does not assume.
double relative_block_steps(index_t p, index_t nev) {
  if (p < 1 || nev < 1)
    HS_THROW(InvalidInput, "relative_block_steps: p = " << p << ", nev = " << nev);
  if (p == 1) return 1.0;
  struct Row {
    double nev;
    WidthTable steps;
  };
  static constexpr Row kRows[] = {{6.0, {1.0, 0.82, 0.68, 0.55, 0.45}},
                                  {11.0, {1.0, 0.65, 0.50, 0.38, 0.31}},
                                  {22.0, {1.0, 0.58, 0.40, 0.29, 0.23}}};
  // Widths between the table's: geometric interpolation in p; beyond 16: the 8 -> 16 trend.
  const auto at = [](const WidthTable& s, index_t q) {
    const double lq = std::log2(static_cast<double>(q));
    const auto lo = static_cast<std::size_t>(std::min(std::floor(lq), 3.0));
    const double f = lq - static_cast<double>(lo);
    return std::exp((1.0 - f) * std::log(s[lo]) + f * std::log(s[lo + 1]));
  };
  const double x = std::clamp(static_cast<double>(nev), kRows[0].nev, kRows[2].nev);
  const std::size_t r = x <= kRows[1].nev ? 0 : 1;
  const double f = (x - kRows[r].nev) / (kRows[r + 1].nev - kRows[r].nev);
  return (1.0 - f) * at(kRows[r].steps, p) + f * at(kRows[r + 1].steps, p);
}

index_t recommended_block_size(index_t n, index_t nev, const WidthTable& step_cost) {
  if (nev < 1 || n < nev)
    HS_THROW(InvalidInput, "recommended_block_size: nev = " << nev << ", n = " << n);
  for (double c : step_cost)
    if (!(c > 0.0) || !std::isfinite(c))
      HS_THROW(InvalidInput, "recommended_block_size: step costs must be positive and finite");
  LanczosSetup s;
  s.n = s.n_free = n;
  s.nev = nev;
  s.ncv = std::min(n, std::max(2 * nev + 1, nev + 16));
  index_t best = 1;
  double best_cost = step_cost[0];
  for (std::size_t i = 1; i < kRuleWidthCount; ++i) {
    const index_t p = kRuleWidths[i];
    if (!block_dimensions(s, p, true).feasible) break;  // wider ones do not fit either
    const double cost = step_cost[i] * relative_block_steps(p, nev);
    if (cost < (1.0 - kRuleMargin) * best_cost) {
      best = p;
      best_cost = cost;
    }
  }
  return best;
}

index_t recommended_block_size(const ShiftInvertShape& shape) {
  return recommended_block_size(shape.n, shape.nev, modelled_step_seconds(shape));
}

}  // namespace hs::eigen_detail
