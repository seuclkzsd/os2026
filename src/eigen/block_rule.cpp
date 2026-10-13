// Block-size rule of shift-invert block Lanczos; see block_rule.hpp. Calibration below.
#include "eigen/block_rule.hpp"

#include <algorithm>
#include <cmath>

#include "eigen/block_lanczos.hpp"
#include "hsolve/error.hpp"

namespace hs::eigen_detail {

namespace {

// ---------------------------------------------------------------------------------------------
// Calibration: bench/blocktune_sweep.cpp on 17 HEP/GHEP data cases (3-D, 2-D, complex, distinct
// and multiple eigenvalues, n = 1e4..1.5e5), 16 threads in the measurement slot of a 2x Xeon Silver
// 4314 node (cores 16-31 = the judging machine's 16 cores), MUMPS L0-OpenMP, the driver's
// adaptive ordering, block steps with BlockOrthonormalizer::try_fast (the solver's default).
// Constants are least-squares fits on relative errors.
//
// Operator application to a block of p columns, apply(p) = a + b p for p >= 2 and
// apply(1) = kSingleRhs (a + b) (MUMPS solves one right-hand side by a different, BLAS2 path that
// does not use 16 threads well), with n_eff = n (real) or 2n (complex: MUMPS factorizes the real
// embedding of order 2n), E = factor entries and the small-front term s = n_eff^2 / E
// (= n_eff / (fill per row): factors with little fill per row consist of many small fronts whose
// per-front overhead dominates both the fixed and the per-column cost):
//   a = kApplyPerEntry E + kApplyPerSmallFront s      (reading the factors once per solve)
//   b = kApplyPerRhsRow n_eff + kApplyPerRhsSmallFront s   (per right-hand side)
// Fit range of a / b (prediction / measurement): 0.44..1.56 / 0.35..1.64; e.g. 3-D
// yee-pec-box-1e5 a = 37.6 ms, b = 1.15 ms; 2-D cav2d-rect-pert-1e5 3.8 / 0.83 ms; random sparse
// syn-zero-cluster-real-1e5 (fill 24 per row) 11.5 / 3.6 ms.
constexpr double kApplyPerEntry = 0.745e-9;
constexpr double kApplyPerSmallFront = 1150e-9;
constexpr double kApplyPerRhsRow = 8.69e-9;
constexpr double kApplyPerRhsSmallFront = 194e-9;
constexpr double kSingleRhs = 1.1;  // median apply(1) / (a + b), range 0.83..1.45
// Orthogonalization of one step against a basis of m vectors (m: three quarters of the cycle's
// ncv, the average basis size of a step) with n_s = n_eff scalars per vector (complex: twice the
// bytes), including the applications of B:
//   block (try_fast):   o(p) = (kOrthoPerBasisEntry n_s m + kOrthoPerRow (nnz_b + n_s)) p
//   p = 1 (CGS2):       o(1) =  kOrthoSinglePerBasisEntry n_s m + kOrthoSinglePerRow (nnz_b + n_s)
// Fit range 0.35..1.82 (block) and 0.59..1.44 (p = 1).
constexpr double kOrthoPerBasisEntry = 0.137e-9;
constexpr double kOrthoPerRow = 0.374e-9;
constexpr double kOrthoSinglePerBasisEntry = 0.15e-9;
constexpr double kOrthoSinglePerRow = 0.772e-9;
//
// Result on the calibration cases: iteration time T(p) / T(1) of the width the rule picks against
// the best width of {1, 2, 4, 8, 16} (medians of 3 interleaved repetitions):
//   3-D, 1e5 (ss-2cubes_sphere, ss-qa8fk-qa8fm, ss-tem152078, syn-lap3d-aniso-1e5, yee-pec-box-1e5;
//     nev 10-12, distinct or few copies): p = 4 or 8, 0.44..0.59 (best 0.44..0.56);
//   3-D with multiple eigenvalues (syn-mem27-46, yee-pec-cube-32) and 3-D complex yee-bloch-1e4:
//     p = 8, 0.21 / 0.22 / 0.41 (= best);
//   little fill (cav2d-rect-pert-1e5 nev 6, cav2d-square-1e4 nev 7, syn-*-1e5 / -1e4 random
//     sparse incl. complex, ss-tem27623): p = 2, 0.56..0.92 (= best within 2 %);
//   syn-multiplicity-real-1e4 (3 distinct values in 14): p = 2, 0.60 (best p = 8 0.42).
// No pick is slower than p = 1; geometric mean over the 17 cases of T(pick) / T(best) = 1.03.
// Without try_fast (the run() kernel throughout) p = 2 lost to p = 1 on cav2d-rect-pert-1e5 (1.20).
// Validation on 14 further data cases (same setup, p in {1, 2, 4, 8}): no pick slower than p = 1
// beyond noise (syn-ghep-multiplicity-real-1e4, 8 distinct values in 22: p = 2 1.02, p = 4 0.81),
// picks within 15 % of the best width except where multiple eigenvalues favour wider blocks
// (that case, yee-pec-cube-16, syn-lap3d-22, syn-multiplicity-complex-1e4: 1.26..1.40).
// End to end (the CLI with --block-size forced, 16 threads in the slot, 5 interleaved runs, all
// valid by tools/eval, certificate passed): solver time of the pick vs p = 1 on 15 cases 1.04x
// (cav2d-rect-pert-1e5) .. 2.03x; p = 1 needed certificate reruns on the two low-fill multiplicity
// cases, where p = 4 beats the pick p = 2 by 12..17 %, while p = 2 beats p = 4 by 3..8 % on the
// four low-fill cases with distinct eigenvalues.

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

/// Widths of the step table of relative_block_steps.
constexpr index_t kStepWidths[] = {1, 2, 4, 8, 16};

}  // namespace

WidthTable modelled_step_seconds(const ShiftInvertShape& shape) {
  if (shape.n < 1 || shape.nev < 1 || !(shape.factor_entries >= 0.0) ||
      !std::isfinite(shape.factor_entries) || !(shape.nnz_b >= 0.0) || !std::isfinite(shape.nnz_b))
    HS_THROW(InvalidInput, "modelled_step_seconds: n = "
                               << shape.n << ", nev = " << shape.nev << ", factor entries "
                               << shape.factor_entries << ", nnz(B) " << shape.nnz_b);
  const double n_eff = static_cast<double>(shape.n) * (shape.complex_scalars ? 2.0 : 1.0);
  // A factor holds at least its diagonal; the bound keeps the small-front term finite.
  const double entries = std::max(shape.factor_entries, n_eff);
  const double small_fronts = n_eff * (n_eff / entries);
  const double a = kApplyPerEntry * entries + kApplyPerSmallFront * small_fronts;
  const double b = kApplyPerRhsRow * n_eff + kApplyPerRhsSmallFront * small_fronts;
  const double rows = shape.nnz_b + n_eff;
  WidthTable t{};
  for (std::size_t i = 0; i < kRuleWidthCount; ++i) {
    const index_t p = kRuleWidths[i];
    const double pd = static_cast<double>(p);
    const double m = average_basis(shape.n, shape.nev, p);
    if (p == 1) {
      t[i] =
          kSingleRhs * (a + b) + kOrthoSinglePerBasisEntry * n_eff * m + kOrthoSinglePerRow * rows;
    } else {
      t[i] = a + b * pd + (kOrthoPerBasisEntry * n_eff * m + kOrthoPerRow * rows) * pd;
    }
  }
  return t;
}

// Block steps relative to p = 1 (operator applications / p of the width-p run over those of the
// p = 1 run, automatic ncv, tol 1e-11; thread independent), medians over the data cases whose
// wanted eigenvalues are distinct: nev = 6 (cav2d-rect-pert-1e5), nev = 10..12 (16 cases),
// nev = 21..26 (syn-zero-cluster-real-1e5, syn-ghep-zero-cluster-complex-1e4). Multiple eigenvalues
// make wider blocks much better (12 cases with nev >= 8: p = 4 0.34, p = 8 0.2), which the rule
// does not assume. Nothing was measured below nev = 6; the row at nev = 1 is "no gain".
double relative_block_steps(index_t p, index_t nev) {
  if (p < 1 || nev < 1)
    HS_THROW(InvalidInput, "relative_block_steps: p = " << p << ", nev = " << nev);
  if (p == 1) return 1.0;
  struct Row {
    double nev;
    double steps[5];  // kStepWidths
  };
  static constexpr Row kRows[] = {{1.0, {1.0, 1.0, 1.0, 1.0, 1.0}},
                                  {6.0, {1.0, 0.82, 0.68, 0.55, 0.45}},
                                  {11.0, {1.0, 0.65, 0.50, 0.38, 0.31}},
                                  {22.0, {1.0, 0.58, 0.40, 0.29, 0.23}}};
  constexpr std::size_t kLastRow = sizeof(kRows) / sizeof(kRows[0]) - 1;
  // Widths between the table's: geometric interpolation in p; beyond 16: no further gain.
  const auto at = [](const Row& row, index_t q) {
    if (q >= kStepWidths[4]) return row.steps[4];
    const double lq = std::log2(static_cast<double>(q));
    const auto lo = static_cast<std::size_t>(std::floor(lq));
    const double f = lq - static_cast<double>(lo);
    return std::exp((1.0 - f) * std::log(row.steps[lo]) + f * std::log(row.steps[lo + 1]));
  };
  const double x = std::min(static_cast<double>(nev), kRows[kLastRow].nev);
  std::size_t r = 0;
  while (r + 1 < kLastRow && x > kRows[r + 1].nev) ++r;
  const double f = (x - kRows[r].nev) / (kRows[r + 1].nev - kRows[r].nev);
  return (1.0 - f) * at(kRows[r], p) + f * at(kRows[r + 1], p);
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
