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
// Floor (P5c review). Minimizing the model alone picked p = 2 on every low-fill shape with
// nev >= 8, below the P4 auto width 4 - slower than P4 wherever multiple eigenvalues make the
// wider block pay, which the rule cannot see before the run. bench_blocktune_e2e (hs::solve
// in-process, 16 threads in the slot, medians of 3 interleaved runs, all certified, eta <= 2e-13):
// syn-ghep-multiplicity-real-1e4 total 0.182 s at p = 2 vs 0.162 s at p = 4, syn-multiplicity-
// complex-1e4 0.288 vs 0.236 s, against ss-tem27623 (distinct) 0.685 vs 0.689 s. Iteration time
// T(p) / T(1) of the sweeps: on 7 low-fill cases with distinct eigenvalues p = 2 beats p = 4 by
// 0.07..0.13 of T(1), on the 4 low-fill cases with multiplicities p = 4 beats p = 2 by 0.05..0.24
// (syn-multiplicity-real-1e4 0.60 -> 0.44, syn-ghep-multiplicity-real-1e4 1.02 -> 0.81,
// syn-multiplicity-complex-1e4 0.84 -> 0.60, syn-lap2d-100 0.55 -> 0.50). Hence nev >= 8 keeps
// p >= 4 (kRuleFloorWidth): never below P4, at a cost of <= 8 % end to end on distinct spectra
// (P5b end-to-end runs of p = 2 vs 4 on four low-fill distinct cases: 3..8 %).
//
// Result: iteration time T(p) / T(1) of the width the rule picks against the best measured width
// (bench_blocktune_sweep, medians of 3 interleaved repetitions, 16 threads in the slot):
//   3-D, 1e5, nev 10-12 (ss-2cubes_sphere, ss-qa8fk-qa8fm, ss-tem152078, syn-lap3d-aniso-1e5,
//     yee-pec-box-1e5): p = 4 or 8, 0.45..0.59 (1.00..1.06 of the best width);
//   3-D with multiple eigenvalues (syn-mem27-46, yee-pec-cube-32, syn-mem125-46) and 3-D complex
//     (yee-bloch-1e4, yee-photonic-*-1e4), ss-si34h36: p = 8, 0.21..0.51 (= best);
//   little fill, nev 6 / 7 (cav2d-rect-pert-1e5, cav2d-square-1e4): p = 2, 0.92 / 0.56 (= best);
//   little fill, nev >= 8 (ss-tem27623, syn-*-1e5 / -1e4 random sparse incl. complex): the floor
//     p = 4, 0.44..0.97 (1.00..1.16 of the best width, p = 2 on the distinct ones);
//   multiplicities the wider blocks exploit (syn-lap3d-22, yee-pec-cube-16; p = 4): 1.30 of p = 8;
//   nev = 1..7 on six cases (yee-pec-box-1e5, ss-qa8fk-qa8fm, ss-tem27623, cav2d-rect-pert-1e5,
//     yee-bloch-1e4, syn-near-singular-complex-1e4; target k replaced in a copy of config.json):
//     3-D p = 4, 0.40..0.77 (<= 1.09 of the best width), the others p = 2, 0.57..0.99 (= best
//     except syn-near-singular-complex-1e4 at nev = 1, 1.30 of p = 4).
// No pick is slower than p = 1 (worst 0.99); geometric mean of T(pick) / T(best) over the 61
// case/nev points 1.035 (the P5b rule without floor and small-nev rows: 1.11, picks up to 1.77 of
// the best width at small nev, where it kept p = 1 or 2). Without try_fast (the run() kernel
// throughout) p = 2 lost to p = 1 on cav2d-rect-pert-1e5 (1.20).
//
// End to end (bench/blocktune_fix_e2e.cpp: the hsolve CLI as tools/bench/compare.py runs it, with
// --block-size 1, 0 = the P4 auto width, and the rule's pick from the main shift's factorization;
// one binary, 16 threads in the slot, 5 interleaved runs; all 155 runs valid by
// python3 -m tools.eval.sweep - 90 complete, 65 consistent, max eta 2.6e-11 - certificate passed).
// Solver time (time_total_s, medians), pick: speed-up vs p = 1 / vs P4 auto:
//   ss-qa8fk-qa8fm 4 (= P4) 2.236 s: 1.25x / 1.00x;   yee-pec-box-1e5 8: 3.117 s, 1.43x / 1.01x;
//   ss-2cubes_sphere 8: 3.187 s, 1.24x / 0.98x;        syn-mem27-46 8: 2.569 s, 1.48x / 1.05x;
//   yee-pec-cube-32 8: 2.392 s, 1.86x / 1.12x;         yee-bloch-1e4 8: 1.034 s, 1.33x / 1.06x;
//   cav2d-rect-pert-1e5 2 (k = 6, P4: 1): 0.333 s, 1.03x / 1.02x;
//   cav2d-square-1e4 2 (k = 7): 0.049 s, 1.15x / 1.19x;
//   floor p = 4 (= P4): ss-tem27623 0.712 s 1.03x, syn-near-singular-complex-1e5 3.230 s 1.13x,
//   syn-ghep-multiplicity-real-1e4 0.163 s 2.01x, syn-multiplicity-complex-1e4 0.246 s 2.45x
//   (p = 1 needs a certificate rerun on the last two).
// The pick never loses to p = 1 (1.03x..2.45x); against P4 it gains up to 1.19x where it moves
// from 4 to 8 or from 1 to 2, and ss-2cubes_sphere is the one case where p = 8 is slower than 4
// (1.6 %; distinct eigenvalues, p = 8 needs 184 applications against 108): the 4 / 8 choice of the
// model cannot see the multiplicities that make 8 pay on syn-mem27-46 and yee-pec-cube-32.

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
// p = 1 run, automatic ncv, tol 1e-11; thread independent), medians over data cases whose wanted
// eigenvalues are distinct:
//   nev = 1, 2, 3, 5 (pooled, 24 runs: yee-pec-box-1e5, ss-qa8fk-qa8fm, ss-tem27623,
//     cav2d-rect-pert-1e5, yee-bloch-1e4, syn-near-singular-complex-1e4 with k replaced; per nev
//     the medians are 0.68..0.72 / 0.55..0.61 / 0.46..0.52 for p = 2 / 4 / 8, no trend in nev:
//     even one wanted pair converges in fewer block steps than single-vector applications, the
//     P5b table's "no gain at nev = 1" was wrong - its p = 1 / 2 picks at small nev took up to
//     1.77x the iteration time of the best width);
//   nev = 7 (the same cases but cav2d-rect-pert-1e5, whose 7th pair is a near-zero mode):
//     0.64 / 0.51 / 0.44, p = 2 rounded up to 0.65 (the nev = 11 value) to keep the table
//     non-increasing in nev;
//   nev = 10..12 (16 cases), nev = 21..26 (syn-zero-cluster-real-1e5,
//     syn-ghep-zero-cluster-complex-1e4).
// p = 16 at nev <= 7 (not measured there; only interpolates widths above 8): 0.82 of p = 8, the
// ratio at nev = 11. The single case of the P5b row at nev = 6 (cav2d-rect-pert-1e5, k = 6:
// 0.82 / 0.68 / 0.55, a case whose p = 1 run converges in one cycle) is replaced by the
// interpolation between nev = 5 and 7. Multiple eigenvalues make wider blocks much better
// (12 cases with nev >= 8: p = 4 0.34, p = 8 0.2), which the model does not assume (see the
// floor in recommended_block_size).
double relative_block_steps(index_t p, index_t nev) {
  if (p < 1 || nev < 1)
    HS_THROW(InvalidInput, "relative_block_steps: p = " << p << ", nev = " << nev);
  if (p == 1) return 1.0;
  struct Row {
    double nev;
    double steps[5];  // kStepWidths
  };
  static constexpr Row kRows[] = {{1.0, {1.0, 0.69, 0.56, 0.49, 0.40}},
                                  {5.0, {1.0, 0.69, 0.56, 0.49, 0.40}},
                                  {7.0, {1.0, 0.65, 0.51, 0.44, 0.36}},
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
  // Floor (see the calibration notes): multiplicities, invisible here, favour the wider block.
  if (nev >= kRuleFloorMinNev && best < kRuleFloorWidth &&
      block_dimensions(s, kRuleFloorWidth, true).feasible)
    best = kRuleFloorWidth;
  return best;
}

index_t recommended_block_size(const ShiftInvertShape& shape) {
  return recommended_block_size(shape.n, shape.nev, modelled_step_seconds(shape));
}

}  // namespace hs::eigen_detail
