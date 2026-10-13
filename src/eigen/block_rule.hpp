// Block-size rule of shift-invert block Lanczos (module eigen, "blocktune"): which Krylov block
// width p a nearest-target run of lanczos_krylov_schur should use, from what the driver knows once
// the shift is factorized. Module-internal (src/eigen), meant for the driver (see "Driver hook-up").
//
// Model. A run of width p costs  steps(p) * step(p),  step(p) = apply(p) + ortho(p):  one operator
// application to p columns (a p-RHS MUMPS solve and B X) plus the block orthogonalization of that
// step. A p-RHS solve reads the factors once for all p columns, so on 3-D factors step(p) grows
// slowly with p (16 threads: apply(8) ~ 1.1 to 1.3 apply(1)), on factors with little fill (2-D
// meshes, random sparse patterns) fast (apply(4) ~ 1.2 to 1.5 apply(1)). The number of block steps
// falls with p, for distinct eigenvalues to ~0.7 (p = 2), ~0.55 (p = 4), ~0.45 (p = 8) of the
// p = 1 applications at any nev (relative_block_steps), with multiple eigenvalues further. The
// rule takes the width minimizing the product of the modelled step costs (modelled_step_seconds)
// and the measured step ratios of distinct-eigenvalue runs, then applies a floor: with
// nev >= kRuleFloorMinNev it never returns fewer than kRuleFloorWidth columns, the P4 auto width,
// because the eigenvalues' multiplicities - invisible before the run - make the wider block pay
// (measured end to end: p = 4 beats p = 2 by 12..22 % on the low-fill multiplicity cases, p = 2
// beats p = 4 by 0..8 % on low-fill distinct ones; block_rule.cpp). Measurements and calibration
// in block_rule.cpp.
//
// Reproducibility. The rule reads no timing: p never depends on the machine load. It is a pure
// function of its inputs, and they are as reproducible as the factorization: with an ordering that
// is the same in every analysis (AMD - the adaptive analysis' choice on 2-D-like and low-fill
// patterns - or METIS, SolverConfig::deterministic = true) factor_entries and so p are identical
// in every run. The default nested-dissection ordering (SCOTCH, ICNTL(7) = 3) orders differently
// from analysis to analysis: factor_entries then vary by up to ~9 % between identical runs
// (measured 2.2..8.8 % on the 3-D data cases), and a shape within that distance of a switch between
// two widths can get either width in different runs - both results valid and equal to tolerance,
// as the SCOTCH orderings already make them (different factors, different rounding). The
// calibration shapes keep their pick for entries within +-10 % (test
// LanczosBlockRuleFix.PinnedPicksSurviveTheOrderingSpread).
//
// Driver hook-up (P5c integration, src/driver/internal.hpp): the P4 rule (block_size_for(cfg,
// nev): 4 for nev >= 8, else 1) has no factor statistics, so
//   1. solve_nearest calls block_size_for(cfg, ShiftInvertShape) after factorize_near: for
//      SolverConfig::block_size == 0 this rule on {n, nev, FactorStats::factor_entries of the
//      main shift's factorization, nnz(B), scalar type};
//   2. the memory plan before the factorization (krylov_reservation) reserves the basis of the
//      widest width the rule returns (kRuleWidths.back() = 8) for nearest targets, falling back
//      to the P4 width when that does not fit; fit_block_size still halves 8 -> 4 with a warning
//      on memory-tight runs;
//   3. the rule is calibrated on nearest-target runs (k wanted pairs next to sigma); spectrum
//      slicing (per-slice nev, definite.cpp) keeps the P4 rule until measured.
// End to end with the driver and the BLAS hook (src/blas, route all; steps.cpp): the 4 -> 8 moves
// on distinct eigenvalues lose 2 - 4 % there (the hook's narrow kernels speed up the p <= 4
// solves the calibration timed without it): the step model wants a recalibration with the hook.
// Scalar-independent; thread-safe (pure functions).
#pragma once

#include <array>
#include <cstddef>

#include "hsolve/types.hpp"

namespace hs::eigen_detail {

/// Block widths the rule chooses from (p = 16 was never measurably better than p = 8 and needs
/// twice the basis memory, block_rule.cpp).
inline constexpr std::array<index_t, 4> kRuleWidths = {1, 2, 4, 8};
inline constexpr std::size_t kRuleWidthCount = kRuleWidths.size();

/// Floor of the rule: runs with at least kRuleFloorMinNev wanted pairs get at least
/// kRuleFloorWidth columns (the P4 auto width, driver block_size_for), when that layout fits.
inline constexpr index_t kRuleFloorMinNev = 8;
inline constexpr index_t kRuleFloorWidth = 4;

/// Per-width numbers, index i <-> kRuleWidths[i].
using WidthTable = std::array<double, kRuleWidthCount>;

/// What the driver knows about a shift-invert Lanczos run once the shift is factorized.
struct ShiftInvertShape {
  index_t n = 0;                ///< order of the pencil (complex scalars: the complex order)
  index_t nev = 0;              ///< wanted pairs of the run
  double factor_entries = 0.0;  ///< entries of the factors of the run's shift
                                ///< (FactorStats::factor_entries; complex scalars: of MUMPS's real
                                ///< embedding of order 2n). See "Reproducibility" above.
  double nnz_b = 0.0;           ///< stored nonzeros of the inner-product matrix B (0: B = I)
  bool complex_scalars = false;
};

/// Modelled seconds of one block step of width p (one operator application to p columns plus the
/// block orthogonalization of one step, automatic ncv) per width of kRuleWidths, calibrated at 16
/// threads (block_rule.cpp). InvalidInput for n < 1, nev < 1, or negative / non-finite
/// factor_entries or nnz_b.
WidthTable modelled_step_seconds(const ShiftInvertShape& shape);

/// Expected block steps of a width-p run (automatic ncv) relative to the applications of the
/// p = 1 run, for nev wanted pairs with distinct eigenvalues (measured at nev = 1, 2, 3, 5, 7,
/// 10..12 and 21..26, block_rule.cpp): 1 for p = 1, non-increasing in p and in nev, below 1 for
/// every p > 1 and nev >= 1 (blocks need fewer steps even for a single wanted pair: p = 2 at
/// nev = 1 took 0.53..0.82 of the p = 1 applications in block steps). p >= 1, nev >= 1, else
/// InvalidInput.
double relative_block_steps(index_t p, index_t nev);

/// Relative advantage a wider width needs over every narrower candidate (the step model is no
/// better than that), so p = 1 stays unless blocks clearly pay.
inline constexpr double kRuleMargin = 0.05;

/// The width minimizing step_cost[i] * relative_block_steps(kRuleWidths[i], nev) over the widths
/// whose block layout fits an operator of order n (eigen_detail::block_dimensions with the
/// automatic ncv, as the driver runs it; p = 1 always fits); a width replaces the best narrower one
/// only when it is cheaper by kRuleMargin. Then the floor: nev >= kRuleFloorMinNev raises the
/// result to kRuleFloorWidth if that layout fits. step_cost: any unit, measured or modelled (all
/// positive and finite, else InvalidInput). nev >= 1, n >= nev (else InvalidInput).
index_t recommended_block_size(index_t n, index_t nev, const WidthTable& step_cost);

/// recommended_block_size(shape.n, shape.nev, modelled_step_seconds(shape)).
index_t recommended_block_size(const ShiftInvertShape& shape);

}  // namespace hs::eigen_detail
