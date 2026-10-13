// Block-size rule of shift-invert block Lanczos (module eigen, P5b "blocktune"): which Krylov block
// width p a run of lanczos_krylov_schur should use, from what the driver knows once the shift is
// factorized. Module-internal (src/eigen), meant to be called by the driver (src/driver
// block_size_for).
//
// A run of width p costs  steps(p) * step(p),  step(p) = apply(p) + ortho(p):  one operator
// application to p columns (a p-RHS MUMPS solve and B X) plus the block orthogonalization of that
// step. A p-RHS solve reads the factors once for all p columns, so on 3-D factors step(p) grows
// slowly with p (16 threads: apply(8) ~ 1.1 to 1.3 apply(1)), on factors with little fill (2-D
// meshes, random sparse patterns) fast (apply(4) ~ 1.2 to 1.5 apply(1)). The number of block steps
// falls with p: for distinct eigenvalues steps(4) ~ 0.5 steps(1), with multiple eigenvalues more.
// The rule minimizes the product with the modelled step costs (modelled_step_seconds) and the
// measured step ratios of distinct-eigenvalue runs (relative_block_steps; multiplicities only make
// wider blocks better than modelled). Deterministic: the inputs are sizes, never a timing, so a
// run's p - and with it its eigenvectors - does not depend on the machine load. Measurements and
// calibration in block_rule.cpp.
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

/// Per-width numbers, index i <-> kRuleWidths[i].
using WidthTable = std::array<double, kRuleWidthCount>;

/// What the driver knows about a shift-invert Lanczos run once the shift is factorized.
struct ShiftInvertShape {
  index_t n = 0;                ///< order of the pencil (complex scalars: the complex order)
  index_t nev = 0;              ///< wanted pairs of the run
  double factor_entries = 0.0;  ///< entries of the factors (FactorStats::factor_entries; complex
                                ///< scalars: of MUMPS's real embedding of order 2n)
  double nnz_b = 0.0;           ///< stored nonzeros of the inner-product matrix B (0: B = I)
  bool complex_scalars = false;
};

/// Modelled seconds of one block step of width p (one operator application to p columns plus the
/// block orthogonalization of one step, automatic ncv) per width of kRuleWidths, calibrated at 16
/// threads (block_rule.cpp). InvalidInput for n < 1, nev < 1, or negative / non-finite
/// factor_entries or nnz_b.
WidthTable modelled_step_seconds(const ShiftInvertShape& shape);

/// Expected block steps of a width-p run (automatic ncv) relative to the applications of the
/// p = 1 run, for nev wanted pairs with distinct eigenvalues (measured, block_rule.cpp): 1 for
/// p = 1, non-increasing in p, and 1 for every p at nev = 1 (no measurement below nev = 6: the
/// table falls linearly from "no gain" at nev = 1 to the measured row at nev = 6). p >= 1,
/// nev >= 1, else InvalidInput.
double relative_block_steps(index_t p, index_t nev);

/// Relative advantage a wider width needs over every narrower candidate (the step model is no
/// better than that), so p = 1 stays unless blocks clearly pay.
inline constexpr double kRuleMargin = 0.05;

/// The width minimizing step_cost[i] * relative_block_steps(kRuleWidths[i], nev) over the widths
/// whose block layout fits an operator of order n (eigen_detail::block_dimensions with the
/// automatic ncv, as the driver runs it; p = 1 always fits); a width replaces the best narrower one
/// only when it is cheaper by kRuleMargin. step_cost: any unit, measured or modelled (all positive
/// and finite, else InvalidInput). nev >= 1, n >= nev (else InvalidInput).
index_t recommended_block_size(index_t n, index_t nev, const WidthTable& step_cost);

/// recommended_block_size(shape.n, shape.nev, modelled_step_seconds(shape)).
index_t recommended_block_size(const ShiftInvertShape& shape);

}  // namespace hs::eigen_detail
