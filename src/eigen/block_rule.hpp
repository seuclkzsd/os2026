// Block-size rule of shift-invert block Lanczos (module eigen, P5b "blocktune"): which Krylov block
// width p a run of lanczos_krylov_schur should use, from what the driver knows before the run.
// Module-internal (src/eigen), called by the driver (src/driver block_size_for).
//
// A run of width p costs  steps(p) * step(p),  step(p) = apply(p) + ortho(p):  one operator
// application to p columns (a p-RHS MUMPS solve and B X) and the block orthogonalization of that
// step. A p-RHS solve reads the factors once for all p columns, so step(p) grows slowly with p on
// 3-D factors (8 threads on 2x Xeon Silver 4314: step(4) ~ 1.0 to 1.1 step(1), step(8) ~ 1.3) and
// fast on sparse 2-D-like factors (step(4) ~ 1.4, step(8) ~ 2), while the number of block steps
// falls with p (the block Krylov space of degree j holds p (j + 1) vectors; for distinct
// eigenvalues steps(4) ~ 0.5 steps(1), with multiple eigenvalues much less). The rule minimizes
// the product with modelled step costs and the measured step ratios of distinct-eigenvalue
// runs (multiplicities only make wider blocks better than modelled). Everything is deterministic:
// the inputs are sizes and the analysis estimate, never a timing, so a run's p - and with it its
// eigenvectors - does not depend on the machine load. Measurements and calibration in
// block_rule.cpp.
// Scalar-independent; thread-safe (pure functions).
#pragma once

#include <array>
#include <cstddef>

#include "hsolve/types.hpp"

namespace hs::eigen_detail {

/// Block widths the rule chooses from.
inline constexpr std::array<index_t, 5> kRuleWidths = {1, 2, 4, 8, 16};
inline constexpr std::size_t kRuleWidthCount = kRuleWidths.size();

/// Per-width numbers, index i <-> kRuleWidths[i].
using WidthTable = std::array<double, kRuleWidthCount>;

/// What the driver knows about a shift-invert Lanczos run before it starts.
struct ShiftInvertShape {
  index_t n = 0;                ///< order of the pencil (complex T: complex order)
  index_t nev = 0;              ///< wanted pairs of the run
  double factor_entries = 0.0;  ///< entries of the factors (MUMPS INFOG(9), or the analysis
                                ///< estimate INFOG(20)); complex T: of the real embedding
  double nnz_b = 0.0;           ///< stored nonzeros of the inner-product matrix B (0: B = I)
  bool complex_scalars = false;
  int threads = 1;              ///< threads of the solves (the model is calibrated at 8)
};

/// Modelled seconds of one block step of width p (one operator application to p columns plus the
/// block orthogonalization of one step, ncv auto) per width of kRuleWidths (block_rule.cpp).
WidthTable modelled_step_seconds(const ShiftInvertShape& shape);

/// Expected block steps of a width-p run relative to p = 1 for nev wanted pairs (distinct
/// eigenvalues; measured, block_rule.cpp). 1 for p = 1; decreasing in p.
double relative_block_steps(index_t p, index_t nev);

/// The width minimizing step_cost[i] * relative_block_steps(kRuleWidths[i], nev) over the widths
/// whose block layout fits an operator of order n (eigen_detail::block_dimensions with the
/// automatic ncv, as the driver runs it; p = 1 always fits). A wider block must be cheaper by
/// kRuleMargin to be chosen over a narrower one (the step model is no better than that), so
/// p = 1 stays unless blocks clearly pay. step_cost: any unit, measured or modelled (all
/// positive and finite, else InvalidInput). nev >= 1, n >= nev (else InvalidInput).
index_t recommended_block_size(index_t n, index_t nev, const WidthTable& step_cost);

/// recommended_block_size(shape.n, shape.nev, modelled_step_seconds(shape)).
index_t recommended_block_size(const ShiftInvertShape& shape);

/// Relative advantage a wider width needs over every narrower candidate (see above).
inline constexpr double kRuleMargin = 0.05;

}  // namespace hs::eigen_detail
