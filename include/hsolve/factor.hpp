// Sparse direct factorization of a Hermitian K = alpha*A + beta*B (module src/factor, MUMPS).
// Scalar set: T in {double, cplx}.
//
// One factorization path, always with inertia:
//   * real T    : MUMPS SYM=2 (LDL^T, 2x2 pivots) on the pencil's upper pattern.
//   * complex T : MUMPS has no Hermitian LDL^H (complex SYM=2 means complex-SYMMETRIC), so K is
//                 factorized through its real symmetric embedding E = [[Re K, -Im K],[Im K, Re K]]
//                 (order 2n, SYM=2). Solves: E [Re x; Im x] = [Re b; Im b]. inertia(K) = inertia(E)/2.
//                 Flops and memory match a complex LU of K; inertia comes for free. Real or
//                 imaginary parts that are exactly zero in both A and B (hence for every real
//                 alpha, beta) are left out of E's pattern. That pays off only when Im K
//                 vanishes on whole regions of the graph: with Im K == 0 E decouples into two
//                 copies of Re K (2-D grid, n = 1e5: 0.9 GFlop, 134 MB vs 3.5 GFlop, 209 MB
//                 with every coupling complex), while phases on every tenth grid row already
//                 cost as much as a fully complex K.
//
// MUMPS facts this contract relies on (verified on 5.8.2 by the P0 design review):
//   * Null-pivot detection needs ICNTL(24)=1 AND CNTL(3) > 0; with CNTL(3)=0 an exactly singular
//     order-3600 matrix factorizes "successfully". With ICNTL(24)=1 MUMPS never fails on null
//     pivots: it fixes them and reports INFOG(28); solves then silently drop that direction.
//   * INFOG(12) (negative pivots) is the exact inertia count when INFOG(28) == 0; near/at a
//     singular shift it is unreliable.
//   * INFOG values above 2^31 are reported negative in units of 1e6 ("-x means x*10^6").
//   * MB in MUMPS means 1e6 bytes.
//   * Workspace errors are usually cured by a larger ICNTL(14) without re-analysis (a
//     saddle-point matrix with ICNTL(14)=0 fails once, then succeeds), but not always: with 1500
//     multipliers on a 40x40 grid and ICNTL(14)=0, 10, 20, 40 all fail (INFO -9). BLR enabled at
//     analysis can be switched off for the factorization; ICNTL(22) (out-of-core) can change
//     between factorizations, and JOB=-2 deletes the factor files.
//   * ICNTL(23) > 0 makes MUMPS size its workspace from the cap, not from the estimate:
//     INFOG(18) (allocated) then approaches cap/2. That is a virtual reservation; resident memory
//     only grows by INFOG(21) (effectively used). 100x100 Laplacian, cap 3000 MB: INFOG(18) =
//     1503, INFOG(21) = 6, VmPeak +1.37 GB. A cap equal to the INFOG(16) estimate can still fail
//     with INFO -19: the smallest caps that worked were 0.96 .. 1.01 x INFOG(16) (+1..2 MB) for
//     Laplacians, random and saddle-point matrices, and 1.29 x INFOG(16) when delayed pivots
//     forced a workspace retry (which grows the workspace beyond the estimate). So
//     "estimate <= cap" does not guarantee success: leave a margin.
//   * With L0-OpenMP tree parallelism (ICNTL(48)=1, the MUMPS default) and more than one thread,
//     repeated factorizations and solves of the same data differ in the last bits from call to
//     call (dynamic assembly order). ICNTL(48)=0 makes them bitwise reproducible for a fixed
//     thread count (FactorOptions::deterministic).
//
// Null-pivot threshold (CNTL(3)), measured on 5.8.2 with ICNTL(8)=7 scaling. MUMPS flags a pivot
// whose remaining row is below CNTL(3)*||A_pre|| (A_pre: the scaled, permuted matrix). At an
// exactly singular shift the decisive pivot is a rounding residue that grows with n for
// delocalized null vectors (larger separators) and depends on the elimination order; at
// |sigma - lambda| = d*||K|| the pivot is larger by a factor of about d / eps. Measured on 2-D
// Laplacians (24 exact eigenvalues across the spectrum) and prescribed-spectrum matrices
// (localized eigenvectors, 31 eigenvalues), real and complex, analysis at a generic shift:
//   * CNTL(3) needed to flag every exact shift: 1e-11 at n = 2550; 1e-9 at n ~ 10^4 (1e-10
//     misses up to 4/24, e.g. on 101x100); 3e-9 at n = 4*10^4 (200x201: 1e-9 misses 1/24);
//     <= 1e-10 for the 12 shifts tried at n = 9*10^4 and for prescribed spectra (n = 3000,
//     12000). Analysing at the singular shift itself, as the driver does for its first shift,
//     gives the same picture (100x100: 1e-10 misses 1/24, 3e-10 none); an analysis at a shift
//     that zeroes the diagonal (2x2-compressed ordering) needed 3e-9 on a 50x51 grid.
//   * Shifts at d*||K|| from an eigenvalue start being (falsely) flagged at CNTL(3) ~ 1e3*d:
//     at 1e-7 for d = 1e-10 (localized modes, smallest grid), at 1e-6 for d = 1e-9. Nothing at
//     d = 1e-10 was flagged by CNTL(3) <= 3e-8, and the inertia of every unflagged near shift
//     was exact.
// Hence the double default 1e-8 (null_pivot_tol): about 3x above the largest value detection
// needed and 10x below the onset of false flags at d = 1e-10. Trade-offs:
//   * The detection margin shrinks as n grows. A missed exact singularity factorizes with a tiny
//     but finite pivot: solves stay backward stable, but the zero eigenvalue is counted on an
//     arbitrary side, which is why inertia endpoints must lie in spectral gaps.
//   * The near-singular guarantee is relative to ||K|| (the driver's scale), not to |lambda|: a
//     shift at 1e-10*|lambda| from an eigenvalue with |lambda| << ||K|| can lie within a few
//     hundred eps*||K|| of it and is then flagged singular, the safe verdict at that distance
//     (100x101 grid: lambda_min = 1.9e-3, sigma = lambda(1 +- 1e-10) is 2.4e-14*||K|| away).
// Single precision (smumps, single_null_pivot_tol): rounding K to float moves every eigenvalue by
// ~eps_s*||K|| (eps_s = 6e-8), so a shift exactly on an eigenvalue of K is, for the float matrix,
// a shift ~eps_s*||K|| away, and its pivot is correspondingly large. The same rule, scaled to the
// smallest distance a Single factor must accept (DESIGN 3.4b: 1e-6*||K||), gives the default 1e-4:
// no shift at d >= 1e-6 was flagged by any CNTL(3) <= 3e-4, and exact shifts on localized
// eigenvectors are caught (prescribed spectra: 31/31). Delocalized ones mostly are not, for any
// threshold (Laplacians at 1e-4: 16/24 for n = 2550, 3/24 for n = 10^4, 2/12 for n = 4*10^4), so
// factorize() cannot certify a Single factor. Whether it is usable (eps_s * cond(K) < 1) shows in
// the refinement residual ShiftInvertOperator::last_relative_residual(), which callers of Single
// (and BLR) factors must check.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

enum class FactorPrecision { Double, Single };

struct FactorOptions {
  FactorPrecision precision = FactorPrecision::Double;  ///< Single: s/cmumps (pair with SolveRefinement)
  int ordering = 7;           ///< ICNTL(7): 7 auto, 5 METIS, 3 SCOTCH, 4 PORD, 0 AMD
  int scaling = 7;            ///< ICNTL(8): 7 = recomputed at every factorization (shift-independent)
  int threads = 0;            ///< ICNTL(16) OpenMP threads; 0 = omp_get_max_threads()
  int mem_relax_pct = 35;     ///< ICNTL(14) initial value; on a workspace error it is doubled (at
                              ///< least +10 points, so 0 still grows) and kept for later calls
  double max_memory_mb = 0;   ///< ICNTL(23) working-memory cap (MB = 1e6 B); 0 = no cap. Needs a
                              ///< margin above estimate().incore_mb, and MUMPS reserves (virtual)
                              ///< memory up to about half of it (see header comment)
  bool blr = false;           ///< block low-rank factorization (ICNTL(35)); pair with SolveRefinement
  double blr_tol = 1e-10;     ///< CNTL(7)
  bool blr_estimates = false; ///< enable BLR at analysis only to obtain INFOG(36..39) estimates
  bool out_of_core = false;   ///< ICNTL(22); can be changed later with set_out_of_core()
  std::string ooc_dir;        ///< out-of-core directory (empty: MUMPS default / TMPDIR)
  double null_pivot_tol = 1e-8;  ///< CNTL(3) in (0, 1) of Double factorizations (ICNTL(24)=1):
                                 ///< pivot rows below this, relative to the preprocessed matrix
                                 ///< norm, are null => factorize() returns false (default: see
                                 ///< the measurements in the header comment)
  double single_null_pivot_tol = 1e-4;  ///< the same for Single factorizations (smumps)
  bool deterministic = true;  ///< ICNTL(48)=0: no L0-OpenMP tree parallelism, so factorizations and
                              ///< solves are bitwise reproducible for a fixed thread count. false
                              ///< (ICNTL(48)=1) saved 10-12% of the factorization time and 20-26%
                              ///< of 16-RHS solve time (2-D Laplacian n = 2.5e5, complex Hermitian
                              ///< n = 1e5; 2 threads, likely more with many threads); results then
                              ///< vary in the last bits from run to run
  int verbosity = 0;          ///< 0 silent; > 0 MUMPS diagnostics (ICNTL(4) = min(verbosity, 4)) on
                              ///< stdout: MUMPS treats Fortran unit 0 (stderr) as "off"
};

/// Matrix inertia (counts of negative / zero / positive eigenvalues). Only meaningful if valid.
struct Inertia {
  index_t neg = -1;
  index_t zero = -1;
  index_t pos = -1;
  bool valid = false;
};

/// Predictions available right after analyze(). All INFOG values decoded (see header comment).
struct FactorEstimate {
  std::int64_t factor_entries = 0;  ///< INFOG(20): estimated entries in the factors
  double incore_mb = 0.0;           ///< INFOG(16)/(17): in-core working memory
  double ooc_mb = 0.0;              ///< INFOG(26)/(27): out-of-core working memory
  double blr_incore_mb = -1.0;      ///< INFOG(36): only when BLR was active at analysis, else -1
  double blr_ooc_mb = -1.0;         ///< INFOG(38): idem
  double flops = 0.0;               ///< RINFOG(1): elimination flops
};

struct FactorStats {
  index_t analyses = 0;      ///< successful + failed analyze() calls
  double analyze_s = 0.0;    ///< accumulated over all analyze() calls
  double factorize_s = 0.0;  ///< accumulated over all factorize() calls
  double solve_s = 0.0;      ///< accumulated over all solve() calls
  index_t factorizations = 0;    ///< successful + failed factorize() calls
  index_t solves = 0;            ///< solve() calls
  index_t rhs_solved = 0;        ///< total right-hand sides
  FactorEstimate estimate;
  std::int64_t factor_entries = 0;  ///< INFOG(9) after the last factorization
  double factor_mb_used = 0.0;      ///< INFOG(21): memory effectively used by the last factorization
  double factor_mb_allocated = 0.0; ///< INFOG(18): memory allocated for it (approaches cap/2 when
                                    ///< max_memory_mb > 0, see header comment)
  index_t null_pivots = 0;          ///< INFOG(28) after the last factorization (of E for complex T)
  index_t delayed_pivots = 0;       ///< INFOG(13) after the last factorization (of E for complex T)
  index_t workspace_retries = 0;    ///< automatic ICNTL(14) increases (all calls)
  int last_info1 = 0;               ///< INFOG(1) of the last MUMPS call
  int last_info2 = 0;               ///< INFOG(2)
};

/// One object = one sparsity pattern (a ShiftedPencil): analyze() once, then factorize() any
/// number of times for different (alpha, beta). Not thread-safe; not copyable.
template <class T>
class SparseFactorization {
 public:
  virtual ~SparseFactorization() = default;

  /// Symbolic analysis on pencil's upper pattern. The values of alpha*A + beta*B are passed to
  /// MUMPS as representative values (analysis-time matching for 2x2 pivots uses them); pass
  /// the first shift you will factorize. The pencil must outlive this object. Calling it again
  /// re-analyses (and discards the factorization). Throws InvalidInput (empty pencil, non-finite
  /// alpha/beta), Unsupported (MUMPS order above 2^31-1), OutOfMemoryBudget (INFO -5/-7/-13)
  /// or FactorizationFailed (other negative INFO).
  virtual void analyze(const ShiftedPencil<T>& pencil, double alpha, double beta) = 0;
  /// Valid after analyze() (Internal before).
  virtual FactorEstimate estimate() const = 0;

  /// Numeric factorization of K = alpha*A + beta*B (real alpha, beta). Requires analyze()
  /// (Internal otherwise).
  /// Returns false when K is (numerically) singular: INFO(1) == -10 or INFOG(28) > 0 null
  /// pivots, or (complex T) an odd INFOG(12): the two copies in E of an eigenvalue at the
  /// shift got opposite signs, i.e. K is singular to working precision although no pivot fell
  /// below CNTL(3). Then stats().null_pivots is set, inertia().valid is false, solve() throws,
  /// and the object stays analysed so the caller can refactorize at a perturbed shift.
  /// Detection is reliable in double precision (see the header comment for its limits) but
  /// only partial for Single: a Single factor that returns true may still be unusable at this
  /// shift, which the refinement residual of ShiftInvertOperator reveals.
  /// Other failures throw hs::Error:
  ///   workspace errors (INFO -8,-9,-14,-15,-17,-20) are retried internally up to 3 times with
  ///   ICNTL(14) doubled (stats().workspace_retries), then FactorizationFailed;
  ///   INFO -13 / -19 (allocation failure / ICNTL(23) cap): OutOfMemoryBudget;
  ///   any other negative INFO: FactorizationFailed (message carries INFO(1), INFO(2)).
  [[nodiscard]] virtual bool factorize(double alpha, double beta) = 0;

  /// Solve K X = B in place; b is n x nrhs column-major with ldb >= n. Multiple right-hand sides
  /// are solved in one MUMPS call (block triangular solves). Throws Internal without a valid
  /// factorization, InvalidInput for nrhs < 0 or ldb < n, OutOfMemoryBudget / FactorizationFailed
  /// for MUMPS errors (as in factorize()). A Single factorization rounds b to float and back.
  virtual void solve(T* b, index_t nrhs, index_t ldb) = 0;

  /// Inertia of the last factorized K: neg = INFOG(12), zero = null pivots (both halved, zero
  /// rounded up, for complex T), pos = n - neg - zero, valid = (last factorize() succeeded, i.e.
  /// no null pivots). With Single or BLR it is the inertia of the rounded / compressed matrix,
  /// exact only when no eigenvalue lies within that perturbation of the shift (compute_inertia
  /// always factorizes in double precision without BLR).
  virtual Inertia inertia() const = 0;

  /// Toggle out-of-core storage of the factors; takes effect at the next factorize().
  virtual void set_out_of_core(bool on) = 0;

  virtual const FactorStats& stats() const = 0;
  virtual const FactorOptions& options() const = 0;
  /// Order n of K (not of the embedding).
  virtual index_t order() const = 0;
};

/// MUMPS-backed factorization (the only backend).
template <class T>
std::unique_ptr<SparseFactorization<T>> make_mumps_factorization(const FactorOptions& options = {});

/// Inertia of alpha*A + beta*B in one call (own analysis + factorization). Always uses double
/// precision without BLR whatever `options` says (inertia of a perturbed matrix is wrong near
/// small gaps). Never fails on singularity: returns valid == false instead.
/// For the B-definiteness test and unit tests; the driver reuses its main factorization object
/// (same pattern) for certificate endpoints. B is HPD iff valid && neg == 0 && zero == 0.
template <class T>
Inertia compute_inertia(const ShiftedPencil<T>& pencil, double alpha, double beta,
                        const FactorOptions& options = {});

}  // namespace hs
