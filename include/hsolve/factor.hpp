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
//   * Workspace errors are cured by a larger ICNTL(14) without re-analysis (measured: a
//     saddle-point matrix with ICNTL(14)=0 fails once, then succeeds); BLR enabled at
//     analysis can be switched off for the factorization; ICNTL(22) (out-of-core) can change
//     between factorizations, and JOB=-2 deletes the factor files.
//
// Null-pivot threshold (null_pivot_tol = CNTL(3)), measured on 5.8.2 with ICNTL(8)=7 scaling.
// MUMPS flags a pivot whose remaining row is below CNTL(3)*||A_pre||. At an exactly singular
// shift the decisive (last root) pivot is roughly c*eps*||K|| / |v_r|^2, v_r being the null
// vector restricted to the root separator, so for delocalized null vectors it GROWS with n and
// depends on the elimination order; at |sigma - lambda| = delta*||K|| it is about
// delta*||K|| / |v_r|^2 >= delta*||K||.
//   * sigma exactly on an eigenvalue of 2-D Laplacians (24 shifts across the spectrum):
//       CNTL(3) = 1e-12 flags 22/24 (n=2500), 18/24 (n=10^4), 7/24 (n=4*10^4);
//       CNTL(3) = 1e-10 flags 24/24 for n = 2500 .. 4*10^4 and 12/12 for n = 9*10^4 (analysis
//       at a generic shift); with the analysis at a shift that zeroes the diagonal (different,
//       2x2-compressed ordering) a 50x51 grid gave 22/24 and needed 3e-9 for 24/24.
//   * |sigma - lambda| >= 1e-10 * ||K|| (delocalized Laplacian modes, n <= 9*10^4, and localized
//     modes of prescribed-spectrum matrices, real and complex): never flagged for any CNTL(3)
//     up to 1e-9; larger thresholds only flag shifts within ~1e-14*||K|| of an eigenvalue.
// Hence the default 1e-10: the largest value for which shifts at delta >= 1e-10 stay regular
// even in the worst case |v_r| ~ 1, with exact singularities caught in all generic-order tests.
// Trade-off: an exact singularity can escape detection (very large n, or a null vector that
// nearly vanishes on the root separator); the factorization then has a tiny but finite pivot
// (solves stay backward stable, the zero eigenvalue is counted on an arbitrary side), which is
// why inertia endpoints must lie in spectral gaps. The "near-singular but regular" guarantee is
// relative to ||K|| (the driver's scale), not to |lambda|.
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
  double max_memory_mb = 0;   ///< ICNTL(23) working-memory cap (MB = 1e6 B); 0 = no cap
  bool blr = false;           ///< block low-rank factorization (ICNTL(35)); pair with SolveRefinement
  double blr_tol = 1e-10;     ///< CNTL(7)
  bool blr_estimates = false; ///< enable BLR at analysis only to obtain INFOG(36..39) estimates
  bool out_of_core = false;   ///< ICNTL(22); can be changed later with set_out_of_core()
  std::string ooc_dir;        ///< out-of-core directory (empty: MUMPS default / TMPDIR)
  double null_pivot_tol = 1e-10;  ///< CNTL(3) in (0, 1) (with ICNTL(24)=1): pivot rows below this,
                                  ///< relative to the preprocessed matrix norm, are null =>
                                  ///< factorize() returns false (default: see header comment)
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
  double factor_mb_used = 0.0;      ///< INFOG(18) after the last factorization
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
