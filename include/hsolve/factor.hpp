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
//     thread count (FactorOptions::deterministic). With ICNTL(48)=1 a factorization must run with
//     the thread count of its analysis, else INFO(1) = -58; analyze() therefore fixes ICNTL(16)
//     (threads = 0 -> omp_get_max_threads() at analysis). With ICNTL(16) > 0 MUMPS restores the
//     caller's OpenMP thread count after each call.
//   * Sequential MUMPS is NOT thread-safe across distinct instances: factorizations share
//     process-global Fortran module state (MUMPS_LOAD_INIT arrays, communication buffers); two
//     instances factorizing concurrently from two std::threads aborted in MUMPS_LOAD_INIT
//     (SIGSEGV) in every trial. All MUMPS calls of the process are therefore serialized by one
//     lock: objects may be used from different threads, but never gain from it.
//   * Two SCOTCH analyses of the same matrix in one process give different orderings (SCOTCH's
//     random state is not reset between calls; flops differed by 10%), while a fresh process
//     reproduces them. An analysis given a previous analysis' pivot order (SYM_PERM -> PERM_IN,
//     ICNTL(7)=1) takes 0.03-0.05 s instead of 0.9-1.3 s (SCOTCH, n ~ 1e5); against AMD it saves
//     little or nothing (0.035 vs 0.043 s on wg-square-1e5, 0.38 vs 0.26-0.30 s on the 5e6-entry
//     complex embedding of syn-near-singular-complex-1e5: reading the input dominates). It does
//     not quite reproduce the elimination (MUMPS rebuilds the tree; the SYM_PERM it returns
//     differs from the one given): +0.4..3.4% factor entries on the 1e5 test cases, +1..7% (AMD,
//     SCOTCH) and up to +15% (METIS) on random sparse matrices.
//   * ICNTL(31)=1 (discard factors) is read at ANALYSIS; set before a factorization it changes
//     nothing. With it, only the pivot signs survive (INFOG(12), INFOG(28)): working memory
//     INFOG(21) -41..-61% and 10-23% less factorization time on the real 1e5 test cases (2
//     threads; no gain on the low-fill complex embedding), and JOB=3 fails with INFO(1) = -44.
//   * JOB=-4 frees the factors and keeps the analysis; JOB=2 and JOB=3 work afterwards.
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
#include <utility>
#include <vector>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

enum class FactorPrecision { Double, Single };

/// FactorOptions::ordering value selecting hsolve's own multithreaded nested dissection
/// (compute_ordering below), handed to MUMPS as its pivot order (PERM_IN, ICNTL(7) = 1).
inline constexpr int kOrderingNestedDissection = 10;

struct FactorOptions {
  FactorPrecision precision = FactorPrecision::Double;  ///< Single: s/cmumps (pair with SolveRefinement)
  int ordering = 7;           ///< ICNTL(7): 7 auto, 5 METIS, 3 SCOTCH, 4 PORD, 0 AMD, 6 QAMD,
                              ///< 2 AMF; or kOrderingNestedDissection (hsolve, multithreaded)
  int ordering_threads = 0;   ///< threads of compute_ordering for kOrderingNestedDissection;
                              ///< 0 = `threads` (the ordering itself does not depend on it)
  int block_analysis = 0;     ///< complex T with a MUMPS ordering: declare the embedding's pairs
                              ///< {i, n+i} as blocks (ICNTL(15) = 1), so that MUMPS orders the
                              ///< n-vertex graph of K. 1 on, 0 off, -1 auto (on when at least
                              ///< half of the off-diagonal couplings are complex). Measured: see
                              ///< the block-analysis note in the header comment
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
  double ordering_s = 0.0;   ///< part of analyze_s spent in compute_ordering (hsolve orderings)
  bool block_analysis = false;  ///< the last analysis declared the embedding's 2x2 blocks
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
/// number of times for different (alpha, beta). Not thread-safe; not copyable. Distinct objects
/// may be used from different threads, but their MUMPS calls run one at a time (header comment).
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
  /// no null pivots, and its factors were not released since: valid <=> solve() works). With
  /// Single or BLR it is the inertia of the rounded / compressed matrix, exact only when no
  /// eigenvalue lies within that perturbation of the shift (compute_inertia always factorizes in
  /// double precision without BLR).
  virtual Inertia inertia() const = 0;

  /// Toggle out-of-core storage of the factors; takes effect at the next factorize().
  virtual void set_out_of_core(bool on) = 0;

  /// Frees the numerical factors and their workspace but keeps the analysis (MUMPS JOB=-4), e.g.
  /// to count inertia with an InertiaCounter without holding both in memory. Afterwards solve()
  /// throws Internal until the next successful factorize(), and inertia().valid is false (it is
  /// the "usable factorization" test of ShiftInvertOperator and the driver), while inertia()'s
  /// counts and stats() keep describing the released factorization: copy inertia() before
  /// releasing if its certified counts are still needed. Returns true if memory was released
  /// (false: nothing to release - inertia() unchanged - or a backend without this feature).
  virtual bool release_factors() { return false; }

  /// Toggle null-pivot detection (ICNTL(24)/CNTL(3), default on); takes effect at the next
  /// factorize(). Off is for shifts inside an eigenvalue cluster denser than
  /// null_pivot_tol*||K||, where every nearby shift looks "singular": MUMPS then keeps the tiny
  /// (nonzero) pivots, and shift-invert stays correct - the near-null directions are the wanted
  /// eigenvectors and get amplified (inverse iteration). factorize() still returns false for an
  /// exactly zero pivot (INFO -10). The inertia of such a factorization is NOT certified near the
  /// shift: inertia counts for certificates must come from factorizations with detection on.
  virtual void set_null_pivot_detection(bool on) = 0;

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
/// small gaps), and discards the factors (ICNTL(31): working memory only). Never fails on
/// singularity: returns valid == false instead.
/// For the B-definiteness test and unit tests; certificate endpoints should use an
/// InertiaCounter on the main factorization (no new ordering). B is HPD iff
/// valid && neg == 0 && zero == 0.
template <class T>
Inertia compute_inertia(const ShiftedPencil<T>& pencil, double alpha, double beta,
                        const FactorOptions& options = {});

struct InertiaCounterStats {
  index_t analyses = 0;        ///< analyses of the counter's own MUMPS instances
  double analyze_s = 0.0;      ///< their accumulated time
  index_t counts = 0;          ///< factorizations (all precisions)
  index_t single_counts = 0;   ///< ... of which in single precision
  double count_s = 0.0;        ///< accumulated time of the counts (assembly + factorization)
  double max_mb_used = 0.0;    ///< largest INFOG(21) of a count (working memory; no factors kept)
  index_t workspace_retries = 0;   ///< automatic ICNTL(14) increases (as in FactorStats)
  bool reused_pivot_order = false; ///< the analyses took the source's pivot order (PERM_IN)
};

/// Inertia ("Sylvester") counts nu(alpha*A + beta*B) on the pattern of an analysed
/// SparseFactorization, without touching it: the source's factors stay valid (no refactorization
/// at the working shift after a certificate), and only the pivot signs are kept.
///
/// Mechanism: one inertia-only MUMPS instance (ICNTL(31)=1) analysed with the source's pivot
/// order (PERM_IN), i.e. without a second fill-reducing ordering. Measured with
/// bench/certpar_inertia.cpp --exp counter (2 threads, the main factors resident, driver path =
/// refactorize the main object at both endpoints; medians of 5 interleaved repetitions):
///   case (ordering)                       driver pair   counter: analysis + pair    single pair
///   syn-mem27-46 (SCOTCH)                   3.85 s       0.06 + 3.27 s  (-13%)      1.82 s
///   yee-pec-cube-32 (SCOTCH)                3.00 s       0.05 + 2.59 s  (-12%)      1.40 s
///   ss-qa8fk-qa8fm (SCOTCH, GHEP)           1.57 s       0.04 + 1.30 s  (-15%)      0.74 s
///   wg-square-1e5 (AMD, 2-D)                0.31 s       0.04 + 0.26 s  ( -2%)      0.22 s
///   syn-near-singular-complex-1e5 (AMD)     1.03 s       0.48 + 0.98 s  (+42%)      0.90 s (wrong)
/// The counter's analysis costs about what an AMD analysis of the same input costs (it reads and
/// processes the whole matrix; the given order only saves the ordering itself), so it pays off
/// when the factorizations dominate the input size - nested-dissection (3-D) cases - and not on
/// low-fill inputs such as the complex embedding above (5e6 entries, no fill). Working memory
/// INFOG(21) of a count: 289 / 213 / 128 / 35 / 73 MB vs 683 / 543 / 326 / 88 / 124 MB of the
/// main factors; with the main factors resident that adds to the peak (RSS +228 / +168 / +79 /
/// +9 / +166 MB) unless the source's factors are released first (release_factors(): no growth).
/// Not done, because measured useless or impossible:
///   * concurrent counts from two threads: sequential MUMPS is not thread-safe (header comment);
///   * both endpoints in one factorization of diag(K(b), -K(a)) (two independent trees for
///     L0-OpenMP): 1-10% slower than two counts, 1.5x the memory.
///
/// Precision. Double counts are exact for shifts farther than O(eps*||K||*growth) from every
/// eigenvalue. Null-pivot detection usually flags closer shifts (valid == false), but it is the
/// same elimination-order-dependent heuristic as in SparseFactorization (header comment) and can
/// miss an exact singularity; that eigenvalue is then counted on an arbitrary side. The counter
/// eliminates in its own order (given pivot order, no compressed 2x2 ordering), so it misses at
/// other shifts than the source would: shifts exactly on 24 eigenvalues of 2-D Laplacians (AMD,
/// CNTL(3) = 1e-8; bench/certpar_fix_probes.cpp --exp nullpivot) were reported valid by the
/// counter at 1/24 (100x101, real and complex; the source flagged all 24) and by both at 1/24
/// each, at different shifts (200x201); every count reported valid lay within [#below,
/// #below + multiplicity]. valid == true therefore certifies a count only when the shift lies in
/// a spectral gap: certificate endpoints must still be placed at gap midpoints.
/// Single-precision counts (smumps; 45-53% faster than double on the 3-D cases, none on the
/// low-fill ones) are the exact counts of a matrix perturbed by O(eps_single*||K||*growth), and
/// only within the float range (alpha = 1e-50 underflows K to the zero matrix: valid == false).
/// With delta = distance from the shift to the nearest eigenvalue in units of (scale + |shift|),
/// scale = ||A||_2/||B||_2: miscounts were seen up to delta ~ 1.6e-6 (27-point stencil, HEP) and
/// ~ 5e-6 (dense complex prescribed spectrum, relative spacing 5e-6: wrong almost everywhere,
/// mostly but not always with null pivots flagged), never at delta >= 1e-5; for a GHEP the
/// perturbation grows with cond(B). The certificate endpoints of the three 3-D cases have
/// delta >= 2.8e-4. A single count is therefore a screening value only, to be trusted when the
/// caller knows every eigenvalue within ~1e-4*cond(B)*(scale + |shift|) of the shift (see the
/// driver recommendation in the certpar report), and to be confirmed in double otherwise.
///
/// Not thread-safe (one caller at a time); MUMPS calls serialize with all other MUMPS objects.
/// Scalar set: T in {double, cplx} (complex: counts of the real embedding, halved, as in
/// SparseFactorization::inertia()).
template <class T>
class InertiaCounter {
 public:
  /// `analysed_source` must come from make_mumps_factorization<T> and have been analysed; its
  /// pencil must outlive the counter (the source itself need not). Used fields of `options`:
  /// threads (0: omp_get_max_threads() now; fixed for the counter's life because L0-OpenMP
  /// factorizations must use the thread count of their analysis), scaling, mem_relax_pct,
  /// max_memory_mb (cap of each count), null_pivot_tol, single_null_pivot_tol, deterministic,
  /// verbosity, and ordering only if the source has no pivot order. Counts are always in-core
  /// and without BLR. Analyses the double-precision instance right away (estimate() is
  /// available). Throws InvalidInput (no MUMPS source / not analysed) or as analyze().
  InertiaCounter(const SparseFactorization<T>& analysed_source, const FactorOptions& options);
  ~InertiaCounter();
  InertiaCounter(const InertiaCounter&) = delete;
  InertiaCounter& operator=(const InertiaCounter&) = delete;

  /// Inertia of K = alpha*A + beta*B, valid == false when K is (numerically) singular (as
  /// SparseFactorization::factorize()). null_pivot_detection == false counts without
  /// ICNTL(24) (for shifts inside clusters denser than the detection threshold; see
  /// set_null_pivot_detection()). FactorPrecision::Single factorizes with smumps (own
  /// inertia-only instance, analysed with the same pivot order at the first single count); see
  /// the class comment for when such a count can be trusted. Throws InvalidInput (non-finite
  /// alpha/beta) or as factorize() for MUMPS errors (memory cap, workspace exhausted).
  Inertia count(double alpha, double beta, bool null_pivot_detection = true,
                FactorPrecision precision = FactorPrecision::Double);

  /// The two counts of a certificate window, one after the other (see the class comment for why
  /// not concurrently).
  std::pair<Inertia, Inertia> count_pair(double alpha1, double beta1, double alpha2, double beta2,
                                         bool null_pivot_detection = true,
                                         FactorPrecision precision = FactorPrecision::Double);

  /// MUMPS estimate of a double count (INFOG(16) etc. of the inertia-only analysis: incore_mb
  /// is the working memory, there are no factors to keep).
  FactorEstimate estimate() const;
  const InertiaCounterStats& stats() const;
  /// Order n of K.
  index_t order() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ------------------------------------------------------------------ hsolve's own orderings

/// Options of compute_ordering (nested dissection). The defaults follow METIS_NodeND.
struct OrderingOptions {
  int threads = 0;              ///< OpenMP threads; 0 = omp_get_max_threads(). The ordering does
                                ///< NOT depend on it (bitwise the same for every thread count)
  index_t leaf_size = 120;      ///< subgraphs of at most this many vertices: minimum degree
  double imbalance = 1.2;       ///< each half of a separator weighs at most imbalance * |V| / 2
  index_t coarsen_to = 100;     ///< multilevel separators coarsen down to this many vertices
  int init_tries = 7;           ///< greedy-growing attempts for the coarsest separator
  int refine_passes = 10;       ///< node-FM refinement passes per level (stop early without gain)
  std::uint64_t seed = 2026;    ///< seed of the (counter-based, thread-independent) random choices
  /// Complex T only. Compressed: order the n-vertex graph of K, then eliminate the two embedding
  /// variables of every vertex back to back (half the graph; right when most couplings are
  /// complex). Full: order the 2n-vertex graph of the embedding itself, which can exploit
  /// couplings without imaginary part (if Im K vanishes on most of them, the embedding nearly
  /// decouples into two copies of Re K: measured 2.95e12 vs 2.24e12 flops for METIS on the 2n
  /// graph on a Bloch-periodic Yee cell, n = 1e5). Auto: Compressed when at least half of the
  /// off-diagonal couplings have an imaginary part, else Full.
  enum class Embedding { Auto, Compressed, Full };
  Embedding embedding = Embedding::Auto;
};

struct OrderingStats {
  double seconds = 0.0;         ///< wall time of compute_ordering (graph construction included)
  int threads = 0;              ///< OpenMP threads used
  index_t vertices = 0;         ///< vertices of the ordered graph (n; 2n for a Full embedding)
  double complex_fraction = 0;  ///< complex T: share of off-diagonal couplings with Im != 0
  bool compressed = false;      ///< complex T: the n-vertex graph of K was ordered
  std::int64_t edges = 0;       ///< its undirected edges
  index_t separators = 0;       ///< separators computed (nonleaf subgraphs)
  index_t leaves = 0;           ///< subgraphs ordered by minimum degree
  index_t top_separator = 0;    ///< vertices of the first separator
};

/// Fill-reducing pivot order of the matrix MUMPS factorizes for pencil (K for real T, its real
/// embedding E of order 2n for complex T) by nested dissection: multilevel vertex separators
/// (sorted heavy-edge matching, greedy growing, one-sided node FM refinement) applied
/// recursively, the two halves of every separator ordered concurrently as OpenMP tasks, leaves by
/// minimum degree. Complex T: either the n-vertex graph of K is ordered and expanded so that the
/// two embedding variables i and n+i of every vertex are eliminated one after the other, or the
/// embedding's own 2n-vertex graph (couplings whose real or imaginary part is zero in A and B
/// left out, as in the factorization) is ordered - see OrderingOptions::embedding.
/// Returns PERM_IN for ICNTL(7) = 1: entry v is the 1-based pivot position of variable v (size n,
/// or 2n). Deterministic: a function of the pencil's pattern and `options` (threads excluded).
/// Thread-safe (no shared state; uses its own OpenMP parallel region). O(nnz log n) work.
/// Throws InvalidInput (bad options), Unsupported (order above the 32-bit MUMPS range).
/// Scalar set: T in {double, cplx}.
template <class T>
std::vector<int> compute_ordering(const ShiftedPencil<T>& pencil, const OrderingOptions& options = {},
                                  OrderingStats* stats = nullptr);

}  // namespace hs
