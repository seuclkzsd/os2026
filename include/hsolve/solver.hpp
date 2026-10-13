// Top-level driver (module src/driver): problem diagnosis, strategy selection, shift management,
// inertia certificate, refinement, residual verification and reporting (docs/DESIGN.md §3.6).
// Scalar set: T in {double, cplx}.
//
// Definitions used throughout (docs/DESIGN.md §4.5):
//   ||A||_2, ||B||_2 : largest |Ritz value| after ~30 Lanczos steps (B = I -> 1). A Ritz value is a
//                      LOWER bound of ||.||_2, which makes eta below an UPPER bound of the true
//                      backward error (conservative acceptance). Typically within a few percent.
//   scale            : ||A||_2 / ||B||_2, the natural eigenvalue magnitude of the problem
//   backward error   : eta = ||A x - lambda B x||_2 / ((||A||_2 + |lambda| ||B||_2) ||x||_2)
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/log.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"
#include "hsolve/util.hpp"

namespace hs {

enum class TargetMode {
  Nearest,   ///< the k eigenvalues closest to sigma
  Interval,  ///< all eigenvalues in [lower, upper]       (B HPD or B = I: they need inertia counts;
  Smallest,  ///< the k algebraically smallest             indefinite B supports Nearest only)
  Largest,   ///< the k algebraically largest
};

struct Target {
  TargetMode mode = TargetMode::Nearest;
  double sigma = 0.0;
  index_t k = 6;
  double lower = 0.0;
  double upper = 0.0;
};

enum class ProblemType {
  Auto,                   ///< decided by the driver (B absent -> Standard; else test B)
  Standard,               ///< A x = lambda x
  GeneralizedDefinite,    ///< B Hermitian positive definite
  GeneralizedIndefinite,  ///< B Hermitian indefinite
};

enum class Method {
  Auto,
  ShiftInvertLanczos,       ///< (A - sigma B)^{-1} B with Hermitian Krylov-Schur (needs B HPD)
  ShiftInvertKrylovSchur,   ///< (A - sigma B)^{-1} B with general Krylov-Schur (any B)
  Lanczos,                  ///< no factorization; extreme eigenvalues of HEP only
};

enum class Toggle { Auto, On, Off };

/// SolverConfig::memory_budget_gb value that derives the budget from the machine (the default).
inline constexpr double kMemoryBudgetAuto = 0.0;

/// Completeness check of a nearest target (SolverConfig::certificate; src/driver/certificate.hpp).
enum class CertificateMode {
  Exact,       ///< double-precision Sylvester counts at both window ends (main object or counter)
  Randomized,  ///< randomized exclusion check on the factorization held at sigma (verify.hpp)
  Auto,        ///< the cheapest method that certifies, predicted from measured times
  Off,         ///< no completeness check
};

struct SolverConfig {
  Target target;
  ProblemType problem = ProblemType::Auto;
  Method method = Method::Auto;
  double tol = 1e-10;               ///< acceptance: backward error eta <= tol (DESIGN §4.5)
  double hermitian_tol = 1e-10;     ///< reject input with ||A-A^H||_F/||A||_F above this;
                                    ///< below it the driver replaces A by (A+A^H)/2
  double inner_tol_factor = 0.1;    ///< Krylov tolerance = tol * inner_tol_factor
  index_t ncv = 0;                  ///< 0 = auto
  index_t block_size = 0;           ///< Krylov block width p of the shift-invert Lanczos runs (HEP /
                                    ///< GHEP, incl. slices; KrylovSchurOptions::block_size): 0 = auto,
                                    ///< 1 = single vector. Auto (src/driver/internal.hpp
                                    ///< block_size_for): nearest targets take the blocktune rule
                                    ///< (src/eigen/block_rule.hpp recommended_block_size: p in
                                    ///< {1, 2, 4, 8} from n, nev, the entries of the shift's
                                    ///< factors, nnz(B) and the scalar type, calibrated at 16
                                    ///< threads; p >= 4 from 8 wanted pairs on; no timing enters);
                                    ///< spectrum slices (interval / smallest / largest, windowed
                                    ///< nearest) p = 4 for >= 8 wanted pairs of the slice, else 1.
                                    ///< A deflated certificate rerun missing m pairs uses
                                    ///< max(q, min(4, m rounded up to even)) (rerun_block_size),
                                    ///< q = p for a configured width, min(p, 4) for the auto one,
                                    ///< so it gets all missing copies of a multiple eigenvalue by
                                    ///< construction only up to that width; it asks for m + 2
                                    ///< pairs but ends once the m nearest have converged. The
                                    ///< memory plan reserves the basis of min(p, 4) only (auto: 4);
                                    ///< p is halved while its basis does not fit beside the
                                    ///< factors (warning, except for an auto width that stays at
                                    ///< or above the reserved one).
                                    ///< GHIEP (src/driver/indefinite.cpp): auto p = 2 for n >= 2e4
                                    ///< and nev >= 4, else 1; a configured p is capped at nev, 64
                                    ///< and n/16, probes (nev <= 2) stay single-vector, and p is
                                    ///< halved (warning) while the run's peak columns exceed 90% of
                                    ///< the budget left beside the current RSS.
  index_t slice_size = 0;           ///< spectrum slicing (interval / smallest / largest, B HPD): at
                                    ///< most this many eigenvalues per shift; 0 = auto (cost model)
  index_t max_restarts = 1000;
  std::uint64_t seed = 2026;
  int threads = 0;                  ///< 0 = OMP_NUM_THREADS / all available
  /// Memory the solve plans with, GiB (DESIGN §3.6-3): factors in core if their estimate fits,
  /// else out of core, else OutOfMemoryBudget (nothing allocated); Krylov block widths and slice
  /// sizes fit into what is left. 0 (kMemoryBudgetAuto, the default) = auto: a share of the
  /// smallest memory limit of the process - memory cgroup, HSOLVE_MEMORY_LIMIT_GB, RLIMIT_AS /
  /// RLIMIT_DATA, MemTotal (resolve_memory_budget; report.memory says which); > 0: exactly this.
  double memory_budget_gb = kMemoryBudgetAuto;
  /// Arithmetic of the shift factorizations (DESIGN §3.4b, §3.6-3; its purpose is memory, not
  /// speed). double (default): always. single: single-precision factors (0.51-0.67x the memory of
  /// double) refined in double precision in every operator application - slower (P6: 0.35-0.59x
  /// end to end). A single run is checked: every refined application must converge (to within
  /// tol * inner_tol_factor), a single factor that is singular even without null-pivot detection
  /// is not used, and the result must be accepted (every pair within tol, the certificate
  /// passed); otherwise the solve starts again with double factors on the same
  /// pivot order (report.precision: fallback and why). The same holds for blr = on. auto: double,
  /// unless its factors do not fit in core within the budget while single ones do - then single,
  /// with the same checks and fallback (measured trade-off: results/p7_membudget.md).
  FactorPrecision factor_precision = FactorPrecision::Double;
  int ordering = -1;                ///< -1: adaptive AMD / nested dissection by cost model (DESIGN
                                    ///< §3.6-3b; nested dissection = SCOTCH, METIS if deterministic);
                                    ///< 0, 2..7: MUMPS ICNTL(7) (7 auto, 5 METIS, 3 SCOTCH, 0 AMD, ...);
                                    ///< 10 (kOrderingNestedDissection, factor.hpp): hsolve's own
                                    ///< multithreaded nested dissection, given to MUMPS as its pivot
                                    ///< order (compute_ordering: the same order for every thread
                                    ///< count and run). 1 (a caller-given order) is rejected
  bool deterministic = false;       ///< true: bitwise-reproducible factorizations/solves for a fixed
                                    ///< thread count (MUMPS ICNTL(48)=0, ~1.75x slower solves; the
                                    ///< adaptive ordering then takes METIS, not SCOTCH, whose orderings
                                    ///< vary from run to run - so does an explicit ordering = 3);
                                    ///< false: L0-OpenMP (results reproducible to tolerance)
  Toggle blr = Toggle::Auto;
  Toggle out_of_core = Toggle::Auto;
  Toggle inertia_check = Toggle::Auto;  ///< Auto = On for HEP/GHEP when affordable
  Toggle inertia_counter = Toggle::Auto;  ///< Sylvester counts of the certificate on an
                                          ///< InertiaCounter (inertia-only factorizations on the
                                          ///< main analysis' pivot order) instead of the main
                                          ///< object; Auto: nearest certificates when the model
                                          ///< says it pays, and always for single / BLR factors
  /// Completeness check of NEAREST targets (HEP/GHEP; src/driver/certificate.hpp):
  ///   exact      double Sylvester counts at both ends of the certificate window (as before);
  ///   randomized the randomized exclusion check (verify.hpp) on the factorization held at sigma,
  ///              p and s from certificate_cost; exact counts when it cannot make a statement
  ///              (gap too small, plan over the memory budget). It needs a proven kappa of B:
  ///              B = I, diagonal or diagonally dominant B, or (P8) a general HPD B whose
  ///              incomplete-Cholesky bounds prove one (src/driver/norms.hpp
  ///              prove_condition_bound); where none is proven a pass would only be a SCREENING
  ///              result: the check is not run there and the exact counts certify (a warning says
  ///              so) - no screening result is ever reported as the verdict;
  ///   auto       the method certificate_cost predicts cheapest among exact counts, single-
  ///              precision counts (only where verify.hpp's rule certifies them with a 10x margin
  ///              on its assumed backward error, certificate.hpp; for a general B with a proven
  ///              lambda_min(B)) and the randomized check (only with a proven kappa), from measured
  ///              times of this run (main factorization, solve calls of the Krylov run at its block
  ///              width);
  ///   off        none (inertia_check = off also turns it off).
  /// A check that detects a missing pair triggers the deflated rerun and checks again; single
  /// counts that exceed the found pairs or contradict the exact count at sigma are checked by
  /// double counts first. Fewer than k selected pairs rerun even after a passed check (it covers
  /// the found pairs only). Window targets (interval / smallest / largest, and nearest targets
  /// handed to the windows by the zero band) always count exactly: their count points are slice
  /// cuts and shifts. The report states what was proven (method, rigorous, failure_probability,
  /// statement). A count whose factorization fails (MUMPS out of memory or another MUMPS error)
  /// moves to the other count source (main factorization or InertiaCounter); when no count can be
  /// made, a nearest target returns its converged pairs with the certificate performed but not
  /// completed (passed = false, count_inertia = -1, note "not completed: ...", success = false); a
  /// window target does so with the pairs of the slices it completed (count_inertia: the window's
  /// count) when a count fails from its first slice on, and ends with the error before (window
  /// ends and cuts: nothing solved yet).
  CertificateMode certificate = CertificateMode::Exact;
  double certificate_delta = 1e-12;  ///< randomized check: bound on the failure probability, (0, 1)
  /// FAULT INJECTION, for tests and measurements only (0 = off): after the first Krylov run of a
  /// nearest target, remove this many found pairs before the completeness check - copies of the
  /// largest multiple eigenvalue among the selected pairs (one copy always stays), then the
  /// farthest selected ones with their Ritz hints - so that the solve must find them missing (a
  /// check detecting a copy, or fewer than k selected pairs) and the deflated rerun recover them.
  index_t certificate_fault_drop = 0;
  /// FAULT INJECTION, for tests only (false = off): the result of an attempt with inexact factors
  /// (single precision, BLR) is not accepted - its first pair is marked above tol - so that the
  /// solve starts again with double factors (report.precision.fallback), as for a real miss.
  bool inexact_fault_reject = false;
  /// FAULT INJECTION, for tests and measurements only (0 = off): a sum of bits making the
  /// factorizations of the certificate fail, so that every recovery path can be exercised
  /// (src/driver/internal.hpp kFault*): 1 every InertiaCounter gets a 1 MB working-memory cap
  /// (its counts fail in MUMPS with INFO -19); 2 counts on the main factorization object fail
  /// (OutOfMemoryBudget); 4 InertiaCounters are predicted not to fit the memory left (count
  /// allowance 0); 8 the refactorization at the working shift before a deflated rerun or a
  /// randomized check fails. Kind of the failures of 1, 2, 8 (at most one): 16 another MUMPS error
  /// (FactorizationFailed; bit 1 then throws at each count instead of the cap), 32 std::bad_alloc.
  /// 64: bits 1 and 2 only from a window target's first slice on. A count that fails moves to the
  /// other count source; with none left the nearest target returns its pairs with the certificate
  /// performed but not completed, a window target the pairs of its completed slices
  /// (docs/FORMATS.md §3).
  index_t certificate_fault_count = 0;
  double zero_filter_tol = -1.0;  ///< < 0 disables; else |lambda| <= zero_filter_tol*scale is a
                                  ///< (non-physical) zero mode, avoided by shift placement (§3.6)
  double complex_tol = 1e-8;      ///< GHIEP: a pair is complex only if |Im lambda| >
                                  ///< complex_tol*max(|lambda|, scale) AND its real projection fails
                                  ///< the backward-error test while the complex pair passes
  index_t max_shift_retries = 5;
};

/// The automatic budget (resolve_memory_budget) of a run whose smallest memory limit is L GiB:
///   budget = max(kAutoBudgetFloorGb, min(kAutoBudgetFraction * L, L - kAutoBudgetHeadroomGb)).
/// The plan (DESIGN §3.6-3) keeps its predictions - the memory in use at planning time, Krylov
/// basis, MUMPS' estimate under the ICNTL(23) cap - within the budget; the rest of L covers what
/// it does not predict (calibration: results/p7_membudget.md). The budget is charged in the
/// measure of L (MemoryBudget::measure); an address-space (or VmData) limit first sets aside what
/// the process reserves there without using it, and the formula applies to the rest of L
/// (MemoryBudget::reserved_gb).
inline constexpr double kAutoBudgetFraction = 0.85;
inline constexpr double kAutoBudgetHeadroomGb = 1.0;
/// Below this no automatic budget goes, whatever the limit (tiny machines: a plan needs room).
inline constexpr double kAutoBudgetFloorGb = 0.5;
/// Budget when no limit is readable at all (no cgroup, environment, rlimit or /proc/meminfo): the
/// fixed default before P7.
inline constexpr double kAutoBudgetFallbackGb = 14.0;

/// The memory budget a solve plans with.
struct MemoryBudget {
  double gb = 0.0;  ///< GiB
  /// "config" (a configured budget); auto: the source of the limit it derives from ("cgroup",
  /// "env", "rlimit_as", "rlimit_data", "mem_total"), or "default" (no limit readable).
  std::string source;
  /// The smallest limit found (MemoryLimits::gib; for auto the L of the formula), GiB; -1: none.
  /// Also probed for a configured budget, so that one above the limit can be reported.
  double limit_gb = -1.0;
  std::string limit_source;               ///< MemoryLimits::source
  std::vector<MemoryLimitSource> limits;  ///< every source probed (detect_memory_limits)
  /// What the budget is charged with: the measure of the limit an automatic budget derives from
  /// (measure_of(source): the address space for RLIMIT_AS, VmData for RLIMIT_DATA), resident
  /// memory otherwise - a configured budget states resident memory.
  MemoryMeasure measure = MemoryMeasure::Resident;
  /// Address-space / VmData budget: what the process held in that measure beyond its resident
  /// memory when the budget was resolved, GiB - reserved, never touched: BLAS buffers, malloc
  /// arenas, thread stacks (3.4 GB at 16 threads). It does not grow with the problem, so the
  /// formula's share applies to the rest of the limit: budget = reserved + share(L - reserved).
  /// 0 for a resident budget.
  double reserved_gb = 0.0;
};

/// configured_gb > 0: that budget; kMemoryBudgetAuto: from the limits (formula above), where an
/// address-space or VmData limit first sets aside `reserved_gib` (MemoryBudget::reserved_gb).
/// Throws InvalidInput for a negative or non-finite configured_gb.
MemoryBudget resolve_memory_budget(double configured_gb, const MemoryLimits& limits,
                                   double reserved_gib = 0.0);
/// The same with the limits of this process (detect_memory_limits()) and, for an address-space
/// or VmData limit, its current reservation beyond resident memory.
MemoryBudget resolve_memory_budget(double configured_gb);

/// The memory budget of a solve, as resolved (SolveReport::memory).
struct MemoryReport {
  double budget_gb = 0.0;                 ///< MemoryBudget::gb
  std::string source;                     ///< MemoryBudget::source
  double limit_gb = -1.0;                 ///< MemoryBudget::limit_gb
  std::string limit_source;               ///< MemoryBudget::limit_source
  std::vector<MemoryLimitSource> limits;  ///< MemoryBudget::limits
  MemoryMeasure measure = MemoryMeasure::Resident;  ///< MemoryBudget::measure
  double reserved_gb = 0.0;                         ///< MemoryBudget::reserved_gb
};

/// The arithmetic of the factorizations (SolverConfig::factor_precision) and its checks
/// (SolveReport::precision).
struct PrecisionReport {
  FactorPrecision requested = FactorPrecision::Double;  ///< SolverConfig::factor_precision
  FactorPrecision used = FactorPrecision::Double;       ///< of the factors of the returned result
  bool blr = false;  ///< the returned result's factors are block low-rank
  /// auto: what the choice compared (the in-core estimates against the factorization cap);
  /// otherwise empty.
  std::string decision;
  bool fallback = false;        ///< an inexact attempt (single / BLR) was abandoned for double
  std::string fallback_reason;  ///< why (refinement that did not converge, a singular single
                                ///< factor, pairs not accepted, certificate failed)
  double attempt_s = 0.0;       ///< wall time of the abandoned attempt (inside the phase times)
};

/// Per-eigenpair verification record.
struct PairReport {
  double lambda_re = 0.0;
  double lambda_im = 0.0;
  double abs_residual = 0.0;    ///< ||A x - lambda B x||_2
  double rel_residual = 0.0;    ///< abs_residual / (|lambda| ||B x||_2)
  double backward_error = 0.0;  ///< abs_residual / ((||A||_2 + |lambda| ||B||_2) ||x||_2)
  bool accepted = false;        ///< backward_error <= tol
};

/// Completeness certificate (HEP/GHEP only; DESIGN §3.6 step 8). Windows by target mode:
/// Nearest: [a, b] with a <= sigma - r, b >= sigma + r (r = max |lambda_i - sigma|), endpoints
/// pushed into spectral gaps; Smallest: (-inf, b]; Largest: [a, +inf); Interval: user bounds.
/// Smallest / largest whose k-th eigenvalue lies in a large cluster of numerically equal
/// eigenvalues (e.g. a null space without zero_filter_tol) require only the copies still needed
/// from it: count_inertia then counts the eigenvalues before the cluster plus those copies (two
/// counts prove the cluster's width; see note), and any such copies are a valid answer.
/// Nearest targets may be checked by another method (SolverConfig::certificate): `method` says
/// which one gave the final verdict and `statement` what it proved.
struct InertiaCertificate {
  bool performed = false;
  double lower = 0.0;          ///< a (-inf for Smallest)
  double upper = 0.0;          ///< b (+inf for Largest)
  index_t count_inertia = -1;  ///< eigenvalues required in the window according to Sylvester inertia
                               ///< (-1 when the final verdict did not count: randomized, or a
                               ///< certificate not completed because no count was possible)
  index_t count_found = -1;    ///< distinct (B-orthogonal) computed eigenvectors in the window
  index_t endpoint_moves = 0;  ///< endpoints moved because inertia came back invalid
  bool passed = false;         ///< counts: count_inertia == count_found; randomized: the exclusion
                               ///< check passed (no further eigenvalue within the window radius,
                               ///< kappa of B proven). Always a proof (rigorous): screening
                               ///< results are never reported as the verdict
  std::string note;            ///< e.g. "not available for GHIEP"; "not completed: no inertia count
                               ///< possible ..." when every count source failed (P8; a window
                               ///< target names the slices it could not solve)
  /// "exact" (double Sylvester counts), "exact_single" (single-precision counts under the
  /// perturbation rule of verify.hpp), "randomized" (exclusion check), "none" (not performed).
  std::string method = "none";
  /// passed is a proof: true for counts (exact_single: under the rule's assumed backward error,
  /// stated); for the randomized check when its kappa is proven (B = I, diagonal or diagonally
  /// dominant B) or its Krylov space covered the complement. false when not passed.
  bool rigorous = false;
  /// Bound on P(passed although a pair within the claimed radius is missing): 0 for counts, the
  /// achieved delta(s) of the randomized check (<= SolverConfig::certificate_delta), 1 = no statement.
  double failure_probability = 1.0;
  index_t solves = 0;           ///< single-vector solves (operator applications) of randomized checks
  double seconds = 0.0;         ///< wall time of the completeness checks, all rounds (Phase
                                ///< certificate: part of SolveReport::time_certificate_s)
  std::string statement;        ///< what was proven, in words (auto: also the predicted costs)
};

/// One shift-invert Krylov run of a sliced or one-sided solve (DESIGN §3.6): spectrum slices of
/// interval / smallest / largest targets, and the one-sided runs that keep the GHIEP zero
/// cluster unwanted (plus GHIEP recovery runs for complex eigenvalues between them). Empty for
/// single-shift nearest solves. reruns counts deflated reruns, including those at a multiple
/// eigenvalue's own shift (sigma is the slice's first shift). A partial cluster slice (see
/// InertiaCertificate) reports the cluster's count in count_inertia and passes with the copies
/// required.
struct SliceReport {
  double lower = 0.0;          ///< slice window (-inf / +inf: one-sided)
  double upper = 0.0;
  double sigma = 0.0;          ///< shift of the run
  std::string which;           ///< "largest_magnitude" (both sides of sigma), "smallest_algebraic"
                               ///< (eigenvalues below sigma), "largest_algebraic" (above sigma)
  index_t count_inertia = -1;  ///< eigenvalues in [lower, upper] by Sylvester inertia (-1: GHIEP)
  index_t count_found = -1;    ///< distinct verified pairs found in [lower, upper]
  index_t nev = 0;             ///< pairs requested from the Krylov solver (first run)
  index_t reruns = 0;          ///< deflated / enlarged reruns
  index_t restarts = 0;
  index_t op_applies = 0;
  bool passed = false;         ///< count_found == count_inertia (GHIEP: run converged)
};

struct SolveReport {
  // problem
  index_t n = 0;
  index_t nnz_a = 0;
  index_t nnz_b = 0;
  bool complex_scalars = false;
  ProblemType problem = ProblemType::Auto;  ///< resolved type
  double norm2_a = 0.0;          ///< Lanczos lower bound of ||A||_2 (conservative eta)
  double norm2_b = 1.0;          ///< idem for B (1 when B = I)
  double scale = 0.0;            ///< norm2_a / norm2_b
  double hermitian_defect_a = 0.0;
  double hermitian_defect_b = 0.0;
  // strategy
  Method method = Method::Auto;  ///< resolved method
  double sigma_requested = 0.0;
  double sigma_used = 0.0;       ///< after near-singular perturbation
  index_t shift_retries = 0;
  FactorOptions factor_options;
  index_t block_size = 1;        ///< Krylov block width p of the first shift-invert run (resolved
                                 ///< SolverConfig::block_size)
  // statistics
  FactorStats factor_stats;
  index_t restarts = 0;
  index_t op_applies = 0;
  // Wall time of hs::solve split into exclusive phases (each instant counted once), so
  // time_total_s == time_setup_s + time_factor_s + time_iterate_s + time_verify_s
  //                 + time_certificate_s:
  double time_total_s = 0.0;
  double time_setup_s = 0.0;    ///< input checks, norm estimates, problem type, planning
  double time_factor_s = 0.0;   ///< symbolic analyses + numeric factorizations of shifts (Krylov
                                ///< runs, rescue), of B (definiteness test) and of A (GHIEP)
  double time_iterate_s = 0.0;  ///< Krylov solvers (operator applications = solves, orthogonalization)
  double time_verify_s = 0.0;   ///< Rayleigh-Ritz refinement, residuals, rescue, selection
  double time_certificate_s = 0.0;  ///< HEP/GHEP Sylvester inertia counts: certificate endpoints,
                                    ///< window ends, slice cuts (main-object factorizations or
                                    ///< InertiaCounter counts incl. its analysis), and the other
                                    ///< completeness checks of nearest targets (randomized
                                    ///< exclusion check, single counts, the auto choice)
  std::int64_t peak_rss_bytes = 0;
  MemoryReport memory;        ///< the memory budget and where it came from
  PrecisionReport precision;  ///< factor arithmetic, its checks and fallback
  // results
  std::vector<PairReport> pairs;
  InertiaCertificate certificate;
  std::vector<SliceReport> slices;  ///< per-run records of sliced / one-sided solves
  index_t zero_cluster_dim = -1;    ///< GHIEP: dim null(A) (zero pivots of A), when it was needed;
                                    ///< -1 = not computed
  index_t zero_modes_filtered = 0;  ///< eigenvalues with |lambda| <= zero_filter_tol*scale left out
                                    ///< (counted by inertia for HEP/GHEP windows)
  bool success = false;
  std::vector<std::string> warnings;

  /// Serialize to a JSON document (stable keys, schema in docs/FORMATS.md §3).
  std::string to_json(int indent = 2) const;
};

template <class T>
struct Solution {
  /// Real eigenpairs (all pairs for HEP/GHEP), ordered by the target: nearest first (Nearest),
  /// ascending (Interval, Smallest), descending (Largest).
  std::vector<double> eigenvalues;
  /// n x m eigenvectors matching `eigenvalues` (B-orthonormal for HEP/GHEP; unit 2-norm GHIEP).
  DenseMatrix<T> eigenvectors;
  /// GHIEP only: genuinely complex pairs (classified per SolverConfig::complex_tol), usually
  /// empty. Real pairs are never moved here.
  std::vector<cplx> complex_eigenvalues;
  DenseMatrix<cplx> complex_eigenvectors;
  /// report.pairs lists the real pairs first (same order), then the complex ones.
  SolveReport report;
};

/// Parse a config.json document (docs/FORMATS.md §2): keys are SolverConfig / Target field names,
/// enums are snake_case strings. Keys absent from the document keep the values of `base`;
/// unknown keys, wrong types and invalid enum strings throw InvalidInput.
SolverConfig parse_config(const std::string& json_text, const SolverConfig& base = {});
/// Serialize every field of cfg (round-trips through parse_config).
std::string config_to_json(const SolverConfig& cfg, int indent = 2);

/// Solve A x = lambda B x (B == nullptr -> B = I) according to cfg. A and B must be square,
/// full storage, same order, Hermitian within cfg.hermitian_tol (symmetrized internally).
/// Throws hs::Error on unrecoverable failure (NotHermitian, OutOfMemoryBudget, ...); a run that
/// completes but misses the tolerance returns with report.success == false.
template <class T>
Solution<T> solve(const CsrMatrix<T>& a, const non_deduced_t<CsrMatrix<T>>* b, const SolverConfig& cfg,
                  Logger* log = nullptr);

}  // namespace hs
