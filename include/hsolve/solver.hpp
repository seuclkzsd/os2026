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
                                    ///< GHEP, incl. slices; KrylovSchurOptions::block_size): 0 = auto
                                    ///< (src/driver/internal.hpp block_size_for: p = 4 for runs of
                                    ///< >= 8 wanted pairs, else 1; depends on nev only), 1 = single
                                    ///< vector. A deflated certificate rerun missing m pairs uses
                                    ///< max(p, min(4, m rounded up to even)) (rerun_block_size), so
                                    ///< it gets all missing copies of a multiple eigenvalue by
                                    ///< construction only up to that width. p is halved (warning)
                                    ///< while the block basis does not fit the memory budget.
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
  double memory_budget_gb = 14.0;   ///< refuse factorizations predicted to exceed this
  FactorPrecision factor_precision = FactorPrecision::Double;
  int ordering = -1;                ///< -1: adaptive AMD / nested dissection by cost model (DESIGN
                                    ///< §3.6-3b; nested dissection = SCOTCH, METIS if deterministic);
                                    ///< >= 0: MUMPS ICNTL(7) (7 auto, 5 METIS, 3 SCOTCH, 0 AMD, ...)
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
  double zero_filter_tol = -1.0;  ///< < 0 disables; else |lambda| <= zero_filter_tol*scale is a
                                  ///< (non-physical) zero mode, avoided by shift placement (§3.6)
  double complex_tol = 1e-8;      ///< GHIEP: a pair is complex only if |Im lambda| >
                                  ///< complex_tol*max(|lambda|, scale) AND its real projection fails
                                  ///< the backward-error test while the complex pair passes
  index_t max_shift_retries = 5;
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
struct InertiaCertificate {
  bool performed = false;
  double lower = 0.0;          ///< a (-inf for Smallest)
  double upper = 0.0;          ///< b (+inf for Largest)
  index_t count_inertia = -1;  ///< eigenvalues required in the window according to Sylvester inertia
  index_t count_found = -1;    ///< distinct (B-orthogonal) computed eigenvectors in the window
  index_t endpoint_moves = 0;  ///< endpoints moved because inertia came back invalid
  bool passed = false;         ///< count_inertia == count_found
  std::string note;            ///< e.g. "not available for GHIEP"
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
                                    ///< InertiaCounter counts incl. its analysis)
  std::int64_t peak_rss_bytes = 0;
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
