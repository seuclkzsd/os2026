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
  Interval,  ///< all eigenvalues in [lower, upper]
  Smallest,  ///< the k algebraically smallest
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
  index_t max_restarts = 1000;
  std::uint64_t seed = 2026;
  int threads = 0;                  ///< 0 = OMP_NUM_THREADS / all available
  double memory_budget_gb = 14.0;   ///< refuse factorizations predicted to exceed this
  FactorPrecision factor_precision = FactorPrecision::Double;
  int ordering = 7;                 ///< MUMPS ICNTL(7)
  Toggle blr = Toggle::Auto;
  Toggle out_of_core = Toggle::Auto;
  Toggle inertia_check = Toggle::Auto;  ///< Auto = On for HEP/GHEP when affordable
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
struct InertiaCertificate {
  bool performed = false;
  double lower = 0.0;          ///< a (-inf for Smallest)
  double upper = 0.0;          ///< b (+inf for Largest)
  index_t count_inertia = -1;  ///< eigenvalues in the window according to Sylvester inertia
  index_t count_found = -1;    ///< distinct (B-orthogonal) computed eigenvectors in the window
  index_t endpoint_moves = 0;  ///< endpoints moved because inertia came back invalid
  bool passed = false;         ///< count_inertia == count_found
  std::string note;            ///< e.g. "not available for GHIEP"
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
  // statistics
  FactorStats factor_stats;
  index_t restarts = 0;
  index_t op_applies = 0;
  double time_total_s = 0.0;
  double time_factor_s = 0.0;
  double time_iterate_s = 0.0;
  double time_verify_s = 0.0;
  std::int64_t peak_rss_bytes = 0;
  // results
  std::vector<PairReport> pairs;
  InertiaCertificate certificate;
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

/// Solve A x = lambda B x (B == nullptr -> B = I) according to cfg. A and B must be square,
/// full storage, same order, Hermitian within cfg.hermitian_tol (symmetrized internally).
/// Throws hs::Error on unrecoverable failure (NotHermitian, OutOfMemoryBudget, ...); a run that
/// completes but misses the tolerance returns with report.success == false.
template <class T>
Solution<T> solve(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const SolverConfig& cfg,
                  Logger* log = nullptr);

}  // namespace hs
