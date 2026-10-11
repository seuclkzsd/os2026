// Top-level driver: problem diagnosis, strategy selection, shift management, inertia
// certificate, refinement, residual verification and reporting (docs/DESIGN.md §3.6).
#pragma once

#include <cstdint>
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
  double zero_filter_tol = -1.0;        ///< < 0 disables zero-mode filtering
  double complex_tol = 1e-8;            ///< |Im lambda| <= complex_tol*|lambda| => real
  index_t max_shift_retries = 5;
};

/// Per-eigenpair verification record.
struct PairReport {
  double lambda_re = 0.0;
  double lambda_im = 0.0;
  double abs_residual = 0.0;    ///< ||A x - lambda B x||_2
  double rel_residual = 0.0;    ///< abs_residual / (|lambda| ||B x||_2)
  double backward_error = 0.0;  ///< abs_residual / ((||A||_1 + |lambda| ||B||_1) ||x||_2)
  bool accepted = false;        ///< backward_error <= tol
};

struct InertiaCertificate {
  bool performed = false;
  double lower = 0.0;          ///< a
  double upper = 0.0;          ///< b
  index_t count_inertia = -1;  ///< nu(A - bB) - nu(A - aB)
  index_t count_found = -1;    ///< computed eigenvalues in [a, b]
  bool passed = false;
};

struct SolveReport {
  // problem
  index_t n = 0;
  index_t nnz_a = 0;
  index_t nnz_b = 0;
  bool complex_scalars = false;
  ProblemType problem = ProblemType::Auto;  ///< resolved type
  double norm1_a = 0.0;
  double norm1_b = 0.0;
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

  /// Serialize to a JSON document (stable keys, documented in docs/REPORT_FORMAT.md).
  std::string to_json(int indent = 2) const;
};

template <class T>
struct Solution {
  /// Real eigenvalues, ordered by the target (nearest first / ascending for interval).
  std::vector<double> eigenvalues;
  /// n x m eigenvectors matching `eigenvalues` (B-orthonormal for HEP/GHEP).
  DenseMatrix<T> eigenvectors;
  /// Filled only when a GHIEP yields genuinely complex eigenvalues; then the real fields are
  /// empty and these hold all returned pairs.
  std::vector<cplx> eigenvalues_complex;
  DenseMatrix<cplx> eigenvectors_complex;
  SolveReport report;
};

/// Solve A x = lambda B x (B == nullptr -> B = I) according to cfg. A and B must be square,
/// Hermitian, full storage, same order. Throws hs::Error on unrecoverable failure; a run that
/// completes but misses the tolerance returns with report.success == false.
template <class T>
Solution<T> solve(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const SolverConfig& cfg,
                  Logger* log = nullptr);

}  // namespace hs
