// Sparse direct factorization of the shifted matrix K = A - sigma*B (MUMPS backend).
#pragma once

#include <memory>
#include <string>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

enum class FactorKind {
  SymmetricIndefinite,  ///< LDL^T with 2x2 pivots (MUMPS SYM=2). REAL matrices only: MUMPS
                        ///< complex SYM=2 means complex-symmetric, not Hermitian.
  General,              ///< LU (MUMPS SYM=0); works for complex Hermitian K, no inertia.
};

enum class FactorPrecision { Double, Single };

struct FactorOptions {
  FactorKind kind = FactorKind::SymmetricIndefinite;
  FactorPrecision precision = FactorPrecision::Double;
  int ordering = 7;          ///< MUMPS ICNTL(7): 7 auto, 5 METIS, 3 SCOTCH, 4 PORD, 0 AMD
  int threads = 0;           ///< OpenMP threads for MUMPS; 0 = omp_get_max_threads()
  int mem_relax_pct = 35;    ///< ICNTL(14) working-space relaxation in percent
  bool blr = false;          ///< block low-rank factorization (ICNTL(35))
  double blr_tol = 1e-10;    ///< CNTL(7) dropping tolerance when blr is on
  bool out_of_core = false;  ///< ICNTL(22)
  std::string ooc_dir;       ///< directory for out-of-core files (empty = TMPDIR)
  bool detect_null_pivots = true;  ///< ICNTL(24)
  int verbosity = 0;               ///< 0 silent; >0 forwards MUMPS diagnostics to stderr
};

/// Matrix inertia (counts of negative / zero / positive eigenvalues).
struct Inertia {
  index_t neg = -1;
  index_t zero = -1;
  index_t pos = -1;
  bool valid = false;
};

/// Predictions available right after analyze().
struct FactorEstimate {
  index_t factor_entries = 0;  ///< INFOG(3) (entries of L/U, 0 if unknown)
  double factor_mb = 0.0;      ///< INFOG(16): in-core memory (MB) for this process
  double total_mb = 0.0;       ///< INFOG(17)
  double flops = 0.0;          ///< RINFOG(1) elimination flops estimate
};

struct FactorStats {
  double analyze_s = 0.0;
  double factorize_s = 0.0;  ///< accumulated over all factorize() calls
  double solve_s = 0.0;      ///< accumulated over all solve() calls
  index_t factorizations = 0;
  index_t solves = 0;        ///< number of solve() calls
  index_t rhs_solved = 0;    ///< total right-hand sides
  FactorEstimate estimate;
  index_t factor_entries = 0;  ///< INFOG(9)/(29) after the last factorization
  double factor_mb_used = 0.0; ///< INFOG(18)/(19) after the last factorization
  index_t null_pivots = 0;     ///< INFOG(28) after the last factorization
  int last_info1 = 0;          ///< INFOG(1) of the last MUMPS call
  int last_info2 = 0;          ///< INFOG(2)
};

/// Interface over a sparse direct solver. One object handles one sparsity pattern:
/// analyze() once, then factorize() any number of times with matrices of that pattern.
/// T in {double, cplx} is the scalar type of K and of the right-hand sides; with
/// FactorPrecision::Single values are rounded to float internally.
/// Not thread-safe: one object must not be used concurrently.
template <class T>
class SparseFactorization {
 public:
  virtual ~SparseFactorization() = default;

  /// Symbolic analysis (ordering). K must be square with full (both triangles) storage;
  /// the backend extracts the triangle it needs.
  virtual void analyze(const CsrMatrix<T>& k) = 0;
  /// Valid after analyze().
  virtual FactorEstimate estimate() const = 0;
  /// Numeric factorization; k must have the pattern given to analyze(). Throws
  /// Error::Code::Singular if MUMPS reports a (numerically) singular matrix.
  virtual void factorize(const CsrMatrix<T>& k) = 0;
  /// Solve K X = B in place. b is n x nrhs column-major with leading dimension ldb >= n.
  virtual void solve(T* b, index_t nrhs, index_t ldb) = 0;
  /// Inertia from the last factorization. valid == false for FactorKind::General.
  virtual Inertia inertia() const = 0;
  virtual const FactorStats& stats() const = 0;
  virtual index_t order() const = 0;
};

/// Create a MUMPS-backed factorization. Throws Error::Code::Unsupported for
/// (complex T, FactorKind::SymmetricIndefinite).
template <class T>
std::unique_ptr<SparseFactorization<T>> make_mumps_factorization(const FactorOptions& options);

/// Inertia of a Hermitian matrix: real T uses LDL^T directly; complex T factorizes
/// real_embedding(k) and halves the counts. One full analysis + factorization per call.
template <class T>
Inertia compute_inertia(const CsrMatrix<T>& k, const FactorOptions& options);

}  // namespace hs
