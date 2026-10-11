// Internals of hs::lanczos_krylov_schur shared by the single-vector solver
// (src/eigen/lanczos_krylov_schur.cpp, which also defines them) and the block solver
// (src/eigen/block_lanczos.cpp), and the block solver's entry point. Module-internal (src/eigen).
// Scalar set: T in {double, cplx}.
//
// Everything that enforces the result contract of eigen.hpp (ranking, convergence flags, history
// records, result assembly) and the numerical thresholds live here once, so that the two solvers
// cannot diverge; lanczos_krylov_schur.cpp documents the design behind the thresholds.
#pragma once

#include <limits>
#include <vector>

#include "eigen/lanczos_detail.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"
#include "hsolve/util.hpp"

namespace hs::eigen_detail {

constexpr double kLanczosEps = std::numeric_limits<double>::epsilon();
/// beta <= kBreakdownTol * ||Op v_j||_M (block solver: per column, |R(i, i)| against the column's
/// input M-norm) is an invariant subspace to rounding ("Breakdown" in lanczos_krylov_schur.cpp).
constexpr double kBreakdownTol = 64 * kLanczosEps;
/// A converged pair is locked (coupling zeroed) once its coupling (block solver: the 2-norm of its
/// coupling column) is <= kLockTol * (smallest reference |theta| of the wanted pairs).
constexpr double kLockTol = 4 * kLanczosEps;
/// A start vector (block solver: start-block column) must keep at least this fraction of its
/// M-norm after projecting out D (and the earlier start columns).
constexpr double kStartMinRatio = 1e-12;
/// A random vector keeps most of its M-norm against an M-orthonormal basis that leaves room;
/// losing more than this fraction means that M (b_inner) is not positive definite.
constexpr double kRandomMinRatio = 1e-8;

/// Adds the lifetime of the scope to a counter.
class ScopedTimer {
 public:
  explicit ScopedTimer(double& acc) : acc_(acc) {}
  ~ScopedTimer() { acc_ += timer_.seconds(); }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  double& acc_;
  Timer timer_;
};

/// Sizes of a validated lanczos_krylov_schur call (independent of block_size).
struct LanczosSetup {
  index_t n = 0;  ///< operator order
  index_t d = 0;  ///< deflation columns (0 without a block or with an empty one)
  index_t nev = 0;
  index_t n_free = 0;  ///< n - d
  /// Single-vector basis size: auto min(n_free, max(2 nev + 1, nev + 16)), or the request (which
  /// must exceed nev) clamped to n_free. ncv == n_free means one cycle spans the free space.
  index_t ncv = 0;
};

/// Every option check of lanczos_krylov_schur except block_size: b_inner order, nev, tol,
/// max_restarts, ncv, deflation block shape, nev <= n - d, initial block shape. Throws
/// InvalidInput (messages name lanczos_krylov_schur). The numerical checks (orthonormal D, usable
/// start vector) happen in the solvers.
template <class T>
LanczosSetup validate_lanczos(const LinearOperator<T>& op, const LinearOperator<T>* b_inner,
                              const KrylovSchurOptions<T>& options);

/// Checks that the d deflation columns already copied to basis(:, 0:d) (ld n) are orthonormal in
/// the solver inner product: max |D^H M D - I| <= 1e-8 and finite, else InvalidInput. With mbasis
/// (store mode) it stores M*D in mbasis(:, 0:d); otherwise, for m != nullptr, chunks of M*D are
/// formed in `scratch` (n x scratch_cols, scratch_cols >= 1), so the check needs no extra n-sized
/// memory.
template <class T>
void check_deflation(index_t n, index_t d, const T* basis, T* mbasis, LinearOperator<T>* m,
                     T* scratch, index_t scratch_cols);

/// V(:, 0:k) <- V(:, 0:m) * S for S of size m x k (m >= k), in place (V: n x m, ld n), one block
/// of 4096 rows at a time (each row of the result depends only on the same row of V).
template <class T>
void rotate_in_place(index_t n, T* v, const DenseMatrix<T>& s);

/// Rayleigh-Ritz state of one cycle and the parts of the result contract built on it. A solver
/// fills theta (ascending, heev order), s (eigenvectors of its projected matrix, column i <->
/// theta[i]) and est (residual estimates, its own formula), then calls classify(); the rest is
/// common to both solvers. m = theta.size() everywhere below.
template <class T>
struct RitzPairs {
  std::vector<double> theta;
  DenseMatrix<T> s;
  std::vector<double> est;
  std::vector<double> ref;     ///< |theta|, or the largest |theta| when theta == 0
  std::vector<char> conv;      ///< est <= tol * ref
  std::vector<index_t> order;  ///< indices into theta by decreasing preference (`which`)

  /// ref, conv and order from theta and est. LargestMagnitude ranks equal |theta| with the
  /// positive value first, so the ranking is a strict total order. O(m log m).
  void classify(double tol, Which which);
  /// Converged pairs among the nev best ranked.
  index_t wanted_converged(index_t nev) const;
  /// History record of this Rayleigh-Ritz step: converged wanted pairs and the largest relative
  /// estimate among the unconverged wanted ones.
  RestartRecord record(index_t restart, index_t nev, index_t op_applies) const;
  /// Smallest reference |theta| among the nev wanted pairs (the locking scale).
  double min_wanted_reference(index_t nev) const;
  /// The result fields of the nev best ranked pairs, converged first, both groups in ranking
  /// order: theta, estimate, vectors = V S(:, selection) for the basis V (n x m, ld n), ritz_all,
  /// nconv, converged. restarts, op_applies and history are left to the caller. O(n m nev).
  HermitianEigResult<T> result(index_t n, index_t nev, const T* basis) const;
};

/// Basis layout of the block solver for block width p > 1 (see lanczos_krylov_schur in eigen.hpp).
/// base = max(setup.ncv rounded up to a multiple of p, round_up(nev, p) + p) is the single-vector
/// basis in whole blocks; ncv = base for a requested ncv, base + 4p (as far as ncv + p <= n_free)
/// for the automatic one. A restart keeps keep = clamp(round_down(nev + (ncv - nev) / 2, p),
/// round_up(nev, p), ncv - p) Ritz pairs, so that ncv - keep is a whole number of blocks.
/// first_probe = base: the first cycle checks convergence at block boundaries from there on.
/// feasible: the free space holds the basis plus the residual block (ncv + p <= n_free) and
/// setup.ncv < n_free; otherwise the call is solved by one single-vector cycle over the free space.
struct BlockDimensions {
  index_t ncv = 0;
  index_t keep = 0;
  index_t first_probe = 0;
  bool feasible = false;
};
BlockDimensions block_dimensions(const LanczosSetup& setup, index_t p, bool auto_ncv);

/// Block Krylov-Schur, options.block_size > 1 (lanczos_krylov_schur_timed dispatches here). Same
/// contract and timing semantics as lanczos_krylov_schur_timed.
template <class T>
HermitianEigResult<T> block_lanczos_krylov_schur_timed(LinearOperator<T>& op,
                                                       LinearOperator<T>* b_inner, Which which,
                                                       const KrylovSchurOptions<T>& options,
                                                       LanczosTimings& timings);

}  // namespace hs::eigen_detail
