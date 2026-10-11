// Krylov eigensolvers operating on a LinearOperator (module src/eigen).
// Scalar set: T in {double, cplx}. Depends only on linear_operator.hpp and src/linalg.
//
// Error contract: the eigensolvers never throw on non-convergence; they return converged=false
// (Error::Code::NotConverged is reserved for the driver). They throw InvalidInput for invalid
// options (nev < 1, nev > available dimension, malformed initial/deflation blocks).
//
// Result contract (both solvers), with m = theta.size():
//   * theta, estimate, vectors (columns) refer to the same pairs index by index.
//   * Entries [0, nconv) are the converged pairs, sorted by the `which` ranking; entries
//     [nconv, m) are the unconverged ones, also in ranking order. converged == (nconv >= nev).
//   * A pair is converged when estimate[i] <= tol * |theta[i]| (|theta| replaced by the largest
//     |Ritz value| when theta[i] == 0).
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Which part of the operator's spectrum is wanted.
enum class Which {
  LargestMagnitude,   ///< largest |theta| (shift-invert)
  LargestAlgebraic,   ///< largest real part
  SmallestAlgebraic,  ///< smallest real part
};

/// Options. Pointer members are call-scoped: read during the call, never retained.
template <class T>
struct KrylovSchurOptions {
  index_t nev = 6;           ///< number of wanted eigenpairs (1 <= nev <= available dimension)
  index_t ncv = 0;           ///< basis size; 0 = auto: min(n_free, max(2*nev + 1, nev + 16))
  index_t max_restarts = 1000;
  double tol = 1e-12;        ///< relative residual-estimate tolerance (see contract above)
  std::uint64_t seed = 2026; ///< deterministic random start vector (linalg::fill_random)
  /// Optional start vector (column 0 is used; must be nonzero). Overrides the random start.
  const DenseMatrix<T>* initial = nullptr;
  /// Hermitian solver only: block D (n x d), orthonormal in the solver inner product. The Krylov
  /// basis is kept orthogonal to span(D) ("deflated restart": find further eigenpairs after the
  /// ones in D). n_free = n - d. krylov_schur_general rejects it (InvalidInput): for a non-normal
  /// operator the orthogonal complement of an invariant subspace is not invariant.
  const DenseMatrix<T>* deflation = nullptr;
  /// Hermitian solver with b_inner only: keep B*V next to V. Saves two B applications per step
  /// but doubles basis memory; default recomputes B*w on demand.
  bool store_bv = false;
};

/// One record per restart (observability / convergence analysis).
struct RestartRecord {
  index_t restart = 0;
  index_t nconv = 0;          ///< converged wanted pairs after this restart
  double max_estimate = 0.0;  ///< largest relative estimate among the wanted unconverged pairs
  index_t op_applies = 0;     ///< cumulative operator applications
};

template <class T>
struct HermitianEigResult {
  std::vector<double> theta;     ///< m == nev Ritz values (contract above)
  DenseMatrix<T> vectors;        ///< n x m Ritz vectors, orthonormal in the solver inner product
  std::vector<double> estimate;  ///< residual estimates ||Op x - theta x|| (inner-product norm)
  std::vector<double> ritz_all;  ///< all Ritz values of the final basis, ascending (gap hints)
  index_t nconv = 0;
  index_t restarts = 0;
  index_t op_applies = 0;
  bool converged = false;
  std::vector<RestartRecord> history;
};

/// Thick-restart Lanczos == Hermitian Krylov-Schur with full CGS2 reorthogonalization and locking.
/// Requires Op to be self-adjoint in <x, y> = y^H M x with M = b_inner (Hermitian positive
/// definite) or the identity when b_inner == nullptr. For shift-invert Op = (A - sigma B)^{-1} B
/// with B HPD, pass b_inner = B.
/// Cost per restart: O(ncv) operator applications + O(n ncv^2) flops (+ up to 3 M applications
/// per step unless store_bv). Memory: n*ncv scalars (+ n*ncv with store_bv) + n*d for deflation.
template <class T>
HermitianEigResult<T> lanczos_krylov_schur(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                           Which which, const KrylovSchurOptions<T>& options);

template <class T>
struct GeneralEigResult {
  /// m Ritz values, m in {nev, nev + 1}: nev + 1 only for real T when a complex-conjugate pair
  /// straddles the cut. Conjugate pairs are adjacent, positive imaginary part first.
  std::vector<cplx> theta;
  DenseMatrix<cplx> vectors;     ///< n x m unit-2-norm eigenvector approximations (one per theta)
  DenseMatrix<T> schur_basis;    ///< n x m orthonormal Q with Op*Q ~= Q*S; leading nconv columns
                                 ///< span an (approximately) Op-invariant subspace
  DenseMatrix<T> schur_form;     ///< m x m S, quasi-upper-triangular for real T, no split 2x2 block
  std::vector<double> estimate;  ///< eigenvector residual estimates ||Op x - theta x||_2
  std::vector<cplx> ritz_all;
  index_t nconv = 0;             ///< never splits a conjugate pair
  index_t restarts = 0;
  index_t op_applies = 0;
  bool converged = false;
  std::vector<RestartRecord> history;
};

/// Krylov-Schur for a general (non-self-adjoint) operator in the Euclidean inner product.
/// Real T works in real arithmetic with a real Schur form (dgees/dtrsen); complex-conjugate
/// Ritz pairs are never split. Used for GHIEP (indefinite B). `deflation` is not supported.
template <class T>
GeneralEigResult<T> krylov_schur_general(LinearOperator<T>& op, Which which,
                                         const KrylovSchurOptions<T>& options);

}  // namespace hs
