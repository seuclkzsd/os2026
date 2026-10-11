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
  std::vector<double> estimate;  ///< residual estimates ||Op x - theta x|| (inner-product norm);
                                 ///< with deflation the span(D) component is excluded (see
                                 ///< lanczos_krylov_schur)
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
/// Deflation: the basis stays M-orthogonal to span(D), so the pairs are those of the deflated
/// operator P Op P (P = I - D D^H M) and estimate[i] measures only ||P (Op x - theta x)||_M. The
/// span(D) component, D^H M (Op x - theta x) = R_D^H M x with R_D the residuals of D's columns,
/// is not included: it vanishes when span(D) is invariant and is bounded by ||R_D|| otherwise.
/// Pass a D accurate to the wanted tolerance, or Rayleigh-Ritz over [D X] / verify the true
/// residuals when the full residual matters.
/// LargestMagnitude ranks equal |theta| with the positive value first.
/// Cost per restart: O(ncv) operator applications + O(n ncv^2) flops (+ 3 M applications per
/// step without store_bv, 1 with it). Memory (scalars): n*(d + ncv + 1) for the basis [D V]
/// (twice that with store_bv, which also keeps M*[D V]) + n (M*w workspace without store_bv)
/// + O(d^2 + ncv*(ncv + 4096)); the returned vectors add n*nev.
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
  DenseMatrix<T> schur_form;     ///< m x m S, quasi-upper-triangular for real T, no split 2x2 block;
                                 ///< spec(S(0:nconv, 0:nconv)) = theta[0, nconv) as a set (the
                                 ///< diagonal order need not follow theta)
  std::vector<double> estimate;  ///< eigenvector residual estimates ||Op x - theta x||_2
  std::vector<cplx> ritz_all;    ///< all Ritz values of the final basis, in `which` ranking order
  index_t nconv = 0;             ///< never splits a conjugate pair
  index_t restarts = 0;
  index_t op_applies = 0;
  bool converged = false;
  std::vector<RestartRecord> history;
};

/// Krylov-Schur for a general (non-self-adjoint) operator in the Euclidean inner product.
/// Real T works in real arithmetic with a real Schur form (dgees/dtrsen); complex-conjugate
/// Ritz pairs are never split. Used for GHIEP (indefinite B). `deflation` is not supported.
/// Options: ncv must be >= min(n, nev + 2) for real T (room for a straddling pair plus one new
/// vector), >= min(n, nev + 1) for complex T; larger than n is clamped to n (exact basis).
/// tol >= 0 and max_restarts >= 0; violations throw InvalidInput. A Ritz value with
/// |theta| <= eps * max|Ritz value| counts as theta == 0 in the convergence test. history holds
/// one record per Rayleigh-Ritz step (restarts + 1 records). Locking is soft (no deflation of b).
/// A single-vector Krylov space sees further copies of an exactly multiple eigenvalue only through
/// rounding, so the result may converge before all copies of such a cluster are found.
/// Cost per restart: about ncv/2 operator applications, O(n ncv^2) flops for CGS2 and the basis
/// update, O(ncv^3) dense work. Memory: n (ncv + 1) scalars plus the n x m outputs.
/// Not thread-safe for a shared `op`; otherwise reentrant.
template <class T>
GeneralEigResult<T> krylov_schur_general(LinearOperator<T>& op, Which which,
                                         const KrylovSchurOptions<T>& options);

}  // namespace hs
