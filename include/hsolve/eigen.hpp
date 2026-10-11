// Krylov eigensolvers operating on a LinearOperator.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Which part of the operator's spectrum is wanted.
enum class Which {
  LargestMagnitude,   ///< largest |theta| (shift-invert)
  LargestAlgebraic,   ///< largest real part
  SmallestAlgebraic,  ///< smallest real part
};

template <class T>
struct KrylovSchurOptions {
  index_t nev = 6;           ///< number of wanted eigenpairs
  index_t ncv = 0;           ///< basis size; 0 = auto: min(n_free, max(2*nev + 1, nev + 16))
  index_t max_restarts = 1000;
  double tol = 1e-12;        ///< converged when residual estimate <= tol * |theta|
  std::uint64_t seed = 2026; ///< seed of the deterministic random start vector
  /// Optional start vector (column 0 is used; must be nonzero). Overrides the random start.
  const DenseMatrix<T>* initial = nullptr;
  /// Optional deflation block D (n x d): the Krylov basis is kept orthogonal to span(D) in the
  /// solver's inner product. D must be orthonormal in that inner product.
  const DenseMatrix<T>* deflation = nullptr;
  /// B * D, required together with `deflation` when a B inner product is used.
  const DenseMatrix<T>* deflation_b = nullptr;
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
  std::vector<double> theta;     ///< nev Ritz values, most wanted first (converged ones first)
  DenseMatrix<T> vectors;        ///< n x nev Ritz vectors, orthonormal in the solver inner product
  std::vector<double> estimate;  ///< residual estimates ||Op x - theta x|| (inner-product norm)
  std::vector<double> ritz_all;  ///< all Ritz values of the final basis (gap hints for the driver)
  index_t nconv = 0;
  index_t restarts = 0;
  index_t op_applies = 0;
  bool converged = false;        ///< nconv >= nev
  std::vector<RestartRecord> history;
};

/// Thick-restart Lanczos == Hermitian Krylov-Schur with full (CGS2) reorthogonalization and
/// locking. Requires Op to be self-adjoint in the inner product <x,y> = y^H * Bip * x, where
/// Bip = b_inner (Hermitian positive definite) or the identity if b_inner == nullptr.
/// Cost per restart: O(ncv) operator applications + O(n * ncv^2) flops; memory: n*ncv scalars
/// (+ n*ncv more for Bip*V when b_inner != nullptr).
template <class T>
HermitianEigResult<T> lanczos_krylov_schur(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                           Which which, const KrylovSchurOptions<T>& options);

template <class T>
struct GeneralEigResult {
  std::vector<cplx> theta;       ///< nev Ritz values, most wanted first (converged ones first)
  DenseMatrix<cplx> vectors;     ///< n x nev unit-2-norm eigenvector approximations
  DenseMatrix<T> schur_basis;    ///< orthonormal Q (n x nev) with Op*Q ~= Q*S
  DenseMatrix<T> schur_form;     ///< S (nev x nev); quasi-upper-triangular for real T
  std::vector<double> estimate;  ///< residual estimates ||Op x - theta x||_2
  std::vector<cplx> ritz_all;
  index_t nconv = 0;
  index_t restarts = 0;
  index_t op_applies = 0;
  bool converged = false;
  std::vector<RestartRecord> history;
};

/// Krylov-Schur for a general (non-self-adjoint) operator in the Euclidean inner product.
/// Real T works in real arithmetic with a real Schur form; complex-conjugate Ritz pairs are
/// never split when truncating. Used for GHIEP (indefinite B).
template <class T>
GeneralEigResult<T> krylov_schur_general(LinearOperator<T>& op, Which which,
                                         const KrylovSchurOptions<T>& options);

}  // namespace hs
