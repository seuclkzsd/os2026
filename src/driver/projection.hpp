// Subspace projection, refinement and residual metrics used by the driver (driver-internal).
#pragma once

#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

/// Y = M * X for a block (M == nullptr means the identity: Y = X).
template <class T>
void apply_block(const CsrMatrix<T>* m, const DenseMatrix<T>& x, DenseMatrix<T>& y);

/// Per-pair residual metrics (docs/DESIGN.md §4.5).
struct PairMetrics {
  double abs_residual = 0.0;    ///< ||A x - lambda B x||_2
  double rel_residual = 0.0;    ///< abs / (|lambda| ||B x||_2), or abs / ||B x||_2 when lambda == 0
  double backward_error = 0.0;  ///< abs / ((||A||_2 + |lambda| ||B||_2) ||x||_2)
};

/// Metrics of column j of (AX, BX, X) with eigenvalue lambda (real or complex T/lambda mixes).
template <class T, class L>
PairMetrics pair_metrics(const T* ax, const T* bx, const T* x, index_t n, L lambda, double norm2_a,
                         double norm2_b);

/// Euclidean orthonormalization of the columns of W in place (CGS2); columns whose norm drops
/// below drop_tol * (original norm) are removed. Returns the number of kept columns (W resized).
template <class T>
index_t orthonormalize_columns(DenseMatrix<T>& w, double drop_tol = 1e-10);

/// Rayleigh-Ritz for a Hermitian-definite pencil restricted to span(X), rank-revealing: blocks
/// may hold dependent columns (the same eigenvector computed twice, columns collapsed by residual
/// correction). The B-Gram matrix X^H B X = V diag(mu) V^H is eigendecomposed, directions with
/// mu <= drop_tol * max(mu) are dropped, and (X^H A X) y = theta (X^H B X) y is solved on the
/// B-orthonormal basis X V_r diag(mu_r)^{-1/2}. X, AX, BX are replaced by the r <= m B-orthonormal
/// Ritz vectors (and their products), lambda by the r Ritz values ascending; returns r. With a
/// well-conditioned X^H B X this is the plain Rayleigh-Ritz (r = m). Cost O(n m^2 + m^3).
template <class T>
index_t rayleigh_ritz_definite(DenseMatrix<T>& x, DenseMatrix<T>& ax, DenseMatrix<T>& bx,
                                  std::vector<double>& lambda, double drop_tol);

}  // namespace hs::driver
