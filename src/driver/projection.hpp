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

/// Rayleigh-Ritz for a Hermitian-definite pencil restricted to span(X): solves
/// (X^H A X) y = mu (X^H B X) y with hegv. On success X, AX, BX are replaced by X*Y, AX*Y, BX*Y
/// (B-orthonormal Ritz vectors) and lambda holds mu ascending. Returns false (inputs untouched)
/// if X^H B X is not positive definite (degenerate subspace).
template <class T>
bool rayleigh_ritz_definite(DenseMatrix<T>& x, DenseMatrix<T>& ax, DenseMatrix<T>& bx,
                            std::vector<double>& lambda);

/// Rayleigh-Ritz for an indefinite Hermitian pencil restricted to span(W) (W Euclidean
/// orthonormal, AW = A W, BW = B W): eigenpairs of the small pencil (W^H A W, W^H B W) computed as
/// eig(H^{-1} G) via LU + Schur. lambda/x receive m complex eigenvalues and unit-2-norm vectors
/// X = W Y. Returns false if H is (numerically) singular, i.e. rcond-like pivot ratio < 1e-12.
template <class T>
bool rayleigh_ritz_indefinite(const DenseMatrix<T>& w, const DenseMatrix<T>& aw, const DenseMatrix<T>& bw,
                              std::vector<cplx>& lambda, DenseMatrix<cplx>& x);

}  // namespace hs::driver
