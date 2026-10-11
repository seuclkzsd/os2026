// Shared test fixtures (library hsolve_test_support). Dense reference operators and matrices with
// known spectra, so module tests do not each re-implement them. Test-only code: densification is
// allowed here for n <= 4000 (docs/DESIGN.md §5). Scalar set: T in {double, cplx}.
//
// Ownership: the lead owns this file; agents may APPEND new fixtures (never change existing
// signatures) and must add tests for them in tests/cpp/test_support_fixtures.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::test {

/// Dense copy of a sparse matrix (verification only; throws for n > 4000).
template <class T>
DenseMatrix<T> to_dense(const CsrMatrix<T>& a);

/// CSR from a dense matrix, dropping entries with |a_ij| <= drop_tol.
template <class T>
CsrMatrix<T> from_dense(const DenseMatrix<T>& d, double drop_tol = 0.0);

/// Op = A for a dense matrix (owned).
template <class T>
class DenseOperator final : public LinearOperator<T> {
 public:
  explicit DenseOperator(DenseMatrix<T> a) : a_(std::move(a)) {}
  index_t size() const override { return a_.rows(); }
  const DenseMatrix<T>& matrix() const { return a_; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  DenseMatrix<T> a_;
};

/// Op = diag(d).
template <class T>
class DiagonalOperator final : public LinearOperator<T> {
 public:
  explicit DiagonalOperator(std::vector<T> d) : d_(std::move(d)) {}
  index_t size() const override { return static_cast<index_t>(d_.size()); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  std::vector<T> d_;
};

/// Op = (A - sigma B)^{-1} B via dense LU (B = I when b == nullptr). n <= 2000. Works for
/// indefinite B (general, non-self-adjoint use) as well as HPD B.
template <class T>
class DenseShiftInvertOperator final : public LinearOperator<T> {
 public:
  DenseShiftInvertOperator(const DenseMatrix<T>& a, const DenseMatrix<T>* b, double sigma);
  index_t size() const override { return lu_.rows(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  DenseMatrix<T> lu_;
  DenseMatrix<T> b_;  ///< empty when B = I
  std::vector<int> ipiv_;
};

/// 1-D Dirichlet Laplacian tridiag(-1, 2, -1) of order n.
template <class T>
CsrMatrix<T> laplacian_1d(index_t n);
/// Its eigenvalues 2 - 2 cos(k pi / (n + 1)), k = 1..n, ascending.
std::vector<double> laplacian_1d_eigenvalues(index_t n);

/// 2-D 5-point Dirichlet Laplacian on an nx x ny grid (order nx*ny, unit spacing).
template <class T>
CsrMatrix<T> laplacian_2d(index_t nx, index_t ny);
/// Its eigenvalues lambda_i(nx) + lambda_j(ny), ascending (exact multiplicities when nx == ny).
std::vector<double> laplacian_2d_eigenvalues(index_t nx, index_t ny);

/// Sparse Hermitian matrix with exactly the given eigenvalues (up to rounding): block diagonal
/// Q_b diag(lambda_b) Q_b^H with random unitary blocks of size block_size, then a random
/// symmetric permutation. nnz ~ n * block_size.
template <class T>
CsrMatrix<T> prescribed_spectrum(const std::vector<double>& eigenvalues, index_t block_size,
                                 std::uint64_t seed);

/// Random sparse Hermitian matrix (indefinite in general) with ~density*n off-diagonals per row.
template <class T>
CsrMatrix<T> random_hermitian(index_t n, double density, std::uint64_t seed);

/// Random sparse Hermitian positive definite matrix (strictly diagonally dominant, diag in [1, 2]+).
template <class T>
CsrMatrix<T> random_hpd(index_t n, double density, std::uint64_t seed);

/// Dense reference eigenvalues of A (b == nullptr) or of the pencil (A, B) with B HPD, ascending.
template <class T>
std::vector<double> dense_eigenvalues(const CsrMatrix<T>& a, const CsrMatrix<T>* b);

}  // namespace hs::test
