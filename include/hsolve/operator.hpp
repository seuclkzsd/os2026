// Linear operators seen by the eigensolvers. Eigensolvers only touch matrices through this
// interface, so the same Krylov code runs on A, on (A - sigma B)^{-1} B, or on a polynomial filter.
#pragma once

#include "hsolve/factor.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

template <class T>
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  /// Order n of the (square) operator.
  virtual index_t size() const = 0;

  /// Y = Op * X for p column-major vectors (X: n x p with ld ldx; Y: n x p with ld ldy).
  /// X and Y must not overlap. Counts p applications.
  void apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
    applications_ += p;
    do_apply(x, ldx, y, ldy, p);
  }

  /// Total number of single-vector applications so far.
  index_t applications() const noexcept { return applications_; }

 protected:
  virtual void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) = 0;

 private:
  index_t applications_ = 0;
};

/// Op = A (sparse matrix-vector product). The matrix must outlive the operator.
template <class T>
class CsrOperator final : public LinearOperator<T> {
 public:
  explicit CsrOperator(const CsrMatrix<T>& a);
  index_t size() const override { return a_->nrows; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  const CsrMatrix<T>* a_;
};

/// Op = K^{-1} * B where K = A - sigma*B is held factorized by `factor`; B = I when b == nullptr.
/// Self-adjoint in the B inner product when K is Hermitian and B is Hermitian positive definite.
/// The factorization and B must outlive the operator.
template <class T>
class ShiftInvertOperator final : public LinearOperator<T> {
 public:
  ShiftInvertOperator(SparseFactorization<T>& factor, const CsrMatrix<T>* b);
  index_t size() const override { return factor_->order(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  SparseFactorization<T>* factor_;
  const CsrMatrix<T>* b_;
};

}  // namespace hs
