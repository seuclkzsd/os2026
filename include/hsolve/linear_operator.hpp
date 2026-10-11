// Abstract linear operator: the only way eigensolvers touch matrices. Kept free of factor/sparse
// dependencies so that src/eigen depends on nothing but this header and src/linalg.
// Scalar set: T in {double, cplx}.
#pragma once

#include "hsolve/types.hpp"

namespace hs {

template <class T>
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  /// Order n of the (square) operator.
  virtual index_t size() const = 0;

  /// Y = Op * X for p column-major vectors (X: n x p with ld ldx >= n; Y: n x p with ld ldy >= n).
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

}  // namespace hs
