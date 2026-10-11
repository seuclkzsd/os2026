// Concrete linear operators (module src/ops). Scalar set: T in {double, cplx}.
// Non-owning: every referenced object must outlive the operator (rvalue constructors deleted).
#pragma once

#include <vector>

#include "hsolve/factor.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Op = A (sparse matrix-vector product via spmm). A must be square (InvalidInput otherwise).
template <class T>
class CsrOperator final : public LinearOperator<T> {
 public:
  explicit CsrOperator(const CsrMatrix<T>& a);
  explicit CsrOperator(CsrMatrix<T>&&) = delete;
  index_t size() const override { return a_->nrows; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  const CsrMatrix<T>* a_;
};

/// Iterative-refinement policy for shift-invert solves (docs/DESIGN.md §3.4b).
/// Refinement stops after max_steps corrections, when ||r_j|| <= rel_tol * ||(B x)_j|| holds for
/// every column j, or when a correction fails to reduce max_j ||r_j|| / ||(B x)_j|| (stagnation
/// at the attainable accuracy, or divergence when eps_factor * cond(K) >= 1: further solves
/// cannot help).
struct SolveRefinement {
  int max_steps = 0;     ///< 0 = off. Needed with single-precision or BLR factorizations.
  double rel_tol = 0.0;  ///< stop when ||r_j|| <= rel_tol * ||(B x)_j|| for every column j
};

/// Op = K^{-1} * B with K = A - sigma*B (B = I when the pencil has no B), where `factor` holds
/// the factorization of K computed from `pencil` at exactly this sigma (alpha = 1, beta = -sigma).
/// Self-adjoint in the B inner product when B is Hermitian positive definite.
///
/// With refinement enabled each block solve is corrected in double precision against the exact
/// K (never formed: r = Bx - (A y - sigma B y) via two spmm), so a single-precision / BLR factor
/// still yields Op to working accuracy. Extra workspace: 2*n*p scalars, kept between applies
/// (sized by the largest block applied so far).
///
/// Construction requires a usable factorization of matching order (factor.inertia().valid, i.e.
/// the last factorize() succeeded), else InvalidInput. The operator records the factorization
/// count and asserts it is unchanged on every apply (refactorizing the same object at another
/// shift would silently change Op): violating this throws hs::Error(Internal).
/// Not thread-safe (the factor and the workspace are shared state).
template <class T>
class ShiftInvertOperator final : public LinearOperator<T> {
 public:
  ShiftInvertOperator(SparseFactorization<T>& factor, const ShiftedPencil<T>& pencil, double sigma,
                      SolveRefinement refinement = {});
  ShiftInvertOperator(SparseFactorization<T>&, ShiftedPencil<T>&&, double, SolveRefinement = {}) = delete;

  index_t size() const override { return factor_->order(); }
  double sigma() const noexcept { return sigma_; }
  /// Total refinement corrections applied so far (each = one extra block solve).
  index_t refinement_steps() const noexcept { return refinement_steps_; }
  /// max_j ||r_j|| / ||(B x)_j|| of the block returned by the last apply, r = Bx - K y computed
  /// in double precision (0 if refinement is off).
  double last_relative_residual() const noexcept { return last_rel_residual_; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  SparseFactorization<T>* factor_;
  const ShiftedPencil<T>* pencil_;
  double sigma_;
  SolveRefinement refinement_;
  index_t factorization_id_;
  index_t refinement_steps_ = 0;
  double last_rel_residual_ = 0.0;
  std::vector<T> bx_;        ///< refinement workspace: B X (n x p)
  std::vector<T> residual_;  ///< refinement workspace: residual / correction (n x p)
};

}  // namespace hs
