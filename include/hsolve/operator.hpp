// Concrete linear operators (module src/ops). Scalar set: T in {double, cplx}.
// Non-owning: every referenced object must outlive the operator (rvalue constructors deleted).
// Sparse products go through the kernels of sparse.hpp; `kernels` selects their KernelPath
// (Auto: AVX-512 when available; Reference reproduces the P1 operators for comparisons). Each
// operator keeps the kernels' scratch between applies (no allocation per apply after the first
// block of a given width).
#pragma once

#include <vector>

#include "hsolve/factor.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Op = A (sparse matrix-vector product via spmm). A must be square (InvalidInput otherwise).
/// Extra memory: n*min(p, 16) scalars of kernel scratch for blocks of p >= 2 columns.
/// Not thread-safe (shared scratch).
template <class T>
class CsrOperator final : public LinearOperator<T> {
 public:
  explicit CsrOperator(const CsrMatrix<T>& a, KernelPath kernels = KernelPath::Auto);
  explicit CsrOperator(CsrMatrix<T>&&, KernelPath = KernelPath::Auto) = delete;
  index_t size() const override { return a_->nrows; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  const CsrMatrix<T>* a_;
  KernelPath kernels_;
  std::vector<double> scratch_;
};

/// Iterative-refinement policy for shift-invert solves (docs/DESIGN.md §3.4b), with the residual
/// r_j = (B x)_j - K y_j of column j computed in double precision. Column j is refined until
/// ||r_j|| <= rel_tol * ||(B x)_j||, until r_j is at the rounding level of its own evaluation
/// (componentwise backward error max_i |r_ij| / (|B| |x_j + sigma y_j| + |A| |y_j|)_i <= 8 eps,
/// see shifted_residual: no correction could reduce it, so none is computed), or until a
/// correction fails to reduce ||r_j|| (stagnation at the attainable accuracy, or divergence when
/// eps_factor * cond(K) >= 1, i.e. sigma is too close to an eigenvalue for the factor's
/// precision): that correction is discarded, so every column returns its best iterate. At most
/// max_steps block corrections are computed per apply.
struct SolveRefinement {
  int max_steps = 0;     ///< 0 = off. Needed with single-precision or BLR factorizations.
  double rel_tol = 0.0;  ///< column j is done when ||r_j|| <= rel_tol * ||(B x)_j||
  bool stop_at_rounding_level = true;  ///< false: ignore the backward-error test (P1 behaviour)
};

/// Op = K^{-1} * B with K = A - sigma*B (B = I when the pencil has no B), where `factor` holds
/// the factorization of K computed from `pencil` at exactly this sigma (alpha = 1, beta = -sigma).
/// Self-adjoint in the B inner product when B is Hermitian positive definite.
///
/// One apply of p columns = B X (spmm) + one p-column block solve; the block solve dominates
/// (bench/solvephase_profile.cpp: 85-99% of an apply on the 1e5 test cases), and its cost per
/// column falls steeply with p (one sweep over the factors serves all p columns).
///
/// With refinement enabled each block solve is corrected in double precision against the exact
/// K (never formed: r = B (x + sigma y) - A y in one fused sweep over A and B, shifted_residual),
/// so a single-precision / BLR factor still yields Op to working accuracy, provided
/// eps_factor * cond(K) < 1. Otherwise the refinement stops early with a large
/// last_relative_residual(): callers using a Single / BLR factor must check it, because
/// factorize() cannot flag every shift that is too close to an eigenvalue for that precision
/// (see factor.hpp). Each correction costs one block solve and one residual sweep. Extra
/// workspace: 2*n*p scalars of iterates plus up to 2*n*min(p, 16) of kernel scratch, kept
/// between applies (sized by the largest block applied so far).
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
                      SolveRefinement refinement = {}, KernelPath kernels = KernelPath::Auto);
  ShiftInvertOperator(SparseFactorization<T>&, ShiftedPencil<T>&&, double, SolveRefinement = {},
                      KernelPath = KernelPath::Auto) = delete;

  index_t size() const override { return factor_->order(); }
  double sigma() const noexcept { return sigma_; }
  /// Refinement block corrections computed so far, over all applies (each = one extra block
  /// solve, whether or not every column accepted its correction).
  index_t refinement_steps() const noexcept { return refinement_steps_; }
  /// max_j ||r_j|| / ||(B x)_j|| of the block returned by the last apply, r = Bx - K y computed
  /// in double precision for the returned Y (0 if refinement is off).
  double last_relative_residual() const noexcept { return last_rel_residual_; }
  /// max_j componentwise backward error of the block returned by the last apply (0 if refinement
  /// is off); a few eps means Op x is as accurate as double precision allows for this K.
  double last_backward_error() const noexcept { return last_backward_error_; }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  /// residual_ = B X - K Y for p columns (leading dimension n), omega = their componentwise
  /// backward errors.
  void residual(const T* x, index_t ldx, const T* y, index_t ldy, index_t p, double* omega);

  SparseFactorization<T>* factor_;
  const ShiftedPencil<T>* pencil_;
  double sigma_;
  SolveRefinement refinement_;
  KernelPath kernels_;
  index_t factorization_id_;
  index_t refinement_steps_ = 0;
  double last_rel_residual_ = 0.0;
  double last_backward_error_ = 0.0;
  std::vector<T> residual_;      ///< refinement workspace: residuals (n x p)
  std::vector<T> candidate_;     ///< refinement workspace: corrected candidate iterates (n x p)
  std::vector<double> scratch_;  ///< sparse-kernel scratch (interleaved copies)
};

/// Op = K^{-1}: one block solve with the factor as it is (no B, no refinement), e.g. the
/// preconditioner T ~ (A - s B)^{-1} of lobpcg / jacobi_davidson (eigen_iterative.hpp), with a
/// factor of any precision. Same construction checks as ShiftInvertOperator (usable
/// factorization; refactorizing the factor afterwards makes apply throw Internal). Cost per apply:
/// a copy of X and one p-column solve. Not thread-safe (the factor is shared state).
template <class T>
class FactorSolveOperator final : public LinearOperator<T> {
 public:
  explicit FactorSolveOperator(SparseFactorization<T>& factor);

  index_t size() const override { return factor_->order(); }

 protected:
  void do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) override;

 private:
  SparseFactorization<T>* factor_;
  index_t factorization_id_;
};

}  // namespace hs
