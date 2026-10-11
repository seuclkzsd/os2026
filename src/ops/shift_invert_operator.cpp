// ShiftInvertOperator: Op = (A - sigma B)^{-1} B with optional double-precision iterative
// refinement (docs/DESIGN.md §3.4b).
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {

namespace {

// Y = B X for p columns (B = I when b == nullptr).
template <class T>
void apply_b(const CsrMatrix<T>* b, index_t n, const T* x, index_t ldx, T* y, index_t ldy,
             index_t p) {
  if (b) {
    spmm(*b, x, ldx, y, ldy, p);
    return;
  }
  for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, y + k * ldy);
}

}  // namespace

template <class T>
ShiftInvertOperator<T>::ShiftInvertOperator(SparseFactorization<T>& factor,
                                            const ShiftedPencil<T>& pencil, double sigma,
                                            SolveRefinement refinement)
    : factor_(&factor),
      pencil_(&pencil),
      sigma_(sigma),
      refinement_(refinement),
      factorization_id_(factor.stats().factorizations) {
  if (factor.order() != pencil.order())
    HS_THROW(InvalidInput, "ShiftInvertOperator: factor order "
                               << factor.order() << " != pencil order " << pencil.order());
  if (!factor.inertia().valid)
    HS_THROW(InvalidInput,
             "ShiftInvertOperator: the factor holds no usable factorization (the last factorize() "
             "failed or was never called)");
  if (!std::isfinite(sigma))
    HS_THROW(InvalidInput, "ShiftInvertOperator: non-finite sigma " << sigma);
  if (refinement.max_steps < 0 || !(refinement.rel_tol >= 0.0))
    HS_THROW(InvalidInput, "ShiftInvertOperator: invalid refinement policy max_steps="
                               << refinement.max_steps << ", rel_tol=" << refinement.rel_tol);
}

template <class T>
void ShiftInvertOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  HS_ASSERT(
      factor_->stats().factorizations == factorization_id_,
      "ShiftInvertOperator(sigma=" << sigma_
                                   << "): its factorization object was refactorized after the "
                                      "operator was built");
  const index_t n = size();
  const CsrMatrix<T>* b = pencil_->b();
  apply_b(b, n, x, ldx, y, ldy, p);
  if (refinement_.max_steps == 0) {
    factor_->solve(y, p, ldy);
    return;
  }

  // Iterative refinement: the factor K~ may be a perturbed K (single precision, BLR), but the
  // residual is computed in double precision with the exact A and B, so the fixed point of
  // y <- y + K~^{-1} (Bx - K y) is the exact K^{-1} B x; each step contracts the error by
  // about ||I - K~^{-1} K|| ~ eps_factor * cond(K).
  const std::size_t len = static_cast<std::size_t>(n * p);
  if (bx_.size() < len) {
    bx_.resize(len);
    residual_.resize(len);
  }
  T* bx = bx_.data();
  T* r = residual_.data();
  std::vector<double> bx_norm(static_cast<std::size_t>(p));
  for (index_t k = 0; k < p; ++k) {
    std::copy(y + k * ldy, y + k * ldy + n, bx + k * n);
    bx_norm[k] = linalg::norm2(n, bx + k * n);
  }
  factor_->solve(y, p, ldy);

  double previous = std::numeric_limits<double>::infinity();
  for (int step = 0;; ++step) {
    // r = Bx - (A y - sigma B y): two spmm, K is never formed.
    std::copy(bx, bx + len, r);
    spmm(pencil_->a(), y, ldy, r, n, p, T(-1), T(1));
    if (b) {
      spmm(*b, y, ldy, r, n, p, T(sigma_), T(1));
    } else {
      for (index_t k = 0; k < p; ++k) linalg::axpy(n, T(sigma_), y + k * ldy, r + k * n);
    }
    double worst = 0.0;
    for (index_t k = 0; k < p; ++k) {
      const double rn = linalg::norm2(n, r + k * n);
      const double rel = rn == 0.0 ? 0.0
                                   : (bx_norm[k] > 0.0 ? rn / bx_norm[k]
                                                       : std::numeric_limits<double>::infinity());
      worst = std::max(worst, rel);
    }
    last_rel_residual_ = worst;
    // Stop when converged, out of steps, or when the last correction did not reduce the
    // residual (attainable accuracy reached, or eps_factor * cond(K) >= 1).
    if (worst <= refinement_.rel_tol || step == refinement_.max_steps || !(worst < previous)) break;
    previous = worst;
    factor_->solve(r, p, n);
    for (index_t k = 0; k < p; ++k) linalg::axpy(n, T(1), r + k * n, y + k * ldy);
    ++refinement_steps_;
  }
}

#define HS_INSTANTIATE_SHIFT_INVERT(T) template class ShiftInvertOperator<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SHIFT_INVERT)
#undef HS_INSTANTIATE_SHIFT_INVERT

}  // namespace hs
