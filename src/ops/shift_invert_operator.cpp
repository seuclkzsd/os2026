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
void ShiftInvertOperator<T>::residual(const T* x, index_t ldx, const T* y, index_t ldy, index_t p) {
  const index_t n = size();
  const CsrMatrix<T>* b = pencil_->b();
  T* r = residual_.data();
  // r = Bx - (A y - sigma B y): B x is recomputed rather than stored (one spmm, or a copy when
  // B = I, against a block solve per step) to keep the workspace at two n x p blocks.
  apply_b(b, n, x, ldx, r, n, p);
  spmm(pencil_->a(), y, ldy, r, n, p, T(-1), T(1));
  if (b) {
    spmm(*b, y, ldy, r, n, p, T(sigma_), T(1));
  } else {
    for (index_t k = 0; k < p; ++k) linalg::axpy(n, T(sigma_), y + k * ldy, r + k * n);
  }
}

template <class T>
void ShiftInvertOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  HS_ASSERT(
      factor_->stats().factorizations == factorization_id_,
      "ShiftInvertOperator(sigma=" << sigma_
                                   << "): its factorization object was refactorized after the "
                                      "operator was built");
  const index_t n = size();
  apply_b(pencil_->b(), n, x, ldx, y, ldy, p);
  if (refinement_.max_steps == 0) {
    factor_->solve(y, p, ldy);
    return;
  }

  // Iterative refinement: the factor K~ may be a perturbed K (single precision, BLR), but the
  // residual is computed in double precision with the exact A and B, so the fixed point of
  // y <- y + K~^{-1} (Bx - K y) is the exact K^{-1} B x; each step contracts the error by
  // about ||I - K~^{-1} K|| ~ eps_factor * cond(K). When that exceeds 1 (sigma too close to an
  // eigenvalue for the factor's precision) corrections make things worse, and at the attainable
  // accuracy they merely stagnate. Hence every column keeps its best iterate: a correction is
  // first applied to a candidate copy and only accepted if it reduces that column's residual;
  // otherwise the column is frozen at its previous iterate (and residual).
  const std::size_t len = static_cast<std::size_t>(n * p);
  if (residual_.size() < len) {
    residual_.resize(len);
    candidate_.resize(len);
  }
  T* r = residual_.data();
  T* c = candidate_.data();
  std::vector<double> bx_norm(static_cast<std::size_t>(p));
  for (index_t k = 0; k < p; ++k) bx_norm[k] = linalg::norm2(n, y + k * ldy);  // y = B x here
  auto relative = [&](index_t k) {
    const double rn = linalg::norm2(n, r + k * n);
    if (rn == 0.0) return 0.0;
    return bx_norm[k] > 0.0 ? rn / bx_norm[k] : std::numeric_limits<double>::infinity();
  };

  factor_->solve(y, p, ldy);
  residual(x, ldx, y, ldy, p);
  std::vector<double> rel(static_cast<std::size_t>(p));
  std::vector<char> active(static_cast<std::size_t>(p));
  index_t remaining = 0;
  for (index_t k = 0; k < p; ++k) {
    rel[k] = relative(k);
    active[k] = rel[k] > refinement_.rel_tol;
    remaining += active[k];
  }
  for (int step = 0; step < refinement_.max_steps && remaining > 0; ++step) {
    // Corrections K~^{-1} r for the active columns, then the candidate iterates y + correction
    // and their residuals. Finished columns keep the one-call block solve with a zero right-hand
    // side (zero correction); their candidate residuals are ignored.
    for (index_t k = 0; k < p; ++k) {
      if (active[k]) {
        std::copy(r + k * n, r + (k + 1) * n, c + k * n);
      } else {
        std::fill(c + k * n, c + (k + 1) * n, T(0));
      }
    }
    factor_->solve(c, p, n);
    ++refinement_steps_;
    for (index_t k = 0; k < p; ++k) linalg::axpy(n, T(1), y + k * ldy, c + k * n);
    residual(x, ldx, c, n, p);
    for (index_t k = 0; k < p; ++k) {
      if (!active[k]) continue;
      const double candidate_rel = relative(k);
      if (candidate_rel < rel[k]) {
        std::copy(c + k * n, c + (k + 1) * n, y + k * ldy);
        rel[k] = candidate_rel;
        if (candidate_rel > refinement_.rel_tol) continue;  // still refining this column
      }
      active[k] = false;  // converged, or the correction was rejected (best iterate kept)
      --remaining;
    }
  }
  last_rel_residual_ = p > 0 ? *std::max_element(rel.begin(), rel.end()) : 0.0;
}

#define HS_INSTANTIATE_SHIFT_INVERT(T) template class ShiftInvertOperator<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SHIFT_INVERT)
#undef HS_INSTANTIATE_SHIFT_INVERT

}  // namespace hs
