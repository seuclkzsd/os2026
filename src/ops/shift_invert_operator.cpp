// ShiftInvertOperator: Op = (A - sigma B)^{-1} B with optional double-precision iterative
// refinement (docs/DESIGN.md §3.4b); FactorSolveOperator: Op = K^{-1}.
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
             index_t p, const KernelOptions& kernels) {
  if (b) {
    spmm(*b, x, ldx, y, ldy, p, T(1), T(0), kernels);
    return;
  }
  for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, y + k * ldy);
}

// A residual column whose componentwise backward error (shifted_residual) is at most this is
// rounding noise of its own evaluation: a correction computed from it cannot reduce it, so the
// column is not refined further. Measured (bench/solvephase_profile.cpp, 4 threads, p = 1, 4,
// 16): one correction of a double MUMPS solve leaves 1.2-2.2 eps on the six 1e5 test cases,
// while a plain solve has 9 .. 2.5e5 eps. Before P4 a column that cannot meet rel_tol
// (syn-near-singular-complex-1e5: ||r|| / ||Bx|| stalls at 5e-11 > 1e-12) kept computing
// corrections until one was rejected: 3-5 block solves per apply instead of 1 (2.3-6.3x the
// apply time).
constexpr double kRoundingLevel = 8.0 * std::numeric_limits<double>::epsilon();

// The factor must hold a usable factorization (the operators' construction check).
template <class T>
void check_usable(const SparseFactorization<T>& factor, const char* who) {
  if (!factor.inertia().valid)
    HS_THROW(InvalidInput,
             who << ": the factor holds no usable factorization (the last factorize() "
                    "failed or was never called)");
}

}  // namespace

template <class T>
ShiftInvertOperator<T>::ShiftInvertOperator(SparseFactorization<T>& factor,
                                            const ShiftedPencil<T>& pencil, double sigma,
                                            SolveRefinement refinement, KernelPath kernels)
    : factor_(&factor),
      pencil_(&pencil),
      sigma_(sigma),
      refinement_(refinement),
      kernels_(kernels),
      factorization_id_(factor.stats().factorizations) {
  if (factor.order() != pencil.order())
    HS_THROW(InvalidInput, "ShiftInvertOperator: factor order "
                               << factor.order() << " != pencil order " << pencil.order());
  check_usable(factor, "ShiftInvertOperator");
  if (!std::isfinite(sigma))
    HS_THROW(InvalidInput, "ShiftInvertOperator: non-finite sigma " << sigma);
  if (refinement.max_steps < 0 || !(refinement.rel_tol >= 0.0))
    HS_THROW(InvalidInput, "ShiftInvertOperator: invalid refinement policy max_steps="
                               << refinement.max_steps << ", rel_tol=" << refinement.rel_tol);
  resolve_kernel_path(kernels);  // an unavailable path fails here, not at the first apply
}

template <class T>
void ShiftInvertOperator<T>::residual(const T* x, index_t ldx, const T* y, index_t ldy, index_t p,
                                      double* omega) {
  // r = Bx - (A y - sigma B y) = B (x + sigma y) - A y: one sweep over A and B (B x is
  // recomputed rather than stored, against a block solve per step, to keep the workspace small),
  // with the componentwise backward error of every column.
  shifted_residual(pencil_->a(), pencil_->b(), sigma_, x, ldx, y, ldy, residual_.data(), size(), p,
                   KernelOptions{kernels_, &scratch_}, omega);
}

template <class T>
void ShiftInvertOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  HS_ASSERT(
      factor_->stats().factorizations == factorization_id_,
      "ShiftInvertOperator(sigma=" << sigma_
                                   << "): its factorization object was refactorized after the "
                                      "operator was built");
  const index_t n = size();
  apply_b(pencil_->b(), n, x, ldx, y, ldy, p, KernelOptions{kernels_, &scratch_});
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
  // otherwise the column is frozen at its previous iterate (and residual). A column whose residual
  // is at the rounding level of its own evaluation (kRoundingLevel) is not refined at all.
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
  std::vector<double> omega(static_cast<std::size_t>(p)), candidate_omega(omega.size());
  residual(x, ldx, y, ldy, p, omega.data());
  std::vector<double> rel(static_cast<std::size_t>(p));
  std::vector<char> active(static_cast<std::size_t>(p));
  const double floor = refinement_.stop_at_rounding_level ? kRoundingLevel : -1.0;
  index_t remaining = 0;
  for (index_t k = 0; k < p; ++k) {
    rel[k] = relative(k);
    active[k] = rel[k] > refinement_.rel_tol && omega[k] > floor;
    remaining += active[k];
  }
  std::vector<index_t> cols;
  for (int step = 0; step < refinement_.max_steps && remaining > 0; ++step) {
    // Corrections K~^{-1} r of the active columns, solved as one block of only those columns
    // (a block solve costs more with every column, zero or not: 46 vs 81 ms for 1 vs 16 columns
    // on syn-mem27-46, 4 threads), packed into the leading columns of c. Then the candidate
    // iterates y + correction (y itself for finished columns) and their residuals; the finished
    // columns' candidate residuals are ignored.
    cols.clear();
    for (index_t k = 0; k < p; ++k)
      if (active[k]) cols.push_back(k);
    const index_t na = static_cast<index_t>(cols.size());
    for (index_t j = 0; j < na; ++j) std::copy(r + cols[j] * n, r + (cols[j] + 1) * n, c + j * n);
    factor_->solve(c, na, n);
    ++refinement_steps_;
    // Unpack in decreasing order: column cols[j] >= j, so its packed source j is never
    // overwritten before it is read.
    for (index_t j = na - 1; j >= 0; --j) {
      const index_t k = cols[j];
      T* ck = c + k * n;
      if (k != j) std::copy(c + j * n, c + (j + 1) * n, ck);
      linalg::axpy(n, T(1), y + k * ldy, ck);
    }
    for (index_t k = 0; k < p; ++k)
      if (!active[k]) std::copy(y + k * ldy, y + k * ldy + n, c + k * n);
    residual(x, ldx, c, n, p, candidate_omega.data());
    for (index_t k = 0; k < p; ++k) {
      if (!active[k]) continue;
      const double candidate_rel = relative(k);
      if (candidate_rel < rel[k]) {
        std::copy(c + k * n, c + (k + 1) * n, y + k * ldy);
        rel[k] = candidate_rel;
        omega[k] = candidate_omega[k];
        // Still refining this column unless converged or at the rounding level.
        if (candidate_rel > refinement_.rel_tol && omega[k] > floor) continue;
      }
      active[k] = false;  // converged, at the rounding level, or the correction was rejected
      --remaining;
    }
  }
  last_rel_residual_ = p > 0 ? *std::max_element(rel.begin(), rel.end()) : 0.0;
  last_backward_error_ = p > 0 ? *std::max_element(omega.begin(), omega.end()) : 0.0;
}

template <class T>
FactorSolveOperator<T>::FactorSolveOperator(SparseFactorization<T>& factor)
    : factor_(&factor), factorization_id_(factor.stats().factorizations) {
  check_usable(factor, "FactorSolveOperator");
}

template <class T>
void FactorSolveOperator<T>::do_apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
  HS_ASSERT(factor_->stats().factorizations == factorization_id_,
            "FactorSolveOperator: its factorization object was refactorized after the operator "
            "was built");
  const index_t n = size();
  for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, y + k * ldy);
  factor_->solve(y, p, ldy);
}

#define HS_INSTANTIATE_SHIFT_INVERT(T)   \
  template class ShiftInvertOperator<T>; \
  template class FactorSolveOperator<T>;
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SHIFT_INVERT)
#undef HS_INSTANTIATE_SHIFT_INVERT

}  // namespace hs
