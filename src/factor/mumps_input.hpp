// Centralized assembled MUMPS input for K = alpha*A + beta*B (module-internal to src/factor).
// T in {double, cplx} is the pencil scalar, U in {double, float} the MUMPS arithmetic.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::mumps {

/// 1-based coordinate (IRN, JCN, A) arrays of the upper triangle of the matrix MUMPS
/// factorizes, built once from a ShiftedPencil pattern and refilled for every (alpha, beta).
///
///  * real T    : the matrix is K itself; entry p of the arrays is entry p of the pencil's
///                upper pattern, and assemble() lets the pencil write straight into A.
///  * complex T : the matrix is the order-2n real embedding E = [[Re K, -Im K], [Im K, Re K]].
///                Its upper triangle is Re K (upper) at (i, j) and at (n+i, n+j), plus the whole
///                -Im K block at (i, n+j) for every j. The strictly lower part of K is not
///                stored, so for an upper entry K(i, j) = v with j > i, Hermitian symmetry
///                gives E(i, n+j) = -Im v and E(j, n+i) = -Im K(j, i) = +Im v; Im K(i, i) = 0.
///                Since alpha and beta are real, Re K = alpha Re A + beta Re B (likewise Im),
///                so a part that is exactly zero in both A and B is zero for EVERY shift and
///                is left out of the pattern (exact, shift-independent pruning; e.g. a complex
///                K whose couplings are mostly real gets an almost block-diagonal E). The
///                diagonal of Re K is always kept.
/// The arrays stay valid (and must not move) while MUMPS uses them; the object keeps a
/// pointer to the pencil, which must outlive it.
template <class T, class U>
class Input {
 public:
  explicit Input(const ShiftedPencil<T>& pencil);
  Input(const Input&) = delete;
  Input& operator=(const Input&) = delete;

  /// Writes the values of alpha*A + beta*B (or of its embedding) into values(). OpenMP, O(nnz).
  void assemble(double alpha, double beta);

  /// Order of the matrix MUMPS sees (n, or 2n for complex T).
  int order() const noexcept { return order_; }
  std::int64_t nnz() const noexcept { return static_cast<std::int64_t>(irn_.size()); }
  int* irn() noexcept { return irn_.data(); }
  int* jcn() noexcept { return jcn_.data(); }
  U* values() noexcept { return values_.data(); }

 private:
  using Scratch = rebind_t<T, U>;  ///< complex<U> for complex T (unused for real T)

  const ShiftedPencil<T>* pencil_;
  int order_ = 0;
  std::vector<int> irn_;
  std::vector<int> jcn_;
  std::vector<U> values_;
  // Complex T only: pencil values are assembled into scratch_, then scattered into values_,
  // laid out as [Re top | Re bottom | -Im upper | +Im mirrored] (see the constructor).
  std::vector<Scratch> scratch_;
  std::vector<index_t> re_src_;  ///< pattern positions whose real part is kept
  std::vector<index_t> im_src_;  ///< off-diagonal pattern positions whose imaginary part is kept
};

}  // namespace hs::mumps
