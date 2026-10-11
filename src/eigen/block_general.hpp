// Internals of hs::krylov_schur_general shared by the single-vector solver and its block mode
// (options.block_size = p > 1), and the block Krylov-Schur basis (module-internal, src/eigen).
// The solver loop and the projected-problem logic live in krylov_schur_general.cpp; this header
// holds what the basis classes of both modes need (phase timing, magnitude scaling) and the block
// basis itself (implemented in block_general.cpp). Scalar set: T in {double, cplx}.
//
// Block Krylov-Schur decomposition (block mode). With m = size() basis vectors and the residual
// block U = V(:, m:m+p),
//     Op V(:, 0:m) = V(:, 0:m) H(0:m, 0:m) + U C,     [V(:, 0:m)  U] orthonormal,
// where C = H(m:m+p, 0:m) is the p x m coupling block (coupling()). A Ritz pair (theta, x = V y)
// then has the residual U C y, i.e. the exact estimate ||C y||_2, and a Schur block Q1 the
// invariant-subspace residual ||C Q1||: the single-vector formulas with the row b^T replaced by C.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "eigen/block_ortho.hpp"
#include "eigen/general_krylov_schur.hpp"
#include "eigen/general_schur.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

// ------------------------------------------------------------------------------ shared helpers

/// Bound of the binary exponent of the operator scale: 2^e and 2^-e stay normal numbers.
constexpr int kMaxScaleExponent = 1000;

/// Adds the lifetime of a scope to *acc; no-op (no clock reads) when acc == nullptr.
class PhaseClock {
 public:
  explicit PhaseClock(double* acc) : acc_(acc) {
    if (acc_) t0_ = std::chrono::steady_clock::now();
  }
  ~PhaseClock() {
    if (acc_)
      *acc_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }
  PhaseClock(const PhaseClock&) = delete;
  PhaseClock& operator=(const PhaseClock&) = delete;

 private:
  double* acc_;
  std::chrono::steady_clock::time_point t0_;
};

inline double* phase(GeneralPhaseTimes* times, double GeneralPhaseTimes::*member) {
  return times ? &(times->*member) : nullptr;
}

/// max_i max(|Re x_i|, |Im x_i|), or +infinity when an entry is NaN or infinite. Serial O(n): it
/// runs once per start column and on the first nonzero operator output.
template <class T>
double max_abs_component(index_t n, const T* x) {
  double amax = 0.0;
  for (index_t i = 0; i < n; ++i) {
    double a = 0.0;
    if constexpr (is_complex_v<T>) {
      a = std::max(std::abs(x[i].real()), std::abs(x[i].imag()));
    } else {
      a = std::abs(x[i]);
    }
    if (!(a <= std::numeric_limits<double>::max())) return std::numeric_limits<double>::infinity();
    amax = std::max(amax, a);
  }
  return amax;
}

/// e with 2^e <= amax < 2^(e+1) for finite amax > 0, clamped to +-kMaxScaleExponent.
inline int binary_exponent(double amax) {
  return std::clamp(std::ilogb(amax), -kMaxScaleExponent, kMaxScaleExponent);
}

// ------------------------------------------------------------------------------ block layout

/// Basis layout of block mode, p > 1 (see krylov_schur_general in eigen.hpp). ncv1 is the
/// validated single-vector basis size (auto or requested, at most n):
///   spare = nev + 1 (complex T) or nev + 2 (real T: room for a conjugate pair straddling nev)
///           rounded up to a multiple of p: a restart keeps at least one Ritz value beyond the
///           wanted set;
///   base  = max(ncv1 rounded up to a multiple of p, spare + 2p): at least two new blocks per
///           cycle;
///   ncv   = base for a requested ncv; base + kAutoExtraBlocks p for the automatic one, as far as
///           the space holds twice the basis plus its residual block (never below base);
///   keep  = nev + (ncv - nev) / 2 rounded down to a multiple of p, clamped to [spare, ncv - p]:
///           the Schur vectors a restart retains (ncv - keep is a whole number of blocks);
///   first_probe = base: the first cycle tests convergence at block boundaries from there on.
/// feasible: ncv1 < n and 2 (ncv + p) <= n; otherwise the call is solved by one single-vector
/// cycle over the whole space (n applications, exact). p > n / 8 is infeasible before any
/// arithmetic (every layout has ncv >= 3p), so no p overflows the rounding.
struct BlockGeneralLayout {
  index_t ncv = 0;
  index_t keep = 0;
  index_t first_probe = 0;
  bool feasible = false;
};

/// Blocks added to base for an automatic ncv (early stop makes the larger basis cheap).
constexpr index_t kAutoExtraBlocks = 4;

BlockGeneralLayout block_general_layout(index_t n, index_t nev, index_t ncv1, bool auto_ncv,
                                        index_t p, bool complex_scalars);

// ------------------------------------------------------------------------------ block basis

/// Storage and growth of the block Krylov-Schur decomposition (file comment). V is
/// n x (capacity + p) and H is (capacity + p) x capacity; the cycle size m grows by whole blocks,
/// m + p <= capacity before each block step. Op stands for the scaled operator Op / op_scale()
/// throughout, as in the single-vector basis (krylov_schur_general.cpp, "Magnitude").
/// Not thread-safe; the operator is applied, never retained.
template <class T>
class BlockKrylovSchurBasis {
 public:
  /// Deficiency threshold of a new block column (BlockOrthonormalizer ratio): a column whose part
  /// outside the basis and the block's earlier columns is below this fraction of its input norm
  /// after both passes is a breakdown and is replaced by a deterministic random vector.
  static constexpr double kBreakdownRatio = 64 * std::numeric_limits<double>::epsilon();

  BlockKrylovSchurBasis(index_t n, index_t capacity, index_t p, std::uint64_t seed);

  index_t n() const noexcept { return n_; }
  index_t p() const noexcept { return p_; }
  index_t capacity() const noexcept { return capacity_; }
  /// Current basis size m (0 after start(), a multiple of p in the first cycle).
  index_t size() const noexcept { return m_; }
  const T* h() const noexcept { return h_.data(); }
  index_t ldh() const noexcept { return h_.rows(); }
  /// The power of two s the operator is divided by: Ritz values of H times s are those of Op.
  double op_scale() const noexcept { return std::ldexp(1.0, scale_exponent_); }
  /// Columns replaced by random vectors so far (breakdowns), start block included.
  index_t replacements() const noexcept { return replacements_; }

  /// Start block V(:, 0:p): the first min(p, initial->cols()) columns of *initial (validated by
  /// the caller: column 0 nonzero, the used columns finite), the others deterministic random
  /// vectors (fill_random streams 0..p-1), orthonormalized; dependent columns other than column 0
  /// are replaced. m = 0. Throws InvalidInput when column 0 is numerically zero after scaling.
  void start(const DenseMatrix<T>* initial);

  /// One block step: W = Op U (one apply of p columns), orthonormalized against V(:, 0:m+p) by
  /// block CGS2 + Cholesky-QR, W = V C + U' R; H(0:m+p, m:m+p) <- C, H(m+p:m+2p, m:m+p) <- R,
  /// U' becomes the residual block and m grows by p. Requires m + p <= capacity. Throws
  /// InvalidInput when the operator output is not finite.
  void add_block(LinearOperator<T>& op, GeneralPhaseTimes* times);

  /// C = H(m:m+p, 0:m), the coupling of the basis to the residual block (p x m).
  DenseMatrix<T> coupling() const;

  /// Thick restart keeping the leading k Schur vectors of ps (the Schur decomposition of
  /// H(0:m, 0:m)) with the coupling c = coupling():
  ///   V(:, 0:k) <- V(:, 0:m) Q(:, 0:k), V(:, k:k+p) <- U, H(0:k, 0:k) <- S(0:k, 0:k),
  ///   H(k:k+p, 0:k) <- C Q(:, 0:k), m <- k.
  /// Requires k to be a block boundary of S and 1 <= k <= m - p.
  void compress(const ProjectedSchur<T>& ps, const DenseMatrix<T>& c, index_t k,
                GeneralPhaseTimes* times);

  /// Y (n x cols, ld ldy) = V(:, 0:m) C with C m x cols (ld ldc).
  void combine(const T* c, index_t ldc, index_t cols, T* y, index_t ldy) const;

  /// The same product handed over by row blocks: consume(r0, rb, block) receives rows
  /// [r0, r0 + rb) of V C as an rb x cols block (ld rb).
  template <class Consume>
  void combine_by_rows(const T* c, index_t ldc, index_t cols, Consume&& consume) const {
    constexpr index_t kRows = 2048;
    aligned_vector<T> block(static_cast<std::size_t>(std::min(kRows, n_) * cols));
    for (index_t r0 = 0; r0 < n_; r0 += kRows) {
      const index_t rb = std::min(kRows, n_ - r0);
      linalg::multiply(rb, m_, cols, v_.data() + r0, n_, c, ldc, block.data(), rb);
      consume(r0, rb, static_cast<const T*>(block.data()));
    }
  }

  /// Frees the basis storage (after the result has been formed); the object is unusable after.
  void release();

 private:
  /// Divides the p operator outputs at V(:, j:j+p) by the operator scale, fixing the scale at the
  /// first nonzero output block, and throws InvalidInput for non-finite output.
  void scale_output(index_t j);

  index_t n_;
  index_t capacity_;
  index_t p_;
  std::uint64_t seed_;
  index_t m_ = 0;
  std::uint64_t next_stream_ = 0;  ///< fill_random stream: 0..p-1 start block, then replacements
  index_t replacements_ = 0;
  bool scale_fixed_ = false;
  int scale_exponent_ = 0;  ///< the iteration works with Op / 2^scale_exponent_
  DenseMatrix<T> v_;
  DenseMatrix<T> h_;
  BlockOrthonormalizer<T> ortho_;
  DenseMatrix<T> coef_;  ///< block CGS2 coefficients of the last block step
  DenseMatrix<T> rblk_;  ///< triangular factor of the last block step
};

}  // namespace hs::eigen_detail
