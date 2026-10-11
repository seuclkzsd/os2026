// Dense vector/block helpers shared by the eigensolvers and the driver (module-internal API).
// Single owner of: deterministic random vectors, inner products/norms, CGS2 orthogonalization.
// T in {double, cplx}. Vectors are contiguous; blocks are column-major with explicit ld.
//
// Reproducibility. Every reduction in this file (dot, norm2 and the projections V^H M w of the CGS
// routines) is summed over fixed row blocks whose boundaries depend only on the problem size: one
// thread sums a block in a fixed order (explicit lane accumulators, no reassociation), block sums
// are added in block order into at most 256 slots, and the slots are added serially. The results
// are therefore bitwise identical for every OpenMP thread count and from run to run (the CGS
// kernels are hand-written for that reason, not BLAS gemv). This holds for one binary: another
// compiler, flags (e.g. FMA contraction) or BLAS may change the last bits.
// The summation order differs from a plain left-to-right sum; the accuracy is the usual
// O(eps * sum |x_i y_i|) of a blocked sum.
//
// Thread safety: all functions are reentrant (no shared state); arrays passed as outputs must not
// be accessed concurrently. Called from inside an active OpenMP region they run on one thread.
#pragma once

#include <cstdint>
#include <functional>

#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"

namespace hs::linalg {

/// Deterministic pseudo-random entries, uniform in [-1, 1] (real and imaginary parts
/// independently for complex T). (seed, stream) select an independent sequence; the result
/// does not depend on the number of threads. O(n).
template <class T>
void fill_random(T* x, index_t n, std::uint64_t seed, std::uint64_t stream = 0);

/// x^H y (conjugate-linear in x). OpenMP-parallel, bitwise independent of the thread count (see
/// the file comment). O(n), reads 2n scalars.
template <class T>
T dot(index_t n, const T* x, const T* y);

/// ||x||_2 (overflow-safe two-pass scaling is not needed for our magnitudes; plain sum of
/// squares, which over/underflows only beyond ~1e154 / below ~1e-154). Deterministic like dot.
template <class T>
double norm2(index_t n, const T* x);

/// y += alpha * x.
template <class T>
void axpy(index_t n, T alpha, const T* x, T* y);

/// x *= alpha.
template <class T>
void scal(index_t n, T alpha, T* x);

/// y = M * x for one vector (used to recompute M*w instead of storing M*V).
template <class T>
using ApplyFn = std::function<void(const T* x, T* y)>;

/// Y = M * X for p columns (X n x p with ld ldx, Y n x p with ld ldy): the block form of ApplyFn.
template <class T>
using ApplyBlockFn = std::function<void(const T* x, index_t ldx, T* y, index_t ldy, index_t p)>;

/// When the CGS routines run their second projection pass.
enum class Reorth {
  /// Always two passes (classical CGS2, "twice is enough": the result is orthogonal to working
  /// precision whenever w is numerically independent of span(V)). Two passes also correct for a
  /// V that is only approximately M-orthonormal: with ||V^H M V - I|| = delta, V^H M w keeps a
  /// relative size O(delta^2) after two passes (O(delta) after one).
  Always,
  /// Daniel-Gragg-Kaufman-Stewart criterion: skip pass 2 when pass 1 removed little of w, i.e.
  /// ||w after pass 1||_M >= eta * ||w before||_M. Rounding in pass 1 leaves a component in
  /// span(V) of relative size O(eps ||w_before|| / ||w_after||) <= O(eps / eta), so the loss of
  /// orthogonality stays at the CGS2 level; heavier cancellation triggers pass 2.
  /// Requires, beyond the general preconditions:
  ///   * V M-orthonormal to working precision (||V^H M V - I|| = O(eps)). A skipped pass leaves
  ///     V^H M w = -(V^H M V - I) c1: for a V that is orthonormal only to delta (e.g. an
  ///     externally supplied block accepted with a looser tolerance) the result is M-orthogonal
  ///     only to O(delta), against O(delta^2) with Always. Use Always for such a V.
  ///   * stored mode: bw == M*w on entry, as the test is computed from Re(w^H bw).
  IfNeeded,
  /// The default: IfNeeded in Euclidean and recompute mode, where the routine forms the norms of
  /// the test itself; Always in stored mode, where they would come from the caller's bw (callers
  /// that keep bw only as scratch, with content unrelated to M*w, stay correct: the projections
  /// never read bw).
  Auto,
};

struct OrthoOptions {
  Reorth reorth = Reorth::Auto;
  /// DGKS threshold in [0, 1] (1/sqrt(2), as in ARPACK and SLEPc), used wherever the DGKS test
  /// runs (IfNeeded; Auto outside stored mode). 1 makes it equivalent to Always except for a w
  /// that is exactly orthogonal already; 0 never runs pass 2 for a nonzero w.
  double eta = 0.70710678118654752;
  /// Recompute mode only: false declares that the caller does not need bw == M*w on exit (e.g.
  /// bw is mere workspace). The norms are then taken from the Pythagorean relation
  /// ||w - V c||_M^2 = ||w||_M^2 - ||c||^2 (V M-orthonormal) where that is accurate: a call
  /// whose first pass passes the DGKS test applies M once (not twice), a call with two passes
  /// twice (not three times). On exit bw holds M times an earlier iterate of w.
  bool mw_on_exit = true;
};

/// What one orthogonalize_cgs2 call did. Norms are M-norms (2-norms in Euclidean mode), computed
/// as sqrt(max(Re(w^H M w), 0)) (NaN propagates).
struct OrthoInfo {
  int passes = 0;            ///< projection passes performed: 0 iff k == 0, else 1 or 2
  double norm_before = 0.0;  ///< ||w|| on entry
  double norm_pass1 = 0.0;   ///< ||w|| after the first pass (== norm_before when k == 0)
  double norm_after = 0.0;   ///< ||w|| on exit (also the return value, see orthogonalize_cgs2)
};

/// Orthogonalize w against the k columns of V (n x k, ld ldv) by block classical Gram-Schmidt
/// with reorthogonalization (CGS2, BLAS2 work), in the inner product <x, y>_M = y^H M x.
/// Three modes:
///   * Euclidean (M = I): bv == nullptr, bw == nullptr, apply_m == nullptr.
///   * Stored M*V:        bv = M*V (n x k, ld ldbv) and bw = M*w on entry; bw is updated
///                        consistently (bw -= BV*c), no application of M. The subtraction
///                        keeps the absolute rounding error O(eps ||M w_entry||): after heavy
///                        cancellation bw and the returned norm are inaccurate relative to the
///                        result (the projections are not affected). Recompute M*w before
///                        storing it as a basis column. The projections use BV and w only: a
///                        caller that discards bw and the norm may pass any bw (scratch) with
///                        the default options (Reorth::Auto runs two passes in this mode), but
///                        not with Reorth::IfNeeded, whose test reads bw.
///   * Recompute:         bv == nullptr, apply_m != nullptr, bw = workspace (n); M*w is
///                        recomputed after every pass, so every norm (and the DGKS test) uses
///                        the M*w of the current w: 2 applications of M with one pass, 3 with two.
/// options.reorth selects the second pass (default Auto: DGKS with eta = 1/sqrt(2) in Euclidean
/// and recompute mode, two passes in stored mode); Always forces two passes, IfNeeded runs the
/// DGKS test in every mode (see Reorth for its preconditions). On exit bw == M*w
/// (stored/recompute modes; see options.mw_on_exit for the exception). h (length k, may be nullptr)
/// receives the accumulated coefficients V^H M w. Returns the M-norm sqrt(Re(w^H M w)) (0 if not
/// positive, also for NaN) or ||w||_2 (NaN propagates) of the result; info (may be nullptr)
/// receives the pass count and the norms before / after each pass.
/// Precondition: the columns of V are M-orthonormal (to working precision where the DGKS test
/// runs, see Reorth); w, bw must not alias V, BV.
/// Cost (a sweep reads the n x k operand once, 4nk flops per pass): two passes take 3 sweeps
/// over V (Euclidean), 1 over BV + 2 over [V BV] (stored), 4 over V + 3 applications of M
/// (recompute); one pass takes 2, 1 + 1, 2 sweeps (+ 2 applications of M). The second pass's
/// projection is fused into the first pass's update while each row block is in cache (not in
/// recompute mode, which needs M*w of the whole updated w first). The previous BLAS2
/// implementation took 4, 6 and 4 sweeps. Workspace: O(256 k) scalars.
/// Deterministic for any thread count (file comment).
template <class T>
double orthogonalize_cgs2(index_t n, index_t k, const T* v, index_t ldv, const T* bv, index_t ldbv,
                          T* w, T* bw, T* h, const ApplyFn<T>* apply_m = nullptr,
                          const OrthoOptions& options = OrthoOptions{}, OrthoInfo* info = nullptr);

/// Block CGS2: orthogonalize the p columns of W (n x p, ld ldw) against the k columns of V
/// (n x k, ld ldv) in the M-inner product, with BLAS3 (gemm) on row blocks. The columns of W are
/// NOT orthogonalized among themselves (follow with a QR of the block, e.g. orthogonalize_cgs2 per
/// column). Same three modes as orthogonalize_cgs2: Euclidean (bv, bw, apply_m all null), stored
/// (bv = M*V with ld ldbv, bw = M*W on entry with ld ldbw, updated by subtraction with the same
/// accuracy caveat) and recompute (apply_m != nullptr, bw = n x p workspace with ld ldbw, M*W
/// recomputed after every pass). On exit bw == M*W (stored/recompute modes).
/// DGKS per block: wherever the DGKS test runs (Reorth::IfNeeded; Auto outside stored mode, as
/// for orthogonalize_cgs2) the second pass runs (on the whole block) unless every column keeps
/// ||w_j after pass 1||_M >= eta * ||w_j before||_M (a zero column forces it).
/// h (k x p, ld ldh, may be nullptr) receives the accumulated V^H M W. norms_before / norms_after
/// (length p each, may be nullptr) receive the column norms on entry / exit, computed as in
/// OrthoInfo. Returns the number of passes: 0 when k == 0 or p == 0, else 1 or 2.
/// Preconditions: V M-orthonormal (as for orthogonalize_cgs2); W, BW do not alias V, BV. Cost: the sweeps of
/// orthogonalize_cgs2 with p columns each (the reads of V are shared by the p columns; 4nkp flops
/// per pass). Workspace: O(k p) scalars per reduction slot, at most ~2 MB of slots.
/// Deterministic for any thread count as long as the single-threaded BLAS gemm is (it is called on
/// fixed row blocks from inside an OpenMP region, where an OpenMP BLAS does not spawn threads).
template <class T>
int orthogonalize_block_cgs2(index_t n, index_t k, index_t p, const T* v, index_t ldv, const T* bv,
                             index_t ldbv, T* w, index_t ldw, T* bw, index_t ldbw, T* h, index_t ldh,
                             double* norms_before, double* norms_after,
                             const ApplyBlockFn<T>* apply_m = nullptr,
                             const OrthoOptions& options = OrthoOptions{});

/// C (k x k) = V^H * W for V (n x k, ld ldv), W (n x k, ld ldw). Used for Gram matrices and
/// Rayleigh quotients. BLAS3.
template <class T>
void gram(index_t n, index_t k, const T* v, index_t ldv, const T* w, index_t ldw, T* c, index_t ldc);

/// Y (n x m) = V (n x k) * S (k x m), Y must not alias V. BLAS3.
template <class T>
void multiply(index_t n, index_t k, index_t m, const T* v, index_t ldv, const T* s, index_t lds, T* y,
              index_t ldy);

}  // namespace hs::linalg
