// General (non-self-adjoint) Krylov-Schur eigensolver (G. W. Stewart, "A Krylov-Schur algorithm
// for large eigenproblems", SIMAX 23(3), 2001) in the Euclidean inner product.
// Contract: include/hsolve/eigen.hpp; design: docs/DESIGN.md §3.5.
//
// Invariant between cycles (the Krylov-Schur decomposition):
//     Op V_m = V_m H_m + v_{m+1} b^T,     [V_m  v_{m+1}] orthonormal,
// where after a restart H_k is (quasi-)triangular and Arnoldi steps append Hessenberg columns.
// One cycle: expand to ncv vectors (Arnoldi, CGS2) -> real/complex Schur form H_m = Q S Q^H ->
// move the wanted Ritz values to the top of S and certify the converged prefix (below) -> reorder
// the best k Ritz values to the top of S (trsen) and truncate: V <- V Q(:, 0:k), H <- S(0:k, 0:k),
// b <- b^T Q(:, 0:k).
//
// Convergence. A Ritz pair (theta, x = V y) passes the eigenvector test when its estimate |b^T y|
// is <= tol * scale(theta), scale = |theta| (or max |Ritz value| for theta numerically 0, see
// ConvergenceScale). nconv is the longest prefix, in `which` ranking order, of wanted values that
// all pass and whose Schur block, moved to the top of S, passes the invariant-subspace test
// ||b^T Q(:, 0:nconv)||_2 <= tol * ||scale(theta[0, nconv))||_2:
//   * a prefix, not a count: a converged value ranked behind an unconverged one is not counted, so
//     nconv >= nev holds exactly when every wanted value converged. (Counting converged blocks in
//     any order is wrong when a conjugate pair straddles nev: the wanted set then holds nev + 1
//     values, and the pair plus nev - 2 others reach nev with the best value still missing.)
//   * the subspace test: Op Q1 - Q1 S11 = v (b^T Q1) for the leading Schur vectors Q1, so span(Q1)
//     is an invariant subspace of Op + E with ||E||_2 = ||b^T Q1||_2. For a non-normal operator
//     this exceeds the eigenvector estimates by up to the condition number of the eigenvector
//     basis (nearly parallel eigenvectors), so the eigenvector test alone does not certify it.
// Locking is soft: converged Ritz pairs stay in the retained set and keep being refined.
// Stewart's optional deflation (zeroing the b entries of converged Schur vectors) is deliberately
// not used: it perturbs the operator by the discarded coupling (up to tol*|theta|), after which
// the estimates of those pairs would read 0 although their true residuals do not.
//
// Estimates versus true residuals. |b^T y| is the residual of the computed Krylov-Schur relation.
// Every compression (V Q, gees, trsen) adds O(eps ||Op||) rounding error E to that relation,
// Op V = V H + v b^T + E, and for x = V y with H y = theta y the true residual is
// ||v (b^T y) + E y||, so the estimate drifts from it by up to ||E|| (measured: ~0.1 eps ||Op||_2
// per restart, growing linearly, i.e. a few 1e-14 ||Op|| after 1000 restarts). Re-orthonormalizing
// V would not remove it: the drift is in the relation, not in the orthogonality of V. The driver
// checks true residuals of every pair (DESIGN §3.6-7).
//
// Magnitude. linalg::norm2 is a plain sum of squares, which over- or underflows for entries beyond
// about 1e+-154. The iteration therefore runs on Op / 2^e, with e fixed by the largest entry of
// the first nonzero operator output. Scaling by a power of two is exact, so the iteration is
// bitwise the same for every e, and the results are multiplied back exactly.
//
// Block mode (block_size = p > 1). The same cycle on a block Krylov-Schur decomposition
// Op V = V H + U C (block_general.hpp: p new vectors per operator application, block CGS2 +
// Cholesky-QR, random replacement of dependent columns): every test above uses the coupling block
// C in place of the row b^T (estimate ||C y||, subspace test ||C Q1||_F), and the selection,
// certification and output code is shared, so both modes follow one contract. Differences: the
// basis grows by whole blocks, so a cycle ends at the largest size <= ncv reachable from the kept
// count; and Rayleigh-Ritz probes at block boundaries end a cycle early once all nev wanted values
// are certified (from the single-vector basis size on in the first cycle, from the first new block
// in later ones), which keeps the larger block basis cheap. With p = 1 the single-vector code runs
// unchanged.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "eigen/block_general.hpp"
#include "eigen/general_krylov_schur.hpp"
#include "eigen/general_schur.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {
namespace eigen_detail {
namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
// CGS2 ("twice is enough") leaves w orthogonal to span(V) to working precision unless the input
// was numerically inside span(V). When more than 1 - kCancellation of ||Op v|| cancelled, an
// extra CGS2 pass decides between a genuine (small) new direction and rounding noise.
constexpr double kCancellation = 1e-8;
// DGKS criterion (Daniel, Gragg, Kaufman, Stewart 1976): if another orthogonalization pass still
// removes more than 1 - 1/sqrt(2) of the norm, the vector is noise inside span(V).
constexpr double kDgks = 0.70710678118654752;
// Rows per block of the in-place basis compression V(:, 0:k) = V(:, 0:m) Q(:, 0:k) and of the
// output products V Y.
constexpr index_t kRowBlock = 2048;
// A random vector orthogonalized against a proper subspace keeps a fraction ~sqrt((n-j)/n) of its
// norm; needing more than one attempt already means numerical trouble.
constexpr int kRandomAttempts = 8;
struct Dimensions {
  index_t n = 0;
  index_t nev = 0;
  index_t ncv = 0;
};

template <class T>
Dimensions validate(index_t n, const KrylovSchurOptions<T>& o) {
  if (o.deflation) {
    HS_THROW(InvalidInput,
             "krylov_schur_general: deflation is not supported (the orthogonal complement of an "
             "invariant subspace of a non-normal operator is not invariant); rerun with a larger "
             "nev/ncv instead");
  }
  if (o.block_size < 1)
    HS_THROW(InvalidInput, "krylov_schur_general: block_size=" << o.block_size << " < 1");
  if (n < 1) HS_THROW(InvalidInput, "krylov_schur_general: empty operator");
  if (o.nev < 1 || o.nev > n) {
    HS_THROW(InvalidInput, "krylov_schur_general: nev=" << o.nev << " outside [1, " << n << "]");
  }
  if (o.max_restarts < 0) {
    HS_THROW(InvalidInput, "krylov_schur_general: max_restarts=" << o.max_restarts << " < 0");
  }
  if (!(o.tol >= 0.0) || !std::isfinite(o.tol)) {
    HS_THROW(InvalidInput, "krylov_schur_general: tol=" << o.tol << " is not a finite value >= 0");
  }
  // Room for the wanted set (nev + 1 values when, for real T, a pair straddles the cut) plus at
  // least one new Arnoldi vector per restart; the whole space always suffices.
  const index_t min_ncv = std::min(n, o.nev + (is_complex_v<T> ? 1 : 2));
  const index_t requested = o.ncv == 0 ? std::max(2 * o.nev + 1, o.nev + 16) : o.ncv;
  if (requested < min_ncv) {
    HS_THROW(InvalidInput, "krylov_schur_general: ncv=" << o.ncv << " < " << min_ncv
                                                        << " (nev=" << o.nev << ", n=" << n << ")");
  }
  if (o.initial) {
    if (o.initial->rows() != n || o.initial->cols() < 1) {
      HS_THROW(InvalidInput, "krylov_schur_general: initial block is "
                                 << o.initial->rows() << " x " << o.initial->cols() << ", need "
                                 << n << " x >=1");
    }
    // The largest entry rather than the norm: a sum of squares could overflow for valid input.
    const double amax = max_abs_component(n, o.initial->col(0));
    if (!(amax > 0.0) || !std::isfinite(amax)) {
      HS_THROW(InvalidInput, "krylov_schur_general: initial vector must be nonzero and finite");
    }
    // Block mode starts from the first min(p, cols) columns: all of them must be finite.
    const index_t used = std::min(o.block_size, o.initial->cols());
    for (index_t c = 1; c < used; ++c) {
      if (!std::isfinite(max_abs_component(n, o.initial->col(c)))) {
        HS_THROW(InvalidInput,
                 "krylov_schur_general: initial block column " << c << " is not finite");
      }
    }
  }
  // Larger requests are clamped: a basis of the whole space is exact.
  return {n, o.nev, std::min(requested, n)};
}

/// Storage of the Krylov-Schur decomposition Op V(:, 0:m) = V(:, 0:m) H(0:m, 0:m) + V(:, m) b^T
/// with b^T = H(m, 0:m) and m = ncv after each expansion. V is n x (ncv+1), H is (ncv+1) x ncv.
/// Op stands for the scaled operator Op / op_scale() throughout (see the file comment).
template <class T>
class KrylovSchurBasis {
 public:
  KrylovSchurBasis(index_t n, index_t ncv, std::uint64_t seed)
      : n_(n), ncv_(ncv), seed_(seed), v_(n, ncv + 1), h_(ncv + 1, ncv) {}

  index_t n() const noexcept { return n_; }
  index_t ncv() const noexcept { return ncv_; }
  /// Basis size of the projected matrix after an expansion (always ncv).
  index_t size() const noexcept { return ncv_; }
  const T* h() const noexcept { return h_.data(); }
  index_t ldh() const noexcept { return h_.rows(); }
  /// True once the basis spans the whole space (only with ncv == n): then b = 0 and the
  /// decomposition Op V = V H is exact.
  bool complete() const noexcept { return complete_; }
  /// The power of two s the operator is divided by: Ritz values of H times s are those of Op.
  double op_scale() const noexcept { return std::ldexp(1.0, scale_exponent_); }

  /// V(:, 0) = v0 / ||v0||, or a deterministic random unit vector (stream 0) when v0 == nullptr.
  void start(const T* v0) {
    T* v = v_.col(0);
    if (v0) {
      std::copy(v0, v0 + n_, v);
      // Bring the largest entry to [1, 2) first (exact): the norm is a plain sum of squares.
      linalg::scal(n_, T(std::ldexp(1.0, -binary_exponent(max_abs_component(n_, v)))), v);
    } else {
      linalg::fill_random(v, n_, seed_, 0);
    }
    linalg::scal(n_, T(1.0 / linalg::norm2(n_, v)), v);
  }

  /// Arnoldi steps j = k .. ncv-1, i.e. columns k.. of H and V(:, k+1 .. ncv).
  /// Throws InvalidInput when the operator returns a non-finite vector.
  void expand(LinearOperator<T>& op, index_t k, GeneralPhaseTimes* times) {
    for (index_t j = k; j < ncv_; ++j) {
      T* w = v_.col(j + 1);
      {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::apply_s));
        op.apply(v_.col(j), n_, w, n_, 1);
      }
      PhaseClock clock(phase(times, &GeneralPhaseTimes::orthogonalize_s));
      const double wnorm = scale_output(w);
      if (!std::isfinite(wnorm)) {
        // Not a breakdown: NaN/Inf would end up in H, where gees either fails or deflates them
        // into "converged" garbage.
        HS_THROW(InvalidInput, "krylov_schur_general: operator output of Arnoldi step "
                                   << j << " is not finite (NaN/Inf entries, or a norm beyond "
                                   << "the double range relative to the first output)");
      }
      T* hj = h_.col(j);
      double beta =
          linalg::orthogonalize_cgs2<T>(n_, j + 1, v_.data(), n_, nullptr, 0, w, nullptr, hj);
      if (j + 1 == n_) {
        // V(:, 0:n) spans the whole space: the remainder of w is rounding noise.
        hj[j + 1] = T(0);
        complete_ = true;
        return;
      }
      if (beta <= kCancellation * wnorm) beta = recheck(j + 1, w, hj, beta);
      if (beta > 0.0) {
        hj[j + 1] = T(beta);
        linalg::scal(n_, T(1.0 / beta), w);
      } else {
        // Breakdown: span(V) is numerically invariant. Continue with a fresh direction and zero
        // coupling, so the relation stays exact and the search covers the rest of the space.
        hj[j + 1] = T(0);
        random_direction(j + 1, w);
      }
    }
  }

  /// b^T = H(ncv, 0:ncv), as a 1 x ncv coupling block.
  DenseMatrix<T> coupling() const {
    DenseMatrix<T> b(1, ncv_);
    for (index_t j = 0; j < ncv_; ++j) b(0, j) = h_(ncv_, j);
    return b;
  }

  /// Keep the leading k Schur vectors of the projected matrix ps (H(0:ncv, 0:ncv) = Q S Q^H):
  ///   V(:, 0:k) <- V(:, 0:ncv) Q(:, 0:k),  V(:, k) <- v_{ncv+1},
  ///   H(0:k, 0:k) <- S(0:k, 0:k),          H(k, 0:k) <- b^T Q(:, 0:k).
  /// Requires k to be a block boundary of S, so that S(0:k, 0:k) is itself a Schur form.
  void compress(const ProjectedSchur<T>& ps, const DenseMatrix<T>& b, index_t k,
                GeneralPhaseTimes* times) {
    const index_t m = ncv_;
    HS_ASSERT(ps.order() == m && k >= 1 && k < m && ps.is_boundary(k),
              "restart keeps k=" << k << " of m=" << m);
    const DenseMatrix<T>& q = ps.vectors();
    const DenseMatrix<T>& s = ps.form();
    {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::update_s));
      // In place by row blocks: each block of V's rows is copied out once and overwritten by the
      // product, so no second n x k buffer is needed (memory budget, DESIGN §3.6-3).
      aligned_vector<T> rows(static_cast<std::size_t>(std::min(kRowBlock, n_) * m));
      for (index_t r0 = 0; r0 < n_; r0 += kRowBlock) {
        const index_t rb = std::min(kRowBlock, n_ - r0);
        for (index_t j = 0; j < m; ++j) {
          std::copy(v_.col(j) + r0, v_.col(j) + r0 + rb, rows.data() + j * rb);
        }
        linalg::multiply(rb, m, k, rows.data(), rb, q.data(), m, v_.data() + r0, n_);
      }
      std::copy(v_.col(m), v_.col(m) + n_, v_.col(k));
    }
    h_.set_zero();
    for (index_t j = 0; j < k; ++j) {
      std::copy(s.col(j), s.col(j) + k, h_.col(j));
      h_(k, j) = row_times_column(m, b.data(), q.col(j));
    }
  }

  /// Y (n x cols, ld ldy) = V(:, 0:ncv) C with C ncv x cols (ld ldc).
  void combine(const T* c, index_t ldc, index_t cols, T* y, index_t ldy) const {
    linalg::multiply(n_, ncv_, cols, v_.data(), n_, c, ldc, y, ldy);
  }

  /// Frees the basis storage (after the result has been formed); the object is unusable after.
  void release() {
    v_ = DenseMatrix<T>();
    h_ = DenseMatrix<T>();
  }

 private:
  /// Divides an operator output by the operator scale, fixing the scale at the first nonzero
  /// output, and returns ||w||_2 (not finite when w has NaN or infinite entries).
  double scale_output(T* w) {
    if (!scale_fixed_) {
      const double amax = max_abs_component(n_, w);
      // An exactly zero output (breakdown) or a non-finite one (the caller throws) fixes nothing.
      if (amax == 0.0 || !std::isfinite(amax)) return linalg::norm2(n_, w);
      scale_exponent_ = binary_exponent(amax);
      scale_fixed_ = true;
    }
    if (scale_exponent_ != 0) linalg::scal(n_, T(std::ldexp(1.0, -scale_exponent_)), w);
    return linalg::norm2(n_, w);
  }

  /// Extra CGS2 pass after heavy cancellation (DGKS test). Returns the new norm if w is a genuine
  /// new direction, 0 if it was rounding noise inside span(V(:, 0:cols)).
  double recheck(index_t cols, T* w, T* hj, double beta) {
    std::vector<T> extra(static_cast<std::size_t>(cols));
    const double beta2 = linalg::orthogonalize_cgs2<T>(n_, cols, v_.data(), n_, nullptr, 0, w,
                                                       nullptr, extra.data());
    for (index_t i = 0; i < cols; ++i) hj[i] += extra[i];
    return beta2 > kDgks * beta ? beta2 : 0.0;
  }

  /// w <- deterministic random unit vector orthogonal to V(:, 0:cols) (cols < n).
  void random_direction(index_t cols, T* w) {
    for (int attempt = 0; attempt < kRandomAttempts; ++attempt) {
      linalg::fill_random(w, n_, seed_, next_stream_++);
      const double before = linalg::norm2(n_, w);
      const double after =
          linalg::orthogonalize_cgs2<T>(n_, cols, v_.data(), n_, nullptr, 0, w, nullptr, nullptr);
      if (after > kCancellation * before) {
        linalg::scal(n_, T(1.0 / after), w);
        return;
      }
    }
    HS_ASSERT(false,
              "no random direction outside a " << cols << "-dimensional subspace of C^" << n_);
  }

  index_t n_;
  index_t ncv_;
  std::uint64_t seed_;
  std::uint64_t next_stream_ = 1;  ///< stream 0 is the start vector
  bool complete_ = false;
  bool scale_fixed_ = false;
  int scale_exponent_ = 0;  ///< the iteration works with Op / 2^scale_exponent_
  DenseMatrix<T> v_;
  DenseMatrix<T> h_;
};

/// Scale of the relative convergence test (DESIGN §4.4) for the Ritz values of one projected
/// matrix: |theta|, or the largest |Ritz value| when theta is numerically zero. "Numerically zero"
/// is |theta| <= m eps ||S||_F, the rounding level of the computed Ritz values (gees is backward
/// stable with an error of that order): such a value cannot be told apart from 0, and a relative
/// test against its rounding-level magnitude could never pass.
struct ConvergenceScale {
  double largest = 0.0;
  double zero_level = 0.0;

  double of(const cplx& theta) const {
    const double a = std::abs(theta);
    return a > zero_level ? a : largest;
  }
};

template <class T>
ConvergenceScale convergence_scale(const ProjectedSchur<T>& ps) {
  ConvergenceScale cs;
  for (const SchurBlock& blk : ps.blocks()) cs.largest = std::max(cs.largest, std::abs(blk.value));
  const index_t m = ps.order();
  // ld == rows, so the m*m entries of S form one vector whose 2-norm is ||S||_F.
  cs.zero_level = static_cast<double>(m) * kEps * linalg::norm2(m * m, ps.form().data());
  return cs;
}

/// value / scale for the convergence history; infinite when a nonzero value meets a zero scale.
double relative(double value, double scale) {
  if (scale > 0.0) return value / scale;
  return value > 0.0 ? kInf : 0.0;
}

/// ||C y||_2^2 for the coupling block C (p x m) of a Krylov-Schur relation. For one row b^T this
/// is exactly abs2(b^T y), as the single-vector solver has always computed it.
template <class T>
double coupling_norm2(const DenseMatrix<T>& c, const T* y) {
  const index_t m = c.cols();
  if (c.rows() == 1) return abs2(row_times_column(m, c.data(), y));
  double s = 0.0;
  for (index_t r = 0; r < c.rows(); ++r) {
    T acc = T(0);
    for (index_t i = 0; i < m; ++i) acc += c(r, i) * y[i];
    s += abs2(acc);
  }
  return s;
}

/// Ritz vectors of the projected matrix with the estimates ||C y||_2 (block_general.hpp); one
/// coupling row is handed to ProjectedSchur::ritz_vectors as it is.
template <class T>
RitzVectors<T> ritz_of(const ProjectedSchur<T>& ps, const DenseMatrix<T>& c) {
  if (c.rows() == 1) return ps.ritz_vectors(c.data());
  const std::vector<T> none(static_cast<std::size_t>(ps.order()), T(0));
  RitzVectors<T> r = ps.ritz_vectors(none.data());
  for (const SchurBlock& blk : ps.blocks()) {
    // A real pair: y = Re + i Im with real C, so ||C y||^2 = ||C Re||^2 + ||C Im||^2.
    double e2 = coupling_norm2(c, r.y.col(blk.start));
    if (blk.size == 2) e2 += coupling_norm2(c, r.y.col(blk.start + 1));
    const double est = std::sqrt(e2);
    for (index_t i = 0; i < blk.size; ++i) r.estimate[blk.start + i] = est;
  }
  return r;
}

/// One Rayleigh-Ritz step. S is reordered so that positions [0, m_out) hold the wanted values and
/// [0, nconv) the certified converged prefix; ritz belongs to S as reordered.
template <class T>
struct Cycle {
  ProjectedSchur<T> ps;
  RitzVectors<T> ritz;
  index_t m_out = 0;  ///< wanted values: nev, or nev + 1 when a pair straddles position nev
  index_t nconv = 0;  ///< certified prefix (file comment)
  /// Largest relative estimate of the wanted values in [nconv, m_out) and relative Schur
  /// coupling of the rejected prefixes (RestartRecord::max_estimate).
  double max_estimate = 0.0;
};

/// The longest prefix P of the wanted blocks (positions [0, m_out), taken in ranking order) whose
/// pairs pass the eigenvector test and which, moved to the top of S, passes the invariant-subspace
/// test ||b^T Q(:, 0:|P|)||_2^2 <= tol^2 sum_{theta in P} scale(theta)^2. A candidate failing the
/// subspace test loses its worst-ranked block and is tried again; *rejected receives the largest
/// relative coupling ||b^T Q1||_2 / ||scale(theta_P)||_2 of the rejected candidates. Leaves P on
/// top of S and cyc.ritz up to date. Returns |P| (number of values).
template <class T>
index_t certified_prefix(Cycle<T>& cyc, const DenseMatrix<T>& b, Which which, double tol,
                         double* rejected) {
  ProjectedSchur<T>& ps = cyc.ps;
  bool moved = false;  // P was moved to the top since `cap` last changed
  for (index_t cap = cyc.m_out; cap > 0;) {
    const ConvergenceScale scale = convergence_scale(ps);
    const std::vector<SchurBlock>& blocks = ps.blocks();
    std::vector<char> prefix(blocks.size(), 0);
    index_t held = 0;
    double scale2 = 0.0;
    for (const index_t blk : ps.ranking(which)) {
      const SchurBlock& sb = blocks[blk];
      if (sb.start >= cyc.m_out) continue;  // not wanted
      const double s = scale.of(sb.value);
      if (held + sb.size > cap || !(cyc.ritz.estimate[sb.start] <= tol * s)) break;
      prefix[blk] = 1;
      held += sb.size;
      scale2 += static_cast<double>(sb.size) * s * s;
    }
    if (held == 0) return 0;
    // The blocks of P hold `held` values, so P = positions [0, held) iff each lies inside it.
    bool on_top = true;
    for (std::size_t blk = 0; blk < blocks.size(); ++blk) {
      if (prefix[blk] && blocks[blk].start + blocks[blk].size > held) on_top = false;
    }
    if (!on_top) {
      if (moved) {
        // Moving P changed the estimates enough (rounding at the threshold) to change P again.
        cap = held - 1;
        moved = false;
        continue;
      }
      // P and the rest of the wanted set stay inside [0, m_out): trsen only moves P upwards. A
      // refused swap (nearly equal eigenvalues, real T) leaves P scattered: certify nothing. The
      // Ritz data must follow the (possibly partial) reordering either way.
      const bool reordered = ps.reorder_to_front(prefix);
      cyc.ritz = ritz_of(ps, b);
      if (!reordered) return 0;
      moved = true;
      continue;  // re-evaluate on the reordered form, whose estimates are recomputed
    }
    double coupling2 = 0.0;
    for (index_t j = 0; j < held; ++j) {
      coupling2 += coupling_norm2(b, ps.vectors().col(j));
    }
    if (coupling2 <= tol * tol * scale2) return held;
    *rejected = std::max(*rejected, relative(std::sqrt(coupling2), std::sqrt(scale2)));
    cap = held - 1;
    moved = false;
  }
  return 0;
}

template <class Basis, class T>
Cycle<T> analyze(const Basis& basis, const DenseMatrix<T>& b, Which which, index_t nev,
                 double tol) {
  Cycle<T> cyc{ProjectedSchur<T>(basis.size(), basis.h(), basis.ldh()), {}, 0, 0, 0.0};
  ProjectedSchur<T>& ps = cyc.ps;
  const std::vector<index_t> ranked = ps.ranking(which);
  index_t count = 0;
  const index_t nb = blocks_for_count(ps.blocks(), ranked, nev, &count);
  std::vector<char> wanted(ps.blocks().size(), 0);
  for (index_t r = 0; r < nb; ++r) wanted[ranked[r]] = 1;
  const bool wanted_on_top = ps.reorder_to_front(wanted);
  // The wanted set fills positions [0, count) in its previous relative order, so a pair that
  // straddles nev in ranking order may sit anywhere inside that block: the cut is count, not nev.
  // Only a refused swap can leave a pair across count; count is nev or nev + 1, so moving the cut
  // by one keeps m_out in {nev, nev + 1} and the pair whole.
  cyc.m_out = count;
  if (!ps.is_boundary(cyc.m_out)) cyc.m_out += cyc.m_out == nev ? 1 : -1;
  HS_ASSERT(ps.is_boundary(cyc.m_out), "no block boundary near " << count);
  cyc.ritz = ritz_of(ps, b);
  double rejected = 0.0;
  // After a refused swap the leading block may hold values other than the wanted ones, so
  // nothing is certified in this cycle (only possible for real T with nearly equal eigenvalues).
  if (wanted_on_top) cyc.nconv = certified_prefix(cyc, b, which, tol, &rejected);
  if (cyc.nconv < cyc.m_out) {
    const ConvergenceScale scale = convergence_scale(ps);
    cyc.max_estimate = rejected;
    for (const SchurBlock& blk : ps.blocks()) {
      if (blk.start < cyc.nconv || blk.start >= cyc.m_out) continue;
      cyc.max_estimate =
          std::max(cyc.max_estimate, relative(cyc.ritz.estimate[blk.start], scale.of(blk.value)));
    }
  }
  return cyc;
}

/// Reorder S so that the wanted values (positions [0, m_out)) and then the best-ranked others,
/// `keep` values in all, lead; returns how many Schur vectors to retain (keep, keep + 1 when a
/// pair straddles and fits within max_keep, otherwise keep - 1). max_keep leaves room for at least
/// one new basis vector (block) per cycle: ncv - 1 for a single vector, m - p in block mode.
template <class T>
index_t select_retained(Cycle<T>& cyc, index_t keep, index_t max_keep, Which which) {
  ProjectedSchur<T>& ps = cyc.ps;
  const std::vector<SchurBlock>& blocks = ps.blocks();
  std::vector<char> selected(blocks.size(), 0);
  index_t held = 0;
  for (std::size_t blk = 0; blk < blocks.size(); ++blk) {
    if (blocks[blk].start < cyc.m_out) {
      selected[blk] = 1;
      held += blocks[blk].size;
    }
  }
  index_t last = -1;
  for (const index_t blk : ps.ranking(which)) {
    if (held >= keep) break;
    if (selected[blk]) continue;
    selected[blk] = 1;
    held += blocks[blk].size;
    last = blk;
  }
  if (held > max_keep) {
    // The last block added is a pair straddling `keep` whose second member does not fit
    // (keep == max_keep): drop it. The wanted set itself fits, m_out <= nev + 1 <= max_keep.
    HS_ASSERT(last >= 0 && blocks[last].size == 2,
              "retained " << held << " of at most " << max_keep);
    selected[last] = 0;
    held -= 2;
  }
  index_t k = held;
  if (!ps.reorder_to_front(selected) && !ps.is_boundary(k)) {
    // A refused swap (nearly equal eigenvalues) left a pair across k: keep it whole if possible.
    k = k + 1 <= max_keep ? k + 1 : k - 1;
  }
  return k;
}

/// The result from the cycle that ends the run: the Schur basis Q1 = V Q(:, 0:m_out), Ritz
/// vectors x = V y = Q1 z (unit 2-norm) and the values scaled back to Op. Releases the basis.
/// Memory: Q1 is formed while V is alive, the n x m_out complex vectors only after V is freed, so
/// the peak is max(V + Q1, Q1 + vectors) rather than V + Q1 + vectors (DESIGN §3.6-3).
template <class Basis, class T>
GeneralEigResult<T> materialize(Basis& basis, const Cycle<T>& cyc, Which which) {
  const ProjectedSchur<T>& ps = cyc.ps;
  const std::vector<SchurBlock>& blocks = ps.blocks();
  const index_t m = ps.order();
  const index_t m_out = cyc.m_out;
  const index_t n = basis.n();
  const double s = basis.op_scale();
  const std::vector<index_t> ranked = ps.ranking(which);
  GeneralEigResult<T> res;
  for (const index_t blk : ranked) {
    res.ritz_all.push_back(s * blocks[blk].value);
    if (blocks[blk].size == 2) res.ritz_all.push_back(std::conj(s * blocks[blk].value));
  }
  // Output order: the certified prefix, then the rest of the wanted block, each by ranking. The
  // prefix is a ranking prefix of the wanted set, so this is the ranking order of all m values.
  std::vector<index_t> order;
  for (const index_t blk : ranked) {
    if (blocks[blk].start < cyc.nconv) order.push_back(blk);
  }
  for (const index_t blk : ranked) {
    if (blocks[blk].start >= cyc.nconv && blocks[blk].start < m_out) order.push_back(blk);
  }
  // Coordinates y of the output Ritz vectors in the Krylov basis; for real T a pair contributes
  // the columns (Re y, Im y) of its positive-imaginary member.
  DenseMatrix<T> coef(m, m_out);
  std::vector<char> pair_at(static_cast<std::size_t>(m_out), 0);
  index_t c = 0;
  for (const index_t blk_index : order) {
    const SchurBlock& blk = blocks[blk_index];
    const double est = s * cyc.ritz.estimate[blk.start];
    for (index_t e = 0; e < blk.size; ++e) {
      const T* y = cyc.ritz.y.col(blk.start + e);
      std::copy(y, y + m, coef.col(c + e));
    }
    if (blk.size == 1) {
      res.theta.push_back(s * blk.value);
      res.estimate.push_back(est);
    } else {
      pair_at[c] = 1;
      res.theta.insert(res.theta.end(), {s * blk.value, std::conj(s * blk.value)});
      res.estimate.insert(res.estimate.end(), {est, est});
    }
    c += blk.size;
  }
  // Q1 = V Q(:, 0:m_out), then V is no longer needed.
  res.schur_basis.resize(n, m_out);
  basis.combine(ps.vectors().data(), m, m_out, res.schur_basis.data(), n);
  basis.release();
  // The output values lead S, so each eigenvector of H is y = Q z with z supported on
  // [0, m_out) (back substitution in the quasi-triangular S): x = V y = Q1 z, z = Q(:, 0:m_out)^H y.
  DenseMatrix<T> z(m_out, m_out);
  {
    const auto mo = static_cast<lapack::fint>(m_out), mm = static_cast<lapack::fint>(m);
    lapack::gemm('C', 'N', mo, mo, mm, T(1), ps.vectors().data(), mm, coef.data(), mm, T(0),
                 z.data(), mo);
  }
  // x = Q1 z, converted to complex columns block by block: no n x m_out buffer of T besides the
  // outputs.
  res.vectors.resize(n, m_out);
  aligned_vector<T> block(static_cast<std::size_t>(std::min(kRowBlock, n) * m_out));
  for (index_t r0 = 0; r0 < n; r0 += kRowBlock) {
    const index_t rb = std::min(kRowBlock, n - r0);
    linalg::multiply(rb, m_out, m_out, res.schur_basis.data() + r0, n, z.data(), m_out,
                     block.data(), rb);
    for (index_t col = 0; col < m_out; ++col) {
      const T* x0 = block.data() + col * rb;
      cplx* out0 = res.vectors.col(col) + r0;
      if constexpr (!is_complex_v<T>) {
        if (pair_at[col]) {
          const T* x1 = x0 + rb;
          cplx* out1 = res.vectors.col(col + 1) + r0;
          for (index_t i = 0; i < rb; ++i) {
            out0[i] = cplx(x0[i], x1[i]);
            out1[i] = cplx(x0[i], -x1[i]);
          }
          ++col;
          continue;
        }
      }
      for (index_t i = 0; i < rb; ++i) out0[i] = cplx(x0[i]);
    }
  }
  for (index_t col = 0; col < m_out; ++col) {
    // One factor for both members of a pair keeps them exact conjugates.
    const cplx inv(1.0 / linalg::norm2(n, res.vectors.col(col)));
    linalg::scal(n, inv, res.vectors.col(col));
    if (pair_at[col]) linalg::scal(n, inv, res.vectors.col(++col));
  }
  res.schur_form.resize(m_out, m_out);
  for (index_t j = 0; j < m_out; ++j) {
    for (index_t i = 0; i < m_out; ++i) res.schur_form(i, j) = T(s) * ps.form()(i, j);
  }
  res.nconv = cyc.nconv;
  return res;
}

/// The single-vector solver (block_size == 1), or one exact cycle over the whole space when
/// dim.ncv == n.
template <class T>
GeneralEigResult<T> run_single(LinearOperator<T>& op, Which which,
                               const KrylovSchurOptions<T>& options, const Dimensions& dim,
                               GeneralPhaseTimes* times) {
  const index_t applies_before = op.applications();
  // Retained Schur vectors per restart: nev plus half of the remaining room, the usual balance
  // between restart frequency and the information kept (adjusted per cycle for pairs).
  const index_t keep = dim.nev + (dim.ncv - dim.nev) / 2;

  KrylovSchurBasis<T> basis(dim.n, dim.ncv, options.seed);
  basis.start(options.initial ? options.initial->col(0) : nullptr);
  std::vector<RestartRecord> history;
  index_t k = 0;
  for (index_t restart = 0;; ++restart) {
    basis.expand(op, k, times);
    const DenseMatrix<T> b = basis.coupling();
    Cycle<T> cyc = [&] {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      return analyze(basis, b, which, dim.nev, options.tol);
    }();
    history.push_back({restart, cyc.nconv, cyc.max_estimate, op.applications() - applies_before});

    // nconv never splits a pair and is at most m_out, so nconv >= nev means nconv == m_out.
    if (basis.complete() || restart == options.max_restarts || cyc.nconv >= dim.nev) {
      GeneralEigResult<T> result = [&] {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::update_s));
        return materialize(basis, cyc, which);
      }();
      result.converged = result.nconv >= dim.nev;
      result.restarts = restart;
      result.op_applies = op.applications() - applies_before;
      result.history = std::move(history);
      return result;
    }
    {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      k = select_retained(cyc, keep, dim.ncv - 1, which);
    }
    basis.compress(cyc.ps, b, k, times);
  }
}

/// Block mode (block_size = p > 1, see eigen.hpp): block Arnoldi expansion by whole blocks up to
/// the layout's capacity, Rayleigh-Ritz probes at block boundaries (early stop), thick restart
/// with the shared selection logic. Falls back to one exact single-vector cycle over the whole
/// space when the layout does not fit (block_general_layout).
template <class T>
GeneralEigResult<T> run_block(LinearOperator<T>& op, Which which,
                              const KrylovSchurOptions<T>& options, const Dimensions& dim,
                              GeneralPhaseTimes* times) {
  const index_t p = options.block_size;
  const BlockGeneralLayout lay =
      block_general_layout(dim.n, dim.nev, dim.ncv, options.ncv == 0, p, is_complex_v<T>);
  if (!lay.feasible) {
    // The space is below twice a block basis plus its residual block: one single-vector cycle
    // over all of it is exact (every copy of a multiple eigenvalue included) and costs n
    // applications, about what a restarted block run of that size takes.
    Dimensions whole = dim;
    whole.ncv = dim.n;
    return run_single(op, which, options, whole, times);
  }
  const index_t applies_before = op.applications();
  BlockKrylovSchurBasis<T> basis(dim.n, lay.ncv, p, options.seed);
  {
    PhaseClock clock(phase(times, &GeneralPhaseTimes::orthogonalize_s));
    basis.start(options.initial);
  }
  std::vector<RestartRecord> history;
  // Smallest basis size probed: the single-vector basis size in the first cycle (a smaller block
  // Krylov space can hold nev converged values that are not the nev wanted ones, see
  // block_lanczos.cpp "Early stop"), the first block after the kept vectors later on.
  index_t probe_from = lay.first_probe;
  for (index_t restart = 0;; ++restart) {
    const index_t m_end = basis.size() + p * ((lay.ncv - basis.size()) / p);
    std::optional<Cycle<T>> cyc;
    DenseMatrix<T> c;
    while (basis.size() < m_end) {
      basis.add_block(op, times);
      if (basis.size() < m_end && basis.size() >= probe_from) {
        c = basis.coupling();
        Cycle<T> probe = [&] {
          PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
          return analyze(basis, c, which, dim.nev, options.tol);
        }();
        if (probe.nconv >= dim.nev) {
          cyc = std::move(probe);
          break;
        }
      }
    }
    if (!cyc) {
      c = basis.coupling();
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      cyc = analyze(basis, c, which, dim.nev, options.tol);
    }
    history.push_back(
        {restart, cyc->nconv, cyc->max_estimate, op.applications() - applies_before});
    if (restart == options.max_restarts || cyc->nconv >= dim.nev) {
      GeneralEigResult<T> result = [&] {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::update_s));
        return materialize(basis, *cyc, which);
      }();
      result.converged = result.nconv >= dim.nev;
      result.restarts = restart;
      result.op_applies = op.applications() - applies_before;
      result.history = std::move(history);
      return result;
    }
    // The cycle ran to m_end > ncv - p; keep at most m - p vectors so a whole block follows.
    const index_t max_keep = basis.size() - p;
    index_t k = 0;
    {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      k = select_retained(*cyc, std::min(lay.keep, max_keep), max_keep, which);
    }
    basis.compress(cyc->ps, c, k, times);
    probe_from = k + p;
  }
}

}  // namespace

template <class T>
GeneralEigResult<T> krylov_schur_general_timed(LinearOperator<T>& op, Which which,
                                               const KrylovSchurOptions<T>& options,
                                               GeneralPhaseTimes* times) {
  PhaseClock total(phase(times, &GeneralPhaseTimes::total_s));
  const Dimensions dim = validate(op.size(), options);
  if (options.block_size > 1) return run_block(op, which, options, dim, times);
  return run_single(op, which, options, dim, times);
}

}  // namespace eigen_detail

template <class T>
GeneralEigResult<T> krylov_schur_general(LinearOperator<T>& op, Which which,
                                         const KrylovSchurOptions<T>& options) {
  return eigen_detail::krylov_schur_general_timed(op, which, options, nullptr);
}

#define HS_INSTANTIATE_KRYLOV_SCHUR_GENERAL(T)                                        \
  template GeneralEigResult<T> krylov_schur_general<T>(LinearOperator<T>&, Which,     \
                                                       const KrylovSchurOptions<T>&); \
  template GeneralEigResult<T> eigen_detail::krylov_schur_general_timed<T>(           \
      LinearOperator<T>&, Which, const KrylovSchurOptions<T>&, eigen_detail::GeneralPhaseTimes*);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_KRYLOV_SCHUR_GENERAL)
#undef HS_INSTANTIATE_KRYLOV_SCHUR_GENERAL

}  // namespace hs
