// General (non-self-adjoint) Krylov-Schur eigensolver (G. W. Stewart, "A Krylov-Schur algorithm
// for large eigenproblems", SIMAX 23(3), 2001) in the Euclidean inner product.
// Contract: include/hsolve/eigen.hpp; design: docs/DESIGN.md §3.5.
//
// Invariant between cycles (the Krylov-Schur decomposition):
//     Op V_m = V_m H_m + v_{m+1} b^T,     [V_m  v_{m+1}] orthonormal,
// where after a restart H_k is (quasi-)triangular and Arnoldi steps append Hessenberg columns.
// One cycle: expand to ncv vectors (Arnoldi, CGS2) -> real/complex Schur form H_m = Q S Q^H ->
// rank the Ritz values -> convergence test on eigenvector residuals -> reorder the best k Ritz
// values to the top of S (trsen) and truncate: V <- V Q(:, 0:k), H <- S(0:k, 0:k), b <- b^T Q.
//
// Locking is soft: converged Ritz pairs stay in the retained set and keep being refined.
// Stewart's optional deflation (zeroing the b entries of leading converged Schur vectors) is
// deliberately not used: it perturbs the operator by the discarded coupling (up to
// tol*|theta|), after which the estimates of those pairs would read 0 although their true
// residuals do not. Without it every reported estimate is the residual ||Op x - theta x|| of the
// actual Krylov relation, up to rounding.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "eigen/general_krylov_schur.hpp"
#include "eigen/general_schur.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/error.hpp"
#include "linalg/dense_ops.hpp"

namespace hs {
namespace eigen_detail {
namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
// CGS2 ("twice is enough") leaves w orthogonal to span(V) to working precision unless the input
// was numerically inside span(V). When more than 1 - kCancellation of ||Op v|| cancelled, an
// extra CGS2 pass decides between a genuine (small) new direction and rounding noise.
constexpr double kCancellation = 1e-8;
// DGKS criterion (Daniel, Gragg, Kaufman, Stewart 1976): if another orthogonalization pass still
// removes more than 1 - 1/sqrt(2) of the norm, the vector is noise inside span(V).
constexpr double kDgks = 0.70710678118654752;
// Rows per block of the in-place basis compression V(:, 0:k) = V(:, 0:m) Q(:, 0:k).
constexpr index_t kRowBlock = 2048;
// A random vector orthogonalized against a proper subspace keeps a fraction ~sqrt((n-j)/n) of its
// norm; needing more than one attempt already means numerical trouble.
constexpr int kRandomAttempts = 8;

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

double* phase(GeneralPhaseTimes* times, double GeneralPhaseTimes::* member) {
  return times ? &(times->*member) : nullptr;
}

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
    const double nrm = linalg::norm2(n, o.initial->col(0));
    if (!(nrm > 0.0) || !std::isfinite(nrm)) {
      HS_THROW(InvalidInput, "krylov_schur_general: initial vector norm " << nrm);
    }
  }
  // Larger requests are clamped: a basis of the whole space is exact.
  return {n, o.nev, std::min(requested, n)};
}

/// Storage of the Krylov-Schur decomposition Op V(:, 0:m) = V(:, 0:m) H(0:m, 0:m) + V(:, m) b^T
/// with b^T = H(m, 0:m) and m = ncv after each expansion. V is n x (ncv+1), H is (ncv+1) x ncv.
template <class T>
class KrylovSchurBasis {
 public:
  KrylovSchurBasis(index_t n, index_t ncv, std::uint64_t seed)
      : n_(n), ncv_(ncv), seed_(seed), v_(n, ncv + 1), h_(ncv + 1, ncv) {}

  index_t ncv() const noexcept { return ncv_; }
  const T* h() const noexcept { return h_.data(); }
  index_t ldh() const noexcept { return h_.rows(); }
  /// True once the basis spans the whole space (only with ncv == n): then b = 0 and the
  /// decomposition Op V = V H is exact.
  bool complete() const noexcept { return complete_; }

  /// V(:, 0) = v0 / ||v0||, or a deterministic random unit vector (stream 0) when v0 == nullptr.
  void start(const T* v0) {
    T* v = v_.col(0);
    if (v0) {
      std::copy(v0, v0 + n_, v);
    } else {
      linalg::fill_random(v, n_, seed_, 0);
    }
    linalg::scal(n_, T(1.0 / linalg::norm2(n_, v)), v);
  }

  /// Arnoldi steps j = k .. ncv-1, i.e. columns k.. of H and V(:, k+1 .. ncv).
  void expand(LinearOperator<T>& op, index_t k, GeneralPhaseTimes* times) {
    for (index_t j = k; j < ncv_; ++j) {
      T* w = v_.col(j + 1);
      {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::apply_s));
        op.apply(v_.col(j), n_, w, n_, 1);
      }
      PhaseClock clock(phase(times, &GeneralPhaseTimes::orthogonalize_s));
      T* hj = h_.col(j);
      const double wnorm = linalg::norm2(n_, w);
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

  /// b^T = H(ncv, 0:ncv).
  std::vector<T> coupling() const {
    std::vector<T> b(static_cast<std::size_t>(ncv_));
    for (index_t j = 0; j < ncv_; ++j) b[j] = h_(ncv_, j);
    return b;
  }

  /// Keep the leading k Schur vectors of the projected matrix ps (H(0:ncv, 0:ncv) = Q S Q^H):
  ///   V(:, 0:k) <- V(:, 0:ncv) Q(:, 0:k),  V(:, k) <- v_{ncv+1},
  ///   H(0:k, 0:k) <- S(0:k, 0:k),          H(k, 0:k) <- b^T Q(:, 0:k).
  /// Requires k to be a block boundary of S, so that S(0:k, 0:k) is itself a Schur form.
  void compress(const ProjectedSchur<T>& ps, const std::vector<T>& b, index_t k,
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

 private:
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
  DenseMatrix<T> v_;
  DenseMatrix<T> h_;
};

/// Convergence test of DESIGN §4.4: estimate <= tol * |theta|. |theta| is replaced by the largest
/// |Ritz value| when theta is numerically zero (|theta| <= eps * largest): such a value cannot be
/// told apart from 0, and a relative test against its rounding-level magnitude could never pass.
double convergence_scale(const cplx& theta, double largest) {
  const double a = std::abs(theta);
  return a > kEps * largest ? a : largest;
}

bool is_converged(double estimate, const cplx& theta, double largest, double tol) {
  return estimate <= tol * convergence_scale(theta, largest);
}

double largest_magnitude(const std::vector<SchurBlock>& blocks) {
  double largest = 0.0;
  for (const SchurBlock& blk : blocks) largest = std::max(largest, std::abs(blk.value));
  return largest;
}

/// The wanted Ritz values of one cycle: the best-ranked blocks holding nev eigenvalues.
struct WantedSet {
  index_t nblocks = 0;          ///< leading entries of the ranking forming the wanted set
  index_t count = 0;            ///< eigenvalues held: nev, or nev + 1 when a pair straddles
  index_t nconv = 0;            ///< eigenvalues in converged wanted blocks
  double max_estimate = 0.0;    ///< largest relative estimate among unconverged wanted ones
  std::vector<char> converged;  ///< per block (positional index); meaningful for wanted ones
};

template <class T>
struct Cycle {
  ProjectedSchur<T> ps;
  std::vector<index_t> ranked;
  WantedSet wanted;
};

template <class T>
Cycle<T> analyze(const KrylovSchurBasis<T>& basis, const std::vector<T>& b, Which which,
                 index_t nev, double tol) {
  ProjectedSchur<T> ps(basis.ncv(), basis.h(), basis.ldh());
  std::vector<index_t> ranked = ps.ranking(which);
  const RitzVectors<T> ritz = ps.ritz_vectors(b.data());
  const std::vector<SchurBlock>& blocks = ps.blocks();
  const double largest = largest_magnitude(blocks);

  WantedSet ws;
  ws.converged.assign(blocks.size(), 0);
  ws.nblocks = blocks_for_count(blocks, ranked, nev, &ws.count);
  for (index_t r = 0; r < ws.nblocks; ++r) {
    const SchurBlock& blk = blocks[ranked[r]];
    const double est = ritz.estimate[blk.start];
    if (is_converged(est, blk.value, largest, tol)) {
      ws.converged[ranked[r]] = 1;
      ws.nconv += blk.size;
    } else {
      const double scale = convergence_scale(blk.value, largest);
      ws.max_estimate = std::max(
          ws.max_estimate, scale > 0.0 ? est / scale : std::numeric_limits<double>::infinity());
    }
  }
  return {std::move(ps), std::move(ranked), std::move(ws)};
}

/// Reorder S so that the `keep` best-ranked Ritz values lead; returns how many Schur vectors to
/// retain (keep, keep + 1 when a pair straddles and fits below ncv, otherwise keep - 1).
template <class T>
index_t select_retained(Cycle<T>& cyc, index_t keep, index_t ncv) {
  ProjectedSchur<T>& ps = cyc.ps;
  index_t held = 0;
  index_t nb = blocks_for_count(ps.blocks(), cyc.ranked, keep, &held);
  if (held > ncv - 1) {
    // The last block is a pair straddling `keep` whose second member does not fit: drop it.
    // This needs keep == ncv - 1 >= nev + 1, so the pair starts at or after position nev and
    // the wanted set (a block prefix holding nev or nev + 1 values) lies entirely before it.
    --nb;
    held -= 2;
  }
  std::vector<char> selected(ps.blocks().size(), 0);
  for (index_t r = 0; r < nb; ++r) selected[cyc.ranked[r]] = 1;
  index_t k = held;
  if (!ps.reorder_to_front(selected) && !ps.is_boundary(k)) {
    // A refused swap (nearly equal eigenvalues) left a pair across k: keep it whole if possible.
    k = k + 1 <= ncv - 1 ? k + 1 : k - 1;
  }
  return k;
}

/// What the result reports, decided on the small projected problem.
template <class T>
struct OutputLayout {
  ProjectedSchur<T> fs;        ///< Schur form reordered for output
  RitzVectors<T> ritz;         ///< Ritz vectors / estimates of fs
  std::vector<index_t> order;  ///< block indices of fs in output order
  index_t m_out = 0;           ///< leading Schur block returned: nev or nev + 1
  index_t nconv = 0;
};

/// Reorders (a copy of) the last cycle's Schur form so that the wanted set leads, converged
/// blocks first, and takes theta, vectors and estimates from the reordered decomposition, so
/// they are consistent with schur_form/schur_basis by construction. nconv is the longest run of
/// converged blocks at the top of S: the converged wanted set after a successful reordering, a
/// conservative count if LAPACK refused a swap (then the leading block may hold a value nearly
/// equal to a wanted one instead of it; the Schur decomposition itself stays exact).
template <class T>
OutputLayout<T> layout_output(const Cycle<T>& cyc, const std::vector<T>& b, Which which,
                              index_t nev, double tol) {
  OutputLayout<T> out{cyc.ps, {}, {}, 0, 0};
  ProjectedSchur<T>& fs = out.fs;
  const WantedSet& ws = cyc.wanted;
  std::vector<char> wanted(fs.blocks().size(), 0);
  for (index_t r = 0; r < ws.nblocks; ++r) wanted[cyc.ranked[r]] = 1;
  const bool wanted_on_top = fs.reorder_to_front(wanted);
  if (wanted_on_top && ws.nconv > 0 && ws.nconv < ws.count) {
    // The wanted set now fills positions [0, count) in its previous relative order. Flags are
    // carried by position, not by block index: a pair may split into two real 1x1 blocks when
    // swapped, but it never leaves its two positions. Reordering inside the leading block (even
    // partially, on a refused swap) never moves anything across position count.
    const std::vector<SchurBlock>& old_blocks = cyc.ps.blocks();
    std::vector<char> converged_at(static_cast<std::size_t>(fs.order()), 0);
    index_t pos = 0;
    for (std::size_t blk = 0; blk < wanted.size(); ++blk) {
      if (!wanted[blk]) continue;
      for (index_t e = 0; e < old_blocks[blk].size; ++e) converged_at[pos++] = ws.converged[blk];
    }
    std::vector<char> converged_first(fs.blocks().size(), 0);
    for (std::size_t blk = 0; blk < fs.blocks().size(); ++blk) {
      converged_first[blk] = converged_at[fs.blocks()[blk].start];
    }
    fs.reorder_to_front(converged_first);
  }
  // A refused swap may leave a pair across position count; count is nev or nev + 1, so moving
  // the cut by one keeps m_out in {nev, nev + 1} and the pair whole.
  out.m_out = ws.count;
  if (!fs.is_boundary(out.m_out)) out.m_out += out.m_out == nev ? 1 : -1;
  HS_ASSERT(fs.is_boundary(out.m_out), "no block boundary near " << ws.count);

  out.ritz = fs.ritz_vectors(b.data());
  const std::vector<SchurBlock>& blocks = fs.blocks();
  const double largest = largest_magnitude(blocks);
  for (const SchurBlock& blk : blocks) {
    if (blk.start >= out.m_out ||
        !is_converged(out.ritz.estimate[blk.start], blk.value, largest, tol)) {
      break;
    }
    out.nconv += blk.size;
  }
  const std::vector<index_t> ranked = fs.ranking(which);
  for (const index_t blk : ranked) {
    if (blocks[blk].start < out.nconv) out.order.push_back(blk);
  }
  for (const index_t blk : ranked) {
    if (blocks[blk].start >= out.nconv && blocks[blk].start < out.m_out) out.order.push_back(blk);
  }
  return out;
}

/// The n-sized part of the result: Ritz vectors x = V y (unit 2-norm) and the Schur basis.
template <class T>
GeneralEigResult<T> materialize(const KrylovSchurBasis<T>& basis, const OutputLayout<T>& out,
                                Which which, index_t n, index_t nev) {
  const ProjectedSchur<T>& fs = out.fs;
  const std::vector<SchurBlock>& blocks = fs.blocks();
  const index_t m = fs.order();
  GeneralEigResult<T> res;
  for (const index_t blk : fs.ranking(which)) {
    res.ritz_all.push_back(blocks[blk].value);
    if (blocks[blk].size == 2) res.ritz_all.push_back(std::conj(blocks[blk].value));
  }
  // All Ritz vectors in one product; real T multiplies the (Re y, Im y) columns of pairs.
  DenseMatrix<T> coef(m, out.m_out);
  index_t c = 0;
  for (const index_t blk : out.order) {
    for (index_t e = 0; e < blocks[blk].size; ++e, ++c) {
      const T* y = out.ritz.y.col(blocks[blk].start + e);
      std::copy(y, y + m, coef.col(c));
    }
  }
  DenseMatrix<T> x(n, out.m_out);
  basis.combine(coef.data(), m, out.m_out, x.data(), n);

  res.vectors.resize(n, out.m_out);
  c = 0;
  for (const index_t blk_index : out.order) {
    const SchurBlock& blk = blocks[blk_index];
    const double est = out.ritz.estimate[blk.start];
    cplx* x0 = res.vectors.col(c);
    if (blk.size == 1) {
      for (index_t i = 0; i < n; ++i) x0[i] = cplx(x(i, c));
      linalg::scal(n, cplx(1.0 / linalg::norm2(n, x0)), x0);
      res.theta.push_back(blk.value);
      res.estimate.push_back(est);
    } else if constexpr (!is_complex_v<T>) {
      cplx* x1 = res.vectors.col(c + 1);
      for (index_t i = 0; i < n; ++i) {
        x0[i] = cplx(x(i, c), x(i, c + 1));
        x1[i] = std::conj(x0[i]);
      }
      const double inv = 1.0 / linalg::norm2(n, x0);
      linalg::scal(n, cplx(inv), x0);
      linalg::scal(n, cplx(inv), x1);
      res.theta.insert(res.theta.end(), {blk.value, std::conj(blk.value)});
      res.estimate.insert(res.estimate.end(), {est, est});
    }
    c += blk.size;
  }

  res.schur_basis.resize(n, out.m_out);
  basis.combine(fs.vectors().data(), m, out.m_out, res.schur_basis.data(), n);
  res.schur_form.resize(out.m_out, out.m_out);
  for (index_t j = 0; j < out.m_out; ++j) {
    std::copy(fs.form().col(j), fs.form().col(j) + out.m_out, res.schur_form.col(j));
  }
  res.nconv = out.nconv;
  res.converged = out.nconv >= nev;
  return res;
}

}  // namespace

template <class T>
GeneralEigResult<T> krylov_schur_general_timed(LinearOperator<T>& op, Which which,
                                               const KrylovSchurOptions<T>& options,
                                               GeneralPhaseTimes* times) {
  PhaseClock total(phase(times, &GeneralPhaseTimes::total_s));
  const Dimensions dim = validate(op.size(), options);
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
    const std::vector<T> b = basis.coupling();
    Cycle<T> cyc = [&] {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      return analyze(basis, b, which, dim.nev, options.tol);
    }();
    history.push_back(
        {restart, cyc.wanted.nconv, cyc.wanted.max_estimate, op.applications() - applies_before});

    const bool last = basis.complete() || restart == options.max_restarts;
    if (last || cyc.wanted.nconv == cyc.wanted.count) {
      const OutputLayout<T> out = [&] {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
        return layout_output(cyc, b, which, dim.nev, options.tol);
      }();
      GeneralEigResult<T> result = [&] {
        PhaseClock clock(phase(times, &GeneralPhaseTimes::update_s));
        return materialize(basis, out, which, dim.n, dim.nev);
      }();
      // The final reordering recomputes the estimates; a pair sitting exactly at the threshold
      // may flip, in which case iterating on is the honest answer while restarts remain.
      if (last || result.converged) {
        result.restarts = restart;
        result.op_applies = op.applications() - applies_before;
        result.history = std::move(history);
        return result;
      }
    }
    {
      PhaseClock clock(phase(times, &GeneralPhaseTimes::projected_s));
      k = select_retained(cyc, keep, dim.ncv);
    }
    basis.compress(cyc.ps, b, k, times);
  }
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
