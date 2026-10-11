// Block thick-restart Lanczos == block Hermitian Krylov-Schur (Zhou & Saad, Numer. Algorithms 47,
// 2008), the block_size > 1 mode of hs::lanczos_krylov_schur. The single-vector solver
// (lanczos_krylov_schur.cpp) documents the shared design decisions (dropped couplings, locking at
// rounding level, local breakdown test, store_bv, deflation semantics); this file only describes
// what blocks change.
//
// Notation. p = block width, M = b_inner or I, every basis M-orthonormal.
//   [D | V | V_m]  storage n x (d + ncv + p): deflation block D, Krylov basis V (ncv columns, a
//                  whole number of blocks after the kept Ritz vectors), residual block V_m.
//   H              ncv x ncv Hermitian projected matrix, R (p x p, upper triangular) the coupling
//                  of the last block to V_m.
// Every cycle maintains the block Krylov-Schur relation
//   Op V = V H + V_m R E^H,   E = the last p columns of I_ncv,
// so a Ritz pair (theta_i, V s_i) has the residual Op x - theta x = V_m R E^H s_i and the exact
// estimate ||R s_i(ncv-p:ncv)||_2. The first cycle is block Lanczos from the start block: H is
// block tridiagonal with diagonal blocks A_j = Herm(V_j^H M Op V_j) and subdiagonal blocks R_j
// (the triangular factors of the block QR of the residual). A restart keeps k Ritz pairs
// (ncv - k a multiple of p):
//   V <- V S(:, 0:k), H <- diag(theta_0..theta_{k-1}), V(:, k:k+p) <- V_m,
//   b = R S(ncv-p:ncv, 0:k),
// with the arrow block H(k:k+p, 0:k) = b (H(0:k, k:k+p) = b^H) followed again by block Lanczos
// steps. As in the single-vector solver only these theoretical entries of H are stored; the other
// block CGS2 coefficients vanish in exact arithmetic.
//
// Block steps. W = Op V_j (one op.apply of p columns) is orthonormalized against [D V] and within
// itself by BlockOrthonormalizer (block CGS2 + Cholesky-QR, BLAS3): W = [D V] C + V_{j+1} R_j.
// A column that is dependent on [D V] and the block's earlier columns to working precision,
// ||new part||_M <= kBreakdownTol * ||Op v||_M (the local test of the single-vector solver applied
// per column), gets R_j(i, i) = 0 and a deterministic random replacement; the dropped part is a
// rounding-level coupling as for a single-vector breakdown. A block keeps its width p throughout,
// which needs d + ncv + p <= n (else the dispatcher solves the free space exactly with one
// single-vector cycle).
//
// Early stop. Every block boundary m also satisfies the relation, with the next block as residual
// block. A Rayleigh-Ritz probe at the boundaries ends a cycle as soon as all nev wanted pairs have
// converged, so the larger block basis (auto ncv: the single-vector one plus 4 blocks) costs little
// on problems that converge early (measured, bench/blockks_shiftinvert.cpp: syn-mem27-46, k = 12,
// p = 8: 48 operator applications with the probe, the full 64 without). Probes start at m = k + p
// in restarted cycles (the kept Ritz vectors carry the earlier cycles), but only at the
// single-vector basis size in the first cycle: a smaller Krylov space can hold nev converged pairs
// that are not the nev wanted ones (a multiple eigenvalue with fewer than its copies yet, measured
// on an operator with three distinct eigenvalues), and the first cycle never accepts on less
// evidence than p = 1 does. The single-vector solver keeps its cycle-end-only test (p = 1 stays
// bitwise unchanged).
//
// Locking. A kept converged pair l gets b(:, l) := 0 once ||b(:, l)||_2 <= kLockTol * (smallest
// reference |theta| of the wanted pairs), the single-vector rule with |b_l| replaced by the norm of
// its coupling column.
//
// Multiplicity. In exact arithmetic a block Krylov space from a start block X contains, for each
// eigenvalue, the M-orthogonal projection of span(X) onto its eigenspace: up to p copies of a
// multiple eigenvalue are found by one run from a random start block (rounding can still add more).
#include "eigen/block_lanczos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "eigen/block_ortho.hpp"
#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

constexpr double kEps = std::numeric_limits<double>::epsilon();
/// Same local breakdown threshold as the single-vector solver, applied per column.
constexpr double kBreakdownTol = 64 * kEps;
/// Same locking threshold as the single-vector solver (on ||b(:, l)||_2).
constexpr double kLockTol = 4 * kEps;
/// A start-block column keeping at most this fraction of its M-norm against D and the earlier
/// start columns is dependent (the single-vector solver's start-vector threshold).
constexpr double kStartMinRatio = 1e-12;
/// Blocks added to the single-vector auto basis size when ncv is chosen automatically (see
/// "Early stop"; measured on shift-invert cases, bench/blockks_shiftinvert.cpp).
constexpr index_t kAutoExtraBlocks = 4;

template <class T>
class BlockThickRestartLanczos {
 public:
  BlockThickRestartLanczos(LinearOperator<T>& op, LinearOperator<T>* b_inner, Which which,
                           const KrylovSchurOptions<T>& options, const LanczosSetup& setup,
                           const BlockDimensions& dims, LanczosTimings& timings);

  HermitianEigResult<T> run();

 private:
  T* vcol(index_t j) { return basis_.col(d_ + j); }
  T* bvcol(index_t j) { return store_bv_ ? bbasis_.col(d_ + j) : nullptr; }
  T* mbasis() { return store_bv_ ? bbasis_.data() : nullptr; }

  void load_deflation();
  void set_start_block();
  bool expand(index_t first);
  void rayleigh_ritz();
  index_t wanted_converged() const;
  index_t record_history();
  void restart();
  HermitianEigResult<T> finish();

  LinearOperator<T>& op_;
  LinearOperator<T>* b_inner_;
  const Which which_;
  const KrylovSchurOptions<T>& opt_;
  LanczosTimings& time_;

  index_t n_ = 0;
  index_t d_ = 0;
  index_t p_ = 0;
  index_t nev_ = 0;
  index_t ncv_ = 0;          ///< basis size of a full cycle (multiple of p)
  index_t keep_ = 0;         ///< Ritz pairs kept per restart (multiple of p)
  index_t m_ = 0;            ///< basis size of the current cycle: ncv_, or less after an early stop
  index_t first_probe_ = 0;  ///< smallest basis size probed in the first cycle
  bool store_bv_ = false;

  DenseMatrix<T> basis_;   ///< [D | V | V_m], n x (d + ncv + p)
  DenseMatrix<T> bbasis_;  ///< M * basis_ (store_bv only)
  BlockOrthonormalizer<T> ortho_;
  DenseMatrix<T> coef_;            ///< block CGS2 coefficients of the current block against [D V]
  DenseMatrix<T> rblk_;            ///< triangular factor of the current block QR
  DenseMatrix<T> h_;               ///< ncv x ncv projected matrix, both triangles stored
  DenseMatrix<T> rlast_;           ///< p x p coupling R of V(:, m-p:m) to V_m = V(:, m:m+p)
  std::uint64_t next_stream_ = 0;  ///< fill_random stream: 0..p-1 start block, then replacements

  // Rayleigh-Ritz results of the current cycle.
  std::vector<double> theta_;   ///< ascending (heev order)
  DenseMatrix<T> s_;            ///< eigenvectors of H, column i <-> theta_[i]
  DenseMatrix<T> coupling_;     ///< p x m: R S(m-p:m, :), the Ritz vectors' couplings to V_m
  std::vector<double> est_;     ///< residual estimates, ||coupling_(:, i)||_2
  std::vector<char> conv_;      ///< est_ <= tol * reference
  std::vector<double> ref_;     ///< |theta|, or max |theta| when theta == 0
  std::vector<index_t> order_;  ///< indices into theta_ by decreasing preference (which_)

  index_t restarts_ = 0;
  index_t op_applies0_ = 0;
  std::vector<RestartRecord> history_;
};

template <class T>
BlockThickRestartLanczos<T>::BlockThickRestartLanczos(LinearOperator<T>& op,
                                                      LinearOperator<T>* b_inner, Which which,
                                                      const KrylovSchurOptions<T>& options,
                                                      const LanczosSetup& setup,
                                                      const BlockDimensions& dims,
                                                      LanczosTimings& timings)
    : op_(op), b_inner_(b_inner), which_(which), opt_(options), time_(timings) {
  n_ = setup.n;
  d_ = setup.d;
  nev_ = setup.nev;
  p_ = options.block_size;
  ncv_ = dims.ncv;
  keep_ = dims.keep;
  first_probe_ = dims.first_probe;
  HS_ASSERT(dims.feasible && ncv_ % p_ == 0 && keep_ % p_ == 0 && keep_ >= nev_ &&
                keep_ + p_ <= ncv_ && d_ + ncv_ + p_ <= n_,
            "block Lanczos layout n=" << n_ << " d=" << d_ << " p=" << p_ << " nev=" << nev_
                                      << " ncv=" << ncv_ << " keep=" << keep_);
  store_bv_ = options.store_bv && b_inner != nullptr;
  basis_.resize(n_, d_ + ncv_ + p_);
  if (store_bv_) bbasis_.resize(n_, d_ + ncv_ + p_);
  ortho_ = BlockOrthonormalizer<T>(n_, p_, b_inner, store_bv_);
  h_.resize(ncv_, ncv_);
  rlast_.resize(p_, p_);
  op_applies0_ = op.applications();
}

template <class T>
void BlockThickRestartLanczos<T>::load_deflation() {
  if (d_ == 0) return;
  const DenseMatrix<T>& dm = *opt_.deflation;
  std::copy(dm.data(), dm.data() + n_ * d_, basis_.data());
  // The Krylov part of the storage is still unused: scratch for chunks of M*D (recompute mode).
  check_deflation(n_, d_, basis_.data(), mbasis(), b_inner_, vcol(0), ncv_ + p_);
}

/// Start block: the leading columns of `initial`, the rest random; orthonormalized against D and
/// within itself. Dependent user columns other than the first are replaced by random vectors.
template <class T>
void BlockThickRestartLanczos<T>::set_start_block() {
  const DenseMatrix<T>* init = opt_.initial;
  const index_t user = init ? std::min(p_, init->cols()) : 0;
  for (index_t c = 0; c < p_; ++c) {
    T* v = vcol(c);
    if (c < user) {
      std::copy(init->col(c), init->col(c) + n_, v);
      const double norm = linalg::norm2(n_, v);
      if (c == 0 && (!(norm > 0.0) || !std::isfinite(norm)))
        HS_THROW(
            InvalidInput,
            "lanczos_krylov_schur: initial vector must be nonzero and finite, 2-norm " << norm);
      if (!std::isfinite(norm))
        HS_THROW(InvalidInput, "lanczos_krylov_schur: initial block column "
                                   << c << " is not finite, 2-norm " << norm);
    } else {
      linalg::fill_random(v, n_, opt_.seed, static_cast<std::uint64_t>(c));
    }
  }
  next_stream_ = static_cast<std::uint64_t>(p_);
  ortho_.run(basis_.data(), mbasis(), d_, kStartMinRatio, opt_.seed, next_stream_, coef_, rblk_);
  if (user > 0 && ortho_.deficient()[0]) {
    // The user's first column carries the caller's intent: never replace it silently. Only a
    // vector in span(D) or a non-positive M-norm can make it dependent (n - d >= ncv + p).
    if (d_ > 0)
      HS_THROW(InvalidInput,
               "lanczos_krylov_schur: initial vector has no component outside span(deflation) in "
               "the solver inner product, or b_inner is not positive definite");
    HS_THROW(InvalidInput, "lanczos_krylov_schur: start vector has M-norm <= "
                               << kStartMinRatio << " times " << ortho_.before()[0]
                               << ": b_inner is not Hermitian positive definite");
  }
}

/// Block Lanczos steps from the block at Krylov column `first` (0 or keep_) to the end of the
/// basis: V_{j+1} R_j = Op V_j - [D V] C, filling the diagonal block A_j and the couplings R_j
/// (the arrow H(0:first, first:first+p) was set by restart()). At a block boundary m = j + p before
/// the end, V(:, 0:m) with the residual block V_{j+1} satisfies the Krylov-Schur relation as well:
/// from m >= first_probe_ (first cycle) or keep_ + p (later cycles) on, a Rayleigh-Ritz probe there
/// ends the cycle early, with basis size m_ = m, when all nev wanted pairs have converged (returns
/// true; the probe's Rayleigh-Ritz state is then current). One heev of order m per probe, against
/// p operator applications per block.
template <class T>
bool BlockThickRestartLanczos<T>::expand(index_t first) {
  const index_t p = p_;
  for (index_t j = first; j < ncv_; j += p) {
    {
      ScopedTimer t(time_.op);
      op_.apply(vcol(j), n_, vcol(j + p), n_, p);
    }
    {
      ScopedTimer t(time_.orthogonalization);
      ortho_.run(basis_.data(), mbasis(), d_ + j + p, kBreakdownTol, opt_.seed, next_stream_, coef_,
                 rblk_);
      // A_j: Op is M-self-adjoint, so the diagonal block is Hermitian up to rounding.
      for (index_t c = 0; c < p; ++c) {
        h_(j + c, j + c) = T(std::real(coef_(d_ + j + c, c)));
        for (index_t r = c + 1; r < p; ++r) {
          const T v = T(0.5) * (coef_(d_ + j + r, c) + conj_if(coef_(d_ + j + c, r)));
          h_(j + r, j + c) = v;
          h_(j + c, j + r) = conj_if(v);
        }
      }
      if (j + p < ncv_) {
        for (index_t c = 0; c < p; ++c)
          for (index_t r = 0; r <= c; ++r) {
            h_(j + p + r, j + c) = rblk_(r, c);
            h_(j + c, j + p + r) = conj_if(rblk_(r, c));
          }
      }
    }
    m_ = j + p;
    rlast_ = rblk_;
    if (m_ < ncv_ && m_ >= (restarts_ == 0 ? first_probe_ : keep_ + p)) {
      rayleigh_ritz();
      if (wanted_converged() >= nev_) return true;
    }
  }
  return false;
}

template <class T>
void BlockThickRestartLanczos<T>::rayleigh_ritz() {
  ScopedTimer t(time_.rayleigh_ritz);
  const index_t m = m_, p = p_;
  const auto mm = static_cast<fint>(m), pp = static_cast<fint>(p);
  s_.resize(m, m);  // H(0:m, 0:m), stored exactly Hermitian (both triangles)
  for (index_t c = 0; c < m; ++c) std::copy(h_.col(c), h_.col(c) + m, s_.col(c));
  theta_.assign(static_cast<std::size_t>(m), 0.0);
  lapack::heev('V', 'L', mm, s_.data(), mm, theta_.data());

  double max_abs = 0.0;
  for (double th : theta_) max_abs = std::max(max_abs, std::abs(th));

  // Couplings of all Ritz vectors to V_m: R * S(m-p:m, :).
  coupling_.resize(p, m);
  lapack::gemm('N', 'N', pp, mm, pp, T(1), rlast_.data(), pp, s_.data() + (m - p), mm, T(0),
               coupling_.data(), pp);
  est_.assign(static_cast<std::size_t>(m), 0.0);
  conv_.assign(static_cast<std::size_t>(m), 0);
  ref_.assign(static_cast<std::size_t>(m), 0.0);
  for (index_t i = 0; i < m; ++i) {
    double e2 = 0.0;
    for (index_t r = 0; r < p; ++r) e2 += abs2(coupling_(r, i));
    est_[i] = std::sqrt(e2);
    ref_[i] = theta_[i] != 0.0 ? std::abs(theta_[i]) : max_abs;
    conv_[i] = est_[i] <= opt_.tol * ref_[i] ? 1 : 0;
  }

  order_.resize(static_cast<std::size_t>(m));
  std::iota(order_.begin(), order_.end(), index_t{0});
  switch (which_) {
    case Which::SmallestAlgebraic:
      break;  // heev returns ascending eigenvalues
    case Which::LargestAlgebraic:
      std::reverse(order_.begin(), order_.end());
      break;
    case Which::LargestMagnitude:
      // Ties in |theta| prefer the positive value, so the ranking is a strict total order.
      std::stable_sort(order_.begin(), order_.end(), [this](index_t a, index_t b) {
        const double fa = std::abs(theta_[a]), fb = std::abs(theta_[b]);
        return fa > fb || (fa == fb && theta_[a] > theta_[b]);
      });
      break;
  }
}

/// Converged pairs among the nev wanted ones of the current Rayleigh-Ritz step.
template <class T>
index_t BlockThickRestartLanczos<T>::wanted_converged() const {
  index_t nconv = 0;
  for (index_t l = 0; l < nev_; ++l) nconv += conv_[order_[l]] ? 1 : 0;
  return nconv;
}

template <class T>
index_t BlockThickRestartLanczos<T>::record_history() {
  RestartRecord rec;
  rec.restart = restarts_;
  for (index_t l = 0; l < nev_; ++l) {
    const index_t i = order_[l];
    if (conv_[i]) {
      ++rec.nconv;
    } else {
      const double rel =
          ref_[i] > 0.0 ? est_[i] / ref_[i] : std::numeric_limits<double>::infinity();
      rec.max_estimate = std::max(rec.max_estimate, rel);
    }
  }
  rec.op_applies = op_.applications() - op_applies0_;
  history_.push_back(rec);
  return rec.nconv;
}

/// Thick restart keeping the keep_ best Ritz pairs (ranking order); converged pairs whose coupling
/// column is at working precision relative to every wanted pair are locked.
template <class T>
void BlockThickRestartLanczos<T>::restart() {
  ScopedTimer t(time_.restart);
  HS_ASSERT(m_ == ncv_, "restart after an early-stopped cycle (m=" << m_ << ")");
  const index_t m = ncv_, k = keep_, p = p_;
  double min_wanted_ref = std::numeric_limits<double>::infinity();
  for (index_t l = 0; l < nev_; ++l) min_wanted_ref = std::min(min_wanted_ref, ref_[order_[l]]);
  const double lock_tol = kLockTol * min_wanted_ref;
  DenseMatrix<T> sk(m, k), b(p, k);
  for (index_t l = 0; l < k; ++l) {
    const index_t i = order_[l];
    std::copy(s_.col(i), s_.col(i) + m, sk.col(l));
    std::copy(coupling_.col(i), coupling_.col(i) + p, b.col(l));
    if (conv_[i] && est_[i] <= lock_tol) std::fill(b.col(l), b.col(l) + p, T(0));
  }

  rotate_in_place(n_, vcol(0), sk);
  // The residual block continues the basis (k + p <= m: the ranges do not overlap).
  std::copy(vcol(m), vcol(m) + n_ * p, vcol(k));
  if (store_bv_) {
    rotate_in_place(n_, bvcol(0), sk);
    std::copy(bvcol(m), bvcol(m) + n_ * p, bvcol(k));
  }

  // Block arrowhead: diag(theta) with the couplings in rows/columns k..k+p-1.
  h_.set_zero();
  for (index_t l = 0; l < k; ++l) {
    h_(l, l) = T(theta_[order_[l]]);
    for (index_t r = 0; r < p; ++r) {
      h_(k + r, l) = b(r, l);
      h_(l, k + r) = conj_if(b(r, l));
    }
  }
}

template <class T>
HermitianEigResult<T> BlockThickRestartLanczos<T>::finish() {
  HermitianEigResult<T> r;
  const index_t m = m_;
  // The wanted pairs are the nev best ranked; converged ones first, both groups in ranking order.
  std::vector<index_t> sel(order_.begin(), order_.begin() + nev_);
  std::stable_partition(sel.begin(), sel.end(), [this](index_t i) { return conv_[i] != 0; });
  DenseMatrix<T> ssel(m, nev_);
  for (index_t l = 0; l < nev_; ++l) {
    const index_t i = sel[l];
    std::copy(s_.col(i), s_.col(i) + m, ssel.col(l));
    r.theta.push_back(theta_[i]);
    r.estimate.push_back(est_[i]);
    if (conv_[i]) ++r.nconv;
  }
  r.vectors.resize(n_, nev_);
  linalg::multiply(n_, m, nev_, vcol(0), n_, ssel.data(), m, r.vectors.data(), n_);
  r.ritz_all = theta_;
  r.restarts = restarts_;
  r.op_applies = op_.applications() - op_applies0_;
  r.converged = r.nconv >= nev_;
  r.history = std::move(history_);
  return r;
}

template <class T>
HermitianEigResult<T> BlockThickRestartLanczos<T>::run() {
  load_deflation();
  {
    ScopedTimer t(time_.orthogonalization);
    set_start_block();
  }
  index_t first = 0;
  for (;;) {
    if (!expand(first)) rayleigh_ritz();  // an early stop leaves its probe's state current
    const index_t nconv = record_history();
    if (nconv >= nev_ || restarts_ == opt_.max_restarts) break;
    restart();
    first = keep_;
    ++restarts_;
  }
  return finish();
}

}  // namespace

BlockDimensions block_dimensions(const LanczosSetup& setup, index_t p, bool auto_ncv) {
  HS_ASSERT(p >= 1, "block_dimensions: p=" << p);
  const auto round_up = [p](index_t x) { return (x + p - 1) / p * p; };
  BlockDimensions dims;
  const index_t nev_up = round_up(setup.nev);
  const index_t base = std::max(round_up(setup.ncv), nev_up + p);
  dims.ncv = base;
  if (auto_ncv) {
    // Up to kAutoExtraBlocks more blocks, as far as the free space holds them with the residual
    // block (never below base).
    const index_t room = (setup.n_free - p) / p * p;
    dims.ncv = std::max(base, std::min(base + kAutoExtraBlocks * p, room));
  }
  dims.keep = std::clamp((setup.nev + (dims.ncv - setup.nev) / 2) / p * p, nev_up, dims.ncv - p);
  dims.first_probe = base;
  dims.feasible = setup.ncv < setup.n_free && dims.ncv + p <= setup.n_free;
  return dims;
}

template <class T>
HermitianEigResult<T> block_lanczos_krylov_schur_timed(LinearOperator<T>& op,
                                                       LinearOperator<T>* b_inner, Which which,
                                                       const KrylovSchurOptions<T>& options,
                                                       LanczosTimings& timings) {
  HS_ASSERT(options.block_size > 1, "block solver called with block_size " << options.block_size);
  const LanczosSetup setup = validate_lanczos(op, b_inner, options);
  const BlockDimensions dims = block_dimensions(setup, options.block_size, options.ncv == 0);
  if (!dims.feasible) {
    // The free space cannot hold a block basis plus its residual block: one single-vector cycle
    // over the whole free space is exact, and cheap since the free space is that small.
    KrylovSchurOptions<T> single = options;
    single.block_size = 1;
    single.ncv = setup.n_free;
    return lanczos_krylov_schur_timed(op, b_inner, which, single, timings);
  }
  timings = LanczosTimings{};
  Timer total;
  BlockThickRestartLanczos<T> solver(op, b_inner, which, options, setup, dims, timings);
  HermitianEigResult<T> result = solver.run();
  timings.total = total.seconds();
  return result;
}

#define HS_INSTANTIATE_BLOCK_LANCZOS(T)                                            \
  template HermitianEigResult<T> block_lanczos_krylov_schur_timed<T>(              \
      LinearOperator<T>&, LinearOperator<T>*, Which, const KrylovSchurOptions<T>&, \
      LanczosTimings&);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_BLOCK_LANCZOS)
#undef HS_INSTANTIATE_BLOCK_LANCZOS

}  // namespace hs::eigen_detail
