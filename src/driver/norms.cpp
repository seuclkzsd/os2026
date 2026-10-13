#include "driver/norms.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hsolve/error.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace hs::driver {

namespace {

/// Ritz values of the Lanczos tridiagonal (alpha, beta) -> estimate fields.
NormEstimate from_tridiagonal(const std::vector<double>& alpha, const std::vector<double>& beta) {
  NormEstimate est;
  const auto m = static_cast<lapack::fint>(alpha.size());
  if (m == 0) return est;
  // Eigenvalues of the m x m symmetric tridiagonal matrix (dense heev; m <= ~30).
  std::vector<double> t(static_cast<std::size_t>(m) * m, 0.0), theta(static_cast<std::size_t>(m));
  for (lapack::fint i = 0; i < m; ++i) {
    t[i + i * m] = alpha[i];
    if (i + 1 < m) t[(i + 1) + i * m] = t[i + (i + 1) * m] = beta[i];
  }
  lapack::heev('N', 'U', m, t.data(), m, theta.data());
  est.steps = static_cast<int>(m);
  est.ritz_min = theta.front();
  est.ritz_max = theta.back();
  est.ritz_min2 = theta[std::min<std::size_t>(1, theta.size() - 1)];
  est.ritz_max2 = theta[theta.size() - std::min<std::size_t>(2, theta.size())];
  est.norm2 = std::max(std::abs(theta.front()), std::abs(theta.back()));
  return est;
}

/// Plain three-term Lanczos on the Hermitian operator y = apply(x) of order n (see estimate_norm2).
template <class T, class Apply>
NormEstimate lanczos_ends(index_t n, Apply&& apply, int steps, std::uint64_t seed, std::uint64_t stream) {
  steps = static_cast<int>(std::min<index_t>(steps, n));
  std::vector<T> v(static_cast<std::size_t>(n)), v_prev(static_cast<std::size_t>(n), T(0)),
      w(static_cast<std::size_t>(n));
  linalg::fill_random(v.data(), n, seed, stream);
  const double nv = linalg::norm2(n, v.data());
  linalg::scal(n, T(1.0 / nv), v.data());

  std::vector<double> alpha, beta;  // tridiagonal: diag alpha, off-diag beta
  double beta_prev = 0.0;
  for (int j = 0; j < steps; ++j) {
    apply(v.data(), w.data());
    // w -= beta_{j-1} v_{j-1};  alpha_j = Re(v^H w) (Hermitian => real);  w -= alpha_j v
    if (j > 0) linalg::axpy(n, T(-beta_prev), v_prev.data(), w.data());
    const double aj = std::real(linalg::dot(n, v.data(), w.data()));
    linalg::axpy(n, T(-aj), v.data(), w.data());
    alpha.push_back(aj);
    const double bj = linalg::norm2(n, w.data());
    // Breakdown: an invariant subspace was found; its Ritz values are exact eigenvalues.
    const double scale = std::max(std::abs(aj), beta_prev);
    if (bj <= 1e-13 * std::max(scale, 1e-300) || j + 1 == steps) break;
    beta.push_back(bj);
    std::swap(v_prev, v);
    for (index_t i = 0; i < n; ++i) v[i] = w[i] / bj;
    beta_prev = bj;
  }
  return from_tridiagonal(alpha, beta);
}

}  // namespace

template <class T>
NormEstimate estimate_norm2(const CsrMatrix<T>& a, int steps, std::uint64_t seed) {
  const index_t n = a.nrows;
  if (n == 0 || a.nnz() == 0) return NormEstimate{};
  return lanczos_ends<T>(
      n, [&](const T* x, T* y) { spmv(a, x, y); }, steps, seed, /*stream=*/0x6e6f726dULL);
}

template <class T>
NormEstimate estimate_pencil_spectrum(const CsrMatrix<T>& a, const CsrMatrix<T>& b, const SolveFn<T>& solve_b,
                                      int steps, std::uint64_t seed) {
  const index_t n = a.nrows;
  if (n == 0 || a.nnz() == 0) return NormEstimate{};
  steps = static_cast<int>(std::min<index_t>(steps, n));
  // Lanczos for Op = B^{-1} A in <x, y>_B = y^H B x: v_j B-orthonormal, alpha_j = v_j^H A v_j,
  // w = B^{-1} A v_j - alpha_j v_j - beta_{j-1} v_{j-1}, beta_j = ||w||_B.
  std::vector<T> v(static_cast<std::size_t>(n)), v_prev(static_cast<std::size_t>(n), T(0)),
      w(static_cast<std::size_t>(n)), bw(static_cast<std::size_t>(n));
  linalg::fill_random(v.data(), n, seed, /*stream=*/0x70656e63ULL);
  spmv(b, v.data(), bw.data());
  const double nv = std::sqrt(std::max(std::real(linalg::dot(n, v.data(), bw.data())), 1e-300));
  linalg::scal(n, T(1.0 / nv), v.data());

  std::vector<double> alpha, beta;
  double beta_prev = 0.0;
  for (int j = 0; j < steps; ++j) {
    spmv(a, v.data(), w.data());
    const double aj = std::real(linalg::dot(n, v.data(), w.data()));  // v^H A v
    solve_b(w.data());                                                 // w = B^{-1} A v
    if (j > 0) linalg::axpy(n, T(-beta_prev), v_prev.data(), w.data());
    linalg::axpy(n, T(-aj), v.data(), w.data());
    alpha.push_back(aj);
    spmv(b, w.data(), bw.data());
    const double bj = std::sqrt(std::max(std::real(linalg::dot(n, w.data(), bw.data())), 0.0));
    const double scale = std::max(std::abs(aj), beta_prev);
    if (!(bj > 1e-13 * std::max(scale, 1e-300)) || j + 1 == steps) break;
    beta.push_back(bj);
    std::swap(v_prev, v);
    for (index_t i = 0; i < n; ++i) v[i] = w[i] / bj;
    beta_prev = bj;
  }
  return from_tridiagonal(alpha, beta);
}

// ------------------------------------------------------------------ positive definiteness proof

namespace {

constexpr double kUnitRoundoff = 0x1p-53;

// Time model of one proof attempt (hpd_attempt_seconds): the incomplete factorization runs level
// by level on up to `threads` threads (a level of w rows keeps about w / kRowsPerBusyThread busy),
// the check on all threads, plus O(n) setup and a barrier per level (bench/anaint_bproof.cpp).
constexpr double kProofSecondsPerFactorWork = 1.8e-9;
constexpr double kProofSecondsPerCheckWork = 4.5e-9;
constexpr double kProofSecondsPerRow = 8e-8;
constexpr double kProofSecondsPerLevel = 5e-6;
constexpr double kRowsPerBusyThread = 8.0;
/// Per-thread workspaces (pattern construction, factorization, check) stay within this many bytes
/// in all (fewer threads otherwise).
constexpr double kWorkspaceBytes = 268435456.0;
/// The factorization runs level by level in parallel only with at least this many rows per level
/// on average (a barrier per level).
constexpr double kMinLevelWidth = 32.0;

/// gamma_m = m u / (1 - m u) (Higham), +inf once m u >= 1/2.
double gamma_m(double m) {
  const double mu = m * kUnitRoundoff;
  return mu < 0.5 ? mu / (1.0 - mu) : std::numeric_limits<double>::infinity();
}

/// Threads whose per-thread workspaces of `per_thread` bytes fit kWorkspaceBytes together.
int workspace_threads(double per_thread) {
  return std::max(1, std::min(max_threads(), static_cast<int>(std::floor(kWorkspaceBytes / std::max(per_thread, 1.0)))));
}

/// Exclusive prefix sum in place: v[0] = 0 on entry, v[i + 1] = count of row i -> row pointers.
void prefix_sum(std::vector<index_t>& v) {
  for (std::size_t i = 1; i < v.size(); ++i) v[i] += v[i - 1];
}

/// Lower triangular sparse pattern with values: CSR rows with sorted columns, the diagonal last in
/// each row, and the transposed structure (CSC, rows sorted, the diagonal first in each column)
/// with cpos[q] = the CSR position of CSC entry q, so that any values on the pattern can be read
/// column-wise (gather).
template <class T>
struct LowerMatrix {
  index_t n = 0;
  std::vector<index_t> ptr;
  std::vector<std::int32_t> col;
  std::vector<T> val;
  std::vector<index_t> cptr;
  std::vector<std::int32_t> crow;
  std::vector<index_t> cpos;

  index_t nnz() const { return static_cast<index_t>(col.size()); }
  /// The CSC structure by a counting sort of the CSR one (rows come out sorted). O(nnz).
  void build_columns() {
    cptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const std::int32_t j : col) ++cptr[static_cast<std::size_t>(j) + 1];
    prefix_sum(cptr);
    std::vector<index_t> next(cptr.begin(), cptr.end() - 1);
    crow.resize(col.size());
    cpos.resize(col.size());
    for (index_t i = 0; i < n; ++i)
      for (index_t p = ptr[i]; p < ptr[i + 1]; ++p) {
        const index_t q = next[col[p]]++;
        crow[q] = static_cast<std::int32_t>(i);
        cpos[q] = p;
      }
  }
  /// out[q] = v[cpos[q]]: values on the pattern in column order.
  void gather(const std::vector<T>& v, std::vector<T>& out) const {
    const auto m = static_cast<index_t>(cpos.size());
    out.resize(cpos.size());
#pragma omp parallel for schedule(static)
    for (index_t q = 0; q < m; ++q) out[q] = v[cpos[q]];
  }
};

/// What an attempt on one fill level needs besides the pattern: the rows by level of the
/// incomplete factorization (level(i) = 1 + max level(j) over the strict lower part of row i;
/// rows ascending within a level) and the attempt's size (IcWork; integer counts, so the time
/// model and the budget decisions are exact and do not depend on the thread count).
struct Schedule {
  IcWork work;
  std::vector<index_t> lptr, rows;
};

template <class T>
Schedule make_schedule(const LowerMatrix<T>& p) {
  Schedule s;
  const index_t n = p.n;
  std::vector<index_t> level(static_cast<std::size_t>(n), 0);
  index_t levels = 0;
  for (index_t i = 0; i < n; ++i) {
    index_t lv = 0;
    for (index_t q = p.ptr[i]; q + 1 < p.ptr[i + 1]; ++q) lv = std::max(lv, level[p.col[q]] + 1);
    level[i] = lv;
    levels = std::max(levels, lv + 1);
  }
  std::int64_t factor = 0, check = 0;
#pragma omp parallel for reduction(+ : factor) schedule(static)
  for (index_t i = 0; i < n; ++i)
    for (index_t q = p.ptr[i]; q < p.ptr[i + 1]; ++q) factor += p.ptr[p.col[q] + 1] - p.ptr[p.col[q]];
#pragma omp parallel for reduction(+ : check) schedule(static)
  for (index_t k = 0; k < n; ++k) {
    const std::int64_t c = p.cptr[k + 1] - p.cptr[k];
    check += c * c;
  }
  s.work.factor = static_cast<double>(factor);
  s.work.check = static_cast<double>(check);
  s.work.levels = levels;
  s.lptr.assign(static_cast<std::size_t>(levels) + 1, 0);
  s.rows.resize(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) ++s.lptr[level[i] + 1];
  prefix_sum(s.lptr);
  std::vector<index_t> next(s.lptr.begin(), s.lptr.end() - 1);
  for (index_t i = 0; i < n; ++i) s.rows[next[level[i]]++] = i;
  return s;
}

/// Jacobi weights w_i = 1 / sqrt(b_ii) of a Hermitian b; false when some diagonal entry is missing
/// or not a positive real number (no proof possible). O(nnz), parallel over rows.
template <class T>
bool jacobi_weights(const CsrMatrix<T>& b, std::vector<double>& w) {
  const index_t n = b.nrows;
  w.assign(static_cast<std::size_t>(n), 0.0);
  bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
  for (index_t i = 0; i < n; ++i) {
    double d = 0.0;
    bool found = false;
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
      if (b.colind[p] == i) {
        found = true;
        d = std::real(b.values[p]);
        if (std::imag(b.values[p]) != 0.0) bad = true;
      }
    const double wi = 1.0 / std::sqrt(d);
    w[i] = wi;
    if (!found || !(d > 0.0) || !std::isfinite(wi) || !(wi > 0.0)) bad = true;
  }
  return !bad;
}

/// The proof's state for one B (see norms.hpp): the scaled lower triangle S of B, the patterns of
/// the fill levels and their schedules, built once and reused by the attempts.
template <class T>
class IcProver {
 public:
  /// w: jacobi_weights of b (all positive).
  IcProver(const CsrMatrix<T>& b, const std::vector<double>& w) : n_(b.nrows), w_(w) {
    s_.n = n_;
    s_.ptr.assign(static_cast<std::size_t>(n_) + 1, 0);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n_; ++i) {
      index_t lower = 0;
      for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1] && b.colind[p] < i; ++p) ++lower;
      s_.ptr[i + 1] = lower + 1;
    }
    prefix_sum(s_.ptr);
    s_.col.resize(static_cast<std::size_t>(s_.ptr[n_]));
    s_.val.resize(static_cast<std::size_t>(s_.ptr[n_]));
    // s_ij = (w_i b_ij) w_j for j <= i: within gamma_2 |s_ij| of w_i b_ij w_j with the stored w
    // (complex b_ij: two real products per component).
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n_; ++i) {
      index_t q = s_.ptr[i];
      for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p) {
        const index_t j = b.colind[p];
        if (j > i) break;  // sorted columns: the strict lower part, then the diagonal
        s_.col[q] = static_cast<std::int32_t>(j);
        s_.val[q] = j < i ? (w_[i] * b.values[p]) * w_[j] : T((w_[i] * std::real(b.values[p])) * w_[i]);
        ++q;
      }
    }
    s_.build_columns();
    s_.gather(s_.val, s_cval_);
  }

  index_t order() const { return n_; }

  /// The pattern of a fill level (0: S itself; 1: built on first use) and its schedule.
  const LowerMatrix<T>& pattern(int level) { return level <= 0 ? s_ : level1(); }
  const Schedule& schedule(int level) {
    const int k = level <= 0 ? 0 : 1;
    if (!sched_built_[k]) {
      sched_[k] = make_schedule(pattern(level));
      sched_built_[k] = true;
    }
    return sched_[k];
  }

  /// Bytes an attempt on `level` allocates beyond the stored S: the level's pattern (level 1: CSR
  /// and CSC structure, values), the factor's values (row and column order) and the larger of the
  /// per-thread workspaces (at most kWorkspaceBytes). Builds the level's pattern.
  double attempt_bytes(int level) {
    const LowerMatrix<T>& p = pattern(level);
    const double entries = static_cast<double>(p.nnz()), nn = static_cast<double>(n_);
    const double structure = entries * (sizeof(T) + 2.0 * sizeof(std::int32_t) + sizeof(index_t)) +
                             2.0 * (nn + 1.0) * sizeof(index_t);
    const double factor = 2.0 * entries * sizeof(T);
    const double ws = static_cast<double>(std::max(1, max_threads())) * nn * (2.0 * sizeof(T) + 20.0);
    return (level <= 0 ? 0.0 : structure) + factor + std::min(kWorkspaceBytes, ws);
  }

  /// One attempt: incomplete factor of S - shift I on the fill level, then the check.
  IcBound attempt(int level, double shift) {
    IcBound r;
    const LowerMatrix<T>& p = pattern(level);
    const Schedule& s = schedule(level);
    r.work = s.work;
    r.factor_entries = p.nnz();
    Timer t;
    std::vector<T> lval;
    const bool ok = factorize(p, s, shift, lval);
    r.factor_s = t.seconds();
    if (!ok) {
      r.breakdown = true;
      return r;
    }
    t.reset();
    std::vector<T> lcval;
    p.gather(lval, lcval);
    check(p, lval, lcval, shift, r);
    r.check_s = t.seconds();
    return r;
  }

  double pattern_seconds() const { return pattern_s_; }

 private:
  /// Fill of row i on level 1 (unsorted): the strict lower part of row i of S plus every j < i
  /// with some k < j such that (i, k) and (j, k) are in S (Saad, Iterative Methods for Sparse
  /// Linear Systems, §10.3.3). mark: per-thread, n entries.
  void level1_row(index_t i, std::vector<std::int32_t>& mark, std::vector<std::int32_t>& row) const {
    row.clear();
    const auto mi = static_cast<std::int32_t>(i);
    for (index_t p = s_.ptr[i]; p + 1 < s_.ptr[i + 1]; ++p) {
      mark[s_.col[p]] = mi;
      row.push_back(s_.col[p]);
    }
    for (index_t p = s_.ptr[i]; p + 1 < s_.ptr[i + 1]; ++p) {
      const index_t k = s_.col[p];
      for (index_t q = s_.cptr[k] + 1; q < s_.cptr[k + 1]; ++q) {  // rows j > k of column k
        const std::int32_t j = s_.crow[q];
        if (j >= i) break;
        if (mark[j] != mi) {
          mark[j] = mi;
          row.push_back(j);
        }
      }
    }
  }

  /// Pattern of fill level 1 (rows in parallel: sizes, then the sorted rows). Values: S on its own
  /// pattern, 0 on the fill.
  const LowerMatrix<T>& level1() {
    if (l1_built_) return l1_;
    Timer t;
    l1_.n = n_;
    l1_.ptr.assign(static_cast<std::size_t>(n_) + 1, 0);
    const int threads = workspace_threads(static_cast<double>(n_) * sizeof(std::int32_t) * 2.0);
#pragma omp parallel num_threads(threads)
    {
      std::vector<std::int32_t> mark(static_cast<std::size_t>(n_), -1), row;
#pragma omp for schedule(dynamic, 256)
      for (index_t i = 0; i < n_; ++i) {
        level1_row(i, mark, row);
        l1_.ptr[i + 1] = static_cast<index_t>(row.size()) + 1;
      }
    }
    prefix_sum(l1_.ptr);
    l1_.col.resize(static_cast<std::size_t>(l1_.ptr[n_]));
    l1_.val.resize(static_cast<std::size_t>(l1_.ptr[n_]));
#pragma omp parallel num_threads(threads)
    {
      std::vector<std::int32_t> mark(static_cast<std::size_t>(n_), -1), row;
#pragma omp for schedule(dynamic, 256)
      for (index_t i = 0; i < n_; ++i) {
        level1_row(i, mark, row);
        std::sort(row.begin(), row.end());
        // values: S where it has an entry (both lists sorted), 0 on the fill
        index_t p = s_.ptr[i], q = l1_.ptr[i];
        const index_t send = s_.ptr[i + 1] - 1;
        for (const std::int32_t j : row) {
          while (p < send && s_.col[p] < j) ++p;
          l1_.col[q] = j;
          l1_.val[q] = p < send && s_.col[p] == j ? s_.val[p] : T(0);
          ++q;
        }
        l1_.col[q] = static_cast<std::int32_t>(i);
        l1_.val[q] = s_.val[send];
      }
    }
    l1_.build_columns();
    l1_built_ = true;
    pattern_s_ += t.seconds();
    return l1_;
  }

  /// Incomplete Cholesky factor of (pattern values) - shift I on the pattern (row-Crout: l_ij =
  /// (s_ij - sum_{k<j} l_ik conj(l_jk)) / l_jj over the pattern, l_ii = sqrt(s_ii - shift -
  /// sum_k |l_ik|^2)), values in CSR order; false when a pivot is not positive. Row i needs the
  /// rows j < i of its pattern only, so the rows of one level of the schedule are computed in
  /// parallel; every row is computed exactly as in the sequential loop (same operations in the same
  /// order), so the factor does not depend on the thread count. Narrow level structures (fewer than
  /// kMinLevelWidth rows per level on average) run sequentially.
  bool factorize(const LowerMatrix<T>& pattern, const Schedule& s, double shift, std::vector<T>& lval) const {
    lval.assign(pattern.val.size(), T(0));
    const index_t levels = s.work.levels;
    int threads = workspace_threads(static_cast<double>(n_) * (sizeof(T) + 8.0));
    if (static_cast<double>(n_) < kMinLevelWidth * static_cast<double>(levels)) threads = 1;
    std::vector<unsigned char> failed(static_cast<std::size_t>(levels), 0);  // a pivot of the level failed
#pragma omp parallel num_threads(threads)
    {
      std::vector<T> x(static_cast<std::size_t>(n_), T(0));
      std::vector<index_t> mark(static_cast<std::size_t>(n_), -1);
      for (index_t v = 0; v < levels; ++v) {
#pragma omp for schedule(dynamic, 16)
        for (index_t r = s.lptr[v]; r < s.lptr[v + 1]; ++r) {
          const index_t i = s.rows[r];
          const index_t end = pattern.ptr[i + 1] - 1;  // the diagonal
          for (index_t p = pattern.ptr[i]; p < end; ++p) {
            x[pattern.col[p]] = pattern.val[p];
            mark[pattern.col[p]] = i;
          }
          double diag = std::real(pattern.val[end]) - shift;
          for (index_t p = pattern.ptr[i]; p < end; ++p) {
            const index_t j = pattern.col[p];
            T sum = x[j];
            const index_t jend = pattern.ptr[j + 1] - 1;
            for (index_t q = pattern.ptr[j]; q < jend; ++q)
              if (mark[pattern.col[q]] == i) sum -= x[pattern.col[q]] * conj_if(lval[q]);
            const T lij = sum / std::real(lval[jend]);
            x[j] = lij;
            lval[p] = lij;
            diag -= abs2(lij);
          }
          // A failed pivot stays 0: rows of the same level do not read it, and the factorization is
          // abandoned after this level.
          if (!(diag > 0.0) || !std::isfinite(diag)) {
#pragma omp atomic write
            failed[v] = 1;
          } else {
            lval[end] = T(std::sqrt(diag));
          }
        }  // implicit barrier: the level is complete
        unsigned char stop;
#pragma omp atomic read
        stop = failed[v];
        if (stop) break;  // written before the barrier only: every thread takes the same decision
      }
    }
    return std::find(failed.begin(), failed.end(), 1) == failed.end();
  }

  /// Rigorous Gershgorin bound of lambda_min(M), M = S - L L^H (exact arithmetic on the stored
  /// numbers): row i of L L^H is accumulated as the sum over k in row i of L of l_ik conj(L(:, k))
  /// (lcval: L's values in column order). With t_ij the computed s_ij - shift delta_ij - (L L^H)_ij
  /// from cnt products,
  ///   |m_ij - shift delta_ij - t_ij| <= e_ij = f gamma_{cnt + 8} (|s_ij| + shift delta_ij + sum_k |l_ik| |l_jk|),
  /// f = 2 real / 4 complex (rounding of the scaled entries - gamma_2 -, the dot product - gamma_cnt,
  /// complex products within sqrt(2) gamma_2 -, the subtractions; a factor 2 of slack), and
  ///   lambda_min(M) >= min_i [(shift + Re t_ii - e_ii)(1 - 4u)
  ///                           - (1 + 2 gamma_{N_i + 4}) sum_{j != i} (|t_ij| + e_ij)]
  /// with the computed sums of N_i nonnegative terms bounded through gamma (|.| of a complex
  /// number within 2u). Rows are independent: the result does not depend on the thread count.
  void check(const LowerMatrix<T>& l, const std::vector<T>& lval, const std::vector<T>& lcval, double shift,
             IcBound& r) const {
    constexpr double kFactor = is_complex_v<T> ? 4.0 : 2.0;
    // Per-thread workspace: acc and the S entry (T), sum of |products| (double), product count
    // (int32), mark (int64); at most kWorkspaceBytes in all.
    const int threads = workspace_threads(static_cast<double>(n_) * (2.0 * sizeof(T) + 8.0 + 4.0 + 8.0));
    double lower = std::numeric_limits<double>::infinity();
    index_t failing = 0;
#pragma omp parallel num_threads(threads) reduction(min : lower) reduction(+ : failing)
    {
      std::vector<T> acc(static_cast<std::size_t>(n_)), sv(static_cast<std::size_t>(n_));
      std::vector<double> aab(static_cast<std::size_t>(n_));
      std::vector<std::int32_t> cnt(static_cast<std::size_t>(n_));
      std::vector<index_t> mark(static_cast<std::size_t>(n_), -1);
      std::vector<std::int32_t> touched;
#pragma omp for schedule(dynamic, 256)
      for (index_t i = 0; i < n_; ++i) {
        touched.clear();
        auto touch = [&](index_t j) {
          if (mark[j] != i) {
            mark[j] = i;
            acc[j] = T(0);
            sv[j] = T(0);
            aab[j] = 0.0;
            cnt[j] = 0;
            touched.push_back(static_cast<std::int32_t>(j));
          }
        };
        for (index_t q = l.ptr[i]; q < l.ptr[i + 1]; ++q) {
          const index_t k = l.col[q];
          const T lik = lval[q];
          const double alik = std::abs(lik);
          for (index_t p = l.cptr[k]; p < l.cptr[k + 1]; ++p) {
            const index_t j = l.crow[p];
            touch(j);
            acc[j] += lik * conj_if(lcval[p]);
            aab[j] += alik * std::abs(lcval[p]);
            ++cnt[j];
          }
        }
        for (index_t p = s_.ptr[i]; p < s_.ptr[i + 1]; ++p) {  // s_ij, j <= i
          touch(s_.col[p]);
          sv[s_.col[p]] = s_.val[p];
        }
        for (index_t p = s_.cptr[i] + 1; p < s_.cptr[i + 1]; ++p) {  // s_ij = conj(s_ji), j > i
          touch(s_.crow[p]);
          sv[s_.crow[p]] = conj_if(s_cval_[p]);
        }
        double diag = -std::numeric_limits<double>::infinity(), off = 0.0;
        for (const std::int32_t j : touched) {
          const double dj = j == i ? shift : 0.0;
          const T t = sv[j] - T(dj) - acc[j];
          const double e =
              kFactor * gamma_m(static_cast<double>(cnt[j]) + 8.0) * (std::abs(sv[j]) + dj + aab[j]);
          if (j == i)
            diag = shift + std::real(t) - e;
          else
            off += std::abs(t) + e;
        }
        const double margin = diag * (1.0 - 4.0 * kUnitRoundoff) -
                              (1.0 + 2.0 * gamma_m(static_cast<double>(touched.size()) + 4.0)) * off;
        const double m = std::isfinite(margin) ? margin : -std::numeric_limits<double>::infinity();
        lower = std::min(lower, m);
        if (!(m > 0.0)) ++failing;
      }
    }
    r.lower_bound = lower;
    r.failing_rows = failing;
  }

  index_t n_ = 0;
  const std::vector<double>& w_;
  LowerMatrix<T> s_;
  std::vector<T> s_cval_;  ///< S's lower values in column order
  LowerMatrix<T> l1_;
  bool l1_built_ = false;
  Schedule sched_[2];
  bool sched_built_[2] = {false, false};
  double pattern_s_ = 0.0;
};

}  // namespace

template <class T>
IcBound incomplete_cholesky_bound(const CsrMatrix<T>& b, int level, double shift) {
  std::vector<double> w;
  if (!jacobi_weights(b, w)) HS_THROW(InvalidInput, "incomplete_cholesky_bound: the diagonal of B is not positive");
  IcProver<T> prover(b, w);
  return prover.attempt(level, shift);
}

double hpd_attempt_seconds(index_t n, const IcWork& w, int threads) {
  threads = std::max(1, threads);
  const double nn = static_cast<double>(n), levels = static_cast<double>(std::max<index_t>(w.levels, 1));
  const bool parallel = threads > 1 && nn >= kMinLevelWidth * levels;
  const double width = parallel ? std::min(static_cast<double>(threads), nn / levels / kRowsPerBusyThread) : 1.0;
  return kProofSecondsPerFactorWork * w.factor / std::max(1.0, width) + kProofSecondsPerCheckWork * w.check / threads +
         kProofSecondsPerRow * nn + (parallel ? kProofSecondsPerLevel * levels : 0.0);
}

// The attempts (norms.hpp): fill level 1 first - on every data case the proof succeeded at, level
// 1 proved it with the first shift (fraction 0.9), while level 0 failed on two of the four
// (ss-qa8fk-qa8fm, syn-ghep-zero-cluster-complex-1e4: dropped fill above the shift in some rows)
// and was at most 10% cheaper elsewhere (bench/anaint_bproof.cpp, 16 threads) -, level 0 only
// when level 1 does not fit the time or memory limit. Per level the shift c = fraction * theta,
// the next (smaller) fraction only after a breakdown: after a Gershgorin failure a smaller shift
// lowers the margins (it did on every row of the scan). Each attempt is paid from opt.max_seconds
// by its predicted time (hpd_attempt_seconds) before it runs. A retry raising only the failing
// rows' shifts to 1.5 times their deficit was tried and dropped: it never turned a failure into a
// proof (ss-qa8fk-qa8fm level 0, ss-tem27623 level 1: breakdown).
template <class T>
HpdProof prove_positive_definite(const CsrMatrix<T>& b, const HpdProofOptions& opt) {
  Timer timer;
  HpdProof r;
  const index_t n = b.nrows;
  std::vector<double> w;
  if (n == 0 || !jacobi_weights(b, w)) {
    r.outcome = "nonpositive_diagonal";
    r.seconds = timer.seconds();
    return r;
  }
  // Lanczos on S = W B W applied as w .* (B (w .* x)): no copy of S before theta says whether an
  // attempt is worth building it.
  std::vector<T> wx(static_cast<std::size_t>(n));
  auto apply = [&](const T* x, T* y) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) wx[i] = w[i] * x[i];
    spmv(b, wx.data(), y);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) y[i] *= w[i];
  };
  const NormEstimate est = lanczos_ends<T>(n, apply, opt.lanczos_steps, opt.seed, /*stream=*/0x68706470ULL);
  r.lanczos_s = timer.seconds();
  r.theta_min = est.ritz_min;
  r.theta_max = est.ritz_max;
  if (est.steps > 0 && est.ritz_min < -kIndefiniteRitzScaled * est.norm2) {
    r.indefinite = true;
    r.outcome = "indefinite";
    r.seconds = timer.seconds();
    return r;
  }
  if (!(est.ritz_min >= opt.min_theta)) {
    r.outcome = "nearly_singular";
    r.seconds = timer.seconds();
    return r;
  }
  const double theta = est.ritz_min;
  Timer ts;
  IcProver<T> prover(b, w);
  r.setup_s = ts.seconds();
  const int threads = std::max(1, max_threads());
  double left = opt.max_seconds;
  // One attempt if its memory and predicted time fit the limits; false: not run.
  auto run = [&](int level, double shift, IcBound& ib) {
    if (prover.attempt_bytes(level) > opt.max_bytes) {
      r.outcome = "too_large";
      return false;
    }
    const double cost = hpd_attempt_seconds(n, prover.schedule(level).work, threads);
    if (cost > left) {
      r.outcome = "too_expensive";
      return false;
    }
    left -= cost;
    ++r.attempts;
    r.level = level;
    r.shift = shift;
    Timer ta;
    ib = prover.attempt(level, shift);
    r.attempts_s += ta.seconds();
    r.factor_entries = ib.factor_entries;
    if (ib.breakdown) {
      r.outcome = "breakdown";
    } else if (ib.lower_bound > 0.0) {
      r.proven = true;
      r.lower_bound = ib.lower_bound;
      r.outcome = "proven";
    } else {
      r.outcome = "gershgorin";
    }
    return true;
  };
  for (const int level : opt.levels) {
    bool ran = false;
    for (const double fraction : opt.fractions) {
      IcBound ib;
      if (!run(level, fraction * theta, ib)) break;
      ran = true;
      if (!ib.breakdown) break;
    }
    if (r.proven || ran) break;  // a lower level only where this one did not fit the limits
  }
  r.pattern_s = prover.pattern_seconds();
  r.seconds = timer.seconds();
  return r;
}

#define HS_INSTANTIATE_NORMS(T)                                                                     \
  template NormEstimate estimate_norm2<T>(const CsrMatrix<T>&, int, std::uint64_t);                 \
  template NormEstimate estimate_pencil_spectrum<T>(const CsrMatrix<T>&, const CsrMatrix<T>&,       \
                                                    const SolveFn<T>&, int, std::uint64_t);  \
  template HpdProof prove_positive_definite<T>(const CsrMatrix<T>&, const HpdProofOptions&);         \
  template IcBound incomplete_cholesky_bound<T>(const CsrMatrix<T>&, int, double);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_NORMS)
#undef HS_INSTANTIATE_NORMS

}  // namespace hs::driver
