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
/// Per-thread workspaces of the factorization and the check stay within this many bytes in all
/// (fewer threads otherwise).
constexpr double kWorkspaceBytes = 268435456.0;
/// The factorization runs level by level in parallel only with at least this many rows per level
/// on average (a barrier per level).
constexpr double kMinLevelWidth = 32.0;

/// gamma_m = m u / (1 - m u) (Higham), +inf once m u >= 1/2.
double gamma_m(double m) {
  const double mu = m * kUnitRoundoff;
  return mu < 0.5 ? mu / (1.0 - mu) : std::numeric_limits<double>::infinity();
}

/// Lower triangular sparse matrix: CSR rows with sorted columns, the diagonal last in each row, and
/// its column structure (CSC, rows sorted, the diagonal first in each column).
template <class T>
struct LowerMatrix {
  index_t n = 0;
  std::vector<index_t> ptr;
  std::vector<std::int32_t> col;
  std::vector<T> val;
  std::vector<index_t> cptr;
  std::vector<std::int32_t> crow;
  std::vector<T> cval;

  index_t nnz() const { return static_cast<index_t>(col.size()); }
  /// Builds the CSC arrays from the CSR ones (counting sort: rows come out sorted).
  void build_columns() {
    cptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const std::int32_t j : col) ++cptr[static_cast<std::size_t>(j) + 1];
    for (index_t j = 0; j < n; ++j) cptr[j + 1] += cptr[j];
    std::vector<index_t> next(cptr.begin(), cptr.end() - 1);
    crow.resize(col.size());
    cval.resize(val.size());
    for (index_t i = 0; i < n; ++i)
      for (index_t p = ptr[i]; p < ptr[i + 1]; ++p) {
        const index_t q = next[col[p]]++;
        crow[q] = static_cast<std::int32_t>(i);
        cval[q] = val[p];
      }
  }
};

/// The proof's state for one B (see norms.hpp): the scaled lower triangle S of B and the patterns of
/// the fill levels, built once and reused by the attempts.
template <class T>
class IcProver {
 public:
  /// ok() == false: some diagonal entry is not a positive real number (no proof possible).
  explicit IcProver(const CsrMatrix<T>& b) : n_(b.nrows) {
    s_.n = n_;
    s_.ptr.assign(static_cast<std::size_t>(n_) + 1, 0);
    w_.assign(static_cast<std::size_t>(n_), 0.0);
    std::vector<double> d(static_cast<std::size_t>(n_), 0.0);
    for (index_t i = 0; i < n_ && ok_; ++i) {
      for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p)
        if (b.colind[p] == i) {
          d[i] = std::real(b.values[p]);
          if (std::imag(b.values[p]) != 0.0) ok_ = false;
        }
      w_[i] = 1.0 / std::sqrt(d[i]);
      if (!(d[i] > 0.0) || !std::isfinite(w_[i]) || !(w_[i] > 0.0)) ok_ = false;
    }
    if (!ok_) return;
    // s_ij = (w_i b_ij) w_j for j <= i: within gamma_2 |s_ij| of w_i b_ij w_j with the stored w
    // (complex b_ij: two real products per component).
    for (index_t i = 0; i < n_; ++i) {
      for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p) {
        const index_t j = b.colind[p];
        if (j >= i) break;  // sorted columns: the strict lower part comes first
        s_.col.push_back(static_cast<std::int32_t>(j));
        s_.val.push_back((w_[i] * b.values[p]) * w_[j]);
      }
      s_.col.push_back(static_cast<std::int32_t>(i));
      s_.val.push_back(T((w_[i] * d[i]) * w_[i]));
      s_.ptr[i + 1] = static_cast<index_t>(s_.col.size());
    }
    s_.build_columns();
  }

  bool ok() const { return ok_; }

  /// y = S x (S Hermitian, from its stored lower triangle and its columns).
  void apply(const T* x, T* y) const {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n_; ++i) {
      T acc(0);
      for (index_t p = s_.ptr[i]; p < s_.ptr[i + 1]; ++p) acc += s_.val[p] * x[s_.col[p]];
      // strict upper part of row i = conjugated strict lower part of column i (diagonal first)
      for (index_t q = s_.cptr[i] + 1; q < s_.cptr[i + 1]; ++q) acc += conj_if(s_.cval[q]) * x[s_.crow[q]];
      y[i] = acc;
    }
  }

  index_t order() const { return n_; }
  /// Stored entries of the Hermitian S (both triangles).
  double entries() const { return static_cast<double>(2 * s_.nnz() - n_); }

  /// Predicted multiply-adds of an attempt on fill level `level`: the factorization sums |row j|
  /// over the entries (i, j) of the pattern, the check sums |column k|^2 over its columns.
  IcWork work(int level) {
    const LowerMatrix<T>& pattern = level <= 0 ? s_ : level1();
    IcWork w;
    std::vector<index_t> lv(static_cast<std::size_t>(n_), 0);
    for (index_t i = 0; i < n_; ++i) {
      for (index_t p = pattern.ptr[i]; p < pattern.ptr[i + 1]; ++p) {
        const index_t j = pattern.col[p];
        w.factor += static_cast<double>(pattern.ptr[j + 1] - pattern.ptr[j]);
        if (j < i) lv[i] = std::max(lv[i], lv[j] + 1);
      }
      w.levels = std::max(w.levels, lv[i] + 1);
    }
    for (index_t k = 0; k < n_; ++k) {
      const double c = static_cast<double>(pattern.cptr[k + 1] - pattern.cptr[k]);
      w.check += c * c;
    }
    return w;
  }

  /// Bytes an attempt on `level` allocates beyond the stored S: the level's pattern (level 1), the
  /// factor (rows and columns) and the larger of the two per-thread workspaces (each capped at
  /// kWorkspaceBytes).
  double attempt_bytes(int level) {
    const LowerMatrix<T>& pattern = level <= 0 ? s_ : level1();
    const double entry = static_cast<double>(sizeof(T) + sizeof(std::int32_t));
    const double matrix = 2.0 * (static_cast<double>(pattern.nnz()) * entry + 8.0 * static_cast<double>(n_));
    const double threads = static_cast<double>(std::max(1, max_threads()));
    const double factor_ws = threads * static_cast<double>(n_) * (sizeof(T) + 8.0);
    const double check_ws = threads * static_cast<double>(n_) * (2.0 * sizeof(T) + 20.0);
    return (level <= 0 ? 0.0 : matrix) + matrix + std::min(kWorkspaceBytes, std::max(factor_ws, check_ws));
  }

  /// One attempt: incomplete factor of S - shift I on the fill level, then the check.
  IcBound attempt(int level, double shift) {
    IcBound r;
    const LowerMatrix<T>& pattern = level <= 0 ? s_ : level1();
    r.work = work(level);
    r.factor_entries = pattern.nnz();
    Timer t;
    LowerMatrix<T> l;
    const bool ok = factorize(pattern, shift, l);
    r.factor_s = t.seconds();
    if (!ok) {
      r.breakdown = true;
      return r;
    }
    t.reset();
    check(l, shift, r);
    r.check_s = t.seconds();
    return r;
  }

 private:
  /// Pattern of fill level 1: (i, j), j < i, in the pattern of S or with some k < j such that
  /// (i, k) and (j, k) are (Saad, Iterative Methods for Sparse Linear Systems, §10.3.3). Values: S
  /// on its own pattern, 0 on the fill.
  const LowerMatrix<T>& level1() {
    if (l1_built_) return l1_;
    l1_.n = n_;
    l1_.ptr.assign(static_cast<std::size_t>(n_) + 1, 0);
    std::vector<index_t> mark(static_cast<std::size_t>(n_), -1);
    std::vector<std::int32_t> row;
    for (index_t i = 0; i < n_; ++i) {
      row.clear();
      for (index_t p = s_.ptr[i]; p + 1 < s_.ptr[i + 1]; ++p) {  // strict lower part of row i
        mark[s_.col[p]] = i;
        row.push_back(s_.col[p]);
      }
      for (index_t p = s_.ptr[i]; p + 1 < s_.ptr[i + 1]; ++p) {
        const index_t k = s_.col[p];
        for (index_t q = s_.cptr[k] + 1; q < s_.cptr[k + 1]; ++q) {  // rows j > k of column k
          const index_t j = s_.crow[q];
          if (j >= i) break;
          if (mark[j] != i) {
            mark[j] = i;
            row.push_back(static_cast<std::int32_t>(j));
          }
        }
      }
      std::sort(row.begin(), row.end());
      // values: S where it has an entry (both lists sorted), 0 on the fill
      index_t p = s_.ptr[i];
      const index_t send = s_.ptr[i + 1] - 1;
      for (const std::int32_t j : row) {
        while (p < send && s_.col[p] < j) ++p;
        l1_.col.push_back(j);
        l1_.val.push_back(p < send && s_.col[p] == j ? s_.val[p] : T(0));
      }
      l1_.col.push_back(static_cast<std::int32_t>(i));
      l1_.val.push_back(s_.val[send]);
      l1_.ptr[i + 1] = static_cast<index_t>(l1_.col.size());
    }
    l1_.build_columns();
    l1_built_ = true;
    return l1_;
  }

  /// Incomplete Cholesky factor of (pattern values) - shift I on the pattern (row-Crout: l_ij =
  /// (s_ij - sum_{k<j} l_ik conj(l_jk)) / l_jj over the pattern, l_ii = sqrt(s_ii - shift -
  /// sum_k |l_ik|^2)); false when a pivot is not positive. Row i needs the rows j < i of its
  /// pattern only, so the rows of one level (level(i) = 1 + max level(j) over its pattern) are
  /// computed in parallel; every row is computed exactly as in the sequential loop (same operations
  /// in the same order), so the factor does not depend on the thread count. Narrow level
  /// structures (fewer than kMinLevelWidth rows per level on average) run sequentially.
  bool factorize(const LowerMatrix<T>& pattern, double shift, LowerMatrix<T>& l) const {
    l.n = n_;
    l.ptr = pattern.ptr;
    l.col = pattern.col;
    l.val.assign(pattern.val.size(), T(0));
    // Levels and the rows of each level (counting sort, rows ascending within a level).
    std::vector<index_t> level(static_cast<std::size_t>(n_), 0);
    index_t levels = 0;
    for (index_t i = 0; i < n_; ++i) {
      index_t lv = 0;
      for (index_t p = pattern.ptr[i]; p + 1 < pattern.ptr[i + 1]; ++p) lv = std::max(lv, level[pattern.col[p]] + 1);
      level[i] = lv;
      levels = std::max(levels, lv + 1);
    }
    std::vector<index_t> lptr(static_cast<std::size_t>(levels) + 1, 0), rows(static_cast<std::size_t>(n_));
    for (index_t i = 0; i < n_; ++i) ++lptr[level[i] + 1];
    for (index_t v = 0; v < levels; ++v) lptr[v + 1] += lptr[v];
    {
      std::vector<index_t> next(lptr.begin(), lptr.end() - 1);
      for (index_t i = 0; i < n_; ++i) rows[next[level[i]]++] = i;
    }
    const double per_thread = static_cast<double>(n_) * (sizeof(T) + 8.0);
    int threads = std::max(
        1, std::min(max_threads(), static_cast<int>(std::floor(kWorkspaceBytes / std::max(per_thread, 1.0)))));
    if (static_cast<double>(n_) < kMinLevelWidth * static_cast<double>(levels)) threads = 1;
    std::vector<unsigned char> failed(static_cast<std::size_t>(levels), 0);  // a pivot of the level failed
#pragma omp parallel num_threads(threads)
    {
      std::vector<T> x(static_cast<std::size_t>(n_), T(0));
      std::vector<index_t> mark(static_cast<std::size_t>(n_), -1);
      for (index_t v = 0; v < levels; ++v) {
#pragma omp for schedule(dynamic, 16)
        for (index_t r = lptr[v]; r < lptr[v + 1]; ++r) {
          const index_t i = rows[r];
          const index_t end = pattern.ptr[i + 1] - 1;  // the diagonal
          for (index_t p = pattern.ptr[i]; p < end; ++p) {
            x[pattern.col[p]] = pattern.val[p];
            mark[pattern.col[p]] = i;
          }
          double diag = std::real(pattern.val[end]) - shift;
          for (index_t p = pattern.ptr[i]; p < end; ++p) {
            const index_t j = pattern.col[p];
            T sum = x[j];
            const index_t jend = l.ptr[j + 1] - 1;
            for (index_t q = l.ptr[j]; q < jend; ++q)
              if (mark[l.col[q]] == i) sum -= x[l.col[q]] * conj_if(l.val[q]);
            const T lij = sum / std::real(l.val[jend]);
            x[j] = lij;
            l.val[p] = lij;
            diag -= abs2(lij);
          }
          // A failed pivot stays 0: rows of the same level do not read it, and the factorization is
          // abandoned after this level.
          if (!(diag > 0.0) || !std::isfinite(diag)) {
#pragma omp atomic write
            failed[v] = 1;
          } else {
            l.val[end] = T(std::sqrt(diag));
          }
        }  // implicit barrier: the level is complete
        unsigned char stop;
#pragma omp atomic read
        stop = failed[v];
        if (stop) break;  // written before the barrier only: every thread takes the same decision
      }
    }
    if (std::find(failed.begin(), failed.end(), 1) != failed.end()) return false;
    l.build_columns();
    return true;
  }


  /// Rigorous Gershgorin bound of lambda_min(M), M = S - L L^H (exact arithmetic on the stored
  /// numbers): row i of L L^H is accumulated as the sum over k in row i of L of l_ik conj(L(:, k)).
  /// With t_ij the computed s_ij - shift delta_ij - (L L^H)_ij from cnt products,
  ///   |m_ij - shift delta_ij - t_ij| <= e_ij = f gamma_{cnt + 8} (|s_ij| + shift delta_ij + sum_k |l_ik| |l_jk|),
  /// f = 2 real / 4 complex (rounding of the scaled entries - gamma_2 -, the dot product - gamma_cnt,
  /// complex products within sqrt(2) gamma_2 -, the subtractions; a factor 2 of slack), and
  ///   lambda_min(M) >= min_i [(shift + Re t_ii - e_ii)(1 - 4u)
  ///                           - (1 + 2 gamma_{N_i + 4}) sum_{j != i} (|t_ij| + e_ij)]
  /// with the computed sums of N_i nonnegative terms bounded through gamma (|.| of a complex
  /// number within 2u). Rows are independent: the result does not depend on the thread count.
  void check(const LowerMatrix<T>& l, double shift, IcBound& r) const {
    constexpr double kFactor = is_complex_v<T> ? 4.0 : 2.0;
    // Per-thread workspace: acc and the S entry (T), sum of |products| (double), product count
    // (int32), mark (int64); at most 256 MB in all.
    const double per_thread = static_cast<double>(n_) * (2.0 * sizeof(T) + 8.0 + 4.0 + 8.0);
    const int threads = std::max(
        1, std::min(max_threads(), static_cast<int>(std::floor(kWorkspaceBytes / std::max(per_thread, 1.0)))));
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
          const T lik = l.val[q];
          const double alik = std::abs(lik);
          for (index_t p = l.cptr[k]; p < l.cptr[k + 1]; ++p) {
            const index_t j = l.crow[p];
            touch(j);
            acc[j] += lik * conj_if(l.cval[p]);
            aab[j] += alik * std::abs(l.cval[p]);
            ++cnt[j];
          }
        }
        for (index_t p = s_.ptr[i]; p < s_.ptr[i + 1]; ++p) {  // s_ij, j <= i
          touch(s_.col[p]);
          sv[s_.col[p]] = s_.val[p];
        }
        for (index_t p = s_.cptr[i] + 1; p < s_.cptr[i + 1]; ++p) {  // s_ij = conj(s_ji), j > i
          touch(s_.crow[p]);
          sv[s_.crow[p]] = conj_if(s_.cval[p]);
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
  bool ok_ = true;
  std::vector<double> w_;
  LowerMatrix<T> s_;
  LowerMatrix<T> l1_;
  bool l1_built_ = false;
};

}  // namespace

template <class T>
IcBound incomplete_cholesky_bound(const CsrMatrix<T>& b, int level, double shift) {
  IcProver<T> prover(b);
  if (!prover.ok()) HS_THROW(InvalidInput, "incomplete_cholesky_bound: the diagonal of B is not positive");
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

// The attempts (norms.hpp): per fill level the shift c = fraction * theta, the next (smaller)
// fraction only after a breakdown - after a Gershgorin failure a smaller shift lowers the margins
// (it did on every row of the scan in bench/anaint_bproof.cpp), the next level is tried. Each
// attempt is paid from opt.max_seconds by its predicted time (hpd_attempt_seconds) before it runs.
// A retry raising only the failing rows' shifts to 1.5 times their deficit was tried and dropped:
// it never turned a failure into a proof (ss-qa8fk-qa8fm level 0, ss-tem27623 level 1: breakdown).
template <class T>
HpdProof prove_positive_definite(const CsrMatrix<T>& b, const HpdProofOptions& opt) {
  Timer timer;
  HpdProof r;
  const index_t n = b.nrows;
  IcProver<T> prover(b);
  if (n == 0 || !prover.ok()) {
    r.outcome = "nonpositive_diagonal";
    r.seconds = timer.seconds();
    return r;
  }
  const NormEstimate est = lanczos_ends<T>(
      n, [&](const T* x, T* y) { prover.apply(x, y); }, opt.lanczos_steps, opt.seed, /*stream=*/0x68706470ULL);
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
  const int threads = std::max(1, max_threads());
  double left = opt.max_seconds;
  // One attempt if its memory and predicted time fit the limits; false: not run.
  auto run = [&](int level, double shift, IcBound& ib) {
    if (prover.attempt_bytes(level) > opt.max_bytes) {
      r.outcome = "too_large";
      return false;
    }
    const double cost = hpd_attempt_seconds(n, prover.work(level), threads);
    if (cost > left) {
      r.outcome = "too_expensive";
      return false;
    }
    left -= cost;
    ++r.attempts;
    r.level = level;
    r.shift = shift;
    ib = prover.attempt(level, shift);
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
  for (int level = 0; level <= opt.max_level && !r.proven; ++level) {
    for (const double fraction : opt.fractions) {
      IcBound ib;
      if (!run(level, fraction * theta, ib) || !ib.breakdown) break;
    }
    if (r.outcome == "too_expensive" || r.outcome == "too_large") break;  // a higher level costs more
  }
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
