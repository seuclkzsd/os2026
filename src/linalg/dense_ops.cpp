#include "linalg/dense_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"

namespace hs::linalg {

namespace {

// SplitMix64: counter-based, so entry i is a pure function of (seed, stream, i) and the fill
// can be parallelised without changing the sequence.
inline std::uint64_t splitmix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline double uniform_pm1(std::uint64_t key) {
  // 53 random bits -> [0, 1) -> [-1, 1)
  return static_cast<double>(splitmix64(key) >> 11) * 0x1.0p-53 * 2.0 - 1.0;
}

// ---- deterministic reductions -----------------------------------------------------------------
// Rows are cut into blocks of a fixed size, consecutive blocks are grouped into at most kMaxSlots
// slots, and each slot is reduced by one thread in block order into its own accumulator. The slot
// accumulators are then added serially in slot order. Every boundary depends only on the sizes,
// never on the thread count, so the result is bitwise reproducible for any number of threads.
// Within a block the sums use kLanes explicit accumulators (vectorizable without reassociation:
// the compiler may not reorder floating-point additions on its own).

constexpr int kLanes = 8;
constexpr index_t kMaxSlots = 256;
constexpr index_t kVecBlock = 1024;  ///< rows per block of dot / norm2
/// Below this many scalars of reduction work a call runs on one thread (results are unchanged).
constexpr index_t kParallelMinWork = 32768;

inline double lane_sum(const double* a) {
  return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

/// sum_t x[t] * y[t] over len doubles: Re(x^H y) for interleaved complex data as well.
inline double ddot(index_t len, const double* x, const double* y) {
  double a[kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int l = 0; l < kLanes; ++l) a[l] += x[t + l] * y[t + l];
  for (int l = 0; t + l < len; ++l) a[l] += x[t + l] * y[t + l];
  return lane_sum(a);
}

/// x^H y over len complex entries stored as interleaved doubles: re = sum_t x[t] y[t] and
/// im = sum_i (x_re y_im - x_im y_re), the latter from lane products x[t] * y[t ^ 1] with the odd
/// lanes subtracted (len * 2 is even, so the remainder holds whole pairs).
inline void zdotc(index_t len, const double* x, const double* y, double& re, double& im) {
  const index_t m = 2 * len;
  double a[kLanes] = {}, b[kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= m; t += kLanes)
    for (int l = 0; l < kLanes; ++l) {
      a[l] += x[t + l] * y[t + l];
      b[l] += x[t + l] * y[t + (l ^ 1)];
    }
  for (int l = 0; t + l < m; ++l) {
    a[l] += x[t + l] * y[t + l];
    b[l] += x[t + l] * y[t + (l ^ 1)];
  }
  re = lane_sum(a);
  im = ((b[0] - b[1]) + (b[4] - b[5])) + ((b[2] - b[3]) + (b[6] - b[7]));
}

/// Fixed partition of [0, n) into blocks of `block` rows grouped into slots.
struct Partition {
  index_t n = 0, block = 1, nblocks = 0, nslots = 0;
  Partition(index_t rows, index_t block_rows, index_t max_slots)
      : n(rows), block(std::max<index_t>(block_rows, 1)) {
    nblocks = (n + block - 1) / block;
    nslots = std::min(nblocks, std::max<index_t>(max_slots, 1));
  }
  index_t first_block(index_t s) const { return s * nblocks / nslots; }
  index_t row(index_t b) const { return std::min(b * block, n); }
};

/// out[0, width) = sum over slots of the per-slot accumulators filled by
/// block_fn(row0, rows, acc) (acc: the slot's `width` doubles, zero-initialised; blocks of one
/// slot arrive in order on one thread). `work` (scalars touched) only decides whether to go
/// parallel. Deterministic for any thread count.
template <class BlockFn>
void reduce_blocks(const Partition& part, index_t width, index_t work, double* out,
                   BlockFn&& block_fn) {
  std::fill(out, out + width, 0.0);
  if (part.nblocks == 0) return;
  if (part.nslots == 1) {
    for (index_t b = 0; b < part.nblocks; ++b)
      block_fn(part.row(b), part.row(b + 1) - part.row(b), out);
    return;
  }
  std::vector<double> acc(static_cast<std::size_t>(part.nslots * width), 0.0);
#pragma omp parallel for schedule(static) if (work >= kParallelMinWork)
  for (index_t s = 0; s < part.nslots; ++s) {
    double* slot = acc.data() + s * width;
    for (index_t b = part.first_block(s); b < part.first_block(s + 1); ++b)
      block_fn(part.row(b), part.row(b + 1) - part.row(b), slot);
  }
  for (index_t s = 0; s < part.nslots; ++s)
    for (index_t j = 0; j < width; ++j) out[j] += acc[static_cast<std::size_t>(s * width + j)];
}

/// Scalars viewed as doubles (std::complex<double> is layout-compatible with double[2]).
template <class T>
const double* dbl(const T* p) {
  return reinterpret_cast<const double*>(p);
}
template <class T>
double* dbl(T* p) {
  return reinterpret_cast<double*>(p);
}
template <class T>
constexpr index_t kDoubles = is_complex_v<T> ? 2 : 1;

// ---- CGS kernels on one row block --------------------------------------------------------------
// All operate on `len` doubles of each column (len = rows, or 2 * rows interleaved for complex
// data) with column stride ldp / ldv in doubles. Columns are processed in register groups (four
// real or two complex); every column is summed with the same lane structure whichever group it
// falls in.

/// Imaginary part of a conjugated complex dot product from the cross-lane accumulators b
/// (b[l] += p[t + l] * x[t + (l ^ 1)]): + even lanes (p_re x_im), - odd lanes (p_im x_re).
inline double cross_sum(const double* b) {
  return ((b[0] - b[1]) + (b[4] - b[5])) + ((b[2] - b[3]) + (b[6] - b[7]));
}

// The kernels below were tuned on Ice Lake (bench/ortho_kernels): two-dimensional accumulator
// arrays a[column][lane] over restrict pointers vectorize well with GCC, separately named arrays
// per column did not (20 instead of 35 GB/s); complex updates vectorize in the explicit pair form.

/// acc[c] += P(:, c)^T x for C columns (real; len rows).
template <int C>
inline void proj_real_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  double a[C][kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < kLanes; ++l) a[c][l] += p[c * ldp + t + l] * x[t + l];
  for (int c = 0; c < C; ++c)
    for (int l = 0; t + l < len; ++l) a[c][l] += p[c * ldp + t + l] * x[t + l];
  for (int c = 0; c < C; ++c) acc[c] += lane_sum(a[c]);
}

/// (acc[2c], acc[2c+1]) += P(:, c)^H x for C columns (complex interleaved; len = 2 * rows).
template <int C>
inline void proj_cplx_cols(index_t len, const double* __restrict p, index_t ldp, const double* __restrict x,
                           double* __restrict acc) {
  double a[C][kLanes] = {}, b[C][kLanes] = {};
  index_t t = 0;
  for (; t + kLanes <= len; t += kLanes)
    for (int c = 0; c < C; ++c)
      for (int l = 0; l < kLanes; ++l) {
        a[c][l] += p[c * ldp + t + l] * x[t + l];
        b[c][l] += p[c * ldp + t + l] * x[t + (l ^ 1)];
      }
  for (int c = 0; c < C; ++c)
    for (int l = 0; t + l < len; ++l) {  // len is even: the remainder holds whole pairs
      a[c][l] += p[c * ldp + t + l] * x[t + l];
      b[c][l] += p[c * ldp + t + l] * x[t + (l ^ 1)];
    }
  for (int c = 0; c < C; ++c) {
    acc[2 * c] += lane_sum(a[c]);
    acc[2 * c + 1] += cross_sum(b[c]);
  }
}

/// x -= V(:, 0:C) c (real; len rows).
template <int C>
inline void update_real_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  double cc[C];
  for (int j = 0; j < C; ++j) cc[j] = c[j];
#pragma omp simd
  for (index_t t = 0; t < len; ++t) {
    double s = 0.0;
    for (int j = 0; j < C; ++j) s += v[j * ldv + t] * cc[j];
    x[t] -= s;
  }
}

/// x -= V(:, 0:C) c (complex interleaved: len = 2 * rows doubles, c interleaved).
template <int C>
inline void update_cplx_cols(index_t len, const double* __restrict v, index_t ldv, const double* __restrict c,
                             double* __restrict x) {
  double cr[C], ci[C];
  for (int j = 0; j < C; ++j) {
    cr[j] = c[2 * j];
    ci[j] = c[2 * j + 1];
  }
  const index_t m = len / 2;
#pragma omp simd
  for (index_t i = 0; i < m; ++i) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < C; ++j) {
      const double a = v[j * ldv + 2 * i], b = v[j * ldv + 2 * i + 1];
      sr += a * cr[j] - b * ci[j];
      si += b * cr[j] + a * ci[j];
    }
    x[2 * i] -= sr;
    x[2 * i + 1] -= si;
  }
}

/// acc[j] += P(:, j)^T x, j < k (real).
inline void proj_real(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) proj_real_cols<8>(len, p + j * ldp, ldp, x, acc + j);
  if (j + 4 <= k) {
    proj_real_cols<4>(len, p + j * ldp, ldp, x, acc + j);
    j += 4;
  }
  for (; j < k; ++j) proj_real_cols<1>(len, p + j * ldp, ldp, x, acc + j);
}

/// (acc[2j], acc[2j+1]) += P(:, j)^H x, j < k (complex).
inline void proj_cplx(index_t len, index_t k, const double* p, index_t ldp, const double* x, double* acc) {
  index_t j = 0;
  for (; j + 4 <= k; j += 4) proj_cplx_cols<4>(len, p + j * ldp, ldp, x, acc + 2 * j);
  for (; j < k; ++j) proj_cplx_cols<1>(len, p + j * ldp, ldp, x, acc + 2 * j);
}

/// x -= V c (real).
inline void update_real(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + 8 <= k; j += 8) update_real_cols<8>(len, v + j * ldv, ldv, c + j, x);
  if (j + 4 <= k) {
    update_real_cols<4>(len, v + j * ldv, ldv, c + j, x);
    j += 4;
  }
  for (; j < k; ++j) update_real_cols<1>(len, v + j * ldv, ldv, c + j, x);
}

/// x -= V c (complex).
inline void update_cplx(index_t len, index_t k, const double* v, index_t ldv, const double* c, double* x) {
  index_t j = 0;
  for (; j + 2 <= k; j += 2) update_cplx_cols<2>(len, v + j * ldv, ldv, c + 2 * j, x);
  for (; j < k; ++j) update_cplx_cols<1>(len, v + j * ldv, ldv, c + 2 * j, x);
}

/// Bytes of the n x k operands per row block. A fused sweep (update, then projection of the
/// updated rows) re-reads its block, which must still be in L2 (1.25 MB per core); a streaming
/// sweep reads every entry once and prefers long per-column runs for the hardware prefetcher.
/// Measured with n = 2e5, k = 16..160, 2 threads: fused blocks of 64-330 KB run fastest (1-2 MB
/// lose 20-40%); streaming sweeps lose ~30% with 2 KB column runs (128 complex rows) against
/// 8-16 KB runs.
constexpr index_t kFusedTileBytes = 128 * 1024;
constexpr index_t kStreamTileBytes = 512 * 1024;

/// Rows per block of a CGS sweep over `matrices` distinct n x k operands (a function of the
/// sizes only: deterministic partition).
inline index_t cgs_block_rows(index_t k, index_t scalar_bytes, index_t matrices, bool fused) {
  const index_t row_bytes = std::max<index_t>(k, 1) * scalar_bytes * std::max<index_t>(matrices, 1);
  const index_t rows = (fused ? kFusedTileBytes : kStreamTileBytes) / row_bytes;
  return std::clamp<index_t>(rows / 32 * 32, 128, 4096);
}

/// One sweep over the rows of the n x k operands: optionally (in this order, per row block)
///   update:     w -= V c and, when bv != nullptr, bw -= BV c         (c != nullptr)
///   projection: proj = P^H x                                         (p != nullptr)
///   norm:       *norm = Re(na^H nb)                                  (na != nullptr)
/// with all pointers to doubles (complex interleaved), P / x / na / nb read after the update.
struct Sweep {
  const double* v = nullptr;
  index_t ldv = 0;
  const double* bv = nullptr;
  index_t ldbv = 0;
  double* w = nullptr;
  double* bw = nullptr;
  const double* c = nullptr;
  const double* p = nullptr;
  index_t ldp = 0;
  const double* x = nullptr;
  const double* na = nullptr;
  const double* nb = nullptr;
};

template <bool Cx>
void run_sweep(index_t n, index_t k, const Sweep& s, double* proj, double* norm) {
  constexpr index_t d = Cx ? 2 : 1;
  const index_t kd = k * d;
  const index_t width = (s.p ? kd : 0) + (s.na ? 1 : 0);
  // Distinct n x k operands: V (and BV) of the update; a fused projection re-reads one of them.
  const bool fused = s.c && s.p;
  const index_t mats = s.c ? (s.bv ? 2 : 1) : (s.p ? 1 : 0);
  const Partition part(n, cgs_block_rows(k, 8 * d, mats, fused), kMaxSlots);
  const index_t work = n * d * std::max<index_t>(k * (mats + (fused ? 1 : 0)), 1);
  auto block = [&](index_t r0, index_t rows, double* acc) {
    const index_t o = r0 * d, len = rows * d;
    if (s.c) {
      if constexpr (Cx) {
        update_cplx(len, k, s.v + o, s.ldv, s.c, s.w + o);
        if (s.bv) update_cplx(len, k, s.bv + o, s.ldbv, s.c, s.bw + o);
      } else {
        update_real(len, k, s.v + o, s.ldv, s.c, s.w + o);
        if (s.bv) update_real(len, k, s.bv + o, s.ldbv, s.c, s.bw + o);
      }
    }
    if (s.p) {
      if constexpr (Cx) {
        proj_cplx(len, k, s.p + o, s.ldp, s.x + o, acc);
      } else {
        proj_real(len, k, s.p + o, s.ldp, s.x + o, acc);
      }
    }
    if (s.na) acc[width - 1] += ddot(len, s.na + o, s.nb + o);
  };
  if (width == 0) {
    // Update only: no reduction, any row distribution gives the same result.
    const index_t nb = part.nblocks;
#pragma omp parallel for schedule(static) if (work >= kParallelMinWork)
    for (index_t b = 0; b < nb; ++b) block(part.row(b), part.row(b + 1) - part.row(b), nullptr);
    return;
  }
  std::vector<double> out(static_cast<std::size_t>(width));
  reduce_blocks(part, width, work, out.data(), block);
  if (s.p) std::copy(out.begin(), out.begin() + kd, proj);
  if (s.na) *norm = out[static_cast<std::size_t>(width - 1)];
}

/// sqrt of a squared norm, keeping NaN (std::max returns its first argument for NaN).
inline double root(double m2) { return std::sqrt(std::max(m2, 0.0)); }

}  // namespace

template <class T>
void fill_random(T* x, index_t n, std::uint64_t seed, std::uint64_t stream) {
  const std::uint64_t base = splitmix64(seed ^ splitmix64(stream + 0x51ED270B27A1F3ULL));
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    const std::uint64_t k = base + 2 * static_cast<std::uint64_t>(i);
    if constexpr (is_complex_v<T>) {
      x[i] = T(uniform_pm1(k), uniform_pm1(k + 1));
    } else {
      x[i] = uniform_pm1(k);
    }
  }
}

template <class T>
T dot(index_t n, const T* x, const T* y) {
  const Partition part(n, kVecBlock, kMaxSlots);
  const double* xd = dbl(x);
  const double* yd = dbl(y);
  if constexpr (is_complex_v<T>) {
    double s[2];
    reduce_blocks(part, 2, 2 * n, s, [&](index_t r0, index_t rows, double* acc) {
      double re, im;
      zdotc(rows, xd + 2 * r0, yd + 2 * r0, re, im);
      acc[0] += re;
      acc[1] += im;
    });
    return T(s[0], s[1]);
  } else {
    double s;
    reduce_blocks(part, 1, n, &s,
                  [&](index_t r0, index_t rows, double* acc) { acc[0] += ddot(rows, xd + r0, yd + r0); });
    return s;
  }
}

template <class T>
double norm2(index_t n, const T* x) {
  const Partition part(n, kVecBlock, kMaxSlots);
  const double* xd = dbl(x);
  constexpr index_t d = kDoubles<T>;
  double s;
  reduce_blocks(part, 1, d * n, &s, [&](index_t r0, index_t rows, double* acc) {
    acc[0] += ddot(d * rows, xd + d * r0, xd + d * r0);
  });
  return std::sqrt(s);
}

template <class T>
void axpy(index_t n, T alpha, const T* x, T* y) {
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

template <class T>
void scal(index_t n, T alpha, T* x) {
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) x[i] *= alpha;
}

template <class T>
double orthogonalize_cgs2(index_t n, index_t k, const T* v, index_t ldv, const T* bv, index_t ldbv,
                          T* w, T* bw, T* h, const ApplyFn<T>* apply_m, const OrthoOptions& options,
                          OrthoInfo* info) {
  const bool stored = bv != nullptr;
  const bool recompute = !stored && apply_m != nullptr;
  HS_ASSERT(!stored || bw != nullptr, "stored mode needs bw = M*w");
  HS_ASSERT(!recompute || bw != nullptr, "recompute mode needs a bw workspace");
  HS_ASSERT(stored || recompute || bw == nullptr, "bw given without bv or apply_m");
  if (!(options.eta >= 0.0 && options.eta <= 1.0))
    HS_THROW(InvalidInput, "orthogonalize_cgs2: eta must lie in [0, 1], got " << options.eta);
  constexpr bool cx = is_complex_v<T>;
  constexpr index_t d = kDoubles<T>;
  const index_t kd = k * d;
  if (h) std::fill(h, h + k, T(0));
  OrthoInfo inf;
  const double* wd = dbl(static_cast<const T*>(w));
  const double* bwd = bw ? dbl(static_cast<const T*>(bw)) : wd;  // M w (M = I: w itself)
  // Squared norm of the current w: Re(w^H M w) with bw == M w, or ||w||^2.
  auto sq_norm = [&]() {
    double m2 = 0.0;
    Sweep s;
    s.na = wd;
    s.nb = bwd;
    run_sweep<cx>(n, 0, s, nullptr, &m2);
    return m2;
  };
  auto finish = [&](double m2) {
    inf.norm_after = root(m2);
    if (info) *info = inf;
    if (bw) return m2 > 0.0 ? std::sqrt(m2) : 0.0;
    return std::sqrt(m2);
  };
  if (k == 0) {
    if (recompute) (*apply_m)(w, bw);
    const double m2 = sq_norm();
    inf.norm_before = inf.norm_pass1 = root(m2);
    return finish(m2);
  }
  std::vector<double> c1(static_cast<std::size_t>(kd)), c2(static_cast<std::size_t>(kd));
  auto accumulate = [&](const std::vector<double>& c) {
    if (h) {
      double* hd = dbl(h);
      for (index_t j = 0; j < kd; ++j) hd[j] += c[static_cast<std::size_t>(j)];
    }
  };
  // Projections use M V: BV against w in stored mode, V against M w otherwise.
  Sweep proj;
  proj.p = stored ? dbl(bv) : dbl(v);
  proj.ldp = (stored ? ldbv : ldv) * d;
  proj.x = recompute ? bwd : wd;
  double m0 = 0.0, m1 = 0.0, m2 = 0.0;

  // Pass 1 projection with the norm before: c1 = (M V)^H w, m0 = w^H M w.
  if (recompute) (*apply_m)(w, bw);
  proj.na = wd;
  proj.nb = bwd;
  run_sweep<cx>(n, k, proj, c1.data(), &m0);
  inf.norm_before = root(m0);
  accumulate(c1);
  inf.passes = 1;

  // Pass 1 update w -= V c1 (bw -= BV c1), fused with the pass 2 projection and the new norm
  // while each row block is in cache. Recompute mode needs M w of the whole updated w first.
  Sweep upd;
  upd.v = dbl(v);
  upd.ldv = ldv * d;
  upd.w = dbl(w);
  if (stored) {
    upd.bv = dbl(bv);
    upd.ldbv = ldbv * d;
    upd.bw = dbl(bw);
  }
  upd.c = c1.data();
  if (recompute) {
    run_sweep<cx>(n, k, upd, nullptr, nullptr);
    (*apply_m)(w, bw);
    m1 = sq_norm();
  } else {
    Sweep fused = upd;
    fused.p = proj.p;
    fused.ldp = proj.ldp;
    fused.x = proj.x;
    fused.na = wd;
    fused.nb = bwd;
    run_sweep<cx>(n, k, fused, c2.data(), &m1);
  }
  inf.norm_pass1 = root(m1);
  // DGKS: no significant cancellation in pass 1 => w is orthogonal to working precision.
  const bool skip = options.reorth == Reorth::IfNeeded && inf.norm_pass1 > 0.0 &&
                    inf.norm_pass1 >= options.eta * inf.norm_before;
  if (skip) return finish(m1);

  // Pass 2.
  if (recompute) {
    proj.na = proj.nb = nullptr;
    run_sweep<cx>(n, k, proj, c2.data(), nullptr);
  }
  accumulate(c2);
  inf.passes = 2;
  upd.c = c2.data();
  if (recompute) {
    run_sweep<cx>(n, k, upd, nullptr, nullptr);
    (*apply_m)(w, bw);
    m2 = sq_norm();
  } else {
    upd.na = wd;
    upd.nb = bwd;
    run_sweep<cx>(n, k, upd, nullptr, &m2);
  }
  return finish(m2);
}

template <class T>
void gram(index_t n, index_t k, const T* v, index_t ldv, const T* w, index_t ldw, T* c, index_t ldc) {
  lapack::gemm('C', 'N', static_cast<lapack::fint>(k), static_cast<lapack::fint>(k),
               static_cast<lapack::fint>(n), T(1), v, static_cast<lapack::fint>(ldv), w,
               static_cast<lapack::fint>(ldw), T(0), c, static_cast<lapack::fint>(ldc));
}

template <class T>
void multiply(index_t n, index_t k, index_t m, const T* v, index_t ldv, const T* s, index_t lds, T* y,
              index_t ldy) {
  lapack::gemm('N', 'N', static_cast<lapack::fint>(n), static_cast<lapack::fint>(m),
               static_cast<lapack::fint>(k), T(1), v, static_cast<lapack::fint>(ldv), s,
               static_cast<lapack::fint>(lds), T(0), y, static_cast<lapack::fint>(ldy));
}

#define HS_INSTANTIATE_DENSE_OPS(T)                                                                 \
  template void fill_random<T>(T*, index_t, std::uint64_t, std::uint64_t);                          \
  template T dot<T>(index_t, const T*, const T*);                                                   \
  template double norm2<T>(index_t, const T*);                                                      \
  template void axpy<T>(index_t, T, const T*, T*);                                                  \
  template void scal<T>(index_t, T, T*);                                                            \
  template double orthogonalize_cgs2<T>(index_t, index_t, const T*, index_t, const T*, index_t, T*, \
                                        T*, T*, const ApplyFn<T>*, const OrthoOptions&,             \
                                        OrthoInfo*);                                                \
  template void gram<T>(index_t, index_t, const T*, index_t, const T*, index_t, T*, index_t);       \
  template void multiply<T>(index_t, index_t, index_t, const T*, index_t, const T*, index_t, T*,    \
                            index_t);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_DENSE_OPS)
#undef HS_INSTANTIATE_DENSE_OPS

}  // namespace hs::linalg
