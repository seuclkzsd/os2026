// Sparse matrix times dense vectors / blocks (docs/DESIGN.md §3.2; P4 "solvephase").
//
// Three paths (KernelPath), dispatched at run time:
//   Reference  the P1 loops: spmm walks every row once per 8 vectors and reads X column-strided
//              (p loads from p cache lines per nonzero); std::complex arithmetic.
//   Portable   interleaved kernels in plain C++: X is first copied into a row-major block Xp
//              (row j holds the q <= 16 entries x[j, k0 .. k0+q)), so a nonzero a_ij costs q
//              contiguous multiply-adds; one pass over A per 16 vectors; explicit complex
//              arithmetic (no NaN-recovery calls; complex in the split form of the AVX-512 path).
//              Compiled for every width q = 1 .. 16 with the accumulators in vector registers of
//              the build's ISA (GCC / Clang vector extension: 2 doubles with SSE2, 4 with AVX), so
//              it is fast on any x86-64 (or other) CPU; Auto runs it on CPUs without AVX-512F.
//   Avx512     the same formulation with AVX-512F intrinsics: one broadcast + ceil(q/8) loads and
//              FMAs per nonzero (complex: 2 broadcasts, ceil(2q/8) loads, 2 FMAs per load; the
//              swap of the imaginary products is done once per row), two nonzeros in flight per
//              row.
// A single vector (spmv, spmm with p = 1) runs the Reference row loop on every path: it is bound by
// the matrix stream and the x gathers, and neither AVX-512 gathers nor split sums beat it; the
// single-column residual is one fused sweep straight from x and y (no interleaved copies).
// Every output row is computed by one thread in a fixed order, so each path is deterministic and
// independent of the thread count. Compiled for x86-64 with GCC/Clang, the AVX-512 code is built
// with function target attributes (no -mavx512f needed) and selected only when
// __builtin_cpu_supports("avx512f") (CPUID and OS register state) says it can run.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/sparse.hpp"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HS_KERNELS_AVX512 1
#include <immintrin.h>
#define HS_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define HS_KERNELS_AVX512 0
#endif

namespace hs {

namespace {

constexpr index_t kChunk = 16;      // vectors per pass of the interleaved kernels
constexpr index_t kRowBlock = 256;  // rows per scheduling unit of the fast kernels
// spmm interleaves X only when every row of X is used often enough to pay for the copy:
// nnz(A) >= reuse * ncols with reuse 2 for p <= 4 and 8 for wider blocks (measured with
// bench/solvephase_kernels.cpp: a diagonal B lost 1.5x at p = 8, 16; cav2d-rect-pert-1e5, 5
// entries per column, broke even at p = 8 and lost up to 1.25x at p = 16, while 10-26 entries per
// column gained 1.2-3.3x).
constexpr index_t kMinReuseNarrow = 2;
constexpr index_t kMinReuseWide = 8;

/// HS_KERNELS_NO_AVX512 set to anything but "" or "0": behave as on a CPU without AVX-512F.
[[maybe_unused]] bool avx512_masked_by_environment() noexcept {
  const char* v = std::getenv("HS_KERNELS_NO_AVX512");
  return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

bool cpu_has_avx512f() noexcept {
#if HS_KERNELS_AVX512
  static const bool ok = [] {
    if (avx512_masked_by_environment()) return false;
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") != 0;
  }();
  return ok;
#else
  return false;
#endif
}

// ------------------------------------------------------------------------- scalar helpers

/// a * b with the plain formula (std::complex's operator* adds NaN/Inf recovery calls).
inline double mul(double a, double b) {
  return a * b;
}
inline cplx mul(cplx a, cplx b) {
  return cplx(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

/// alpha * s + beta * y, with beta == 0 never reading y.
template <class T>
inline T combine(T alpha, T s, T beta, const T& y) {
  return beta == T(0) ? mul(alpha, s) : mul(alpha, s) + mul(beta, y);
}

/// |z| in the 1-norm of the components (|Re z| + |Im z| for complex: within sqrt(2) of |z|), the
/// magnitude used by the componentwise backward error of shifted_residual.
inline double mag(double v) {
  return std::abs(v);
}
inline double mag(cplx v) {
  return std::abs(v.real()) + std::abs(v.imag());
}

/// |r| / s, the componentwise backward error of one entry: 0 / 0 = 0, r / 0 = inf, and inf as
/// soon as r or s is not finite, so that a residual holding an Inf or a NaN never looks
/// converged (an overflowing row has r = +-inf and s = inf: inf / inf would be a NaN, which the
/// max reductions silently drop). The result is never NaN.
inline double ratio(double r, double s) {
  if (r == 0.0) return 0.0;
  if (!std::isfinite(r) || !std::isfinite(s) || !(s > 0.0))
    return std::numeric_limits<double>::infinity();
  return r / s;
}

/// Per-column maxima of the row blocks of a residual (blocks x q), reduced in any order: max is
/// exact, so the result is deterministic.
class BlockMax {
 public:
  BlockMax(double* out, index_t n, index_t q)
      : out_(out),
        q_(q),
        v_(out ? static_cast<std::size_t>(((n + kRowBlock - 1) / kRowBlock) * q) : 0, 0.0) {}
  bool on() const noexcept { return out_ != nullptr; }
  double* row(index_t i0) noexcept { return v_.data() + (i0 / kRowBlock) * q_; }
  void finish() {
    if (!out_) return;
    for (index_t k = 0; k < q_; ++k) out_[k] = 0.0;
    for (std::size_t t = 0; t < v_.size(); ++t) {
      const index_t k = static_cast<index_t>(t) % q_;
      out_[k] = std::max(out_[k], v_[t]);
    }
  }

 private:
  double* out_;
  index_t q_;
  std::vector<double> v_;
};

/// Scalars per vector entry in the double-precision scratch.
template <class T>
constexpr index_t kWords = is_complex_v<T> ? 2 : 1;

/// Scratch of `doubles` doubles: the caller's workspace (grown if needed) or a per-call buffer.
class Scratch {
 public:
  Scratch(std::vector<double>* workspace, index_t doubles) {
    const std::size_t len = static_cast<std::size_t>(std::max<index_t>(doubles, 1));
    if (workspace) {
      if (workspace->size() < len) workspace->resize(len);
      data_ = workspace->data();
    } else {
      own_.reset(new double[len]);  // default-initialised: no zeroing pass
      data_ = own_.get();
    }
  }
  double* data() const noexcept { return data_; }

 private:
  std::unique_ptr<double[]> own_;
  double* data_ = nullptr;
};

/// xp[(j * q + k) * w + c] = component c of (x[j + k * ldx] + s * z[j + k * ldz]) for j < m,
/// k < q (w = 1 real, 2 complex); z == nullptr means s * z = 0.
template <class T>
void interleave(index_t m, index_t q, const T* x, index_t ldx, double s, const T* z, index_t ldz,
                double* xp) {
#pragma omp parallel for schedule(static)
  for (index_t j = 0; j < m; ++j) {
    double* row = xp + j * q * kWords<T>;
    for (index_t k = 0; k < q; ++k) {
      T v = x[j + k * ldx];
      if (z) v += s * z[j + k * ldz];
      if constexpr (is_complex_v<T>) {
        row[2 * k] = v.real();
        row[2 * k + 1] = v.imag();
      } else {
        row[k] = v;
      }
    }
  }
}

/// Runs body(row_begin, row_end) over [0, n) in blocks of kRowBlock rows (OpenMP, static).
template <class F>
void for_row_blocks(index_t n, F&& body) {
  const index_t blocks = (n + kRowBlock - 1) / kRowBlock;
#pragma omp parallel for schedule(static)
  for (index_t blk = 0; blk < blocks; ++blk)
    body(blk * kRowBlock, std::min(n, (blk + 1) * kRowBlock));
}

// ----------------------------------------------------------------------------- Reference

template <class T>
void spmv_reference(const CsrMatrix<T>& a, const T* x, T* y, T alpha, T beta) {
  const index_t n = a.nrows;
  const index_t* rp = a.rowptr.data();
  const std::int32_t* ci = a.colind.data();
  const T* v = a.values.data();
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    T sum = T(0);
    for (index_t p = rp[i]; p < rp[i + 1]; ++p) sum += v[p] * x[ci[p]];
    // beta == 0 must not read y (it may hold garbage/NaN).
    y[i] = (beta == T(0)) ? alpha * sum : alpha * sum + beta * y[i];
  }
}

template <class T>
void spmm_reference(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p,
                    T alpha, T beta) {
  if (p == 1) {
    spmv_reference(a, x, y, alpha, beta);
    return;
  }
  const index_t n = a.nrows;
  const index_t* rp = a.rowptr.data();
  const std::int32_t* ci = a.colind.data();
  const T* v = a.values.data();
  constexpr index_t kBlock = 8;  // vectors processed together per row pass
#pragma omp parallel
  {
    T acc[kBlock];
#pragma omp for schedule(static)
    for (index_t i = 0; i < n; ++i) {
      for (index_t k0 = 0; k0 < p; k0 += kBlock) {
        const index_t kb = std::min(kBlock, p - k0);
        for (index_t k = 0; k < kb; ++k) acc[k] = T(0);
        for (index_t q = rp[i]; q < rp[i + 1]; ++q) {
          const T aq = v[q];
          const T* xc = x + ci[q] + k0 * ldx;
          for (index_t k = 0; k < kb; ++k) acc[k] += aq * xc[k * ldx];
        }
        for (index_t k = 0; k < kb; ++k) {
          T& yk = y[i + (k0 + k) * ldy];
          yk = (beta == T(0)) ? alpha * acc[k] : alpha * acc[k] + beta * yk;
        }
      }
    }
  }
}

template <class T>
void shifted_residual_reference(const CsrMatrix<T>& a, const CsrMatrix<T>* b, double sigma,
                                const T* x, index_t ldx, const T* y, index_t ldy, T* r, index_t ldr,
                                index_t p, double* omega) {
  const index_t n = a.nrows;
  if (b) {
    spmm_reference(*b, x, ldx, r, ldr, p, T(1), T(0));
  } else {
    for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, r + k * ldr);
  }
  spmm_reference(a, y, ldy, r, ldr, p, T(-1), T(1));
  if (b) {
    spmm_reference(*b, y, ldy, r, ldr, p, T(sigma), T(1));
  } else {
    for (index_t k = 0; k < p; ++k)
      for (index_t i = 0; i < n; ++i) r[i + k * ldr] += T(sigma) * y[i + k * ldy];
  }
  if (!omega) return;
  // Componentwise backward error with the scale s = |B| |w| + |A| |y|, w = x + sigma y (B = I:
  // |w|).
  for (index_t k = 0; k < p; ++k) {
    const T* xk = x + k * ldx;
    const T* yk = y + k * ldy;
    double w = 0.0;
#pragma omp parallel for schedule(static) reduction(max : w)  // max is exact: deterministic
    for (index_t i = 0; i < n; ++i) {
      double sc = 0.0;
      if (b) {
        for (index_t t = b->rowptr[i]; t < b->rowptr[i + 1]; ++t) {
          const index_t j = b->colind[t];
          sc += mag(b->values[t]) * mag(xk[j] + sigma * yk[j]);
        }
      } else {
        sc = mag(xk[i] + sigma * yk[i]);
      }
      for (index_t t = a.rowptr[i]; t < a.rowptr[i + 1]; ++t)
        sc += mag(a.values[t]) * mag(yk[a.colind[t]]);
      w = std::max(w, ratio(mag(r[i + k * ldr]), sc));
    }
    omega[k] = w;
  }
}

// ------------------------------------------------------------------------------ Portable

/// Calls f(std::integral_constant<int, Q>{}) with Q = q, for 1 <= q <= kChunk. The portable
/// kernels are compiled for every chunk width so that their accumulators live in registers: with
/// a run-time width they were memory operands (a load and a store per multiply-add) and the real
/// spmm ran at 0.77-0.88x the Reference loop for p = 4, 8. Per width, with the vector-extension
/// accumulators below, the Portable path is 1.4-3.0x (spmm) and 1.1-4.5x (residual) faster than
/// Reference for p = 2 .. 16 on the six 1e5 test cases, in SSE2-only and AVX2 builds at 4
/// threads (bench/solvephase_kernels.cpp; cases whose spmm keeps the strided loop excepted).
template <int Q = 1, class F>
void with_width(index_t q, F&& f) {
  if constexpr (Q <= kChunk) {
    if (q == Q) {
      f(std::integral_constant<int, Q>{});
      return;
    }
    with_width<Q + 1>(q, std::forward<F>(f));
  }
}

/// kLanes doubles as one value (GCC / Clang vector extension): one register of the build's ISA,
/// 4 doubles with AVX, 2 with SSE2 only (a wider extension vector is split into slow pieces there).
/// Fixing the width keeps the portable accumulators in registers whatever the auto-vectorizer's
/// cost model would pick (for complex q = 8 it chose 2-double vectors on AVX2 and spilled half of
/// them to the stack on every nonzero).
#if defined(__GNUC__) || defined(__clang__)
#if defined(__AVX__)
constexpr int kLanes = 4;
#else
constexpr int kLanes = 2;
#endif
typedef double VD __attribute__((vector_size(8 * kLanes)));
typedef long long VI __attribute__((vector_size(8 * kLanes)));
inline VD load_vd(const double* p) {
  VD v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
inline VD abs_vd(VD v) {
  return (VD)((VI)v & static_cast<long long>(0x7fffffffffffffffULL));
}
#else
#error "src/sparse/kernels.cpp needs the GCC / Clang vector extension (vector_size)"
#endif

/// Accumulators of one output row for Q interleaved vectors: kW = Q (real) or 2Q (complex) words,
/// kV full VD values and kT < kLanes scalar tail words. Complex products use the split form of the
/// AVX-512 path: re += Re(a) [xr, xi] and im += Im(a) [xr, xi] word by word, combined once per
/// row by value(); elementwise, hence no shuffles.
template <int Q, class T>
struct RowAcc {
  static constexpr bool kComplex = is_complex_v<T>;
  static constexpr index_t kW = Q * kWords<T>;
  static constexpr int kV = static_cast<int>(kW / kLanes), kT = static_cast<int>(kW % kLanes);
  VD re[kV ? kV : 1] = {}, im[kComplex && kV ? kV : 1] = {}, sc[kV ? kV : 1] = {};
  double re_t[kT ? kT : 1] = {}, im_t[kComplex && kT ? kT : 1] = {}, sc_t[kT ? kT : 1] = {};

  static double word(const VD* v, const double* tail, index_t w) {
    return w < kLanes * kV ? v[w / kLanes][w % kLanes] : tail[w - kLanes * kV];
  }
  /// Entry k of the row (k < Q).
  T value(int k) const {
    if constexpr (kComplex) {
      return T(word(re, re_t, 2 * k) - word(im, im_t, 2 * k + 1),
               word(re, re_t, 2 * k + 1) + word(im, im_t, 2 * k));
    } else {
      return word(re, re_t, k);
    }
  }
  /// Scale s_k of the backward error (accumulated magnitudes).
  double scale(int k) const {
    if constexpr (kComplex) {
      return word(sc, sc_t, 2 * k) + word(sc, sc_t, 2 * k + 1);
    } else {
      return word(sc, sc_t, k);
    }
  }
  /// re = the kW words at w (and, with S, sc = their magnitudes): the B = I start of a residual
  /// row.
  template <bool S>
  void start_from(const double* w) {
    for (int c = 0; c < kV; ++c) {
      re[c] = load_vd(w + kLanes * c);
      if constexpr (S) sc[c] = abs_vd(re[c]);
    }
    for (int c = 0; c < kT; ++c) {
      re_t[c] = w[kLanes * kV + c];
      if constexpr (S) sc_t[c] = std::abs(re_t[c]);
    }
  }
};

/// acc += sign * sum_t v[t] * Xp[ci[t], :] restricted to the accumulator vectors [C0, C1) (and the
/// tail words when C1 == kV) over row i of m; with S also the magnitudes mag(v[t]) |Xp[ci[t], :]|.
template <int C0, int C1, int Q, bool S, class T>
inline void row_accumulate_vectors(const CsrMatrix<T>& m, index_t i, const double* xp,
                                   double sign, RowAcc<Q, T>& acc) {
  using Acc = RowAcc<Q, T>;
  constexpr int kTail = C1 == Acc::kV ? Acc::kT : 0;
  const index_t* rp = m.rowptr.data();
  const std::int32_t* ci = m.colind.data();
  const T* v = m.values.data();
  for (index_t t = rp[i]; t < rp[i + 1]; ++t) {
    const double* xr = xp + static_cast<index_t>(ci[t]) * Acc::kW;
    double vr = 0.0, vi = 0.0;
    if constexpr (Acc::kComplex) {
      vr = sign * v[t].real();
      vi = sign * v[t].imag();
    } else {
      vr = sign * v[t];
    }
    const double mv = S ? mag(v[t]) : 0.0;
    for (int c = C0; c < C1; ++c) {
      const VD x = load_vd(xr + kLanes * c);
      acc.re[c] += vr * x;
      if constexpr (Acc::kComplex) acc.im[c] += vi * x;
      if constexpr (S) acc.sc[c] += mv * abs_vd(x);
    }
    for (int c = 0; c < kTail; ++c) {
      const double x = xr[kLanes * Acc::kV + c];
      acc.re_t[c] += vr * x;
      if constexpr (Acc::kComplex) acc.im_t[c] += vi * x;
      if constexpr (S) acc.sc_t[c] += mv * std::abs(x);
    }
  }
}

/// acc += sign * sum_t v[t] * Xp[ci[t], :] over row i of m (Xp interleaved, Q wide); with S also
/// the magnitudes. Rows wider than the register file are swept in groups of accumulator vectors
/// (at most 12 vector registers of accumulators per sweep; the row's indices and values are
/// re-read from L1), so that no accumulator lives on the stack.
template <int Q, bool S, class T, int C0 = 0>
inline void row_accumulate_portable(const CsrMatrix<T>& m, index_t i, const double* xp,
                                    double sign, RowAcc<Q, T>& acc) {
  using Acc = RowAcc<Q, T>;
  constexpr int kArrays = 1 + (Acc::kComplex ? 1 : 0) + (S ? 1 : 0);
  constexpr int kGroup = 12 / kArrays;
  constexpr int C1 = C0 + kGroup < Acc::kV ? C0 + kGroup : Acc::kV;
  row_accumulate_vectors<C0, C1, Q, S>(m, i, xp, sign, acc);
  if constexpr (C1 < Acc::kV) row_accumulate_portable<Q, S, T, C1>(m, i, xp, sign, acc);
}

template <class T>
inline T acc_value(const double* acc, index_t k) {
  if constexpr (is_complex_v<T>) {
    return T(acc[2 * k], acc[2 * k + 1]);
  } else {
    return acc[k];
  }
}

/// Y[:, 0..Q) = alpha * A * Xp + beta * Y for one interleaved chunk of Q <= 16 vectors.
template <int Q, class T>
void spmm_chunk_portable(const CsrMatrix<T>& a, const double* xp, T* y, index_t ldy, T alpha,
                         T beta) {
  for_row_blocks(a.nrows, [&](index_t i0, index_t i1) {
    for (index_t i = i0; i < i1; ++i) {
      RowAcc<Q, T> acc;
      row_accumulate_portable<Q, false>(a, i, xp, 1.0, acc);
      for (int k = 0; k < Q; ++k) {
        T& yk = y[i + k * ldy];
        yk = combine(alpha, acc.value(k), beta, yk);
      }
    }
  });
}

/// R[:, 0..Q) = B Wp - A Yp, or (B = I) Wp - A Yp (Wp = X + sigma Y, Yp = Y interleaved). With S,
/// also the row-block maxima of |r| / s (s = |B| |Wp| + |A| |Yp|, B = I: |Wp| + |A| |Yp|).
template <int Q, bool S, class T>
void residual_chunk_portable(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const double* wp,
                             const double* yp, T* r, index_t ldr, BlockMax& bm) {
  constexpr index_t words = RowAcc<Q, T>::kW;
  for_row_blocks(a.nrows, [&](index_t i0, index_t i1) {
    double* wmax = S ? bm.row(i0) : nullptr;
    for (index_t i = i0; i < i1; ++i) {
      RowAcc<Q, T> acc;
      if (b) {
        row_accumulate_portable<Q, S>(*b, i, wp, 1.0, acc);
      } else {
        acc.template start_from<S>(wp + i * words);
      }
      row_accumulate_portable<Q, S>(a, i, yp, -1.0, acc);
      for (int k = 0; k < Q; ++k) {
        const T v = acc.value(k);
        r[i + k * ldr] = v;
        if constexpr (S) wmax[k] = std::max(wmax[k], ratio(mag(v), acc.scale(k)));
      }
    }
  });
}

/// One column (q = 1) of the residual straight from x and y, without interleaved copies (for a
/// single vector they would only add two passes): r_i = sum_B b_ij (x_j + sigma y_j) - sum_A a_ij
/// y_j, or (B = I) x_i + sigma y_i - sum_A a_ij y_j.
template <class T>
void residual_column(const CsrMatrix<T>& a, const CsrMatrix<T>* b, double sigma, const T* x,
                     const T* y, T* r, BlockMax& bm) {
  const index_t* ra = a.rowptr.data();
  const std::int32_t* ca = a.colind.data();
  const T* va = a.values.data();
  for_row_blocks(a.nrows, [&](index_t i0, index_t i1) {
    double* wmax = bm.on() ? bm.row(i0) : nullptr;
    for (index_t i = i0; i < i1; ++i) {
      T s = T(0);
      double sc = 0.0;
      if (b) {
        for (index_t t = b->rowptr[i]; t < b->rowptr[i + 1]; ++t) {
          const index_t j = b->colind[t];
          const T w = x[j] + sigma * y[j];
          s += mul(b->values[t], w);
          if (wmax) sc += mag(b->values[t]) * mag(w);
        }
      } else {
        s = x[i] + sigma * y[i];
        if (wmax) sc = mag(s);
      }
      T acc = T(0);
      for (index_t t = ra[i]; t < ra[i + 1]; ++t) {
        acc += mul(va[t], y[ca[t]]);
        if (wmax) sc += mag(va[t]) * mag(y[ca[t]]);
      }
      r[i] = s - acc;
      if (wmax) wmax[0] = std::max(wmax[0], ratio(mag(r[i]), sc));
    }
  });
}

// -------------------------------------------------------------------------------- AVX-512

#if HS_KERNELS_AVX512

/// Mask of the valid lanes of chunk c (8 doubles) of a row holding `words` doubles.
inline __mmask8 lane_mask(index_t words, int c) {
  const index_t left = words - 8 * static_cast<index_t>(c);
  return left >= 8 ? static_cast<__mmask8>(0xFF) : static_cast<__mmask8>((1u << left) - 1u);
}

/// Real: acc[c] += sign * a_it * Xp[j, 8c .. 8c+8) over row i of m, two nonzeros in flight;
/// with S also sacc[c] += |a_it| |Xp[j, chunk c]|.
template <int NC, bool S>
HS_TARGET_AVX512 inline void row_real_avx512(const CsrMatrix<double>& m, index_t i,
                                             const double* xp, index_t q, const __mmask8* mask,
                                             double sign, __m512d* acc, __m512d* sacc) {
  const index_t* rp = m.rowptr.data();
  const std::int32_t* ci = m.colind.data();
  const double* v = m.values.data();
  __m512d acc2[NC];
  for (int c = 0; c < NC; ++c) acc2[c] = _mm512_setzero_pd();
  index_t t = rp[i];
  const index_t e = rp[i + 1];
  for (; t + 1 < e; t += 2) {
    const double* x0 = xp + static_cast<index_t>(ci[t]) * q;
    const double* x1 = xp + static_cast<index_t>(ci[t + 1]) * q;
    const __m512d a0 = _mm512_set1_pd(sign * v[t]);
    const __m512d a1 = _mm512_set1_pd(sign * v[t + 1]);
    for (int c = 0; c < NC; ++c) {
      const __m512d x0v = _mm512_maskz_loadu_pd(mask[c], x0 + 8 * c);
      const __m512d x1v = _mm512_maskz_loadu_pd(mask[c], x1 + 8 * c);
      acc[c] = _mm512_fmadd_pd(a0, x0v, acc[c]);
      acc2[c] = _mm512_fmadd_pd(a1, x1v, acc2[c]);
      if constexpr (S) {
        sacc[c] = _mm512_fmadd_pd(_mm512_abs_pd(a0), _mm512_abs_pd(x0v), sacc[c]);
        sacc[c] = _mm512_fmadd_pd(_mm512_abs_pd(a1), _mm512_abs_pd(x1v), sacc[c]);
      }
    }
  }
  if (t < e) {
    const double* x0 = xp + static_cast<index_t>(ci[t]) * q;
    const __m512d a0 = _mm512_set1_pd(sign * v[t]);
    for (int c = 0; c < NC; ++c) {
      const __m512d x0v = _mm512_maskz_loadu_pd(mask[c], x0 + 8 * c);
      acc[c] = _mm512_fmadd_pd(a0, x0v, acc[c]);
      if constexpr (S) sacc[c] = _mm512_fmadd_pd(_mm512_abs_pd(a0), _mm512_abs_pd(x0v), sacc[c]);
    }
  }
  for (int c = 0; c < NC; ++c) acc[c] = _mm512_add_pd(acc[c], acc2[c]);
}

/// Complex: (accr[c], acci[c]) += sign * (Re a, Im a) * Xp[j, chunk c] over row i of m; the
/// product is assembled by finish_complex(). With S also sacc[c] += mag(a) [|xr|, |xi|] (the two
/// lanes of a vector entry are added by the caller).
template <int NC, bool S>
HS_TARGET_AVX512 inline void row_complex_avx512(const CsrMatrix<cplx>& m, index_t i,
                                                const double* xp, index_t q, const __mmask8* mask,
                                                double sign, __m512d* accr, __m512d* acci,
                                                __m512d* sacc) {
  const index_t* rp = m.rowptr.data();
  const std::int32_t* ci = m.colind.data();
  const double* v = reinterpret_cast<const double*>(m.values.data());
  const index_t w = 2 * q;
  for (index_t t = rp[i]; t < rp[i + 1]; ++t) {
    const double* x0 = xp + static_cast<index_t>(ci[t]) * w;
    const __m512d ar = _mm512_set1_pd(sign * v[2 * t]);
    const __m512d ai = _mm512_set1_pd(sign * v[2 * t + 1]);
    __m512d am = _mm512_setzero_pd();
    if constexpr (S) am = _mm512_set1_pd(std::abs(v[2 * t]) + std::abs(v[2 * t + 1]));
    for (int c = 0; c < NC; ++c) {
      const __m512d xv = _mm512_maskz_loadu_pd(mask[c], x0 + 8 * c);
      accr[c] = _mm512_fmadd_pd(ar, xv, accr[c]);
      acci[c] = _mm512_fmadd_pd(ai, xv, acci[c]);
      if constexpr (S) sacc[c] = _mm512_fmadd_pd(am, _mm512_abs_pd(xv), sacc[c]);
    }
  }
}

/// [re, im] pairs of sum_t a_t x_t from accr = sum Re(a) [xr, xi], acci = sum Im(a) [xr, xi]:
/// re = accr_even - acci_odd, im = accr_odd + acci_even.
HS_TARGET_AVX512 inline __m512d finish_complex(__m512d accr, __m512d acci) {
  return _mm512_fmaddsub_pd(accr, _mm512_set1_pd(1.0), _mm512_permute_pd(acci, 0x55));
}

template <class T, int NC>
HS_TARGET_AVX512 void spmm_rows_avx512(const CsrMatrix<T>& a, const double* xp, index_t q, T* y,
                                       index_t ldy, T alpha, T beta, index_t i0, index_t i1) {
  const index_t words = q * kWords<T>;
  __mmask8 mask[NC];
  for (int c = 0; c < NC; ++c) mask[c] = lane_mask(words, c);
  alignas(64) double out[2 * kChunk] = {};  // zeroed once per row block, as below
  for (index_t i = i0; i < i1; ++i) {
    if constexpr (is_complex_v<T>) {
      __m512d accr[NC], acci[NC];
      for (int c = 0; c < NC; ++c) accr[c] = acci[c] = _mm512_setzero_pd();
      row_complex_avx512<NC, false>(a, i, xp, q, mask, 1.0, accr, acci, nullptr);
      for (int c = 0; c < NC; ++c) _mm512_store_pd(out + 8 * c, finish_complex(accr[c], acci[c]));
    } else {
      __m512d acc[NC];
      for (int c = 0; c < NC; ++c) acc[c] = _mm512_setzero_pd();
      row_real_avx512<NC, false>(a, i, xp, q, mask, 1.0, acc, nullptr);
      for (int c = 0; c < NC; ++c) _mm512_store_pd(out + 8 * c, acc[c]);
    }
    for (index_t k = 0; k < q; ++k) {
      T& yk = y[i + k * ldy];
      yk = combine(alpha, acc_value<T>(out, k), beta, yk);
    }
  }
}

template <class T, int NC, bool S>
HS_TARGET_AVX512 void residual_rows_avx512(const CsrMatrix<T>& a, const CsrMatrix<T>* b,
                                           const double* wp, const double* yp, index_t q, T* r,
                                           index_t ldr, index_t i0, index_t i1, double* wmax) {
  const index_t words = q * kWords<T>;
  __mmask8 mask[NC];
  for (int c = 0; c < NC; ++c) mask[c] = lane_mask(words, c);
  // Zeroed once per row block (the intrinsic stores below always cover entries 0 .. q-1, which
  // GCC cannot prove: -Wmaybe-uninitialized without -march=native).
  alignas(64) double out[2 * kChunk] = {}, sout[2 * kChunk] = {};
  for (index_t i = i0; i < i1; ++i) {
    // B = I: the accumulators start from the row of Wp (and |Wp|), so that R and the backward
    // error come from the same vector arithmetic whether or not the latter is requested.
    __m512d start[NC], sacc[NC];
    for (int c = 0; c < NC; ++c) {
      start[c] = b ? _mm512_setzero_pd() : _mm512_maskz_loadu_pd(mask[c], wp + i * words + 8 * c);
      sacc[c] = S && !b ? _mm512_abs_pd(start[c]) : _mm512_setzero_pd();
    }
    if constexpr (is_complex_v<T>) {
      __m512d accr[NC], acci[NC];
      for (int c = 0; c < NC; ++c) {
        accr[c] = start[c];
        acci[c] = _mm512_setzero_pd();
      }
      if (b) row_complex_avx512<NC, S>(*b, i, wp, q, mask, 1.0, accr, acci, sacc);
      row_complex_avx512<NC, S>(a, i, yp, q, mask, -1.0, accr, acci, sacc);
      for (int c = 0; c < NC; ++c) _mm512_store_pd(out + 8 * c, finish_complex(accr[c], acci[c]));
    } else {
      __m512d acc[NC];
      for (int c = 0; c < NC; ++c) acc[c] = start[c];
      if (b) row_real_avx512<NC, S>(*b, i, wp, q, mask, 1.0, acc, sacc);
      row_real_avx512<NC, S>(a, i, yp, q, mask, -1.0, acc, sacc);
      for (int c = 0; c < NC; ++c) _mm512_store_pd(out + 8 * c, acc[c]);
    }
    if constexpr (S)
      for (int c = 0; c < NC; ++c) _mm512_store_pd(sout + 8 * c, sacc[c]);
    for (index_t k = 0; k < q; ++k) {
      const T v = acc_value<T>(out, k);
      r[i + k * ldr] = v;
      if constexpr (S) {
        const double sc = is_complex_v<T> ? sout[2 * k] + sout[2 * k + 1] : sout[k];
        wmax[k] = std::max(wmax[k], ratio(mag(v), sc));
      }
    }
  }
}

template <class T, class F>
void with_chunk_width(index_t q, F&& f) {
  // NC = number of 8-double chunks of an interleaved row of q vectors.
  const index_t words = q * kWords<T>;
  if (words <= 8) {
    f(std::integral_constant<int, 1>{});
  } else if (words <= 16) {
    f(std::integral_constant<int, 2>{});
  } else if (words <= 24) {
    f(std::integral_constant<int, 3>{});
  } else {
    f(std::integral_constant<int, 4>{});
  }
}

#endif  // HS_KERNELS_AVX512

// --------------------------------------------------------------------------- dispatchers

template <class T>
void spmm_chunk(const CsrMatrix<T>& a, const double* xp, index_t q, T* y, index_t ldy, T alpha,
                T beta, KernelPath path) {
#if HS_KERNELS_AVX512
  if (path == KernelPath::Avx512) {
    with_chunk_width<T>(q, [&](auto nc) {
      for_row_blocks(a.nrows, [&](index_t i0, index_t i1) {
        spmm_rows_avx512<T, decltype(nc)::value>(a, xp, q, y, ldy, alpha, beta, i0, i1);
      });
    });
    return;
  }
#endif
  static_cast<void>(path);
  with_width(q, [&](auto qc) {
    spmm_chunk_portable<decltype(qc)::value>(a, xp, y, ldy, alpha, beta);
  });
}

template <class T>
void residual_chunk(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const double* wp,
                    const double* yp, index_t q, T* r, index_t ldr, KernelPath path, BlockMax& bm) {
#if HS_KERNELS_AVX512
  if (path == KernelPath::Avx512) {
    with_chunk_width<T>(q, [&](auto nc) {
      for_row_blocks(a.nrows, [&](index_t i0, index_t i1) {
        if (bm.on()) {
          residual_rows_avx512<T, decltype(nc)::value, true>(a, b, wp, yp, q, r, ldr, i0, i1,
                                                             bm.row(i0));
        } else {
          residual_rows_avx512<T, decltype(nc)::value, false>(a, b, wp, yp, q, r, ldr, i0, i1,
                                                              nullptr);
        }
      });
    });
    return;
  }
#endif
  static_cast<void>(path);
  with_width(q, [&](auto qc) {
    constexpr int kQ = decltype(qc)::value;
    if (bm.on()) {
      residual_chunk_portable<kQ, true>(a, b, wp, yp, r, ldr, bm);
    } else {
      residual_chunk_portable<kQ, false>(a, b, wp, yp, r, ldr, bm);
    }
  });
}

}  // namespace

bool kernel_path_available(KernelPath path) noexcept {
  return path != KernelPath::Avx512 || cpu_has_avx512f();
}

KernelPath resolve_kernel_path(KernelPath path) {
  switch (path) {
    case KernelPath::Auto:
      return cpu_has_avx512f() ? KernelPath::Avx512 : KernelPath::Portable;
    case KernelPath::Avx512:
      if (!cpu_has_avx512f())
        HS_THROW(Unsupported,
                 "sparse kernels: KernelPath::Avx512 requested, but this CPU / OS does not "
                 "support AVX-512F (or the library was built without it)");
      return path;
    case KernelPath::Reference:
    case KernelPath::Portable:
      return path;
  }
  HS_THROW(InvalidInput, "sparse kernels: unknown KernelPath " << static_cast<int>(path));
}

template <class T>
void spmv(const CsrMatrix<T>& a, const T* x, T* y, T alpha, T beta, const KernelOptions& options) {
  // One vector: every path runs the plain row loop. It is bound by the matrix stream and the x
  // gathers; AVX-512 gathers (8 nonzeros + a horizontal sum per row) measured 1.7-3x slower on
  // Ice Lake, two partial sums no faster (bench/solvephase_kernels.cpp, p = 1).
  resolve_kernel_path(options.path);  // validates the request
  spmv_reference(a, x, y, alpha, beta);
}

template <class T>
void spmm(const CsrMatrix<T>& a, const T* x, index_t ldx, T* y, index_t ldy, index_t p, T alpha,
          T beta, const KernelOptions& options) {
  const KernelPath path = resolve_kernel_path(options.path);
  if (p <= 0) return;
  if (path == KernelPath::Reference) {
    spmm_reference(a, x, ldx, y, ldy, p, alpha, beta);
    return;
  }
  if (p == 1) {
    spmv_reference(a, x, y, alpha, beta);
    return;
  }
  const index_t m = a.ncols;
  if (a.nnz() < (p <= 4 ? kMinReuseNarrow : kMinReuseWide) * m) {
    // Rows of X are used too rarely: the interleaved copy (one read and one write of X) costs
    // more than the strided reads it saves (see kMinReuseNarrow / kMinReuseWide).
    spmm_reference(a, x, ldx, y, ldy, p, alpha, beta);
    return;
  }
  Scratch xp(options.workspace, m * std::min(p, kChunk) * kWords<T>);
  for (index_t k0 = 0; k0 < p; k0 += kChunk) {
    const index_t q = std::min(kChunk, p - k0);
    interleave<T>(m, q, x + k0 * ldx, ldx, 0.0, nullptr, 0, xp.data());
    spmm_chunk(a, xp.data(), q, y + k0 * ldy, ldy, alpha, beta, path);
  }
}

template <class T>
void shifted_residual(const CsrMatrix<T>& a, const non_deduced_t<CsrMatrix<T>>* b, double sigma,
                      const T* x, index_t ldx, const T* y, index_t ldy, T* r, index_t ldr,
                      index_t p, const KernelOptions& options, double* backward_error) {
  if (!a.square() || (b && (b->nrows != a.nrows || !b->square())))
    HS_THROW(InvalidInput, "shifted_residual: A (" << a.nrows << "x" << a.ncols
                                                   << ") and B must be square "
                                                   << "of equal order");
  const KernelPath path = resolve_kernel_path(options.path);
  if (p <= 0) return;
  if (path == KernelPath::Reference) {
    shifted_residual_reference(a, b, sigma, x, ldx, y, ldy, r, ldr, p, backward_error);
    return;
  }
  const index_t n = a.nrows;
  const index_t q_max = p == 1 ? 0 : std::min(p, kChunk);
  // [Yp | Wp]: Y interleaved (for A) and W = X + sigma Y interleaved (for B, or the start value
  // of every row when B = I).
  const index_t block = n * q_max * kWords<T>;
  Scratch buf(options.workspace, 2 * block);
  double* yp = buf.data();
  double* wp = buf.data() + block;
  for (index_t k0 = 0; k0 < p; k0 += kChunk) {
    const index_t q = std::min(kChunk, p - k0);
    const T* xk = x + k0 * ldx;
    const T* yk = y + k0 * ldy;
    BlockMax bm(backward_error ? backward_error + k0 : nullptr, n, q);
    if (q == 1) {
      residual_column(a, b, sigma, xk, yk, r + k0 * ldr, bm);
    } else {
      interleave<T>(n, q, yk, ldy, 0.0, nullptr, 0, yp);
      interleave<T>(n, q, xk, ldx, sigma, yk, ldy, wp);
      residual_chunk(a, b, wp, yp, q, r + k0 * ldr, ldr, path, bm);
    }
    bm.finish();
  }
}

#define HS_INSTANTIATE_KERNELS(T)                                                               \
  template void spmv<T>(const CsrMatrix<T>&, const T*, T*, T, T, const KernelOptions&);         \
  template void spmm<T>(const CsrMatrix<T>&, const T*, index_t, T*, index_t, index_t, T, T,     \
                        const KernelOptions&);                                                  \
  template void shifted_residual<T>(const CsrMatrix<T>&, const non_deduced_t<CsrMatrix<T>>*,    \
                                    double, const T*, index_t, const T*, index_t, T*, index_t,  \
                                    index_t, const KernelOptions&, double*);

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_KERNELS)
#undef HS_INSTANTIATE_KERNELS

}  // namespace hs
