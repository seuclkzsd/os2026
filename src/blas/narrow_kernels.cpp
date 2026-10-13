// BLAS hook: narrow dgemm kernels, C = alpha * op(A) * B + beta * C for 1 <= N <= 64 columns,
// a small triangular solve, and their parallel split (blas_hook.hpp). Shapes of MUMPS'
// multi-right-hand-side solves (N = p = 1..16) and of hsolve's block products: memory-bound, so
// the kernels stream A exactly once and never pack it (OpenBLAS' level-3 path copies A into a
// packed buffer first, which for N <= 8 costs about as much as the multiplication; its
// small-matrix kernels only take M*N*K <= 1e6), and tiny calls avoid OpenBLAS' fixed cost of
// ~1-2 us per call (buffer allocation, packing).
//
// Instruction sets, chosen at run time (Isa, from CPUID): the SIMD kernels are written once in
// simd_kernels.hpp for a vector traits type and compiled here twice - AVX-512F (V512, 8 doubles
// per register) and AVX2 + FMA (V256, 4 doubles) - each set of functions with its own
// target("...") attribute (no global -m flags: the rest of the binary stays x86-64 baseline).
// Isa::Portable runs the plain C++ loops below (reference and fallback; on non-x86 CPUs the only
// kernels compiled, whatever Isa the caller passes). Provenance of the SIMD
// code: simd_kernels.hpp and docs/BLAS.md (adapted from OpenBLAS 0.3.34, BSD-3-Clause).
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <memory>

#include "blas/blas_hook.hpp"

#if HS_BLAS_X86
#include <immintrin.h>
#endif

namespace hs::blas {
namespace {

using idx = std::ptrdiff_t;

constexpr int kMaxN = 64;

// Row chunk of the NN kernels: the chunk's partial results (rows x N doubles) stay in L1/L2.
fint nn_row_chunk(fint n) {
  return n <= 8 ? 512 : 256;
}

// Fixed work split of a parallel narrow product (a function of the shape and the thread count
// only): rb row blocks of rows_per rows (a multiple of `align`, at least min_rows unless m is
// smaller) times sk chunks of k_per (a multiple of 8) of the K dimension, rb * sk <= threads. K is
// split only when the row blocks leave threads idle, into chunks of at least kMinKChunk.
struct Split {
  fint rb = 1, rows_per = 0, sk = 1, k_per = 0;
};

constexpr fint kMinKChunk = 256;

Split make_split(int threads, fint m, fint k, fint min_rows, fint align) {
  Split s;
  const fint t = std::max(threads, 1);
  s.rb = std::clamp<fint>(m / std::max<fint>(min_rows, 1), 1, t);
  s.rows_per = ((m + s.rb - 1) / s.rb + align - 1) / align * align;
  s.rb = (m + s.rows_per - 1) / s.rows_per;
  s.sk = std::clamp<fint>(k / kMinKChunk, 1, std::max<fint>(t / s.rb, 1));
  s.k_per = ((k + s.sk - 1) / s.sk + 7) / 8 * 8;
  s.sk = (k + s.k_per - 1) / s.k_per;
  return s;
}

// Runs kernel(i0, i1, k0, k1, alpha, beta, c, ldc), which sets rows [i0, i1) of
// c = alpha * op(A)(:, k0:k1) * B(k0:k1, :) + beta * c, over make_split's blocks on up to `threads`
// threads (serially inside a parallel region or for one block). With sk > 1 the chunks write
// partial products (alpha = 1, beta = 0) that are then summed in chunk order per element:
// C = alpha * (P_0 + P_1 + ...) + beta * C. Each block's arithmetic depends on the split only, so
// the result is the same for a fixed thread count (and, with sk = 1, for any thread count).
template <class F>
void run_split(int threads, fint m, fint n, fint k, double alpha, double beta, double* c, fint ldc,
               fint min_rows, fint align, const F& kernel) {
  if (threads <= 1 || omp_in_parallel()) {
    kernel(0, m, 0, k, alpha, beta, c, ldc);
    return;
  }
  const Split s = make_split(threads, m, k, min_rows, align);
  const int tasks = static_cast<int>(s.rb * s.sk);
  if (tasks <= 1) {
    kernel(0, m, 0, k, alpha, beta, c, ldc);
    return;
  }
  if (s.sk == 1) {
#pragma omp parallel num_threads(tasks)
    for (int t = omp_get_thread_num(); t < tasks; t += omp_get_num_threads()) {
      const fint i0 = static_cast<fint>(t) * s.rows_per;
      kernel(i0, std::min<fint>(m, i0 + s.rows_per), 0, k, alpha, beta, c, ldc);
    }
    return;
  }
  const std::size_t plane = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  std::unique_ptr<double[]> part(new double[plane * static_cast<std::size_t>(s.sk)]);
  const fint red_per = ((m + tasks - 1) / tasks + 7) / 8 * 8;
#pragma omp parallel num_threads(tasks)
  {
    for (int t = omp_get_thread_num(); t < tasks; t += omp_get_num_threads()) {
      const fint r = t / s.sk, q = t % s.sk;
      const fint i0 = r * s.rows_per, k0 = q * s.k_per;
      kernel(i0, std::min<fint>(m, i0 + s.rows_per), k0, std::min<fint>(k, k0 + s.k_per), 1.0, 0.0,
             part.get() + plane * static_cast<std::size_t>(q), m);
    }
#pragma omp barrier
    for (int t = omp_get_thread_num(); t < tasks; t += omp_get_num_threads()) {
      const fint i0 = static_cast<fint>(t) * red_per, i1 = std::min<fint>(m, i0 + red_per);
      for (fint j = 0; j < n; ++j) {
        const double* pj = part.get() + static_cast<idx>(j) * m;
        double* cj = c + static_cast<idx>(j) * ldc;
        for (fint i = i0; i < i1; ++i) {
          double sum = pj[i];
          for (fint q = 1; q < s.sk; ++q) sum += pj[plane * static_cast<std::size_t>(q) + i];
          cj[i] = beta == 0.0 ? alpha * sum : alpha * sum + beta * cj[i];
        }
      }
    }
  }
}

// C(rows, j) = alpha * t(rows, j) + beta * C (beta == 0: C not read).
void finish_block(fint rows, fint n, double alpha, const double* t, fint ldt, double beta,
                  double* c, fint ldc) {
  for (fint j = 0; j < n; ++j) {
    const double* tj = t + static_cast<idx>(j) * ldt;
    double* cj = c + static_cast<idx>(j) * ldc;
    if (beta == 0.0) {
      for (fint i = 0; i < rows; ++i) cj[i] = alpha * tj[i];
    } else {
      for (fint i = 0; i < rows; ++i) cj[i] = alpha * tj[i] + beta * cj[i];
    }
  }
}

// ------------------------------------------------------------------------------------ portable

void nn_rows_portable(fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
                      const double* b, fint ldb, double beta, double* c, fint ldc) {
  const fint rc = nn_row_chunk(n);
  alignas(64) double t[4096];
  for (fint r0 = i0; r0 < i1; r0 += rc) {
    const fint rows = std::min(rc, i1 - r0);
    // Columns in groups that fit t (n <= 64: 4096 / rows >= 8 columns per group).
    const fint group = std::max<fint>(1, 4096 / rows);
    for (fint j0 = 0; j0 < n; j0 += group) {
      const fint nj = std::min(group, n - j0);
      std::fill(t, t + static_cast<idx>(rows) * nj, 0.0);
      for (fint kk = 0; kk < k; ++kk) {
        const double* ak = a + static_cast<idx>(kk) * lda + r0;
        for (fint j = 0; j < nj; ++j) {
          const double bkj = b[kk + static_cast<idx>(j0 + j) * ldb];
          double* tj = t + static_cast<idx>(j) * rows;
          for (fint i = 0; i < rows; ++i) tj[i] += ak[i] * bkj;
        }
      }
      finish_block(rows, nj, alpha, t, rows, beta, c + r0 + static_cast<idx>(j0) * ldc, ldc);
    }
  }
}

void tn_rows_portable(fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
                      const double* b, fint ldb, double beta, double* c, fint ldc) {
  for (fint i = i0; i < i1; ++i) {
    const double* ai = a + static_cast<idx>(i) * lda;
    for (fint j = 0; j < n; ++j) {
      const double* bj = b + static_cast<idx>(j) * ldb;
      double s = 0.0;
      for (fint kk = 0; kk < k; ++kk) s += ai[kk] * bj[kk];
      double& cij = c[i + static_cast<idx>(j) * ldc];
      cij = beta == 0.0 ? alpha * s : alpha * s + beta * cij;
    }
  }
}

// ------------------------------------------------------------------------- small triangular solve
// B(m x n) := op(A)^{-1} B (side L, alpha = 1), the reference BLAS loop orders: op(A) = A with
// A lower / upper - column-oriented forward / backward substitution (axpy with the contiguous
// column of A, vectorized over rows); op(A) = A^T - dot products of the contiguous column i of A
// with the solved part of the column of B. No skipping of zero entries (as OpenBLAS). The unit
// diagonal is never read.

struct DotPortable {
  static double dot(const double* x, const double* y, fint len) {
    double s = 0.0;
    for (fint j = 0; j < len; ++j) s += x[j] * y[j];
    return s;
  }
};

template <class Dot, bool kUnit>
inline __attribute__((always_inline)) void trsm_small_body(bool upper, bool trans, fint m, fint n,
                                                           const double* a, fint lda, double* b,
                                                           fint ldb) {
  if (!trans) {
    // op(A) = A: column-oriented; forward for lower, backward for upper.
    for (fint step = 0; step < m; ++step) {
      const fint j = upper ? m - 1 - step : step;
      const double* aj = a + static_cast<idx>(j) * lda;
      const fint i0 = upper ? 0 : j + 1;
      const fint i1 = upper ? j : m;
      for (fint c = 0; c < n; ++c) {
        double* bc = b + static_cast<idx>(c) * ldb;
        double x = bc[j];
        if (!kUnit) x /= aj[j];
        bc[j] = x;
        for (fint i = i0; i < i1; ++i) bc[i] -= aj[i] * x;
      }
    }
  } else {
    // op(A) = A^T: dot form; forward for upper A (A^T lower), backward for lower A.
    for (fint step = 0; step < m; ++step) {
      const fint i = upper ? step : m - 1 - step;
      const double* ai = a + static_cast<idx>(i) * lda;
      const fint j0 = upper ? 0 : i + 1;
      const fint len = upper ? i : m - 1 - i;
      for (fint c = 0; c < n; ++c) {
        double* bc = b + static_cast<idx>(c) * ldb;
        double x = bc[i] - Dot::dot(ai + j0, bc + j0, len);
        if (!kUnit) x /= ai[i];
        bc[i] = x;
      }
    }
  }
}

void trsm_small_portable(bool upper, bool trans, bool unit, fint m, fint n, const double* a,
                         fint lda, double* b, fint ldb) {
  if (unit)
    trsm_small_body<DotPortable, true>(upper, trans, m, n, a, lda, b, ldb);
  else
    trsm_small_body<DotPortable, false>(upper, trans, m, n, a, lda, b, ldb);
}

// ------------------------------------------------------------------------------- SIMD kernels
#if HS_BLAS_X86

#define HS_T512 __attribute__((target("avx512f"), always_inline)) inline
#define HS_T256 __attribute__((target("avx2,fma"), always_inline)) inline

// AVX-512F: 8 doubles per zmm, 32 registers (blocks of 4 x 4 accumulators), mask registers.
struct V512 {
  using reg = __m512d;
  using mask = __mmask8;
  static constexpr int W = 8;
  static constexpr int kGroup = 4;
  static constexpr fint kRegRows = 32;
  HS_T512 static mask tail(fint r) { return static_cast<__mmask8>((1u << r) - 1u); }
  HS_T512 static reg zero() { return _mm512_setzero_pd(); }
  HS_T512 static reg set1(double x) { return _mm512_set1_pd(x); }
  HS_T512 static reg loadu(const double* p) { return _mm512_loadu_pd(p); }
  HS_T512 static reg load(const double* p) { return _mm512_load_pd(p); }
  HS_T512 static void store(double* p, reg v) { _mm512_store_pd(p, v); }
  HS_T512 static reg maskz_loadu(mask m, const double* p) { return _mm512_maskz_loadu_pd(m, p); }
  HS_T512 static void mask_storeu(double* p, mask m, reg v) { _mm512_mask_storeu_pd(p, m, v); }
  HS_T512 static reg fmadd(reg a, reg b, reg c) { return _mm512_fmadd_pd(a, b, c); }
  HS_T512 static reg fnmadd(reg a, reg b, reg c) { return _mm512_fnmadd_pd(a, b, c); }
  HS_T512 static reg mul(reg a, reg b) { return _mm512_mul_pd(a, b); }
  HS_T512 static reg add(reg a, reg b) { return _mm512_add_pd(a, b); }
  // {sum a0, sum a1, sum a2, sum a3}: pairwise adds within 128-bit lanes, then across lanes - the
  // reduction scheme of OpenBLAS' REDUCE_4 (dgemm_small_kernel_tn_skylakex.c) with fewer
  // cross-lane permutes. Fixed order: deterministic.
  HS_T512 static void reduce4(reg a0, reg a1, reg a2, reg a3, double* out) {
    const __m512d u01 = _mm512_add_pd(_mm512_unpacklo_pd(a0, a1), _mm512_unpackhi_pd(a0, a1));
    const __m512d u23 = _mm512_add_pd(_mm512_unpacklo_pd(a2, a3), _mm512_unpackhi_pd(a2, a3));
    const __m512d z = _mm512_add_pd(_mm512_shuffle_f64x2(u01, u23, 0x88),   // lanes 0, 2 | 0, 2
                                    _mm512_shuffle_f64x2(u01, u23, 0xDD));  // lanes 1, 3 | 1, 3
    const __m512d p = _mm512_shuffle_f64x2(z, z, 0xD8);                      // lanes 0, 2, 1, 3
    _mm256_store_pd(out, _mm256_add_pd(_mm512_castpd512_pd256(p), _mm512_extractf64x4_pd(p, 1)));
  }
};

// AVX2 + FMA: 4 doubles per ymm, 16 registers (blocks of 4 x 2 accumulators), vector masks.
struct V256 {
  using reg = __m256d;
  using mask = __m256i;
  static constexpr int W = 4;
  static constexpr int kGroup = 2;
  static constexpr fint kRegRows = 16;
  HS_T256 static mask tail(fint r) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x(r), _mm256_setr_epi64x(0, 1, 2, 3));
  }
  HS_T256 static reg zero() { return _mm256_setzero_pd(); }
  HS_T256 static reg set1(double x) { return _mm256_set1_pd(x); }
  HS_T256 static reg loadu(const double* p) { return _mm256_loadu_pd(p); }
  HS_T256 static reg load(const double* p) { return _mm256_load_pd(p); }
  HS_T256 static void store(double* p, reg v) { _mm256_store_pd(p, v); }
  HS_T256 static reg maskz_loadu(mask m, const double* p) { return _mm256_maskload_pd(p, m); }
  HS_T256 static void mask_storeu(double* p, mask m, reg v) { _mm256_maskstore_pd(p, m, v); }
  HS_T256 static reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b, c); }
  HS_T256 static reg fnmadd(reg a, reg b, reg c) { return _mm256_fnmadd_pd(a, b, c); }
  HS_T256 static reg mul(reg a, reg b) { return _mm256_mul_pd(a, b); }
  HS_T256 static reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
  // {sum a0, sum a1, sum a2, sum a3}: horizontal adds of pairs, then the two 128-bit halves.
  HS_T256 static void reduce4(reg a0, reg a1, reg a2, reg a3, double* out) {
    const __m256d h01 = _mm256_hadd_pd(a0, a1);  // a0[0]+a0[1], a1[0]+a1[1], a0[2]+a0[3], ...
    const __m256d h23 = _mm256_hadd_pd(a2, a3);
    _mm256_store_pd(out, _mm256_add_pd(_mm256_permute2f128_pd(h01, h23, 0x20),
                                       _mm256_permute2f128_pd(h01, h23, 0x31)));
  }
};

namespace avx512 {
using V = V512;
#define HS_SIMD __attribute__((target("avx512f")))
#include "blas/simd_kernels.hpp"
#undef HS_SIMD
}  // namespace avx512

namespace avx2 {
using V = V256;
#define HS_SIMD __attribute__((target("avx2,fma")))
#include "blas/simd_kernels.hpp"
#undef HS_SIMD
}  // namespace avx2

#undef HS_T512
#undef HS_T256

#endif  // HS_BLAS_X86

// Rows [i0, i1) of C = alpha * op(A) * B + beta * C on the given instruction set.
void nn_rows(Isa isa, fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
             const double* b, fint ldb, double beta, double* c, fint ldc) {
  switch (isa) {
#if HS_BLAS_X86
    case Isa::Avx512:
      avx512::nn_rows(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    case Isa::Avx2:
      avx2::nn_rows(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
#else
    case Isa::Avx512:
    case Isa::Avx2:
#endif
    case Isa::Portable:
      nn_rows_portable(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
  }
}

void tn_rows(Isa isa, fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
             const double* b, fint ldb, double beta, double* c, fint ldc) {
  switch (isa) {
#if HS_BLAS_X86
    case Isa::Avx512:
      avx512::tn_rows(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    case Isa::Avx2:
      avx2::tn_rows(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
#else
    case Isa::Avx512:
    case Isa::Avx2:
#endif
    case Isa::Portable:
      tn_rows_portable(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
  }
}

}  // namespace

void narrow_gemm_nn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc,
                    fint min_rows) {
  if (m <= 0 || n <= 0 || n > kMaxN) return;
  run_split(threads, m, n, k, alpha, beta, c, ldc, min_rows, 8,
            [&](fint i0, fint i1, fint k0, fint k1, double al, double be, double* cc, fint ldcc) {
              nn_rows(isa, i0, i1, n, k1 - k0, al, a + static_cast<idx>(k0) * lda, lda, b + k0,
                      ldb, be, cc, ldcc);
            });
}

void narrow_gemm_tn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc,
                    fint min_rows) {
  if (m <= 0 || n <= 0 || n > kMaxN) return;
  run_split(threads, m, n, k, alpha, beta, c, ldc, min_rows, 4,
            [&](fint i0, fint i1, fint k0, fint k1, double al, double be, double* cc, fint ldcc) {
              tn_rows(isa, i0, i1, n, k1 - k0, al, a + k0, lda, b + k0, ldb, be, cc, ldcc);
            });
}

void small_trsm_left(Isa isa, char uplo, char trans, char diag, fint m, fint n, const double* a,
                     fint lda, double* b, fint ldb) {
  if (m <= 0 || n <= 0) return;
  const bool upper = uplo == 'U' || uplo == 'u';
  const bool tr = !(trans == 'N' || trans == 'n');
  const bool unit = diag == 'U' || diag == 'u';
  switch (isa) {
#if HS_BLAS_X86
    case Isa::Avx512:
      avx512::trsm_small(upper, tr, unit, m, n, a, lda, b, ldb);
      break;
    case Isa::Avx2:
      avx2::trsm_small(upper, tr, unit, m, n, a, lda, b, ldb);
      break;
#else
    case Isa::Avx512:
    case Isa::Avx2:
#endif
    case Isa::Portable:
      trsm_small_portable(upper, tr, unit, m, n, a, lda, b, ldb);
      break;
  }
}

}  // namespace hs::blas
