// BLAS hook: narrow dgemm kernels, C = alpha * op(A) * B + beta * C for 1 <= N <= 64 columns,
// and a small triangular solve (blas_hook.hpp). Shapes of MUMPS' multi-right-hand-side solves (N = p = 2..16) and of narrow
// panel updates: memory-bound, so the kernels stream A exactly once and never pack it (OpenBLAS'
// level-3 path copies A into a packed buffer first, which for N <= 8 costs about as much as the
// multiplication; its small-matrix kernels only take M*N*K <= 1e6).
//
// Provenance (docs/BLAS.md): the AVX-512 code is adapted from OpenBLAS 0.3.34, BSD-3-Clause,
// Copyright (c) 2021 The OpenBLAS Project:
//   * NN: the column-group update of kernel/x86_64/dgemv_n_microk_skylakex-4.c (4 columns of A
//     per pass, broadcast x values held in registers, y updated in 8-row zmm steps) generalized
//     from one right-hand side to a group of up to 4, over a row chunk whose partial results live
//     in a small L1-resident buffer; A is read in long unit-stride runs (whole column segments of
//     the chunk), which the hardware prefetchers follow, instead of the 32-row x K tiles of
//     dgemm_small_kernel_nn_skylakex.c (one cache line per column per tile: strided, latency-bound
//     for the large leading dimensions of MUMPS fronts).
//   * TN: dgemm_small_kernel_tn_skylakex.c's scheme - zmm accumulators for a block of C, each a
//     dot product of a column of A with a column of B in 8-lane chunks, masked tails, horizontal
//     reduction - with the block shape changed to 4 columns of A x up to 4 columns of B.
// Every C element is computed in a fixed order that does not depend on the row split (NN: one
// FMA chain over k; TN: 8 lane chains over k, then a fixed reduction tree), so results are
// identical for any thread count.
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>

#include "blas/blas_hook.hpp"

namespace hs::blas {
namespace {

using idx = std::ptrdiff_t;

constexpr int kMaxN = 64;

// Row chunk of the NN kernels: the chunk's partial results (rows x N doubles) stay in L1/L2.
fint nn_row_chunk(fint n) {
  return n <= 8 ? 512 : 256;
}

// Runs f(i0, i1) over [0, m) split into `threads` fixed blocks (multiples of `align` rows).
template <class F>
void split_rows(int threads, fint m, fint align, const F& f) {
  if (threads <= 1 || omp_in_parallel() || m <= align) {
    f(0, m);
    return;
  }
  const fint per = ((m + threads - 1) / threads + align - 1) / align * align;
#pragma omp parallel num_threads(threads)
  {
    const fint i0 = static_cast<fint>(omp_get_thread_num()) * per;
    const fint i1 = std::min<fint>(m, i0 + per);
    if (i0 < i1) f(i0, i1);
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

// ------------------------------------------------------------------------------------- AVX-512

#define HS_AVX512 __attribute__((target("avx512f")))

// t(rows x NG, ld ldt; ldt = rows rounded up to 8, 64-byte aligned, padding rows writable) +=
// A(rows x KQ) * B(KQ x NG): one tile of KQ consecutive columns of A, applied in column order.
template <int KQ, int NG>
HS_AVX512 void nn_tile_avx512(fint rows, const double* a, fint lda, const double* b, fint ldb,
                              double* t, fint ldt) {
  __m512d bb[KQ][NG];
  const double* ap[KQ];
  for (int q = 0; q < KQ; ++q) {
    ap[q] = a + static_cast<idx>(q) * lda;
    for (int j = 0; j < NG; ++j) bb[q][j] = _mm512_set1_pd(b[q + static_cast<idx>(j) * ldb]);
  }
  const fint full = rows & ~7;
  fint i = 0;
  for (; i < full; i += 8) {
    __m512d x[KQ];
    for (int q = 0; q < KQ; ++q) x[q] = _mm512_loadu_pd(ap[q] + i);
    for (int j = 0; j < NG; ++j) {
      double* tj = t + static_cast<idx>(j) * ldt + i;
      __m512d acc = _mm512_load_pd(tj);
      for (int q = 0; q < KQ; ++q) acc = _mm512_fmadd_pd(x[q], bb[q][j], acc);
      _mm512_store_pd(tj, acc);
    }
  }
  if (i < rows) {
    const __mmask8 tail = static_cast<__mmask8>((1u << (rows - i)) - 1u);
    __m512d x[KQ];
    for (int q = 0; q < KQ; ++q) x[q] = _mm512_maskz_loadu_pd(tail, ap[q] + i);
    for (int j = 0; j < NG; ++j) {
      double* tj = t + static_cast<idx>(j) * ldt + i;
      __m512d acc = _mm512_load_pd(tj);
      for (int q = 0; q < KQ; ++q) acc = _mm512_fmadd_pd(x[q], bb[q][j], acc);
      _mm512_store_pd(tj, acc);
    }
  }
}

// t(rows x ng) += A(rows x KQ) * B(KQ x ng), column groups of 4 (the A tile stays in L1).
template <int KQ>
HS_AVX512 void nn_tiles_avx512(fint rows, fint ng, const double* a, fint lda, const double* b,
                               fint ldb, double* t, fint ldt) {
  for (fint j0 = 0; j0 < ng; j0 += 4) {
    const double* bj = b + static_cast<idx>(j0) * ldb;
    double* tj = t + static_cast<idx>(j0) * ldt;
    switch (std::min<fint>(4, ng - j0)) {
      case 4:
        nn_tile_avx512<KQ, 4>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      case 3:
        nn_tile_avx512<KQ, 3>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      case 2:
        nn_tile_avx512<KQ, 2>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      default:
        nn_tile_avx512<KQ, 1>(rows, a, lda, bj, ldb, tj, ldt);
        break;
    }
  }
}

HS_AVX512 void nn_rows_avx512(fint i0, fint i1, fint n, fint k, double alpha, const double* a,
                              fint lda, const double* b, fint ldb, double beta, double* c,
                              fint ldc) {
  const fint rc = nn_row_chunk(n);
  alignas(64) double t[4096];
  for (fint r0 = i0; r0 < i1; r0 += rc) {
    const fint rows = std::min(rc, i1 - r0);
    const fint ldt = (rows + 7) & ~7;
    const fint group = std::max<fint>(4, 4096 / ldt / 4 * 4);  // columns per pass through t
    for (fint g0 = 0; g0 < n; g0 += group) {
      const fint ng = std::min(group, n - g0);
      std::fill(t, t + static_cast<idx>(ldt) * ng, 0.0);
      const double* ar = a + r0;
      const double* bg = b + static_cast<idx>(g0) * ldb;
      fint kk = 0;
      for (; kk + 4 <= k; kk += 4)
        nn_tiles_avx512<4>(rows, ng, ar + static_cast<idx>(kk) * lda, lda, bg + kk, ldb, t, ldt);
      for (; kk < k; ++kk)
        nn_tiles_avx512<1>(rows, ng, ar + static_cast<idx>(kk) * lda, lda, bg + kk, ldb, t, ldt);
      finish_block(rows, ng, alpha, t, ldt, beta, c + r0 + static_cast<idx>(g0) * ldc, ldc);
    }
  }
}

// out(IB x NG, ld IB) = A(:, 0..IB)^T * B(:, 0..NG), A k x IB (lda), B k x NG (ldb).
template <int IB, int NG>
HS_AVX512 void tn_block_avx512(fint k, const double* a, fint lda, const double* b, fint ldb,
                               double* out) {
  __m512d acc[IB][NG];
  for (int r = 0; r < IB; ++r)
    for (int j = 0; j < NG; ++j) acc[r][j] = _mm512_setzero_pd();
  fint kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    __m512d bv[NG];
    for (int j = 0; j < NG; ++j) bv[j] = _mm512_loadu_pd(b + static_cast<idx>(j) * ldb + kk);
    for (int r = 0; r < IB; ++r) {
      const __m512d av = _mm512_loadu_pd(a + static_cast<idx>(r) * lda + kk);
      for (int j = 0; j < NG; ++j) acc[r][j] = _mm512_fmadd_pd(av, bv[j], acc[r][j]);
    }
  }
  if (kk < k) {
    const __mmask8 tail = static_cast<__mmask8>((1u << (k - kk)) - 1u);
    __m512d bv[NG];
    for (int j = 0; j < NG; ++j)
      bv[j] = _mm512_maskz_loadu_pd(tail, b + static_cast<idx>(j) * ldb + kk);
    for (int r = 0; r < IB; ++r) {
      const __m512d av = _mm512_maskz_loadu_pd(tail, a + static_cast<idx>(r) * lda + kk);
      for (int j = 0; j < NG; ++j) acc[r][j] = _mm512_fmadd_pd(av, bv[j], acc[r][j]);
    }
  }
  for (int r = 0; r < IB; ++r)
    for (int j = 0; j < NG; ++j) out[r + j * IB] = _mm512_reduce_add_pd(acc[r][j]);
}

template <int IB>
HS_AVX512 void tn_block_dispatch(int ng, fint k, const double* a, fint lda, const double* b,
                                 fint ldb, double* out) {
  switch (ng) {
    case 4:
      tn_block_avx512<IB, 4>(k, a, lda, b, ldb, out);
      break;
    case 3:
      tn_block_avx512<IB, 3>(k, a, lda, b, ldb, out);
      break;
    case 2:
      tn_block_avx512<IB, 2>(k, a, lda, b, ldb, out);
      break;
    default:
      tn_block_avx512<IB, 1>(k, a, lda, b, ldb, out);
      break;
  }
}

HS_AVX512 void tn_rows_avx512(fint i0, fint i1, fint n, fint k, double alpha, const double* a,
                              fint lda, const double* b, fint ldb, double beta, double* c,
                              fint ldc) {
  double out[16];
  for (fint i = i0; i < i1; i += 4) {
    const int ib = static_cast<int>(std::min<fint>(4, i1 - i));
    const double* ai = a + static_cast<idx>(i) * lda;
    for (fint j0 = 0; j0 < n; j0 += 4) {
      const int ng = static_cast<int>(std::min<fint>(4, n - j0));
      const double* bj = b + static_cast<idx>(j0) * ldb;
      switch (ib) {
        case 4:
          tn_block_dispatch<4>(ng, k, ai, lda, bj, ldb, out);
          break;
        case 3:
          tn_block_dispatch<3>(ng, k, ai, lda, bj, ldb, out);
          break;
        case 2:
          tn_block_dispatch<2>(ng, k, ai, lda, bj, ldb, out);
          break;
        default:
          tn_block_dispatch<1>(ng, k, ai, lda, bj, ldb, out);
          break;
      }
      for (int j = 0; j < ng; ++j)
        for (int r = 0; r < ib; ++r) {
          double& cij = c[i + r + static_cast<idx>(j0 + j) * ldc];
          const double s = out[r + j * ib];
          cij = beta == 0.0 ? alpha * s : alpha * s + beta * cij;
        }
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

struct DotAvx512 {
  HS_AVX512 static double dot(const double* x, const double* y, fint len) {
    __m512d acc = _mm512_setzero_pd();
    fint j = 0;
    for (; j + 8 <= len; j += 8)
      acc = _mm512_fmadd_pd(_mm512_loadu_pd(x + j), _mm512_loadu_pd(y + j), acc);
    if (j < len) {
      const __mmask8 tail = static_cast<__mmask8>((1u << (len - j)) - 1u);
      acc = _mm512_fmadd_pd(_mm512_maskz_loadu_pd(tail, x + j), _mm512_maskz_loadu_pd(tail, y + j),
                            acc);
    }
    return _mm512_reduce_add_pd(acc);
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

HS_AVX512 void trsm_small_avx512(bool upper, bool trans, bool unit, fint m, fint n,
                                 const double* a, fint lda, double* b, fint ldb) {
  if (unit)
    trsm_small_body<DotAvx512, true>(upper, trans, m, n, a, lda, b, ldb);
  else
    trsm_small_body<DotAvx512, false>(upper, trans, m, n, a, lda, b, ldb);
}

#undef HS_AVX512

}  // namespace

void narrow_gemm_nn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc) {
  if (m <= 0 || n <= 0 || n > kMaxN) return;
  const bool avx = isa == Isa::Avx512;
  split_rows(threads, m, 8, [&](fint i0, fint i1) {
    if (avx)
      nn_rows_avx512(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else
      nn_rows_portable(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  });
}

void narrow_gemm_tn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc) {
  if (m <= 0 || n <= 0 || n > kMaxN) return;
  const bool avx = isa == Isa::Avx512;
  split_rows(threads, m, 4, [&](fint i0, fint i1) {
    if (avx)
      tn_rows_avx512(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else
      tn_rows_portable(i0, i1, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  });
}


void small_trsm_left(Isa isa, char uplo, char trans, char diag, fint m, fint n, const double* a,
                     fint lda, double* b, fint ldb) {
  if (m <= 0 || n <= 0) return;
  const bool upper = uplo == 'U' || uplo == 'u';
  const bool tr = !(trans == 'N' || trans == 'n');
  const bool unit = diag == 'U' || diag == 'u';
  if (isa == Isa::Avx512)
    trsm_small_avx512(upper, tr, unit, m, n, a, lda, b, ldb);
  else
    trsm_small_portable(upper, tr, unit, m, n, a, lda, b, ldb);
}

}  // namespace hs::blas
