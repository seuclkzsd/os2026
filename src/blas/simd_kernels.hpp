// BLAS hook: the SIMD kernels of narrow_kernels.cpp, written once for a vector traits type and
// compiled once per instruction set. NOT a normal header: narrow_kernels.cpp includes it once per
// ISA (hence no include guard), inside namespace hs::blas::<anonymous>::<isa>, after defining
//   V        the traits: register type reg, width W (doubles per register), mask type and tail(r)
//            (first r lanes), zero / set1 / loadu / load / store (aligned) / maskz_loadu /
//            mask_storeu / fmadd / fnmadd / mul / add, reduce4 (four horizontal sums), kGroup
//            (columns of B per register block) and kRegRows (rows of the NN register path);
//   HS_SIMD  the function attribute that selects the instruction set (target("...")).
// It includes nothing (it is expanded inside a namespace); <immintrin.h> and blas_hook.hpp come
// from narrow_kernels.cpp, as do idx, nn_row_chunk and finish_block.
//
// Provenance (docs/BLAS.md): adapted from OpenBLAS 0.3.34, BSD-3-Clause, Copyright (c) 2021 The
// OpenBLAS Project:
//   * NN: the column-group update of kernel/x86_64/dgemv_n_microk_skylakex-4.c (4 columns of A
//     per pass, broadcast x values held in registers, y updated one register of rows at a time)
//     generalized from one right-hand side to a group, over a row chunk whose partial results live
//     in a small L1-resident buffer; A is read in long unit-stride runs (whole column segments of
//     the chunk), which the hardware prefetchers follow, instead of the 32-row x K tiles of
//     dgemm_small_kernel_nn_skylakex.c (one cache line per column per tile: strided, latency-bound
//     for the large leading dimensions of MUMPS fronts).
//   * TN: dgemm_small_kernel_tn_skylakex.c's scheme - register accumulators for a block of C, each
//     a dot product of a column of A with a column of B in W-lane chunks, masked tails, and four
//     horizontal sums reduced together (its REDUCE_4) - with the block shape changed to 4 columns
//     of A x kGroup columns of B.
//   * NN with at most kRegRows rows: register accumulators over all of k (the small-matrix
//     kernel's structure), no row-chunk buffer.
//   * Small triangular solve: the reference BLAS loop orders (project's own code).
// Every C element is computed in a fixed order that does not depend on the row split (NN: one FMA
// chain over k, then alpha * t + beta * c with separately rounded products, on either NN path; TN:
// W lane chains over k, then the same reduction tree for every row), so results are identical for
// any row split. "Separately rounded" needs -ffp-contract=off (CMake target hsolve_blas): GCC
// contracts C++ by default and would turn the register path's mul + add (vector arithmetic in
// the intrinsics) into an FMA while the chunked path's finish_block (baseline ISA, no FMA) rounds
// both products - test_asmkern_fix checks it with inexact alpha and beta.

using reg = typename V::reg;

// t(rows x NG, ld ldt; ldt = rows rounded up to 8, 64-byte aligned, padding rows writable) +=
// A(rows x KQ) * B(KQ x NG): one tile of KQ consecutive columns of A, applied in column order.
template <int KQ, int NG>
HS_SIMD void nn_tile(fint rows, const double* a, fint lda, const double* b, fint ldb, double* t,
                     fint ldt) {
  reg bb[KQ][NG];
  const double* ap[KQ];
  for (int q = 0; q < KQ; ++q) {
    ap[q] = a + static_cast<idx>(q) * lda;
    for (int j = 0; j < NG; ++j) bb[q][j] = V::set1(b[q + static_cast<idx>(j) * ldb]);
  }
  const fint full = rows - rows % V::W;
  fint i = 0;
  for (; i < full; i += V::W) {
    reg x[KQ];
    for (int q = 0; q < KQ; ++q) x[q] = V::loadu(ap[q] + i);
    for (int j = 0; j < NG; ++j) {
      double* tj = t + static_cast<idx>(j) * ldt + i;
      reg acc = V::load(tj);
      for (int q = 0; q < KQ; ++q) acc = V::fmadd(x[q], bb[q][j], acc);
      V::store(tj, acc);
    }
  }
  if (i < rows) {
    const auto tail = V::tail(rows - i);
    reg x[KQ];
    for (int q = 0; q < KQ; ++q) x[q] = V::maskz_loadu(tail, ap[q] + i);
    for (int j = 0; j < NG; ++j) {
      double* tj = t + static_cast<idx>(j) * ldt + i;
      reg acc = V::load(tj);
      for (int q = 0; q < KQ; ++q) acc = V::fmadd(x[q], bb[q][j], acc);
      V::store(tj, acc);
    }
  }
}

// t(rows x ng) += A(rows x KQ) * B(KQ x ng), column groups of kGroup (the A tile stays in L1).
template <int KQ>
HS_SIMD void nn_tiles(fint rows, fint ng, const double* a, fint lda, const double* b, fint ldb,
                      double* t, fint ldt) {
  for (fint j0 = 0; j0 < ng; j0 += V::kGroup) {
    const double* bj = b + static_cast<idx>(j0) * ldb;
    double* tj = t + static_cast<idx>(j0) * ldt;
    switch (std::min<fint>(V::kGroup, ng - j0)) {
      case 4:
        nn_tile<KQ, 4>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      case 3:
        nn_tile<KQ, 3>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      case 2:
        nn_tile<KQ, 2>(rows, a, lda, bj, ldb, tj, ldt);
        break;
      default:
        nn_tile<KQ, 1>(rows, a, lda, bj, ldb, tj, ldt);
        break;
    }
  }
}

// C(m x NG) = alpha * A(m x k) * B(k x NG) + beta * C for m <= MB * W rows, accumulated in
// registers (no t buffer: for the tiny products of MUMPS' 2-D solves). Per element the same
// arithmetic as the chunked path - one FMA chain over k from 0, then alpha * t + beta * c with
// separately rounded products (no contraction: -ffp-contract=off) - so the result does not
// depend on which path a row takes.
template <int MB, int NG>
HS_SIMD void nn_reg(fint m, fint k, double alpha, const double* a, fint lda, const double* b,
                    fint ldb, double beta, double* c, fint ldc) {
  const auto last = V::tail(m - V::W * (MB - 1));
  reg acc[MB][NG];
  for (int r = 0; r < MB; ++r)
    for (int j = 0; j < NG; ++j) acc[r][j] = V::zero();
  for (fint kk = 0; kk < k; ++kk) {
    const double* ak = a + static_cast<idx>(kk) * lda;
    reg x[MB];
    for (int r = 0; r < MB - 1; ++r) x[r] = V::loadu(ak + V::W * r);
    x[MB - 1] = V::maskz_loadu(last, ak + V::W * (MB - 1));
    for (int j = 0; j < NG; ++j) {
      const reg bj = V::set1(b[kk + static_cast<idx>(j) * ldb]);
      for (int r = 0; r < MB; ++r) acc[r][j] = V::fmadd(x[r], bj, acc[r][j]);
    }
  }
  const reg va = V::set1(alpha);
  const reg vb = V::set1(beta);
  for (int j = 0; j < NG; ++j)
    for (int r = 0; r < MB; ++r) {
      double* cj = c + static_cast<idx>(j) * ldc + V::W * r;
      const auto mk = r == MB - 1 ? last : V::tail(V::W);
      reg v = V::mul(va, acc[r][j]);
      if (beta != 0.0) v = V::add(v, V::mul(vb, V::maskz_loadu(mk, cj)));
      V::mask_storeu(cj, mk, v);
    }
}

template <int MB>
HS_SIMD void nn_reg_dispatch(fint m, fint n, fint k, double alpha, const double* a, fint lda,
                             const double* b, fint ldb, double beta, double* c, fint ldc) {
  for (fint j0 = 0; j0 < n; j0 += V::kGroup) {
    const double* bj = b + static_cast<idx>(j0) * ldb;
    double* cj = c + static_cast<idx>(j0) * ldc;
    switch (std::min<fint>(V::kGroup, n - j0)) {
      case 4:
        nn_reg<MB, 4>(m, k, alpha, a, lda, bj, ldb, beta, cj, ldc);
        break;
      case 3:
        nn_reg<MB, 3>(m, k, alpha, a, lda, bj, ldb, beta, cj, ldc);
        break;
      case 2:
        nn_reg<MB, 2>(m, k, alpha, a, lda, bj, ldb, beta, cj, ldc);
        break;
      default:
        nn_reg<MB, 1>(m, k, alpha, a, lda, bj, ldb, beta, cj, ldc);
        break;
    }
  }
}

// Rows [i0, i1) of C = alpha * A * B + beta * C (A m x k, B k x n).
HS_SIMD void nn_rows(fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
                     const double* b, fint ldb, double beta, double* c, fint ldc) {
  const fint m = i1 - i0;
  static_assert(V::kRegRows <= 4 * V::W, "register path: at most 4 registers of rows");
  if (m <= V::kRegRows) {
    const double* ai = a + i0;
    double* ci = c + i0;
    switch ((m + V::W - 1) / V::W) {
      case 4:
        nn_reg_dispatch<4>(m, n, k, alpha, ai, lda, b, ldb, beta, ci, ldc);
        break;
      case 3:
        nn_reg_dispatch<3>(m, n, k, alpha, ai, lda, b, ldb, beta, ci, ldc);
        break;
      case 2:
        nn_reg_dispatch<2>(m, n, k, alpha, ai, lda, b, ldb, beta, ci, ldc);
        break;
      default:
        nn_reg_dispatch<1>(m, n, k, alpha, ai, lda, b, ldb, beta, ci, ldc);
        break;
    }
    return;
  }
  const fint rc = nn_row_chunk(n);
  alignas(64) double t[4096];
  for (fint r0 = i0; r0 < i1; r0 += rc) {
    const fint rows = std::min(rc, i1 - r0);
    const fint ldt = (rows + 7) & ~7;
    const fint group = std::max<fint>(4, 4096 / ldt / 4 * 4);  // columns per pass through t
    for (fint g0 = 0; g0 < n; g0 += group) {
      const fint ng = std::min(group, n - g0);
      for (idx q = 0; q < static_cast<idx>(ldt) * ng; ++q) t[q] = 0.0;
      const double* ar = a + r0;
      const double* bg = b + static_cast<idx>(g0) * ldb;
      fint kk = 0;
      for (; kk + 4 <= k; kk += 4)
        nn_tiles<4>(rows, ng, ar + static_cast<idx>(kk) * lda, lda, bg + kk, ldb, t, ldt);
      for (; kk < k; ++kk)
        nn_tiles<1>(rows, ng, ar + static_cast<idx>(kk) * lda, lda, bg + kk, ldb, t, ldt);
      finish_block(rows, ng, alpha, t, ldt, beta, c + r0 + static_cast<idx>(g0) * ldc, ldc);
    }
  }
}

// C(i..i+IB, j..j+NG) = alpha * A(:, i..i+IB)^T * B(:, j..j+NG) + beta * C, A k x IB (lda),
// B k x NG (ldb), IB <= 4 (missing rows reduce zero accumulators: every row of C gets the same
// reduction tree wherever it lies).
template <int IB, int NG>
HS_SIMD void tn_block(fint k, const double* a, fint lda, const double* b, fint ldb, double alpha,
                      double beta, double* c, fint ldc) {
  reg acc[4][NG];
  for (int r = 0; r < 4; ++r)
    for (int j = 0; j < NG; ++j) acc[r][j] = V::zero();
  fint kk = 0;
  for (; kk + V::W <= k; kk += V::W) {
    reg bv[NG];
    for (int j = 0; j < NG; ++j) bv[j] = V::loadu(b + static_cast<idx>(j) * ldb + kk);
    for (int r = 0; r < IB; ++r) {
      const reg av = V::loadu(a + static_cast<idx>(r) * lda + kk);
      for (int j = 0; j < NG; ++j) acc[r][j] = V::fmadd(av, bv[j], acc[r][j]);
    }
  }
  if (kk < k) {
    const auto tail = V::tail(k - kk);
    reg bv[NG];
    for (int j = 0; j < NG; ++j) bv[j] = V::maskz_loadu(tail, b + static_cast<idx>(j) * ldb + kk);
    for (int r = 0; r < IB; ++r) {
      const reg av = V::maskz_loadu(tail, a + static_cast<idx>(r) * lda + kk);
      for (int j = 0; j < NG; ++j) acc[r][j] = V::fmadd(av, bv[j], acc[r][j]);
    }
  }
  alignas(32) double sums[4];
  for (int j = 0; j < NG; ++j) {
    double* cj = c + static_cast<idx>(j) * ldc;
    V::reduce4(acc[0][j], acc[1][j], acc[2][j], acc[3][j], sums);
    for (int r = 0; r < IB; ++r)
      cj[r] = beta == 0.0 ? alpha * sums[r] : alpha * sums[r] + beta * cj[r];
  }
}

template <int IB>
HS_SIMD void tn_block_dispatch(int ng, fint k, const double* a, fint lda, const double* b,
                               fint ldb, double alpha, double beta, double* c, fint ldc) {
  switch (ng) {
    case 4:
      tn_block<IB, 4>(k, a, lda, b, ldb, alpha, beta, c, ldc);
      break;
    case 3:
      tn_block<IB, 3>(k, a, lda, b, ldb, alpha, beta, c, ldc);
      break;
    case 2:
      tn_block<IB, 2>(k, a, lda, b, ldb, alpha, beta, c, ldc);
      break;
    default:
      tn_block<IB, 1>(k, a, lda, b, ldb, alpha, beta, c, ldc);
      break;
  }
}

// Rows [i0, i1) of C = alpha * A^T * B + beta * C (A k x m, B k x n).
HS_SIMD void tn_rows(fint i0, fint i1, fint n, fint k, double alpha, const double* a, fint lda,
                     const double* b, fint ldb, double beta, double* c, fint ldc) {
  for (fint i = i0; i < i1; i += 4) {
    const int ib = static_cast<int>(std::min<fint>(4, i1 - i));
    const double* ai = a + static_cast<idx>(i) * lda;
    for (fint j0 = 0; j0 < n; j0 += V::kGroup) {
      const int ng = static_cast<int>(std::min<fint>(V::kGroup, n - j0));
      const double* bj = b + static_cast<idx>(j0) * ldb;
      double* cij = c + i + static_cast<idx>(j0) * ldc;
      switch (ib) {
        case 4:
          tn_block_dispatch<4>(ng, k, ai, lda, bj, ldb, alpha, beta, cij, ldc);
          break;
        case 3:
          tn_block_dispatch<3>(ng, k, ai, lda, bj, ldb, alpha, beta, cij, ldc);
          break;
        case 2:
          tn_block_dispatch<2>(ng, k, ai, lda, bj, ldb, alpha, beta, cij, ldc);
          break;
        default:
          tn_block_dispatch<1>(ng, k, ai, lda, bj, ldb, alpha, beta, cij, ldc);
          break;
      }
    }
  }
}

// ------------------------------------------------------------------------- small triangular solve
// B(m x n) := op(A)^{-1} B (side L, alpha = 1), the reference BLAS loop orders with up to 4
// right-hand sides per pass, so that every chunk of the triangle is loaded once for all of them
// (MUMPS solves 1, 2 or 4 at a time). The unit diagonal is never read.

// op(A) = A: x_j for the columns of the group, then rows [i0, i1) of each column updated with
// column j of A (b_i -= a_ij x_j, one FMA per element): forward for lower, backward for upper.
template <bool kUnit>
HS_SIMD void trsm_col(bool upper, fint m, fint n, const double* a, fint lda, double* b,
                      fint ldb) {
  for (fint c0 = 0; c0 < n; c0 += 4) {
    const int nc = static_cast<int>(std::min<fint>(4, n - c0));
    double* bc[4];
    for (int q = 0; q < nc; ++q) bc[q] = b + static_cast<idx>(c0 + q) * ldb;
    for (fint step = 0; step < m; ++step) {
      const fint j = upper ? m - 1 - step : step;
      const double* aj = a + static_cast<idx>(j) * lda;
      reg xs[4];
      for (int q = 0; q < nc; ++q) {
        double x = bc[q][j];
        if (!kUnit) x /= aj[j];
        bc[q][j] = x;
        xs[q] = V::set1(x);
      }
      const fint i0 = upper ? 0 : j + 1;
      const fint i1 = upper ? j : m;
      for (fint i = i0; i < i1; i += V::W) {
        const auto mk = V::tail(std::min<fint>(V::W, i1 - i));
        const reg av = V::maskz_loadu(mk, aj + i);
        for (int q = 0; q < nc; ++q) {
          const reg bv = V::maskz_loadu(mk, bc[q] + i);
          V::mask_storeu(bc[q] + i, mk, V::fnmadd(av, xs[q], bv));
        }
      }
    }
  }
}

// op(A) = A^T: x_i = (b_i - A(j-range, i) . x(j-range)) / a_ii, the dot products of a group
// computed together and reduced with reduce4: forward for upper A (A^T lower), backward for lower.
template <bool kUnit>
HS_SIMD void trsm_dot(bool upper, fint m, fint n, const double* a, fint lda, double* b,
                      fint ldb) {
  for (fint c0 = 0; c0 < n; c0 += 4) {
    const int nc = static_cast<int>(std::min<fint>(4, n - c0));
    double* bc[4];
    for (int q = 0; q < nc; ++q) bc[q] = b + static_cast<idx>(c0 + q) * ldb;
    alignas(32) double sums[4];
    for (fint step = 0; step < m; ++step) {
      const fint i = upper ? step : m - 1 - step;
      const double* ai = a + static_cast<idx>(i) * lda;
      const fint j0 = upper ? 0 : i + 1;
      const fint j1 = upper ? i : m;
      reg acc[4] = {V::zero(), V::zero(), V::zero(), V::zero()};
      for (fint j = j0; j < j1; j += V::W) {
        const auto mk = V::tail(std::min<fint>(V::W, j1 - j));
        const reg av = V::maskz_loadu(mk, ai + j);
        for (int q = 0; q < nc; ++q)
          acc[q] = V::fmadd(av, V::maskz_loadu(mk, bc[q] + j), acc[q]);
      }
      V::reduce4(acc[0], acc[1], acc[2], acc[3], sums);
      for (int q = 0; q < nc; ++q) {
        double x = bc[q][i] - sums[q];
        if (!kUnit) x /= ai[i];
        bc[q][i] = x;
      }
    }
  }
}

HS_SIMD void trsm_small(bool upper, bool trans, bool unit, fint m, fint n, const double* a,
                        fint lda, double* b, fint ldb) {
  if (trans) {
    if (unit)
      trsm_dot<true>(upper, m, n, a, lda, b, ldb);
    else
      trsm_dot<false>(upper, m, n, a, lda, b, ldb);
  } else {
    if (unit)
      trsm_col<true>(upper, m, n, a, lda, b, ldb);
    else
      trsm_col<false>(upper, m, n, a, lda, b, ldb);
  }
}
