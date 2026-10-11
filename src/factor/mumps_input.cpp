#include "factor/mumps_input.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "hsolve/error.hpp"

namespace hs::mumps {

namespace {

constexpr std::uint8_t kHasRe = 1;
constexpr std::uint8_t kHasIm = 2;

// For every entry of the pencil's upper pattern, record whether m stores a nonzero real and/or
// imaginary part there. Both the pattern rows and the CSR rows are sorted ascending and the
// pattern contains upper(m), so one merge walk per row suffices. O(nnz).
void mark_parts(const ShiftedPencil<cplx>& pencil, const CsrMatrix<cplx>& m,
                std::vector<std::uint8_t>& flags) {
  const auto& prow = pencil.rowptr();
  const auto& pcol = pencil.colind();
  const index_t n = pencil.order();
#pragma omp parallel for schedule(dynamic, 512)
  for (index_t i = 0; i < n; ++i) {
    const auto* row_begin = m.colind.data() + m.rowptr[i];
    const auto* row_end = m.colind.data() + m.rowptr[i + 1];
    index_t q = m.rowptr[i] +
                (std::lower_bound(row_begin, row_end, static_cast<std::int32_t>(i)) - row_begin);
    index_t o = prow[i];
    for (; q < m.rowptr[i + 1]; ++q) {
      while (pcol[o] != m.colind[q]) ++o;
      if (m.values[q].real() != 0.0) flags[o] |= kHasRe;
      if (m.values[q].imag() != 0.0) flags[o] |= kHasIm;
    }
  }
}

}  // namespace

template <class T, class U>
Input<T, U>::Input(const ShiftedPencil<T>& pencil) : pencil_(&pencil) {
  const index_t n = pencil.order();
  const auto& prow = pencil.rowptr();
  const auto& pcol = pencil.colind();
  constexpr index_t kMaxOrder = std::numeric_limits<int>::max();  // MUMPS_INT is 32-bit

  if constexpr (!is_complex_v<T>) {
    if (n > kMaxOrder)
      HS_THROW(Unsupported, "MUMPS input: order " << n << " exceeds the 32-bit MUMPS_INT range");
    order_ = static_cast<int>(n);
    const index_t nnz = pencil.nnz();
    irn_.resize(static_cast<std::size_t>(nnz));
    jcn_.resize(static_cast<std::size_t>(nnz));
    values_.resize(static_cast<std::size_t>(nnz));
#pragma omp parallel for schedule(dynamic, 512)
    for (index_t i = 0; i < n; ++i) {
      for (index_t p = prow[i]; p < prow[i + 1]; ++p) {
        irn_[p] = static_cast<int>(i + 1);
        jcn_[p] = pcol[p] + 1;
      }
    }
  } else {
    if (2 * n > kMaxOrder)
      HS_THROW(Unsupported, "MUMPS input: embedding order 2n = "
                                << 2 * n << " exceeds the 32-bit MUMPS_INT range");
    order_ = static_cast<int>(2 * n);

    // Which real / imaginary parts can ever be nonzero (alpha, beta are real).
    std::vector<std::uint8_t> flags(static_cast<std::size_t>(pencil.nnz()), 0);
    mark_parts(pencil, pencil.a(), flags);
    if (pencil.b()) mark_parts(pencil, *pencil.b(), flags);  // B = I adds only real diagonals

    // Pass 1, kept source positions in pattern (row-major) order: the diagonal (first entry of
    // each pattern row) always keeps its real part; its imaginary part is zero by Hermitian
    // symmetry.
    for (index_t i = 0; i < n; ++i) {
      for (index_t p = prow[i]; p < prow[i + 1]; ++p) {
        const bool diagonal = (p == prow[i]);
        if (diagonal || (flags[p] & kHasRe)) re_src_.push_back(p);
        if (!diagonal && (flags[p] & kHasIm)) im_src_.push_back(p);
      }
    }
    flags = {};
    re_src_.shrink_to_fit();
    im_src_.shrink_to_fit();

    // Layout of the 2*nre + 2*nim upper-triangle entries of E:
    //   [0, nre)               Re K(i, j) at (i, j)
    //   [nre, 2 nre)           Re K(i, j) at (n+i, n+j)
    //   [2 nre, 2 nre + nim)   -Im K(i, j) at (i, n+j)     (j > i)
    //   [2 nre + nim, end)     +Im K(i, j) at (j, n+i)     (mirror of the strictly lower K(j, i))
    const index_t nre = static_cast<index_t>(re_src_.size());
    const index_t nim = static_cast<index_t>(im_src_.size());
    const std::size_t total = static_cast<std::size_t>(2 * nre + 2 * nim);
    irn_.resize(total);
    jcn_.resize(total);
    values_.resize(total);
    // Pass 2: the same row sweep meets the kept positions in the same order and supplies rows.
    const int ni = static_cast<int>(n);
    index_t kr = 0, ki = 0;
    for (index_t row = 0; row < n; ++row) {
      const int i = static_cast<int>(row);
      for (index_t p = prow[row]; p < prow[row + 1]; ++p) {
        const int j = pcol[p];
        if (kr < nre && re_src_[kr] == p) {
          irn_[kr] = i + 1;
          jcn_[kr] = j + 1;
          irn_[nre + kr] = ni + i + 1;
          jcn_[nre + kr] = ni + j + 1;
          ++kr;
        }
        if (ki < nim && im_src_[ki] == p) {
          irn_[2 * nre + ki] = i + 1;
          jcn_[2 * nre + ki] = ni + j + 1;
          irn_[2 * nre + nim + ki] = j + 1;
          jcn_[2 * nre + nim + ki] = ni + i + 1;
          ++ki;
        }
      }
    }
    scratch_.resize(static_cast<std::size_t>(pencil.nnz()));
  }
}

template <class T, class U>
void Input<T, U>::assemble(double alpha, double beta) {
  if constexpr (!is_complex_v<T>) {
    pencil_->assemble(alpha, beta, values_.data());
  } else {
    pencil_->assemble(alpha, beta, scratch_.data());
    const index_t nre = static_cast<index_t>(re_src_.size());
    const index_t nim = static_cast<index_t>(im_src_.size());
    U* v = values_.data();
#pragma omp parallel for schedule(static)
    for (index_t k = 0; k < nre; ++k) {
      const U re = scratch_[re_src_[k]].real();
      v[k] = re;
      v[nre + k] = re;
    }
#pragma omp parallel for schedule(static)
    for (index_t k = 0; k < nim; ++k) {
      const U im = scratch_[im_src_[k]].imag();
      v[2 * nre + k] = -im;
      v[2 * nre + nim + k] = im;
    }
  }
}

#define HS_INSTANTIATE_MUMPS_INPUT(T) \
  template class Input<T, double>;    \
  template class Input<T, float>;

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_MUMPS_INPUT)
#undef HS_INSTANTIATE_MUMPS_INPUT

}  // namespace hs::mumps
