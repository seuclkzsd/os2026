#include "eigen/general_schur.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "hsolve/error.hpp"
#include "linalg/blas_lapack.hpp"

namespace hs::eigen_detail {

namespace {

using lapack::fint;

/// Sort key, smaller is better.
double rank_key(const cplx& value, Which which) {
  switch (which) {
    case Which::LargestMagnitude:
      return -std::abs(value);
    case Which::LargestAlgebraic:
      return -value.real();
    case Which::SmallestAlgebraic:
      return value.real();
  }
  HS_THROW(Internal, "unknown Which value");
}

}  // namespace

template <class T>
ProjectedSchur<T>::ProjectedSchur(index_t m, const T* h, index_t ldh)
    : s_(m, m), q_(m, m), w_(static_cast<std::size_t>(m)) {
  HS_ASSERT(m >= 1 && ldh >= m, "projected matrix " << m << " x " << m << ", ld " << ldh);
  for (index_t j = 0; j < m; ++j) std::copy(h + j * ldh, h + j * ldh + m, s_.col(j));
  const auto mm = static_cast<fint>(m);
  lapack::gees(mm, s_.data(), mm, w_.data(), q_.data(), mm);
  refresh_blocks();
}

template <class T>
void ProjectedSchur<T>::refresh_blocks() {
  const index_t m = order();
  blocks_.clear();
  for (index_t i = 0; i < m;) {
    SchurBlock blk;
    blk.start = i;
    if constexpr (is_complex_v<T>) {
      blk.value = w_[i];
    } else {
      // Standardized real Schur form: a nonzero subdiagonal entry marks a 2x2 block, and such
      // a block always holds a complex-conjugate pair (LAPACK triangularizes real pairs).
      if (i + 1 < m && s_(i + 1, i) != 0.0) {
        blk.size = 2;
        blk.value = cplx(w_[i].real(), std::abs(w_[i].imag()));
      } else {
        blk.value = cplx(s_(i, i), 0.0);
      }
    }
    blocks_.push_back(blk);
    i += blk.size;
  }
}

template <class T>
bool ProjectedSchur<T>::is_boundary(index_t p) const {
  HS_ASSERT(p >= 0 && p <= order(), "boundary " << p << " outside [0, " << order() << "]");
  if constexpr (is_complex_v<T>) {
    return true;
  } else {
    return p == 0 || p == order() || s_(p, p - 1) == 0.0;
  }
}

template <class T>
std::vector<index_t> ProjectedSchur<T>::ranking(Which which) const {
  std::vector<double> key(blocks_.size());
  for (std::size_t b = 0; b < blocks_.size(); ++b) key[b] = rank_key(blocks_[b].value, which);
  std::vector<index_t> order(blocks_.size());
  std::iota(order.begin(), order.end(), index_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&key](index_t a, index_t b) { return key[a] < key[b]; });
  return order;
}

template <class T>
bool ProjectedSchur<T>::reorder_to_front(const std::vector<char>& selected_blocks) {
  HS_ASSERT(selected_blocks.size() == blocks_.size(),
            "one flag per block expected: " << selected_blocks.size() << " vs " << blocks_.size());
  const index_t m = order();
  // LAPACK wants one flag per eigenvalue; flagging both members of a pair keeps it together.
  std::vector<fint> select(static_cast<std::size_t>(m), 0);
  for (std::size_t b = 0; b < blocks_.size(); ++b) {
    if (!selected_blocks[b]) continue;
    for (index_t i = 0; i < blocks_[b].size; ++i) select[blocks_[b].start + i] = 1;
  }
  const auto mm = static_cast<fint>(m);
  fint info = 0;
  lapack::trsen(mm, s_.data(), mm, q_.data(), mm, select, w_.data(), &info);
  refresh_blocks();
  return info == 0;
}

template <class T>
RitzVectors<T> ProjectedSchur<T>::ritz_vectors(const T* b) const {
  const index_t m = order();
  const auto mm = static_cast<fint>(m);
  RitzVectors<T> r;
  r.y = q_;  // back-transformation input: eigenvectors of S become eigenvectors of H = Q S Q^H
  DenseMatrix<T> t = s_;  // ztrevc takes a writable T (it restores it); keep S untouched
  lapack::trevc_backtransform(mm, t.data(), mm, r.y.data(), mm);
  r.estimate.assign(static_cast<std::size_t>(m), 0.0);
  for (const SchurBlock& blk : blocks_) {
    T* y0 = r.y.col(blk.start);
    if (blk.size == 1) {
      double nrm = 0.0;
      for (index_t i = 0; i < m; ++i) nrm += abs2(y0[i]);
      nrm = std::sqrt(nrm);
      for (index_t i = 0; i < m; ++i) y0[i] /= nrm;
      r.estimate[blk.start] = std::abs(row_times_column(m, b, y0));
    } else {
      // Real pair: y = Re + i Im; b is real, so |b^T y|^2 = (b^T Re)^2 + (b^T Im)^2.
      T* y1 = r.y.col(blk.start + 1);
      double nrm = 0.0;
      for (index_t i = 0; i < m; ++i) nrm += abs2(y0[i]) + abs2(y1[i]);
      nrm = std::sqrt(nrm);
      for (index_t i = 0; i < m; ++i) {
        y0[i] /= nrm;
        y1[i] /= nrm;
      }
      const double est =
          std::hypot(std::abs(row_times_column(m, b, y0)), std::abs(row_times_column(m, b, y1)));
      r.estimate[blk.start] = est;
      r.estimate[blk.start + 1] = est;
    }
  }
  return r;
}

index_t blocks_for_count(const std::vector<SchurBlock>& blocks, const std::vector<index_t>& ranked,
                         index_t count, index_t* held) {
  index_t nb = 0, total = 0;
  while (total < count && nb < static_cast<index_t>(ranked.size())) {
    total += blocks[ranked[nb]].size;
    ++nb;
  }
  HS_ASSERT(total >= count, "only " << total << " eigenvalues available, " << count << " needed");
  *held = total;
  return nb;
}

#define HS_INSTANTIATE_GENERAL_SCHUR(T) \
  template class ProjectedSchur<T>;     \
  template struct RitzVectors<T>;

HS_FOR_EACH_SCALAR(HS_INSTANTIATE_GENERAL_SCHUR)
#undef HS_INSTANTIATE_GENERAL_SCHUR

}  // namespace hs::eigen_detail
