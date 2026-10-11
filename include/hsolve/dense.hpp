// Column-major dense matrix with 64-byte aligned storage (ld == rows).
// Used for Krylov bases, eigenvector blocks and small projected problems. Never used to
// hold a densified copy of a sparse input matrix (see docs/DESIGN.md §5).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

#include "hsolve/error.hpp"
#include "hsolve/types.hpp"

namespace hs {

template <class T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() noexcept = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    std::size_t bytes = ((n * sizeof(T) + Alignment - 1) / Alignment) * Alignment;
    void* p = std::aligned_alloc(Alignment, bytes);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }

  template <class U>
  bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
    return true;
  }
  template <class U>
  bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
    return false;
  }
};

template <class T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

template <class T>
class DenseMatrix {
 public:
  DenseMatrix() = default;
  /// Zero-initialised rows x cols matrix.
  DenseMatrix(index_t rows, index_t cols) { resize(rows, cols); }

  index_t rows() const noexcept { return rows_; }
  index_t cols() const noexcept { return cols_; }
  index_t ld() const noexcept { return rows_; }
  bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

  T* data() noexcept { return buf_.data(); }
  const T* data() const noexcept { return buf_.data(); }
  T* col(index_t j) noexcept { return buf_.data() + j * rows_; }
  const T* col(index_t j) const noexcept { return buf_.data() + j * rows_; }

  T& operator()(index_t i, index_t j) noexcept { return buf_[static_cast<std::size_t>(i + j * rows_)]; }
  const T& operator()(index_t i, index_t j) const noexcept {
    return buf_[static_cast<std::size_t>(i + j * rows_)];
  }

  /// Reshape to rows x cols, zero-filled (previous contents are discarded).
  void resize(index_t rows, index_t cols) {
    if (rows < 0 || cols < 0) HS_THROW(InvalidInput, "DenseMatrix: negative size " << rows << "x" << cols);
    rows_ = rows;
    cols_ = cols;
    buf_.assign(static_cast<std::size_t>(rows * cols), T(0));
  }

  /// Change the number of columns keeping the leading min(cols, old cols) columns.
  void resize_cols(index_t cols) {
    if (cols < 0) HS_THROW(InvalidInput, "DenseMatrix: negative cols " << cols);
    buf_.resize(static_cast<std::size_t>(rows_ * cols), T(0));
    cols_ = cols;
  }

  void set_zero() { std::fill(buf_.begin(), buf_.end(), T(0)); }

 private:
  index_t rows_ = 0;
  index_t cols_ = 0;
  aligned_vector<T> buf_;
};

}  // namespace hs
