// Scratch memory of orthogonalize_block_cgs2 against its documented bound. The global operator
// new / delete of this test executable count the live bytes, so the peak during a call is the
// scratch the call allocates (inputs are DenseMatrix, which uses aligned_alloc and is not counted;
// OpenMP and BLAS do not allocate through operator new).
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/error.hpp"
#include "hsolve/types.hpp"
#include "linalg/dense_ops.hpp"

namespace {

std::atomic<long long> g_live{0};
std::atomic<long long> g_peak{0};

constexpr std::size_t kHeader = 16;  // keeps the default new alignment (16)

void* counted_alloc(std::size_t size) {
  void* raw = std::malloc(size + kHeader);
  if (!raw) throw std::bad_alloc();
  *static_cast<std::size_t*>(raw) = size;
  const long long live = g_live.fetch_add(static_cast<long long>(size)) + static_cast<long long>(size);
  long long peak = g_peak.load();
  while (live > peak && !g_peak.compare_exchange_weak(peak, live)) {
  }
  return static_cast<char*>(raw) + kHeader;
}

void counted_free(void* p) noexcept {
  if (!p) return;
  void* raw = static_cast<char*>(p) - kHeader;
  g_live.fetch_sub(static_cast<long long>(*static_cast<std::size_t*>(raw)));
  std::free(raw);
}

}  // namespace

void* operator new(std::size_t size) { return counted_alloc(size); }
void* operator new[](std::size_t size) { return counted_alloc(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(size);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { counted_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { counted_free(p); }

namespace {

using hs::cplx;
using hs::index_t;
namespace la = hs::linalg;

/// Peak bytes allocated through operator new during f() beyond those live before it.
template <class F>
long long peak_scratch(F&& f) {
  const long long base = g_live.load();
  g_peak.store(base);
  f();
  return g_peak.load() - base;
}

template <class T>
class LinalgDenseOpsWorkspace : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LinalgDenseOpsWorkspace, Scalars);

// Review finding: the header promised "at most ~2 MB of slots", but at least 32 reduction slots
// are kept, so wide k x p blocks take more. The documented bound is max(2 MB, 32 s) with
// s = 8 (k p d + p) bytes per slot, plus the coefficient blocks; the measured scratch must stay
// within orthogonalize_block_cgs2_workspace_bytes, which must follow that formula.
TYPED_TEST(LinalgDenseOpsWorkspace, BlockCgsScratchWithinDocumentedBound) {
  using T = TypeParam;
  constexpr double d = hs::is_complex_v<T> ? 2.0 : 1.0;
  struct Shape {
    index_t n, k, p;
  };
  // 4096 rows give 32 row blocks (32 slots) for the wide shapes; the narrow one reaches the 2 MB
  // target only for larger n, the bound covers any n.
  for (const Shape& sh : {Shape{4096, 512, 64}, Shape{4096, 300, 40}, Shape{20000, 24, 6}, Shape{3000, 7, 1}}) {
    const index_t n = sh.n, k = sh.k, p = sh.p;
    hs::DenseMatrix<T> v(n, k), w(n, p);
    la::fill_random(v.data(), n * k, 7, 0);
    la::fill_random(w.data(), n * p, 7, 1);
    std::vector<double> before(static_cast<std::size_t>(p)), after(static_cast<std::size_t>(p));
    la::OrthoOptions two;
    two.reorth = la::Reorth::Always;  // every sweep runs (V need not be orthonormal here)
    const long long used = peak_scratch([&] {
      la::orthogonalize_block_cgs2<T>(n, k, p, v.data(), n, nullptr, 0, w.data(), n, nullptr, 0, nullptr, 0,
                                      before.data(), after.data(), nullptr, two);
    });
    const double bound = static_cast<double>(la::orthogonalize_block_cgs2_workspace_bytes<T>(k, p));
    EXPECT_LE(static_cast<double>(used), bound) << "k=" << k << " p=" << p;
    // The formula of the header: slots <= max(2 MB, 32 s), + one result vector of s bytes, + c1,
    // c2 (k p scalars each) and three norm vectors.
    const double s = 8.0 * (static_cast<double>(k * p) * d + static_cast<double>(p));
    const double documented = std::max(2.0 * 1024 * 1024, 32.0 * s) + s +
                              2.0 * static_cast<double>(k * p) * sizeof(T) + 24.0 * static_cast<double>(p);
    EXPECT_LE(bound, documented) << "k=" << k << " p=" << p;
    if (k == 512 && hs::is_complex_v<T>) {
      // The header's example: 32 slots of 525 KB, i.e. well beyond 2 MB, and actually allocated.
      EXPECT_GT(static_cast<double>(used), 16.5e6);
      EXPECT_LT(bound, 20e6);
    }
  }
  EXPECT_THROW(la::orthogonalize_block_cgs2_workspace_bytes<T>(-1, 2), hs::Error);
}

}  // namespace
