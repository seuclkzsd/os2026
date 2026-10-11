// block_size = 1 must keep the single-vector thick-restart Lanczos solver of the base commit
// 17a2a79 bit for bit (module eigen, P3 "blockks"). This file pins that behaviour:
//   * exact integers (operator applications, restarts, converged count, history length) of a set
//     of configurations covering both scalar types, all `which` modes, the B inner product in both
//     store modes, deflation, 100+ restarts, a user start vector, a full-space cycle and the
//     row-blocked basis rotation (n > 4096): enforced everywhere;
//   * a 64-bit FNV-1a hash of every output bit (theta, estimate, vectors, ritz_all, history):
//     enforced where the platform fingerprint (bits of the BLAS/LAPACK and OpenMP-reduction
//     kernels the solver uses, on fixed inputs) matches one recorded with the reference
//     generator below; elsewhere the hash check is skipped with a note (the reduction order of
//     linalg::dot depends on the OpenMP thread count, BLAS kernels on the CPU, so the bits of any
//     run depend on the platform). Recorded for 1 and 2 threads, which covers the project's test
//     runs (tools/dev/test.sh with HS_CORES of 2 cores).
// Reference generator: compile this file against the base solver, e.g.
//   git show 17a2a79:src/eigen/lanczos_krylov_schur.cpp > base.cpp
//   c++ <flags of test_lanczos> -c base.cpp; link base.o + this file's object before libhsolve.a
//   (base.o then provides hs::lanczos_krylov_schur, the archive's solver is not pulled in) and
//   run it with HS_LANCZOS_GOLDEN_PRINT=1 under guard.py --threads 1 and --threads 2; it prints
//   the tables below. Recorded on the project server (Xeon Gold 6336Y, conda env os2026
//   OpenBLAS, -march=native release build). The same program linked against the current library
//   printed identical tables at 72195c9.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hsolve/eigen.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::HermitianEigResult;
using hs::index_t;
using hs::KrylovSchurOptions;
using hs::Which;

/// FNV-1a 64 over raw bytes.
class Fnv {
 public:
  void bytes(const void* p, std::size_t len) {
    const auto* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < len; ++i) {
      h_ ^= c[i];
      h_ *= 1099511628211ull;
    }
  }
  template <class X>
  void value(const X& x) {
    bytes(&x, sizeof(X));
  }
  template <class X>
  void vec(const std::vector<X>& v) {
    value(v.size());
    if (!v.empty()) bytes(v.data(), v.size() * sizeof(X));
  }
  template <class T>
  void mat(const DenseMatrix<T>& m) {
    value(m.rows());
    value(m.cols());
    bytes(m.data(), static_cast<std::size_t>(m.rows() * m.cols()) * sizeof(T));
  }
  std::uint64_t get() const { return h_; }

 private:
  std::uint64_t h_ = 1469598103934665603ull;
};

template <class T>
std::uint64_t result_hash(const HermitianEigResult<T>& r) {
  Fnv f;
  f.vec(r.theta);
  f.vec(r.estimate);
  f.mat(r.vectors);
  f.vec(r.ritz_all);
  f.value(r.nconv);
  f.value(r.restarts);
  f.value(r.op_applies);
  f.value(r.converged);
  for (const auto& h : r.history) {
    f.value(h.restart);
    f.value(h.nconv);
    f.value(h.max_estimate);
    f.value(h.op_applies);
  }
  return f.get();
}

/// Bits of the kernels the solver's results depend on, on fixed inputs: OpenMP reductions
/// (linalg::dot, norm2), CGS2, dense operator applications (gemm, getrs) and heev.
template <class T>
std::uint64_t platform_fingerprint() {
  Fnv f;
  for (index_t n : {120, 1000, 9001}) {
    std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    hs::linalg::fill_random(x.data(), n, 11, 0);
    hs::linalg::fill_random(y.data(), n, 11, 1);
    f.value(hs::linalg::dot(n, x.data(), y.data()));
    f.value(hs::linalg::norm2(n, x.data()));
  }
  const index_t n = 120, k = 10;
  const auto a = hs::test::to_dense(hs::test::random_hermitian<T>(n, 0.08, 61));
  const auto b = hs::test::to_dense(hs::test::random_hpd<T>(n, 0.08, 62));
  hs::test::DenseOperator<T> aop(a);
  hs::test::DenseShiftInvertOperator<T> si(a, &b, 0.05);
  DenseMatrix<T> x(n, k), y(n, k);
  for (index_t j = 0; j < k; ++j) hs::linalg::fill_random(x.col(j), n, 12, j);
  aop.apply(x.data(), n, y.data(), n, 1);
  f.mat(y);
  si.apply(x.data(), n, y.data(), n, k);
  f.mat(y);
  std::vector<T> w(x.col(k - 1), x.col(k - 1) + n), h(static_cast<std::size_t>(k));
  f.value(hs::linalg::orthogonalize_cgs2<T>(n, k - 1, x.data(), n, nullptr, n, w.data(), nullptr,
                                            h.data()));
  f.vec(w);
  DenseMatrix<T> s(30, 30);
  for (index_t j = 0; j < 30; ++j)
    for (index_t i = 0; i < 30; ++i) s(i, j) = a(i, j);
  std::vector<double> ev(30);
  hs::lapack::heev('V', 'L', 30, s.data(), 30, ev.data());
  f.vec(ev);
  f.mat(s);
  return f.get();
}

/// One configuration's pinned outcome.
struct Pinned {
  index_t op_applies, restarts, nconv, history;
};

template <class T>
struct Reference;

using HashTable = std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>>;

// Generated by the reference generator (see the file comment) from 17a2a79. The counts were the
// same at 1, 2, 4 and 16 OpenMP threads. Hashes: {platform fingerprint, one hash per
// configuration} for 1 and 2 threads. With 3 or more threads the OpenMP reductions of linalg::dot
// combine the partial sums in thread-arrival order, so bits (fingerprint included) vary from run
// to run (measured at 4 and 16 threads); such a fingerprint matches no entry.
template <>
struct Reference<double> {
  static std::vector<Pinned> counts() {
    return {{112, 11, 8, 12}, {120, 12, 8, 13}, {120, 12, 8, 13}, {38, 9, 5, 10},
            {38, 9, 5, 10},   {39, 7, 4, 8},    {39, 7, 4, 8},    {948, 132, 10, 133},
            {100, 10, 4, 11}, {20, 0, 4, 1},    {40, 7, 5, 8},    {48, 9, 5, 10}};
  }
  static HashTable hashes() {
    return {{0x48c42e4e2b62a6daull,  // 1 thread
             {0xd323a15b2be8cbbdull, 0xcd26581167222bffull, 0x8a9647487de40ce6ull,
              0xb813dee3478b374aull, 0x710d6c50bbaa3b57ull, 0x57c3110e5942cf3aull,
              0xcff3b38aed233e5eull, 0xc3ea02bea1566a00ull, 0xf356381a3bac37cfull,
              0xa5d3b70b749c1175ull, 0x32523877bcb90ceeull, 0x5ef742e136de827eull}},
            {0x0f7bbc15bc9be297ull,  // 2 threads
             {0xe6a437088f09d0f7ull, 0x2e0d625794a9ab99ull, 0xb8a43330746c25c2ull,
              0x09fb7802a21bea9full, 0xdc1611adc319daf8ull, 0x3f475de69ec771f8ull,
              0x594300f50fa079e7ull, 0xebb4c329628b8b73ull, 0x024b7f3db620651full,
              0x25f9861bcaaa9706ull, 0x6599fa1e85b3cc27ull, 0xc752e0ebc46be421ull}}};
  }
};
template <>
struct Reference<cplx> {
  static std::vector<Pinned> counts() {
    return {{120, 12, 8, 13}, {120, 12, 8, 13}, {120, 12, 8, 13}, {35, 8, 5, 9},
            {35, 8, 5, 9},    {39, 7, 4, 8},    {39, 7, 4, 8},    {934, 130, 10, 131},
            {108, 11, 4, 12}, {20, 0, 4, 1},    {40, 7, 5, 8},    {44, 8, 5, 9}};
  }
  static HashTable hashes() {
    return {{0x6fa3160feba245ffull,  // 1 thread
             {0x8723b24bc3b35b75ull, 0x0d106c5cd07f94d8ull, 0x9e981b9001d7a61cull,
              0x18d64b423e37625aull, 0x4925750ef0f765beull, 0x7784b44cdd64f787ull,
              0x2e100d9abb4f1cb2ull, 0xfd66d94035b8f5bbull, 0xc1ee8a163671e457ull,
              0xa7cd7106b7186b23ull, 0x24a6f768267647b1ull, 0xc7e53f70e454f558ull}},
            {0x800b37088663789full,  // 2 threads
             {0x3a6e144565ef89e3ull, 0x25d1e6471e6094bcull, 0xb60312508bda30e3ull,
              0xee1fa6e7067ee8f7ull, 0xb6c61276f4493a53ull, 0x9d70567ef6f03c9dull,
              0x57ba903de2363218ull, 0x3b1c9ff304b40939ull, 0x953d7fcd506823ebull,
              0xc0a14d10ffa80e7aull, 0xfe3ef286da2842a5ull, 0x89658a7290570e9aull}}};
  }
};

/// The configurations, in a fixed order (index = position in the reference tables).
template <class T>
std::vector<HermitianEigResult<T>> run_configurations() {
  std::vector<HermitianEigResult<T>> out;
  // 1-3: explicit operator with spread-out ends, every `which`, automatic ncv.
  std::vector<double> cubic(300);
  for (index_t i = 0; i < 300; ++i) {
    const double t = -1.0 + 2.0 * static_cast<double>(i) / 299.0;
    cubic[i] = 10.0 * t * t * t + t + 0.05;
  }
  hs::test::DenseOperator<T> dense(
      hs::test::to_dense(hs::test::prescribed_spectrum<T>(cubic, 8, 1)));
  for (Which w : {Which::LargestMagnitude, Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
    KrylovSchurOptions<T> opt;
    opt.nev = 8;
    out.push_back(hs::lanczos_krylov_schur<T>(dense, nullptr, w, opt));
  }
  // 4-7: shift-invert pencil in the B inner product, restarts, recompute and store_bv, then a
  // deflated run on the first result in both modes.
  const index_t n = 120;
  const auto bd = hs::test::to_dense(hs::test::random_hpd<T>(n, 0.08, 62));
  hs::test::DenseShiftInvertOperator<T> si(
      hs::test::to_dense(hs::test::random_hermitian<T>(n, 0.08, 61)), &bd, 0.05);
  hs::test::DenseOperator<T> bop(bd);
  KrylovSchurOptions<T> opt;
  opt.nev = 5;
  opt.ncv = 11;
  for (bool store : {false, true}) {
    opt.store_bv = store;
    out.push_back(hs::lanczos_krylov_schur<T>(si, &bop, Which::LargestMagnitude, opt));
  }
  const DenseMatrix<T> defl = out[3].vectors;
  opt.deflation = &defl;
  opt.nev = 4;
  for (bool store : {false, true}) {
    opt.store_bv = store;
    out.push_back(hs::lanczos_krylov_schur<T>(si, &bop, Which::LargestMagnitude, opt));
  }
  // 8: 100+ restarts (1-D Laplacian, clustered top end).
  hs::test::DenseOperator<T> lap(hs::test::to_dense(hs::test::laplacian_1d<T>(400)));
  KrylovSchurOptions<T> many;
  many.nev = 10;
  many.ncv = 24;
  many.tol = 1e-10;
  out.push_back(hs::lanczos_krylov_schur<T>(lap, nullptr, Which::LargestAlgebraic, many));
  // 9: user start vector.
  DenseMatrix<T> x0(300, 1);
  for (index_t i = 0; i < 300; ++i) x0(i, 0) = T(1.0 + static_cast<double>(i % 5));
  KrylovSchurOptions<T> start;
  start.nev = 4;
  start.initial = &x0;
  out.push_back(hs::lanczos_krylov_schur<T>(dense, nullptr, Which::LargestAlgebraic, start));
  // 10: full-space cycle (ncv = n).
  std::vector<double> small;
  for (index_t i = 0; i < 20; ++i) small.push_back(std::pow(-1.0, i) * (1.0 + i));
  hs::test::DenseOperator<T> tiny(
      hs::test::to_dense(hs::test::prescribed_spectrum<T>(small, 8, 81)));
  KrylovSchurOptions<T> full;
  full.nev = 4;
  full.ncv = 20;
  out.push_back(hs::lanczos_krylov_schur<T>(tiny, nullptr, Which::LargestMagnitude, full));
  // 11-12: n > 2 * 4096 (row-blocked rotation of V and M*V), diagonal M, then 70 deflation
  // columns (chunked deflation check).
  const index_t big = 9001, top = 80;
  std::vector<T> av(static_cast<std::size_t>(big)), mv(static_cast<std::size_t>(big));
  for (index_t i = 0; i < big - top; ++i)
    av[i] = T(static_cast<double>(i) / static_cast<double>(big - top));
  for (index_t k = 0; k < top; ++k)
    av[big - top + k] = T(1.0 + std::pow(1.1, static_cast<double>(k + 1)));
  for (index_t i = 0; i < big; ++i)
    mv[i] = T(1.0 + 3.0 * std::fmod(0.6180339887 * static_cast<double>(i), 1.0));
  hs::test::DiagonalOperator<T> aop(av), mop(mv);
  KrylovSchurOptions<T> rows;
  rows.nev = 5;
  rows.ncv = 12;
  rows.store_bv = true;
  out.push_back(hs::lanczos_krylov_schur<T>(aop, &mop, Which::LargestAlgebraic, rows));
  DenseMatrix<T> d70(big, 70);
  for (index_t c = 0; c < 70; ++c)
    d70(big - 1 - c, c) = T(1.0 / std::sqrt(std::real(mv[big - 1 - c])));
  rows.deflation = &d70;
  rows.store_bv = false;
  out.push_back(hs::lanczos_krylov_schur<T>(aop, &mop, Which::LargestAlgebraic, rows));
  return out;
}

template <class T>
class LanczosBitwiseTyped : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LanczosBitwiseTyped, Scalars);

TYPED_TEST(LanczosBitwiseTyped, BlockSizeOneReproducesTheBaseCommitSolver) {
  using T = TypeParam;
  const auto results = run_configurations<T>();
  const std::uint64_t fingerprint = platform_fingerprint<T>();
  if (std::getenv("HS_LANCZOS_GOLDEN_PRINT")) {
    std::printf("[golden] %s counts:", hs::is_complex_v<T> ? "cplx" : "double");
    for (const auto& r : results)
      std::printf(" {%ld, %ld, %ld, %zu},", static_cast<long>(r.op_applies),
                  static_cast<long>(r.restarts), static_cast<long>(r.nconv), r.history.size());
    std::printf("\n[golden] %s hashes: {0x%016llxull, {", hs::is_complex_v<T> ? "cplx" : "double",
                static_cast<unsigned long long>(fingerprint));
    for (const auto& r : results)
      std::printf("0x%016llxull, ", static_cast<unsigned long long>(result_hash(r)));
    std::printf("}},\n");
  }
  const auto counts = Reference<T>::counts();
  ASSERT_EQ(counts.size(), results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    SCOPED_TRACE("configuration " + std::to_string(i + 1));
    EXPECT_EQ(results[i].op_applies, counts[i].op_applies);
    EXPECT_EQ(results[i].restarts, counts[i].restarts);
    EXPECT_EQ(results[i].nconv, counts[i].nconv);
    EXPECT_EQ(static_cast<index_t>(results[i].history.size()), counts[i].history);
  }
  for (const auto& [fp, hashes] : Reference<T>::hashes()) {
    if (fp != fingerprint) continue;
    ASSERT_EQ(hashes.size(), results.size());
    for (std::size_t i = 0; i < results.size(); ++i)
      EXPECT_EQ(result_hash(results[i]), hashes[i]) << "configuration " << i + 1;
    return;
  }
  std::cout << "[ info     ] platform fingerprint 0x" << std::hex << fingerprint << std::dec
            << " has no recorded reference: bitwise check skipped (counts checked)\n";
}

}  // namespace
