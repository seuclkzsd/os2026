// block_size = 1 must keep the single-vector general Krylov-Schur solver of the base commit
// a8ed8f4 bit for bit (module ghiep, P4: the block mode shares the selection, certification and
// output code). This file pins that behaviour, as test_lanczos_bitwise.cpp does for the Hermitian
// solver:
//   * exact integers (operator applications, restarts, nconv, history length, result size) of a
//     set of configurations covering both scalar types, every `which` mode, conjugate pairs at
//     the cut, shift-invert of an indefinite pencil, hundreds of restarts on the minimal basis, a
//     user start vector, a breakdown, a whole-space cycle and the row-blocked basis compression
//     (n > 2048): enforced everywhere;
//   * a 64-bit FNV-1a hash of every output bit except the Ritz vectors (theta, estimate, ritz_all,
//     schur_form, schur_basis, history, nconv, restarts, op_applies, converged): enforced where the
//     platform fingerprint (bits of the kernels the solver uses, on fixed inputs) matches a
//     recorded one, else skipped with a note. The Ritz vectors are excluded on purpose: since the
//     memory change of P4 they are formed as Q1 z (Schur basis times projected eigenvectors) after
//     the Krylov basis is freed, instead of V y, which changes their last bits (by <= 2e-15 for
//     unit vectors in the measurement of that commit); every other output is unchanged.
// Reference generator: compile this file against the base commit's library (e.g. git archive
// a8ed8f4 into a scratch tree, build libhsolve there, compile this file with that tree's include
// and src directories and link it with this tree's libhsolve_test_support.a - the fixtures are
// unchanged - and gtest_main) and run it with HS_KSG_GOLDEN_PRINT=1 under guard.py --threads 1 and
// --threads 2; it prints the tables below. Recorded on the project server (Xeon Gold 6336Y, conda
// env os2026 OpenBLAS, -march=native release build).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "hsolve/eigen.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::DenseMatrix;
using hs::GeneralEigResult;
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

/// Hash of every output except the Ritz vectors (file comment).
template <class T>
std::uint64_t result_hash(const GeneralEigResult<T>& r) {
  Fnv f;
  f.vec(r.theta);
  f.vec(r.estimate);
  f.vec(r.ritz_all);
  f.mat(r.schur_form);
  f.mat(r.schur_basis);
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
/// (linalg::dot, norm2), CGS2, gemm (dense operator, basis products), gees, trsen and trevc.
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
  const index_t n = 120, k = 10, m = 24;
  DenseMatrix<T> a(n, n), x(n, k), y(n, k);
  for (index_t j = 0; j < n; ++j) hs::linalg::fill_random(a.col(j), n, 13, j);
  for (index_t j = 0; j < k; ++j) hs::linalg::fill_random(x.col(j), n, 12, j);
  hs::test::DenseOperator<T> aop(a);
  aop.apply(x.data(), n, y.data(), n, k);
  f.mat(y);
  // Dense LU solves of a full pencil as in configuration 5 (their bits depend on the thread
  // count: measured, OMP_NUM_THREADS 1 against 2).
  const index_t ns = 150;
  const auto ah = hs::test::to_dense(hs::test::random_hermitian<T>(ns, 1.0, 21));
  const auto bh = hs::test::to_dense(hs::test::random_hermitian<T>(ns, 1.0, 22));
  hs::test::DenseShiftInvertOperator<T> si(ah, &bh, 0.137);
  DenseMatrix<T> xs(ns, k), ys(ns, k);
  for (index_t j = 0; j < k; ++j) hs::linalg::fill_random(xs.col(j), ns, 14, j);
  si.apply(xs.data(), ns, ys.data(), ns, k);
  f.mat(ys);
  std::vector<T> w(x.col(k - 1), x.col(k - 1) + n), h(static_cast<std::size_t>(k));
  f.value(hs::linalg::orthogonalize_cgs2<T>(n, k - 1, x.data(), n, nullptr, n, w.data(), nullptr,
                                            h.data()));
  f.vec(w);
  DenseMatrix<T> s(m, m), q(m, m);
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < m; ++i) s(i, j) = a(i, j);
  std::vector<cplx> ev(static_cast<std::size_t>(m));
  const auto fm = static_cast<hs::lapack::fint>(m);
  hs::lapack::gees(fm, s.data(), fm, ev.data(), q.data(), fm);
  f.vec(ev);
  f.mat(s);
  f.mat(q);
  std::vector<hs::lapack::fint> select(static_cast<std::size_t>(m), 0);
  select[static_cast<std::size_t>(m - 1)] = 1;
  if (!hs::is_complex_v<T> && s(m - 1, m - 2) != T(0)) select[static_cast<std::size_t>(m - 2)] = 1;
  hs::lapack::fint info = 0;
  hs::lapack::trsen(fm, s.data(), fm, q.data(), fm, select, ev.data(), &info);
  f.value(info);
  f.mat(s);
  hs::lapack::trevc_backtransform(fm, s.data(), fm, q.data(), fm);
  f.mat(q);
  return f.get();
}

/// One configuration's pinned outcome.
struct Pinned {
  index_t op_applies, restarts, nconv, history, m;
};

template <class T>
struct Reference;

using HashTable = std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>>;

// Generated by the reference generator (file comment) from a8ed8f4. Counts: the same for
// OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 1/1, 2/1 (the ctest setting) and 2/2. Hashes:
// {platform fingerprint, one hash per configuration} for 1 and 2 OpenMP threads (the OpenBLAS
// thread count changed no bit; with 3 or more OpenMP threads the reductions of linalg::dot combine
// partial sums in arrival order, see test_lanczos_bitwise.cpp).
template <>
struct Reference<double> {
  static std::vector<Pinned> counts() {
    return {{78, 7, 9, 8, 9},     {78, 7, 9, 8, 9},   {715, 87, 9, 88, 9}, {56, 13, 4, 14, 4},
            {74, 6, 11, 7, 11},   {87, 78, 5, 79, 5}, {50, 4, 4, 5, 4},    {100, 22, 5, 23, 5},
            {20, 0, 5, 1, 5},     {811, 159, 6, 160, 6}};
  }
  static HashTable hashes() {
    return {{0x543819d6bf4fef43ull,  // 1 OpenMP thread
             {0x7a5a1f9682fc24edull, 0x8d607ba91bf7376cull, 0x55d17b44b4975a44ull,
              0x9bf17c8fe34ffdccull, 0xb2122ae4cb21a904ull, 0x20b251044c17515dull,
              0x929968989cfdac79ull, 0x8a34eaab4b933772ull, 0xe1cf25a2e4731927ull,
              0x2b07ac339c8c813eull}},
            {0xe5ec73e3bc29e284ull,  // 2 OpenMP threads
             {0x7a5a1f9682fc24edull, 0x8d607ba91bf7376cull, 0x55d17b44b4975a44ull,
              0x9bf17c8fe34ffdccull, 0xaeab3818622ebd7bull, 0x20b251044c17515dull,
              0x929968989cfdac79ull, 0x8a34eaab4b933772ull, 0xe1cf25a2e4731927ull,
              0x2b07ac339c8c813eull}}};
  }
};
template <>
struct Reference<cplx> {
  static std::vector<Pinned> counts() {
    return {{113, 11, 9, 12, 9},  {113, 11, 9, 12, 9},   {361, 42, 9, 43, 9}, {71, 15, 4, 16, 4},
            {74, 6, 10, 7, 10},   {107, 100, 5, 101, 5}, {68, 6, 4, 7, 4},    {104, 23, 5, 24, 5},
            {20, 0, 4, 1, 4},     {816, 160, 6, 161, 6}};
  }
  static HashTable hashes() {
    return {{0xb91abbb6b5fc5ebaull,  // 1 OpenMP thread
             {0xc0d1c122c9a49a16ull, 0xc16937381e759a2cull, 0xf5efd51a09ec5c12ull,
              0xdebec9bec91778c6ull, 0x16e2ed4ca2e24c89ull, 0xb2f0a515d37b0255ull,
              0x9bbc07824f1e81dfull, 0x22cbb3d2d8e64e74ull, 0xb14b46c602d50f50ull,
              0x06640797c48173e6ull}},
            {0x95c8c2e3edca9b56ull,  // 2 OpenMP threads
             {0xc0d1c122c9a49a16ull, 0xc16937381e759a2cull, 0xf5efd51a09ec5c12ull,
              0xdebec9bec91778c6ull, 0x07afd6c4e70abf26ull, 0xb2f0a515d37b0255ull,
              0x9bbc07824f1e81dfull, 0x22cbb3d2d8e64e74ull, 0xb14b46c602d50f50ull,
              0x06640797c48173e6ull}}};
  }
};

/// A = X D X^{-1}-like test matrix: random entries scaled by 0.3 / sqrt(n) plus a diagonal
/// 1 / (1 + 0.2 i) (distinct, non-normal); real T gets conjugate pairs from the random part.
template <class T>
DenseMatrix<T> test_matrix(index_t n, std::uint64_t seed) {
  DenseMatrix<T> a(n, n);
  for (index_t j = 0; j < n; ++j) hs::linalg::fill_random(a.col(j), n, seed, j);
  for (index_t j = 0; j < n; ++j)
    for (index_t i = 0; i < n; ++i) a(i, j) *= 0.3 / std::sqrt(static_cast<double>(n));
  for (index_t i = 0; i < n; ++i) a(i, i) += T(1.0 / (1.0 + 0.2 * static_cast<double>(i)));
  return a;
}

/// The configurations, in a fixed order (index = position in the reference tables).
template <class T>
std::vector<GeneralEigResult<T>> run_configurations() {
  std::vector<GeneralEigResult<T>> out;
  // 1-3: non-normal dense operator, every `which`, automatic ncv.
  hs::test::DenseOperator<T> dense(test_matrix<T>(200, 7));
  for (Which w : {Which::LargestMagnitude, Which::LargestAlgebraic, Which::SmallestAlgebraic}) {
    KrylovSchurOptions<T> opt;
    opt.nev = 9;
    out.push_back(hs::krylov_schur_general<T>(dense, w, opt));
  }
  // 4: requested basis, tighter tol.
  KrylovSchurOptions<T> req;
  req.nev = 4;
  req.ncv = 11;
  req.tol = 1e-13;
  out.push_back(hs::krylov_schur_general<T>(dense, Which::LargestMagnitude, req));
  // 5: shift-invert of a pencil with indefinite B (complex eigenvalues).
  const index_t n = 150;
  const auto ah = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 21));
  DenseMatrix<T> bh = hs::test::to_dense(hs::test::random_hermitian<T>(n, 1.0, 22));
  for (index_t j = 0; j < n; ++j) {
    for (index_t i = 0; i < n; ++i) bh(i, j) *= 0.15;
    bh(j, j) += T((j % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.01 * static_cast<double>(j)));
  }
  hs::test::DenseShiftInvertOperator<T> si(ah, &bh, 0.137);
  KrylovSchurOptions<T> sio;
  sio.nev = 10;
  out.push_back(hs::krylov_schur_general<T>(si, Which::LargestMagnitude, sio));
  // 6: minimal basis (ncv = nev + 2), hundreds of restarts.
  KrylovSchurOptions<T> tight;
  tight.nev = 5;
  tight.ncv = 7;
  tight.tol = 1e-11;
  tight.max_restarts = 400;
  out.push_back(hs::krylov_schur_general<T>(dense, Which::LargestMagnitude, tight));
  // 7: user start vector.
  DenseMatrix<T> x0(200, 1);
  for (index_t i = 0; i < 200; ++i) x0(i, 0) = T(1.0 + static_cast<double>(i % 5));
  KrylovSchurOptions<T> start;
  start.nev = 4;
  start.initial = &x0;
  out.push_back(hs::krylov_schur_general<T>(dense, Which::LargestAlgebraic, start));
  // 8: breakdown - a start vector inside a 3-dimensional invariant subspace of a diagonal Op.
  std::vector<T> diag(60);
  for (index_t i = 0; i < 60; ++i) diag[i] = T(1.0 + 0.5 * static_cast<double>(i));
  hs::test::DiagonalOperator<T> dop(diag);
  DenseMatrix<T> inv(60, 1);
  inv(2, 0) = T(1), inv(10, 0) = T(-2), inv(30, 0) = T(0.5);
  KrylovSchurOptions<T> brk;
  brk.nev = 5;
  brk.ncv = 12;
  brk.initial = &inv;
  out.push_back(hs::krylov_schur_general<T>(dop, Which::LargestMagnitude, brk));
  // 9: whole-space cycle (ncv >= n).
  hs::test::DenseOperator<T> tiny(test_matrix<T>(20, 9));
  KrylovSchurOptions<T> whole;
  whole.nev = 4;
  whole.ncv = 20;
  out.push_back(hs::krylov_schur_general<T>(tiny, Which::SmallestAlgebraic, whole));
  // 10: n > 2 * 2048 (row-blocked compression and outputs).
  const index_t big = 5000;
  std::vector<T> dv(static_cast<std::size_t>(big));
  for (index_t i = 0; i < big; ++i)
    dv[i] = T(std::pow(0.999, static_cast<double>(i)) * (i % 2 ? -1.0 : 1.0));
  hs::test::DiagonalOperator<T> bop(dv);
  KrylovSchurOptions<T> rows;
  rows.nev = 6;
  rows.ncv = 16;
  rows.tol = 1e-10;
  out.push_back(hs::krylov_schur_general<T>(bop, Which::LargestMagnitude, rows));
  return out;
}

template <class T>
class KrylovSchurGeneralBitwise : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(KrylovSchurGeneralBitwise, Scalars);

TYPED_TEST(KrylovSchurGeneralBitwise, BlockSizeOneReproducesTheBaseCommitSolver) {
  using T = TypeParam;
  const auto results = run_configurations<T>();
  const std::uint64_t fingerprint = platform_fingerprint<T>();
  if (std::getenv("HS_KSG_GOLDEN_PRINT")) {
    std::printf("[golden] %s counts:", hs::is_complex_v<T> ? "cplx" : "double");
    for (const auto& r : results)
      std::printf(" {%ld, %ld, %ld, %zu, %zu},", static_cast<long>(r.op_applies),
                  static_cast<long>(r.restarts), static_cast<long>(r.nconv), r.history.size(),
                  r.theta.size());
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
    EXPECT_EQ(static_cast<index_t>(results[i].theta.size()), counts[i].m);
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
