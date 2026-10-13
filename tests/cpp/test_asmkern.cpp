// Unit tests of the BLAS hook (src/blas, module blas, P5 "asmkern"; docs/BLAS.md): the route
// configuration parser, the router (dgemm N = 1 / M = 1 -> dgemv, dtrsm N = 1 -> dtrsv, narrow
// kernels, parallel splits) against the real BLAS for every routed shape class - edge sizes,
// transposes, padded leading dimensions, alpha / beta special values, NaN propagation and the
// beta = 0 rule (C not read) -, the narrow kernels and the small triangular solve on every
// instruction set the CPU has (portable, AVX2, AVX-512) against a long-double reference / the real
// dtrsm, bitwise independence of the thread count, the forced fallbacks without AVX-512 / AVX2
// (HS_BLAS_NO_AVX512, HS_BLAS_NO_AVX2), concurrent use from many threads, the lookup of the real
// routines (also in libraries opened with dlopen(RTLD_LOCAL), as under Python), and the
// profiler's helpers.
// The real BLAS is the test executable's (OpenBLAS), found by the hook through dlsym(RTLD_NEXT).
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "blas/blas_hook.hpp"
#include "blas/blas_profile.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

// The hook finds the real BLAS with dlsym(RTLD_NEXT): this reference keeps the BLAS library in
// the executable's dependencies (the toolchain links with --as-needed).
[[maybe_unused, gnu::used]] void (*const kLinkBlas)(const char*, const char*, const hs::lapack::fint*,
                                         const hs::lapack::fint*, const hs::lapack::fint*,
                                         const double*, const double*, const hs::lapack::fint*,
                                         const double*, const hs::lapack::fint*, const double*,
                                         double*, const hs::lapack::fint*, hs::lapack::fstrlen,
                                         hs::lapack::fstrlen) = &hs::lapack::dgemm_;

using hs::blas::fint;
using hs::blas::Isa;
using hs::blas::Route;
using hs::blas::RouteConfig;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// What the CPU reports (the hook's SIMD kernels exist on x86 only).
bool cpu_avx2_fma() {
#if HS_BLAS_X86
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool cpu_avx512f() {
#if HS_BLAS_X86
  return __builtin_cpu_supports("avx512f") != 0;
#else
  return false;
#endif
}

std::vector<double> random_vec(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& x : v) x = u(g);
  return v;
}

bool is_t(char c) {
  return c == 'T' || c == 't' || c == 'C' || c == 'c';
}

// Long-double reference of C = alpha op(A) op(B) + beta C (beta == 0: C not read), and the
// elementwise error bound gamma * (|alpha| |op(A)| |op(B)| + |beta| |C|).
void ref_gemm(char ta, char tb, fint m, fint n, fint k, double alpha, const double* a, fint lda,
              const double* b, fint ldb, double beta, const double* c_in, fint ldc,
              std::vector<double>* c_out, std::vector<double>* bound) {
  c_out->assign(c_in, c_in + static_cast<std::size_t>(ldc) * n);
  bound->assign(c_out->size(), 0.0);
  for (fint j = 0; j < n; ++j)
    for (fint i = 0; i < m; ++i) {
      long double s = 0.0L, sa = 0.0L;
      for (fint p = 0; p < k; ++p) {
        const double av = is_t(ta) ? a[p + static_cast<std::size_t>(i) * lda]
                                   : a[i + static_cast<std::size_t>(p) * lda];
        const double bv = is_t(tb) ? b[j + static_cast<std::size_t>(p) * ldb]
                                   : b[p + static_cast<std::size_t>(j) * ldb];
        s += static_cast<long double>(av) * bv;
        sa += std::abs(static_cast<long double>(av) * bv);
      }
      const std::size_t idx = i + static_cast<std::size_t>(j) * ldc;
      const double cv = c_in[idx];
      const long double r = alpha * s + (beta == 0.0 ? 0.0L : static_cast<long double>(beta) * cv);
      (*c_out)[idx] = static_cast<double>(r);
      (*bound)[idx] = 4.0 * (k + 2) * std::numeric_limits<double>::epsilon() *
                      static_cast<double>(std::abs(alpha) * sa +
                                          (beta == 0.0 ? 0.0 : std::abs(beta * cv)));
    }
}

void expect_close(const std::vector<double>& got, const std::vector<double>& want,
                  const std::vector<double>& bound, const std::string& what) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(want[i])) {
      ASSERT_TRUE(std::isnan(got[i])) << what << " at " << i;
      continue;
    }
    ASSERT_LE(std::abs(got[i] - want[i]), bound[i] + 1e-300) << what << " at " << i;
  }
}

// Every instruction set this CPU can run (the kernels are tested on each).
std::vector<Isa> isas() {
  std::vector<Isa> v{Isa::Portable};
  if (hs::blas::cpu_has_avx2_fma()) v.push_back(Isa::Avx2);
  if (hs::blas::cpu_has_avx512f()) v.push_back(Isa::Avx512);
  return v;
}

// Calls the real dgemm (the reference the routes are compared with in the router tests).
void real_dgemm(char ta, char tb, fint m, fint n, fint k, double alpha, const double* a, fint lda,
                const double* b, fint ldb, double beta, double* c, fint ldc) {
  hs::blas::real_blas().dgemm(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1,
                              1);
}

class EnvGuard {
 public:
  EnvGuard(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    had_ = old != nullptr;
    if (had_) old_ = old;
    setenv(name, value, 1);
  }
  ~EnvGuard() {
    if (had_)
      setenv(name_.c_str(), old_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

 private:
  std::string name_, old_;
  bool had_ = false;
};

// ------------------------------------------------------------------------------ configuration

TEST(BlasHookConfig, ParsesTokensAndReportsUnknownOnes) {
  std::string unknown;
  RouteConfig c = hs::blas::parse_route_config("gemv1, trsv1,maxn=8,minkb=64,threads=3", &unknown);
  EXPECT_TRUE(unknown.empty()) << unknown;
  EXPECT_TRUE(c.gemm_n1_gemv);
  EXPECT_EQ(c.trsv_max_n, 1);
  EXPECT_FALSE(c.narrow);
  EXPECT_FALSE(c.parallel);
  EXPECT_EQ(c.narrow_max_n, 8);
  EXPECT_EQ(c.parallel_min_bytes, 64 * 1024);
  EXPECT_EQ(c.threads, 3);

  c = hs::blas::parse_route_config("all", &unknown);
  EXPECT_TRUE(c.gemm_n1_gemv && c.narrow && c.parallel);
  EXPECT_EQ(c.trsv_max_n, 0);
  EXPECT_EQ(c.small_trsm_max_m, 512);
  c = hs::blas::parse_route_config("narrow,rows=128", &unknown);
  EXPECT_EQ(c.nn_split_rows, 128);
  EXPECT_TRUE(unknown.empty()) << unknown;
  c = hs::blas::parse_route_config("all,off", &unknown);
  EXPECT_FALSE(c.gemm_n1_gemv || c.trsv_max_n > 0 || c.narrow || c.parallel);

  c = hs::blas::parse_route_config("gemv1,bogus,maxn=0,maxn=65,threads=x,rows=7,par", &unknown);
  EXPECT_EQ(unknown, "bogus,maxn=0,maxn=65,threads=x,rows=7");
  EXPECT_TRUE(c.gemm_n1_gemv);
  EXPECT_TRUE(c.parallel);
  EXPECT_EQ(c.narrow_max_n, 16);

  unknown.clear();
  c = hs::blas::parse_route_config("", &unknown);
  EXPECT_TRUE(unknown.empty());
  EXPECT_FALSE(c.gemm_n1_gemv || c.trsv_max_n > 0 || c.narrow || c.parallel);
}

TEST(BlasHookConfig, BuildDefaultParses) {
  std::string unknown;
  hs::blas::parse_route_config(hs::blas::default_route_spec(), &unknown);
  EXPECT_TRUE(unknown.empty()) << "HSOLVE_BLAS_ROUTE_DEFAULT has unknown tokens: " << unknown;
}

TEST(BlasHookConfig, ToStringRoundTrips) {
  for (const char* spec : {"off", "gemv1", "all,maxn=4,minkb=16,threads=2", "narrow,portable",
                           "trsv1,par,minkb=0", "trsv=7"}) {
    const RouteConfig c = hs::blas::parse_route_config(spec);
    std::string unknown;
    const RouteConfig d = hs::blas::parse_route_config(hs::blas::to_string(c), &unknown);
    EXPECT_TRUE(unknown.empty()) << spec;
    EXPECT_EQ(hs::blas::to_string(c), hs::blas::to_string(d)) << spec;
    EXPECT_EQ(c.isa, d.isa);
  }
}

// HS_BLAS_NO_AVX512 / HS_BLAS_NO_AVX2 force the lower instruction sets (as on a CPU without
// them), and the route tokens cap the ISA.
TEST(BlasHookConfig, ForcedFallbackWithoutAvx512) {
  const fint m = 300, n = 4, k = 70;
  const auto a = random_vec(static_cast<std::size_t>(m) * k, 1);
  const auto b = random_vec(static_cast<std::size_t>(k) * n, 2);
  const auto c0 = random_vec(static_cast<std::size_t>(m) * n, 3);
  const bool has_avx2 = cpu_avx2_fma();
  {
    EnvGuard g("HS_BLAS_NO_AVX512", "1");
    EXPECT_FALSE(hs::blas::cpu_has_avx512f());
    const RouteConfig c = hs::blas::parse_route_config("all,minkb=1000000");
    EXPECT_EQ(c.isa, has_avx2 ? Isa::Avx2 : Isa::Portable);
    EXPECT_EQ(hs::blas::best_isa(), c.isa);
    std::vector<double> got = c0, want, bound;
    ref_gemm('N', 'N', m, n, k, 1.5, a.data(), m, b.data(), k, -1.0, c0.data(), m, &want, &bound);
    EXPECT_EQ(hs::blas::dgemm(c, 'N', 'N', m, n, k, 1.5, a.data(), m, b.data(), k, -1.0,
                              got.data(), m),
              has_avx2 ? Route::Narrow : Route::Real);
    expect_close(got, want, bound, "avx2 fallback");
  }
  EXPECT_EQ(hs::blas::parse_route_config("all,avx2").isa,
            std::min(Isa::Avx2, hs::blas::best_isa()));
  EXPECT_EQ(hs::blas::parse_route_config("all,avx2,portable").isa, Isa::Portable);
  {
    EnvGuard g1("HS_BLAS_NO_AVX512", "1");
    EnvGuard g2("HS_BLAS_NO_AVX2", "yes");
    EXPECT_FALSE(hs::blas::cpu_has_avx2_fma());
    const RouteConfig c = hs::blas::parse_route_config("all,minkb=1000000");
    EXPECT_EQ(c.isa, Isa::Portable);
    // The narrow route needs SIMD kernels: without them the call goes to the real dgemm and the
    // result is the real routine's.
    std::vector<double> got = c0, want = c0;
    EXPECT_EQ(hs::blas::dgemm(c, 'N', 'N', m, n, k, 1.5, a.data(), m, b.data(), k, -1.0,
                              got.data(), m),
              Route::Real);
    real_dgemm('N', 'N', m, n, k, 1.5, a.data(), m, b.data(), k, -1.0, want.data(), m);
    EXPECT_EQ(got, want);
  }
  {
    EnvGuard g("HS_BLAS_NO_AVX512", "0");
    EXPECT_EQ(hs::blas::cpu_has_avx512f(), cpu_avx512f());
    EXPECT_EQ(hs::blas::cpu_has_avx2_fma(), has_avx2);
  }
}

// ---------------------------------------------------------------------------------- kernels

TEST(BlasHookKernels, NarrowGemmMatchesLongDoubleReference) {
  const fint ms[] = {1, 7, 8, 9, 31, 33, 517, 1100};
  const fint ns[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64};
  const fint ks[] = {1, 3, 4, 5, 9, 64, 131};
  const double alphas[] = {1.0, -0.75};
  const double betas[] = {0.0, 1.0, -0.5};
  std::uint64_t seed = 10;
  for (Isa isa : isas())
    for (char ta : {'N', 'T'})
      for (fint m : ms)
        for (fint n : ns)
          for (fint k : ks) {
            if (static_cast<long>(m) * n * k > 1'000'000) continue;
            const fint lda = (ta == 'N' ? m : k) + 3;
            const fint ldb = k + 1;
            const fint ldc = m + 2;
            const auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), ++seed);
            const auto b = random_vec(static_cast<std::size_t>(ldb) * n, ++seed);
            const auto c0 = random_vec(static_cast<std::size_t>(ldc) * n, ++seed);
            for (double alpha : alphas)
              for (double beta : betas) {
                std::vector<double> c = c0, want, bound;
                if (beta == 0.0) std::fill(c.begin(), c.end(), kNaN);  // must not be read
                ref_gemm(ta, 'N', m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(),
                         ldc, &want, &bound);
                // padding rows of C (i >= m) must stay untouched
                for (fint j = 0; j < n; ++j)
                  for (fint i = m; i < ldc; ++i) want[i + static_cast<std::size_t>(j) * ldc] =
                      c[i + static_cast<std::size_t>(j) * ldc];
                if (ta == 'N')
                  hs::blas::narrow_gemm_nn(isa, 1, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                                           beta, c.data(), ldc);
                else
                  hs::blas::narrow_gemm_tn(isa, 1, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                                           beta, c.data(), ldc);
                for (fint j = 0; j < n; ++j)
                  for (fint i = m; i < ldc; ++i) {
                    const std::size_t idx = i + static_cast<std::size_t>(j) * ldc;
                    if (beta == 0.0) {
                      ASSERT_TRUE(std::isnan(c[idx]));
                      c[idx] = want[idx] = 0.0;
                    }
                  }
                expect_close(c, want, bound,
                             std::string("isa ") + hs::blas::isa_name(isa) +
                                 " " + ta + " m=" + std::to_string(m) + " n=" + std::to_string(n) +
                                 " k=" + std::to_string(k) + " alpha=" + std::to_string(alpha) +
                                 " beta=" + std::to_string(beta));
              }
          }
}

// Row splits without a K split (k < 512) give every element the same arithmetic: the result does
// not depend on the thread count. min_rows = 8 / 4 makes small row blocks: with m = 61 the
// per-thread blocks of NN take the register path, the whole (one thread) the chunked one.
TEST(BlasHookKernels, NarrowGemmIsBitwiseIndependentOfThreads) {
  for (const fint m : {2049, 61})
  for (const fint n : {6, 3})
  for (const fint k : {300, 5})
  for (Isa isa : isas())
    for (char ta : {'N', 'T'}) {
      const fint lda = ta == 'N' ? m : k;
      const auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), 5);
      const auto b = random_vec(static_cast<std::size_t>(k) * n, 6);
      const auto c0 = random_vec(static_cast<std::size_t>(m) * n, 7);
      std::vector<double> ref;
      for (int threads : {1, 2, 3, 4, 7}) {
        std::vector<double> c = c0;
        if (ta == 'N')
          hs::blas::narrow_gemm_nn(isa, threads, m, n, k, -1.0, a.data(), lda, b.data(), k, 1.0,
                                   c.data(), m, 8);
        else
          hs::blas::narrow_gemm_tn(isa, threads, m, n, k, -1.0, a.data(), lda, b.data(), k, 1.0,
                                   c.data(), m, 4);
        if (ref.empty())
          ref = c;
        else
          EXPECT_EQ(c, ref) << hs::blas::isa_name(isa) << " " << ta << " m=" << m
                            << " threads " << threads;
      }
    }
}

// Parallel split with K chunks (few rows of C, long K: MUMPS' backward-solve updates above its L0
// layer, hsolve's Gram products): accurate against the long-double reference, padding rows and
// (beta = 0) C untouched / unread, and bitwise reproducible for a fixed thread count.
TEST(BlasHookKernels, NarrowGemmSplitKIsAccurateAndDeterministic) {
  std::uint64_t seed = 300;
  for (char ta : {'N', 'T'})
    for (const fint m : {3, 40, 300, 700})
      for (const fint n : {2, 4, 9})
        for (const fint k : {1100, 3001}) {
          const fint lda = (ta == 'N' ? m : k) + 5;
          const fint ldb = k + 3;
          const fint ldc = m + 2;
          const auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), ++seed);
          const auto b = random_vec(static_cast<std::size_t>(ldb) * n, ++seed);
          const auto c0 = random_vec(static_cast<std::size_t>(ldc) * n, ++seed);
          for (double beta : {0.0, -0.5}) {
            std::vector<double> cin = c0, want, bound;
            if (beta == 0.0) std::fill(cin.begin(), cin.end(), kNaN);  // must not be read
            ref_gemm(ta, 'N', m, n, k, 0.75, a.data(), lda, b.data(), ldb, beta, cin.data(), ldc,
                     &want, &bound);
            for (fint j = 0; j < n; ++j)
              for (fint i = m; i < ldc; ++i) want[i + static_cast<std::size_t>(j) * ldc] = 0.0;
            for (Isa isa : isas())
              for (const int threads : {2, 7, 16})
                for (const fint rows : {fint{8}, hs::blas::kNnSplitRows}) {
                  auto run = [&] {
                    std::vector<double> c = cin;
                    if (ta == 'N')
                      hs::blas::narrow_gemm_nn(isa, threads, m, n, k, 0.75, a.data(), lda, b.data(),
                                               ldb, beta, c.data(), ldc, rows);
                    else
                      hs::blas::narrow_gemm_tn(isa, threads, m, n, k, 0.75, a.data(), lda, b.data(),
                                               ldb, beta, c.data(), ldc, rows);
                    return c;
                  };
                  std::vector<double> c = run();
                  const std::vector<double> again = run();
                  const std::string what =
                      std::string(hs::blas::isa_name(isa)) + " " + ta +
                      " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" +
                      std::to_string(k) + " threads=" + std::to_string(threads) +
                      " rows=" + std::to_string(rows) + " beta=" + std::to_string(beta);
                  for (std::size_t i = 0; i < c.size(); ++i)
                    ASSERT_TRUE(c[i] == again[i] || (std::isnan(c[i]) && std::isnan(again[i])))
                        << what << ": not reproducible at " << i;
                  for (fint j = 0; j < n; ++j)
                    for (fint i = m; i < ldc; ++i) {  // padding rows untouched
                      const std::size_t q = i + static_cast<std::size_t>(j) * ldc;
                      if (beta == 0.0)
                        ASSERT_TRUE(std::isnan(c[q])) << what;
                      else
                        ASSERT_EQ(c[q], c0[q]) << what;
                      c[q] = 0.0;
                    }
                  expect_close(c, want, bound, what);
                }
          }
        }
}

TEST(BlasHookKernels, NarrowGemmPropagatesNaN) {
  const fint m = 40, n = 3, k = 17;
  for (Isa isa : isas())
    for (char ta : {'N', 'T'}) {
      const fint lda = ta == 'N' ? m : k;
      auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), 8);
      const auto b = random_vec(static_cast<std::size_t>(k) * n, 9);
      // NaN in op(A)(5, 2): row 5 of C becomes NaN in every column, other rows stay finite.
      if (ta == 'N')
        a[5 + 2 * static_cast<std::size_t>(lda)] = kNaN;
      else
        a[2 + 5 * static_cast<std::size_t>(lda)] = kNaN;
      std::vector<double> c(static_cast<std::size_t>(m) * n, 0.0);
      if (ta == 'N')
        hs::blas::narrow_gemm_nn(isa, 1, m, n, k, 1.0, a.data(), lda, b.data(), k, 0.0, c.data(),
                                 m);
      else
        hs::blas::narrow_gemm_tn(isa, 1, m, n, k, 1.0, a.data(), lda, b.data(), k, 0.0, c.data(),
                                 m);
      for (fint j = 0; j < n; ++j)
        for (fint i = 0; i < m; ++i)
          EXPECT_EQ(std::isnan(c[i + static_cast<std::size_t>(j) * m]), i == 5)
              << ta << " " << i << "," << j;
    }
}

// ----------------------------------------------------------------------------------- router

struct GemmCase {
  char ta, tb;
  fint m, n, k;
};

// Every routed dgemm shape class and its neighbours, against the real dgemm.
TEST(BlasHookRouter, DgemmRoutesMatchRealBlas) {
  std::vector<GemmCase> cases;
  for (char ta : {'N', 'T', 'c'})
    for (char tb : {'N', 't'})
      for (fint m : {1, 2, 9, 130, 700})
        for (fint n : {1, 2, 4, 5, 16, 17})
          for (fint k : {1, 7, 64}) cases.push_back({ta, tb, m, n, k});
  const char* specs[] = {"off", "gemv1", "narrow", "par,minkb=0,threads=3",
                         "all,minkb=0,threads=4", "all,maxn=4,minkb=1,threads=2", "all,portable"};
  std::uint64_t seed = 100;
  for (const char* spec : specs) {
    const RouteConfig cfg = hs::blas::parse_route_config(spec);
    for (const GemmCase& g : cases) {
      const fint lda = (is_t(g.ta) ? g.k : g.m) + 1;
      const fint ldb = (is_t(g.tb) ? g.n : g.k) + 2;
      const fint ldc = g.m + 1;
      const auto a = random_vec(static_cast<std::size_t>(lda) * (is_t(g.ta) ? g.m : g.k), ++seed);
      const auto b = random_vec(static_cast<std::size_t>(ldb) * (is_t(g.tb) ? g.k : g.n), ++seed);
      const auto c0 = random_vec(static_cast<std::size_t>(ldc) * g.n, ++seed);
      for (double beta : {0.0, 1.0, 0.25}) {
        std::vector<double> got = c0, want, bound;
        if (beta == 0.0) std::fill(got.begin(), got.end(), kNaN);
        ref_gemm(g.ta, g.tb, g.m, g.n, g.k, -1.0, a.data(), lda, b.data(), ldb, beta, got.data(),
                 ldc, &want, &bound);
        const Route r = hs::blas::dgemm(cfg, g.ta, g.tb, g.m, g.n, g.k, -1.0, a.data(), lda,
                                        b.data(), ldb, beta, got.data(), ldc);
        // Which route: N = 1 / M = 1 -> gemv (or its parallel split), narrow only for op(B) = B.
        if (!cfg.gemm_n1_gemv && !cfg.narrow && !cfg.parallel) {
          EXPECT_EQ(r, Route::Real);
        }
        if (cfg.gemm_n1_gemv && g.n == 1 && !cfg.parallel) {
          EXPECT_EQ(r, Route::Gemv) << spec;
        }
        if (r == Route::Narrow) {
          EXPECT_FALSE(is_t(g.tb));
          EXPECT_TRUE(g.n >= 2 && g.n <= cfg.narrow_max_n);
        }
        for (fint j = 0; j < g.n; ++j)
          for (fint i = g.m; i < ldc; ++i) {
            const std::size_t idx = i + static_cast<std::size_t>(j) * ldc;
            if (beta == 0.0) {
              ASSERT_TRUE(std::isnan(got[idx]));  // padding untouched
            }
            got[idx] = want[idx] = 0.0;
          }
        expect_close(got, want, bound,
                     std::string(spec) + " " + g.ta + g.tb + " m=" + std::to_string(g.m) +
                         " n=" + std::to_string(g.n) + " k=" + std::to_string(g.k) +
                         " beta=" + std::to_string(beta) + " route " + hs::blas::route_name(r));
      }
    }
  }
}

// alpha = 0 and K = 0 are quick-return cases of the reference BLAS (C = beta C, A and B not
// read; OpenBLAS' small-matrix path does read them when alpha = 0: 0 * NaN = NaN). They are never
// routed, so the hook reproduces whatever the real routine does, bit for bit.
TEST(BlasHookRouter, DgemmQuickReturnCasesGoToTheRealRoutine) {
  const RouteConfig cfg = hs::blas::parse_route_config("all,minkb=0,threads=2");
  const fint m = 50, k = 20;
  auto same = [](const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
      if (!(x[i] == y[i] || (std::isnan(x[i]) && std::isnan(y[i])))) return false;
    return true;
  };
  for (fint n : {1, 4})
    for (bool nan_operands : {false, true}) {
      std::vector<double> a = random_vec(static_cast<std::size_t>(m) * k, 1);
      std::vector<double> b = random_vec(static_cast<std::size_t>(k) * n, 2);
      if (nan_operands) {
        std::fill(a.begin(), a.end(), kNaN);
        std::fill(b.begin(), b.end(), kNaN);
      }
      const auto c0 = random_vec(static_cast<std::size_t>(m) * n, 3);
      std::vector<double> got = c0, want = c0;
      EXPECT_EQ(hs::blas::dgemm(cfg, 'N', 'N', m, n, k, 0.0, a.data(), m, b.data(), k, 0.5,
                                got.data(), m),
                Route::Real);
      real_dgemm('N', 'N', m, n, k, 0.0, a.data(), m, b.data(), k, 0.5, want.data(), m);
      EXPECT_TRUE(same(got, want)) << "alpha = 0, n = " << n << " nan " << nan_operands;
      got = want = c0;
      EXPECT_EQ(hs::blas::dgemm(cfg, 'N', 'N', m, n, 0, 1.0, a.data(), m, b.data(), 1, -2.0,
                                got.data(), m),
                Route::Real);
      real_dgemm('N', 'N', m, n, 0, 1.0, a.data(), m, b.data(), 1, -2.0, want.data(), m);
      EXPECT_TRUE(same(got, want)) << "k = 0, n = " << n;
      for (std::size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i], -2.0 * c0[i]);
    }
}

TEST(BlasHookRouter, GemvRouteAndParallelSplitPropagateNaN) {
  const RouteConfig cfg = hs::blas::parse_route_config("gemv1,par,minkb=0,threads=3");
  const fint m = 333, k = 41;
  auto a = random_vec(static_cast<std::size_t>(m) * k, 3);
  auto b = random_vec(k, 4);
  b[7] = kNaN;
  std::vector<double> c(m, 0.0);
  const Route r = hs::blas::dgemm(cfg, 'N', 'N', m, 1, k, 1.0, a.data(), m, b.data(), k, 0.0,
                                  c.data(), m);
  EXPECT_EQ(r, Route::Parallel);
  for (double x : c) EXPECT_TRUE(std::isnan(x));
  b[7] = 0.5;
  a[10 + 3 * static_cast<std::size_t>(m)] = kNaN;
  std::fill(c.begin(), c.end(), 0.0);
  hs::blas::dgemm(cfg, 'N', 'N', m, 1, k, 1.0, a.data(), m, b.data(), k, 0.0, c.data(), m);
  for (fint i = 0; i < m; ++i) EXPECT_EQ(std::isnan(c[i]), i == 10) << i;
}

// The small triangular solve against the real dtrsm: every uplo / trans / diag, edge orders,
// padded leading dimensions; a unit diagonal is never read (NaN there), NaN in B propagates.
TEST(BlasHookKernels, SmallTrsmMatchesRealDtrsm) {
  std::uint64_t seed = 500;
  for (Isa isa : isas())
    for (char uplo : {'U', 'L', 'u'})
      for (char tr : {'N', 'T', 'C'})
        for (char diag : {'N', 'U'})
          for (fint m : {1, 2, 7, 8, 9, 17, 64, 100})
            for (fint n : {1, 2, 4, 5}) {
              const fint lda = m + 3, ldb = m + 1;
              auto a = random_vec(static_cast<std::size_t>(lda) * m, ++seed);
              for (fint j = 0; j < m; ++j)
                for (fint i = 0; i < m; ++i) {
                  double& v = a[i + static_cast<std::size_t>(j) * lda];
                  v = i == j ? (diag == 'U' ? kNaN : 2.0 + 0.5 * v) : v / m;
                }
              const auto b0 = random_vec(static_cast<std::size_t>(ldb) * n, ++seed);
              std::vector<double> got = b0, want = b0;
              hs::blas::small_trsm_left(isa, uplo, tr, diag, m, n, a.data(), lda, got.data(), ldb);
              const char s = 'L';
              const double one = 1.0;
              hs::blas::real_blas().dtrsm(&s, &uplo, &tr, &diag, &m, &n, &one, a.data(), &lda,
                                          want.data(), &ldb, 1, 1, 1, 1);
              for (std::size_t i = 0; i < got.size(); ++i)
                ASSERT_NEAR(got[i], want[i], 1e-13 * (1.0 + std::abs(want[i])))
                    << hs::blas::isa_name(isa) << " " << uplo << tr << diag
                    << " m=" << m << " n=" << n << " at " << i;
            }
  // NaN in B: the solved entries that depend on it become NaN, the others stay finite.
  for (Isa isa : isas()) {
    const fint m = 20;
    std::vector<double> a(static_cast<std::size_t>(m) * m, 0.0);
    for (fint i = 0; i < m; ++i) a[i + static_cast<std::size_t>(i) * m] = 1.0;
    for (fint i = 1; i < m; ++i) a[i + static_cast<std::size_t>(i - 1) * m] = 0.5;  // lower band
    std::vector<double> b(m, 1.0);
    b[5] = kNaN;
    hs::blas::small_trsm_left(isa, 'L', 'N', 'N', m, 1, a.data(), m, b.data(), m);
    for (fint i = 0; i < m; ++i) EXPECT_EQ(std::isnan(b[i]), i >= 5) << i;
  }
}

TEST(BlasHookRouter, SmallTrsmLimitDependsOnTheInstructionSet) {
  const RouteConfig p = hs::blas::parse_route_config("strsm=512,portable");
  EXPECT_EQ(hs::blas::small_trsm_limit(p), 32);
  EXPECT_EQ(hs::blas::small_trsm_limit(hs::blas::parse_route_config("strsm=16,portable")), 16);
  const RouteConfig a2 = hs::blas::parse_route_config("strsm=512,avx2");
  EXPECT_EQ(hs::blas::small_trsm_limit(a2), a2.isa == Isa::Avx2 ? 256 : 32);
  const RouteConfig best = hs::blas::parse_route_config("strsm=512");
  if (best.isa == Isa::Avx512) {
    EXPECT_EQ(hs::blas::small_trsm_limit(best), 512);
  }
  // The cap applies to routing: m = 40 goes to the real dtrsm on the portable kernels.
  std::vector<double> a(40 * 40, 0.0), b(40 * 2, 1.0);
  for (int i = 0; i < 40; ++i) a[i * 41] = 2.0;
  EXPECT_EQ(hs::blas::dtrsm(p, 'L', 'U', 'N', 'N', 40, 2, 1.0, a.data(), 40, b.data(), 40),
            Route::Real);
  EXPECT_EQ(hs::blas::dtrsm(p, 'L', 'U', 'N', 'N', 32, 2, 1.0, a.data(), 40, b.data(), 40),
            Route::SmallTrsm);
}

TEST(BlasHookRouter, DtrsmSmallRoute) {
  const RouteConfig cfg = hs::blas::parse_route_config("strsm=32,trsv=4,maxn=8");
  const fint lda = 70;
  auto a = random_vec(static_cast<std::size_t>(lda) * 64, 600);
  for (fint i = 0; i < 64; ++i) a[i + static_cast<std::size_t>(i) * lda] += 4.0;
  for (fint m : {8, 32, 33})
    for (fint n : {1, 4, 8, 9})
      for (double alpha : {1.0, 2.0}) {
        auto b = random_vec(static_cast<std::size_t>(m) * n, 601);
        const Route r =
            hs::blas::dtrsm(cfg, 'L', 'U', 'T', 'U', m, n, alpha, a.data(), lda, b.data(), m);
        const Route want = alpha != 1.0                ? Route::Real
                           : (m <= 32 && n <= 8)       ? Route::SmallTrsm
                           : n <= 4                    ? Route::Trsv
                                                       : Route::Real;
        EXPECT_EQ(r, want) << m << " " << n << " " << alpha;
      }
}

TEST(BlasHookRouter, DtrsmNarrowRoutesToTrsvAndMatchesReal) {
  const RouteConfig on = hs::blas::parse_route_config("trsv=2");
  const RouteConfig off = hs::blas::parse_route_config("off");
  std::uint64_t seed = 300;
  for (char side : {'L', 'R'})
    for (char uplo : {'U', 'L'})
      for (char tr : {'N', 'T'})
        for (char diag : {'N', 'U'})
          for (fint m : {1, 2, 17, 300})
            for (fint n : {1, 2, 3})
              for (double alpha : {1.0, -2.0}) {
                const fint na = side == 'L' ? m : n;
                const fint lda = na + 2, ldb = m + 1;
                auto a = random_vec(static_cast<std::size_t>(lda) * na, ++seed);
                for (fint i = 0; i < na; ++i) a[i + static_cast<std::size_t>(i) * lda] += 3.0;
                for (fint j = 0; j < na; ++j)
                  for (fint i = 0; i < na; ++i)
                    if (i != j) a[i + static_cast<std::size_t>(j) * lda] /= na;
                const auto b0 = random_vec(static_cast<std::size_t>(ldb) * n, ++seed);
                std::vector<double> got = b0, want = b0;
                const Route r = hs::blas::dtrsm(on, side, uplo, tr, diag, m, n, alpha, a.data(),
                                                lda, got.data(), ldb);
                EXPECT_EQ(hs::blas::dtrsm(off, side, uplo, tr, diag, m, n, alpha, a.data(), lda,
                                          want.data(), ldb),
                          Route::Real);
                EXPECT_EQ(r, (side == 'L' && n <= 2 && alpha == 1.0) ? Route::Trsv : Route::Real);
                for (std::size_t i = 0; i < got.size(); ++i)
                  ASSERT_NEAR(got[i], want[i], 1e-13 * (1.0 + std::abs(want[i])))
                      << side << uplo << tr << diag << " m=" << m << " n=" << n;
              }
}

TEST(BlasHookRouter, DgemvParallelSplitMatchesRealAndIsDeterministic) {
  const RouteConfig par = hs::blas::parse_route_config("par,minkb=0,threads=4");
  const RouteConfig off = hs::blas::parse_route_config("off");
  for (char t : {'N', 'T'})
    for (fint m : {1, 9, 1000})
      for (fint n : {1, 13, 700})
        for (fint inc : {1, 2}) {
          const fint lda = m + 3;
          const auto a = random_vec(static_cast<std::size_t>(lda) * n, 11);
          const fint lx = t == 'N' ? n : m;
          const fint ly = t == 'N' ? m : n;
          const auto x = random_vec(static_cast<std::size_t>(lx) * inc, 12);
          const auto y0 = random_vec(static_cast<std::size_t>(ly) * inc, 13);
          std::vector<double> got = y0, again = y0, want = y0;
          hs::blas::dgemv(par, t, m, n, 0.5, a.data(), lda, x.data(), inc, -1.0, got.data(), inc);
          hs::blas::dgemv(par, t, m, n, 0.5, a.data(), lda, x.data(), inc, -1.0, again.data(),
                          inc);
          hs::blas::dgemv(off, t, m, n, 0.5, a.data(), lda, x.data(), inc, -1.0, want.data(),
                          inc);
          EXPECT_EQ(got, again);
          for (std::size_t i = 0; i < got.size(); ++i)
            ASSERT_NEAR(got[i], want[i], 1e-12 * (1.0 + std::abs(want[i])))
                << t << " m=" << m << " n=" << n << " inc=" << inc;
        }
}

// A^T B with few rows of C and a long K (hsolve's Gram products V^T W): split over K.
TEST(BlasHookRouter, SplitKMatchesRealAndIsDeterministic) {
  const RouteConfig cfg = hs::blas::parse_route_config("par,minkb=0,threads=4");
  for (fint m : {1, 3, 15, 16})
    for (fint n : {1, 2, 5})
      for (double beta : {0.0, -1.0}) {
        const fint k = 40001, lda = k + 3, ldb = k + 1;
        const auto a = random_vec(static_cast<std::size_t>(lda) * m, 31);
        const auto b = random_vec(static_cast<std::size_t>(ldb) * n, 32);
        const auto c0 = random_vec(static_cast<std::size_t>(m) * n, 33);
        std::vector<double> got = c0, again = c0, want, bound;
        if (beta == 0.0) {
          std::fill(got.begin(), got.end(), kNaN);
          std::fill(again.begin(), again.end(), kNaN);
        }
        ref_gemm('T', 'N', m, n, k, 0.5, a.data(), lda, b.data(), ldb, beta, got.data(), m, &want,
                 &bound);
        const Route r = hs::blas::dgemm(cfg, 'T', 'N', m, n, k, 0.5, a.data(), lda, b.data(), ldb,
                                        beta, got.data(), m);
        hs::blas::dgemm(cfg, 'T', 'N', m, n, k, 0.5, a.data(), lda, b.data(), ldb, beta,
                        again.data(), m);
        if (m < 16) {
          EXPECT_EQ(r, Route::SplitK) << m << " " << n;
        } else {
          EXPECT_NE(r, Route::SplitK) << m << " " << n;
        }
        EXPECT_EQ(got, again);
        expect_close(got, want, bound, "splitk m=" + std::to_string(m) + " n=" + std::to_string(n));
      }
}

TEST(BlasHookRouter, ParallelThreadRule) {
  RouteConfig c = hs::blas::parse_route_config("par,minkb=64,threads=8");
  EXPECT_EQ(hs::blas::parallel_threads(c, 64 * 1024.0), 1);       // one block of work
  EXPECT_EQ(hs::blas::parallel_threads(c, 3 * 64 * 1024.0), 3);   // >= minkb per thread
  EXPECT_EQ(hs::blas::parallel_threads(c, 3.9 * 64 * 1024.0), 3);
  EXPECT_EQ(hs::blas::parallel_threads(c, 1e12), 8);              // capped by threads
  int inside = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    inside = hs::blas::parallel_threads(c, 1e12);  // no nested parallelism
  }
  EXPECT_EQ(inside, 1);
}

// Without the parallel route a large call outside parallel regions stays with the real dgemm
// (which may thread it); the same call inside a parallel region, or a small one, takes the
// serial narrow kernel; with the parallel route the large call is split.
TEST(BlasHookRouter, NarrowRouteLeavesLargeSerialCallsToTheRealBlas) {
  if (hs::blas::best_isa() == Isa::Portable) GTEST_SKIP() << "narrow kernels need AVX2 or AVX-512";
  const RouteConfig narrow = hs::blas::parse_route_config("narrow,minkb=64");
  const RouteConfig both = hs::blas::parse_route_config("narrow,par,minkb=64,threads=4");
  const fint m = 1000, n = 4, k = 40;  // A: 320 KB >= 2 * 64 KB
  const auto a = random_vec(static_cast<std::size_t>(m) * k, 21);
  const auto b = random_vec(static_cast<std::size_t>(k) * n, 22);
  std::vector<double> c(static_cast<std::size_t>(m) * n, 0.0);
  auto call = [&](const RouteConfig& cfg, fint rows) {
    return hs::blas::dgemm(cfg, 'N', 'N', rows, n, k, 1.0, a.data(), m, b.data(), k, 0.0,
                           c.data(), m);
  };
  EXPECT_EQ(call(narrow, m), Route::Real);
  EXPECT_EQ(call(narrow, 100), Route::Narrow);  // 32 KB
  EXPECT_EQ(call(both, m), Route::Parallel);
  Route inside = Route::Real;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    inside = call(narrow, m);
  }
  EXPECT_EQ(inside, Route::Narrow);
}

// Many threads calling the router at once (as MUMPS' L0 subtrees do) get the serial results.
TEST(BlasHookRouter, ConcurrentCallsAreThreadSafe) {
  const RouteConfig cfg = hs::blas::parse_route_config("all,minkb=0,threads=2");
  const fint m = 257, k = 64;
  const auto a = random_vec(static_cast<std::size_t>(m) * k, 21);
  const auto b = random_vec(static_cast<std::size_t>(k) * 16, 22);
  const auto c0 = random_vec(static_cast<std::size_t>(m) * 16, 23);
  std::vector<std::vector<double>> serial(16);
  for (fint n = 1; n <= 16; ++n) {
    serial[n - 1] = c0;
    hs::blas::dgemm(cfg, 'N', 'N', m, n, k, -1.0, a.data(), m, b.data(), k, 1.0,
                    serial[n - 1].data(), m);
  }
  std::vector<std::thread> pool;
  std::vector<int> bad(16, 0);
  for (int t = 0; t < 16; ++t)
    pool.emplace_back([&, t] {
      for (int rep = 0; rep < 20; ++rep) {
        const fint n = 1 + (t + rep) % 16;
        std::vector<double> c = c0;
        hs::blas::dgemm(cfg, 'N', 'N', m, n, k, -1.0, a.data(), m, b.data(), k, 1.0, c.data(), m);
        if (c != serial[n - 1]) ++bad[t];
      }
    });
  for (auto& th : pool) th.join();
  for (int t = 0; t < 16; ++t) EXPECT_EQ(bad[t], 0) << "thread " << t;
  // Inside an OpenMP region the parallel route runs serially in each thread: the results are
  // those of the same configuration with one thread.
  const RouteConfig one = hs::blas::parse_route_config("all,minkb=0,threads=1");
  std::vector<std::vector<double>> serial1(16);
  for (fint n = 1; n <= 16; ++n) {
    serial1[n - 1] = c0;
    hs::blas::dgemm(one, 'N', 'N', m, n, k, -1.0, a.data(), m, b.data(), k, 1.0,
                    serial1[n - 1].data(), m);
  }
  int bad_omp = 0;
#pragma omp parallel num_threads(4) reduction(+ : bad_omp)
  for (int rep = 0; rep < 10; ++rep) {
    const fint n = 1 + (omp_get_thread_num() * 5 + rep) % 16;
    std::vector<double> c = c0;
    hs::blas::dgemm(cfg, 'N', 'N', m, n, k, -1.0, a.data(), m, b.data(), k, 1.0, c.data(), m);
    if (c != serial1[n - 1]) ++bad_omp;
  }
  EXPECT_EQ(bad_omp, 0);
}

// ------------------------------------------------------------------- lookup of the real routines

// In the test executable (no hook symbols) the next definition of dgemm_ is the global one, the
// real BLAS's, and real_blas() uses it.
TEST(BlasHookResolve, NextDefinitionIsTheGlobalBlas) {
  void* p = hs::blas::next_definition("dgemm_");
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p, dlsym(RTLD_DEFAULT, "dgemm_"));
  EXPECT_EQ(reinterpret_cast<void*>(hs::blas::real_blas().dgemm), p);
  EXPECT_FALSE(hs::blas::real_blas().library.empty());
  EXPECT_EQ(hs::blas::next_definition("hs_no_such_symbol_xyz"), nullptr);
}

// A library opened with dlopen(RTLD_LOCAL) is outside the global scope that dlsym(RTLD_NEXT /
// RTLD_DEFAULT) searches - MUMPS and the BLAS under the Python module - and the fallback still
// finds its definitions, skipping the object it is told to skip. The library here is the hook's
// own shared object, whose hs::blas functions the executable does not export.
TEST(BlasHookResolve, FindsDefinitionsInLocallyOpenedLibraries) {
  void* lib = dlopen(HSOLVE_BLASHOOK_SO, RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(lib, nullptr) << dlerror();
  const char* sym = "_ZN2hs4blas15next_definitionEPKc";  // hs::blas::next_definition(const char*)
  void* in_lib = dlsym(lib, sym);
  ASSERT_NE(in_lib, nullptr);
  ASSERT_EQ(dlsym(RTLD_DEFAULT, sym), nullptr) << "the executable exports " << sym;
  EXPECT_EQ(hs::blas::find_loaded_definition(sym, nullptr), in_lib);
  Dl_info info{};
  ASSERT_NE(dladdr(in_lib, &info), 0);
  EXPECT_EQ(hs::blas::find_loaded_definition(sym, info.dli_fbase), nullptr);
  // dgemm_ is defined by the hook object too; skipping it leaves the real BLAS.
  void* real = hs::blas::find_loaded_definition("dgemm_", info.dli_fbase);
  EXPECT_EQ(real, dlsym(RTLD_DEFAULT, "dgemm_"));
  EXPECT_NE(real, dlsym(lib, "dgemm_"));
  dlclose(lib);
}

// ---------------------------------------------------------------------------------- profiler

TEST(BlasHookProfile, SizeClassesAndFlags) {
  namespace prof = hs::blas::prof;
  EXPECT_EQ(prof::size_class(0), 0);
  EXPECT_EQ(prof::size_class(-5), 0);
  EXPECT_EQ(prof::size_class(1), 1);
  EXPECT_EQ(prof::size_class(2), 2);
  EXPECT_EQ(prof::size_class(3), 2);
  EXPECT_EQ(prof::size_class(4), 3);
  EXPECT_EQ(prof::size_class(1023), 10);
  EXPECT_EQ(prof::size_class(1024), 11);
  EXPECT_EQ(prof::gemm_flags('N', 'n'), 0);
  EXPECT_EQ(prof::gemm_flags('t', 'N'), 1);
  EXPECT_EQ(prof::gemm_flags('N', 'C'), 2);
  EXPECT_EQ(prof::trsm_flags('R', 'l', 'T', 'u'), 15);
  EXPECT_EQ(prof::trsm_flags('L', 'U', 'N', 'N'), 0);
  EXPECT_EQ(prof::gemv_flags('T'), 1);
  EXPECT_STREQ(prof::op_name(prof::Op::Idamax), "idamax");
  EXPECT_STREQ(prof::phase_name(prof::Phase::SolveP), "solvep");
  EXPECT_STREQ(hs::blas::route_name(Route::Narrow), "narrow");
  prof::set_phase(prof::Phase::Count);
  EXPECT_EQ(prof::phase(), prof::Phase::Count);
  prof::set_phase(prof::Phase::None);
}

}  // namespace
