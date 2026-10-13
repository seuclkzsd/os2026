// Regression tests of the BLAS hook's review fixes (module blas, P5c; docs/BLAS.md):
//   * the parallel routes (dgemv split, gemv1 + par, parallel_gemm, split-K, narrow + par) compute
//     every block when the OpenMP runtime forms a smaller team than they ask for - one thread
//     (max-active-levels 0, below) or two (OMP_THREAD_LIMIT=2: ctest
//     test_asmkern_fix.ShortTeamUnderThreadLimit2 runs the AsmkernFixShortTeam suite so);
//   * the narrow NN kernels give bitwise the same result for every row split (thread count) also
//     for general alpha / beta, i.e. the register path (<= kRegRows rows) and the chunked path
//     round alpha * t + beta * c alike (hsolve_blas is compiled with -ffp-contract=off).
// The real BLAS is the test executable's (OpenBLAS), found by the hook through dlsym(RTLD_NEXT).
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "blas/blas_hook.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

// Keeps the BLAS library in the executable's dependencies (the toolchain links with --as-needed).
[[maybe_unused, gnu::used]] void (*const kLinkBlas)(
    const char*, const char*, const hs::lapack::fint*, const hs::lapack::fint*,
    const hs::lapack::fint*, const double*, const double*, const hs::lapack::fint*, const double*,
    const hs::lapack::fint*, const double*, double*, const hs::lapack::fint*, hs::lapack::fstrlen,
    hs::lapack::fstrlen) = &hs::lapack::dgemm_;

using hs::blas::fint;
using hs::blas::Isa;
using hs::blas::Route;
using hs::blas::RouteConfig;

std::vector<double> random_vec(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& x : v) x = u(g);
  return v;
}

std::vector<Isa> isas() {
  std::vector<Isa> v{Isa::Portable};
  if (hs::blas::cpu_has_avx2_fma()) v.push_back(Isa::Avx2);
  if (hs::blas::cpu_has_avx512f()) v.push_back(Isa::Avx512);
  return v;
}

// Team the runtime forms for `num_threads(want)` under the current settings.
int team_size(int want) {
  int got = 0;
#pragma omp parallel num_threads(want)
  {
#pragma omp single
    got = omp_get_num_threads();
  }
  return got;
}

// Sets max-active-levels for the scope (0: every parallel region runs on one thread).
class MaxActiveLevels {
 public:
  explicit MaxActiveLevels(int levels) : old_(omp_get_max_active_levels()) {
    omp_set_max_active_levels(levels);
  }
  ~MaxActiveLevels() { omp_set_max_active_levels(old_); }
  MaxActiveLevels(const MaxActiveLevels&) = delete;
  MaxActiveLevels& operator=(const MaxActiveLevels&) = delete;

 private:
  int old_;
};

constexpr int kThreads = 8;  // what the routes ask for (threads=8 in the configurations)

// Long-double reference of y(i) = alpha * sum_p A(i, p) x(p) + beta * y(i) for i < rows, with
// A(i, p) = a[i * si + p * sp], x(p) = x[p * sx], y(i) = y[i * sy], and the error bound
// 4 (len + 2) eps (|alpha| sum |A(i, p) x(p)| + |beta y(i)|). Plain loops, not the real BLAS: a
// threaded OpenMP OpenBLAS call deadlocks in a team smaller than it plans for, which these tests
// create on purpose.
void ref_products(fint rows, fint len, double alpha, const double* a, std::size_t si,
                  std::size_t sp, const double* x, std::size_t sx, double beta, const double* y,
                  std::size_t sy, std::vector<double>* want, std::vector<double>* bound) {
  want->resize(rows);
  bound->resize(rows);
  for (fint i = 0; i < rows; ++i) {
    long double s = 0.0L, sa = 0.0L;
    for (fint p = 0; p < len; ++p) {
      const long double t = static_cast<long double>(a[i * si + p * sp]) * x[p * sx];
      s += t;
      sa += t < 0 ? -t : t;
    }
    const double yi = y[i * sy];
    (*want)[i] = static_cast<double>(alpha * s + static_cast<long double>(beta) * yi);
    (*bound)[i] = 4.0 * (len + 2) * std::numeric_limits<double>::epsilon() *
                      static_cast<double>(std::abs(alpha) * sa + std::abs(beta * yi)) +
                  1e-300;
  }
}

// C = alpha op(A) B + beta C through `cfg` against the long-double reference; asserts the route
// taken. A block left uncomputed is off by O(1), far above the rounding bound.
void check_gemm(const RouteConfig& cfg, char ta, fint m, fint n, fint k, Route want_route,
                const std::string& what) {
  const fint lda = (ta == 'N' ? m : k) + 3, ldb = k + 1, ldc = m + 2;
  const auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), 41);
  const auto b = random_vec(static_cast<std::size_t>(ldb) * n, 42);
  const auto c0 = random_vec(static_cast<std::size_t>(ldc) * n, 43);
  const double alpha = 0.7, beta = -0.3;
  std::vector<double> got = c0;
  const Route r = hs::blas::dgemm(cfg, ta, 'N', m, n, k, alpha, a.data(), lda, b.data(), ldb,
                                  beta, got.data(), ldc);
  ASSERT_EQ(r, want_route) << what << ": route " << hs::blas::route_name(r);
  // op(A)(i, p) = A(i, p) at a[i + p lda] ('N') or A(p, i) at a[p + i lda] ('T').
  const std::size_t si = ta == 'N' ? 1 : lda, sp = ta == 'N' ? lda : 1;
  std::vector<double> want, bound;
  for (fint j = 0; j < n; ++j) {
    const std::size_t cj = static_cast<std::size_t>(j) * ldc;
    ref_products(m, k, alpha, a.data(), si, sp, b.data() + static_cast<std::size_t>(j) * ldb, 1,
                 beta, c0.data() + cj, 1, &want, &bound);
    for (fint i = 0; i < m; ++i)
      ASSERT_LE(std::abs(got[cj + i] - want[i]), bound[i])
          << what << " " << ta << " m=" << m << " n=" << n << " k=" << k << " at (" << i << ", "
          << j << ")";
    for (fint i = m; i < ldc; ++i) ASSERT_EQ(got[cj + i], c0[cj + i]) << what << " padding";
  }
}

void check_gemv(const RouteConfig& cfg, char t, fint m, fint n, fint inc, const std::string& what) {
  const fint lda = m + 3;
  const auto a = random_vec(static_cast<std::size_t>(lda) * n, 51);
  const fint lx = t == 'N' ? n : m, ly = t == 'N' ? m : n;
  const auto x = random_vec(static_cast<std::size_t>(lx) * inc, 52);
  const auto y0 = random_vec(static_cast<std::size_t>(ly) * inc, 53);
  std::vector<double> got = y0;
  const Route r =
      hs::blas::dgemv(cfg, t, m, n, 0.7, a.data(), lda, x.data(), inc, -0.3, got.data(), inc);
  ASSERT_EQ(r, Route::Parallel) << what;
  std::vector<double> want, bound;
  ref_products(ly, lx, 0.7, a.data(), t == 'N' ? 1 : lda, t == 'N' ? lda : 1, x.data(), inc, -0.3,
               y0.data(), inc, &want, &bound);
  for (fint i = 0; i < ly; ++i)
    ASSERT_LE(std::abs(got[static_cast<std::size_t>(i) * inc] - want[i]), bound[i])
        << what << " dgemv " << t << " m=" << m << " n=" << n << " inc=" << inc << " at " << i;
}

// Every parallel route with threads=8, against the long-double reference.
void check_parallel_routes(const std::string& what) {
  const RouteConfig par = hs::blas::parse_route_config("par,minkb=0,threads=8");
  const RouteConfig gemv1 = hs::blas::parse_route_config("gemv1,par,minkb=0,threads=8");
  for (char t : {'N', 'T'})
    for (fint inc : {1, 2}) check_gemv(par, t, 4000, 50, inc, what);  // parallel_gemv
  for (char ta : {'N', 'T'}) {
    check_gemm(gemv1, ta, 4000, 1, 50, Route::Parallel, what + " gemv1");  // parallel_gemv, N = 1
    check_gemm(par, ta, 4000, 4, 50, Route::Parallel, what + " par");      // parallel_gemm
  }
  check_gemm(par, 'T', 3, 2, 100000, Route::SplitK, what + " splitk");     // splitk_gemm_tn
  if (hs::blas::best_isa() != Isa::Portable) {                              // run_split
    const RouteConfig narrow = hs::blas::parse_route_config("narrow,par,minkb=0,threads=8");
    for (char ta : {'N', 'T'}) check_gemm(narrow, ta, 4000, 4, 50, Route::Parallel, what + " narrow");
  }
}

// max-active-levels 0: every parallel region gets one thread.
TEST(AsmkernFixShortTeam, ParallelRoutesComputeEveryBlockOnAOneThreadTeam) {
  const MaxActiveLevels inactive(0);
  ASSERT_EQ(team_size(kThreads), 1) << "the runtime ignored max-active-levels 0";
  check_parallel_routes("max-active-levels 0");
}

// OMP_THREAD_LIMIT=2 (ctest test_asmkern_fix.ShortTeamUnderThreadLimit2): teams of at most 2.
TEST(AsmkernFixShortTeam, ParallelRoutesComputeEveryBlockUnderAThreadLimit) {
  if (omp_get_thread_limit() >= kThreads)
    GTEST_SKIP() << "needs OMP_THREAD_LIMIT < " << kThreads
                 << " (ctest test_asmkern_fix.ShortTeamUnderThreadLimit2)";
  ASSERT_LT(team_size(kThreads), kThreads);
  check_parallel_routes("OMP_THREAD_LIMIT=" + std::to_string(omp_get_thread_limit()));
}

// The full team: same checks (the routes and the comparison are right in the normal case).
TEST(AsmkernFixShortTeam, ParallelRoutesAreAccurateOnAFullTeam) {
  if (team_size(kThreads) != kThreads) GTEST_SKIP() << "the runtime forms no team of " << kThreads;
  check_parallel_routes("full team");
}

// Row splits without a K split (k < 512) leave every element's arithmetic unchanged for general
// alpha / beta (inexact products, where an FMA in only one of the two NN paths would show): the
// result is bitwise the same for every thread count. min_rows = 8 makes blocks of 8-32 rows,
// which take the NN register path, while the whole (one thread) takes the chunked path.
TEST(AsmkernFixNarrowDeterminism, NnAndTnAreBitwiseIndependentOfTheRowSplitForGeneralScalars) {
  const double scalars[][2] = {{0.7, -0.3}, {1.3, 0.9}, {0.7, 0.0}, {-1.0, 1.0}};
  std::uint64_t seed = 70;
  for (Isa isa : isas())
    for (char ta : {'N', 'T'})
      for (const fint m : {33, 40, 61, 130, 1025})
        for (const fint n : {1, 3, 6})
          for (const fint k : {5, 300}) {
            const fint lda = (ta == 'N' ? m : k) + 1;
            const auto a = random_vec(static_cast<std::size_t>(lda) * (ta == 'N' ? k : m), ++seed);
            const auto b = random_vec(static_cast<std::size_t>(k) * n, ++seed);
            const auto c0 = random_vec(static_cast<std::size_t>(m) * n, ++seed);
            for (const auto& s : scalars) {
              std::vector<double> ref;
              for (int threads : {1, 2, 3, 4, 7, 16}) {
                std::vector<double> c = c0;
                if (ta == 'N')
                  hs::blas::narrow_gemm_nn(isa, threads, m, n, k, s[0], a.data(), lda, b.data(), k,
                                           s[1], c.data(), m, 8);
                else
                  hs::blas::narrow_gemm_tn(isa, threads, m, n, k, s[0], a.data(), lda, b.data(), k,
                                           s[1], c.data(), m, 4);
                if (ref.empty()) {
                  ref = c;
                  continue;
                }
                std::size_t differ = 0;
                for (std::size_t i = 0; i < c.size(); ++i) differ += c[i] != ref[i] ? 1 : 0;
                ASSERT_EQ(differ, 0u) << hs::blas::isa_name(isa) << " " << ta << " m=" << m
                                      << " n=" << n << " k=" << k << " alpha=" << s[0]
                                      << " beta=" << s[1] << " threads=" << threads;
              }
            }
          }
}

// The NN register path (m <= kRegRows: 32 rows for AVX-512, 16 for AVX2) against the chunked
// path on the same rows: the first m rows of an (m + 8)-row product equal the m-row product.
TEST(AsmkernFixNarrowDeterminism, NnRegisterPathRoundsLikeTheChunkedPath) {
  for (Isa isa : isas())
    for (const fint m : {8, 16, 32})
      for (const fint n : {1, 4, 5})
        for (const fint k : {3, 64, 301}) {
          const fint big = m + 8, lda = big;
          const auto a = random_vec(static_cast<std::size_t>(lda) * k, 81);
          const auto b = random_vec(static_cast<std::size_t>(k) * n, 82);
          const auto c0 = random_vec(static_cast<std::size_t>(lda) * n, 83);
          std::vector<double> whole = c0, part = c0;
          hs::blas::narrow_gemm_nn(isa, 1, big, n, k, 0.7, a.data(), lda, b.data(), k, -0.3,
                                   whole.data(), lda);
          hs::blas::narrow_gemm_nn(isa, 1, m, n, k, 0.7, a.data(), lda, b.data(), k, -0.3,
                                   part.data(), lda);
          for (fint j = 0; j < n; ++j)
            for (fint i = 0; i < m; ++i) {
              const std::size_t q = i + static_cast<std::size_t>(j) * lda;
              ASSERT_EQ(part[q], whole[q]) << hs::blas::isa_name(isa) << " m=" << m << " n=" << n
                                           << " k=" << k << " at (" << i << ", " << j << ")";
            }
        }
}

}  // namespace
