// Unit tests for the certificate-cost facilities of src/factor: InertiaCounter (exact counts for
// shifted Laplacians, prescribed spectra, generalized pencils and saddle points, real and through
// the complex embedding; the source factorization stays valid; single-precision counts at wide
// gaps; error paths), inertia-only (discard-factors) MUMPS instances and their memory, pivot-order
// reuse, and thread safety (distinct objects used from two threads; thread count fixed against
// later OpenMP changes). Analytic or dense reference spectra throughout.
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "factor/mumps_api.hpp"
#include "factor/mumps_factorization.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/dense_ops.hpp"
#include "support/fixtures.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::FactorOptions;
using hs::FactorPrecision;
using hs::index_t;
using hs::InertiaCounter;
using hs::ShiftedPencil;

// ------------------------------------------------------------------------------- helpers

index_t count_below(const std::vector<double>& w, double sigma) {
  return std::lower_bound(w.begin(), w.end(), sigma) - w.begin();
}

// `count` shifts in the middle of spectral gaps wider than min_gap, spread over the spectrum.
std::vector<double> mid_gap_shifts(const std::vector<double>& w, int count, double min_gap) {
  std::vector<double> s;
  const index_t n = static_cast<index_t>(w.size());
  for (int t = 1; t <= count; ++t) {
    for (index_t k = n * t / (count + 1); k < n; ++k) {
      if (k > 0 && w[k] - w[k - 1] > min_gap) {
        s.push_back(0.5 * (w[k - 1] + w[k]));
        break;
      }
    }
  }
  return s;
}

// Distinct, irregularly spaced eigenvalues in about [-2, 3].
std::vector<double> irregular_spectrum(index_t n) {
  std::vector<double> w(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) w[i] = -2.0 + 5.0 * (i + 0.5) / n + 0.4 * std::sin(1.3 * i) / n;
  std::sort(w.begin(), w.end());
  return w;
}

// Saddle point [[M, C], [C^T, 0]], M the SPD m x m-grid Laplacian, C random m^2 x n2: the zero
// block forces 2x2 / delayed pivots; with C of full column rank the inertia is (n2, 0, m^2).
CsrMatrix<double> saddle_point(index_t m, index_t n2, int per_column, std::uint64_t seed) {
  const auto lap = hs::test::laplacian_2d<double>(m, m);
  const index_t n1 = lap.nrows;
  std::vector<index_t> r, c;
  std::vector<double> v;
  for (index_t i = 0; i < n1; ++i)
    for (index_t p = lap.rowptr[i]; p < lap.rowptr[i + 1]; ++p) {
      r.push_back(i);
      c.push_back(lap.colind[p]);
      v.push_back(lap.values[p]);
    }
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<index_t> row(0, n1 - 1);
  for (index_t j = 0; j < n2; ++j)
    for (int t = 0; t < per_column; ++t) {
      const index_t i = row(g);
      const double x = u(g);
      r.push_back(i), c.push_back(n1 + j), v.push_back(x);
      r.push_back(n1 + j), c.push_back(i), v.push_back(x);
    }
  return hs::csr_from_coo<double>(n1 + n2, n1 + n2, r, c, v);
}

template <class T>
CsrMatrix<T> as_scalar(const CsrMatrix<double>& a) {
  if constexpr (hs::is_complex_v<T>) {
    return hs::to_complex(a);
  } else {
    return a;
  }
}

hs::Error::Code error_code_of(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const hs::Error& e) {
    return e.code();
  }
  ADD_FAILURE() << "expected an hs::Error";
  return hs::Error::Code::Internal;
}

// Expects exact counts at every shift of `shifts` from a counter on an analysed main object.
template <class T>
void expect_exact_counts(InertiaCounter<T>& counter, const std::vector<double>& w,
                         const std::vector<double>& shifts) {
  const index_t n = static_cast<index_t>(w.size());
  ASSERT_FALSE(shifts.empty());
  for (double sigma : shifts) {
    const hs::Inertia in = counter.count(1.0, -sigma);
    EXPECT_TRUE(in.valid) << "sigma=" << sigma;
    EXPECT_EQ(in.neg, count_below(w, sigma)) << "sigma=" << sigma;
    EXPECT_EQ(in.zero, 0) << "sigma=" << sigma;
    EXPECT_EQ(in.pos, n - count_below(w, sigma)) << "sigma=" << sigma;
  }
}

using Scalars = ::testing::Types<double, cplx>;
template <class T>
class FactorInertiaCounter : public ::testing::Test {};
TYPED_TEST_SUITE(FactorInertiaCounter, Scalars);
template <class T>
class FactorDiscard : public ::testing::Test {};
TYPED_TEST_SUITE(FactorDiscard, Scalars);
template <class T>
class FactorPivotOrder : public ::testing::Test {};
TYPED_TEST_SUITE(FactorPivotOrder, Scalars);
template <class T>
class FactorThreads : public ::testing::Test {};
TYPED_TEST_SUITE(FactorThreads, Scalars);

// ------------------------------------------------------------------------------- InertiaCounter

TYPED_TEST(FactorInertiaCounter, ShiftedLaplacian2DCountsAreExact) {
  using T = TypeParam;
  const index_t nx = 60, ny = 45;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.ordering = 3;  // SCOTCH, as the driver picks for 3-D-like graphs
  auto f = hs::make_mumps_factorization<T>(o);
  f->analyze(pencil, 1.0, -4.0);
  InertiaCounter<T> counter(*f, o);
  EXPECT_EQ(counter.order(), nx * ny);
  expect_exact_counts(counter, w, mid_gap_shifts(w, 12, 1e-7));
  const auto& st = counter.stats();
  EXPECT_TRUE(st.reused_pivot_order);
  EXPECT_EQ(st.analyses, 1);
  EXPECT_EQ(st.counts, 12);
  EXPECT_EQ(st.single_counts, 0);
  EXPECT_GT(st.count_s, 0.0);
  EXPECT_GT(st.max_mb_used, 0.0);
  EXPECT_EQ(f->stats().factorizations, 0);  // the source was never touched
}

// prescribed_spectrum<cplx> is genuinely complex Hermitian (full -Im K block in the embedding).
TYPED_TEST(FactorInertiaCounter, PrescribedSpectrumCountsAreExact) {
  using T = TypeParam;
  const index_t n = 1500;
  const auto w = irregular_spectrum(n);
  const auto a = hs::test::prescribed_spectrum<T>(w, 12, 2026);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, 0.0);
  InertiaCounter<T> counter(*f, {});
  expect_exact_counts(counter, w, mid_gap_shifts(w, 10, 1e-6));
  // count_pair: the two counts of a certificate window.
  const double lo = 0.5 * (w[200] + w[201]), hi = 0.5 * (w[900] + w[901]);
  const auto [ia, ib] = counter.count_pair(1.0, -lo, 1.0, -hi);
  EXPECT_TRUE(ia.valid && ib.valid);
  EXPECT_EQ(ib.neg - ia.neg, 700);
}

// B HPD: nu_-(A - sigma B) counts the generalized eigenvalues below sigma (Sylvester).
TYPED_TEST(FactorInertiaCounter, GeneralizedPencilMatchesDenseCount) {
  using T = TypeParam;
  const index_t n = 300;
  const auto a = hs::test::random_hermitian<T>(n, 0.03, 11);
  const auto b = hs::test::random_hpd<T>(n, 0.02, 12);
  const auto w = hs::test::dense_eigenvalues(a, &b);
  ShiftedPencil<T> pencil(a, &b);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -0.3);
  InertiaCounter<T> counter(*f, {});
  expect_exact_counts(counter, w, mid_gap_shifts(w, 8, 1e-6));
  // inertia of B itself (the driver's definiteness test) on the same counter
  const auto ib = counter.count(0.0, 1.0);
  EXPECT_TRUE(ib.valid && ib.neg == 0 && ib.zero == 0 && ib.pos == n);
}

// A zero diagonal block needs 2x2 / delayed pivots; the counter's analysis sees no matching or
// compressed ordering (given pivot order), so this checks that pivoting still yields exact counts.
TYPED_TEST(FactorInertiaCounter, SaddlePointCountsAreExact) {
  using T = TypeParam;
  const auto a = as_scalar<T>(saddle_point(20, 60, 3, 7));
  const index_t n = a.nrows;
  const auto w = hs::test::dense_eigenvalues(a, static_cast<const CsrMatrix<T>*>(nullptr));
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, 0.0);
  InertiaCounter<T> counter(*f, {});
  const auto in0 = counter.count(1.0, 0.0);
  EXPECT_TRUE(in0.valid);
  EXPECT_EQ(in0.neg, 60);
  EXPECT_EQ(in0.pos, n - 60);
  expect_exact_counts(counter, w, mid_gap_shifts(w, 9, 1e-6));
}

// The point of the counter: the source's factorization survives (solves bitwise unchanged).
TYPED_TEST(FactorInertiaCounter, SourceFactorizationStaysValid) {
  using T = TypeParam;
  const index_t nx = 40, ny = 41, n = nx * ny;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  const double sigma = mid_gap_shifts(w, 3, 1e-6)[1];
  f->analyze(pencil, 1.0, -sigma);
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  const auto in_main = f->inertia();
  std::vector<T> rhs(static_cast<std::size_t>(n));
  hs::linalg::fill_random(rhs.data(), n, 5, 0);
  std::vector<T> x1 = rhs;
  f->solve(x1.data(), 1, n);

  InertiaCounter<T> counter(*f, f->options());
  expect_exact_counts(counter, w, mid_gap_shifts(w, 6, 1e-6));
  std::vector<T> x2 = rhs;
  f->solve(x2.data(), 1, n);
  EXPECT_EQ(x1, x2);
  EXPECT_EQ(f->stats().factorizations, 1);
  EXPECT_EQ(f->inertia().neg, in_main.neg);
  EXPECT_TRUE(f->inertia().valid);
}

// A shift exactly on an eigenvalue: valid == false, never thrown; a nearby shift counts exactly.
TYPED_TEST(FactorInertiaCounter, SingularShiftIsReportedInvalid) {
  using T = TypeParam;
  const index_t nx = 50, ny = 50;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, -3.0);
  InertiaCounter<T> counter(*f, {});
  for (index_t k : {index_t{100}, index_t{1300}}) {
    const double lambda = w[k];
    EXPECT_FALSE(counter.count(1.0, -lambda).valid) << "lambda=" << lambda;
    const double sigma = lambda + 8e-8;
    const auto in = counter.count(1.0, -sigma);
    EXPECT_TRUE(in.valid);
    EXPECT_EQ(in.neg, count_below(w, sigma));
  }
}

// Single precision (screening): exact when every eigenvalue is far (relative to ||K||) from the
// shift; mid-gap shifts with gaps >= 1e-3 * ||K|| here.
TYPED_TEST(FactorInertiaCounter, SinglePrecisionCountsAreExactAtWideGaps) {
  using T = TypeParam;
  const index_t n = 1200;
  const auto w = irregular_spectrum(n);  // gaps ~ 5/n ~ 4e-3, ||K|| <= 5
  const auto a = hs::test::prescribed_spectrum<T>(w, 10, 99);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  f->analyze(pencil, 1.0, 0.0);
  InertiaCounter<T> counter(*f, {});
  for (double sigma : mid_gap_shifts(w, 9, 1e-3)) {
    const auto in = counter.count(1.0, -sigma, true, FactorPrecision::Single);
    EXPECT_TRUE(in.valid) << "sigma=" << sigma;
    EXPECT_EQ(in.neg, count_below(w, sigma)) << "sigma=" << sigma;
  }
  EXPECT_EQ(counter.stats().single_counts, 9);
  EXPECT_EQ(counter.stats().analyses, 2);  // the single instance is analysed on first use
  EXPECT_TRUE(counter.stats().reused_pivot_order);
}

// A Single (smumps) source passes its pivot order to the double counter as well.
TYPED_TEST(FactorInertiaCounter, SinglePrecisionSourceGivesExactDoubleCounts) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(30, 31);
  const auto w = hs::test::laplacian_2d_eigenvalues(30, 31);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.precision = FactorPrecision::Single;
  auto f = hs::make_mumps_factorization<T>(o);
  f->analyze(pencil, 1.0, -1.0);
  InertiaCounter<T> counter(*f, o);  // counts in double whatever o.precision says
  expect_exact_counts(counter, w, mid_gap_shifts(w, 5, 1e-6));
  EXPECT_TRUE(counter.stats().reused_pivot_order);
}

TYPED_TEST(FactorInertiaCounter, MisuseIsRejected) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(10, 10);
  ShiftedPencil<T> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<T>();
  EXPECT_EQ(error_code_of([&] { InertiaCounter<T> c(*f, {}); }), hs::Error::Code::InvalidInput);
  f->analyze(pencil, 1.0, 0.0);
  InertiaCounter<T> counter(*f, {});
  EXPECT_EQ(error_code_of([&] { counter.count(std::nan(""), 1.0); }), hs::Error::Code::InvalidInput);
  EXPECT_EQ(error_code_of([&] { counter.count(1.0, INFINITY); }), hs::Error::Code::InvalidInput);
  FactorOptions bad;
  bad.null_pivot_tol = 0.0;
  EXPECT_EQ(error_code_of([&] { InertiaCounter<T> c(*f, bad); }), hs::Error::Code::InvalidInput);
}

// ------------------------------------------------------------------------------- discard mode

// ICNTL(31)=1 must be set before the analysis; then the factorization keeps only its working
// memory (no factor entries) and refuses to solve, with exactly the same pivot signs.
TYPED_TEST(FactorDiscard, KeepsOnlyWorkingMemoryAndTheSameInertia) {
  using T = TypeParam;
  const index_t nx = 150, ny = 150;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  const double sigma = mid_gap_shifts(w, 3, 1e-7)[0];

  hs::mumps::MumpsFactorization<T, double> keep(FactorOptions{});
  keep.analyze(pencil, 1.0, -sigma);
  ASSERT_TRUE(keep.factorize(1.0, -sigma));

  hs::mumps::MumpsFactorization<T, double> discard(FactorOptions{});
  discard.set_discard_factors(true);
  discard.set_pivot_order(keep.pivot_order());
  discard.analyze(pencil, 1.0, -sigma);
  EXPECT_TRUE(discard.factors_discarded());
  ASSERT_TRUE(discard.factorize(1.0, -sigma));
  EXPECT_EQ(discard.inertia().neg, keep.inertia().neg);
  EXPECT_EQ(discard.inertia().neg, count_below(w, sigma));
  EXPECT_EQ(discard.stats().factor_entries, 0);
  EXPECT_GT(keep.stats().factor_entries, 0);
  // Measured on 1e5 cases: -57..-61%; this 2-D grid: well below 0.7x as well.
  EXPECT_LT(discard.stats().factor_mb_used, 0.7 * keep.stats().factor_mb_used);
  EXPECT_LT(discard.estimate().incore_mb, keep.estimate().incore_mb);
  std::vector<T> b(static_cast<std::size_t>(a.nrows), T(1));
  EXPECT_EQ(error_code_of([&] { discard.solve(b.data(), 1, a.nrows); }), hs::Error::Code::Internal);

  // Switched on after the analysis it has no effect until the next analysis.
  hs::mumps::MumpsFactorization<T, double> late(FactorOptions{});
  late.set_pivot_order(keep.pivot_order());
  late.analyze(pencil, 1.0, -sigma);
  late.set_discard_factors(true);
  ASSERT_TRUE(late.factorize(1.0, -sigma));
  EXPECT_FALSE(late.factors_discarded());
  EXPECT_GT(late.stats().factor_entries, 0);
  late.solve(b.data(), 1, a.nrows);  // factors kept: solves work
}

// release_factors() (JOB=-4) frees the factors but keeps the analysis: the counter can then run
// without both in memory, and the object refactorizes and solves as before.
TYPED_TEST(FactorDiscard, ReleasedFactorsRefactorizeFromTheSameAnalysis) {
  using T = TypeParam;
  const index_t nx = 60, ny = 50, n = nx * ny;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  ShiftedPencil<T> pencil(a, nullptr);
  const auto shifts = mid_gap_shifts(w, 4, 1e-6);
  auto f = hs::make_mumps_factorization<T>();
  EXPECT_FALSE(f->release_factors());  // nothing analysed yet
  f->analyze(pencil, 1.0, -shifts[0]);
  EXPECT_FALSE(f->release_factors());  // nothing factorized yet
  ASSERT_TRUE(f->factorize(1.0, -shifts[0]));
  std::vector<T> x1(static_cast<std::size_t>(n), T(1));
  f->solve(x1.data(), 1, n);
  const index_t neg = f->inertia().neg;

  EXPECT_TRUE(f->release_factors());
  EXPECT_FALSE(f->release_factors());  // already released
  std::vector<T> b(static_cast<std::size_t>(n), T(1));
  EXPECT_EQ(error_code_of([&] { f->solve(b.data(), 1, n); }), hs::Error::Code::Internal);
  // inertia().valid is the "solve() works" test of other modules (ShiftInvertOperator, the
  // driver's rescue): false now, while the counts still describe the released factorization.
  EXPECT_FALSE(f->inertia().valid);
  EXPECT_EQ(f->inertia().neg, neg);
  EXPECT_EQ(error_code_of([&] { hs::ShiftInvertOperator<T> op(*f, pencil, shifts[0]); }),
            hs::Error::Code::InvalidInput);

  InertiaCounter<T> counter(*f, {});
  expect_exact_counts(counter, w, shifts);

  ASSERT_TRUE(f->factorize(1.0, -shifts[0]));
  EXPECT_TRUE(f->inertia().valid);
  hs::ShiftInvertOperator<T> op(*f, pencil, shifts[0]);  // accepted again
  std::vector<T> x2(static_cast<std::size_t>(n), T(1));
  f->solve(x2.data(), 1, n);
  for (index_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(x2[i] - x1[i]), 0.0, 1e-12 * std::abs(x1[i]));
  EXPECT_EQ(f->stats().analyses, 1);
}

// compute_inertia discards its factors: same counts as before.
TYPED_TEST(FactorDiscard, ComputeInertiaIsUnchanged) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(40, 30);
  const auto w = hs::test::laplacian_2d_eigenvalues(40, 30);
  ShiftedPencil<T> pencil(a, nullptr);
  for (double sigma : mid_gap_shifts(w, 4, 1e-6)) {
    const auto in = hs::compute_inertia(pencil, 1.0, -sigma);
    EXPECT_TRUE(in.valid);
    EXPECT_EQ(in.neg, count_below(w, sigma));
  }
}

// ------------------------------------------------------------------------------- pivot order

// SYM_PERM -> PERM_IN does not reproduce the elimination exactly (MUMPS rebuilds the tree), but
// nearly: measured +0.4..3.4% entries on the 1e5 cases, +1..7% (AMD, SCOTCH) and up to +15%
// (METIS) on random sparse matrices.
TYPED_TEST(FactorPivotOrder, IsAPermutationAndNearlyReproducesTheFill) {
  using T = TypeParam;
  const auto grid = hs::test::laplacian_2d<T>(70, 71);
  const auto rnd = hs::test::random_hermitian<T>(2000, 0.003, 3);
  for (const auto* a : {&grid, &rnd}) {
    const bool structured = (a == &grid);
    ShiftedPencil<T> pencil(*a, nullptr);
    FactorOptions o;
    o.ordering = 5;  // METIS
    hs::mumps::MumpsFactorization<T, double> main(o);
    EXPECT_TRUE(main.pivot_order().empty());
    main.analyze(pencil, 1.0, -1.0);
    const std::vector<int>& p = main.pivot_order();
    const index_t m = (hs::is_complex_v<T> ? 2 : 1) * a->nrows;  // order of the MUMPS matrix
    ASSERT_EQ(static_cast<index_t>(p.size()), m);
    std::vector<int> sorted = p;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> iota(static_cast<std::size_t>(m));
    std::iota(iota.begin(), iota.end(), 1);
    EXPECT_EQ(sorted, iota);

    hs::mumps::MumpsFactorization<T, double> again(o);
    again.set_pivot_order(p);
    again.analyze(pencil, 1.0, -1.0);
    const double e0 = static_cast<double>(main.estimate().factor_entries);
    const double e1 = static_cast<double>(again.estimate().factor_entries);
    EXPECT_GE(e1 / e0, 0.95) << (structured ? "grid" : "random");
    EXPECT_LE(e1 / e0, structured ? 1.05 : 1.25) << (structured ? "grid" : "random");
    // and the factorization given the order has the exact inertia
    ASSERT_TRUE(main.factorize(1.0, -1.0));
    ASSERT_TRUE(again.factorize(1.0, -1.0));
    EXPECT_EQ(again.inertia().neg, main.inertia().neg);

    hs::mumps::MumpsFactorization<T, double> wrong(o);
    wrong.set_pivot_order(std::vector<int>(p.begin(), p.end() - 1));
    EXPECT_EQ(error_code_of([&] { wrong.analyze(pencil, 1.0, 0.0); }), hs::Error::Code::InvalidInput);
  }
}

// ------------------------------------------------------------------------------- threads

// The MUMPS call lock must exclude in every build. As a std::mutex it was a no-op in executables
// linked without libpthread (the ASan debug preset, where libstdc++'s __gthread_active_p() is
// false), and DistinctObjectsFromTwoThreadsAreSafe crashed in MUMPS_LOAD_INIT. The test must not
// pull libpthread into the link itself, or it would hide exactly that: std::this_thread::sleep_for
// does (nanosleep binds to libpthread first), hence the yield loop on steady_clock.
TEST(FactorCallLock, ExcludesASecondThread) {
  auto& m = hs::mumps::call_mutex();
  std::atomic<bool> trying{false}, acquired{false};
  m.lock();
  std::thread t([&] {
    trying = true;
    m.lock();
    acquired = true;
    m.unlock();
  });
  while (!trying) std::this_thread::yield();
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  while (!acquired && std::chrono::steady_clock::now() < until) std::this_thread::yield();
  EXPECT_FALSE(acquired.load()) << "the second thread entered the held MUMPS call lock";
  m.unlock();
  t.join();
  EXPECT_TRUE(acquired.load());
}

// Sequential MUMPS is not thread-safe across instances (factor.hpp); the module serializes all
// MUMPS calls, so distinct objects driven from two threads must give exactly the single-threaded
// results. (Without the lock the first concurrent factorization aborted in MUMPS_LOAD_INIT.)
TYPED_TEST(FactorThreads, DistinctObjectsFromTwoThreadsAreSafe) {
  using T = TypeParam;
  const index_t nx = 40, ny = 37, n = nx * ny;
  const auto a = hs::test::laplacian_2d<T>(nx, ny);
  const auto w = hs::test::laplacian_2d_eigenvalues(nx, ny);
  const auto shifts = mid_gap_shifts(w, 6, 1e-6);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.threads = 1;
  o.deterministic = true;

  auto work = [&](int id, std::vector<index_t>& counts, std::vector<T>& x) {
    omp_set_num_threads(1);
    auto f = hs::make_mumps_factorization<T>(o);
    f->analyze(pencil, 1.0, -shifts[0]);
    InertiaCounter<T> counter(*f, o);
    for (int round = 0; round < 15; ++round)
      for (double s : shifts) counts.push_back(counter.count(1.0, -s).neg);
    if (!f->factorize(1.0, -shifts[static_cast<std::size_t>(id)])) counts.push_back(-1);
    x.assign(static_cast<std::size_t>(n), T(1));
    f->solve(x.data(), 1, n);
  };
  std::vector<index_t> c1, c2, r1, r2;
  std::vector<T> x1, x2, y1, y2;
  work(0, r1, y1);  // references, one after the other
  work(1, r2, y2);
  std::thread t1([&] { work(0, c1, x1); });
  std::thread t2([&] { work(1, c2, x2); });
  t1.join();
  t2.join();
  EXPECT_EQ(c1, r1);
  EXPECT_EQ(c2, r2);
  EXPECT_EQ(x1, y1);  // deterministic factorizations: bitwise identical solves
  EXPECT_EQ(x2, y2);
  for (std::size_t i = 0; i < shifts.size(); ++i) EXPECT_EQ(r1[i], count_below(w, shifts[i]));
}

// With L0-OpenMP (deterministic = false) a factorization must use the thread count of its
// analysis (MUMPS INFO -58 otherwise). The counter fixes it at construction (threads = 0 ->
// omp_get_max_threads() then), so a caller that changes the OpenMP setting afterwards is safe.
TYPED_TEST(FactorThreads, CounterKeepsItsThreadCountWhenTheCallerChangesOpenMP) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(50, 40);
  const auto w = hs::test::laplacian_2d_eigenvalues(50, 40);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.deterministic = false;
  o.threads = 0;
  const int saved = omp_get_max_threads();
  omp_set_num_threads(2);
  auto f = hs::make_mumps_factorization<T>(o);
  f->analyze(pencil, 1.0, -1.0);
  InertiaCounter<T> counter(*f, o);
  omp_set_num_threads(1);
  expect_exact_counts(counter, w, mid_gap_shifts(w, 3, 1e-6));
  EXPECT_EQ(omp_get_max_threads(), 1);  // MUMPS restores the caller's setting
  omp_set_num_threads(saved);
}

// The same for a plain factorization: threads = 0 is resolved at analyze() (before, a caller
// that lowered its OpenMP thread count after the analysis got FactorizationFailed, INFO -58).
TYPED_TEST(FactorThreads, FactorizationKeepsTheThreadCountOfItsAnalysis) {
  using T = TypeParam;
  const auto a = hs::test::laplacian_2d<T>(80, 80);
  const auto w = hs::test::laplacian_2d_eigenvalues(80, 80);
  ShiftedPencil<T> pencil(a, nullptr);
  FactorOptions o;
  o.deterministic = false;
  o.threads = 0;
  const int saved = omp_get_max_threads();
  omp_set_num_threads(2);
  auto f = hs::make_mumps_factorization<T>(o);
  f->analyze(pencil, 1.0, -1.0);
  omp_set_num_threads(1);
  const double sigma = mid_gap_shifts(w, 3, 1e-6)[1];
  ASSERT_TRUE(f->factorize(1.0, -sigma));
  EXPECT_EQ(f->inertia().neg, count_below(w, sigma));
  EXPECT_EQ(omp_get_max_threads(), 1);
  EXPECT_EQ(f->options().threads, 0);  // the options are reported as given
  omp_set_num_threads(saved);
}

}  // namespace
