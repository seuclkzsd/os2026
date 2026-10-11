// Tests of src/linalg/dense_ops: deterministic reductions (bitwise identical for any thread count)
// and the CGS orthogonalization routines.
#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/types.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::index_t;
namespace la = hs::linalg;

/// Sets the OpenMP thread count for its lifetime and restores the previous one.
class ThreadCount {
 public:
  explicit ThreadCount(int n) : saved_(omp_get_max_threads()) { omp_set_num_threads(n); }
  ~ThreadCount() { omp_set_num_threads(saved_); }
  ThreadCount(const ThreadCount&) = delete;
  ThreadCount& operator=(const ThreadCount&) = delete;

 private:
  int saved_;
};

template <class T>
bool bitwise_equal(const T& a, const T& b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
bool bitwise_equal(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

constexpr int kThreadCounts[] = {1, 3, 4};

template <class T>
class LinalgDenseOpsReduce : public ::testing::Test {};
using Scalars = ::testing::Types<double, cplx>;
TYPED_TEST_SUITE(LinalgDenseOpsReduce, Scalars);

TYPED_TEST(LinalgDenseOpsReduce, DotAndNorm2MatchExtendedPrecisionReference) {
  using T = TypeParam;
  // Sizes around the lane width (8), the block (1024) and beyond 256 blocks (several blocks per
  // reduction slot).
  for (index_t n : {0, 1, 7, 8, 9, 1023, 1024, 1025, 5000, 300001}) {
    std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    la::fill_random(x.data(), n, 11, 0);
    la::fill_random(y.data(), n, 11, 1);
    long double re = 0, im = 0, nn = 0, mag = 0;
    for (index_t i = 0; i < n; ++i) {
      const std::complex<long double> xi(std::real(x[i]), std::imag(x[i]));
      const std::complex<long double> yi(std::real(y[i]), std::imag(y[i]));
      const auto p = std::conj(xi) * yi;
      re += p.real();
      im += p.imag();
      nn += std::norm(xi);
      mag += std::abs(xi) * std::abs(yi);
    }
    const T d = la::dot(n, x.data(), y.data());
    const double tol = 1e-15 * static_cast<double>(mag) + 1e-300;
    EXPECT_NEAR(std::real(d), static_cast<double>(re), tol) << "n=" << n;
    EXPECT_NEAR(std::imag(d), static_cast<double>(im), tol) << "n=" << n;
    const double nrm = la::norm2(n, x.data());
    EXPECT_NEAR(nrm, std::sqrt(static_cast<double>(nn)), 1e-15 * std::sqrt(static_cast<double>(nn)) + 1e-300)
        << "n=" << n;
    // x^H x is real and equals norm2^2 up to rounding.
    EXPECT_EQ(std::imag(la::dot(n, x.data(), x.data())), 0.0);
  }
}

TYPED_TEST(LinalgDenseOpsReduce, BitwiseIdenticalForAnyThreadCountAndRepetition) {
  using T = TypeParam;
  for (index_t n : {20000, 300001}) {
    std::vector<T> x(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    la::fill_random(x.data(), n, 5, 0);
    la::fill_random(y.data(), n, 5, 1);
    // Scale entries over many orders of magnitude so that any change of summation order would
    // show in the last bits.
    for (index_t i = 0; i < n; ++i) x[i] *= std::ldexp(1.0, static_cast<int>(i % 41) - 20);
    T d_ref;
    double n_ref;
    {
      ThreadCount tc(1);
      d_ref = la::dot(n, x.data(), y.data());
      n_ref = la::norm2(n, x.data());
    }
    for (int threads : kThreadCounts) {
      ThreadCount tc(threads);
      for (int rep = 0; rep < 3; ++rep) {
        EXPECT_TRUE(bitwise_equal(la::dot(n, x.data(), y.data()), d_ref)) << "n=" << n << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(la::norm2(n, x.data()), n_ref)) << "n=" << n << " threads=" << threads;
      }
    }
  }
}

TYPED_TEST(LinalgDenseOpsReduce, NonFiniteEntriesPropagate) {
  using T = TypeParam;
  const index_t n = 5000;
  std::vector<T> x(static_cast<std::size_t>(n), T(1.0));
  x[3333] = T(std::nan(""));
  EXPECT_TRUE(std::isnan(la::norm2(n, x.data())));
  EXPECT_TRUE(std::isnan(std::real(la::dot(n, x.data(), x.data()))));
  x[3333] = T(std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isinf(la::norm2(n, x.data())));
}

// ---- CGS orthogonalization ----------------------------------------------------------------------

enum class Mode { Euclidean, Stored, Recompute };
constexpr Mode kModes[] = {Mode::Euclidean, Mode::Stored, Mode::Recompute};
const char* mode_name(Mode m) {
  return m == Mode::Euclidean ? "euclidean" : m == Mode::Stored ? "stored" : "recompute";
}

la::OrthoOptions policy(la::Reorth r, double eta = 0.70710678118654752) {
  la::OrthoOptions o;
  o.reorth = r;
  o.eta = eta;
  return o;
}

/// Inner-product setting of one test: M = diag(m) with m_i in [1, 100] (identity in Euclidean
/// mode), basis V (n x cap) and, for the stored mode, BV = M V.
template <class T>
struct OrthoSetup {
  index_t n, cap;
  Mode mode;
  std::vector<double> m;
  hs::DenseMatrix<T> v, bv;
  la::ApplyFn<T> apply;
  index_t k = 0;  ///< columns of V built so far

  OrthoSetup(const OrthoSetup&) = delete;  // `apply` captures this
  OrthoSetup& operator=(const OrthoSetup&) = delete;
  OrthoSetup(index_t n_, index_t cap_, Mode mode_) : n(n_), cap(cap_), mode(mode_), v(n_, cap_), bv(n_, cap_) {
    m.resize(static_cast<std::size_t>(n));
    for (index_t i = 0; i < n; ++i)
      m[i] = mode == Mode::Euclidean ? 1.0 : 1.0 + 99.0 * static_cast<double>((i * 7919) % 1000) / 999.0;
    apply = [this](const T* x, T* y) {
      for (index_t i = 0; i < n; ++i) y[i] = m[i] * x[i];
    };
  }

  /// Orthogonalizes w (bw = M w in stored mode) against V(:, 0:k) with the setup's mode.
  double orthogonalize(T* w, T* bw, T* h, const la::OrthoOptions& opt, la::OrthoInfo* info) const {
    switch (mode) {
      case Mode::Euclidean:
        return la::orthogonalize_cgs2<T>(n, k, v.data(), n, nullptr, 0, w, nullptr, h, nullptr, opt, info);
      case Mode::Stored:
        return la::orthogonalize_cgs2<T>(n, k, v.data(), n, bv.data(), n, w, bw, h, nullptr, opt, info);
      case Mode::Recompute:
        return la::orthogonalize_cgs2<T>(n, k, v.data(), n, nullptr, 0, w, bw, h, &apply, opt, info);
    }
    return 0.0;
  }

  /// Appends x (orthogonalized and M-normalized) as column k. Returns the pass count. In stored
  /// mode M w is recomputed for the new column, as callers must (the updated bw carries the
  /// cancellation error, see orthogonalize_cgs2).
  int append(const std::vector<T>& x, const la::OrthoOptions& opt) {
    T* w = v.col(k);
    T* bw = bv.col(k);
    std::copy(x.begin(), x.end(), w);
    if (mode == Mode::Stored) apply(w, bw);
    la::OrthoInfo info;
    double nrm = orthogonalize(w, mode == Mode::Euclidean ? nullptr : bw, nullptr, opt, &info);
    if (mode == Mode::Stored) {
      apply(w, bw);
      nrm = std::sqrt(std::real(la::dot(n, w, bw)));
    }
    EXPECT_GT(nrm, 0.0);
    la::scal(n, T(1.0 / nrm), w);
    if (mode == Mode::Stored) la::scal(n, T(1.0 / nrm), bw);
    ++k;
    return info.passes;
  }

  /// max |V^H M V - I| over the first k columns (M V recomputed exactly).
  double loss_of_orthogonality() const {
    hs::DenseMatrix<T> mv(n, k), g(k, k);
    for (index_t j = 0; j < k; ++j) apply(v.col(j), mv.col(j));
    la::gram(n, k, v.data(), n, mv.data(), n, g.data(), k);
    double worst = 0.0;
    for (index_t j = 0; j < k; ++j)
      for (index_t i = 0; i < k; ++i) worst = std::max(worst, std::abs(g(i, j) - T(i == j ? 1.0 : 0.0)));
    return worst;
  }
};

template <class T>
class LinalgDenseOpsCgs : public ::testing::Test {};
TYPED_TEST_SUITE(LinalgDenseOpsCgs, Scalars);

TYPED_TEST(LinalgDenseOpsCgs, NearlyDependentInputsStayOrthonormal) {
  using T = TypeParam;
  const index_t n = 3000, cap = 48;
  for (Mode mode : kModes)
    for (la::Reorth reorth : {la::Reorth::Always, la::Reorth::IfNeeded}) {
      OrthoSetup<T> s(n, cap, mode);
      const auto opt = policy(reorth);
      std::vector<T> x(static_cast<std::size_t>(n)), r(static_cast<std::size_t>(n));
      la::fill_random(x.data(), n, 21, 0);
      int two_pass = 0, one_pass = 0;
      for (index_t j = 0; j < cap; ++j) {
        // Every sixth vector is fresh (independent of span(V)); in between graded dependence
        // x_j = x_{j-1} + delta r_j with delta = 1e-2 .. 1e-10, i.e. ||x_j after projection|| /
        // ||x_j|| ~ delta (heavy cancellation).
        const double delta = std::pow(10.0, -static_cast<double>(j % 6) * 2.0);
        la::fill_random(r.data(), n, 21, static_cast<std::uint64_t>(j + 1));
        if (j % 6 == 0) {
          x = r;
        } else {
          la::axpy(n, T(delta), r.data(), x.data());
        }
        const int passes = s.append(x, opt);
        (passes == 2 ? two_pass : one_pass) += 1;
        if (j % 6 != 0) {
          EXPECT_EQ(passes, 2) << mode_name(mode) << " j=" << j;
        }
      }
      // Column 0 runs no pass (k == 0); with DGKS the fresh vectors j = 6, 12, ... take one.
      EXPECT_EQ(one_pass, reorth == la::Reorth::Always ? 1 : static_cast<int>((cap + 5) / 6)) << mode_name(mode);
      const double loss = s.loss_of_orthogonality();
      EXPECT_LE(loss, 1e-14) << mode_name(mode) << (reorth == la::Reorth::Always ? " always" : " dgks");
    }
}

TYPED_TEST(LinalgDenseOpsCgs, CoefficientsNormsAndMw) {
  using T = TypeParam;
  const index_t n = 5000, k = 30;
  for (Mode mode : kModes)
    for (double target : {1.0, 0.3, 1e-8}) {
      OrthoSetup<T> s(n, k, mode);
      std::vector<T> x(static_cast<std::size_t>(n));
      for (index_t j = 0; j < k; ++j) {
        la::fill_random(x.data(), n, 5, static_cast<std::uint64_t>(j));
        s.append(x, policy(la::Reorth::Always));
      }
      // w0 = V y + r_perp scaled so that ||w0 after projection||_M / ||w0||_M ~ target.
      std::vector<T> y(static_cast<std::size_t>(k)), w0(static_cast<std::size_t>(n)), bw(static_cast<std::size_t>(n));
      la::fill_random(y.data(), k, 6, 0);
      la::fill_random(x.data(), n, 6, 1);
      if (mode == Mode::Stored) s.apply(x.data(), bw.data());
      const double xn = s.orthogonalize(x.data(), mode == Mode::Euclidean ? nullptr : bw.data(), nullptr,
                                        policy(la::Reorth::Always), nullptr);
      la::multiply(n, k, 1, s.v.data(), n, y.data(), k, w0.data(), n);
      const double yn = la::norm2(k, y.data());
      const double scale = target < 1.0 ? target * yn / std::sqrt(1.0 - target * target) / xn : 1.0 / xn;
      if (target == 1.0) std::fill(w0.begin(), w0.end(), T(0));
      la::axpy(n, T(scale), x.data(), w0.data());

      for (la::Reorth reorth : {la::Reorth::Always, la::Reorth::IfNeeded}) {
        std::vector<T> w = w0, h(static_cast<std::size_t>(k));
        if (mode == Mode::Stored) s.apply(w.data(), bw.data());
        la::OrthoInfo info;
        const double nrm =
            s.orthogonalize(w.data(), mode == Mode::Euclidean ? nullptr : bw.data(), h.data(), policy(reorth), &info);
        const char* what = reorth == la::Reorth::Always ? "always" : "dgks";
        // Passes: DGKS skips pass 2 exactly when pass 1 kept >= eta of the norm.
        EXPECT_EQ(info.passes, reorth == la::Reorth::Always || target < 0.7 ? 2 : 1) << mode_name(mode) << " " << what;
        // Norms: before = ||w0||_M, pass 1 ~ target * before, after = return value.
        double before2 = 0.0;
        for (index_t i = 0; i < n; ++i) before2 += s.m[i] * hs::abs2(w0[i]);
        EXPECT_NEAR(info.norm_before, std::sqrt(before2), 1e-13 * std::sqrt(before2));
        EXPECT_NEAR(info.norm_pass1 / info.norm_before, target, 1e-6 * target);
        EXPECT_EQ(info.norm_after, nrm);
        // w0 = V h + w (the coefficients account for everything removed).
        std::vector<T> rec(static_cast<std::size_t>(n));
        la::multiply(n, k, 1, s.v.data(), n, h.data(), k, rec.data(), n);
        double err = 0.0;
        for (index_t i = 0; i < n; ++i) err = std::max(err, std::abs(rec[i] + w[i] - w0[i]));
        EXPECT_LE(err, 1e-14 * info.norm_before) << mode_name(mode) << " " << what;
        // h = V^H M w0 = y up to rounding (|y_j| <= 1).
        for (index_t j = 0; target < 1.0 && j < k; ++j) EXPECT_NEAR(std::abs(h[j] - y[j]), 0.0, 1e-13) << j;
        // Result: M-orthogonal to V, return value = ||w||_M, bw == M w.
        // Stored mode updates bw by subtraction: its error is relative to M w0, not to M w (it
        // dominates after heavy cancellation), and so is the error of the returned norm.
        double after2 = 0.0, worst_bw = 0.0, wn = 0.0, w0n = 0.0;
        for (index_t i = 0; i < n; ++i) {
          after2 += s.m[i] * hs::abs2(w[i]);
          wn = std::max(wn, std::abs(w[i]));
          w0n = std::max(w0n, std::abs(w0[i]));
          if (mode != Mode::Euclidean) worst_bw = std::max(worst_bw, std::abs(bw[i] - s.m[i] * w[i]));
        }
        const bool stored = mode == Mode::Stored;
        EXPECT_NEAR(nrm, std::sqrt(after2), 1e-13 * (stored ? info.norm_before : std::sqrt(after2)))
            << mode_name(mode) << " " << what;
        EXPECT_LE(worst_bw, 1e-13 * 100.0 * (stored ? w0n : wn)) << mode_name(mode) << " " << what;
        std::vector<T> mw(static_cast<std::size_t>(n));
        s.apply(w.data(), mw.data());
        double proj = 0.0;
        for (index_t j = 0; j < k; ++j) proj = std::max(proj, std::abs(la::dot(n, s.v.col(j), mw.data())));
        EXPECT_LE(proj, 1e-14 * nrm) << mode_name(mode) << " " << what << " target=" << target;
      }
    }
}

TYPED_TEST(LinalgDenseOpsCgs, EtaControlsTheSecondPass) {
  using T = TypeParam;
  const index_t n = 2000, k = 10;
  OrthoSetup<T> s(n, k, Mode::Euclidean);
  std::vector<T> x(static_cast<std::size_t>(n));
  for (index_t j = 0; j < k; ++j) {
    la::fill_random(x.data(), n, 9, static_cast<std::uint64_t>(j));
    s.append(x, policy(la::Reorth::Always));
  }
  // w = v_0 + 0.5 * (unit vector orthogonal to V): ratio 0.5 / sqrt(1.25) = 0.447.
  std::vector<T> w0(static_cast<std::size_t>(n));
  la::fill_random(x.data(), n, 9, 100);
  const double xn = s.orthogonalize(x.data(), nullptr, nullptr, policy(la::Reorth::Always), nullptr);
  for (index_t i = 0; i < n; ++i) w0[i] = s.v(i, 0) + T(0.5 / xn) * x[i];
  auto passes = [&](la::OrthoOptions opt) {
    std::vector<T> w = w0;
    la::OrthoInfo info;
    s.orthogonalize(w.data(), nullptr, nullptr, opt, &info);
    return info.passes;
  };
  EXPECT_EQ(passes(policy(la::Reorth::IfNeeded, 0.5)), 2);
  EXPECT_EQ(passes(policy(la::Reorth::IfNeeded, 0.4)), 1);
  EXPECT_EQ(passes(policy(la::Reorth::IfNeeded, 0.0)), 1);
  EXPECT_EQ(passes(policy(la::Reorth::IfNeeded, 1.0)), 2);
  EXPECT_EQ(passes(policy(la::Reorth::Always, 0.0)), 2);
  // The defaults: DGKS with 1/sqrt(2).
  EXPECT_EQ(la::OrthoOptions{}.reorth, la::Reorth::IfNeeded);
  EXPECT_DOUBLE_EQ(la::OrthoOptions{}.eta, 1.0 / std::sqrt(2.0));
  std::vector<T> w = w0;
  EXPECT_THROW(s.orthogonalize(w.data(), nullptr, nullptr, policy(la::Reorth::IfNeeded, 1.5), nullptr), hs::Error);
  EXPECT_THROW(s.orthogonalize(w.data(), nullptr, nullptr, policy(la::Reorth::IfNeeded, std::nan("")), nullptr),
               hs::Error);
}

TYPED_TEST(LinalgDenseOpsCgs, EmptyBasisAndNonFiniteInput) {
  using T = TypeParam;
  const index_t n = 3000;
  for (Mode mode : kModes) {
    OrthoSetup<T> s(n, 4, mode);
    std::vector<T> w(static_cast<std::size_t>(n)), bw(static_cast<std::size_t>(n), T(7.0)), h(4, T(5.0));
    la::fill_random(w.data(), n, 3, 0);
    if (mode == Mode::Stored) s.apply(w.data(), bw.data());
    const std::vector<T> w0 = w;
    la::OrthoInfo info;
    // k == 0: w unchanged, M w computed in recompute mode, the norm returned.
    const double nrm = s.orthogonalize(w.data(), mode == Mode::Euclidean ? nullptr : bw.data(), h.data(),
                                       la::OrthoOptions{}, &info);
    EXPECT_TRUE(w == w0);
    EXPECT_EQ(info.passes, 0);
    EXPECT_EQ(info.norm_before, nrm);
    double m2 = 0.0;
    for (index_t i = 0; i < n; ++i) {
      m2 += s.m[i] * hs::abs2(w[i]);
      if (mode != Mode::Euclidean) {
        EXPECT_NEAR(std::abs(bw[i] - s.m[i] * w[i]), 0.0, 1e-13);
      }
    }
    EXPECT_NEAR(nrm, std::sqrt(m2), 1e-13 * std::sqrt(m2));
    // NaN input: two passes (no DGKS skip on NaN), NaN 2-norm or 0 M-norm, no crash.
    s.append(w0, la::OrthoOptions{});
    w = w0;
    w[17] = T(std::nan(""));
    if (mode == Mode::Stored) s.apply(w.data(), bw.data());
    const double bad = s.orthogonalize(w.data(), mode == Mode::Euclidean ? nullptr : bw.data(), nullptr,
                                       la::OrthoOptions{}, &info);
    EXPECT_EQ(info.passes, 2) << mode_name(mode);
    EXPECT_TRUE(std::isnan(info.norm_after)) << mode_name(mode);
    if (mode == Mode::Euclidean) {
      EXPECT_TRUE(std::isnan(bad));
    } else {
      EXPECT_EQ(bad, 0.0);
    }
  }
}

TYPED_TEST(LinalgDenseOpsCgs, BitwiseIdenticalForAnyThreadCount) {
  using T = TypeParam;
  // (n, k): several reduction slots of one block each (all modes); more than 256 blocks, i.e.
  // several blocks per slot; a basis wide enough for every column group of the kernels plus
  // remainders. The last two are mode-independent properties: Euclidean only (test time).
  const std::pair<index_t, index_t> shapes[] = {{20000, 23}, {1100000, 2}, {6000, 77}};
  for (auto [n, k] : shapes)
    for (Mode mode : kModes) {
      if (k != 23 && mode != Mode::Euclidean) continue;
      OrthoSetup<T> s(n, k, mode);
      std::vector<T> x(static_cast<std::size_t>(n));
      for (index_t j = 0; j < k; ++j) {
        la::fill_random(x.data(), n, 13, static_cast<std::uint64_t>(j));
        s.append(x, policy(la::Reorth::Always));
      }
      std::vector<T> w0(static_cast<std::size_t>(n));
      la::fill_random(w0.data(), n, 14, 0);
      // A large component in span(V) forces two passes (ratio < 0.2 in every mode).
      const double big = 30.0 * std::sqrt(static_cast<double>(n));
      for (index_t i = 0; i < n; ++i) w0[i] += T(big) * s.v(i, k / 2);
      struct Out {
        std::vector<T> w, bw, h;
        double nrm;
      };
      auto run = [&]() {
        Out o{w0, std::vector<T>(static_cast<std::size_t>(n)), std::vector<T>(static_cast<std::size_t>(k)), 0.0};
        if (mode == Mode::Stored) s.apply(o.w.data(), o.bw.data());
        o.nrm = s.orthogonalize(o.w.data(), mode == Mode::Euclidean ? nullptr : o.bw.data(), o.h.data(),
                                la::OrthoOptions{}, nullptr);
        return o;
      };
      Out ref;
      {
        ThreadCount tc(1);
        ref = run();
      }
      for (int threads : kThreadCounts) {  // threads == 1 repeats the reference run
        ThreadCount tc(threads);
        const Out o = run();
        EXPECT_TRUE(bitwise_equal(o.w, ref.w)) << "n=" << n << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(o.bw, ref.bw)) << "n=" << n << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(o.h, ref.h)) << "n=" << n << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(o.nrm, ref.nrm)) << "n=" << n << " " << mode_name(mode) << " threads=" << threads;
      }
    }
}

// ---- block CGS -----------------------------------------------------------------------------------

/// Runs orthogonalize_block_cgs2 on W (n x p, ld n) with the setup's mode. bw: n x p (ld n), M W
/// on entry in stored mode, workspace in recompute mode.
template <class T>
int block_orthogonalize(const OrthoSetup<T>& s, index_t p, T* w, T* bw, T* h, double* before, double* after,
                        const la::OrthoOptions& opt) {
  const index_t n = s.n;
  const la::ApplyBlockFn<T> apply = [&s](const T* x, index_t ldx, T* y, index_t ldy, index_t cols) {
    for (index_t j = 0; j < cols; ++j) s.apply(x + j * ldx, y + j * ldy);
  };
  switch (s.mode) {
    case Mode::Euclidean:
      return la::orthogonalize_block_cgs2<T>(n, s.k, p, s.v.data(), n, nullptr, 0, w, n, nullptr, 0, h, s.k, before,
                                             after, nullptr, opt);
    case Mode::Stored:
      return la::orthogonalize_block_cgs2<T>(n, s.k, p, s.v.data(), n, s.bv.data(), n, w, n, bw, n, h, s.k, before,
                                             after, nullptr, opt);
    case Mode::Recompute:
      return la::orthogonalize_block_cgs2<T>(n, s.k, p, s.v.data(), n, nullptr, 0, w, n, bw, n, h, s.k, before, after,
                                             &apply, opt);
  }
  return -1;
}

/// Fills s with s.cap random M-orthonormal columns.
template <class T>
void fill_random_basis(OrthoSetup<T>& s, std::uint64_t seed) {
  std::vector<T> x(static_cast<std::size_t>(s.n));
  while (s.k < s.cap) {
    la::fill_random(x.data(), s.n, seed, static_cast<std::uint64_t>(s.k));
    s.append(x, policy(la::Reorth::Always));
  }
}

/// W (n x p): column j = V y_j + ratio_j-weighted component outside span(V) (ratio 1: random).
template <class T>
hs::DenseMatrix<T> block_input(const OrthoSetup<T>& s, const std::vector<double>& ratios, std::uint64_t seed) {
  const index_t n = s.n, k = s.k, p = static_cast<index_t>(ratios.size());
  hs::DenseMatrix<T> w(n, p);
  std::vector<T> y(static_cast<std::size_t>(k)), x(static_cast<std::size_t>(n));
  for (index_t j = 0; j < p; ++j) {
    la::fill_random(x.data(), n, seed, static_cast<std::uint64_t>(2 * j));
    if (ratios[j] >= 1.0) {
      std::copy(x.begin(), x.end(), w.col(j));
      continue;
    }
    la::fill_random(y.data(), k, seed, static_cast<std::uint64_t>(2 * j + 1));
    // x_perp: x orthogonalized against V (M-norm xn).
    std::vector<T> bx(static_cast<std::size_t>(n));
    if (s.mode == Mode::Stored) s.apply(x.data(), bx.data());
    const double xn =
        s.orthogonalize(x.data(), s.mode == Mode::Euclidean ? nullptr : bx.data(), nullptr, policy(la::Reorth::Always), nullptr);
    la::multiply(n, k, 1, s.v.data(), n, y.data(), k, w.col(j), n);
    const double yn = la::norm2(k, y.data()), r = ratios[j];
    la::axpy(n, T(r * yn / std::sqrt(1.0 - r * r) / xn), x.data(), w.col(j));
  }
  return w;
}

template <class T>
class LinalgDenseOpsBlockCgs : public ::testing::Test {};
TYPED_TEST_SUITE(LinalgDenseOpsBlockCgs, Scalars);

TYPED_TEST(LinalgDenseOpsBlockCgs, AgreesWithColumnwiseCgsInAllModes) {
  using T = TypeParam;
  const index_t n = 3000, k = 30;
  for (Mode mode : kModes) {
    OrthoSetup<T> s(n, k, mode);
    fill_random_basis(s, 31);
    // Block A: all columns independent of span(V); block B: one column nearly in span(V).
    const std::vector<std::vector<double>> blocks = {{1.0, 1.0, 1.0}, {1.0, 0.3, 1e-8, 1.0, 0.9}};
    for (const auto& ratios : blocks)
      for (la::Reorth reorth : {la::Reorth::Always, la::Reorth::IfNeeded}) {
        const index_t p = static_cast<index_t>(ratios.size());
        const hs::DenseMatrix<T> w0 = block_input(s, ratios, 32);
        hs::DenseMatrix<T> w = w0, bw(n, p), h(k, p);
        if (mode == Mode::Stored)
          for (index_t j = 0; j < p; ++j) s.apply(w.col(j), bw.col(j));
        std::vector<double> before(static_cast<std::size_t>(p)), after(static_cast<std::size_t>(p));
        const int passes = block_orthogonalize(s, p, w.data(), bw.data(), h.data(), before.data(), after.data(),
                                               policy(reorth));
        const bool all_independent = ratios.size() == 3;
        EXPECT_EQ(passes, reorth == la::Reorth::IfNeeded && all_independent ? 1 : 2) << mode_name(mode);
        for (index_t j = 0; j < p; ++j) {
          // The same column through orthogonalize_cgs2 with the same number of passes.
          std::vector<T> wj(w0.col(j), w0.col(j) + n), bwj(static_cast<std::size_t>(n)), hj(static_cast<std::size_t>(k));
          if (mode == Mode::Stored) s.apply(wj.data(), bwj.data());
          la::OrthoInfo info;
          const double nj = s.orthogonalize(wj.data(), mode == Mode::Euclidean ? nullptr : bwj.data(), hj.data(),
                                            policy(passes == 2 ? la::Reorth::Always : la::Reorth::IfNeeded), &info);
          EXPECT_EQ(info.passes, passes);
          EXPECT_NEAR(before[j], info.norm_before, 1e-13 * info.norm_before);
          EXPECT_NEAR(after[j], nj, 1e-12 * info.norm_before) << mode_name(mode) << " col " << j;
          double dw = 0.0, dh = 0.0, wmax = 0.0;
          for (index_t i = 0; i < n; ++i) {
            dw = std::max(dw, std::abs(w(i, j) - wj[i]));
            wmax = std::max(wmax, std::abs(w0(i, j)));
          }
          for (index_t i = 0; i < k; ++i) dh = std::max(dh, std::abs(h(i, j) - hj[i]));
          EXPECT_LE(dw, 1e-14 * wmax) << mode_name(mode) << " col " << j;
          EXPECT_LE(dh, 1e-13) << mode_name(mode) << " col " << j;
          // M-orthogonality of the result and bw == M w.
          std::vector<T> mw(static_cast<std::size_t>(n));
          s.apply(w.col(j), mw.data());
          double proj = 0.0, dbw = 0.0, bwmax = 0.0;
          for (index_t c = 0; c < k; ++c) proj = std::max(proj, std::abs(la::dot(n, s.v.col(c), mw.data())));
          for (index_t i = 0; mode != Mode::Euclidean && i < n; ++i) {
            dbw = std::max(dbw, std::abs(bw(i, j) - mw[i]));
            bwmax = std::max(bwmax, 100.0 * std::abs(w0(i, j)));
          }
          EXPECT_LE(proj, 1e-14 * after[j]) << mode_name(mode) << " col " << j;
          EXPECT_LE(dbw, 1e-13 * bwmax) << mode_name(mode) << " col " << j;
        }
      }
  }
}

TYPED_TEST(LinalgDenseOpsBlockCgs, EdgeCases) {
  using T = TypeParam;
  const index_t n = 500;
  for (Mode mode : kModes) {
    OrthoSetup<T> s(n, 2, mode);
    hs::DenseMatrix<T> w(n, 2), bw(n, 2);
    la::fill_random(w.data(), 2 * n, 41, 0);
    if (mode == Mode::Stored)
      for (index_t j = 0; j < 2; ++j) s.apply(w.col(j), bw.col(j));
    const hs::DenseMatrix<T> w0 = w;
    std::vector<double> before(2), after(2);
    // k == 0: nothing removed, norms reported, M W computed in recompute mode.
    EXPECT_EQ(block_orthogonalize<T>(s, 2, w.data(), bw.data(), nullptr, before.data(), after.data(), la::OrthoOptions{}), 0);
    for (index_t j = 0; j < 2; ++j) {
      EXPECT_TRUE(std::equal(w.col(j), w.col(j) + n, w0.col(j)));
      EXPECT_EQ(before[j], after[j]);
      double m2 = 0.0;
      for (index_t i = 0; i < n; ++i) m2 += s.m[i] * hs::abs2(w(i, j));
      EXPECT_NEAR(after[j], std::sqrt(m2), 1e-13 * std::sqrt(m2));
      for (index_t i = 0; mode != Mode::Euclidean && i < n; ++i) EXPECT_EQ(bw(i, j), s.m[i] * w(i, j));
    }
    // p == 0: no work.
    s.append(std::vector<T>(w0.col(0), w0.col(0) + n), la::OrthoOptions{});
    EXPECT_EQ(block_orthogonalize<T>(s, 0, w.data(), bw.data(), nullptr, nullptr, nullptr, la::OrthoOptions{}), 0);
    // Invalid eta.
    EXPECT_THROW(block_orthogonalize<T>(s, 2, w.data(), bw.data(), nullptr, nullptr, nullptr,
                                     policy(la::Reorth::IfNeeded, -0.1)),
                 hs::Error);
  }
}

TYPED_TEST(LinalgDenseOpsBlockCgs, BitwiseIdenticalForAnyThreadCount) {
  using T = TypeParam;
  // (n, k, p): one block per reduction slot (all modes); several blocks per slot, as wide k x p
  // partials get fewer slots (Euclidean only, test time).
  const std::array<index_t, 3> shapes[] = {{20000, 23, 4}, {30000, 64, 48}};
  for (const auto& [n, k, p] : shapes)
    for (Mode mode : kModes) {
      if (k != 23 && mode != Mode::Euclidean) continue;
      OrthoSetup<T> s(n, k, mode);
      fill_random_basis(s, 51);
      std::vector<double> ratios(static_cast<std::size_t>(p), 1.0);
      ratios[1] = 0.1;  // two passes
      const hs::DenseMatrix<T> w0 = block_input(s, ratios, 52);
      struct Out {
        hs::DenseMatrix<T> w, bw, h;
        std::vector<double> after;
      };
      auto run = [&]() {
        Out o{w0, hs::DenseMatrix<T>(n, p), hs::DenseMatrix<T>(k, p), std::vector<double>(static_cast<std::size_t>(p))};
        if (mode == Mode::Stored)
          for (index_t j = 0; j < p; ++j) s.apply(o.w.col(j), o.bw.col(j));
        EXPECT_EQ(block_orthogonalize<T>(s, p, o.w.data(), o.bw.data(), o.h.data(), nullptr, o.after.data(),
                                      la::OrthoOptions{}),
                  2);
        return o;
      };
      auto same = [](const hs::DenseMatrix<T>& a, const hs::DenseMatrix<T>& b) {
        return a.rows() == b.rows() && a.cols() == b.cols() &&
               std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.rows() * a.cols()) * sizeof(T)) == 0;
      };
      Out ref;
      {
        ThreadCount tc(1);
        ref = run();
      }
      for (int threads : kThreadCounts) {
        ThreadCount tc(threads);
        const Out o = run();
        EXPECT_TRUE(same(o.w, ref.w)) << "n=" << n << " k=" << k << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(same(o.bw, ref.bw)) << "n=" << n << " k=" << k << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(same(o.h, ref.h)) << "n=" << n << " k=" << k << " " << mode_name(mode) << " threads=" << threads;
        EXPECT_TRUE(bitwise_equal(o.after, ref.after)) << "n=" << n << " " << mode_name(mode) << " threads=" << threads;
      }
    }
}

}  // namespace
