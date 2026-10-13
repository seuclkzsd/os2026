// factorize_near's choice between a perturbed shift and a factorization at the shift without
// null-pivot detection (module anaint, P5c fix; src/driver/steps.cpp). Regression: on
// syn-zero-cluster-real-1e5 (sigma0 = 0 inside 10^4 eigenvalues spaced 2e-9) the factors of the
// nested dissections flagged 3 null pivots at sigma0 and none at the perturbed shift 1e-8, AMD's
// 1 and 1: the nested dissections ran at 1e-8, half the wanted window's radius away, and the
// certificate counted 34 eigenvalues for the 21 found - a rerun on every seed. The decision is
// made here on a scripted factorization, so that it does not depend on MUMPS' pivots.
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"

namespace {

using hs::cplx;
using hs::index_t;

/// What factorize(1, -s) does at a shift s: with detection on, `nulls` null pivots (0: success);
/// with detection off, success unless `zero_pivot`.
struct Outcome {
  index_t nulls = 0;
  bool zero_pivot = false;
};

/// A SparseFactorization that plays a script: outcomes by shift, every call recorded.
template <class T>
class ScriptedFactorization final : public hs::SparseFactorization<T> {
 public:
  explicit ScriptedFactorization(std::map<double, Outcome> script) : script_(std::move(script)) {}
  void analyze(const hs::ShiftedPencil<T>&, double, double) override {}
  hs::FactorEstimate estimate() const override { return {}; }
  bool factorize(double alpha, double beta) override {
    const double s = -beta / alpha;
    calls.emplace_back(s, detection_);
    const auto it = script_.find(s);
    const Outcome o = it == script_.end() ? Outcome{} : it->second;
    ++stats_.factorizations;
    stats_.null_pivots = detection_ ? o.nulls : 0;
    return detection_ ? o.nulls == 0 : !o.zero_pivot;
  }
  void solve(T*, index_t, index_t) override {}
  hs::Inertia inertia() const override { return {}; }
  void set_out_of_core(bool) override {}
  void set_null_pivot_detection(bool on) override { detection_ = on; }
  const hs::FactorStats& stats() const override { return stats_; }
  const hs::FactorOptions& options() const override { return options_; }
  index_t order() const override { return 10; }

  std::vector<std::pair<double, bool>> calls;  ///< (shift, detection on)

 private:
  std::map<double, Outcome> script_;
  bool detection_ = true;
  hs::FactorStats stats_;
  hs::FactorOptions options_;
};

struct Result {
  double sigma = 0.0;
  bool detection_off = false;
  index_t shift_retries = 0;
  std::vector<std::pair<double, bool>> calls;
};

constexpr double kSigma = 0.0;
constexpr double kScale = 1.0;
const double kPerturbed = kSigma + 1e-8 * kScale;

template <class T>
Result run(std::map<double, Outcome> script, bool on_eigenvalue = false, index_t max_shift_retries = 5) {
  hs::SolverConfig cfg;
  cfg.max_shift_retries = max_shift_retries;
  hs::SolveReport rep;
  hs::driver::PhaseClock clock;
  hs::driver::Ctx c{cfg, nullptr, rep, clock};
  ScriptedFactorization<T> fac(std::move(script));
  Result r;
  r.sigma = hs::driver::factorize_near(c, fac, kSigma, kScale, r.detection_off, on_eigenvalue);
  r.shift_retries = rep.shift_retries;
  r.calls = fac.calls;
  return r;
}

TEST(AnaintFixShift, RegularShiftIsFactorizedOnce) {
  const Result r = run<double>({});
  EXPECT_EQ(r.sigma, kSigma);
  EXPECT_FALSE(r.detection_off);
  EXPECT_EQ(r.calls.size(), 1u);
  EXPECT_EQ(r.shift_retries, 0);
}

// One eigenvalue at sigma0 (one null pivot; two in the real embedding of a complex pencil): the
// perturbed shift, as before.
TEST(AnaintFixShift, IsolatedEigenvalueTakesThePerturbedShift) {
  for (const bool complex : {false, true}) {
    const std::map<double, Outcome> script{{kSigma, {complex ? 2 : 1, false}}};
    const Result r = complex ? run<cplx>(script) : run<double>(script);
    EXPECT_EQ(r.sigma, kPerturbed) << complex;
    EXPECT_FALSE(r.detection_off) << complex;
    EXPECT_EQ(r.shift_retries, 1) << complex;
    ASSERT_EQ(r.calls.size(), 2u) << complex;
    EXPECT_EQ(r.calls[1], std::make_pair(kPerturbed, true)) << complex;
  }
}

// Several eigenvalues' worth of null pivots at sigma0 (the nested dissections' factors on
// syn-zero-cluster-real-1e5): factorized at sigma0 without detection even though the perturbed
// shift would not be flagged - the perturbation is not tried.
TEST(AnaintFixShift, ClusterOfNullPivotsStaysAtTheShift) {
  for (const bool complex : {false, true}) {
    const std::map<double, Outcome> script{{kSigma, {3, false}}};  // 3: two eigenvalues in the embedding
    const Result r = complex ? run<cplx>(script) : run<double>(script);
    EXPECT_EQ(r.sigma, kSigma) << complex;
    EXPECT_TRUE(r.detection_off) << complex;
    EXPECT_EQ(r.shift_retries, 2) << complex;
    ASSERT_EQ(r.calls.size(), 2u) << complex;
    EXPECT_EQ(r.calls[1], std::make_pair(kSigma, false)) << complex;
  }
}

// One null pivot at sigma0, but the perturbed shift flagged too (AMD's factors there): a cluster
// as well, as before.
TEST(AnaintFixShift, PerturbedShiftSingularToo) {
  const Result r = run<double>({{kSigma, {1, false}}, {kPerturbed, {2, false}}});
  EXPECT_EQ(r.sigma, kSigma);
  EXPECT_TRUE(r.detection_off);
  ASSERT_EQ(r.calls.size(), 3u);
  EXPECT_EQ(r.calls[1], std::make_pair(kPerturbed, true));
  EXPECT_EQ(r.calls[2], std::make_pair(kSigma, false));
}

// A cluster whose factorization without detection meets an exactly zero pivot: the perturbed
// shift after all; nothing works -> Singular.
TEST(AnaintFixShift, ExactZeroPivotFallsBackToThePerturbedShift) {
  Result r = run<double>({{kSigma, {4, true}}});
  EXPECT_EQ(r.sigma, kPerturbed);
  EXPECT_FALSE(r.detection_off);
  ASSERT_EQ(r.calls.size(), 3u);
  EXPECT_EQ(r.calls[1], std::make_pair(kSigma, false));
  EXPECT_EQ(r.calls[2], std::make_pair(kPerturbed, true));
  try {
    run<double>({{kSigma, {4, true}}, {kPerturbed, {1, false}}});
    ADD_FAILURE() << "expected Singular";
  } catch (const hs::Error& e) {
    EXPECT_EQ(e.code(), hs::Error::Code::Singular);
  }
}

// A shift placed on a (multiple) eigenvalue on purpose keeps the perturbation, whatever the null
// pivots; without shift retries a cluster is factorized at sigma0 right away.
TEST(AnaintFixShift, OnEigenvalueAndNoRetries) {
  const Result on = run<double>({{kSigma, {5, false}}}, /*on_eigenvalue=*/true);
  EXPECT_EQ(on.sigma, kPerturbed);
  EXPECT_EQ(on.shift_retries, 0);  // expected, not reported
  const Result none = run<double>({{kSigma, {1, false}}}, false, /*max_shift_retries=*/0);
  EXPECT_EQ(none.sigma, kSigma);
  EXPECT_TRUE(none.detection_off);
  EXPECT_EQ(none.calls.size(), 2u);
  EXPECT_EQ(none.shift_retries, 1);
}

TEST(AnaintFixShift, NullPivotsPerEigenvalue) {
  using hs::driver::near_null_eigenvalues;
  EXPECT_EQ(near_null_eigenvalues(1, false), 1);
  EXPECT_EQ(near_null_eigenvalues(2, false), 2);
  EXPECT_EQ(near_null_eigenvalues(1, true), 1);
  EXPECT_EQ(near_null_eigenvalues(2, true), 1);
  EXPECT_EQ(near_null_eigenvalues(3, true), 2);
}

}  // namespace
