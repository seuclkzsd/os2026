// The nearest target's fast completeness checks: the auto choice, the randomized exclusion check
// and single-precision counts (certificate.hpp; the exact counts stay in definite.cpp).
#include "driver/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "driver/projection.hpp"
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/util.hpp"

namespace hs::driver {

namespace {

/// Converged found pair (the deflation criterion of definite.cpp's reruns, keep_converged).
constexpr double kConvergedFactor = 10.0;
/// Relative tolerance of "copies of one multiple eigenvalue" for the fault injection (as in
/// bench/certfast_measure.cpp).
constexpr double kCopyTol = 1e-8;
/// Ratio of the single-precision to the double count of an InertiaCounter: 3-D fill (factor
/// entries / input entries > 20, the adaptive ordering's 3-D criterion, steps.cpp) measured
/// 0.54-0.61 (verify.hpp, factor.hpp InertiaCounter), low fill ~1.
constexpr double kSingleRatio3d = 0.55;
constexpr double kSingleRatioLowFill = 1.0;
constexpr double kThreeDFill = 20.0;
/// Share of the memory left in the budget that a randomized check may plan for.
constexpr double kCheckMemoryShare = 0.9;

std::string sec(double s) {
  std::ostringstream os;
  os.precision(3);
  os << s << " s";
  return os.str();
}

std::string predicted(double s) { return std::isfinite(s) ? sec(s) : std::string("n/a"); }

}  // namespace

template <class T>
NearestCertifier<T>::NearestCertifier(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac,
                                      double sigma, bool detection_off, double scale, double n2a, double n2b,
                                      const CertificateTimings& timings)
    : c_(c),
      pencil_(pencil),
      fac_(fac),
      sigma_(sigma),
      detection_off_(detection_off),
      scale_(scale),
      n2a_(n2a),
      n2b_(n2b),
      timings_(timings) {}

template <class T>
NearestCertifier<T>::~NearestCertifier() = default;

// The pencil's ||A||_1 and min b_ii, computed once (O(nnz)), and only when the single-count rule
// is evaluated at all.
template <class T>
const SingleCountRule& NearestCertifier<T>::single_rule() {
  if (rule_.a_norm_1 < 0.0) {
    rule_.a_norm_1 = norm_one(pencil_.a());
    // ||r||_{B^{-1}} <= ||r||_2 / sqrt(min b_ii) for a diagonal B (B-normalized x); B = I: ||r||_2.
    // For other B the bound only feeds the rule, which then does not apply anyway.
    if (const CsrMatrix<T>* b = pencil_.b()) {
      double bmin = std::numeric_limits<double>::infinity();
      for (index_t i = 0; i < b->nrows; ++i)
        for (index_t q = b->rowptr[i]; q < b->rowptr[i + 1]; ++q)
          if (b->colind[q] == i) bmin = std::min(bmin, std::real(b->values[q]));
      inv_sqrt_bmin_ = bmin > 0.0 ? 1.0 / std::sqrt(bmin) : std::numeric_limits<double>::infinity();
    }
  }
  return rule_;
}

template <class T>
SingleWindow NearestCertifier<T>::single_window(const NearestWindow& w, const Block<T>& all) {
  (void)single_rule();  // inv_sqrt_bmin_
  SingleWindow sw;
  sw.lower = w.lower;
  sw.upper = w.upper;
  sw.wanted_lower = w.target - w.radius;
  sw.wanted_upper = w.target + w.radius;
  const index_t n = all.x.rows();
  for (index_t j = 0; j < all.size(); ++j) {
    const double l = all.lambda[j];
    if (!std::isfinite(l) || l < w.lower || l > w.upper) continue;
    const PairMetrics pm = pair_metrics(all.ax.col(j), all.bx.col(j), all.x.col(j), n, l, n2a_, n2b_);
    sw.found.push_back(l);
    sw.found_error.push_back(pm.abs_residual * inv_sqrt_bmin_);
  }
  return sw;
}

template <class T>
bool NearestCertifier<T>::converged(const Block<T>& blk, index_t j) const {
  const PairMetrics pm =
      pair_metrics(blk.ax.col(j), blk.bx.col(j), blk.x.col(j), blk.x.rows(), blk.lambda[j], n2a_, n2b_);
  return std::isfinite(blk.lambda[j]) && pm.backward_error <= kConvergedFactor * c_.cfg.tol;
}

template <class T>
std::vector<const T*> NearestCertifier<T>::deflation_columns(const NearestWindow& w,
                                                             const RoundPairs<T>& pairs) const {
  std::vector<const T*> cols;
  const auto in_window = [&](double l) { return std::isfinite(l) && l >= w.lower && l <= w.upper; };
  std::vector<char> taken(static_cast<std::size_t>(pairs.all.size()), 0);
  if (pairs.selected) {
    for (index_t j : pairs.selected_columns) taken[static_cast<std::size_t>(j)] = 1;
    for (index_t j = 0; j < pairs.selected->size(); ++j)
      if (in_window(pairs.selected->lambda[j]) && converged(*pairs.selected, j))
        cols.push_back(pairs.selected->x.col(j));
  }
  for (index_t j = 0; j < pairs.all.size(); ++j)
    if (!taken[static_cast<std::size_t>(j)] && in_window(pairs.all.lambda[j]) && converged(pairs.all, j))
      cols.push_back(pairs.all.x.col(j));
  return cols;
}

template <class T>
DenseMatrix<T> NearestCertifier<T>::deflation_block(const NearestWindow& w, const RoundPairs<T>& pairs) const {
  const std::vector<const T*> cols = deflation_columns(w, pairs);
  const index_t n = pairs.all.x.rows(), m = static_cast<index_t>(cols.size());
  DenseMatrix<T> x(n, m);
  for (index_t j = 0; j < m; ++j) std::copy(cols[j], cols[j] + n, x.col(j));
  return x;
}

// The randomized check needs kappa, and only a proven one certifies: the Lanczos estimate of a
// general B's kappa (~0.1 s at n = 1e5) is computed only for a run() that asks for a screening
// result (the solve path never runs the check without a proven kappa).
template <class T>
const StartScaling& NearestCertifier<T>::scaling(bool estimate) {
  if (!proven_scaling_)
    proven_scaling_ = std::make_unique<StartScaling>(start_scaling_for<T>(pencil_.b(), 100.0, false));
  if (proven_scaling_->rigorous || !estimate) return *proven_scaling_;
  if (!estimated_scaling_)
    estimated_scaling_ = std::make_unique<StartScaling>(start_scaling_for<T>(pencil_.b(), 100.0, true));
  return *estimated_scaling_;
}

template <class T>
double NearestCertifier<T>::check_budget_bytes() const {
  const double avail = budget_left(c_);
  // ExclusionOptions: 0 means "no limit", so an exhausted budget is one byte, not zero.
  return std::max(1.0, kCheckMemoryShare * avail);
}

template <class T>
std::string NearestCertifier<T>::prefix() const {
  return auto_note_.empty() ? std::string() : auto_note_ + "; ";
}

// Radii of the randomized claim (certificate.hpp): r_edge holds every wanted eigenvalue, r_pass is
// the window's half-width on its nearer side, r_claim lies between at kClaimFraction of the theta
// gap from r_edge. eps is that of exclusion_check for a negligible coupling.
namespace {
struct Radii {
  double edge = 0.0, pass = 0.0, claim = 0.0, epsilon = 0.0;
  bool valid = false;
};
Radii randomized_radii(const NearestWindow& w, double sigma) {
  Radii r;
  r.edge = w.radius + std::abs(sigma - w.target);
  r.pass = std::min(sigma - w.lower, w.upper - sigma);
  if (!(r.edge > 0.0) || !(r.pass > r.edge) || !std::isfinite(r.pass)) return r;
  const double th_edge = 1.0 / r.edge, tau = 1.0 / r.pass;
  const double th_claim = th_edge - kClaimFraction * (th_edge - tau);
  r.claim = 1.0 / th_claim;
  const double th_eff = th_claim * (1.0 - ExclusionOptions{}.rounding_margin);
  r.epsilon = 1.0 - tau / th_eff;
  r.valid = r.claim > r.edge && r.pass > r.claim && r.epsilon > 0.0;
  return r;
}
}  // namespace

template <class T>
CertificateChoice NearestCertifier<T>::choose(const NearestWindow& w, const RoundPairs<T>& pairs) {
  const SolverConfig& cfg = c_.cfg;
  CertificateChoice ch;
  const index_t n = pencil_.order();
  const MumpsInputSize in_size = mumps_input_size(pencil_);
  const FactorEstimate est = fac_.estimate();
  const int threads = fac_.options().threads;

  CertificateCostInputs in;
  in.n = n;
  in.deflated = std::min<index_t>(static_cast<index_t>(deflation_columns(w, pairs).size()), n - 1);
  in.complex_scalars = is_complex_v<T>;
  in.factorize_s = timings_.factorize_s > 0.0 ? timings_.factorize_s : model_factor_s(est, in_size.order, threads);
  const double analysis = model_analysis_s(in_size.order, in_size.entries, static_cast<double>(est.factor_entries));
  // The exact counts run on this certifier's counter once single counts built it (exact_counts);
  // its double instance is analysed at the first double count (deferred, see single()).
  in.counter_analysis_s = counter_ && counter_->double_analysed() ? 0.0 : analysis;
  // Single counts analyse the counter's single instance at the first one (a counter of an
  // earlier round has done it); the double instance only where a double check follows.
  in.single_analysis_s = counter_ && counter_->stats().single_counts > 0 ? 0.0 : analysis;
  in.single_ratio = static_cast<double>(est.factor_entries) > kThreeDFill * in_size.entries ? kSingleRatio3d
                                                                                             : kSingleRatioLowFill;
  // Measured solve calls of the Krylov run; t_apply(q) = apply1 (1 + rhs_ratio (q - 1)).
  const double q = std::max(1.0, timings_.rhs_per_call);
  const double call = timings_.solve_call_s > 0.0 ? timings_.solve_call_s
                                                  : model_solve_s(est) * (1.0 + in.rhs_ratio * (q - 1.0));
  in.apply1_s = call / (1.0 + in.rhs_ratio * (q - 1.0));
  in.delta = cfg.certificate_delta;
  in.max_memory_bytes = check_budget_bytes();
  // Randomized: never after an undecided check; auto takes it only with a proven kappa, and asks
  // for kappa (start_scaling_for: O(nnz(B)) for a general B) only when the check, at the smallest
  // possible kappa = 1, would beat the exact counts.
  const Radii rad = randomized_radii(w, sigma_);
  ch.epsilon = rad.valid ? rad.epsilon : 0.0;
  in.epsilon = rad.valid && !randomized_failed_ ? rad.epsilon : 0.0;
  in.kappa = 1.0;
  if (!rad.valid) {
    ch.randomized_reason = "the window leaves no gap beyond the wanted radius";
  } else if (randomized_failed_) {
    ch.randomized_reason = "could not decide in an earlier round";
  } else if (cfg.certificate == CertificateMode::Auto && pencil_.b()) {
    in.single_valid = false;
    const CertificateCostEstimate optimistic = certificate_cost(in);
    if (!(optimistic.randomized_s * in.preference < optimistic.exact_s)) {
      in.epsilon = 0.0;
      ch.randomized_reason = "predicted no cheaper than the exact counts";
    }
  }
  if (in.epsilon > 0.0) {
    const StartScaling& sc = scaling(cfg.certificate == CertificateMode::Randomized);
    ch.kappa_rigorous = sc.rigorous;
    in.kappa = std::isfinite(sc.kappa) && sc.kappa >= 1.0 ? sc.kappa : 1.0;
    if (cfg.certificate != CertificateMode::Randomized && !sc.rigorous) {
      in.epsilon = 0.0;
      ch.randomized_reason = "no proven kappa of B (general B): not a certificate";
    }
  }
  // The single-count rule (O(nnz(A)) and the residuals of the window's pairs) is evaluated only
  // when single counts would be cheaper than the exact counts if it held - not on low-fill inputs,
  // whose single counts save nothing (kSingleRatioLowFill).
  in.single_valid = !single_failed_;
  if (single_failed_) {
    ch.single_reason = "single counts could not decide in an earlier round";
  } else if (const CertificateCostEstimate optimistic = certificate_cost(in);
             !(optimistic.exact_single_s < optimistic.exact_s)) {
    in.single_valid = false;
    ch.single_reason = "predicted no cheaper than the exact counts";
  } else {
    const SingleWindow sw = single_window(w, pairs.all);  // wanted region: the selected pairs' disc
    if (sw.lower < sw.wanted_lower && sw.wanted_upper < sw.upper) {
      // auto demands the safety margin over the rule's assumed backward error (kAutoSingleSafety)
      SingleCountRule rule = single_rule();
      if (cfg.certificate == CertificateMode::Auto) rule.backward_factor *= kAutoSingleSafety;
      const SingleCountValidity v = single_count_validity(pencil_, sw, rule);
      in.single_valid = v.valid();
      ch.single_reason = v.reason;
    } else {
      in.single_valid = false;
      ch.single_reason = "window does not strictly contain the wanted region";
    }
  }
  ch.inputs = in;
  ch.estimate = certificate_cost(in);
  ch.method = ch.estimate.recommended;
  return ch;
}

template <class T>
CheckOutcome NearestCertifier<T>::check(const NearestWindow& w, const RoundPairs<T>& pairs, int round) {
  const SolverConfig& cfg = c_.cfg;
  if (cfg.certificate == CertificateMode::Exact || cfg.certificate == CertificateMode::Off) return {};
  PhaseScope phase(c_.clock, Phase::Certificate);
  fallback_note_.clear();
  if (cfg.certificate == CertificateMode::Randomized && pencil_.b() && !scaling(false).rigorous) {
    // A general B: kappa can only be estimated, so a pass would be a SCREENING result, not a
    // certificate - and the check costs more than the counts it cannot replace. Not run (the
    // note goes into every round's statement).
    fallback_note_ = "randomized check not run: the kappa of B is not proven (" + scaling(false).note +
                     "), so a pass would be a screening result, not a certificate";
    if (!randomized_failed_)
      warn(c_, "certificate=randomized: the kappa of B is not proven (B neither diagonal nor diagonally "
               "dominant); the exact counts certify instead of a screening-only randomized check");
    randomized_failed_ = true;
    return {};
  }
  const Timer choice_timer;
  const CertificateChoice ch = choose(w, pairs);
  const double choice_s = choice_timer.seconds();
  CertificateMethod method = ch.method;
  if (cfg.certificate == CertificateMode::Randomized)
    method = randomized_failed_ ? CertificateMethod::Exact : CertificateMethod::Randomized;
  const CertificateCostEstimate& e = ch.estimate;
  if (cfg.certificate == CertificateMode::Auto) {
    std::ostringstream os;
    os.precision(3);
    os << "auto (round " << round << "): predicted exact counts " << predicted(e.exact_s) << ", single counts "
       << predicted(e.exact_single_s);
    if (!std::isfinite(e.exact_single_s)) os << " [" << ch.single_reason << "]";
    os << ", randomized check " << predicted(e.randomized_s);
    if (e.block_size > 0)
      os << " [p = " << e.block_size << ", s = " << e.block_steps << ", eps = " << ch.epsilon << "]";
    else if (!ch.randomized_reason.empty())
      os << " [" << ch.randomized_reason << "]";
    else
      os << " [no plan within the resource budget]";
    os << "; chose " << certificate_method_name(method);
    auto_note_ = os.str();
  }
  // The prediction and every input of certificate_cost (bench/certint_model.cpp re-evaluates it).
  const CertificateCostInputs& in = ch.inputs;
  event(c_, "certificate_choice",
        {{"round", static_cast<std::int64_t>(round)}, {"method", std::string(certificate_method_name(method))},
         {"exact_s", e.exact_s}, {"single_s", std::isfinite(e.exact_single_s) ? e.exact_single_s : -1.0},
         {"randomized_s", std::isfinite(e.randomized_s) ? e.randomized_s : -1.0},
         {"block_size", static_cast<std::int64_t>(e.block_size)},
         {"block_steps", static_cast<std::int64_t>(e.block_steps)}, {"epsilon", ch.epsilon},
         {"kappa_rigorous", ch.kappa_rigorous}, {"factorize_s", timings_.factorize_s},
         {"solve_call_s", timings_.solve_call_s}, {"rhs_per_call", timings_.rhs_per_call},
         {"in_n", static_cast<std::int64_t>(in.n)}, {"in_deflated", static_cast<std::int64_t>(in.deflated)},
         {"in_complex", in.complex_scalars}, {"in_factorize_s", in.factorize_s},
         {"in_counter_analysis_s", in.counter_analysis_s}, {"in_single_analysis_s", in.single_analysis_s},
         {"in_count_ratio", in.count_ratio}, {"in_single_ratio", in.single_ratio},
         {"in_single_valid", in.single_valid}, {"in_apply1_s", in.apply1_s}, {"in_rhs_ratio", in.rhs_ratio},
         {"in_stream_bytes_per_s", in.stream_bytes_per_s}, {"in_epsilon", in.epsilon}, {"in_delta", in.delta},
         {"in_kappa", in.kappa}, {"in_max_krylov_vectors", static_cast<std::int64_t>(in.max_krylov_vectors)},
         {"in_max_memory_bytes", in.max_memory_bytes}, {"in_preference", in.preference},
         {"choice_s", choice_s}});
  if (method == CertificateMethod::Randomized) {
    const CheckOutcome out = randomized(w, pairs, e.block_size, round);
    if (out.verdict == CheckOutcome::Verdict::Passed && !c_.rep.certificate.rigorous) {
      // Not reached with a proven kappa (above); a screening pass never becomes the verdict.
      randomized_failed_ = true;
      fallback_note_ = c_.rep.certificate.statement.substr(prefix().size());
      return {};
    }
    if (out.verdict != CheckOutcome::Verdict::Undecided) return out;
    // auto: the next cheapest method that certifies
    if (cfg.certificate == CertificateMode::Auto && e.exact_single_s < e.exact_s) method = CertificateMethod::ExactSingle;
  }
  if (method == CertificateMethod::ExactSingle) return single(w, pairs.all, kAutoSingleSafety);
  return {};
}

template <class T>
CheckOutcome NearestCertifier<T>::run(CertificateMethod method, const NearestWindow& w,
                                      const RoundPairs<T>& pairs, int round) {
  PhaseScope phase(c_.clock, Phase::Certificate);
  switch (method) {
    case CertificateMethod::Randomized: return randomized(w, pairs, choose(w, pairs).estimate.block_size, round);
    case CertificateMethod::ExactSingle: return single(w, pairs.all, 1.0);
    case CertificateMethod::Exact: break;
  }
  return {};
}

// Undecided results mark the method as failed: it is not tried again in a later round (its
// window differs, but whatever stopped it - a narrow gap, a large coupling, a miscount - is
// likely to persist; the exact counts decide instead).
template <class T>
CheckOutcome NearestCertifier<T>::randomized(const NearestWindow& w, const RoundPairs<T>& pairs,
                                             index_t block_size, int round) {
  const SolverConfig& cfg = c_.cfg;
  InertiaCertificate& cert = c_.rep.certificate;
  const Radii rad = randomized_radii(w, sigma_);
  randomized_failed_ = true;  // reset below on a verdict
  if (!rad.valid || block_size < 1) {
    std::ostringstream os;
    os.precision(6);
    if (!rad.valid)
      os << "randomized check impossible: the window [" << w.lower << ", " << w.upper
         << "] leaves no gap beyond the wanted radius " << w.radius << " around sigma = " << sigma_;
    else
      os << "randomized check: no plan within the resource budget (eps = " << rad.epsilon
         << ", delta = " << cfg.certificate_delta << ", " << check_budget_bytes() << " bytes)";
    fallback_note_ = os.str();
    return {};
  }
  const DenseMatrix<T> x = deflation_block(w, pairs);
  const index_t m = x.cols();
  refactorize_at(c_, fac_, sigma_, detection_off_);  // no-op unless counts moved the main object
  DriverShiftInvert<T> op(cfg, fac_, pencil_, sigma_);  // inexact factors: refined and checked
  std::unique_ptr<CsrOperator<T>> b_op;
  if (pencil_.b()) b_op = std::make_unique<CsrOperator<T>>(*pencil_.b());
  ExclusionOptions o;
  o.pass_radius = rad.pass;
  o.delta = cfg.certificate_delta;
  o.block_size = block_size;
  o.seed = cfg.seed + 7919u * static_cast<std::uint64_t>(round + 1);  // a fresh start block per round
  set_start_scaling(o, scaling(true));
  o.max_memory_bytes = check_budget_bytes();
  o.tol = cfg.tol * cfg.inner_tol_factor;
  ExclusionResult<T> res;
  try {
    res = exclusion_check<T>(op, b_op.get(), x, sigma_, rad.claim, o);
  } catch (const Error& err) {
    // Rejected input (e.g. a rank-deficient X) or memory: the exact counts decide instead.
    if (err.code() != Error::Code::InvalidInput && err.code() != Error::Code::OutOfMemoryBudget) throw;
    fallback_note_ = std::string("randomized check could not run (") + to_string(err.code()) + "): " + err.what();
    return {};
  }
  cert.solves += res.op_applications;
  event(c_, "randomized_check",
        {{"round", static_cast<std::int64_t>(round)}, {"status", std::string(exclusion_status_name(res.status))},
         {"claim_radius", rad.claim}, {"pass_radius", rad.pass}, {"epsilon", res.epsilon},
         {"coupling", res.coupling}, {"deflated", static_cast<std::int64_t>(m)},
         {"block_size", static_cast<std::int64_t>(res.block_size)},
         {"block_steps", static_cast<std::int64_t>(res.block_steps)},
         {"op_applications", static_cast<std::int64_t>(res.op_applications)},
         {"failure_probability", res.failure_probability}, {"rigorous", res.rigorous},
         {"found_inside", static_cast<std::int64_t>(res.found_inside)},
         {"certified_inside", static_cast<std::int64_t>(res.certified_inside)},
         {"new_count", static_cast<std::int64_t>(res.new_count)}, {"setup_s", res.setup_s},
         {"krylov_s", res.krylov_s}});
  const bool decisive = res.found_inside == res.certified_inside;
  if (res.status == ExclusionStatus::Inconclusive ||
      (res.status == ExclusionStatus::Passed && !decisive)) {
    fallback_note_ = res.status == ExclusionStatus::Passed
                         ? "randomized check passed, but " + std::to_string(res.found_inside - res.certified_inside) +
                               " found eigenvalue(s) lie within the coupling e = " + fmt(res.coupling) +
                               " of the claim radius: no statement about them"
                         : "randomized check inconclusive: " + res.statement;
    return {};
  }
  randomized_failed_ = false;
  cert.performed = true;
  cert.method = certificate_method_name(CertificateMethod::Randomized);
  cert.lower = sigma_ - rad.claim;
  cert.upper = sigma_ + rad.claim;
  cert.count_inertia = -1;
  cert.count_found = res.found_inside;
  if (res.status == ExclusionStatus::Detected) {
    cert.passed = false;
    cert.rigorous = false;
    cert.failure_probability = 1.0;
    cert.statement = prefix() + "randomized check: " + res.statement;
    fallback_note_ = "randomized check: " + res.statement;  // in case the counts decide after all
    CheckOutcome out;
    out.verdict = CheckOutcome::Verdict::Missing;
    out.missing = std::max<index_t>(1, res.new_count);
    return out;
  }
  cert.passed = true;
  cert.rigorous = res.rigorous;
  cert.failure_probability = res.failure_probability;
  cert.statement = prefix() + "randomized check: " + res.statement + "; " + std::to_string(res.found_inside) +
                   " found pairs lie within that radius, so the selected pairs are the nearest ones" +
                   (res.rigorous ? std::string(" with this probability") : std::string(" (SCREENING only)"));
  if (!res.rigorous)
    warn(c_, "certificate: the randomized check passed as a SCREENING result, not a certificate (kappa of B is "
             "only estimated: " +
                 scaling(true).note + ")");
  return {CheckOutcome::Verdict::Passed, 0};
}

template <class T>
CheckOutcome NearestCertifier<T>::single(const NearestWindow& w, const Block<T>& all, double safety) {
  InertiaCertificate& cert = c_.rep.certificate;
  single_failed_ = true;  // reset below on a verdict
  const SingleWindow sw = single_window(w, all);
  if (!(sw.lower < sw.wanted_lower && sw.wanted_upper < sw.upper)) {
    fallback_note_ = "single counts: the window does not strictly contain the wanted region";
    return {};
  }
  SingleCountRule rule = single_rule();
  rule.backward_factor *= safety;
  const SingleCountValidity v = single_count_validity(pencil_, sw, rule);
  if (!v.valid()) {
    fallback_note_ = "single counts do not certify this window: " + v.reason;
    return {};
  }
  // nu(sigma) of the main factorization, an exact count when it is a double, full-rank one with
  // null-pivot detection, taken while the object still holds K(sigma) (per-side check below).
  if (!nu_sigma_taken_ && c_.held.shift == sigma_ && fac_.inertia().valid) {
    nu_sigma_taken_ = true;
    const bool exact = fac_.options().precision == FactorPrecision::Double && !fac_.options().blr;
    if (exact && !detection_off_ && !c_.held.detection_off) nu_sigma_ = fac_.inertia().neg;
  }
  // As the exact counts on a counter (MainFactors::Release): the main factors are needed again
  // only for a rerun, which refactorizes; the counts never hold both in memory.
  if (fac_.inertia().valid && fac_.release_factors()) c_.held = FactorHeld{};
  if (!counter_) {
    // The double instance only for a check of the single counts (exact_counts): deferred.
    counter_ = std::make_unique<InertiaCounter<T>>(fac_, fac_.options(), /*defer_double_analysis=*/true);
    event(c_, "inertia_counter", {{"analyze_s", counter_->stats().analyze_s}, {"double_deferred", true},
                                  {"owner", std::string("certifier")}});
  }
  SingleCertificate s = single_precision_certificate(*counter_, pencil_, sw, rule);
  const auto found = static_cast<index_t>(sw.found.size());
  const bool counted = s.lower.valid && s.upper.valid;
  if (counted && (single_fault_lower_ != 0 || single_fault_upper_ != 0)) {  // fault injection (tests)
    s.lower.neg += single_fault_lower_;
    s.upper.neg += single_fault_upper_;
    s.count = s.upper.neg - s.lower.neg;
    s.certified = s.count == found;
  }
  // Per-side check against nu(sigma): nu(sigma) - nu_s(a) found pairs below sigma, nu_s(b) -
  // nu(sigma) at or above it.
  index_t found_below = 0;
  for (double l : sw.found) found_below += l < sigma_ ? 1 : 0;
  const bool sides_ok = !counted || nu_sigma_ < 0 ||
                        (nu_sigma_ - s.lower.neg == found_below && s.upper.neg - nu_sigma_ == found - found_below);
  event(c_, "single_counts", {{"lower", w.lower}, {"upper", w.upper}, {"count", static_cast<std::int64_t>(s.count)},
                              {"found", static_cast<std::int64_t>(found)}, {"certified", s.certified},
                              {"nu_lower", static_cast<std::int64_t>(counted ? s.lower.neg : -1)},
                              {"nu_upper", static_cast<std::int64_t>(counted ? s.upper.neg : -1)},
                              {"nu_sigma", static_cast<std::int64_t>(nu_sigma_)}, {"sides_ok", sides_ok},
                              {"backward_factor", rule.backward_factor}, {"eta_lower", v.eta_lower},
                              {"eta_upper", v.eta_upper}, {"seconds", s.seconds},
                              {"counter_analyses", static_cast<std::int64_t>(counter_->stats().analyses)}});
  std::ostringstream os;
  os.precision(6);
  if (!counted || s.count < found) {
    os << "single counts did not certify (valid " << counted << ", count " << s.count << " vs found " << found << ")";
    fallback_note_ = os.str();
    return {};
  }
  if (s.certified && sides_ok) {
    single_failed_ = false;
    cert.performed = true;
    cert.method = certificate_method_name(CertificateMethod::ExactSingle);
    cert.lower = w.lower;
    cert.upper = w.upper;
    cert.count_inertia = s.count;
    cert.count_found = found;
    cert.passed = true;
    cert.rigorous = true;
    cert.failure_probability = 0.0;
    os << prefix() << "single-precision Sylvester counts nu(b) - nu(a) = " << s.count << " at [" << w.lower << ", "
       << w.upper << "] equal the " << found << " found pairs there"
       << (nu_sigma_ >= 0 ? " (on either side of sigma, with the exact count at sigma)" : "")
       << "; by the perturbation rule (verify.hpp; B = I or diagonal B, backward error of the single LDL^T assumed <= "
       << rule.backward_factor
       << " u_s ||K||_1, as double counts assume a backward-stable LDL^T) every eigenvalue in [a + eta_a, b - eta_b) "
          "= ["
       << w.lower + v.eta_lower << ", " << w.upper - v.eta_upper << ") is found, which holds the wanted ["
       << sw.wanted_lower << ", " << sw.wanted_upper << "]";
    cert.statement = os.str();
    return {CheckOutcome::Verdict::Passed, 0};
  }
  // More counted than found, or per-side counts contradicting nu(sigma): double counts on the same
  // counter decide (certificate.hpp). They are kept for the caller's exact counts (exact_counts).
  index_t nu_a = 0, nu_b = 0;
  if (!exact_counts(w.lower, w.upper, nu_a, nu_b)) {
    os << "single counts nu(b) - nu(a) = " << s.count << " vs " << found
       << " found pairs could not be checked in double (a window end is singular)";
    fallback_note_ = os.str();
    return {};
  }
  if (nu_a != s.lower.neg || nu_b != s.upper.neg) {
    // The single LDL^T's backward error exceeded the rule's assumption here: not used again.
    os << "single-precision counts nu_s(a) = " << s.lower.neg << ", nu_s(b) = " << s.upper.neg
       << " disagree with the double counts nu(a) = " << nu_a << ", nu(b) = " << nu_b << " at the same ends [" << w.lower
       << ", " << w.upper << "] (the backward error of the single LDL^T exceeded the rule's assumption; single counts "
       << "are not used again in this solve)";
    fallback_note_ = os.str();
    warn(c_, "certificate: " + os.str());
    return {};
  }
  if (s.count == found) {  // both precisions agree, nu(sigma) does not: the double counts certify
    os << "single-precision counts per side of sigma contradict nu(sigma) = " << nu_sigma_
       << " of the main factorization; the double counts confirm the single ones";
    fallback_note_ = os.str();
    return {};
  }
  // A deficit both precisions agree on: an eigenvalue in [a, b] is not found - rerun.
  single_failed_ = false;
  cert.performed = true;
  cert.method = certificate_method_name(CertificateMethod::ExactSingle);
  cert.lower = w.lower;
  cert.upper = w.upper;
  cert.count_inertia = s.count;
  cert.count_found = found;
  cert.passed = false;
  cert.rigorous = false;
  cert.failure_probability = 1.0;
  os << "single-precision Sylvester counts nu(b) - nu(a) = " << s.count << " at [" << w.lower << ", " << w.upper
     << "] exceed the " << found << " found pairs there, and the double counts on the same InertiaCounter agree: "
     << s.count - found << " eigenvalue(s) in the window are not found";
  fallback_note_ = os.str();  // the statement if the exact counts decide after all (last round)
  cert.statement = prefix() + os.str();
  return {CheckOutcome::Verdict::Missing, s.count - found};
}

template <class T>
bool NearestCertifier<T>::exact_counts(double a, double b, index_t& nu_a, index_t& nu_b) {
  if (!counter_) return false;
  if (!(kept_.counted && kept_.a == a && kept_.b == b)) {
    PhaseScope phase(c_.clock, Phase::Certificate);
    if (fac_.inertia().valid && fac_.release_factors()) c_.held = FactorHeld{};
    const auto counts = counter_->count_pair(1.0, -a, 1.0, -b, true, FactorPrecision::Double);
    kept_ = KeptCounts{a, b, counts.first.neg, counts.second.neg, true, counts.first.valid && counts.second.valid};
    event(c_, "certifier_double_counts", {{"lower", a}, {"upper", b}, {"valid", kept_.valid},
                                          {"nu_lower", static_cast<std::int64_t>(kept_.nu_a)},
                                          {"nu_upper", static_cast<std::int64_t>(kept_.nu_b)}});
  }
  if (!kept_.valid) return false;
  nu_a = kept_.nu_a;
  nu_b = kept_.nu_b;
  return true;
}

template <class T>
void NearestCertifier<T>::record_exact(bool on_counter) {
  InertiaCertificate& cert = c_.rep.certificate;
  cert.method = certificate_method_name(CertificateMethod::Exact);
  cert.rigorous = cert.passed;
  cert.failure_probability = cert.passed ? 0.0 : 1.0;
  std::ostringstream os;
  os.precision(6);
  os << prefix();
  if (!fallback_note_.empty()) os << fallback_note_ << "; therefore ";
  os << "double-precision Sylvester counts (" << (on_counter ? "InertiaCounter" : "main factorization")
     << "): nu(b) - nu(a) = " << cert.count_inertia << " eigenvalue(s) in [" << cert.lower << ", " << cert.upper
     << "], " << cert.count_found << " distinct found pair(s) there";
  if (cert.passed) os << ": every eigenvalue in the window is found";
  cert.statement = os.str();
  fallback_note_.clear();
}

// ------------------------------------------------------------------------------ fault injection

template <class T>
std::vector<double> drop_found_pairs(Block<T>& all, std::vector<double>& hints, const std::vector<double>& selected,
                                     double target, index_t count) {
  std::vector<double> removed;
  double r = 0.0;
  for (double l : selected) r = std::max(r, std::abs(l - target));
  std::vector<double> sel = selected;
  auto same = [&](double x, double y) { return std::abs(x - y) <= kCopyTol * std::max({std::abs(x), std::abs(y), r}); };
  auto remove_column = [&](double lambda, bool drop_hints) {
    index_t best = -1;
    for (index_t j = 0; j < all.size(); ++j)
      if (std::isfinite(all.lambda[j]) && (best < 0 || std::abs(all.lambda[j] - lambda) < std::abs(all.lambda[best] - lambda)))
        best = j;
    if (best < 0) return false;
    std::vector<index_t> keep;
    for (index_t j = 0; j < all.size(); ++j)
      if (j != best) keep.push_back(j);
    removed.push_back(all.lambda[best]);
    all = take_columns(all, keep);
    if (drop_hints) hints.erase(std::remove_if(hints.begin(), hints.end(), [&](double h) { return same(h, lambda); }),
                                hints.end());
    auto it = std::min_element(sel.begin(), sel.end(), [&](double x, double y) { return std::abs(x - lambda) < std::abs(y - lambda); });
    if (it != sel.end()) sel.erase(it);
    return true;
  };
  for (index_t d = 0; d < count && !sel.empty(); ++d) {
    // Largest multiplet among the still selected eigenvalues.
    double best_l = 0.0;
    std::size_t best_mult = 1;
    for (double l : sel) {
      std::size_t mult = 0;
      for (double x : sel) mult += same(l, x) ? 1 : 0;
      if (mult > best_mult) {
        best_mult = mult;
        best_l = l;
      }
    }
    bool ok = false;
    if (best_mult > 1) {
      ok = remove_column(best_l, false);
    } else {
      const double far = *std::max_element(sel.begin(), sel.end(), [&](double x, double y) {
        return std::abs(x - target) < std::abs(y - target);
      });
      ok = remove_column(far, true);
    }
    if (!ok) break;
  }
  return removed;
}

#define HS_INSTANTIATE_CERTIFICATE(T)                                                                        \
  template class NearestCertifier<T>;                                                                        \
  template std::vector<double> drop_found_pairs<T>(Block<T>&, std::vector<double>&, const std::vector<double>&, \
                                                   double, index_t);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_CERTIFICATE)
#undef HS_INSTANTIATE_CERTIFICATE

}  // namespace hs::driver
