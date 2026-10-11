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
struct NearestCertifier<T>::Found {
  std::vector<index_t> window;     ///< columns of `all` with a finite lambda in [lower, upper]
  std::vector<double> lambda;      ///< their eigenvalues
  std::vector<double> error;       ///< their residual bounds ||A x - lambda B x||_{B^{-1}}
};

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

template <class T>
typename NearestCertifier<T>::Found NearestCertifier<T>::found_in_window(const NearestWindow& w,
                                                                         const Block<T>& all) const {
  Found f;
  const index_t n = all.x.rows();
  // ||r||_{B^{-1}} <= ||r||_2 / sqrt(min b_ii) for a diagonal B (B-normalized x); B = I: ||r||_2.
  // For other B the bound only feeds the single-count rule, which then does not apply anyway.
  double inv_sqrt_bmin = 1.0;
  if (const CsrMatrix<T>* b = pencil_.b()) {
    double bmin = std::numeric_limits<double>::infinity();
    for (index_t i = 0; i < b->nrows; ++i)
      for (index_t q = b->rowptr[i]; q < b->rowptr[i + 1]; ++q)
        if (b->colind[q] == i) bmin = std::min(bmin, std::real(b->values[q]));
    inv_sqrt_bmin = bmin > 0.0 ? 1.0 / std::sqrt(bmin) : std::numeric_limits<double>::infinity();
  }
  for (index_t j = 0; j < all.size(); ++j) {
    const double l = all.lambda[j];
    if (!std::isfinite(l) || l < w.lower || l > w.upper) continue;
    const PairMetrics pm = pair_metrics(all.ax.col(j), all.bx.col(j), all.x.col(j), n, l, n2a_, n2b_);
    f.window.push_back(j);
    f.lambda.push_back(l);
    f.error.push_back(pm.abs_residual * inv_sqrt_bmin);
  }
  return f;
}

template <class T>
bool NearestCertifier<T>::converged(const Block<T>& blk, index_t j) const {
  const PairMetrics pm =
      pair_metrics(blk.ax.col(j), blk.bx.col(j), blk.x.col(j), blk.x.rows(), blk.lambda[j], n2a_, n2b_);
  return std::isfinite(blk.lambda[j]) && pm.backward_error <= kConvergedFactor * c_.cfg.tol;
}

template <class T>
DenseMatrix<T> NearestCertifier<T>::deflation_block(const NearestWindow& w, const RoundPairs<T>& pairs) const {
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
  const index_t n = pairs.all.x.rows(), m = static_cast<index_t>(cols.size());
  DenseMatrix<T> x(n, m);
  for (index_t j = 0; j < m; ++j) std::copy(cols[j], cols[j] + n, x.col(j));
  return x;
}

// The randomized check needs kappa, and only a proven one counts for auto: the Lanczos estimate
// of a general B's kappa (~0.1 s at n = 1e5) is computed only when the mode is randomized.
template <class T>
const StartScaling& NearestCertifier<T>::scaling() {
  if (!scaling_)
    scaling_ = std::make_unique<StartScaling>(
        start_scaling_for<T>(pencil_.b(), 100.0, c_.cfg.certificate == CertificateMode::Randomized));
  return *scaling_;
}

template <class T>
double NearestCertifier<T>::check_budget_bytes() const {
  const double avail = c_.cfg.memory_budget_gb * 1073741824.0 -
                       static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
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
  const Found f = found_in_window(w, pairs.all);
  const index_t n = pencil_.order();
  const MumpsInputSize in_size = mumps_input_size(pencil_);
  const FactorEstimate est = fac_.estimate();
  const int threads = fac_.options().threads;

  CertificateCostInputs in;
  in.n = n;
  in.deflated = std::min<index_t>(deflation_block(w, pairs).cols(), n - 1);
  in.complex_scalars = is_complex_v<T>;
  in.factorize_s = timings_.factorize_s > 0.0 ? timings_.factorize_s : model_factor_s(est, in_size.order, threads);
  const double analysis = model_analysis_s(in_size.order, in_size.entries, static_cast<double>(est.factor_entries));
  in.counter_analysis_s = analysis;
  // A fresh counter analyses its double instance at construction and its single one at the first
  // single count; a counter of an earlier round has done both.
  in.single_analysis_s = counter_ ? (counter_->stats().single_counts > 0 ? 0.0 : analysis) : 2.0 * analysis;
  in.single_ratio = static_cast<double>(est.factor_entries) > kThreeDFill * in_size.entries ? kSingleRatio3d
                                                                                             : kSingleRatioLowFill;
  // The single-count rule on this window (wanted region: the selected pairs' disc).
  SingleWindow sw;
  sw.lower = w.lower;
  sw.upper = w.upper;
  sw.wanted_lower = w.target - w.radius;
  sw.wanted_upper = w.target + w.radius;
  sw.found = f.lambda;
  sw.found_error = f.error;
  bool single_ok = false;
  if (sw.lower < sw.wanted_lower && sw.wanted_upper < sw.upper) {
    const SingleCountValidity v = single_count_validity(pencil_, sw);
    single_ok = v.valid();
    ch.single_reason = v.reason;
  } else {
    ch.single_reason = "window does not strictly contain the wanted region";
  }
  if (single_failed_) {
    single_ok = false;
    ch.single_reason = "single counts did not certify in an earlier round";
  }
  in.single_valid = single_ok;
  // Measured solve calls of the Krylov run; t_apply(q) = apply1 (1 + rhs_ratio (q - 1)).
  const double q = std::max(1.0, timings_.rhs_per_call);
  const double call = timings_.solve_call_s > 0.0 ? timings_.solve_call_s
                                                  : model_solve_s(est) * (1.0 + in.rhs_ratio * (q - 1.0));
  in.apply1_s = call / (1.0 + in.rhs_ratio * (q - 1.0));
  const Radii rad = randomized_radii(w, sigma_);
  const StartScaling& sc = scaling();
  ch.kappa_rigorous = sc.rigorous;
  ch.epsilon = rad.valid ? rad.epsilon : 0.0;
  // Randomized: never after an undecided check; auto takes it only with a proven kappa.
  const bool randomized_allowed =
      rad.valid && !randomized_failed_ && (cfg.certificate == CertificateMode::Randomized || sc.rigorous);
  in.epsilon = randomized_allowed ? rad.epsilon : 0.0;
  in.delta = cfg.certificate_delta;
  in.kappa = std::isfinite(sc.kappa) && sc.kappa >= 1.0 ? sc.kappa : 1.0;
  in.max_memory_bytes = check_budget_bytes();
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
  const CertificateChoice ch = choose(w, pairs);
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
    else if (!ch.kappa_rigorous)
      os << " [no proven kappa of B (general B): not a certificate, not considered]";
    os << "; chose " << certificate_method_name(method);
    auto_note_ = os.str();
  }
  event(c_, "certificate_choice",
        {{"round", static_cast<std::int64_t>(round)}, {"method", std::string(certificate_method_name(method))},
         {"exact_s", e.exact_s}, {"single_s", std::isfinite(e.exact_single_s) ? e.exact_single_s : -1.0},
         {"randomized_s", std::isfinite(e.randomized_s) ? e.randomized_s : -1.0},
         {"block_size", static_cast<std::int64_t>(e.block_size)},
         {"block_steps", static_cast<std::int64_t>(e.block_steps)}, {"epsilon", ch.epsilon},
         {"kappa_rigorous", ch.kappa_rigorous}, {"factorize_s", timings_.factorize_s},
         {"solve_call_s", timings_.solve_call_s}, {"rhs_per_call", timings_.rhs_per_call}});
  if (method == CertificateMethod::Randomized) {
    const CheckOutcome out = randomized(w, pairs, e.block_size, round);
    if (out.verdict != CheckOutcome::Verdict::Undecided) return out;
    // auto: the next cheapest method that certifies
    if (cfg.certificate == CertificateMode::Auto && e.exact_single_s < e.exact_s) method = CertificateMethod::ExactSingle;
  }
  if (method == CertificateMethod::ExactSingle) return single(w, pairs.all);
  return {};
}

template <class T>
CheckOutcome NearestCertifier<T>::run(CertificateMethod method, const NearestWindow& w,
                                      const RoundPairs<T>& pairs, int round) {
  PhaseScope phase(c_.clock, Phase::Certificate);
  switch (method) {
    case CertificateMethod::Randomized: return randomized(w, pairs, choose(w, pairs).estimate.block_size, round);
    case CertificateMethod::ExactSingle: return single(w, pairs.all);
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
  SolveRefinement refine;
  if (fac_.options().precision == FactorPrecision::Single || fac_.options().blr) {
    refine.max_steps = 10;
    refine.rel_tol = cfg.tol * cfg.inner_tol_factor * 0.1;
  }
  ShiftInvertOperator<T> op(fac_, pencil_, sigma_, refine);
  std::unique_ptr<CsrOperator<T>> b_op;
  if (pencil_.b()) b_op = std::make_unique<CsrOperator<T>>(*pencil_.b());
  ExclusionOptions o;
  o.pass_radius = rad.pass;
  o.delta = cfg.certificate_delta;
  o.block_size = block_size;
  o.seed = cfg.seed + 7919u * static_cast<std::uint64_t>(round + 1);  // a fresh start block per round
  set_start_scaling(o, scaling());
  o.max_memory_bytes = check_budget_bytes();
  o.tol = cfg.tol * cfg.inner_tol_factor;
  ExclusionResult<T> res;
  try {
    res = exclusion_check<T>(op, b_op.get(), x, sigma_, rad.claim, o);
  } catch (const Error& err) {
    if (err.code() != Error::Code::InvalidInput) throw;
    fallback_note_ = std::string("randomized check rejected its input: ") + err.what();
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
                 scaling().note + ")");
  return {CheckOutcome::Verdict::Passed, 0};
}

template <class T>
CheckOutcome NearestCertifier<T>::single(const NearestWindow& w, const Block<T>& all) {
  InertiaCertificate& cert = c_.rep.certificate;
  single_failed_ = true;  // reset below on a verdict
  const Found f = found_in_window(w, all);
  SingleWindow sw;
  sw.lower = w.lower;
  sw.upper = w.upper;
  sw.wanted_lower = w.target - w.radius;
  sw.wanted_upper = w.target + w.radius;
  sw.found = f.lambda;
  sw.found_error = f.error;
  if (!(sw.lower < sw.wanted_lower && sw.wanted_upper < sw.upper)) {
    fallback_note_ = "single counts: the window does not strictly contain the wanted region";
    return {};
  }
  const SingleCountValidity v = single_count_validity(pencil_, sw);
  if (!v.valid()) {
    fallback_note_ = "single counts do not certify this window: " + v.reason;
    return {};
  }
  // As the exact counts on a counter (MainFactors::Release): the main factors are needed again
  // only for a rerun, which refactorizes; the counts never hold both in memory.
  if (fac_.inertia().valid && fac_.release_factors()) c_.held = FactorHeld{};
  if (!counter_) counter_ = std::make_unique<InertiaCounter<T>>(fac_, fac_.options());
  const SingleCertificate s = single_precision_certificate(*counter_, pencil_, sw);
  event(c_, "single_counts", {{"lower", w.lower}, {"upper", w.upper}, {"count", static_cast<std::int64_t>(s.count)},
                              {"found", static_cast<std::int64_t>(f.window.size())}, {"certified", s.certified},
                              {"eta_lower", v.eta_lower}, {"eta_upper", v.eta_upper}, {"seconds", s.seconds}});
  if (!s.certified) {
    std::ostringstream os;
    os << "single counts did not certify (valid " << (s.lower.valid && s.upper.valid) << ", count " << s.count
       << " vs found " << f.window.size() << ")";
    fallback_note_ = os.str();
    return {};
  }
  single_failed_ = false;
  cert.performed = true;
  cert.method = certificate_method_name(CertificateMethod::ExactSingle);
  cert.lower = w.lower;
  cert.upper = w.upper;
  cert.count_inertia = s.count;
  cert.count_found = static_cast<index_t>(f.window.size());
  cert.passed = true;
  cert.rigorous = true;
  cert.failure_probability = 0.0;
  std::ostringstream os;
  os.precision(6);
  os << prefix() << "single-precision Sylvester counts nu(b) - nu(a) = " << s.count << " at [" << w.lower << ", "
     << w.upper << "] equal the " << f.window.size()
     << " found pairs there; by the perturbation rule (verify.hpp; B = I or diagonal B, backward error of the "
        "single LDL^T assumed <= 1e3 u_s ||K||_1, as double counts assume a backward-stable LDL^T) every "
        "eigenvalue in [a + eta_a, b - eta_b) = ["
     << w.lower + v.eta_lower << ", " << w.upper - v.eta_upper << ") is found, which holds the wanted ["
     << sw.wanted_lower << ", " << sw.wanted_upper << "]";
  cert.statement = os.str();
  return {CheckOutcome::Verdict::Passed, 0};
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
