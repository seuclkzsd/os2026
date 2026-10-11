// Cost model of the completeness checks (include/hsolve/verify.hpp, "(3)").
#include <cmath>
#include <limits>

#include "hsolve/error.hpp"
#include "hsolve/verify.hpp"

namespace hs {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
/// Block sizes the randomized estimate chooses from.
constexpr index_t kBlockSizes[] = {1, 2, 4, 8};

/// Seconds of one block application of q vectors.
double apply_seconds(const CertificateCostInputs& in, index_t q) {
  return in.apply1_s * (1.0 + in.rhs_ratio * static_cast<double>(q - 1));
}

}  // namespace

const char* certificate_method_name(CertificateMethod m) {
  switch (m) {
    case CertificateMethod::Exact: return "exact";
    case CertificateMethod::ExactSingle: return "exact_single";
    case CertificateMethod::Randomized: return "randomized";
  }
  return "unknown";
}

CertificateCostEstimate certificate_cost(const CertificateCostInputs& in) {
  if (in.n < 1 || in.deflated < 0 || in.deflated >= in.n)
    HS_THROW(InvalidInput, "certificate_cost: need n >= 1 and 0 <= deflated < n");
  for (double v : {in.factorize_s, in.counter_analysis_s, in.count_ratio, in.single_ratio, in.apply1_s, in.rhs_ratio})
    if (!(v >= 0.0) || !std::isfinite(v))
      HS_THROW(InvalidInput, "certificate_cost: times and ratios must be finite and >= 0");
  if (!(in.stream_bytes_per_s > 0.0) || !(in.preference >= 1.0) || !(in.kappa >= 1.0))
    HS_THROW(InvalidInput, "certificate_cost: need stream_bytes_per_s > 0, preference >= 1, kappa >= 1");
  if (in.max_krylov_vectors < 0 || !(in.max_memory_bytes >= 0.0) || !std::isfinite(in.max_memory_bytes))
    HS_THROW(InvalidInput, "certificate_cost: need max_krylov_vectors >= 0 and a finite max_memory_bytes >= 0");

  CertificateCostEstimate est;
  const double count_s = in.count_ratio * in.factorize_s;
  est.exact_s = std::min(2.0 * in.factorize_s, in.counter_analysis_s + 2.0 * count_s);
  est.exact_single_s = in.single_valid ? in.counter_analysis_s + 2.0 * in.single_ratio * count_s : kInf;

  est.randomized_s = kInf;
  const index_t m = in.deflated, big_n = in.n - m;
  const double scalar_bytes = in.complex_scalars ? 16.0 : 8.0;
  if (in.epsilon > 0.0 && in.delta > 0.0 && in.delta < 1.0) {
    for (index_t p : kBlockSizes) {
      const index_t s = exclusion_block_steps(in.epsilon, in.delta, big_n, p, in.kappa, in.complex_scalars);
      if (s < 1) continue;
      // exclusion_check refuses a plan beyond its resource budget (Inconclusive, no run).
      const ExclusionRunSize size = exclusion_run_size(in.n, m, s, p, 1, in.complex_scalars);
      if ((in.max_krylov_vectors > 0 && size.krylov_vectors > in.max_krylov_vectors) ||
          (in.max_memory_bytes > 0.0 && size.bytes > in.max_memory_bytes))
        continue;
      double t = static_cast<double>(s) * apply_seconds(in, p);
      if (!in.coupling_known && m > 0) t += apply_seconds(in, m);
      // Block CGS2 (two passes) of step j sweeps [X V_j] (m + j p columns) twice per pass.
      double cols = 0.0;
      for (index_t j = 1; j <= s; ++j) cols += static_cast<double>(m + j * p);
      t += 4.0 * cols * static_cast<double>(in.n) * scalar_bytes / in.stream_bytes_per_s;
      if (t < est.randomized_s) {
        est.randomized_s = t;
        est.block_size = p;
        est.block_steps = s;
        // A space covering the complement is exact (exclusion_block_steps' fallback).
        est.failure_bound = s * p >= big_n ? 0.0
                                           : exclusion_failure_bound(in.epsilon, s, big_n, p, in.kappa,
                                                                     in.complex_scalars);
      }
    }
  }
  const double best_exact = std::min(est.exact_s, est.exact_single_s);
  est.recommended = est.exact_single_s < est.exact_s ? CertificateMethod::ExactSingle : CertificateMethod::Exact;
  if (est.randomized_s * in.preference < best_exact) est.recommended = CertificateMethod::Randomized;
  return est;
}

}  // namespace hs
