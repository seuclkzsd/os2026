// Completeness check of the NEAREST target (HEP / GHEP) by the method SolverConfig::certificate
// selects (driver-internal; definite.cpp calls it once per certificate round). Scalar set of the
// templates: T in {double, cplx}, instantiated in certificate.cpp.
//
// Methods (include/hsolve/verify.hpp):
//   exact       nu(b) - nu(a) by double-precision LDL^T counts at the window ends [a, b] - the
//               certificate of DESIGN §3.6-8, counted by definite.cpp itself (count_below on its
//               CountSource; on this certifier's InertiaCounter when single counts built one,
//               exact_counts), unchanged;
//   exact_single the same two counts in single precision on an InertiaCounter, only where the
//               perturbation rule of verify.hpp (2) makes them certify the window (B = I or
//               diagonal B, every found eigenvalue and the wanted region eta away from the ends);
//   randomized  exclusion_check (verify.hpp (1)) on the factorization already held at sigma:
//               block Lanczos on the operator deflated by the found pairs X, from a Gaussian start
//               block, with a failure-probability bound delta (SolverConfig::certificate_delta).
// 'auto' predicts each method's seconds with certificate_cost from this run's MEASURED main
// factorization time and solve-call time (the Krylov run's block width) and runs the cheapest
// that certifies: randomized only with a proven kappa (B = I, diagonal or diagonally dominant B at
// O(nnz(B)); P8: a general HPD B through the incomplete-Cholesky bounds of norms.hpp,
// prove_condition_bound, which also give single counts the lambda_min(B) their rule needs), single
// counts only where the rule holds with kAutoSingleSafety times its assumed backward error
// (below); otherwise, and whenever a fast check cannot decide (gap too small, plan over the memory
// budget, a count that disagrees), the exact counts. Mode 'randomized' runs the check whenever a
// plan exists and kappa is proven; with a general B whose bounds are not proven (kappa only
// estimable) the check could only SCREEN, so it is not run and the exact counts certify (warning,
// statement). No screening result ever becomes the report's verdict (a pass without proof would
// read as a certificate downstream).
//
// The randomized claim. sigma = the held shift, target = target.sigma (|sigma - target| > 0 only
// after a near-singular perturbation), r = max |lambda - target| over the selected pairs, so
// every wanted eigenvalue lies within r_edge = r + |sigma - target| of sigma. The pass radius r'
// is the certificate window's half-width on its nearer side (the ends sit mid-way to the next
// Ritz hints, as for the counts), and the claim radius r_c lies between: 1/r_c = 1/r_edge -
// kClaimFraction (1/r_edge - 1/r'), which keeps every found eigenvalue within r_edge more than
// the coupling e = ||Op X - X H11||_B inside the claim (Kahan: certified_inside) as long as e is
// below that fraction of the theta gap, at a 1% cost in eps. The check certifies when it passes
// AND found_inside == certified_inside: then, with probability >= 1 - delta(s), the pencil has
// exactly found_inside eigenvalues within r_c of sigma, all approximated by found pairs, so the
// selected pairs (all within r_edge < r_c) are the nearest ones. X = the converged found pairs
// (backward error <= 10 tol) in the window [a, b]: the selected ones as residual correction left
// them, and the other computed ones; an unconverged pair is left out and, if it is an eigenvalue
// within r', detected. A check still detecting after the last rerun leaves the final verdict to
// the exact counts (the solve never fails where the counts pass). Every check covers the found
// pairs' disc / window only: with fewer than k selected pairs definite.cpp reruns even after a
// pass (the k-th nearest may lie beyond the window).
//
// Single counts. The rule's one assumption is the backward error of MUMPS' single LDL^T, <= c g
// u_s ||K||_1 with c g = SingleCountRule::backward_factor = 1e3. A P5 measurement violated it: on
// yee-pec-box-1e5 (16 threads, 1 of ~50 processes, deterministic = false) the single counts gave
// nu(b) - nu(a) = 12 against the double counts' 10 at the same ends, an eigenvalue at least 2.96e3
// u_s ||K||_1 / b_min from an end counted on the wrong side (its 35475-fold null space at 0 or
// the eigenvalue 102.57). Hence:
//   * auto takes single counts only where the rule holds at kAutoSingleSafety x backward_factor;
//   * a single count above the found pairs, or per-side counts that contradict the exact count
//     nu(sigma) of the main factorization (B-definite pencils: nu(sigma) - nu_s(a) found pairs in
//     [a, sigma), nu_s(b) - nu(sigma) in [sigma, b]), is checked by double counts on the same
//     InertiaCounter (analysed already; two counts, far cheaper than a needless rerun): a deficit
//     both precisions agree on triggers the deflated rerun; single counts that disagree with the
//     double ones are not used again in this solve, and the double counts give the verdict;
//   * single counts BELOW the found pairs, or an invalid (singular) single count, leave the
//     verdict to the exact counts (duplicates / ghosts are theirs to report).
// On a verdict "missing" (exact: count > found; exact_single: the confirmed deficit; randomized:
// Detected, new_count Ritz values beyond the pass threshold) definite.cpp runs its deflated rerun
// and checks again with the same method. Detection of an eigenvalue in (r_c, r'] (an unconverged
// Ritz hint placed the window end too far) is not an error.
//
// Failures (P8). A factorization of the checks that fails - MUMPS out of memory (INFO -13, -19
// under the ICNTL(23) cap), another MUMPS error, std::bad_alloc - never ends the solve:
//   * this certifier's InertiaCounter (single counts, the double check of single counts,
//     exact_counts) is abandoned for the solve (counter_failure()); so is one whose estimate
//     exceeds count_allowance_mb (internal.hpp) - it is not even tried. The exact counts then
//     come from definite.cpp's CountSource, which skips its own counter too after a memory
//     failure (same pivot order, same working memory) and counts on the main object;
//   * a randomized check whose refactorization at sigma or whose solves fail (an hs::Error of
//     memory or MUMPS, std::bad_alloc) leaves the verdict to the counts;
//   * when no count can be made (CountFailure), record_not_completed() reports the certificate as
//     performed but not completed and definite.cpp returns the converged pairs uncertified
//     (success = false); a deflated rerun that cannot refactorize at sigma returns the pairs found.
// The fault bits of SolverConfig::certificate_fault_count exercise each path.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"
#include "hsolve/verify.hpp"

namespace hs::driver {

/// Fraction of the theta gap between the farthest wanted eigenvalue and the pass threshold that
/// separates the randomized claim radius from that eigenvalue (header comment).
constexpr double kClaimFraction = 0.01;

/// auto takes single counts only where the rule of verify.hpp (2) holds with this multiple of its
/// assumed backward error (SingleCountRule::backward_factor): 1e4 u_s ||K||_1 against the 2.96e3
/// or more of the miscount measured on yee-pec-box-1e5 (header comment). Margins of the rule on
/// the P5 3-D cases at 16 threads (largest factor that still holds the window): yee-pec-box-1e5
/// 2.96e3, yee-photonic-rods-1e4 5.7e3, yee-pec-box-1e4 6.9e3, yee-pec-cube-32 1.0e4,
/// yee-bloch-1e4 1.3e4, yee-pec-cube-16 3.9e4. Measured with the margin (bench_certint_e2e, gpu02
/// measurement slot, 16 threads, 5 interleaved repetitions, medians; every run certified, tools/eval
/// 96/96 pass): auto now takes the exact counts on yee-pec-box-1e5 / -1e4, yee-photonic-rods-1e4
/// and yee-bloch-1e4 (auto/exact 0.99-1.00x; single counts had given yee-pec-box-1e5 3.20 -> 2.87 s,
/// against tuned SLEPc now 1.02x instead of 1.15x: tools/bench/compare.py, 5 repetitions),
/// the randomized check on yee-pec-cube-32 (1.09x), syn-mem27-46 (1.29x) and syn-mem125-46 (1.19x),
/// single counts only on yee-pec-cube-16 (1.05x); elsewhere it matches exact within noise.
constexpr double kAutoSingleSafety = 10.0;

/// A general B's condition bounds (b_bounds(): an incomplete-Cholesky proof plus one O(nnz(B))
/// pass): the time budget of the proof's incomplete-Cholesky attempts and level-1 pattern
/// (HpdProofOptions::max_seconds, each paid by its predicted time before it runs) is this share of
/// the predicted time of the exact counts they could save. The Lanczos estimate of theta and the
/// scaled copy S are not charged, so a whole proof can take longer than that budget (16 threads,
/// certificate = randomized, 3 runs each: cav2d-rect-pert-1e5 0.016-0.023 s against 0.016-0.021 s,
/// ss-tem27623 0.026-0.048 s against 0.024-0.032 s). Proofs that succeed cost 0.004-0.08 s at 16
/// threads on the data cases (bench/anaint_bproof.cpp).
constexpr double kBoundsProofShare = 0.25;

/// What the nearest path measured before its certificate (inputs of the auto choice). Zero
/// entries fall back to the run-time models of DESIGN §3.6-3b.
struct CertificateTimings {
  double factorize_s = 0.0;   ///< one numeric factorization of K(sigma)
  double solve_call_s = 0.0;  ///< one solve call of the Krylov run (all of its right-hand sides)
  double rhs_per_call = 1.0;  ///< right-hand sides per solve call (the run's block width)
};

/// One round's window, computed by definite.cpp as for the exact counts.
struct NearestWindow {
  double target = 0.0;              ///< selection centre (target.sigma)
  double radius = 0.0;              ///< r = max |lambda - target| over the selected pairs
  double lower = 0.0, upper = 0.0;  ///< window ends a < target - r, b > target + r (mid-gap)
};

/// The found pairs of a round. `all`: every computed pair as the Krylov runs left them
/// (B-orthonormal X with A X, B X in sync; definite.cpp's run.all). `selected`: the selected pairs
/// after residual correction and Rayleigh-Ritz (definite.cpp's sel), taken from the columns
/// `selected_columns` of `all`; nullptr: those columns of `all` as they are. The randomized check
/// deflates the corrected selected pairs and the other converged pairs of `all` in the window -
/// the uncorrected copies in `all` may miss the convergence test (a near-singular shift).
template <class T>
struct RoundPairs {
  RoundPairs(const Block<T>& all_pairs, const Block<T>* selected_pairs = nullptr,
             std::vector<index_t> columns = {})
      : all(all_pairs), selected(selected_pairs), selected_columns(std::move(columns)) {}
  const Block<T>& all;
  const Block<T>* selected = nullptr;
  std::vector<index_t> selected_columns;
};

/// Verdict of a fast check. Undecided: the caller counts exactly (exact mode, auto choosing the
/// exact counts, or a fast check that could not decide) - exact_counts() first, which reuses the
/// certifier's counter and the double counts it may already have made - then calls
/// record_exact(). Passed covers the found pairs only (fewer than k selected: the caller reruns).
struct CheckOutcome {
  enum class Verdict { Passed, Missing, Undecided };
  Verdict verdict = Verdict::Undecided;
  index_t missing = 0;  ///< Missing: pairs the deflated rerun should look for (>= 1)
};

/// The prediction behind an auto choice, for the report.
struct CertificateChoice {
  CertificateMethod method = CertificateMethod::Exact;
  CertificateCostInputs inputs;  ///< what certificate_cost was given (logged: bench/certint_model.cpp
                                 ///< re-evaluates it with other times, e.g. a 16-thread run's)
  CertificateCostEstimate estimate;
  /// kappa of B proven (auto considers the randomized check only then); true when kappa was not
  /// needed (the check was out of the question before: randomized_reason)
  bool kappa_rigorous = true;
  std::string single_reason;      ///< why single counts are not considered ("" when the rule holds)
  std::string randomized_reason;  ///< why the randomized check is not considered ("" when it is)
  double epsilon = 0.0;           ///< eps of the randomized estimate (coupling taken as 0)
};

/// Where a nearest certificate's two exact counts were made, for its statement: by_helpers of them
/// by helper processes of a multi-process solve (on their InertiaCounters, DESIGN §3.9), the rest by
/// the leader on its InertiaCounter (leader_counter) or on its main factorization.
std::string count_sources(index_t by_helpers, bool leader_counter);

/// The nearest target's fast completeness checks, one object per solve (kept across the
/// certificate rounds: start scaling, counter, failed methods). Fills SolveReport::certificate on
/// a verdict; all of its work is charged to Phase::Certificate (except a refactorization at sigma
/// that a randomized check after main-object counts needs: Phase::Factor). Not thread-safe; every
/// referenced object must outlive it.
template <class T>
class NearestCertifier {
 public:
  /// sigma / detection_off: the shift the main object factorized for the Krylov run (as
  /// factorize_near returned it). n2a, n2b: norm estimates (backward errors of the found pairs).
  NearestCertifier(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double sigma,
                   bool detection_off, double scale, double n2a, double n2b, const CertificateTimings& timings);
  ~NearestCertifier();
  NearestCertifier(const NearestCertifier&) = delete;
  NearestCertifier& operator=(const NearestCertifier&) = delete;

  /// This round's fast check, if the mode asks for one and it is possible. `all`: every computed
  /// pair (B-orthonormal X with A X, B X in sync). Exact mode returns Undecided without doing
  /// anything; otherwise choose() then run(), and in auto the single counts when a randomized
  /// check could not decide and they are predicted cheaper than the exact counts.
  CheckOutcome check(const NearestWindow& w, const RoundPairs<T>& pairs, int round);

  /// Runs one method on this round's window: Randomized (block width from choose()) or
  /// ExactSingle; Exact returns Undecided. Fills the report on a verdict.
  CheckOutcome run(CertificateMethod method, const NearestWindow& w, const RoundPairs<T>& pairs, int round);

  /// Report fields after the exact counts of this round (cert.passed etc. already set by the
  /// caller); source: where the counts were made, for the statement ("InertiaCounter", "main
  /// factorization", or one per count in a multi-process solve: count_sources, driver/helpers.hpp).
  void record_exact(const std::string& source);

  /// Report fields when no count could be made at this round's window [a, b] (CountFailure: every
  /// count source failed, `why`): the certificate was performed but not completed - passed =
  /// false, rigorous = false, failure_probability = 1, count_inertia = -1 (nothing counted),
  /// count_found = `found`, method "exact" (the counts that failed), and a statement and note that
  /// say the found pairs are returned uncertified (docs/FORMATS.md §3).
  void record_not_completed(double a, double b, index_t found, const std::string& why);

  /// Why this certifier's InertiaCounter could not count (its single or double counts failed, or
  /// its estimate did not fit the count allowance); "" while it did not fail.
  const std::string& counter_failure() const { return counter_failure_; }
  /// The failure was one of memory (OutOfMemoryBudget, std::bad_alloc): another InertiaCounter on
  /// the same pivot order (CountSource's) would not fit either.
  bool counter_memory_failure() const { return counter_memory_failure_; }

  /// The exact (double-precision) counts nu(a), nu(b) of a window on this certifier's
  /// InertiaCounter, which single counts built (its double instance deferred: analysed at the
  /// first double count): no second counter and no main-object factorization. Counted with null-pivot
  /// detection at a and b themselves (gap midpoints, as count_below with s_is_gap); the last
  /// window's counts are kept, so a window counted to check single counts is not counted again.
  /// false: no counter yet, K(a) / K(b) flagged singular, or the counter failed (its estimate
  /// over count_allowance_mb, a MUMPS error: counter_failure(); not tried again) - the caller
  /// counts (count_below on its CountSource). Releases the main factors first
  /// (MainFactors::Release); Phase::Certificate.
  bool exact_counts(double a, double b, index_t& nu_a, index_t& nu_b);

  /// The auto choice for window w (no side effect besides the start scaling, computed once:
  /// without the Lanczos kappa estimate of a general B unless the mode is randomized).
  CertificateChoice choose(const NearestWindow& w, const RoundPairs<T>& pairs);

  /// FAULT INJECTION, for tests only: adds these numbers to every later single-precision count at
  /// a window's lower / upper end (as a single LDL^T beyond the rule's assumed backward error
  /// would miscount), so that the double-count check of single counts can be exercised.
  void inject_single_miscount(index_t at_lower, index_t at_upper) {
    single_fault_lower_ = at_lower;
    single_fault_upper_ = at_upper;
  }

 private:
  /// The single-count window of w: the found pairs of `all` in [lower, upper] with their residual
  /// bounds ||A x - lambda B x||_{B^{-1}} <= ||r||_2 inv_sqrt_bmin (1 / sqrt of a lower bound on
  /// lambda_min(B)), the wanted region the selected pairs' disc.
  SingleWindow single_window(const NearestWindow& w, const Block<T>& all, double inv_sqrt_bmin);
  /// The single-count rule with the pencil's ||A||_1 (computed at the first call); prove: for a
  /// general B also its proven lambda_min (b_bounds(); SingleCountRule::b_min_proven).
  const SingleCountRule& single_rule(bool prove);
  /// Columns of the randomized check's deflation block X (RoundPairs comment), pointers into
  /// `pairs`; deflation_block copies them.
  std::vector<const T*> deflation_columns(const NearestWindow& w, const RoundPairs<T>& pairs) const;
  DenseMatrix<T> deflation_block(const NearestWindow& w, const RoundPairs<T>& pairs) const;
  bool converged(const Block<T>& blk, index_t j) const;
  CheckOutcome randomized(const NearestWindow& w, const RoundPairs<T>& pairs, index_t block_size, int round);
  /// safety: multiple of the rule's backward factor the window must hold with (auto:
  /// kAutoSingleSafety; run(): 1).
  CheckOutcome single(const NearestWindow& w, const Block<T>& all, double safety);
  /// Start scaling and kappa of B: the proven bound - B = I, diagonal or diagonally dominant B
  /// (O(nnz(B))), or a general B's bound from b_bounds() (P8) - or, with estimate, a general B's
  /// estimated kappa where no bound is proven (only a randomized check run() asks for it on such a
  /// B: a screening result). Computed once each.
  const StartScaling& scaling(bool estimate);
  /// A general (non-diagonal) B's proven bounds (norms.hpp prove_condition_bound): kappa of the
  /// Jacobi-scaled B and lambda_min(B) >= b_min. Computed once, at the first use - only where the
  /// randomized check or single counts could beat the exact counts (choose()) or the mode asks for
  /// the randomized check - its attempts within kBoundsProofShare of the predicted exact counts'
  /// time (kBoundsProofShare comment: not the whole proof).
  const ConditionBound& b_bounds();
  double bounds_budget_s() const;
  double check_budget_bytes() const;
  std::string prefix() const;
  /// The InertiaCounter faults of SolverConfig::certificate_fault_count apply to this certifier's
  /// counter (not with kFaultLate: those are for window targets only).
  bool counter_faults() const { return !fault(c_.cfg, kFaultLate); }

  Ctx& c_;
  const ShiftedPencil<T>& pencil_;
  SparseFactorization<T>& fac_;
  double sigma_;
  bool detection_off_;
  double scale_, n2a_, n2b_;
  /// 1 / sqrt(lambda_min(B) bound): min b_ii of a diagonal B, 1 for B = I, a general B's proven
  /// b_min, +inf without one: turns ||r||_2 into a bound on ||r||_{B^{-1}} (the single-count rule's
  /// error bars); set by single_rule().
  double inv_sqrt_bmin_ = 1.0;
  SingleCountRule rule_;  ///< a_norm_1 < 0 until single_rule() computed it
  CertificateTimings timings_;
  std::unique_ptr<StartScaling> proven_scaling_, estimated_scaling_;
  std::unique_ptr<ConditionBound> b_bounds_;  ///< see b_bounds()
  bool b_diagonal_ = true;                    ///< B = I or diagonal (set by single_rule())
  double b_diag_min_ = 1.0;                   ///< min b_ii (set by single_rule())
  std::unique_ptr<InertiaCounter<T>> counter_;  ///< single counts (created at the first one; its
                                                ///< double instance deferred to the first double count)
  /// exact_counts' last window [a, b] (counted) and its counts (valid: both nonsingular)
  struct KeptCounts {
    double a = 0.0, b = 0.0;
    index_t nu_a = 0, nu_b = 0;
    bool counted = false, valid = false;
  } kept_;
  /// nu(sigma) of the main factorization (exact: double, full rank, null-pivot detection on),
  /// taken at the first single count while the main object holds K(sigma); -1 = unknown
  index_t nu_sigma_ = -1;
  bool nu_sigma_taken_ = false;
  index_t single_fault_lower_ = 0, single_fault_upper_ = 0;  ///< see inject_single_miscount
  std::string counter_failure_;          ///< see counter_failure()
  bool counter_memory_failure_ = false;  ///< see counter_memory_failure()
  /// Records a failure of counter_ (its counts are not tried again in this solve).
  void counter_failed(const std::string& why, bool memory);
  bool randomized_failed_ = false;  ///< an earlier round's check could not decide: not again
  bool single_failed_ = false;      ///< single counts could not decide (invalid / below found /
                                    ///< contradicted by double counts): not again
  std::string auto_note_;           ///< auto: the last prediction, prefixed to the statement
  std::string fallback_note_;       ///< why this round's fast check left the verdict to the counts
};

/// FAULT INJECTION (SolverConfig::certificate_fault_drop): removes `count` pairs from `all`
/// before the completeness check - copies of the largest multiple eigenvalue among `selected`
/// (eigenvalues within 1e-8 max(|lambda|, r) of each other; one copy always stays), then the
/// selected eigenvalue farthest from `target`, whose Ritz hints are removed too (the run never
/// saw it). Returns the removed eigenvalues.
template <class T>
std::vector<double> drop_found_pairs(Block<T>& all, std::vector<double>& hints, const std::vector<double>& selected,
                                     double target, index_t count);

}  // namespace hs::driver
