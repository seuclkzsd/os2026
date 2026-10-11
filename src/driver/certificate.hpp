// Completeness check of the NEAREST target (HEP / GHEP) by the method SolverConfig::certificate
// selects (driver-internal; definite.cpp calls it once per certificate round). Scalar set of the
// templates: T in {double, cplx}, instantiated in certificate.cpp.
//
// Methods (include/hsolve/verify.hpp):
//   exact       nu(b) - nu(a) by double-precision LDL^T counts at the window ends [a, b] - the
//               certificate of DESIGN §3.6-8, counted by definite.cpp itself (count_below on its
//               CountSource), unchanged;
//   exact_single the same two counts in single precision on an InertiaCounter, only where the
//               perturbation rule of verify.hpp (2) makes them certify the window (B = I or
//               diagonal B, every found eigenvalue and the wanted region eta away from the ends);
//   randomized  exclusion_check (verify.hpp (1)) on the factorization already held at sigma:
//               block Lanczos on the operator deflated by the found pairs X, from a Gaussian start
//               block, with a failure-probability bound delta (SolverConfig::certificate_delta).
// 'auto' predicts each method's seconds with certificate_cost from this run's MEASURED main
// factorization time and solve-call time (the Krylov run's block width) and runs the cheapest
// that certifies: randomized only with a proven kappa (B = I, diagonal or diagonally dominant B),
// single counts only where the rule holds; otherwise, and whenever a fast check cannot decide
// (gap too small, plan over the memory budget, a count that disagrees), the exact counts. Mode
// 'randomized' runs the check whenever a plan exists; with an estimated kappa its pass is a
// SCREENING result (rigorous = false, said in the statement and a warning), never a certificate.
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
// k selected pairs (all within r_edge < r_c) are k nearest ones. X = the converged found pairs
// (backward error <= 10 tol) in the window [a, b]; an unconverged pair is left out and, if it is
// an eigenvalue within r', detected.
//
// On a verdict "missing" (exact: count > found; randomized: Detected, new_count Ritz values
// beyond the pass threshold) definite.cpp runs its deflated rerun and checks again. Detection of
// an eigenvalue in (r_c, r'] (an unconverged Ritz hint placed the window end too far) is not an
// error: the rerun finds it, the next window ends mid-way to it.
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

/// Verdict of a fast check. Undecided: the caller counts exactly (exact mode, auto choosing the
/// exact counts, or a fast check that could not decide), then calls record_exact().
struct CheckOutcome {
  enum class Verdict { Passed, Missing, Undecided };
  Verdict verdict = Verdict::Undecided;
  index_t missing = 0;  ///< Missing: pairs the deflated rerun should look for (>= 1)
};

/// The prediction behind an auto choice, for the report.
struct CertificateChoice {
  CertificateMethod method = CertificateMethod::Exact;
  CertificateCostEstimate estimate;
  bool kappa_rigorous = true;     ///< randomized considered only with a proven kappa (auto)
  std::string single_reason;      ///< why the single-count rule fails ("" when it holds)
  double epsilon = 0.0;           ///< eps of the randomized estimate (coupling taken as 0)
};

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
  CheckOutcome check(const NearestWindow& w, const Block<T>& all, int round);

  /// Runs one method on this round's window: Randomized (block width from choose()) or
  /// ExactSingle; Exact returns Undecided. Fills the report on a verdict.
  CheckOutcome run(CertificateMethod method, const NearestWindow& w, const Block<T>& all, int round);

  /// Report fields after the exact counts of this round (cert.passed etc. already set by the
  /// caller); on_counter: the counts came from an InertiaCounter.
  void record_exact(bool on_counter);

  /// The auto choice for window w (no side effect besides the start scaling, computed once).
  CertificateChoice choose(const NearestWindow& w, const Block<T>& all);

 private:
  struct Found;
  Found found_in_window(const NearestWindow& w, const Block<T>& all) const;
  CheckOutcome randomized(const NearestWindow& w, const Block<T>& all, index_t block_size, int round);
  CheckOutcome single(const NearestWindow& w, const Block<T>& all);
  const StartScaling& scaling();
  double check_budget_bytes() const;
  std::string prefix() const;

  Ctx& c_;
  const ShiftedPencil<T>& pencil_;
  SparseFactorization<T>& fac_;
  double sigma_;
  bool detection_off_;
  double scale_, n2a_, n2b_;
  CertificateTimings timings_;
  std::unique_ptr<StartScaling> scaling_;
  std::unique_ptr<InertiaCounter<T>> counter_;  ///< single counts (created at the first one)
  bool randomized_failed_ = false;  ///< an earlier round's check could not decide: not again
  bool single_failed_ = false;
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
