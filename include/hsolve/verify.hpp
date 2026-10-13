// Completeness checks that cost solves instead of factorizations (module src/verify, "certfast").
// Scalar set: T in {double, cplx}.
//
// The exact Sylvester certificate (DESIGN §3.6-8) counts nu(A - aB) and nu(A - bB): two extra LDL^T
// factorizations, which on 3-D cases cost as much as the whole Krylov phase. This module offers
// three cheaper building blocks for a driver policy certificate = exact | randomized | auto:
//   (1) exclusion_check: a randomized test on the factorization already held at the shift sigma
//       (block Lanczos on the deflated shift-invert operator from a Gaussian start block), with a
//       rigorous failure probability;
//   (2) single_precision_certificate: Sylvester counts in single precision (InertiaCounter,
//       ~2x cheaper on 3-D cases) where a perturbation argument makes them certify a window;
//   (3) certificate_cost: predicted seconds of the exact counts vs the randomized check, and a
//       recommendation for 'auto'.
// Nothing here prints; every function is deterministic for a fixed seed and thread count.
//
// ======================================================================== (1) exclusion check
// Setting. A x = lambda B x with B Hermitian positive definite (B = I allowed), the factorization
// of K = A - sigma B held, Op = K^{-1} B (B-self-adjoint, eigenvalues theta = 1/(lambda - sigma)).
// X (n x m) holds found eigenvectors (any basis of their span; it is B-orthonormalized here, two
// Cholesky-QR passes). The question: does the pencil have more than the found eigenvalues within
// distance r ("radius") of sigma, i.e. more eigenvalues with |theta| >= 1/r?
//
// Notation. P = I - X X^H B (B-orthogonal projector onto the complement of span X, dimension
// N = n - m). In a B-orthonormal basis [X Y] the operator is M = [[H11, E^H], [E, H22]] with
// H11 = X^H B Op X (its eigenvalues mu_i approximate the found theta_i), H22 = the deflated
// operator P Op P on the complement (what Lanczos with deflation = X iterates on), and the coupling
// e = ||E||_2 = ||Op X - X H11||_B (zero when span X is exactly invariant; computed with one block
// application of Op to X, or supplied by the caller as ExclusionOptions::coupling_bound).
//
// Step 1 (deterministic reduction, Weyl). ||M - diag(H11, H22)||_2 = e, so for every t:
//   #{eigenvalues of M with |theta| >= t} <= #{mu_i : |mu_i| >= t - e} + #{eig(H22) : |.| >= t - e}.
// Hence, with Theta_c = 1/r - e and m_in = #{|mu_i| >= Theta_c} (found_inside): if the pencil has
// more than m_in eigenvalues within r, then H22 has an eigenvalue with |theta| >= Theta_c.
// A missed copy of a found multiple eigenvalue is such an eigenvalue: its eigenvector lies in the
// complement. Conversely (Kahan) at least #{|mu_i| >= 1/r + e} eigenvalues lie within r
// (certified_inside), so the two counts agree unless a found eigenvalue sits within e of the edge.
//
// Step 2 (Krylov lower bound). Block Lanczos with deflation = X from a start block S G (G n x p
// Gaussian, S a positive diagonal start_scaling, default I), s block steps, no restart: its Ritz
// values on the complement satisfy rho = max |Ritz| <= max |eig(H22)| (Cauchy interlacing), and
// the check PASSES iff rho < tau = 1/r' (r' = pass_radius > r). Let Theta = max |eig(H22)| with a
// B-unit eigenvector u, and beta = |<u, b>_B| / ||b||_B for one start column b (projected).
// For eps in (0, 1) the event rho < (1 - eps) Theta implies beta^2 < t*(eps, s), where t* is the
// smaller of two Chebyshev bounds (y = p(Op) b in the Krylov space, Rayleigh quotient argument of
// Kuczynski & Wozniakowski, SIAM J. Matrix Anal. Appl. 13 (1992) 1094-1122, here for an
// indefinite operator):
//   direct   p = T_{s-1}(l(theta)), l: [-Theta, (1-eps) Theta] -> [-1, 1]:
//            t*_1 = (2 - eps) / (eps T_{s-1}(x1)^2 + 2 - eps),           x1 = (2 + eps)/(2 - eps);
//   squared  p = T_k(l(theta^2)) with k = floor((s - 2) / 2), l: [0, (1-eps)^2 Theta^2] -> [-1,1],
//            so that y and Op y lie in the space and max |Ritz| >= ||Op y||_B / ||y||_B:
//            t*_2 = (1-eps)^2 / (e2 T_k(x2)^2 + (1-eps)^2), e2 = 1-(1-eps)^2, x2 = (1+e2)/(1-e2).
// (The squared bound folds both signs of theta into one interval; it is the stronger one for wide
// gaps, eps -> 1, where the direct one saturates at acosh(3) per step.) Multiplicities only help:
// the projection of b onto a multiple eigenspace is at least its component along one fixed u.
//
// Step 3 (probability). In B-orthonormal eigencoordinates of the complement the start column is
// zeta = U^H B S g, Gaussian with covariance Sigma whose eigenvalues lie in the spectrum of S B S,
// so beta^2 = |zeta_1|^2/||zeta||^2 >= |w^H eta|^2/||eta||^2 / kappa with eta standard Gaussian and
// kappa >= cond(S B S). For B = I (b_inner == nullptr) kappa = cond(S^2) = (max s_i / min s_i)^2
// is computed here (1 without S); with b_inner the caller supplies it (b_condition; exactly 1 for
// a diagonal B with S = B^{-1/2}). A small s_i on a missed eigenvector shrinks its start
// component, which is exactly what kappa pays for. |w^H eta|^2/||eta||^2 is the first coordinate
// of a uniform unit vector:
//   real T:    P(z1^2 < x) <= sqrt(N x)       (Beta(1/2, (N-1)/2) density bound, Gautschi);
//   complex T: P(|z1|^2 < x) <= (N - 1) x     (Beta(1, N - 1)).
// The block Krylov space contains each column's Krylov space and the p columns are independent:
//   P(pass and more than m_in eigenvalues within r) <= F(kappa * t*(eps, s))^p =: delta(s),
// with eps = 1 - tau / Theta_eff, Theta_eff = Theta_c (1 - rounding_margin). The number of block
// steps is the smallest s with delta(s) <= ExclusionOptions::delta (exclusion_block_steps), fixed
// BEFORE the run (no data-dependent stopping). When the Krylov space spans the whole complement
// (small N), rho is max |eig(H22)| exactly and delta = 0.
// Example (real, N = 1e5, kappa = 1, p = 1): eps = 0.99 needs s = 8 block steps for 1e-12; eps =
// 0.25 needs s = 28 for 1e-6 and 46 for 1e-12. The gap between 1/r and 1/r' is the lever: tau must
// stay above the next legitimate |theta| (the nearest eigenvalue beyond r), so eps <= 1 - r/d_next.
//
// Rounding. The statement is for the Krylov space of the computed operator in exact arithmetic.
// (a) X is orthonormal only to ~1e-14 even after re-orthonormalization; the solver therefore runs
// two CGS2 passes against it (eigen: Reorth::Always with deflation), leaving components in span X
// of relative size O(delta_X^2 + u). (b) Full reorthogonalization makes the computed Ritz values
// those of an operator within O(u ||Op||) of H22 (Paige); shift-invert solves add a relative error
// of about u * scale / dist(sigma, spectrum). Both are absorbed by rounding_margin (default 1e-6
// relative, versus ~1e-10 for r ~ 1e-6 scale); a radius below ~1e4 u scale is out of scope.
// (c) The random numbers come from linalg::fill_random (SplitMix64, Box-Muller): the probability
// is with respect to an ideal Gaussian start, the usual idealization of randomized algorithms.
//
// Rigour of kappa. For a general B that is not diagonally dominant, start_scaling_for's kappa is
// an estimate (StartScaling::rigorous == false). With b_condition_rigorous = false (what
// set_start_scaling copies) the result is flagged rigorous = false and the statement says that
// kappa is assumed: a screening result, not a certificate. Runs covering the complement involve
// no kappa and stay rigorous.
//
// Cost: one block application of Op to X (unless coupling_bound >= 0) plus s block applications of
// p columns, block CGS2 against [X V] (2 passes), O((s p)^3) for the projected eigenproblem;
// memory about n (3m + s p + 3p + 1) scalars (exclusion_run_size). No factorization. s grows like
// log(N kappa / delta) / sqrt(eps): a narrow gap plans large bases (real, N = 1e5, p = 4,
// delta = 1e-12: eps = 1e-3 -> 1480 vectors, 1.2 GB; eps = 1e-4 -> 4992 vectors, 4 GB). The
// resource budget (max_krylov_vectors, max_memory_bytes) therefore ends such a plan Inconclusive
// BEFORE anything is allocated or applied; the caller falls back to exact counts.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

struct ExclusionOptions {
  /// r' > radius: the check passes iff every Ritz value on the complement has |theta| < 1/r'.
  /// Must stay below the distance of the nearest eigenvalue not in X (else that legitimate
  /// eigenvalue is "detected"); the gap between radius and r' sets eps (see the header comment).
  double pass_radius = 0.0;
  double delta = 1e-12;       ///< target failure probability, in (0, 1)
  index_t block_size = 4;     ///< p >= 1 start vectors (one op.apply of p columns per step)
  std::uint64_t seed = 2026;  ///< Gaussian start block (deterministic)
  /// Optional positive diagonal S (n entries): start columns S g. With B diagonal pass
  /// S = B^{-1/2} (start_scaling_for) and kappa = 1. Without b_inner (B = I) the check charges
  /// kappa = (max s_i / min s_i)^2 = cond(S^2) itself. Call-scoped pointer. set_start_scaling
  /// sets this, b_condition and b_condition_rigorous from a StartScaling.
  const std::vector<double>* start_scaling = nullptr;
  /// With b_inner: kappa >= cond(S B S), an upper bound the caller vouches for (start_scaling_for
  /// gives S and kappa). Required (>= 1): the default 0 makes forgetting it an error rather than
  /// a silently optimistic statement. Without b_inner: optional (0 = unset, else >= 1); kappa =
  /// max(cond(S^2), b_condition), cond(S^2) computed here.
  double b_condition = 0.0;
  /// With b_inner: false when b_condition is an estimate, not a proven bound (StartScaling::
  /// rigorous == false, e.g. a general non-diagonally-dominant B). The result is then flagged
  /// rigorous = false and its statement says that kappa is assumed. Without b_inner kappa is
  /// computed exactly and this is ignored.
  bool b_condition_rigorous = true;
  /// >= 0: caller's bound on e = ||Op X - X H11||_B (2-norm), e.g. sqrt(sum of the Lanczos
  /// estimates^2) of the Ritz vectors spanning X; saves the block application of Op to X.
  /// < 0 (default): computed (rigorous path).
  double coupling_bound = -1.0;
  /// Required with coupling_bound >= 0 (one value per column of x), ignored otherwise: the
  /// eigenvalues of H11 = X^H B Op X, e.g. the Ritz values theta_i of the Lanczos run whose Ritz
  /// vectors span X (then H11 = diag(theta_i) in exact arithmetic), or 1/(lambda_i - sigma) for
  /// the pencil's Rayleigh-Ritz values on span X (equal to O(e^2 / gap)). Call-scoped pointer.
  const std::vector<double>* deflated_theta = nullptr;
  double rounding_margin = 1e-6;  ///< relative safety margin on 1/r (see Rounding), in [0, 0.5)
  index_t max_block_steps = 0;    ///< cap on s (0: none); a capped run may end Inconclusive
  /// Resource budget of the Krylov run, checked on the plan BEFORE anything is allocated or
  /// applied: a plan (s from delta, after max_block_steps) whose run (exclusion_run_size) needs
  /// more Krylov vectors or more bytes ends Inconclusive with no operator application, and the
  /// caller falls back to exact counts. 0 = no limit. The defaults keep a narrow gap from
  /// allocating gigabytes (or a full-space run for a small N) on the 16/32 GB machine; at N = 1e5
  /// (delta = 1e-12, kappa = 1) they admit eps >= 0.01 for p = 4 and 0.005 for p = 1 (the
  /// measured data cases plan at most 144 vectors, eps >= 0.07); at N = 1e6 the bytes bind
  /// first (real: eps >= 0.1 for p = 1). A driver should derive max_memory_bytes from its own
  /// memory budget (what the factorization leaves of SolverConfig::memory_budget_gb).
  index_t max_krylov_vectors = 512;
  double max_memory_bytes = 1073741824.0;  ///< 1 GiB
  /// Ritz vectors returned on detection (the best by |theta|; new_count still counts all). 1 keeps
  /// the block layout smallest (3 block steps at least, against 4 for p of them).
  index_t max_new_pairs = 1;
  double tol = 1e-10;             ///< a returned pair is flagged converged when est <= tol |theta|
};

enum class ExclusionStatus {
  Passed,       ///< no Ritz value with |theta| >= 1/r'; the probability statement holds
  Detected,     ///< some Ritz value has |theta| >= 1/r' (a missing pair, or one in (r, r'])
  Inconclusive  ///< no statement: eps <= 0 (gap too small / coupling too large), step cap hit,
                ///< plan beyond the resource budget, or kappa not finite
};

const char* exclusion_status_name(ExclusionStatus s);

template <class T>
struct ExclusionResult {
  ExclusionStatus status = ExclusionStatus::Inconclusive;
  double sigma = 0.0, radius = 0.0, pass_radius = 0.0;
  double claim_threshold = 0.0;  ///< Theta_eff = (1/r - e)(1 - rounding_margin)
  double pass_threshold = 0.0;   ///< tau = 1/r'
  double coupling = 0.0;         ///< e used (computed or supplied, plus a rounding guard)
  double epsilon = 0.0;          ///< 1 - tau / Theta_eff (<= 0: no statement possible)
  double kappa = 1.0;            ///< kappa of the statement (see ExclusionOptions::b_condition)
  /// false when the probability statement rests on a kappa that is only an estimate
  /// (b_condition_rigorous == false and the run did not cover the complement): a Passed result is
  /// then a screening result, not a certificate (the statement says so).
  bool rigorous = true;
  index_t deflated = 0;          ///< m = X.cols()
  index_t dimension = 0;         ///< N = n - m
  std::vector<double> deflated_theta;  ///< eigenvalues mu_i of H11, by decreasing |mu|
  index_t found_inside = 0;      ///< m_in = #{|mu_i| >= Theta_c}: the bound of the statement
  index_t certified_inside = 0;  ///< #{|mu_i| >= 1/r + e}: eigenvalues certainly within r
  index_t block_size = 0;        ///< p
  index_t block_steps_planned = 0;  ///< s from delta (0 when no run)
  index_t krylov_vectors_planned = 0;  ///< exclusion_run_size of the plan (0 when no plan)
  double memory_bytes_planned = 0.0;
  bool over_budget = false;      ///< the plan exceeded the resource budget: no run (Inconclusive)
  index_t block_steps = 0;       ///< block steps performed (Krylov space K_s)
  bool exact = false;            ///< the Krylov space spanned the whole complement (delta = 0)
  double max_ritz = 0.0;         ///< rho = max |Ritz value| on the complement
  std::vector<double> ritz_all;  ///< all Ritz values of the run, ascending
  double failure_probability = 1.0;  ///< delta(s) achieved; 0 if exact; 1 without a statement
  std::string statement;         ///< the probability statement in words
  /// Ritz values with |theta| >= tau: count in ritz_all, of which within r (|theta| >= 1/r).
  index_t new_count = 0;
  index_t new_inside = 0;
  /// The returned candidate pairs (|theta| >= tau among the max_new_pairs best by |theta|):
  /// lambda = sigma + 1/theta, residual estimates ||P(Op x - theta x)||_B, converged flags.
  std::vector<double> new_theta, new_lambda, new_estimate;
  std::vector<char> new_converged;
  DenseMatrix<T> new_vectors;    ///< n x (#returned), B-orthonormal, B-orthogonal to X
  index_t op_applications = 0;   ///< single-vector applications of op (coupling + Krylov)
  index_t b_applications = 0;    ///< single-vector applications of b_inner
  double setup_s = 0.0;          ///< orthonormalization of X and the coupling
  double krylov_s = 0.0;         ///< the block Lanczos run
};

/// delta(s) of the header comment: bound on P(pass and more than m_in eigenvalues within r) for
/// relative gap eps in (0, 1) (<= 0: returns 1), s >= 1 block steps, complement dimension N,
/// block size p, kappa >= 1, real or complex scalars. O(1); monotone non-increasing in s and eps.
double exclusion_failure_bound(double epsilon, index_t block_steps, index_t dimension,
                               index_t block_size, double kappa, bool complex_scalars);

/// Smallest s >= 1 with exclusion_failure_bound(...) <= delta; returns
/// ceil(dimension / block_size) + 1 (a space covering the complement, exact) when that is
/// smaller, and -1 for eps <= 0 or delta outside (0, 1). The bound is monotone in s: doubling
/// then bisection, O(log s) evaluations of the bound.
index_t exclusion_block_steps(double epsilon, double delta, index_t dimension, index_t block_size,
                              double kappa, bool complex_scalars);

/// Size of the Krylov run of exclusion_check for s planned block steps (the layout of
/// lanczos_krylov_schur, eigen.hpp: ncv = s p, at least 3p in block mode with one returned pair;
/// a free space N below twice the block basis plus its residual block, or s p >= N, is searched by
/// one single-vector cycle over the whole complement).
struct ExclusionRunSize {
  index_t krylov_vectors = 0;  ///< Krylov basis vectors (N when the run covers the complement)
  bool covers = false;         ///< the run spans the whole complement (exact, delta = 0)
  /// Peak memory of the run: n (3m [X, B X, the solver's copy of X] + krylov_vectors + 1 or p
  /// [residual block] + 2p [start block, B W workspace] + max_new_pairs [returned vectors])
  /// scalars + 3 krylov_vectors^2 scalars (projected matrices).
  double bytes = 0.0;
};
ExclusionRunSize exclusion_run_size(index_t n, index_t deflated, index_t block_steps,
                                    index_t block_size, index_t max_new_pairs,
                                    bool complex_scalars);

/// The randomized exclusion check (header comment). op = (A - sigma B)^{-1} B, self-adjoint and
/// positive-definite-inner-product-compatible with b_inner = B (nullptr: B = I); x = n x m found
/// eigenvectors (m >= 0, full column rank; B-orthonormalized internally, x is not modified);
/// radius r > 0 (the claim: no eigenvalue beyond the found ones within r of sigma);
/// opt.pass_radius > r. x may hold any full-rank basis of the found span, with columns of any
/// relative scaling (the rank test sees the column-equilibrated X: cond_B(X D^{-1/2}) above ~1e6 is
/// rejected). Throws InvalidInput for invalid sizes/options, a rank-deficient x, or a non-finite
/// operator output. Cost and memory: header comment; a plan beyond opt's resource budget returns
/// Inconclusive before the Krylov run. Not thread-safe for a shared op.
template <class T>
ExclusionResult<T> exclusion_check(LinearOperator<T>& op, LinearOperator<T>* b_inner,
                                   const DenseMatrix<T>& x, double sigma, double radius,
                                   const ExclusionOptions& opt);

/// Start scaling S and the bound kappa >= cond(S B S) for ExclusionOptions.
struct StartScaling {
  std::vector<double> scaling;  ///< empty: S = I
  double kappa = 1.0;           ///< rigorous bound when `rigorous`, else an estimate (see below)
  bool rigorous = true;
  std::string note;
};

/// Copies sc into opt: start_scaling = &sc.scaling (empty: none; sc must outlive the check),
/// b_condition = sc.kappa, b_condition_rigorous = sc.rigorous.
void set_start_scaling(ExclusionOptions& opt, const StartScaling& sc);

/// B == nullptr: S = I, kappa = 1. B diagonal (positive): S = B^{-1/2}, kappa = 1 (exact).
/// Otherwise Jacobi scaling S = diag(B)^{-1/2}; kappa = Gershgorin upper / lower bound of S B S
/// when that lower bound is positive (diagonally dominant: rigorous), else an ESTIMATE:
/// safety * (Gershgorin upper bound) / (smallest Ritz value of a 40-step Lanczos run on S B S) -
/// an assumption, not a bound (the Ritz value over-estimates lambda_min). kappa enters the step
/// count only as log(kappa) / (2 acosh(.)), so a generous safety factor is cheap. estimate =
/// false skips that run: such a B then gets kappa = +inf, rigorous = false (for a caller that
/// only wants a proven kappa, e.g. the driver's auto choice; ~0.1 s at n = 1e5).
/// Throws InvalidInput for a non-positive diagonal. O(nnz(B)) (+ 40 B applications).
template <class T>
StartScaling start_scaling_for(const CsrMatrix<T>* b, double safety = 100.0, bool estimate = true);

// ===================================================== (2) single-precision screening counts
// A single-precision LDL^T count of K(s) = A - s B is the exact inertia of K(s) + E with
// ||E||_2 <= c g u_s ||K(s)||_1 (u_s = 2^-24 unit roundoff, g the element growth of MUMPS'
// threshold pivoting, c a modest constant; the float rounding of the entries is included). For
// B = I or a diagonal B > 0 the eigenvalues of (A + E, B) are within eta = ||E||_2 / min b_ii of
// those of (A, B) (Weyl in the B^{-1/2} congruence), hence, with valid single counts at a < b:
//   #{eigenvalues in [a + eta_a, b - eta_b)} <= nu_s(b) - nu_s(a).
// If moreover every found eigenvalue (with its error bar) lies in [a + eta_a, b - eta_b) and
// nu_s(b) - nu_s(a) equals the number of found pairs, the found pairs are ALL the eigenvalues in
// that shrunken window, which must contain the wanted region. Unknown eigenvalues within eta of an
// end are outside the shrunken window, so no gap assumption about them is needed.
// The one non-rigorous input is c g: MUMPS' threshold pivoting has no useful a priori growth
// bound. SingleCountRule::backward_factor = c g defaults to 1e3, i.e. eta ~ 6e-5 ||K||_1 / b_min,
// against the largest miscount distance measured on the certpar test set, 5e-6 (scale + |s|)
// (factor.hpp, InertiaCounter). A later measurement violated the default: on yee-pec-box-1e5 (16
// threads, 1 of ~50 processes) single counts were off by 2 at ends 2.96e3 or more u_s ||K||_1 /
// b_min away from every eigenvalue - c g is an assumption with no a priori bound, which callers
// must guard (the driver: a 10x margin in auto, and double counts on any count above the found
// pairs; src/driver/certificate.hpp). For a general (non-diagonal) B no cheap rigorous bound on
// lambda_min(B) exists: the rule does not apply and no certificate is claimed.

struct SingleCountRule {
  double backward_factor = 1e3;  ///< assumed c g (see above)
  /// ||A||_1 of the pencil when the caller has it (>= 0), else computed (O(nnz(A)); a complex A's
  /// |a_ij| are hypot calls: ~10 ms per 1e6 entries, which a caller checking several windows of
  /// one pencil should not pay each time).
  double a_norm_1 = -1.0;
};

/// One certificate window for single counts.
struct SingleWindow {
  double lower = 0.0, upper = 0.0;                ///< count points a < b
  double wanted_lower = 0.0, wanted_upper = 0.0;  ///< region to certify, a < wl <= wu < b
  std::vector<double> found;        ///< eigenvalues of the found distinct pairs counted in [a, b]
  std::vector<double> found_error;  ///< their error bars (empty: 0), e.g. residual bounds
};

struct SingleCountValidity {
  bool applicable = false;  ///< B = I or B diagonal with min b_ii > 0
  bool margins_ok = false;  ///< [a + eta_a, b - eta_b) holds the wanted region and every found
                            ///< eigenvalue +- its error bar
  double eta_lower = 0.0, eta_upper = 0.0;  ///< eigenvalue perturbation bounds at a and b
  double b_min = 0.0;       ///< min b_ii (1 for B = I)
  std::string reason;       ///< why not valid ("" when valid)
  bool valid() const { return applicable && margins_ok; }
};

/// The rule alone (no factorization): O(nnz(A) + nnz(B) + #found), without nnz(A) when
/// rule.a_norm_1 is given. Throws InvalidInput for a malformed window (a >= b, wanted region
/// outside (a, b), found_error size mismatch) or a non-finite a_norm_1.
template <class T>
SingleCountValidity single_count_validity(const ShiftedPencil<T>& pencil, const SingleWindow& w,
                                          const SingleCountRule& rule = {});

struct SingleCertificate {
  SingleCountValidity validity;
  Inertia lower, upper;  ///< single-precision counts (only when validity.valid())
  index_t count = -1;    ///< nu_s(b) - nu_s(a), -1 when not counted
  bool certified = false;
  double seconds = 0.0;  ///< time of the two counts
};

/// Counts nu(K(a)), nu(K(b)) in single precision with `counter` (on `pencil`) only when the rule
/// holds; certified = valid rule && both counts valid && count == found.size(). Otherwise the
/// caller must fall back to double counts: never a certificate where the rule fails.
template <class T>
SingleCertificate single_precision_certificate(InertiaCounter<T>& counter,
                                               const ShiftedPencil<T>& pencil,
                                               const SingleWindow& w,
                                               const SingleCountRule& rule = {});

// ======================================================================= (3) cost model
// Predicted seconds of each completeness check from quantities the driver has measured by then:
// the main factorization's time and the per-block operator application time of its Krylov run.
//   exact      2 counts: min(2 t_factor [refactorize the main object, today's driver],
//              t_counter_analysis + 2 count_ratio t_factor [InertiaCounter])
//   single     t_counter_analysis + 2 count_ratio single_ratio t_factor (only when the rule holds)
//   randomized t_apply(m) [coupling, unless known] + s(p) t_apply(p) + CGS2 sweeps, with
//              t_apply(q) = t_apply1 (1 + rhs_ratio (q - 1)) and s(p) = exclusion_block_steps;
//              the block size p in {1, 2, 4, 8} minimizing the total among those whose plan fits
//              the check's resource budget is chosen (+inf when none fits).
// Default ratios measured at 2 threads (bench/certpar_inertia.cpp, bench/certfast_measure.cpp).
//
// Measured (bench/certfast_measure.cpp: 2 threads on cores 30,33 of the shared machine, X =
// hsolve's own solution via the CLI, the driver's ordering and window, medians of 5 interleaved
// repetitions; seconds). exact: driver path (refactorize the main object at both window ends) /
// InertiaCounter double pair / single pair (whether the rule of (2) certifies it). randomized:
// the driver's pass radius r' = the half-width of its certificate window (ends mid-gap to the
// next Ritz hint, the nearer side; eps follows), the default p = 4 [p = 1], including the block
// application of Op to X. d/r: distance of the next Ritz hint over the claim radius.
// case                            k d/r  B    eps   exact drv/cnt/single    rand 1e-6/1e-12 [p=1]
// syn-mem27-46                   12 387  I    0.995 3.10/2.86/1.62 cert     0.50/0.50 [0.50/0.59]
// yee-pec-cube-32                12 4.0  I    0.60  2.80/2.56/1.37 cert     1.04/1.03 [1.07/1.93]
// yee-pec-box-1e5                10 1.36 diag 0.15  3.12/2.76/1.59 cert     2.31/3.31 [2.64/4.58]
// ss-qa8fk-qa8fm                 10 1.34 gen* 0.14  1.34/1.10/0.64 not cert 2.03/2.61 [1.97/2.95]
// syn-multiplicity-real-1e4      14 1.16 I    0.073 0.025/0.024/invalid     0.13/0.18 [0.09/0.17]
// syn-multiplicity-complex-1e4   14 1.16 I    0.073 0.081/0.075/invalid     0.27/0.34 [0.17/0.26]
// syn-ghep-multiplicity-real-1e4 22 6.0  gen* 0.71  0.027/0.027/not cert    0.07/0.08 [0.05/0.06]
// gen*: a general B whose kappa start_scaling_for only estimates (2.6e3, 1.3e4): the randomized
// result there is a screening result, not a certificate (ExclusionResult::rigorous = false).
// Every complete X passed (no false alarm) and every X with one copy of the largest found
// multiplet removed (6-, 6-, 2-, -, 12-, 12-, 12-fold; qa8 has no multiple found eigenvalue) or
// the found eigenvalue farthest from sigma removed was detected, in every repetition, delta and p
// (the new Ritz value within 4e-12 of the removed eigenvalue). Against the fastest exact option
// that certifies (single counts where their rule holds, else the counter pair) the check wins on
// syn-mem27-46 (3.3x at both deltas; 5.8x against the counter pair) and yee-pec-cube-32 (1.3x),
// and loses everywhere else: on yee-pec-box-1e5 (narrow gap: eps 0.15, 20-26 block steps of 4)
// it is 1.5x / 2.1x slower than the certified single counts and at 1e-12 slower even than the
// double counter pair (3.31 vs 2.76 s); the 1e4 cases have cheap factorizations. So the check pays
// off only where factorizations are expensive (3-D fill) AND the gap d/r is wide. Against the
// earlier 3-repetition run of the same bench the medians moved by up to 26% (shared machine),
// not the ordering.
// The bench's second pass radius "hint", r' = r + 0.9 (d_hint - r), is an experiment, not what
// the driver does: it widens eps (box 0.24, qa8 0.23) and measured 1.86 / 2.55 s on
// yee-pec-box-1e5 and 1.57 / 1.96 s on ss-qa8fk-qa8fm (p = 4), up to 38% faster on the 1e4 cases
// and within 5% on syn-mem27-46 and yee-pec-cube-32 - still slower than the single counts on the
// box. A Ritz value under-estimates |theta| (Cauchy interlacing), so d_hint >= d_next and an
// unconverged hint can put r' beyond the next eigenvalue, which is then Detected: a false alarm
// costing the check plus the exact fallback. It passed on all seven X here; nothing more is
// validated.
// certificate_cost with the measured inputs (window policy) recommended the measured fastest
// certifying method in all seven cases; its randomized estimate was within 26% of the measurement
// on the four 1e5 cases and up to 55% above it on the 1e4 cases (not changing the choice).

enum class CertificateMethod { Exact, ExactSingle, Randomized };
const char* certificate_method_name(CertificateMethod m);

struct CertificateCostInputs {
  index_t n = 0;
  index_t deflated = 0;          ///< m
  bool complex_scalars = false;
  double factorize_s = 0.0;      ///< one double numeric factorization of K (measured)
  double counter_analysis_s = 0.0;  ///< InertiaCounter analysis (PERM_IN), ~AMD analysis time
  /// Analyses of the single-count path; < 0: counter_analysis_s. A fresh InertiaCounter analyses
  /// its double instance at construction and its single one at the first single count (two).
  double single_analysis_s = -1.0;
  double count_ratio = 0.9;      ///< double count / factorize_s (InertiaCounter; measured 0.87-0.94)
  double single_ratio = 0.55;    ///< single count / double count (3-D: 0.54-0.61; low fill: ~1)
  bool single_valid = false;     ///< single_count_validity holds
  double apply1_s = 0.0;         ///< one single-vector application of Op (solve + B product)
  double rhs_ratio = 0.12;       ///< marginal cost of an extra right-hand side in a block solve
                                 ///< (measured 0.07-0.14, MUMPS, 2 threads)
  double stream_bytes_per_s = 8e9;  ///< CGS2 sweep bandwidth
  bool coupling_known = false;   ///< e supplied: no application of Op to X
  double epsilon = 0.0;          ///< relative gap of the exclusion check (ExclusionResult)
  double delta = 1e-12;
  double kappa = 1.0;
  /// The check's resource budget (ExclusionOptions, same defaults): a block size whose plan
  /// exceeds it is not considered (the check would end Inconclusive without a run).
  index_t max_krylov_vectors = ExclusionOptions{}.max_krylov_vectors;
  double max_memory_bytes = ExclusionOptions{}.max_memory_bytes;
  /// Randomized is recommended only when cheaper than the best exact option by this factor.
  double preference = 1.2;
};

struct CertificateCostEstimate {
  double exact_s = 0.0;
  double exact_single_s = 0.0;  ///< +inf when the single rule does not hold
  double randomized_s = 0.0;    ///< +inf when eps <= 0 or no plan fits the resource budget
  index_t block_size = 0;       ///< p of the randomized estimate
  index_t block_steps = 0;
  double failure_bound = 1.0;
  CertificateMethod recommended = CertificateMethod::Exact;
};

/// O(1). Throws InvalidInput for negative times / sizes.
CertificateCostEstimate certificate_cost(const CertificateCostInputs& in);

}  // namespace hs
