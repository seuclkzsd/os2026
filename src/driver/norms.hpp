// Spectral-norm and spectrum-end estimation for Hermitian sparse matrices and pencils
// (driver-internal).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs::driver {

struct NormEstimate {
  double norm2 = 0.0;      ///< max |Ritz value|: a LOWER bound of ||A||_2 (conservative eta)
  double ritz_min = 0.0;   ///< smallest Ritz value (estimate of lambda_min, from inside)
  double ritz_max = 0.0;   ///< largest Ritz value (estimate of lambda_max, from inside)
  double ritz_min2 = 0.0;  ///< second smallest Ritz value (== ritz_min after 1 step)
  double ritz_max2 = 0.0;  ///< second largest Ritz value (== ritz_max after 1 step)
  int steps = 0;           ///< Lanczos steps actually taken (fewer on breakdown)
};

/// Plain three-term Lanczos (no reorthogonalization, O(n) memory) on a Hermitian matrix.
/// Without reorthogonalization the Ritz values may contain ghost copies, but they still lie in
/// [lambda_min, lambda_max] up to O(eps ||A||) (Paige), so max |Ritz| is a lower bound of
/// ||A||_2 that is typically within a few percent after 30 steps. Cost: steps SpMVs.
template <class T>
NormEstimate estimate_norm2(const CsrMatrix<T>& a, int steps = 30, std::uint64_t seed = 2026);

/// In-place solve with B: x <- B^{-1} x for one vector.
template <class T>
using SolveFn = std::function<void(T* x)>;

/// The same Lanczos for the pencil (A, B), B Hermitian positive definite: Op = B^{-1} A is
/// self-adjoint in the B inner product, its Ritz values lie in [lambda_min, lambda_max] of the
/// pencil. Each step costs one A product, one B product and one solve_b. norm2 is max |Ritz|.
template <class T>
NormEstimate estimate_pencil_spectrum(const CsrMatrix<T>& a, const CsrMatrix<T>& b, const SolveFn<T>& solve_b,
                                      int steps = 30, std::uint64_t seed = 2026);

// ------------------------------------------------------------------ positive definiteness proof
//
// A proof that a Hermitian B is positive definite without its exact factorization (DESIGN §3.6-1,
// the B test of hs::solve). With D = diag(B) > 0 and S = D^{-1/2} B D^{-1/2} (unit diagonal), any
// lower triangular L splits S = L L^H + M, and L L^H is positive semidefinite whatever L is, so
//   lambda_min(B) > 0  <=  lambda_min(S) >= lambda_min(M) >= min_i (m_ii - sum_{j != i} |m_ij|) > 0
// (Weyl, then Gershgorin on M = S - L L^H). L is an incomplete Cholesky factor of S - c I on a
// fill level (0: the pattern of B, 1: one level of fill): M is then c I plus the dropped fill (the
// entries of L L^H outside the pattern) plus rounding, and the Gershgorin test holds when the
// dropped fill of every row sums to less than c. The entries of M are computed from the stored
// floating-point L and the scaled entries with a rigorous a-priori bound on every rounding error
// (Higham, Accuracy and Stability, 2nd ed., Lemma 3.1 / §3.1: dot products and sums, with a factor
// 2 of slack; complex products), so a positive bound is a proof for the stored B, not an
// estimate - unlike an inertia count, it does not rest on the growth factor of a pivoted
// factorization. A failed proof says nothing about B (the caller then counts its inertia).
// Lanczos on S (no reorthogonalization; S applied as w .* (B (w .* x)), w = diag(B)^{-1/2}, so
// that S is stored only when an attempt follows) sets the shifts: its smallest Ritz value theta
// lies above lambda_min(S) (up to rounding), and one below -kIndefiniteRitzScaled proves S - hence
// B - indefinite (Paige: Ritz values stay inside the spectrum's hull up to O(eps ||S||_2)).

/// Options of prove_positive_definite.
struct HpdProofOptions {
  int lanczos_steps = 20;  ///< steps of the Lanczos estimate theta of lambda_min(S)
  /// Fill levels (0 or 1) in order of preference: a later one is tried only when the earlier one
  /// does not fit the time or memory limit (norms.cpp: why level 1 first).
  std::vector<int> levels{1, 0};
  /// Uniform shift c = fraction * theta of a level's first attempt, and the smaller fractions
  /// tried after a breakdown (a pivot of the incomplete factorization not positive).
  std::vector<double> fractions{0.9, 0.5, 0.2};
  /// Time budget (seconds, predicted by hpd_attempt_seconds before each attempt): an attempt beyond
  /// what is left is not run (outcome too_expensive). The driver passes a fraction of the predicted
  /// time of the factorization of B that a proof saves.
  double max_seconds = 1e300;
  /// Memory limit (bytes) of an attempt beyond the stored lower triangle of S (IcProver: pattern,
  /// factor, workspaces); a level beyond it is not tried (outcome too_large).
  double max_bytes = 1e300;
  /// No attempt when theta < this. The check needs the fill an attempt drops below the shift
  /// c < theta in every row; with theta < 0.1 that held on none of the data cases (bench/
  /// anaint_bproof.cpp: syn-ghep-multiplicity-real-1e4 0.043, syn-ghep-near-singular-real-1e5
  /// 0.035, ss-tem152078 0.072 failed at every level and shift, ss-tem152078's attempts costing
  /// 3.7x the factorization of B), while every proven case had theta >= 0.14.
  double min_theta = 0.1;
  std::uint64_t seed = 2026;
};

/// Size of one attempt on a fill level: predicted multiply-adds of the incomplete factorization
/// (parallel over the rows of a level) and of the check (parallel over rows), and the levels.
struct IcWork {
  double factor = 0.0;
  double check = 0.0;
  index_t levels = 0;
};

/// One attempt of the proof: incomplete Cholesky factor of S - c I on a fill level, then the
/// rigorous Gershgorin bound of lambda_min(S - L L^H).
struct IcBound {
  bool breakdown = false;     ///< a pivot of the incomplete factorization was not positive
  double lower_bound = -1.0;  ///< rigorous unless breakdown: lambda_min(S) >= lower_bound
  index_t failing_rows = 0;   ///< rows whose Gershgorin margin is not positive
  index_t factor_entries = 0; ///< entries of the incomplete factor (lower triangle with diagonal)
  IcWork work;
  double factor_s = 0.0, check_s = 0.0;
};

struct HpdProof {
  bool proven = false;         ///< B is Hermitian positive definite (rigorous)
  bool indefinite = false;     ///< the Lanczos run on S proved a negative eigenvalue
  double lower_bound = 0.0;    ///< proven: lambda_min(D^{-1/2} B D^{-1/2}) >= lower_bound > 0
  double theta_min = 0.0;      ///< Lanczos estimates (from inside) of the ends of the spectrum of S
  double theta_max = 0.0;
  int level = -1;              ///< fill level of the last attempt (-1: none)
  double shift = 0.0;          ///< its uniform shift c
  int attempts = 0;            ///< incomplete factorizations computed
  index_t factor_entries = 0;  ///< of the last attempt
  double seconds = 0.0;        ///< all of it; the parts:
  double lanczos_s = 0.0;      ///< the diagonal check and the Lanczos run on S
  double setup_s = 0.0;        ///< the stored S (rows and columns; only when theta >= min_theta)
  double pattern_s = 0.0;      ///< the level-1 pattern (when built)
  double attempts_s = 0.0;     ///< the attempts (factorizations and checks)
  /// "proven", "indefinite", "nonpositive_diagonal", "nearly_singular" (theta < min_theta),
  /// "breakdown" / "gershgorin" / "too_expensive" / "too_large" (why the last attempt did not
  /// prove it).
  std::string outcome = "none";
};

/// A Ritz value of the Lanczos run on S below -kIndefiniteRitzScaled * max|Ritz| proves S (hence
/// B) indefinite (same rule as kIndefiniteRitz for B itself, internal.hpp).
constexpr double kIndefiniteRitzScaled = 1e-10;

/// The proof above for an exactly Hermitian b (as hs::solve's working copies are: the attempts
/// read its lower triangle, the Lanczos run all of it). O(lanczos_steps nnz(b)), then S and the
/// level-1 pattern (parallel over rows) and per attempt the incomplete factorization (parallel over
/// the rows of a level) and the check (parallel over rows); per-thread workspaces of ~5 n scalars,
/// threads reduced to keep them under 256 MB. The outcome is a deterministic function of b, the
/// options and the thread count (the budget's time model divides the work by the threads; the
/// bound itself does not depend on them). Does not modify b.
template <class T>
HpdProof prove_positive_definite(const CsrMatrix<T>& b, const HpdProofOptions& opt = {});

/// Predicted seconds of one attempt on an order-n matrix of the given size on `threads` threads
/// (model fitted with bench/anaint_bproof.cpp, see norms.cpp).
double hpd_attempt_seconds(index_t n, const IcWork& w, int threads);

/// One attempt with a uniform shift (exposed for tests and bench/anaint_bproof.cpp): level 0 or 1,
/// shift c >= 0 (absolute: S has a unit diagonal). Requires a positive diagonal (InvalidInput).
template <class T>
IcBound incomplete_cholesky_bound(const CsrMatrix<T>& b, int level, double shift);

}  // namespace hs::driver
