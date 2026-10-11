// Eigensolvers that work on the pencil (A, B) itself instead of a spectral transformation
// (module src/eigen):
//   * lobpcg          - Knyazev's locally optimal block preconditioned conjugate gradient method
//                       for the algebraically smallest / largest eigenpairs. Needs only products
//                       with A, B and (optionally) a preconditioner: the factorization-free path.
//   * jacobi_davidson - Hermitian Jacobi-Davidson for the eigenpairs nearest a target tau
//                       (interior or exterior). The correction equation is solved approximately
//                       with a user preconditioner, e.g. a SINGLE-precision factorization of
//                       A - tau B: the outer residual is computed in double precision, so the
//                       eigenpairs are double-precision accurate although the factor is not.
// Scalar set: T in {double, cplx}. Depends only on linear_operator.hpp, eigen.hpp and src/linalg.
//
// Operators. A must be Hermitian, B (nullptr = identity) Hermitian positive definite; neither is
// checked (a non-Hermitian A or an indefinite B gives meaningless results, B^H B-type failures
// show up as InvalidInput when a B-norm is not positive). The preconditioner is any linear
// operator approximating an inverse of A - s B for a shift s near the wanted eigenvalues
// (LOBPCG: preferably Hermitian definite; JD: any, see JdInnerSolver). All operators are applied
// to blocks; none is retained after the call. Not thread-safe with respect to the operators.
//
// Convergence test (both solvers). A pair (theta, x) is converged when its backward error
//     eta = ||A x - theta B x||_2 / ((norm_a + |theta| norm_b) ||x||_2)
// is <= tol, with norm_a ~ ||A||_2, norm_b ~ ||B||_2 taken from the options or, when 0, estimated
// as the largest |Ritz value| of 30 Lanczos steps (a lower bound, so eta is conservative), i.e.
// exactly the acceptance test of the driver (docs/DESIGN.md §4.5). Both solvers recompute the
// residuals of the returned vectors from fresh products with A and B before returning, so
// `estimate` is the true backward error, not a recurrence value.
//
// Result contract (shared with include/hsolve/eigen.hpp), m = theta.size():
//   * theta, vectors (columns), estimate, residual refer to the same pairs index by index.
//   * [0, nconv) are the converged pairs (estimate <= tol), [nconv, m) the unconverged ones, both
//     groups in ranking order (lobpcg: by `which`; jacobi_davidson: by |theta - tau|, nearest
//     first). converged == (nconv >= nev).
//   * vectors are B-orthonormal (orthonormal when B == nullptr).
//   * Neither solver throws on non-convergence (converged = false). Invalid options throw
//     InvalidInput; so does an operator output with Inf/NaN.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/linear_operator.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Common result of lobpcg and jacobi_davidson (contract above).
template <class T>
struct IterativeEigResult {
  std::vector<double> theta;     ///< m eigenvalue approximations of (A, B), ranking order per group
  DenseMatrix<T> vectors;        ///< n x m, B-orthonormal
  std::vector<double> estimate;  ///< backward errors eta (see the convergence test above)
  std::vector<double> residual;  ///< ||A x - theta B x||_2 (fresh products)
  index_t nconv = 0;
  index_t iterations = 0;        ///< lobpcg: block iterations; jacobi_davidson: outer iterations
  index_t op_applies = 0;        ///< single-vector applications of A (incl. norm estimation)
  index_t b_applies = 0;         ///< single-vector applications of B (0 when B == nullptr)
  index_t precond_applies = 0;   ///< single-vector applications of the preconditioner
  index_t inner_iterations = 0;  ///< jacobi_davidson: total inner (correction) iterations
  index_t fallback_expansions = 0;  ///< jacobi_davidson: expansions by the residual or a random
                                    ///< vector because the correction was inside span[Q V]
  index_t tracking_steps = 0;    ///< jacobi_davidson: correction equations shifted by tau, not theta
  index_t restarts = 0;          ///< lobpcg: iterations whose P block was dropped (ill-conditioned
                                 ///< basis); jacobi_davidson: thick restarts of the search space
  double norm_a = 0.0;           ///< the norms used in the backward error
  double norm_b = 1.0;
  bool converged = false;
  /// One record per iteration: restart = iteration index, nconv = converged wanted pairs,
  /// max_estimate = largest backward error among the wanted unconverged pairs (lobpcg) or of the
  /// current approximation (jacobi_davidson), op_applies = cumulative A applications.
  std::vector<RestartRecord> history;
};

// ------------------------------------------------------------------------------------- LOBPCG

/// Options of lobpcg. Pointer members are call-scoped: read during the call, never retained.
template <class T>
struct LobpcgOptions {
  index_t nev = 6;           ///< wanted pairs, 1 <= nev <= block_size
  /// Block size m (columns of X). 0 = auto: min(n - d, nev + max(2, min(nev, 8))). Extra columns
  /// beyond nev are guard vectors that speed up the convergence of the wanted ones.
  index_t block_size = 0;
  double tol = 1e-10;        ///< backward-error tolerance (see the convergence test)
  index_t max_iterations = 1000;
  std::uint64_t seed = 2026;  ///< deterministic random start columns (linalg::fill_random)
  /// Optional start block (n x p, 1 <= p <= block_size); the remaining columns are random.
  const DenseMatrix<T>* initial = nullptr;
  /// Optional constraint block Y (n x d), B-orthonormal (|Y^H B Y - I| <= 1e-8): the iteration
  /// stays in the B-orthogonal complement of span(Y) ("deflation" of known eigenvectors). The
  /// residuals are full residuals, so pass Y accurate to about tol.
  const DenseMatrix<T>* deflation = nullptr;
  double norm_a = 0.0;  ///< ||A||_2 for the backward error; 0 = estimate (30 Lanczos steps)
  double norm_b = 0.0;  ///< ||B||_2; 0 = estimate (1 when B == nullptr)
};

/// LOBPCG (Knyazev 2001) with soft locking and a robust basis (Hetmaniuk & Lehoucq 2006; Duersch,
/// Shao, Yang & Gu 2018): every iteration B-orthonormalizes the preconditioned residuals W of the
/// active (unconverged) columns against [Y X P] and among themselves (block CGS2 + SVQB, dropping
/// numerically dependent directions), solves the Rayleigh-Ritz problem on [X P W] with the
/// explicitly computed Gram matrix, and builds the next P as the part of the new X outside the old
/// one, orthonormalized inside the small space. When the Gram matrix of [X P W] is ill conditioned
/// P is dropped for that iteration (counted in `restarts`).
/// which: SmallestAlgebraic or LargestAlgebraic (LargestMagnitude throws InvalidInput).
/// precond: nullptr (plain block steepest descent/CG directions R) or T ~ (A - s B)^{-1} with s
/// beyond the wanted end of the spectrum (Hermitian definite of either sign; the sign does not
/// matter, only the spanned directions).
/// Returns m == nev pairs (the nev best ranked of the block).
/// Cost per iteration: block applications of A, B and T to the active columns (B: one more block
/// per extra orthonormalization round), O(n m^2) dense work for the Gram matrices and updates.
/// Memory: n * m * (3 blocks [X P W] for each of S, A S and B S (B S only with B) + 1 residual
/// block) + n d (B Y) + O(m^2) small matrices; the result adds n * nev.
template <class T>
IterativeEigResult<T> lobpcg(LinearOperator<T>& a, LinearOperator<T>* b, LinearOperator<T>* precond,
                             Which which, const LobpcgOptions<T>& options);

// ---------------------------------------------------------------------------- Jacobi-Davidson

/// How Ritz approximations are extracted from the search space V (B-orthonormal).
enum class JdExtraction {
  Auto,      ///< Ritz (measured: fewer solves and no failures where Harmonic failed, see
             ///< jacobi_davidson)
  Harmonic,  ///< harmonic Ritz vectors w.r.t. tau (Petrov space (A - tau I) V), with Ritz
             ///< polishing and a certified Ritz override for eigenvalues at tau; B == nullptr only
  Ritz,      ///< standard Rayleigh-Ritz, Ritz values nearest tau
};

/// Solver for the correction equation (I - B Q~ Q~^H)(A - theta B)(I - Q~ Q~^H B) t = -r,
/// Q~ = [locked vectors, current approximation u].
enum class JdInnerSolver {
  Auto,    ///< Gmres with a preconditioner, Minres without
  Gmres,   ///< right-preconditioned flexible GMRES; any preconditioner (oblique projection with
           ///< K^{-1} B Q~, Sleijpen & van der Vorst / Fokkema et al.)
  Minres,  ///< preconditioned MINRES; the preconditioner MUST be Hermitian positive definite
};

/// Options of jacobi_davidson. Pointer members are call-scoped.
template <class T>
struct JdOptions {
  index_t nev = 6;               ///< wanted pairs nearest tau, 1 <= nev <= n
  double tol = 1e-10;            ///< backward-error tolerance (see the convergence test)
  index_t max_iterations = 1000; ///< outer iterations (search-space expansions)
  /// Search-space size kept after a thick restart; 0 = auto: min(n, max(nev, 10)).
  index_t min_basis = 0;
  /// Search-space size that triggers a thick restart; 0 = auto: min(n, max(2 min_basis, min_basis
  /// + 10)). Must exceed min_basis (unless it reaches the available dimension).
  index_t max_basis = 0;
  /// Maximum inner iterations per correction equation (Krylov steps). 0: no inner solve,
  /// t = -M~ r (preconditioned Davidson / Olsen step). Inner stopping (see jacobi_davidson):
  /// tau-shifted equations at relative residual 0.25 and, with a preconditioner, at most 5 steps;
  /// theta-shifted ones at 2^-(j+1), j = theta-shifted outer iterations on the current pair.
  index_t inner_iterations = 20;
  std::uint64_t seed = 2026;  ///< deterministic random start block
  /// Subspace-iteration steps S <- orth(M B S) applied to the random start block before the first
  /// extraction (only with a preconditioner M and without `initial`; costs start_iterations * p
  /// preconditioner applications, p = start block size); 0 = none. See jacobi_davidson.
  index_t start_iterations = 2;
  /// Optional start block (n x p, p >= 1), B-orthonormalized and used as the initial search
  /// space instead of the random start block.
  const DenseMatrix<T>* initial = nullptr;
  /// Optional block Y (n x d) of known eigenvectors, B-orthonormal (|Y^H B Y - I| <= 1e-8): they
  /// are treated as already locked (the search space and every correction stay B-orthogonal to
  /// them) and are not returned, so the call finds nev further pairs (deflated rerun after an
  /// inertia certificate found missing eigenvalues). nev + d <= n.
  const DenseMatrix<T>* deflation = nullptr;
  JdExtraction extraction = JdExtraction::Auto;
  JdInnerSolver inner_solver = JdInnerSolver::Auto;
  double norm_a = 0.0;  ///< ||A||_2 for the backward error; 0 = estimate (30 Lanczos steps)
  double norm_b = 0.0;  ///< ||B||_2; 0 = estimate (1 when B == nullptr)
};

/// Hermitian Jacobi-Davidson (Sleijpen & van der Vorst 1996; JDQR-style locking and thick restart,
/// Fokkema, Sleijpen & van der Vorst 1998) for the nev eigenpairs of A x = lambda B x nearest
/// `target`.
/// Each outer iteration extracts the approximation u nearest tau from V (standard Ritz by default,
/// or harmonic, see JdExtraction), takes theta = u^H A u (u^H B u = 1) and r = A u - theta B u (from the
/// stored products, refreshed at locking), locks u when its backward error is <= tol (it leaves V
/// and joins the deflation block Q, which keeps every later correction B-orthogonal to it), and
/// otherwise expands V by the approximate solution t of the deflated correction equation.
/// Target tracking: the correction equation is shifted by tau while u is rough (with M ~
/// (A - tau B)^{-1} the expansion then acts like shift-and-invert at tau, so V collects the
/// eigenvectors nearest tau in order) and by theta once ||r|| / ||B u|| <= 0.1 * (distance to the
/// nearest Ritz value of another eigenvalue) (fast local convergence, like inexact Rayleigh
/// quotient iteration).
/// Start space: p = min(nev, max_basis - 1, n - d) random vectors. With a preconditioner they first
/// get start_iterations (default 2) steps of subspace iteration S <- orth(M B S): with M ~
/// (A - tau B)^{-1} this brings the eigenvectors nearest tau - in particular several copies of a
/// multiple eigenvalue - into V together. Every later expansion amplifies only the copy inside the
/// current approximation u, so from a plain random block the other copies stay diluted and a
/// farther eigenvalue is locked instead (measured, bench/lobpcg_jd_fix_synthetic.cpp: with a
/// 6-, 8- or 10-fold eigenvalue and nev = 6, 24 of 24 preconditioned runs returned 4-5 copies with
/// converged = true without the block iteration, 0 of 24 with it, at 36% fewer solves in total).
/// M only affects the speed, never the accuracy (r is exact), so a single-precision factor of
/// A - tau B yields double-precision eigenpairs. Measured (bench/lobpcg_jd_memory.cpp, 2 threads,
/// nev = 12, tol = 1e-10, 5 runs each with a fresh, non-deterministic MUMPS factor; min / median /
/// max of preconditioner solves and of the total time incl. analysis and factorization):
///   syn-mem27-46 (tau = 0 lies 3e-6 and 5e-6 from two 6-fold eigenvalues, ||A|| = 3.6):
///     JD + single factor  123 / 133 / 141 solves,  7.5 / 8.0 / 9.6 s, factor 337 MB, peak 0.55 GiB
///     shift-invert Lanczos, double factor  36 solves,  5.2 / 5.4 / 6.5 s,  651 MB, 0.70 GiB
///     single factor + refinement (the driver's fallback)  303 / 396 / 396 solves,
///       13.7 / 19.0 / 25.0 s, 337 MB, 0.46 GiB; 2 of 5 runs not accepted (eta 2e-8, 2e-5)
///     former JD defaults (harmonic extraction, no block iteration)  312 / 665 / 2170 solves,
///       18 / 31 / 117 s
///   yee-pec-cube-32 (tau = 53.16 between two 6-fold eigenvalues):
///     JD + single factor  196 / 200 / 206 solves,  8.8 / 9.0 / 11.1 s, 282 MB, 0.45 GiB
///     shift-invert Lanczos, double factor  68 solves,  5.8 / 6.1 / 7.7 s,  548 MB, 0.57 GiB
///     single factor + refinement  300 / 381 / 613 solves,  12.1 / 15.1 / 24.6 s, 0.36 GiB;
///       1 of 5 runs missed a copy of a 6-fold eigenvalue (bare Lanczos, no certificate)
///     former JD defaults  207 / 214 / 230 solves,  9.9 / 10.2 / 12.2 s
///   yee-pec-cube-32 with tau 9.5e-9 and 2.5e-11 from the 6-fold eigenvalue 49.21342550952...
///   (eps_s cond(A - tau B) > 1; at 2.5e-11 the double factor is singular, at 9.5e-9 double
///   shift-invert Lanczos ends at eta 2e-8 with a wrong set; the single factor was reported
///   singular in 2 of 12 runs): JD 219 - 259 and 231 - 333 solves (5 runs each), 10 - 19 s, all
///   correct; single factor + refinement fails (eta 7e-2); former JD defaults at 9.5e-9: 608
///   solves with a wrong set (a copy missing, converged = true) or 6339 solves / 241 s; at
///   2.5e-11: no convergence in 1000 iterations (7230 solves, 323 s, eta 0.2).
/// So JD with a single factor costs about 1.5x the time of double shift-invert Lanczos at half the
/// factor memory, and is faster and more reliable than single factor + refinement.
/// The pairs found are the nearest ones in practice, NOT provably: in particular without a
/// preconditioner (or with start_iterations = 0) copies of an eigenvalue of multiplicity >= about
/// nev - 1 can be lost to farther eigenvalues while converged = true (measured: 28 of 32
/// unpreconditioned runs of the synthetic study). A missing eigenvalue is not detected here; the
/// driver's inertia certificate detects it, and a rerun with the copies found as `deflation`
/// recovers the missing ones.
/// Returns m == nev pairs when converged; otherwise the locked pairs plus the best current
/// approximations (m = nconv + min(nev - nconv, current basis size)).
/// Cost per outer iteration: 1 + (inner iterations) applications each of A and B (with B), as many
/// preconditioner applications (+1 for M B u with GMRES, +1 per locked pair once),
/// O(n (nconv + basis size)) for the B-orthogonalization and projections.
/// Memory (scalars): n * (d + nev + max_basis) for [Y Q V], the same again for B [Y Q V] (B only)
/// and for (A - tau B) V, n * (d + nev + 1) for M B [Y Q u] (GMRES with a
/// preconditioner), inner workspace n * (2 inner_iterations + 1) (GMRES) or 7 n (MINRES), plus 7 n
/// work vectors, plus 2 n p (3 n p with B) during the start block iteration; the result adds
/// n * nev. The start block iteration costs start_iterations * p preconditioner applications.
template <class T>
IterativeEigResult<T> jacobi_davidson(LinearOperator<T>& a, LinearOperator<T>* b,
                                      LinearOperator<T>* precond, double target,
                                      const JdOptions<T>& options);

}  // namespace hs
