// Driver-internal building blocks shared by the solve paths (src/driver/*.cpp only).
// Scalar set of the templates: T in {double, cplx}, instantiated in the .cpp that defines them.
#pragma once

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "driver/norms.hpp"
#include "hsolve/dense.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"
#include "hsolve/util.hpp"

namespace hs::driver {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ------------------------------------------------------------------------------ time accounting

/// Phases of a solve. Every instant of hs::solve is charged to exactly one phase (exclusive
/// time), so the report's time_setup_s + time_factor_s + time_iterate_s + time_verify_s equals
/// time_total_s:
///   Setup   input checks, norm estimates, problem-type decision, planning (the default);
///   Factor  symbolic analyses and numeric factorizations (shifts, inertia counts, B, A);
///   Iterate the Krylov solvers (operator applications = solves, orthogonalization, restarts);
///   Verify  Rayleigh-Ritz refinement, residuals, rescue, selection, certificate bookkeeping.
enum class Phase { Setup = 0, Factor = 1, Iterate = 2, Verify = 3 };

class PhaseClock {
 public:
  /// Makes p the running phase; the time since the last switch goes to the previous phase,
  /// which is returned.
  Phase switch_to(Phase p);
  /// Accumulated seconds of phase p, including the running interval if p is running.
  double seconds(Phase p) const;
  double total() const;

 private:
  Timer timer_;
  double last_ = 0.0;
  Phase current_ = Phase::Setup;
  double acc_[4] = {0.0, 0.0, 0.0, 0.0};
};

/// Charges its lifetime to phase p (nested scopes: the innermost wins; the enclosing phase
/// resumes on destruction).
class PhaseScope {
 public:
  PhaseScope(PhaseClock& clock, Phase p) : clock_(clock), prev_(clock.switch_to(p)) {}
  ~PhaseScope() { clock_.switch_to(prev_); }
  PhaseScope(const PhaseScope&) = delete;
  PhaseScope& operator=(const PhaseScope&) = delete;

 private:
  PhaseClock& clock_;
  Phase prev_;
};

// ------------------------------------------------------------------------------ context

/// What the main factorization object currently holds, so that a factorization done for an
/// inertia count can serve as the shift of the next Krylov run without refactorizing.
struct FactorHeld {
  double shift = std::numeric_limits<double>::quiet_NaN();  ///< K(shift) factorized; NaN: other
  bool detection_off = false;  ///< factorized without null-pivot detection (dense cluster)
};

struct Ctx {
  const SolverConfig& cfg;
  Logger* log;
  SolveReport& rep;
  PhaseClock& clock;
  FactorHeld held{};
};

void warn(Ctx& c, const std::string& msg);
void event(Ctx& c, const std::string& name, std::initializer_list<LogField> fields);
std::string fmt(double x);

// ------------------------------------------------------------------------------ pair blocks

/// Pairs being refined: eigenvalues with their vectors and A*X, B*X kept in sync.
template <class T>
struct Block {
  std::vector<double> lambda;
  DenseMatrix<T> x, ax, bx;

  index_t size() const { return static_cast<index_t>(lambda.size()); }
};

template <class T>
void recompute_products(const CsrMatrix<T>& a, const CsrMatrix<T>* b, Block<T>& blk);
template <class T>
Block<T> take_columns(const Block<T>& src, const std::vector<index_t>& cols);
/// dst <- [dst src] (dst may be empty).
template <class T>
void append_columns(DenseMatrix<T>& dst, const DenseMatrix<T>& src);

// ------------------------------------------------------------------------------ factorization

FactorOptions factor_options_from(const SolverConfig& cfg, double cap_mb);

/// Factorize K(sigma0) (DESIGN §3.6-4): perturb the shift on a null pivot, and factorize at sigma0
/// without null-pivot detection when every nearby shift looks singular (dense cluster). Returns
/// the shift used; detection_off tells later refactorizations at that shift to do the same.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off);

/// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
/// No-op when the object already holds exactly that factorization (Ctx::held).
template <class T>
void refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);
/// The same, returning false instead of throwing when K(sigma) is now flagged singular.
template <class T>
bool try_refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);

/// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2), moving s outward (direction
/// `outward`, steps step * 4^i) while K(s) is singular. Returns the final s via `s`. With an
/// exact (double, non-BLR) factor the object then holds K(s) (Ctx::held), counted with null-pivot
/// detection unless *detection_off (optional) reports otherwise. s_is_gap: s lies mid-way between
/// computed eigenvalues (count there without detection if flagged); otherwise a flagged s is first
/// moved by step / 10 (it may sit on an eigenvalue) - see steps.cpp.
template <class T>
index_t count_below(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double& s, double outward,
                    double step, bool* detection_off = nullptr, bool s_is_gap = false);

/// Symbolic analysis at sigma0 and the memory-budget decision (DESIGN §3.6-3).
template <class T>
void analyze_within_budget(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma0,
                           bool already_analyzed = false);

/// Adaptive fill-reducing ordering (DESIGN §3.6-3b): AMD probe + cost model, SCOTCH if it pays.
template <class T>
std::unique_ptr<SparseFactorization<T>> adaptive_analysis(Ctx& c, const FactorOptions& fopts,
                                                          const ShiftedPencil<T>& pencil, double sigma0,
                                                          index_t expected_factorizations, index_t expected_solves);

// ------------------------------------------------------------------------------ cost model

/// Run-time model of the shift-invert building blocks on this machine, from the analysis
/// estimate (the adaptive-ordering model of DESIGN §3.6-3b plus the Krylov basis work). Only
/// ratios matter: it decides how many eigenvalues one shift should serve (slice capacity) and
/// when another inertia count is cheaper than computing extra eigenpairs.
struct CostModel {
  double t_factor = 0.0;    ///< seconds per numeric factorization
  double t_solve = 0.0;     ///< seconds per single right-hand-side solve
  double vec_bytes = 0.0;   ///< bytes of one n-vector
  double stream_bw = 1.0;   ///< bytes/s of the BLAS2 passes over the Krylov basis
  /// Seconds per converged pair in a Lanczos run that converges c pairs (solves + CGS2 passes over
  /// a basis of ~2c vectors).
  double per_pair(index_t c) const;
  /// Extra seconds for one more pair in a run that converges c pairs.
  double marginal_pair(index_t c) const;
  /// Pairs per shift minimizing (factorizations + solves + basis work) per pair.
  index_t best_slice() const;
};

CostModel make_cost_model(const FactorEstimate& est, index_t n, int threads, bool complex_scalars);

/// Ritz estimates of the ends of the spectrum of the pencil (from inside): lowest two and
/// highest two Ritz values of a short Lanczos run (A alone for B = I, B^{-1}A otherwise).
struct SpectrumEstimate {
  double lo1 = 0.0, lo2 = 0.0, hi1 = 0.0, hi2 = 0.0;
  bool valid = false;
};

// ------------------------------------------------------------------------------ solve paths

/// HEP / GHEP (B HPD): shift-invert Lanczos, Rayleigh-Ritz, rescue, inertia certificate;
/// nearest targets with one shift, interval / smallest / largest with spectrum slicing.
template <class T>
Solution<T> solve_definite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                           double n2b, double scale, double sigma0, const SpectrumEstimate& spectrum);

/// GHIEP (B indefinite): shift-invert general Krylov-Schur, projected refinement, real/complex
/// classification.
template <class T>
Solution<T> solve_indefinite(Ctx& c, const ShiftedPencil<T>& pencil, SparseFactorization<T>& fac, double n2a,
                             double n2b, double scale);

}  // namespace hs::driver
