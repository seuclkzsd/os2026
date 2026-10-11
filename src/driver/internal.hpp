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
/// time), so the report's time_setup_s + time_factor_s + time_iterate_s + time_verify_s +
/// time_certificate_s equals time_total_s:
///   Setup       input checks, norm estimates, problem-type decision, planning (the default);
///   Factor      symbolic analyses and numeric factorizations of shifts, of B (definiteness
///               test) and of A (GHIEP zero cluster);
///   Iterate     the Krylov solvers (operator applications = solves, orthogonalization, restarts);
///   Verify      Rayleigh-Ritz refinement, residuals, rescue, selection, certificate bookkeeping;
///   Certificate Sylvester inertia counts of the HEP/GHEP path (count_below: certificate
///               endpoints, window ends, slice cuts, cluster checks), including the
///               InertiaCounter's analysis.
enum class Phase { Setup = 0, Factor = 1, Iterate = 2, Verify = 3, Certificate = 4 };
constexpr int kPhases = 5;

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
  double acc_[kPhases] = {0.0, 0.0, 0.0, 0.0, 0.0};
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

// ------------------------------------------------------------------------------ run-time models

/// Run-time models of the MUMPS building blocks on this machine (DESIGN §3.6-3b). Deterministic
/// functions of the input and the analysis estimate, so that every decision built on them
/// (ordering, InertiaCounter, slice size) is the same in every run of the same problem. `order`
/// and `entries` describe the matrix MUMPS factorizes (mumps_input_size).
double model_factor_s(const FactorEstimate& e, double order, int threads);  ///< one factorization
double model_solve_s(const FactorEstimate& e);                               ///< one 1-RHS solve
/// One MUMPS analysis with AMD ordering (input conversion, ordering, symbolic factorization) of an
/// input with `entries` upper-triangle entries whose factors hold `factor_entries` (the analysis
/// estimate), fitted on the data cases (steps.cpp). MUMPS' analysis is sequential: no threads.
double model_analysis_s(double order, double entries, double factor_entries);
/// An InertiaCounter count (inertia-only factorization) saves this fraction of a factorization.
constexpr double kCounterSaving = 0.13;
/// A SCOTCH analysis costs this many AMD analyses of the same input (adaptive ordering; measured
/// 2.2..17, median ~9 on the 1e5 data cases, bench/drvint_profile --exp analysis; 7 keeps the
/// P3 calibration, which picks the same ordering as before on every data case).
constexpr double kNdAnalysisOverAmd = 7.0;

struct MumpsInputSize {
  double order = 0.0;    ///< n, or 2n for complex T (real symmetric embedding)
  double entries = 0.0;  ///< upper-triangle entries passed to MUMPS
};
template <class T>
MumpsInputSize mumps_input_size(const ShiftedPencil<T>& pencil);

// ------------------------------------------------------------------------------ factorization

FactorOptions factor_options_from(const SolverConfig& cfg, double cap_mb);

/// Factorize K(sigma0) (DESIGN §3.6-4): perturb the shift on a null pivot, and factorize at sigma0
/// without null-pivot detection when every nearby shift looks singular (dense cluster). Returns
/// the shift used; detection_off tells later refactorizations at that shift to do the same.
/// on_eigenvalue: sigma0 was placed on an eigenvalue on purpose (a multiple eigenvalue's own
/// shift): the perturbation is expected and not reported as a warning.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off,
                      bool on_eigenvalue = false);

/// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
/// No-op when the object already holds exactly that factorization (Ctx::held).
template <class T>
void refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);
/// The same, returning false instead of throwing when K(sigma) is now flagged singular.
template <class T>
bool try_refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);

/// Where the Sylvester counts of a solve path come from (DESIGN §3.6-8):
///   * the main factorization object, refactorized at each count point: it then holds K(s)
///     (Ctx::held), which a Krylov run at s reuses (one-sided slice anchors);
///   * an InertiaCounter, built at the first count from the main analysis' pivot order
///     (factor.hpp): inertia-only factorizations (measured 10-23% cheaper) that leave the main
///     factors intact, so reruns and rescue at the working shift need no refactorization. When
///     the main factors' working memory (INFOG(21)) plus the counter's estimate exceed the
///     factorization cap, the main factors are released first (then they are refactorized when
///     needed again, as with main-object counts).
/// Inexact main factors (single precision, BLR) always count on the counter: their own inertia is
/// that of the rounded / compressed matrix. Not thread-safe; the pencil and the main object must
/// outlive it.
template <class T>
class CountSource {
 public:
  /// prefer_counter: count on the counter even with an exact main factor (see counter_pays()).
  CountSource(SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, bool prefer_counter);
  ~CountSource();
  CountSource(const CountSource&) = delete;
  CountSource& operator=(const CountSource&) = delete;

  SparseFactorization<T>& fac() { return fac_; }
  const ShiftedPencil<T>& pencil() const { return pencil_; }
  bool uses_counter() const { return use_counter_; }
  /// Inertia of K(s) = A - s B (valid == false: K(s) numerically singular). Main object: Ctx::held
  /// describes the factorization it now holds. Charged to Phase::Certificate.
  Inertia count(Ctx& c, double s, bool detection);

 private:
  SparseFactorization<T>& fac_;
  const ShiftedPencil<T>& pencil_;
  bool use_counter_ = false;
  std::unique_ptr<InertiaCounter<T>> counter_;
};

/// Rule for the nearest target's certificate (two counts, exact main factor): an InertiaCounter
/// pays when the two counts' saving, kCounterSaving of a factorization each (model of DESIGN
/// §3.6-3b), exceeds its analysis (model_analysis_s: about what an AMD analysis of the input
/// costs). Reruns after a failed certificate save a refactorization on top (not counted).
/// cfg.inertia_counter On / Off override it.
template <class T>
bool counter_pays(const Ctx& c, const SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, index_t counts);

/// nu_-(A - s B): number of eigenvalues below s (DESIGN §4.2), moving s outward (direction
/// `outward`, steps step * 4^i) while K(s) is singular. Returns the final s via `s`. Counting on
/// the main object, that object then holds K(s) (Ctx::held), counted with null-pivot detection
/// unless *detection_off (optional) reports otherwise. s_is_gap: s lies mid-way between
/// computed eigenvalues (count there without detection if flagged); otherwise a flagged s is first
/// moved by step / 10 (it may sit on an eigenvalue) - see steps.cpp. *near_eigenvalue (optional)
/// is set when the requested s was flagged singular: an eigenvalue (or a cluster) lies within the
/// null-pivot threshold of the returned s, which then is no place for a shift.
template <class T>
index_t count_below(Ctx& c, CountSource<T>& src, double& s, double outward, double step,
                    bool* detection_off = nullptr, bool s_is_gap = false, bool* near_eigenvalue = nullptr);

/// Symbolic analysis at sigma0 and the memory-budget decision (DESIGN §3.6-3).
template <class T>
void analyze_within_budget(Ctx& c, SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, double sigma0,
                           bool already_analyzed = false);

/// The adaptive ordering's decision from the AMD probe's estimate (deterministic, see steps.cpp).
struct OrderingDecision {
  double fill_ratio = 0.0;      ///< AMD factor entries / nnz(K)
  double saving_s = 0.0;        ///< predicted saving of nested dissection on the run
  double amd_analysis_s = 0.0;  ///< model_analysis_s of the input and the AMD estimate
  double nd_cost_s = 0.0;       ///< kNdAnalysisOverAmd * amd_analysis_s
  bool use_nd = false;          ///< saving_s > nd_cost_s
};
OrderingDecision decide_ordering(const FactorEstimate& amd, const MumpsInputSize& in, double pencil_nnz,
                                 int threads, index_t expected_factorizations, index_t expected_solves);

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

// ------------------------------------------------------------------------------ block Krylov

/// Columns of n-vectors a shift-invert lanczos_krylov_schur run holds (driver use: no store_bv):
/// p = 1: the basis [D V v] plus the M*w workspace, d + ncv + 2; p > 1: [D V V_m] plus the n x p
/// M*W workspace, d + ncv_b + 2p with ncv_b from eigen_detail::block_dimensions (the larger
/// block basis), or d + n_free + 2 when the block layout does not fit the free space (the solver
/// then runs one single-vector cycle over it). ncv: SolverConfig::ncv (0 = auto).
index_t lanczos_basis_columns(index_t n, index_t d, index_t nev, index_t ncv, index_t p);

/// Widest block the auto rule of block_size_for() chooses.
constexpr index_t kMaxAutoBlockSize = 8;

/// Block width of the definite path's first shift-invert Lanczos run for nev wanted pairs:
/// SolverConfig::block_size if > 0, else the auto rule (see steps.cpp).
template <class T>
index_t block_size_for(const Ctx& c, const SparseFactorization<T>& fac, index_t n, index_t nev);

/// p, halved while the basis of a run of width p (lanczos_basis_columns, plus `extra_columns`
/// n-vectors the caller keeps, e.g. the candidate block X, AX, BX) would not fit the memory left
/// in the budget (warning when reduced).
template <class T>
index_t fit_block_size(Ctx& c, index_t n, index_t nev, index_t d, index_t extra_columns, index_t p);

/// Block width of a deflated rerun that misses `missing` pairs after a run with width p:
/// max(p, missing rounded up to even), so that a random start block has components in every
/// missing copy of a multiple eigenvalue and one run finds them all (eigen.hpp, block mode).
index_t rerun_block_size(index_t p, index_t missing);

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
