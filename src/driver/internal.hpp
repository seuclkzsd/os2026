// Driver-internal building blocks shared by the solve paths (src/driver/*.cpp only).
// Scalar set of the templates: T in {double, cplx}, instantiated in the .cpp that defines them.
#pragma once

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
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
/// A nested-dissection analysis (SCOTCH, or METIS for deterministic runs: nd_ordering) costs this
/// many AMD analyses of the same input (adaptive ordering; measured 2.2..17, median ~9 on the 1e5
/// data cases, bench/drvint_profile --exp analysis; METIS median 9.8 vs SCOTCH 9.3 on the cases
/// that pick it, bench/drvint_fix_ordering; 7 keeps the P3 calibration, which picks the same
/// ordering as before on every data case).
constexpr double kNdAnalysisOverAmd = 7.0;

struct MumpsInputSize {
  double order = 0.0;    ///< n, or 2n for complex T (real symmetric embedding)
  double entries = 0.0;  ///< upper-triangle entries passed to MUMPS
};
template <class T>
MumpsInputSize mumps_input_size(const ShiftedPencil<T>& pencil);

// ------------------------------------------------------------------------------ factorization

FactorOptions factor_options_from(const SolverConfig& cfg, double cap_mb);

/// Factorize K(sigma0) (DESIGN §3.6-4): perturb the shift by 1e-8 * max(scale, |sigma0|) on null
/// pivots of one eigenvalue (isolated eigenvalue at sigma0), and factorize at sigma0 without
/// null-pivot detection inside a dense cluster - several eigenvalues' worth of null pivots
/// (near_null_eigenvalues >= 2), or the perturbed shift singular as well. Returns the shift used;
/// detection_off tells later refactorizations at that shift to do the same. on_eigenvalue: sigma0
/// was placed on an eigenvalue on purpose (a multiple eigenvalue's own shift): the perturbation is
/// expected (whatever the null pivots), and not reported as a warning.
template <class T>
double factorize_near(Ctx& c, SparseFactorization<T>& fac, double sigma0, double scale, bool& detection_off,
                      bool on_eigenvalue = false);
/// Eigenvalues the null pivots of a factorization stand for: a complex T factorizes the real
/// embedding, in which every eigenvalue appears twice.
index_t near_null_eigenvalues(index_t null_pivots, bool complex_scalars);

/// Refactorize K(sigma) the way factorize_near() did (needed after certificate factorizations).
/// No-op when the object already holds exactly that factorization (Ctx::held).
template <class T>
void refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);
/// The same, returning false instead of throwing when K(sigma) is now flagged singular.
template <class T>
bool try_refactorize_at(Ctx& c, SparseFactorization<T>& fac, double sigma, bool detection_off);

/// What a CountSource counting on its InertiaCounter does with the main factors, decided anew
/// before every count (the main object may hold factors again since the previous count: a rerun,
/// a rescue or the next slice refactorized it).
enum class MainFactors {
  /// Keep them, unless their working memory (INFOG(21)) plus the counter's estimate exceed the
  /// factorization cap (FactorOptions::max_memory_mb, DESIGN §3.6-3): then release them. For
  /// paths that run Krylov solves at a shift between counts (windows: slice shifts, rescue).
  KeepWithinCap,
  /// Always release them (MUMPS JOB=-4) before a count: for counts after which the factors are
  /// needed only on a failed certificate (nearest target: one refactorization for the rerun, as
  /// with main-object counts). The counts then never add the counter's working memory to the main
  /// factors'. Nearest, six 3-D 1e5 cases, 4 threads, medians of 6 interleaved runs: peak RSS
  /// 0.48..0.85 GiB, 21..26% below keeping them resident and 2..7% below main-object counts, at
  /// the certificate time of the resident variant (0.05..0.14 s below main-object counts).
  Release,
};

/// Where the Sylvester counts of a solve path come from (DESIGN §3.6-8):
///   * the main factorization object, refactorized at each count point: it then holds K(s)
///     (Ctx::held), which a Krylov run at s reuses (one-sided slice anchors);
///   * an InertiaCounter, built at the first count from the main analysis' pivot order
///     (factor.hpp): inertia-only factorizations (measured 10-23% cheaper) that do not touch the
///     main factors; MainFactors says whether those are released before a count (then they are
///     refactorized when needed again, as with main-object counts).
/// Inexact main factors (single precision, BLR) always count on the counter: their own inertia is
/// that of the rounded / compressed matrix. Not thread-safe; the pencil and the main object must
/// outlive it.
template <class T>
class CountSource {
 public:
  /// prefer_counter: count on the counter even with an exact main factor (see counter_pays()).
  /// main: what counter counts do with the main factors (no effect on main-object counts).
  CountSource(SparseFactorization<T>& fac, const ShiftedPencil<T>& pencil, bool prefer_counter,
              MainFactors main = MainFactors::KeepWithinCap);
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
  MainFactors main_ = MainFactors::KeepWithinCap;
  std::unique_ptr<InertiaCounter<T>> counter_;
};

/// Rule for the nearest target's certificate (`counts` counts, exact main factor): an
/// InertiaCounter pays when the counts' saving, kCounterSaving of a factorization each (model of
/// DESIGN §3.6-3b), exceeds its analysis (model_analysis_s: about what an AMD analysis of the
/// input costs). e: the main analysis' estimate, in: mumps_input_size of the pencil, threads: of
/// the factorizations. cfg.inertia_counter On / Off override it. Deterministic.
bool counter_pays(const SolverConfig& cfg, const FactorEstimate& e, const MumpsInputSize& in, int threads,
                  index_t counts);

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

/// The P4 rule of the adaptive ordering (AMD probe; one MUMPS nested dissection, nd_ordering, if
/// the P3 model's saving exceeds kNdAnalysisOverAmd AMD analyses). Replaced in the driver by
/// choose_ordering / adaptive_analysis below; kept as the reference the P5 choice is compared
/// against (bench/anaint_decide.cpp, bench/drvint_*). Deterministic.
struct OrderingDecision {
  double fill_ratio = 0.0;      ///< AMD factor entries / nnz(K)
  double saving_s = 0.0;        ///< predicted saving of nested dissection on the run
  double amd_analysis_s = 0.0;  ///< model_analysis_s of the input and the AMD estimate
  double nd_cost_s = 0.0;       ///< kNdAnalysisOverAmd * amd_analysis_s
  bool use_nd = false;          ///< saving_s > nd_cost_s
};
OrderingDecision decide_ordering(const FactorEstimate& amd, const MumpsInputSize& in, double pencil_nnz,
                                 int threads, index_t expected_factorizations, index_t expected_solves);

/// MUMPS ICNTL(7) of the P4 rule's nested dissection: SCOTCH (3), or METIS (5, fixed seed) when
/// cfg.deterministic asks for bitwise-reproducible runs - SCOTCH's orderings vary from run to run
/// (measurements in steps.cpp).
int nd_ordering(const SolverConfig& cfg);

// ------------------------------------------------------------------------------ ordering choice

/// What a solve plans to do with the factors of its one analysis, from the configuration and the
/// problem type known before the analysis (ProblemType::Auto: B's inertia is still to be counted,
/// one factorization). Deterministic (no timing); calibrated on the data cases' end-to-end runs
/// (plan_work in steps.cpp). The adaptive ordering prices every candidate on it.
struct PlannedWork {
  double factorizations = 1.0;  ///< numeric factorizations of the main object (shifts, B, A's null space)
  double counts = 0.0;          ///< certificate inertia counts (an InertiaCounter's or the main object's)
  double solve_calls = 0.0;     ///< multi-RHS solve calls of the Krylov runs
  index_t block = 1;            ///< right-hand sides per call (block Krylov width)
};
/// scale: ||A||_2 / ||B||_2 (GHIEP: a shift in the zero band means one-sided runs).
PlannedWork plan_work(const SolverConfig& cfg, ProblemType type, index_t n, double scale);

/// One candidate fill-reducing ordering of the adaptive analysis, priced on the planned work.
struct OrderingCandidate {
  int ordering = 0;         ///< FactorOptions::ordering (0 AMD, kOrderingNestedDissection, 5, 4, 3)
  bool analysed = false;    ///< flops / entries are its own analysis estimate (else predicted)
  double analysis_s = 0.0;  ///< predicted analysis time still to pay (0 once analysed)
  double flops = 0.0;       ///< factorization flops (RINFOG(1))
  double entries = 0.0;     ///< factor entries (INFOG(20))
  double numeric_s = 0.0;   ///< predicted time of the planned factorizations, counts and solves
  double incore_mb = 0.0;   ///< in-core MB of the factors (estimate; predicted for the others)
  /// What the choice compares: the analysed candidates' numeric_s; for the others their analysis
  /// plus numeric_s less the expected gain of knowing their actual estimate (choose_ordering).
  /// Factors beyond the in-core cap score kOverCapScore-scaled by their size (steps.cpp).
  double score_s = 0.0;
};

/// Orderings the adaptive analysis chooses from, in order of preference on near-equal scores: AMD,
/// SCOTCH, hsolve's nested dissection (multithreaded), METIS, PORD (steps.cpp: why this order).
std::vector<int> candidate_orderings(const SolverConfig& cfg);
/// FactorOptions::ordering of MUMPS' SCOTCH (ICNTL(7) = 3).
constexpr int kOrderingScotch = 3;
/// Whether adaptive_analysis takes `ordering` on its prediction: once it is the best-scoring
/// candidate it is analysed and used without comparing its estimate (SCOTCH: its orderings differ
/// between runs, so a comparison could choose differently in identical runs).
bool taken_on_prediction(int ordering);

/// Sizes of the graphs the orderings work on: MUMPS orders its input (mumps_input_size), hsolve's
/// nested dissection the same graph - or, for a complex pencil whose couplings are mostly complex,
/// the n-vertex graph of K (compute_ordering's Embedding::Auto rule), with pencil.nnz() entries.
struct OrderingInput {
  MumpsInputSize mumps;
  MumpsInputSize nd;
};
template <class T>
OrderingInput ordering_input(const ShiftedPencil<T>& pencil);

/// The ordering cost model (steps.cpp: fit and measurements). Predicted wall time of one analysis
/// with `ordering` (MUMPS orderings are serial, hsolve's nested dissection runs on `threads`) of an
/// input of sizes `in` whose factors hold `factor_entries`.
double ordering_analysis_s(int ordering, const OrderingInput& in, double factor_entries, int threads);
/// Predicted seconds of one numeric factorization (the first on its analysis) of an input of sizes
/// `in` whose factors take `flops` and hold `entries`, on `threads` (the fit of steps.cpp).
double factorization_s(const MumpsInputSize& in, double flops, double entries, int threads);
/// Predicted time of the planned numeric work with factors of `flops` and `entries` on `threads`:
/// factorizations, certificate counts (on an InertiaCounter when counter_pays, which then adds the
/// counter's analysis) and the solve calls of width w.block. amd_fill > 0: the factors are AMD's,
/// with this fill ratio (entries / nnz of the pencil) - at low fill its numeric phases parallelize
/// worse (steps.cpp); 0 for a nested dissection.
double planned_numeric_s(const SolverConfig& cfg, const PlannedWork& w, const MumpsInputSize& in, double flops,
                         double entries, int threads, double amd_fill = 0.0);
/// Predicted (flops, entries) of `ordering` on an input whose AMD analysis estimated `amd`
/// (pencil_nnz: upper entries of the pencil), scaled from the actual estimate `anchor` of the
/// analysed ordering `anchor_ordering` (AMD itself, or a nested dissection: the relative
/// prediction between two nested dissections is much tighter than from AMD).
std::pair<double, double> predicted_shape(int ordering, const FactorEstimate& amd, double pencil_nnz,
                                          int anchor_ordering, const FactorEstimate& anchor);

/// The choice step of adaptive_analysis: given the candidates (at least one analysed), the index
/// of the one to take if it is analysed, or to analyse next. Scores ties go to the earlier
/// candidate; a candidate must beat an earlier one by kOrderingMargin of its score. Pure function.
std::size_t choose_ordering(const std::vector<OrderingCandidate>& candidates);
/// A later candidate must score below (1 - kOrderingMargin) times an earlier one to be preferred.
constexpr double kOrderingMargin = 0.05;

/// The analysis loop of adaptive_analysis on candidates priced by callbacks (pure: testable without
/// MUMPS). cand[0] must be AMD; candidates' `ordering` set. analyse(i) gives candidate i its actual
/// estimate (analysed, score_s); predict(i) re-prices the candidates not analysed yet from the
/// analysed candidate i; release(i) frees i's analysis. Calls analyse(0) and predict(0), then
/// analyses the best-scoring candidate (choose_ordering) while it is not analysed yet; each newly
/// analysed candidate is compared with the one kept so far by choose_ordering's rule for the two,
/// and the loser is released; a candidate taken on its prediction ends the loop as the choice
/// unless its score is over the cap. Returns the choice: analysed and never released.
std::size_t run_ordering_choice(std::vector<OrderingCandidate>& cand, const std::function<void(std::size_t)>& analyse,
                                const std::function<void(std::size_t)>& predict,
                                const std::function<void(std::size_t)>& release);

/// Adaptive fill-reducing ordering (DESIGN §3.6-3b): AMD probe, then the candidate orderings
/// priced on `work` by the ordering cost model; the most promising one is analysed, its actual
/// estimate replaces the prediction, and so on until an analysed candidate is the best choice
/// (steps.cpp). Every input of the decisions is deterministic (analysis estimates of
/// deterministic orderings, models, the thread count), so identical runs choose the same ordering.
/// cfg.deterministic runs keep the P4 rule (decide_ordering: AMD or METIS).
template <class T>
std::unique_ptr<SparseFactorization<T>> adaptive_analysis(Ctx& c, const FactorOptions& fopts,
                                                          const ShiftedPencil<T>& pencil, double sigma0,
                                                          const PlannedWork& work);

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

/// Block width of the auto rule, used for runs of at least kAutoBlockMinNev wanted pairs.
constexpr index_t kMaxAutoBlockSize = 4;
constexpr index_t kAutoBlockMinNev = 8;

/// Block width of a shift-invert Lanczos run of the definite path for nev wanted pairs:
/// cfg.block_size if > 0, else the auto rule kMaxAutoBlockSize for nev >= kAutoBlockMinNev and 1
/// below (measurements in steps.cpp). Deterministic (no timing enters).
index_t block_size_for(const SolverConfig& cfg, index_t nev);

/// p, halved while the basis of a run of width p (lanczos_basis_columns, plus `extra_columns`
/// n-vectors the caller keeps, e.g. the candidate block X, AX, BX) would not fit the memory left
/// in the budget (warning when reduced).
template <class T>
index_t fit_block_size(Ctx& c, index_t n, index_t nev, index_t d, index_t extra_columns, index_t p);

/// Block width of a deflated rerun that misses `missing` pairs after a run with width p:
/// max(p, min(kMaxAutoBlockSize, missing rounded up to even)), so that a random start block has
/// components in every missing copy of a multiple eigenvalue of multiplicity up to that width and
/// one run finds them all (eigen.hpp, block mode). More missing copies than the width are left to
/// rounding and further reruns; see steps.cpp for why it is capped.
index_t rerun_block_size(index_t p, index_t missing);

/// Ritz estimates of the ends of the spectrum of the pencil (from inside): lowest two and
/// highest two Ritz values of a short Lanczos run (A alone for B = I, B^{-1}A otherwise).
struct SpectrumEstimate {
  double lo1 = 0.0, lo2 = 0.0, hi1 = 0.0, hi2 = 0.0;
  bool valid = false;
};

// ------------------------------------------------------------------------------ definiteness of B

enum class Definiteness { Positive, NotPositive, Unknown };

/// A Ritz value of B's norm-estimate Lanczos below -kIndefiniteRitz * ||B||_2 proves B indefinite
/// (test_b_definiteness).
constexpr double kIndefiniteRitz = 1e-10;

/// hs::solve gives the attempts of prove_positive_definite (norms.hpp) at most this fraction of the
/// predicted time of the factorization of B a proof saves (factorization_s of the main analysis'
/// estimate, against hpd_attempt_seconds). Beyond the attempts a proof costs its Lanczos run and,
/// once theta passes the gate, the stored S and the level-1 pattern. At 16 threads (bench/
/// anaint_bproof.cpp, P5c) a failed proof cost 0.054 s on ss-tem27623 against a 0.058 s
/// factorization of B (the worst ratio; theta passed the gate, the attempt failed), 0.001..0.09 s
/// elsewhere (ss-tem152078 0.09 s against 0.6 s, before the Lanczos ran on the implicitly scaled B);
/// a successful one 0.004..0.08 s against 0.026..0.20 s.
constexpr double kBProofBudget = 0.5;

/// Whether B is HPD, as far as decided before the analysis, without a factorization.
struct BDefiniteness {
  Definiteness verdict = Definiteness::Unknown;
  Inertia inertia;           ///< valid when the test yields it (diagonal B, dominant B)
  std::string how = "none";  ///< "diagonal", "dominant", "diagonal_entry", "lanczos"; "none": Unknown
};

/// The cheap tests of B's definiteness (DESIGN §3.6-1), in this order: a diagonal B by the signs of
/// its diagonal; strict diagonal dominance with a positive diagonal (HPD by Gershgorin); a diagonal
/// entry <= 0 (not HPD); a Ritz value of `nb` - the norm-estimate Lanczos of B (estimate_norm2) -
/// below -kIndefiniteRitz * nb.norm2 (not HPD: Ritz values stay inside the spectrum's hull up to
/// O(eps ||B||_2)). Unknown: B's inertia must be counted. O(nnz(B)); does not modify b.
template <class T>
BDefiniteness test_b_definiteness(const CsrMatrix<T>& b, const NormEstimate& nb);

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

/// GHIEP: |lambda| <= kGhiepZeroBand * scale is numerically zero - the band of A's null-space
/// cluster, kept away from the shifts (indefinite.cpp; zero_filter_tol can widen it).
constexpr double kGhiepZeroBand = 1e-8;

/// Krylov block size of a GHIEP shift-invert run with nev wanted pairs on an order-n pencil, before
/// the memory fit: cfg.block_size if positive - capped at nev (a block wider than the wanted set
/// only adds basis work), 64 and n / 16, and 1 for the side probes (nev <= 2) - else 2 for
/// n >= 2e4 and nev >= 4, 1 below (measurements in indefinite.cpp). Deterministic.
index_t ghiep_block_size(const SolverConfig& cfg, index_t n, index_t nev);

/// n-vectors of T the GHIEP path keeps beside the factorization (memory plan of DESIGN §3.6-3,
/// reserved before the analysis): the first run's krylov_schur_general_peak_columns (block size of
/// ghiep_block_size, refinement workspace for single / BLR factors) plus the plain run's Schur
/// basis and the union refinement basis held while later runs factorize (indefinite.cpp).
template <class T>
index_t ghiep_reserve_columns(const SolverConfig& cfg, index_t n);

}  // namespace hs::driver
