// hs::solve - the driver (docs/DESIGN.md §3.6). Orchestrates diagnosis, analysis within the memory
// budget, the problem-type decision and the solve path (definite.cpp / indefinite.cpp); all
// numerics live in the lower layers. Shared steps: internal.hpp / steps.cpp.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "io/csr_assembly.hpp"

namespace hs {

namespace {

using driver::Ctx;
using driver::event;

/// hs::solve sets the calling thread's OpenMP thread count when cfg.threads > 0 (MUMPS and every
/// parallel loop of the solve follow it). A library call must not leave its host process changed:
/// the previous value is restored when the solve returns or throws.
class OmpThreadsScope {
 public:
  explicit OmpThreadsScope(int threads) {
#ifdef _OPENMP
    if (threads > 0) {
      previous_ = omp_get_max_threads();
      omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
  }
  ~OmpThreadsScope() {
#ifdef _OPENMP
    if (previous_ > 0) omp_set_num_threads(previous_);
#endif
  }
  OmpThreadsScope(const OmpThreadsScope&) = delete;
  OmpThreadsScope& operator=(const OmpThreadsScope&) = delete;

 private:
  int previous_ = 0;  ///< 0: nothing to restore
};

// Every stored value finite (hs::load_matrix rejects non-finite values too; the scaled defect
// below requires it), and at least one nonzero: the zero matrix has no eigenproblem to solve
// (A = 0: every eigenvalue is 0; B = 0: no finite eigenvalue) and would otherwise fail deep in
// the solve (zero norm estimates, a singular K(sigma) at every shift).
template <class T>
void check_values(const CsrMatrix<T>& m, const char* name) {
  bool finite = true, nonzero = false;
  const index_t nnz = m.nnz();
#pragma omp parallel for reduction(&& : finite) reduction(|| : nonzero) schedule(static)
  for (index_t p = 0; p < nnz; ++p) {
    const T v = m.values[p];
    finite = finite && std::isfinite(std::real(v)) && std::isfinite(std::imag(v));
    nonzero = nonzero || v != T(0);
  }
  if (!finite) {
    for (index_t i = 0; i < m.nrows; ++i)
      for (index_t p = m.rowptr[i]; p < m.rowptr[i + 1]; ++p)
        if (!std::isfinite(std::real(m.values[p])) || !std::isfinite(std::imag(m.values[p])))
          HS_THROW(InvalidInput, name << " has a non-finite entry at (" << i << ", " << m.colind[p] << ")");
  }
  if (!nonzero)
    HS_THROW(InvalidInput, name << " is the zero matrix (no nonzero entry): "
                                << (name[0] == 'B' ? "the pencil (A, 0) has no finite eigenvalues"
                                                   : "every eigenvalue is 0, there is nothing to compute"));
}

// Exactly Hermitian working copy of an input matrix (DESIGN §3.6-1), by the rules of hs::load_matrix
// (src/io/load.cpp), so that in-memory input (Python) and file input are treated alike at any
// magnitude: the relative defect ||M - M^H||_F / ||M||_F comes from scaled sums of squares
// (io::hermitian_defect_scaled; the unscaled hs::hermitian_defect reports 0 for entries below
// ~1e-154 and NaN above ~1e154, which let a non-Hermitian matrix through), and the Hermitian part
// is taken exactly when some m_ij != conj(m_ji) - decided by comparing values, not by the norm.
template <class T>
const CsrMatrix<T>* hermitian_input(const CsrMatrix<T>& m, double tol, const char* name, double& defect,
                                    CsrMatrix<T>& store) {
  m.validate();
  if (!m.square()) HS_THROW(InvalidInput, name << " is not square (" << m.nrows << "x" << m.ncols << ")");
  if (m.nrows == 0) HS_THROW(InvalidInput, "empty matrix");
  check_values(m, name);
  const io::HermitianDefect d = io::hermitian_defect_scaled(m);
  defect = d.relative;
  if (!(d.relative <= tol))
    HS_THROW(NotHermitian, name << " is not Hermitian: ||M-M^H||_F/||M||_F = " << d.relative << " > hermitian_tol " << tol);
  if (!d.exact) {
    store = hermitian_part(m);
    return &store;
  }
  return &m;
}

// Representative shift of the symbolic analysis (its values feed MUMPS' analysis-time matching):
// the target shift, the interval centre, or just beyond the estimated end of the spectrum
// (generalized problems: A's Ritz estimates divided by ||B||_2, a rough guess that is enough here).
double initial_shift(const Target& t, const driver::NormEstimate& na, double scale, double norm2_b) {
  switch (t.mode) {
    case TargetMode::Nearest: return t.sigma;
    case TargetMode::Interval: return 0.5 * (t.lower + t.upper);
    case TargetMode::Smallest:
    case TargetMode::Largest: {
      const double nb = norm2_b > 0 ? norm2_b : 1.0;
      const double spread = std::max(na.ritz_max - na.ritz_min, 1e-12 * std::max(scale, 1.0)) / nb;
      return t.mode == TargetMode::Smallest ? na.ritz_min / nb - 0.05 * spread : na.ritz_max / nb + 0.05 * spread;
    }
  }
  return t.sigma;
}

// Is B diagonal (lumped / Yee mass matrices)?
template <class T>
bool is_diagonal(const CsrMatrix<T>& b) {
  const index_t n = b.nrows;
  if (b.nnz() > n) return false;
  for (index_t i = 0; i < n; ++i) {
    const index_t len = b.rowptr[i + 1] - b.rowptr[i];
    if (len > 1 || (len == 1 && b.colind[b.rowptr[i]] != i)) return false;
  }
  return true;
}

using driver::Definiteness;

// O(nnz) screen of a Hermitian B before its factorization: every row strictly diagonally
// dominant with a positive diagonal proves B HPD (Gershgorin); a diagonal entry <= 0 (e_i^H B e_i)
// proves it is not. Everything else needs the inertia of B.
template <class T>
Definiteness screen_definiteness(const CsrMatrix<T>& b) {
  bool dominant = true;
  for (index_t i = 0; i < b.nrows; ++i) {
    double diag = 0.0, off = 0.0;
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p) {
      if (b.colind[p] == i) {
        diag = std::real(b.values[p]);
      } else {
        off += std::abs(b.values[p]);
      }
    }
    if (!(diag > 0.0)) return Definiteness::NotPositive;
    dominant = dominant && diag > off;
  }
  return dominant ? Definiteness::Positive : Definiteness::Unknown;
}

}  // namespace

// Tests that decide before the analysis, cheapest first (internal.hpp):
//   * a diagonal B: HPD iff its diagonal is positive (lumped / Yee mass matrices);
//   * screen_definiteness: O(nnz), dominant with a positive diagonal proves HPD, a diagonal entry
//     <= 0 proves not HPD;
//   * the norm-estimate Lanczos of B (already run for ||B||_2): its Ritz values lie in
//     [lambda_min(B), lambda_max(B)] up to O(eps ||B||_2) (Paige: without reorthogonalization
//     Ritz values may repeat, they do not leave the spectrum's hull), so one below
//     -kIndefiniteRitz * ||B||_2 - about 1e6 times the rounding level - proves a negative
//     eigenvalue. Lee-Sun-Cendes waveguide pencils show one at once (B's negative part is not
//     small); before, their B was factorized for its inertia: 0.19 s, 10% of wg-square-1e5 (P4).
// Unknown: B's inertia on the main analysis decides (one factorization).
template <class T>
driver::BDefiniteness driver::test_b_definiteness(const CsrMatrix<T>& b, const NormEstimate& nb) {
  BDefiniteness r;
  const index_t n = b.nrows;
  if (is_diagonal(b)) {
    r.inertia.neg = r.inertia.zero = r.inertia.pos = 0;
    for (index_t i = 0; i < n; ++i) {
      const double d = (b.rowptr[i + 1] > b.rowptr[i]) ? std::real(b.values[b.rowptr[i]]) : 0.0;
      (d > 0.0 ? r.inertia.pos : d < 0.0 ? r.inertia.neg : r.inertia.zero)++;
    }
    r.inertia.valid = true;
    r.verdict = (r.inertia.neg == 0 && r.inertia.zero == 0) ? Definiteness::Positive : Definiteness::NotPositive;
    r.how = "diagonal";
    return r;
  }
  r.verdict = screen_definiteness(b);
  if (r.verdict == Definiteness::Positive) {
    r.inertia.neg = r.inertia.zero = 0;
    r.inertia.pos = n;
    r.inertia.valid = true;
    r.how = "dominant";
  } else if (r.verdict == Definiteness::NotPositive) {
    r.how = "diagonal_entry";  // not HPD; its inertia is not needed (counts stay -1, valid == false)
  } else if (nb.steps > 0 && nb.ritz_min < -kIndefiniteRitz * nb.norm2) {
    r.verdict = Definiteness::NotPositive;
    r.how = "lanczos";
  }
  return r;
}

// ------------------------------------------------------------------------------------- solve

template <class T>
Solution<T> solve(const CsrMatrix<T>& a_in, const non_deduced_t<CsrMatrix<T>>* b_in, const SolverConfig& cfg,
                  Logger* log) {
  const OmpThreadsScope omp_threads(cfg.threads);
  driver::PhaseClock clock;  // starts in Phase::Setup
  SolveReport rep;
  Ctx c{cfg, log, rep, clock};

  // 1. Diagnosis: exact Hermitian inputs, norms, scale.
  CsrMatrix<T> a_store, b_store;
  const CsrMatrix<T>* a = hermitian_input(a_in, cfg.hermitian_tol, "A", rep.hermitian_defect_a, a_store);
  const CsrMatrix<T>* b = nullptr;
  if (b_in) {
    if (b_in->nrows != a->nrows) HS_THROW(InvalidInput, "B has order " << b_in->nrows << ", A has " << a->nrows);
    b = hermitian_input(*b_in, cfg.hermitian_tol, "B", rep.hermitian_defect_b, b_store);
  }
  const index_t n = a->nrows;
  rep.n = n;
  rep.nnz_a = a->nnz();
  rep.nnz_b = b ? b->nnz() : 0;
  rep.complex_scalars = is_complex_v<T>;
  const auto na = driver::estimate_norm2(*a, 30, cfg.seed);
  const driver::NormEstimate nb = b ? driver::estimate_norm2(*b, 30, cfg.seed) : driver::NormEstimate{};
  rep.norm2_a = na.norm2;
  rep.norm2_b = b ? nb.norm2 : 1.0;
  rep.scale = rep.norm2_b > 0 ? rep.norm2_a / rep.norm2_b : rep.norm2_a;
  const double scale = std::max(rep.scale, 1e-300);
  event(c, "problem", {{"n", static_cast<std::int64_t>(n)}, {"nnz_a", static_cast<std::int64_t>(rep.nnz_a)},
                       {"nnz_b", static_cast<std::int64_t>(rep.nnz_b)}, {"complex", rep.complex_scalars},
                       {"norm2_a", rep.norm2_a}, {"norm2_b", rep.norm2_b}});

  ShiftedPencil<T> pencil(*a, b);
  if (b && cfg.problem == ProblemType::Standard) HS_THROW(InvalidInput, "problem 'standard' requested but B was given");
  if (!b && (cfg.problem == ProblemType::GeneralizedDefinite || cfg.problem == ProblemType::GeneralizedIndefinite))
    HS_THROW(InvalidInput, "a generalized problem type was requested but no B was given");
  if (cfg.method == Method::Lanczos)
    HS_THROW(Unsupported, "method 'lanczos' (factorization-free) is not implemented; use 'auto', "
                          "'shift_invert_lanczos' or 'shift_invert_krylov_schur'");

  // 2. Problem type as far as it is known before the analysis: given, B = I, or B's definiteness
  //    proven without a factorization (test_b_definiteness). Auto here: B's inertia on the main
  //    analysis decides below. The memory plan and the ordering choice follow the planned type.
  ProblemType type = cfg.problem;
  Timer tb;
  driver::BDefiniteness bdef;
  if (!b) {
    type = ProblemType::Standard;
  } else if (type == ProblemType::Auto) {
    bdef = driver::test_b_definiteness(*b, nb);
    if (bdef.verdict != Definiteness::Unknown)
      type = bdef.verdict == Definiteness::Positive ? ProblemType::GeneralizedDefinite
                                                    : ProblemType::GeneralizedIndefinite;
  }
  auto b_inertia_event = [&](const Inertia& ib, const std::string& how) {
    event(c, "b_inertia", {{"neg", static_cast<std::int64_t>(ib.neg)}, {"zero", static_cast<std::int64_t>(ib.zero)},
                           {"pos", static_cast<std::int64_t>(ib.pos)}, {"valid", ib.valid},
                           {"screened", how != "diagonal" && how != "factorization" && how != "compute_inertia"},
                           {"how", how}, {"ritz_min", nb.ritz_min}, {"seconds", tb.seconds()}});
  };
  if (b && cfg.problem == ProblemType::Auto && type != ProblemType::Auto) b_inertia_event(bdef.inertia, bdef.how);

  // 3. Memory budget for the factorization (DESIGN §3.6-3): what the solve path keeps beside the
  //    factors - the block Lanczos basis plus candidate blocks (definite), or the general
  //    Krylov-Schur peak plus the bases held for the union refinement (indefinite); both while the
  //    type is still open.
  const double budget = cfg.memory_budget_gb * 1073741824.0;
  // Definite path: the block Lanczos basis of the P4 width (reserved_block_size; DESIGN §3.6-3;
  // block layout of eigen.hpp). A wider width - an explicit block_size, or the blocktune rule's
  // 8 for a nearest target, picked after the factorization - is not reserved: the factorization
  // has priority, and fit_block_size() takes the wider width only where the factors leave room
  // for its basis.
  const double free_bytes =
      budget - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0));
  const driver::KrylovReservation reserve = driver::krylov_reservation<T>(cfg, type, n);
  const double krylov_bytes = static_cast<double>(n) * sizeof(T) * static_cast<double>(reserve.columns);
  const double cap_mb = (free_bytes - krylov_bytes) * 0.95 / 1e6;
  event(c, "memory_plan", {{"reserved_block_size", static_cast<std::int64_t>(reserve.block_size)},
                           {"reserved_columns", static_cast<std::int64_t>(reserve.columns)},
                           {"krylov_mb", krylov_bytes / 1e6}, {"cap_mb", cap_mb}});
  if (cap_mb <= 0)
    HS_THROW(OutOfMemoryBudget, "matrices and Krylov basis alone exceed memory_budget_gb = " << cfg.memory_budget_gb);
  const FactorOptions fopts = driver::factor_options_from(cfg, cap_mb);
  rep.factor_options = fopts;

  // 4. One analysis serves every factorization on the pencil's pattern: the B-definiteness test,
  //    K(sigma), certificate endpoints and reruns. ordering < 0 selects it adaptively, priced on
  //    the work the planned problem type will do with the factors (plan_work).
  const double sigma0 = initial_shift(cfg.target, na, scale, rep.norm2_b);
  std::unique_ptr<SparseFactorization<T>> fac;
  if (cfg.ordering < 0) {
    const driver::PlannedWork work = driver::plan_work(cfg, type, n, scale);
    fac = driver::adaptive_analysis(c, fopts, pencil, sigma0, work);
    driver::analyze_within_budget(c, *fac, pencil, sigma0, /*already_analyzed=*/true);
  } else {
    fac = make_mumps_factorization<T>(fopts);
    driver::analyze_within_budget(c, *fac, pencil, sigma0);
  }
  rep.factor_options = fac->options();

  // 5. B's definiteness when the cheap tests left it open. First the rigorous incomplete-Cholesky
  //    proof (driver::prove_positive_definite, norms.hpp: B HPD, or B indefinite through a Lanczos
  //    run on the scaled B) within kBProofBudget of the predicted time of the factorization it
  //    saves; smallest / largest targets factorize B anyway (spectrum estimate below) and skip it.
  //    Otherwise B's inertia: B is HPD iff its exact LDL^T has no negative or zero pivots
  //    (Sylvester), counted on the main analysis.
  const bool b_diagonal = b && is_diagonal(*b);
  const bool one_end = cfg.target.mode == TargetMode::Smallest || cfg.target.mode == TargetMode::Largest;
  if (b && type == ProblemType::Auto && !(one_end && !b_diagonal)) {
    driver::PhaseScope phase(clock, driver::Phase::Factor);
    tb.reset();
    const FactorEstimate est = fac->estimate();
    driver::HpdProofOptions po;
    po.max_seconds = driver::kBProofBudget * driver::factorization_s(driver::mumps_input_size(pencil), est.flops,
                                                                    static_cast<double>(est.factor_entries),
                                                                    fopts.threads);
    // Its memory: what the budget leaves beside the process now (the proof's arrays are freed
    // before anything else is allocated), and at most the factorization cap.
    po.max_bytes = std::max(0.0, budget - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0)));
    if (fopts.max_memory_mb > 0) po.max_bytes = std::min(po.max_bytes, fopts.max_memory_mb * 1e6);
    po.seed = cfg.seed;
    const driver::HpdProof proof = driver::prove_positive_definite(*b, po);
    event(c, "b_proof", {{"outcome", proof.outcome}, {"proven", proof.proven}, {"indefinite", proof.indefinite},
                         {"lower_bound", proof.lower_bound}, {"theta_min", proof.theta_min},
                         {"theta_max", proof.theta_max}, {"level", static_cast<std::int64_t>(proof.level)},
                         {"shift", proof.shift}, {"attempts", static_cast<std::int64_t>(proof.attempts)},
                         {"budget_s", po.max_seconds}, {"max_bytes", po.max_bytes}, {"seconds", proof.seconds},
                         {"setup_s", proof.setup_s}, {"lanczos_s", proof.lanczos_s}, {"pattern_s", proof.pattern_s},
                         {"pattern_predicted_s", proof.pattern_predicted_s}, {"attempts_s", proof.attempts_s},
                         {"stored_bytes", proof.stored_bytes}, {"level1_bound", proof.level1_bound},
                         {"level1_entries", static_cast<std::int64_t>(proof.level1_entries)}});
    if (proof.proven) {
      Inertia ib;
      ib.neg = ib.zero = 0;
      ib.pos = n;
      ib.valid = true;
      type = ProblemType::GeneralizedDefinite;
      b_inertia_event(ib, "incomplete_cholesky");
    } else if (proof.indefinite) {
      type = ProblemType::GeneralizedIndefinite;
      b_inertia_event(Inertia{}, "lanczos_scaled");
    }
  }
  bool b_factorized = false;  // the main object holds the factorization of B
  if (b && type == ProblemType::Auto) {
    tb.reset();
    Inertia ib;
    std::string how;
    if (fopts.precision == FactorPrecision::Double && !fopts.blr) {
      driver::PhaseScope phase(clock, driver::Phase::Factor);
      if (fac->factorize(0.0, 1.0)) ib = fac->inertia();  // false: B singular -> not HPD
      b_factorized = ib.valid;
      how = "factorization";
    } else {
      driver::PhaseScope phase(clock, driver::Phase::Factor);
      ib = compute_inertia(pencil, 0.0, 1.0, fopts);  // forces an exact double factorization
      how = "compute_inertia";
    }
    type = (ib.valid && ib.neg == 0 && ib.zero == 0) ? ProblemType::GeneralizedDefinite
                                                     : ProblemType::GeneralizedIndefinite;
    b_inertia_event(ib, how);
  }
  rep.problem = type;
  Method method = cfg.method;
  if (method == Method::Auto)
    method = type == ProblemType::GeneralizedIndefinite ? Method::ShiftInvertKrylovSchur : Method::ShiftInvertLanczos;
  if (method == Method::ShiftInvertLanczos && type == ProblemType::GeneralizedIndefinite)
    HS_THROW(InvalidInput, "shift_invert_lanczos needs B positive definite; B is indefinite");
  rep.method = method;

  // Smallest / largest: Ritz estimates of the ends of the spectrum seed the inertia search
  // (A alone, or B^{-1}A in the B inner product with B's factorization on the main analysis).
  driver::SpectrumEstimate spectrum;
  if (one_end && method == Method::ShiftInvertLanczos) {
    driver::NormEstimate e = na;
    if (b) {
      driver::SolveFn<T> solve_b;
      std::vector<double> inv_diag;
      if (b_diagonal) {
        for (index_t i = 0; i < n; ++i)
          inv_diag.push_back(b->rowptr[i + 1] > b->rowptr[i] ? 1.0 / std::real(b->values[b->rowptr[i]]) : 0.0);
        solve_b = [&](T* x) {
          for (index_t i = 0; i < n; ++i) x[i] *= inv_diag[i];
        };
      } else {
        if (!b_factorized) {
          driver::PhaseScope phase(clock, driver::Phase::Factor);
          if (!fac->factorize(0.0, 1.0)) HS_THROW(InvalidInput, "B is singular: not a definite generalized problem");
        }
        c.held = driver::FactorHeld{};
        solve_b = [&](T* x) { fac->solve(x, 1, n); };
      }
      e = driver::estimate_pencil_spectrum(*a, *b, solve_b, 30, cfg.seed);
    }
    spectrum = {e.ritz_min, e.ritz_min2, e.ritz_max, e.ritz_max2, true};
    event(c, "spectrum_estimate", {{"lo1", spectrum.lo1}, {"lo2", spectrum.lo2}, {"hi1", spectrum.hi1}, {"hi2", spectrum.hi2}});
  }

  // 6-11. Solve along the chosen path.
  Solution<T> sol = (method == Method::ShiftInvertKrylovSchur)
                        ? driver::solve_indefinite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale)
                        : driver::solve_definite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale, sigma0, spectrum);
  rep.factor_stats = fac->stats();
  // Exclusive phase times: they add up to the total by construction (driver::PhaseClock).
  rep.time_setup_s = clock.seconds(driver::Phase::Setup);
  rep.time_factor_s = clock.seconds(driver::Phase::Factor);
  rep.time_iterate_s = clock.seconds(driver::Phase::Iterate);
  rep.time_verify_s = clock.seconds(driver::Phase::Verify);
  rep.time_certificate_s = clock.seconds(driver::Phase::Certificate);
  rep.time_total_s =
      rep.time_setup_s + rep.time_factor_s + rep.time_iterate_s + rep.time_verify_s + rep.time_certificate_s;
  rep.peak_rss_bytes = peak_rss_bytes();
  event(c, "done", {{"success", rep.success}, {"pairs", static_cast<std::int64_t>(rep.pairs.size())},
                    {"seconds", rep.time_total_s}});
  sol.report = std::move(rep);
  return sol;
}

#define HS_INSTANTIATE_SOLVE(T)                                                                     \
  template Solution<T> solve<T>(const CsrMatrix<T>&, const non_deduced_t<CsrMatrix<T>>*, const SolverConfig&, \
                                Logger*);                                                               \
  template driver::BDefiniteness driver::test_b_definiteness<T>(const CsrMatrix<T>&, const driver::NormEstimate&);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE)
#undef HS_INSTANTIATE_SOLVE

}  // namespace hs
