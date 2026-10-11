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

namespace hs {

namespace {

using driver::Ctx;
using driver::event;

// Exactly Hermitian working copy of an input matrix (DESIGN §3.6-1).
template <class T>
const CsrMatrix<T>* hermitian_input(const CsrMatrix<T>& m, double tol, const char* name, double& defect,
                                    CsrMatrix<T>& store) {
  m.validate();
  if (!m.square()) HS_THROW(InvalidInput, name << " is not square (" << m.nrows << "x" << m.ncols << ")");
  defect = hermitian_defect(m);
  if (defect > tol)
    HS_THROW(NotHermitian, name << " is not Hermitian: ||M-M^H||_F/||M||_F = " << defect << " > hermitian_tol " << tol);
  if (defect > 0.0) {
    store = hermitian_part(m);
    return &store;
  }
  return &m;
}

// Shift used for the analysis and the main factorization (Smallest/Largest: just beyond the
// estimated end of the spectrum; standard problems only).
double initial_shift(const Target& t, const driver::NormEstimate& na, double scale, bool has_b) {
  switch (t.mode) {
    case TargetMode::Nearest: return t.sigma;
    case TargetMode::Interval: return 0.5 * (t.lower + t.upper);
    case TargetMode::Smallest:
    case TargetMode::Largest: {
      if (has_b) HS_THROW(Unsupported, "smallest/largest targets are implemented for standard problems only (roadmap P3)");
      const double spread = std::max(na.ritz_max - na.ritz_min, 1e-12 * std::max(scale, 1.0));
      return t.mode == TargetMode::Smallest ? na.ritz_min - 0.05 * spread : na.ritz_max + 0.05 * spread;
    }
  }
  return t.sigma;
}

}  // namespace

// ------------------------------------------------------------------------------------- solve

template <class T>
Solution<T> solve(const CsrMatrix<T>& a_in, const non_deduced_t<CsrMatrix<T>>* b_in, const SolverConfig& cfg,
                  Logger* log) {
  driver::PhaseClock clock;  // starts in Phase::Setup
  SolveReport rep;
  Ctx c{cfg, log, rep, clock};
#ifdef _OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#endif

  // 1. Diagnosis: exact Hermitian inputs, norms, scale.
  CsrMatrix<T> a_store, b_store;
  const CsrMatrix<T>* a = hermitian_input(a_in, cfg.hermitian_tol, "A", rep.hermitian_defect_a, a_store);
  const CsrMatrix<T>* b = nullptr;
  if (b_in) {
    if (b_in->nrows != a->nrows) HS_THROW(InvalidInput, "B has order " << b_in->nrows << ", A has " << a->nrows);
    b = hermitian_input(*b_in, cfg.hermitian_tol, "B", rep.hermitian_defect_b, b_store);
  }
  const index_t n = a->nrows;
  if (n == 0) HS_THROW(InvalidInput, "empty matrix");
  rep.n = n;
  rep.nnz_a = a->nnz();
  rep.nnz_b = b ? b->nnz() : 0;
  rep.complex_scalars = is_complex_v<T>;
  const auto na = driver::estimate_norm2(*a, 30, cfg.seed);
  rep.norm2_a = na.norm2;
  rep.norm2_b = b ? driver::estimate_norm2(*b, 30, cfg.seed).norm2 : 1.0;
  rep.scale = rep.norm2_b > 0 ? rep.norm2_a / rep.norm2_b : rep.norm2_a;
  const double scale = std::max(rep.scale, 1e-300);
  event(c, "problem", {{"n", static_cast<std::int64_t>(n)}, {"nnz_a", static_cast<std::int64_t>(rep.nnz_a)},
                       {"nnz_b", static_cast<std::int64_t>(rep.nnz_b)}, {"complex", rep.complex_scalars},
                       {"norm2_a", rep.norm2_a}, {"norm2_b", rep.norm2_b}});

  ShiftedPencil<T> pencil(*a, b);
  if (b && cfg.problem == ProblemType::Standard) HS_THROW(InvalidInput, "problem 'standard' requested but B was given");
  if (!b && (cfg.problem == ProblemType::GeneralizedDefinite || cfg.problem == ProblemType::GeneralizedIndefinite))
    HS_THROW(InvalidInput, "a generalized problem type was requested but no B was given");
  if (cfg.method == Method::Lanczos) HS_THROW(Unsupported, "factorization-free Lanczos is roadmap P3");

  // 2. Memory budget for the factorization (DESIGN §3.6-3).
  const double budget = cfg.memory_budget_gb * 1073741824.0;
  const index_t k_guess = cfg.target.mode == TargetMode::Interval ? std::min<index_t>(n, 64) : cfg.target.k;
  const index_t ncv_guess = cfg.ncv > 0 ? cfg.ncv : std::max<index_t>(2 * (k_guess + 16) + 1, k_guess + 32);
  const double krylov_bytes = static_cast<double>(n) * sizeof(T) * static_cast<double>(ncv_guess + 6 * k_guess);
  const double cap_mb = (budget - static_cast<double>(std::max<std::int64_t>(current_rss_bytes(), 0)) - krylov_bytes) * 0.95 / 1e6;
  if (cap_mb <= 0)
    HS_THROW(OutOfMemoryBudget, "matrices and Krylov basis alone exceed memory_budget_gb = " << cfg.memory_budget_gb);
  const FactorOptions fopts = driver::factor_options_from(cfg, cap_mb);
  rep.factor_options = fopts;
  // 3. One analysis serves every factorization on the pencil's pattern: the B-definiteness test,
  //    K(sigma), certificate endpoints and reruns. ordering < 0 selects it adaptively.
  const double sigma0 = initial_shift(cfg.target, na, scale, b != nullptr);
  std::unique_ptr<SparseFactorization<T>> fac;
  if (cfg.ordering < 0) {
    const index_t kk = cfg.target.mode == TargetMode::Interval ? 32 : cfg.target.k;
    const bool cert = cfg.inertia_check != Toggle::Off;
    fac = driver::adaptive_analysis(c, fopts, pencil, sigma0, 1 + (cert ? 2 : 0) + (b ? 1 : 0), std::max<index_t>(40, 3 * kk));
    driver::analyze_within_budget(c, *fac, pencil, sigma0, /*already_analyzed=*/true);
  } else {
    fac = make_mumps_factorization<T>(fopts);
    driver::analyze_within_budget(c, *fac, pencil, sigma0);
  }
  rep.factor_options = fac->options();

  // 4. Problem type: B is HPD iff its exact LDL^T has no negative or zero pivots (Sylvester).
  ProblemType type = cfg.problem;
  if (!b) {
    type = ProblemType::Standard;
  } else if (type == ProblemType::Auto) {
    Timer tb;
    Inertia ib;
    // A diagonal B (e.g. lumped/Yee mass matrices) is HPD iff its diagonal is positive: no
    // factorization needed. Otherwise count the inertia of B on the main analysis.
    bool diagonal = b->nnz() <= n;
    for (index_t i = 0; diagonal && i < n; ++i)
      diagonal = (b->rowptr[i + 1] - b->rowptr[i] == 0) || (b->rowptr[i + 1] - b->rowptr[i] == 1 && b->colind[b->rowptr[i]] == i);
    if (diagonal) {
      ib.neg = ib.zero = ib.pos = 0;
      for (index_t i = 0; i < n; ++i) {
        const double d = (b->rowptr[i + 1] > b->rowptr[i]) ? std::real(b->values[b->rowptr[i]]) : 0.0;
        (d > 0.0 ? ib.pos : d < 0.0 ? ib.neg : ib.zero)++;
      }
      ib.valid = true;
    } else if (fopts.precision == FactorPrecision::Double && !fopts.blr) {
      driver::PhaseScope phase(clock, driver::Phase::Factor);
      if (fac->factorize(0.0, 1.0)) ib = fac->inertia();  // false: B singular -> not HPD
    } else {
      driver::PhaseScope phase(clock, driver::Phase::Factor);
      ib = compute_inertia(pencil, 0.0, 1.0, fopts);  // forces an exact double factorization
    }
    type = (ib.valid && ib.neg == 0 && ib.zero == 0) ? ProblemType::GeneralizedDefinite
                                                     : ProblemType::GeneralizedIndefinite;
    event(c, "b_inertia", {{"neg", static_cast<std::int64_t>(ib.neg)}, {"zero", static_cast<std::int64_t>(ib.zero)},
                           {"pos", static_cast<std::int64_t>(ib.pos)}, {"valid", ib.valid}, {"seconds", tb.seconds()}});
  }
  rep.problem = type;
  Method method = cfg.method;
  if (method == Method::Auto)
    method = type == ProblemType::GeneralizedIndefinite ? Method::ShiftInvertKrylovSchur : Method::ShiftInvertLanczos;
  if (method == Method::ShiftInvertLanczos && type == ProblemType::GeneralizedIndefinite)
    HS_THROW(InvalidInput, "shift_invert_lanczos needs B positive definite; B is indefinite");
  rep.method = method;

  // 5-11. Solve along the chosen path.
  Solution<T> sol = (method == Method::ShiftInvertKrylovSchur)
                        ? driver::solve_indefinite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale)
                        : driver::solve_definite(c, pencil, *fac, rep.norm2_a, rep.norm2_b, scale, sigma0);
  rep.factor_stats = fac->stats();
  // Exclusive phase times: they add up to the total by construction (driver::PhaseClock).
  rep.time_setup_s = clock.seconds(driver::Phase::Setup);
  rep.time_factor_s = clock.seconds(driver::Phase::Factor);
  rep.time_iterate_s = clock.seconds(driver::Phase::Iterate);
  rep.time_verify_s = clock.seconds(driver::Phase::Verify);
  rep.time_total_s = rep.time_setup_s + rep.time_factor_s + rep.time_iterate_s + rep.time_verify_s;
  rep.peak_rss_bytes = peak_rss_bytes();
  event(c, "done", {{"success", rep.success}, {"pairs", static_cast<std::int64_t>(rep.pairs.size())},
                    {"seconds", rep.time_total_s}});
  sol.report = std::move(rep);
  return sol;
}

#define HS_INSTANTIATE_SOLVE(T) \
  template Solution<T> solve<T>(const CsrMatrix<T>&, const non_deduced_t<CsrMatrix<T>>*, const SolverConfig&, Logger*);
HS_FOR_EACH_SCALAR(HS_INSTANTIATE_SOLVE)
#undef HS_INSTANTIATE_SOLVE

}  // namespace hs
