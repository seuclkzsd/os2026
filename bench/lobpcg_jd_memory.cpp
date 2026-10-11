// Memory-limited fallback study: shift-invert Lanczos with a double-precision factor versus
// Jacobi-Davidson preconditioned by a SINGLE-precision factor of A - sigma B (P3, module
// lobpcg_jd). One mode per process, so that the peak RSS reported by tools/run/guard.py belongs to
// that mode alone. Prints one JSON line on stdout.
//
//   bench_lobpcg_jd_memory --case NAME --mode MODE [--k K] [--sigma S] [--tol 1e-10]
//                          [--ordering 3] [--inner 20] [--deterministic 0] [--history 0]
//                          [--max-iterations 1000] [--data-root DIR]
// Modes:
//   si-double  ShiftInvertOperator on a double factor + lanczos_krylov_schur (driver's inner path)
//   si-single  the same on a single factor with double-precision iterative refinement (the
//              driver's existing memory fallback, DESIGN §3.4b)
//   jd-single  jacobi_davidson, preconditioner = solves with a single factor (no refinement)
//   jd-double  jacobi_davidson, preconditioner = solves with a double factor (reference)
//   driver-double / driver-single   hs::solve with factor_precision double / single
// Factorization-free path (the k SMALLEST eigenvalues, shift s below the spectrum: from --sigma,
// else lambda_min estimate - 5% of the spectral spread; B needs --sigma):
//   lobpcg         lobpcg without preconditioner (no factorization at all)
//   lobpcg-single  lobpcg preconditioned by solves with a single factor of A - s B
//   si-smallest    shift-invert Lanczos at s with a double factor (the driver's Smallest path)
// Case data: <data-root>/cases/NAME/{A,B}.npz + config.json (sigma, k); the reference eigenvalues
// <data-root>/refs/NAME/eigenvalues.npy are read only for the accuracy report (evaluation code).
// Run (2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 30,33 --mem-gb 8 --timeout 1200 --
//       build/release/bench_lobpcg_jd_memory --case syn-mem27-46 --mode jd-single
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/norms.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Args {
  std::string case_name, mode, data_root;
  long long k = -1;
  std::optional<double> sigma;
  double tol = 1e-10;
  int ordering = 3;
  long long inner = 20;
  bool deterministic = false;
  bool history = false;
  long long max_iterations = 1000;
};

/// y = K^{-1} x with the factor's solve (no B, no refinement): the JD preconditioner.
template <class T>
class FactorSolveOperator final : public hs::LinearOperator<T> {
 public:
  explicit FactorSolveOperator(hs::SparseFactorization<T>& f) : f_(f) {}
  hs::index_t size() const override { return f_.order(); }

 protected:
  void do_apply(const T* x, hs::index_t ldx, T* y, hs::index_t ldy, hs::index_t p) override {
    const hs::index_t n = size();
    for (hs::index_t c = 0; c < p; ++c) std::copy(x + c * ldx, x + c * ldx + n, y + c * ldy);
    f_.solve(y, p, ldy);
  }

 private:
  hs::SparseFactorization<T>& f_;
};

/// Backward errors of the pairs (lambda, X) with the driver's definition (DESIGN §4.5).
template <class T>
std::vector<double> backward_errors(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::DenseMatrix<T>& x,
                                    const std::vector<double>& lambda, double na, double nb) {
  const hs::index_t n = x.rows(), m = x.cols();
  std::vector<T> ax(static_cast<std::size_t>(n)), bx(static_cast<std::size_t>(n));
  std::vector<double> eta;
  for (hs::index_t j = 0; j < m; ++j) {
    hs::spmv(a, x.col(j), ax.data());
    if (b) {
      hs::spmv(*b, x.col(j), bx.data());
    } else {
      std::copy(x.col(j), x.col(j) + n, bx.begin());
    }
    hs::linalg::axpy(n, T(-lambda[j]), bx.data(), ax.data());
    const double r = hs::linalg::norm2(n, ax.data());
    eta.push_back(r / ((na + std::abs(lambda[j]) * nb) * hs::linalg::norm2(n, x.col(j))));
  }
  return eta;
}

/// Accuracy against the reference: the k values nearest sigma, compared as sorted lists.
json compare_with_reference(const fs::path& ref_file, std::vector<double> got, double sigma, double scale) {
  json out;
  if (!fs::exists(ref_file)) return out;
  auto ref = hs::npy_to_vector<double>(hs::read_npy(ref_file));
  std::stable_sort(ref.begin(), ref.end(),
                   [sigma](double p, double q) { return std::abs(p - sigma) < std::abs(q - sigma); });
  ref.resize(std::min(ref.size(), got.size()));
  std::sort(ref.begin(), ref.end());
  std::sort(got.begin(), got.end());
  double max_abs = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  out["max_abs_err_vs_ref"] = max_abs;
  out["max_rel_err_vs_ref"] = max_abs / scale;  // relative to ||A|| / ||B||, as the set check of compare.py
  out["set_matches_ref"] = max_abs <= 1e-8 * scale;
  return out;
}

template <class T>
json run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, double sigma, hs::index_t k,
         const fs::path& ref_file) {
  json out;
  const auto ea = hs::driver::estimate_norm2(a);
  const double na = ea.norm2;
  const double nb = b ? hs::driver::estimate_norm2(*b).norm2 : 1.0;
  out["norm_a"] = na;
  out["norm_b"] = nb;
  const bool smallest = args.mode == "lobpcg" || args.mode == "lobpcg-single" || args.mode == "si-smallest";
  if (smallest) {
    // The k smallest eigenvalues; shift below the spectrum as the driver places it for Smallest
    // targets (initial_shift in src/driver/solver.cpp). With B the shift must be given.
    if (!args.sigma) {
      if (b) HS_THROW(InvalidInput, "smallest modes with B need --sigma below the spectrum");
      sigma = ea.ritz_min - 0.05 * (ea.ritz_max - ea.ritz_min);
    }
  }
  std::vector<double> lambda;
  hs::DenseMatrix<T> x;
  hs::Timer total;

  if (args.mode.rfind("driver-", 0) == 0) {
    hs::SolverConfig cfg;
    cfg.target.mode = hs::TargetMode::Nearest;
    cfg.target.sigma = sigma;
    cfg.target.k = k;
    cfg.tol = args.tol;
    cfg.factor_precision = args.mode == "driver-single" ? hs::FactorPrecision::Single : hs::FactorPrecision::Double;
    const auto sol = hs::solve(a, b, cfg);
    lambda = sol.eigenvalues;
    x = sol.eigenvectors;
    const auto& r = sol.report;
    out["time_factor_s"] = r.time_factor_s;
    out["time_iterate_s"] = r.time_iterate_s;
    out["success"] = r.success;
    out["certificate_passed"] = r.certificate.passed;
    out["factor_mb_used"] = r.factor_stats.factor_mb_used;
    out["factor_entries"] = static_cast<long long>(r.factor_stats.factor_entries);
    out["solves"] = static_cast<long long>(r.factor_stats.rhs_solved);
    out["factorizations"] = static_cast<long long>(r.factor_stats.factorizations);
  } else {
    const bool single = args.mode == "si-single" || args.mode == "jd-single" || args.mode == "lobpcg-single";
    const bool jd = args.mode.rfind("jd-", 0) == 0;
    const bool lobpcg = args.mode.rfind("lobpcg", 0) == 0;
    if (!jd && !lobpcg && args.mode != "si-double" && args.mode != "si-single" && args.mode != "si-smallest")
      HS_THROW(InvalidInput, "unknown mode " << args.mode);
    hs::ShiftedPencil<T> pencil(a, b);
    hs::FactorOptions fo;
    fo.precision = single ? hs::FactorPrecision::Single : hs::FactorPrecision::Double;
    fo.ordering = args.ordering;
    fo.threads = hs::max_threads();
    fo.deterministic = args.deterministic;
    auto fac = hs::make_mumps_factorization<T>(fo);
    if (args.mode != "lobpcg") {  // the factorization-free path needs none
      fac->analyze(pencil, 1.0, -sigma);
      if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "A - sigma B singular at sigma = " << sigma);
    }
    const double t_factor = total.seconds();
    hs::Timer it;
    if (lobpcg) {
      FactorSolveOperator<T> prec(*fac);
      hs::CsrOperator<T> aop(a);
      std::unique_ptr<hs::CsrOperator<T>> bop;
      if (b) bop = std::make_unique<hs::CsrOperator<T>>(*b);
      hs::LobpcgOptions<T> o;
      o.nev = k;
      o.tol = args.tol;
      o.max_iterations = args.max_iterations;
      o.norm_a = na;
      o.norm_b = nb;
      const auto r = hs::lobpcg<T>(aop, bop.get(), single ? &prec : nullptr, hs::Which::SmallestAlgebraic, o);
      lambda = r.theta;
      x = r.vectors;
      out["converged"] = r.converged;
      out["iterations"] = static_cast<long long>(r.iterations);
      out["restarts"] = static_cast<long long>(r.restarts);
      out["op_applies"] = static_cast<long long>(r.op_applies);
      out["precond_applies"] = static_cast<long long>(r.precond_applies);
    } else if (jd) {
      FactorSolveOperator<T> prec(*fac);
      hs::CsrOperator<T> aop(a);
      std::unique_ptr<hs::CsrOperator<T>> bop;
      if (b) bop = std::make_unique<hs::CsrOperator<T>>(*b);
      hs::JdOptions<T> o;
      o.nev = k;
      o.tol = args.tol;
      o.inner_iterations = args.inner;
      o.norm_a = na;
      o.norm_b = nb;
      const auto r = hs::jacobi_davidson<T>(aop, bop.get(), &prec, sigma, o);
      lambda = r.theta;
      x = r.vectors;
      out["converged"] = r.converged;
      out["iterations"] = static_cast<long long>(r.iterations);
      out["inner_iterations"] = static_cast<long long>(r.inner_iterations);
      out["tracking_steps"] = static_cast<long long>(r.tracking_steps);
      out["fallback_expansions"] = static_cast<long long>(r.fallback_expansions);
      out["restarts"] = static_cast<long long>(r.restarts);
      out["op_applies"] = static_cast<long long>(r.op_applies);
      out["precond_applies"] = static_cast<long long>(r.precond_applies);
      if (args.history) {
        json h = json::array();
        for (const auto& rec : r.history)
          h.push_back({rec.restart, rec.nconv, rec.max_estimate, static_cast<long long>(rec.op_applies)});
        out["history"] = h;  // [iteration, locked, eta of the current approximation, A applications]
      }
    } else {
      hs::SolveRefinement refine;
      if (single) {
        refine.max_steps = 10;
        refine.rel_tol = args.tol * 0.1 * 0.1;  // the driver's choice (solver.cpp)
      }
      hs::ShiftInvertOperator<T> op(*fac, pencil, sigma, refine);
      std::unique_ptr<hs::CsrOperator<T>> bop;
      if (b) bop = std::make_unique<hs::CsrOperator<T>>(*b);
      hs::KrylovSchurOptions<T> ko;
      ko.nev = k;
      ko.tol = args.tol * 0.1;  // driver: tol * inner_tol_factor
      const auto r = hs::lanczos_krylov_schur<T>(op, bop.get(), hs::Which::LargestMagnitude, ko);
      for (double th : r.theta) lambda.push_back(sigma + 1.0 / th);
      x = r.vectors;
      out["converged"] = r.converged;
      out["restarts"] = static_cast<long long>(r.restarts);
      out["op_applies"] = static_cast<long long>(r.op_applies);
      out["refinement_steps"] = static_cast<long long>(op.refinement_steps());
    }
    const auto& st = fac->stats();
    out["time_analyze_s"] = st.analyze_s;
    out["time_factor_s"] = t_factor;
    out["time_iterate_s"] = it.seconds();
    out["time_solve_s"] = st.solve_s;
    out["factor_mb_used"] = st.factor_mb_used;
    out["factor_mb_estimate"] = st.estimate.incore_mb;
    out["factor_entries"] = static_cast<long long>(st.factor_entries);
    out["solves"] = static_cast<long long>(st.rhs_solved);
  }
  out["time_total_s"] = total.seconds();
  out["peak_rss_bytes"] = static_cast<long long>(hs::peak_rss_bytes());
  const auto eta = backward_errors(a, b, x, lambda, na, nb);
  double max_eta = 0.0;
  for (double e : eta) max_eta = std::max(max_eta, e);
  out["found"] = static_cast<long long>(lambda.size());
  out["max_backward_error"] = max_eta;
  out["all_accepted"] = max_eta <= args.tol && static_cast<hs::index_t>(lambda.size()) == k;
  std::vector<double> sorted = lambda;
  std::sort(sorted.begin(), sorted.end());
  out["eigenvalues"] = sorted;
  out["reference"] = compare_with_reference(ref_file, lambda, sigma, na / nb);
  out["sigma_used"] = sigma;
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  const char* root = std::getenv("HS_DATA_ROOT");
  a.data_root = root ? root : "data";
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) HS_THROW(InvalidInput, "missing value for " << key);
    const std::string val = argv[++i];
    if (key == "--case") a.case_name = val;
    else if (key == "--mode") a.mode = val;
    else if (key == "--k") a.k = std::stoll(val);
    else if (key == "--sigma") a.sigma = std::stod(val);
    else if (key == "--tol") a.tol = std::stod(val);
    else if (key == "--ordering") a.ordering = std::stoi(val);
    else if (key == "--inner") a.inner = std::stoll(val);
    else if (key == "--deterministic") a.deterministic = std::stoi(val) != 0;
    else if (key == "--history") a.history = std::stoi(val) != 0;
    else if (key == "--max-iterations") a.max_iterations = std::stoll(val);
    else if (key == "--data-root") a.data_root = val;
    else HS_THROW(InvalidInput, "unknown option " << key);
  }
  if (a.case_name.empty() || a.mode.empty()) HS_THROW(InvalidInput, "--case and --mode are required");
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    const fs::path dir = fs::path(args.data_root) / "cases" / args.case_name;
    std::ifstream cf(dir / "config.json");
    if (!cf) HS_THROW(Io, "cannot read " << (dir / "config.json").string());
    const json cfg = json::parse(cf);
    const double sigma = args.sigma ? *args.sigma : cfg.at("target").value("sigma", 0.0);
    const hs::index_t k = args.k > 0 ? args.k : cfg.at("target").value("k", 6);
    hs::Timer load;
    hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(dir / "B.npz")) lb = hs::load_matrix(dir / "B.npz");
    const double t_load = load.seconds();
    const fs::path ref = fs::path(args.data_root) / "refs" / args.case_name / "eigenvalues.npy";
    json out;
    const bool cx = la.is_complex() || (lb && lb->is_complex());
    if (cx) {
      hs::CsrMatrix<hs::cplx> a = la.is_complex() ? std::get<1>(std::move(la.matrix)) : hs::to_complex(std::get<0>(la.matrix));
      hs::CsrMatrix<hs::cplx> b;
      if (lb) b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(lb->matrix));
      out = run<hs::cplx>(args, a, lb ? &b : nullptr, sigma, k, ref);
    } else {
      const auto& a = std::get<0>(la.matrix);
      out = run<double>(args, a, lb ? &std::get<0>(lb->matrix) : nullptr, sigma, k, ref);
    }
    out["case"] = args.case_name;
    out["mode"] = args.mode;
    out["n"] = static_cast<long long>(la.order());
    out["k"] = static_cast<long long>(k);
    out["sigma"] = sigma;
    out["tol"] = args.tol;
    out["threads"] = hs::max_threads();
    out["ordering"] = args.ordering;
    out["time_load_s"] = t_load;
    std::cout << out.dump() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_lobpcg_jd_memory: %s\n", e.what());
    return 2;
  }
}
