// Measurements behind the P4 driver integration (module drvint): where a whole HEP/GHEP run spends
// its time outside the Krylov iteration, and the inputs of the driver's run-time models.
//
//   bench_drvint_profile --case DIR [--threads t] [--reps r] [--ordering o] [--lower a --upper b]
//                        --exp E[,E...]
//
// DIR holds A.npz (and B.npz) and config.json (target sigma). Experiments (one JSON line each):
//   setup     the steps hs::solve takes before its first analysis, each timed separately: loading
//             A (and B), hermitian_defect, estimate_norm2 (30 Lanczos steps), the ShiftedPencil,
//             the diagonal / diagonally-dominant test of B
//   analysis  AMD (ICNTL(7)=0) and SCOTCH (ICNTL(7)=3) analyses of K(sigma), interleaved; the MUMPS
//             input size (order, upper entries: driver::mumps_input_size) and the predictions of
//             driver::model_analysis_s - the data of its fit (steps.cpp)
//   counter   certificate counts at [lower, upper] (default sigma -+ 10% of |sigma| + 1e-3 scale):
//             refactorizing the main object (the old driver path) vs an InertiaCounter built from
//             it (analysis + two counts), interleaved, with the main factors at sigma resident;
//             also the driver::counter_pays decision
//   solves    one MUMPS solve with p = 1, 2, 4, 8, 16 right-hand sides on the factorization at
//             sigma: seconds per right-hand side (the block Krylov trade-off)
//   output    writing an n x 12 eigenvector block with hs::write_npy_matrix
// --ordering: of the main factorization (counter, solves); -1 = the driver's adaptive choice.
// Run through tools/run/guard.py, e.g. (one command line)
//   python3 tools/run/guard.py --skip-busy-check --cores 24,25,26,27 --mem-gb 6 --
//     build/release/bench_drvint_profile --case data/cases/syn-mem27-46 --threads 4 --reps 3
//     --exp setup,analysis,counter,solves,output
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Args {
  fs::path case_dir;
  std::vector<std::string> exps;
  double lower = kNaN, upper = kNaN;
  int ordering = -1;
  int threads = 4;
  int reps = 3;
};

void emit(json j) { std::cout << j.dump() << std::endl; }

double median(std::vector<double> v) {
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

hs::FactorOptions options_for(int ordering, int threads) {
  hs::FactorOptions o;
  o.ordering = ordering;
  o.threads = threads;
  o.deterministic = false;  // the driver default (SolverConfig::deterministic = false)
  return o;
}

template <class T>
bool diagonally_dominant(const hs::CsrMatrix<T>& b) {
  for (index_t i = 0; i < b.nrows; ++i) {
    double diag = 0.0, off = 0.0;
    for (index_t p = b.rowptr[i]; p < b.rowptr[i + 1]; ++p) {
      if (b.colind[p] == i) {
        diag = std::real(b.values[p]);
      } else {
        off += std::abs(b.values[p]);
      }
    }
    if (!(diag > off)) return false;
  }
  return true;
}

// The driver's adaptive ordering (driver::decide_ordering on an AMD probe) for a nearest target
// with certificate: 3 (+1 with B) factorizations, max(40, 3k) solves (solver.cpp).
template <class T>
int adaptive_ordering(const hs::ShiftedPencil<T>& pencil, double sigma, int threads, index_t k, bool has_b) {
  auto f = hs::make_mumps_factorization<T>(options_for(0, threads));
  f->analyze(pencil, 1.0, -sigma);
  const auto d = hs::driver::decide_ordering(f->estimate(), hs::driver::mumps_input_size(pencil),
                                             static_cast<double>(pencil.nnz()), threads, 3 + (has_b ? 1 : 0),
                                             std::max<index_t>(40, 3 * k));
  return d.use_nd ? 3 : 0;
}

template <class T>
void exp_setup(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  std::vector<double> t_def_a, t_def_b, t_norm_a, t_norm_b, t_pencil, t_dd;
  bool dd = false;
  for (int r = 0; r < args.reps; ++r) {
    hs::Timer t;
    (void)hs::hermitian_defect(a);
    t_def_a.push_back(t.seconds());
    t = hs::Timer();
    (void)hs::driver::estimate_norm2(a, 30, 2026);
    t_norm_a.push_back(t.seconds());
    if (b) {
      t = hs::Timer();
      (void)hs::hermitian_defect(*b);
      t_def_b.push_back(t.seconds());
      t = hs::Timer();
      (void)hs::driver::estimate_norm2(*b, 30, 2026);
      t_norm_b.push_back(t.seconds());
      t = hs::Timer();
      dd = diagonally_dominant(*b);
      t_dd.push_back(t.seconds());
    }
    t = hs::Timer();
    hs::ShiftedPencil<T> pencil(a, b);
    t_pencil.push_back(t.seconds());
  }
  emit({{"exp", "setup"}, {"hermitian_defect_a_s", median(t_def_a)}, {"hermitian_defect_b_s", median(t_def_b)},
        {"norm2_a_s", median(t_norm_a)}, {"norm2_b_s", median(t_norm_b)}, {"pencil_s", median(t_pencil)},
        {"b_dominance_s", median(t_dd)}, {"b_diagonally_dominant", dd}});
}

template <class T>
void exp_analysis(const Args& args, const hs::ShiftedPencil<T>& pencil, double sigma) {
  std::vector<double> t_amd, t_nd;
  hs::FactorEstimate e_amd, e_nd;
  for (int r = 0; r < args.reps; ++r) {
    for (const int ord : {0, 3}) {
      auto f = hs::make_mumps_factorization<T>(options_for(ord, args.threads));
      hs::Timer t;
      f->analyze(pencil, 1.0, -sigma);
      (ord == 0 ? t_amd : t_nd).push_back(t.seconds());
      (ord == 0 ? e_amd : e_nd) = f->estimate();
    }
  }
  const auto in = hs::driver::mumps_input_size(pencil);
  emit({{"exp", "analysis"}, {"order", in.order}, {"entries", in.entries}, {"amd_s", median(t_amd)},
        {"scotch_s", median(t_nd)}, {"amd_s_all", t_amd}, {"scotch_s_all", t_nd},
        {"model_amd_s", hs::driver::model_analysis_s(in.order, in.entries, static_cast<double>(e_amd.factor_entries))},
        {"amd_flops", e_amd.flops}, {"amd_entries", e_amd.factor_entries}, {"scotch_flops", e_nd.flops},
        {"scotch_entries", e_nd.factor_entries},
        {"model_factor_amd_s", hs::driver::model_factor_s(e_amd, in.order, args.threads)},
        {"model_factor_scotch_s", hs::driver::model_factor_s(e_nd, in.order, args.threads)}});
}

template <class T>
void exp_counter(const Args& args, const hs::ShiftedPencil<T>& pencil, double sigma, double lower, double upper,
                 int ordering) {
  auto fac = hs::make_mumps_factorization<T>(options_for(ordering, args.threads));
  fac->analyze(pencil, 1.0, -sigma);
  std::vector<double> t_main_pair, t_counter_an, t_counter_pair, t_main_fact;
  hs::index_t nu_main_a = -1, nu_main_b = -1, nu_c_a = -1, nu_c_b = -1;
  double counter_mb = 0.0, main_mb = 0.0;
  for (int r = 0; r < args.reps; ++r) {
    hs::Timer t;
    if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "K(sigma) singular");
    t_main_fact.push_back(t.seconds());
    main_mb = fac->stats().factor_mb_used;
    {  // the counter, with the main factors resident
      hs::Timer tc;
      hs::InertiaCounter<T> counter(*fac, fac->options());
      t_counter_an.push_back(tc.seconds());
      counter_mb = counter.estimate().incore_mb;
      hs::Timer tp;
      const auto ia = counter.count(1.0, -lower);
      const auto ib = counter.count(1.0, -upper);
      t_counter_pair.push_back(tp.seconds());
      nu_c_a = ia.valid ? ia.neg : -1;
      nu_c_b = ib.valid ? ib.neg : -1;
    }
    hs::Timer tm;  // the old driver path: refactorize the main object at both ends
    nu_main_a = fac->factorize(1.0, -lower) ? fac->inertia().neg : -1;
    nu_main_b = fac->factorize(1.0, -upper) ? fac->inertia().neg : -1;
    t_main_pair.push_back(tm.seconds());
  }
  const double an = median(t_counter_an), cp = median(t_counter_pair), mp = median(t_main_pair);
  const auto in = hs::driver::mumps_input_size(pencil);
  const double model_saving =
      2 * hs::driver::kCounterSaving * hs::driver::model_factor_s(fac->estimate(), in.order, args.threads);
  const double model_an =
      hs::driver::model_analysis_s(in.order, in.entries, static_cast<double>(fac->estimate().factor_entries));
  emit({{"exp", "counter"}, {"ordering", ordering}, {"lower", lower}, {"upper", upper},
        {"main_factorize_s", median(t_main_fact)}, {"main_pair_s", mp}, {"counter_analyze_s", an},
        {"counter_pair_s", cp}, {"count_saving", 1.0 - cp / mp}, {"net_saving_s", mp - cp - an},
        {"main_counts", {nu_main_a, nu_main_b}}, {"counter_counts", {nu_c_a, nu_c_b}},
        {"main_mb", main_mb}, {"counter_mb", counter_mb},
        {"model_saving_s", model_saving},
        {"model_analysis_s", model_an}, {"rule_uses_counter", model_saving > model_an}});
}

template <class T>
void exp_solves(const Args& args, const hs::ShiftedPencil<T>& pencil, double sigma, int ordering) {
  auto fac = hs::make_mumps_factorization<T>(options_for(ordering, args.threads));
  fac->analyze(pencil, 1.0, -sigma);
  if (!fac->factorize(1.0, -sigma)) HS_THROW(Singular, "K(sigma) singular");
  const index_t n = pencil.order();
  json per_rhs = json::object();
  const std::vector<index_t> ps{1, 2, 4, 8, 16};
  std::vector<std::vector<double>> t(ps.size());
  hs::DenseMatrix<T> x(n, 16);
  for (int r = 0; r < args.reps + 1; ++r) {
    for (std::size_t q = 0; q < ps.size(); ++q) {
      hs::linalg::fill_random(x.data(), n * ps[q], 7 + static_cast<std::uint64_t>(r), 0);
      hs::Timer tm;
      fac->solve(x.data(), ps[q], n);
      if (r > 0) t[q].push_back(tm.seconds() / static_cast<double>(ps[q]));  // r = 0: warm-up
    }
  }
  for (std::size_t q = 0; q < ps.size(); ++q) per_rhs[std::to_string(ps[q])] = median(t[q]);
  emit({{"exp", "solves"}, {"ordering", ordering}, {"factor_entries", fac->stats().factor_entries},
        {"per_rhs_s", per_rhs}, {"model_solve_s", hs::driver::model_solve_s(fac->estimate())}});
}

template <class T>
void exp_output(const Args& args, index_t n) {
  hs::DenseMatrix<T> x(n, 12);
  hs::linalg::fill_random(x.data(), n * 12, 3, 0);
  const fs::path tmp = fs::temp_directory_path() / ("drvint_profile_" + std::to_string(::getpid()) + ".npy");
  std::vector<double> t;
  for (int r = 0; r < args.reps; ++r) {
    hs::Timer tm;
    hs::write_npy_matrix(tmp, x);
    t.push_back(tm.seconds());
  }
  fs::remove(tmp);
  emit({{"exp", "output"}, {"columns", 12}, {"write_s", median(t)}});
}

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, double sigma, index_t k,
        double load_a_s, double load_b_s) {
  emit({{"case", args.case_dir.filename().string()}, {"n", a.nrows}, {"nnz_a", a.nnz()},
        {"nnz_b", b ? b->nnz() : 0}, {"complex", hs::is_complex_v<T>}, {"threads", args.threads},
        {"sigma", sigma}, {"load_a_s", load_a_s}, {"load_b_s", load_b_s}});
  hs::ShiftedPencil<T> pencil(a, b);
  const double scale = std::max(1e-300, hs::driver::estimate_norm2(a, 30, 2026).norm2 /
                                            (b ? hs::driver::estimate_norm2(*b, 30, 2026).norm2 : 1.0));
  const double lower = std::isnan(args.lower) ? sigma - 0.1 * std::abs(sigma) - 1e-3 * scale : args.lower;
  const double upper = std::isnan(args.upper) ? sigma + 0.1 * std::abs(sigma) + 1e-3 * scale : args.upper;
  int ordering = args.ordering;
  for (const auto& e : args.exps) {
    if ((e == "counter" || e == "solves") && ordering < 0) ordering = adaptive_ordering(pencil, sigma, args.threads, k, b != nullptr);
    if (e == "setup") exp_setup(args, a, b);
    else if (e == "analysis") exp_analysis(args, pencil, sigma);
    else if (e == "counter") exp_counter(args, pencil, sigma, lower, upper, ordering);
    else if (e == "solves") exp_solves(args, pencil, sigma, ordering);
    else if (e == "output") exp_output<T>(args, a.nrows);
    else HS_THROW(InvalidInput, "unknown experiment " << e);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](int& j) -> std::string {
      if (j + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++j];
    };
    if (k == "--case") args.case_dir = need(i);
    else if (k == "--exp") {
      std::stringstream ss(need(i));
      std::string e;
      while (std::getline(ss, e, ',')) args.exps.push_back(e);
    }
    else if (k == "--lower") args.lower = std::stod(need(i));
    else if (k == "--upper") args.upper = std::stod(need(i));
    else if (k == "--ordering") args.ordering = std::stoi(need(i));
    else if (k == "--threads") args.threads = std::stoi(need(i));
    else if (k == "--reps") args.reps = std::stoi(need(i));
    else {
      std::cerr << "unknown argument " << k << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty() || args.exps.empty()) {
    std::cerr << "usage: bench_drvint_profile --case DIR --exp E[,E...] [--threads t] [--reps r]\n";
    return 2;
  }
  try {
    std::ifstream f(args.case_dir / "config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const hs::SolverConfig cfg = hs::parse_config(ss.str());
    const double sigma = cfg.target.mode == hs::TargetMode::Nearest ? cfg.target.sigma : 0.0;
    hs::Timer t;
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    const double load_a = t.seconds();
    std::optional<hs::LoadedMatrix> lb;
    t = hs::Timer();
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const double load_b = lb ? t.seconds() : 0.0;
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b, sigma, cfg.target.k, load_a, load_b);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, a, b ? &*b : nullptr, sigma, cfg.target.k, load_a, load_b);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
