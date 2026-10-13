// The B-definiteness test of hs::solve on a data case (module anaint, P5): the rigorous incomplete-
// Cholesky proof of driver::prove_positive_definite (src/driver/norms.hpp) against the numeric
// factorization of B it replaces - MUMPS on the pencil's analysis, as the driver counts B's inertia
// when no cheap test decides.
//
//   bench_anaint_bproof --case DIR [--threads 4] [--ordering 0] [--scan 1] [--reps 3]
//
// DIR holds A.npz and B.npz. Output: one JSON line - n, nnz_b, the proof (outcome, seconds,
// theta_min / theta_max of the scaled B, the attempt that decided: level, shift, lower bound,
// factor entries; median seconds over --reps runs, and of its parts: setup_s (the scaled B), lanczos_s,
// pattern_s (level-1 fill pattern), attempts_s), with --scan 1 every (level, shift fraction)
// attempt on a grid (lower bound, failing rows, breakdown, factorization / check seconds), and the
// MUMPS side: analysis of the pencil with FactorOptions::ordering --ordering (the driver's options
// otherwise), then factorize(0 A + 1 B) - median seconds of --reps factorizations - with B's
// inertia. The numbers quoted in src/driver/solver.cpp come from
//   python3 tools/run/guard.py --skip-busy-check --cores 20-23 --mem-gb 14 --timeout 900 --
//     build/release/bench_anaint_bproof --case data/cases/$c --threads 4 --ordering <driver's choice>
#include <omp.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  fs::path case_dir;
  int threads = 4;
  int ordering = 0;
  bool scan = false;
  int reps = 3;
};

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <class T>
int run(const Args& args, const hs::SolverConfig& cfg_in, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>& b) {
  omp_set_num_threads(args.threads);
  json out = {{"case", args.case_dir.filename().string()}, {"n", b.nrows},       {"nnz_b", b.nnz()},
              {"complex", hs::is_complex_v<T>},             {"threads", args.threads}, {"ordering", args.ordering}};
  // The proof, as the driver runs it.
  std::vector<double> secs, setup, lanczos, pattern, attempts;
  hs::driver::HpdProof proof;
  for (int r = 0; r < std::max(1, args.reps); ++r) {
    proof = hs::driver::prove_positive_definite(b);
    secs.push_back(proof.seconds);
    setup.push_back(proof.setup_s);
    lanczos.push_back(proof.lanczos_s);
    pattern.push_back(proof.pattern_s);
    attempts.push_back(proof.attempts_s);
  }
  out["proof"] = {{"outcome", proof.outcome},      {"proven", proof.proven},           {"seconds", median(secs)},
                  {"setup_s", median(setup)},      {"lanczos_s", median(lanczos)},     {"pattern_s", median(pattern)},
                  {"attempts_s", median(attempts)},
                  {"theta_min", proof.theta_min},  {"theta_max", proof.theta_max},     {"level", proof.level},
                  {"shift", proof.shift},          {"lower_bound", proof.lower_bound}, {"attempts", proof.attempts},
                  {"factor_entries", proof.factor_entries}};
  if (args.scan && proof.theta_min > 0) {
    json grid = json::array();
    for (const int level : {0, 1})
      for (const double frac : {0.9, 0.7, 0.5, 0.3, 0.15, 0.05}) {
        const auto ib = hs::driver::incomplete_cholesky_bound(b, level, frac * proof.theta_min);
        grid.push_back({{"level", level},
                        {"fraction", frac},
                        {"breakdown", ib.breakdown},
                        {"lower_bound", ib.lower_bound},
                        {"failing_rows", ib.failing_rows},
                        {"factor_entries", ib.factor_entries},
                        {"work_factor", ib.work.factor},
                        {"work_check", ib.work.check},
                        {"levels", ib.work.levels},
                        {"predicted_s", hs::driver::hpd_attempt_seconds(b.nrows, ib.work, args.threads)},
                        {"factor_s", ib.factor_s},
                        {"check_s", ib.check_s}});
      }
    out["scan"] = grid;
  }
  // MUMPS: B's inertia on the pencil's analysis.
  const hs::ShiftedPencil<T> pencil(a, &b);
  hs::SolverConfig cfg = cfg_in;
  cfg.threads = args.threads;
  cfg.ordering = args.ordering;
  const hs::FactorOptions fo = hs::driver::factor_options_from(cfg, 0.0);
  auto f = hs::make_mumps_factorization<T>(fo);
  hs::Timer t;
  f->analyze(pencil, 1.0, -cfg.target.sigma);
  out["mumps_analyze_s"] = t.seconds();
  std::vector<double> fs_;
  hs::Inertia in;
  for (int r = 0; r < std::max(1, args.reps); ++r) {
    t.reset();
    const bool ok = f->factorize(0.0, 1.0);
    fs_.push_back(t.seconds());
    in = ok ? f->inertia() : hs::Inertia{};
  }
  out["mumps_factorize_b_s"] = median(fs_);
  out["b_inertia"] = {{"valid", in.valid}, {"neg", in.neg}, {"zero", in.zero}, {"pos", in.pos}};
  std::cout << out.dump() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (k == "--case")
        args.case_dir = v;
      else if (k == "--threads")
        args.threads = std::stoi(v);
      else if (k == "--ordering")
        args.ordering = std::stoi(v);
      else if (k == "--scan")
        args.scan = std::stoi(v) != 0;
      else if (k == "--reps")
        args.reps = std::stoi(v);
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty() || !fs::exists(args.case_dir / "B.npz")) {
      std::cerr << "usage: bench_anaint_bproof --case DIR (with A.npz and B.npz) [...]\n";
      return 2;
    }
    std::ifstream cf(args.case_dir / "config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    const hs::SolverConfig cfg = hs::parse_config(ss.str());
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    auto lb = hs::load_matrix(args.case_dir / "B.npz");
    if (!la.is_complex() && !lb.is_complex())
      return run<double>(args, cfg, std::get<hs::CsrMatrix<double>>(la.matrix),
                         std::get<hs::CsrMatrix<double>>(lb.matrix));
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
    };
    const auto a = to_c(la);
    const auto b = to_c(lb);
    return run<hs::cplx>(args, cfg, a, b);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
