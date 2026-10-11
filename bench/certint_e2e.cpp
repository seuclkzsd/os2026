// End-to-end measurement of the completeness-check modes (module certint, SolverConfig::certificate
// = exact | auto | randomized) on one data case: hs::solve with the case's config.json, the modes
// interleaved over the repetitions (the order rotating), plus the detection experiment
// (certificate_fault_drop: one copy of the largest found multiplet removed before the check).
//
//   bench_certint_e2e --case DIR --out DIR [--threads t] [--reps r] [--modes exact,auto,randomized]
//                     [--drop d] [--delta 1e-12] [--log DIR]
//
// Per run one JSON line ("exp": "run"): total and phase times, the certificate (method, rigorous,
// failure_probability, solves, seconds, passed, counts), factorizations and solves, success, the
// eigenvalues; per mode a "summary" line with the medians. The first repetition of each mode writes
// its result directory OUT/<case>/hsolve-<mode>[-drop<d>]/ (eigenvalues.npy, eigenvectors.npy,
// report.json; docs/FORMATS.md §3), the layout tools/eval/sweep.py evaluates (--log DIR: the
// driver's event log of every run as DIR/<case>-<mode>-rep<r>.jsonl):
//   python -m tools.eval.sweep OUT --out OUT/eval.md      (under tools/run/guard.py)
// Run through tools/run/guard.py (P5: gpu02, cores 24-27, 4 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 24-27 --mem-gb 14 --timeout 3000 --
//     build/release/bench_certint_e2e --case data/cases/<c> --out <run> --threads 4 --reps 3
// The process peak RSS covers all modes (the modes share one process).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  fs::path case_dir, out;
  int threads = 4;
  int reps = 3;
  std::vector<std::string> modes{"exact", "auto", "randomized"};
  index_t drop = 0;
  double delta = 1e-12;
  fs::path log_dir;
};

void emit(const json& j) { std::cout << j.dump() << std::endl; }

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const std::size_t h = v.size() / 2;
  return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p);
  if (!in) HS_THROW(Io, "cannot read " << p.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const std::string name = args.case_dir.filename().string();
  hs::SolverConfig base = hs::parse_config(read_text(args.case_dir / "config.json"));
  base.threads = args.threads;
  base.certificate_delta = args.delta;
  base.certificate_fault_drop = args.drop;
  std::map<std::string, std::vector<json>> records;
  const std::size_t nm = args.modes.size();
  for (int rep = 0; rep < args.reps; ++rep)
    for (std::size_t q = 0; q < nm; ++q) {
      const std::string mode = args.modes[(q + static_cast<std::size_t>(rep)) % nm];
      const hs::SolverConfig cfg = hs::parse_config(R"({"certificate": ")" + mode + R"("})", base);
      std::unique_ptr<hs::Logger> log;
      if (!args.log_dir.empty()) {
        fs::create_directories(args.log_dir);
        log = std::make_unique<hs::Logger>(
            0, args.log_dir / (name + "-" + mode + "-rep" + std::to_string(rep) + ".jsonl"));
      }
      hs::Timer wall;
      const hs::Solution<T> sol = hs::solve(a, b, cfg, log.get());
      const double wall_s = wall.seconds();
      const hs::SolveReport& r = sol.report;
      const auto& c = r.certificate;
      json j{{"exp", "run"},
             {"case", name},
             {"mode", mode},
             {"drop", args.drop},
             {"rep", rep},
             {"n", r.n},
             {"k", cfg.target.k},
             {"wall_s", wall_s},
             {"time_total_s", r.time_total_s},
             {"time_factor_s", r.time_factor_s},
             {"time_iterate_s", r.time_iterate_s},
             {"time_verify_s", r.time_verify_s},
             {"time_certificate_s", r.time_certificate_s},
             {"success", r.success},
             {"method", c.method},
             {"rigorous", c.rigorous},
             {"passed", c.passed},
             {"failure_probability", c.failure_probability},
             {"cert_solves", c.solves},
             {"cert_seconds", c.seconds},
             {"count_inertia", c.count_inertia},
             {"count_found", c.count_found},
             {"statement", c.statement},
             {"factorizations", r.factor_stats.factorizations},
             {"solves", r.factor_stats.solves},
             {"rhs_solved", r.factor_stats.rhs_solved},
             {"factorize_s", r.factor_stats.factorize_s},
             {"solve_s", r.factor_stats.solve_s},
             {"op_applies", r.op_applies},
             {"block_size", r.block_size},
             {"ordering", r.factor_options.ordering},
             {"warnings", r.warnings},
             {"eigenvalues", sol.eigenvalues},
             {"peak_rss_gib", static_cast<double>(r.peak_rss_bytes) / 1073741824.0}};
      emit(j);
      records[mode].push_back(j);
      if (rep == 0) {
        const fs::path dir =
            args.out / name / ("hsolve-" + mode + (args.drop > 0 ? "-drop" + std::to_string(args.drop) : ""));
        fs::create_directories(dir);
        hs::write_npy_vector(dir / "eigenvalues.npy", sol.eigenvalues);
        hs::write_npy_matrix(dir / "eigenvectors.npy", sol.eigenvectors);
        std::ofstream rep_out(dir / "report.json");
        rep_out << r.to_json(2) << '\n';
        if (!rep_out) HS_THROW(Io, "cannot write " << (dir / "report.json").string());
      }
    }
  for (const auto& mode : args.modes) {
    const auto& v = records[mode];
    std::vector<double> tot, cert, fac, it, wall;
    bool all_success = true;
    std::map<std::string, int> methods;
    for (const json& j : v) {
      tot.push_back(j["time_total_s"].get<double>());
      cert.push_back(j["time_certificate_s"].get<double>());
      fac.push_back(j["time_factor_s"].get<double>());
      it.push_back(j["time_iterate_s"].get<double>());
      wall.push_back(j["wall_s"].get<double>());
      all_success = all_success && j["success"].get<bool>();
      ++methods[j["method"].get<std::string>()];
    }
    emit({{"exp", "summary"},
          {"case", name},
          {"mode", mode},
          {"drop", args.drop},
          {"reps", v.size()},
          {"median_total_s", median(tot)},
          {"median_certificate_s", median(cert)},
          {"median_factor_s", median(fac)},
          {"median_iterate_s", median(it)},
          {"median_wall_s", median(wall)},
          {"all_success", all_success},
          {"methods", methods}});
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
    else if (k == "--out") args.out = need(i);
    else if (k == "--threads") args.threads = std::stoi(need(i));
    else if (k == "--reps") args.reps = std::stoi(need(i));
    else if (k == "--modes") args.modes = split(need(i));
    else if (k == "--drop") args.drop = std::stoll(need(i));
    else if (k == "--delta") args.delta = std::stod(need(i));
    else if (k == "--log") args.log_dir = need(i);
    else {
      std::cerr << "unknown argument " << k << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty() || args.out.empty() || args.reps < 1 || args.modes.empty()) {
    std::cerr << "usage: bench_certint_e2e --case DIR --out DIR [--threads t] [--reps r] "
                 "[--modes exact,auto,randomized] [--drop d] [--delta x] [--log DIR]\n";
    return 2;
  }
  omp_set_num_threads(args.threads);
  try {
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& mm) {
      return mm.is_complex() ? std::move(std::get<hs::CsrMatrix<hs::cplx>>(mm.matrix))
                             : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(mm.matrix)));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, a, b ? &*b : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
