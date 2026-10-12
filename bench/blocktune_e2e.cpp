// End-to-end hs::solve per Krylov block size (module eigen, P5b "blocktune").
//
// For each case: load A (and B) with hs::load_matrix, parse its config.json (hs::parse_config,
// the CLI's path), set threads = hs::max_threads() and the memory budget (--budget-gb), then run
// hs::solve once per block size of --p (SolverConfig::block_size; 0 = the driver's auto rule),
// --reps times, interleaved (repetitions outside, the order of the block sizes rotating by one
// per repetition). One "run" record per solve: the resolved block size, success, certificate
// (performed / passed / inertia count), the largest backward error of the returned pairs, operator
// applications, restarts, the report's time split (total / setup / factor / iterate / verify /
// certificate), the MUMPS solve time and right-hand sides, and the largest difference of the
// sorted eigenvalues to the first run of the case, |lambda - mu| / (|mu| + scale) (tools/eval's
// yardstick). The first record of a case ("case") holds the factorization shape the block-size
// rule reads (n, factor entries, nnz(B), threads, scalar type).
// This is the CLI's computation after loading (apps/hsolve_cli.cpp calls hs::solve the same way);
// output files and tools/eval are left out.
//
// Usage (via the resource guard):
//   python3 tools/run/guard.py --skip-busy-check --cores 24-31 --mem-gb 16 --timeout 3600 --
//     build/release/bench_blocktune_e2e [--data DIR] [--p 1,0,4,8] [--reps 5] [--budget-gb 14]
//     CASE...
// DIR defaults to $HS_DATA_ROOT or ./data; CASE is a directory name under DIR/cases.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using json = nlohmann::json;

struct Args {
  std::filesystem::path data;
  std::vector<index_t> p = {1, 0};
  int reps = 5;
  double budget_gb = 14.0;
  std::vector<std::string> cases;
};

Args parse_args(int argc, char** argv) {
  Args a;
  const char* root = std::getenv("HS_DATA_ROOT");
  a.data = root ? std::filesystem::path(root) : std::filesystem::path("data");
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--data") {
      a.data = value();
    } else if (s == "--p") {
      a.p.clear();
      std::stringstream ss(value());
      std::string item;
      while (std::getline(ss, item, ',')) a.p.push_back(std::stoll(item));
    } else if (s == "--reps") {
      a.reps = std::stoi(value());
    } else if (s == "--budget-gb") {
      a.budget_gb = std::stod(value());
    } else {
      a.cases.push_back(s);
    }
  }
  if (a.cases.empty()) throw std::runtime_error("no case given");
  return a;
}

template <class T>
CsrMatrix<T> as_scalar(hs::LoadedMatrix&& m) {
  if constexpr (std::is_same_v<T, cplx>) {
    if (!m.is_complex()) return hs::to_complex(std::get<CsrMatrix<double>>(m.matrix));
    return std::move(std::get<CsrMatrix<cplx>>(m.matrix));
  } else {
    return std::move(std::get<CsrMatrix<double>>(m.matrix));
  }
}

template <class T>
void run_case(const Args& args, const std::string& name, const hs::SolverConfig& base,
              hs::LoadedMatrix&& la, std::optional<hs::LoadedMatrix>&& lb) {
  const CsrMatrix<T> a = as_scalar<T>(std::move(la));
  std::optional<CsrMatrix<T>> b_store;
  if (lb) b_store = as_scalar<T>(std::move(*lb));
  const CsrMatrix<T>* b = b_store ? &*b_store : nullptr;
  std::vector<double> ref;
  bool head = false;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (std::size_t i = 0; i < args.p.size(); ++i) {
      const index_t p = args.p[(i + static_cast<std::size_t>(rep)) % args.p.size()];
      hs::SolverConfig cfg = base;
      cfg.block_size = p;
      hs::Timer wall;
      const hs::Solution<T> sol = hs::solve<T>(a, b, cfg);
      const double wall_s = wall.seconds();
      const hs::SolveReport& r = sol.report;
      if (!head) {
        json h = {{"record", "case"},
                  {"case", name},
                  {"n", r.n},
                  {"nnz_a", r.nnz_a},
                  {"nnz_b", r.nnz_b},
                  {"complex", r.complex_scalars},
                  {"k", cfg.target.k},
                  {"threads", cfg.threads},
                  {"factor_entries", r.factor_stats.factor_entries},
                  {"estimate_entries", r.factor_stats.estimate.factor_entries},
                  {"estimate_flops", r.factor_stats.estimate.flops},
                  {"ordering", r.factor_options.ordering},
                  {"scale", r.scale}};
        std::cout << h.dump() << std::endl;
        head = true;
      }
      double max_eta = 0.0;
      for (const auto& pr : r.pairs) max_eta = std::max(max_eta, pr.backward_error);
      std::vector<double> sorted = sol.eigenvalues;
      std::sort(sorted.begin(), sorted.end());
      if (ref.empty()) ref = sorted;
      double diff = sorted.size() == ref.size() ? 0.0 : INFINITY;
      for (std::size_t j = 0; j < sorted.size() && j < ref.size(); ++j)
        diff = std::max(diff, std::abs(sorted[j] - ref[j]) / (std::abs(ref[j]) + r.scale));
      json rec = {{"record", "run"},
                  {"case", name},
                  {"p_config", p},
                  {"block_size", r.block_size},
                  {"rep", rep},
                  {"success", r.success},
                  {"pairs", static_cast<index_t>(sol.eigenvalues.size())},
                  {"cert_performed", r.certificate.performed},
                  {"cert_passed", r.certificate.passed},
                  {"cert_count", r.certificate.count_inertia},
                  {"max_eta", max_eta},
                  {"op_applies", r.op_applies},
                  {"restarts", r.restarts},
                  {"time_total_s", r.time_total_s},
                  {"time_setup_s", r.time_setup_s},
                  {"time_factor_s", r.time_factor_s},
                  {"time_iterate_s", r.time_iterate_s},
                  {"time_verify_s", r.time_verify_s},
                  {"time_certificate_s", r.time_certificate_s},
                  {"solve_s", r.factor_stats.solve_s},
                  {"rhs_solved", r.factor_stats.rhs_solved},
                  {"factorizations", r.factor_stats.factorizations},
                  {"wall_s", wall_s},
                  {"set_diff_to_first", diff},
                  {"warnings", r.warnings},
                  {"peak_rss_gib", static_cast<double>(r.peak_rss_bytes) / (1u << 30)}};
      std::cout << rec.dump() << std::endl;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    for (const std::string& name : args.cases) {
      const auto dir = args.data / "cases" / name;
      std::ifstream in(dir / "config.json");
      std::ostringstream text;
      text << in.rdbuf();
      hs::SolverConfig cfg = hs::parse_config(text.str());
      cfg.threads = hs::max_threads();
      cfg.memory_budget_gb = args.budget_gb;
      hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
      std::optional<hs::LoadedMatrix> lb;
      if (std::filesystem::exists(dir / "B.npz")) lb = hs::load_matrix(dir / "B.npz");
      const bool complex = la.is_complex() || (lb && lb->is_complex());
      if (complex) {
        run_case<cplx>(args, name, cfg, std::move(la), std::move(lb));
      } else {
        run_case<double>(args, name, cfg, std::move(la), std::move(lb));
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "bench_blocktune_e2e: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
