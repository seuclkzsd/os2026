// End-to-end GHIEP solves (hs::solve) for a list of Krylov block sizes p (module ghiep, P4 fixes):
// the measurement behind the driver's block-size rule (indefinite.cpp, ghiep_block_size), with the
// target shift and k overridable so that the one-sided zero-cluster path (sigma = 0) of a data case
// runs without editing its config.json.
//
// Per case: load A and B (real; data/cases/<case>/{A,B}.npz) and config.json; per repetition (outer
// loop, so slow load drifts of the shared machine affect every p alike) and per p, hs::solve with
// the configuration, threads from the environment (the guard sets OMP_NUM_THREADS), sigma / k
// replaced when given, and block size p via SolverConfig::block_size (p = 0 leaves the driver's
// automatic rule). One JSON line per solve: phase times, operator applications, solves,
// factorizations, success, largest backward error, real / complex counts and the largest relative
// eigenvalue difference to the first p of the list.
//
// p > 0 needs SolverConfig::block_size (interface request 2 of module ghiep: `index_t block_size =
// 0;`, 0 = auto, 1 = single vector). Without it the program exits with an error: build it on a
// tree that has the field (the driver's accessor configured_block_size picks it up unchanged).
//
// Usage (via the resource guard, 2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 28,29 --mem-gb 4 --timeout 3600 --
//     build/release/bench_ghiep_fix_e2e [--data DIR] [--p 1,2,4,8] [--reps 3] [--sigma S] [--k K]
//     CASE...
// DIR defaults to $HS_DATA_ROOT or ./data.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;

struct Args {
  std::filesystem::path data;
  std::vector<index_t> p = {1, 2, 4, 8};
  int reps = 3;
  std::optional<double> sigma;
  std::optional<index_t> k;
  std::vector<std::string> cases;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

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
    if (s == "--data")
      a.data = value();
    else if (s == "--p")
      a.p = parse_list(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--sigma")
      a.sigma = std::stod(value());
    else if (s == "--k")
      a.k = std::stoll(value());
    else
      a.cases.push_back(s);
  }
  if (a.cases.empty()) throw std::runtime_error("no case given");
  return a;
}

// SolverConfig::block_size when the field exists (interface request 2 of module ghiep).
template <class C>
auto set_block_size(C& cfg, index_t p, int) -> decltype(cfg.block_size = p, true) {
  cfg.block_size = p;
  return true;
}
template <class C>
bool set_block_size(C&, index_t, long) {
  return false;
}

std::string read_text(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

double max_rel_diff(std::vector<double> got, std::vector<double> ref) {
  if (got.size() != ref.size()) return std::numeric_limits<double>::infinity();
  std::sort(got.begin(), got.end());
  std::sort(ref.begin(), ref.end());
  double d = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i)
    d = std::max(d, std::abs(got[i] - ref[i]) / std::max(std::abs(ref[i]), 1e-300));
  return d;
}

void run_case(const Args& args, const std::string& name) {
  const auto dir = args.data / "cases" / name;
  hs::SolverConfig base = hs::parse_config(read_text(dir / "config.json"));
  if (base.target.mode != hs::TargetMode::Nearest) throw std::runtime_error(name + ": target is not 'nearest'");
  if (args.sigma) base.target.sigma = *args.sigma;
  if (args.k) base.target.k = *args.k;
  const hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
  const hs::LoadedMatrix lb = hs::load_matrix(dir / "B.npz");
  if (la.is_complex() || lb.is_complex()) throw std::runtime_error(name + ": complex cases are not supported");
  const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
  const auto& b = std::get<hs::CsrMatrix<double>>(lb.matrix);
  std::vector<double> ref;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (const index_t p : args.p) {
      hs::SolverConfig cfg = base;
      if (p != 0 && !set_block_size(cfg, p, 0))
        throw std::runtime_error("this build has no SolverConfig::block_size (interface request 2): p > 0 needs it");
      const hs::Solution<double> sol = hs::solve(a, &b, cfg);
      const hs::SolveReport& r = sol.report;
      double worst_eta = 0.0;
      for (const auto& pr : r.pairs) worst_eta = std::max(worst_eta, pr.backward_error);
      if (ref.empty()) ref = sol.eigenvalues;
      std::cout << json{{"record", "e2e"},
                        {"case", name},
                        {"sigma", cfg.target.sigma},
                        {"k", cfg.target.k},
                        {"p", p},
                        {"rep", rep},
                        {"threads", hs::max_threads()},
                        {"success", r.success},
                        {"total_s", r.time_total_s},
                        {"setup_s", r.time_setup_s},
                        {"factor_s", r.time_factor_s},
                        {"iterate_s", r.time_iterate_s},
                        {"verify_s", r.time_verify_s},
                        {"op_applies", r.op_applies},
                        {"restarts", r.restarts},
                        {"solve_calls", r.factor_stats.solves},
                        {"solve_s", r.factor_stats.solve_s},
                        {"factorizations", r.factor_stats.factorizations},
                        {"runs", r.slices.size()},
                        {"real", sol.eigenvalues.size()},
                        {"complex", sol.complex_eigenvalues.size()},
                        {"max_eta", worst_eta},
                        {"max_rel_diff_to_first_p", max_rel_diff(sol.eigenvalues, ref)},
                        {"warnings", r.warnings},
                        {"lambda", sol.eigenvalues},
                        {"peak_rss_gib", static_cast<double>(hs::peak_rss_bytes()) / (1u << 30)}}
                       .dump()
                << std::endl;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    for (const std::string& name : args.cases) run_case(args, name);
  } catch (const std::exception& e) {
    std::cerr << "bench_ghiep_fix_e2e: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
