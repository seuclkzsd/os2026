// hsolve command-line front end: read matrices (+ config), solve, write results (docs/FORMATS.md).
//
//   hsolve --A A.npz [--B B.npz] [--config config.json] --out DIR [overrides...] [--gpu MODE]
//
// --gpu off|auto|on (or HS_GPU): GPU offload of the factorizations' large dense updates by the BLAS
// hook linked into this executable (include/hsolve/gpu.hpp, docs/BLAS.md §10); default off. When
// the mode is not off, report.json gets a "gpu" object (device, what was offloaded, or why the CPU
// path ran - also when this hsolve was built without the GPU module or without the hook).
//
// Exit status: 0 = all pairs accepted (and certificate passed), 1 = finished but not successful,
// 2 = error (message on stderr).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/gpu.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& p) {
  std::ifstream in(p);
  if (!in) HS_THROW(Io, "cannot read " << p.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

// The GPU offload's report when its mode (or the --gpu value) is not off (empty otherwise): the
// hook's, or - when the hook cannot offload - the mode asked for and why the CPU path ran.
// `asked`: the --gpu value ("" when not given).
nlohmann::json gpu_report(const std::string& asked) {
  const bool wanted = !asked.empty() && asked != "off";
  const std::string text = hs::gpu_offload_report_json();
  if (text.empty()) {
    if (!wanted) return {};
    return {{"mode", asked}, {"available", false}, {"reason", "no BLAS hook in this executable"}};
  }
  nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded()) return {};
  if (j.value("mode", "off") != "off") return j;
  if (!wanted) return {};
  // A hook that ignored the request (built without the GPU module): say so in the report.
  return {{"mode", asked},
          {"available", false},
          {"reason", j.value("reason", std::string("the BLAS hook did not take the GPU mode"))}};
}

template <class T>
void write_outputs(const fs::path& out, const hs::Solution<T>& sol, const nlohmann::json& gpu) {
  fs::create_directories(out);
  hs::write_npy_vector(out / "eigenvalues.npy", sol.eigenvalues);
  hs::write_npy_matrix(out / "eigenvectors.npy", sol.eigenvectors);
  const fs::path cev = out / "complex_eigenvalues.npy", cvec = out / "complex_eigenvectors.npy";
  if (!sol.complex_eigenvalues.empty()) {
    hs::write_npy_vector(cev, sol.complex_eigenvalues);
    hs::write_npy_matrix(cvec, sol.complex_eigenvectors);
  } else {
    fs::remove(cev);
    fs::remove(cvec);
  }
  std::ofstream rep(out / "report.json");
  if (gpu.is_null()) {
    rep << sol.report.to_json(2) << '\n';
  } else {
    nlohmann::json r = nlohmann::json::parse(sol.report.to_json(2));
    r["gpu"] = gpu;
    rep << r.dump(2) << '\n';
  }
  if (!rep) HS_THROW(Io, "cannot write " << (out / "report.json").string());
}

void print_gpu(const nlohmann::json& g) {
  if (g.is_null()) return;
  if (g.value("factorizations_offloaded", 0) > 0)
    std::printf("gpu (%s): %s, %d of %d factorizations offloaded, %.0f GFlop on the GPU (%.0f%% of their "
                "estimate), set-up %.2fs\n",
                g.value("mode", "").c_str(), g.value("device", "").c_str(), g.value("factorizations_offloaded", 0),
                g.value("factorizations", 0), g.value("gpu_gflop", 0.0), 100.0 * g.value("gpu_share", 0.0),
                g.value("init_s", 0.0));
  else if (g.contains("reason"))
    std::printf("gpu (%s): CPU path - %s\n", g.value("mode", "").c_str(), g.value("reason", "").c_str());
  else
    std::printf("gpu (%s): CPU path - no factorization large enough\n", g.value("mode", "").c_str());
}

template <class T>
void print_summary(const hs::Solution<T>& sol) {
  const auto& r = sol.report;
  std::printf("n=%lld  problem=%s  time=%.3fs  peak_rss=%.2f GiB  success=%s\n", static_cast<long long>(r.n),
              r.problem == hs::ProblemType::Standard               ? "standard"
              : r.problem == hs::ProblemType::GeneralizedDefinite ? "generalized_definite"
                                                                   : "generalized_indefinite",
              r.time_total_s, static_cast<double>(r.peak_rss_bytes) / 1073741824.0, r.success ? "yes" : "no");
  std::printf("%5s %24s %24s %12s %4s\n", "#", "Re(lambda)", "Im(lambda)", "backward_err", "ok");
  for (std::size_t i = 0; i < r.pairs.size(); ++i) {
    const auto& p = r.pairs[i];
    std::printf("%5zu %24.16e %24.16e %12.3e %4s\n", i, p.lambda_re, p.lambda_im, p.backward_error,
                p.accepted ? "yes" : "NO");
  }
  const auto& c = r.certificate;
  if (c.performed && c.note.rfind("not completed", 0) == 0)  // no count possible: pairs uncertified
    std::printf("certificate (%s): [%g, %g] found=%lld -> NOT COMPLETED (%s)\n", c.method.c_str(), c.lower, c.upper,
                static_cast<long long>(c.count_found), c.note.c_str());
  else if (c.performed && c.count_inertia >= 0)
    std::printf("certificate (%s): [%g, %g] inertia=%lld found=%lld -> %s\n", c.method.c_str(), c.lower, c.upper,
                static_cast<long long>(c.count_inertia), static_cast<long long>(c.count_found),
                c.passed ? (c.rigorous ? "passed" : "passed (screening only)") : "FAILED");
  else if (c.performed)  // randomized check: no count, a failure probability
    std::printf("certificate (%s): [%g, %g] found=%lld -> %s, failure probability <= %.3g%s\n", c.method.c_str(),
                c.lower, c.upper, static_cast<long long>(c.count_found), c.passed ? "passed" : "FAILED",
                c.failure_probability, c.rigorous ? "" : " (screening only: kappa of B assumed)");
  for (const auto& w : r.warnings) std::printf("warning: %s\n", w.c_str());
}

template <class T>
int run(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg, hs::Logger& log,
        const fs::path& out, const std::vector<std::string>& load_notes, bool quiet, const std::string& gpu_asked) {
  hs::Solution<T> sol = hs::solve(a, b, cfg, &log);
  for (const auto& note : load_notes) sol.report.warnings.push_back("input: " + note);
  const nlohmann::json gpu = gpu_report(gpu_asked);
  write_outputs(out, sol, gpu);
  if (!quiet) {
    print_summary(sol);
    print_gpu(gpu);
  }
  return sol.report.success ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"hsolve - eigensolver for large sparse Hermitian (indefinite) eigenproblems"};
  std::string a_path, b_path, config_path, out_dir = "hsolve_out", log_path;
  double sigma = 0.0, tol = 0.0, zero_filter = 0.0;
  long long k = 0, smallest = 0, largest = 0, slice_size = 0, block_size = 0;
  std::vector<double> interval;
  int threads = -1;
  std::string precision, certificate, budget, gpu_mode;
  bool verbose = false, quiet = false, version = false;

  app.add_option("--A", a_path, "matrix A (.npz scipy sparse, .npy dense, .mtx)")->required();
  app.add_option("--B", b_path, "matrix B (generalized problem)");
  app.add_option("--config", config_path, "config.json (docs/FORMATS.md section 2)");
  app.add_option("--out", out_dir, "output directory")->capture_default_str();
  auto* o_sigma = app.add_option("--sigma", sigma, "target shift (nearest mode)");
  auto* o_k = app.add_option("--k", k, "number of eigenpairs (nearest mode)");
  auto* o_interval = app.add_option("--interval", interval, "all eigenvalues in [LO, HI]")->expected(2);
  auto* o_small = app.add_option("--smallest", smallest, "k algebraically smallest");
  auto* o_large = app.add_option("--largest", largest, "k algebraically largest");
  auto* o_tol = app.add_option("--tol", tol, "backward-error tolerance");
  auto* o_threads = app.add_option("--threads", threads, "threads (0 = all available)");
  auto* o_budget = app.add_option("--memory-budget-gb", budget,
                                  "memory budget in GiB, or auto (default: a share of the smallest memory limit of "
                                  "the process - cgroup, HSOLVE_MEMORY_LIMIT_GB, rlimit, MemTotal)");
  auto* o_prec = app.add_option("--precision", precision,
                                "factorization precision: double | single (half the factor memory, slower; checked, "
                                "falls back to double) | auto (single only when double does not fit in core)")
                     ->check(CLI::IsMember({"double", "single", "auto"}));
  auto* o_slice = app.add_option("--slice-size", slice_size, "max eigenvalues per shift in sliced targets (0 = auto)");
  auto* o_block = app.add_option("--block-size", block_size,
                                 "Krylov block width of the shift-invert runs - HEP/GHEP Lanczos and GHIEP "
                                 "Krylov-Schur (0 = auto, 1 = single vector)");
  auto* o_zero = app.add_option("--zero-filter-tol", zero_filter, "drop |lambda| <= TOL*scale (zero modes); < 0 off");
  auto* o_cert = app.add_option("--certificate", certificate,
                                "completeness check of nearest HEP/GHEP targets: exact (double Sylvester counts) | "
                                "randomized (exclusion check on the held factorization; exact counts where kappa of "
                                "B is not proven) | auto (the cheapest that certifies, predicted from measured "
                                "times) | off")
                     ->check(CLI::IsMember({"exact", "randomized", "auto", "off"}));
  auto* o_gpu = app.add_option("--gpu", gpu_mode,
                               "GPU offload of the factorizations' large dense updates (BLAS hook, FP64 cuBLAS on the "
                               "first visible GPU, loaded at run time): off (default; or HS_GPU) | auto (when the "
                               "factorization is large enough) | on; without a usable GPU the CPU path runs")
                    ->check(CLI::IsMember({"off", "auto", "on"}));
  app.add_option("--log", log_path, "JSON-lines event log file");
  app.add_flag("-v,--verbose", verbose, "debug output");
  app.add_flag("-q,--quiet", quiet, "no summary on stdout");
  app.add_flag("--version", version, "print version and build information");
  o_interval->excludes(o_small)->excludes(o_large);
  o_small->excludes(o_large);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e) == 0 ? 0 : 2;
  }
  if (version) {
    std::printf("%s\n", hs::build_info().c_str());
    return 0;
  }

  try {
    hs::SolverConfig cfg;
    if (!config_path.empty()) cfg = hs::parse_config(read_text(config_path));
    if (*o_sigma) cfg.target.sigma = sigma, cfg.target.mode = hs::TargetMode::Nearest;
    if (*o_k) cfg.target.k = static_cast<hs::index_t>(k);
    if (*o_interval) {
      cfg.target.mode = hs::TargetMode::Interval;
      cfg.target.lower = interval[0];
      cfg.target.upper = interval[1];
    }
    if (*o_small) cfg.target.mode = hs::TargetMode::Smallest, cfg.target.k = static_cast<hs::index_t>(smallest);
    if (*o_large) cfg.target.mode = hs::TargetMode::Largest, cfg.target.k = static_cast<hs::index_t>(largest);
    if (*o_tol) cfg.tol = tol;
    if (*o_threads) cfg.threads = threads;
    if (*o_budget) {  // "auto" or a number of GiB (range checked with the merged configuration below)
      std::size_t used = 0;
      double gb = 0.0;
      if (budget == "auto") {
        used = budget.size();
        gb = hs::kMemoryBudgetAuto;
      } else {
        try {
          gb = std::stod(budget, &used);
        } catch (const std::exception&) {
          used = 0;
        }
      }
      if (budget.empty() || used != budget.size())
        HS_THROW(InvalidInput, "--memory-budget-gb must be a number of GiB or auto, got '" << budget << "'");
      cfg.memory_budget_gb = gb;
    }
    if (*o_slice) cfg.slice_size = static_cast<hs::index_t>(slice_size);
    if (*o_block) cfg.block_size = static_cast<hs::index_t>(block_size);
    if (*o_zero) cfg.zero_filter_tol = zero_filter;
    if (*o_prec) cfg = hs::parse_config("{\"factor_precision\": \"" + precision + "\"}", cfg);
    if (*o_cert) cfg = hs::parse_config("{\"certificate\": \"" + certificate + "\"}", cfg);
    // Re-validate the merged configuration (overrides bypass parse_config's checks otherwise).
    cfg = hs::parse_config(hs::config_to_json(cfg));
    if (*o_gpu) {
      const hs::GpuControl rc = hs::gpu_offload_set_mode(gpu_mode);
      if (rc != hs::GpuControl::Ok && gpu_mode != "off")
        std::fprintf(stderr, "hsolve: --gpu %s: %s; running on the CPU\n", gpu_mode.c_str(),
                     rc == hs::GpuControl::NotBuilt ? "this hsolve was built without the GPU module"
                                                    : "no BLAS hook in this executable");
    }
    const std::string gpu_asked = *o_gpu ? gpu_mode : std::string();

    // Create the output directory (and the log's directory) before anything writes into them.
    fs::create_directories(out_dir);
    if (!log_path.empty() && fs::path(log_path).has_parent_path()) fs::create_directories(fs::path(log_path).parent_path());
    hs::Logger log(quiet ? 0 : (verbose ? 2 : 1), log_path);
    log.event("start", {{"version", std::string(hs::version())}, {"A", a_path}, {"B", b_path}, {"out", out_dir}});

    hs::LoadOptions lo;
    lo.hermitian_tol = cfg.hermitian_tol;
    hs::LoadedMatrix la = hs::load_matrix(a_path, lo);
    std::vector<std::string> notes;
    for (const auto& n : la.notes) notes.push_back("A: " + n);
    std::optional<hs::LoadedMatrix> lb;
    if (!b_path.empty()) {
      lb = hs::load_matrix(b_path, lo);
      for (const auto& n : lb->notes) notes.push_back("B: " + n);
    }
    log.event("loaded", {{"n", static_cast<std::int64_t>(la.order())}, {"complex", la.is_complex() || (lb && lb->is_complex())}});

    const bool complex_problem = la.is_complex() || (lb && lb->is_complex());
    if (complex_problem) {
      // A real A or B is promoted by the rvalue overload of to_complex, which frees the real copy.
      hs::CsrMatrix<hs::cplx> a = la.is_complex() ? std::get<1>(std::move(la.matrix))
                                                  : hs::to_complex(std::get<0>(std::move(la.matrix)));
      hs::CsrMatrix<hs::cplx> b;
      if (lb)
        b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(std::move(lb->matrix)));
      return run<hs::cplx>(a, lb ? &b : nullptr, cfg, log, out_dir, notes, quiet, gpu_asked);
    }
    const auto& a = std::get<0>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<0>(lb->matrix) : nullptr;
    return run<double>(a, b, cfg, log, out_dir, notes, quiet, gpu_asked);
  } catch (const hs::Error& e) {
    std::fprintf(stderr, "hsolve: error [%s]: %s\n", hs::to_string(e.code()), e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hsolve: error: %s\n", e.what());
    return 2;
  }
}
