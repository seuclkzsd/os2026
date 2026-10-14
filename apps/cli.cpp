// hsolve command-line front end: read matrices (+ config), solve, write results (docs/FORMATS.md).
//
//   hsolve --A A.npz [--B B.npz] [--config config.json] --out DIR [overrides...]
//   mpiexec -n R hsolve-mpi <the same arguments>     (apps/hsolve_mpi.cpp, docs/PARALLEL.md)
//
// Exit status: 0 = all pairs accepted (and certificate passed), 1 = finished but not successful,
// 2 = error (message on stderr).
#include "cli.hpp"

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

#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/parallel.hpp"
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

template <class T>
void write_outputs(const fs::path& out, const hs::Solution<T>& sol) {
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
  rep << sol.report.to_json(2) << '\n';
  if (!rep) HS_THROW(Io, "cannot write " << (out / "report.json").string());
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
  const auto& p = r.parallel;
  if (p.processes > 1)
    std::printf("processes: %d (leader %d threads, helpers %d)  helper counts=%lld slices=%lld fallbacks=%lld  "
                "wait=%.3fs\n",
                p.processes, p.threads, p.helper_threads, static_cast<long long>(p.helper_counts),
                static_cast<long long>(p.helper_slices), static_cast<long long>(p.fallbacks), p.wait_s);
  for (const auto& w : r.warnings) std::printf("warning: %s\n", w.c_str());
}

template <class T>
int run_solve(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg, hs::Logger& log,
              const fs::path& out, const std::vector<std::string>& load_notes, bool quiet, hs::Transport* peers,
              const hs::ParallelOptions& popts) {
  hs::Solution<T> sol = peers ? hs::solve_parallel(a, b, cfg, &log, *peers, popts) : hs::solve(a, b, cfg, &log);
  for (const auto& note : load_notes) sol.report.warnings.push_back("input: " + note);
  write_outputs(out, sol);
  if (!quiet) print_summary(sol);
  return sol.report.success ? 0 : 1;
}

/// Rank 0 of a session: ends the helpers' serve() however the solve ends.
class ReleaseHelpers {
 public:
  explicit ReleaseHelpers(hs::Transport* peers) : peers_(peers) {}
  ~ReleaseHelpers() {
    try {
      if (peers_) hs::release_helpers(*peers_);
    } catch (...) {
    }
  }
  ReleaseHelpers(const ReleaseHelpers&) = delete;
  ReleaseHelpers& operator=(const ReleaseHelpers&) = delete;

 private:
  hs::Transport* peers_;
};

}  // namespace

int hs::cli::run(int argc, char** argv, hs::Transport* peers) {
  CLI::App app{"hsolve - eigensolver for large sparse Hermitian (indefinite) eigenproblems"};
  std::string a_path, b_path, config_path, out_dir = "hsolve_out", log_path;
  double sigma = 0.0, tol = 0.0, zero_filter = 0.0;
  long long k = 0, smallest = 0, largest = 0, slice_size = 0, block_size = 0;
  std::vector<double> interval;
  int threads = -1;
  std::string precision, certificate, budget, helper_counts = "auto", helper_slices = "auto";
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
  app.add_option("--helper-counts", helper_counts,
                 "hsolve-mpi: the nearest certificate's two counts on two helper processes at once - auto (when "
                 "they pay against the helpers' preparation), on, off (docs/PARALLEL.md)")
      ->check(CLI::IsMember({"auto", "on", "off"}));
  app.add_option("--helper-slices", helper_slices,
                 "hsolve-mpi: spectrum slices on the helper processes - auto (when a slice outweighs the helpers' "
                 "preparation), on, off")
      ->check(CLI::IsMember({"auto", "on", "off"}));
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
    if (!peers || peers->rank() == 0) std::printf("%s\n", hs::build_info().c_str());
    return 0;
  }
  if (peers && peers->rank() > 0) {
    // A helper: the same arguments on every rank; it only needs the log settings. Its events go to
    // <log>.rank<r> (not to stderr unless -v: one summary, the leader's).
    try {
      const std::string own = log_path.empty() ? std::string() : log_path + ".rank" + std::to_string(peers->rank());
      if (!own.empty() && fs::path(own).has_parent_path()) fs::create_directories(fs::path(own).parent_path());
      hs::Logger log(verbose ? 2 : 0, own);
      hs::serve(*peers, &log);
      return 0;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "hsolve (helper rank %d): error: %s\n", peers->rank(), e.what());
      return 2;
    }
  }
  const ReleaseHelpers release(peers);
  const auto toggle = [](const std::string& s) {
    return s == "on" ? hs::Toggle::On : s == "off" ? hs::Toggle::Off : hs::Toggle::Auto;
  };
  const hs::ParallelOptions popts{toggle(helper_counts), toggle(helper_slices)};

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
      return run_solve<hs::cplx>(a, lb ? &b : nullptr, cfg, log, out_dir, notes, quiet, peers, popts);
    }
    const auto& a = std::get<0>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<0>(lb->matrix) : nullptr;
    return run_solve<double>(a, b, cfg, log, out_dir, notes, quiet, peers, popts);
  } catch (const hs::Error& e) {
    std::fprintf(stderr, "hsolve: error [%s]: %s\n", hs::to_string(e.code()), e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hsolve: error: %s\n", e.what());
    return 2;
  }
}
