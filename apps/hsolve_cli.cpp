// hsolve command-line front end: read matrices (+ config), solve, write results (docs/FORMATS.md).
//
//   hsolve --A A.npz [--B B.npz] [--config config.json] --out DIR [overrides...]
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

#include "hsolve/error.hpp"
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
  if (r.certificate.performed)
    std::printf("certificate: [%g, %g] inertia=%lld found=%lld -> %s\n", r.certificate.lower, r.certificate.upper,
                static_cast<long long>(r.certificate.count_inertia), static_cast<long long>(r.certificate.count_found),
                r.certificate.passed ? "passed" : "FAILED");
  for (const auto& w : r.warnings) std::printf("warning: %s\n", w.c_str());
}

template <class T>
int run(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg, hs::Logger& log,
        const fs::path& out, const std::vector<std::string>& load_notes, bool quiet) {
  hs::Solution<T> sol = hs::solve(a, b, cfg, &log);
  for (const auto& note : load_notes) sol.report.warnings.push_back("input: " + note);
  write_outputs(out, sol);
  if (!quiet) print_summary(sol);
  return sol.report.success ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"hsolve - eigensolver for large sparse Hermitian (indefinite) eigenproblems"};
  std::string a_path, b_path, config_path, out_dir = "hsolve_out", log_path;
  double sigma = 0.0, tol = 0.0, budget = 0.0;
  long long k = 0, smallest = 0, largest = 0;
  std::vector<double> interval;
  int threads = -1;
  std::string precision;
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
  auto* o_budget = app.add_option("--memory-budget-gb", budget, "memory budget in GiB");
  auto* o_prec = app.add_option("--precision", precision, "factorization precision: double|single");
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
    if (*o_budget) cfg.memory_budget_gb = budget;
    if (*o_prec) cfg = hs::parse_config("{\"factor_precision\": \"" + precision + "\"}", cfg);
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
      hs::CsrMatrix<hs::cplx> a = la.is_complex() ? std::get<1>(std::move(la.matrix)) : hs::to_complex(std::get<0>(la.matrix));
      hs::CsrMatrix<hs::cplx> b;
      if (lb) b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(lb->matrix));
      return run<hs::cplx>(a, lb ? &b : nullptr, cfg, log, out_dir, notes, quiet);
    }
    const auto& a = std::get<0>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<0>(lb->matrix) : nullptr;
    return run<double>(a, b, cfg, log, out_dir, notes, quiet);
  } catch (const hs::Error& e) {
    std::fprintf(stderr, "hsolve: error [%s]: %s\n", hs::to_string(e.code()), e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hsolve: error: %s\n", e.what());
    return 2;
  }
}
