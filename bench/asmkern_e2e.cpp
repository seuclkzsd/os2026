// End-to-end A/B runs of the hsolve CLI under different BLAS configurations (module blas, P5
// "asmkern"; docs/BLAS.md): each configuration is a set of environment assignments (LD_PRELOAD
// of the BLAS hook or of another BLAS, HS_BLAS_ROUTE, OPENBLAS_CORETYPE, ...) given to the CLI
// process only.
//
//   bench_asmkern_e2e --cli build/release/hsolve --cases syn-mem27-46,wg-square-1e5
//       [--data data/cases] [--threads 4,8] [--reps 5] [--out DIR] [--run-timeout 600]
//       --config base= --config 'hook=LD_PRELOAD=/abs/libhsolve_blashook.so;HS_BLAS_ROUTE=gemv1'
//       [--override '{"ordering": 5}'] [--override 'wg-square-1e5:{"ordering": 0}']
//
// --override merges JSON into the case's config.json (all cases, or CASE: only), e.g. to fix
// the ordering so that every configuration factorizes the same matrix in the same order (the
// adaptive choice depends on a measured analysis time, and SCOTCH orderings differ between
// processes - DESIGN 3.4, 3.6-3b - which would add noise unrelated to the BLAS).
// Repetitions are interleaved (case x threads x configurations, the configuration order rotated
// per repetition). A run with T threads is pinned to the first T CPUs of this process' affinity
// mask (start the bench through tools/run/guard.py --cores ...) and gets --threads T and
// OMP_NUM_THREADS=T. Per case and thread count the table gives, per configuration, the medians of
// the report's phase times (total, factor, iterate, certificate; DESIGN §3.6-0) and MUMPS phase
// times (factor_stats: numeric factorizations incl. inertia counts with their number, solves with
// the number of right-hand sides, solve time per right-hand side), the speedup of the median total
// against the first configuration, the largest backward error over all runs,
// whether every run succeeded with a passed certificate (or none performed), and the largest
// relative difference of the sorted eigenvalues against the first configuration's first run
// (equal accuracy check), and the range of the main factorization's factor entries over the runs
// (equal work check: a fixed ordering must give the same factorization under every BLAS). A run
// that exceeds --run-timeout seconds is killed and counts as failed. Every run is also written as
// one JSON line to OUT/runs.jsonl.
#include <fcntl.h>
#include <signal.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct Config {
  std::string name;
  std::vector<std::pair<std::string, std::string>> env;
};

struct Run {
  bool ok = false;
  double total = 0, setup = 0, factor = 0, iterate = 0, verify = 0, cert = 0;
  double max_eta = 0;
  double entries = 0;  // factor entries of the main factorization (equal work check)
  // MUMPS phase times from the report's factor_stats: numeric factorizations (incl. the
  // certificate's inertia counts) and solves, with their counts.
  double numeric = 0, solve = 0, factorizations = 0, rhs = 0;
  bool cert_ok = true;
  std::vector<double> lambdas;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    std::size_t end = s.find(sep, pos);
    if (end == std::string::npos) end = s.size();
    if (end > pos) out.push_back(s.substr(pos, end - pos));
    pos = end + 1;
  }
  return out;
}

Config parse_config(const std::string& spec) {
  const std::size_t eq = spec.find('=');
  if (eq == std::string::npos) {
    std::fprintf(stderr, "config '%s': NAME=[VAR=VALUE;...] expected\n", spec.c_str());
    std::exit(2);
  }
  Config c;
  c.name = spec.substr(0, eq);
  for (const std::string& kv : split(spec.substr(eq + 1), ';')) {
    const std::size_t e = kv.find('=');
    if (e == std::string::npos) {
      std::fprintf(stderr, "config '%s': bad assignment '%s'\n", spec.c_str(), kv.c_str());
      std::exit(2);
    }
    c.env.emplace_back(kv.substr(0, e), kv.substr(e + 1));
  }
  return c;
}

double median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// The case's config.json with the override's keys replaced (JSON merge), written next to the
// run's outputs; the case file itself when there is no override.
fs::path run_config(const fs::path& case_dir, const json& override_cfg, const fs::path& out_dir) {
  if (override_cfg.empty()) return case_dir / "config.json";
  json c;
  std::ifstream(case_dir / "config.json") >> c;
  c.merge_patch(override_cfg);
  const fs::path p = out_dir.parent_path() / (out_dir.filename().string() + ".config.json");
  std::ofstream(p) << c.dump(2) << "\n";
  return p;
}

Run run_once(const std::string& cli, const fs::path& case_dir, int threads, const Config& cfg,
             const json& override_cfg, const fs::path& out_dir, double timeout_s) {
  fs::remove_all(out_dir);
  fs::create_directories(out_dir);
  std::vector<std::string> args{cli,        "--A",      (case_dir / "A.npz").string(),
                                "--config", run_config(case_dir, override_cfg, out_dir).string(),
                                "--threads", std::to_string(threads),
                                "--out",    out_dir.string(),
                                "-q"};
  if (fs::exists(case_dir / "B.npz")) {
    args.push_back("--B");
    args.push_back((case_dir / "B.npz").string());
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  sched_getaffinity(0, sizeof mask, &mask);
  cpu_set_t child_mask;
  CPU_ZERO(&child_mask);
  int taken = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE && taken < threads; ++cpu)
    if (CPU_ISSET(cpu, &mask)) {
      CPU_SET(cpu, &child_mask);
      ++taken;
    }
  const pid_t pid = fork();
  if (pid == 0) {
    sched_setaffinity(0, sizeof child_mask, &child_mask);
    const std::string t = std::to_string(threads);
    setenv("OMP_NUM_THREADS", t.c_str(), 1);
    setenv("OPENBLAS_NUM_THREADS", t.c_str(), 1);
    for (const auto& [k, v] : cfg.env) setenv(k.c_str(), v.c_str(), 1);
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    const int fd = ::open("/dev/null", O_WRONLY);
    if (fd >= 0) dup2(fd, 1);
    execv(cli.c_str(), argv.data());
    _exit(127);
  }
  // Wait, killing the run after timeout_s (a configuration that computes garbage may not finish).
  int status = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (waitpid(pid, &status, WNOHANG) == 0) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return Run{};
    }
    usleep(20000);
  }
  Run r;
  std::ifstream in(out_dir / "report.json");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !in) return r;
  json j;
  in >> j;
  r.ok = j.value("success", false);
  r.total = j.value("time_total_s", 0.0);
  r.setup = j.value("time_setup_s", 0.0);
  r.factor = j.value("time_factor_s", 0.0);
  r.iterate = j.value("time_iterate_s", 0.0);
  r.verify = j.value("time_verify_s", 0.0);
  r.cert = j.value("time_certificate_s", 0.0);
  if (j.contains("factor_stats")) {
    const json& st = j["factor_stats"];
    r.entries = st.value("factor_entries", 0.0);
    r.numeric = st.value("factorize_s", 0.0);
    r.factorizations = st.value("factorizations", 0.0);
    r.solve = st.value("solve_s", 0.0);
    r.rhs = st.value("rhs_solved", 0.0);
  }
  for (const auto& p : j.at("pairs")) {
    r.max_eta = std::max(r.max_eta, p.value("backward_error", 0.0));
    r.lambdas.push_back(p.value("lambda_re", 0.0));
  }
  std::sort(r.lambdas.begin(), r.lambdas.end());
  const json& c = j.at("certificate");
  if (c.value("performed", false)) r.cert_ok = c.value("passed", false);
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string cli = "build/release/hsolve", data = "data/cases", out = "asmkern_e2e_out";
  std::vector<std::string> cases;
  std::vector<int> threads{4};
  int reps = 5;
  double run_timeout = 600.0;
  std::vector<Config> configs;
  json global_override = json::object();
  std::map<std::string, json> case_override;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--cli") {
      cli = v;
    } else if (a == "--data") {
      data = v;
    } else if (a == "--cases") {
      cases = split(v, ',');
    } else if (a == "--threads") {
      threads.clear();
      for (const auto& t : split(v, ',')) threads.push_back(std::atoi(t.c_str()));
    } else if (a == "--reps") {
      reps = std::atoi(v.c_str());
    } else if (a == "--run-timeout") {
      run_timeout = std::atof(v.c_str());
    } else if (a == "--out") {
      out = v;
    } else if (a == "--config") {
      configs.push_back(parse_config(v));
    } else if (a == "--override") {
      const std::size_t colon = v.find(':');
      if (!v.empty() && v[0] == '{') {
        global_override.merge_patch(json::parse(v));
      } else if (colon != std::string::npos) {
        case_override[v.substr(0, colon)].merge_patch(json::parse(v.substr(colon + 1)));
      } else {
        std::fprintf(stderr, "--override JSON or CASE:JSON expected\n");
        return 2;
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (cases.empty() || configs.empty()) {
    std::fprintf(stderr, "usage: see the header of bench/asmkern_e2e.cpp\n");
    return 2;
  }
  cli = fs::absolute(cli).string();
  fs::create_directories(out);
  std::ofstream jl(fs::path(out) / "runs.jsonl", std::ios::app);
  const std::size_t nc = configs.size();
  // results[case][threads][config] -> runs
  std::map<std::string, std::map<int, std::vector<std::vector<Run>>>> res;
  for (int r = 0; r < reps; ++r)
    for (const std::string& cs : cases)
      for (int t : threads) {
        auto& slot = res[cs][t];
        slot.resize(nc);
        for (std::size_t q = 0; q < nc; ++q) {
          const std::size_t ci = (q + static_cast<std::size_t>(r)) % nc;
          const Config& cfg = configs[ci];
          const fs::path od = fs::path(out) / ("run-" + cs + "-t" + std::to_string(t) + "-" +
                                               cfg.name);
          json ov = global_override;
          if (case_override.count(cs)) ov.merge_patch(case_override.at(cs));
          const Run run = run_once(cli, fs::path(data) / cs, t, cfg, ov, od, run_timeout);
          slot[ci].push_back(run);
          jl << json{{"case", cs},          {"threads", t},          {"config", cfg.name},
                     {"rep", r},            {"ok", run.ok},          {"total", run.total},
                     {"setup", run.setup},  {"factor", run.factor},  {"iterate", run.iterate},
                     {"verify", run.verify}, {"certificate", run.cert}, {"max_eta", run.max_eta},
                     {"cert_ok", run.cert_ok}, {"factor_entries", run.entries},
                     {"numeric", run.numeric}, {"factorizations", run.factorizations},
                     {"solve", run.solve},     {"rhs", run.rhs}}
                    .dump()
             << "\n";
          jl.flush();
          std::fprintf(stderr, "rep %d %s t=%d %s: %.3f s%s\n", r, cs.c_str(), t, cfg.name.c_str(),
                       run.total, run.ok ? "" : " FAILED");
        }
      }
  std::printf("| case | threads | config | total s | factor s | iterate s | certificate s | "
              "MUMPS numeric s (#) | MUMPS solve s (rhs) | ms per rhs | "
              "speedup | max eta | all ok | eig rel diff | factor entries min..max |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
  for (const std::string& cs : cases)
    for (int t : threads) {
      const auto& slot = res[cs][t];
      const double base_total = [&] {
        std::vector<double> v;
        for (const Run& x : slot[0]) v.push_back(x.total);
        return median(v);
      }();
      const std::vector<double>& base_l = slot[0].empty() ? std::vector<double>{} : slot[0][0].lambdas;
      for (std::size_t ci = 0; ci < nc; ++ci) {
        std::vector<double> tot, fac, it, ce, num, sol, per_rhs, nfac, nrhs;
        double emin = INFINITY, emax = 0;
        double eta = 0, diff = 0;
        bool ok = true;
        for (const Run& x : slot[ci]) {
          tot.push_back(x.total);
          fac.push_back(x.factor);
          it.push_back(x.iterate);
          ce.push_back(x.cert);
          num.push_back(x.numeric);
          sol.push_back(x.solve);
          nfac.push_back(x.factorizations);
          nrhs.push_back(x.rhs);
          if (x.rhs > 0) per_rhs.push_back(1e3 * x.solve / x.rhs);
          eta = std::max(eta, x.max_eta);
          ok = ok && x.ok && x.cert_ok;
          emin = std::min(emin, x.entries);
          emax = std::max(emax, x.entries);
          if (x.lambdas.size() != base_l.size()) {
            diff = INFINITY;
          } else {
            for (std::size_t i = 0; i < base_l.size(); ++i)
              diff = std::max(diff, std::abs(x.lambdas[i] - base_l[i]) /
                                        std::max(1e-300, std::abs(base_l[i])));
          }
        }
        std::printf("| %s | %d | %s | %.3f | %.3f | %.3f | %.3f | %.3f (%.0f) | %.3f (%.0f) | %.2f | "
                    "%.3f | %.1e | %s | %.1e | %.4g..%.4g |\n",
                    cs.c_str(), t, configs[ci].name.c_str(), median(tot), median(fac), median(it),
                    median(ce), median(num), median(nfac), median(sol), median(nrhs),
                    median(per_rhs), base_total / median(tot), eta, ok ? "yes" : "NO", diff, emin,
                    emax);
      }
    }
  return 0;
}
