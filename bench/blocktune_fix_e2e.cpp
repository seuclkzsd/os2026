// End-to-end CLI runs per Krylov block size (module eigen, "blocktune"): the method of the
// end-to-end table of the block-size rule (src/eigen/block_rule.cpp), reproducible from the repo.
//
// For each case and each entry of --p, --reps repetitions, interleaved (repetitions outermost,
// then the cases, then the case's entries rotated by one per repetition), this runs the hsolve CLI
// exactly as tools/bench/compare.py does (--threads = hs::max_threads(), --memory-budget-gb =
// 0.9 * --mem-gb, -q) plus --block-size, with the result directory RUN/<case>/<label>-r<rep>/
// (docs/FORMATS.md §3; an earlier one is removed first). Entries of --p:
//   0     label "auto": --block-size 0, the driver's auto width (P4 rule: 4 for k >= 8, else 1);
//   p>0   label "p<p>": --block-size p;
//   rule  label "rule": eigen_detail::recommended_block_size for the case's shape, from an untimed
//         warm-up CLI run per case (--block-size 1 --log, result in RUN-warmup/<case>/): n, k,
//         nnz(B), scalar type from report.json, and the entries of the main shift's factorization
//         from the log's "factorization" event (what solve_nearest has when it picks the width).
//         A pick that another entry already runs (the same explicit p, or "auto" when the driver's
//         auto width is that p) is not run twice: the case record names the entry it equals.
// One JSON object per line on stdout: a "case" record per case (shape, the rule's pick and the
// modelled step costs, the deterministic AMD-probe estimate of the "ordering" event, the driver's
// auto width), then a "run" record per run (exit status, wall time, and from report.json: success,
// block size, operator applications, restarts, the time split, factorizations, right-hand sides,
// MUMPS solve time, factor entries, certificate, peak RSS, warnings).
// Accuracy is judged by tools/eval afterwards, outside the slot (build/test lane), on the raw run:
//   python3 tools/run/guard.py --skip-busy-check --timeout 3600 -- python3 -m tools.eval.sweep RUN
//       --out OUT.md
//
// Usage (timed: the measurement slot, 16 threads = the judging configuration):
//   python3 tools/run/guard.py --exclusive --timeout 2700 --
//     build/release/bench_blocktune_fix_e2e --run RUN [--cli build/release/hsolve] [--data DIR]
//     [--p 1,0,rule] [--reps 5] [--mem-gb 32] CASE...
// DIR defaults to $HS_DATA_ROOT or ./data; CASE is a directory name under DIR/cases.
#include <spawn.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "eigen/block_rule.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

extern char** environ;

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

/// --p entry standing for the rule's pick.
constexpr index_t kRule = -1;

struct Args {
  fs::path run;
  fs::path cli = "build/release/hsolve";
  fs::path data;
  std::vector<index_t> p = {1, 0, kRule};
  int reps = 5;
  double mem_gb = 32.0;
  std::vector<std::string> cases;
};

Args parse_args(int argc, char** argv) {
  Args a;
  const char* root = std::getenv("HS_DATA_ROOT");
  a.data = root ? fs::path(root) : fs::path("data");
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--run") {
      a.run = value();
    } else if (s == "--cli") {
      a.cli = value();
    } else if (s == "--data") {
      a.data = value();
    } else if (s == "--p") {
      a.p.clear();
      std::stringstream ss(value());
      std::string item;
      while (std::getline(ss, item, ',')) {
        const index_t p = item == "rule" ? kRule : std::stoll(item);
        if (p < kRule) throw std::runtime_error("bad --p entry " + item);
        a.p.push_back(p);
      }
    } else if (s == "--reps") {
      a.reps = std::stoi(value());
    } else if (s == "--mem-gb") {
      a.mem_gb = std::stod(value());
    } else {
      a.cases.push_back(s);
    }
  }
  if (a.run.empty()) throw std::runtime_error("--run DIR is required");
  if (a.cases.empty()) throw std::runtime_error("no case given");
  if (a.p.empty()) throw std::runtime_error("--p is empty");
  return a;
}

std::string label(index_t p) {
  if (p == kRule) return "rule";
  if (p == 0) return "auto";
  return "p" + std::to_string(p);
}

json read_json(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  json j;
  in >> j;
  return j;
}

/// Runs argv[0] with argv (no shell), waits; returns the exit status (128 + signal if killed).
int spawn_and_wait(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  pid_t pid = 0;
  const int err = posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
  if (err != 0) throw std::runtime_error("posix_spawn " + args[0] + ": " + std::strerror(err));
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) throw std::runtime_error(std::string("waitpid: ") + std::strerror(errno));
  }
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
}

struct Case {
  std::string name;
  fs::path dir;
  index_t k = 0;
  bool has_b = false;
  index_t rule_p = 0;  // 0: not computed
  std::vector<index_t> entries;  // the --p entries run for this case
};

std::vector<std::string> cli_command(const Args& args, const Case& c, const fs::path& out,
                                     index_t block_size, int threads) {
  std::ostringstream budget;
  budget << 0.9 * args.mem_gb;
  std::vector<std::string> cmd = {args.cli.string(), "--A", (c.dir / "A.npz").string()};
  if (c.has_b) {
    cmd.push_back("--B");
    cmd.push_back((c.dir / "B.npz").string());
  }
  for (const std::string& s : {std::string("--config"), (c.dir / "config.json").string(),
                               std::string("--out"), out.string(), std::string("--threads"),
                               std::to_string(threads), std::string("--memory-budget-gb"),
                               budget.str(), std::string("-q"), std::string("--block-size"),
                               std::to_string(block_size)})
    cmd.push_back(s);
  return cmd;
}

/// Untimed warm-up run (block size 1, event log) and the rule's pick for the case's shape.
json warm_up(const Args& args, Case& c, int threads) {
  const fs::path dir = args.run.parent_path() / (args.run.filename().string() + "-warmup") / c.name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path out = dir / "result", log = dir / "events.jsonl";
  std::vector<std::string> cmd = cli_command(args, c, out, 1, threads);
  cmd.push_back("--log");
  cmd.push_back(log.string());
  const int rc = spawn_and_wait(cmd);
  if (rc != 0 && rc != 1)
    throw std::runtime_error(c.name + ": warm-up CLI run failed, exit status " +
                             std::to_string(rc));
  const json rep = read_json(out / "report.json");
  json head = {{"record", "case"}, {"case", c.name}, {"k", c.k}};
  double entries = -1.0;
  std::ifstream in(log);
  for (std::string line; std::getline(in, line);) {
    const json e = json::parse(line, nullptr, false);
    if (e.is_discarded()) continue;
    if (e.value("event", "") == "factorization" && entries < 0.0)
      entries = e.at("entries").get<double>();
    if (e.value("event", "") == "ordering") {
      head["amd_entries"] = e.value("amd_entries", json());
      head["ordering_choice"] = e.value("choice", json());
    }
  }
  if (entries < 0.0) throw std::runtime_error(c.name + ": no factorization event in " + log.string());
  hs::eigen_detail::ShiftInvertShape shape;
  shape.n = rep.at("n").get<index_t>();
  shape.nev = std::min(shape.n, c.k);
  shape.factor_entries = entries;
  shape.nnz_b = rep.at("nnz_b").get<double>();
  shape.complex_scalars = rep.at("complex_scalars").get<bool>();
  c.rule_p = hs::eigen_detail::recommended_block_size(shape);
  const auto cost = hs::eigen_detail::modelled_step_seconds(shape);
  head["n"] = shape.n;
  head["nnz_b"] = shape.nnz_b;
  head["complex"] = shape.complex_scalars;
  head["factor_entries"] = entries;
  head["rule_block_size"] = c.rule_p;
  head["rule_step_s"] = std::vector<double>(cost.begin(), cost.end());
  return head;
}

json run_record(const Case& c, index_t entry, index_t block_size, int rep, int rc, double wall_s,
                const fs::path& out) {
  json rec = {{"record", "run"}, {"case", c.name},   {"config", label(entry)},
              {"block_size_arg", block_size},        {"rep", rep},
              {"rc", rc},        {"wall_s", wall_s}, {"dir", out.string()}};
  if (!fs::exists(out / "report.json")) {
    rec["report_error"] = "no report.json";
    return rec;
  }
  const json r = read_json(out / "report.json");
  for (const char* key : {"success", "block_size", "op_applies", "restarts", "time_total_s",
                          "time_setup_s", "time_factor_s", "time_iterate_s", "time_verify_s",
                          "time_certificate_s", "peak_rss_bytes", "warnings", "certificate"})
    rec[key] = r.value(key, json());
  const json f = r.value("factor_stats", json::object());
  for (const char* key : {"factorizations", "rhs_solved", "solve_s", "factor_entries"})
    rec[key] = f.value(key, json());
  return rec;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const int threads = hs::max_threads();
    if (!fs::exists(args.cli)) throw std::runtime_error("no CLI at " + args.cli.string());
    std::vector<Case> cases;
    const bool want_rule = std::find(args.p.begin(), args.p.end(), kRule) != args.p.end();
    for (const std::string& name : args.cases) {
      Case c;
      c.name = name;
      c.dir = args.data / "cases" / name;
      const json cfg = read_json(c.dir / "config.json");
      const json& target = cfg.at("target");
      if (target.value("mode", "nearest") != "nearest")
        throw std::runtime_error(name + ": target is not 'nearest'");
      c.k = target.at("k").get<index_t>();
      c.has_b = fs::exists(c.dir / "B.npz");
      json head = {{"record", "case"}, {"case", name}, {"k", c.k}};
      std::ostringstream text;
      text << std::ifstream(c.dir / "config.json").rdbuf();
      hs::SolverConfig auto_cfg = hs::parse_config(text.str());
      auto_cfg.block_size = 0;
      // The driver's auto width before its memory fit (P4: block_size_for).
      const index_t auto_p = hs::driver::block_size_for(auto_cfg, c.k);
      if (want_rule) head = warm_up(args, c, threads);
      head["auto_block_size"] = auto_p;
      head["threads"] = threads;
      for (index_t e : args.p) {
        if (e == kRule) {
          // Already run under another entry?
          const auto same = std::find_if(args.p.begin(), args.p.end(), [&](index_t q) {
            return q == c.rule_p || (q == 0 && auto_p == c.rule_p);
          });
          if (same != args.p.end()) {
            head["rule_same_as"] = label(*same);
            continue;
          }
        }
        c.entries.push_back(e);
      }
      std::cout << head.dump() << std::endl;
      cases.push_back(std::move(c));
    }
    for (int rep = 0; rep < args.reps; ++rep) {
      for (const Case& c : cases) {
        const std::size_t m = c.entries.size();
        for (std::size_t i = 0; i < m; ++i) {
          const index_t e = c.entries[(i + static_cast<std::size_t>(rep)) % m];
          const index_t block_size = e == kRule ? c.rule_p : e;
          const fs::path out = args.run / c.name / (label(e) + "-r" + std::to_string(rep));
          fs::remove_all(out);
          fs::create_directories(out.parent_path());
          hs::Timer wall;
          const int rc = spawn_and_wait(cli_command(args, c, out, block_size, threads));
          std::cout << run_record(c, e, block_size, rep, rc, wall.seconds(), out).dump()
                    << std::endl;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "bench_blocktune_fix_e2e: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
