// BLAS profile report (module blas, P5 "asmkern"; docs/BLAS.md): turns the JSON files written by
// the BLAS hook's profiler (HS_BLAS_PROFILE, src/blas/blas_profile.hpp) into markdown tables.
//
//   bench_asmkern_report [--top N] [--peak GFLOPS_PER_CORE] [--stream GBS_PER_CORE] FILE...
//
// Several files (e.g. the processes of one run, or repetitions) are summed. Output:
//   1. MUMPS phases: calls, wall seconds, right-hand sides; BLAS time inside the phase, split into
//      calls outside OpenMP parallel regions (wall seconds; above MUMPS' L0 layer) and inside
//      (thread-seconds; the L0 subtrees), and the share of the phase's wall time spent outside.
//   2. Per phase and routine (x in/out of parallel regions x route): calls, seconds, GFlop/s, GB/s
//      (minimum traffic: every operand once), and the efficiency against the given per-core peak
//      and STREAM bandwidth when provided (outside parallel regions the BLAS may use several
//      cores: the percentage is then per process, against threads x per-core).
//   3. The N most expensive shape classes per phase (routine, operand flags, size classes).
// Size class c of a dimension covers [2^(c-1), 2^c - 1].
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Acc {
  long long calls = 0;
  double seconds = 0.0, flops = 0.0, bytes = 0.0;
  void add(const nlohmann::json& c) {
    calls += c.at("calls").get<long long>();
    seconds += c.at("seconds").get<double>();
    flops += c.at("flops").get<double>();
    bytes += c.at("bytes").get<double>();
  }
};

struct Mumps {
  long long calls = 0, rhs = 0;
  double seconds = 0.0;
};

std::string cls(int c) {
  if (c <= 0) return "0";
  const long long lo = 1LL << (c - 1);
  const long long hi = (1LL << c) - 1;
  if (lo == hi) return std::to_string(lo);
  auto fmt = [](long long v) {
    if (v >= 1024 && v % 1024 == 0) return std::to_string(v / 1024) + "k";
    if (v >= 1023 && (v + 1) % 1024 == 0) return std::to_string((v + 1) / 1024) + "k-1";
    return std::to_string(v);
  };
  return fmt(lo) + ".." + fmt(hi);
}

const char* const kPhases[] = {"analysis", "factorization", "count", "solve1", "solvep", "other",
                               "none"};

}  // namespace

int main(int argc, char** argv) {
  int top = 12;
  double peak = 0.0, stream = 0.0;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--top" && i + 1 < argc) {
      top = std::atoi(argv[++i]);
    } else if (a == "--peak" && i + 1 < argc) {
      peak = std::atof(argv[++i]);
    } else if (a == "--stream" && i + 1 < argc) {
      stream = std::atof(argv[++i]);
    } else {
      files.push_back(a);
    }
  }
  if (files.empty()) {
    std::fprintf(stderr,
                 "usage: bench_asmkern_report [--top N] [--peak GF/core] [--stream GB/s/core] "
                 "FILE...\n");
    return 2;
  }
  std::map<std::string, Mumps> mumps;
  // (phase, op, par, route) and (phase, op, flags, mc, nc, kc, par, route)
  std::map<std::tuple<std::string, std::string, int, std::string>, Acc> by_op;
  std::map<std::tuple<std::string, std::string, std::string, int, int, int, int, std::string>, Acc>
      by_shape;
  int threads = 0;
  std::string route, blas;
  for (const std::string& f : files) {
    std::ifstream in(f);
    if (!in) {
      std::fprintf(stderr, "cannot open %s\n", f.c_str());
      return 1;
    }
    nlohmann::json j;
    in >> j;
    threads = std::max(threads, j.value("omp_max_threads", 0));
    route = j.value("route", "");
    blas = j.value("blas", "");
    for (const auto& m : j.at("mumps")) {
      Mumps& s = mumps[m.at("phase").get<std::string>()];
      s.calls += m.at("calls").get<long long>();
      s.seconds += m.at("seconds").get<double>();
      s.rhs += m.at("rhs").get<long long>();
    }
    for (const auto& c : j.at("calls")) {
      const std::string ph = c.at("phase"), op = c.at("op"), fl = c.at("flags"),
                        rt = c.at("route");
      const int par = c.at("par").get<int>();
      by_op[{ph, op, par, rt}].add(c);
      by_shape[{ph, op, fl, c.at("mc").get<int>(), c.at("nc").get<int>(), c.at("kc").get<int>(),
                par, rt}]
          .add(c);
    }
  }
  std::printf("files: %zu, OpenMP threads: %d, route: %s, BLAS: %s\n\n", files.size(), threads,
              route.c_str(), blas.c_str());

  // 1. phases
  std::printf("| phase | MUMPS calls | MUMPS wall s | rhs | BLAS s outside par (wall) | "
              "BLAS thread-s inside par | outside / wall |\n|---|---|---|---|---|---|---|\n");
  for (const char* ph : kPhases) {
    double out = 0.0, in = 0.0;
    for (const auto& [key, a] : by_op)
      if (std::get<0>(key) == ph) (std::get<2>(key) ? in : out) += a.seconds;
    const auto it = mumps.find(ph);
    if (it == mumps.end() && out + in == 0.0) continue;
    const Mumps m = it == mumps.end() ? Mumps{} : it->second;
    std::printf("| %s | %lld | %.3f | %lld | %.3f | %.3f | %s |\n", ph, m.calls, m.seconds, m.rhs,
                out, in,
                m.seconds > 0 ? (std::to_string(static_cast<int>(100.0 * out / m.seconds + 0.5)) +
                                 "%")
                                    .c_str()
                              : "-");
  }

  // 2. per phase and routine
  auto eff = [&](const Acc& a, int par) {
    char buf[96];
    const double gf = a.seconds > 0 ? a.flops / a.seconds * 1e-9 : 0.0;
    const double gb = a.seconds > 0 ? a.bytes / a.seconds * 1e-9 : 0.0;
    const double cores = par ? 1.0 : std::max(1, threads);
    std::string pk = peak > 0 && a.flops > 0
                         ? std::to_string(static_cast<int>(100.0 * gf / (peak * cores) + 0.5)) + "%"
                         : "-";
    std::string st = stream > 0
                         ? std::to_string(static_cast<int>(100.0 * gb / (stream * cores) + 0.5)) + "%"
                         : "-";
    std::snprintf(buf, sizeof buf, "%.1f | %.1f | %s | %s", gf, gb, pk.c_str(), st.c_str());
    return std::string(buf);
  };
  std::printf("\n| phase | routine | in par | route | calls | seconds | GFlop/s | GB/s | "
              "%% peak | %% STREAM |\n|---|---|---|---|---|---|---|---|---|---|\n");
  for (const char* ph : kPhases) {
    std::vector<std::pair<std::tuple<std::string, std::string, int, std::string>, Acc>> rows;
    for (const auto& kv : by_op)
      if (std::get<0>(kv.first) == ph) rows.push_back(kv);
    std::sort(rows.begin(), rows.end(),
              [](const auto& x, const auto& y) { return x.second.seconds > y.second.seconds; });
    for (const auto& [key, a] : rows) {
      if (a.seconds < 1e-4) continue;
      std::printf("| %s | %s | %d | %s | %lld | %.3f | %s |\n", ph, std::get<1>(key).c_str(),
                  std::get<2>(key), std::get<3>(key).c_str(), a.calls, a.seconds,
                  eff(a, std::get<2>(key)).c_str());
    }
  }

  // 3. top shape classes
  std::printf("\n| phase | routine | flags | m | n | k | in par | route | calls | seconds | "
              "GFlop/s | GB/s | %% peak | %% STREAM |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
  for (const char* ph : kPhases) {
    std::vector<std::pair<std::tuple<std::string, std::string, std::string, int, int, int, int,
                                     std::string>,
                          Acc>>
        rows;
    for (const auto& kv : by_shape)
      if (std::get<0>(kv.first) == ph) rows.push_back(kv);
    std::sort(rows.begin(), rows.end(),
              [](const auto& x, const auto& y) { return x.second.seconds > y.second.seconds; });
    int shown = 0;
    for (const auto& [key, a] : rows) {
      if (shown++ >= top || a.seconds < 1e-3) break;
      std::printf("| %s | %s | %s | %s | %s | %s | %d | %s | %lld | %.3f | %s |\n", ph,
                  std::get<1>(key).c_str(), std::get<2>(key).c_str(),
                  cls(std::get<3>(key)).c_str(), cls(std::get<4>(key)).c_str(),
                  cls(std::get<5>(key)).c_str(), std::get<6>(key), std::get<7>(key).c_str(),
                  a.calls, a.seconds, eff(a, std::get<6>(key)).c_str());
    }
  }
  return 0;
}
