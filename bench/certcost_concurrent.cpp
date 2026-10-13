// Feasibility and gain of CONCURRENT certificate counts (P8 "certcost", task 2). The two exact
// counts nu(a), nu(b) of a nearest certificate are independent inertia-only factorizations on the
// same pivot order; run concurrently with the threads split (8 + 8 of 16) they could take one
// 8-thread count's time instead of two 16-thread ones. Sequential MUMPS 5.8.2 is not thread-safe
// across instances (factor.hpp; bench_certpar_inertia --exp concurrent --mumps-lock 0 aborts in
// MUMPS_LOAD_INIT), so in one process the only isolation is a second process; this program
// measures what any concurrent scheme could gain and what the fork-based one does gain.
//
//   bench_certcost_concurrent --case DIR --lower a --upper b [--sigma s] [--ordering o]
//                             [--threads t] [--reps r] [--at a|b] [--scaling s] [--detection 0|1]
//                             [--set-blas 0|1] --exp E[,E...]
//
// DIR holds A.npz (B.npz) and config.json (sigma); [a, b] is the certificate window (a hsolve
// report's certificate.lower / upper). o: a DETERMINISTIC ordering (5 METIS, 10 hsolve's nested
// dissection, 0 AMD; default 10), so that separate processes count on the same pivot order. The main
// analysis runs on all threads (OMP_NUM_THREADS); the InertiaCounters (as the driver's: inertia-only,
// on the main analysis' pivot order) on t threads (OpenMP and OpenBLAS, default all).
// Experiments:
//   seq   count(a) then count(b) on one counter with t threads, --reps times (the driver's
//         certificate counts): per count and per pair;
//   one   count(--at) only, --reps times, on t threads: run alone, and as two processes at once on
//         disjoint halves of the slot (taskset), for the process-isolated bound of every concurrent
//         scheme;
//   fork  the in-process scheme by process isolation: two counters of t threads analysed before the
//         loop; per rep the process fork()s, the child counts nu(b) on a fresh pthread (libgomp's
//         thread pool does not survive fork(); a new thread gets a new pool) and returns it through
//         a pipe, the parent counts nu(a) meanwhile, then waits: wall time per pair, and the counts
//         must equal seq's.
// --scaling: ICNTL(8) of the counters (default 7, the driver's; 0 none, to bound what a reused
// scaling could save), --detection 0: counts without null-pivot detection (its cost), --set-blas 1:
// openblas_set_num_threads(t) before every count (with the OpenMP OpenBLAS of the conda environment
// this hung the 8-thread count in the build lane: off by default).
// Output: one JSON object per line on stdout. Run through tools/run/guard.py.
//
// P8 commands (gpu01 measurement slot, 16 threads; cases, windows and results in
// results/p8_certcost.md §3), each one command line:
//   env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 --
//     build/release/bench_certcost_concurrent --case data/cases/<case> --lower <a> --upper <b>
//     --threads 16 --reps 5 --exp seq
//   (the same with --threads 8 --exp seq,fork) and, for the two-process bound,
//   env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 -- bash -c
//     'taskset -c 16-23 build/release/bench_certcost_concurrent ... --threads 8 --exp one --at a &
//      taskset -c 24-31 build/release/bench_certcost_concurrent ... --threads 8 --exp one --at b &
//      wait'
// The script tools/bench/certcost_concurrent.sh runs all of it for the P8 cases.
#include <omp.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

extern "C" void openblas_set_num_threads(int);

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Args {
  fs::path case_dir;
  std::vector<std::string> exps;
  double sigma = kNaN, lower = kNaN, upper = kNaN;
  int ordering = hs::kOrderingNestedDissection;
  int threads = 0;
  int reps = 5;
  bool set_blas = false;
  int scaling = 7;        // ICNTL(8) of the counters (the driver's: 7)
  bool detection = true;  // null-pivot detection of the counts (the driver's: on)
  std::string at = "a";
};

bool g_set_blas = false;
int g_scaling = 7;
bool g_detection = true;

void emit(const json& j) { std::cout << j.dump() << std::endl; }

double median(std::vector<double> v) {
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <class T>
std::unique_ptr<hs::InertiaCounter<T>> make_counter(const hs::SparseFactorization<T>& fac, int threads) {
  hs::FactorOptions o = fac.options();
  o.threads = threads;
  o.scaling = g_scaling;
  return std::make_unique<hs::InertiaCounter<T>>(fac, o);
}

/// One count on `threads` OpenMP and OpenBLAS threads.
template <class T>
hs::Inertia count(hs::InertiaCounter<T>& c, double s, int threads) {
  if (g_set_blas) openblas_set_num_threads(threads);
  return c.count(1.0, -s, g_detection);
}

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  const int all = omp_get_max_threads();
  const int t = args.threads > 0 ? args.threads : all;
  hs::FactorOptions fo;
  fo.ordering = args.ordering;
  fo.threads = all;
  fo.deterministic = false;  // L0-OpenMP, the driver default
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer ta;
  fac->analyze(pencil, 1.0, -args.sigma);
  const hs::FactorEstimate est = fac->estimate();
  emit({{"exp", "setup"}, {"case", args.case_dir.filename().string()}, {"n", pencil.order()},
        {"complex", hs::is_complex_v<T>}, {"ordering", args.ordering}, {"threads_all", all}, {"threads", t},
        {"lower", args.lower}, {"upper", args.upper}, {"analyze_s", ta.seconds()},
        {"factor_entries", est.factor_entries}, {"flops", est.flops}, {"incore_mb", est.incore_mb}});
  index_t seq_nu_a = -1, seq_nu_b = -1;
  for (const std::string& e : args.exps) {
    if (e == "seq") {
      auto c = make_counter(*fac, t);
      std::vector<double> ta_s, tb_s, pair_s;
      for (int r = 0; r < args.reps; ++r) {
        hs::Timer tp;
        hs::Timer t1;
        const hs::Inertia ia = count(*c, args.lower, t);
        const double s1 = t1.seconds();
        hs::Timer t2;
        const hs::Inertia ib = count(*c, args.upper, t);
        const double s2 = t2.seconds();
        pair_s.push_back(tp.seconds());
        ta_s.push_back(s1);
        tb_s.push_back(s2);
        seq_nu_a = ia.neg;
        seq_nu_b = ib.neg;
        if (!ia.valid || !ib.valid) HS_THROW(Internal, "a window end is singular");
      }
      emit({{"exp", "seq"}, {"threads", t}, {"count_a_s", median(ta_s)}, {"count_b_s", median(tb_s)},
            {"pair_s", median(pair_s)}, {"pair_all_s", pair_s}, {"nu_a", seq_nu_a}, {"nu_b", seq_nu_b},
            {"counter_mb", c->estimate().incore_mb}, {"counter_analyze_s", c->stats().analyze_s}});
    } else if (e == "one") {
      const double s = args.at == "b" ? args.upper : args.lower;
      auto c = make_counter(*fac, t);
      std::vector<double> ts;
      index_t nu = -1;
      for (int r = 0; r < args.reps; ++r) {
        hs::Timer t1;
        nu = count(*c, s, t).neg;
        ts.push_back(t1.seconds());
      }
      emit({{"exp", "one"}, {"at", args.at}, {"threads", t}, {"count_s", median(ts)}, {"count_all_s", ts},
            {"nu", nu}});
    } else if (e == "fork") {
      auto ca = make_counter(*fac, t);
      auto cb = make_counter(*fac, t);
      std::vector<double> pair_s, parent_s;
      std::vector<int> child_status;
      bool agree = true;
      index_t nu_a = -1, nu_b = -1;
      for (int r = 0; r < args.reps; ++r) {
        int fds[2];
        if (pipe(fds) != 0) HS_THROW(Internal, "pipe failed");
        hs::Timer tp;
        const pid_t pid = fork();
        if (pid < 0) HS_THROW(Internal, "fork failed");
        if (pid == 0) {
          // Child: a fresh pthread (a new libgomp pool); the result through the pipe; _exit (no
          // atexit handlers, no stdio flush of the parent's buffers).
          close(fds[0]);
          std::int64_t out[2] = {-1, 0};
          std::thread th([&]() {
            try {
              const hs::Inertia in = count(*cb, args.upper, t);
              out[0] = in.neg;
              out[1] = in.valid ? 1 : 0;
            } catch (...) {
              out[0] = -2;
            }
          });
          th.join();
          const ssize_t w = write(fds[1], out, sizeof(out));
          _exit(w == static_cast<ssize_t>(sizeof(out)) ? 0 : 1);
        }
        close(fds[1]);
        hs::Timer t1;
        const hs::Inertia ia = count(*ca, args.lower, t);
        parent_s.push_back(t1.seconds());
        std::int64_t in[2] = {-1, 0};
        const ssize_t got = read(fds[0], in, sizeof(in));
        close(fds[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        pair_s.push_back(tp.seconds());
        nu_a = ia.neg;
        nu_b = got == static_cast<ssize_t>(sizeof(in)) ? in[0] : -3;
        const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && in[1] == 1 && ia.valid;
        child_status.push_back(WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status));
        agree = agree && ok && (seq_nu_a < 0 || (nu_a == seq_nu_a && nu_b == seq_nu_b));
      }
      emit({{"exp", "fork"}, {"threads", t}, {"pair_s", median(pair_s)}, {"pair_all_s", pair_s},
            {"parent_count_s", median(parent_s)}, {"child_status", child_status}, {"nu_a", nu_a}, {"nu_b", nu_b}, {"agree", agree}});
    } else {
      HS_THROW(InvalidInput, "unknown --exp " << e);
    }
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
    try {
      if (k == "--case") args.case_dir = need(i);
      else if (k == "--lower") args.lower = std::stod(need(i));
      else if (k == "--upper") args.upper = std::stod(need(i));
      else if (k == "--sigma") args.sigma = std::stod(need(i));
      else if (k == "--ordering") args.ordering = std::stoi(need(i));
      else if (k == "--threads") args.threads = std::stoi(need(i));
      else if (k == "--reps") args.reps = std::stoi(need(i));
      else if (k == "--at") args.at = need(i);
      else if (k == "--set-blas") args.set_blas = std::stoi(need(i)) != 0;
      else if (k == "--scaling") args.scaling = std::stoi(need(i));
      else if (k == "--detection") args.detection = std::stoi(need(i)) != 0;
      else if (k == "--exp") {
        std::stringstream ss(need(i));
        std::string e;
        while (std::getline(ss, e, ',')) args.exps.push_back(e);
      } else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty() || args.exps.empty() || std::isnan(args.lower) || std::isnan(args.upper)) {
    std::cerr << "usage: bench_certcost_concurrent --case DIR --lower a --upper b --exp E[,E...] [...]\n";
    return 2;
  }
  g_set_blas = args.set_blas;
  g_scaling = args.scaling;
  g_detection = args.detection;
  try {
    if (std::isnan(args.sigma)) {
      std::ifstream f(args.case_dir / "config.json");
      std::stringstream ss;
      ss << f.rdbuf();
      args.sigma = hs::parse_config(ss.str()).target.sigma;
    }
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(m.matrix)
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(m.matrix));
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
