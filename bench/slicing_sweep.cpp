// Measurement program for spectrum slicing and one-sided targets (module driver, P3 "slicing").
// Runs hs::solve on a case for several settings and prints one JSON line per run (stdout):
//
//   bench_slicing_sweep --A A.npz [--B B.npz] --threads 2 interval LO HI SIZE...
//       interval target, slice_size = each SIZE (0 = the cost model's choice)
//   bench_slicing_sweep --A A.npz [--B B.npz] smallest K | largest K
//       the inertia-search one-sided path, then (standard problems) the pre-P3 path for
//       comparison: nearest K at 5% of the Ritz spread beyond the end of the spectrum ("legacy")
//   bench_slicing_sweep --A A.npz --B B.npz nearest SIGMA K [ZERO_FILTER_TOL]
//       nearest target (GHIEP zero-cluster handling)
//
// Run it through tools/run/guard.py (docs/DESIGN.md §6). Fields: setting, wall and phase times,
// factorizations, solves, slices, certificate counts, success, worst backward error, and the
// eigenvalues (to compare against data/refs).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"

namespace {

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: bench_slicing_sweep --A A.npz [--B B.npz] [--threads T] "
               "(interval LO HI SIZE... | smallest K | largest K | nearest SIGMA K [ZTOL])\n");
  std::exit(2);
}

template <class T>
nlohmann::json run_one(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg,
                       const std::string& setting) {
  hs::Timer wall;
  const auto sol = hs::solve(a, b, cfg);
  const double wall_s = wall.seconds();
  const auto& r = sol.report;
  double worst = 0.0;
  for (const auto& p : r.pairs) worst = std::max(worst, p.backward_error);
  nlohmann::json j = {{"setting", setting},
                      {"wall_s", wall_s},
                      {"total_s", r.time_total_s},
                      {"setup_s", r.time_setup_s},
                      {"factor_s", r.time_factor_s},
                      {"iterate_s", r.time_iterate_s},
                      {"verify_s", r.time_verify_s},
                      {"factorizations", r.factor_stats.factorizations},
                      {"solves", r.factor_stats.rhs_solved},
                      {"solve_s", r.factor_stats.solve_s},
                      {"restarts", r.restarts},
                      {"slices", r.slices.size()},
                      {"count_inertia", r.certificate.count_inertia},
                      {"count_found", r.certificate.count_found},
                      {"certificate", r.certificate.passed},
                      {"zero_cluster_dim", r.zero_cluster_dim},
                      {"pairs", sol.eigenvalues.size() + sol.complex_eigenvalues.size()},
                      {"success", r.success},
                      {"max_backward_error", worst},
                      {"peak_rss_gib", static_cast<double>(r.peak_rss_bytes) / 1073741824.0},
                      {"eigenvalues", sol.eigenvalues}};
  return j;
}

template <class T>
void sweep(const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, int threads, const std::vector<std::string>& args) {
  hs::SolverConfig base;
  base.threads = threads;
  base.tol = 1e-10;
  base.memory_budget_gb = 7.0;
  const std::string mode = args.at(0);
  auto emit = [&](nlohmann::json j) { std::printf("%s\n", j.dump().c_str()); std::fflush(stdout); };
  if (mode == "interval") {
    if (args.size() < 4) usage();
    for (std::size_t i = 3; i < args.size(); ++i) {
      hs::SolverConfig cfg = base;
      cfg.target.mode = hs::TargetMode::Interval;
      cfg.target.lower = std::stod(args[1]);
      cfg.target.upper = std::stod(args[2]);
      cfg.slice_size = std::stoll(args[i]);
      emit(run_one(a, b, cfg, "interval slice_size=" + args[i]));
    }
  } else if (mode == "smallest" || mode == "largest") {
    if (args.size() < 2) usage();
    const bool smallest = mode == "smallest";
    hs::SolverConfig cfg = base;
    cfg.target.mode = smallest ? hs::TargetMode::Smallest : hs::TargetMode::Largest;
    cfg.target.k = std::stoll(args[1]);
    emit(run_one(a, b, cfg, mode + " inertia-search"));
    if (!b) {
      // Pre-P3 standard path: nearest k to a shift 5% of the Ritz spread beyond the estimated end.
      const auto na = hs::driver::estimate_norm2(a, 30, base.seed);
      const double spread = na.ritz_max - na.ritz_min;
      hs::SolverConfig legacy = base;
      legacy.target.mode = hs::TargetMode::Nearest;
      legacy.target.k = cfg.target.k;
      legacy.target.sigma = smallest ? na.ritz_min - 0.05 * spread : na.ritz_max + 0.05 * spread;
      emit(run_one(a, b, legacy, mode + " legacy-nearest-beyond-end"));
    }
  } else if (mode == "nearest") {
    if (args.size() < 3) usage();
    hs::SolverConfig cfg = base;
    cfg.target.mode = hs::TargetMode::Nearest;
    cfg.target.sigma = std::stod(args[1]);
    cfg.target.k = std::stoll(args[2]);
    if (args.size() > 3) cfg.zero_filter_tol = std::stod(args[3]);
    emit(run_one(a, b, cfg, "nearest sigma=" + args[1] + " k=" + args[2] + (args.size() > 3 ? " ztol=" + args[3] : "")));
  } else {
    usage();
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string a_path, b_path;
  int threads = 2;
  std::vector<std::string> rest;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--A" && i + 1 < argc) {
      a_path = argv[++i];
    } else if (s == "--B" && i + 1 < argc) {
      b_path = argv[++i];
    } else if (s == "--threads" && i + 1 < argc) {
      threads = std::atoi(argv[++i]);
    } else {
      rest.push_back(s);
    }
  }
  if (a_path.empty() || rest.empty()) usage();
  try {
    auto la = hs::load_matrix(a_path);
    std::variant<std::monostate, hs::LoadedMatrix> lb;
    if (!b_path.empty()) lb = hs::load_matrix(b_path);
    const bool cpx = la.is_complex() || (lb.index() == 1 && std::get<1>(lb).is_complex());
    if (cpx) {
      auto a = la.is_complex() ? std::get<1>(la.matrix) : hs::to_complex(std::get<0>(la.matrix));
      hs::CsrMatrix<hs::cplx> b;
      if (lb.index() == 1) {
        auto& m = std::get<1>(lb);
        b = m.is_complex() ? std::get<1>(m.matrix) : hs::to_complex(std::get<0>(m.matrix));
      }
      sweep<hs::cplx>(a, lb.index() == 1 ? &b : nullptr, threads, rest);
    } else {
      const auto& a = std::get<0>(la.matrix);
      const hs::CsrMatrix<double>* b = lb.index() == 1 ? &std::get<0>(std::get<1>(lb).matrix) : nullptr;
      sweep<double>(a, b, threads, rest);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_slicing_sweep: %s\n", e.what());
    return 2;
  }
  return 0;
}
