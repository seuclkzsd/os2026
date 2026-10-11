// Cost components of a fill-reducing ordering for the driver's adaptive choice (module anaint, P5):
// what one solve pays per ordering - the analysis (with the ordering itself), the numeric
// factorization, one refactorization (a certificate count on the main object), an InertiaCounter
// (its analysis on the reused order, one count) and the solves with 1 and p right-hand sides
// (block Krylov: HEP p = 4, GHIEP p = 2) - with the FactorOptions of the driver
// (driver::factor_options_from: L0-OpenMP, threads, memory cap).
//
//   bench_anaint_orderings --case DIR --ordering O [--threads 4] [--solves 5] [--block-solves 3]
//       [--blocks 2,4] [--max-mb 11000] [--max-gflops 3000] [--nd-serial 0|1]
//
// O: FactorOptions::ordering (0 AMD, 3 SCOTCH, 4 PORD, 5 METIS, 10 hsolve nested dissection).
// DIR holds A.npz (and B.npz) and config.json (shift = target.sigma). The numeric steps are skipped
// (skipped_numeric = true) when the analysis estimate exceeds --max-mb in-core MB or --max-gflops
// GFlop (AMD / AMF on the large 3-D cases: minutes per factorization). With --nd-serial 1
// and O = 10 the ordering is also computed alone on 1 and on `threads` threads (compute_ordering),
// which separates its serial work W from the parallel run's time (Brent: t_p <= W / p + C).
// Output: one JSON line - order / entries (mumps_input_size), analyze_s, ordering_s, est_flops,
// est_entries, est_incore_mb, factorize_s, factor_entries, factor_mb, refactorize_s,
// counter_setup_s, counter_count_s, counter_incore_mb, solve1_s, solve_p_s {p: s} (medians per
// call), neg / valid of the factorizations. One process per measurement, orderings interleaved,
// through tools/run/guard.py; the commands behind the numbers quoted in src/driver/steps.cpp:
//   for rep in 0 1 2; do for c in <cases>; do for o in 0 10 4 3; do   (order rotated by rep)
//     python3 tools/run/guard.py --skip-busy-check --cores 28-31 --mem-gb 14 --timeout 900 --
//       build/release/bench_anaint_orderings --case data/cases/$c --ordering $o --threads 4 --nd-serial 1
//   done; done; done
// (gpu02, 2x Xeon Silver 4314; the driver's choice per case comes from bench_anaint_decide).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  fs::path case_dir;
  int ordering = 0;
  int threads = 4;
  int solves = 5;
  int block_solves = 3;
  std::vector<int> blocks{2, 4};
  double max_mb = 11000.0;
  double max_gflops = 3000.0;
  bool nd_serial = false;
};

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <class T>
void fill(std::vector<T>& x, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (auto& xi : x) {
    if constexpr (hs::is_complex_v<T>)
      xi = T(u(rng), u(rng));
    else
      xi = u(rng);
  }
}

template <class T>
int run(const Args& args, const hs::SolverConfig& cfg_in, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  hs::SolverConfig cfg = cfg_in;
  cfg.threads = args.threads;
  cfg.ordering = args.ordering;
  omp_set_num_threads(args.threads);
  const double sigma = cfg.target.sigma;
  const hs::FactorOptions fo = hs::driver::factor_options_from(cfg, 0.0);
  const auto in = hs::driver::mumps_input_size(pencil);
  json out = {{"case", args.case_dir.filename().string()},
              {"n", pencil.order()},
              {"order", in.order},
              {"entries", in.entries},
              {"pencil_nnz", pencil.nnz()},
              {"nnz_a", a.nnz()},
              {"nnz_b", b ? b->nnz() : 0},
              {"complex", hs::is_complex_v<T>},
              {"has_b", b != nullptr},
              {"k", cfg.target.k},
              {"ordering", args.ordering},
              {"threads", args.threads},
              {"sigma", sigma}};

  if (args.nd_serial && args.ordering == hs::kOrderingNestedDissection) {
    for (const int th : {1, args.threads}) {
      hs::OrderingOptions oo;
      oo.threads = th;
      hs::OrderingStats st;
      hs::Timer t;
      const auto perm = hs::compute_ordering(pencil, oo, &st);
      out["nd_ordering_s_" + std::to_string(th)] = t.seconds();
      out["nd_vertices"] = st.vertices;
      out["nd_edges"] = st.edges;
      out["nd_separators"] = st.separators;
      out["nd_top_separator"] = st.top_separator;
    }
  }

  auto f = hs::make_mumps_factorization<T>(fo);
  hs::Timer t;
  f->analyze(pencil, 1.0, -sigma);
  out["analyze_s"] = t.seconds();
  out["ordering_s"] = f->stats().ordering_s;
  const hs::FactorEstimate est = f->estimate();
  out["est_flops"] = est.flops;
  out["est_entries"] = est.factor_entries;
  out["est_incore_mb"] = est.incore_mb;
  if (est.incore_mb > args.max_mb || est.flops > args.max_gflops * 1e9) {
    out["skipped_numeric"] = true;
    std::cout << out.dump() << std::endl;
    return 0;
  }
  out["skipped_numeric"] = false;

  // Main factorization at sigma (one perturbation if K(sigma) is flagged singular, as the driver).
  double s = sigma;
  t.reset();
  bool ok = f->factorize(1.0, -s);
  if (!ok) {
    s = sigma + 1e-8 * std::max(1.0, std::abs(sigma));
    ok = f->factorize(1.0, -s);
  }
  out["factorize_s"] = t.seconds();
  out["sigma_used"] = s;
  out["factor_entries"] = f->stats().factor_entries;
  out["factor_mb"] = f->stats().factor_mb_used;
  json negs = json::array({f->inertia().neg}), valid = json::array({ok});
  if (ok) {
    const index_t n = pencil.order();
    std::mt19937_64 rng(1);
    std::vector<double> s1;
    std::vector<T> x(static_cast<std::size_t>(n));
    for (int k = 0; k < args.solves; ++k) {
      fill(x, rng);
      t.reset();
      f->solve(x.data(), 1, n);
      s1.push_back(t.seconds());
    }
    out["solve1_s"] = median(s1);
    json sp = json::object();
    for (const int p : args.blocks) {
      std::vector<double> tp;
      std::vector<T> xb(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
      for (int k = 0; k < args.block_solves; ++k) {
        fill(xb, rng);
        t.reset();
        f->solve(xb.data(), p, n);
        tp.push_back(t.seconds());
      }
      sp[std::to_string(p)] = median(tp);
    }
    out["solve_p_s"] = sp;
  }
  // A certificate count on the main object: refactorization at a nearby shift.
  const double delta = 1e-3 * std::max(1.0, std::abs(sigma));
  t.reset();
  valid.push_back(f->factorize(1.0, -(sigma + delta)));
  out["refactorize_s"] = t.seconds();
  negs.push_back(f->inertia().neg);
  // The same count on an InertiaCounter (main factors released first, as the nearest certificate).
  f->release_factors();
  t.reset();
  hs::InertiaCounter<T> counter(*f, fo);
  out["counter_setup_s"] = t.seconds();
  out["counter_incore_mb"] = counter.estimate().incore_mb;
  t.reset();
  const hs::Inertia ci = counter.count(1.0, -(sigma - delta));
  out["counter_count_s"] = t.seconds();
  negs.push_back(ci.neg);
  valid.push_back(ci.valid);
  out["neg"] = negs;
  out["valid"] = valid;
  std::cout << out.dump() << std::endl;
  return 0;
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string e;
  while (std::getline(ss, e, ',')) v.push_back(std::stoi(e));
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (k == "--case")
        args.case_dir = v;
      else if (k == "--ordering")
        args.ordering = std::stoi(v);
      else if (k == "--threads")
        args.threads = std::stoi(v);
      else if (k == "--solves")
        args.solves = std::stoi(v);
      else if (k == "--block-solves")
        args.block_solves = std::stoi(v);
      else if (k == "--blocks")
        args.blocks = parse_list(v);
      else if (k == "--max-mb")
        args.max_mb = std::stod(v);
      else if (k == "--max-gflops")
        args.max_gflops = std::stod(v);
      else if (k == "--nd-serial")
        args.nd_serial = std::stoi(v) != 0;
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty()) {
      std::cerr << "usage: bench_anaint_orderings --case DIR --ordering O [...]\n";
      return 2;
    }
    std::ifstream cf(args.case_dir / "config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    const hs::SolverConfig cfg = hs::parse_config(ss.str());
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, cfg, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, cfg, a, b ? &*b : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
