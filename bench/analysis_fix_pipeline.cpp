// End-to-end cost of a fill-reducing ordering for the driver (module analysis, P4 review fix):
// not analysis + one factorization, but what the driver pays with it - the analysis, the main
// factorization, the two certificate inertia counts at the window endpoints (refactorizations
// of the same analysed object, or an InertiaCounter reusing its pivot order) and the solves of
// the eigensolver, all with the factor entries this ordering produced.
//
//   bench_analysis_fix_pipeline --case DIR --ordering O [--threads t] [--ordering-threads t]
//       [--sigma s] [--delta d] [--refactorizations F] [--counter 0|1] [--solves S]
//       [--block-solves B] [--block p] [--l0 0|1]
//
// O: FactorOptions::ordering (0 AMD, 3 SCOTCH, 5 METIS, 7 auto, 10 hsolve nested dissection with
// --ordering-threads threads, 0 = --threads). DIR holds A.npz (and B.npz) and config.json (sigma
// = target.sigma unless --sigma). Steps, all through hs::make_mumps_factorization with the
// driver's FactorOptions (threads, deterministic = !l0; l0 = 1 is the driver default):
//   analyze at sigma; factorize at sigma (main); S solves with one right-hand side and B solves
//   with p (block Krylov) on the main factors; F refactorizations at sigma -/+ delta (count_below
//   in src/driver/steps.cpp; delta default 1e-3 * max(1, |sigma|)); with --counter 1 an
//   InertiaCounter on the analysed object and its count_pair at sigma -/+ delta.
// Output: one JSON line - analyze_s (ordering_s inside), factorize_s (main), refactorize_s[],
// counter_setup_s, counter_pair_s, counter_est_flops / _entries (the counter's own analysis of the
// reused order), solve1_s / solveP_s (medians per call), est_flops, factor_entries, neg / valid
// of every factorization (all orderings must agree). Run one process per measurement, orderings
// interleaved, through tools/run/guard.py. Reproduction of the end-to-end table in
// include/hsolve/factor.hpp (cores 38,39; medians of the 3 repetitions; A+3f = analyze_s +
// factorize_s + both refactorize_s, A+f+InertiaCounter = analyze_s + factorize_s +
// counter_setup_s + counter_pair_s; AMD only on the 2-D cases - on the 3-D ones it costs 3-7x
// METIS' flops, see the ordering table):
//   for rep in 1 2 3; do for c in syn-mem27-46 yee-pec-box-1e5 ss-qa8fk-qa8fm wg-square-1e5
//       cav2d-rect-pert-1e5 wg-square-1e6; do for o in 0 3 5 10; do
//     python3 tools/run/guard.py --skip-busy-check --cores 38,39 --mem-gb 4 --timeout 900 --
//       build/release/bench_analysis_fix_pipeline --case $HS_DATA_ROOT/cases/$c --ordering $o
//       --threads 2 --counter 1
//   done; done; done
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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
  int ordering = 5;
  int threads = 0;
  int ordering_threads = 0;
  double sigma = std::numeric_limits<double>::quiet_NaN();
  double delta = 0.0;  // 0: 1e-3 * max(1, |sigma|)
  int refactorizations = 2;
  bool counter = false;
  int solves = 10;
  int block_solves = 3;
  int block = 8;
  bool l0 = true;
};

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <class T>
int run(Args a, const hs::CsrMatrix<T>& am, const hs::CsrMatrix<T>* bm) {
  const hs::ShiftedPencil<T> pencil(am, bm);
  if (a.threads <= 0) a.threads = omp_get_max_threads();
  if (std::isnan(a.sigma)) {
    std::ifstream f(a.case_dir / "config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    a.sigma = hs::parse_config(ss.str()).target.sigma;
  }
  const double delta = a.delta > 0.0 ? a.delta : 1e-3 * std::max(1.0, std::abs(a.sigma));
  hs::FactorOptions fo;
  fo.ordering = a.ordering;
  fo.ordering_threads = a.ordering_threads;
  fo.threads = a.threads;
  fo.deterministic = !a.l0;
  json out = {{"case", a.case_dir.filename().string()},
              {"n", pencil.order()},
              {"nnz_upper", pencil.nnz()},
              {"complex", hs::is_complex_v<T>},
              {"ordering", a.ordering},
              {"threads", a.threads},
              {"ordering_threads", a.ordering_threads},
              {"l0", a.l0},
              {"sigma", a.sigma},
              {"delta", delta}};
  auto f = hs::make_mumps_factorization<T>(fo);
  hs::Timer t;
  f->analyze(pencil, 1.0, -a.sigma);
  out["analyze_s"] = t.seconds();
  out["ordering_s"] = f->stats().ordering_s;
  out["est_flops"] = f->estimate().flops;
  out["est_entries"] = f->estimate().factor_entries;
  out["block_analysis"] = f->stats().block_analysis;

  // Main factorization, then the solves on it, then the certificate refactorizations.
  t.reset();
  const bool main_ok = f->factorize(1.0, -a.sigma);
  out["factorize_s"] = t.seconds();
  out["factor_entries"] = f->stats().factor_entries;
  json negs = json::array({f->inertia().neg}), valid = json::array({main_ok});
  if (main_ok) {
    const index_t n = pencil.order();
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto fill = [&](std::vector<T>& x) {
      for (auto& xi : x) {
        if constexpr (hs::is_complex_v<T>)
          xi = T(u(rng), u(rng));
        else
          xi = u(rng);
      }
    };
    std::vector<double> s1, sp;
    std::vector<T> x(static_cast<std::size_t>(n));
    for (int k = 0; k < a.solves; ++k) {
      fill(x);
      t.reset();
      f->solve(x.data(), 1, n);
      s1.push_back(t.seconds());
    }
    std::vector<T> xb(static_cast<std::size_t>(n) * a.block);
    for (int k = 0; k < a.block_solves; ++k) {
      fill(xb);
      t.reset();
      f->solve(xb.data(), a.block, n);
      sp.push_back(t.seconds());
    }
    out["solve1_s"] = median(s1);
    out["block"] = a.block;
    out["solveP_s"] = median(sp);
  }
  json refac = json::array();
  for (int k = 0; k < a.refactorizations; ++k) {
    t.reset();
    const bool ok = f->factorize(1.0, -(a.sigma + (k % 2 ? delta : -delta)));
    refac.push_back(t.seconds());
    negs.push_back(f->inertia().neg);
    valid.push_back(ok);
  }
  out["refactorize_s"] = refac;
  out["neg"] = negs;
  out["valid"] = valid;

  if (a.counter) {
    t.reset();
    hs::InertiaCounter<T> counter(*f, fo);
    out["counter_setup_s"] = t.seconds();
    out["counter_est_flops"] = counter.estimate().flops;
    out["counter_est_entries"] = counter.estimate().factor_entries;
    t.reset();
    const auto [lo, hi] = counter.count_pair(1.0, -(a.sigma - delta), 1.0, -(a.sigma + delta));
    out["counter_pair_s"] = t.seconds();
    out["counter_neg"] = {lo.neg, hi.neg};
    out["counter_valid"] = {lo.valid, hi.valid};
  }
  std::cout << out.dump() << std::endl;
  return 0;
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
      else if (k == "--ordering-threads")
        args.ordering_threads = std::stoi(v);
      else if (k == "--sigma")
        args.sigma = std::stod(v);
      else if (k == "--delta")
        args.delta = std::stod(v);
      else if (k == "--refactorizations")
        args.refactorizations = std::stoi(v);
      else if (k == "--counter")
        args.counter = std::stoi(v) != 0;
      else if (k == "--solves")
        args.solves = std::stoi(v);
      else if (k == "--block-solves")
        args.block_solves = std::stoi(v);
      else if (k == "--block")
        args.block = std::stoi(v);
      else if (k == "--l0")
        args.l0 = std::stoi(v) != 0;
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty()) {
      std::cerr << "usage: bench_analysis_fix_pipeline --case DIR --ordering O [...]\n";
      return 2;
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
