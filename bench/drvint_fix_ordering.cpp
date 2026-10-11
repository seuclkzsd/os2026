// Nested-dissection orderings for the adaptive analysis (driver fix of the drvint module): analysis
// time, fill and run-to-run stability of AMD, SCOTCH and METIS analyses of K(sigma), next to the
// driver's ordering decision (driver::decide_ordering on the AMD probe, nearest target with
// certificate: 3 (+1 with B) factorizations, max(40, 3k) solves - solver.cpp).
//
//   bench_drvint_fix_ordering --case DIR [--threads t] [--reps r] [--orderings 0,3,5] [--factorize]
//
// One JSON line per case: per ordering (ICNTL(7)) the median analysis seconds, every repetition's
// factor-entry estimate (an ordering whose estimates differ between the repetitions of one process
// is not reproducible), and with --factorize the median factorization seconds at sigma. The
// orderings are interleaved within each repetition. Run through tools/run/guard.py, e.g.
//   python3 tools/run/guard.py --skip-busy-check --cores 24,25,26,27 --mem-gb 6 --
//     build/release/bench_drvint_fix_ordering --case data/cases/syn-mem27-46 --threads 4 --reps 3
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
#include <sstream>
#include <string>
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
  std::vector<int> orderings{0, 3, 5};
  int threads = 4;
  int reps = 3;
  bool factorize = false;
};

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

hs::FactorOptions options_for(int ordering, int threads) {
  hs::FactorOptions o;
  o.ordering = ordering;
  o.threads = threads;
  o.deterministic = false;  // the driver default (SolverConfig::deterministic = false)
  return o;
}

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, double sigma, index_t k) {
  const hs::ShiftedPencil<T> pencil(a, b);
  const auto in = hs::driver::mumps_input_size(pencil);
  json out = {{"case", args.case_dir.filename().string()}, {"n", a.nrows}, {"order", in.order},
              {"entries", in.entries}, {"complex", hs::is_complex_v<T>}, {"threads", args.threads}};
  std::map<int, std::vector<double>> t_an, t_fac;
  std::map<int, std::vector<std::int64_t>> entries;
  std::map<int, hs::FactorEstimate> est;
  for (int r = 0; r < args.reps; ++r) {
    for (const int ord : args.orderings) {
      auto f = hs::make_mumps_factorization<T>(options_for(ord, args.threads));
      hs::Timer t;
      f->analyze(pencil, 1.0, -sigma);
      t_an[ord].push_back(t.seconds());
      est[ord] = f->estimate();
      entries[ord].push_back(est[ord].factor_entries);
      if (args.factorize) {
        hs::Timer tf;
        if (!f->factorize(1.0, -sigma)) HS_THROW(Singular, "K(sigma) singular");
        t_fac[ord].push_back(tf.seconds());
      }
    }
  }
  if (est.count(0)) {
    const auto d = hs::driver::decide_ordering(est[0], in, static_cast<double>(pencil.nnz()), args.threads,
                                               3 + (b ? 1 : 0), std::max<index_t>(40, 3 * k));
    out["decision"] = {{"use_nd", d.use_nd}, {"saving_s", d.saving_s}, {"amd_analysis_model_s", d.amd_analysis_s},
                       {"nd_cost_s", d.nd_cost_s}, {"fill_ratio", d.fill_ratio}};
  }
  json per = json::object();
  for (const int ord : args.orderings) {
    json o = {{"analyze_s", median(t_an[ord])}, {"analyze_s_all", t_an[ord]}, {"factor_entries_all", entries[ord]},
              {"flops", est[ord].flops}, {"incore_mb", est[ord].incore_mb},
              {"stable", std::adjacent_find(entries[ord].begin(), entries[ord].end(), std::not_equal_to<>()) ==
                             entries[ord].end()}};
    if (args.factorize) o["factorize_s"] = median(t_fac[ord]);
    per[std::to_string(ord)] = o;
  }
  out["orderings"] = per;
  std::cout << out.dump() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need = [&](int& j) -> std::string {
      if (j + 1 >= argc) throw std::runtime_error("missing value for " + key);
      return argv[++j];
    };
    if (key == "--case") {
      args.case_dir = need(i);
    } else if (key == "--orderings") {
      args.orderings.clear();
      std::stringstream ss(need(i));
      std::string e;
      while (std::getline(ss, e, ',')) args.orderings.push_back(std::stoi(e));
    } else if (key == "--threads") {
      args.threads = std::stoi(need(i));
    } else if (key == "--reps") {
      args.reps = std::stoi(need(i));
    } else if (key == "--factorize") {
      args.factorize = true;
    } else {
      std::cerr << "unknown argument " << key << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty()) {
    std::cerr << "usage: bench_drvint_fix_ordering --case DIR [--threads t] [--reps r] [--orderings 0,3,5] "
                 "[--factorize]\n";
    return 2;
  }
  try {
    std::ifstream f(args.case_dir / "config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const hs::SolverConfig cfg = hs::parse_config(ss.str());
    const double sigma = cfg.target.mode == hs::TargetMode::Nearest ? cfg.target.sigma : 0.0;
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b, sigma, cfg.target.k);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, a, b ? &*b : nullptr, sigma, cfg.target.k);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
