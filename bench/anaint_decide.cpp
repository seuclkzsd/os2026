// The driver's ordering choice on a data case (module anaint, P5), without solving: the planned
// work (driver::plan_work for the problem type known before the analysis), the analyses and
// prices of driver::adaptive_analysis (its "ordering" log event) and, for comparison, the P4 rule
// (driver::decide_ordering on the same AMD estimate). The decisions depend on the analysis
// estimates and the thread count only, so the choice at thread counts the machine does not have
// (--threads 4,8,16) is the one the driver would make there; wall times are those of this machine.
//
//   bench_anaint_decide --case DIR [--threads 4[,8,16]]
//
// DIR holds A.npz (and B.npz) and config.json. Output: one JSON line per thread count - type,
// work (factorizations, counts, solve calls, block), choice (FactorOptions::ordering), candidates
// ("ordering:a|p/score s[/analysis wall s]", a = analysed, p = predicted), p4_choice (0 AMD,
// 3 SCOTCH). The per-case choices quoted in src/driver/steps.cpp come from
//   python3 tools/run/guard.py --skip-busy-check --cores 28-31 --mem-gb 14 --timeout 900 --
//     build/release/bench_anaint_decide --case data/cases/$c --threads 4,8,16
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/internal.hpp"
#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

const char* type_name(hs::ProblemType t) {
  switch (t) {
    case hs::ProblemType::Standard: return "standard";
    case hs::ProblemType::GeneralizedDefinite: return "generalized_definite";
    case hs::ProblemType::GeneralizedIndefinite: return "generalized_indefinite";
    case hs::ProblemType::Auto: break;
  }
  return "undecided";
}

template <class T>
int run(const fs::path& case_dir, const std::vector<int>& threads, const hs::SolverConfig& cfg_in,
        const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  // The problem type and scale as hs::solve knows them before its analysis.
  const auto na = hs::driver::estimate_norm2(a, 30, cfg_in.seed);
  hs::driver::NormEstimate nb;
  hs::ProblemType type = b ? hs::ProblemType::Auto : hs::ProblemType::Standard;
  if (b) {
    nb = hs::driver::estimate_norm2(*b, 30, cfg_in.seed);
    const auto d = hs::driver::test_b_definiteness(*b, nb);
    if (d.verdict == hs::driver::Definiteness::Positive) type = hs::ProblemType::GeneralizedDefinite;
    if (d.verdict == hs::driver::Definiteness::NotPositive) type = hs::ProblemType::GeneralizedIndefinite;
  }
  const double scale = std::max(b ? na.norm2 / nb.norm2 : na.norm2, 1e-300);
  for (const int th : threads) {
    hs::SolverConfig cfg = cfg_in;
    cfg.threads = th;
    omp_set_num_threads(th);
    const auto work = hs::driver::plan_work(cfg, type, pencil.order(), scale);
    const fs::path log_path = fs::temp_directory_path() / ("anaint_decide_" + std::to_string(::getpid()) + ".jsonl");
    int choice = -1;
    json ev;
    {
      hs::Logger log(0, log_path);
      hs::SolveReport rep;
      hs::driver::PhaseClock clock;
      hs::driver::Ctx c{cfg, &log, rep, clock};
      auto fac = hs::driver::adaptive_analysis<T>(c, hs::driver::factor_options_from(cfg, 0.0), pencil,
                                                  cfg.target.sigma, work);
      choice = fac->options().ordering;
    }
    {
      std::ifstream in(log_path);
      std::string line;
      while (std::getline(in, line)) {
        const auto j = json::parse(line);
        if (j.value("event", "") == "ordering") ev = j;
      }
    }
    fs::remove(log_path);
    hs::FactorEstimate amd;
    amd.flops = ev.value("amd_flops", 0.0);
    amd.factor_entries = ev.value("amd_entries", std::int64_t{0});
    const index_t k = cfg.target.mode == hs::TargetMode::Interval ? 32 : cfg.target.k;
    const auto p4 = hs::driver::decide_ordering(amd, hs::driver::mumps_input_size(pencil),
                                                static_cast<double>(pencil.nnz()), th,
                                                1 + (cfg.inertia_check != hs::Toggle::Off ? 2 : 0) + (b ? 1 : 0),
                                                std::max<index_t>(40, 3 * k));
    json out = {{"case", case_dir.filename().string()},
                {"threads", th},
                {"n", pencil.order()},
                {"complex", hs::is_complex_v<T>},
                {"type", type_name(type)},
                {"work", {{"factorizations", work.factorizations}, {"counts", work.counts},
                          {"solve_calls", work.solve_calls}, {"block", work.block}}},
                {"choice", choice},
                {"fill_ratio", ev.value("fill_ratio", 0.0)},
                {"candidates", ev.value("candidates", std::string())},
                {"p4_choice", p4.use_nd ? hs::driver::nd_ordering(cfg) : 0}};
    std::cout << out.dump() << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  fs::path case_dir;
  std::vector<int> threads{4};
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (k == "--case") {
        case_dir = v;
      } else if (k == "--threads") {
        threads.clear();
        std::stringstream ss(v);
        std::string e;
        while (std::getline(ss, e, ',')) threads.push_back(std::stoi(e));
      } else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (case_dir.empty()) {
      std::cerr << "usage: bench_anaint_decide --case DIR [--threads 4,8,16]\n";
      return 2;
    }
    std::ifstream cf(case_dir / "config.json");
    std::stringstream ss;
    ss << cf.rdbuf();
    const hs::SolverConfig cfg = hs::parse_config(ss.str());
    auto la = hs::load_matrix(case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(case_dir / "B.npz")) lb = hs::load_matrix(case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(case_dir, threads, cfg, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(std::move(m.matrix)));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(case_dir, threads, cfg, a, b ? &*b : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
