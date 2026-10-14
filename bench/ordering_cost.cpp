// What a fill-reducing ordering costs a whole solve (P9 "ordering"): the analysis (with the
// ordering itself), the MUMPS estimates (flops, factor entries, in-core memory), the main numeric
// factorization, the certificate's inertia-only counts (an InertiaCounter on the main order after
// the main factors are released - the nearest certificate's path on the high-fill cases) and the
// solves with 1 and p right-hand sides, all with the driver's FactorOptions
// (driver::factor_options_from: L0-OpenMP, threads).
//
//   bench_ordering_cost --case DIR --ordering O [--threads 16] [--numeric 0|1] [--counts 1]
//       [--solves 5] [--block-solves 3] [--blocks 4,8] [--max-mb 27000] [--perm-in FILE]
//       [--perm-out FILE] [--nd-leaf L] [--nd-imbalance F] [--nd-coarsen C] [--nd-tries T]
//       [--nd-passes P] [--nd-runs R] [--nd-finalists F] [--nd-compress 0|1]
//       [--nd-top-depth D] [--nd-top-size S] [--nd-top-runs R] [--nd-top-finalists F] [--nd-seed S] [--nd-embedding auto|compressed|full] [--nd-profile 0|1]
//
// O: FactorOptions::ordering - 0 AMD, 2 AMF, 3 SCOTCH, 4 PORD, 5 METIS, 6 QAMD (MUMPS, serial,
// inside the analysis), 10 hsolve's nested dissection (hs::compute_ordering with --threads
// threads and the --nd-* OrderingOptions, timed apart, then handed to MUMPS as PERM_IN), or
// 11 (real T) the same nested dissection through its internal interface
// (hs::ordering::nested_dissection; --nd-mode M sets the FM variant, 0 two-sided, 1 one-sided, 2
// both; profile as with --nd-profile 1), or -1 with --perm-in FILE: a pivot order from a .npy file (int32, 1-based pivot positions, as
// --perm-out writes them: the analysed object's SYM_PERM). DIR holds A.npz (and B.npz) and
// config.json (sigma = target.sigma). The numeric steps are skipped when --numeric 0 or the
// in-core estimate exceeds --max-mb MB. --nd-profile 1 (real T) first runs the nested dissection
// alone (hs::ordering::nested_dissection, --nd-* options) and reports its DissectionStats: wall
// time, work, critical path, and where the work went (separator stages, extraction, leaves).
// Output: one JSON line - order / entries (mumps_input_size), analyze_s (ordering included),
// ordering_s (hsolve ND only: compute_ordering), given_s (= analyze_s - ordering_s for ND and
// --perm-in), nd_* (OrderingStats), est_flops, est_entries, est_incore_mb, factorize_s,
// factor_entries, factor_mb, counter_setup_s, count_s[] (one per count), solve1_s,
// solve_p_s {p: s} (medians per call), neg / valid of the factorizations (must agree between
// orderings). One process per measurement, orderings interleaved, through tools/run/guard.py in
// the measurement slot; for solve times matching the CLI (which links the BLAS hook) preload
// build/release/libhsolve_blashook.so. Commands behind results/p9_ordering.md: see there.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include "factor/ordering_detail.hpp"
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
  int ordering = 10;
  int threads = 16;
  bool numeric = true;
  int counts = 1;
  int solves = 5;
  int block_solves = 3;
  std::vector<int> blocks{4, 8};
  double max_mb = 27000.0;
  fs::path perm_in, perm_out;
  hs::OrderingOptions nd;
  bool nd_profile = false;
  int nd_mode = -1;  // SeparatorParams::refine_mode for --ordering 11 (-1: default)
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

// --nd-profile 1 (real T): hs::ordering::nested_dissection on the pencil's graph with the --nd-*
// options, timed apart, with its DissectionStats (work, critical path, where the work went).
template <class T>
json nd_profile(const hs::ShiftedPencil<T>& pencil, const hs::OrderingOptions& o, int threads, int mode = -1,
                std::vector<int>* perm = nullptr) {
  if constexpr (hs::is_complex_v<T>) {
    if (perm) HS_THROW(Unsupported, "--ordering 11 needs a real pencil");
    return json{{"skipped", "complex"}};
  } else {
    hs::Timer t;
    hs::ordering::Graph g = hs::ordering::graph_from_upper(static_cast<std::int32_t>(pencil.order()),
                                                           pencil.rowptr().data(), pencil.colind().data());
    hs::ordering::Graph cg;
    std::vector<std::int32_t> first, members;
    const bool compressed =
        o.compress && hs::ordering::compress_graph(g, hs::ordering::kCompressionRatio, threads, cg, first, members);
    if (compressed) g = std::move(cg);
    const double graph_s = t.seconds();
    hs::ordering::DissectionParams p;
    p.separator.imbalance = o.imbalance;
    p.separator.coarsen_to = static_cast<std::int32_t>(o.coarsen_to);
    p.separator.init_tries = o.init_tries;
    p.separator.refine_passes = o.refine_passes;
    p.separator.runs = o.separator_runs;
    p.separator.finalists = o.separator_finalists;
    p.top_depth = o.top_depth;
    p.top_size = o.top_size > 0 ? o.top_size : std::numeric_limits<std::int64_t>::max();
    p.top_runs = o.top_runs;
    p.top_finalists = o.top_finalists;
    p.leaf_size = static_cast<std::int32_t>(o.leaf_size);
    p.seed = o.seed;
    if (mode >= 0) p.separator.refine_mode = mode;
    hs::ordering::DissectionStats st;
    t.reset();
    const auto order = hs::ordering::nested_dissection(std::move(g), p, threads, &st);
    const double seconds = t.seconds();
    if (perm) {
      perm->assign(static_cast<std::size_t>(pencil.order()), 0);
      int pos = 0;
      for (const std::int32_t q : order) {
        if (!compressed) {
          (*perm)[q] = ++pos;
          continue;
        }
        for (std::int32_t m = first[q]; m < first[q + 1]; ++m) (*perm)[members[m]] = ++pos;
      }
    }
    return json{{"graph_s", graph_s},
                {"ordered_vertices", static_cast<std::int64_t>(order.size())},
                {"seconds", seconds},
                {"work_s", st.work_s},
                {"critical_path_s", st.critical_path_s},
                {"top_separator_s", st.top_separator_s},
                {"sep_coarsen_s", st.separator.coarsen_s},
                {"sep_runs_s", st.separator.runs_s},
                {"sep_uncoarsen_s", st.separator.uncoarsen_s},
                {"sep_small_s", st.separator.small_s},
                {"extract_s", st.extract_s},
                {"leaf_s", st.leaf_s},
                {"separators", st.separators},
                {"leaves", st.leaves},
                {"depth_separator", st.depth_separator}};
  }
}

template <class T>
int run(const Args& args, const hs::SolverConfig& cfg_in, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  hs::SolverConfig cfg = cfg_in;
  cfg.threads = args.threads;
  cfg.ordering = args.ordering < 0 || args.ordering == 11 ? 0 : args.ordering;
  omp_set_num_threads(args.threads);
  const double sigma = cfg.target.sigma;
  hs::FactorOptions fo = hs::driver::factor_options_from(cfg, 0.0);
  const auto in = hs::driver::mumps_input_size(pencil);
  json out = {{"case", args.case_dir.filename().string()},
              {"n", pencil.order()},
              {"order", in.order},
              {"entries", in.entries},
              {"pencil_nnz", pencil.nnz()},
              {"complex", hs::is_complex_v<T>},
              {"has_b", b != nullptr},
              {"k", cfg.target.k},
              {"ordering", args.ordering},
              {"threads", args.threads},
              {"sigma", sigma}};

  if (args.nd_profile && args.ordering != 11) out["nd_profile"] = nd_profile(pencil, args.nd, args.threads);
  auto f = hs::make_mumps_factorization<T>(fo);
  double ordering_s = 0.0;
  hs::Timer t;
  if (args.ordering == hs::kOrderingNestedDissection) {
    hs::OrderingOptions oo = args.nd;
    oo.threads = args.threads;
    hs::OrderingStats st;
    std::vector<int> perm = hs::compute_ordering(pencil, oo, &st);
    ordering_s = t.seconds();
    out["nd_vertices"] = st.vertices;
    out["nd_edges"] = st.edges;
    out["nd_separators"] = st.separators;
    out["nd_leaves"] = st.leaves;
    out["nd_top_separator"] = st.top_separator;
    out["nd_ordered_vertices"] = st.ordered_vertices;
    out["nd_compressed"] = st.compressed;
    f->set_pivot_order(std::move(perm));
  } else if (args.ordering == 11) {
    std::vector<int> perm;
    out["nd_profile"] = nd_profile(pencil, args.nd, args.threads, args.nd_mode, &perm);
    ordering_s = t.seconds();
    f->set_pivot_order(std::move(perm));
  } else if (args.ordering < 0) {
    if (args.perm_in.empty()) HS_THROW(InvalidInput, "--ordering -1 needs --perm-in");
    std::vector<int> perm = hs::npy_to_vector<std::int32_t>(hs::read_npy(args.perm_in));
    ordering_s = t.seconds();
    f->set_pivot_order(std::move(perm));
  }
  f->analyze(pencil, 1.0, -sigma);
  const double analyze_s = t.seconds();
  out["analyze_s"] = analyze_s;
  out["ordering_s"] = ordering_s;
  out["given_s"] = analyze_s - ordering_s;
  if (!args.perm_out.empty()) hs::write_npy_vector(args.perm_out, std::vector<std::int32_t>(f->pivot_order().begin(), f->pivot_order().end()));
  const hs::FactorEstimate est = f->estimate();
  out["est_flops"] = est.flops;
  out["est_entries"] = est.factor_entries;
  out["est_incore_mb"] = est.incore_mb;
  out["est_ooc_mb"] = est.ooc_mb;
  if (!args.numeric || est.incore_mb > args.max_mb) {
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
  // Certificate counts: an InertiaCounter on the main order, main factors released first (the
  // nearest certificate's path where counter_pays), at sigma -/+ delta alternately.
  if (args.counts > 0) {
    f->release_factors();
    t.reset();
    hs::InertiaCounter<T> counter(*f, fo);
    out["counter_setup_s"] = t.seconds();
    out["counter_incore_mb"] = counter.estimate().incore_mb;
    const double delta = 1e-3 * std::max(1.0, std::abs(sigma));
    json cs = json::array();
    for (int c = 0; c < args.counts; ++c) {
      t.reset();
      const hs::Inertia ci = counter.count(1.0, -(sigma + (c % 2 ? delta : -delta)));
      cs.push_back(t.seconds());
      negs.push_back(ci.neg);
      valid.push_back(ci.valid);
    }
    out["count_s"] = cs;
  }
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
      else if (k == "--numeric")
        args.numeric = std::stoi(v) != 0;
      else if (k == "--counts")
        args.counts = std::stoi(v);
      else if (k == "--solves")
        args.solves = std::stoi(v);
      else if (k == "--block-solves")
        args.block_solves = std::stoi(v);
      else if (k == "--blocks")
        args.blocks = parse_list(v);
      else if (k == "--max-mb")
        args.max_mb = std::stod(v);
      else if (k == "--perm-in")
        args.perm_in = v;
      else if (k == "--perm-out")
        args.perm_out = v;
      else if (k == "--nd-leaf")
        args.nd.leaf_size = std::stoll(v);
      else if (k == "--nd-imbalance")
        args.nd.imbalance = std::stod(v);
      else if (k == "--nd-coarsen")
        args.nd.coarsen_to = std::stoll(v);
      else if (k == "--nd-tries")
        args.nd.init_tries = std::stoi(v);
      else if (k == "--nd-passes")
        args.nd.refine_passes = std::stoi(v);
      else if (k == "--nd-compress")
        args.nd.compress = std::stoi(v) != 0;
      else if (k == "--nd-top-depth")
        args.nd.top_depth = std::stoi(v);
      else if (k == "--nd-top-size")
        args.nd.top_size = std::stoll(v);
      else if (k == "--nd-top-runs")
        args.nd.top_runs = std::stoi(v);
      else if (k == "--nd-top-finalists")
        args.nd.top_finalists = std::stoi(v);
      else if (k == "--nd-runs")
        args.nd.separator_runs = std::stoi(v);
      else if (k == "--nd-finalists")
        args.nd.separator_finalists = std::stoi(v);
      else if (k == "--nd-seed")
        args.nd.seed = std::stoull(v);
      else if (k == "--nd-mode")
        args.nd_mode = std::stoi(v);
      else if (k == "--nd-profile")
        args.nd_profile = std::stoi(v) != 0;
      else if (k == "--nd-embedding")
        args.nd.embedding = v == "compressed" ? hs::OrderingOptions::Embedding::Compressed
                            : v == "full"     ? hs::OrderingOptions::Embedding::Full
                                              : hs::OrderingOptions::Embedding::Auto;
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty()) {
      std::cerr << "usage: bench_ordering_cost --case DIR --ordering O [...]\n";
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
