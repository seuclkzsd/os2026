// Measurements behind the ordering/analysis options of src/factor (module analysis, P4): what
// MUMPS' symbolic phase (JOB=1) costs with each fill-reducing ordering, how it splits into the
// ordering itself and the rest (symbolic factorization, tree, mapping, estimates), and what the
// resulting factors cost in the numeric factorization - an ordering that is cheap to compute but
// doubles the factorization is no gain.
//
//   bench_analysis_orderings --case DIR [--exp orderings|separator|dissect] [--sigma s]
//       [--threads t] [--reps r] [--l0 0|1] [--orderings LIST] [--icntl K=V[,K=V...]]
//       [--blocks 0|1] [--split 0|1] [--factorize 0|1] [--verbose 0|1]
//       [--nd-leaf L] [--nd-coarsen C] [--nd-tries T] [--nd-passes P] [--nd-imbalance F]
//       [--nd-seed S] [--nd-embedding auto|compressed|full] [--nd-mode M] [--nd-task K]
//
// DIR holds A.npz (and B.npz) and config.json (sigma = target.sigma unless --sigma).
// --exp orderings (default): LIST is a comma-separated list of orderings: MUMPS ICNTL(7) values
// (0 AMD, 2 AMF, 3 SCOTCH, 4 PORD, 5 METIS, 6 QAMD, 7 auto) and hsolve's nested dissection
// computed outside MUMPS and passed as PERM_IN (`pnd`: hs::compute_ordering with --threads
// threads, `pnd:T` with T threads; --nd-* set OrderingOptions). Each repetition runs every
// ordering once, interleaved (o1 o2 .. o1 o2 ..), and prints one JSON line per (repetition,
// ordering):
//   analyze_s     wall time of JOB=1 (for `pnd`: ordering_s + the PERM_IN analysis)
//   ordering_s    `pnd` only: time of hs::compute_ordering
//   given_s       --split 1: JOB=1 of a second instance given the first one's SYM_PERM (PERM_IN,
//                 ICNTL(7)=1) - the non-ordering part of the analysis (graph build, symbolic
//                 factorization, tree, mapping, estimates); analyze_s - given_s ~ ordering time
//   est_entries / est_flops / est_mb   INFOG(20), RINFOG(1), INFOG(16)
//   factorize_s / entries / flops / neg / nulls   --factorize 1: numeric JOB=2 at sigma
//   infog7 / infog32 / infog24   ordering effectively used, analysis type, effective ICNTL(12)
// --icntl sets further MUMPS controls for every instance (e.g. 58=1 symbolic factorization option,
// 12=1, 6=0, 15=-2). --blocks 1 (complex T) declares the real embedding's 2x2 structure to MUMPS
// (ICNTL(15)=1 with blocks {i, n+i}), so that it orders the n-vertex graph of K (analysis by
// block). Two SCOTCH analyses in one process differ (SCOTCH's random state is not reset): run one
// repetition per process (--reps 1 in a shell loop) when the first-call behaviour matters.
// --exp separator: the top-level vertex separator of the pencil's graph for --reps seeds, with
// the multilevel trace (vertices per level, separator projected / refined per level, time in
// coarsening, initial separators and refinement). --nd-mode selects the FM variant (0 two-sided,
// 1 one-sided, 2 both).
// --exp dissect: hs::ordering::nested_dissection with --threads threads: time, work, critical
// path (with 1 thread: the wall time with unlimited threads), separator vertices per depth, and
// the MUMPS estimate of the order (real T). --nd-task sets the task granularity.
// Output: JSON lines on stdout. Run through tools/run/guard.py.
//
// Reproduction of the numbers in include/hsolve/factor.hpp (2 threads, cores 38,39, one process
// per measurement, 3 interleaved repetitions, medians):
//   for rep in 1 2 3; do for c in <cases>; do for o in 0 6 2 3 5 4 pnd:1 pnd:2; do
//     python3 tools/run/guard.py --skip-busy-check --cores 38,39 --mem-gb 4 --timeout 900 --
//       build/release/bench_analysis_orderings --case $HS_DATA_ROOT/cases/$c --threads 2
//       --reps 1 --orderings $o --split 1 --factorize 1
//   done; done; done
//   (--blocks 0 / --blocks 1 alternated for the ICNTL(15) rows; --exp dissect --threads 1 for
//   work and critical path)
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "factor/mumps_api.hpp"
#include "factor/mumps_input.hpp"
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

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Args {
  fs::path case_dir;
  double sigma = kNaN;
  int threads = 0;
  int reps = 1;
  bool l0 = true;  // ICNTL(48)=1, the driver default (SolverConfig::deterministic = false)
  std::vector<std::string> orderings{"0", "6", "3", "5", "4"};
  std::vector<std::pair<int, int>> icntl;
  bool blocks = false;
  bool split = true;
  bool factorize = true;
  bool verbose = false;
  hs::OrderingOptions nd;  // --nd-* flags (threads from `pnd:T` or --threads)
  std::string exp = "orderings";
  int nd_mode =
      -1;  // --nd-mode: SeparatorParams::refine_mode for --exp separator|dissect (-1: default)
  int nd_task = 0;  // --nd-task: DissectionParams::task_size for --exp dissect (0: default)
};

void emit(const json& j) {
  std::cout << j.dump() << std::endl;
}

// One raw dmumps instance on the pencil, with the controls of src/factor/mumps_factorization.cpp
// plus the overrides under test.
template <class T>
class Raw {
 public:
  static constexpr int kEmb = hs::is_complex_v<T> ? 2 : 1;

  Raw(const hs::ShiftedPencil<T>& pencil, const Args& a, int ordering) : in_(pencil) {
    const int stream = a.verbose ? 6 : -1;
    m_.icntl(1) = stream;
    m_.icntl(2) = -1;
    m_.icntl(3) = stream;
    m_.icntl(4) = a.verbose ? 2 : 0;
    m_.icntl(5) = 0;
    m_.icntl(7) = ordering;
    m_.icntl(8) = 7;
    m_.icntl(13) = 1;
    m_.icntl(14) = 35;
    m_.icntl(16) = a.threads;
    m_.icntl(18) = 0;
    m_.icntl(20) = 0;
    m_.icntl(21) = 0;
    m_.icntl(24) = 1;
    m_.cntl(3) = 1e-8;
    m_.icntl(48) = a.l0 ? 1 : 0;
    for (const auto& [k, v] : a.icntl) m_.icntl(k) = v;
    auto& s = m_.raw();
    s.n = in_.order();
    s.nz = 0;
    s.nnz = in_.nnz();
    s.irn = in_.irn();
    s.jcn = in_.jcn();
    s.a = in_.values();
    if (a.blocks && kEmb == 2) {
      // Blocks {i, n+i} of the real embedding (1-based), ICNTL(15) = 1.
      const int n = in_.order() / 2;
      blkptr_.resize(static_cast<std::size_t>(n) + 1);
      blkvar_.resize(static_cast<std::size_t>(2 * n));
      for (int b = 0; b <= n; ++b) blkptr_[b] = 2 * b + 1;
      for (int b = 0; b < n; ++b) {
        blkvar_[2 * b] = b + 1;
        blkvar_[2 * b + 1] = n + b + 1;
      }
      m_.icntl(15) = 1;
      s.nblk = n;
      s.blkptr = blkptr_.data();
      s.blkvar = blkvar_.data();
    }
  }

  double analyze(double alpha, double beta, const std::vector<int>* perm = nullptr) {
    in_.assemble(alpha, beta);
    if (perm) {
      perm_ = *perm;
      m_.icntl(7) = 1;
      m_.raw().perm_in = perm_.data();
    }
    hs::Timer t;
    const int info = m_.run(hs::mumps::kAnalyze);
    const double sec = t.seconds();
    m_.raw().perm_in = nullptr;
    if (info < 0)
      HS_THROW(FactorizationFailed, "analysis INFO(1)=" << info << " INFO(2)=" << m_.infog(2));
    return sec;
  }

  std::vector<int> sym_perm() {
    const auto& s = m_.raw();
    return std::vector<int>(s.sym_perm, s.sym_perm + s.n);
  }

  json estimates() {
    return {{"est_entries", hs::mumps::decode_count(m_.infog(20))},
            {"est_flops", m_.rinfog(1)},
            {"est_mb", m_.infog(16)},
            {"infog7", m_.infog(7)},
            {"infog32", m_.infog(32)},
            {"infog24", m_.infog(24)},
            {"tree_nodes", m_.infog(6)},
            {"max_front", m_.infog(5)}};
  }

  json factorize(double alpha, double beta) {
    in_.assemble(alpha, beta);
    hs::Timer t;
    const int info = m_.run(hs::mumps::kFactorize);
    const double sec = t.seconds();
    return {{"factorize_s", sec},
            {"factorize_info1", info},
            {"entries", hs::mumps::decode_count(m_.infog(9))},
            {"flops", m_.rinfog(3)},
            {"neg", m_.infog(12) / kEmb},
            {"nulls", m_.infog(28)},
            {"infog21_mb", m_.infog(21)}};
  }

  int order() const { return in_.order(); }

 private:
  hs::mumps::Instance<double> m_;
  hs::mumps::Input<T, double> in_;
  std::vector<int> perm_;
  std::vector<int> blkptr_, blkvar_;
};

// The pivot order of an hsolve ordering ("pnd" or "pnd:T") for the MUMPS matrix of the pencil.
template <class T>
std::vector<int> hsolve_pivot_order(const hs::ShiftedPencil<T>& pencil, const std::string& spec,
                                    const Args& a, double& seconds, json& info) {
  hs::OrderingOptions o = a.nd;
  o.threads = a.threads;
  const auto colon = spec.find(':');
  if (colon != std::string::npos) o.threads = std::stoi(spec.substr(colon + 1));
  hs::OrderingStats st;
  hs::Timer t;
  std::vector<int> perm = hs::compute_ordering(pencil, o, &st);
  seconds = t.seconds();
  info = {{"ordering_threads", st.threads},
          {"separators", st.separators},
          {"leaves", st.leaves},
          {"top_separator", st.top_separator},
          {"graph_vertices", st.vertices},
          {"graph_edges", st.edges},
          {"compressed", st.compressed},
          {"complex_fraction", st.complex_fraction},
          {"nd_leaf", o.leaf_size},
          {"nd_coarsen_to", o.coarsen_to},
          {"nd_tries", o.init_tries},
          {"nd_passes", o.refine_passes},
          {"nd_imbalance", o.imbalance},
          {"nd_seed", o.seed}};
  return perm;
}

template <class T>
void measure(const Args& a, const hs::ShiftedPencil<T>& pencil, const std::string& ord, int rep) {
  const double sigma = a.sigma;
  const bool own = ord.rfind("pnd", 0) == 0;
  json out = {{"rep", rep}, {"ordering", ord}};
  std::vector<int> perm;
  double t_order = 0.0;
  if (own) {
    json info;
    perm = hsolve_pivot_order(pencil, ord, a, t_order, info);
    out["ordering_s"] = t_order;
    out.update(info);
  }
  Raw<T> r(pencil, a, own ? 1 : std::stoi(ord));
  const double t_ana = r.analyze(1.0, -sigma, own ? &perm : nullptr);
  out["analyze_s"] = t_order + t_ana;
  out["mumps_analyze_s"] = t_ana;
  out.update(r.estimates());
  if (a.split && !own) {
    const std::vector<int> sp = r.sym_perm();
    Raw<T> g(pencil, a, 1);
    out["given_s"] = g.analyze(1.0, -sigma, &sp);
    const json e = g.estimates();
    out["given_est_entries"] = e["est_entries"];
    out["given_est_flops"] = e["est_flops"];
  }
  if (a.factorize) out.update(r.factorize(1.0, -sigma));
  emit(out);
}

// --exp separator: the top-level separator of the pencil's graph (n vertices) for --reps seeds,
// with the per-level trace of the multilevel scheme (sizes, projected and refined separators).
template <class T>
void exp_separator(const Args& a, const hs::ShiftedPencil<T>& pencil) {
  const hs::ordering::Graph g = hs::ordering::graph_from_upper(
      static_cast<std::int32_t>(pencil.order()), pencil.rowptr().data(), pencil.colind().data());
  hs::ordering::SeparatorParams p;
  p.imbalance = a.nd.imbalance;
  p.coarsen_to = static_cast<std::int32_t>(a.nd.coarsen_to);
  p.init_tries = a.nd.init_tries;
  p.refine_passes = a.nd.refine_passes;
  if (a.nd_mode >= 0) p.refine_mode = a.nd_mode;
  for (int rep = 0; rep < a.reps; ++rep) {
    hs::ordering::SeparatorTrace tr;
    hs::Timer t;
    const auto where =
        hs::ordering::vertex_separator(g, a.nd.seed + static_cast<std::uint64_t>(rep), p, &tr);
    const double sec = t.seconds();
    const auto chk = hs::ordering::check_separator(g, where);
    emit({{"exp", "separator"},
          {"rep", rep},
          {"seconds", sec},
          {"valid", chk.valid},
          {"separator", chk.weight[2]},
          {"part0", chk.weight[0]},
          {"part1", chk.weight[1]},
          {"level_vertices", tr.vertices},
          {"level_edges", tr.edges},
          {"projected", tr.projected},
          {"refined", tr.refined},
          {"coarsen_s", tr.coarsen_s},
          {"init_s", tr.init_s},
          {"refine_s", tr.refine_s}});
  }
}

// --exp dissect: hs::ordering::nested_dissection on the pencil's graph with --threads threads:
// time and separator vertices per dissection depth, then the MUMPS estimate of the order.
template <class T>
void exp_dissect(const Args& a, const hs::ShiftedPencil<T>& pencil) {
  hs::ordering::DissectionParams p;
  p.separator.imbalance = a.nd.imbalance;
  p.separator.coarsen_to = static_cast<std::int32_t>(a.nd.coarsen_to);
  p.separator.init_tries = a.nd.init_tries;
  p.separator.refine_passes = a.nd.refine_passes;
  if (a.nd_mode >= 0) p.separator.refine_mode = a.nd_mode;
  p.leaf_size = static_cast<std::int32_t>(a.nd.leaf_size);
  p.seed = a.nd.seed;
  if (a.nd_task > 0) p.task_size = a.nd_task;
  for (int rep = 0; rep < a.reps; ++rep) {
    hs::ordering::Graph g = hs::ordering::graph_from_upper(
        static_cast<std::int32_t>(pencil.order()), pencil.rowptr().data(), pencil.colind().data());
    hs::ordering::DissectionStats st;
    hs::Timer t;
    const auto order = hs::ordering::nested_dissection(std::move(g), p, a.threads, &st);
    const double sec = t.seconds();
    json out = {{"exp", "dissect"},
                {"rep", rep},
                {"seconds", sec},
                {"threads", st.threads},
                {"separators", st.separators},
                {"leaves", st.leaves},
                {"depth_separator", st.depth_separator},
                {"top_separator_s", st.top_separator_s},
                {"work_s", st.work_s},
                {"critical_path_s", st.critical_path_s}};
    if constexpr (!hs::is_complex_v<T>) {
      std::vector<int> perm(order.size());
      for (std::size_t k = 0; k < order.size(); ++k) perm[order[k]] = static_cast<int>(k + 1);
      Raw<T> r(pencil, a, 1);
      out["mumps_analyze_s"] = r.analyze(1.0, -a.sigma, &perm);
      out.update(r.estimates());
    }
    emit(out);
  }
}

template <class T>
int run(Args args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  if (args.threads <= 0) args.threads = omp_get_max_threads();
  if (std::isnan(args.sigma)) {
    std::ifstream f(args.case_dir / "config.json");
    std::stringstream ss;
    ss << f.rdbuf();
    args.sigma = hs::parse_config(ss.str()).target.sigma;
  }
  json icntl = json::object();
  for (const auto& [k, v] : args.icntl) icntl[std::to_string(k)] = v;
  const char* scotch_threads = std::getenv("SCOTCH_PTHREAD_NUMBER");
  emit({{"exp", "setup"},
        {"case", args.case_dir.filename().string()},
        {"n", pencil.order()},
        {"nnz_upper", pencil.nnz()},
        {"complex", hs::is_complex_v<T>},
        {"threads", args.threads},
        {"l0", args.l0},
        {"sigma", args.sigma},
        {"icntl", icntl},
        {"blocks", args.blocks},
        {"scotch_pthread_number", scotch_threads ? scotch_threads : ""}});
  if (args.exp == "dissect") {
    exp_dissect(args, pencil);
    return 0;
  }
  if (args.exp == "separator") {
    exp_separator(args, pencil);
    return 0;
  }
  for (int rep = 0; rep < args.reps; ++rep)
    for (const auto& o : args.orderings) measure(args, pencil, o, rep);
  return 0;
}

std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string e; std::getline(ss, e, ',');)
    if (!e.empty()) out.push_back(e);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << argv[i] << "\n";
      std::exit(2);
    }
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (k == "--case")
        args.case_dir = need(i);
      else if (k == "--sigma")
        args.sigma = std::stod(need(i));
      else if (k == "--threads")
        args.threads = std::stoi(need(i));
      else if (k == "--reps")
        args.reps = std::stoi(need(i));
      else if (k == "--l0")
        args.l0 = std::stoi(need(i)) != 0;
      else if (k == "--orderings")
        args.orderings = split_list(need(i));
      else if (k == "--icntl") {
        for (const auto& kv : split_list(need(i))) {
          const auto eq = kv.find('=');
          if (eq == std::string::npos) HS_THROW(InvalidInput, "--icntl expects K=V, got " << kv);
          args.icntl.emplace_back(std::stoi(kv.substr(0, eq)), std::stoi(kv.substr(eq + 1)));
        }
      } else if (k == "--blocks")
        args.blocks = std::stoi(need(i)) != 0;
      else if (k == "--split")
        args.split = std::stoi(need(i)) != 0;
      else if (k == "--factorize")
        args.factorize = std::stoi(need(i)) != 0;
      else if (k == "--verbose")
        args.verbose = std::stoi(need(i)) != 0;
      else if (k == "--nd-leaf")
        args.nd.leaf_size = std::stoll(need(i));
      else if (k == "--nd-coarsen")
        args.nd.coarsen_to = std::stoll(need(i));
      else if (k == "--nd-tries")
        args.nd.init_tries = std::stoi(need(i));
      else if (k == "--nd-passes")
        args.nd.refine_passes = std::stoi(need(i));
      else if (k == "--nd-imbalance")
        args.nd.imbalance = std::stod(need(i));
      else if (k == "--nd-seed")
        args.nd.seed = std::stoull(need(i));
      else if (k == "--nd-embedding") {
        const std::string v = need(i);
        args.nd.embedding = v == "full"         ? hs::OrderingOptions::Embedding::Full
                            : v == "compressed" ? hs::OrderingOptions::Embedding::Compressed
                                                : hs::OrderingOptions::Embedding::Auto;
      } else if (k == "--exp")
        args.exp = need(i);
      else if (k == "--nd-mode")
        args.nd_mode = std::stoi(need(i));
      else if (k == "--nd-task")
        args.nd_task = std::stoi(need(i));
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    if (args.case_dir.empty()) {
      std::cerr << "usage: bench_analysis_orderings --case DIR [--orderings 0,6,3,5,4,pnd] ...\n";
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
