// Block size and basis layout of shift-invert Lanczos on the data cases (module eigen, P5b
// "blocktune").
//
// For each case (target "nearest"): load A (and B) with hs::load_matrix, analyse and factorize
// K = A - sigma B as the driver does (MUMPS LDL^T, L0-OpenMP i.e. FactorOptions::deterministic =
// false, the adaptive ordering: AMD probe and driver::decide_ordering with the driver's expected
// factorizations / solves, SCOTCH when nested dissection pays; --ordering N forces ICNTL(7) = N),
// then
//   1. "apply" records: one ShiftInvertOperator application to a block of p columns (spmm with B
//      + one multi-RHS MUMPS solve), median over --apply-reps, per block and per column, for every
//      p of --p and --apply-p;
//   2. "run" records: lanczos_krylov_schur (B inner product, LargestMagnitude, nev = case k,
//      tol = --tol, default 1e-11 = the driver's tol * inner_tol_factor) per configuration:
//      p = 1 with each requested ncv of --ncv1 (0 = auto, the driver's setting), and every p > 1 of
//      --p with every block setting of --extra x --min-ncv x --keep x --fast
//      (eigen_detail::BlockTuning; ncv auto as in the driver). Recorded: operator applications, restarts, the convergence
//      history (applications and converged wanted pairs per Rayleigh-Ritz step), the time split
//      (operator / solve / orthogonalization / Rayleigh-Ritz / restart / total), the largest
//      relative residual ||A x - lambda B x|| / (|lambda| ||B x||), the largest relative difference
//      of the sorted eigenvalues to the first configuration (p = 1, first ncv1) and the number of
//      distinct values (relative 1e-9). Repetitions are the outer loop (interleaved), and the
//      order of the configurations rotates by one per repetition.
// Output: one JSON object per line on stdout.
//
// Usage (via the resource guard):
//   python3 tools/run/guard.py --skip-busy-check --cores 24-31 --mem-gb 16 --timeout 3600 --
//     build/release/bench_blocktune_sweep [--data DIR] [--p 1,2,4,8,16] [--apply-p 32]
//     [--ncv1 0] [--extra 4] [--min-ncv 0] [--keep 0.5] [--fast 0] [--reps 3] [--apply-reps 5]
//     [--ordering auto|N] [--tol 1e-11] CASE...
// DIR defaults to $HS_DATA_ROOT or ./data; CASE is a directory name under DIR/cases.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "driver/internal.hpp"
#include "eigen/block_lanczos.hpp"
#include "eigen/lanczos_detail.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::DenseMatrix;
using hs::index_t;
using json = nlohmann::json;

struct Args {
  std::filesystem::path data;
  std::vector<index_t> p = {1, 2, 4, 8, 16};
  std::vector<index_t> apply_p;
  std::vector<index_t> ncv1 = {0};
  std::vector<index_t> extra = {4};
  std::vector<index_t> min_ncv = {0};
  std::vector<double> keep = {0.5};
  std::vector<index_t> fast = {0};
  int reps = 3;
  int apply_reps = 5;
  int ordering = -1;  // -1 = the driver's adaptive choice
  double tol = 1e-11;
  std::vector<std::string> cases;
};

template <class X>
std::vector<X> parse_list(const std::string& s) {
  std::vector<X> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if constexpr (std::is_same_v<X, double>) {
      out.push_back(std::stod(item));
    } else {
      out.push_back(static_cast<X>(std::stoll(item)));
    }
  }
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  const char* root = std::getenv("HS_DATA_ROOT");
  a.data = root ? std::filesystem::path(root) : std::filesystem::path("data");
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--data") {
      a.data = value();
    } else if (s == "--p") {
      a.p = parse_list<index_t>(value());
    } else if (s == "--apply-p") {
      a.apply_p = parse_list<index_t>(value());
    } else if (s == "--ncv1") {
      a.ncv1 = parse_list<index_t>(value());
    } else if (s == "--extra") {
      a.extra = parse_list<index_t>(value());
    } else if (s == "--min-ncv") {
      a.min_ncv = parse_list<index_t>(value());
    } else if (s == "--keep") {
      a.keep = parse_list<double>(value());
    } else if (s == "--fast") {
      a.fast = parse_list<index_t>(value());
    } else if (s == "--reps") {
      a.reps = std::stoi(value());
    } else if (s == "--apply-reps") {
      a.apply_reps = std::stoi(value());
    } else if (s == "--ordering") {
      const std::string v = value();
      a.ordering = v == "auto" ? -1 : std::stoi(v);
    } else if (s == "--tol") {
      a.tol = std::stod(value());
    } else {
      a.cases.push_back(s);
    }
  }
  if (a.cases.empty()) throw std::runtime_error("no case given");
  return a;
}

template <class T>
CsrMatrix<T> as_scalar(hs::LoadedMatrix&& m) {
  if constexpr (std::is_same_v<T, cplx>) {
    if (!m.is_complex()) return hs::to_complex(std::get<CsrMatrix<double>>(m.matrix));
    return std::move(std::get<CsrMatrix<cplx>>(m.matrix));
  } else {
    return std::move(std::get<CsrMatrix<double>>(m.matrix));
  }
}

/// Largest ||A x - lambda B x|| / (|lambda| ||B x||) over the columns of x.
template <class T>
double max_relative_residual(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const DenseMatrix<T>& x,
                             const std::vector<double>& lambda) {
  const index_t n = x.rows(), k = x.cols();
  DenseMatrix<T> ax(n, k), bx(n, k);
  hs::spmm(a, x.data(), n, ax.data(), n, k);
  if (b) {
    hs::spmm(*b, x.data(), n, bx.data(), n, k);
  } else {
    bx = x;
  }
  double worst = 0.0;
  for (index_t j = 0; j < k; ++j) {
    const double bn = hs::linalg::norm2(n, bx.col(j));
    hs::linalg::axpy(n, T(-lambda[j]), bx.col(j), ax.col(j));
    worst = std::max(worst, hs::linalg::norm2(n, ax.col(j)) / (std::abs(lambda[j]) * bn));
  }
  return worst;
}

/// One solver configuration of the sweep.
struct Config {
  index_t p = 1;
  index_t ncv = 0;  // p = 1: requested ncv (0 = auto); p > 1: always auto
  hs::eigen_detail::BlockTuning tuning;
  std::string name() const {
    std::ostringstream os;
    if (p == 1) {
      os << "p1-ncv" << ncv;
    } else {
      os << "p" << p << "-e" << tuning.extra_blocks << "-m" << tuning.min_auto_ncv << "-k"
         << tuning.keep_fraction << "-f" << (tuning.fast_ortho ? 1 : 0);
    }
    return os.str();
  }
};

std::vector<Config> make_configs(const Args& args) {
  std::vector<Config> out;
  for (index_t p : args.p) {
    if (p == 1) {
      for (index_t ncv : args.ncv1) {
        Config c;
        c.ncv = ncv;
        out.push_back(c);
      }
      continue;
    }
    for (index_t e : args.extra)
      for (index_t m : args.min_ncv)
        for (double k : args.keep)
          for (index_t f : args.fast) {
            Config c;
            c.p = p;
            c.tuning.extra_blocks = e;
            c.tuning.min_auto_ncv = m;
            c.tuning.keep_fraction = k;
            c.tuning.fast_ortho = f != 0;
            out.push_back(c);
          }
  }
  return out;
}

template <class T>
std::unique_ptr<hs::SparseFactorization<T>> analyse(const Args& args, const hs::ShiftedPencil<T>& pencil,
                                                    double sigma, index_t k, json& head) {
  hs::FactorOptions fo;
  fo.threads = hs::max_threads();
  fo.deterministic = false;  // the driver's default (L0-OpenMP)
  hs::Timer t;
  if (args.ordering >= 0) {
    fo.ordering = args.ordering;
    auto fac = hs::make_mumps_factorization<T>(fo);
    fac->analyze(pencil, 1.0, -sigma);
    head["ordering"] = args.ordering;
    head["analyze_s"] = t.seconds();
    return fac;
  }
  // The driver's adaptive choice (solver.cpp: certificate on, B given or not).
  fo.ordering = 0;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -sigma);
  const index_t expected_f = 1 + 2 + (pencil.b() ? 1 : 0);
  const index_t expected_s = std::max<index_t>(40, 3 * k);
  const auto d = hs::driver::decide_ordering(fac->estimate(), hs::driver::mumps_input_size(pencil),
                                             static_cast<double>(pencil.nnz()), fo.threads, expected_f,
                                             expected_s);
  head["fill_ratio"] = d.fill_ratio;
  if (d.use_nd) {
    fac.reset();
    fo.ordering = 3;
    fac = hs::make_mumps_factorization<T>(fo);
    fac->analyze(pencil, 1.0, -sigma);
  }
  head["ordering"] = fo.ordering;
  head["analyze_s"] = t.seconds();
  return fac;
}

template <class T>
void run_case(const Args& args, const std::string& name, hs::LoadedMatrix&& la,
              std::optional<hs::LoadedMatrix>&& lb, double sigma0, index_t k) {
  const CsrMatrix<T> a = as_scalar<T>(std::move(la));
  std::optional<CsrMatrix<T>> b_store;
  if (lb) b_store = as_scalar<T>(std::move(*lb));
  const CsrMatrix<T>* b = b_store ? &*b_store : nullptr;
  const index_t n = a.nrows;
  hs::ShiftedPencil<T> pencil(a, b);

  json head = {{"record", "case"}, {"case", name},           {"n", n},
               {"nnz_a", a.nnz()}, {"nnz_b", b ? b->nnz() : 0}, {"complex", hs::is_complex_v<T>},
               {"k", k},           {"threads", hs::max_threads()}};
  auto fac = analyse<T>(args, pencil, sigma0, k, head);
  double sigma = sigma0;
  hs::Timer t_fa;
  bool ok = fac->factorize(1.0, -sigma);
  for (int r = 0; !ok && r < 3; ++r) {  // the driver's rule: nudge the shift off an eigenvalue
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
    ok = fac->factorize(1.0, -sigma);
  }
  if (!ok) throw std::runtime_error(name + ": singular at every tried shift");
  head["factorize_s"] = t_fa.seconds();
  head["sigma"] = sigma;
  head["factor_entries"] = fac->stats().factor_entries;
  head["estimate_entries"] = fac->stats().estimate.factor_entries;
  head["estimate_flops"] = fac->stats().estimate.flops;
  std::cout << head.dump() << std::endl;
  hs::ShiftInvertOperator<T> op(*fac, pencil, sigma);
  std::unique_ptr<hs::CsrOperator<T>> bop;
  if (b) bop = std::make_unique<hs::CsrOperator<T>>(*b);

  // 1. Operator application per block width.
  std::vector<index_t> ps = args.p;
  ps.insert(ps.end(), args.apply_p.begin(), args.apply_p.end());
  std::sort(ps.begin(), ps.end());
  ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
  for (index_t p : ps) {
    DenseMatrix<T> x(n, p), y(n, p);
    for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.col(j), n, 7, j);
    op.apply(x.data(), n, y.data(), n, p);  // warm-up
    std::vector<double> times;
    for (int r = 0; r < args.apply_reps; ++r) {
      hs::Timer t;
      op.apply(x.data(), n, y.data(), n, p);
      times.push_back(t.seconds());
    }
    std::sort(times.begin(), times.end());
    const double med = times[times.size() / 2];
    json rec = {{"record", "apply"},
                {"case", name},
                {"p", p},
                {"median_s", med},
                {"min_s", times.front()},
                {"max_s", times.back()},
                {"median_ms_per_vector", 1e3 * med / static_cast<double>(p)}};
    std::cout << rec.dump() << std::endl;
  }

  // 2. Eigensolver runs, interleaved: repetitions outside, configurations rotated per repetition.
  const std::vector<Config> configs = make_configs(args);
  std::vector<double> ref_lambda;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (std::size_t i = 0; i < configs.size(); ++i) {
      const Config& c = configs[(i + static_cast<std::size_t>(rep)) % configs.size()];
      hs::KrylovSchurOptions<T> ko;
      ko.nev = k;
      ko.ncv = c.ncv;
      ko.tol = args.tol;
      ko.block_size = c.p;
      hs::eigen_detail::LanczosTimings tm;
      const double s0 = fac->stats().solve_s;
      const index_t calls0 = fac->stats().solves;
      const auto r =
          c.p == 1 ? hs::eigen_detail::lanczos_krylov_schur_timed<T>(
                         op, bop.get(), hs::Which::LargestMagnitude, ko, tm)
                   : hs::eigen_detail::block_lanczos_krylov_schur_timed<T>(
                         op, bop.get(), hs::Which::LargestMagnitude, ko, tm, c.tuning);
      const double solve_s = fac->stats().solve_s - s0;
      std::vector<double> lambda;
      for (double th : r.theta) lambda.push_back(sigma + 1.0 / th);
      const double res = max_relative_residual(a, b, r.vectors, lambda);
      std::vector<double> sorted = lambda;
      std::sort(sorted.begin(), sorted.end());
      if (ref_lambda.empty()) ref_lambda = sorted;
      double diff = 0.0;
      for (std::size_t j = 0; j < sorted.size() && j < ref_lambda.size(); ++j)
        diff = std::max(diff, std::abs(sorted[j] - ref_lambda[j]) /
                                  std::max(std::abs(ref_lambda[j]), 1e-300));
      index_t distinct = sorted.empty() ? 0 : 1;
      for (std::size_t j = 1; j < sorted.size(); ++j)
        if (std::abs(sorted[j] - sorted[j - 1]) > 1e-9 * std::abs(sorted[j])) ++distinct;
      json hist = json::array();
      for (const auto& h : r.history) hist.push_back({h.op_applies, h.nconv, h.max_estimate});
      json rec = {{"record", "run"},
                  {"case", name},
                  {"config", c.name()},
                  {"p", c.p},
                  {"ncv_option", c.ncv},
                  {"extra", c.tuning.extra_blocks},
                  {"min_ncv", c.tuning.min_auto_ncv},
                  {"keep", c.tuning.keep_fraction},
                  {"fast_ortho", c.tuning.fast_ortho},
                  {"final_basis", static_cast<index_t>(r.ritz_all.size())},
                  {"rep", rep},
                  {"converged", r.converged},
                  {"nconv", r.nconv},
                  {"restarts", r.restarts},
                  {"op_applies", r.op_applies},
                  {"solve_calls", fac->stats().solves - calls0},
                  {"total_s", tm.total},
                  {"op_s", tm.op},
                  {"solve_s", solve_s},
                  {"orth_s", tm.orthogonalization},
                  {"rr_s", tm.rayleigh_ritz},
                  {"restart_s", tm.restart},
                  {"max_rel_residual", res},
                  {"max_rel_diff_to_first", diff},
                  {"distinct_values", distinct},
                  {"history", hist},
                  {"peak_rss_gib", static_cast<double>(hs::peak_rss_bytes()) / (1u << 30)}};
      std::cout << rec.dump() << std::endl;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    for (const std::string& name : args.cases) {
      const auto dir = args.data / "cases" / name;
      json cfg;
      std::ifstream(dir / "config.json") >> cfg;
      const auto& target = cfg.at("target");
      if (target.at("mode") != "nearest")
        throw std::runtime_error(name + ": target is not 'nearest'");
      const double sigma = target.at("sigma").get<double>();
      const index_t k = target.at("k").get<index_t>();
      hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
      std::optional<hs::LoadedMatrix> lb;
      if (std::filesystem::exists(dir / "B.npz")) lb = hs::load_matrix(dir / "B.npz");
      const bool complex = la.is_complex() || (lb && lb->is_complex());
      if (complex) {
        run_case<cplx>(args, name, std::move(la), std::move(lb), sigma, k);
      } else {
        run_case<double>(args, name, std::move(la), std::move(lb), sigma, k);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "bench_blocktune_sweep: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
