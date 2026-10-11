// Block Krylov-Schur in the real shift-invert setting (module eigen, P3 "blockks").
//
// For each data case: load A (and B) with hs::load_matrix, factorize K = A - sigma B at the case's
// sigma (config.json target, MUMPS LDL^T, ordering as the driver picks for 3-D graphs: SCOTCH),
// then
//   1. "apply" records: time of one ShiftInvertOperator application to a block of p columns
//      (spmm with B + one multi-RHS MUMPS solve), per block and per column;
//   2. "run" records: hs::lanczos_krylov_schur (B inner product, LargestMagnitude, nev = case k,
//      tol = 1e-11 = the driver's tol * inner_tol_factor) for every block size p and basis size
//      ncv: iteration time split (operator / orthogonalization / Rayleigh-Ritz / restart), MUMPS
//      solve time, operator applications, restarts, and the accuracy of the result (largest
//      relative residual ||A x - lambda B x|| / (|lambda| ||B x||), largest relative eigenvalue
//      difference to the first run (first p of the list) with the same ncv option, distinct
//      values among the k found). Repetitions are interleaved across configurations.
// Output: one JSON object per line on stdout.
//
// Usage (via the resource guard, e.g. 2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 24,25 --mem-gb 6 --timeout 3600 --
//     build/release/bench_blockks_shiftinvert [--data DIR] [--p 1,2,4,8] [--ncv 0,48]
//     [--reps 3] [--apply-reps 10] [--ordering 3] [--tol 1e-11] CASE...
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
  std::vector<index_t> p = {1, 2, 4, 8};
  std::vector<index_t> ncv = {0};
  int reps = 3;
  int apply_reps = 10;
  int ordering = 3;
  double tol = 1e-11;
  std::vector<std::string> cases;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
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
    if (s == "--data")
      a.data = value();
    else if (s == "--p")
      a.p = parse_list(value());
    else if (s == "--ncv")
      a.ncv = parse_list(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--apply-reps")
      a.apply_reps = std::stoi(value());
    else if (s == "--ordering")
      a.ordering = std::stoi(value());
    else if (s == "--tol")
      a.tol = std::stod(value());
    else
      a.cases.push_back(s);
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

template <class T>
void run_case(const Args& args, const std::string& name, hs::LoadedMatrix&& la,
              std::optional<hs::LoadedMatrix>&& lb, double sigma0, index_t k) {
  const CsrMatrix<T> a = as_scalar<T>(std::move(la));
  std::optional<CsrMatrix<T>> b_store;
  if (lb) b_store = as_scalar<T>(std::move(*lb));
  const CsrMatrix<T>* b = b_store ? &*b_store : nullptr;
  const index_t n = a.nrows;
  hs::ShiftedPencil<T> pencil(a, b);

  hs::FactorOptions fo;
  fo.ordering = args.ordering;
  fo.threads = hs::max_threads();
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer t_an;
  fac->analyze(pencil, 1.0, -sigma0);
  const double analyze_s = t_an.seconds();
  double sigma = sigma0;
  hs::Timer t_fa;
  bool ok = fac->factorize(1.0, -sigma);
  for (int r = 0; !ok && r < 3; ++r) {  // the driver's rule: nudge the shift off an eigenvalue
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
    ok = fac->factorize(1.0, -sigma);
  }
  if (!ok) throw std::runtime_error(name + ": singular at every tried shift");
  const double factorize_s = t_fa.seconds();
  hs::ShiftInvertOperator<T> op(*fac, pencil, sigma);
  std::unique_ptr<hs::CsrOperator<T>> bop;
  if (b) bop = std::make_unique<hs::CsrOperator<T>>(*b);

  json head = {{"record", "case"},
               {"case", name},
               {"n", n},
               {"nnz_a", a.nnz()},
               {"nnz_b", b ? b->nnz() : 0},
               {"complex", hs::is_complex_v<T>},
               {"sigma", sigma},
               {"k", k},
               {"threads", hs::max_threads()},
               {"ordering", args.ordering},
               {"analyze_s", analyze_s},
               {"factorize_s", factorize_s},
               {"factor_entries", fac->stats().factor_entries}};
  std::cout << head.dump() << std::endl;

  // 1. Operator application per block width.
  for (index_t p : args.p) {
    DenseMatrix<T> x(n, p), y(n, p);
    for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.col(j), n, 7, j);
    op.apply(x.data(), n, y.data(), n, p);  // warm-up
    std::vector<double> times;
    double solve_s = 0.0;
    for (int r = 0; r < args.apply_reps; ++r) {
      const double s0 = fac->stats().solve_s;
      hs::Timer t;
      op.apply(x.data(), n, y.data(), n, p);
      times.push_back(t.seconds());
      solve_s += fac->stats().solve_s - s0;
    }
    std::sort(times.begin(), times.end());
    const double med = times[times.size() / 2];
    json rec = {{"record", "apply"},
                {"case", name},
                {"p", p},
                {"median_s", med},
                {"min_s", times.front()},
                {"median_ms_per_vector", 1e3 * med / static_cast<double>(p)},
                {"solve_s_per_block", solve_s / args.apply_reps}};
    std::cout << rec.dump() << std::endl;
  }

  // 2. Eigensolver runs. Repetitions form the outer loop, so that slow drifts of the machine load
  // (shared server) affect every configuration alike.
  std::vector<std::vector<double>> ref_lambda(args.ncv.size());
  for (int rep = 0; rep < args.reps; ++rep) {
    for (std::size_t ic = 0; ic < args.ncv.size(); ++ic) {
      const index_t ncv = args.ncv[ic];
      for (index_t p : args.p) {
        hs::KrylovSchurOptions<T> ko;
        ko.nev = k;
        ko.ncv = ncv;
        ko.tol = args.tol;
        ko.block_size = p;
        hs::eigen_detail::LanczosTimings tm;
        const double s0 = fac->stats().solve_s;
        const index_t rhs0 = fac->stats().rhs_solved, calls0 = fac->stats().solves;
        const auto r = hs::eigen_detail::lanczos_krylov_schur_timed<T>(
            op, bop.get(), hs::Which::LargestMagnitude, ko, tm);
        const double solve_s = fac->stats().solve_s - s0;
        std::vector<double> lambda;
        for (double th : r.theta) lambda.push_back(sigma + 1.0 / th);
        const double res = max_relative_residual(a, b, r.vectors, lambda);
        std::vector<double> sorted = lambda;
        std::sort(sorted.begin(), sorted.end());
        if (ref_lambda[ic].empty()) ref_lambda[ic] = sorted;
        double diff = 0.0;
        for (std::size_t i = 0; i < sorted.size(); ++i)
          diff =
              std::max(diff, std::abs(sorted[i] - ref_lambda[ic][i]) / std::abs(ref_lambda[ic][i]));
        // Distinct values (relative 1e-9) among the result: copies of multiple eigenvalues.
        index_t distinct = sorted.empty() ? 0 : 1;
        for (std::size_t i = 1; i < sorted.size(); ++i)
          if (std::abs(sorted[i] - sorted[i - 1]) > 1e-9 * std::abs(sorted[i])) ++distinct;
        index_t eff_ncv = static_cast<index_t>(r.ritz_all.size());
        json rec = {{"record", "run"},
                    {"case", name},
                    {"p", p},
                    {"ncv_option", ncv},
                    {"ncv", eff_ncv},
                    {"rep", rep},
                    {"converged", r.converged},
                    {"nconv", r.nconv},
                    {"restarts", r.restarts},
                    {"op_applies", r.op_applies},
                    {"solve_calls", fac->stats().solves - calls0},
                    {"rhs_solved", fac->stats().rhs_solved - rhs0},
                    {"total_s", tm.total},
                    {"op_s", tm.op},
                    {"solve_s", solve_s},
                    {"orth_s", tm.orthogonalization},
                    {"rr_s", tm.rayleigh_ritz},
                    {"restart_s", tm.restart},
                    {"max_rel_residual", res},
                    {"max_rel_diff_to_first_p", diff},
                    {"distinct_values", distinct},
                    {"lambda", sorted},
                    {"peak_rss_gib", static_cast<double>(hs::peak_rss_bytes()) / (1u << 30)}};
        std::cout << rec.dump() << std::endl;
      }
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
    std::cerr << "bench_blockks_shiftinvert: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
