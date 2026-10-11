// Block Krylov-Schur against single-vector Lanczos at MATCHED ACCURACY in the real shift-invert
// path (module eigen, P3 "blockks" review fixes).
//
// Why: an early-stopped block run (block_size > 1) ends as soon as every wanted estimate passes
// tol, while p = 1 tests only at cycle ends and usually overshoots tol by orders of magnitude, so
// equal tol is not equal accuracy (review finding: residuals 10-2700x larger at p = 8). This
// program compares the time to reach at least the accuracy that p = 1 delivers today.
//
// For each data case: load A (and B) with hs::load_matrix, factorize K = A - sigma B at the case's
// sigma (MUMPS LDL^T, SCOTCH ordering as the driver picks for 3-D graphs; the driver's nudge when
// K is singular), then run hs::lanczos_krylov_schur (B inner product, LargestMagnitude, nev = case
// k, automatic ncv) for
//   * p = 1 at each tol of --p1-tols (default 1e-9,1e-10,1e-11; 1e-11 = the driver's tol *
//     inner_tol_factor is the reference),
//   * every p of --p (default 2,4,8) at each tol of --block-tols (default 1e-11..1e-14),
// with the repetitions interleaved across configurations (outer loop), so slow drifts of a shared
// machine's load hit every configuration alike. Accuracy of a run: the largest relative residual
// ||A x - lambda B x|| / (|lambda| ||B x||) over the k returned pairs.
// Output (JSON lines on stdout):
//   "case"    sizes, sigma, factorization;
//   "run"     one per run: times (total, operator, MUMPS solve, orthogonalization), applications,
//             restarts, residual, eigenvalues;
//   "summary" per case and p: medians at the reference, and "matched": the loosest block tol whose
//             median residual is <= --match-factor (default 2) times the reference's (p = 1 at
//             --ref-tol; the largest residual over k pairs varies by about that much between runs
//             of equal accuracy) with its median time and speedup = t(p = 1, ref) / t(p, matched
//             tol), or null when no block tol in the list gets there; "equal_tol": the speedup at
//             the reference tol itself (not accuracy-matched); "p1_matched": the loosest p = 1
//             tol whose residual is <= the factor times the block run's at the reference tol, and
//             the block speedup against it (the comparison at the block run's own accuracy).
//   "overall" geometric means of the matched and equal-tol speedups per p over the cases.
//
// Usage (via the resource guard, 2 threads as in the review):
//   python3 tools/run/guard.py --skip-busy-check --cores 24,25 --mem-gb 8 --timeout 7200 --
//     build/release/bench_blockks_fix_matched [--data DIR] [--p 2,4,8] [--p1-tols 1e-9,1e-10,1e-11]
//     [--block-tols 1e-11,1e-12,1e-13,1e-14] [--ref-tol 1e-11] [--match-factor 2] [--reps 3]
//     [--ordering 3] CASE...
// DIR defaults to $HS_DATA_ROOT or ./data; CASE is a directory name under DIR/cases whose target
// mode is "nearest" (HEP/GHEP with B positive definite: the Lanczos path).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
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
  std::vector<index_t> p = {2, 4, 8};
  std::vector<double> p1_tols = {1e-9, 1e-10, 1e-11};
  std::vector<double> block_tols = {1e-11, 1e-12, 1e-13, 1e-14};
  double ref_tol = 1e-11;
  double match_factor = 2.0;
  int reps = 3;
  int ordering = 3;
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
    if (s == "--data")
      a.data = value();
    else if (s == "--p")
      a.p = parse_list<index_t>(value());
    else if (s == "--p1-tols")
      a.p1_tols = parse_list<double>(value());
    else if (s == "--block-tols")
      a.block_tols = parse_list<double>(value());
    else if (s == "--ref-tol")
      a.ref_tol = std::stod(value());
    else if (s == "--match-factor")
      a.match_factor = std::stod(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--ordering")
      a.ordering = std::stoi(value());
    else
      a.cases.push_back(s);
  }
  if (a.cases.empty()) throw std::runtime_error("no case given");
  if (std::find(a.p1_tols.begin(), a.p1_tols.end(), a.ref_tol) == a.p1_tols.end())
    a.p1_tols.push_back(a.ref_tol);
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

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Config {
  index_t p;
  double tol;
};

struct Samples {
  std::vector<double> total, residual, applies, restarts;
};

/// Geometric-mean accumulator.
struct GeoMean {
  double log_sum = 0.0;
  int count = 0;
  void add(double x) {
    log_sum += std::log(x);
    ++count;
  }
  double value() const { return count ? std::exp(log_sum / count) : 0.0; }
};

/// Per-p geometric means over the cases.
struct Overall {
  std::map<index_t, GeoMean> matched, equal;
};

template <class T>
void run_case(const Args& args, const std::string& name, hs::LoadedMatrix&& la,
              std::optional<hs::LoadedMatrix>&& lb, double sigma0, index_t k, Overall& overall) {
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

  std::cout << json{{"record", "case"},
                    {"case", name},
                    {"n", n},
                    {"complex", hs::is_complex_v<T>},
                    {"generalized", b != nullptr},
                    {"sigma", sigma},
                    {"k", k},
                    {"threads", hs::max_threads()},
                    {"analyze_s", analyze_s},
                    {"factorize_s", factorize_s}}
                   .dump()
            << std::endl;

  std::vector<Config> configs;
  for (double tol : args.p1_tols) configs.push_back({1, tol});
  for (index_t p : args.p)
    for (double tol : args.block_tols) configs.push_back({p, tol});
  std::vector<Samples> samples(configs.size());
  for (int rep = 0; rep < args.reps; ++rep) {
    for (std::size_t ic = 0; ic < configs.size(); ++ic) {
      const Config& c = configs[ic];
      hs::KrylovSchurOptions<T> ko;
      ko.nev = k;
      ko.tol = c.tol;
      ko.block_size = c.p;
      hs::eigen_detail::LanczosTimings tm;
      const double s0 = fac->stats().solve_s;
      const auto r = hs::eigen_detail::lanczos_krylov_schur_timed<T>(
          op, bop.get(), hs::Which::LargestMagnitude, ko, tm);
      const double solve_s = fac->stats().solve_s - s0;
      std::vector<double> lambda;
      for (double th : r.theta) lambda.push_back(sigma + 1.0 / th);
      const double res = max_relative_residual(a, b, r.vectors, lambda);
      std::vector<double> sorted = lambda;
      std::sort(sorted.begin(), sorted.end());
      samples[ic].total.push_back(tm.total);
      samples[ic].residual.push_back(res);
      samples[ic].applies.push_back(static_cast<double>(r.op_applies));
      samples[ic].restarts.push_back(static_cast<double>(r.restarts));
      std::cout << json{{"record", "run"},
                        {"case", name},
                        {"p", c.p},
                        {"tol", c.tol},
                        {"rep", rep},
                        {"converged", r.converged},
                        {"ncv", static_cast<index_t>(r.ritz_all.size())},
                        {"op_applies", r.op_applies},
                        {"restarts", r.restarts},
                        {"total_s", tm.total},
                        {"op_s", tm.op},
                        {"solve_s", solve_s},
                        {"orth_s", tm.orthogonalization},
                        {"max_rel_residual", res},
                        {"lambda", sorted}}
                       .dump()
                << std::endl;
    }
  }

  // Summaries: medians over the repetitions.
  auto med = [&](std::size_t ic, std::vector<double> Samples::*f) {
    return median(samples[ic].*f);
  };
  auto index_of = [&](index_t p, double tol) {
    for (std::size_t ic = 0; ic < configs.size(); ++ic)
      if (configs[ic].p == p && configs[ic].tol == tol) return ic;
    throw std::runtime_error("configuration missing");
  };
  const std::size_t ref = index_of(1, args.ref_tol);
  const double t_ref = med(ref, &Samples::total), res_ref = med(ref, &Samples::residual);
  for (index_t p : args.p) {
    const std::size_t eq = index_of(p, args.ref_tol);
    json s = {{"record", "summary"},
              {"case", name},
              {"p", p},
              {"ref_tol", args.ref_tol},
              {"p1_total_s", t_ref},
              {"p1_residual", res_ref},
              {"p1_applies", med(ref, &Samples::applies)},
              {"equal_tol_total_s", med(eq, &Samples::total)},
              {"equal_tol_residual", med(eq, &Samples::residual)},
              {"equal_tol_applies", med(eq, &Samples::applies)},
              {"equal_tol_speedup", t_ref / med(eq, &Samples::total)}};
    overall.equal[p].add(t_ref / med(eq, &Samples::total));
    // Loosest block tol whose residual is within match_factor of p = 1's at the reference tol.
    std::optional<std::size_t> matched;
    for (double tol : args.block_tols) {
      const std::size_t ic = index_of(p, tol);
      if (med(ic, &Samples::residual) <= args.match_factor * res_ref &&
          (!matched || tol > configs[*matched].tol))
        matched = ic;
    }
    if (matched) {
      const double tm = med(*matched, &Samples::total);
      s["matched_tol"] = configs[*matched].tol;
      s["matched_total_s"] = tm;
      s["matched_residual"] = med(*matched, &Samples::residual);
      s["matched_applies"] = med(*matched, &Samples::applies);
      s["matched_speedup"] = t_ref / tm;
      overall.matched[p].add(t_ref / tm);
    } else {
      s["matched_tol"] = nullptr;  // no block tol in the list reaches p = 1's accuracy
    }
    // The other direction: the loosest p = 1 tol about as accurate as the block run at ref tol.
    std::optional<std::size_t> p1m;
    for (double tol : args.p1_tols) {
      const std::size_t ic = index_of(1, tol);
      if (med(ic, &Samples::residual) <= args.match_factor * med(eq, &Samples::residual) &&
          (!p1m || tol > configs[*p1m].tol))
        p1m = ic;
    }
    if (p1m) {
      s["p1_matched_tol"] = configs[*p1m].tol;
      s["p1_matched_total_s"] = med(*p1m, &Samples::total);
      s["p1_matched_speedup"] = med(*p1m, &Samples::total) / med(eq, &Samples::total);
    }
    std::cout << s.dump() << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    Overall acc;
    int failed = 0;
    for (const std::string& name : args.cases) {
      try {  // a failing case (e.g. memory) is reported and the others still run
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
          run_case<cplx>(args, name, std::move(la), std::move(lb), sigma, k, acc);
        } else {
          run_case<double>(args, name, std::move(la), std::move(lb), sigma, k, acc);
        }
      } catch (const std::exception& e) {
        ++failed;
        std::cout << json{{"record", "error"}, {"case", name}, {"what", e.what()}}.dump()
                  << std::endl;
      }
    }
    json overall = {{"record", "overall"}, {"cases", args.cases.size()}};
    for (const auto& [p, g] : acc.matched) {
      overall["matched_geomean_p" + std::to_string(p)] = g.value();
      overall["matched_cases_p" + std::to_string(p)] = g.count;
    }
    for (const auto& [p, g] : acc.equal)
      overall["equal_tol_geomean_p" + std::to_string(p)] = g.value();
    std::cout << overall.dump() << std::endl;
    if (failed > 0) return 1;
  } catch (const std::exception& e) {
    std::cerr << "bench_blockks_fix_matched: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
