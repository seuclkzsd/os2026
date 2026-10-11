// Jacobi-Davidson option study for the memory-limited fallback (fix round of module lobpcg_jd):
// jacobi_davidson preconditioned by solves with a single- (or double-) precision factor of
// A - s I (standard problems only), with explicit extraction and start-iteration options,
// repeated with a FRESH factorization per run (MUMPS is non-deterministic by default, as in the
// driver, so the factor and with it the JD iteration path differ between runs: report
// distributions, not one run).
// Prints one JSON line per run on stdout. The si-double / si-single references are the modes of
// bench_lobpcg_jd_memory (one mode per process, for the peak RSS).
//
//   bench_lobpcg_jd_fix_options --case NAME [--precision single|double]
//       [--extraction auto|harmonic|ritz] [--start N] [--repeat R] [--k K] [--sigma S]
//       [--precond-sigma S] [--tol 1e-10] [--inner 20] [--ordering 3] [--deterministic 0]
//       [--max-iterations 1000] [--data-root DIR]
// --start: JdOptions::start_iterations (-1 = library default). --precond-sigma: shift s of the
// factor (default: the target sigma), e.g. a slightly moved shift when sigma is an eigenvalue.
// Case data: <data-root>/cases/NAME/{A,B}.npz + config.json; reference eigenvalues
// <data-root>/refs/NAME/eigenvalues.npy (accuracy report only).
// Run (2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 30,33 --mem-gb 8 --timeout 3600 --
//       build/release/bench_lobpcg_jd_fix_options --case syn-mem27-46 --extraction ritz --repeat 5
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/norms.hpp"
#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Args {
  std::string case_name, data_root, precision = "single", extraction = "auto";
  long long k = -1, start = -1, repeat = 1, inner = 20, max_iterations = 1000;
  std::optional<double> sigma, precond_sigma;
  double tol = 1e-10;
  int ordering = 3;
  bool deterministic = false;
};

hs::JdExtraction parse_extraction(const std::string& s) {
  if (s == "auto") return hs::JdExtraction::Auto;
  if (s == "harmonic") return hs::JdExtraction::Harmonic;
  if (s == "ritz") return hs::JdExtraction::Ritz;
  HS_THROW(InvalidInput, "unknown extraction " << s);
}

/// Sorted values of `got` against the reference values nearest sigma (sorted).
json compare_with_reference(const fs::path& ref_file, std::vector<double> got, double sigma, double scale) {
  json out;
  if (!fs::exists(ref_file)) return out;
  auto ref = hs::npy_to_vector<double>(hs::read_npy(ref_file));
  std::stable_sort(ref.begin(), ref.end(),
                   [sigma](double p, double q) { return std::abs(p - sigma) < std::abs(q - sigma); });
  ref.resize(std::min(ref.size(), got.size()));
  std::sort(ref.begin(), ref.end());
  std::sort(got.begin(), got.end());
  double max_abs = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) max_abs = std::max(max_abs, std::abs(got[i] - ref[i]));
  out["max_abs_err_vs_ref"] = max_abs;
  out["set_matches_ref"] = max_abs <= 1e-8 * scale;
  return out;
}

template <class T>
void run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, double sigma, hs::index_t k,
         const fs::path& ref_file, json base) {
  if (b) HS_THROW(InvalidInput, "standard problems only (B == nullptr); use bench_lobpcg_jd_memory with B");
  const double na = hs::driver::estimate_norm2(a).norm2;
  const double ps = args.precond_sigma ? *args.precond_sigma : sigma;
  base["norm_a"] = na;
  base["precond_sigma"] = ps;
  hs::CsrOperator<T> aop(a);
  hs::ShiftedPencil<T> pencil(a, nullptr);
  for (long long rep = 0; rep < args.repeat; ++rep) {
    json out = base;
    out["run"] = rep;
    hs::FactorOptions fo;
    fo.precision = args.precision == "double" ? hs::FactorPrecision::Double : hs::FactorPrecision::Single;
    fo.ordering = args.ordering;
    fo.threads = hs::max_threads();
    fo.deterministic = args.deterministic;
    auto fac = hs::make_mumps_factorization<T>(fo);
    hs::Timer tf;
    fac->analyze(pencil, 1.0, -ps);
    if (!fac->factorize(1.0, -ps)) {
      out["error"] = "singular factor";
      std::cout << out.dump() << std::endl;
      continue;
    }
    out["time_factor_s"] = tf.seconds();
    // B == nullptr: the shift-invert operator without refinement is exactly y = K^{-1} x.
    hs::ShiftInvertOperator<T> prec(*fac, pencil, ps);
    hs::JdOptions<T> o;
    o.nev = k;
    o.tol = args.tol;
    o.inner_iterations = args.inner;
    o.max_iterations = args.max_iterations;
    o.extraction = parse_extraction(args.extraction);
    if (args.start >= 0) o.start_iterations = args.start;
    o.norm_a = na;
    out["start_iterations"] = static_cast<long long>(o.start_iterations);
    hs::Timer tj;
    const auto r = hs::jacobi_davidson<T>(aop, nullptr, &prec, sigma, o);
    out["time_jd_s"] = tj.seconds();
    out["converged"] = r.converged;
    out["iterations"] = static_cast<long long>(r.iterations);
    out["inner_iterations"] = static_cast<long long>(r.inner_iterations);
    out["tracking_steps"] = static_cast<long long>(r.tracking_steps);
    out["restarts"] = static_cast<long long>(r.restarts);
    out["solves"] = static_cast<long long>(r.precond_applies);
    out["factor_mb_used"] = fac->stats().factor_mb_used;
    double max_eta = 0.0;
    for (double e : r.estimate) max_eta = std::max(max_eta, e);
    out["found"] = static_cast<long long>(r.theta.size());
    out["max_backward_error"] = max_eta;
    out["reference"] = compare_with_reference(ref_file, r.theta, sigma, na);
    out["peak_rss_bytes"] = static_cast<long long>(hs::peak_rss_bytes());
    std::cout << out.dump() << std::endl;
  }
}

Args parse(int argc, char** argv) {
  Args a;
  const char* root = std::getenv("HS_DATA_ROOT");
  a.data_root = root ? root : "data";
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) HS_THROW(InvalidInput, "missing value for " << key);
    const std::string val = argv[++i];
    if (key == "--case") a.case_name = val;
    else if (key == "--precision") a.precision = val;
    else if (key == "--extraction") a.extraction = val;
    else if (key == "--start") a.start = std::stoll(val);
    else if (key == "--repeat") a.repeat = std::stoll(val);
    else if (key == "--k") a.k = std::stoll(val);
    else if (key == "--sigma") a.sigma = std::stod(val);
    else if (key == "--precond-sigma") a.precond_sigma = std::stod(val);
    else if (key == "--tol") a.tol = std::stod(val);
    else if (key == "--inner") a.inner = std::stoll(val);
    else if (key == "--ordering") a.ordering = std::stoi(val);
    else if (key == "--deterministic") a.deterministic = std::stoi(val) != 0;
    else if (key == "--max-iterations") a.max_iterations = std::stoll(val);
    else if (key == "--data-root") a.data_root = val;
    else HS_THROW(InvalidInput, "unknown option " << key);
  }
  if (a.case_name.empty()) HS_THROW(InvalidInput, "--case is required");
  if (a.precision != "single" && a.precision != "double") HS_THROW(InvalidInput, "--precision single|double");
  parse_extraction(a.extraction);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse(argc, argv);
    const fs::path dir = fs::path(args.data_root) / "cases" / args.case_name;
    std::ifstream cf(dir / "config.json");
    if (!cf) HS_THROW(Io, "cannot read " << (dir / "config.json").string());
    const json cfg = json::parse(cf);
    const double sigma = args.sigma ? *args.sigma : cfg.at("target").value("sigma", 0.0);
    const hs::index_t k = args.k > 0 ? args.k : cfg.at("target").value("k", 6);
    hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(dir / "B.npz")) lb = hs::load_matrix(dir / "B.npz");
    const fs::path ref = fs::path(args.data_root) / "refs" / args.case_name / "eigenvalues.npy";
    json base;
    base["case"] = args.case_name;
    base["n"] = static_cast<long long>(la.order());
    base["k"] = static_cast<long long>(k);
    base["sigma"] = sigma;
    base["tol"] = args.tol;
    base["precision"] = args.precision;
    base["extraction"] = args.extraction;
    base["deterministic"] = args.deterministic;
    base["threads"] = hs::max_threads();
    const bool cx = la.is_complex() || (lb && lb->is_complex());
    if (cx) {
      hs::CsrMatrix<hs::cplx> a =
          la.is_complex() ? std::get<1>(std::move(la.matrix)) : hs::to_complex(std::get<0>(la.matrix));
      hs::CsrMatrix<hs::cplx> b;
      if (lb) b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(lb->matrix));
      run<hs::cplx>(args, a, lb ? &b : nullptr, sigma, k, ref, base);
    } else {
      const auto& a = std::get<0>(la.matrix);
      run<double>(args, a, lb ? &std::get<0>(lb->matrix) : nullptr, sigma, k, ref, base);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_lobpcg_jd_fix_options: %s\n", e.what());
    return 2;
  }
}
