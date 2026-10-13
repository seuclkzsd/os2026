// Experiment P6 "mixprec", stage A: what a single-precision factorization of the shift-invert
// matrix K = A - sigma B costs and buys at the driver's operating point, double vs single
// (dmumps / smumps; complex K goes through the real embedding, i.e. smumps too).
//
// For one data case and the report.json of a baseline hsolve run on it (the driver's shift
// sigma_used, its resolved ordering, factor cap and block width p): one double analysis with the
// driver's ordering (time, estimates, one factorization = the driver's numeric phase), then a
// double and a single object analysed with THAT analysis' pivot order (PERM_IN, identical
// elimination for both). The single object is first factorized with null-pivot detection
// (single_null_pivot_tol, "single_probe"); a flagged shift is measured without detection (the
// refinement residual then decides usability). Per --exp:
//   phases  numeric factorization time (interleaved repetitions, alternating order), factor
//           memory (INFOG(21) effectively used, entries, RSS growth at the first factorization),
//           inertia at sigma (single vs double), solve time for 1 and for p right-hand sides;
//           accuracy of a plain solve (normwise ||r||/||Bx||, componentwise backward error omega,
//           forward error of the single solution against the double one); mixed-precision
//           iterative refinement (ShiftInvertOperator: residual in double with A and B,
//           corrections with the single factor) to the double solve's ||r||/||Bx|| ("dlevel")
//           and with the driver's Single settings (V1: rel_tol = tol * 0.1 * 0.1, 10 steps):
//           steps and time per accurate application against the double operator.
//   v2      (HEP / GHEP only) the shift-invert Lanczos of the driver (nev = k, block p, Krylov tol
//           tol * 0.1) with the double operator, with the refined single operator (V1; --no-v1
//           leaves it out) and with the PLAIN single operator (V2: Krylov tolerances --v2-tols,
//           nev = k + g for every g in --v2-guards), each followed by Rayleigh-Ritz in double. V2
//           then refines its Ritz pairs until the backward errors eta of the k pairs nearest sigma
//           are <= tol (the driver's acceptance, Lanczos norm estimates), at most --v2-max-steps
//           steps (default 8): --v2-mode rii (default) = residual inverse iteration
//           X <- X - S^{-1} R, R = A X - B X diag(lambda), S = the single factor, then
//           Rayleigh-Ritz; davidson = Rayleigh-Ritz over [X, C], C = S^{-1} R B-orthonormalized
//           against X, keeping per old vector the Ritz vector of largest B-overlap (diverges on
//           the 3-D cases: the single solve's error in C mixes spurious interior Ritz pairs in).
//           --v2-refine-factor double refines with the double factor instead (diagnostic: is a
//           slow refinement the factor's precision or the inverse-iteration rate?). Times,
//           applications, steps, the eta history and the eigenvalue deviation from the double
//           arm |lambda - mu| / (scale + |mu|).
// Repetitions are interleaved with the arm order rotated (v2: reps / 2 + 1 rounds). With
// --single-only the double object is analysed (estimates) but never factorized (cases whose
// double factors do not fit the memory).
// Output: JSON lines on stdout ("kind": setup | single_probe | factor | accuracy | refine | v2 |
// note | median | end), progress on stderr.
//
// Usage (measurement slot: 16 threads = the cores given by the guard):
//   python3 tools/run/guard.py --exclusive --timeout 1800 -- build/release/bench_mixprec_phases
//     --case data/cases/CASE --report RUN/report.json [--exp phases,v2] [--reps 5] [--inner 3]
//     [--p P] [--v2-tols 1e-11,1e-8] [--v2-guards 0] [--v2-mode rii|davidson]
//     [--v2-max-steps 8] [--v2-max-restarts 300] [--v2-refine-factor single|double] [--no-v1]
//     [--single-only]
// Linked with the BLAS hook exactly like the hsolve executable (route "all" unless HS_BLAS_ROUTE
// says otherwise): the hook routes the d* BLAS of dmumps; smumps' s* calls go to OpenBLAS.
// Numbers quoted in results/p6_mixprec.md: gpu02 slot (cores 16-31), base 1bd75f9 + this program;
// the exact commands are in that file's method section.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "driver/norms.hpp"
#include "driver/projection.hpp"
#include "factor/mumps_factorization.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::DenseMatrix;
using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Args {
  fs::path dir;
  fs::path report;
  std::vector<std::string> exps = {"phases", "v2"};
  int reps = 5;
  int inner = 3;
  index_t p = 0;  // 0: the report's block_size
  std::vector<double> v2_tols;  // empty: tol * 0.1 and 1e-8
  std::string v2_mode = "davidson";
  bool single_only = false;
  int v2_max_restarts = 300;
  std::vector<index_t> v2_guards = {0};  // extra Ritz pairs carried through the V2 refinement
  int v2_max_steps = 8;
  bool v1_arm = true;  // the refined-operator Lanczos arm (V1) in --exp v2
  bool refine_with_double = false;  // diagnostic: refine the V2 Ritz pairs with the double factor
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--case")
      a.dir = value();
    else if (s == "--report")
      a.report = value();
    else if (s == "--exp")
      a.exps = split(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--inner")
      a.inner = std::stoi(value());
    else if (s == "--p")
      a.p = std::stoll(value());
    else if (s == "--v2-tols") {
      a.v2_tols.clear();
      for (const auto& t : split(value())) a.v2_tols.push_back(std::stod(t));
    } else if (s == "--v2-mode")
      a.v2_mode = value();
    else if (s == "--v2-max-restarts")
      a.v2_max_restarts = std::stoi(value());
    else if (s == "--v2-guards") {
      a.v2_guards.clear();
      for (const auto& g : split(value())) a.v2_guards.push_back(std::stoll(g));
    } else if (s == "--v2-max-steps")
      a.v2_max_steps = std::stoi(value());
    else if (s == "--v2-refine-factor")
      a.refine_with_double = value() == "double";
    else if (s == "--no-v1")
      a.v1_arm = false;
    else if (s == "--single-only")
      a.single_only = true;
    else
      throw std::runtime_error("unknown option " + s);
  }
  if (a.dir.empty() || a.report.empty())
    throw std::runtime_error("usage: bench_mixprec_phases --case DIR --report REPORT.json [options]");
  if (a.v2_mode != "davidson" && a.v2_mode != "rii") throw std::runtime_error("--v2-mode davidson|rii");
  return a;
}

json read_json(const fs::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  return json::parse(in);
}

bool has(const std::vector<std::string>& v, const std::string& s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

void emit(const json& j) {
  std::printf("%s\n", j.dump().c_str());
  std::fflush(stdout);
}

void progress(const std::string& s) {
  std::fprintf(stderr, "[mixprec] %s\n", s.c_str());
  std::fflush(stderr);
}

double gib(std::int64_t bytes) { return static_cast<double>(bytes) / 1073741824.0; }

template <class T>
double rel_norm(index_t n, const T* r, const T* ref) {
  const double d = hs::linalg::norm2(n, ref);
  return d > 0 ? hs::linalg::norm2(n, r) / d : std::numeric_limits<double>::infinity();
}

/// Per-column accuracy of Y as a solution of K Y = B X: ||r_j|| / ||(B X)_j|| and omega_j.
template <class T>
struct Accuracy {
  std::vector<double> rel, omega;
  double max_rel() const { return *std::max_element(rel.begin(), rel.end()); }
  double max_omega() const { return *std::max_element(omega.begin(), omega.end()); }
};

template <class T>
Accuracy<T> accuracy(const hs::ShiftedPencil<T>& pencil, double sigma, const DenseMatrix<T>& x,
                     const DenseMatrix<T>& bx, const DenseMatrix<T>& y, index_t p) {
  const index_t n = x.rows();
  DenseMatrix<T> r(n, p);
  Accuracy<T> acc;
  acc.omega.resize(static_cast<std::size_t>(p));
  hs::shifted_residual(pencil.a(), pencil.b(), sigma, x.data(), n, y.data(), n, r.data(), n, p, {},
                       acc.omega.data());
  for (index_t j = 0; j < p; ++j) acc.rel.push_back(rel_norm(n, r.col(j), bx.col(j)));
  return acc;
}

// ------------------------------------------------------------------------------- the V2 arm

/// Rayleigh-Ritz state of a block of approximate eigenpairs of a definite pencil.
template <class T>
struct Pairs {
  DenseMatrix<T> x, ax, bx;
  std::vector<double> lambda;
};

template <class T>
std::vector<double> backward_errors(const Pairs<T>& s, double n2a, double n2b) {
  std::vector<double> eta;
  const index_t n = s.x.rows();
  for (index_t j = 0; j < s.x.cols(); ++j)
    eta.push_back(hs::driver::pair_metrics(s.ax.col(j), s.bx.col(j), s.x.col(j), n, s.lambda[j], n2a, n2b)
                      .backward_error);
  return eta;
}

/// Largest backward error among the k pairs nearest sigma (the wanted ones).
template <class T>
double max_eta_nearest(const Pairs<T>& s, double sigma, index_t k, double n2a, double n2b) {
  const std::vector<double> eta = backward_errors(s, n2a, n2b);
  std::vector<index_t> idx(eta.size());
  for (std::size_t j = 0; j < idx.size(); ++j) idx[j] = static_cast<index_t>(j);
  std::stable_sort(idx.begin(), idx.end(), [&](index_t i, index_t j) {
    return std::abs(s.lambda[i] - sigma) < std::abs(s.lambda[j] - sigma);
  });
  double worst = 0;
  for (index_t j = 0; j < std::min<index_t>(k, static_cast<index_t>(idx.size())); ++j)
    worst = std::max(worst, eta[idx[j]]);
  return worst;
}

/// Keeps the k pairs nearest sigma (the nearest-target selection).
template <class T>
void keep_nearest(Pairs<T>& s, double sigma, index_t k) {
  const index_t m = s.x.cols(), n = s.x.rows();
  if (m <= k) return;
  std::vector<index_t> idx(static_cast<std::size_t>(m));
  for (index_t j = 0; j < m; ++j) idx[j] = j;
  std::stable_sort(idx.begin(), idx.end(), [&](index_t i, index_t j) {
    return std::abs(s.lambda[i] - sigma) < std::abs(s.lambda[j] - sigma);
  });
  idx.resize(static_cast<std::size_t>(k));
  std::sort(idx.begin(), idx.end());
  Pairs<T> out;
  out.x.resize(n, k);
  out.ax.resize(n, k);
  out.bx.resize(n, k);
  for (index_t j = 0; j < k; ++j) {
    std::copy(s.x.col(idx[j]), s.x.col(idx[j]) + n, out.x.col(j));
    std::copy(s.ax.col(idx[j]), s.ax.col(idx[j]) + n, out.ax.col(j));
    std::copy(s.bx.col(idx[j]), s.bx.col(idx[j]) + n, out.bx.col(j));
    out.lambda.push_back(s.lambda[idx[j]]);
  }
  s = std::move(out);
}

/// Ritz pairs of a Lanczos result mapped to (A, B): products, Rayleigh-Ritz (as the driver).
template <class T>
Pairs<T> ritz_pairs(const hs::ShiftedPencil<T>& pencil, hs::HermitianEigResult<T>& res) {
  Pairs<T> s;
  s.x = std::move(res.vectors);
  hs::driver::apply_block(&pencil.a(), s.x, s.ax);
  hs::driver::apply_block(pencil.b(), s.x, s.bx);
  hs::driver::rayleigh_ritz_definite(s.x, s.ax, s.bx, s.lambda, 1e-8);
  return s;
}

/// One refinement step of the V2 arm with the single factor S (see the header comment).
template <class T>
void refine_step(hs::SparseFactorization<T>& single, const hs::ShiftedPencil<T>& pencil, double sigma,
                 const std::string& mode, Pairs<T>& s) {
  const index_t n = s.x.rows(), k = s.x.cols();
  DenseMatrix<T> c(n, k);
  for (index_t j = 0; j < k; ++j)
    for (index_t i = 0; i < n; ++i) c(i, j) = s.ax(i, j) - T(s.lambda[j]) * s.bx(i, j);
  single.solve(c.data(), k, n);
  if (mode == "rii") {
    for (index_t j = 0; j < k; ++j) hs::linalg::axpy(n, T(-1), c.col(j), s.x.col(j));
    hs::driver::apply_block(&pencil.a(), s.x, s.ax);
    hs::driver::apply_block(pencil.b(), s.x, s.bx);
    hs::driver::rayleigh_ritz_definite(s.x, s.ax, s.bx, s.lambda, 1e-8);
    return;
  }
  // davidson: Rayleigh-Ritz over [X, C], keep the k nearest sigma. C is B-orthogonalized against
  // X (B-orthonormal after the previous Rayleigh-Ritz) and its columns B-normalized, twice: the
  // corrections are tiny next to X and nearly parallel to it, so the joined block's Gram matrix
  // would otherwise be too ill-conditioned for the rank-revealing Rayleigh-Ritz.
  namespace lp = hs::lapack;
  DenseMatrix<T> ac, bc, g(k, k);
  std::vector<double> before(static_cast<std::size_t>(k));
  hs::driver::apply_block(pencil.b(), c, bc);
  for (index_t j = 0; j < k; ++j)
    before[j] = std::sqrt(std::max(std::real(hs::linalg::dot(n, c.col(j), bc.col(j))), 0.0));
  for (int pass = 0; pass < 2; ++pass) {
    lp::gemm('C', 'N', static_cast<lp::fint>(k), static_cast<lp::fint>(k), static_cast<lp::fint>(n), T(1),
             s.bx.data(), static_cast<lp::fint>(n), c.data(), static_cast<lp::fint>(n), T(0), g.data(),
             static_cast<lp::fint>(k));
    lp::gemm('N', 'N', static_cast<lp::fint>(n), static_cast<lp::fint>(k), static_cast<lp::fint>(k), T(-1),
             s.x.data(), static_cast<lp::fint>(n), g.data(), static_cast<lp::fint>(k), T(1), c.data(),
             static_cast<lp::fint>(n));
    hs::driver::apply_block(pencil.b(), c, bc);
    for (index_t j = 0; j < k; ++j) {
      const double bn = std::sqrt(std::max(std::real(hs::linalg::dot(n, c.col(j), bc.col(j))), 0.0));
      // A column with nothing left beside X (relative 1e-12) is noise: zero it (the Rayleigh-Ritz
      // drops zero columns).
      const T scale = bn > 1e-12 * before[j] && bn > 0 ? T(1.0 / bn) : T(0);
      hs::linalg::scal(n, scale, c.col(j));
      if (pass == 0 && scale != T(0)) before[j] = 1.0;  // compare the second pass with the unit column
    }
  }
  hs::driver::apply_block(&pencil.a(), c, ac);
  hs::driver::apply_block(pencil.b(), c, bc);
  auto join = [&](DenseMatrix<T>& left, const DenseMatrix<T>& right) {
    left.resize_cols(k + right.cols());
    std::copy(right.data(), right.data() + n * right.cols(), left.col(k));
  };
  const DenseMatrix<T> bx_old = s.bx;
  join(s.x, c);
  join(s.ax, ac);
  join(s.bx, bc);
  hs::driver::rayleigh_ritz_definite(s.x, s.ax, s.bx, s.lambda, 1e-8);
  // Selection: standard Rayleigh-Ritz for interior eigenvalues can put spurious Ritz values (mixes
  // of correction directions) next to sigma, so "the k nearest sigma" may pick them. Keep instead,
  // greedily, the Ritz vector with the largest B-overlap |y^H B x_old| for every previous vector.
  const index_t r = s.x.cols();
  DenseMatrix<T> m(r, k);
  lp::gemm('C', 'N', static_cast<lp::fint>(r), static_cast<lp::fint>(k), static_cast<lp::fint>(n), T(1),
           s.x.data(), static_cast<lp::fint>(n), bx_old.data(), static_cast<lp::fint>(n), T(0), m.data(),
           static_cast<lp::fint>(r));
  std::vector<char> row_used(static_cast<std::size_t>(r)), col_used(static_cast<std::size_t>(k));
  std::vector<index_t> keep;
  for (index_t t = 0; t < std::min(r, k); ++t) {
    double best = -1;
    index_t bi = 0, bj = 0;
    for (index_t j = 0; j < k; ++j)
      for (index_t i = 0; i < r; ++i)
        if (!row_used[i] && !col_used[j] && std::abs(m(i, j)) > best) best = std::abs(m(i, j)), bi = i, bj = j;
    row_used[bi] = col_used[bj] = 1;
    keep.push_back(bi);
  }
  std::sort(keep.begin(), keep.end());
  Pairs<T> out;
  out.x.resize(n, static_cast<index_t>(keep.size()));
  out.ax.resize(n, out.x.cols());
  out.bx.resize(n, out.x.cols());
  for (std::size_t q = 0; q < keep.size(); ++q) {
    const index_t i = keep[q], jq = static_cast<index_t>(q);
    std::copy(s.x.col(i), s.x.col(i) + n, out.x.col(jq));
    std::copy(s.ax.col(i), s.ax.col(i) + n, out.ax.col(jq));
    std::copy(s.bx.col(i), s.bx.col(i) + n, out.bx.col(jq));
    out.lambda.push_back(s.lambda[i]);
  }
  s = std::move(out);
  static_cast<void>(sigma);
}

// ------------------------------------------------------------------------------------- run

template <class T>
void run(const Args& args, const std::string& name, const CsrMatrix<T>& a, const CsrMatrix<T>* b) {
  const json rep = read_json(args.report);
  const json cfg = read_json(args.dir / "config.json");
  const double sigma = rep.at("sigma_used").get<double>();
  const std::string problem = rep.at("problem").get<std::string>();
  const json& fo_j = rep.at("factor_options");
  const double tol = cfg.value("tol", 1e-10);
  const index_t k = cfg.at("target").value("k", static_cast<index_t>(6));
  const index_t p = args.p > 0 ? args.p : std::max<index_t>(1, rep.value("block_size", static_cast<index_t>(1)));
  const index_t n = a.nrows;
  const int threads = hs::max_threads();

  hs::FactorOptions fo;
  fo.ordering = fo_j.at("ordering").get<int>();
  fo.deterministic = fo_j.value("deterministic", false);
  fo.threads = threads;
  fo.max_memory_mb = fo_j.value("max_memory_mb", 0.0);
  hs::ShiftedPencil<T> pencil(a, b);

  // Driver path: one analysis with the driver's ordering, one factorization.
  hs::mumps::MumpsFactorization<T, double> ref(fo);
  hs::Timer t;
  ref.analyze(pencil, 1.0, -sigma);
  const double analyze_s = t.seconds();
  const hs::FactorEstimate est_d = ref.estimate();
  double ref_factor_s = -1.0;
  std::int64_t rss0 = hs::current_rss_bytes();
  json setup = {{"kind", "setup"},        {"case", name},
                {"n", n},                 {"complex", hs::is_complex_v<T>},
                {"problem", problem},     {"sigma", sigma},
                {"tol", tol},             {"k", k},
                {"p", p},                 {"threads", threads},
                {"ordering", fo.ordering}, {"max_memory_mb", fo.max_memory_mb},
                {"analyze_s", analyze_s}, {"ordering_s", ref.stats().ordering_s},
                {"est_double", {{"entries", est_d.factor_entries}, {"incore_mb", est_d.incore_mb},
                                {"flops", est_d.flops}}},
                {"blas_route", std::getenv("HS_BLAS_ROUTE") ? std::getenv("HS_BLAS_ROUTE") : "(default)"}};
  if (!args.single_only) {
    t.reset();
    const bool ok = ref.factorize(1.0, -sigma);
    ref_factor_s = t.seconds();
    setup["driver_numeric_s"] = ref_factor_s;
    setup["driver_numeric_ok"] = ok;
    setup["driver_factor_mb_used"] = ref.stats().factor_mb_used;
    setup["driver_factor_entries"] = ref.stats().factor_entries;
    static_cast<void>(ref.release_factors());
  }
  const std::vector<int> order = ref.pivot_order();

  // The two objects of the comparison: the same given pivot order (identical elimination).
  std::unique_ptr<hs::mumps::MumpsFactorization<T, double>> dbl;
  if (!args.single_only) dbl = std::make_unique<hs::mumps::MumpsFactorization<T, double>>(fo);
  hs::FactorOptions fs_opts = fo;
  fs_opts.precision = hs::FactorPrecision::Single;
  auto sgl = std::make_unique<hs::mumps::MumpsFactorization<T, float>>(fs_opts);
  if (dbl) {
    dbl->set_pivot_order(order);
    dbl->analyze(pencil, 1.0, -sigma);
    setup["analyze_given_order_s"] = dbl->stats().analyze_s;
  }
  sgl->set_pivot_order(order);
  sgl->analyze(pencil, 1.0, -sigma);
  const hs::FactorEstimate est_s = sgl->estimate();
  setup["est_single"] = {{"entries", est_s.factor_entries}, {"incore_mb", est_s.incore_mb}, {"flops", est_s.flops}};
  rss0 = hs::current_rss_bytes();
  setup["rss_before_factor_gib"] = gib(rss0);
  emit(setup);

  // ---------------------------------------------------------------- numeric factorizations
  // Probe: the single factorization with null-pivot detection (single_null_pivot_tol). A flagged
  // shift is not necessarily unusable in single precision (the threshold is relative to the
  // scaled matrix: small but well-determined pivots trip it); the measurements then factorize
  // without detection, and the refinement residual decides usability.
  const std::int64_t rss_probe = hs::current_rss_bytes();
  const bool single_detection_ok = sgl->factorize(1.0, -sigma);
  const double rss_s_probe = gib(hs::current_rss_bytes() - rss_probe);
  emit({{"kind", "single_probe"}, {"case", name}, {"ok", single_detection_ok},
        {"null_pivots", sgl->stats().null_pivots}, {"seconds", sgl->stats().factorize_s}});
  if (!single_detection_ok) sgl->set_null_pivot_detection(false);
  std::vector<double> fd, fs;
  bool d_ok = true, s_ok = true;
  double rss_d = 0, rss_s = rss_s_probe;
  progress(name + ": factorizations");
  for (int r = 0; r < args.reps; ++r) {
    for (int arm = 0; arm < 2; ++arm) {
      const bool single = (arm + r) % 2 == 1;
      if (!single && !dbl) continue;
      hs::SparseFactorization<T>& f = single ? static_cast<hs::SparseFactorization<T>&>(*sgl) : *dbl;
      const double before = f.stats().factorize_s;
      const std::int64_t rss_before = hs::current_rss_bytes();
      const bool ok = f.factorize(1.0, -sigma);
      const double dt = f.stats().factorize_s - before;
      if (r == 0 && !single) rss_d = gib(hs::current_rss_bytes() - rss_before);
      (single ? fs : fd).push_back(dt);
      (single ? s_ok : d_ok) = ok && (single ? s_ok : d_ok);
      emit({{"kind", "factor"}, {"case", name}, {"rep", r}, {"precision", single ? "single" : "double"},
            {"seconds", dt}, {"ok", ok}});
    }
  }
  auto fstats = [&](hs::SparseFactorization<T>& f, double rss_growth) {
    const hs::Inertia in = f.inertia();
    return json{{"factor_mb_used", f.stats().factor_mb_used}, {"factor_mb_allocated", f.stats().factor_mb_allocated},
                {"factor_entries", f.stats().factor_entries}, {"null_pivots", f.stats().null_pivots},
                {"delayed_pivots", f.stats().delayed_pivots}, {"workspace_retries", f.stats().workspace_retries},
                {"inertia_neg", in.neg}, {"inertia_zero", in.zero}, {"inertia_valid", in.valid},
                {"rss_growth_first_factorization_gib", rss_growth}};
  };
  json fmed = {{"kind", "median"},          {"what", "factor"},
               {"case", name},              {"single_s", median(fs)},
               {"single_ok", s_ok},         {"single_detection_ok", single_detection_ok},
               {"single", fstats(*sgl, rss_s)}, {"driver_numeric_s", ref_factor_s}};
  if (dbl) {
    fmed["double_s"] = median(fd);
    fmed["double_ok"] = d_ok;
    fmed["double"] = fstats(*dbl, rss_d);
    fmed["speedup"] = median(fd) / median(fs);
  }
  emit(fmed);
  if (!s_ok) {
    emit({{"kind", "note"}, {"case", name}, {"text", "single factorization flagged singular: no solves"}});
    return;
  }

  // ---------------------------------------------------------------------------------- solves
  std::vector<index_t> widths = {1};
  if (p > 1) widths.push_back(p);
  DenseMatrix<T> x(n, p), bx(n, p);
  for (index_t j = 0; j < p; ++j) hs::linalg::fill_random(x.col(j), n, 2026, static_cast<std::uint64_t>(j));
  hs::driver::apply_block(b, x, bx);

  if (has(args.exps, "phases")) {
    progress(name + ": solves");
    DenseMatrix<T> y(n, p);
    for (index_t w : widths) {
      std::vector<double> sd, ss;
      for (int r = 0; r < args.reps; ++r)
        for (int arm = 0; arm < 2; ++arm) {
          const bool single = (arm + r) % 2 == 1;
          if (!single && !dbl) continue;
          hs::SparseFactorization<T>& f = single ? static_cast<hs::SparseFactorization<T>&>(*sgl) : *dbl;
          for (int i = 0; i < args.inner; ++i) {
            std::copy(bx.data(), bx.data() + n * w, y.data());
            hs::Timer ts;
            f.solve(y.data(), w, n);
            (single ? ss : sd).push_back(ts.seconds());
          }
        }
      json m = {{"kind", "median"}, {"what", "solve"}, {"case", name}, {"nrhs", w}, {"single_s", median(ss)}};
      if (dbl) {
        m["double_s"] = median(sd);
        m["speedup"] = median(sd) / median(ss);
      }
      emit(m);
    }

    // Accuracy of plain solves and of the refined single operator.
    progress(name + ": accuracy");
    for (index_t w : widths) {
      DenseMatrix<T> yd(n, w), ys(n, w);
      DenseMatrix<T> xw(n, w), bxw(n, w);
      std::copy(x.data(), x.data() + n * w, xw.data());
      std::copy(bx.data(), bx.data() + n * w, bxw.data());
      std::optional<Accuracy<T>> acc_d;
      if (dbl) {
        std::copy(bxw.data(), bxw.data() + n * w, yd.data());
        dbl->solve(yd.data(), w, n);
        acc_d = accuracy(pencil, sigma, xw, bxw, yd, w);
      }
      std::copy(bxw.data(), bxw.data() + n * w, ys.data());
      sgl->solve(ys.data(), w, n);
      const Accuracy<T> acc_s = accuracy(pencil, sigma, xw, bxw, ys, w);
      json j = {{"kind", "accuracy"}, {"case", name}, {"nrhs", w},
                {"single_rel", acc_s.max_rel()}, {"single_omega", acc_s.max_omega()}};
      if (acc_d) {
        double fwd = 0;
        for (index_t c = 0; c < w; ++c) {
          std::vector<T> dif(static_cast<std::size_t>(n));
          for (index_t i = 0; i < n; ++i) dif[i] = ys(i, c) - yd(i, c);
          fwd = std::max(fwd, rel_norm(n, dif.data(), yd.col(c)));
        }
        j["double_rel"] = acc_d->max_rel();
        j["double_omega"] = acc_d->max_omega();
        j["single_forward_error"] = fwd;
      }
      emit(j);

      // Refinement to the double level (dlevel) and with the driver's V1 settings.
      const double dlevel = acc_d ? acc_d->max_rel() : 1e-15;
      struct Policy {
        const char* name;
        hs::SolveRefinement r;
      };
      const Policy policies[] = {{"dlevel", {20, dlevel, true}}, {"v1", {10, tol * 0.1 * 0.1, true}}};
      for (const Policy& pol : policies) {
        hs::ShiftInvertOperator<T> op(*sgl, pencil, sigma, pol.r);
        DenseMatrix<T> yr(n, w);
        op.apply(xw.data(), n, yr.data(), n, w);
        const Accuracy<T> acc_r = accuracy(pencil, sigma, xw, bxw, yr, w);
        emit({{"kind", "refine"}, {"case", name}, {"nrhs", w}, {"policy", pol.name}, {"rel_tol", pol.r.rel_tol},
              {"steps", op.refinement_steps()}, {"rel", acc_r.max_rel()}, {"omega", acc_r.max_omega()},
              {"reached", acc_r.max_rel() <= pol.r.rel_tol || acc_r.max_omega() <= 8 * 2.220446049250313e-16}});
      }
    }

    // Time per accurate application: double operator (driver's double path, no refinement) vs
    // single + refinement to dlevel vs single + V1 refinement vs plain single.
    progress(name + ": applications");
    for (index_t w : widths) {
      double dlevel = 1e-15;
      if (dbl) {
        DenseMatrix<T> yd(n, w), xw(n, w), bxw(n, w);
        std::copy(x.data(), x.data() + n * w, xw.data());
        std::copy(bx.data(), bx.data() + n * w, bxw.data());
        std::copy(bxw.data(), bxw.data() + n * w, yd.data());
        dbl->solve(yd.data(), w, n);
        dlevel = accuracy(pencil, sigma, xw, bxw, yd, w).max_rel();
      }
      std::vector<std::unique_ptr<hs::ShiftInvertOperator<T>>> ops;
      std::vector<std::string> names;
      if (dbl) {
        ops.push_back(std::make_unique<hs::ShiftInvertOperator<T>>(*dbl, pencil, sigma));
        names.push_back("double");
      }
      ops.push_back(std::make_unique<hs::ShiftInvertOperator<T>>(*sgl, pencil, sigma));
      names.push_back("single_plain");
      ops.push_back(std::make_unique<hs::ShiftInvertOperator<T>>(*sgl, pencil, sigma,
                                                                 hs::SolveRefinement{20, dlevel, true}));
      names.push_back("single_dlevel");
      ops.push_back(std::make_unique<hs::ShiftInvertOperator<T>>(
          *sgl, pencil, sigma, hs::SolveRefinement{10, tol * 0.1 * 0.1, true}));
      names.push_back("single_v1");
      std::vector<std::vector<double>> times(ops.size());
      std::vector<index_t> steps0(ops.size());
      for (std::size_t o = 0; o < ops.size(); ++o) steps0[o] = ops[o]->refinement_steps();
      DenseMatrix<T> y(n, w);
      const index_t m = static_cast<index_t>(ops.size());
      for (int r = 0; r < args.reps; ++r)
        for (index_t q = 0; q < m; ++q) {
          const index_t o = (q + r) % m;
          for (int i = 0; i < args.inner; ++i) {
            hs::Timer ta;
            ops[o]->apply(x.data(), n, y.data(), n, w);
            times[o].push_back(ta.seconds());
          }
        }
      json j = {{"kind", "median"}, {"what", "apply"}, {"case", name}, {"nrhs", w}, {"dlevel", dlevel}};
      for (std::size_t o = 0; o < ops.size(); ++o) {
        const double applies = static_cast<double>(args.reps) * args.inner;
        j[names[o] + "_s"] = median(times[o]);
        j[names[o] + "_steps_per_apply"] = static_cast<double>(ops[o]->refinement_steps() - steps0[o]) / applies;
      }
      emit(j);
    }
  }

  // ------------------------------------------------------------------------------ V2 arm
  if (has(args.exps, "v2")) {
    if (problem == "generalized_indefinite") {
      emit({{"kind", "note"}, {"case", name}, {"text", "v2 arm: HEP/GHEP only (GHIEP not simulated)"}});
      return;
    }
    const auto na = hs::driver::estimate_norm2(a, 30, 2026);
    const double n2a = na.norm2;
    const double n2b = b ? hs::driver::estimate_norm2(*b, 30, 2026).norm2 : 1.0;
    const double scale = n2a / n2b;
    std::unique_ptr<hs::CsrOperator<T>> b_op;
    if (b) b_op = std::make_unique<hs::CsrOperator<T>>(*b);
    std::vector<double> v2_tols = args.v2_tols;
    if (v2_tols.empty()) v2_tols = {tol * 0.1, 1e-8};
    struct Arm {
      std::string name;
      bool single;
      hs::SolveRefinement refine;
      double krylov_tol;
      index_t guard;
    };
    std::vector<Arm> arms;
    if (dbl) arms.push_back({"double", false, {}, tol * 0.1, 0});
    if (args.v1_arm) arms.push_back({"v1", true, {10, tol * 0.1 * 0.1, true}, tol * 0.1, 0});
    for (double kt : v2_tols)
      for (index_t g : args.v2_guards) {
        std::ostringstream os;
        os << "v2_" << kt << "_g" << g;
        arms.push_back({os.str(), true, {}, kt, g});
      }
    std::vector<double> mu;  // eigenvalues of the double arm's first repetition
    const index_t m = static_cast<index_t>(arms.size());
    std::vector<std::vector<json>> results(arms.size());
    for (int r = 0; r < std::max(1, args.reps / 2 + 1); ++r) {
      for (index_t q = 0; q < m; ++q) {
        const index_t ai = (q + r) % m;
        const Arm& arm = arms[ai];
        progress(name + ": v2 rep " + std::to_string(r) + " arm " + arm.name);
        hs::SparseFactorization<T>& f = arm.single ? static_cast<hs::SparseFactorization<T>&>(*sgl) : *dbl;
        hs::ShiftInvertOperator<T> op(f, pencil, sigma, arm.refine);
        hs::KrylovSchurOptions<T> ko;
        const index_t nev = std::min(k + arm.guard, n / 2);
        ko.nev = nev;
        ko.tol = arm.krylov_tol;
        ko.seed = 2026;
        ko.block_size = p;
        ko.max_restarts = arm.single && arm.refine.max_steps == 0 ? args.v2_max_restarts : 1000;
        const double solve0 = f.stats().solve_s;
        const index_t rhs0 = f.stats().rhs_solved;
        hs::Timer tk;
        hs::HermitianEigResult<T> res = hs::lanczos_krylov_schur<T>(op, b_op.get(), hs::Which::LargestMagnitude, ko);
        const double krylov_s = tk.seconds();
        const double krylov_solve_s = f.stats().solve_s - solve0;
        const index_t krylov_rhs = f.stats().rhs_solved - rhs0;
        hs::Timer tr;
        Pairs<T> s = ritz_pairs(pencil, res);
        keep_nearest(s, sigma, nev);
        const double eta_rr = max_eta_nearest(s, sigma, k, n2a, n2b);
        const double rr_s = tr.seconds();
        json j = {{"kind", "v2"},
                  {"case", name},
                  {"rep", r},
                  {"arm", arm.name},
                  {"krylov_tol", arm.krylov_tol},
                  {"guard", arm.guard},
                  {"converged", res.converged},
                  {"nconv", res.nconv},
                  {"restarts", res.restarts},
                  {"op_applies", res.op_applies},
                  {"krylov_s", krylov_s},
                  {"krylov_solve_s", krylov_solve_s},
                  {"krylov_rhs", krylov_rhs},
                  {"refinement_steps_in_krylov", op.refinement_steps()},
                  {"rr_s", rr_s},
                  {"eta_after_rr", eta_rr},
                  {"max_estimate", res.history.empty() ? 0.0 : res.history.back().max_estimate}};
        // V2: refine the Ritz pairs with the single factor until eta <= tol.
        if (arm.single && arm.refine.max_steps == 0) {
          std::vector<double> hist = {eta_rr};
          hs::Timer tf;
          const double rs0 = f.stats().solve_s;
          int steps = 0;
          while (hist.back() > tol && steps < args.v2_max_steps) {
            if (args.refine_with_double && dbl)
              refine_step<T>(*dbl, pencil, sigma, args.v2_mode, s);
            else
              refine_step<T>(*sgl, pencil, sigma, args.v2_mode, s);
            hist.push_back(max_eta_nearest(s, sigma, k, n2a, n2b));
            ++steps;
          }
          j["refine_mode"] = args.v2_mode;
          j["refine_factor"] = args.refine_with_double && dbl ? "double" : "single";
          j["refine_steps"] = steps;
          j["refine_s"] = tf.seconds();
          j["refine_solve_s"] = f.stats().solve_s - rs0;
          j["eta_history"] = hist;
          j["eta_final"] = hist.back();
          j["accepted"] = hist.back() <= tol;
        } else {
          j["eta_final"] = eta_rr;
          j["accepted"] = eta_rr <= tol;
        }
        keep_nearest(s, sigma, k);
        std::vector<double> lam = s.lambda;
        std::sort(lam.begin(), lam.end());
        if (arm.name == "double" && mu.empty()) mu = lam;
        if (!mu.empty() && mu.size() == lam.size()) {
          double dev = 0;
          for (std::size_t i = 0; i < lam.size(); ++i)
            dev = std::max(dev, std::abs(lam[i] - mu[i]) / (scale + std::abs(mu[i])));
          j["max_dev_vs_double"] = dev;
        }
        j["lambda"] = lam;
        emit(j);
        results[ai].push_back(j);
      }
    }
    for (std::size_t ai = 0; ai < arms.size(); ++ai) {
      std::vector<double> ks, ref_s, tot;
      for (const json& j : results[ai]) {
        ks.push_back(j["krylov_s"].get<double>());
        const double rf = j.value("refine_s", 0.0);
        ref_s.push_back(rf);
        tot.push_back(j["krylov_s"].get<double>() + j["rr_s"].get<double>() + rf);
      }
      emit({{"kind", "median"}, {"what", "v2"}, {"case", name}, {"arm", arms[ai].name}, {"krylov_s", median(ks)},
            {"refine_s", median(ref_s)}, {"iterate_total_s", median(tot)},
            {"op_applies", results[ai].front()["op_applies"]}, {"accepted", results[ai].front()["accepted"]}});
    }
  }
  emit({{"kind", "end"}, {"case", name}, {"peak_rss_gib", gib(hs::peak_rss_bytes())}});
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const std::string name = args.dir.filename().string();
    hs::LoadedMatrix la = hs::load_matrix(args.dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.dir / "B.npz")) lb = hs::load_matrix(args.dir / "B.npz");
    if (la.is_complex() || (lb && lb->is_complex())) {
      auto as_complex = [](hs::LoadedMatrix& m) {
        return m.is_complex() ? std::get<CsrMatrix<cplx>>(std::move(m.matrix))
                              : hs::to_complex(std::get<CsrMatrix<double>>(m.matrix));
      };
      const CsrMatrix<cplx> a = as_complex(la);
      std::optional<CsrMatrix<cplx>> b;
      if (lb) b = as_complex(*lb);
      run<cplx>(args, name, a, b ? &*b : nullptr);
    } else {
      const auto& a = std::get<CsrMatrix<double>>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<CsrMatrix<double>>(lb->matrix) : nullptr;
      run<double>(args, name, a, b);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_mixprec_phases: %s\n", e.what());
    return 1;
  }
  return 0;
}
