// Block Krylov-Schur on the GHIEP path (module ghiep, P4): block size p against p = 1 on real
// indefinite-B cases (wg-* waveguides), at equal tolerance with accuracy and set checks.
//
// Mode "ks" (default), eigensolver level. Per case: load A and B, factorize K = A - sigma B at the
// case's sigma (config.json target; MUMPS LDL^T, --ordering, L0-OpenMP as the driver's default),
// then
//   1. "apply" records: one ShiftInvertOperator application to a block of p columns (spmm with B
//      + one multi-RHS solve), median over --apply-reps, per block and per column;
//   2. "run" records: hs::krylov_schur_general (LargestMagnitude, nev = case k, tol = --tol, the
//      driver's tol * inner_tol_factor = 1e-11 by default) for every p: time split by phase,
//      operator applications, solve calls, restarts, and the accuracy of the k nearest Ritz pairs
//      lambda = sigma + 1/theta: largest backward error ||A x - lambda B x|| / ((||A||_1 +
//      |lambda| ||B||_1) ||x||) (1-norms: a cheap, slightly optimistic scale, the same for every
//      p), largest relative eigenvalue difference to the first p of the list, set size.
// Mode "e2e": hs::solve with the case's config.json (threads from OMP_NUM_THREADS) for every p,
// via SolverConfig::block_size; p = 0 leaves the configuration alone (the driver's automatic
// rule). p > 0 requires that field (see the module's interface request; without it the bench
// exits with an error). Records the report's phase times, applications, success,
// largest backward error, real/complex counts and the eigenvalue differences to the first p.
// Repetitions are the outer loop in both modes, so slow load drifts of the shared machine affect
// every p alike; summarize with medians.
// Output: one JSON object per line on stdout.
//
// Usage (via the resource guard, 2 threads):
//   python3 tools/run/guard.py --skip-busy-check --cores 28,29 --mem-gb 4 --timeout 3600 --
//     build/release/bench_ghiep_block [--mode ks|e2e] [--data DIR] [--p 1,2,4,8] [--reps 3]
//     [--ncv 0] [--apply-reps 9] [--ordering 0] [--tol 1e-11] CASE...
// --ncv (ks mode): requested basis sizes, each run with every p (0 = the solver's auto rule).
// DIR defaults to $HS_DATA_ROOT or ./data; CASE is a directory name under DIR/cases, or a
// synthetic GHIEP pencil generated in process, syn-ghiep-<kind>-<n> with kind wg / wg0 / cplx
// (banded, known spectrum, zero cluster; see synthetic_case).
#include <algorithm>
#include <array>
#include <cmath>
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
#include <type_traits>
#include <utility>
#include <vector>

#include "eigen/general_krylov_schur.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/solver.hpp"
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
  std::string mode = "ks";
  std::filesystem::path data;
  std::vector<index_t> p = {1, 2, 4, 8};
  std::vector<index_t> ncv = {0};  ///< ks mode: requested basis sizes (0 = auto)
  int reps = 3;
  int apply_reps = 9;
  int ordering = 0;
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
    if (s == "--mode")
      a.mode = value();
    else if (s == "--data")
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
  if (a.mode != "ks" && a.mode != "e2e") throw std::runtime_error("--mode must be ks or e2e");
  return a;
}

// SolverConfig::block_size when the field exists (interface request of this module).
template <class C>
auto set_block_size(C& cfg, index_t p, int) -> decltype(cfg.block_size = p, true) {
  cfg.block_size = p;
  return true;
}
template <class C>
bool set_block_size(C&, index_t, long) {
  return false;
}

std::string read_text(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

/// ||A x - lambda B x|| / ((n1a + |lambda| n1b) ||x||) for a complex vector x (real A, B: applied
/// to the real and imaginary parts).
double backward_error(const CsrMatrix<double>& a, const CsrMatrix<double>& b, const cplx* x,
                      cplx lambda, double n1a, double n1b) {
  const index_t n = a.nrows;
  DenseMatrix<double> parts(n, 2), ax(n, 2), bx(n, 2);
  for (index_t i = 0; i < n; ++i) parts(i, 0) = x[i].real(), parts(i, 1) = x[i].imag();
  hs::spmm(a, parts.data(), n, ax.data(), n, 2);
  hs::spmm(b, parts.data(), n, bx.data(), n, 2);
  double r2 = 0.0, x2 = 0.0;
  for (index_t i = 0; i < n; ++i) {
    const cplx r = cplx(ax(i, 0), ax(i, 1)) - lambda * cplx(bx(i, 0), bx(i, 1));
    r2 += std::norm(r);
    x2 += std::norm(x[i]);
  }
  return std::sqrt(r2) / ((n1a + std::abs(lambda) * n1b) * std::sqrt(x2));
}

double max_rel_diff(std::vector<double> got, std::vector<double> ref) {
  if (got.size() != ref.size()) return std::numeric_limits<double>::infinity();
  std::sort(got.begin(), got.end());
  std::sort(ref.begin(), ref.end());
  double d = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i)
    d = std::max(d, std::abs(got[i] - ref[i]) / std::max(std::abs(ref[i]), 1e-300));
  return d;
}

void run_ks(const Args& args, const std::string& name, const CsrMatrix<double>& a,
            const CsrMatrix<double>& b, double sigma0, index_t k) {
  const index_t n = a.nrows;
  hs::ShiftedPencil<double> pencil(a, &b);
  hs::FactorOptions fo;
  fo.ordering = args.ordering;
  fo.deterministic = false;  // the driver's default (L0-OpenMP)
  fo.threads = hs::max_threads();
  auto fac = hs::make_mumps_factorization<double>(fo);
  hs::Timer t_an;
  fac->analyze(pencil, 1.0, -sigma0);
  const double analyze_s = t_an.seconds();
  double sigma = sigma0;
  hs::Timer t_fa;
  bool ok = fac->factorize(1.0, -sigma);
  for (int r = 0; !ok && r < 3; ++r) {
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
    ok = fac->factorize(1.0, -sigma);
  }
  if (!ok) throw std::runtime_error(name + ": singular at every tried shift");
  const double factorize_s = t_fa.seconds();
  hs::ShiftInvertOperator<double> op(*fac, pencil, sigma);
  const double n1a = hs::norm_one(a), n1b = hs::norm_one(b);
  std::cout << json{{"record", "case"},          {"mode", "ks"},
                    {"case", name},              {"n", n},
                    {"sigma", sigma},            {"k", k},
                    {"threads", hs::max_threads()}, {"ordering", args.ordering},
                    {"analyze_s", analyze_s},    {"factorize_s", factorize_s}}
                   .dump()
            << std::endl;

  for (const index_t p : args.p) {
    if (args.apply_reps <= 0) break;
    DenseMatrix<double> x(n, p), y(n, p);
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
    std::cout << json{{"record", "apply"}, {"case", name}, {"p", p}, {"median_s", med},
                      {"median_ms_per_vector", 1e3 * med / static_cast<double>(p)}}
                     .dump()
              << std::endl;
  }

  std::vector<double> ref;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (const index_t ncv : args.ncv) {
    for (const index_t p : args.p) {
      hs::KrylovSchurOptions<double> ko;
      ko.nev = k;
      ko.ncv = ncv;
      ko.tol = args.tol;
      ko.block_size = p;
      hs::eigen_detail::GeneralPhaseTimes tm;
      const auto s0 = fac->stats();
      const auto r = hs::eigen_detail::krylov_schur_general_timed<double>(
          op, hs::Which::LargestMagnitude, ko, &tm);
      const auto s1 = fac->stats();
      std::vector<double> lambda;
      double worst_eta = 0.0, worst_imag = 0.0;
      for (index_t j = 0; j < std::min<index_t>(k, static_cast<index_t>(r.theta.size())); ++j) {
        const cplx l = sigma + 1.0 / r.theta[j];
        lambda.push_back(l.real());
        worst_imag = std::max(worst_imag, std::abs(l.imag()) / std::abs(l));
        worst_eta = std::max(worst_eta, backward_error(a, b, r.vectors.col(j), l, n1a, n1b));
      }
      if (ref.empty()) ref = lambda;
      json hist = json::array();
      for (const auto& h : r.history) hist.push_back({h.nconv, h.max_estimate, h.op_applies});
      std::cout << json{{"record", "run"},
                        {"case", name},
                        {"p", p},
                        {"ncv_option", ncv},
                        {"rep", rep},
                        {"converged", r.converged},
                        {"nconv", r.nconv},
                        {"restarts", r.restarts},
                        {"op_applies", r.op_applies},
                        {"ncv_final", r.ritz_all.size()},
                        {"solve_calls", s1.solves - s0.solves},
                        {"solve_s", s1.solve_s - s0.solve_s},
                        {"total_s", tm.total_s},
                        {"apply_s", tm.apply_s},
                        {"orth_s", tm.orthogonalize_s},
                        {"projected_s", tm.projected_s},
                        {"update_s", tm.update_s},
                        {"max_eta_1norm", worst_eta},
                        {"max_rel_imag", worst_imag},
                        {"max_rel_diff_to_first_p", max_rel_diff(lambda, ref)},
                        {"lambda", lambda},
                        {"history", hist},
                        {"peak_rss_gib", static_cast<double>(hs::peak_rss_bytes()) / (1u << 30)}}
                       .dump()
                << std::endl;
    }
    }
  }
}

void run_e2e(const Args& args, const std::string& name, const CsrMatrix<double>& a,
             const CsrMatrix<double>& b, const hs::SolverConfig& base) {
  std::vector<double> ref;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (const index_t p : args.p) {
      hs::SolverConfig cfg = base;
      if (p != 0 && !set_block_size(cfg, p, 0))
        throw std::runtime_error("this build has no SolverConfig::block_size: e2e needs it");
      const hs::Solution<double> sol = hs::solve(a, &b, cfg);
      const hs::SolveReport& r = sol.report;
      double worst_eta = 0.0;
      for (const auto& pr : r.pairs) worst_eta = std::max(worst_eta, pr.backward_error);
      if (ref.empty()) ref = sol.eigenvalues;
      std::cout << json{{"record", "e2e"},
                        {"case", name},
                        {"p", p},
                        {"rep", rep},
                        {"success", r.success},
                        {"total_s", r.time_total_s},
                        {"setup_s", r.time_setup_s},
                        {"factor_s", r.time_factor_s},
                        {"iterate_s", r.time_iterate_s},
                        {"verify_s", r.time_verify_s},
                        {"op_applies", r.op_applies},
                        {"restarts", r.restarts},
                        {"solve_calls", r.factor_stats.solves},
                        {"solve_s", r.factor_stats.solve_s},
                        {"factorizations", r.factor_stats.factorizations},
                        {"runs", r.slices.size()},
                        {"real", sol.eigenvalues.size()},
                        {"complex", sol.complex_eigenvalues.size()},
                        {"max_eta", worst_eta},
                        {"max_rel_diff_to_first_p", max_rel_diff(sol.eigenvalues, ref)},
                        {"warnings", r.warnings},
                        {"lambda", sol.eigenvalues},
                        {"peak_rss_gib", static_cast<double>(hs::peak_rss_bytes()) / (1u << 30)}}
                       .dump()
                << std::endl;
    }
  }
}

// ------------------------------------------------------------------- synthetic GHIEP pencils

/// 2 x 2 Householder reflector I - 2 v v^T / (v^T v) from a deterministic v seeded by `index`.
std::array<std::array<double, 2>, 2> reflector(index_t index) {
  const double phi = 0.7 + 1.3 * std::sin(0.37 * static_cast<double>(index));
  const double v[2] = {1.0, std::cos(phi)};
  const double vv = v[0] * v[0] + v[1] * v[1];
  std::array<std::array<double, 2>, 2> q{};
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c) q[r][c] = (r == c ? 1.0 : 0.0) - 2.0 / vv * v[r] * v[c];
  return q;
}

/// Banded real GHIEP pencil with a prescribed spectrum, the construction of
/// tests/cpp/test_ghiep_driver.cpp: A = Q^T D_A Q, B = Q^T D_B Q, Q = G2 G1 products of 2 x 2
/// reflectors on the index pairs (0, 1), (2, 3), ... and (1, 2), (3, 4), ...; D_B = diag(+-1),
/// D_A = diag(lambda D_B) for the real eigenvalues, [[x, y], [y, -x]] / diag(1, -1) per complex
/// pair x +- iy. lam.size() even.
std::pair<CsrMatrix<double>, CsrMatrix<double>> synthetic_pencil(const std::vector<double>& lam,
                                                                 const std::vector<cplx>& pairs) {
  const auto nreal = static_cast<index_t>(lam.size());
  const index_t n = nreal + 2 * static_cast<index_t>(pairs.size());
  std::vector<std::vector<std::pair<index_t, double>>> q(static_cast<std::size_t>(n));
  for (index_t k = 0; k < n; ++k) {
    std::vector<std::pair<index_t, double>> g2;
    if (k == 0 || (k == n - 1 && k % 2 == 1)) {
      g2 = {{k, 1.0}};
    } else {
      const index_t base = k - (k + 1) % 2;
      const auto r = reflector(base);
      g2 = {{base, r[k - base][0]}, {base + 1, r[k - base][1]}};
    }
    for (const auto& [l, gv] : g2) {
      const auto r = reflector(l / 2 + 7919);
      const index_t base = l - l % 2;
      q[k].push_back({base, gv * r[l % 2][0]});
      q[k].push_back({base + 1, gv * r[l % 2][1]});
    }
  }
  struct DEntry {
    index_t k, l;
    double a, b;
  };
  std::vector<DEntry> d;
  for (index_t i = 0; i < nreal; ++i) {
    const double s = (i % 3 == 0) ? -1.0 : 1.0;
    d.push_back({i, i, lam[i] * s, s});
  }
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const index_t i = nreal + 2 * static_cast<index_t>(p);
    d.push_back({i, i, pairs[p].real(), 1.0});
    d.push_back({i + 1, i + 1, -pairs[p].real(), -1.0});
    d.push_back({i, i + 1, pairs[p].imag(), 0.0});
    d.push_back({i + 1, i, pairs[p].imag(), 0.0});
  }
  std::vector<index_t> rows, cols;
  std::vector<double> av, bv;
  for (const DEntry& e : d)
    for (const auto& [i, qki] : q[e.k])
      for (const auto& [j, qlj] : q[e.l]) {
        rows.push_back(i);
        cols.push_back(j);
        av.push_back(qki * e.a * qlj);
        bv.push_back(qki * e.b * qlj);
      }
  return {hs::csr_from_coo(n, n, rows, cols, av), hs::csr_from_coo(n, n, rows, cols, bv)};
}

/// Synthetic case by name, "syn-ghiep-<kind>-<n>" (n even):
///   wg    n / 10 zeros (the cluster), 64 negative eigenvalues in [-1, -0.1], positives from 0.05
///         spaced 3 / #positives (dense), sigma = -0.507 (between two negatives), k = 12;
///   wg0   the same pencil, sigma = 0 (cluster mode: one-sided runs), k = 10;
///   cplx  the same spectrum on n - 8 real values plus four complex pairs -0.5 +- 0.004i,
///         -0.52 +- 0.01i, -0.47 +- 0.002i, -0.55 +- 0.02i near sigma = -0.505, k = 16.
bool synthetic_case(const std::string& name, CsrMatrix<double>& a, CsrMatrix<double>& b,
                    hs::SolverConfig& cfg) {
  const std::string prefix = "syn-ghiep-";
  if (name.rfind(prefix, 0) != 0) return false;
  const std::string rest = name.substr(prefix.size());
  const auto dash = rest.find('-');
  if (dash == std::string::npos) throw std::runtime_error(name + ": expected syn-ghiep-<kind>-<n>");
  const std::string kind = rest.substr(0, dash);
  const index_t n = std::stoll(rest.substr(dash + 1));
  if (n < 1000 || n % 2 != 0) throw std::runtime_error(name + ": n must be even and >= 1000");
  auto waveguide_like = [](index_t m) {
    std::vector<double> lam(static_cast<std::size_t>(m / 10), 0.0);
    for (int i = 0; i < 64; ++i) lam.push_back(-0.1 - 0.9 * i / 63.0);
    const auto pos = static_cast<index_t>(m - static_cast<index_t>(lam.size()));
    for (index_t i = 0; i < pos; ++i)
      lam.push_back(0.05 + 3.0 * static_cast<double>(i) / static_cast<double>(pos));
    return lam;
  };
  cfg = hs::SolverConfig{};
  cfg.tol = 1e-10;
  std::pair<CsrMatrix<double>, CsrMatrix<double>> p;
  if (kind == "wg" || kind == "wg0") {
    p = synthetic_pencil(waveguide_like(n), {});
    cfg.target.sigma = kind == "wg" ? -0.507 : 0.0;
    cfg.target.k = kind == "wg" ? 12 : 10;
  } else if (kind == "cplx") {
    p = synthetic_pencil(waveguide_like(n - 8),
                         {cplx(-0.5, 0.004), cplx(-0.52, 0.01), cplx(-0.47, 0.002), cplx(-0.55, 0.02)});
    cfg.target.sigma = -0.505;
    cfg.target.k = 16;
  } else {
    throw std::runtime_error(name + ": unknown synthetic kind '" + kind + "'");
  }
  a = std::move(p.first);
  b = std::move(p.second);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    for (const std::string& name : args.cases) {
      CsrMatrix<double> syn_a, syn_b;
      hs::SolverConfig cfg;
      std::optional<hs::LoadedMatrix> la, lb;
      const CsrMatrix<double>* a = &syn_a;
      const CsrMatrix<double>* b = &syn_b;
      if (!synthetic_case(name, syn_a, syn_b, cfg)) {
        const auto dir = args.data / "cases" / name;
        cfg = hs::parse_config(read_text(dir / "config.json"));
        la = hs::load_matrix(dir / "A.npz");
        lb = hs::load_matrix(dir / "B.npz");
        if (la->is_complex() || lb->is_complex())
          throw std::runtime_error(name + ": complex cases are not supported by this bench");
        a = &std::get<CsrMatrix<double>>(la->matrix);
        b = &std::get<CsrMatrix<double>>(lb->matrix);
      }
      if (cfg.target.mode != hs::TargetMode::Nearest)
        throw std::runtime_error(name + ": target is not 'nearest'");
      if (args.mode == "ks")
        run_ks(args, name, *a, *b, cfg.target.sigma, cfg.target.k);
      else
        run_e2e(args, name, *a, *b, cfg);
    }
  } catch (const std::exception& e) {
    std::cerr << "bench_ghiep_block: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
