// End-to-end effect of the P4 "solvephase" kernels on a shift-invert Lanczos run.
//
// For one data case (HEP / GHEP): factorize K = A - sigma B at the case's sigma with the driver's
// factor options (L0-OpenMP, --ordering), then run hs::eigen_detail::lanczos_krylov_schur_timed
// (Op = K^{-1} B, B inner product, LargestMagnitude, nev = case k, tol = 1e-11 = the driver's
// tol * inner_tol_factor) for every block size in --p, with the operators on KernelPath
// Reference (the P1 sparse products, "before") and Auto ("after"), repetitions interleaved over
// block sizes and paths. Same factorization for both, so the comparison isolates the sparse
// kernels (B X in the operator, the B applications of the orthogonalization) - the MUMPS solves
// are identical. Accuracy is checked at equal tolerance: largest relative residual
// ||A x - lambda B x|| / (|lambda| ||B x||) of the returned pairs and the largest relative
// eigenvalue difference to the "before" run of the same block size.
// Output: one JSON object per line ("kind": "run" per repetition, "median" per configuration).
//
// Usage (via the resource guard; threads = cores given):
//   python3 tools/run/guard.py --skip-busy-check --cores 34,35,36,37 --mem-gb 6 --timeout 3600 --
//     build/release/bench_solvephase_lanczos CASE_DIR [--ordering amd|scotch] [--p 1,8]
//     [--reps 5] [--tol 1e-11]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
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
using hs::KernelPath;
using json = nlohmann::json;

struct Args {
  std::filesystem::path dir;
  std::string ordering = "scotch";
  std::vector<index_t> p = {1, 8};
  int reps = 5;
  double tol = 1e-11;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

Args parse_args(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("usage: bench_solvephase_lanczos CASE_DIR [options]");
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
    const std::string v = argv[++i];
    if (s == "--ordering")
      a.ordering = v;
    else if (s == "--p")
      a.p = parse_list(v);
    else if (s == "--reps")
      a.reps = std::stoi(v);
    else if (s == "--tol")
      a.tol = std::stod(v);
    else
      throw std::runtime_error("unknown option " + s);
  }
  return a;
}

json read_json(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  return json::parse(in);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/// Largest ||A x - lambda B x|| / (|lambda| ||B x||) over the converged pairs (reference kernels).
template <class T>
double max_relative_residual(const CsrMatrix<T>& a, const CsrMatrix<T>* b,
                             const hs::HermitianEigResult<T>& r, double sigma) {
  const index_t n = a.nrows;
  std::vector<T> ax(static_cast<std::size_t>(n)), bx(ax.size());
  double worst = 0.0;
  for (index_t j = 0; j < r.nconv; ++j) {
    const double lambda = sigma + 1.0 / r.theta[j];
    const T* x = r.vectors.col(j);
    hs::spmv(a, x, ax.data(), T(1), T(0), {KernelPath::Reference});
    if (b) {
      hs::spmv(*b, x, bx.data(), T(1), T(0), {KernelPath::Reference});
    } else {
      std::copy(x, x + n, bx.begin());
    }
    double num = 0.0;
    for (index_t i = 0; i < n; ++i) num += hs::abs2(ax[i] - lambda * bx[i]);
    worst = std::max(worst, std::sqrt(num) / (std::abs(lambda) * hs::linalg::norm2(n, bx.data())));
  }
  return worst;
}

template <class T>
void run(const Args& args, const std::string& name, const CsrMatrix<T>& a, const CsrMatrix<T>* b,
         double sigma0, index_t nev) {
  hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;
  fo.ordering = args.ordering == "amd" ? 0 : args.ordering == "metis" ? 5 : 3;
  fo.deterministic = false;  // driver default
  fo.threads = hs::max_threads();
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -sigma0);
  double sigma = sigma0;
  for (int r = 0; !fac->factorize(1.0, -sigma); ++r) {
    if (r == 3) throw std::runtime_error("no regular shift near sigma");
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
  }
  const KernelPath paths[] = {KernelPath::Reference, KernelPath::Auto};
  std::map<std::pair<index_t, int>, std::map<std::string, std::vector<double>>> samples;
  std::map<index_t, std::vector<double>> before_theta;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (index_t p : args.p) {
      for (int k = 0; k < 2; ++k) {
        hs::ShiftInvertOperator<T> op(*fac, pencil, sigma, {}, paths[k]);
        std::unique_ptr<hs::CsrOperator<T>> b_op;
        if (b) b_op = std::make_unique<hs::CsrOperator<T>>(*b, paths[k]);
        hs::KrylovSchurOptions<T> ko;
        ko.nev = nev;
        ko.tol = args.tol;
        ko.seed = 2026;
        ko.block_size = p;
        hs::eigen_detail::LanczosTimings t;
        const double solve0 = fac->stats().solve_s;
        const auto res = hs::eigen_detail::lanczos_krylov_schur_timed<T>(
            op, b_op.get(), hs::Which::LargestMagnitude, ko, t);
        const double solve_s = fac->stats().solve_s - solve0;
        std::vector<double> theta(res.theta.begin(), res.theta.begin() + res.nconv);
        double eig_dev = 0.0;
        if (k == 0 && rep == 0) before_theta[p] = theta;
        const auto& ref = before_theta[p];
        for (std::size_t i = 0; i < std::min(ref.size(), theta.size()); ++i) {
          const double l0 = sigma + 1.0 / ref[i], l1 = sigma + 1.0 / theta[i];
          eig_dev = std::max(eig_dev, std::abs(l1 - l0) / std::abs(l0));
        }
        const double resid = max_relative_residual(a, b, res, sigma);
        json rec = {{"kind", "run"},
                    {"case", name},
                    {"rep", rep},
                    {"block_size", p},
                    {"path", k == 0 ? "reference" : "auto"},
                    {"threads", hs::max_threads()},
                    {"total_s", t.total},
                    {"op_s", t.op},
                    {"solve_s", solve_s},
                    {"ortho_s", t.orthogonalization},
                    {"rr_s", t.rayleigh_ritz},
                    {"restart_s", t.restart},
                    {"op_applies", res.op_applies},
                    {"restarts", res.restarts},
                    {"nconv", res.nconv},
                    {"max_rel_residual", resid},
                    {"max_rel_eig_dev_vs_before", eig_dev}};
        std::printf("%s\n", rec.dump().c_str());
        std::fflush(stdout);
        auto& s = samples[{p, k}];
        s["total_s"].push_back(t.total);
        s["op_s"].push_back(t.op);
        s["solve_s"].push_back(solve_s);
        s["ortho_s"].push_back(t.orthogonalization);
        s["other_s"].push_back(t.total - t.op - t.orthogonalization);
        s["max_rel_residual"].push_back(resid);
        s["op_applies"].push_back(static_cast<double>(res.op_applies));
      }
    }
  }
  for (index_t p : args.p) {
    for (int k = 0; k < 2; ++k) {
      json rec = {{"kind", "median"},
                  {"case", name},
                  {"block_size", p},
                  {"path", k == 0 ? "reference" : "auto"},
                  {"threads", hs::max_threads()}};
      for (const auto& [key, v] : samples[{p, k}]) rec[key] = median(v);
      rec["speedup_total_vs_reference"] =
          median(samples[{p, 0}]["total_s"]) / median(samples[{p, k}]["total_s"]);
      std::printf("%s\n", rec.dump().c_str());
    }
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto meta = read_json(args.dir / "meta.json");
    const auto cfg = read_json(args.dir / "config.json");
    if (meta.at("problem").get<std::string>() == "GHIEP")
      throw std::runtime_error("GHIEP case: the Lanczos path needs B positive definite");
    const double sigma = cfg.at("target").at("sigma").get<double>();
    const index_t nev = cfg.at("target").at("k").get<index_t>();
    const std::string name = args.dir.filename().string();
    hs::LoadedMatrix la = hs::load_matrix(args.dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (std::filesystem::exists(args.dir / "B.npz")) lb = hs::load_matrix(args.dir / "B.npz");
    if (la.is_complex() || (lb && lb->is_complex())) {
      auto as_complex = [](hs::LoadedMatrix& m) {
        return m.is_complex() ? std::get<CsrMatrix<cplx>>(std::move(m.matrix))
                              : hs::to_complex(std::get<CsrMatrix<double>>(m.matrix));
      };
      const CsrMatrix<cplx> a = as_complex(la);
      std::optional<CsrMatrix<cplx>> b;
      if (lb) b = as_complex(*lb);
      run<cplx>(args, name, a, b ? &*b : nullptr, sigma, nev);
    } else {
      const auto& a = std::get<CsrMatrix<double>>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<CsrMatrix<double>>(lb->matrix) : nullptr;
      run<double>(args, name, a, b, sigma, nev);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_solvephase_lanczos: %s\n", e.what());
    return 1;
  }
  return 0;
}
