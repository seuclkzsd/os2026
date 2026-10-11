// Time split of the real shift-invert eigensolver path on a data case: load A (and B), factorize
// K = A - sigma B at the case's configured shift (MUMPS, as the driver does), then run the Krylov
// solver the driver uses on Op = K^{-1} B: thick-restart Lanczos in the B inner product for
// HEP/GHEP (B applied on demand: the recompute mode of linalg::orthogonalize_cgs2), general
// Krylov-Schur for GHIEP (Euclidean CGS2). Prints one JSON line per repetition with the phase
// times (operator = factor solves, orthogonalization = CGS2 incl. B applications, ...) and the
// loss of orthogonality of the returned basis: max |X^H M X - I| for the Lanczos Ritz vectors
// (M = B, or I), max |Q^H Q - I| for the general solver's Schur basis.
// Built against two library versions it gives a before/after comparison of src/linalg changes.
// Run under the guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 6 --timeout 900 --
//       build/release/bench_ortho_shift_invert $HS_DATA_ROOT/cases/cav2d-rect-pert-1e5 3
// Arguments: <case dir> [repetitions = 3] [ncv = 0 (solver default)] [nev = 0 (the case's k)]
// (a larger nev widens the Krylov basis, i.e. the share of orthogonalization in the run)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include <nlohmann/json.hpp>

#include "eigen/general_krylov_schur.hpp"
#include "eigen/lanczos_detail.hpp"
#include "hsolve/eigen.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

using hs::cplx;
using hs::index_t;

nlohmann::json read_json(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", p.string().c_str());
    std::exit(2);
  }
  return nlohmann::json::parse(in);
}

/// max |X^H Y - I| for X, Y n x m (Y = M X).
template <class T>
double gram_deviation(const hs::DenseMatrix<T>& x, const hs::DenseMatrix<T>& y) {
  const index_t n = x.rows(), m = x.cols();
  hs::DenseMatrix<T> g(m, m);
  hs::lapack::gemm('C', 'N', static_cast<hs::lapack::fint>(m), static_cast<hs::lapack::fint>(m),
                   static_cast<hs::lapack::fint>(n), T(1), x.data(), static_cast<hs::lapack::fint>(n), y.data(),
                   static_cast<hs::lapack::fint>(n), T(0), g.data(), static_cast<hs::lapack::fint>(m));
  double dev = 0.0;
  for (index_t j = 0; j < m; ++j)
    for (index_t i = 0; i < m; ++i) dev = std::max(dev, std::abs(g(i, j) - T(i == j ? 1.0 : 0.0)));
  return dev;
}

template <class T>
void run(const std::string& name, const std::string& problem, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b,
         double sigma, index_t k, double tol, int reps, index_t ncv) {
  hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;  // driver defaults: ordering auto, deterministic, all OpenMP threads
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer tf;
  fac->analyze(pencil, 1.0, -sigma);
  // A shift on an eigenvalue (null pivot): move it slightly, as the driver does.
  double used = sigma;
  for (int r = 0; !fac->factorize(1.0, -used); ++r) {
    if (r == 3) {
      std::fprintf(stderr, "%s: no regular shift near %g\n", name.c_str(), sigma);
      std::exit(3);
    }
    used = sigma + 1e-8 * std::max(1.0, std::abs(sigma)) * std::pow(10.0, r);
  }
  const double factor_s = tf.seconds();
  hs::ShiftInvertOperator<T> op(*fac, pencil, used);
  std::unique_ptr<hs::CsrOperator<T>> b_op;
  if (b) b_op = std::make_unique<hs::CsrOperator<T>>(*b);
  hs::KrylovSchurOptions<T> ko;
  ko.nev = k;
  ko.ncv = ncv;
  ko.tol = tol * 0.1;  // SolverConfig::inner_tol_factor
  ko.seed = 2026;
  for (int rep = 0; rep < reps; ++rep) {
    nlohmann::json out = {{"case", name},       {"problem", problem}, {"n", a.nrows},
                          {"complex", hs::is_complex_v<T>},           {"nev", k},
                          {"sigma", used},      {"threads", hs::max_threads()},
                          {"factor_s", factor_s}, {"rep", rep}};
    if (problem == "GHIEP") {
      hs::eigen_detail::GeneralPhaseTimes t;
      const auto r = hs::eigen_detail::krylov_schur_general_timed<T>(op, hs::Which::LargestMagnitude, ko, &t);
      out.update({{"solver", "krylov_schur_general"}, {"converged", r.converged}, {"nconv", r.nconv},
                  {"restarts", r.restarts}, {"op_applies", r.op_applies}, {"total_s", t.total_s},
                  {"op_s", t.apply_s}, {"ortho_s", t.orthogonalize_s}, {"projected_s", t.projected_s},
                  {"update_s", t.update_s}, {"ortho_pct", 100.0 * t.orthogonalize_s / t.total_s},
                  {"loss_of_orthogonality", gram_deviation(r.schur_basis, r.schur_basis)}});
    } else {
      hs::eigen_detail::LanczosTimings t;
      const auto r = hs::eigen_detail::lanczos_krylov_schur_timed<T>(op, b_op.get(), hs::Which::LargestMagnitude, ko, t);
      out.update({{"solver", "lanczos"}, {"converged", r.converged}, {"nconv", r.nconv}, {"restarts", r.restarts},
                  {"op_applies", r.op_applies}, {"total_s", t.total}, {"op_s", t.op},
                  {"ortho_s", t.orthogonalization}, {"rayleigh_ritz_s", t.rayleigh_ritz},
                  {"restart_s", t.restart}, {"ortho_pct", 100.0 * t.orthogonalization / t.total},
                  {"theta0", r.theta.empty() ? 0.0 : r.theta[0]}});
      hs::DenseMatrix<T> mx = r.vectors;
      if (b_op) b_op->apply(r.vectors.data(), a.nrows, mx.data(), a.nrows, r.vectors.cols());
      out["loss_of_orthogonality"] = gram_deviation(r.vectors, mx);
    }
    std::printf("%s\n", out.dump().c_str());
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_ortho_shift_invert <case dir> [reps] [ncv] [nev]\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  const int reps = argc >= 3 ? std::atoi(argv[2]) : 3;
  const index_t ncv = argc >= 4 ? std::atoll(argv[3]) : 0;
  const auto meta = read_json(dir / "meta.json");
  const auto cfg = read_json(dir / "config.json");
  const std::string problem = meta.at("problem").get<std::string>();
  const double sigma = cfg.at("target").at("sigma").get<double>();
  const index_t nev_override = argc >= 5 ? std::atoll(argv[4]) : 0;
  const index_t k = nev_override > 0 ? nev_override : cfg.at("target").at("k").get<index_t>();
  const double tol = cfg.value("tol", 1e-10);
  const std::string name = dir.filename().string();
  hs::LoadedMatrix la = hs::load_matrix(dir / "A.npz");
  const bool has_b = std::filesystem::exists(dir / "B.npz");
  std::optional<hs::LoadedMatrix> lb;
  if (has_b) lb = hs::load_matrix(dir / "B.npz");
  if (la.is_complex() || (lb && lb->is_complex())) {
    auto as_complex = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<cplx>>(std::move(m.matrix))
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(m.matrix));
    };
    const hs::CsrMatrix<cplx> a = as_complex(la);
    std::optional<hs::CsrMatrix<cplx>> b;
    if (lb) b = as_complex(*lb);
    run<cplx>(name, problem, a, b ? &*b : nullptr, sigma, k, tol, reps, ncv);
  } else {
    const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
    run<double>(name, problem, a, b, sigma, k, tol, reps, ncv);
  }
  return 0;
}
