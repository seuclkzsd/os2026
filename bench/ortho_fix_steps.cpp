// Fixed-step time split of the real shift-invert eigensolver path (review of the ortho module).
// Same setup as bench/ortho_shift_invert.cpp (load A and B, factorize K = A - sigma B with MUMPS
// at the case's shift, run the driver's Krylov solver on Op = K^{-1} B: Lanczos in the B inner
// product for HEP/GHEP, general Krylov-Schur for GHIEP), but with tol = 1e-300 and a fixed
// max_restarts: no pair converges, so every library version performs exactly the same number of
// steps (restarts + 1 cycles, the same number of operator applications), and the times compare
// the same computation. A converging run can change its restart count with the last bits of the
// arithmetic (wg-square-1e5, nev = 60: 245 vs 214 applications between two library versions),
// which then mixes a different amount of work into a "speedup".
// Prints one JSON line per repetition (phase times, per-application times, loss of orthogonality
// of the returned basis). Built against two library versions it gives a before/after comparison.
// Run under the guard, e.g. (one line)
//   python3 tools/run/guard.py --skip-busy-check --cores 26,27 --mem-gb 6 --timeout 900 --
//       build/release/bench_ortho_fix_steps $HS_DATA_ROOT/cases/wg-square-1e5 3 1 60
// Arguments: <case dir> <restarts> [repetitions = 1] [nev = the case's k] [ncv = 0 (default)]
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
         double sigma, index_t nev, index_t ncv, index_t restarts, int reps) {
  hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;  // driver defaults
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer tf;
  fac->analyze(pencil, 1.0, -sigma);
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
  ko.nev = nev;
  ko.ncv = ncv;
  ko.tol = 1e-300;  // nothing converges: exactly restarts + 1 cycles
  ko.max_restarts = restarts;
  ko.seed = 2026;
  for (int rep = 0; rep < reps; ++rep) {
    nlohmann::json out = {{"case", name},       {"problem", problem}, {"n", a.nrows}, {"complex", hs::is_complex_v<T>},
                          {"nev", nev},         {"sigma", used},      {"threads", hs::max_threads()},
                          {"factor_s", factor_s}, {"rep", rep}};
    double ortho = 0.0, opt = 0.0, total = 0.0;
    index_t applies = 0;
    if (problem == "GHIEP") {
      hs::eigen_detail::GeneralPhaseTimes t;
      const auto r = hs::eigen_detail::krylov_schur_general_timed<T>(op, hs::Which::LargestMagnitude, ko, &t);
      ortho = t.orthogonalize_s;
      opt = t.apply_s;
      total = t.total_s;
      applies = r.op_applies;
      out.update({{"solver", "krylov_schur_general"}, {"nconv", r.nconv}, {"restarts", r.restarts},
                  {"projected_s", t.projected_s}, {"update_s", t.update_s},
                  {"loss_of_orthogonality", gram_deviation(r.schur_basis, r.schur_basis)}});
    } else {
      hs::eigen_detail::LanczosTimings t;
      const auto r = hs::eigen_detail::lanczos_krylov_schur_timed<T>(op, b_op.get(), hs::Which::LargestMagnitude, ko, t);
      ortho = t.orthogonalization;
      opt = t.op;
      total = t.total;
      applies = r.op_applies;
      hs::DenseMatrix<T> mx = r.vectors;
      if (b_op) b_op->apply(r.vectors.data(), a.nrows, mx.data(), a.nrows, r.vectors.cols());
      out.update({{"solver", "lanczos"}, {"nconv", r.nconv}, {"restarts", r.restarts},
                  {"rayleigh_ritz_s", t.rayleigh_ritz}, {"restart_s", t.restart},
                  {"loss_of_orthogonality", gram_deviation(r.vectors, mx)}});
    }
    out.update({{"op_applies", applies}, {"total_s", total}, {"op_s", opt}, {"ortho_s", ortho},
                {"ortho_ms_per_apply", 1e3 * ortho / static_cast<double>(std::max<index_t>(applies, 1))},
                {"total_ms_per_apply", 1e3 * total / static_cast<double>(std::max<index_t>(applies, 1))},
                {"ortho_pct", 100.0 * ortho / total}});
    std::printf("%s\n", out.dump().c_str());
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_ortho_fix_steps <case dir> <restarts> [reps] [nev] [ncv]\n");
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  const index_t restarts = std::atoll(argv[2]);
  const int reps = argc >= 4 ? std::atoi(argv[3]) : 1;
  const auto meta = read_json(dir / "meta.json");
  const auto cfg = read_json(dir / "config.json");
  const std::string problem = meta.at("problem").get<std::string>();
  const double sigma = cfg.at("target").at("sigma").get<double>();
  const index_t nev_arg = argc >= 5 ? std::atoll(argv[4]) : 0;
  const index_t nev = nev_arg > 0 ? nev_arg : cfg.at("target").at("k").get<index_t>();
  const index_t ncv = argc >= 6 ? std::atoll(argv[5]) : 0;
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
    run<cplx>(name, problem, a, b ? &*b : nullptr, sigma, nev, ncv, restarts, reps);
  } else {
    const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
    const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
    run<double>(name, problem, a, b, sigma, nev, ncv, restarts, reps);
  }
  return 0;
}
