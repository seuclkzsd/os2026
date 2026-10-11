// Synthetic robustness study of jacobi_davidson (fix round of module lobpcg_jd): small problems
// with known spectra where the target lies ON an eigenvalue (simple, 25-fold, an exact null
// space) and where an eigenvalue of multiplicity >= nev lies next to the target, run with the
// extraction / start-iteration options given, real and complex, with and without a
// preconditioner (float-rounded dense inverse of A - s I: what a single-precision factor
// factorizes). One JSON line per run on stdout, then a summary line per (extraction,
// preconditioner) group: runs, failures (not converged or not the nearest set), solves.
//
//   bench_lobpcg_jd_fix_synthetic [--start N] [--extraction auto|harmonic|ritz|both]
//                                 [--set tau|mult|all]
// --start: JdOptions::start_iterations (-1 = library default); --extraction both = harmonic and
// ritz. Run (2 threads, a few seconds):
//   python3 tools/run/guard.py --skip-busy-check --cores 30,33 --mem-gb 4 --timeout 900 --
//       build/release/bench_lobpcg_jd_fix_synthetic --start 0 --extraction both
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/eigen_iterative.hpp"
#include "hsolve/error.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "linalg/blas_lapack.hpp"
#include "support/fixtures.hpp"
// The bench targets do not link hsolve_test_support (CMakeLists.txt is lead-owned), so the
// fixture generators (prescribed_spectrum, laplacian_2d, DenseOperator, ...) are compiled in.
#include "support/fixtures.cpp"  // NOLINT(bugprone-suspicious-include)

using json = nlohmann::json;

namespace {

using hs::DenseMatrix;
using hs::index_t;
using hs::JdExtraction;

template <class T>
DenseMatrix<T> float_rounded_inverse(const hs::CsrMatrix<T>& a, double shift) {
  DenseMatrix<T> k = hs::test::to_dense(a);
  const auto n = static_cast<hs::lapack::fint>(k.rows());
  for (index_t i = 0; i < n; ++i) k(i, i) -= T(shift);
  for (index_t i = 0; i < k.rows() * k.cols(); ++i) {
    if constexpr (hs::is_complex_v<T>) {
      k.data()[i] = T(static_cast<float>(k.data()[i].real()), static_cast<float>(k.data()[i].imag()));
    } else {
      k.data()[i] = static_cast<float>(k.data()[i]);
    }
  }
  std::vector<hs::lapack::fint> piv(static_cast<std::size_t>(n));
  if (hs::lapack::getrf(n, k.data(), n, piv.data()) != 0) HS_THROW(Singular, "singular A - s I");
  DenseMatrix<T> inv(n, n);
  for (index_t i = 0; i < n; ++i) inv(i, i) = T(1);
  hs::lapack::getrs(n, n, k.data(), n, piv.data(), inv.data(), n);
  return inv;
}

/// max | |theta_i - tau| - |lambda_i - tau| | over the sorted distances (ties on both sides of
/// tau count as correct).
double distance_error(const std::vector<double>& theta, std::vector<double> spectrum, double tau) {
  std::vector<double> dg, dw;
  for (double v : theta) dg.push_back(std::abs(v - tau));
  for (double v : spectrum) dw.push_back(std::abs(v - tau));
  std::sort(dg.begin(), dg.end());
  std::sort(dw.begin(), dw.end());
  double e = 0.0;
  for (std::size_t i = 0; i < dg.size(); ++i) e = std::max(e, std::abs(dg[i] - dw[i]));
  return e;
}

struct Group {
  int runs = 0, failures = 0;
  long long solves = 0, iterations = 0;
};

struct Study {
  long long start = -1;
  std::vector<JdExtraction> extractions;
  std::map<std::string, Group> groups;

  template <class T>
  void run(const std::string& name, const hs::CsrMatrix<T>& a, const std::vector<double>& spectrum, double tau,
           index_t nev, bool with_prec, double prec_shift, std::uint64_t seed = 2026, index_t max_iterations = 3000) {
    hs::CsrOperator<T> op(a);
    std::unique_ptr<hs::test::DenseOperator<T>> prec;
    if (with_prec) prec = std::make_unique<hs::test::DenseOperator<T>>(float_rounded_inverse(a, prec_shift));
    for (JdExtraction ex : extractions) {
      hs::JdOptions<T> o;
      o.nev = nev;
      o.tol = 1e-11;
      o.extraction = ex;
      o.max_iterations = max_iterations;
      o.seed = seed;
      if (start >= 0) o.start_iterations = start;
      const auto r = hs::jacobi_davidson<T>(op, nullptr, prec.get(), tau, o);
      const double err = distance_error(r.theta, spectrum, tau);
      const bool ok = r.converged && err <= 1e-8;
      const std::string ex_name = ex == JdExtraction::Ritz ? "ritz" : (ex == JdExtraction::Harmonic ? "harmonic" : "auto");
      json line = {{"problem", name},
                   {"scalar", hs::is_complex_v<T> ? "complex" : "real"},
                   {"extraction", ex_name},
                   {"preconditioner", with_prec},
                   {"start_iterations", static_cast<long long>(o.start_iterations)},
                   {"converged", r.converged},
                   {"nearest_set", err <= 1e-8},
                   {"distance_error", err},
                   {"iterations", static_cast<long long>(r.iterations)},
                   {"solves", static_cast<long long>(r.precond_applies)},
                   {"theta", r.theta}};
      std::cout << line.dump() << "\n";
      Group& g = groups[ex_name + (with_prec ? "/preconditioned" : "/plain")];
      ++g.runs;
      g.failures += ok ? 0 : 1;
      g.solves += r.precond_applies;
      g.iterations += r.iterations;
    }
  }

  template <class T>
  void target_on_eigenvalue() {
    {  // 25 x 25 grid Laplacian, tau = 4: multiplicity 25
      const auto a = hs::test::laplacian_2d<T>(25, 25);
      run<T>("lap2d-25 tau=4 (25-fold) nev=6", a, hs::test::laplacian_2d_eigenvalues(25, 25), 4.0, 6, true, 4.0 + 1e-3);
    }
    {  // -3:0.02:3, tau = the eigenvalue 0
      std::vector<double> s;
      for (int i = 0; i < 300; ++i) s.push_back(-3.0 + 0.02 * i);
      const auto a = hs::test::prescribed_spectrum<T>(s, 6, 3);
      for (bool p : {false, true}) run<T>("grid tau=eigenvalue nev=5", a, s, s[150], 5, p, s[150] + 1e-3);
    }
    {  // 10 exact zeros + 0.5:0.02:..., tau = 0
      std::vector<double> s(10, 0.0);
      for (int i = 0; s.size() < 300; ++i) s.push_back(0.5 + 0.02 * i);
      const auto a = hs::test::prescribed_spectrum<T>(s, 6, 4);
      for (index_t nev : {6, 12})
        run<T>("null space tau=0 nev=" + std::to_string(nev), a, s, 0.0, nev, true, -1e-3);
    }
    for (double delta : {0.0, 1e-15, 1e-13, 1e-12}) {  // one eigenvalue at 0.3 + delta
      std::vector<double> s;
      for (int i = 0; i < 299; ++i) s.push_back(-3.0 + 0.0200384 * i);
      s.push_back(0.3 + delta);
      const auto a = hs::test::prescribed_spectrum<T>(s, 6, 5);
      char nm[64];
      std::snprintf(nm, sizeof nm, "simple at tau+%g nev=3", delta);
      run<T>(nm, a, s, 0.3, 3, true, 0.3 - 1e-3);
    }
    {  // floor(i / 3): triple eigenvalues, tau = 50 (no preconditioner)
      std::vector<double> s;
      for (int i = 0; i < 300; ++i) s.push_back(static_cast<double>(i / 3));
      const auto a = hs::test::prescribed_spectrum<T>(s, 1, 6);
      run<T>("triples tau=50 nev=4", a, s, 50.0, 4, false, 0.0);
    }
  }

  template <class T>
  void multiplicity() {
    for (index_t mult : {5, 6, 8, 10}) {  // {0.5 x mult, 1.0, 1.01, ...}, tau = 0.45 below them
      std::vector<double> w(static_cast<std::size_t>(mult), 0.5);
      for (index_t i = 0; i < 300 - mult; ++i) w.push_back(1.0 + 0.01 * static_cast<double>(i));
      const auto a = hs::test::prescribed_spectrum<T>(w, 8, 3);
      for (bool p : {false, true})
        for (std::uint64_t seed : {2026ULL, 7ULL})
          run<T>("multiplicity " + std::to_string(mult) + " tau=0.45 nev=6", a, w, 0.45, 6, p, 0.45, seed, 2000);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    Study st;
    std::string extraction = "both", set = "all";
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (i + 1 >= argc) HS_THROW(InvalidInput, "missing value for " << key);
      const std::string val = argv[++i];
      if (key == "--start") st.start = std::stoll(val);
      else if (key == "--extraction") extraction = val;
      else if (key == "--set") set = val;
      else HS_THROW(InvalidInput, "unknown option " << key);
    }
    if (extraction == "both") st.extractions = {JdExtraction::Harmonic, JdExtraction::Ritz};
    else if (extraction == "auto") st.extractions = {JdExtraction::Auto};
    else if (extraction == "harmonic") st.extractions = {JdExtraction::Harmonic};
    else if (extraction == "ritz") st.extractions = {JdExtraction::Ritz};
    else HS_THROW(InvalidInput, "unknown extraction " << extraction);
    if (set == "tau" || set == "all") {
      st.target_on_eigenvalue<double>();
      st.target_on_eigenvalue<hs::cplx>();
    }
    if (set == "mult" || set == "all") {
      st.multiplicity<double>();
      st.multiplicity<hs::cplx>();
    }
    for (const auto& [name, g] : st.groups)
      std::cout << json({{"summary", name},
                         {"start_iterations",
                          st.start >= 0 ? st.start : static_cast<long long>(hs::JdOptions<double>().start_iterations)},
                         {"runs", g.runs},
                         {"failures", g.failures},
                         {"solves", g.solves},
                         {"iterations", g.iterations}})
                       .dump()
                << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_lobpcg_jd_fix_synthetic: %s\n", e.what());
    return 2;
  }
}
