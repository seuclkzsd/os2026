// Measurement / regression probes for the driver's sliced and one-sided targets (P3 slicing review
// fixes). Each probe builds a small problem with a known spectrum, runs hs::solve and prints one
// JSON line per run (stdout): correctness against the reference (set error, duplicates,
// B-orthonormality), report (success, certificate, slices, warnings) and cost (wall and phase
// times, factorizations, solves).
//
//   bench_slicing_fix_probe PROBE [SEED]
//
// PROBE (finding of the review it reproduces):
//   mult_top_largest      largest 4 of a spectrum whose top eigenvalue is 10-fold      (F1)
//   cluster_k             smallest / largest k = 5, 6, 7 around a 6-fold eigenvalue  (F1, F2)
//   lap4                  30x30 Laplacian intervals around the 30-fold eigenvalue 4     (F3)
//   cluster_mid           40-fold 0.5 inside [0.2, 0.8], slice_size 10 and auto         (F3)
//   psd_smallest          smallest / largest 5 next to a 600-dim null space            (F4)
//   near_cluster          smallest 5 inside 40 close, distinct eigenvalues              (F4)
//   anchor                1-D Laplacian, interval [-1, w_i] (w_i an eigenvalue), double /
//                         single / BLR factors                                         (F6)
//   mult                  12 copies of 1.0 in [0.9, 1.15], 30 copies in [0.9, 1.1]      (F8)
//   largest_lap2d         largest / smallest 10 of a 100x100 Laplacian: counts         (F9)
//   nearest_few           nearest 30 with a zero band, only 20 nonzero eigenvalues     (F10)
//   ghiep_complex         GHIEP zero cluster, a complex pair between the side shifts    (F7)
//   ghiep_cplx_cluster    GHIEP zero cluster, complex Hermitian pencil (no complex pair)
//   ghiep_plain_edge      GHIEP zero cluster, side shifts beyond a plain run's disk edge (F5)
//   all                   every probe above
//
// PROBE_LOG=<dir> writes each run's JSON-lines event log to <dir>/<probe>.jsonl. Run it through
// tools/run/guard.py (docs/DESIGN.md §6).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/error.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/util.hpp"
#include "support/fixtures.cpp"  // test fixtures (known spectra) compiled into the probe

namespace {

using hs::cplx;
using hs::index_t;

std::uint64_t g_seed = 2026;

hs::SolverConfig base_cfg() {
  hs::SolverConfig cfg;
  cfg.tol = 1e-10;
  cfg.threads = 2;
  cfg.memory_budget_gb = 4.0;
  cfg.seed = g_seed;
  return cfg;
}

hs::SolverConfig interval_cfg(double lo, double hi, index_t slice_size = 0) {
  auto c = base_cfg();
  c.target.mode = hs::TargetMode::Interval;
  c.target.lower = lo;
  c.target.upper = hi;
  c.slice_size = slice_size;
  return c;
}

hs::SolverConfig end_cfg(hs::TargetMode m, index_t k, index_t slice_size = 0) {
  auto c = base_cfg();
  c.target.mode = m;
  c.target.k = k;
  c.slice_size = slice_size;
  return c;
}

std::vector<double> in_interval(const std::vector<double>& w, double lo, double hi) {
  std::vector<double> o;
  for (double x : w)
    if (x >= lo && x <= hi) o.push_back(x);
  return o;
}

// Largest relative difference of the sorted sets; infinity on a size mismatch.
double set_error(std::vector<double> got, std::vector<double> want) {
  if (got.size() != want.size()) return INFINITY;
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  double e = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i)
    e = std::max(e, std::abs(got[i] - want[i]) / std::max(1.0, std::abs(want[i])));
  return e;
}

// max |X^H B X - I| (B == nullptr: identity), dense; the probes are small.
template <class T>
double orth_defect(const hs::DenseMatrix<T>& x, const hs::CsrMatrix<T>* b) {
  const index_t n = x.rows(), m = x.cols();
  std::vector<T> bx(static_cast<std::size_t>(n));
  double worst = 0.0;
  for (index_t j = 0; j < m; ++j) {
    if (b) {
      hs::spmv(*b, x.col(j), bx.data());
    } else {
      std::copy(x.col(j), x.col(j) + n, bx.begin());
    }
    for (index_t i = 0; i < m; ++i) {
      T s = T(0);
      for (index_t r = 0; r < n; ++r) s += hs::conj_if(x(r, i)) * bx[r];
      worst = std::max(worst, std::abs(s - T(i == j ? 1.0 : 0.0)));
    }
  }
  return worst;
}

template <class T>
void run(const std::string& name, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b, const hs::SolverConfig& cfg,
         const std::vector<double>& want, bool check_orth = true) {
  nlohmann::json j;
  j["probe"] = name;
  j["seed"] = cfg.seed;
  hs::Timer wall;
  try {
    // PROBE_LOG=<dir>: JSON-lines event log per run (<dir>/<probe>.jsonl).
    std::unique_ptr<hs::Logger> log;
    if (const char* dir = std::getenv("PROBE_LOG"))
      log = std::make_unique<hs::Logger>(0, std::string(dir) + "/" + name + ".jsonl");
    const auto sol = hs::solve(a, b, cfg, log.get());
    const auto& r = sol.report;
    j["wall_s"] = wall.seconds();
    j["phases_s"] = {r.time_setup_s, r.time_factor_s, r.time_iterate_s, r.time_verify_s};
    j["total_s"] = r.time_total_s;
    j["success"] = r.success;
    j["m"] = sol.eigenvalues.size();
    j["want_m"] = want.size();
    j["set_error"] = set_error(sol.eigenvalues, want);
    std::vector<double> g = sol.eigenvalues;
    std::sort(g.begin(), g.end());
    index_t dup = 0;
    for (std::size_t i = 1; i < g.size(); ++i)
      dup += std::abs(g[i] - g[i - 1]) <= 1e-13 * std::max(1.0, std::abs(g[i]));
    j["equal_values"] = dup;
    if (check_orth && b == nullptr) j["orth_defect"] = orth_defect(sol.eigenvectors, b);
    if (check_orth && b != nullptr && r.problem != hs::ProblemType::GeneralizedIndefinite)
      j["orth_defect"] = orth_defect(sol.eigenvectors, b);
    j["cert"] = {r.certificate.count_inertia, r.certificate.count_found, r.certificate.passed};
    j["slices"] = r.slices.size();
    std::vector<double> shifts;
    for (const auto& sl : r.slices) shifts.push_back(sl.sigma);
    j["shifts"] = shifts;
    // Distance of the nearest slice shift to an eigenvalue of `want` (relative to |want|).
    double closest = INFINITY;
    for (double sh : shifts)
      for (double w : want) closest = std::min(closest, std::abs(sh - w) / std::max(1e-300, std::abs(w)));
    j["shift_rel_dist"] = closest;
    index_t reruns = 0;
    for (const auto& s : r.slices) reruns += s.reruns;
    j["reruns"] = reruns;
    j["factorizations"] = r.factor_stats.factorizations;
    j["solves"] = r.factor_stats.rhs_solved;
    double worst = 0.0;
    for (const auto& p : r.pairs) worst = std::max(worst, p.backward_error);
    j["worst_eta"] = worst;
    j["zero_filtered"] = r.zero_modes_filtered;
    j["zero_cluster_dim"] = r.zero_cluster_dim;
    if (!sol.complex_eigenvalues.empty()) {
      std::vector<std::vector<double>> z;
      for (auto c : sol.complex_eigenvalues) z.push_back({c.real(), c.imag()});
      j["complex"] = z;
    }
    if (sol.eigenvalues.size() <= 12) j["eigenvalues"] = sol.eigenvalues;
    j["warnings"] = r.warnings;
  } catch (const hs::Error& e) {
    j["wall_s"] = wall.seconds();
    j["error"] = e.what();
  }
  std::printf("%s\n", j.dump().c_str());
  std::fflush(stdout);
}

const hs::CsrMatrix<double>* const kNoB = nullptr;

// GHIEP pencil (A, B) = Y^T (D_A, D_B) Y with real eigenvalues lam (B signs +-1) and complex pairs
// x +- iy from 2x2 blocks A = [[x, y], [y, -x]], B = diag(1, -1); Y = I + coupling * dense.
struct Pencil {
  hs::CsrMatrix<double> a, b;
};
Pencil ghiep_with_complex(const std::vector<double>& lam, const std::vector<cplx>& pairs, double coupling) {
  const index_t nr = static_cast<index_t>(lam.size());
  const index_t n = nr + 2 * static_cast<index_t>(pairs.size());
  hs::DenseMatrix<double> da(n, n), db(n, n), y(n, n), ad(n, n), bd(n, n);
  for (index_t i = 0; i < nr; ++i) {
    const double s = (i % 3 == 0) ? -1.0 : 1.0;
    da(i, i) = lam[i] * s;
    db(i, i) = s;
  }
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const index_t i = nr + 2 * static_cast<index_t>(p);
    da(i, i) = pairs[p].real();
    da(i, i + 1) = da(i + 1, i) = pairs[p].imag();
    da(i + 1, i + 1) = -pairs[p].real();
    db(i, i) = 1.0;
    db(i + 1, i + 1) = -1.0;
  }
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j)
      y(i, j) = (i == j ? 1.0 : 0.0) + coupling * std::sin(1.7 * i + 0.9 * j * j + 0.1) / std::sqrt(double(n));
  // ad = Y^T da Y, bd = Y^T db Y (da, db have at most 2 entries per row).
  for (index_t r = 0; r < n; ++r)
    for (index_t c = r; c < n; ++c) {
      double sa = 0.0, sb = 0.0;
      for (index_t i = 0; i < n; ++i)
        for (index_t q = std::max<index_t>(0, i - 1); q <= std::min(n - 1, i + 1); ++q) {
          if (da(i, q) != 0.0) sa += y(i, r) * da(i, q) * y(q, c);
          if (db(i, q) != 0.0) sb += y(i, r) * db(i, q) * y(q, c);
        }
      ad(r, c) = ad(c, r) = sa;
      bd(r, c) = bd(c, r) = sb;
    }
  return {hs::test::from_dense(ad), hs::test::from_dense(bd)};
}

// The pencil of tests/cpp/test_driver_cluster.cpp (make_pencil): Y^H diag(lambda_i s_i) Y,
// Y^H diag(s_i) Y with s_i = -1 for i % 3 == 0, complex Y for T = cplx.
template <class T>
std::pair<hs::CsrMatrix<T>, hs::CsrMatrix<T>> congruent_pencil(const std::vector<double>& lam, double coupling) {
  const auto n = static_cast<index_t>(lam.size());
  hs::DenseMatrix<T> y(n, n), ad(n, n), bd(n, n);
  std::vector<double> sgn(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) {
    for (index_t j = 0; j < n; ++j) {
      const double re = std::sin(1.7 * i + 0.9 * j * j + 0.1), im = std::cos(0.7 * i * j + 0.3);
      if constexpr (hs::is_complex_v<T>) {
        y(i, j) = T(i == j ? 1.0 : 0.0) + coupling * T(re, im) / std::sqrt(2.0 * n);
      } else {
        y(i, j) = (i == j ? 1.0 : 0.0) + coupling * re / std::sqrt(double(n));
      }
    }
    sgn[i] = (i % 3 == 0) ? -1.0 : 1.0;
  }
  for (index_t r = 0; r < n; ++r)
    for (index_t c = r; c < n; ++c) {
      T sa = T(0), sb = T(0);
      for (index_t i = 0; i < n; ++i) {
        sa += hs::conj_if(y(i, r)) * (lam[i] * sgn[i]) * y(i, c);
        sb += hs::conj_if(y(i, r)) * sgn[i] * y(i, c);
      }
      ad(r, c) = sa;
      ad(c, r) = hs::conj_if(sa);
      bd(r, c) = sb;
      bd(c, r) = hs::conj_if(sb);
    }
  for (index_t r = 0; r < n; ++r) {
    ad(r, r) = T(std::real(ad(r, r)));
    bd(r, r) = T(std::real(bd(r, r)));
  }
  return {hs::test::from_dense(ad), hs::test::from_dense(bd)};
}

void probe(const std::string& p) {
  const bool all = p == "all";
  if (all || p == "mult_top_largest") {
    std::vector<double> lam(10, 3.0);
    for (int i = 1; i <= 300; ++i) lam.push_back(-0.01 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 41);
    run("mult_top_largest_k4", a, kNoB, end_cfg(hs::TargetMode::Largest, 4), std::vector<double>(4, 3.0));
  }
  if (all || p == "cluster_k") {
    std::vector<double> lam = {1.0, 1.0, 1.0, 1.5, 1.5};
    for (int i = 0; i < 6; ++i) lam.push_back(2.5);
    for (int i = 1; i <= 400; ++i) lam.push_back(2.5 + 0.37 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 48);
    for (index_t k : {5, 6, 7})
      run("cluster_smallest_k" + std::to_string(k), a, kNoB, end_cfg(hs::TargetMode::Smallest, k),
          std::vector<double>(lam.begin(), lam.begin() + k));
    std::vector<double> neg(lam);
    for (auto& x : neg) x = -x;
    const auto an = hs::test::prescribed_spectrum<double>(neg, 4, 49);
    for (index_t k : {5, 6, 7})
      run("cluster_largest_k" + std::to_string(k), an, kNoB, end_cfg(hs::TargetMode::Largest, k),
          std::vector<double>(neg.begin(), neg.begin() + k));
  }
  if (all || p == "lap4") {
    const auto a = hs::test::laplacian_2d<double>(30, 30);
    const auto w = hs::test::laplacian_2d_eigenvalues(30, 30);
    for (double h : {0.1, 0.15, 0.2})
      run("lap4_pm" + std::to_string(h).substr(0, 4), a, kNoB, interval_cfg(4.0 - h, 4.0 + h),
          in_interval(w, 4.0 - h - 1e-12, 4.0 + h + 1e-12));
  }
  if (all || p == "cluster_mid") {
    std::vector<double> lam(40, 0.5);
    for (int i = 0; i < 300; ++i) lam.push_back(-1.0 + 0.01 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 46);
    run("cluster_mid_s10", a, kNoB, interval_cfg(0.2, 0.8, 10), in_interval(lam, 0.2, 0.8));
    run("cluster_mid_auto", a, kNoB, interval_cfg(0.2, 0.8), in_interval(lam, 0.2, 0.8));
  }
  if (all || p == "psd_smallest") {
    std::vector<double> lam(600, 0.0);
    for (int i = 1; i <= 1400; ++i) lam.push_back(0.3 + 0.01 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 40);
    run("psd600_smallest5", a, kNoB, end_cfg(hs::TargetMode::Smallest, 5), std::vector<double>(5, 0.0), false);
    std::vector<double> neg(lam);
    for (auto& x : neg) x = -x;
    const auto an = hs::test::prescribed_spectrum<double>(neg, 4, 42);
    run("nsd600_largest5", an, kNoB, end_cfg(hs::TargetMode::Largest, 5), std::vector<double>(5, 0.0), false);
  }
  if (all || p == "near_cluster") {
    // The 5 smallest end inside 40 close but distinct eigenvalues (spacing 1e-6): not a
    // numerically equal cluster, so the partial-cluster shortcut must fall back to all 40.
    std::vector<double> lam = {0.2, 0.3, 0.4};
    for (int i = 0; i < 40; ++i) lam.push_back(1.0 + 1e-6 * i);
    for (int i = 1; i <= 300; ++i) lam.push_back(2.0 + 0.01 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 52);
    run("near_cluster_smallest5", a, kNoB, end_cfg(hs::TargetMode::Smallest, 5),
        std::vector<double>(lam.begin(), lam.begin() + 5));
  }
  if (all || p == "anchor") {
    struct Case {
      index_t n;
      int idx;
      hs::FactorPrecision prec;
      bool blr;
    };
    const Case cases[] = {{400, 9, hs::FactorPrecision::Double, false}, {400, 9, hs::FactorPrecision::Double, true},
                          {400, 19, hs::FactorPrecision::Double, true}, {400, 9, hs::FactorPrecision::Single, false},
                          {400, 4, hs::FactorPrecision::Single, false}, {1000, 15, hs::FactorPrecision::Single, false}};
    for (const auto& cs : cases) {
      const auto a = hs::test::laplacian_1d<double>(cs.n);
      const auto w = hs::test::laplacian_1d_eigenvalues(cs.n);
      auto cfg = interval_cfg(-1.0, w[cs.idx]);
      cfg.factor_precision = cs.prec;
      cfg.blr = cs.blr ? hs::Toggle::On : hs::Toggle::Off;
      const std::string name = "anchor_n" + std::to_string(cs.n) + "_w" + std::to_string(cs.idx) + "_" +
                               (cs.blr ? "blr" : cs.prec == hs::FactorPrecision::Single ? "single" : "double");
      run(name, a, kNoB, cfg, std::vector<double>(w.begin(), w.begin() + cs.idx + 1));
    }
  }
  if (all || p == "mult") {
    for (int copies : {12, 30}) {
      std::vector<double> lam(static_cast<std::size_t>(copies), 1.0);
      for (int i = 1; i <= 200; ++i) lam.push_back(0.5 + 0.0071 * i);
      const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 23);
      const double lo = 0.9, hi = copies == 12 ? 1.15 : 1.1;
      run("mult" + std::to_string(copies), a, kNoB, interval_cfg(lo, hi), in_interval(lam, lo, hi));
    }
  }
  if (all || p == "largest_lap2d") {
    const auto a = hs::test::laplacian_2d<double>(100, 100);
    const auto w = hs::test::laplacian_2d_eigenvalues(100, 100);
    run("largest10_lap2d100", a, kNoB, end_cfg(hs::TargetMode::Largest, 10), std::vector<double>(w.end() - 10, w.end()),
        false);
    run("smallest10_lap2d100", a, kNoB, end_cfg(hs::TargetMode::Smallest, 10),
        std::vector<double>(w.begin(), w.begin() + 10), false);
  }
  if (all || p == "nearest_few") {
    std::vector<double> lam(80, 0.0);
    for (int i = 1; i <= 20; ++i) lam.push_back(0.3 + 0.01 * i);
    const auto a = hs::test::prescribed_spectrum<double>(lam, 4, 39);
    auto cfg = base_cfg();
    cfg.target.sigma = 0.01;
    cfg.target.k = 30;
    cfg.zero_filter_tol = 1e-6;
    run("nearest_few", a, kNoB, cfg, std::vector<double>(lam.begin() + 80, lam.end()));
  }
  if (all || p == "ghiep_cplx_cluster") {
    // 30 zeros, 20 negative, 50 positive eigenvalues; complex Hermitian congruence; sigma = 0, k = 8.
    std::vector<double> lam(30, 0.0);
    for (int i = 0; i < 20; ++i) lam.push_back(-0.15 - 0.05 * i);
    for (int i = 0; i < 50; ++i) lam.push_back(0.2 + 0.07 * i);
    const auto [a, b] = congruent_pencil<cplx>(lam, 0.3);
    auto cfg = base_cfg();
    cfg.target.sigma = 0.0;
    cfg.target.k = 8;
    run("ghiep_cplx_cluster", a, &b, cfg, {-0.15, -0.2, -0.25, -0.3, 0.2, 0.27, 0.34, -0.35}, false);
  }
  if (all || p == "ghiep_plain_edge") {
    // A plain run at sigma = 0.01 converges cluster copies and real pairs out to a disk edge that
    // lies on a double eigenvalue; the side runs must be placed beyond it, not on it.
    std::vector<double> lam(30, 0.0);
    for (int i = 0; i < 20; ++i) lam.push_back(-0.15 - 0.05 * i);
    for (int i = 0; i < 50; ++i) lam.push_back(0.2 + 0.07 * i);
    for (int i = 0; i < 6; ++i) lam.push_back(0.2 + 0.07 * i);  // doubles: the disk ends on one
    const auto [a, b] = congruent_pencil<double>(lam, 0.3);
    for (index_t k : {8, 12, 16, 24}) {
      auto cfg = base_cfg();
      cfg.target.sigma = 0.01;
      cfg.target.k = k;
      cfg.zero_filter_tol = 1e-9;
      std::vector<double> nz;
      for (double x : lam)
        if (x != 0.0) nz.push_back(x);
      std::stable_sort(nz.begin(), nz.end(),
                       [](double x, double y) { return std::abs(x - 0.01) < std::abs(y - 0.01); });
      std::vector<double> want(nz.begin(), nz.begin() + k);
      run("ghiep_plain_edge_k" + std::to_string(k), a, &b, cfg, want, false);
    }
  }
  if (all || p == "ghiep_complex") {
    std::vector<double> lam(30, 0.0);
    for (int i = 0; i < 20; ++i) lam.push_back(-0.15 - 0.05 * i);
    for (int i = 0; i < 40; ++i) lam.push_back(0.2 + 0.07 * i);
    const auto pen = ghiep_with_complex(lam, {cplx(0.02, 0.1)}, 0.3);
    for (double sigma : {1e-3, 0.05}) {  // (sigma = 0 ties 0.2 with -0.2)
      auto cfg = base_cfg();
      cfg.target.sigma = sigma;
      cfg.target.k = 4;
      // Expected: 0.02 +- 0.1i (complex), -0.15, 0.2 (real).
      run("ghiep_complex_sigma" + std::to_string(sigma).substr(0, 5), pen.a, &pen.b, cfg, {-0.15, 0.2}, false);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_slicing_fix_probe PROBE [SEED]\n");
    return 2;
  }
  if (argc > 2) g_seed = std::strtoull(argv[2], nullptr, 10);
  probe(argv[1]);
  return 0;
}
