// The level-1 pattern of the B proof (module anaint, P5c fix; src/driver/norms.cpp): its O(n)
// bounds (Level1Size: visits of a construction pass, entries bound) against the pattern built -
// exact entries, time of the first pass and of the whole construction - and the predictions
// hpd_count_seconds / hpd_fill_seconds. The time model's constants (kCount*, kFill*; thread
// exponents from --threads 16, 4, 1) are fitted on this output by nonnegative least squares on
// the relative error; the arrowheads' patterns are quadratic.
//
//   bench_anaint_fix_bproof (--case DIR | --arrow N | --stencil K) [--threads 16] [--reps 3]
//                           [--proof 1]
//
// --case: B.npz of a data case; --arrow N: the order-N arrowhead b_ii = 1, b_0j = 0.632 /
// sqrt(N - 1) (one dense column: n^2 / 2 level-1 entries); --stencil K: the 27-point stencil on
// a K^3 grid (HPD, not diagonally dominant). Output: one JSON line - n, nnz_b, visits,
// entries_bound, entries, seconds (median of --reps constructions), predicted_s; with --proof 1
// also prove_positive_definite with the driver's level order and no limits (outcome, pattern_s,
// attempts_s, pattern_predicted_s). The constants quoted in norms.cpp come from the gpu02
// measurement slot (16 threads; the outer guard holds the slot, nested ones share it):
//   python3 tools/run/guard.py --exclusive --timeout 900 -- bash -c '
//     for c in <cases with B>; do build/release/bench_anaint_fix_bproof --case data/cases/$c; done;
//     for n in 2000 5000 10000 20000; do build/release/bench_anaint_fix_bproof --arrow $n; done;
//     for k in 20 40 60; do build/release/bench_anaint_fix_bproof --stencil $k; done'
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "driver/norms.hpp"
#include "hsolve/error.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

hs::CsrMatrix<double> arrowhead(index_t n) {
  const double c = 0.632 / std::sqrt(static_cast<double>(n - 1));
  std::vector<index_t> r, cc;
  std::vector<double> v;
  for (index_t i = 0; i < n; ++i) r.push_back(i), cc.push_back(i), v.push_back(1.0);
  for (index_t j = 1; j < n; ++j) {
    r.push_back(0), cc.push_back(j), v.push_back(c);
    r.push_back(j), cc.push_back(0), v.push_back(c);
  }
  return hs::csr_from_coo<double>(n, n, r, cc, v);
}

/// 27-point stencil: 1 on the diagonal, -1/27 to the 26 neighbours (HPD; strictly diagonally
/// dominant, which the driver's screen would catch first - only the pattern matters here).
hs::CsrMatrix<double> stencil27(index_t k) {
  std::vector<index_t> r, c;
  std::vector<double> v;
  auto id = [&](index_t i, index_t j, index_t l) { return i + k * (j + k * l); };
  for (index_t l = 0; l < k; ++l)
    for (index_t j = 0; j < k; ++j)
      for (index_t i = 0; i < k; ++i)
        for (index_t dl = -1; dl <= 1; ++dl)
          for (index_t dj = -1; dj <= 1; ++dj)
            for (index_t di = -1; di <= 1; ++di) {
              const index_t ii = i + di, jj = j + dj, ll = l + dl;
              if (ii < 0 || jj < 0 || ll < 0 || ii >= k || jj >= k || ll >= k) continue;
              r.push_back(id(i, j, l));
              c.push_back(id(ii, jj, ll));
              v.push_back(di == 0 && dj == 0 && dl == 0 ? 1.0 : -1.0 / 27.0);
            }
  const index_t n = k * k * k;
  return hs::csr_from_coo<double>(n, n, r, c, v);
}

template <class T>
json measure(const hs::CsrMatrix<T>& b, int threads, int reps, bool proof) {
  json out = {{"n", b.nrows}, {"nnz_b", b.nnz()}, {"complex", hs::is_complex_v<T>}, {"threads", threads}};
  std::vector<double> secs, count;
  hs::driver::Level1Size s;
  for (int r = 0; r < std::max(1, reps); ++r) {
    s = hs::driver::level1_pattern_size(b, true);
    secs.push_back(s.seconds);
    count.push_back(s.count_seconds);
  }
  s.seconds = median(secs);
  out["visits"] = s.visits;
  out["entries_bound"] = s.entries_bound;
  out["entries"] = s.entries;
  out["seconds"] = s.seconds;
  out["count_seconds"] = median(count);
  out["count_predicted_s"] = hs::driver::hpd_count_seconds(b.nrows, s.visits, threads);
  out["fill_predicted_s"] = hs::driver::hpd_fill_seconds(static_cast<double>(s.entries), threads);
  if (proof) {
    const auto p = hs::driver::prove_positive_definite(b);
    out["proof"] = {{"outcome", p.outcome},       {"level", p.level},
                    {"seconds", p.seconds},       {"pattern_s", p.pattern_s},
                    {"attempts_s", p.attempts_s}, {"pattern_predicted_s", p.pattern_predicted_s},
                    {"theta_min", p.theta_min}};
    // The attempts' predicted times on both levels (the proof's gates use them).
    for (const int level : {0, 1}) {
      const auto ib = hs::driver::incomplete_cholesky_bound(b, level, 0.9 * std::max(p.theta_min, 0.0));
      out["proof"]["attempt" + std::to_string(level) + "_predicted_s"] =
          hs::driver::hpd_attempt_seconds(b.nrows, ib.work, threads);
      out["proof"]["attempt" + std::to_string(level) + "_s"] = ib.factor_s + ib.check_s;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string case_dir;
  index_t arrow = 0, stencil = 0;
  int threads = 16, reps = 3;
  bool proof = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        return 2;
      }
      const std::string v = argv[++i];
      if (k == "--case")
        case_dir = v;
      else if (k == "--arrow")
        arrow = std::stoll(v);
      else if (k == "--stencil")
        stencil = std::stoll(v);
      else if (k == "--threads")
        threads = std::stoi(v);
      else if (k == "--reps")
        reps = std::stoi(v);
      else if (k == "--proof")
        proof = std::stoi(v) != 0;
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    }
    omp_set_num_threads(threads);
    json out;
    if (!case_dir.empty()) {
      auto lb = hs::load_matrix(fs::path(case_dir) / "B.npz");
      out = lb.is_complex() ? measure(std::get<hs::CsrMatrix<hs::cplx>>(lb.matrix), threads, reps, proof)
                            : measure(std::get<hs::CsrMatrix<double>>(lb.matrix), threads, reps, proof);
      out["case"] = fs::path(case_dir).filename().string();
    } else if (arrow > 1) {
      out = measure(arrowhead(arrow), threads, reps, proof);
      out["case"] = "arrow-" + std::to_string(arrow);
    } else if (stencil > 1) {
      out = measure(stencil27(stencil), threads, reps, proof);
      out["case"] = "stencil27-" + std::to_string(stencil);
    } else {
      std::cerr << "usage: bench_anaint_fix_bproof (--case DIR | --arrow N | --stencil K) [...]\n";
      return 2;
    }
    std::cout << out.dump() << std::endl;
  } catch (const hs::Error& e) {
    // B without a positive diagonal (GHIEP cases): no proof is attempted on it.
    std::cout << json{{"case", case_dir}, {"skipped", e.what()}}.dump() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
