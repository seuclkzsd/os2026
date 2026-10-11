// Iterative refinement of the shift-invert apply Y = (A - sigma B)^{-1} B X (module ops, P4
// "solvephase"): the three P4 changes measured one at a time, against a replica of the P1 operator
// (base a8ed8f4), at equal accuracy.
//
// Arms (same factorization; repetitions interleaved over p and arms, the arm order rotated from
// one repetition to the next; each sample the mean of --inner applies):
//   p1        replica of the P1 ShiftInvertOperator::do_apply: B X and the residual (three spmm
//             passes) on KernelPath::Reference, every correction solves all p columns (zero
//             right-hand sides for the finished ones), no rounding-level stop
//   p1_auto   the same loop with KernelPath::Auto kernels (fused one-sweep residual)
//                                                   -> p1 / p1_auto = the kernel effect
//   active    hs::ShiftInvertOperator, Auto kernels, stop_at_rounding_level = false: corrections
//             solve only the still-active columns   -> p1_auto / active = the active-column effect
//   p4        hs::ShiftInvertOperator, Auto kernels, stop_at_rounding_level = true (the default)
//                                                   -> active / p4 = the rounding-level stop
// Per arm: ms per apply and per vector, corrections and right-hand sides solved per apply, and of
// the returned block the largest ||r|| / ||B x|| and componentwise backward error (recomputed
// with the Reference residual, the same for every arm) and the largest relative deviation from
// the p1 arm's block.
// The driver enables refinement only for Single / BLR factors (src/driver/definite.cpp,
// indefinite.cpp): --precision single is the production configuration; --precision double
// measures an unattainable --ref-tol with a double factor.
// Output: one JSON object per line ("kind": "setup" | "rep" | "median").
//
// Usage (via the resource guard; threads = cores given):
//   python3 tools/run/guard.py --skip-busy-check --cores 34,35,36,37 --mem-gb 6 --timeout 1800 --
//     build/release/bench_solvephase_fix_refine CASE_DIR [--ordering amd|scotch|metis|auto]
//     [--precision double|single] [--p 1,8] [--reps 7] [--inner 3] [--ref-tol 1e-12]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/operator.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using hs::KernelPath;
using json = nlohmann::json;

struct Args {
  std::filesystem::path dir;
  std::string ordering = "auto";
  bool single = false;
  std::vector<index_t> p = {1, 8};
  int reps = 7;
  int inner = 3;
  double ref_tol = 1e-12;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

Args parse_args(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("usage: bench_solvephase_fix_refine CASE_DIR [options]");
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
    const std::string v = argv[++i];
    if (s == "--ordering")
      a.ordering = v;
    else if (s == "--precision")
      a.single = v == "single";
    else if (s == "--p")
      a.p = parse_list(v);
    else if (s == "--reps")
      a.reps = std::stoi(v);
    else if (s == "--inner")
      a.inner = std::stoi(v);
    else if (s == "--ref-tol")
      a.ref_tol = std::stod(v);
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

int ordering_code(const std::string& s) {
  if (s == "amd") return 0;
  if (s == "scotch") return 3;
  if (s == "metis") return 5;
  if (s == "auto") return 7;
  throw std::runtime_error("unknown ordering " + s);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/// Replica of the P1 refinement operator (ShiftInvertOperator::do_apply at base a8ed8f4), with the
/// sparse products on a chosen KernelPath: Reference reproduces P1 exactly (its spmm was the
/// Reference loop, its residual the three passes shifted_residual's Reference path computes).
template <class T>
class P1Refinement {
 public:
  P1Refinement(hs::SparseFactorization<T>& f, const hs::ShiftedPencil<T>& pencil, double sigma,
               hs::SolveRefinement refine, KernelPath path)
      : f_(f), pencil_(pencil), sigma_(sigma), refine_(refine), kernels_{path, &scratch_} {}
  index_t refinement_steps() const { return steps_; }

  void apply(const T* x, index_t ldx, T* y, index_t ldy, index_t p) {
    const index_t n = pencil_.order();
    const CsrMatrix<T>* b = pencil_.b();
    if (b) {
      hs::spmm(*b, x, ldx, y, ldy, p, T(1), T(0), kernels_);
    } else {
      for (index_t k = 0; k < p; ++k) std::copy(x + k * ldx, x + k * ldx + n, y + k * ldy);
    }
    const std::size_t len = static_cast<std::size_t>(n * p);
    if (r_.size() < len) {
      r_.resize(len);
      c_.resize(len);
    }
    T* r = r_.data();
    T* c = c_.data();
    std::vector<double> bx_norm(static_cast<std::size_t>(p));
    for (index_t k = 0; k < p; ++k) bx_norm[k] = hs::linalg::norm2(n, y + k * ldy);
    auto relative = [&](index_t k) {
      const double rn = hs::linalg::norm2(n, r + k * n);
      if (rn == 0.0) return 0.0;
      return bx_norm[k] > 0.0 ? rn / bx_norm[k] : std::numeric_limits<double>::infinity();
    };
    auto residual = [&](const T* yy, index_t ldyy) {
      hs::shifted_residual(pencil_.a(), b, sigma_, x, ldx, yy, ldyy, r, n, p, kernels_);
    };
    f_.solve(y, p, ldy);
    residual(y, ldy);
    std::vector<double> rel(static_cast<std::size_t>(p));
    std::vector<char> active(static_cast<std::size_t>(p));
    index_t remaining = 0;
    for (index_t k = 0; k < p; ++k) {
      rel[k] = relative(k);
      active[k] = rel[k] > refine_.rel_tol;
      remaining += active[k];
    }
    for (int step = 0; step < refine_.max_steps && remaining > 0; ++step) {
      for (index_t k = 0; k < p; ++k) {
        if (active[k]) {
          std::copy(r + k * n, r + (k + 1) * n, c + k * n);
        } else {
          std::fill(c + k * n, c + (k + 1) * n, T(0));
        }
      }
      f_.solve(c, p, n);
      ++steps_;
      for (index_t k = 0; k < p; ++k) hs::linalg::axpy(n, T(1), y + k * ldy, c + k * n);
      residual(c, n);
      for (index_t k = 0; k < p; ++k) {
        if (!active[k]) continue;
        const double candidate_rel = relative(k);
        if (candidate_rel < rel[k]) {
          std::copy(c + k * n, c + (k + 1) * n, y + k * ldy);
          rel[k] = candidate_rel;
          if (candidate_rel > refine_.rel_tol) continue;
        }
        active[k] = false;
        --remaining;
      }
    }
  }

 private:
  hs::SparseFactorization<T>& f_;
  const hs::ShiftedPencil<T>& pencil_;
  double sigma_;
  hs::SolveRefinement refine_;
  std::vector<double> scratch_;
  hs::KernelOptions kernels_;
  std::vector<T> r_, c_;
  index_t steps_ = 0;
};

template <class T>
void run(const Args& args, const std::string& name, const CsrMatrix<T>& a, const CsrMatrix<T>* b,
         double sigma0) {
  const index_t n = a.nrows;
  hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;
  fo.ordering = ordering_code(args.ordering);
  fo.deterministic = false;  // driver default (L0-OpenMP)
  fo.threads = hs::max_threads();
  if (args.single) fo.precision = hs::FactorPrecision::Single;
  auto fac = hs::make_mumps_factorization<T>(fo);
  fac->analyze(pencil, 1.0, -sigma0);
  double sigma = sigma0;
  for (int r = 0; !fac->factorize(1.0, -sigma); ++r) {
    if (r == 3) throw std::runtime_error("no regular shift near sigma");
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
  }
  json setup = {
      {"kind", "setup"},
      {"case", name},
      {"n", n},
      {"complex", hs::is_complex_v<T>},
      {"threads", hs::max_threads()},
      {"precision", args.single ? "single" : "double"},
      {"ref_tol", args.ref_tol},
      {"sigma", sigma},
      {"kernel_path",
       hs::resolve_kernel_path(KernelPath::Auto) == KernelPath::Avx512 ? "avx512" : "portable"}};
  std::printf("%s\n", setup.dump().c_str());
  std::fflush(stdout);

  hs::SolveRefinement refine;
  refine.max_steps = 10;
  refine.rel_tol = args.ref_tol;
  hs::SolveRefinement no_stop = refine;
  no_stop.stop_at_rounding_level = false;
  P1Refinement<T> p1(*fac, pencil, sigma, refine, KernelPath::Reference);
  P1Refinement<T> p1_auto(*fac, pencil, sigma, refine, KernelPath::Auto);
  hs::ShiftInvertOperator<T> active(*fac, pencil, sigma, no_stop), p4(*fac, pencil, sigma, refine);
  struct Arm {
    std::string name;
    std::function<void(const T*, T*, index_t)> apply;
    std::function<index_t()> steps;
  };
  const std::vector<Arm> arms = {
      {"p1", [&](const T* x, T* y, index_t p) { p1.apply(x, n, y, n, p); },
       [&] { return p1.refinement_steps(); }},
      {"p1_auto", [&](const T* x, T* y, index_t p) { p1_auto.apply(x, n, y, n, p); },
       [&] { return p1_auto.refinement_steps(); }},
      {"active", [&](const T* x, T* y, index_t p) { active.apply(x, n, y, n, p); },
       [&] { return active.refinement_steps(); }},
      {"p4", [&](const T* x, T* y, index_t p) { p4.apply(x, n, y, n, p); },
       [&] { return p4.refinement_steps(); }},
  };

  const index_t pmax = *std::max_element(args.p.begin(), args.p.end());
  std::vector<T> x(static_cast<std::size_t>(n * pmax)), r(x.size());
  for (index_t j = 0; j < pmax; ++j)
    hs::linalg::fill_random(x.data() + j * n, n, 2026, static_cast<std::uint64_t>(j));
  std::map<std::string, std::vector<T>> out;
  for (const auto& arm : arms) out[arm.name].resize(x.size());
  // [p][arm][metric] -> samples
  std::map<index_t, std::map<std::string, std::map<std::string, std::vector<double>>>> s;
  const std::size_t na = arms.size();
  for (int rep = 0; rep < args.reps; ++rep) {
    for (index_t p : args.p) {
      json rec = {{"kind", "rep"}, {"case", name}, {"rep", rep}, {"p", p}};
      for (std::size_t o = 0; o < na; ++o) {
        const Arm& arm = arms[(o + static_cast<std::size_t>(rep)) % na];
        T* y = out[arm.name].data();
        const index_t steps0 = arm.steps(), rhs0 = fac->stats().rhs_solved;
        hs::Timer t;
        for (int i = 0; i < args.inner; ++i) arm.apply(x.data(), y, p);
        const double ms = 1e3 * t.seconds() / args.inner;
        auto& m = s[p][arm.name];
        m["ms"].push_back(ms);
        m["steps"].push_back(static_cast<double>(arm.steps() - steps0) / args.inner);
        m["rhs"].push_back(static_cast<double>(fac->stats().rhs_solved - rhs0) / args.inner);
        rec[arm.name + "_ms"] = ms;
      }
      // Accuracy of the returned blocks, all with the same (Reference) residual.
      const T* y1 = out["p1"].data();
      for (const auto& arm : arms) {
        const T* y = out[arm.name].data();
        std::vector<double> omega(static_cast<std::size_t>(p));
        hs::shifted_residual(a, b, sigma, x.data(), n, y, n, r.data(), n, p,
                             {KernelPath::Reference}, omega.data());
        double rel = 0.0, dev = 0.0;
        std::vector<T> bx(static_cast<std::size_t>(n));
        for (index_t k = 0; k < p; ++k) {
          if (b) {
            hs::spmv(*b, x.data() + k * n, bx.data(), T(1), T(0), {KernelPath::Reference});
          } else {
            std::copy(x.data() + k * n, x.data() + (k + 1) * n, bx.begin());
          }
          rel = std::max(rel,
                         hs::linalg::norm2(n, r.data() + k * n) / hs::linalg::norm2(n, bx.data()));
          double d = 0.0;
          for (index_t i = 0; i < n; ++i) d += hs::abs2(y[i + k * n] - y1[i + k * n]);
          dev = std::max(dev, std::sqrt(d) / hs::linalg::norm2(n, y1 + k * n));
        }
        auto& m = s[p][arm.name];
        m["rel_residual"].push_back(rel);
        m["backward_error"].push_back(*std::max_element(omega.begin(), omega.end()));
        m["dev_vs_p1"].push_back(dev);
      }
      std::printf("%s\n", rec.dump().c_str());
      std::fflush(stdout);
    }
  }
  for (index_t p : args.p) {
    json rec = {{"kind", "median"},
                {"case", name},
                {"p", p},
                {"threads", hs::max_threads()},
                {"precision", args.single ? "single" : "double"}};
    for (const auto& arm : arms) {
      const auto& m = s[p][arm.name];
      json r = {{"ms", median(m.at("ms"))},
                {"ms_per_vec", median(m.at("ms")) / static_cast<double>(p)},
                {"steps", median(m.at("steps"))},
                {"rhs", median(m.at("rhs"))},
                {"rel_residual", median(m.at("rel_residual"))},
                {"backward_error", median(m.at("backward_error"))},
                {"dev_vs_p1", median(m.at("dev_vs_p1"))}};
      rec[arm.name] = r;
    }
    const auto ms = [&](const char* arm) { return median(s[p][arm].at("ms")); };
    rec["kernel_effect"] = ms("p1") / ms("p1_auto");
    rec["active_column_effect"] = ms("p1_auto") / ms("active");
    rec["rounding_stop_effect"] = ms("active") / ms("p4");
    rec["total_vs_p1"] = ms("p1") / ms("p4");
    std::printf("%s\n", rec.dump().c_str());
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto cfg = read_json(args.dir / "config.json");
    const double sigma = cfg.at("target").at("sigma").get<double>();
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
      run<cplx>(args, name, a, b ? &*b : nullptr, sigma);
    } else {
      const auto& a = std::get<CsrMatrix<double>>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<CsrMatrix<double>>(lb->matrix) : nullptr;
      run<double>(args, name, a, b, sigma);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_solvephase_fix_refine: %s\n", e.what());
    return 1;
  }
  return 0;
}
