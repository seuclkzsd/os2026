// Time split of one shift-invert application Y = (A - sigma B)^{-1} B X (module ops, P4
// "solvephase"), the inner loop of every Krylov method, for p = 1, 2, 4, ... right-hand sides,
// before (P1 kernels, KernelPath::Reference) and after (KernelPath::Auto) the P4 kernels, in the
// same process with repetitions interleaved over p and paths.
//
// For one data case: load A (and B) with hs::load_matrix, factorize K = A - sigma B at the case's
// sigma (config.json target) with the driver's factor options (MUMPS LDL^T, L0-OpenMP unless
// --det 1, the ordering given by --ordering: the driver picks SCOTCH on 3-D graphs, AMD on 2-D
// ones and on the complex embedding; --precision single for a smumps factor), then for every p
// time (per call; the medians divide by p):
//   bx_{ref,new}    Y = B X (spmm; a copy when B = I) - what ShiftInvertOperator does first
//   solve           SparseFactorization::solve(Y, p): one MUMPS JOB=3 call, incl. the wrapper's
//                   pack/unpack of the real embedding (complex T) / float rounding (single)
//   pack_{old,new}  the wrapper's pack + unpack alone, replicated here (complex T only): old =
//                   zero-initialised buffer (base a8ed8f4), new = default-initialised (P4)
//   resid_{ref,new} r = B X - (A Y - sigma B Y): three spmm passes (ref) / one fused sweep (new)
//   op_{ref,new}    ShiftInvertOperator::apply without refinement (the driver's double factor)
//   opr_{ref,new}   ShiftInvertOperator::apply with refinement (max_steps 10, rel_tol --ref-tol,
//                   1e-12 = the driver's tol * inner_tol_factor * 0.1 for Single / BLR factors);
//                   ref_steps = corrections per apply, ref_rel_residual = residual returned;
//                   ref = P1 refinement (no rounding-level stop), new = P4 (stop_at_rounding_level)
//   omega_solve     componentwise backward error of a plain solve (max over the p columns);
//   omega_refined_* that of the refined block (ShiftInvertOperator::last_backward_error)
// A STREAM triad on the same threads and the factor size give the solve's effective bandwidth
// (two sweeps over the factor entries; dmumps entries, of the embedding for complex T).
// Output: one JSON object per line on stdout ("kind": "setup" | "rep" | "median").
//
// Usage (via the resource guard; the thread count is the number of cores given):
//   python3 tools/run/guard.py --skip-busy-check --cores 34,35,36,37 --mem-gb 6 --timeout 1800 --
//     build/release/bench_solvephase_profile CASE_DIR [--ordering amd|scotch|metis|auto]
//     [--det 0|1] [--precision double|single] [--p 1,2,4,8,16] [--reps 7] [--inner 3]
//     [--ref-tol 1e-12]
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
  bool deterministic = false;
  bool single = false;
  std::vector<index_t> p = {1, 2, 4, 8, 16};
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
  if (argc < 2) throw std::runtime_error("usage: bench_solvephase_profile CASE_DIR [options]");
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--ordering")
      a.ordering = value();
    else if (s == "--det")
      a.deterministic = std::stoi(value()) != 0;
    else if (s == "--precision")
      a.single = value() == "single";
    else if (s == "--p")
      a.p = parse_list(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--inner")
      a.inner = std::stoi(value());
    else if (s == "--ref-tol")
      a.ref_tol = std::stod(value());
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

/// Best-of-5 STREAM triad a = b + s c on 3 x 64 MB arrays (GB/s, 24 bytes per element).
double stream_triad_gbs() {
  const index_t n = index_t(1) << 23;
  std::vector<double> a(n), b(n), c(n);
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < n; ++i) {
    a[i] = 0.0;
    b[i] = 1.0;
    c[i] = 2.0;
  }
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    hs::Timer t;
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) a[i] = b[i] + 0.5 * c[i];
    best = std::min(best, t.seconds());
  }
  if (a[n / 2] != 2.0) std::fprintf(stderr, "triad check failed\n");
  return 24.0 * static_cast<double>(n) / best / 1e9;
}

/// The wrapper's pack + unpack for complex T (mumps_factorization.cpp solve()): [Re; Im] per
/// column into a buffer allocated per call, zero-initialised (zeroed, base a8ed8f4) or not (P4).
template <class T>
void replicate_pack(T* b, index_t n, index_t nrhs, index_t ldb, bool zeroed) {
  if constexpr (hs::is_complex_v<T>) {
    const index_t m = 2 * n;
    const std::size_t len = static_cast<std::size_t>(m * nrhs);
    std::vector<double> zero_buf;
    std::unique_ptr<double[]> raw_buf;
    double* packed = nullptr;
    if (zeroed) {
      zero_buf.resize(len);
      packed = zero_buf.data();
    } else {
      raw_buf.reset(new double[len]);
      packed = raw_buf.get();
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (index_t k = 0; k < nrhs; ++k)
      for (index_t i = 0; i < n; ++i) {
        const T v = b[i + k * ldb];
        packed[i + k * m] = v.real();
        packed[n + i + k * m] = v.imag();
      }
#pragma omp parallel for collapse(2) schedule(static)
    for (index_t k = 0; k < nrhs; ++k)
      for (index_t i = 0; i < n; ++i) b[i + k * ldb] = T(packed[i + k * m], packed[n + i + k * m]);
  }
}

template <class F>
double time_mean(int inner, F&& f) {
  hs::Timer t;
  for (int i = 0; i < inner; ++i) f();
  return t.seconds() / inner;
}

template <class T>
void run(const Args& args, const std::string& name, const std::string& problem,
         const CsrMatrix<T>& a, const CsrMatrix<T>* b, double sigma0) {
  const index_t n = a.nrows;
  hs::ShiftedPencil<T> pencil(a, b);
  hs::FactorOptions fo;
  fo.ordering = ordering_code(args.ordering);
  fo.deterministic = args.deterministic;
  fo.threads = hs::max_threads();
  if (args.single) fo.precision = hs::FactorPrecision::Single;
  auto fac = hs::make_mumps_factorization<T>(fo);
  hs::Timer ta;
  fac->analyze(pencil, 1.0, -sigma0);
  const double analyze_s = ta.seconds();
  double sigma = sigma0;
  hs::Timer tf;
  for (int r = 0; !fac->factorize(1.0, -sigma); ++r) {
    if (r == 3) throw std::runtime_error("no regular shift near sigma");
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
  }
  const double factorize_s = tf.seconds();
  const auto& st = fac->stats();
  const double factor_bytes = (args.single ? 4.0 : 8.0) * static_cast<double>(st.factor_entries);
  const double triad = stream_triad_gbs();
  json setup = {
      {"kind", "setup"},
      {"case", name},
      {"problem", problem},
      {"n", n},
      {"complex", hs::is_complex_v<T>},
      {"nnz_a", a.nnz()},
      {"nnz_b", b ? b->nnz() : 0},
      {"threads", hs::max_threads()},
      {"ordering", args.ordering},
      {"deterministic", args.deterministic},
      {"precision", args.single ? "single" : "double"},
      {"kernel_path",
       hs::resolve_kernel_path(KernelPath::Auto) == KernelPath::Avx512 ? "avx512" : "portable"},
      {"sigma", sigma},
      {"analyze_s", analyze_s},
      {"factorize_s", factorize_s},
      {"factor_entries", st.factor_entries},
      {"factor_mb_used", st.factor_mb_used},
      {"flops", st.estimate.flops},
      {"triad_gbs", triad}};
  std::printf("%s\n", setup.dump().c_str());
  std::fflush(stdout);

  const index_t pmax = *std::max_element(args.p.begin(), args.p.end());
  std::vector<T> x(static_cast<std::size_t>(n * pmax)), y(x.size()), r(x.size());
  for (index_t j = 0; j < pmax; ++j)
    hs::linalg::fill_random(x.data() + j * n, n, 2026, static_cast<std::uint64_t>(j));
  hs::SolveRefinement refine;
  refine.max_steps = 10;
  refine.rel_tol = args.ref_tol;
  const KernelPath kRef = KernelPath::Reference, kNew = KernelPath::Auto;
  hs::ShiftInvertOperator<T> op_ref(*fac, pencil, sigma, {}, kRef),
      op_new(*fac, pencil, sigma, {}, kNew);
  hs::SolveRefinement refine_p1 = refine;
  refine_p1.stop_at_rounding_level =
      false;  // P1 refinement: a correction even at the rounding level
  hs::ShiftInvertOperator<T> opr_ref(*fac, pencil, sigma, refine_p1, kRef),
      opr_new(*fac, pencil, sigma, refine, kNew);
  std::vector<double> ws;
  const hs::KernelOptions o_ref{kRef, nullptr}, o_new{kNew, &ws};

  std::map<index_t, std::map<std::string, std::vector<double>>> samples;
  for (int rep = 0; rep < args.reps; ++rep) {
    for (index_t p : args.p) {
      auto& s = samples[p];
      auto bx = [&](const hs::KernelOptions& o) {
        if (b) {
          hs::spmm(*b, x.data(), n, y.data(), n, p, T(1), T(0), o);
        } else {
          std::copy(x.data(), x.data() + n * p, y.data());
        }
      };
      json rec = {{"kind", "rep"}, {"case", name}, {"rep", rep}, {"p", p}};
      auto record = [&](const std::string& key, double seconds) {
        rec[key + "_ms"] = 1e3 * seconds;
        s[key].push_back(1e3 * seconds / static_cast<double>(p));
      };
      record("bx_ref", time_mean(args.inner, [&] { bx(o_ref); }));
      record("bx_new", time_mean(args.inner, [&] { bx(o_new); }));
      record("solve", time_mean(args.inner, [&] { fac->solve(y.data(), p, n); }));
      if (hs::is_complex_v<T>) {
        record("pack_old", time_mean(args.inner, [&] { replicate_pack(y.data(), n, p, n, true); }));
        record("pack_new",
               time_mean(args.inner, [&] { replicate_pack(y.data(), n, p, n, false); }));
      }
      // y now holds (a power of) K^{-1} B x: a valid iterate for timing the residual.
      record("resid_ref", time_mean(args.inner, [&] {
               hs::shifted_residual(a, b, sigma, x.data(), n, y.data(), n, r.data(), n, p, o_ref);
             }));
      record("resid_new", time_mean(args.inner, [&] {
               hs::shifted_residual(a, b, sigma, x.data(), n, y.data(), n, r.data(), n, p, o_new);
             }));
      record("op_ref", time_mean(args.inner, [&] { op_ref.apply(x.data(), n, y.data(), n, p); }));
      record("op_new", time_mean(args.inner, [&] { op_new.apply(x.data(), n, y.data(), n, p); }));
      {
        // Componentwise backward error of a plain solve (no refinement).
        std::vector<double> omega(static_cast<std::size_t>(p));
        hs::shifted_residual(a, b, sigma, x.data(), n, y.data(), n, r.data(), n, p, o_new,
                             omega.data());
        rec["omega_solve"] = *std::max_element(omega.begin(), omega.end());
      }
      for (auto* o : {&opr_ref, &opr_new}) {
        const bool is_ref = (o == &opr_ref);
        const index_t steps0 = o->refinement_steps();
        record(is_ref ? "opr_ref" : "opr_new",
               time_mean(args.inner, [&] { o->apply(x.data(), n, y.data(), n, p); }));
        const double steps = static_cast<double>(o->refinement_steps() - steps0) / args.inner;
        rec[is_ref ? "ref_steps_ref" : "ref_steps_new"] = steps;
        rec[is_ref ? "ref_rel_residual_ref" : "ref_rel_residual_new"] = o->last_relative_residual();
        rec[is_ref ? "omega_refined_ref" : "omega_refined_new"] = o->last_backward_error();
        s[is_ref ? "steps_ref" : "steps_new"].push_back(
            steps);  // corrections per apply (not per vector)
      }
      std::printf("%s\n", rec.dump().c_str());
      std::fflush(stdout);
      s["solve_call"].push_back(rec["solve_ms"].get<double>() * static_cast<double>(p));
    }
  }
  for (index_t p : args.p) {
    auto& s = samples[p];
    const double solve_ms = median(s["solve_call"]) / static_cast<double>(p);
    json rec = {{"kind", "median"},
                {"case", name},
                {"p", p},
                {"threads", hs::max_threads()},
                {"precision", args.single ? "single" : "double"}};
    for (const auto& [key, v] : s) {
      if (key == "solve_call") continue;
      if (key.rfind("steps_", 0) == 0) {
        rec[key] = median(v);
        continue;
      }
      rec[key + "_ms_per_vec"] = median(v);
    }
    rec["solve_call_ms"] = solve_ms;
    // Effective bandwidth of one solve call: two sweeps (forward, backward) over the factors.
    rec["solve_gbs"] = 2.0 * factor_bytes / (solve_ms * 1e-3) / 1e9;
    std::printf("%s\n", rec.dump().c_str());
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto meta = read_json(args.dir / "meta.json");
    const auto cfg = read_json(args.dir / "config.json");
    const std::string problem = meta.at("problem").get<std::string>();
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
      run<cplx>(args, name, problem, a, b ? &*b : nullptr, sigma);
    } else {
      const auto& a = std::get<CsrMatrix<double>>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<CsrMatrix<double>>(lb->matrix) : nullptr;
      run<double>(args, name, problem, a, b, sigma);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_solvephase_profile: %s\n", e.what());
    return 1;
  }
  return 0;
}
