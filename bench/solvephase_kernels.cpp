// Sparse kernel paths on the case matrices (module ops/sparse, P4 "solvephase"): time of
// spmm (Y = M X, M = B when the case has one, else A) and of the refinement residual
// shifted_residual (R = B X - (A Y - sigma B Y)) for p vectors, on KernelPath Reference (the P1
// loops), Portable and Avx512, repetitions interleaved over paths (order rotated per repetition,
// each sample the mean of --inner calls). Each record also gives the
// largest deviation from the Reference result relative to max |reference| (accuracy check).
// Output: one JSON object per line ("kind": "median"), times in ms per call and per vector.
//
// Usage (via the resource guard; threads = cores given):
//   python3 tools/run/guard.py --skip-busy-check --cores 34,35,36,37 --mem-gb 6 --timeout 1800 --
//     build/release/bench_solvephase_kernels CASE_DIR [--p 1,2,4,8,16] [--reps 9] [--inner 10]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "hsolve/io.hpp"
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
  std::vector<index_t> p = {1, 2, 4, 8, 16};
  int reps = 9;
  int inner = 10;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

Args parse_args(int argc, char** argv) {
  if (argc < 2)
    throw std::runtime_error("usage: bench_solvephase_kernels CASE_DIR [--p LIST] [--reps N]");
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
    const std::string v = argv[++i];
    if (s == "--p")
      a.p = parse_list(v);
    else if (s == "--reps")
      a.reps = std::stoi(v);
    else if (s == "--inner")
      a.inner = std::stoi(v);
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

const char* path_name(KernelPath p) {
  switch (p) {
    case KernelPath::Auto:
      return "auto";
    case KernelPath::Reference:
      return "reference";
    case KernelPath::Portable:
      return "portable";
    case KernelPath::Avx512:
      return "avx512";
  }
  return "?";
}

template <class T>
double max_dev(const std::vector<T>& y, const std::vector<T>& ref) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    num = std::max(num, std::abs(y[i] - ref[i]));
    den = std::max(den, std::abs(ref[i]));
  }
  return den > 0 ? num / den : num;
}

template <class T>
void run(const Args& args, const std::string& name, const CsrMatrix<T>& a, const CsrMatrix<T>* b,
         double sigma) {
  const index_t n = a.nrows;
  const CsrMatrix<T>& m = b ? *b : a;
  std::vector<KernelPath> paths{KernelPath::Reference, KernelPath::Portable};
  if (hs::kernel_path_available(KernelPath::Avx512)) paths.push_back(KernelPath::Avx512);
  const index_t pmax = *std::max_element(args.p.begin(), args.p.end());
  std::vector<T> x(static_cast<std::size_t>(n * pmax)), y(x.size());
  for (index_t j = 0; j < pmax; ++j) {
    hs::linalg::fill_random(x.data() + j * n, n, 7, static_cast<std::uint64_t>(j));
    hs::linalg::fill_random(y.data() + j * n, n, 8, static_cast<std::uint64_t>(j));
  }
  std::vector<double> ws;
  // [op][p][path] -> times (s); reference results for the accuracy check.
  std::map<std::string, std::map<index_t, std::map<KernelPath, std::vector<double>>>> t;
  std::map<std::string, std::map<index_t, std::map<KernelPath, double>>> dev;
  const std::size_t np = paths.size();
  for (int rep = 0; rep < args.reps; ++rep) {
    for (index_t p : args.p) {
      std::vector<T> ref_mx(static_cast<std::size_t>(n * p)), ref_r(ref_mx.size()),
          out(ref_mx.size());
      hs::spmm(m, x.data(), n, ref_mx.data(), n, p, T(1), T(0), {KernelPath::Reference});
      hs::shifted_residual(a, b, sigma, x.data(), n, y.data(), n, ref_r.data(), n, p,
                           {KernelPath::Reference});
      // The path order rotates from one repetition to the next (no path always runs first, on
      // colder caches); each sample is the mean of --inner calls.
      for (std::size_t o = 0; o < np; ++o) {
        const KernelPath path = paths[(o + static_cast<std::size_t>(rep)) % np];
        const hs::KernelOptions opt{path, &ws};
        hs::Timer t1;
        for (int i = 0; i < args.inner; ++i)
          hs::spmm(m, x.data(), n, out.data(), n, p, T(1), T(0), opt);
        t["spmm"][p][path].push_back(t1.seconds() / args.inner);
        dev["spmm"][p][path] = std::max(dev["spmm"][p][path], max_dev(out, ref_mx));
        hs::Timer t2;
        for (int i = 0; i < args.inner; ++i)
          hs::shifted_residual(a, b, sigma, x.data(), n, y.data(), n, out.data(), n, p, opt);
        t["residual"][p][path].push_back(t2.seconds() / args.inner);
        dev["residual"][p][path] = std::max(dev["residual"][p][path], max_dev(out, ref_r));
      }
    }
  }
  for (const auto& [op, per_p] : t)
    for (const auto& [p, per_path] : per_p) {
      const double ref_ms = 1e3 * median(per_path.at(KernelPath::Reference));
      for (const auto& [path, v] : per_path) {
        const double ms = 1e3 * median(v);
        json rec = {{"kind", "median"},
                    {"case", name},
                    {"op", op},
                    {"matrix", op == "spmm" ? (b ? "B" : "A") : (b ? "A,B" : "A")},
                    {"nnz", op == "spmm" ? m.nnz() : a.nnz() + (b ? b->nnz() : 0)},
                    {"complex", hs::is_complex_v<T>},
                    {"threads", hs::max_threads()},
                    {"p", p},
                    {"path", path_name(path)},
                    {"ms", ms},
                    {"ms_per_vec", ms / p},
                    {"speedup_vs_reference", ref_ms / ms},
                    {"max_rel_dev", dev[op][p][path]}};
        std::printf("%s\n", rec.dump().c_str());
      }
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
      auto as_complex = [](hs::LoadedMatrix& mat) {
        return mat.is_complex() ? std::get<CsrMatrix<cplx>>(std::move(mat.matrix))
                                : hs::to_complex(std::get<CsrMatrix<double>>(mat.matrix));
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
    std::fprintf(stderr, "bench_solvephase_kernels: %s\n", e.what());
    return 1;
  }
  return 0;
}
