// GPU offload through the BLAS hook linked into an executable (P10 "gpu"; src/gpu/offload.hpp,
// docs/BLAS.md §10): MUMPS factorizations made by hsolve's backend in this process pass their
// dgemm calls through the hook (CMake hsolve_link_blas_hook), whose dmumps_c wrapper opens the
// offload window. Checked:
//   * the inertia of an indefinite 3-D Laplacian is the exact Sylvester count with the GPU off and
//     on (and the solutions agree): the offload is IEEE double arithmetic like the CPU BLAS;
//   * without a GPU, mode on runs the CPU path and the report says why ("degrade mode");
//   * the hsolve executable with --gpu on finds the same eigenvalues and certificate count as
//     with the GPU off, and its report carries the gpu object.
// GPU use is opt-in, as in test_gpu.cpp: only with HSOLVE_TEST_GPU_DEVICE (e.g. 7 on gpu02),
// otherwise the process sees no GPU (CUDA_VISIBLE_DEVICES is set to the empty list before main).
// HS_GPU_OPTS makes every dgemm outside OpenMP parallel regions eligible (min_speedup=0,
// min_gflop=0): these problems are far too small for the model to offload anything.
#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hsolve/factor.hpp"
#include "hsolve/gpu.hpp"
#include "hsolve/sparse.hpp"

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

const std::string kGpuDevice = [] {
  const char* d = std::getenv("HSOLVE_TEST_GPU_DEVICE");
  const std::string dev = d != nullptr ? d : "";
  setenv("CUDA_VISIBLE_DEVICES", dev.c_str(), 1);
  setenv("HS_GPU_OPTS", "min_speedup=0,min_gflop=0,mem_mb=512,slot_mb=2", 1);
  unsetenv("HS_GPU");
  return dev;
}();

bool gpu_requested() {
  return !kGpuDevice.empty();
}

// Whether the hook in this executable carries the GPU module (CMake HSOLVE_GPU).
#if HSOLVE_WITH_GPU
constexpr bool kGpuModule = true;
#else
constexpr bool kGpuModule = false;
#endif

// 7-point Laplacian on nx x ny x nz with coefficients 1, 1.3, 1.7 (no multiple eigenvalues).
hs::CsrMatrix<double> laplacian_3d(hs::index_t nx, hs::index_t ny, hs::index_t nz) {
  const double cx = 1.0, cy = 1.3, cz = 1.7;
  std::vector<hs::index_t> r, c;
  std::vector<double> v;
  auto id = [&](hs::index_t i, hs::index_t j, hs::index_t k) { return i + nx * (j + ny * k); };
  for (hs::index_t k = 0; k < nz; ++k)
    for (hs::index_t j = 0; j < ny; ++j)
      for (hs::index_t i = 0; i < nx; ++i) {
        const hs::index_t p = id(i, j, k);
        r.push_back(p), c.push_back(p), v.push_back(2 * (cx + cy + cz));
        auto add = [&](hs::index_t q, double w) { r.push_back(p), c.push_back(q), v.push_back(-w); };
        if (i > 0) add(id(i - 1, j, k), cx);
        if (i + 1 < nx) add(id(i + 1, j, k), cx);
        if (j > 0) add(id(i, j - 1, k), cy);
        if (j + 1 < ny) add(id(i, j + 1, k), cy);
        if (k > 0) add(id(i, j, k - 1), cz);
        if (k + 1 < nz) add(id(i, j, k + 1), cz);
      }
  return hs::csr_from_coo<double>(nx * ny * nz, nx * ny * nz, r, c, v);
}

std::vector<double> laplacian_3d_eigenvalues(hs::index_t nx, hs::index_t ny, hs::index_t nz) {
  auto w1 = [](hs::index_t n, double c) {
    std::vector<double> w;
    for (hs::index_t i = 1; i <= n; ++i) w.push_back(c * (2.0 - 2.0 * std::cos(M_PI * i / (n + 1.0))));
    return w;
  };
  std::vector<double> out;
  for (double a : w1(nx, 1.0))
    for (double b : w1(ny, 1.3))
      for (double c : w1(nz, 1.7)) out.push_back(a + b + c);
  std::sort(out.begin(), out.end());
  return out;
}

struct Result {
  hs::Inertia inertia;
  std::vector<double> x;
};

// Factorizes A - sigma I (hsolve's MUMPS backend, the driver's default options) and solves for
// two right-hand sides.
Result factor_and_solve(const hs::CsrMatrix<double>& a, double sigma) {
  hs::ShiftedPencil<double> pencil(a, nullptr);
  auto f = hs::make_mumps_factorization<double>();
  f->analyze(pencil, 1.0, -sigma);
  EXPECT_TRUE(f->factorize(1.0, -sigma));
  Result r;
  r.inertia = f->inertia();
  const hs::index_t n = a.nrows, nrhs = 2;
  r.x.assign(static_cast<std::size_t>(n) * nrhs, 0.0);
  std::mt19937_64 g(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (double& v : r.x) v = u(g);
  f->solve(r.x.data(), nrhs, n);
  return r;
}

// The shift halfway across the widest gap among 64 consecutive eigenvalues a third into the
// spectrum (well inside it, far from both neighbours), and the number of eigenvalues below it.
std::pair<double, hs::index_t> gap_shift(const std::vector<double>& w) {
  std::size_t best = w.size() / 3;
  for (std::size_t i = best; i + 1 < w.size() && i < w.size() / 3 + 64; ++i)
    if (w[i + 1] - w[i] > w[best + 1] - w[best]) best = i;
  return {0.5 * (w[best] + w[best + 1]), static_cast<hs::index_t>(best + 1)};
}

TEST(GpuHook, InertiaIsTheExactCountWithTheGpuOffAndOn) {
  if (!kGpuModule) GTEST_SKIP() << "built without the GPU module";
  const hs::index_t nx = 26, ny = 25, nz = 24;
  const auto a = laplacian_3d(nx, ny, nz);
  const auto [sigma, below] = gap_shift(laplacian_3d_eigenvalues(nx, ny, nz));
  ASSERT_EQ(hs::gpu_offload_set_mode("off"), hs::GpuControl::Ok);
  const Result cpu = factor_and_solve(a, sigma);
  ASSERT_EQ(hs::gpu_offload_set_mode("on"), hs::GpuControl::Ok);
  const Result gpu = factor_and_solve(a, sigma);
  ASSERT_EQ(hs::gpu_offload_set_mode("off"), hs::GpuControl::Ok);
  EXPECT_TRUE(cpu.inertia.valid);
  EXPECT_TRUE(gpu.inertia.valid);
  EXPECT_EQ(cpu.inertia.neg, below);
  EXPECT_EQ(gpu.inertia.neg, below);
  EXPECT_EQ(gpu.inertia.zero, 0);
  double diff = 0.0, nrm = 0.0;
  for (std::size_t i = 0; i < cpu.x.size(); ++i) {
    diff = std::max(diff, std::abs(cpu.x[i] - gpu.x[i]));
    nrm = std::max(nrm, std::abs(cpu.x[i]));
  }
  EXPECT_LE(diff, 1e-9 * nrm);
  const json r = json::parse(hs::gpu_offload_report_json());
  if (gpu_requested()) {
    EXPECT_EQ(r.value("available", false), true) << r.dump();
    EXPECT_GE(r.value("factorizations_offloaded", 0), 1) << r.dump();
    EXPECT_GT(r.value("calls", 0), 0) << r.dump();
    EXPECT_GT(r.value("gpu_gflop", 0.0), 0.0) << r.dump();
    EXPECT_EQ(r.value("errors", -1), 0) << r.dump();
  } else {
    EXPECT_EQ(r.value("available", true), false) << r.dump();
    EXPECT_FALSE(r.value("reason", std::string()).empty()) << r.dump();
    EXPECT_EQ(r.value("factorizations_offloaded", -1), 0) << r.dump();
  }
}

#ifdef HSOLVE_HOOKED_CLI
// Runs the hsolve executable; returns its report (null when it failed).
json run_cli(const fs::path& dir, const std::string& tag, const std::string& extra) {
  const fs::path out = dir / tag;
  const std::string cmd = std::string("'") + HSOLVE_HOOKED_CLI + "' --A '" + (dir / "A.mtx").string() +
                          "' --sigma 3.3 --k 8 --out '" + out.string() + "' --threads 2 " + extra +
                          " > '" + (dir / (tag + ".log")).string() + "' 2>&1";
  const int rc = std::system(cmd.c_str());
  std::ifstream log(dir / (tag + ".log"));
  const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(WIFEXITED(rc) && WEXITSTATUS(rc) == 0) << tag << " rc " << rc << ": " << text;
  std::ifstream rep(out / "report.json");
  if (!rep) return json();
  return json::parse(rep);
}
#endif

TEST(GpuHook, HsolveWithGpuOnFindsTheSameEigenpairsAndCount) {
#ifndef HSOLVE_HOOKED_CLI
  GTEST_SKIP() << "needs the hooked hsolve executable";
#else
  if (!kGpuModule) GTEST_SKIP() << "built without the GPU module";
  const fs::path dir = fs::temp_directory_path() / ("hsolve_gpu_hook_" + std::to_string(static_cast<long>(getpid())));
  fs::create_directories(dir);
  const auto a = laplacian_3d(22, 21, 20);
  {
    std::ofstream out(dir / "A.mtx");
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    hs::index_t lower = 0;
    for (hs::index_t i = 0; i < a.nrows; ++i)
      for (hs::index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p) lower += a.colind[p] <= i;
    out << a.nrows << " " << a.ncols << " " << lower << "\n";
    out.precision(17);
    for (hs::index_t i = 0; i < a.nrows; ++i)
      for (hs::index_t p = a.rowptr[i]; p < a.rowptr[i + 1]; ++p)
        if (a.colind[p] <= i) out << i + 1 << " " << a.colind[p] + 1 << " " << a.values[p] << "\n";
  }
  const json cpu = run_cli(dir, "cpu", "");
  const json gpu = run_cli(dir, "gpu", "--gpu on");
  ASSERT_FALSE(cpu.is_null());
  ASSERT_FALSE(gpu.is_null());
  EXPECT_FALSE(cpu.contains("gpu"));  // default off: the report is today's
  ASSERT_TRUE(gpu.contains("gpu")) << gpu.dump();
  const json& g = gpu["gpu"];
  EXPECT_EQ(g.value("mode", ""), "on");
  EXPECT_TRUE(cpu.value("success", false));
  EXPECT_TRUE(gpu.value("success", false));
  ASSERT_EQ(cpu["pairs"].size(), gpu["pairs"].size());
  for (std::size_t i = 0; i < cpu["pairs"].size(); ++i) {
    const double l1 = cpu["pairs"][i]["lambda_re"], l2 = gpu["pairs"][i]["lambda_re"];
    EXPECT_NEAR(l1, l2, 1e-10 * std::abs(l1));
  }
  EXPECT_EQ(cpu["certificate"]["count_inertia"], gpu["certificate"]["count_inertia"]);
  EXPECT_EQ(cpu["certificate"]["passed"], gpu["certificate"]["passed"]);
  if (gpu_requested()) {
    EXPECT_EQ(g.value("available", false), true) << g.dump();
    EXPECT_GE(g.value("factorizations_offloaded", 0), 1) << g.dump();
    EXPECT_GT(g.value("gpu_gflop", 0.0), 0.0) << g.dump();
  } else {
    EXPECT_EQ(g.value("available", true), false) << g.dump();
  }
  // auto: far below the threshold (3e12 estimated flops) - no CUDA library is even looked for.
  const json automode = run_cli(dir, "auto", "--gpu auto");
  ASSERT_FALSE(automode.is_null());
  ASSERT_TRUE(automode.contains("gpu")) << automode.dump();
  EXPECT_EQ(automode["gpu"].value("mode", ""), "auto");
  EXPECT_TRUE(automode["gpu"]["available"].is_null()) << automode["gpu"].dump();
  EXPECT_EQ(automode["gpu"].value("factorizations_offloaded", -1), 0);
  EXPECT_EQ(cpu["certificate"]["count_inertia"], automode["certificate"]["count_inertia"]);
  fs::remove_all(dir);
#endif
}

}  // namespace
