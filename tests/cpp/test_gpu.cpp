// GPU module (src/gpu, P10 "gpu"; docs/BLAS.md §10): options, run-time loading, the dgemm engine
// and the process-level offload controller.
//
// GPU use is opt-in for tests: the GPU tests run only when HSOLVE_TEST_GPU_DEVICE names a device
// (e.g. HSOLVE_TEST_GPU_DEVICE=7 on gpu02 - the one GPU this project may use; run them through
// tools/bench/gpu_slot.sh). Otherwise this process sees no GPU at all (CUDA_VISIBLE_DEVICES is set
// to the empty list before main), so a plain ctest run on a shared GPU node never touches a GPU,
// and the fallback tests below check the CPU-only behaviour ("degrade mode").
#include <gtest/gtest.h>
#include <omp.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gpu/cuda_lib.hpp"
#include "gpu/gemm_engine.hpp"
#include "gpu/offload.hpp"

extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
                       const double*, const int*, const double*, const int*, const double*, double*,
                       const int*, std::size_t, std::size_t);

namespace {

// Before main() and before anything could initialize CUDA: only the named device, or none.
const std::string kGpuDevice = [] {
  const char* d = std::getenv("HSOLVE_TEST_GPU_DEVICE");
  const std::string dev = d != nullptr ? d : "";
  setenv("CUDA_VISIBLE_DEVICES", dev.c_str(), 1);
  return dev;
}();

bool gpu_requested() {
  return !kGpuDevice.empty();
}

#define REQUIRE_GPU()                                                             \
  do {                                                                            \
    if (!gpu_requested()) GTEST_SKIP() << "set HSOLVE_TEST_GPU_DEVICE to run GPU tests"; \
  } while (0)

// Options that make the engine take every shape and use small slots (many tiles per call).
hs::gpu::EngineOptions small_engine() {
  hs::gpu::EngineOptions o;
  o.mem_bytes = std::size_t{256} << 20;
  o.slot_bytes = std::size_t{1} << 20;
  o.min_flops = 0.0;
  o.min_speedup = 0.0;
  return o;
}

struct Problem {
  char ta, tb;
  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
  std::vector<double> a, b, c;
};

Problem make_problem(char ta, char tb, int m, int n, int k, int pad, double alpha, double beta,
                     std::uint64_t seed) {
  Problem p{ta, tb, m, n, k, 0, 0, 0, alpha, beta, {}, {}, {}};
  const bool at = ta == 'T', bt = tb == 'T';
  p.lda = (at ? k : m) + pad;
  p.ldb = (bt ? n : k) + pad;
  p.ldc = m + pad;
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  p.a.resize(static_cast<std::size_t>(p.lda) * (at ? m : k));
  p.b.resize(static_cast<std::size_t>(p.ldb) * (bt ? k : n));
  p.c.resize(static_cast<std::size_t>(p.ldc) * n);
  for (double& v : p.a) v = u(g);
  for (double& v : p.b) v = u(g);
  for (double& v : p.c) v = u(g);
  return p;
}

// max |C - Cref| / (|alpha| k max|A| max|B| + |beta| max|C0|) over the m x n block; the padding
// rows of C must be unchanged (exactly).
double rel_error(const Problem& p, const std::vector<double>& c, const std::vector<double>& cref,
                 bool* padding_ok) {
  double err = 0.0;
  *padding_ok = true;
  for (int j = 0; j < p.n; ++j)
    for (int i = 0; i < p.ldc; ++i) {
      const std::size_t q = i + static_cast<std::size_t>(j) * p.ldc;
      if (i < p.m)
        err = std::max(err, std::abs(c[q] - cref[q]));
      else if (std::memcmp(&c[q], &p.c[q], sizeof(double)) != 0)
        *padding_ok = false;
    }
  const double scale = std::abs(p.alpha) * p.k + std::abs(p.beta) + 1e-300;
  return err / scale;
}

std::vector<double> reference(const Problem& p) {
  std::vector<double> c = p.c;
  dgemm_(&p.ta, &p.tb, &p.m, &p.n, &p.k, &p.alpha, p.a.data(), &p.lda, p.b.data(), &p.ldb, &p.beta,
         c.data(), &p.ldc, 1, 1);
  return c;
}

// ---- without a GPU (always) ----

TEST(GpuOptions, ModesParse) {
  hs::gpu::Mode m = hs::gpu::Mode::On;
  EXPECT_TRUE(hs::gpu::parse_mode("off", &m));
  EXPECT_EQ(m, hs::gpu::Mode::Off);
  EXPECT_TRUE(hs::gpu::parse_mode("auto", &m));
  EXPECT_EQ(m, hs::gpu::Mode::Auto);
  EXPECT_TRUE(hs::gpu::parse_mode("on", &m));
  EXPECT_EQ(m, hs::gpu::Mode::On);
  EXPECT_FALSE(hs::gpu::parse_mode("ON", &m));
  EXPECT_FALSE(hs::gpu::parse_mode("", &m));
  EXPECT_STREQ(hs::gpu::mode_name(hs::gpu::Mode::Auto), "auto");
}

TEST(GpuOptions, OffloadOptionsParse) {
  std::string unknown;
  const hs::gpu::OffloadOptions o = hs::gpu::parse_offload_options(
      "device=1, mem_mb=512,slot_mb=8,pack=3,slots=5,hybrid=1,cpu_threads=6,min_gflop=2,"
      "cpu_gflops=500,host_gbs=20,min_speedup=1.5,auto_min_tflop=0.5,fault_after=4",
      &unknown);
  EXPECT_EQ(unknown, "");
  EXPECT_EQ(o.engine.device, 1);
  EXPECT_EQ(o.engine.mem_bytes, std::size_t{512} << 20);
  EXPECT_EQ(o.engine.slot_bytes, std::size_t{8} << 20);
  EXPECT_EQ(o.engine.pack_threads, 3);
  EXPECT_EQ(o.engine.slots, 5);
  EXPECT_TRUE(o.engine.hybrid);
  EXPECT_EQ(o.engine.cpu_threads, 6);
  EXPECT_DOUBLE_EQ(o.engine.min_flops, 2e9);
  EXPECT_DOUBLE_EQ(o.engine.cpu_gflops, 500.0);
  EXPECT_DOUBLE_EQ(o.engine.host_gbs, 20.0);
  EXPECT_DOUBLE_EQ(o.engine.min_speedup, 1.5);
  EXPECT_DOUBLE_EQ(o.auto_min_flops, 0.5e12);
  EXPECT_EQ(o.engine.inject_fault_after_tiles, 4);
  // Defaults, and what is rejected.
  const hs::gpu::OffloadOptions d = hs::gpu::parse_offload_options("", &unknown);
  EXPECT_FALSE(d.engine.hybrid);
  EXPECT_EQ(d.engine.pack_threads, 2);
  EXPECT_DOUBLE_EQ(d.auto_min_flops, 3e12);
  hs::gpu::parse_offload_options("pack=9,slots=1,mem_mb=x,bogus=1,hybrid=2,device", &unknown);
  EXPECT_EQ(unknown, "pack=9,slots=1,mem_mb=x,bogus=1,hybrid=2,device");
}

TEST(GpuLoader, LoadsOrSaysWhy) {
  std::string why;
  const hs::gpu::CudaLib* lib = hs::gpu::load(&why);
  if (lib == nullptr) {
    EXPECT_FALSE(why.empty());
  } else {
    EXPECT_NE(lib->Dgemm, nullptr);
    EXPECT_NE(lib->MemcpyAsync, nullptr);
    EXPECT_FALSE(lib->cudart_path.empty());
    EXPECT_EQ(hs::gpu::load(), lib);  // loaded once
  }
}

TEST(GpuFallback, NoVisibleDeviceMeansNoEngine) {
  if (gpu_requested()) GTEST_SKIP() << "a GPU is visible in this run";
  std::string why;
  auto e = hs::gpu::GemmEngine::create(small_engine(), &why);
  EXPECT_EQ(e, nullptr);
  EXPECT_FALSE(why.empty());
}

// The process-level controller without a GPU: mode on, a factorization asks for the GPU, the
// engine cannot be created - the window stays closed (every call stays on the CPU), the report
// says why. (The controller is process-wide: this is the only test of this executable using it.)
TEST(GpuOffload, DegradesToTheCpuWithoutAGpu) {
  if (gpu_requested()) GTEST_SKIP() << "a GPU is visible in this run";
  // auto below the threshold: CUDA is not even looked for.
  ASSERT_TRUE(hs::gpu::set_mode("auto"));
  hs::gpu::factorization_begin(1e9);
  EXPECT_FALSE(hs::gpu::window_open());
  hs::gpu::factorization_end();
  EXPECT_NE(hs::gpu::report_json().find("\"available\": null"), std::string::npos) << hs::gpu::report_json();
  ASSERT_TRUE(hs::gpu::set_mode("on"));
  EXPECT_EQ(hs::gpu::mode(), hs::gpu::Mode::On);
  EXPECT_FALSE(hs::gpu::set_mode("sometimes"));
  hs::gpu::factorization_begin(1e13);
  EXPECT_FALSE(hs::gpu::window_open());
  Problem p = make_problem('N', 'N', 64, 64, 64, 0, 1.0, 1.0, 5);
  EXPECT_FALSE(hs::gpu::offload_dgemm('N', 'N', 64, 64, 64, 1.0, p.a.data(), 64, p.b.data(), 64, 1.0,
                                      p.c.data(), 64, dgemm_));
  hs::gpu::factorization_end();
  const std::string r = hs::gpu::report_json();
  EXPECT_NE(r.find("\"mode\": \"on\""), std::string::npos) << r;
  EXPECT_NE(r.find("\"available\": false"), std::string::npos) << r;
  EXPECT_NE(r.find("\"reason\": "), std::string::npos) << r;
  EXPECT_NE(r.find("\"factorizations\": 2"), std::string::npos) << r;
  EXPECT_NE(r.find("\"factorizations_offloaded\": 0"), std::string::npos) << r;
  ASSERT_TRUE(hs::gpu::set_mode("off"));
  hs::gpu::factorization_begin(1e13);  // off: not even counted
  EXPECT_FALSE(hs::gpu::window_open());
  hs::gpu::factorization_end();
}

// ---- with a GPU (HSOLVE_TEST_GPU_DEVICE) ----

TEST(GpuEngine, DescribesTheDevice) {
  REQUIRE_GPU();
  std::string why;
  auto e = hs::gpu::GemmEngine::create(small_engine(), &why);
  ASSERT_NE(e, nullptr) << why;
  const hs::gpu::EngineInfo& i = e->info();
  EXPECT_GT(i.multiprocessors, 0);
  EXPECT_GT(i.device_total_bytes, 0u);
  EXPECT_GT(i.runtime_version, 0);
  EXPECT_GT(i.cublas_version, 0);
  EXPECT_GT(i.init_seconds, 0.0);
  EXPECT_EQ(i.options.slots, 6);
  EXPECT_FALSE(e->broken());
}

// Every transpose combination, both split directions (m <= n: column tiles; m > n: row tiles),
// alpha / beta special values, padded leading dimensions, 1 MiB slots (many tiles): equal to the
// CPU BLAS up to rounding; the padding rows of C untouched; GPU-only and hybrid.
TEST(GpuEngine, MatchesTheCpuBlas) {
  REQUIRE_GPU();
  for (const bool hybrid : {false, true}) {
    hs::gpu::EngineOptions o = small_engine();
    o.hybrid = hybrid;
    o.cpu_threads = 2;
    std::string why;
    auto e = hs::gpu::GemmEngine::create(o, &why);
    ASSERT_NE(e, nullptr) << why;
    std::uint64_t seed = 1;
    for (const char ta : {'N', 'T'})
      for (const char tb : {'N', 'T'})
        for (const auto& [m, n, k] : {std::tuple{300, 700, 260}, std::tuple{700, 300, 260},
                                      std::tuple{33, 1000, 77}, std::tuple{1000, 40, 513}})
          for (const auto& [alpha, beta] : {std::pair{1.0, 1.0}, std::pair{-1.0, 0.0},
                                            std::pair{2.5, -0.5}})
            for (const int pad : {0, 3}) {
              Problem p = make_problem(ta, tb, m, n, k, pad, alpha, beta, ++seed);
              const std::vector<double> cref = reference(p);
              std::vector<double> c = p.c;
              ASSERT_TRUE(e->dgemm(ta, tb, m, n, k, alpha, p.a.data(), p.lda, p.b.data(), p.ldb, beta,
                                   c.data(), p.ldc, dgemm_))
                  << ta << tb << " " << m << "x" << n << "x" << k;
              bool padding_ok = false;
              const double err = rel_error(p, c, cref, &padding_ok);
              EXPECT_LE(err, 1e-15 * 8) << "hybrid " << hybrid << " " << ta << tb << " " << m << "x" << n
                                        << "x" << k << " alpha " << alpha << " beta " << beta;
              EXPECT_TRUE(padding_ok);
            }
    const hs::gpu::EngineStats s = e->stats();
    EXPECT_GT(s.tiles, s.calls);  // several tiles per call
    EXPECT_GT(s.gpu_flops, 0.0);
    EXPECT_EQ(s.errors, 0);
  }
}

TEST(GpuEngine, BetaZeroNeverReadsC) {
  REQUIRE_GPU();
  std::string why;
  auto e = hs::gpu::GemmEngine::create(small_engine(), &why);
  ASSERT_NE(e, nullptr) << why;
  Problem p = make_problem('N', 'T', 200, 500, 300, 0, -1.0, 0.0, 77);
  std::fill(p.c.begin(), p.c.end(), std::numeric_limits<double>::quiet_NaN());
  const std::vector<double> cref = reference(p);
  std::vector<double> c = p.c;
  ASSERT_TRUE(e->dgemm('N', 'T', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, 0.0,
                       c.data(), p.ldc, dgemm_));
  for (std::size_t i = 0; i < c.size(); ++i) {
    ASSERT_TRUE(std::isfinite(c[i])) << i;
    EXPECT_NEAR(c[i], cref[i], 1e-12 * p.k);
  }
}

// GPU-only calls are reproducible: the tiles depend on the shape only.
TEST(GpuEngine, GpuOnlyCallsAreBitwiseReproducible) {
  REQUIRE_GPU();
  std::string why;
  auto e = hs::gpu::GemmEngine::create(small_engine(), &why);
  ASSERT_NE(e, nullptr) << why;
  Problem p = make_problem('N', 'N', 400, 900, 333, 5, -1.0, 1.0, 9);
  std::vector<double> c1 = p.c, c2 = p.c;
  ASSERT_TRUE(e->dgemm('N', 'N', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, p.beta,
                       c1.data(), p.ldc, dgemm_));
  ASSERT_TRUE(e->dgemm('N', 'N', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, p.beta,
                       c2.data(), p.ldc, dgemm_));
  EXPECT_EQ(std::memcmp(c1.data(), c2.data(), c1.size() * sizeof(double)), 0);
}

// The model: small or thin calls stay on the CPU, large ones are taken; min_flops is a floor.
TEST(GpuEngine, ModelDecidesWhatIsOffloaded) {
  REQUIRE_GPU();
  hs::gpu::EngineOptions o;  // the defaults (measured on gpu02)
  o.mem_bytes = std::size_t{512} << 20;
  o.cpu_gflops = 640.0;
  std::string why;
  auto e = hs::gpu::GemmEngine::create(o, &why);
  ASSERT_NE(e, nullptr) << why;
  EXPECT_TRUE(e->eligible('N', 'N', 1000, 9000, 4000));   // a front's contribution-block update
  EXPECT_TRUE(e->eligible('N', 'N', 2000, 8000, 4000));
  EXPECT_FALSE(e->eligible('N', 'N', 1000, 9000, 512));   // too little work per byte moved
  EXPECT_FALSE(e->eligible('N', 'N', 4000, 4000, 128));   // a rank-128 panel update
  EXPECT_FALSE(e->eligible('N', 'N', 250, 1000, 4000));   // a triangle stripe
  EXPECT_FALSE(e->eligible('N', 'N', 64, 64, 64));
  EXPECT_GT(e->predicted_speedup(2000, 8000, 4000), e->predicted_speedup(1000, 9000, 4000));
  Problem p = make_problem('N', 'N', 64, 64, 64, 0, 1.0, 1.0, 3);
  EXPECT_FALSE(e->dgemm('N', 'N', 64, 64, 64, 1.0, p.a.data(), 64, p.b.data(), 64, 1.0, p.c.data(), 64,
                        dgemm_));
  EXPECT_EQ(e->stats().calls, 0);
}

// Two threads call at once: one call may be declined (busy), the caller then computes it itself;
// both results are right.
TEST(GpuEngine, ConcurrentCallersAreServedOrDeclined) {
  REQUIRE_GPU();
  std::string why;
  auto e = hs::gpu::GemmEngine::create(small_engine(), &why);
  ASSERT_NE(e, nullptr) << why;
  std::vector<Problem> ps;
  for (int t = 0; t < 4; ++t) ps.push_back(make_problem('N', 'N', 300, 1200, 400, 0, -1.0, 1.0, 100 + t));
  std::vector<std::vector<double>> out(ps.size());
  std::atomic<int> declined{0};
  std::vector<std::thread> th;
  for (std::size_t t = 0; t < ps.size(); ++t)
    th.emplace_back([&, t] {
      const Problem& p = ps[t];
      out[t] = p.c;
      for (int r = 0; r < 3; ++r) {
        std::vector<double> c = p.c;
        if (!e->dgemm('N', 'N', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, p.beta,
                      c.data(), p.ldc, dgemm_)) {
          ++declined;
          dgemm_("N", "N", &p.m, &p.n, &p.k, &p.alpha, p.a.data(), &p.lda, p.b.data(), &p.ldb, &p.beta,
                 c.data(), &p.ldc, 1, 1);
        }
        out[t] = c;
      }
    });
  for (auto& t : th) t.join();
  for (std::size_t t = 0; t < ps.size(); ++t) {
    bool padding_ok = false;
    EXPECT_LE(rel_error(ps[t], out[t], reference(ps[t]), &padding_ok), 1e-14);
  }
  EXPECT_EQ(e->stats().declined_busy, declined.load());
  EXPECT_GT(e->stats().calls, 0);
}

// A CUDA failure in the middle of a call (injected after 3 tiles): the call still completes
// correctly on the CPU, the engine is broken from then on and declines further calls.
TEST(GpuEngine, AFailedTileIsRecomputedOnTheCpu) {
  REQUIRE_GPU();
  hs::gpu::EngineOptions o = small_engine();
  o.inject_fault_after_tiles = 3;
  std::string why;
  auto e = hs::gpu::GemmEngine::create(o, &why);
  ASSERT_NE(e, nullptr) << why;
  Problem p = make_problem('N', 'N', 300, 2000, 300, 2, -1.0, 1.0, 42);
  const std::vector<double> cref = reference(p);
  std::vector<double> c = p.c;
  ASSERT_TRUE(e->dgemm('N', 'N', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, p.beta,
                       c.data(), p.ldc, dgemm_));
  bool padding_ok = false;
  EXPECT_LE(rel_error(p, c, cref, &padding_ok), 1e-15 * 8);
  EXPECT_TRUE(padding_ok);
  EXPECT_TRUE(e->broken());
  EXPECT_NE(e->error().find("injected"), std::string::npos) << e->error();
  const hs::gpu::EngineStats s = e->stats();
  EXPECT_GT(s.cpu_flops, 0.0);  // the lost and unclaimed tiles
  EXPECT_GE(s.errors, 1);
  std::vector<double> c2 = p.c;
  EXPECT_FALSE(e->dgemm('N', 'N', p.m, p.n, p.k, p.alpha, p.a.data(), p.lda, p.b.data(), p.ldb, p.beta,
                        c2.data(), p.ldc, dgemm_));
}

}  // namespace
