// GPU offload of dense dgemm (P10 "gpu", src/gpu, docs/BLAS.md §10, results/p10_gpu.md §2): the
// machine's transfer and FP64 rates, and the offload engine (gemm_engine.hpp) against the CPU BLAS on
// the shapes of MUMPS' factorization updates.
//
//   bench_gpu_dgemm info
//   bench_gpu_dgemm copy [--mb 16,64,256] [--reps 5]
//   bench_gpu_dgemm devgemm [--shapes m:n:k,...] [--reps 5]
//   bench_gpu_dgemm gemm [--shapes m:n:k[:ta:tb],...] [--ld L] [--reps 5]
//                        [--modes cpu,gpu,hyb,gpu-p3-s6,hyb-c8,...] [--slot-mb 16] [--mem-mb 1024]
//
// info: loads the CUDA libraries, initializes the engine (device 0 of CUDA_VISIBLE_DEVICES) and
// prints its description, set-up time and the CPUs its pipeline threads may use. copy: host memcpy
// from pageable into pinned memory (one thread), pinned / pageable / strided-pageable
// host-to-device and device-to-host copies, both directions at once, and cudaHostRegister +
// cudaHostUnregister of touched pageable memory (MB/s, best of reps). devgemm: cublasDgemm on
// device-resident operands (GFlop/s, median). gemm: per shape, the modes interleaved per repetition
// on the same operands (C restored before every run): cpu = the BLAS dgemm (all OpenMP threads),
// gpu = the engine as the hook uses it (GPU only: 2 packing threads, 6 slots of --slot-mb,
// non-temporal stores, guided tiles), hyb = the hybrid engine; options appended with '-': -p<N> N
// issuing (packing) threads, -s<N> N pipeline slots, -m<MB> slot size, -k<KB> pack/upload chunk,
// -c<N> N BLAS threads for the CPU share (default max - N - 1), -nt0 regular stores when packing.
// Every mode offloads every shape here (the offload model is printed as model_speedup, not
// applied). Output per shape and mode: wall time median [min-max], GFlop/s, the GPU's share of the
// flops, the speedup over cpu and the largest difference to the cpu result relative to max |C|.
// --ld L stores all three operands with leading dimension L (MUMPS' fronts: lda = ldb = ldc =
// NFRONT); default the tight one.
// Output: one JSON object per line on stdout (tools/bench/gpu_dgemm_table.py makes the tables).
// Run in the measurement slot with exactly one idle GPU visible (tools/bench/gpu_slot.sh).
//
// P10 commands (gpu02, slot 16-31, GPU 7; one slot call; results/p10_gpu.md §2 and §9):
//   S="tools/bench/gpu_slot.sh 7 900 --"; M=results/raw/p10gpu/micro.jsonl
//   $S build/release/bench_gpu_dgemm info > $M
//   $S build/release/bench_gpu_dgemm copy --mb 16,64,256 --reps 5 >> $M
//   $S build/release/bench_gpu_dgemm devgemm --reps 5 --shapes 1000:9000:4000,1000:4000:2048,...
//   $S build/release/bench_gpu_dgemm gemm --ld 15800 --slot-mb 16 --reps 3 --shapes ... >> $M
//      --modes cpu,gpu,gpu-p1,gpu-p3,gpu-s3,gpu-m32,gpu-nt0,gpu-k1024-nt0,hyb,hyb-c8   (same line)
// P10 review fixes (results/p10_gpu.md §10): OpenMP unbound / bound, 3 rounds alternating, and a
// chunk larger than the staging ring:
//   G="gemm --ld 15800 --reps 3 --modes cpu,gpu --shapes 1000:9000:4000,2000:8000:4000,1000:4000:2048"
//   $S build/release/bench_gpu_dgemm $G;  $S env OMP_PROC_BIND=close OMP_PLACES=cores \
//      build/release/bench_gpu_dgemm $G
//   $S build/release/bench_gpu_dgemm gemm --ld 15800 --reps 3 --shapes 2000:8000:4000,1000:9000:4000 \
//      --modes cpu,gpu,gpu-k65536
#include <omp.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gpu/cuda_lib.hpp"
#include "gpu/gemm_engine.hpp"

extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
                       const double*, const int*, const double*, const int*, const double*, double*,
                       const int*, std::size_t, std::size_t);

namespace {

using json = nlohmann::json;
using hs::gpu::CudaLib;

double now_s() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

struct Args {
  std::map<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, sep))
    if (!t.empty()) out.push_back(t);
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void fill(double* p, std::size_t n, std::uint64_t seed) {
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    std::uint64_t x = (static_cast<std::uint64_t>(i) + 1) * 0x9E3779B97F4A7C15ULL ^ seed;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 29;
    p[i] = static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0;
  }
}

const CudaLib* need_lib() {
  std::string why;
  const CudaLib* L = hs::gpu::load(&why);
  if (L == nullptr) {
    std::fprintf(stderr, "bench_gpu_dgemm: %s\n", why.c_str());
    std::exit(2);
  }
  return L;
}

void check(const CudaLib* L, int e, const char* what) {
  if (e != 0) {
    std::fprintf(stderr, "bench_gpu_dgemm: %s: %s\n", what, hs::gpu::error_text(L, e).c_str());
    std::exit(2);
  }
}

json info_json(const hs::gpu::EngineInfo& i) {
  return json{{"device", i.device_name},
              {"pci_bus_id", i.pci_bus_id},
              {"cc", std::to_string(i.compute_major) + "." + std::to_string(i.compute_minor)},
              {"sms", i.multiprocessors},
              {"driver", i.driver_version},
              {"runtime", i.runtime_version},
              {"cublas", i.cublas_version},
              {"total_mib", i.device_total_bytes >> 20},
              {"free_mib", i.device_free_bytes >> 20},
              {"cudart", i.cudart_path},
              {"cublas_path", i.cublas_path},
              {"init_s", i.init_seconds},
              {"chunk_bytes", i.options.chunk_bytes},
              {"pipeline_cpus", i.pipeline_cpus},
              {"pipeline_affinity_widened", i.pipeline_affinity_widened}};
}

int cmd_info() {
  const double t0 = now_s();
  std::string why;
  hs::gpu::EngineOptions opt;
  auto eng = hs::gpu::GemmEngine::create(opt, &why);
  if (!eng) {
    std::printf("%s\n", json{{"what", "info"}, {"ok", false}, {"why", why}}.dump().c_str());
    return 1;
  }
  json j = info_json(eng->info());
  j["what"] = "info";
  j["ok"] = true;
  j["create_s"] = now_s() - t0;
  std::printf("%s\n", j.dump().c_str());
  return 0;
}

int cmd_copy(const Args& args) {
  const CudaLib* L = need_lib();
  check(L, L->SetDevice(0), "cudaSetDevice");
  check(L, L->Free(nullptr), "context");
  const int reps = std::stoi(args.get("reps", "5"));
  hs::gpu::cudaStream_t s0 = nullptr, s1 = nullptr;
  check(L, L->StreamCreateWithFlags(&s0, hs::gpu::kCudaStreamNonBlocking), "stream");
  check(L, L->StreamCreateWithFlags(&s1, hs::gpu::kCudaStreamNonBlocking), "stream");
  for (const std::string& mbs : split(args.get("mb", "16,64,256"), ',')) {
    const std::size_t bytes = std::stoull(mbs) << 20;
    const std::size_t n = bytes / 8;
    std::vector<double> page(n), page2(n);
    fill(page.data(), n, 1);
    fill(page2.data(), n, 2);
    double *pin = nullptr, *pin2 = nullptr, *dev = nullptr, *dev2 = nullptr;
    check(L, L->HostAlloc(reinterpret_cast<void**>(&pin), bytes, 0), "cudaHostAlloc");
    check(L, L->HostAlloc(reinterpret_cast<void**>(&pin2), bytes, 0), "cudaHostAlloc");
    check(L, L->Malloc(reinterpret_cast<void**>(&dev), bytes), "cudaMalloc");
    check(L, L->Malloc(reinterpret_cast<void**>(&dev2), bytes), "cudaMalloc");
    std::memset(pin, 0, bytes);
    std::memset(pin2, 0, bytes);
    auto best = [&](auto&& fn) {
      double b = 1e30;
      for (int r = 0; r < reps; ++r) {
        const double t = now_s();
        fn();
        b = std::min(b, now_s() - t);
      }
      return static_cast<double>(bytes) / b / 1e6;  // MB/s
    };
    json j{{"what", "copy"}, {"mb", bytes >> 20}};
    j["memcpy_pageable_to_pinned"] = best([&] { std::memcpy(pin, page.data(), bytes); });
    j["memcpy_pinned_to_pageable"] = best([&] { std::memcpy(page2.data(), pin2, bytes); });
    j["h2d_pinned"] = best([&] {
      check(L, L->MemcpyAsync(dev, pin, bytes, hs::gpu::kCudaMemcpyHostToDevice, s0), "h2d");
      check(L, L->StreamSynchronize(s0), "sync");
    });
    j["d2h_pinned"] = best([&] {
      check(L, L->MemcpyAsync(pin2, dev, bytes, hs::gpu::kCudaMemcpyDeviceToHost, s0), "d2h");
      check(L, L->StreamSynchronize(s0), "sync");
    });
    j["h2d_pageable"] = best([&] {
      check(L, L->MemcpyAsync(dev, page.data(), bytes, hs::gpu::kCudaMemcpyHostToDevice, s0), "h2d");
      check(L, L->StreamSynchronize(s0), "sync");
    });
    j["d2h_pageable"] = best([&] {
      check(L, L->MemcpyAsync(page2.data(), dev, bytes, hs::gpu::kCudaMemcpyDeviceToHost, s0), "d2h");
      check(L, L->StreamSynchronize(s0), "sync");
    });
    // Both directions at once: MB/s of each direction.
    j["bidir_pinned_each"] = best([&] {
      check(L, L->MemcpyAsync(dev, pin, bytes, hs::gpu::kCudaMemcpyHostToDevice, s0), "h2d");
      check(L, L->MemcpyAsync(pin2, dev2, bytes, hs::gpu::kCudaMemcpyDeviceToHost, s1), "d2h");
      check(L, L->StreamSynchronize(s0), "sync");
      check(L, L->StreamSynchronize(s1), "sync");
    });
    // Register + unregister of touched pageable memory (the alternative to staging copies).
    j["register_unregister"] = best([&] {
      check(L, L->HostRegister(page.data(), bytes, hs::gpu::kCudaHostRegisterDefault), "register");
      check(L, L->HostUnregister(page.data()), "unregister");
    });
    // Strided pageable source (columns of 4000 doubles, 15800 apart: a MUMPS front's U block),
    // copied by the driver's own staging (cudaMemcpy2DAsync from pageable memory).
    {
      const std::size_t rows = 4000, ld = 15800, cols = n / ld;
      if (cols > 0) {
        const double b2 = [&] {
          double b = 1e30;
          for (int r = 0; r < reps; ++r) {
            const double t = now_s();
            check(L, L->Memcpy2DAsync(dev, rows * 8, page.data(), ld * 8, rows * 8, cols,
                                      hs::gpu::kCudaMemcpyHostToDevice, s0), "h2d 2d");
            check(L, L->StreamSynchronize(s0), "sync");
            b = std::min(b, now_s() - t);
          }
          return b;
        }();
        j["h2d_pageable_2d_strided"] = static_cast<double>(rows * cols * 8) / b2 / 1e6;
      }
    }
    j["h2d_registered"] = [&] {
      check(L, L->HostRegister(page.data(), bytes, hs::gpu::kCudaHostRegisterDefault), "register");
      const double r = best([&] {
        check(L, L->MemcpyAsync(dev, page.data(), bytes, hs::gpu::kCudaMemcpyHostToDevice, s0), "h2d");
        check(L, L->StreamSynchronize(s0), "sync");
      });
      check(L, L->HostUnregister(page.data()), "unregister");
      return r;
    }();
    std::printf("%s\n", j.dump().c_str());
    std::fflush(stdout);
    L->Free(dev);
    L->Free(dev2);
    L->FreeHost(pin);
    L->FreeHost(pin2);
  }
  return 0;
}

struct Shape {
  int m, n, k;
  char ta = 'N', tb = 'N';
};

std::vector<Shape> parse_shapes(const std::string& s) {
  std::vector<Shape> out;
  for (const std::string& t : split(s, ',')) {
    const auto f = split(t, ':');
    if (f.size() != 3 && f.size() != 5) {
      std::fprintf(stderr, "bench_gpu_dgemm: bad shape '%s' (m:n:k[:ta:tb])\n", t.c_str());
      std::exit(2);
    }
    Shape sh{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2])};
    if (f.size() == 5) sh.ta = f[3][0], sh.tb = f[4][0];
    out.push_back(sh);
  }
  return out;
}

int cmd_devgemm(const Args& args) {
  const CudaLib* L = need_lib();
  check(L, L->SetDevice(0), "cudaSetDevice");
  hs::gpu::cublasHandle_t h = nullptr;
  if (L->BlasCreate(&h) != 0) return 2;
  const int reps = std::stoi(args.get("reps", "5"));
  for (const Shape& s : parse_shapes(args.get("shapes", "1000:8000:4000,2000:8000:4000,4000:4000:4000,8192:8192:8192"))) {
    const std::size_t na = static_cast<std::size_t>(s.m) * s.k, nb = static_cast<std::size_t>(s.k) * s.n,
                      nc = static_cast<std::size_t>(s.m) * s.n;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    check(L, L->Malloc(reinterpret_cast<void**>(&a), 8 * na), "malloc");
    check(L, L->Malloc(reinterpret_cast<void**>(&b), 8 * nb), "malloc");
    check(L, L->Malloc(reinterpret_cast<void**>(&c), 8 * nc), "malloc");
    const double alpha = -1.0, beta = 1.0;
    std::vector<double> t;
    for (int r = 0; r <= reps; ++r) {
      check(L, L->DeviceSynchronize(), "sync");
      const double t0 = now_s();
      L->Dgemm(h, 0, 0, s.m, s.n, s.k, &alpha, a, s.m, b, s.k, &beta, c, s.m);
      check(L, L->DeviceSynchronize(), "sync");
      if (r > 0) t.push_back(now_s() - t0);  // the first run warms up
    }
    const double tm = median(t);
    std::printf("%s\n", json{{"what", "devgemm"},
                             {"m", s.m},
                             {"n", s.n},
                             {"k", s.k},
                             {"s", tm},
                             {"gflops", 2.0 * s.m * s.n * static_cast<double>(s.k) / tm / 1e9}}
                            .dump()
                            .c_str());
    std::fflush(stdout);
    L->Free(a);
    L->Free(b);
    L->Free(c);
  }
  return 0;
}

int cmd_gemm(const Args& args) {
  const int reps = std::stoi(args.get("reps", "5"));
  const long ldarg = std::stol(args.get("ld", "0"));
  const auto modes = split(args.get("modes", "cpu,gpu,hyb"), ',');
  hs::gpu::EngineOptions base;
  base.slot_bytes = std::stoull(args.get("slot-mb", "16")) << 20;
  base.mem_bytes = std::stoull(args.get("mem-mb", "1024")) << 20;
  base.min_flops = 0.0;  // every shape is offloaded here (the model is printed, not applied)
  base.min_speedup = 0.0;
  std::map<std::string, std::unique_ptr<hs::gpu::GemmEngine>> engines;
  auto engine_for = [&](const std::string& mode) -> hs::gpu::GemmEngine* {
    if (mode == "cpu") return nullptr;
    auto it = engines.find(mode);
    if (it != engines.end()) return it->second.get();
    // gpu | hyb, then options: -p<N> issuing (packing) threads, -c<N> BLAS threads of the CPU
    // share, -nt0 plain memcpy instead of non-temporal stores when packing.
    hs::gpu::EngineOptions o = base;
    const auto parts = split(mode, '-');
    if (parts.empty() || (parts[0] != "gpu" && parts[0] != "hyb")) {
      std::fprintf(stderr, "bench_gpu_dgemm: unknown mode '%s'\n", mode.c_str());
      std::exit(2);
    }
    o.hybrid = parts[0] == "hyb";
    for (std::size_t i = 1; i < parts.size(); ++i) {
      const std::string& t = parts[i];
      if (t[0] == 'p') o.pack_threads = std::stoi(t.substr(1));
      else if (t[0] == 's') o.slots = std::stoi(t.substr(1));
      else if (t[0] == 'm') o.slot_bytes = std::stoull(t.substr(1)) << 20;
      else if (t[0] == 'k') o.chunk_bytes = std::stoull(t.substr(1)) << 10;
      else if (t[0] == 'c') o.cpu_threads = std::stoi(t.substr(1));
      else if (t == "nt0") o.stream_stores = false;
      else {
        std::fprintf(stderr, "bench_gpu_dgemm: unknown mode option '%s'\n", t.c_str());
        std::exit(2);
      }
    }
    std::string why;
    auto e = hs::gpu::GemmEngine::create(o, &why);
    if (!e) {
      std::fprintf(stderr, "bench_gpu_dgemm: engine: %s\n", why.c_str());
      std::exit(2);
    }
    std::printf("%s\n", [&] {
      json j = info_json(e->info());
      j["what"] = "engine";
      j["mode"] = mode;
      return j.dump();
    }().c_str());
    auto* p = e.get();
    engines[mode] = std::move(e);
    return p;
  };
  for (const std::string& mode : modes) engine_for(mode);

  for (const Shape& s : parse_shapes(args.get("shapes", "1000:8000:4000"))) {
    const bool ta = s.ta == 'T' || s.ta == 't', tb = s.tb == 'T' || s.tb == 't';
    const int ra = ta ? s.k : s.m, ca = ta ? s.m : s.k;
    const int rb = tb ? s.n : s.k, cb = tb ? s.k : s.n;
    const int lda = std::max<long>(ra, ldarg), ldb = std::max<long>(rb, ldarg),
              ldc = std::max<long>(s.m, ldarg);
    std::vector<double> a(static_cast<std::size_t>(lda) * ca), b(static_cast<std::size_t>(ldb) * cb),
        c0(static_cast<std::size_t>(ldc) * s.n), c(c0.size()), cref(c0.size());
    fill(a.data(), a.size(), 11);
    fill(b.data(), b.size(), 12);
    fill(c0.data(), c0.size(), 13);
    const double alpha = -1.0, beta = 1.0;
    const double flops = 2.0 * s.m * s.n * static_cast<double>(s.k);
    std::map<std::string, std::vector<double>> times;
    std::map<std::string, double> diff, share;
    auto run = [&](const std::string& mode) {
      std::memcpy(c.data(), c0.data(), 8 * c.size());
      hs::gpu::GemmEngine* e = engine_for(mode);
      const hs::gpu::EngineStats before = e ? e->stats() : hs::gpu::EngineStats{};
      const double t0 = now_s();
      if (e == nullptr) {
        dgemm_(&s.ta, &s.tb, &s.m, &s.n, &s.k, &alpha, a.data(), &lda, b.data(), &ldb, &beta, c.data(),
               &ldc, 1, 1);
      } else if (!e->dgemm(s.ta, s.tb, s.m, s.n, s.k, alpha, a.data(), lda, b.data(), ldb, beta,
                           c.data(), ldc, dgemm_)) {
        std::fprintf(stderr, "bench_gpu_dgemm: engine declined %d:%d:%d (%s)\n", s.m, s.n, s.k,
                     e->error().c_str());
        std::exit(2);
      }
      times[mode].push_back(now_s() - t0);
      if (e != nullptr) {
        const hs::gpu::EngineStats after = e->stats();
        share[mode] += (after.gpu_flops - before.gpu_flops) / flops;
      }
      if (mode == "cpu") {
        cref = c;
      } else if (!cref.empty()) {
        double dmax = 0.0, cmax = 0.0;
        for (int j = 0; j < s.n; ++j)
          for (int i = 0; i < s.m; ++i) {
            const std::size_t q = i + static_cast<std::size_t>(j) * ldc;
            dmax = std::max(dmax, std::abs(c[q] - cref[q]));
            cmax = std::max(cmax, std::abs(cref[q]));
          }
        diff[mode] = std::max(diff[mode], dmax / std::max(cmax, 1e-300));
      }
    };
    // Warm-up of every mode once, then interleaved repetitions (order reversed every other one).
    run("cpu");
    for (const std::string& mode : modes)
      if (mode != "cpu") run(mode);
    times.clear();
    share.clear();
    for (int r = 0; r < reps; ++r) {
      std::vector<std::string> order = modes;
      if (r % 2) std::reverse(order.begin(), order.end());
      if (std::find(order.begin(), order.end(), "cpu") == order.end()) order.insert(order.begin(), "cpu");
      for (const std::string& mode : order) run(mode);
    }
    for (const std::string& mode : modes) {
      const auto& t = times[mode];
      const double tm = median(t);
      json j{{"what", "gemm"},  {"m", s.m},  {"n", s.n},     {"k", s.k},
             {"ta", std::string(1, s.ta)}, {"tb", std::string(1, s.tb)}, {"ld", ldc},
             {"mode", mode},    {"s", tm},   {"s_min", *std::min_element(t.begin(), t.end())},
             {"s_max", *std::max_element(t.begin(), t.end())},
             {"gflops", flops / tm / 1e9}};
      if (mode != "cpu") {
        j["gpu_share"] = share[mode] / static_cast<double>(t.size());
        j["max_rel_diff"] = diff[mode];
        j["speedup"] = median(times["cpu"]) / tm;
        j["model_speedup"] = engine_for(mode)->predicted_speedup(s.m, s.n, s.k);
      }
      std::printf("%s\n", j.dump().c_str());
    }
    std::fflush(stdout);
  }
  for (auto& [mode, e] : engines) {
    const auto st = e->stats();
    std::printf("%s\n", json{{"what", "engine_stats"},
                             {"mode", mode},
                             {"calls", st.calls},
                             {"tiles", st.tiles},
                             {"gpu_flops", st.gpu_flops},
                             {"cpu_flops", st.cpu_flops},
                             {"h2d_mb", st.h2d_bytes / 1e6},
                             {"d2h_mb", st.d2h_bytes / 1e6},
                             {"errors", st.errors},
                             {"error", e->error()},
                             {"seconds", st.seconds},
                             {"shared_s", st.shared_s},
                             {"pack_s", st.pack_s},
                             {"slot_wait_s", st.slot_wait_s},
                             {"sync_s", st.sync_s},
                             {"add_s", st.add_s},
                             {"cpu_s", st.cpu_s},
                             {"tail_s", st.tail_s}}
                            .dump()
                            .c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_gpu_dgemm info|copy|devgemm|gemm [--key value]...\n");
    return 2;
  }
  const std::string cmd = argv[1];
  Args args;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) {
      std::fprintf(stderr, "bench_gpu_dgemm: expected --key, got '%s'\n", k.c_str());
      return 2;
    }
    args.kv[k.substr(2)] = argv[i + 1];
  }
  if (cmd == "info") return cmd_info();
  if (cmd == "copy") return cmd_copy(args);
  if (cmd == "devgemm") return cmd_devgemm(args);
  if (cmd == "gemm") return cmd_gemm(args);
  std::fprintf(stderr, "bench_gpu_dgemm: unknown command '%s'\n", cmd.c_str());
  return 2;
}
