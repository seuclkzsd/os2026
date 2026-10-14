// GPU module: process-level GPU offload of the BLAS hook (offload.hpp).
#include "gpu/offload.hpp"

#include <omp.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

// Mode when neither HS_GPU nor set_mode() says otherwise (CMake cache HSOLVE_GPU_DEFAULT).
#ifndef HSOLVE_GPU_DEFAULT
#define HSOLVE_GPU_DEFAULT "off"
#endif

namespace hs::gpu {
namespace {

[[noreturn]] void fatal(const std::string& msg) noexcept {
  const std::string line = "hsolve GPU offload: " + msg + "\n";
  // write(2), not stdio: may run inside a Fortran BLAS call of any thread.
  [[maybe_unused]] const ssize_t rc = ::write(2, line.data(), line.size());
  std::abort();
}

double now_s() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

bool parse_double(const std::string& s, double lo, double hi, double* out) {
  if (s.empty()) return false;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == nullptr || *end != '\0' || !std::isfinite(v) || v < lo || v > hi) return false;
  *out = v;
  return true;
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += ch;
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", v);
  return buf;
}

struct State {
  std::atomic<int> mode{-1};  // -1: not resolved yet
  std::atomic<bool> window{false};
  std::atomic<GemmEngine*> engine{nullptr};  // never deleted: usable until the process ends
  std::mutex mu;  // everything below
  bool opts_read = false;
  OffloadOptions opts;
  bool tried = false;  // engine creation attempted
  std::string reason;  // why no engine
  double init_s = 0.0;
  long long factorizations = 0;  // seen while the mode was not off
  long long offloaded = 0;       // of which offloaded
  double est_flops_offloaded = 0.0;
  double window_s = 0.0;
  double window_t0 = 0.0;
};

State& state() {
  static State* s = new State;  // leaked on purpose: MUMPS may run during static destruction
  return *s;
}

// HS_GPU_REPORT=<file> ("%p" = process id): report_json() is written there at exit.
void write_report_at_exit() {
  const char* v = std::getenv("HS_GPU_REPORT");
  if (v == nullptr || *v == '\0') return;
  std::string path = v;
  const std::size_t p = path.find("%p");
  if (p != std::string::npos) path.replace(p, 2, std::to_string(::getpid()));
  if (std::FILE* f = std::fopen(path.c_str(), "w")) {
    const std::string j = report_json();
    std::fwrite(j.data(), 1, j.size(), f);
    std::fputc('\n', f);
    std::fclose(f);
  }
}

void register_report_at_exit() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* r = std::getenv("HS_GPU_REPORT");
    if (r != nullptr && *r != '\0') std::atexit(write_report_at_exit);
  });
}

const OffloadOptions& options_locked(State& s) {
  if (!s.opts_read) {
    const char* v = std::getenv("HS_GPU_OPTS");
    std::string unknown;
    s.opts = parse_offload_options(v != nullptr ? v : "", &unknown);
    if (!unknown.empty()) fatal("unknown HS_GPU_OPTS token(s) '" + unknown + "'");
    s.opts_read = true;
  }
  return s.opts;
}

}  // namespace

const char* mode_name(Mode m) noexcept {
  switch (m) {
    case Mode::Off:
      return "off";
    case Mode::Auto:
      return "auto";
    case Mode::On:
      return "on";
  }
  return "?";
}

bool parse_mode(const std::string& s, Mode* out) noexcept {
  if (s == "off") *out = Mode::Off;
  else if (s == "auto") *out = Mode::Auto;
  else if (s == "on") *out = Mode::On;
  else return false;
  return true;
}

OffloadOptions parse_offload_options(const std::string& spec, std::string* unknown) {
  OffloadOptions o;
  std::size_t pos = 0;
  while (pos <= spec.size()) {
    std::size_t end = spec.find(',', pos);
    if (end == std::string::npos) end = spec.size();
    std::string tok = spec.substr(pos, end - pos);
    pos = end + 1;
    tok.erase(0, tok.find_first_not_of(" \t"));
    tok.erase(tok.find_last_not_of(" \t") + 1);
    if (tok.empty()) continue;
    const std::size_t eq = tok.find('=');
    const std::string key = tok.substr(0, eq);
    const std::string val = eq == std::string::npos ? "" : tok.substr(eq + 1);
    double v = 0.0;
    EngineOptions& e = o.engine;
    bool ok = eq != std::string::npos;
    if (!ok) {
    } else if (key == "device" && parse_double(val, 0, 64, &v) && v == std::floor(v)) {
      e.device = static_cast<int>(v);
    } else if (key == "mem_mb" && parse_double(val, 64, 1 << 20, &v)) {
      e.mem_bytes = static_cast<std::size_t>(v) << 20;
    } else if (key == "slot_mb" && parse_double(val, 1, 1024, &v)) {
      e.slot_bytes = static_cast<std::size_t>(v) << 20;
    } else if (key == "chunk_kb" && parse_double(val, 0, 1 << 20, &v)) {
      e.chunk_bytes = static_cast<std::size_t>(v) << 10;
    } else if (key == "nt" && (val == "0" || val == "1")) {
      e.stream_stores = val == "1";
    } else if (key == "pack" && parse_double(val, 1, 4, &v) && v == std::floor(v)) {
      e.pack_threads = static_cast<int>(v);
    } else if (key == "slots" && parse_double(val, 2, 8, &v) && v == std::floor(v)) {
      e.slots = static_cast<int>(v);
    } else if (key == "hybrid" && (val == "0" || val == "1")) {
      e.hybrid = val == "1";
    } else if (key == "cpu_threads" && parse_double(val, 0, 4096, &v) && v == std::floor(v)) {
      e.cpu_threads = static_cast<int>(v);
    } else if (key == "min_gflop" && parse_double(val, 0, 1e9, &v)) {
      e.min_flops = v * 1e9;
    } else if (key == "cpu_gflops" && parse_double(val, 0, 1e9, &v)) {
      e.cpu_gflops = v;
    } else if (key == "host_gbs" && parse_double(val, 0.1, 1e6, &v)) {
      e.host_gbs = v;
    } else if (key == "min_speedup" && parse_double(val, 0, 1e6, &v)) {
      e.min_speedup = v;
    } else if (key == "auto_min_tflop" && parse_double(val, 0, 1e9, &v)) {
      o.auto_min_flops = v * 1e12;
    } else if (key == "fault_after" && parse_double(val, -1, 1e15, &v) && v == std::floor(v)) {
      e.inject_fault_after_tiles = static_cast<std::int64_t>(v);
    } else {
      ok = false;
    }
    if (!ok && unknown != nullptr) {
      if (!unknown->empty()) *unknown += ',';
      *unknown += tok;
    }
  }
  return o;
}

Mode mode() noexcept {
  State& s = state();
  int m = s.mode.load(std::memory_order_acquire);
  if (m >= 0) return static_cast<Mode>(m);
  const char* v = std::getenv("HS_GPU");
  const std::string name = v != nullptr && *v != '\0' ? v : HSOLVE_GPU_DEFAULT;
  Mode parsed = Mode::Off;
  if (!parse_mode(name, &parsed))
    fatal("HS_GPU must be off, auto or on, got '" + name + "'");
  int expected = -1;
  s.mode.compare_exchange_strong(expected, static_cast<int>(parsed), std::memory_order_acq_rel);
  register_report_at_exit();
  return static_cast<Mode>(s.mode.load(std::memory_order_acquire));
}

bool set_mode(const std::string& name) noexcept {
  Mode m = Mode::Off;
  if (!parse_mode(name, &m)) return false;
  state().mode.store(static_cast<int>(m), std::memory_order_release);
  register_report_at_exit();
  return true;
}

void factorization_begin(double est_flops) {
  const Mode m = mode();
  if (m == Mode::Off) return;
  State& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  ++s.factorizations;
  const OffloadOptions& o = options_locked(s);
  const bool want = m == Mode::On || (est_flops >= 0.0 && est_flops >= o.auto_min_flops);
  if (!want) return;
  if (!s.tried) {
    s.tried = true;
    const double t0 = now_s();
    std::unique_ptr<GemmEngine> e;
    try {  // called from the dmumps_c wrapper: nothing may propagate into MUMPS' C / Fortran frames
      e = GemmEngine::create(o.engine, &s.reason);
    } catch (const std::exception& ex) {
      s.reason = std::string("GPU set-up failed: ") + ex.what();
    }
    s.init_s = now_s() - t0;
    s.engine.store(e.release(), std::memory_order_release);
  }
  GemmEngine* e = s.engine.load(std::memory_order_acquire);
  if (e == nullptr || e->broken()) return;
  ++s.offloaded;
  if (est_flops > 0.0) s.est_flops_offloaded += est_flops;
  s.window_t0 = now_s();
  s.window.store(true, std::memory_order_release);
}

void factorization_end() {
  State& s = state();
  if (!s.window.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lk(s.mu);
  s.window.store(false, std::memory_order_release);
  s.window_s += now_s() - s.window_t0;
}

bool window_open() noexcept {
  return state().window.load(std::memory_order_acquire);
}

bool offload_dgemm(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta, double* c, int ldc, CpuDgemm real) {
  State& s = state();
  if (!s.window.load(std::memory_order_acquire) || omp_in_parallel()) return false;
  GemmEngine* e = s.engine.load(std::memory_order_acquire);
  return e != nullptr && e->dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, real);
}

std::string report_json() {
  const Mode m = mode();
  State& s = state();
  std::lock_guard<std::mutex> lk(s.mu);
  GemmEngine* e = s.engine.load(std::memory_order_acquire);
  std::string j = "{\"mode\": " + json_string(mode_name(m));
  if (!s.tried)
    j += ", \"available\": null";  // never needed: no CUDA library was loaded
  else
    j += std::string(", \"available\": ") + (e != nullptr ? "true" : "false");
  if (!s.reason.empty()) j += ", \"reason\": " + json_string(s.reason);
  j += ", \"factorizations\": " + std::to_string(s.factorizations);
  j += ", \"factorizations_offloaded\": " + std::to_string(s.offloaded);
  if (e != nullptr) {
    const EngineInfo& inf = e->info();
    const EngineStats st = e->stats();
    j += ", \"device\": " + json_string(inf.device_name);
    j += ", \"pci_bus_id\": " + json_string(inf.pci_bus_id);
    j += ", \"compute_capability\": " +
         json_string(std::to_string(inf.compute_major) + "." + std::to_string(inf.compute_minor));
    j += ", \"cuda_runtime\": " + std::to_string(inf.runtime_version);
    j += ", \"cublas\": " + std::to_string(inf.cublas_version);
    j += ", \"init_s\": " + num(s.init_s);
    j += ", \"device_mem_mb\": " + std::to_string(inf.options.mem_bytes >> 20);
    j += ", \"pinned_mb\": " + std::to_string(inf.pinned_bytes >> 20);
    j += ", \"pipeline_cpus\": " + std::to_string(inf.pipeline_cpus);
    j += ", \"window_s\": " + num(s.window_s);
    j += ", \"calls\": " + std::to_string(st.calls);
    j += ", \"tiles\": " + std::to_string(st.tiles);
    j += ", \"gpu_gflop\": " + num(st.gpu_flops / 1e9);
    j += ", \"cpu_gflop_in_calls\": " + num(st.cpu_flops / 1e9);
    j += ", \"estimated_gflop\": " + num(s.est_flops_offloaded / 1e9);
    j += ", \"gpu_share\": " +
         num(s.est_flops_offloaded > 0.0 ? st.gpu_flops / s.est_flops_offloaded : 0.0);
    j += ", \"call_s\": " + num(st.seconds);
    j += ", \"predicted_cpu_s\": " + num(st.predicted_cpu_s);
    j += ", \"h2d_gb\": " + num(st.h2d_bytes / 1e9);
    j += ", \"d2h_gb\": " + num(st.d2h_bytes / 1e9);
    j += ", \"pipeline_s\": {\"shared\": " + num(st.shared_s) + ", \"pack\": " + num(st.pack_s) +
         ", \"slot_wait\": " + num(st.slot_wait_s) + ", \"sync\": " + num(st.sync_s) +
         ", \"add\": " + num(st.add_s) + ", \"cpu\": " + num(st.cpu_s) + ", \"tail\": " + num(st.tail_s) +
         "}";
    j += ", \"declined_busy\": " + std::to_string(st.declined_busy);
    j += ", \"errors\": " + std::to_string(st.errors);
    if (e->broken()) j += ", \"error\": " + json_string(e->error());
  } else if (s.tried) {
    j += ", \"init_s\": " + num(s.init_s);
  }
  return j + "}";
}

}  // namespace hs::gpu
