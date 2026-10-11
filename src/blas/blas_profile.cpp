// BLAS call profiler (blas_profile.hpp).
#include "blas/blas_profile.hpp"

#include <omp.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace hs::blas::prof {
namespace {

struct Stat {
  std::int64_t calls = 0;
  double seconds = 0.0;
  double flops = 0.0;
  double bytes = 0.0;
};

struct MumpsStat {
  std::int64_t calls = 0;
  double seconds = 0.0;
  std::int64_t rhs = 0;
};

// Key layout (low to high): route 3 bits | in parallel 1 | phase 3 | k class 6 | n class 6 |
// m class 6 | flags 4 | op 4.
std::uint64_t make_key(Op op, std::uint8_t flags, int mc, int nc, int kc, Phase ph, bool par,
                       std::uint8_t route) {
  std::uint64_t key = static_cast<std::uint64_t>(op) & 0xFu;
  key = (key << 4) | (flags & 0xFu);
  key = (key << 6) | static_cast<std::uint64_t>(mc & 63);
  key = (key << 6) | static_cast<std::uint64_t>(nc & 63);
  key = (key << 6) | static_cast<std::uint64_t>(kc & 63);
  key = (key << 3) | static_cast<std::uint64_t>(static_cast<unsigned>(ph) & 7u);
  key = (key << 1) | (par ? 1u : 0u);
  key = (key << 3) | (route & 7u);
  return key;
}

struct Table {
  std::unordered_map<std::uint64_t, Stat> calls;
};

// Process-level registry of the per-thread tables (never freed: threads may outlive the dump).
// A pthread mutex (as the MUMPS call lock, src/factor/mumps_api.hpp): std::mutex can be a no-op
// in executables linked without libpthread.
pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;
std::vector<Table*>* g_tables = nullptr;
MumpsStat g_mumps[static_cast<int>(Phase::kCount)];
std::atomic<int> g_phase{0};
std::string* g_route_info = nullptr;
std::string* g_blas_info = nullptr;

thread_local Table* t_table = nullptr;

Table& my_table() {
  if (t_table == nullptr) {
    auto* t = new Table;
    pthread_mutex_lock(&g_registry_mutex);
    if (g_tables == nullptr) g_tables = new std::vector<Table*>;
    g_tables->push_back(t);
    pthread_mutex_unlock(&g_registry_mutex);
    t_table = t;
  }
  return *t_table;
}

const char* profile_path() noexcept {
  static const char* const path = [] {
    const char* v = std::getenv("HS_BLAS_PROFILE");
    return (v != nullptr && *v != '\0') ? v : nullptr;
  }();
  return path;
}

std::string flags_string(Op op, unsigned flags) {
  switch (op) {
    case Op::Dgemm:
      return std::string{(flags & 1u) ? 'T' : 'N', (flags & 2u) ? 'T' : 'N'};
    case Op::Dtrsm:
      return std::string{(flags & 1u) ? 'R' : 'L', (flags & 2u) ? 'L' : 'U',
                         (flags & 4u) ? 'T' : 'N', (flags & 8u) ? 'U' : 'N'};
    case Op::Dgemv:
      return std::string{(flags & 1u) ? 'T' : 'N'};
    default:
      return "";
  }
}

const char* route_label(unsigned r) {
  static const char* const names[] = {"real",     "gemv",   "trsv",  "narrow",
                                      "parallel", "splitk", "strsm", "?"};
  return names[r & 7u];
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (const char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    if (static_cast<unsigned char>(ch) >= 0x20) out += ch;
  }
  return out;
}

void write_json(std::FILE* f) {
  std::map<std::uint64_t, Stat> merged;
  MumpsStat mumps[static_cast<int>(Phase::kCount)];
  std::string route, blas;
  pthread_mutex_lock(&g_registry_mutex);
  if (g_tables != nullptr)
    for (const Table* t : *g_tables)
      for (const auto& [key, s] : t->calls) {
        Stat& m = merged[key];
        m.calls += s.calls;
        m.seconds += s.seconds;
        m.flops += s.flops;
        m.bytes += s.bytes;
      }
  for (int p = 0; p < static_cast<int>(Phase::kCount); ++p) mumps[p] = g_mumps[p];
  if (g_route_info != nullptr) route = *g_route_info;
  if (g_blas_info != nullptr) blas = *g_blas_info;
  pthread_mutex_unlock(&g_registry_mutex);

  char exe[512] = {0};
  const ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (len > 0) exe[len] = '\0';
  std::fprintf(f,
               "{\"pid\": %d, \"exe\": \"%s\", \"omp_max_threads\": %d, \"route\": \"%s\", "
               "\"blas\": \"%s\",\n",
               static_cast<int>(getpid()), json_escape(exe).c_str(), omp_get_max_threads(),
               json_escape(route).c_str(), json_escape(blas).c_str());
  std::fprintf(f, " \"mumps\": [");
  bool first = true;
  for (int p = 0; p < static_cast<int>(Phase::kCount); ++p) {
    const MumpsStat& s = mumps[p];
    if (s.calls == 0) continue;
    std::fprintf(f, "%s\n  {\"phase\": \"%s\", \"calls\": %lld, \"seconds\": %.6f, \"rhs\": %lld}",
                 first ? "" : ",", phase_name(static_cast<Phase>(p)),
                 static_cast<long long>(s.calls), s.seconds, static_cast<long long>(s.rhs));
    first = false;
  }
  std::fprintf(f, "],\n \"calls\": [");
  first = true;
  for (const auto& [key, s] : merged) {
    std::uint64_t k = key;
    const unsigned route_id = k & 7u;
    k >>= 3;
    const bool par = (k & 1u) != 0;
    k >>= 1;
    const unsigned ph = k & 7u;
    k >>= 3;
    const unsigned kc = k & 63u;
    k >>= 6;
    const unsigned nc = k & 63u;
    k >>= 6;
    const unsigned mc = k & 63u;
    k >>= 6;
    const unsigned flags = k & 0xFu;
    k >>= 4;
    const Op op = static_cast<Op>(k & 0xFu);
    std::fprintf(f,
                 "%s\n  {\"op\": \"%s\", \"flags\": \"%s\", \"mc\": %u, \"nc\": %u, \"kc\": %u, "
                 "\"phase\": \"%s\", \"par\": %d, \"route\": \"%s\", \"calls\": %lld, "
                 "\"seconds\": %.9g, \"flops\": %.9g, \"bytes\": %.9g}",
                 first ? "" : ",", op_name(op), flags_string(op, flags).c_str(), mc, nc, kc,
                 phase_name(static_cast<Phase>(ph)), par ? 1 : 0, route_label(route_id),
                 static_cast<long long>(s.calls), s.seconds, s.flops, s.bytes);
    first = false;
  }
  std::fprintf(f, "]}\n");
}

struct ExitDump {
  ~ExitDump() { dump(); }
};
ExitDump g_exit_dump;

}  // namespace

const char* op_name(Op op) noexcept {
  static const char* const names[] = {"dgemm", "dtrsm", "dgemv", "daxpy", "dcopy",
                                      "dscal", "dswap", "dnrm2", "idamax"};
  const auto i = static_cast<unsigned>(op);
  return i < static_cast<unsigned>(Op::kCount) ? names[i] : "?";
}

const char* phase_name(Phase p) noexcept {
  static const char* const names[] = {"none",   "analysis", "factorization", "count",
                                      "solve1", "solvep",   "other"};
  const auto i = static_cast<unsigned>(p);
  return i < static_cast<unsigned>(Phase::kCount) ? names[i] : "?";
}

int size_class(std::int64_t x) noexcept {
  int c = 0;
  while (x > 0) {
    ++c;
    x >>= 1;
  }
  return c;
}

bool enabled() noexcept {
  return profile_path() != nullptr;
}

void set_phase(Phase p) noexcept {
  g_phase.store(static_cast<int>(p), std::memory_order_relaxed);
}

Phase phase() noexcept {
  return static_cast<Phase>(g_phase.load(std::memory_order_relaxed));
}

namespace {
bool is_trans(char c) {
  return !(c == 'N' || c == 'n');
}
}  // namespace

std::uint8_t gemm_flags(char ta, char tb) noexcept {
  return static_cast<std::uint8_t>((is_trans(ta) ? 1u : 0u) | (is_trans(tb) ? 2u : 0u));
}

std::uint8_t trsm_flags(char side, char uplo, char ta, char diag) noexcept {
  return static_cast<std::uint8_t>(((side == 'R' || side == 'r') ? 1u : 0u) |
                                   ((uplo == 'L' || uplo == 'l') ? 2u : 0u) |
                                   (is_trans(ta) ? 4u : 0u) |
                                   ((diag == 'U' || diag == 'u') ? 8u : 0u));
}

std::uint8_t gemv_flags(char t) noexcept {
  return is_trans(t) ? 1u : 0u;
}

double now() noexcept {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

void record(Op op, std::uint8_t flags, std::int64_t m, std::int64_t n, std::int64_t k,
            double flops, double bytes, double seconds, std::uint8_t route) noexcept {
  const std::uint64_t key = make_key(op, flags, size_class(m), size_class(n), size_class(k),
                                     phase(), omp_in_parallel() != 0, route);
  Stat& s = my_table().calls[key];
  ++s.calls;
  s.seconds += seconds;
  s.flops += flops;
  s.bytes += bytes;
}

void record_mumps(Phase p, double seconds, std::int64_t nrhs) noexcept {
  pthread_mutex_lock(&g_registry_mutex);
  MumpsStat& s = g_mumps[static_cast<int>(p)];
  ++s.calls;
  s.seconds += seconds;
  s.rhs += nrhs;
  pthread_mutex_unlock(&g_registry_mutex);
}

void set_info(const std::string& route, const std::string& blas_library) {
  pthread_mutex_lock(&g_registry_mutex);
  if (g_route_info == nullptr) g_route_info = new std::string;
  if (g_blas_info == nullptr) g_blas_info = new std::string;
  *g_route_info = route;
  *g_blas_info = blas_library;
  pthread_mutex_unlock(&g_registry_mutex);
}

void dump() noexcept {
  const char* pattern = profile_path();
  if (pattern == nullptr) return;
  // A process that made no BLAS or MUMPS call (e.g. a wrapper that only inherited the
  // environment) writes nothing, so it cannot clobber the file of the process that did.
  pthread_mutex_lock(&g_registry_mutex);
  bool any = g_tables != nullptr && !g_tables->empty();
  for (const MumpsStat& s : g_mumps) any = any || s.calls > 0;
  pthread_mutex_unlock(&g_registry_mutex);
  if (!any) return;
  std::string path;
  for (const char* p = pattern; *p != '\0'; ++p) {
    if (p[0] == '%' && p[1] == 'p') {
      path += std::to_string(static_cast<long long>(getpid()));
      ++p;
    } else {
      path += *p;
    }
  }
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (f == nullptr) return;
  write_json(f);
  std::fclose(f);
}

}  // namespace hs::blas::prof
