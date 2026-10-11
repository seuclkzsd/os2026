// End-to-end memory profile of hsolve (module memio): where does the peak RSS come from?
//
//   bench_memio_e2e --A A.npz [--B B.npz] [--config config.json] [--out DIR] [--sample-ms 2]
//   bench_memio_e2e --A A.npz [--B B.npz] [--config config.json] --analyze-only
//
// Full mode replays apps/hsolve_cli.cpp in one process - load A and B (hs::load_matrix), promote
// to complex as the CLI does, hs::solve with a JSON-lines Logger, write eigenvalues/eigenvectors
// with write_npy_vector/matrix - while a sampler thread records VmRSS every --sample-ms. The
// bench's own marks and the driver's log events (problem, ordering, analysis, b_inertia,
// factorization, lanczos/krylov_schur, certificate, done) split the run into phases; for each
// phase one JSON line gives its time span, the RSS at its start and end and the largest sample
// in between (spikes shorter than the sampling period can be missed: the final line also gives
// the kernel's VmHWM). A "components" line gives the sizes the phases are made of: the input CSR
// arrays, the pencil pattern, MUMPS's input arrays, the factor (MUMPS INFOG(21)/(18) from the
// report), the Krylov basis and the eigenvector output, computed from the problem dimensions.
//
// --analyze-only builds the pencil and runs the MUMPS analysis with AMD and with SCOTCH at the
// configured shift, printing the factor estimates (INFOG(16/17) in-core MB etc.) and the RSS
// after each analysis: use it first to check that a full run fits the memory limit.
// Run through tools/run/guard.py.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

namespace fs = std::filesystem;
using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double mb(double bytes) { return bytes / 1e6; }

std::int64_t rss_bytes() {
  // /proc/self/statm: size resident ... in pages (cheap to read).
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return -1;
  long long size = 0, resident = 0;
  const int got = std::fscanf(f, "%lld %lld", &size, &resident);
  std::fclose(f);
  return got == 2 ? resident * static_cast<std::int64_t>(::sysconf(_SC_PAGESIZE)) : -1;
}

std::int64_t hwm_bytes() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind("VmHWM:", 0) == 0) return std::atoll(line.c_str() + 6) * 1024;
  return -1;
}

// Samples VmRSS on a background thread; marks label instants on the same clock.
class Profiler {
 public:
  explicit Profiler(int period_ms) : t0_(Clock::now()), period_(period_ms) {
    samples_.reserve(1 << 18);
    thread_ = std::thread([this] {
      while (!stop_) {
        const std::int64_t r = rss_bytes();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          samples_.emplace_back(now(), r);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(period_));
      }
    });
  }
  ~Profiler() { stop(); }
  void stop() {
    if (thread_.joinable()) {
      stop_ = true;
      thread_.join();
    }
  }
  double now() const { return std::chrono::duration<double>(Clock::now() - t0_).count(); }
  void mark(const std::string& name) {
    const std::int64_t r = rss_bytes();
    std::lock_guard<std::mutex> lock(mutex_);
    marks_.emplace_back(now(), name);
    samples_.emplace_back(marks_.back().first, r);
  }
  void add_marks(std::vector<std::pair<double, std::string>> m) {
    std::lock_guard<std::mutex> lock(mutex_);
    marks_.insert(marks_.end(), m.begin(), m.end());
  }
  // One JSON line per phase between consecutive marks.
  void report(std::ostream& os) {
    stop();
    std::sort(marks_.begin(), marks_.end());
    std::sort(samples_.begin(), samples_.end());
    const auto rss_at = [&](double t) {
      std::int64_t r = -1;
      for (const auto& s : samples_) {
        if (s.first > t) break;
        r = s.second;
      }
      return r;
    };
    for (std::size_t k = 0; k + 1 < marks_.size(); ++k) {
      const double a = marks_[k].first, b = marks_[k + 1].first;
      std::int64_t peak = rss_at(a);
      for (const auto& s : samples_)
        if (s.first >= a && s.first <= b) peak = std::max(peak, s.second);
      json j = {{"phase", marks_[k].second + " -> " + marks_[k + 1].second},
                {"t_start", a},
                {"seconds", b - a},
                {"rss_start_mb", mb(static_cast<double>(rss_at(a)))},
                {"rss_end_mb", mb(static_cast<double>(rss_at(b)))},
                {"peak_mb", mb(static_cast<double>(peak))}};
      os << j.dump() << '\n';
    }
  }

 private:
  Clock::time_point t0_;
  int period_;
  std::atomic<bool> stop_{false};
  std::mutex mutex_;
  std::vector<std::pair<double, std::int64_t>> samples_;
  std::vector<std::pair<double, std::string>> marks_;
  std::thread thread_;
};

template <class T>
double csr_bytes(const CsrMatrix<T>& a) {
  return static_cast<double>(a.rowptr.capacity() * 8 + a.colind.capacity() * 4 + a.values.capacity() * sizeof(T));
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p);
  if (!in) HS_THROW(Io, "cannot read " << p.string());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

// Krylov basis size the eigensolvers choose by default (eigen module rules).
index_t default_ncv(index_t n, index_t k) { return std::min(n, std::max(2 * k + 1, k + 16)); }

template <class T>
void analyze_only(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const hs::SolverConfig& cfg) {
  const double sigma = cfg.target.sigma;
  std::int64_t r0 = rss_bytes();
  const hs::ShiftedPencil<T> pencil(a, b);
  std::cout << json{{"step", "pencil"}, {"pattern_nnz", pencil.nnz()},
                    {"rss_growth_mb", mb(static_cast<double>(rss_bytes() - r0))}}
                   .dump()
            << std::endl;
  for (const int variant : {0, 3, 4}) {  // AMD, SCOTCH, SCOTCH with BLR estimates
    hs::FactorOptions o;
    o.ordering = variant == 0 ? 0 : 3;
    o.blr_estimates = variant == 4;
    o.threads = cfg.threads;
    r0 = rss_bytes();
    const auto t0 = Clock::now();
    auto fac = hs::make_mumps_factorization<T>(o);
    fac->analyze(pencil, 1.0, -sigma);
    const hs::FactorEstimate e = fac->estimate();
    std::cout << json{{"step", "analysis"},
                      {"ordering", variant == 0 ? "amd" : (variant == 3 ? "scotch" : "scotch+blr_estimates")},
                      {"blr_incore_mb", e.blr_incore_mb},
                      {"blr_ooc_mb", e.blr_ooc_mb},
                      {"seconds", std::chrono::duration<double>(Clock::now() - t0).count()},
                      {"factor_entries", e.factor_entries},
                      {"factor_mb_8B_per_entry", mb(8.0 * static_cast<double>(e.factor_entries))},
                      {"incore_mb", e.incore_mb},
                      {"ooc_mb", e.ooc_mb},
                      {"gflops", e.flops / 1e9},
                      {"analysis_rss_growth_mb", mb(static_cast<double>(rss_bytes() - r0))}}
                     .dump()
              << std::endl;
  }
}

template <class T>
int run(const CsrMatrix<T>& a, const CsrMatrix<T>* b, const hs::SolverConfig& cfg, Profiler& prof,
        const fs::path& out, const fs::path& log_path, double load_and_promote_mb) {
  // Analytic components (bytes).
  const index_t n = a.nrows;
  double pattern_nnz = 0;
  {
    const hs::ShiftedPencil<T> pencil(a, b);  // only to count the pattern; freed at once
    pattern_nnz = static_cast<double>(pencil.nnz());
  }
  prof.mark("pattern counted");
  const index_t k = cfg.target.k;
  const index_t ncv = cfg.ncv > 0 ? cfg.ncv : default_ncv(n, k);
  json comp;
  comp["components"] = true;
  comp["n"] = n;
  comp["input_csr_mb"] = mb(csr_bytes(a) + (b ? csr_bytes(*b) : 0.0));
  comp["load_and_promote_rss_mb"] = load_and_promote_mb;
  comp["pencil_mb"] = mb(8.0 * static_cast<double>(n + 1) + 4.0 * pattern_nnz + static_cast<double>(n));
  comp["pattern_nnz"] = pattern_nnz;
  // mumps::Input: real T: int irn, jcn + double value per pattern entry; complex T: the 2n real
  // embedding (at most 4 entries per pattern entry, 16 B each) + a complex scratch copy of the
  // pencil values + 8 B source indices (upper bound).
  comp["mumps_input_mb"] = hs::is_complex_v<T> ? mb(pattern_nnz * (4 * 16.0 + 16.0 + 2 * 8.0)) : mb(16.0 * pattern_nnz);
  comp["ncv_default"] = ncv;
  comp["krylov_basis_mb"] = mb(static_cast<double>(n) * static_cast<double>(ncv + 1) * sizeof(T));
  comp["eigenvectors_out_mb"] = mb(static_cast<double>(n) * static_cast<double>(k) * sizeof(T));

  std::optional<hs::Logger> log;
  log.emplace(0, log_path);
  const double t_log0 = prof.now();
  prof.mark("solve begin");
  hs::Solution<T> sol = hs::solve(a, b, cfg, &*log);
  prof.mark("solve end");
  log.reset();
  // Driver events as phase marks (the logger's clock starts at t_log0).
  std::vector<std::pair<double, std::string>> events;
  {
    std::ifstream in(log_path);
    std::string line;
    while (std::getline(in, line)) {
      const json j = json::parse(line, nullptr, false);
      if (j.is_discarded() || !j.contains("t") || !j.contains("event")) continue;
      if (j.contains("message")) continue;  // warn/info/debug lines, not phase events
      events.emplace_back(t_log0 + j["t"].get<double>(), "event " + j["event"].get<std::string>());
    }
  }
  prof.add_marks(events);

  fs::create_directories(out);
  hs::write_npy_vector(out / "eigenvalues.npy", sol.eigenvalues);
  hs::write_npy_matrix(out / "eigenvectors.npy", sol.eigenvectors);
  prof.mark("outputs written");

  const hs::SolveReport& r = sol.report;
  comp["factor_entries"] = r.factor_stats.factor_entries;
  comp["factor_values_mb"] = mb(8.0 * static_cast<double>(r.factor_stats.factor_entries));  // real (embedding)
  comp["mumps_mb_used_infog21"] = r.factor_stats.factor_mb_used;
  comp["mumps_mb_allocated_infog18"] = r.factor_stats.factor_mb_allocated;
  comp["factorizations"] = r.factor_stats.factorizations;
  comp["analyses"] = r.factor_stats.analyses;
  comp["ordering"] = r.factor_options.ordering;
  comp["solution_vectors_mb"] =
      mb(static_cast<double>(sol.eigenvectors.rows() * sol.eigenvectors.cols()) * sizeof(T) +
         static_cast<double>(sol.complex_eigenvectors.rows() * sol.complex_eigenvectors.cols()) * sizeof(cplx));
  comp["report_peak_rss_mb"] = mb(static_cast<double>(r.peak_rss_bytes));
  comp["time_total_s"] = r.time_total_s;
  comp["success"] = r.success;
  comp["pairs"] = r.pairs.size();
  std::cout << comp.dump() << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string a_path, b_path, config_path;
    fs::path out = fs::temp_directory_path() / ("bench_memio_e2e." + std::to_string(::getpid()));
    int sample_ms = 2;
    bool analyze = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto next = [&]() -> std::string {
        if (i + 1 >= argc) HS_THROW(InvalidInput, arg << " needs a value");
        return argv[++i];
      };
      if (arg == "--A") {
        a_path = next();
      } else if (arg == "--B") {
        b_path = next();
      } else if (arg == "--config") {
        config_path = next();
      } else if (arg == "--out") {
        out = next();
      } else if (arg == "--sample-ms") {
        sample_ms = std::max(1, std::stoi(next()));
      } else if (arg == "--analyze-only") {
        analyze = true;
      } else {
        HS_THROW(InvalidInput, "unknown argument " << arg);
      }
    }
    if (a_path.empty()) HS_THROW(InvalidInput, "--A is required");
    hs::SolverConfig cfg;
    if (!config_path.empty()) cfg = hs::parse_config(read_text(config_path));

    Profiler prof(sample_ms);
    prof.mark("start");
    hs::LoadOptions lo;
    lo.hermitian_tol = cfg.hermitian_tol;
    hs::LoadedMatrix la = hs::load_matrix(a_path, lo);
    prof.mark("A loaded");
    std::optional<hs::LoadedMatrix> lb;
    if (!b_path.empty()) {
      lb = hs::load_matrix(b_path, lo);
      prof.mark("B loaded");
    }
    const fs::path log_path = out.string() + ".log.jsonl";
    int rc = 0;
    // As apps/hsolve_cli.cpp: a real matrix in a complex problem is copied by to_complex while
    // its LoadedMatrix stays alive.
    if (la.is_complex() || (lb && lb->is_complex())) {
      CsrMatrix<cplx> a = la.is_complex() ? std::get<1>(std::move(la.matrix)) : hs::to_complex(std::get<0>(la.matrix));
      CsrMatrix<cplx> b;
      if (lb) b = lb->is_complex() ? std::get<1>(std::move(lb->matrix)) : hs::to_complex(std::get<0>(lb->matrix));
      prof.mark("promoted");
      const double loaded = mb(static_cast<double>(rss_bytes()));
      if (analyze) {
        analyze_only<cplx>(a, lb ? &b : nullptr, cfg);
        return 0;
      }
      rc = run<cplx>(a, lb ? &b : nullptr, cfg, prof, out, log_path, loaded);
    } else {
      const auto& a = std::get<0>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<0>(lb->matrix) : nullptr;
      const double loaded = mb(static_cast<double>(rss_bytes()));
      if (analyze) {
        analyze_only<double>(a, b, cfg);
        return 0;
      }
      rc = run<double>(a, b, cfg, prof, out, log_path, loaded);
    }
    prof.report(std::cout);
    std::cout << json{{"final_vmhwm_mb", mb(static_cast<double>(hwm_bytes()))},
                      {"peak_rss_mb_getrusage", mb(static_cast<double>(hs::peak_rss_bytes()))}}
                     .dump()
              << std::endl;
    fs::remove_all(out);
    fs::remove(log_path);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_memio_e2e: %s\n", e.what());
    return 2;
  }
}
