// Numeric LDL^T throughput of the factorizations a nearest solve makes (P9 "factspeed"): the main
// factorization at sigma (factors kept) and the certificate's two inertia-only counts at the window
// ends a, b on the main analysis' pivot order - exactly as the driver makes them (main object:
// FactorOptions of the driver; counts: InertiaCounter's instances, i.e. ICNTL(31) = 1, PERM_IN =
// the main SYM_PERM, the main object's options). Per factorization it reports wall time, CPU time
// (user / system), page faults and MUMPS' statistics (RINFOG(1-3), INFOG(9/11-14/18/21/25/28/29),
// the swaps INFOG(48), the 2x2 pivots KEEP(229/230)) and the achieved rate RINFOG(3) / wall.
//
//   bench_factspeed_factor --case DIR [--sigma s] [--lower a --upper b] [--jobs main,a,b]
//       [--reps r] [--ordering o] [--perm-in FILE] [--perm-out FILE] [--threads t]
//       [--relax pct] [--cap-mb mb] [--release 0|1] [--deterministic 0|1]
//       [--icntl i=v,...] [--keep i=v,...] [--cntl i=v,...] [--main-cntl i=v,...]
//       [--mumps-stats 0|1] [--thp-disable 0|1] [--sample-us us --sample-out FILE]
//
// DIR holds A.npz (B.npz) and config.json (sigma). a, b: a hsolve report's certificate.lower /
// upper (the jobs a, b need them). o: ICNTL(7) (default 3 = SCOTCH, the driver's choice at 1e6) or
// 10 (hsolve's nested dissection). SCOTCH is not reproducible from run to run (factor.hpp), so A/B
// comparisons fix the pivot order: --perm-out writes the main analysis' SYM_PERM (int32, raw) and
// --perm-in analyses with it (ICNTL(7) = 1), identically in every run. --jobs: the sequence of
// factorizations per repetition (main = factorize K(sigma) keeping the factors, released after the
// job unless --release 0, as the driver releases them before its counts; a / b = count at a / b on
// the counter instance, created and analysed once). --cap-mb: the working-memory cap of every
// instance (the driver's max_memory_mb; default 25000, the 32 GiB slot's in-core plan);
// --relax: ICNTL(14) (driver 35). --icntl/--keep/--cntl: extra MUMPS controls for every instance
// (MumpsControlOverrides: set after hsolve's own before each analysis and factorization; KEEP is
// MUMPS-internal); --main-cntl only for the main object (e.g. 1=0.01 for the pivot threshold CNTL(1):
// never for counts, their inertia must be that of K). --mumps-stats 1: MUMPS' global statistics
// (ICNTL(3) = 6, ICNTL(4) = 2: elapsed times, the time and flops under the L0 layer, delayed and
// 2x2 pivots) on stdout between the JSON lines. --thp-disable 1: prctl(PR_SET_THP_DISABLE) before
// anything is allocated (transparent huge pages off for this process). --sample-us: a SIGPROF
// sampler (setitimer ITIMER_PROF, every `us` microseconds of process CPU time, i.e. each busy thread
// is sampled about every `us` * threads microseconds of wall time; the interrupted instruction
// pointer of whichever thread the kernel signals) during the factorizations; --sample-out FILE gets
// one line per (job, object, offset) with its sample count ("job<TAB>object<TAB>0xbase<TAB>
// 0xoffset<TAB>count"), symbolised by tools/bench/factspeed_symbolize.py (nm), and the JSON line "samples"
// a per-job summary by object file. CPU time includes system time (page faults count at the faulting
// instruction).
// Output: one JSON object per line on stdout. Run through tools/run/guard.py (measurement slot for
// timings: --exclusive).
//
// P9 commands (gpu01 measurement slot, results/p9_factspeed.md has the windows and every command):
//   G="python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 --"
//   $G build/release/bench_factspeed_factor --case $D/syn-mem27-100 --lower -0.0005369991406518608
//      --upper 0.0005978355069250973 --perm-out $S/syn100.perm --jobs main,a,b --mumps-stats 1
//   (then --perm-in $S/syn100.perm for every comparison; one command line each)
#include <omp.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <ucontext.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "factor/mumps_factorization.hpp"
#include "hsolve/error.hpp"
#include "hsolve/factor.hpp"
#include "hsolve/io.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"

#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41
#endif

namespace {

using hs::index_t;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void emit(const json& j) { std::cout << j.dump() << std::endl; }

struct Args {
  fs::path case_dir;
  double sigma = kNaN, lower = kNaN, upper = kNaN;
  std::vector<std::string> jobs = {"main", "a", "b"};
  int reps = 1;
  int ordering = 3;
  std::string perm_in, perm_out;
  int threads = 0;
  int relax = 35;
  double cap_mb = 25000.0;
  bool release = true;
  bool deterministic = false;
  hs::mumps::MumpsControlOverrides all, main_only;
  bool mumps_stats = false;
  bool thp_disable = false;
  int sample_us = 0;
  std::string sample_out;
};

// ---------------------------------------------------------------- SIGPROF sampler
// The handler only stores the interrupted RIP (tagged with the current job in the top byte; user
// addresses are below 2^47) into a preallocated array with one atomic increment: async-signal-safe.
namespace sampler {
constexpr std::size_t kCapacity = std::size_t{1} << 24;  // 16M samples, 128 MB
std::uintptr_t* g_buf = nullptr;
std::atomic<std::size_t> g_count{0};
std::atomic<int> g_tag{0};

void handler(int, siginfo_t*, void* ctx) {
  const auto* uc = static_cast<const ucontext_t*>(ctx);
  const auto pc = static_cast<std::uintptr_t>(uc->uc_mcontext.gregs[REG_RIP]);
  const std::size_t i = g_count.fetch_add(1, std::memory_order_relaxed);
  if (i < kCapacity)
    g_buf[i] = (pc & ((std::uintptr_t{1} << 56) - 1)) |
               (static_cast<std::uintptr_t>(g_tag.load(std::memory_order_relaxed)) << 56);
}

void install(int us) {
  g_buf = new std::uintptr_t[kCapacity];
  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGPROF, &sa, nullptr) != 0) HS_THROW(Internal, "sigaction(SIGPROF) failed");
  (void)us;
}

void set_timer(int us) {
  itimerval it;
  it.it_interval.tv_sec = us / 1000000;
  it.it_interval.tv_usec = us % 1000000;
  it.it_value = it.it_interval;
  if (setitimer(ITIMER_PROF, &it, nullptr) != 0) HS_THROW(Internal, "setitimer failed");
}

void stop() {
  itimerval it;
  std::memset(&it, 0, sizeof(it));
  setitimer(ITIMER_PROF, &it, nullptr);
}

// Executable mappings of the process (/proc/self/maps): [start, end) -> (object, load base =
// start - file offset). No dladdr: it needs libdl, which the bench targets do not link.
struct Mapping {
  std::uintptr_t start, end, base;
  std::string object;
};
std::vector<Mapping> read_maps() {
  std::vector<Mapping> out;
  std::ifstream f("/proc/self/maps");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ls(line);
    std::string range, perms, off, dev, inode, path;
    ls >> range >> perms >> off >> dev >> inode >> path;
    if (perms.size() < 3 || perms[2] != 'x' || path.empty() || path[0] != '/') continue;
    const auto dash = range.find('-');
    Mapping m;
    m.start = std::stoul(range.substr(0, dash), nullptr, 16);
    m.end = std::stoul(range.substr(dash + 1), nullptr, 16);
    m.base = m.start - std::stoul(off, nullptr, 16);
    m.object = path;
    out.push_back(m);
  }
  std::sort(out.begin(), out.end(), [](const Mapping& x, const Mapping& y) { return x.start < y.start; });
  return out;
}

// Writes "job<TAB>object<TAB>0xbase<TAB>0xoffset<TAB>count" lines (offset = pc - load base) and
// returns a per-job summary by object.
json dump(const std::vector<std::string>& tags, const std::string& path) {
  const std::size_t n = std::min(g_count.load(), kCapacity);
  const std::vector<Mapping> maps = read_maps();
  std::map<std::tuple<int, std::string, std::uintptr_t, std::uintptr_t>, std::size_t> by_offset;
  std::map<std::pair<int, std::string>, std::size_t> by_object;
  for (std::size_t i = 0; i < n; ++i) {
    const int tag = static_cast<int>(g_buf[i] >> 56);
    const std::uintptr_t pc = g_buf[i] & ((std::uintptr_t{1} << 56) - 1);
    auto it = std::upper_bound(maps.begin(), maps.end(), pc,
                               [](std::uintptr_t v, const Mapping& m) { return v < m.start; });
    std::string obj = "?";
    std::uintptr_t base = 0;
    if (it != maps.begin() && pc < std::prev(it)->end) {
      obj = std::prev(it)->object;
      base = std::prev(it)->base;
    }
    ++by_offset[{tag, obj, base, pc - base}];
    ++by_object[{tag, fs::path(obj).filename().string()}];
  }
  if (!path.empty()) {
    std::ofstream f(path);
    for (const auto& [k, c] : by_offset) {
      const auto& [tag, obj, base, off] = k;
      char hex[64];
      std::snprintf(hex, sizeof(hex), "0x%lx\t0x%lx", static_cast<unsigned long>(base),
                    static_cast<unsigned long>(off));
      f << (tag < static_cast<int>(tags.size()) ? tags[tag] : "?") << '\t' << obj << '\t' << hex
        << '\t' << c << '\n';
    }
  }
  json j = json::object();
  for (const auto& [k, c] : by_object) {
    const std::string job = k.first < static_cast<int>(tags.size()) ? tags[k.first] : "?";
    j[job][k.second] = c;
  }
  return {{"samples", n}, {"dropped", g_count.load() > kCapacity ? g_count.load() - kCapacity : 0},
          {"by_object", j}};
}
}  // namespace sampler

// ---------------------------------------------------------------- one factorization

struct Usage {
  double user = 0, sys = 0;
  long minflt = 0, majflt = 0;
  static Usage now() {
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    Usage u;
    u.user = r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec;
    u.sys = r.ru_stime.tv_sec + 1e-6 * r.ru_stime.tv_usec;
    u.minflt = r.ru_minflt;
    u.majflt = r.ru_majflt;
    return u;
  }
};

template <class T>
using Fac = hs::mumps::MumpsFactorization<T, double>;

template <class T>
json factor_record(const Fac<T>& f) {
  const auto& m = f.instance();
  const hs::FactorStats& s = f.stats();
  return {{"rinfog1", m.rinfog(1)},
          {"rinfog2", m.rinfog(2)},
          {"rinfog3", m.rinfog(3)},
          {"infog9", hs::mumps::decode_count(m.infog(9))},
          {"infog11", m.infog(11)},
          {"infog12", m.infog(12)},
          {"infog13", m.infog(13)},
          {"infog14", m.infog(14)},
          {"infog18", m.infog(18)},
          {"infog21", m.infog(21)},
          {"infog25", m.infog(25)},
          {"infog28", m.infog(28)},
          {"infog29", hs::mumps::decode_count(m.infog(29))},
          {"infog48", m.infog(48)},
          {"infog49", m.infog(49)},
          {"keep229", m.keep(229)},
          {"keep230", m.keep(230)},
          {"icntl23_mb", s.workspace_reserved_mb},
          {"workspace_retries", s.workspace_retries}};
}

std::vector<int> read_perm(const std::string& path, index_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) HS_THROW(Io, "cannot open " << path);
  std::vector<int> p(static_cast<std::size_t>(n));
  f.read(reinterpret_cast<char*>(p.data()), static_cast<std::streamsize>(p.size() * sizeof(int)));
  if (f.gcount() != static_cast<std::streamsize>(p.size() * sizeof(int)))
    HS_THROW(Io, path << " holds fewer than " << n << " int32 entries");
  return p;
}

template <class T>
int run(const Args& args, const hs::CsrMatrix<T>& a, const hs::CsrMatrix<T>* b) {
  const hs::ShiftedPencil<T> pencil(a, b);
  const int threads = args.threads > 0 ? args.threads : omp_get_max_threads();
  hs::FactorOptions fo;  // the driver's (factor_options_from) for an in-core double solve
  fo.ordering = args.ordering;
  fo.threads = threads;
  fo.deterministic = args.deterministic;
  fo.mem_relax_pct = args.relax;
  fo.max_memory_mb = args.cap_mb;
  fo.ooc_on_memory_error = false;
  fo.verbosity = args.mumps_stats ? 1 : 0;
  hs::mumps::MumpsControlOverrides ov = args.all;
  if (args.mumps_stats) ov.icntl.push_back({4, 2});  // global statistics only (ICNTL(2) stays off)
  hs::mumps::MumpsControlOverrides main_ov = ov;
  main_ov.cntl.insert(main_ov.cntl.end(), args.main_only.cntl.begin(), args.main_only.cntl.end());
  main_ov.icntl.insert(main_ov.icntl.end(), args.main_only.icntl.begin(), args.main_only.icntl.end());
  main_ov.keep.insert(main_ov.keep.end(), args.main_only.keep.begin(), args.main_only.keep.end());

  Fac<T> main(fo);
  main.set_control_overrides(main_ov);
  const index_t mumps_n = pencil.order() * (hs::is_complex_v<T> ? 2 : 1);
  if (!args.perm_in.empty()) main.set_pivot_order(read_perm(args.perm_in, mumps_n));
  hs::Timer ta;
  main.analyze(pencil, 1.0, -args.sigma);
  const double analyze_s = ta.seconds();
  if (!args.perm_out.empty()) {
    std::ofstream f(args.perm_out, std::ios::binary);
    const auto& p = main.pivot_order();
    f.write(reinterpret_cast<const char*>(p.data()), static_cast<std::streamsize>(p.size() * sizeof(int)));
  }
  const hs::FactorEstimate est = main.estimate();
  emit({{"event", "analysis"}, {"case", args.case_dir.filename().string()}, {"n", pencil.order()},
        {"complex", hs::is_complex_v<T>}, {"threads", threads}, {"ordering", args.ordering},
        {"perm_in", !args.perm_in.empty()}, {"analyze_s", analyze_s},
        {"ordering_s", main.stats().ordering_s}, {"est_entries", est.factor_entries},
        {"est_flops", est.flops}, {"est_incore_mb", est.incore_mb},
        {"est_max_front", main.instance().infog(5)}, {"nodes", main.instance().infog(6)},
        {"keep", {{"4", main.instance().keep(4)}, {"7", main.instance().keep(7)},
                  {"8", main.instance().keep(8)}, {"227", main.instance().keep(227)},
                  {"420", main.instance().keep(420)}, {"468", main.instance().keep(468)},
                  {"57", main.instance().keep(57)}, {"58", main.instance().keep(58)},
                  {"218", main.instance().keep(218)}, {"424", main.instance().keep(424)}}}});

  // The counter instance exactly as InertiaCounter::Impl::make_instance builds it.
  std::unique_ptr<Fac<T>> counter;
  auto counter_instance = [&]() -> Fac<T>& {
    if (!counter) {
      hs::FactorOptions co = fo;
      co.out_of_core = false;
      counter = std::make_unique<Fac<T>>(co);
      counter->set_control_overrides(ov);
      counter->set_discard_factors(true);
      counter->set_pivot_order(main.pivot_order());
      hs::Timer tc;
      counter->analyze(pencil, 1.0, 0.0);
      emit({{"event", "counter_analysis"}, {"analyze_s", tc.seconds()},
            {"est_incore_mb", counter->estimate().incore_mb}, {"est_flops", counter->estimate().flops}});
    }
    return *counter;
  };

  std::vector<std::string> tags = {"none"};
  for (const std::string& j : args.jobs) tags.push_back(j);
  if (args.sample_us > 0) sampler::install(args.sample_us);
  for (int r = 0; r < args.reps; ++r) {
    for (std::size_t ji = 0; ji < args.jobs.size(); ++ji) {
      const std::string& job = args.jobs[ji];
      Fac<T>* f = nullptr;
      double shift = args.sigma;
      if (job == "main") {
        f = &main;
      } else if (job == "a" || job == "b") {
        shift = job == "a" ? args.lower : args.upper;
        if (std::isnan(shift)) HS_THROW(InvalidInput, "job " << job << " needs --lower/--upper");
        f = &counter_instance();
      } else {
        HS_THROW(InvalidInput, "unknown job " << job);
      }
      std::cout.flush();
      const double before = f->stats().factorize_s;
      const Usage u0 = Usage::now();
      if (args.sample_us > 0) {
        sampler::g_tag.store(static_cast<int>(ji + 1));
        sampler::set_timer(args.sample_us);
      }
      hs::Timer tw;
      const bool ok = f->factorize(1.0, -shift);
      const double wall = tw.seconds();
      if (args.sample_us > 0) {
        sampler::stop();
        sampler::g_tag.store(0);
      }
      const Usage u1 = Usage::now();
      json rec = factor_record(*f);
      const hs::Inertia in = f->inertia();
      rec.update({{"event", "factorization"}, {"job", job}, {"rep", r}, {"shift", shift},
                  {"ok", ok}, {"wall_s", wall}, {"factorize_s", f->stats().factorize_s - before},
                  {"user_s", u1.user - u0.user}, {"sys_s", u1.sys - u0.sys},
                  {"minflt", u1.minflt - u0.minflt}, {"majflt", u1.majflt - u0.majflt},
                  {"gflops", f->instance().rinfog(3) / wall * 1e-9}, {"neg", in.neg},
                  {"zero", in.zero}, {"valid", in.valid}, {"peak_rss_mb", hs::peak_rss_bytes() / 1e6},
                  {"rss_mb", hs::current_rss_bytes() / 1e6}});
      emit(rec);
      if (job == "main" && args.release) main.release_factors();
    }
  }
  if (args.sample_us > 0) {
    json s = sampler::dump(tags, args.sample_out);
    s["event"] = "samples";
    s["sample_us"] = args.sample_us;
    emit(s);
  }
  return 0;
}

// "i=v,i=v" -> pairs
template <class V>
std::vector<std::pair<int, V>> parse_pairs(const std::string& s) {
  std::vector<std::pair<int, V>> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto eq = item.find('=');
    if (eq == std::string::npos) throw std::runtime_error("expected i=v, got " + item);
    const int i = std::stoi(item.substr(0, eq));
    if constexpr (std::is_same_v<V, int>) out.push_back({i, std::stoi(item.substr(eq + 1))});
    else out.push_back({i, std::stod(item.substr(eq + 1))});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](int& j) -> std::string {
      if (j + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++j];
    };
    try {
      if (k == "--case") args.case_dir = need(i);
      else if (k == "--sigma") args.sigma = std::stod(need(i));
      else if (k == "--lower") args.lower = std::stod(need(i));
      else if (k == "--upper") args.upper = std::stod(need(i));
      else if (k == "--jobs") {
        args.jobs.clear();
        std::stringstream ss(need(i));
        std::string e;
        while (std::getline(ss, e, ',')) args.jobs.push_back(e);
      } else if (k == "--reps") args.reps = std::stoi(need(i));
      else if (k == "--ordering") args.ordering = std::stoi(need(i));
      else if (k == "--perm-in") args.perm_in = need(i);
      else if (k == "--perm-out") args.perm_out = need(i);
      else if (k == "--threads") args.threads = std::stoi(need(i));
      else if (k == "--relax") args.relax = std::stoi(need(i));
      else if (k == "--cap-mb") args.cap_mb = std::stod(need(i));
      else if (k == "--release") args.release = std::stoi(need(i)) != 0;
      else if (k == "--deterministic") args.deterministic = std::stoi(need(i)) != 0;
      else if (k == "--icntl") args.all.icntl = parse_pairs<int>(need(i));
      else if (k == "--keep") args.all.keep = parse_pairs<int>(need(i));
      else if (k == "--cntl") args.all.cntl = parse_pairs<double>(need(i));
      else if (k == "--main-cntl") args.main_only.cntl = parse_pairs<double>(need(i));
      else if (k == "--main-icntl") args.main_only.icntl = parse_pairs<int>(need(i));
      else if (k == "--main-keep") args.main_only.keep = parse_pairs<int>(need(i));
      else if (k == "--mumps-stats") args.mumps_stats = std::stoi(need(i)) != 0;
      else if (k == "--thp-disable") args.thp_disable = std::stoi(need(i)) != 0;
      else if (k == "--sample-us") args.sample_us = std::stoi(need(i));
      else if (k == "--sample-out") args.sample_out = need(i);
      else {
        std::cerr << "unknown argument " << k << "\n";
        return 2;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return 2;
    }
  }
  if (args.case_dir.empty()) {
    std::cerr << "usage: bench_factspeed_factor --case DIR [--lower a --upper b] [--jobs main,a,b] [...]\n";
    return 2;
  }
  if (args.thp_disable) {
    const int rc = prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    emit({{"event", "thp_disable"}, {"rc", rc}, {"errno", rc == 0 ? 0 : errno}});
  }
  try {
    if (std::isnan(args.sigma)) {
      std::ifstream f(args.case_dir / "config.json");
      std::stringstream ss;
      ss << f.rdbuf();
      args.sigma = hs::parse_config(ss.str()).target.sigma;
    }
    auto la = hs::load_matrix(args.case_dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (fs::exists(args.case_dir / "B.npz")) lb = hs::load_matrix(args.case_dir / "B.npz");
    const bool cplx = la.is_complex() || (lb && lb->is_complex());
    if (!cplx) {
      const auto& a = std::get<hs::CsrMatrix<double>>(la.matrix);
      const hs::CsrMatrix<double>* b = lb ? &std::get<hs::CsrMatrix<double>>(lb->matrix) : nullptr;
      return run<double>(args, a, b);
    }
    auto to_c = [](hs::LoadedMatrix& m) {
      return m.is_complex() ? std::get<hs::CsrMatrix<hs::cplx>>(m.matrix)
                            : hs::to_complex(std::get<hs::CsrMatrix<double>>(m.matrix));
    };
    const auto a = to_c(la);
    std::optional<hs::CsrMatrix<hs::cplx>> b;
    if (lb) b = to_c(*lb);
    return run<hs::cplx>(args, a, b ? &*b : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
