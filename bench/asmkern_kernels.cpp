// Kernel measurements of the BLAS hook (module blas, P5 "asmkern"; docs/BLAS.md).
//
//   bench_asmkern_kernels peak   [--n 4096] [--reps 5]
//       dgemm GFlop/s of the real BLAS (C = A B, square n) at OMP_NUM_THREADS threads.
//   bench_asmkern_kernels stream [--mb 1024] [--reps 10]
//       memory bandwidth on the cores given (OpenMP, static schedule, first touch by the same
//       threads): read (sum of one array), triad (a = b + s c; 3 arrays counted, no write
//       allocate) and update (y += s x; 3 transfers per element counted), GB/s, best of reps.
//   bench_asmkern_kernels shapes [--shapes SPEC] [--routes LIST] [--reps 7] [--pool-mb 1024]
//       [--inpar 0|1]
//       per-call time of BLAS calls of the given shapes, the real routine against hook routes,
//       interleaved (variant order rotated per repetition), median over repetitions; each
//       repetition cycles through a pool of distinct operands (pool-mb in total) so that the
//       matrix comes from memory as in MUMPS' solves (cold). SPEC: ';'-separated
//       op:flags:m:n:k[:lda] with op gemm (flags NN/TN), trsm (flags side uplo trans [diag], e.g.
//       LLN or LUTU, k unused),
//       gemv (flags N/T, n unused: m x k matrix). LIST: ','-separated route specifications
//       (HS_BLAS_ROUTE syntax) with '+' instead of ',' inside one, e.g. "gemv1,narrow+par".
//       Prints per shape: real time, each route's time, speedup, GB/s of the matrix, and the max
//       relative difference to the real routine's result. --inpar 1: every thread makes its own
//       calls (own operand pool) at the same time inside one parallel region, as MUMPS' L0
//       subtrees do (concurrent callers, the BLAS single-threaded); times are per thread.
// Output: markdown tables on stdout. Run through tools/run/guard.py (cores, OMP_NUM_THREADS).
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "blas/blas_hook.hpp"
#include "linalg/blas_lapack.hpp"

namespace {

// The hook finds the real BLAS with dlsym(RTLD_NEXT): this reference keeps the BLAS library in
// the executable's dependencies (the toolchain links with --as-needed).
[[maybe_unused, gnu::used]] void (*const kLinkBlas)(const char*, const char*, const hs::lapack::fint*,
                                         const hs::lapack::fint*, const hs::lapack::fint*,
                                         const double*, const double*, const hs::lapack::fint*,
                                         const double*, const hs::lapack::fint*, const double*,
                                         double*, const hs::lapack::fint*, hs::lapack::fstrlen,
                                         hs::lapack::fstrlen) = &hs::lapack::dgemm_;

using hs::blas::fint;

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::map<std::string, std::string> parse_opts(int argc, char** argv, int first) {
  std::map<std::string, std::string> o;
  for (int i = first; i + 1 < argc; i += 2) o[argv[i]] = argv[i + 1];
  return o;
}

double* alloc(std::size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(double) + 64) != 0) std::abort();
  return static_cast<double*>(p);
}

// Parallel first touch with random values in [-1, 1) (deterministic per index).
void fill(double* x, std::size_t n, unsigned seed) {
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull ^ (static_cast<std::uint64_t>(seed) << 32);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    x[i] = static_cast<double>(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
  }
}

int run_peak(const std::map<std::string, std::string>& o) {
  const fint n = o.count("--n") ? std::atoi(o.at("--n").c_str()) : 4096;
  const int reps = o.count("--reps") ? std::atoi(o.at("--reps").c_str()) : 5;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  double* a = alloc(nn);
  double* b = alloc(nn);
  double* c = alloc(nn);
  fill(a, nn, 1);
  fill(b, nn, 2);
  fill(c, nn, 3);
  const auto& rb = hs::blas::real_blas();
  const double one = 1.0, zero = 0.0;
  std::vector<double> t;
  for (int r = 0; r < reps + 1; ++r) {
    const double t0 = now();
    rb.dgemm("N", "N", &n, &n, &n, &one, a, &n, b, &n, &zero, c, &n, 1, 1);
    if (r > 0) t.push_back(now() - t0);
  }
  const double gf = 2.0 * n * static_cast<double>(n) * n / median(t) * 1e-9;
  std::printf("| peak dgemm | n | threads | median s | GFlop/s | GFlop/s per thread |\n"
              "|---|---|---|---|---|---|\n| %s | %d | %d | %.4f | %.1f | %.1f |\n",
              rb.library.c_str(), n, omp_get_max_threads(), median(t), gf,
              gf / omp_get_max_threads());
  return 0;
}

int run_stream(const std::map<std::string, std::string>& o) {
  const double mb = o.count("--mb") ? std::atof(o.at("--mb").c_str()) : 1024.0;
  const int reps = o.count("--reps") ? std::atoi(o.at("--reps").c_str()) : 10;
  const std::size_t n = static_cast<std::size_t>(mb * 1e6 / 8.0);
  double* x = alloc(n);
  double* y = alloc(n);
  double* z = alloc(n);
  fill(x, n, 1);
  fill(y, n, 2);
  fill(z, n, 3);
  double best_read = 0, best_triad = 0, best_update = 0, sink = 0;
  for (int r = 0; r < reps; ++r) {
    double t0 = now();
    double s = 0;
#pragma omp parallel for simd schedule(static) reduction(+ : s)
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    best_read = std::max(best_read, 8.0 * n / (now() - t0) * 1e-9);
    sink += s;
    t0 = now();
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) z[i] = x[i] + 0.5 * y[i];
    best_triad = std::max(best_triad, 24.0 * n / (now() - t0) * 1e-9);
    t0 = now();
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) y[i] += 1e-3 * x[i];
    best_update = std::max(best_update, 24.0 * n / (now() - t0) * 1e-9);
  }
  std::printf("| stream | threads | MB per array | read GB/s | triad GB/s | update GB/s |\n"
              "|---|---|---|---|---|---|\n| omp | %d | %.0f | %.1f | %.1f | %.1f |\n",
              omp_get_max_threads(), mb, best_read, best_triad, best_update);
  if (sink == 12345.678) std::printf("%g\n", sink);
  return 0;
}

struct Shape {
  std::string op, flags;
  fint m = 0, n = 0, k = 0, ld = 0;
};

std::vector<Shape> parse_shapes(const std::string& spec) {
  std::vector<Shape> out;
  std::size_t pos = 0;
  while (pos < spec.size()) {
    std::size_t end = spec.find(';', pos);
    if (end == std::string::npos) end = spec.size();
    const std::string s = spec.substr(pos, end - pos);
    pos = end + 1;
    if (s.empty()) continue;
    std::vector<std::string> f;
    std::size_t p = 0;
    while (true) {
      const std::size_t q = s.find(':', p);
      f.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
      if (q == std::string::npos) break;
      p = q + 1;
    }
    if (f.size() < 5) {
      std::fprintf(stderr, "bad shape '%s'\n", s.c_str());
      std::exit(2);
    }
    Shape sh;
    sh.op = f[0];
    sh.flags = f[1];
    sh.m = std::atoi(f[2].c_str());
    sh.n = std::atoi(f[3].c_str());
    sh.k = std::atoi(f[4].c_str());
    sh.ld = f.size() > 5 ? std::atoi(f[5].c_str()) : 0;
    out.push_back(sh);
  }
  return out;
}

const char* kDefaultShapes =
    "gemm:NN:1024:1:256;gemm:TN:256:1:1024;gemm:NN:4096:1:512;gemm:TN:512:1:4096;"
    "gemm:NN:1024:4:256;gemm:TN:256:4:1024;gemm:NN:4096:4:512;gemm:TN:512:4:4096;"
    "gemm:NN:1024:16:256;gemm:TN:256:16:1024;gemm:NN:4096:16:512;gemm:TN:512:16:4096;"
    "trsm:LLN:256:1:0;trsm:LLT:256:1:0;trsm:LLN:1024:1:0;trsm:LLT:1024:1:0;"
    "trsm:LLN:1024:4:0;trsm:LLT:1024:4:0;"
    "gemv:N:1024:0:256;gemv:T:1024:0:256;gemv:N:4096:0:1024;gemv:T:4096:0:1024";

// One shape: operand pool and a call per route.
struct Case {
  Shape s;
  fint lda = 0, ldb = 0, ldc = 0;
  std::size_t a_elems = 0, b_elems = 0, c_elems = 0;
  int pool = 1;
  std::vector<double*> as;
  double* b = nullptr;
  double* c0 = nullptr;  // initial C / B (trsm) contents
  double* c = nullptr;   // working copy
  double a_bytes = 0.0;  // matrix bytes one call reads

  void setup(const Shape& sh, double pool_mb) {
    s = sh;
    if (s.op == "gemm") {
      const bool ta = s.flags[0] == 'T';
      const fint rows = ta ? s.k : s.m;
      const fint cols = ta ? s.m : s.k;
      lda = std::max(s.ld, rows);
      a_elems = static_cast<std::size_t>(lda) * cols;
      ldb = s.k;
      b_elems = static_cast<std::size_t>(ldb) * s.n;
      ldc = s.m;
      c_elems = static_cast<std::size_t>(ldc) * s.n;
      a_bytes = 8.0 * rows * cols;
    } else if (s.op == "trsm") {
      lda = std::max(s.ld, s.m);
      a_elems = static_cast<std::size_t>(lda) * s.m;
      ldb = s.m;
      c_elems = static_cast<std::size_t>(ldb) * s.n;
      a_bytes = 8.0 * 0.5 * s.m * s.m;
    } else if (s.op == "gemv") {
      lda = std::max(s.ld, s.m);
      a_elems = static_cast<std::size_t>(lda) * s.k;
      const bool t = s.flags[0] == 'T';
      b_elems = static_cast<std::size_t>(t ? s.m : s.k);
      c_elems = static_cast<std::size_t>(t ? s.k : s.m);
      a_bytes = 8.0 * s.m * s.k;
    } else {
      std::fprintf(stderr, "unknown op %s\n", s.op.c_str());
      std::exit(2);
    }
    pool = std::max(1, static_cast<int>(pool_mb * 1e6 / (8.0 * a_elems)));
    pool = std::min(pool, 4096);
    for (int p = 0; p < pool; ++p) {
      double* a = alloc(a_elems);
      fill(a, a_elems, 100 + p);
      if (s.op == "trsm") {  // well-conditioned triangle: unit-ish diagonal, small off-diagonal
        for (fint j = 0; j < s.m; ++j)
          for (fint i = 0; i < s.m; ++i) {
            double& v = a[i + static_cast<std::size_t>(j) * lda];
            v = i == j ? 2.0 + 0.5 * v : v / s.m;
          }
      }
      as.push_back(a);
    }
    if (b_elems) {
      b = alloc(b_elems);
      fill(b, b_elems, 7);
    }
    c0 = alloc(c_elems);
    c = alloc(c_elems);
    fill(c0, c_elems, 9);
  }

  // One call on pool operand p with the given route configuration (nullptr = real routine).
  void call(const hs::blas::RouteConfig* cfg, int p) {
    const auto& rb = hs::blas::real_blas();
    const double* a = as[p];
    const char f0 = s.flags[0];
    if (s.op == "gemm") {
      const double alpha = -1.0, beta = 1.0;
      const char tb = 'N';
      if (cfg)
        hs::blas::dgemm(*cfg, f0, tb, s.m, s.n, s.k, alpha, a, lda, b, ldb, beta, c, ldc);
      else
        rb.dgemm(&f0, &tb, &s.m, &s.n, &s.k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
    } else if (s.op == "trsm") {
      const double alpha = 1.0;
      const char side = s.flags[0], uplo = s.flags[1], tr = s.flags[2];
      const char diag = s.flags.size() > 3 ? s.flags[3] : 'N';
      if (cfg)
        hs::blas::dtrsm(*cfg, side, uplo, tr, diag, s.m, s.n, alpha, a, lda, c, ldb);
      else
        rb.dtrsm(&side, &uplo, &tr, &diag, &s.m, &s.n, &alpha, a, &lda, c, &ldb, 1, 1, 1, 1);
    } else {
      const double alpha = -1.0, beta = 1.0;
      const fint one = 1;
      if (cfg)
        hs::blas::dgemv(*cfg, f0, s.m, s.k, alpha, a, lda, b, one, beta, c, one);
      else
        rb.dgemv(&f0, &s.m, &s.k, &alpha, a, &lda, b, &one, &beta, c, &one, 1);
    }
  }

  // Seconds per call over the pool (C reset before; for trsm the right-hand side is reset before
  // every call, inside the timing - an m x n copy against the m^2/2 triangle - so that repeated
  // solves cannot drift into denormals).
  double time(const hs::blas::RouteConfig* cfg) {
    std::memcpy(c, c0, c_elems * sizeof(double));
    const int calls = std::max(pool, 8);
    const bool reset = s.op == "trsm";
    const double t0 = now();
    for (int i = 0; i < calls; ++i) {
      if (reset) std::memcpy(c, c0, c_elems * sizeof(double));
      call(cfg, i % pool);
    }
    return (now() - t0) / calls;
  }

  // Max relative difference of one call's result against the real routine's, same operands.
  double check(const hs::blas::RouteConfig* cfg) {
    std::vector<double> ref(c_elems), got(c_elems);
    std::memcpy(c, c0, c_elems * sizeof(double));
    call(nullptr, 0);
    std::memcpy(ref.data(), c, c_elems * sizeof(double));
    std::memcpy(c, c0, c_elems * sizeof(double));
    call(cfg, 0);
    std::memcpy(got.data(), c, c_elems * sizeof(double));
    double num = 0, den = 0;
    for (std::size_t i = 0; i < c_elems; ++i) {
      num = std::max(num, std::abs(ref[i] - got[i]));
      den = std::max(den, std::abs(ref[i]));
    }
    return den > 0 ? num / den : num;
  }
};

int run_shapes(const std::map<std::string, std::string>& o) {
  const std::string spec = o.count("--shapes") ? o.at("--shapes") : kDefaultShapes;
  const std::string routes_spec = o.count("--routes") ? o.at("--routes") : "gemv1,trsv1,narrow,par";
  const int reps = o.count("--reps") ? std::atoi(o.at("--reps").c_str()) : 7;
  const double pool_mb = o.count("--pool-mb") ? std::atof(o.at("--pool-mb").c_str()) : 1024.0;
  const bool inpar = o.count("--inpar") && std::atoi(o.at("--inpar").c_str()) != 0;
  std::vector<std::string> names;
  std::vector<hs::blas::RouteConfig> cfgs;
  std::size_t pos = 0;
  while (pos < routes_spec.size()) {
    std::size_t end = routes_spec.find(',', pos);
    if (end == std::string::npos) end = routes_spec.size();
    std::string r = routes_spec.substr(pos, end - pos);
    pos = end + 1;
    names.push_back(r);
    std::replace(r.begin(), r.end(), '+', ',');
    std::string unknown;
    cfgs.push_back(hs::blas::parse_route_config(r, &unknown));
    if (!unknown.empty()) {
      std::fprintf(stderr, "unknown route token(s) %s\n", unknown.c_str());
      return 2;
    }
  }
  std::printf("threads %d, %s, BLAS %s, ISA %s, pool %.0f MB, reps %d\n\n",
              omp_get_max_threads(),
              inpar ? "calls inside a parallel region by every thread (us and GB/s per thread)"
                    : "calls from one thread",
              hs::blas::real_blas().library.c_str(), hs::blas::isa_name(hs::blas::best_isa()), pool_mb,
              reps);
  std::printf("| shape | real us | real GB/s |");
  for (const auto& nm : names) std::printf(" %s us | x | GB/s | rel diff |", nm.c_str());
  std::printf("\n|---|---|---|");
  for (std::size_t i = 0; i < names.size(); ++i) std::printf("---|---|---|---|");
  std::printf("\n");
  for (const Shape& sh : parse_shapes(spec)) {
    // inpar: one operand set per thread, every call made inside a parallel region by all threads
    // at once (as in MUMPS' L0 subtrees: concurrent callers, no nested threading).
    const int nthr = inpar ? omp_get_max_threads() : 1;
    std::vector<Case> cs(nthr);
    if (inpar) {
#pragma omp parallel num_threads(nthr)
      cs[omp_get_thread_num()].setup(sh, pool_mb / nthr);
    } else {
      cs[0].setup(sh, pool_mb);
    }
    auto timed = [&](const hs::blas::RouteConfig* cfg) {
      if (!inpar) return cs[0].time(cfg);
      std::vector<double> tt(nthr);
#pragma omp parallel num_threads(nthr)
      tt[omp_get_thread_num()] = cs[omp_get_thread_num()].time(cfg);
      double sum = 0;
      for (double x : tt) sum += x;
      return sum / nthr;
    };
    const std::size_t nv = cfgs.size() + 1;
    std::vector<std::vector<double>> t(nv);
    timed(nullptr);  // warm-up
    for (int r = 0; r < reps; ++r)
      for (std::size_t q = 0; q < nv; ++q) {
        const std::size_t v = (q + r) % nv;
        t[v].push_back(timed(v == 0 ? nullptr : &cfgs[v - 1]));
      }
    const double t_real = median(t[0]);
    std::printf("| %s %s %dx%dx%d | %.2f | %.1f |", sh.op.c_str(), sh.flags.c_str(), sh.m, sh.n,
                sh.k, t_real * 1e6, cs[0].a_bytes / t_real * 1e-9);
    for (std::size_t v = 1; v < nv; ++v) {
      const double tv = median(t[v]);
      std::printf(" %.2f | %.2f | %.1f | %.1e |", tv * 1e6, t_real / tv,
                  cs[0].a_bytes / tv * 1e-9, cs[0].check(&cfgs[v - 1]));
    }
    std::printf("\n");
    std::fflush(stdout);
    for (Case& c : cs) {
      for (double* a : c.as) std::free(a);
      std::free(c.b);
      std::free(c.c0);
      std::free(c.c);
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_asmkern_kernels peak|stream|shapes [options]\n");
    return 2;
  }
  const std::string exp = argv[1];
  const auto o = parse_opts(argc, argv, 2);
  if (exp == "peak") return run_peak(o);
  if (exp == "stream") return run_stream(o);
  if (exp == "shapes") return run_shapes(o);
  std::fprintf(stderr, "unknown experiment %s\n", exp.c_str());
  return 2;
}
