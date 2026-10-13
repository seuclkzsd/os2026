// BLAS hook: real-BLAS resolution, configuration and the router (blas_hook.hpp).
#include <dlfcn.h>
#include <link.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "blas/blas_hook.hpp"

// Route specification used when HS_BLAS_ROUTE is unset (CMake cache HSOLVE_BLAS_ROUTE_DEFAULT).
#ifndef HSOLVE_BLAS_ROUTE_DEFAULT
#define HSOLVE_BLAS_ROUTE_DEFAULT ""
#endif

namespace hs::blas {
namespace {

[[noreturn]] void fatal(const std::string& msg) noexcept {
  const std::string line = "hsolve BLAS hook: " + msg + "\n";
  // write(2), not stdio: may run inside a Fortran BLAS call of any thread.
  [[maybe_unused]] const ssize_t rc = ::write(2, line.data(), line.size());
  std::abort();
}

void* resolve_next(const char* name) noexcept {
  void* p = next_definition(name);
  if (p == nullptr)
    fatal(std::string("no definition of ") + name +
          " after the hook (is a BLAS library loaded after it?)");
  return p;
}

char upper(char c) {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

// 'N' -> false, 'T'/'C' -> true; anything else is invalid (*ok = false).
bool trans_flag(char c, bool* ok) {
  c = upper(c);
  if (c == 'N') return false;
  if (c == 'T' || c == 'C') return true;
  *ok = false;
  return false;
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

bool parse_int(const std::string& s, long long lo, long long hi, long long* out) {
  if (s.empty()) return false;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (end == nullptr || *end != '\0' || v < lo || v > hi) return false;
  *out = v;
  return true;
}

// Gemv with the real routine on `threads` fixed blocks of y (rows of A for 'N', columns of A for
// 'T'); incx, incy > 0. Each element of y is computed by one real dgemv call on one thread.
void parallel_gemv(int threads, bool trans, fint m, fint n, double alpha, const double* a,
                   fint lda, const double* x, fint incx, double beta, double* y, fint incy) {
  const RealBlas& rb = real_blas();
  const fint len = trans ? n : m;
  // Blocks of whole cache lines of y for unit stride (8 doubles), else of single elements.
  const fint align = incy == 1 ? 8 : 1;
  const fint per = ((len + threads - 1) / threads + align - 1) / align * align;
  const char t = trans ? 'T' : 'N';
#pragma omp parallel num_threads(threads)
  {
    const fint i0 = static_cast<fint>(omp_get_thread_num()) * per;
    const fint i1 = std::min<fint>(len, i0 + per);
    if (i0 < i1) {
      const fint cnt = i1 - i0;
      if (!trans) {
        rb.dgemv(&t, &cnt, &n, &alpha, a + i0, &lda, x, &incx, &beta,
                 y + static_cast<std::ptrdiff_t>(i0) * incy, &incy, 1);
      } else {
        rb.dgemv(&t, &m, &cnt, &alpha, a + static_cast<std::ptrdiff_t>(i0) * lda, &lda, x, &incx,
                 &beta, y + static_cast<std::ptrdiff_t>(i0) * incy, &incy, 1);
      }
    }
  }
}

// Gemm (op(B) = B or B^T) with the real routine on `threads` fixed row blocks of C.
void parallel_gemm(int threads, bool ta, bool tb, fint m, fint n, fint k, double alpha,
                   const double* a, fint lda, const double* b, fint ldb, double beta, double* c,
                   fint ldc) {
  const RealBlas& rb = real_blas();
  const fint per = ((m + threads - 1) / threads + 7) / 8 * 8;
  const char cta = ta ? 'T' : 'N';
  const char ctb = tb ? 'T' : 'N';
#pragma omp parallel num_threads(threads)
  {
    const fint i0 = static_cast<fint>(omp_get_thread_num()) * per;
    const fint i1 = std::min<fint>(m, i0 + per);
    if (i0 < i1) {
      const fint cnt = i1 - i0;
      const double* ab = ta ? a + static_cast<std::ptrdiff_t>(i0) * lda : a + i0;
      rb.dgemm(&cta, &ctb, &cnt, &n, &k, &alpha, ab, &lda, b, &ldb, &beta, c + i0, &ldc, 1, 1);
    }
  }
}

// C = alpha * A^T * B + beta * C with A k x m, B k x n, split over K into `threads` fixed chunks:
// each thread forms its partial product with the real dgemm (beta = 0), the partials are added
// in thread order, so the result depends only on the shape and the thread count.
void splitk_gemm_tn(int threads, fint m, fint n, fint k, double alpha, const double* a, fint lda,
                    const double* b, fint ldb, double beta, double* c, fint ldc) {
  const RealBlas& rb = real_blas();
  const fint per = ((k + threads - 1) / threads + 7) / 8 * 8;
  const std::size_t mn = static_cast<std::size_t>(m) * n;
  std::vector<double> part(mn * threads, 0.0);
#pragma omp parallel num_threads(threads)
  {
    const int t = omp_get_thread_num();
    const fint k0 = static_cast<fint>(t) * per;
    const fint k1 = std::min<fint>(k, k0 + per);
    if (k0 < k1) {
      const fint kc = k1 - k0;
      const double one = 1.0, zero = 0.0;
      const char tr = 'T', nt = 'N';
      rb.dgemm(&tr, &nt, &m, &n, &kc, &one, a + k0, &lda, b + k0, &ldb, &zero, part.data() + mn * t,
               &m, 1, 1);
    }
  }
  for (fint j = 0; j < n; ++j)
    for (fint i = 0; i < m; ++i) {
      const std::size_t q = i + static_cast<std::size_t>(j) * m;
      double s = part[q];
      for (int t = 1; t < threads; ++t) s += part[q + mn * t];
      double& cij = c[i + static_cast<std::size_t>(j) * ldc];
      cij = beta == 0.0 ? alpha * s : alpha * s + beta * cij;
    }
}

}  // namespace

void* find_loaded_definition(const char* name, const void* skip_base) noexcept {
  // Names first, lookups afterwards: dl_iterate_phdr holds the loader's lock during the callback.
  std::vector<std::string> objects;
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) -> int {
        if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0')
          static_cast<std::vector<std::string>*>(data)->emplace_back(info->dlpi_name);
        return 0;
      },
      &objects);
  for (const std::string& obj : objects) {
    void* h = dlopen(obj.c_str(), RTLD_LAZY | RTLD_NOLOAD);  // already loaded: no side effects
    if (h == nullptr) continue;                              // e.g. the vDSO
    void* p = dlsym(h, name);
    dlclose(h);
    Dl_info di{};
    if (p != nullptr && dladdr(p, &di) != 0 && di.dli_fbase != skip_base) return p;
  }
  return nullptr;
}

void* next_definition(const char* name) noexcept {
  if (void* p = dlsym(RTLD_NEXT, name)) return p;
  Dl_info self{};
  if (dladdr(reinterpret_cast<void*>(&next_definition), &self) == 0) return nullptr;
  return find_loaded_definition(name, self.dli_fbase);
}

const RealBlas& real_blas() noexcept {
  static const RealBlas rb = [] {
    RealBlas r;
    r.dgemm = reinterpret_cast<DgemmFn>(resolve_next("dgemm_"));
    r.dtrsm = reinterpret_cast<DtrsmFn>(resolve_next("dtrsm_"));
    r.dgemv = reinterpret_cast<DgemvFn>(resolve_next("dgemv_"));
    r.dtrsv = reinterpret_cast<DtrsvFn>(resolve_next("dtrsv_"));
    r.daxpy = reinterpret_cast<DaxpyFn>(resolve_next("daxpy_"));
    r.dcopy = reinterpret_cast<DcopyFn>(resolve_next("dcopy_"));
    r.dscal = reinterpret_cast<DscalFn>(resolve_next("dscal_"));
    r.dswap = reinterpret_cast<DswapFn>(resolve_next("dswap_"));
    r.dnrm2 = reinterpret_cast<Dnrm2Fn>(resolve_next("dnrm2_"));
    r.idamax = reinterpret_cast<IdamaxFn>(resolve_next("idamax_"));
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(r.dgemm), &info) != 0 && info.dli_fname != nullptr)
      r.library = info.dli_fname;
    return r;
  }();
  return rb;
}

bool cpu_has_avx512f() noexcept {
  if (env_flag("HS_BLAS_NO_AVX512")) return false;
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") != 0;
}

bool cpu_has_avx2_fma() noexcept {
  if (env_flag("HS_BLAS_NO_AVX2")) return false;
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
}

Isa best_isa() noexcept {
  if (cpu_has_avx512f()) return Isa::Avx512;
  if (cpu_has_avx2_fma()) return Isa::Avx2;
  return Isa::Portable;
}

const char* isa_name(Isa isa) noexcept {
  switch (isa) {
    case Isa::Avx512:
      return "avx512";
    case Isa::Avx2:
      return "avx2";
    case Isa::Portable:
      return "portable";
  }
  return "?";
}

RouteConfig parse_route_config(const std::string& spec, std::string* unknown) {
  RouteConfig c;
  Isa cap = Isa::Avx512;
  std::size_t pos = 0;
  while (pos <= spec.size()) {
    std::size_t end = spec.find(',', pos);
    if (end == std::string::npos) end = spec.size();
    std::string tok = spec.substr(pos, end - pos);
    tok.erase(0, tok.find_first_not_of(" \t"));
    tok.erase(tok.find_last_not_of(" \t") + 1);
    pos = end + 1;
    if (tok.empty()) continue;
    const std::size_t eq = tok.find('=');
    const std::string key = tok.substr(0, eq);
    const std::string val = eq == std::string::npos ? "" : tok.substr(eq + 1);
    long long v = 0;
    bool ok = true;
    if (eq == std::string::npos) {
      if (key == "off") {
        c = RouteConfig{};
      } else if (key == "gemv1") {
        c.gemm_n1_gemv = true;
      } else if (key == "trsv1") {
        c.trsv_max_n = std::max(c.trsv_max_n, 1);
      } else if (key == "narrow") {
        c.narrow = true;
      } else if (key == "par") {
        c.parallel = true;
      } else if (key == "all") {
        c.gemm_n1_gemv = c.narrow = c.parallel = true;
        c.small_trsm_max_m = 512;
      } else if (key == "portable") {
        cap = Isa::Portable;
      } else if (key == "avx2") {
        cap = std::min(cap, Isa::Avx2);
      } else {
        ok = false;
      }
    } else if (key == "strsm" && parse_int(val, 1, 4096, &v)) {
      c.small_trsm_max_m = static_cast<int>(v);
    } else if (key == "trsv" && parse_int(val, 1, 64, &v)) {
      c.trsv_max_n = static_cast<int>(v);
    } else if (key == "maxn" && parse_int(val, 1, 64, &v)) {
      c.narrow_max_n = static_cast<int>(v);
    } else if (key == "rows" && parse_int(val, 8, 65536, &v)) {
      c.nn_split_rows = static_cast<int>(v);
    } else if (key == "minkb" && parse_int(val, 0, 1LL << 30, &v)) {
      c.parallel_min_bytes = static_cast<std::int64_t>(v) * 1024;
    } else if (key == "threads" && parse_int(val, 0, 4096, &v)) {
      c.threads = static_cast<int>(v);
    } else {
      ok = false;
    }
    if (!ok && unknown != nullptr) {
      if (!unknown->empty()) *unknown += ',';
      *unknown += tok;
    }
  }
  c.isa = std::min(cap, best_isa());
  return c;
}

std::string to_string(const RouteConfig& c) {
  std::string s;
  auto add = [&s](const std::string& t) {
    if (!s.empty()) s += ',';
    s += t;
  };
  if (c.gemm_n1_gemv) add("gemv1");
  if (c.trsv_max_n > 0) add("trsv=" + std::to_string(c.trsv_max_n));
  if (c.small_trsm_max_m > 0) add("strsm=" + std::to_string(c.small_trsm_max_m));
  if (c.narrow) add("narrow");
  if (c.parallel) add("par");
  if (s.empty()) add("off");
  add("maxn=" + std::to_string(c.narrow_max_n));
  add("rows=" + std::to_string(c.nn_split_rows));
  add("minkb=" + std::to_string(c.parallel_min_bytes / 1024));
  add("threads=" + std::to_string(c.threads));
  if (c.isa != Isa::Avx512) add(isa_name(c.isa));
  return s;
}

const char* default_route_spec() noexcept {
  return HSOLVE_BLAS_ROUTE_DEFAULT;
}

const RouteConfig& active_route_config() noexcept {
  static const RouteConfig cfg = [] {
    const char* v = std::getenv("HS_BLAS_ROUTE");
    std::string unknown;
    RouteConfig c = parse_route_config(v != nullptr ? v : default_route_spec(), &unknown);
    if (!unknown.empty()) fatal("unknown HS_BLAS_ROUTE token(s) '" + unknown + "'");
    return c;
  }();
  return cfg;
}

const char* route_name(Route r) noexcept {
  switch (r) {
    case Route::Real:
      return "real";
    case Route::Gemv:
      return "gemv";
    case Route::Trsv:
      return "trsv";
    case Route::Narrow:
      return "narrow";
    case Route::Parallel:
      return "parallel";
    case Route::SplitK:
      return "splitk";
    case Route::SmallTrsm:
      return "strsm";
  }
  return "?";
}

int small_trsm_limit(const RouteConfig& cfg) noexcept {
  const int cap = cfg.isa == Isa::Avx512 ? 4096 : cfg.isa == Isa::Avx2 ? 256 : 32;
  return std::min(cfg.small_trsm_max_m, cap);
}

int parallel_threads(const RouteConfig& cfg, double bytes) noexcept {
  if (omp_in_parallel()) return 1;
  const int max_t = cfg.threads > 0 ? cfg.threads : omp_get_max_threads();
  const double per = std::max<double>(1.0, static_cast<double>(cfg.parallel_min_bytes));
  const double want = std::floor(bytes / per);
  if (want < 2.0) return 1;
  return std::max(1, std::min(max_t, static_cast<int>(want)));
}

Route dgemm(const RouteConfig& cfg, char ta_c, char tb_c, fint m, fint n, fint k, double alpha,
            const double* a, fint lda, const double* b, fint ldb, double beta, double* c,
            fint ldc) {
  const RealBlas& rb = real_blas();
  bool ok = true;
  const bool ta = trans_flag(ta_c, &ok);
  const bool tb = trans_flag(tb_c, &ok);
  const fint nrow_a = ta ? k : m;
  const fint nrow_b = tb ? n : k;
  ok = ok && m >= 0 && n >= 0 && k >= 0 && lda >= std::max<fint>(1, nrow_a) &&
       ldb >= std::max<fint>(1, nrow_b) && ldc >= std::max<fint>(1, m);
  auto forward = [&] {
    rb.dgemm(&ta_c, &tb_c, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
    return Route::Real;
  };
  if (!ok || m == 0 || n == 0 || k == 0 || alpha == 0.0) return forward();
  const double a_bytes = 8.0 * static_cast<double>(m) * static_cast<double>(k);
  const int threads = cfg.parallel ? parallel_threads(cfg, a_bytes) : 1;

  // Narrow kernels: split over threads by the parallel route, serially inside parallel regions
  // and for small calls; without the parallel route a large call outside parallel regions stays
  // with the real routine, which may thread it (one thread of the narrow kernel would not keep up).
  const bool large_serial = !cfg.parallel && !omp_in_parallel() &&
                            a_bytes >= 2.0 * static_cast<double>(cfg.parallel_min_bytes);
  if (cfg.narrow && cfg.isa != Isa::Portable && !tb && n >= 2 && n <= cfg.narrow_max_n &&
      !large_serial) {
    if (ta)
      narrow_gemm_tn(cfg.isa, threads, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else
      narrow_gemm_nn(cfg.isa, threads, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                     cfg.nn_split_rows);
    return threads > 1 ? Route::Parallel : Route::Narrow;
  }
  if (ta && !tb && n <= cfg.narrow_max_n && threads > 1 && m < 4 * threads &&
      k >= 1024 * threads) {
    splitk_gemm_tn(threads, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return Route::SplitK;
  }
  if (cfg.gemm_n1_gemv && n == 1) {
    // c(m) = alpha * op(A) * op(B)(:, 1) + beta * c: op(B) column = B(:,1) or B(1,:) (stride ldb).
    const fint incx = tb ? ldb : 1;
    const fint one = 1;
    if (threads > 1) {
      parallel_gemv(threads, ta, ta ? k : m, ta ? m : k, alpha, a, lda, b, incx, beta, c, one);
      return Route::Parallel;
    }
    const char t = ta ? 'T' : 'N';
    const fint rows = ta ? k : m;
    const fint cols = ta ? m : k;
    rb.dgemv(&t, &rows, &cols, &alpha, a, &lda, b, &incx, &beta, c, &one, 1);
    return Route::Gemv;
  }
  if (cfg.gemm_n1_gemv && m == 1) {
    // c(1, n) = alpha * op(A)(1, :) * op(B) + beta * c: c^T = op(B)^T * op(A)(1,:)^T.
    // op(A) row = A(1,:) (stride lda) or A(:,1) (unit stride); c^T has stride ldc.
    const fint incx = ta ? 1 : lda;
    const char t = tb ? 'N' : 'T';
    const fint rows = tb ? n : k;
    const fint cols = tb ? k : n;
    rb.dgemv(&t, &rows, &cols, &alpha, b, &ldb, a, &incx, &beta, c, &ldc, 1);
    return Route::Gemv;
  }
  if (n >= 2 && n <= cfg.narrow_max_n && !tb && threads > 1 && m >= 16 * threads) {
    parallel_gemm(threads, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return Route::Parallel;
  }
  return forward();
}

Route dtrsm(const RouteConfig& cfg, char side, char uplo, char ta, char diag, fint m, fint n,
            double alpha, const double* a, fint lda, double* b, fint ldb) {
  const RealBlas& rb = real_blas();
  const char s = upper(side), u = upper(uplo), t = upper(ta), d = upper(diag);
  const bool valid = (s == 'L' || s == 'R') && (u == 'U' || u == 'L') &&
                     (t == 'N' || t == 'T' || t == 'C') && (d == 'U' || d == 'N') && m >= 0 &&
                     n >= 0 && lda >= std::max<fint>(1, s == 'L' ? m : n) &&
                     ldb >= std::max<fint>(1, m);
  if (valid && s == 'L' && n >= 1 && n <= cfg.narrow_max_n && m >= 1 &&
      m <= small_trsm_limit(cfg) && alpha == 1.0) {
    small_trsm_left(cfg.isa, u, t, d, m, n, a, lda, b, ldb);
    return Route::SmallTrsm;
  }
  if (valid && s == 'L' && n >= 1 && n <= cfg.trsv_max_n && m >= 1 && alpha == 1.0) {
    const fint one = 1;
    for (fint j = 0; j < n; ++j)
      rb.dtrsv(&uplo, &ta, &diag, &m, a, &lda, b + static_cast<std::ptrdiff_t>(j) * ldb, &one, 1, 1,
               1);
    return Route::Trsv;
  }
  rb.dtrsm(&side, &uplo, &ta, &diag, &m, &n, &alpha, a, &lda, b, &ldb, 1, 1, 1, 1);
  return Route::Real;
}

Route dgemv(const RouteConfig& cfg, char t_c, fint m, fint n, double alpha, const double* a,
            fint lda, const double* x, fint incx, double beta, double* y, fint incy) {
  const RealBlas& rb = real_blas();
  bool ok = true;
  const bool trans = trans_flag(t_c, &ok);
  ok = ok && m >= 0 && n >= 0 && lda >= std::max<fint>(1, m) && incx != 0 && incy != 0;
  if (cfg.parallel && ok && m > 0 && n > 0 && alpha != 0.0 && incx > 0 && incy > 0) {
    const int threads =
        parallel_threads(cfg, 8.0 * static_cast<double>(m) * static_cast<double>(n));
    if (threads > 1) {
      parallel_gemv(threads, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      return Route::Parallel;
    }
  }
  rb.dgemv(&t_c, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
  return Route::Real;
}

}  // namespace hs::blas
