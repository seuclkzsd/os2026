// BLAS hook: real-BLAS resolution, configuration and the router (blas_hook.hpp).
#include <dlfcn.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "blas/blas_hook.hpp"

namespace hs::blas {
namespace {

[[noreturn]] void fatal(const std::string& msg) noexcept {
  const std::string line = "hsolve BLAS hook: " + msg + "\n";
  // write(2), not stdio: may run inside a Fortran BLAS call of any thread.
  [[maybe_unused]] const ssize_t rc = ::write(2, line.data(), line.size());
  std::abort();
}

void* resolve_next(const char* name) noexcept {
  void* p = dlsym(RTLD_NEXT, name);
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

}  // namespace

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

RouteConfig parse_route_config(const std::string& spec, std::string* unknown) {
  RouteConfig c;
  bool portable = false;
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
        c.trsm_n1_trsv = true;
      } else if (key == "narrow") {
        c.narrow = true;
      } else if (key == "par") {
        c.parallel = true;
      } else if (key == "all") {
        c.gemm_n1_gemv = c.trsm_n1_trsv = c.narrow = c.parallel = true;
      } else if (key == "portable") {
        portable = true;
      } else {
        ok = false;
      }
    } else if (key == "maxn" && parse_int(val, 1, 64, &v)) {
      c.narrow_max_n = static_cast<int>(v);
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
  c.isa = (!portable && cpu_has_avx512f()) ? Isa::Avx512 : Isa::Portable;
  return c;
}

std::string to_string(const RouteConfig& c) {
  std::string s;
  auto add = [&s](const std::string& t) {
    if (!s.empty()) s += ',';
    s += t;
  };
  if (c.gemm_n1_gemv) add("gemv1");
  if (c.trsm_n1_trsv) add("trsv1");
  if (c.narrow) add("narrow");
  if (c.parallel) add("par");
  if (s.empty()) add("off");
  add("maxn=" + std::to_string(c.narrow_max_n));
  add("minkb=" + std::to_string(c.parallel_min_bytes / 1024));
  add("threads=" + std::to_string(c.threads));
  if (c.isa == Isa::Portable) add("portable");
  return s;
}

const RouteConfig& active_route_config() noexcept {
  static const RouteConfig cfg = [] {
    const char* v = std::getenv("HS_BLAS_ROUTE");
    std::string unknown;
    RouteConfig c = parse_route_config(v != nullptr ? v : "", &unknown);
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
  }
  return "?";
}

int parallel_threads(const RouteConfig& cfg, double bytes) noexcept {
  if (omp_in_parallel()) return 1;
  const int max_t = cfg.threads > 0 ? cfg.threads : omp_get_max_threads();
  const double per = std::max<double>(1.0, static_cast<double>(cfg.parallel_min_bytes));
  const double want = bytes / per;
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
  if (n >= 2 && n <= cfg.narrow_max_n && !tb) {
    if (cfg.narrow && cfg.isa == Isa::Avx512) {
      if (ta)
        narrow_gemm_tn(cfg.isa, threads, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      else
        narrow_gemm_nn(cfg.isa, threads, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      return threads > 1 ? Route::Parallel : Route::Narrow;
    }
    if (threads > 1 && m >= 16 * threads) {
      parallel_gemm(threads, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      return Route::Parallel;
    }
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
  if (cfg.trsm_n1_trsv && valid && s == 'L' && n == 1 && m >= 1 && alpha == 1.0) {
    const fint one = 1;
    rb.dtrsv(&uplo, &ta, &diag, &m, a, &lda, b, &one, 1, 1, 1);
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
