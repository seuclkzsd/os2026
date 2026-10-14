// BLAS hook: the Fortran BLAS entry points and MUMPS entry-point wrappers (blas_hook.hpp).
// Only in libhsolve_blashook.so (LD_PRELOAD) and, with -DHSOLVE_BLAS_HOOK=ON, in the hsolve
// executable (CMake hsolve_link_blas_hook, which keeps MUMPS loaded and exports the entry
// points) - never in libhsolve or a test executable (it would capture their BLAS calls), except
// test_asmkern_fix_exehook, which tests that linkage.
//
// dgemm_ / dtrsm_ / dgemv_ go through the router with the process configuration
// (HS_BLAS_ROUTE); the level-1 routines MUMPS uses are forwarded unchanged and only timed.
// dmumps_c / smumps_c set the profiler phase around the real call (next_definition); MUMPS calls
// are serialized by hsolve's MUMPS call lock, so one process-wide phase is enough. dmumps_c also
// opens the GPU offload window around factorizations (JOB 2, or 4 / 5 / 6 which include one) when
// the GPU module is built in (src/gpu/offload.hpp; the mode decides, default off).
// C entry points for applications (found with dlsym(RTLD_DEFAULT), include/hsolve/gpu.hpp):
// hsolve_gpu_set_mode, hsolve_gpu_report_json.
#include <dmumps_c.h>
#include <smumps_c.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

#include "blas/blas_hook.hpp"
#include "blas/blas_profile.hpp"

#if HSOLVE_WITH_GPU
#include "gpu/offload.hpp"
#endif

namespace {

using hs::blas::fint;
using hs::blas::fstrlen;
namespace prof = hs::blas::prof;

// Configuration and profiler provenance, set up once before the first routed call.
const hs::blas::RouteConfig& config() noexcept {
  static const bool init = [] {
    if (prof::enabled())
      prof::set_info(hs::blas::to_string(hs::blas::active_route_config()),
                     hs::blas::real_blas().library);
    return true;
  }();
  (void)init;
  return hs::blas::active_route_config();
}

struct Timer {
  double t0;
  Timer() : t0(prof::enabled() ? prof::now() : 0.0) {}
  double seconds() const { return prof::now() - t0; }
};

double dd(fint x) {
  return static_cast<double>(x);
}

using DmumpsFn = void (*)(DMUMPS_STRUC_C*);
using SmumpsFn = void (*)(SMUMPS_STRUC_C*);

template <class Fn>
Fn next_symbol(const char* name) {
  void* p = hs::blas::next_definition(name);
  if (p == nullptr) {
    const std::string msg = std::string("hsolve BLAS hook: no definition of ") + name + "\n";
    [[maybe_unused]] const ssize_t rc = ::write(2, msg.data(), msg.size());
    std::abort();
  }
  return reinterpret_cast<Fn>(p);
}

template <class Struc>
prof::Phase mumps_phase(const Struc* id) {
  switch (id->job) {
    case 1:
      return prof::Phase::Analysis;
    case 2:
      return id->icntl[30] == 1 ? prof::Phase::Count : prof::Phase::Factorization;
    case 3:
      return id->nrhs > 1 ? prof::Phase::SolveP : prof::Phase::Solve1;
    default:
      return prof::Phase::OtherMumps;
  }
}

// `gpu`: the call may offload a factorization's dgemm calls (double precision only: single
// precision MUMPS calls sgemm, which the hook does not route).
template <class Struc, class Fn>
void mumps_call(Fn real, Struc* id, bool gpu) {
  const prof::Phase ph = mumps_phase(id);
  const long long nrhs = id->job == 3 ? (id->nrhs > 0 ? id->nrhs : 1) : 0;
#if HSOLVE_WITH_GPU
  const bool fact = gpu && (id->job == 2 || id->job == 4 || id->job == 5 || id->job == 6);
  // RINFOG(1): the analysis' flop estimate of the elimination (unknown before JOB 4 / 6 analyse).
  if (fact) hs::gpu::factorization_begin(id->job == 2 || id->job == 5 ? id->rinfog[0] : -1.0);
#else
  (void)gpu;
#endif
  prof::set_phase(ph);
  const Timer t;
  real(id);
  if (prof::enabled()) prof::record_mumps(ph, t.seconds(), nrhs);
  prof::set_phase(prof::Phase::None);
#if HSOLVE_WITH_GPU
  if (fact) hs::gpu::factorization_end();
#endif
}

#if !HSOLVE_WITH_GPU
// Without the GPU module the mode a program asks for (hsolve_gpu_set_mode, else HS_GPU) is still
// reported, with the reason the CPU path runs: the degrade-mode report of `hsolve --gpu on|auto`.
constexpr const char* kGpuModes[] = {"off", "auto", "on"};
std::atomic<int> g_gpu_asked{-1};  // index into kGpuModes; -1: not set

int gpu_mode_index(const char* s) noexcept {
  if (s == nullptr) return -1;
  for (int i = 0; i < 3; ++i)
    if (std::strcmp(s, kGpuModes[i]) == 0) return i;
  return -1;
}
#endif

}  // namespace

extern "C" {

void dmumps_c(DMUMPS_STRUC_C* id) {
  static const DmumpsFn real = next_symbol<DmumpsFn>("dmumps_c");
  mumps_call(real, id, true);
}

void smumps_c(SMUMPS_STRUC_C* id) {
  static const SmumpsFn real = next_symbol<SmumpsFn>("smumps_c");
  mumps_call(real, id, false);
}

void dgemm_(const char* ta, const char* tb, const fint* m, const fint* n, const fint* k,
            const double* alpha, const double* a, const fint* lda, const double* b,
            const fint* ldb, const double* beta, double* c, const fint* ldc, fstrlen, fstrlen) {
  const Timer t;
  const hs::blas::Route r =
      hs::blas::dgemm(config(), *ta, *tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
  if (prof::enabled()) {
    const double mm = dd(*m), nn = dd(*n), kk = dd(*k);
    const double cbytes = (*beta == 0.0 ? 1.0 : 2.0) * mm * nn;
    prof::record(prof::Op::Dgemm, prof::gemm_flags(*ta, *tb), *m, *n, *k, 2.0 * mm * nn * kk,
                 8.0 * (mm * kk + kk * nn + cbytes), t.seconds(), static_cast<std::uint8_t>(r));
  }
}

void dtrsm_(const char* side, const char* uplo, const char* ta, const char* diag, const fint* m,
            const fint* n, const double* alpha, const double* a, const fint* lda, double* b,
            const fint* ldb, fstrlen, fstrlen, fstrlen, fstrlen) {
  const Timer t;
  const hs::blas::Route r = hs::blas::dtrsm(config(), *side, *uplo, *ta, *diag, *m, *n, *alpha, a,
                                            *lda, b, *ldb);
  if (prof::enabled()) {
    const bool left = *side == 'L' || *side == 'l';
    const double tri = left ? dd(*m) : dd(*n);
    const double other = left ? dd(*n) : dd(*m);
    prof::record(prof::Op::Dtrsm, prof::trsm_flags(*side, *uplo, *ta, *diag), *m, *n, 0,
                 tri * tri * other, 8.0 * (0.5 * tri * tri + 2.0 * tri * other), t.seconds(),
                 static_cast<std::uint8_t>(r));
  }
}

void dgemv_(const char* tr, const fint* m, const fint* n, const double* alpha, const double* a,
            const fint* lda, const double* x, const fint* incx, const double* beta, double* y,
            const fint* incy, fstrlen) {
  const Timer t;
  const hs::blas::Route r =
      hs::blas::dgemv(config(), *tr, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
  if (prof::enabled()) {
    const bool trans = !(*tr == 'N' || *tr == 'n');
    const double xl = trans ? dd(*m) : dd(*n);
    const double yl = trans ? dd(*n) : dd(*m);
    prof::record(prof::Op::Dgemv, prof::gemv_flags(*tr), *m, *n, 0, 2.0 * dd(*m) * dd(*n),
                 8.0 * (dd(*m) * dd(*n) + xl + (*beta == 0.0 ? 1.0 : 2.0) * yl), t.seconds(),
                 static_cast<std::uint8_t>(r));
  }
}

void daxpy_(const fint* n, const double* alpha, const double* x, const fint* incx, double* y,
            const fint* incy) {
  const Timer t;
  hs::blas::real_blas().daxpy(n, alpha, x, incx, y, incy);
  if (prof::enabled())
    prof::record(prof::Op::Daxpy, 0, *n, 0, 0, 2.0 * dd(*n), 24.0 * dd(*n), t.seconds(), 0);
}

void dcopy_(const fint* n, const double* x, const fint* incx, double* y, const fint* incy) {
  const Timer t;
  hs::blas::real_blas().dcopy(n, x, incx, y, incy);
  if (prof::enabled())
    prof::record(prof::Op::Dcopy, 0, *n, 0, 0, 0.0, 16.0 * dd(*n), t.seconds(), 0);
}

void dscal_(const fint* n, const double* alpha, double* x, const fint* incx) {
  const Timer t;
  hs::blas::real_blas().dscal(n, alpha, x, incx);
  if (prof::enabled())
    prof::record(prof::Op::Dscal, 0, *n, 0, 0, dd(*n), 16.0 * dd(*n), t.seconds(), 0);
}

void dswap_(const fint* n, double* x, const fint* incx, double* y, const fint* incy) {
  const Timer t;
  hs::blas::real_blas().dswap(n, x, incx, y, incy);
  if (prof::enabled())
    prof::record(prof::Op::Dswap, 0, *n, 0, 0, 0.0, 32.0 * dd(*n), t.seconds(), 0);
}

double dnrm2_(const fint* n, const double* x, const fint* incx) {
  const Timer t;
  const double r = hs::blas::real_blas().dnrm2(n, x, incx);
  if (prof::enabled())
    prof::record(prof::Op::Dnrm2, 0, *n, 0, 0, 2.0 * dd(*n), 8.0 * dd(*n), t.seconds(), 0);
  return r;
}

// GPU offload control (include/hsolve/gpu.hpp). 0 = set; -1 = invalid name; -2 = this hook was
// built without the GPU module (HSOLVE_GPU=OFF): the mode is only recorded for the report.
int hsolve_gpu_set_mode(const char* mode) {
#if HSOLVE_WITH_GPU
  return mode != nullptr && hs::gpu::set_mode(mode) ? 0 : -1;
#else
  const int m = gpu_mode_index(mode);
  if (m < 0) return -1;
  g_gpu_asked.store(m);
  return -2;
#endif
}

// The offload's state and totals as a JSON object; the pointer stays valid until the calling
// thread's next call.
const char* hsolve_gpu_report_json() {
  static thread_local std::string buf;
#if HSOLVE_WITH_GPU
  buf = hs::gpu::report_json();
#else
  // The mode asked for (set_mode, else a valid HS_GPU; the GPU build aborts on an invalid one).
  int m = g_gpu_asked.load();
  if (m < 0) m = std::max(0, gpu_mode_index(std::getenv("HS_GPU")));
  buf = std::string("{\"mode\": \"") + kGpuModes[m] +
        "\", \"available\": false, \"reason\": \"built without the GPU module (HSOLVE_GPU=OFF)\"}";
#endif
  return buf.c_str();
}

fint idamax_(const fint* n, const double* x, const fint* incx) {
  const Timer t;
  const fint r = hs::blas::real_blas().idamax(n, x, incx);
  if (prof::enabled())
    prof::record(prof::Op::Idamax, 0, *n, 0, 0, 0.0, 8.0 * dd(*n), t.seconds(), 0);
  return r;
}

}  // extern "C"
