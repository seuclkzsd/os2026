// BLAS hook (module blas, P5 "asmkern"; docs/BLAS.md): routes selected shape classes of the dense
// BLAS calls MUMPS makes (and hsolve's own) to other existing kernels and forwards everything else
// to the real BLAS, plus a per-call profiler (blas_profile.hpp).
// Scalar set: double only. MUMPS' factorizations and solves of real and complex problems run in
// double (complex Hermitian matrices are factorized through their real symmetric embedding);
// zgemm & co. are never called by MUMPS here, single precision (smumps) is not routed.
//
// Not part of libhsolve. Library hsolve_blas holds this header's functions; the Fortran symbols
// (dgemm_, dtrsm_, dgemv_, level-1 routines for the profiler, dmumps_c/smumps_c for the phases)
// are in hook_symbols.cpp, which goes into
//   * libhsolve_blashook.so, used through LD_PRELOAD (any executable, also python): its symbols
//     come before the BLAS library in the global lookup scope, so MUMPS' calls land here;
//   * the hsolve executable when configured with -DHSOLVE_BLAS_HOOK=ON: definitions in the
//     executable are preferred by the dynamic linker for MUMPS' calls.
// Use one or the other, not both (the calls would pass through the hook twice).
// The real routines are found with dlsym(RTLD_NEXT, name): the next definition after the object
// that holds the hook (the BLAS library, or another preloaded BLAS such as BLIS or libxsmmext).
//
// Configuration: environment variable HS_BLAS_ROUTE (parse_route_config), read once per process.
// Default (unset or "off"): every call is forwarded unchanged.
//
// Thread safety: every function here may be called concurrently from any number of threads
// (MUMPS' L0-OpenMP subtrees call BLAS from inside parallel regions). Process-level state: the
// resolved function pointers and the active configuration, both written once (function-local
// statics) and immutable afterwards, and the profiler's per-thread tables (blas_profile.hpp).
//
// Results: a route changes the operation order of a call (rounding-level differences against the
// real routine), never its semantics; for a fixed thread count every route is deterministic (each
// output element is computed by one thread in a fixed order, the work split depends only on the
// shape and the thread count).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hs::blas {

using fint = int;             // Fortran INTEGER (LP64)
using fstrlen = std::size_t;  // hidden CHARACTER length argument (gfortran ABI)

extern "C" {
using DgemmFn = void (*)(const char*, const char*, const fint*, const fint*, const fint*,
                         const double*, const double*, const fint*, const double*, const fint*,
                         const double*, double*, const fint*, fstrlen, fstrlen);
using DtrsmFn = void (*)(const char*, const char*, const char*, const char*, const fint*,
                         const fint*, const double*, const double*, const fint*, double*,
                         const fint*, fstrlen, fstrlen, fstrlen, fstrlen);
using DgemvFn = void (*)(const char*, const fint*, const fint*, const double*, const double*,
                         const fint*, const double*, const fint*, const double*, double*,
                         const fint*, fstrlen);
using DtrsvFn = void (*)(const char*, const char*, const char*, const fint*, const double*,
                         const fint*, double*, const fint*, fstrlen, fstrlen, fstrlen);
using DaxpyFn = void (*)(const fint*, const double*, const double*, const fint*, double*,
                         const fint*);
using DcopyFn = void (*)(const fint*, const double*, const fint*, double*, const fint*);
using DscalFn = void (*)(const fint*, const double*, double*, const fint*);
using DswapFn = void (*)(const fint*, double*, const fint*, double*, const fint*);
using Dnrm2Fn = double (*)(const fint*, const double*, const fint*);
using IdamaxFn = fint (*)(const fint*, const double*, const fint*);
}

/// The real BLAS behind the hook (dlsym(RTLD_NEXT), resolved on first use). A routine that cannot
/// be found aborts the process with a message on stderr: a Fortran BLAS call has no way to report
/// an error, and continuing would compute garbage.
struct RealBlas {
  DgemmFn dgemm = nullptr;
  DtrsmFn dtrsm = nullptr;
  DgemvFn dgemv = nullptr;
  DtrsvFn dtrsv = nullptr;
  DaxpyFn daxpy = nullptr;
  DcopyFn dcopy = nullptr;
  DscalFn dscal = nullptr;
  DswapFn dswap = nullptr;
  Dnrm2Fn dnrm2 = nullptr;
  IdamaxFn idamax = nullptr;
  std::string library;  ///< file that provides dgemm_ (dladdr), for the profile's provenance
};
const RealBlas& real_blas() noexcept;

/// Instruction set of the hook's own kernels.
enum class Isa { Portable, Avx512 };

/// AVX-512F (and the OS saving its state) per CPUID, unless the environment variable
/// HS_BLAS_NO_AVX512 is set to anything but "" or "0" (tests force the fallback with it). Not
/// cached: reads the environment at every call.
bool cpu_has_avx512f() noexcept;

/// Which shape classes leave the real routine. Default: none (pure forwarding).
struct RouteConfig {
  /// dgemm with N = 1 (or M = 1), K >= 1, alpha != 0 -> the real dgemv (A read once, not packed).
  bool gemm_n1_gemv = false;
  /// dtrsm with SIDE = 'L', 1 <= N <= trsv_max_n, alpha = 1 -> N calls of the real dtrsv, one per
  /// column (no level-3 packing; OpenBLAS' dtrsm costs ~1 us even for an 8 x 8 triangle).
  /// 0 = off.
  int trsv_max_n = 0;
  /// dtrsm with SIDE = 'L', M <= small_trsm_max_m, 1 <= N <= narrow_max_n, alpha = 1 -> the
  /// hook's small_trsm_left (before the trsv route). 0 = off.
  int small_trsm_max_m = 0;
  /// dgemm with 2 <= N <= narrow_max_n, op(B) = B, alpha != 0, Isa::Avx512 -> the hook's narrow
  /// kernels (narrow_gemm_nn / narrow_gemm_tn: A streamed once, no packing).
  bool narrow = false;
  /// Outside OpenMP parallel regions, calls that touch at least 2 * parallel_min_bytes of A run on
  /// up to `threads` threads (at least parallel_min_bytes each), split into fixed blocks: dgemv
  /// and dgemm with N <= narrow_max_n by rows of C (the routes above or the real dgemm per block);
  /// dgemm A^T B with too few rows of C to split (M < 4 threads) by K, the partial products summed
  /// in thread order. OpenBLAS threads dgemv only from m*n >= 460800, narrow dgemm from
  /// m*n*k > 262144 and never splits K, so most fronts above MUMPS' L0 layer and hsolve's Gram
  /// products V^T W (k = n) run on one core.
  bool parallel = false;
  int narrow_max_n = 16;
  std::int64_t parallel_min_bytes = std::int64_t{1} << 18;
  /// Threads of the parallel route; 0 = omp_get_max_threads() at the call.
  int threads = 0;
  Isa isa = Isa::Portable;
};

/// Parses a route specification: comma-separated tokens
///   off | gemv1 | trsv1 | trsv=<1..64> | strsm=<1..4096> | narrow | par | all | maxn=<1..64> |
///   minkb=<int> | threads=<int> | portable
/// ("trsv1" = trsv=1, "all" = gemv1,trsv=16,strsm=64,narrow,par). isa is Avx512 when
/// cpu_has_avx512f(),
/// unless "portable" is
/// given. Unknown or malformed tokens are skipped and appended (comma-separated) to `unknown`
/// when given.
RouteConfig parse_route_config(const std::string& spec, std::string* unknown = nullptr);

/// Canonical text of a configuration (parse_route_config(to_string(c)) == c up to isa).
std::string to_string(const RouteConfig& c);

/// The specification used when HS_BLAS_ROUTE is unset: the build's HSOLVE_BLAS_ROUTE_DEFAULT
/// (CMake cache variable; "" = off).
const char* default_route_spec() noexcept;

/// The process-wide configuration: parse_route_config(getenv("HS_BLAS_ROUTE") or
/// default_route_spec()), read once. An unknown token aborts the process with a message (a
/// misspelt route must not run silently).
const RouteConfig& active_route_config() noexcept;

/// How a call was executed (profiler key).
enum class Route : std::uint8_t {
  Real = 0,
  Gemv = 1,
  Trsv = 2,
  Narrow = 3,
  Parallel = 4,
  SplitK = 5,
  SmallTrsm = 6
};
const char* route_name(Route r) noexcept;

// ---- routed operations: Fortran BLAS semantics (column-major, op chars 'N'/'T'/'C' in any case,
// 'C' = 'T' for real data). Each returns the route it took. Invalid arguments and quick-return
// cases (a zero dimension, alpha = 0, K = 0) always go to the real routine, which reports errors
// through its XERBLA and keeps the reference semantics (e.g. beta = 0 never reads C).
Route dgemm(const RouteConfig& cfg, char ta, char tb, fint m, fint n, fint k, double alpha,
            const double* a, fint lda, const double* b, fint ldb, double beta, double* c,
            fint ldc);
Route dtrsm(const RouteConfig& cfg, char side, char uplo, char ta, char diag, fint m, fint n,
            double alpha, const double* a, fint lda, double* b, fint ldb);
Route dgemv(const RouteConfig& cfg, char t, fint m, fint n, double alpha, const double* a,
            fint lda, const double* x, fint incx, double beta, double* y, fint incy);

// ---- the hook's kernels (narrow_kernels.cpp; AVX-512 code adapted from OpenBLAS'
// dgemm_small_kernel_{nn,tn}_skylakex.c, BSD-3-Clause, provenance in docs/BLAS.md). Valid
// arguments only (the router checks): m, k >= 1, 1 <= n <= 64, lda/ldb/ldc large enough.
// beta == 0 never reads C. `threads` > 1 splits the rows of C into fixed blocks over an OpenMP
// parallel region (inside a parallel region the call runs serially). Isa::Portable runs plain
// C++ loops (reference and fallback for CPUs without AVX-512).

/// C(m x n) = alpha * A * B + beta * C, A m x k (lda), B k x n (ldb), C m x n (ldc).
void narrow_gemm_nn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc);
/// C(m x n) = alpha * A^T * B + beta * C, A k x m (lda), B k x n (ldb), C m x n (ldc).
void narrow_gemm_tn(Isa isa, int threads, fint m, fint n, fint k, double alpha, const double* a,
                    fint lda, const double* b, fint ldb, double beta, double* c, fint ldc);

/// B(m x n) := op(A)^{-1} B for side L and alpha = 1 (dtrsm semantics; uplo / trans / diag
/// characters in any case, 'C' = 'T'), the reference BLAS loop orders without level-3 packing:
/// for the small triangles of MUMPS' solves (m <= 64 or so), where OpenBLAS' dtrsm spends ~1 us
/// per call in overhead. Serial; a diagonal marked unit is never read. This kernel is the
/// project's own (no existing open-source small-trsm kernel fits: LIBXSMM's is for packed
/// batches, BLIS' small trsm is in AMD's fork only); provenance in docs/BLAS.md.
void small_trsm_left(Isa isa, char uplo, char trans, char diag, fint m, fint n, const double* a,
                     fint lda, double* b, fint ldb);

/// Threads the parallel route would use for a call touching `bytes` of matrix data: 1 inside a
/// parallel region or for bytes < 2 * parallel_min_bytes, else
/// min(cfg.threads or omp_get_max_threads(), floor(bytes / parallel_min_bytes)).
int parallel_threads(const RouteConfig& cfg, double bytes) noexcept;

}  // namespace hs::blas
