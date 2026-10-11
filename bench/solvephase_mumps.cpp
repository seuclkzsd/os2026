// Inside one MUMPS solve (JOB=3) of the shift-invert operator (module ops, P4 "solvephase").
//
// Drives dmumps directly (the library's MUMPS input assembly, src/factor/mumps_input.hpp, with the
// driver's controls: ICNTL(8)=7 scaling, ICNTL(13)=1, ICNTL(24)=1 with CNTL(3)=1e-8, L0-OpenMP
// unless --det 1, the given ordering) so that solve-time controls can be varied on one
// factorization, and interposes the BLAS routines MUMPS calls (dtrsm_, dgemv_, dgemm_, dcopy_ are
// defined in this executable; libdmumps_seq binds to them, they forward to the CBLAS entry points
// of OpenBLAS) to split the solve time:
//   seq_*  BLAS time outside OpenMP parallel regions (fronts above the L0 layer: wall time, the
//          BLAS library may use several threads internally)
//   par_*  BLAS time inside parallel regions (L0 subtrees: summed over threads)
//   other  solve wall time - seq BLAS time (MUMPS' own code + the L0 phase)
// Experiments (--exp):
//   split    time split of a p-column solve for every p in --p (default)
//   icntl27  solve time for ICNTL(27) (RHS blocking) in --icntl27 (0 = p) for every p in --p
//   icntl21  centralized solution (ICNTL(21) = 0) vs distributed SOL_LOC (1) scattered back
//   partrsm  per-column solve time with the sequential dtrsm calls of few right-hand sides done
//            by a blocked OpenMP-parallel triangular solve (this file, --nb block size) instead of
//            OpenBLAS (single-threaded for N = 1), interleaved with plain solves; checks that the
//            solutions agree
//   gemvfwd  solve time with every dgemm of one right-hand side (N = 1) done as one dgemv,
//            interleaved with plain solves (is OpenBLAS' dgemm slower than dgemv for N = 1?)
//   skinny   solve time with every dgemm of N <= 16 right-hand sides done by this file's
//            AVX-512 skinny kernel (reads A once, no packing), interleaved with plain solves
//   solvethreads  solve time with ICNTL(16) (solve threads) in --solve-threads, the factorization
//            done with all threads; checks the solutions against an all-thread solve
// Output: one JSON object per line on stdout.
//
// Usage (via the resource guard; threads = cores given):
//   python3 tools/run/guard.py --skip-busy-check --cores 34,35,36,37 --mem-gb 6 --timeout 1800 --
//     build/release/bench_solvephase_mumps CASE_DIR
//     [--exp split|icntl21|icntl27|partrsm|gemvfwd|skinny|solvethreads] [--ordering amd|scotch]
//     [--det 0|1] [--p 1,2,4,8,16] [--reps 5] [--icntl27 -32,0,32] [--solve-threads 1,2,4]
//     [--nb 256] [--min-m 512] [--max-n 4]
// Measured with it (4 cores of the 2x Xeon 6336Y server, see the P4 solvephase report):
// ICNTL(27) and ICNTL(21) change nothing (+-5%), the parallel trsm nothing (the sequential
// triangles above L0 are small), dgemv for N = 1 saves 7-11% of a one-column solve on the real
// cases, the skinny kernel 0-19% for N <= 2 but loses for N >= 8.
#include <dmumps_c.h>
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "factor/mumps_input.hpp"
#include "hsolve/io.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/util.hpp"
#include "linalg/dense_ops.hpp"

// ------------------------------------------------------------------------------ BLAS interposer

namespace probe {

enum Kind { kTrsm = 0, kGemv = 1, kGemm = 2, kCopy = 3, kKinds = 4 };
constexpr int kMaxThreads = 128;

struct alignas(64) Slot {
  double seconds[kKinds][2] = {};  // [kind][in parallel region]
  double bytes[kKinds][2] = {};    // matrix bytes read
  long long calls[kKinds][2] = {};
};

Slot slots[kMaxThreads];
std::atomic<bool> recording{false};
std::atomic<bool> parallel_trsm{false};
std::atomic<bool> gemm_as_gemv{false};  // dgemm with N = 1 forwarded to dgemv (exp gemvfwd)
int trsm_nb = 256;
int trsm_min_m = 512;
int trsm_max_n = 4;

// The real BLAS behind the interposed Fortran symbols: OpenBLAS' CBLAS entry points, which call
// its kernels directly (not the dtrsm_ / dgemv_ / ... symbols defined below).
extern "C" {
void cblas_dtrsm(int layout, int side, int uplo, int transa, int diag, int m, int n, double alpha,
                 const double* a, int lda, double* b, int ldb);
void cblas_dgemv(int layout, int trans, int m, int n, double alpha, const double* a, int lda,
                 const double* x, int incx, double beta, double* y, int incy);
void cblas_dgemm(int layout, int ta, int tb, int m, int n, int k, double alpha, const double* a,
                 int lda, const double* b, int ldb, double beta, double* c, int ldc);
void cblas_dcopy(int n, const double* x, int incx, double* y, int incy);
}
constexpr int kColMajor = 102;
int cb_trans(char t) {
  return (t == 'N' || t == 'n') ? 111 : 112;
}
int cb_uplo(char u) {
  return (u == 'U' || u == 'u') ? 121 : 122;
}
int cb_diag(char d) {
  return (d == 'U' || d == 'u') ? 132 : 131;
}
int cb_side(char s) {
  return (s == 'L' || s == 'l') ? 141 : 142;
}

void real_trsm(char side, char uplo, char trans, char diag, int m, int n, double alpha,
               const double* a, int lda, double* b, int ldb) {
  cblas_dtrsm(kColMajor, cb_side(side), cb_uplo(uplo), cb_trans(trans), cb_diag(diag), m, n, alpha,
              a, lda, b, ldb);
}
void real_gemv(char trans, int m, int n, double alpha, const double* a, int lda, const double* x,
               int incx, double beta, double* y, int incy) {
  cblas_dgemv(kColMajor, cb_trans(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}
void real_gemm(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda,
               const double* b, int ldb, double beta, double* c, int ldc) {
  cblas_dgemm(kColMajor, cb_trans(ta), cb_trans(tb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

struct Scope {
  Kind kind;
  double bytes;
  double t0 = 0.0;
  bool on;
  Scope(Kind k, double b) : kind(k), bytes(b), on(recording.load(std::memory_order_relaxed)) {
    if (on) t0 = omp_get_wtime();
  }
  ~Scope() {
    if (!on) return;
    const int par = omp_in_parallel() ? 1 : 0;
    Slot& s = slots[omp_get_thread_num() % kMaxThreads];
    s.seconds[kind][par] += omp_get_wtime() - t0;
    s.bytes[kind][par] += bytes;
    ++s.calls[kind][par];
  }
};

void reset() {
  for (auto& s : slots) s = Slot{};
}

Slot total() {
  Slot t;
  for (const auto& s : slots)
    for (int k = 0; k < kKinds; ++k)
      for (int p = 0; p < 2; ++p) {
        t.seconds[k][p] += s.seconds[k][p];
        t.bytes[k][p] += s.bytes[k][p];
        t.calls[k][p] += s.calls[k][p];
      }
  return t;
}

/// B = alpha * op(A)^{-1} B, side = 'L', for an M x N block B with few columns: blocks of nb rows,
/// the diagonal block solved by one thread (OpenBLAS), the update of the remaining rows by all
/// threads (one dgemv / dgemm per row chunk; OpenBLAS runs single-threaded inside the region).
void blocked_parallel_trsm(char uplo, char trans, char diag, int m, int n, double alpha,
                           const double* a, int lda, double* b, int ldb) {
  const bool lower_op = (uplo == 'L') == (trans == 'N');  // op(A) lower -> forward substitution
  const int nb = trsm_nb;
  if (alpha != 1.0) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) b[i + static_cast<std::size_t>(j) * ldb] *= alpha;
  }
#pragma omp parallel
  {
    const int nblocks = (m + nb - 1) / nb;
    for (int step = 0; step < nblocks; ++step) {
      // Block [k0, k1) solved now; rows still to update: below (forward) or above (backward).
      const int k0 = lower_op ? step * nb : std::max(0, m - (step + 1) * nb);
      const int k1 = lower_op ? std::min(m, k0 + nb) : m - step * nb;
      const int kb = k1 - k0;
#pragma omp single
      real_trsm('L', uplo, trans, diag, kb, n, 1.0, a + k0 + static_cast<std::size_t>(k0) * lda,
                lda, b + k0, ldb);
      // implicit barrier
      const int r_begin = lower_op ? k1 : 0;
      const int r_end = lower_op ? m : k0;
      const int rows = r_end - r_begin;
      if (rows > 0) {
        const int chunk = 256;
        const int nchunks = (rows + chunk - 1) / chunk;
#pragma omp for schedule(static)
        for (int c = 0; c < nchunks; ++c) {
          const int r0 = r_begin + c * chunk;
          const int rc = std::min(chunk, r_end - r0);
          double* bo = b + r0;
          const double* bk = b + k0;
          if (trans == 'N') {
            // B[r0:r0+rc] -= A[r0:r0+rc, k0:k1] * B[k0:k1]
            const double* ap = a + r0 + static_cast<std::size_t>(k0) * lda;
            if (n == 1) {
              real_gemv('N', rc, kb, -1.0, ap, lda, bk, 1, 1.0, bo, 1);
            } else {
              real_gemm('N', 'N', rc, n, kb, -1.0, ap, lda, bk, ldb, 1.0, bo, ldb);
            }
          } else {
            // B[r0:r0+rc] -= A[k0:k1, r0:r0+rc]^T * B[k0:k1]
            const double* ap = a + k0 + static_cast<std::size_t>(r0) * lda;
            if (n == 1) {
              real_gemv('T', kb, rc, -1.0, ap, lda, bk, 1, 1.0, bo, 1);
            } else {
              real_gemm('T', 'N', rc, n, kb, -1.0, ap, lda, bk, ldb, 1.0, bo, ldb);
            }
          }
        }
        // implicit barrier
      }
    }
  }
}

std::map<std::string, std::pair<long long, double>>
    trsm_shapes;  // "side uplo trans diag" -> calls, MB

// ---- skinny GEMM (exp skinny): C = alpha op(A) B + beta C for N <= 16 right-hand sides that reads
// A exactly once, without the packing copy OpenBLAS makes of A for any N (a memory-bound update:
// A is a factor block, B / C the few right-hand sides). AVX-512F; OpenMP over rows of C outside
// parallel regions (MUMPS' upper tree), serial inside (L0 subtrees). B is first transposed into a
// small row-major buffer (k x N) so that the N values of one row of B are one cache line.
std::atomic<bool> skinny_gemm{false};
int skinny_max_n = 16;

#define HS_BENCH_AVX512 __attribute__((target("avx512f")))

/// C[r0:r1, 0:n] = alpha A[r0:r1, :] Bt^T + beta C (A m x k column-major, Bt k x n row-major).
template <int NB>
HS_BENCH_AVX512 void skinny_nn(int r0, int r1, int n, int k, double alpha, const double* a, int lda,
                               const double* bt, double beta, double* c, int ldc) {
  for (int r = r0; r < r1; r += 8) {
    const int rows = std::min(8, r1 - r);
    const __mmask8 mk =
        rows == 8 ? static_cast<__mmask8>(0xFF) : static_cast<__mmask8>((1u << rows) - 1u);
    __m512d acc[NB];
    for (int j = 0; j < NB; ++j) acc[j] = _mm512_setzero_pd();
    const double* ar = a + r;
    for (int l = 0; l < k; ++l) {
      const __m512d av = _mm512_maskz_loadu_pd(mk, ar + static_cast<std::size_t>(l) * lda);
      const double* bl = bt + static_cast<std::size_t>(l) * n;
      for (int j = 0; j < NB; ++j)
        if (j < n) acc[j] = _mm512_fmadd_pd(av, _mm512_set1_pd(bl[j]), acc[j]);
    }
    for (int j = 0; j < NB; ++j) {
      if (j >= n) break;
      double* cj = c + r + static_cast<std::size_t>(j) * ldc;
      __m512d out = _mm512_mul_pd(_mm512_set1_pd(alpha), acc[j]);
      if (beta != 0.0)
        out = _mm512_fmadd_pd(_mm512_set1_pd(beta), _mm512_maskz_loadu_pd(mk, cj), out);
      _mm512_mask_storeu_pd(cj, mk, out);
    }
  }
}

/// C[i0:i1, 0:n] = alpha A[:, i0:i1]^T B + beta C (A k x m column-major, B k x n column-major).
template <int NB>
HS_BENCH_AVX512 void skinny_tn(int i0, int i1, int n, int k, double alpha, const double* a, int lda,
                               const double* b, int ldb, double beta, double* c, int ldc) {
  for (int i = i0; i < i1; ++i) {
    const double* ai = a + static_cast<std::size_t>(i) * lda;
    __m512d acc[NB];
    for (int j = 0; j < NB; ++j) acc[j] = _mm512_setzero_pd();
    for (int l = 0; l < k; l += 8) {
      const int len = std::min(8, k - l);
      const __mmask8 mk =
          len == 8 ? static_cast<__mmask8>(0xFF) : static_cast<__mmask8>((1u << len) - 1u);
      const __m512d av = _mm512_maskz_loadu_pd(mk, ai + l);
      for (int j = 0; j < NB; ++j)
        if (j < n)
          acc[j] = _mm512_fmadd_pd(
              av, _mm512_maskz_loadu_pd(mk, b + l + static_cast<std::size_t>(j) * ldb), acc[j]);
    }
    for (int j = 0; j < NB; ++j) {
      if (j >= n) break;
      double& cij = c[i + static_cast<std::size_t>(j) * ldc];
      const double v = alpha * _mm512_reduce_add_pd(acc[j]);
      cij = beta != 0.0 ? v + beta * cij : v;
    }
  }
}

template <int NB>
void skinny_dispatch(bool trans, int m, int n, int k, double alpha, const double* a, int lda,
                     const double* b, int ldb, double beta, double* c, int ldc) {
  const bool par =
      !omp_in_parallel() && static_cast<double>(m) * k >= 32768.0 && omp_get_max_threads() > 1;
  if (!trans) {
    std::vector<double> bt(static_cast<std::size_t>(k) * n);
    for (int l = 0; l < k; ++l)
      for (int j = 0; j < n; ++j)
        bt[static_cast<std::size_t>(l) * n + j] = b[l + static_cast<std::size_t>(j) * ldb];
    const int chunk = 64, nchunks = (m + chunk - 1) / chunk;
#pragma omp parallel for schedule(static) if (par)
    for (int q = 0; q < nchunks; ++q)
      skinny_nn<NB>(q * chunk, std::min(m, (q + 1) * chunk), n, k, alpha, a, lda, bt.data(), beta,
                    c, ldc);
  } else {
    const int chunk = 16, nchunks = (m + chunk - 1) / chunk;
#pragma omp parallel for schedule(static) if (par)
    for (int q = 0; q < nchunks; ++q)
      skinny_tn<NB>(q * chunk, std::min(m, (q + 1) * chunk), n, k, alpha, a, lda, b, ldb, beta, c,
                    ldc);
  }
}

/// Returns false (caller uses the real dgemm) for shapes this does not handle.
bool skinny(char ta, char tb, int m, int n, int k, double alpha, const double* a, int lda,
            const double* b, int ldb, double beta, double* c, int ldc) {
  if (!skinny_gemm.load(std::memory_order_relaxed) || n > skinny_max_n || n < 1 || m < 1 || k < 1)
    return false;
  if (!(tb == 'N' || tb == 'n')) return false;
  const bool trans = !(ta == 'N' || ta == 'n');
  if (n <= 4) {
    skinny_dispatch<4>(trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (n <= 8) {
    skinny_dispatch<8>(trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    skinny_dispatch<16>(trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  return true;
}

}  // namespace probe

extern "C" {

void dtrsm_(const char* side, const char* uplo, const char* ta, const char* diag, const int* m,
            const int* n, const double* alpha, const double* a, const int* lda, double* b,
            const int* ldb, std::size_t, std::size_t, std::size_t, std::size_t) {
  const double mm = *m;
  probe::Scope scope(probe::kTrsm, 4.0 * mm * mm + 8.0 * mm * *n);
  if (scope.on && !omp_in_parallel()) {
    std::string key{*side, ' ', *uplo, ' ', *ta, ' ', *diag};
    auto& e = probe::trsm_shapes[key];
    ++e.first;
    e.second += 4.0 * mm * mm;
  }
  if (probe::parallel_trsm.load(std::memory_order_relaxed) && !omp_in_parallel() && *side == 'L' &&
      *m >= probe::trsm_min_m && *n <= probe::trsm_max_n && omp_get_max_threads() > 1) {
    probe::blocked_parallel_trsm(*uplo, *ta, *diag, *m, *n, *alpha, a, *lda, b, *ldb);
    return;
  }
  probe::real_trsm(*side, *uplo, *ta, *diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

void dgemv_(const char* t, const int* m, const int* n, const double* alpha, const double* a,
            const int* lda, const double* x, const int* incx, const double* beta, double* y,
            const int* incy, std::size_t) {
  probe::Scope scope(probe::kGemv, 8.0 * static_cast<double>(*m) * *n);
  probe::real_gemv(*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

void dgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
            const double* beta, double* c, const int* ldc, std::size_t, std::size_t) {
  probe::Scope scope(probe::kGemm, 8.0 * static_cast<double>(*m) * *k);
  if (probe::skinny(*ta, *tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc)) return;
  if (probe::gemm_as_gemv.load(std::memory_order_relaxed) && *n == 1) {
    // C(m x 1) = alpha op(A) B + beta C as one dgemv (B is a column, or a row when tb = 'T').
    const char t = (*ta == 'N' || *ta == 'n') ? 'N' : 'T';
    const int rows = t == 'N' ? *m : *k, cols = t == 'N' ? *k : *m;
    const int incx = (*tb == 'N' || *tb == 'n') ? 1 : *ldb;
    probe::real_gemv(t, rows, cols, *alpha, a, *lda, b, incx, *beta, c, 1);
    return;
  }
  probe::real_gemm(*ta, *tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

void dcopy_(const int* n, const double* x, const int* incx, double* y, const int* incy) {
  probe::Scope scope(probe::kCopy, 8.0 * *n);
  probe::cblas_dcopy(*n, x, *incx, y, *incy);
}

}  // extern "C"

// ------------------------------------------------------------------------------------- bench

namespace {

using hs::cplx;
using hs::CsrMatrix;
using hs::index_t;
using json = nlohmann::json;

struct Args {
  std::filesystem::path dir;
  std::string exp = "split";
  std::string ordering = "scotch";
  bool deterministic = false;
  std::vector<index_t> p = {1, 2, 4, 8, 16};
  std::vector<index_t> icntl27 = {-32, 1, 2, 4, 8, 16};
  std::vector<index_t> solve_threads = {1, 2, 4};
  int reps = 5;
};

std::vector<index_t> parse_list(const std::string& s) {
  std::vector<index_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::stoll(item));
  return out;
}

Args parse_args(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("usage: bench_solvephase_mumps CASE_DIR [options]");
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return argv[++i];
    };
    if (s == "--exp")
      a.exp = value();
    else if (s == "--ordering")
      a.ordering = value();
    else if (s == "--det")
      a.deterministic = std::stoi(value()) != 0;
    else if (s == "--p")
      a.p = parse_list(value());
    else if (s == "--icntl27")
      a.icntl27 = parse_list(value());
    else if (s == "--solve-threads")
      a.solve_threads = parse_list(value());
    else if (s == "--reps")
      a.reps = std::stoi(value());
    else if (s == "--nb")
      probe::trsm_nb = std::stoi(value());
    else if (s == "--min-m")
      probe::trsm_min_m = std::stoi(value());
    else if (s == "--max-n")
      probe::trsm_max_n = std::stoi(value());
    else
      throw std::runtime_error("unknown option " + s);
  }
  return a;
}

json read_json(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + p.string());
  return json::parse(in);
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

/// One dmumps instance (SYM=2, sequential) with the library's controls.
class Mumps {
 public:
  explicit Mumps(const Args& args) {
    s_.sym = 2;
    s_.par = 1;
    s_.comm_fortran = -987654;
    s_.job = -1;
    dmumps_c(&s_);
    icntl(1) = -1;
    icntl(2) = -1;
    icntl(3) = -1;
    icntl(4) = 0;
    icntl(5) = 0;
    icntl(7) = args.ordering == "amd"     ? 0
               : args.ordering == "metis" ? 5
               : args.ordering == "auto"  ? 7
                                          : 3;
    icntl(8) = 7;
    icntl(13) = 1;
    icntl(14) = 35;
    icntl(16) = omp_get_max_threads();
    icntl(24) = 1;
    s_.cntl[2] = 1e-8;
    icntl(48) = args.deterministic ? 0 : 1;
  }
  ~Mumps() {
    s_.job = -2;
    dmumps_c(&s_);
  }
  int& icntl(int i) { return s_.icntl[i - 1]; }
  int infog(int i) const { return s_.infog[i - 1]; }
  DMUMPS_STRUC_C& raw() { return s_; }
  void run(int job) {
    s_.job = job;
    dmumps_c(&s_);
    if (s_.infog[0] < 0)
      throw std::runtime_error("MUMPS job " + std::to_string(job) +
                               " failed: INFOG(1)=" + std::to_string(s_.infog[0]) +
                               " INFOG(2)=" + std::to_string(s_.infog[1]));
  }
  void solve(double* rhs, int nrhs, int lrhs) {
    s_.rhs = rhs;
    s_.nrhs = nrhs;
    s_.lrhs = lrhs;
    run(3);
    s_.rhs = nullptr;
  }

 private:
  DMUMPS_STRUC_C s_{};
};

json slot_json(const probe::Slot& t, double wall) {
  static const char* names[] = {"trsm", "gemv", "gemm", "copy"};
  json j;
  double seq = 0.0;
  for (int k = 0; k < probe::kKinds; ++k) {
    j[std::string("seq_") + names[k] + "_ms"] = 1e3 * t.seconds[k][0];
    j[std::string("par_") + names[k] + "_ms"] = 1e3 * t.seconds[k][1];
    j[std::string("seq_") + names[k] + "_mb"] = t.bytes[k][0] / 1e6;
    j[std::string("par_") + names[k] + "_mb"] = t.bytes[k][1] / 1e6;
    j[std::string("seq_") + names[k] + "_calls"] = t.calls[k][0];
    j[std::string("par_") + names[k] + "_calls"] = t.calls[k][1];
    seq += t.seconds[k][0];
  }
  j["other_ms"] = 1e3 * (wall - seq);
  return j;
}

template <class T>
void run(const Args& args, const std::string& name, const hs::ShiftedPencil<T>& pencil,
         double sigma0) {
  hs::mumps::Input<T, double> input(pencil);
  Mumps m(args);
  auto& s = m.raw();
  input.assemble(1.0, -sigma0);
  s.n = input.order();
  s.nz = 0;
  s.nnz = input.nnz();
  s.irn = input.irn();
  s.jcn = input.jcn();
  s.a = input.values();
  hs::Timer ta;
  m.run(1);
  const double analyze_s = ta.seconds();
  double sigma = sigma0;
  hs::Timer tf;
  for (int r = 0;; ++r) {
    input.assemble(1.0, -sigma);
    m.run(2);
    if (m.infog(28) == 0) break;
    if (r == 3) throw std::runtime_error("no regular shift near sigma");
    sigma = sigma0 + 1e-8 * std::max(1.0, std::abs(sigma0)) * std::pow(10.0, r);
  }
  const double factorize_s = tf.seconds();
  const index_t order = input.order();
  json setup = {{"kind", "setup"},
                {"case", name},
                {"order", order},
                {"threads", omp_get_max_threads()},
                {"ordering", args.ordering},
                {"deterministic", args.deterministic},
                {"analyze_s", analyze_s},
                {"factorize_s", factorize_s},
                {"factor_entries", hs::index_t(m.infog(9) >= 0 ? m.infog(9) : -1e6 * m.infog(9))},
                {"sigma", sigma}};
  std::printf("%s\n", setup.dump().c_str());
  std::fflush(stdout);

  const index_t pmax = *std::max_element(args.p.begin(), args.p.end());
  std::vector<double> x(static_cast<std::size_t>(order * pmax)), y(x.size()), z(x.size());
  for (index_t j = 0; j < pmax; ++j)
    hs::linalg::fill_random(x.data() + j * order, order, 2026, static_cast<std::uint64_t>(j));

  auto timed_solve = [&](index_t p, bool record) {
    std::copy(x.begin(), x.begin() + order * p, y.begin());
    probe::reset();
    probe::recording = record;
    hs::Timer t;
    m.solve(y.data(), static_cast<int>(p), static_cast<int>(order));
    const double wall = t.seconds();
    probe::recording = false;
    return wall;
  };
  timed_solve(1, false);  // warm-up (first solve allocates MUMPS' solve workspace)

  if (args.exp == "split") {
    for (int rep = 0; rep < args.reps; ++rep) {
      for (index_t p : args.p) {
        const double plain = timed_solve(p, false);
        probe::trsm_shapes.clear();
        const double wall = timed_solve(p, true);
        json rec = slot_json(probe::total(), wall);
        rec.update({{"kind", "split"},
                    {"case", name},
                    {"rep", rep},
                    {"p", p},
                    {"plain_ms", 1e3 * plain},
                    {"recorded_ms", 1e3 * wall}});
        json shapes;
        for (const auto& [k, v] : probe::trsm_shapes) shapes[k] = {v.first, v.second / 1e6};
        rec["seq_trsm_shapes"] = shapes;
        std::printf("%s\n", rec.dump().c_str());
        std::fflush(stdout);
      }
    }
  } else if (args.exp == "icntl21") {
    // Distributed solution (ICNTL(21) = 1: SOL_LOC / ISOL_LOC, scattered back here) against the
    // centralized one overwriting the right-hand side (ICNTL(21) = 0, the library's setting).
    std::map<index_t, std::vector<double>> central, distributed;
    std::map<index_t, double> diff;
    std::vector<double> sol(static_cast<std::size_t>(order * pmax));
    std::vector<int> isol(static_cast<std::size_t>(order));
    for (int rep = 0; rep < args.reps; ++rep) {
      for (index_t p : args.p) {
        m.icntl(21) = 0;
        central[p].push_back(1e3 * timed_solve(p, false));
        std::copy(y.begin(), y.begin() + order * p, z.begin());
        m.icntl(21) = 1;
        auto& s = m.raw();
        s.sol_loc = sol.data();
        s.lsol_loc = static_cast<int>(order);
        s.isol_loc = isol.data();
        std::copy(x.begin(), x.begin() + order * p, y.begin());
        hs::Timer t;
        m.solve(y.data(), static_cast<int>(p), static_cast<int>(order));
        for (index_t k = 0; k < p; ++k)
          for (index_t i = 0; i < order; ++i) y[isol[i] - 1 + k * order] = sol[i + k * order];
        distributed[p].push_back(1e3 * t.seconds());
        m.icntl(21) = 0;
        s.sol_loc = nullptr;
        s.isol_loc = nullptr;
        double num = 0.0, den = 0.0;
        for (index_t i = 0; i < order * p; ++i) {
          num = std::max(num, std::abs(y[i] - z[i]));
          den = std::max(den, std::abs(z[i]));
        }
        diff[p] = std::max(diff[p], num / den);
      }
    }
    for (index_t p : args.p)
      std::printf("%s\n", json({{"kind", "icntl21"},
                                {"case", name},
                                {"p", p},
                                {"threads", omp_get_max_threads()},
                                {"central_ms", median(central[p])},
                                {"distributed_ms", median(distributed[p])},
                                {"speedup", median(central[p]) / median(distributed[p])},
                                {"max_rel_diff", diff[p]}})
                              .dump()
                              .c_str());
  } else if (args.exp == "icntl27") {
    // Every value of --icntl27 for every p of --p, interleaved (0 in the list = "p itself").
    std::map<std::pair<index_t, index_t>, std::vector<double>> per;
    for (int rep = 0; rep < args.reps; ++rep) {
      for (index_t p : args.p) {
        for (index_t v : args.icntl27) {
          m.icntl(27) = static_cast<int>(v == 0 ? p : v);
          per[{p, v}].push_back(1e3 * timed_solve(p, false));
        }
      }
    }
    m.icntl(27) = -32;
    for (index_t p : args.p)
      for (index_t v : args.icntl27)
        std::printf("%s\n", json({{"kind", "icntl27"},
                                  {"case", name},
                                  {"p", p},
                                  {"icntl27", v},
                                  {"solve_ms", median(per[{p, v}])},
                                  {"ms_per_vec", median(per[{p, v}]) / p}})
                                .dump()
                                .c_str());
  } else if (args.exp == "partrsm" || args.exp == "gemvfwd" || args.exp == "skinny") {
    // A/B on the same factorization: plain solves vs solves with the BLAS substitution on.
    std::atomic<bool>& flag = args.exp == "partrsm"   ? probe::parallel_trsm
                              : args.exp == "gemvfwd" ? probe::gemm_as_gemv
                                                      : probe::skinny_gemm;
    std::map<index_t, std::vector<double>> plain, alt;
    std::map<index_t, double> diff;
    for (int rep = 0; rep < args.reps; ++rep) {
      for (index_t p : args.p) {
        flag = false;
        plain[p].push_back(1e3 * timed_solve(p, false));
        std::copy(y.begin(), y.begin() + order * p, z.begin());
        flag = true;
        alt[p].push_back(1e3 * timed_solve(p, false));
        flag = false;
        double num = 0.0, den = 0.0;
        for (index_t i = 0; i < order * p; ++i) {
          num = std::max(num, std::abs(y[i] - z[i]));
          den = std::max(den, std::abs(z[i]));
        }
        diff[p] = std::max(diff[p], num / den);
      }
    }
    for (index_t p : args.p)
      std::printf("%s\n", json({{"kind", args.exp},
                                {"case", name},
                                {"p", p},
                                {"threads", omp_get_max_threads()},
                                {"nb", probe::trsm_nb},
                                {"plain_ms", median(plain[p])},
                                {"alt_ms", median(alt[p])},
                                {"speedup", median(plain[p]) / median(alt[p])},
                                {"max_rel_diff", diff[p]}})
                              .dump()
                              .c_str());
  } else if (args.exp == "solvethreads") {
    // ICNTL(16) changed between the factorization (all threads) and the solves.
    const int all = omp_get_max_threads();
    std::map<std::pair<index_t, index_t>, std::vector<double>> per;
    std::map<std::pair<index_t, index_t>, double> diff;
    std::map<std::pair<index_t, index_t>, std::string> failed;
    for (int rep = 0; rep < args.reps; ++rep) {
      for (index_t p : args.p) {
        m.icntl(16) = all;
        timed_solve(p, false);
        std::copy(y.begin(), y.begin() + order * p, z.begin());
        for (index_t t : args.solve_threads) {
          const auto key = std::make_pair(p, t);
          m.icntl(16) = static_cast<int>(t);
          try {
            per[key].push_back(1e3 * timed_solve(p, false));
          } catch (const std::exception& e) {
            failed[key] = e.what();
            continue;
          }
          double num = 0.0, den = 0.0;
          for (index_t i = 0; i < order * p; ++i) {
            num = std::max(num, std::abs(y[i] - z[i]));
            den = std::max(den, std::abs(z[i]));
          }
          diff[key] = std::max(diff[key], num / den);
        }
      }
    }
    m.icntl(16) = all;
    for (index_t p : args.p)
      for (index_t t : args.solve_threads) {
        const auto key = std::make_pair(p, t);
        json rec = {{"kind", "solvethreads"},
                    {"case", name},
                    {"p", p},
                    {"factor_threads", all},
                    {"solve_threads", t}};
        if (failed.count(key)) {
          rec["error"] = failed[key];
        } else {
          rec.update({{"solve_ms", median(per[key])},
                      {"ms_per_vec", median(per[key]) / p},
                      {"max_rel_diff", diff[key]}});
        }
        std::printf("%s\n", rec.dump().c_str());
      }
  } else {
    throw std::runtime_error("unknown --exp " + args.exp);
  }
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto cfg = read_json(args.dir / "config.json");
    const double sigma = cfg.at("target").at("sigma").get<double>();
    const std::string name = args.dir.filename().string();
    hs::LoadedMatrix la = hs::load_matrix(args.dir / "A.npz");
    std::optional<hs::LoadedMatrix> lb;
    if (std::filesystem::exists(args.dir / "B.npz")) lb = hs::load_matrix(args.dir / "B.npz");
    if (la.is_complex() || (lb && lb->is_complex())) {
      auto as_complex = [](hs::LoadedMatrix& mat) {
        return mat.is_complex() ? std::get<CsrMatrix<cplx>>(std::move(mat.matrix))
                                : hs::to_complex(std::get<CsrMatrix<double>>(mat.matrix));
      };
      const CsrMatrix<cplx> a = as_complex(la);
      std::optional<CsrMatrix<cplx>> b;
      if (lb) b = as_complex(*lb);
      hs::ShiftedPencil<cplx> pencil(a, b ? &*b : nullptr);
      run<cplx>(args, name, pencil, sigma);
    } else {
      const auto& a = std::get<CsrMatrix<double>>(la.matrix);
      const CsrMatrix<double>* b = lb ? &std::get<CsrMatrix<double>>(lb->matrix) : nullptr;
      hs::ShiftedPencil<double> pencil(a, b);
      run<double>(args, name, pencil, sigma);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_solvephase_mumps: %s\n", e.what());
    return 1;
  }
  return 0;
}
