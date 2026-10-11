// BLAS call profiler of the BLAS hook (module blas; docs/BLAS.md). Scalar set: none (records).
//
// Enabled when the environment variable HS_BLAS_PROFILE names an output file ("%p" in it is
// replaced by the process id); disabled, a record costs one load of a cached flag. Each call is
// accumulated under (routine, operand flags, size classes of m / n / k, phase, inside an OpenMP
// parallel region, route) with its count, seconds, flop count and the bytes it must move at least
// (every operand read once, outputs read and written once). Phases are set by the MUMPS
// entry-point wrappers (hook_symbols.cpp): analysis (JOB 1), factorization (JOB 2), count (JOB 2
// with ICNTL(31) = 1: the inertia-only factorizations of the certificate and of compute_inertia),
// solve1 / solvep (JOB 3 with NRHS = 1 / > 1), other MUMPS jobs; calls outside MUMPS are phase
// none. Seconds of calls inside parallel regions are summed over threads (thread-seconds); those
// outside are wall seconds of the calling thread (the BLAS may use more threads inside).
// At process exit the tables of all threads are merged and written as one JSON document (format
// in docs/BLAS.md); bench/asmkern_report.cpp turns such files into tables.
//
// Size class of a dimension x: 0 for x <= 0, else 1 + floor(log2 x), i.e. class c holds
// [2^(c-1), 2^c - 1] (1 -> 1, 2..3 -> 2, 4..7 -> 3, 16..31 -> 5, ...).
//
// Thread safety: record() and the phase calls may run concurrently on any threads; each thread
// owns its table (registered once under a mutex), merged only when dumping.
#pragma once

#include <cstdint>
#include <string>

namespace hs::blas::prof {

enum class Op : std::uint8_t {
  Dgemm, Dtrsm, Dgemv, Daxpy, Dcopy, Dscal, Dswap, Dnrm2, Idamax,
  kCount
};

enum class Phase : std::uint8_t { None, Analysis, Factorization, Count, Solve1, SolveP, OtherMumps,
                                  kCount };

const char* op_name(Op op) noexcept;
const char* phase_name(Phase p) noexcept;
int size_class(std::int64_t x) noexcept;

/// True when HS_BLAS_PROFILE is set (read once).
bool enabled() noexcept;

void set_phase(Phase p) noexcept;
Phase phase() noexcept;

/// Operand flags: gemm bit0 = op(A) transposed, bit1 = op(B) transposed; trsm bit0 side R,
/// bit1 uplo L, bit2 transposed, bit3 unit diagonal; gemv bit0 transposed.
std::uint8_t gemm_flags(char ta, char tb) noexcept;
std::uint8_t trsm_flags(char side, char uplo, char ta, char diag) noexcept;
std::uint8_t gemv_flags(char t) noexcept;

/// Monotonic seconds (clock_gettime).
double now() noexcept;

/// Accumulates one call. `route` is hs::blas::Route as an integer (0 = real routine).
void record(Op op, std::uint8_t flags, std::int64_t m, std::int64_t n, std::int64_t k,
            double flops, double bytes, double seconds, std::uint8_t route) noexcept;

/// Accumulates one MUMPS call of the given phase (wall seconds, right-hand sides for solves).
void record_mumps(Phase p, double seconds, std::int64_t nrhs) noexcept;

/// Free-form provenance written into the JSON header ("route", "blas"), set once at start-up.
void set_info(const std::string& route, const std::string& blas_library);

/// Writes the merged tables now (also done automatically at exit; later calls rewrite the file).
void dump() noexcept;

}  // namespace hs::blas::prof
