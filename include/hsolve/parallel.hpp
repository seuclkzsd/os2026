// Multi-process solves (module src/parallel; DESIGN §3.9): one solve spread over several processes
// - MPI ranks of hsolve-mpi (src/mpi, optional build) - that each own a sequential MUMPS and their
// OpenMP threads. Process 0 (the leader) runs the solve exactly as hs::solve does; the others
// (helpers) only serve independent pieces of it that the leader hands them:
//   * the second Sylvester count of a nearest target's exact certificate, concurrently with the
//     leader's first (its own inertia-only factorization on the leader's pivot order);
//   * spectrum slices of interval / smallest / largest targets (and of nearest targets handed to
//     the windows): factorization at the slice's shift, Krylov run, verification, deflated reruns.
// What pays and why (one process per socket: each gets its own memory bandwidth) is measured in
// results/p10_mpi.md. The results equal those of the single-process path up to the usual
// tolerance: counts are exact integers either way, and a slice is solved by the same code on the
// same pivot order (only the thread count of its factorization may differ). Every helper task that
// fails - a MUMPS or memory error, an exception, a fault injected for tests - is reported back
// and the leader does the work itself (the helper is not used again in that solve); a helper
// PROCESS that dies ends the MPI job (MPI's default error handler), which hsolve cannot catch.
// Scalar set: T in {double, cplx}.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "hsolve/log.hpp"
#include "hsolve/solver.hpp"
#include "hsolve/sparse.hpp"
#include "hsolve/types.hpp"

namespace hs {

/// Point-to-point byte messages between the processes of one parallel session (ranks 0..size-1;
/// rank 0 is the leader). Messages between two processes arrive in the order they were sent.
/// Implementations: hs::mpi::MpiTransport (src/mpi, MPI_COMM_WORLD) and
/// make_thread_transports() (threads of one process, for tests). One object per process; not
/// thread-safe (each process uses its transport from one thread).
class Transport {
 public:
  static constexpr int kAnySource = -1;
  virtual ~Transport() = default;
  virtual int rank() const = 0;
  virtual int size() const = 0;
  /// Sends `bytes` to rank `to` (!= rank()); returns once the buffer may be reused.
  virtual void send(int to, const std::vector<std::uint8_t>& bytes) = 0;
  /// Blocks until a message from rank `from` (kAnySource: any rank) arrives and returns it;
  /// *source (optional) receives the sender. While blocked the process does not spin on a core
  /// for long (helpers wait through whole solves; MpiTransport polls with sleeps).
  virtual std::vector<std::uint8_t> receive(int from, int* source = nullptr) = 0;
  /// Whether a message from `from` (kAnySource: any) has arrived (does not block).
  virtual bool ready(int from) = 0;
};

/// n transports over in-process queues (rank i = element i), to be used from n threads: the
/// session logic without MPI (tests). MUMPS calls of all threads are serialized by the process-wide
/// MUMPS lock (DESIGN §3.4), so the helpers' work is correct but not concurrent.
std::vector<std::unique_ptr<Transport>> make_thread_transports(int n);

/// What a multi-process solve hands to its helpers (hsolve-mpi: --helper-counts, --helper-slices).
/// Both need at least two usable helpers and room in the memory budget for the work at once.
struct ParallelOptions {
  /// The nearest certificate's two exact counts on two helpers at once. Auto: when they pay - the
  /// main factorization measured in this solve is long enough against the helpers' preparation
  /// (DESIGN §3.9); On: whenever possible; Off: never (the leader counts).
  Toggle helper_counts = Toggle::Auto;
  /// The slices of window targets on the helpers. Auto: when one slice's predicted work (the
  /// slicing cost model) exceeds the helpers' preparation; On: whenever possible; Off: never.
  Toggle helper_slices = Toggle::Auto;
};

/// Rank 0 of `peers`: hs::solve(a, b, cfg, log) with the helpers 1..size-1, which must be in
/// serve(). With size() == 1 it is hs::solve itself. report.parallel describes the session
/// (DESIGN §3.9, docs/FORMATS.md §3). a and b are only read; the helpers get copies of the
/// matrices the solve factorizes (after its Hermitian check) when they first get work.
template <class T>
Solution<T> solve_parallel(const CsrMatrix<T>& a, const non_deduced_t<CsrMatrix<T>>* b, const SolverConfig& cfg,
                           Logger* log, Transport& peers, const ParallelOptions& options = {});

/// Ranks 1..size-1: serve the leader's tasks - any number of solves - until the leader calls
/// release_helpers(). log (optional): the helper's own events. Returns normally then; throws only
/// for a broken session (a message it cannot decode: Internal).
void serve(Transport& peers, Logger* log = nullptr);

/// Rank 0: ends serve() on every helper (after the last solve_parallel; idempotent per session).
void release_helpers(Transport& peers);

}  // namespace hs
