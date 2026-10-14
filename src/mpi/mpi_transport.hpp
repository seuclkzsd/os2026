// hs::Transport over MPI (module src/mpi, built only with -DHSOLVE_WITH_MPI=ON; DESIGN §3.9).
// The ranks of MPI_COMM_WORLD are the session's processes. hsolve itself links the SEQUENTIAL
// MUMPS: MPI here only carries the session's messages, MUMPS never sees it.
#pragma once

#include <cstdint>
#include <vector>

#include "hsolve/parallel.hpp"

namespace hs::mpi {

/// Needs MPI initialized (MPI_THREAD_FUNNELED or above: only the thread that drives the session
/// calls MPI) and not finalized for its lifetime. A message is a header (its size) and up to
/// kChunk-sized parts, so that any size passes MPI's int counts. Waiting does not spin on a core:
/// receive() polls MPI_Iprobe with sleeps growing from 50 us to 2 ms (helpers wait through whole
/// solves; the leader waits while helpers work), so a message is picked up within ~2 ms.
class MpiTransport final : public Transport {
 public:
  MpiTransport();
  int rank() const override { return rank_; }
  int size() const override { return size_; }
  void send(int to, const std::vector<std::uint8_t>& bytes) override;
  std::vector<std::uint8_t> receive(int from, int* source = nullptr) override;
  bool ready(int from) override;

  static constexpr std::size_t kChunk = std::size_t{1} << 30;

 private:
  int rank_ = 0, size_ = 1;
};

/// The MPI library this process resolved MPI_Init from (dladdr; "" if unknown): hsolve-mpi checks
/// that it is not the sequential MUMPS' libmpiseq stub, which defines MPI_Init too
/// (docs/PARALLEL.md: link order).
const char* mpi_library();

/// MPI_Init_thread(MPI_THREAD_FUNNELED) and MPI_Finalize, MPI_COMM_WORLD's size. Executables call
/// these instead of including <mpi.h>: on an include path with MUMPS' sequential headers its
/// mpi.h stub (libseq) shadows the real one. init() returns false when MPI_Init_thread failed.
bool init(int* argc, char*** argv);
void finalize();
int world_size();

}  // namespace hs::mpi
