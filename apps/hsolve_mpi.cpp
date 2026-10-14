// hsolve-mpi: the hsolve command line (apps/cli.cpp) as one MPI job (built with HSOLVE_WITH_MPI;
// DESIGN §3.9, docs/PARALLEL.md). Rank 0 is hsolve - it reads, solves and writes the results -
// and hands independent work (the nearest certificate's two counts, spectrum slices) to the other
// ranks, which serve it. The intended layout is the leader on all cores plus one helper per socket
// (hsolve-mpirun sets it up):
//
//   hsolve-mpirun [--ranks R] [--] --A A.npz [--B B.npz] --config config.json --out DIR ...
//   mpiexec -n R [binding] hsolve-mpi --A ... (the same arguments on every rank)
//
// With one rank it is hsolve. Exit status as hsolve (rank 0's; helpers exit 0).
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "hsolve/parallel.hpp"
#include "mpi/mpi_transport.hpp"

int main(int argc, char** argv) {
  if (!hs::mpi::init(&argc, &argv)) {
    std::fprintf(stderr, "hsolve-mpi: MPI_Init_thread failed\n");
    return 2;
  }
  // The sequential MUMPS hsolve links brings libmpiseq, whose stub MPI_Init would make every rank a
  // lone rank 0: the real MPI library must come first in the dynamic linker's lookup (CMake links
  // it ahead of MUMPS; docs/PARALLEL.md).
  const std::string lib = hs::mpi::mpi_library();
  if (lib.find("mpiseq") != std::string::npos) {
    std::fprintf(stderr, "hsolve-mpi: MPI_Init resolves to %s (the sequential MUMPS' stub), not to MPI\n",
                 lib.c_str());
    hs::mpi::finalize();
    return 2;
  }
  int rc = 2;
  {
    hs::mpi::MpiTransport peers;
    rc = hs::cli::run(argc, argv, &peers);
  }
  hs::mpi::finalize();
  return rc;
}
