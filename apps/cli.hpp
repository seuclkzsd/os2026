// The hsolve command line (docs/FORMATS.md), shared by the two executables:
//   hsolve      run(argc, argv, nullptr): one process, hs::solve;
//   hsolve-mpi  run(argc, argv, &transport) on every MPI rank (apps/hsolve_mpi.cpp): rank 0 parses,
//               loads and solves with hs::solve_parallel, the other ranks serve it (hs::serve);
//               the arguments are the same on every rank (DESIGN §3.9, docs/PARALLEL.md).
#pragma once

namespace hs {
class Transport;
}

namespace hs::cli {

/// Exit status: 0 = all pairs accepted (and certificate passed), 1 = finished but not successful,
/// 2 = error (message on stderr). With a transport, rank 0 releases the helpers on every path
/// (helpers return 0 once released).
int run(int argc, char** argv, Transport* peers);

}  // namespace hs::cli
