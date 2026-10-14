#!/usr/bin/env bash
# Build the P9 standalone MUMPS bench (bench/mpi_factor_bench.c) twice, outside the CMake build:
#   build/mpi/mpi_factor_bench   MPI MUMPS 5.8.2 of the os2026-ref env (Open MPI 5, ScaLAPACK,
#                                PT-SCOTCH, ParMETIS), through its mpicc
#   build/mpi/seq_factor_bench   the sequential MUMPS 5.8.2 hsolve links (os2026 env, libseq MPI
#                                stubs), i.e. hsolve's 1 x 16 path
# Both with the os2026 env's conda gcc (the ref env ships no compiler; OMPI_CC points its mpicc at
# it). Usage, on a cluster node from the work tree (the build/test lane):
#   tools/dev/cluster_run.sh c005 -- bash tools/bench/mpi_build.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source env/activate.sh
ref="${HS_REF_ENV:?HS_REF_ENV is not set (env/hosts)}"
seq="${HS_CONDA_ENV:?HS_CONDA_ENV is not set (env/hosts)}"
cc="$seq/bin/x86_64-conda-linux-gnu-gcc"
out="$root/build/mpi"
mkdir -p "$out"
flags=(-O2 -std=c11 -Wall -Wextra)
echo "MPI:  $("$ref/bin/mpiexec" --version | head -1); MUMPS $(ls "$ref"/conda-meta | grep -o '^mumps-mpi-[0-9.]*')"
echo "seq:  MUMPS $(ls "$seq"/conda-meta | grep -o '^mumps-seq-[0-9.]*')"
OMPI_CC="$cc" "$ref/bin/mpicc" "${flags[@]}" -I"$ref/include" bench/mpi_factor_bench.c \
  -o "$out/mpi_factor_bench" -L"$ref/lib" -Wl,-rpath,"$ref/lib" -Wl,-rpath-link,"$ref/lib" \
  -ldmumps -lmumps_common -lm
"$cc" "${flags[@]}" -DHS_SEQ_MUMPS -I"$seq/include" -I"$seq/include/mumps_seq" bench/mpi_factor_bench.c \
  -o "$out/seq_factor_bench" -L"$seq/lib" -Wl,-rpath,"$seq/lib" -Wl,-rpath-link,"$seq/lib" \
  -ldmumps_seq -lmumps_common_seq -lmpiseq_seq -lm
echo "built $out/mpi_factor_bench $out/seq_factor_bench"
