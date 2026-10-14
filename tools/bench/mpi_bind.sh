#!/usr/bin/env bash
# Pin this MPI rank to its block of the CPUs it inherited, then exec the command (P9, used by
# tools/bench/mpi_factor_runs.sh as `mpiexec -n R --bind-to none tools/bench/mpi_bind.sh CMD...`).
#
# The inherited CPUs are the guard's cpuset (the measurement slot, 16-31): rank r of R takes the r-th
# of R equal contiguous blocks, like tools/bench/baseline_slepc.py --bind core (mpiexec's own
# --bind-to core binds outside the guard's cpuset, docs/DESIGN.md section 6). The OpenMP runtime of
# the command starts after the pinning, so OMP_PLACES=cores / OMP_PROC_BIND=close (set by the
# caller) place each rank's threads inside its block. Rank and size come from Open MPI / PMIx.
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}
size=${OMPI_COMM_WORLD_SIZE:-1}
list=$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)
cpus=()
IFS=, read -r -a parts <<< "$list"
for p in "${parts[@]}"; do
  if [[ $p == *-* ]]; then
    for ((c = ${p%-*}; c <= ${p#*-}; c++)); do cpus+=("$c"); done
  else
    cpus+=("$p")
  fi
done
n=${#cpus[@]}
if (( size < 1 || rank >= size || n % size != 0 )); then
  echo "mpi_bind.sh: cannot split CPUs $list into $size equal blocks (rank $rank)" >&2
  exit 2
fi
per=$((n / size))
block=$(IFS=,; echo "${cpus[*]:rank*per:per}")
exec taskset -c "$block" "$@"
