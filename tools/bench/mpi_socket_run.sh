#!/usr/bin/env bash
# P10 "mpi": ONE run of a process / thread layout on the cores this script inherits - the cpuset of
# the caller's guard (`guard.py --exclusive --cores C`; C = 16-31: one socket, 8-15,16-23: 8 cores of
# each socket on the cluster nodes). tools/bench/mpi_layouts.py runs it once per (case, layout,
# repetition), each under its own guard; results/p10_mpi.md has the commands.
#
#   bash tools/bench/mpi_socket_run.sh LAYOUT KIND [ARGS...]
#
# LAYOUT (the inherited cores split into equal contiguous halves for the 8-thread layouts):
#   1x16    one process, T = all inherited cores (unbound inside them, as hsolve runs)
#   1x16i   the same under `numactl --interleave=all` (pages spread over the NUMA nodes)
#   1x8     one process on the first half (T = half); the other half idles
#   2x8     two processes at once, one on each half (T = half each), independent of each other
#   2x8m    ONE MPI job of two ranks (Open MPI of the os2026-ref env), rank r pinned to half r by
#           tools/bench/mpi_bind.sh (KIND bench: MPI MUMPS; KIND hsolve: build/release/hsolve-mpi)
#   1+2x8   KIND hsolve: hsolve's multi-process mode through its launcher
#           (build/release/hsolve-mpirun --ranks 3): the leader on all cores, a helper on each half
# KIND:
#   bench   build/mpi/seq_factor_bench (2x8m: build/mpi/mpi_factor_bench) with hsolve's BLAS hook
#           preloaded (build/release/libhsolve_blashook.so, route all: as the hsolve executable links
#           it; it changes the solves, not the factorization - docs/BLAS.md); ARGS go to every
#           process. 2x8 adds --sync DIR --peers 2 --peer I: both processes start each timed phase
#           together (bench/mpi_factor_bench.c), so their counts / factorizations / solves overlap.
#   hsolve  build/release/hsolve (2x8m: mpiexec ... build/release/hsolve-mpi), ARGS = the CLI
#           arguments without --threads (set here: T). 2x8 is not defined for hsolve (two
#           independent solves are no layout of one solve).
# Every process gets OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = T. The bench processes print JSON
# lines; this script prefixes each line of process I with {"proc":I,"cores":"...",...} merged in.
set -euo pipefail
layout=$1 kind=$2
shift 2
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
half=$((n / 2))
(( n >= 2 && n % 2 == 0 )) || { echo "mpi_socket_run.sh: $n inherited CPUs ($list) cannot be halved" >&2; exit 2; }
join() { local IFS=,; echo "$*"; }
all=$(join "${cpus[@]}")
h0=$(join "${cpus[@]:0:half}")
h1=$(join "${cpus[@]:half:half}")
hook="$root/build/release/libhsolve_blashook.so"
ref="${HS_REF_ENV:-}"
tag() {  # prefix JSON lines of process $1 on cores $2
  python3 -c 'import json, sys
proc, cores = int(sys.argv[1]), sys.argv[2]
for line in sys.stdin:
    line = line.strip()
    if line.startswith("{"):
        d = json.loads(line); d.update(proc=proc, cores=cores); print(json.dumps(d), flush=True)
    elif line:
        print(line, file=sys.stderr, flush=True)' "$1" "$2"
}
threads_env() { echo "OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1"; }
case "$kind" in
  bench) seq=("$root/build/mpi/seq_factor_bench") mpi=("$root/build/mpi/mpi_factor_bench") ;;
  hsolve) seq=("$root/build/release/hsolve") mpi=("$root/build/release/hsolve-mpi") ;;
  *) echo "mpi_socket_run.sh: KIND must be bench or hsolve" >&2; exit 2 ;;
esac
targs() { echo "--threads $1"; }  # both programs take --threads T
preload=()
[ "$kind" = bench ] && preload=(LD_PRELOAD="$hook")
case "$layout" in
  1x16|1x16i|1x8)
    cores=$all t=$n
    [ "$layout" = 1x8 ] && cores=$h0 t=$half
    pre=()
    [ "$layout" = 1x16i ] && pre=(numactl --interleave=all)
    # shellcheck disable=SC2046
    env ${preload[@]+"${preload[@]}"} $(threads_env "$t") taskset -c "$cores" ${pre[@]+"${pre[@]}"} \
      "${seq[@]}" "$@" $(targs "$t") | tag 0 "$cores"
    ;;
  2x8)
    [ "$kind" = bench ] || { echo "mpi_socket_run.sh: 2x8 is for KIND bench (independent processes)" >&2; exit 2; }
    sync=$(mktemp -d /tmp/hs-sync.XXXXXX)
    trap 'rm -rf "$sync"' EXIT
    out0=$(mktemp) out1=$(mktemp)
    # shellcheck disable=SC2046
    env "${preload[@]}" $(threads_env "$half") taskset -c "$h0" "${seq[@]}" "$@" $(targs "$half") \
      --sync "$sync" --peers 2 --peer 0 > "$out0" &
    p0=$!
    # shellcheck disable=SC2046
    env "${preload[@]}" $(threads_env "$half") taskset -c "$h1" "${seq[@]}" "$@" $(targs "$half") \
      --sync "$sync" --peers 2 --peer 1 > "$out1" &
    p1=$!
    rc=0
    wait "$p0" || rc=$?
    wait "$p1" || rc=$?
    tag 0 "$h0" < "$out0"
    tag 1 "$h1" < "$out1"
    rm -f "$out0" "$out1"
    exit "$rc"
    ;;
  2x8m)
    [ -n "$ref" ] || { echo "mpi_socket_run.sh: HS_REF_ENV is not set (env/hosts)" >&2; exit 2; }
    cmd=("$ref/bin/mpiexec" -n 2 --bind-to none -x OMP_NUM_THREADS="$half" -x OPENBLAS_NUM_THREADS="$half"
         -x OMP_PLACES=cores -x OMP_PROC_BIND=close)
    [ "$kind" = bench ] && cmd+=(-x LD_PRELOAD="$hook")
    # shellcheck disable=SC2046
    "${cmd[@]}" "$root/tools/bench/mpi_bind.sh" "${mpi[@]}" "$@" $(targs "$half") | tag 0 "$all"
    ;;
  1+2x8)
    # hsolve's multi-process mode as its launcher lays it out: the leader on all inherited cores, two
    # helpers on the halves (one per NUMA node when the cores span two).
    [ "$kind" = hsolve ] || { echo "mpi_socket_run.sh: 1+2x8 is for KIND hsolve" >&2; exit 2; }
    "$root/build/release/hsolve-mpirun" --ranks 3 -- "$@" | tag 0 "$all"
    ;;
  *) echo "mpi_socket_run.sh: unknown LAYOUT $layout (1x16, 1x16i, 1x8, 2x8, 2x8m, 1+2x8)" >&2; exit 2 ;;
esac
