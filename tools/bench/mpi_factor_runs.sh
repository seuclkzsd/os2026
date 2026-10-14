#!/usr/bin/env bash
# P9 "mpi": run bench/mpi_factor_bench.c (built by tools/bench/mpi_build.sh) for several MPI-rank x
# OpenMP-thread layouts of the 16 slot cores, interleaved (the layout order rotates per
# repetition), each run under its own nested guard (the caller's guard holds the slot):
#   seq1x16   build/mpi/seq_factor_bench, 16 threads, unbound like hsolve (sequential MUMPS of the
#             os2026 env = hsolve's path)
#   RxT       build/mpi/mpi_factor_bench under the ref env's mpiexec, R ranks x T threads, each rank
#             pinned to its block of the slot by tools/bench/mpi_bind.sh, OMP_PLACES=cores,
#             OMP_PROC_BIND=close inside it (1x16 = the MPI library with one rank)
#   ...h      (suffix, e.g. seq1x16h, 4x4h) the same with hsolve's BLAS hook preloaded
#             (build/release/libhsolve_blashook.so, route all - what the hsolve executable links;
#             it changes the solves, not the factorization, docs/BLAS.md)
# Usage (from the work tree on the node, one command line):
#   env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- \
#     bash tools/bench/mpi_factor_runs.sh OUT TAG MATRIX ORDER REPS LAYOUTS [BENCH ARGS...]
# ORDER: perm:FILE (PERM_IN), ord:N (ICNTL(7) = N, sequential analysis), par:N (ICNTL(28) = 2 with
# ICNTL(29) = N; MPI layouts only). LAYOUTS: comma list, e.g. seq1x16,1x16,2x8,4x4,8x2,16x1.
# BENCH ARGS go to every run (e.g. --modes count,full --nrhs 16 --expect-neg 20003).
# HS_RUN_MEM_GB (default 32) is each run's RSS cap (nested guard --mem-gb; the lane allows <= 12).
# HS_REP0 (default 0) numbers the repetitions from there, so that one campaign can be split into
# several guard calls (each <= 45 min) and keep its rotation: REPS=1 with HS_REP0=0, 1, 2.
# Output: OUT/TAG.jsonl, the bench's JSON lines with "layout", "rep", "order" added, and one
# {"event":"guard",...} line per run (exit status, wall time, process-tree peak RSS).
set -u
out=$1 tag=$2 matrix=$3 order=$4 reps=$5 layouts=$6
shift 6
extra=("$@")
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ref="${HS_REF_ENV:?HS_REF_ENV is not set (env/hosts)}"
mem="${HS_RUN_MEM_GB:-32}"
mkdir -p "$out"
f="$out/$tag.jsonl"
case "$order" in
  perm:*) oargs=(--perm "${order#perm:}") ;;
  ord:*) oargs=(--ordering "${order#ord:}") ;;
  par:*) oargs=(--par-analysis "${order#par:}") ;;
  *) echo "mpi_factor_runs.sh: ORDER must be perm:FILE, ord:N or par:N" >&2; exit 2 ;;
esac
IFS=, read -r -a lay <<< "$layouts"
nl=${#lay[@]}
annotate() {  # add layout / rep / order to the JSON lines on stdin
  python3 -c 'import json, sys
lay, rep, order = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in sys.stdin:
    line = line.strip()
    if line.startswith("{"):
        d = json.loads(line); d.update(layout=lay, rep=rep, order=order); print(json.dumps(d), flush=True)' "$@"
}
r0="${HS_REP0:-0}"
for ((r = r0; r < r0 + reps; r++)); do
  for ((i = 0; i < nl; i++)); do
    l=${lay[$(((i + r) % nl))]}
    gsum="$out/$tag.guard.$l.r$r.json"
    base=${l%h} preload=()
    [ "$base" != "$l" ] && preload=(LD_PRELOAD="$root/build/release/libhsolve_blashook.so")
    if [ "$base" = seq1x16 ]; then
      cmd=(env ${preload[@]+"${preload[@]}"} OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 "$root/build/mpi/seq_factor_bench"
           --threads 16)
    else
      ranks=${base%x*} threads=${base#*x}
      cmd=("$ref/bin/mpiexec" -n "$ranks" --bind-to none -x OMP_NUM_THREADS="$threads"
           -x OPENBLAS_NUM_THREADS="$threads" -x OMP_PLACES=cores -x OMP_PROC_BIND=close)
      [ ${#preload[@]} -gt 0 ] && cmd+=(-x "${preload[0]}")
      cmd+=("$root/tools/bench/mpi_bind.sh" "$root/build/mpi/mpi_factor_bench" --threads "$threads")
    fi
    echo "== $tag $l rep $r $(date +%T)" >&2
    python3 "$root/tools/run/guard.py" --timeout 1500 --mem-gb "$mem" --summary "$gsum" -- \
      "${cmd[@]}" --matrix "$matrix" "${oargs[@]}" --label "$tag" ${extra[@]+"${extra[@]}"} | annotate "$l" "$r" "$order" >> "$f"
    python3 -c 'import json, sys
g = json.load(open(sys.argv[1]))
print(json.dumps({"event": "guard", "layout": sys.argv[2], "rep": int(sys.argv[3]), "order": sys.argv[4],
                  "returncode": g.get("returncode"), "killed": g.get("killed"), "wall_s": g.get("wall_s"),
                  "peak_rss_gib": g.get("peak_rss_gib")}))' "$gsum" "$l" "$r" "$order" >> "$f"
  done
done
echo "done $tag $(date +%T)" >&2
