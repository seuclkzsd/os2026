#!/usr/bin/env bash
# P10 "mpi", review fix (results/p10_mpi.md sections 3.1 and 7): plain hsolve against its
# multi-process mode where the review found it wrong. Run from the repository root on a cluster
# node (tools/dev/cluster_run.sh), detached; each run is its own guard call.
#
#   tools/bench/mpi_review_runs.sh tight  OUT   two sockets (--cores 8-15,16-23, guard --exclusive):
#        hsolve (16 threads) vs hsolve-mpirun --ranks 3 at the same explicit --memory-budget-gb,
#        5 interleaved repetitions: syn-mem27-46 nearest at 1.0 and 1.1 GiB, yee-pec-box-1e5
#        [40, 400] at 2.5 GiB (section 7.1). The lane cores 8-15 must be idle meanwhile.
#   tools/bench/mpi_review_runs.sh launch OUT   one socket (--cores 16-31, guard --exclusive): hsolve
#        vs hsolve-mpirun without --ranks (one NUMA node: the plain hsolve), syn-mem27-46 and
#        cav2d-rect-pert-1e5, 5 interleaved repetitions; compare the guards' wall times (section 7.3).
#   tools/bench/mpi_review_runs.sh calib  OUT   build/test lane (guard --skip-busy-check): one
#        hsolve-mpirun --ranks 3 run per case with --helper-counts on --helper-slices on and a 10 GiB
#        budget, for the helpers_ready / helpers_drop events of the memory predictions (section 3.1).
#
# Output: OUT/<tag>-<mode>-r<rep>/ (result), .guard.json (guard summary), .log (+ .rank<r>: event
# logs), .stdout. Data: $HS_DATA_ROOT or data/. Tabulate with the reports and guard summaries.
set -u
mode="${1:-}" out="${2:-}"
[ -n "$mode" ] && [ -n "$out" ] || { sed -n '2,20p' "$0" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source env/activate.sh
D="${HS_DATA_ROOT:-$root/data}"
mkdir -p "$out"

run() {  # placement tag case mode rep [hsolve args]
  local place=$1 tag=$2 c=$3 m=$4 r=$5
  shift 5
  local o="$out/$tag-$m-r$r"
  rm -rf "$o" "$o".log*
  local B=()
  [ -f "$D/cases/$c/B.npz" ] && B=(--B "$D/cases/$c/B.npz")
  local g cmd
  case "$place" in
    two) g=(--exclusive --cores 8-15,16-23 --mem-gb 32 --timeout 900 --lock-wait 900) ;;
    one) g=(--exclusive --cores 16-31 --mem-gb 32 --timeout 600 --lock-wait 900) ;;
    lane) g=(--skip-busy-check --mem-gb 11 --timeout 900) ;;
  esac
  case "$m" in
    plain) cmd=(build/release/hsolve) ;;
    mpi) cmd=(build/release/hsolve-mpirun --ranks 3 --) ;;
    launcher) cmd=(build/release/hsolve-mpirun --) ;;
  esac
  [ "$place" = two ] && [ "$m" = plain ] && cmd+=(--threads 16)
  python3 tools/run/guard.py "${g[@]}" --summary "$o.guard.json" -- "${cmd[@]}" --A "$D/cases/$c/A.npz" \
    ${B[@]+"${B[@]}"} --config "$D/cases/$c/config.json" --out "$o" --log "$o.log" -q "$@" > "$o.stdout" 2>&1
  echo "rc=$? $tag $m r$r"
}

case "$mode" in
  tight)
    for r in 0 1 2 3 4; do
      if [ $((r % 2)) = 0 ]; then m1=plain m2=mpi; else m1=mpi m2=plain; fi
      run two m46b1 syn-mem27-46 $m1 $r --memory-budget-gb 1.0
      run two m46b1 syn-mem27-46 $m2 $r --memory-budget-gb 1.0
      run two m46b11 syn-mem27-46 $m2 $r --memory-budget-gb 1.1
      run two m46b11 syn-mem27-46 $m1 $r --memory-budget-gb 1.1
      run two ybb25 yee-pec-box-1e5 $m1 $r --memory-budget-gb 2.5 --interval 40 400
      run two ybb25 yee-pec-box-1e5 $m2 $r --memory-budget-gb 2.5 --interval 40 400
    done ;;
  launch)
    for r in 0 1 2 3 4; do
      if [ $((r % 2)) = 0 ]; then m1=plain m2=launcher; else m1=launcher m2=plain; fi
      for c in syn-mem27-46 cav2d-rect-pert-1e5; do
        run one "$c" "$c" $m1 $r
        run one "$c" "$c" $m2 $r
      done
    done ;;
  calib)
    on=(--memory-budget-gb 10 --helper-counts on --helper-slices on)
    run lane m46 syn-mem27-46 mpi 0 "${on[@]}"
    run lane ybb yee-pec-box-1e5 mpi 0 "${on[@]}"
    run lane qa8 ss-qa8fk-qa8fm mpi 0 "${on[@]}"
    run lane cav cav2d-rect-pert-1e5 mpi 0 "${on[@]}"
    run lane ybbiv yee-pec-box-1e5 mpi 0 "${on[@]}" --interval 40 400
    run lane m46iv syn-mem27-46 mpi 0 "${on[@]}" --interval 0.1 0.3
    run lane ycube yee-pec-cube-32 mpi 0 "${on[@]}" --interval 40 400 ;;
  *) echo "mpi_review_runs.sh: MODE must be tight, launch or calib" >&2; exit 2 ;;
esac
