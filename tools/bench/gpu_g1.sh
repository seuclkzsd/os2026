#!/usr/bin/env bash
# GPU experiment G1 (results/gpu_g1.md): bench_gpu_ssids per case and backend variant, repetitions
# interleaved, every timed process in the node's measurement slot (16 cores, 32 GiB).
#
#   tools/bench/gpu_g1.sh [--reps R] [--variants V,V...] [--out FILE] [--gpu IDX] [--lane]
#                         [--bench-args "ARGS"] CASE...
#
# CASE: a directory under data/cases, or a path. Variants (backend:ordering of bench/gpu_ssids.cpp;
# default all six, in an order rotated per repetition): mumps:driver mumps:metis ssids-cpu:metis
# ssids-cpu:mumps ssids-gpu:metis ssids-gpu:mumps. Each (repetition, case) is one
# `tools/run/guard.py --exclusive --timeout 900` call running all variants, one process each, so the
# slot is held at most 15 minutes at a time and other callers get it in between; the guard's exit
# status 75 (slot busy for HS_LOCK_WAIT_S) stops the script with 75. Environments: mumps as hsolve
# runs (CUDA_VISIBLE_DEVICES=""), ssids-cpu with OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE and no GPU,
# ssids-gpu the same on GPU IDX (default 7: gpu02's GPU on the slot's socket). Before each ssids-gpu
# process that GPU must be idle (no compute process, < 16 MiB used), else the script stops with
# status 3. --lane: the build/test lane instead of the slot (smoke runs, no timings). --bench-args:
# extra arguments for every process (word-split), e.g. "--factorizations 3".
# Output: one JSON line per process - the bench's line plus "rep" and "variant" ("failed": true if
# it exited non-zero) - appended to FILE (default results/gpu_g1.jsonl); stderr goes to FILE.log.
# Report: python3 tools/bench/gpu_g1_report.py FILE
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
self="$root/tools/bench/gpu_g1.sh"
reps=3
variants="mumps:driver,mumps:metis,ssids-cpu:metis,ssids-cpu:mumps,ssids-gpu:metis,ssids-gpu:mumps"
out="results/gpu_g1.jsonl"
gpu=7
lane=0
bench_args=""
batch_rep=""
cases=()
while [ $# -gt 0 ]; do
  case "$1" in
    --reps) reps="$2"; shift 2 ;;
    --variants) variants="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --lane) lane=1; shift ;;
    --bench-args) bench_args="$2"; shift 2 ;;
    --batch) batch_rep="$2"; shift 2 ;;  # internal: one (repetition, case) inside the guard
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) cases+=("$1"); shift ;;
  esac
done
[ ${#cases[@]} -gt 0 ] || { sed -n '2,23p' "$0" >&2; exit 2; }
bench="$root/build/spral/bench_gpu_ssids"
[ -x "$bench" ] || { echo "gpu_g1: $bench missing (tools/dev/build.sh spral)" >&2; exit 2; }
IFS=, read -r -a vlist <<< "$variants"
read -r -a extra <<< "$bench_args"

gpu_idle() {
  local uuid used
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
  [ "${used// /}" -lt 16 ] && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
}

if [ -n "$batch_rep" ]; then
  # All variants of one case, rotated by the repetition so that no variant always runs first.
  dir="${cases[0]}"
  n=${#vlist[@]}
  for ((i = 0; i < n; i++)); do
    v="${vlist[$(((i + batch_rep) % n))]}"
    backend="${v%%:*}" ordering="${v#*:}"
    case "$backend" in
      mumps) env_args=(CUDA_VISIBLE_DEVICES=) ;;
      ssids-cpu) env_args=(CUDA_VISIBLE_DEVICES= OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE) ;;
      ssids-gpu)
        if ! gpu_idle; then echo "gpu_g1: GPU $gpu is in use - stopping" >&2; exit 3; fi
        env_args=(CUDA_VISIBLE_DEVICES="$gpu" OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE) ;;
      *) echo "gpu_g1: unknown variant $v" >&2; exit 2 ;;
    esac
    if line="$(env "${env_args[@]}" "$bench" --case "$dir" --backend "$backend" --ordering "$ordering" \
                 ${extra[@]+"${extra[@]}"} 2>>"$out.log")"; then
      python3 -c 'import json, sys; d = json.loads(sys.argv[1]); d.update(rep=int(sys.argv[2]), variant=sys.argv[3]); print(json.dumps(d))' \
        "$line" "$batch_rep" "$v" >> "$out"
    else
      echo "gpu_g1: $v on $dir failed (see $out.log)" >&2
      printf '{"case": "%s", "variant": "%s", "rep": %d, "failed": true}\n' "$(basename "$dir")" "$v" "$batch_rep" >> "$out"
    fi
  done
  exit 0
fi

mkdir -p "$(dirname "$out")"
if [ "$lane" = 1 ]; then guard=(python3 tools/run/guard.py --skip-busy-check --timeout 900 --)
else guard=(python3 tools/run/guard.py --exclusive --timeout 900 --); fi
for ((rep = 1; rep <= reps; rep++)); do
  for c in "${cases[@]}"; do
    dir="$c"; [ -d "$dir" ] || dir="data/cases/$c"
    [ -d "$dir" ] || { echo "gpu_g1: no case $c" >&2; exit 2; }
    echo "gpu_g1: rep $rep/$reps $c" >&2
    status=0
    "${guard[@]}" "$self" --batch "$rep" --variants "$variants" --out "$out" --gpu "$gpu" \
      --bench-args "$bench_args" "$dir" || status=$?
    [ "$status" = 0 ] || { echo "gpu_g1: batch exited with $status" >&2; exit "$status"; }
  done
done
