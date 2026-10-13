#!/usr/bin/env bash
# P8 task 2: are concurrent certificate counts (two inertia-only factorizations, 8 + 8 of the 16
# slot threads) faster than the driver's sequential ones (16 threads each)? Runs, for every case
# line of CASES (name lower upper ordering), inside ONE measurement-slot guard call (the caller's):
#   seq16  bench_certcost_concurrent --threads 16 --exp seq       (cores 16-31)
#   one8   the same --threads 8 --exp one --at a, alone          (cores 16-23)
#   two8   two processes at once: --at a on 16-23, --at b on 24-31 (process isolation: the upper
#          bound of any concurrent scheme)
#   fork8  --threads 8 --exp seq,fork (the in-process scheme by fork(); counts must agree)
# Usage (from the work tree, one command line):
#   env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 -- bash tools/bench/certcost_concurrent.sh
#     OUTDIR REPS CASES_FILE
# CASES_FILE: one "case lower upper ordering" per line (the windows of hsolve reports;
# results/p8_certcost.md §3 lists the ones used). Output: OUTDIR/<case>.jsonl (one JSON per line,
# with a "variant" key).
set -u
out=$1 reps=$2 cases=$3
mkdir -p "$out"
bin=build/release/bench_certcost_concurrent
tag() { python3 -c 'import json,sys
v=sys.argv[1]
for l in sys.stdin:
    l=l.strip()
    if l.startswith("{"):
        d=json.loads(l); d["variant"]=v; print(json.dumps(d))' "$1"; }
while read -r name lower upper ord; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  d=data/cases/$name
  f=$out/$name.jsonl
  : > "$f"
  common=(--case "$d" --lower "$lower" --upper "$upper" --ordering "$ord" --reps "$reps")
  echo "== $name $(date +%T)" >&2
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 taskset -c 16-31 "$bin" "${common[@]}" --threads 16 --exp seq | tag seq16 >> "$f"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 16-23 "$bin" "${common[@]}" --threads 8 --exp one --at a | tag one8 >> "$f"
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 16-23 "$bin" "${common[@]}" --threads 8 --exp one --at a > "$out/$name.two_a" &
  pa=$!
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 24-31 "$bin" "${common[@]}" --threads 8 --exp one --at b > "$out/$name.two_b" &
  pb=$!
  wait $pa $pb
  tag two8a < "$out/$name.two_a" >> "$f"
  tag two8b < "$out/$name.two_b" >> "$f"
  rm -f "$out/$name.two_a" "$out/$name.two_b"
  OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 taskset -c 16-31 "$bin" "${common[@]}" --threads 8 --exp seq,fork | tag fork8 >> "$f"
done < "$cases"
echo "done $(date +%T)" >&2
