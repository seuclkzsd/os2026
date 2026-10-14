#!/usr/bin/env python3
"""GPU offload of the numeric factorizations, A/B at equal pivot order (P10 "gpu", results/p10_gpu.md).

Runs build/release/bench_factspeed_factor (the driver's main factorization at sigma and the exact
certificate's two inertia-only counts at the window ends a, b; bench/factspeed_factor.cpp) with the
BLAS hook preloaded (build/release/libhsolve_blashook.so) under each GPU mode (HS_GPU=off / on / ...),
repetitions interleaved (variant order reversed every other repetition), every run on the same saved
pivot order (hsolve's nested dissection, ordering 10, written once per case with --perm-out), so that
the variants factorize the same tree with the same flops. Per run: the wall time, inertia (negative /
zero pivots, validity) and MUMPS statistics of each factorization, and the hook's GPU report
(HS_GPU_REPORT). Inertia must be identical in every run of a case (checked and reported).

  python3 tools/bench/gpu_factor_ab.py --out FILE.jsonl [--reps 3] [--variants off,on] [--jobs main,a,b]
      [--root DIR] [--perm-dir DIR] [--env K=V ...] [--variant-env VARIANT:K=V ...] CASE[:LOWER:UPPER] ...

CASE is a directory name under ROOT/cases (default $HOME/p7_p7-membudget/data, the 1e6 cases); the
certificate window LOWER:UPPER is needed for the jobs a / b (known for syn-mem27-100 and
yee-pec-box-1e6: the P7/P9 windows). Run it in the measurement slot with one GPU visible:
  tools/bench/gpu_slot.sh 7 2700 -- python3 tools/bench/gpu_factor_ab.py --out OUT.jsonl --reps 5 \\
      syn-mem27-100 yee-pec-box-1e6
A second invocation with the same --out appends (the summary covers every run in the file).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "build/release/bench_factspeed_factor"
HOOK = REPO / "build/release/libhsolve_blashook.so"
# Certificate windows of the 1e6 cases (hsolve reports of P7/P9; results/p9_factspeed.md §5).
WINDOWS = {
    "syn-mem27-100": (-0.0005369991406518608, 0.0005978355069250973),
    "yee-pec-box-1e6": (8.087001904223854, 96.1897089202103),
}


def parse_case(spec: str) -> tuple[str, float | None, float | None]:
    parts = spec.split(":")
    if len(parts) == 3:
        return parts[0], float(parts[1]), float(parts[2])
    lo, hi = WINDOWS.get(spec, (None, None))
    return spec, lo, hi


def run_one(case_dir: Path, lo, hi, jobs: str, perm: Path, env: dict[str, str], extra: list[str]) -> dict:
    cmd = [str(BENCH), "--case", str(case_dir), "--jobs", jobs] + extra
    if lo is not None:
        cmd += ["--lower", repr(lo), "--upper", repr(hi)]
    if perm.exists():
        cmd += ["--perm-in", str(perm)]
    else:
        cmd += ["--ordering", "10", "--perm-out", str(perm)]
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as rf:
        report_path = Path(rf.name)
    full_env = {**os.environ, **env, "LD_PRELOAD": str(HOOK), "HS_GPU_REPORT": str(report_path)}
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, env=full_env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    facts = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        if j.get("event") == "factorization":
            facts.append({k: j.get(k) for k in ("job", "rep", "wall_s", "neg", "zero", "valid", "ok", "rinfog1",
                                                 "rinfog3", "infog29", "peak_rss_mb", "user_s", "sys_s")})
        elif j.get("event") == "analysis":
            facts.append({"job": "analysis", "wall_s": j.get("analyze_s"), "est_flops": j.get("est_flops")})
    gpu = None
    try:
        text = report_path.read_text().strip()
        gpu = json.loads(text) if text else None
    except (OSError, json.JSONDecodeError):
        gpu = None
    report_path.unlink(missing_ok=True)
    return {"rc": proc.returncode, "process_s": wall, "factorizations": facts, "gpu": gpu,
            "stderr_tail": proc.stderr[-800:] if proc.returncode else ""}


def summarize(lines: list[dict]) -> str:
    """Per (case, label, job, variant): median wall time [min-max] and the speedup against the
    same label's 'off' variant; the inertia must be the same in every run of a (case, job)."""
    out = ["| case | label | job | variant | runs | wall s median [min-max] | speedup vs off | inertia (neg) |",
           "|---|---|---|---|---|---|---|---|"]
    by: dict[tuple, list[float]] = {}
    inertia: dict[tuple, set] = {}
    for r in lines:
        for f in r["factorizations"]:
            if f["job"] == "analysis":
                continue
            key = (r["case"], r.get("label", ""), f["job"], r["variant"])
            by.setdefault(key, []).append(f["wall_s"])
            inertia.setdefault((r["case"], f["job"]), set()).add((f.get("neg"), f.get("zero"), f.get("valid")))
    for (case, label, job, var), ws in sorted(by.items()):
        base = by.get((case, label, job, "off"))
        sp = f"{statistics.median(base) / statistics.median(ws):.3f}" if base and var != "off" else "-"
        inn = inertia[(case, job)]
        inn_s = ", ".join(str(n) for n, _, _ in sorted(inn, key=str)) + ("" if len(inn) == 1 else " DIFFER")
        out.append(f"| {case} | {label} | {job} | {var} | {len(ws)} | {statistics.median(ws):.2f} "
                   f"[{min(ws):.2f}-{max(ws):.2f}] | {sp} | {inn_s} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", default="off,on", help="HS_GPU values, run interleaved")
    ap.add_argument("--jobs", default="main,a,b")
    ap.add_argument("--root", type=Path, default=Path.home() / "p7_p7-membudget/data")
    ap.add_argument("--perm-dir", type=Path, default=Path.home() / "p10gpu/perm")
    ap.add_argument("--env", action="append", default=[], help="K=V for every run")
    ap.add_argument("--variant-env", action="append", default=[], help="VARIANT:K=V for one variant's runs")
    ap.add_argument("--bench-args", default="", help="extra bench_factspeed_factor arguments (space separated)")
    ap.add_argument("--label", default="", help="label stored with every run (e.g. the configuration)")
    ap.add_argument("--summary-only", action="store_true", help="print the summary of --out and exit")
    args = ap.parse_args(argv)
    if not args.summary_only:
        if not BENCH.exists() or not HOOK.exists():
            raise SystemExit(f"build first: {BENCH} / {HOOK} missing")
        variants = args.variants.split(",")
        common = dict(kv.split("=", 1) for kv in args.env)
        per_variant: dict[str, dict[str, str]] = {}
        for spec in args.variant_env:
            v, kv = spec.split(":", 1)
            k, val = kv.split("=", 1)
            per_variant.setdefault(v, {})[k] = val
        args.perm_dir.mkdir(parents=True, exist_ok=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for rep in range(args.reps):
            for spec in args.cases:
                name, lo, hi = parse_case(spec)
                perm = args.perm_dir / f"{name}.nd.perm"
                order = variants if rep % 2 == 0 else variants[::-1]
                for v in order:
                    env = {**common, **per_variant.get(v, {}), "HS_GPU": v}
                    r = run_one(args.root / "cases" / name, lo, hi, args.jobs, perm, env, args.bench_args.split())
                    r.update(case=name, variant=v, rep=rep, env=env, label=args.label, bench_args=args.bench_args)
                    with args.out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    walls = " ".join(f"{f['job']}={f['wall_s']:.2f}" for f in r["factorizations"])
                    g = r["gpu"] or {}
                    print(f"{name} rep {rep} {v}: rc={r['rc']} {walls} gpu_gflop={g.get('gpu_gflop')} "
                          f"calls={g.get('calls')} init_s={g.get('init_s')}", flush=True)
    lines = [json.loads(x) for x in args.out.read_text().splitlines() if x.strip()]
    print(summarize(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
