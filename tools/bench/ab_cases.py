#!/usr/bin/env python3
"""A/B regression of two hsolve binaries on data cases: interleaved runs and a ratio table.

  run    run every (case, binary, repetition) in ABBA order inside the measurement slot: repetition r
         runs A then B for even r and B then A for odd r, repetitions outermost (every case's r-th
         pair before any (r+1)-th). Each run is the binary's default configuration (the case's
         config.json, --threads T, no other option) under its own nested guard, so the raw run
         directory has the layout tools/eval/sweep.py evaluates:
             RUN/<case>/<label>-r<rep>/            result directory (eigenvectors kept for tools/eval)
             RUN/<case>/guard-<label>-r<rep>.json  guard summary
             RUN/<case>/stderr-<label>-r<rep>.txt  the run's stderr (solver errors, guard summary line)
             RUN/runs.jsonl                        one line per run (host, binary, times, phases)
         Runs whose guard summary exists are skipped (resume); none starts after --deadline s.
  table  medians of report time_total_s per (case, binary) over the valid runs of one or more raw
         run directories, ratio B/A, geometric mean and the cases slower than --threshold. A run is
         valid when it exited 0, was not killed, claims success and tools/eval/sweep.py's
         evaluation.json in its result directory says pass; a ratio enters the geometric mean only
         when every run of both binaries on that case was valid.

Measurement slot (docs/DESIGN.md §6): wrap `run` in one --exclusive guard per <= 45 min part, e.g.
  python3 tools/run/guard.py --exclusive --timeout 2700 -- python3 tools/bench/ab_cases.py run \\
      --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_new --reps 0,1,2 --run-dir $S/A-c003 case...
then evaluate in the build lane and tabulate:
  python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3600 -- \\
      python3 -m tools.eval.sweep $S/A-c003 --out $S/A-c003/eval.md
  python3 tools/bench/ab_cases.py table --base main --new new --md out.md --json out.json $S/A-*
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "tools/run/guard.py"
DATA = Path(os.environ.get("HS_DATA_ROOT", str(REPO / "data")))
PROBLEM = {"standard": "HEP", "generalized_definite": "GHEP", "generalized_indefinite": "GHIEP"}


def hsolve_command(binary: Path, case_dir: Path, out: Path, threads: int) -> list[str]:
    cmd = [str(binary), "--A", str(case_dir / "A.npz")]
    if (case_dir / "B.npz").exists():
        cmd += ["--B", str(case_dir / "B.npz")]
    return cmd + ["--config", str(case_dir / "config.json"), "--threads", str(threads), "--out", str(out), "-q"]


def report_fields(out: Path) -> dict[str, Any]:
    rep = json.loads((out / "report.json").read_text())
    fs = rep.get("factor_stats") or {}
    cert = rep.get("certificate") or {}
    return {"success": rep.get("success"), "time_total_s": rep.get("time_total_s"), "problem": rep.get("problem"),
            "n": rep.get("n"), "block_size": rep.get("block_size"), "factorizations": fs.get("factorizations"),
            "numeric_s": fs.get("factorize_s"), "analysis_s": fs.get("analyze_s"), "rhs_solved": fs.get("rhs_solved"),
            "certificate_method": cert.get("method"), "time_certificate_s": rep.get("time_certificate_s"),
            "peak_rss_bytes": rep.get("peak_rss_bytes")}


def cmd_run(a: argparse.Namespace) -> int:
    if os.environ.get("HS_GUARDED") != "1":
        raise SystemExit("ab_cases run: start it under tools/run/guard.py (--exclusive for timings)")
    if os.environ.get("HS_SLOT_HELD") is None:
        print("ab_cases: not inside the measurement slot - these times are not measurements", flush=True)
    bins = dict(b.split("=", 1) for b in a.bin)
    labels = list(bins)
    if len(labels) != 2:
        raise SystemExit("ab_cases run: give exactly two --bin LABEL=PATH")
    for lb, p in bins.items():
        if not Path(p).is_file():
            raise SystemExit(f"ab_cases run: binary {lb}={p} not found")
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    t0 = time.time()
    with open(run_dir / "runs.jsonl", "a") as log:
        for rep in (int(x) for x in a.reps.split(",")):
            order = labels if rep % 2 == 0 else labels[::-1]
            for name in a.cases:
                for lb in order:
                    run = f"{lb}-r{rep}"
                    gsum = run_dir / name / f"guard-{run}.json"
                    if gsum.exists():
                        continue
                    if time.time() - t0 > a.deadline:
                        print("deadline reached: no further run started", flush=True)
                        return 0
                    out = run_dir / name / run
                    shutil.rmtree(out, ignore_errors=True)
                    out.mkdir(parents=True, exist_ok=True)
                    g = [sys.executable, str(GUARD), "--skip-busy-check", "--threads", str(a.threads),
                         "--timeout", str(a.timeout), "--summary", str(gsum)]
                    if a.cores:
                        g += ["--cores", a.cores]
                    cmd = hsolve_command(Path(bins[lb]), DATA / "cases" / name, out, a.threads)
                    p = subprocess.run(g + ["--"] + cmd, capture_output=True, text=True)
                    (run_dir / name / f"stderr-{run}.txt").write_text(p.stderr)   # the solver's errors
                    s = json.loads(gsum.read_text()) if gsum.exists() else {"returncode": p.returncode}
                    rec: dict[str, Any] = {"case": name, "label": lb, "rep": rep, "host": host, "binary": bins[lb],
                                           "rc": s.get("returncode"), "killed": s.get("killed"),
                                           "wall_s": s.get("wall_s"), "peak_rss_gib": s.get("peak_rss_gib")}
                    try:
                        rec.update(report_fields(out))
                    except (OSError, ValueError) as err:
                        rec["error"] = f"{type(err).__name__}: {err}; {p.stderr[-300:]}"
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"{name:34s} {run:10s} rc={rec['rc']} t={rec.get('time_total_s')} "
                          f"wall={rec.get('wall_s')}", flush=True)
    return 0


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def collect(run_dirs: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """case -> label -> runs (from runs.jsonl, with the evaluation verdict of each result directory)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rd in run_dirs:
        for line in (rd / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            ev_path = rd / r["case"] / f"{r['label']}-r{r['rep']}" / "evaluation.json"
            ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
            r["verdict"] = ev["verdict"] if ev else "not evaluated"
            r["completeness"] = ev["completeness"]["status"] if ev else None
            r["valid"] = (r.get("rc") == 0 and not r.get("killed") and bool(r.get("success"))
                          and r["verdict"] == "pass")
            out.setdefault(r["case"], {}).setdefault(r["label"], []).append(r)
    return out


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over the valid runs (over all runs that report a time when none is valid)."""
    t = [r["time_total_s"] for r in runs if r["valid"] and r.get("time_total_s") is not None] or \
        [r["time_total_s"] for r in runs if r.get("time_total_s") is not None]

    def common(key: str) -> Any:
        vals = sorted({str(r.get(key)) for r in runs if r.get(key) is not None})
        return "/".join(vals) if vals else None
    return {"reps": len(runs), "valid": sum(r["valid"] for r in runs),
            "ok": bool(runs) and all(r["valid"] for r in runs),
            "time_s": _median(t), "time_min": min(t) if t else None, "time_max": max(t) if t else None, "times": t,
            "wall_s": _median([r["wall_s"] for r in runs if r.get("wall_s") is not None]),
            "peak_rss_gib": max((r["peak_rss_gib"] for r in runs if r.get("peak_rss_gib") is not None), default=None),
            "factorizations": common("factorizations"), "block_size": common("block_size"),
            "certificate_method": common("certificate_method"), "completeness": common("completeness"),
            "hosts": sorted({r["host"] for r in runs}), "binaries": sorted({r["binary"] for r in runs}),
            "invalid": sorted({f"r{r['rep']}: rc={r.get('rc')} verdict={r['verdict']}"
                               for r in runs if not r["valid"]})}


def cmd_table(a: argparse.Namespace) -> int:
    data = collect([Path(p) for p in a.run_dirs])
    rows = []
    for name in sorted(data):
        base, new = summarize(data[name].get(a.base, [])), summarize(data[name].get(a.new, []))
        any_run = (data[name].get(a.new) or data[name].get(a.base))[0]
        ratio = new["time_s"] / base["time_s"] if base["time_s"] and new["time_s"] else None
        rows.append({"case": name, "n": any_run.get("n"), "problem": PROBLEM.get(any_run.get("problem"),
                                                                                 any_run.get("problem")),
                     "hosts": sorted(set(base["hosts"]) | set(new["hosts"])), a.base: base, a.new: new,
                     "ratio": ratio, "comparable": base["ok"] and new["ok"] and ratio is not None})
    comp = [r for r in rows if r["comparable"]]
    geo = math.exp(sum(math.log(r["ratio"]) for r in comp) / len(comp)) if comp else None
    slower = [r for r in comp if r["ratio"] > 1 + a.threshold]
    faster = [r for r in comp if r["ratio"] < 1 - a.threshold]
    doc = {"base": a.base, "new": a.new, "threshold": a.threshold, "geomean_ratio": geo, "cases": len(rows),
           "comparable": len(comp), "slower": [r["case"] for r in slower], "faster": [r["case"] for r in faster],
           "rows": rows}
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    lines = [f"| case | n | type | node | {a.base} s [min-max] | {a.new} s [min-max] | {a.new}/{a.base} | "
             f"valid {a.base} / {a.new} | numeric # {a.base} / {a.new} | block p {a.base} / {a.new} | "
             f"certificate {a.new} |", "|" + "---|" * 11]
    for r in rows:
        b, n = r[a.base], r[a.new]
        mark = " **slower**" if r in slower else ("" if r["comparable"] else " (invalid runs: not compared)")
        lines.append(
            f"| {r['case']} | {r['n']} | {r['problem']} | {','.join(r['hosts'])} | "
            f"{f(b['time_s'], 3)} [{f(b['time_min'], 3)}-{f(b['time_max'], 3)}] | "
            f"{f(n['time_s'], 3)} [{f(n['time_min'], 3)}-{f(n['time_max'], 3)}] | "
            f"{f(r['ratio'], 3)}{mark} | {b['valid']}/{b['reps']} / {n['valid']}/{n['reps']} | "
            f"{b['factorizations']} / {n['factorizations']} | {b['block_size']} / {n['block_size']} | "
            f"{n['certificate_method'] or '-'} |")
    lines += ["", f"Geometric mean of {a.new}/{a.base} over the {len(comp)} cases valid in every run: "
              f"**{f(geo, 3)}**; faster by > {a.threshold:.0%}: {len(faster)}; slower by > {a.threshold:.0%}: "
              f"{len(slower)}" + (" (" + ", ".join(f"{r['case']} {r['ratio']:.3f}" for r in slower) + ")"
                                  if slower else "") + "."]
    bad = [r for r in rows if not r["comparable"]]
    if bad:
        lines.append("Not comparable (an invalid or missing run): " + "; ".join(
            f"{r['case']} ({a.base}: {', '.join(r[a.base]['invalid']) or 'ok'}; "
            f"{a.new}: {', '.join(r[a.new]['invalid']) or 'ok'})" for r in bad) + ".")
    md = "\n".join(lines) + "\n"
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1) + "\n")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="interleaved ABBA runs (inside guard.py --exclusive)")
    r.add_argument("cases", nargs="+")
    r.add_argument("--bin", action="append", required=True, help="LABEL=PATH, twice (A first)")
    r.add_argument("--reps", default="0,1,2", help="repetition indices, e.g. 0,1,2 or 1,2 to resume a part")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--cores", help="cpulist of the nested guards (default: the slot / HS_CORES)")
    r.add_argument("--timeout", type=int, default=1500, help="per run, seconds")
    r.add_argument("--deadline", type=float, default=2400, help="no run starts after so many seconds")
    t = sub.add_parser("table", help="medians, ratios, geometric mean")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--base", default="main")
    t.add_argument("--new", default="new")
    t.add_argument("--threshold", type=float, default=0.03)
    t.add_argument("--md")
    t.add_argument("--json")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    sys.exit(main())
