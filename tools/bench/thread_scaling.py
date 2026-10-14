#!/usr/bin/env python3
"""Thread scaling of one hsolve binary on data cases: interleaved runs at several thread counts and a
speedup / parallel-efficiency table with the phase split (P9 integration, results/p9_measurements.md).

  run    run every (case, thread count T, repetition) inside the measurement slot. Repetitions are
         outermost, then cases; within a case the thread counts go up for even repetitions and down
         for odd ones. Each run is the binary's default configuration (the case's config.json,
         --threads T, no other option) under its own nested guard pinned to the first T cores of the
         slot (--first-core, default the slot's first core: 16 on the cluster nodes), so the outer
         guard (--exclusive, the whole slot) keeps the other cores of the socket idle. The raw run
         directory has the layout of tools/bench/ab_cases.py (label t<T>), which tools/eval/sweep.py
         evaluates:
             RUN/<case>/t<T>-r<rep>/            result directory (eigenvectors kept for tools/eval)
             RUN/<case>/guard-t<T>-r<rep>.json  guard summary
             RUN/<case>/stderr-t<T>-r<rep>.txt  the run's stderr
             RUN/runs.jsonl                     one line per run (host, threads, times, phases)
         Runs whose guard summary exists are skipped (resume); none starts after --deadline s.
  table  per (case, T): median report time_total_s over the valid runs, speedup t(1) / t(T) (the
         smallest measured T when 1 is missing) and parallel efficiency speedup x T_min / T, with the
         medians of the phases: analysis (factor_stats.analyze_s: orderings and symbolic analyses of
         the main factorization object), numeric factorizations (factor_stats.factorize_s), solves
         (factor_stats.solve_s: the block triangular solves of the Krylov iteration and of the
         refinement), iteration without the solves (time_iterate_s - solves), certificate
         (time_certificate_s: the inertia counts) and the rest (setup, verify, other analyses).
         A run is valid as in ab_cases.py (exit 0, not killed, success claimed, tools/eval pass); a
         speedup is given only when every run of both thread counts was valid.

Measurement slot (docs/DESIGN.md §6): wrap `run` in one --exclusive guard per <= 45 min part:
  python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- python3 tools/bench/thread_scaling.py \\
      run --bin build/release/hsolve --threads 1,2,4,8,16 --reps 0,1,2 --run-dir $S/T case...
then evaluate in the build lane and tabulate:
  python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3600 -- \\
      python3 -m tools.eval.sweep $S/T --out $S/T/eval.md          # --root DATA for other data roots
  python3 tools/bench/thread_scaling.py table --md out.md --json out.json $S/T
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bench.ab_cases import GUARD, PROBLEM, collect, hsolve_command, report_fields  # noqa: E402

PHASES = ("analysis", "numeric", "solves", "iterate_other", "certificate", "rest")


def phases(out: Path) -> dict[str, float]:
    """The phase split of one report (see the module docstring)."""
    rep = json.loads((out / "report.json").read_text())
    fs = rep.get("factor_stats") or {}
    total = rep["time_total_s"]
    analysis, numeric, solves = fs.get("analyze_s", 0.0), fs.get("factorize_s", 0.0), fs.get("solve_s", 0.0)
    iterate_other = max(0.0, rep.get("time_iterate_s", 0.0) - solves)
    cert = rep.get("time_certificate_s", 0.0)
    rest = max(0.0, total - analysis - numeric - solves - iterate_other - cert)
    return {"analysis": analysis, "numeric": numeric, "solves": solves, "iterate_other": iterate_other,
            "certificate": cert, "rest": rest}


def cmd_run(a: argparse.Namespace) -> int:
    if os.environ.get("HS_GUARDED") != "1":
        raise SystemExit("thread_scaling run: start it under tools/run/guard.py (--exclusive for timings)")
    if os.environ.get("HS_SLOT_HELD") is None:
        print("thread_scaling: not inside the measurement slot - these times are not measurements", flush=True)
    binary = Path(a.bin)
    if not binary.is_file():
        raise SystemExit(f"thread_scaling run: binary {binary} not found")
    threads = [int(x) for x in a.threads.split(",")]
    first = a.first_core
    if first is None:
        slot = os.environ.get("HS_MEASURE_CORES", "")
        first = int(slot.split("-")[0].split(",")[0]) if slot else min(os.sched_getaffinity(0))
    allowed = os.sched_getaffinity(0)
    for t in threads:
        missing = set(range(first, first + t)) - allowed
        if missing:
            raise SystemExit(f"thread_scaling run: cores {sorted(missing)} for T = {t} are not available")
    data = Path(a.root)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    t0 = time.time()
    with open(run_dir / "runs.jsonl", "a") as log:
        for rep in (int(x) for x in a.reps.split(",")):
            order = threads if rep % 2 == 0 else threads[::-1]
            for name in a.cases:
                for t in order:
                    label = f"t{t}"
                    run = f"{label}-r{rep}"
                    gsum = run_dir / name / f"guard-{run}.json"
                    if gsum.exists():
                        continue
                    if time.time() - t0 > a.deadline:
                        print("deadline reached: no further run started", flush=True)
                        return 0
                    out = run_dir / name / run
                    shutil.rmtree(out, ignore_errors=True)
                    out.mkdir(parents=True, exist_ok=True)
                    g = [sys.executable, str(GUARD), "--skip-busy-check", "--cores", f"{first}-{first + t - 1}",
                         "--threads", str(t), "--timeout", str(a.timeout), "--summary", str(gsum)]
                    cmd = hsolve_command(binary, data / "cases" / name, out, t)
                    p = subprocess.run(g + ["--"] + cmd, capture_output=True, text=True)
                    (run_dir / name / f"stderr-{run}.txt").write_text(p.stderr)
                    s = json.loads(gsum.read_text()) if gsum.exists() else {"returncode": p.returncode}
                    rec: dict[str, Any] = {"case": name, "label": label, "threads": t, "rep": rep, "host": host,
                                           "binary": str(binary), "rc": s.get("returncode"),
                                           "killed": s.get("killed"), "wall_s": s.get("wall_s"),
                                           "peak_rss_gib": s.get("peak_rss_gib")}
                    try:
                        rec.update(report_fields(out))
                        rec["phases"] = phases(out)
                    except (OSError, ValueError, KeyError) as err:
                        rec["error"] = f"{type(err).__name__}: {err}; {p.stderr[-300:]}"
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"{name:34s} {run:8s} rc={rec['rc']} t={rec.get('time_total_s')} "
                          f"wall={rec.get('wall_s')}", flush=True)
    return 0


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def cmd_table(a: argparse.Namespace) -> int:
    data = collect([Path(p) for p in a.run_dirs])
    doc: dict[str, Any] = {"cases": []}
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    lines = []
    for name in sorted(data, key=lambda c: (a.order.index(c) if c in a.order else len(a.order), c)):
        by_t = {int(lb[1:]): runs for lb, runs in data[name].items() if lb.startswith("t")}
        any_run = next(iter(by_t.values()))[0]
        rows = []
        for t in sorted(by_t):
            runs = by_t[t]
            valid = [r for r in runs if r["valid"]]
            times = [r["time_total_s"] for r in valid]
            ph = {k: _median([r["phases"][k] for r in valid if "phases" in r]) for k in PHASES}
            rows.append({"threads": t, "reps": len(runs), "valid": len(valid), "ok": len(valid) == len(runs),
                         "time_s": _median(times), "time_min": min(times) if times else None,
                         "time_max": max(times) if times else None, "times": times, "phases": ph,
                         "peak_rss_gib": max((r["peak_rss_gib"] for r in runs if r.get("peak_rss_gib")),
                                             default=None),
                         "block_size": "/".join(sorted({str(r.get("block_size")) for r in runs})),
                         "certificate_method": "/".join(sorted({str(r.get("certificate_method")) for r in runs}))})
        base = rows[0]
        for r in rows:
            ok = base["ok"] and r["ok"] and base["time_s"] and r["time_s"]
            r["speedup"] = base["time_s"] / r["time_s"] if ok else None
            r["efficiency"] = r["speedup"] * base["threads"] / r["threads"] if ok else None
            r["phase_speedup"] = {k: (base["phases"][k] / r["phases"][k]
                                      if ok and base["phases"][k] and r["phases"][k] else None) for k in PHASES}
        doc["cases"].append({"case": name, "n": any_run.get("n"),
                             "problem": PROBLEM.get(any_run.get("problem"), any_run.get("problem")),
                             "hosts": sorted({r["host"] for runs in by_t.values() for r in runs}), "rows": rows})
        lines += [f"**{name}** (n = {any_run.get('n')}, "
                  f"{PROBLEM.get(any_run.get('problem'), any_run.get('problem'))}, certificate "
                  f"{rows[-1]['certificate_method']}, block p {rows[-1]['block_size']})", "",
                  f"| threads | total s [min-max] | speedup | efficiency | analysis | numeric fact. | solves | "
                  f"iteration (other) | certificate | rest | valid | peak GiB |", "|" + "---|" * 12]
        for r in rows:
            p = r["phases"]
            lines.append(f"| {r['threads']} | {f(r['time_s'], 3)} [{f(r['time_min'], 3)}-{f(r['time_max'], 3)}] | "
                         f"{f(r['speedup'])}x | {f(r['efficiency'])} | {f(p['analysis'], 3)} | "
                         f"{f(p['numeric'], 3)} | {f(p['solves'], 3)} | {f(p['iterate_other'], 3)} | "
                         f"{f(p['certificate'], 3)} | {f(p['rest'], 3)} | {r['valid']}/{r['reps']} | "
                         f"{f(r['peak_rss_gib'])} |")
        last = rows[-1]
        if last["speedup"]:
            lines.append("")
            lines.append(f"Phase speedups {base['threads']} -> {last['threads']} threads: " + ", ".join(
                f"{k} {f(v)}x" for k, v in last["phase_speedup"].items() if v))
        lines.append("")
    md = "\n".join(lines)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1) + "\n")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="interleaved runs at several thread counts (inside guard.py --exclusive)")
    r.add_argument("cases", nargs="+")
    r.add_argument("--bin", required=True, help="the hsolve binary")
    r.add_argument("--threads", default="1,2,4,8,16", help="thread counts, ascending")
    r.add_argument("--reps", default="0,1,2", help="repetition indices, e.g. 0,1,2 or 2 to resume a part")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--root", default=os.environ.get("HS_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")),
                   help="data root with cases/ (default $HS_DATA_ROOT or the repository's data/)")
    r.add_argument("--first-core", type=int, help="first core of the runs (default: the slot's first)")
    r.add_argument("--timeout", type=int, default=1500, help="per run, seconds")
    r.add_argument("--deadline", type=float, default=2400, help="no run starts after so many seconds")
    t = sub.add_parser("table", help="speedup, efficiency and phases per case and thread count")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--order", nargs="*", default=[], help="case order of the tables (others after, sorted)")
    t.add_argument("--md")
    t.add_argument("--json")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    sys.exit(main())
