#!/usr/bin/env python3
"""Thread scaling of one hsolve binary on data cases: interleaved runs at several thread counts and
configurations, and a speedup / parallel-efficiency table with the phase split (P9 integration,
results/p9_measurements.md §5).

  run    run every (case, thread count T, configuration, repetition) inside the measurement slot.
         Repetitions are outermost, then cases; within a case the thread counts go up for even
         repetitions and down for odd ones, and at each T the configurations run in the order given
         (reversed for odd repetitions). A configuration (--variant, repeatable) is
             default                the case's config.json as it is (label t<T>; the only one when
                                    no --variant is given)
             NAME[@T,T...]=JSON     the case's config.json with JSON's top-level keys replaced
                                    (label NAME-t<T>; NAME of letters, digits, '_' and '.'), at
                                    every --threads count or only at the counts after '@'
         e.g. --variant default --variant 'nd={"ordering":10}' --variant 'scotch@1={"ordering":3}'
         runs the adaptive default and the own nested dissection at every T and SCOTCH at T = 1.
         A fixed ordering is how the speedup of one and the same computation is measured: the
         adaptive ordering's cost model prices the parallel nested dissection by the thread count,
         so the default configuration may factorize another pivot order at T = 1 than at T = 16.
         Each run is hsolve --config <the configuration> --threads T (no other option) under its own
         nested guard pinned to the first T cores of the slot (--first-core, default the slot's
         first core: 16 on the cluster nodes), so the outer guard (--exclusive, the whole slot) keeps
         the other cores of the socket idle. The raw run directory has the layout of
         tools/bench/ab_cases.py, which tools/eval/sweep.py evaluates:
             RUN/<case>/<label>-r<rep>/            result directory (eigenvectors kept for tools/eval)
             RUN/<case>/guard-<label>-r<rep>.json  guard summary
             RUN/<case>/stderr-<label>-r<rep>.txt  the run's stderr
             RUN/<case>/config-<NAME>.json         a configuration's config.json (not for default)
             RUN/runs.jsonl                        one line per run (host, threads, times, phases)
         Runs whose guard summary exists are skipped (resume); none starts after --deadline s.
  table  per (case, configuration, T) the median report time_total_s over the valid runs and
           speedup, efficiency  t(T0) / t(T) of the same configuration, T0 its smallest measured T,
                                and that x T0 / T: the parallel speedup of one computation only
                                where the ordering is the same at T0 and T - a row whose ordering
                                differs is marked '*', its phase speedups and the summary flagged
           vs fastest           t_fastest / t(T), t_fastest the median of the case's fastest
                                configuration at the smallest T measured (every run valid), so that
                                no speedup is taken against a slower serial configuration; the
                                summary's efficiency is this speedup x T_min / T
         with the medians of an exclusive phase split of the report's five exclusive phases
         (docs/FORMATS.md §3; the columns add up to time_total_s):
           analysis       factor_stats.analyze_s (ordering and symbolic analysis of the main
                          factorization object; at most time_factor_s)
           factorization  the rest of time_factor_s: the numeric factorizations of the factor phase
                          (at sigma; B or A where the problem type needs them) and the other
                          candidate analyses of the adaptive ordering
           solves         factor_stats.solve_s, at most time_iterate_s: the block solves of the
                          Krylov iteration
           iteration      the rest of time_iterate_s (orthogonalization, restarts, products)
           certificate    time_certificate_s: the inertia counts (on the InertiaCounter, or
                          refactorizations of the main object - which factor_stats.factorize_s also
                          counts, so it is not used)
           rest           time_setup_s + time_verify_s
         the ordering factorized (factor_options.ordering: 0 AMD, 3 SCOTCH, 4 PORD, 5 METIS, 10 own
         nested dissection; the adaptive ordering's cost model depends on the thread count) and the
         main factor's entries (factor_stats.factor_entries; '-' where the runs differ). The table
         reads every run's report.json again, so it applies this split to older raw runs too. A run
         is valid as in ab_cases.py (exit 0, not killed, success claimed, tools/eval pass); a speedup
         is given only when every run of both rows was valid.

Measurement slot (docs/DESIGN.md §6): wrap `run` in one --exclusive guard per <= 45 min part:
  python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- python3 tools/bench/thread_scaling.py \\
      run --bin build/release/hsolve --threads 1,2,4,8,16 --reps 0,1,2 --run-dir $S/T \\
      --variant default --variant 'nd={"ordering":10}' case...
then evaluate in the build lane and tabulate:
  python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3600 -- \\
      python3 -m tools.eval.sweep $S/T --out $S/T/eval.md          # --root DATA for other data roots
  python3 tools/bench/thread_scaling.py table --md out.md --json out.json $S/T
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bench.ab_cases import GUARD, PROBLEM, collect, hsolve_command, report_fields  # noqa: E402

PHASES = ("analysis", "numeric", "solves", "iterate_other", "certificate", "rest")
LABELS = dict(zip(PHASES, ("analysis", "factorization", "solves", "iteration", "certificate", "rest")))
ORDERINGS = {0: "AMD", 2: "AMF", 3: "SCOTCH", 4: "PORD", 5: "METIS", 6: "QAMD", 7: "auto", 10: "ND"}
DEFAULT = "default"
_VARIANT = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)(?:@(?P<threads>[0-9]+(?:,[0-9]+)*))?(?:=(?P<json>.+))?$", re.S)
_LABEL = re.compile(r"^(?:(?P<name>[A-Za-z0-9_.]+)-)?t(?P<threads>[0-9]+)$")


@dataclass
class Variant:
    """One configuration of a case: its replacements of config.json keys and its thread counts."""
    name: str
    threads: list[int] | None = None      # None: every --threads count
    overrides: dict[str, Any] = field(default_factory=dict)

    def label(self, t: int) -> str:
        return f"t{t}" if self.name == DEFAULT else f"{self.name}-t{t}"


def parse_variant(spec: str) -> Variant:
    """default | NAME[@T,T...]=JSON (a non-empty object of top-level config.json keys)."""
    m = _VARIANT.match(spec)
    if not m:
        raise ValueError(f"bad --variant '{spec}': default or NAME[@T,T...]=JSON")
    name, threads, js = m["name"], m["threads"], m["json"]
    if name == DEFAULT:
        if threads or js:
            raise ValueError("--variant default is the case's own configuration at every thread count")
        return Variant(DEFAULT)
    if js is None:
        raise ValueError(f"--variant '{spec}': a configuration other than default needs =JSON")
    overrides = json.loads(js)
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"--variant '{spec}': JSON must be a non-empty object")
    return Variant(name, [int(t) for t in threads.split(",")] if threads else None, overrides)


def parse_label(label: str) -> tuple[str, int] | None:
    """(configuration, T) of a run label t<T> / NAME-t<T>; None for other labels."""
    m = _LABEL.match(label)
    return (m["name"] or DEFAULT, int(m["threads"])) if m else None


def variant_config(case_dir: Path, case_run_dir: Path, v: Variant) -> Path:
    """The configuration's config.json: the case's own, or a merged copy in the case's run directory."""
    own = case_dir / "config.json"
    if v.name == DEFAULT:
        return own
    cfg = json.loads(own.read_text()) if own.is_file() else {}
    cfg.update(v.overrides)
    path = case_run_dir / f"config-{v.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    return path


def schedule(cases: list[str], threads: list[int], variants: list[Variant],
             reps: list[int]) -> list[tuple[int, str, int, Variant]]:
    """(rep, case, T, configuration) in run order: repetitions outermost, then cases; thread counts
    up for even repetitions and down for odd ones; at each T the configurations that run at T, in
    the given order for even repetitions and reversed for odd ones."""
    out = []
    for rep in reps:
        order = threads if rep % 2 == 0 else threads[::-1]
        vorder = variants if rep % 2 == 0 else variants[::-1]
        for name in cases:
            for t in order:
                out += [(rep, name, t, v) for v in vorder if not v.threads or t in v.threads]
    return out


def phases(out: Path) -> dict[str, float]:
    """The exclusive phase split of one report (see the module docstring); sums to time_total_s."""
    rep = json.loads((out / "report.json").read_text())
    fs = rep.get("factor_stats") or {}
    factor, iterate = rep.get("time_factor_s", 0.0), rep.get("time_iterate_s", 0.0)
    analysis = min(fs.get("analyze_s", 0.0), factor)
    solves = min(fs.get("solve_s", 0.0), iterate)
    return {"analysis": analysis, "numeric": factor - analysis, "solves": solves,
            "iterate_other": iterate - solves, "certificate": rep.get("time_certificate_s", 0.0),
            "rest": rep.get("time_setup_s", 0.0) + rep.get("time_verify_s", 0.0)}


def ordering(out: Path) -> str:
    """The ordering the run factorized with (report factor_options.ordering)."""
    o = (json.loads((out / "report.json").read_text()).get("factor_options") or {}).get("ordering")
    return ORDERINGS.get(o, str(o))


def factor_entries(out: Path) -> int | None:
    """The main factor's entries (report factor_stats.factor_entries)."""
    return (json.loads((out / "report.json").read_text()).get("factor_stats") or {}).get("factor_entries")


def cmd_run(a: argparse.Namespace) -> int:
    if os.environ.get("HS_GUARDED") != "1":
        raise SystemExit("thread_scaling run: start it under tools/run/guard.py (--exclusive for timings)")
    if os.environ.get("HS_SLOT_HELD") is None:
        print("thread_scaling: not inside the measurement slot - these times are not measurements", flush=True)
    binary = Path(a.bin)
    if not binary.is_file():
        raise SystemExit(f"thread_scaling run: binary {binary} not found")
    threads = [int(x) for x in a.threads.split(",")]
    try:
        variants = [parse_variant(s) for s in (a.variant or [DEFAULT])]
    except ValueError as err:
        raise SystemExit(f"thread_scaling run: {err}")
    if len({v.name for v in variants}) != len(variants):
        raise SystemExit("thread_scaling run: the configurations need different names")
    for v in variants:
        if v.threads and not set(v.threads) <= set(threads):
            raise SystemExit(f"thread_scaling run: {v.name}'s thread counts {v.threads} are not all in --threads")
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
    configs = {(name, v.name): variant_config(data / "cases" / name, run_dir / name, v)
               for name in a.cases for v in variants}
    host = socket.gethostname().split(".")[0]
    t0 = time.time()
    with open(run_dir / "runs.jsonl", "a") as log:
        for rep, name, t, v in schedule(a.cases, threads, variants, [int(x) for x in a.reps.split(",")]):
            label = v.label(t)
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
            cmd[cmd.index("--config") + 1] = str(configs[(name, v.name)])
            p = subprocess.run(g + ["--"] + cmd, capture_output=True, text=True)
            (run_dir / name / f"stderr-{run}.txt").write_text(p.stderr)
            s = json.loads(gsum.read_text()) if gsum.exists() else {"returncode": p.returncode}
            rec: dict[str, Any] = {"case": name, "label": label, "variant": v.name, "overrides": v.overrides,
                                   "threads": t, "rep": rep, "host": host, "binary": str(binary),
                                   "rc": s.get("returncode"), "killed": s.get("killed"), "wall_s": s.get("wall_s"),
                                   "peak_rss_gib": s.get("peak_rss_gib")}
            try:
                rec.update(report_fields(out))
                rec["phases"] = phases(out)
            except (OSError, ValueError, KeyError) as err:
                rec["error"] = f"{type(err).__name__}: {err}; {p.stderr[-300:]}"
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{name:34s} {run:14s} rc={rec['rc']} t={rec.get('time_total_s')} wall={rec.get('wall_s')}",
                  flush=True)
    return 0


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _reread(run_dirs: list[Path], r: dict[str, Any]) -> None:
    """Phases, ordering and factor entries of run r from its report.json, when it is there."""
    for rd in run_dirs:
        out = rd / r["case"] / f"{r['label']}-r{r['rep']}"
        if (out / "report.json").is_file():
            r["phases"], r["ordering"], r["factor_entries"] = phases(out), ordering(out), factor_entries(out)
            return


def _row(t: int, runs: list[dict[str, Any]], run_dirs: list[Path]) -> dict[str, Any]:
    for r in runs:
        _reread(run_dirs, r)
    valid = [r for r in runs if r["valid"]]
    times = [r["time_total_s"] for r in valid]
    entries = {r.get("factor_entries") for r in runs}
    return {"threads": t, "reps": len(runs), "valid": len(valid), "ok": len(valid) == len(runs),
            "time_s": _median(times), "time_min": min(times) if times else None,
            "time_max": max(times) if times else None, "times": times,
            "phases": {k: _median([r["phases"][k] for r in valid if "phases" in r]) for k in PHASES},
            "peak_rss_gib": max((r["peak_rss_gib"] for r in runs if r.get("peak_rss_gib")), default=None),
            "block_size": "/".join(sorted({str(r.get("block_size")) for r in runs})),
            "ordering": "/".join(sorted({str(r.get("ordering", "-")) for r in runs})),
            "factor_entries": entries.pop() if len(entries) == 1 else None,
            "certificate_method": "/".join(sorted({str(r.get("certificate_method")) for r in runs}))}


def _ratio(base: dict[str, Any] | None, r: dict[str, Any]) -> float | None:
    ok = base is not None and base["ok"] and r["ok"] and base["time_s"] and r["time_s"]
    return base["time_s"] / r["time_s"] if ok else None


def case_table(name: str, labels: dict[str, list[dict[str, Any]]], run_dirs: list[Path]) -> dict[str, Any]:
    """One case: its configurations' rows per T with the speedups of the module docstring."""
    by_v: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for lb, runs in labels.items():
        p = parse_label(lb)
        if p:
            by_v.setdefault(p[0], {})[p[1]] = runs
    variants = [{"variant": v, "rows": [_row(t, by_v[v][t], run_dirs) for t in sorted(by_v[v])]}
                for v in sorted(by_v, key=lambda v: (v != DEFAULT, v))]
    t_min = min(v["rows"][0]["threads"] for v in variants)
    at_min = [(v["variant"], v["rows"][0]) for v in variants if v["rows"][0]["threads"] == t_min]
    fast = min((x for x in at_min if x[1]["ok"] and x[1]["time_s"]), key=lambda x: x[1]["time_s"], default=None)
    for v in variants:
        base = v["rows"][0]
        for r in v["rows"]:
            r["speedup"] = _ratio(base, r)
            r["efficiency"] = r["speedup"] * base["threads"] / r["threads"] if r["speedup"] else None
            r["same_ordering"] = r["ordering"] == base["ordering"]
            r["phase_speedup"] = {k: (base["phases"][k] / r["phases"][k]
                                      if r["speedup"] and base["phases"][k] and r["phases"][k] else None)
                                  for k in PHASES}
            r["speedup_vs_fastest"] = _ratio(fast[1] if fast else None, r)
            r["efficiency_vs_fastest"] = (r["speedup_vs_fastest"] * t_min / r["threads"]
                                          if r["speedup_vs_fastest"] else None)
    any_run = next(iter(next(iter(by_v.values())).values()))[0]
    return {"case": name, "n": any_run.get("n"),
            "problem": PROBLEM.get(any_run.get("problem"), any_run.get("problem")),
            "hosts": sorted({r["host"] for runs in labels.values() for r in runs}),
            "fastest": {"variant": fast[0], "threads": t_min, "time_s": fast[1]["time_s"],
                        "ordering": fast[1]["ordering"]} if fast else None,
            "variants": variants, "rows": variants[0]["rows"]}


def markdown(cases: list[dict[str, Any]]) -> str:
    """The summary (one line per case and configuration) and the per-case tables."""
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    ts = sorted({r["threads"] for c in cases for v in c["variants"] for r in v["rows"]})
    lines = ["| case | configuration | ordering | " + " | ".join(f"T = {t}" for t in ts)
             + " | speedup | vs fastest | efficiency vs fastest |", "|" + "---|" * (len(ts) + 6)]
    for c in cases:
        for v in c["variants"]:
            rows = {r["threads"]: r for r in v["rows"]}
            last = v["rows"][-1]
            lines.append(f"| {c['case']} | {v['variant']} | "
                         f"{'/'.join(dict.fromkeys(r['ordering'] for r in v['rows']))} | "
                         + " | ".join(f(rows[t]["time_s"]) if t in rows else "" for t in ts)
                         + f" | {f(last['speedup'])}x{'' if last['same_ordering'] else ' (orderings differ)'} | "
                         f"{f(last['speedup_vs_fastest'])}x | {f(last['efficiency_vs_fastest'])} |")
    lines.append("")
    for c in cases:
        fast = c["fastest"]
        lines += [f"**{c['case']}** (n = {c['n']}, {c['problem']}; fastest at T = "
                  + (f"{fast['threads']}: {fast['variant']} ({fast['ordering']}) {fast['time_s']:.3f} s)"
                     if fast else "-: no configuration with every run valid)"), ""]
        for v in c["variants"]:
            rows = v["rows"]
            lines += [f"*{v['variant']}* (certificate {rows[-1]['certificate_method']}, block p "
                      f"{rows[-1]['block_size']})", "",
                      "| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | "
                      "solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |",
                      "|" + "---|" * 15]
            for r in rows:
                p = r["phases"]
                lines.append(f"| {r['threads']} | {f(r['time_s'], 3)} [{f(r['time_min'], 3)}-{f(r['time_max'], 3)}] | "
                             f"{f(r['speedup'])}x | {f(r['efficiency'])} | {f(r['speedup_vs_fastest'])}x | "
                             f"{f(p['analysis'], 3)} | {f(p['numeric'], 3)} | {f(p['solves'], 3)} | "
                             f"{f(p['iterate_other'], 3)} | {f(p['certificate'], 3)} | {f(p['rest'], 3)} | "
                             f"{r['ordering']}{'' if r['same_ordering'] else ' *'} | "
                             f"{'-' if r['factor_entries'] is None else r['factor_entries']} | "
                             f"{r['valid']}/{r['reps']} | {f(r['peak_rss_gib'])} |")
            last, base = rows[-1], rows[0]
            if len(rows) > 1 and last["speedup"]:
                note = "" if last["same_ordering"] else (
                    f" - orderings differ ({base['ordering']} at {base['threads']}, {last['ordering']} at "
                    f"{last['threads']}): not the parallel speedup of one computation")
                lines += ["", f"Phase speedups {base['threads']} -> {last['threads']} threads: " + ", ".join(
                    f"{LABELS[k]} {f(x)}x" for k, x in last["phase_speedup"].items() if x) + note]
            lines.append("")
    return "\n".join(lines)


def cmd_table(a: argparse.Namespace) -> int:
    run_dirs = [Path(p) for p in a.run_dirs]
    data = collect(run_dirs)
    names = sorted(data, key=lambda c: (a.order.index(c) if c in a.order else len(a.order), c))
    cases = [case_table(name, data[name], run_dirs) for name in names]
    md = markdown(cases)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="interleaved runs at several thread counts (inside guard.py --exclusive)")
    r.add_argument("cases", nargs="+")
    r.add_argument("--bin", required=True, help="the hsolve binary")
    r.add_argument("--threads", default="1,2,4,8,16", help="thread counts, ascending")
    r.add_argument("--variant", action="append",
                   help="configuration: default | NAME[@T,T...]=JSON (repeatable; default: default alone)")
    r.add_argument("--reps", default="0,1,2", help="repetition indices, e.g. 0,1,2 or 2 to resume a part")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--root", default=os.environ.get("HS_DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")),
                   help="data root with cases/ (default $HS_DATA_ROOT or the repository's data/)")
    r.add_argument("--first-core", type=int, help="first core of the runs (default: the slot's first)")
    r.add_argument("--timeout", type=int, default=1500, help="per run, seconds")
    r.add_argument("--deadline", type=float, default=2400, help="no run starts after so many seconds")
    t = sub.add_parser("table", help="speedup, efficiency and phases per case, configuration and thread count")
    t.add_argument("run_dirs", nargs="+")
    t.add_argument("--order", nargs="*", default=[], help="case order of the tables (others after, sorted)")
    t.add_argument("--md")
    t.add_argument("--json")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    sys.exit(main())
