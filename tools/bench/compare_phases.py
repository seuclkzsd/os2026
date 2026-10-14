#!/usr/bin/env python3
"""Per-phase medians of a tools/bench/compare.py campaign, from the runs' own reports (P9).

compare.py's table gives the MUMPS phases (analysis, numeric, solves) of every solver; this adds
hsolve's exclusive phases from each run's report.json - setup, factor (analyses + main numeric
factorization), iterate, certificate (the inertia counts), verify - and SLEPc's -log_view split
(setup = MUMPS analysis + numeric factorization, solve, eigenvectors), as medians [min-max] over the
valid repetitions, so the two solvers' time can be compared phase by phase.

A run is valid exactly when compare.py counts it valid, judged from the files it leaves: its guard
summary (<case>/guard-<run>.json) shows exit status 0 and no kill, its report.json claims success,
and its evaluation.json (tools/eval) has the verdict "pass". compare.py writes evaluation.json only
when tools/eval could evaluate the result, so a run without one is invalid, as is a run without a
guard summary. The first row of each table counts the valid runs of each solver.

Tests: tests/py/test_bench_mpi.py.

usage: python3 tools/bench/compare_phases.py RESULTS_RAW_DIR/TAG [--json OUT.json]
       (RESULTS_RAW_DIR/TAG = <work tree>/results/raw/<compare tag>, holding <case>/<solver>-r<rep>/)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.bench import slepc_log  # noqa: E402

HSOLVE_KEYS = ("time_total_s", "time_setup_s", "time_factor_s", "time_iterate_s", "time_certificate_s",
               "time_verify_s")
RUN_DIR = re.compile(r"^(?P<solver>[a-z-]+?)(?:-r(?P<rep>\d+))?$")


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_valid(case_dir: Path, run: Path) -> bool:
    """compare.py's validity of one run (module docstring): guard exit 0 and no kill, report.json
    claiming success, evaluation.json with verdict "pass"; a missing or unreadable file is invalid."""
    guard = _json(case_dir / f"guard-{run.name}.json")
    rep = _json(run / "report.json")
    ev = _json(run / "evaluation.json")
    if guard is None or rep is None or ev is None:
        return False
    return (guard.get("returncode") == 0 and not guard.get("killed") and bool(rep.get("success"))
            and ev.get("verdict") == "pass")


def run_phases(case_dir: Path, run: Path, solver: str) -> dict | None:
    """Phase times of one run (None when it is not valid: run_valid)."""
    if not run_valid(case_dir, run):
        return None
    rep = json.loads((run / "report.json").read_text())
    if solver.startswith("hsolve"):
        fs = rep.get("factor_stats") or {}
        out = {k: rep.get(k) for k in HSOLVE_KEYS}
        out.update(analyze_s=fs.get("analyze_s"), factorize_s=fs.get("factorize_s"), solve_s=fs.get("solve_s"),
                   rhs_solved=fs.get("rhs_solved"),
                   certificate_method=(rep.get("certificate") or {}).get("method"))
        return out
    ph = slepc_log.read_phases(case_dir / f"log_view-{run.name}.txt")
    out = {"time_total_s": ph.get("solver_s") if ph else None}
    if ph:
        out.update({k: ph.get(k) for k in ("setup_s", "eps_solve_s", "vectors_s", "analysis_s", "numeric_s",
                                           "solve_s", "solves", "ortho_s")})
    return out


def collect(raw: Path) -> dict:
    """{case: {solver: {"runs": all runs, "valid": valid runs, "phases": {phase: [values over the
    valid runs]}}}}."""
    doc: dict = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "valid": 0, "phases": defaultdict(list)}))
    for case_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for run in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            m = RUN_DIR.match(run.name)
            if not m:
                continue
            entry = doc[case_dir.name][m.group("solver")]
            entry["runs"] += 1
            ph = run_phases(case_dir, run, m.group("solver"))
            if ph is None:
                continue
            entry["valid"] += 1
            for k, v in ph.items():
                if v is not None:
                    entry["phases"][k].append(v)
    return doc


def fmt(vals: list) -> str:
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return ", ".join(sorted(set(map(str, vals)))) or "-"
    med = statistics.median(nums)
    return f"{med:.2f} [{min(nums):.2f}-{max(nums):.2f}]" if len(nums) > 1 else f"{med:.2f}"


def _solver_key(name: str) -> tuple[int, str]:
    """compare.py's solver order (hsolve variants, then slepc), unknown names last."""
    known = ("hsolve", "hsolve-auto", "hsolve-nocert", "slepc")
    return (known.index(name) if name in known else len(known), name)


def render(doc: dict) -> str:
    out = []
    for case, by_solver in doc.items():
        solvers = [by_solver[s] for s in sorted(by_solver, key=_solver_key)]
        keys = sorted({k for s in solvers for k in s["phases"]})
        out.append(f"**{case}** (valid runs; medians [min-max], seconds)\n")
        out.append("| phase | " + " | ".join(sorted(by_solver, key=_solver_key)) + " |")
        out.append("|---|" + "---|" * len(solvers))
        out.append("| valid runs | " + " | ".join(f"{s['valid']}/{s['runs']}" for s in solvers) + " |")
        for k in keys:
            out.append(f"| {k} | " + " | ".join(fmt(s["phases"].get(k, [])) for s in solvers) + " |")
        out.append("")
    return "\n".join(out) + ("\n" if out else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    doc = collect(a.raw)
    print(render(doc), end="")
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
