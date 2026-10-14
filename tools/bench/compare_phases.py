#!/usr/bin/env python3
"""Per-phase medians of a tools/bench/compare.py campaign, from the runs' own reports (P9).

compare.py's table gives the MUMPS phases (analysis, numeric, solves) of every solver; this adds
hsolve's exclusive phases from each run's report.json - setup, factor (analyses + main numeric
factorization), iterate, certificate (the inertia counts), verify - and SLEPc's -log_view split
(setup = MUMPS analysis + numeric factorization, solve, eigenvectors), as medians [min-max] over the
valid repetitions, so the two solvers' time can be compared phase by phase.

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


def run_phases(case_dir: Path, run: Path, solver: str) -> dict | None:
    """Phase times of one run (None when it left no report, or its evaluation failed)."""
    rep_file, ev_file = run / "report.json", run / "evaluation.json"
    if not rep_file.exists():
        return None
    if ev_file.exists() and json.loads(ev_file.read_text()).get("verdict") != "pass":
        return None
    rep = json.loads(rep_file.read_text())
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
    """{case: {solver: {phase: [values]}}} over the valid runs."""
    doc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for case_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for run in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            m = RUN_DIR.match(run.name)
            if not m:
                continue
            solver = m.group("solver")
            ph = run_phases(case_dir, run, solver)
            if ph is None:
                continue
            for k, v in ph.items():
                if v is not None:
                    doc[case_dir.name][solver][k].append(v)
    return doc


def fmt(vals: list) -> str:
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return ", ".join(sorted(set(map(str, vals)))) or "-"
    med = statistics.median(nums)
    return f"{med:.2f} [{min(nums):.2f}-{max(nums):.2f}]" if len(nums) > 1 else f"{med:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    doc = collect(a.raw)
    for case, solvers in doc.items():
        keys = sorted({k for s in solvers.values() for k in s})
        print(f"**{case}** (valid runs; medians [min-max], seconds)\n")
        print("| phase | " + " | ".join(solvers) + " |")
        print("|---|" + "---|" * len(solvers))
        for k in keys:
            print(f"| {k} | " + " | ".join(fmt(solvers[s].get(k, [])) for s in solvers) + " |")
        print()
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
