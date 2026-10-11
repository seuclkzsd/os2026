"""Parser of PETSc's ASCII -log_view report, reduced to the phases of a SLEPc shift-invert solve.

Pure Python (no numpy/PETSc): used by tools/bench/tune_slepc.py and compare.py in the project env
and testable without the os2026-ref env. The report lists, per log stage, one line per event:

    MatSolve              31 1.0 6.9520e-02 1.0 6.78e+07 1.5 7.8e+01 2.1e+04 5.0e+00 23 54 ...
    <name>             count ratio  max-time ratio  max-flop ratio  messages len reductions ...

Times are the maximum over the MPI ranks and INCLUSIVE (MatMult of the shift-invert operator
contains its MatSolve, EPSSolve contains everything below it), so phases are read from disjoint
events, never summed across nesting levels. tools/bench/baseline_slepc.py runs each phase in its own
stage (hs_setup = EPSSetUp, hs_solve = EPSSolve, hs_vectors = the eigenvector computation EPSSolve
defers, hs_output = residual checks and gathering), so:

    analysis_s      MatCholFctrSym + MatLUFactorSym   (MUMPS analysis incl. ordering)
    numeric_s       MatCholFctrNum + MatLUFactorNum   (numeric factorizations; count in factorizations)
    solves          MatSolve count, solve_s its time  (one per operator application, 1 RHS each)
    ortho_s         BVOrthogonalize + BVOrthogonalizeV (+ BVOrthogonalizeVec where present)
    setup_s         stage hs_setup (EPSSetUp event when the stage is absent)
    eps_solve_s     stage hs_solve (EPSSolve event when absent)
    vectors_s       stage hs_vectors (0 when absent)
    solver_s        setup_s + eps_solve_s + vectors_s: the solver time compared with hsolve's
                    time_total_s (tools/bench/compare.py)
Counts and times of the per-event phases are summed over the solver stages (hs_setup, hs_solve,
hs_vectors) - or over all stages of a log without them - so the output stage never counts.
"""
from __future__ import annotations

import re
from typing import Any

SOLVER_STAGES = ("hs_setup", "hs_solve", "hs_vectors")
ANALYSIS_EVENTS = ("MatCholFctrSym", "MatLUFactorSym")
NUMERIC_EVENTS = ("MatCholFctrNum", "MatLUFactorNum")
SOLVE_EVENTS = ("MatSolve",)
ORTHO_EVENTS = ("BVOrthogonalize", "BVOrthogonalizeV", "BVOrthogonalizeVec")

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_STAGE_HEADER = re.compile(r"^--- Event Stage (\d+): (.+?)\s*$")
_STAGE_SUMMARY = re.compile(rf"^\s*(\d+):\s+(.+?):\s+({_FLOAT})\s+({_FLOAT})%")
_EVENT = re.compile(rf"^(\S+)\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})")
_TOTAL_TIME = re.compile(rf"^Time \(sec\):\s+({_FLOAT})")
_PROCS = re.compile(r" with (\d+) process(?:es)?,")
_THREADS = re.compile(r"^Using (\d+) OpenMP threads")


def parse_log_view(text: str) -> dict[str, Any]:
    """Structured -log_view report: {"time_s", "ranks", "omp_threads", "stages": {name: {"index",
    "time_s", "percent"}}, "events": {stage: {event: {"count", "count_ratio", "time_s",
    "time_ratio", "flop"}}}}. Raises ValueError when the text holds no event table."""
    out: dict[str, Any] = {"time_s": None, "ranks": None, "omp_threads": None, "stages": {}, "events": {}}
    stage = None
    in_summary = False
    for line in text.splitlines():
        if line.startswith("Object Type"):   # the object tables repeat the stage headers: done
            break
        if out["time_s"] is None and (m := _TOTAL_TIME.match(line)):
            out["time_s"] = float(m.group(1))
            continue
        if out["ranks"] is None and (m := _PROCS.search(line)):
            out["ranks"] = int(m.group(1))
        if out["omp_threads"] is None and (m := _THREADS.match(line)):
            out["omp_threads"] = int(m.group(1))
            continue
        if line.startswith("Summary of Stages:"):
            in_summary = True
            continue
        if in_summary:
            if m := _STAGE_SUMMARY.match(line):
                out["stages"][m.group(2).strip()] = {"index": int(m.group(1)), "time_s": float(m.group(3)),
                                                     "percent": float(m.group(4))}
                continue
            if out["stages"] and not line.strip().startswith(("Avg", "-")) and line.strip():
                in_summary = False
        if m := _STAGE_HEADER.match(line):
            stage = m.group(2)
            out["events"].setdefault(stage, {})
            continue
        if stage is not None and (m := _EVENT.match(line)):
            out["events"][stage][m.group(1)] = {
                "count": int(m.group(2)), "count_ratio": float(m.group(3)), "time_s": float(m.group(4)),
                "time_ratio": float(m.group(5)), "flop": float(m.group(6))}
    if not out["events"]:
        raise ValueError("no -log_view event table found")
    return out


def _sum(events_by_stage: dict[str, dict[str, Any]], stages: list[str], names: tuple[str, ...],
         field: str) -> float:
    return sum(events_by_stage.get(s, {}).get(e, {}).get(field, 0) for s in stages for e in names)


def phases(log: dict[str, Any]) -> dict[str, Any]:
    """Per-phase times and counts (module docstring) of a parse_log_view() result."""
    ev = log["events"]
    have_stages = all(s in log["stages"] for s in SOLVER_STAGES)
    stages = list(SOLVER_STAGES) if have_stages else list(ev)

    def stage_time(name: str, event: str | None) -> float:
        if name in log["stages"]:
            return log["stages"][name]["time_s"]
        return _sum(ev, list(ev), (event,), "time_s") if event else 0.0

    setup = stage_time("hs_setup", "EPSSetUp")
    eps_solve = stage_time("hs_solve", "EPSSolve")
    vectors = stage_time("hs_vectors", None)
    return {
        "analysis_s": _sum(ev, stages, ANALYSIS_EVENTS, "time_s"),
        "analyses": int(_sum(ev, stages, ANALYSIS_EVENTS, "count")),
        "numeric_s": _sum(ev, stages, NUMERIC_EVENTS, "time_s"),
        "factorizations": int(_sum(ev, stages, NUMERIC_EVENTS, "count")),
        "solve_s": _sum(ev, stages, SOLVE_EVENTS, "time_s"),
        "solves": int(_sum(ev, stages, SOLVE_EVENTS, "count")),
        "ortho_s": _sum(ev, stages, ORTHO_EVENTS, "time_s"),
        "setup_s": setup, "eps_solve_s": eps_solve, "vectors_s": vectors,
        "solver_s": setup + eps_solve + vectors,
        "staged": have_stages, "ranks": log.get("ranks"), "omp_threads": log.get("omp_threads"),
    }


def read_phases(path) -> dict[str, Any] | None:
    """phases() of a -log_view file, None when it is missing or holds no event table."""
    try:
        with open(path) as f:
            return phases(parse_log_view(f.read()))
    except (OSError, ValueError):
        return None
