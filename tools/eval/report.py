"""Aggregate evaluation JSONs (tools/eval/evaluate.py) into a markdown table under results/.

Inputs are evaluation files or directories searched recursively for evaluation.json. Verdict counts
per solver head the page - every run counts, including runs that could not be evaluated (crashed,
killed, malformed result: 'fail' documents of tools/eval/sweep.py), and runs whose guard summary
shows a non-zero exit status are counted and marked; one row per (case, solver), sorted; failing
rows are explained below the table, followed by result-format issues and free-form notes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import case as cio
from tools.eval.evaluate import EVAL_FILE

DEFAULT_OUT = cio.REPO_ROOT / "results" / "eval_summary.md"

LEGEND = (
    "Metrics (tools/eval/evaluate.py, recomputed from the case matrices and the returned vectors): "
    "ref = reference kind used (discrete-exact / analytic / dense: certified complete; solver: ARPACK "
    "shift-invert, uncertified; pooled: accepted pairs of all solvers of this run, uncertified; "
    "n/c = not comparable, e.g. continuum-model values; - = none); solve = solver-reported time of the "
    "solve proper, like for like: hsolve `time_total_s` (all of hs::solve incl. its residual check and "
    "inertia certificate, after the CLI loaded the matrices), SLEPc baseline `time_setup_s + "
    "time_solve_s` (EPSSetUp incl. the MUMPS factorization + EPSSolve; without Python case loading, "
    "PETSc assembly and pair extraction), scipy baseline `time_total_s` (splu + ARPACK); "
    "wall / peak = wall time and peak RSS of the whole process tree measured by tools/run/guard.py, "
    "everything included (peak marked * is solver-reported, when no guard summary exists); eta = backward error "
    "||Ax - lambda Bx||_2 / ((||A||_2 + |lambda| ||B||_2) ||x||_2); rel res = ||Ax - lambda Bx||_2 / "
    "(|lambda| ||Bx||_2), which stays informative where |lambda| ||B||_2 ||x||_2 dwarfs the residual "
    "and eta tells little (e.g. a badly conditioned B); eig err = |lambda - mu|/|mu| / "
    "|lambda - mu|/(|mu| + ||A||_2/||B||_2) against the matched reference value (not against a pooled "
    "reference, which holds the solvers' own values); angle = largest "
    "principal angle between computed and reference eigenspaces per eigenvalue cluster (needs "
    "reference eigenvectors); B-orth = ||X^H B X - I||_2 (HEP/GHEP; not part of the verdict); "
    "completeness = status (found/expected, + pairs outside the target set): complete/incomplete "
    "against a certified reference, consistent/inconsistent against an uncertified one; claim = the "
    "solver's own success flag vs the verdict (over = success claimed but the evaluation fails, "
    "under = failure reported but the evaluation passes). Verdict pass = every eta <= tol, no "
    "duplicate or degenerate vectors, completeness not incomplete/inconsistent, no spurious eigenvalue, "
    "and the run was not killed; a run that left no result or one that cannot be evaluated is a fail "
    "('not evaluated'); '(exit N)' marks a run whose guard summary shows a non-zero exit status."
)
CLAIM = {"agrees": "ok", "overclaim": "**over**", "underclaim": "under", None: "-"}


def collect(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for p in paths:
        files += sorted(p.rglob(EVAL_FILE)) if p.is_dir() else [p]
    return [json.loads(f.read_text()) for f in files]


def _exit_code(d: dict[str, Any]) -> int | None:
    """The guard-measured exit status of the run when it is not 0 (None: 0 or unknown)."""
    rc = (d.get("measured") or {}).get("returncode")
    return rc if rc not in (None, 0) else None


def verdict_counts(docs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per solver: number of pass / fail verdicts (fail includes not evaluated runs), of runs that
    could not be evaluated, of overclaims / underclaims and of runs with a non-zero exit status."""
    out: dict[str, dict[str, int]] = {}
    for d in docs:
        c = out.setdefault(d["solver"], {"pass": 0, "fail": 0, "not_evaluated": 0, "overclaim": 0,
                                         "underclaim": 0, "nonzero_exit": 0})
        c["pass" if d["verdict"] == "pass" else "fail"] += 1
        c["not_evaluated"] += d.get("evaluated", True) is False
        c["nonzero_exit"] += _exit_code(d) is not None
        claim = d["summary"].get("solver_claim")
        if claim in ("overclaim", "underclaim"):
            c[claim] += 1
    return dict(sorted(out.items()))


def _num(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _target(t: dict[str, Any] | None) -> str:
    if t is None:
        return "-"
    return f"[{t['lower']:g}, {t['upper']:g}]" if t["mode"] == "interval" else str(t["k"])


def _solve_time(d: dict[str, Any]) -> float | None:
    """solve_time_s (like for like, see LEGEND); documents written before it existed: time_total_s."""
    return d["solve_time_s"] if "solve_time_s" in d else d.get("time_total_s")


def _verdict(d: dict[str, Any]) -> str:
    text = "pass" if d["verdict"] == "pass" else ("**fail**" if d.get("evaluated", True) else "**fail** (not evaluated)")
    rc = _exit_code(d)
    return text if rc is None else f"{text} (exit {rc})"


def _counts(c: dict[str, int]) -> str:
    extra = [f"{c['not_evaluated']} not evaluated" if c["not_evaluated"] else "",
             f"{c['overclaim']} overclaim(s)" if c["overclaim"] else "",
             f"{c['underclaim']} underclaim(s)" if c["underclaim"] else "",
             f"{c['nonzero_exit']} non-zero exit(s)" if c["nonzero_exit"] else ""]
    return f"{c['pass']} pass / {c['fail']} fail" + "".join(f", {e}" for e in extra if e)


def _reference(r: dict[str, Any] | None) -> str:
    if r is None:
        return "-"
    return r["kind"] if r.get("comparable", True) else f"{r['kind']} (n/c)"


def _peak_mib(d: dict[str, Any]) -> str:
    measured = (d.get("measured") or {}).get("peak_rss_bytes")
    if measured is not None:
        return f"{measured / 2**20:.0f}"
    reported = d.get("peak_rss_bytes")
    return f"{reported / 2**20:.0f}*" if reported else "-"


def _completeness(c: dict[str, Any]) -> str:
    exp = c.get("expected")
    if exp is None:
        return c["status"]
    exp_s = f"{exp[0]}-{exp[1]}" if isinstance(exp, list) else str(exp)
    extra = f", +{len(c['extra'])} extra" if c.get("extra") else ""
    return f"{c['status']} ({c['found']}/{exp_s}{extra})"


def render(docs: list[dict[str, Any]], title: str, notes: list[str] | None = None,
           preamble: list[str] | None = None) -> str:
    """Markdown summary. `preamble` lines (e.g. provenance) go below the title; `notes` (e.g. runs
    killed by the guard, which leave no evaluation) are listed verbatim at the end."""
    docs = sorted(docs, key=lambda d: (d["case"], d["solver"]))
    counts = "; ".join(f"**{s}**: {_counts(c)}" for s, c in verdict_counts(docs).items())
    lines = [f"# {title}", ""] + (list(preamble) + [""] if preamble else [])
    lines += [f"Verdicts - {counts}", "", LEGEND, "",
              "| case | solver | problem | n | k | ref | solve [s] | wall [s] | peak [MiB] | max eta "
              "| max rel res | max eig err (rel / scaled) | max angle | B-orth | completeness | claim | verdict |",
              "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for d in docs:
        s = d["summary"]
        lines.append(
            f"| {d['case']} | {d['solver']} | {d['problem'] or '-'} | {_num(d['n'], 'd')} | {_target(d['target'])} "
            f"| {_reference(d.get('reference'))} | {_num(_solve_time(d), '.3g')} "
            f"| {_num((d.get('measured') or {}).get('wall_s'), '.3g')} | {_peak_mib(d)} "
            f"| {_num(s['max_backward_error'], '.1e')} | {_num(s['max_rel_residual'], '.1e')} "
            f"| {_num(s['max_eig_rel_error'], '.1e')} / {_num(s['max_eig_scaled_error'], '.1e')} "
            f"| {_num(s['max_subspace_angle'], '.1e')} | {_num(s['b_orthogonality'], '.1e')} "
            f"| {_completeness(d['completeness'])} | {CLAIM[s.get('solver_claim')]} | {_verdict(d)} |")
    failing = [d for d in docs if d["verdict"] != "pass"]
    if failing:
        lines += ["", "## Failures", ""]
        lines += [f"- {d['case']} / {d['solver']}: " + "; ".join(d["reasons"]) for d in failing]
    issues = [(d, i) for d in docs for i in (d.get("format") or {}).get("issues", [])]
    if issues:
        lines += ["", "## Result format issues (docs/FORMATS.md §3; not part of the verdict)", ""]
        lines += [f"- {d['case']} / {d['solver']}: {i}" for d, i in issues]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate evaluation JSONs into a markdown table.")
    ap.add_argument("inputs", nargs="+", type=Path, help=f"evaluation JSON files or directories ({EVAL_FILE})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown file (default: %(default)s)")
    ap.add_argument("--title", default="Evaluation summary")
    ap.add_argument("--note", action="append", default=[],
                    help="line for a Notes section, e.g. a run that left no evaluation (repeatable)")
    args = ap.parse_args(argv)
    docs = collect(args.inputs)
    if not docs:
        ap.error("no evaluation files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(docs, args.title, args.note))
    print(f"{len(docs)} evaluation(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
