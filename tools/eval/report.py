"""Aggregate evaluation JSONs (tools/eval/evaluate.py) into a markdown table under results/.

Inputs are evaluation files or directories searched recursively for evaluation.json. One row per
(case, solver), sorted; failing rows are explained below the table.
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
    "Metrics (tools/eval/evaluate.py): time = solver-reported `time_total_s`; peak memory = solver-reported "
    "`peak_rss_bytes`; backward error eta = ||Ax - lambda Bx||_2 / ((||A||_2 + |lambda| ||B||_2) ||x||_2) "
    "recomputed independently; eigenvalue error vs the reference as relative |lambda - mu|/|mu| and scaled "
    "|lambda - mu|/(|mu| + ||A||_2/||B||_2); completeness found/expected, + computed pairs outside the "
    "target set (certified only against a complete reference; consistent/inconsistent = against an "
    "uncertified solver reference)."
)


def collect(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for p in paths:
        files += sorted(p.rglob(EVAL_FILE)) if p.is_dir() else [p]
    return [json.loads(f.read_text()) for f in files]


def _num(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _target(t: dict[str, Any]) -> str:
    return f"[{t['lower']:g}, {t['upper']:g}]" if t["mode"] == "interval" else str(t["k"])


def _completeness(c: dict[str, Any]) -> str:
    exp = c.get("expected")
    if exp is None:
        return c["status"]
    exp_s = f"{exp[0]}-{exp[1]}" if isinstance(exp, list) else str(exp)
    extra = f", +{len(c['extra'])} extra" if c.get("extra") else ""
    return f"{c['status']} ({c['found']}/{exp_s}{extra})"


def render(docs: list[dict[str, Any]], title: str) -> str:
    docs = sorted(docs, key=lambda d: (d["case"], d["solver"]))
    lines = [f"# {title}", "", LEGEND, "",
             "| case | solver | problem | n | k | time [s] | peak mem [MiB] | max backward error "
             "| max eig err (rel / scaled) | completeness | verdict |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for d in docs:
        s = d["summary"]
        mem = d.get("peak_rss_bytes")
        lines.append(
            f"| {d['case']} | {d['solver']} | {d['problem']} | {d['n']} | {_target(d['target'])} "
            f"| {_num(d.get('time_total_s'), '.3g')} | {_num(mem / 2**20 if mem else None, '.0f')} "
            f"| {_num(s['max_backward_error'], '.1e')} "
            f"| {_num(s['max_eig_rel_error'], '.1e')} / {_num(s['max_eig_scaled_error'], '.1e')} "
            f"| {_completeness(d['completeness'])} | {'pass' if d['verdict'] == 'pass' else '**fail**'} |")
    failing = [d for d in docs if d["verdict"] != "pass"]
    if failing:
        lines += ["", "## Failures", ""]
        lines += [f"- {d['case']} / {d['solver']}: " + "; ".join(d["reasons"]) for d in failing]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate evaluation JSONs into a markdown table.")
    ap.add_argument("inputs", nargs="+", type=Path, help=f"evaluation JSON files or directories ({EVAL_FILE})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown file (default: %(default)s)")
    ap.add_argument("--title", default="Evaluation summary")
    args = ap.parse_args(argv)
    docs = collect(args.inputs)
    if not docs:
        ap.error("no evaluation files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(docs, args.title))
    print(f"{len(docs)} evaluation(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
