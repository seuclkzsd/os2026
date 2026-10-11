"""Evaluate every result directory of a raw benchmark run and write the markdown summary.

Raw run layout (tools/bench/compare.py and the experiment scripts):
    RUN/<case>/<solver>/              result directory (docs/FORMATS.md §3)
    RUN/<case>/guard-<solver>.json    tools/run/guard.py --summary of that solver's run
    RUN/<case>/guard.json             the same for a case with a single solver directory
A guard summary without a result directory (killed, crashed, timed out) becomes a note of the
report, as does a result directory that cannot be evaluated or a run with a non-zero exit status.

Per case the matrices are loaded and their norms estimated once for all solvers. Cases without a
comparable stored reference but with results of two or more solvers are evaluated against the pooled
accepted pairs of all of them (evaluate.pooled_reference): uncertified, but a solver that missed an
eigenvalue another one found is caught.

Evaluation JSONs go to EVAL_DIR/<case>/<solver>/evaluation.json (default: into each result
directory, like evaluate.py's CLI). Must run under tools/run/guard.py.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools import case as cio
from tools.eval import common, evaluate, report


@dataclass
class Run:
    case: str
    solver: str
    result_dir: Path | None          # None: the run left no result directory
    guard: dict[str, Any] | None     # tools/run/guard.py summary, if one was written


def discover(run_dir: Path, cases: list[str] | None = None, solvers: list[str] | None = None) -> list[Run]:
    """Every (case, solver) of a raw run, from result directories and guard summaries alike."""
    runs = []
    for cdir in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        if cases and cdir.name not in cases:
            continue
        subdirs = sorted(p for p in cdir.iterdir() if p.is_dir())
        results = {p.name: p for p in subdirs if (p / "report.json").exists()}
        guards: dict[str, dict[str, Any]] = {}
        for g in sorted(cdir.glob("guard*.json")):
            if g.stem.startswith("guard-"):
                solver = g.stem[len("guard-"):]
            else:   # guard.json belongs to the case's only solver directory
                solver = subdirs[0].name if len(subdirs) == 1 else "unknown"
            guards[solver] = json.loads(g.read_text())
        for solver in sorted(set(results) | set(guards)):
            if not solvers or solver in solvers:
                runs.append(Run(cdir.name, solver, results.get(solver), guards.get(solver)))
    return runs


def _status(guard: dict[str, Any] | None) -> str:
    if guard is None:
        return "no guard summary"
    killed = f", killed: {guard['killed']}" if guard.get("killed") else ""
    return f"exit status {guard.get('returncode')}{killed}, wall {guard.get('wall_s')} s"


def _evaluate_case(name: str, runs: list[Run], root: Path, pool: bool,
                   notes: list[str]) -> list[tuple[Run, dict[str, Any]]]:
    c = cio.load_case(name, root=root)
    norms = evaluate.estimate_norms(c.A, c.B)
    done = []
    for r in runs:
        try:
            done.append((r, evaluate.evaluate(c, r.result_dir, root=root, solver=r.solver, norms=norms,
                                              guard=r.guard)))
        except Exception as err:  # a malformed result is reported, the sweep goes on
            notes.append(f"{name} / {r.solver}: evaluation failed: {type(err).__name__}: {err}")
    uncompared = all(d["reference"] is None or not d["reference"]["comparable"] for _, d in done)
    if pool and len(done) >= 2 and uncompared:
        pooled = evaluate.pooled_reference([d for _, d in done], norms.scale)
        if pooled is not None:
            done = [(r, evaluate.evaluate(c, r.result_dir, root=root, solver=r.solver, norms=norms,
                                          reference=pooled, guard=r.guard)) for r, _ in done]
    return done


def sweep(run_dir: Path, *, root: Path = cio.DEFAULT_ROOT, eval_dir: Path | None = None,
          cases: list[str] | None = None, solvers: list[str] | None = None, pool: bool = True,
          log: Callable[[str], None] = lambda line: None) -> tuple[list[dict[str, Any]], list[str]]:
    """Evaluate a raw run; returns (evaluation documents, report notes) and writes the documents."""
    runs = discover(run_dir, cases, solvers)
    docs: list[dict[str, Any]] = []
    notes: list[str] = []
    for name in dict.fromkeys(r.case for r in runs):
        case_runs = [r for r in runs if r.case == name]
        for r in case_runs:
            if r.result_dir is None:
                notes.append(f"{name} / {r.solver}: no result directory ({_status(r.guard)})")
            elif r.guard is not None and (r.guard.get("returncode") != 0 or r.guard.get("killed")):
                notes.append(f"{name} / {r.solver}: result written but the run ended with {_status(r.guard)}")
        present = [r for r in case_runs if r.result_dir is not None]
        if not present:
            continue
        try:
            done = _evaluate_case(name, present, Path(root), pool, notes)
        except Exception as err:  # the case itself cannot be loaded
            notes.append(f"{name}: case cannot be evaluated: {type(err).__name__}: {err}")
            continue
        for r, doc in done:
            out = (Path(eval_dir) / name / r.solver if eval_dir else r.result_dir) / evaluate.EVAL_FILE
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2) + "\n")
            s = doc["summary"]
            log(f"{name} [{r.solver}]: {doc['verdict']} ref={doc['reference'] and doc['reference']['kind']} "
                f"max_eta={s['max_backward_error']} completeness={doc['completeness']['status']}"
                + "".join(f"\n  - {reason}" for reason in doc["reasons"]))
            docs.append(doc)
    return docs, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate every result directory of a raw run; write a markdown summary.")
    ap.add_argument("run", type=Path, help="raw run directory RUN/<case>/<solver>/")
    ap.add_argument("--root", type=Path, default=cio.DEFAULT_ROOT, help="data root (default: %(default)s)")
    ap.add_argument("--eval-dir", type=Path, help="where evaluation JSONs go (default: inside each result directory)")
    ap.add_argument("--out", type=Path, help="markdown summary (default: results/eval_<run name>.md)")
    ap.add_argument("--title", help="report title (default: Evaluation of <run name>)")
    ap.add_argument("--cases", nargs="+", help="only these cases")
    ap.add_argument("--solvers", nargs="+", help="only these solver directories")
    ap.add_argument("--no-pool", action="store_true", help="no pooled cross-solver reference for unreferenced cases")
    args = ap.parse_args(argv)
    common.require_guard("sweep")
    docs, notes = sweep(args.run, root=args.root, eval_dir=args.eval_dir, cases=args.cases, solvers=args.solvers,
                        pool=not args.no_pool, log=lambda line: print(line, flush=True))
    if not docs and not notes:
        ap.error(f"no results found under {args.run}")
    out = args.out or cio.REPO_ROOT / "results" / f"eval_{args.run.name}.md"
    preamble = [f"Run `{args.run}`, data root `{args.root}`; {len(docs)} evaluation(s) by "
                f"`python -m tools.eval.sweep` (tools/eval/sweep.py)."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.render(docs, args.title or f"Evaluation of {args.run.name}", notes, preamble))
    for solver, c in report.verdict_counts(docs).items():
        print(f"{solver}: {c['pass']} pass, {c['fail']} fail, {c['overclaim']} overclaim, {c['underclaim']} underclaim")
    print(f"{len(docs)} evaluation(s), {len(notes)} note(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
