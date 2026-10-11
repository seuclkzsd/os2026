"""Evaluate every result directory of a raw benchmark run and write the markdown summary.

Raw run layout (tools/bench/compare.py and the experiment scripts):
    RUN/<case>/<solver>/              result directory (docs/FORMATS.md §3)
    RUN/<case>/guard-<solver>.json    tools/run/guard.py --summary of that solver's run
    RUN/<case>/guard.json             the same for a case with a single solver directory
Every (case, solver) run is a row of the report, so the verdict counts cover crashes too: a guard
summary without a result directory (crashed, killed, timed out), a result that cannot be evaluated
(e.g. malformed files) and a case that cannot be loaded all become 'fail' documents
(evaluate.not_evaluated_document, evaluated=false) with the reason. A run the guard killed, or a
signal ended, fails even if it left a result (evaluate.evaluate); other non-zero exit statuses are
the solver's own failure report (hsolve: 1 = not successful) and are marked in the report.

Per case the matrices are loaded and their norms estimated once for all solvers. Cases without a
comparable stored reference but with results of two or more solvers are evaluated against the pooled
accepted pairs of all of them (evaluate.pooled_reference): uncertified, but a solver that missed an
eigenvalue another one found is caught.

Evaluation JSONs go to EVAL_DIR/<case>/<solver>/evaluation.json (default: into each result
directory, like evaluate.py's CLI; a run without a result directory then has no JSON, only its row).
The sweep is deterministic for a given raw run, data root and thread count, so the report's preamble
gives the command that regenerates both. Must run under tools/run/guard.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
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


def _evaluate_case(name: str, runs: list[Run], root: Path, pool: bool) -> list[tuple[Run, dict[str, Any]]]:
    """Documents of one case's runs, in the order of `runs`."""
    try:
        c = cio.load_case(name, root=root)
        norms = evaluate.estimate_norms(c.A, c.B)
    except Exception as err:  # the case itself cannot be loaded: every run of it fails
        why = f"case cannot be evaluated: {type(err).__name__}: {err}"
        return [(r, evaluate.not_evaluated_document(name, r.solver, why, result_dir=r.result_dir, guard=r.guard))
                for r in runs]
    docs: dict[int, dict[str, Any]] = {}
    for i, r in enumerate(runs):
        if r.result_dir is None:
            docs[i] = evaluate.not_evaluated_document(c, r.solver, "no result directory", guard=r.guard)
            continue
        try:
            docs[i] = evaluate.evaluate(c, r.result_dir, root=root, solver=r.solver, norms=norms, guard=r.guard)
        except Exception as err:  # a malformed result fails, the sweep goes on
            docs[i] = evaluate.not_evaluated_document(c, r.solver, f"evaluation failed: {type(err).__name__}: {err}",
                                                      result_dir=r.result_dir, guard=r.guard)
    evaluated = [i for i, d in docs.items() if d["evaluated"]]
    uncompared = all(docs[i]["reference"] is None or not docs[i]["reference"]["comparable"] for i in evaluated)
    if pool and len(evaluated) >= 2 and uncompared:
        pooled = evaluate.pooled_reference([docs[i] for i in evaluated], norms.scale)
        if pooled is not None:
            for i in evaluated:
                r = runs[i]
                docs[i] = evaluate.evaluate(c, r.result_dir, root=root, solver=r.solver, norms=norms,
                                            reference=pooled, guard=r.guard)
    return [(r, docs[i]) for i, r in enumerate(runs)]


def sweep(run_dir: Path, *, root: Path = cio.DEFAULT_ROOT, eval_dir: Path | None = None,
          cases: list[str] | None = None, solvers: list[str] | None = None, pool: bool = True,
          log: Callable[[str], None] = lambda line: None) -> list[dict[str, Any]]:
    """Evaluate a raw run: one evaluation document per (case, solver) run, written as JSON."""
    runs = discover(run_dir, cases, solvers)
    docs: list[dict[str, Any]] = []
    for name in dict.fromkeys(r.case for r in runs):
        for r, doc in _evaluate_case(name, [r for r in runs if r.case == name], Path(root), pool):
            out_dir = Path(eval_dir) / name / r.solver if eval_dir else r.result_dir
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / evaluate.EVAL_FILE).write_text(json.dumps(doc, indent=2) + "\n")
            s = doc["summary"]
            log(f"{name} [{r.solver}]: {doc['verdict']} ref={doc['reference'] and doc['reference']['kind']} "
                f"max_eta={s['max_backward_error']} completeness={doc['completeness']['status']}"
                + "".join(f"\n  - {reason}" for reason in doc["reasons"]))
            docs.append(doc)
    return docs


def regenerate_command(argv: list[str]) -> str:
    """The command line that reproduces this sweep (and the evaluation JSONs), minus the free-text
    --title / --note options, which only change the page text."""
    keep, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in ("--title", "--note"):
            skip = True
        elif not a.startswith(("--title=", "--note=")):
            keep.append(a)
    cpus, spans = sorted(os.sched_getaffinity(0)), []   # the guard's pinning, as a cpulist
    for c in cpus:
        if spans and c == spans[-1][1] + 1:
            spans[-1][1] = c
        else:
            spans.append([c, c])
    cores = ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)
    return (f"python3 tools/run/guard.py --skip-busy-check --cores {cores} --mem-gb <GiB> -- "
            "python -m tools.eval.sweep " + shlex.join(keep))


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
    ap.add_argument("--note", action="append", default=[], help="extra line for the Notes section (repeatable)")
    argv = sys.argv[1:] if argv is None else argv
    args = ap.parse_args(argv)
    common.require_guard("sweep")
    docs = sweep(args.run, root=args.root, eval_dir=args.eval_dir, cases=args.cases, solvers=args.solvers,
                 pool=not args.no_pool, log=lambda line: print(line, flush=True))
    if not docs:
        ap.error(f"no results found under {args.run}")
    out = args.out or cio.REPO_ROOT / "results" / f"eval_{args.run.name}.md"
    where = (f"`{args.eval_dir.resolve()}/<case>/<solver>/{evaluate.EVAL_FILE}`" if args.eval_dir
             else f"each result directory (`<run>/<case>/<solver>/{evaluate.EVAL_FILE}`)")
    preamble = [f"Run `{args.run}`, data root `{args.root}`; {len(docs)} evaluation(s) by tools/eval/sweep.py. "
                f"Evaluation JSONs (not under version control) are written to {where}; the sweep is deterministic "
                f"for a given raw run, data root and thread count, so this command regenerates them and the "
                f"table (add the page's --title/--note options to reproduce its text exactly): "
                f"`{regenerate_command(argv)}`."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.render(docs, args.title or f"Evaluation of {args.run.name}", args.note, preamble))
    for solver, c in report.verdict_counts(docs).items():
        print(f"{solver}: {c['pass']} pass, {c['fail']} fail ({c['not_evaluated']} not evaluated), "
              f"{c['overclaim']} overclaim, {c['underclaim']} underclaim, {c['nonzero_exit']} non-zero exit")
    print(f"{len(docs)} evaluation(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
