#!/usr/bin/env python3
"""Head-to-head benchmark: hsolve vs reference solvers on the same cases, cores and tolerance.

Every solver run goes through tools/run/guard.py (core pinning, RSS watchdog, peak memory incl. all
MPI ranks), so wall time and peak RSS are measured the same way for all solvers. Solvers (--solvers):
  hsolve        default configuration (inertia certificate on)
  hsolve-nocert inertia certificate off (pure speed; same accuracy, no completeness proof)
  slepc         SLEPc Krylov-Schur + shift-invert + MUMPS (tools/bench/baseline_slepc.py, env os2026-ref)
SLEPc configuration (--slepc-profile):
  default       1 rank x P threads, SLEPc/MUMPS defaults (--slepc-ranks R: R ranks x P/R threads)
  best          the tuned profile of the case for this core count (tools/bench/slepc_profiles.json,
                written by tools/bench/tune_slepc.py; an interval profile only for the interval it was
                tuned for), else the case class's profile, else the default. The table says which:
                a class profile is a heuristic that was never tuned or validated on that case, and
                the default is untuned, so a speedup against either is not a speedup against the
                tuned baseline (the speedup table marks it).
Solver time, equal scope for both (the comparison's main column):
  hsolve        report time_total_s: all of hs::solve after the CLI loaded the matrices (norm
                estimates, analysis, factorizations, iterations, Rayleigh-Ritz, residual checks,
                inertia certificate)
  slepc         -log_view setup + solve + eigenvector computation (EPSSetUp incl. MUMPS analysis
                and factorization, EPSSolve, EPSComputeVectors incl. purification;
                tools/bench/slepc_log.py), excluding Python/MPI start-up, case loading and output
Phase columns: analysis / numeric factorization (count) / solves (count of right-hand sides, time) /
orthogonalization (SLEPc BVOrthogonalize*) / hsolve's iterate and verify phases. Peak memory: the
guard's process-tree peak (includes loading; for SLEPc every rank holds the full case while
loading) and the solver-reported peak (hsolve peak_rss_bytes; SLEPc summed VmHWM of its solver
phases).

Validity (equal accuracy): every result is checked with tools/eval at the case tolerance against the
run's target (the case norms are estimated once per case): backward errors with conservative 2-norm
estimates, completeness against the case reference (certified against a complete reference;
against an incomplete solver reference only a consistency check, status "consistent"), duplicate
vectors. A run is VALID when it ended normally (exit code 0, not killed), the solver claims success
and the evaluation passes; the run's own result directory is cleared before it starts, so a failed
run can never be credited with an earlier run's files. 'ok' = valid in every repetition; the table
lists the valid runs per solver, and a speedup is printed only when both solvers were valid in every
repetition. The eigenvalue sets are compared too (informational): with the evaluation's yardstick
|lambda - mu| / (|mu| + ||A||_2/||B||_2), against the target set of a complete reference and against
SLEPc's first valid run.
--reps N runs every (case, solver) N times INTERLEAVED (rotating solver order) and reports medians
with [min-max]. --target replaces every case's config target (e.g. an interval, for spectrum
slicing; a nearest target's k also drives the profile's k-rules). Eigenvectors are deleted after the
evaluation (evaluation.json stays) unless --keep-vectors. Writes <out-dir>/<tag>.json and .md.

compare.py evaluates in-process, so it runs under tools/run/guard.py itself (like tune_slepc.py);
its solver runs get their own nested guards. --exclusive makes each nested guard take the
machine-wide lock: do not combine it with an --exclusive outer guard (the nested guard would wait
for the lock its parent holds). For exclusive measurements wrap compare.py in an --exclusive guard
and leave --exclusive out.

usage: python tools/run/guard.py --skip-busy-check --cores 40,41 --mem-gb 4 --timeout 7200 -- \\
           python tools/bench/compare.py --tag NAME --cores 40,41 [--slepc-profile best] [--reps 3] case...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402
from tools.bench import slepc_log  # noqa: E402
from tools.bench.slepc_profile import (PROFILE_FILE, SlepcConfig, default_config, load_profiles,  # noqa: E402
                                       lookup, mpiexec_command, parse_cpulist)

HSOLVE = REPO / "build/release/hsolve"
GUARD = REPO / "tools/run/guard.py"
PHASE_KEYS = ("analysis_s", "numeric_s", "factorizations", "solves", "solve_s", "ortho_s", "iterate_s", "verify_s")
# evaluation of one result directory -> {verdict, reasons, completeness, certified, max_eta}
Judge = Callable[[Path, dict], dict]


def guarded(cmd: list[str], summary: Path, cores: str, mem_gb: float, timeout: int, threads: int,
            exclusive: bool) -> dict:
    summary.parent.mkdir(parents=True, exist_ok=True)
    g = [sys.executable, str(GUARD), "--cores", cores, "--threads", str(threads), "--mem-gb", str(mem_gb),
         "--timeout", str(timeout), "--summary", str(summary)]
    g += ["--exclusive"] if exclusive else ["--skip-busy-check"]
    proc = subprocess.run(g + ["--"] + cmd, cwd=REPO, capture_output=True, text=True)
    s = json.loads(summary.read_text()) if summary.exists() else {"returncode": proc.returncode, "killed": "no summary"}
    s["stderr_tail"] = proc.stderr[-1500:]
    return s


def slepc_config(c: case.Case, tgt: dict[str, Any], ncores: int, args) -> tuple[SlepcConfig, str, str]:
    """(configuration, where it came from, scope: case | class | default) for the slepc solver on
    case c with the effective target tgt."""
    mode = tgt.get("mode", "nearest")
    if args.slepc_profile == "best":
        hit = lookup(load_profiles(args.slepc_profile_file), c.name, c.meta["problem"], mode, ncores,
                     target=tgt, config_target=c.config.get("target"))
        if hit is not None:
            cfg, entry = hit
            return cfg, f"{entry['scope']} profile {entry['name']}", entry["scope"]
        return default_config(ncores), "default (no tuned profile)", "default"
    ranks = args.slepc_ranks
    if ncores % ranks:
        raise SystemExit(f"--slepc-ranks {ranks} does not divide the {ncores} cores")
    return SlepcConfig(ranks, ncores // ranks), "default", "default"


def hsolve_command(c: case.Case, solver: str, out: Path, ncores: int, args, target: dict | None) -> list[str]:
    cmd = [str(HSOLVE), "--A", str(c.path / "A.npz"), "--config", str(c.path / "config.json"),
           "--out", str(out), "--threads", str(ncores), "--memory-budget-gb", str(args.mem_gb * 0.9), "-q"]
    if c.B is not None:
        cmd[3:3] = ["--B", str(c.path / "B.npz")]
    if solver == "hsolve-nocert" or target is not None:
        cfg = dict(c.config)
        if solver == "hsolve-nocert":
            cfg["inertia_check"] = "off"
        if target is not None:
            cfg["target"] = target
        cfg_path = out.parent / f"config-{solver}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg))
        cmd[cmd.index("--config") + 1] = str(cfg_path)
    return cmd


def hsolve_phases(rep: dict[str, Any]) -> dict[str, Any]:
    fs = rep.get("factor_stats") or {}
    return {"analysis_s": fs.get("analyze_s"), "numeric_s": fs.get("factorize_s"),
            "factorizations": fs.get("factorizations"), "solves": fs.get("rhs_solved"), "solve_s": fs.get("solve_s"),
            "ortho_s": None, "iterate_s": rep.get("time_iterate_s"), "verify_s": rep.get("time_verify_s")}


def make_judge(c: case.Case, tgt: dict[str, Any], norms) -> Judge:
    """tools/eval of a result directory of case c for the effective target tgt (norms: the case's
    evaluate.estimate_norms, computed once). Writes evaluation.json next to the result."""
    from tools.eval import evaluate
    ce = dataclasses.replace(c, config={**c.config, "target": dict(tgt)})

    def judge(out: Path, guard: dict) -> dict:
        try:
            doc = evaluate.evaluate(ce, out, norms=norms, guard=guard)
        except Exception as err:   # malformed result: an invalid run, not a crash of the comparison
            return {"verdict": "fail", "reasons": [f"evaluation failed: {type(err).__name__}: {err}"],
                    "completeness": None, "certified": False, "max_eta": None}
        (out / evaluate.EVAL_FILE).write_text(json.dumps(doc, indent=1) + "\n")
        return {"verdict": doc["verdict"], "reasons": doc["reasons"], "completeness": doc["completeness"]["status"],
                "certified": bool(doc["completeness"]["certified"]), "max_eta": doc["summary"]["max_backward_error"]}
    return judge


def clear_outputs(*paths: Path) -> None:
    """Remove what an earlier run with the same tag left at these paths (directories or files)."""
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def run_once(solver: str, c: case.Case, tgt: dict[str, Any], out: Path, args, override: dict | None,
             judge: Judge | None = None) -> dict[str, Any]:
    """One guarded run on case c for the effective target tgt (override: the --target replacing the
    config's, handed to the solver), evaluated by judge; the entry holds the measurements."""
    ncores = len(parse_cpulist(args.cores))
    log_view, gsum = out.parent / f"log_view-{out.name}.txt", out.parent / f"guard-{out.name}.json"
    clear_outputs(out, log_view, gsum)
    entry: dict[str, Any] = {"solver": solver}
    if solver.startswith("hsolve"):
        cmd = hsolve_command(c, solver, out, ncores, args, override)
    elif solver == "slepc":
        cfg, source, scope = slepc_config(c, tgt, ncores, args)
        k = int(tgt["k"]) if tgt.get("mode", "nearest") == "nearest" else None
        cmd = mpiexec_command(cfg, c.name, out, k=k, target=override, log_view=log_view, root=args.root)
        entry.update(config=cfg.label(), config_source=source, config_scope=scope)
    else:
        raise SystemExit(f"unknown solver {solver}")
    g = guarded(cmd, gsum, args.cores, args.mem_gb, args.timeout, ncores, args.exclusive)
    run_ok = g.get("returncode") == 0 and not g.get("killed")
    entry.update(wall_s=g.get("wall_s"), peak_rss_gib=g.get("peak_rss_gib"), rc=g.get("returncode"),
                 killed=g.get("killed"))
    why = [] if run_ok else [f"run failed (rc={g.get('returncode')}, killed={g.get('killed')})"]
    if not run_ok:
        entry["error"] = g.get("stderr_tail", "")[-300:]
    if not (out / "report.json").exists():
        entry.update(success=False, valid=False, why=why + ["no report.json"])
        entry.setdefault("error", g.get("stderr_tail", "")[-300:])
        return entry
    res = case.load_result(out)
    rep = res.report
    etas = [p.get("backward_error", np.nan) for p in rep.get("pairs", [])]
    solver_eta = float(np.nanmax(etas)) if etas and not np.all(np.isnan(etas)) else None
    entry.update(claimed_success=bool(rep.get("success")), success=bool(rep.get("success")) and run_ok,
                 solver_max_eta=solver_eta, max_eta=solver_eta,   # replaced by the evaluation's
                 pairs=len(res.eigenvalues), eigenvalues=[float(x) for x in np.asarray(res.eigenvalues).real],
                 solver_peak_bytes=rep.get("peak_rss_bytes"))
    if not rep.get("success"):
        why.append("solver claims no success")
    if judge is not None:
        ev = judge(out, g)
        entry.update(verdict=ev["verdict"], eval_reasons=ev["reasons"], completeness=ev["completeness"],
                     certified=ev["certified"], max_eta=ev["max_eta"])
        if ev["verdict"] != "pass":
            why += ev["reasons"] or ["evaluation failed"]
        if not args.keep_vectors:
            for f in ("eigenvectors.npy", "complex_eigenvectors.npy"):
                (out / f).unlink(missing_ok=True)
    entry.update(valid=not why, why=why)
    if solver == "slepc":
        ph = slepc_log.read_phases(log_view)
        if ph is not None:
            entry.update(solver_s=ph["solver_s"], solver_s_scope="log_view setup+solve+vectors",
                         phases={k: ph.get(k) for k in PHASE_KEYS})
        else:
            entry.update(solver_s=(rep.get("time_setup_s") or 0) + (rep.get("time_solve_s") or 0),
                         solver_s_scope="report time_setup_s+time_solve_s", phases={})
    else:
        entry.update(solver_s=rep.get("time_total_s"), solver_s_scope="report time_total_s",
                     phases=hsolve_phases(rep))
    return entry


def nearest_set(w: np.ndarray, sigma: float, k: int) -> np.ndarray:
    w = np.asarray(w).real
    return np.sort(w[np.argsort(np.abs(w - sigma), kind="stable")[:k]])


def expected_values(ref: case.Reference, tgt: dict[str, Any]) -> np.ndarray | None:
    """The target's eigenvalues from a complete reference (nearest: k nearest; interval: inside)."""
    if ref is None or not ref.info.get("complete") or ref.info.get("kind") not in ("dense", "discrete-exact", "analytic"):
        return None
    w = np.asarray(ref.eigenvalues).real
    if tgt.get("mode", "nearest") == "interval":
        return np.sort(w[(w >= float(tgt["lower"])) & (w <= float(tgt["upper"]))])
    return nearest_set(w, float(tgt.get("sigma", 0.0)), int(tgt["k"]))


def set_error(got: np.ndarray, want: np.ndarray, scale: float) -> float | None:
    """Largest |lambda - mu| / (|mu| + scale) between two sorted sets of equal size (the evaluation's
    yardstick: a backward-stable solver at tol is off by about tol in it); None if sizes differ."""
    if got.size != want.size:
        return None
    if not got.size:
        return 0.0
    w = np.sort(want)
    return float(np.max(np.abs(np.sort(got) - w) / np.maximum(np.abs(w) + scale, 1e-300)))


def check_sets(row: dict[str, Any], sets: dict[str, list[np.ndarray]], want: np.ndarray | None,
               scale: float) -> None:
    """Informational set errors of the valid repetitions (worst case kept): against the target set of
    a complete reference (want) and against SLEPc's first valid run. Validity itself is decided by
    tools/eval (run_once), whose completeness test resolves ties and multiplicities."""
    for s, vs in sets.items():
        e = row["solvers"][s]
        if want is not None:
            errs = [set_error(v, want, scale) for v in vs]
            e["set_err_vs_ref"] = None if None in errs else max(errs)
            sizes = sorted({v.size for v in vs if v.size != want.size})
            if sizes:
                e["set_mismatch"] = f"{'/'.join(map(str, sizes))} values, reference has {want.size}"
        if s != "slepc" and "slepc" in sets:
            diffs = [set_error(v, sets["slepc"][0], scale) for v in vs]
            e["set_diff_vs_slepc"] = None if None in diffs else max(diffs)


def _median(vals: list[Any]) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(statistics.median(vals)) if vals else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over the repetitions of one (case, solver); 'ok' only if every repetition was valid."""
    times = [e.get("solver_s") for e in entries if e.get("solver_s") is not None]
    valid = [bool(e.get("valid")) for e in entries]
    out: dict[str, Any] = {
        "reps": len(entries), "valid_runs": sum(valid), "ok": bool(entries) and all(valid),
        "solver_s": _median(times), "solver_s_min": min(times) if times else None,
        "solver_s_max": max(times) if times else None, "solver_s_all": times,
        "solver_s_scope": next((e["solver_s_scope"] for e in entries if e.get("solver_s_scope")), None),
        "wall_s": _median([e.get("wall_s") for e in entries]),
        "peak_rss_gib": max((e["peak_rss_gib"] for e in entries if e.get("peak_rss_gib") is not None), default=None),
        "solver_peak_gib": max((e["solver_peak_bytes"] / 2**30 for e in entries if e.get("solver_peak_bytes")),
                               default=None),
        "max_eta": max((e["max_eta"] for e in entries if e.get("max_eta") is not None), default=None),
        "completeness": sorted({e["completeness"] for e in entries if e.get("completeness")}),
        "pairs": entries[-1].get("pairs") if entries else None,
        "phases": {k: _median([(e.get("phases") or {}).get(k) for e in entries]) for k in PHASE_KEYS},
        "errors": [e["error"] for e in entries if e.get("error")][:1],
        "invalid_reasons": sorted({r for e in entries if not e.get("valid") for r in (e.get("why") or ["invalid"])}),
    }
    for key in ("config", "config_source", "config_scope"):
        if any(e.get(key) for e in entries):
            out[key] = next(e[key] for e in entries if e.get(key))
    return out


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


# What a SLEPc speedup is measured against, unless it is the case's own tuned profile.
_BASELINE_NOTE = {"class": "vs class profile", "default": "vs untuned default"}


def speedup_cell(e: dict[str, Any], sl: dict[str, Any], profile: str) -> str:
    """Speedup of a solver over SLEPc: median solver time (wall in parentheses), only when both were
    valid in every repetition; marked when SLEPc did not run the case's tuned profile."""
    if not (e.get("solver_s") and sl.get("solver_s")):
        return "-"
    if not (e.get("ok") and sl.get("ok")):
        bad = [f"{name} valid {x.get('valid_runs', 0)}/{x.get('reps', 0)}" for name, x in (("solver", e), ("slepc", sl))
               if not x.get("ok")]
        return "- (" + ", ".join(bad) + ")"
    cell = f"{sl['solver_s'] / e['solver_s']:.2f}x"
    if e.get("wall_s") and sl.get("wall_s"):
        cell += f" ({sl['wall_s'] / e['wall_s']:.2f}x)"
    note = _BASELINE_NOTE.get(sl.get("config_scope", "case"))
    if profile == "best" and note:
        cell += f" ({note})"
    return cell


def render_markdown(doc: dict[str, Any]) -> str:
    """The report of a compare.py JSON document (one row per case and solver, then speedups)."""
    lines = [f"# {doc['tag']}", "",
             f"cores `{doc['cores']}` ({doc['ncores']}), SLEPc profile `{doc['slepc_profile']}`, "
             f"{doc['reps']} interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / "
             "SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). "
             "Peak GiB: guard process tree / solver-reported. Valid = exit 0, success claimed and tools/eval "
             "pass at the case tol (completeness: complete = certified by a complete reference, consistent = "
             "agrees with an incomplete one); set errors |lambda-mu|/(|mu|+||A||/||B||) of the valid runs.", "",
             "| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | "
             "numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | valid | evaluation |",
             "|" + "---|" * 17]
    for r in doc["rows"]:
        for s, e in r["solvers"].items():
            ph = e.get("phases") or {}
            t = e.get("solver_s")
            span = f" [{_f(e.get('solver_s_min'))}-{_f(e.get('solver_s_max'))}]" if e.get("reps", 1) > 1 and t else ""
            conf = e.get("config", "")
            if e.get("config_source"):
                conf = f"`{conf}` ({e['config_source']})"
            chk = list(e.get("completeness") or [])
            if e.get("set_err_vs_ref") is not None:
                chk.append(f"ref {e['set_err_vs_ref']:.0e}")
            if e.get("set_diff_vs_slepc") is not None:
                chk.append(f"slepc {e['set_diff_vs_slepc']:.0e}")
            if e.get("set_mismatch"):
                chk.append(e["set_mismatch"])
            if e.get("invalid_reasons"):
                chk.append("invalid: " + "; ".join(e["invalid_reasons"])[:160])
            num = _f(ph.get("numeric_s"))
            if ph.get("factorizations") is not None:
                num += f" ({ph['factorizations']:.0f})"
            solves = "-" if ph.get("solves") is None else f"{ph['solves']:.0f} / {_f(ph.get('solve_s'))}"
            valid = f"{'yes' if e.get('ok') else 'NO'} ({e.get('valid_runs', 0)}/{e.get('reps', 0)})"
            lines.append("| " + " | ".join([
                r["case"], str(r["n"]), r["problem"], r["target_desc"], s, conf, _f(t) + span, _f(e.get("wall_s")),
                _f(ph.get("analysis_s")), num, solves, _f(ph.get("ortho_s")), _f(ph.get("iterate_s")),
                _f(ph.get("verify_s")), f"{_f(e.get('peak_rss_gib'))} / {_f(e.get('solver_peak_gib'))}",
                valid, "; ".join(chk)]) + " |")
    others = [s for s in doc["solvers"] if s != "slepc"]
    if "slepc" in doc["solvers"] and others:
        lines += ["", "Speedup vs SLEPc (median solver time; wall in parentheses), given only when both solvers "
                  "were valid in every repetition. 'vs class profile' / 'vs untuned default': SLEPc did not run "
                  "a profile tuned for this case, so this is no comparison with the tuned baseline.", "",
                  "| case | " + " | ".join(others) + " |", "|" + "---|" * (1 + len(others))]
        for r in doc["rows"]:
            sl = r["solvers"].get("slepc", {})
            cells = [speedup_cell(r["solvers"].get(s, {}), sl, doc["slepc_profile"]) for s in others]
            lines.append(f"| {r['case']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def target_desc(tgt: dict[str, Any]) -> str:
    if tgt.get("mode", "nearest") == "interval":
        return f"[{float(tgt['lower']):g}, {float(tgt['upper']):g}]"
    return f"k={tgt.get('k')} @ {float(tgt.get('sigma', 0.0)):.4g}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", default=os.environ.get("HS_CORES", "24-30,33-41"))
    ap.add_argument("--solvers", default="hsolve,hsolve-nocert,slepc")
    ap.add_argument("--slepc-profile", choices=("default", "best"), default="default")
    ap.add_argument("--slepc-profile-file", default=str(PROFILE_FILE))
    ap.add_argument("--slepc-ranks", type=int, default=1, help="ranks of the default SLEPc profile")
    ap.add_argument("--reps", type=int, default=1, help="interleaved repetitions per (case, solver)")
    ap.add_argument("--target", default=None, help="JSON Target replacing every case's config target")
    ap.add_argument("--root", default=str(case.DEFAULT_ROOT))
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--exclusive", action="store_true",
                    help="nested solver guards take the exclusive lock (not under an --exclusive outer guard)")
    ap.add_argument("--keep-vectors", action="store_true", help="keep the eigenvectors of every result")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results", help="where <tag>.json/.md go")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tools.eval import common, evaluate
    common.require_guard("compare")
    solvers = args.solvers.split(",")
    override = json.loads(args.target) if args.target else None
    ncores = len(parse_cpulist(args.cores))
    rows = []
    for name in args.cases:
        c = case.load_case(name, root=args.root)
        tgt = dict(override) if override else dict(c.config["target"])
        ref = case.load_ref(name, root=args.root)
        t0 = time.perf_counter()
        norms = evaluate.estimate_norms(c.A, c.B)
        norm_s = time.perf_counter() - t0
        judge = make_judge(c, tgt, norms)
        runs: dict[str, list[dict[str, Any]]] = {s: [] for s in solvers}
        for rep in range(args.reps):
            order = solvers[rep % len(solvers):] + solvers[:rep % len(solvers)]
            for s in order:
                out = REPO / "results/raw" / args.tag / name / (s if args.reps == 1 else f"{s}-r{rep}")
                e = run_once(s, c, tgt, out, args, override, judge)
                runs[s].append(e)
                print(f"{name:34s} {s:14s} rep {rep} solver={_f(e.get('solver_s'), '{:.3f}')} wall={e.get('wall_s')} "
                      f"GiB={e.get('peak_rss_gib')} valid={e.get('valid')} eta={e.get('max_eta')} "
                      f"{e.get('completeness')} {'; '.join(e.get('eval_reasons') or [])[:120]}", flush=True)
        row = {"case": name, "n": c.n, "problem": c.meta["problem"], "target": tgt, "target_desc": target_desc(tgt),
               "k": tgt.get("k"), "norms": {"norm2_a": norms.norm2_a, "norm2_b": norms.norm2_b, "scale": norms.scale,
                                            "time_s": norm_s}, "solvers": {}}
        sets: dict[str, list[np.ndarray]] = {}
        for s in solvers:
            row["solvers"][s] = summarize(runs[s])
            good = [np.asarray(e["eigenvalues"]) for e in runs[s]
                    if e.get("valid") and e.get("eigenvalues") is not None]
            if good:
                sets[s] = good
        check_sets(row, sets, expected_values(ref, tgt), norms.scale)
        rows.append(row)
    doc = {"tag": args.tag, "cores": args.cores, "ncores": ncores, "solvers": solvers,
           "slepc_profile": args.slepc_profile, "slepc_ranks": args.slepc_ranks, "reps": args.reps,
           "target_override": override, "evaluated": True, "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = render_markdown(doc)
    (args.out_dir / f"{args.tag}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
