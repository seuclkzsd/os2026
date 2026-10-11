#!/usr/bin/env python3
"""Head-to-head benchmark: hsolve vs reference solvers on the same cases, cores and tolerance.

Every run goes through tools/run/guard.py (core pinning, RSS watchdog, peak memory incl. all MPI
ranks), so wall time and peak RSS are measured the same way for all solvers. Solvers (--solvers):
  hsolve        default configuration (inertia certificate on)
  hsolve-nocert inertia certificate off (pure speed; same accuracy, no completeness proof)
  slepc         SLEPc Krylov-Schur + shift-invert + MUMPS (tools/bench/baseline_slepc.py, env os2026-ref)
SLEPc configuration (--slepc-profile):
  default       1 rank x P threads, SLEPc/MUMPS defaults (--slepc-ranks R: R ranks x P/R threads)
  best          the tuned profile of the case for this core count (tools/bench/slepc_profiles.json,
                written by tools/bench/tune_slepc.py; else the case class's profile, else default -
                the table says which was used)
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
--reps N runs every (case, solver) N times INTERLEAVED (rotating solver order) and reports medians
with [min-max]. Accuracy is compared on the eigenvalue SETS: against the case's complete reference
when it has one (a set off by more than SET_TOL marks the run "NO (set)" even when the solver claimed
success), otherwise between solvers. --target replaces every case's config target (e.g. an
interval, for spectrum slicing). Writes <out-dir>/<tag>.json and <out-dir>/<tag>.md.

usage: python tools/bench/compare.py --tag NAME --cores 40,41 [--slepc-profile best] [--reps 3] case...
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402
from tools.bench import slepc_log  # noqa: E402
from tools.bench.slepc_profile import (PROFILE_FILE, SlepcConfig, default_config, load_profiles,  # noqa: E402
                                       lookup, mpiexec_command)

HSOLVE = REPO / "build/release/hsolve"
GUARD = REPO / "tools/run/guard.py"
PHASE_KEYS = ("analysis_s", "numeric_s", "factorizations", "solves", "solve_s", "ortho_s", "iterate_s", "verify_s")
# A returned eigenvalue set differing from a complete reference's target set by more than this
# (relative to the largest |eigenvalue|) has missed or misplaced an eigenvalue: a backward-stable
# solver at tol ~ 1e-10 is off by ~tol only.
SET_TOL = 1e-6


def case_cores(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


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


def slepc_config(c: case.Case, mode: str, ncores: int, args) -> tuple[SlepcConfig, str]:
    """(configuration, where it came from) for the slepc solver on case c."""
    if args.slepc_profile == "best":
        hit = lookup(load_profiles(args.slepc_profile_file), c.name, c.meta["problem"], mode, ncores)
        if hit is not None:
            cfg, entry = hit
            return cfg, f"{entry['scope']} profile {entry['name']}"
        return default_config(ncores), "default (no tuned profile)"
    ranks = args.slepc_ranks
    if ncores % ranks:
        raise SystemExit(f"--slepc-ranks {ranks} does not divide the {ncores} cores")
    return SlepcConfig(ranks, ncores // ranks), "default"


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


def run_once(solver: str, c: case.Case, mode: str, out: Path, args, target: dict | None) -> dict[str, Any]:
    """One guarded run; the entry holds the measurements and, while the result exists, its path."""
    ncores = len(case_cores(args.cores))
    entry: dict[str, Any] = {"solver": solver}
    if solver.startswith("hsolve"):
        cmd = hsolve_command(c, solver, out, ncores, args, target)
    elif solver == "slepc":
        cfg, source = slepc_config(c, mode, ncores, args)
        k = int(c.config["target"]["k"]) if mode == "nearest" else None
        cmd = mpiexec_command(cfg, c.name, out, k=k, target=target, log_view=out.parent / f"log_view-{out.name}.txt",
                              root=args.root)
        entry.update(config=cfg.label(), config_source=source)
    else:
        raise SystemExit(f"unknown solver {solver}")
    g = guarded(cmd, out.parent / f"guard-{out.name}.json", args.cores, args.mem_gb, args.timeout, ncores,
                args.exclusive)
    entry.update(wall_s=g.get("wall_s"), peak_rss_gib=g.get("peak_rss_gib"), rc=g.get("returncode"),
                 killed=g.get("killed"))
    if not (out / "report.json").exists():
        entry.update(success=False, error=g.get("stderr_tail", "")[-300:])
        return entry
    res = case.load_result(out)
    rep = res.report
    etas = [p.get("backward_error", np.nan) for p in rep.get("pairs", [])]
    entry.update(success=bool(rep.get("success")), max_eta=float(np.nanmax(etas)) if etas else None,
                 pairs=len(res.eigenvalues), eigenvalues=[float(x) for x in np.asarray(res.eigenvalues).real],
                 solver_peak_bytes=rep.get("peak_rss_bytes"))
    if solver == "slepc":
        ph = slepc_log.read_phases(out.parent / f"log_view-{out.name}.txt")
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
    if got.size != want.size:
        return None
    return float(np.max(np.abs(np.sort(got) - np.sort(want))) / max(scale, 1e-300)) if got.size else 0.0


def check_sets(row: dict[str, Any], sets: dict[str, list[np.ndarray]], want: np.ndarray | None,
               scale: float) -> None:
    """Eigenvalue-set checks of every successful repetition, worst case kept: against the target set
    of a complete reference (want; a wrong set marks the solver not ok) and against SLEPc's first."""
    for s, vs in sets.items():
        e = row["solvers"][s]
        if want is not None:
            errs = [set_error(v, want, scale) for v in vs]
            e["set_err_vs_ref"] = None if None in errs else max(errs)
            sizes = sorted({v.size for v in vs if v.size != want.size})
            if sizes:
                e["set_mismatch"] = f"{'/'.join(map(str, sizes))} values, reference has {want.size}"
            if e["set_err_vs_ref"] is None or e["set_err_vs_ref"] > SET_TOL:
                e["ok"] = False
                e["set_wrong"] = True
        if s != "slepc" and "slepc" in sets:
            diffs = [set_error(v, sets["slepc"][0], scale) for v in vs]
            e["set_diff_vs_slepc"] = None if None in diffs else max(diffs)


def _median(vals: list[Any]) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(statistics.median(vals)) if vals else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over the repetitions of one (case, solver); 'ok' only if every repetition succeeded."""
    times = [e.get("solver_s") for e in entries if e.get("solver_s") is not None]
    out: dict[str, Any] = {
        "reps": len(entries), "ok": bool(entries) and all(e.get("success") for e in entries),
        "solver_s": _median(times), "solver_s_min": min(times) if times else None,
        "solver_s_max": max(times) if times else None, "solver_s_all": times,
        "solver_s_scope": next((e["solver_s_scope"] for e in entries if e.get("solver_s_scope")), None),
        "wall_s": _median([e.get("wall_s") for e in entries]),
        "peak_rss_gib": max((e["peak_rss_gib"] for e in entries if e.get("peak_rss_gib") is not None), default=None),
        "solver_peak_gib": max((e["solver_peak_bytes"] / 2**30 for e in entries if e.get("solver_peak_bytes")),
                               default=None),
        "max_eta": max((e["max_eta"] for e in entries if e.get("max_eta") is not None), default=None),
        "pairs": entries[-1].get("pairs") if entries else None,
        "phases": {k: _median([(e.get("phases") or {}).get(k) for e in entries]) for k in PHASE_KEYS},
        "errors": [e["error"] for e in entries if e.get("error")][:1],
    }
    for key in ("config", "config_source"):
        if any(e.get(key) for e in entries):
            out[key] = next(e[key] for e in entries if e.get(key))
    return out


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def render_markdown(doc: dict[str, Any]) -> str:
    """The report of a compare.py JSON document (one row per case and solver, then speedups)."""
    lines = [f"# {doc['tag']}", "",
             f"cores `{doc['cores']}` ({doc['ncores']}), SLEPc profile `{doc['slepc_profile']}`, "
             f"{doc['reps']} interleaved repetition(s): medians [min-max]. Solver time = hsolve time_total_s / "
             "SLEPc -log_view setup+solve+vectors (same scope: no process start-up, loading or output). "
             "Peak GiB: guard process tree / solver-reported.", "",
             "| case | n | type | target | solver | configuration | solver time (s) | wall (s) | analysis | "
             "numeric (#) | solves (#/s) | ortho | iterate | verify | peak GiB | ok | set check |",
             "|" + "---|" * 17]
    for r in doc["rows"]:
        for s, e in r["solvers"].items():
            ph = e.get("phases") or {}
            t = e.get("solver_s")
            span = f" [{_f(e.get('solver_s_min'))}-{_f(e.get('solver_s_max'))}]" if e.get("reps", 1) > 1 and t else ""
            conf = e.get("config", "")
            if e.get("config_source"):
                conf = f"`{conf}` ({e['config_source']})"
            chk = []
            if e.get("set_err_vs_ref") is not None:
                chk.append(f"ref {e['set_err_vs_ref']:.0e}")
            if e.get("set_diff_vs_slepc") is not None:
                chk.append(f"slepc {e['set_diff_vs_slepc']:.0e}")
            if e.get("set_mismatch"):
                chk.append(e["set_mismatch"])
            num = _f(ph.get("numeric_s"))
            if ph.get("factorizations") is not None:
                num += f" ({ph['factorizations']:.0f})"
            solves = "-" if ph.get("solves") is None else f"{ph['solves']:.0f} / {_f(ph.get('solve_s'))}"
            lines.append("| " + " | ".join([
                r["case"], str(r["n"]), r["problem"], r["target_desc"], s, conf, _f(t) + span, _f(e.get("wall_s")),
                _f(ph.get("analysis_s")), num, solves, _f(ph.get("ortho_s")), _f(ph.get("iterate_s")),
                _f(ph.get("verify_s")), f"{_f(e.get('peak_rss_gib'))} / {_f(e.get('solver_peak_gib'))}",
                "yes" if e.get("ok") else ("NO (set)" if e.get("set_wrong") else "NO"), "; ".join(chk)]) + " |")
    others = [s for s in doc["solvers"] if s != "slepc"]
    if "slepc" in doc["solvers"] and others:
        lines += ["", "Speedup vs SLEPc (median solver time; wall in parentheses):", "",
                  "| case | " + " | ".join(others) + " |", "|" + "---|" * (1 + len(others))]
        for r in doc["rows"]:
            sl = r["solvers"].get("slepc", {})
            cells = []
            for s in others:
                e = r["solvers"].get(s, {})
                if e.get("solver_s") and sl.get("solver_s"):
                    cell = f"{sl['solver_s'] / e['solver_s']:.2f}x"
                    if e.get("wall_s") and sl.get("wall_s"):
                        cell += f" ({sl['wall_s'] / e['wall_s']:.2f}x)"
                    if not (e.get("ok") and sl.get("ok")):
                        cell += " (not both ok)"
                else:
                    cell = "-"
                cells.append(cell)
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
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results", help="where <tag>.json/.md go")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    solvers = args.solvers.split(",")
    override = json.loads(args.target) if args.target else None
    ncores = len(case_cores(args.cores))
    rows = []
    for name in args.cases:
        c = case.load_case(name, root=args.root)
        tgt = dict(override) if override else dict(c.config["target"])
        mode = tgt.get("mode", "nearest")
        ref = case.load_ref(name, root=args.root)
        runs: dict[str, list[dict[str, Any]]] = {s: [] for s in solvers}
        for rep in range(args.reps):
            order = solvers[rep % len(solvers):] + solvers[:rep % len(solvers)]
            for s in order:
                out = REPO / "results/raw" / args.tag / name / (s if args.reps == 1 else f"{s}-r{rep}")
                e = run_once(s, c, mode, out, args, override)
                runs[s].append(e)
                print(f"{name:34s} {s:14s} rep {rep} solver={_f(e.get('solver_s'), '{:.3f}')} wall={e.get('wall_s')} "
                      f"GiB={e.get('peak_rss_gib')} ok={e.get('success')} eta={e.get('max_eta')}", flush=True)
        row = {"case": name, "n": c.n, "problem": c.meta["problem"], "target": tgt, "target_desc": target_desc(tgt),
               "k": tgt.get("k"), "solvers": {}}
        sets: dict[str, list[np.ndarray]] = {}
        for s in solvers:
            row["solvers"][s] = summarize(runs[s])
            good = [np.asarray(e["eigenvalues"]) for e in runs[s]
                    if e.get("success") and e.get("eigenvalues") is not None]
            if good:
                sets[s] = good
        scale = float(np.max(np.abs(ref.eigenvalues.real))) if ref is not None else max(
            [float(np.max(np.abs(v))) for vs in sets.values() for v in vs if v.size] or [1.0])
        check_sets(row, sets, expected_values(ref, tgt), scale)
        rows.append(row)
    doc = {"tag": args.tag, "cores": args.cores, "ncores": ncores, "solvers": solvers,
           "slepc_profile": args.slepc_profile, "slepc_ranks": args.slepc_ranks, "reps": args.reps,
           "target_override": override, "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = render_markdown(doc)
    (args.out_dir / f"{args.tag}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
