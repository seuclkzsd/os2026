#!/usr/bin/env python3
"""Head-to-head benchmark: hsolve vs reference solvers on the same cases, cores and tolerance.

Every run goes through tools/run/guard.py (core pinning, RSS watchdog, peak memory incl. all MPI
ranks); wall time and peak RSS are taken from the guard summary so all solvers are measured the
same way. Solvers (select with --solvers):
  hsolve        default configuration (inertia certificate on)
  hsolve-nocert inertia certificate off (pure speed; same accuracy, no completeness proof)
  slepc         SLEPc Krylov-Schur + shift-invert + MUMPS (tools/bench/baseline_slepc.py, env os2026-ref)
Accuracy is compared on the eigenvalue SETS: against the case's complete reference when it has
one, otherwise between solvers. Writes results/<tag>.json and results/<tag>.md.

usage: python tools/bench/compare.py --tag NAME --cores 28,29,40,41 [--slepc-ranks 1] case...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402

REF_ENV = Path(os.environ.get("HS_REF_ENV", Path.home() / "miniconda3/envs/os2026-ref"))
HSOLVE = REPO / "build/release/hsolve"
GUARD = REPO / "tools/run/guard.py"


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


def run_solver(solver: str, c: str, out: Path, args) -> dict:
    ncores = len(case_cores(args.cores))
    if solver.startswith("hsolve"):
        cs = case.load_case(c)
        cmd = [str(HSOLVE), "--A", str(cs.path / "A.npz"), "--config", str(cs.path / "config.json"),
               "--out", str(out), "--threads", str(ncores), "--memory-budget-gb", str(args.mem_gb * 0.9), "-q"]
        if cs.B is not None:
            cmd[3:3] = ["--B", str(cs.path / "B.npz")]
        if solver == "hsolve-nocert":
            cfg = dict(cs.config, inertia_check="off")
            cfg_path = out.parent / "config-nocert.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg))
            cmd[cmd.index("--config") + 1] = str(cfg_path)
        return guarded(cmd, out.parent / f"guard-{solver}.json", args.cores, args.mem_gb, args.timeout, ncores,
                       args.exclusive)
    if solver == "slepc":
        ranks = args.slepc_ranks
        t = max(1, ncores // ranks)
        cmd = [str(REF_ENV / "bin/mpiexec"), "-n", str(ranks), "--bind-to", "none", "-x", f"OMP_NUM_THREADS={t}",
               "-x", f"OPENBLAS_NUM_THREADS={t}", str(REF_ENV / "bin/python"),
               str(REPO / "tools/bench/baseline_slepc.py"), "--case", c, "--out", str(out)]
        return guarded(cmd, out.parent / f"guard-{solver}.json", args.cores, args.mem_gb, args.timeout, ncores,
                       args.exclusive)
    raise SystemExit(f"unknown solver {solver}")


def case_cores(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def nearest_set(w: np.ndarray, sigma: float, k: int) -> np.ndarray:
    w = np.asarray(w).real
    return np.sort(w[np.argsort(np.abs(w - sigma), kind="stable")[:k]])


def set_error(got: np.ndarray, want: np.ndarray, scale: float) -> float | None:
    if got.size != want.size:
        return None
    return float(np.max(np.abs(np.sort(got) - np.sort(want))) / max(scale, 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", default=os.environ.get("HS_CORES", "24-30,33-41"))
    ap.add_argument("--solvers", default="hsolve,hsolve-nocert,slepc")
    ap.add_argument("--slepc-ranks", type=int, default=1)
    ap.add_argument("--mem-gb", type=float, default=8.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--exclusive", action="store_true")
    args = ap.parse_args()
    solvers = args.solvers.split(",")
    rows = []
    for c in args.cases:
        cs = case.load_case(c)
        t = cs.config["target"]
        sigma, k = float(t.get("sigma", 0.0)), int(t.get("k", 0))
        ref = case.load_ref(c)
        row = {"case": c, "n": cs.n, "problem": cs.meta["problem"], "k": k, "solvers": {}}
        sets = {}
        for s in solvers:
            out = REPO / "results/raw" / args.tag / c / s
            g = run_solver(s, c, out, args)
            entry = {"wall_s": g.get("wall_s"), "peak_rss_gib": g.get("peak_rss_gib"), "rc": g.get("returncode"),
                     "killed": g.get("killed")}
            if (out / "report.json").exists():
                res = case.load_result(out)
                rep = res.report
                etas = [p.get("backward_error", np.nan) for p in rep.get("pairs", [])]
                entry.update(success=rep.get("success"), max_eta=float(np.nanmax(etas)) if etas else None,
                             pairs=len(res.eigenvalues))
                sets[s] = np.asarray(res.eigenvalues)
            else:
                entry.update(success=False, error=g.get("stderr_tail", "")[-300:])
            row["solvers"][s] = entry
            print(f"{c:34s} {s:14s} wall={entry['wall_s']} GiB={entry['peak_rss_gib']} ok={entry.get('success')} "
                  f"eta={entry.get('max_eta')}", flush=True)
        scale = float(np.max(np.abs(ref.eigenvalues.real))) if ref is not None else max(
            [float(np.max(np.abs(v))) for v in sets.values() if v.size] or [1.0])
        if ref is not None and ref.info.get("complete") and ref.info.get("kind") in ("dense", "discrete-exact", "analytic"):
            want = nearest_set(ref.eigenvalues, sigma, k)
            for s, v in sets.items():
                row["solvers"][s]["set_err_vs_ref"] = set_error(v, want, scale)
        base = sets.get("slepc")
        if base is not None:
            for s, v in sets.items():
                if s != "slepc":
                    row["solvers"][s]["set_diff_vs_slepc"] = set_error(v, base, scale)
        rows.append(row)
    out_json = REPO / "results" / f"{args.tag}.json"
    out_json.write_text(json.dumps({"cores": args.cores, "solvers": solvers, "slepc_ranks": args.slepc_ranks,
                                    "rows": rows}, indent=1))
    # Markdown summary
    lines = [f"# {args.tag}", "", f"cores `{args.cores}`, SLEPc ranks {args.slepc_ranks}, wall/peak from guard.py", "",
             "| case | n | type | k | " + " | ".join(f"{s} time (s) | {s} GiB | {s} ok" for s in solvers) +
             " | speedup vs slepc | set check |", "|" + "---|" * (4 + 3 * len(solvers) + 2)]
    for r in rows:
        cells = [r["case"], str(r["n"]), r["problem"], str(r["k"])]
        for s in solvers:
            e = r["solvers"][s]
            cells += [f"{e['wall_s']:.2f}" if e.get("wall_s") is not None else "-",
                      f"{e['peak_rss_gib']:.2f}" if e.get("peak_rss_gib") is not None else "-",
                      "yes" if e.get("success") else "NO"]
        hs, sl = r["solvers"].get("hsolve", {}), r["solvers"].get("slepc", {})
        sp = f"{sl['wall_s'] / hs['wall_s']:.2f}x" if hs.get("wall_s") and sl.get("wall_s") else "-"
        chk = []
        for s in solvers:
            e = r["solvers"][s]
            if e.get("set_err_vs_ref") is not None:
                chk.append(f"{s}:ref {e['set_err_vs_ref']:.0e}")
            if e.get("set_diff_vs_slepc") is not None:
                chk.append(f"{s}:slepc {e['set_diff_vs_slepc']:.0e}")
        cells += [sp, "; ".join(chk)]
        lines.append("| " + " | ".join(cells) + " |")
    (REPO / "results" / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
