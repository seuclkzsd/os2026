#!/usr/bin/env python3
"""P10 "mpi": process / thread layouts on one socket and on two sockets, at 16 cores per run.

Each (repetition, case, layout) is ONE top-level guard call - `guard.py --exclusive --cores C
--mem-gb 32` - running tools/bench/mpi_socket_run.sh LAYOUT KIND; the layouts are interleaved
(their order rotates per repetition and case). A layout is PLACEMENT:SPLIT:
  PLACEMENT  one = --one-cores (default 16-31: the 16 cores of NUMA node 1, the measurement slot);
             two = --two-cores (default 8-15,16-23: 8 cores of each socket) - the guard's busy check
             covers them, so the node's lane (8-15) must be idle: build on 0-7 (HS_CORES) meanwhile;
  SPLIT      1x16, 1x16i (numactl --interleave=all), 1x8 (first half, the rest of the cores idle),
             2x8 (two independent processes, KIND bench), 2x8m (one MPI job of two ranks).
KIND bench runs build/mpi/seq_factor_bench (mpi_factor_bench for 2x8m) on a K(sigma) dump with a
fixed pivot order (--kdir: <case>.K.bin from tools/bench/mpi_dump_kernel.py, its .json for the exact
inertia, <case>.perm: hsolve's raw pivot order from bench_factspeed_factor --perm-out); KIND hsolve
runs the CLI (build/release/hsolve; 2x8m: mpiexec ... build/release/hsolve-mpi) on the case with its
config (or --target), output under OUT/TAG/<case>/<layout>-r<rep>/.

  python3 tools/bench/mpi_layouts.py run --tag T --kind bench|hsolve --layouts one:1x16,two:1x16,...
      --case NAME [--case NAME ...] [--reps 3] [--rep0 0] [--out OUT] [--root DATA] [--kdir DIR]
      [--bench-args "--modes count,full --nrhs 16 --solve-reps 3"] [--target JSON]
      [--timeout 1500] [--lock-wait 600]
  python3 tools/bench/mpi_layouts.py eval --tag T [--out OUT] [--root DATA]    (lane: tools/eval)
  python3 tools/bench/mpi_layouts.py table --tag T [--out OUT] [--base one:1x16] [--md FILE]

`run` itself is a light driver (it only waits for the guards): start it detached on the node, outside
any guard. Output: OUT/TAG.jsonl - the bench's JSON lines (with case, layout, rep, proc, cores) or
one {"event": "hsolve", ...} line per solver run, and one {"event": "guard", ...} line per run.
`eval` judges every hsolve result with tools/eval at the case tolerance (evaluation.json next to
it; a run is VALID when it exited 0, claimed success and the evaluation passed) and must run under
a guard in the build/test lane. `table` prints medians [min-max] per case and layout and the
speedups against --base (bench: one count, the pair of counts - the slower of the two concurrent
ones for 2x8 -, one factorization keeping the factors, one 16-RHS solve; hsolve: time_total_s and
its phases, valid runs only).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GUARD = REPO / "tools/run/guard.py"
RUN = REPO / "tools/bench/mpi_socket_run.sh"
SPLITS = ("1x16", "1x16i", "1x8", "2x8", "2x8m", "1+2x8")
DEFAULT_OUT = Path(os.environ.get("HOME", "/tmp")) / "p10mpi" / "runs"


def parse_layouts(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in spec.split(","):
        place, _, split = item.partition(":")
        if place not in ("one", "two") or split not in SPLITS:
            raise SystemExit(f"bad layout '{item}': PLACEMENT (one, two) : SPLIT ({', '.join(SPLITS)})")
        out.append((place, split))
    return out


def layout_name(place: str, split: str) -> str:
    return f"{place}:{split}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def hsolve_args(root: Path, name: str, out: Path, target: dict | None) -> list[str]:
    """CLI arguments of the case (without --threads, which the layout sets)."""
    cdir = root / "cases" / name
    args = ["--A", str(cdir / "A.npz")]
    if (cdir / "B.npz").exists():
        args += ["--B", str(cdir / "B.npz")]
    cfg_path = cdir / "config.json"
    if target is not None:
        cfg = json.loads(cfg_path.read_text())
        cfg["target"] = target
        cfg_path = out.parent / f"config-{out.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg))
    return args + ["--config", str(cfg_path), "--out", str(out), "-q"]


def bench_args(kdir: Path, name: str, extra: str) -> list[str]:
    side = json.loads((kdir / f"{name}.K.bin.json").read_text())
    args = ["--matrix", str(kdir / f"{name}.K.bin"), "--perm-raw", str(kdir / f"{name}.perm"), "--label", name]
    neg = (side.get("inertia") or {}).get("neg")
    if neg is not None:
        args += ["--expect-neg", str(int(neg))]
    return args + extra.split()


def cmd_run(a: argparse.Namespace) -> int:
    layouts = parse_layouts(a.layouts)
    cores = {"one": a.one_cores, "two": a.two_cores}
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / f"{a.tag}.jsonl"
    target = json.loads(a.target) if a.target else None
    for r in range(a.rep0, a.rep0 + a.reps):
        for ci, name in enumerate(a.case):
            k = (r + ci) % len(layouts)
            for place, split in layouts[k:] + layouts[:k]:
                lay = layout_name(place, split)
                gsum = a.out / a.tag / name / f"guard-{place}-{split}-r{r}.json"
                gsum.parent.mkdir(parents=True, exist_ok=True)
                if gsum.exists():
                    gsum.unlink()
                if a.kind == "bench":
                    args = bench_args(a.kdir, name, a.bench_args)
                    out = None
                else:
                    out = a.out / a.tag / name / f"{place}-{split}-r{r}"
                    if out.exists():
                        subprocess.run(["rm", "-rf", str(out)], check=True)
                    args = hsolve_args(a.root, name, out, target)
                cmd = [sys.executable, str(GUARD), "--exclusive", "--cores", cores[place], "--mem-gb", str(a.mem_gb),
                       "--timeout", str(a.timeout), "--lock-wait", str(a.lock_wait), "--summary", str(gsum), "--",
                       "bash", str(RUN), split, a.kind] + args
                t0 = time.time()
                print(f"== {time.strftime('%H:%M:%S')} {a.tag} {name} {lay} rep {r}", file=sys.stderr, flush=True)
                proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                meta = {"tag": a.tag, "case": name, "layout": lay, "placement": place, "split": split, "rep": r,
                        "kind": a.kind}
                with log.open("a") as f:
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("{"):
                            d = json.loads(line)
                            f.write(json.dumps({**meta, **d}) + "\n")
                    g = json.loads(gsum.read_text()) if gsum.exists() else {"returncode": proc.returncode,
                                                                            "killed": "no summary"}
                    entry = {**meta, "event": "guard", "returncode": g.get("returncode"), "killed": g.get("killed"),
                             "wall_s": g.get("wall_s"), "peak_rss_gib": g.get("peak_rss_gib"),
                             "peak_sampled_gib": g.get("peak_sampled_gib"),
                             "cores": cores[place], "stderr_tail": proc.stderr[-600:] if proc.returncode else ""}
                    if out is not None:
                        entry["out"] = str(out)
                        rep_path = out / "report.json"
                        if rep_path.exists():
                            rep = json.loads(rep_path.read_text())
                            entry["report"] = summarize_report(rep)
                    f.write(json.dumps(entry) + "\n")
                print(f"   rc={g.get('returncode')} killed={g.get('killed')} wall={time.time() - t0:.1f}s "
                      f"peak={g.get('peak_rss_gib')}", file=sys.stderr, flush=True)
                if g.get("returncode") == 75:
                    raise SystemExit("the measurement slot stayed busy (guard exit 75); stopping")
    return 0


def summarize_report(rep: dict[str, Any]) -> dict[str, Any]:
    fs = rep.get("factor_stats") or {}
    cert = rep.get("certificate") or {}
    keys = ("time_total_s", "time_setup_s", "time_factor_s", "time_iterate_s", "time_verify_s", "time_certificate_s",
            "success", "peak_rss_bytes", "block_size", "op_applies")
    out = {k: rep.get(k) for k in keys}
    out.update(analyze_s=fs.get("analyze_s"), factorize_s=fs.get("factorize_s"), factorizations=fs.get("factorizations"),
               solve_s=fs.get("solve_s"), rhs_solved=fs.get("rhs_solved"), ordering=(rep.get("factor_options") or {})
               .get("ordering"), cert_method=cert.get("method"), cert_passed=cert.get("passed"),
               slices=len(rep.get("slices") or []), parallel=rep.get("parallel"))
    return out


def cmd_eval(a: argparse.Namespace) -> int:
    from tools import case
    from tools.bench.compare import make_judge
    from tools.eval import common, evaluate
    common.require_guard("mpi_layouts eval")
    log = a.out / f"{a.tag}.jsonl"
    rows = [r for r in read_jsonl(log) if r.get("event") == "guard" and r.get("out")]
    verdicts_path = a.out / f"{a.tag}.eval.json"
    verdicts = json.loads(verdicts_path.read_text()) if verdicts_path.exists() else {}
    judges: dict[str, Any] = {}
    for r in rows:
        out = Path(r["out"])
        if str(out) in verdicts:
            continue
        name = r["case"]
        if name not in judges:
            c = case.load_case(name, root=a.root)
            cfg_override = out.parent / f"config-{out.name}.json"
            tgt = json.loads(cfg_override.read_text())["target"] if cfg_override.exists() else dict(c.config["target"])
            judges[name] = make_judge(c, tgt, evaluate.estimate_norms(c.A, c.B))
        g = {"returncode": r.get("returncode"), "killed": r.get("killed"), "wall_s": r.get("wall_s"),
             "peak_rss_gib": r.get("peak_rss_gib")}
        run_ok = r.get("returncode") == 0 and not r.get("killed")
        if not (out / "report.json").exists():
            verdicts[str(out)] = {"valid": False, "why": ["no report.json"]}
        else:
            ev = judges[name](out, g)
            rep = json.loads((out / "report.json").read_text())
            why = [] if run_ok else [f"run failed (rc={r.get('returncode')}, killed={r.get('killed')})"]
            if not rep.get("success"):
                why.append("solver claims no success")
            if ev["verdict"] != "pass":
                why += ev["reasons"] or ["evaluation failed"]
            verdicts[str(out)] = {"valid": not why, "why": why, "verdict": ev["verdict"],
                                  "completeness": ev["completeness"], "max_eta": ev["max_eta"]}
            for f in ("eigenvectors.npy", "complex_eigenvectors.npy"):
                (out / f).unlink(missing_ok=True)
        print(f"{name} {r['layout']} rep {r['rep']}: valid={verdicts[str(out)]['valid']} "
              f"{'; '.join(verdicts[str(out)].get('why') or [])[:200]}", flush=True)
        verdicts_path.write_text(json.dumps(verdicts, indent=1) + "\n")
    return 0


def _med(v: list[float]) -> float | None:
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def _span(v: list[float]) -> str:
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    m = statistics.median(v)
    return f"{m:.3f}" if len(v) == 1 else f"{m:.3f} [{min(v):.3f}-{max(v):.3f}]"


def bench_table(rows: list[dict[str, Any]], base: str) -> list[str]:
    """Per case and layout: medians over the repetitions of one count, the pair of counts (2x8: the
    slower of the two concurrent counts; else twice one count), one factorization, one solve."""
    per: dict[tuple[str, str], dict[str, list[float]]] = {}
    ok: dict[tuple[str, str], list[bool]] = {}
    for r in rows:
        key = (r["case"], r["layout"])
        d = per.setdefault(key, {"count": [], "pair": [], "factor": [], "solve": [], "peak": []})
        if r.get("event") == "guard":
            d["peak"].append(r.get("peak_rss_gib"))
            ok.setdefault(key, []).append(r.get("returncode") == 0 and not r.get("killed"))
    # group the bench events of one run (rep) by mode
    runs: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("event") in ("factorize", "solve", "run"):
            runs.setdefault((r["case"], r["layout"], r["rep"]), []).append(r)
    for (name, lay, rep), ev in runs.items():
        d = per[(name, lay)]
        cnt = [e["s"] for e in ev if e["event"] == "factorize" and e.get("mode") == "count"]
        fac = [e["s"] for e in ev if e["event"] == "factorize" and e.get("mode") == "full"]
        sol = [e["s"] for e in ev if e["event"] == "solve"]
        procs = {e.get("proc", 0) for e in ev}
        bad = [e for e in ev if e["event"] == "run" and not e.get("inertia_ok", True)]
        if bad:
            ok.setdefault((name, lay), []).append(False)
        if cnt:
            d["count"].append(statistics.median(cnt))
            d["pair"].append(max(cnt) if len(procs) > 1 else 2 * statistics.median(cnt))
        if fac:
            d["factor"].append(statistics.median(fac))
        if sol:
            d["solve"].append(statistics.median(sol))
    lines = ["| case | layout | runs ok | one count s | count pair s | x vs base | factorization s | x vs base | "
             "16-RHS solve s | x vs base | peak GiB |", "|" + "---|" * 11]
    cases = sorted({k[0] for k in per})
    for name in cases:
        b = per.get((name, base), {})
        for (cname, lay), d in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if cname != name:
                continue

            def ratio(key: str) -> str:
                mb, ml = _med(b.get(key, [])), _med(d[key])
                return "-" if not (mb and ml) else f"{mb / ml:.2f}"
            o = ok.get((name, lay), [])
            lines.append(f"| {name} | {lay} | {sum(o)}/{len(o)} | {_span(d['count'])} | {_span(d['pair'])} | "
                         f"{ratio('pair')} | {_span(d['factor'])} | {ratio('factor')} | {_span(d['solve'])} | "
                         f"{ratio('solve')} | {_med(d['peak']) or 0:.1f} |")
    return lines


HS_PHASES = ("time_total_s", "time_factor_s", "time_iterate_s", "time_certificate_s", "time_verify_s", "time_setup_s")


def hsolve_table(rows: list[dict[str, Any]], verdicts: dict[str, Any], base: str) -> list[str]:
    per: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("event") == "guard" and r.get("out"):
            per.setdefault((r["case"], r["layout"]), []).append(r)
    lines = ["| case | layout | valid | total s | x vs base | wall s | factor s | iterate s | certificate s | "
             "verify+setup s | peak GiB (tree / process peaks) |", "|" + "---|" * 11]
    for name in sorted({k[0] for k in per}):
        def med(lay: str, key: str) -> float | None:
            vals = [(e.get("report") or {}).get(key) for e in per.get((name, lay), [])
                    if (verdicts.get(e["out"]) or {}).get("valid")]
            return _med(vals)
        for (cname, lay), es in sorted(per.items()):
            if cname != name:
                continue
            valid = [bool((verdicts.get(e["out"]) or {}).get("valid")) for e in es]
            tot = [(e.get("report") or {}).get("time_total_s") for e, v in zip(es, valid) if v]
            mb, ml = med(base, "time_total_s"), _med(tot)
            vs = (med(lay, "time_verify_s") or 0) + (med(lay, "time_setup_s") or 0)
            wall = _med([e.get("wall_s") for e, v in zip(es, valid) if v])
            tree = max((e.get("peak_sampled_gib") or 0.0 for e in es), default=0.0)
            sums = max((e.get("peak_rss_gib") or 0.0 for e in es), default=0.0)
            lines.append(f"| {name} | {lay} | {sum(valid)}/{len(valid)} | {_span(tot)} | "
                         f"{'-' if not (mb and ml) else f'{mb / ml:.2f}'} | {wall or 0:.2f} | "
                         f"{med(lay, 'time_factor_s') or 0:.2f} | "
                         f"{med(lay, 'time_iterate_s') or 0:.2f} | {med(lay, 'time_certificate_s') or 0:.2f} | "
                         f"{vs:.2f} | {tree:.1f} / {sums:.1f} |")
    return lines


def cmd_table(a: argparse.Namespace) -> int:
    rows = read_jsonl(a.out / f"{a.tag}.jsonl")
    for r in rows:  # runs logged before peak_sampled_gib was recorded: from their guard summaries
        if r.get("event") == "guard" and r.get("peak_sampled_gib") is None:
            g = a.out / a.tag / r["case"] / f"guard-{r['placement']}-{r['split']}-r{r['rep']}.json"
            if g.exists():
                r["peak_sampled_gib"] = json.loads(g.read_text()).get("peak_sampled_gib")
    vpath = a.out / f"{a.tag}.eval.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    kinds = {r.get("kind") for r in rows}
    lines = [f"## {a.tag}", ""]
    if "bench" in kinds:
        lines += bench_table([r for r in rows if r.get("kind") == "bench"], a.base)
    if "hsolve" in kinds:
        lines += hsolve_table([r for r in rows if r.get("kind") == "hsolve"], verdicts, a.base)
    text = "\n".join(lines) + "\n"
    if a.md:
        Path(a.md).write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "eval", "table"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--root", type=Path, default=Path(os.environ.get("HS_DATA_ROOT", str(REPO / "data"))))
        if name == "run":
            p.add_argument("--kind", choices=("bench", "hsolve"), required=True)
            p.add_argument("--layouts", required=True)
            p.add_argument("--case", action="append", required=True)
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--rep0", type=int, default=0)
            p.add_argument("--kdir", type=Path, default=DEFAULT_OUT.parent / "k")
            p.add_argument("--bench-args", default="--modes count,full --nrhs 16 --solve-reps 3")
            p.add_argument("--target", default=None, help="JSON target replacing the case config's")
            p.add_argument("--one-cores", default="16-31")
            p.add_argument("--two-cores", default="8-15,16-23")
            p.add_argument("--mem-gb", type=float, default=32.0)
            p.add_argument("--timeout", type=int, default=1500)
            p.add_argument("--lock-wait", type=int, default=600)
        if name == "table":
            p.add_argument("--base", default="one:1x16")
            p.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "eval": cmd_eval, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
