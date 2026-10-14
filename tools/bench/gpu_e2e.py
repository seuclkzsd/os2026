#!/usr/bin/env python3
"""Whole hsolve runs with the GPU offload off / on (P10 "gpu", results/p10_gpu.md).

Every run is tools/bench/compare.py's hsolve run (run_once: its own nested guard on the given cores
and RSS cap, the case's default configuration, tools/eval evaluation at the case tolerance - valid =
exit 0, success claimed, evaluation passed) with the environment variable HS_GPU set to the
variant (off / auto / on; the hsolve executable's BLAS hook reads it). Variants run interleaved, the
order reversed every other repetition. Recorded per run: solver time (report time_total_s), the
phases, the certificate (count, passed), the eigenvalues and the report's gpu object. Runs are
appended to <out-dir>/<tag>.jsonl, so the repetitions can be split over several slot calls; the
summary (<tag>.md, printed) covers every run in the file.

  python3 tools/bench/gpu_e2e.py --tag NAME [--reps 1] [--variants off,on] [--cores 16-31]
      [--mem-gb 32] [--timeout 1800] [--root DIR] [--out-dir DIR] [--summary-only] CASE ...

Run it in the measurement slot with exactly one GPU visible (the nested guards share the lock):
  tools/bench/gpu_slot.sh 7 2700 -- python3 tools/bench/gpu_e2e.py --tag p10-e2e-1e6 --reps 2 \\
      --root ~/p7_p7-membudget/data --mem-gb 32 syn-mem27-100 yee-pec-box-1e6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402
from tools.bench import compare  # noqa: E402

GPU_KEYS = ("mode", "available", "reason", "device", "init_s", "factorizations", "factorizations_offloaded",
            "calls", "gpu_gflop", "estimated_gflop", "gpu_share", "call_s", "predicted_cpu_s", "h2d_gb", "d2h_gb",
            "errors", "pipeline_s")


def run_variant(variant: str, c, tgt, out: Path, args, judge) -> dict:
    """One evaluated hsolve run with HS_GPU=variant."""
    old = os.environ.get("HS_GPU")
    os.environ["HS_GPU"] = variant
    try:
        e = compare.run_once("hsolve", c, tgt, out, args, None, judge)
    finally:
        if old is None:
            os.environ.pop("HS_GPU", None)
        else:
            os.environ["HS_GPU"] = old
    rep_path = out / "report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        g = rep.get("gpu") or {}
        cert = rep.get("certificate") or {}
        e["gpu"] = {k: g.get(k) for k in GPU_KEYS if k in g}
        e["certificate"] = {k: cert.get(k) for k in ("method", "performed", "passed", "count_inertia",
                                                       "count_found", "lower", "upper")}
        e["times"] = {k: rep.get(k) for k in ("time_total_s", "time_factor_s", "time_iterate_s",
                                               "time_certificate_s", "time_verify_s")}
    return e


def summarize(lines: list[dict]) -> str:
    out = ["| case | variant | valid | solver s median [min-max] | speedup vs off | certificate count | "
           "GPU factorizations | GPU GFlop (share) | GPU call s / CPU-model s | max eig. diff vs off |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    cases = sorted({r["case"] for r in lines})
    for name in cases:
        rows = [r for r in lines if r["case"] == name]
        variants = sorted({r["variant"] for r in rows}, key=lambda v: (v != "off", v))
        base = [r for r in rows if r["variant"] == "off" and r.get("valid")]
        base_t = statistics.median([r["solver_s"] for r in base]) if base else None
        base_eigs = np.asarray(base[0]["eigenvalues"]) if base else None
        for v in variants:
            rs = [r for r in rows if r["variant"] == v]
            good = [r for r in rs if r.get("valid")]
            ts = [r["solver_s"] for r in good]
            t = statistics.median(ts) if ts else None
            sp = f"{base_t / t:.3f}" if (t and base_t and v != "off" and len(good) == len(rs)) else "-"
            counts = sorted({(r.get("certificate") or {}).get("count_inertia") for r in rs}, key=str)
            g = [r.get("gpu") or {} for r in rs]
            nf = sorted({(x.get("factorizations_offloaded"), x.get("factorizations")) for x in g if x}, key=str)
            gf = [x.get("gpu_gflop") for x in g if x.get("gpu_gflop") is not None]
            sh = [x.get("gpu_share") for x in g if x.get("gpu_share") is not None]
            cs = [x.get("call_s") for x in g if x.get("call_s") is not None]
            ps = [x.get("predicted_cpu_s") for x in g if x.get("predicted_cpu_s") is not None]
            diff = "-"
            if base_eigs is not None and v != "off" and good:
                d = max(float(np.max(np.abs(np.sort(np.asarray(r["eigenvalues"])) - np.sort(base_eigs))
                                     / (np.abs(base_eigs).max() + 1e-300))) for r in good
                        if len(r["eigenvalues"]) == len(base_eigs))
                diff = f"{d:.1e}"
            out.append(
                f"| {name} | {v} | {len(good)}/{len(rs)} | "
                + (f"{t:.2f} [{min(ts):.2f}-{max(ts):.2f}]" if t else "-")
                + f" | {sp} | {', '.join(str(x) for x in counts)} | "
                + (", ".join(f"{a}/{b}" for a, b in nf) if nf else "-") + " | "
                + (f"{statistics.median(gf):.0f} ({statistics.median(sh):.2f})" if gf else "-") + " | "
                + (f"{statistics.median(cs):.1f} / {statistics.median(ps):.1f}" if cs else "-")
                + f" | {diff} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--variants", default="off,on")
    ap.add_argument("--cores", default=os.environ.get("HS_MEASURE_CORES", "16-31"))
    ap.add_argument("--mem-gb", type=float, default=32.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--root", default=str(case.DEFAULT_ROOT))
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/raw/p10gpu")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    jsonl = args.out_dir / f"{args.tag}.jsonl"
    if not args.summary_only:
        from tools.eval import common, evaluate
        common.require_guard("gpu_e2e")
        variants = args.variants.split(",")
        run_args = SimpleNamespace(cores=args.cores, mem_gb=args.mem_gb, timeout=args.timeout, exclusive=False,
                                   keep_vectors=False, root=args.root)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name in args.cases:
            c = case.load_case(name, root=args.root)
            tgt = dict(c.config["target"])
            norms = evaluate.estimate_norms(c.A, c.B)
            judge = compare.make_judge(c, tgt, norms)
            done = sum(1 for x in (jsonl.read_text().splitlines() if jsonl.exists() else [])
                       if json.loads(x)["case"] == name)
            first_rep = done // max(1, len(variants))
            for rep in range(first_rep, first_rep + args.reps):
                order = variants if rep % 2 == 0 else variants[::-1]
                for v in order:
                    out = REPO / "results/raw" / args.tag / name / f"{v}-r{rep}"
                    t0 = time.time()
                    e = run_variant(v, c, tgt, out, run_args, judge)
                    e.update(case=name, variant=v, rep=rep, started=t0)
                    with jsonl.open("a") as f:
                        f.write(json.dumps(e) + "\n")
                    g = e.get("gpu") or {}
                    print(f"{name} rep {rep} {v}: solver={e.get('solver_s')} valid={e.get('valid')} "
                          f"count={(e.get('certificate') or {}).get('count_inertia')} gpu_gflop={g.get('gpu_gflop')} "
                          f"why={'; '.join(e.get('why') or [])[:200]}", flush=True)
    lines = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()] if jsonl.exists() else []
    md = summarize(lines)
    (args.out_dir / f"{args.tag}.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
