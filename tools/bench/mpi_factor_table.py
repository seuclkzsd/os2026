#!/usr/bin/env python3
"""Tables of the P9 MPI x OpenMP MUMPS runs (tools/bench/mpi_factor_runs.sh output, P9 "mpi").

Reads the JSON lines of one or more runs files (bench/mpi_factor_bench.c events annotated with
layout / rep / order, plus the guard lines) and prints one markdown table per (file, order): per
layout the medians over the repetitions of the analysis, the inertia-only count (numeric
factorization with ICNTL(31) = 1), the numeric factorization keeping the factors, one block solve
(--nrhs right-hand sides; the median over every solve call), the factorization rate, MUMPS'
memory (INFOG(22) total over the ranks, INFOG(21) the largest rank) and the guard's process-tree
peak RSS, with the ratios against the baseline layout (default seq1x16: hsolve's sequential MUMPS,
16 threads). "inertia" is the INFOG(12) of every factorization of the layout: one value when they
all agree, and whether it equals the bench's --expect-neg (the exact count of the case) in every
run.

usage: python3 tools/bench/mpi_factor_table.py RUNS.jsonl... [--base seq1x16] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def summarize(lines: list[dict]) -> dict[str, dict[str, dict]]:
    """{order: {layout: summary}} of one runs file."""
    acc: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for d in lines:
        key = (d.get("order", "?"), d.get("layout", "?"))
        a = acc[key]
        ev = d.get("event")
        if ev == "analysis":
            a[f"analysis_{d['mode']}"].append(d["s"])
            a["flops_est"].append(d["flops_est"])
        elif ev == "factorize":
            a[f"factor_{d['mode']}"].append(d["s"])
            a["neg"].append(d["neg"])
            if d["mode"] == "full":
                a["flops"].append(d["flops"])
                a["mem_total_mb"].append(d["mem_total_mb"])
                a["mem_max_mb"].append(d["mem_max_mb"])
                a["entries"].append(d["entries"])
            else:
                a["count_mem_total_mb"].append(d["mem_total_mb"])
        elif ev == "solve":
            a["solve"].append(d["s"])
            a["nrhs"].append(d["nrhs"])
            a["residual"].append(d["rel_residual"])
        elif ev == "run":
            a["inertia_ok"].append(bool(d["inertia_ok"]))
            a["expect_neg"].append(d["expect_neg"])
            a["ranks"].append(d["ranks"])
            a["threads"].append(d["threads"])
        elif ev == "guard":
            a["peak_gib"].append(d["peak_rss_gib"])
            a["rc"].append(d["returncode"])
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (order, layout), a in acc.items():
        fl = median(a["flops"])
        ff = median(a["factor_full"])
        out[order][layout] = {
            "runs": len(a["rc"]), "failed_runs": sum(1 for r in a["rc"] if r != 0),
            "analysis_count_s": median(a["analysis_count"]), "analysis_full_s": median(a["analysis_full"]),
            "count_s": median(a["factor_count"]), "count_all_s": a["factor_count"],
            "factor_s": ff, "factor_all_s": a["factor_full"],
            "solve_s": median(a["solve"]), "nrhs": a["nrhs"][0] if a["nrhs"] else None,
            "max_residual": max(a["residual"]) if a["residual"] else None,
            "flops": fl, "gflops_per_s": fl / ff / 1e9 if fl and ff else None,
            "flops_est": median(a["flops_est"]), "entries": median(a["entries"]),
            "mem_total_mb": median(a["mem_total_mb"]), "mem_max_mb": median(a["mem_max_mb"]),
            "count_mem_total_mb": median(a["count_mem_total_mb"]), "peak_gib": max(a["peak_gib"]) if a["peak_gib"] else None,
            "neg": sorted(set(a["neg"])), "expect_neg": a["expect_neg"][0] if a["expect_neg"] else None,
            "inertia_ok": bool(a["inertia_ok"]) and all(a["inertia_ok"]) and len(a["inertia_ok"]) == len(a["rc"]),
        }
    return out


def _f(x, fmt="{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def _ratio(base, x) -> str:
    return "-" if base is None or x is None or x == 0 else f"{base / x:.2f}"


def table(name: str, order: str, rows: dict[str, dict], base: str) -> str:
    b = rows.get(base, {})
    head = (f"**{name}**, order `{order}` (medians; ratio = {base} time / layout time, > 1 = faster than {base})\n\n"
            "| layout | runs ok | analysis s (count / full) | count s | ratio | factor s | ratio | GF/s | solve s (nrhs) | ratio | "
            "flops | MUMPS MB total (max rank) | count MB total | peak GiB | inertia |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for lay in sorted(rows, key=lambda k: (k != base, -int(k.split("x")[-1].rstrip("h") or 0), k)):
        r = rows[lay]
        neg = ",".join(str(v) for v in r["neg"])
        lines.append(
            f"| {lay} | {r['runs'] - r['failed_runs']}/{r['runs']} | {_f(r['analysis_count_s'])} / {_f(r['analysis_full_s'])} | "
            f"{_f(r['count_s'])} | {_ratio(b.get('count_s'), r['count_s'])} | {_f(r['factor_s'])} | "
            f"{_ratio(b.get('factor_s'), r['factor_s'])} | {_f(r['gflops_per_s'], '{:.0f}')} | "
            f"{_f(r['solve_s'], '{:.3f}')} ({r['nrhs']}) | {_ratio(b.get('solve_s'), r['solve_s'])} | "
            f"{_f(r['flops'], '{:.3e}')} | {_f(r['mem_total_mb'], '{:.0f}')} ({_f(r['mem_max_mb'], '{:.0f}')}) | "
            f"{_f(r['count_mem_total_mb'], '{:.0f}')} | {_f(r['peak_gib'])} | "
            f"{neg} {'ok' if r['inertia_ok'] else 'MISMATCH'} |")
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--base", default="seq1x16")
    ap.add_argument("--json", type=Path, help="also write the summaries as JSON")
    a = ap.parse_args(argv)
    doc = {}
    for path in a.runs:
        lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip().startswith("{")]
        summ = summarize(lines)
        doc[path.stem] = summ
        for order, rows in summ.items():
            print(table(path.stem, order, rows, a.base))
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
