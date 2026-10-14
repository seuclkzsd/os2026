#!/usr/bin/env python3
"""Tables of the P9 MPI x OpenMP MUMPS runs (tools/bench/mpi_factor_runs.sh output, P9 "mpi").

Reads the JSON lines of one or more runs files (bench/mpi_factor_bench.c events annotated with
layout / rep / order, plus the guard lines) and prints one markdown table per (file, order): per
layout the medians over the repetitions of the analysis, the inertia-only count (numeric
factorization with ICNTL(31) = 1), the numeric factorization keeping the factors, one block solve
(--nrhs right-hand sides; the median over every solve call), the factorization rate, MUMPS'
memory (INFOG(22) total over the ranks, INFOG(21) the largest rank) and the guard's process-tree
peak RSS, with the ratios against the baseline layout (default seq1x16: hsolve's sequential MUMPS,
16 threads). "flop balance" (bench builds from 872d39a on) is the elimination flops of the busiest
rank over the mean of the ranks (RINFO(3); 1 = perfectly balanced). "inertia" is the INFOG(12)
of every factorization of the layout (one value when they all agree) and its verdict: "ok" when
every run of the layout (one per guard line) ended with the bench's "run" event, the bench's own
check passed in each and every factorization's INFOG(12) equals the bench's --expect-neg (the
exact count of the case); "unchecked" when the runs were made without --expect-neg (the counts
then only agree with each other); "MISMATCH" otherwise - a count off, counts that disagree, or a
run that ended without its "run" event (a crash: its inertia is unknown). Lines of the bench run
directly (no layout annotation, no guard lines) are grouped under layout "?", one run per "run"
event.

Tests: tests/py/test_bench_mpi.py.

usage: python3 tools/bench/mpi_factor_table.py RUNS.jsonl... [--base seq1x16] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def inertia_verdict(negs: list[int], run_ok: list[bool], expect: list[int], runs: int) -> str:
    """"ok" / "unchecked" / "MISMATCH" of one layout (module docstring), "-" when nothing was
    factorized. negs: INFOG(12) of every factorization; run_ok / expect: inertia_ok / expect_neg of
    every "run" event; runs: the layout's runs (guard lines)."""
    if runs == 0 and not negs:
        return "-"
    if runs == 0 or len(run_ok) != runs or not all(run_ok) or not negs or len(set(expect)) > 1:
        return "MISMATCH"
    if expect[0] < 0:   # no --expect-neg: the bench checked only that its counts agree
        return "unchecked" if len(set(negs)) == 1 else "MISMATCH"
    return "ok" if all(v == expect[0] for v in negs) else "MISMATCH"


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
                if d.get("rank_flops_sum"):   # busiest rank's elimination flops over the mean
                    a["imbalance"].append(d["rank_flops_max"] * d["ranks"] / d["rank_flops_sum"])
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
        runs = len(a["rc"]) or len(a["inertia_ok"])    # direct bench output has no guard lines
        verdict = inertia_verdict(a["neg"], a["inertia_ok"], a["expect_neg"], runs)
        out[order][layout] = {
            "runs": runs, "failed_runs": sum(1 for r in a["rc"] if r != 0),
            "analysis_count_s": median(a["analysis_count"]), "analysis_full_s": median(a["analysis_full"]),
            "count_s": median(a["factor_count"]), "count_all_s": a["factor_count"],
            "factor_s": ff, "factor_all_s": a["factor_full"],
            "solve_s": median(a["solve"]), "nrhs": a["nrhs"][0] if a["nrhs"] else None,
            "max_residual": max(a["residual"]) if a["residual"] else None,
            "flops": fl, "gflops_per_s": fl / ff / 1e9 if fl and ff else None,
            "flops_est": median(a["flops_est"]), "entries": median(a["entries"]),
            "mem_total_mb": median(a["mem_total_mb"]), "mem_max_mb": median(a["mem_max_mb"]),
            "count_mem_total_mb": median(a["count_mem_total_mb"]), "peak_gib": max(a["peak_gib"]) if a["peak_gib"] else None,
            "imbalance": median(a["imbalance"]),
            "neg": sorted(set(a["neg"])), "expect_neg": a["expect_neg"][0] if a["expect_neg"] else None,
            "inertia": verdict, "inertia_ok": verdict == "ok",
        }
    return out


def _f(x, fmt="{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def _ratio(base, x) -> str:
    return "-" if base is None or x is None or x == 0 else f"{base / x:.2f}"


def _layout_key(layout: str, base: str) -> tuple:
    """Table order: the base layout, then by threads per rank (descending; RxT[h]), then the name;
    names that are no layout (e.g. "?" of direct bench output) last."""
    m = re.fullmatch(r"(?:seq)?(\d+)x(\d+)h?", layout)
    return (layout != base, m is None, -int(m.group(2)) if m else 0, layout)


def table(name: str, order: str, rows: dict[str, dict], base: str) -> str:
    b = rows.get(base, {})
    head = (f"**{name}**, order `{order}` (medians; ratio = {base} time / layout time, > 1 = faster than {base})\n\n"
            "| layout | runs ok | analysis s (count / full) | count s | ratio | factor s | ratio | GF/s | solve s (nrhs) | ratio | "
            "flops | flop balance | MUMPS MB total (max rank) | count MB total | peak GiB | inertia |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for lay in sorted(rows, key=lambda k: _layout_key(k, base)):
        r = rows[lay]
        neg = ",".join(str(v) for v in r["neg"])
        lines.append(
            f"| {lay} | {r['runs'] - r['failed_runs']}/{r['runs']} | {_f(r['analysis_count_s'])} / {_f(r['analysis_full_s'])} | "
            f"{_f(r['count_s'])} | {_ratio(b.get('count_s'), r['count_s'])} | {_f(r['factor_s'])} | "
            f"{_ratio(b.get('factor_s'), r['factor_s'])} | {_f(r['gflops_per_s'], '{:.0f}')} | "
            f"{_f(r['solve_s'], '{:.3f}')} ({r['nrhs']}) | {_ratio(b.get('solve_s'), r['solve_s'])} | "
            f"{_f(r['flops'], '{:.3e}')} | {_f(r['imbalance'])} | "
            f"{_f(r['mem_total_mb'], '{:.0f}')} ({_f(r['mem_max_mb'], '{:.0f}')}) | "
            f"{_f(r['count_mem_total_mb'], '{:.0f}')} | {_f(r['peak_gib'])} | "
            f"{neg} {r['inertia']} |")
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
