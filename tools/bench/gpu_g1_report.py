#!/usr/bin/env python3
"""Tables of the GPU experiment G1 (results/gpu_g1.md) from tools/bench/gpu_g1.sh's JSON lines.

    python3 tools/bench/gpu_g1_report.py results/gpu_g1.jsonl [--ref results/cmp-p16-best.json]

Per case and variant (backend:ordering of bench/gpu_ssids.cpp): medians over the repetitions of
analysis, cold factorization (the first), warm factorization (median of the later ones), one- and
p-column solves, worst backward errors, host peak RSS, GPU peak memory, and the speedups of the
factorization / solves against mumps:driver (hsolve's configuration). Inertia: every factorization of
every variant and repetition must report the same number of negative pivots and no zero pivot;
the table says "ok" or lists what differed.

Projection (--ref, the hsolve 16-core reference runs of tools/bench/compare.py): hsolve's measured
phases with the factorizations and solves scaled by the measured SSIDS / MUMPS ratios
  analysis  * analyse_X / analyse_M     (ssids:mumps: + the MUMPS analysis that made the order)
  numeric   * (cold_X + (F-1) warm_X) / (cold_M + (F-1) warm_M)      F = the run's factorizations
  solves    * solve1_X / solve1_M
  certificate (hsolve - hsolve-nocert, when its counts are not main factorizations)
            * warm_X / (0.87 warm_M)     (MUMPS counts discard the factors: 13% cheaper)
  + cuda_init_s for GPU variants; everything else (iteration, verification) unchanged.
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REF_VARIANT = "mumps:driver"
COUNTER_SAVING = 0.13  # src/driver/internal.hpp kCounterSaving


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (abs(x) < 1e-3 and x != 0):
        return f"{x:.1e}"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def summarize(runs: list[dict]) -> dict:
    ok = [r for r in runs if not r.get("failed")]
    s = {"runs": len(runs), "ok_runs": len(ok)}
    if not ok:
        return s
    s["analyse"] = med([r["analyse_s"] for r in ok])
    # ssids:mumps orders with the driver's MUMPS analysis (untimed in analyse_s): count it here.
    s["analyse_total"] = med([r["analyse_s"] + (r.get("order_source") or {}).get("analyse_s", 0.0) for r in ok])
    s["cold"] = med([r["factor_s"][0] for r in ok])
    s["warm"] = med([med(r["factor_s"][1:]) if len(r["factor_s"]) > 1 else r["factor_s"][0] for r in ok])
    s["solve1"] = med([r["solve1_s"] for r in ok])
    s["solveP"] = med([r["solveP_s"] for r in ok])
    s["block"] = ok[0].get("block")
    s["resid"] = max(max(r.get("resid1", 0.0), r.get("residP", 0.0)) for r in ok)
    s["rss"] = med([r["rss_peak_gib"] for r in ok])
    s["gpu_peak"] = med([r.get("gpu_mem_peak_mib") for r in ok])
    s["gpu_ctx"] = med([r.get("gpu_mem_ctx_mib") for r in ok])
    s["cuda_init"] = med([r.get("cuda_init_s") for r in ok])
    s["negs"] = sorted({n for r in ok for n in r["neg"]})
    s["zeros"] = sorted({z for r in ok for z in r["zero"]})
    s["oks"] = all(all(r["ok"]) for r in ok)
    st = ok[0].get("stats", {})
    s["entries"] = st.get("factor_entries")
    s["flops"] = st.get("factor_flops") or st.get("est_flops")
    s["mumps_ordering"] = ok[0].get("mumps_ordering") or (ok[0].get("order_source") or {}).get("mumps_ordering")
    return s


def ratio(a, b):
    return a / b if a is not None and b else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--ref", type=Path, default=Path("results/cmp-p16-best.json"))
    args = ap.parse_args()
    runs = defaultdict(list)
    order = []
    for line in args.jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Runs with extra bench arguments / SSIDS environment (tuning) form variants of their own.
        extra = " ".join(x for x in (r.get("bench_args", ""), r.get("ssids_env", "")) if x)
        key = (r["case"], r["variant"] + (f" [{extra}]" if extra else ""))
        if r["case"] not in order:
            order.append(r["case"])
        runs[key].append(r)
    summ = {k: summarize(v) for k, v in runs.items()}

    print("| case | variant | ordering | analyse s | factor cold s | factor warm s | solve-1 s | solve-p s "
          "| factor speedup | solve-1 speedup | solve-p speedup | neg | inertia | max resid | host peak GiB "
          "| GPU peak MiB | runs |")
    print("|" + "---|" * 17)
    for case in order:
        ref = next((s for (c, v), s in summ.items() if c == case and v.split(" ")[0] == REF_VARIANT), {})
        negs_all = sorted({n for (c, _), s in summ.items() if c == case for n in s.get("negs", [])})
        for (c, v), s in sorted(summ.items(), key=lambda kv: kv[0][1]):
            if c != case:
                continue
            if not s.get("ok_runs"):
                print(f"| {case} | {v} | | failed ({s['runs']} runs) |" + " |" * 13)
                continue
            inertia = "ok" if len(negs_all) == 1 and s["zeros"] == [0] and s["oks"] else \
                f"neg {s['negs']} zero {s['zeros']} (case: {negs_all})"
            ordering = v.split(":")[1]
            if s.get("mumps_ordering"):
                ordering += f" ({s['mumps_ordering']})"
            print(f"| {case} | {v} | {ordering} | {fmt(s['analyse'])} | {fmt(s['cold'])} | {fmt(s['warm'])} "
                  f"| {fmt(s['solve1'], 4)} | {fmt(s['solveP'], 4)} "
                  f"| {fmt(ratio(ref.get('warm'), s['warm']), 2)}x | {fmt(ratio(ref.get('solve1'), s['solve1']), 2)}x "
                  f"| {fmt(ratio(ref.get('solveP'), s['solveP']), 2)}x | {s['negs'][0] if len(s['negs']) == 1 else s['negs']} "
                  f"| {inertia} | {fmt(s['resid'])} | {fmt(s['rss'], 2)} | {fmt(s['gpu_peak'], 0)} "
                  f"| {s['ok_runs']}/{s['runs']} |")
    gpu_rows = [(c, v, s) for (c, v), s in summ.items() if v.startswith("ssids-gpu") and s.get("cuda_init") is not None]
    if gpu_rows:
        inits = [s["cuda_init"] for _, _, s in gpu_rows]
        firsts = [s["cold"] - s["warm"] for _, _, s in gpu_rows]
        print()
        print(f"CUDA context creation (cuda_init_s, medians of {len(gpu_rows)} case/variant groups): "
              f"{fmt(min(inits))}..{fmt(max(inits))} s; first-factorization extra (cold - warm): "
              f"{fmt(min(firsts))}..{fmt(max(firsts))} s; GPU memory after the context: "
              + ", ".join(sorted({fmt(s['gpu_ctx'], 0) for _, _, s in gpu_rows})) + " MiB")

    if not args.ref.exists():
        return 0
    ref = json.loads(args.ref.read_text())
    print()
    print("Projection of hsolve's 16-core solver time (s) with the measured ratios (see the docstring):")
    print()
    print("| case | hsolve ref | nocert ref | variant | analysis | numeric (F) | solves | certificate "
          "| projected hsolve | projected nocert | vs ref |")
    print("|" + "---|" * 11)
    for row in ref["rows"]:
        case = row["case"]
        m = next((s for (c, v), s in summ.items() if c == case and v.split(" ")[0] == REF_VARIANT), None)
        if not m or not m.get("ok_runs"):
            continue
        h, hn = row["solvers"]["hsolve"], row["solvers"]["hsolve-nocert"]
        ph, pn = h["phases"], hn["phases"]
        cert_extra = h["solver_s"] - hn["solver_s"]
        counts_in_numeric = ph["factorizations"] > pn["factorizations"]
        for v in sorted(k[1] for k in summ if k[0] == case and k[1].startswith("ssids")):
            x = summ[(case, v)]
            if not x.get("ok_runs"):
                continue

            def proj(phases, cert):
                f = int(phases["factorizations"])
                r_num = (x["cold"] + (f - 1) * x["warm"]) / (m["cold"] + (f - 1) * m["warm"])
                a = phases["analysis_s"] * x["analyse_total"] / m["analyse"]
                n = phases["numeric_s"] * r_num
                s = phases["solve_s"] * x["solve1"] / m["solve1"]
                c = cert * x["warm"] / ((1 - COUNTER_SAVING) * m["warm"]) if cert > 0 else 0.0
                base = (phases["analysis_s"] + phases["numeric_s"] + phases["solve_s"] + cert)
                total = h_or_n_total(phases) - base + a + n + s + c + (x["cuda_init"] or 0.0)
                return a, n, s, c, total

            def h_or_n_total(phases):
                return h["solver_s"] if phases is ph else hn["solver_s"]

            cert = 0.0 if counts_in_numeric else max(cert_extra, 0.0)
            a, n, s, c, th = proj(ph, cert)
            _, _, _, _, tn = proj(pn, 0.0)
            print(f"| {case} | {fmt(h['solver_s'], 2)} | {fmt(hn['solver_s'], 2)} | {v} | {fmt(a, 2)} "
                  f"| {fmt(n, 2)} ({int(ph['factorizations'])}) | {fmt(s, 2)} | {fmt(c, 2)} | {fmt(th, 2)} "
                  f"| {fmt(tn, 2)} | {fmt(h['solver_s'] / th, 2)}x / {fmt(hn['solver_s'] / tn, 2)}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
