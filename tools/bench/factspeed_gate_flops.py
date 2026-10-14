#!/usr/bin/env python3
"""End-to-end A/B of tools/bench/ab_cases.py runs, pooled and normalised for the pivot order's flops.

    python3 tools/bench/factspeed_gate_flops.py [--base main] [--new new] [--boot 10000]
        [--gate-reps 5] [--bar 1.05] [--md OUT.md] [--json OUT.json] RUNDIR [RUNDIR...]

P9 "factspeed" (results/p9_factspeed.md §4.5): at n = 1e6 SCOTCH draws a different pivot order in
every run (docs/DESIGN.md §3.4), and the flops of those orders scatter by more than the effect of a
faster factorization, so a 5-repetition median ratio of two binaries passes or fails a 1.05x bar by
the luck of the draw. This tool takes every valid run (ab_cases' rule: exit 0, success, tools/eval
pass) of one or more ab_cases run directories - several samples of the same two builds - and
reports per case, per sample and pooled:
  * the median solver time time_total_s of each binary [min-max], the ratio base/new (a speedup);
  * the medians of the pivot order's flops (report factor_stats.estimate.flops, RINFOG(1)) and of
    the factorization rate (numeric_s + time_certificate_s) / flops in s per Tflop - the main
    factorization and the certificate's two counts all eliminate on that order - and its ratio;
  * the rest of a run (time_total_s - numeric_s - time_certificate_s: analysis, iteration, I/O)
    and the end-to-end speedup at equal flops: (rest + F * rate) of base over that of new, F the
    median flops of all runs of the case (both binaries, all samples);
  * pooled: a bootstrap interval (2.5-97.5%) of both speedups, resampling each binary's runs, and
    the probability that a gate of --gate-reps runs per binary (median ratio) reaches --bar,
    drawing that many runs per binary from the pooled ones.
The run directories need the result directories (report.json, evaluation.json) of every run.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.bench.ab_cases import collect  # noqa: E402


def load(run_dirs: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """case -> label -> valid runs, each with its sample (run directory), flops, rate and rest."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rd in run_dirs:
        for case, labels in collect([rd]).items():
            for label, runs in labels.items():
                for r in runs:
                    if not r["valid"]:
                        continue
                    rep = json.loads((rd / case / f"{label}-r{r['rep']}" / "report.json").read_text())
                    flops = ((rep.get("factor_stats") or {}).get("estimate") or {}).get("flops")
                    fac = r["numeric_s"] + r["time_certificate_s"]
                    out.setdefault(case, {}).setdefault(label, []).append({
                        "sample": str(rd), "rep": r["rep"], "time": r["time_total_s"], "flops": flops,
                        "fac": fac, "rate": fac / (flops / 1e12), "rest": r["time_total_s"] - fac})
    return out


def med(xs: list[float]) -> float:
    return float(statistics.median(xs))


def stats(base: list[dict[str, Any]], new: list[dict[str, Any]], f_ref: float) -> dict[str, Any]:
    """Medians of both binaries' runs and the speedups (base over new: > 1 means new is faster);
    f_ref in flops, rates in s per Tflop."""
    tf = f_ref / 1e12

    def side(runs: list[dict[str, Any]]) -> dict[str, Any]:
        t = [r["time"] for r in runs]
        return {"runs": len(runs), "time": med(t), "min": min(t), "max": max(t),
                "flops": med([r["flops"] for r in runs]), "rate": med([r["rate"] for r in runs]),
                "rest": med([r["rest"] for r in runs])}
    b, n = side(base), side(new)
    return {"base": b, "new": n, "speedup": b["time"] / n["time"], "rate_speedup": b["rate"] / n["rate"],
            "equal_flops_speedup": (b["rest"] + tf * b["rate"]) / (n["rest"] + tf * n["rate"])}


def bootstrap(base: list[dict[str, Any]], new: list[dict[str, Any]], f_ref: float, n_boot: int,
              seed: int = 2026) -> dict[str, list[float]]:
    rng = random.Random(seed)
    sp, eq = [], []
    for _ in range(n_boot):
        s = stats([rng.choice(base) for _ in base], [rng.choice(new) for _ in new], f_ref)
        sp.append(s["speedup"])
        eq.append(s["equal_flops_speedup"])

    def interval(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]
    return {"speedup": interval(sp), "equal_flops_speedup": interval(eq)}


def gate_probability(base: list[dict[str, Any]], new: list[dict[str, Any]], reps: int, bar: float,
                     n_boot: int, seed: int = 2027) -> float:
    """Share of gates of `reps` runs per binary (drawn from the pooled runs) whose median speedup
    reaches `bar`."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        tb = med([rng.choice(base)["time"] for _ in range(reps)])
        tn = med([rng.choice(new)["time"] for _ in range(reps)])
        hits += tb / tn >= bar
    return hits / n_boot


def row(case: str, label: str, s: dict[str, Any]) -> str:
    """One markdown table row; the pooled row carries the bootstrap intervals."""
    sb, sn = s["base"], s["new"]
    boot = s.get("bootstrap")
    ci = (lambda k: f" [{boot[k][0]:.3f}-{boot[k][1]:.3f}]") if boot else (lambda k: "")
    return (f"| {case} | {label} | {sb['runs']} / {sn['runs']} | {sb['time']:.2f} [{sb['min']:.2f}-{sb['max']:.2f}] | "
            f"{sn['time']:.2f} [{sn['min']:.2f}-{sn['max']:.2f}] | {s['speedup']:.3f}{ci('speedup')} | "
            f"{sb['flops']:.3g} / {sn['flops']:.3g} | {sb['rate']:.2f} / {sn['rate']:.2f} | "
            f"{s['rate_speedup']:.3f} | {sb['rest']:.1f} / {sn['rest']:.1f} | "
            f"{s['equal_flops_speedup']:.3f}{ci('equal_flops_speedup')} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--base", default="main")
    ap.add_argument("--new", default="new")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples of the pooled runs")
    ap.add_argument("--gate-reps", type=int, default=5, help="runs per binary of one gate")
    ap.add_argument("--bar", type=float, default=1.05, help="speedup a gate must reach")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    dirs = [Path(p) for p in a.run_dirs]
    data = load(dirs)
    doc: dict[str, Any] = {"base": a.base, "new": a.new, "run_dirs": [str(d) for d in dirs],
                           "gate_reps": a.gate_reps, "bar": a.bar, "cases": {}}
    notes = []
    lines = [f"| case | sample | runs {a.base} / {a.new} | {a.base} s [min-max] | {a.new} s [min-max] | speedup | "
             f"flops {a.base} / {a.new} | s per Tflop {a.base} / {a.new} | per-flop speedup | "
             f"rest s {a.base} / {a.new} | speedup at equal flops |", "|" + "---|" * 11]
    for case in sorted(data):
        base, new = data[case].get(a.base, []), data[case].get(a.new, [])
        if not base or not new:
            continue
        f_ref = med([r["flops"] for r in base + new])
        entry: dict[str, Any] = {"flops_ref": f_ref, "samples": {}}
        groups = [(str(d), [r for r in base if r["sample"] == str(d)], [r for r in new if r["sample"] == str(d)])
                  for d in dirs] + [("pooled", base, new)]
        for name, b, n in groups:
            if not b or not n:
                continue
            s = stats(b, n, f_ref)
            if name == "pooled":
                s["bootstrap"] = bootstrap(b, n, f_ref, a.boot)
                s["gate_pass_probability"] = gate_probability(b, n, a.gate_reps, a.bar, a.boot)
                entry["pooled"] = s
                notes.append(f"{case} {s['gate_pass_probability']:.2f}")
            else:
                entry["samples"][name] = s
            label = "**pooled**" if name == "pooled" else "/".join(Path(name).parts[-2:])
            lines.append(row(case, label, s))
        doc["cases"][case] = entry
    lines += ["", "speedup = median time of the base / median time of the new binary (> 1: new faster); "
              "pooled intervals: bootstrap 2.5-97.5% over the runs of each binary.",
              "", f"Probability that a gate of {a.gate_reps} runs per binary (median ratio) reaches "
              f"{a.bar}x, drawn from the pooled runs: " + ", ".join(notes) + "."]
    md = "\n".join(lines) + "\n"
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1) + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
