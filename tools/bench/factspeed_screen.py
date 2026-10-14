#!/usr/bin/env python3
"""Interleaved A/B screening of factorization variants with bench_factspeed_factor (P9 "factspeed").

    python3 tools/bench/factspeed_screen.py run SPEC.json RUNDIR
    python3 tools/bench/factspeed_screen.py table RUNDIR [--base NAME] [--md OUT.md] [--json OUT.json]

SPEC.json:
  {"bench": "/abs/bench_factspeed_factor",          # copied out of the work tree first
   "reps": 3,                                        # repetitions; variant order rotates per rep
   "jobs": "a",                                      # --jobs of every run (main, a, b)
   "common_args": ["--mumps-stats", "1"],
   "cases": {"syn100": {"case": "/abs/case dir", "lower": -5.37e-4, "upper": 5.98e-4,
                        "perm": "/abs/syn100.perm"}},  # --perm-in: the same pivot order everywhere
   "variants": {"base": {}, "k420": {"args": ["--keep", "420=256"]},
                "v3": {"env": {"LD_PRELOAD": "/abs/libdmumps_seq.so ..."}}},
   "timeout": 900}
`run` executes reps x cases x variants (one bench process each, output RUNDIR/<case>/<variant>-r<rep>.out)
and must itself run inside the measurement slot (guard.py --exclusive: the nested runs inherit its
cores and lock). `table` reports per case, job and variant the median wall time [min-max] over the
repetitions, the speedup against --base (median ratio), the rate RINFOG(3)/wall, MUMPS' time under
the L0 layer, and checks that every run of a case and job gave the same inertia (negative count
and validity) as the base - a variant that changes a count is flagged.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time


def run(spec_path, rundir):
    spec = json.load(open(spec_path))
    os.makedirs(rundir, exist_ok=True)
    json.dump(spec, open(os.path.join(rundir, "spec.json"), "w"), indent=1)
    names = list(spec["variants"])
    for rep in range(spec.get("reps", 3)):
        for cname, case in spec["cases"].items():
            os.makedirs(os.path.join(rundir, cname), exist_ok=True)
            order = names[rep % len(names):] + names[:rep % len(names)]
            if rep % 2 == 1:
                order = order[::-1]
            for vname in order:
                v = spec["variants"][vname]
                cmd = [spec["bench"], "--case", case["case"], "--jobs", spec.get("jobs", "a")]
                for key in ("lower", "upper", "sigma"):
                    if key in case:
                        cmd += ["--" + key, repr(case[key])]
                if "perm" in case:
                    cmd += ["--perm-in", case["perm"]]
                cmd += case.get("args", []) + spec.get("common_args", []) + v.get("args", [])
                env = dict(os.environ)
                env.update(v.get("env", {}))
                out = os.path.join(rundir, cname, "%s-r%d.out" % (vname, rep))
                t0 = time.time()
                with open(out, "w") as f:
                    f.write("# cmd: %s\n# env: %s\n" % (" ".join(cmd), json.dumps(v.get("env", {}))))
                    f.flush()
                    try:
                        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                            timeout=spec.get("timeout", 900)).returncode
                    except subprocess.TimeoutExpired:
                        rc = "timeout"
                print("%s %s r%d rc=%s %.1fs" % (cname, vname, rep, rc, time.time() - t0), flush=True)


def parse(path):
    """Factorization records of one run, each with the MUMPS 'under L0' time that precedes it."""
    recs, l0 = [], None
    with open(path) as f:
        for line in f:
            m = re.search(r"Elapsed time under L0\s+=\s+([0-9.]+)", line)
            if m:
                l0 = float(m.group(1))
            if line.startswith("{"):
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("event") == "factorization":
                    j["l0_s"] = l0
                    l0 = None
                    recs.append(j)
    return recs


def table(rundir, base, md_out, json_out):
    rows, result = [], {}
    for cname in sorted(os.listdir(rundir)):
        cdir = os.path.join(rundir, cname)
        if not os.path.isdir(cdir):
            continue
        data = {}  # (job, variant) -> list of records
        for fn in sorted(os.listdir(cdir)):
            m = re.match(r"(.+)-r(\d+)\.out$", fn)
            if not m:
                continue
            for r in parse(os.path.join(cdir, fn)):
                data.setdefault((r["job"], m.group(1)), []).append(r)
        jobs = sorted({k[0] for k in data})
        for job in jobs:
            variants = sorted({k[1] for k in data if k[0] == job}, key=lambda v: (v != base, v))
            bmed = None
            bneg = None
            if (job, base) in data:
                bmed = statistics.median(r["wall_s"] for r in data[(job, base)])
                bneg = {(r["neg"], r["valid"]) for r in data[(job, base)]}
            for v in variants:
                rs = data[(job, v)]
                w = [r["wall_s"] for r in rs]
                med = statistics.median(w)
                l0 = [r["l0_s"] for r in rs if r["l0_s"] is not None]
                negs = {(r["neg"], r["valid"]) for r in rs}
                same = bneg is None or negs == bneg
                row = {"case": cname, "job": job, "variant": v, "n": len(rs), "median_s": med,
                       "min_s": min(w), "max_s": max(w), "speedup": (bmed / med) if bmed else None,
                       "gflops": statistics.median(r["gflops"] for r in rs),
                       "l0_s": statistics.median(l0) if l0 else None,
                       "sys_s": statistics.median(r["sys_s"] for r in rs),
                       "inertia": sorted(negs), "inertia_as_base": same,
                       "ok": all(r["ok"] for r in rs)}
                rows.append(row)
    lines = ["| case | job | variant | runs | wall s median [min-max] | vs %s | GFlop/s | under L0 s | sys s | inertia (neg, valid) |" % base,
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %d | %.2f [%.2f-%.2f] | %s | %.0f | %s | %.1f | %s%s |" % (
            r["case"], r["job"], r["variant"], r["n"], r["median_s"], r["min_s"], r["max_s"],
            "%.3f" % r["speedup"] if r["speedup"] else "-", r["gflops"],
            "%.2f" % r["l0_s"] if r["l0_s"] is not None else "-", r["sys_s"],
            ", ".join("%d/%s" % (n, "v" if ok else "SINGULAR") for n, ok in r["inertia"]),
            "" if r["inertia_as_base"] else " **DIFFERS**"))
    text = "\n".join(lines)
    print(text)
    if md_out:
        open(md_out, "w").write(text + "\n")
    if json_out:
        json.dump(rows, open(json_out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("rundir")
    t = sub.add_parser("table")
    t.add_argument("rundir")
    t.add_argument("--base", default="base")
    t.add_argument("--md")
    t.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.spec, a.rundir)
    else:
        table(a.rundir, a.base, a.md, a.json)


if __name__ == "__main__":
    sys.exit(main())
