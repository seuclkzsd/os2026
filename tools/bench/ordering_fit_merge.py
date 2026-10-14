#!/usr/bin/env python3
"""Merge the raw runs of the ordering cost model fit (bench/anaint_fix_fit.cpp) after a change of
hsolve's nested dissection (P9, results/p9_ordering.md section 6).

  ordering_fit_merge.py --base DIR --nd RUNS [--nd RUNS ...] [--extra RUNS ...] --cases DIR [...]
                        --out OUT

--base DIR: the earlier fit data (ord16.jsonl, ord4.jsonl: one bench_anaint_orderings line per case,
ordering and run, raw or wrapped as {"returncode", "bench"}); its nested dissection rows (ordering
10) are dropped, every other row kept. --nd RUNS: directories of new bench_anaint_orderings outputs
(one JSON file per run, named o16_<case>_... for 16 threads and o4_<case>_... for 4 threads, as
run_cmds.sh writes them); their ordering-10 rows replace the dropped ones, the first directory
that has a case wins. --extra RUNS: directories whose rows of the other orderings are added (new
cases). --cases DIR: case directories (each holding A.npz, ...) linked into OUT/cases for
bench_anaint_fix_fit --data. Writes OUT/ord16.jsonl, OUT/ord4.jsonl, OUT/cases/.

P9:  ordering_fit_merge.py --base ~/p5b_anaint_fix --nd ~/p9_p9-ordering/m8 --nd ~/p9_p9-ordering/m5 \\
       --extra ~/p9_p9-ordering/m4 --cases ~/os2026/data/cases/* ~/p7_p7-membudget/data/cases/syn-mem27-100 \\
       ~/p7_p7-membudget/data/cases/yee-pec-box-1e6 --out ~/p9_p9-ordering/fit2
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def rows(path: str):
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        b = r.get("bench", r)
        if r.get("returncode", 0) != 0 or "case" not in b:
            continue
        yield b


def dir_rows(d: str, prefix: str):
    for f in sorted(glob.glob(os.path.join(d, f"{prefix}_*.json"))):
        yield from rows(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--nd", action="append", default=[])
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--cases", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cases"), exist_ok=True)
    for name, prefix in (("ord16", "o16"), ("ord4", "o4")):
        keep = [b for b in rows(os.path.join(a.base, f"{name}.jsonl")) if b["ordering"] != 10]
        nd: list[dict] = []
        seen: set[str] = set()
        for d in a.nd:
            new = [b for b in dir_rows(d, prefix) if b["ordering"] == 10 and b["case"] not in seen]
            nd += new
            seen |= {b["case"] for b in new}
        extra = [b for d in a.extra for b in dir_rows(d, prefix) if b["ordering"] != 10]
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as fh:
            for b in keep + nd + extra:
                fh.write(json.dumps(b) + "\n")
        print(f"{name}: {len(keep)} kept, {len(nd)} nested dissection rows, {len(extra)} extra rows")
    for c in a.cases:
        link = os.path.join(a.out, "cases", os.path.basename(os.path.normpath(c)))
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(c), link)
    print("cases", len(os.listdir(os.path.join(a.out, "cases"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
