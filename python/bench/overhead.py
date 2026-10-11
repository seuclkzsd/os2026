"""Overhead of the in-process Python API on a test case (numbers in docs/PYTHON.md).

    PYTHONPATH=build/release/python python python/bench/overhead.py CASE --mode memory|path

  memory  scipy.sparse.load_npz (untimed), then hsolve.solve(A, B, config=config.json) (timed)
  path    hsolve.solve("A.npz", "B.npz", config=config.json) (timed; C++ loads the files)

Prints one JSON line per repeat: wall time of hsolve.solve, the solver's own time_total_s
(report), their difference (conversion + marshalling for 'memory'; + file loading for 'path'),
the success flag and the eigenvalues' agreement with the first repeat. Run each mode in its own
process through tools/run/guard.py, which reports the peak RSS; compare with the CLI
(build/release/hsolve --A ... --config ... --out DIR) under the same guard settings.
Cases live in $HS_DATA_ROOT/cases/<CASE> (docs/FORMATS.md section 1).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import hsolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--mode", choices=("memory", "path"), default="memory")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--data-root", default=os.environ.get("HS_DATA_ROOT", "data"))
    args = ap.parse_args()

    d = Path(args.data_root) / "cases" / args.case
    a_path, b_path, cfg = d / "A.npz", d / "B.npz", d / "config.json"
    b_path = b_path if b_path.exists() else None
    if args.mode == "memory":
        t = time.perf_counter()
        a = sp.load_npz(a_path)
        b = sp.load_npz(b_path) if b_path else None
        load_s = time.perf_counter() - t
    else:
        a, b, load_s = a_path, b_path, 0.0
    first = None
    for rep in range(args.repeat):
        t = time.perf_counter()
        res = hsolve.solve(a, b, config=cfg)
        wall = time.perf_counter() - t
        inner = res.report["time_total_s"]
        w = np.concatenate([res.eigenvalues.astype(complex), res.complex_eigenvalues])
        first = w if first is None else first
        diff = float(np.max(np.abs(np.sort_complex(w) - np.sort_complex(first)))) if w.size == first.size else None
        print(json.dumps({
            "case": args.case, "mode": args.mode, "repeat": rep, "n": res.report["n"],
            "nnz_a": res.report["nnz_a"], "scipy_load_s": round(load_s, 4), "solve_wall_s": round(wall, 4),
            "time_total_s": round(inner, 4), "overhead_s": round(wall - inner, 4),
            "success": res.success, "max_eig_diff_vs_first": diff,
        }), flush=True)


if __name__ == "__main__":
    main()
