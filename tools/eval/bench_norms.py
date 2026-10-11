"""Measure the ||.||_2 estimate of tools/eval/evaluate.py against the settings it replaced.

For every case and matrix (A, and B when present) ARPACK eigsh(k=1, which='LM') runs twice from the
same deterministic start vector:
    new   tol=evaluate.NORM_TOL, ncv=evaluate.NORM_NCV   (evaluate.hermitian_norm2's settings)
    old   tol=1e-8, scipy's default ncv (= 20 for k=1)   (the settings they replaced)
and the matrix-vector products, the wall time and the relative difference of the two estimates are
printed (and written as JSON with --json). This is the measurement behind the norm paragraph of the
evaluate.py docstring. Products are deterministic; times depend on the machine's load. Must run
under tools/run/guard.py, e.g.

    python3 tools/run/guard.py --skip-busy-check --cores 28,29 --mem-gb 8 --timeout 1800 -- \\
        python -m tools.eval.bench_norms syn-zero-cluster-real-1e5 wg-square-1e5 wg-halfslab-1e5 \\
        ss-qa8fk-qa8fm yee-pec-box-1e5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from tools import case as cio
from tools.eval import common, evaluate

OLD_TOL = 1e-8
SETTINGS = {"new": {"tol": evaluate.NORM_TOL, "ncv": evaluate.NORM_NCV}, "old": {"tol": OLD_TOL, "ncv": None}}


class _Counting(sla.LinearOperator):
    """m as a LinearOperator that counts its products."""

    def __init__(self, m: sp.spmatrix):
        super().__init__(m.dtype, m.shape)
        self.m, self.products = m, 0

    def _matvec(self, x):
        self.products += 1
        return self.m @ x


def measure(m: sp.spmatrix, tol: float, ncv: int | None) -> dict[str, Any]:
    """||m||_2 estimate as evaluate.hermitian_norm2 computes it above DENSE_MAX, with its cost."""
    n = m.shape[0]
    op = _Counting(m)
    t0 = time.perf_counter()
    w = sla.eigsh(op, k=1, which="LM", tol=tol, ncv=None if ncv is None else min(ncv, n),
                  v0=common.start_vector(n, m.dtype, 0), return_eigenvectors=False)
    return {"norm2": float(np.max(np.abs(w))), "products": op.products, "seconds": time.perf_counter() - t0}


def bench_case(name: str, root: str | Path) -> list[dict[str, Any]]:
    c = cio.load_case(name, root=root)
    rows = []
    for label, m in (("A", c.A), ("B", c.B)):
        if m is None:
            continue
        runs = {key: measure(m, **cfg) for key, cfg in SETTINGS.items()}
        rel = abs(runs["new"]["norm2"] - runs["old"]["norm2"]) / runs["old"]["norm2"]
        rows.append({"case": name, "matrix": label, "n": c.n, "nnz": int(m.nnz), **runs, "rel_diff": rel})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cost and agreement of the evaluator's ||.||_2 estimate (new vs old settings).")
    ap.add_argument("cases", nargs="+", help="case names (under ROOT/cases)")
    ap.add_argument("--root", default=str(cio.DEFAULT_ROOT), help="data root (default: %(default)s)")
    ap.add_argument("--json", type=Path, help="write the measurements here")
    args = ap.parse_args(argv)
    common.require_guard("bench_norms")
    rows = []
    print(f"{'case':32s} mat  {'new mv':>7s} {'old mv':>7s} {'new s':>7s} {'old s':>7s}  rel diff")
    for name in args.cases:
        for r in bench_case(name, args.root):
            rows.append(r)
            print(f"{r['case']:32s} {r['matrix']:3s}  {r['new']['products']:7d} {r['old']['products']:7d} "
                  f"{r['new']['seconds']:7.2f} {r['old']['seconds']:7.2f}  {r['rel_diff']:.1e}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
