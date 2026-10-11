"""Memory held by a matrix loaded through hsolve.load_matrix vs scipy.sparse.load_npz.

    PYTHONPATH=build/release/python python python/bench/load_memory.py PATH.npz --mode hsolve|scipy

Run each mode in its own process (through tools/run/guard.py): imports numpy, scipy.sparse and
hsolve, records the resident set size, loads the matrix, drops temporaries (gc), and prints one
JSON line with the RSS growth, the bytes of the three CSR arrays the result exposes, their
dtypes and which of them are views of the C++ buffers ('cpp') or numpy-allocated ('numpy').
The growth exceeds the exposed bytes when something else stays alive, e.g. a C++ buffer that
scipy copied instead of using it (numbers in docs/PYTHON.md section 3).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import scipy.sparse as sp

import hsolve

_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE


def owner(arr: np.ndarray) -> str:
    while isinstance(arr.base, np.ndarray):
        arr = arr.base
    if arr.flags.owndata:
        return "numpy"
    return "cpp" if type(arr.base).__name__ == "PyCapsule" else type(arr.base).__name__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--mode", choices=("hsolve", "scipy"), required=True)
    args = ap.parse_args()
    gc.collect()
    before = rss_bytes()
    t = time.perf_counter()
    if args.mode == "hsolve":
        m = hsolve.load_matrix(args.path)
    else:
        m = sp.csr_array(sp.load_npz(args.path).tocsr())
    seconds = time.perf_counter() - t
    gc.collect()
    grown = rss_bytes() - before
    arrays = {"indptr": m.indptr, "indices": m.indices, "data": m.data}
    exposed = sum(a.nbytes for a in arrays.values())
    mib = 1 << 20
    print(json.dumps({
        "mode": args.mode, "path": args.path, "n": m.shape[0], "nnz": int(m.nnz),
        "load_s": round(seconds, 3), "rss_growth_mib": round(grown / mib, 1),
        "exposed_mib": round(exposed / mib, 1),
        "arrays": {k: f"{a.dtype}/{owner(a)}" for k, a in arrays.items()},
    }))


if __name__ == "__main__":
    main()
