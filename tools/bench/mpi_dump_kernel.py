#!/usr/bin/env python3
"""Write K(sigma) = A - sigma B of a data case as the binary input of bench/mpi_factor_bench.c (P9).

The bench is a standalone C program built against either MUMPS library (the MPI MUMPS of the
os2026-ref env, or the sequential MUMPS hsolve links) and cannot read npz files, so this converter
hands it the matrix hsolve factorizes for the case's nearest target: the upper triangle (i <= j,
1-based) of A - sigma B on the union pattern of A and B, as MUMPS takes a symmetric matrix (SYM = 2:
only one of a_ij / a_ji, duplicates are summed). Real cases only (a complex Hermitian case would be
the 2n real embedding; not needed for the P9 cases).

File layout (little endian), magic "HSKMAT1\\0":
    char[8] magic | int64 n | int64 nnz | int32 irn[nnz] | int32 jcn[nnz] | float64 a[nnz]
A JSON side file <out>.json records the case, sigma, n, nnz and, when the case reference is
complete, the exact inertia of K(sigma) (eigenvalues below / at / above sigma) - what every
factorization of the bench must reproduce.

usage (through tools/run/guard.py, in the os2026 env):
    python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 900 -- \\
        python3 tools/bench/mpi_dump_kernel.py --root DATA --case syn-mem27-100 --out K.bin [--sigma S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402

MAGIC = b"HSKMAT1\0"


def kernel_upper(A: sp.spmatrix, B: sp.spmatrix | None, sigma: float) -> sp.coo_matrix:
    """Upper triangle of A - sigma B on the union pattern (explicit zeros kept), sorted by column
    then row."""
    A = sp.csr_matrix(A)
    if np.iscomplexobj(A.data) or (B is not None and np.iscomplexobj(B.data)):
        raise SystemExit("mpi_dump_kernel: complex cases are not supported (real embedding not written)")
    K = A.astype(np.float64)
    if B is not None and sigma != 0.0:
        # Union pattern: add the B pattern with zeros first, so no entry of either is dropped.
        Bc = sp.csr_matrix(B, dtype=np.float64)
        K = K + Bc.multiply(0.0) + (-sigma) * Bc
    U = sp.triu(K, k=0, format="csc")
    U.sort_indices()
    return U.tocoo()


def write_kernel(path: Path, U: sp.coo_matrix) -> None:
    n = U.shape[0]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(np.array([n, U.nnz], dtype="<i8").tobytes())
        f.write((U.row.astype("<i4") + 1).tobytes())
        f.write((U.col.astype("<i4") + 1).tobytes())
        f.write(U.data.astype("<f8").tobytes())


def exact_inertia(ref: case.Reference | None, sigma: float, n: int) -> dict | None:
    """(below, at, above) of K(sigma) from a complete reference spectrum, else None."""
    if ref is None or not ref.info.get("complete") or len(ref.eigenvalues) != n:
        return None
    w = np.asarray(ref.eigenvalues).real
    return {"neg": int(np.sum(w < sigma)), "zero": int(np.sum(w == sigma)), "pos": int(np.sum(w > sigma))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", default=str(case.DEFAULT_ROOT))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma", type=float, default=None, help="default: the case config's nearest sigma")
    args = ap.parse_args(argv)
    c = case.load_case(args.case, root=args.root)
    tgt = c.config["target"]
    sigma = args.sigma if args.sigma is not None else float(tgt.get("sigma", 0.0))
    U = kernel_upper(c.A, c.B, sigma)
    write_kernel(args.out, U)
    info = {"case": c.name, "problem": c.meta["problem"], "sigma": sigma, "n": int(U.shape[0]),
            "nnz_upper": int(U.nnz), "file": str(args.out),
            "inertia": exact_inertia(case.load_ref(c.name, root=args.root), sigma, U.shape[0])}
    Path(str(args.out) + ".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
