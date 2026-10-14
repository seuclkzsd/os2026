#!/usr/bin/env python3
"""Write K(sigma) = A - sigma B of a data case as the binary input of bench/mpi_factor_bench.c (P9).

The bench is a standalone C program built against either MUMPS library (the MPI MUMPS of the
os2026-ref env, or the sequential MUMPS hsolve links) and cannot read npz files, so this converter
hands it the matrix hsolve factorizes for the case's nearest target - the values hsolve's
ShiftedPencil assembles (include/hsolve/sparse.hpp) on its pattern:
  * A and B as hsolve's loader passes them on (include/hsolve/io.hpp, default LoadOptions):
    duplicates summed, exact zeros dropped, and a matrix that is not exactly symmetric replaced by
    its symmetric part (A + A^T) / 2 on the union pattern of A and A^T;
  * the pattern is the upper triangle (i <= j) of A, the upper triangle of B (B = I for a standard
    problem: K(sigma) = A - sigma I) and the whole diagonal, structurally: an entry where
    a_ij - sigma b_ij cancels to 0 stays an (explicit zero) entry, as in hsolve;
  * 1-based indices, sorted by column then row; MUMPS takes it as a symmetric matrix (SYM = 2).
Real cases only (a complex Hermitian case would be the 2n real embedding; not needed for P9).

File layout (little endian), magic "HSKMAT1\\0":
    char[8] magic | int64 n | int64 nnz | int32 irn[nnz] | int32 jcn[nnz] | float64 a[nnz]
A JSON side file <out>.json records the case, sigma, n, nnz and, when the case reference is
complete and Sylvester's law gives it (HEP, or GHEP with B positive definite; not GHIEP), the
exact inertia of K(sigma) - eigenvalues below / at / above sigma - which every factorization of
the bench must reproduce.

Tests: tests/py/test_bench_mpi.py.

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
_INT32_MAX = 2**31 - 1


def loaded_upper(M: sp.spmatrix, what: str = "A") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, cols, values) of the upper triangle (row <= col) of M as hsolve's loader passes it
    on: duplicates summed, exact zeros dropped, and the symmetric part (M + M^T) / 2 when M is not
    exactly symmetric (0.5 a_ij + 0.5 a_ji equals (a_ij + a_ji) / 2 exactly; the diagonal is kept
    as it is). Indices are int64; a position may repeat (the caller sums the duplicates)."""
    if np.iscomplexobj(M.data):
        raise SystemExit(f"mpi_dump_kernel: {what} is complex; complex cases are not supported "
                         "(real embedding not written)")
    if M.shape[0] != M.shape[1]:
        raise SystemExit(f"mpi_dump_kernel: {what} is not square")
    M = sp.csr_matrix(M, dtype=np.float64, copy=True)
    M.sum_duplicates()
    M.eliminate_zeros()
    C = M.tocoo()
    r, c, v = C.row.astype(np.int64), C.col.astype(np.int64), C.data
    up = r <= c
    if (M != M.T).nnz == 0:                       # exactly symmetric: hsolve keeps it as it is
        return r[up], c[up], v[up]
    lo = ~up                                      # strict lower entries, mirrored to the upper triangle
    half = np.where(r[up] == c[up], 1.0, 0.5)
    return (np.concatenate([r[up], c[lo]]), np.concatenate([c[up], r[lo]]),
            np.concatenate([v[up] * half, 0.5 * v[lo]]))


def kernel_upper(A: sp.spmatrix, B: sp.spmatrix | None, sigma: float) -> sp.coo_matrix:
    """Upper triangle of A - sigma B (B = I when None) on hsolve's pencil pattern: upper(A) union
    upper(B) union the diagonal, explicit zeros kept (module docstring), sorted by column then row."""
    n = A.shape[0]
    if B is not None and B.shape != A.shape:
        raise SystemExit(f"mpi_dump_kernel: B is {B.shape}, A is {A.shape}")
    ra, ca, va = loaded_upper(A, "A")
    diag = np.arange(n, dtype=np.int64)
    if B is None:
        rb, cb, vb = diag, diag, np.full(n, -sigma)
    else:
        rb, cb, vb = loaded_upper(B, "B")
        vb = (-sigma) * vb
    rows = np.concatenate([ra, rb, diag])
    cols = np.concatenate([ca, cb, diag])
    vals = np.concatenate([va, vb, np.zeros(n)])  # the zeros put the whole diagonal in the pattern
    # COO -> CSC sums the duplicates of each position and keeps the entries that sum to zero.
    U = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    U.sum_duplicates()
    U.sort_indices()
    return U.tocoo()


def write_kernel(path: Path, U: sp.coo_matrix) -> None:
    n = U.shape[0]
    if n > _INT32_MAX:
        raise SystemExit(f"mpi_dump_kernel: n = {n} does not fit the int32 indices of the file")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(np.array([n, U.nnz], dtype="<i8").tobytes())
        f.write((U.row.astype(np.int64) + 1).astype("<i4").tobytes())
        f.write((U.col.astype(np.int64) + 1).astype("<i4").tobytes())
        f.write(U.data.astype("<f8").tobytes())


def read_kernel(path: Path) -> sp.coo_matrix:
    """The matrix of a file written by write_kernel (0-based COO, entries in file order)."""
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not an HSKMAT1 file")
    n, nnz = (int(x) for x in np.frombuffer(raw, dtype="<i8", count=2, offset=8))
    if len(raw) != 24 + 16 * nnz:
        raise ValueError(f"{path}: {len(raw)} bytes, expected {24 + 16 * nnz} for nnz = {nnz}")
    irn = np.frombuffer(raw, dtype="<i4", count=nnz, offset=24).astype(np.int64) - 1
    jcn = np.frombuffer(raw, dtype="<i4", count=nnz, offset=24 + 4 * nnz).astype(np.int64) - 1
    a = np.frombuffer(raw, dtype="<f8", count=nnz, offset=24 + 8 * nnz)
    return sp.coo_matrix((a.copy(), (irn, jcn)), shape=(n, n))


def exact_inertia(ref: case.Reference | None, sigma: float, n: int, problem: str) -> dict | None:
    """(below, at, above) of K(sigma) = A - sigma B from a complete reference spectrum, else None.
    Sylvester's law gives it for HEP and for GHEP (B positive definite); for GHIEP (B indefinite)
    the eigenvalue counts do not determine the inertia of K(sigma): None."""
    if problem not in ("HEP", "GHEP"):
        return None
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
            "inertia": exact_inertia(case.load_ref(c.name, root=args.root), sigma, U.shape[0],
                                     c.meta["problem"])}
    Path(str(args.out) + ".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
