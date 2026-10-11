"""Common on-disk format for test cases (shared by every generator and by tools/eval).

A case is a directory::

    <root>/<name>/
        A.npz              scipy.sparse.save_npz, CSR, full (both triangles) Hermitian storage
        B.npz              optional, same format (generalized problems)
        ref_eigenvalues.npy  optional, sorted ascending; exact (analytic) or trusted reference
        meta.json          description, provenance and expectations (schema below)

meta.json keys (all optional unless marked *):
    name*            case name (= directory name)
    family*          generator family, e.g. "waveguide", "yee", "synthetic", "suitesparse"
    problem*         "HEP" | "GHEP" | "GHIEP"
    n*, nnz_A*, nnz_B, dtype*   ("float64" | "complex128")
    description      one paragraph
    params           generator parameters (dict) - enough to regenerate the case bit-for-bit
    reference        {"kind": "analytic"|"discrete-exact"|"dense"|"solver"|"none", "note": str,
                      "count": int}   (eigenvalues themselves live in ref_eigenvalues.npy)
    inertia          {"A": [neg, zero, pos], "B": [...], "pencil": [neg, zero, pos]} when known
    targets          list of suggested queries [{"sigma": float, "k": int, "why": str}, ...]
    scenarios        list of tags: "zero-cluster", "multiplicity", "near-singular",
                     "indefinite-B", "complex-hermitian", "memory"
    license, citation  for third-party matrices
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (registers sp.linalg)

SCHEMA_VERSION = 1


def _check_hermitian(m: sp.spmatrix, name: str, tol: float = 1e-12) -> None:
    m = sp.csr_matrix(m)
    diff = m - m.conj().T
    nrm = sp.linalg.norm(m)
    if nrm > 0 and sp.linalg.norm(diff) > tol * nrm:
        raise ValueError(f"{name} is not Hermitian (relative defect {sp.linalg.norm(diff) / nrm:.3e})")


def save_case(root: str | Path, name: str, A: sp.spmatrix, B: sp.spmatrix | None = None,
              meta: dict[str, Any] | None = None, ref_eigenvalues: np.ndarray | None = None,
              check: bool = True) -> Path:
    """Write a case directory. A/B are converted to canonical CSR (sorted, no duplicates)."""
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    A = sp.csr_matrix(A)
    A.sum_duplicates()
    A.sort_indices()
    if check:
        _check_hermitian(A, "A")
    sp.save_npz(out / "A.npz", A, compressed=True)
    meta = dict(meta or {})
    meta.setdefault("name", name)
    meta["schema_version"] = SCHEMA_VERSION
    meta["n"] = int(A.shape[0])
    meta["nnz_A"] = int(A.nnz)
    meta["dtype"] = "complex128" if np.iscomplexobj(A.data) else "float64"
    if B is not None:
        B = sp.csr_matrix(B)
        B.sum_duplicates()
        B.sort_indices()
        if B.shape != A.shape:
            raise ValueError("B shape differs from A")
        if check:
            _check_hermitian(B, "B")
        sp.save_npz(out / "B.npz", B, compressed=True)
        meta["nnz_B"] = int(B.nnz)
        if np.iscomplexobj(B.data):
            meta["dtype"] = "complex128"
    elif (out / "B.npz").exists():
        (out / "B.npz").unlink()
    if ref_eigenvalues is not None:
        ref = np.sort(np.asarray(ref_eigenvalues, dtype=float))
        np.save(out / "ref_eigenvalues.npy", ref)
        meta.setdefault("reference", {})
        meta["reference"]["count"] = int(ref.size)
    for key in ("family", "problem"):
        if key not in meta:
            raise ValueError(f"meta[{key!r}] is required")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_jsonable))
    return out


def load_case(path: str | Path) -> tuple[sp.csr_matrix, sp.csr_matrix | None, dict[str, Any], np.ndarray | None]:
    """Return (A, B or None, meta, ref_eigenvalues or None)."""
    path = Path(path)
    A = sp.load_npz(path / "A.npz").tocsr()
    B = sp.load_npz(path / "B.npz").tocsr() if (path / "B.npz").exists() else None
    meta = json.loads((path / "meta.json").read_text())
    ref = np.load(path / "ref_eigenvalues.npy") if (path / "ref_eigenvalues.npy").exists() else None
    return A, B, meta, ref


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x)}")
