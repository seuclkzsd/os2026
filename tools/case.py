"""Single owner of every on-disk format shared by generators, evaluator, baselines and the CLI.

Formats are specified in docs/FORMATS.md; this module is the reference implementation. Nothing
else in tools/ may parse case/ref/result files directly (import from here instead).

Layout (root defaults to $HS_DATA_ROOT or <repo>/data):
    <root>/cases/<name>/A.npz          CSR, full Hermitian storage (scipy.sparse.save_npz)
    <root>/cases/<name>/B.npz          optional
    <root>/cases/<name>/meta.json      provenance / expectations (no reference values!)
    <root>/cases/<name>/config.json    default solver configuration (SolverConfig schema)
    <root>/refs/<name>/eigenvalues.npy reference eigenvalues (ascending) - never given to the solver
    <root>/refs/<name>/eigenvectors.npy optional (n x m, Fortran order)
    <root>/refs/<name>/ref.json        {"kind", "note", "count", "complete", "window"}
Result directory (written by bin/hsolve and by tools/eval/baseline_scipy.py):
    eigenvalues.npy, eigenvectors.npy, [complex_eigenvalues.npy, complex_eigenvectors.npy],
    report.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (registers sp.linalg)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
# HS_DATA_ROOT lets git worktrees share one data directory (data/ is not version-controlled).
DEFAULT_ROOT = Path(os.environ.get("HS_DATA_ROOT", str(REPO_ROOT / "data")))

PROBLEMS = ("HEP", "GHEP", "GHIEP")
REF_KINDS = ("analytic", "discrete-exact", "dense", "solver", "partial-analytic")
SCENARIOS = ("zero-cluster", "multiplicity", "near-singular", "indefinite-B", "complex-hermitian",
             "memory", "clustered", "large")
TARGET_MODES = ("nearest", "interval", "smallest", "largest")


# ----------------------------------------------------------------------------------- helpers

def _canonical_csr(m: sp.spmatrix | sp.sparray) -> sp.csr_matrix:
    m = sp.csr_matrix(m)
    m.sum_duplicates()
    m.sort_indices()
    return m


def hermitian_defect(m: sp.spmatrix) -> float:
    m = sp.csr_matrix(m)
    nrm = sp.linalg.norm(m)
    return 0.0 if nrm == 0 else float(sp.linalg.norm(m - m.conj().T) / nrm)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable) + "\n")


def default_config(sigma: float, k: int, tol: float = 1e-10, **extra: Any) -> dict[str, Any]:
    """A SolverConfig document for a 'nearest k' query (keys documented in docs/FORMATS.md §2)."""
    cfg: dict[str, Any] = {"target": {"mode": "nearest", "sigma": float(sigma), "k": int(k)}, "tol": tol}
    cfg.update(extra)
    return cfg


# ------------------------------------------------------------------------------------- cases

@dataclass
class Case:
    name: str
    path: Path
    A: sp.csr_matrix
    B: sp.csr_matrix | None
    meta: dict[str, Any]
    config: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.A.shape[0])


def save_case(name: str, A, B=None, *, meta: dict[str, Any], config: dict[str, Any] | None = None,
              root: str | Path = DEFAULT_ROOT, check_hermitian: float = 1e-12) -> Path:
    """Write <root>/cases/<name>/. meta must contain 'family' and 'problem'. Returns the case dir."""
    if "family" not in meta or "problem" not in meta:
        raise ValueError("meta needs 'family' and 'problem'")
    if meta["problem"] not in PROBLEMS:
        raise ValueError(f"meta['problem'] must be one of {PROBLEMS}")
    bad = [s for s in meta.get("scenarios", []) if s not in SCENARIOS]
    if bad:
        raise ValueError(f"unknown scenario tags {bad}; allowed {SCENARIOS}")
    for key in ("reference", "ref_eigenvalues", "eigenvalues"):
        if key in meta:
            raise ValueError(f"meta must not carry reference data ('{key}'); use save_ref()")
    out = Path(root) / "cases" / name
    out.mkdir(parents=True, exist_ok=True)
    A = _canonical_csr(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if check_hermitian is not None and hermitian_defect(A) > check_hermitian:
        raise ValueError(f"A is not Hermitian (defect {hermitian_defect(A):.3e})")
    sp.save_npz(out / "A.npz", A, compressed=True)
    full = dict(meta)
    full.update(schema_version=SCHEMA_VERSION, name=name, n=int(A.shape[0]), nnz_A=int(A.nnz),
                dtype="complex128" if np.iscomplexobj(A.data) else "float64")
    if B is not None:
        B = _canonical_csr(B)
        if B.shape != A.shape:
            raise ValueError("B shape differs from A")
        if check_hermitian is not None and hermitian_defect(B) > check_hermitian:
            raise ValueError(f"B is not Hermitian (defect {hermitian_defect(B):.3e})")
        sp.save_npz(out / "B.npz", B, compressed=True)
        full["nnz_B"] = int(B.nnz)
        if np.iscomplexobj(B.data):
            full["dtype"] = "complex128"
    elif (out / "B.npz").exists():
        (out / "B.npz").unlink()
    _dump(out / "meta.json", full)
    _dump(out / "config.json", config if config is not None else default_config(0.0, 6))
    return out


def load_case(name_or_path: str | Path, root: str | Path = DEFAULT_ROOT) -> Case:
    path = Path(name_or_path)
    if not (path / "A.npz").exists():
        path = Path(root) / "cases" / str(name_or_path)
    A = sp.load_npz(path / "A.npz").tocsr()
    B = sp.load_npz(path / "B.npz").tocsr() if (path / "B.npz").exists() else None
    meta = json.loads((path / "meta.json").read_text())
    config = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    return Case(name=path.name, path=path, A=A, B=B, meta=meta, config=config)


def list_cases(root: str | Path = DEFAULT_ROOT) -> list[str]:
    d = Path(root) / "cases"
    return sorted(p.name for p in d.iterdir() if (p / "A.npz").exists()) if d.exists() else []


# -------------------------------------------------------------------------------- references

@dataclass
class Reference:
    eigenvalues: np.ndarray            # ascending
    eigenvectors: np.ndarray | None    # n x m, columns match eigenvalues
    info: dict[str, Any] = field(default_factory=dict)


def save_ref(name: str, eigenvalues, *, kind: str, note: str = "", complete: bool = False,
             window: tuple[float, float] | None = None, eigenvectors: np.ndarray | None = None,
             root: str | Path = DEFAULT_ROOT) -> Path:
    """Write <root>/refs/<name>/. complete=True means: every eigenvalue inside `window`
    (or the whole spectrum if window is None) is listed - lets the evaluator detect misses."""
    if kind not in REF_KINDS:
        raise ValueError(f"kind must be one of {REF_KINDS}")
    out = Path(root) / "refs" / name
    out.mkdir(parents=True, exist_ok=True)
    w = np.asarray(eigenvalues)
    order = np.argsort(w.real if np.iscomplexobj(w) else w, kind="stable")
    w = w[order]
    np.save(out / "eigenvalues.npy", w)
    if eigenvectors is not None:
        np.save(out / "eigenvectors.npy", np.asfortranarray(np.asarray(eigenvectors)[:, order]))
    elif (out / "eigenvectors.npy").exists():
        (out / "eigenvectors.npy").unlink()
    _dump(out / "ref.json", {"schema_version": SCHEMA_VERSION, "kind": kind, "note": note,
                             "count": int(w.size), "complete": bool(complete),
                             "window": list(window) if window is not None else None})
    return out


def load_ref(name: str, root: str | Path = DEFAULT_ROOT) -> Reference | None:
    d = Path(root) / "refs" / name
    if not (d / "eigenvalues.npy").exists():
        return None
    vecs = np.load(d / "eigenvectors.npy") if (d / "eigenvectors.npy").exists() else None
    info = json.loads((d / "ref.json").read_text()) if (d / "ref.json").exists() else {}
    return Reference(np.load(d / "eigenvalues.npy"), vecs, info)


# ----------------------------------------------------------------------------------- results

@dataclass
class Result:
    eigenvalues: np.ndarray                 # real pairs, solver order
    eigenvectors: np.ndarray                # n x m
    complex_eigenvalues: np.ndarray | None
    complex_eigenvectors: np.ndarray | None
    report: dict[str, Any]


def save_result(out_dir: str | Path, eigenvalues, eigenvectors, report: dict[str, Any],
                complex_eigenvalues=None, complex_eigenvectors=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "eigenvalues.npy", np.asarray(eigenvalues, dtype=float))
    np.save(out / "eigenvectors.npy", np.asfortranarray(eigenvectors))
    for fname, arr in (("complex_eigenvalues.npy", complex_eigenvalues),
                       ("complex_eigenvectors.npy", complex_eigenvectors)):
        if arr is not None and np.size(arr):
            np.save(out / fname, arr)
        elif (out / fname).exists():
            (out / fname).unlink()
    report = dict(report)
    report.setdefault("schema_version", SCHEMA_VERSION)
    _dump(out / "report.json", report)
    return out


def load_result(out_dir: str | Path) -> Result:
    out = Path(out_dir)
    cev = out / "complex_eigenvalues.npy"
    cvec = out / "complex_eigenvectors.npy"
    return Result(
        eigenvalues=np.load(out / "eigenvalues.npy"),
        eigenvectors=np.load(out / "eigenvectors.npy"),
        complex_eigenvalues=np.load(cev) if cev.exists() else None,
        complex_eigenvectors=np.load(cvec) if cvec.exists() else None,
        report=json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {},
    )
