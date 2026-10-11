"""Curated SuiteSparse Matrix Collection cases (electromagnetics and related indefinite problems).

Pipeline per matrix (archives are untrusted data):
  1. download https://sparse.tamu.edu/MM/<Group>/<Name>.tar.gz into a NEW empty directory
     <root>/downloads/<Name>/ (socket timeout, overall deadline, size cap; offline or HTTP errors
     raise Skipped with the reason). A finished download is recorded in download.json (url,
     bytes, sha256) and reused on later runs; anything else in the directory is never trusted:
     an incomplete directory is only replaced with --refresh.
  2. extract with tarfile filter='data' (no absolute paths, links out of the tree, devices) after
     checking the total unpacked size, and pick <Name>.mtx.
  3. scipy.io.mmread; a 'general' file holding a single triangle is symmetrized; explicit zeros are
     dropped; the matrix must be Hermitian (complex-symmetric matrices are rejected: they are not
     valid cases); an O(eps) defect is removed by taking the Hermitian part.
  4. problem by inspection: one matrix -> HEP; a pair -> GHEP when B is proven HPD by a symmetric
     diagonal-pivoting factorization (Sylvester inertia), GHIEP otherwise.
  5. targets: sigma inside the spectrum from a cheap Rayleigh-Ritz estimate (never from reference
     solutions): Krylov Ritz values of a Hermitian pencil with HPD B lie in [lambda_min, lambda_max]
     (Courant-Fischer), so any sigma between the extreme Ritz values is inside the spectrum.
  6. tools.case.save_case with license CC-BY 4.0 and the collection's citation. No reference is
     written (none is known).

Usage (from the repository root):
    python -m tools.gen.suitesparse --list
    python -m tools.gen.suitesparse ss-tem27623 ss-qa8fk-qa8fm [--root DIR] [--max-mb 200] [--refresh]
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tools import case
from tools.gen.synthetic import case_summary, hermitian_part

FAMILY = "suitesparse"
BASE_URL = "https://sparse.tamu.edu/MM"
LICENSE = "CC-BY 4.0"
CITATION = ("T. A. Davis and Y. Hu, The University of Florida Sparse Matrix Collection, ACM "
            "Transactions on Mathematical Software 38(1), 1:1-1:25, 2011, doi:10.1145/2049662.2049663")
HERMITIAN_TOL = 1e-12     # relative Frobenius defect accepted (then removed) as rounding
UNPACK_FACTOR = 20        # unpacked archive may be at most this many times the download cap


class Skipped(Exception):
    """A matrix or case was skipped: offline, over a size cap, unsafe archive or not Hermitian."""


@dataclass(frozen=True)
class Matrix:
    """One collection matrix. n, nnz, field and spd are the collection index values (ssstats.csv,
    2023-10); field/Hermitian-ness are re-checked on the downloaded file."""
    group: str
    name: str
    n: int
    nnz: int
    field: str
    spd: bool
    archive_mb: float
    notes: str


@dataclass(frozen=True)
class CaseSpec:
    """A case built from matrix `a` (and `b` for a pencil)."""
    a: str
    b: str | None = None
    notes: str = ""


REGISTRY: dict[str, Matrix] = {m.name: m for m in (
    Matrix("Guettel", "TEM27623", 27623, 439469, "real", False, 6.2,
           "electromagnetics (S. Guettel): FE matrix of a transient-electromagnetics (TEM) model; "
           "real symmetric, not positive definite per the index"),
    Matrix("Guettel", "TEM152078", 152078, 6459326, "real", False, 80.3,
           "larger TEM electromagnetics model of the same family as TEM27623"),
    Matrix("Guettel", "TEM181302", 181302, 7839010, "real", False, 98.7,
           "largest TEM electromagnetics model of the family"),
    Matrix("Um", "2cubes_sphere", 101492, 1647264, "real", True, 11.8,
           "electromagnetics FE model (two cubes inside a sphere), symmetric positive definite: "
           "interior sigma makes A - sigma I indefinite"),
    Matrix("Um", "offshore", 259789, 4242673, "real", True, 33.6,
           "electromagnetics FE model of an offshore (marine) EM survey, symmetric positive definite"),
    Matrix("Dziekonski", "gsm_106857", 589446, 21758924, "real", False, 152.7,
           "electromagnetics FE model (Dziekonski), real symmetric indefinite; large (589k)"),
    Matrix("Cunningham", "qa8fk", 66127, 1660579, "real", False, 4.9,
           "3-D acoustics FE stiffness matrix; pairs with the mass matrix qa8fm"),
    Matrix("Cunningham", "qa8fm", 66127, 1660579, "real", True, 4.4,
           "3-D acoustics FE mass matrix (SPD) belonging to qa8fk"),
    Matrix("GHS_indef", "helm3d01", 32226, 428444, "real", False, 3.0,
           "3-D Helmholtz operator, symmetric indefinite"),
    Matrix("GHS_indef", "helm2d03", 392257, 2741935, "real", False, 9.4,
           "2-D Helmholtz operator, symmetric indefinite"),
    Matrix("PARSEC", "Si34H36", 97569, 5156379, "real", False, 15.6,
           "real-space DFT Hamiltonian (PARSEC), symmetric indefinite, interior spectrum of interest"),
    Matrix("PARSEC", "Ge87H76", 112985, 7892195, "real", False, 34.8,
           "real-space DFT Hamiltonian (PARSEC), symmetric indefinite"),
    Matrix("PARSEC", "Ga10As10H30", 113081, 6115633, "real", False, 24.7,
           "real-space DFT Hamiltonian (PARSEC), symmetric indefinite"),
    Matrix("Sinclair", "3Dspectralwave2", 292008, 12935272, "complex", False, 61.7,
           "3-D spectral wave model, complex with symmetric pattern: valid only if Hermitian "
           "(complex-symmetric matrices are rejected after download)"),
)}

CASES: dict[str, CaseSpec] = {
    "ss-tem27623": CaseSpec("TEM27623"),
    "ss-tem152078": CaseSpec("TEM152078"),
    "ss-tem181302": CaseSpec("TEM181302"),
    "ss-2cubes_sphere": CaseSpec("2cubes_sphere"),
    "ss-offshore": CaseSpec("offshore"),
    "ss-gsm_106857": CaseSpec("gsm_106857"),
    "ss-qa8fk-qa8fm": CaseSpec("qa8fk", "qa8fm", "acoustic stiffness/mass pencil K x = lambda M x"),
    "ss-helm3d01": CaseSpec("helm3d01"),
    "ss-helm2d03": CaseSpec("helm2d03"),
    "ss-si34h36": CaseSpec("Si34H36"),
    "ss-ge87h76": CaseSpec("Ge87H76"),
    "ss-ga10as10h30": CaseSpec("Ga10As10H30"),
    "ss-3dspectralwave2": CaseSpec("3Dspectralwave2"),
}


# ================================================================================ download


def _download(url: str, dest: Path, *, max_bytes: int, timeout: float, max_seconds: float) -> dict[str, Any]:
    """Stream url to dest under a size cap; returns {url, bytes, sha256}. Raises Skipped."""
    req = urllib.request.Request(url, headers={"User-Agent": "hsolve-testgen/1"})
    deadline = time.monotonic() + max_seconds
    digest, total = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            length = resp.headers.get("Content-Length")
            if length is not None and int(length) > max_bytes:
                raise Skipped(f"{url}: {int(length) / 1e6:.1f} MB exceeds the {max_bytes / 1e6:.1f} MB cap")
            while chunk := resp.read(1 << 20):
                total += len(chunk)
                if total > max_bytes:
                    raise Skipped(f"{url}: more than the {max_bytes / 1e6:.1f} MB cap")
                if time.monotonic() > deadline:
                    raise Skipped(f"{url}: download exceeded {max_seconds:.0f} s")
                digest.update(chunk)
                out.write(chunk)
            if length is not None and total != int(length):
                raise Skipped(f"{url}: truncated transfer ({total} of {int(length)} bytes)")
    except Skipped:
        dest.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        dest.unlink(missing_ok=True)
        raise Skipped(f"{url}: download failed ({exc}); offline?") from exc
    return {"url": url, "bytes": total, "sha256": digest.hexdigest()}


def _extract_mtx(archive: Path, dest: Path, name: str, max_unpacked: int) -> Path:
    """Extract the archive into dest (filter='data') and return the path of <name>.mtx."""
    try:
        with tarfile.open(archive, "r:gz") as tf:
            unpacked = sum(m.size for m in tf.getmembers() if m.isfile())
            if unpacked > max_unpacked:
                raise Skipped(f"{archive.name}: unpacks to {unpacked / 1e6:.0f} MB "
                              f"(cap {max_unpacked / 1e6:.0f} MB)")
            tf.extractall(dest, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise Skipped(f"{archive.name}: unusable archive ({exc})") from exc
    found = sorted(p for p in dest.rglob(f"{name}.mtx") if p.is_file())
    if len(found) != 1:
        raise Skipped(f"{archive.name}: expected exactly one {name}.mtx, found {len(found)}")
    return found[0]


def obtain(matrix: Matrix, downloads: Path, *, base_url: str = BASE_URL, max_mb: float = 200.0,
           timeout: float = 60.0, max_seconds: float = 1800.0,
           refresh: bool = False) -> tuple[Path, dict[str, Any]]:
    """Download and extract one matrix into downloads/<Name>/; returns (.mtx path, download record).

    The directory must be new and empty; a completed earlier download (download.json, written
    last) is reused; refresh=True deletes the directory and starts over. A failed fresh download
    removes the directory it created, so a rerun starts clean.
    """
    folder = downloads / matrix.name
    extract_dir = folder / "extract"
    record_path = folder / "download.json"
    if folder.exists() and refresh:
        shutil.rmtree(folder)
    if folder.exists():
        if not record_path.is_file():
            raise Skipped(f"{folder} exists without a completed download; rerun with --refresh")
        record = json.loads(record_path.read_text())
        mtx = extract_dir / record["mtx"]
        if not mtx.is_file():
            raise Skipped(f"{mtx} is missing; rerun with --refresh")
        return mtx, record
    folder.mkdir(parents=True)
    try:
        url = f"{base_url}/{matrix.group}/{matrix.name}.tar.gz"
        archive = folder / f"{matrix.name}.tar.gz"
        max_bytes = int(max_mb * 1e6)
        record = _download(url, archive, max_bytes=max_bytes, timeout=timeout, max_seconds=max_seconds)
        mtx = _extract_mtx(archive, extract_dir, matrix.name, UNPACK_FACTOR * max_bytes)
    except Skipped:
        shutil.rmtree(folder)
        raise
    record["mtx"] = str(mtx.relative_to(extract_dir))
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return mtx, record


# ============================================================================== conversion


def _restore_triangle(a: sp.csr_matrix) -> tuple[sp.csr_matrix, str | None]:
    """Full Hermitian storage from a matrix holding a single triangle (same rule as src/io's
    Storage::Auto: one strict triangle empty, the other not)."""
    lower, upper = sp.tril(a, -1, format="csr"), sp.triu(a, 1, format="csr")
    if lower.nnz and not upper.nnz:
        strict, kind = lower, "lower"
    elif upper.nnz and not lower.nnz:
        strict, kind = upper, "upper"
    else:
        return a, None
    full = strict + strict.conj().T + sp.diags(a.diagonal())
    return full.tocsr(), f"'general' file stores only the {kind} triangle: symmetrized"


def read_hermitian(path: Path) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Read a MatrixMarket file as a full Hermitian CSR matrix. Raises Skipped otherwise.

    Returns (A, info) with info = {mm_header, notes}.
    """
    rows, cols, entries, fmt, field, symmetry = scipy.io.mminfo(path)
    header = {"rows": rows, "cols": cols, "entries": entries, "format": fmt, "field": field,
              "symmetry": symmetry}
    if rows != cols:
        raise Skipped(f"{path.name}: not square ({rows} x {cols})")
    if field == "pattern" or symmetry == "skew-symmetric":
        raise Skipped(f"{path.name}: {field}/{symmetry} matrix is not a Hermitian operator")
    a = sp.csr_matrix(scipy.io.mmread(path, spmatrix=False))
    a.sum_duplicates()
    notes = []
    if symmetry == "general":
        a, note = _restore_triangle(a)
        if note:
            notes.append(note)
    stored = a.nnz
    a.eliminate_zeros()
    if a.nnz < stored:
        notes.append(f"dropped {stored - a.nnz} explicit zeros")
    defect = case.hermitian_defect(a)
    if defect > HERMITIAN_TOL:
        if np.iscomplexobj(a.data) and spla.norm(a - a.T) <= HERMITIAN_TOL * spla.norm(a):
            raise Skipped(f"{path.name}: complex symmetric (A = A^T), not Hermitian; not a valid case")
        raise Skipped(f"{path.name}: not Hermitian (relative defect {defect:.2e})")
    if defect > 0:
        a = hermitian_part(a)
        notes.append(f"Hermitian defect {defect:.2e} removed (Hermitian part)")
    if np.iscomplexobj(a.data) and not np.any(a.data.imag):
        a = sp.csr_matrix(a.real)
        notes.append("complex file with zero imaginary parts: stored as real")
    return a, {"mm_header": header, "notes": notes}


# ============================================================================== inspection


def hpd_inertia(b: sp.spmatrix) -> list[int] | None:
    """Exact inertia [neg, zero, pos] of a Hermitian matrix via LU with symmetric diagonal
    pivoting, or None when the factorization had to leave the diagonal.

    SuperLU in SymmetricMode with diag_pivot_thresh=0 pivots on the diagonal whenever it is
    nonzero, so perm_r == perm_c and P^T B P = L U with U = D L^H: the signs of diag(U) give the
    inertia (Sylvester). An HPD matrix never produces a zero diagonal pivot, hence a factorization
    that pivots off the diagonal (or breaks down) proves B is not HPD.
    """
    try:
        lu = spla.splu(sp.csc_matrix(b), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options={"SymmetricMode": True})
    except RuntimeError:  # exactly singular
        return None
    if not np.array_equal(lu.perm_r, lu.perm_c):
        return None
    d = lu.U.diagonal().real
    neg, zero = int(np.count_nonzero(d < 0)), int(np.count_nonzero(d == 0))
    return [neg, zero, int(d.size) - neg - zero]


def ritz_interval(a: sp.spmatrix, b: sp.spmatrix | None = None, steps: int = 40,
                  seed: int = 0) -> tuple[float, float]:
    """Extreme Ritz values of the pencil (A, B), B HPD (None = I), on a Krylov subspace.

    Basis: K_m(D^{-1} A, v0) with D = diag(B) (Jacobi-preconditioned, so no solve with B is needed),
    orthonormalized by CGS2; Rayleigh-Ritz on (V^H A V, V^H B V). By Courant-Fischer every Ritz
    value lies in [lambda_min, lambda_max] for ANY subspace, so the returned (lo, hi) satisfy
    lambda_min <= lo <= hi <= lambda_max: points between them are inside the spectrum. Cost:
    `steps` products with A and O(n steps^2) orthogonalization.
    """
    n = a.shape[0]
    cplx = np.iscomplexobj(a.data) or (b is not None and np.iscomplexobj(b.data))
    dtype = complex if cplx else float
    diag = None if b is None else b.diagonal().real
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0)
    basis = np.empty((n, min(steps, n)), dtype=dtype)
    m = 0
    while m < basis.shape[1]:
        scale = np.linalg.norm(v)
        for _ in range(2):  # CGS2: two passes keep the basis orthonormal to working precision
            v = v - basis[:, :m] @ (basis[:, :m].conj().T @ v)
        nv = np.linalg.norm(v)
        if nv <= 1e-10 * scale:  # invariant subspace reached
            break
        basis[:, m] = v / nv
        v = a @ basis[:, m]
        if diag is not None:
            v = v / diag
        m += 1
    q = basis[:, :m]
    ah = q.conj().T @ (a @ q)
    bh = np.eye(m) if b is None else q.conj().T @ (b @ q)
    theta = scipy.linalg.eigh((ah + ah.conj().T) / 2, (bh + bh.conj().T) / 2, eigvals_only=True)
    return float(theta[0]), float(theta[-1])


def gershgorin_interval(a: sp.spmatrix) -> tuple[float, float]:
    """Gershgorin bounds [min(a_ii - r_i), max(a_ii + r_i)] of a Hermitian matrix."""
    a = sp.csr_matrix(a)
    d = a.diagonal().real
    r = np.asarray(abs(a).sum(axis=1)).ravel() - np.abs(d)
    return float(np.min(d - r)), float(np.max(d + r))


def choose_targets(a: sp.spmatrix, b: sp.spmatrix | None, problem: str,
                   k: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Targets inside the spectrum from Ritz bounds; returns (targets, selection record)."""
    if problem == "GHIEP":
        return ([{"sigma": 0.0, "k": k, "why": "B indefinite: no Courant-Fischer bounds; origin"}],
                {"method": "convention (B indefinite)"})
    lo, hi = ritz_interval(a, b)
    targets = []
    if lo < 0.0 < hi:
        targets.append({"sigma": 0.0, "k": k, "why": "indefinite: sigma=0 lies between the extreme Ritz values"})
    targets.append({"sigma": lo + 0.01 * (hi - lo), "k": k, "why": "low end: 1% into the Ritz interval"})
    targets.append({"sigma": lo + 0.5 * (hi - lo), "k": k, "why": "deep interior: middle of the Ritz interval"})
    record: dict[str, Any] = {"method": "Rayleigh-Ritz on a 40-step Krylov space (Courant-Fischer)",
                              "ritz_interval": [lo, hi]}
    if b is None:
        record["gershgorin_interval"] = list(gershgorin_interval(a))
    return targets, record


# ================================================================================== cases


def fetch_case(name: str, *, root: str | Path = case.DEFAULT_ROOT, cases: dict[str, CaseSpec] = CASES,
               registry: dict[str, Matrix] = REGISTRY, base_url: str = BASE_URL, max_mb: float = 200.0,
               timeout: float = 60.0, max_seconds: float = 1800.0, refresh: bool = False) -> Path:
    """Download, convert, inspect and save one case; returns the case directory. Raises Skipped."""
    spec = cases[name]
    downloads = Path(root) / "downloads"
    mats, sources = [], []
    for key in (spec.a, spec.b):
        if key is None:
            continue
        matrix = registry[key]
        mtx, record = obtain(matrix, downloads, base_url=base_url, max_mb=max_mb, timeout=timeout,
                             max_seconds=max_seconds, refresh=refresh)
        m, info = read_hermitian(mtx)
        mats.append(m)
        sources.append({"group": matrix.group, "name": matrix.name, "url": record["url"],
                        "archive_bytes": record["bytes"], "archive_sha256": record["sha256"],
                        "collection_notes": matrix.notes, **info})
    a = mats[0]
    b = mats[1] if len(mats) > 1 else None
    inertia: dict[str, list[int]] = {}
    if b is None:
        problem = "HEP"
    else:
        if b.shape != a.shape:
            raise Skipped(f"{name}: A is {a.shape}, B is {b.shape}")
        b_inertia = hpd_inertia(b)
        problem = "GHEP" if b_inertia is not None and b_inertia[0] == b_inertia[1] == 0 else "GHIEP"
        if b_inertia is not None:
            inertia["B"] = b_inertia
    targets, selection = choose_targets(a, b, problem)
    complex_ = np.iscomplexobj(a.data) or (b is not None and np.iscomplexobj(b.data))
    scenarios = [s for s, on in (("complex-hermitian", complex_), ("indefinite-B", problem == "GHIEP"),
                                 ("large", a.shape[0] >= 100_000)) if on]
    description = "; ".join(filter(None, [spec.notes, registry[spec.a].notes]))
    meta = {"family": FAMILY, "problem": problem, "description": description,
            "params": {"source": "SuiteSparse Matrix Collection", "matrices": sources,
                       "target_selection": selection},
            "targets": targets, "scenarios": scenarios, "license": LICENSE,
            "citation": f"{CITATION}; matrices: " + ", ".join(f"{s['group']}/{s['name']}" for s in sources)}
    if inertia:
        meta["inertia"] = inertia
    return case.save_case(name, a, b, meta=meta, config=case.default_config(targets[0]["sigma"], targets[0]["k"]),
                          root=root)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.gen.suitesparse",
                                 description="Download SuiteSparse matrices and convert them to cases.")
    ap.add_argument("cases", nargs="*", help="case names (see --list)")
    ap.add_argument("--list", action="store_true", help="list cases with archive sizes and exit")
    ap.add_argument("--root", type=Path, default=case.DEFAULT_ROOT, help="data root")
    ap.add_argument("--base-url", default=BASE_URL, help="collection (or mirror) MM/ URL")
    ap.add_argument("--max-mb", type=float, default=200.0, help="size cap per archive (MB)")
    ap.add_argument("--timeout", type=float, default=60.0, help="socket timeout (s)")
    ap.add_argument("--max-seconds", type=float, default=1800.0, help="deadline per download (s)")
    ap.add_argument("--refresh", action="store_true", help="discard earlier downloads and fetch again")
    args = ap.parse_args(argv)
    if args.list or not args.cases:
        for cname, spec in CASES.items():
            keys = [k for k in (spec.a, spec.b) if k]
            mb = sum(REGISTRY[k].archive_mb for k in keys)
            print(f"{cname:22s} {'+'.join(REGISTRY[k].group + '/' + k for k in keys):40s} {mb:7.1f} MB")
        return 0
    unknown = [c for c in args.cases if c not in CASES]
    if unknown:
        ap.error(f"unknown cases {unknown}; see --list")
    failures = 0
    for cname in args.cases:
        t0 = time.perf_counter()
        try:
            path = fetch_case(cname, root=args.root, base_url=args.base_url, max_mb=args.max_mb,
                              timeout=args.timeout, max_seconds=args.max_seconds, refresh=args.refresh)
        except Skipped as exc:
            failures += 1
            print(json.dumps({"case": cname, "skipped": str(exc)}), flush=True)
            continue
        summary = case_summary(path, args.root) | {"seconds": round(time.perf_counter() - t0, 2)}
        print(json.dumps(summary), flush=True)
    return 1 if failures == len(args.cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
