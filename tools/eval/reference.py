"""Reference eigenpairs for cases that lack an exact one (data/refs/<case>, docs/FORMATS.md §1).

Kinds written here:
  dense   n <= DENSE_MAX: the whole finite spectrum by LAPACK, with eigenvectors; complete=True,
          window=None. HEP: eigh(A); GHEP: eigh(A, B) (B-orthonormal vectors); GHIEP: QZ eig(A, B)
          (unit 2-norm vectors). A GHIEP eigenvalue with |Im| <= complex_tol * max(|lambda|, scale)
          is a rounding-perturbed real one and is stored with Im = 0; the array becomes complex only
          when genuinely complex eigenvalues exist. Infinite eigenvalues of a singular B (|lambda| >
          INFINITE_FACTOR * scale; for HPD B every |lambda| <= scale * cond(B)) are dropped and counted
          in the note.
  solver  n > DENSE_MAX: ARPACK shift-invert (tools/eval/shift_invert.py) with tol=1e-14 for the
          k + extra eigenvalues nearest the config target sigma, so the k requested ones are interior
          to what was computed. The margin can run into a huge eigenvalue cluster (an EM null space
          right after the k-th value) that ARPACK cannot resolve, so that attempt gets a bounded
          number of restarts and the reference falls back to exactly k pairs. complete=False:
          single-vector ARPACK offers no completeness guarantee and can silently return only some
          copies of a multiple eigenvalue (seen: 9 of 300), which the note flags. window =
          [sigma - r, sigma + r], r the largest distance found. Every pair must pass a backward-error
          check before anything is written.
Replacement policy: exact references (analytic, discrete-exact, or an unlabelled one) are never replaced
unless --force. Continuum-model references (partial-analytic, common.CONTINUUM_KINDS) are not
eigenvalues of the matrix pencil, so a case holding only one "lacks an exact one" and gets a discrete
reference; the replaced files are first archived to refs/<case>/superseded/<kind>/ (load_ref ignores
subdirectories) so the generator's continuum data is kept. A dense reference replaces a solver one,
never the other way round.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la

from tools import case as cio
from tools.eval import common, evaluate, shift_invert

SOLVER_TOL = 1e-14          # ARPACK accuracy for solver references
VERIFY_TOL = 1e-11          # max backward error accepted for a solver reference pair
EXTRA_MAXITER = 100         # ARPACK restarts allowed for the k + extra attempt
INFINITE_FACTOR = 1e12
# Usefulness for judging a solver; kinds not listed (analytic, discrete-exact, unlabelled) are exact.
KIND_RANK = {**{k: -1 for k in common.CONTINUUM_KINDS}, "solver": 0, "dense": 1}
EXACT_RANK = 2
REF_FILES = ("eigenvalues.npy", "eigenvectors.npy", "ref.json")


@dataclass
class RefData:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kind: str
    complete: bool
    window: tuple[float, float] | None
    note: str


def _real_if_possible(w: np.ndarray, V: np.ndarray, real_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero the imaginary part of eigenvalues classified real; drop complex dtypes that carry no
    imaginary information (real pencil, real spectrum -> real eigenvectors)."""
    w = np.where(real_mask, w.real, w)
    if real_mask.all():
        w = w.real
        if np.iscomplexobj(V) and not np.any(V.imag):
            V = V.real
    return w, V


def dense_reference(c: cio.Case, problem: str, *, ctol: float, dense_max: int = common.DENSE_MAX) -> RefData:
    Ad = common.to_dense(c.A, dense_max)
    if problem == "HEP":
        w, V = la.eigh(Ad)
        return RefData(w, V, "dense", True, None, "scipy.linalg.eigh(A), full spectrum")
    Bd = common.to_dense(c.B, dense_max)
    if problem == "GHEP":
        w, V = la.eigh(Ad, Bd)
        return RefData(w, V, "dense", True, None, "scipy.linalg.eigh(A, B), full spectrum, B-orthonormal vectors")
    w, V = la.eig(Ad, Bd)
    scale = evaluate.estimate_norms(c.A, c.B, dense_max).scale
    finite = np.isfinite(w) & (np.abs(w) <= INFINITE_FACTOR * scale)
    w, V = w[finite], V[:, finite]
    V = V / np.linalg.norm(V, axis=0)
    real = common.near_real(w, scale, ctol)
    w, V = _real_if_possible(w, V, real)
    note = (f"scipy.linalg.eig(A, B) (QZ), finite spectrum; {int(np.count_nonzero(~finite))} infinite "
            f"eigenvalue(s) dropped; {int(np.count_nonzero(~real))} complex (complex_tol={ctol:g})")
    return RefData(w, V, "dense", True, None, note)


def solver_reference(c: cio.Case, problem: str, tgt: dict, *, ctol: float, seed: int,
                     extra: int | None = None) -> RefData:
    if tgt["mode"] != "nearest":
        raise ValueError(f"solver references support 'nearest' targets only (got {tgt['mode']!r})")
    sigma, k = float(tgt["sigma"]), int(tgt["k"])
    extra = max(k, 10) if extra is None else extra
    attempts = ([(min(k + extra, c.n - 2), EXTRA_MAXITER)] if extra > 0 else []) + [(k, None)]
    notes = []
    for k_ref, maxiter in attempts:
        run = shift_invert.shift_invert_eigs(c.A, c.B, problem, sigma, k_ref, tol=SOLVER_TOL, maxiter=maxiter,
                                             seed=seed)
        if run.converged:
            break
        notes.append(f"{k_ref} pairs did not converge within {maxiter} restarts; fell back to k={k}")
    else:
        raise RuntimeError(f"ARPACK did not converge ({run.eigenvalues.size} of {k} pairs)")
    norms = evaluate.estimate_norms(c.A, c.B)
    w, V = np.asarray(run.eigenvalues), run.eigenvectors
    w, V = _real_if_possible(w.astype(complex), V, common.near_real(w, norms.scale, ctol)
                             if problem == "GHIEP" else np.ones(w.size, dtype=bool))
    metrics = evaluate.pair_metrics(c.A, c.B, w, V, norms, VERIFY_TOL)
    eta = max(p["backward_error"] for p in metrics)
    if eta > VERIFY_TOL:
        raise RuntimeError(f"solver reference failed verification: max backward error {eta:.2e} > {VERIFY_TOL:g}")
    r = float(np.max(np.abs(w - sigma)))
    multiplicity = int(np.bincount(evaluate.reference_clusters(w, norms.scale, evaluate.CLUSTER_RTOL)).max())
    if multiplicity > 1:
        notes.append(f"contains an eigenvalue of multiplicity >= {multiplicity}: ARPACK may have missed copies")
    note = "; ".join([f"ARPACK shift-invert ({run.method}) sigma={sigma:g}, {k_ref} pairs, tol={SOLVER_TOL:g}, "
                      f"max backward error {eta:.2e} (norms: {norms.method_a}/{norms.method_b})"] + notes)
    return RefData(w, V, "solver", False, (sigma - r, sigma + r), note)


def _archive(ref_dir: Path, kind: str) -> None:
    """Copy the current reference files to ref_dir/superseded/<kind>/ before they are replaced."""
    dest = ref_dir / "superseded" / kind
    dest.mkdir(parents=True, exist_ok=True)
    for name in REF_FILES:
        if (ref_dir / name).exists():
            shutil.copy2(ref_dir / name, dest / name)


def build_reference(name: str, root=cio.DEFAULT_ROOT, *, force: bool = False,
                    dense_max: int = common.DENSE_MAX, extra: int | None = None) -> str:
    """Write the best reference this tool can produce for case `name`; returns a status line."""
    c = cio.load_case(name, root=root)
    problem = common.problem_type(c)
    kind = "dense" if c.n <= dense_max else "solver"
    existing = cio.load_ref(c.name, root=root)
    old = existing.info.get("kind") if existing is not None else None
    if existing is not None and not force and KIND_RANK.get(old, EXACT_RANK) >= KIND_RANK[kind]:
        return f"{c.name}: kept existing {old or 'unlabelled'} reference (use --force to replace)"
    if kind == "dense":
        data = dense_reference(c, problem, ctol=common.complex_tol(c), dense_max=dense_max)
    else:
        data = solver_reference(c, problem, common.target(c), ctol=common.complex_tol(c),
                                seed=common.seed(c), extra=extra)
    if old in common.CONTINUUM_KINDS:   # archive only once the replacement exists
        _archive(Path(root) / "refs" / c.name, old)
    cio.save_ref(c.name, data.eigenvalues, kind=data.kind, note=data.note, complete=data.complete,
                 window=data.window, eigenvectors=data.eigenvectors, root=root)
    return f"{c.name}: wrote {data.kind} reference ({data.eigenvalues.size} eigenvalues) - {data.note}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build reference eigenpairs for cases lacking an exact one.")
    ap.add_argument("cases", nargs="*", help="case names (under ROOT/cases)")
    ap.add_argument("--all", action="store_true", help="every case under ROOT/cases")
    ap.add_argument("--root", default=str(cio.DEFAULT_ROOT), help="data root (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="replace any existing reference, even exact ones")
    ap.add_argument("--dense-max", type=int, default=common.DENSE_MAX, help="largest n for a dense reference")
    ap.add_argument("--extra", type=int, help="solver references: pairs beyond k (default max(k, 10))")
    args = ap.parse_args(argv)
    names = cio.list_cases(args.root) if args.all else args.cases
    if not names:
        ap.error("no cases given (names or --all)")
    common.require_guard("reference")
    failures = 0
    for name in names:
        try:
            print(build_reference(name, args.root, force=args.force, dense_max=args.dense_max, extra=args.extra),
                  flush=True)
        except Exception as err:  # report and continue with the remaining cases
            failures += 1
            print(f"{name}: FAILED {type(err).__name__}: {err}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
