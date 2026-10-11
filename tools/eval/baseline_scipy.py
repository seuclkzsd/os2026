"""scipy/ARPACK baseline solver: writes a result directory (docs/FORMATS.md §3) for one case.

  HEP/GHEP  eigsh(A, k, M=B, sigma=sigma, which='LM')
  GHIEP     eigs on OP = (A - sigma B)^-1 B in the Euclidean inner product (default 'operator'
            variant). The textbook eigs(A, k, M=B, sigma=sigma) silently returns wrong eigenpairs
            for indefinite B (tools/eval/shift_invert.py); it stays available as
            --ghiep-variant mode3 to document that failure.
Only the config 'nearest' target is supported. ARPACK runs with scipy's defaults (tol=0, i.e.
machine precision; ncv=min(n, max(2k+1, 20))) and a deterministic start vector from the config seed.

Output conventions: pairs ordered nearest sigma first; HEP/GHEP vectors B-normalized column by
column (eigsh already returns them B-orthonormal; the complex-Hermitian path goes through eigs),
GHIEP vectors unit 2-norm. GHIEP pairs are split into real and complex ones with the SolverConfig
rule (DESIGN §3.6-10): complex only if |Im lambda| > complex_tol * max(|lambda|, scale) AND its real
projection fails the backward-error test while the complex pair passes. The real projection of
(lambda, x) is (Re lambda, x) for complex matrices; for real matrices x and conj(x) span the real
plane {Re x, Im x} holding the eigenvectors of the real eigenvalue(s) the conjugate pair
approximates, and Rayleigh-Ritz of the pencil on that plane yields them.

report.json uses SolveReport key names where they apply. time_total_s = time_factor_s (splu) +
time_iterate_s (ARPACK), i.e. the scipy solve only; the verification done for this report (norm
estimates, residuals) is timed in time_verify_s and NOT included, since scipy itself does not
verify. peak_rss_bytes is the process high-water mark (ru_maxrss) right after the solve, so it
includes the interpreter and the loaded matrices. Must run under tools/run/guard.py.
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp

from tools import case as cio
from tools.eval import common, evaluate, shift_invert


def _b_normalize(V: np.ndarray, B: sp.spmatrix | None) -> np.ndarray:
    BV = V if B is None else B @ V
    return V / np.sqrt(np.real(np.sum(V.conj() * BV, axis=0)))


def _real_projection(A, B, lam: complex, x: np.ndarray, real_matrices: bool) -> tuple[np.ndarray, np.ndarray]:
    """Real eigenpair candidates (values, n x p vectors) closest to the computed pair (lam, x)."""
    if not real_matrices:
        return np.array([lam.real]), x[:, None]
    Q = la.orth(np.column_stack([x.real, x.imag]))
    mu, Y = la.eig(Q.T @ (A @ Q), Q.T @ (B @ Q))
    if np.all(np.isfinite(mu)) and not np.any(mu.imag):
        Z = Q @ Y.real
        return mu.real, Z / np.linalg.norm(Z, axis=0)
    # The projected pencil keeps a complex pair: fall back to the phase-aligned real part of x.
    p = int(np.argmax(np.abs(x)))
    z = (x * (abs(x[p]) / x[p])).real
    return np.array([lam.real]), (z / np.linalg.norm(z))[:, None]


def split_ghiep(A, B, w: np.ndarray, V: np.ndarray, norms: evaluate.Norms, tol: float, ctol: float,
                real_matrices: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(real values, real vectors, complex values, complex vectors, notes) from eigs output."""
    w = np.asarray(w, dtype=complex)
    V = V / np.linalg.norm(V, axis=0)
    real_w, real_v, cplx_w, cplx_v, notes = [], [], [], [], []
    reps = []   # (lambda, x) to classify; a real pencil's conjugate pair is represented once
    for i, lam in enumerate(w):
        if real_matrices and lam.imag == 0:     # real-arithmetic ARPACK: exactly real pair
            real_w.append(np.array([lam.real]))
            real_v.append(V[:, i:i + 1].real)
        elif real_matrices and lam.imag < 0:
            partner = np.any((w.imag > 0) & (np.abs(w - lam.conjugate()) <= 1e-12 * max(abs(lam), norms.scale)))
            if not partner:                       # ARPACK returned only this half of the pair
                reps.append((lam.conjugate(), V[:, i].conj()))
        else:
            reps.append((lam, V[:, i]))
    for lam, x in reps:
        pr_w, pr_v = _real_projection(A, B, lam, x, real_matrices)
        eta_real = max(p["backward_error"] for p in evaluate.pair_metrics(A, B, pr_w, pr_v, norms, tol))
        eta_cplx = evaluate.pair_metrics(A, B, np.array([lam]), x[:, None], norms, tol)[0]["backward_error"]
        if not common.near_real(lam, norms.scale, ctol) and eta_real > tol and eta_cplx <= tol:
            cplx_w.append(np.array([lam] + ([lam.conjugate()] if real_matrices else [])))
            cplx_v.append(np.column_stack([x] + ([x.conj()] if real_matrices else [])))
        else:
            if lam.imag != 0:
                notes.append(f"lambda={lam:.6g} classified real (|Im| small or real projection passes)")
            real_w.append(pr_w)
            real_v.append(pr_v)
    n = V.shape[0]
    vec_dtype = float if real_matrices else complex
    rw = np.concatenate(real_w) if real_w else np.empty(0)
    rv = np.concatenate(real_v, axis=1).astype(vec_dtype) if real_v else np.empty((n, 0), dtype=vec_dtype)
    cw = np.concatenate(cplx_w) if cplx_w else np.empty(0, dtype=complex)
    cv = np.concatenate(cplx_v, axis=1) if cplx_v else np.empty((n, 0), dtype=complex)
    return rw, rv, cw, cv, notes


def _nearest_first(w: np.ndarray, V: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.abs(w - sigma), kind="stable")
    return w[order], V[:, order]


def solve_case(c: cio.Case, *, arpack_tol: float = 0.0, ncv: int | None = None,
               ghiep_variant: str = "operator") -> dict[str, Any]:
    """Run the baseline on a loaded case. Returns the save_result keyword arguments."""
    problem = common.problem_type(c)
    tgt = common.target(c)
    if tgt["mode"] != "nearest":
        raise ValueError(f"case {c.name}: the scipy baseline supports 'nearest' targets only (got {tgt['mode']!r})")
    sigma, k, tol = float(tgt["sigma"]), int(tgt["k"]), common.tol(c)
    n = c.n
    real_matrices = not common.is_complex(c)
    vec_dtype = float if real_matrices else complex
    warnings: list[str] = []
    rw, rv = np.empty(0), np.empty((n, 0), dtype=vec_dtype)
    cw, cv = np.empty(0, dtype=complex), np.empty((n, 0), dtype=complex)
    run = None
    try:
        run = shift_invert.shift_invert_eigs(c.A, c.B, problem, sigma, k, tol=arpack_tol, ncv=ncv,
                                             seed=common.seed(c), ghiep_variant=ghiep_variant)
    except shift_invert.FactorizationError as err:
        warnings.append(str(err))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # KiB on Linux

    t0 = time.perf_counter()
    norms = evaluate.estimate_norms(c.A, c.B)
    if run is not None:
        if not run.converged:
            warnings.append(f"ARPACK did not converge: {run.eigenvalues.size} of {k} pairs")
        if problem == "GHIEP":
            if ghiep_variant == "mode3":
                warnings.append("mode3: ARPACK generalized mode 3 assumes a B-inner product, invalid for indefinite B")
            rw, rv, cw, cv, notes = split_ghiep(c.A, c.B, run.eigenvalues, run.eigenvectors, norms, tol,
                                                common.complex_tol(c), real_matrices)
            warnings += notes
        else:
            rw = np.asarray(run.eigenvalues).real   # complex Hermitian goes through eigs
            rv = _b_normalize(run.eigenvectors, c.B)
        rw, rv = _nearest_first(rw, rv, sigma)
        cw, cv = _nearest_first(cw, cv, sigma)
    pairs = (evaluate.pair_metrics(c.A, c.B, rw, rv, norms, tol)
             + evaluate.pair_metrics(c.A, c.B, cw, cv, norms, tol))
    time_verify = time.perf_counter() - t0

    report: dict[str, Any] = {
        "solver": "scipy", "scipy_version": scipy.__version__,
        "n": n, "nnz_a": int(c.A.nnz), "nnz_b": int(c.B.nnz) if c.B is not None else 0,
        "complex_scalars": not real_matrices, "problem": common.PROBLEM_ENUM[problem],
        "norm2_a": norms.norm2_a, "norm2_b": norms.norm2_b, "scale": norms.scale,
        "method": run.method if run is not None else "scipy_shift_invert",
        "sigma_requested": sigma, "sigma_used": sigma, "shift_retries": 0,
        "op_applies": run.op_applies if run is not None else 0,
        "time_factor_s": run.time_factor_s if run is not None else 0.0,
        "time_iterate_s": run.time_iterate_s if run is not None else 0.0,
        "time_total_s": (run.time_factor_s + run.time_iterate_s) if run is not None else 0.0,
        "time_verify_s": time_verify,
        "peak_rss_bytes": int(peak_rss),
        "arpack": {"k": k, "ncv": ncv, "tol": arpack_tol, "converged": bool(run is not None and run.converged),
                   "ghiep_variant": ghiep_variant if problem == "GHIEP" else None},
        "pairs": pairs,
        "success": bool(run is not None and run.converged and len(pairs) >= k and all(p["accepted"] for p in pairs)),
        "warnings": warnings,
    }
    return {"eigenvalues": rw, "eigenvectors": rv, "report": report,
            "complex_eigenvalues": cw if cw.size else None, "complex_eigenvectors": cv if cw.size else None}


def run_baseline(case_ref: str | Path, out_dir: str | Path, *, root=cio.DEFAULT_ROOT, arpack_tol: float = 0.0,
                 ncv: int | None = None, ghiep_variant: str = "operator") -> Path:
    c = cio.load_case(case_ref, root=root)
    return cio.save_result(out_dir, **solve_case(c, arpack_tol=arpack_tol, ncv=ncv, ghiep_variant=ghiep_variant))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="scipy/ARPACK shift-invert baseline for one case.")
    ap.add_argument("case", help="case name (under ROOT/cases) or case directory")
    ap.add_argument("--out", type=Path, required=True, help="result directory to write")
    ap.add_argument("--root", default=str(cio.DEFAULT_ROOT), help="data root (default: %(default)s)")
    ap.add_argument("--arpack-tol", type=float, default=0.0, help="ARPACK tol (0 = machine precision)")
    ap.add_argument("--ncv", type=int, help="ARPACK ncv (default: scipy's)")
    ap.add_argument("--ghiep-variant", choices=shift_invert.GHIEP_VARIANTS, default="operator")
    args = ap.parse_args(argv)
    common.require_guard("baseline_scipy")
    out = run_baseline(args.case, args.out, root=args.root, arpack_tol=args.arpack_tol, ncv=args.ncv,
                       ghiep_variant=args.ghiep_variant)
    rep = cio.load_result(out).report
    print(f"{args.case}: success={rep['success']} pairs={len(rep['pairs'])} time={rep['time_total_s']:.3g}s "
          f"peak_rss={rep['peak_rss_bytes'] / 2**20:.0f}MiB -> {out}")
    for w in rep["warnings"]:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
