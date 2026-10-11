"""Shift-invert eigensolves with scipy/ARPACK, shared by the scipy baseline and solver-based references.

K = A - sigma B is factorized once with SuperLU (scipy.sparse.linalg.splu with scipy's default COLAMD
ordering - the same factorization eigsh/eigs build internally when sigma is given). Passing it in
as OPinv lets us time the factorization separately and count operator applications.

Variants (theta = 1/(lambda - sigma), DESIGN §4.1):
  HEP/GHEP          eigsh(A, k, M=B, sigma, which='LM', OPinv=K^-1): ARPACK mode 3 with the
                    B-inner product, valid because B is HPD; returns lambda directly.
  GHIEP 'operator'  eigs(OP, k, which='LM') with OP = K^-1 B in the Euclidean inner product, then
                    lambda = sigma + 1/theta. Correct for any nonsingular K.
  GHIEP 'mode3'     the textbook call eigs(A, k, M=B, sigma, OPinv=K^-1). ARPACK's generalized
                    mode 3 orthogonalizes with sqrt|x^T B x|, which is not an inner product when B
                    is indefinite: on a 400-order real-spectrum pencil it returned eigenvalue errors
                    ~1e-1 and residuals ~0.7 with no warning. Kept only to demonstrate (and let the
                    evaluator detect) this failure of the standard tool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from tools.eval import common

GHIEP_VARIANTS = ("operator", "mode3")


class FactorizationError(RuntimeError):
    """K = A - sigma B could not be factorized (exactly singular, or out of memory)."""


@dataclass
class ArpackRun:
    eigenvalues: np.ndarray      # lambda (complex128 for eigs-based runs), ARPACK order
    eigenvectors: np.ndarray     # n x m, columns match eigenvalues
    converged: bool              # False: partial results from ArpackNoConvergence
    method: str                  # SolveReport::method string for the result report
    k: int
    ncv: int | None
    tol: float
    op_applies: int
    time_factor_s: float
    time_iterate_s: float


def shift_invert_eigs(A: sp.csr_matrix, B: sp.csr_matrix | None, problem: str, sigma: float, k: int, *,
                      tol: float = 0.0, ncv: int | None = None, maxiter: int | None = None,
                      seed: int = common.DEFAULT_SEED, ghiep_variant: str = "operator") -> ArpackRun:
    """The k eigenpairs of A x = lambda B x nearest sigma (largest |theta|) via ARPACK.

    tol is ARPACK's relative Ritz-value accuracy (0 = machine precision, scipy's default); maxiter
    bounds the implicit restarts (None = scipy's n*10). Raises FactorizationError when K cannot be
    factorized and ValueError for unsupported arguments; non-convergence returns the converged
    subset with converged=False."""
    n = A.shape[0]
    if problem not in common.PROBLEM_ENUM:
        raise ValueError(f"unknown problem {problem!r}")
    if ghiep_variant not in GHIEP_VARIANTS:
        raise ValueError(f"ghiep_variant must be one of {GHIEP_VARIANTS}")
    # ARPACK needs k < n (symmetric driver) and k < n - 1 (nonsymmetric driver).
    if not 1 <= k < n - 1:
        raise ValueError(f"ARPACK needs 1 <= k < n - 1 (k={k}, n={n}); use a dense solve instead")
    dtype = np.result_type(A.dtype, B.dtype if B is not None else A.dtype, np.float64)

    t0 = time.perf_counter()
    eye = sp.identity(n, dtype=dtype, format="csr")
    K = (A - sigma * (B if B is not None else eye)).tocsc()
    try:
        lu = sla.splu(K)
    except (RuntimeError, MemoryError) as err:   # SuperLU: "Factor is exactly singular"
        raise FactorizationError(f"splu(A - sigma B) failed at sigma={sigma:g}: {err}") from err
    time_factor = time.perf_counter() - t0

    applies = 0

    def solve(x: np.ndarray) -> np.ndarray:
        nonlocal applies
        applies += 1
        return lu.solve(np.asarray(x, dtype=dtype))

    def op(x: np.ndarray) -> np.ndarray:
        return solve(B @ x if B is not None else x)

    v0 = common.start_vector(n, dtype, seed)
    opinv = sla.LinearOperator((n, n), matvec=solve, dtype=dtype)
    explicit_op = problem == "GHIEP" and ghiep_variant == "operator"
    method = ("scipy_eigsh_shift_invert" if problem != "GHIEP" else
              "scipy_eigs_shift_invert_operator" if explicit_op else "scipy_eigs_mode3")
    t1 = time.perf_counter()
    try:
        if problem != "GHIEP":
            w, v = sla.eigsh(A, k=k, M=B, sigma=sigma, which="LM", OPinv=opinv, v0=v0, ncv=ncv, tol=tol,
                             maxiter=maxiter)
        elif explicit_op:
            w, v = sla.eigs(sla.LinearOperator((n, n), matvec=op, dtype=dtype), k=k, which="LM", v0=v0,
                            ncv=ncv, tol=tol, maxiter=maxiter)
        else:
            w, v = sla.eigs(A, k=k, M=B, sigma=sigma, which="LM", OPinv=opinv, v0=v0, ncv=ncv, tol=tol,
                            maxiter=maxiter)
        converged = True
    except sla.ArpackNoConvergence as err:
        w, v = err.eigenvalues, err.eigenvectors
        converged = False
    time_iterate = time.perf_counter() - t1
    w = np.asarray(w)
    if explicit_op:   # ARPACK returned theta = 1/(lambda - sigma)
        w = sigma + 1.0 / w.astype(complex)
    return ArpackRun(eigenvalues=w, eigenvectors=np.asarray(v).reshape(n, w.size), converged=converged,
                     method=method, k=k, ncv=ncv, tol=tol, op_applies=applies, time_factor_s=time_factor,
                     time_iterate_s=time_iterate)
