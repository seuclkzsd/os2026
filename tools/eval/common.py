"""Helpers shared by the evaluation tools (reference, baseline, evaluator).

Case interpretation follows docs/FORMATS.md: the problem type comes from meta.json (the source of
truth written by the generator), the query and tolerances from config.json with SolverConfig
defaults (include/hsolve/solver.hpp) for omitted keys.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from tools import case as cio

# DESIGN §5: test/evaluation code may densify matrices up to this order (never the solver).
DENSE_MAX = 4000
# SolverConfig defaults (include/hsolve/solver.hpp).
DEFAULT_TOL = 1e-10
DEFAULT_COMPLEX_TOL = 1e-8
DEFAULT_SEED = 2026
DEFAULT_TARGET: dict[str, Any] = {"mode": "nearest", "sigma": 0.0, "k": 6, "lower": 0.0, "upper": 0.0}
# meta.json problem names -> SolveReport::problem enum strings (snake_case, docs/FORMATS.md §2).
PROBLEM_ENUM = {"HEP": "standard", "GHEP": "generalized_definite", "GHIEP": "generalized_indefinite"}


def problem_type(c: cio.Case) -> str:
    """'HEP' | 'GHEP' | 'GHIEP' from meta.json; B must be present exactly for the generalized ones."""
    problem = c.meta.get("problem", "HEP" if c.B is None else "GHEP")
    if problem not in cio.PROBLEMS:
        raise ValueError(f"case {c.name}: unknown problem {problem!r}")
    if (problem == "HEP") != (c.B is None):
        raise ValueError(f"case {c.name}: problem {problem} but B is {'absent' if c.B is None else 'present'}")
    return problem


def target(c: cio.Case) -> dict[str, Any]:
    """The config.json target with SolverConfig defaults filled in, validated."""
    t = dict(DEFAULT_TARGET)
    t.update(c.config.get("target", {}))
    if t["mode"] not in cio.TARGET_MODES:
        raise ValueError(f"case {c.name}: target mode {t['mode']!r} not in {cio.TARGET_MODES}")
    if t["mode"] == "interval":
        if not float(t["lower"]) <= float(t["upper"]):
            raise ValueError(f"case {c.name}: interval target needs lower <= upper")
    elif int(t["k"]) < 1:
        raise ValueError(f"case {c.name}: target k must be >= 1")
    return t


def tol(c: cio.Case) -> float:
    return float(c.config.get("tol", DEFAULT_TOL))


def complex_tol(c: cio.Case) -> float:
    return float(c.config.get("complex_tol", DEFAULT_COMPLEX_TOL))


def seed(c: cio.Case) -> int:
    return int(c.config.get("seed", DEFAULT_SEED))


def is_complex(c: cio.Case) -> bool:
    return np.iscomplexobj(c.A.data) or (c.B is not None and np.iscomplexobj(c.B.data))


def start_vector(n: int, dtype: np.dtype, seed_value: int) -> np.ndarray:
    """Deterministic ARPACK start vector (ARPACK's own default v0 is random and irreproducible)."""
    rng = np.random.default_rng(seed_value)
    v = rng.uniform(-1.0, 1.0, n)
    if np.issubdtype(dtype, np.complexfloating):
        v = v + 1j * rng.uniform(-1.0, 1.0, n)
    return v.astype(dtype)


def near_real(lam: np.ndarray, scale: float, ctol: float) -> np.ndarray:
    """|Im lambda| <= complex_tol * max(|lambda|, scale): the magnitude half of the GHIEP real/complex
    rule (DESIGN §3.6-10). Such values are real eigenvalues perturbed by rounding."""
    lam = np.asarray(lam, dtype=complex)
    return np.abs(lam.imag) <= ctol * np.maximum(np.abs(lam), scale)


def to_dense(m, dense_max: int = DENSE_MAX) -> np.ndarray:
    """Dense copy of a sparse matrix, refused above dense_max (DESIGN §5 red line)."""
    if m.shape[0] > dense_max:
        raise ValueError(f"refusing to densify a matrix of order {m.shape[0]} > {dense_max}")
    return m.toarray()


def require_guard(prog: str) -> None:
    """CLIs that can run heavy numerics must be launched through tools/run/guard.py (DESIGN §6)."""
    if os.environ.get("HS_GUARDED") != "1":
        raise SystemExit(f"{prog}: run under tools/run/guard.py (docs/DESIGN.md §6); HS_GUARDED is not set")
