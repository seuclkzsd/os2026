"""hsolve - sparse Hermitian (indefinite) eigensolver, in-process Python API.

    import hsolve
    res = hsolve.solve(A, B, sigma=-2.5e5, k=10)      # A, B: scipy.sparse or numpy arrays
    res.eigenvalues, res.eigenvectors, res.report["success"]

A and B may be any scipy.sparse matrix/array (csr, csc and coo are converted without
intermediate copies; other formats via .tocsr()), dense numpy arrays, or paths to .npz/.npy/.mtx
files (loaded exactly as the hsolve command line does). See docs/PYTHON.md.
"""
from __future__ import annotations

import json
import numbers
import operator
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _hsolve

__all__ = [
    "solve",
    "load_matrix",
    "default_config",
    "version",
    "build_info",
    "SolveResult",
    "HsolveError",
    "InvalidInputError",
    "HsolveIOError",
    "NotHermitianError",
    "UnsupportedError",
    "FactorizationFailedError",
    "SingularError",
    "OutOfMemoryBudgetError",
    "NotConvergedError",
    "InternalError",
    "HsolveWarning",
]

# --------------------------------------------------------------------------------- exceptions


class HsolveError(Exception):
    """Base class of every hsolve error. ``code`` is the hs::Error code name (snake_case)."""

    code = "unknown"

    def __init__(self, message: str = "", code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class InvalidInputError(HsolveError, ValueError):
    """Malformed or unsupported input: bad shapes, dtypes, configuration values, k, ..."""

    code = "invalid_input"


class HsolveIOError(HsolveError, OSError):
    """File system or file format error (matrix files, config.json)."""

    code = "io"


class NotHermitianError(HsolveError, ValueError):
    """A (or B) fails the Hermitian check: ||M - M^H||_F / ||M||_F > hermitian_tol."""

    code = "not_hermitian"


class UnsupportedError(HsolveError, NotImplementedError):
    """A valid request this build cannot serve yet (e.g. interval targets with indefinite B)."""

    code = "unsupported"


class FactorizationFailedError(HsolveError, RuntimeError):
    """The sparse LDL^T factorization (MUMPS) failed."""

    code = "factorization_failed"


class SingularError(HsolveError, ArithmeticError):
    """A - sigma B is numerically singular at and near the requested shift."""

    code = "singular"


class OutOfMemoryBudgetError(HsolveError, MemoryError):
    """The predicted memory exceeds the memory budget (memory_budget_gb, or the automatic budget
    derived from the memory limits of the process; nothing was allocated)."""

    code = "out_of_memory_budget"


class NotConvergedError(HsolveError, RuntimeError):
    code = "not_converged"


class InternalError(HsolveError, RuntimeError):
    """A broken internal invariant (a bug - please report it)."""

    code = "internal"


class HsolveWarning(UserWarning):
    """Issued when a solve finishes without meeting the tolerance (report['success'] is False)."""


_ERROR_CLASSES = (
    InvalidInputError,
    HsolveIOError,
    NotHermitianError,
    UnsupportedError,
    FactorizationFailedError,
    SingularError,
    OutOfMemoryBudgetError,
    NotConvergedError,
    InternalError,
)
_hsolve._error_classes.update({cls.code: cls for cls in _ERROR_CLASSES})

# ----------------------------------------------------------------------------------- results


@dataclass(repr=False, eq=False)
class SolveResult:
    """Result of :func:`solve`.

    eigenvalues          float64 (m,): real eigenvalues ordered by the target (nearest: by
                         distance to sigma; interval/smallest: ascending; largest: descending).
    eigenvectors         (n, m), Fortran order; complex128 when A or B has a complex dtype (the
                         array's, or the value type stored in the file), else float64.
                         B-orthonormal for HEP/GHEP, unit 2-norm for indefinite B.
    complex_eigenvalues  complex128 (c,): genuinely complex eigenvalues (indefinite B only).
    complex_eigenvectors complex128 (n, c).
    report               dict, the SolveReport (same keys as report.json of the CLI).
    config               dict, the complete effective configuration (a valid config.json).
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    complex_eigenvalues: np.ndarray
    complex_eigenvectors: np.ndarray
    report: dict
    config: dict

    @property
    def success(self) -> bool:
        """True when every pair met the tolerance (and the inertia certificate passed)."""
        return bool(self.report.get("success", False))

    def __repr__(self) -> str:
        n = self.eigenvectors.shape[0]
        return (
            f"SolveResult(n={n}, eigenvalues={self.eigenvalues.size}, "
            f"complex_eigenvalues={self.complex_eigenvalues.size}, "
            f"problem={self.report.get('problem')!r}, success={self.success})"
        )


# ----------------------------------------------------------------------------------- helpers


class _Default:
    """Marks a keyword argument the caller did not pass (so config values are not overridden)."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


_MODES = ("nearest", "interval", "smallest", "largest")
_STORAGES = ("auto", "full", "upper", "lower")
_VALUE_DTYPES = tuple(np.dtype(t) for t in (np.float32, np.float64, np.complex64, np.complex128))


def _real(x: Any, name: str) -> float:
    """A real number: int, float, numpy integer/floating scalar or 0-d array. Strings, bools and
    complex values are rejected (float() would parse '0.5'; k and threads reject '2' as well)."""
    if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype.kind in "iuf":
        return float(x)
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}")
    return float(x)


def _integer(x: Any, name: str) -> int:
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{name} must be an integer, got {type(x).__name__}") from e


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic) or (isinstance(o, np.ndarray) and o.ndim == 0):
        return o.item()
    raise TypeError(f"config value of type {type(o).__name__} is not JSON serializable")


def _read_config(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return json.loads(json.dumps(config, default=_json_default))  # deep copy, plain types
    if isinstance(config, (str, bytes, os.PathLike)):
        path = os.fsdecode(config)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise HsolveIOError(f"cannot read config file {path}: {e}") from e
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise InvalidInputError(f"config file {path}: invalid JSON: {e}") from None
        if not isinstance(doc, dict):
            raise InvalidInputError(f"config file {path}: top level must be a JSON object")
        return doc
    raise TypeError(f"config must be a dict, a path to config.json or None, got {type(config).__name__}")


def _merge_config(config, sigma, k, mode, lower, upper, tol, threads) -> str:
    cfg = _read_config(config)
    target = cfg.get("target", {})
    if not isinstance(target, dict):
        raise InvalidInputError("config: 'target' must be an object")
    target = dict(target)
    explicit_mode = not isinstance(mode, _Default)
    interval_bounds = lower is not None or upper is not None
    if explicit_mode:
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a string, got {type(mode).__name__}")
        if mode not in _MODES:
            raise InvalidInputError(f"mode = {mode!r} is not one of {_MODES}")
        target["mode"] = mode
    elif interval_bounds:
        if sigma is not None:
            raise InvalidInputError("give sigma (nearest) or lower/upper (interval), not both, or pass mode=")
        target["mode"] = "interval"
    elif sigma is not None:
        target["mode"] = "nearest"
    if sigma is not None:
        target["sigma"] = _real(sigma, "sigma")
    if lower is not None:
        target["lower"] = _real(lower, "lower")
    if upper is not None:
        target["upper"] = _real(upper, "upper")
    if not isinstance(k, _Default):
        target["k"] = _integer(k, "k")
    if target:
        cfg["target"] = target
    if not isinstance(tol, _Default):
        cfg["tol"] = _real(tol, "tol")
    if not isinstance(threads, _Default):
        cfg["threads"] = _integer(threads, "threads")
    try:
        return json.dumps(cfg, default=_json_default, allow_nan=False)
    except ValueError as e:
        raise InvalidInputError(f"config: {e} (JSON has no inf/nan)") from None


def _values(arr: Any, name: str) -> np.ndarray:
    """Numeric array with a dtype the C++ side reads directly (float32/64, complex64/128)."""
    arr = np.asarray(arr)
    dt = arr.dtype
    if dt in _VALUE_DTYPES and dt.isnative:
        return arr
    if dt.kind == "c":
        target = np.complex64 if dt.itemsize == 8 else np.complex128
    elif dt.kind == "f":
        target = np.float32 if dt.itemsize == 4 else np.float64
    elif dt.kind in "biu":
        target = np.float64
    else:
        raise InvalidInputError(f"{name}: unsupported dtype {dt}")
    return arr.astype(target)


def _indices(*arrays: Any) -> list[np.ndarray]:
    """Contiguous 1-D index arrays sharing one native dtype, int32 or int64 (no copy if so)."""
    out = [np.asarray(a) for a in arrays]
    dts = {a.dtype for a in out}
    ok = len(dts) == 1 and next(iter(dts)) in (np.dtype(np.int32), np.dtype(np.int64))
    if not ok:
        out = [a.astype(np.int64) for a in out]
    return [np.ascontiguousarray(a) for a in out]


def _is_sparse(m: Any) -> bool:
    sp = sys.modules.get("scipy.sparse")  # a scipy.sparse object implies the module is loaded
    return sp is not None and sp.issparse(m)


def _matrix_arg(m: Any, name: str) -> tuple:
    """Describe a matrix for the C++ side without copying it (see module.cpp: matrix_arg)."""
    if isinstance(m, (str, bytes, os.PathLike)):
        return ("path", os.fsdecode(m))
    if _is_sparse(m):
        shape = tuple(int(s) for s in m.shape)
        if len(shape) != 2:
            raise InvalidInputError(f"{name}: expected a 2-D sparse matrix, got shape {shape}")
        fmt = m.format
        if fmt not in ("csr", "csc", "coo"):
            m = m.tocsr()  # bsr, dia, lil, dok: one conversion by scipy
            fmt = "csr"
        data = np.ascontiguousarray(_values(m.data, f"{name}.data"))
        if fmt == "coo":
            row, col = _indices(m.row, m.col)
            arg = ("coo", shape, row, col, data)
        else:
            indptr, indices = _indices(m.indptr, m.indices)
            arg = (fmt, shape, indptr, indices, data)
        return arg
    if hasattr(m, "matvec") and not isinstance(m, np.ndarray):
        raise TypeError(f"{name}: hsolve factorizes the matrix and needs its entries, not a LinearOperator")
    arr = np.asarray(m)
    if arr.dtype == object:
        raise InvalidInputError(f"{name}: cannot interpret {type(m).__name__} as a numeric matrix")
    if arr.ndim != 2:
        raise InvalidInputError(f"{name}: expected a 2-D array, got {arr.ndim}-D")
    arr = _values(arr, name)
    return ("dense", tuple(int(s) for s in arr.shape), arr)


# --------------------------------------------------------------------------------------- API


def solve(
    A: Any,
    B: Any = None,
    *,
    sigma: float | None = None,
    k: int = _Default(6),
    mode: str = _Default("nearest"),
    lower: float | None = None,
    upper: float | None = None,
    tol: float = _Default(1e-10),
    config: dict | str | os.PathLike | None = None,
    threads: int = _Default(0),
    verbose: int = 0,
    storage: str = "auto",
) -> SolveResult:
    """Solve A x = lambda B x (B=None: A x = lambda x) for Hermitian A, B; B may be indefinite.

    A, B     scipy.sparse matrices/arrays (csr/csc/coo direct; other formats via .tocsr()),
             dense numpy arrays (exact zeros are absent entries), or paths to .npz/.npy/.mtx.
             Values float32/64 or complex64/128 (others are cast); indices int32 or int64.
             Complex A with real B (or vice versa) solves the complex problem.
    sigma    shift for mode='nearest' (the k eigenvalues closest to sigma).
    k        number of eigenpairs (nearest, smallest, largest).
    mode     'nearest' | 'interval' (all eigenvalues in [lower, upper]) | 'smallest' | 'largest'.
             Inferred when omitted: lower/upper -> 'interval', sigma -> 'nearest'.
    tol      acceptance threshold on the backward error of every pair.
    config   dict or path to config.json (docs/FORMATS.md section 2); keyword arguments that are
             passed explicitly override it. Memory: config["memory_budget_gb"] is "auto" (the
             default, also 0) - a share of the smallest memory limit of the process: memory
             cgroup, the HSOLVE_MEMORY_LIMIT_GB environment variable (GiB), RLIMIT_AS/RLIMIT_DATA,
             MemTotal; under RLIMIT_AS it is planned in address space, which also counts what the
             process reserves without using it (BLAS buffers, malloc arenas) - or a number of GiB
             (resident memory); factors beyond it go out of core (result.report["memory"] says
             which budget, in which measure and why). config["factor_precision"]: "double"
             (default), "single" (half the factor memory, slower; checked, falls back to double
             rather than return a wrong answer) or "auto" (single only when the double factors do
             not fit in core but single ones do; result.report["precision"]).
    threads  OpenMP threads (0 = OMP_NUM_THREADS / all available).
    verbose  0 quiet, 1 progress events, 2 debug (written to the process's stderr).
    storage  'auto' (a matrix with one empty strict triangle is mirrored, as the CLI does),
             'full', 'upper' or 'lower' (only that triangle is stored).

    The GIL is released during conversion and solve. Raises an HsolveError subclass on failure;
    a run that finishes without meeting tol returns with result.success False and issues an
    HsolveWarning.
    """
    if storage not in _STORAGES:
        raise InvalidInputError(f"storage = {storage!r} is not one of {_STORAGES}")
    config_json = _merge_config(config, sigma, k, mode, lower, upper, tol, threads)
    verbose = _integer(verbose, "verbose")
    a_arg = _matrix_arg(A, "A")
    b_arg = None if B is None else _matrix_arg(B, "B")
    w, v, cw, cv, report_json, config_out, complex_dtype = _hsolve.solve(a_arg, b_arg, config_json, verbose,
                                                                         storage)
    if complex_dtype and v.dtype != np.complex128:
        # A complex dtype (array or file) whose imaginary parts are all zero is solved as real;
        # the eigenvectors still come back complex, whichever way the matrix was passed.
        v = v.astype(np.complex128, order="F")
    result = SolveResult(w, v, cw, cv, json.loads(report_json), json.loads(config_out))
    if not result.success:
        warnings.warn(
            "hsolve: the solve finished without meeting the tolerance (see result.report['pairs'] "
            "and result.report['warnings'])",
            HsolveWarning,
            stacklevel=2,
        )
    return result


def load_matrix(path: str | os.PathLike, *, storage: str = "auto", hermitian_tol: float = 1e-10,
                return_info: bool = False):
    """Load a matrix file exactly as the hsolve command line does (.npz scipy sparse, .npy dense,
    .mtx MatrixMarket): duplicates summed, exact zeros dropped, one stored triangle mirrored
    (storage), Hermitian check (NotHermitianError above hermitian_tol; below it A is replaced by
    its Hermitian part), complex input with zero imaginary parts returned as real.

    Returns a scipy.sparse.csr_array whose data and indices are the loaded C++ buffers (no copy;
    indptr is a fresh int32 array of n + 1 entries, since scipy wants one index dtype - int64
    indptr would make it copy all indices), and with return_info=True also a dict
    {source_format, storage, hermitian_defect, notes}. Only when n or nnz exceeds the int32 range
    are the indices an int64 copy (indptr then the C++ buffer).
    """
    import scipy.sparse as sp

    if storage not in _STORAGES:
        raise InvalidInputError(f"storage = {storage!r} is not one of {_STORAGES}")
    n, indptr, indices, data, info = _hsolve.load_matrix(os.fsdecode(path), storage,
                                                         _real(hermitian_tol, "hermitian_tol"))
    mat = sp.csr_array((data, indices, indptr), shape=(n, n))  # same index dtypes: kept as they are
    mat.has_canonical_format = True  # rows sorted, no duplicates (hs::CsrMatrix invariant)
    return (mat, info) if return_info else mat


def default_config() -> dict:
    """The complete default configuration (every key of docs/FORMATS.md section 2)."""
    return json.loads(_hsolve.parse_config("{}"))


def version() -> str:
    """hsolve library version, e.g. '0.1.0'."""
    return _hsolve.version()


def build_info() -> str:
    """Compiler, flags and OpenMP version the extension was built with."""
    return _hsolve.build_info()


__version__ = version()
