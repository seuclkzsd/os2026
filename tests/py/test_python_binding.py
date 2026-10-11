"""Tests of the in-process Python API (python/hsolve, docs/PYTHON.md).

The extension is imported from the build tree: $HSOLVE_PYTHON_DIR, default build/release/python
(built by tools/dev/build.sh release when pybind11 is available). Small problems only.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

_ROOT = Path(__file__).resolve().parents[2]
_PY_DIR = Path(os.environ.get("HSOLVE_PYTHON_DIR", _ROOT / "build" / "release" / "python"))
if not (_PY_DIR / "hsolve" / "__init__.py").is_file():
    pytest.skip(f"hsolve Python package not built in {_PY_DIR} (tools/dev/build.sh release)",
                allow_module_level=True)
sys.path.insert(0, str(_PY_DIR))
import hsolve  # noqa: E402
from hsolve import _hsolve  # noqa: E402

DETERMINISTIC = {"deterministic": True, "threads": 2}


# ------------------------------------------------------------------------------------ fixtures


def lap1d(n: int) -> sp.csr_matrix:
    e = np.ones(n)
    return sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csr")


def lap1d_eigs(n: int) -> np.ndarray:
    return 2.0 - 2.0 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1))


def lap2d(nx: int, ny: int, shift: float = 0.0) -> sp.csr_matrix:
    a = sp.kronsum(lap1d(nx), lap1d(ny), format="csr")
    return (a - shift * sp.identity(nx * ny, format="csr")).tocsr()


def lap2d_eigs(nx: int, ny: int, shift: float = 0.0) -> np.ndarray:
    return np.sort(np.add.outer(lap1d_eigs(nx), lap1d_eigs(ny)).ravel() - shift)


def mass1d(n: int) -> sp.csr_matrix:
    e = np.ones(n)
    return sp.diags([e[:-1], 4 * e, e[:-1]], [-1, 0, 1], format="csr") / 6.0


def random_hermitian(n: int, density: float, seed: int, complex_: bool = True) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    r = sp.random(n, n, density=density, random_state=rng, format="csr")
    if complex_:
        r = r + 1j * sp.random(n, n, density=density, random_state=rng, format="csr")
    a = r + r.conj().T + sp.diags(rng.standard_normal(n))
    return a.tocsr()


def nearest(values: np.ndarray, sigma: complex, k: int) -> np.ndarray:
    values = np.asarray(values)
    return values[np.argsort(np.abs(values - sigma), kind="stable")[:k]]


def assert_same_set(got, want, rtol: float, scale: float = 1.0):
    got = np.sort_complex(np.asarray(got, dtype=complex))
    want = np.sort_complex(np.asarray(want, dtype=complex))
    assert got.shape == want.shape, (got, want)
    np.testing.assert_allclose(got, want, rtol=0, atol=rtol * scale)


def residuals(a, b, w, v) -> np.ndarray:
    bv = v if b is None else b @ v
    r = a @ v - bv * w[np.newaxis, :]
    return np.linalg.norm(r, axis=0) / np.maximum(np.abs(w), 1.0)


def solve_quiet(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", hsolve.HsolveWarning)
        return hsolve.solve(*args, **kw)


# --------------------------------------------------------------------------- agreement: HEP/GHEP


def test_hep_real_matches_eigsh():
    a = lap2d(30, 25, shift=4.0)  # indefinite, simple spectrum (nx != ny)
    sigma, k = 0.3, 6
    res = solve_quiet(a, sigma=sigma, k=k)
    assert res.success and res.report["problem"] == "standard"
    w_ref, v_ref = sla.eigsh(a.tocsc(), k=k, sigma=sigma)
    assert_same_set(res.eigenvalues, w_ref, 1e-10)
    assert_same_set(res.eigenvalues, nearest(lap2d_eigs(30, 25, 4.0), sigma, k), 1e-10)
    # nearest first, Fortran-ordered float64 eigenvectors, orthonormal, small residuals
    d = np.abs(res.eigenvalues - sigma)
    assert np.all(np.diff(d) >= -1e-12)
    v = res.eigenvectors
    assert v.dtype == np.float64 and v.shape == (a.shape[0], k) and v.flags.f_contiguous
    np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-10)
    assert residuals(a, None, res.eigenvalues, v).max() < 1e-8
    # same subspace as eigsh (simple eigenvalues: |cos| = 1 per vector)
    order = np.argsort(res.eigenvalues)
    cos = np.abs(np.sum(v[:, order] * v_ref[:, np.argsort(w_ref)], axis=0))
    np.testing.assert_allclose(cos, 1.0, atol=1e-8)


def test_ghep_real_matches_eigh():
    n = 200
    a, b = lap1d(n) * (n + 1) ** 2, mass1d(n)
    sigma, k = 3000.0, 5
    res = solve_quiet(a, b, sigma=sigma, k=k)
    assert res.report["problem"] == "generalized_definite"
    w_all = la.eigh(a.toarray(), b.toarray(), eigvals_only=True)
    assert_same_set(res.eigenvalues, nearest(w_all, sigma, k), 1e-10, scale=sigma)
    v = res.eigenvectors
    np.testing.assert_allclose(v.T @ (b @ v), np.eye(k), atol=1e-10)  # B-orthonormal
    assert residuals(a, b, res.eigenvalues, v).max() < 1e-7


def test_hep_complex_matches_eigh():
    a = random_hermitian(200, 0.03, seed=11)
    sigma, k = 0.1, 6
    res = solve_quiet(a, sigma=sigma, k=k)
    assert res.report["complex_scalars"] is True
    w_all = la.eigh(a.toarray(), eigvals_only=True)
    assert_same_set(res.eigenvalues, nearest(w_all, sigma, k), 1e-10)
    v = res.eigenvectors
    assert v.dtype == np.complex128 and v.flags.f_contiguous
    np.testing.assert_allclose(v.conj().T @ v, np.eye(k), atol=1e-10)
    assert residuals(a, None, res.eigenvalues, v).max() < 1e-8


def test_ghep_complex_a_real_b_promotes_b():
    n = 150
    a = random_hermitian(n, 0.04, seed=5) + 4.0 * sp.identity(n)
    b = mass1d(n)  # real B, complex A: B is promoted inside the conversion
    sigma, k = 2.0, 4
    res = solve_quiet(a, b, sigma=sigma, k=k)
    assert res.report["problem"] == "generalized_definite" and res.report["complex_scalars"]
    w_all = la.eigh(a.toarray(), b.toarray(), eigvals_only=True)
    assert_same_set(res.eigenvalues, nearest(w_all, sigma, k), 1e-9)
    v = res.eigenvectors
    np.testing.assert_allclose(v.conj().T @ (b @ v), np.eye(k), atol=1e-10)


def test_smallest_largest_and_interval_modes():
    n = 300
    a, exact = lap1d(n), lap1d_eigs(n)
    res = solve_quiet(a, mode="smallest", k=4)
    np.testing.assert_allclose(res.eigenvalues, exact[:4], atol=1e-11)  # ascending
    res = solve_quiet(a, mode="largest", k=3)
    np.testing.assert_allclose(res.eigenvalues, exact[::-1][:3], atol=1e-11)  # descending
    lo, hi = 0.5, 0.62
    res = solve_quiet(a, lower=lo, upper=hi)  # mode inferred: interval
    assert res.config["target"]["mode"] == "interval"
    np.testing.assert_allclose(res.eigenvalues, exact[(exact >= lo) & (exact <= hi)], atol=1e-11)
    assert res.report["certificate"]["passed"]


# ------------------------------------------------------------------------------------ GHIEP


def ghiep_pencil(n: int = 60, seed: int = 7):
    """A = Y^T C Y, B = Y^T J Y (B indefinite): real eigenvalues lam_i plus one complex pair
    0.31 +- 0.05i from the block C = [[a, b], [b, -a]], J = [[0, 1], [1, 0]]."""
    rng = np.random.default_rng(seed)
    y = np.eye(n) + 0.3 * rng.standard_normal((n, n)) / np.sqrt(n)
    lam = np.linspace(-5.0, 5.0, n - 2)
    sgn = np.where(np.arange(n - 2) % 3 == 0, -1.0, 1.0)
    c, j = np.zeros((n, n)), np.zeros((n, n))
    c[: n - 2, : n - 2] = np.diag(lam * sgn)
    j[: n - 2, : n - 2] = np.diag(sgn)
    c[n - 2 :, n - 2 :] = [[0.05, 0.31], [0.31, -0.05]]
    j[n - 2 :, n - 2 :] = [[0.0, 1.0], [1.0, 0.0]]
    a, b = y.T @ c @ y, y.T @ j @ y
    a, b = np.triu(a) + np.triu(a, 1).T, np.triu(b) + np.triu(b, 1).T  # exactly symmetric
    return sp.csr_matrix(a), sp.csr_matrix(b)


def test_ghiep_matches_eig_with_complex_pair():
    a, b = ghiep_pencil()
    sigma, k = 0.3, 6
    res = solve_quiet(a, b, sigma=sigma, k=k)
    assert res.report["problem"] == "generalized_indefinite"
    w_all = la.eig(a.toarray(), b.toarray(), right=False)
    want = nearest(w_all[np.isfinite(w_all)], sigma, k)
    got = np.concatenate([res.eigenvalues.astype(complex), res.complex_eigenvalues])
    assert_same_set(got, want, 1e-8)
    # the genuinely complex pair comes back separately, with complex vectors
    assert res.complex_eigenvalues.dtype == np.complex128
    assert_same_set(res.complex_eigenvalues, [0.31 + 0.05j, 0.31 - 0.05j], 1e-8)
    cv = res.complex_eigenvectors
    assert cv.shape == (a.shape[0], 2) and cv.dtype == np.complex128 and cv.flags.f_contiguous
    assert residuals(a, b, res.complex_eigenvalues, cv).max() < 1e-8
    assert res.eigenvectors.shape == (a.shape[0], k - 2)
    np.testing.assert_allclose(np.linalg.norm(res.eigenvectors, axis=0), 1.0, atol=1e-12)
    assert residuals(a, b, res.eigenvalues, res.eigenvectors).max() < 1e-8


# ---------------------------------------------------------------------------- input formats


def _base_matrix() -> sp.csr_matrix:
    a = random_hermitian(120, 0.05, seed=3, complex_=False) + 2.0 * sp.identity(120)
    return a.tocsr()


def _unsorted_with_duplicates(a: sp.csr_matrix) -> sp.csr_matrix:
    """Same matrix as `a`, but every row reversed and each entry split into two duplicates."""
    indptr, indices, data = [0], [], []
    for i in range(a.shape[0]):
        cols = a.indices[a.indptr[i] : a.indptr[i + 1]][::-1]
        vals = a.data[a.indptr[i] : a.indptr[i + 1]][::-1]
        indices += list(cols) + list(cols)
        data += list(vals * 0.5) + list(vals * 0.5)  # halves sum back exactly
        indptr.append(len(indices))
    m = sp.csr_matrix((np.array(data), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
                      shape=a.shape)
    assert not m.has_sorted_indices
    return m


def _coo_duplicates_and_cancellations(a) -> sp.coo_matrix:
    """COO of `a` with every entry split into two halves (shuffled) plus a symmetric pair of
    entries that cancel to exact zeros: duplicates are summed, the zeros dropped and compacted."""
    c = a.tocoo()
    j = int(np.flatnonzero(sp.csr_matrix(a)[[0]].toarray().ravel()[1:] == 0)[0]) + 1  # empty (0, j)
    rows = np.r_[c.row, c.row, 0, 0, j, j]
    cols = np.r_[c.col, c.col, j, j, 0, 0]
    vals = np.r_[c.data * 0.5, c.data * 0.5, 3.0, -3.0, 3.0, -3.0].astype(c.data.dtype)
    p = np.random.default_rng(2).permutation(rows.size)
    return sp.coo_matrix((vals[p], (rows[p], cols[p])), shape=a.shape)


def _int64(m):
    m = m.copy()
    if m.format == "coo":
        m.row, m.col = m.row.astype(np.int64), m.col.astype(np.int64)
    else:
        m.indptr, m.indices = m.indptr.astype(np.int64), m.indices.astype(np.int64)
    return m


FORMATS = {
    "csr": lambda a: a,
    "csc": lambda a: a.tocsc(),
    "coo": lambda a: a.tocoo(),
    "csr_array": lambda a: sp.csr_array(a),
    "csc_array": lambda a: sp.csc_array(a),
    "coo_array": lambda a: sp.coo_array(a),
    "csr_int64": lambda a: _int64(a),
    "csc_int64": lambda a: _int64(a.tocsc()),
    "coo_int64": lambda a: _int64(a.tocoo()),
    "coo_shuffled": lambda a: (lambda c, p: sp.coo_matrix((c.data[p], (c.row[p], c.col[p])), shape=c.shape))(
        a.tocoo(), np.random.default_rng(1).permutation(a.nnz)),
    "csr_unsorted_duplicates": _unsorted_with_duplicates,
    "coo_duplicates_cancellations": _coo_duplicates_and_cancellations,
    "csr_explicit_zeros": lambda a: _with_explicit_zero(a),
    "bsr": lambda a: a.tobsr(),
    "lil": lambda a: a.tolil(),
    "dense": lambda a: a.toarray(),
    "dense_fortran": lambda a: np.asfortranarray(a.toarray()),
    "dense_strided_view": lambda a: np.repeat(np.repeat(a.toarray(), 2, axis=0), 2, axis=1)[::2, ::2],
    "np_matrix": lambda a: a.todense(),
    "upper_triangle": lambda a: sp.triu(a, format="csr"),
    "lower_triangle_csc": lambda a: sp.tril(a, format="csc"),
    "complex128_zero_imag": lambda a: a.astype(np.complex128),
}


def _with_explicit_zero(a: sp.csr_matrix) -> sp.csr_matrix:
    m = a.tolil()
    m[0, a.shape[1] - 1] = 1.0  # create entries, then zero them: stored explicit zeros
    m[a.shape[0] - 1, 0] = 1.0
    m = m.tocsr()
    m.data[(m.indices == a.shape[1] - 1) & (np.repeat(np.arange(a.shape[0]), np.diff(m.indptr)) == 0)] = 0.0
    m.data[(m.indices == 0) & (np.repeat(np.arange(a.shape[0]), np.diff(m.indptr)) == a.shape[0] - 1)] = 0.0
    assert m.nnz == a.nnz + 2
    return m


@pytest.fixture(scope="module")
def format_reference():
    a = _base_matrix()
    return a, solve_quiet(a, sigma=2.1, k=5, config=DETERMINISTIC)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_input_formats_give_identical_results(format_reference, fmt):
    """Every format converts to the same CsrMatrix: deterministic solves agree bit for bit."""
    a, ref = format_reference
    m = FORMATS[fmt](a)
    res = solve_quiet(m, sigma=2.1, k=5, config=DETERMINISTIC)
    np.testing.assert_array_equal(res.eigenvalues, ref.eigenvalues)
    if fmt == "complex128_zero_imag":  # solved as real, returned as complex like the input
        assert res.eigenvectors.dtype == np.complex128
        assert any("solved as real" in w for w in res.report["warnings"])
        np.testing.assert_array_equal(res.eigenvectors.real, ref.eigenvectors)
    else:
        np.testing.assert_array_equal(res.eigenvectors, ref.eigenvectors)
    if fmt in ("upper_triangle", "lower_triangle_csc"):
        want = "upper" if fmt == "upper_triangle" else "lower"
        assert any(f"storage: {want} triangle mirrored" in w for w in res.report["warnings"])
    if fmt in ("csr_unsorted_duplicates", "coo_duplicates_cancellations"):
        assert any("duplicates: summed" in w for w in res.report["warnings"])
    if fmt in ("csr_explicit_zeros", "coo_duplicates_cancellations"):
        assert any("zeros: dropped" in w for w in res.report["warnings"])


COMPLEX_FORMATS = ("csr", "csc", "coo", "coo_duplicates_cancellations", "csr_int64", "dense", "dense_fortran",
                   "upper_triangle", "lower_triangle_csc")


@pytest.mark.parametrize("fmt", COMPLEX_FORMATS)
def test_complex_input_formats_give_identical_results(fmt):
    a = random_hermitian(100, 0.05, seed=17) + 1.5 * sp.identity(100)
    a = a.tocsr()
    ref = solve_quiet(a, sigma=1.4, k=4, config=DETERMINISTIC)
    res = solve_quiet(FORMATS[fmt](a), sigma=1.4, k=4, config=DETERMINISTIC)
    assert res.eigenvectors.dtype == np.complex128
    np.testing.assert_array_equal(res.eigenvalues, ref.eigenvalues)
    np.testing.assert_array_equal(res.eigenvectors, ref.eigenvectors)


def test_float32_input_is_converted_exactly():
    a = _base_matrix()
    a32 = a.astype(np.float32)
    res = solve_quiet(a32, sigma=2.1, k=5, config=DETERMINISTIC)
    ref = solve_quiet(a32.astype(np.float64), sigma=2.1, k=5, config=DETERMINISTIC)
    np.testing.assert_array_equal(res.eigenvalues, ref.eigenvalues)
    # complex64: same, through the complex path
    c = random_hermitian(100, 0.05, seed=9) + 3.0 * sp.identity(100)
    c64 = c.astype(np.complex64)
    res = solve_quiet(c64.tocsc(), sigma=0.5, k=4, config=DETERMINISTIC)
    ref = solve_quiet(c64.astype(np.complex128), sigma=0.5, k=4, config=DETERMINISTIC)
    np.testing.assert_array_equal(res.eigenvalues, ref.eigenvalues)
    np.testing.assert_array_equal(res.eigenvectors, ref.eigenvectors)


def test_complex_csc_is_not_conjugated_by_mistake():
    """CSC arrays are A^T; the conversion conjugates them back (A^H = A). A wrong sign would
    flip the eigenvectors to their conjugates: check A v = lambda v on the original A."""
    a = random_hermitian(120, 0.05, seed=21)
    res = solve_quiet(a.tocsc(), sigma=0.0, k=4)
    assert residuals(a, None, res.eigenvalues, res.eigenvectors).max() < 1e-8


def test_eigenvectors_are_not_copied():
    res = solve_quiet(lap1d(100), sigma=1.0, k=3)
    v = res.eigenvectors
    assert not v.flags.owndata and type(v.base).__name__ == "PyCapsule"


# ------------------------------------------------------------------------------- configuration


def test_config_dict_vs_file(tmp_path):
    a = lap2d(20, 17, shift=3.0)
    cfg = {"target": {"mode": "nearest", "sigma": 0.25, "k": 5}, "tol": 1e-11, "seed": 7,
           "deterministic": True, "threads": 2}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    r_dict = solve_quiet(a, config=cfg)
    r_file = solve_quiet(a, config=path)
    r_str = solve_quiet(a, config=str(path))
    assert r_dict.config == r_file.config == r_str.config
    assert r_dict.config["target"]["k"] == 5 and r_dict.config["tol"] == 1e-11 and r_dict.config["seed"] == 7
    np.testing.assert_array_equal(r_dict.eigenvalues, r_file.eigenvalues)
    np.testing.assert_array_equal(r_dict.eigenvectors, r_str.eigenvectors)
    assert cfg["target"]["k"] == 5  # the caller's dict is not modified
    # the effective config is a complete config.json: feeding it back reproduces the run
    r_again = solve_quiet(a, config=r_dict.config)
    np.testing.assert_array_equal(r_again.eigenvalues, r_dict.eigenvalues)
    assert set(r_dict.config) == set(hsolve.default_config())


def test_explicit_keywords_override_config():
    a = lap1d(200)
    cfg = {"target": {"mode": "nearest", "sigma": 0.5, "k": 5}, "tol": 1e-9}
    res = solve_quiet(a, config=cfg, k=3, tol=1e-12)
    assert res.config["target"] == {**res.config["target"], "sigma": 0.5, "k": 3}
    assert res.config["tol"] == 1e-12 and res.eigenvalues.size == 3
    res = solve_quiet(a, config=cfg)  # defaults do not override the config
    assert res.config["target"]["k"] == 5 and res.config["tol"] == 1e-9
    res = solve_quiet(a, config={"target": {"mode": "largest", "k": 2}}, sigma=0.3)  # sigma -> nearest
    assert res.config["target"]["mode"] == "nearest"
    np.testing.assert_allclose(res.eigenvalues, nearest(lap1d_eigs(200), 0.3, 2), atol=1e-11)
    res = solve_quiet(a, sigma=np.float64(0.7), k=np.int64(2), threads=np.int32(1))  # numpy scalars
    assert res.config["threads"] == 1


# ------------------------------------------------------------------------------------- errors


def test_error_mapping():
    a = lap1d(50).tolil()
    a[0, 1] = 5.0
    with pytest.raises(hsolve.NotHermitianError) as ei:
        hsolve.solve(a.tocsr(), k=2)
    assert ei.value.code == "not_hermitian" and isinstance(ei.value, ValueError)
    assert isinstance(ei.value, hsolve.HsolveError)

    good = lap1d(50)
    for bad_k in (0, -3, 51):
        with pytest.raises(hsolve.InvalidInputError) as ei:
            hsolve.solve(good, k=bad_k)
        assert ei.value.code == "invalid_input"
    with pytest.raises(TypeError):
        hsolve.solve(good, k=2.5)
    with pytest.raises(TypeError):
        hsolve.solve(good, k="6")
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(good, mode="biggest")
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(good, sigma=1.0, lower=0.0, upper=1.0)
    with pytest.raises(hsolve.InvalidInputError, match="unknown key"):
        hsolve.solve(good, config={"tolerance": 1e-8})
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(good, tol=float("nan"))
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(good, storage="triangle")


def test_input_errors():
    good = lap1d(30)
    with pytest.raises(hsolve.InvalidInputError, match="not square"):
        hsolve.solve(sp.random(30, 20, density=0.2, format="csr", random_state=0), k=2)
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(good, lap1d(31), k=2)  # B of a different order
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(np.zeros((3, 3, 3)), k=1)
    nan = good.copy()
    nan.data[3] = np.nan
    with pytest.raises(hsolve.InvalidInputError, match="non-finite"):
        hsolve.solve(nan, k=2)
    broken = good.copy()
    broken.indices = broken.indices.copy()
    broken.indices[5] = 1000  # column outside the matrix (scipy does not check lazily built arrays)
    with pytest.raises(hsolve.InvalidInputError, match="out of range"):
        hsolve.solve(broken, k=2)
    with pytest.raises(hsolve.InvalidInputError, match="other strict triangle"):
        hsolve.solve(good, k=2, storage="upper")
    with pytest.raises(TypeError):
        hsolve.solve(sla.aslinearoperator(good), k=2)
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(np.array([["a", "b"], ["c", "d"]], dtype=object), k=1)


def test_file_errors(tmp_path):
    with pytest.raises(hsolve.HsolveIOError) as ei:
        hsolve.solve(tmp_path / "missing.npz", k=2)
    assert ei.value.code == "io" and isinstance(ei.value, OSError)
    with pytest.raises(hsolve.HsolveIOError):
        hsolve.solve(lap1d(10), k=2, config=tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(hsolve.InvalidInputError):
        hsolve.solve(lap1d(10), k=2, config=tmp_path / "bad.json")
    with pytest.raises(hsolve.HsolveIOError):
        hsolve.load_matrix(tmp_path / "missing.npz")


def test_unsupported_maps_to_not_implemented():
    a, b = ghiep_pencil(30)
    with pytest.raises(hsolve.UnsupportedError) as ei:
        hsolve.solve(a, b, lower=-1.0, upper=1.0)  # interval targets need definite B (roadmap)
    assert ei.value.code == "unsupported" and isinstance(ei.value, NotImplementedError)


# ----------------------------------------------------------------------------- load_matrix


def test_load_matrix_roundtrip_and_solve_from_path(tmp_path):
    a = random_hermitian(80, 0.06, seed=4)
    sp.save_npz(tmp_path / "A.npz", sp.triu(a, format="csc"))  # upper triangle, csc on disk
    m, info = hsolve.load_matrix(tmp_path / "A.npz", return_info=True)
    assert isinstance(m, sp.csr_array) and m.dtype == np.complex128 and m.has_canonical_format
    assert info["source_format"] == "npz-csc" and info["storage"] == "upper"
    assert (abs(m - a)).max() == 0
    with pytest.raises(hsolve.NotHermitianError):  # a triangle declared as full storage
        hsolve.load_matrix(tmp_path / "A.npz", storage="full")
    # a path is loaded by the C++ reader exactly like the CLI does; same result as in memory
    r_path = solve_quiet(tmp_path / "A.npz", sigma=0.2, k=4, config=DETERMINISTIC)
    r_mem = solve_quiet(a, sigma=0.2, k=4, config=DETERMINISTIC)
    np.testing.assert_array_equal(r_path.eigenvalues, r_mem.eigenvalues)
    assert any("storage: upper triangle mirrored" in w for w in r_path.report["warnings"])


# ---------------------------------------------------------------------- GIL, threads, warnings


def test_gil_is_released_during_solve():
    a = lap2d(120, 110, shift=2.0)  # ~0.2 s solve: long enough to observe the other thread
    ticks: list[float] = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.002)

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    time.sleep(0.02)
    t0 = time.perf_counter()
    res = hsolve.solve(a, sigma=0.0, k=8)
    t1 = time.perf_counter()
    stop.set()
    th.join()
    assert res.success
    solve_s = res.report["time_total_s"]
    assert solve_s > 0.05, "test problem too small to observe the GIL"
    during = [t for t in ticks if t0 < t < t1]
    # With the GIL held the ticker could not run at all while C++ works (solve_s of t1 - t0);
    # released, it ticks every ~2 ms. Ask for a fifth of that (the cores are shared with OpenMP).
    assert len(during) >= max(5, 0.2 * solve_s / 0.002), (len(during), solve_s, t1 - t0)


def test_concurrent_solves_from_threads():
    """Solves from several Python threads are serialised inside the module and all correct."""
    n, exact = 2000, lap1d_eigs(2000)
    out: dict[int, hsolve.SolveResult] = {}
    errors: list[BaseException] = []

    def run(i: int):
        try:
            out[i] = solve_quiet(lap1d(n), sigma=0.5 + 0.3 * i, k=4)
        except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for i in range(3):
        assert_same_set(out[i].eigenvalues, nearest(exact, 0.5 + 0.3 * i, 4), 1e-11)


def test_verbose_logs_to_stderr(capfd):
    solve_quiet(lap1d(40), sigma=1.0, k=2, verbose=1)
    err = capfd.readouterr().err
    assert "problem n=40" in err and "done success=true" in err
    solve_quiet(lap1d(40), sigma=1.0, k=2)
    assert capfd.readouterr().err == ""


def test_empty_results_and_empty_matrix():
    res = solve_quiet(lap1d(50), lower=10.0, upper=11.0)  # no eigenvalue in the interval
    assert res.eigenvalues.shape == (0,) and res.eigenvectors.shape == (50, 0)
    assert res.complex_eigenvalues.shape == (0,) and res.complex_eigenvectors.shape == (50, 0)
    with pytest.raises(hsolve.InvalidInputError, match="empty"):
        hsolve.solve(sp.csr_matrix((0, 0)), k=1)


def test_threads_argument_does_not_leak():
    before = _hsolve.max_threads()
    solve_quiet(lap1d(100), sigma=1.0, k=2, threads=1)
    assert _hsolve.max_threads() == before


def test_unsuccessful_solve_warns():
    # tol below rounding level: no pair can be accepted (the driver's rescue cannot help)
    with pytest.warns(hsolve.HsolveWarning):
        res = hsolve.solve(lap1d(200), sigma=0.9, k=3, tol=1e-20, config={"max_restarts": 3})
    assert not res.success and res.eigenvalues.size == 3
    assert not any(p["accepted"] for p in res.report["pairs"])


def test_version():
    v = hsolve.version()
    assert v == hsolve.__version__ and len(v.split(".")) == 3
    assert "hsolve" in hsolve.build_info()
