"""Tests of the evaluation tools (tools/eval): metrics, references, scipy baseline, report.

Every case is tiny and created in tmp_path through tools/case.py; exact eigenpairs come from the
dense reference (LAPACK), so the expected metrics are known."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from tools import case as cio
from tools.eval import baseline_scipy, common, evaluate, reference, report

TRIPLE = 1.0                  # eigenvalue of multiplicity 3 in the prescribed spectra
SIGMA = -0.3                  # 4 nearest: -0.6, 0.2, -1.0, 0.7 (no tie); the 5th is the triple
SPECTRUM = np.array([-4.0, -3.0, -2.5, -1.0, -0.6, 0.2, 0.7, TRIPLE, TRIPLE, TRIPLE, 1.6, 2.5, 3.3, 4.0,
                     5.5, 7.0, -6.0, 8.0, 9.5, -8.5])


def _prescribed(eigs, rng, complex_=False):
    """Dense Hermitian Q diag(eigs) Q^H with a random unitary Q."""
    n = len(eigs)
    z = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if complex_ else 0.0)
    q, _ = np.linalg.qr(z)
    a = (q * eigs) @ q.conj().T
    return (a + a.conj().T) / 2


def _case(root, name, A, B=None, *, problem, sigma, k, **cfg):
    cio.save_case(name, sp.csr_matrix(A), None if B is None else sp.csr_matrix(B),
                  meta={"family": "test", "problem": problem}, config=cio.default_config(sigma, k, **cfg), root=root)


def _hermitian_case(root, name, problem="HEP", complex_=False, sigma=SIGMA, k=4, **cfg):
    rng = np.random.default_rng(7)
    A = _prescribed(SPECTRUM, rng, complex_)
    B = _prescribed(np.linspace(0.5, 3.0, SPECTRUM.size), rng, complex_) if problem == "GHEP" else None
    _case(root, name, A, B, problem=problem, sigma=sigma, k=k, **cfg)
    reference.build_reference(name, root)
    return cio.load_ref(name, root)


def _nearest(ref, sigma, k):
    idx = np.argsort(np.abs(ref.eigenvalues - sigma), kind="stable")[:k]
    return ref.eigenvalues[idx].copy(), ref.eigenvectors[:, idx].copy()


def _save(out, w, V, wc=None, Vc=None):
    return cio.save_result(out, w, V, {"solver": "exact", "time_total_s": 0.5, "peak_rss_bytes": 2**20,
                                       "success": True}, wc, Vc)


# ----------------------------------------------------------------------------- exact solutions

@pytest.mark.parametrize("problem,complex_", [("HEP", False), ("HEP", True), ("GHEP", False), ("GHEP", True)])
def test_exact_solution_passes(tmp_path, problem, complex_):
    ref = _hermitian_case(tmp_path, "c", problem, complex_)
    assert ref.info["kind"] == "dense" and ref.info["complete"] and ref.info["window"] is None
    w, V = _nearest(ref, SIGMA, 4)
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V), root=tmp_path)
    s = doc["summary"]
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["complex_scalars"] == complex_ and doc["norms"]["norm_method"]["A"] == "dense"
    assert s["max_backward_error"] < 1e-14 and s["max_eig_rel_error"] < 1e-12 and s["b_orthogonality"] < 1e-12
    assert doc["completeness"]["status"] == "complete" and doc["completeness"]["certified"]
    assert doc["completeness"]["found"] == doc["completeness"]["expected"] == 4
    assert [p["ref_index"] for p in doc["pairs"]] == list(np.argsort(np.abs(ref.eigenvalues - SIGMA))[:4])
    assert np.isclose(doc["norms"]["norm2_a"], 9.5)


@pytest.mark.parametrize("tgt", [{"mode": "smallest", "k": 3}, {"mode": "largest", "k": 2},
                                 {"mode": "interval", "lower": -1.5, "upper": 0.5}])
def test_other_target_modes(tmp_path, tgt):
    ref = _hermitian_case(tmp_path, "c")
    cfg = json.loads((tmp_path / "cases/c/config.json").read_text())
    cfg["target"] = tgt
    (tmp_path / "cases/c/config.json").write_text(json.dumps(cfg))
    vals = ref.eigenvalues
    idx = {"smallest": np.arange(3), "largest": np.arange(vals.size - 2, vals.size),
           "interval": np.flatnonzero((vals >= -1.5) & (vals <= 0.5))}[tgt["mode"]]
    doc = evaluate.evaluate("c", _save(tmp_path / "res", vals[idx], ref.eigenvectors[:, idx]), root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["completeness"]["status"] == "complete"
    # one eigenvalue short is detected for every mode
    doc = evaluate.evaluate("c", _save(tmp_path / "res2", vals[idx[1:]], ref.eigenvectors[:, idx[1:]]), root=tmp_path)
    assert doc["verdict"] == "fail" and doc["completeness"]["status"] == "incomplete"


# ------------------------------------------------------------------------------ detected defects

def test_perturbed_vector_is_detected(tmp_path):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 4)
    V[:, 1] += 1e-6 * np.random.default_rng(1).standard_normal(V.shape[0])
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V), root=tmp_path)
    assert doc["verdict"] == "fail" and "backward error" in doc["reasons"][0]
    assert [p["accepted"] for p in doc["pairs"]] == [True, False, True, True]
    assert doc["pairs"][1]["backward_error"] > 1e-8
    assert doc["completeness"]["status"] == "complete"     # the eigenvalue itself is still right


def test_missing_eigenvalue_is_detected(tmp_path):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 5)      # drop the nearest, add the 5th nearest instead
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w[1:], V[:, 1:]), root=tmp_path)
    comp = doc["completeness"]
    assert doc["verdict"] == "fail" and comp["status"] == "incomplete" and comp["found"] == 3
    assert [m["lambda_re"] for m in comp["missing"]] == pytest.approx([w[0]])
    assert comp["beyond_target"] == [3]


def test_duplicated_eigenvector_is_detected(tmp_path):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 4)
    w[2], V[:, 2] = w[0], V[:, 0]
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V), root=tmp_path)
    assert doc["verdict"] == "fail"
    assert doc["duplicates"] == [{"index": 2, "duplicate_of": 0, "distance": pytest.approx(0.0, abs=1e-7)}]
    assert doc["pairs"][2]["duplicate"] and doc["summary"]["b_orthogonality"] > 0.9
    assert doc["completeness"]["status"] == "incomplete"


def test_find_duplicates_tolerance():
    X = np.eye(5)[:, :3].copy()
    X = np.column_stack([X, X[:, 0] + 1e-9 * np.eye(5)[:, 4], X[:, 1] + 1e-2 * np.eye(5)[:, 4]])
    assert [(d["index"], d["duplicate_of"]) for d in evaluate.find_duplicates(X)] == [(3, 0)]


def test_spurious_eigenvalue_is_detected(tmp_path):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 4)
    w[3] += 0.05    # not an eigenvalue at all: wrong value, residual fails too
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V), root=tmp_path)
    assert doc["completeness"]["spurious"] == [3] and doc["verdict"] == "fail"


# -------------------------------------------------------------------------- multiplicity clusters

def test_multiplicity_cluster_accepts_any_basis(tmp_path):
    ref = _hermitian_case(tmp_path, "c", sigma=1.05, k=3)
    w, V = _nearest(ref, 1.05, 3)
    rot, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V @ rot), root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    (cluster,) = doc["clusters"]
    assert cluster["multiplicity"] == 3 and len(cluster["computed_indices"]) == 3
    assert cluster["max_angle"] < 1e-12


def test_cluster_cut_by_k_accepts_any_subspace(tmp_path):
    ref = _hermitian_case(tmp_path, "c", sigma=1.05, k=2)
    w, V = _nearest(ref, 1.05, 3)
    mix, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((3, 2)))
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w[:2], V @ mix), root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["completeness"]["expected"] == 2 and doc["completeness"]["found"] == 2
    assert doc["clusters"][0]["max_angle"] < 1e-12
    # a vector outside the eigenspace is caught by the subspace angle
    V[:, 1] = ref.eigenvectors[:, np.argmin(np.abs(ref.eigenvalues - 0.7))]
    doc = evaluate.evaluate("c", _save(tmp_path / "res2", w[:2], V[:, :2]), root=tmp_path)
    assert doc["summary"]["max_subspace_angle"] > 1.0 and doc["verdict"] == "fail"


# ------------------------------------------------------------------------------------- GHIEP

def _ghiep_dense(rng, lam_real):
    """Real symmetric A, B (B indefinite) with eigenvalues da/db plus the complex pair +-0.5i from the
    block A2 = [[0, .5], [.5, 0]], B2 = diag(1, -1); congruence by a random W keeps the spectrum."""
    db = np.array([1, -1, 1, -1, 1, 1, -1, 1.0])
    da = lam_real * db
    A0 = np.zeros((10, 10))
    B0 = np.zeros((10, 10))
    A0[:8, :8], B0[:8, :8] = np.diag(da), np.diag(db)
    A0[8:, 8:], B0[8:, 8:] = [[0, 0.5], [0.5, 0]], np.diag([1.0, -1.0])
    W = np.eye(10) + 0.3 * rng.standard_normal((10, 10))
    sym = lambda m: (m + m.T) / 2
    return sym(W.T @ A0 @ W), sym(W.T @ B0 @ W)


def test_ghiep_indefinite_b_with_complex_pair(tmp_path):
    lam_real = np.array([-2.0, -0.8, 0.3, 0.9, 1.7, 3.0, -5.0, 6.0])
    A, B = _ghiep_dense(np.random.default_rng(5), lam_real)
    _case(tmp_path, "g", A, B, problem="GHIEP", sigma=0.0, k=4)
    reference.build_reference("g", tmp_path)
    ref = cio.load_ref("g", tmp_path)
    assert np.iscomplexobj(ref.eigenvalues) and ref.info["complete"]
    assert np.sort_complex(ref.eigenvalues[ref.eigenvalues.imag != 0]) == pytest.approx([-0.5j, 0.5j])
    assert np.sort(ref.eigenvalues[ref.eigenvalues.imag == 0].real) == pytest.approx(np.sort(lam_real))
    idx = np.argsort(np.abs(ref.eigenvalues))[:4]          # 0.3, +-0.5i, -0.8
    is_real = ref.eigenvalues[idx].imag == 0
    r, c = idx[is_real], idx[~is_real]
    out = _save(tmp_path / "res", ref.eigenvalues[r].real, ref.eigenvectors[:, r].real,
                ref.eigenvalues[c], ref.eigenvectors[:, c])
    doc = evaluate.evaluate("g", out, root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["summary"]["n_complex"] == 2 and doc["summary"]["b_orthogonality"] is None
    assert doc["summary"]["max_backward_error"] < 1e-13
    assert doc["completeness"]["status"] == "complete"


def _sparse_ghiep(n, seed):
    """Sparse real-spectrum GHIEP: A = W^T Da W, B = W^T Db W with Db indefinite, W unit upper
    triangular -> lambda = da/db, all real."""
    rng = np.random.default_rng(seed)
    W = sp.identity(n) + 0.3 * sp.triu(sp.random(n, n, density=0.02, random_state=rng), 1)
    da = rng.uniform(-5, 5, n)
    db = rng.uniform(0.5, 2, n) * np.where(rng.random(n) < 0.3, -1.0, 1.0)
    A = (W.T @ sp.diags(da) @ W).tocsr()
    B = (W.T @ sp.diags(db) @ W).tocsr()
    return (A + A.T) / 2, (B + B.T) / 2, np.sort(da / db)


def test_baseline_ghiep_operator_passes_and_mode3_fails(tmp_path):
    A, B, lam = _sparse_ghiep(200, 11)
    _case(tmp_path, "g", A, B, problem="GHIEP", sigma=0.37, k=6)
    reference.build_reference("g", tmp_path)
    ref = cio.load_ref("g", tmp_path)
    assert not np.iscomplexobj(ref.eigenvalues) and ref.eigenvalues == pytest.approx(lam, rel=1e-9, abs=1e-9)
    out = baseline_scipy.run_baseline("g", tmp_path / "op", root=tmp_path)
    rep = cio.load_result(out).report
    assert rep["success"] and rep["problem"] == "generalized_indefinite" and rep["op_applies"] > 0
    doc = evaluate.evaluate("g", out, root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["solver"] == "scipy" and doc["summary"]["n_real"] == 6
    # the textbook eigs(A, k, M=B, sigma) is wrong for indefinite B, and the evaluator says so
    out = baseline_scipy.run_baseline("g", tmp_path / "mode3", root=tmp_path, ghiep_variant="mode3")
    doc = evaluate.evaluate("g", out, root=tmp_path)
    assert doc["verdict"] == "fail" and doc["summary"]["max_backward_error"] > 1e-6


# ---------------------------------------------------------------------------------- baseline

def _laplacian_case(root, name, n=300, shift=1.0, sigma=0.0, k=6, analytic=True):
    """1-D Dirichlet Laplacian minus shift*I: indefinite, eigenvalues 2 - 2cos(j pi/(n+1)) - shift."""
    A = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]) - shift * sp.identity(n)
    _case(root, name, A, problem="HEP", sigma=sigma, k=k)
    if analytic:
        j = np.arange(1, n + 1)
        cio.save_ref(name, 2 - 2 * np.cos(j * np.pi / (n + 1)) - shift, kind="analytic", complete=True, root=root)


def test_baseline_hep_end_to_end_with_analytic_reference(tmp_path):
    _laplacian_case(tmp_path, "lap")
    out = baseline_scipy.run_baseline("lap", tmp_path / "res", root=tmp_path)
    res = cio.load_result(out)
    rep = res.report
    assert rep["success"] and rep["method"] == "scipy_eigsh_shift_invert" and rep["problem"] == "standard"
    assert rep["time_total_s"] == pytest.approx(rep["time_factor_s"] + rep["time_iterate_s"])
    assert rep["peak_rss_bytes"] > 0 and len(rep["pairs"]) == 6
    assert list(np.abs(res.eigenvalues)) == sorted(np.abs(res.eigenvalues))     # nearest sigma=0 first
    doc = evaluate.evaluate("lap", out, root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["reference"]["kind"] == "analytic" and doc["completeness"]["certified"]
    assert doc["summary"]["max_eig_rel_error"] < 1e-10
    # the analytic reference is kept; --force replaces it by a dense one
    assert "kept existing analytic" in reference.build_reference("lap", tmp_path)
    assert "wrote dense" in reference.build_reference("lap", tmp_path, force=True)


@pytest.mark.parametrize("problem", ["HEP", "GHEP"])
def test_baseline_complex_hermitian(tmp_path, problem):
    _hermitian_case(tmp_path, "c", problem, complex_=True)
    out = baseline_scipy.run_baseline("c", tmp_path / "res", root=tmp_path)
    res = cio.load_result(out)
    assert res.report["complex_scalars"] and np.iscomplexobj(res.eigenvectors)
    doc = evaluate.evaluate("c", out, root=tmp_path)
    assert doc["verdict"] == "pass", doc["reasons"]
    assert doc["summary"]["b_orthogonality"] < 1e-10


def test_baseline_singular_shift_writes_failed_result(tmp_path):
    A = sp.diags(np.arange(-5.0, 15.0))       # sigma = 0 is an exact eigenvalue: K is singular
    _case(tmp_path, "s", A, problem="HEP", sigma=0.0, k=3)
    out = baseline_scipy.run_baseline("s", tmp_path / "res", root=tmp_path)
    rep = cio.load_result(out).report
    assert not rep["success"] and rep["pairs"] == [] and "splu" in rep["warnings"][0]
    doc = evaluate.evaluate("s", out, root=tmp_path)
    assert doc["verdict"] == "fail" and "no eigenpairs returned" in doc["reasons"]


# ---------------------------------------------------------------------- references and norms

def test_solver_reference_then_dense_upgrade(tmp_path):
    _laplacian_case(tmp_path, "lap", n=200, sigma=0.3, k=5, analytic=False)
    msg = reference.build_reference("lap", tmp_path, dense_max=50)
    ref = cio.load_ref("lap", tmp_path)
    assert "wrote solver" in msg and ref.info["kind"] == "solver" and not ref.info["complete"]
    assert ref.eigenvalues.size == 15 and ref.info["window"][0] < 0.3 < ref.info["window"][1]
    w, V = _nearest(ref, 0.3, 5)
    doc = evaluate.evaluate("lap", _save(tmp_path / "res", w, V), root=tmp_path, dense_max=50)
    assert doc["verdict"] == "pass" and doc["completeness"]["status"] == "consistent"
    assert doc["norms"]["norm_method"]["A"] == "eigsh"
    doc = evaluate.evaluate("lap", _save(tmp_path / "res2", w[1:], V[:, 1:]), root=tmp_path, dense_max=50)
    assert doc["completeness"]["status"] == "incomplete"        # 4 distinct < k regardless of the reference
    # a dense reference outranks the solver one
    assert "wrote dense" in reference.build_reference("lap", tmp_path)
    assert cio.load_ref("lap", tmp_path).info["kind"] == "dense"


def test_solver_reference_falls_back_to_k_pairs(tmp_path, monkeypatch):
    _laplacian_case(tmp_path, "lap", n=200, sigma=0.3, k=5, analytic=False)
    monkeypatch.setattr(reference, "EXTRA_MAXITER", 1)    # the k + extra attempt cannot converge
    msg = reference.build_reference("lap", tmp_path, dense_max=50)
    ref = cio.load_ref("lap", tmp_path)
    assert "fell back to k=5" in msg and ref.eigenvalues.size == 5
    assert ref.eigenvalues == pytest.approx(np.sort(_nearest(ref, 0.3, 5)[0]))


def test_solver_reference_flags_multiplicity(tmp_path):
    _hermitian_case(tmp_path, "c", sigma=1.05, k=3)
    msg = reference.build_reference("c", tmp_path, force=True, dense_max=10, extra=2)
    assert "wrote solver" in msg and "multiplicity >= 3" in msg


def test_continuum_reference_is_not_compared_and_gets_replaced(tmp_path):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 4)
    # continuum-model values: the discrete eigenvalues are O(h^2) = 1e-3 away
    cio.save_ref("c", ref.eigenvalues + 1e-3, kind="partial-analytic", complete=True, root=tmp_path)
    doc = evaluate.evaluate("c", _save(tmp_path / "res", w, V), root=tmp_path)
    assert doc["reference"]["kind"] == "partial-analytic" and not doc["reference"]["comparable"]
    assert doc["verdict"] == "pass" and doc["completeness"]["status"] == "unknown"
    assert all(p["ref_index"] is None for p in doc["pairs"])
    assert "wrote dense" in reference.build_reference("c", tmp_path)
    archived = tmp_path / "refs/c/superseded/partial-analytic"
    assert json.loads((archived / "ref.json").read_text())["kind"] == "partial-analytic"
    assert np.load(archived / "eigenvalues.npy") == pytest.approx(ref.eigenvalues + 1e-3)
    doc = evaluate.evaluate("c", tmp_path / "res", root=tmp_path)
    assert doc["reference"]["comparable"] and doc["completeness"]["status"] == "complete"


def test_norm_estimate_eigsh_is_a_tight_lower_bound():
    rng = np.random.default_rng(2)
    M = sp.random(300, 300, density=0.05, random_state=rng) - 0.5 * sp.random(300, 300, density=0.05, random_state=rng)
    M = (M + M.T).tocsr()
    exact, method = evaluate.hermitian_norm2(M)
    est, method_est = evaluate.hermitian_norm2(M, dense_max=100)
    assert (method, method_est) == ("dense", "eigsh")
    assert est <= exact * (1 + 1e-12) and est == pytest.approx(exact, rel=1e-7)


def test_dense_refuses_large_matrices():
    with pytest.raises(ValueError):
        common.to_dense(sp.identity(10, format="csr"), dense_max=5)


# ------------------------------------------------------------------------------- CLI and report

def test_cli_and_report(tmp_path, monkeypatch):
    ref = _hermitian_case(tmp_path, "c")
    w, V = _nearest(ref, SIGMA, 4)
    good = _save(tmp_path / "good", w, V)
    bad = _save(tmp_path / "bad", w[:3], V[:, :3])
    monkeypatch.delenv("HS_GUARDED", raising=False)
    with pytest.raises(SystemExit):
        evaluate.main(["c", str(good), "--root", str(tmp_path)])
    monkeypatch.setenv("HS_GUARDED", "1")
    assert evaluate.main(["c", str(good), "--root", str(tmp_path), "--solver", "exact"]) == 0
    assert evaluate.main(["c", str(bad), "--root", str(tmp_path), "--solver", "truncated"]) == 0
    doc = json.loads((good / evaluate.EVAL_FILE).read_text())
    assert doc["verdict"] == "pass" and doc["schema_version"] == cio.SCHEMA_VERSION
    md = tmp_path / "results" / "summary.md"
    assert report.main([str(tmp_path / "good"), str(bad / evaluate.EVAL_FILE), "--out", str(md)]) == 0
    text = md.read_text()
    assert "| c | exact | HEP | 20 | 4 | 0.5 | 1 |" in text and "complete (4/4) | pass |" in text
    assert "incomplete (3/4) | **fail** |" in text and "c / truncated: " in text
