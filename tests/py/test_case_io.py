"""Round-trip tests for tools/case.py (the single owner of on-disk formats)."""
import numpy as np
import pytest
import scipy.sparse as sp

from tools import case


def _herm(n, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.2, random_state=rng, format="csr")
    if complex_:
        a = a + 1j * sp.random(n, n, density=0.2, random_state=rng, format="csr")
    return (a + a.conj().T).tocsr()


def test_case_roundtrip(tmp_path):
    A = _herm(20, complex_=True)
    B = sp.identity(20, format="csr") * 2.0
    d = case.save_case("c1", A, B, meta={"family": "test", "problem": "GHEP", "scenarios": ["multiplicity"]},
                       config=case.default_config(0.5, 4), root=tmp_path)
    assert d == tmp_path / "cases" / "c1"
    c = case.load_case("c1", root=tmp_path)
    assert abs(c.A - A).max() == 0 and abs(c.B - B).max() == 0
    assert c.meta["dtype"] == "complex128" and c.meta["n"] == 20
    assert c.config["target"] == {"mode": "nearest", "sigma": 0.5, "k": 4}
    assert case.list_cases(tmp_path) == ["c1"]


def test_meta_rejects_reference_data_and_bad_tags(tmp_path):
    A = _herm(5)
    with pytest.raises(ValueError):
        case.save_case("x", A, meta={"family": "t", "problem": "HEP", "reference": [1, 2]}, root=tmp_path)
    with pytest.raises(ValueError):
        case.save_case("x", A, meta={"family": "t", "problem": "HEP", "scenarios": ["nope"]}, root=tmp_path)
    with pytest.raises(ValueError):
        case.save_case("x", sp.random(5, 5, density=0.5, random_state=1), meta={"family": "t", "problem": "HEP"},
                       root=tmp_path)


def test_ref_and_result_roundtrip(tmp_path):
    case.save_ref("c1", [3.0, 1.0, 2.0], kind="analytic", complete=True, window=(0.0, 4.0), root=tmp_path)
    r = case.load_ref("c1", root=tmp_path)
    assert list(r.eigenvalues) == [1.0, 2.0, 3.0] and r.info["complete"] and r.info["window"] == [0.0, 4.0]
    assert case.load_ref("missing", root=tmp_path) is None
    vecs = np.eye(4)[:, :2]
    out = case.save_result(tmp_path / "res", [1.0, 2.0], vecs, {"success": True})
    res = case.load_result(out)
    assert res.eigenvectors.flags.f_contiguous and res.report["schema_version"] == case.SCHEMA_VERSION
    assert res.complex_eigenvalues is None
