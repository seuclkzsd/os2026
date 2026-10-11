"""Offline tests for tools/gen/suitesparse.py: a file:// mirror of tiny .tar.gz archives."""
import io
import json
import tarfile

import numpy as np
import pytest
import scipy.io
import scipy.linalg as sl
import scipy.sparse as sp

from tools import case
from tools.gen import suitesparse as ss

GROUP = "Test"


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    m = sp.random(n, n, density=0.3, random_state=rng)
    return (m @ m.T + n * sp.identity(n)).tocsr()


def _sym(n, seed, complex_=False):
    rng = np.random.default_rng(seed)
    m = sp.random(n, n, density=0.3, random_state=rng)
    if complex_:
        m = m + 1j * sp.random(n, n, density=0.3, random_state=rng)
    return (m + m.conj().T - 0.5 * sp.identity(n)).tocsr()


def _mtx_bytes(matrix, symmetry):
    buf = io.BytesIO()
    scipy.io.mmwrite(buf, sp.coo_matrix(matrix), symmetry=symmetry)
    return buf.getvalue()


def _publish(mirror, name, matrix, symmetry, extra_members=()):
    """Write MM/<GROUP>/<name>.tar.gz containing <name>/<name>.mtx (+ extra raw members)."""
    folder = mirror / "MM" / GROUP
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / f"{name}.tar.gz", "w:gz") as tf:
        for member, data in [(f"{name}/{name}.mtx", _mtx_bytes(matrix, symmetry)), *extra_members]:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _registry(*names):
    return {n: ss.Matrix(GROUP, n, 0, 0, "real", False, 0.1, f"test matrix {n}") for n in names}


def _fetch(tmp_path, cname, cases, registry, **kw):
    return ss.fetch_case(cname, root=tmp_path / "data", cases=cases, registry=registry,
                         base_url=(tmp_path / "mirror" / "MM").as_uri(), **kw)


def _assert_sigma_inside(c):
    dense = sl.eigh(c.A.toarray(), None if c.B is None else c.B.toarray(), eigvals_only=True)
    for t in c.meta["targets"]:
        assert dense[0] <= t["sigma"] <= dense[-1]


@pytest.mark.parametrize("symmetry,complex_", [("symmetric", False), ("hermitian", True),
                                               ("general", False), ("general", True)])
def test_hermitian_matrix_becomes_hep_case(tmp_path, symmetry, complex_):
    a = _sym(30, 1, complex_)
    _publish(tmp_path / "mirror", "M1", a, symmetry)
    path = _fetch(tmp_path, "t-m1", {"t-m1": ss.CaseSpec("M1")}, _registry("M1"))
    c = case.load_case(path)
    assert abs(c.A - a).max() < 1e-15 * abs(a).max()
    assert c.B is None and c.meta["problem"] == "HEP" and c.meta["family"] == "suitesparse"
    assert c.meta["license"] == "CC-BY 4.0" and "Davis" in c.meta["citation"] and "Test/M1" in c.meta["citation"]
    src = c.meta["params"]["matrices"][0]
    assert src["mm_header"]["symmetry"] == symmetry and len(src["archive_sha256"]) == 64
    assert ("complex-hermitian" in c.meta["scenarios"]) == complex_
    assert c.config["target"]["sigma"] == c.meta["targets"][0]["sigma"]
    _assert_sigma_inside(c)
    sel = c.meta["params"]["target_selection"]
    lo_g, hi_g = sel["gershgorin_interval"]
    assert lo_g <= sel["ritz_interval"][0] <= sel["ritz_interval"][1] <= hi_g


def test_single_stored_triangle_is_symmetrized(tmp_path):
    a = _sym(25, 2)
    _publish(tmp_path / "mirror", "Low", sp.tril(a), "general")
    c = case.load_case(_fetch(tmp_path, "t-low", {"t-low": ss.CaseSpec("Low")}, _registry("Low")))
    assert abs(c.A - a).max() < 1e-15 * abs(a).max()
    assert any("lower triangle" in note for note in c.meta["params"]["matrices"][0]["notes"])


def test_explicit_zeros_dropped_and_real_complex_demoted(tmp_path):
    path = tmp_path / "z.mtx"
    path.write_text("%%MatrixMarket matrix coordinate complex general\n3 3 5\n"
                    "1 1 2.0 0.0\n2 2 3.0 0.0\n1 2 0.0 0.0\n2 1 0.0 0.0\n3 3 1.0 0.0\n")
    m, info = ss.read_hermitian(path)
    assert m.dtype == np.float64 and m.nnz == 3 and m.diagonal().tolist() == [2.0, 3.0, 1.0]
    assert info["notes"] == ["dropped 2 explicit zeros", "complex file with zero imaginary parts: stored as real"]
    path.write_text("%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n1 1\n")
    with pytest.raises(ss.Skipped, match="pattern"):
        ss.read_hermitian(path)


def test_pencils_classified_by_inertia_of_b(tmp_path):
    mirror = tmp_path / "mirror"
    _publish(mirror, "K", _sym(30, 4), "symmetric")
    _publish(mirror, "M", _spd(30, 5), "symmetric")
    _publish(mirror, "S", _sym(30, 6), "symmetric")
    cases = {"t-km": ss.CaseSpec("K", "M", "pencil"), "t-ks": ss.CaseSpec("K", "S")}
    reg = _registry("K", "M", "S")
    c = case.load_case(_fetch(tmp_path, "t-km", cases, reg))
    assert c.meta["problem"] == "GHEP" and c.meta["inertia"]["B"] == [0, 0, 30]
    assert c.meta["description"].startswith("pencil")
    _assert_sigma_inside(c)
    c = case.load_case(_fetch(tmp_path, "t-ks", cases, reg))
    assert c.meta["problem"] == "GHIEP" and "indefinite-B" in c.meta["scenarios"]


def test_pencil_from_auxiliary_member_downloads_archive_once(tmp_path, monkeypatch):
    """Guettel TEM layout: <Name>/<Name>.mtx (curl-curl C) and <Name>/<Name>_M.mtx (mass M)."""
    c_mat, m_mat = _sym(30, 11), _spd(30, 12)
    _publish(tmp_path / "mirror", "P", c_mat, "symmetric", [("P/P_M.mtx", _mtx_bytes(m_mat, "symmetric"))])
    reg = _registry("P") | {"P_M": ss.Matrix(GROUP, "P", None, None, "real", True, 0.1, "mass", member="P_M")}
    calls = []
    download = ss._download
    monkeypatch.setattr(ss, "_download", lambda *a, **kw: calls.append(a[0]) or download(*a, **kw))
    c = case.load_case(_fetch(tmp_path, "t-p", {"t-p": ss.CaseSpec("P", "P_M")}, reg, refresh=True))
    assert len(calls) == 1 and c.meta["problem"] == "GHEP"
    assert abs(c.A - c_mat).max() < 1e-15 * abs(c_mat).max() and abs(c.B - m_mat).max() < 1e-15 * abs(m_mat).max()
    files = [s["file"] for s in c.meta["params"]["matrices"]]
    assert files == ["P.mtx", "P_M.mtx"] and c.meta["citation"].endswith("matrices: Test/P")
    reg_missing = reg | {"P_K": ss.Matrix(GROUP, "P", None, None, "real", True, 0.1, "absent", member="P_K")}
    with pytest.raises(ss.Skipped, match="P_K.mtx"):
        _fetch(tmp_path, "t-pk", {"t-pk": ss.CaseSpec("P", "P_K")}, reg_missing)


def test_complex_symmetric_and_nonsymmetric_rejected(tmp_path):
    rng = np.random.default_rng(7)
    m = sp.random(20, 20, density=0.3, random_state=rng) * (1 + 2j)
    _publish(tmp_path / "mirror", "CS", m + m.T + sp.identity(20), "general")
    _publish(tmp_path / "mirror", "NS", sp.random(20, 20, density=0.3, random_state=rng) + sp.identity(20),
             "general")
    reg = _registry("CS", "NS")
    cases = {"t-cs": ss.CaseSpec("CS"), "t-ns": ss.CaseSpec("NS")}
    with pytest.raises(ss.Skipped, match="complex symmetric"):
        _fetch(tmp_path, "t-cs", cases, reg)
    with pytest.raises(ss.Skipped, match="not Hermitian"):
        _fetch(tmp_path, "t-ns", cases, reg)
    assert not (tmp_path / "data" / "cases" / "t-cs").exists()


def test_offline_size_cap_and_unsafe_archives_skip_cleanly(tmp_path):
    reg = _registry("M1", "Evil")
    cases = {"t-m1": ss.CaseSpec("M1"), "t-evil": ss.CaseSpec("Evil")}
    with pytest.raises(ss.Skipped, match="download failed"):          # nothing published: offline
        _fetch(tmp_path, "t-m1", cases, reg)
    assert not (tmp_path / "data" / "downloads" / "M1").exists()     # failed fresh download cleaned
    _publish(tmp_path / "mirror", "M1", _sym(30, 1), "symmetric")
    with pytest.raises(ss.Skipped, match="cap"):
        _fetch(tmp_path, "t-m1", cases, reg, max_mb=1e-4)
    _publish(tmp_path / "mirror", "Evil", _sym(5, 1), "symmetric", [("../escape.mtx", b"%%x\n")])
    with pytest.raises(ss.Skipped, match="unusable archive"):
        _fetch(tmp_path, "t-evil", cases, reg)
    assert not (tmp_path / "data" / "downloads" / "Evil").exists()


def test_completed_download_is_reused_and_refresh_refetches(tmp_path):
    reg = _registry("M1")
    cases = {"t-m1": ss.CaseSpec("M1")}
    _publish(tmp_path / "mirror", "M1", _sym(30, 1), "symmetric")
    _fetch(tmp_path, "t-m1", cases, reg)
    record = json.loads((tmp_path / "data" / "downloads" / "M1" / "download.json").read_text())
    assert record["bytes"] > 0 and record["url"].endswith(f"{GROUP}/M1.tar.gz")
    (tmp_path / "mirror" / "MM" / GROUP / "M1.tar.gz").unlink()      # go "offline"
    _fetch(tmp_path, "t-m1", cases, reg)                             # served from the record
    with pytest.raises(ss.Skipped):
        _fetch(tmp_path, "t-m1", cases, reg, refresh=True)
    (tmp_path / "data" / "downloads" / "M1").mkdir(parents=True)     # leftover without a record
    with pytest.raises(ss.Skipped, match="refresh"):
        _fetch(tmp_path, "t-m1", cases, reg)


@pytest.mark.parametrize("complex_", [False, True])
def test_hpd_inertia_matches_dense(complex_):
    for seed in range(3):
        spd = _spd(40, seed)
        if complex_:
            h = _sym(40, seed, True)
            spd = (h @ h.conj().T + 40 * sp.identity(40)).tocsr()
        assert ss.hpd_inertia(spd) == [0, 0, 40]
        indef = _sym(40, seed + 10, complex_)
        w = np.linalg.eigvalsh(indef.toarray())
        inertia = ss.hpd_inertia(indef)
        assert inertia is None or inertia == [int(np.sum(w < 0)), 0, int(np.sum(w > 0))]
    zero_diag = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert ss.hpd_inertia(zero_diag) is None                  # off-diagonal pivot: not HPD


@pytest.mark.parametrize("complex_", [False, True])
def test_ritz_interval_is_inside_the_spectrum(complex_):
    a = _sym(120, 8, complex_)
    lo, hi = ss.ritz_interval(a, steps=15)
    w = np.linalg.eigvalsh(a.toarray())
    tol = 1e-12 * np.abs(w).max()                      # extreme Ritz values may converge
    assert w[0] - tol <= lo < hi <= w[-1] + tol and lo < 0 < hi
    b = _spd(120, 9)
    lo, hi = ss.ritz_interval(a, b, steps=15)
    w = sl.eigh(a.toarray(), b.toarray(), eigvals_only=True)
    tol = 1e-12 * np.abs(w).max()
    assert w[0] - tol <= lo < hi <= w[-1] + tol
    lo, hi = ss.ritz_interval(sp.identity(50, format="csr") * 3.0)  # invariant subspace at once
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


def test_registry_and_cli(tmp_path, capsys):
    assert {"TEM27623", "qa8fk", "qa8fm", "helm3d01", "Si34H36", "Ge87H76", "2cubes_sphere",
            "3Dspectralwave2", "gsm_106857", "offshore", "helm2d03", "Ga10As10H30",
            "TEM152078", "TEM181302", "TEM27623_M"} <= set(ss.REGISTRY)
    for spec in ss.CASES.values():
        assert spec.a in ss.REGISTRY and (spec.b is None or spec.b in ss.REGISTRY)
    assert ss.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert "Cunningham/qa8fk+Cunningham/qa8fm" in listing
    assert "Guettel/TEM27623+Guettel/TEM27623_M" in listing and "    6.2 MB" in listing
    rc = ss.main(["ss-tem27623", "--root", str(tmp_path), "--base-url", (tmp_path / "empty").as_uri()])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["case"] == "ss-tem27623" and "skipped" in out
