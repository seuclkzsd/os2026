"""P7 memory budget and precision through the Python API: memory_budget_gb "auto" (the default)
derived from HSOLVE_MEMORY_LIMIT_GB and the other limits, a configured budget, the report's
"memory" and "precision" records, and factor_precision "single" that falls back to double instead
of returning a wrong answer.

The extension is imported from the build tree: $HSOLVE_PYTHON_DIR, default build/release/python.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[2]
_PY_DIR = Path(os.environ.get("HSOLVE_PYTHON_DIR", _ROOT / "build" / "release" / "python"))
if not (_PY_DIR / "hsolve" / "__init__.py").is_file():
    pytest.skip(f"hsolve Python package not built in {_PY_DIR} (tools/dev/build.sh release)",
                allow_module_level=True)
sys.path.insert(0, str(_PY_DIR))
import hsolve  # noqa: E402


def lap2d(nx: int, ny: int, shift: float = 0.0) -> sp.csr_matrix:
    e = [np.ones(m) for m in (nx, ny)]
    l1 = [sp.diags([-x[:-1], 2 * x, -x[:-1]], [-1, 0, 1], format="csr") for x in e]
    return (sp.kronsum(l1[0], l1[1], format="csr") - shift * sp.identity(nx * ny, format="csr")).tocsr()


def lap2d_eigs(nx: int, ny: int, shift: float = 0.0) -> np.ndarray:
    w = [2.0 - 2.0 * np.cos(np.arange(1, m + 1) * np.pi / (m + 1)) for m in (nx, ny)]
    return np.sort((w[0][:, None] + w[1][None, :]).ravel()) - shift


def test_default_budget_is_auto_and_old_configs_still_work():
    assert hsolve.default_config()["memory_budget_gb"] == "auto"
    a = lap2d(20, 21)
    for budget in ("auto", 0, 14.0):
        r = hsolve.solve(a, sigma=1.0, k=4, config={"memory_budget_gb": budget})
        assert r.success
        assert r.config["memory_budget_gb"] == ("auto" if budget in ("auto", 0) else budget)
        assert (r.report["memory"]["source"] == "config") == (budget == 14.0)
    with pytest.raises(hsolve.InvalidInputError, match="memory_budget_gb"):
        hsolve.solve(a, sigma=1.0, k=4, config={"memory_budget_gb": -1})


def test_auto_budget_follows_the_declared_limit(monkeypatch):
    monkeypatch.setenv("HSOLVE_MEMORY_LIMIT_GB", "2")
    r = hsolve.solve(lap2d(20, 21), sigma=1.0, k=4)
    m = r.report["memory"]
    assert 0 < m["limit_gb"] <= 2.0          # the declaration, or a smaller limit of this machine
    assert m["source"] == m["limit_source"]
    assert 0 < m["budget_gb"] < m["limit_gb"]
    names = [s["name"] for s in m["limits"]]
    assert names == ["cgroup", "env", "rlimit_as", "rlimit_data", "mem_total"]
    env = m["limits"][1]
    assert env["gib"] == 2.0 and "HSOLVE_MEMORY_LIMIT_GB" in env["detail"]


def test_single_precision_never_returns_a_wrong_answer():
    # sigma 1e-9 ||K|| from an eigenvalue: eps_single * cond(K) >> 1; the single attempt is
    # abandoned and the double result returned (before P7: pairs above tol, failed certificate)
    nx, ny, shift = 70, 71, 2.0
    w = lap2d_eigs(nx, ny, shift)
    sigma = w[2500] + 8e-9
    r = hsolve.solve(lap2d(nx, ny, shift), sigma=sigma, k=6, config={"factor_precision": "single"})
    assert r.success
    p = r.report["precision"]
    assert (p["requested"], p["used"], p["fallback"]) == ("single", "double", True)
    assert p["fallback_reason"] and p["attempt_s"] > 0
    want = np.sort(w[np.argsort(np.abs(w - sigma), kind="stable")[:6]])
    np.testing.assert_allclose(np.sort(r.eigenvalues), want, atol=1e-9)
