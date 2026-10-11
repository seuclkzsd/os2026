"""Tests for tools/gen/synthetic.py: exact spectra against dense eigensolvers on small sizes."""
import json

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp

from tools import case
from tools.gen import synthetic as syn


def _max_rel_err(computed, exact):
    computed, exact = np.sort(computed), np.sort(exact)
    return np.max(np.abs(computed - exact)) / max(1.0, np.max(np.abs(exact)))


def _assert_hermitian(a):
    assert (a - a.conj().T).count_nonzero() == 0
    if np.iscomplexobj(a.data):
        assert not np.any(a.diagonal().imag)


# ------------------------------------------------------------------------- spectrum utilities

def test_distinct_levels_merges_only_near_equal_values():
    lev, counts = syn.distinct_levels([2.0, 0.0, 1.0, 0.0, 1.0 + 1e-14, 1.0])
    assert lev.tolist() == [0.0, 1.0, 2.0] and counts.tolist() == [2, 3, 1]
    lev, counts = syn.distinct_levels([1e-20, 2e-20])
    assert counts.tolist() == [1, 1]


def test_nearest_k_closed_skips_ties():
    w = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0]
    assert syn.nearest_k_closed(w, 0.9, 1) == 3          # the triple at distance 0.1
    assert syn.nearest_k_closed(w, 0.9, 4) == 5          # then the pair at distance 0.9
    assert syn.nearest_k_closed([-1.0, 1.0, 3.0], 0.0, 1) == 2   # mirrored around sigma
    assert syn.nearest_k_closed(w, 10.0, 6) == 6
    with pytest.raises(ValueError):
        syn.nearest_k_closed(w, 0.0, 7)


def test_targets_and_inertia():
    w = np.array([-2.0, 0.0, 0.0, 1.0, 1.0, 3.0])
    t = syn.target_between_levels(w, 2, 1, "why")
    assert t == {"sigma": pytest.approx(1.8), "k": 2, "why": "why"}
    with pytest.raises(ValueError):
        syn.target_between_levels(w, 3, 1, "last level has no successor")
    with pytest.raises(ValueError):
        syn.target_at(np.zeros(10), 1.0, 1, "huge cluster", k_max=5)
    assert syn.inertia_counts(w) == [1, 2, 3]
    assert syn.inertia_counts(w, zero_tol=1.5) == [1, 4, 1]


def test_hermitian_part_is_bitwise_hermitian():
    rng = np.random.default_rng(0)
    m = sp.random(30, 30, density=0.2, random_state=rng) + 1j * sp.random(30, 30, density=0.2, random_state=rng)
    h = syn.hermitian_part(m @ m.conj().T * (1 + 1e-15))
    _assert_hermitian(h)


# ------------------------------------------------------------- (a, d) tensor-product operators

@pytest.mark.parametrize("shape,spacing,shift", [((7, 9), None, 0.0), ((7, 9), (0.5, 1.5), 3.0),
                                                 ((4, 5, 6), (1.0, 0.7, 1.3), 5.5)])
def test_laplacian_matches_dense(shape, spacing, shift):
    a = syn.laplacian(shape, spacing, shift)
    assert a.shape == (np.prod(shape),) * 2 and a.dtype == np.float64
    _assert_hermitian(a)
    w = syn.laplacian_eigenvalues(shape, spacing, shift)
    assert _max_rel_err(sl.eigvalsh(a.toarray()), w) < 1e-13


def test_laplacian_equal_sizes_give_bitwise_multiplicities():
    """Permuted index tuples must give bitwise-equal values: every eigenvalue of a triple with
    three distinct indices occurs >= 6 times (accidental degeneracies such as
    t_k + t_(n+1-k) = 4 may merge levels further, hence >=)."""
    _, counts = np.unique(syn.laplacian_eigenvalues((5, 5, 5)), return_counts=True)
    assert counts[counts >= 6].sum() >= 5 * 4 * 3
    _, counts = np.unique(syn.laplacian_eigenvalues((8, 8)), return_counts=True)
    assert counts[counts >= 2].sum() >= 8 * 7


@pytest.mark.parametrize("order,points", [(1, 27), (2, 125)])
def test_stencil_operator_matches_dense_and_has_full_stencil(order, points):
    cells = (5, 6, 7)
    c = syn.stencil_shift(cells, order, 0.2)
    a = syn.stencil_operator(cells, order, c)
    _assert_hermitian(a)
    w = syn.stencil_eigenvalues(cells, order, c)
    assert _max_rel_err(sl.eigvalsh(a.toarray()), w) < 1e-13
    neg, zero, pos = syn.inertia_counts(w)
    assert zero == 0 and 0.1 * w.size < neg < 0.3 * w.size
    grid = (2 * order + 3,) * 3                              # has a point with all neighbours
    shifted = syn.stencil_operator(grid, order, syn.stencil_shift(grid, order, 0.05))
    assert shifted.getnnz(axis=1).max() == points


def test_shift_between_levels_avoids_eigenvalues():
    w = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    c = syn.shift_between_levels(w, 0.5)
    assert 1.0 < c < 2.0
    with pytest.raises(ValueError):
        syn.laplacian((3,), None)


# ------------------------------------------------------------- (b, c) prescribed spectra

@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("kind", syn.SPECTRUM_KINDS)
def test_prescribed_hermitian_matches_spectrum(kind, complex_):
    spec = syn.spectrum_preset(kind, 300, seed=1)
    assert spec.eigenvalues.size == 300 and np.all(np.diff(spec.eigenvalues) >= 0)
    a = syn.prescribed_hermitian(spec.eigenvalues, complex_=complex_, seed=2, grouping=spec.grouping)
    assert np.iscomplexobj(a.data) == complex_
    _assert_hermitian(a)
    assert a.getnnz(axis=1).max() <= 3 * 8
    assert _max_rel_err(sl.eigvalsh(a.toarray()), spec.eigenvalues) < 1e-13
    for t in spec.targets:
        assert t["k"] == syn.nearest_k_closed(spec.eigenvalues, t["sigma"], t["k"])


def test_spectrum_presets_have_their_features():
    w = syn.spectrum_preset("zero-cluster", 1000, 3).eigenvalues
    assert np.count_nonzero(np.abs(w) <= 1e-6) == 100
    _, counts = syn.distinct_levels(syn.spectrum_preset("multiplicity", 1000, 3).eigenvalues)
    assert counts.max() >= 6 and counts[counts > 1].sum() >= 600
    spec = syn.spectrum_preset("near-singular", 1000, 3)
    for t in spec.targets:
        gap = np.min(np.abs(spec.eigenvalues - t["sigma"]))
        assert gap == pytest.approx(1e-12 * abs(t["sigma"]), rel=1e-3)
    w = np.abs(syn.spectrum_preset("wide-range", 1000, 3).eigenvalues)
    assert w.min() >= 1e-6 and w.max() <= 1e6 and w.max() / w.min() > 1e10
    with pytest.raises(ValueError):
        syn.spectrum_preset("nope", 10, 0)


def test_magnitude_grouping_keeps_rows_local():
    """With grouping='magnitude' (unpermuted) row i only mixes eigenvalues l <= i + 11 (its offset
    block straddles two aligned blocks), so |a_ij| <= max_(l <= i+11) |lambda_l| by Cauchy-Schwarz:
    rounding stays relative to the local magnitude. A random grouping puts ~1e6 into every row."""
    w = 10.0 ** np.linspace(-6, 6, 400)
    bound = w[np.minimum(np.arange(400) + 11, 399)] * (1 + 1e-12)
    a = syn.prescribed_hermitian(w, block=8, seed=4, grouping="magnitude", permute=False)
    assert np.all(abs(a).max(axis=1).toarray().ravel() <= bound)
    a = syn.prescribed_hermitian(w, block=8, seed=4, grouping="random", permute=False)
    assert not np.all(abs(a).max(axis=1).toarray().ravel() <= bound)
    with pytest.raises(ValueError):
        syn.prescribed_hermitian(w, grouping="sorted")


@pytest.mark.parametrize("complex_", [False, True])
def test_congruence_pair_matches_spectrum(complex_):
    spec = syn.spectrum_preset("multiplicity", 250, seed=5)
    a, b = syn.congruence_pair(spec.eigenvalues, b_cond=1e3, complex_=complex_, seed=6)
    assert np.iscomplexobj(a.data) == complex_ == np.iscomplexobj(b.data)
    _assert_hermitian(a)
    _assert_hermitian(b)
    bd = b.toarray()
    sl.cholesky(bd)                                     # B is HPD
    assert np.linalg.cond(bd) <= 9 * 1e3
    w = sl.eigh(a.toarray(), bd, eigvals_only=True)
    assert _max_rel_err(w, spec.eigenvalues) < 1e-12
    # rows i..i+3 of M (W's band) touch at most 4 offset blocks, widened by the band of W
    assert a.getnnz(axis=1).max() <= 4 * 8 + 3


# ------------------------------------------------------------------------- case writers / CLI

def test_case_writers_roundtrip(tmp_path):
    paths = [syn.make_laplacian_case("t-lap", (6, 6), None, 0.1, root=tmp_path),
             syn.make_prescribed_case("t-pre", "near-singular", 200, True, 7, root=tmp_path),
             syn.make_pair_case("t-pair", "zero-cluster", 200, False, 8, 1e2, root=tmp_path),
             syn.make_memory_case("t-mem", (5, 5, 5), 2, 0.05, root=tmp_path)]
    for path in paths:
        c = case.load_case(path)
        ref = case.load_ref(c.name, root=tmp_path)
        assert ref.info["kind"] == "discrete-exact" and ref.info["complete"] and ref.info["window"] is None
        assert ref.eigenvalues.size == c.n == sum(c.meta["inertia"]["A"])
        assert c.config["target"] == {"mode": "nearest", "sigma": c.meta["targets"][0]["sigma"],
                                      "k": c.meta["targets"][0]["k"]}
        dense = sl.eigh(c.A.toarray(), None if c.B is None else c.B.toarray(), eigvals_only=True)
        assert _max_rel_err(dense, ref.eigenvalues) < 1e-12
    assert case.load_case(paths[1]).meta["dtype"] == "complex128"
    pair = case.load_case(paths[2])
    assert pair.meta["problem"] == "GHEP" and pair.meta["inertia"]["B"] == [0, 0, 200]
    assert "multiplicity" in case.load_case(paths[0]).meta["scenarios"]


def test_write_case_without_reference_drops_only_its_own_stale_reference(tmp_path):
    a = syn.laplacian((3, 3))
    target = [{"sigma": 0.0, "k": 2, "why": "test"}]
    meta = {"family": "synthetic", "problem": "HEP"}
    syn.write_case("t", a, None, meta=meta, targets=target, root=tmp_path,
                   eigenvalues=syn.laplacian_eigenvalues((3, 3)))
    assert case.load_ref("t", root=tmp_path).info["kind"] == "discrete-exact"
    syn.write_case("t", a, None, meta=meta, targets=target, root=tmp_path)
    assert case.load_ref("t", root=tmp_path) is None
    case.save_ref("t", [1.0], kind="solver", root=tmp_path)    # another tool's reference stays
    syn.write_case("t", a, None, meta=meta, targets=target, root=tmp_path)
    assert case.load_ref("t", root=tmp_path).info["kind"] == "solver"


def test_cli_lists_and_builds(tmp_path, capsys):
    presets = {"t-cli": lambda root: syn.make_laplacian_case("t-cli", (4, 5), None, 0.2, root=root)}
    assert syn.run_presets(presets, ["--list"], prog="p", description="d") == 0
    assert capsys.readouterr().out.split() == ["t-cli"]
    assert syn.run_presets(presets, ["all", "--root", str(tmp_path)], prog="p", description="d") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["case"] == "t-cli" and summary["n"] == 20 and summary["problem"] == "HEP"
    assert summary["case_bytes"] > 0 and summary["ref_bytes"] > 0 and summary["nnz_B"] == 0
    with pytest.raises(SystemExit):
        syn.run_presets(presets, ["missing"], prog="p", description="d")
    assert {"syn-lap3d-22", "syn-mem125-46", "syn-ghep-near-singular-real-1e5"} <= set(syn.PRESETS)
