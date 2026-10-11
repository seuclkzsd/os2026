"""Tests for tools/gen/yee.py: exact discrete Yee spectra against dense eigensolvers."""
import numpy as np
import pytest
import scipy.linalg as sl

from tools import case
from tools.gen import synthetic as syn
from tools.gen import yee

GENERIC = (0.3 * np.pi, 0.2 * np.pi, 1.1 * np.pi)

GRIDS = [
    yee.YeeGrid((3, 4, 5), (1.0, 0.7, 1.3)),                   # anisotropic PEC box
    yee.YeeGrid((4, 4, 4)),                                    # PEC cube: TE/TM degeneracies
    yee.YeeGrid((3, 4, 5), (1.0, 0.7, 1.3), "bloch", GENERIC),  # complex Hermitian
    yee.YeeGrid((3, 3, 4), boundary="bloch"),                  # Gamma point: 3 constant fields
    yee.YeeGrid((3, 3, 4), boundary="bloch", phases=(np.pi, 0.0, 0.0)),  # zone edge: real
    yee.YeeGrid((1, 2, 3), boundary="bloch", phases=(0.5, 1.0, 2.0)),    # one-cell period
]


def _dense_pencil(a, b):
    return sl.eigh(a.toarray(), b.toarray(), eigvals_only=True)


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: f"{g.boundary}-{'x'.join(map(str, g.cells))}")
def test_exact_spectrum_matches_dense(grid):
    eps, mu = 2.0, 1.5
    a, b = yee.assemble(grid, eps, mu)
    n = grid.edge_count
    assert a.shape == b.shape == (n, n) and np.iscomplexobj(a.data) == grid.is_complex
    assert (a - a.conj().T).count_nonzero() == 0
    exact = yee.exact_eigenvalues(grid, eps, mu)
    assert exact.size == n
    scale = exact.max()
    assert np.max(np.abs(_dense_pencil(a, b) - exact)) < 1e-13 * scale
    assert np.max(np.abs(sl.eigvalsh(yee.standard_form(a, b).toarray()) - exact)) < 1e-13 * scale


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: f"{g.boundary}-{'x'.join(map(str, g.cells))}")
def test_null_space_and_multiplicities(grid):
    """Null space = gradients (+2 constant fields at k = 0) and the TE/TM multiplicity rules:
    level counts of the formula equal the clusters of the dense spectrum."""
    a, b = yee.assemble(grid)
    curl, grad = yee.operators(grid)
    assert (curl @ grad).count_nonzero() == 0
    dense = _dense_pencil(a, b)
    exact = yee.exact_eigenvalues(grid)
    null = grid.null_space_dimension()
    assert np.count_nonzero(np.abs(dense) < 1e-10 * dense.max()) == null == np.count_nonzero(exact == 0)
    _, dense_counts = syn.distinct_levels(np.where(np.abs(dense) < 1e-10 * dense.max(), 0, dense), rtol=1e-9)
    _, exact_counts = syn.distinct_levels(exact, rtol=1e-9)
    assert dense_counts.tolist() == exact_counts.tolist()


def test_counts_and_pec_mode_rules():
    g = yee.YeeGrid((3, 4, 5))
    assert g.edge_counts == (3 * 3 * 4, 2 * 4 * 4, 2 * 3 * 5) and g.null_space_dimension() == 2 * 3 * 4
    assert yee.YeeGrid((3, 4, 5), boundary="bloch").edge_count == 3 * 60
    # cube: lowest level = indices (1,1,0) and permutations, one polarization each -> 3;
    # next level (1,1,1): two polarizations -> 2
    _, counts = syn.distinct_levels(yee.exact_eigenvalues(yee.YeeGrid((4, 4, 4))))
    assert counts[1:3].tolist() == [3, 2]
    s1 = (2 * 4 * np.sin(np.pi / 8)) ** 2
    assert yee.exact_eigenvalues(yee.YeeGrid((4, 4, 4)))[27] == pytest.approx(2 * s1)


def test_bitwise_degeneracies_in_cube():
    """Permuted mode indices of a cube give bitwise-equal eigenvalues: each triple of three
    distinct nonzero indices (6 permutations x 2 polarizations) yields a value occurring >= 12
    times (>= because s_m + s_(N-m) is constant, which merges some levels mathematically)."""
    _, bitwise = np.unique(yee.exact_eigenvalues(yee.YeeGrid((6, 6, 6))), return_counts=True)
    assert bitwise[bitwise >= 12].sum() >= 5 * 4 * 3 * 2


def test_material_averaging():
    g = yee.YeeGrid((3, 4, 5), boundary="bloch", phases=GENERIC)
    a1, b1 = yee.assemble(g, 2.5, 1.25)
    a2, b2 = yee.assemble(g, np.full(g.cells, 2.5), np.full(g.cells, 1.25))
    assert abs(a1 - a2).max() < 1e-14 * abs(a1).max() and abs(b1 - b2).max() < 1e-14 * abs(b1).max()
    eps = yee.photonic_permittivity(g, "spheres", 0.25, 13.0)
    _, b = yee.assemble(g, eps)
    assert b.diagonal().min() > 0
    with pytest.raises(ValueError):
        yee.assemble(g, np.ones((3, 4, 4)))
    with pytest.raises(ValueError):
        yee.assemble(g, -1.0)
    with pytest.raises(ValueError):
        yee.exact_eigenvalues(g, eps)


@pytest.mark.parametrize("shape", ["spheres", "rods"])
def test_photonic_crystal(shape):
    g = yee.YeeGrid((6, 6, 6), boundary="bloch", phases=(0.4 * np.pi, 0.25 * np.pi, 0.1 * np.pi))
    eps = yee.photonic_permittivity(g, shape, 0.25, 13.0)
    assert eps.shape == g.cells and eps.min() == 1.0 and eps.max() == 13.0
    assert np.allclose(eps, eps[::-1, ::-1, ::-1])           # inclusion centred in the cell
    if shape == "rods":
        assert np.all(eps == eps[:, :, :1])                  # invariant along z
    a, b = yee.assemble(g, eps)
    w = _dense_pencil(a, b)
    assert np.count_nonzero(np.abs(w) < 1e-10 * w.max()) == g.null_space_dimension()
    assert w.min() > -1e-10 * w.max()                        # A PSD, B HPD
    with pytest.raises(ValueError):
        yee.photonic_permittivity(g, "cubes", 0.25, 13.0)


def test_grid_validation():
    with pytest.raises(ValueError):
        yee.YeeGrid((1, 4, 4))                               # PEC needs interior nodes
    with pytest.raises(ValueError):
        yee.YeeGrid((4, 4, 4), phases=(0.1, 0.0, 0.0))       # phases need Bloch boundary
    with pytest.raises(ValueError):
        yee.YeeGrid((4, 4, 4), boundary="pmc")
    assert yee.YeeGrid((2, 2, 2), boundary="bloch", phases=(2 * np.pi, 0, 0)).phases == (0.0, 0.0, 0.0)


def test_case_writers(tmp_path):
    g = yee.YeeGrid((4, 3, 5), (1.0, 0.8, 1.2))
    for form, problem in (("standard", "HEP"), ("generalized", "GHEP")):
        name = f"t-yee-{form}"
        c = case.load_case(yee.make_homogeneous_case(name, g, form, tmp_path))
        ref = case.load_ref(name, root=tmp_path)
        assert c.meta["problem"] == problem and (c.B is not None) == (form == "generalized")
        assert ref.info["kind"] == "discrete-exact" and ref.info["complete"] and ref.eigenvalues.size == c.n
        assert c.meta["inertia"]["A"] == [0, g.null_space_dimension(), c.n - g.null_space_dimension()]
        assert {"zero-cluster", "multiplicity"} <= set(c.meta["scenarios"])
        assert c.config["target"]["sigma"] == c.meta["targets"][0]["sigma"]
        dense = sl.eigh(c.A.toarray(), None if c.B is None else c.B.toarray(), eigvals_only=True)
        assert np.max(np.abs(dense - ref.eigenvalues)) < 1e-12 * ref.eigenvalues.max()
    gp = yee.YeeGrid((6, 6, 6), boundary="bloch", phases=(0.4 * np.pi, 0.25 * np.pi, 0.1 * np.pi))
    case.save_ref("t-phc", [1.0], kind="dense", root=tmp_path)    # stale reference must disappear
    c = case.load_case(yee.make_photonic_case("t-phc", gp, "generalized", "spheres", 0.25, 13.0, tmp_path))
    assert case.load_ref("t-phc", root=tmp_path) is None
    assert c.meta["dtype"] == "complex128" and c.meta["scenarios"] == ["zero-cluster", "complex-hermitian"]
    assert c.meta["targets"][0]["sigma"] > 0 and c.meta["inertia"]["B"] == [0, 0, c.n]
    assert set(yee.PRESETS) >= {"yee-pec-cube-16", "yee-pec-box-1e5", "yee-bloch-1e4",
                                "yee-photonic-spheres-1e4"}
