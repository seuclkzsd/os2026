"""Tests for tools/gen/waveguide.py: FEM identities, inertia structure, real spectrum, O(h^2)
convergence to the analytic modes, the a-priori error bound, and the case generator / CLI.

Dense eigen-solves only for n <= 2000; larger checks use sparse shift-invert (scipy splu + eigs).
"""
import shlex

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from tools import case
from tools.gen import mesh2d
from tools.gen import waveguide as wg

K0 = wg.wavenumber(50.0)          # 10 mm square guide at 50 GHz: the competition's real case
A_MM = B_MM = 10.0


def _disc(nx, *, eps=1.0, family="waveguide", **kw):
    spec = wg.CaseSpec(family=family, nx=nx, eps_slab=eps, **kw)
    mesh, eps_r, mu_r = wg.build_mesh(spec, *wg.resolve_grid(spec))
    return wg.discretize(mesh), eps_r, mu_r


def _nearest(A, B, sigma, k):
    """k eigenvalues of A x = lambda B x nearest to sigma (OP = (A - sigma B)^{-1} B, Euclidean
    Arnoldi, valid for indefinite B), sorted ascending by real part."""
    lu = spla.splu((A - sigma * B).tocsc())
    op = spla.LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x), dtype=float)
    theta = spla.eigs(op, k=k, which="LM", tol=1e-13, return_eigenvectors=False)
    lam = sigma + 1.0 / theta
    return lam[np.argsort(lam.real)]


def _real(lam, scale=1.0):
    assert np.abs(lam.imag).max() <= 1e-9 * scale
    return lam.real


# ----------------------------------------------------------------------------- FEM identities

def test_discrete_gradient_identities_with_weights():
    """G = T_tt D (Whitney forms contain the gradients of P1), S_tt D = 0, D^T T_tt D = S_zz,
    for arbitrary per-triangle weights on a perturbed mesh."""
    disc, _, _ = _disc(7, ny=5, perturb=0.2, seed=4, pattern="diagonal")
    w = np.random.default_rng(0).uniform(0.5, 3.0, disc.mesh.num_triangles)
    T, G, D = wg.mass_tt(disc, w), wg.grad_tz(disc, w), wg.incidence(disc)
    assert abs(G - T @ D).max() <= 1e-12 * abs(G).max()
    assert abs(wg.curl_tt(disc, w) @ D).max() <= 1e-12 * abs(wg.curl_tt(disc, w)).max()
    assert abs(D.T @ T @ D - wg.stiff_zz(disc, w)).max() <= 1e-12 * abs(wg.stiff_zz(disc, w)).max()
    # Each interior edge touches at most two interior nodes, with signs +1 (head) / -1 (tail).
    assert set(np.unique(D.data)) <= {-1.0, 1.0} and np.all(np.diff(D.indptr) <= 2)


def test_element_integrals_on_reference_triangle():
    """Mass/stiffness of the unit right triangle against hand-computed values."""
    mesh = mesh2d.TriMesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), triangles=np.array([[0, 1, 2]]))
    el = wg.element_matrices(mesh, mesh2d.edge_topology(mesh))
    assert np.allclose(el.nmass[0], np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 24.0)
    assert np.allclose(el.stiff[0], 0.5 * np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]))
    # Edges (0,1), (1,2), (2,0); edge (2,0) is stored as 0 -> 2 (sign -1).  curl N_ij = 1 / area
    # for counter-clockwise (i, j), so int curl N_k curl N_l = s_k s_l / area = 2 s_k s_l.
    s = np.array([1.0, 1.0, -1.0])
    assert np.allclose(el.curl[0], 2.0 * np.outer(s, s))
    # int N_01 . N_01 = 1/3 (classical value for the unit right triangle's short edge).
    assert np.isclose(el.mass[0, 0, 0], 1.0 / 3.0)
    assert np.allclose(el.mass[0], el.mass[0].T) and np.all(np.linalg.eigvalsh(el.mass[0]) > 0)


@pytest.mark.parametrize("eps", [1.0, 2.25])
def test_pencil_exactly_symmetric_and_inertia_of_B_and_null_of_A(eps):
    """B is congruent to blkdiag(T_tt, -k0^2 T_zz): inertia (#nodes neg, 0, #edges pos) on any
    mesh; A has exactly #interior-nodes zero eigenvalues."""
    disc, eps_r, mu_r = _disc(8, ny=6, eps=eps, perturb=0.2, seed=1)
    A, B = wg.waveguide_pencil(disc, K0, eps_r, mu_r)
    ne, nn = disc.num_edge_dofs, disc.num_node_dofs
    assert (A - A.T).nnz == 0 and (B - B.T).nnz == 0
    assert wg.dense_inertia(B) == [nn, 0, ne]
    assert wg.dense_inertia(A)[1] == nn
    assert A[ne:, :].nnz == 0                       # the e_z block of A is structurally zero


def test_coarse_mesh_reproduces_slide_case_structure():
    """9 x 9 cells (n = 289 vs 293 in the slide): A = (#nodes + 12 TE below cutoff) / #nodes / rest,
    B = #nodes / 0 / #edges, pencil = 18 / #nodes / #edges - 18, all eigenvalues real; slide:
    A 78-/66 0/149+, B 66-/227+, pencil 18-/66 0/209+."""
    disc, _, _ = _disc(9)
    ne, nn = disc.num_edge_dofs, disc.num_node_dofs
    A, B = wg.waveguide_pencil(disc, K0)
    assert (ne, nn) == (225, 64)
    assert wg.dense_inertia(A) == [nn + 12, nn, ne - nn - 12]
    assert wg.dense_inertia(B) == [nn, 0, ne]
    lam = wg.dense_pencil_eigenvalues(A, B)
    scale = np.abs(lam).max()
    lam = _real(lam, scale)
    assert wg.count_signs(lam, 1e-8 * scale) == [18, nn, ne - 18]
    assert -1.1 < lam.min() and lam[lam < 0].max() > -0.12      # slide: negatives in about [-1, -0.01]


@pytest.mark.parametrize("nx,perturb,pattern", [(16, 0.0, "unionjack"), (11, 0.2, "diagonal")])
def test_hollow_guide_spectrum_real_with_18_propagating_modes(nx, perturb, pattern):
    disc, _, _ = _disc(nx, perturb=perturb, seed=2, pattern=pattern)
    A, B = wg.waveguide_pencil(disc, K0)
    lam = wg.dense_pencil_eigenvalues(A, B)
    scale = np.abs(lam).max()
    assert np.abs(lam.imag).max() <= 1e-9 * scale              # dense QZ: the whole spectrum is real
    assert wg.count_signs(lam.real, 1e-8 * scale) == [18, disc.num_node_dofs, disc.num_edge_dofs - 18]


def test_slab_guide_complex_pairs_only_among_evanescent_modes():
    """Indefinite B: LSE/LSM branches of opposite sign characteristic collide into complex pairs,
    but only above the zero cluster; the propagating modes stay real with positive B-type."""
    disc, eps_r, mu_r = _disc(12, eps=2.25)
    A, B = wg.waveguide_pencil(disc, K0, eps_r, mu_r)
    lam, vec = sla.eig(A.toarray(), B.toarray())
    tol = 1e-8 * np.abs(lam).max()
    cplx = np.abs(lam.imag) > tol
    assert cplx.any() and lam.real[cplx].min() > 0.1
    neg = ~cplx & (lam.real < -tol)
    btype = np.real(np.einsum("ij,ij->j", vec.conj(), B.toarray() @ vec))
    assert np.all(btype[neg] > 0)


def test_unionjack_keeps_symmetry_protected_degeneracy():
    """TE1,0 / TE0,1 are exactly degenerate on the D4-symmetric union-jack mesh and split on the
    'diagonal' mesh ('multiplicity' versus 'clustered' scenarios)."""
    for pattern, degenerate in (("unionjack", True), ("diagonal", False)):
        disc, _, _ = _disc(8, family="cavity2d", pattern=pattern)
        S, T = wg.cavity_pencil(disc)
        kc2 = sla.eigh(S.toarray(), T.toarray(), eigvals_only=True)
        first = kc2[kc2 > 1e-8 * kc2.max()][:2]
        split = (first[1] - first[0]) / first[0]
        assert (split < 1e-10) if degenerate else (split > 1e-6)


def test_cavity_null_space_dimension():
    disc, _, _ = _disc(9, ny=7, family="cavity2d", perturb=0.15, seed=3)
    S, T = wg.cavity_pencil(disc)
    kc2 = sla.eigh(S.toarray(), T.toarray(), eigvals_only=True)
    scale = kc2.max()
    assert wg.count_signs(kc2, 1e-9 * scale) == [0, disc.num_node_dofs, disc.num_edge_dofs - disc.num_node_dofs]


# ------------------------------------------------------------------- convergence / error bound

def _max_errors(family, levels, **kw):
    """Max |lambda_h - lambda| over a block of low modes on successively refined meshes, together
    with the a-priori bound; the block is chosen away from the zero cluster."""
    if family == "hollow":
        exact = np.array([m.value for m in wg.hollow_guide_modes(A_MM, B_MM, K0, 0.0)])      # 18 modes
        shift = K0 * K0
    elif family == "slab":
        exact = np.array([m.value for m in wg.slab_guide_modes(A_MM, B_MM, K0, 5.0, 2.25, 1.0, -0.15)])
        shift = 2.25 * K0 * K0
    else:
        exact = np.array([m.value for m in wg.rectangle_cutoffs(A_MM, B_MM, 1.6, tm=False)])
        shift = 0.0
    errors, bounds = [], []
    for nx in levels:
        if family == "cavity":
            disc, _, _ = _disc(nx, family="cavity2d", **kw)
            A, B = wg.cavity_pencil(disc)
        else:
            disc, eps_r, mu_r = _disc(nx, eps=2.25 if family == "slab" else 1.0, **kw)
            A, B = wg.waveguide_pencil(disc, K0, eps_r, mu_r)
        sigma = 0.5 * (exact[0] + exact[-1])
        lam = _real(_nearest(A, B, sigma, exact.size + 4))
        lam = lam[np.abs(lam) > 1e-8][: exact.size]      # drop zero-cluster members, keep the block
        h = mesh2d.max_edge_length(disc.mesh, disc.topo)
        errors.append(np.abs(lam - exact).max())
        bounds.append(np.min(wg.error_bound(exact, shift, h) - np.abs(lam - exact)))
    return np.array(errors), np.array(bounds)


@pytest.mark.parametrize("family", ["hollow", "slab", "cavity"])
def test_convergence_is_second_order(family):
    """-lambda -> beta^2 (waveguide, hollow and slab-loaded) and kc^2 (cavity) at O(h^2)."""
    err, slack = _max_errors(family, (16, 32, 64))
    ratios = err[:-1] / err[1:]
    assert np.all(err[1:] < err[:-1])
    assert 3.5 < ratios[-1] < 4.5, ratios
    assert np.all(slack > 0)                      # the a-priori bound holds on every level


@pytest.mark.parametrize("family", ["hollow", "slab", "cavity"])
@pytest.mark.parametrize("pattern,perturb", [("diagonal", 0.0), ("unionjack", 0.24)])
def test_error_bound_holds_on_other_meshes(family, pattern, perturb):
    _, slack = _max_errors(family, (16, 32), pattern=pattern, perturb=perturb, seed=5)
    assert np.all(slack > 0)


# --------------------------------------------------------------------------- analytic spectra

def test_rectangle_cutoffs_and_hollow_modes():
    te = wg.rectangle_cutoffs(10.0, 5.0, (np.pi / 5.0) ** 2 * 1.01, tm=False)
    assert [m.label for m in te] == ["TE1,0", "TE0,1", "TE2,0"]          # TE0,1 and TE2,0 degenerate
    modes = wg.hollow_guide_modes(A_MM, B_MM, K0, 0.0)
    assert sum(m.label.startswith("TE") for m in modes) == 12 and len(modes) == 18
    # cutoff f_mn = 15 sqrt(m^2 + n^2) GHz: TE1,3 propagates (47.4 GHz), TE2,3 does not (54.1 GHz).
    assert {"TE1,3", "TM3,1"} <= {m.label for m in modes} and "TE2,3" not in {m.label for m in modes}


def test_slab_modes_reduce_to_hollow_and_scan_misses_no_root():
    hollow = wg.hollow_guide_modes(A_MM, B_MM, K0, 1.0)
    empty = wg.slab_guide_modes(A_MM, B_MM, K0, 5.0, 1.0, 1.0, 1.0)
    assert np.allclose([m.value for m in empty], [m.value for m in hollow], atol=1e-12)
    for eps, mu, width in ((2.25, 1.0, 5.0), (4.0, 2.0, 3.0), (2.25, 1.0, 10.0)):
        coarse = {m.label: m.value for m in wg.slab_guide_modes(A_MM, B_MM, K0, width, eps, mu, 1.0)}
        fine = {m.label: m.value for m in wg.slab_guide_modes(A_MM, B_MM, K0, width, eps, mu, 1.0, refine=4)}
        assert coarse.keys() == fine.keys()
        assert np.allclose([coarse[k] for k in coarse], [fine[k] for k in coarse], atol=1e-12)
        assert min(coarse.values()) > -K0 * K0 * eps * mu                 # Rayleigh bound


def test_fully_filled_guide_matches_scaled_hollow_guide():
    """slab_width = a: homogeneous eps, so beta^2 = eps k0^2 - kc^2."""
    filled = wg.slab_guide_modes(A_MM, B_MM, K0, A_MM, 2.25, 1.0, 0.0)
    kc2 = wg.rectangle_cutoffs(A_MM, B_MM, 2.25 * K0 * K0, tm=True)
    assert np.allclose(sorted(m.value for m in filled), sorted(m.value - 2.25 * K0 * K0 for m in kc2))


# ------------------------------------------------------------------------------- generator

def test_resolve_grid_matches_target_n():
    for family, target in (("waveguide", 1000), ("waveguide", 10_000), ("cavity2d", 10_000)):
        nx, ny = wg.resolve_grid(wg.CaseSpec(family=family, target_n=target))
        assert nx == ny and abs(wg.dof_count(family, nx, ny) - target) / target < 0.05
    nx, ny = wg.resolve_grid(wg.CaseSpec(target_n=5000, a=20.0, b=10.0))
    assert nx == 2 * ny
    assert wg.resolve_grid(wg.CaseSpec(nx=12)) == (12, 12)
    assert wg.dof_count("waveguide", 9, 9) == 289 == (2 * 9 - 1) ** 2


@pytest.mark.parametrize("kw", [dict(nx=4, target_n=100), dict(ny=4, target_n=100), dict(nx=4, pattern="x"),
                                dict(family="cavity2d", nx=4, eps_slab=2.0), dict(nx=4, freq_ghz=-1.0)])
def test_case_spec_rejects_bad_input(kw):
    with pytest.raises(ValueError):
        wg.CaseSpec(**kw)


def _check_targets_against_discrete(gen, A, B):
    """Each suggested 'nearest k' query selects exactly a block: the k-th and (k+1)-th distances
    to sigma differ, and the propagating target selects only negative eigenvalues."""
    for t in gen.meta["targets"]:
        lam = _nearest(A, B, t["sigma"], t["k"] + 4)
        dist = np.sort(np.abs(lam - t["sigma"]))
        assert dist[t["k"] - 1] < dist[t["k"]] - 1e-9
        if "propagating" in t["why"]:
            nearest = lam[np.argsort(np.abs(lam - t["sigma"]))][: t["k"]]
            assert np.all(nearest.real < 0)


def test_generate_model_path_window_and_targets_hold():
    """n > DENSE_LIMIT: inertia, window and targets come from the analytic model + error bound;
    verify them on the discrete problem with sparse shift-invert."""
    gen = wg.generate(wg.CaseSpec(nx=24), "wg_model")
    A, B = gen.A, gen.B
    nn = gen.meta["mesh"]["interior_nodes"]
    assert A.shape[0] > wg.DENSE_LIMIT and gen.meta["problem"] == "GHIEP"
    assert gen.meta["inertia"]["pencil"] == [18, nn, A.shape[0] - nn - 18]
    lo, hi = gen.ref["window"]
    listed = gen.ref["eigenvalues"]
    inside = int(((listed > lo) & (listed < hi)).sum())
    lam = _real(_nearest(A, B, 0.5 * (lo + hi), inside + 6))
    assert int(((lam > lo) & (lam < hi)).sum()) == inside
    _check_targets_against_discrete(gen, A, B)

    cav = wg.generate(wg.CaseSpec(family="cavity2d", nx=27), "cav_model")
    assert cav.A.shape[0] > wg.DENSE_LIMIT and cav.meta["problem"] == "GHEP"
    _check_targets_against_discrete(cav, cav.A, cav.B)
    lo, hi = cav.ref["window"]
    listed = cav.ref["eigenvalues"]
    inside = int(((listed > lo) & (listed < hi)).sum())
    lam = _real(_nearest(cav.A, cav.B, 0.5 * (lo + hi), inside + 6))
    assert int(((lam > lo) & (lam < hi)).sum()) == inside


def test_generate_dense_path_records_exact_counts():
    gen = wg.generate(wg.CaseSpec(nx=8, eps_slab=2.25), "slab_dense")
    n = gen.A.shape[0]
    inertia = gen.meta["inertia"]
    assert inertia["B"] == [49, 0, 176] and inertia["A"][1] == 49
    assert sum(inertia["pencil"]) + gen.meta["complex_eigenvalue_count"] == n
    assert gen.meta["complex_eigenvalue_count"] > 0
    assert len(gen.meta["targets"]) == 1 and "propagating" in gen.meta["targets"][0]["why"]
    _check_targets_against_discrete(gen, gen.A, gen.B)


def test_cli_writes_case_and_command_regenerates_it(tmp_path):
    root = tmp_path / "data"
    assert wg.main(["--family", "waveguide", "--nx", "9", "--perturb", "0.1", "--seed", "7",
                    "--name", "wg_cli", "--root", str(root)]) == 0
    c = case.load_case("wg_cli", root=root)
    ref = case.load_ref("wg_cli", root=root)
    meta = c.meta
    assert meta["n"] == 289 and meta["family"] == "waveguide" and meta["problem"] == "GHIEP"
    assert {"indefinite-B", "zero-cluster"} <= set(meta["scenarios"])
    assert meta["inertia"]["pencil"][1] == 64 and meta["complex_eigenvalue_count"] == 0
    assert c.config["target"] == {"mode": "nearest", "sigma": meta["targets"][0]["sigma"],
                                  "k": meta["targets"][0]["k"]}
    assert ref.info["kind"] == "partial-analytic" and "O(h^2)" in ref.info["note"]
    assert np.all(np.diff(ref.eigenvalues) >= 0) and ref.info["count"] == ref.eigenvalues.size

    args = shlex.split(meta["params"]["command"])[3:]          # drop "python -m tools.gen.waveguide"
    assert wg.main(args + ["--root", str(tmp_path / "again")]) == 0
    again = case.load_case("wg_cli", root=tmp_path / "again")
    assert (again.A != c.A).nnz == 0 and (again.B != c.B).nnz == 0

    assert wg.main(["--family", "cavity2d", "--target-n", "300", "--name", "cav_cli", "--root", str(root)]) == 0
    cav = case.load_case("cav_cli", root=root)
    assert cav.meta["problem"] == "GHEP" and cav.meta["inertia"]["B"] == [0, 0, cav.meta["n"]]
    with pytest.raises(SystemExit):
        wg.main(["--nx", "4", "--target-n", "100", "--name", "x", "--root", str(root)])
