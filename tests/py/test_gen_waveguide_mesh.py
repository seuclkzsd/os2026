"""Tests for tools/gen/mesh2d.py (structured rectangle triangulations and edge topology)."""
import numpy as np
import pytest

from tools.gen import mesh2d


def _triangle_set(mesh, coords):
    """Triangles as frozensets of rounded vertex coordinates (orientation-free comparison)."""
    key = np.round(coords, 9)
    return {frozenset(map(tuple, key[t])) for t in mesh.triangles}


def test_grid_ticks_uniform_segments_with_breakpoints():
    t = mesh2d.grid_ticks(10.0, 7, [3.0])
    assert t.size == 8 and t[0] == 0.0 and t[-1] == 10.0 and 3.0 in t
    lo, hi = t[t <= 3.0], t[t >= 3.0]
    assert np.allclose(np.diff(lo), np.diff(lo)[0]) and np.allclose(np.diff(hi), np.diff(hi)[0])
    assert lo.size - 1 == 2 and hi.size - 1 == 5           # 7 cells split 3 : 7 -> 2 + 5
    assert np.allclose(mesh2d.grid_ticks(4.0, 4), [0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        mesh2d.grid_ticks(10.0, 1, [5.0])                  # fewer cells than segments
    with pytest.raises(ValueError):
        mesh2d.grid_ticks(10.0, 4, [10.0])                 # breakpoint on the boundary


@pytest.mark.parametrize("pattern", mesh2d.PATTERNS)
@pytest.mark.parametrize("nx,ny", [(2, 2), (3, 5), (8, 8)])
def test_topology_counts_and_orientation(pattern, nx, ny):
    mesh = mesh2d.structured_rectangle(mesh2d.grid_ticks(10.0, nx), mesh2d.grid_ticks(6.0, ny), pattern=pattern)
    topo = mesh2d.edge_topology(mesh)
    assert mesh.num_triangles == 2 * nx * ny
    assert np.all(mesh2d.triangle_areas(mesh) > 0)
    assert np.isclose(mesh2d.triangle_areas(mesh).sum(), 60.0)
    nodes_in, edges_in = mesh2d.grid_interior_counts(nx, ny)
    assert int((~topo.boundary_node).sum()) == nodes_in
    assert int((~topo.boundary_edge).sum()) == edges_in
    assert int(topo.boundary_edge.sum()) == 2 * (nx + ny)
    # Euler characteristic of a disc: V - E + F = 1.
    assert mesh.num_nodes - topo.num_edges + mesh.num_triangles == 1
    # Global orientation lo -> hi and the per-triangle signs agree with the local edge orientation.
    assert np.all(topo.edges[:, 0] < topo.edges[:, 1])
    for k, (a, b) in enumerate(mesh2d.LOCAL_EDGES):
        ta, tb = mesh.triangles[:, a], mesh.triangles[:, b]
        e = topo.edges[topo.tri_edges[:, k]]
        assert np.array_equal(e[:, 0], np.minimum(ta, tb)) and np.array_equal(e[:, 1], np.maximum(ta, tb))
        assert np.array_equal(topo.tri_signs[:, k], np.where(ta < tb, 1.0, -1.0))


def test_perturbation_is_seeded_valid_and_respects_constraints():
    xs = mesh2d.grid_ticks(10.0, 12, [4.0])
    ys = mesh2d.grid_ticks(8.0, 9)
    base = mesh2d.structured_rectangle(xs, ys)
    for seed in range(5):
        m = mesh2d.structured_rectangle(xs, ys, perturb=0.249, seed=seed, straight_x=[4.0])
        assert np.all(mesh2d.triangle_areas(m) > 0)
        topo = mesh2d.edge_topology(m)
        assert np.array_equal(m.nodes[topo.boundary_node], base.nodes[topo.boundary_node])
        on_line = np.isclose(base.nodes[:, 0], 4.0)
        assert np.array_equal(m.nodes[on_line, 0], base.nodes[on_line, 0])      # interface stays straight
        assert not np.array_equal(m.nodes[on_line, 1], base.nodes[on_line, 1])  # but nodes slide along it
    a = mesh2d.structured_rectangle(xs, ys, perturb=0.2, seed=3)
    b = mesh2d.structured_rectangle(xs, ys, perturb=0.2, seed=3)
    c = mesh2d.structured_rectangle(xs, ys, perturb=0.2, seed=4)
    assert np.array_equal(a.nodes, b.nodes) and not np.array_equal(a.nodes, c.nodes)
    with pytest.raises(ValueError):
        mesh2d.structured_rectangle(xs, ys, perturb=0.25)
    with pytest.raises(ValueError):
        mesh2d.structured_rectangle(xs, ys, straight_x=[4.5])                    # not a grid line


def test_unionjack_has_square_symmetry_for_even_n():
    """Even union-jack grids of a square are invariant under the 90 degree rotation (group D4),
    odd ones and the 'diagonal' pattern are not."""
    def rotation_invariant(n, pattern):
        mesh = mesh2d.structured_rectangle(mesh2d.grid_ticks(1.0, n), mesh2d.grid_ticks(1.0, n), pattern=pattern)
        rotated = np.column_stack([1.0 - mesh.nodes[:, 1], mesh.nodes[:, 0]])
        return _triangle_set(mesh, mesh.nodes) == _triangle_set(mesh, rotated)

    assert rotation_invariant(4, "unionjack") and rotation_invariant(8, "unionjack")
    assert not rotation_invariant(5, "unionjack")
    assert not rotation_invariant(4, "diagonal")


def test_max_edge_length():
    mesh = mesh2d.structured_rectangle(mesh2d.grid_ticks(3.0, 3), mesh2d.grid_ticks(4.0, 2))
    assert np.isclose(mesh2d.max_edge_length(mesh, mesh2d.edge_topology(mesh)), np.hypot(1.0, 2.0))
