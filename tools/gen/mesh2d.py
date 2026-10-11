"""Structured triangulations of rectangles for the 2-D waveguide / cavity generators.

Conventions (shared with tools/gen/waveguide.py)
-------------------------------------------------
* Coordinates are plain floats in the caller's length unit (the generators use millimetres).
* Triangles are counter-clockwise (positive signed area).
* Global edge orientation runs from the lower to the higher global node index.  This is the
  orientation of the Whitney edge functions and of the signed edge-node incidence matrix D
  (D[e, hi] = +1, D[e, lo] = -1), so that grad(L_j) = sum_e D[e, j] N_e holds exactly.
* Boundary edges are the edges owned by exactly one triangle; boundary nodes are their end
  points.  This topological definition is mesh-independent, so the PEC condition (drop boundary
  edges and boundary nodes) never relies on coordinate comparisons.

Mesh family
-----------
A tensor grid xs x ys (piecewise uniform, with optional breakpoints that become grid lines, e.g.
material interfaces), each cell split into two right triangles:

* ``unionjack`` (default): the diagonal alternates with the cell parity, so every diagonal joins
  two nodes with even i + j.  For a square with nx == ny even the mesh has the full symmetry
  group D4 of the square, which keeps symmetry-protected degeneracies of the continuous problem
  (e.g. TE10 / TE01) exactly degenerate in the discrete one ("multiplicity" scenario).
* ``diagonal``: all diagonals point the same way (only a C2v-like symmetry; such degeneracies split
  at O(h^2)).

Optional seeded random interior perturbation turns the grid into an unstructured-looking mesh.
Every interior node moves by at most ``perturb`` (p) times the width of its narrower adjacent grid
cell in each coordinate.  The bound MAX_PERTURB = 1/4 comes from the worst case of a right
triangle with legs hx, hy and hypotenuse L: vertex and opposite edge can approach each other along
the edge normal by at most 2 * 2p hx hy / L, which stays below the height hx hy / L for p < 1/4
(legs: 2p h < h).  The constructor re-checks every signed area and refuses an invalid mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

PATTERNS = ("unionjack", "diagonal")
MAX_PERTURB = 0.25
# Local edge k of a triangle joins local vertices LOCAL_EDGES[k] (oriented first -> second).
LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))


@dataclass(frozen=True)
class TriMesh:
    nodes: np.ndarray       # (nv, 2) float64 coordinates
    triangles: np.ndarray   # (nt, 3) int64 node indices, counter-clockwise

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class EdgeTopology:
    edges: np.ndarray          # (ne, 2) int64, edges[e] = (lo, hi) with lo < hi
    tri_edges: np.ndarray      # (nt, 3) int64 global edge of local edge k
    tri_signs: np.ndarray      # (nt, 3) float64: +1 if local orientation matches lo -> hi, else -1
    boundary_edge: np.ndarray  # (ne,) bool
    boundary_node: np.ndarray  # (nv,) bool

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])


# ------------------------------------------------------------------------------- construction

def grid_ticks(length: float, ncells: int, breakpoints: Sequence[float] = ()) -> np.ndarray:
    """``ncells + 1`` coordinates on [0, length], uniform between consecutive breakpoints.

    Breakpoints (strictly inside (0, length)) become grid lines.  Cells are distributed over the
    segments proportionally to their lengths (largest remainder, at least one per segment), so
    the cell size is as uniform as the breakpoints allow.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    cuts = sorted(float(p) for p in breakpoints)
    if any(not 0.0 < p < length for p in cuts) or len(set(cuts)) != len(cuts):
        raise ValueError(f"breakpoints must be distinct and inside (0, {length}): {breakpoints}")
    bounds = np.array([0.0, *cuts, float(length)])
    seg = np.diff(bounds)
    if ncells < seg.size:
        raise ValueError(f"need at least {seg.size} cells for breakpoints {cuts}")
    ideal = ncells * seg / length
    counts = np.maximum(1, np.floor(ideal).astype(int))
    while counts.sum() < ncells:            # largest remainder first
        counts[np.argmax(ideal - counts)] += 1
    while counts.sum() > ncells:            # only possible when the max(1, .) floor kicked in
        counts[np.argmax(np.where(counts > 1, counts - ideal, -np.inf))] -= 1
    parts = [np.linspace(bounds[i], bounds[i + 1], counts[i] + 1)[:-1] for i in range(seg.size)]
    return np.concatenate([*parts, [float(length)]])


def structured_rectangle(xs: Sequence[float], ys: Sequence[float], *, pattern: str = "unionjack",
                         perturb: float = 0.0, seed: int = 0, straight_x: Sequence[float] = (),
                         straight_y: Sequence[float] = ()) -> TriMesh:
    """Triangulate the tensor grid xs x ys (two triangles per cell, see module docstring).

    ``perturb`` in [0, 1/4) moves interior nodes randomly (``np.random.default_rng(seed)``);
    nodes on the grid columns listed in ``straight_x`` (rows in ``straight_y``) keep their x (y)
    coordinate so that material interfaces stay straight lines.  Boundary nodes never move.
    Node (i, j) (x index i, y index j) has global index j * (nx + 1) + i.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or ys.size < 2:
        raise ValueError("xs and ys need at least two coordinates each")
    if np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
        raise ValueError("grid coordinates must be strictly increasing")
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    if not 0.0 <= perturb < MAX_PERTURB:
        raise ValueError(f"perturb must lie in [0, {MAX_PERTURB}) to guarantee a valid mesh")
    nx, ny = xs.size - 1, ys.size - 1
    frozen_cols = _matching(xs, straight_x)
    frozen_rows = _matching(ys, straight_y)

    gx, gy = np.meshgrid(xs, ys)                       # (ny+1, nx+1), node (i, j) at [j, i]
    if perturb > 0.0:
        # Draw for every node so the result does not depend on which nodes are frozen.
        shake = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(2, ny + 1, nx + 1))
        gx = gx + shake[0] * _admissible_moves(xs, frozen_cols, perturb)[None, :]
        gy = gy + shake[1] * _admissible_moves(ys, frozen_rows, perturb)[:, None]
        # Boundary nodes stay put: zero the moves on the outer rows / columns in both directions.
        gx[[0, -1], :] = xs[None, :]
        gy[:, [0, -1]] = ys[:, None]
    nodes = np.column_stack([gx.ravel(), gy.ravel()])

    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    n00 = j * (nx + 1) + i
    n10, n01, n11 = n00 + 1, n00 + nx + 1, n00 + nx + 2
    main_diag = np.ones(i.size, dtype=bool) if pattern == "diagonal" else (i + j) % 2 == 0
    # Diagonal n00-n11: (n00, n10, n11), (n00, n11, n01).  Diagonal n10-n01: (n00, n10, n01),
    # (n10, n11, n01).  All four are counter-clockwise.
    t1 = np.where(main_diag[:, None], np.column_stack([n00, n10, n11]), np.column_stack([n00, n10, n01]))
    t2 = np.where(main_diag[:, None], np.column_stack([n00, n11, n01]), np.column_stack([n10, n11, n01]))
    triangles = np.concatenate([t1, t2]).astype(np.int64)
    mesh = TriMesh(nodes=nodes, triangles=triangles)
    if np.any(triangle_areas(mesh) <= 0.0):
        raise RuntimeError("triangulation produced a degenerate or inverted triangle")
    return mesh


def _matching(ticks: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Indices of grid ticks equal (to rounding) to the requested line positions."""
    idx = []
    scale = float(ticks[-1] - ticks[0])
    for v in values:
        k = int(np.argmin(np.abs(ticks - v)))
        if abs(ticks[k] - v) > 1e-12 * scale:
            raise ValueError(f"straight line at {v} is not a grid line")
        idx.append(k)
    return np.array(idx, dtype=int)


def _admissible_moves(ticks: np.ndarray, frozen: np.ndarray, perturb: float) -> np.ndarray:
    """Per-line maximal displacement normal to the grid lines: perturb * narrower adjacent cell;
    zero for the two boundary lines and for the frozen line indices (they stay straight)."""
    h = np.diff(ticks)
    move = np.zeros(ticks.size)
    move[1:-1] = perturb * np.minimum(h[:-1], h[1:])
    move[frozen] = 0.0
    return move


# ----------------------------------------------------------------------------------- topology

def edge_topology(mesh: TriMesh) -> EdgeTopology:
    """Unique edges (sorted by (lo, hi)), triangle-to-edge map with orientation signs, boundary."""
    tri = mesh.triangles
    loc_a = tri[:, [a for a, _ in LOCAL_EDGES]]
    loc_b = tri[:, [b for _, b in LOCAL_EDGES]]
    lo = np.minimum(loc_a, loc_b)
    hi = np.maximum(loc_a, loc_b)
    key = lo.ravel() * np.int64(mesh.num_nodes) + hi.ravel()
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    edges = np.column_stack([uniq // mesh.num_nodes, uniq % mesh.num_nodes]).astype(np.int64)
    boundary_edge = counts == 1
    if np.any(counts > 2):
        raise ValueError("non-manifold mesh: an edge is shared by more than two triangles")
    boundary_node = np.zeros(mesh.num_nodes, dtype=bool)
    boundary_node[edges[boundary_edge].ravel()] = True
    return EdgeTopology(edges=edges, tri_edges=inverse.reshape(tri.shape).astype(np.int64),
                        tri_signs=np.where(loc_a < loc_b, 1.0, -1.0),
                        boundary_edge=boundary_edge, boundary_node=boundary_node)


def triangle_areas(mesh: TriMesh) -> np.ndarray:
    """Signed areas (positive for counter-clockwise triangles)."""
    p = mesh.nodes[mesh.triangles]
    d1 = p[:, 1] - p[:, 0]
    d2 = p[:, 2] - p[:, 0]
    return 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])


def max_edge_length(mesh: TriMesh, topo: EdgeTopology) -> float:
    """Mesh size h = longest edge."""
    d = mesh.nodes[topo.edges[:, 1]] - mesh.nodes[topo.edges[:, 0]]
    return float(np.sqrt((d * d).sum(axis=1)).max())


def grid_interior_counts(nx: int, ny: int) -> tuple[int, int]:
    """(interior nodes, interior edges) of an nx x ny two-triangles-per-cell grid (any pattern).

    Edges: nx(ny+1) horizontal + ny(nx+1) vertical + nx*ny diagonals, minus 2(nx+ny) on the
    boundary, i.e. 3*nx*ny - nx - ny.
    """
    return (nx - 1) * (ny - 1), 3 * nx * ny - nx - ny
