#!/usr/bin/env python3
"""Geometric nested dissection of the data cases' structured grids - a reference for hsolve's
graph-only ordering (results/p9_ordering.md section 2). The solver never sees coordinates; this
script knows the generators' index layouts and cuts boxes by planes:

  lattice NX NY NZ OUT.npy [LEAF]  points of a 27-point stencil (tools/gen/synthetic.py
                                   stencil_operator, Kronecker order: index = (x*NY + y)*NZ + z);
                                   a coordinate plane of points is a separator
  yee CX CY CZ OUT.npy [LEAF]      edge unknowns of a PEC Yee grid of CX x CY x CZ cells
                                   (tools/gen/yee.py: x-, y-, z-edge blocks, each in C order over
                                   (cells along the edge, interior nodes across)); in doubled
                                   coordinates the unknowns on an even plane (the edges lying in a
                                   node plane) separate the two sides of the curl-curl graph

Each box is cut across its longest side at the (even) plane nearest its middle; halves first,
plane last; boxes of at most LEAF (default 64) unknowns stay in natural order. Writes MUMPS
PERM_IN (1-based pivot position of every unknown, int32) for
`bench_ordering_cost --ordering -1 --perm-in OUT.npy`.

Example (P9): python3 tools/bench/ordering_geometric.py yee 92 71 54 geo_yee1e6.npy
"""
from __future__ import annotations

import sys

import numpy as np


def lattice_coords(nx: int, ny: int, nz: int) -> np.ndarray:
    g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack([2 * c.ravel() for c in g], axis=1).astype(np.int32)  # even: every plane cuts


def yee_coords(cx: int, cy: int, cz: int) -> np.ndarray:
    cells = (cx, cy, cz)
    nodes = [c - 1 for c in cells]
    blocks = []
    for d in range(3):
        shape = list(nodes)
        shape[d] = cells[d]
        g = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        blocks.append(np.stack([(2 * g[e] + (1 if e == d else 2)).ravel() for e in range(3)], axis=1))
    return np.concatenate(blocks).astype(np.int32)


def dissect(x: np.ndarray, leaf: int) -> np.ndarray:
    order: list[np.ndarray] = []
    # Explicit stack: a box pushes its plane, then its right half, then its left half (popped first).
    out_stack: list[tuple] = [("box", np.arange(x.shape[0]), x.min(axis=0), x.max(axis=0) + 1)]
    while out_stack:
        item = out_stack.pop()
        if item[0] == "plane":
            order.append(item[1])
            continue
        _, idx, lo, hi = item
        if idx.size <= leaf:
            order.append(idx)
            continue
        ext = hi - lo
        d = int(np.argmax(ext))
        mid = (int(lo[d]) + int(hi[d])) // 2
        if mid % 2:
            mid += 1
        if mid <= lo[d] or mid >= hi[d]:
            order.append(idx)
            continue
        c = x[idx, d]
        left, right, sep = idx[c < mid], idx[c > mid], idx[c == mid]
        lhi = hi.copy()
        lhi[d] = mid
        rlo = lo.copy()
        rlo[d] = mid
        out_stack.append(("plane", sep))
        out_stack.append(("box", right, rlo, hi))
        out_stack.append(("box", left, lo, lhi))
    return np.concatenate(order)


def main(argv: list[str]) -> int:
    if len(argv) < 5 or argv[0] not in ("lattice", "yee"):
        print(__doc__, file=sys.stderr)
        return 2
    dims = [int(a) for a in argv[1:4]]
    out = argv[4]
    leaf = int(argv[5]) if len(argv) > 5 else 64
    x = lattice_coords(*dims) if argv[0] == "lattice" else yee_coords(*dims)
    o = dissect(x, leaf)
    n = x.shape[0]
    if o.size != n or np.unique(o).size != n:
        raise SystemExit("not a permutation")
    perm = np.empty(n, dtype=np.int32)
    perm[o] = np.arange(1, n + 1, dtype=np.int32)
    np.save(out, perm)
    print(f"wrote {out}: {n} unknowns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
