"""3-D Yee-grid curl-curl eigenproblems: PEC cavities, Bloch-periodic cells, photonic crystals.

Discretization: the Yee scheme in matrix form (Weiland's finite integration technique, FIT) on a
box of Nx x Ny x Nz cells of size hx x hy x hz.
  * Unknowns u_e: the line integral of E along each primal edge (u_e = h_e E_e). PEC walls remove
    the edges lying in them (tangential E = 0); a Bloch-periodic box keeps all 3 Nx Ny Nz edges.
  * C (faces x edges) is the topological curl, (C u)_f = circulation of u around face f, entries
    +-1 times the Bloch phase exp(i k_d L_d) for edges reached by wrapping around the box.
    G (edges x nodes) is the topological gradient and C G = 0 exactly (also in floating point:
    the two products cancelling in each entry are the same numbers).
  * Hodge (material) matrices are diagonal: M_eps = diag(eps_e |e~| / |e|) and
    M_nu = diag(nu_f |f~| / |f|), nu = 1/mu, |e~| the dual facet area crossed by edge e, |f~| the
    dual edge length through face f.
  A = C^H M_nu C (Hermitian PSD, null space = range(G)), B = M_eps (diagonal HPD): a GHEP whose
  eigenvalues are omega^2. The standard form (HEP) is B^{-1/2} A B^{-1/2}.

  In field unknowns E_e = u_e / h_e the same pencil reads A_E = C_E^T diag(V_f / mu) C_E,
  B_E = diag(eps V_e) (V = dual volume, C_E = difference quotients); the forms are congruent via
  D = diag(h_e) and share eigenvalues. The integral form is used because its B is not a multiple
  of I for homogeneous media once hx, hy, hz differ, i.e. the GHEP cases have a non-trivial B.

Material averaging (standard FIT): eps on an edge = arithmetic mean over the 4 cells around it
(parallel combination across the dual facet); 1/mu on a face = arithmetic mean over the 2 cells
its dual edge crosses (series combination along the dual edge).

Exact discrete spectra for homogeneous eps, mu (lambda = omega^2):
  PEC box  : (1/(eps mu)) sum_d (2/h_d sin(m_d pi / (2 N_d)))^2, m_d = 0..N_d-1: two modes (TE/TM)
             when all three m_d >= 1, one when exactly one m_d = 0, none when two vanish; plus the
             gradient null space of dimension (Nx-1)(Ny-1)(Nz-1).
  Bloch box: (1/(eps mu)) sum_d (2/h_d sin(q_d h_d / 2))^2, q_d = k_d + 2 pi m_d / L_d,
             m_d = 0..N_d-1, twice per wavevector, plus one zero per wavevector (its gradient); at
             k = 0 the wavevector m = 0 gives three zeros instead (constant fields).
Both are checked against dense eigensolvers on small grids in tests/py/test_gen_yee.py.

Usage (from the repository root):
    python -m tools.gen.yee --list
    python -m tools.gen.yee yee-pec-cube-16 yee-bloch-1e4 [--root DIR]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp

from tools.gen.synthetic import (distinct_levels, hermitian_part, run_presets, target_between_levels,
                                 write_case)

FAMILY = "yee"
BOUNDARIES = ("pec", "bloch")
FORMS = ("standard", "generalized")


@dataclass(frozen=True)
class YeeGrid:
    """Box of cells[d] cells of total length lengths[d]; boundary 'pec' or 'bloch'.

    phases[d] = k_d L_d is the Bloch phase accumulated over one period (radians, reduced modulo
    2 pi); only meaningful for 'bloch'. Unknown nodes per dimension: N_d - 1 interior nodes (PEC)
    or N_d nodes (periodic).
    """
    cells: tuple[int, int, int]
    lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    boundary: str = "pec"
    phases: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if len(self.cells) != 3 or len(self.lengths) != 3 or len(self.phases) != 3:
            raise ValueError("cells, lengths and phases need 3 entries")
        if min(self.cells) < (2 if self.boundary == "pec" else 1) or min(self.lengths) <= 0:
            raise ValueError("need >= 2 cells per dimension for PEC (>= 1 periodic) and lengths > 0")
        if self.boundary == "pec" and any(p != 0.0 for p in self.phases):
            raise ValueError("Bloch phases require boundary='bloch'")
        object.__setattr__(self, "cells", tuple(int(c) for c in self.cells))
        object.__setattr__(self, "lengths", tuple(float(x) for x in self.lengths))
        object.__setattr__(self, "phases", tuple(_reduce_phase(p) for p in self.phases))

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(L / N for L, N in zip(self.lengths, self.cells))

    def nodes(self, d: int) -> int:
        return self.cells[d] - 1 if self.boundary == "pec" else self.cells[d]

    @property
    def node_count(self) -> int:
        return self.nodes(0) * self.nodes(1) * self.nodes(2)

    @property
    def edge_counts(self) -> tuple[int, int, int]:
        """Unknown x-, y-, z-edges (cells along their own direction, nodes across)."""
        n = [self.nodes(d) for d in range(3)]
        c = self.cells
        return (c[0] * n[1] * n[2], n[0] * c[1] * n[2], n[0] * n[1] * c[2])

    @property
    def edge_count(self) -> int:
        return sum(self.edge_counts)

    @property
    def is_complex(self) -> bool:
        return any(np.iscomplexobj(_wrap_factor(p)) for p in self.phases)

    def null_space_dimension(self) -> int:
        """dim ker(A) = rank(G) + harmonic fields: interior nodes (PEC); all nodes, plus the two
        extra constant fields at k = 0, for a periodic box (ker G = constants only when k = 0)."""
        if self.boundary == "pec":
            return self.node_count
        return self.node_count + (2 if all(p == 0.0 for p in self.phases) else 0)


def _reduce_phase(phase: float) -> float:
    """phase modulo 2 pi in [0, 2 pi).

    np.mod rounds a tiny negative phase (e.g. -1e-17 on a k-path through Gamma) up to exactly
    2 pi; _wrap_factor would then snap exp(2 pi i) to 1 and assemble the k = 0 matrix, while the
    spectrum formula and the null-space count still saw k != 0. Such a phase is 0 to rounding.
    """
    p = float(np.mod(phase, 2 * np.pi))
    return 0.0 if p >= 2 * np.pi else p


def _wrap_factor(phase: float) -> float | complex:
    """exp(i phase), exactly +-1 for multiples of pi so that real cases stay real."""
    r = phase / np.pi
    if r == round(r):
        return 1.0 if round(r) % 2 == 0 else -1.0
    return complex(np.exp(1j * phase))


def _difference_1d(grid: YeeGrid, d: int) -> sp.csr_matrix:
    """Cells x unknown nodes along dimension d: (D u)_i = u_{i+1} - u_i.

    PEC: boundary nodes carry no unknown (u_0 = u_N = 0). Bloch: u_N = exp(i k L) u_0.
    """
    n = grid.cells[d]
    if grid.boundary == "pec":
        return sp.diags([np.ones(n - 1), -np.ones(n - 1)], [0, -1], shape=(n, n - 1), format="csr")
    i = np.arange(n)
    vals = np.concatenate((-np.ones(n), np.ones(n))).astype(complex if grid.is_complex else float)
    vals[-1] = _wrap_factor(grid.phases[d])
    return sp.csr_matrix((vals, (np.concatenate((i, i)), np.concatenate((i, (i + 1) % n)))),
                         shape=(n, n))


def _average_1d(grid: YeeGrid, d: int) -> sp.csr_matrix:
    """Unknown nodes x cells along d: mean of the two cells adjacent to each node."""
    n = grid.cells[d]
    if grid.boundary == "pec":
        return sp.diags([np.full(n - 1, 0.5), np.full(n - 1, 0.5)], [0, 1], shape=(n - 1, n), format="csr")
    i = np.arange(n)
    return sp.csr_matrix((np.full(2 * n, 0.5), (np.concatenate((i, i)), np.concatenate((i, (i - 1) % n)))),
                         shape=(n, n))


def _kron3(a: sp.spmatrix, b: sp.spmatrix, c: sp.spmatrix) -> sp.csr_matrix:
    """a x b x c with the first (x) index varying slowest (C order of an (Nx, Ny, Nz) array)."""
    return sp.kron(sp.kron(a, b, format="csr"), c, format="csr")


def operators(grid: YeeGrid) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Topological curl C (faces x edges) and gradient G (edges x nodes).

    Edge blocks: x-edges (cell, node, node), y-edges (node, cell, node), z-edges (node, node, cell);
    face blocks: x-faces (node, cell, cell), y-faces (cell, node, cell), z-faces (cell, cell, node),
    each in C order. (C u)_x = d_y u_z - d_z u_y and cyclic, so C G = 0 because the 1-D difference
    factors acting on different dimensions commute.
    """
    D = [_difference_1d(grid, d) for d in range(3)]
    In = [sp.identity(grid.nodes(d), format="csr") for d in range(3)]
    Ic = [sp.identity(grid.cells[d], format="csr") for d in range(3)]
    curl = sp.bmat([
        [None, -_kron3(In[0], Ic[1], D[2]), _kron3(In[0], D[1], Ic[2])],
        [_kron3(Ic[0], In[1], D[2]), None, -_kron3(D[0], In[1], Ic[2])],
        [-_kron3(Ic[0], D[1], In[2]), _kron3(D[0], Ic[1], In[2]), None],
    ], format="csr")
    grad = sp.vstack([_kron3(D[0], In[1], In[2]), _kron3(In[0], D[1], In[2]),
                      _kron3(In[0], In[1], D[2])], format="csr")
    return curl, grad


def _cell_field(grid: YeeGrid, value) -> np.ndarray | float:
    if np.ndim(value) == 0:
        if float(value) <= 0:
            raise ValueError("material parameters must be positive")
        return float(value)
    arr = np.asarray(value, dtype=float)
    if arr.shape != grid.cells or arr.min() <= 0:
        raise ValueError(f"cell material array must have shape {grid.cells} and be positive")
    return arr


def _averaged(field: np.ndarray | float, maps: Sequence[sp.csr_matrix]) -> list[np.ndarray]:
    """Per-block averages of a cell field (a homogeneous scalar is used as is, exactly)."""
    if isinstance(field, float):
        return [np.full(m.shape[0], field) for m in maps]
    return [m @ field.ravel() for m in maps]


def assemble(grid: YeeGrid, eps=1.0, mu=1.0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """FIT pencil (A, B) = (C^H M_nu C, M_eps). eps, mu: positive scalars or (Nx, Ny, Nz) arrays
    of cell values. A is real symmetric unless a Bloch phase is not a multiple of pi."""
    hx, hy, hz = grid.spacing
    P = [_average_1d(grid, d) for d in range(3)]
    Ic = [sp.identity(grid.cells[d], format="csr") for d in range(3)]
    eps_e = _averaged(_cell_field(grid, eps),
                      [_kron3(Ic[0], P[1], P[2]), _kron3(P[0], Ic[1], P[2]), _kron3(P[0], P[1], Ic[2])])
    nu_f = _averaged(1.0 / _cell_field(grid, mu), [_kron3(P[0], Ic[1], Ic[2]), _kron3(Ic[0], P[1], Ic[2]),
                                _kron3(Ic[0], Ic[1], P[2])])
    edge_geom = (hy * hz / hx, hx * hz / hy, hx * hy / hz)    # |e~| / |e|
    face_geom = (hx / (hy * hz), hy / (hx * hz), hz / (hx * hy))  # |f~| / |f|
    m_eps = np.concatenate([e * g for e, g in zip(eps_e, edge_geom)])
    m_nu = np.concatenate([v * g for v, g in zip(nu_f, face_geom)])
    curl, _ = operators(grid)
    a = curl.conj().T @ sp.diags(m_nu) @ curl
    return hermitian_part(a), sp.diags(m_eps, format="csr")


def standard_form(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    """B^{-1/2} A B^{-1/2} for diagonal HPD B (same eigenvalues, eigenvectors scaled by B^{1/2})."""
    s = sp.diags(1.0 / np.sqrt(b.diagonal().real))
    return hermitian_part(s @ a @ s)


def exact_eigenvalues(grid: YeeGrid, eps: float = 1.0, mu: float = 1.0) -> np.ndarray:
    """Complete exact spectrum of the homogeneous pencil (see module docstring), ascending.

    The three per-dimension terms of each wavevector are summed in sorted order, so permuted
    index triples of equal dimensions give bitwise-equal eigenvalues (exact multiplicities).
    """
    if np.ndim(eps) or np.ndim(mu):
        raise ValueError("exact spectra exist only for homogeneous (scalar) eps and mu")
    h = grid.spacing
    terms = []
    for d in range(3):
        m = np.arange(grid.cells[d])
        if grid.boundary == "pec":
            arg = m * np.pi / (2 * grid.cells[d])
        else:
            arg = grid.phases[d] / (2 * grid.cells[d]) + m * np.pi / grid.cells[d]
        terms.append((2.0 / h[d] * np.sin(arg)) ** 2)
    idx = np.meshgrid(*[np.arange(t.size) for t in terms], indexing="ij")
    s = np.sort(np.stack([terms[d][idx[d].ravel()] for d in range(3)], axis=1), axis=1)
    g = (s[:, 0] + s[:, 1]) + s[:, 2]
    if grid.boundary == "pec":
        # Polarizations per mode index triple: E_x needs m_y, m_z >= 1 (sines across x), cyclic.
        active = np.count_nonzero(np.stack([t[i.ravel()] > 0 for t, i in zip(terms, idx)], axis=1), axis=1)
        mult = np.where(active == 3, 2, np.where(active == 2, 1, 0))
    else:
        mult = np.where(g > 0, 2, 0)
    w = np.concatenate((np.repeat(g, mult), np.zeros(grid.null_space_dimension())))
    if w.size != grid.edge_count:
        raise AssertionError("mode count does not match the number of edges")
    return np.sort(w / (eps * mu))


def photonic_permittivity(grid: YeeGrid, shape: str, radius: float, eps_in: float,
                          eps_out: float = 1.0, samples: int = 3) -> np.ndarray:
    """Cell permittivity of one unit cell of a photonic crystal (inclusion centred in the box).

    shape 'spheres': sphere of radius radius*min(L) (simple cubic lattice of spheres);
    shape 'rods'   : cylinder along z of radius radius*min(Lx, Ly) (square lattice of rods).
    Each cell gets the volume fraction measured on samples^3 sub-points (arithmetic average),
    which smooths the staircase without changing the FIT averaging rules.
    """
    if shape not in ("spheres", "rods"):
        raise ValueError("shape must be 'spheres' or 'rods'")
    h = grid.spacing
    coords = [((np.arange(n * samples) + 0.5) / samples) * hd - L / 2
              for n, hd, L in zip(grid.cells, h, grid.lengths)]
    x, y, z = np.meshgrid(*coords, indexing="ij", sparse=True)
    if shape == "spheres":
        inside = x**2 + y**2 + z**2 <= (radius * min(grid.lengths)) ** 2
    else:
        inside = np.broadcast_to(x**2 + y**2 <= (radius * min(grid.lengths[:2])) ** 2,
                                 tuple(n * samples for n in grid.cells))
    nx, ny, nz = grid.cells
    frac = inside.reshape(nx, samples, ny, samples, nz, samples).mean(axis=(1, 3, 5))
    return eps_out + (eps_in - eps_out) * frac


# ======================================================================= preset case writers


def _pencil(grid: YeeGrid, form: str, eps=1.0, mu=1.0) -> tuple[sp.csr_matrix, sp.csr_matrix | None, str]:
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")
    a, b = assemble(grid, eps, mu)
    return (standard_form(a, b), None, "HEP") if form == "standard" else (a, b, "GHEP")


def _meta(grid: YeeGrid, problem: str, description: str, params: dict[str, Any],
          homogeneous: bool) -> dict[str, Any]:
    """Case metadata. The null-space dimension (hence the inertia) is topological and holds for
    any material; the TE/TM pair degeneracy ('multiplicity') only for homogeneous media."""
    n, null = grid.edge_count, grid.null_space_dimension()
    inertia: dict[str, list[int]] = {"A": [0, null, n - null]}
    if problem == "GHEP":
        inertia.update(B=[0, 0, n], pencil=[0, null, n - null])
    scenarios = ["zero-cluster"] + (["multiplicity"] if homogeneous else [])
    if grid.is_complex:
        scenarios.append("complex-hermitian")
    if n >= 100_000:
        scenarios.append("large")
    return {"family": FAMILY, "problem": problem, "description": description,
            "params": {"generator": "yee", "cells": list(grid.cells), "lengths": list(grid.lengths),
                       "boundary": grid.boundary, "phases": list(grid.phases), **params},
            "inertia": inertia, "scenarios": scenarios}


def _exact_targets(w: np.ndarray, cond_b: float) -> list[dict[str, Any]]:
    """Low modes (between the 3rd and 4th nonzero levels; level 0 is the null space) and a
    higher band (25th level): both far from the gradient null space relative to their k.
    cond_b = cond(B) of the stored pencil (1 for the standard form) sets the decidable k."""
    return [target_between_levels(w, 3, 10, "lowest resonances above the gradient null space",
                                  cond_b=cond_b),
            target_between_levels(w, 25, 20, "higher band, many polarization-degenerate pairs",
                                  cond_b=cond_b)]


def make_homogeneous_case(name: str, grid: YeeGrid, form: str, root: str | Path) -> Path:
    """Empty PEC cavity or homogeneous Bloch cell (eps = mu = 1) with its exact spectrum."""
    a, b, problem = _pencil(grid, form)
    w = exact_eigenvalues(grid)
    cond_b = 1.0 if b is None else float(b.diagonal().max() / b.diagonal().min())
    what = "PEC cavity" if grid.boundary == "pec" else "Bloch-periodic cell"
    meta = _meta(grid, problem, f"empty {what} {grid.cells} cells, curl-curl ({form} form)",
                 {"form": form, "eps": 1.0, "mu": 1.0}, homogeneous=True)
    return write_case(name, a, b, meta=meta, targets=_exact_targets(w, cond_b), root=root, eigenvalues=w,
                      ref_note=("discrete Yee/FIT spectrum: sum_d (2/h_d sin(theta_d))^2 with TE/TM "
                                "multiplicities and the gradient null space; exact up to assembly rounding"))


def make_photonic_case(name: str, grid: YeeGrid, form: str, shape: str, radius: float,
                       eps_in: float, root: str | Path) -> Path:
    """Photonic-crystal unit cell at a Bloch wavevector; no exact reference exists."""
    eps = photonic_permittivity(grid, shape, radius, eps_in)
    a, b, problem = _pencil(grid, form, eps=eps)
    # Targets from the effective-medium estimate (homogeneous cell with the volume-averaged eps):
    # a physically plausible frequency window, not a reference spectrum of this matrix.
    eps_avg = float(eps.mean())
    lev, _ = distinct_levels(exact_eigenvalues(grid, eps=eps_avg))
    targets = [{"sigma": float(lev[6] + 0.4 * (lev[7] - lev[6])), "k": 10,
                "why": f"low bands (effective-medium estimate with eps_avg={eps_avg:.4g})"}]
    meta = _meta(grid, problem, f"photonic crystal ({shape}, eps={eps_in:g}, r={radius:g}) unit cell, "
                                f"{grid.cells} cells ({form} form)",
                 {"form": form, "shape": shape, "radius": radius, "eps_in": eps_in, "eps_out": 1.0,
                  "mu": 1.0, "eps_avg": eps_avg}, homogeneous=False)
    return write_case(name, a, b, meta=meta, targets=targets, root=root)


def _presets() -> dict[str, Callable[[Path], Path]]:
    k_generic = (0.3 * np.pi, 0.2 * np.pi, 0.1 * np.pi)
    k_crystal = (0.4 * np.pi, 0.25 * np.pi, 0.1 * np.pi)
    cube16 = YeeGrid((16, 16, 16))
    box1e4 = YeeGrid((20, 16, 12), (1.0, 0.7, 0.45))
    cube32 = YeeGrid((32, 32, 32))
    box1e5 = YeeGrid((44, 34, 26), (1.0, 0.75, 0.55))
    bloch1e4 = YeeGrid((16, 15, 14), (1.0, 0.9, 0.8), "bloch", k_generic)
    crystal1e4 = YeeGrid((15, 15, 15), (1.0, 1.0, 1.0), "bloch", k_crystal)
    return {
        "yee-pec-cube-16": lambda root: make_homogeneous_case("yee-pec-cube-16", cube16, "standard", root),
        "yee-pec-box-1e4": lambda root: make_homogeneous_case("yee-pec-box-1e4", box1e4, "generalized", root),
        "yee-pec-cube-32": lambda root: make_homogeneous_case("yee-pec-cube-32", cube32, "standard", root),
        "yee-pec-box-1e5": lambda root: make_homogeneous_case("yee-pec-box-1e5", box1e5, "generalized", root),
        "yee-bloch-1e4": lambda root: make_homogeneous_case("yee-bloch-1e4", bloch1e4, "standard", root),
        "yee-photonic-spheres-1e4": lambda root: make_photonic_case(
            "yee-photonic-spheres-1e4", crystal1e4, "generalized", "spheres", 0.25, 13.0, root),
        "yee-photonic-rods-1e4": lambda root: make_photonic_case(
            "yee-photonic-rods-1e4", crystal1e4, "standard", "rods", 0.2, 8.9, root),
    }


PRESETS = _presets()


def main(argv: Sequence[str] | None = None) -> int:
    return run_presets(PRESETS, argv, prog="python -m tools.gen.yee",
                       description="Generate Yee-grid curl-curl cases into <root>/cases.")


if __name__ == "__main__":
    raise SystemExit(main())
