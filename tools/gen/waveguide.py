"""Waveguide-family test cases: 2-D vector finite-element eigenproblems from electromagnetics.

Families
--------
``waveguide``
    Propagation constants of a PEC rectangular waveguide (hollow, or loaded with a dielectric /
    magnetic slab) in the Lee-Sun-Cendes mixed formulation: lowest-order Nedelec (Whitney) edge
    elements for the scaled transverse field e_t = beta E_t and P1 nodal elements for the scaled
    axial field e_z = -j E_z.  Substituting into the functional
        F = int 1/mu |curl E|^2 - k0^2 eps |E|^2,   E = (E_t + z E_z) exp(-j beta z),
    and multiplying by beta^2 gives
        int 1/mu |curl e_t|^2 - k0^2 eps |e_t|^2 + beta^2 (1/mu |grad e_z + e_t|^2 - k0^2 eps |e_z|^2),
    whose stationarity is the generalized eigenproblem (GHIEP, B indefinite)
        A = [[S_tt/mu - k0^2 eps T_tt, 0], [0, 0]],
        B = [[T_tt/mu, G/mu], [G^T/mu, S_zz/mu - k0^2 eps T_zz]],     A x = lambda B x,
        lambda = -beta^2  (propagating modes: lambda < 0, evanescent: lambda > 0),
    with S_tt = int curl N_i curl N_j, T_tt = int N_i . N_j, G = int N_i . grad L_j,
    S_zz = int grad L_i . grad L_j, T_zz = int L_i L_j (eps, mu piecewise constant per triangle).
    References: J.-F. Lee, D.-K. Sun, Z. J. Cendes, IEEE Trans. MTT 39(8):1262-1271, 1991;
    J. Jin, The Finite Element Method in Electromagnetics (Wiley), ch. 8.
``cavity2d``
    The 2-D TE cutoff problem S_tt x = kc^2 T_tt x on the same edge space (GHEP, B = T_tt HPD).
    Its null space is the discrete gradient space, of dimension #interior nodes.

Structure of the discrete problems (the expectations written into meta.json rely on it)
----------------------------------------------------------------------------------------
* Whitney forms reproduce gradients exactly: grad L_j = sum_e D[e, j] N_e with the signed
  edge-node incidence D, hence G = T_tt D, D^T T_tt D = S_zz and S_tt D = 0 (curl grad = 0),
  also for every per-triangle weight (X^mu below denotes the 1/mu-weighted, X^eps the
  eps-weighted matrix).
* B is congruent to blkdiag(T_tt^mu, -k0^2 T_zz^eps): with G^mu = T_tt^mu D the Schur complement
  is S_zz^mu - k0^2 T_zz^eps - (G^mu)^T (T_tt^mu)^{-1} G^mu = -k0^2 T_zz^eps.  Hence
  inertia(B) = (#interior nodes negative, 0, #interior edges positive) on every mesh.
* A has exactly #interior nodes zero eigenvalues (its zz block) unless k0^2 is a discrete
  curl-curl eigenvalue.  The vectors x = [0; psi] are pencil eigenvectors with lambda = 0 (the
  non-physical zero cluster a solver must filter); 0 is semisimple unless k0^2 is also a discrete
  TM cutoff.
* Hollow guide: the pencil decouples exactly into discrete TE modes (lambda = kc^2 - k0^2, kc^2 a
  nonzero curl-curl eigenvalue) and TM modes (kc^2 a P1 Dirichlet-Laplacian eigenvalue), so the
  whole spectrum is real.
* Slab-loaded guide: the continuous LSE / LSM families are real, but B is indefinite and the
  evanescent part (lambda > 0) mixes eigenvalues of both sign characteristics (x^H B x > 0 for
  LSE-like, < 0 for LSM-like modes).  Where two such branches nearly cross, the discretization
  couples them and they collide into complex-conjugate pairs (dense checks: 24 / 46 / 52 complex
  eigenvalues for 8 / 12 / 16 cells per side, all with Re lambda > 0.46).  The propagating modes
  (lambda < 0) all have positive type and stay real, so they remain a well-posed target.

Units: lengths in mm and frequency in GHz, so k0 [rad/mm] = 2 pi f / c with c = 299.792458 mm/ns
and every eigenvalue is in mm^-2.  This reproduces the competition's real case (10 mm square air
guide at 50 GHz: 18 propagating modes with lambda in about [-1, 0)).

Reference values are those of the continuous model (kind "partial-analytic"); the discrete
eigenvalues differ by O(h^2), bounded a priori by ``error_bound``.  Reference windows and target
suggestions are placed in spectral gaps that this bound cannot close (``safe_cuts``), so their
counts are valid for the discrete problem.  Cases with n <= DENSE_LIMIT are additionally checked
with dense eigensolvers and their exact discrete inertia is recorded.

Usage: python -m tools.gen.waveguide --family waveguide --a 10 --b 10 --freq 50 --target-n 1000 \
           --name wg_square_1e3            (see --help; output via tools/case.py)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import brentq

from tools import case
from tools.gen.mesh2d import (LOCAL_EDGES, PATTERNS, EdgeTopology, TriMesh, edge_topology, grid_interior_counts,
                              grid_ticks, max_edge_length, structured_rectangle, triangle_areas)

C0_MM_PER_NS = 299.792458   # speed of light in mm/ns = mm * GHz
FAMILIES = ("waveguide", "cavity2d")
DENSE_LIMIT = 2000          # dense verification of inertia / spectrum only for n <= DENSE_LIMIT
LARGE_N = 500_000           # scenario tags "large" and "memory" from this size on
CAVITY_LISTED = 64          # cavity2d references list the lowest CAVITY_LISTED nonzero kc^2
# A-priori bound |lambda_h - lambda| <= ERR_CONST * h^2 * kt^4 with h the longest mesh edge and
# kt^2 = lambda + k0^2 max(eps mu) (waveguide) or kt^2 = kc^2 (cavity2d), the transverse
# wavenumber of the mode.  Measured constants on union-jack / diagonal meshes, unperturbed and
# perturbed (0.24), 16 and 32 cells per side (tests/py/test_gen_waveguide.py re-checks the bound):
# <= 0.025 cavity2d (Nedelec TE cutoffs), <= 0.069 hollow guide (dominated by the P1 TM cutoffs),
# <= 0.047 slab-loaded guide; 0.15 leaves at least a 2x margin.
ERR_CONST = 0.15


def wavenumber(freq_ghz: float) -> float:
    """Free-space wavenumber k0 in rad/mm for a frequency in GHz."""
    return 2.0 * np.pi * freq_ghz / C0_MM_PER_NS


# --------------------------------------------------------------------------- element matrices

@dataclass(frozen=True)
class ElementMatrices:
    """Per-triangle matrices, edge rows/columns already in global orientation (lo -> hi)."""
    curl: np.ndarray    # (nt, 3, 3)  int curl N_k curl N_l             -> S_tt
    mass: np.ndarray    # (nt, 3, 3)  int N_k . N_l                     -> T_tt
    grad: np.ndarray    # (nt, 3, 3)  int N_k . grad L_q  (edge x node) -> G
    stiff: np.ndarray   # (nt, 3, 3)  int grad L_p . grad L_q           -> S_zz
    nmass: np.ndarray   # (nt, 3, 3)  int L_p L_q                       -> T_zz


def element_matrices(mesh: TriMesh, topo: EdgeTopology) -> ElementMatrices:
    """Closed-form P1 / Whitney element integrals in barycentric coordinates.

    With Delta the area, g_ab = grad L_a . grad L_b (constant) and M_ab = int L_a L_b =
    Delta (1 + d_ab) / 12, the Whitney function of the local edge (i -> j) is
    N_ij = L_i grad L_j - L_j grad L_i, so
        curl N_ij = 2 grad L_i x grad L_j                         (constant),
        int N_ij . N_mn = M_im g_jn - M_in g_jm - M_jm g_in + M_jn g_im,
        int N_ij . grad L_q = Delta/3 (g_jq - g_iq).
    """
    p = mesh.nodes[mesh.triangles]                         # (nt, 3, 2)
    area = triangle_areas(mesh)
    # grad L_i = (y_j - y_k, x_k - x_j) / (2 Delta) for (i, j, k) cyclic.
    nxt, prv = [1, 2, 0], [2, 0, 1]
    grads = np.stack([p[:, nxt, 1] - p[:, prv, 1], p[:, prv, 0] - p[:, nxt, 0]], axis=-1)
    grads /= (2.0 * area)[:, None, None]
    g = np.einsum("tad,tbd->tab", grads, grads)            # (nt, 3, 3)
    nmass = (area / 12.0)[:, None, None] * (np.ones((3, 3)) + np.eye(3))[None]

    i_loc = np.array([e[0] for e in LOCAL_EDGES])
    j_loc = np.array([e[1] for e in LOCAL_EDGES])
    sign = topo.tri_signs                                    # (nt, 3)
    curl = 2.0 * (grads[:, i_loc, 0] * grads[:, j_loc, 1] - grads[:, i_loc, 1] * grads[:, j_loc, 0])
    curl *= sign

    def pick(m: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return m[:, rows[:, None], cols[None, :]]

    mass = (pick(nmass, i_loc, i_loc) * pick(g, j_loc, j_loc) - pick(nmass, i_loc, j_loc) * pick(g, j_loc, i_loc)
            - pick(nmass, j_loc, i_loc) * pick(g, i_loc, j_loc) + pick(nmass, j_loc, j_loc) * pick(g, i_loc, i_loc))
    mass *= sign[:, :, None] * sign[:, None, :]
    # The four products are symmetric term by term, but their summation order is not;
    # symmetrize so that every assembled matrix is exactly (bitwise) symmetric.
    mass = 0.5 * (mass + mass.transpose(0, 2, 1))
    grad = (area / 3.0)[:, None, None] * (g[:, j_loc, :] - g[:, i_loc, :]) * sign[:, :, None]
    return ElementMatrices(curl=area[:, None, None] * curl[:, :, None] * curl[:, None, :],
                           mass=mass, grad=grad, stiff=area[:, None, None] * g, nmass=nmass)


# --------------------------------------------------------------------------------- assembly

@dataclass(frozen=True)
class Discretization:
    """Mesh + topology + element matrices + PEC dof numbering (boundary edges/nodes dropped)."""
    mesh: TriMesh
    topo: EdgeTopology
    elem: ElementMatrices
    edge_dof: np.ndarray   # (ne,) interior-edge number, -1 for PEC boundary edges
    node_dof: np.ndarray   # (nv,) interior-node number, -1 for PEC boundary nodes

    @property
    def num_edge_dofs(self) -> int:
        return int((self.edge_dof >= 0).sum())

    @property
    def num_node_dofs(self) -> int:
        return int((self.node_dof >= 0).sum())

    def tri_edge_dofs(self) -> np.ndarray:
        return self.edge_dof[self.topo.tri_edges]

    def tri_node_dofs(self) -> np.ndarray:
        return self.node_dof[self.mesh.triangles]


def discretize(mesh: TriMesh) -> Discretization:
    topo = edge_topology(mesh)
    edge_dof = np.full(topo.num_edges, -1, dtype=np.int64)
    edge_dof[~topo.boundary_edge] = np.arange(int((~topo.boundary_edge).sum()))
    node_dof = np.full(mesh.num_nodes, -1, dtype=np.int64)
    node_dof[~topo.boundary_node] = np.arange(int((~topo.boundary_node).sum()))
    return Discretization(mesh=mesh, topo=topo, elem=element_matrices(mesh, topo),
                          edge_dof=edge_dof, node_dof=node_dof)


def _assemble(shape: tuple[int, int], parts, symmetric: bool) -> sp.csr_matrix:
    """Sum element blocks into a CSR matrix.  parts: (row dofs (nt,p), col dofs (nt,q),
    values (nt,p,q)); entries with a negative (PEC) dof are dropped.  ``symmetric`` replaces the
    result by (M + M^T)/2, which is exact and removes the rounding asymmetry left by the
    order-dependent summation of duplicates."""
    rows, cols, vals = [], [], []
    for r_dofs, c_dofs, values in parts:
        r = np.broadcast_to(r_dofs[:, :, None], values.shape)
        c = np.broadcast_to(c_dofs[:, None, :], values.shape)
        keep = (r >= 0) & (c >= 0)
        rows.append(r[keep])
        cols.append(c[keep])
        vals.append(values[keep])
    m = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=shape).tocsr()
    m.sum_duplicates()
    if symmetric:
        m = ((m + m.T) * 0.5).tocsr()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def _weights(disc: Discretization, w) -> np.ndarray:
    """Per-triangle coefficient array (w: None -> 1, scalar, or (nt,) array)."""
    nt = disc.mesh.num_triangles
    return np.ones(nt) if w is None else np.broadcast_to(np.asarray(w, dtype=float), (nt,))


def curl_tt(disc: Discretization, w=None) -> sp.csr_matrix:
    """S_tt = sum_T w_T int curl N_i curl N_j over interior edges."""
    ed = disc.tri_edge_dofs()
    n = disc.num_edge_dofs
    return _assemble((n, n), [(ed, ed, disc.elem.curl * _weights(disc, w)[:, None, None])], True)


def mass_tt(disc: Discretization, w=None) -> sp.csr_matrix:
    """T_tt = sum_T w_T int N_i . N_j over interior edges."""
    ed = disc.tri_edge_dofs()
    n = disc.num_edge_dofs
    return _assemble((n, n), [(ed, ed, disc.elem.mass * _weights(disc, w)[:, None, None])], True)


def grad_tz(disc: Discretization, w=None) -> sp.csr_matrix:
    """G = sum_T w_T int N_i . grad L_j (interior edges x interior nodes)."""
    shape = (disc.num_edge_dofs, disc.num_node_dofs)
    values = disc.elem.grad * _weights(disc, w)[:, None, None]
    return _assemble(shape, [(disc.tri_edge_dofs(), disc.tri_node_dofs(), values)], False)


def stiff_zz(disc: Discretization, w=None) -> sp.csr_matrix:
    """S_zz = sum_T w_T int grad L_i . grad L_j over interior nodes."""
    nd = disc.tri_node_dofs()
    n = disc.num_node_dofs
    return _assemble((n, n), [(nd, nd, disc.elem.stiff * _weights(disc, w)[:, None, None])], True)


def mass_zz(disc: Discretization, w=None) -> sp.csr_matrix:
    """T_zz = sum_T w_T int L_i L_j over interior nodes."""
    nd = disc.tri_node_dofs()
    n = disc.num_node_dofs
    return _assemble((n, n), [(nd, nd, disc.elem.nmass * _weights(disc, w)[:, None, None])], True)


def incidence(disc: Discretization) -> sp.csr_matrix:
    """Signed edge-node incidence D (interior edges x interior nodes): D[e, hi] = +1, D[e, lo] = -1."""
    keep = disc.edge_dof >= 0
    e = disc.edge_dof[keep]
    lo = disc.node_dof[disc.topo.edges[keep, 0]]
    hi = disc.node_dof[disc.topo.edges[keep, 1]]
    rows = np.concatenate([e[hi >= 0], e[lo >= 0]])
    cols = np.concatenate([hi[hi >= 0], lo[lo >= 0]])
    vals = np.concatenate([np.ones(int((hi >= 0).sum())), -np.ones(int((lo >= 0).sum()))])
    return sp.csr_matrix((vals, (rows, cols)), shape=(disc.num_edge_dofs, disc.num_node_dofs))


def waveguide_pencil(disc: Discretization, k0: float, eps_r=None, mu_r=None):
    """(A, B) of the Lee-Sun-Cendes pencil; unknowns ordered [interior edges; interior nodes].

    eps_r, mu_r: per-triangle relative permittivity / permeability (None, scalar or (nt,) array).
    """
    eps = _weights(disc, eps_r)[:, None, None]
    inv_mu = 1.0 / _weights(disc, mu_r)[:, None, None]
    ne = disc.num_edge_dofs
    n = ne + disc.num_node_dofs
    el = disc.elem
    ed = disc.tri_edge_dofs()
    nd = disc.tri_node_dofs()
    nd = np.where(nd >= 0, nd + ne, -1)
    k2 = k0 * k0
    A = _assemble((n, n), [(ed, ed, inv_mu * el.curl - k2 * eps * el.mass)], True)
    grad = inv_mu * el.grad
    B = _assemble((n, n), [(ed, ed, inv_mu * el.mass), (ed, nd, grad), (nd, ed, grad.transpose(0, 2, 1)),
                           (nd, nd, inv_mu * el.stiff - k2 * eps * el.nmass)], True)
    return A, B


def cavity_pencil(disc: Discretization):
    """(S_tt, T_tt): 2-D TE cutoff problem S_tt x = kc^2 T_tt x of a hollow PEC rectangle."""
    return curl_tt(disc), mass_tt(disc)


# ------------------------------------------------------------------------------ dense checks

def dense_inertia(m: sp.spmatrix, rel_tol: float = 1e-9) -> list[int]:
    """[neg, zero, pos] of a small symmetric matrix; |mu| <= rel_tol * max|mu| counts as zero."""
    w = np.linalg.eigvalsh(m.toarray())
    tol = rel_tol * float(np.abs(w).max())
    return [int((w < -tol).sum()), int((np.abs(w) <= tol).sum()), int((w > tol).sum())]


def dense_pencil_eigenvalues(A: sp.spmatrix, B: sp.spmatrix) -> np.ndarray:
    """All eigenvalues of a small pencil (QZ); B must be nonsingular."""
    return sla.eigvals(A.toarray(), B.toarray())


def count_signs(values: np.ndarray, tol: float) -> list[int]:
    """[#(v < -tol), #(|v| <= tol), #(v > tol)]."""
    return [int((values < -tol).sum()), int((np.abs(values) <= tol).sum()), int((values > tol).sum())]


# --------------------------------------------------------------------------- analytic spectra

@dataclass(frozen=True)
class Mode:
    value: float   # pencil eigenvalue: lambda = -beta^2 (waveguide) or kc^2 (cavity2d), mm^-2
    label: str     # TE/TM m,n (hollow) or LSE/LSM m,n (slab-loaded)


def rectangle_cutoffs(a: float, b: float, kc2_max: float, *, tm: bool) -> list[Mode]:
    """kc^2 = (m pi/a)^2 + (n pi/b)^2 <= kc2_max of TE_mn (m, n >= 0, not both 0) and, if ``tm``,
    TM_mn (m, n >= 1) of a PEC a x b rectangle, ascending."""
    kmax = np.sqrt(max(kc2_max, 0.0))
    modes = []
    for m in range(int(a * kmax / np.pi) + 1):
        for n in range(int(b * kmax / np.pi) + 1):
            kc2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            if (m, n) == (0, 0) or kc2 > kc2_max:
                continue
            modes.append(Mode(kc2, f"TE{m},{n}"))
            if tm and m > 0 and n > 0:
                modes.append(Mode(kc2, f"TM{m},{n}"))
    return sorted(modes, key=lambda md: (md.value, md.label))


def hollow_guide_modes(a: float, b: float, k0: float, lam_max: float) -> list[Mode]:
    """lambda = kc^2 - k0^2 of every TE/TM mode of an air-filled guide with lambda <= lam_max."""
    return [Mode(md.value - k0 * k0, md.label) for md in rectangle_cutoffs(a, b, lam_max + k0 * k0, tm=True)]


def _cos_sinc(s: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """C = cos(sqrt(s) L) and Sn = sin(sqrt(s) L) / sqrt(s): entire functions of s (cosh / sinh
    for s < 0, Sn(0) = L), so the characteristic functions below have no poles or branch cuts."""
    r = np.sqrt(np.abs(s))
    z = r * length
    osc = s >= 0
    c = np.where(osc, np.cos(z), np.cosh(z))
    zs = np.where(z == 0.0, 1.0, z)
    sn = length * np.where(z == 0.0, 1.0, np.where(osc, np.sin(zs), np.sinh(zs)) / zs)
    return c, sn


def _slab_characteristic(lam, ky2, k0, d, length, eps1, mu1, family):
    """Characteristic function of the LSE^x / LSM^x modes of a guide with a slab (eps1, mu1) on
    0 < x < d and air on d < x < d + length, at lambda = -beta^2 and ky = n pi / b.

    With s_i = k0^2 eps_i mu_i - ky^2 + lambda (= kx_i^2) in each region:
    * LSE (E_x = 0, F = x psi): u = psi/eps solves u'' + s u = 0 with u = 0 on both walls, u and
      u'/mu continuous; shooting from x = 0 gives u(a) proportional to
          mu_2 C1 Sn2 + mu_1 C2 Sn1.
    * LSM (H_x = 0, A = x phi): v = phi/mu with v' = 0 on both walls, v and v'/eps continuous;
      the shooting function v'(a) is proportional to
          eps_2 s1 Sn1 C2 + eps_1 s2 Sn2 C1.
    Both are shooting functions of regular Sturm-Liouville problems in lambda, so their zeros are
    exactly the (simple) eigenvalues.  Air has eps_2 = mu_2 = 1.
    """
    s1 = k0 * k0 * eps1 * mu1 - ky2 + lam
    s2 = k0 * k0 - ky2 + lam
    c1, sn1 = _cos_sinc(s1, d)
    c2, sn2 = _cos_sinc(s2, length)
    if family == "LSE":
        return c1 * sn2 + mu1 * c2 * sn1
    return s1 * sn1 * c2 + eps1 * s2 * sn2 * c1


def slab_guide_modes(a: float, b: float, k0: float, slab_width: float, eps_slab: float,
                     mu_slab: float, lam_max: float, *, refine: int = 1) -> list[Mode]:
    """lambda = -beta^2 <= lam_max of all modes of a PEC guide loaded with a full-height slab
    (eps_slab, mu_slab) on 0 < x < slab_width, air elsewhere (R. F. Harrington, Time-Harmonic
    Electromagnetic Fields, partially filled waveguides).  Modes vary as cos / sin(n pi y / b):
    LSE_mn (n >= 0, m = 1, 2, ...) and LSM_mn (n >= 1, m = 0, 1, ...), m counting the roots of the
    characteristic function upwards in lambda, i.e. the Sturm-Liouville order, so the labels map
    to TE/TM_mn in the empty limit.

    No eigenvalue lies below ky^2 - k0^2 max(eps mu) (Rayleigh quotient), so each family is
    scanned upwards from just below that bound on a grid of step (pi/a)^2 / (1024 refine), far
    finer than the eigenvalue spacing of the 1-D problems, and every sign change is refined with
    Brent's method.  ``refine`` > 1 only serves to verify that no root pair was missed.
    """
    if not 0.0 <= slab_width <= a:
        raise ValueError("slab_width must lie in [0, a]")
    k2max = k0 * k0 * max(1.0, eps_slab * mu_slab)
    step = (np.pi / a) ** 2 / (1024.0 * refine)
    modes = []
    n = 0
    while (n * np.pi / b) ** 2 - k2max < lam_max:
        ky2 = (n * np.pi / b) ** 2
        grid = np.arange(ky2 - k2max - step, lam_max + step, step)
        for family in ("LSE", "LSM") if n > 0 else ("LSE",):
            def f(x, fam=family):
                return _slab_characteristic(x, ky2, k0, slab_width, a - slab_width, eps_slab, mu_slab, fam)
            vals = f(grid)
            roots = list(grid[vals == 0.0])
            change = np.nonzero(vals[:-1] * vals[1:] < 0.0)[0]
            roots += [brentq(f, grid[i], grid[i + 1], xtol=1e-15, rtol=4.0 * np.finfo(float).eps) for i in change]
            first = 1 if family == "LSE" else 0
            for m, lam in enumerate(sorted(r for r in roots if r <= lam_max), start=first):
                modes.append(Mode(float(lam), f"{family}{m},{n}"))
        n += 1
    return sorted(modes, key=lambda md: (md.value, md.label))


# ------------------------------------------------------- error model, safe gaps and targets

def error_bound(values, kt2_shift: float, h: float) -> np.ndarray:
    """A-priori discretization error ERR_CONST h^2 kt^4 with kt^2 = value + kt2_shift."""
    kt2 = np.maximum(np.asarray(values, dtype=float) + kt2_shift, 0.0)
    return ERR_CONST * h * h * kt2 * kt2


def safe_cuts(values: np.ndarray, errs: np.ndarray) -> list[tuple[int, float]]:
    """Cuts that no perturbed eigenvalue can cross.  For ascending values with error bars, a cut
    after index i exists when max_{j<=i}(v_j + e_j) < min_{j>i}(v_j - e_j); it is placed at the
    midpoint of that free interval.  Returns [(i, cut)] for every such gap."""
    upper = np.maximum.accumulate(values + errs)
    lower = np.minimum.accumulate((values - errs)[::-1])[::-1]
    return [(i, 0.5 * (upper[i] + lower[i + 1])) for i in range(values.size - 1) if upper[i] < lower[i + 1]]


@dataclass(frozen=True)
class Target:
    """A suggested 'nearest k to sigma' query (meta.json targets)."""
    sigma: float
    k: int
    why: str

    def as_dict(self) -> dict:
        return {"sigma": float(self.sigma), "k": int(self.k), "why": self.why}


def nearest_block(cuts: list[tuple[int, float]], first: int, min_size: int,
                  lower_edge: float) -> tuple[float, int] | None:
    """(sigma, k) of a robust 'nearest k' query for the block of sorted values starting at index
    ``first`` (all of them above ``lower_edge``): the block ends at the first safe cut leaving at
    least ``min_size`` values in it and sigma is the midpoint of [lower_edge, cut].  Every block
    member then lies strictly closer to sigma than every other eigenvalue, even after the
    worst-case discretization shift.  None if the spectrum has no such cut."""
    for i, c in cuts:
        if i - first + 1 >= min_size:
            return 0.5 * (lower_edge + c), i - first + 1
    return None


@dataclass(frozen=True)
class Spectrum:
    """Sorted eigenvalue model with error bars, the zero cluster included as exact zeros."""
    values: np.ndarray
    errs: np.ndarray
    zero_first: int   # index range [zero_first, zero_last] of the zero cluster
    zero_last: int
    exact: bool       # True: dense discrete spectrum; False: continuous model + error bound

    @staticmethod
    def build(values, errs, zero_tol: float, exact: bool) -> "Spectrum":
        order = np.argsort(values, kind="stable")
        v = np.asarray(values, dtype=float)[order]
        e = np.asarray(errs, dtype=float)[order]
        zeros = np.nonzero(np.abs(v) <= zero_tol)[0]
        return Spectrum(v, e, int(zeros[0]), int(zeros[-1]), exact)

    def cuts(self) -> list[tuple[int, float]]:
        return safe_cuts(self.values, self.errs)

    def cut_below_zero(self) -> tuple[int, float] | None:
        below = [(i, c) for i, c in self.cuts() if i < self.zero_first]
        return below[-1] if below else None

    def cut_above_zero(self) -> float | None:
        above = [c for i, c in self.cuts() if i == self.zero_last]
        return above[0] if above else None

    def zero_isolated(self) -> bool:
        """The zero cluster is separated from both neighbours, so the counts below / above it
        are those of the discrete problem."""
        below = self.cut_below_zero()
        return (below is not None and below[0] == self.zero_first - 1 and self.cut_above_zero() is not None)


# ---------------------------------------------------------------------------- case definition

@dataclass(frozen=True)
class CaseSpec:
    """All generator inputs; together with the code version they determine the case exactly."""
    family: str = "waveguide"
    a: float = 10.0                    # mm (x extent)
    b: float = 10.0                    # mm (y extent)
    freq_ghz: float = 50.0             # waveguide only
    eps_slab: float = 1.0              # waveguide only: slab on 0 < x < slab_width
    mu_slab: float = 1.0
    slab_width: float | None = None    # mm, default a / 2
    nx: int | None = None              # cells along x (alternative: target_n)
    ny: int | None = None              # cells along y (default: square-ish cells)
    target_n: int | None = None        # choose nx, ny so that n is closest to target_n
    pattern: str = "unionjack"
    perturb: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.a <= 0 or self.b <= 0 or self.freq_ghz <= 0 or self.eps_slab <= 0 or self.mu_slab <= 0:
            raise ValueError("a, b, freq, eps_slab and mu_slab must be positive")
        if self.family == "cavity2d" and (self.eps_slab, self.mu_slab) != (1.0, 1.0):
            raise ValueError("cavity2d is the hollow TE cutoff problem: no slab")
        if (self.nx is None) == (self.target_n is None):
            raise ValueError("give exactly one of nx (optionally with ny) and target_n")
        if self.ny is not None and self.nx is None:
            raise ValueError("ny requires nx")
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")

    @property
    def width(self) -> float:
        """Resolved slab width (0 for an empty guide)."""
        if self.family != "waveguide" or (self.eps_slab, self.mu_slab) == (1.0, 1.0):
            return 0.0
        w = self.a / 2.0 if self.slab_width is None else float(self.slab_width)
        if not 0.0 <= w <= self.a:
            raise ValueError("slab_width must lie in [0, a]")
        return w

    @property
    def loaded(self) -> bool:
        return self.width > 0.0


def dof_count(family: str, nx: int, ny: int) -> int:
    """n of the generated pencil: interior edges (+ interior nodes for the waveguide)."""
    nodes, edges = grid_interior_counts(nx, ny)
    return edges + nodes if family == "waveguide" else edges


def resolve_grid(spec: CaseSpec) -> tuple[int, int]:
    """Cells (nx, ny); with target_n the cells stay as square as a/b allows and n is closest to
    the target (n grows monotonically with nx, so the scan stops after passing it)."""
    if spec.nx is not None:
        nx = spec.nx
        ny = spec.ny if spec.ny is not None else max(2, round(nx * spec.b / spec.a))
    else:
        best = None
        nx = 2
        while True:
            ny = max(2, round(nx * spec.b / spec.a))
            n = dof_count(spec.family, nx, ny)
            if best is None or abs(n - spec.target_n) < best[0]:
                best = (abs(n - spec.target_n), nx, ny)
            if n > spec.target_n:
                break
            nx += 1
        nx, ny = best[1], best[2]
    if nx < 2 or ny < 2:
        raise ValueError("need at least 2 x 2 cells (interior nodes)")
    return nx, ny


def build_mesh(spec: CaseSpec, nx: int, ny: int) -> tuple[TriMesh, np.ndarray, np.ndarray]:
    """Mesh plus per-triangle eps_r, mu_r.  A slab interface is a grid line that the random
    perturbation keeps straight, so every triangle lies on one side and the material is exact."""
    w = spec.width
    interface = [w] if 0.0 < w < spec.a else []
    mesh = structured_rectangle(grid_ticks(spec.a, nx, interface), grid_ticks(spec.b, ny),
                                pattern=spec.pattern, perturb=spec.perturb, seed=spec.seed,
                                straight_x=interface)
    in_slab = mesh.nodes[mesh.triangles][:, :, 0].mean(axis=1) < w
    eps_r = np.where(in_slab, spec.eps_slab, 1.0)
    mu_r = np.where(in_slab, spec.mu_slab, 1.0)
    return mesh, eps_r, mu_r


def command_line(spec: CaseSpec, name: str, nx: int, ny: int) -> str:
    """CLI invocation that regenerates the case bit for bit (resolved grid, all parameters)."""
    args = [f"--family {spec.family}", f"--a {spec.a!r}", f"--b {spec.b!r}"]
    if spec.family == "waveguide":
        args += [f"--freq {spec.freq_ghz!r}", f"--eps-slab {spec.eps_slab!r}", f"--mu-slab {spec.mu_slab!r}"]
        if spec.loaded:
            args.append(f"--slab-width {spec.width!r}")
    args += [f"--nx {nx}", f"--ny {ny}", f"--pattern {spec.pattern}", f"--perturb {spec.perturb!r}",
             f"--seed {spec.seed}", f"--name {name}"]
    return "python -m tools.gen.waveguide " + " ".join(args)


@dataclass
class GeneratedCase:
    name: str
    A: sp.csr_matrix
    B: sp.csr_matrix
    meta: dict
    config: dict
    ref: dict          # keyword arguments of tools.case.save_ref
    timings: dict


def _analytic_modes(spec: CaseSpec, k0: float, kt2_shift: float) -> tuple[list[Mode], list[Mode]]:
    """(listed, validation) continuous-model modes.  Waveguide: listed = lambda <= k0^2 max(eps mu)
    (all propagating plus the first evanescent modes); cavity2d: the lowest CAVITY_LISTED kc^2
    (with ties).  The validation set extends to twice the listed kt^2 range so that window edges
    near the end of the listing are checked against the next unlisted eigenvalues too."""
    if spec.family == "waveguide":
        lam_list = kt2_shift
        lam_ext = 3.0 * kt2_shift
        if spec.loaded:
            ext = slab_guide_modes(spec.a, spec.b, k0, spec.width, spec.eps_slab, spec.mu_slab, lam_ext)
        else:
            ext = hollow_guide_modes(spec.a, spec.b, k0, lam_ext)
        return [m for m in ext if m.value <= lam_list], ext
    kc2 = 4.0 * np.pi * CAVITY_LISTED / (spec.a * spec.b)   # Weyl: N(kc^2) ~ a b kc^2 / (4 pi)
    while len(rectangle_cutoffs(spec.a, spec.b, kc2, tm=False)) < CAVITY_LISTED:
        kc2 *= 1.5
    ext = rectangle_cutoffs(spec.a, spec.b, 2.0 * kc2, tm=False)
    last = ext[CAVITY_LISTED - 1].value
    return [m for m in ext if m.value <= last], ext


def generate(spec: CaseSpec, name: str) -> GeneratedCase:
    """Assemble the pencil and derive meta (inertia, scenarios, targets), config and reference."""
    t0 = time.perf_counter()
    nx, ny = resolve_grid(spec)
    mesh, eps_r, mu_r = build_mesh(spec, nx, ny)
    disc = discretize(mesh)
    h = max_edge_length(mesh, disc.topo)
    ne, nn = disc.num_edge_dofs, disc.num_node_dofs
    wave = spec.family == "waveguide"
    if wave:
        k0 = wavenumber(spec.freq_ghz)
        A, B = waveguide_pencil(disc, k0, eps_r, mu_r)
        kt2_shift = k0 * k0 * max(1.0, spec.eps_slab * spec.mu_slab)
    else:
        k0 = 0.0
        A, B = cavity_pencil(disc)
        kt2_shift = 0.0
    n = A.shape[0]
    t_assemble = time.perf_counter() - t0

    listed, validation = _analytic_modes(spec, k0, kt2_shift)
    ext_values = np.array([m.value for m in validation])
    # The zero cluster is exactly 0 in the discrete problem: append it with a zero error bar.
    model = Spectrum.build(np.append(ext_values, 0.0), np.append(error_bound(ext_values, kt2_shift, h), 0.0),
                           0.0, exact=False)
    window = _reference_window(model, len(listed), kt2_shift, wave)
    inertia = _inertia(spec, n, ne, nn, k0, h, model)

    t1 = time.perf_counter()
    targets_from = model
    complex_count = None if spec.loaded else 0     # real spectrum proven for hollow guide / cavity
    if n <= DENSE_LIMIT:
        inertia, complex_count, targets_from = _dense_check(A, B, spec, inertia)
    t_dense = time.perf_counter() - t1
    # The evanescent region of a loaded guide may hold complex pairs: no robust target there.
    targets = _targets(targets_from, wave, evanescent=not spec.loaded, null_dim=nn)
    if not targets:
        raise ValueError("mesh too coarse: no target can be separated from its neighbours")

    meta = {
        "family": spec.family,
        "problem": "GHIEP" if wave else "GHEP",
        "description": _description(spec, ne, nn, sum(m.value < 0 for m in listed)),
        "params": {**{k: v for k, v in asdict(spec).items() if k not in ("nx", "ny")}, "nx": nx, "ny": ny,
                   "slab_width": spec.width, "k0_rad_per_mm": k0 if wave else None,
                   "units": {"length": "mm", "frequency": "GHz", "eigenvalue": "mm^-2"},
                   "generator": "tools.gen.waveguide", "command": command_line(spec, name, nx, ny)},
        "mesh": {"nx": nx, "ny": ny, "nodes": mesh.num_nodes, "triangles": mesh.num_triangles,
                 "interior_edges": ne, "interior_nodes": nn, "h_max_mm": h,
                 "dof_layout": "[interior edges (e_t); interior nodes (e_z)]" if wave else "[interior edges]"},
        "inertia": inertia,
        "complex_eigenvalue_count": complex_count,
        "scenarios": _scenarios(spec, nx, ny, n, listed),
        "targets": [t.as_dict() for t in targets],
        "citation": "J.-F. Lee, D.-K. Sun, Z. J. Cendes, Full-wave analysis of dielectric waveguides using "
                    "tangential vector finite elements, IEEE Trans. MTT 39(8):1262-1271, 1991",
    }
    config = case.default_config(targets[0].sigma, targets[0].k)
    ref = {"eigenvalues": np.array([m.value for m in listed]), "kind": "partial-analytic",
           "note": _reference_note(spec, listed, window, h, kt2_shift, nn, wave),
           "complete": window is not None, "window": window}
    return GeneratedCase(name=name, A=A, B=B, meta=meta, config=config, ref=ref,
                         timings={"assemble_s": t_assemble, "dense_check_s": t_dense})


def _reference_window(model: Spectrum, num_listed: int, kt2_shift: float, wave: bool):
    """Window [lo, hi] in which the listed values are complete for the discrete problem: both
    edges are safe cuts and the zero cluster (never listed) lies outside.  None if none exists.

    Waveguide: everything below the highest safe cut under the zero cluster (all of it listed);
    lo is a lower bound of the whole discrete spectrum according to the model.  Cavity: from the
    cut above the zero cluster to the highest safe cut that leaves only listed values below it
    (the listing is the first num_listed values after the zero entry)."""
    if wave:
        top = model.cut_below_zero()
        lo = min(-kt2_shift, float(np.min(model.values - 2.0 * model.errs)))
        return None if top is None else (float(lo), float(top[1]))
    lo = model.cut_above_zero()
    inside = [c for i, c in model.cuts() if model.zero_last < i <= model.zero_last + num_listed]
    return None if lo is None or not inside else (float(lo), float(inside[-1]))


def _inertia(spec: CaseSpec, n: int, ne: int, nn: int, k0: float, h: float, model: Spectrum) -> dict:
    """Inertia expectations that hold without a dense check (see module docstring); counts that
    depend on where discrete eigenvalues fall relative to k0^2 are only given when the error
    bound guarantees them, otherwise null.  "pencil" counts real eigenvalues by sign; for a loaded
    guide the positive count is unknown (complex pairs may take part of the evanescent region)."""
    if spec.family == "cavity2d":
        return {"A": [0, nn, ne - nn], "B": [0, 0, ne], "pencil": [0, nn, ne - nn]}
    inertia = {"B": [nn, 0, ne], "A": [None, nn, None], "pencil": [None, nn, None]}
    if model.zero_isolated():
        neg = model.zero_first        # model values below the zero cluster
        inertia["pencil"] = [neg, nn, None if spec.loaded else n - nn - neg]
    if not spec.loaded:
        # A_tt = S_tt - k0^2 T_tt: negative on the nn gradients and on the TE modes below k0^2.
        te = np.array([m.value for m in rectangle_cutoffs(spec.a, spec.b, 4.0 * k0 * k0, tm=False)])
        if np.all(np.abs(te - k0 * k0) > error_bound(te, 0.0, h)):
            below = int((te < k0 * k0).sum())
            inertia["A"] = [nn + below, nn, ne - nn - below]
    return inertia


def _dense_check(A, B, spec: CaseSpec, inertia: dict) -> tuple[dict, int, Spectrum]:
    """Small cases: exact discrete spectrum (dense QZ / symmetric-definite solver).  Verifies the
    structural expectations (inertia, real spectrum of the hollow guide and the cavity, complex
    pairs of the loaded guide only above the zero cluster), replaces model-based counts by exact
    ones ("pencil" counts the real eigenvalues by sign) and returns them together with the number
    of complex eigenvalues and the real discrete spectrum (error bars = rounding tolerance) for
    target selection."""
    wave = spec.family == "waveguide"
    lam = dense_pencil_eigenvalues(A, B) if wave else sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    tol = 1e-8 * float(np.abs(lam).max())
    cplx = np.abs(np.imag(lam)) > tol
    if cplx.any() and (not spec.loaded or np.real(lam[cplx]).min() <= tol):
        raise RuntimeError("dense check: complex eigenvalues where the theory predicts real ones")
    real = np.real(lam[~cplx])
    exact = {"A": dense_inertia(A), "B": dense_inertia(B), "pencil": count_signs(real, tol)}
    for key, expected in inertia.items():
        if any(e is not None and e != x for e, x in zip(expected, exact[key])):
            raise RuntimeError(f"dense check: inertia of {key} is {exact[key]}, expected {expected}")
    return exact, int(cplx.sum()), Spectrum.build(real, np.full(real.size, tol), tol, exact=True)


def _targets(spec_model: Spectrum, wave: bool, evanescent: bool, null_dim: int) -> list[Target]:
    """Robust 'nearest k' suggestions (see nearest_block).  Waveguide: all propagating modes and,
    if ``evanescent``, the first evanescent group; cavity2d: the lowest and an interior group."""
    s = spec_model
    cuts = s.cuts()
    basis = ("the dense discrete spectrum" if s.exact else
             "the continuous-model eigenvalues widened by the a-priori discretization error bound")
    robust = f"sigma lies mid-way between two spectral gaps that {basis} cannot close, so the k nearest " \
             "eigenvalues are exactly this group"
    out = []
    zero_cut = s.cut_above_zero()
    if wave:
        top = s.cut_below_zero()
        if top is not None:
            k = top[0] + 1
            # Nothing lies below the block, so its lower edge is its own lowest error-bar end.
            sigma = 0.5 * (float(np.min(s.values[:k] - s.errs[:k])) + top[1])
            what = ("all propagating modes (lambda = -beta^2 < 0)" if k == s.zero_first else
                    f"the lowest {k} of the {s.zero_first} propagating modes (the others are not separable "
                    "from the zero cluster on this mesh)")
            out.append(Target(sigma, k, f"{what}; {robust}; the zero cluster (lambda = 0) is excluded."))
        if evanescent and zero_cut is not None:
            blk = nearest_block(cuts, s.zero_last + 1, 6, zero_cut)
            if blk is not None:
                out.append(Target(blk[0], blk[1], "the least evanescent modes just above the zero cluster "
                                                  f"(lambda > 0, close to cutoff); {robust}."))
        return out
    if zero_cut is not None:
        blk = nearest_block(cuts, s.zero_last + 1, 6, zero_cut)
        if blk is not None:
            out.append(Target(blk[0], blk[1], "the lowest nonzero TE cutoffs kc^2 next to the "
                                              f"{null_dim}-fold zero cluster; {robust}."))
        start = [(i, c) for i, c in cuts if i >= s.zero_last + 30]
        if start:
            blk = nearest_block(cuts, start[0][0] + 1, 6, start[0][1])
            if blk is not None:
                out.append(Target(blk[0], blk[1], f"an interior group of cutoffs (about the 30th "
                                                  f"nonzero eigenvalue on); {robust}."))
    return out


def _scenarios(spec: CaseSpec, nx: int, ny: int, n: int, listed: list[Mode]) -> list[str]:
    tags = ["indefinite-B", "zero-cluster"] if spec.family == "waveguide" else ["zero-cluster"]
    d4 = (not spec.loaded and spec.a == spec.b and nx == ny and nx % 2 == 0
          and spec.pattern == "unionjack" and spec.perturb == 0.0)
    if d4:
        tags.append("multiplicity")   # symmetry-protected exact double eigenvalues (e.g. TE1,0/TE0,1)
    v = np.array([m.value for m in listed])
    if np.any(np.diff(v) <= 1e-9 * np.maximum(1.0, np.abs(v[1:]))):
        tags.append("clustered")      # continuum degeneracies that the mesh splits at O(h^2)
    if n >= LARGE_N:
        tags += ["large", "memory"]
    return tags


def _description(spec: CaseSpec, ne: int, nn: int, nprop: int) -> str:
    if spec.family == "cavity2d":
        return (f"2-D TE cutoff problem S_tt x = kc^2 T_tt x (Whitney edge elements, PEC) of a {spec.a} x {spec.b} mm "
                f"rectangle: GHEP with B HPD and a {nn}-fold zero eigenvalue (discrete gradients).")
    fill = (f"loaded with an eps_r = {spec.eps_slab}, mu_r = {spec.mu_slab} slab on 0 < x < {spec.width} mm"
            if spec.loaded else "air-filled")
    text = (f"Lee-Sun-Cendes FEM (Whitney edges + P1 nodes) of a {spec.a} x {spec.b} mm PEC waveguide, {fill}, "
            f"at {spec.freq_ghz} GHz: A x = lambda B x with lambda = -beta^2 [mm^-2]; A and B indefinite "
            f"(B: {nn} negative / {ne} positive), {nn}-fold zero eigenvalue (null space of A, non-physical), "
            f"{nprop} propagating modes in the continuous model.")
    if spec.loaded:
        text += (" The evanescent part (lambda > 0) contains complex-conjugate eigenvalue pairs: LSE/LSM branches "
                 "of opposite B-sign characteristic that the discretization couples where they nearly cross.")
    return text


def _reference_note(spec: CaseSpec, listed: list[Mode], window, h: float, kt2_shift: float, nn: int,
                    wave: bool) -> str:
    values = np.array([m.value for m in listed])
    err = error_bound(values, kt2_shift, h)
    what = ("lambda = -beta^2 of all modes with lambda <= k0^2 max(eps mu)" if wave else
            f"the lowest {len(listed)} nonzero TE cutoffs kc^2")
    family = ("closed form" if not spec.loaded and wave else
              "LSE/LSM transcendental equations (Brent)" if wave else "closed form (m pi/a)^2 + (n pi/b)^2")
    win = (f"complete=true: every discrete eigenvalue in the window [{window[0]:.6g}, {window[1]:.6g}] "
           "corresponds to a listed value (both edges lie in gaps the error bound cannot close)"
           if window is not None else "complete=false: the mesh is too coarse to certify any window")
    return (f"Continuous-model (partial-analytic) values, {family}: {what}, in mm^-2. The discrete "
            f"eigenvalues differ by O(h^2): |lambda_h - lambda| <= {ERR_CONST} h^2 (lambda + {kt2_shift:.6g})^2 "
            f"with h = {h:.6g} mm (largest bound over the listing {err.max():.3g}, at the lowest mode "
            f"{err.min():.3g}); compare with that tolerance, not with the solver tolerance. The {nn}-fold zero "
            f"eigenvalue of the discrete pencil (non-physical null space) is not listed. {win}. "
            + ("Positive (evanescent) listed values may appear as complex-conjugate pairs in the discrete pencil "
               "(see meta complex_eigenvalue_count). " if spec.loaded else "")
            + f"Labels (ascending): {' '.join(m.label for m in listed)}.")


def write(gen: GeneratedCase, root: str | Path = case.DEFAULT_ROOT) -> tuple[Path, Path]:
    """Write case and reference through tools/case.py; returns (case dir, ref dir)."""
    case_dir = case.save_case(gen.name, gen.A, gen.B, meta=gen.meta, config=gen.config, root=root)
    ref_dir = case.save_ref(gen.name, root=root, **gen.ref)
    return case_dir, ref_dir


# ------------------------------------------------------------------------------------- CLI

def parse_args(argv=None) -> tuple[CaseSpec, str, Path]:
    ap = argparse.ArgumentParser(prog="python -m tools.gen.waveguide", description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=FAMILIES, default="waveguide")
    ap.add_argument("--name", required=True, help="case name (directory under <root>/cases and <root>/refs)")
    ap.add_argument("--a", type=float, default=10.0, help="width along x [mm]")
    ap.add_argument("--b", type=float, default=10.0, help="height along y [mm]")
    ap.add_argument("--freq", type=float, default=50.0, help="frequency [GHz] (waveguide)")
    ap.add_argument("--eps-slab", type=float, default=1.0, help="slab relative permittivity (waveguide)")
    ap.add_argument("--mu-slab", type=float, default=1.0, help="slab relative permeability (waveguide)")
    ap.add_argument("--slab-width", type=float, default=None, help="slab on 0 < x < width [mm], default a/2")
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--nx", type=int, help="cells along x")
    size.add_argument("--target-n", type=int, help="choose the grid so that n is closest to this")
    ap.add_argument("--ny", type=int, default=None, help="cells along y (with --nx; default square cells)")
    ap.add_argument("--pattern", choices=PATTERNS, default="unionjack")
    ap.add_argument("--perturb", type=float, default=0.0, help="random interior node shift, fraction of h, < 0.25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=case.DEFAULT_ROOT, help="data root (default $HS_DATA_ROOT)")
    ns = ap.parse_args(argv)
    spec = CaseSpec(family=ns.family, a=ns.a, b=ns.b, freq_ghz=ns.freq, eps_slab=ns.eps_slab, mu_slab=ns.mu_slab,
                    slab_width=ns.slab_width, nx=ns.nx, ny=ns.ny, target_n=ns.target_n, pattern=ns.pattern,
                    perturb=ns.perturb, seed=ns.seed)
    return spec, ns.name, ns.root


def main(argv=None) -> int:
    spec, name, root = parse_args(argv)
    gen = generate(spec, name)
    t0 = time.perf_counter()
    case_dir, ref_dir = write(gen, root)
    sizes = {p.name: p.stat().st_size for d in (case_dir, ref_dir) for p in sorted(d.iterdir())}
    print(json.dumps({"name": name, "n": gen.A.shape[0], "nnz_A": gen.A.nnz, "nnz_B": gen.B.nnz,
                      **gen.timings, "write_s": time.perf_counter() - t0, "case_dir": str(case_dir),
                      "file_bytes": sizes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
