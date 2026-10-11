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
  whole spectrum is real.  Each family is a symmetric-definite problem (the cavity2d pencil is
  the TE family alone), which the error model below exploits.
* Sign characteristic: for a simple real eigenvalue with eigenvector x = [e; z], differentiating
  A x = lambda B x in k0^2 gives
      d lambda / d k0^2 = -(e^H T_tt^eps e - lambda z^H T_zz^eps z) / x^H B x,
  and the numerator is negative for lambda < 0.  A propagating mode therefore has positive type
  (x^H B x > 0) exactly when it is a forward wave (beta^2 grows with frequency).
* Slab-loaded guide: every continuous LSE / LSM family is a regular Sturm-Liouville problem in
  lambda (weight 1/mu resp. 1/eps), so the continuous spectrum is real, and lambda decreases
  with k0^2: all continuous propagating modes are forward waves of positive type.  The evanescent
  part (lambda > 0) mixes both sign characteristics (x^H B x > 0 for LSE-like, < 0 for LSM-like
  modes).  Where two branches of opposite type nearly cross, the discretization couples them and
  they collide into complex-conjugate pairs (dense checks, eps_r = 2.25: 24 / 46 / 52 complex
  eigenvalues for 8 / 12 / 16 cells per side, all with Re lambda > 0.46).  The discrete
  propagating modes inherit the positive type only on meshes that resolve them: on coarse meshes
  of high-contrast slabs a discrete propagating eigenvalue can have negative type (eps_r = 30 on
  a 2 mm slab, 12 cells) or join a complex pair (eps_r = 30 on a 1 mm slab, 16 cells: a pair at
  Re lambda = -0.149).  The dense check records such pairs and targets keep clear of them.

Units: lengths in mm and frequency in GHz, so k0 [rad/mm] = 2 pi f / c with c = 299.792458 mm/ns
and every eigenvalue is in mm^-2.  This reproduces the competition's real case (10 mm square air
guide at 50 GHz: 18 propagating modes with lambda in about [-1, 0)).

Reference values are those of the continuous model (kind "partial-analytic"); the discrete
eigenvalues differ by O(h^2), bounded a priori by ``error_bound``.  Reference windows and target
suggestions are placed in spectral gaps that this bound cannot close (``safe_cuts``), so their
counts are valid for the discrete problem.  Cases with n <= DENSE_LIMIT are additionally checked
with dense eigensolvers: their exact discrete inertia is recorded, the reference window is
verified and the targets are taken from the exact spectrum.

Usage: python -m tools.gen.waveguide --family waveguide --a 10 --b 10 --freq 50 --target-n 1000 \
           --name wg_square_1e3            (see --help; output via tools/case.py)
"""
from __future__ import annotations

import argparse
import json
import operator
import shlex
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
WAVEGUIDE_EVANESCENT = 16   # waveguide references list at least this many evanescent modes
# A-priori bound |lambda_h - lambda| <= ERR_CONST * h^2 * kt^4 with h the longest mesh edge and
# kt^2 = lambda + k0^2 max(eps mu) (waveguide) or kt^2 = kc^2 (cavity2d), the transverse
# wavenumber of the mode, for the k-th discrete eigenvalue against the k-th continuous one (in
# ascending order, per DISCRETE_FAMILY where the discrete problem decouples).  Measured constants
# on union-jack / diagonal meshes, unperturbed and perturbed (0.24), 16 and 32 cells per side
# (tests/py/test_gen_waveguide.py re-checks the bound): <= 0.025 Nedelec TE family (cavity2d),
# <= 0.077 P1 TM family, <= 0.069 hollow guide (both families merged), <= 0.047 slab-loaded guide
# (eps_r 2.25; eps_r 4 with mu_r 2; mu_r 3); 0.15 leaves a margin of about 2.
ERR_CONST = 0.15
# Mode families that the discrete problem keeps apart exactly (hollow guide and cavity2d, see the
# module docstring), each the sorted spectrum of one symmetric-definite problem: TE = nonzero
# Nedelec curl-curl eigenvalues, TM = P1 Dirichlet-Laplacian eigenvalues.  The LSE / LSM modes of a
# loaded guide are coupled by the discretization and belong to no family (see safe_cuts).
DISCRETE_FAMILY = {"TE": 0, "TM": 1}
NO_FAMILY = -1


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


def safe_cuts(values: np.ndarray, errs: np.ndarray, family: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Cuts that no perturbed eigenvalue can cross, for ascending values with error bars.

    A cut after index i separates values[:i+1] from values[i+1:].  A value without a family
    (family[j] < 0, or ``family`` None) is only known to have a discrete counterpart within its
    bar, so each one constrains the cut: max_{j<=i}(v_j + e_j) < cut < min_{j>i}(v_j - e_j).
    The members of a family f >= 0 are matched to the sorted eigenvalues of one symmetric-definite
    discrete problem in order (k-th to k-th), so the discrete members below the cut are bounded
    by the last continuous member below it and those above by the first member above it.  Only
    these two members constrain the cut.  The bars of high, poorly resolved members can therefore
    not close gaps far below them, although they grow like kt^4.  A family continues beyond its
    last validated member, so no cut may lie above that member.  Each cut is the midpoint of its
    free interval.  Returns [(i, cut)] for every gap."""
    v = np.asarray(values, dtype=float)
    e = np.asarray(errs, dtype=float)
    fam = np.full(v.size, NO_FAMILY) if family is None else np.asarray(family)
    free = fam < 0
    upper = np.maximum.accumulate(np.where(free, v + e, -np.inf))                 # j <= i
    lower = np.minimum.accumulate(np.where(free, v - e, np.inf)[::-1])[::-1]
    lower = np.append(lower[1:], np.inf)                                         # j > i
    index = np.arange(v.size)
    for f in np.unique(fam[~free]):
        members = np.nonzero(fam == f)[0]
        last = np.searchsorted(members, index, side="right") - 1                 # -1: none <= i
        nxt = last + 1
        upper = np.maximum(upper, np.where(last >= 0, (v + e)[members[np.maximum(last, 0)]], -np.inf))
        # No validated member above the cut: the family's continuation is unknown, refuse.
        lower = np.minimum(lower, np.where(nxt < members.size,
                                           (v - e)[members[np.minimum(nxt, members.size - 1)]], -np.inf))
    return [(i, 0.5 * (upper[i] + lower[i])) for i in range(v.size - 1) if upper[i] < lower[i]]


@dataclass(frozen=True)
class Target:
    """A suggested 'nearest k to sigma' query (meta.json targets)."""
    sigma: float
    k: int
    why: str

    def as_dict(self) -> dict:
        return {"sigma": float(self.sigma), "k": int(self.k), "why": self.why}


def disk_is_clear(obstacles, lower_edge: float, cut: float) -> bool:
    """No obstacle (complex eigenvalue) lies in the closed disk with diameter [lower_edge, cut],
    i.e. at distance <= radius from the 'nearest k' shift sigma = midpoint."""
    distance = np.abs(np.asarray(obstacles) - 0.5 * (lower_edge + cut))
    return not np.any(distance <= 0.5 * (cut - lower_edge))


def nearest_block(cuts: list[tuple[int, float]], first: int, min_size: int, lower_edge: float,
                  obstacles=()) -> tuple[float, int] | None:
    """(sigma, k) of a robust 'nearest k' query for the block of sorted real values starting at
    index ``first`` (all of them above ``lower_edge``): the block ends at the first safe cut
    leaving at least ``min_size`` values in it and sigma is the midpoint of [lower_edge, cut].
    Every block member then lies strictly closer to sigma than every other real eigenvalue, even
    after the worst-case discretization shift; ``obstacles`` (complex eigenvalues) must lie
    outside that disk too.  Larger blocks have disks containing this one, so an obstacle inside
    ends the search.  None if no block qualifies."""
    for i, c in cuts:
        if i - first + 1 >= min_size:
            return (0.5 * (lower_edge + c), i - first + 1) if disk_is_clear(obstacles, lower_edge, c) else None
    return None


@dataclass(frozen=True)
class Spectrum:
    """Sorted eigenvalue model with error bars, the zero cluster included as exact zeros."""
    values: np.ndarray
    errs: np.ndarray
    family: np.ndarray   # DISCRETE_FAMILY id per value, NO_FAMILY if none (see safe_cuts)
    zero_first: int      # index range [zero_first, zero_last] of the zero cluster
    zero_last: int
    exact: bool          # True: dense discrete spectrum; False: continuous model + error bound

    @staticmethod
    def build(values, errs, zero_tol: float, exact: bool, family=None) -> "Spectrum":
        order = np.argsort(values, kind="stable")
        v = np.asarray(values, dtype=float)[order]
        e = np.asarray(errs, dtype=float)[order]
        fam = np.full(v.size, NO_FAMILY) if family is None else np.asarray(family, dtype=int)[order]
        zeros = np.nonzero(np.abs(v) <= zero_tol)[0]
        return Spectrum(v, e, fam, int(zeros[0]), int(zeros[-1]), exact)

    def cuts(self) -> list[tuple[int, float]]:
        return safe_cuts(self.values, self.errs, self.family)

    def floor(self, widen: float = 1.0) -> float:
        """Lower bound of the whole discrete spectrum according to the model, with the bars
        multiplied by ``widen``: the lowest bar end over the values without a family and the
        lowest member of each family (order matching puts the other members above it)."""
        relevant = self.family < 0
        for f in np.unique(self.family[~relevant]):
            relevant[np.argmax(self.family == f)] = True
        return float(np.min((self.values - widen * self.errs)[relevant]))

    def cut_below_zero(self) -> tuple[int, float] | None:
        below = [(i, c) for i, c in self.cuts() if i < self.zero_first]
        return below[-1] if below else None

    def cut_above_zero(self) -> float | None:
        above = [c for i, c in self.cuts() if i == self.zero_last]
        return above[0] if above else None

    def zero_isolated(self) -> bool:
        """The zero cluster is separated from both neighbours (or nothing lies below it), so
        the counts below / above it are those of the discrete problem."""
        below = self.cut_below_zero()
        clear_below = self.zero_first == 0 or (below is not None and below[0] == self.zero_first - 1)
        return clear_below and self.cut_above_zero() is not None


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
        # Plain Python numbers (e.g. from numpy scalars of a scripted sweep), so that params and the
        # recorded command line hold exactly round-tripping values; integers must be integral.
        for name in ("a", "b", "freq_ghz", "eps_slab", "mu_slab", "slab_width", "perturb"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("nx", "ny", "target_n", "seed"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, operator.index(getattr(self, name)))
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.a <= 0 or self.b <= 0 or self.freq_ghz <= 0 or self.eps_slab <= 0 or self.mu_slab <= 0:
            raise ValueError("a, b, freq, eps_slab and mu_slab must be positive")
        if self.family == "cavity2d" and (self.eps_slab, self.mu_slab) != (1.0, 1.0):
            raise ValueError("cavity2d is the hollow TE cutoff problem: no slab")
        if self.slab_width is not None and not 0.0 < self.slab_width <= self.a:
            raise ValueError("slab_width must lie in (0, a]; an air-filled guide keeps eps_slab = mu_slab = 1")
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
        return self.a / 2.0 if self.slab_width is None else self.slab_width

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
    """Shell command (POSIX quoting) that regenerates the case bit for bit: the resolved grid and
    every input as given, floats as round-tripping reprs, options in --key=value form so that no
    value (e.g. a name starting with '-') can be taken for an option."""
    args = {"family": spec.family, "a": repr(spec.a), "b": repr(spec.b)}
    if spec.family == "waveguide":
        args.update({"freq": repr(spec.freq_ghz), "eps-slab": repr(spec.eps_slab), "mu-slab": repr(spec.mu_slab)})
        if spec.slab_width is not None:
            args["slab-width"] = repr(spec.slab_width)
    args.update({"nx": str(nx), "ny": str(ny), "pattern": spec.pattern, "perturb": repr(spec.perturb),
                 "seed": str(spec.seed), "name": name})
    return shlex.join(["python", "-m", "tools.gen.waveguide", *(f"--{k}={v}" for k, v in args.items())])


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
    """(listed, validation) continuous-model modes, ties included.  Waveguide: listed = every
    propagating mode and the evanescent ones up to lambda = k0^2 max(eps mu), but at least the
    WAVEGUIDE_EVANESCENT least evanescent modes (below cutoff the guide has no propagating mode);
    cavity2d: the lowest CAVITY_LISTED kc^2.  The validation set extends to twice the listed kt^2
    range so that window edges near the end of the listing are checked against the next
    unlisted eigenvalues too."""
    if spec.family == "waveguide":
        def modes(lam_max: float) -> list[Mode]:
            if spec.loaded:
                return slab_guide_modes(spec.a, spec.b, k0, spec.width, spec.eps_slab, spec.mu_slab, lam_max)
            return hollow_guide_modes(spec.a, spec.b, k0, lam_max)

        lam = max(kt2_shift, (np.pi / max(spec.a, spec.b)) ** 2)
        while len(positive := [m.value for m in modes(lam) if m.value > 0.0]) < WAVEGUIDE_EVANESCENT:
            lam *= 1.5
        lam_list = max(kt2_shift, sorted(positive)[WAVEGUIDE_EVANESCENT - 1])
        ext = modes(2.0 * lam_list + kt2_shift)      # kt^2 = lambda + kt2_shift doubled
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
        kt2_shift = k0 * k0 * (max(1.0, spec.eps_slab * spec.mu_slab) if spec.loaded else 1.0)
    else:
        k0 = 0.0
        A, B = cavity_pencil(disc)
        kt2_shift = 0.0
    n = A.shape[0]
    t_assemble = time.perf_counter() - t0

    listed, validation = _analytic_modes(spec, k0, kt2_shift)
    listed_values = np.array([m.value for m in listed])
    ext_values = np.array([m.value for m in validation])
    family = [DISCRETE_FAMILY.get(m.label[:2], NO_FAMILY) for m in validation]
    # The zero cluster is exactly 0 in the discrete problem: append it with a zero error bar.
    model = Spectrum.build(np.append(ext_values, 0.0), np.append(error_bound(ext_values, kt2_shift, h), 0.0),
                           0.0, exact=False, family=np.append(family, NO_FAMILY))
    propagating = wave and bool(np.any(listed_values < 0.0))
    # The evanescent part of a loaded guide may hold complex-conjugate pairs that only a dense
    # check can locate (module docstring): the model certifies no window or target there.
    evanescent_real = not spec.loaded
    window = (_reference_window(model, listed_values, kt2_shift, below_zero=propagating)
              if propagating or evanescent_real else None)
    inertia = _inertia(spec, n, ne, nn, k0, h, model)

    t1 = time.perf_counter()
    targets_from = model
    complex_values = None if spec.loaded else np.empty(0)   # real spectrum proven for hollow guide / cavity
    if n <= DENSE_LIMIT:
        inertia, complex_values, targets_from = _dense_check(A, B, spec, inertia, window, listed_values)
    t_dense = time.perf_counter() - t1
    obstacles = np.empty(0) if complex_values is None else complex_values
    targets = _targets(targets_from, wave, evanescent=evanescent_real or targets_from.exact, null_dim=nn,
                       obstacles=obstacles)
    if not targets:
        raise ValueError(_refusal(spec, targets_from, h, propagating))

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
        # Unknown (null) for a loaded guide without dense check; min real part null if none.
        "complex_eigenvalue_count": None if complex_values is None else int(complex_values.size),
        "complex_eigenvalue_min_real": (float(np.real(complex_values).min())
                                        if complex_values is not None and complex_values.size else None),
        "scenarios": _scenarios(spec, nx, ny, n, listed),
        "targets": [t.as_dict() for t in targets],
        "citation": "J.-F. Lee, D.-K. Sun, Z. J. Cendes, Full-wave analysis of dielectric waveguides using "
                    "tangential vector finite elements, IEEE Trans. MTT 39(8):1262-1271, 1991",
    }
    config = case.default_config(targets[0].sigma, targets[0].k)
    ref = {"eigenvalues": listed_values, "kind": "partial-analytic",
           "note": _reference_note(spec, listed, window, h, kt2_shift, nn, wave, propagating),
           "complete": window is not None, "window": window}
    return GeneratedCase(name=name, A=A, B=B, meta=meta, config=config, ref=ref,
                         timings={"assemble_s": t_assemble, "dense_check_s": t_dense})


def _reference_window(model: Spectrum, listed: np.ndarray, kt2_shift: float, below_zero: bool):
    """Window [lo, hi] in which the listed values are complete for the discrete problem: both
    edges are safe cuts and the zero cluster (never listed) lies outside.  None if none exists.

    below_zero (waveguide with propagating modes): everything below the highest safe cut under
    the zero cluster (all of it listed); lo lies below the continuous spectrum (-kt2_shift) and
    below the model's lower bound of the discrete one (bars doubled for margin).  Otherwise
    (cavity2d, hollow guide below cutoff): from the cut above the zero cluster to the highest
    safe cut that leaves only listed values below it (the listing holds the lowest positive
    values of the model)."""
    if below_zero:
        top = model.cut_below_zero()
        lo = min(-kt2_shift, model.floor(widen=2.0))
        return None if top is None else (float(lo), float(top[1]))
    lo = model.cut_above_zero()
    num_positive = int((listed > 0.0).sum())
    inside = [c for i, c in model.cuts() if model.zero_last < i <= model.zero_last + num_positive]
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


def _dense_check(A, B, spec: CaseSpec, inertia: dict, window, listed: np.ndarray) -> tuple[dict, np.ndarray, Spectrum]:
    """Small cases: exact discrete spectrum (dense QZ / symmetric-definite solver).  Verifies the
    model-based claims (inertia, the reference window, the real spectrum of the hollow guide and
    the cavity) and replaces the model-based counts by exact ones ("pencil" counts the real
    eigenvalues by sign).  Returns them with the complex eigenvalues and with the real discrete
    spectrum (error bars = rounding tolerance) for target selection.  Only a loaded guide has
    complex eigenvalues: in its evanescent part and, on coarse meshes of high-contrast slabs,
    also among its propagating modes (module docstring)."""
    wave = spec.family == "waveguide"
    lam = dense_pencil_eigenvalues(A, B) if wave else sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    tol = 1e-8 * float(np.abs(lam).max())
    cplx = np.abs(np.imag(lam)) > tol
    if cplx.any() and not spec.loaded:
        raise RuntimeError("dense check: complex eigenvalues, but the hollow guide and the cavity have a real spectrum")
    real = np.real(lam[~cplx])
    complex_values = lam[cplx]
    exact = {"A": dense_inertia(A), "B": dense_inertia(B), "pencil": count_signs(real, tol)}
    for key, expected in inertia.items():
        if any(e is not None and e != x for e, x in zip(expected, exact[key])):
            raise RuntimeError(f"dense check: inertia of {key} is {exact[key]}, expected {expected}")
    if window is not None:
        lo, hi = window
        found = int(((real > lo) & (real < hi)).sum())
        expected = int(((listed > lo) & (listed < hi)).sum())
        if found != expected or np.any((np.real(complex_values) > lo) & (np.real(complex_values) < hi)):
            raise RuntimeError(f"dense check: the reference window ({lo:.6g}, {hi:.6g}) holds {found} real "
                               f"eigenvalues (or complex ones) but {expected} listed values")
    return exact, complex_values, Spectrum.build(real, np.full(real.size, tol), tol, exact=True)


def _targets(spec_model: Spectrum, wave: bool, evanescent: bool, null_dim: int,
             obstacles: np.ndarray) -> list[Target]:
    """Robust 'nearest k' suggestions (see nearest_block) that keep the complex eigenvalues
    ``obstacles`` out of their disks.  Waveguide: all propagating modes (or the largest lowest
    group that can be separated) and, if ``evanescent``, the first evanescent group; cavity2d:
    the lowest and an interior group."""
    s = spec_model
    cuts = s.cuts()
    basis = ("the dense discrete spectrum" if s.exact else
             "the continuous-model eigenvalues widened by the a-priori discretization error bound")
    robust = f"sigma lies mid-way between two spectral gaps that {basis} cannot close, so the k nearest " \
             "eigenvalues are exactly this group"
    if obstacles.size:
        robust += " (every complex eigenvalue lies farther from sigma)"
    out = []
    zero_cut = s.cut_above_zero()
    if wave:
        # Largest block from the bottom first; disks shrink with the block.
        for i, c in reversed([(i, c) for i, c in cuts if i < s.zero_first]):
            k = i + 1
            # Nothing lies below the block, so its lower edge is its own lowest error-bar end.
            low = float(np.min(s.values[:k] - s.errs[:k]))
            if not disk_is_clear(obstacles, low, c):
                continue
            hurdle = "the zero cluster or from complex eigenvalues" if obstacles.size else "the zero cluster"
            what = ("all propagating modes (lambda = -beta^2 < 0)" if k == s.zero_first else
                    f"the lowest {k} of the {s.zero_first} propagating modes (the others are not separable "
                    f"from {hurdle} on this mesh)")
            out.append(Target(0.5 * (low + c), k, f"{what}; {robust}; the zero cluster (lambda = 0) is excluded."))
            break
        if evanescent and zero_cut is not None:
            blk = nearest_block(cuts, s.zero_last + 1, 6, zero_cut, obstacles)
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
        text += (" The discrete pencil may contain complex-conjugate eigenvalue pairs (see complex_eigenvalue_count): "
                 "LSE/LSM branches of opposite B-sign characteristic that the discretization couples where they "
                 "nearly cross, in the evanescent part (lambda > 0) and, on coarse meshes of high-contrast slabs, "
                 "also among the propagating modes.")
    return text


def _refusal(spec: CaseSpec, s: Spectrum, h: float, propagating: bool) -> str:
    """Why no robust target exists (generate raises ValueError with this message)."""
    if s.exact:
        return ("no robust target: no group of real eigenvalues of the dense discrete spectrum is separated from "
                "the zero cluster and from the complex eigenvalues")
    if spec.loaded and not propagating:
        return (f"no robust target: the loaded guide has no propagating mode at {spec.freq_ghz} GHz, and its "
                "evanescent spectrum may hold complex-conjugate pairs that only the dense check "
                f"(n <= {DENSE_LIMIT}) locates")
    return (f"no robust target: for n > {DENSE_LIMIT} targets rest on the a-priori error bound {ERR_CONST} h^2 kt^4 "
            f"(h = {h:.3g} mm), which on this mesh is too wide to separate any target group from its neighbours; "
            f"refine the mesh (the bound shrinks like h^2) or use n <= {DENSE_LIMIT} (exact dense spectrum)")


def _reference_note(spec: CaseSpec, listed: list[Mode], window, h: float, kt2_shift: float, nn: int,
                    wave: bool, propagating: bool) -> str:
    values = np.array([m.value for m in listed])
    err = error_bound(values, kt2_shift, h)
    what = (f"lambda = -beta^2 of all propagating modes and of the least evanescent ones (lambda <= "
            f"{values.max():.6g})" if wave else f"the lowest {len(listed)} nonzero TE cutoffs kc^2")
    family = ("closed form" if not spec.loaded and wave else
              "LSE/LSM transcendental equations (Brent)" if wave else "closed form (m pi/a)^2 + (n pi/b)^2")
    if window is not None:
        win = (f"complete=true: every discrete eigenvalue in the window [{window[0]:.6g}, {window[1]:.6g}] "
               "corresponds to a listed value (both edges lie in gaps the error bound cannot close)")
    elif spec.loaded and not propagating:
        win = ("complete=false: no propagating mode at this frequency, and the evanescent spectrum of a loaded "
               "guide may hold complex-conjugate pairs")
    else:
        win = "complete=false: the mesh is too coarse to certify any window"
    return (f"Continuous-model (partial-analytic) values, {family}: {what}, in mm^-2. The discrete "
            f"eigenvalues differ by O(h^2): |lambda_h - lambda| <= {ERR_CONST} h^2 (lambda + {kt2_shift:.6g})^2 "
            f"with h = {h:.6g} mm (largest bound over the listing {err.max():.3g}, at the lowest mode "
            f"{err.min():.3g}); compare with that tolerance, not with the solver tolerance. The {nn}-fold zero "
            f"eigenvalue of the discrete pencil (non-physical null space) is not listed. {win}. "
            + ("Listed values outside the window, evanescent ones in particular, may appear as complex-conjugate "
               "pairs in the discrete pencil (see meta complex_eigenvalue_count / complex_eigenvalue_min_real). "
               if spec.loaded else "")
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
