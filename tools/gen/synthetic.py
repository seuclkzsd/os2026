"""Synthetic Hermitian eigenproblems with exactly known spectra, plus a memory-stress family.

Families (all matrices are written through tools/case.py):
  (a) Kronecker-sum Dirichlet Laplacians in 2-D / 3-D (unit or given spacing), shifted by c so that
      A = L - c I is indefinite. Equal sizes and spacings give exact multiplicities.
  (b) Prescribed-spectrum Hermitian matrices, real or complex: diag(lambda) under two staggered
      layers of random block-unitary similarities and a random symmetric permutation. Presets:
      dense cluster near zero, many multiplicities, near-singular target, wide dynamic range.
  (c) Generalized pairs A x = lambda B x with B sparse HPD: A = W^H M W, B = W^H W for a sparse,
      well-conditioned W and a prescribed-spectrum M, so the pencil's eigenvalues are those of M.
  (d) "memory": 3-D tensor-product stencils (27-point trilinear FEM, 125-point fourth-order) whose
      sparse factors fill heavily at n ~ 1e5..1e6. Their spectra are exact too (shared sine basis).

Every exact spectrum is saved as a reference of kind 'discrete-exact' with complete=True and
window=None (the whole spectrum is listed). "Exact" means: the eigenvalues of the matrix as it is
stored, up to the rounding committed while assembling it (documented per family in ref.json).

The module also hosts small exact-spectrum utilities reused by tools/gen/yee.py: distinct levels,
unambiguous "nearest k" targets, inertia counts, case writing and the preset command line.

Usage (from the repository root):
    python -m tools.gen.synthetic --list
    python -m tools.gen.synthetic syn-lap3d-22 syn-zero-cluster-real-1e4 [--root DIR]
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp

from tools import case

FAMILY = "synthetic"

# ============================================================================ spectrum utilities


def distinct_levels(eigenvalues, rtol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distinct levels of a spectrum and their multiplicities.

    Neighbours closer than rtol relative to their magnitude form one level. Exact multiplicities
    are bitwise equal in our formulas; the tolerance additionally merges mathematically equal
    values that differ in the last bits (accidental degeneracies evaluated along different paths).
    """
    w = np.sort(np.asarray(eigenvalues, dtype=float))
    if w.size == 0:
        return w, np.zeros(0, dtype=np.int64)
    new_level = np.diff(w) > rtol * np.maximum(np.abs(w[1:]), np.abs(w[:-1]))
    starts = np.concatenate(([0], np.nonzero(new_level)[0] + 1))
    counts = np.diff(np.concatenate((starts, [w.size])))
    return w[starts], counts


def nearest_k_closed(eigenvalues, sigma: float, k_min: int, rel_gap: float = 1e-6) -> int:
    """Smallest k >= k_min for which "the k eigenvalues nearest to sigma" is unambiguous.

    Distances to sigma tie for multiplets and for eigenvalues mirrored around sigma; a target
    whose answer depends on tie-breaking cannot be evaluated. We require the k-th and (k+1)-th
    distances to differ by more than rel_gap * (|sigma| + d_(k+1)), i.e. well above the eigenvalue
    accuracy a backward-stable solver reaches at the default tolerance.
    """
    d = np.sort(np.abs(np.asarray(eigenvalues, dtype=float) - sigma))
    if not 1 <= k_min <= d.size:
        raise ValueError(f"k_min={k_min} outside [1, {d.size}]")
    separated = (d[1:] - d[:-1]) > rel_gap * (abs(sigma) + d[1:])
    hits = np.nonzero(separated[k_min - 1:])[0]
    return k_min + int(hits[0]) if hits.size else int(d.size)


def target_between_levels(eigenvalues, level: int, k_min: int, why: str, *, frac: float = 0.4,
                          k_max: int = 500) -> dict[str, Any]:
    """A 'nearest k' target with sigma = lev[level] + frac * (lev[level+1] - lev[level]).

    frac != 0.5 keeps sigma off the midpoint (equidistant neighbours); k is enlarged to the next
    unambiguous count. k_max guards against a preset whose k would swallow a huge cluster.
    """
    lev, _ = distinct_levels(eigenvalues)
    if not 0 <= level < lev.size - 1:
        raise ValueError(f"level {level} outside [0, {lev.size - 2}]")
    sigma = float(lev[level] + frac * (lev[level + 1] - lev[level]))
    return target_at(eigenvalues, sigma, k_min, why, k_max=k_max)


def target_at(eigenvalues, sigma: float, k_min: int, why: str, *, k_max: int = 500) -> dict[str, Any]:
    """A 'nearest k' target at a given sigma with the smallest unambiguous k >= k_min."""
    k = nearest_k_closed(eigenvalues, sigma, k_min)
    if k > k_max:
        raise ValueError(f"target at sigma={sigma:.6g} needs k={k} > {k_max} to be unambiguous")
    return {"sigma": float(sigma), "k": int(k), "why": why}


def inertia_counts(eigenvalues, zero_tol: float = 0.0) -> list[int]:
    """[negative, zero, positive] counts with |lambda| <= zero_tol counted as zero."""
    w = np.asarray(eigenvalues, dtype=float)
    zero = int(np.count_nonzero(np.abs(w) <= zero_tol))
    neg = int(np.count_nonzero(w < -zero_tol))
    return [neg, zero, int(w.size) - neg - zero]


def hermitian_part(m: sp.spmatrix) -> sp.csr_matrix:
    """(M + M^H) / 2 in CSR without stored zeros.

    Assembling through sparse products leaves O(eps) asymmetry; the averaged matrix is Hermitian
    bitwise (entry (j,i) is computed from the same two operands as entry (i,j)), which is what the
    solver's single-triangle factorization and the full-matrix residuals both assume.
    """
    m = sp.csr_matrix(m)
    h = ((m + m.conj().T) * 0.5).tocsr()
    h.sum_duplicates()
    h.eliminate_zeros()
    h.sort_indices()
    return h


def drop_generated_reference(name: str, root: str | Path) -> None:
    """Remove an exact reference written by an earlier generation of the same case (the case is
    now generated without one). References of other kinds belong to the tools that computed them
    (e.g. solver references for cases without an exact spectrum) and are left alone."""
    ref = case.load_ref(name, root=root)
    if ref is None or ref.info.get("kind") != "discrete-exact":
        return
    ref_dir = Path(root) / "refs" / name
    for fname in ("eigenvalues.npy", "eigenvectors.npy", "ref.json"):
        (ref_dir / fname).unlink(missing_ok=True)


def write_case(name: str, A, B, *, meta: dict[str, Any], targets: list[dict[str, Any]],
               root: str | Path, eigenvalues=None, ref_note: str = "") -> Path:
    """Save a case whose default configuration is its first target; save the complete exact
    spectrum as reference when it is known (otherwise drop a stale exact reference)."""
    meta = dict(meta)
    meta["targets"] = targets
    config = case.default_config(targets[0]["sigma"], targets[0]["k"])
    out = case.save_case(name, A, B, meta=meta, config=config, root=root)
    if eigenvalues is None:
        drop_generated_reference(name, root)
    else:
        case.save_ref(name, eigenvalues, kind="discrete-exact", note=ref_note, complete=True,
                      window=None, root=root)
    return out


def run_presets(presets: dict[str, Callable[[Path], Path]], argv: Sequence[str] | None, *,
                prog: str, description: str) -> int:
    """Command line shared by the generators: list presets or build some, one JSON line each."""
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("presets", nargs="*", help="preset names ('all' = every preset)")
    ap.add_argument("--list", action="store_true", help="list the presets and exit")
    ap.add_argument("--root", type=Path, default=case.DEFAULT_ROOT, help="data root")
    args = ap.parse_args(argv)
    if args.list or not args.presets:
        print("\n".join(presets))
        return 0
    names = list(presets) if args.presets == ["all"] else args.presets
    unknown = [n for n in names if n not in presets]
    if unknown:
        ap.error(f"unknown presets {unknown}; see --list")
    for name in names:
        t0 = time.perf_counter()
        path = presets[name](args.root)
        seconds = time.perf_counter() - t0
        print(json.dumps(case_summary(path, args.root) | {"seconds": round(seconds, 2)}), flush=True)
    return 0


def case_summary(path: Path, root: str | Path) -> dict[str, Any]:
    """n, nnz, problem, dtype and on-disk size of a written case (+ its reference)."""
    c = case.load_case(path)
    files = [p for d in (path, Path(root) / "refs" / c.name) if d.exists() for p in d.iterdir()]
    return {"case": c.name, "problem": c.meta["problem"], "dtype": c.meta["dtype"], "n": c.n,
            "nnz_A": int(c.A.nnz), "nnz_B": int(c.B.nnz) if c.B is not None else 0,
            "disk_bytes": int(sum(p.stat().st_size for p in files))}


# ====================================================== (a, d) tensor-product (Kronecker) operators


def second_difference(n: int) -> sp.csr_matrix:
    """T = tridiag(-1, 2, -1) of order n (1-D Dirichlet second difference, unit spacing)."""
    e = np.ones(n)
    return sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csr")


def second_difference_eigenvalues(n: int) -> np.ndarray:
    """Eigenvalues of T: 4 sin^2(j pi / (2 (n+1))), j = 1..n, ascending.

    The sine form avoids the cancellation in 2 - 2 cos(theta) for the smallest eigenvalues.
    """
    return 4.0 * np.sin(np.arange(1, n + 1) * np.pi / (2.0 * (n + 1))) ** 2


def _kron_all(factors: Sequence[sp.spmatrix]) -> sp.csr_matrix:
    out = sp.csr_matrix(factors[0])
    for f in factors[1:]:
        out = sp.kron(out, f, format="csr")
    return out


def tensor_operator(stiff: Sequence[sp.spmatrix], mass: Sequence[sp.spmatrix | None],
                    shift: float = 0.0) -> sp.csr_matrix:
    """A = sum_d (M_1 x ... x K_d x ... x M_D) - shift (M_1 x ... x M_D), x = Kronecker product.

    mass[d] = None means the identity (then A is the Kronecker sum of the K_d). Index order is
    C order: the first dimension varies slowest.
    """
    mats = [sp.identity(k.shape[0], format="csr") if m is None else sp.csr_matrix(m)
            for k, m in zip(stiff, mass)]
    terms = [_kron_all([stiff[e] if e == d else mats[e] for e in range(len(stiff))])
             for d in range(len(stiff))]
    a = functools.reduce(lambda x, y: x + y, terms)
    if shift != 0.0:
        a = a - shift * _kron_all(mats)
    return hermitian_part(a)


def tensor_eigenvalues(stiff_eigs: Sequence[np.ndarray], mass_eigs: Sequence[np.ndarray | None],
                       shift: float = 0.0) -> np.ndarray:
    """Exact spectrum of tensor_operator when K_d and M_d share eigenvectors (both polynomials in
    the same T_d): lambda = sum_d kappa_d prod_{e != d} mu_e - shift prod_e mu_e, ascending.

    Each index tuple's (kappa, mu) pairs are sorted by kappa before evaluation, so eigenvalues that
    are equal because coinciding dimensions are permuted (exact multiplicities) are computed by
    the same floating-point operations and come out bitwise equal.
    """
    dims = len(stiff_eigs)
    kap = [np.asarray(k, dtype=float) for k in stiff_eigs]
    mu = [np.ones_like(k) if m is None else np.asarray(m, dtype=float) for k, m in zip(kap, mass_eigs)]
    idx = np.meshgrid(*[np.arange(k.size) for k in kap], indexing="ij")
    K = np.stack([kap[d][idx[d].ravel()] for d in range(dims)], axis=1)
    M = np.stack([mu[d][idx[d].ravel()] for d in range(dims)], axis=1)
    order = np.argsort(K, axis=1, kind="stable")
    K = np.take_along_axis(K, order, axis=1)
    M = np.take_along_axis(M, order, axis=1)
    lam = np.zeros(K.shape[0])
    for d in range(dims):
        term = K[:, d].copy()
        for e in range(dims):
            if e != d:
                term *= M[:, e]
        lam += term
    if shift != 0.0:
        lam -= shift * np.prod(M, axis=1)
    return np.sort(lam)


def shift_between_levels(eigenvalues, quantile: float) -> float:
    """A shift strictly between two distinct levels near the given quantile of the spectrum, so
    that the shifted operator is indefinite (about `quantile` of it negative) and nonsingular."""
    lev, counts = distinct_levels(eigenvalues)
    j = int(np.searchsorted(np.cumsum(counts), quantile * np.sum(counts)))
    j = min(max(j, 0), lev.size - 2)
    return float(lev[j] + 0.4 * (lev[j + 1] - lev[j]))


def _spacing(shape: Sequence[int], spacing: Sequence[float] | None) -> list[float]:
    h = [1.0] * len(shape) if spacing is None else [float(s) for s in spacing]
    if len(h) != len(shape) or len(shape) not in (2, 3) or min(shape) < 1 or min(h) <= 0:
        raise ValueError("need 2 or 3 dimensions with positive sizes and spacings")
    return h


def laplacian(shape: Sequence[int], spacing: Sequence[float] | None = None,
              shift: float = 0.0) -> sp.csr_matrix:
    """Dirichlet Laplacian -Delta_h on a 2-D/3-D grid of interior points minus shift * I.

    Kronecker sum of T_d / h_d^2 (5-point / 7-point stencil). Order n = prod(shape).
    """
    h = _spacing(shape, spacing)
    return tensor_operator([second_difference(n) / hd**2 for n, hd in zip(shape, h)],
                           [None] * len(shape), shift)


def laplacian_eigenvalues(shape: Sequence[int], spacing: Sequence[float] | None = None,
                          shift: float = 0.0) -> np.ndarray:
    """sum_d 4 sin^2(j_d pi / (2(n_d+1))) / h_d^2 - shift, all index tuples, ascending."""
    h = _spacing(shape, spacing)
    return tensor_eigenvalues([second_difference_eigenvalues(n) / hd**2 for n, hd in zip(shape, h)],
                              [None] * len(shape), shift)


def stencil_factors(n: int, order: int) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """1-D factors (K, M) of the memory-family stencils and their eigenvalues (kappa, mu).

    order 1: K = T, M = I - T/6 = tridiag(1, 4, 1)/6 (trilinear FEM stiffness/mass; 3-D stencil of
             27 points).
    order 2: K = T + T^2/12 (fourth-order second difference [1,-16,30,-16,1]/12 in the interior),
             M = (I - T/6)^2; radius-2 factors give a 5x5x5 = 125-point stencil (complete only
             for shift != 0: without the mass term the (2,2,1)-type couplings cancel exactly).
    K and M are polynomials in T, so all factors share T's sine eigenvectors and the 3-D spectrum
    follows from kappa = p(t), mu = q(t) (mu > 0 because t < 4).
    """
    t_mat = second_difference(n)
    eye = sp.identity(n, format="csr")
    t = second_difference_eigenvalues(n)
    m1 = eye - t_mat / 6.0
    if order == 1:
        return t_mat, m1, t, 1.0 - t / 6.0
    if order == 2:
        return ((t_mat + (t_mat @ t_mat) / 12.0).tocsr(), (m1 @ m1).tocsr(),
                t + t**2 / 12.0, (1.0 - t / 6.0) ** 2)
    raise ValueError("stencil order must be 1 (27-point) or 2 (125-point)")


def stencil_operator(cells: Sequence[int], order: int, shift: float) -> sp.csr_matrix:
    """3-D tensor-product stencil A = sum_d K_d x M x M (perm.) - shift M x M x M (see stencil_factors)."""
    f = [stencil_factors(n, order) for n in cells]
    return tensor_operator([x[0] for x in f], [x[1] for x in f], shift)


def stencil_eigenvalues(cells: Sequence[int], order: int, shift: float) -> np.ndarray:
    f = [stencil_factors(n, order) for n in cells]
    return tensor_eigenvalues([x[2] for x in f], [x[3] for x in f], shift)


def stencil_shift(cells: Sequence[int], order: int, quantile: float) -> float:
    """Shift c making the stencil operator indefinite and nonsingular.

    A's eigenvalues are prod(mu) * (r - c) with r = sum_d kappa_d / mu_d (the eigenvalues of the
    pencil (sum K x M x M, M x M x M)); placing c between two distinct r-levels gives a fraction
    `quantile` of negative eigenvalues and no zero eigenvalue.
    """
    f = [stencil_factors(n, order) for n in cells]
    r = tensor_eigenvalues([x[2] / x[3] for x in f], [None] * len(cells))
    return shift_between_levels(r, quantile)


# ============================================================ (b) prescribed-spectrum matrices


def _haar_blocks(count: int, size: int, rng: np.random.Generator, complex_: bool) -> np.ndarray:
    """count Haar-distributed unitary (orthogonal) matrices of order size, shape (count, s, s).

    QR of a Gaussian matrix with R's diagonal phases moved into Q (Mezzadri's correction)."""
    x = rng.standard_normal((count, size, size))
    if complex_:
        x = x + 1j * rng.standard_normal((count, size, size))
    q, r = np.linalg.qr(x)
    d = np.diagonal(r, axis1=1, axis2=2)
    return q * (d / np.abs(d))[:, None, :]


def block_unitary(n: int, block: int, offset: int, rng: np.random.Generator,
                  complex_: bool) -> sp.csr_matrix:
    """Block-diagonal random unitary with blocks starting at offset, offset+block, ... (a leading
    block of size offset and a trailing remainder block when they do not divide evenly)."""
    bounds = sorted(set([0, n] + list(range(offset if offset > 0 else block, n, block))))
    sizes = np.diff(bounds)
    rows, cols, vals = [], [], []
    for s in np.unique(sizes):
        starts = np.asarray(bounds[:-1])[sizes == s]
        q = _haar_blocks(starts.size, int(s), rng, complex_)
        local = np.arange(s)
        rows.append(np.broadcast_to(starts[:, None, None] + local[None, :, None], q.shape).ravel())
        cols.append(np.broadcast_to(starts[:, None, None] + local[None, None, :], q.shape).ravel())
        vals.append(q.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def prescribed_hermitian(eigenvalues, *, block: int = 8, complex_: bool = False, seed: int = 0,
                         grouping: str = "random", permute: bool = True) -> sp.csr_matrix:
    """Sparse Hermitian matrix with exactly the given eigenvalues (up to rounding).

    A = P Q diag(lambda) Q^H P^T, Q = Q2 Q1: Q1 block diagonal with Haar unitary blocks of size
    `block`, Q2 the same with blocks offset by block/2 (the stagger couples neighbouring blocks, so
    the matrix does not split into independent blocks), P a random permutation hiding the band.
    About 3*block nonzeros per row.

    grouping: which eigenvalues share a block. 'random' mixes the whole spectrum; 'magnitude'
    groups eigenvalues of similar |lambda| so that the O(eps * local norm) rounding of the
    similarity stays relative to each eigenvalue's own size (needed for wide dynamic ranges, where
    mixing would bury the small eigenvalues under eps * max|lambda|).
    """
    lam = np.asarray(eigenvalues, dtype=float)
    n = lam.size
    rng = np.random.default_rng(seed)
    if grouping == "random":
        d = lam[rng.permutation(n)]
    elif grouping == "magnitude":
        d = lam[np.argsort(np.abs(lam), kind="stable")]
    else:
        raise ValueError("grouping must be 'random' or 'magnitude'")
    q = block_unitary(n, block, block // 2, rng, complex_) @ block_unitary(n, block, 0, rng, complex_)
    a = q @ sp.diags(d) @ q.conj().T
    if permute:
        p = rng.permutation(n)
        a = sp.csr_matrix(a)[p][:, p]
    return hermitian_part(a)


@dataclass
class Spectrum:
    """A prescribed spectrum with its evaluation targets and scenario tags."""
    kind: str
    eigenvalues: np.ndarray  # ascending
    targets: list[dict[str, Any]]
    scenarios: list[str]
    grouping: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)


SPECTRUM_KINDS = ("zero-cluster", "multiplicity", "near-singular", "wide-range")


def spectrum_preset(kind: str, n: int, seed: int) -> Spectrum:
    """Spectra for the solver's failure scenarios (docs: competition item e). Values are O(1)
    except 'wide-range'.

    zero-cluster : n/10 eigenvalues uniform in [-1e-6, 1e-6] (low-frequency breakdown), the rest
                   uniform in +-[0.05, 1].
    multiplicity : ~60% of the spectrum in multiplets of multiplicity 2..12, the rest simple.
    near-singular: uniform in [-1, 1]; targets sigma = lambda_i (1 + 1e-12).
    wide-range   : |lambda| log-uniform in [1e-6, 1e6], random signs.
    """
    rng = np.random.default_rng(seed)
    if kind == "zero-cluster":
        width, nc = 1e-6, max(20, n // 10)
        rest = rng.uniform(0.05, 1.0, n - nc) * rng.choice([-1.0, 1.0], n - nc)
        w = np.sort(np.concatenate((rng.uniform(-width, width, nc), rest)))
        targets = [target_at(w, 0.0, 20, f"sigma=0 inside a cluster of {nc} eigenvalues in [-1e-6, 1e-6]"),
                   target_at(w, 0.5, 10, "regular part of the spectrum")]
        return Spectrum(kind, w, targets, ["zero-cluster", "clustered"], "random",
                        f"{nc} eigenvalues clustered in [-1e-6, 1e-6], others in +-[0.05, 1]",
                        {"cluster_size": nc, "cluster_halfwidth": width})
    if kind == "multiplicity":
        values, mults, filled = [], [], 0
        while filled < 0.6 * n:
            m = int(min(rng.integers(2, 13), n - filled))
            values.append(rng.uniform(-1.0, 1.0))
            mults.append(m)
            filled += m
        w = np.sort(np.concatenate((np.repeat(values, mults), rng.uniform(-1.0, 1.0, n - filled))))
        lev, _ = distinct_levels(w)
        top = int(np.argmax(mults))
        level = int(np.searchsorted(lev, values[top]))
        targets = [target_between_levels(w, level, mults[top] + 2,
                                         f"next to a {mults[top]}-fold eigenvalue {values[top]:.6g}"),
                   target_at(w, 0.0, 40, "several multiplets around sigma=0")]
        return Spectrum(kind, w, targets, ["multiplicity"], "random",
                        f"{len(values)} multiplets (multiplicity 2..12) covering {filled} eigenvalues",
                        {"multiplets": len(values), "max_multiplicity": int(max(mults))})
    if kind == "near-singular":
        w = np.sort(rng.uniform(-1.0, 1.0, n))
        targets = []
        for anchor in (0.3, -0.7):
            lam_i = float(w[np.argmin(np.abs(w - anchor))])
            targets.append(target_at(w, lam_i * (1 + 1e-12), 10,
                                     f"sigma = lambda_i (1 + 1e-12), lambda_i = {lam_i:.17g}"))
        return Spectrum(kind, w, targets, ["near-singular"], "random",
                        "simple spectrum uniform in [-1, 1]; targets within 1e-12 (relative) of an eigenvalue")
    if kind == "wide-range":
        w = np.sort(10.0 ** rng.uniform(-6.0, 6.0, n) * rng.choice([-1.0, 1.0], n))
        targets = [target_at(w, 1.0, 10, "|lambda| ~ 1 in a spectrum spanning 1e-6..1e6"),
                   target_at(w, 1e-4, 10, "small eigenvalues: absolute accuracy ~ tol * 1e6 limits them")]
        return Spectrum(kind, w, targets, ["clustered"], "magnitude",
                        "|lambda| log-uniform in [1e-6, 1e6] with random signs (12 decades)")
    raise ValueError(f"unknown spectrum kind {kind!r}; choose from {SPECTRUM_KINDS}")


# =================================================================== (c) generalized HPD pairs


def congruence_factor(n: int, *, b_cond: float, bandwidth: int, complex_: bool,
                      rng: np.random.Generator) -> sp.csr_matrix:
    """W = D (I + N): N strictly lower banded (half bandwidth `bandwidth`, ~half the band filled)
    with |N_ij| <= 0.5/bandwidth, so ||N||_inf <= 1/2 and cond(I + N) <= 3; D diagonal with
    log-uniform entries in [1, sqrt(b_cond)]. Hence cond(W^H W) <~ 9 b_cond: B can be made as
    ill-conditioned as a real mass matrix without making the exact eigenvalues inaccurate."""
    rows, cols, vals = [], [], []
    for j in range(1, bandwidth + 1):
        r = np.arange(j, n)
        keep = rng.random(r.size) < 0.5
        v = rng.uniform(-1.0, 1.0, keep.sum())
        if complex_:
            v = v * np.exp(2j * np.pi * rng.random(keep.sum()))
        rows.append(r[keep])
        cols.append(r[keep] - j)
        vals.append(v * (0.5 / bandwidth))
    lower = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
    d = 10.0 ** rng.uniform(0.0, 0.5 * np.log10(b_cond), n)
    return (sp.diags(d) @ (sp.identity(n) + lower)).tocsr()


def congruence_pair(eigenvalues, *, b_cond: float = 100.0, bandwidth: int = 3, block: int = 8,
                    complex_: bool = False, seed: int = 0,
                    grouping: str = "random") -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Pencil (A, B), B HPD, whose eigenvalues are exactly the given ones (up to rounding).

    A = P W^H M W P^T, B = P W^H W P^T with M = prescribed_hermitian(eigenvalues) in its banded
    (unpermuted) form: A x = lambda B x  <=>  M (W P^T x) = lambda (W P^T x). Building M unpermuted
    keeps W's band aligned with M's band, so A has ~3*block + 4*bandwidth nonzeros per row instead
    of the fill a random W would cause. Rounding perturbs eigenvalues by ~eps cond(B) |lambda|_max.
    Inertia(A) = inertia(M) by Sylvester's law.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    n = lam.size
    m = prescribed_hermitian(lam, block=block, complex_=complex_, seed=seed, grouping=grouping,
                             permute=False)
    rng = np.random.default_rng([seed, 1])
    w = congruence_factor(n, b_cond=b_cond, bandwidth=bandwidth, complex_=complex_, rng=rng)
    p = rng.permutation(n)
    a = (w.conj().T @ m @ w).tocsr()[p][:, p]
    b = (w.conj().T @ w).tocsr()[p][:, p]
    return hermitian_part(a), hermitian_part(b)


# ======================================================================= preset case writers


def _rounding_note(what: str) -> str:
    return f"exact eigenvalues of the stored matrix up to assembly rounding: {what}"


def make_laplacian_case(name: str, shape: Sequence[int], spacing: Sequence[float] | None,
                        quantile: float, root: str | Path) -> Path:
    """Shifted Dirichlet Laplacian, shift between levels at the given quantile."""
    base = laplacian_eigenvalues(shape, spacing)
    c = shift_between_levels(base, quantile)
    w = laplacian_eigenvalues(shape, spacing, c)
    a = laplacian(shape, spacing, c)
    multiple = len(set(zip(shape, spacing or [1.0] * len(shape)))) < len(shape)
    targets = [target_at(w, 0.0, 12, f"sigma=0: interior of the shifted spectrum ({quantile:.0%} negative)"),
               target_between_levels(w, int(np.searchsorted(distinct_levels(w)[0], 0.5 * w[-1])), 12,
                                     "middle of the spectrum")]
    meta = {"family": FAMILY, "problem": "HEP",
            "description": f"{len(shape)}-D Dirichlet Laplacian {tuple(shape)} shifted by c={c:.6g}",
            "params": {"generator": "synthetic.laplacian", "shape": list(shape),
                       "spacing": list(spacing) if spacing else None, "shift": c, "quantile": quantile},
            "inertia": {"A": inertia_counts(w)},
            "scenarios": ["multiplicity"] if multiple else []}
    return write_case(name, a, None, meta=meta, targets=targets, root=root, eigenvalues=w,
                      ref_note=_rounding_note("Kronecker sum of tridiag(-1,2,-1)/h^2 minus c I; "
                                              "sum_d 4 sin^2(j pi/(2(n_d+1)))/h_d^2 - c"))


def make_prescribed_case(name: str, kind: str, n: int, complex_: bool, seed: int,
                         root: str | Path) -> Path:
    spec = spectrum_preset(kind, n, seed)
    a = prescribed_hermitian(spec.eigenvalues, complex_=complex_, seed=seed + 1, grouping=spec.grouping)
    meta = {"family": FAMILY, "problem": "HEP",
            "description": f"prescribed spectrum '{kind}': {spec.description}",
            "params": {"generator": "synthetic.prescribed_hermitian", "kind": kind, "n": n,
                       "complex": complex_, "spectrum_seed": seed, "matrix_seed": seed + 1, "block": 8,
                       "grouping": spec.grouping, **spec.params},
            "inertia": {"A": inertia_counts(spec.eigenvalues)},
            "scenarios": spec.scenarios + (["complex-hermitian"] if complex_ else [])}
    return write_case(name, a, None, meta=meta, targets=spec.targets, root=root,
                      eigenvalues=spec.eigenvalues,
                      ref_note=_rounding_note("unitary similarity of diag(lambda), error ~ eps * "
                                              "max|lambda| over each eigenvalue's block neighbourhood"))


def make_pair_case(name: str, kind: str, n: int, complex_: bool, seed: int, b_cond: float,
                   root: str | Path) -> Path:
    spec = spectrum_preset(kind, n, seed)
    a, b = congruence_pair(spec.eigenvalues, b_cond=b_cond, complex_=complex_, seed=seed + 1,
                           grouping=spec.grouping)
    meta = {"family": FAMILY, "problem": "GHEP",
            "description": f"HPD pencil with prescribed spectrum '{kind}' via congruence: {spec.description}",
            "params": {"generator": "synthetic.congruence_pair", "kind": kind, "n": n,
                       "complex": complex_, "spectrum_seed": seed, "matrix_seed": seed + 1, "block": 8,
                       "bandwidth": 3, "b_cond": b_cond, "grouping": spec.grouping, **spec.params},
            "inertia": {"A": inertia_counts(spec.eigenvalues), "B": [0, 0, n],
                        "pencil": inertia_counts(spec.eigenvalues)},
            "scenarios": spec.scenarios + (["complex-hermitian"] if complex_ else [])}
    return write_case(name, a, b, meta=meta, targets=spec.targets, root=root,
                      eigenvalues=spec.eigenvalues,
                      ref_note=_rounding_note(f"congruence W^H (.) W with cond(B) <~ {9 * b_cond:g}, "
                                              "error ~ eps * cond(B) * max|lambda|"))


def make_memory_case(name: str, cells: Sequence[int], order: int, quantile: float,
                     root: str | Path) -> Path:
    c = stencil_shift(cells, order, quantile)
    w = stencil_eigenvalues(cells, order, c)
    a = stencil_operator(cells, order, c)
    points = (2 * order + 1) ** 3
    meta = {"family": FAMILY, "problem": "HEP",
            "description": (f"memory stress: 3-D {points}-point tensor-product stencil on "
                            f"{'x'.join(map(str, cells))} points, shifted to {quantile:.0%} negative"),
            "params": {"generator": "synthetic.stencil_operator", "cells": list(cells), "order": order,
                       "stencil_points": points, "shift": c, "quantile": quantile},
            "inertia": {"A": inertia_counts(w)},
            "scenarios": ["memory", "large"] if a.shape[0] >= 100_000 else ["memory"]}
    return write_case(name, a, None, meta=meta,
                      targets=[target_at(w, 0.0, 10, "sigma=0 inside the indefinite spectrum")],
                      root=root, eigenvalues=w,
                      ref_note=_rounding_note("K, M polynomials in tridiag(-1,2,-1); "
                                              "sum_d kappa_d prod mu_e - c prod mu_e"))


def _presets() -> dict[str, Callable[[Path], Path]]:
    p: dict[str, Callable[[Path], Path]] = {}

    def add(name: str, fn: Callable[..., Path], *args: Any) -> None:
        p[name] = lambda root, fn=fn, name=name, args=args: fn(name, *args, root=root)

    add("syn-lap2d-100", make_laplacian_case, (100, 100), None, 0.05)
    add("syn-lap3d-22", make_laplacian_case, (22, 22, 22), None, 0.05)
    add("syn-lap3d-aniso-1e5", make_laplacian_case, (60, 45, 36), (1.0, 1.3, 0.8), 0.03)
    for i, kind in enumerate(SPECTRUM_KINDS):
        for complex_ in (False, True):
            tag = "complex" if complex_ else "real"
            add(f"syn-{kind}-{tag}-1e4", make_prescribed_case, kind, 10_000, complex_, 100 + 10 * i)
    add("syn-zero-cluster-real-1e5", make_prescribed_case, "zero-cluster", 100_000, False, 200)
    add("syn-near-singular-complex-1e5", make_prescribed_case, "near-singular", 100_000, True, 210)
    add("syn-ghep-multiplicity-real-1e4", make_pair_case, "multiplicity", 10_000, False, 300, 1e3)
    add("syn-ghep-zero-cluster-complex-1e4", make_pair_case, "zero-cluster", 10_000, True, 310, 1e2)
    add("syn-ghep-near-singular-real-1e5", make_pair_case, "near-singular", 100_000, False, 320, 1e3)
    add("syn-mem27-46", make_memory_case, (46, 46, 46), 1, 0.02)
    add("syn-mem125-46", make_memory_case, (46, 46, 46), 2, 0.02)
    add("syn-mem27-100", make_memory_case, (100, 100, 100), 1, 0.02)
    return p


PRESETS = _presets()


def main(argv: Sequence[str] | None = None) -> int:
    return run_presets(PRESETS, argv, prog="python -m tools.gen.synthetic",
                       description="Generate synthetic cases with exact spectra into <root>/cases.")


if __name__ == "__main__":
    raise SystemExit(main())
