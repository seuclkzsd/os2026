"""Independent evaluation of an eigensolver result directory against a case (docs/FORMATS.md §4).

Everything is recomputed from the case matrices and the stored vectors; the solver's report.json is
used only for bookkeeping (time, peak memory, its own success claim).

Per computed pair (lambda, x) (DESIGN §4.5):
  abs_residual    r   = ||A x - lambda B x||_2
  rel_residual        = r / (|lambda| ||B x||_2)            (r / ||B x||_2 when lambda == 0)
  backward_error  eta = r / ((||A||_2 + |lambda| ||B||_2) ||x||_2)
Norms: exact (dense eigvalsh, max |eigenvalue|) for n <= DENSE_MAX. Above that, the largest-magnitude
Ritz value of ARPACK eigsh(which='LM'): for a Hermitian matrix every Ritz value lies inside the
spectrum, so it is a LOWER bound of ||.||_2 and eta is over-estimated (conservative), never
optimistic. Only if ARPACK fails do we fall back to the 1-norm, an upper bound (optimistic eta) -
reported in norm_method. ARPACK runs with tol=NORM_TOL, ncv=NORM_NCV: measured on five 1e5 cases
(A and B of syn-zero-cluster-real-1e5, wg-square-1e5, wg-halfslab-1e5, ss-qa8fk-qa8fm,
yee-pec-box-1e5) it agrees with scipy's default ncv at tol=1e-8 to <= 1.4e-10 relative and needs up
to 4.4x fewer products (4901 -> 1121, 23.8 s -> 7.0 s on the zero-cluster Laplacian, whose dense
upper spectrum made the norm estimate the dominant cost of an evaluation); the one increase is a
diagonal B converging at once (21 -> 41 products: ncv is the minimum).

Comparison with the reference (data/refs/<case>). Continuum-model references (kind partial-analytic,
common.CONTINUUM_KINDS) are O(h^2) away from the matrix eigenvalues: they are reported with
comparable=false and not used below; build a discrete one with tools/eval/reference.py.
  * matching: one-to-one, |lambda - mu| <= match_rtol * (|mu| + scale), scale = ||A||_2/||B||_2;
    maximal number of matches first, then minimal total normalized distance (Hungarian algorithm).
    The (|mu| + scale) yardstick is the eigenvalue error a backward-stable solver can guarantee:
    relative for large |mu|, relative to the spectrum scale near 0 (zero clusters).
  * eigenvalue errors: eig_abs_error |lambda - mu|; eig_rel_error |lambda - mu|/|mu| (null for
    mu == 0); eig_scaled_error |lambda - mu|/(|mu| + scale), the one comparable with eta.
  * completeness: expected = the target's reference eigenvalues (nearest: the k nearest sigma;
    smallest/largest: the k extreme ones; interval: all inside). Reference eigenvalues whose target
    key (distance to sigma, value, interval end) ties with the k-th one within
    tie_rtol * (|mu| + scale) are interchangeable: any of them may complete the set, so a
    multiplicity cluster cut by k accepts any subset of its eigenspace; a reference eigenvalue the
    set needs and the result lacks is 'missing', a shortfall among tied ones is the
    'boundary_deficit' (with 'boundary_candidates' that could fill it). tie_rtol defaults to
    TIE_FACTOR * tol, the resolution an accepted pair guarantees: for a Hermitian matrix
    |lambda - mu| <= ||r||_2/||x||_2 = eta (||A||_2 + |lambda|), so eigenvalues closer than about
    tol * (|mu| + scale) cannot be told apart by the acceptance test, while farther ones can (a
    dense cluster such as a 1e-6-wide zero cluster is resolved, not waved through). For a definite
    pencil the bound grows by up to cond(B): raise --tie-rtol for ill-conditioned B. Status:
        complete / incomplete      ref.complete and its window covers the needed range (certified)
        consistent / inconsistent  same test against an uncertified (solver) reference
        unknown                    no usable reference
    Independently of any reference, fewer than k distinct vectors is 'incomplete'. Computed pairs
    outside the expected set are 'extra' = 'beyond_target' (matched to another reference
    eigenvalue) + 'unmatched' (no reference eigenvalue within match_rtol); extra pairs are not an
    error, but unmatched ones inside the window of a complete reference are 'spurious'.
  * clusters: reference eigenvalues chained by gaps <= cluster_rtol * (|mu| + scale) (consecutive in
    (Re, Im) order). With reference eigenvectors, the principal angles between the span of the
    computed vectors matched into a cluster and the cluster's reference eigenspace are reported
    (Euclidean; an eigenspace is a set, so any basis of it is accepted).
Duplicates: a vector within relative distance dup_tol of the span of the vectors before it (B-norm
for GHEP, 2-norm otherwise) duplicates an earlier one. B-orthogonality ||X^H B X - I||_2 (HEP/GHEP).
Non-finite residuals are reported as Infinity (Python's JSON extension), never hidden.
Verdict 'pass' iff pairs were returned, every eta <= tol, no duplicates, completeness is not
incomplete/inconsistent and no computed eigenvalue is spurious (unmatched where a complete reference
lists every eigenvalue). Disagreement with an uncertified reference fails too: it means either the
solver or the reference is wrong and needs a look.

Pooled references: a case without a comparable stored reference can be checked against the accepted,
distinct pairs of several solvers (pooled_reference; tools/eval/sweep.py does this per case). It is
uncertified (consistent / inconsistent), but sound in the direction that matters: an accepted pair
of a Hermitian problem is backed by a true eigenvalue within eta (||A||_2 + |lambda|) (residual
bound; for a definite pencil the bound grows with cond(B), for GHIEP it holds only as far as the
eigenvalue is well conditioned), and
B-orthonormal accepted vectors in one cluster are backed by as many eigenvalues, so a solver that
lacks a pooled eigenvalue nearer than its own k-th one, or fewer copies of a multiple one than another
solver proved to exist, missed it. Eigenvalue errors are not reported against it (it holds the
solvers' own values).

Bookkeeping that is reported but not part of the verdict:
  * format (docs/FORMATS.md §3): eigenvalues.npy ordered for the target (inversions within the tie
    band are ties), HEP/GHEP vectors B-normalized, GHIEP vectors of unit 2-norm (FORMAT_NORM_TOL).
  * solver_claim: the solver's own report.json 'success' against the verdict - 'agrees',
    'overclaim' (success claimed, evaluation fails: the dangerous case) or 'underclaim'.
  * time and peak memory as reported by the solver (peak_rss_bytes, or the SLEPc baseline's
    peak_rss_bytes_sum over its MPI ranks) and, when the caller passes the tools/run/guard.py summary
    of the run, as measured by the guard ('measured': wall time and peak RSS of the process tree, the
    same yardstick for every solver).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import linear_sum_assignment

from tools import case as cio
from tools.eval import common

NORM_TOL = 1e-6        # ARPACK relative accuracy of the ||.||_2 estimate (module docstring)
NORM_NCV = 40          # ARPACK subspace size of the ||.||_2 estimate
MATCH_RTOL = 1e-6
CLUSTER_RTOL = 1e-6
TIE_FACTOR = 10.0      # default tie_rtol = TIE_FACTOR * tol (module docstring, completeness)
DUP_TOL = 1e-4
FORMAT_NORM_TOL = 1e-6  # vector normalization deviation reported as a format issue
POOLED_KIND = "pooled"  # in-memory reference kind of pooled_reference (never written to data/refs)
EVAL_FILE = "evaluation.json"


# --------------------------------------------------------------------------------------- norms

@dataclass(frozen=True)
class Norms:
    norm2_a: float
    norm2_b: float
    method_a: str
    method_b: str

    @property
    def scale(self) -> float:
        """||A||_2 / ||B||_2: the natural eigenvalue magnitude of the pencil."""
        return self.norm2_a / self.norm2_b if self.norm2_b > 0 else self.norm2_a


def hermitian_norm2(m: sp.spmatrix, dense_max: int = common.DENSE_MAX) -> tuple[float, str]:
    """(||m||_2, method) for a Hermitian sparse matrix; see the module docstring for the methods."""
    n = m.shape[0]
    if n <= dense_max:
        w = la.eigvalsh(common.to_dense(m, dense_max))
        return (float(np.max(np.abs(w))) if n else 0.0), "dense"
    try:
        w = sla.eigsh(m, k=1, which="LM", tol=NORM_TOL, ncv=min(NORM_NCV, n),
                      v0=common.start_vector(n, m.dtype, 0), return_eigenvectors=False)
        return float(np.max(np.abs(w))), "eigsh"
    except sla.ArpackNoConvergence as err:
        if err.eigenvalues.size:
            return float(np.max(np.abs(err.eigenvalues))), "eigsh-unconverged"
    return float(sla.norm(m, 1)), "one-norm"


def estimate_norms(A: sp.spmatrix, B: sp.spmatrix | None, dense_max: int = common.DENSE_MAX) -> Norms:
    na, ma = hermitian_norm2(A, dense_max)
    if B is None:
        return Norms(na, 1.0, ma, "identity")
    nb, mb = hermitian_norm2(B, dense_max)
    return Norms(na, nb, ma, mb)


# ------------------------------------------------------------------------------- pair metrics

def _finite_or_inf(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.inf)


def pair_metrics(A: sp.spmatrix, B: sp.spmatrix | None, lambdas, X, norms: Norms, tol: float) -> list[dict]:
    """PairReport records (include/hsolve/solver.hpp) for the columns of X. NaN/inf -> inf."""
    lam = np.asarray(lambdas)
    if lam.size == 0:
        return []
    X = np.asarray(X)
    if X.shape != (A.shape[0], lam.size):
        raise ValueError(f"eigenvectors have shape {X.shape}, expected {(A.shape[0], lam.size)}")
    BX = X if B is None else B @ X
    R = A @ X - BX * lam[np.newaxis, :]
    r = np.linalg.norm(R, axis=0)
    bx = np.linalg.norm(BX, axis=0)
    xn = np.linalg.norm(X, axis=0)
    lam_abs = np.abs(lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = _finite_or_inf(r / np.where(lam_abs > 0, lam_abs * bx, bx))
        eta = _finite_or_inf(r / ((norms.norm2_a + lam_abs * norms.norm2_b) * xn))
    r = _finite_or_inf(r)
    lc = lam.astype(complex)
    return [{"lambda_re": float(lc[i].real), "lambda_im": float(lc[i].imag), "abs_residual": float(r[i]),
             "rel_residual": float(rel[i]), "backward_error": float(eta[i]), "accepted": bool(eta[i] <= tol)}
            for i in range(lam.size)]


def b_orthogonality(X: np.ndarray, B: sp.spmatrix | None) -> float:
    """||X^H B X - I||_2 (loss of B-orthonormality of the returned basis)."""
    G = X.conj().T @ (X if B is None else B @ X)
    return float(la.norm(G - np.eye(G.shape[0]), 2))


def find_duplicates(X: np.ndarray, W: sp.spmatrix | None = None, dup_tol: float = DUP_TOL) -> list[dict]:
    """Columns that add (almost) no new direction: relative W-distance to the span of the earlier
    accepted columns <= dup_tol. Incremental Cholesky of the normalized Gram matrix: when columns
    0..p-1 are accepted with Gn = L L^H, the new column's squared distance is 1 - ||L^-1 g||^2.
    Gram-based distances are accurate to ~sqrt(eps), ample for dup_tol >= 1e-6. Zero or non-finite
    columns count as duplicates of nothing (duplicate_of = None)."""
    m = X.shape[1]
    G = X.conj().T @ (X if W is None else W @ X)
    d = np.sqrt(np.maximum(np.real(np.diag(G)), 0.0))
    L = np.zeros((m, m), dtype=G.dtype)
    basis: list[int] = []
    out = []
    for j in range(m):
        if not np.isfinite(d[j]) or d[j] == 0.0 or not np.all(np.isfinite(G[:, j])):
            out.append({"index": j, "duplicate_of": None, "distance": 0.0})
            continue
        p = len(basis)
        g = G[basis, j] / (d[basis] * d[j])
        c = la.solve_triangular(L[:p, :p], g, lower=True) if p else g
        dist = float(np.sqrt(max(1.0 - float(np.real(np.vdot(c, c))), 0.0)))
        if dist <= dup_tol:
            closest = basis[int(np.argmax(np.abs(g)))]
            out.append({"index": j, "duplicate_of": closest, "distance": dist})
        else:
            L[p, :p] = c.conj()
            L[p, p] = dist
            basis.append(j)
    return out


# ---------------------------------------------------------------------- reference comparison

def match_to_reference(lambdas: np.ndarray, ref_vals: np.ndarray, scale: float, rtol: float) -> np.ndarray:
    """Reference index matched to each computed eigenvalue (-1: none within tolerance), one-to-one."""
    lam = np.asarray(lambdas, dtype=complex)
    out = np.full(lam.size, -1, dtype=np.int64)
    ref = np.asarray(ref_vals, dtype=complex)
    if lam.size == 0 or ref.size == 0:
        return out
    tol = rtol * (np.abs(ref) + scale) + np.finfo(float).tiny
    dist = np.abs(lam[:, None] - ref[None, :]) / tol[None, :]
    feasible = dist <= 1.0
    rows = np.flatnonzero(feasible.any(axis=1))
    cols = np.flatnonzero(feasible.any(axis=0))
    if rows.size == 0:
        return out
    sub = dist[np.ix_(rows, cols)]
    # Infeasible cost exceeds any sum of feasible ones (<= 1 each), so the optimum maximizes the
    # number of feasible matches first and only then minimizes the total distance.
    cost = np.where(sub <= 1.0, sub, rows.size + 1.0)
    ri, ci = linear_sum_assignment(cost)
    ok = cost[ri, ci] <= 1.0
    out[rows[ri[ok]]] = cols[ci[ok]]
    return out


def reference_clusters(ref_vals: np.ndarray, scale: float, rtol: float) -> np.ndarray:
    """Cluster id of every reference eigenvalue: chains of neighbours (in (Re, Im) order) closer than
    rtol * (max |mu| + scale)."""
    ref = np.asarray(ref_vals, dtype=complex)
    ids = np.zeros(ref.size, dtype=np.int64)
    if ref.size == 0:
        return ids
    order = np.lexsort((ref.imag, ref.real))
    s = ref[order]
    gap_tol = rtol * (np.maximum(np.abs(s[1:]), np.abs(s[:-1])) + scale)
    ids[order] = np.concatenate(([0], np.cumsum(np.abs(np.diff(s)) > gap_tol)))
    return ids


@dataclass
class ExpectedSet:
    required: np.ndarray     # reference indices that must be found
    optional: np.ndarray     # tied at the boundary: interchangeable
    need_optional: int       # how many of `optional` complete the set (nearest/smallest/largest)
    window: tuple[float, float] | None   # real range a reference must list completely; None = all


def expected_set(ref_vals: np.ndarray, tgt: dict[str, Any], scale: float, rtol: float) -> ExpectedSet:
    """Reference indices the target asks for. Values tied with the k-th one (or with an interval end)
    within rtol * (|mu| + scale) are interchangeable `optional` ones (rtol is the evaluation's
    tie_rtol); `window` is the real range a reference must list completely for the set to be certain."""
    ref = np.asarray(ref_vals, dtype=complex)
    if tgt["mode"] == "interval":
        lo, hi = float(tgt["lower"]), float(tgt["upper"])
        blo, bhi = rtol * (abs(lo) + scale), rtol * (abs(hi) + scale)
        x = ref.real
        inside = (ref.imag == 0) & (x >= lo - blo) & (x <= hi + bhi)
        boundary = inside & ((x < lo + blo) | (x > hi - bhi))
        return ExpectedSet(np.flatnonzero(inside & ~boundary), np.flatnonzero(boundary), 0, (lo - blo, hi + bhi))
    k = int(tgt["k"])
    if ref.size <= k:   # everything is expected; only a full-spectrum reference can certify it
        return ExpectedSet(np.arange(ref.size), np.empty(0, dtype=np.int64), 0, None)
    sigma = float(tgt.get("sigma", 0.0))
    key = {"nearest": np.abs(ref - sigma), "smallest": ref.real, "largest": -ref.real}[tgt["mode"]]
    order = np.argsort(key, kind="stable")
    kth, mu_k = key[order[k - 1]], ref[order[k - 1]]
    band = rtol * (abs(mu_k) + scale)
    required = np.flatnonzero(key < kth - band)
    optional = np.flatnonzero(np.abs(key - kth) <= band)
    if optional.size == k - required.size:   # no tie at the boundary: every one of them is needed
        required, optional = np.sort(np.concatenate([required, optional])), np.empty(0, dtype=np.int64)
    window = {"nearest": (sigma - kth - band, sigma + kth + band),
              "smallest": (-np.inf, mu_k.real + band),
              "largest": (mu_k.real - band, np.inf)}[tgt["mode"]]
    return ExpectedSet(required, optional, k - required.size, window)


def _covers(info: dict[str, Any], window: tuple[float, float] | None) -> bool:
    ref_window = info.get("window")
    if ref_window is None:
        return True
    return window is not None and ref_window[0] <= window[0] and ref_window[1] >= window[1]


def _ref_entry(ref: np.ndarray, j: int) -> dict[str, Any]:
    mu = complex(ref[j])
    return {"ref_index": int(j), "lambda_re": mu.real, "lambda_im": mu.imag}


def assess_completeness(tgt: dict[str, Any], lambdas: np.ndarray, ref: cio.Reference | None,
                        exp: ExpectedSet | None, match: np.ndarray, n_distinct: int, n: int) -> dict[str, Any]:
    """Completeness block of the evaluation (statuses in the module docstring). exp is the target's
    expected set in `ref` (None without a usable reference)."""
    lam = np.asarray(lambdas, dtype=complex)
    out: dict[str, Any] = {"status": "unknown", "certified": False, "expected": None, "found": None,
                           "missing": [], "boundary_deficit": 0, "boundary_candidates": [],
                           "extra": [], "beyond_target": [], "unmatched": [], "spurious": [],
                           "distinct_returned": n_distinct}
    k_needed = min(int(tgt["k"]), n) if tgt["mode"] != "interval" else 0
    if ref is not None and exp is not None:
        vals = np.asarray(ref.eigenvalues, dtype=complex)
        matched = set(int(j) for j in match[match >= 0])
        in_target = set(exp.required.tolist()) | set(exp.optional.tolist())
        missing = [int(j) for j in exp.required if j not in matched]
        opt_found = [int(j) for j in exp.optional if j in matched]
        deficit = max(0, exp.need_optional - len(opt_found))
        out["missing"] = [_ref_entry(vals, j) for j in missing]
        out["boundary_deficit"] = deficit
        if deficit:
            out["boundary_candidates"] = [_ref_entry(vals, j) for j in exp.optional if j not in matched]
        if tgt["mode"] == "interval":
            out["expected"] = [int(exp.required.size), int(exp.required.size + exp.optional.size)]
            out["found"] = exp.required.size - len(missing) + len(opt_found)
        else:
            out["expected"] = int(exp.required.size + exp.need_optional)
            out["found"] = exp.required.size - len(missing) + min(len(opt_found), exp.need_optional)
        info = ref.info
        complete_ref = bool(info.get("complete", False))
        rw = info.get("window")
        for i, j in enumerate(match):
            if j >= 0 and int(j) not in in_target:
                out["beyond_target"].append(i)
            elif j < 0:
                out["unmatched"].append(i)
                inside = rw is None or rw[0] <= lam[i].real <= rw[1]
                if complete_ref and inside:
                    out["spurious"].append(i)
        out["extra"] = sorted(out["beyond_target"] + out["unmatched"])
        if exp.window is None and vals.size < k_needed and rw is not None:
            out["status"] = "unknown"   # fewer reference values than requested and no full spectrum
        elif complete_ref and _covers(info, exp.window):
            out["certified"] = True
            out["status"] = "complete" if not missing and not deficit else "incomplete"
        else:
            out["status"] = "consistent" if not missing and not deficit else "inconsistent"
    if n_distinct < k_needed:
        out["status"] = "incomplete"
    return out


def cluster_records(ref: cio.Reference, match: np.ndarray, exp: ExpectedSet, X: np.ndarray,
                    scale: float, rtol: float) -> list[dict[str, Any]]:
    """One record per reference cluster touched by a matched pair or by the expected set."""
    vals = np.asarray(ref.eigenvalues, dtype=complex)
    ids = reference_clusters(vals, scale, rtol)
    touched = sorted(set(ids[match[match >= 0]].tolist()) | set(ids[exp.required].tolist())
                     | set(ids[exp.optional].tolist()))
    vecs = ref.eigenvectors
    if vecs is not None and vecs.shape != (X.shape[0], vals.size):
        vecs = None
    records = []
    for cid in touched:
        members = np.flatnonzero(ids == cid)
        comp = [int(i) for i in np.flatnonzero(np.isin(match, members))]
        rec: dict[str, Any] = {"ref_indices": members.tolist(), "multiplicity": int(members.size),
                               "lambda_min": float(vals[members].real.min()),
                               "lambda_max": float(vals[members].real.max()),
                               "computed_indices": comp, "angles": None, "max_angle": None}
        if vecs is not None and comp:
            ang = la.subspace_angles(X[:, comp], vecs[:, members])
            rec["angles"] = sorted(float(a) for a in ang)
            rec["max_angle"] = float(np.max(ang))
        records.append(rec)
    return records


def pooled_reference(docs: list[dict[str, Any]], scale: float, rtol: float = MATCH_RTOL) -> cio.Reference | None:
    """Uncertified in-memory reference from the accepted, non-duplicate pairs of several evaluations
    of one case (module docstring). Eigenvalues of all solvers are chained into clusters (gaps <=
    rtol * (|mu| + scale)); each cluster keeps the values of the solver that returned the most copies
    in it - a lower bound of its multiplicity, never a guess above what one solver proved. None when
    no solver has an accepted pair."""
    sources = [d["solver"] for d in docs]
    per_solver = [np.array([complex(p["lambda_re"], p["lambda_im"]) for p in d["pairs"]
                            if p["accepted"] and not p["duplicate"]], dtype=complex) for d in docs]
    values = np.concatenate(per_solver) if per_solver else np.empty(0, dtype=complex)
    if values.size == 0:
        return None
    owner = np.repeat(np.arange(len(per_solver)), [v.size for v in per_solver])
    ids = reference_clusters(values, scale, rtol)
    pooled = []
    for cid in np.unique(ids):
        counts = [np.count_nonzero((ids == cid) & (owner == s)) for s in range(len(per_solver))]
        pooled.append(values[(ids == cid) & (owner == int(np.argmax(counts)))])
    w = np.concatenate(pooled)
    w = w[np.lexsort((w.imag, w.real))]
    if not np.any(w.imag):
        w = w.real
    info = {"kind": POOLED_KIND, "complete": False, "window": None, "count": int(w.size), "sources": sources,
            "note": f"accepted distinct pairs pooled from {', '.join(sources)} (uncertified)"}
    return cio.Reference(w, None, info)


def format_checks(problem: str, tgt: dict[str, Any], lam_real: np.ndarray, X: np.ndarray, B: sp.spmatrix | None,
                  scale: float, tie_rtol: float) -> dict[str, Any]:
    """docs/FORMATS.md §3 conventions of a result directory (module docstring). lam_real are the
    eigenvalues.npy values; X holds every returned vector (real pairs first, then complex ones)."""
    lam = np.asarray(lam_real, dtype=float)
    key = {"nearest": np.abs(lam - float(tgt.get("sigma", 0.0))), "interval": lam, "smallest": lam,
           "largest": -lam}[tgt["mode"]]
    inversions = np.flatnonzero(np.diff(key) < -tie_rtol * (np.abs(lam[1:]) + scale))
    issues = []
    if inversions.size:
        issues.append(f"eigenvalues.npy is not in '{tgt['mode']}' target order at position(s) "
                      f"{(inversions + 1).tolist()}")
    max_dev = None
    if X.shape[1]:
        if problem == "GHIEP" or B is None:
            dev, what = np.abs(np.linalg.norm(X, axis=0) - 1.0), "of unit 2-norm"
        else:
            bnorm = np.sqrt(np.maximum(np.real(np.sum(X.conj() * (B @ X), axis=0)), 0.0))
            dev, what = np.abs(bnorm - 1.0), "B-normalized"
        max_dev = float(np.max(_finite_or_inf(dev)))
        if max_dev > FORMAT_NORM_TOL:
            issues.append(f"eigenvectors are not {what}: max deviation {max_dev:.1e}")
    return {"order_ok": not inversions.size, "max_normalization_error": max_dev, "issues": issues}


def _reported_peak(report: dict[str, Any]) -> int | None:
    """Solver-reported peak RSS: SolveReport's peak_rss_bytes, or the SLEPc baseline's sum over ranks."""
    for key in ("peak_rss_bytes", "peak_rss_bytes_sum"):
        if report.get(key) is not None:
            return int(report[key])
    return None


def _measured(guard: dict[str, Any] | None) -> dict[str, Any] | None:
    """The tools/run/guard.py summary of the run, reduced to what the evaluation reports."""
    if guard is None:
        return None
    gib = guard.get("peak_rss_gib")
    return {"wall_s": guard.get("wall_s"), "peak_rss_bytes": int(gib * 2**30) if gib is not None else None,
            "returncode": guard.get("returncode"), "killed": guard.get("killed"), "cores": guard.get("cores"),
            "threads": guard.get("threads")}


def _claim(solver_success: Any, verdict: str) -> str | None:
    if solver_success is None:
        return None
    if bool(solver_success) == (verdict == "pass"):
        return "agrees"
    return "overclaim" if solver_success else "underclaim"


# ----------------------------------------------------------------------------------- driver

def _max(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(max(vals)) if vals else None


def _data_root(c: cio.Case, root: str | Path) -> Path:
    return c.path.parent.parent if c.path.parent.name == "cases" else Path(root)


def evaluate(case_ref: str | Path | cio.Case, result_dir: str | Path, *, root: str | Path = cio.DEFAULT_ROOT,
             solver: str | None = None, tol: float | None = None, match_rtol: float = MATCH_RTOL,
             cluster_rtol: float = CLUSTER_RTOL, tie_rtol: float | None = None, dup_tol: float = DUP_TOL,
             dense_max: int = common.DENSE_MAX, norms: Norms | None = None,
             reference: cio.Reference | None = None, guard: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluation document (docs/FORMATS.md §4) for one (case, result directory). case_ref is a case
    name, a case directory or an already loaded case. tol defaults to the case config's tol, tie_rtol
    to TIE_FACTOR * tol. Callers evaluating several results of one case pass the case's norms
    (estimate_norms) to estimate them once. `reference` replaces data/refs/<case> (e.g. a
    pooled_reference); `guard` is the tools/run/guard.py summary of the solver run, if any."""
    c = case_ref if isinstance(case_ref, cio.Case) else cio.load_case(case_ref, root=root)
    problem = common.problem_type(c)
    tgt = common.target(c)
    tol = common.tol(c) if tol is None else float(tol)
    tie_rtol = TIE_FACTOR * tol if tie_rtol is None else float(tie_rtol)
    res = cio.load_result(result_dir)
    data_root = _data_root(c, root)
    if reference is not None:
        stored_ref, ref_source = reference, reference.info.get("note", "in-memory reference")
    else:
        stored_ref, ref_source = cio.load_ref(c.name, root=data_root), str(data_root / "refs" / c.name)
    comparable = stored_ref is not None and stored_ref.info.get("kind") not in common.CONTINUUM_KINDS
    ref = stored_ref if comparable else None
    norms = estimate_norms(c.A, c.B, dense_max) if norms is None else norms
    scale = norms.scale

    n = c.n
    if np.iscomplexobj(res.eigenvalues):
        raise ValueError(f"{result_dir}: eigenvalues.npy must be real; complex pairs go to complex_eigenvalues.npy")
    lam_real = np.asarray(res.eigenvalues, dtype=float).ravel()
    X = np.asarray(res.eigenvectors).reshape(n, -1) if res.eigenvectors.size else np.zeros((n, 0))
    if X.shape[1] != lam_real.size:
        raise ValueError(f"{result_dir}: {lam_real.size} eigenvalues but eigenvectors {res.eigenvectors.shape}")
    lambdas = lam_real.astype(complex)
    kinds = ["real"] * lam_real.size
    if res.complex_eigenvalues is not None and res.complex_eigenvalues.size:
        lc = np.asarray(res.complex_eigenvalues, dtype=complex).ravel()
        Xc = np.asarray(res.complex_eigenvectors).reshape(n, lc.size)
        lambdas = np.concatenate([lambdas, lc])
        X = np.concatenate([X.astype(np.result_type(X, Xc)), Xc], axis=1)
        kinds += ["complex"] * lc.size
    # Real arithmetic unless complex pairs are present (real pairs then carry Im = 0 exactly).
    pairs = pair_metrics(c.A, c.B, lambdas if "complex" in kinds else lam_real, X, norms, tol)

    match = (match_to_reference(lambdas, ref.eigenvalues, scale, match_rtol) if ref is not None
             else np.full(lambdas.size, -1, dtype=np.int64))
    dups = find_duplicates(X, c.B if problem == "GHEP" else None, dup_tol) if lambdas.size else []
    dup_of = {d["index"]: d["duplicate_of"] for d in dups}
    ref_vals = np.asarray(ref.eigenvalues, dtype=complex) if ref is not None else None
    # A pooled reference holds computed values, partly this solver's own: it decides completeness,
    # but an "error" against it would be zero by construction for its contributors.
    errors_meaningful = ref is not None and ref.info.get("kind") != POOLED_KIND
    for i, p in enumerate(pairs):
        p.update(index=i, kind=kinds[i], ref_index=None, ref_lambda_re=None, ref_lambda_im=None,
                 eig_abs_error=None, eig_rel_error=None, eig_scaled_error=None,
                 duplicate=i in dup_of, duplicate_of=dup_of.get(i))
        if match[i] >= 0:
            mu = ref_vals[match[i]]
            p.update(ref_index=int(match[i]), ref_lambda_re=float(mu.real), ref_lambda_im=float(mu.imag))
            if errors_meaningful:
                err = float(abs(lambdas[i] - mu))
                p.update(eig_abs_error=err, eig_rel_error=err / abs(mu) if mu != 0 else None,
                         eig_scaled_error=err / (abs(mu) + scale) if abs(mu) + scale > 0 else None)

    exp = expected_set(ref_vals, tgt, scale, tie_rtol) if ref is not None and ref_vals.size else None
    completeness = assess_completeness(tgt, lambdas, ref, exp, match, lambdas.size - len(dups), n)
    clusters = cluster_records(ref, match, exp, X, scale, cluster_rtol) if exp is not None else []
    b_orth = b_orthogonality(X, c.B) if problem != "GHIEP" and lambdas.size else None
    fmt = format_checks(problem, tgt, lam_real, X, c.B, scale, tie_rtol)

    reasons = []
    if not pairs:
        reasons.append("no eigenpairs returned")
    bad = sum(not p["accepted"] for p in pairs)
    if bad:
        reasons.append(f"{bad} of {len(pairs)} pairs have backward error > tol={tol:g}")
    if dups:
        reasons.append(f"{len(dups)} duplicate eigenvector(s)")
    if completeness["status"] in ("incomplete", "inconsistent"):
        reasons.append(f"completeness {completeness['status']}: {len(completeness['missing'])} missing, "
                       f"boundary deficit {completeness['boundary_deficit']}, "
                       f"{completeness['distinct_returned']} distinct returned")
    if completeness["spurious"]:
        reasons.append(f"{len(completeness['spurious'])} spurious eigenvalue(s) absent from the complete reference")

    report = res.report
    verdict = "fail" if reasons else "pass"
    summary = {
        "n_pairs": len(pairs), "n_real": kinds.count("real"), "n_complex": kinds.count("complex"),
        "n_accepted": len(pairs) - bad, "n_duplicates": len(dups),
        "max_abs_residual": _max(p["abs_residual"] for p in pairs),
        "max_rel_residual": _max(p["rel_residual"] for p in pairs),
        "max_backward_error": _max(p["backward_error"] for p in pairs),
        "max_eig_abs_error": _max(p["eig_abs_error"] for p in pairs),
        "max_eig_rel_error": _max(p["eig_rel_error"] for p in pairs),
        "max_eig_scaled_error": _max(p["eig_scaled_error"] for p in pairs),
        "max_subspace_angle": _max(r["max_angle"] for r in clusters),
        "b_orthogonality": b_orth,
        "solver_claim": _claim(report.get("success"), verdict),
    }
    return {
        "schema_version": cio.SCHEMA_VERSION,
        "case": c.name,
        "solver": solver or report.get("solver") or Path(result_dir).name,
        "inputs": {"case": str(c.path), "result": str(Path(result_dir)),
                   "reference": ref_source if stored_ref is not None else None},
        "problem": problem, "n": n, "nnz_a": int(c.A.nnz), "nnz_b": int(c.B.nnz) if c.B is not None else 0,
        "complex_scalars": common.is_complex(c),
        "config": c.config, "target": tgt, "tol": tol,
        "tolerances": {"match_rtol": match_rtol, "cluster_rtol": cluster_rtol, "tie_rtol": tie_rtol,
                       "dup_tol": dup_tol},
        "time_total_s": report.get("time_total_s"), "peak_rss_bytes": _reported_peak(report),
        "measured": _measured(guard),
        "solver_success": report.get("success"),
        "solver_warnings": list(report.get("warnings", [])),
        "norms": {"norm2_a": norms.norm2_a, "norm2_b": norms.norm2_b, "scale": scale,
                  "norm_method": {"A": norms.method_a, "B": norms.method_b}},
        "reference": None if stored_ref is None else {
            "kind": stored_ref.info.get("kind"), "comparable": comparable, "complete": stored_ref.info.get("complete"),
            "window": stored_ref.info.get("window"), "count": int(stored_ref.eigenvalues.size),
            "has_eigenvectors": stored_ref.eigenvectors is not None},
        "pairs": pairs,
        "duplicates": dups,
        "completeness": completeness,
        "clusters": clusters,
        "format": fmt,
        "summary": summary,
        "verdict": verdict,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a solver result directory against a case.")
    ap.add_argument("case", help="case name (under ROOT/cases) or case directory")
    ap.add_argument("result", type=Path, help="result directory (docs/FORMATS.md §3)")
    ap.add_argument("--root", default=str(cio.DEFAULT_ROOT), help="data root (default: %(default)s)")
    ap.add_argument("--out", type=Path, help=f"output JSON (default: RESULT/{EVAL_FILE})")
    ap.add_argument("--solver", help="solver label (default: report.json 'solver' or the directory name)")
    ap.add_argument("--tol", type=float, help="acceptance threshold on eta (default: config tol)")
    ap.add_argument("--match-rtol", type=float, default=MATCH_RTOL)
    ap.add_argument("--cluster-rtol", type=float, default=CLUSTER_RTOL)
    ap.add_argument("--tie-rtol", type=float, help=f"target-set tie band (default: {TIE_FACTOR:g} * tol)")
    ap.add_argument("--dup-tol", type=float, default=DUP_TOL)
    ap.add_argument("--dense-max", type=int, default=common.DENSE_MAX)
    ap.add_argument("--guard", type=Path, help="tools/run/guard.py --summary JSON of the solver run")
    args = ap.parse_args(argv)
    common.require_guard("evaluate")
    guard = json.loads(args.guard.read_text()) if args.guard else None
    doc = evaluate(args.case, args.result, root=args.root, solver=args.solver, tol=args.tol,
                   match_rtol=args.match_rtol, cluster_rtol=args.cluster_rtol, tie_rtol=args.tie_rtol,
                   dup_tol=args.dup_tol, dense_max=args.dense_max, guard=guard)
    out = args.out or args.result / EVAL_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    s = doc["summary"]
    print(f"{doc['case']} [{doc['solver']}]: {doc['verdict']}  pairs={s['n_pairs']} "
          f"max_eta={s['max_backward_error']} completeness={doc['completeness']['status']} -> {out}")
    for reason in doc["reasons"]:
        print(f"  - {reason}")
    for issue in doc["format"]["issues"]:
        print(f"  format: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
