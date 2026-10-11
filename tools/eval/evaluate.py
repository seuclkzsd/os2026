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
reported in norm_method.

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
    smallest/largest: the k extreme ones; interval: all inside). Reference eigenvalues tied with the
    k-th (within cluster_rtol) are interchangeable: any of them may complete the set, so a
    multiplicity cluster cut by k accepts any subset of its eigenspace. Status:
        complete / incomplete      ref.complete and its window covers the needed range (certified)
        consistent / inconsistent  same test against an uncertified (solver) reference
        unknown                    no usable reference
    Independently of any reference, fewer than k distinct vectors is 'incomplete'.
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

NORM_TOL = 1e-8        # ARPACK relative accuracy of the ||.||_2 estimate
MATCH_RTOL = 1e-6
CLUSTER_RTOL = 1e-6
DUP_TOL = 1e-4
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
        w = sla.eigsh(m, k=1, which="LM", tol=NORM_TOL, v0=common.start_vector(n, m.dtype, 0),
                      return_eigenvectors=False)
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
    within rtol * (|mu| + scale) are interchangeable `optional` ones; `window` is the real range a
    reference must list completely for the set to be certain."""
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
                           "beyond_target": [], "unmatched": [], "spurious": [], "distinct_returned": n_distinct}
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


# ----------------------------------------------------------------------------------- driver

def _max(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(max(vals)) if vals else None


def _data_root(c: cio.Case, root: str | Path) -> Path:
    return c.path.parent.parent if c.path.parent.name == "cases" else Path(root)


def evaluate(case_ref: str | Path, result_dir: str | Path, *, root: str | Path = cio.DEFAULT_ROOT,
             solver: str | None = None, tol: float | None = None, match_rtol: float = MATCH_RTOL,
             cluster_rtol: float = CLUSTER_RTOL, dup_tol: float = DUP_TOL,
             dense_max: int = common.DENSE_MAX) -> dict[str, Any]:
    """Evaluation document (docs/FORMATS.md §4) for one (case, result directory)."""
    c = cio.load_case(case_ref, root=root)
    problem = common.problem_type(c)
    tgt = common.target(c)
    tol = common.tol(c) if tol is None else float(tol)
    res = cio.load_result(result_dir)
    data_root = _data_root(c, root)
    stored_ref = cio.load_ref(c.name, root=data_root)
    comparable = stored_ref is not None and stored_ref.info.get("kind") not in common.CONTINUUM_KINDS
    ref = stored_ref if comparable else None
    norms = estimate_norms(c.A, c.B, dense_max)
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
    for i, p in enumerate(pairs):
        p.update(index=i, kind=kinds[i], ref_index=None, ref_lambda_re=None, ref_lambda_im=None,
                 eig_abs_error=None, eig_rel_error=None, eig_scaled_error=None,
                 duplicate=i in dup_of, duplicate_of=dup_of.get(i))
        if match[i] >= 0:
            mu = ref_vals[match[i]]
            err = float(abs(lambdas[i] - mu))
            p.update(ref_index=int(match[i]), ref_lambda_re=float(mu.real), ref_lambda_im=float(mu.imag),
                     eig_abs_error=err, eig_rel_error=err / abs(mu) if mu != 0 else None,
                     eig_scaled_error=err / (abs(mu) + scale) if abs(mu) + scale > 0 else None)

    exp = expected_set(ref_vals, tgt, scale, cluster_rtol) if ref is not None and ref_vals.size else None
    completeness = assess_completeness(tgt, lambdas, ref, exp, match, lambdas.size - len(dups), n)
    clusters = cluster_records(ref, match, exp, X, scale, cluster_rtol) if exp is not None else []
    b_orth = b_orthogonality(X, c.B) if problem != "GHIEP" and lambdas.size else None

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
    }
    return {
        "schema_version": cio.SCHEMA_VERSION,
        "case": c.name,
        "solver": solver or report.get("solver") or Path(result_dir).name,
        "inputs": {"case": str(c.path), "result": str(Path(result_dir)),
                   "reference": str(data_root / "refs" / c.name) if stored_ref is not None else None},
        "problem": problem, "n": n, "nnz_a": int(c.A.nnz), "nnz_b": int(c.B.nnz) if c.B is not None else 0,
        "complex_scalars": common.is_complex(c),
        "config": c.config, "target": tgt, "tol": tol,
        "tolerances": {"match_rtol": match_rtol, "cluster_rtol": cluster_rtol, "dup_tol": dup_tol},
        "time_total_s": report.get("time_total_s"), "peak_rss_bytes": report.get("peak_rss_bytes"),
        "solver_success": report.get("success"),
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
        "summary": summary,
        "verdict": "fail" if reasons else "pass",
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
    ap.add_argument("--dup-tol", type=float, default=DUP_TOL)
    ap.add_argument("--dense-max", type=int, default=common.DENSE_MAX)
    args = ap.parse_args(argv)
    common.require_guard("evaluate")
    doc = evaluate(args.case, args.result, root=args.root, solver=args.solver, tol=args.tol,
                   match_rtol=args.match_rtol, cluster_rtol=args.cluster_rtol, dup_tol=args.dup_tol,
                   dense_max=args.dense_max)
    out = args.out or args.result / EVAL_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    s = doc["summary"]
    print(f"{doc['case']} [{doc['solver']}]: {doc['verdict']}  pairs={s['n_pairs']} "
          f"max_eta={s['max_backward_error']} completeness={doc['completeness']['status']} -> {out}")
    for reason in doc["reasons"]:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
