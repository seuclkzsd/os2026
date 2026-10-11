#!/usr/bin/env python3
"""SLEPc reference solver (strong open-source baseline): Krylov-Schur + shift-and-invert with an
MUMPS factorization - the same direct solver hsolve uses - so the comparison isolates the
eigensolver/orchestration layer.

Runs in the separate env `os2026-ref` (petsc4py/slepc4py 3.26, MUMPS 5.8.2 MPI), under MPI:

    mpiexec -n P --bind-to none -x OMP_NUM_THREADS=T -x OPENBLAS_NUM_THREADS=T \\
        $REF/bin/python tools/bench/baseline_slepc.py --case NAME --out DIR

(always through tools/run/guard.py). Configuration choices that favour SLEPc are deliberate:
the problem type comes from the case metadata (hsolve detects it itself), and B HPD uses a
Cholesky (MUMPS SYM=1) factorization. The convergence test is EPS_CONV_NORM,
||r|| / (||A|| + |lambda| ||B||), the same backward-error notion hsolve accepts with.
Results are written with tools/case.save_result (docs/FORMATS.md §3), so tools/eval can check
them exactly like hsolve's.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402  (numpy/scipy only)

import petsc4py  # noqa: E402

petsc4py.init(sys.argv[:1] + [a for a in sys.argv[1:] if a.startswith("-") and not a.startswith("--")])
from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402


def distributed_aij(comm, m, sym_hint: str | None):
    """PETSc MPIAIJ from a scipy CSR matrix; every rank holds the full CSR and inserts its rows."""
    n = m.shape[0]
    layout = PETSc.Vec().createMPI(n, comm=comm)
    r0, r1 = layout.getOwnershipRange()
    layout.destroy()
    ip = m.indptr[r0:r1 + 1]
    rows = (ip - ip[0]).astype(PETSc.IntType)
    cols = m.indices[ip[0]:ip[-1]].astype(PETSc.IntType)
    vals = m.data[ip[0]:ip[-1]].astype(PETSc.ScalarType)
    a = PETSc.Mat().createAIJ(size=((r1 - r0, n), (r1 - r0, n)), csr=(rows, cols, vals), comm=comm)
    a.assemble()
    a.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    a.setOption(PETSc.Mat.Option.HERMITIAN, True)
    if sym_hint == "spd":
        a.setOption(PETSc.Mat.Option.SPD, True)
    return a


def gather_to_zero(comm, value: float) -> list[float]:
    """Gather one float per rank onto rank 0 using a PETSc vector (no mpi4py dependency)."""
    v = PETSc.Vec().createMPI((1, comm.getSize()), comm=comm)
    v.setValue(comm.getRank(), value)
    v.assemble()
    scatter, v0 = PETSc.Scatter.toZero(v)
    scatter.scatter(v, v0, False, PETSc.Scatter.Mode.FORWARD)
    out = [float(x) for x in v0.getArray()] if comm.getRank() == 0 else []
    scatter.destroy(), v0.destroy(), v.destroy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", default=str(case.DEFAULT_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=None, help="default: the case config's tol")
    ap.add_argument("--ncv", type=int, default=0)
    ap.add_argument("--max-it", type=int, default=1000)
    args, _ = ap.parse_known_args()

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    t0 = time.perf_counter()
    cs = case.load_case(args.case, root=args.root)
    target = cs.config["target"]
    if target.get("mode", "nearest") != "nearest":
        raise SystemExit("baseline_slepc: only 'nearest' targets are implemented")
    sigma, k = float(target["sigma"]), int(target["k"])
    tol = args.tol if args.tol is not None else float(cs.config.get("tol", 1e-10))
    problem = cs.meta["problem"]
    if np.iscomplexobj(cs.A.data) or (cs.B is not None and np.iscomplexobj(cs.B.data)):
        if PETSc.ScalarType is not np.complex128:
            raise SystemExit("baseline_slepc: complex case needs a complex-scalar PETSc build")

    a = distributed_aij(comm, cs.A, None)
    b = distributed_aij(comm, cs.B, "spd" if problem == "GHEP" else None) if cs.B is not None else None
    t_load = time.perf_counter() - t0

    eps = SLEPc.EPS().create(comm)
    eps.setOperators(a, b)
    eps.setProblemType({"HEP": SLEPc.EPS.ProblemType.HEP, "GHEP": SLEPc.EPS.ProblemType.GHEP,
                        "GHIEP": SLEPc.EPS.ProblemType.GHIEP}[problem])
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=k, ncv=args.ncv if args.ncv > 0 else PETSc.DECIDE)
    eps.setTarget(sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTolerances(tol, args.max_it)
    eps.setConvergenceTest(SLEPc.EPS.Conv.NORM)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.CHOLESKY)  # MUMPS LDL^T (SYM=2) for indefinite K, SYM=1 when SPD
    pc.setFactorSolverType("mumps")
    eps.setFromOptions()  # allow -mat_mumps_icntl_* etc. on the command line

    t1 = time.perf_counter()
    eps.setUp()  # includes the MUMPS analysis + factorization of A - sigma B
    t_setup = time.perf_counter() - t1
    t2 = time.perf_counter()
    eps.solve()
    t_solve = time.perf_counter() - t2
    nconv = eps.getConverged()
    its = eps.getIterationNumber()

    m = min(nconv, k)
    lambdas, vectors, pairs = [], [], []
    xr, _ = a.createVecs()
    for i in range(m):
        lam = eps.getEigenpair(i, xr)
        err = eps.computeError(i, SLEPc.EPS.ErrorType.BACKWARD)
        scatter, x0 = PETSc.Scatter.toZero(xr)
        scatter.scatter(xr, x0, False, PETSc.Scatter.Mode.FORWARD)
        if rank == 0:
            lambdas.append(lam.real)
            vectors.append(np.array(x0.getArray()).copy())
            pairs.append({"lambda_re": float(lam.real), "lambda_im": float(lam.imag),
                          "backward_error": float(err), "accepted": bool(err <= tol)})
        scatter.destroy()
        x0.destroy()

    rss = gather_to_zero(comm, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0)
    total = time.perf_counter() - t0
    if rank == 0:
        n = cs.n
        x = np.column_stack(vectors) if vectors else np.zeros((n, 0))
        report = {
            "solver": "slepc-krylovschur-sinvert-mumps",
            "slepc_version": ".".join(map(str, SLEPc.Sys.getVersion())),
            "mpi_ranks": comm.getSize(),
            "problem": problem, "n": n, "nnz_a": int(cs.A.nnz), "nnz_b": int(cs.B.nnz) if cs.B is not None else 0,
            "sigma_requested": sigma, "k": k, "tol": tol, "nconv": int(nconv), "iterations": int(its),
            "time_total_s": total, "time_load_s": t_load, "time_setup_s": t_setup, "time_solve_s": t_solve,
            "peak_rss_bytes_sum": int(sum(rss)), "peak_rss_bytes_max_rank": int(max(rss)),
            "pairs": pairs, "success": bool(m == k and all(p["accepted"] for p in pairs)),
            "warnings": [] if m == k else [f"only {nconv} pairs converged"],
        }
        case.save_result(args.out, lambdas, x, report)
        print(f"slepc: n={n} k={k} nconv={nconv} its={its} setup={t_setup:.2f}s solve={t_solve:.2f}s "
              f"total={total:.2f}s rss_sum={sum(rss) / 2**30:.2f}GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
