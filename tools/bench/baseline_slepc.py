#!/usr/bin/env python3
"""SLEPc reference solver (strong open-source baseline): Krylov-Schur + shift-and-invert with an
MUMPS factorization - the same direct solver hsolve uses - so the comparison isolates the
eigensolver/orchestration layer.

Runs in the separate env `os2026-ref` (petsc4py/slepc4py 3.26, MUMPS 5.8.2 MPI), under MPI:

    mpiexec -n P --bind-to none -x OMP_NUM_THREADS=T -x OPENBLAS_NUM_THREADS=T \\
        $REF/bin/python tools/bench/baseline_slepc.py --case NAME --out DIR [PETSc options]

(always through tools/run/guard.py; tools/bench/tune_slepc.py and compare.py build this command).
Arguments starting with "--" are this script's own (each takes one value: "--name value" or
"--name=value"); every other argument is handed to PETSc in order, e.g.
"-st_mat_mumps_icntl_7 5 -eps_ncv 40 -log_view :log.txt". The KSP/PC/factor of the spectral
transformation carry the prefix "st_", so MUMPS controls are -st_mat_mumps_icntl_<i> /
-st_mat_mumps_cntl_<i>. Options PETSc never queried are listed in report.json "unused_options"
(a misspelt option would otherwise silently tune nothing).

Configuration choices that favour SLEPc are deliberate: the problem type comes from the case
metadata (hsolve detects it itself) and B of a GHEP is flagged SPD. K = A - sigma B is factorized by
MUMPS LDL^T (PCCHOLESKY, SYM=2). The convergence test is EPS_CONV_NORM,
||r|| / (||A|| + |lambda| ||B||), the same backward-error notion hsolve accepts with (SLEPc takes
||.||_inf, an upper bound of the 2-norm, so its estimate is the more lenient one; tools/eval checks
the results with conservative 2-norm estimates). Targets: 'nearest' (Krylov-Schur, which =
TARGET_MAGNITUDE) and 'interval' (Krylov-Schur spectrum slicing with MUMPS inertia; HEP/GHEP);
--target replaces the case config's target (JSON, docs/FORMATS.md §2 Target keys).

Timing (report.json; every phase starts and ends at an MPI barrier, so rank 0's clock is the
wall time of the phase on all ranks):
    time_setup_s    EPSSetUp: MUMPS analysis + numeric factorization (+ slicing's inertia counts)
    time_solve_s    EPSSolve + the eigenvector computation it defers (EPSComputeVectors, run by
                    the first EPSGetEigenpair: purification for generalized problems), i.e. what
                    it takes to hold the eigenpairs; time_eps_solve_s / time_vectors_s split it
    time_load_s     case loading + PETSc assembly (not solver time); time_total_s everything
Each phase is a PETSc log stage (hs_setup, hs_solve, hs_vectors, hs_output), so -log_view breaks
it down per event (tools/bench/slepc_log.py parses it). peak_rss_bytes is the summed VmHWM of all
ranks over the solver phases (the high-water mark is reset after loading, Linux clear_refs; the
loaded case's arrays are freed first): the solver's own footprint, comparable with hsolve's
peak_rss_bytes. peak_rss_bytes_with_load (ru_maxrss, includes every rank's full copy of the
case) and the per-rank maximum are kept for reference.
Results are written with tools/case.save_result (docs/FORMATS.md §3), so tools/eval can check
them exactly like hsolve's.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import sys
import time
from pathlib import Path


def rank_cpus(allowed: list[int], rank: int, size: int) -> list[int]:
    """The rank-th of `size` equal contiguous blocks of the sorted allowed CPUs."""
    cpus = sorted(set(allowed))
    if size < 1 or not 0 <= rank < size or len(cpus) % size:
        raise ValueError(f"cannot split CPUs {cpus} into {size} equal blocks (rank {rank})")
    per = len(cpus) // size
    return cpus[rank * per:(rank + 1) * per]


def bind_rank(argv: list[str], env=os.environ) -> list[int] | None:
    """--bind core: pin this MPI rank to its block of the CPUs it inherited (the guard's cpuset),
    before numpy/OpenBLAS/OpenMP load, so every thread they create inherits the block. mpiexec's
    own --bind-to core is not used: it binds to cores outside the inherited cpuset (measured:
    ranks of a run pinned to CPUs 40,41 were bound to 0/48 and 1/49). Rank and size come from the
    launcher's environment (Open MPI / PMIx). Returns the CPUs, None without --bind core."""
    want = None
    for i, tok in enumerate(argv):
        if tok == "--bind" and i + 1 < len(argv):
            want = argv[i + 1]
        elif tok.startswith("--bind="):
            want = tok.split("=", 1)[1]
    if want in (None, "none"):
        return None
    if want != "core":
        raise SystemExit(f"baseline_slepc: --bind must be none or core, got {want!r}")
    rank = int(env.get("OMPI_COMM_WORLD_RANK", env.get("PMIX_RANK", "0")))
    size = int(env.get("OMPI_COMM_WORLD_SIZE", "1"))
    cpus = rank_cpus(sorted(os.sched_getaffinity(0)), rank, size)
    os.sched_setaffinity(0, cpus)
    return cpus


if __name__ == "__main__":
    BOUND_CPUS = bind_rank(sys.argv[1:])

import numpy as np  # noqa: E402  (after the rank binding)

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import case  # noqa: E402  (numpy/scipy only)

# Stage names of the -log_view breakdown (tools/bench/slepc_log.py reads them).
STAGES = ("hs_setup", "hs_solve", "hs_vectors", "hs_output")
PROBLEM_TYPES = ("HEP", "GHEP", "GHIEP")


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """(own, petsc): "--name value" / "--name=value" pairs are this script's own options, every
    other token goes to PETSc unchanged and in order (option values such as "-0.5" included)."""
    own: list[str] = []
    petsc: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and len(tok) > 2:
            own.append(tok)
            if "=" not in tok:
                if i + 1 >= len(argv):
                    raise SystemExit(f"baseline_slepc: option {tok} needs a value")
                own.append(argv[i + 1])
                i += 1
        else:
            petsc.append(tok)
        i += 1
    return own, petsc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", default=str(case.DEFAULT_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=None, help="default: the case config's tol")
    ap.add_argument("--ncv", type=int, default=0, help="0: SLEPc default (-eps_ncv overrides)")
    ap.add_argument("--max-it", type=int, default=1000)
    ap.add_argument("--bind", default="none", choices=("none", "core"),
                    help="core: pin each rank to its block of the inherited CPUs (bind_rank)")
    ap.add_argument("--target", default=None,
                    help='JSON Target replacing the config one, e.g. {"mode":"interval","lower":1,"upper":200}')
    return ap


def resolve_target(config_target: dict, override: str | None) -> dict:
    """The query: the case config's target, or the --target JSON. Only nearest and interval."""
    tgt = dict(json.loads(override)) if override else dict(config_target)
    mode = tgt.get("mode", "nearest")
    tgt["mode"] = mode
    if mode == "nearest":
        tgt["sigma"] = float(tgt.get("sigma", 0.0))
        tgt["k"] = int(tgt["k"])
        if tgt["k"] < 1:
            raise SystemExit("baseline_slepc: target k must be >= 1")
    elif mode == "interval":
        tgt["lower"], tgt["upper"] = float(tgt["lower"]), float(tgt["upper"])
        if not tgt["lower"] < tgt["upper"]:
            raise SystemExit("baseline_slepc: interval target needs lower < upper")
    else:
        raise SystemExit(f"baseline_slepc: target mode {mode!r} is not implemented (nearest, interval)")
    return tgt


def petsc_option_names(petsc_args: list[str]) -> list[str]:
    """Names (without '-') of the options in a PETSc argument list; values such as "-0.5" or
    ":log.txt" are skipped."""
    return [tok[1:] for tok in petsc_args if re.match(r"^-[A-Za-z]", tok)]


def option_defaults(mode: str) -> list[tuple[str, str]]:
    """PETSc options-database defaults (command-line values win). The shift-invert solver choice
    is set in the code AND here: multi-communicator spectrum slicing (-eps_krylovschur_partitions
    > 1) builds one ST/KSP/PC per partition from the options database (prefix st_) and silently
    falls back to PETSc's native Cholesky without them (measured: 190 s instead of MUMPS). Slicing
    also needs the inertia of the MUMPS factorization: the root front must not go to ScaLAPACK
    (ICNTL(13)=1) and null pivots are detected (ICNTL(24)=1, CNTL(3)), as in SLEPc's ex25."""
    out = [("st_ksp_type", "preonly"), ("st_pc_type", "cholesky"), ("st_pc_factor_mat_solver_type", "mumps")]
    if mode == "interval":
        out += [("st_mat_mumps_icntl_13", "1"), ("st_mat_mumps_icntl_24", "1"), ("st_mat_mumps_cntl_3", "1e-12")]
    return out


def result_order(lambdas: np.ndarray, tgt: dict) -> np.ndarray:
    """Permutation into the docs/FORMATS.md §3 order: nearest first (nearest), ascending (interval)."""
    lam = np.asarray(lambdas, dtype=float)
    key = np.abs(lam - tgt["sigma"]) if tgt["mode"] == "nearest" else lam
    return np.argsort(key, kind="stable")


def vm_hwm_bytes() -> int | None:
    """This process's VmHWM (peak resident set) in bytes, None where /proc is unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def reset_hwm() -> bool:
    """Reset VmHWM to the current RSS (Linux >= 4.0, /proc/self/clear_refs '5')."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    own, petsc_args = split_argv(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(own)

    import petsc4py
    petsc4py.init([sys.argv[0]] + petsc_args)
    from petsc4py import PETSc
    from slepc4py import SLEPc

    def distributed_aij(comm, m, spd: bool):
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
        if spd:
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

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    stages = {name: PETSc.Log.Stage(name) for name in STAGES}
    t0 = time.perf_counter()
    cs = case.load_case(args.case, root=args.root)
    tgt = resolve_target(cs.config.get("target", {}), args.target)
    tol = args.tol if args.tol is not None else float(cs.config.get("tol", 1e-10))
    problem = cs.meta["problem"]
    if problem not in PROBLEM_TYPES:
        raise SystemExit(f"baseline_slepc: unknown problem {problem!r}")
    if tgt["mode"] == "interval" and problem == "GHIEP":
        raise SystemExit("baseline_slepc: spectrum slicing needs a definite problem (HEP/GHEP)")
    if np.iscomplexobj(cs.A.data) or (cs.B is not None and np.iscomplexobj(cs.B.data)):
        if PETSc.ScalarType is not np.complex128:
            raise SystemExit("baseline_slepc: complex case needs a complex-scalar PETSc build")
    n, nnz_a, nnz_b = cs.n, int(cs.A.nnz), int(cs.B.nnz) if cs.B is not None else 0

    a = distributed_aij(comm, cs.A, False)
    b = distributed_aij(comm, cs.B, problem == "GHEP") if cs.B is not None else None
    del cs  # every rank held the full CSR: free it before the solver's memory is measured
    gc.collect()
    comm.barrier()
    t_load = time.perf_counter() - t0
    hwm_reset = reset_hwm()

    eps = SLEPc.EPS().create(comm)
    eps.setOperators(a, b)
    eps.setProblemType({"HEP": SLEPc.EPS.ProblemType.HEP, "GHEP": SLEPc.EPS.ProblemType.GHEP,
                        "GHIEP": SLEPc.EPS.ProblemType.GHIEP}[problem])
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    if tgt["mode"] == "nearest":
        eps.setDimensions(nev=tgt["k"], ncv=args.ncv if args.ncv > 0 else PETSc.DECIDE)
        eps.setTarget(tgt["sigma"])
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    else:
        eps.setInterval(tgt["lower"], tgt["upper"])
        eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    eps.setTolerances(tol, args.max_it)
    eps.setConvergenceTest(SLEPc.EPS.Conv.NORM)
    opts = PETSc.Options()
    for name, value in option_defaults(tgt["mode"]):
        if not opts.hasName(name):   # command-line values win
            opts.setValue(name, value)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.CHOLESKY)  # MUMPS LDL^T (SYM=2) for indefinite K, SYM=1 when SPD
    pc.setFactorSolverType("mumps")
    eps.setFromOptions()  # -eps_*, -st_*, -st_mat_mumps_icntl_* ... from the command line

    def phase(name, fn):
        comm.barrier()
        stages[name].push()
        t = time.perf_counter()
        out = fn()
        comm.barrier()
        dt = time.perf_counter() - t
        stages[name].pop()
        return out, dt

    _, t_setup = phase("hs_setup", eps.setUp)
    _, t_eps_solve = phase("hs_solve", eps.solve)
    nconv = eps.getConverged()
    its = eps.getIterationNumber()
    m = min(nconv, tgt["k"]) if tgt["mode"] == "nearest" else nconv

    def compute_vectors():
        vecs, vals = [], []
        for i in range(m):
            v = a.createVecLeft()
            vals.append(eps.getEigenpair(i, v))  # the first call runs EPSComputeVectors
            vecs.append(v)
        return vals, vecs

    (vals, vecs), t_vectors = phase("hs_vectors", compute_vectors)

    def output():
        lambdas, vectors, pairs = [], [], []
        for i in range(m):
            err = eps.computeError(i, SLEPc.EPS.ErrorType.BACKWARD)
            scatter, x0 = PETSc.Scatter.toZero(vecs[i])
            scatter.scatter(vecs[i], x0, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                lam = vals[i]
                lambdas.append(lam.real)
                vectors.append(np.array(x0.getArray()).copy())
                pairs.append({"lambda_re": float(lam.real), "lambda_im": float(lam.imag),
                              "backward_error": float(err), "accepted": bool(err <= tol)})
            scatter.destroy()
            x0.destroy()
        return lambdas, vectors, pairs

    (lambdas, vectors, pairs), t_output = phase("hs_output", output)

    hwm = vm_hwm_bytes()
    solver_peak = gather_to_zero(comm, float(hwm if hwm is not None and hwm_reset else -1))
    with_load = gather_to_zero(comm, float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0)
    opts = PETSc.Options()
    unused = [name for name in petsc_option_names(petsc_args) if not opts.used(name)]
    aff = sorted(os.sched_getaffinity(0))
    aff_first, aff_count = gather_to_zero(comm, float(aff[0])), gather_to_zero(comm, float(len(aff)))
    total = time.perf_counter() - t0
    if rank == 0:
        order = result_order(np.asarray(lambdas), tgt)
        lam_sorted = [lambdas[i] for i in order]
        x = np.column_stack([vectors[i] for i in order]) if vectors else np.zeros((n, 0))
        pairs = [pairs[i] for i in order]
        has_peak = all(v >= 0 for v in solver_peak)
        success = all(p["accepted"] for p in pairs) and (m == tgt["k"] if tgt["mode"] == "nearest" else True)
        warnings = [] if tgt["mode"] != "nearest" or m == tgt["k"] else [f"only {nconv} pairs converged"]
        if unused:
            warnings.append(f"unused PETSc options: {', '.join(unused)}")
        report = {
            "solver": "slepc-krylovschur-sinvert-mumps",
            "slepc_version": ".".join(map(str, SLEPc.Sys.getVersion())),
            "mpi_ranks": comm.getSize(),
            "petsc_options": petsc_args, "unused_options": unused, "bind": args.bind,
            "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
            "rank_cpus": [{"first": int(f), "count": int(c)} for f, c in zip(aff_first, aff_count)],
            "problem": problem, "n": n, "nnz_a": nnz_a, "nnz_b": nnz_b, "target": tgt,
            "sigma_requested": tgt.get("sigma"), "k": tgt.get("k"), "tol": tol, "nconv": int(nconv),
            "iterations": int(its),
            "time_total_s": total, "time_load_s": t_load, "time_setup_s": t_setup,
            "time_solve_s": t_eps_solve + t_vectors, "time_eps_solve_s": t_eps_solve,
            "time_vectors_s": t_vectors, "time_output_s": t_output,
            "peak_rss_bytes": int(sum(solver_peak)) if has_peak else int(sum(with_load)),
            "peak_rss_scope": "solver phases (VmHWM reset after loading)" if has_peak else "whole process (ru_maxrss)",
            "peak_rss_bytes_max_rank": int(max(solver_peak)) if has_peak else int(max(with_load)),
            "peak_rss_bytes_with_load": int(sum(with_load)),
            "pairs": pairs, "success": bool(success), "warnings": warnings,
        }
        case.save_result(args.out, lam_sorted, x, report)
        print(f"slepc: n={n} target={tgt['mode']} nconv={nconv} its={its} setup={t_setup:.2f}s "
              f"solve={t_eps_solve + t_vectors:.2f}s total={total:.2f}s "
              f"peak={report['peak_rss_bytes'] / 2**30:.2f}GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
