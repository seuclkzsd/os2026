#!/usr/bin/env python3
"""Tune the SLEPc baseline (tools/bench/baseline_slepc.py) per case and core budget P: find the
fastest configuration whose result passes tools/eval at the case tolerance, so that a speedup
claimed against it is a claim against the strongest open-source setup we can build.

Search space (SlepcConfig, tools/bench/slepc_profile.py) - greedy coordinate descent from the
untuned default (1 rank x P threads, SLEPc/MUMPS defaults); each group's candidates are measured
against the incumbent, which moves to the fastest VALID one if it is faster by more than --margin:
  accuracy   only while the incumbent is invalid: -eps_tol tol/10, -eps_nev k+2 / k+4,
             -eps_ncv 3k, and tol/10 with k+4. SLEPc's convergence test judges the Krylov
             residual estimate of the shift-inverted operator, and the true residual of a pair it
             accepts can exceed tol: baseline_slepc.py's EPSComputeError then rejects it as well
             (24 of the 25 backward-error failures of the 2-core tuning), so a tighter -eps_tol is
             the remedy. The ||.||_inf norm of EPS_CONV_NORM (an upper bound of ||.||_2) makes the
             test more lenient than tools/eval's only by ||A||_inf/||A||_2, about 1.3x on
             yee-pec-box-1e5. Single-vector Krylov can also miss copies of a multiple eigenvalue
             (the -eps_nev padding).
  layout     ranks x threads over the divisors of P (1xP, ..., Px1), --bind-to none | core
  ordering   MUMPS ICNTL(7) AMD/AMF/SCOTCH/PORD/METIS/QAMD (default 7 = automatic choice), and
             with >1 rank parallel analysis ICNTL(28)=2 with ICNTL(29) PT-SCOTCH / ParMETIS
  layout     again, with the chosen ordering
  mumps      ICNTL(48)=0 (L0-OpenMP tree parallelism off), ICNTL(58)=1 (symbolic factorization
             variant), ICNTL(13)=1 (no ScaLAPACK root, >1 rank), ICNTL(8)=0 (no scaling)
  blr        nearest: block low-rank factorization ICNTL(35)=2 with CNTL(7) 1e-12 / 1e-10
  krylov     nearest: -eps_ncv k+8/2k/3k/4k (default max(2k, k+15)), -eps_krylovschur_restart
             0.3/0.7, generalized: -eps_purify 0, GHIEP: -eps_gen_non_hermitian (GNHEP Krylov-Schur)
  prune      last: the incumbent without each one of its options; a removal wins when it stays
             valid and is not slower by more than --margin (greedy order can leave a setting
             behind that a later group made unnecessary)
  slicing    interval targets (spectrum slicing with MUMPS inertia): -eps_krylovschur_partitions
             (divisors of the rank count) and the per-shift -eps_krylovschur_nev 10/20/60
             (the per-shift ncv/mpd follow SLEPc's defaults for that nev)
The convergence test is fixed: EPS_CONV_NORM backward error at the case tol (baseline_slepc.py);
the tuner only tightens it (-eps_tol) when that is needed to pass the evaluation.

Metric: the solver time setup + solve + eigenvector computation from -log_view
(tools/bench/slepc_log.py), the same scope as hsolve's time_total_s; MPI/Python start-up and case
loading are excluded. Screening runs each configuration --screen-reps times (min); the final
confirmation runs the default, the descent's start (--from-profile: the stored profile), the best
and up to --confirm-top runners-up --confirm-reps times INTERLEAVED (rotating order) and compares
MEDIANS; a configuration is valid only if every confirmation run passes the evaluation.
Randomly failing cases: when a configuration of the case was valid in some runs and invalid in
others - in this tuning, or recorded as "flaky" in a stored profile of the case and target mode at
any core count (multi-rank MUMPS is not deterministic; SLEPc's GHIEP Krylov-Schur on
wg-square-1e5 misses eigenvalues in some runs, with 1 and with 2 ranks) -
every configuration still valid after the regular confirmation gets further interleaved runs up to
--flaky-confirm-reps (default 20), all of which it must pass. 5 passing runs cannot tell a
configuration that fails in a quarter of its runs from a sound one (it passes 5 of 5 with
probability 0.24); 20 can (0.003).

Every trial runs under tools/run/guard.py (nested: the tuner itself must run under guard, it
evaluates results in-process with the case norms estimated once). Output:
    results/raw/<tag>/<case>[@interval]/trials/<NNN>/{result/, log_view.txt, guard.json, stdout.txt}
    results/raw/<tag>/<case>[@interval]/tune.json      every trial, group decisions, confirmation
--groups restricts the descent to some groups and --from-profile starts it at the case's stored
profile, so a tuning can be extended (e.g. --groups blr --from-profile) without repeating it.
With --write-profiles the best configuration becomes the case's profile in
tools/bench/slepc_profiles.json (and class profiles are re-derived), which
tools/bench/compare.py --slepc-profile best uses. --report renders tune.json files as markdown.

usage:
  python tools/run/guard.py --skip-busy-check --cores 40,41 --mem-gb 4 --timeout 21600 -- \\
      python tools/bench/tune_slepc.py --cores 40,41 --mem-gb 3.5 --tag tune-p2 --write-profiles \\
      syn-mem27-46 wg-square-1e5
  python tools/bench/tune_slepc.py --report results/raw/tune-p2/*/tune.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.bench import slepc_log  # noqa: E402
from tools.bench.slepc_profile import (MUMPS, PROFILE_FILE, SlepcConfig, class_profiles,  # noqa: E402
                                       default_config, expand_value, load_profiles, lookup, mpiexec_command,
                                       parse_cpulist, save_profiles, upsert)

GUARD = REPO / "tools/run/guard.py"
ORDERINGS = {"0": "AMD", "2": "AMF", "3": "SCOTCH", "4": "PORD", "5": "METIS", "6": "QAMD"}
PAR_ORDERINGS = {"1": "PT-SCOTCH", "2": "ParMETIS"}


def layouts(cores: int) -> list[tuple[int, int, str]]:
    """(ranks, threads, bind) splits of a core budget: every divisor as rank count, each unbound
    (processes and threads float within the guard's cpuset) and bound (ranks pinned to disjoint
    core blocks, OpenMP threads to cores; not for a single serial process)."""
    out = []
    for r in range(1, cores + 1):
        if cores % r == 0:
            out.append((r, cores // r, "none"))
            if cores > 1:
                out.append((r, cores // r, "core"))
    return out


@dataclasses.dataclass
class Context:
    """What the candidate generators need to know about the tuning problem."""
    cores: int
    problem: str            # HEP | GHEP | GHIEP
    mode: str               # nearest | interval
    k: int | None           # nearest: wanted pairs
    tol: float


Group = Callable[[SlepcConfig, Context], list[SlepcConfig]]


def g_accuracy(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    tight = f"{ctx.tol / 10:g}"
    cands = [inc.with_options(eps_tol=tight)]
    if ctx.mode == "nearest":
        cands += [inc.with_options(eps_nev="k+2"), inc.with_options(eps_nev="k+4"),
                  inc.with_options(eps_ncv="3k"), inc.with_options(eps_tol=tight, eps_nev="k+4")]
    return cands


def g_layout(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    # parallel analysis / partitions the layout cannot use are dropped, not the layout
    return [inc.with_layout(r, t, b).fitted() for r, t, b in layouts(ctx.cores)]


def g_ordering(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    base = inc.with_options(**{MUMPS + "icntl_28": None, MUMPS + "icntl_29": None})
    out = [base.with_options(**{MUMPS + "icntl_7": None})]
    out += [base.with_options(**{MUMPS + "icntl_7": o}) for o in ORDERINGS]
    if inc.ranks > 1:
        out += [base.with_options(**{MUMPS + "icntl_7": None, MUMPS + "icntl_28": "2", MUMPS + "icntl_29": p})
                for p in PAR_ORDERINGS]
    return out


def g_mumps(name: str, value: str, when: Callable[[SlepcConfig, Context], bool] = lambda c, x: True) -> Group:
    def group(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
        return [inc.with_options(**{MUMPS + name: value})] if when(inc, ctx) else []
    group.__name__ = f"mumps_{name}"
    return group


def g_ncv(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    if ctx.mode != "nearest":
        return []
    nev = int(expand_value(inc.option("eps_nev") or "k", ctx.k))
    seen = {inc.option("eps_ncv") and expand_value(inc.option("eps_ncv"), ctx.k)
            or str(max(2 * nev, nev + 15))}   # SLEPc's default ncv
    out = []
    for rule in ("k+8", "2k", "3k", "4k", None):
        value = str(max(2 * nev, nev + 15)) if rule is None else expand_value(rule, ctx.k)
        if value in seen or int(value) <= nev:
            continue
        seen.add(value)
        out.append(inc.with_options(eps_ncv=rule))
    return out


def g_restart(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    return [inc.with_options(eps_krylovschur_restart=v) for v in ("0.3", "0.7")] if ctx.mode == "nearest" else []


def g_purify(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    return [inc.with_options(eps_purify="0")] if ctx.problem != "HEP" else []


def g_problem_type(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    return [inc.with_options(eps_gen_non_hermitian="1")] if ctx.problem == "GHIEP" else []


def g_partitions(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    if ctx.mode != "interval":
        return []
    out = [inc.with_options(eps_krylovschur_partitions=p) for p in range(1, inc.ranks + 1)
           if inc.ranks % p == 0 and str(p) != (inc.option("eps_krylovschur_partitions") or "1")]
    return [c.with_options(eps_krylovschur_partitions=None) if c.option("eps_krylovschur_partitions") == "1"
            else c for c in out]


def g_slice_nev(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    if ctx.mode != "interval":
        return []
    return [inc.with_options(eps_krylovschur_nev=v) for v in ("10", "20", "60")
            if v != inc.option("eps_krylovschur_nev")]


def g_blr(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    """Block low-rank factorization (ICNTL(35)=2) at two compression tolerances CNTL(7); whether
    the eigenpairs of the perturbed operator still pass the evaluation decides (no refinement)."""
    return [inc.with_options(**{MUMPS + "icntl_35": "2", MUMPS + "cntl_7": eps}) for eps in ("1e-12", "1e-10")]


def g_prune(inc: SlepcConfig, ctx: Context) -> list[SlepcConfig]:
    """The incumbent without one of its options each: a setting an earlier group needed (e.g. the
    accuracy padding) may be unnecessary after a later one (e.g. no purification)."""
    out = []
    for name, _ in inc.options:
        c = inc.with_options(**{name: None})
        if name == MUMPS + "icntl_28":      # parallel analysis comes as a pair
            c = c.with_options(**{MUMPS + "icntl_29": None})
        elif name == MUMPS + "icntl_29":
            continue
        elif name == MUMPS + "icntl_35":    # so does BLR
            c = c.with_options(**{MUMPS + "cntl_7": None})
        elif name == MUMPS + "cntl_7" and inc.option(MUMPS + "icntl_35"):
            continue
        out.append(c)
    return out


# Groups whose candidates only remove settings: they win when valid and not slower (pick()).
SIMPLIFYING = ("prune",)


def plan(mode: str, groups: list[str] | None = None) -> list[tuple[str, Group]]:
    """Ordered (name, generator) groups of the coordinate descent (module docstring), restricted
    to `groups` when given (names as listed here; unknown names are an error)."""
    multi_rank = lambda c, x: c.ranks > 1  # noqa: E731
    threaded = lambda c, x: c.threads > 1  # noqa: E731
    mumps = [("icntl_48", g_mumps("icntl_48", "0", threaded)), ("icntl_58", g_mumps("icntl_58", "1")),
             ("icntl_13", g_mumps("icntl_13", "1", multi_rank)), ("icntl_8", g_mumps("icntl_8", "0"))]
    if mode == "interval":
        full = [("accuracy", g_accuracy), ("layout", g_layout), ("partitions", g_partitions),
                ("slice_nev", g_slice_nev), ("ordering", g_ordering), ("layout2", g_layout),
                ("partitions2", g_partitions)] + mumps[:2] + [("prune", g_prune)]
    else:
        full = ([("accuracy", g_accuracy), ("layout", g_layout), ("ordering", g_ordering), ("layout2", g_layout)]
                + mumps + [("blr", g_blr), ("ncv", g_ncv), ("restart", g_restart), ("purify", g_purify),
                           ("problem_type", g_problem_type), ("prune", g_prune)])
    if groups is None:
        return full
    unknown = sorted(set(groups) - {n for n, _ in full})
    if unknown:
        raise SystemExit(f"tune_slepc: unknown group(s) {unknown} for {mode} targets; "
                         f"known: {[n for n, _ in full]}")
    return [(n, g) for n, g in full if n in groups]


# --------------------------------------------------------------------------------- measuring

def trial_time(t: dict[str, Any]) -> float | None:
    """Solver time of a trial: -log_view setup + solve + vectors, else report's setup + solve."""
    ph = t.get("phases")
    if ph and ph.get("solver_s") is not None:
        return float(ph["solver_s"])
    rep = t.get("report") or {}
    if rep.get("time_setup_s") is not None and rep.get("time_solve_s") is not None:
        return float(rep["time_setup_s"]) + float(rep["time_solve_s"])
    return None


class Tuner:
    def __init__(self, case_name: str, args: argparse.Namespace, target_override: dict[str, Any] | None):
        from tools import case as cio
        from tools.eval import evaluate
        self.cio, self.evaluate = cio, evaluate
        self.args = args
        c = cio.load_case(case_name, root=args.root)
        self.config_target = dict(c.config.get("target", {}))
        if target_override is not None:
            c = dataclasses.replace(c, config={**c.config, "target": target_override})
        self.case = c
        tgt = dict(c.config.get("target", {}))
        self.mode = tgt.get("mode", "nearest")
        self.target = tgt
        self.target_override = target_override
        self.ctx = Context(cores=len(parse_cpulist(args.cores)), problem=c.meta["problem"], mode=self.mode,
                           k=int(tgt["k"]) if self.mode == "nearest" else None,
                           tol=float(c.config.get("tol", 1e-10)))
        self.dir = REPO / "results/raw" / args.tag / (case_name + ("" if self.mode == "nearest" else f"@{self.mode}"))
        self.trials: list[dict[str, Any]] = []
        self.cache: dict[str, list[dict[str, Any]]] = {}
        # random failures recorded by an earlier tuning of this case and target mode, at any core
        # count (they come from the solver, not the layout): confirm with the long series at once
        self.flaky = any(p["scope"] == "case" and p["name"] == c.name and p["mode"] == self.mode
                         and p.get("tuned", {}).get("flaky") for p in load_profiles(args.profiles_file))
        t0 = time.perf_counter()
        self.norms = evaluate.estimate_norms(c.A, c.B)
        self.norm_time = time.perf_counter() - t0

    def log(self, msg: str) -> None:
        print(f"[tune {self.case.name}] {msg}", flush=True)

    def run(self, cfg: SlepcConfig, phase: str, timeout: float) -> dict[str, Any]:
        """One measured trial of cfg (measure()), judged (validity()) and recorded."""
        t = self.measure(cfg, phase, timeout, len(self.trials))
        t["valid"], t["why"] = self.validity(t)
        t["solver_s"] = trial_time(t)
        self.trials.append(t)
        self.cache.setdefault(t["label"], []).append(t)
        ph = t["phases"] or {}
        self.log(f"#{t['id']:03d} {phase:12s} {t['label']:70s} solver={_fmt(t['solver_s'])} "
                 f"ana={_fmt(ph.get('analysis_s'))} num={_fmt(ph.get('numeric_s'))} "
                 f"solves={ph.get('solves')}/{_fmt(ph.get('solve_s'))} wall={t['guard'].get('wall_s')} "
                 f"{'VALID' if t['valid'] else 'invalid: ' + str(t['why'])}")
        self.save()
        return t

    def measure(self, cfg: SlepcConfig, phase: str, timeout: float, tid: int) -> dict[str, Any]:
        """One guarded, evaluated baseline run of cfg: the trial record before its verdict."""
        tdir = self.dir / "trials" / f"{tid:03d}"
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True)
        res, log_path, gpath = tdir / "result", tdir / "log_view.txt", tdir / "guard.json"
        cmd = mpiexec_command(cfg, self.case.name, res, k=self.ctx.k, target=self.target_override,
                              log_view=log_path, root=self.args.root)
        g = [sys.executable, str(GUARD), "--cores", self.args.cores, "--threads", str(cfg.cores),
             "--mem-gb", str(self.args.mem_gb), "--timeout", str(int(timeout)), "--summary", str(gpath),
             "--skip-busy-check", "--"]
        proc = subprocess.run(g + cmd, cwd=REPO, capture_output=True, text=True)
        (tdir / "stdout.txt").write_text((proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:]))
        guard = json.loads(gpath.read_text()) if gpath.exists() else {"returncode": proc.returncode,
                                                                       "killed": "no guard summary"}
        t: dict[str, Any] = {"id": tid, "phase": phase, "label": cfg.label(), "config": cfg.to_json(),
                             "guard": {k: guard.get(k) for k in ("returncode", "killed", "wall_s", "peak_rss_gib")},
                             "report": None, "phases": slepc_log.read_phases(log_path), "eval": None}
        if (res / "report.json").exists():
            rep = json.loads((res / "report.json").read_text())
            t["report"] = {k: rep.get(k) for k in ("time_setup_s", "time_solve_s", "time_eps_solve_s", "time_vectors_s",
                                                   "time_load_s", "nconv", "iterations", "peak_rss_bytes",
                                                   "unused_options", "success", "mpi_ranks")}
            try:
                doc = self.evaluate.evaluate(self.case, res, norms=self.norms, guard=guard)
                t["eval"] = {"verdict": doc["verdict"], "reasons": doc["reasons"],
                             "max_eta": doc["summary"]["max_backward_error"],
                             "completeness": doc["completeness"]["status"], "pairs": doc["summary"]["n_pairs"]}
            except Exception as err:  # malformed result: an invalid trial, not a crash of the tuner
                t["eval"] = {"verdict": "fail", "reasons": [f"evaluation failed: {type(err).__name__}: {err}"]}
            if not self.args.keep_vectors:
                for f in ("eigenvectors.npy", "complex_eigenvectors.npy"):
                    (res / f).unlink(missing_ok=True)
        return t

    def validity(self, t: dict[str, Any]) -> tuple[bool, str | None]:
        g = t["guard"]
        if g.get("killed") or g.get("returncode") not in (0,):
            return False, f"run failed (rc={g.get('returncode')}, killed={g.get('killed')})"
        if t["report"] is None:
            return False, "no report.json"
        if t["report"].get("unused_options"):
            return False, f"unused options {t['report']['unused_options']}"
        if t["eval"] is None or t["eval"]["verdict"] != "pass":
            return False, "; ".join((t["eval"] or {}).get("reasons", ["not evaluated"]))[:300]
        if trial_time(t) is None:
            return False, "no timing"
        return True, None

    def screen(self, cfg: SlepcConfig, phase: str, timeout: float) -> dict[str, Any]:
        """Screening measurement of cfg (cached by label): {valid, time = min over reps}."""
        runs = list(self.cache.get(cfg.label(), []))
        while len(runs) < self.args.screen_reps:
            runs.append(self.run(cfg, phase, timeout))
            if not runs[-1]["valid"]:
                break
        valid = all(r["valid"] for r in runs)
        times = [r["solver_s"] for r in runs if r["solver_s"] is not None]
        return {"config": cfg, "valid": valid, "time": min(times) if times and valid else None,
                "wall": max((r["guard"].get("wall_s") or 0) for r in runs)}

    def tune(self) -> dict[str, Any]:
        args = self.args
        default = default_config(self.ctx.cores)
        self.log(f"{self.ctx} norms in {self.norm_time:.1f}s -> {self.dir}")
        d = self.screen(default, "default", args.timeout)
        inc = d
        decisions = [{"group": "default", "incumbent": default.label(), "time": d["time"], "valid": d["valid"]}]
        screened = {default.label(): d}
        start = self.start_config()
        if start is not None and start.label() != default.label():
            inc = self.screen(start, "start", args.timeout)
            screened[start.label()] = inc
            decisions.append({"group": "start", "incumbent": start.label(), "time": inc["time"], "valid": inc["valid"]})
            if not inc["valid"] and d["valid"]:
                inc = d
        groups = args.groups.split(",") if args.groups else None
        for name, gen in plan(self.mode, groups):
            if name == "accuracy" and inc["valid"]:
                continue
            cands = [c for c in gen(inc["config"], self.ctx) if c.label() != inc["config"].label()]
            if not cands:
                continue
            ref_wall = inc["wall"] or args.timeout
            timeout = min(args.timeout, max(args.min_trial_timeout, args.timeout_factor * ref_wall))
            results = []
            for c in cands:
                r = self.screen(c, name, timeout)
                screened[c.label()] = r
                results.append(r)
            if name in SIMPLIFYING and inc["valid"]:
                # dropping an option needs no gain: the simpler one wins unless it is slower
                best = pick(results + [inc], args.margin)
                moved = best is not inc
            else:
                best = pick(results, args.margin)
                moved = best is not None and (not inc["valid"] or best["time"] < inc["time"] * (1.0 - args.margin))
            if moved:
                inc = best
            decisions.append({"group": name, "candidates": [r["config"].label() for r in results],
                              "incumbent": inc["config"].label(), "time": inc["time"], "valid": inc["valid"],
                              "moved": moved})
            self.log(f"group {name}: {'-> ' if moved else 'keep '}{inc['config'].label()} ({_fmt(inc['time'])})")
        confirm = self.confirm(default, inc, screened, start)
        return self.summary(default, decisions, confirm)

    def stored_profile(self) -> tuple[SlepcConfig, dict[str, Any]] | None:
        """The case's own stored profile for this core count and target (None without one)."""
        hit = lookup(load_profiles(self.args.profiles_file), self.case.name, self.ctx.problem, self.mode,
                     self.ctx.cores, target=self.target, config_target=self.config_target)
        return hit if hit is not None and hit[1]["scope"] == "case" else None

    def start_config(self) -> SlepcConfig | None:
        """--from-profile: the case's stored profile for this core count and target."""
        if not self.args.from_profile:
            return None
        hit = self.stored_profile()
        if hit is None:
            self.log("no stored case profile: starting from the default")
            return None
        return hit[0]

    def confirm(self, default: SlepcConfig, inc: dict[str, Any], screened: dict[str, dict[str, Any]],
                start: SlepcConfig | None = None) -> dict[str, Any]:
        """Interleaved repetitions of the default, the start, the best and the runners-up; medians
        decide. A configuration is valid only if all its runs of this tuning (screening included)
        passed. A randomly failing case (self.flaky, or observed here) extends the series of every
        configuration still valid to --flaky-confirm-reps runs."""
        args = self.args
        fixed: list[SlepcConfig] = []
        for c in (default, start, inc["config"]):
            if c is not None and c.label() not in {u.label() for u in fixed}:
                fixed.append(c)
        pool = sorted((r for r in screened.values() if r["valid"]), key=lambda r: r["time"])
        runners = [r["config"] for r in pool if r["config"].label() not in {u.label() for u in fixed}]
        uniq = fixed + runners[:args.confirm_top]
        timeout = min(args.timeout, max(args.min_trial_timeout,
                                        args.timeout_factor * max(screened[c.label()]["wall"] or 0 for c in uniq)))
        runs: dict[str, list[dict[str, Any]]] = {c.label(): [] for c in uniq}
        self.interleave(uniq, 0, args.confirm_reps, runs, timeout)
        if random_failures(self.trials):
            self.flaky = True
        extended: list[SlepcConfig] = []
        if self.flaky and args.flaky_confirm_reps > args.confirm_reps:
            extended = [c for c in uniq if all(t["valid"] for t in self.cache[c.label()])]
            if extended:
                self.log(f"random failures: {len(extended)} configuration(s) to {args.flaky_confirm_reps} runs")
                self.interleave(extended, args.confirm_reps, args.flaky_confirm_reps, runs, timeout)
        stats = {}
        for c in uniq:
            st = {"config": c.to_json(), **aggregate(runs[c.label()])}
            every = self.cache[c.label()]
            if not all(t["valid"] for t in every):     # e.g. failed its screening run
                st["valid"] = False
                st["invalid_reasons"] = sorted({t["why"] for t in every if not t["valid"] and t["why"]})
            st["runs_total"] = len(every)
            stats[c.label()] = st
        return {"configs": stats, "best": choose_best(stats, args.margin), "default": default.label(),
                "start": start.label() if start is not None else None,
                "extended": [c.label() for c in extended]}

    def interleave(self, cfgs: list[SlepcConfig], first: int, stop: int, runs: dict[str, list[dict[str, Any]]],
                   timeout: float) -> None:
        """Confirmation repetitions first..stop-1 of cfgs, in rotating order."""
        for rep in range(first, stop):
            order = cfgs[rep % len(cfgs):] + cfgs[:rep % len(cfgs)]
            for c in order:
                runs[c.label()].append(self.run(c, f"confirm{rep}", timeout))

    def summary(self, default: SlepcConfig, decisions: list[dict[str, Any]], confirm: dict[str, Any]) -> dict[str, Any]:
        doc = {"case": self.case.name, "problem": self.ctx.problem, "mode": self.mode, "target": self.target,
               "target_override": self.target_override, "cores": self.args.cores, "ncores": self.ctx.cores,
               "tol": self.ctx.tol, "n": self.case.n, "tag": self.args.tag,
               "date": datetime.datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename,
               "screen_reps": self.args.screen_reps, "confirm_reps": self.args.confirm_reps,
               "flaky_confirm_reps": self.args.flaky_confirm_reps, "flaky": self.flaky,
               "random_failures": random_failures(self.trials),
               "margin": self.args.margin, "default": default.label(),
               "decisions": decisions, "confirmation": confirm, "trials": self.trials}
        apply_best(doc, confirm["best"])
        self.save(doc)
        return doc

    def save(self, doc: dict[str, Any] | None = None) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "tune.json"
        if doc is None:
            doc = {"case": self.case.name, "mode": self.mode, "in_progress": True, "trials": self.trials}
        path.write_text(json.dumps(doc, indent=1) + "\n")


def random_failures(trials: list[dict[str, Any]]) -> list[str]:
    """Labels of the configurations that were valid in some trials and invalid in others."""
    seen: dict[str, set[bool]] = {}
    for t in trials:
        seen.setdefault(t["label"], set()).add(bool(t["valid"]))
    return sorted(label for label, v in seen.items() if len(v) == 2)


def choose_best(stats: dict[str, dict[str, Any]], margin: float) -> str | None:
    """The confirmation's winner (label): among the configurations valid in every repetition whose
    median is within `margin` of the fastest median, the one with the fewest options (then the
    fastest) - the rule pick() applies to screening. None when none is valid."""
    valid = {k: s for k, s in stats.items() if s["valid"] and s["median_s"] is not None}
    if not valid:
        return None
    fastest = min(s["median_s"] for s in valid.values())
    near = [k for k, s in valid.items() if s["median_s"] <= fastest * (1.0 + margin)]
    return min(near, key=lambda k: (len(SlepcConfig.from_json(valid[k]["config"]).options), valid[k]["median_s"]))


def apply_best(doc: dict[str, Any], best: str | None) -> dict[str, Any]:
    """Set the best-dependent fields of a tune.json document for the confirmation winner `best`."""
    stats = doc["confirmation"]["configs"]
    d, b = stats[doc["default"]], (stats[best] if best else None)
    doc["confirmation"]["best"] = best
    doc.update(best=best, best_config=b["config"] if b else None, default_median_s=d["median_s"],
               default_valid=d["valid"], best_median_s=b["median_s"] if b else None,
               speedup_vs_default=(d["median_s"] / b["median_s"]) if b and d["median_s"] else None)
    return doc


def pick(results: list[dict[str, Any]], margin: float) -> dict[str, Any] | None:
    """The winner of a group's screening results: among the valid ones within `margin` of the
    fastest, the one with the fewest options (then the fastest) - a difference inside the noise
    does not justify a further non-default setting. None when none is valid."""
    valid = [r for r in results if r["valid"] and r["time"] is not None]
    if not valid:
        return None
    fastest = min(r["time"] for r in valid)
    near = [r for r in valid if r["time"] <= fastest * (1.0 + margin)]
    return min(near, key=lambda r: (len(r["config"].options), r["time"]))


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median/min/max of the solver time and median phases of repeated runs of one config."""
    times = [r["solver_s"] for r in runs if r["solver_s"] is not None]
    keys = ("analysis_s", "numeric_s", "solve_s", "solves", "ortho_s", "setup_s", "eps_solve_s", "vectors_s",
            "factorizations")
    ph = {}
    for key in keys:
        vals = [r["phases"][key] for r in runs if r.get("phases") and r["phases"].get(key) is not None]
        ph[key] = statistics.median(vals) if vals else None
    walls = [r["guard"]["wall_s"] for r in runs if r["guard"].get("wall_s") is not None]
    peaks = [r["report"]["peak_rss_bytes"] for r in runs if r.get("report") and r["report"].get("peak_rss_bytes")]
    etas = [r["eval"]["max_eta"] for r in runs if r.get("eval") and r["eval"].get("max_eta") is not None]
    return {"valid": bool(runs) and all(r["valid"] for r in runs), "n": len(runs),
            "invalid_reasons": sorted({r["why"] for r in runs if not r["valid"] and r["why"]}),
            "median_s": statistics.median(times) if times else None, "min_s": min(times) if times else None,
            "max_s": max(times) if times else None, "times_s": times, "phases": ph,
            "median_wall_s": statistics.median(walls) if walls else None,
            "peak_rss_bytes": max(peaks) if peaks else None, "max_eta": max(etas) if etas else None,
            "trial_ids": [r["id"] for r in runs]}


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"


# ----------------------------------------------------------------------------------- reports

def report_markdown(docs: list[dict[str, Any]]) -> str:
    """Markdown tables of tune.json documents: best configuration per case vs the default."""
    lines = ["| case | type | target | P | default (s) | best (s) | speedup | best configuration | "
             "analysis | numeric (#) | solves (#/s) | ortho | valid |", "|" + "---|" * 13]
    for d in docs:
        conf = d["confirmation"]["configs"]
        best = conf.get(d["best"]) if d.get("best") else None
        df = conf.get(d["default"], {})
        tgt = d["target"]
        tdesc = f"k={tgt['k']}" if d["mode"] == "nearest" else f"[{tgt['lower']:g}, {tgt['upper']:g}]"
        if best is None:
            lines.append(f"| {d['case']} | {d['problem']} | {tdesc} | {d['ncores']} | {_fmt(df.get('median_s'))} | "
                         "- | - | no valid configuration | | | | | NO |")
            continue
        ph = best["phases"]
        sp = d.get("speedup_vs_default")
        lines.append(
            f"| {d['case']} | {d['problem']} | {tdesc} | {d['ncores']} | {_fmt(df.get('median_s'))}"
            f"{'' if df.get('valid') else ' (invalid)'} | {_fmt(best['median_s'])} | "
            f"{'-' if sp is None else f'{sp:.2f}x'} | `{d['best']}` | {_fmt(ph['analysis_s'])} | "
            f"{_fmt(ph['numeric_s'])} ({_num(ph['factorizations'])}) | {_num(ph['solves'])} / {_fmt(ph['solve_s'])} | "
            f"{_fmt(ph['ortho_s'])} | yes |")
    lines += ["", "Confirmation runs (interleaved; median [min-max] solver time, s):", ""]
    for d in docs:
        flaky = d.get("random_failures") or []
        note = f", randomly failing: {len(flaky)} configuration(s) valid in some runs only" if flaky else ""
        lines.append(f"- **{d['case']}** ({d['confirm_reps']} reps{note}): "
                     + "; ".join(
                         f"`{label}` {_fmt(s['median_s'])} [{_fmt(s['min_s'])}-{_fmt(s['max_s'])}] ({s['n']} runs)"
                         f"{'' if s['valid'] else ' INVALID (' + '; '.join(s['invalid_reasons'])[:120] + ')'}"
                         for label, s in d["confirmation"]["configs"].items()))
    return "\n".join(lines) + "\n"


def _num(x: Any) -> str:
    return "-" if x is None else f"{int(round(x))}"


# ------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    from tools import case as cio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--cores", default=os.environ.get("HS_CORES", "24-30,33-41"), help="cpulist; P = its size")
    ap.add_argument("--tag", default="tune")
    ap.add_argument("--root", default=str(cio.DEFAULT_ROOT))
    ap.add_argument("--mem-gb", type=float, default=3.5, help="guard RSS limit of each trial")
    ap.add_argument("--timeout", type=float, default=900, help="limit of the default run (s)")
    ap.add_argument("--timeout-factor", type=float, default=5.0, help="later trials: factor x incumbent wall")
    ap.add_argument("--min-trial-timeout", type=float, default=60)
    ap.add_argument("--target", default=None, help="JSON Target replacing the cases' config target")
    ap.add_argument("--screen-reps", type=int, default=1)
    ap.add_argument("--confirm-reps", type=int, default=5)
    ap.add_argument("--confirm-top", type=int, default=2, help="runners-up in the confirmation")
    ap.add_argument("--flaky-confirm-reps", type=int, default=20,
                    help="confirmation runs per configuration of a randomly failing case (docstring)")
    ap.add_argument("--margin", type=float, default=0.03, help="relative gain a candidate needs to replace the incumbent")
    ap.add_argument("--keep-vectors", action="store_true", help="keep eigenvectors.npy of every trial")
    ap.add_argument("--groups", default=None, help="comma list: run only these search groups (see plan())")
    ap.add_argument("--from-profile", action="store_true",
                    help="start the descent from the case's stored profile (the default is still measured)")
    ap.add_argument("--write-profiles", action="store_true",
                    help="store each case's best configuration in --profiles-file (class profiles re-derived)")
    ap.add_argument("--profiles-file", default=str(PROFILE_FILE),
                    help=f"profile file (default {PROFILE_FILE.relative_to(REPO)})")
    ap.add_argument("--report", nargs="+", type=Path, help="render these tune.json files as markdown and exit")
    ap.add_argument("--rescore", nargs="+", type=Path,
                    help="re-select the confirmation winner of these tune.json files with the current rule "
                         "(choose_best, their own margin), rewrite them (and the profiles with --write-profiles)")
    return ap


def profile_entry(doc: dict[str, Any]) -> dict[str, Any] | None:
    if not doc.get("best"):
        return None
    best = doc["confirmation"]["configs"][doc["best"]]
    flaky = bool(doc.get("flaky")) or bool(random_failures(doc.get("trials", [])))
    return {"scope": "case", "name": doc["case"], "mode": doc["mode"], "cores": doc["ncores"],
            "target": doc["target_override"], "config": doc["best_config"],
            "tuned": {"tag": doc["tag"], "date": doc["date"], "host": doc["host"], "valid": True,
                      "solver_s": best["median_s"], "default_solver_s": doc["default_median_s"],
                      "default_valid": doc["default_valid"], "speedup_vs_default": doc["speedup_vs_default"],
                      "reps": best["n"], "flaky": flaky, "phases": best["phases"]}}


def update_profiles(path: str, docs: list[dict[str, Any]], root: str) -> None:
    from tools import case as cio
    profiles = load_profiles(path)
    for doc in docs:
        entry = profile_entry(doc)
        if entry is not None:
            profiles = upsert(profiles, entry)
    case_profiles = [p for p in profiles if p["scope"] == "case"]
    problems = {}
    for p in case_profiles:
        try:
            problems[p["name"]] = cio.load_case(p["name"], root=root).meta["problem"]
        except (OSError, ValueError, KeyError):
            pass
    save_profiles(case_profiles + class_profiles(case_profiles, problems), path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        docs = [json.loads(p.read_text()) for p in args.report]
        print(report_markdown([d for d in docs if not d.get("in_progress")]), end="")
        return 0
    if args.rescore:
        docs = []
        for p in args.rescore:
            doc = json.loads(p.read_text())
            if doc.get("in_progress"):
                continue
            apply_best(doc, choose_best(doc["confirmation"]["configs"], float(doc.get("margin", args.margin))))
            p.write_text(json.dumps(doc, indent=1) + "\n")
            docs.append(doc)
        if args.write_profiles:
            update_profiles(args.profiles_file, docs, args.root)
        print(report_markdown(docs), end="")
        return 0
    if not args.cases:
        raise SystemExit("tune_slepc: no cases given")
    from tools.eval import common
    common.require_guard("tune_slepc")
    target = json.loads(args.target) if args.target else None
    docs = []
    for c in args.cases:
        doc = Tuner(c, args, target).tune()
        docs.append(doc)
        if args.write_profiles:
            update_profiles(args.profiles_file, [doc], args.root)
        print(f"[tune {c}] best {doc['best']} median {_fmt(doc['best_median_s'])} s; default "
              f"{_fmt(doc['default_median_s'])} s ({'valid' if doc['default_valid'] else 'INVALID'}); "
              f"speedup {doc['speedup_vs_default'] and round(doc['speedup_vs_default'], 2)}", flush=True)
    print(report_markdown(docs), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
