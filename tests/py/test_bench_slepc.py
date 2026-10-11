"""Tests of the SLEPc baseline tooling (tools/bench): option handling of baseline_slepc.py, the
-log_view parser, configurations/profiles, the tuner's search logic and reports, and compare.py's
tables. No PETSc needed: everything runs on fixtures (log excerpts captured from real
baseline_slepc.py runs with PETSc 3.26 / SLEPc 3.26, trimmed) and synthetic documents."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from tools import case as cio
from tools.bench import baseline_slepc, compare, slepc_log, slepc_profile, tune_slepc
from tools.bench.slepc_profile import SlepcConfig

# -log_view of baseline_slepc.py with its stages (yee-pec-box-1e4, 2 ranks), trimmed; the hs_output
# MatSolve and the column-0 line under "Object Type" are added to check what is (not) counted.
STAGED_LOG = """\
------------------------------------------------------------------ PETSc Performance Summary: ------------------------------------------------------------------

tools/bench/baseline_slepc.py on a  named CISL-NF5468M6 with 2 processes, by zhlin on Sun Oct 11 19:36:24 2026
Using 1 OpenMP threads
Using PETSc Release Version 3.26.0, Sep 28, 2026

                         Max       Max/Min     Avg       Total
Time (sec):           2.989e-01     1.000   2.989e-01
Objects:              0.000e+00     0.000   0.000e+00

Summary of Stages:   ----- Time ------  ----- Flop ------  --- Messages ---  -- Message Lengths --  -- Reductions --
                        Avg     %Total     Avg     %Total    Count   %Total     Avg         %Total    Count   %Total
 0:      Main Stage: 7.2203e-02  24.2%  0.0000e+00   0.0%  4.000e+00   3.0%  1.007e+04        1.1%  3.600e+01  14.9%
 1:        hs_setup: 1.2438e-01  41.6%  2.7331e+06   1.3%  4.000e+00   3.0%  1.007e+04        1.1%  1.400e+01   5.8%
 2:        hs_solve: 7.4420e-02  24.9%  1.5747e+08  75.8%  7.800e+01  59.1%  2.067e+04       44.7%  9.500e+01  39.3%
 3:      hs_vectors: 2.5428e-02   8.5%  4.4147e+07  21.3%  2.400e+01  18.2%  2.431e+04       16.2%  2.400e+01   9.9%
 4:       hs_output: 2.4615e-03   0.8%  3.3645e+06   1.6%  2.200e+01  16.7%  6.038e+04       36.9%  5.500e+01  22.7%

------------------------------------------------------------------------------------------------------------------------
See the 'Profiling' chapter of the users' manual for details on interpreting output.
------------------------------------------------------------------------------------------------------------------------
Event                Count      Time (sec)     Flop                              --- Global ---  --- Stage ----  Total
                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %T %F %M %L %R  %T %F %M %L %R Mflop/s
------------------------------------------------------------------------------------------------------------------------

--- Event Stage 0: Main Stage

MatAssemblyEnd         5 1.0 2.9831e-03 1.0 0.00e+00 0.0 8.0e+00 1.0e+04 1.5e+01  0  0  5  2  6   0  0  5  2  6     0

--- Event Stage 1: hs_setup

MatCholFctrSym         1 1.0 8.4650e-02 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00 28  0  0  0  0  68  0  0  0  0     0
MatCholFctrNum         1 1.0 3.5322e-02 1.0 1.57e+06 1.5 0.0e+00 0.0e+00 0.0e+00 12  1  0  0  0  28 95  0  0  0    73
EPSSetUp               1 1.0 1.2434e-01 1.0 1.64e+06 1.5 4.0e+00 1.0e+04 1.4e+01 42  1  3  1  6 100 100 100 100 100    22

--- Event Stage 2: hs_solve

MatMult               62 1.0 7.1471e-02 1.0 1.18e+08 1.5 1.4e+02 2.5e+04 5.8e+01 69 75 95 87 22  69 75 95 87 23   146
MatSolve              31 1.0 6.9520e-02 1.0 6.78e+07 1.5 7.8e+01 2.1e+04 5.0e+00 23 54 59 45  2  93 71 100 100  5  1618
EPSSolve               1 1.0 7.4327e-02 1.0 9.03e+07 1.3 7.8e+01 2.1e+04 9.5e+01 25 76 59 45 39 100 100 100 100 100  2119
STMatSolve            31 1.0 6.9727e-02 1.0 6.78e+07 1.5 7.8e+01 2.1e+04 3.6e+01 23 54 59 45 15  94 71 100 100 38  1613
BVOrthogonalizeV      31 1.0 3.5643e-03 1.0 1.86e+07 1.0 0.0e+00 0.0e+00 5.7e+01  1 18  0  0 24   5 24  0  0 60 10412
BVScale               53 1.0 1.2155e-02 101.5 2.67e+05 1.0 0.0e+00 0.0e+00 0.0e+00  0  0  0  0  0   0  0  0  0  0    44

--- Event Stage 3: hs_vectors

MatSolve              12 1.0 2.4638e-02 1.0 2.62e+07 1.5 2.4e+01 2.4e+04 0.0e+00  8 21 18 16  0  97 99 100 100  0  1767
STMatSolve            12 1.0 2.4709e-02 1.0 2.62e+07 1.5 2.4e+01 2.4e+04 1.2e+01  8 21 18 16  5  97 99 100 100 50  1762

--- Event Stage 4: hs_output

MatSolve               2 1.0 5.0000e-03 1.0 2.62e+07 1.5 2.4e+01 2.4e+04 0.0e+00  8 21 18 16  0  97 99 100 100  0  1767
------------------------------------------------------------------------------------------------------------------------

Object Type          Creations   Destructions. Reports information only for process 0.

--- Event Stage 0: Main Stage

MatSolve              99 1.0 9.9000e+01 1.0 2.62e+07 1.5 2.4e+01 2.4e+04 0.0e+00  8 21 18 16  0  97 99 100 100  0  1767
              Vector    10             33
"""

# The same report of the original single-stage baseline (yee-pec-box-1e4, 2 ranks), trimmed.
UNSTAGED_LOG = """\
tools/bench/baseline_slepc.py on a  named CISL-NF5468M6 with 1 process, by zhlin on Sun Oct 11 19:31:20 2026
Using 2 OpenMP threads
Time (sec):           1.949e+00     1.000   1.949e+00

Summary of Stages:   ----- Time ------  ----- Flop ------  --- Messages ---  -- Message Lengths --  -- Reductions --
                        Avg     %Total     Avg     %Total    Count   %Total     Avg         %Total    Count   %Total
 0:      Main Stage: 1.9486e+00 100.0%  2.6049e+08 100.0%  1.520e+02 100.0%  2.691e+04      100.0%  2.470e+02  92.9%

--- Event Stage 0: Main Stage

MatSolve              53 1.0 1.3340e+00 1.0 1.16e+08 1.5 1.2e+02 2.2e+04 5.0e+00 68 74 80 66  2  68 74 80 66  2   144
MatCholFctrSym         1 1.0 2.2114e-01 1.0 0.00e+00 0.0 0.0e+00 0.0e+00 0.0e+00 11  0  0  0  0  11  0  0  0  0     0
MatCholFctrNum         1 1.0 7.6260e-02 1.0 1.57e+06 1.5 0.0e+00 0.0e+00 0.0e+00  4  1  0  0  0   4  1  0  0  0    34
EPSSetUp               1 1.0 3.1426e-01 1.0 1.64e+06 1.5 4.0e+00 1.0e+04 1.4e+01 16  1  3  1  5  16  1  3  1  6     9
EPSSolve               1 1.0 1.1061e+00 1.0 1.14e+08 1.3 9.2e+01 2.1e+04 1.2e+02 57 76 61 48 44  57 76 61 48 47   180
BVOrthogonalizeV      38 1.0 5.4572e-02 1.0 2.39e+07 1.0 0.0e+00 0.0e+00 7.2e+01  3 18  0  0 27   3 18  0  0 29   876
"""


# ------------------------------------------------------------------------- baseline_slepc options

def test_baseline_split_argv_own_and_petsc_options():
    own, petsc = baseline_slepc.split_argv(
        ["--case", "c1", "-st_mat_mumps_icntl_7", "5", "--out=/tmp/o", "-eps_target", "-0.5", "-eps_view",
         "--target", '{"mode":"interval","lower":1,"upper":2}', "-log_view", ":log.txt"])
    assert own == ["--case", "c1", "--out=/tmp/o", "--target", '{"mode":"interval","lower":1,"upper":2}']
    assert petsc == ["-st_mat_mumps_icntl_7", "5", "-eps_target", "-0.5", "-eps_view", "-log_view", ":log.txt"]
    args = baseline_slepc.build_parser().parse_args(own)
    assert (args.case, args.out, args.bind) == ("c1", "/tmp/o", "none")
    with pytest.raises(SystemExit):
        baseline_slepc.split_argv(["--case"])


def test_baseline_petsc_option_names_skip_values():
    names = baseline_slepc.petsc_option_names(["-eps_target", "-0.5", "-log_view", ":x.txt", "-eps_purify", "0",
                                               "-st_mat_mumps_cntl_3", "1e-12", "-eps_view"])
    assert names == ["eps_target", "log_view", "eps_purify", "st_mat_mumps_cntl_3", "eps_view"]


def test_baseline_resolve_target_and_order():
    cfg_t = {"mode": "nearest", "sigma": 2.0, "k": 3}
    assert baseline_slepc.resolve_target(cfg_t, None) == {"mode": "nearest", "sigma": 2.0, "k": 3}
    t = baseline_slepc.resolve_target(cfg_t, '{"mode": "interval", "lower": 1, "upper": 5}')
    assert t == {"mode": "interval", "lower": 1.0, "upper": 5.0}
    for bad in ('{"mode": "interval", "lower": 5, "upper": 1}', '{"mode": "smallest", "k": 2}',
                '{"mode": "nearest", "sigma": 0, "k": 0}'):
        with pytest.raises(SystemExit):
            baseline_slepc.resolve_target(cfg_t, bad)
    lam = np.array([4.0, 1.5, 2.2, -1.0])
    assert baseline_slepc.result_order(lam, cfg_t).tolist() == [2, 1, 0, 3]   # |lam - 2|: 2 .5 .2 3
    assert baseline_slepc.result_order(lam, t).tolist() == [3, 1, 2, 0]       # ascending


def test_baseline_option_defaults_put_mumps_in_the_database():
    near = dict(baseline_slepc.option_defaults("nearest"))
    assert near == {"st_ksp_type": "preonly", "st_pc_type": "cholesky", "st_pc_factor_mat_solver_type": "mumps"}
    inter = dict(baseline_slepc.option_defaults("interval"))
    assert inter["st_mat_mumps_icntl_13"] == "1" and inter["st_mat_mumps_icntl_24"] == "1"
    assert inter["st_pc_factor_mat_solver_type"] == "mumps"


def test_baseline_rank_binding(monkeypatch):
    assert baseline_slepc.rank_cpus([41, 40, 43, 42], 1, 2) == [42, 43]
    assert baseline_slepc.rank_cpus([40, 41], 0, 1) == [40, 41]
    with pytest.raises(ValueError):
        baseline_slepc.rank_cpus([40, 41, 42], 0, 2)
    calls = []
    monkeypatch.setattr(baseline_slepc.os, "sched_getaffinity", lambda pid: {40, 41})
    monkeypatch.setattr(baseline_slepc.os, "sched_setaffinity", lambda pid, cpus: calls.append(list(cpus)))
    assert baseline_slepc.bind_rank(["--case", "x"], env={}) is None
    assert baseline_slepc.bind_rank(["--bind", "none"], env={}) is None
    assert calls == []
    env = {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2"}
    assert baseline_slepc.bind_rank(["--case", "x", "--bind", "core"], env=env) == [41]
    assert baseline_slepc.bind_rank(["--bind=core"], env={"PMIX_RANK": "0"}) == [40, 41]
    assert calls == [[41], [40, 41]]
    with pytest.raises(SystemExit):
        baseline_slepc.bind_rank(["--bind", "socket"], env={})


# ----------------------------------------------------------------------------------- slepc_log

def test_log_view_parse_staged():
    log = slepc_log.parse_log_view(STAGED_LOG)
    assert log["ranks"] == 2 and log["omp_threads"] == 1 and log["time_s"] == pytest.approx(0.2989)
    assert list(log["stages"]) == ["Main Stage", "hs_setup", "hs_solve", "hs_vectors", "hs_output"]
    assert log["stages"]["hs_solve"]["time_s"] == pytest.approx(0.07442)
    assert log["events"]["hs_solve"]["MatSolve"] == {"count": 31, "count_ratio": 1.0, "time_s": pytest.approx(0.06952),
                                                     "time_ratio": 1.0, "flop": pytest.approx(6.78e7)}
    assert log["events"]["hs_solve"]["BVScale"]["time_ratio"] == pytest.approx(101.5)
    # the object table's repeated stage header does not add events
    assert log["events"]["Main Stage"].keys() == {"MatAssemblyEnd"}
    ph = slepc_log.phases(log)
    assert ph["staged"]
    assert ph["analysis_s"] == pytest.approx(0.08465) and ph["analyses"] == 1
    assert ph["numeric_s"] == pytest.approx(0.035322) and ph["factorizations"] == 1
    assert ph["solves"] == 43 and ph["solve_s"] == pytest.approx(0.06952 + 0.024638)   # hs_output excluded
    assert ph["ortho_s"] == pytest.approx(0.0035643)
    assert ph["setup_s"] == pytest.approx(0.12438) and ph["eps_solve_s"] == pytest.approx(0.07442)
    assert ph["vectors_s"] == pytest.approx(0.025428)
    assert ph["solver_s"] == pytest.approx(0.12438 + 0.07442 + 0.025428)


def test_log_view_parse_unstaged_falls_back_to_events():
    ph = slepc_log.phases(slepc_log.parse_log_view(UNSTAGED_LOG))
    assert not ph["staged"] and ph["ranks"] == 1 and ph["omp_threads"] == 2
    assert ph["solves"] == 53 and ph["solve_s"] == pytest.approx(1.334)
    assert ph["setup_s"] == pytest.approx(0.31426) and ph["eps_solve_s"] == pytest.approx(1.1061)
    assert ph["vectors_s"] == 0.0 and ph["solver_s"] == pytest.approx(0.31426 + 1.1061)
    assert ph["analysis_s"] == pytest.approx(0.22114) and ph["ortho_s"] == pytest.approx(0.054572)


def test_log_view_errors(tmp_path):
    with pytest.raises(ValueError):
        slepc_log.parse_log_view("no table here\n")
    assert slepc_log.read_phases(tmp_path / "missing.txt") is None
    (tmp_path / "log.txt").write_text(STAGED_LOG)
    assert slepc_log.read_phases(tmp_path / "log.txt")["solves"] == 43


# ------------------------------------------------------------------------------- slepc_profile

def test_config_options_labels_and_k_rules():
    c = SlepcConfig.make(2, 1, "core", {"st_mat_mumps_icntl_7": 5, "eps_ncv": "3k", "eps_nev": "k+4"})
    assert c.options == (("eps_ncv", "3k"), ("eps_nev", "k+4"), ("st_mat_mumps_icntl_7", "5"))
    assert c.label() == "2x1/core eps_ncv=3k eps_nev=k+4 st_mat_mumps_icntl_7=5"
    assert c.petsc_args(k=10) == ["-eps_ncv", "30", "-eps_nev", "14", "-st_mat_mumps_icntl_7", "5"]
    assert c.with_options(eps_ncv=None, eps_purify=0).label() == "2x1/core eps_nev=k+4 eps_purify=0 st_mat_mumps_icntl_7=5"
    assert c.with_layout(1, 2, "none").label().startswith("1x2 eps_ncv")
    assert SlepcConfig.from_json(c.to_json()) == c and c.cores == 2
    assert [slepc_profile.expand_value(v, 12) for v in ("k", "2k", "k+15", "3k-2", "40", "0.7")] == \
        ["12", "24", "27", "34", "40", "0.7"]
    with pytest.raises(ValueError):
        slepc_profile.expand_value("2k", None)
    for bad in (dict(ranks=0), dict(bind="socket"), dict(options=(("-eps_ncv", "3"),)),
                dict(options=(("b", "1"), ("a", "2")))):
        with pytest.raises(ValueError):
            SlepcConfig(**bad)


def test_mpiexec_command(tmp_path):
    cfg = SlepcConfig.make(2, 1, "core", {"eps_ncv": "2k"})
    cmd = slepc_profile.mpiexec_command(cfg, "c1", tmp_path / "out", k=6, target={"mode": "interval", "lower": 1,
                                                                                    "upper": 2},
                                        log_view=tmp_path / "log.txt", ref_env=tmp_path / "ref")
    assert cmd[:5] == [str(tmp_path / "ref/bin/mpiexec"), "-n", "2", "--bind-to", "none"]
    assert "OMP_NUM_THREADS=1" in cmd and "OMP_PROC_BIND=close" not in cmd
    i = cmd.index("--bind")
    assert cmd[i + 1] == "core"
    assert cmd[cmd.index("--target") + 1] == '{"mode":"interval","lower":1,"upper":2}'
    assert cmd[cmd.index("-eps_ncv") + 1] == "12"
    assert cmd[-2:] == ["-log_view", f":{tmp_path / 'log.txt'}"]
    hybrid = slepc_profile.mpiexec_command(SlepcConfig(1, 2, "core"), "c1", "o", ref_env=tmp_path)
    assert "OMP_PROC_BIND=close" in hybrid and "OMP_PLACES=cores" in hybrid
    plain = slepc_profile.mpiexec_command(slepc_profile.default_config(2), "c1", "o", ref_env=tmp_path)
    assert "--bind" not in plain and "OMP_NUM_THREADS=2" in plain


def _profile(scope, name, mode="nearest", cores=2, ranks=1, solver_s=1.0, valid=True, **opts):
    return {"scope": scope, "name": name, "mode": mode, "cores": cores,
            "config": SlepcConfig.make(ranks, cores // ranks, "none", opts).to_json(),
            "tuned": {"valid": valid, "solver_s": solver_s}}


def test_profile_lookup_upsert_and_roundtrip(tmp_path):
    profiles = [_profile("case", "a", ranks=2, eps_ncv="3k"), _profile("class", "GHEP/nearest", eps_purify="0"),
                _profile("case", "a", mode="interval")]
    cfg, entry = slepc_profile.lookup(profiles, "a", "GHEP", "nearest", 2)
    assert entry["scope"] == "case" and cfg.ranks == 2 and cfg.option("eps_ncv") == "3k"
    cfg, entry = slepc_profile.lookup(profiles, "b", "GHEP", "nearest", 2)
    assert entry["scope"] == "class" and cfg.option("eps_purify") == "0"
    assert slepc_profile.lookup(profiles, "b", "HEP", "nearest", 2) is None
    assert slepc_profile.lookup(profiles, "b", "GHEP", "nearest", 16) is None   # tuned for 2 cores only
    assert slepc_profile.lookup(profiles, "a", "GHEP", "interval", 2)[1]["mode"] == "interval"
    new = _profile("case", "a", ranks=1, eps_ncv="4k")
    profiles = slepc_profile.upsert(profiles, new)
    assert sum(p["scope"] == "case" and p["name"] == "a" and p["mode"] == "nearest" for p in profiles) == 1
    path = tmp_path / "profiles.json"
    slepc_profile.save_profiles(profiles, path)
    loaded = slepc_profile.load_profiles(path)
    assert sorted(json.dumps(p, sort_keys=True) for p in loaded) == sorted(json.dumps(p, sort_keys=True) for p in profiles)
    assert loaded[0]["scope"] == "case"
    assert slepc_profile.load_profiles(tmp_path / "none.json") == []
    path.write_text(json.dumps({"schema_version": 99, "profiles": []}))
    with pytest.raises(ValueError):
        slepc_profile.load_profiles(path)


def test_profile_lookup_matches_the_tuned_interval():
    tuned = {"mode": "interval", "lower": 40, "upper": 200}
    profiles = [{**_profile("case", "a", mode="interval", ranks=2, eps_krylovschur_nev="60"), "target": tuned},
                _profile("case", "b", mode="interval", ranks=2, eps_krylovschur_nev="20"),   # tuned on config target
                _profile("class", "GHEP/interval", mode="interval")]
    hit = slepc_profile.lookup(profiles, "a", "GHEP", "interval", 2, target={"mode": "interval", "lower": 40.0,
                                                                              "upper": 200.0})
    assert hit[1]["scope"] == "case" and hit[0].option("eps_krylovschur_nev") == "60"
    other = {"mode": "interval", "lower": 1, "upper": 200}
    assert slepc_profile.lookup(profiles, "a", "GHEP", "interval", 2, target=other)[1]["scope"] == "class"
    # a profile without a stored target was tuned on the case config's target
    assert slepc_profile.lookup(profiles, "b", "GHEP", "interval", 2, target=other,
                                config_target=other)[1]["scope"] == "case"
    assert slepc_profile.lookup(profiles, "b", "GHEP", "interval", 2, target=tuned,
                                config_target=other)[1]["scope"] == "class"


def test_config_fitted_drops_options_the_layout_cannot_use():
    c = SlepcConfig.make(1, 2, "none", {"eps_krylovschur_partitions": 2, "st_mat_mumps_icntl_28": 2,
                                        "st_mat_mumps_icntl_29": 1, "eps_ncv": "2k"})
    assert c.fitted().label() == "1x2 eps_ncv=2k"
    four = c.with_layout(4, 1)
    assert four.fitted() == four                                             # 2 partitions of 4 ranks
    assert four.with_options(eps_krylovschur_partitions=3).fitted().option("eps_krylovschur_partitions") is None


def test_class_profiles_vote():
    profiles = [_profile("case", "a", ranks=2, solver_s=5.0, st_mat_mumps_icntl_7="3", eps_purify="0"),
                _profile("case", "b", ranks=2, solver_s=1.0, st_mat_mumps_icntl_7="5"),
                _profile("case", "c", ranks=1, solver_s=2.0, st_mat_mumps_icntl_7="5", eps_purify="0"),
                _profile("case", "d", ranks=1, solver_s=9.0, valid=False),        # invalid: ignored
                _profile("case", "h", ranks=1, solver_s=1.0)]                      # other class
    out = slepc_profile.class_profiles(profiles, {"a": "GHEP", "b": "GHEP", "c": "GHEP", "d": "GHEP", "h": "HEP"})
    by = {p["name"]: p for p in out}
    assert set(by) == {"GHEP/nearest", "HEP/nearest"}
    g = SlepcConfig.from_json(by["GHEP/nearest"]["config"])
    assert (g.ranks, g.threads) == (2, 1)                        # 2 of 3 cases
    # options are voted among the cases with that layout (a, b): ties go to the slower case a
    assert g.option("st_mat_mumps_icntl_7") == "3" and g.option("eps_purify") == "0"
    assert by["GHEP/nearest"]["from_cases"] == ["a", "c", "b"]   # by tuned solver time
    assert by["GHEP/nearest"]["validated"] is False


def test_class_profiles_keep_accuracy_and_fit_the_layout():
    # accuracy padding one case needed survives the vote (yee-pec-box-1e5 needed eps_nev=k+4)
    profiles = [_profile("case", "yee", ranks=1, solver_s=7.0, eps_nev="k+4", eps_purify="0"),
                _profile("case", "ss", ranks=1, solver_s=3.0, eps_ncv="3k", eps_purify="0", eps_nev="k+2"),
                _profile("case", "cav", ranks=1, solver_s=0.6, eps_purify="0", eps_tol="1e-11")]
    g = SlepcConfig.from_json(slepc_profile.class_profiles(profiles, dict.fromkeys(("yee", "ss", "cav"), "GHEP"))[0]
                              ["config"])
    assert g.label() == "1x2 eps_nev=k+4 eps_purify=0 eps_tol=1e-11"
    # interval tunings: bound and unbound 2-rank layouts split the layout vote; the 1-rank layout wins
    # the tie (slowest case) and must not inherit the 2-rank cases' partitions (SLEPc rejects them)
    inter = [_profile("case", "a", mode="interval", ranks=2, solver_s=1.0, eps_krylovschur_partitions="2"),
             {**_profile("case", "b", mode="interval", ranks=2, solver_s=2.0, eps_krylovschur_partitions="2"),
              "config": SlepcConfig.make(2, 1, "core", {"eps_krylovschur_partitions": "2"}).to_json()},
             _profile("case", "c", mode="interval", ranks=1, solver_s=3.0)]
    g = SlepcConfig.from_json(slepc_profile.class_profiles(inter, dict.fromkeys("abc", "GHEP"))[0]["config"])
    assert g.label() == "1x2"


# --------------------------------------------------------------------------------- tune_slepc

def _ctx(**kw):
    base = dict(cores=2, problem="GHEP", mode="nearest", k=10, tol=1e-10)
    base.update(kw)
    return tune_slepc.Context(**base)


def test_tune_parser_keeps_cases_after_flags():
    args = tune_slepc.build_parser().parse_args(["--write-profiles", "caseA", "caseB", "--cores", "40,41"])
    assert args.cases == ["caseA", "caseB"] and args.write_profiles


def test_tune_layouts_and_groups():
    assert tune_slepc.layouts(2) == [(1, 2, "none"), (1, 2, "core"), (2, 1, "none"), (2, 1, "core")]
    assert [r for r, _, b in tune_slepc.layouts(16) if b == "none"] == [1, 2, 4, 8, 16]
    assert tune_slepc.layouts(1) == [(1, 1, "none")]
    names = [n for n, _ in tune_slepc.plan("nearest")]
    assert names[:3] == ["accuracy", "layout", "ordering"] and "purify" in names and "slice_nev" not in names
    assert "partitions" in [n for n, _ in tune_slepc.plan("interval")]
    inc = SlepcConfig(1, 2)
    ords = tune_slepc.g_ordering(inc, _ctx())
    assert {c.option("st_mat_mumps_icntl_7") for c in ords} == {None, "0", "2", "3", "4", "5", "6"}
    par = [c for c in tune_slepc.g_ordering(SlepcConfig(2, 1), _ctx()) if c.option("st_mat_mumps_icntl_28")]
    assert sorted(c.option("st_mat_mumps_icntl_29") for c in par) == ["1", "2"]
    lay = tune_slepc.g_layout(SlepcConfig.make(2, 1, "none", {"st_mat_mumps_icntl_28": 2, "st_mat_mumps_icntl_29": 1}),
                              _ctx())
    assert all(c.option("st_mat_mumps_icntl_28") is None for c in lay if c.ranks == 1)
    assert all(c.option("st_mat_mumps_icntl_28") == "2" for c in lay if c.ranks == 2)
    # k = 18: SLEPc's default ncv max(2k, k+15) = 36 = 2k, so "2k" is not a new candidate
    ncv = [c.option("eps_ncv") for c in tune_slepc.g_ncv(inc, _ctx(k=18))]
    assert ncv == ["k+8", "3k", "4k"]
    assert tune_slepc.g_purify(inc, _ctx(problem="HEP")) == []
    assert tune_slepc.g_problem_type(inc, _ctx(problem="GHIEP"))[0].option("eps_gen_non_hermitian") == "1"
    acc = tune_slepc.g_accuracy(inc, _ctx())
    assert acc[0].option("eps_tol") == "1e-11" and any(c.option("eps_nev") == "k+4" for c in acc)
    assert [c.option("eps_tol") for c in tune_slepc.g_accuracy(inc, _ctx(mode="interval", k=None))] == ["1e-11"]
    parts = tune_slepc.g_partitions(SlepcConfig(2, 1), _ctx(mode="interval", k=None))
    assert [c.option("eps_krylovschur_partitions") for c in parts] == ["2"]
    back = tune_slepc.g_partitions(SlepcConfig.make(2, 1, "none", {"eps_krylovschur_partitions": 2}),
                                   _ctx(mode="interval", k=None))
    assert [c.option("eps_krylovschur_partitions") for c in back] == [None]
    icntl48 = dict(tune_slepc.plan("nearest"))["icntl_48"]
    assert icntl48(SlepcConfig(2, 1), _ctx()) == [] and len(icntl48(SlepcConfig(1, 2), _ctx())) == 1
    blr = tune_slepc.g_blr(inc, _ctx())
    assert [(c.option("st_mat_mumps_icntl_35"), c.option("st_mat_mumps_cntl_7")) for c in blr] == \
        [("2", "1e-12"), ("2", "1e-10")]


def test_tune_prune_drops_one_setting_at_a_time():
    inc = SlepcConfig.make(2, 1, "none", {"eps_nev": "k+4", "eps_purify": "0", "st_mat_mumps_icntl_28": "2",
                                          "st_mat_mumps_icntl_29": "1", "st_mat_mumps_icntl_35": "2",
                                          "st_mat_mumps_cntl_7": "1e-12"})
    labels = [c.label() for c in tune_slepc.g_prune(inc, _ctx())]
    assert labels == [
        "2x1 eps_purify=0 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1 st_mat_mumps_icntl_35=2",
        "2x1 eps_nev=k+4 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1 st_mat_mumps_icntl_35=2",
        "2x1 eps_nev=k+4 eps_purify=0 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_35=2",   # pair 28/29
        "2x1 eps_nev=k+4 eps_purify=0 st_mat_mumps_icntl_28=2 st_mat_mumps_icntl_29=1"]
    assert "prune" in tune_slepc.SIMPLIFYING and [n for n, _ in tune_slepc.plan("nearest")][-1] == "prune"
    # simplifying selection: the incumbent competes, a valid removal within the margin wins
    inc_r = {"config": inc, "valid": True, "time": 1.00}
    drop = {"config": inc.with_options(eps_nev=None), "valid": True, "time": 1.02}
    assert tune_slepc.pick([drop, inc_r], 0.03) is drop
    assert tune_slepc.pick([{**drop, "time": 1.10}, inc_r], 0.03) is inc_r


def test_tune_plan_group_selection():
    assert [n for n, _ in tune_slepc.plan("nearest", ["ncv", "blr"])] == ["blr", "ncv"]   # plan order kept
    assert [n for n, _ in tune_slepc.plan("interval", ["slice_nev"])] == ["slice_nev"]
    with pytest.raises(SystemExit):
        tune_slepc.plan("interval", ["blr"])
    args = tune_slepc.build_parser().parse_args(["c1", "--groups", "blr", "--from-profile"])
    assert args.groups == "blr" and args.from_profile and args.cases == ["c1"]


def test_tune_pick_prefers_simpler_within_margin():
    simple = SlepcConfig.make(1, 2, "none", {"eps_nev": "k+4"})
    fancy = simple.with_options(eps_tol="1e-11")
    res = [{"config": fancy, "valid": True, "time": 1.00}, {"config": simple, "valid": True, "time": 1.02},
           {"config": SlepcConfig(2, 1), "valid": False, "time": None}]
    assert tune_slepc.pick(res, 0.03)["config"] == simple
    assert tune_slepc.pick(res, 0.01)["config"] == fancy
    assert tune_slepc.pick(res[2:], 0.03) is None


def _trial(tid, t, valid=True, solves=10):
    return {"id": tid, "solver_s": t, "valid": valid, "why": None if valid else "1 of 4 pairs have backward error > tol",
            "phases": {"analysis_s": 0.5, "numeric_s": 0.2, "solve_s": t - 0.8, "solves": solves, "ortho_s": 0.01,
                       "setup_s": 0.7, "eps_solve_s": t - 0.7, "vectors_s": 0.0, "factorizations": 1},
            "guard": {"wall_s": t + 2.0}, "report": {"peak_rss_bytes": 2**30}, "eval": {"max_eta": 1e-13}}


def test_tune_aggregate_and_report():
    agg = tune_slepc.aggregate([_trial(0, 2.0), _trial(1, 3.0), _trial(2, 2.5)])
    assert agg["valid"] and agg["median_s"] == 2.5 and (agg["min_s"], agg["max_s"]) == (2.0, 3.0)
    assert agg["phases"]["solves"] == 10 and agg["median_wall_s"] == 4.5 and agg["trial_ids"] == [0, 1, 2]
    bad = tune_slepc.aggregate([_trial(0, 2.0), _trial(1, 1.0, valid=False)])
    assert not bad["valid"] and bad["invalid_reasons"] == ["1 of 4 pairs have backward error > tol"]
    best = SlepcConfig.make(2, 1, "none", {"eps_nev": "k+4"})
    doc = {"case": "c1", "problem": "GHEP", "mode": "nearest", "target": {"mode": "nearest", "sigma": 1.0, "k": 4},
           "ncores": 2, "confirm_reps": 3, "default": "1x2", "best": best.label(), "speedup_vs_default": 1.6,
           "confirmation": {"configs": {"1x2": {**bad, "config": SlepcConfig(1, 2).to_json()},
                                        best.label(): {**agg, "config": best.to_json()}}}}
    md = tune_slepc.report_markdown([doc])
    row = next(line for line in md.splitlines() if line.startswith("| c1 "))
    assert "(invalid)" in row and "1.60x" in row and f"`{best.label()}`" in row and "| yes |" in row
    assert "INVALID" in md and "2.500 [2.000-3.000]" in md
    entry = tune_slepc.profile_entry({**doc, "date": "d", "host": "h", "tag": "t", "target_override": None,
                                      "best_config": best.to_json(), "default_median_s": 2.0, "default_valid": False})
    assert entry["scope"] == "case" and entry["cores"] == 2 and entry["tuned"]["solver_s"] == 2.5
    assert tune_slepc.profile_entry({**doc, "best": None}) is None


def test_tune_choose_best_and_rescore(tmp_path):
    plain, blr = SlepcConfig(2, 1), SlepcConfig.make(2, 1, "none", {"st_mat_mumps_icntl_35": 2,
                                                                    "st_mat_mumps_cntl_7": "1e-10"})
    stats = {"1x2": {**tune_slepc.aggregate([_trial(0, 1.10)]), "config": SlepcConfig(1, 2).to_json()},
             plain.label(): {**tune_slepc.aggregate([_trial(1, 0.570)]), "config": plain.to_json()},
             blr.label(): {**tune_slepc.aggregate([_trial(2, 0.559)]), "config": blr.to_json()},
             "bad": {**tune_slepc.aggregate([_trial(3, 0.3, valid=False)]), "config": SlepcConfig(2, 1, "core").to_json()}}
    assert tune_slepc.choose_best(stats, 0.03) == plain.label()     # 2 % slower, no extra options
    assert tune_slepc.choose_best(stats, 0.01) == blr.label()
    assert tune_slepc.choose_best({"bad": stats["bad"]}, 0.03) is None
    doc = {"case": "c1", "problem": "GHEP", "mode": "nearest", "target": {"mode": "nearest", "sigma": 0, "k": 2},
           "ncores": 2, "confirm_reps": 1, "margin": 0.03, "default": "1x2",
           "confirmation": {"configs": stats, "best": blr.label()}}
    tune_slepc.apply_best(doc, blr.label())
    path = tmp_path / "tune.json"
    path.write_text(json.dumps(doc))
    assert tune_slepc.main(["--rescore", str(path)]) == 0
    doc = json.loads(path.read_text())
    assert doc["best"] == plain.label() and doc["confirmation"]["best"] == plain.label()
    assert doc["best_config"] == plain.to_json() and doc["speedup_vs_default"] == pytest.approx(1.10 / 0.570)


def test_tune_update_profiles_derives_classes(tmp_path):
    cio.save_case("tiny", sp.identity(3, format="csr"), sp.identity(3, format="csr") * 2.0,
                  meta={"family": "test", "problem": "GHEP"}, config=cio.default_config(0.0, 1), root=tmp_path)
    doc = {"case": "tiny", "mode": "nearest", "ncores": 2, "target_override": None, "tag": "t", "date": "d",
           "host": "h", "best": "2x1", "best_config": SlepcConfig(2, 1).to_json(), "default_median_s": 3.0,
           "default_valid": True, "speedup_vs_default": 1.5,
           "confirmation": {"configs": {"2x1": tune_slepc.aggregate([_trial(0, 2.0)])}}}
    path = tmp_path / "profiles.json"
    tune_slepc.update_profiles(str(path), [doc], str(tmp_path))
    profiles = slepc_profile.load_profiles(path)
    assert [(p["scope"], p["name"]) for p in profiles] == [("case", "tiny"), ("class", "GHEP/nearest")]


def test_tune_trial_time_prefers_log_view():
    assert tune_slepc.trial_time({"phases": {"solver_s": 1.5}, "report": {"time_setup_s": 1, "time_solve_s": 2}}) == 1.5
    assert tune_slepc.trial_time({"phases": None, "report": {"time_setup_s": 1, "time_solve_s": 2}}) == 3
    assert tune_slepc.trial_time({"phases": None, "report": None}) is None


def test_tune_random_failures_and_flaky_profiles():
    trials = [{**_trial(0, 1.0), "label": "a"}, {**_trial(1, 1.0, valid=False), "label": "a"},
              {**_trial(2, 1.0, valid=False), "label": "b"}, {**_trial(3, 1.0), "label": "c"}]
    assert tune_slepc.random_failures(trials) == ["a"]                # b fails every time: not random
    best = SlepcConfig(2, 1)
    doc = {"case": "c1", "mode": "nearest", "ncores": 2, "target_override": None, "tag": "t", "date": "d", "host": "h",
           "best": best.label(), "best_config": best.to_json(), "default_median_s": 2.0, "default_valid": True,
           "speedup_vs_default": 2.0, "trials": trials,
           "confirmation": {"configs": {best.label(): tune_slepc.aggregate([_trial(4, 1.0)])}}}
    assert tune_slepc.profile_entry(doc)["tuned"]["flaky"] is True    # derived from the trials of old documents
    assert tune_slepc.profile_entry({**doc, "trials": trials[2:]})["tuned"]["flaky"] is False


class _SyntheticTuner(tune_slepc.Tuner):
    """Tuner whose trials are synthetic: model(cfg, n) -> (solver time, valid) for the n-th run of cfg."""

    def __init__(self, model, case_name, args):
        super().__init__(case_name, args, None)
        self.model = model
        self.dir = Path(args.root) / "tune"      # keep tune.json out of the repository

    def measure(self, cfg, phase, timeout, tid):
        t, ok = self.model(cfg, len(self.cache.get(cfg.label(), [])))
        return {"id": tid, "phase": phase, "label": cfg.label(), "config": cfg.to_json(),
                "guard": {"returncode": 0, "killed": None, "wall_s": t + 2.0, "peak_rss_gib": 0.1},
                "report": {"unused_options": [], "success": ok, "peak_rss_bytes": 2**20},
                "phases": {"solver_s": t, "solves": 10},
                "eval": {"verdict": "pass" if ok else "fail", "reasons": [] if ok else ["2 missing"], "max_eta": 1e-13}}


def _tuner_args(tmp_path, *extra):
    cio.save_case("tiny", sp.diags(np.arange(1.0, 9.0)).tocsr(), sp.identity(8, format="csr"),
                  meta={"family": "test", "problem": "GHEP"}, config=cio.default_config(0.0, 2), root=tmp_path)
    return tune_slepc.build_parser().parse_args(["tiny", "--root", str(tmp_path), "--cores", "40,41",
                                                 "--profiles-file", str(tmp_path / "profiles.json"), *extra])


def _speed_model(cfg, n):
    """2x1 beats 1x2, AMF beats the other orderings, no purification saves time; BLR 1e-12 is a bit
    faster but fails every 4th run (like wg-square-1e5), BLR 1e-10 always fails; the rest is slower."""
    t, known = 2.0, {"st_mat_mumps_icntl_7", "st_mat_mumps_icntl_35", "st_mat_mumps_cntl_7", "eps_purify"}
    t -= 0.4 if cfg.ranks == 2 else 0.0
    t -= 0.5 if cfg.option("st_mat_mumps_icntl_7") == "2" else 0.0
    t += 0.3 if cfg.option("st_mat_mumps_icntl_7") not in (None, "2") else 0.0
    t -= 0.2 if cfg.option("eps_purify") == "0" else 0.0
    t += 0.3 * len({n for n, _ in cfg.options} - known)
    if cfg.option("st_mat_mumps_icntl_35"):
        return t - 0.1, cfg.option("st_mat_mumps_cntl_7") == "1e-12" and n % 4 != 3
    return t, True


def test_tune_descent_and_flaky_confirmation(tmp_path):
    tuner = _SyntheticTuner(_speed_model, "tiny", _tuner_args(tmp_path, "--confirm-reps", "5"))
    assert not tuner.flaky
    doc = tuner.tune()
    groups = [d["group"] for d in doc["decisions"]]
    # accuracy skipped (the default is valid), icntl_48 has no candidate on 1-thread ranks, GHEP: no problem_type
    assert groups == ["default", "layout", "ordering", "layout2", "icntl_58", "icntl_13", "icntl_8", "blr", "ncv",
                      "restart", "purify", "prune"]
    by = {d["group"]: d for d in doc["decisions"]}
    assert by["layout"]["incumbent"] == "2x1" and by["ordering"]["incumbent"] == "2x1 st_mat_mumps_icntl_7=2"
    assert not by["icntl_58"]["moved"] and not by["layout2"]["moved"]
    blr = "2x1 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_35=2 st_mat_mumps_icntl_7=2"
    assert by["blr"]["moved"] and by["blr"]["incumbent"] == blr     # its single screening run passed
    inc = "2x1 eps_purify=0 st_mat_mumps_cntl_7=1e-12 st_mat_mumps_icntl_35=2 st_mat_mumps_icntl_7=2"
    assert by["purify"]["incumbent"] == inc and by["prune"]["incumbent"] == inc   # dropping BLR is 12 % slower
    conf = doc["confirmation"]
    assert list(conf["configs"])[:2] == ["1x2", inc]
    # BLR failed its 4th run: the case fails at random, so every configuration still valid got 20 runs
    assert doc["flaky"] and inc in doc["random_failures"] and not conf["configs"][inc]["valid"]
    assert conf["extended"] and all(conf["configs"][c]["n"] == 20 and conf["configs"][c]["valid"]
                                    for c in conf["extended"])
    assert doc["best"] == "2x1 eps_purify=0 st_mat_mumps_icntl_7=2" and doc["default_valid"]
    assert tune_slepc.profile_entry(doc)["tuned"]["flaky"]
    assert json.loads((tmp_path / "tune" / "tune.json").read_text())["best"] == doc["best"]


def test_tune_accuracy_group_only_while_invalid(tmp_path):
    def model(cfg, n):   # valid only with eps_nev padding; the larger padding costs more
        return 2.0 + (0.1 if cfg.option("eps_nev") == "k+4" else 0.0) + (0.1 if cfg.option("eps_tol") else 0.0), \
            cfg.option("eps_nev") is not None
    tuner = _SyntheticTuner(model, "tiny", _tuner_args(tmp_path, "--groups", "accuracy,purify",
                                                       "--confirm-reps", "2"))
    doc = tuner.tune()
    assert [d["group"] for d in doc["decisions"]] == ["default", "accuracy", "purify"]
    assert doc["decisions"][1]["incumbent"] == "1x2 eps_nev=k+2"
    assert not doc["default_valid"] and doc["best"] == "1x2 eps_nev=k+2"
    assert not doc["flaky"] and not doc["confirmation"]["extended"]   # failing every time is not random
    assert all(s["n"] == 2 for s in doc["confirmation"]["configs"].values())


def test_tune_confirmation_includes_the_stored_profile_and_its_flaky_flag(tmp_path):
    args = _tuner_args(tmp_path, "--groups", "ncv", "--from-profile", "--confirm-reps", "2",
                       "--flaky-confirm-reps", "3", "--confirm-top", "1")
    stored = SlepcConfig.make(2, 1, "none", {"st_mat_mumps_icntl_7": "2"})
    slepc_profile.save_profiles([{**_profile("case", "tiny", ranks=2), "config": stored.to_json(),
                                  "tuned": {"valid": True, "solver_s": 1.0, "flaky": True}}], args.profiles_file)
    tuner = _SyntheticTuner(lambda cfg, n: (_speed_model(cfg, n)[0], True), "tiny", args)
    assert tuner.flaky                                  # recorded by the earlier tuning
    doc = tuner.tune()
    conf = doc["confirmation"]
    assert conf["start"] == stored.label() and stored.label() in conf["configs"]
    # the stored flag alone extends the confirmation of every valid configuration
    assert all(s["n"] == 3 for s in conf["configs"].values()) and len(conf["configs"]) == 3


# ------------------------------------------------------------------------------------ compare

def _entry(solver, t, ok=True, **extra):
    return {"solver": solver, "solver_s": t, "solver_s_scope": "x", "wall_s": t + 1, "peak_rss_gib": 0.5,
            "solver_peak_bytes": 2**29, "success": ok, "max_eta": 1e-14, "pairs": 4, "eigenvalues": [1.0],
            "phases": {"analysis_s": 0.1, "numeric_s": 0.2, "factorizations": 1, "solves": 20, "solve_s": t / 2},
            **extra}


def test_compare_summarize_medians():
    s = compare.summarize([_entry("slepc", 2.0, config="1x2", config_source="default"), _entry("slepc", 4.0),
                           _entry("slepc", 3.0, ok=False)])
    assert s["solver_s"] == 3.0 and (s["solver_s_min"], s["solver_s_max"]) == (2.0, 4.0) and not s["ok"]
    assert s["phases"]["solve_s"] == 1.5 and s["config"] == "1x2" and s["solver_peak_gib"] == 0.5
    assert compare.summarize([_entry("hsolve", 1.0)])["ok"]


def test_compare_render_markdown():
    doc = {"tag": "t", "cores": "40,41", "ncores": 2, "slepc_profile": "best", "reps": 3,
           "solvers": ["hsolve", "slepc"],
           "rows": [{"case": "c1", "n": 10, "problem": "HEP", "target_desc": "k=4 @ 1",
                     "solvers": {"hsolve": {**compare.summarize([_entry("hsolve", 1.0), _entry("hsolve", 1.2)]),
                                            "set_err_vs_ref": 1e-15},
                                 "slepc": {**compare.summarize([_entry("slepc", 2.4, config="2x1",
                                                                       config_source="case profile c1")])}}},
                    {"case": "c2", "n": 10, "problem": "GHEP", "target_desc": "[1, 2]",
                     "solvers": {"hsolve": compare.summarize([_entry("hsolve", 1.0)]),
                                 "slepc": compare.summarize([_entry("slepc", 3.0, ok=False)])}}]}
    md = compare.render_markdown(doc)
    lines = md.splitlines()
    assert lines[0] == "# t" and "SLEPc profile `best`" in md
    hs = next(line for line in lines if line.startswith("| c1 | 10 | HEP | k=4 @ 1 | hsolve"))
    assert "1.10 [1.00-1.20]" in hs and "ref 1e-15" in hs and "| yes |" in hs
    sl = next(line for line in lines if line.startswith("| c1 | 10 | HEP | k=4 @ 1 | slepc"))
    assert "`2x1` (case profile c1)" in sl and "0.20 (1)" in sl and "20 / 1.20" in sl
    assert "| c1 | 2.18x (1.62x) |" in md
    assert "| c2 | 3.00x (2.00x) (not both ok) |" in md


def test_compare_reference_sets_and_errors():
    ref = cio.Reference(np.array([-1.0, 0.5, 1.0, 2.0, 3.0]), None, {"kind": "dense", "complete": True})
    assert compare.expected_values(ref, {"mode": "nearest", "sigma": 0.9, "k": 2}).tolist() == [0.5, 1.0]
    assert compare.expected_values(ref, {"mode": "interval", "lower": 0.0, "upper": 2.5}).tolist() == [0.5, 1.0, 2.0]
    assert compare.expected_values(cio.Reference(ref.eigenvalues, None, {"kind": "solver", "complete": False}),
                                   {"mode": "nearest", "sigma": 0, "k": 1}) is None
    assert compare.set_error(np.array([1.0, 2.0]), np.array([2.0, 1.0 + 1e-3]), 1.0) == pytest.approx(1e-3)
    assert compare.set_error(np.array([1.0]), np.array([1.0, 2.0]), 1.0) is None
    assert compare.target_desc({"mode": "interval", "lower": 40, "upper": 200}) == "[40, 200]"


def test_compare_check_sets_marks_wrong_sets():
    want = np.array([1.0, 2.0, 2.0])
    row = {"solvers": {"hsolve": {"ok": True}, "slepc": {"ok": True}}}
    sets = {"hsolve": [np.array([2.0, 1.0, 2.0]), np.array([1.0, 2.0, 2.0 + 1e-12])],
            "slepc": [np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])]}   # 2nd rep missed a copy
    compare.check_sets(row, sets, want, 3.0)
    hs, sl = row["solvers"]["hsolve"], row["solvers"]["slepc"]
    assert hs["ok"] and hs["set_err_vs_ref"] < 1e-12 and hs["set_diff_vs_slepc"] < 1e-12
    assert not sl["ok"] and sl["set_wrong"] and sl["set_err_vs_ref"] == pytest.approx(1 / 3)
    row = {"solvers": {"slepc": {"ok": True}}}
    compare.check_sets(row, {"slepc": [np.array([1.0, 2.0])]}, want, 3.0)
    assert row["solvers"]["slepc"]["set_mismatch"] == "2 values, reference has 3" and not row["solvers"]["slepc"]["ok"]
    row = {"solvers": {"slepc": {"ok": True}}}
    compare.check_sets(row, {"slepc": [np.array([1.0])]}, None, 3.0)   # no reference: nothing to judge
    assert row["solvers"]["slepc"] == {"ok": True}
    doc = {"tag": "t", "cores": "1", "ncores": 1, "slepc_profile": "default", "reps": 1, "solvers": ["slepc"],
           "rows": [{"case": "c", "n": 3, "problem": "HEP", "target_desc": "k=3 @ 0",
                     "solvers": {"slepc": {**compare.summarize([_entry("slepc", 1.0)]), "ok": False, "set_wrong": True}}}]}
    assert "| NO (set) |" in compare.render_markdown(doc)


def test_compare_slepc_config_and_hsolve_command(tmp_path):
    cio.save_case("tiny", sp.identity(3, format="csr"), meta={"family": "test", "problem": "HEP"},
                  config=cio.default_config(0.0, 1), root=tmp_path)
    c = cio.load_case("tiny", root=tmp_path)
    pf = tmp_path / "p.json"
    slepc_profile.save_profiles([_profile("case", "tiny", ranks=2, eps_ncv="3k")], pf)
    args = compare.build_parser().parse_args(["tiny", "--tag", "t", "--cores", "40,41", "--slepc-profile", "best",
                                              "--slepc-profile-file", str(pf), "--root", str(tmp_path)])
    cfg, src = compare.slepc_config(c, "nearest", 2, args)
    assert cfg.label() == "2x1 eps_ncv=3k" and src == "case profile tiny"
    cfg, src = compare.slepc_config(c, "interval", 2, args)
    assert cfg == slepc_profile.default_config(2) and src == "default (no tuned profile)"
    args.slepc_profile, args.slepc_ranks = "default", 2
    assert compare.slepc_config(c, "nearest", 2, args) == (SlepcConfig(2, 1), "default")
    out = tmp_path / "raw" / "hsolve-nocert"
    cmd = compare.hsolve_command(c, "hsolve-nocert", out, 2, args, {"mode": "interval", "lower": 0, "upper": 1})
    cfg_doc = json.loads((out.parent / "config-hsolve-nocert.json").read_text())
    assert cfg_doc["inertia_check"] == "off" and cfg_doc["target"]["mode"] == "interval"
    assert cmd[cmd.index("--config") + 1] == str(out.parent / "config-hsolve-nocert.json")
    assert "--B" not in cmd and cmd[cmd.index("--threads") + 1] == "2"
    ph = compare.hsolve_phases({"factor_stats": {"analyze_s": 1.0, "factorize_s": 2.0, "factorizations": 3,
                                                 "rhs_solved": 40, "solve_s": 0.5},
                                "time_iterate_s": 4.0, "time_verify_s": 0.3})
    assert ph == {"analysis_s": 1.0, "numeric_s": 2.0, "factorizations": 3, "solves": 40, "solve_s": 0.5,
                  "ortho_s": None, "iterate_s": 4.0, "verify_s": 0.3}
