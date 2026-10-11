"""SLEPc baseline configurations and the tuned-profile file.

A configuration (SlepcConfig) is an MPI-rank x OpenMP-thread layout, the mpiexec binding and a set
of PETSc options for tools/bench/baseline_slepc.py (MUMPS controls carry the spectral transform's
prefix: st_mat_mumps_icntl_7 = MUMPS ICNTL(7)). Option values may be k-rules - "2k", "k+15",
"3k+1" - expanded with the target's k when the command is built, so one profile serves any k.

Profiles (tools/bench/slepc_profiles.json, written by tools/bench/tune_slepc.py, read by
tools/bench/compare.py --slepc-profile best):

    {"schema_version": 1, "profiles": [
       {"scope": "case",  "name": "syn-mem27-46", "mode": "nearest", "cores": 2,
        "config": {...}, "tuned": {...measurements...}},
       {"scope": "class", "name": "HEP/nearest", "mode": "nearest", "cores": 2,
        "config": {...}, "from_cases": [...]}]}

lookup() takes the case's own profile for the core count and target mode, else the class
"<problem>/<mode>" profile for that core count, else None (= the default configuration). A profile
is only valid for the core count it was tuned at (the layout is part of it).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
REF_ENV = Path(os.environ.get("HS_REF_ENV", Path.home() / "miniconda3/envs/os2026-ref"))
BASELINE = REPO / "tools/bench/baseline_slepc.py"
PROFILE_FILE = REPO / "tools/bench/slepc_profiles.json"
SCHEMA_VERSION = 1
BINDS = ("none", "core")
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_K_RULE = re.compile(r"^(\d*)k([+-]\d+)?$")


def expand_value(value: str, k: int | None) -> str:
    """A k-rule ("2k", "k+15", "3k-2") evaluated with k; any other value unchanged."""
    m = _K_RULE.match(value)
    if not m:
        return value
    if k is None:
        raise ValueError(f"option value {value!r} is a k-rule but the target has no k")
    return str(int(m.group(1) or 1) * int(k) + int(m.group(2) or 0))


@dataclass(frozen=True)
class SlepcConfig:
    """Layout + PETSc options of one baseline run. options: sorted (name, value) pairs, names
    without the leading '-'. Use make() to build one from a dict."""
    ranks: int = 1
    threads: int = 1
    bind: str = "none"
    options: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.ranks < 1 or self.threads < 1:
            raise ValueError(f"ranks and threads must be >= 1, got {self.ranks}x{self.threads}")
        if self.bind not in BINDS:
            raise ValueError(f"bind must be one of {BINDS}, got {self.bind!r}")
        names = [n for n, _ in self.options]
        if names != sorted(set(names)):
            raise ValueError("options must be sorted by name and unique (use SlepcConfig.make)")
        for n, v in self.options:
            if not _OPTION_NAME.match(n):
                raise ValueError(f"bad PETSc option name {n!r} (lower case, no leading '-')")
            if not isinstance(v, str):
                raise ValueError(f"option {n} value must be a string, got {v!r}")

    @staticmethod
    def make(ranks: int = 1, threads: int = 1, bind: str = "none",
             options: dict[str, Any] | Iterable[tuple[str, Any]] = ()) -> "SlepcConfig":
        items = options.items() if isinstance(options, dict) else options
        opts = {str(n).lstrip("-"): str(v) for n, v in items if v is not None}
        return SlepcConfig(ranks, threads, bind, tuple(sorted(opts.items())))

    @property
    def cores(self) -> int:
        return self.ranks * self.threads

    def option(self, name: str) -> str | None:
        return dict(self.options).get(name)

    def with_options(self, **changes: Any) -> "SlepcConfig":
        """Copy with options set (None removes one)."""
        opts = dict(self.options)
        for n, v in changes.items():
            if v is None:
                opts.pop(n, None)
            else:
                opts[n] = str(v)
        return SlepcConfig.make(self.ranks, self.threads, self.bind, opts)

    def with_layout(self, ranks: int, threads: int, bind: str | None = None) -> "SlepcConfig":
        return SlepcConfig(ranks, threads, self.bind if bind is None else bind, self.options)

    def label(self) -> str:
        """Short unique description, e.g. '2x1/core st_mat_mumps_icntl_7=5 eps_ncv=3k'."""
        head = f"{self.ranks}x{self.threads}" + ("" if self.bind == "none" else f"/{self.bind}")
        return " ".join([head] + [f"{n}={v}" for n, v in self.options])

    def petsc_args(self, k: int | None = None) -> list[str]:
        out: list[str] = []
        for n, v in self.options:
            out += [f"-{n}", expand_value(v, k)]
        return out

    def to_json(self) -> dict[str, Any]:
        return {"ranks": self.ranks, "threads": self.threads, "bind": self.bind, "options": dict(self.options)}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "SlepcConfig":
        return SlepcConfig.make(int(d.get("ranks", 1)), int(d.get("threads", 1)), d.get("bind", "none"),
                                d.get("options", {}))


def default_config(cores: int) -> SlepcConfig:
    """The untuned baseline: one MPI rank with every core as an OpenMP thread, SLEPc/MUMPS
    defaults (what tools/bench/compare.py ran before profiles existed, --slepc-ranks 1)."""
    return SlepcConfig(1, cores)


def mpiexec_command(cfg: SlepcConfig, case_name: str, out_dir: str | Path, *, k: int | None = None,
                    target: dict[str, Any] | None = None, log_view: str | Path | None = None,
                    root: str | Path | None = None, ref_env: Path = REF_ENV) -> list[str]:
    """argv running baseline_slepc.py with this configuration (launch it under tools/run/guard.py)."""
    # Binding is done by baseline_slepc.py itself (--bind core: each rank takes its block of the
    # guard's cpuset); mpiexec's --bind-to core would bind outside that cpuset.
    cmd = [str(ref_env / "bin/mpiexec"), "-n", str(cfg.ranks), "--bind-to", "none",
           "-x", f"OMP_NUM_THREADS={cfg.threads}", "-x", f"OPENBLAS_NUM_THREADS={cfg.threads}"]
    if cfg.bind == "core" and cfg.threads > 1:   # OpenMP threads pinned within the rank's block
        cmd += ["-x", "OMP_PROC_BIND=close", "-x", "OMP_PLACES=cores"]
    cmd += [str(ref_env / "bin/python"), str(BASELINE), "--case", case_name, "--out", str(out_dir)]
    if cfg.bind != "none":
        cmd += ["--bind", cfg.bind]
    if root is not None:
        cmd += ["--root", str(root)]
    if target is not None:
        cmd += ["--target", json.dumps(target, separators=(",", ":"))]
    cmd += cfg.petsc_args(k)
    if log_view is not None:
        cmd += ["-log_view", f":{log_view}"]
    return cmd


# ------------------------------------------------------------------------------------ profiles

def class_name(problem: str, mode: str) -> str:
    return f"{problem}/{mode}"


def load_profiles(path: str | Path = PROFILE_FILE) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{p}: schema_version {doc.get('schema_version')} != {SCHEMA_VERSION}")
    return list(doc.get("profiles", []))


def save_profiles(profiles: list[dict[str, Any]], path: str | Path = PROFILE_FILE) -> None:
    order = {"case": 0, "class": 1}
    profiles = sorted(profiles, key=lambda p: (order.get(p["scope"], 2), p["name"], p["mode"], p["cores"]))
    doc = {"schema_version": SCHEMA_VERSION,
           "note": "written by tools/bench/tune_slepc.py; read by tools/bench/compare.py --slepc-profile best "
                   "(tools/bench/slepc_profile.py documents the format)",
           "profiles": profiles}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def upsert(profiles: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """profiles with `entry` replacing the one of the same (scope, name, mode, cores)."""
    key = (entry["scope"], entry["name"], entry["mode"], entry["cores"])
    return [p for p in profiles if (p["scope"], p["name"], p["mode"], p["cores"]) != key] + [entry]


def lookup(profiles: list[dict[str, Any]], case_name: str, problem: str, mode: str,
           cores: int) -> tuple[SlepcConfig, dict[str, Any]] | None:
    """(config, profile entry) for a case: its own profile, else its class's, else None."""
    for scope, name in (("case", case_name), ("class", class_name(problem, mode))):
        for p in profiles:
            if p["scope"] == scope and p["name"] == name and p["mode"] == mode and int(p["cores"]) == cores:
                return SlepcConfig.from_json(p["config"]), p
    return None


def class_profiles(profiles: list[dict[str, Any]], problems: dict[str, str]) -> list[dict[str, Any]]:
    """Class profiles ("<problem>/<mode>", per core count) from the valid case profiles. problems
    maps case name -> problem type. Per knob - the layout (ranks, threads, bind) and each option -
    the value most of the class's cases chose (an option absent in most cases stays at its default);
    ties go to the case with the largest tuned solver time (where the choice matters most). A
    heuristic fallback for untuned cases: the combination itself was not measured as a whole."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for p in profiles:
        if p["scope"] == "case" and p.get("tuned", {}).get("valid", False) and p["name"] in problems:
            groups.setdefault((class_name(problems[p["name"]], p["mode"]), p["mode"], int(p["cores"])), []).append(p)
    out = []
    for (cname, mode, cores), members in sorted(groups.items()):
        members = sorted(members, key=lambda p: -float(p.get("tuned", {}).get("solver_s", 0.0) or 0.0))
        cfgs = [SlepcConfig.from_json(p["config"]) for p in members]

        def vote(values: list[Any]) -> Any:
            counts = Counter(values)
            best = max(counts.values())
            return next(v for v in values if counts[v] == best)   # members are ordered by solver time

        ranks, threads, bind = vote([(c.ranks, c.threads, c.bind) for c in cfgs])
        names = sorted({n for c in cfgs for n, _ in c.options})
        opts = {}
        for n in names:
            v = vote([c.option(n) for c in cfgs])
            if v is not None:
                opts[n] = v
        out.append({"scope": "class", "name": cname, "mode": mode, "cores": cores,
                    "config": SlepcConfig.make(ranks, threads, bind, opts).to_json(),
                    "from_cases": [p["name"] for p in members]})
    return out
