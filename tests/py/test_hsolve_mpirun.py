"""Tests of the multi-process launcher apps/hsolve-mpirun (DESIGN §3.9, docs/PARALLEL.md) with stand-ins
for hsolve, hsolve-mpi and mpiexec - no MPI, no solve:

* on one NUMA node (the run pinned to one or two of the allowed CPUs) it runs the plain hsolve with the
  same arguments, without mpiexec (review p10mpi, finding 3: it used to start a one-rank MPI job);
* --ranks R starts an MPI job whose ranks get their cores and thread counts through the environment, so
  that a launcher without Open MPI's -x (Hydra: MPICH, Intel MPI, rank in PMI_RANK) works too, and
  Open MPI-only options are passed only to Open MPI (finding 4)."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "apps" / "hsolve-mpirun"

pytestmark = pytest.mark.skipif(shutil.which("taskset") is None or not hasattr(os, "sched_getaffinity"),
                                reason="needs taskset and CPU affinity (Linux)")


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture()
def fakes(tmp_path):
    """hsolve and hsolve-mpi that print what they got; mpiexec stand-ins for Open MPI and Hydra that
    start each rank with the rank variable of their kind (and log that they were called)."""
    log = tmp_path / "mpiexec.log"
    _script(tmp_path / "hsolve", 'echo "plain hsolve: $*"\n')
    _script(tmp_path / "hsolve-mpi",
            'echo "rank ${OMPI_COMM_WORLD_RANK:-$PMI_RANK} threads $OMP_NUM_THREADS '
            'cpus $(taskset -pc $$ | sed "s/.*: //") args $*"\n')
    launch = ('[ "$1" = -n ] || exit 3; n=$2; shift 2\n'
              'while [ "${1#-}" != "$1" ]; do case "$1" in -x | --bind-to) shift 2 ;; *) shift ;; esac; done\n'
              'for ((r = 0; r < n; r++)); do {var}=$r "$@" & done; wait\n')
    ompi = _script(tmp_path / "mpiexec-ompi",
                   f'echo "$*" >> {log}\n'
                   'if [ "$1" = --version ]; then echo "mpiexec (Open MPI) 5.0.11"; exit 0; fi\n'
                   + launch.replace("{var}", "OMPI_COMM_WORLD_RANK"))
    hydra = _script(tmp_path / "mpiexec-hydra",
                    f'echo "$*" >> {log}\n'
                    'if [ "$1" = --version ]; then echo "HYDRA build details:"; exit 0; fi\n'
                    + launch.replace("{var}", "PMI_RANK"))
    env = dict(os.environ, HSOLVE=str(tmp_path / "hsolve"), HSOLVE_MPI=str(tmp_path / "hsolve-mpi"))
    env.pop("HSOLVE_MPIRUN_PLAN", None)
    return {"ompi": ompi, "hydra": hydra, "log": log, "env": env}


def _cpus(k):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < k:
        pytest.skip(f"needs {k} allowed CPUs")
    return cpus[:k]


def _run(cpus, args, env):
    cmd = ["taskset", "-c", ",".join(map(str, cpus)), "bash", str(LAUNCHER), *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)


def _same_node(cpus):
    nodes = set()
    for c in cpus:
        hits = list(Path("/sys/devices/system/cpu", f"cpu{c}").glob("node[0-9]*"))
        nodes.add(hits[0].name if hits else "node0")
    return len(nodes) == 1


def test_one_numa_node_runs_plain_hsolve(fakes):
    cpus = _cpus(1)
    r = _run(cpus, ["--mpiexec", str(fakes["ompi"]), "--", "--A", "a b.npz", "--threads", "0"], fakes["env"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "plain hsolve: --A a b.npz --threads 0"
    assert not fakes["log"].exists()  # no mpiexec, not even its --version
    d = _run(cpus, ["--dry-run", "--", "--A", "x.npz"], fakes["env"])
    assert d.returncode == 0, d.stderr
    assert "plain hsolve, no MPI job" in d.stdout
    assert d.stdout.strip().endswith(f"{fakes['env']['HSOLVE']} --A x.npz")


@pytest.mark.parametrize("kind", ["ompi", "hydra"])
def test_forced_ranks_reach_every_rank_through_the_environment(fakes, kind):
    cpus = _cpus(2)
    if not _same_node(cpus):
        pytest.skip("the first two allowed CPUs are on two NUMA nodes")
    r = _run(cpus, ["--ranks", "3", "--mpiexec", str(fakes[kind]), "--", "--A", "x.npz"], fakes["env"])
    assert r.returncode == 0, r.stderr
    lines = sorted(r.stdout.strip().splitlines())
    a, b = cpus
    assert lines == [f"rank 0 threads 2 cpus {a},{b} args --A x.npz",
                     f"rank 1 threads 1 cpus {a} args --A x.npz",
                     f"rank 2 threads 1 cpus {b} args --A x.npz"]
    launched = fakes["log"].read_text().splitlines()[-1]
    if kind == "ompi":
        assert "--bind-to none -x HSOLVE_MPIRUN_PLAN -x HSOLVE_MPIRUN_EXE" in launched
    else:
        assert "-x" not in launched.split() and "--bind-to" not in launched


def test_a_rank_without_a_rank_variable_is_an_error(fakes, tmp_path):
    cpus = _cpus(2)
    if not _same_node(cpus):
        pytest.skip("the first two allowed CPUs are on two NUMA nodes")
    bare = _script(tmp_path / "mpiexec-bare",
                   'if [ "$1" = --version ]; then echo "some MPI"; exit 0; fi\n'
                   'n=$2; shift 2; for ((r = 0; r < n; r++)); do "$@" & done; wait\n')
    r = _run(cpus, ["--ranks", "3", "--mpiexec", str(bare), "--", "--A", "x.npz"], fakes["env"])
    assert "sets no rank variable" in r.stderr
    assert "rank" not in r.stdout
