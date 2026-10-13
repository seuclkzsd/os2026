"""Tests of the resource guard (tools/run/guard.py): the node scheduling policy (measurement slot,
per-run memory cap, lock wait and holder record, nested guards) and signal handling.

Every test runs the guard as a subprocess with its own lock file and policy variables, pinned to
cores this process may use, and runs only trivial commands."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[2] / "tools/run/guard.py"
CPUS = sorted(os.sched_getaffinity(0))
LANE, SLOT = str(CPUS[0]), str(CPUS[-1])   # the same core when only one is available
PRINT_AFFINITY = [sys.executable, "-c",
                  "import json, os; print(json.dumps({'aff': sorted(os.sched_getaffinity(0)), "
                  "'held': os.environ.get('HS_SLOT_HELD')}))"]


def _env(lock, **policy):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HS_")}
    env.update(HS_LOCK=str(lock), HS_CORES=LANE, HS_MEM_GB="1", HS_MIN_AVAIL_GB="0")
    env.update({k: str(v) for k, v in policy.items()})
    return env


def _guard(lock, args, cmd=("true",), **policy):
    return subprocess.run([sys.executable, str(GUARD), "--skip-busy-check", *args, "--", *cmd],
                          env=_env(lock, **policy), capture_output=True, text=True, timeout=60)


def _summary(stderr):
    return json.loads(stderr.strip().splitlines()[-1].split("[guard] ", 1)[1])


def test_busy_check_covers_hyperthread_siblings():
    sys.path.insert(0, str(GUARD.parent))
    import guard
    cpus = guard.with_siblings([int(SLOT)])
    assert int(SLOT) in cpus
    sib = Path(f"/sys/devices/system/cpu/cpu{SLOT}/topology/thread_siblings_list")
    if sib.exists():
        assert cpus == guard.parse_cpulist(sib.read_text())


def test_without_policy_any_core_runs_unlocked(tmp_path):
    r = _guard(tmp_path / "l", ["--cores", SLOT], PRINT_AFFINITY)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"aff": [int(SLOT)], "held": None}
    assert _summary(r.stderr)["exclusive"] is False


def test_slot_cores_need_the_lock(tmp_path):
    r = _guard(tmp_path / "l", ["--cores", SLOT], HS_MEASURE_CORES=SLOT)
    assert r.returncode == 2
    assert "measurement slot" in r.stderr


def test_exclusive_defaults_to_the_slot_and_the_cap(tmp_path):
    lock = tmp_path / "l"
    r = _guard(lock, ["--exclusive", "--timeout", "30"], PRINT_AFFINITY,
               HS_MEASURE_CORES=SLOT, HS_MEM_CAP_GB=3)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"aff": [int(SLOT)], "held": str(lock)}
    s = _summary(r.stderr)
    assert (s["cores"], s["mem_limit_gib"], s["exclusive"]) == (SLOT, 3.0, True)
    assert not Path(str(lock) + ".owner").exists()   # the holder record goes with the lock


def test_per_run_memory_cap(tmp_path):
    r = _guard(tmp_path / "l", ["--exclusive", "--timeout", "30", "--mem-gb", "5"], HS_MEM_CAP_GB=4)
    assert r.returncode == 2
    assert "HS_MEM_CAP_GB" in r.stderr


def test_shared_memory_ceiling_outside_the_slot(tmp_path):
    lock = tmp_path / "l"
    r = _guard(lock, ["--mem-gb", "2"], HS_MAX_SHARED_MEM_GB=1)
    assert r.returncode == 2
    assert "HS_MAX_SHARED_MEM_GB" in r.stderr
    # inside a run that holds the lock the ceiling is the per-run cap
    r = _guard(lock, ["--mem-gb", "2"], HS_MAX_SHARED_MEM_GB=1, HS_SLOT_HELD=lock)
    assert r.returncode == 0, r.stderr


def test_slot_runs_need_a_bounded_timeout(tmp_path):
    lock = tmp_path / "l"
    assert _guard(lock, ["--exclusive"], HS_SLOT_MAX_S=10).returncode == 2
    assert _guard(lock, ["--exclusive", "--timeout", "20"], HS_SLOT_MAX_S=10).returncode == 2
    r = _guard(lock, ["--exclusive", "--timeout", "5"], HS_SLOT_MAX_S=10)
    assert r.returncode == 0, r.stderr


def test_busy_slot_gives_up_and_names_the_holder(tmp_path):
    lock = tmp_path / "l"
    status = [sys.executable, str(GUARD), "--status"]
    assert _run_status(status, lock).startswith("free")
    with open(lock, "a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        Path(str(lock) + ".owner").write_text(json.dumps(
            {"pid": 4242, "dir": "wt-holder", "since": "2026-10-13 16:00:00", "cmd": ["bench", "x"]}))
        t0 = time.time()
        r = _guard(lock, ["--exclusive", "--timeout", "30", "--lock-wait", "1"])
        assert r.returncode == 75
        assert 1 <= time.time() - t0 < 30
        assert "held by pid 4242 (wt-holder)" in r.stderr
        assert "busy: held by pid 4242 (wt-holder)" in _run_status(status, lock)
    assert _run_status(status, lock).startswith("free")


def test_waiting_run_gets_the_slot_when_it_is_released(tmp_path):
    lock = tmp_path / "l"
    with open(lock, "a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        waiter = subprocess.Popen([sys.executable, str(GUARD), "--skip-busy-check", "--exclusive", "--timeout", "30",
                                   "--lock-wait", "30", "--", "true"], env=_env(lock),
                                  stderr=subprocess.PIPE, text=True)
        time.sleep(1.0)
        assert waiter.poll() is None                  # still waiting
    t0 = time.time()                                  # released
    assert waiter.wait(timeout=30) == 0
    assert time.time() - t0 < 5
    assert "waiting up to 30 s" in waiter.stderr.read()


def _run_status(cmd, lock):
    return subprocess.run(cmd, env=_env(lock), capture_output=True, text=True, timeout=30, check=True).stdout


def test_nested_exclusive_shares_the_held_lock(tmp_path):
    lock = tmp_path / "l"
    inner = [sys.executable, str(GUARD), "--skip-busy-check", "--exclusive", "--timeout", "30",
             "--lock-wait", "2", "--", *PRINT_AFFINITY]
    r = _guard(lock, ["--exclusive", "--timeout", "30", "--lock-wait", "2"], inner,
               HS_MEASURE_CORES=SLOT, HS_SLOT_MAX_S=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"aff": [int(SLOT)], "held": str(lock)}


def _start_sleeper(tmp_path, ignore_hup=False):
    pidfile = tmp_path / "child.pid"
    summary = tmp_path / "summary.json"
    cmd = [sys.executable, str(GUARD), "--skip-busy-check", "--summary", str(summary), "--",
           "bash", "-c", f"echo $$ > {pidfile}; exec sleep 60"]
    # explicit either way: the test runner itself may run with SIGHUP ignored (nohup)
    pre = lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN if ignore_hup else signal.SIG_DFL)  # noqa: E731
    guard = subprocess.Popen(cmd, env=_env(tmp_path / "l"), stderr=subprocess.PIPE, text=True, preexec_fn=pre)
    for _ in range(200):
        if pidfile.exists() and pidfile.read_text().strip():
            return guard, int(pidfile.read_text()), summary
        time.sleep(0.05)
    guard.kill()
    pytest.fail("the guarded child did not start")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:          # a zombie is not running any more
        return f.read().split(")")[-1].split()[0] != "Z"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_termination_signal_kills_the_tree(tmp_path, sig):
    guard, child, summary = _start_sleeper(tmp_path)
    guard.send_signal(sig)
    assert guard.wait(timeout=20) == 137
    assert not _alive(child)
    assert json.loads(summary.read_text())["killed"] == f"guard received {sig.name}"


def test_ignored_sighup_stays_ignored(tmp_path):
    guard, child, _ = _start_sleeper(tmp_path, ignore_hup=True)
    try:
        guard.send_signal(signal.SIGHUP)
        time.sleep(1.0)
        assert guard.poll() is None and _alive(child)
    finally:
        guard.send_signal(signal.SIGTERM)
        guard.wait(timeout=20)
    assert not _alive(child)
