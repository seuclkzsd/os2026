#!/usr/bin/env python3
"""Resource-guarded command runner for the shared development machine.

Every experiment and every non-trivial test must be launched through this
script.  It enforces the project's resource policy:

  * CPU:    the child is pinned (sched_setaffinity, inherited by all
            descendants) to a fixed set of physical cores, and the usual
            threading environment variables are set to the same count.
  * Memory: the summed RSS of the whole process tree is sampled every
            ``--interval`` seconds; the tree is killed when it exceeds
            ``--mem-gb``, or when system-wide MemAvailable drops below
            ``--min-avail-gb`` AND the tree itself is large (>= max(2 GiB,
            mem_gb/4)), so small builds/tests of parallel sessions are never
            collateral damage.  The reported peak is the max of the sampled sum,
            the summed kernel high-water marks (VmHWM) and ru_maxrss, so short
            spikes between samples are not missed.  The cap is exported to the
            child as HSOLVE_MEMORY_LIMIT_GB (GiB; the smaller of --mem-gb and
            a value inherited from an enclosing guard or the caller), from which
            hsolve's automatic memory budget is derived (docs/DESIGN.md §3.6-3).
  * Time:   optional wall-clock timeout; SIGTERM / SIGHUP to the guard kill
            the process tree as well (a SIGHUP ignored by nohup stays ignored).
  * Mutual exclusion: ``--exclusive`` takes a machine-wide flock so that
            heavy experiments run one at a time.  Guards started inside an
            ``--exclusive`` run (HS_SLOT_HELD) share the held lock instead of
            waiting for it.

Node scheduling (set by env/hosts/<host>.sh on the cluster nodes; unset = no policy):
  HS_MEASURE_CORES      the measurement slot (cluster: 16-31 = NUMA node 1).  A run pinned to any
                        of these cores must hold the lock (--exclusive), so timed runs never share
                        the socket; --exclusive without --cores pins to the whole slot.
  HS_MEM_CAP_GB         per-run RSS ceiling (cluster: 32 GiB, the judging machine); --mem-gb above
                        it is refused; --exclusive without --mem-gb uses it.
  HS_MAX_SHARED_MEM_GB  RSS ceiling of runs that do not hold the lock.
  HS_SLOT_MAX_S         longest --timeout of a run that takes the lock, so no run hogs the slot.
  HS_LOCK_WAIT_S        default of --lock-wait: give up after so many seconds of waiting for the
                        lock (exit status 75, the holder is printed; 0 = wait as long as it takes).
``--status`` prints whether the slot is free and who holds it.

A JSON summary (exit status, wall time, peak RSS, kill reason, cores) is
printed to stderr as the last line and optionally written to ``--summary``.

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CORES = os.environ.get("HS_CORES", "24-30,33-41")
MEMORY_LIMIT_ENV = "HSOLVE_MEMORY_LIMIT_GB"  # hs::kMemoryLimitEnv: the cap the child may plan with
DEFAULT_MEM_GB = float(os.environ.get("HS_MEM_GB", "30"))
DEFAULT_MIN_AVAIL_GB = float(os.environ.get("HS_MIN_AVAIL_GB", "12"))
# During parallel multi-agent phases set HS_MAX_SHARED_MEM_GB (e.g. 8): larger --mem-gb then
# requires --exclusive, so heavy runs never overlap.
MAX_SHARED_MEM_GB = float(os.environ.get("HS_MAX_SHARED_MEM_GB", "inf"))
MEASURE_CORES = os.environ.get("HS_MEASURE_CORES", "")
MEM_CAP_GB = float(os.environ.get("HS_MEM_CAP_GB", "inf"))
SLOT_MAX_S = float(os.environ.get("HS_SLOT_MAX_S", "inf"))
LOCK_WAIT_S = float(os.environ.get("HS_LOCK_WAIT_S", "0"))
LOCK_PATH = os.environ.get("HS_LOCK", "/tmp/hsolve-guard-exclusive.lock")
OWNER_PATH = LOCK_PATH + ".owner"
EXIT_SLOT_BUSY = 75  # EX_TEMPFAIL: the lock stayed taken for --lock-wait seconds
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def parse_cpulist(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def cpu_times() -> dict[int, tuple[int, int]]:
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = list(map(int, p[1:]))
                out[int(p[0][3:])] = (sum(v), v[3] + v[4])
    return out


def busy_cores(cpus: list[int], window: float = 1.0, threshold: float = 50.0) -> list[int]:
    """Return the subset of ``cpus`` whose utilisation over ``window`` s exceeds ``threshold`` %."""
    a = cpu_times()
    time.sleep(window)
    b = cpu_times()
    busy = []
    for c in cpus:
        tot = b[c][0] - a[c][0]
        idle = b[c][1] - a[c][1]
        if tot > 0 and 100.0 * (1 - idle / tot) > threshold:
            busy.append(c)
    return busy


def with_siblings(cpus: list[int]) -> list[int]:
    """cpus plus their hyperthread siblings: a job on a sibling slows a pinned run as much as one on
    the core itself."""
    out = set(cpus)
    for c in cpus:
        try:
            out.update(parse_cpulist(Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text()))
        except OSError:
            pass
    return sorted(out)


def child_memory_limit(mem_gb: float, inherited: str | None) -> float:
    """The memory limit the child is told (MEMORY_LIMIT_ENV, GiB): this guard's cap, or a smaller
    one declared by an enclosing guard or the caller (a malformed inherited value is ignored)."""
    limit = mem_gb
    try:
        outer = float(inherited) if inherited else math.inf
    except ValueError:
        outer = math.inf
    if 0 < outer < limit:
        limit = outer
    return limit


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("inf")


def tree_pids(root: int) -> list[int]:
    """All live descendants of ``root`` (including root), via /proc/<pid>/task/*/children."""
    pids, stack = [], [root]
    while stack:
        pid = stack.pop()
        pids.append(pid)
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{task}/children") as f:
                    stack.extend(int(x) for x in f.read().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return pids


def tree_rss_bytes(root: int) -> tuple[int, int]:
    """(sum of current RSS, sum of VmHWM) over the process tree, in bytes."""
    rss = hwm = 0
    for pid in tree_pids(root):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        hwm += int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rss, hwm


def slot_owner() -> str:
    """Who holds the lock, from the record its holder wrote (best effort)."""
    try:
        o = json.loads(Path(OWNER_PATH).read_text())
        return (f"held by pid {o['pid']} ({o['dir']}) since {o['since']}: "
                f"{' '.join(o['cmd'])[:200]}")
    except (OSError, ValueError, KeyError):
        return "holder unknown"


def try_lock(fd) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def blocking_lock(fd, wait_s: float) -> bool:
    """Block in flock for at most wait_s seconds (0 = no limit). A blocked waiter is woken as soon as
    the holder releases, so the holder's next run (a new process) does not overtake it."""
    if wait_s <= 0:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return True

    def expire(_signum, _frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, wait_s)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except TimeoutError:      # also when it fired just after the lock was taken: the caller closes fd
        return False
    finally:
        signal.signal(signal.SIGALRM, old)


def acquire_slot(wait_s: float, cmd: list[str]):
    """Take the machine-wide lock, waiting at most wait_s seconds (0 = no limit); None on timeout.
    The holder records itself in OWNER_PATH for waiting runs and --status."""
    fd = open(LOCK_PATH, "a")
    if not try_lock(fd):
        limit = f" up to {wait_s:.0f} s" if wait_s > 0 else ""
        print(f"[guard] waiting{limit} for exclusive lock {LOCK_PATH}: {slot_owner()}", file=sys.stderr, flush=True)
        if not blocking_lock(fd, wait_s):
            fd.close()
            return None
    record = {"pid": os.getpid(), "dir": os.path.basename(os.getcwd()),
              "since": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd}
    tmp = f"{OWNER_PATH}.{os.getpid()}"
    try:
        Path(tmp).write_text(json.dumps(record))
        os.replace(tmp, OWNER_PATH)
    except OSError:
        pass
    return fd


def release_slot(fd) -> None:
    try:
        os.unlink(OWNER_PATH)
    except OSError:
        pass
    fcntl.flock(fd, fcntl.LOCK_UN)
    fd.close()


class Terminated(Exception):
    """SIGTERM or SIGHUP received by the guard."""


def on_signal(signum, _frame) -> None:
    raise Terminated(signal.Signals(signum).name)


def kill_tree(proc: subprocess.Popen, grace: float = 5.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", help=f"cpulist to pin to (default: {DEFAULT_CORES}; "
                    f"with the lock: HS_MEASURE_CORES if set)")
    ap.add_argument("--threads", type=int, default=0, help="threads to advertise via env (default: #cores)")
    ap.add_argument("--mem-gb", type=float, help=f"RSS limit of the process tree in GiB (default: {DEFAULT_MEM_GB}; "
                    f"with the lock: HS_MEM_CAP_GB if set)")
    ap.add_argument("--min-avail-gb", type=float, default=DEFAULT_MIN_AVAIL_GB,
                    help="kill if system MemAvailable falls below this (GiB)")
    ap.add_argument("--timeout", type=float, default=0, help="wall-clock limit in seconds (0 = none)")
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval in seconds")
    ap.add_argument("--exclusive", action="store_true", help="serialize with other --exclusive runs")
    ap.add_argument("--lock-wait", type=float, default=LOCK_WAIT_S,
                    help="give up after waiting this many seconds for the lock, exit 75 (0 = no limit)")
    ap.add_argument("--status", action="store_true", help="print whether the lock is free and exit")
    ap.add_argument("--skip-busy-check", action="store_true", help="do not abort when pinned cores are busy")
    ap.add_argument("--nice", type=int, default=5)
    ap.add_argument("--summary", type=Path, help="also write the JSON summary to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (after --)")
    args = ap.parse_args(argv)

    if args.status:
        with open(LOCK_PATH, "a") as fd:
            if try_lock(fd):
                print(f"free ({LOCK_PATH}; measurement slot {MEASURE_CORES or 'not set'})")
            else:
                print(f"busy: {slot_owner()}")
        return 0

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("missing command")

    # An ancestor guard took the lock: share it (waiting for it would deadlock).
    inherited = os.environ.get("HS_SLOT_HELD") == LOCK_PATH
    slot = args.exclusive or inherited
    cores_spec = args.cores or (MEASURE_CORES if slot and MEASURE_CORES else DEFAULT_CORES)
    mem_gb = args.mem_gb if args.mem_gb is not None else (
        MEM_CAP_GB if slot and MEM_CAP_GB != float("inf") else DEFAULT_MEM_GB)
    if mem_gb > MEM_CAP_GB:
        ap.error(f"--mem-gb {mem_gb} > HS_MEM_CAP_GB={MEM_CAP_GB} (per-run limit)")
    if mem_gb > MAX_SHARED_MEM_GB and not slot:
        ap.error(f"--mem-gb {mem_gb} > HS_MAX_SHARED_MEM_GB={MAX_SHARED_MEM_GB}: add --exclusive")

    cpus = parse_cpulist(cores_spec)
    allowed = os.sched_getaffinity(0)
    if not set(cpus) <= allowed:
        ap.error(f"cores {sorted(set(cpus) - allowed)} are not available to this process")
    in_slot = sorted(set(cpus) & set(parse_cpulist(MEASURE_CORES)))
    if in_slot and not slot:
        ap.error(f"cores {in_slot} belong to the measurement slot HS_MEASURE_CORES={MEASURE_CORES}: "
                 f"add --exclusive (one run at a time) or use other cores")
    if args.exclusive and not inherited and SLOT_MAX_S != float("inf") and not 0 < args.timeout <= SLOT_MAX_S:
        ap.error(f"--exclusive needs 0 < --timeout <= HS_SLOT_MAX_S={SLOT_MAX_S:g} s")
    nthreads = args.threads or len(cpus)

    lock_fd = None
    if args.exclusive and not inherited:
        lock_fd = acquire_slot(args.lock_wait, cmd)
        if lock_fd is None:
            print(f"[guard] gave up after {args.lock_wait:g} s: exclusive lock {LOCK_PATH} {slot_owner()}",
                  file=sys.stderr, flush=True)
            return EXIT_SLOT_BUSY

    if not args.skip_busy_check:
        busy = busy_cores(with_siblings(cpus))
        if len(busy) > max(1, len(cpus) // 8):
            print(f"[guard] refusing to start: pinned cores busy {busy}; "
                  f"choose other cores with --cores", file=sys.stderr)
            if lock_fd is not None:
                release_slot(lock_fd)
            return 3

    avail = mem_available_gb()
    if avail - mem_gb < args.min_avail_gb:
        print(f"[guard] note: MemAvailable={avail:.1f} GiB; the run will be killed if it drops "
              f"below {args.min_avail_gb} GiB", file=sys.stderr)

    env = dict(os.environ)
    for var in THREAD_VARS:
        env[var] = str(nthreads)
    env["HS_GUARDED"] = "1"
    env[MEMORY_LIMIT_ENV] = f"{child_memory_limit(mem_gb, os.environ.get(MEMORY_LIMIT_ENV)):g}"
    if slot:
        env["HS_SLOT_HELD"] = LOCK_PATH
    else:
        env.pop("HS_SLOT_HELD", None)

    def preexec() -> None:
        os.setpgrp()
        os.sched_setaffinity(0, cpus)
        os.nice(args.nice)

    signal.signal(signal.SIGTERM, on_signal)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, on_signal)
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, preexec_fn=preexec)
    peak = 0
    peak_hwm = 0
    reason = None
    big_tree = max(2.0, mem_gb / 4) * 2**30
    try:
        while proc.poll() is None:
            rss, hwm = tree_rss_bytes(proc.pid)
            peak = max(peak, rss)
            peak_hwm = max(peak_hwm, hwm)
            if rss > mem_gb * 2**30:
                reason = f"rss {rss / 2**30:.2f} GiB > limit {mem_gb} GiB"
            elif rss >= big_tree and mem_available_gb() < args.min_avail_gb:
                reason = f"system MemAvailable < {args.min_avail_gb} GiB while tree uses {rss / 2**30:.1f} GiB"
            elif args.timeout and time.time() - t0 > args.timeout:
                reason = f"timeout {args.timeout}s"
            if reason:
                print(f"[guard] killing process tree: {reason}", file=sys.stderr, flush=True)
                kill_tree(proc)
                break
            # Wait on the child instead of sleeping, so wall_s is exact (not quantized to interval).
            try:
                proc.wait(timeout=args.interval)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        reason = "interrupted"
        kill_tree(proc)
    except Terminated as e:
        reason = f"guard received {e}"
        print(f"[guard] killing process tree: {reason}", file=sys.stderr, flush=True)
        kill_tree(proc)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) is on_signal:
            signal.signal(sig, signal.SIG_DFL)
    rc = proc.wait()
    wall = time.time() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    # ru_maxrss of children is a max over all reaped children of this guard (KiB on Linux).
    peak_reported = max(peak, peak_hwm, ru1.ru_maxrss * 1024)
    summary = {
        "cmd": cmd,
        "returncode": rc,
        "killed": reason,
        "wall_s": round(wall, 3),
        "peak_rss_gib": round(peak_reported / 2**30, 3),
        "peak_sampled_gib": round(peak / 2**30, 3),
        "cpu_s": round(cpu_s, 3),
        "cores": cores_spec,
        "threads": nthreads,
        "mem_limit_gib": mem_gb,
        "exclusive": slot,
    }
    line = json.dumps(summary, ensure_ascii=False)
    print(f"[guard] {line}", file=sys.stderr, flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(line + "\n")
    if lock_fd is not None:
        release_slot(lock_fd)
    if reason:
        return 137
    return rc


if __name__ == "__main__":
    sys.exit(main())
