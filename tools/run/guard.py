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
            spikes between samples are not missed.
  * Time:   optional wall-clock timeout.
  * Mutual exclusion: ``--exclusive`` takes a machine-wide flock so that
            heavy experiments run one at a time.

A JSON summary (exit status, wall time, peak RSS, kill reason, cores) is
printed to stderr as the last line and optionally written to ``--summary``.

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CORES = os.environ.get("HS_CORES", "24-30,33-41")
DEFAULT_MEM_GB = float(os.environ.get("HS_MEM_GB", "30"))
DEFAULT_MIN_AVAIL_GB = float(os.environ.get("HS_MIN_AVAIL_GB", "12"))
# During parallel multi-agent phases set HS_MAX_SHARED_MEM_GB (e.g. 8): larger --mem-gb then
# requires --exclusive, so heavy runs never overlap.
MAX_SHARED_MEM_GB = float(os.environ.get("HS_MAX_SHARED_MEM_GB", "inf"))
LOCK_PATH = os.environ.get("HS_LOCK", "/tmp/hsolve-guard-exclusive.lock")
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
    ap.add_argument("--cores", default=DEFAULT_CORES, help="cpulist to pin to (default: %(default)s)")
    ap.add_argument("--threads", type=int, default=0, help="threads to advertise via env (default: #cores)")
    ap.add_argument("--mem-gb", type=float, default=DEFAULT_MEM_GB, help="RSS limit of the process tree in GiB")
    ap.add_argument("--min-avail-gb", type=float, default=DEFAULT_MIN_AVAIL_GB,
                    help="kill if system MemAvailable falls below this (GiB)")
    ap.add_argument("--timeout", type=float, default=0, help="wall-clock limit in seconds (0 = none)")
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval in seconds")
    ap.add_argument("--exclusive", action="store_true", help="serialize with other --exclusive runs")
    ap.add_argument("--skip-busy-check", action="store_true", help="do not abort when pinned cores are busy")
    ap.add_argument("--nice", type=int, default=5)
    ap.add_argument("--summary", type=Path, help="also write the JSON summary to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run (after --)")
    args = ap.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("missing command")

    if args.mem_gb > MAX_SHARED_MEM_GB and not args.exclusive:
        ap.error(f"--mem-gb {args.mem_gb} > HS_MAX_SHARED_MEM_GB={MAX_SHARED_MEM_GB}: add --exclusive")

    cpus = parse_cpulist(args.cores)
    allowed = os.sched_getaffinity(0)
    if not set(cpus) <= allowed:
        ap.error(f"cores {sorted(set(cpus) - allowed)} are not available to this process")
    nthreads = args.threads or len(cpus)

    lock_fd = None
    if args.exclusive:
        lock_fd = open(LOCK_PATH, "w")
        print(f"[guard] waiting for exclusive lock {LOCK_PATH}", file=sys.stderr, flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

    if not args.skip_busy_check:
        busy = busy_cores(cpus)
        if len(busy) > max(1, len(cpus) // 8):
            print(f"[guard] refusing to start: pinned cores busy {busy}; "
                  f"choose other cores with --cores", file=sys.stderr)
            return 3

    avail = mem_available_gb()
    if avail - args.mem_gb < args.min_avail_gb:
        print(f"[guard] note: MemAvailable={avail:.1f} GiB; the run will be killed if it drops "
              f"below {args.min_avail_gb} GiB", file=sys.stderr)

    env = dict(os.environ)
    for var in THREAD_VARS:
        env[var] = str(nthreads)
    env["HS_GUARDED"] = "1"

    def preexec() -> None:
        os.setpgrp()
        os.sched_setaffinity(0, cpus)
        os.nice(args.nice)

    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, preexec_fn=preexec)
    peak = 0
    peak_hwm = 0
    reason = None
    big_tree = max(2.0, args.mem_gb / 4) * 2**30
    try:
        while proc.poll() is None:
            rss, hwm = tree_rss_bytes(proc.pid)
            peak = max(peak, rss)
            peak_hwm = max(peak_hwm, hwm)
            if rss > args.mem_gb * 2**30:
                reason = f"rss {rss / 2**30:.2f} GiB > limit {args.mem_gb} GiB"
            elif rss >= big_tree and mem_available_gb() < args.min_avail_gb:
                reason = f"system MemAvailable < {args.min_avail_gb} GiB while tree uses {rss / 2**30:.1f} GiB"
            elif args.timeout and time.time() - t0 > args.timeout:
                reason = f"timeout {args.timeout}s"
            if reason:
                print(f"[guard] killing process tree: {reason}", file=sys.stderr, flush=True)
                kill_tree(proc)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        reason = "interrupted"
        kill_tree(proc)
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
        "cores": args.cores,
        "threads": nthreads,
        "mem_limit_gib": args.mem_gb,
    }
    line = json.dumps(summary, ensure_ascii=False)
    print(f"[guard] {line}", file=sys.stderr, flush=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(line + "\n")
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    if reason:
        return 137
    return rc


if __name__ == "__main__":
    sys.exit(main())
