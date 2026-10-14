#!/usr/bin/env python3
"""Symbolise the SIGPROF samples of bench_factspeed_factor (P9 "factspeed").

    python3 tools/bench/factspeed_symbolize.py SAMPLES [--top 40] [--json OUT] [--job JOB]

SAMPLES: the --sample-out file of bench_factspeed_factor, one line per (job, object, load base,
offset) with its count. Each object is symbolised with `nm -n --defined-only` (its .symtab, else
.dynsym with -D); an offset falls into the last symbol at or below it (an object without symbols
at that address reports "<object>+0xpage"). Prints per job the share of samples by object and the
top symbols; --json writes {job: {"objects": {...}, "symbols": [[symbol, object, count], ...]}}.
Run where the objects live (the cluster node), e.g. through env/run.sh, so that nm sees the same
files the bench loaded. Samples are process CPU time (user + system), see the bench header.
"""
import argparse
import bisect
import collections
import json
import os
import subprocess
import sys


def elf_type(path):
    """2 = ET_EXEC (absolute addresses), 3 = ET_DYN (offsets from the load base)."""
    try:
        with open(path, "rb") as f:
            head = f.read(18)
        return int.from_bytes(head[16:18], "little")
    except OSError:
        return 0


def symbols(path):
    """Sorted (address, name) of the defined function symbols of `path`."""
    out = []
    for extra in ([], ["-D"]):
        try:
            res = subprocess.run(["nm", "-n", "--defined-only", "-C"] + extra + [path],
                                 capture_output=True, text=True, check=False)
        except FileNotFoundError:
            sys.exit("nm not found (run through env/run.sh)")
        for line in res.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[1] in "tTwWiI":
                try:
                    out.append((int(parts[0], 16), parts[2]))
                except ValueError:
                    pass
        if out:
            break
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json")
    ap.add_argument("--job", help="only this job")
    args = ap.parse_args()

    rows = []
    with open(args.samples) as f:
        for line in f:
            job, obj, base, off, count = line.rstrip("\n").split("\t")
            if args.job and job != args.job:
                continue
            rows.append((job, obj, int(base, 16), int(off, 16), int(count)))

    tables = {}
    for obj in sorted({r[1] for r in rows}):
        tables[obj] = symbols(obj) if os.path.exists(obj) else []
    kinds = {obj: elf_type(obj) for obj in tables}

    by_sym = collections.defaultdict(collections.Counter)
    by_obj = collections.defaultdict(collections.Counter)
    for job, obj, base, off, count in rows:
        name = os.path.basename(obj)
        by_obj[job][name] += count
        table = tables[obj]
        addr = off + base if kinds[obj] == 2 else off
        i = bisect.bisect_right(table, (addr, "￿")) - 1
        sym = table[i][1] if i >= 0 else "%s+0x%x" % (name, addr & ~0xfff)
        by_sym[job][(sym, name)] += count

    result = {}
    for job in by_obj:
        total = sum(by_obj[job].values())
        print("== job %s: %d samples" % (job, total))
        for name, c in by_obj[job].most_common():
            print("  %6.2f%%  %s" % (100.0 * c / total, name))
        print("  -- top symbols")
        for (sym, name), c in by_sym[job].most_common(args.top):
            print("  %6.2f%%  %-60s %s" % (100.0 * c / total, sym[:60], name))
        result[job] = {"total": total, "objects": dict(by_obj[job]),
                       "symbols": [[s, n, c] for (s, n), c in by_sym[job].most_common()]}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
