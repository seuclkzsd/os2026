#!/usr/bin/env python3
"""Symbolise and aggregate the SIGPROF samples of bench_factspeed_factor (P9 "factspeed").

    python3 tools/bench/factspeed_symbolize.py SAMPLES [--job JOB] [--t0 S] [--t1 S]
        [--top 40] [--timeline BIN_S] [--threads] [--json OUT]

SAMPLES: the --sample-out file of bench_factspeed_factor ("# obj" lines with the mapped objects and
their load bases, then one line per sample: job, microseconds since the job's start, thread id,
object index, offset). Each object is symbolised with `nm -n --defined-only -C` (its .symtab, else
.dynsym with -D); an offset falls into the last function symbol at or below it.

Output per job (optionally only samples in [t0, t1) seconds of the job):
  * shares by object, by category (CATEGORIES below: BLAS kernel / packing, MUMPS panel, assembly,
    copies, memset, OpenMP spin, ...) and the top symbols;
  * --timeline BIN_S: the category shares per time bin (what runs when: MUMPS factorizes the L0
    subtrees first, then the fronts above them);
  * --threads: samples per thread and the share of them in OpenMP spin (idle) per thread.
--json writes {job: {"objects", "categories", "symbols", "timeline"}}.
Run where the objects live (the cluster node), e.g. through env/run.sh. Samples are process CPU
time (user + system).
"""
import argparse
import bisect
import collections
import json
import os
import re
import subprocess
import sys

# (category, regex on the symbol name, regex on the object basename); first match wins.
CATEGORIES = [
    ("blas_kernel", r"(gemm|trsm|gemv|syrk|axpy|dot|scal)_(small_)?kernel|_kernel_", r"openblas"),
    ("blas_pack", r"copy|inner_thread|exec_threads|blas_thread|gemm_[nt][nt]$", r"openblas"),
    ("blas_other", r".", r"openblas|blashook"),
    ("spin", r"__kmp_wait|__kmp_hardware_timestamp|sched_yield|kmp_flag|__kmp_.*barrier|__kmp_yield", r"libomp|libgomp|libc"),
    ("omp_other", r".", r"libomp|libgomp"),
    ("panel", r"fac_mq_ldlt|fac_i_ldlt|minmax_pivot|swap_ldlt|fac_sq_ldlt|fac_t_ldlt|fac_n_|fac_mq", r"mumps"),
    ("assembly", r"asm|arrow|build_sort_index|mumps_sort|sorted_merge", r"mumps"),
    ("copies", r"copy|compact|perform_copies|cbstatic2dynamic|dm_|stack|ooc", r"mumps"),
    ("scaling", r"scal", r"mumps"),
    ("mumps_other", r".", r"mumps"),
    ("memset_memcpy", r"mem(set|cpy|move)", r"libc"),
    ("malloc", r"malloc|free|mmap|munmap|brk|sbrk", r"libc"),
    ("libc_other", r".", r"libc|ld-"),
]


def category(sym, obj):
    for name, srx, orx in CATEGORIES:
        if re.search(orx, obj) and re.search(srx, sym):
            return name
    return "other"


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
    ap.add_argument("--job", help="only this job")
    ap.add_argument("--t0", type=float, default=0.0, help="window start, seconds of the job")
    ap.add_argument("--t1", type=float, default=float("inf"), help="window end, seconds of the job")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--timeline", type=float, default=0.0, help="time bin in seconds (0: none)")
    ap.add_argument("--threads", action="store_true", help="per-thread samples and spin share")
    ap.add_argument("--json")
    args = ap.parse_args()

    objects = {}  # index -> (path, base)
    rows = []
    with open(args.samples) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "# obj":
                objects[int(parts[1])] = (parts[2], int(parts[3], 16))
                continue
            job, t_us, tid, idx, off = parts
            t = int(t_us) * 1e-6
            if (args.job and job != args.job) or not (args.t0 <= t < args.t1):
                continue
            rows.append((job, t, int(tid), int(idx), int(off, 16)))

    tables = {i: symbols(p) if os.path.exists(p) else [] for i, (p, _) in objects.items()}
    kinds = {i: elf_type(p) for i, (p, _) in objects.items()}
    cache = {}

    def resolve(idx, off):
        key = (idx, off)
        if key not in cache:
            if idx < 0:
                cache[key] = ("?", "?", "other")
            else:
                path, base = objects[idx]
                name = os.path.basename(path)
                table = tables[idx]
                addr = off + base if kinds[idx] == 2 else off
                i = bisect.bisect_right(table, (addr, "￿")) - 1
                sym = table[i][1] if i >= 0 else "%s+0x%x" % (name, addr & ~0xfff)
                cache[key] = (sym, name, category(sym, name))
        return cache[key]

    by_obj = collections.defaultdict(collections.Counter)
    by_sym = collections.defaultdict(collections.Counter)
    by_cat = collections.defaultdict(collections.Counter)
    timeline = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    per_thread = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for job, t, tid, idx, off in rows:
        sym, name, cat = resolve(idx, off)
        by_obj[job][name] += 1
        by_sym[job][(sym, name)] += 1
        by_cat[job][cat] += 1
        if args.timeline > 0:
            timeline[job][int(t / args.timeline)][cat] += 1
        if args.threads:
            per_thread[job][tid][cat] += 1

    cats = [c[0] for c in CATEGORIES] + ["other"]
    result = {}
    for job in by_obj:
        total = sum(by_obj[job].values())
        print("== job %s: %d samples (t in [%g, %g) s)" % (job, total, args.t0, args.t1))
        for name, c in by_obj[job].most_common():
            print("  %6.2f%%  %s" % (100.0 * c / total, name))
        print("  -- categories")
        for cat, c in by_cat[job].most_common():
            print("  %6.2f%%  %s" % (100.0 * c / total, cat))
        print("  -- top symbols")
        for (sym, name), c in by_sym[job].most_common(args.top):
            print("  %6.2f%%  %-62s %s [%s]" % (100.0 * c / total, sym[:62], name, category(sym, name)))
        tl = []
        if args.timeline > 0:
            used = [c for c in cats if by_cat[job][c] > 0]
            print("  -- timeline (bin %g s): samples, then %% per category: %s" % (args.timeline, " ".join(used)))
            for b in sorted(timeline[job]):
                cnt = timeline[job][b]
                n = sum(cnt.values())
                print("  %7.2f %6d  %s" % (b * args.timeline, n,
                                           " ".join("%5.1f" % (100.0 * cnt[c] / n) for c in used)))
                tl.append({"t": b * args.timeline, "samples": n, "categories": dict(cnt)})
        if args.threads:
            print("  -- threads: samples, %% spin, %% blas_kernel")
            for tid in sorted(per_thread[job], key=lambda k: -sum(per_thread[job][k].values())):
                cnt = per_thread[job][tid]
                n = sum(cnt.values())
                print("  %8d %7d %6.1f %6.1f" % (tid, n, 100.0 * cnt["spin"] / n, 100.0 * cnt["blas_kernel"] / n))
        result[job] = {"total": total, "objects": dict(by_obj[job]), "categories": dict(by_cat[job]),
                       "symbols": [[s, n, c] for (s, n), c in by_sym[job].most_common()], "timeline": tl}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
