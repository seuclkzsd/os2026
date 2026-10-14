#!/usr/bin/env python3
"""Markdown tables from bench_gpu_dgemm's JSON lines (P10 "gpu", results/p10_gpu.md section 2).

  build/release/bench_gpu_dgemm gemm ... > gemm.jsonl
  python3 tools/bench/gpu_dgemm_table.py gemm.jsonl [copy.jsonl devgemm.jsonl ...]

gemm lines: one row per shape, one column per mode: median ms and speedup over the CPU BLAS
(cpu mode), plus the offload model's predicted speedup of the default engine; copy lines: MB/s per
transfer kind and size; devgemm lines: cuBLAS GFlop/s per shape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    return out


def gemm_table(rows: list[dict]) -> str:
    modes: list[str] = []
    shapes: list[tuple] = []
    cell: dict[tuple, dict] = {}
    for r in rows:
        sh = (r["m"], r["n"], r["k"], r.get("ta", "N"), r.get("tb", "N"))
        if sh not in shapes:
            shapes.append(sh)
        if r["mode"] not in modes:
            modes.append(r["mode"])
        cell[(sh, r["mode"])] = r
    head = "| m x n x k | " + " | ".join(modes) + " | model (default) |"
    out = [head, "|" + "---|" * (len(modes) + 2)]
    for sh in shapes:
        cells = []
        model = None
        for m in modes:
            r = cell.get((sh, m))
            if r is None:
                cells.append("-")
                continue
            if m == "cpu":
                cells.append(f"{r['s'] * 1e3:.1f} ms ({r['gflops']:.0f} GF/s)")
            else:
                cells.append(f"{r['s'] * 1e3:.1f} ms ({r['speedup']:.2f}x)")
                if m == "gpu":
                    model = r.get("model_speedup")
        trans = "" if sh[3:] == ("N", "N") else f" {sh[3]}{sh[4]}"
        model_cell = f"{model:.2f}x" if model is not None else "-"
        out.append(f"| {sh[0]} x {sh[1]} x {sh[2]}{trans} | " + " | ".join(cells) + f" | {model_cell} |")
    return "\n".join(out)


def copy_table(rows: list[dict]) -> str:
    keys = ["memcpy_pageable_to_pinned", "h2d_pinned", "d2h_pinned", "bidir_pinned_each", "h2d_pageable",
            "d2h_pageable", "h2d_pageable_2d_strided", "h2d_registered", "register_unregister"]
    out = ["| MB | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for r in rows:
        out.append(f"| {r['mb']} | " + " | ".join(f"{r.get(k, float('nan')) / 1e3:.1f}" for k in keys) + " |")
    return "\n".join(out) + "\n(GB/s)"


def devgemm_table(rows: list[dict]) -> str:
    out = ["| m x n x k | ms | GFlop/s |", "|---|---|---|"]
    for r in rows:
        out.append(f"| {r['m']} x {r['n']} x {r['k']} | {r['s'] * 1e3:.2f} | {r['gflops']:.0f} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    rows = load(argv)
    parts = []
    for what, fn in (("copy", copy_table), ("devgemm", devgemm_table), ("gemm", gemm_table)):
        sel = [r for r in rows if r.get("what") == what]
        if sel:
            parts.append(fn(sel))
    print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
