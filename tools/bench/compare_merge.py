#!/usr/bin/env python3
"""Merge compare.py documents of one campaign split over several nodes into one table.

  python3 tools/bench/compare_merge.py --tag cmp-p5c-best --out-dir results \\
      results/raw/cmp-p5c-a.json@gpu01 results/raw/cmp-p5c-b.json@c003 ...

Every part must come from the same comparison (cores, solvers, SLEPc profile and ranks,
repetitions, --target): only the case rows are concatenated, in the order of the arguments. The
merged document keeps compare.py's layout (<tag>.json, and <tag>.md rendered by
compare.render_markdown) and adds "parts": the source tag, node (the optional @HOST suffix) and cases
of each part; the markdown ends with that list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.bench.compare import render_markdown  # noqa: E402

SAME = ("cores", "ncores", "solvers", "slepc_profile", "slepc_ranks", "reps", "target_override", "evaluated")


def merge(parts: list[tuple[dict, str | None]], tag: str) -> dict:
    """One compare.py document from several parts (each: document, node or None)."""
    first = parts[0][0]
    for doc, _ in parts[1:]:
        diff = [k for k in SAME if doc.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"compare_merge: {doc['tag']} differs from {first['tag']} in {', '.join(diff)}")
    seen: set[str] = set()
    rows = []
    for doc, _ in parts:
        for r in doc["rows"]:
            if r["case"] in seen:
                raise SystemExit(f"compare_merge: case {r['case']} appears in two parts")
            seen.add(r["case"])
            rows.append(r)
    return {"tag": tag, **{k: first.get(k) for k in SAME}, "rows": rows,
            "parts": [{"tag": d["tag"], "host": h, "cases": [r["case"] for r in d["rows"]]} for d, h in parts]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", help="compare.py JSON, optionally with @HOST")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    a = ap.parse_args(argv)
    parts = []
    for p in a.parts:
        path, _, host = p.partition("@")
        parts.append((json.loads(Path(path).read_text()), host or None))
    doc = merge(parts, a.tag)
    md = render_markdown(doc) + "\nParts (one compare.py call each; all solvers of a case ran on the same node): " + \
        "; ".join(f"{p['tag']} on {p['host'] or '?'}: {', '.join(p['cases'])}" for p in doc["parts"]) + ".\n"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"{a.tag}.json").write_text(json.dumps(doc, indent=1) + "\n")
    (a.out_dir / f"{a.tag}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
