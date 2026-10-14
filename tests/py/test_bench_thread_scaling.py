"""tools/bench/thread_scaling.py: the phase split of a report and the speedup / efficiency table over
a raw run directory (synthetic runs; the run itself needs the measurement slot)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench import thread_scaling as ts


def report(setup: float, factor: float, iterate: float, verify: float, cert: float, analysis: float,
           factorize: float, solves: float, ordering: int = 10) -> dict:
    return {"time_total_s": setup + factor + iterate + verify + cert, "time_setup_s": setup,
            "time_factor_s": factor, "time_iterate_s": iterate, "time_verify_s": verify,
            "time_certificate_s": cert, "n": 1000, "problem": "standard", "success": True, "block_size": 4,
            "factor_options": {"ordering": ordering},
            "factor_stats": {"analyze_s": analysis, "factorize_s": factorize, "solve_s": solves}}


def test_phases_are_exclusive_and_split_the_total(tmp_path: Path):
    # Certificate counts on the main object: factor_stats.factorize_s (3.5) also holds them, the
    # split takes the factorization from the factor phase instead and never counts them twice.
    (tmp_path / "report.json").write_text(json.dumps(report(0.25, 5.0, 3.0, 0.25, 1.5, 1.0, 3.5, 2.5, 3)))
    p = ts.phases(tmp_path)
    assert p == pytest.approx({"analysis": 1.0, "numeric": 4.0, "solves": 2.5, "iterate_other": 0.5,
                               "certificate": 1.5, "rest": 0.5})
    assert sum(p.values()) == pytest.approx(10.0)
    assert ts.ordering(tmp_path) == "SCOTCH"
    # Analysis or solves beyond their phase are capped by it.
    (tmp_path / "report.json").write_text(json.dumps(report(0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.5, 3.0)))
    p = ts.phases(tmp_path)
    assert p["analysis"] == 1.0 and p["numeric"] == 0.0 and p["solves"] == 1.0 and p["iterate_other"] == 0.0


def write_run(rd: Path, case: str, t: int, rep: int, total: float, verdict: str = "pass") -> dict:
    out = rd / case / f"t{t}-r{rep}"
    out.mkdir(parents=True)
    (out / "evaluation.json").write_text(json.dumps({"verdict": verdict, "completeness": {"status": "complete"}}))
    rec = {"case": case, "label": f"t{t}", "threads": t, "rep": rep, "host": "h", "binary": "b", "rc": 0,
           "killed": None, "wall_s": total + 1, "peak_rss_gib": 1.0, "success": True, "time_total_s": total,
           "n": 1000, "problem": "standard", "block_size": 4, "certificate_method": "exact",
           "phases": {"analysis": 0.1 * total, "numeric": 0.5 * total, "solves": 0.2 * total,
                      "iterate_other": 0.0, "certificate": 0.2 * total, "rest": 0.0}}
    with open(rd / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def test_table_speedup_and_efficiency(tmp_path: Path):
    rd = tmp_path / "T"
    rd.mkdir()
    for rep, scale in enumerate((1.0, 1.1, 0.9)):
        write_run(rd, "c", 1, rep, 8.0 * scale)
        write_run(rd, "c", 4, rep, 2.5 * scale)
        write_run(rd, "c", 16, rep, 1.0 * scale)
    write_run(rd, "d", 2, 0, 4.0)
    write_run(rd, "d", 8, 0, 2.0, verdict="fail")    # an invalid run: no speedup for that row
    out = tmp_path / "t.json"
    assert ts.main(["table", str(rd), "--json", str(out), "--md", str(tmp_path / "t.md")]) == 0
    doc = json.loads(out.read_text())
    c = {r["threads"]: r for r in doc["cases"][0]["rows"]}
    assert c[1]["speedup"] == pytest.approx(1.0) and c[4]["speedup"] == pytest.approx(3.2)
    assert c[4]["efficiency"] == pytest.approx(0.8) and c[16]["efficiency"] == pytest.approx(0.5)
    assert c[16]["phase_speedup"]["numeric"] == pytest.approx(8.0)
    d = {r["threads"]: r for r in doc["cases"][1]["rows"]}
    assert d[2]["speedup"] == pytest.approx(1.0)       # the smallest measured thread count is the base
    assert d[8]["speedup"] is None and d[8]["valid"] == 0
    assert "| 16 | 1.000 [0.900-1.100] | 8.00x | 0.50 |" in (tmp_path / "t.md").read_text()


def test_table_rereads_the_reports(tmp_path: Path):
    rd = tmp_path / "T"
    rd.mkdir()
    for t, factor in ((1, 4.0), (2, 2.0)):
        write_run(rd, "c", t, 0, factor + 1.0)
        (rd / "c" / f"t{t}-r0" / "report.json").write_text(
            json.dumps(report(0.5, factor, 0.5, 0.0, 0.0, 1.0, 9.9, 0.25, 0 if t == 1 else 10)))
    out = tmp_path / "t.json"
    assert ts.main(["table", str(rd), "--json", str(out)]) == 0
    rows = json.loads(out.read_text())["cases"][0]["rows"]
    assert [r["ordering"] for r in rows] == ["AMD", "ND"]
    assert rows[1]["phases"]["numeric"] == pytest.approx(1.0)       # from the report, not runs.jsonl
    assert rows[1]["phase_speedup"]["numeric"] == pytest.approx(3.0)
