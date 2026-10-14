"""Tests of the P10 "mpi" layout campaign tool (tools/bench/mpi_layouts.py): the layout names, the
bench table (one count, the pair of counts - the slower of two concurrent ones -, factorization,
solve; ratios to a base layout; a failed inertia check or run counts against "runs ok") and the
hsolve table (valid runs only, phases). Synthetic JSON lines; no MUMPS, MPI or guard needed."""
import json

import pytest

from tools.bench import mpi_layouts


def test_layouts_parse_and_reject():
    assert mpi_layouts.parse_layouts("one:1x16,two:2x8,two:1+2x8") == [("one", "1x16"), ("two", "2x8"),
                                                                         ("two", "1+2x8")]
    for bad in ("three:1x16", "one:4x4", "1x16"):
        with pytest.raises(SystemExit):
            mpi_layouts.parse_layouts(bad)


def _bench_rows(case, lay, rep, procs, count, factor, solve, ok=True):
    rows = []
    for p in range(procs):
        rows += [{"kind": "bench", "case": case, "layout": lay, "rep": rep, "proc": p, "event": "factorize",
                  "mode": "count", "s": count[p]},
                 {"kind": "bench", "case": case, "layout": lay, "rep": rep, "proc": p, "event": "factorize",
                  "mode": "full", "s": factor},
                 {"kind": "bench", "case": case, "layout": lay, "rep": rep, "proc": p, "event": "solve",
                  "mode": "full", "s": solve},
                 {"kind": "bench", "case": case, "layout": lay, "rep": rep, "proc": p, "event": "run",
                  "inertia_ok": ok}]
    rows.append({"kind": "bench", "case": case, "layout": lay, "rep": rep, "event": "guard", "returncode": 0,
                 "killed": None, "peak_rss_gib": 1.0})
    return rows


def _cells(lines, layout):
    row = next(l for l in lines if f"| {layout} |" in l)
    return [c.strip() for c in row.strip("|").split("|")]


def test_bench_table_pairs_and_ratios():
    rows = []
    for rep in range(3):
        rows += _bench_rows("c", "one:1x16", rep, 1, [10.0], 11.0, 1.0)
        rows += _bench_rows("c", "two:2x8", rep, 2, [6.0, 7.0], 12.0, 1.0)
    lines = mpi_layouts.bench_table(rows, "one:1x16")
    base, two = _cells(lines, "one:1x16"), _cells(lines, "two:2x8")
    assert base[2] == "3/3" and base[4].startswith("20.000")  # sequential pair = 2 x one count
    assert two[4].startswith("7.000") and two[5] == "2.86"     # concurrent pair = the slower count
    assert two[7] == "0.92"                                     # factorization 11 / 12


def test_bench_table_counts_a_wrong_inertia_against_the_layout():
    rows = _bench_rows("c", "one:1x16", 0, 1, [10.0], 11.0, 1.0)
    rows += _bench_rows("c", "two:2x8", 0, 2, [6.0, 7.0], 12.0, 1.0, ok=False)
    lines = mpi_layouts.bench_table(rows, "one:1x16")
    assert _cells(lines, "two:2x8")[2].startswith("1/")


def test_hsolve_table_uses_valid_runs_only():
    rows, verdicts = [], {}
    for rep, (t, ok) in enumerate([(10.0, True), (12.0, True), (1.0, False)]):
        out = f"/x/one-1x16-r{rep}"
        rows.append({"kind": "hsolve", "case": "c", "layout": "one:1x16", "rep": rep, "event": "guard", "out": out,
                     "returncode": 0, "killed": None, "peak_rss_gib": 2.0,
                     "report": {"time_total_s": t, "time_factor_s": 1.0, "time_iterate_s": 2.0,
                                "time_certificate_s": 3.0, "time_verify_s": 0.5, "time_setup_s": 0.5}})
        verdicts[out] = {"valid": ok}
        out2 = f"/x/two-1+2x8-r{rep}"
        rows.append({"kind": "hsolve", "case": "c", "layout": "two:1+2x8", "rep": rep, "event": "guard",
                     "out": out2, "returncode": 0, "killed": None, "peak_rss_gib": 3.0,
                     "report": {"time_total_s": 5.5, "time_factor_s": 1.0, "time_iterate_s": 2.0,
                                "time_certificate_s": 1.5, "time_verify_s": 0.5, "time_setup_s": 0.5}})
        verdicts[out2] = {"valid": True}
    lines = mpi_layouts.hsolve_table(rows, verdicts, "one:1x16")
    base, mpi = _cells(lines, "one:1x16"), _cells(lines, "two:1+2x8")
    assert base[2] == "2/3" and base[3].startswith("11.000")   # median of the two valid runs
    assert mpi[2] == "3/3" and mpi[4] == "2.00"                 # 11 / 5.5
    assert mpi[7] == "1.50"


def test_cli_table_reads_the_campaign_files(tmp_path, capsys):
    rows = _bench_rows("c", "one:1x16", 0, 1, [10.0], 11.0, 1.0)
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert mpi_layouts.main(["table", "--tag", "t", "--out", str(tmp_path), "--md", str(tmp_path / "t.md")]) == 0
    assert "one:1x16" in (tmp_path / "t.md").read_text()
