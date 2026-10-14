"""tools/bench/factspeed_gate_flops.py: pooling of ab_cases run directories and flop normalisation."""
import json
from pathlib import Path

import pytest

from tools.bench import factspeed_gate_flops as gf


def make_run_dir(root: Path, runs: list[tuple[str, int, float, float, float, float, str]]) -> Path:
    """runs: (label, rep, time_total_s, numeric_s, time_certificate_s, flops, verdict)."""
    root.mkdir(parents=True)
    lines = []
    for label, rep, total, numeric, cert, flops, verdict in runs:
        d = root / "case1" / f"{label}-r{rep}"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({"factor_stats": {"estimate": {"flops": flops}}}))
        (d / "evaluation.json").write_text(json.dumps({"verdict": verdict, "completeness": {"status": "complete"}}))
        lines.append(json.dumps({"case": "case1", "label": label, "rep": rep, "host": "h", "binary": f"/b/{label}",
                                 "rc": 0, "killed": None, "success": True, "time_total_s": total,
                                 "numeric_s": numeric, "time_certificate_s": cert}))
    (root / "runs.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_equal_flops_speedup_removes_the_pivot_orders_luck(tmp_path):
    # Both binaries spend 10 s outside the factorizations; main factorizes at 3 s per Tflop, new at
    # 2 s per Tflop, but new drew cheaper orders (8 vs 10 Tflop): the raw speedup (40/26) overstates
    # the per-flop one; at the median flops of all runs (9 Tflop) it is (10 + 27) / (10 + 18).
    runs = [("main", r, 40.0, 10.0, 20.0, 10e12, "pass") for r in range(3)]
    runs += [("new", r, 26.0, 5.0, 11.0, 8e12, "pass") for r in range(3)]
    rd = make_run_dir(tmp_path / "s1", runs)
    out = tmp_path / "t.json"
    assert gf.main([str(rd), "--boot", "50", "--json", str(out), "--md", str(tmp_path / "t.md")]) == 0
    doc = json.loads(out.read_text())["cases"]["case1"]
    p = doc["pooled"]
    assert p["speedup"] == pytest.approx(40.0 / 26.0)
    assert p["rate_speedup"] == pytest.approx(3.0 / 2.0)
    assert doc["flops_ref"] == pytest.approx(9e12)
    assert p["equal_flops_speedup"] == pytest.approx((10 + 27) / (10 + 18))
    lo, hi = p["bootstrap"]["speedup"]
    assert lo == pytest.approx(40.0 / 26.0) and hi == pytest.approx(40.0 / 26.0)  # identical runs
    assert p["gate_pass_probability"] == 1.0  # every 5-run gate sees 40 / 26 >= 1.05
    gf.main([str(rd), "--boot", "50", "--bar", "1.6", "--json", str(out)])
    assert json.loads(out.read_text())["cases"]["case1"]["pooled"]["gate_pass_probability"] == 0.0


def test_gate_probability_counts_the_draws_that_reach_the_bar():
    base = [{"time": t} for t in (100.0, 100.0, 110.0, 110.0)]
    new = [{"time": 100.0}]
    # Median of 1 draw per binary: base 110 (ratio 1.1) or 100 (ratio 1.0) with probability 1/2 each.
    p = gf.gate_probability(base, new, reps=1, bar=1.05, n_boot=4000)
    assert p == pytest.approx(0.5, abs=0.05)


def test_samples_are_pooled_and_invalid_runs_dropped(tmp_path):
    s1 = make_run_dir(tmp_path / "s1", [("main", 0, 30.0, 5.0, 10.0, 5e12, "pass"),
                                        ("new", 0, 20.0, 4.0, 6.0, 5e12, "pass"),
                                        ("new", 1, 99.0, 4.0, 6.0, 5e12, "fail")])
    s2 = make_run_dir(tmp_path / "s2", [("main", 0, 34.0, 5.0, 10.0, 5e12, "pass"),
                                        ("new", 0, 24.0, 4.0, 6.0, 5e12, "pass")])
    data = gf.load([s1, s2])
    assert len(data["case1"]["main"]) == 2 and len(data["case1"]["new"]) == 2  # the failed run is dropped
    out = tmp_path / "t.json"
    gf.main([str(s1), str(s2), "--boot", "20", "--json", str(out)])
    doc = json.loads(out.read_text())["cases"]["case1"]
    assert set(doc["samples"]) == {str(s1), str(s2)}
    assert doc["samples"][str(s1)]["speedup"] == pytest.approx(30.0 / 20.0)
    assert doc["pooled"]["speedup"] == pytest.approx(32.0 / 22.0)
