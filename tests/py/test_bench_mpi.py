"""Tests of the P9 "mpi" measurement tools (tools/bench): the K(sigma) converter of
bench/mpi_factor_bench.c (mpi_dump_kernel.py: hsolve's pencil pattern and values, the file format,
the exact inertia from a complete reference), the layout tables (mpi_factor_table.py: medians,
ratios, the inertia verdict) and the per-phase split of a compare.py campaign (compare_phases.py:
only the runs compare.py counts valid). Synthetic inputs only; no MPI or MUMPS needed."""
import json

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from tools import case as cio
from tools.bench import compare_phases, mpi_dump_kernel, mpi_factor_table


def _dense_upper(U: sp.coo_matrix) -> np.ndarray:
    return np.triu(U.toarray())


def _pattern(U: sp.coo_matrix) -> set:
    return set(zip(U.row.tolist(), U.col.tolist()))


# ------------------------------------------------------------------------------- mpi_dump_kernel

def test_kernel_upper_is_the_union_pattern_of_disjoint_a_and_b():
    # A: diagonal (no (3,3)) + (0,1); B: diagonal (no (3,3)) + (2,3): disjoint off-diagonal patterns.
    A = sp.csr_matrix(np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 5.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0] * 4]))
    B = sp.csr_matrix(np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.5],
                                [0.0, 0.0, 0.5, 0.0]]))
    sigma = 0.75
    U = mpi_dump_kernel.kernel_upper(A, B, sigma)
    np.testing.assert_array_equal(_dense_upper(U), np.triu(A.toarray() - sigma * B.toarray()))
    # upper(A) union upper(B) union the whole diagonal: 4 diagonal + (0,1) + (2,3); (3,3) is in the
    # pattern although neither A nor B stores it (hsolve's pencil always holds the diagonal)
    assert _pattern(U) == {(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (2, 3)} and U.nnz == 6
    assert np.all(U.row <= U.col)
    # sorted by column, then row
    assert list(zip(U.col.tolist(), U.row.tolist())) == sorted(zip(U.col.tolist(), U.row.tolist()))


def test_kernel_upper_keeps_cancellations_and_drops_stored_zeros():
    # a_01 = sigma * b_01 exactly: K_01 = 0 stays an entry (hsolve's pattern is structural);
    # the explicitly stored zero a_02 / a_20 is dropped, as hsolve's loader drops exact zeros.
    A = sp.csr_matrix((np.array([3.0, 0.5, 0.0, 0.5, 3.0, 0.0, 3.0]),
                       (np.array([0, 0, 0, 1, 1, 2, 2]), np.array([0, 1, 2, 0, 1, 0, 2]))), shape=(3, 3))
    assert A.nnz == 7
    B = sp.csr_matrix(np.array([[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    U = mpi_dump_kernel.kernel_upper(A, B, 2.0)
    assert _pattern(U) == {(0, 0), (1, 1), (2, 2), (0, 1)}
    K = U.tocsr()
    assert K[0, 1] == 0.0 and K[0, 0] == 1.0 and K[2, 2] == 1.0


def test_kernel_upper_standard_problem_is_a_minus_sigma_identity():
    A = sp.csr_matrix(np.array([[2.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 2.0]]))   # a_11 not stored
    assert A.nnz == 6
    U = mpi_dump_kernel.kernel_upper(A, None, 1.5)
    np.testing.assert_array_equal(_dense_upper(U), np.triu(A.toarray() - 1.5 * np.eye(3)))
    assert (1, 1) in _pattern(U) and U.nnz == 5
    U0 = mpi_dump_kernel.kernel_upper(A, None, 0.0)          # sigma = 0: A itself, diagonal included
    np.testing.assert_array_equal(_dense_upper(U0), np.triu(A.toarray()))
    assert U0.nnz == 5 and U0.tocsr()[1, 1] == 0.0


def test_kernel_upper_sums_duplicates_and_takes_the_symmetric_part():
    eps = 2.0**-52
    # COO with a duplicated (0,0); a_01 = 1, a_10 = 1 + eps: not exactly symmetric, so hsolve
    # factorizes (A + A^T) / 2; a lower-only entry (2,1) appears mirrored at (1,2) with half its value.
    A = sp.coo_matrix((np.array([1.0, 2.0, 1.0, 1.0 + eps, 4.0, 8.0, 5.0]),
                       (np.array([0, 0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 0, 1, 1, 2]))), shape=(3, 3))
    U = mpi_dump_kernel.kernel_upper(A, None, 0.0).tocsr()
    assert U[0, 0] == 3.0 and U[1, 1] == 4.0 and U[2, 2] == 5.0
    assert U[0, 1] == (1.0 + (1.0 + eps)) / 2 and U[1, 2] == 4.0
    assert U.nnz == 5


def test_kernel_upper_rejects_complex_and_mismatched_inputs():
    with pytest.raises(SystemExit, match="complex"):
        mpi_dump_kernel.kernel_upper(sp.identity(2, dtype=complex, format="csr"), None, 0.0)
    with pytest.raises(SystemExit, match="B is"):
        mpi_dump_kernel.kernel_upper(sp.identity(2, format="csr"), sp.identity(3, format="csr"), 0.0)


def test_write_and_read_kernel_round_trip(tmp_path):
    A = sp.csr_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]))
    U = mpi_dump_kernel.kernel_upper(A, None, 1.0)
    path = tmp_path / "K.bin"
    mpi_dump_kernel.write_kernel(path, U)
    raw = path.read_bytes()
    assert raw[:8] == b"HSKMAT1\0" and len(raw) == 24 + 16 * U.nnz
    n, nnz = np.frombuffer(raw, dtype="<i8", count=2, offset=8)
    assert (n, nnz) == (3, 5)
    irn = np.frombuffer(raw, dtype="<i4", count=nnz, offset=24)
    jcn = np.frombuffer(raw, dtype="<i4", count=nnz, offset=24 + 4 * nnz)
    assert irn.min() == 1 and jcn.max() == 3 and np.all(irn <= jcn)      # 1-based, upper triangle
    assert list(zip(jcn.tolist(), irn.tolist())) == [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    R = mpi_dump_kernel.read_kernel(path)
    np.testing.assert_array_equal(R.toarray(), U.toarray())
    with pytest.raises(ValueError, match="bytes"):
        (tmp_path / "short.bin").write_bytes(raw[:-8])
        mpi_dump_kernel.read_kernel(tmp_path / "short.bin")


def test_exact_inertia_counts_ties_and_needs_a_complete_reference():
    ref = cio.Reference(np.array([-1.0, 0.5, 0.5, 2.0]), None, {"complete": True})
    assert mpi_dump_kernel.exact_inertia(ref, 0.5, 4, "HEP") == {"neg": 1, "zero": 2, "pos": 1}
    assert mpi_dump_kernel.exact_inertia(ref, 0.0, 4, "GHEP") == {"neg": 1, "zero": 0, "pos": 3}
    assert mpi_dump_kernel.exact_inertia(ref, 0.5, 4, "GHIEP") is None       # Sylvester does not apply
    assert mpi_dump_kernel.exact_inertia(ref, 0.5, 5, "HEP") is None         # not the whole spectrum
    assert mpi_dump_kernel.exact_inertia(cio.Reference(ref.eigenvalues, None, {"complete": False}),
                                         0.5, 4, "HEP") is None
    assert mpi_dump_kernel.exact_inertia(None, 0.5, 4, "HEP") is None


def test_dump_kernel_main_writes_matrix_and_inertia(tmp_path):
    rng = np.random.default_rng(7)
    n = 12
    A = sp.diags([np.full(n - 1, -1.0), np.linspace(-2.0, 3.0, n), np.full(n - 1, -1.0)], [-1, 0, 1], format="csr")
    B = sp.diags([np.full(n - 1, 0.1), 1.0 + rng.random(n), np.full(n - 1, 0.1)], [-1, 0, 1], format="csr")
    sigma = 0.3
    cio.save_case("tiny-ghep", A, B, meta={"family": "test", "problem": "GHEP"},
                  config=cio.default_config(sigma, 2), root=tmp_path)
    w =scipy.linalg.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    cio.save_ref("tiny-ghep", w, kind="dense", complete=True, root=tmp_path)
    out = tmp_path / "K.bin"
    assert mpi_dump_kernel.main(["--case", "tiny-ghep", "--root", str(tmp_path), "--out", str(out)]) == 0
    info = json.loads((tmp_path / "K.bin.json").read_text())
    K = mpi_dump_kernel.read_kernel(out).toarray()
    np.testing.assert_allclose(K, np.triu(A.toarray() - sigma * B.toarray()), rtol=0, atol=1e-15)
    assert info["n"] == n and info["nnz_upper"] == 2 * n - 1 and info["sigma"] == sigma
    full = K + np.triu(K, 1).T
    neg = int(np.sum(np.linalg.eigvalsh(full) < 0))
    assert info["inertia"] == {"neg": neg, "zero": 0, "pos": n - neg} and 0 < neg < n


# ------------------------------------------------------------------------------ mpi_factor_table

def _layout_run(layout, rep, count_s, full_s, solve_s, neg=7, expect=7, ok=True, rc=0, ranks=1, run_event=True,
                order="perm"):
    """The JSON lines of one tools/bench/mpi_factor_runs.sh run (bench events + guard line)."""
    tag = {"layout": layout, "rep": rep, "order": order}
    lines = [{"event": "analysis", "mode": "count", "s": 0.5, "flops_est": 1e9, **tag},
             {"event": "factorize", "mode": "count", "s": count_s, "neg": neg, "mem_total_mb": 50, **tag},
             {"event": "analysis", "mode": "full", "s": 0.6, "flops_est": 1e9, **tag},
             {"event": "factorize", "mode": "full", "s": full_s, "neg": neg, "flops": 2e9, "mem_total_mb": 100,
              "mem_max_mb": 100 // ranks, "entries": 1e6, "ranks": ranks, "rank_flops_max": 2e9 / ranks * 1.1,
              "rank_flops_sum": 2e9, **tag},
             {"event": "solve", "mode": "full", "s": solve_s, "nrhs": 16, "rel_residual": 1e-15, **tag}]
    if run_event:
        lines.append({"event": "run", "ranks": ranks, "threads": 16 // ranks, "expect_neg": expect,
                      "inertia_ok": ok, **tag})
    lines.append({"event": "guard", "returncode": rc, "killed": None, "wall_s": 9.0, "peak_rss_gib": 1.0 + rep,
                  **tag})
    return lines


def test_factor_table_medians_ratios_and_inertia():
    lines = []
    for rep, (c, f, s) in enumerate([(10.0, 12.0, 1.0), (11.0, 13.0, 1.2), (30.0, 11.0, 0.8)]):
        lines += _layout_run("seq1x16", rep, c, f, s)
    for rep, (c, f, s) in enumerate([(20.0, 24.0, 2.0), (22.0, 26.0, 2.4), (21.0, 25.0, 2.2)]):
        lines += _layout_run("2x8", rep, c, f, s, ranks=2)
    summ = mpi_factor_table.summarize(lines)
    seq, l28 = summ["perm"]["seq1x16"], summ["perm"]["2x8"]
    assert (seq["count_s"], seq["factor_s"], seq["solve_s"]) == (11.0, 12.0, 1.0)
    assert (l28["count_s"], l28["factor_s"], l28["solve_s"]) == (21.0, 25.0, 2.2)
    assert seq["runs"] == 3 and seq["failed_runs"] == 0 and seq["peak_gib"] == 3.0
    assert l28["gflops_per_s"] == pytest.approx(2e9 / 25.0 / 1e9) and l28["imbalance"] == pytest.approx(1.1)
    assert seq["inertia"] == "ok" and seq["inertia_ok"] and l28["neg"] == [7]
    md = mpi_factor_table.table("t", "perm", summ["perm"], "seq1x16")
    row = next(r for r in md.splitlines() if r.startswith("| 2x8 "))
    assert "| 21.00 | 0.52 | 25.00 | 0.48 |" in row and "| 2.200 (16) | 0.45 |" in row and row.endswith("7 ok |")
    assert md.splitlines()[4].startswith("| seq1x16 | 3/3 |")      # the base layout comes first


def test_factor_table_inertia_verdicts():
    base = _layout_run("seq1x16", 0, 1.0, 1.0, 0.1) + _layout_run("seq1x16", 1, 1.0, 1.0, 0.1)
    assert mpi_factor_table.summarize(base)["perm"]["seq1x16"]["inertia"] == "ok"
    # a run that ended without its "run" event (crashed): its inertia is unknown
    crashed = _layout_run("seq1x16", 0, 1.0, 1.0, 0.1) + _layout_run("seq1x16", 1, 1.0, 1.0, 0.1, rc=-9,
                                                                       run_event=False)
    r = mpi_factor_table.summarize(crashed)["perm"]["seq1x16"]
    assert r["inertia"] == "MISMATCH" and not r["inertia_ok"] and (r["runs"], r["failed_runs"]) == (2, 1)
    # a count off the exact one, even where the run event claims success
    off = _layout_run("4x4", 0, 1.0, 1.0, 0.1, ranks=4) + _layout_run("4x4", 1, 1.0, 1.0, 0.1, ranks=4, neg=6)
    r = mpi_factor_table.summarize(off)["perm"]["4x4"]
    assert r["inertia"] == "MISMATCH" and r["neg"] == [6, 7]
    # the bench's own failed check (exit status 1)
    failed = _layout_run("2x8", 0, 1.0, 1.0, 0.1, ranks=2, neg=5, ok=False, rc=1)
    assert mpi_factor_table.summarize(failed)["perm"]["2x8"]["inertia"] == "MISMATCH"
    # without --expect-neg the counts are only compared with each other
    free = _layout_run("1x16", 0, 1.0, 1.0, 0.1, expect=-1) + _layout_run("1x16", 1, 1.0, 1.0, 0.1, expect=-1)
    assert mpi_factor_table.summarize(free)["perm"]["1x16"]["inertia"] == "unchecked"
    free[1]["neg"] = 8
    assert mpi_factor_table.summarize(free)["perm"]["1x16"]["inertia"] == "MISMATCH"
    # one table per order
    two = base + _layout_run("seq1x16", 0, 2.0, 2.0, 0.2, order="metis")
    assert set(mpi_factor_table.summarize(two)) == {"perm", "metis"}


def test_factor_table_reads_direct_bench_output():
    # lines of the bench run by hand: no layout / rep / order annotation and no guard lines
    direct = [{k: v for k, v in d.items() if k not in ("layout", "rep", "order")}
              for d in _layout_run("x", 0, 3.0, 4.0, 0.5) + _layout_run("x", 1, 3.0, 4.0, 0.5) if d["event"] != "guard"]
    lines = direct + _layout_run("seq1x16", 0, 3.0, 4.0, 0.5, order="?") + _layout_run("16x1", 0, 6.0, 8.0, 1.0,
                                                                                        ranks=16, order="?")
    summ = mpi_factor_table.summarize(lines)
    r = summ["?"]["?"]
    assert (r["runs"], r["failed_runs"], r["inertia"], r["factor_s"]) == (2, 0, "ok", 4.0)
    md = mpi_factor_table.table("t", "?", summ["?"], "seq1x16").splitlines()
    assert [row.split(" | ")[0] for row in md[4:]] == ["| seq1x16", "| 16x1", "| ?"]
    assert mpi_factor_table.inertia_verdict([], [], [], 0) == "-"
    assert mpi_factor_table.inertia_verdict([7], [], [], 0) == "MISMATCH"     # factorized, then no run event


# -------------------------------------------------------------------------------- compare_phases

def _campaign_run(case_dir, run, total, *, rc=0, killed=None, success=True, verdict="pass", guard=True):
    """One run directory as tools/bench/compare.py leaves it (hsolve report)."""
    d = case_dir / run
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({
        "success": success, "time_total_s": total, "time_factor_s": total / 2, "time_iterate_s": total / 4,
        "time_certificate_s": total / 4, "time_setup_s": 0.0, "time_verify_s": 0.0,
        "factor_stats": {"analyze_s": 0.1, "factorize_s": 0.2, "solve_s": 0.3, "rhs_solved": 16},
        "certificate": {"method": "exact"}}))
    if verdict is not None:
        (d / "evaluation.json").write_text(json.dumps({"verdict": verdict}))
    if guard:
        (case_dir / f"guard-{run}.json").write_text(json.dumps({"returncode": rc, "killed": killed}))


def test_compare_phases_counts_only_valid_runs(tmp_path):
    case_dir = tmp_path / "c1"
    _campaign_run(case_dir, "hsolve-r0", 10.0)
    _campaign_run(case_dir, "hsolve-r1", 12.0)
    _campaign_run(case_dir, "hsolve-r2", 1.0, verdict=None)          # tools/eval raised: no evaluation.json
    _campaign_run(case_dir, "hsolve-r3", 2.0, verdict="fail")
    _campaign_run(case_dir, "hsolve-r4", 3.0, rc=1)                  # exit status 1
    _campaign_run(case_dir, "hsolve-r5", 4.0, killed="memory")
    _campaign_run(case_dir, "hsolve-r6", 5.0, success=False)          # the solver claims no success
    _campaign_run(case_dir, "hsolve-r7", 6.0, guard=False)            # no guard summary
    _campaign_run(case_dir, "hsolve-auto-r0", 7.0)
    (case_dir / "slepc-r0").mkdir()                                   # no report.json at all
    doc = compare_phases.collect(tmp_path)
    hs = doc["c1"]["hsolve"]
    assert (hs["valid"], hs["runs"]) == (2, 8)
    assert hs["phases"]["time_total_s"] == [10.0, 12.0] and hs["phases"]["certificate_method"] == ["exact"] * 2
    assert doc["c1"]["hsolve-auto"]["valid"] == 1 and doc["c1"]["slepc"] == {"runs": 1, "valid": 0, "phases": {}}
    md = compare_phases.render(doc)
    assert "| phase | hsolve | hsolve-auto | slepc |" in md      # compare.py's order, not the directory order
    assert "| valid runs | 2/8 | 1/1 | 0/1 |" in md
    row = next(r for r in md.splitlines() if r.startswith("| time_total_s |"))
    assert row == "| time_total_s | 11.00 [10.00-12.00] | 7.00 | - |"
    assert compare_phases.main([str(tmp_path), "--json", str(tmp_path / "p.json")]) == 0
    assert json.loads((tmp_path / "p.json").read_text())["c1"]["hsolve"]["valid"] == 2
