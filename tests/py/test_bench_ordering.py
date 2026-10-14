"""Tests of the P9 ordering tools (tools/bench): the geometric dissection references
(ordering_geometric.py) and the merge of the cost model's fit data (ordering_fit_merge.py)."""
import json
import sys

import numpy as np
import pytest
import scipy.sparse as sp

from tools.bench import ordering_fit_merge, ordering_geometric as geo
from tools.gen import yee


def _order(x, leaf):
    o = geo.dissect(x, leaf)
    assert o.size == x.shape[0] and np.unique(o).size == x.shape[0]
    return o


def test_lattice_order_is_a_permutation_with_the_middle_plane_last():
    x = geo.lattice_coords(6, 5, 4)
    o = _order(x, 8)
    # The longest side (x, 6 points) is cut first; its plane (5 x 4 points) is eliminated last.
    last = x[o[-20:]]
    assert np.all(last[:, 0] == last[0, 0])
    assert 0 < last[0, 0] < 10


@pytest.mark.parametrize("cells", [(4, 3, 3), (5, 4, 3)])
def test_yee_planes_separate_the_curl_curl_graph(cells):
    grid = yee.YeeGrid(cells)
    a, _ = yee.assemble(grid)
    pattern = sp.coo_matrix(a)
    x = geo.yee_coords(*cells)
    assert x.shape[0] == a.shape[0] == grid.edge_count
    # Every even plane perpendicular to every axis separates the unknowns on its two sides.
    for d in range(3):
        for plane in range(2, 2 * cells[d] - 1, 2):
            side = np.sign(x[:, d] - plane)
            cross = (side[pattern.row] * side[pattern.col]) < 0
            assert not np.any(cross), (d, plane)
    _order(x, 4)


def test_main_writes_perm_in(tmp_path):
    out = tmp_path / "p.npy"
    assert geo.main(["yee", "3", "3", "3", str(out), "4"]) == 0
    perm = np.load(out)
    n = yee.YeeGrid((3, 3, 3)).edge_count
    assert perm.dtype == np.int32 and sorted(perm.tolist()) == list(range(1, n + 1))
    assert geo.main(["bad"]) == 2


def _line(case, ordering, flops):
    return json.dumps({"case": case, "ordering": ordering, "est_flops": flops})


def test_fit_merge_replaces_the_nested_dissection_rows(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    for name in ("ord16", "ord4"):
        (base / f"{name}.jsonl").write_text("\n".join([
            _line("a", 0, 1.0), _line("a", 10, 2.0),
            json.dumps({"returncode": 0, "bench": json.loads(_line("b", 10, 3.0))}),
            json.dumps({"returncode": 1, "bench": json.loads(_line("b", 3, 9.0))})]) + "\n")
    nd1, nd2, extra = tmp_path / "nd1", tmp_path / "nd2", tmp_path / "extra"
    for d in (nd1, nd2, extra):
        d.mkdir()
    (nd1 / "o16_a_10.json").write_text(_line("a", 10, 20.0) + "\n")
    (nd2 / "o16_a_10.json").write_text(_line("a", 10, 99.0) + "\n")  # nd1 wins for a
    (nd2 / "o16_b_10.json").write_text(_line("b", 10, 30.0) + "\n")
    (nd2 / "o4_b_10_r1.json").write_text(_line("b", 10, 31.0) + "\n")
    (extra / "o16_c_3.json").write_text(_line("c", 3, 5.0) + "\n")
    (extra / "o16_c_10.json").write_text(_line("c", 10, 6.0) + "\n")  # ND of --extra: ignored
    case = tmp_path / "cases" / "a"
    case.mkdir(parents=True)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--base", str(base), "--nd", str(nd1), "--nd", str(nd2),
                                      "--extra", str(extra), "--cases", str(case), "--out", str(out)])
    assert ordering_fit_merge.main() == 0
    r16 = [json.loads(l) for l in (out / "ord16.jsonl").read_text().splitlines()]
    assert sorted((r["case"], r["ordering"], r["est_flops"]) for r in r16) == [
        ("a", 0, 1.0), ("a", 10, 20.0), ("b", 10, 30.0), ("c", 3, 5.0)]
    r4 = [json.loads(l) for l in (out / "ord4.jsonl").read_text().splitlines()]
    assert sorted((r["case"], r["ordering"], r["est_flops"]) for r in r4) == [("a", 0, 1.0), ("b", 10, 31.0)]
    assert (out / "cases" / "a").is_symlink()
