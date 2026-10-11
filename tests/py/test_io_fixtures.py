"""The committed src/io fixtures (tests/data/io) are what tests/data/make_io_fixtures.py produces,
and scipy reads them as expected.json claims (an independent check of the C++ test oracle)."""
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "data" / "io"
SCRIPT = ROOT / "tests" / "data" / "make_io_fixtures.py"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    out = tmp_path_factory.mktemp("io_fixtures")
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], check=True)
    return out


def test_fixtures_stay_small():
    assert sum(p.stat().st_size for p in FIXTURES.iterdir()) < 200_000


def test_regeneration_reproduces_committed_fixtures(regenerated):
    names = sorted(p.name for p in FIXTURES.iterdir())
    assert sorted(p.name for p in regenerated.iterdir()) == names
    fresh = json.loads((regenerated / "expected.json").read_text())
    assert {k: v for k, v in fresh.items() if k != "generator"} == \
           {k: v for k, v in EXPECTED.items() if k != "generator"}
    # Deflate output depends on the zlib build; npz bytes are only comparable with the same one.
    same_zlib = EXPECTED["generator"]["zlib"] == zlib.ZLIB_RUNTIME_VERSION
    for name in names:
        if name == "expected.json" or (name.endswith(".npz") and not same_zlib):
            continue
        assert (FIXTURES / name).read_bytes() == (regenerated / name).read_bytes(), name


# load cases whose expected matrix is exactly what scipy reads: no triangle restored, no
# Hermitian part taken (duplicates are summed by scipy as well; "real" only drops zero imag).
_PLAIN_NOTES = {"real", "ignored", "unchecked", "duplicates", "zeros"}
# Python's zipfile takes the last end-record signature it finds, so it cannot parse an archive
# comment, or bytes appended after the archive, that contain one.
_PYTHON_UNREADABLE = {"comment_with_signature.npz", "trailing_signature.npz"}


@pytest.mark.parametrize("case", [c for c in EXPECTED["load"]
                                  if c["file"].endswith(".npz") and "error" not in c
                                  and c["storage"] == "full" and set(c["notes"]) <= _PLAIN_NOTES
                                  and c["file"] not in _PYTHON_UNREADABLE],
                         ids=lambda c: c["name"])
def test_scipy_reads_what_expected_json_claims(case):
    m = sp.load_npz(FIXTURES / case["file"])
    dense = m.toarray().astype(np.complex128)
    want = np.array(case["re"], dtype=np.complex128)
    if case["im"] is not None:
        want += 1j * np.array(case["im"])
    assert m.shape == (case["n"], case["n"])
    assert np.array_equal(dense, want)
