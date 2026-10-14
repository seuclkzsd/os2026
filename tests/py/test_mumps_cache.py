"""tools/dev/mumps_cache.sh when the MUMPS build fails: the make log that build_mumps.sh names survives
(the cache builds in a temporary directory that is removed on exit; p9int review), and no install is
left at the cache prefix. The Fortran compiler is replaced by one that answers --version and
-print-file-name like the environment's (same cache key inputs) but compiles nothing, so make fails
within seconds of unpacking the tarball. Needs the project environment (env/run.sh)."""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tools" / "dev" / "mumps_cache.sh"


@pytest.mark.skipif(not (os.environ.get("CONDA_PREFIX") and os.environ.get("CC") and os.environ.get("FC")),
                    reason="needs the project environment (env/run.sh)")
def test_failed_build_keeps_the_make_log(tmp_path: Path):
    fc = tmp_path / "bin" / "broken-fc"
    fc.parent.mkdir()
    fc.write_text("#!/bin/sh\n"
                  f'case "$1" in --version|-print-file-name=*) exec "{os.environ["FC"]}" "$@" ;; esac\n'
                  'echo "broken-fc: no Fortran compiler in this test" >&2\nexit 1\n')
    fc.chmod(fc.stat().st_mode | stat.S_IXUSR)
    cache = tmp_path / "cache"
    env = dict(os.environ, HS_MUMPS_CACHE=str(cache), FC=str(fc), HS_JOBS="2")
    p = subprocess.run([str(_SCRIPT), "ensure"], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode != 0, p.stderr[-2000:]
    m = re.search(r"kept for diagnosis: (\S+/build\.log)", p.stderr)
    assert m, p.stderr[-2000:]
    log = Path(m.group(1))
    assert log.is_file() and "broken-fc: no Fortran compiler in this test" in log.read_text()
    assert log.parent.parent.parent.parent == cache and log.parent.parent.parent.name.startswith(".tmp-")
    assert f"(log {log})" in p.stderr       # the path build_mumps.sh printed is the one kept
    assert (log.parent / "Makefile.inc").is_file()
    prefix = subprocess.run([str(_SCRIPT), "prefix"], env=env, capture_output=True, text=True,
                            check=True).stdout.strip()
    assert not Path(prefix).exists()
    assert not (log.parent.parent.parent / Path(prefix).name).exists(), "the partial install is removed"
