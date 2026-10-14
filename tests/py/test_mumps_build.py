"""The MUMPS the built hsolve executable and Python module load (P9 integration; CMakeLists.txt
HSOLVE_MUMPS_ROOT, tools/dev/build.sh HS_MUMPS, docs/DESIGN.md §3.4).

The default build links MUMPS 5.8.2 built from third_party/mumps/ with the L0-blocking patch
(tools/dev/mumps_cache.sh); HS_MUMPS=env links the environment's conda-forge mumps-seq. Both carry
the same library names and sonames, so only the RPATH order decides which one the dynamic loader
maps. These tests read the build directory's CMakeCache.txt and check that
  * the hsolve executable resolves every MUMPS library to the configured install
    (LD_TRACE_LOADED_OBJECTS: the loader's own resolution, as ldd prints it), and
  * the Python module, after a solve, has exactly those libraries mapped (/proc/self/maps of a
    fresh interpreter), and
  * both report the configured MUMPS and default block sizes in their build information.
(The C++ side: tests/cpp/test_factor_mumps_build.cpp, for the unit-test executables.)
Build directory: $HSOLVE_BUILD_DIR, default build/release (the Python package in its python/,
or $HSOLVE_PYTHON_DIR as in the other binding tests).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BUILD = Path(os.environ.get("HSOLVE_BUILD_DIR", _ROOT / "build" / "release"))
_PY_DIR = Path(os.environ.get("HSOLVE_PYTHON_DIR", _BUILD / "python"))
_CACHE = _BUILD / "CMakeCache.txt"
if not _CACHE.is_file():
    pytest.skip(f"no configured build in {_BUILD} (tools/dev/build.sh release)", allow_module_level=True)

# The libraries of conda-forge's mumps-seq / tools/dev/build_mumps.sh: the ones every hsolve binary
# uses (double and single precision, the common part) must be loaded; the others are checked where
# loaded (an --as-needed link drops the unused complex arithmetics).
_REQUIRED = ("libdmumps_seq.so", "libsmumps_seq.so", "libmumps_common_seq.so")
_MUMPS_LIBS = _REQUIRED + ("libzmumps_seq.so", "libcmumps_seq.so", "libpord_seq.so", "libmpiseq_seq.so")


def cmake_cache(name: str) -> str:
    for line in _CACHE.read_text().splitlines():
        m = re.match(rf"^{re.escape(name)}(:[A-Z]+)?=(.*)$", line)
        if m:
            return m.group(2)
    raise KeyError(name)


def mumps_root() -> str:
    return cmake_cache("HSOLVE_MUMPS_ROOT")


def expected_blocking(root: str) -> str:
    configured = cmake_cache("HSOLVE_MUMPS_BLOCKING")
    return configured or ("p9" if root else "mumps")


def check_paths(paths: dict[str, str], root: str) -> None:
    """The required MUMPS libraries loaded; every loaded one under root/lib when a root is
    configured, else not from a build_mumps.sh / mumps_cache.sh install."""
    for lib in _REQUIRED:
        assert lib in paths, f"{lib} not loaded: {sorted(paths)}"
    for lib in _MUMPS_LIBS:
        if lib not in paths:
            continue
        real = os.path.realpath(paths[lib])
        if root:
            want = os.path.realpath(os.path.join(root, "lib"))
            assert os.path.dirname(real) == want, f"{lib} -> {real}, not from HSOLVE_MUMPS_ROOT {root}"
        else:
            assert "/mumps-5.8.2-" not in real, f"{lib} -> {real}: a source build in an env build"


def test_cli_resolves_the_configured_mumps():
    exe = _BUILD / "hsolve"
    if not exe.is_file():
        pytest.skip(f"{exe} not built")
    env = dict(os.environ, LD_TRACE_LOADED_OBJECTS="1")
    out = subprocess.run([str(exe)], env=env, capture_output=True, text=True, check=True).stdout
    paths = {}
    for line in out.splitlines():
        m = re.match(r"^\s*(\S+) => (\S+) \(0x", line)
        if m:
            paths[m.group(1)] = m.group(2)
    check_paths(paths, mumps_root())


def test_cli_build_info_names_mumps():
    exe = _BUILD / "hsolve"
    if not exe.is_file():
        pytest.skip(f"{exe} not built")
    root = mumps_root()
    info = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True).stdout
    label = os.path.basename(root.rstrip("/")) if root else "env"
    assert f"mumps={label} " in info, info
    assert f"mumps_blocking={expected_blocking(root)}" in info, info


_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, scipy.sparse as sp
import hsolve
n = 400
e = np.ones(n)
a = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csr")
res = hsolve.solve(a, sigma=1.0, k=4, threads=2)
maps = {}
for line in open("/proc/self/maps"):
    path = line.split()[-1] if "/" in line else ""
    name = path.rsplit("/", 1)[-1]
    if name:
        maps.setdefault(name, set()).add(path)
print(json.dumps({"success": res.success, "build_info": hsolve.build_info(),
                  "maps": {k: sorted(v) for k, v in maps.items()}}))
"""


def test_python_module_maps_the_configured_mumps():
    if not (_PY_DIR / "hsolve" / "__init__.py").is_file():
        pytest.skip(f"hsolve Python package not built in {_PY_DIR}")
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _PROBE, str(_PY_DIR)], env=env, capture_output=True,
                         text=True, check=True).stdout
    probe = json.loads(out.strip().splitlines()[-1])
    assert probe["success"]
    root = mumps_root()
    paths = {}
    for lib in _MUMPS_LIBS:
        mapped = probe["maps"].get(lib, [])
        assert len(mapped) <= 1, f"{lib}: one copy mapped, got {mapped}"
        if mapped:
            paths[lib] = mapped[0]
    check_paths(paths, root)
    label = os.path.basename(root.rstrip("/")) if root else "env"
    assert f"mumps={label} " in probe["build_info"]
    assert f"mumps_blocking={expected_blocking(root)}" in probe["build_info"]


_TARBALL = _ROOT / "third_party" / "mumps" / "MUMPS_5.8.2.tar.gz"
_PINNED = "eb515aa688e6dbab414bb6e889ff4c8b23f1691a843c68da5230a33ac4db7039"


def test_shipped_tarball_is_the_pinned_one():
    import hashlib
    h = hashlib.sha256(_TARBALL.read_bytes()).hexdigest()
    assert h == _PINNED
    for script in ("tools/dev/build_mumps.sh", "tools/dev/mumps_cache.sh"):
        assert _PINNED in (_ROOT / script).read_text(), script


def test_cache_prefix_is_keyed_by_the_build_inputs(tmp_path):
    if not (os.environ.get("CONDA_PREFIX") and os.environ.get("CC") and os.environ.get("FC")):
        pytest.skip("needs the project environment (env/run.sh)")
    script = str(_ROOT / "tools/dev/mumps_cache.sh")

    def run(*args, **env):
        return subprocess.run([script, *args], env=dict(os.environ, **env), capture_output=True, text=True,
                              check=True).stdout.strip()

    prefix = run("prefix", HS_MUMPS_CACHE=str(tmp_path))
    assert prefix == run("prefix", HS_MUMPS_CACHE=str(tmp_path))
    assert re.fullmatch(re.escape(str(tmp_path)) + r"/mumps-5\.8\.2-[0-9a-f]{16}", prefix), prefix
    desc = run("describe")
    assert f"tarball-sha256 {_PINNED}" in desc
    assert "patch mumps-5.8.2-l0-blocking.patch " in desc and "flags --isa nocona --opt O2" in desc
    # Other compilers (or any other input) give another install.
    assert run("prefix", HS_MUMPS_CACHE=str(tmp_path), FC=os.environ["CC"]) != prefix
    assert not any(tmp_path.iterdir()), "prefix / describe must not build"
