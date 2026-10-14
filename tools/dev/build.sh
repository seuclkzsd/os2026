#!/usr/bin/env bash
# Configure + build a preset under the resource guard.  Usage: tools/dev/build.sh [release|debug|portable]
# HS_JOBS (default 4) sets the build parallelism; HS_CORES pins the build (see env/activate.sh).
#
# MUMPS (every preset links it; the Python module too) - HS_MUMPS selects which:
#   source (default)  MUMPS 5.8.2 built from the repository's tarball (third_party/mumps/) with
#                     patches/mumps-5.8.2-l0-blocking.patch, cached by tools/dev/mumps_cache.sh
#                     (first build ~1.5 min, then reused); hsolve then gives MUMPS the P9 block sizes
#                     by default (HS_MUMPS_BLOCKING=mumps restores MUMPS' own at run time)
#   env               the environment's conda-forge mumps-seq (the pre-P9 build: MUMPS' block sizes)
#   /abs/prefix       an install of tools/dev/build_mumps.sh (P9 block sizes by default)
# The choice is passed as -DHSOLVE_MUMPS_ROOT (empty = the environment's) on every call, so switching
# reconfigures the same build directory (docs/DESIGN.md §3.4).
set -euo pipefail
preset="${1:-release}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source env/activate.sh
# Builds peak at ~0.5 GiB; stay within HS_MAX_SHARED_MEM_GB during parallel multi-agent phases.
build_mem="${HS_BUILD_MEM_GB:-$(python3 -c "import os; print(min(12.0, float(os.environ.get('HS_MAX_SHARED_MEM_GB', 'inf'))))")}"
guard=(python3 tools/run/guard.py --skip-busy-check --mem-gb "$build_mem" --timeout 3600 --)
mumps="${HS_MUMPS:-source}"
case "$mumps" in
  source)
    "${guard[@]}" tools/dev/mumps_cache.sh ensure >/dev/null
    mumps_root="$(tools/dev/mumps_cache.sh prefix)" ;;
  env) mumps_root="" ;;
  /*) [ -f "$mumps/include/dmumps_c.h" ] || { echo "build.sh: HS_MUMPS=$mumps is no MUMPS install" >&2; exit 2; }
      mumps_root="$mumps" ;;
  *) echo "build.sh: HS_MUMPS must be source, env or an absolute install prefix, got '$mumps'" >&2; exit 2 ;;
esac
"${guard[@]}" cmake --preset "$preset" -DHSOLVE_MUMPS_ROOT="$mumps_root" >/dev/null
"${guard[@]}" cmake --build --preset "$preset" -j "${HS_JOBS:-4}"
