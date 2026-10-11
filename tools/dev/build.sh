#!/usr/bin/env bash
# Configure + build a preset under the resource guard.  Usage: tools/dev/build.sh [release|debug|portable]
# HS_JOBS (default 4) sets the build parallelism; HS_CORES pins the build (see env/activate.sh).
set -euo pipefail
preset="${1:-release}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source env/activate.sh
# Builds peak at ~0.5 GiB; stay within HS_MAX_SHARED_MEM_GB during parallel multi-agent phases.
build_mem="${HS_BUILD_MEM_GB:-$(python3 -c "import os; print(min(12.0, float(os.environ.get('HS_MAX_SHARED_MEM_GB', 'inf'))))")}"
guard=(python3 tools/run/guard.py --skip-busy-check --mem-gb "$build_mem" --timeout 3600 --)
"${guard[@]}" cmake --preset "$preset" >/dev/null
"${guard[@]}" cmake --build --preset "$preset" -j "${HS_JOBS:-4}"
