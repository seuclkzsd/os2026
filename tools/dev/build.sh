#!/usr/bin/env bash
# Configure + build a preset under the resource guard.  Usage: tools/dev/build.sh [release|debug|portable]
# HS_JOBS (default 4) sets the build parallelism; HS_CORES pins the build (see env/activate.sh).
set -euo pipefail
preset="${1:-release}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# shellcheck disable=SC1091
source env/activate.sh
guard=(python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3600 --)
"${guard[@]}" cmake --preset "$preset" >/dev/null
"${guard[@]}" cmake --build --preset "$preset" -j "${HS_JOBS:-4}"
