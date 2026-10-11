#!/usr/bin/env bash
# Build and run C++ unit tests (ctest) and Python tool tests (pytest) under the resource guard.
# Usage: tools/dev/test.sh [release|debug] [ctest -R regex]
set -euo pipefail
preset="${1:-release}"
regex="${2:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
tools/dev/build.sh "$preset"
# shellcheck disable=SC1091
source env/activate.sh
guard=(python3 tools/run/guard.py --skip-busy-check --mem-gb 8 --timeout 3600 --threads 4 --)
if [[ -n "$regex" ]]; then
  "${guard[@]}" ctest --preset "$preset" -R "$regex"
else
  "${guard[@]}" ctest --preset "$preset"
  if compgen -G "tests/py/test_*.py" >/dev/null; then
    "${guard[@]}" python -m pytest -q tests/py
  fi
fi
