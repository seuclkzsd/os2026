#!/usr/bin/env bash
# Build and run C++ unit tests (ctest) and Python tool tests (pytest) under the resource guard.
# Usage: tools/dev/test.sh [release|debug] [ctest -R regex]
# ctest names are "<test executable>.<Suite>.<Name>", e.g. -R test_sparse or -R SparsePencil.
# A regex that matches no test is an error (noTestsAction), never a silent pass.
set -euo pipefail
preset="${1:-release}"
regex="${2:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
tools/dev/build.sh "$preset"
# shellcheck disable=SC1091
source env/activate.sh
guard=(python3 tools/run/guard.py --skip-busy-check --mem-gb "${HS_TEST_MEM_GB:-6}" --timeout 3600 --)
if [[ -n "$regex" ]]; then
  "${guard[@]}" ctest --preset "$preset" --no-tests=error -R "$regex"
else
  "${guard[@]}" ctest --preset "$preset" --no-tests=error
  if compgen -G "tests/py/test_*.py" >/dev/null; then
    "${guard[@]}" python -m pytest -q tests/py
  fi
fi
