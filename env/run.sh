#!/usr/bin/env bash
# Run a command inside the project environment without sourcing anything into the caller:
#   env/run.sh <cmd> [args...]        (works from any working directory, e.g. a git worktree)
# Equivalent to `source env/activate.sh && <cmd>`; needed where sourcing is not possible.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/activate.sh"
exec "$@"
