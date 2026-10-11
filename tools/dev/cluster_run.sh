#!/usr/bin/env bash
# Run a command on a cluster node (gpu01 / gpu02 / c001) on exactly the committed state of the
# current git worktree, in its own remote work directory (tools/dev/deploy_cluster.sh must have run).
#
#   tools/dev/cluster_run.sh HOST [--allow-dirty] -- CMD [ARGS...]
#
# 1. pushes HEAD to the remote "cluster" as refs/heads/wt/<name>; <name> = $HS_CLUSTER_NAME or the
#    basename of the worktree (uncommitted changes are not transferred: commit first; --allow-dirty
#    only silences the check);
# 2. on HOST: clones $HOME/git/os2026.git into $HOME/os2026-wt/<name> once, checks out that commit,
#    links data/ to the shared $HOME/os2026/data;
# 3. runs `env/run.sh CMD ARGS...` there (env/hosts/<host>.sh sets cores, memory and environments).
# Example:
#   tools/dev/cluster_run.sh gpu02 -- tools/dev/build.sh release
#   tools/dev/cluster_run.sh gpu02 -- python3 tools/run/guard.py --cores 16-23 -- build/release/hsolve ...
# The home directory is shared by all nodes, so one <name> must not build on two nodes at once.
set -euo pipefail
usage() { sed -n '2,16p' "$0" >&2; exit 2; }
[ $# -ge 3 ] || usage
HOST="$1"; shift
dirty_ok=0
[ "${1:-}" = "--allow-dirty" ] && { dirty_ok=1; shift; }
[ "${1:-}" = "--" ] || usage
shift
top="$(git rev-parse --show-toplevel)"
name="${HS_CLUSTER_NAME:-$(basename "$top")}"
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "cluster_run: bad work directory name '$name'" >&2; exit 2; }
if [ "$dirty_ok" = 0 ] && [ -n "$(git -C "$top" status --porcelain --untracked-files=no)" ]; then
  echo "cluster_run: uncommitted changes in $top are not transferred; commit them (or --allow-dirty)" >&2
  exit 2
fi
sha="$(git -C "$top" rev-parse HEAD)"
git -C "$top" push -q -f cluster "HEAD:refs/heads/wt/$name"
printf -v cmd '%q ' "$@"
ssh -o BatchMode=yes "$HOST" "set -e
  wt=\$HOME/os2026-wt/$name
  if [ ! -d \$wt/.git ]; then mkdir -p \$HOME/os2026-wt && git clone -q \$HOME/git/os2026.git \$wt; fi
  cd \$wt && git fetch -q origin wt/$name && git checkout -q -f $sha
  [ -e data ] || ln -s \$HOME/os2026/data data
  exec env/run.sh $cmd"
