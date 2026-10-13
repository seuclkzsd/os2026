#!/usr/bin/env bash
# Run a command on a cluster node (gpu01 / gpu02 / c001) on exactly the committed state of the
# current git worktree, in its own remote work directory (tools/dev/deploy_cluster.sh must have run).
#
#   tools/dev/cluster_run.sh HOST [--wip | --allow-dirty] -- CMD [ARGS...]
#
# 1. pushes HEAD to the remote "cluster" as refs/heads/wt/<name>; <name> = $HS_CLUSTER_NAME or the
#    basename of the worktree. Uncommitted changes are refused unless --wip (transfer the working
#    tree's tracked files - stage new files with `git add` first - as a temporary commit made by
#    `git stash create`, which touches neither the working tree, the index nor the stash list) or
#    --allow-dirty (run HEAD and ignore the changes);
# 2. on HOST: clones $HOME/git/os2026.git into $HOME/os2026-wt/<name> once, checks out that commit,
#    links data/ to the shared $HOME/os2026/data;
# 3. runs `env/run.sh CMD ARGS...` there (env/hosts/<host>.sh sets cores, memory and environments;
#    HS_CORES, HS_JOBS, HS_MEM_GB, HS_MIN_AVAIL_GB and HS_LOCK_WAIT_S set in the caller's
#    environment are forwarded and take precedence; the scheduling limits and data paths are not:
#    the node uses its own limits and the work directory's data/ link).
# Scheduling on the nodes (docs/DESIGN.md §6): builds, tests and smoke runs use the shared lane
# (default HS_CORES); timed runs take the node's measurement slot - 16 cores, 32 GiB, one run at a
# time - with guard.py --exclusive.
# Example:
#   tools/dev/cluster_run.sh gpu02 -- tools/dev/build.sh release
#   tools/dev/cluster_run.sh gpu02 -- python3 tools/run/guard.py --exclusive --timeout 900 -- build/release/hsolve ...
#   tools/dev/cluster_run.sh gpu02 -- python3 tools/run/guard.py --status
# The home directory is shared by all nodes, so one <name> must not build on two nodes at once.
set -euo pipefail
usage() { sed -n '2,25p' "$0" >&2; exit 2; }
[ $# -ge 3 ] || usage
HOST="$1"; shift
dirty_ok=0 wip=0
case "${1:-}" in
  --allow-dirty) dirty_ok=1; shift ;;
  --wip) wip=1; shift ;;
esac
[ "${1:-}" = "--" ] || usage
shift
top="$(git rev-parse --show-toplevel)"
name="${HS_CLUSTER_NAME:-$(basename "$top")}"
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "cluster_run: bad work directory name '$name'" >&2; exit 2; }
sha="$(git -C "$top" rev-parse HEAD)"
if [ -n "$(git -C "$top" status --porcelain --untracked-files=no)" ]; then
  if [ "$wip" = 1 ]; then
    sha="$(git -C "$top" stash create "cluster_run wip")"
  elif [ "$dirty_ok" = 0 ]; then
    echo "cluster_run: uncommitted changes in $top are not transferred; commit them, or pass --wip" \
         "(transfer them) or --allow-dirty (ignore them)" >&2
    exit 2
  fi
fi
git -C "$top" push -q -f cluster "$sha:refs/heads/wt/$name"
printf -v cmd '%q ' "$@"
fwd=""
for v in HS_CORES HS_JOBS HS_MEM_GB HS_MIN_AVAIL_GB HS_LOCK_WAIT_S; do
  [ -n "${!v:-}" ] && fwd+="$v=$(printf '%q' "${!v}") "
done
ssh -o BatchMode=yes "$HOST" "set -e
  wt=\$HOME/os2026-wt/$name
  if [ ! -d \$wt/.git ]; then mkdir -p \$HOME/os2026-wt && git clone -q \$HOME/git/os2026.git \$wt; fi
  cd \$wt && git fetch -q origin wt/$name && git checkout -q -f $sha
  [ -e data ] || ln -s \$HOME/os2026/data data
  exec env ${fwd}env/run.sh $cmd"
