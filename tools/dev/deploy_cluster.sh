#!/usr/bin/env bash
# Deploy the project environments, code and test data to the cluster share (gpu01 / gpu02 / c001:
# CentOS 7, glibc 2.17, no internet), from the development server's conda package cache.
#
#   tools/dev/deploy_cluster.sh [HOST]        (HOST: any node mounting /share, default gpu01)
#
# Steps (idempotent; existing environments and clones are kept):
#   1. conda list --explicit --md5 of os2026 and os2026-ref -> file:// lists of the cached packages;
#      libfabric1 2.x needs glibc 2.25, so the SLEPc env gets libfabric1 1.22 (glibc 2.14, provides
#      every FABRIC_1.x symbol version OpenMPI 5 uses; downloaded here into the package cache);
#   2. rsync the packages to $REMOTE_DEPLOY/pkgs and create $HOME/envs/{os2026,os2026-ref} offline;
#   3. bare repository $HOME/git/os2026.git (git remote "cluster"), push main, clone $HOME/os2026;
#   4. rsync data/ (not under version control) into the clone.
# The node-side settings (conda binary, env prefixes, cores, memory) are in env/hosts/cluster-node.sh.
# MUMPS needs no step here: the repository carries its source tarball (third_party/mumps/), and the
# first tools/dev/build.sh on any node compiles the default patched MUMPS (~1.5 min) into the shared
# $HOME/deps/mumps-cache (HS_MUMPS_CACHE, env/hosts/cluster-node.sh), which every later build of any
# clone with the same build inputs reuses (tools/dev/mumps_cache.sh).
set -euo pipefail
HOST="${1:-gpu01}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
PKGS="$("$CONDA" info --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["pkgs_dirs"][0])')"
REMOTE_HOME="$(ssh -o BatchMode=yes "$HOST" 'echo $HOME')"
REMOTE_DEPLOY="$REMOTE_HOME/hsolve-deploy"
LIBFABRIC_COMPAT="libfabric1-1.22.0-ha594dbc_3.conda"
LIBFABRIC_URL="https://conda.anaconda.org/conda-forge/linux-64/$LIBFABRIC_COMPAT"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT

echo "[1/4] package lists"
[ -e "$PKGS/$LIBFABRIC_COMPAT" ] || curl -sSfL -o "$PKGS/$LIBFABRIC_COMPAT" "$LIBFABRIC_URL"
for env in os2026 os2026-ref; do
  "$CONDA" list -n "$env" --explicit --md5 | grep '^http' > "$work/$env.urls"
  if [ "$env" = os2026-ref ]; then
    sed -i -E "s#^.*/libfabric1-[^/]*\$#$LIBFABRIC_URL#" "$work/$env.urls"
  fi
  { echo '@EXPLICIT'
    sed -E "s#^https?://[^#]*/([^/#]+)(\#.*)?\$#file://$REMOTE_DEPLOY/pkgs/\1\2#" "$work/$env.urls"; } > "$work/$env.local.explicit.txt"
done
cat "$work"/*.urls | sed -E 's#.*/([^/#]+)(\#.*)?$#\1#' | sort -u > "$work/files.txt"

echo "[2/4] packages -> $HOST:$REMOTE_DEPLOY, environments"
ssh "$HOST" "mkdir -p $REMOTE_DEPLOY/pkgs $REMOTE_HOME/envs"
(cd "$PKGS" && rsync -a --files-from="$work/files.txt" ./ "$HOST:$REMOTE_DEPLOY/pkgs/")
rsync -a "$work"/*.local.explicit.txt "$HOST:$REMOTE_DEPLOY/"
ssh "$HOST" "set -e; for e in os2026 os2026-ref; do
  if [ -d $REMOTE_HOME/envs/\$e ]; then echo \"  \$e exists\"; continue; fi
  /share/apps/anaconda3/bin/conda create -y -q -p $REMOTE_HOME/envs/\$e --offline \
    --file $REMOTE_DEPLOY/\$e.local.explicit.txt > $REMOTE_DEPLOY/create-\$e.log 2>&1 && echo \"  \$e created\"
done"

echo "[3/4] code"
ssh "$HOST" "mkdir -p $REMOTE_HOME/git && cd $REMOTE_HOME/git && { [ -d os2026.git ] || git init -q --bare os2026.git; } \
  && cd os2026.git && git symbolic-ref HEAD refs/heads/main"
git -C "$REPO" remote get-url cluster >/dev/null 2>&1 || git -C "$REPO" remote add cluster "$HOST:$REMOTE_HOME/git/os2026.git"
git -C "$REPO" push -q cluster main
ssh "$HOST" "cd $REMOTE_HOME && { [ -d os2026/.git ] || git clone -q git/os2026.git os2026; } && cd os2026 \
  && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1"

echo "[4/4] data"
rsync -a "$REPO/data/" "$HOST:$REMOTE_HOME/os2026/data/"
echo "done. Build and test on the node:  ssh $HOST 'cd ~/os2026 && env/run.sh tools/dev/build.sh release && env/run.sh tools/dev/test.sh release'"
