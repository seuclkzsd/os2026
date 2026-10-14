#!/usr/bin/env bash
# The MUMPS of the default build: MUMPS 5.8.2 from third_party/mumps/, patched with
# patches/mumps-5.8.2-l0-blocking.patch, built by tools/dev/build_mumps.sh with its portable defaults
# (nocona -O2: conda-forge's flags) against the project environment - built once per cache key and
# then reused (P9 integration; docs/DESIGN.md §3.4, results/p9_factspeed.md).
#
#   env/run.sh tools/dev/mumps_cache.sh ensure     build unless cached; print the install prefix
#   env/run.sh tools/dev/mumps_cache.sh prefix     print the install prefix (whether built or not)
#   env/run.sh tools/dev/mumps_cache.sh describe   print the cache key's inputs
#
# tools/dev/build.sh runs `ensure` (under guard.py) and links hsolve with -DHSOLVE_MUMPS_ROOT=<prefix>.
# Cache: $HS_MUMPS_CACHE (default <repo>/build/deps; the cluster nodes share $HOME/deps/mumps-cache,
# env/hosts/cluster-node.sh), one install per key, <cache>/mumps-5.8.2-<key>/. The key is the
# sha256 of: the tarball's checksum, each patch's, the build flags, the two build scripts
# (build_mumps.sh, tools/deps/common.sh), the C and Fortran compilers' versions and the environment
# prefix ($CONDA_PREFIX: the libraries link its METIS / SCOTCH / OpenBLAS by RPATH). Any change of
# these builds a new install next to the old ones (delete <cache> to clean up, after the build
# directories that link it). The install's CACHE-KEY.txt holds the inputs, BUILD-INFO.txt the build.
# Concurrent callers (several work trees sharing a cache) do not lock: each builds in its own
# temporary directory and renames the finished install into place; the first rename wins, the others
# discard theirs. The install is relocatable (the libraries find each other through $ORIGIN) but the
# binaries linked to it record its path: do not move or delete an install that builds still use.
# A failed build keeps its build directory, <cache>/.tmp-<key>-XXXXXX/build/mumps-5.8.2-<key>/
# (build.log: the whole make output), and names it; delete that .tmp directory after reading it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ver=5.8.2
sha=eb515aa688e6dbab414bb6e889ff4c8b23f1691a843c68da5230a33ac4db7039  # = build_mumps.sh's
patches=("$repo/patches/mumps-5.8.2-l0-blocking.patch")
flags=(--isa nocona --opt O2)

die() {
  echo "mumps_cache: $*" >&2
  exit 1
}
[ -n "${CONDA_PREFIX:-}" ] && [ -n "${CC:-}" ] && [ -n "${FC:-}" ] ||
  die "run through env/run.sh (the project environment's compilers and libraries)"
for p in "${patches[@]}" "$here/build_mumps.sh" "$repo/tools/deps/common.sh"; do
  [ -f "$p" ] || die "missing $p"
done
cc_version="$("$CC" --version | head -1)" || die "cannot run CC=$CC"
fc_version="$("$FC" --version | head -1)" || die "cannot run FC=$FC"

describe() {
  local p
  echo "mumps $ver tarball-sha256 $sha"
  for p in "${patches[@]}"; do
    echo "patch $(basename "$p") $(sha256sum "$p" | cut -d' ' -f1)"
  done
  echo "flags ${flags[*]}"
  echo "script build_mumps.sh $(sha256sum "$here/build_mumps.sh" | cut -d' ' -f1)"
  echo "script common.sh $(sha256sum "$repo/tools/deps/common.sh" | cut -d' ' -f1)"
  echo "cc $cc_version"
  echo "fc $fc_version"
  echo "env $CONDA_PREFIX"
}

cache="${HS_MUMPS_CACHE:-$repo/build/deps}"
[[ "$cache" == /* ]] || cache="$PWD/$cache"  # the prefix ends up in RPATHs: absolute
key="$(describe | sha256sum | cut -c1-16)"
prefix="$cache/mumps-$ver-$key"

case "${1:-}" in
  describe) describe ;;
  prefix) echo "$prefix" ;;
  ensure)
    if [ -f "$prefix/CACHE-KEY.txt" ]; then
      echo "mumps_cache: reusing $prefix" >&2
    else
      mkdir -p "$cache"
      tmp="$(mktemp -d "$cache/.tmp-$key-XXXXXX")"
      trap 'rm -rf "$tmp"' EXIT
      args=()
      for p in "${patches[@]}"; do args+=(--patch "$p"); done
      echo "mumps_cache: building $prefix (key inputs: $prefix/CACHE-KEY.txt)" >&2
      if ! HS_DEPS="$tmp" HS_DEPS_BUILD="$tmp/build" "$here/build_mumps.sh" "$key" "${flags[@]}" "${args[@]}" >&2; then
        # build_mumps.sh printed the tail of its make log and named it: keep the build directory
        # (build.log, Makefile.inc, the partial build) instead of removing it with $tmp; the
        # partial install goes.
        rm -rf "$tmp/mumps-$ver-$key"
        if [ -f "$tmp/build/mumps-$ver-$key/build.log" ]; then
          trap - EXIT
          die "building MUMPS failed; kept for diagnosis: $tmp/build/mumps-$ver-$key/build.log" \
              "(and that directory's Makefile.inc and partial build) - delete $tmp afterwards"
        fi
        die "building MUMPS failed (before make; output above)"
      fi
      describe > "$tmp/mumps-$ver-$key/CACHE-KEY.txt"
      # rename(2) onto an existing non-empty directory fails: a concurrent caller finished first.
      if ! mv -T "$tmp/mumps-$ver-$key" "$prefix" 2>/dev/null; then
        [ -f "$prefix/CACHE-KEY.txt" ] || die "could not install $prefix"
        echo "mumps_cache: $prefix was installed meanwhile by another build; using it" >&2
      fi
    fi
    echo "$prefix"
    ;;
  *) sed -n '2,24p' "$0" >&2; exit 2 ;;
esac
