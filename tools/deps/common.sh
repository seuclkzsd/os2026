# Shared helpers of tools/deps/build_*.sh (sourced, not executed). docs/BLAS.md describes the builds.
#
# Layout (all under $HS_DEPS, default $HOME/deps):
#   src/<tarball>            pinned source tarballs. The cluster nodes have no internet: download on
#                            a machine that has (URLs in each build script), check the sha256 there,
#                            rsync into src/; every build checks the sha256 again before unpacking.
#   build/<lib>-<ver>-<name> fresh build tree per build (removed first when it exists)
#   <lib>-<ver>-<name>/      install prefix; BUILD-INFO.txt records tarball, checksum, flags, compiler
# The build tools come from the project environment: run the scripts through env/run.sh (or after
# `source env/activate.sh`) so that CC / FC / CXX are conda-forge's GCC 13 and libgomp is the
# environment's (the one MUMPS and hsolve use). HS_JOBS (default 4) sets the build parallelism.
set -euo pipefail

HS_DEPS="${HS_DEPS:-$HOME/deps}"
HS_DEPS_SRC="${HS_DEPS_SRC:-$HS_DEPS/src}"
HS_DEPS_BUILD="${HS_DEPS_BUILD:-$HS_DEPS/build}"

deps_die() {
  echo "deps: $*" >&2
  exit 1
}

# deps_check FILE SHA256: the tarball exists and has the pinned checksum.
deps_check() {
  local file="$1" want="$2" got
  [ -f "$file" ] || deps_die "missing source tarball $file (download it, check sha256 $want, copy it there)"
  got="$(sha256sum "$file" | cut -d' ' -f1)"
  [ "$got" = "$want" ] || deps_die "sha256 mismatch for $file: got $got, want $want"
}

# deps_unpack TARBALL DIR: DIR is created fresh and receives the tarball's single top directory.
deps_unpack() {
  local tarball="$1" dir="$2"
  rm -rf "$dir"
  mkdir -p "$dir"
  tar -xzf "$tarball" -C "$dir" --strip-components=1 --no-same-owner
}

# deps_info PREFIX TARBALL SHA256 FLAGS...: provenance file of an install.
deps_info() {
  local prefix="$1" tarball="$2" sha="$3"
  shift 3
  {
    echo "tarball: $(basename "$tarball")"
    echo "sha256: $sha"
    echo "flags: $*"
    echo "cc: $(${CC:-cc} --version | head -1)"
    echo "host: $(hostname -s) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$prefix/BUILD-INFO.txt"
}

deps_jobs() {
  echo "${HS_JOBS:-4}"
}

deps_require_env() {
  [ -n "${CONDA_PREFIX:-}" ] && [ -n "${CC:-}" ] ||
    deps_die "run through env/run.sh (conda-forge compilers and libgomp of the project environment)"
}
