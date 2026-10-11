#!/usr/bin/env bash
# Build BLIS 2.1 from the pinned tag tarball (docs/BLAS.md).
#
#   env/run.sh tools/deps/build_blis.sh NAME [configure arguments ...]
#
# Installs into $HS_DEPS/blis-2.1-NAME: lib/libblis.so.* with the BLAS compatibility layer
# (Fortran dgemm_, dtrsm_, ...; 32-bit integers) and CBLAS; LAPACK is not part of BLIS, so it is
# used as LD_PRELOAD=<prefix>/lib/libblis.so.4 in front of the environment's OpenBLAS (MUMPS'
# BLAS calls then bind to BLIS, LAPACK stays OpenBLAS'). Default configure arguments:
# --enable-threading=openmp --enable-cblas, configuration family x86_64 (runtime dispatch over
# skx / haswell / zen* / sandybridge / penryn / generic kernels). Arguments given replace the
# configuration name (the last word, e.g. "skx") or add options ("--enable-threading=openmp skx").
# Source: https://github.com/flame/blis/archive/refs/tags/2.1.tar.gz
# (sha256 = conda-forge blis-feedstock recipe). License BSD-3-Clause.
main() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=tools/deps/common.sh
  source "$here/common.sh"
  deps_require_env
  [ $# -ge 1 ] || deps_die "usage: build_blis.sh NAME [configure arguments ...]"
  local name="$1"
  shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || deps_die "bad NAME '$name'"
  local ver=2.1
  local sha=901752ec596cd63421ea45a6a06a9f34491cf5ae01cea412ece05f97a2690a96
  local tarball="$HS_DEPS_SRC/blis-$ver.tar.gz"
  local build="$HS_DEPS_BUILD/blis-$ver-$name"
  local prefix="$HS_DEPS/blis-$ver-$name"
  deps_check "$tarball" "$sha"
  deps_unpack "$tarball" "$build"
  local args=(--prefix="$prefix" --enable-threading=openmp --enable-cblas --disable-static
              --enable-shared)
  if [ $# -gt 0 ]; then args+=("$@"); else args+=(x86_64); fi
  echo "blis $ver -> $prefix: ${args[*]}"
  rm -rf "$prefix"
  (cd "$build" && CC="$CC" CXX="$CXX" ./configure "${args[@]}" >"$build/configure.log" 2>&1 &&
     make -j"$(deps_jobs)" >"$build/build.log" 2>&1 && make install >"$build/install.log" 2>&1) ||
    { tail -40 "$build"/*.log >&2; deps_die "build failed (logs in $build)"; }
  local lib
  for lib in "$prefix"/lib/libblis.so.?; do deps_check_glibc "$lib"; done
  deps_info "$prefix" "$tarball" "$sha" "${args[@]}"
  echo "done: $prefix"
}
main "$@"
exit
