#!/usr/bin/env bash
# Build OpenBLAS 0.3.34 from the pinned release tarball (docs/BLAS.md).
#
#   env/run.sh tools/deps/build_openblas.sh NAME [MAKEVAR=VALUE ...]
#
# Installs into $HS_DEPS/openblas-0.3.34-NAME (lib/libopenblas.so.0 + libblas.so.3 / liblapack.so.3
# links, so that LD_LIBRARY_PATH=<prefix>/lib replaces the environment's OpenBLAS in hsolve, MUMPS
# and Python at run time). The base flags are conda-forge's (openblas-feedstock build.sh for
# linux-64): DYNAMIC_ARCH=1 TARGET=PRESCOTT (runtime CPU dispatch, every x86-64 CPU),
# USE_OPENMP=1 with the environment's libgomp, NUM_THREADS=128, NO_AFFINITY=1, BUILD_BFLOAT16=1;
# SMALL_MATRIX_OPT=1 is OpenBLAS' default on x86-64. Extra MAKEVAR=VALUE arguments come after the
# base flags and override them, e.g.
#   build_openblas.sh dynamic                                   # conda-forge's build, rebuilt here
#   build_openblas.sh dynamic-fwd GEMM_GEMV_FORWARD=1           # dgemm with N = 1 or M = 1 -> gemv
#   build_openblas.sh skx-fwd DYNAMIC_ARCH=0 TARGET=SKYLAKEX GEMM_GEMV_FORWARD=1   # this CPU only
# Source: https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34.tar.gz
# (sha256 also listed by Spack's openblas package and conda-forge's recipe). License BSD-3-Clause.
main() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=tools/deps/common.sh
  source "$here/common.sh"
  deps_require_env
  [ $# -ge 1 ] || deps_die "usage: build_openblas.sh NAME [MAKEVAR=VALUE ...]"
  local name="$1"
  shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || deps_die "bad NAME '$name'"
  local ver=0.3.34
  local sha=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed
  local tarball="$HS_DEPS_SRC/OpenBLAS-$ver.tar.gz"
  local build="$HS_DEPS_BUILD/openblas-$ver-$name"
  local prefix="$HS_DEPS/openblas-$ver-$name"
  deps_check "$tarball" "$sha"
  deps_unpack "$tarball" "$build"
  # conda-forge drops --gc-sections (breaks OpenBLAS' tests and symbol export); FFLAGS as theirs;
  # glibc 2.17 compatibility of libm calls (common.sh, deps_glibc_compat).
  export LDFLAGS="${LDFLAGS//-Wl,--gc-sections/} $(deps_glibc_compat "$build")"
  export FFLAGS="${FFLAGS:-} -frecursive"
  local flags=(DYNAMIC_ARCH=1 TARGET=PRESCOTT BINARY=64 NO_LAPACK=0 BUILD_BFLOAT16=1
               NO_AFFINITY=1 USE_THREAD=1 USE_OPENMP=1 NUM_THREADS=128 "CC=$CC" "FC=$FC"
               "HOSTCC=$CC" CFLAGS="${CFLAGS:-}" "$@")
  echo "openblas $ver -> $prefix: ${flags[*]}"
  (cd "$build" && make -j"$(deps_jobs)" "${flags[@]}" shared >"$build/build.log" 2>&1) ||
    { tail -40 "$build/build.log" >&2; deps_die "build failed (log $build/build.log)"; }
  rm -rf "$prefix"
  (cd "$build" && make "${flags[@]}" install PREFIX="$prefix" >"$build/install.log" 2>&1) ||
    { tail -40 "$build/install.log" >&2; deps_die "install failed"; }
  # The names the environment's MUMPS and hsolve load (DT_NEEDED): libblas.so.3, liblapack.so.3,
  # libcblas.so.3 are links to libopenblas in conda-forge; libopenblas.so.0 is installed already.
  local so
  so="$(cd "$prefix/lib" && ls libopenblas*-r$ver.so | head -1)"
  for l in libblas.so.3 liblapack.so.3 libcblas.so.3; do ln -sf "$so" "$prefix/lib/$l"; done
  deps_check_glibc "$prefix/lib/$so"
  deps_info "$prefix" "$tarball" "$sha" "${flags[@]}"
  grep -E "^(Core|Library|Architecture)" "$build/Makefile.conf_last" 2>/dev/null >>"$prefix/BUILD-INFO.txt" || true
  echo "done: $prefix"
}
main "$@"
exit
