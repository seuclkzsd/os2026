#!/usr/bin/env bash
# Build LIBXSMM 1.17 from the pinned tag tarball (docs/BLAS.md).
#
#   env/run.sh tools/deps/build_libxsmm.sh NAME [MAKEVAR=VALUE ...]
#
# Installs into $HS_DEPS/libxsmm-1.17-NAME: lib/libxsmm.so (JIT code generator: AVX-512 / AVX2 /
# SSE kernels chosen by CPUID at run time) and lib/libxsmmext.so, the BLAS call wrapper - as
# LD_PRELOAD=libxsmmext.so it intercepts dgemm_ (WRAP=1, the default here; WRAP=3 also dgemv_)
# and runs problems up to the THRESHOLD (M*N*K, default 64^3 in 1.17) on JIT kernels, larger ones
# on the next BLAS (dlsym(RTLD_NEXT)). Built shared (STATIC=0), thread-safe, without OpenMP and
# without a BLAS dependency (BLAS=0: the fallback is found at run time). 1.17 is the last release
# with this wrapper and the classic small-GEMM interface (2.x replaced it by TPP kernels).
# Source: https://github.com/libxsmm/libxsmm/archive/refs/tags/1.17.tar.gz
# (sha256 = Spack's libxsmm package). License BSD-3-Clause.
main() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=tools/deps/common.sh
  source "$here/common.sh"
  deps_require_env
  [ $# -ge 1 ] || deps_die "usage: build_libxsmm.sh NAME [MAKEVAR=VALUE ...]"
  local name="$1"
  shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || deps_die "bad NAME '$name'"
  local ver=1.17
  local sha=8b642127880e92e8a75400125307724635ecdf4020ca4481e5efe7640451bb92
  local tarball="$HS_DEPS_SRC/libxsmm-$ver.tar.gz"
  local build="$HS_DEPS_BUILD/libxsmm-$ver-$name"
  local prefix="$HS_DEPS/libxsmm-$ver-$name"
  deps_check "$tarball" "$sha"
  deps_unpack "$tarball" "$build"
  local flags=(STATIC=0 BLAS=0 WRAP=1 OMP=0 "CC=$CC" "CXX=$CXX" "FC=$FC" FORTRAN=0 "$@")
  echo "libxsmm $ver -> $prefix: ${flags[*]}"
  rm -rf "$prefix"
  (cd "$build" && make -j"$(deps_jobs)" "${flags[@]}" PREFIX="$prefix" >"$build/build.log" 2>&1 &&
     make "${flags[@]}" PREFIX="$prefix" install-minimal >"$build/install.log" 2>&1) ||
    { tail -40 "$build"/*.log >&2; deps_die "build failed (logs in $build)"; }
  deps_info "$prefix" "$tarball" "$sha" "${flags[@]}"
  echo "done: $prefix"
}
main "$@"
exit
