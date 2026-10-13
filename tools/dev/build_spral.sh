#!/usr/bin/env bash
# Offline, reproducible build of SPRAL (its SSIDS solver: multifrontal LDL^T with inertia, OpenMP CPU
# path + CUDA path) for the GPU experiment G1: bench/gpu_ssids.cpp, results/gpu_g1.md. Experiment
# only - hsolve itself does not use SPRAL (docs/DEPENDENCIES.md).
#
#   tools/dev/build_spral.sh [--cpu-only]
#
# Run it inside the project environment and the resource guard (build/test lane), e.g.
#   tools/dev/cluster_run.sh gpu02 -- python3 tools/run/guard.py --skip-busy-check --timeout 3600 --
#     tools/dev/build_spral.sh                                                  (one command line)
# Sources (the cluster has no internet: download them elsewhere into $SPRAL_SRC_DIR, default
# $HOME/src; the checksums are verified):
#   spral-2025.09.18.tar.gz  https://github.com/ralna/spral/archive/refs/tags/v2025.09.18.tar.gz
#   meson-1.11.1.pyz         https://github.com/mesonbuild/meson/releases/download/1.11.1/meson.pyz
# Toolchain: the conda env's gcc / g++ / gfortran 13.4, ninja, OpenBLAS 0.3.34 (BLAS and LAPACK),
# METIS 5.1 (32-bit idx_t); CUDA $CUDA_HOME (default /usr/local/cuda-12.2) nvcc for sm_70 (V100).
# nvcc's host compiler is $SPRAL_CUDA_HOST_CXX (default: the conda g++; nvcc 12.2 officially stops
# at gcc 12, hence -allow-unsupported-compiler). No hwloc: SSIDS then treats the machine as one
# region of omp_get_max_threads() threads (bench/gpu_ssids.cpp passes its topology explicitly).
# C, C++ and Fortran are compiled with -O3 -march=native, like hsolve (HSOLVE_NATIVE).
# Output: $SPRAL_PREFIX (default $HOME/opt/spral-2025.09.18-cuda12.2, -cpu with --cpu-only):
#   lib/libspral.so (RPATH: the conda lib and CUDA lib64), include/spral*.h, bin/spral_ssids, the C
#   examples in examples/C, BUILDINFO (versions, checksums, flags). The build never touches a GPU:
#   meson's CUDA sanity check program runs with CUDA_VISIBLE_DEVICES="".
set -euo pipefail
SPRAL_VERSION=2025.09.18
SPRAL_SHA256=1358168ac95297049e4fc810e54a16e0c765796cfbaa156e09979e2620b7dae7
MESON_VERSION=1.11.1
MESON_SHA256=05f25d74eab0d1a9b26f1d2fe482677b9c2b9543f974016c8102a5f96b2b73f5
CUDA_ARCH=70

gpu=true
case "${1:-}" in
  "") ;;
  --cpu-only) gpu=false ;;
  *) sed -n '2,25p' "$0" >&2; exit 2 ;;
esac
[ -n "${CONDA_PREFIX:-}" ] || { echo "build_spral: run inside the project environment (env/run.sh)" >&2; exit 2; }
src_dir="${SPRAL_SRC_DIR:-$HOME/src}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.2}"
suffix=$([ "$gpu" = true ] && echo "cuda$(basename "$cuda_home" | sed 's/^cuda-//')" || echo cpu)
prefix="${SPRAL_PREFIX:-$HOME/opt/spral-$SPRAL_VERSION-$suffix}"
build_root="${SPRAL_BUILD_DIR:-$HOME/build/spral-$SPRAL_VERSION-$suffix}"
tarball="$src_dir/spral-$SPRAL_VERSION.tar.gz"
meson_pyz="$src_dir/meson-$MESON_VERSION.pyz"

echo "[1/4] sources"
check() { echo "$2  $1" | sha256sum -c --quiet - || { echo "build_spral: checksum mismatch: $1" >&2; exit 1; }; }
check "$tarball" "$SPRAL_SHA256"
check "$meson_pyz" "$MESON_SHA256"
rm -rf "$build_root"
mkdir -p "$build_root"
tar -xzf "$tarball" -C "$build_root"
srcdir="$build_root/spral-$SPRAL_VERSION"

echo "[2/4] configure (gpu=$gpu)"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export FC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
# env/activate.sh already resets CFLAGS / CXXFLAGS; the conda activation's Fortran flags
# (-march=nocona -O2 ...) would override meson's optimisation the same way.
export FFLAGS="-pipe" FCFLAGS="-pipe"
rpath="-Wl,-rpath,$CONDA_PREFIX/lib"
cuda_opts=()
if [ "$gpu" = true ]; then
  host_cxx="${SPRAL_CUDA_HOST_CXX:-$CXX}"
  export NVCC="$cuda_home/bin/nvcc" CUDA_PATH="$cuda_home"
  # Every nvcc call (including meson's sanity check) gets the host compiler.
  export NVCC_PREPEND_FLAGS="-ccbin=$host_cxx -allow-unsupported-compiler"
  export CUDA_VISIBLE_DEVICES=""
  rpath+=" -Wl,-rpath,$cuda_home/lib64"
  cuda_opts=("-Dcuda_args=-gencode=arch=compute_$CUDA_ARCH,code=sm_$CUDA_ARCH")
fi
export LDFLAGS="${LDFLAGS:-} $rpath"
native=-march=native
python3 "$meson_pyz" setup "$build_root/build" "$srcdir" \
  --prefix="$prefix" --buildtype=release --wrap-mode=nodownload \
  -Dgpu="$gpu" -Dexamples=true -Dtests=false -Dmodules=false -Dopenmp=true \
  -Dlibblas=openblas -Dlibblas_path="$CONDA_PREFIX/lib" -Dlibblas_include="$CONDA_PREFIX/include" \
  -Dliblapack=openblas -Dliblapack_path="$CONDA_PREFIX/lib" \
  -Dlibmetis=metis -Dlibmetis_path="$CONDA_PREFIX/lib" -Dlibmetis_version=5 \
  -Dlibhwloc=hwloc-not-used \
  -Dc_args="$native" -Dcpp_args="$native" -Dfortran_args="$native" \
  "${cuda_opts[@]}" | tee "$build_root/configure.log" | grep -E "Compiler for|Dependency|Library|Message|WARNING|ERROR" || true
grep -q "^Build targets in project" "$build_root/configure.log" || { echo "build_spral: meson setup failed" >&2; exit 1; }
if [ "$gpu" = true ] && ! grep -q -- "-DHAVE_NVCC" "$build_root/build/build.ninja"; then
  echo "build_spral: CUDA path not enabled (nvcc or the cuda dependency was not found)" >&2; exit 1
fi

echo "[3/4] build (${HS_JOBS:-4} jobs)"
ninja -C "$build_root/build" -j "${HS_JOBS:-4}" > "$build_root/build.log" 2>&1 || { tail -40 "$build_root/build.log" >&2; exit 1; }

echo "[4/4] install -> $prefix"
rm -rf "$prefix"
python3 "$meson_pyz" install -C "$build_root/build" > "$build_root/install.log" 2>&1 || { tail -20 "$build_root/install.log" >&2; exit 1; }
{
  echo "spral $SPRAL_VERSION sha256 $SPRAL_SHA256"
  echo "meson $MESON_VERSION sha256 $MESON_SHA256"
  echo "gpu $gpu  cuda_home $cuda_home  arch sm_$CUDA_ARCH  host_cxx ${host_cxx:-none}"
  "$CXX" --version | head -1
  "$FC" --version | head -1
  [ "$gpu" = true ] && "$NVCC" --version | tail -2 | head -1
  echo "flags $native -O3 (meson release)  LDFLAGS $LDFLAGS"
  echo "built $(date -Is) on $(hostname -s) by tools/dev/build_spral.sh"
} > "$prefix/BUILDINFO"
cat "$prefix/BUILDINFO"
ldd "$prefix/lib/libspral.so" | grep -E "cublas|cudart|openblas|metis|gfortran|gomp|not found" || true
