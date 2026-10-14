#!/usr/bin/env bash
# Build sequential MUMPS 5.8.2 (OpenMP) from the official source tarball against the project
# environment's METIS, SCOTCH, OpenBLAS and gfortran (P9 "factspeed"; results/p9_factspeed.md,
# docs/BLAS.md section 9).
#
#   env/run.sh tools/dev/build_mumps.sh NAME [--isa ISA] [--opt LEVEL] [--gemmt] [--patch FILE]...
#
# The default build (tools/dev/build.sh) does not call this directly but through
# tools/dev/mumps_cache.sh, which builds the default configuration (these defaults plus
# patches/mumps-5.8.2-l0-blocking.patch) once per cache key and reuses it. Use this script itself
# for other configurations (link one with HS_MUMPS=<prefix> tools/dev/build.sh).
#
#   --isa   -march of every MUMPS object. Default nocona (SSE3; with -mtune=haswell
#            -ftree-vectorize: conda-forge's own flags) - portable to any x86-64 CPU, and the
#            configuration the P9 gates measured (results/p9_factspeed.md §3.3-4.5). Opt-in, NOT
#            portable, ~0-0.4% faster in the P9 screens: x86-64-v3 needs AVX2, FMA, BMI1/2,
#            LZCNT, F16C, MOVBE (Intel Core / Xeon from Haswell, AMD from Excavator / Zen; NOT
#            Sandy / Ivy Bridge Xeons, NOT the many Atom / Celeron / Pentium parts without AVX2
#            sold well after 2015); skylake-avx512 / native need the build CPU's ISA. On a CPU
#            without them the libraries die with SIGILL in the first factorization.
#   --opt    optimisation level (default O2, conda-forge's; O3 is opt-in, 1.000-1.005x).
#   --gemmt  -DGEMMT_AVAILABLE: MUMPS' symmetric (LDL^T) updates call the BLAS extension dgemmt
#            (OpenBLAS >= 0.3.22 provides it) instead of blocks of dgemm (MUMPS' INSTALL file).
#   --patch  a patch (git diff format, -p1 against the tarball's top directory) applied after
#            unpacking; several allowed, in order.
#
# Installs into $HS_DEPS/mumps-5.8.2-NAME (HS_DEPS default $HOME/deps, tools/deps/common.sh):
#   lib/lib{s,d,c,z}mumps_seq.so, libmumps_common_seq.so, libpord_seq.so, libmpiseq_seq.so - the
#     names AND sonames of conda-forge's mumps-seq, with RUNPATH $ORIGIN, so that they replace the
#     environment's libraries at run time (LD_PRELOAD of the seven files, or a hsolve linked against
#     them: cmake -DHSOLVE_MUMPS_ROOT=<prefix>, CMakeLists.txt), and the static archives;
#   include/ (the tarball's headers, mumps_int_def.h, libseq's mpi*.h under include/mumps_seq);
#   BUILD-INFO.txt (tarball, sha256, flags, compiler, host).
# The libraries are checked to need no glibc symbol newer than 2.17 (the cluster nodes').
#
# Source: third_party/mumps/MUMPS_5.8.2.tar.gz in this repository - the official
# https://mumps-solver.org/MUMPS_5.8.2.tar.gz unchanged (sha256 below, the same as conda-forge's
# mumps recipe; checked before every unpack) - so that an offline checkout builds (the cluster
# nodes have no internet). License: CeCILL-C (third_party/mumps/README.md, THIRD_PARTY_NOTICES).
# Patches are applied with patch(1), or with `git apply` where patch is not installed.
# Build time ~1.5 min with HS_JOBS=4 (build lane: run through guard.py).
# Conda-forge's build (recipe mumps 5.8.2, build 3): Makefile.inc from their Makefile.conda.SEQ,
# FFLAGS = the toolchain's "-march=nocona -mtune=haswell -ftree-vectorize -fPIC
# -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe" + -fallow-argument-mismatch,
# -DALLOW_NON_INIT, no GEMMT; orderings -Dmetis -Dpord -Dscotch. Their patches concern flang and
# METIS < 5.1.0's option numbering (MUMPS 5.8.2 reads METIS_OPTION_NUMBERING from metis.h itself);
# none is needed with gfortran and the environment's METIS 5.1.0.
main() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=tools/deps/common.sh
  source "$here/../deps/common.sh"
  deps_require_env
  [ $# -ge 1 ] || deps_die "usage: build_mumps.sh NAME [--isa ISA] [--opt O2|O3] [--gemmt] [--patch FILE]..."
  local name="$1"
  shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || deps_die "bad NAME '$name'"
  local isa=nocona opt=O2 gemmt=0 patches=()  # portable defaults (conda-forge's flags; header)
  while [ $# -gt 0 ]; do
    case "$1" in
      --isa) isa="$2"; shift 2 ;;
      --opt) opt="$2"; shift 2 ;;
      --gemmt) gemmt=1; shift ;;
      --patch) patches+=("$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2 ;;
      *) deps_die "unknown option $1" ;;
    esac
  done
  local ver=5.8.2
  local sha=eb515aa688e6dbab414bb6e889ff4c8b23f1691a843c68da5230a33ac4db7039
  local tarball="$here/../../third_party/mumps/MUMPS_$ver.tar.gz"
  local build="$HS_DEPS_BUILD/mumps-$ver-$name"
  local prefix="$HS_DEPS/mumps-$ver-$name"
  deps_check "$tarball" "$sha"
  deps_unpack "$tarball" "$build"
  local p
  for p in ${patches[@]+"${patches[@]}"}; do
    if command -v patch >/dev/null; then
      (cd "$build" && patch -p1 --forward < "$p") || deps_die "patch $p does not apply"
    else  # git apply as a plain patch: the ceiling keeps it from finding an enclosing repository
      (cd "$build" && GIT_CEILING_DIRECTORIES="$(dirname "$build")" git apply -p1 "$p") ||
        deps_die "patch $p does not apply (git apply; patch(1) is not installed)"
    fi
  done

  local tune=""
  [ "$isa" = nocona ] && tune="-mtune=haswell -ftree-vectorize"
  local common="-$opt -march=$isa $tune -fPIC -pipe"
  local defs="-DALLOW_NON_INIT"
  [ "$gemmt" = 1 ] && defs="$defs -DGEMMT_AVAILABLE"
  # gfortran pre-includes the sysroot's finclude/math-vector-fortran.h (glibc 2.28: libmvec), so
  # vectorised loops over LOG etc. call _ZGV*_logf@GLIBC_2.22 - not on glibc 2.17 (cluster nodes),
  # and the libraries would not load. -nostdinc drops that pre-include; the compiler's own module
  # directory (omp_lib.mod) is added back explicitly. MUMPS' Fortran includes come through -I.
  local fnostd
  fnostd="-nostdinc -I$("$FC" -print-file-name=finclude)"
  local compat
  compat="$(deps_glibc_compat "$build")"
  # conda's LDFLAGS minus --gc-sections and --as-needed (the whole archives must stay whole).
  local ldflags="${LDFLAGS//-Wl,--gc-sections/}"
  ldflags="${ldflags//-Wl,--as-needed/}"
  cat > "$build/Makefile.inc" <<EOF
# Generated by tools/dev/build_mumps.sh ($name)
PLAT = _seq
LIBEXT = .a
LIBEXT_SHARED = .so
SONAME = -soname
SHARED_OPT = -shared
FPIC_OPT = -fPIC
OUTC = -o
OUTF = -o
RM = /bin/rm -f
CC = $CC
FC = $FC
FL = $FC
AR = ar vr \$(NOTHING_)  # keeps the space MUMPS needs before the archive name
RANLIB = ranlib
ISCOTCH = -I$CONDA_PREFIX/include
LSCOTCH = -L$CONDA_PREFIX/lib -lesmumps -lscotch -lscotcherr
IMETIS = -I$CONDA_PREFIX/include
LMETIS = -L$CONDA_PREFIX/lib -lmetis
LPORDDIR = \$(topdir)/PORD/lib/
IPORD = -I\$(topdir)/PORD/include/
LPORD = -L\$(LPORDDIR) -lpord\$(PLAT)
ORDERINGSF = -Dmetis -Dpord -Dscotch
ORDERINGSC = \$(ORDERINGSF)
LORDERINGS = \$(LMETIS) \$(LPORD) \$(LSCOTCH)
IORDERINGSF = \$(ISCOTCH)
IORDERINGSC = \$(IMETIS) \$(IPORD) \$(ISCOTCH)
LAPACK = -L$CONDA_PREFIX/lib -llapack -lblas
INCSEQ = -I\$(topdir)/libseq
LIBSEQ = \$(LAPACK) -L\$(topdir)/libseq -lmpiseq\$(PLAT)
LIBBLAS = -L$CONDA_PREFIX/lib -lblas
LIBOTHERS = -lpthread
CDEFS = -DAdd_
OPTF = $common -fopenmp -fallow-argument-mismatch $fnostd $defs
OPTC = $common -fopenmp -I.
OPTL = -fopenmp
INCS = \$(INCSEQ)
LIBS = \$(LIBSEQ)
LIBSEQNEEDED = libseqneeded
EOF
  echo "mumps $ver -> $prefix: $common $defs"
  (cd "$build" && make -j"$(deps_jobs)" all >"$build/build.log" 2>&1) ||
    { tail -40 "$build/build.log" >&2; deps_die "build failed (log $build/build.log)"; }

  # Shared libraries with conda-forge's names and sonames from the static archives (MUMPS' own
  # shared targets set no soname on most of them; conda-forge patches them with patchelf).
  rm -rf "$prefix"
  mkdir -p "$prefix/lib" "$prefix/include/mumps_seq"
  local lib="$build/lib" seq="$build/libseq" pord="$build/PORD/lib"
  # $ORIGIN first in the (DT_)RPATH: libdmumps_seq.so must find its own libmumps_common_seq.so
  # ahead of the environment's ($CONDA_PREFIX/lib comes with conda's LDFLAGS).
  local link=("$FC" -shared -fopenmp "-Wl,-rpath,\$ORIGIN" $ldflags $compat)
  local so
  shlib() {  # shlib NAME ARCHIVE [LIBS...]
    local n="$1" a="$2"
    shift 2
    "${link[@]}" "-Wl,-soname,$n" -o "$prefix/lib/$n" -Wl,--whole-archive "$a" -Wl,--no-whole-archive "$@" ||
      deps_die "link of $n failed"
  }
  shlib libpord_seq.so "$pord/libpord_seq.a"
  shlib libmpiseq_seq.so "$seq/libmpiseq_seq.a"
  shlib libmumps_common_seq.so "$lib/libmumps_common_seq.a" -L"$prefix/lib" -lpord_seq -lmpiseq_seq \
    -L"$CONDA_PREFIX/lib" -lesmumps -lscotch -lscotcherr -lmetis -llapack -lblas -lpthread
  local a
  for a in s d c z; do
    shlib "lib${a}mumps_seq.so" "$lib/lib${a}mumps_seq.a" -L"$prefix/lib" -lmumps_common_seq -lpord_seq \
      -lmpiseq_seq -L"$CONDA_PREFIX/lib" -lesmumps -lscotch -lscotcherr -lmetis -llapack -lblas -lpthread
  done
  ln -s libmpiseq_seq.so "$prefix/lib/libmpiseq.so"
  local ar
  for ar in "$lib"/*.a "$pord/libpord_seq.a" "$seq/libmpiseq_seq.a"; do
    [ -e "$prefix/lib/$(basename "$ar")" ] || cp "$ar" "$prefix/lib/"
  done
  cp "$build"/include/*.h "$prefix/include/"
  cp "$seq"/mpi*.h "$prefix/include/mumps_seq/"
  for so in "$prefix"/lib/*.so; do
    [ -L "$so" ] || deps_check_glibc "$so"
  done
  deps_info "$prefix" "$tarball" "$sha" "isa=$isa opt=$opt gemmt=$gemmt OPTF=$common $defs patches=${patches[*]+${patches[*]}}"
  echo "fc: $($FC --version | head -1)" >>"$prefix/BUILD-INFO.txt"
  echo "done: $prefix"
}
main "$@"
exit
