# Source this file to enter the project environment:  source env/activate.sh
# Activates the conda env and sets the resource policy defaults (docs/DESIGN.md §6).
# Per-host settings (conda location, environment prefix, cores, memory limits) live in
# env/hosts/<short hostname>.sh (cluster nodes, the development server); without such a file the
# defaults below apply - on any other machine (a user's, the judges') every CPU this shell may use.
_hs_host_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hosts/$(hostname -s).sh"
# shellcheck disable=SC1090
[ -f "$_hs_host_file" ] && source "$_hs_host_file"
_conda_bin="${HS_CONDA_BIN:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}"
eval "$("$_conda_bin" shell.bash hook)"
conda activate "${HS_CONDA_ENV:-os2026}"
unset _conda_bin _hs_host_file
# Some user profiles prepend other toolchains (e.g. an old cmake); make the env win.
export PATH="$CONDA_PREFIX/bin:$PATH"
# conda compiler activation injects -march=nocona -O2 ...; CMake owns optimisation flags.
export CFLAGS="-pipe" CXXFLAGS="-pipe" CPPFLAGS="" DEBUG_CFLAGS="" DEBUG_CXXFLAGS=""
# guard.py pins runs to HS_CORES; a list naming CPUs this machine lacks would make every guarded
# build and test fail, so the generic default is the shell's own CPU list.
export HS_CORES="${HS_CORES:-$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)}"
export HS_MEM_GB="${HS_MEM_GB:-30}"            # RSS limit enforced by tools/run/guard.py
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" # interactive default; guard.py overrides per run
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
# numpy's MADV_HUGEPAGE triggers failing THP compaction on this machine: first touch of fresh
# arrays becomes ~100x slower (measured by the generator agents). Disable it for every tool.
export NUMPY_MADVISE_HUGEPAGE=0
