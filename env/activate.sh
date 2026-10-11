# Source this file to enter the project environment:  source env/activate.sh
# Activates the conda env and sets the resource policy defaults (docs/DESIGN.md §6).
_conda_bin="$(command -v conda || echo "$HOME/miniconda3/bin/conda")"
eval "$("$_conda_bin" shell.bash hook)"
conda activate os2026
unset _conda_bin
# Some user profiles prepend other toolchains (e.g. an old cmake); make the env win.
export PATH="$CONDA_PREFIX/bin:$PATH"
# conda compiler activation injects -march=nocona -O2 ...; CMake owns optimisation flags.
export CFLAGS="-pipe" CXXFLAGS="-pipe" CPPFLAGS="" DEBUG_CFLAGS="" DEBUG_CXXFLAGS=""
export HS_CORES="${HS_CORES:-24-30,33-41}"     # 16 physical cores on NUMA node 1
export HS_MEM_GB="${HS_MEM_GB:-30}"            # RSS limit enforced by tools/run/guard.py
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" # interactive default; guard.py overrides per run
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
