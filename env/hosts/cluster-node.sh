# Cluster nodes gpu01 / gpu02 / c001: 2x Xeon Silver 4314 (2 x 16 cores, AVX-512), 251 GB, CentOS 7
# (glibc 2.17), no internet. The environments were created offline from the development server's
# conda package cache (conda list --explicit --md5 -> file:// URLs; see docs/DEPENDENCIES.md):
#   conda create -p $HOME/envs/os2026 --offline --file os2026.local.explicit.txt   (same for os2026-ref)
# Sourced by env/activate.sh through the per-host links env/hosts/<host>.sh.
export HS_CONDA_BIN="${HS_CONDA_BIN:-/share/apps/anaconda3/bin/conda}"
export HS_CONDA_ENV="${HS_CONDA_ENV:-$HOME/envs/os2026}"
export HS_REF_ENV="${HS_REF_ENV:-$HOME/envs/os2026-ref}"   # SLEPc baseline (tools/bench)
export HS_CORES="${HS_CORES:-16-31}"                      # 16 physical cores = NUMA node 1
export HS_MEM_GB="${HS_MEM_GB:-60}"                       # per-run RSS limit (tools/run/guard.py)
export HS_MAX_SHARED_MEM_GB="${HS_MAX_SHARED_MEM_GB:-64}" # total budget for parallel jobs
