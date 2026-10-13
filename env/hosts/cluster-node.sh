# Cluster nodes gpu01 / gpu02 / c001: 2x Xeon Silver 4314 (2 x 16 cores, AVX-512), 251 GB, CentOS 7
# (glibc 2.17), no internet. The environments were created offline from the development server's
# conda package cache (conda list --explicit --md5 -> file:// URLs; see docs/DEPENDENCIES.md):
#   conda create -p $HOME/envs/os2026 --offline --file os2026.local.explicit.txt   (same for os2026-ref)
# Sourced by env/activate.sh through the per-host links env/hosts/<host>.sh.
export HS_CONDA_BIN="${HS_CONDA_BIN:-/share/apps/anaconda3/bin/conda}"
export HS_CONDA_ENV="${HS_CONDA_ENV:-$HOME/envs/os2026}"
export HS_REF_ENV="${HS_REF_ENV:-$HOME/envs/os2026-ref}"   # SLEPc baseline (tools/bench)
# Scheduling (docs/DESIGN.md §6, tools/run/guard.py): one run at a time in the measurement slot,
# sized like the judging machine (16 cores, 32 GiB); builds, unit tests and smoke runs share a small
# lane on the other socket. With two callers per node that is at most 24 of the 32 physical cores
# and ~56 GiB; the rest of the node stays free for other work.
export HS_MEASURE_CORES="${HS_MEASURE_CORES:-16-31}"      # NUMA node 1: guard --exclusive only
export HS_MEM_CAP_GB="${HS_MEM_CAP_GB:-32}"               # per-run RSS ceiling (any run)
export HS_SLOT_MAX_S="${HS_SLOT_MAX_S:-2700}"             # longest run holding the slot
export HS_LOCK_WAIT_S="${HS_LOCK_WAIT_S:-300}"            # then exit 75 and name the holder
export HS_CORES="${HS_CORES:-8-15}"                       # build/test lane: 8 cores of NUMA node 0
export HS_JOBS="${HS_JOBS:-4}"                            # build parallelism per caller
export HS_MEM_GB="${HS_MEM_GB:-12}"                       # default RSS limit outside the slot
export HS_MAX_SHARED_MEM_GB="${HS_MAX_SHARED_MEM_GB:-12}" # RSS ceiling outside the slot
