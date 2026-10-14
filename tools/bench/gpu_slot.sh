#!/usr/bin/env bash
# Run a command in the node's measurement slot with exactly one GPU visible, after checking that
# this GPU is idle (P10 "gpu"; results/p10_gpu.md). On the shared cluster nodes most GPUs belong to
# other jobs: the command sees only the named GPU (CUDA_VISIBLE_DEVICES), and it does not start
# when another process uses that GPU (nvidia-smi compute apps on its bus id, or more than
# HS_GPU_IDLE_MIB of its memory in use).
#
#   tools/bench/gpu_slot.sh GPU TIMEOUT_S [guard options...] -- CMD [ARGS...]
#
# e.g. tools/bench/gpu_slot.sh 7 900 -- build/release/bench_gpu_dgemm info
# The guard options (e.g. --lock-wait 600, --mem-gb 28) go to tools/run/guard.py, which runs CMD
# with --exclusive (the slot: cores 16-31, <= 32 GiB, one run at a time) and --timeout TIMEOUT_S.
# Run it from the repository root inside the project environment (tools/dev/cluster_run.sh does).
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,13p' "$0" >&2; exit 2; }
gpu="$1"; timeout_s="$2"; shift 2
gopts=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do gopts+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "gpu_slot: missing -- before the command" >&2; exit 2; }
shift
idle_mib="${HS_GPU_IDLE_MIB:-64}"
bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$gpu")"
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
apps="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i "^$bus" || true)"
if [ -n "$apps" ] || [ "${used// /}" -gt "$idle_mib" ]; then
  echo "gpu_slot: GPU $gpu ($bus) is in use: ${used} MiB, processes: ${apps:-none}" >&2
  exit 75
fi
echo "gpu_slot: GPU $gpu ($bus) idle (${used// /} MiB used)" >&2
exec env CUDA_VISIBLE_DEVICES="$gpu" python3 tools/run/guard.py --exclusive --timeout "$timeout_s" \
  ${gopts[@]+"${gopts[@]}"} -- "$@"
