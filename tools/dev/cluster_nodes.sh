#!/usr/bin/env bash
# Which cluster nodes are free for hsolve runs right now (docs/DESIGN.md §6), and pick one.
#
#   tools/dev/cluster_nodes.sh [--pick] [HOST...]
#
# Hosts default to $HS_CLUSTER_HOSTS or "gpu01 gpu02 c001 c002 c003 c004 c005" (identical nodes:
# 2x Xeon Silver 4314, 251 GB, shared home). Per node, sampled over one second: our measurement
# slot (lock free, or its holder), busy logical CPUs on the slot socket (cores 16-31 and their
# hyperthreads) and on the whole node, and MemAvailable. A node is idle for a timed run when the slot
# lock is free, no slot CPU is busy and at least 64 GiB are available. --pick prints the idle node
# with the fewest busy CPUs (exit status 1 if there is none). c001-c005 are shared with other users:
# use them only while they are idle; their jobs keep priority.
set -euo pipefail
pick=0
if [ "${1:-}" = "--pick" ]; then pick=1; shift; fi
hosts=("$@")
[ ${#hosts[@]} -gt 0 ] || read -r -a hosts <<< "${HS_CLUSTER_HOSTS:-gpu01 gpu02 c001 c002 c003 c004 c005}"
# shellcheck disable=SC2016
probe='set -e
lock=/tmp/hsolve-guard-exclusive.lock
if flock -n "$lock" true 2>/dev/null; then slot=free; else slot="held:$(head -c 200 "$lock.owner" 2>/dev/null | tr -d "\n ")"; fi
cpus=$(for c in $(seq 16 31); do cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list; done | tr "\n" ,)
a=$(grep "^cpu[0-9]" /proc/stat); sleep 1; b=$(grep "^cpu[0-9]" /proc/stat)
busy=$(paste <(echo "$a") <(echo "$b") | awk -v slot="$cpus" -F"[ \t]+" '"'"'
  BEGIN { n = split(slot, parts, ","); for (i = 1; i <= n; i++) {
            if (parts[i] ~ /-/) { split(parts[i], r, "-"); for (c = r[1]; c <= r[2]; c++) s[c] = 1 }
            else if (parts[i] != "") s[parts[i]] = 1 } }
  { m = NF / 2; ta = 0; tb = 0; for (i = 2; i <= 9; i++) { ta += $i; tb += $(i + m) }
    idle = ($(5 + m) + $(6 + m)) - ($5 + $6); tot = tb - ta; cpu = substr($1, 4)
    if (tot > 0 && 100 * (1 - idle / tot) > 50) { all++; if (cpu in s) sl++ } }
  END { printf "%d %d", sl, all }'"'"')
mem=$(awk "/MemAvailable/ { print int(\$2 / 1048576) }" /proc/meminfo)
echo "$busy $mem $slot"'
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for h in "${hosts[@]}"; do
  ( timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "$probe" > "$tmp/$h" 2>/dev/null || echo "unreachable" > "$tmp/$h" ) &
done
wait
best="" best_busy=1000000
[ "$pick" = 1 ] || printf '%-6s %-9s %9s %9s %8s  %s\n' host verdict slot_busy node_busy avail_GB slot
for h in "${hosts[@]}"; do
  read -r sl all mem slot < "$tmp/$h" || true
  if [ "${sl:-}" = unreachable ] || [ -z "${slot:-}" ]; then
    [ "$pick" = 1 ] || printf '%-6s %-9s\n' "$h" unreachable
    continue
  fi
  verdict=busy
  if [ "$slot" = free ] && [ "$sl" -eq 0 ] && [ "$mem" -ge 64 ]; then
    verdict=idle
    if [ "$all" -lt "$best_busy" ]; then best=$h best_busy=$all; fi
  fi
  [ "$pick" = 1 ] || printf '%-6s %-9s %9s %9s %8s  %s\n' "$h" "$verdict" "$sl/32" "$all/64" "$mem" "$slot"
done
if [ "$pick" = 1 ]; then
  [ -n "$best" ] || { echo "cluster_nodes: no idle node" >&2; exit 1; }
  echo "$best"
fi
