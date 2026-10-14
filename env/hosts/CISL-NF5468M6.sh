# Development server (2x Xeon Gold 6336Y, 48 physical cores, 125 GB, shared with other users). Since
# the end of P4 it runs no builds, tests or experiments (docs/DESIGN.md §6); these are the limits
# that applied while it did. Sourced by env/activate.sh (short hostname).
export HS_CORES="${HS_CORES:-24-30,33-41}"     # 16 physical cores on NUMA node 1
export HS_MEM_GB="${HS_MEM_GB:-30}"            # RSS limit enforced by tools/run/guard.py
