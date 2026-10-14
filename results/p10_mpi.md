# P10 "mpi": hsolve over several processes - one per socket

P10 task "mpi" (branch `p10mpi` from p9-merge 90e7483). Question: can multi-process (MPI) execution
make hsolve faster at the judging budget of 16 cores per run, and where? P9 measured that on ONE
16-core socket MPI loses (`results/p9_mpi.md`: MPI MUMPS 2 x 8 .. 16 x 1 = 0.85 .. 0.59x of 1 x 16,
two concurrent 8-thread counts 0.92x): every factorization is limited by the socket's memory
bandwidth. A judging machine with 16 cores may also be two sockets of 8 - then a process per socket
has its own memory bandwidth. This report measures that first (section 2), implements the layouts
that pay as an optional multi-process mode of hsolve (section 3; `docs/DESIGN.md` §3.9,
`docs/PARALLEL.md`) and measures it end to end (section 4).

__ANSWERS__

## 1. Setup

* **Node.** c005 (2x Xeon Silver 4314: two sockets = NUMA nodes 0 (cores 0-15) and 1 (16-31),
  distance 20 vs 10, AVX-512, 251 GB; no InfiniBand), idle by `tools/dev/cluster_nodes.sh` before
  every batch, every timed run under its own `guard.py --exclusive` (the node's measurement-slot
  lock: one run at a time) with `--mem-gb 32` and the cores of its placement:
  * **one socket**: `--cores 16-31` (the measurement slot, NUMA node 1);
  * **two sockets**: `--cores 8-15,16-23` (8 cores of each socket; the guard's busy check covers
    them and their hyperthreads, so nothing else ran on 8-15 meanwhile - builds and evaluations
    ran before or after the timed batches).
  Every run uses at most 16 cores in total; hyperthreads (32-63) are never used.
* **Layouts** (`tools/bench/mpi_socket_run.sh`, one run each; `tools/bench/mpi_layouts.py` runs
  them interleaved - the layout order rotates per repetition and case - and tabulates):
  `1x16` one process, 16 threads (unbound inside the cores, as hsolve runs); `1x16i` the same under
  `numactl --interleave=all`; `1x8` one process on the first 8 cores (the other 8 idle); `2x8` two
  independent processes, one per half of the cores (per socket on two sockets), started together;
  `2x8m` one MPI job of 2 ranks x 8 threads (MPI MUMPS); `1+2x8` hsolve's multi-process mode
  (section 3) through its launcher: the leader on all 16 cores, a helper on each half.
* **Builds.** hsolve as of each section (`tools/dev/build.sh release`: conda gcc 13.4, -O3, BLAS hook
  linked, route all; sequential MUMPS 5.8.2 of conda-forge, OpenBLAS 0.3.34 OpenMP); `hsolve-mpi`
  links Open MPI 5.0.11 of the os2026-ref environment (`HSOLVE_WITH_MPI=AUTO`). The phase bench is
  P9's `bench/mpi_factor_bench.c` (built by `tools/bench/mpi_build.sh` against the sequential and
  the MPI MUMPS 5.8.2), with hsolve's MUMPS controls, hsolve's BLAS hook preloaded, on K(sigma)
  of each case (`tools/bench/mpi_dump_kernel.py`) and hsolve's own pivot order
  (`bench_factspeed_factor --ordering 10 --perm-out`, its nested dissection; AMD for the 2-D
  cav2d-rect-pert-1e5, which hsolve's adaptive ordering picks there).
* **Cases.** 1e6: syn-mem27-100 (HEP, n = 1e6, sigma = 0, k = 12; 6.73e12 flops with hsolve's order),
  yee-pec-box-1e6 (GHEP, n = 1027733, k = 10; 5.44e12 flops) from `$HOME/p7_p7-membudget/data`;
  1e5: syn-mem27-46 (HEP, 6.7e10 flops), yee-pec-box-1e5 (GHEP, 6.0e10), ss-qa8fk-qa8fm (GHEP,
  1.8e10), cav2d-rect-pert-1e5 (2-D GHEP, AMD, 1.7e8), wg-square-1e5 (GHIEP, the waveguide); intervals
  (spectrum slicing, complete references): syn-mem27-46 [0.1, 0.3] (655 eigenvalues),
  yee-pec-box-1e5 [40, 400] (108), yee-pec-cube-32 [40, 400] (271).
* **Validity.** Every hsolve run is judged by tools/eval at the case tolerance against its target
  (`mpi_layouts.py eval`: exit 0, success claimed, evaluation pass - backward errors and
  completeness against the complete reference); bench runs check INFOG(12) against the exact
  inertia of the complete reference (1e6, syn-mem27-46, yee-pec-box-1e5).

## 2. Measure first: what a process per socket buys (screening, 3 interleaved repetitions)

### 2.1 Phases

`mpi_layouts.py run --kind bench` (tags a1e6c, a1e6, a1e5). Per layout: one inertia-only count
(ICNTL(31) = 1, the certificate's), the **pair** of counts a nearest certificate needs (two counts
one after the other = 2 x one count; for 2x8 the two run at the same time, one per process, and
the pair takes the slower one), one numeric factorization keeping the factors, one 16-RHS block
solve; medians [min-max] in seconds, x = speedup over `one:1x16` (hsolve's layout on one socket).
Every count reproduced the exact inertia.

**1e6** (a1e6c: counts only; a1e6: the rest):

| case | layout | one count | count pair | x | factorization | x | 16-RHS solve | x |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | one:1x16 | 21.59 | 43.19 [43.18-43.36] | 1.00 | 22.94 [22.93-23.01] | 1.00 | 0.842 | 1.00 |
| syn-mem27-100 | one:1x8 | 26.96 | 53.93 | 0.80 | 28.24 | 0.81 | 0.934 | 0.90 |
| syn-mem27-100 | one:2x8 (two counts at once on one socket) | 47.95 | 47.95 [47.75-48.00] | 0.90 | - | - | - | - |
| syn-mem27-100 | two:1x16 | 17.33 | 34.65 [34.04-36.47] | 1.25 | 18.07 [17.85-18.98] | 1.27 | 0.879 | 0.96 |
| syn-mem27-100 | two:1x16i | 17.07 | 34.14 | 1.27 | 18.08 | 1.27 | 0.808 | 1.04 |
| syn-mem27-100 | **two:2x8** (one count per socket at once) | 27.07 | **27.08** [27.07-27.11] | **1.59** | - | - | - | - |
| yee-pec-box-1e6 | one:1x16 | 17.83 | 35.67 [35.55-35.71] | 1.00 | 18.91 [18.91-18.95] | 1.00 | 0.791 | 1.00 |
| yee-pec-box-1e6 | one:1x8 | 22.37 | 44.75 | 0.79 | 23.66 | 0.80 | 0.849 | 0.93 |
| yee-pec-box-1e6 | one:2x8 | 39.50 | 39.51 [39.50-39.60] | 0.90 | - | - | - | - |
| yee-pec-box-1e6 | two:1x16 | 14.87 | 29.74 [28.74-31.27] | 1.20 | 15.97 [15.87-16.04] | 1.18 | 0.896 | 0.88 |
| yee-pec-box-1e6 | two:1x16i | 14.86 | 29.72 | 1.19 | 15.71 | 1.20 | 0.804 | 0.98 |
| yee-pec-box-1e6 | **two:2x8** | 22.45 | **22.47** [22.44-22.48] | **1.59** | - | - | - | - |

**1e5** (a1e5; pair / factorization / solve, x over one:1x16):

| case | one:1x16 | one:1x8 | one:2x8 | two:1x16 | two:1x16i | **two:2x8** |
|---|---|---|---|---|---|---|
| syn-mem27-46 pair | 0.846 | 1.013 (0.83) | 0.804 (1.05) | 1.221 (0.69) | 1.080 (0.78) | **0.521 (1.62)** |
| syn-mem27-46 factorization / solve | 0.462 / 0.054 | 0.548 / 0.053 | 0.911 / 0.072 (both at once) | 0.570 / 0.062 | 0.508 / 0.058 | 0.551 / 0.054 (both at once) |
| yee-pec-box-1e5 pair | 0.852 | 0.967 (0.88) | 0.761 (1.12) | 1.116 (0.76) | 1.064 (0.80) | **0.503 (1.70)** |
| yee-pec-box-1e5 factorization / solve | 0.450 / 0.054 | 0.518 / 0.054 | 0.858 / 0.073 | 0.491 / 0.063 | 0.513 / 0.062 | 0.519 / 0.054 |
| ss-qa8fk-qa8fm pair | 0.374 | 0.411 (0.91) | 0.293 (1.28) | 0.595 (0.63) | 0.482 (0.78) | **0.228 (1.64)** |
| ss-qa8fk-qa8fm factorization / solve | 0.198 / 0.032 | 0.209 / 0.030 | 0.328 / 0.038 | 0.238 / 0.037 | 0.244 / 0.037 | 0.213 / 0.030 |
| cav2d-rect-pert-1e5 pair | 0.090 | 0.104 (0.87) | 0.066 (1.36) | 0.170 (0.53) | 0.114 (0.79) | **0.072 (1.26)** |
| cav2d-rect-pert-1e5 factorization / solve | 0.036 / 0.018 | 0.047 / 0.029 | 0.053 / 0.032 | 0.052 / 0.026 | 0.064 / 0.023 | 0.050 / 0.029 |

What the phases say:

* **Two processes, one per socket, do not slow each other down.** Each of two concurrent 8-thread
  counts on its own socket takes what one 8-thread count takes alone (1e6: 27.07 vs 26.96 s,
  22.45 vs 22.37 s; 1e5: 0.517 vs 0.506 s, 0.489 vs 0.483 s), and two concurrent factorizations
  equal one alone (syn-mem27-46 0.551 vs 0.548 s): each socket has its own memory bandwidth. On ONE
  socket two concurrent processes share it and each takes 1.6-1.8x longer (P9: 0.92x for the pair;
  here 0.90x at 1e6, 1.05-1.36x at 1e5 where a single count does not saturate the socket).
* **Hence the certificate's two counts pay most**: the pair at once on two sockets takes 1.59x less
  than hsolve's pair on one socket (1e6) and 1.62-1.70x less at 1e5 - and against the fairer
  baseline on a two-socket machine, one process spreading its 16 threads over both sockets
  (`two:1x16`), 1.28x (syn-mem27-100: 34.65 -> 27.08 s), 1.32x (yee-pec-box-1e6: 29.74 -> 22.47 s)
  and 2.0-2.6x at 1e5.
* **One process over two sockets is good at 1e6, poor at 1e5.** `two:1x16` factorizes 1.18-1.27x
  faster than one socket at 1e6 (it has both sockets' bandwidth) but 0.70-0.92x at 1e5, and its
  counts are worse still (0.53-0.76x; the first numeric phase of a fresh MUMPS instance places its
  memory on one socket by first touch, which half the threads then read remotely); interleaving the
  pages (`1x16i`) helps a little (0.78-0.80x). A single factorization on one socket's 8 cores
  (`1x8`) is 0.80x of one socket's 16 at 1e6, 0.78-0.95x at 1e5.
* **So a process per socket helps where the work splits into independent factorizations** (the two
  counts, spectrum slices), and the leader of the sequential part (analysis, main factorization,
  Krylov) should keep all cores: at 1e6 a leader on one socket's 8 cores would lose 1.6x on its
  main factorization against all 16 spread (28.2 vs 18.1 s).

### 2.2 MPI MUMPS across sockets (task 1b)

`2x8m`: one numeric factorization by the MPI MUMPS of the os2026-ref environment, 2 ranks x 8
threads, each rank pinned to one half of the cores (`tools/bench/mpi_bind.sh`), same matrix, order
and controls (ICNTL(13) = 1: exact inertia), against the sequential MUMPS spread over the same
two-socket cores (`two:1x16`) and against hsolve's layout on one socket (`one:1x16`):

| case | factorization 2x8m s | speedup over two:1x16 | over one:1x16 (one socket, 2x8m on it: speedup) | 16-RHS solve speedup (two sockets) | count speedup (two sockets) |
|---|---|---|---|---|---|
| syn-mem27-100 | 19.06 | **0.95** (18.07) | 1.20 (22.94; on one socket 27.01 s: 0.85) | 1.08 | 0.95 |
| yee-pec-box-1e6 | 14.58 | **1.10** (15.97) | 1.30 (18.91; on one socket 22.89 s: 0.83) | 1.34 | 1.08 |
| syn-mem27-46 | 0.448 | 1.27 (0.570) | 1.03 (0.462; on one socket 0.560 s: 0.82) | 1.35 | 1.20 |
| yee-pec-box-1e5 | 0.375 | 1.31 (0.491) | 1.20 (0.450; on one socket 0.493 s: 0.91) | 1.05 | 1.27 |
| ss-qa8fk-qa8fm | 0.173 | 1.38 (0.238) | 1.14 (0.198; on one socket 0.213 s: 0.93) | 1.28 | 1.23 |
| cav2d-rect-pert-1e5 | 0.046 | 1.13 (0.052) | 0.78 (0.036; on one socket 0.045 s: 0.80) | 0.74 | 1.55 |

(speedup = the baseline's time / 2x8m's; baselines in parentheses, same campaign.)

* Unlike one socket (0.80-0.93x, P9's verdict), across two sockets MPI MUMPS 2x8 beats the
  sequential MUMPS spread over both sockets at 1e5 (1.13-1.38x per factorization) and is mixed at 1e6
  (0.95x syn-mem27-100, 1.10x yee-pec-box-1e6). Against hsolve's sequential MUMPS on ONE socket it
  gains at 1e6 what the second socket's bandwidth gives any layout (1.20 / 1.30x; `two:1x16` 1.27 /
  1.18x) and at 1e5 0.78-1.20x per factorization; the certificate's pair as two MPI counts one after
  the other is 0.77-0.97x of one:1x16's pair at 1e5.
* Against the per-socket concurrency of section 2.1 it loses where hsolve has independent
  factorizations: the certificate's pair as two MPI counts takes 37.2 / 28.2 s (1e6) against 27.1 /
  22.5 s for one count per socket, 1.0 / 0.88 s against 0.52 / 0.50 s at 1e5.
* Not adopted: the main factorization would have to become a distributed MUMPS object (hsolve
  links the sequential MUMPS, whose libmpiseq defines MPI symbols; solves, refinement and the
  Krylov basis would have to follow the distributed factors), for at most 1.10x on one of the two
  1e6 cases and 0.95x on the other. The measured gains are per-socket concurrency of independent
  work, which needs no distributed factorization at all.

## 3. The implementation: leader + one helper per socket (task 2)

`docs/DESIGN.md` §3.9 is the contract, `docs/PARALLEL.md` the user's page; the measurements of
section 2 decided every point:

* **Layout.** `mpiexec -n R hsolve-mpi <hsolve's arguments>`, laid out by `hsolve-mpirun`: rank 0, the
  **leader**, bound to all cores (16 threads), runs `hs::solve` exactly as the single process does -
  analysis, main factorization, Krylov, verification, slice cuts; ranks 1..S, the **helpers**, one per
  NUMA node (socket) with that node's cores. The leader never computes while helpers do and vice
  versa (each waits in a sleeping poll, `MpiTransport::receive`), so no core is ever oversubscribed
  and the run never uses more than its 16 cores. On one NUMA node `hsolve-mpirun` starts plain hsolve
  (R = 1).
* **What helpers do.** (1) The nearest certificate's two exact counts nu(a), nu(b), one per helper,
  at the same time (each on its own InertiaCounter on the leader's pivot order; `count_below`'s rules,
  so the integer counts are those of the single process). (2) The slices of window targets
  (interval / smallest / largest, and nearest targets handed to the windows by the zero band): the
  leader counts the cuts as before and hands the slices to idle helpers, which run the same
  `run_slice`; the leader merges. Everything else stays on the leader: section 2 found no other
  independent factorizations (MPI MUMPS for the main factorization: section 2.2).
* **When.** At least two usable helpers (one alone would compete with the leader for cores or run on
  half of them); the memory budget - charged with the helpers' memory, so it covers the process
  tree - must hold the work at once (two counts after the leader released its main factors; a
  slice per helper, each getting an equal share of what is left); and cost rules (`--helper-counts`,
  `--helper-slices` auto, the default): a count (0.9 x the measured main factorization) must outweigh
  the helpers' preparation (two analyses on the given order + 0.05 s), a slice's predicted work its
  helper's preparation. With two or more helpers a window of at least 32 eigenvalues is cut into at
  least as many slices as there are helpers. The problem (the factorized matrices, configuration,
  pivot order) is sent only before the helpers' first task: sent right after the analysis, the
  helpers' preparation slowed the leader's main factorization by 0.2-0.35 s at 1e5 (screening).
* **Failures.** A helper task that fails (MUMPS or memory error, an exception; fault bit 128 injects
  it) is answered with the error; the leader does the work itself and stops using that helper in
  this solve (`report.parallel.fallbacks`). A helper process that dies ends the MPI job (MPI's default
  error handler).
* **Build.** `HSOLVE_WITH_MPI=AUTO` (on the cluster: the Open MPI 5.0.11 of the os2026-ref
  environment); the hsolve executable is unchanged. Each process keeps the sequential MUMPS; its
  libmpiseq defines `MPI_Init` and three more stubs, so MPI is linked first (checked at start-up).
* **Tests** (whole suite green, section 6): over thread transports (`test_parallel_session`: the
  wire format, the transport, counts on two helpers, slices on the helpers, one helper, every
  failure kind, the cost rules and Off, the driver hooks against scripted helpers: the memory rule,
  a failed count, real and complex) and over MPI with 1, 2 and 3 ranks (`test_mpi_session.ranks{1,2,3}`:
  the real MPI library is linked, nearest certificate, interval slices, failing helpers - results
  equal to `hs::solve`'s); pytest for the campaign tool.

__SECTION4__

## 6. Commands

On c005 from the work tree of this branch (`HS_CLUSTER_NAME=p10mpi tools/dev/cluster_run.sh c005 -- ...`),
`D=$HOME/p7_p7-membudget/data`, `K=$HOME/p10mpi/k`, raw output in `$HOME/p10mpi/runs/<tag>*`
(`<tag>.jsonl` - bench JSON lines or one entry per hsolve run with its report summary and guard
summary -, `<tag>.eval.json`, the result directories). Builds, dumps and evaluations in the
build/test lane, never during a timed batch.

```bash
tools/dev/build.sh release && bash tools/bench/mpi_build.sh                       # lane
# K(sigma) and hsolve's pivot order per case (lane; 1e6: P9's K dumps, $HOME/p9_p9-mpi/k):
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 900 -- \
  python3 tools/bench/mpi_dump_kernel.py --root data --case syn-mem27-46 --out $K/syn-mem27-46.K.bin
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 900 -- \
  build/release/bench_factspeed_factor --case data/cases/syn-mem27-46 --ordering 10 --jobs "" --perm-out $K/syn-mem27-46.perm
#   (cav2d-rect-pert-1e5: --ordering 0, hsolve's choice there)
# 2.1 / 2.2 phase campaigns (each run: guard.py --exclusive --cores <placement> --mem-gb 32):
python3 tools/bench/mpi_layouts.py run --tag a1e5 --kind bench --reps 3 \
  --layouts one:1x16,one:1x8,one:2x8,two:1x16,two:1x16i,two:2x8,two:2x8m,one:2x8m \
  --case syn-mem27-46 --case yee-pec-box-1e5 --case ss-qa8fk-qa8fm --case cav2d-rect-pert-1e5 --timeout 600
python3 tools/bench/mpi_layouts.py run --tag a1e6c --kind bench --reps 3 --layouts one:1x16,one:2x8,two:1x16,two:2x8 \
  --case syn-mem27-100 --case yee-pec-box-1e6 --bench-args "--modes count" --timeout 900
python3 tools/bench/mpi_layouts.py run --tag a1e6 --kind bench --reps 3 \
  --layouts one:1x16,one:1x8,two:1x16,two:1x16i,two:2x8m,one:2x8m --case syn-mem27-100 --case yee-pec-box-1e6 --timeout 1200
python3 tools/bench/mpi_layouts.py table --tag a1e5        # (a1e6c, a1e6)
__COMMANDS4__
```
