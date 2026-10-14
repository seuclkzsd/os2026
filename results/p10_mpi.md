# P10 "mpi": hsolve over several processes - one per socket

P10 task "mpi" (branch `p10mpi` from p9-merge 90e7483). Question: can multi-process (MPI) execution
make hsolve faster at the judging budget of 16 cores per run, and where? P9 measured that on ONE
16-core socket MPI loses (`results/p9_mpi.md`: MPI MUMPS 2 x 8 .. 16 x 1 = 0.85 .. 0.59x of 1 x 16,
two concurrent 8-thread counts 0.92x): every factorization is limited by the socket's memory
bandwidth. A judging machine with 16 cores may also be two sockets of 8 - then a process per socket
has its own memory bandwidth. This report measures that first (section 2), implements the layouts
that pay as an optional multi-process mode of hsolve (section 3; `docs/DESIGN.md` §3.9,
`docs/PARALLEL.md`) and measures it end to end (section 4).

**Answers.**

* **Processes pay where they get their own memory bandwidth - one per socket - and where the work
  splits into independent factorizations.** Two processes, one per socket, each make an 8-thread
  inertia count in the time one alone takes (27.1 vs 27.0 s at 1e6), so the certificate's two counts
  at once take 27.1 / 22.5 s instead of 35.2 / 29.7 s for one process spreading 16 threads over both
  sockets and 43.2 / 35.7 s on one socket (1e5: 0.49-0.54 s instead of 1.1-1.2 s / 0.85 s). On ONE
  socket two processes slow each other down (each 1.4-1.8x), confirming P9 (section 2).
* **MPI MUMPS across two sockets (task 1b)** factorizes 1.16-1.39x faster than the sequential MUMPS
  spread over both at 1e5, but only 0.95x / 1.09x at 1e6, and needs a distributed main factorization;
  for the certificate it loses to one count per socket (37.3 vs 27.1 s). Not adopted (section 2.2).
* **Implemented: hsolve-mpi, a leader + one helper per socket** (`mpiexec -n 1+S hsolve-mpi`, or
  `hsolve-mpirun`, which reads the NUMA topology; `docs/PARALLEL.md`, DESIGN §3.9). The leader runs the
  solve on all cores exactly as hsolve does; the helpers, each on its socket, make the nearest
  certificate's two counts at once and solve the slices of window targets, while the leader waits -
  no core is ever shared. Results equal hsolve's (exact integer counts, the same slices; every run of
  the final code valid - an earlier version gave slice helpers too little memory and missed 4 pairs
  of one window, which tools/eval caught, section 4); a failing helper task is redone by the leader;
  the single-process hsolve and its default are unchanged. Tests run the protocol over threads and
  over MPI with 1, 2 and 3 ranks.
* **Measured end to end (5 interleaved repetitions, every run valid; section 4).** On two sockets (8 +
  8 cores) against hsolve spread over the same cores: **1e6 nearest 1.10x** (syn-mem27-100 75.4 -> 68.3
  s, yee-pec-box-1e6 64.1 -> 58.5 s; against hsolve on one socket 1.28x / 1.23x), 1e5 3-D nearest
  1.06-1.14x, **windows with several slices 1.16-1.75x** (syn-mem27-46 [0.1, 0.3], 655 eigenvalues:
  32.9 -> 18.8 s). On one socket the mode loses (0.86-0.95x; 1.11-1.17x on the windows with many
  slices), so the launcher runs plain hsolve there.
* **Honest limits (section 5):** the gain needs two sockets (a 16-core judging machine may be one);
  only the certificate's counts and the slices run in parallel - main factorization and Krylov stay
  one sequential computation; MPI start-up costs ~1.8-2.0 s of wall time, more than the gain on 1e5
  nearest targets; the process tree's peak grows 1.3-1.4x at 1e6; GHIEP gets nothing.

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

## 2. Measure first: what a process per socket buys (5 interleaved repetitions)

### 2.1 Phases

`mpi_layouts.py run --kind bench` (tags a1e6c, a1e6, a1e5). Per layout: one inertia-only count
(ICNTL(31) = 1, the certificate's), the **pair** of counts a nearest certificate needs (two counts
one after the other = 2 x one count; for 2x8 the two run at the same time, one per process, and
the pair takes the slower one), one numeric factorization keeping the factors, one 16-RHS block
solve; medians [min-max] in seconds, x = speedup over `one:1x16` (hsolve's layout on one socket).
Every count reproduced the exact inertia.

**1e6** (a1e6c, counts only: the counts of one:1x16, one:2x8, two:1x16 and two:2x8; a1e6: the rest):

| case | layout | one count | count pair | x | factorization | x | 16-RHS solve | x |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | one:1x16 | 21.59 | 43.19 [43.11-43.36] | 1.00 | 22.93 [22.92-23.01] | 1.00 | 0.842 | 1.00 |
| syn-mem27-100 | one:1x8 | 26.95 | 53.90 | 0.80 | 28.24 | 0.81 | 0.934 | 0.90 |
| syn-mem27-100 | one:2x8 (two counts at once on one socket) | 47.95 | 47.95 [47.75-48.06] | 0.90 | - | - | - | - |
| syn-mem27-100 | two:1x16 | 17.59 | 35.18 [34.04-36.47] | 1.23 | 18.07 [17.66-18.98] | 1.27 | 0.879 | 0.96 |
| syn-mem27-100 | two:1x16i | 17.07 | 34.14 | 1.27 | 18.08 | 1.27 | 0.812 | 1.04 |
| syn-mem27-100 | **two:2x8** (one count per socket at once) | 27.07 | **27.10** [27.07-27.11] | **1.59** | - | - | - | - |
| yee-pec-box-1e6 | one:1x16 | 17.83 | 35.67 [35.55-35.74] | 1.00 | 18.94 [18.91-18.95] | 1.00 | 0.790 | 1.00 |
| yee-pec-box-1e6 | one:1x8 | 22.37 | 44.75 | 0.79 | 23.66 | 0.80 | 0.848 | 0.93 |
| yee-pec-box-1e6 | one:2x8 | 39.51 | 39.52 [39.50-39.60] | 0.90 | - | - | - | - |
| yee-pec-box-1e6 | two:1x16 | 14.87 | 29.74 [28.74-31.27] | 1.20 | 15.97 [15.55-16.19] | 1.19 | 0.913 | 0.87 |
| yee-pec-box-1e6 | two:1x16i | 14.79 | 29.58 | 1.20 | 15.64 | 1.21 | 0.803 | 0.98 |
| yee-pec-box-1e6 | **two:2x8** | 22.45 | **22.46** [22.43-22.48] | **1.59** | - | - | - | - |

**1e5** (a1e5; pair / factorization / solve, x over one:1x16):

| case | one:1x16 | one:1x8 | one:2x8 | two:1x16 | two:1x16i | **two:2x8** |
|---|---|---|---|---|---|---|
| syn-mem27-46 pair | 0.849 | 1.013 (0.84) | 0.813 (1.04) | 1.221 [1.165-2.190] (0.70) | 1.080 (0.79) | **0.538** [0.507-0.557] **(1.58)** |
| syn-mem27-46 factorization / solve | 0.462 / 0.054 | 0.548 / 0.053 | 0.912 / 0.073 (both at once) | 0.521 / 0.066 | 0.508 / 0.058 | 0.551 / 0.054 (both at once) |
| yee-pec-box-1e5 pair | 0.852 | 0.950 (0.90) | 0.763 (1.12) | 1.116 [1.039-2.672] (0.76) | 1.064 (0.80) | **0.490** [0.481-0.518] **(1.74)** |
| yee-pec-box-1e5 factorization / solve | 0.451 / 0.054 | 0.516 / 0.054 | 0.858 / 0.073 | 0.491 / 0.062 | 0.513 / 0.062 | 0.519 / 0.054 |
| ss-qa8fk-qa8fm pair | 0.367 | 0.411 (0.89) | 0.293 (1.25) | 1.031 [0.571-1.578] (0.36) | 0.486 (0.76) | **0.220** [0.199-0.232] **(1.67)** |
| ss-qa8fk-qa8fm factorization / solve | 0.198 / 0.031 | 0.209 / 0.030 | 0.329 / 0.038 | 0.241 / 0.037 | 0.244 / 0.037 | 0.213 / 0.030 |
| cav2d-rect-pert-1e5 pair | 0.089 | 0.115 (0.78) | 0.079 (1.13) | 0.184 [0.157-0.268] (0.49) | 0.115 (0.77) | **0.072** [0.052-0.083] **(1.24)** |
| cav2d-rect-pert-1e5 factorization / solve | 0.035 / 0.018 | 0.047 / 0.029 | 0.056 / 0.032 | 0.056 / 0.026 | 0.050 / 0.023 | 0.050 / 0.029 |

What the phases say:

* **Two processes, one per socket, do not slow each other down.** Each of two concurrent 8-thread
  counts on its own socket takes what one 8-thread count takes alone (1e6: 27.07 vs 26.95 s,
  22.45 vs 22.37 s; 1e5: 0.520 vs 0.507 s, 0.484 vs 0.475 s), and two concurrent factorizations
  equal one alone (syn-mem27-46 0.551 vs 0.548 s): each socket has its own memory bandwidth. On ONE
  socket two concurrent processes share it and each takes 1.4-1.8x longer (P9: 0.92x for the pair;
  here 0.90x at 1e6, 1.04-1.25x at 1e5 where a single count does not saturate the socket).
* **Hence the certificate's two counts pay most**: the pair at once on two sockets takes 1.59x less
  than hsolve's pair on one socket (1e6) and 1.58-1.74x less at 1e5 (3-D cases) - and against the fairer
  baseline on a two-socket machine, one process spreading its 16 threads over both sockets
  (`two:1x16`), 1.30x (syn-mem27-100: 35.18 -> 27.10 s), 1.32x (yee-pec-box-1e6: 29.74 -> 22.46 s)
  and 2.3-4.7x at 1e5 (4.7x: ss-qa8fk-qa8fm, whose `two:1x16` pair varied 0.57-1.58 s).
* **One process over two sockets is good at 1e6, poor at 1e5.** `two:1x16` factorizes 1.19-1.27x
  faster than one socket at 1e6 (it has both sockets' bandwidth) but 0.63-0.92x at 1e5, and its
  counts are worse still (0.36-0.76x; the first numeric phase of a fresh MUMPS instance places its
  memory on one socket by first touch, which half the threads then read remotely); interleaving the
  pages (`1x16i`) helps a little (0.76-0.80x). A single factorization on one socket's 8 cores
  (`1x8`) is 0.80-0.81x of one socket's 16 at 1e6, 0.75-0.95x at 1e5.
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
| syn-mem27-100 | 19.04 | **0.95** (18.07) | 1.20 (22.93; on one socket 26.98 s: 0.85) | 1.08 | 0.95 |
| yee-pec-box-1e6 | 14.60 | **1.09** (15.97) | 1.30 (18.94; on one socket 22.89 s: 0.83) | 1.37 | 1.08 |
| syn-mem27-46 | 0.448 | 1.16 (0.521) | 1.03 (0.462; on one socket 0.555 s: 0.83) | 1.43 | 1.19 |
| yee-pec-box-1e5 | 0.375 | 1.31 (0.491) | 1.20 (0.451; on one socket 0.494 s: 0.91) | 1.05 | 1.27 |
| ss-qa8fk-qa8fm | 0.173 | 1.39 (0.241) | 1.14 (0.198; on one socket 0.213 s: 0.93) | 1.28 | 2.19 (two:1x16's counts varied) |
| cav2d-rect-pert-1e5 | 0.046 | 1.22 (0.056) | 0.76 (0.035; on one socket 0.045 s: 0.78) | 0.74 | 1.67 |

(speedup = the baseline's time / 2x8m's; baselines in parentheses, same campaign.)

* Unlike one socket (0.78-0.93x, P9's verdict), across two sockets MPI MUMPS 2x8 beats the
  sequential MUMPS spread over both sockets at 1e5 (1.16-1.39x per factorization) and is mixed at 1e6
  (0.95x syn-mem27-100, 1.09x yee-pec-box-1e6). Against hsolve's sequential MUMPS on ONE socket it
  gains at 1e6 what the second socket's bandwidth gives any layout (1.20 / 1.30x; `two:1x16` 1.27 /
  1.19x) and at 1e5 0.76-1.20x per factorization; the certificate's pair as two MPI counts one after
  the other is 0.78-0.97x of one:1x16's pair at 1e5.
* Against the per-socket concurrency of section 2.1 it loses where hsolve has independent
  factorizations: the certificate's pair as two MPI counts takes 37.3 / 28.2 s (1e6) against 27.1 /
  22.5 s for one count per socket, 1.03 / 0.88 s against 0.54 / 0.49 s at 1e5.
* Not adopted: the main factorization would have to become a distributed MUMPS object (hsolve
  links the sequential MUMPS, whose libmpiseq defines MPI symbols; solves, refinement and the
  Krylov basis would have to follow the distributed factors), for at most 1.09x on one of the two
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

## 4. End to end (claims: 5 interleaved repetitions, every run valid)

`mpi_layouts.py run --kind hsolve`, then `eval` (tools/eval on every result) and `table`. Solver time =
the report's `time_total_s` (from the matrices in memory to the result; the guard's wall time adds
loading and, for the multi-process mode, the MPI start-up - both shown). Phases are hsolve's
exclusive phases of the leader (DESIGN §3.6-0; waiting for helpers is charged to certificate /
iterate). Peak: the guard's sampled process tree (all ranks) / the sum of the processes' own peaks.
Layouts: `one:` = 16 cores of one socket, `two:` = 8 cores of each socket; `1x16` = hsolve,
`1+2x8` = hsolve's multi-process mode (leader 16 threads, two helpers x 8 threads). Binaries: 1e5 and
intervals from the final code (tags d1e5, div1-3, 142/142 runs valid); 1e6 from campaign C (c1e6,
40/40 valid) at 1c39bf6 - the later commits change only the cost rules (which hand these counts to
the helpers as before), the slice helpers' memory share and the report; a run of each 1e6 case with
the final binary (d1e6s) gives the same: 67.2 s / 57.9 s, 2 helper counts each.

**Nearest targets (the certificate's two counts on the helpers):**

| case | layout | solver s, median [min-max] | x vs one:1x16 | x vs two:1x16 | wall s | factor | iterate | certificate | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | one:1x16 | 87.41 [87.26-87.63] | 1.00 | - | 88.4 | 26.76 | 16.11 | 43.78 | 13.1 |
| syn-mem27-100 | one:1+2x8 | 93.87 [92.85-94.14] | 0.93 | - | 96.5 | 26.84 | 16.11 | 50.14 | 18.2 / 30.0 |
| syn-mem27-100 | two:1x16 | 75.38 [72.60-76.65] | 1.16 | 1.00 | 76.3 | 22.06 | 16.40 | 35.91 | 13.1 |
| syn-mem27-100 | **two:1+2x8** | **68.28** [67.40-70.04] | **1.28** | **1.10** | 70.9 | 22.35 | 15.92 | **29.19** | 18.0 / 30.1 |
| yee-pec-box-1e6 | one:1x16 | 72.21 [72.12-72.58] | 1.00 | - | 73.0 | 22.52 | 13.14 | 36.05 | 11.3 |
| yee-pec-box-1e6 | one:1+2x8 | 77.67 [77.44-77.81] | 0.93 | - | 80.2 | 22.62 | 13.12 | 41.40 | 14.9 / 24.7 |
| yee-pec-box-1e6 | two:1x16 | 64.09 [62.85-64.24] | 1.13 | 1.00 | 64.9 | 19.44 | 14.28 | 29.68 | 11.3 |
| yee-pec-box-1e6 | **two:1+2x8** | **58.49** [58.21-58.72] | **1.23** | **1.10** | 60.9 | 19.23 | 14.31 | **24.17** | 14.9 / 24.6 |
| syn-mem27-46 | one:1x16 | 2.016 [1.996-2.192] | 1.00 | - | 2.18 | 0.82 | 0.26 | 0.86 | 0.9 |
| syn-mem27-46 | one:1+2x8 | 2.273 [2.232-2.281] | 0.89 | - | 4.24 | 0.85 | 0.26 | 1.09 | 1.4 / 1.9 |
| syn-mem27-46 | two:1x16 | 2.260 [2.107-2.441] | 0.89 | 1.00 | 2.42 | 0.91 | 0.29 | 1.01 | 0.9 |
| syn-mem27-46 | **two:1+2x8** | **1.988** [1.978-2.077] | 1.01 | **1.14** | 3.90 | 0.88 | 0.28 | **0.76** | 1.3 / 1.9 |
| yee-pec-box-1e5 | one:1x16 | 2.453 [2.431-2.488] | 1.00 | - | 2.62 | 0.82 | 0.75 | 0.84 | 0.8 |
| yee-pec-box-1e5 | one:1+2x8 | 2.637 [2.623-2.661] | 0.93 | - | 4.59 | 0.82 | 0.75 | 1.01 | 1.2 / 1.6 |
| yee-pec-box-1e5 | two:1x16 | 2.784 [2.655-2.853] | 0.88 | 1.00 | 2.99 | 0.91 | 0.83 | 0.97 | 0.8 |
| yee-pec-box-1e5 | **two:1+2x8** | **2.540** [2.477-2.560] | 0.97 | **1.10** | 4.39 | 0.88 | 0.83 | **0.72** | 1.2 / 1.7 |
| ss-qa8fk-qa8fm | one:1x16 | 1.502 [1.493-1.512] | 1.00 | - | 1.73 | 0.56 | 0.51 | 0.38 | 0.7 |
| ss-qa8fk-qa8fm | one:1+2x8 | 1.678 [1.670-1.693] | 0.90 | - | 3.63 | 0.56 | 0.52 | 0.53 | 0.9 / 1.1 |
| ss-qa8fk-qa8fm | two:1x16 | 1.740 [1.661-1.844] | 0.86 | 1.00 | 1.98 | 0.61 | 0.55 | 0.48 | 0.7 |
| ss-qa8fk-qa8fm | **two:1+2x8** | **1.637** [1.616-1.662] | 0.92 | **1.06** | 3.54 | 0.61 | 0.53 | **0.43** | 0.9 / 1.1 |
| cav2d-rect-pert-1e5 (2-D) | one:1x16 | 0.308 [0.294-0.334] | 1.00 | - | 0.44 | 0.10 | 0.11 | 0.06 | 0.2 |
| cav2d-rect-pert-1e5 | one:1+2x8 | 0.323 [0.306-0.347] | 0.95 | - | 2.25 | 0.10 | 0.12 | 0.06 | 0.3 |
| cav2d-rect-pert-1e5 | two:1x16 | 0.535 [0.410-0.591] | 0.57 | 1.00 | 0.72 | 0.18 | 0.16 | 0.09 | 0.2 |
| cav2d-rect-pert-1e5 | two:1+2x8 (no helper work: cost rule) | 0.400 [0.362-0.471] | 0.77 | 1.34 | 2.28 | 0.12 | 0.14 | 0.09 | 0.3 |

**Window targets (spectrum slicing; the slices on the helpers):**

| case, interval (eigenvalues, slices) | layout | solver s | x vs one:1x16 | x vs two:1x16 | wall s | factor | iterate | certificate | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 [0.1, 0.3] (655, 4) | one:1x16 | 31.17 [31.03-31.51] | 1.00 | - | 32.2 | 2.16 | 24.70 | 2.29 | 2.5 |
| | one:1+2x8 | 26.68 [26.64-26.83] | 1.17 | - | 29.4 | 0.37 | 23.75 | 2.31 | 4.2 / 5.5 |
| | two:1x16 | 32.89 [32.35-33.26] | 0.95 | 1.00 | 33.9 | 2.35 | 25.75 | 2.49 | 2.4 |
| | **two:1+2x8** | **18.78** [18.70-18.94] | **1.66** | **1.75** | 21.5 | 0.39 | 15.74 | 2.41 | 4.2 / 5.5 |
| yee-pec-cube-32 [40, 400] (271, 2) | one:1x16 | 13.89 [13.71-13.92] | 1.00 | - | 14.4 | 1.07 | 10.91 | 1.11 | 1.7 |
| | one:1+2x8 | 12.46 [12.32-12.55] | 1.11 | - | 14.6 | 0.33 | 10.92 | 1.12 | 3.2 / 4.2 |
| | two:1x16 | 15.31 [15.21-15.78] | 0.91 | 1.00 | 15.8 | 1.31 | 11.79 | 1.35 | 2.1 |
| | **two:1+2x8** | **10.15** [9.97-10.19] | **1.37** | **1.51** | 12.3 | 0.39 | 8.29 | 1.35 | 3.1 / 4.2 |
| yee-pec-box-1e5 [40, 400] (108; 1 slice, 2 with helpers) | one:1x16 | 5.849 [5.836-5.852] | 1.00 | - | 6.21 | 0.80 | 3.75 | 0.90 | 1.4 |
| | one:1+2x8 | 6.801 [6.791-6.842] | 0.86 | - | 8.89 | 0.37 | 5.02 | 1.33 | 2.6 / 3.4 |
| | two:1x16 | 6.442 [6.229-6.500] | 0.91 | 1.00 | 6.74 | 1.06 | 3.93 | 1.01 | 1.7 |
| | **two:1+2x8** | **5.548** [5.520-5.574] | 1.05 | **1.16** | 7.61 | 0.38 | 3.62 | 1.48 | 2.7 / 3.4 |

What the claims say:

* **Two sockets: the multi-process mode is faster than one process spread over both** - 1e6 nearest
  1.10x (68.3 vs 75.4 s, 58.5 vs 64.1 s; certificate 35.9 -> 29.2 s, 29.7 -> 24.2 s), 1e5 3-D nearest
  1.06-1.14x (certificate 0.97-1.01 -> 0.72-0.76 s), windows with several slices 1.16x, 1.51x, 1.75x.
  The main factorization and the Krylov phase are the leader's as before (equal within noise).
* **Against hsolve on one socket** (the 16-core layout on these nodes): 1e6 1.23-1.28x (half of it the
  second socket's bandwidth for the leader: `two:1x16` alone is 1.13-1.16x),
  windows 1.05-1.66x (1.05x: the window whose second slice exists only for the helpers); 1e5
  nearest 0.92-1.01x (3-D), 0.77x (cav2d).
* **On one socket the mode loses** (0.89-0.95x nearest, 0.86x on yee-pec-box-1e5's window): the
  helpers share the socket's memory bandwidth and pay their preparation; it still gains on the windows
  with many slices (1.11x, 1.17x), where the leader alone would factorize one slice after the other.
  `hsolve-mpirun` therefore starts one process on a single NUMA node.
* **Wall time.** Starting the MPI job (prterun, MPI_Init of three processes) adds ~1.8-2.0 s at 1e5
  (wall 3.5-4.4 s against 2.0-3.0 s); at 1e6 the helpers' gain outweighs it (70.9 vs 76.3 s, 60.9 vs
  64.9 s), on the windows with several slices too (21.5 vs 33.9 s, 12.3 vs 15.8 s), not on the 1e5
  nearest cases and the 108-eigenvalue window.
* **Memory.** The helpers hold their own copies of the problem and their counters: the process tree
  peaks at 18.0 / 14.9 GiB (sampled; sum of the processes' own peaks 30.1 / 24.6 GiB) against 13.1 /
  11.3 GiB at 1e6 - within the 32 GiB of a run, but on a 16 GiB machine the 1e6 cases run out of core
  anyway (P7) and the two counts at once would not fit: the memory rule then keeps them on the leader.
* **Results are the single-process ones**: the same eigenvalues (every run valid against the complete
  references), the same certificate counts (`2 of the 2 counts made by helper processes`), the same
  slices.
* **The 2-D case and the GHIEP case get no helper work** (cav2d-rect-pert-1e5: a 0.035 s main
  factorization against ~0.18 s of helper preparation, the cost rule keeps the counts on the leader;
  wg-square-1e5: GHIEP has no inertia counts). Yet the multi-process leader's run is shorter than the
  plain process's on two sockets (cav2d 0.400 vs 0.535 s): section 4.1 separates that from the
  helpers.
* **A bug the campaign caught.** The first version gave each slice helper only the memory one slice
  needs; on yee-pec-cube-32's 167-pair slice its Krylov basis did not fit, the helper halved its block
  width to 1 and missed 4 pairs (tools/eval: invalid, campaign C tag civ3); helpers now get an equal
  share of what the budget leaves (e2c443d), measured above (div3).

### 4.1 What the helpers do (tag f1e5, final binary, 5 repetitions, all valid)

Two controls on two sockets: `two:1x16r` - hsolve-mpi with ONE rank through the launcher (the same
start-up, Open MPI loaded, no helpers) - and `two:1x16g` - plain hsolve with glibc's malloc keeping
freed memory (`MALLOC_MMAP_MAX_=0`, no trimming; Open MPI may change the allocator's behaviour):

| case | two:1x16 | two:1x16r | two:1x16g | two:1+2x8 | certificate: two:1x16 / 1+2x8 |
|---|---|---|---|---|---|
| syn-mem27-46 | 2.330 [2.090-2.494] | 2.353 [2.301-2.420] | 2.176 [2.156-2.209] | **1.999** [1.963-2.069] | 1.02 / **0.77** |
| yee-pec-box-1e5 | 2.794 [2.637-2.925] | 2.932 [2.859-3.004] | 2.815 [2.711-3.206] | **2.486** [2.462-2.558] | 0.95 / **0.72** |
| cav2d-rect-pert-1e5 (no helper work) | 0.427 [0.333-0.457] | 0.448 [0.404-0.594] | 0.400 [0.318-0.522] | 0.376 [0.346-0.443] | 0.07 / 0.08 |
| wg-square-1e5 (GHIEP, no helper work) | 0.557 [0.543-0.944] | 0.620 [0.590-0.822] | 0.767 [0.631-1.725] | 0.592 [0.583-0.671] | - |

* Neither the launch path (`1x16r`: 0.95-0.99x of plain hsolve on the 3-D cases) nor the allocator
  (`1x16g`: 0.99-1.07x) explains the multi-process gain: it is the certificate's counts on the helpers
  (0.95-1.02 s -> 0.72-0.77 s), 1.12x / 1.17x end to end here.
* The cases without helper work show only the run-to-run spread of one process spread over two
  sockets at 1e5 (ranges overlap; plain hsolve's 0.333-0.457 s on cav2d, 0.543-0.944 s on
  wg-square-1e5); the 1.34x of cav2d in the table above is that spread, not a gain.

## 5. The "并行方案" table and the limits

**Per phase** (section 2, the phase bench, 5 repetitions; syn-mem27-100 / yee-pec-box-1e6, seconds;
16 cores in every row):

| R x T | what | placement | one count | the certificate's two counts | one factorization | 16-RHS solve |
|---|---|---|---|---|---|---|
| 1 x 16 | hsolve (sequential MUMPS, OpenMP) | one socket | 21.6 / 17.8 | 43.2 / 35.7 | 22.9 / 18.9 | 0.84 / 0.79 |
| 1 x 16 | hsolve | two sockets (8 + 8) | 17.6 / 14.9 | 35.2 / 29.7 | 18.1 / 16.0 | 0.88 / 0.91 |
| 1 x 8 | hsolve on half the cores | one socket | 27.0 / 22.4 | 53.9 / 44.7 | 28.2 / 23.7 | 0.93 / 0.85 |
| 2 x 8 | two hsolve processes, one count each, at once | one socket | 47.9 / 39.5 each | 48.0 / 39.5 | - | - |
| 2 x 8 | the same - **hsolve's helpers** | two sockets | 27.1 / 22.5 each | **27.1 / 22.5** | - | - |
| 2 x 8 | MPI MUMPS (one distributed factorization) | one socket | 25.8 / 21.6 | 51.6 / 43.2 | 27.0 / 22.9 | 0.98 / 0.87 |
| 2 x 8 | MPI MUMPS | two sockets | 18.6 / 14.1 | 37.3 / 28.2 | 19.0 / 14.6 | 0.81 / 0.67 |

(Counts of the 1 x 16 and the concurrent 2 x 8 rows from the count campaign a1e6c, which ran these layouts
interleaved with each other; the rest from a1e6.)

**End to end** (section 4, 5 repetitions, solver time; speedup over hsolve on one socket / over hsolve
on the same two sockets):

| layout (16 cores) | placement | handed to helpers | 1e6 nearest: syn-mem27-100 / yee-pec-box-1e6 | 1e5 3-D nearest: syn-mem27-46 / yee-pec-box-1e5 / ss-qa8fk-qa8fm | windows: syn-mem27-46 (4 slices) / yee-pec-cube-32 (2) / yee-pec-box-1e5 (2) |
|---|---|---|---|---|---|
| 1 x 16 (hsolve) | one socket | - | 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 1 + 2 x 8 (hsolve-mpi) | one socket | counts, slices | 0.93 / 0.93 | 0.89 / 0.93 / 0.90 | 1.17 / 1.11 / 0.86 |
| 1 x 16 (hsolve) | two sockets | - | 1.16 / 1.13 | 0.89 / 0.88 / 0.86 | 0.95 / 0.91 / 0.91 |
| **1 + 2 x 8 (hsolve-mpi)** | **two sockets** | counts, slices | **1.28 / 1.23** (1.10 / 1.10) | **1.01 / 0.97 / 0.92** (1.14 / 1.10 / 1.06) | **1.66 / 1.37 / 1.05** (1.75 / 1.51 / 1.16) |

Per phase in those runs (two sockets, hsolve -> hsolve-mpi): the certificate 35.9 -> 29.2 s and 29.7 ->
24.2 s at 1e6, 0.97-1.01 -> 0.72-0.76 s at 1e5; the windows' slices (iterate) 25.8 -> 15.7 s
(syn-mem27-46) and 11.8 -> 8.3 s (yee-pec-cube-32); analysis, main factorization and Krylov run on the
leader as in hsolve and take the same time.

**Limits** (honest ones):

* **One socket: no gain.** Helpers on one socket share its memory bandwidth: 0.86-0.95x, except
  windows with many slices (1.11-1.17x). `hsolve-mpirun` starts plain hsolve on one NUMA node - the
  judging machine (16 cores) may well be one socket, and then the multi-process mode changes nothing.
* **Two sockets: modest for nearest targets.** Only the certificate's two counts run concurrently -
  the main factorization and the Krylov run are one sequential computation, which the leader runs on
  all cores as hsolve does; at 1e6 the certificate is ~40-45% of the run, so 1.10x. Windows with
  several slices gain most (1.5-1.75x). GHIEP (no counts), the randomized certificate (no counts when
  it certifies), single-slice windows and problems too small for the helpers' preparation (cost
  rules) get nothing.
* **MPI start-up.** ~1.8-2.0 s of wall time (prterun, three MPI_Inits) before the solve; it outweighs
  the gain on the 1e5 nearest cases and the 108-eigenvalue window, not at 1e6 or on the larger windows.
* **Memory.** Each helper holds a copy of the problem and its own counter or factors: the process tree
  peaks 1.3-1.4x higher at 1e6 (18.0 vs 13.1 GiB, 14.9 vs 11.3 GiB). The budget covers the tree, so
  under a tight limit the counts stay on the leader rather than overrun it.
* **Emulated topology.** The two-socket layouts use 8 of each socket's 16 cores (the run's 16 cores);
  each socket then serves 8 cores with its whole memory bandwidth and L3, as a 2 x 8-core machine
  would, but a machine with smaller sockets may have less bandwidth per socket.
* **MPI MUMPS stays out.** A distributed main factorization gained 0.95-1.09x at 1e6 on two sockets
  (1.16-1.39x at 1e5) against the sequential MUMPS spread over both, would need a distributed main
  object in place of hsolve's, and loses to the per-socket counts for the certificate.
* **Failures.** A helper task that fails is redone by the leader (tests: fault bit 128, every kind);
  a helper process that dies ends the MPI job.

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
# (the same with --reps 2 --rep0 3 completes them to 5 repetitions)
# 4: whole solves (each run: guard.py --exclusive --cores <placement> --mem-gb 32); L = the four layouts
L=one:1x16,two:1x16,two:1+2x8,one:1+2x8
python3 tools/bench/mpi_layouts.py run --tag c1e6 --kind hsolve --reps 5 --layouts $L \
  --case syn-mem27-100 --case yee-pec-box-1e6 --root $D --timeout 1500                    # at 1c39bf6
python3 tools/bench/mpi_layouts.py run --tag d1e5 --kind hsolve --reps 5 --layouts $L \
  --case syn-mem27-46 --case yee-pec-box-1e5 --case ss-qa8fk-qa8fm --case cav2d-rect-pert-1e5 --timeout 600
python3 tools/bench/mpi_layouts.py run --tag div1 --kind hsolve --reps 5 --layouts $L --case syn-mem27-46 \
  --target '{"mode": "interval", "lower": 0.1, "upper": 0.3}' --timeout 900
#   div2: yee-pec-box-1e5 [40, 400]; div3: yee-pec-cube-32 [40, 400]; d1e6s: the 1e6 cases, two:1+2x8, 1 rep
python3 tools/bench/mpi_layouts.py run --tag f1e5 --kind hsolve --reps 5 --layouts two:1x16,two:1x16r,two:1x16g,two:1+2x8 \
  --case syn-mem27-46 --case yee-pec-box-1e5 --case cav2d-rect-pert-1e5 --case wg-square-1e5 --timeout 600
# validity (lane) and tables:
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 2700 -- \
  python3 tools/bench/mpi_layouts.py eval --tag d1e5                                       # (--root $D for 1e6)
python3 tools/bench/mpi_layouts.py table --tag d1e5                                        # (f1e5: --base two:1x16)
# one multi-process solve by hand (two sockets' worth of cores):
python3 tools/run/guard.py --exclusive --cores 8-15,16-23 --mem-gb 32 --timeout 900 -- \
  build/release/hsolve-mpirun -- --A data/cases/syn-mem27-46/A.npz --config data/cases/syn-mem27-46/config.json --out OUT
```
