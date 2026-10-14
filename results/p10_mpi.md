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
* **Measured end to end (5 interleaved repetitions, every run valid; section 4, with the binary of
  the review fix).** On two sockets (8 + 8 cores), against hsolve spread over the same cores:
  - **1e6 nearest: 1.09-1.11x** (syn-mem27-100 75.8 -> 68.5 s, yee-pec-box-1e6 64.1 -> 59.0 s);
    against hsolve on one socket 1.26x / 1.23x;
  - 1e5 3-D nearest: 1.10-1.20x;
  - **windows with several slices: 1.15-1.74x** (syn-mem27-46 [0.1, 0.3], 655 eigenvalues:
    33.0 -> 19.0 s).

  On one socket the mode loses (0.86-0.93x; 1.11-1.16x on the windows with many slices), so the
  launcher runs the plain hsolve there, without an MPI job.
* **Review fixes (section 7).**
  - The helpers' memory rules are now decided before the problem is sent, and the helpers drop the
    problem whenever the leader keeps the work or their batch is done. Under a tight budget the
    multi-process mode then certifies wherever hsolve does: syn-mem27-46 at 1.0 GiB went from 0/5
    to 5/5. It also keeps hsolve's slices: yee-pec-box-1e5 [40, 400] at 2.5 GiB went from 0.78x to
    1.00x.
  - On one NUMA node the launcher execs the plain hsolve, without the +1.7-1.9 s of a one-rank
    MPI job.
  - The launcher and the tests pass Open MPI's options only to Open MPI.
  - The notices cover hsolve-mpi's MPI libraries.
  - The certificate statement names each count's source.
* **Honest limits (section 5):** the gain needs two sockets (a 16-core judging machine may be one);
  only the certificate's counts and the slices run in parallel - main factorization and Krylov stay
  one sequential computation; MPI start-up costs ~1.7-2.0 s of wall time, more than the gain on 1e5
  nearest targets; the process tree's peak grows 1.3-1.4x at 1e6; GHIEP gets nothing; only Open MPI
  was tested.

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
  and the run never uses more than its 16 cores. On one NUMA node `hsolve-mpirun` execs the plain
  hsolve with the same arguments - no MPI job (since the review fix, section 7.3; before, it started a
  one-rank MPI job).
* **What helpers do.** (1) The nearest certificate's two exact counts nu(a), nu(b), one per helper,
  at the same time (each on its own InertiaCounter on the leader's pivot order; `count_below`'s rules,
  so the integer counts are those of the single process). (2) The slices of window targets
  (interval / smallest / largest, and nearest targets handed to the windows by the zero band): the
  leader counts the cuts as before and hands the slices to idle helpers, which run the same
  `run_slice`; the leader merges. Everything else stays on the leader: section 2 found no other
  independent factorizations (MPI MUMPS for the main factorization: section 2.2).
* **When.** These conditions decide:
  - at least two usable helpers (one alone would compete with the leader for cores or run on half
    of them);
  - cost rules (`--helper-counts`, `--helper-slices` auto, the default): a count (0.9 x the measured
    main factorization) must outweigh the helpers' preparation (two analyses on the given order +
    0.05 s), and a slice's predicted work must outweigh its helper's preparation;
  - the memory budget must hold the work at once: two counts after the leader released its main
    factors, or a slice per helper, each getting an equal share of what is left. The budget is
    charged with what the helpers hold for the solve, i.e. their resident memory above their idle
    level, so it covers the process tree. Since the review fix (section 7) this rule is checked
    from predictions BEFORE the problem is sent (section 3.1), and again with what the helpers
    report once they have it. The helpers drop the problem (Drop / Dropped) whenever the leader
    keeps the work, and after each batch of their work. The leader's own counts and slices then
    run under the budget hs::solve would have.

  With two or more helpers, a window of at least 32 eigenvalues is cut into at least as many
  slices as there are helpers. Since the fix this happens only when the helpers will take those
  slices: the cost rule must pass and the memory must be predicted to fit; otherwise the
  sequential plan stays.

  The problem (the factorized matrices, configuration, pivot order) is sent only before the
  helpers' first task. Sent right after the analysis, the helpers' preparation slowed the leader's
  main factorization by 0.2-0.35 s at 1e5 (screening).
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
  equal to `hs::solve`'s); pytest for the campaign tool. Review fix (section 7): the memory
  rules before and after the problem is sent and the drops (FakeHelpers with a memory model; Drop
  over threads and over 3 MPI ranks), the window cut only for helpers that take the slices, the
  certificate's count sources, and the launcher (`tests/py/test_hsolve_mpirun.py`: one NUMA node =
  plain hsolve, the plan through the environment for an Open MPI- and a Hydra-style mpiexec).

### 3.1 Memory predictions of the helpers' work (review fix)

The memory rules are decided before the helpers get the problem, from what the leader knows:
- a count on a helper is predicted at `kHelperCountShare` = 0.45 of the main analysis' in-core
  estimate (MUMPS INFOG(16)), since a helper's inertia-only instance keeps no factors;
- what a helper holds once it has the problem is predicted at `kHelperProblemCopies` = 2.5 x the
  bytes of the matrices it gets: its copies, its pencil and, for counts, its counter's analysis.

Both are measured on c005 (`tools/bench/mpi_review_runs.sh calib`: hsolve-mpirun --ranks 3 in the lane, helper work forced on,
10 GiB budget; the leader's main analysis with 8 threads, the helpers' with 4). helpers_ready gives
their count estimate and what they hold above their idle level:

| case | main in-core estimate MB | a helper's count estimate MB (x main) | matrices sent MB | held per helper MB (x matrices) | drop s |
|---|---|---|---|---|---|
| syn-mem27-46 (nearest) | 718 | 320 (0.45) | 31.4 | 80.2 (2.56) | 0.010 |
| yee-pec-box-1e5 (nearest) | 659 | 264 (0.40) | 20.1 | 54.4 (2.70) | 0.009 |
| ss-qa8fk-qa8fm (nearest) | 324 | 122 (0.38) | 41.2 | 74.4 (1.81) | 0.008 |
| cav2d-rect-pert-1e5 (nearest) | 52 | 20 (0.38) | 13.9 | 30.9 (2.23) | 0.008 |
| yee-pec-box-1e5 [40, 400] (slices) | 659 | - | 20.1 | 24.8 (1.23) | 0.014 |
| syn-mem27-46 [0.1, 0.3] (slices) | 718 | - | 31.4 | 38.2 (1.22) | 0.016 |
| yee-pec-cube-32 [40, 400] (slices) | 588 | - | 15.1 | 19.2 (1.27) | 0.018 |

The count share at 1e6 was 0.495 (syn-mem27-100, 8-thread helpers: 7965 MB against 16101 MB, from
the review's 16 GiB run). A slice helper holds only its copies before its first slice: it analyses
when that slice runs, inside its granted share, so 2.5 is high for slices. The error is
one-sided. A prediction that is too high keeps work on the leader. One that is too low is caught
by the check after the problem was sent, and then only the preparation is wasted (section 7.1:
syn-mem27-46 at 1.1 GiB in the lane). Dropping the problem after a batch, which frees the helpers'
factors or count workspace, took 0.008-0.018 s at 1e5.

## 4. End to end (claims: 5 interleaved repetitions, every run valid)

`mpi_layouts.py run --kind hsolve`, then `eval` (tools/eval on every result) and `table`. Solver time =
the report's `time_total_s` (from the matrices in memory to the result; the guard's wall time adds
loading and, for the multi-process mode, the MPI start-up - both shown). Phases are hsolve's
exclusive phases of the leader (DESIGN §3.6-0; waiting for helpers is charged to certificate /
iterate). Peak: the guard's sampled process tree (all ranks) / the sum of the processes' own peaks.
Layouts: `one:` = 16 cores of one socket, `two:` = 8 cores of each socket; `1x16` = hsolve,
`1+2x8` = hsolve's multi-process mode (leader 16 threads, two helpers x 8 threads).

Binaries are those of the review fix (section 7), at 54e5219: hsolve-mpi with the memory rules
predicted before the problem is sent, and Drop. Tags g1e5, giv1-3 and g1e6, 192/192 runs valid. At
32 GiB the helpers get the same work as before the fix: 2 counts each nearest target, the same slices.
The drops of the problem after each batch take 0.012-0.016 s at 1e6. Before the fix (tags d1e5,
div1-3, c1e6) the multi-process medians were within 2% of these: section 7.4.

**Nearest targets (the certificate's two counts on the helpers):**

| case | layout | solver s, median [min-max] | x vs one:1x16 | x vs two:1x16 | wall s | factor | iterate | certificate | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | one:1x16 | 86.66 [86.44-87.38] | 1.00 | - | 87.5 | 26.75 | 15.37 | 43.77 | 13.2 |
| syn-mem27-100 | one:1+2x8 | 94.00 [93.74-94.22] | 0.92 | - | 96.7 | 26.91 | 16.11 | 50.26 | 18.3 / 29.5 |
| syn-mem27-100 | two:1x16 | 75.83 [74.07-77.88] | 1.14 | 1.00 | 76.7 | 22.38 | 16.83 | 36.04 | 13.2 |
| syn-mem27-100 | **two:1+2x8** | **68.53** [67.63-69.70] | **1.26** | **1.11** | 71.1 | 22.14 | 16.34 | **29.21** | 18.3 / 29.5 |
| yee-pec-box-1e6 | one:1x16 | 72.33 [72.18-72.54] | 1.00 | - | 73.0 | 22.55 | 13.14 | 36.12 | 11.3 |
| yee-pec-box-1e6 | one:1+2x8 | 77.73 [77.58-77.81] | 0.93 | - | 80.3 | 22.66 | 13.12 | 41.39 | 14.9 / 24.5 |
| yee-pec-box-1e6 | two:1x16 | 64.12 [62.81-65.24] | 1.13 | 1.00 | 64.9 | 19.48 | 14.14 | 29.85 | 11.3 |
| yee-pec-box-1e6 | **two:1+2x8** | **58.97** [58.77-59.44] | **1.23** | **1.09** | 61.5 | 19.88 | 14.65 | **24.21** | 14.6 / 24.5 |
| syn-mem27-46 | one:1x16 | 2.009 [1.998-2.030] | 1.00 | - | 2.18 | 0.83 | 0.26 | 0.87 | 0.9 |
| syn-mem27-46 | one:1+2x8 | 2.258 [2.219-2.283] | 0.89 | - | 4.19 | 0.84 | 0.25 | 1.09 | 1.4 / 1.8 |
| syn-mem27-46 | two:1x16 | 2.402 [2.279-2.475] | 0.84 | 1.00 | 2.58 | 1.01 | 0.29 | 1.01 | 0.9 |
| syn-mem27-46 | **two:1+2x8** | **2.008** [1.987-2.031] | 1.00 | **1.20** | 3.94 | 0.88 | 0.29 | **0.77** | 1.3 / 1.8 |
| yee-pec-box-1e5 | one:1x16 | 2.444 [2.425-2.481] | 1.00 | - | 2.62 | 0.81 | 0.75 | 0.83 | 0.8 |
| yee-pec-box-1e5 | one:1+2x8 | 2.650 [2.635-2.679] | 0.92 | - | 4.61 | 0.83 | 0.75 | 1.01 | 1.1 / 1.6 |
| yee-pec-box-1e5 | two:1x16 | 2.882 [2.794-3.097] | 0.85 | 1.00 | 3.08 | 1.02 | 0.83 | 0.97 | 0.8 |
| yee-pec-box-1e5 | **two:1+2x8** | **2.483** [2.470-2.591] | 0.98 | **1.16** | 4.41 | 0.88 | 0.82 | **0.73** | 1.1 / 1.6 |
| ss-qa8fk-qa8fm | one:1x16 | 1.514 [1.505-1.528] | 1.00 | - | 1.72 | 0.56 | 0.51 | 0.38 | 0.7 |
| ss-qa8fk-qa8fm | one:1+2x8 | 1.685 [1.679-1.707] | 0.90 | - | 3.75 | 0.56 | 0.51 | 0.53 | 0.9 / 1.0 |
| ss-qa8fk-qa8fm | two:1x16 | 1.825 [1.796-2.158] | 0.83 | 1.00 | 2.08 | 0.70 | 0.56 | 0.47 | 0.8 |
| ss-qa8fk-qa8fm | **two:1+2x8** | **1.654** [1.608-1.741] | 0.92 | **1.10** | 3.68 | 0.61 | 0.53 | **0.44** | 0.9 / 1.0 |
| cav2d-rect-pert-1e5 (2-D) | one:1x16 | 0.300 [0.297-0.342] | 1.00 | - | 0.44 | 0.10 | 0.11 | 0.06 | 0.2 |
| cav2d-rect-pert-1e5 | one:1+2x8 | 0.332 [0.314-0.349] | 0.90 | - | 2.33 | 0.12 | 0.12 | 0.06 | 0.2 / 0.3 |
| cav2d-rect-pert-1e5 | two:1x16 | 0.473 [0.414-0.594] | 0.63 | 1.00 | 0.72 | 0.18 | 0.16 | 0.08 | 0.2 |
| cav2d-rect-pert-1e5 | two:1+2x8 (no helper work: cost rule) | 0.401 [0.386-0.432] | 0.75 | 1.18 | 2.33 | 0.13 | 0.14 | 0.07 | 0.3 / 0.3 |

**Window targets (spectrum slicing; the slices on the helpers):**

| case, interval (eigenvalues, slices) | layout | solver s | x vs one:1x16 | x vs two:1x16 | wall s | factor | iterate | certificate | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 [0.1, 0.3] (655, 4) | one:1x16 | 31.16 [31.04-31.37] | 1.00 | - | 32.1 | 2.20 | 24.67 | 2.29 | 2.2 |
| | one:1+2x8 | 26.79 [26.57-26.85] | 1.16 | - | 29.6 | 0.37 | 23.88 | 2.30 | 4.2 / 5.5 |
| | two:1x16 | 33.04 [32.27-33.21] | 0.94 | 1.00 | 34.0 | 2.48 | 25.49 | 2.49 | 2.5 |
| | **two:1+2x8** | **18.95** [18.86-18.99] | **1.64** | **1.74** | 21.8 | 0.38 | 15.90 | 2.47 | 4.2 / 5.5 |
| yee-pec-cube-32 [40, 400] (271, 2) | one:1x16 | 13.83 [13.77-13.99] | 1.00 | - | 14.3 | 1.05 | 10.87 | 1.11 | 2.1 |
| | one:1+2x8 | 12.47 [12.33-12.58] | 1.11 | - | 14.7 | 0.34 | 10.90 | 1.11 | 3.1 / 4.2 |
| | two:1x16 | 15.24 [15.13-15.62] | 0.91 | 1.00 | 15.7 | 1.27 | 11.65 | 1.36 | 2.1 |
| | **two:1+2x8** | **10.15** [10.12-10.37] | **1.36** | **1.50** | 12.4 | 0.34 | 8.36 | 1.35 | 3.1 / 4.1 |
| yee-pec-box-1e5 [40, 400] (108; 1 slice, 2 with helpers) | one:1x16 | 5.839 [5.828-5.873] | 1.00 | - | 6.21 | 0.81 | 3.75 | 0.90 | 1.6 |
| | one:1+2x8 | 6.807 [6.785-6.875] | 0.86 | - | 8.91 | 0.36 | 5.03 | 1.32 | 2.5 / 3.4 |
| | two:1x16 | 6.367 [6.291-6.477] | 0.92 | 1.00 | 6.65 | 0.93 | 3.95 | 1.01 | 1.8 |
| | **two:1+2x8** | **5.553** [5.513-5.598] | 1.05 | **1.15** | 7.64 | 0.37 | 3.66 | 1.45 | 2.7 / 3.4 |

What the claims say:

* **Two sockets: the multi-process mode is faster than one process spread over both.**
  - 1e6 nearest: 1.11x / 1.09x (68.5 vs 75.8 s, 59.0 vs 64.1 s). The certificate drops from 36.0 to
    29.2 s and from 29.9 to 24.2 s.
  - 1e5 3-D nearest: 1.10-1.20x. The certificate drops from 0.97-1.01 s to 0.73-0.77 s.
  - Windows with several slices: 1.15x, 1.50x, 1.74x.
  - The main factorization and the Krylov phase are the leader's as before (equal within noise).
* **Against hsolve on one socket** (the 16-core layout on these nodes):
  - 1e6: 1.23-1.26x. About half of it is the second socket's bandwidth for the leader: `two:1x16`
    alone is 1.13-1.14x.
  - Windows: 1.05-1.64x. The 1.05x is the window whose second slice exists only for the helpers.
  - 1e5 nearest: 0.92-1.00x (3-D) and 0.75x (cav2d).
* **On one socket the mode loses:** 0.89-0.93x on nearest targets and 0.86x on yee-pec-box-1e5's
  window. The helpers share the socket's memory bandwidth and pay their preparation. The mode still
  gains on the windows with many slices (1.11x, 1.16x), where the leader alone would factorize one
  slice after the other. `hsolve-mpirun` therefore runs the plain hsolve on a single NUMA node: no
  MPI job since the review fix (section 7.3).
* **Wall time.** Starting the MPI job (prterun, MPI_Init of three processes) adds ~1.7-2.0 s at 1e5:
  wall 3.7-4.6 s against 2.1-3.1 s.
  - The helpers' gain outweighs it at 1e6 (71.1 vs 76.7 s, 61.5 vs 64.9 s) and on the windows with
    several slices (21.8 vs 34.0 s, 12.4 vs 15.7 s).
  - It does not on the 1e5 nearest cases and the 108-eigenvalue window.
* **Memory.** The helpers hold their own copies of the problem and their counters while they count.
  - At 1e6 the process tree peaks at 18.3 / 14.6 GiB (sampled; the sum of the processes' own peaks
    is 29.5 / 24.5 GiB), against 13.2 / 11.3 GiB for hsolve. That is within the 32 GiB of a run.
  - With a 16 GiB limit the 1e6 cases run out of core anyway (P7), and two counts at once do not
    fit. Since the fix the leader decides that before it sends anything: section 7.2 shows the
    tree at the single process's peak.
* **Results are the single-process ones**: the same eigenvalues (every run valid against the complete
  references), the same certificate counts (`2 of the 2 counts made by helper processes`), the same
  slices.
* **The 2-D case and the GHIEP case get no helper work.**
  - cav2d-rect-pert-1e5: a 0.035 s main factorization against ~0.18 s of helper preparation, so the
    cost rule keeps the counts on the leader.
  - wg-square-1e5: GHIEP has no inertia counts.
  - Yet the multi-process leader's run is shorter than the plain process's on two sockets (cav2d
    0.401 vs 0.473 s). Section 4.1 separates that from the helpers.
* **A bug the campaign caught.** The first version gave each slice helper only the memory one slice
  needs. On yee-pec-cube-32's 167-pair slice its Krylov basis did not fit, so the helper halved its
  block width to 1 and missed 4 pairs (tools/eval: invalid, campaign C tag civ3). Helpers now get an
  equal share of what the budget leaves (e2c443d), measured above (giv3).

### 4.1 What the helpers do (tag f1e5, binary f265491 before the review fix, 5 repetitions, all valid)

Two controls on two sockets:
- `two:1x16r` is hsolve-mpi with ONE rank through the launcher (`--ranks 1`): the same start-up,
  Open MPI loaded, no helpers. Before the review fix this was also what the launcher did by itself
  on one NUMA node; section 7.3 has the fix.
- `two:1x16g` is plain hsolve with glibc's malloc keeping freed memory (`MALLOC_MMAP_MAX_=0`, no
  trimming), since Open MPI may change the allocator's behaviour.

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
  wg-square-1e5); the 1.18x of cav2d in the table above is that spread, not a gain.

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
| 1 + 2 x 8 (hsolve-mpi) | one socket | counts, slices | 0.92 / 0.93 | 0.89 / 0.92 / 0.90 | 1.16 / 1.11 / 0.86 |
| 1 x 16 (hsolve) | two sockets | - | 1.14 / 1.13 | 0.84 / 0.85 / 0.83 | 0.94 / 0.91 / 0.92 |
| **1 + 2 x 8 (hsolve-mpi)** | **two sockets** | counts, slices | **1.26 / 1.23** (1.11 / 1.09) | **1.00 / 0.98 / 0.92** (1.20 / 1.16 / 1.10) | **1.64 / 1.36 / 1.05** (1.74 / 1.50 / 1.15) |

(Binary of the review fix, 54e5219; tags g1e6, g1e5, giv1-3.)

Per phase in those runs (two sockets, hsolve -> hsolve-mpi):
- the certificate: 36.0 -> 29.2 s and 29.9 -> 24.2 s at 1e6, 0.97-1.01 -> 0.73-0.77 s at 1e5;
- the windows' slices (iterate): 25.5 -> 15.9 s (syn-mem27-46) and 11.7 -> 8.4 s (yee-pec-cube-32);
- analysis, main factorization and Krylov run on the leader as in hsolve and take the same time.

**Limits** (honest ones):

* **One socket: no gain.** Helpers on one socket share its memory bandwidth: 0.86-0.93x, except on
  windows with many slices (1.11-1.16x). `hsolve-mpirun` therefore runs the plain hsolve on one NUMA
  node, without an MPI job or its start-up (section 7.3). The judging machine (16 cores) may well be
  one socket, and then the multi-process mode changes nothing.
* **Two sockets: modest for nearest targets.** Only the certificate's two counts run concurrently -
  the main factorization and the Krylov run are one sequential computation, which the leader runs on
  all cores as hsolve does; at 1e6 the certificate is ~40-45% of the run, so 1.09-1.11x. Windows with
  several slices gain most (1.5-1.74x). GHIEP (no counts), the randomized certificate (no counts when
  it certifies), single-slice windows and problems too small for the helpers' preparation (cost
  rules) get nothing.
* **MPI start-up.** ~1.7-2.0 s of wall time (prterun, three MPI_Inits) before the solve. It outweighs
  the gain on the 1e5 nearest cases and the 108-eigenvalue window, but not at 1e6 or on the larger
  windows.
* **Memory.** While it works, each helper holds a copy of the problem and its own counter or factors.
  The process tree peaks 1.3-1.4x higher at 1e6 (18.3 vs 13.2 GiB, 14.6 vs 11.3 GiB).
  - The budget is charged with what the helpers hold for the solve. Their own idle processes (MPI,
    libraries, ~50 MB each) are not charged.
  - Under a tight limit the counts and slices stay on the leader. Since the review fix this is
    decided before the problem is sent, and the helpers drop it whenever the leader keeps the work.
    The leader then certifies wherever hsolve does (section 7.1), and at 1e6 under 16 GiB the tree
    stays at the single process's peak (section 7.2).
  - The prediction can still be too low near the edge. The helpers then drop the problem once
    they report their actual needs, and only their preparation is lost (section 3.1).
* **Emulated topology.** The two-socket layouts use 8 of each socket's 16 cores (the run's 16 cores);
  each socket then serves 8 cores with its whole memory bandwidth and L3, as a 2 x 8-core machine
  would, but a machine with smaller sockets may have less bandwidth per socket.
* **MPI MUMPS stays out.** A distributed main factorization gained 0.95-1.09x at 1e6 on two sockets
  (1.16-1.39x at 1e5) against the sequential MUMPS spread over both, would need a distributed main
  object in place of hsolve's, and loses to the per-socket counts for the certificate.
* **Failures.** A helper task that fails is redone by the leader (tests: fault bit 128, every kind);
  a helper process that dies ends the MPI job.
* **Other MPIs untested.** Only Open MPI 5.0.11 is installed on the nodes and in the offline package
  cache. The launcher and the tests pass Open MPI's options only to Open MPI and give the plan through
  the environment, so Hydra-based MPIs (MPICH, Intel MPI) should work. That is shown only with
  stand-in launchers (`tests/py/test_hsolve_mpirun.py`).

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
# 7 (review fix, at 54e5219; raw output in $HOME/p10mpi/fix/): memory-tight budgets and the launcher on
# one NUMA node (interleaved, each run its own guard --exclusive), then the end-to-end tags with event logs
bash tools/bench/mpi_review_runs.sh tight $HOME/p10mpi/fix/tight
bash tools/bench/mpi_review_runs.sh launch $HOME/p10mpi/fix/launch
O=$HOME/p10mpi/fix/runs
python3 tools/bench/mpi_layouts.py run --out $O --kind hsolve --events --tag g16 --reps 1 \
  --layouts two:1x16,two:1+2x8 --case syn-mem27-100 --root $D --mem-gb 16 --timeout 1500
python3 tools/bench/mpi_layouts.py run --out $O --kind hsolve --events --tag g1e5 --reps 5 --layouts $L \
  --case syn-mem27-46 --case yee-pec-box-1e5 --case ss-qa8fk-qa8fm --case cav2d-rect-pert-1e5 --timeout 600
#   giv1-3: as div1-3 with --out $O --events; g1e6: as c1e6 with --out $O --events; then eval / table --out $O
# one multi-process solve by hand (two sockets' worth of cores):
python3 tools/run/guard.py --exclusive --cores 8-15,16-23 --mem-gb 32 --timeout 900 -- \
  build/release/hsolve-mpirun -- --A data/cases/syn-mem27-46/A.npz --config data/cases/syn-mem27-46/config.json --out OUT
```

## 7. Review fixes (review p10mpi; branch p10mpi-fix)

The review of P10 found six problems. All six are fixed. Each fix comes with a regression test
(section 3, Tests) and the measurements below. The disposition and the commit of each:

| # | finding | disposition | commit |
|---|---|---|---|
| 1 | major: hsolve-mpi did not certify where hsolve certifies under the same budget. The helpers were prepared before the memory rule ran; their copies and analyses stayed until the end of the solve and were charged to the leader's count allowance. | fixed: memory rule predicted before sending, Drop / Dropped, budget charged above the helpers' idle level (sections 3, 3.1, 7.1, 7.2) | 1279778 |
| 2 | major: the window was cut into one slice per helper before it was known that a helper would take a slice | fixed: cut only when the cost rule passes and two helpers' slices are predicted to fit; the leader releases its main factors only when the helpers take the slices (section 7.1) | 1279778 |
| 3 | major: on one NUMA node the launcher started a one-rank MPI job (+1.7-1.9 s) instead of plain hsolve | fixed: the launcher execs the plain hsolve (section 7.3) | 8b1db53 |
| 4 | minor: the launcher and the ctest registration used Open MPI-only options | fixed: the plan reaches the ranks through the environment, with the rank read from OMPI_COMM_WORLD_RANK / PMIX_RANK / PMI_RANK / MV2_COMM_WORLD_RANK; Open MPI options are passed only to Open MPI (launcher and tests). The docs say only Open MPI was tested: no other MPI is installed or in the offline package cache, so MPICH / Intel MPI are covered by stand-in tests only | 8b1db53, a7c5583 |
| 5 | minor: THIRD_PARTY_NOTICES, DEPENDENCIES and FORMATS were not updated for hsolve-mpi | fixed: notices 3.7 (Open MPI and its run-time libraries, from hsolve-mpi's ldd and the conda packages' licenses), DEPENDENCIES P10 entry, FORMATS (fault bit 128, 0..255; report.parallel; --helper-counts / --helper-slices) | 2258ffd |
| 6 | minor: the certificate statement named the InertiaCounter even when the leader's fallback count used the main factorization | fixed: the statement names each count's source (count_sources) | cca48b1 |

### 7.1 Memory-tight budgets (findings 1 and 2; two sockets, 5 interleaved repetitions)

The review's experiment ran again on the fixed binary. Each run had its own guard: `--exclusive
--cores 8-15,16-23 --mem-gb 32` and the same explicit `--memory-budget-gb` for both modes. Plain
hsolve used 16 threads; the other mode was `hsolve-mpirun --ranks 3` (`tools/bench/mpi_review_runs.sh
tight`). "Before" is the review's result at f265491.

| case, budget | mode | runs ok / certified | solver s | wall s | certificate s | leader factorizations, slices | helper work | before the fix |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 nearest, 1.0 GiB | hsolve | 5/5, 5/5 | 2.286 [2.151-2.476] | 2.44 | 1.033 | 1, - | - | 5/5 certified |
| | hsolve-mpi | 5/5, **5/5** | 2.228 [2.153-2.263] | 4.08 | 0.964 | 1, - | none: predicted 809 MB > 578 MB allowance, nothing sent | **0/5 certified** (rc 1: certificate NOT COMPLETED) |
| syn-mem27-46 nearest, 1.1 GiB | hsolve | 5/5, 5/5 | 2.280 [2.157-2.434] | 2.44 | 1.015 | 1, - | - | |
| | hsolve-mpi | 5/5, 5/5 | 2.273 [2.167-2.299] | 4.14 | 1.008 | 1, - | none (predicted) | |
| yee-pec-box-1e5 [40, 400], 2.5 GiB | hsolve | 5/5, 5/5 | 6.267 [5.966-6.418] | 6.55 | 0.983 | 3, 1 | - | 6.263 s |
| | hsolve-mpi | 5/5, 5/5 | **6.265** [6.127-6.298] | 8.24 | 0.987 | **3, 1** | none: the cut for the helpers skipped (2 x 1185 MB > 1840 MB predicted) | **8.006 s, 5 factorizations, 2 slices (0.78x)** |

* The multi-process mode now computes exactly what hsolve computes when the budget leaves no room
  for the helpers. The leader counts on its InertiaCounter and certifies; the window keeps
  hsolve's single slice and its 3 factorizations. Solver times are equal. The wall time adds the
  MPI start-up of the three-rank job (~1.7-1.9 s).
* A lane sweep (8 cores: the leader on 8 threads, two helpers on 4) covered the edge cases:
  - syn-mem27-46 nearest, 0.8-1.5 GiB: both modes certify at every budget. The counts are declined
    from the prediction at 0.8-1.0 GiB. At 1.1 GiB the prediction passed but the helpers' own
    estimate did not fit (2 x 320 MB > 607 MB): the helpers dropped the problem, and the leader
    certified on its counter. That run wasted the preparation (certificate 1.32 vs 1.03 s), the
    one-sided error of section 3.1. From 1.2 GiB the helpers count.
  - yee-pec-box-1e5 [40, 400], 1.5-2.5 GiB: the plan equals hsolve's (1 slice, 3 factorizations,
    6.1 s each). From 3.0 GiB the helpers take two slices. The review saw them used only from
    3.5 GiB, with a wasted cut below that.

### 7.2 The judging scale under 16 GiB (finding 1; g16: syn-mem27-100, two sockets, guard --mem-gb 16, 1 run each)

With a 16 GiB limit the budget is 13.6 GiB, and the 1e6 case runs out of core in both modes (P7).

| layout | solver s | certificate s | wall s | tree peak GiB (sampled) | helper work | before the fix (review, rB16) |
|---|---|---|---|---|---|---|
| two:1x16 (hsolve) | 170.9 | 36.3 | 171.8 | 7.26 | - | 180.7 s, 7.1 GiB |
| two:1+2x8 (hsolve-mpi) | 171.7 | 34.9 | 174.5 | **7.11** | none: predicted 16119 MB > 11942 MB, nothing sent (share 0, wait 0) | 175.6 s; helpers prepared (share 0.52 s + wait 1.48 s) then declined; **peak 10.1 GiB** |

The helpers no longer receive a 1e6 problem whose counts cannot fit. The process tree stays at
the single process's peak.

### 7.3 The launcher on one NUMA node (finding 3; one socket, cores 16-31, 5 interleaved repetitions)

The launcher without `--ranks` (`tools/bench/mpi_review_runs.sh launch`) was run against plain
`hsolve` with the same arguments. Wall time is the guard's.

| case | mode | solver s | wall s | report.parallel.processes |
|---|---|---|---|---|
| syn-mem27-46 | hsolve | 2.012 [1.981-2.063] | 2.175 [2.124-2.245] | 1 |
| | hsolve-mpirun | 2.003 [1.985-2.030] | 2.177 [2.126-2.275] | 1 |
| cav2d-rect-pert-1e5 | hsolve | 0.292 [0.289-0.306] | 0.408 [0.404-0.436] | 1 |
| | hsolve-mpirun | 0.295 [0.289-0.299] | 0.420 [0.403-0.468] | 1 |

Before the fix this started a one-rank MPI job. In section 4.1 that job (two:1x16r) cost +1.7-1.9 s
of wall time per run and 0.95-0.99x of solver time. On a single-socket judging machine,
`hsolve-mpirun` is now the plain hsolve.

### 7.4 End to end after the fix (32 GiB; section 4)

The claims of section 4 were measured again on the fixed binary (54e5219). The later commit 4a7029c
changes only an error path that these runs never take: a helper that fails to prepare is dropped
as well. Tags:
- g1e5: 1e5 nearest;
- giv1-3: the windows;
- g1e6: 1e6 nearest.

Each tag ran 5 interleaved repetitions of the four layouts, and every run was judged by tools/eval.
At 32 GiB the helpers get the same work as before the fix. The drops add 0.008-0.018 s at 1e5 and
0.012-0.016 s at 1e6 (they free 1.2-1.4 GB of the helpers' copies and analyses). The predictions
pass with room:
- syn-mem27-100: 2 x (0.45 x 16101 + 2.5 x 330) = 16.1 GB predicted, and the helpers then report
  2 x 7965 MB of counts;
- yee-pec-box-1e6: the helpers report 2 x 6426 MB.

The two-socket speedup over hsolve on the same cores, before and after the fix:

| target | before (f265491 / 1c39bf6; d1e5, div1-3, c1e6) | after (54e5219; g1e5, giv1-3, g1e6) |
|---|---|---|
| syn-mem27-100 / yee-pec-box-1e6 nearest | 1.10 / 1.10 (68.28 / 58.49 s) | 1.11 / 1.09 (68.53 / 58.97 s) |
| syn-mem27-46 / yee-pec-box-1e5 / ss-qa8fk-qa8fm nearest | 1.14 / 1.10 / 1.06 (1.988 / 2.540 / 1.637 s) | 1.20 / 1.16 / 1.10 (2.008 / 2.483 / 1.654 s) |
| windows syn-mem27-46 / yee-pec-cube-32 / yee-pec-box-1e5 | 1.75 / 1.51 / 1.16 (18.78 / 10.15 / 5.548 s) | 1.74 / 1.50 / 1.15 (18.95 / 10.15 / 5.553 s) |

The multi-process times are unchanged within 2%. The larger 1e5 ratios come from the plain process
spread over two sockets, which was slower in this campaign (2.402 vs 2.260 s on syn-mem27-46; its
two-socket spread at 1e5, section 4.1). Every run of g1e5, giv1-3, g1e6 and g16 was valid
(tools/eval: 194/194).
