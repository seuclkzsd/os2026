# P9 "mpi": hsolve against tuned SLEPc at 1e6, and the upper bound of hybrid MPI + OpenMP MUMPS

P9 measurement task "mpi" (branch `p9-mpi` from main a831d79; no change to hsolve's code, defaults or
build). Two questions:

1. **Where we stand at 1e6.** On the two 3-D 1e6 cases in core (16 cores, 32 GiB), how does hsolve
   (certificate exact / auto / off) compare with SLEPc configured like the best 16-core configurations
   of the analogous 1e5 cases, phase by phase?
2. **Upper bound of MPI MUMPS for hsolve's factorizations.** At 1e6 a run is dominated by three numeric
   LDLᵀ factorizations on one pivot order (the main one and the certificate's two inertia-only counts).
   Would the MPI MUMPS of the os2026-ref env, with R ranks x T threads on the same 16 cores, factorize the
   same matrix on the same pivot order faster than the sequential MUMPS hsolve links with 16 OpenMP
   threads - by at least 1.3x, the bar for a design proposal?

**Answers.**

* **At 1e6 hsolve wins on syn-mem27-100 and loses on yee-pec-box-1e6 with the exact certificate**
  (medians of 3 interleaved runs, all valid): syn-mem27-100 **1.15x** (hsolve 108.0 s vs SLEPc 123.9 s),
  yee-pec-box-1e6 **0.74x** (92.4 vs 68.3 s). The exact certificate - two inertia-only factorizations,
  54.3 / 47.6 s, half of hsolve's run - is the whole gap: without it hsolve is 2.33x / 1.57x faster, with
  `certificate = auto` (the randomized check, P8) 1.73x / 1.04x. SLEPc's best yee configuration
  factorizes with BLR (`cntl_7 = 1e-10`): its numeric factorization costs what hsolve's does (24.4 vs
  24.5 s) and makes its 78 single-RHS solves cheap; on syn BLR made every SLEPc pair inaccurate (12 of
  12 above tol). hsolve's solves are 5-8x cheaper per right-hand side (blocks + BLAS hook).
* **MPI MUMPS is slower than hsolve's path at every layout - a negative result.** Same matrix, same
  pivot order (hsolve's SCOTCH order as PERM_IN), same MUMPS 5.8.2 and controls, 16 cores of the slot,
  medians of 3 interleaved runs: the numeric factorization keeping the factors runs at
  **0.84-0.85x** of hsolve's sequential 1 x 16 path with 2 ranks x 8 threads, 0.72-0.73x at 4 x 4,
  0.60-0.63x at 8 x 2 and 0.59x at 16 x 1; the inertia-only counts the same (0.83 / 0.71-0.72 /
  0.57-0.61 / 0.56-0.57x); 16-RHS block solves 0.79-0.94x; memory grows with the ranks (process-tree
  peak 14.0 -> 28.8 GiB at 16 x 1 on syn-mem27-100). MPI MUMPS with one rank equals the sequential
  library (1.00-1.01x), so the two builds are the same code and the loss is the MPI parallelism itself.
  Nothing changes the verdict (section 3): MUMPS' own METIS order (2 x 8: 0.85-0.90x), a parallel
  ParMETIS analysis (analysis + factorization 1.2-1.6x hsolve's), distributed input (0.84-0.85x), the
  ScaLAPACK root (0.54-0.80x, and a wrong inertia). The flops are balanced within 2-8% at 2 x 8 and
  4 x 4; the 16 cores of the socket are the limit: 8 -> 16 threads speed one factorization up only
  1.24x, and even the certificate's two counts as two concurrent 8-thread processes - the best case of
  any count server, no MPI traffic at all - are 0.92-0.93x of hsolve's sequential pair.
* Every MPI factorization with the root front outside ScaLAPACK (`ICNTL(13) = 1`, as hsolve sets it)
  returned the exact inertia (INFOG(12) = 20003 / 337613, the counts of the complete references);
  with the ScaLAPACK root (`ICNTL(13) = 0`, MUMPS' parallel default) INFOG(12) leaves out the root's
  negative pivots and is wrong (section 3.5) - unusable for a certificate.
* No configuration reaches 1.3x, so there is **no design proposal**: hsolve keeps the sequential MUMPS.
  The levers at 1e6 remain the factorization kernel itself and the ordering (P9 factspeed / ordering)
  and the certificate's cost (section 4).

## 1. Setup

* **Node.** c005 (2x Xeon Silver 4314, 2 x 16 cores, AVX-512, 251 GB; idle by
  `tools/dev/cluster_nodes.sh` before the campaign, and every timed run's guard checked its cores and
  their hyperthreads idle), measurement slot cores 16-31 (NUMA node 1), `guard.py --exclusive --mem-gb
  32`, one run at a time, 16 threads; 2026-10-14 09:26-13:22, the batches chained one guard call after
  the other. Builds, conversions, evaluations and the test suite in the build/test lane (cores 8-15).
* **hsolve.** main a831d79 (this branch adds only measurement tools), `tools/dev/build.sh release`
  (conda gcc 13.4, `-O3`, BLAS hook linked, route all), md5 5f93952b3005dfaa8ec96fc97d27d23b; sequential
  MUMPS 5.8.2 (conda-forge `mumps-seq`, OpenMP), SCOTCH 7.0.11 (int64), METIS 5.1.0, OpenBLAS 0.3.34
  (OpenMP).
* **SLEPc / MPI MUMPS.** env os2026-ref: PETSc / SLEPc 3.26.0, MUMPS 5.8.2 (`mumps-mpi`, OpenMP),
  ScaLAPACK 2.2.0, PT-SCOTCH 7.0.11 (int32), ParMETIS 4.0.3, METIS 5.1.0, Open MPI 5.0.11, OpenBLAS
  0.3.34 (OpenMP). The MUMPS version equals hsolve's, so no separate apples-to-apples version run was
  needed; its 1 x 16 run is in every table as a check of the builds.
* **Cases** (`$HOME/p7_p7-membudget/data` on the cluster; P7): syn-mem27-100 (HEP, n = 1e6, 27-point
  3-D stencil, sigma = 0, k = 12; K = A has 20003 negative eigenvalues), yee-pec-box-1e6 (GHEP, n =
  1027733, empty PEC cavity, sigma = 52.9 above the gradient null space, k = 10; K(sigma) = A - sigma B
  has 337613 negative eigenvalues: the 337610-dimensional null space and 3 resonances). Both references
  are complete spectra, so every run is checked against the exact answer and every count against the
  exact inertia.

## 2. Head-to-head at 1e6 (task 1)

### 2.1 The SLEPc configuration

No SLEPc profile existed at 1e6. Starting from the best 16-core configurations of the analogous 1e5
cases (`results/baseline_tuning_p16.md`: syn-mem27-46 `1x16` with the defaults; yee-pec-box-1e5
`4x4/core eps_nev=k+2` with BLR `cntl_7 = 1e-10` and a ParMETIS analysis), five variants per case were
screened, one run each, through `compare.py --solvers slepc` (profile files: variant 1 = the chosen one
is `results/cmp-p9-1e6-profiles.json`):

| variant | case | configuration | solver s | valid | analysis | numeric | solves (#/s) | peak GiB |
|---|---|---|---|---|---|---|---|---|
| 1 | syn-mem27-100 | `1x16` (the 1e5 best) | **124.76** | yes | 14.29 | 23.24 | 118 / 85.14 | 12.6 |
| 2 | syn-mem27-100 | `4x4/core icntl_28=2 icntl_29=2` | - | no: MUMPS INFOG(1) = -9 (workspace) in the factorization | - | - | - | 18.3 |
| 3 | syn-mem27-100 | `16x1/core icntl_28=2 icntl_29=2` | 151.32 | yes | 4.27 | 51.19 | 118 / 93.97 | 30.4 |
| 4 | syn-mem27-100 | `1x16 cntl_7=1e-10 icntl_35=2` (BLR) | 91.31 | **no**: 12 of 12 pairs eta > tol (max 2.0e-10) | 14.70 | 26.21 | 94 / 48.75 | 11.9 |
| 5 | syn-mem27-100 | `4x4/core` (sequential analysis) | 137.08 | yes | 15.48 | 36.58 | 119 / 83.10 | 17.5 |
| 1 | yee-pec-box-1e6 | `4x4/core eps_nev=k+2 cntl_7=1e-10 icntl_28=2 icntl_29=2 icntl_35=2` (the 1e5 best) | **68.35** | yes | 6.74 | 24.45 | 78 / 35.46 | 15.4 |
| 2 | yee-pec-box-1e6 | `4x4/core eps_nev=k+2 icntl_28=2 icntl_29=2` (no BLR) | 84.94 | yes | 6.43 | 33.06 | 64 / 44.14 | 16.3 |
| 3 | yee-pec-box-1e6 | `1x16 eps_nev=k+2` | 85.56 | yes | 13.48 | 24.15 | 64 / 46.31 | 12.3 |
| 4 | yee-pec-box-1e6 | `1x16 eps_nev=k+2 cntl_7=1e-10 icntl_35=2` (BLR) | 77.09 | yes | 13.86 | 22.97 | 78 / 38.21 | 10.6 |
| 5 | yee-pec-box-1e6 | `4x4/core eps_nev=k+2 cntl_7=1e-10 icntl_35=2` (sequential analysis) | 70.87 | yes | 13.98 | 25.17 | 65 / 30.41 | 16.8 |

(MUMPS controls carry the prefix `st_mat_mumps_`; solver s = `-log_view` setup + solve + vectors.)
The 1e5 winners are also the 1e6 winners. Already here the MPI layouts factorize slower than one
rank with 16 threads (syn-mem27-100: 36.6 s at 4 x 4, 51.2 s at 16 x 1 against 23.2 s; yee-pec-box-1e6
without BLR 33.1 s at 4 x 4 against 24.1 s) - section 3 measures this on hsolve's own matrix and order.

### 2.2 Result

`results/cmp-p9-1e6.md` / `.json` (`compare.py`, 3 interleaved repetitions, tools/eval on every run
against the complete references; `--mem-gb 32`, so hsolve ran its automatic budget of 27.2 GiB and
factorized in core):

| case | solver | solver s, median [min-max] | valid | peak GiB | speedup vs SLEPc |
|---|---|---|---|---|---|
| syn-mem27-100 | hsolve (exact) | 108.02 [103.47-108.81] | 3/3 | 15.2 | **1.15x** |
| syn-mem27-100 | hsolve-auto (randomized) | 71.79 [69.27-72.48] | 3/3 | 16.2 | 1.73x |
| syn-mem27-100 | hsolve-nocert | 53.18 [52.19-56.43] | 3/3 | 15.5 | 2.33x |
| syn-mem27-100 | SLEPc `1x16` | 123.90 [117.27-124.95] | 3/3 | 12.6 | - |
| yee-pec-box-1e6 | hsolve (exact) | 92.44 [90.77-92.60] | 3/3 | 13.2 | **0.74x** |
| yee-pec-box-1e6 | hsolve-auto (randomized) | 65.49 [64.64-66.31] | 3/3 | 14.0 | 1.04x |
| yee-pec-box-1e6 | hsolve-nocert | 43.48 [42.91-43.88] | 3/3 | 13.0 | 1.57x |
| yee-pec-box-1e6 | SLEPc `4x4/core` BLR | 68.27 [59.80-68.31] | 3/3 | 15.4 | - |

All eigenvalue sets agree with the complete references to 3e-17..2e-14 (relative to |lambda| + ||A||/||B||).

### 2.3 Per-phase split

`tools/bench/compare_phases.py results/raw/cmp-p9-1e6-<case>` (medians over the 3 runs, seconds;
hsolve's phases are exclusive and add up to its total, SLEPc's are `-log_view` events):

| phase | syn hsolve | syn auto | syn nocert | syn SLEPc | yee hsolve | yee auto | yee nocert | yee SLEPc |
|---|---|---|---|---|---|---|---|---|
| total | 108.02 | 71.79 | 53.18 | 123.90 | 92.44 | 65.49 | 43.48 | 68.27 |
| MUMPS analysis | 6.08 | 5.86 | 5.88 | 14.29 | 4.49 | 4.51 | 4.49 | 6.72 |
| main numeric factorization | 28.49 | 29.41 | 28.14 | 23.23 | 24.46 | 23.54 | 23.15 | 24.40 (BLR) |
| hsolve factor phase (ordering choice, analyses, main factorization) | 35.32 | 35.99 | 34.77 | - | 29.94 | 29.08 | 28.71 | - |
| iterate (Krylov incl. its solves) | 17.31 | 17.79 | 17.66 | 86.29 (EPSSolve) | 14.23 | 14.22 | 14.23 | 29.51 (EPSSolve) |
| solves: RHS / s | 168 / 15.72 | 260 / 31.09 | 168 / 16.05 | 117 / 84.26 | 136 / 11.97 | 258 / 30.24 | 136 / 11.97 | 78 / 35.48 |
| s per RHS | 0.094 | 0.120 | 0.096 | 0.720 | 0.088 | 0.117 | 0.088 | 0.455 |
| certificate | **54.26** (2 exact counts) | 16.94 (randomized) | 0 | - | **47.55** (2 exact counts) | 21.69 (randomized) | 0 | - |
| verify + setup | 0.75 | 0.74 | 0.74 | - | 0.54 | 0.51 | 0.52 | - |
| orthogonalization / vectors | - | - | - | 1.75 / 0.03 | - | - | - | 1.18 / 7.32 |

* **The certificate is the gap.** hsolve's run is three factorizations of ~25-28 s each on one pivot
  order (main + two inertia-only counts): 28.5 + 54.3 s of 108 s on syn-mem27-100, 24.5 + 47.6 of 92.4
  s on yee-pec-box-1e6. SLEPc factorizes once. Everything else hsolve does faster: the analysis
  (SCOTCH 4.5-6.1 s vs MUMPS' sequential METIS 14.3 s / ParMETIS 6.7 s) and the Krylov phase (17.3 vs
  86.3 s, 14.2 vs 29.5 + 7.3 s): hsolve's block solves cost 0.09 s per right-hand side against SLEPc's
  0.72 s (full-rank factors, one RHS per call) and 0.46 s (BLR factors).
* **SLEPc's yee configuration wins with BLR.** Low-rank compression at `cntl_7 = 1e-10` keeps SLEPc's
  4 x 4 factorization at 24.4 s (33.1 s without BLR) and its solves at 0.46 s per RHS. hsolve does not
  use BLR by default (rejected at 1e5 in P1); a BLR factor's inertia is that of the compressed matrix,
  so the exact certificate would still need its own double-precision full-rank counts (DESIGN §3.4) -
  the user's decision for the default.
* **SLEPc's 1 x 16 numeric factorization of syn-mem27-100 is faster than hsolve's** (23.2 vs 28.5 s):
  MUMPS' automatic ordering takes METIS there (INFOG(7) = 5, section 3.5), hsolve's adaptive choice is
  SCOTCH. In the same bench (section 3.3) METIS needs 6.76e12 flops and 23.1 s against 8.95e12 flops and
  28.8 s for the SCOTCH order - input for the ordering lever (P9 ordering).

## 3. Hybrid MPI + OpenMP MUMPS: the upper bound (task 2)

### 3.1 Method

* **Bench.** `bench/mpi_factor_bench.c`, one C source built twice by `tools/bench/mpi_build.sh`
  (not part of the CMake build): `build/mpi/mpi_factor_bench` against the MPI MUMPS of os2026-ref and
  `build/mpi/seq_factor_bench` against the sequential MUMPS hsolve links (libseq stubs). Both use
  hsolve's MUMPS controls (src/factor/mumps_factorization.cpp with the driver defaults): SYM = 2, PAR = 1,
  ICNTL(8) = 7, ICNTL(13) = 1, ICNTL(14) = 35, ICNTL(16) = threads per rank, ICNTL(24) = 1 with CNTL(3) =
  1e-8, ICNTL(48) = 1 (L0-OpenMP, per rank in MPI), no BLR, in core, centralized input on rank 0
  (ICNTL(18) = 0, like hsolve). Per run, two MUMPS instances one after the other: **count** (analysis
  with ICNTL(31) = 1, one inertia-only numeric factorization - the InertiaCounter's work) and **full**
  (analysis, numeric factorization keeping the factors, three block solves of 16 right-hand sides from
  a centralized dense RHS, residual of the first column checked). Phases start and end at MPI barriers.
* **Matrix and order.** `tools/bench/mpi_dump_kernel.py` writes K(sigma) of the case (upper triangle of
  A - sigma B on the union pattern, exactly what hsolve's pencil hands MUMPS) and its exact inertia from
  the complete reference. hsolve's adaptive ordering picks SCOTCH on both cases (P7 logs, 16 threads);
  the sequential bench ran that analysis once at 16 threads (SCOTCH inside MUMPS, as hsolve) and wrote its
  SYM_PERM; every layout then analyses with it as PERM_IN (ICNTL(7) = 1), as hsolve's InertiaCounter
  does with the main order. Flops of this order: syn-mem27-100 8.95e12 (hsolve's P7 run: 8.04e12 - SCOTCH
  varies from run to run, factor.hpp), yee-pec-box-1e6 7.13e12 (6.91e12).
* **Layouts.** `tools/bench/mpi_factor_runs.sh`: `seq1x16` (sequential library, 16 threads unbound, as
  hsolve runs), `seq1x16h` (the same with hsolve's BLAS hook preloaded, as the hsolve executable links
  it), `RxT` = R ranks x T threads of the MPI library under `mpiexec --bind-to none`, each rank pinned
  by `tools/bench/mpi_bind.sh` to its block of the slot's 16 cores (rank r of R: cores 16 + r T ..
  16 + (r + 1) T - 1; the split checked with `Cpus_allowed_list` in the lane), threads placed with `OMP_PLACES=cores
  OMP_PROC_BIND=close` inside it, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = T`. Every run under its
  own guard (`--mem-gb 32`: the whole process tree), the layout order rotated per repetition, 3
  repetitions, one guard call per (case, repetition).

### 3.2 hsolve's order (PERM_IN)

`tools/bench/mpi_factor_table.py` over the runs (medians of 3; ratio = seq1x16 time / layout time, > 1
would be faster than hsolve's path; count = one inertia-only numeric factorization, factor = the
numeric factorization keeping the factors, solve = one 16-RHS block solve; GF/s from MUMPS' RINFOG(3)
of the factorization; MUMPS MB = INFOG(22) total over the ranks (INFOG(21) of the largest rank); peak
= the guard's process tree; inertia = INFOG(12) of every factorization of the layout against the exact
count):

**syn-mem27-100** (flops 8.95e12 with this order):

| layout | count s | ratio | factor s [min-max] | ratio | GF/s | solve s | ratio | MUMPS MB total (max rank) | count MB total | peak GiB | inertia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seq1x16 (hsolve) | 27.37 | 1.00 | 28.82 [28.78-29.06] | 1.00 | 310 | 1.038 | 1.00 | 15665 | 5945 | 14.04 | 20003 exact |
| seq1x16h (+ BLAS hook) | 27.36 | 1.00 | 28.79 | 1.00 | 311 | 0.936 | 1.11 | 15665 | 5945 | 14.02 | 20003 exact |
| 1x16 (MPI library) | 27.41 | 1.00 | 28.89 | 1.00 | 310 | 1.041 | 1.00 | 15665 | 5945 | 14.06 | 20003 exact |
| 2x8 | 33.07 | **0.83** | 34.32 [34.29-34.34] | **0.84** | 261 | 1.127 | 0.92 | 17297 (8661) | 8268 | 16.97 | 20003 exact |
| 4x4 | 38.54 | 0.71 | 39.79 [39.75-39.85] | 0.72 | 225 | 1.111 | 0.93 | 18215 (4810) | 10683 | 20.58 | 20003 exact |
| 8x2 | 45.09 | 0.61 | 45.92 [45.68-46.14] | 0.63 | 195 | 1.174 | 0.88 | 19916 (3044) | 11336 | 23.54 | 20003 exact |
| 16x1 | 48.33 | 0.57 | 48.64 [48.62-48.70] | 0.59 | 189 | 1.308 | 0.79 | 19888 (1678) | 11216 | 28.76 | 20003 exact |

**yee-pec-box-1e6** (flops 7.13e12):

| layout | count s | ratio | factor s [min-max] | ratio | GF/s | solve s | ratio | MUMPS MB total (max rank) | count MB total | peak GiB | inertia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seq1x16 (hsolve) | 22.69 | 1.00 | 24.15 [23.92-24.17] | 1.00 | 295 | 0.968 | 1.00 | 13340 | 4747 | 11.91 | 337613 exact |
| seq1x16h (+ BLAS hook) | 22.69 | 1.00 | 24.04 | 1.00 | 297 | 0.870 | 1.11 | 13340 | 4747 | 11.90 | 337613 exact |
| 1x16 (MPI library) | 22.64 | 1.00 | 23.98 | 1.01 | 297 | 0.969 | 1.00 | 13340 | 4747 | 11.93 | 337613 exact |
| 2x8 | 27.40 | **0.83** | 28.39 [28.25-28.42] | **0.85** | 251 | 1.030 | 0.94 | 16255 (8415) | 8061 | 15.43 | 337613 exact |
| 4x4 | 31.50 | 0.72 | 33.26 [32.79-33.26] | 0.73 | 214 | 1.070 | 0.90 | 16258 (5009) | 9096 | 18.80 | 337613 exact |
| 8x2 | 40.17 | 0.57 | 40.59 [40.57-40.60] | 0.60 | 176 | 1.112 | 0.87 | 21580 (3928) | 14116 | 25.24 | 337613 exact |
| 16x1 | 40.66 | 0.56 | 40.77 [40.77-40.80] | 0.59 | 175 | 1.225 | 0.79 | 19870 (2177) | 12307 | 26.99 | 337613 exact |

* Every MPI layout is slower than hsolve's sequential 1 x 16 path, monotonically with the number of
  ranks; 2 x 8 is the best at 0.83-0.85x. Repetitions agree to 1.5%.
* The MPI library run as one rank equals the sequential library (0.99-1.01x on every phase): the two
  conda builds of MUMPS 5.8.2 are the same code, and nothing but the MPI distribution changes.
* The analyses with the given order take 0.6-0.8 s in every layout (hsolve's InertiaCounter: 0.6-0.66
  s). The bench's sequential factorization reproduces hsolve's in its runs: syn-mem27-100 28.8 s here
  (8.95e12 flops) against 27.0-28.5 s in hsolve (8.04e12, its own SCOTCH order), counts 27.4 s against
  hsolve's certificate of 51.6-55.5 s for two; yee-pec-box-1e6 24.2 s against 24.0-24.6 s, counts 22.7
  s against 46.6-47.8 s for two.
* hsolve's BLAS hook (route all) speeds up the 16-RHS solves by 11% (0.936 vs 1.038 s, 0.870 vs 0.968 s)
  and leaves the factorization unchanged, as docs/BLAS.md says.
* Memory grows with the ranks: the MPI factorization holds 10-62% more MUMPS memory in total, the
  counts 1.4-3.0x (yee-pec-box-1e6 at 8 x 2: 4.7 -> 14.1 GB), and the process tree peaks at
  23.5-28.8 GiB at 8 x 2 and 16 x 1 against 12-14 GiB - within 32 GiB here, but not on a 16 GiB machine.

### 3.3 MUMPS' own METIS (sequential analysis)

The same with ICNTL(7) = 5 (each mode analyses for itself; medians of 3):

| case | layout | analysis s | count s | ratio | factor s | ratio | GF/s | solve s | ratio | flops | MUMPS MB total | peak GiB | inertia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | seq1x16 | 13.65 | 21.75 | 1.00 | 23.12 | 1.00 | 293 | 0.919 | 1.00 | 6.76e12 | 13746 | 11.98 | exact |
| syn-mem27-100 | 1x16 | 13.96 | 21.81 | 1.00 | 23.17 | 1.00 | 292 | 0.923 | 1.00 | 6.76e12 | 13746 | 11.98 | exact |
| syn-mem27-100 | 2x8 | 14.05 | 26.05 | **0.83** | 27.14 | **0.85** | 249 | 1.026 | 0.90 | 6.76e12 | 15010 | 14.37 | exact |
| syn-mem27-100 | 4x4 | 14.06 | 29.82 | 0.73 | 30.92 | 0.75 | 219 | 1.005 | 0.91 | 6.76e12 | 15444 | 15.79 | exact |
| syn-mem27-100 | 8x2 | 14.08 | 35.28 | 0.62 | 36.57 | 0.63 | 185 | 1.054 | 0.87 | 6.76e12 | 16987 | 19.39 | exact |
| syn-mem27-100 | 16x1 | 14.15 | 35.40 | 0.61 | 35.96 | 0.64 | 188 | 1.146 | 0.80 | 6.76e12 | 15630 | 22.47 | exact |
| yee-pec-box-1e6 | seq1x16 | 12.95 | 22.58 | 1.00 | 23.99 | 1.00 | 294 | 0.962 | 1.00 | 7.05e12 | 13467 | 11.57 | exact |
| yee-pec-box-1e6 | 1x16 | 13.30 | 22.56 | 1.00 | 23.96 | 1.00 | 294 | 0.963 | 1.00 | 7.05e12 | 13467 | 11.60 | exact |
| yee-pec-box-1e6 | 2x8 | 13.41 | 25.26 | **0.89** | 26.52 | **0.90** | 266 | 1.028 | 0.94 | 7.05e12 | 14694 | 13.65 | exact |
| yee-pec-box-1e6 | 4x4 | 13.42 | 29.39 | 0.77 | 30.40 | 0.79 | 232 | 1.071 | 0.90 | 7.05e12 | 16549 | 17.91 | exact |
| yee-pec-box-1e6 | 8x2 | 13.44 | 37.24 | 0.61 | 37.73 | 0.64 | 187 | 1.116 | 0.86 | 7.05e12 | 17056 | 20.14 | exact |
| yee-pec-box-1e6 | 16x1 | 13.55 | 37.71 | 0.60 | 37.89 | 0.63 | 186 | 1.213 | 0.79 | 7.05e12 | 17655 | 25.90 | exact |

* The order does not change the verdict: MPI 0.83-0.90x at 2 x 8, down to 0.60-0.64x at 16 x 1.
* **An observation for the ordering lever (P9 ordering):** on syn-mem27-100 MUMPS' METIS needs 24%
  fewer flops than this SCOTCH order (6.76e12 vs 8.95e12; hsolve's own SCOTCH runs 8.04e12) and
  factorizes in 23.1 s instead of 28.8 s (counts 21.8 vs 27.4 s), for an analysis of 13.7-13.9 s
  instead of SCOTCH's ~6 s. Over hsolve's analysis and three factorizations that is about 13.9 + 23.1 +
  2 x 21.8 = 81 s against 6.1 + 28.8 + 2 x 27.4 = 90 s with this SCOTCH order, and against 6.0 + 27.0 +
  51.8 = 85 s in hsolve's P7 run (SCOTCH at 8.04e12); hsolve's cost model priced METIS at 128 s against
  SCOTCH's 109 s on this case (P7 log, `candidates`). On yee-pec-box-1e6 METIS and SCOTCH need the same flops (7.05e12 vs 7.13e12) and
  SCOTCH's analysis is 8 s cheaper. This is also why SLEPc's 1 x 16 factorization of syn-mem27-100
  takes 23.2 s against hsolve's 28.5 s (section 2.3): MUMPS' automatic choice there is METIS (section 3.5).

### 3.4 Parallel analysis (ParMETIS)

ICNTL(28) = 2, ICNTL(29) = 2 (what SLEPc's yee profile uses), factors kept, 1 run per layout (ratio
against the sequential path of section 3.2):

| case | layout | analysis s | factor s | vs seq1x16 (hsolve's order, 28.82 / 24.15 s) | flops | GF/s | solve s | peak GiB | inertia |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 2x8 | 11.32 | 29.40 | 0.98 | 7.35e12 | 250 | 1.057 | 15.50 | exact |
| syn-mem27-100 | 4x4 | 7.24 | 35.70 | 0.81 | 7.83e12 | 219 | 1.043 | 17.62 | exact |
| syn-mem27-100 | 8x2 | 5.41 | 46.85 | 0.62 | 8.66e12 | 185 | 1.138 | 24.36 | exact |
| syn-mem27-100 | 16x1 | 4.90 | 49.50 | 0.58 | 9.63e12 | 195 | 1.329 | 28.03 | exact |
| yee-pec-box-1e6 | 2x8 | 10.65 | 27.11 | 0.89 | 7.05e12 | 260 | 1.047 | 13.20 | exact |
| yee-pec-box-1e6 | 4x4 | 6.92 | 31.94 | 0.76 | 7.34e12 | 230 | 1.082 | 16.17 | exact |
| yee-pec-box-1e6 | 8x2 | 5.16 | 35.72 | 0.68 | 7.06e12 | 198 | 1.122 | 18.93 | exact |
| yee-pec-box-1e6 | 16x1 | 4.40 | 36.78 | 0.66 | 6.99e12 | 190 | 1.240 | 22.82 | exact |

The parallel analysis scales (11.3 -> 4.9 s) but its orders cost more flops with more ranks (up to
1.4x METIS' on syn-mem27-100), and the factorization runs at the same 190-260 GF/s as with the other
orders: analysis + factorization
is 40.7-54.4 s (syn-mem27-100) and 37.8-41.2 s (yee-pec-box-1e6) against hsolve's 34.9 / 28.7 s
(SCOTCH analysis ~6.1 / 4.5 s + factorization).

### 3.5 ScaLAPACK root, distributed input, MUMPS' automatic ordering

hsolve's order, factors kept and counts, medians of 2 interleaved runs (each set with its own seq1x16
baseline: 28.76 / 23.95 s and 28.77 / 23.99 s, as in section 3.2):

| case | variant | layout | count s | ratio | factor s | ratio | flops | peak GiB | INFOG(12) |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | ScaLAPACK root, ICNTL(13) = 0 | 2x8 | 40.03 | 0.68 | 41.25 | 0.70 | 9.35e12 | 18.02 | **19535 (exact 20003)** |
| syn-mem27-100 | ScaLAPACK root | 4x4 | 41.18 | 0.66 | 42.33 | 0.68 | 9.35e12 | 20.99 | 19535 |
| syn-mem27-100 | ScaLAPACK root | 8x2 | 47.16 | 0.58 | 48.54 | 0.59 | 9.35e12 | 24.33 | 19535 |
| syn-mem27-100 | ScaLAPACK root | 16x1 | 50.70 | 0.54 | 51.22 | 0.56 | 9.58e12 | 29.41 | 19535 |
| yee-pec-box-1e6 | ScaLAPACK root | 2x8 | 29.00 | 0.78 | 30.02 | 0.80 | 7.32e12 | 15.44 | **333223 (exact 337613)** |
| yee-pec-box-1e6 | ScaLAPACK root | 4x4 | 32.82 | 0.69 | 34.34 | 0.70 | 7.32e12 | 19.24 | 333219 |
| yee-pec-box-1e6 | ScaLAPACK root | 8x2 | 41.37 | 0.55 | 41.99 | 0.57 | 7.32e12 | 25.82 | 333219 |
| yee-pec-box-1e6 | ScaLAPACK root | 16x1 | 41.85 | 0.54 | 41.95 | 0.57 | 7.32e12 | 27.52 | 333219 |
| syn-mem27-100 | distributed input, ICNTL(18) = 3 | 2x8 | 32.96 | 0.83 | 34.32 | 0.84 | 8.95e12 | 17.00 | 20003 |
| syn-mem27-100 | distributed input | 4x4 | 38.26 | 0.71 | 39.60 | 0.73 | 8.95e12 | 20.65 | 20003 |
| yee-pec-box-1e6 | distributed input | 2x8 | 27.33 | 0.83 | 28.32 | 0.85 | 7.13e12 | 15.46 | 337613 |
| yee-pec-box-1e6 | distributed input | 4x4 | 31.40 | 0.72 | 33.20 | 0.72 | 7.13e12 | 18.88 | 337613 |

* **The ScaLAPACK root is no way out.** MUMPS' parallel default factorizes the root front with
  ScaLAPACK (an LU: 3-4% more flops here), whose pivots INFOG(12) does not count - the inertia is short
  by 468 and 4390-4394 negative eigenvalues, as the MUMPS 5.8.2 user guide says under ICNTL(13)
  ("Processing the root sequentially (ICNTL(13) > 0) can be useful when the user is interested in the
  inertia"; with ICNTL(13) = 1 MUMPS splits the root instead "to enable node level parallelism"). It would void the
  certificate and is slower anyway (0.54-0.80x). The same was seen at 1e5 (syn-mem27-46, 2 x 4: 1842
  against 1950).
* **Distributed input** (each rank hands MUMPS its slice of the entries) changes nothing (0.83-0.85x
  at 2 x 8; hsolve's centralized input is not the bottleneck).
* **MUMPS' automatic ordering** (ICNTL(7) = 7, what SLEPc's `1x16` profile runs) takes METIS on both
  cases (INFOG(7) = 5; same flops and times as section 3.3).

### 3.6 Why MPI does not pay on one 16-core socket

**Load balance** (bench 872d39a: the elimination flops RINFO(3) of the busiest rank over the mean of the
ranks; hsolve's order, factors kept, 1 run per layout):

| case | layout | factor s | ratio | GF/s | flop balance (max / mean) | ideal time at seq1x16's per-core rate, balanced / with this balance |
|---|---|---|---|---|---|---|
| syn-mem27-100 | seq1x16 | 28.85 | 1.00 | 310 | 1.00 | 28.9 |
| syn-mem27-100 | 2x8 | 34.56 | 0.83 | 259 | 1.02 | 28.9 / 29.4 |
| syn-mem27-100 | 4x4 | 39.93 | 0.72 | 224 | 1.08 | 28.9 / 31.2 |
| syn-mem27-100 | 8x2 | 46.29 | 0.62 | 193 | 1.12 | 28.9 / 32.3 |
| syn-mem27-100 | 16x1 | 48.68 | 0.59 | 189 | 1.17 | 29.6 / 34.6 |
| yee-pec-box-1e6 | seq1x16 | 24.19 | 1.00 | 295 | 1.00 | 24.2 |
| yee-pec-box-1e6 | 2x8 | 28.40 | 0.85 | 251 | 1.05 | 24.2 / 25.4 |
| yee-pec-box-1e6 | 4x4 | 32.88 | 0.74 | 217 | 1.06 | 24.2 / 25.6 |
| yee-pec-box-1e6 | 8x2 | 40.70 | 0.59 | 175 | 1.19 | 24.2 / 28.8 |
| yee-pec-box-1e6 | 16x1 | 40.82 | 0.59 | 175 | 1.32 | 24.2 / 31.9 |

**The socket is the limit, not the parallelism** (commands in section 5; one slot guard call, 2 runs per case;
hsolve's order, sequential MUMPS, inertia-only counts, every count exact):

| case | one count, 16 threads | one count alone, 8 threads (16-23) | two counts at once, 8 + 8 threads (16-23, 24-31): each / pair wall | sequential pair at 16 threads | pair: sequential / concurrent |
|---|---|---|---|---|---|
| syn-mem27-100 | 27.37 | 33.93-34.00 | 58.95-59.05 / 59.79-59.86 | 54.74 | **0.92** |
| yee-pec-box-1e6 | 22.69 | 28.06 | 47.95-48.33 / 48.94-49.06 | 45.38 | **0.93** |

* The MPI layouts lose 15-41% of the factorization rate although their flops are balanced within 2-8%
  at 2 x 8 and 4 x 4: at seq1x16's per-core rate (19.4 / 18.4 GF/s) the work of the busiest rank would
  take 29.4 / 25.4 s at 2 x 8, against the measured 34.6 / 28.4 s. The imbalance (up to 1.17-1.32 at
  16 x 1) explains only part of the loss; the rest is what the distribution itself costs: MUMPS maps
  the large fronts at the top of the tree - where a 3-D 1e6 factorization spends most of its flops - on
  several processes as 1-D row-distributed (type 2) nodes, whose master factors the pivot block and
  ships panels and contribution blocks to the others as MPI messages (copies through shared memory),
  and with ICNTL(13) = 1 the root is a chain of such nodes; inside one process the same fronts are
  factorized in place by multithreaded BLAS 3.
* Doubling the threads of one factorization from 8 to 16 gains only 1.24x (34.0 -> 27.4 s,
  28.1 -> 22.7 s), and two 8-thread factorizations side by side each slow down 1.7x (34.0 -> 59.0 s,
  28.1 -> 48.2 s): the factorization is limited by what the 16 cores of the socket share (memory
  bandwidth, the shared L3, the AVX-512 all-core clock), not by a lack of parallel work. So the
  certificate's two counts run concurrently as two processes - the best case of any count server or
  helper-rank scheme, without any MPI communication - are slower than hsolve's sequential pair at 1e6
  as well (0.92-0.93x; P8 measured 0.99-1.12x on the 3-D 1e5 cases).

## 4. Conclusion (task 3): negative, no design proposal

* **Bar not reached.** The best MPI layout, 2 ranks x 8 threads, factorizes at 0.83-0.90x of hsolve's
  sequential MUMPS on the same 16 cores (any order, any input distribution), and every layout with more
  ranks is slower still; the inertia-only counts and the block solves follow. Nothing approaches the
  1.3x that would justify an MPI helper (MPI singleton + `MPI_Comm_spawn`, or a count server), so no
  design is proposed and docs/DESIGN.md is unchanged.
* **Projected effect, had hsolve adopted the best layout** (2 x 8, hsolve's order; section 2.3 phases
  scaled by the section 3.2 ratios): syn-mem27-100 108.0 -> ~125 s (6.1 + 28.5 / 0.84 + 54.3 / 0.83 +
  (17.3 - 15.7) + 15.7 / 0.92 + 1.1), yee-pec-box-1e6 92.4 -> ~106 s - 0.86-0.87x. At 1e5, where hsolve
  takes 0.3-3 s, the helper ranks alone would cost about as much as the whole solve: starting an Open
  MPI 5.0.11 job that only calls MPI_Init / MPI_Finalize takes 1.65-1.80 s on c005 (1-8 ranks, 3 runs
  each, lane; section 5), before any matrix is distributed.
* **Not an artefact of the setup.** The MPI library with one rank equals the sequential one; the
  ranks were pinned to disjoint core blocks of the slot; the result holds for hsolve's order, MUMPS'
  METIS and ParMETIS, centralized and distributed input; the exact inertia requires the sequential root
  (ICNTL(13) = 1), and the ScaLAPACK root is slower too; SLEPc's own screening shows the same ranking
  (section 2.1). The spread over repetitions is under 1.5%.
* **Where the 1e6 time can come from instead** (inputs for the other P9 levers, not measured further
  here): the factorization kernel (310 / 295 GF/s of a ~600 GF/s FP64 peak of 16 Silver 4314 cores at
  AVX-512 frequency; P9 factspeed), the ordering (METIS saves 24% of the flops of SCOTCH on
  syn-mem27-100, section 3.3; P9 ordering), and the certificate's cost - the exact counts are the whole
  gap to SLEPc on yee-pec-box-1e6 (0.74x with them, 1.57x without), where SLEPc's BLR factorization
  wins; `certificate = auto` (P8, randomized) gives 1.73x / 1.04x.

## 5. Commands

On c005 from the work tree of this branch (`tools/dev/cluster_run.sh c005 -- ...`, HS_CLUSTER_NAME=p9-mpi),
`D=$HOME/p7_p7-membudget/data`, `S=$HOME/p9_p9-mpi` (raw output there and in `results/raw/`, not committed).
Slot calls hold `guard.py --exclusive --timeout 2700 --mem-gb 32`; everything else ran in the build/test lane.

```bash
tools/dev/build.sh release && bash tools/bench/mpi_build.sh            # lane
# K(sigma) and its exact inertia (lane), per case:
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 900 -- \
  python3 tools/bench/mpi_dump_kernel.py --root $D --case syn-mem27-100 --out $S/k/syn-mem27-100.K.bin
# SLEPc screening (slot), variant v = 1..5; v1 = results/cmp-p9-1e6-profiles.json, the others as in 2.1:
python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- python3 tools/bench/compare.py \
  syn-mem27-100 yee-pec-box-1e6 --tag p9-screen-v$v --cores 16-31 --solvers slepc --slepc-profile best \
  --slepc-profile-file prof_v$v.json --reps 1 --root $D --mem-gb 32 --timeout 1500 --out-dir $S/cmp
# head-to-head (slot), one call per case, then the merge and the phases (lane):
python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- python3 tools/bench/compare.py \
  $CASE --tag cmp-p9-1e6-$CASE --cores 16-31 --solvers hsolve,hsolve-auto,hsolve-nocert,slepc \
  --slepc-profile best --slepc-profile-file results/cmp-p9-1e6-profiles.json --reps 3 --root $D \
  --mem-gb 32 --timeout 1500 --out-dir $S/cmp
python3 tools/bench/compare_merge.py --tag cmp-p9-1e6 --out-dir results \
  $S/cmp/cmp-p9-1e6-syn-mem27-100.json@c005 $S/cmp/cmp-p9-1e6-yee-pec-box-1e6.json@c005
python3 tools/bench/compare_phases.py results/raw/cmp-p9-1e6-$CASE
# hsolve's order, once per case (slot): SCOTCH inside the sequential MUMPS at 16 threads
OMP_NUM_THREADS=16 build/mpi/seq_factor_bench --matrix $S/k/$CASE.K.bin --ordering 3 --threads 16 \
  --write-perm $S/k/$CASE.perm.bin --modes full --nrhs 16 --solve-reps 1 --expect-neg $NEG
# layouts, one slot call per (case, repetition r); ORDER/TAG/LAYOUTS/EXTRA per section:
#   3.2  perm:$S/k/$CASE.perm.bin  perm      seq1x16,seq1x16h,1x16,2x8,4x4,8x2,16x1  r = 0..2
#   3.3  ord:5                     metis     seq1x16,1x16,2x8,4x4,8x2,16x1           r = 0..2
#   3.4  par:2                     parmetis  2x8,4x4,8x2,16x1  (--modes full)        r = 0
#   3.5  perm:...                  root0     seq1x16,2x8,4x4,8x2,16x1 (--icntl13 0)  r = 0..1
#        perm:...                  dist3     seq1x16,2x8,4x4 (--dist 3)              r = 0..1
#        ord:7                     auto7     seq1x16,1x16                            r = 0
#   3.6  perm:...                  balance   seq1x16,2x8,4x4,8x2,16x1 (--modes full --solve-reps 1)
HS_REP0=$r python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- \
  bash tools/bench/mpi_factor_runs.sh $S/m $CASE-$TAG $S/k/$CASE.K.bin $ORDER 1 $LAYOUTS \
  --nrhs 16 --solve-reps 3 --expect-neg $NEG $EXTRA
python3 tools/bench/mpi_factor_table.py $S/m/$CASE-$TAG.jsonl         # the tables (lane)
# 3.6, two concurrent counts (inside one slot guard call; per case, 2 repetitions):
A=(--matrix $S/k/$CASE.K.bin --perm $S/k/$CASE.perm.bin --threads 8 --modes count --expect-neg $NEG)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 16-23 build/mpi/seq_factor_bench "${A[@]}"  # one8
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 16-23 build/mpi/seq_factor_bench "${A[@]}" &  # two8
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 24-31 build/mpi/seq_factor_bench "${A[@]}"; wait
# 4, MPI start-up (lane): hello.c = MPI_Init + MPI_Finalize, built with the ref env's mpicc
time $HS_REF_ENV/bin/mpiexec -n $R --bind-to none ./hello                # R = 1, 2, 4, 8
```

NEG = 20003 (syn-mem27-100) and 337613 (yee-pec-box-1e6), from `$S/k/$CASE.K.bin.json`.
