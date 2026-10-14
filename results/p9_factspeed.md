# P9 "factspeed": the numeric LDL^T at 1e6 on 16 threads

**Question.** In core at n = 1e6 (32 GiB slot, 16 threads) the 3-D runs are three numeric LDL^T
factorizations on one pivot order: the main one at sigma and the exact certificate's two
inertia-only counts nu(a), nu(b) (syn-mem27-100: 109 s = factor 35.2 + iterate 17.7 + certificate
55.3; yee-pec-box-1e6: 92 s = 29.8 + 14.4 + 47.3; results/p7_membudget.md §4). The certificate stays
the exact double-precision Sylvester count (user decision), so the lever is the factorization
itself. Where does a MUMPS 5.8.2 factorization of these matrices spend its time, and what makes it
faster - runtime settings, MUMPS controls, a MUMPS built from source for the CPU, patches of its hot
loops or BLAS routes?

**Answers.**

* **Profile (§2).** One factorization takes 22-26 s at 292-316 GFlop/s, **40-43% of the slot's
  DGEMM peak (728-744 GFlop/s)**. Two phases: MUMPS' L0 subtrees (tree parallelism, one thread per
  subtree) take 6.3-7.5 s (27-29%) for 13-14% of the flops at 131-152 GFlop/s, the fronts above them
  (node parallelism, multithreaded BLAS) 16-18 s at 360-376 GFlop/s. BLAS is 65-69% of the
  factorization wall time; the CPU time is 55% dgemm kernel, 9% OpenBLAS packing, 10% the LDL^T
  panels' rank-1 updates (MUMPS Fortran), 6% assembly, 5% zeroing new fronts (memset), 4% stack copies,
  10% OpenMP threads spinning idle. Above the L0 layer the updates of a front's remaining fully
  summed columns are rank-128 dgemm calls (MUMPS' full-rank panel width KEEP(420) = 128) that run at
  **350 GFlop/s against 585-620 for the rank >= 512 contribution-block updates** - 4.9-5.4 s of the
  12.4-13.9 s of dgemm above L0. The factorization is memory-bandwidth bound: 4 -> 16 threads gives
  1.84x (L0 2.7x, above it 1.95x), and in the L0 phase each thread falls from 34 GFlop/s (4 threads)
  to 9 (16).
* **What helps (§3.2): MUMPS' block sizes.** The full-rank panel width KEEP(420) = 256 and the
  contribution-block row blocks KEEP(58) = 2000 / KEEP(57) = 4000 - internal parameters that
  dmumps_c.h exposes and whose meaning the source makes clear (the blocking of the same
  threshold-pivoted LDL^T, §3.2) - make every factorization **1.050-1.056x faster** (main and count,
  both cases, the saved SCOTCH orders, 3 interleaved repetitions, identical inertia in every run;
  1.036-1.058x on METIS orders, §4.2). 192 / 320 / 384 / 512 are slower than 256.
* **Measured negative (§3):** OMP_PROC_BIND / OMP_PLACES (1.00x), OMP_WAIT_POLICY active (1.00x) or
  passive (0.93x), transparent huge pages off (0.96x); a higher, lower or no L0 layer (KEEP(408):
  0.91-1.00x; ICNTL(48) = 0: 0.95x); more amalgamation (KEEP(1): 0.99x); bigger triangle stripes
  (KEEP(7/8): 0.99x); **MUMPS 5.8.2 rebuilt from the official source** with -O3 -march=x86-64-v3 or
  skylake-avx512 (1.000-1.005x alone, +0.4% on top of the new block sizes), so the reproducible build
  path (tools/dev/build_mumps.sh, CMake option HSOLVE_MUMPS_ROOT) stays opt-in; GEMMT (OpenBLAS'
  dgemmt is a loop of dgemv calls).
* **A MUMPS patch (§3.4):** `patches/mumps-5.8.2-l0-blocking.patch` lets the small fronts under the
  L0 layer keep MUMPS' panel width and a narrower zeroed band (two KEEP entries MUMPS does not use,
  read only when set): another 1.8-2.1% per factorization with the conda-forge flags, 1.073-1.079x
  over MUMPS' defaults for a count.
* **Gate (§4): not passed, so opt-in.** End to end (5 interleaved runs, every run valid, certificate
  exact): syn-mem27-100 1.066x, yee-pec-box-1e6 **1.032x** in the default configuration - where
  SCOTCH's run-to-run scatter (8.5e12-1.34e13 flops on syn-mem27-100) dwarfs the effect - and
  1.038x / 1.027x with the ordering fixed (METIS); no regression at 1e5 (41 cases, geometric mean
  1.003x). The block sizes are therefore opt-in (`HS_MUMPS_BLOCKING=p9`), MUMPS' own stay the default.
  **With the patched MUMPS as well the bar is met: 1.125x / 1.066x end to end** (§4.4; ~1.05x on
  yee-pec-box-1e6 normalised for its runs' luckier SCOTCH orders) - but building MUMPS from source is
  not the default build; a build with `-DHSOLVE_MUMPS_ROOT=<that install>` takes the P9 block sizes
  by default.

## 1. Setup

* **Where.** gpu01 (2x Xeon Silver 4314, 2.4 GHz, AVX-512 with two FMA units per core, CentOS 7,
  kernel 3.10, glibc 2.17, THP "always"), the measurement slot: cores 16-31 = NUMA node 1, 32 GiB,
  one run at a time (`tools/run/guard.py --exclusive`); 4- and 8-thread runs on the slot's first
  cores while holding all of it. Builds and tests in the lane (cores 8-15). 2026-10-14.
* **Software.** The project environment: conda-forge MUMPS 5.8.2 (mumps-seq build 3: -O2
  -march=nocona -mtune=haswell, no GEMMT), OpenBLAS 0.3.34 (DYNAMIC_ARCH, OpenMP, SkylakeX kernels),
  LLVM OpenMP (conda-forge's libgomp.so.1 is libomp), SCOTCH 7.0.11, METIS 5.1.0, gcc/gfortran 13.4.
* **Cases.** syn-mem27-100 (HEP, 27-point 3-D, n = 1e6, nnz(A) = 2.65e7, sigma = 0, k = 12) and
  yee-pec-box-1e6 (GHEP, Yee edge elements, n = 1 027 733, sigma = 52.9, k = 10), in
  `$HOME/p7_p7-membudget/data` on the cluster; the certificate windows [a, b] of the P7 reports
  ([-5.37e-4, 5.98e-4] and [8.087, 96.19]).
* **Measurement program.** `bench/factspeed_factor.cpp` (`bench_factspeed_factor`) makes the
  driver's three factorizations exactly as the driver makes them: the main object with the driver's
  FactorOptions (SCOTCH, scaling 7, L0-OpenMP, ICNTL(14) = 35, the in-core cap of the 32 GiB plan
  ~25 GB) factorized at sigma and released (JOB = -4), then one inertia-only instance (ICNTL(31) = 1)
  analysed on the main SYM_PERM (PERM_IN) - InertiaCounter's instance - counting at a and b. SCOTCH is
  not reproducible from run to run (§4.1), so every A/B compares variants on **one saved pivot order**
  per case (`--perm-out` once, `--perm-in` after): the same tree, the same flops in every run. Per
  factorization it reports wall / user / system time, page faults, RINFOG(1-3), INFOG(9, 11-14, 18,
  21, 25, 28, 29, 48), the 2x2 pivots KEEP(229/230), MUMPS' own timers (`--mumps-stats 1`: time and
  flops under the L0 layer) and, with `--sample-us`, a SIGPROF sampler. Variants: extra MUMPS
  controls through `MumpsControlOverrides` (a measurement hook on MumpsFactorization, written after
  hsolve's own settings before each analysis and factorization), environment variables, and other
  MUMPS libraries through LD_PRELOAD of all seven (the sampler's mapped-object list confirms only the
  preloaded ones are mapped).
* **Screens.** `tools/bench/factspeed_screen.py`: per repetition every variant once, the order
  rotated and reversed every other repetition, 3 repetitions, medians [min-max] of the wall time of
  one factorization; every run's inertia (negative count, validity) must equal the base's (no
  variant ever changed a count). Unless said otherwise: count a (the certificate's lower count, an
  inertia-only factorization), 16 threads.
* **Peak.** OpenBLAS dgemm in the slot at 16 threads (`bench_asmkern_kernels peak`): 727.6 GFlop/s
  (n = 4096), 744.1 (n = 8192), 46.5 per thread; STREAM-like 36 GB/s read (docs/BLAS.md §1).

## 2. Profile (task 1)

### 2.1 The three factorizations

`bench_factspeed_factor --jobs main,a,b --mumps-stats 1` with the BLAS hook preloaded for its
profiler (route all, as in hsolve; it routes nothing in a factorization) and the sampler, one fresh
SCOTCH analysis per case (these are the pivot orders all later runs use):

| case | job | wall s | RINFOG(3) flops | RINFOG(2) (assembly) | INFOG(29) entries | largest front | delayed | 2x2 pivots | GFlop/s | % of peak | under L0: s (flops) | above L0: s, GFlop/s | sys s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | main | 25.75 | 7.556e12 | 2.58e9 | 1.232e9 | 15800 | 5 | 532 | 293 | 39% | 7.45 (9.73e11) | 18.30, 360 | 12.6 |
| syn-mem27-100 | a | 24.13 | 7.564e12 | 2.54e9 | 1.242e9 | 15800 | 0 | 523 | 314 | 42% | 6.54 (9.80e11) | 17.58, 374 | 6.2 |
| syn-mem27-100 | b | 24.15 | 7.564e12 | 2.54e9 | 1.242e9 | 15800 | 6 | 435 | 313 | 42% | 6.54 (9.80e11) | 17.60, 374 | 6.0 |
| yee-pec-box-1e6 | main | 23.48 | 6.865e12 | 2.34e9 | 1.149e9 | 14290 | 0 | 148 | 292 | 39% | 7.10 (9.41e11) | 16.38, 362 | 11.8 |
| yee-pec-box-1e6 | a | 22.36 | 6.877e12 | 2.50e9 | 1.166e9 | 14289 | 0 | 25 | 308 | 41% | 6.32 (9.54e11) | 16.04, 369 | 6.0 |
| yee-pec-box-1e6 | b | 22.54 | 6.877e12 | 2.50e9 | 1.166e9 | 14289 | 0 | 223 | 305 | 41% | 6.35 (9.54e11) | 16.19, 366 | 6.0 |

(% of peak against 744 GFlop/s. "a"/"b" are inertia-only: INFOG(9) = 0, the factors are discarded;
INFOG(29) counts the entries computed. No null pivots, no memory compressions (INFOG(14) = 0), no
tiny pivots, 0 workspace retries.) The counts cost 94-95% of a main factorization: discarding the
factors saves the factor compaction and half of the page faults (system time 6 vs 12 s of CPU), not
flops. Analysis: 6.3 s (syn, SCOTCH) / 4.5 s (yee); the counter's analysis on the given order 0.6 s.

### 2.2 BLAS against the rest (HS_BLAS_PROFILE)

| case | phase | MUMPS wall s | dgemm outside parallel regions: s (GFlop/s) | dgemm inside (L0 subtrees): thread-s (GFlop/s per thread) | BLAS share of wall |
|---|---|---|---|---|---|
| syn-mem27-100 | main | 25.73 | 13.87 (484) | 48.06 (22.3) | 66% |
| syn-mem27-100 | counts a + b | 48.25 | 27.64 (486) | 93.93 (23.0) | 69% |
| yee-pec-box-1e6 | main | 23.47 | 12.38 (489) | 45.92 (22.5) | 65% |
| yee-pec-box-1e6 | counts a + b | 44.89 | 24.60 (492) | 89.76 (23.3) | 67% |

(BLAS share = (outside + inside / 16) / wall. dtrsm, dgemv and level-1 calls are below 0.1 s.)
dgemm outside parallel regions (the fronts above the L0 layer, one call at a time, OpenBLAS threads
each call) by the rank k of the update, main factorization of syn-mem27-100 (yee alike):

| k | what (dfac_front_aux.F) | s | GFlop/s |
|---|---|---|---|
| 32-63 | updates inside a 128-column panel (inner blocks of KEEP(6) = 32) | 0.39 | 165 |
| 128-255 | the panel's update of the front's remaining fully summed columns (FAC_SQ_LDLT, KEEP(420) = 128) | 5.42 | 349 |
| 512-1023 | contribution-block updates (FAC_T_LDLT, k = pivots of the front) | 0.74 | 583 |
| 1024-2047 | " | 2.43 | 584 |
| 2048-4095 | " | 3.01 | 590 |
| 4096-8191 | " | 1.88 | 600 |

A rank-128 update C -= A B of a large front reads and writes C once per 2 x 128 flops per entry: 8
flops per byte of C traffic, i.e. at 36 GB/s it cannot pass ~290-580 GFlop/s depending on how much of
C stays in cache - it runs at 350. Inside the L0 subtrees each thread calls dgemm on its own small
and medium fronts at 22-23 GFlop/s per thread (the single-thread peak at 16 busy cores is ~46).

### 2.3 Where the CPU time goes (SIGPROF sampler)

`--sample-us 1000` (ITIMER_PROF; the kernel delivers ~1.6 samples per wall millisecond, spread over
the running threads: 35-42k samples per factorization), symbolised and grouped by
`tools/bench/factspeed_symbolize.py`; the saved orders, MUMPS' default block sizes; shares of the
samples, i.e. of the CPU time of all 16 threads (system time included: page faults count at the
faulting instruction); "L0" = the first 6.3 s of a factorization, "above" = from 7.6 s on:

| case | job | window | dgemm kernel | OpenBLAS packing / sync | OpenMP idle spin | LDL^T panel | assembly | zeroing fronts | stack / factor copies |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | main | all | 52.7 | 8.5 | 9.5 | 9.7 | 6.2 | 5.3 | 7.6 |
| syn-mem27-100 | a | all | 55.6 | 8.7 | 9.7 | 10.3 | 6.2 | 4.9 | 4.1 |
| syn-mem27-100 | a | L0 | 34.7 | 7.3 | 1.3 | 26.4 | 12.4 | 10.6 | 5.5 |
| syn-mem27-100 | a | above | 66.4 | 9.3 | 13.0 | 3.2 | 3.4 | 2.3 | 2.2 |
| yee-pec-box-1e6 | main | all | 51.9 | 8.6 | 9.4 | 10.2 | 6.6 | 5.1 | 7.7 |
| yee-pec-box-1e6 | a | all | 52.5 | 8.9 | 11.1 | 10.3 | 7.0 | 5.1 | 4.5 |
| yee-pec-box-1e6 | a | L0 | 33.9 | 7.5 | 1.1 | 26.7 | 12.5 | 10.0 | 6.7 |
| yee-pec-box-1e6 | a | above | 61.8 | 9.5 | 15.3 | 3.1 | 4.5 | 2.8 | 2.9 |

(Symbols: dgemm_kernel_SKYLAKEX; dgemm_oncopy / itcopy, inner_thread; __kmp_wait_template,
__kmp_hardware_timestamp; dmumps_fac_mq_ldlt's OpenMP loop body, dmumps_fac_i_ldlt;
dmumps_ldlt_asm_niv12, dmumps_fac_asm_niv1; __memset_sse2 of glibc 2.17; dmumps_copy_cb_left_to_right,
dmumps_dm_cbstatic2dynamic, and in the main factorization dmumps_compact_factors_sym. Counts b like a.)

In the L0 phase each thread works alone on its subtrees: a quarter of its time goes into the
BLAS-2-like panel updates (each pivot updates its inner block's remaining columns over all rows of
the front: MUMPS' PIVOT_OPTION 3, needed for the threshold test), a tenth into zeroing new fronts,
an eighth into assembly; dgemm only a third. Above it the dgemm kernel dominates; idle spinning
(threads waiting at the end of OpenBLAS' and MUMPS' parallel regions while others finish - the
master's own serial work is ~3%) is the next item, then OpenBLAS' packing. The timeline (1-s bins)
is flat within each phase; the spin share rises to 6% only in the last second of the L0 phase.

### 2.4 Thread scaling and the L0 layer

`bench_factspeed_factor --threads T` on the saved orders, T on the slot's first T cores (MUMPS'
L0 layer is chosen for T threads at the analysis):

| case | job | T | wall s | GFlop/s | under L0: s (GFlop/s) | above L0: s (GFlop/s) |
|---|---|---|---|---|---|---|
| syn-mem27-100 | main | 4 | 45.74 | 165 | 18.98 (129) | 26.76 (191) |
| syn-mem27-100 | main | 8 | 30.78 | 246 | 11.19 (156) | 19.60 (297) |
| syn-mem27-100 | main | 16 | 25.51 | 297 | 7.37 (133) | 18.14 (363) |
| syn-mem27-100 | a | 4 | 44.31 | 171 | 17.88 (137) | 26.43 (193) |
| syn-mem27-100 | a | 8 | 29.16 | 259 | 10.08 (174) | 19.08 (305) |
| syn-mem27-100 | a | 16 | 24.04 | 315 | 6.53 (150) | 17.51 (376) |
| yee-pec-box-1e6 | main | 4 | 42.74 | 161 | 17.07 (123) | 25.67 (186) |
| yee-pec-box-1e6 | main | 8 | 28.40 | 242 | 10.23 (156) | 18.16 (291) |
| yee-pec-box-1e6 | main | 16 | 23.55 | 292 | 7.06 (135) | 16.49 (359) |
| yee-pec-box-1e6 | a | 4 | 41.96 | 164 | 16.17 (130) | 25.79 (185) |
| yee-pec-box-1e6 | a | 8 | 27.33 | 252 | 9.26 (173) | 18.07 (292) |
| yee-pec-box-1e6 | a | 16 | 22.53 | 305 | 6.27 (152) | 16.26 (364) |

4 -> 16 threads: 1.84-1.86x (counts), 1.79-1.81x (main). The L0 layer at 16 threads holds less of
the tree (9.8e11 flops) than at 4 (2.4e12), and its rate stays at 130-175 GFlop/s whatever the
thread count: the L0 phase is bound by the socket's memory bandwidth (the per-thread rate falls from
34 to 9 GFlop/s). The fronts above it gain 1.95x from 4 to 16 threads. Without the L0 layer
(ICNTL(48) = 0, every front node-parallel) a count takes 0.95x the time (§3.2): the L0 layer is worth
5%.

## 3. Experiments (task 2)

All screens: count a, 16 threads, the saved pivot orders, 3 interleaved repetitions, wall time of
the factorization, medians [min-max]; the inertia of every run equals the base's.

### 3.1 Runtime settings (a)

| variant | syn-mem27-100 count a, s | vs base | under L0 s | sys s (CPU) |
|---|---|---|---|---|
| base (libomp defaults: KMP_BLOCKTIME 200 ms spin, no binding; THP always) | 24.05 [24.04-24.07] | 1.000 | 6.53 | 6.1 |
| OMP_PROC_BIND=close OMP_PLACES=cores | 24.00 [23.99-24.00] | 1.002 | 6.53 | 6.1 |
| OMP_WAIT_POLICY=active | 23.95 [23.95-23.98] | 1.004 | 6.52 | 6.2 |
| OMP_WAIT_POLICY=passive | 25.78 [25.72-25.82] | 0.933 | 6.52 | 9.8 |
| transparent huge pages off (prctl PR_SET_THP_DISABLE) | 24.94 [24.84-24.98] | 0.964 | 6.70 | 13.2 |

* Binding changes nothing: the guard already confines the process to the 16 cores of one NUMA node,
  16 threads on 16 cores do not migrate, and the memory is local (first touch by the slot's threads
  on node 1). Passive waiting puts idle threads to sleep between parallel regions (futex calls,
  wake-up latency): 7% slower.
* THP "always" (the node's setting) is right: without huge pages the system time doubles (4 KiB page
  faults on 5-15 GB of fresh workspace per factorization). glibc 2.17 has no malloc tunables for
  huge pages; with THP the remaining system time (6 s of CPU = ~0.4 s of wall per count) bounds what
  reusing the workspace across factorizations could save - not tried.

### 3.2 MUMPS controls (b)

MUMPS' LDL^T of a front (dfac_front_LDLT_type1.F, dfac_front_aux.F) eliminates the fully summed
columns in **outer panels of KEEP(420) columns** (NBLR_ORIG; MUMPS sets 4 * KEEP(6) = 128 at JOB=-1),
each made of inner blocks of KEEP(6) = 32 (KEEP(5) = 16 for fronts with at most KEEP(3) = 96 fully
summed columns) whose pivots are chosen one at a time with threshold pivoting (CNTL(1) = 0.01) and
applied to the block's remaining columns over all rows of the front (FAC_MQ_LDLT, BLAS-2 Fortran);
after each inner block a dgemm updates the rest of the outer panel (rank 32), after each outer panel
a dgemm updates the front's remaining fully summed columns (rank KEEP(420), in stripes of KEEP(8) =
120 rows once more than KEEP(7) = 150 remain), and at the end the contribution block is updated once
with all pivots of the front (FAC_T_LDLT: blocks of KEEP(58) = 1000 rows once the contribution block
has more than KEEP(57) = 2000, diagonal sub-blocks of KEEP(218) = 250). When a front is assembled,
its lower triangle plus a band of max(KEEP(7), KEEP(8), KEEP(218)) - 1 = 249 entries beyond the
diagonal is zeroed (dfac_asm_master_m.F): the region those dgemm calls read. The KEEP array is part
of the C structure (dmumps_c.h: `keep[500]`); these entries set block sizes only - every update
is the same arithmetic on the same entries, the pivot choices are made by the same test. The pivot
threshold CNTL(1) was not changed: 2x2 and delayed pivots are rare here (0-6 delayed, 25-532 2x2
pivots for 1e6 unknowns), a smaller threshold has nothing to gain.

**Panel width and contribution-block blocks** (main factorization and count a of both cases):

| variant | syn main s | vs base | syn count a s | vs base | yee main s | vs base | yee count a s | vs base |
|---|---|---|---|---|---|---|---|---|
| base (KEEP(420) = 128) | 25.35 [25.34-25.55] | 1.000 | 23.92 [23.86-24.10] | 1.000 | 23.44 [23.44-23.57] | 1.000 | 22.55 [22.46-22.58] | 1.000 |
| KEEP(420) = 192 | 24.43 [24.39-24.54] | 1.038 | 22.93 [22.93-22.98] | 1.043 | 22.66 [22.63-22.69] | 1.035 | 21.74 [21.68-21.75] | 1.037 |
| KEEP(420) = 256 | 24.13 [24.12-24.17] | 1.051 | 22.68 [22.67-22.73] | 1.055 | 22.47 [22.42-22.49] | 1.043 | 21.58 [21.49-21.60] | 1.045 |
| **KEEP(420) = 256, KEEP(58) = 2000, KEEP(57) = 4000** | 24.15 [24.00-24.16] | **1.050** | 22.65 [22.49-22.68] | **1.056** | 22.27 [22.24-22.41] | **1.053** | 21.35 [21.34-21.47] | **1.056** |
| KEEP(420) = 320 | 24.18 [24.17-24.25] | 1.048 | 22.73 [22.72-22.83] | 1.052 | 22.40 [22.37-22.43] | 1.047 | 21.56 [21.55-21.57] | 1.046 |
| KEEP(420) = 384 | 24.36 [24.29-24.63] | 1.041 | 22.87 [22.86-23.14] | 1.046 | 22.51 [22.46-22.58] | 1.041 | 21.65 [21.57-21.66] | 1.041 |

and in the first screen (count a of syn-mem27-100, base 24.09 s): KEEP(420) = 256 1.058x, = 512
1.028x; KEEP(6) = 64 with KEEP(420) = 256 1.042x (wider inner blocks: more BLAS-2 work per pivot);
KEEP(58) = 2000 alone 1.012x; KEEP(7/8) = 600/480 (4x fewer, taller triangle stripes) 0.991x.

* The rank of the trailing update is what limits the large fronts (§2.2): at 256 it is no longer
  memory-bound; wider panels lose again, because the rank-32 updates inside a panel then cover wider
  panels over all rows of the front, and because the L0 phase gets slower (6.51 -> 6.68 s at 256,
  6.89 s at 384: its small fronts prefer narrow panels). 256 is the best of 128-512 on all four
  factorizations; the larger contribution-block row blocks add 0.1-1.1% (fewer, larger dgemm calls
  on the biggest fronts).
* **Inertia** identical in every run of every variant (syn 20003 / 19997, yee 337613 / 337610).

**L0 layer, amalgamation, zeroed band** (count a):

| variant | syn s | vs ref | under L0 s | yee s | vs ref | under L0 s |
|---|---|---|---|---|---|---|
| ref: base | 23.96 [23.95-24.03] | 1.000 | 6.55 | | | |
| KEEP(408) = 97 (stricter L0 balance) | 24.03 [23.93-24.13] | 0.997 | 6.54 | | | |
| KEEP(408) = 70 (looser: more under L0) | 24.72 [24.71-24.75] | 0.969 | 8.59 | | | |
| KEEP(408) = 50 | 26.34 [26.20-26.39] | 0.910 | 12.30 | | | |
| ICNTL(48) = 0 (no L0 layer) | 25.16 [25.16-25.19] | 0.952 | - | | | |
| ref: KEEP(420) = 256 | 22.73 [22.71-22.83] | 1.000 | 6.69 | 21.60 [21.54-21.63] | 1.000 | 6.46 |
| + KEEP(1) = 8 (amalgamation; MUMPS 5) | 22.86 [22.83-22.88] | 0.994 | 6.82 | 21.69 [21.64-21.70] | 0.996 | 6.57 |
| + KEEP(1) = 16 | 23.08 [23.05-23.09] | 0.985 | 7.10 | 21.90 [21.87-21.96] | 0.986 | 6.81 |
| + KEEP(218) = 64 (zeroed band 149) | 22.70 [22.69-22.74] | 1.001 | 6.39 | 21.67 [21.65-21.71] | 0.997 | 6.19 |
| + KEEP(7/8/218) = 64 (zeroed band 63) | 22.80 [22.77-22.89] | 0.997 | 6.34 | 21.61 [21.58-21.63] | 0.999 | 6.13 |

* MUMPS chooses the L0 layer by descending from the root until the flop loads of the threads are
  within KEEP(408) = 90% (ana_omp_m.F); a looser balance puts more of the tree under the layer and
  loses, a stricter one changes nothing; the layer itself is worth 5%.
* More amalgamation (fewer, larger small fronts) costs more than it saves.
* A narrower zeroed band (smaller KEEP(7/8/218)) makes the **L0 phase 0.3 s faster** (less memset
  on its small fronts) but the large fronts above it 0.3 s slower (narrower stripes): no net gain
  from one global setting - §3.4 tries separate settings under and above the L0 layer.
* Not tried: ICNTL(14) (no memory compressions happen, INFOG(14) = 0), ICNTL(49) (compaction after
  the factorization: memory only), the pivot threshold (see above), PIVOT_OPTION KEEP(468) and the
  a-priori pivot estimates KEEP(269) (they change what the threshold test sees: never for counts,
  whose inertia must be that of the double LDL^T of K with MUMPS' pivoting).

### 3.3 MUMPS 5.8.2 rebuilt from the official source (c)

`tools/dev/build_mumps.sh NAME --isa ISA --opt LEVEL [--gemmt] [--patch FILE]`: the tarball
https://mumps-solver.org/MUMPS_5.8.2.tar.gz (sha256 eb515aa6...4db7039, the same as conda-forge's
recipe; downloaded on the development server, checked, copied to the node), built with the
environment's gfortran / gcc 13.4 against its METIS, SCOTCH and OpenBLAS: static archives with -fPIC,
then shared libraries with conda-forge's names and sonames (RPATH $ORIGIN first); gfortran's libmvec
pre-include is disabled and libm calls are wrapped so that nothing needs glibc > 2.17 (checked).
~90 s per build in the lane. Variants: nocona -O2 (conda-forge's flags: the control), x86-64-v3 -O3
(AVX2 + FMA, any x86-64 CPU since ~2015), skylake-avx512 -O3; swapped in by LD_PRELOAD of all seven:

| variant | syn count a s | vs base | yee count a s | vs base |
|---|---|---|---|---|
| base (conda-forge) | 23.93 [23.91-23.98] | 1.000 | 22.50 [22.43-22.51] | 1.000 |
| rebuilt, nocona -O2 | 23.96 [23.95-24.02] | 0.999 | 22.54 [22.51-22.56] | 0.999 |
| rebuilt, x86-64-v3 -O3 | 23.93 [23.89-24.02] | 1.000 | 22.48 [22.48-22.56] | 1.001 |
| rebuilt, skylake-avx512 -O3 | 23.92 [23.91-23.96] | 1.001 | 22.39 [22.37-22.46] | 1.005 |
| KEEP(420) = 256 | 22.78 [22.78-22.87] | 1.050 | 21.55 [21.50-21.70] | 1.044 |
| KEEP(420) = 256, rebuilt x86-64-v3 -O3 | 22.70 [22.69-22.74] | 1.054 | 21.48 [21.48-21.49] | 1.048 |
| KEEP(420) = 256, rebuilt skylake-avx512 -O3 | 22.66 [22.60-22.68] | 1.056 | 21.47 [21.43-21.53] | 1.048 |

* Vectorising MUMPS' own loops for the CPU gains nothing alone and 0.3-0.4% with the wider panels:
  the panel updates, assembly and copies of §2.3 are bound by memory (and, in the panels, by the
  latency of short rows in L2), not by instruction width. Not worth replacing the environment's
  library: the build path stays **opt-in**. `cmake -DHSOLVE_MUMPS_ROOT=$HOME/deps/mumps-5.8.2-NAME`
  links hsolve against such an install with its lib/ first in the RPATH (`test_factor_mumps_build`
  checks that the mapped MUMPS is the configured one; checked with the x86-64-v3 build).
* GEMMT (`--gemmt`, MUMPS' -DGEMMT_AVAILABLE, recommended by MUMPS' INSTALL file) was built but not
  timed: OpenBLAS 0.3.34 implements dgemmt (interface/gemmt.c) as one dgemv per column of the
  triangle - BLAS-2 for MUMPS' largest updates; it pays only with a BLAS whose GEMMT is blocked (MKL,
  NEC). A recursive GEMMT in the BLAS hook would only reshape calls whose shape is not the limit
  (taller stripes: 0.99x above).

### 3.4 A patch: separate block sizes under the L0 layer (d)

§2.3 and §3.2 show two kinds of fronts that want different blocking: the large fronts above the L0
layer want wide panels (KEEP(420) = 256), the small ones under it want MUMPS' 128 and a narrow
zeroed band (their L0 phase loses 0.17 s with 256-wide panels and gains 0.3 s with 64-wide stripes),
but MUMPS has one global setting. `patches/mumps-5.8.2-l0-blocking.patch` (20 hunks in 12 files: all four
arithmetics, applied by `tools/dev/build_mumps.sh --patch`) reads two KEEP entries that MUMPS 5.8.2
does not use, only while it factorizes the L0 subtrees (KEEP(405) = 1, set by dfac_b.F around
DMUMPS_FAC_L0_OMP): KEEP(498) > 0 replaces KEEP(420) as the panel width (dfac_front_LDLT_type1.F),
KEEP(499) > 0 replaces KEEP(7/8) for the triangle stripes (FAC_SQ_LDLT) and KEEP(218) for the
contribution block's diagonal sub-blocks (FAC_T_LDLT), and the zeroed band (dfac_asm_master_m.F)
shrinks to KEEP(499) - 1 accordingly, so it still covers every block a dgemm reads with beta = 1.
Unset (0, MUMPS' value after JOB = -1) they leave MUMPS' behaviour unchanged; hsolve sets KEEP(498) =
128 and KEEP(499) = 64 (`kL0PanelWidth`, `kL0Block`), which the environment's unpatched library
ignores. The same counts in every run (and `test_factor_panel`, `test_factor`, `test_factor_counter`
pass against the patched build).

On top of KEEP(420) = 256 / KEEP(58) = 2000 (count a, conda-forge's library = the reference):

| variant | syn s | vs ref | under L0 s | yee s | vs ref | under L0 s |
|---|---|---|---|---|---|---|
| ref: conda-forge MUMPS | 22.57 [22.45-22.64] | 1.000 | 6.87 | 21.45 [21.34-21.48] | 1.000 | 6.62 |
| patched, nocona -O2, KEEP(498) = 128 only | 22.51 [22.51-22.54] | 1.003 | 6.69 | 21.32 [21.23-21.34] | 1.006 | 6.46 |
| patched, nocona -O2, KEEP(499) = 64 only | 22.42 [22.35-22.44] | 1.007 | 6.60 | 21.32 [21.13-21.42] | 1.006 | 6.41 |
| **patched, nocona -O2, both** | 22.17 [22.14-22.20] | **1.018** | 6.44 | 21.01 [20.92-21.06] | **1.021** | 6.21 |
| patched, x86-64-v3 -O3, both | 22.19 [22.13-22.22] | 1.017 | 6.45 | 20.98 [20.86-21.04] | 1.023 | 6.20 |

Together with the block sizes of §3.2 a count is **1.073-1.079x** faster than with MUMPS' defaults
(syn 23.92 -> 22.17 s, yee 22.55 -> 21.01 s; the conda-forge flags suffice). Only the L0 phase changes
(6.87 -> 6.44 s, 6.62 -> 6.21 s).

## 4. Gate

The candidate for a default: the block sizes of §3.2 (KEEP(420) = 256, KEEP(58) = 2000, KEEP(57) =
4000) with the environment's MUMPS - `hsolve_k256c`, i.e. today's code with `HS_MUMPS_BLOCKING=p9` -
against main a831d79 (`hsolve_main`), `tools/bench/ab_cases.py` (ABBA, each run the case's default
configuration under its own nested guard, 16 threads, 32 GiB: in core), solver time
`time_total_s`, every run evaluated by tools/eval. The ab_cases tables of every gate run are in
`results/p9_factspeed_gates.json`, those of the screens of §3 in `results/p9_factspeed_screens.json`.

### 4.1 End to end at 1e6, default configuration (5 interleaved repetitions)

| case | main s, median [min-max] | P9 blocks s | speedup | valid | numeric s (main / P9, medians) | certificate s | SCOTCH flops of the runs |
|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 114.88 [107.51-124.75] | 107.80 [102.24-141.42] | 1.066 | 5/5, 5/5 | 30.4 / 28.2 | 58.9 / 54.0 | 8.5e12 - 1.34e13 |
| yee-pec-box-1e6 | 89.63 [87.46-91.71] | 86.81 [83.83-89.25] | 1.032 | 5/5, 5/5 | 23.8 / 22.3 | 45.6 / 44.2 | 6.7e12 - 7.3e12 |

Every run passes tools/eval (complete 12/12 and 10/10 against the discrete-exact references, max eta
1.7e-11 / 3.2e-12, certificate exact passed). **The ordering decides more than the factorization**:
SCOTCH, which the adaptive ordering picks for both cases, is not reproducible from run to run
(docs/DESIGN.md §3.4 P8) and at 1e6 its pivot orders cost between 8.5e12 and 1.34e13 flops on
syn-mem27-100 (102-141 s per run) and 6.7e12-7.3e12 on yee-pec-box-1e6, so five repetitions do not
resolve a few percent. Per flop the P9 blocks are 1.064x faster on yee-pec-box-1e6 (numeric
factorization 295 -> 314 GFlop/s over the ten runs).

### 4.2 End to end at 1e6 with the ordering fixed (METIS, 5 interleaved repetitions)

The same with each case's config plus `"ordering": 5` (METIS with a fixed seed: the same pivot
order in every run; data root `$S/data_metis`, links to the P7 data):

| case | main s | P9 blocks s | speedup | analysis s | numeric s (main / P9) | iterate s | certificate s (main / P9) |
|---|---|---|---|---|---|---|---|
| syn-mem27-100 | 98.05 [97.97-98.71] | 94.46 [94.08-95.33] | **1.038** | 14.1 | 23.4 / 22.2 (1.054) | 15.4-16.1 | 44.45 / 42.0 (1.058) |
| yee-pec-box-1e6 | 98.57 [98.48-98.63] | 95.99 [95.76-96.20] | **1.027** | 13.4 | 24.2 / 23.35 (1.036) | 14.3-14.4 | 46.2 / 44.4 (1.040) |

All 20 runs valid. The factorizations get their 4-6%, the rest of the run (METIS' serial analysis
14 s, the Krylov iteration, I/O) none: **1.03-1.04x end to end - below the 1.05x a new default
needs.**

### 4.3 No regression at 1e5

`ab_cases.py` ABBA, 3 repetitions, all 41 data cases but ss-3dspectralwave2 (HEP, GHEP, GHIEP, real
and complex, 289 to 998 001 unknowns, incl. wg-square-1e6): all 246 runs valid, **geometric mean
new/main 0.997 (1.003x)**, 5 cases faster by more than 3% (ss-ge87h76 0.952, ss-si34h36 0.956 - the
big-front cases -, syn-wide-range-real-1e4, wg-square-pert-1e4, wg-square-1e3), 3 slower by more than
3% with overlapping ranges. Those three, 5 more ABBA repetitions: syn-mem125-46 0.995, wg-halfslab-1e4
0.979, cav2d-square-1e4 1.038 (0.040 vs 0.041 s; min 0.038 both, ranges 0.038-0.055 / 0.038-0.050: noise
on a 40 ms run; its fronts are all narrower than either panel width).

### 4.4 With the patched MUMPS

`hsolve_l0p`: the P9 block sizes plus KEEP(498) = 128 / KEEP(499) = 64, linked against MUMPS built
from source with conda-forge's flags and `patches/mumps-5.8.2-l0-blocking.patch`
(`-DHSOLVE_MUMPS_ROOT=$HOME/deps/mumps-5.8.2-nocona-O2-l0p`), default configuration, 5 interleaved
repetitions against main:

| case | main s | patched + P9 s | speedup | valid | numeric GFlop/s (main / new, per run) | flops of the runs (main / new) |
|---|---|---|---|---|---|---|
| syn-mem27-100 | 112.02 [104.97-113.12] | 99.53 [95.99-107.24] | **1.125** | 5/5, 5/5 | 299-307 / 325-335 | 8.4-9.25e12 / 8.06-9.37e12 |
| yee-pec-box-1e6 | 89.70 [86.63-91.20] | 84.15 [81.41-85.46] | **1.066** | 5/5, 5/5 | 283-297 / 316-325 | 6.65-7.19e12 / 6.74-7.08e12 |

Every run passes tools/eval (complete, certificate exact passed). Per flop the main factorization
is 1.08-1.10x faster. The SCOTCH orders again differ between the runs: on yee-pec-box-1e6 the new
binary's runs drew 2% fewer flops in the median (6.80e12 vs 6.95e12); normalised to equal flops its
end-to-end gain is ~1.05x. **1e5 regression of this build** (the 41 cases, ABBA, 3 repetitions): all
246 runs valid, geometric mean new/main 0.998 (1.002x), 6 cases faster by more than 3% (ss-ge87h76
0.927, ss-si34h36 0.933, syn-multiplicity-real-1e4 0.945, wg-halfslab-1e5 0.954, cav2d-square-1e4
0.966, syn-wide-range-real-1e4 0.968), 5 slower by more than 3% - all 15-200 ms runs with overlapping
ranges, and with 5 more ABBA repetitions: syn-lap2d-100 1.025, syn-multiplicity-complex-1e4 1.003,
wg-square-1e3 0.978, wg-square-n289 0.918, wg-square-pert-1e4 0.898 (noise).

**Verdict.** The P9 block sizes with the environment's MUMPS pass the 1e5 regression check but not
the 1e6 end-to-end bar (1.032x on yee-pec-box-1e6 with SCOTCH, 1.027x with METIS): with the default
build MUMPS' own block sizes stay the default and `HS_MUMPS_BLOCKING=p9` (environment of the CLI or
of Python) opts in. With the patched MUMPS the whole gate is met (1.125x / 1.066x at 1e6, no
regression at 1e5), but that build is not the default build: it needs the MUMPS tarball (4.5 MB,
CeCILL-C: shippable in the repository) and a MUMPS build step (~90 s) before hsolve's in
tools/dev/build.sh and the deployment scripts - the lead's files, and a change to every build that
should be reviewed as such. **Recommendation: make the patched build the default build.** A build
configured with `-DHSOLVE_MUMPS_ROOT=<install of build_mumps.sh>` uses the P9 block sizes by default
(CMake `HSOLVE_MUMPS_BLOCKING`, overridable by HS_MUMPS_BLOCKING at run time), so today the whole
configuration is two commands: `env/run.sh tools/dev/build_mumps.sh l0p --isa nocona --opt O2 --patch
patches/mumps-5.8.2-l0-blocking.patch` and `cmake ... -DHSOLVE_MUMPS_ROOT=$HOME/deps/mumps-5.8.2-l0p`.

## 5. Conclusion

* **What the factorization is.** At n = 1e6 in core on 16 cores MUMPS 5.8.2 factorizes at 40-43% of
  DGEMM peak. It is bound by the socket's memory bandwidth, not by arithmetic: 4 -> 16 threads gives
  1.8x, the L0 subtrees (a quarter of the time, an eighth of the flops) stay at 130-175 GFlop/s at any
  thread count, and above them the rank-128 panel updates run at 350 GFlop/s. Compiler flags, thread
  binding, wait policies and huge pages cannot change that; the blocking can, a little.
* **Delivered (opt-in).** `HS_MUMPS_BLOCKING=p9`: MUMPS' full-rank panel width 256 and
  contribution-block row blocks 2000 make every factorization 1.04-1.06x faster (main and counts,
  both 1e6 cases), exact counts unchanged; end to end 1.03-1.04x (1.066x on syn-mem27-100 in the
  default configuration), no regression at 1e5. Below the 1.05x end-to-end bar, so not the default.
  A reproducible MUMPS source build (`tools/dev/build_mumps.sh`, CMake `HSOLVE_MUMPS_ROOT`, glibc 2.17
  safe, conda-forge sonames) and a reviewable patch (`patches/mumps-5.8.2-l0-blocking.patch`: L0-layer
  blocks KEEP(498/499), read only when set) add 1.8-2.1% per factorization (§3.4); together **1.125x
  and 1.066x end to end** (§4.4), past the bar - the recommendation for the lead is to make the
  patched build the default build (ship or fetch the tarball, build MUMPS before hsolve). Measurement
  tools: `bench/factspeed_factor.cpp` (the driver's factorizations, MUMPS statistics, a SIGPROF
  sampler), `tools/bench/factspeed_symbolize.py`, `tools/bench/factspeed_screen.py`, the
  `MumpsControlOverrides` / `instance()` hooks.
* **Negative results.** Runtime settings (binding 1.00x, active waiting 1.00x, passive 0.93x, THP off
  0.96x); MUMPS' L0 layer moved or off (0.91-1.00x); more amalgamation (0.99x); taller triangle
  stripes (0.99x); MUMPS rebuilt with -O3 for x86-64-v3 / AVX-512 (1.000-1.005x); GEMMT (OpenBLAS'
  dgemmt is BLAS-2). The pivot threshold and PIVOT_OPTION were left alone for the counts.
* **The bigger lever is not the factorization.** At 1e6 the adaptive ordering's SCOTCH orders scatter
  by 1.6x in flops between runs of the same case (syn-mem27-100: 8.5e12 - 1.34e13, 102-141 s per run)
  - more than every factorization setting measured here together; METIS with a fixed seed gives
  6.8e12 on syn-mem27-100 (below every SCOTCH order seen) but a 14 s serial analysis. That is the
  ordering module's (P9 "ordering") question; the numbers are in §4.1-4.2.
* **Open.** (1) Above the L0 layer 13-15% of the CPU time is threads waiting at the end of OpenBLAS'
  and MUMPS' parallel regions; within MUMPS' structure (one front at a time, BLAS-threaded) it is not
  reachable by settings. (2) The L0 phase: a quarter of its time is the panels' BLAS-2 rank-1
  updates over all rows of each front (MUMPS' PIVOT_OPTION 3, required by its threshold test on full
  columns), a tenth zeroing fronts; a left-looking panel with the CB rows updated by BLAS-3 would need
  a different pivoting scheme. (3) The patched build as the default needs the MUMPS tarball shipped
  or fetched by the deployment and a build step before hsolve's (tools/dev/build.sh, the deployment
  scripts: lead's files).

## 6. Commands

All on gpu01 through `tools/dev/cluster_run.sh gpu01 [--wip] -- ...` (HS_CLUSTER_NAME=p9-factspeed),
from the work tree of branch p9-factspeed. `S=$HOME/p9_p9-factspeed`, `D=$HOME/p7_p7-membudget/data/cases`,
`G="python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 --"` (measurement slot; each
campaign one $G call), `L="python3 tools/run/guard.py --skip-busy-check --mem-gb 8 --timeout 3600 --"`
(build lane). Timed binaries are copies in `$S/bin`: `bench_factspeed_factor` (built at 6b8cea2: no
block-size setting, i.e. MUMPS' defaults unless `--keep`), `libhsolve_blashook.so`,
`bench_asmkern_kernels`, `hsolve_main` (main a831d79, own clone `~/os2026-wt/p9-factspeed-main`,
`tools/dev/build.sh release`), `hsolve_k256c` / `hsolve_k256c2` (the P9 block sizes compiled in
unconditionally: the code of 602b24f with `HS_MUMPS_BLOCKING=p9`), `hsolve_l0p` (the same linked with
`-DHSOLVE_MUMPS_ROOT=$HOME/deps/mumps-5.8.2-nocona-O2-l0p`).

```bash
# MUMPS builds (lane; tarball in $HOME/deps/src, sha256 checked by the script)
$L tools/dev/build_mumps.sh nocona-O2 --isa nocona --opt O2
$L tools/dev/build_mumps.sh v3-O3 --isa x86-64-v3 --opt O3
$L tools/dev/build_mumps.sh skx-O3 --isa skylake-avx512 --opt O3
$L tools/dev/build_mumps.sh v3-O3-gemmt --isa x86-64-v3 --opt O3 --gemmt
$L tools/dev/build_mumps.sh nocona-O2-l0p --isa nocona --opt O2 --patch patches/mumps-5.8.2-l0-blocking.patch
$L tools/dev/build_mumps.sh v3-O3-l0p --isa x86-64-v3 --opt O3 --patch patches/mumps-5.8.2-l0-blocking.patch
cmake -S . -B build-l0p -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release \
  -DHSOLVE_MUMPS_ROOT=$HOME/deps/mumps-5.8.2-nocona-O2-l0p -DHSOLVE_BUILD_PYTHON=OFF -DHSOLVE_BUILD_BENCH=OFF

# 2.1-2.2 profile with fresh SCOTCH orders (saved for everything else); peak
B=$S/bin; P="env LD_PRELOAD=$B/libhsolve_blashook.so HS_BLAS_ROUTE=all"
SYN="--case $D/syn-mem27-100 --lower -0.0005369991406518608 --upper 0.0005978355069250973"
YEE="--case $D/yee-pec-box-1e6 --lower 8.087001904223854 --upper 96.1897089202103"
$G bash -c "$P HS_BLAS_PROFILE=$S/prof/syn100.blas.json $B/bench_factspeed_factor $SYN --perm-out $S/perm/syn100.perm \
    --mumps-stats 1 --sample-us 1000 --sample-out $S/prof/syn100.samples > $S/prof/syn100.out;
  (the same for yee1e6 with \$YEE); $B/bench_asmkern_kernels peak --n 4096 --reps 3; $B/bench_asmkern_kernels peak --n 8192 --reps 2"
build/release/bench_asmkern_report $S/prof/syn100.blas.json
# 2.3 profile on the saved orders, MUMPS' blocks and KEEP(420) = 256 (prof2.sh), symbolised in the lane
$G bash -c "$P HS_BLAS_PROFILE=$S/prof2/syn100-k128.blas.json $B/bench_factspeed_factor $SYN --perm-in $S/perm/syn100.perm \
    --jobs main,a,b --mumps-stats 1 --sample-us 1000 --sample-out $S/prof2/syn100-k128.samples > $S/prof2/syn100-k128.out; ..."
python3 tools/bench/factspeed_symbolize.py $S/prof2/syn100-k128.samples [--t1 6.3 | --t0 7.6] [--timeline 1] [--threads] --json OUT
# 2.4 thread scaling: per T in 4 8 16, inside one $G
python3 tools/run/guard.py --exclusive --cores 16-$((15+T)) --threads T --mem-gb 32 --timeout 900 -- \
  $B/bench_factspeed_factor $SYN --perm-in $S/perm/syn100.perm --jobs main,a --threads T --mumps-stats 1

# 3.x screens: one $G per spec; the spec lists the cases (with --perm-in), the job (a, or main,a for
# the panel widths), 3 repetitions, and the variants as extra arguments / environment:
$G python3 tools/bench/factspeed_screen.py run $S/specE.json $S/screenE
python3 tools/bench/factspeed_screen.py table $S/screenE --base base --md table.md --json table.json
#   specA: --keep 420=256 | 420=512 | 7=600,8=480 | 57=4000,58=2000 | 6=64,420=256
#   specB: LD_PRELOAD of the seven libraries of $HOME/deps/mumps-5.8.2-{nocona-O2,v3-O3,skx-O3}/lib, +- --keep 420=256
#   specC: OMP_PROC_BIND=close OMP_PLACES=cores | OMP_WAIT_POLICY=active | =passive | --thp-disable 1
#   specD: --icntl 48=0 | --keep 408=70 | 408=50 | 408=97
#   specE: --keep 420=192 | 256 | 320 | 384 | 420=256,57=4000,58=2000 (jobs main,a)
#   specF: --keep 420=256 plus 1=8 | 1=16 | 218=64 | 7=64,8=64,218=64
#   specG: --keep 420=256,57=4000,58=2000 plus LD_PRELOAD of nocona-O2-l0p / v3-O3-l0p with 498=128,499=64 | 498=128 | 499=64

# 4. gates (ab_cases: ABBA, each run its own nested guard in the slot, 32 GiB)
$G env HS_DATA_ROOT=$HOME/p7_p7-membudget/data python3 tools/bench/ab_cases.py run --bin main=$B/hsolve_main \
  --bin new=$B/hsolve_k256c --reps 0,1,2,3,4 --run-dir $S/gate1e6 --timeout 600 --deadline 2450 syn-mem27-100 yee-pec-box-1e6
#   the same with HS_DATA_ROOT=$S/data_metis (configs + "ordering": 5) -> $S/gate1e6m, and with hsolve_l0p -> $S/gate1e6l0p
$G python3 tools/bench/ab_cases.py run --bin main=$B/hsolve_main --bin new=$B/hsolve_k256c --reps 0,1,2 \
  --run-dir $S/gate1e5 --timeout 600 --deadline 2450 $(ls data/cases | grep -v '^ss-3dspectralwave2$')
#   the same with hsolve_l0p -> $S/gate1e5l0p; the cases slower by > 3% again with --reps 0,1,2,3,4
#   -> $S/gate1e5b (hsolve_k256c), $S/gate1e5l0pb (hsolve_l0p)
python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3600 -- python3 -m tools.eval.sweep $S/gate1e6 \
  --root $HOME/p7_p7-membudget/data --out $S/gate1e6/eval.md          # gate1e5: default root
python3 tools/bench/ab_cases.py table --base main --new new --md table.md --json table.json $S/gate1e5
```
