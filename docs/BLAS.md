# Dense BLAS under MUMPS: profile, drop-in alternatives, BLAS hook (P5 "asmkern")

MUMPS spends its numeric factorization in dense BLAS-3 on frontal matrices and its solves in
narrow BLAS-2/3 calls. This page records where that time goes in hsolve, what the existing
open-source kernels (OpenBLAS built from source, BLIS, LIBXSMM) change, and the BLAS hook
(`src/blas`) that routes selected shape classes to other kernels. Every number has its command in
section 8; all runs on gpu02 (2x Xeon Silver 4314, CentOS 7), cores 16-23 (NUMA node 1) only,
through `tools/run/guard.py`, release build, 2026-10-12.

RESULTS_SUMMARY

## 1. Machine reference (cores 16-23)

`bench_asmkern_kernels peak --n 4096` (dgemm of the environment's OpenBLAS 0.3.34, best of two
sessions) and `bench_asmkern_kernels stream --mb 800` (OpenMP, first touch by the same threads,
best of 10):

| threads | dgemm GFlop/s (SkylakeX kernels) | per thread | dgemm with Haswell (AVX2) kernels | read GB/s | triad GB/s | update y += a x GB/s |
|---|---|---|---|---|---|---|
| 1 | 65.3 / 71.1 | 65-71 | 40.1 / 40.3 | 5.3 (*) | 8.5 | 12.3 |
| 4 | 237.4 / 252.3 | 59-63 | 152.7 / 162.3 | 17.2 (*) | 19.0 | 25.1 |
| 8 | 345.5 / 377.7 | 43-47 | 266.7 / 276.4 | 24.7 (*) | 25.4 | 27.7 |

(*) the first read kernel was not vectorized (latency-bound sum; fixed afterwards with
`omp parallel for simd`), so "read" understates; triad and update are the bandwidth references.

* The Silver 4314 executes two AVX-512 FMAs per cycle: one thread reaches 65-71 GFlop/s, i.e. 32
  flops/cycle at ~2.2 GHz; forcing OpenBLAS' Haswell (AVX2) kernels (`OPENBLAS_CORETYPE=Haswell`)
  gives 40 GFlop/s. (P5 planning assumed one FMA unit; the measurement says two.) The AVX-512
  kernels that OpenBLAS' DYNAMIC_ARCH picks on this CPU (SkylakeX) are the right ones.
* Per-thread dgemm rate falls from ~68 (1 thread) to ~61 (4) and ~45 (8 threads); the memory
  bandwidth available to these 8 cores is only ~25-28 GB/s (other jobs share the socket), about
  12 GB/s for one core.

## 2. Profile: where factorization and solves spend their time

Method: `libhsolve_blashook.so` preloaded with routing off (`HS_BLAS_ROUTE` unset) and
`HS_BLAS_PROFILE=<file>`; the hook times every dgemm/dtrsm/dgemv and the level-1 calls MUMPS makes,
keyed by routine x operand flags x size classes (powers of two) x MUMPS phase (from the `dmumps_c`
JOB and ICNTL(31): analysis, factorization, inertia count, solve with 1 / p right-hand sides) x
inside / outside OpenMP parallel regions. Outside parallel regions (MUMPS' fronts above its L0
layer, and hsolve's own calls) the time is wall time; inside (the L0 subtrees) it is summed over
threads, "in/T" below divides it by the thread count. Profiling cost: none measurable (route-off
hook and profiled runs within +-3% of plain runs, which vary as much between repetitions).
Default solver configuration of each case (ordering by the adaptive rule, block size by
`block_size_for`: p = 4 for the 3-D cases, GHIEP p = 2), one run each:

| case | thr | hsolve total s | factor + certificate s | iterate s | MUMPS fact. + counts wall s | BLAS in it: outside + inside/T s (share) | dgemm GFlop/s per thread inside / outside | MUMPS solves wall s | BLAS in them (share) | small calls (m < 128) inside, thread-s | hsolve's own BLAS: outside + inside/T s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 1 | 12.13 | 10.29 | 1.69 | 7.30 | 4.63 (63%) | - / 54 | 1.38 | 1.20 (87%) | - | 0.13 |
| syn-mem27-46 | 4 | 4.05 | 3.33 | 0.63 | 2.37 | 0.76 + 0.49 (53%) | 47 / 47 | 0.50 | 0.22 + 0.21 (86%) | 0.37 | 0.07 |
| syn-mem27-46 | 8 | 3.16 | 2.52 | 0.57 | 1.68 | 0.49 + 0.28 (46%) | 36 / 39 | 0.48 | 0.23 + 0.18 (86%) | 0.63 | 0.08 |
| yee-pec-box-1e5 | 1 | 13.21 | 9.74 | 3.36 | 7.15 | 4.57 (64%) | - / 53 | 2.73 | 2.32 (85%) | - | 0.33 |
| yee-pec-box-1e5 | 4 | 4.73 | 3.34 | 1.32 | 2.35 | 0.75 + 0.53 (55%) | 46 / 45 | 1.11 | 0.43 + 0.48 (82%) | 0.85 | 0.21 |
| yee-pec-box-1e5 | 8 | 3.66 | 2.47 | 1.14 | 1.68 | 0.56 + 0.22 (46%) | 36 / 38 | 0.95 | 0.50 + 0.29 (83%) | 1.20 | 0.20 |
| ss-qa8fk-qa8fm | 1 | 8.13 | 5.87 | 2.11 | 4.04 | 2.31 (57%) | - / 50 | 1.68 | 1.41 (84%) | - | 0.24 |
| ss-qa8fk-qa8fm | 4 | 3.02 | 2.02 | 0.92 | 1.28 | 0.34 + 0.26 (47%) | 46 / 42 | 0.68 | 0.28 + 0.29 (83%) | 0.57 | 0.15 |
| ss-qa8fk-qa8fm | 8 | 2.37 | 1.52 | 0.80 | 0.92 | 0.22 + 0.15 (40%) | 38 / 33 | 0.59 | 0.29 + 0.21 (84%) | 0.79 | 0.14 |
| wg-square-1e5 | 1 | 1.74 | 0.45 | 1.09 | 0.40 | 0.08 (20%) | - / 41 | 0.94 | 0.56 (59%) | - | 0.22 |
| wg-square-1e5 | 4 | 0.93 | 0.28 | 0.54 | 0.20 | 0.03 + 0.01 (18%) | 31 / 22 | 0.44 | 0.10 + 0.17 (62%) | 0.66 | 0.04 + 0.05 |
| wg-square-1e5 | 8 | 0.69 | 0.22 | 0.39 | 0.14 | 0.02 + 0.00 (19%) | 33 / 13 | 0.30 | 0.07 + 0.11 (61%) | 0.84 | 0.04 + 0.04 |
| wg-square-1e6 | 1 | 32.73 | 16.66 | 13.69 | 4.09 | 1.00 (24%) | - / 45 | 9.66 | 5.88 (61%) | - | 2.56 |
| wg-square-1e6 | 4 | 15.32 | 6.92 | 6.94 | 1.91 | 0.36 + 0.04 (21%) | 28 / 28 | 4.85 | 1.29 + 1.63 (60%) | 6.53 | 0.54 + 0.66 |
| wg-square-1e6 | 8 | 8.08 | 2.12 | 4.76 | 1.57 | 0.32 + 0.11 (28%) | 41 / 35 | 3.37 | 0.85 + 1.29 (63%) | 9.35 | 0.49 + 0.65 |

(wg-square-1e6: the adaptive ordering picked SCOTCH at 1 and 4 threads - analysis 4.3 s of the
6.9 s "factor" - and AMD at 8; GHIEP has no certificate. Per-shape tables: `bench_asmkern_report`
on the profiles, section 8.)

Findings:

* **Factorization (incl. the certificate's inertia counts) is 72-82% of the 3-D runs** at 4
  threads and BLAS is about half of it (46-64%; the rest is MUMPS' own assembly, pivoting and
  scheduling, which no BLAS can speed up). Essentially all of that BLAS time is dgemm on medium
  fronts (m 128-1k, n 128-1k, k 32-511; NN), inside the L0 subtrees (one thread per call) and
  above them (OpenBLAS threading the call). It runs at **45-54 GFlop/s per thread at 1-4
  threads, 70-85% of OpenBLAS' own large-dgemm rate at the same thread count** (65-71 / 59-63), and
  36-41 at 8 threads (large dgemm: 43-47). dtrsm, the level-1 calls (idamax, dswap, dscal, ...)
  and dgemv are below 1% of the factorization. So a better dgemm could shave at most ~15-25% of
  the BLAS part, i.e. ~6-13% of a factorization.
* **Solves: BLAS is 60-87% of MUMPS' solve time**, all of it narrow (N = p = 4 for the 3-D cases,
  2 for GHIEP): dgemm NN/TN with N = 4 and dtrsm with 32-63-sized triangles. Inside the L0 subtrees
  each thread streams the factors at 6-8 GB/s (4 threads: ~25-30 GB/s together, i.e. the STREAM
  limit of these cores); above the L0 layer the calls run on one core (OpenBLAS threads neither
  narrow dgemm below m*n*k = 262144 nor dtrsm with N < 8) at 12-19 GB/s.
* **2-D / GHIEP solves are made of millions of tiny calls**: wg-square-1e6 makes 3 million dtrsm
  and as many dgemm calls per run with 8-31-sized triangles and 2 right-hand sides; OpenBLAS' dtrsm
  costs 1-1.5 us per call inside the parallel region (0.3 us for one caller with warm caches: the
  extra is the concurrent callers and cold operands), dgemm 0.6-1.5 us. These are 6.5-9.4
  thread-seconds per run (35-40% of the solve phase).
* **hsolve's own BLAS calls** (block orthogonalization of the Krylov basis, Gram products V^T W
  with k = n) are 2-8% of the run; the long-K products A^T B with few rows run on one core
  (OpenBLAS never splits K): e.g. 48 x 4 x 1e5 at 9.5 GB/s, 2 x 2 x 1e6 at 3.6 GB/s.
* Analysis (ordering, symbolic) makes no BLAS calls.
* The level-1 and dgemv calls MUMPS makes are negligible everywhere (< 0.01 s).

## 3. Drop-in alternatives

DROPIN

## 4. The BLAS hook (`src/blas`)

HOOK

## 5. End to end

E2E

## 6. Recommendation and portable build story

RECOMMEND

## 7. Provenance and licenses

| component | version / source | license | used as |
|---|---|---|---|
| OpenBLAS | 0.3.34, release tarball (`tools/deps/build_openblas.sh`, sha256 `cd7e1298...`) | BSD-3-Clause | drop-in builds; the hook's AVX-512 narrow kernels adapt the structure of its `kernel/x86_64/dgemv_n_microk_skylakex-4.c` (NN) and `dgemm_small_kernel_tn_skylakex.c` (TN) - Copyright (c) 2021 The OpenBLAS Project |
| BLIS | 2.1, tag tarball (`tools/deps/build_blis.sh`, sha256 `901752ec...`) | BSD-3-Clause | drop-in experiment only |
| LIBXSMM | 1.17, tag tarball (`tools/deps/build_libxsmm.sh`, sha256 `8b642127...`) | BSD-3-Clause | drop-in experiment only (`libxsmmext` call wrapper) |
| small triangular solve (`small_trsm_left`) | the project's own, loop orders of the reference BLAS dtrsm | project | hook route `strsm` |

Checksums were taken from the downloads and cross-checked against Spack's package recipes
(OpenBLAS, LIBXSMM) and conda-forge's recipe (BLIS, OpenBLAS). No existing small-trsm kernel
fits MUMPS' calls: LIBXSMM's trsm generator is for packed batches of matrices, BLIS' small-trsm
path exists only in AMD's fork (AOCL-BLIS).

## 8. Reproduction

REPRO
