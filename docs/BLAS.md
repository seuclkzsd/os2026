# Dense BLAS under MUMPS: profile, drop-in alternatives, BLAS hook (P5 "asmkern")

MUMPS spends its numeric factorization in dense BLAS-3 on frontal matrices and its solves in
narrow BLAS-2/3 calls (complex problems go through the real symmetric embedding, so all of it is
real double precision). This page records where that time goes in hsolve, what existing
open-source kernels (OpenBLAS built from source, BLIS, LIBXSMM) change, and the BLAS hook
(`src/blas`) that routes selected shape classes to other kernels. Every number has its command in
section 8.

Where: gpu01 (2x Xeon Silver 4314, CentOS 7), the measurement slot - cores 16-31 = NUMA node 1,
16 physical cores, 32 GiB, one run at a time (`tools/run/guard.py --exclusive`), i.e. the judging
configuration; runs with fewer threads are pinned to the first cores of the slot while holding all
of it. Release build, conda-forge OpenBLAS 0.3.34 (DYNAMIC_ARCH, OpenMP, SkylakeX kernels),
MUMPS 5.8.2 (sequential), 2026-10-12/13. Binaries: the screening of sections 3 / 4.4 and the
route-off profiles of section 2.1 (except wg-square-1e6) were built from commit 6da4c7a, before
the review fixes of the hook (short OpenMP teams, `-ffp-contract=off`, the in-executable link),
which change no timed code path's work; everything else from e985acb. Section 2 also keeps an
earlier 1/4/8-thread profile from gpu02 cores 16-23 (taken before the slot existed).

**Summary**

* The cores have two AVX-512 FMA units (one core: 78 GFlop/s dgemm = 32 flop/cycle); 16 cores
  reach 747 GFlop/s. Memory bandwidth saturates from 4 cores on: 36 GB/s read, 26-34 GB/s triad
  for 4-16 cores, 13-17 GB/s for one core.
* Numeric factorization: BLAS is 33-54% of MUMPS' factorization wall time on the 3-D cases, almost
  all dgemm on medium fronts. Inside MUMPS' L0 subtrees dgemm runs at 47 GFlop/s per thread with 4
  threads and 24 with 16 (bandwidth- and frequency-bound); no drop-in BLAS and no route changes the
  factorization time by more than a few percent.
* Solves: BLAS is 83-87% of the 3-D solve time, narrow dgemm / dtrsm streaming the factors at an
  effective 16-22 GB/s. At 16 threads 61-64% of the solve time are calls made one at a time above
  MUMPS' L0 layer, where OpenBLAS uses a single core for narrow shapes - which is why solves stop
  scaling at 4-8 threads (section 2.2).
* Drop-in BLAS libraries do not help: OpenBLAS rebuilt for SkylakeX (incl. GEMM_GEMV_FORWARD, a
  multithreading threshold of 1) 0.97-1.07x end to end, LIBXSMM 0.97-1.02x, BLIS 0.46-0.57x on
  the 2-D cases (its small-call overhead).
* The BLAS hook (route `all`: parallel split of the calls above the L0 layer, split-K for long
  Gram products, AVX-512/AVX2 narrow kernels, a small triangular solve kernel, dgemm N = 1 ->
  dgemv) cuts the solve time per right-hand side by 15-28% and the total by 6-15% in screening
  (section 4.4). Confirmed with `tools/bench/compare.py` on the six head-to-head configurations,
  5 interleaved repetitions, every run valid at equal accuracy: **1.05-1.16x end to end** (section
  5); a one-column solve becomes 21-31% faster per right-hand side (section 2.2).
* Recommendation (section 6): the hook linked into hsolve with route `all` is the default now
  (`HSOLVE_BLAS_HOOK=ON`, `HSOLVE_BLAS_ROUTE_DEFAULT=all`); keep the environment's OpenBLAS;
  runtime CPU dispatch makes the same build right for an unknown judging CPU.

## 1. Machine reference (gpu01 slot)

`bench_asmkern_kernels peak --n 4096 --reps 3` (dgemm of the environment's OpenBLAS) and
`bench_asmkern_kernels stream --mb 1024 --reps 10` (OpenMP, static schedule, first touch by the same
threads; best of 10), pinned to cores 16..16+T-1, two sessions:

| threads | dgemm GFlop/s | per thread | read GB/s | triad GB/s | update y += a x GB/s |
|---|---|---|---|---|---|
| 1 | 77.7 / 77.5 | 77.6 | 14.0 / 14.1 | 12.6 / 12.7 | 17.2 / 17.4 |
| 4 | 301.0 / 300.6 | 75.2 | 36.5 / 36.4 | 26.2 / 26.3 | 32.1 / 32.1 |
| 8 | 550.7 / 552.9 | 69.0 | 37.1 / 37.2 | 29.7 / 29.7 | 32.2 / 32.1 |
| 16 | 747.3 / 742.0 | 46.6 | 36.1 / 36.2 | 33.9 / 34.0 | 31.9 / 32.2 |

* One core: 77.6 GFlop/s = 32 flop/cycle at ~2.4 GHz, i.e. two AVX-512 FMA units per core (the
  P5 plan assumed one). At 16 threads the per-thread rate falls to 47 (AVX-512 frequency licence,
  shared L3 and memory); the AVX-512 kernels that OpenBLAS' DYNAMIC_ARCH picks (SkylakeX) are the
  right ones (OPENBLAS_CORETYPE=Haswell, AVX2: 40 GFlop/s on one core, gpu02).
* Bandwidth: one core streams 13-17 GB/s; four cores already reach the socket's ~36 GB/s read
  bandwidth, 16 cores add nothing for reads and ~30% for triad. Anything memory-bound - every solve
  - gets at most ~2.6 times the bandwidth of one core, however many threads run.

## 2. Profile: where factorization and solves spend their time

Method: `libhsolve_blashook.so` preloaded into a plain hsolve with routing off
(`HS_BLAS_ROUTE=off`) and `HS_BLAS_PROFILE=<file>` (with the hook linked into hsolve, the default
build now, the same without the preload); the hook times every dgemm/dtrsm/dgemv and the level-1
calls MUMPS makes, keyed by routine x operand flags x size classes (powers of two) x MUMPS phase
(from the `dmumps_c` JOB and ICNTL(31): analysis, factorization, inertia count, solve with 1 / p
right-hand sides) x inside / outside OpenMP parallel regions. Outside parallel regions (MUMPS'
fronts above its L0 layer, and hsolve's own calls) the time is wall time; inside (the L0 subtrees)
it is summed over threads, "in/T" below divides it by the thread count. Profiling cost: none
measurable (route-off hook and profiled runs within +-3% of plain runs). Default solver
configuration of each case (adaptive ordering, block size by `block_size_for`: p = 4 for the 3-D
cases, GHIEP p = 2), one run each; `bench_asmkern_report` gives the per-shape tables.

### 2.1 Default block size, 4 / 8 / 16 threads (gpu01 slot)

Columns: hsolve total; MUMPS factorization + inertia-count wall; the BLAS part of it outside
parallel regions (wall) + inside / T, and their share; dgemm GFlop/s per thread inside parallel
regions / per process outside; MUMPS solve wall (right-hand sides, ms per right-hand side); the BLAS
part of it and its share; the share of the solve wall spent in BLAS calls outside parallel regions;
effective solve bandwidth (minimum BLAS traffic / solve wall); hsolve's own BLAS calls.

| case | thr | total s | fact+count wall s | BLAS out + in/T s (share) | dgemm GF/s in (per thr) / out | solve wall s (rhs, ms/rhs) | BLAS out + in/T s (share) | out / solve wall | solve GB/s | own BLAS out + in/T s |
|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | 4 | 3.99 | 2.273 | 0.748 + 0.472 (54%) | 47 / 192 | 0.495 (40, 12.39) | 0.215 + 0.207 (85%) | 43% | 17.9 | 0.073 + 0.000 |
| syn-mem27-46 | 8 | 2.97 | 1.548 | 0.479 + 0.244 (47%) | 39 / 331 | 0.453 (44, 10.29) | 0.222 + 0.162 (85%) | 49% | 21.7 | 0.074 + 0.000 |
| syn-mem27-46 | 16 | 2.82 | 1.389 | 0.469 + 0.106 (41%) | 24 / 424 | 0.465 (44, 10.58) | 0.292 + 0.112 (87%) | 63% | 21.2 | 0.070 + 0.000 |
| yee-pec-box-1e5 | 4 | 4.76 | 2.343 | 0.780 + 0.487 (54%) | 47 / 185 | 1.066 (84, 12.69) | 0.407 + 0.478 (83%) | 38% | 17.8 | 0.206 + 0.000 |
| yee-pec-box-1e5 | 8 | 3.53 | 1.599 | 0.495 + 0.260 (47%) | 38 / 314 | 0.878 (84, 10.45) | 0.425 + 0.303 (83%) | 48% | 21.5 | 0.189 + 0.000 |
| yee-pec-box-1e5 | 16 | 3.41 | 1.438 | 0.509 + 0.102 (42%) | 24 / 420 | 0.961 (84, 11.44) | 0.616 + 0.210 (86%) | 64% | 20.0 | 0.180 + 0.000 |
| ss-qa8fk-qa8fm | 4 | 3.10 | 1.329 | 0.328 + 0.249 (43%) | 46 / 166 | 0.693 (100, 6.93) | 0.285 + 0.292 (83%) | 41% | 16.4 | 0.155 + 0.000 |
| ss-qa8fk-qa8fm | 8 | 2.35 | 0.886 | 0.196 + 0.152 (39%) | 38 / 267 | 0.573 (100, 5.73) | 0.274 + 0.207 (84%) | 48% | 19.7 | 0.137 + 0.000 |
| ss-qa8fk-qa8fm | 16 | 2.37 | 0.909 | 0.226 + 0.076 (33%) | 24 / 329 | 0.637 (100, 6.37) | 0.389 + 0.155 (85%) | 61% | 18.1 | 0.132 + 0.000 |
| wg-square-1e5 | 4 | 0.98 | 0.210 | 0.032 + 0.006 (18%) | 28 / 88 | 0.443 (62, 7.14) | 0.107 + 0.171 (63%) | 24% | 8.4 | 0.041 + 0.046 |
| wg-square-1e5 | 8 | 0.72 | 0.145 | 0.022 + 0.005 (19%) | 30 / 102 | 0.292 (62, 4.71) | 0.072 + 0.108 (62%) | 25% | 12.7 | 0.035 + 0.037 |
| wg-square-1e5 | 16 | 0.65 | 0.131 | 0.022 + 0.002 (19%) | 28 / 109 | 0.252 (62, 4.07) | 0.079 + 0.083 (64%) | 31% | 14.7 | 0.035 + 0.030 |
| cav2d-rect-pert-1e5 | 4 | 0.68 | 0.269 | 0.021 + 0.002 (9%) | 11 / 38 | 0.275 (22, 12.50) | 0.067 + 0.073 (51%) | 24% | 4.4 | 0.003 + 0.000 |
| cav2d-rect-pert-1e5 | 8 | 0.48 | 0.191 | 0.019 + 0.001 (11%) | 14 / 39 | 0.186 (22, 8.44) | 0.040 + 0.052 (49%) | 22% | 6.6 | 0.002 + 0.000 |
| cav2d-rect-pert-1e5 | 16 | 0.39 | 0.148 | 0.015 + 0.001 (11%) | 19 / 36 | 0.133 (22, 6.06) | 0.015 + 0.051 (49%) | 11% | 9.2 | 0.003 + 0.000 |
| wg-square-1e6 | 4 | 14.50 | 1.926 | 0.353 + 0.034 (20%) | 26 / 111 | 4.894 (64, 76.47) | 1.300 + 1.623 (60%) | 27% | 7.6 | 0.542 + 0.663 |
| wg-square-1e6 | 8 | 8.19 | 1.575 | 0.326 + 0.113 (28%) | 42 / 284 | 3.369 (64, 52.64) | 0.848 + 1.282 (63%) | 25% | 15.3 | 0.486 + 0.651 |
| wg-square-1e6 | 16 | 7.26 | 1.302 | 0.213 + 0.098 (24%) | 30 / 398 | 2.827 (64, 44.17) | 0.662 + 1.065 (61%) | 23% | 18.2 | 0.469 + 0.628 |

(wg-square-1e6: the adaptive ordering chose SCOTCH at 4 threads - analysis 4.1 s - and AMD at 8
and 16 - analysis 0.5 s; GHIEP has no certificate.)

Findings:

* **Factorization** (incl. the certificate's inertia counts) is 38-57% of the 3-D runs, and BLAS
  is 33-54% of it; the rest is MUMPS' own assembly, pivoting and scheduling, which no BLAS changes.
  The BLAS time is dgemm on medium fronts (m 128-1k, n 128-1k, k 32-511; NN): inside the L0
  subtrees one thread per call at 47 GFlop/s per thread with 4 threads (63% of the large-dgemm
  rate), 24 with 16 threads (52%); above them OpenBLAS threads each call over the team (166-424
  GFlop/s per process, at most 57% of the 16-core peak - the fronts are too small to use 16 cores).
  dtrsm, dgemv and the level-1 calls are below 1%. The room for a faster dgemm is the 37-48% gap
  to the large-matrix rate within the BLAS part, i.e. below ~20% of a factorization; the drop-ins
  of section 3 recover none of it.
* **Solves: BLAS is 83-87% of MUMPS' solve time on the 3-D cases**, narrow dgemm NN/TN with N = p
  and dtrsm with 32-511-sized triangles, i.e. streaming the factors: 16-22 GB/s effective, below
  the 34-36 GB/s the cores can stream. The solve time hardly improves from 8 to 16 threads and even
  grows (yee-pec-box-1e5: 0.88 -> 0.96 s) because the share outside parallel regions grows from
  38-43% (4 threads) to 61-64% (16 threads): above MUMPS' L0 layer the fronts are processed one at
  a time, and OpenBLAS runs these narrow calls at 4-26 GB/s (section 4.3) - dgemm with
  m n k <= 262144 and every dtrsm with N < 8 on one core, larger narrow dgemm with its packing
  overhead.
* **2-D / GHIEP solves are millions of tiny calls**: wg-square-1e6 makes ~3 million dtrsm and as
  many dgemm calls per run with 8-31-sized triangles and p = 2; OpenBLAS' dtrsm costs 2.3 us per
  call of a 12-sized triangle inside the parallel region, the hook's small-trsm kernel 0.4 us
  (section 4.3).
* **hsolve's own BLAS calls** (block orthogonalization, Gram products V^T W with k = n) are 1-10%
  of a run; the long-K products A^T B with few rows ran on one core (OpenBLAS never splits K):
  48 x 4 x 1e5 at 11 GB/s.
* Analysis makes no BLAS calls; MUMPS' level-1 and dgemv calls are negligible (< 0.01 s).

### 2.2 One right-hand side at a time: why it barely scales beyond 4 threads

The two 3-D cases with `"block_size": 1` (every MUMPS solve has one right-hand side), route off and
route `all`, 4 / 8 / 16 threads, same columns as above:

| case | route | thr | total s | solve wall s (rhs, ms/rhs) | BLAS out + in/T s (share) | out / solve wall | solve GB/s |
|---|---|---|---|---|---|---|---|
| syn-mem27-46 | off | 4 | 4.89 | 1.422 (36, 39.51) | 0.662 + 0.597 (88%) | 47% | 21.6 |
| syn-mem27-46 | off | 8 | 3.84 | 1.360 (36, 37.78) | 0.736 + 0.474 (89%) | 54% | 22.5 |
| syn-mem27-46 | off | 16 | 3.82 | 1.528 (36, 42.45) | 1.028 + 0.352 (90%) | 67% | 20.0 |
| syn-mem27-46 | all | 4 | 4.44 | 1.064 (36, 29.57) | 0.419 + 0.491 (85%) | 39% | 28.1 |
| syn-mem27-46 | all | 8 | 3.77 | 1.080 (36, 30.00) | 0.491 + 0.446 (87%) | 45% | 29.8 |
| syn-mem27-46 | all | 16 | 3.32 | 1.078 (36, 29.95) | 0.595 + 0.334 (86%) | 55% | 28.4 |
| yee-pec-box-1e5 | off | 4 | 5.54 | 2.049 (50, 40.97) | 0.874 + 0.890 (86%) | 43% | 21.0 |
| yee-pec-box-1e5 | off | 8 | 4.57 | 2.105 (50, 42.11) | 1.276 + 0.594 (89%) | 61% | 19.9 |
| yee-pec-box-1e5 | off | 16 | 4.48 | 2.276 (50, 45.53) | 1.583 + 0.460 (90%) | 70% | 18.1 |
| yee-pec-box-1e5 | all | 4 | 5.01 | 1.563 (50, 31.26) | 0.549 + 0.725 (81%) | 35% | 26.8 |
| yee-pec-box-1e5 | all | 8 | 3.98 | 1.497 (50, 29.94) | 0.674 + 0.589 (84%) | 45% | 28.7 |
| yee-pec-box-1e5 | all | 16 | 3.84 | 1.578 (50, 31.56) | 0.890 + 0.444 (85%) | 56% | 27.0 |

Why the time per right-hand side does not fall beyond 4 threads (it even rises, 39.5 -> 42.5 ms
and 41.0 -> 45.5 ms from 4 to 16):

1. **It is a stream of the factors.** One right-hand side reads every factor entry twice (forward
   and backward substitution, ~0.9 GB for syn-mem27-46) and does 2 flops per entry read: 20-22 GB/s
   effective. The socket delivers ~36 GB/s of reads from 4 cores on (section 1), so the L0 part,
   which already runs on all threads, has little bandwidth left to gain.
2. **Above MUMPS' L0 layer it runs on one core.** 47% of the solve wall at 4 threads, 67-70% at
   16, is spent in BLAS calls outside parallel regions: MUMPS processes the upper fronts one at a
   time and calls dgemm with N = 1 and dtrsm with N = 1 on them, which OpenBLAS runs on one core
   (the profile: 15 264 such dgemm calls at 17 GB/s, one core's bandwidth). More threads only
   shrink the L0 part; the serial part stays (Amdahl), and at 16 threads the extra L0 threads
   compete with it for bandwidth.
3. MUMPS' own per-front work in the solve (gathers, scatters of the right-hand side: the 10-14%
   that is not BLAS) is serial above the L0 layer as well.

With route `all` the calls above the L0 layer use all threads (gemv1 + par: the dgemm N = 1 calls
become dgemv split by rows over the threads), the effective rate rises to 27-30 GB/s and the time
per right-hand side drops 21-31% (to 30-32 ms), flat from 4 to 16 threads - at the bandwidth limit,
which no kernel can pass. The rest of the gap to perfect scaling is MUMPS' sequential traversal of
the upper tree, which a BLAS cannot change.

### 2.3 Earlier profile, gpu02 cores 16-23, 1 / 4 / 8 threads (2026-10-12)

The same method before the measurement slot existed (8 cores of a socket shared with other jobs;
its 1-thread rows show the serial split):

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

(wg-square-1e6 there: the adaptive ordering picked SCOTCH at 1 and 4 threads - analysis 4.3 s of
the 6.9 s "factor" - and AMD at 8.)

## 3. Drop-in alternatives

Built from the pinned release tarballs with `tools/deps` (section 7) on the cluster nodes and
swapped in at run time without relinking: OpenBLAS variants through `LD_PRELOAD` of their
`libopenblas.so.0` (MUMPS and hsolve then bind every BLAS and LAPACK symbol to it), BLIS and
LIBXSMM's `libxsmmext` (dgemm_ call wrapper, JIT kernels up to M N K = 64^3, the rest to the next
BLAS) through `LD_PRELOAD` in front of the environment's OpenBLAS (LAPACK stays OpenBLAS').

| name | library, build |
|---|---|
| base | conda-forge OpenBLAS 0.3.34 (DYNAMIC_ARCH, OpenMP; SkylakeX kernels on this CPU) |
| skx | OpenBLAS 0.3.34, `DYNAMIC_ARCH=0 TARGET=SKYLAKEX` (conda-forge's other flags) |
| skxfwd | skx + `GEMM_GEMV_FORWARD=1` (dgemm with N = 1 or M = 1 -> dgemv) |
| skxfwdt1 | skxfwd + `GEMM_MULTITHREAD_THRESHOLD=1` (threads narrow dgemm from m n k > 65536) |
| blis | BLIS 2.1, `--enable-threading=openmp --enable-cblas x86_64` (runtime dispatch: skx kernels) |
| xsmm | LIBXSMM 1.17, `libxsmmext` with `WRAP=1` (dgemm_), no OpenMP |

End to end with `bench_asmkern_e2e` (hsolve CLI, 16 threads in the slot, 3 interleaved
repetitions, medians; the ordering method fixed per case - SCOTCH for the 3-D cases, whose
orderings still vary between runs by up to ~6% in factor entries, AMD for the 2-D ones, identical
in every run - instead of the adaptive choice, which depends on a measured analysis time; every
run succeeded with a passed certificate, eigenvalues within 1e-12 relative of the base run):
total-time speedup against base, and in parentheses MUMPS' solve time per right-hand side in ms.

| case | base | skx | skxfwd | skxfwdt1 | blis | xsmm |
|---|---|---|---|---|---|---|
| syn-mem27-46 | 2.672 s (10.59) | 0.994 (10.62) | 0.975 (10.97) | 1.022 (9.90) | 0.993 (11.82) | 0.972 (10.65) |
| yee-pec-box-1e5 | 3.187 s (10.88) | 1.054 (10.85) | 1.042 (10.62) | 1.054 (10.30) | 0.974 (13.05) | 1.012 (10.96) |
| ss-qa8fk-qa8fm | 2.226 s (6.43) | 1.043 (6.16) | 1.053 (6.31) | 1.074 (5.49) | 1.008 (7.14) | 1.002 (6.32) |
| wg-square-1e5 | 0.595 s (3.99) | 1.010 (3.95) | 1.008 (3.93) | 1.009 (4.00) | 0.461 (15.25) | 0.975 (4.02) |
| cav2d-rect-pert-1e5 | 0.340 s (5.91) | 1.004 (5.82) | 1.041 (5.14) | 1.050 (5.15) | 0.455 (24.03) | 0.994 (5.88) |
| wg-square-1e6 | 7.209 s (43.24) | 1.004 (43.19) | 1.020 (43.27) | 1.023 (41.49) | 0.571 (130.50) | 1.016 (43.65) |

* The numeric factorization changes by a few percent at most: MUMPS' numeric time with the
  OpenBLAS SkylakeX builds is 1-6% lower on yee-pec-box-1e5, ss-qa8fk-qa8fm and wg-square-1e6 and
  equal on the other three cases; BLIS and LIBXSMM are within +-3% of base. MUMPS' medium fronts
  run at about the same rate in every library's AVX-512 kernels.
* The OpenBLAS rebuilds gain -3% to +7% end to end, mostly in the solves (the multithreading
  threshold of 1 lets OpenBLAS thread more of the narrow dgemm above the L0 layer); for half the
  cases within the 3-run noise.
* BLIS is 2-3x slower on the 2-D solves: its per-call overhead on the tiny dtrsm/dgemm calls of
  the L0 subtrees (15-24 ms per right-hand side instead of 4-6). LIBXSMM's JIT kernels only take
  dgemm, and MUMPS' dgemm calls are either memory-bound (no gain from a better kernel) or above its
  64^3 threshold.

## 4. The BLAS hook (`src/blas`)

### 4.1 Routes

The hook replaces dgemm_, dtrsm_, dgemv_ (and, for the profiler, the level-1 routines MUMPS calls)
and sends each call by shape to an existing kernel; everything else - and every call while the
configuration is `off` - goes unchanged to the real BLAS, found after the hook in the lookup order.
Configuration: `HS_BLAS_ROUTE` (comma-separated tokens, read once per process; unknown tokens abort
with a message), else the build's `HSOLVE_BLAS_ROUTE_DEFAULT` (empty = off).

| token | shape class | goes to |
|---|---|---|
| `gemv1` | dgemm with N = 1 (or M = 1) | the real dgemv (A read once, never packed) |
| `strsm=<m>` | dtrsm side L, M <= m (capped: 4096 AVX-512, 256 AVX2, 32 portable), 1 <= N <= maxn, alpha = 1 | the hook's small triangular solve (reference loop order, up to 4 right-hand sides per pass over the triangle; no packing) |
| `trsv=<n>`, `trsv1` | dtrsm side L, N <= n, alpha = 1 | N calls of the real dtrsv |
| `narrow` | dgemm NN / TN with 2 <= N <= maxn | the hook's narrow kernels (AVX-512F or AVX2+FMA, adapted from OpenBLAS' SkylakeX small-matrix / dgemv kernels: A streamed once, not packed); without `par`, calls outside parallel regions with >= 2 x minkb of A stay with the real routine |
| `par` | outside OpenMP parallel regions, calls touching >= 2 x minkb of A | fixed blocks over `threads` threads: dgemv and narrow dgemm by rows of C (narrow kernels or the real routine per block), dgemm A^T B with M < 4 x threads and K >= 1024 x threads by K (partial products summed in block order) |
| `all` | | `gemv1,strsm=512,narrow,par` |
| `maxn=<n>`, `rows=<r>`, `minkb=<kb>`, `threads=<t>` | | narrow width limit (16), NN split block (64 rows), work per thread (256 KiB), threads of the parallel route (0 = omp_get_max_threads()) |
| `avx2`, `portable` | | cap the instruction set of the hook's kernels |

Instruction set: chosen at run time from CPUID - AVX-512F, else AVX2 + FMA, else the portable C++
loops (the narrow route needs AVX2 at least; on non-x86 CPUs only the portable loops are compiled).
`HS_BLAS_NO_AVX512=1` / `HS_BLAS_NO_AVX2=1` force the fallbacks (tests, comparisons).
`HS_BLAS_PROFILE=<file>` (`%p` = pid) turns on the call profiler of section 2 in any configuration.

Correctness and threads (tests: `test_asmkern`, `test_asmkern_fix`, `test_asmkern_fix_exehook`):
every routed shape class is compared with the real BLAS / a long-double reference (edge sizes,
transposes, leading dimensions, alpha / beta special values, beta = 0 never reading C, NaN
propagation), on every instruction set the CPU has and with the fallbacks forced. Inside OpenMP
parallel regions (MUMPS' L0 subtrees) every route runs serially in the calling thread - no
oversubscription with MUMPS' own threads; outside, the parallel route uses at most `threads`
threads. A route changes the rounding of a call, never its result beyond rounding; for a fixed
thread count every route is deterministic (fixed work split, each element computed by one thread
in a fixed order), and the narrow kernels without a K split give the same result for every thread
count (`hsolve_blas` is compiled with `-ffp-contract=off` so that both NN code paths round alike).
The fixed blocks of the parallel routes are dealt round-robin over the team the OpenMP runtime
actually forms, so a smaller team (OMP_THREAD_LIMIT, OMP_DYNAMIC, max-active-levels) still computes
every block. Thread safety: the configuration and the resolved routines are written once;
concurrent calls from any number of threads are tested.

### 4.2 Using it

Two ways, not both at once (the calls would pass through the hook twice):

1. **Preloaded** (any executable that does not carry the hook, nothing rebuilt):
   `tools/dev/build.sh release` builds `build/release/libhsolve_blashook.so`; for an hsolve built
   with `-DHSOLVE_BLAS_HOOK=OFF` in `build-plain`:

   ```bash
   LD_PRELOAD=$PWD/build/release/libhsolve_blashook.so HS_BLAS_ROUTE=all \
     build-plain/hsolve --A A.npz --B B.npz --config config.json --out out --threads 16
   ```

   **Python module**: the same preload in front of the interpreter. The extension module, MUMPS
   and OpenBLAS are opened with `dlopen(RTLD_LOCAL)` there; the preloaded hook still comes first
   in their lookup, and it finds the real routines in those libraries itself
   (`find_loaded_definition`):

   ```bash
   LD_PRELOAD=$PWD/build/release/libhsolve_blashook.so HS_BLAS_ROUTE=all \
     PYTHONPATH=build/release/python python3 -c "import hsolve; r = hsolve.solve('A.npz', config='config.json'); print(r.report['success'])"
   ```

   numpy / scipy BLAS calls of the same process also pass through the hook (forwarded or routed
   like MUMPS'; the routes are correct for any caller).
2. **Linked into hsolve** (the default build; explicitly `cmake ... -DHSOLVE_BLAS_HOOK=ON
   -DHSOLVE_BLAS_ROUTE_DEFAULT=all`): the executable carries the hook (CMake function
   `hsolve_link_blas_hook`: the MUMPS libraries stay in DT_NEEDED with `--no-as-needed` - with
   conda-forge's `-Wl,--as-needed` they used to be dropped and hsolve aborted with "no definition
   of dmumps_c" - and a dynamic list exports the entry points so that MUMPS' calls bind to them).
   `HS_BLAS_ROUTE` still overrides the built-in default at run time (`HS_BLAS_ROUTE=off` = plain
   forwarding). `test_asmkern_fix_exehook` links the same way in every build and checks that
   MUMPS' BLAS calls reach the hook; with the option ON it also runs the hooked hsolve.

### 4.3 Kernel measurements (16 threads, gpu01 slot)

`bench_asmkern_kernels shapes`: MUMPS solve shapes (op flags m x n x k; trsm m x n), per-call time
of the real routine and of the routes, interleaved, median of 7, operands cycling through a 1 GB
pool (cold, as in MUMPS' solves). Inside a parallel region every one of the 16 threads calls at the
same time (times per call and thread):

| shape | real us | narrow | strsm=512 |
|---|---|---|---|
| trsm LUTU 12 x 2 | 2.30 | - | 0.43 (5.3x) |
| trsm LUNU 12 x 2 | 2.33 | - | 0.47 (5.0x) |
| trsm LUTU 24 x 2 | 2.57 | - | 1.89 (1.36x) |
| trsm LUNU 24 x 4 | 2.62 | - | 1.93 (1.35x) |
| trsm LUTU 48 x 4 | 6.95 | - | 6.62 (1.05x) |
| trsm LUTU 24 x 1 | 2.47 | - | 1.88 (1.32x) |
| gemm TN 12 x 2 x 12 | 2.39 | 0.44 (5.4x) | - |
| gemm TN 24 x 2 x 24 | 2.79 | 1.97 (1.42x) | - |
| gemm TN 96 x 4 x 24, NN 24 x 4 x 96 | 7.8 | 1.00-1.01x | - |
| gemm TN 384 x 4 x 96, NN 96 x 4 x 384 | 125 | 1.00-1.01x | - |

Calls from one thread outside parallel regions (the fronts above MUMPS' L0 layer, hsolve's Gram
products):

| shape | real us (GB/s) | par | narrow+par | strsm=512 | gemv1+par |
|---|---|---|---|---|---|
| gemm NN 400 x 4 x 1500 | 186.8 (25.7) | 1.37x | 1.38x (35.6 GB/s) | - | - |
| gemm TN 1500 x 4 x 400 | 185.9 (25.8) | 1.43x | 1.42x | - | - |
| gemm NN 100 x 4 x 700 | 50.3 (11.1) | 1.50x | 1.98x (22.0 GB/s) | - | - |
| gemm TN 700 x 4 x 100 | 80.2 (7.0) | 1.74x | 3.16x (22.1 GB/s) | - | - |
| gemm TN 400 x 4 x 48 | 26.6 (5.8) | 1.00x | 2.24x | - | - |
| gemm TN 48 x 4 x 100000 (Gram) | 3475 (11.0) | 3.16x (split-K, 34.9 GB/s) | 3.31x | - | - |
| gemm NN 2000 x 1 x 2000 | 1003 (31.9) | 1.00x | - | - | 1.04x |
| trsm LUNU 48 x 4 | 2.86 (3.2) | - | - | 1.56x | - |
| trsm LUTU 200 x 4 | 40.0 (4.0) | - | - | 1.93x | - |
| trsm LUNU 400 x 4 | 133.2 (4.8) | - | - | 1.50x | - |

* Inside the L0 subtrees the 16 concurrent callers share the socket's bandwidth (2.4 GB/s per
  thread = 38 GB/s together): medium narrow products are memory-bound and no kernel can be faster;
  the gain is on tiny calls, where OpenBLAS' fixed cost of ~2 us per call (buffer allocation,
  packing) dominates - the small trsm and the narrow kernels take 0.4 us for 12-sized operands.
* Outside them the real routine runs on one core at 6-26 GB/s; the parallel split reaches the
  socket's 35-37 GB/s on large calls, and with the narrow kernels (no packing) 22 GB/s on medium
  ones. Results differ from the real routine by rounding only (relative difference <= 2.2e-15).
* AVX2 kernels (HS_BLAS_NO_AVX512=1): the same speedups within 1-2% except the narrow TN kernel on
  medium shapes (2.45x instead of 3.16x for 700 x 4 x 100) and the small trsm from m = 48 (1.23x
  instead of 1.56x; capped at 256). Portable loops: the small trsm still gains 5x on 12-sized
  triangles (capped at 32), the parallel split gains as much as with SIMD kernels (it calls the
  real routine per block); the narrow route is off.

### 4.4 Route variants end to end (screening)

Same runs as the drop-in table of section 3 (bench_asmkern_e2e, 16 threads, 3 interleaved
repetitions, fixed orderings, all runs ok): total speedup against base (solve ms per right-hand
side).

| case | base | all | gemv1,strsm=512,par (no narrow) | all, AVX2 forced |
|---|---|---|---|---|
| syn-mem27-46 | 2.672 s (10.59) | 1.058 (8.41) | 1.066 (8.93) | 1.051 (8.56) |
| yee-pec-box-1e5 | 3.187 s (10.88) | 1.148 (8.33) | 1.117 (8.92) | 1.139 (8.54) |
| ss-qa8fk-qa8fm | 2.226 s (6.43) | 1.132 (4.61) | 1.118 (4.79) | 1.139 (4.80) |
| wg-square-1e5 | 0.595 s (3.99) | 1.102 (3.25) | 1.093 (3.44) | 1.107 (3.31) |
| cav2d-rect-pert-1e5 | 0.340 s (5.91) | 1.100 (4.43) | 1.120 (4.42) | 1.114 (4.45) |
| wg-square-1e6 | 7.209 s (43.24) | 1.100 (36.58) | 1.093 (37.77) | 1.091 (37.12) |

* Every gain is in the solves (solve time per right-hand side -15% to -28%); MUMPS' numeric
  factorization time is unchanged.
* Most of it comes from the parallel split and the small trsm; the narrow kernels take another
  0-7% off the solve time per right-hand side, within the noise of the totals; AVX2 is within 1-2%
  of AVX-512 (the routes are memory-bound).
* Equal accuracy: largest backward error <= 3.1e-12, eigenvalues within 1e-12 relative of base,
  every certificate passed (as for the drop-ins).

Where the gain comes from - the profile of section 2.1 at 16 threads with route `all` (default
configuration, one run each):

| case | total s (off) | solve wall s (rhs, ms/rhs) | BLAS out + in/T s (share) | out / solve wall | solve GB/s | own BLAS out + in/T s |
|---|---|---|---|---|---|---|
| syn-mem27-46 | 2.61 (2.82) | 0.356 (44, 8.09) | 0.194 + 0.102 (83%) | 54% | 27.4 | 0.016 + 0.000 |
| yee-pec-box-1e5 | 2.95 (3.41) | 0.711 (84, 8.47) | 0.370 + 0.207 (81%) | 52% | 27.1 | 0.051 + 0.000 |
| ss-qa8fk-qa8fm | 2.09 (2.37) | 0.467 (100, 4.67) | 0.234 + 0.141 (80%) | 50% | 25.2 | 0.042 + 0.000 |
| wg-square-1e5 | 0.54 (0.65) | 0.206 (62, 3.32) | 0.067 + 0.045 (54%) | 33% | 18.0 | 0.020 + 0.030 |
| cav2d-rect-pert-1e5 | 0.32 (0.39) | 0.099 (22, 4.49) | 0.013 + 0.012 (25%) | 13% | 12.4 | 0.001 + 0.000 |
| wg-square-1e6 | 6.57 (7.26) | 2.355 (64, 36.80) | 0.390 + 0.866 (53%) | 17% | 21.8 | 0.253 + 0.622 |

The solve's BLAS time above the L0 layer (out) drops by 34-41% on the 3-D cases and
wg-square-1e6 (0.29-0.66 s -> 0.19-0.39 s), the L0 part by 19-76% on the 2-D cases (small trsm,
narrow kernels on tiny shapes) and little on the 3-D ones, and hsolve's own calls outside parallel
regions by 1.8-4.4x (split-K Gram products, parallel split); the factorization columns (not shown)
stay within the run-to-run noise (+-5%).

## 5. End to end (tools/bench/compare.py)

The six head-to-head configurations of results/cmp-p16-best.md and cmp-p16-best-interval.md
(`compare.py --solvers hsolve --cores 16-31`, 16 threads, each case's default configuration incl.
the adaptive ordering, tools/eval evaluation), plain hsolve (`build/release/hsolve`, the hook not
linked) against hsolve built with `-DHSOLVE_BLAS_HOOK=ON -DHSOLVE_BLAS_ROUTE_DEFAULT=all`, both at
commit e985acb; 5 interleaved repetitions (variant order alternating), solver time = report
`time_total_s`, medians [min-max]. 2026-10-13, gpu01 slot.

| case | target | plain s | hook (`all`) s | speedup | valid | MUMPS numeric s (#) plain / hook | MUMPS solve s (rhs) plain / hook | max eta plain / hook | evaluation (both) |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | k=12 @ 0 | 2.774 [2.743-2.863] | 2.652 [2.584-2.723] | 1.046 | 5/5, 5/5 | 0.506 / 0.519 (1) | 0.457 / 0.362 (44) | 1.4e-12 / 1.8e-12 | complete, set err 2e-17 |
| yee-pec-box-1e5 | k=10 @ 52.85 | 3.289 [3.257-3.391] | 2.907 [2.827-3.045] | 1.131 | 5/5, 5/5 | 0.511 / 0.504 (1) | 0.896 / 0.700 (84) | 1.6e-12 / 1.0e-12 | complete, 3e-17 |
| wg-square-1e5 | k=18 @ -0.5272 | 0.613 [0.590-0.635] | 0.536 [0.530-0.551] | 1.145 | 5/5, 5/5 | 0.129 / 0.126 (2) | 0.246 / 0.201 (62) | 3.7e-13 / 3.7e-13 | consistent, 0 |
| ss-qa8fk-qa8fm | k=10 @ 12.82 | 2.285 [2.239-2.383] | 2.016 [1.989-2.033] | 1.134 | 5/5, 5/5 | 0.884 / 0.900 (4) | 0.634 / 0.442 (100) | 7.4e-13 / 9.5e-13 | consistent, 0 |
| cav2d-rect-pert-1e5 | k=6 @ 0.1418 | 0.344 [0.338-0.395] | 0.309 [0.307-0.341] | 1.115 | 5/5, 5/5 | 0.138 / 0.138 (4) | 0.128 / 0.097 (22) | 6.4e-15 / 6.1e-15 | consistent, 0 |
| yee-pec-box-1e5 | [40, 200] | 4.437 [4.345-4.459] | 3.812 [3.785-3.835] | 1.164 | 5/5, 5/5 | 1.511 / 1.476 (3) | 1.574 / 1.198 (144) | 1.0e-11 / 2.1e-11 | complete, 4e-17 |

* **The hook is faster on every configuration, 1.05-1.16x**, with every run of both variants valid,
  the same completeness verdicts and the same set errors against the references; the backward
  errors stay far below the cases' tolerances. Except cav2d-rect-pert-1e5 (ranges touching at
  0.338 / 0.341) the min-max ranges do not overlap.
* All of it is solve time: MUMPS' solve phase -18% to -30%, numeric factorization unchanged.
* Against the tuned SLEPc of results/cmp-p16-best*.md (same slot and commands, measured in another
  session, so indicative only): syn-mem27-46 1.26x -> 1.32x, yee-pec-box-1e5 1.03x -> 1.15x,
  wg-square-1e5 1.50x -> 1.66x, ss-qa8fk-qa8fm 0.64x -> 0.73x, cav2d-rect-pert-1e5 0.64x -> 0.71x,
  interval [40, 200] 2.37x -> 2.74x. Where hsolve still trails, its four numeric factorizations
  (B-definiteness test, main, two inertia counts: 0.90 s and 0.14 s) against SLEPc's one (0.34 s,
  0.02 s) cost more than the whole remaining gap; no BLAS route changes the factorizations.

## 6. Recommendation and portable build story

* **Default: the hook linked into hsolve with route `all`** (`HSOLVE_BLAS_HOOK=ON`,
  `HSOLVE_BLAS_ROUTE_DEFAULT=all`, the CMake defaults since this measurement). It is 5-16% faster
  end to end on all six head-to-head configurations at equal accuracy (section 5), costs nothing
  measurable where it does not help (unrouted calls are forwarded; the factorization is
  unchanged), and keeps hsolve's guarantees: deterministic for a fixed thread count, no threads
  beyond MUMPS' own inside its parallel regions. Switches: `HS_BLAS_ROUTE=off` at run time (pure
  forwarding), `-DHSOLVE_BLAS_HOOK=OFF` at build time, `-DHSOLVE_BLAS_ROUTE_DEFAULT=` (empty) for a
  hook that forwards unless `HS_BLAS_ROUTE` says otherwise (the default also applies to
  `libhsolve_blashook.so`). The Python module is not linked with the hook; preload
  `libhsolve_blashook.so` there (section 4.2; checked for correctness, not timed).
  Existing build directories keep their cached `HSOLVE_BLAS_HOOK` / `HSOLVE_BLAS_ROUTE_DEFAULT`
  values: reconfigure with `-UHSOLVE_BLAS_HOOK -UHSOLVE_BLAS_ROUTE_DEFAULT` to pick up the
  defaults.
* **Keep the environment's OpenBLAS** (conda-forge, DYNAMIC_ARCH): rebuilding it for the CPU,
  BLIS and LIBXSMM bring nothing or lose (section 3), and DYNAMIC_ARCH picks the right kernels on
  any x86-64 CPU at run time.
* **Unknown judging CPU.** The hook is built without `-march=native`; its SIMD kernels carry
  their own target attributes and are chosen by CPUID at run time (AVX-512F, else AVX2 + FMA,
  else portable loops; on non-x86 CPUs only the portable loops are compiled). AVX2 gives the same
  end-to-end gain as AVX-512 here (section 4.4), and the largest part of the gain - the parallel
  split above MUMPS' L0 layer and split-K - calls the real BLAS per block and needs no SIMD code at
  all. The thresholds (256 KiB of A per thread for the parallel split, triangles up to 512 for the
  small trsm, N <= 16) follow this machine's ratio of one core's to the socket's bandwidth (1 : 2.6)
  and OpenBLAS' per-call overhead; on a CPU where one core can already saturate the memory the
  parallel split should gain less (not measured; it only starts at 512 KiB of A, where a parallel
  region's few microseconds are small against the call). The rest
  of hsolve follows the build presets: `release` compiles with `-march=native` (build on the
  judging machine), `portable` without (a binary built elsewhere); the hook behaves the same in
  both.
* Not pursued: an own dgemm for the factorization (at most ~20% of a factorization, section 2.1,
  and no existing kernel tried does better than OpenBLAS'), and BLIS / LIBXSMM.

## 7. Provenance and licenses

| component | version / source | license | used as |
|---|---|---|---|
| OpenBLAS | 0.3.34, release tarball (`tools/deps/build_openblas.sh`, sha256 `cd7e1298...`) | BSD-3-Clause | drop-in builds; the hook's narrow kernels adapt the structure of its `kernel/x86_64/dgemv_n_microk_skylakex-4.c` (NN; Copyright (c) 2014 The OpenBLAS Project) and `dgemm_small_kernel_{nn,tn}_skylakex.c` (TN, REDUCE_4; Copyright (c) 2021 The OpenBLAS Project) |
| BLIS | 2.1, tag tarball (`tools/deps/build_blis.sh`, sha256 `901752ec...`) | BSD-3-Clause | drop-in experiment only |
| LIBXSMM | 1.17, tag tarball (`tools/deps/build_libxsmm.sh`, sha256 `8b642127...`) | BSD-3-Clause | drop-in experiment only (`libxsmmext` call wrapper) |
| small triangular solve (`small_trsm_left`) | the project's own, loop orders of the reference BLAS dtrsm | project | hook route `strsm` |

OpenBLAS' copyright notices, its three license conditions and its disclaimer are reproduced in
full in `src/blas/simd_kernels.hpp` (BSD-3-Clause clause 1, source redistributions;
`src/blas/narrow_kernels.cpp` points there) and here, for the documentation of a binary
distribution of hsolve built with the hook (clause 2):

> Copyright (c) 2014, The OpenBLAS Project<br>
> Copyright (c) 2021, The OpenBLAS Project<br>
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted
> provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice, this list of
>    conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of
>    conditions and the following disclaimer in the documentation and/or other materials provided
>    with the distribution.
> 3. Neither the name of the OpenBLAS project nor the names of its contributors may be used to
>    endorse or promote products derived from this software without specific prior written
>    permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
> IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
> FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
> IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
> OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Checksums were taken from the downloads and cross-checked against Spack's package recipes
(OpenBLAS, LIBXSMM) and conda-forge's recipe (BLIS, OpenBLAS). No existing small-trsm kernel fits
MUMPS' calls: LIBXSMM's trsm generator is for packed batches of matrices, BLIS' small-trsm path
exists only in AMD's fork (AOCL-BLIS).

## 8. Reproduction

All runs from the repository root through `env/run.sh` on the cluster node
(`tools/dev/cluster_run.sh gpu01 -- ...`). `G` below is the slot guard,
`python3 tools/run/guard.py --exclusive --timeout 2700 --` (one run at a time on cores 16-31);
runs with T < 16 threads go through a nested `python3 tools/run/guard.py --exclusive --cores
16-$((16+T-1)) --threads T --` inside it. Binaries that a detached campaign times are copied out
of the work tree first.

```bash
# builds. The measurements here were made at e985acb, where the release build's hsolve was plain
# and the hooked one came from build/hook; since then the default build carries the hook, and a
# plain hsolve (build-plain below) needs -DHSOLVE_BLAS_HOOK=OFF.
tools/dev/build.sh release      # libhsolve_blashook.so, bench_asmkern_*, hsolve (hooked by default)
cmake -S . -B build-plain -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release \
  -DHSOLVE_BLAS_HOOK=OFF -DHSOLVE_BUILD_TESTS=OFF -DHSOLVE_BUILD_BENCH=OFF -DHSOLVE_BUILD_PYTHON=OFF
cmake --build build-plain --target hsolve_cli
# drop-in libraries (tarballs in $HOME/deps/src, sha256 checked by the scripts)
tools/deps/build_openblas.sh skx DYNAMIC_ARCH=0 TARGET=SKYLAKEX
tools/deps/build_openblas.sh skx-fwd DYNAMIC_ARCH=0 TARGET=SKYLAKEX GEMM_GEMV_FORWARD=1
tools/deps/build_openblas.sh skx-fwd-t1 DYNAMIC_ARCH=0 TARGET=SKYLAKEX GEMM_GEMV_FORWARD=1 GEMM_MULTITHREAD_THRESHOLD=1
tools/deps/build_blis.sh omp
tools/deps/build_libxsmm.sh wrap1

# 1. machine reference, T = 1 4 8 16, two sessions
K=build/release/bench_asmkern_kernels
$K peak --n 4096 --reps 3;  $K stream --mb 1024 --reps 10

# 2. profiles (route off; all; one column: config.json with "block_size": 1 merged in), with a
#    plain hsolve (built with -DHSOLVE_BLAS_HOOK=OFF, as here) and the preloaded hook ...
env LD_PRELOAD=$PWD/build/release/libhsolve_blashook.so HS_BLAS_ROUTE=off HS_BLAS_PROFILE=prof.json \
  build-plain/hsolve --A data/cases/C/A.npz [--B data/cases/C/B.npz] --config data/cases/C/config.json \
  --out out --threads T -q
#    ... or with the default build, whose hsolve carries the hook (no preload: never both)
env HS_BLAS_ROUTE=off HS_BLAS_PROFILE=prof.json build/release/hsolve ...
build/release/bench_asmkern_report prof.json      # per-phase, per-routine and per-shape tables

# 4.3 kernels at 16 threads (HS_BLAS_NO_AVX512=1: AVX2; plus HS_BLAS_NO_AVX2=1: portable)
$K shapes --reps 7 --pool-mb 1024 --inpar 1 --routes narrow,strsm=512,gemv1 --shapes \
  "trsm:LUTU:12:2:0;trsm:LUNU:12:2:0;trsm:LUTU:24:2:0;trsm:LUNU:24:4:0;trsm:LUTU:48:4:0;gemm:TN:12:2:12;gemm:TN:24:2:24;gemm:TN:96:4:24;gemm:NN:24:4:96;gemm:TN:384:4:96;gemm:NN:96:4:384;trsm:LUTU:24:1:0"
$K shapes --reps 7 --pool-mb 1024 --routes par,narrow+par,strsm=512,gemv1+par --shapes \
  "gemm:NN:400:4:1500;gemm:TN:1500:4:400;gemm:NN:100:4:700;gemm:TN:700:4:100;gemm:TN:400:4:48;gemm:TN:48:4:100000;gemm:NN:2000:1:2000;trsm:LUNU:48:4:0;trsm:LUTU:200:4:0;trsm:LUNU:400:4:0"

# 3. / 4.4 screening: one slot call, 3 interleaved repetitions, fixed orderings
H=$PWD/build/release/libhsolve_blashook.so D=$HOME/deps
$G build/release/bench_asmkern_e2e --cli build-plain/hsolve --threads 16 --reps 3 --run-timeout 300 \
  --cases syn-mem27-46,yee-pec-box-1e5,ss-qa8fk-qa8fm,wg-square-1e5,cav2d-rect-pert-1e5,wg-square-1e6 \
  --override 'syn-mem27-46:{"ordering": 3}' --override 'yee-pec-box-1e5:{"ordering": 3}' \
  --override 'ss-qa8fk-qa8fm:{"ordering": 3}' --override 'wg-square-1e5:{"ordering": 0}' \
  --override 'cav2d-rect-pert-1e5:{"ordering": 0}' --override 'wg-square-1e6:{"ordering": 0}' \
  --config base= --config "all=LD_PRELOAD=$H;HS_BLAS_ROUTE=all" \
  --config "strsmpar=LD_PRELOAD=$H;HS_BLAS_ROUTE=gemv1,strsm=512,par" \
  --config "allavx2=LD_PRELOAD=$H;HS_BLAS_ROUTE=all,avx2" \
  --config "skx=LD_PRELOAD=$D/openblas-0.3.34-skx/lib/libopenblas.so.0" \
  --config "skxfwd=LD_PRELOAD=$D/openblas-0.3.34-skx-fwd/lib/libopenblas.so.0" \
  --config "skxfwdt1=LD_PRELOAD=$D/openblas-0.3.34-skx-fwd-t1/lib/libopenblas.so.0" \
  --config "blis=LD_PRELOAD=$D/blis-2.1-omp/lib/libblis.so.4" \
  --config "xsmm=LD_PRELOAD=$D/libxsmm-1.17-wrap1/lib/libxsmmext.so.1 $D/libxsmm-1.17-wrap1/lib/libxsmm.so.1" \
  --out screen > screen/table.md
```

Section 5: compare.py runs `build/release/hsolve` of the repository it lives in, so each variant
gets a scratch copy of the repository - `tools/` copied, `data` linked to the work tree's,
`build/release/hsolve` linked to the plain (`build-plain/hsolve`; at e985acb the release build's)
or the hooked (`build/release/hsolve` now; at e985acb `build/hook/hsolve`) executable. Per
repetition R (1-3 in one slot call, 4-5 in a second; variant order plain, hook for odd R and hook,
plain for even R), for each variant V with scratch copy `$REPO_V`:

```bash
$G python3 $REPO_V/tools/bench/compare.py --tag e2e-V-rR --cores 16-31 --solvers hsolve --reps 1 \
  --mem-gb 28 --timeout 600 --out-dir OUT syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5
$G python3 $REPO_V/tools/bench/compare.py --tag e2e-int-V-rR --cores 16-31 --solvers hsolve --reps 1 \
  --mem-gb 28 --timeout 600 --target '{"mode": "interval", "lower": 40, "upper": 200}' --out-dir OUT yee-pec-box-1e5
```

(one outer `$G` holds the slot for all calls of a block of repetitions; compare.py's nested guards
share it). Section 5 takes per case and variant the median solver time (`time_total_s`, the scope
of results/cmp-p16-best.md) over the five repetitions.
