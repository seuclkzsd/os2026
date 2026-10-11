# P4 measurements

Collected from the P4 commit messages (hash in brackets), the header comments of the code, the P4
module reports and `results/baseline_tuning.md`. Every number keeps its method line; nothing here
was re-measured for this page.

Common setup unless a section says otherwise: the shared 2x Xeon Gold 6336Y server, every run under
`tools/run/guard.py` (`--skip-busy-check`, `--mem-gb` <= 8) on 2 or 4 cores of NUMA node 1, release
build, 2026-10-11. Each module had its own core slot (analysis 38,39; certfast 30,33; ghiep 28,29;
baseline 40,41; solvephase 34-37; drvint 24-27). The machine was loaded (load average ~34-44 on 96
hardware threads); identical runs moved by 10-40% between sessions, so every comparison below is an
interleaved same-session measurement (medians), and absolute times from different sections must not
be combined. "Base" is a8ed8f4 (P3) unless a section says otherwise. No 16-core run of P4 is
recorded here: the exclusive 16-core runs (SLEPc tuning, head-to-head, analysis, solves, GHIEP block
size) are pending on the cluster nodes (section 8).

## 1. solvephase - shift-invert apply, sparse kernels, refinement

**Split of one apply** (`bench_solvephase_profile`, 4 threads on cores 34-37, the driver's factor
options: L0-OpenMP, SCOTCH for 3-D, AMD for 2-D and the complex embedding; ms per column, final
run) [d3ee372, solvephase report]:

| case | apply p=1 | apply p=16 | gain | solve p=1 / p=16 | B*X p=1 / p=16 |
|---|---|---|---|---|---|
| syn-mem27-46 | 56.4 | 6.4 | 8.8x | 60.7 / 6.26 | 0.05 / 0.09 (copy, B = I) |
| yee-pec-box-1e5 | 60.9 | 6.6 | 9.2x | 63.4 / 6.8 | 0.05 / 0.08 (diagonal B) |
| wg-square-1e5 | 17.8 | 3.9 | 4.6x | 15.0 / 3.08 | 0.70 / 0.30 |
| ss-qa8fk-qa8fm | 28.3 | 3.4 | 8.3x | 31.3 / 3.27 | 0.66 / 0.13 |
| cav2d-rect-pert-1e5 | 13.3 | 3.1 | 4.3x | 13.2 / 2.98 | 0.19 / 0.12 |
| syn-near-singular-complex-1e5 | 48.3 | 10.8 | 4.5x | 54.7 / 11.3 | pack/unpack 0.07 / 0.23 |

- The block solve is 80-100% of an apply, B*X at most ~10% (wg-square-1e5, p = 16) [5bae9c5]. At
  1 and 2 threads the p = 1 -> 16 gain per column is 3.7-8.2x.
- One-column solve at 1 / 2 / 4 threads (earlier run): syn-mem27-46 110 / 81 / 46 ms,
  yee-pec-box-1e5 153 / 100 / 45, ss-qa8fk-qa8fm 60 / 56 / 26, wg-square-1e5 28 / 27 / 13,
  cav2d-rect-pert-1e5 19 / 18.5 / 11.5, complex 53 / 64 / 46 (no scaling).
- Effective bandwidth at 4 threads (two sweeps over the factor): 14-20 GB/s on the 3-D cases against
  a STREAM triad of 22-27 GB/s on the same cores; 4-8 GB/s on 2-D, 1-2 GB/s on the complex case.
- Backward error of a plain double solve: 9 to 2.5e5 eps; one correction brings it to 1.2-2.2 eps.
- A block solve costs more with every column, zero right-hand side or not: syn-mem27-46, 4 threads,
  46 ms for 1 column, 81 ms for 16 [346e522].

**Inside one MUMPS solve** (`bench_solvephase_mumps --exp split`, 4 threads, BLAS interposed; "above
L0" = upper tree, node by node with threaded BLAS, wall time; "L0" = parallel subtrees, summed over
threads):

| case | above L0 | L0 subtrees | rest | solve wall |
|---|---|---|---|---|
| syn-mem27-46 | 31.5 ms, 343 MB (41% of the factor bytes) | 103 ms, 494 MB | 32 ms | 66 ms |
| yee-pec-box-1e5 | 26 ms, 293 MB | 118 ms, 541 MB | - | - |
| ss-qa8fk-qa8fm | 17 ms, 139 MB | 54 ms, 274 MB | - | - |

wg-square-1e5 and cav2d-rect-pert-1e5: 6-7 ms above L0 for 21-40 MB, rest 14 ms (per-node
overhead). The complex case has no L0 layer: 50k sequential BLAS calls, 4 threads (87 ms) slower than
1 (57 ms). Above-L0 bandwidth 11-18 GB/s. MUMPS' own (non-BLAS) time grows with p: 32 -> 74 ms from
p = 1 to 16 on syn-mem27-46.

**MUMPS solve options** (`bench_solvephase_mumps`, 4 threads, interleaved medians): ICNTL(27) = -32
vs p vs 32 (p = 2..16, 9 reps) +-5%; ICNTL(21) = 1 0.83-1.01x; ICNTL(16) below the analysis thread
count refused with INFO -58 under L0-OpenMP, and with ICNTL(48) = 0 1/2/4 solve threads within 15%;
the OpenMP wait policy and a parallel trsm no gain; an AVX-512 "skinny" GEMM up to 1.19x for N <= 2
but 0.66-0.95x for N >= 8. Only routing dgemm calls with N = 1 to dgemv helped: 1.07-1.11x on
one-column solves of the five real 1e5 cases, 0.99x on the complex one (`--exp gemvfwd`; not
adopted, needs a BLAS hook in src/factor).

**Sparse kernels** (`bench_solvephase_kernels`, six 1e5 cases, p = 2..16, 4 threads, medians,
binaries alternated):
- Portable kernels compiled per chunk width, builds without -march=native (x86-64 SSE2 baseline
  and x86-64-v3): Portable vs Reference spmm 1.4-3.0x (was 0.77-1.1x), shifted_residual 1.1-4.9x
  (was 0.83-2.4x); vs the old Portable code 1.2-3.1x except complex p = 16 on SSE2 (0.94-0.99x)
  [41c126e].
- With AVX-512 (-march=native): on chunks narrower than 8 doubles per row the per-width Portable
  kernels beat the intrinsics (spmm real p = 2, 4: 2.2-2.9x vs 1.9-2.1x over Reference; residual
  3.6-4.9x vs 2.6-3.7x; complex p = 2 1.7x vs 1.4-1.5x); they tie at 8 doubles per row and lose at
  16 complex vectors (3.2x vs 3.8x). Hence Auto = AVX-512 only for >= 8 doubles per row [5d27a15].
- Strided fallback: a diagonal B lost 1.5x with the interleaved copy at p = 8, 16 (yee-pec-box-1e5)
  [be0c3f1]; cav2d-rect-pert-1e5 (5 entries per column) broke even at p = 8 and lost at 16 [70b5f3a].
- Fused residual vs the three-pass Reference, native build, cav2d-rect-pert-1e5, 4 threads, 21 reps x
  3 alternating rounds: 1.83-1.93x at p = 8, 1.35-1.39x at p = 16; over all runs p = 16 0.95-1.43x
  (median ~1.2x), p = 8 0.99-2.08x (fix-solvephase review).

**Refinement** (`bench_solvephase_fix_refine`, 4 threads, p = 1 / 8, arms rotated, accuracy
recomputed with one residual) [0e28bf2]:
- syn-near-singular-complex-1e5, double factor, rel_tol 1e-12 (unattainable): P4 1.96x / 2.78x faster
  than the P1 replica (rounding-level stop 1.55x / 2.15x, kernels 1.33x / 1.13x, active-column solves
  0.95x / 1.14x), same backward error 2.7e-16. The first measurement [48db355] ("1 correction per
  apply instead of 3-5, 2.3-6.3x less apply time") overstated it.
- Single-precision factors (the only refinement the driver enables): wg-square-1e5 1.07x / 1.09x,
  ss-qa8fk-qa8fm 1.10x / 1.13x, syn-mem27-46 1.06x / 1.01x; the stop never triggers there.
- syn-mem27-46 at sigma = 0 with a single factor: the refinement does not converge (10 steps,
  ||r||/||Bx|| ~ 5e-6); the driver does not check this yet (solvephase report).

**Lanczos end to end** (`bench_solvephase_fix_lanczos`, 4 threads, 9 reps, arm order alternating)
[0e28bf2]: ss-qa8fk-qa8fm block 8 total 1.063x [1.032, 1.094] (kernel time 1.22x, solves 1.004x),
block 1 (A/A control) 1.007x [0.925, 1.068]; cav2d-rect-pert-1e5 block 8 1.011x, block 1 1.017x.
Equal residuals and applications in both arms. Block 8 vs 1 (`bench_solvephase_lanczos`, 4
threads): 3.6x on syn-mem27-46, 1.7x on ss-qa8fk-qa8fm, 0.5x on cav2d-rect-pert-1e5 (k = 6, more
applications).

## 2. analysis - orderings and hsolve's nested dissection

**Orderings** (`bench_analysis_orderings`, 2 threads, cores 38,39, medians of 3 interleaved
one-process runs; analysis JOB=1 s / RINFOG(1) GFlop / numeric factorization s; hsolve ND: analysis
with 1 | 2 threads) [9cfc453, include/hsolve/factor.hpp]:

| case | AMD | METIS | SCOTCH | hsolve ND |
|---|---|---|---|---|
| syn-mem27-46 | 0.07/212/4.31 | 1.11/65/1.67 | 1.51/71/1.80 | 1.50 \| 0.85/66/1.78 |
| yee-pec-box-1e5 | 0.10/487/8.99 | 1.22/73/1.95 | 1.47/72/1.95 | 1.38 \| 0.75/81/2.17 |
| ss-qa8fk-qa8fm | 0.09/73/1.91 | 0.72/20/0.69 | 1.00/23/0.76 | 0.95 \| 0.55/18/0.61 |
| wg-square-1e5 | 0.06/1.2/0.19 | 0.70/0.55/0.15 | 0.72/0.59/0.15 | 0.82 \| 0.47/0.59/0.14 |
| cav2d-rect-pert-1e5 | 0.03/0.17/0.09 | 0.46/0.13/0.08 | 0.36/0.15/0.09 | 0.54 \| 0.32/0.15/0.09 |
| wg-square-1e6 | 0.72/55/2.55 | 8.54/19/1.64 | 7.30/18/1.62 | 9.98 \| 5.85/19/1.66 |

- At 2 threads ND's analysis is 1.3-1.6x faster than MUMPS' serial METIS; serially it is 1.2-1.35x
  slower. Its flops are within -7..+12% of METIS (+8..12% on yee-pec-box-1e5, wg-square-1e5,
  cav2d-rect-pert-1e5; -7% on ss-qa8fk-qa8fm; equal on syn-mem27-46, wg-square-1e6); no knob
  closed the yee gap (seeds, passes 20, tries 14, coarsen_to 200, imbalance 1.1/1.35, leaf 60/250:
  7.9-8.5e10 vs METIS 7.25e10). 2-thread speedup of the ordering 1.7-1.85x; task granularity 512
  vertices gives 1.83x, 4096 gave 1.25x [3a922d5].
- PORD 0.5-0.8 s and 1.0-2x METIS' flops (syn-mem27-46 0.50 s / 89 GF / 2.01 s); AMF about half of
  AMD's flops at AMD's cost on 3-D.
- The ordering is 90-97% of a METIS/SCOTCH analysis: the same analysis given the order (PERM_IN)
  takes 0.03-0.08 s at n ~ 1e5 (real), 0.35-0.49 s on the complex 1e5 case, 0.48-0.61 s at 1e6
  (MUMPS' timer on syn-mem27-46: METIS 1.00 of 1.05 s, symbolic factorization 0.026 s).
- Work W and critical path C of the dissection (1 thread) [ca5a7a5]: syn-mem27-46 1.41 / 0.29 s,
  yee-pec-box-1e5 1.23 / 0.22, ss-qa8fk-qa8fm 0.90 / 0.18, wg-square-1e5 0.74 / 0.13, wg-square-1e6
  9.6 / 1.4. Brent bound W/16 + C at 16 threads: <= 0.38 / 0.30 / 0.24 / 0.17 / 2.0 s plus the
  PERM_IN analysis, vs METIS' serial 1.11 / 1.22 / 0.72 / 0.70 / 8.5 s - an estimate, not a
  measurement (only 2 cores were available). Two serial runs side by side on cores 38 and 39 did not
  slow each other (1.42 vs 1.40 s).
- Analysis options: ICNTL(58) = 1 slows the given-order analysis 0.053 -> 0.220 s (syn-mem27-46),
  0.51 -> 0.87 s (wg-square-1e6); ICNTL(6) = 0 no change; SCOTCH_PTHREAD_NUMBER 1 vs 2: 1.43 vs
  1.47 s.
- Block analysis ICNTL(15) on syn-near-singular-complex-1e5 (every coupling complex): AMD analysis
  0.37 -> 0.28 s, SCOTCH 3.0 -> 0.30 s, flops 3.5e8 -> 3.3e8; factorization SCOTCH 0.638 vs 0.643 s,
  AMD +3% and +20% in two A/B series. Bloch-periodic Yee cell (n = 1e5, 3% complex couplings): METIS
  2.24e12 -> 2.69e12 flops with blocks.
- Complex embedding in compute_ordering: Bloch Yee 1e5 full embedding 2.53-2.59e12 vs compressed
  2.93-3.12e12 flops (3 seeds; METIS on the 2n graph 2.24e12); fully complex case compressed 3.53e8
  in 0.75 s vs full 3.82e8 in 3.0 s.
- Separator balance (`bench_analysis_fix_imbalance`, grid Laplacians, flops relative to imbalance
  1.2): 0.67-1.0x for 1.3-1.5, 0.75-1.7x for 1.7-1.8, 1.6-5x for 1.9, 170-250x for 1.999 [2dcf98b].
  On the real cases 1.5 costs +14..+70% flops on 3-D (syn-mem27-46 65.9 -> 98.9 GF, yee-pec-box-1e5
  81.0 -> 92.5, ss-qa8fk-qa8fm 18.5 -> 31.4) and -0..-14% on 2-D: the default stays 1.2 [3c644a6].

**What the driver pays** (`bench_analysis_fix_pipeline`, 2 threads, medians of 3 interleaved
one-process runs; A+f / A+3f / A+f+InertiaCounter pair [s] / 1-RHS solve [ms]) [0c1b0df]:

| case | METIS | SCOTCH | hsolve ND (2 threads) |
|---|---|---|---|
| syn-mem27-46 | 2.80/6.39/5.73/61 | 3.62/7.92/7.00/78 | 2.52/5.82/5.71/64 |
| yee-pec-box-1e5 | 3.15/6.94/6.26/75 | 3.34/7.05/6.52/71 | 2.75/6.59/6.47/76 |
| ss-qa8fk-qa8fm | 1.52/2.90/2.79/38 | 1.81/3.36/3.18/44 | 1.26/2.65/2.54/37 |
| wg-square-1e5 | 0.86/1.14/1.14/21 | 0.85/1.11/1.13/19 | 0.63/0.94/0.92/22 |
| cav2d-rect-pert-1e5 | 0.58/0.80/0.81/21 | 0.46/0.63/0.65/16 | 0.44/0.66/0.68/24 |
| wg-square-1e6 | 11.1/15.5/15.2/260 | 9.46/13.2/13.5/298 | 8.32/12.2/12.4/292 |

AMD on the 2-D cases: 0.23/0.57/0.56/22, 0.12/0.28/0.29/16, 3.82/9.87/9.11/275. ND's gain shrinks by
its extra fill: on yee-pec-box-1e5 it is 0.40 s for A+f, 0.35 s for A+3f and -0.22 s with the
InertiaCounter counts (break-even at 2 threads); a solve costs ND 0.95-1.16x METIS' time. AMD wins on
the 2-D cases. The driver does not use ND yet (DESIGN §3.6-3b).

## 3. drvint - driver integration (HEP/GHEP)

**Analysis-time model** (`bench_drvint_profile --exp analysis`, 42 data cases, median of 3 analyses
each, 1 thread of MUMPS analysis) [7899da8]: t = 3.4e-7 s * order + 2.5e-8 s * entries + 3.6e-10 s *
AMD factor entries predicts every case with n >= 1e4 within 0.41x .. 1.44x (median 0.95; order and
entries alone 0.18x .. 1.50x). The measured probe time it replaces varied 3x between runs of one input
(0.086 vs 0.258 s, syn-ghep-near-singular-real-1e5). SCOTCH / AMD analysis 2.2 .. 17, median ~9. With
the unchanged factor 7 every data case keeps its P3 ordering except wg-square-1e6, which sat at the
threshold (saving / SCOTCH cost 0.78 measured, 1.02 modelled).

**Ordering reproducibility** (`bench_drvint_fix_ordering`, 42 data cases, 4 threads, 3 interleaved
analyses per process) [1888a72]: the SCOTCH estimates varied between the analyses of one process on
every case, the METIS ones on none. On the 20 cases where the probe picks nested dissection METIS
analyses take 0.52x .. 1.98x SCOTCH's time (median 0.84x; ss-ge87h76 4.87 -> 2.53 s, but
syn-mem27-46 0.82 -> 1.06 s, yee-pec-cube-32 0.64 -> 0.98 s, yee-pec-box-1e5 0.81 -> 1.19 s,
ss-tem152078 1.07 -> 1.77 s); factorizations 0.84x .. 1.06x; nested-dissection / AMD analysis median
9.3 (SCOTCH), 9.8 (METIS). Three identical `deterministic = true` CLI runs of syn-lap3d-22 with SCOTCH
gave three different factor sizes (~1.7-1.8e6 entries) and eigenvectors; METIS gave 1515509 entries
and bitwise equal outputs.

**B definiteness screen** [d33b3bc]: the factorization of B it avoids costs 0.07 s of 0.62 s on
cav2d-rect-pert-1e5 and 0.10 s of 1.36 s on wg-square-1e5 (4 threads). Among the data cases only
cav2d-square-1e4 is diagonally dominant; the waveguide and other GHEP mass matrices are not.

**Certificate counts, main object vs InertiaCounter** (`bench_drvint_profile --exp counter`, 4
threads, medians of 5, main factors resident; main pair -> counter analysis + pair) [9d23e1d]:

| case | main object | counter | net |
|---|---|---|---|
| syn-mem27-46 | 1.959 | 0.063 + 1.758 (count -10%) | -0.14 s |
| yee-pec-cube-32 | 1.671 | 0.047 + 1.341 (-20%) | -0.28 s |
| yee-pec-box-1e5 | 1.904 | 0.057 + 1.626 (-15%) | -0.22 s |
| ss-qa8fk-qa8fm | 0.762 | 0.039 + 0.624 (-18%) | -0.10 s |
| ss-2cubes_sphere | 1.537 | 0.061 + 1.278 (-17%) | -0.20 s |
| syn-lap3d-aniso | 1.240 | 0.039 + 1.054 (-15%) | -0.15 s |
| cav2d-rect-pert | 0.115 | 0.024 + 0.098 (-15%) | +0.01 s (rule: no) |
| wg-square-1e5 | 0.187 | 0.041 + 0.165 (-12%) | +0.02 s (rule: no) |

Identical counts everywhere; `counter_pays` picks the counter exactly where the net saving is
positive. Counter working memory 135-313 MB next to 323-650 MB of main factors. Per right-hand side
solve cost at p = 1 / 4 / 8 / 16: syn-mem27-46 52.5 / 16.2 / 9.5 / 6.0 ms, yee-pec-box-1e5 54.2 / 16.1
/ 9.6 / 6.5 ms, wg-square-1e5 14.6 / 5.1 / 3.7 / 3.1 ms, cav2d-rect-pert-1e5 11.9 / 4.2 / 3.2 / 2.7 ms.

**Main factors during counter counts** (hsolve CLI, six 3-D 1e5 cases, 4 threads, medians of 6
interleaved runs; certificate s / peak RSS GiB; resident = ffb1f7f, release = 667d8e9, main =
667d8e9 with `inertia_counter = off`) [667d8e9]:

| case | resident | release (default) | main object |
|---|---|---|---|
| syn-mem27-46 | 1.59 / 1.09 | 1.62 / 0.85 | 1.76 / 0.86 |
| yee-pec-cube-32 | 1.31 / 0.87 | 1.32 / 0.66 | 1.43 / 0.68 |
| ss-qa8fk-qa8fm | 0.66 / 0.60 | 0.67 / 0.48 | 0.77 / 0.51 |
| yee-pec-box-1e5 | 1.61 / 1.00 | 1.62 / 0.77 | 1.74 / 0.79 |
| ss-2cubes_sphere | 1.31 / 0.89 | 1.33 / 0.68 | 1.44 / 0.70 |
| syn-lap3d-aniso-1e5 | 1.00 / 0.70 | 1.08 / 0.52 | 1.13 / 0.56 |

End to end the counter saves 0.05-0.14 s (5-13%, analysis included) of certificate time against
main-object counts; totals are within noise (3.13-4.92 s vs 3.38-4.96 s). Releasing the main factors
puts the peak 21-26% below resident counts and 2-7% below main-object counts. Same eigenvalues,
certificates and residuals in all three variants (eta <= 7.8e-12). Before this fix the resident
policy had raised the peak of syn-mem125-46 from 2.37 to 3.38 GiB (+43%; drvint report).

**Auto block width** (hsolve `--block-size 1 / 4 / 8` (ncv auto) on the 30 HEP/GHEP data cases, 4
threads, medians of 3 interleaved runs; iterate phase p = 1 -> 4) [f2737c7, src/driver/steps.cpp]:
- 3-D 1e5 (syn-mem27-46, yee-pec-cube-32, yee-pec-box-1e5, ss-2cubes_sphere, ss-qa8fk-qa8fm,
  syn-lap3d-aniso-1e5; k = 10..12): 1.4x .. 2.5x faster; p = 8 better than 4 only where multiple
  eigenvalues dominate (syn-mem27-46 3.6x, yee-pec-cube-32 3.0x);
- low-fill 1e5 (zero-cluster, near-singular; k = 10..21): 1.10x .. 1.16x;
- 3-D complex 1e4 (yee-bloch, yee-photonic-*): 2.0x; multiple eigenvalues (syn-*multiplicity*):
  1.4x .. 2.2x in total, no certificate rerun left;
- small low-fill 1e4: up to +0.1 s (syn-ghep-zero-cluster-complex-1e4 0.58 -> 0.80 s,
  syn-near-singular-complex-1e4 0.33 -> 0.46 s), the 11 syn-*-1e4 cases together 3.91 -> 3.73 s;
- k = 6, 7 (cav2d-*): p = 4 slower (0.37 -> 0.55 s, 22 -> 60 applications).
Rerun width (5 interleaved runs, block_size = 1 so that reruns happen): syn-ghep-multiplicity-real-1e4
(9 missing) single-vector rerun 59 applications / iterate 0.221 s, p = 10: 200 / 0.314 s, cap 4:
0.226 s; syn-multiplicity-complex-1e4 0.444 / 0.403 / 0.416 s (noise).

**Verify phase** (driver loops over columns in parallel; medians of 3 interleaved runs, 4 threads)
[e99f115]: syn-zero-cluster-real-1e5 (k = 21) 0.207 -> 0.155 s, syn-near-singular-complex-1e5
0.170 -> 0.130 s, syn-ghep-zero-cluster-complex-1e4 (k = 26) 0.046 -> 0.030 s, syn-mem27-46
0.071 -> 0.040 s. Other setup steps at n ~ 1e5: hermitian_defect <= 26 ms, the two norm estimates
<= 0.1 s, the pencil 2-6 ms.

**End to end** (drvint report, measured at ffb1f7f - before the review fixes 1888a72, 667d8e9,
9cee2c4; hsolve CLI on cores 24-27, 4 threads, tol 1e-10, medians of 3 interleaved runs
(syn-mem125-46 and ss-tem152078: 1 run); base = a8ed8f4, p=1 = `block_size = 1`, auto = new
defaults; phases of auto: setup / factor / iterate / verify / certificate s; SLEPc = compare.py
solver time (setup + solve), 1 rank x 4 threads and 4 ranks x 1 thread, with no 4-core tuned profile
(the profiles of section 6 are 2-core); * = SLEPc result not accepted (backward error or
incomplete); n/a = complex case (the os2026-ref PETSc build is real-only); last column = best SLEPc
time / hsolve auto, > 1 means hsolve is faster). All 96 evaluations (32 cases x 3 variants) pass
tools.eval.sweep; completeness "complete" against certified references, "consistent" otherwise.

| case | n | type | k | p | base | p=1 | auto | auto vs base | auto phases | max eta | SLEPc 1x4 / 4x1 | vs best SLEPc |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| cav2d-rect-pert-1e5 | 99459 | GHEP | 6 | 1 | 0.66 | 0.63 | 0.63 | 1.06x | .02/.16/.30/.02/.12 | 5.6e-15 | 1.42 / 0.74 | 1.18x |
| ss-2cubes_sphere | 101492 | HEP | 10 | 4 | 5.94 | 5.96 | 5.45 | 1.09x | .06/2.16/1.70/.03/1.49 | 6.9e-15 | 3.23* / 3.18* | 0.58x |
| ss-qa8fk-qa8fm | 66127 | GHEP | 10 | 4 | 4.45 | 3.89 | 3.69 | 1.21x | .06/1.63/1.21/.02/.73 | 3.5e-13 | 2.73 / 2.35 | 0.64x |
| syn-ghep-near-singular-real-1e5 | 1e5 | GHEP | 10 | 4 | 2.62 | 2.03 | 2.19 | 1.19x | .09/.48/1.32/.05/.25 | 1.2e-15 | 1.44 / 1.13 | 0.51x |
| syn-lap3d-aniso-1e5 | 97200 | HEP | 12 | 4 | 4.48 | 3.98 | 3.92 | 1.14x | .01/1.29/1.43/.05/1.15 | 2.8e-12 | 3.47 / 3.57 | 0.89x |
| syn-mem27-46 | 97336 | HEP | 12 | 4 | 5.99 | 5.74 | 5.06 | 1.18x | .06/2.07/.90/.05/1.99 | 1.9e-12 | 3.97 / 4.17 | 0.79x |
| syn-near-singular-complex-1e5 | 1e5 | HEP | 10 | 4 | 4.69 | 4.81 | 5.08 | 0.92x | .16/1.24/2.73/.09/.86 | 4.8e-11 | n/a | - |
| syn-zero-cluster-real-1e5 | 1e5 | HEP | 21 | 4 | 2.43 | 2.43 | 2.21 | 1.10x | .06/.45/1.24/.13/.34 | 6.6e-15 | 1.87 / 2.86 | 0.85x |
| yee-pec-box-1e5 | 109744 | GHEP | 10 | 4 | 6.56 | 6.14 | 6.09 | 1.08x | .03/2.10/1.85/.04/1.94 | 9.6e-13 | 4.10* / 4.05* | 0.66x |
| yee-pec-cube-32 | 92256 | HEP | 12 | 4 | 7.00 | 6.73 | 4.84 | 1.45x | .02/1.66/1.45/.04/1.59 | 1.9e-12 | 4.23* / 3.86* | 0.80x |
| syn-mem125-46 | 97336 | HEP | 15 | 4 | 30.45 | 30.42 | 21.62 | 1.41x | .26/7.40/5.84/.08/8.05 | 1.8e-12 | - | - |
| ss-tem152078 | 152078 | GHEP | 10 | 4 | 9.66 | 8.55 | 8.02 | 1.20x | .29/3.39/2.36/.07/1.90 | 4.2e-21 | - | - |
| syn-ghep-multiplicity-real-1e4 | 1e4 | GHEP | 22 | 4 | 0.36 | 0.35 | 0.21 | 1.75x | | 5.4e-16 | 0.40 / 0.33 | 1.59x |
| syn-multiplicity-complex-1e4 | 1e4 | HEP | 14 | 4 | 0.58 | 0.62 | 0.34 | 1.71x | | 9.0e-14 | n/a | - |
| syn-multiplicity-real-1e4 | 1e4 | HEP | 14 | 4 | 0.18 | 0.18 | 0.13 | 1.45x | | 3.6e-14 | 0.22 / 0.22 | 1.73x |
| syn-lap2d-100 | 1e4 | HEP | 12 | 4 | 0.16 | 0.12 | 0.10 | 1.57x | | 3.2e-13 | 0.12 / 0.14 | 1.11x |
| yee-bloch-1e4 | 10080 | HEP | 10 | 4 | 1.81 | 1.89 | 1.52 | 1.19x | | 2.9e-12 | n/a | - |
| yee-photonic-rods-1e4 / -spheres-1e4 | 10125 | HEP / GHEP | 10 | 4 | 1.88 / 1.95 | 1.93 / 1.93 | 1.64 / 1.85 | 1.15x / 1.05x | | 1.6e-12 / 2.5e-12 | n/a | - |
| syn-near-singular-complex-1e4 | 1e4 | HEP | 10 | 4 | 0.30 | 0.29 | 0.38 | 0.81x | | | n/a | - |

- Total over the 32 cases 97.2 -> 80.0 s (1.21x). Real 3-D nearest cases: hsolve auto at 0.58x to
  0.89x of the best SLEPc; the gap is the certificate (two count factorizations, 1.2 to 2.0 s, 30-40%
  of the run). SLEPc's results on yee-pec-box-1e5, yee-pec-cube-32 and ss-2cubes_sphere are not
  accepted; hsolve's are certified.
- Auto block width loses on some small low-fill complex 1e4 cases (above; syn-zero-cluster-complex-1e4
  0.50 -> 0.57 s) and on ss-helm3d01 (0.98 -> 1.13 s, within noise of p = 1 at 1.08 s).
- Not run: ss-3dspectralwave2 (SCOTCH factor estimate ~16 GB > the 6 GiB limit), ss-si34h36 and
  ss-ge87h76 (159-265 s per run). Window targets (interval / smallest / largest) were not measured
  with blocks. GHIEP is section 4.

## 4. ghiep - block Krylov-Schur for indefinite B, post-processing memory

**Block step kernel** (`bench_ghiep_fix_ortho`, n = 1e5, k = 24..72, 2 threads on cores 28-29, 21
rounds, real; fast path `block_cgs_cholqr2` vs the two-pass `BlockOrthonormalizer`) [997bfe4]: p = 2..8
1.3-2.0x on Krylov-like blocks (80% cancellation, DGKS second pass) and 1.8-3.1x on random blocks; p =
16 1.3-1.6x; p = 64 1.0-1.3x; loss of orthogonality <= 4e-15 for both (the earlier "2.3-2.7x" came
from an uncommitted scratch bench). Inside krylov_schur_general (`bench_ghiep_block`, tol 1e-11,
medians of 3): orthogonalization 0.237 -> 0.162 s (wg-square-1e5, p = 4), 0.551 -> 0.290 s
(wg-halfslab-1e5, p = 4); no step of p = 2..8 on four wg-* cases used the fallback [6295caf].

**Memory** (`bench_memio_e2e`, wg-square-1e6, SCOTCH fixed, 2 threads, cores 28-29):
- result phase of krylov_schur_general (vectors formed as Q1 z after the basis is freed): 736 -> 448 MB
  (n = 998001, ncv = 37, m = 18) [feb126e];
- post-processing (Schur-basis union, column-block projection, batches of 4, factors released):
  peak RSS 2886 -> 1760 MB, post-processing 3.50 -> 2.44 s, its peak now at its start (1464 MB)
  [ca29887]; with the p = 4 rule 1889 MB [32e7a92]; final build with p = 2: 1766 MB (report) /
  1.65 GiB (guard), post-processing 1.29 s with no growth (base: +1150 MB spike) (ghiep report).

**End to end vs base** (hsolve CLI, 2 threads, interleaved, medians):
- wg-square-1e5 2.33 -> 1.84 s, wg-halfslab-1e5 3.49 -> 2.58 s (3 runs); sigma = 0 cluster-mode
  variants 9.97 -> 8.43 s and 15.44 -> 12.26 s with peak RSS 0.45 -> 0.22 GB and 0.64 -> 0.26 GB (2
  runs); same eigenvalue sets, eta <= 1.6e-12 [25fa6e9].
- wg-square-1e5: iterate 1.57 -> 1.19 s, verify 0.25 -> 0.18 s, peak 0.275 -> 0.176 GiB, 31 multi-RHS
  solve calls instead of 57 single ones; wg-halfslab-1e5 peak 0.361 -> 0.202 GiB; wg-square-1e6 (SCOTCH
  fixed, 2 runs) 37.2 -> 25.0 s, iterate 22.6 -> 11.3 s, verify 3.0 -> 1.9 s, guard peak 2.67 ->
  1.67 GiB, eta <= 2.2e-13 (ghiep report).
- Candidate refinement (squared-magnitude pivot, parallel loops): verify phase wg-square-1e5
  0.178 -> 0.099 s, wg-halfslab-1e5 0.237 -> 0.200 s [25eac09].

**Block size p = 1 / 2 / 4 / 8** (re-measured with committed programs: `bench_ghiep_fix_e2e` on the
data cases (`--sigma 0` for the one-sided rows), `bench_ghiep_block --mode e2e` on the synthetic
pencils, `--mode ks` for per-vector costs; 2 threads, cores 28-29, 3 interleaved repetitions,
medians; seconds) [740f1a7]:

| case | p = 1 | p = 2 | p = 4 | p = 8 |
|---|---|---|---|---|
| wg-square-1e5 | 2.09 | 1.56 | 1.57 | 1.69 |
| wg-square-1e5, sigma 0 | 9.9 | 8.6 | 14.6 | 10.1 |
| wg-halfslab-1e5 | 3.89 | 2.46 | 3.10 | 3.40 |
| wg-halfslab-1e5, sigma 0 | 14.5 | 11.4 | 10.6 | 12.7 |

Synthetic pencils (syn-ghiep-{wg,wg0}-{20480,102400}, syn-ghiep-cplx-102400): p = 2 1.06-1.14x,
p = 4 0.97-1.12x, p = 8 0.79-1.12x as fast as p = 1. Over the nine cases p = 2 never lost (1.06-1.58x),
p = 4 lost up to 1.5x; every solve succeeded (eta <= 3e-12) with the same eigenvalues (<= 3e-10
relative, as between repetitions of p = 1). n ~ 1e4 (wg-square-pert-1e4, wg-halfslab-1e4): p = 2
1.03-1.19x on totals of 0.11-0.16 s (noise level), so p = 1 there. Solve cost per vector p = 1 / 2 /
4 / 8: 20.8 / 11.5 / 7.9 / 5.5 ms (wg-square-1e5), 277 / 150 / 105 / 65 ms (wg-square-1e6);
applications on wg-square-1e5 (k = 18): 57 / 62 / 84 / 128, at sigma = 0 (whole solve) 284 / 400 /
620 / 855 (src/driver/indefinite.cpp). Cluster mode is slow whatever p: wg-square-1e5 at sigma 0
takes 294 applications with p = 1, syn-ghiep-wg0-102400 640 (the inner-band-edge probes never
converge, ~50 applications each).

## 5. certfast - completeness checks that cost solves (src/verify, not yet called by the driver)

`bench_certfast_measure`, 2 threads on cores 30,33, X = hsolve's own CLI solution (the driver's
ordering and certificate window), medians of 5 interleaved repetitions, seconds [8b2c527,
include/hsolve/verify.hpp]. exact: driver path (refactorize the main object at both window ends) /
InertiaCounter double pair / single pair (certified by the single-count rule?). randomized:
exclusion_check with the driver's window as pass radius, default p = 4 [p = 1], delta 1e-6 / 1e-12,
including the block application of Op to X. d/r: next Ritz hint over the claim radius.

| case | k | d/r | B | eps | exact drv / cnt / single | rand 1e-6 / 1e-12 [p = 1] |
|---|---|---|---|---|---|---|
| syn-mem27-46 | 12 | 387 | I | 0.995 | 3.10 / 2.86 / 1.62 cert | 0.50 / 0.50 [0.50 / 0.59] |
| yee-pec-cube-32 | 12 | 4.0 | I | 0.60 | 2.80 / 2.56 / 1.37 cert | 1.04 / 1.03 [1.07 / 1.93] |
| yee-pec-box-1e5 | 10 | 1.36 | diagonal | 0.15 | 3.12 / 2.76 / 1.59 cert | 2.31 / 3.31 [2.64 / 4.58] |
| ss-qa8fk-qa8fm | 10 | 1.34 | general* | 0.14 | 1.34 / 1.10 / 0.64 not cert | 2.03 / 2.61 [1.97 / 2.95] |
| syn-multiplicity-real-1e4 | 14 | 1.16 | I | 0.073 | 0.025 / 0.024 / invalid | 0.13 / 0.18 [0.09 / 0.17] |
| syn-multiplicity-complex-1e4 | 14 | 1.16 | I | 0.073 | 0.081 / 0.075 / invalid | 0.27 / 0.34 [0.17 / 0.26] |
| syn-ghep-multiplicity-real-1e4 | 22 | 6.0 | general* | 0.71 | 0.027 / 0.027 / not cert | 0.07 / 0.08 [0.05 / 0.06] |

- \* kappa only estimated (2.6e3, 1.3e4): the randomized result there is a screening result.
- Every complete X passed (no false alarm); every X with one copy of the largest found multiplet
  removed, or the farthest found eigenvalue removed, was detected in every repetition, delta and p
  (new Ritz value within 4e-12 of the removed eigenvalue).
- Against the fastest certifying exact option the check wins only on syn-mem27-46 (3.3x; 5.8x
  against the counter pair) and yee-pec-cube-32 (1.3x); it loses on the narrow gap of
  yee-pec-box-1e5 (20-26 block steps of 4: 1.5x / 2.1x slower than the certified single counts) and
  on the cheap 1e4 factorizations. `certificate_cost` recommended the measured fastest certifying
  method in all seven cases; its randomized estimate was within 26% on the 1e5 cases, up to 55% above
  on the 1e4 cases.
- The "hint" pass radius r' = r + 0.9 (d_hint - r) (bench experiment, not a driver policy): 1.86 /
  2.55 s on yee-pec-box-1e5 and 1.57 / 1.96 s on ss-qa8fk-qa8fm (p = 4), up to 38% faster on the 1e4
  cases, within 5% on syn-mem27-46 and yee-pec-cube-32; it risks false alarms (d_hint >= d_next).
- Against an earlier 3-repetition run of the same bench (which took the better of window / hint)
  the medians moved by up to 26%; the ranking of the methods did not change.
- Empirical miss rate (test VerifyBound.EmpiricalMissRateStaysBelowTheBound, 200 seeds per Krylov
  size) stays below the reported bound [3bab294].

## 6. baseline - tuned SLEPc and the fair comparison (2 cores)

Full method, configurations and pitfalls: `results/baseline_tuning.md` (tune_slepc.py, cores 40,41,
5 interleaved runs, 21 for the randomly failing wg-square-1e5; every run judged by tools/eval at the
case tol). The hsolve of this comparison is the P3 code (the baseline branch is based on a8ed8f4),
i.e. before the P4 block Krylov, InertiaCounter and GHIEP changes.

- Tuned vs default SLEPc: 1.13x syn-mem27-46, 1.20x ss-qa8fk-qa8fm, 1.07-1.29x wg-square-1e5 (two
  sessions), 1.94x cav2d-rect-pert-1e5, 1.03x (noise) on the interval case. The default is invalid on
  yee-pec-box-1e5 (5/5 runs miss a copy of a multiple eigenvalue) and fails at random on
  wg-square-1e5; almost every wg-square-1e5 configuration fails at random, the stored profile passed
  63 of 63 runs.
- Head-to-head (solver time, same scope; hsolve / hsolve without certificate / tuned SLEPc, s; all
  75 + 9 runs valid):

| case | hsolve | nocert | SLEPc | hsolve speedup | nocert speedup |
|---|---|---|---|---|---|
| syn-mem27-46 | 9.57 | 6.26 | 4.88 | 0.51x | 0.78x |
| yee-pec-box-1e5 | 9.98 | 8.30 | 6.65 | 0.67x | 0.80x |
| wg-square-1e5 | 2.54 | 2.47 | 2.63 | 1.04x | 1.07x |
| ss-qa8fk-qa8fm | 7.64 | 5.70 | 3.53 | 0.46x | 0.62x |
| cav2d-rect-pert-1e5 | 1.00 | 0.80 | 0.66 | 0.66x | 0.83x |
| yee-pec-box-1e5 [40, 200] | 18.01 | 18.91 | 26.89 | 1.49x | 1.42x |

- The gap: hsolve's analysis (1.44 s vs SLEPc's PORD 0.52 s on syn-mem27-46) and its 3-4
  factorizations against SLEPc's 1 (numeric time 2.8-4.5x SLEPc's); per solve hsolve is within 10% of
  SLEPc on four cases and 22% slower on ss-qa8fk-qa8fm; on the interval case it needs half the solves.

## 7. Cluster nodes and platform-dependent results

- `KrylovSchurGeneralBitwise`: on the cluster nodes (Xeon Silver 4314) the kernel fingerprint equals
  the server's (Gold 6336Y), but OpenBLAS's products in configuration 10 (n = 5000) differ in the last
  bits; the reference hashes are now keyed by fingerprint and CPU model (the outputs of the merged
  main and of the pre-merge ghiep branch on gpu01 are bitwise identical) [39552d2].
- `mpiexec --bind-to core` ignores the guard's cpuset: under a 40,41 pin the SLEPc ranks were bound
  to CPUs 0/48 and 1/49 (about 25 short trials, ~1 min in total, were discarded);
  `baseline_slepc.py --bind core` pins each rank to its block of the inherited cpuset [58fdc65,
  baseline report].

## 8. Not measured / pending

- 16 threads: the SLEPc 16-core tuning and head-to-head (commands in `results/baseline_tuning.md`),
  the ND analysis at 16 threads (`bench_analysis_orderings --threads 16 --orderings 5,3,pnd:16,pnd:1
  --factorize 1`, one process per measurement, >= 3 interleaved reps; `--exp dissect` for the
  dissection alone), the solve phase at 8-16 threads (`bench_solvephase_mumps --exp split|gemvfwd|
  skinny`, `bench_solvephase_profile --p 1,2,4,8,16`), the GHIEP block size (`bench_ghiep_fix_e2e` /
  `bench_ghiep_block --mode e2e`).
- Window targets with block Krylov (slice cost model calibrated for p = 1), the randomized
  certificate end to end (not wired), the factorization of the 1e5 Bloch case with block analysis
  (~3 GB of factors, too much for the 4 GB limit of that slot; only analysis estimates were taken).
