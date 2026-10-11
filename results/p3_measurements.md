# P3 measurements

Collected from the P3 commit messages (hash in brackets), the header comments of the code, and the
P3 module reports. Every number keeps its method line; nothing here was re-measured for this page.

Common setup unless a section says otherwise: the shared 2x Xeon Gold 6336Y server, every run under
`tools/run/guard.py` with 2 cores (`--skip-busy-check`, `--mem-gb` <= 8), release build, 2026-10-11.
The machine was loaded (load average ~37-54) during the P3 runs; identical runs varied by 10-30% (up
to ~2x for the block Krylov runs), so methods were interleaved and only same-session comparisons are
quoted as A vs B. "Base" is commit 17a2a79 (P2) built from the same bench source.

## 1. memio - input assembly, pencil, end-to-end memory

**Load** (`bench_memio_load --grid 70 --mtx-grid 45`; 27-point Hermitian matrix, n = 343000,
nnz = 9.0e6, real and complex; peak = VmHWM after the load (reset via /proc/self/clear_refs) minus
VmRSS before, divided by the final CSR bytes; time = minimum of interleaved A,B,B,A,A,B runs, base
library vs new) [d96d12d, de3746f, a4f3e75]. Peak / result, real / complex:

| layout | base | P3 | load time |
|---|---|---|---|
| npz csr full | 1.03 / 1.01 | 1.00 | - |
| npz csr upper / lower | 2.52 / 2.50 | 1.22 / 1.34 | -25..-35% |
| npz csc full | 2.02 / 2.02 | 1.34 / 1.20 | -7 / -12% |
| npz csc upper | 2.52 / 2.50 | 1.22 / 1.34 | -28 / -30% |
| npz coo full | 2.35 / 2.21 | 1.34 / 1.20 | -1 / -10% |
| npz coo upper | 2.52 / 2.50 | 1.22 / 1.34 | -26 / -31% |
| npz csr, f4 values + int64 indices | 1.32 / 1.39 | 1.02 / 1.01 | -26 / -23% |
| Hermitian part, asymmetric pattern | 3.00 / 3.00 | 1.70 / 1.82 | -18 / -24% |
| near-Hermitian, symmetric pattern | 1.02 / 1.01 | unchanged (in place) | - |
| MatrixMarket general (n = 91125, nnz = 2.35e6) | 2.49 / 2.29 | 1.20 / 1.11 (incl. 4 MiB line buffer) | +18 / +10% (counting pass) |
| MatrixMarket symmetric | 2.52 / 2.50 | 1.22 / 1.33 | +3 / -8% |

csc/coo full went 1.36 / 1.22 -> 1.335 / 1.202 and csr 1.025 -> 1.002 with the per-row work arrays
removed [a4f3e75]. Every loaded matrix equals the generator's (bitwise for exactly Hermitian input).
Caveat: the `npz-csr-nearherm-asym` layout measures the in-place (symmetric) case when grid % 4 == 0
(grid 40: 1.020 / 1.014, grid 41: 1.724 / 1.819); the default grid 70 is unaffected.

**Few entries per row** (`bench_memio_fix_rows`, n = 3e6, cores 34,35) [4d1d680, a4f3e75], peak /
result before -> after: diagonal real coo 1.80 -> 1.22, csc 1.80 -> 1.22, csr 1.40 -> 1.00, mtx
1.87 -> 1.07; tridiagonal real coo/csc 1.46 -> 1.28, csr 1.18 -> 1.00, mtx 1.39 -> 1.03; tridiagonal
complex coo/csc 1.30 -> 1.18, csr 1.12 -> 1.00, mtx 1.26 -> 1.02; one stored triangle unchanged 1.55
/ 1.59. Load times unchanged within noise (MatrixMarket tridiagonal 1.398 -> 1.395 s real).

**Oversized header claims** (hsolve CLI on a 1.2 KB npz whose header claims 3.75e8 elements, cores
34,35) [088c825]: 2.80 GiB / 1.32 s before the Io error -> 0.071 GiB / 0.035 s.

**ShiftedPencil** (`bench_memio_pencil --A --B` on case files; minimum of 6 interleaved runs x 30
assemblies, old vs new; FNV-1a hash of the assembled values identical in all cases) [5aa21e7]:

| case | retained MB | assemble ms |
|---|---|---|
| wg-square-1e6 (A+B) | 112.5 -> 33.2 | 42.0 -> 24.3 |
| ss-tem152078 (A+B) | 69.8 -> 14.6 | 16.5 -> 11.3 |
| ss-ge87h76 (A only) | 17.6 -> 17.0 | 9.4 -> 9.3 |
| ss-3dspectralwave2 (complex, A only) | 31.1 -> 28.9 | 36.1 -> 36.4 |
| syn-ghep-near-singular-real-1e5 (B much sparser than A: merge path) | 11.1 -> 6.5 | 4.1 -> 5.7 |

Construction on wg-square-1e6 0.137 -> 0.063 s; no 16 MB transient per assemble. An intermediate
variant without the whole-row fast path was 35-76% slower on A-only and sparse-B cases.

**End to end** (`bench_memio_e2e`: the CLI replayed in one process, VmRSS sampled every 2 ms, driver
log events as phase marks) [b6b3fa8 and the memio report]. wg-square-1e6 (n = 998001, nnz A = 3.7e6,
B = 1.06e7, GHIEP, k = 18), ordering pinned to SCOTCH:
- peak 2853 / 2861 MB with the new pencil vs 2962 / 2919 MB with the old one (-83 MB, -2.8%);
- phases: loaded A+B 207 MB (CSR 188 MB) -> norms 224 -> SCOTCH analysis 550-610 (pencil 33, MUMPS
  input 97, MUMPS analysis ~0.3 GB) -> B factorization for the definiteness test 1239-1278 (factor
  INFOG(21) 828 MB; later factorizations reuse it) -> Krylov-Schur peak 1975-2014 (basis n x 38 x 8 B
  = 303 MB plus complex result vectors 303 MB) -> GHIEP post-processing sets the whole-run peak:
  +1.18 GB on top of ~1.2 GB of live MUMPS state;
- the post-processing needs ~88 B n m (n x m x (16 result vectors + 3 x 16 for W, AW, BW + 16 + 8)),
  1.67 GB at m = 19, while the driver budgets ~0.86 GB for the Krylov phase.
- Two identical runs with the adaptive ordering chose SCOTCH (72.4M factor entries, INFOG(21) 828 MB,
  peak 2.92 GB) and AMD (97.0M, 1140 MB, 3.10 GB): the choice compares against the measured AMD
  analysis time. With ordering = 3 fixed, 4 runs gave 72.02M-72.75M factor entries; ss-ge87h76 SCOTCH
  estimates were 5.80e8 vs 5.91e8 entries in two processes.
- During the adaptive analysis the AMD object (input 97 MB + analysis ~136 MB) stays alive while SCOTCH
  analyses: phase peak 949 MB vs 700 MB after.

Other cases: ss-tem152078 (GHEP, n = 152078) peak 1.14 GB, the MUMPS factor ~0.7 GB of it;
ss-ge87h76 SCOTCH in-core estimate 8.4-8.8 GB (AMD 26.9 GB, BLR 7.2 GB), not run under the 8 GiB limit
[b6b3fa8]. The complex `mumps::Input` keeps 85 B per pattern entry: 564 MB on ss-3dspectralwave2 vs
29 MB pencil and 261 MB input CSR.

## 2. ortho - deterministic reductions and CGS kernels

**Reductions** (`bench_ortho_fix_kernels reduce`, cores 26,27, new vs the OpenMP-reduction versions
interleaved, three runs, speedup ranges) [c25d774]:

| | n = 1e4 | n = 1e5 | n = 2e6 |
|---|---|---|---|
| real dot | 3.2-3.5x | 6.4-6.7x | 1.76-1.85x |
| complex dot | 1.03-1.05x | 1.81-1.83x | 1.09-1.10x |
| real norm2 | 4.1-4.2x | 8.1x | 4.6-4.7x |
| complex norm2 | 2.1x | 3.5-3.6x | 1.59-1.64x |

The first, non-interleaved measurement [78795e6] ("dot/norm2 2-7x") overstated the complex dot.
Results are bitwise identical for 1, 3 and 4 threads (tests).

**One CGS call** (`bench_ortho_kernels cgs 200000 64`; legacy = four BLAS gemv sweeps; methods
interleaved, medians of 9) [28fcce0]:

| mode | legacy | two passes | one DGKS pass |
|---|---|---|---|
| real Euclidean | 16.6 ms | 11.9 (1.39x) | 8.2 (2.02x) |
| complex Euclidean | 36.7 | 32.7 (1.12x) | 21.9 (1.68x) |
| real stored M*V | 28.5 | 24.5 (1.17x) | 15.1 (1.89x) |
| complex stored | 64.6 | 63.2 (1.02x) | 38.5 (1.68x) |
| recompute (M diagonal) | - | 0.89-1.09x (same 4 sweeps) | 1.79x complex / 2.16x real |

max |V^H M q| after the call 4e-18 .. 2e-17 for legacy, Always and DGKS alike. Small n after the
vector-extension kernels and the >= 64 row blocks rule (`bench_ortho_fix_kernels cgs`, two passes,
speedup over the in-process legacy CGS2; before c25d774 -> after f66fc84, chained over the two
commits' separate sessions) [c25d774, f66fc84]: complex n = 1e3, k = 30: Euclidean 0.53 -> 1.16, stored 0.62 ->
1.16, recompute 0.46 -> 1.00; complex n = 1e4: 0.89 -> 1.25 / 0.89 -> 1.16 / 0.86 -> 1.01; real n =
1e4: 2.01 -> 2.77 / 1.96 -> 2.30 / 1.52 -> 2.16. The complex two-pass CGS is now at least on par with
the legacy code at every measured size (n = 1e3 .. 2e5); one DGKS pass 1.5-2x. One-thread kernel
bandwidth (n = 1e3, L2): complex projection 25.5 -> 60.8 GB/s, real 37.8 -> 60.7 GB/s.

**Block CGS2** (`bench_ortho_kernels block 200000 64 8`, two passes, medians of 5) [fc2f76f]:

| mode | 8 legacy calls | 8 new column calls | block | speedup |
|---|---|---|---|---|
| real Euclidean | 275.6 ms | 182.2 | 39.5 | 7.0x / 4.6x |
| complex Euclidean | 486.0 | 373.1 | 93.3 | 5.2x / 4.0x |
| real stored | 336.5 | 260.0 | 79.2 | 4.3x / 3.3x |
| complex stored | 708.2 | 602.2 | 136.5 | 5.2x / 4.4x |
| real recompute | 235.4 | 192.6 | 66.9 | 3.5x / 2.9x |
| complex recompute | 494.0 | 480.7 | 141.5 | 3.5x / 3.4x |

Loss max |V^H M q| 5e-18 .. 2e-17 for all three.

**Lanczos** (tests/cpp/test_lanczos_perf.cpp, DISABLED benchmarks, n = 2e5, ncv = 101, Euclidean;
same test source built against base, 4373b71 and the final branch, 5 rounds with rotated order;
orthogonalization ms per Lanczos step, medians) [d3565c0]: shift-invert spectrum real 17.09 -> 11.28
(1.51x), complex 36.25 -> 26.13 (1.39x); uniform spectrum real 20.95 (4373b71) -> 19.25. DGKS skipped
pass 2 in 35% (real) / 28% (complex) of the Lanczos steps [28fcce0]. max relative eigenvalue error
unchanged (real 2.2e-15 -> 2.4e-15, complex 8.0e-15 -> 5.4e-16).

**Real shift-invert path** (`bench_ortho_shift_invert <case> 1 0 <nev>`, base vs 4373b71, medians of
3 interleaved rounds) [4373b71]: at the configured k orthogonalization is only 8-11% of the iteration:
cav2d-rect-pert-1e5 GHEP k = 6 0.052 -> 0.045 s (1.16x), syn-near-singular-complex-1e5 k = 10 0.251
-> 0.215 s (1.17x), wg-square-1e5 GHIEP k = 18 0.199 -> 0.161 s (1.24x); total iteration 0.98-1.02x.
At nev = 60 the first claim (1.65x per operator application on wg-square-1e5) compared runs with
different restart counts; at equal work (`bench_ortho_fix_steps`, tol = 1e-300, 3 restarts = 214
applications for every build, per-round paired ratios, base / version) [b622d6e]:

| case | 4373b71 | final branch | total time (4373b71 / final) |
|---|---|---|---|
| wg-square-1e5 GHIEP (3 rounds) | 1.13x (1.20x) | 1.32x (1.31x) | 0.99x / 1.09x |
| cav2d-rect-pert-1e5 GHEP (3) | 1.01x (1.04x) | 1.05x (1.08x) | 0.99x / 1.00x |
| syn-ghep-zero-cluster-complex-1e4 GHEP (5) | 1.04x (1.02x) | 1.12x (1.09x) | 1.02x / 1.04x |

(in parentheses: normalized by each run's operator time). Loss of orthogonality of the returned
bases: base 1e-14 .. 2.1e-14, final 9.1e-15 .. 1.5e-14.

`OrthoOptions::mw_on_exit = false` (one B application less per step) gave no detectable effect on
cav2d-rect-pert-1e5 (B 5 nnz/row) and an incomparable path on yee-pec-box-1e5 [00a3f98]; the default
stays. Correctness: with DGKS reading a scratch bw in stored mode, store_bv Lanczos had returned
max |X^H M X - I| = 12.1; after Reorth::Auto all 48 store_bv probe runs stay <= 1.7e-14 [545b321].

## 3. certpar - certificate inertia counts, InertiaCounter

Method: `bench_certpar_inertia` (windows and orderings in its header), 2 threads on cores 36,37, load
~41, before/after timings medians of 5 interleaved repetitions.

Baseline hsolve at 2 threads, certificate share = total factorize time minus the main factorization:
syn-mem27-46 total 9.66 s, certificate ~3.57 s (37%); yee-pec-cube-32 9.83 s, ~2.86 s (29%);
ss-qa8fk-qa8fm 6.55 s, ~1.50 s (23%).

**ICNTL(31) = 1** (same pivot order for all variants; endpoint pair time and INFOG(21), keep ->
discard): syn-mem27-46 3.19 -> 2.75 s, 650 -> 256 MB; yee-pec-cube-32 2.51 -> 2.25 s, 546 -> 216 MB;
ss-qa8fk-qa8fm 1.31 -> 1.09 s, 321 -> 138 MB; wg-square-1e5 0.31 -> 0.24 s, 89 -> 35 MB;
syn-near-singular-complex-1e5 1.14 -> 1.20 s, 124 -> 73 MB. Set at factorization instead of analysis:
no change in factors or memory.

**Given pivot order** (analysis fresh -> with PERM_IN): SCOTCH cases 1.34 -> 0.05, 1.02 -> 0.04,
0.93 -> 0.03 s; AMD cases 0.043 -> 0.035 s (wg), 0.26-0.30 -> 0.38 s (complex). Factor entries with the
given order +0.4% to +3.4%.

**Concurrency**: without the lock 10 of 10 runs crashed (rc -11, "PB allocation in MUMPS_LOAD_INIT");
with it 2 x 50 rounds were consistent. The block-diagonal pair diag(K(b), -K(a)) took 2.77 / 2.20 /
1.13 / 0.26 s vs 2.73 / 2.16 / 1.03 / 0.25 s for two sequential counts, with 394 vs 254 MB. In the
ASan debug preset std::mutex was a no-op (3 of 3 crashes); the pthread lock passes 5 of 5 [7ecbe1d].

**Single precision** (pair time double -> single, factors discarded): 2.88 -> 1.66, 2.17 -> 1.25,
1.12 -> 0.68, 0.25 -> 0.21, complex 1.18 -> 1.08 s. Largest distance at which a count was wrong, in
units of (scale + |s|): 1.6e-6 (syn-mem27), 1.3e-8 (yee), 1.2e-8 (qa8fk), 9e-8 (wg), ~5e-6 (complex:
wrong at nearly every shift). The 3-D certificate endpoints lie at least 2.8e-4 from every eigenvalue.

**InertiaCounter** (certificate pair with the main factors resident; driver path = refactorize the
main object at both endpoints) [590c143, factor.hpp]:

| case (ordering) | driver pair | counter: analysis + pair | single pair |
|---|---|---|---|
| syn-mem27-46 (SCOTCH) | 3.85 s | 0.06 + 3.27 s (-13%) | 1.82 s |
| yee-pec-cube-32 (SCOTCH) | 3.00 | 0.05 + 2.59 (-12%) | 1.40 |
| ss-qa8fk-qa8fm (SCOTCH, GHEP) | 1.57 | 0.04 + 1.30 (-15%) | 0.74 |
| wg-square-1e5 (AMD, 2-D) | 0.31 | 0.04 + 0.26 (-2%) | 0.22 |
| syn-near-singular-complex-1e5 (AMD) | 1.03 | 0.48 + 0.98 (+42%) | 0.90 (wrong count) |

Working memory INFOG(21) of a count 289 / 213 / 128 / 35 / 73 MB vs 683 / 543 / 326 / 88 / 124 MB of
the main factors; RSS +228 / +168 / +79 / +9 / +166 MB with the main factors resident, no growth after
`release_factors()`. Not integrated in the driver yet (P4).

**Missed null pivots** (`bench_certpar_fix_probes --exp nullpivot`, shifts exactly on 24 eigenvalues
of 2-D Dirichlet Laplacians, AMD, CNTL(3) = 1e-8) [850fbea]: 100x101 real source 0/24 missed, counter
1/24; complex 0/24 vs 1/24; 200x201 real 1/24 each, at different shifts. Every count reported valid lay
within [#below, #below + multiplicity].

## 4. blockks - block Krylov-Schur

**Shift-invert path** (`bench_blockks_shiftinvert --reps 5 --ncv 0,64 syn-mem27-46 yee-pec-cube-32
ss-qa8fk-qa8fm`, cores 24,25, 2 threads, MUMPS SCOTCH at the case sigma, nev = case k, tol 1e-11,
B inner product, repetitions interleaved; load 38-44, absolute times varied up to ~2x between runs;
peak RSS 0.75 GiB) [72195c9]. Auto ncv; per p = 1 / 2 / 4 / 8:

| case | iteration s | MUMPS solve s | applications | restarts | speedup vs p = 1 |
|---|---|---|---|---|---|
| syn-mem27-46 (n = 97336, k = 12, sigma = 0) | 3.45 / 1.66 / 1.24 / 0.96 | 3.36 / 1.57 / 1.15 / 0.85 | 36 / 34 / 40 / 48 | 1 / 0 / 0 / 0 | 2.08 / 2.78 / 3.58x (best of 5: 1.85 / 2.31 / 2.89x) |
| yee-pec-cube-32 (n = 92256, k = 12) | 5.06 / 3.00 / 2.51 / 1.83 | 4.89 / 2.78 / 2.26 / 1.58 | 68 / 70 / 88 / 96 | 5 / 3 / 3 / 1 | 1.69 / 2.01 / 2.77x (best of 5: 1.78 / 2.08 / 2.84x) |
| ss-qa8fk-qa8fm (n = 66127, k = 10, GHEP) | 2.77 / 2.40 / 2.25 / 2.50 | 2.46 / 1.93 / 1.75 / 1.87 | 50 / 68 / 100 / 160 | 3 / 3 / 3 / 3 | 1.15 / 1.23 / 1.11x |

Operator application per vector (one multi-RHS MUMPS solve plus B spmm), speedup vs p = 1 for
p = 2 / 4 / 8: mem27 1.61 / 2.60 / 4.56x, cube 1.76 / 2.90 / 4.42x, qa8fk 1.72 / 2.82 / 4.56x (earlier
runs 1.6-1.9 / 2.5-2.8 / 4.2-5.7x). Orthogonalization 0.07-0.51 s. Eigenvalues match p = 1 to <=
3.3e-12 (mem27), 6.9e-14 (cube), 2.4e-13 (qa8fk) relative; both cubic cases have two 6-fold
eigenvalues among the 12 wanted, every p found all copies (p = 1 through rounding). First-cycle probe,
p = 8, auto ncv (64, probes from 32) vs ncv = 64 requested: mem27 48 vs 64 applications, 0.96 vs
1.35 s; cube 96 vs 96, 1.83 vs 1.92 s; qa8fk 160 vs 160, 2.50 vs 2.44 s. With p = 4 on syn-mem27-46
the application count (40 or 44) is decided by rounding together with the non-reproducible
factorization [2830207].

Negative results: with the single-vector auto ncv (32 for p = 8), p = 8 was 0.66x on ss-qa8fk-qa8fm
(272 applications vs 50): this motivated the +4 blocks rule. Without probes, p = 8 at ncv 40 / 48 /
64 / 96 vs p = 1 auto: mem27 2.28 / 3.35 / 1.91 / 1.50x, cube 2.28 / 2.06 / 2.25 / 2.38x, qa8fk 0.79 /
0.94 / 1.00 / 1.13x.

**Layout rules** (scratch sweep: random shift-invert pencils, B inner product, LargestMagnitude,
tol 1e-11, ncv = nev + 3, n = 150/250, nev 3/5/7/12, 3 shifts, 10 seeds, real and complex = 480 runs
per p) [9d6d94b]: wrong converged sets for p = 1 / 2 / 3 / 4 / 8 went 1 / 5 / 51 / 44 / 5 -> 1 / 1 / 1
/ 0 / 0 with the spare Ritz pair and two new blocks, with 40-66% fewer applications for p > 1.

**Matched accuracy** (`bench_blockks_fix_matched --reps 3`, cores 24,25, 2 threads, auto ncv, 30
nearest HEP/GHEP cases; p = 1 at tol 1e-9/1e-10/1e-11, p = 2/4/8 at 1e-11..1e-14; "matched" = the
loosest block tol whose median largest relative residual is within 2x of p = 1's at 1e-11) [2830207].
Speedup vs p = 1, equal tol / matched ("-" = no block tol down to 1e-14 reaches it):

| case | k | p = 1 s | p = 2 | p = 4 | p = 8 |
|---|---|---|---|---|---|
| cav2d-square-1e4 | 7 | 0.05 | 1.23 / 1.23 | 0.98 / 0.98 | 0.65 / 0.65 |
| syn-multiplicity-real-1e4 | 14 | 0.16 | 1.61 / 1.70 | 1.82 / 1.76 | 1.79 / 1.66 |
| syn-multiplicity-cplx-1e4 | 14 | 0.44 | 1.72 / 1.42 | 1.97 / 1.97 | 1.95 / - |
| syn-near-singular-real-1e4 | 10 | 0.07 | 1.16 / 1.04 | 0.85 / 0.85 | 0.62 / 0.62 |
| syn-near-singular-cplx-1e4 | 10 | 0.18 | 1.02 / - | 0.86 / - | 0.62 / - |
| syn-wide-range-real-1e4 | 10 | 0.05 | 1.19 / - | 0.91 / 0.81 | 0.62 / 0.51 |
| syn-wide-range-cplx-1e4 | 10 | 0.15 | 1.09 / 0.97 | 0.97 / 0.82 | 0.70 / - |
| syn-ghep-multiplicity-1e4 | 22 | 0.23 | 1.31 / 1.31 | 1.66 / 1.66 | 1.50 / 1.50 |
| yee-bloch-1e4 | 10 | 1.21 | 1.77 / 1.77 | 1.82 / 1.66 | 1.72 / 1.58 |
| yee-pec-box-1e4 | 11 | 0.26 | 1.77 / 1.53 | 1.47 / 1.23 | 1.12 / - |
| yee-pec-cube-16 | 12 | 0.35 | 1.59 / 1.59 | 1.65 / 1.65 | 2.33 / 2.33 |
| yee-photonic-rods-1e4 | 10 | 0.82 | 1.41 / 1.32 | 1.52 / 1.30 | 1.29 / 1.21 |
| yee-photonic-spheres-1e4 | 10 | 0.90 | 1.27 / 1.25 | 1.24 / 1.24 | 1.29 / 1.15 |
| syn-lap2d-100 | 12 | 0.12 | 1.62 / 1.57 | 1.47 / 1.34 | 1.11 / 1.05 |
| syn-mem27-46 | 12 | 2.96 | 1.82 / 1.82 | 2.23 / 2.23 | 2.88 / 2.88 |
| yee-pec-cube-32 | 12 | 5.03 | 1.80 / 1.88 | 2.13 / 1.97 | 2.83 / 2.83 |
| ss-qa8fk-qa8fm | 10 | 2.67 | 1.20 / - | 1.30 / 1.30 | 1.14 / 1.16 |
| cav2d-rect-pert-1e5 | 6 | 0.59 | 0.98 / 0.94 | 0.83 / 0.77 | 0.67 / - |
| ss-helm3d01 | 10 | 0.85 | 1.27 / 1.27 | 1.27 / 1.27 | 1.11 / 1.11 |
| ss-tem27623 | 10 | 0.78 | 1.16 / - | 1.04 / - | 0.95 / - |
| yee-pec-box-1e5 | 10 | 6.69 | 1.80 / 2.02 | 1.72 / 1.83 | 2.08 / 1.99 |
| syn-lap3d-22 | 12 | 0.19 | 1.80 / 1.80 | 1.70 / 1.70 | 1.94 / 1.94 |
| syn-lap3d-aniso-1e5 | 12 | 3.40 | 1.41 / 1.41 | 1.26 / 1.26 | 1.13 / 1.13 |
| syn-mem125-46 | 15 | 26.21 | 2.00 / 2.00 | 2.21 / 2.21 | 3.17 / 3.17 |
| ss-si34h36 | 10 | 28.29 | 1.53 / 1.53 | 1.87 / 1.87 | 2.27 / 2.27 |
| ss-2cubes_sphere | 10 | 3.49 | 1.21 / 1.09 | 1.20 / 1.20 | 1.02 / 0.91 |
| syn-ghep-near-sing-real-1e5 | 10 | 1.20 | 0.95 / 0.95 | 0.81 / 0.81 | 0.73 / 0.65 |
| ss-ge87h76 | 10 | 29.32 | 1.40 / 1.40 | 1.39 / 1.39 | 1.81 / 1.81 |
| ss-tem152078 | 10 | 5.13 | 1.15 / - | 1.11 / - | 1.07 / - |
| syn-near-singular-cplx-1e5 | 10 | 2.60 | 1.22 / 1.22 | 1.05 / - | 0.90 / - |

Geometric means, matched (cases matched): p = 2 1.41x (25), p = 4 1.36x (26), p = 8 1.37x (22); equal
tol 1.39x / 1.35x / 1.27x over 30. Slower than 0.9x when matched: p = 2 none, p = 4 5, p = 8 4 cases.
Not matched at any tol: p = 2 5, p = 4 4, p = 8 8 cases (higher residual floors, e.g.
syn-near-singular-complex-1e4: residuals p = 1 / 2 / 8 about 2-5e-10 / 0.5-2.4e-9 / 0.2-5e-8 at every
tol; not caused by the Cholesky-QR condition limit). Conclusion: p = 1 stays the driver default; block
mode is meant for deflated reruns with p >= the missing copies; a default change needs an end-to-end
driver measurement.

## 5. lobpcg_jd - Jacobi-Davidson and LOBPCG

**Final defaults** (`bench_lobpcg_jd_memory`, 2 threads, nev = 12, tol = 1e-10, 5 runs each with a
fresh, non-deterministic MUMPS factor; min / median / max of preconditioner solves and of the total
time incl. analysis and factorization) [29e6bcb, eigen_iterative.hpp]:

| case | method | solves | total s | factor | peak |
|---|---|---|---|---|---|
| syn-mem27-46 (tau = 0 lies 3e-6 and 5e-6 from two 6-fold eigenvalues, ‖A‖ = 3.6) | JD + single factor | 123 / 133 / 141 | 7.5 / 8.0 / 9.6 | 337 MB | 0.55 GiB |
| | shift-invert Lanczos, double factor | 36 | 5.2 / 5.4 / 6.5 | 651 MB | 0.70 GiB |
| | single factor + refinement (driver's fallback) | 303 / 396 / 396 | 13.7 / 19.0 / 25.0 | 337 MB | 0.46 GiB; 2 of 5 runs not accepted (eta 2e-8, 2e-5) |
| | former JD defaults (harmonic, no start-block iteration) | 312 / 665 / 2170 | 18 / 31 / 117 | | |
| yee-pec-cube-32 (tau = 53.16 between two 6-fold eigenvalues) | JD + single factor | 196 / 200 / 206 | 8.8 / 9.0 / 11.1 | 282 MB | 0.45 GiB |
| | shift-invert Lanczos, double factor | 68 | 5.8 / 6.1 / 7.7 | 548 MB | 0.57 GiB |
| | single factor + refinement | 300 / 381 / 613 | 12.1 / 15.1 / 24.6 | | 0.36 GiB; 1 of 5 runs missed a copy of a 6-fold eigenvalue |
| | former JD defaults | 207 / 214 / 230 | 9.9 / 10.2 / 12.2 | | |

yee-pec-cube-32 with tau 9.5e-9 and 2.5e-11 from the 6-fold eigenvalue 49.21342550952...
(eps_s cond(A - tau B) > 1; at 2.5e-11 the double factor is singular, at 9.5e-9 double shift-invert
Lanczos ends at eta 2e-8 with a wrong set; the single factor was reported singular in 2 of 12 runs):
JD 219-259 and 231-333 solves (5 runs each), 10-19 s, all correct; single factor + refinement fails
(eta 7e-2); former JD defaults at 9.5e-9: 608 solves with a wrong set (a copy missing, converged =
true) or 6339 solves / 241 s; at 2.5e-11 no convergence in 1000 iterations (7230 solves, 323 s).

**Options** (`bench_lobpcg_jd_fix_options`, 5 fresh-factor runs, solves min/med/max) [29e6bcb]:
syn-mem27-46 Ritz 129/135/136 vs harmonic 147/165/288 (432/916/1882 without the start-block
iteration); yee-pec-cube-32 196/200/206 vs 194/201/218. **Synthetic** (`bench_lobpcg_jd_fix_synthetic`,
6/8/10-fold eigenvalue, nev = 6): 24 of 24 preconditioned runs returned 4-5 copies with converged = true
without the start-block iteration, 0 of 24 with it, 36% fewer solves in total; unpreconditioned runs
still lose copies (28 of 32): the driver's certificate and a deflated rerun recover them.

**Implementation round** (same bench, cores 30,33, ordering 3, k = 12, tol 1e-10, single runs; JD
still with the former defaults, the si-* and driver-* rows are unaffected by the later JD changes):

| case | mode | factor s | iterate s | total s | factor MB | peak GiB | solves | max eta |
|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 | si-double | 2.99 | 2.21 | 5.23 | 645 | 0.686 | 36 | 7.7e-13 |
| | si-single (refinement) | 2.42 | 16.45 | 18.89 | 343 | 0.462 | 396 | 1.0e-11 |
| | driver-double | | | 8.23 | | 0.755 | | |
| | driver-single | | | 26.23 | | 1.058 | | |
| yee-pec-cube-32 (sigma = 53.158) | si-double | 2.32 | 3.51 | 5.85 | 551 | 0.569 | 68 | 5.2e-13 |
| | si-single | 1.87 | 21.41 | 23.29 | 285 | 0.373 | 529 | |
| | driver-double | | | 8.70 | | 0.617 | | |
| | driver-single | | | 18.17 | | 0.873 | | |

In single-precision mode the certificate's double `compute_inertia` factorization raises the peak above
the double path (0.873 vs 0.617 GiB, 1.058 vs 0.755 GiB). Close shift yee-pec-cube-32 sigma = 49.2134
(2.5e-5 from a 6-fold eigenvalue, eps_s cond ~ 30): si-single fails (eta 0.11, refinement diverges);
driver-single 33.1 s, 0.858 GiB, 6 of 12 pairs, certificate failed; raw si-double eta 2.9e-9 before the
driver's Rayleigh-Ritz / rescue; JD with a single factor (former defaults) 30.1 s, 0.438 GiB, 585
solves, correct set, eta 7.2e-11.

**LOBPCG, factorization-free path** (syn-mem27-46, the k smallest): plain lobpcg 15.1 s / 0.16 GiB;
lobpcg preconditioned by a single-precision factor 8.7 s / 0.54 GiB; shift-invert Lanczos with a double
factor 12.7 s / 0.69 GiB.

## 6. slicing - windows, one-sided targets, zero cluster

Method: `bench_slicing_sweep` and the hsolve CLI (`--slice-size`, `--zero-filter-tol`), cores 40,41,
2 threads, `--mem-gb 8`; correctness against the case references.

**Interval slicing** (all certificates passed; multiplicity-2 spectra):
- yee-pec-box-1e4 [1000, 2000], 1075 eigenvalues. Final binary, same session: single shift (old
  behaviour) 49.8 s, 0.89 GiB; auto (19 slices) 11.1 s, 0.20 GiB, 40 factorizations, 2964 solves;
  max relative error vs the exact reference 2.2e-15. Earlier sweep [bec16ab]: single 70.4 s; slice
  sizes 16 / 32 / 64 / 128 / 256: 29.3 / 17.8 / 11.8 / 13.1 / 17.5 s.
- yee-pec-box-1e4 [1, 2000], 1461 eigenvalues: 26 slices, 15.1 s, 0.23 GiB, exact.
- yee-pec-box-1e5 [382, 1209], 499 eigenvalues. Session 1: single shift 213 s; sizes 32 / 64 / 128 /
  256: 218 / 155 / 150 / 179 s; the first model's auto choice (3 slices) 180.6 s. After recalibration
  (session 2, more load) [12d560f]: auto (4 slices) 230.1 s vs best fixed size 128 235.1 s; error
  3.5e-15. yee-pec-box-1e4 [1000, 2000] then: auto 13.4-14.5 s vs fixed 64 / 96 13.8 / 14.0 s.

**Smallest / largest**:
- ss-qa8fk-qa8fm (GHEP; Unsupported before): smallest 10 in 11.0 s (7 factorizations, certificate
  16/16); largest 10 in 14.4 s (certificate 17/17). Illinois count search: smallest 10 12.35 -> 10.98 s,
  79 -> 69 Lanczos applications [12d560f].
- syn-lap2d-100: smallest 10 0.13 s (1 factorization, 58 solves) vs the legacy path (nearest at 5% of
  the spread beyond the end) 0.47 s (3 factorizations, 242 solves); largest 10 0.18 vs 0.45 s.
- syn-lap3d-aniso-1e5 smallest 10: 10.8 s (4 factorizations, 79 solves) vs legacy 17.8 s (3, 210).
- yee-pec-box-1e4 smallest 20 with zero_filter_tol 1e-6: exact, 3135 null-space eigenvalues excluded by
  inertia, 0.9 s.
- Review fixes [722965d, 38c23e5]: yee-pec-cube-16 largest 3: 0 pairs / 65 s -> 3 exact pairs in
  0.87 s; smallest 6 / 12 with zero filter: eta 1.3e-9 / 6.5e-9 and 28 / 26 factorizations ->
  7.7e-13 / 1.2e-13 and 16 / 14; PSD matrix with a 600-dim null space, smallest 5: 19 s and 600 pairs
  -> 5 null vectors in 0.06 s; yee-pec-box-1e4 smallest 10 (multiplet rerun shift at 1% of its gap
  instead of on it): 215 -> 87 solves, 1.37 -> 1.03 s.

**Nearest with a zero band** [d0012f7, 82811df]: yee-pec-box-1e4, sigma = 1, k = 6, zero_filter_tol
1e-6: returned 1 of 6 pairs before; now 6/6 exact (1.5e-15 vs the discrete-exact reference),
certificate 6/6 with 3135 null-space eigenvalues excluded by inertia, 5 factorizations, 68 solves,
0.88 s. Radius mid-gap after the k-th Ritz value: far pairs eta 6.9e-11 -> 7.5e-14 / 1.1e-13.

**Near-singular shift** (hsolve CLI, case default config) [e30d029]: syn-zero-cluster-real-1e5 base
32.9 s, 10 factorizations, 549 solves -> 2.6-3.5 s, 6 factorizations, 65 solves (8.7e-17 vs the exact
reference); syn-near-singular-{real,complex}-1e4, -complex-1e5 and syn-ghep-near-singular-real-1e5
unchanged (exact, 4-5 factorizations). syn-zero-cluster-real-1e4: 15 -> 8 factorizations [d0012f7],
then 7 with gap endpoints counted in place, no certificate rerun [4f90f9c].

**GHIEP zero cluster**, wg-square-1e5 (scale 333, dim null(A) = 24649) [3b7a4b6, 680d533]:
- sigma = 1e-4, k = 10: base returned 10 zeros; with side shifts at the plain run's disk edge 514
  solves, 14.4 s; mid-gap shifts: the 10 nearest nonzero modes, eta <= 1e-15, 256 solves, 7.7 s; with
  zero_filter_tol 1e-6 221 solves, 6.6 s.
- sigma = 0, k = 10 (no plain run): 238 solves, 7.7 s. sigma = -0.5, k = 22: base 18 modes + 4 zeros;
  now 18 propagating + 4 evanescent modes, 253 solves, 8.0 s.
- The two case targets (sigma = -0.527 / 0.311) stay on the plain path (2 factorizations).
  wg-square-1e3, sigma = 0, k = 10: equals the dense reference (5.8e-14), cluster dimension 225.
- Review fixes [376f27a]: wg-square-1e3 (sigma 0.01, k 40, zero_filter_tol 1e-9) eta 6.5e-11 .. 1.8e-10
  -> <= 8.5e-13 in five runs; a complex pair 0.02 +- 0.1i between the side shifts (pencil with 30 zeros,
  sigma = 1e-3, k = 4) is now found (before: four real pairs, success = true).

## 7. pybind - Python API overhead

Method: guard `--cores 38,39` (2 cores), release build.

**Conversion** (`bench_pybind_convert 100 5`: 3-D 7-point Laplacian, n = 1e6, nnz = 6.94e6, median of
5; floor = any owning CsrMatrix copy (allocate, zero-fill, memcpy, widen indptr) 0.036 s): csr
int32/f64 0.048 s (1.34x floor); csc, no transpose 0.049 s (1.36x); csr int64 0.050 s (1.38x); csr
float32 0.049 s (1.36x); complex128 with zero imaginary parts 0.063 s (1.77x); csr with every row
reversed 0.105 s (2.94x); coo row-ordered 0.064 s (1.79x); coo shuffled 0.290 s (8.09x); upper triangle +
mirroring 0.172 s; dense 2744^2 scan 0.0086 s. Negative result: shuffled COO is 13% slower than
`hs::csr_from_coo` (0.257 s); a pair array would save ~65 ms but costs 16 extra bytes per entry at
peak, so the memory-lean version was kept. Insertion sort for short rows: 0.118 -> 0.105 s (reversed
rows), 0.317 -> 0.290 s (shuffled COO).

**End to end** (`python/bench/overhead.py CASE --mode memory|path` vs the CLI, two repeats; overhead =
wall time of `solve()` minus the solver's `time_total_s`; peak CLI / memory / path):

| case | CLI s | in-memory overhead s | path overhead s | peak GiB |
|---|---|---|---|---|
| cav2d-rect-pert-1e5 (GHEP, nnz 0.50M) | 0.97 / 0.82 | 0.006-0.013 | 0.056-0.066 | 0.12 / 0.18 / 0.17 |
| wg-square-1e5 (GHIEP) | 1.97 / 1.83 | 0.010-0.019 | 0.049-0.059 | 0.28 / 0.36 / 0.34 |
| syn-near-singular-complex-1e5 (nnz 2.4M) | 5.08 / 4.67 | 0.032-0.041 | 0.316-0.344 | 0.40 / 0.50 / 0.46 |
| yee-pec-box-1e5 (GHEP, nnz 1.39M) | 9.34 / 9.16 | 0.038-0.046 | 0.177-0.191 | 0.71 / 0.78 / 0.76 |

In-memory input costs 0.4-1.6% of the solve; path input costs the npz loading, which the CLI pays
too. Peak difference: ~0.07 GiB for the interpreter with numpy/scipy, plus scipy's own copy of the
matrix for in-memory input. Repeats agree to within 1e-13. GIL check: with the GIL release removed a
background thread ticked once during a 0.19 s solve (>= 16 required); with it the test passes.

**load_matrix memory** (`python/bench/load_memory.py`, RSS growth after loading, 2 runs) [08b8620]:
yee-pec-box-1e5 A 30.7 -> 17.0 / 18.2 MiB, syn-near-singular-complex-1e5 A 66.8 / 68.2 -> 46.9 MiB,
wg-square-1e6 A 80.6 -> 48.3 / 47.2 MiB; `scipy.sparse.load_npz` 18.6, 47.0, 48.4-49.5 MiB.

## 8. eval - independent evaluation

Method: `tools/run/guard.py` on cores 28,29, mem <= 8 GiB; `python -m tools.eval.sweep`.

**Verdicts**:
- run2 (`results/eval_run2.md`): hsolve 39 pass / 0 fail. Completeness: 25 complete against a
  certified reference (23 discrete-exact, 2 dense); 11 consistent against an ARPACK solver reference;
  3 unknown (no reference: ss-2cubes_sphere, ss-ge87h76, ss-si34h36). Max backward error 4.2e-11;
  0 format issues; 0 overclaims.
- cmp4b (`results/eval_cmp4b.md`): hsolve 15 pass / 0 fail (7 complete, 8 consistent: 7 solver
  references, 1 pooled), max backward error 9.5e-13. SLEPc 10 pass / 5 fail, 1 overclaim:
  ss-2cubes_sphere 5 of 10 pairs at backward error ~1.3e-5; wg-halfslab-1e5 2 of 28 pairs at 1.8e-10;
  yee-pec-box-1e4 misses one copy of the double eigenvalue 107.69 and returns 118.69 instead while
  claiming success (the overclaim); yee-pec-box-1e5 misses one copy and has 1 pair above tol;
  yee-pec-cube-32 misses one copy and has 7 of 12 pairs above tol (worst 2.2e-7). ss-tem27623: SLEPc
  passes on backward error (2.2e-13) but its relative residual reaches 6.8e-5, eigenvalue error 4.2e-7
  and subspace angle 5.1e-3; hsolve 4.1e-13, 9.3e-16, 1.0e-12.
- scipy baseline (18 real cases): 18 pass (13 complete, 5 consistent); max backward error 5.0e-15;
  solve times 0.016 to 12.8 s. On complex Hermitian cases its vectors are not orthogonal inside
  degenerate eigenspaces (B-orth 0.82 on yee-bloch-1e4).
- The committed pages' time column is `time_total_s`, whose scope differs between hsolve and SLEPc;
  the evaluator now reports a like-for-like `solve_time_s` [4ee52ce] (e.g. SLEPc syn-ghep-near-singular
  2.02 -> 1.46 s, cav2d-rect-pert 1.04 -> 0.842 s), and the pages are to be regenerated with it.

**Norm estimate** (`tools/eval/bench_norms.py`, ARPACK eigsh k = 1 'LM', tol 1e-6 / ncv 40 vs tol 1e-8
/ default ncv, same start vector, five ~1e5 cases) [d9533d3, 4c4a589]: agreement <= 1.4e-10 relative;
products 4901 -> 1121 on syn-zero-cluster-real-1e5 (wall 55.7 -> 14.1 s), 1501 -> 601 on wg-square-1e5
A (7.4 -> 7.9 s: a restart with ncv 40 costs more), 1651 -> 601 on wg-halfslab-1e5 A; one increase, a
diagonal B 21 -> 41 products; 93.6 -> 49.3 s summed over the nine matrices.

**Sweep wall time**: cmp4b 30.5 s and 54.7 s in two runs (the former per-result evaluator took
116 s); run2 87.9 to 123.9 s.

**Negative result**: SuperLU (MMD_AT_PLUS_A) had not finished factorizing ss-2cubes_sphere after
600 s at 4.7 GiB (killed by the guard), so ARPACK + SuperLU references are out of reach for the 3-D
SuiteSparse cases (Si34H36 and Ge87H76 have 3.1x and 4.8x its nonzeros).
