# P9 "ordering": fewer flops through hsolve's own nested dissection

P9 asked whether the fill-reducing ordering can cut the three numeric LDL^T factorizations that
dominate an in-core n = 1e6 run (the main factorization and the exact certificate's two inertia
counts on the same pivot order), and to make the adaptive ordering pick the fastest ordering for the
whole run at 1e6 and at 1e5. Branch `p9-ordering` from main a831d79. Every timing is from the gpu02
measurement slot (`guard.py --exclusive`, cores 16-31, 16 threads, 32 GiB), 2026-10-14, unless it
says "build lane" (cores 8-15, 8 threads, analysis estimates only - no timing conclusions).

**Result.** hsolve's nested dissection (ND) now gives the smallest factors of every ordering tried
on the 3-D data cases and is the adaptive ordering's preferred candidate:

* a **bug fix**: the trial bisections' `taskwait` also waited for the sibling half spawned from the
  same task - the 16-thread dissection of a 1e6 graph took 7.4 s for 18 s of work, now 2.4 s
  (with everything below);
* **flow refinement** of every separator (minimum vertex cut in a band of 2 BFS layers, max flow):
  FM stops in local minima on meshes; the cut finds the plane separators. MUMPS flops
  yee-pec-box-1e6 7.39e12 -> 5.43e12 (SCOTCH 6.75-7.09e12, METIS 7.05e12, a geometric dissection
  5.29e12), yee-pec-box-1e5 -26%, ss-2cubes_sphere -16%, ss-ge87h76 -16%; -10% geomean over 19
  cases. **Review fix**: the driver skips it where the planned numeric time is under 4.5x the
  ordering time - on the smallest 3-D cases fewer flops did not mean less time (yee-pec-cube-16:
  -17% flops, +9% whole run, systematic; section 8.3);
* **graph compression** (indistinguishable vertices, as METIS and SCOTCH do; leaves by vertex
  weight): ss-tem152078 ordering 0.70 -> 0.26 s - the cause of the P5c regression -, the block
  syn-* cases 5-10x faster to order;
* concurrent contraction / splitting of large graphs and, chosen by the cost model where the
  numeric work dwarfs the ordering, more effort on the top two levels' separators (ss-ge87h76
  -12% flops);
* the cost model refit on the new ND, the ND before SCOTCH among the candidates.

Whole run, final code (gate, section 8; slot, ABBA, every run valid, exact certificate):
**syn-mem27-100 111.0 -> 89.0 s (1.25x), yee-pec-box-1e6 89.7 -> 73.7 s (1.22x)** (5 runs each;
the draft: 1.27x / 1.21x); n <= 1e5: **geometric mean 0.919** over 26 data cases (3 runs each;
faster by > 3% on 18, 0.76-0.97; no case slower beyond noise in the 10-30-run re-checks except
wg-square-n289, n = 289, +1.6 ms on 18 ms with main's own pivot order - section 8.2). The draft's
one regression, yee-pec-cube-16 (+9%, systematic), is gone (1.015 over 20 runs, main's order).

## 1. Setup

Code: branch `p9-ordering` (commits in order): 48e23b5 bench_ordering_cost; 0daf807 taskgroup;
7c3d269 / 2c4594b compression, finalists, effort, `FactorOptions::nested_dissection`; 537ff8e
concurrent contraction; 6d96a08 flow refinement; edb3909 leaves by weight; 10be12a cost model and
candidate order (section 6); 4aa74bf / 2c9aa2a formatting, bench header. Review fixes (branch
`p9-ordering-fix`): be69e74 `bench_ordering_cost --mumps-verbosity`; 0b2da5a the flow rule (section
6, 8.3); 50d8351 the concurrent-split test; then documents. Binaries: `tools/dev/build.sh release`
on gpu02 (build lane), copies in `~/p9_p9-ordering/bin/`, the review fix's scripts and raw data in
`~/p9_p9-ordering/fix/`:

| label | commit | md5 | used for |
|---|---|---|---|
| main | a831d79 | b9d1e1657ad05d10c5e1aa1565c7f649 (`hsolve`) | gate baseline |
| v0 | 48e23b5 (main + bench) | 655a5963c96ecb3fc91e10efed776f6c | m1 (1e6, MUMPS orderings, P5c ND) |
| v1 | 0daf807 | de6296303e1149726ef4a77e3ed2140b | m2 (1e5, MUMPS orderings, ND with the taskgroup fix) |
| v5 | 2c4594b + internal options | ed9040ce621608fe3af1e6efb719045f | m3a / m3b (effort, compression) |
| v12 | 6d96a08 (flow default) | 0c1a5be730041a465407f2bec31871e1 | m6 (1e6), m7 (1e5), m5 fit data |
| v14 | edb3909 | dd6b2c729352c4ced41b6ea009d24c4c (`bench_anaint_orderings`) | m8 fit data |
| v15 | 10be12a's model code before the refit constants | 139cff34c4887338c2becdea920514b2 (`bench_anaint_fix_fit`) | the refit (section 6) |
| v16 = new | 10be12a | a6fd066291418828da0483a9930468a4 (`hsolve`) | the P9 draft's gate (section 8: G5, G6) |
| fix0 | ac022a6 + `--mumps-verbosity` (be69e74) | abc0a6499cdbf6709ce4a243b1c7bff4 (`bench_ordering_cost`) | F1 (flow on / off, section 8.3) |
| fix1 | 0b2da5a (review fix: the flow rule) | 3f2cd9bab4960df8e622c9a8e0d580e7 (`hsolve`), 26c252064631adccbb47565e65a15dd5 (`bench_anaint_decide`), d9fdfb1ad3b701e27b3f3622bed4f100 (`bench_anaint_fix_fit`) | the gate of the final code (section 8: G7, G8), choices, replay |

Benches: `bench/ordering_cost.cpp` (new: analysis with the ordering timed apart, MUMPS estimates,
main factorization, InertiaCounter counts after releasing the main factors - the certificate's
path at this size -, 1/4/8-RHS solves; every ND option; `--perm-in` for external orders; run with
the BLAS hook preloaded, `LD_PRELOAD=build/release/libhsolve_blashook.so`, so that solves time as
in the CLI) and `bench/anaint_orderings.cpp` (the fit data, as in P5c: no preload). "Run model" in
the tables = analysis + factorization + counter setup + 2 counts + (40 + 4k)/4 four-RHS solves
(`plan_work` of a nearest target, P4 block width).

Campaigns (slot; one outer `guard.py --exclusive --timeout <= 2700` per part, one nested guard per
measurement, `~/p9_p9-ordering/{run_plan,run_cmds,chain}.sh`):

| campaign | time | content |
|---|---|---|
| m1 | 09:21-09:32 | 1e6: SCOTCH, METIS, P5c ND, PORD (+ AMD on wg-square-1e6), 1 rep, v0 |
| m2 | 09:33-09:58 | 10 1e5 3-D cases x AMD/SCOTCH/METIS/PORD/ND, 3 interleaved reps, v1 |
| m3a / m3b | 10:03-10:14 / 10:40-10:56 | ND with and without top effort (no flow), 1e5 and 1e6, 2-3 reps, v5 (m3b stopped after 2 reps) |
| m4 | 10:14-10:39 | fit data (first ND, superseded) and the 1e6 cases' MUMPS orderings for the fit |
| m6 | 10:55-11:15 | 1e6 final ND with / without effort, 3 reps, v12 |
| m5 | 11:07-11:21 | fit data: ND 16 threads x 1, 4 threads `--nd-serial 1` x 3 (cores 16-19), v12 |
| m7 | 11:21-11:30 | 1e5 final ND (+ effort on ss-ge87h76, ss-si34h36), 3 reps, v12 |
| m8 | 11:29-11:40 | fit data again with v14 (leaves by weight) |
| G6a / G6b | 11:45-12:04 / 12:04-12:16 | gate at 1e6: main vs new, ABBA, reps 0-2 / 3-4 |
| G5 | 12:16-12:31 | gate at 1e5: 26 cases, main vs new, ABBA, 3 reps |
| G5s | 12:33-12:36 | re-check of 5 small cases, ABBA, 10 reps |
| m9 / m10 | 12:37-12:40 / 12:46-12:50 | yee-pec-cube-16 and other small cases: ND with / without flow, flow only above 5000 vertices, 5 reps |
| | | *review fix (`~/p9_p9-ordering/fix/`)* |
| V1 | 14:14-14:16 | re-check of yee-pec-cube-16, yee-pec-box-1e4, yee-photonic-rods-1e4: main vs new (v16), ABBA, 10 reps |
| F1 | 14:27-14:30 | 11 small ND cases, ND with / without flow (`bench_ordering_cost`, fix0), 5 interleaved reps |
| G7s / G7n | 14:48-14:49 / 14:49-14:50 | 9 small cases main vs fix1; the 4 cases the flow rule changes, v16 vs fix1; ABBA, 10 reps |
| G7 | 14:50-15:04 | gate at 1e5: the 26 cases of G5, main vs fix1, ABBA, 3 reps |
| G8a / G8b | 15:04-15:22 / 15:23-15:35 | gate at 1e6: main vs fix1, ABBA, reps 0-2 / 3-4 |
| G7r / L1 | 15:22-15:23 / 15:35-15:36 | 10-rep re-check (main vs fix1, ABBA) of 7 cases; logged runs (`--log`) of 3 small ND cases, 6 reps |
| G7t / G7u | 15:39 / 15:40 | wg-square-n289 main vs fix1, ABBA, 30 reps: as configured / with AMD forced |

## 2. What an ordering costs at 1e6 (task 1)

16 threads, slot; medians (reps); flops = MUMPS RINFOG(1) (SCOTCH's spread over the reps in
brackets: its multithreaded orderings differ from run to run); in-core GB = MUMPS' estimate of the
in-core factorization (INFOG(17), `est_incore_mb` of `bench_ordering_cost`, analysed without a
memory cap; AMD's from the m4 fit runs, 16 threads; 1 GB = 1000 MB); count = one InertiaCounter
count (main factors released).

| case | ordering | reps | analysis s (ordering) | flops | factor entries | in-core GB | factorize s | count s | 4-RHS solve s | run model s |
|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | AMD | - | 0.75 (P7 logs) | 3.89e13 | 2.45e9 | 41.3 | - | - | - | (numeric not run: 5.7x ND's flops) |
| syn-mem27-100 | SCOTCH | 3 | 6.1-6.3 | 8.89e12 (1.11e13 in m1) | 1.32e9 | 18.6 (18.5-20.3) | 29.6 | 28.3 | 0.715 | 108.7 |
| syn-mem27-100 | METIS | 3 | 14.1 | 6.76e12 | 1.16e9 | 16.1 | 23.8 | 22.3 | 0.624 | 96.9 |
| syn-mem27-100 | PORD | 1 | 5.3 | 9.67e12 | 1.35e9 | 17.2 | 30.9 | 29.2 | 0.708 | 110.9 |
| syn-mem27-100 | ND P5c (m1, taskwait bug) | 1 | 7.99 (7.38) | 6.81e12 | 1.17e9 | 16.2 | 24.0 | 22.4 | 0.627 | 91.3 |
| syn-mem27-100 | **ND P9** | 3 | **3.05 (2.44)** | **6.72e12** | 1.16e9 | 16.1 | **23.5** | **22.1** | 0.628 | **85.2** |
| syn-mem27-100 | ND P9 + effort | 3 | 3.31 (2.71) | 6.75e12 | 1.16e9 | 16.2 | 23.4 | 21.9 | 0.630 | 85.0 |
| syn-mem27-100 | geometric (lattice) | - | - | 6.72e12 | 1.16e9 | - | - | - | - | (reference: coordinates known) |
| yee-pec-box-1e6 | AMD | - | 0.92 (P7 logs) | 6.28e13 | 3.49e9 | 57.3 | - | - | - | - |
| yee-pec-box-1e6 | SCOTCH | 3 | 4.3-4.7 | 7.01e12 (0.97-1.03) | 1.15e9 | 15.9 (15.6-16.0) | 24.0 | 23.1 | 0.629 | 88.1 |
| yee-pec-box-1e6 | METIS | 3 | 13.4 | 7.05e12 | 1.16e9 | 15.8 | 24.7 | 23.1 | 0.635 | 97.7 |
| yee-pec-box-1e6 | PORD | 1 | 7.0 | 1.30e13 | 1.88e9 | 23.5 | 44.6 | 42.2 | 0.984 | 156.3 |
| yee-pec-box-1e6 | ND P5c (no flow, taskgroup) | 1 | 2.56 (1.92) | 7.39e12 | 1.20e9 | 16.6 | 25.7 | 25.0 | 0.657 | 92.0 |
| yee-pec-box-1e6 | **ND P9** | 3 | **2.58 (1.99)** | **5.43e12** | **1.02e9** | **13.9** | **19.4** | **18.2** | **0.560** | **70.2** |
| yee-pec-box-1e6 | ND P9 + effort | 3 | 2.71 (2.15) | 5.45e12 | 1.02e9 | 13.9 | 19.5 | 18.2 | 0.565 | 70.5 |
| yee-pec-box-1e6 | geometric (edge coordinates) | - | - | 5.29e12 | 0.99e9 | - | - | - | - | (reference) |
| wg-square-1e6 | AMD | 3 | 0.54 | 5.47e10 | 8.73e7 | 1.30 | 0.672 | 0.539 | 0.095 | 5.34 |
| wg-square-1e6 | SCOTCH | 2 | 2.6 | 1.82e10 | 6.20e7 | 1.02 (1.02-1.06) | 0.452 | 0.350 | 0.077 | 6.37 |
| wg-square-1e6 | METIS | 1 | 8.2 | 1.91e10 | 5.78e7 | 0.98 | 0.496 | 0.346 | 0.080 | 12.2 |
| wg-square-1e6 | ND P9 | 3 | 1.80 (1.33) | 1.76e10 | 5.83e7 | 0.97 | 0.434 | 0.359 | 0.081 | 5.70 |

* At 1e6 the numeric work follows the flops: every ordering factorizes at 0.28-0.30 TFlop/s; a count
  takes 0.94-0.96 of a factorization.
* Memory: with the automatic budget of a 32-GiB run the factorization cap is 25.5-27.7 GB (report
  `factor_options.max_memory_mb`); every nested dissection fits at 1e6 (13.9-20.3 GB; ND P9 the
  smallest on yee-pec-box-1e6, -2.0 GB against SCOTCH), AMD (41-57 GB) does not, PORD on
  yee-pec-box-1e6 (23.5 GB) barely. The InertiaCounter of the counts runs after the main factors are
  released; its estimate is 4.8 GB (ND P9, yee-pec-box-1e6) to 7.0-8.4 GB (SCOTCH, syn-mem27-100).
* ss-3dspectralwave2 (n = 292008 complex, not in the gate set) is where the estimate decides the run
  (16 threads, build lane, analysis only, `~/p9_p9-ordering/fix/spec16`): AMD 94.9 GB, PORD 41.5,
  SCOTCH 27.2 (28.3 in a run), ND 25.4, ND + effort 24.6, METIS 23.3 GB. In the review's slot runs
  (R6, 2 each) main took SCOTCH - 28.3 GB, over its cap of 25.5 GB - and ran out of core (319 s, of
  which 18 out-of-core solves 121 s); the new binary took ND with effort (24.6 GB, cap 26.0 GB) in
  core: 160 s. Without a cap (`bench_anaint_decide`) main chooses METIS there, the new binary ND +
  effort.
* **SCOTCH (main's choice on both 3-D cases) is not the best ordering at 1e6**: on syn-mem27-100 its
  16-thread orders vary between 8.5e12 (P7) and 1.11e13 flops (1.25-1.63x METIS / ND); on
  yee-pec-box-1e6 it equals METIS. ND P9 saves **18.5 s (syn-mem27-100) and 14.4 s
  (yee-pec-box-1e6) of the three factorizations against SCOTCH's median run**, and its parallel
  ordering 2.0-2.4 s against SCOTCH's 4.3-6.3 s serial analysis.
* Geometric references (`tools/bench/ordering_geometric.py lattice 100 100 100` / `yee 92 71 54`:
  coordinate bisection of the lattice / of the Yee edge grid with plane separators, then
  `bench_ordering_cost --ordering -1 --perm-in`): ND P9 is within 0% (syn-mem27-100) and 2.7%
  (yee-pec-box-1e6) of them. Before the flow refinement hsolve's ND
  was 40% above the Yee reference while its top separator was only 12-14% above the plane (8472-8615
  vs 7543 unknowns): the deeper, irregular separators and their larger borders cost the rest.
* wg-square-1e6 (2-D, GHIEP: no counts in the real run): AMD stays cheapest for the whole run
  (one factorization, ~64 RHS); ND P9's ordering (1.3 s) eats its smaller numeric work.

## 3. The 1e5 3-D cases (task 1)

16 threads, slot; MUMPS orderings and the P5c ND (with the taskgroup fix: v1) from m2 (3 reps;
ss-* 2 reps), ND P9 from m7 (3 reps). Bold: the cheapest run model.

| case | ordering | reps | analysis s | flops | factor entries | in-core MB | factorize s | count s | 4-RHS solve s | run model s |
|---|---|---|---|---|---|---|---|---|---|---|
| ss-2cubes_sphere | AMD | 2 | 0.12 | 2.61e+11 | 8.87e+07 | 1477 | 1.274 | 1.233 | 0.0541 | 5.01 |
| ss-2cubes_sphere | SCOTCH | 2 | 0.69 | 5.24e+10 (0.99-1.01) | 4.19e+07 | 621 (613-621) | 0.421 | 0.360 | 0.0282 | 2.46 |
| ss-2cubes_sphere | METIS | 2 | 1.21 | 5.91e+10 | 4.29e+07 | 650 | 0.467 | 0.389 | 0.0289 | 3.09 |
| ss-2cubes_sphere | PORD | 2 | 1.24 | 7.63e+10 | 4.82e+07 | 679 | 0.519 | 0.423 | 0.0312 | 3.30 |
| ss-2cubes_sphere | ND, taskgroup fix only | 2 | 0.32 | 5.65e+10 | 4.25e+07 | 609 | 0.434 | 0.368 | 0.0279 | 2.11 |
| ss-2cubes_sphere | ND P9 | 3 | 0.35 | 4.77e+10 | 3.94e+07 | 591 | 0.401 | 0.344 | 0.0265 | **2.03** |
| ss-ge87h76 | SCOTCH | 2 | 3.36 | 4.89e+12 (0.99-1.01) | 5.75e+08 | 9958 (9924-10294) | 15.546 | 15.302 | 0.2992 | 55.66 |
| ss-ge87h76 | METIS | 2 | 2.42 | 5.76e+12 | 6.19e+08 | 10623 | 18.373 | 17.952 | 0.3204 | 63.28 |
| ss-ge87h76 | PORD | 2 | 5.80 | 8.95e+12 | 7.72e+08 | 11503 | 25.211 | 25.288 | 0.4012 | 89.78 |
| ss-ge87h76 | ND, taskgroup fix only | 2 | 1.05 | 6.12e+12 | 6.41e+08 | 10881 | 19.408 | 18.609 | 0.3305 | 64.44 |
| ss-ge87h76 | ND P9 | 3 | 1.92 | 5.11e+12 | 5.88e+08 | 10115 | 16.481 | 15.996 | 0.3058 | 56.67 |
| ss-ge87h76 | ND P9 + effort | 3 | 1.70 | 4.49e+12 | 5.59e+08 | 9596 | 14.815 | 14.334 | 0.2883 | **51.11** |
| ss-qa8fk-qa8fm | AMD | 2 | 0.07 | 7.29e+10 | 4.7e+07 | 727 | 0.525 | 0.450 | 0.0304 | 2.15 |
| ss-qa8fk-qa8fm | SCOTCH | 2 | 0.48 | 2.17e+10 (0.98-1.02) | 2.59e+07 | 371 (367-376) | 0.234 | 0.182 | 0.0180 | 1.48 |
| ss-qa8fk-qa8fm | METIS | 2 | 0.73 | 1.98e+10 | 2.4e+07 | 349 | 0.231 | 0.174 | 0.0170 | 1.69 |
| ss-qa8fk-qa8fm | PORD | 2 | 0.57 | 2.24e+10 | 2.58e+07 | 356 | 0.239 | 0.185 | 0.0182 | 1.58 |
| ss-qa8fk-qa8fm | ND, taskgroup fix only | 2 | 0.21 | 1.85e+10 | 2.34e+07 | 340 | 0.226 | 0.164 | 0.0162 | **1.13** |
| ss-qa8fk-qa8fm | ND P9 | 3 | 0.22 | 1.84e+10 | 2.33e+07 | 339 | 0.221 | 0.163 | 0.0162 | 1.13 |
| ss-si34h36 | SCOTCH | 2 | 2.59 | 3.37e+12 | 4.44e+08 | 7759 (7654-7858) | 11.215 | 10.933 | 0.2319 | 40.42 |
| ss-si34h36 | METIS | 2 | 1.93 | 4.05e+12 | 4.86e+08 | 8389 | 13.346 | 12.851 | 0.2536 | 46.17 |
| ss-si34h36 | PORD | 2 | 4.35 | 5.9e+12 | 5.83e+08 | 8284 | 17.128 | 16.775 | 0.2963 | 61.06 |
| ss-si34h36 | ND, taskgroup fix only | 2 | 0.81 | 3.67e+12 | 4.72e+08 | 8047 | 12.286 | 11.873 | 0.2444 | 41.84 |
| ss-si34h36 | ND P9 | 3 | 1.02 | 3.26e+12 | 4.43e+08 | 7612 | 10.868 | 10.628 | 0.2301 | 37.85 |
| ss-si34h36 | ND P9 + effort | 3 | 1.12 | 3.19e+12 | 4.34e+08 | 7436 | 10.729 | 10.395 | 0.2257 | **37.26** |
| ss-tem152078 | AMD | 2 | 0.16 | 3.01e+11 | 1.22e+08 | 1903 | 1.597 | 1.450 | 0.0738 | 6.27 |
| ss-tem152078 | SCOTCH | 2 | 0.74 | 7.05e+10 (0.99-1.01) | 6.43e+07 | 876 (873-876) | 0.599 | 0.482 | 0.0407 | 3.25 |
| ss-tem152078 | METIS | 2 | 1.79 | 7.24e+10 | 6.28e+07 | 880 | 0.606 | 0.477 | 0.0408 | 4.31 |
| ss-tem152078 | PORD | 2 | 1.05 | 8.04e+10 | 6.7e+07 | 920 | 0.646 | 0.528 | 0.0434 | 3.76 |
| ss-tem152078 | ND, taskgroup fix only | 3 | 0.70 | 7.58e+10 | 6.46e+07 | 895 | 0.625 | 0.499 | 0.0415 | 3.29 |
| ss-tem152078 | ND P9 | 3 | 0.40 | 7.18e+10 | 6.31e+07 | 873 | 0.589 | 0.481 | 0.0415 | **2.92** |
| syn-lap3d-aniso-1e5 | AMD | 2 | 0.08 | 8.89e+10 | 4.47e+07 | 730 | 0.538 | 0.473 | 0.0295 | 2.24 |
| syn-lap3d-aniso-1e5 | SCOTCH | 2 | 0.47 | 4.34e+10 (0.99-1.01) | 3.46e+07 | 507 (506-523) | 0.363 | 0.288 | 0.0239 | 1.97 |
| syn-lap3d-aniso-1e5 | METIS | 2 | 0.80 | 4.38e+10 | 3.12e+07 | 476 | 0.334 | 0.269 | 0.0227 | 2.21 |
| syn-lap3d-aniso-1e5 | PORD | 2 | 0.92 | 4.03e+10 | 2.97e+07 | 434 | 0.316 | 0.250 | 0.0220 | 2.25 |
| syn-lap3d-aniso-1e5 | ND, taskgroup fix only | 2 | 0.20 | 4.09e+10 | 3.17e+07 | 484 | 0.320 | 0.278 | 0.0237 | 1.63 |
| syn-lap3d-aniso-1e5 | ND P9 | 3 | 0.20 | 3.78e+10 | 2.97e+07 | 455 | 0.302 | 0.256 | 0.0216 | **1.52** |
| syn-mem125-46 | AMD | 3 | 0.19 | 1.53e+12 | 2.65e+08 | 4614 | 5.879 | 5.491 | 0.1449 | 20.85 |
| syn-mem125-46 | SCOTCH | 3 | 1.88 | 4.98e+11 (0.96-1.03) | 1.63e+08 | 2550 (2539-2586) | 2.457 | 2.309 | 0.0886 | 11.35 |
| syn-mem125-46 | METIS | 3 | 2.16 | 4.69e+11 | 1.59e+08 | 2494 | 2.387 | 2.208 | 0.0881 | 11.33 |
| syn-mem125-46 | PORD | 3 | 1.99 | 7.18e+11 | 1.87e+08 | 2874 | 3.123 | 2.896 | 0.1002 | 13.57 |
| syn-mem125-46 | ND, taskgroup fix only | 3 | 0.95 | 4.71e+11 | 1.6e+08 | 2506 | 2.365 | 2.179 | 0.0874 | **10.02** |
| syn-mem125-46 | ND P9 | 3 | 1.29 | 4.72e+11 | 1.6e+08 | 2509 | 2.349 | 2.175 | 0.0880 | 10.35 |
| syn-mem27-46 | AMD | 3 | 0.07 | 2.12e+11 | 8.23e+07 | 1277 | 1.086 | 1.027 | 0.0487 | 4.34 |
| syn-mem27-46 | SCOTCH | 3 | 0.72 | 7.67e+10 (0.92-1.00) | 5.33e+07 | 780 (750-782) | 0.520 | 0.448 | 0.0325 | 2.90 |
| syn-mem27-46 | METIS | 3 | 1.08 | 6.51e+10 | 4.9e+07 | 720 | 0.497 | 0.413 | 0.0318 | 3.16 |
| syn-mem27-46 | PORD | 3 | 0.51 | 8.92e+10 | 5.43e+07 | 740 | 0.569 | 0.488 | 0.0337 | 2.85 |
| syn-mem27-46 | ND, taskgroup fix only | 3 | 0.32 | 6.59e+10 | 4.98e+07 | 727 | 0.485 | 0.415 | 0.0321 | 2.39 |
| syn-mem27-46 | ND P9 | 3 | 0.30 | 6.55e+10 | 4.97e+07 | 727 | 0.481 | 0.413 | 0.0314 | **2.35** |
| yee-pec-box-1e5 | AMD | 3 | 0.09 | 4.87e+11 | 1.26e+08 | 2184 | 2.160 | 2.021 | 0.0752 | 7.85 |
| yee-pec-box-1e5 | SCOTCH | 3 | 0.64 | 6.69e+10 (0.93-1.11) | 5.1e+07 | 715 (692-743) | 0.518 | 0.417 | 0.0327 | 2.70 |
| yee-pec-box-1e5 | METIS | 3 | 1.23 | 7.25e+10 | 5.19e+07 | 739 | 0.540 | 0.467 | 0.0333 | 3.43 |
| yee-pec-box-1e5 | PORD | 3 | 0.77 | 1.42e+11 | 7.6e+07 | 996 | 0.914 | 0.700 | 0.0473 | 4.08 |
| yee-pec-box-1e5 | ND, taskgroup fix only | 3 | 0.29 | 8.1e+10 | 5.49e+07 | 794 | 0.572 | 0.496 | 0.0363 | 2.63 |
| yee-pec-box-1e5 | ND P9 | 3 | 0.30 | 5.96e+10 | 4.73e+07 | 672 | 0.469 | 0.405 | 0.0316 | **2.27** |
| yee-pec-cube-32 | AMD | 3 | 0.07 | 2.86e+11 | 9.09e+07 | 1451 | 1.353 | 1.261 | 0.0543 | 5.19 |
| yee-pec-cube-32 | SCOTCH | 3 | 0.56 | 5.85e+10 (0.96-1.05) | 4.34e+07 | 644 (632-654) | 0.454 | 0.358 | 0.0280 | 2.39 |
| yee-pec-cube-32 | METIS | 3 | 1.03 | 6.02e+10 | 4.34e+07 | 644 | 0.453 | 0.369 | 0.0282 | 2.89 |
| yee-pec-cube-32 | PORD | 3 | 0.66 | 9.4e+10 | 5.21e+07 | 725 | 0.566 | 0.504 | 0.0335 | 3.01 |
| yee-pec-cube-32 | ND, taskgroup fix only | 3 | 0.29 | 6.54e+10 | 4.53e+07 | 682 | 0.475 | 0.410 | 0.0293 | 2.27 |
| yee-pec-cube-32 | ND P9 | 3 | 0.26 | 5.16e+10 | 4.06e+07 | 590 | 0.388 | 0.331 | 0.0280 | **1.98** |

* ND P9 has the smallest flops on 7 of the 10 cases (METIS 0.6-0.7% fewer on syn-mem27-46 /
  syn-mem125-46, SCOTCH 1.8% fewer on ss-tem152078) and a cheaper run model than every MUMPS ordering
  on all 10 (ss-ge87h76 with the effort, which the cost model takes there: section 5.4): 7-24% below
  the best MUMPS ordering (SCOTCH on 9 of them, METIS on syn-mem125-46).
* Its analysis is the fastest of the nested dissections everywhere (0.20-1.9 s against SCOTCH's
  0.47-3.4 s).
* The flow costs analysis time on the high-degree graphs (syn-mem125-46, degree 114: ordering
  0.79 -> 1.13 s) without flops gain there; still the cheapest run.
* Memory (in-core MB, MUMPS' estimate without a cap): ND P9 (with the effort where the driver takes
  it) needs the least on 7 of the 10 cases and at most 4.8% more than the smallest on the others
  (syn-lap3d-aniso-1e5: PORD 434 against 455 MB); AMD needs 1.6-2.9x as much. Every nested
  dissection of these cases fits even a 16-GiB run (cap about 13.5 GB; largest: ss-ge87h76,
  9.6-11.5 GB).

## 4. ss-tem152078 (task 3)

P5c measured ss-tem152078 1.067x slower than P4 (results/p5c_measurements.md section 2): the cost
model took hsolve's ND, whose ordering (1.24-1.38 s) took twice SCOTCH's whole analysis (0.73 s),
because its one-thread work was 2.1x the fitted W ("no input statistic tried - degree, fill,
separator sizes - accounts for it").

**Cause**: half of ss-tem152078's vertices are indistinguishable from another one (identical closed
neighbourhoods: `compress_ratio.py` gives 76108 classes for 152078 vertices - two unknowns per
mesh entity). SCOTCH (in MUMPS' ESMUMPS strategy) and METIS compress such graphs before ordering;
hsolve's ND did not and ordered twice the graph, with twice the edges (mean degree 41.5). Second
cause, for every case: the taskwait bug (section 5.1).

**Fix**: `compress_graph` (section 5.3), on by default. ss-tem152078 at 16 threads (slot): ND
ordering 0.70 -> 0.26 s (P5c 1.24 s), analysis 0.40 s against SCOTCH's 0.74 s, flops 7.18e10
(SCOTCH 7.05e10, METIS 7.24e10), run model 2.92 s against SCOTCH's 3.25 s and P5c ND's 3.29 s; the
cost model's ND graph size now accounts for the merge (`indistinguishable_ratio`, section 6), so the
2.1x W outlier is gone (W predicted / measured 0.66-1.24 over 40 cases). Whole run (gate, section
8.2; main also takes ND here): **5.51 -> 4.41 s (0.800)** with the final code (the draft: 5.48 ->
4.47 s), analysis 1.36 -> 0.40 s, flops 7.58e10 -> 7.14e10 - faster than P4's 5.11 s measured in
P5c (other node).

## 5. What changed in the nested dissection (task 2)

### 5.1 The taskwait bug

`vertex_separator` ran its 5 trial bisections as tasks and waited with `#pragma omp taskwait`. The
dissection calls it from inside a task that has already spawned the other half of the previous
separator as a child task - taskwait waits for all children, so every separator on the critical
path also waited for a whole sibling subtree. syn-mem27-100 (build lane, `bench_ordering_cost
--nd-profile 1`): 18.1 s of work, critical path 2.3 s on one thread; 7.7 s on 4 threads, 7.3 s on
8, 7.4 s on 16; the trial runs' summed wall time 16.3 s against 3.2 s of work. With a `taskgroup`
(waits only for the runs): 3.6 s on 8 threads, the same order. 16-thread slot: 7.38 -> 2.89 s
(m1 -> m3b). The same bug made P5c's 16-thread ND look serial (nd_serial 0.383).

### 5.2 Concurrent contraction and splitting (537ff8e)

Graphs of >= 20000 vertices are contracted and split into the halves of a separator by `taskloop`s
(two passes: row lengths, then rows at their offsets), giving exactly the serial loops' graphs:
orders unchanged (syn-mem27-100 6.8089e12 flops before and after); ordering 3.72 -> 3.43 s
(syn-mem27-100), 0.32 -> 0.24 s (yee-pec-box-1e5), build lane. Matching and FM stay serial.

### 5.3 Compression and leaves by weight (2c4594b, edb3909)

`compress_graph` merges vertices with identical closed neighbourhoods when at most 85% remain
(METIS' rule): each vertex is compared with its neighbours only (indistinguishable vertices are
adjacent), O(edges), parallel; minimum degree on the leaves counts neighbour weights. Classes:
ss-tem152078 0.50, syn-near-singular-complex-1e5 0.125, the block syn-* cases; every other data
case 1.0. A first version kept leaves of 120 *compressed* vertices (up to 1920 MUMPS variables on the
complex block cases): MUMPS' tree-parallel L0 layer then idled - syn-near-singular-complex-1e5
factorized in 1.09 s against 0.40 s with the uncompressed order (build lane). Leaves now hold at
most 120 of vertex *weight*: 0.43 s, ordering 0.04 s instead of 0.20 s.

### 5.4 Trial separators carried down, effort on the top (2c4594b)

METIS' two-stage scheme keeps the best of 5 trial bisections *of the mid graph* (four contractions
down) and refines it back up. `SeparatorParams::finalists` refines the best F of them down to the
graph and keeps the best there; `OrderingOptions::top_*` does so (with R runs) only for the top
levels, where most threads would idle. Without flow refinement (build lane, analysis only, 12
3-D cases): runs 10 / finalists 10 everywhere -6.3% flops geomean (ss-ge87h76 -25%), at +20-60%
ordering time; top 4 levels with 16 / 16 -3.7% (1e5) at +40%. With the flow refinement (section
5.5) the gain shrinks to ss-ge87h76 (-12%: 5.11e12 -> 4.49e12, slot 56.7 -> 51.1 s run model) and
ss-si34h36 (-2%), nothing on the 1e6 cases (m6). The driver therefore takes it only where the
planned numeric time is at least 30x the predicted ordering time (`kNdEffortWorkRatio`; top 2
levels, 16 runs, 16 finalists: ordering x1.1-1.2), see section 6.

### 5.5 Flow refinement (6d96a08)

`FlowRefiner` (ordering_separator.cpp, after Sanders & Schulz, "Advanced multilevel node separator
algorithms", SEA 2016): after FM, the separator S plus up to 2 BFS layers on each side - as many as
keep any cut balanced (a half gains at most the other side's band) - become a split-vertex network
(capacity = vertex weight on v_in -> v_out, unbounded along the edges, the rest of half 0 the
source, of half 1 the sink); Dinic's maximum flow gives the smallest separator in the band. Of the
two extreme minimum cuts the better balanced one replaces S if it is smaller (or equal and better
balanced), polished by one FM round. It never enlarges a separator (test
`FactorOrderingSeparator.FlowRefinementNeverEnlargesTheSeparator`) and finds the 576-plane of a 24^3
grid for most seeds.

Variants (build lane, `bench_ordering_cost --ordering 11`, MUMPS flops relative to the ND before):

| variant | geomean flops (19 cases) | yee-pec-box-1e5 | yee-pec-box-1e6 | ss-ge87h76 | ordering time (8 thr) |
|---|---|---|---|---|---|
| flow at every level, also in the trial runs | - | 0.742 | 0.731 | - | 2-3x (yee-1e6 6.4 s, syn-mem27-100 10.8 s) |
| every level, not in the trials | - | 0.733 | 0.724 | - | 1.6-2.8x |
| **input graph of each separator only, depth 2 (default)** | **0.902** | **0.736** | **0.735** | **0.835** | **1.0-2.7x (yee-1e5 0.31 vs 0.31 s, 1e6 +0.3-0.8 s)** |
| ... depth 3 | 0.901 | 0.732 | 0.722 | 0.835 | slightly more |
| ... depth 1 | 0.898 (10 cases) | 0.794 | 0.791 | 0.853 | less |
| + effort top 4 x 16 (no finalist flow) | 0.854 | 0.727 | 0.718 | 0.723 | 3-4x |

The coarse levels' networks carry vertex weights and needed 78-95 Dinic phases (0.4-1.7 s each at
the top of syn-mem27-100) for no better separators; on the input graph the plane-like separators
need 2 phases. The implementation builds the CSR network directly (32-bit arcs and capacities):
1.2-1.7x faster than the first version, same cuts.

### 5.6 Parameters measured without gain (negative results)

Build lane, analysis only, MUMPS flops relative to the default ND (before the flow refinement;
`sw1`): leaf size 60 / 250 / 500: 0.999 / 1.002 / 1.013 geomean; imbalance 1.1 / 1.3: 1.000 /
1.006 (case-dependent +-10%); coarsen_to 50 / 200: 1.008 / 0.995; init tries 14: 0.992; FM passes
20: 1.000; FM two-sided / both instead of one-sided: mixed (+8% to -7%, ss-2cubes_sphere +33%
two-sided); other seeds: +-5% (ss-tem152078, ss-qa8fk-qa8fm up to 8%). None adopted. The geometric
references (section 2) bound what separators can still give on the regular grids. Flow refinement
variants: a balance guard on the cut (halves within 2% or 5% of the weight unless the old separator
was worse) costs yee-pec-box-1e5 8% flops and ss-2cubes_sphere 4-5%; flow only on graphs of >= 5000 /
20000 vertices: geomean 0.858 / 0.871 against 0.853 (8 cases) at hardly less ordering time - the
top separators' flows are the expensive ones.

## 6. The cost model (task 2) and the candidates

Three changes (10be12a), all in `src/driver/steps.cpp`:

1. **Candidates**: AMD, hsolve's ND, SCOTCH, METIS, PORD (P5c: SCOTCH before ND). With the flow
   refinement ND needs 0.73-1.0x SCOTCH's flops on the 3-D cases, its analysis is parallel and its
   order reproducible; SCOTCH stays a candidate, still taken on its prediction.
2. **ND graph size after compression**: `OrderingInput::nd` is the graph after merging
   indistinguishable vertices (`ordering::indistinguishable_ratio`: an order-independent hash of
   the closed neighbourhood and the degree, compared along the pattern's edges, no graph built;
   r of the vertices and ~r^2 of the edges remain). Without it the refit W missed the block syn-*
   cases by 5-10x and the 1-thread ND work fit collapsed (predicted / measured 0.05-1.6).
3. **Refit** with `bench_anaint_fix_fit` on the P5c raw runs (`~/p5b_anaint_fix/ord16.jsonl`,
   `ord4.jsonl`: AMD / SCOTCH / METIS / PORD rows kept) with every ND row replaced by P9 runs (m8,
   m5: `bench_anaint_orderings --ordering 10`, 16 threads x 1 and 4 threads `--nd-serial 1` x 3 on
   cores 16-19, slot) and the two 1e6 3-D cases added (m4: all MUMPS orderings, m5: ND):

   ```bash
   S=$HOME/p9_p9-ordering
   python3 tools/bench/ordering_fit_merge.py --base $HOME/p5b_anaint_fix --nd $S/m8 --nd $S/m5 \
     --extra $S/m4 --cases data/cases/* $HOME/p7_p7-membudget/data/cases/{syn-mem27-100,yee-pec-box-1e6} \
     --out $S/fit2                          # ord16.jsonl, ord4.jsonl, cases/ (links)
   python3 tools/run/guard.py --skip-busy-check --mem-gb 11 --timeout 1100 -- \
     build/release/bench_anaint_fix_fit --ord16 $S/fit2/ord16.jsonl --ord4 $S/fit2/ord4.jsonl \
     --data $S/fit2/cases --loo 1 --verbose 1
   ```

   | constant | P5c | P9 | why |
   |---|---|---|---|
   | nd_work_vertex / edge / fe | 7.03e-6 / 6.54e-8 / 3.85e-9 | 4.65e-6 / 7.47e-7 / 2.03e-9 | flow work grows with the edges; compressed sizes |
   | nd_serial | 0.383 | 0.222 | the taskwait bug |
   | rho METIS / PORD / SCOTCH | 1.033 / 1.126 / 0.626 | 0.686 / 0.582 / 0.377 | relative to the new W |
   | ND shape (ln flops ratio at fill 1, 4, 10, 40, 150, 1000) | 0.81 1.00 -0.52 -1.03 -1.32 -1.90 | 0.76 0.93 -0.54 -1.10 -1.56 -2.07 | flow refinement |
   | SCOTCH shape | 0.88 1.01 -0.49 -1.03 -1.43 -1.87 | 0.88 1.01 -0.49 -1.03 -1.39 -2.03 | 1e6 cases |
   | AMD low-fill factorization / solve | 1.86 / 3.15 | 1.49 / 2.82 | relative to the new ND |
   | factorization, solve, thread exponents | - | within 4-15% | refit on the same data |

   Ranges of predicted / measured: W [0.66, 1.24] (P5c [0.47, 1.38], ss-tem152078 2.1x outside),
   ND analysis [0.42, 1.74] (the 1e6 cases overpredicted 1.7x: their ordering scales better than
   the median), shapes as in P5c.

**Replay** (the driver's `OrderingPricer` / `run_ordering_choice` on the measured components of 38
cases at 16 threads, AMD probe always paid): mean excess over the best ordering **1.1%** in-sample
and leave-one-case-out (P5c's model on P5c's data: 4.3% / 5.0%); summed 205.35 s against the best
205.31 s. The best ordering is ND on every 3-D case and AMD on the 2-D ones; the remaining excess
is on four n <= 1e5 low-fill syn-* cases where METIS is best (1.0-26.6%, at most 0.02 s each). The
P5c constants with the new candidate order replay to the same 1.1%: the order and the faster ND
carry the choice, the refit makes the predictions right (the 1e6 ND predicted within 1.5x instead
of 2.6x).

Choices at 16 threads (`bench_anaint_decide`, build lane - the choice does not depend on timing;
all 42 data cases and the two 1e6 3-D cases, `~/p9_p9-ordering/fix/decide_final.jsonl` (0b2da5a)
for the final code, main a831d79's from the review's run of its
`bench_anaint_decide` and the main runs of the gate, which agree):

| | main a831d79 (P5c model) | P9 (after the review fix) |
|---|---|---|
| syn-mem27-100, yee-pec-box-1e5, ss-2cubes_sphere, syn-mem125-46, yee-pec-cube-32 | SCOTCH | ND |
| yee-pec-box-1e6, ss-ge87h76, ss-si34h36 | SCOTCH | ND + effort |
| ss-3dspectralwave2 | METIS (SCOTCH out of core in a 32-GiB run, section 2) | ND + effort |
| ss-tem152078, ss-qa8fk-qa8fm, syn-lap3d-aniso-1e5, syn-mem27-46, yee-bloch-1e4, yee-photonic-rods-1e4 / -spheres-1e4, the block syn-* cases (real) | ND | ND |
| syn-near-singular-complex-1e5 and the complex block syn-* 1e4 cases | ND | ND + effort |
| yee-pec-cube-16, yee-pec-box-1e4, ss-tem27623, ss-helm3d01, wg-square-n289 | ND | ND without flow (section 8.3) |
| the 2-D cases (wg-*, cav2d-*, syn-lap2d-100) and syn-lap3d-22 | AMD | AMD |

The P9 draft of this section said main chose SCOTCH on all of ss-tem152078,
syn-near-singular-complex-1e5 and ss-3dspectralwave2 too (or AMD): wrong - the table above is the
measured choice (review finding).

**Effort rule** (`kNdEffortWorkRatio` = 30, top 2 levels, 16 runs / 16 finalists): priced as
ordering x1.2, flops x0.97, entries x0.99 (build lane: ordering x1.05-1.2 on the large cases). m7:
ss-ge87h76 flops 5.11e12 -> 4.49e12, run model 56.7 -> 51.1 s (its ordering even 1.76 -> 1.54 s:
other separators, less flow work below); ss-si34h36 37.9 -> 37.3 s; m6: no change at 1e6. It is
taken on yee-pec-box-1e6, where it does not pay (+0.15 s, 0.2%).

**Flow rule** (review fix, 0b2da5a; `kNdFlowWorkRatio` = 4.5): on the same ratio, the predicted
planned numeric time over the predicted ordering time (`OrderingCandidate::nd_work_ratio`, both
priced as fitted, with the flow), the dissection runs without the flow refinement below 4.5, priced
as ordering / 1.15, flops / 0.90, entries / 0.95; an analysed dissection without it anchors the other
candidates' predictions as the fitted one would (flops x0.90, entries x0.95). The ordering event logs
`10n` and the ratio (`/r3.62`). Section 8.3 has the measurements behind it; the decision ratios of the
data cases: 2.2 (wg-square-n289), 3.27-3.62 (ss-helm3d01, yee-pec-box-1e4, ss-tem27623,
yee-pec-cube-16: no flow), then 5.61 (syn-lap3d-aniso-1e5), 5.78 (ss-qa8fk-qa8fm), 5.85 (the Yee 1e4
cells), 5.93 (syn-mem27-46), 6.96 ... 23 (flow; syn-mem27-100 21.2), 30 and more effort
(yee-pec-box-1e6 33.2, ss-ge87h76 44.8, the complex block syn-* 34-55, ss-3dspectralwave2 71.8). The
chosen orderings are the P9 draft's everywhere. The replay of the refit (`bench_anaint_fix_fit`, same
data) gives the same output with 0b2da5a as before, 1.1% in-sample and leave-one-case-out: no
fitted case changes its choice.


## 7. Correctness and determinism

* Every order is a function of the graph and the options only (not of the thread count): the
  taskloops write each output row from one task; compression compares neighbours; the flow is
  deterministic; finalists and runs have fixed seeds. Tests (`tests/cpp/test_ordering_effort.cpp`,
  16 tests): compression (vector fields, ratio, no-op graphs, blocks in compute_ordering, thread
  counts), the concurrent split of a 27000-vertex graph against its serial definition (review fix),
  weighted minimum degree, finalists, top effort by depth and size, flow refinement never
  enlarging a separator and finding planes, option ranges, `FactorOptions::nested_dissection`, the
  driver's effort rule and flow rule (review fix: the options, the rule over a sweep of the AMD
  flops crossing both thresholds - flags, prices and the logged ratio -, an analysed anchor without
  the flow); the existing ordering tests unchanged except the candidate order and the
  analysis-time property (now at the fitted 16 threads). `tests/py/test_bench_ordering.py`: the
  geometric references (even planes separate the Yee curl-curl graph of `tools/gen/yee.py`) and
  the fit data merge.
* The certificate path is unchanged: InertiaCounter counts reuse the main analysis' order (PERM_IN),
  whatever ordering produced it.

## 8. Gate: end-to-end

Two rounds, both main (a831d79) against the branch, each case's default configuration (config.json,
`--threads 16`, CLI defaults: automatic memory budget from the slot's 32 GiB, certificate exact),
`tools/bench/ab_cases.py run` in ABBA order inside one outer slot guard per part, every run
evaluated by `tools/eval` (`python3 -m tools.eval.sweep RUN --root DATA`), tables by `ab_cases.py
table`:

* the P9 draft, new = v16 (10be12a): G6 (1e6), G5 (1e5), G5s (10-rep re-check);
* **the final code, new = fix1 (0b2da5a: the flow rule of section 8.3)**: G8 (1e6), G7 (1e5), G7s
  and G7r (10-rep re-checks), G7n (fix1 against v16 on the four cases the rule changes). Later
  commits change comments, tests and documents only.

Machine-readable: `results/p9_gate_1e6.json`, `results/p9_gate_1e5.json`,
`results/p9_gate_1e4_recheck.json` (the draft) and `results/p9_gate_fix_1e6.json`,
`results/p9_gate_fix_1e5.json`, `results/p9_gate_fix_recheck.json` (the final code).

```bash
S=$HOME/p9_p9-ordering   # gpu02, from the work tree; fix/gate.sh, fix/post.sh, fix/eval.sh
HS_DATA_ROOT=$HOME/p7_p7-membudget/data python3 tools/run/guard.py --exclusive --timeout 2700 \
  --mem-gb 32 -- python3 tools/bench/ab_cases.py run --bin main=$S/bin/hsolve_main \
  --bin new=$S/bin/hsolve_fix1 --reps 0,1,2 --run-dir $S/fix/G8a --timeout 1500 --deadline 2300 \
  syn-mem27-100 yee-pec-box-1e6                # G8b: --reps 3,4; G6a/G6b: hsolve_v16
python3 tools/run/guard.py --exclusive --timeout 2700 --mem-gb 32 -- python3 tools/bench/ab_cases.py \
  run --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_fix1 --reps 0,1,2 --run-dir $S/fix/G7 \
  --timeout 1500 --deadline 2300 CASE...       # the 26 cases below; G5: hsolve_v16
python3 -m tools.eval.sweep $S/fix/G8a --root $HOME/p7_p7-membudget/data --out $S/fix/G8a/eval.md
python3 tools/bench/ab_cases.py table --base main --new new --json OUT.json $S/fix/G8a $S/fix/G8b
```

### 8.1 n = 1e6 in core: 5 interleaved runs

The final code (G8a + G8b, main vs fix1):

| case | main s [min-max] | new s [min-max] | new / main | speedup | valid main / new | certificate |
|---|---|---|---|---|---|---|
| syn-mem27-100 | 111.00 [106.63-117.07] | 89.02 [88.21-89.13] | 0.802 | **1.25x** | 5/5 / 5/5 | exact |
| yee-pec-box-1e6 | 89.70 [86.58-92.71] | 73.68 [73.49-73.76] | 0.821 | **1.22x** | 5/5 / 5/5 | exact |

Medians of the phases (reports):

| case | binary | ordering | total s | factor s (analysis, numeric) | iterate s (solves) | certificate s | flops | factor entries | peak RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | main | SCOTCH | 111.00 | 36.28 (5.96, 29.56) | 17.62 (16.02) | 56.78 | 8.83e12 | 1.35e9 | 15.5 |
| syn-mem27-100 | new | ND | 89.02 | 27.37 (3.02, 23.52) | 16.20 (14.60) | 44.70 | 6.72e12 | 1.19e9 | 13.1 |
| yee-pec-box-1e6 | main | SCOTCH | 89.70 | 29.30 (4.56, 23.86) | 14.27 (11.99) | 45.46 | 6.98e12 | 1.18e9 | 13.0 |
| yee-pec-box-1e6 | new | ND (+ effort) | 73.68 | 23.02 (2.71, 19.37) | 13.21 (10.94) | 36.87 | 5.45e12 | 1.04e9 | 11.3 |

The gain is the factorizations (main + 2 certificate counts: -18.1 s / -13.1 s), the analysis
(-2.9 s / -1.9 s) and the solves (-1.4 s / -1.1 s, fewer factor entries); peak RSS -1.7..-2.4 GiB.
Both cases pass the gate (>= 1.05x, every run valid, certificate exact). The flow rule does not
touch them (ratios 21 and 33: flow on, as in the draft): the draft's runs (G6, v16) gave
113.56 -> 89.37 s (1.27x) and 89.13 -> 73.68 s (1.21x), the same new times within 0.4%; main's
SCOTCH varies more (106.6-117.1 s on syn-mem27-100: its orders differ from run to run).

### 8.2 n <= 1e5: 26 data cases, ABBA, 3 repetitions

The final code (G7, main vs fix1):

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | gpu02 | 0.295 [0.295-0.330] | 0.294 [0.293-0.296] | 0.996 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | gpu02 | 0.039 [0.039-0.041] | 0.041 [0.038-0.043] | 1.029 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | gpu02 | 2.889 [2.877-2.978] | 2.457 [2.448-2.462] | 0.850 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-ge87h76 | 112985 | HEP | gpu02 | 55.313 [54.701-55.994] | 51.940 [51.871-51.984] | 0.939 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-helm3d01 | 32226 | HEP | gpu02 | 0.550 [0.544-0.608] | 0.502 [0.500-0.695] | 0.913 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | gpu02 | 1.696 [1.672-1.713] | 1.516 [1.501-1.520] | 0.894 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-si34h36 | 97569 | HEP | gpu02 | 40.115 [38.996-40.791] | 36.822 [36.762-36.909] | 0.918 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-tem152078 | 152078 | GHEP | gpu02 | 5.511 [5.462-5.537] | 4.411 [4.403-4.438] | 0.800 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | gpu02 | 0.593 [0.582-0.610] | 0.555 [0.550-0.567] | 0.936 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | gpu02 | 1.290 [1.280-1.320] | 0.975 [0.970-0.975] | 0.756 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu02 | 0.061 [0.055-0.062] | 0.059 [0.055-0.059] | 0.966 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu02 | 0.132 [0.132-0.139] | 0.136 [0.132-0.150] | 1.026 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | gpu02 | 2.097 [2.094-2.111] | 1.784 [1.776-1.791] | 0.851 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem125-46 | 97336 | HEP | gpu02 | 11.241 [11.188-11.371] | 10.746 [10.735-10.813] | 0.956 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-mem27-46 | 97336 | HEP | gpu02 | 2.415 [2.365-2.451] | 2.045 [2.029-2.077] | 0.847 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | gpu02 | 2.455 [2.425-2.489] | 2.219 [2.205-2.223] | 0.904 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | gpu02 | 1.208 [1.199-1.217] | 0.927 [0.916-0.931] | 0.767 | 3/3 / 3/3 | 6 / 6 | 4 / 4 | exact |
| wg-halfslab-1e5 | 99225 | GHIEP | gpu02 | 0.696 [0.695-0.723] | 0.746 [0.700-0.767] | 1.071 **slower** | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-1e5 | 99225 | GHIEP | gpu02 | 0.480 [0.480-0.484] | 0.489 [0.482-0.563] | 1.018 | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-pert-1e4 | 9801 | GHIEP | gpu02 | 0.060 [0.058-0.062] | 0.061 [0.060-0.062] | 1.021 | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| yee-bloch-1e4 | 10080 | HEP | gpu02 | 0.823 [0.823-0.879] | 0.810 [0.807-0.814] | 0.984 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | gpu02 | 0.198 [0.197-0.206] | 0.192 [0.191-0.195] | 0.967 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | gpu02 | 2.951 [2.879-2.955] | 2.475 [2.473-2.482] | 0.839 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | gpu02 | 0.209 [0.202-0.247] | 0.205 [0.204-0.242] | 0.980 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | gpu02 | 2.253 [2.221-2.273] | 1.941 [1.938-1.961] | 0.862 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | gpu02 | 0.920 [0.914-0.925] | 0.837 [0.826-0.850] | 0.910 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |

Geometric mean of new/main over the 26 cases valid in every run: **0.919**; faster by > 3%: 18; slower by > 3%: 1 (wg-halfslab-1e5 1.071).

* **Geometric mean new / main 0.919** over 26 cases, every run valid (156/156), certificate exact
  where planned; faster by > 3% on 18 cases (0.76-0.97), the other 3-D cases within 3%
  (yee-bloch-1e4 0.984, yee-pec-cube-16 0.980); the 2-D / GHIEP cases and syn-lap3d-22 run AMD in
  both binaries (same factors).
* The one case above 3% here, wg-halfslab-1e5 (1.071), runs AMD in both binaries with the same
  factors; its Krylov-Schur needs 44-48 solves from run to run in either binary (the L0-OpenMP
  factorization is not bitwise reproducible): 44, 44, 46 for main against 48, 48, 44 here. With 10
  repetitions (G7r) it is 1.007 (p = 0.19).

10- and 20-repetition re-checks (G7s + G7r, main vs fix1, ABBA; every run valid):

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-square-1e4 | 9976 | GHEP | gpu02 | 0.042 [0.038-0.050] | 0.042 [0.039-0.051] | 1.015 | 20/20 / 20/20 | 3 / 3 | 2 / 2 | exact |
| ss-helm3d01 | 32226 | HEP | gpu02 | 0.548 [0.541-0.581] | 0.501 [0.498-0.550] | 0.915 | 10/10 / 10/10 | 3 / 3 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | gpu02 | 0.585 [0.577-0.661] | 0.553 [0.545-0.587] | 0.945 | 10/10 / 10/10 | 4 / 4 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu02 | 0.060 [0.055-0.107] | 0.057 [0.056-0.073] | 0.963 | 10/10 / 10/10 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu02 | 0.137 [0.131-0.269] | 0.137 [0.130-0.188] | 1.002 | 20/20 / 20/20 | 3 / 3 | 4 / 4 | exact |
| wg-halfslab-1e5 | 99225 | GHIEP | gpu02 | 0.741 [0.693-0.758] | 0.746 [0.701-0.762] | 1.007 | 10/10 / 10/10 | 1 / 1 | 2 / 2 | none |
| wg-square-1e5 | 99225 | GHIEP | gpu02 | 0.479 [0.475-0.486] | 0.483 [0.478-0.489] | 1.008 | 10/10 / 10/10 | 1 / 1 | 2 / 2 | none |
| wg-square-n289 | 289 | GHIEP | gpu02 | 0.015 [0.013-0.018] | 0.016 [0.013-0.017] | 1.031 **slower** | 10/10 / 10/10 | 1 / 1 | 1 / 1 | none |
| wg-square-pert-1e4 | 9801 | GHIEP | gpu02 | 0.061 [0.059-0.070] | 0.061 [0.059-0.094] | 0.993 | 20/20 / 20/20 | 1 / 1 | 1 / 1 | none |
| yee-pec-box-1e4 | 10064 | GHEP | gpu02 | 0.193 [0.190-0.210] | 0.196 [0.189-0.247] | 1.013 | 20/20 / 20/20 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | gpu02 | 0.204 [0.196-0.293] | 0.207 [0.200-0.286] | 1.015 | 20/20 / 20/20 | 3 / 3 | 4 / 4 | exact |

Geometric mean of new/main over the 11 cases valid in every run: **0.991**; faster by > 3%: 3; slower by > 3%: 1 (wg-square-n289 1.031).

* yee-pec-cube-16 1.015 and yee-pec-box-1e4 1.013 over 20 runs each (G7s alone: 1.026 / 1.023,
  p = 0.09 / 0.04; G7r alone: 1.005 / 1.002, p = 0.37 / 0.43): no systematic difference left - they
  now run main's pivot order (bitwise the same SYM_PERM, `fix/permcmp.sh`; also ss-tem27623,
  ss-helm3d01, wg-square-n289).
* wg-square-n289 (n = 289, GHIEP, 15-19 ms) 1.031 here, 1.090 in 30 more repetitions (G7t: 17.8 ->
  19.4 ms, p = 0.045). Same pivot order and factors as main; the time is in the solves on those
  identical factors (57 calls, 5.2 -> 5.8 ms) and +0.1 ms in the analysis. With AMD forced in both
  binaries (G7u, 30 reps, a copy of the case with `ordering: 0`) the binaries are equal (19.1 / 18.9
  ms, 0.988). Not an ordering effect we could find (event logs, L1, are too noisy for it: flushed to
  the shared home, they move even the matrix load by 7 ms); reported as the one residual (1.6 ms).
* wg-square-1e5 1.008 (p = 0.02, +4 ms of 0.48 s, AMD in both): below 1%.

The P9 draft (G5, main vs v16; G5s re-check) for the record - the same 26 cases, geometric mean 0.920:

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | gpu02 | 0.295 [0.292-0.304] | 0.295 [0.295-0.299] | 1.001 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | gpu02 | 0.039 [0.039-0.040] | 0.042 [0.041-0.043] | 1.075 **slower** | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | gpu02 | 2.890 [2.849-2.908] | 2.470 [2.467-2.500] | 0.855 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-ge87h76 | 112985 | HEP | gpu02 | 56.605 [55.257-56.686] | 51.884 [51.719-52.120] | 0.917 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-helm3d01 | 32226 | HEP | gpu02 | 0.549 [0.548-0.551] | 0.528 [0.527-0.533] | 0.963 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | gpu02 | 1.695 [1.683-1.772] | 1.543 [1.511-1.548] | 0.910 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-si34h36 | 97569 | HEP | gpu02 | 40.251 [39.639-40.430] | 36.780 [36.671-36.781] | 0.914 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-tem152078 | 152078 | GHEP | gpu02 | 5.483 [5.478-5.760] | 4.466 [4.430-4.489] | 0.815 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | gpu02 | 0.593 [0.580-0.595] | 0.531 [0.531-0.548] | 0.896 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | gpu02 | 1.282 [1.280-1.283] | 0.970 [0.963-0.977] | 0.757 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu02 | 0.057 [0.054-0.062] | 0.062 [0.061-0.063] | 1.094 **slower** | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu02 | 0.137 [0.136-0.143] | 0.136 [0.132-0.137] | 0.997 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | gpu02 | 2.083 [2.068-2.085] | 1.811 [1.779-1.910] | 0.869 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem125-46 | 97336 | HEP | gpu02 | 11.393 [11.364-11.510] | 10.786 [10.691-10.791] | 0.947 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-mem27-46 | 97336 | HEP | gpu02 | 2.407 [2.365-2.435] | 2.036 [2.027-2.068] | 0.846 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | gpu02 | 2.477 [2.474-2.488] | 2.205 [2.195-2.258] | 0.890 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | gpu02 | 1.212 [1.202-1.216] | 0.933 [0.921-0.933] | 0.770 | 3/3 / 3/3 | 6 / 6 | 4 / 4 | exact |
| wg-halfslab-1e5 | 99225 | GHIEP | gpu02 | 0.734 [0.696-0.744] | 0.750 [0.729-0.750] | 1.021 | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-1e5 | 99225 | GHIEP | gpu02 | 0.480 [0.478-0.504] | 0.483 [0.482-0.499] | 1.006 | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-pert-1e4 | 9801 | GHIEP | gpu02 | 0.067 [0.061-0.073] | 0.063 [0.061-0.065] | 0.952 | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| yee-bloch-1e4 | 10080 | HEP | gpu02 | 0.828 [0.828-0.829] | 0.813 [0.812-0.910] | 0.981 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | gpu02 | 0.192 [0.192-0.202] | 0.180 [0.179-0.180] | 0.937 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | gpu02 | 2.987 [2.888-3.003] | 2.491 [2.490-2.491] | 0.834 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | gpu02 | 0.222 [0.204-0.233] | 0.225 [0.222-0.260] | 1.011 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | gpu02 | 2.290 [2.215-2.338] | 1.939 [1.932-2.074] | 0.847 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | gpu02 | 0.916 [0.911-0.917] | 0.838 [0.826-0.847] | 0.915 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |

Geometric mean of new/main over the 26 cases valid in every run: **0.920**; faster by > 3%: 18; slower by > 3%: 2 (cav2d-square-1e4 1.075, syn-lap2d-100 1.094).

The two draft cases above 3% there (cav2d-square-1e4 1.075, syn-lap2d-100 1.094) were 0.973 / 0.966 in G5s; yee-pec-cube-16 was 1.011 there and 1.074 in G5s (section 8.3).

10-repetition re-check of the small cases (G5s, ABBA):

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-square-1e4 | 9976 | GHEP | gpu02 | 0.043 [0.039-0.073] | 0.041 [0.039-0.067] | 0.973 | 10/10 / 10/10 | 3 / 3 | 2 / 2 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu02 | 0.058 [0.054-0.075] | 0.056 [0.055-0.068] | 0.966 | 10/10 / 10/10 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu02 | 0.137 [0.130-0.150] | 0.136 [0.132-0.166] | 0.997 | 10/10 / 10/10 | 3 / 3 | 4 / 4 | exact |
| wg-square-pert-1e4 | 9801 | GHIEP | gpu02 | 0.064 [0.060-0.083] | 0.065 [0.061-0.074] | 1.019 | 10/10 / 10/10 | 1 / 1 | 1 / 1 | none |
| yee-pec-cube-16 | 10800 | HEP | gpu02 | 0.206 [0.203-0.287] | 0.221 [0.218-0.231] | 1.074 **slower** | 10/10 / 10/10 | 3 / 3 | 4 / 4 | exact |

Geometric mean of new/main over the 5 cases valid in every run: **1.005**; faster by > 3%: 1; slower by > 3%: 1 (yee-pec-cube-16 1.074).

### 8.3 yee-pec-cube-16: the flow refinement at the small end (review finding)

The draft (v16) reported yee-pec-cube-16 (n = 10800, HEP, 0.2 s) as "+7%, ranges overlapping" and
blamed MUMPS' 4-column solves alone. The review showed it systematic; re-measured here (V1):

| run set (slot, ABBA) | reps | main s | v16 s | v16 / main | one-sided Mann-Whitney p (v16 slower) |
|---|---|---|---|---|---|
| G5s (draft) | 10 | 0.206 | 0.221 | 1.074 | 0.032 |
| review R3 | 5 | 0.206 | 0.229 | 1.108 | 0.016 |
| review R4 | 10 | 0.213 | 0.225 | 1.059 | 0.15 |
| review R7 | 15 | 0.210 | 0.228 | 1.085 | 0.006 |
| V1 (this fix) | 10 | 0.209 | 0.228 | 1.090 | 0.013 |

Where the 19 ms go (V1, medians of the reports, main -> v16): analysis 43.7 -> 48.1 ms (the flow's
ordering time), the 3 factorizations 92.2 -> 97.0 ms, the solves 43.4 -> 51.5 ms (22 -> 23 calls -
one Lanczos restart more - and 1.97 -> 2.24 ms per 4-column call), the certificate's two counts
57.1 -> 61.9 ms. Not the solves alone: every numeric phase on the flow-refined order is slower,
although it has 17% fewer flops (6.91e8 against 8.37e8) and 5% fewer entries. Bench F1 (16 threads,
slot, 5 interleaved reps, ND with / without the flow, everything else equal):

| case | ratio (driver) | flops with / without | ordering ms | factorization ms | 2 counts ms | 1-col / 4-col solve ms | run model change |
|---|---|---|---|---|---|---|---|
| yee-pec-cube-16 | 3.62 | 0.83 | 58.0 / 52.3 | 39.4 / 35.7 | 52.9 / 50.9 | 1.65 / 2.17 vs 1.33 / 1.84 | **+19 ms** |
| yee-pec-box-1e4 | 3.46 | 0.76 | 47.6 / 48.7 | 29.5 / 31.5 | 45.2 / 46.7 | 1.15 / 1.68 vs 1.20 / 1.79 | -7 ms |
| ss-tem27623 | 3.47 | 0.89 | 91.1 / 72.8 | 53.6 / 64.0 | 76.6 / 94.1 | 3.21 / 4.22 vs 3.71 / 4.84 | -24 ms |
| ss-helm3d01 | 3.27 | 0.89 | 119.7 / 85.9 | 67.6 / 72.0 | 105.8 / 111.5 | 4.55 / 5.67 vs 4.82 / 5.91 | **+18 ms** |
| yee-photonic-rods-1e4 | 5.85 | 0.90 | 102.8 / 94.0 | 165.8 / 187.4 | 294.4 / 340.9 | 8.12 / 9.16 vs 9.00 / 10.13 | -81 ms |
| 6 block syn-* cases | 10-50 | 1.00 | +-2.5 ms | equal | equal | equal | none (separators of 1-3 vertices) |

(run model: analysis + factorization + 2 counts + 22 four-column solves.) MUMPS' own statistics of
yee-pec-cube-16 (`bench_ordering_cost --mumps-verbosity 2`, build lane, 8 threads): the
flow-refined order has 357 instead of 403 tree nodes and puts more of the work under the
tree-parallel L0 layer (1.65e8 against 1.54e8 flops): at this size the time follows the tree's shape
and the per-front overhead rather than the flops (the mechanism inside MUMPS' L0 scheduling was not
isolated). The elimination trees of both orders (fundamental supernodes, `fix/tree.py`) differ in
height (109 against 74 supernodes for yee-pec-cube-16) but so do yee-pec-box-1e4's (99 against
81), where the flow order is the faster one: no static tree statistic tried separates the two cases.

**Fix** (0b2da5a, section 6 "Flow rule"): the driver runs the dissection without the flow where the
planned numeric time is under 4.5x the predicted ordering time. At the small end the flow's net
effect is even (+19, -7, -24, +18 ms on the four cases between 3.3 and 3.6) and its risk one-sided
against main (no flow = main's order there), from 5.6 on it gains (section 6). The draft's
alternatives were measured and rejected: flow only on graphs of >= 5000 vertices keeps the slow
order (m10: yee-pec-cube-16 flops 7.2e8, 4-column solve 2.18 ms), >= 20000 restores it but also
drops the flow on yee-photonic-rods-1e4 and yee-bloch-1e4 (ordered as 10125- / 10080-vertex
graphs), where it gains up to 9%; a balance guard on the cut (the draft, section 5.6) leaves
yee-pec-cube-16's order unchanged and costs yee-pec-box-1e5 8% flops.

Result (fix1, slot, ABBA, every run valid, certificate exact):

| case | ratio | v16 / main (draft) | fix1 / main: G7 (3 reps); G7s + G7r (20) | fix1 / v16 (G7n, 10) |
|---|---|---|---|---|
| yee-pec-cube-16 | 3.62 | 1.090 (V1, p = 0.013) | 0.980; 1.015 | 0.960 |
| yee-pec-box-1e4 | 3.46 | 0.951 (V1) | 0.967; 1.013 | 1.054 |
| ss-tem27623 | 3.47 | 0.896 (G5) | 0.936; 0.945 (10) | 1.033 |
| ss-helm3d01 | 3.27 | 0.963 (G5) | 0.913; 0.915 (10) | 0.947 |
| wg-square-n289 | 2.20 | 1.034 (review R7, 15) | -; 1.031 (10), 1.090 (G7t, 30) | - |

The regression is gone: yee-pec-cube-16 runs main's pivot order (bitwise) at main's time (1.015
over 20 runs, p > 0.09 in either half). The rule gives back yee-pec-box-1e4's flow gain (about 1.0
against main instead of 0.95) and part of ss-tem27623's (0.945 instead of 0.896) and takes
ss-helm3d01 further (0.915 instead of 0.963); over the four, fix1 against v16 -11 ms (G7n:
-9, +10, +17, -29 ms). The 26-case geometric mean is unchanged (0.919 against the draft's 0.920),
1e6 untouched. wg-square-n289 (section 8.2) keeps a 1.6 ms residual that is not in the ordering.

## 9. Open issues

1. **The AMD probe on 3-D cases.** Every adaptive analysis starts with an AMD analysis (0.07-0.19 s
   at 1e5, 0.8-1.0 s at 1e6) whose only use on a 3-D case is the fill ratio that rules AMD out. A
   cheap classifier (BFS level structure, or the ND estimate first when the graph looks 3-D) or
   running the probe concurrently with the parallel ND ordering would save 3-5% of a 1e5 3-D run and
   ~1% at 1e6. Not done: the classifier must not misjudge the 2-D cases, where ND first would cost
   more than the probe (wg-square-1e5: ND ordering 0.15 s against a 0.4 s run).
2. **Flow refinement on high-degree graphs.** On syn-mem125-46 (degree 114) the flow costs 0.34 s
   of ordering (0.79 -> 1.13 s) and gains nothing (its separators are already planes); a band depth
   by degree, or skipping the flow where FM's separator is already a plane, would save it. The cost
   model prices it (W's edge term), so the choice is unaffected. Its ratio (7.65) keeps the flow on
   under the review's flow rule; a per-separator criterion (flow only where it shrinks the separator
   by enough to pay for its own time) would subsume both.
3. **The ND analysis model at 1e6** overpredicts 1.7x (the median serial fraction fits the 1e5
   cases); harmless for the choices measured (ND wins there by its flops), but the effort rule
   then sees the ordering as dearer than it is.
4. **Distance to the geometric references**: 0% (syn-mem27-100) and 2.7% (yee-pec-box-1e6) - little
   left in the separators of regular grids; further factorization gains have to come from the
   numeric kernels (P9 factspeed) or from fewer factorizations.
5. **`deterministic = true`** still takes the P4 rule (AMD or METIS); hsolve's ND is reproducible
   and better - switching needs `tests/cpp/test_drvint_fix.cpp` to accept any deterministic
   ordering.
6. **Low-fill block cases** (4 syn-* cases of n <= 1e5): METIS is best, ND within 1-27% (<= 0.02 s).
7. Not measured: other thread counts than 16 for the new ND (the model scales by its fitted serial
   fraction), CPUs without AVX-512 (unaffected: the ordering is scalar code).
8. **The flow rule's threshold** (4.5) lies in a gap of the data (ratios 3.6 to 5.6); the small
   cases' time on a given order is predicted neither by its flops nor by any static tree statistic
   tried (section 8.3), so the rule is a guard against a regression, not a model of MUMPS'
   tree-parallel layer. A case between 3.6 and 5.6 may go either way by a few percent.
9. **wg-square-n289** (n = 289) is 1.6 ms (9%) slower than main with main's own pivot order in 30
   runs, the time in solves on identical factors (section 8.2); not traced.

