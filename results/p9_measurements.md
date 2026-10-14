# P9 integration: final measurements of the integrated build

P9 integration (branch `p9int` from p9-merge 90e7483 = main a831d79 + P9 "ordering", "factspeed" and
"mpi") makes the patched MUMPS source build the default build (user decision 2026-10-14) and measures
the integrated build end to end: against main a831d79 at 1e6 and on the 1e5 data cases, head to head
against tuned SLEPc, and its thread scaling for the "并行方案" section of the solution document. The
certificate is the default `exact` in every hsolve run unless a column says otherwise
(`hsolve-auto` = `certificate = auto`, `hsolve-nocert` = off).

**Answers.**

* **The default build** now compiles MUMPS 5.8.2 from the official tarball shipped in
  `third_party/mumps/` with the L0-blocking patch (portable flags), cached by its build inputs, and
  links the CLI, the Python module and the tests to it; `HS_MUMPS=env` falls back to the environment's
  conda-forge MUMPS, `HS_MUMPS_BLOCKING=mumps` to MUMPS' block sizes at run time (section 6).
* **n = 1e6 against main a831d79** (8 interleaved repetitions, every run valid, certificate exact):
  syn-mem27-100 111.2 -> 83.2 s (**1.34x**), yee-pec-box-1e6 88.2 -> 69.1 s (**1.28x**),
  wg-square-1e6 (2-D GHIEP) 5.94 -> 5.94 s (section 2.1). On the same (deterministic) pivot order the
  P9 block sizes with the patch contribute **1.056x / 1.053x** end to end and 1.065-1.083x per
  factorization (section 2.2) - the factspeed gate without SCOTCH's scatter, passed on both cases; the
  ordering contributes 1.27x / 1.21x.
* **n <= 3e5, 41 data cases** (3 repetitions): geometric mean **0.910**, 28 faster by > 3%, none
  slower by > 3% after a 10-repetition re-check of the seven not faster (geometric mean 0.999);
  ss-3dspectralwave2 312 -> 151 s (its factor now fits in core) (section 3).
* **Against tuned SLEPc** (16 cores, 5 repetitions, 160 runs valid): with the exact certificate
  syn-mem27-46 **1.77x**, yee-pec-box-1e5 **1.37x**, wg-square-1e5 **1.89x**, the interval
  **3.14x**, syn-mem27-100 **1.41x** (main: 1.04x), ss-qa8fk-qa8fm 0.96x, cav2d-rect-pert-1e5 0.74x,
  yee-pec-box-1e6 0.86x (main: 0.67x); without the certificate 0.94-3.15x, `certificate = auto`
  1.08x on yee-pec-box-1e6 (section 4).
* **Multithreading** (1 -> 16 threads on one socket, 5 repetitions; re-measured after the review,
  section 5): the default configuration at 16 threads against the fastest of six measured 1-thread
  configurations (the default and each of hsolve's orderings AMD / own ND / SCOTCH / METIS / PORD):
  3-D n ~ 1e5 cases **3.57-4.51x** (efficiency 0.22-0.28), complex / 2-D cases 2.25-2.71x; at equal
  pivot order (the 16-thread run's ordering at every thread count) 3-D **4.26-5.09x** (1 -> 2 threads
  1.72-1.91x), ordering and analysis 5.3-5.8x, factorizations 4.1-6.7x, certificate counts 4.4-5.4x,
  block solves 2.6-3.3x (memory bandwidth); yee-pec-box-1e6 4 -> 16 threads 1.82x. The first
  campaign's 4.2-5.4x divided by slower 1-thread orderings (section 5). At 1 thread the adaptive
  ordering (its model fitted at 16 threads) picks SCOTCH on two cases where METIS / PORD are 1.21x /
  1.27x faster: an open item outside the judged configuration.

## 1. Setup

* **Where.** gpu01 (2x Xeon Silver 4314, 2 x 16 cores, AVX-512, 251 GB, CentOS 7, glibc 2.17), the
  measurement slot: cores 16-31 = NUMA node 1, 16 threads, 32 GiB, one run at a time
  (`tools/run/guard.py --exclusive`, each slot call <= 2700 s, every solver run its own nested guard);
  builds, tests and evaluations in the build/test lane (cores 8-15). 2026-10-14 17:36-20:39, three
  chained scripts of consecutive slot calls (`$HOME/p9int/campaign.sh`, `campaign2.sh`,
  `campaign3.sh`), and the review fix 21:56-22:29 (`$HOME/p9int-fix/campaign_fix.sh`, two slot
  calls), nothing else on the slot's socket.
* **Binaries** (fixed copies in `$HOME/p9int/bin` on the cluster share):

  | label | what | md5 |
  |---|---|---|
  | `main` | main a831d79, `tools/dev/build.sh release` in `~/os2026-wt/p9-factspeed-main` on gpu01 (P9 factspeed's baseline copy `hsolve_main`): conda-forge MUMPS 5.8.2, MUMPS' block sizes, P5c ordering model | b1c973c5b627cbe08aff558ad74be3d7 |
  | `new` (`hsolve_int`) | this branch at b50a718 (the code of every later commit: those change tests, tools, scripts and documents only), `tools/dev/build.sh release` in `~/os2026-wt/p9int` - the default build: MUMPS 5.8.2 from `third_party/mumps/` + `patches/mumps-5.8.2-l0-blocking.patch`, nocona -O2 (install `$HOME/deps/mumps-cache/mumps-5.8.2-df9061400e43252d`), P9 block sizes by default, P9 ordering | e0d2ef6343cf642a8ee35efb89342d13 |
  | `blk` | `new` run with `HS_MUMPS_BLOCKING=mumps` (wrapper `hsolve_int_mumpsblk`: MUMPS' own block sizes; the patch's KEEP(498/499) unset, so the libraries behave as unpatched ones) - sections 2.2 and 3 | (= `new`) |

  Both are the CLI with the BLAS hook linked (route all), gcc 13.4 `-O3 -march=native`.
* **Validity.** Every timed run is evaluated by `tools/eval` (`python3 -m tools.eval.sweep`: backward
  errors at the case tol, completeness against the case reference, duplicates); a run is valid when it
  exited 0, claims success and the evaluation passes. Ratios and speedups use medians of valid runs and
  are given only when every run of both sides was valid.
* **Cases.** 1e6: `$HOME/p7_p7-membudget/data` (syn-mem27-100: HEP, 27-point 3-D, n = 1e6, k = 12;
  yee-pec-box-1e6: GHEP, Yee edge elements, n = 1 027 733, k = 10; wg-square-1e6: GHIEP waveguide,
  n = 998 001, k = 18), each with its config.json (CLI defaults otherwise: automatic memory budget from
  the 32-GiB guard cap, certificate exact). 1e5 and smaller: the 41 other data cases of `data/cases`.
* **Campaign** (slot calls in this order; 849 timed solver runs, every one valid; the review fix
  added part 5b, 360 runs, every one valid):

  | part | slot time | content | runs |
  |---|---|---|---|
  | 2.1 | 17:36-18:25 | 1e6, main vs new, 8 reps ABBA | 48 |
  | 3 | 18:25-19:03 | 41 data cases, main vs new, 3 reps ABBA | 246 |
  | 4 | 19:03-19:54 | compare.py: syn-mem27-100, yee-pec-box-1e6, five 1e5 nearest cases, the interval, 4 solvers x 5 reps | 160 |
  | 5 | 19:54-20:14 | thread scaling: six 1e5 cases x 1/2/4/8/16 threads, yee-pec-box-1e6 x 4/8/16, 3 reps | 99 |
  | 2.2 | 20:14-20:37 | 1e6, blk vs new, 4 reps ABBA | 16 |
  | 3 (re-check) | 20:37-20:39 | seven 1e5 cases, main vs new and blk vs new, 10 reps ABBA | 280 |
| 5b (review fix) | 21:56-22:29 | thread scaling again, `new`: six 1e5 cases, default at 1/2/4/8/16 threads, the 16-thread ordering fixed at 1/2/4/8/16 (three cases), every ordering at 1 thread, 5 reps; supersedes part 5's 1e5 tables | 360 |

## 2. n = 1e6 against main a831d79 (task 3a)

### 2.1 Eight interleaved repetitions (ABBA)

`tools/bench/ab_cases.py`, repetition r runs main then new for even r and new then main for odd r,
8 repetitions, 48 runs, all valid (`results/p9_measurements_1e6.json`):

| case | type | main s [min-max] | new s [min-max] | new / main | speedup | valid main / new | certificate |
|---|---|---|---|---|---|---|---|
| syn-mem27-100 | HEP | 111.21 [102.49-117.23] | 83.21 [82.55-83.39] | 0.748 | **1.336x** | 8/8 / 8/8 | exact |
| yee-pec-box-1e6 | GHEP | 88.24 [86.70-93.93] | 69.14 [68.96-69.31] | 0.784 | **1.276x** | 8/8 / 8/8 | exact |
| wg-square-1e6 | GHIEP | 5.938 [5.818-5.945] | 5.940 [5.853-5.999] | 1.000 | 1.00x | 8/8 / 8/8 | none (B indefinite) |

Every new run is faster than every main run on the two 3-D cases (syn 1.229-1.420x, yee
1.251-1.362x pairwise extremes). main's spread is its SCOTCH order (different every run, P8/P9); the
integrated build orders with the deterministic own nested dissection, so its runs agree within 1%.
wg-square-1e6 is a 2-D cross-section ordered by AMD in both builds (identical flops); its numeric
factorization is 0.7 s of 5.9 s, so the block sizes do not show.

Medians of an exclusive phase split of the report's five exclusive phases over the valid runs
(`tools/bench/thread_scaling.py` `phases`: analysis = factor_stats.analyze_s; factorization = the rest
of the factor phase, i.e. the main numeric factorization at sigma and the adaptive ordering's other
analyses; solves = factor_stats.solve_s with the number of right-hand sides; iteration = the rest of
the iteration phase; certificate = the two inertia-only counts and the InertiaCounter's analysis;
rest = setup + verify; ordering 3 = SCOTCH, 10 = own nested dissection, 0 = AMD):

| case | binary | total s | analysis | factorization | solves (RHS) | iteration | certificate | rest | flops | factor entries | ordering | peak RSS GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-100 | main | 111.21 | 5.94 | 30.13 | 16.04 (168) | 1.60 | 56.78 | 0.74 | 9.01e12 | 1.36e9 | 3 | 15.9 |
| syn-mem27-100 | new | 83.21 | 3.03 | 22.41 | 14.55 (168) | 1.60 | 40.91 | 0.74 | 6.72e12 | 1.19e9 | 10 | 13.1 |
| yee-pec-box-1e6 | main | 88.24 | 4.46 | 24.33 | 11.92 (136) | 2.27 | 44.83 | 0.52 | 6.87e12 | 1.18e9 | 3 | 13.0 |
| yee-pec-box-1e6 | new | 69.14 | 2.72 | 18.81 | 10.92 (136) | 2.27 | 33.93 | 0.52 | 5.45e12 | 1.04e9 | 10 | 11.3 |
| wg-square-1e6 | main | 5.94 | 0.54 | 0.67 | 2.34 (64) | 1.29 | - | 1.08 | 5.47e10 | 9.7e7 | 0 | 1.9 |
| wg-square-1e6 | new | 5.94 | 0.54 | 0.68 | 2.34 (64) | 1.29 | - | 1.08 | 5.47e10 | 9.7e7 | 0 | 1.9 |

The gain is in the three factorizations (main factorization + certificate: syn -23.6 s, yee -16.4 s),
the analysis (-2.9 s / -1.7 s) and the solves (-1.5 s / -1.0 s: fewer factor entries); peak RSS
-2.8 / -1.7 GiB. The exact certificate is still about half of the run (40.9 of 83.2 s, 33.9 of 69.1 s).

### 2.2 The share of the block sizes and the patch (same pivot order)

The integrated build against itself with MUMPS' own block sizes (`blk` = `HS_MUMPS_BLOCKING=mumps`:
KEEP(420) = 128, KEEP(57/58) = 2000 / 1000, the patch's KEEP(498/499) unset, so the patched libraries
behave as unpatched ones - P9 factspeed measured a rebuild alone at 1.000-1.005x). The own nested
dissection is deterministic, so both factorize the same pivot order with the same flops: this is the
factspeed gate without SCOTCH's scatter, the question P9 factspeed's review left open
(`results/p9_factspeed.md` §4.5). 4 ABBA repetitions, 16 runs, all valid
(`results/p9_measurements_blocking_1e6.json`):

| case | blk s [min-max] | new s [min-max] | speedup | factorization blk / new | certificate blk / new | solves blk / new |
|---|---|---|---|---|---|---|
| syn-mem27-100 | 87.56 [87.33-88.51] | 82.88 [82.44-83.31] | **1.056x** | 24.10 / 22.40 (1.076x) | 44.28 / 40.87 (1.083x) | 13.92 / 14.25 |
| yee-pec-box-1e6 | 72.82 [72.77-73.06] | 69.13 [69.03-69.24] | **1.053x** | 20.04 / 18.81 (1.065x) | 36.39 / 33.87 (1.074x) | 10.92 / 10.96 |

Every new run is faster than every blk run (ranges disjoint). Per factorization the P9 block sizes
with the patch give 1.065-1.083x - P9 factspeed's 1.073-1.079x per count on fixed orders - and end to
end 1.053-1.056x, past the 1.05x bar on both cases with the pivot order fixed (factspeed's pooled
estimate at equal flops was 1.055x on yee-pec-box-1e6, 1.061x on syn-mem27-100). Together with
section 2.1: the ordering (P9 "ordering", blk against main) accounts for 1.27x / 1.21x and the block
sizes for 1.056x / 1.053x of the 1.34x / 1.28x (the products, 1.34x / 1.28x, agree with the measured
totals; different sessions of the same day and slot).

## 3. n <= 3e5: the 41 other data cases against main a831d79 (task 3b)

`tools/bench/ab_cases.py`, every data case except wg-square-1e6 (section 2), 3 interleaved
repetitions (ABBA), 246 runs, all valid (`results/p9_measurements_1e5.json`); each case's default
configuration (`--threads 16`, certificate exact where B is definite - GHIEP cases have none).
"numeric #" = factorizations of the main object (the certificate's counts on a separate
InertiaCounter, the large 3-D cases, are not in it); "block p" = the reported block width.

| case | n | type | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | 0.293 [0.292-0.322] | 0.295 [0.295-0.305] | 1.008 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | 0.042 [0.042-0.043] | 0.042 [0.041-0.042] | 0.991 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | 2.833 [2.793-2.880] | 2.480 [2.410-2.540] | 0.875 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-3dspectralwave2 | 292008 | HEP | 312.426 [309.590-336.684] | 151.227 [151.005-151.322] | 0.484 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-ge87h76 | 112985 | HEP | 56.293 [54.626-58.209] | 48.353 [48.239-48.390] | 0.859 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-helm3d01 | 32226 | HEP | 0.548 [0.544-0.565] | 0.508 [0.505-0.518] | 0.927 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | 1.687 [1.671-1.699] | 1.491 [1.484-1.504] | 0.884 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-si34h36 | 97569 | HEP | 38.974 [38.460-40.184] | 34.314 [34.239-34.402] | 0.880 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-tem152078 | 152078 | GHEP | 5.407 [5.386-5.429] | 4.344 [4.338-4.386] | 0.803 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | 0.579 [0.578-0.588] | 0.557 [0.552-0.594] | 0.961 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-multiplicity-real-1e4 | 10000 | GHEP | 0.129 [0.121-0.139] | 0.119 [0.113-0.120] | 0.920 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | 1.289 [1.286-1.307] | 0.983 [0.970-0.998] | 0.763 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-ghep-zero-cluster-complex-1e4 | 10000 | GHEP | 0.288 [0.281-0.319] | 0.264 [0.263-0.269] | 0.919 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | 0.057 [0.057-0.061] | 0.061 [0.056-0.062] | 1.068 **slower** | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | 0.138 [0.135-0.139] | 0.136 [0.134-0.137] | 0.980 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | 2.059 [2.055-2.065] | 1.767 [1.764-1.774] | 0.858 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem125-46 | 97336 | HEP | 11.168 [11.092-11.193] | 10.452 [10.398-10.453] | 0.936 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-mem27-46 | 97336 | HEP | 2.355 [2.340-2.363] | 1.995 [1.991-2.085] | 0.847 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-multiplicity-complex-1e4 | 10000 | HEP | 0.199 [0.197-0.223] | 0.194 [0.188-0.201] | 0.975 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-multiplicity-real-1e4 | 10000 | HEP | 0.091 [0.089-0.095] | 0.079 [0.078-0.083] | 0.869 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-near-singular-complex-1e4 | 10000 | HEP | 0.223 [0.218-0.225] | 0.212 [0.212-0.215] | 0.954 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | 2.484 [2.466-2.495] | 2.229 [2.212-2.275] | 0.898 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-real-1e4 | 10000 | HEP | 0.099 [0.094-0.103] | 0.083 [0.082-0.099] | 0.837 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-wide-range-complex-1e4 | 10000 | HEP | 0.177 [0.175-0.179] | 0.171 [0.167-0.197] | 0.968 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-wide-range-real-1e4 | 10000 | HEP | 0.083 [0.082-0.114] | 0.073 [0.070-0.084] | 0.887 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-zero-cluster-complex-1e4 | 10000 | HEP | 0.291 [0.290-0.293] | 0.286 [0.281-0.287] | 0.980 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e4 | 10000 | HEP | 0.120 [0.118-0.122] | 0.106 [0.104-0.108] | 0.887 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | 1.225 [1.211-1.240] | 0.929 [0.921-0.933] | 0.759 | 3/3 / 3/3 | 6 / 6 | 4 / 4 | exact |
| wg-halfslab-1e4 | 9801 | GHIEP | 0.080 [0.071-0.082] | 0.072 [0.072-0.078] | 0.898 | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| wg-halfslab-1e5 | 99225 | GHIEP | 0.710 [0.699-0.757] | 0.746 [0.718-0.750] | 1.050 **slower** | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-1e3 | 961 | GHIEP | 0.023 [0.020-0.043] | 0.025 [0.019-0.039] | 1.107 **slower** | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| wg-square-1e5 | 99225 | GHIEP | 0.484 [0.481-0.520] | 0.485 [0.483-0.486] | 1.004 | 3/3 / 3/3 | 1 / 1 | 2 / 2 | none |
| wg-square-n289 | 289 | GHIEP | 0.013 [0.012-0.014] | 0.013 [0.013-0.015] | 0.965 | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| wg-square-pert-1e4 | 9801 | GHIEP | 0.063 [0.060-0.063] | 0.064 [0.061-0.068] | 1.019 | 3/3 / 3/3 | 1 / 1 | 1 / 1 | none |
| yee-bloch-1e4 | 10080 | HEP | 0.831 [0.815-0.851] | 0.820 [0.811-0.829] | 0.987 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | 0.192 [0.191-0.201] | 0.190 [0.189-0.192] | 0.989 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | 2.896 [2.880-2.902] | 2.447 [2.430-2.495] | 0.845 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | 0.211 [0.203-0.215] | 0.214 [0.202-0.304] | 1.016 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | 2.238 [2.211-2.243] | 1.910 [1.902-1.984] | 0.853 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | 0.900 [0.900-0.907] | 0.833 [0.829-0.834] | 0.926 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-spheres-1e4 | 10125 | GHEP | 0.960 [0.956-0.964] | 0.905 [0.881-0.920] | 0.942 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |

Geometric mean of new / main over the 41 cases: **0.910** (HEP 0.888 over 24, GHEP 0.908 over 11,
GHIEP 1.005 over 6; cases of >= 1 s 0.816 over 14, smaller ones 0.962 over 27); faster by > 3%: 28,
slower by > 3% in this screen: 3 (syn-lap2d-100 +4 ms, wg-halfslab-1e5 +36 ms, wg-square-1e3 +2 ms;
re-checked below). Summed medians 451.2 -> 272.5 s (without ss-3dspectralwave2 138.7 -> 121.2 s).

* **Where it comes from.** The adaptive ordering changed on seven cases, all SCOTCH -> own nested
  dissection (ss-2cubes_sphere, ss-3dspectralwave2, ss-ge87h76, ss-si34h36, syn-mem125-46,
  yee-pec-box-1e5, yee-pec-cube-32: 0.48-0.94); the other 3-D cases already used the dissection in
  main and gain from its faster analysis (taskgroup, compression: syn-mem27-46 analysis 0.62 -> 0.29 s,
  ss-tem152078 1.33 -> 0.41 s) and from the block sizes; the 2-D and GHIEP cases are ordered by AMD in
  both builds and change within noise.
* **ss-3dspectralwave2** (n = 292 008, the largest data case): main's SCOTCH factor needs 29.1 GB in
  core, over the 25.5 GB cap of the automatic budget, so main factorizes out of core and its 144
  solves take 124 s; the dissection's factor (24.6 GB estimate, 1.73e9 entries against 1.90e9)
  stays in core: 312.4 -> 151.2 s (2.07x), peak RSS 13.7 -> 18.6 GiB.
* **ss-tem152078** (the P5c regression, analysis-dominated): 5.41 -> 4.34 s (P9 ordering's
  compression).

**Re-check** of the seven cases not faster (ratio >= 1.0) in this screen, 10 ABBA repetitions each,
140 runs against main and 140 against `blk` (section 2.2), all valid
(`results/p9_measurements_1e5_recheck.json`):

| case | type | main s [min-max] | new s [min-max] | new / main | blk s [min-max] | new s [min-max] | new / blk |
|---|---|---|---|---|---|---|---|
| syn-lap2d-100 | HEP | 0.061 [0.053-0.088] | 0.059 [0.053-0.075] | 0.982 | 0.059 [0.055-0.108] | 0.060 [0.058-0.085] | 1.024 |
| wg-halfslab-1e5 | GHIEP | 0.736 [0.700-0.786] | 0.748 [0.701-0.757] | 1.017 | 0.746 [0.706-0.757] | 0.716 [0.702-0.749] | 0.960 |
| wg-square-1e3 | GHIEP | 0.023 [0.019-0.032] | 0.023 [0.019-0.024] | 0.962 | 0.024 [0.022-0.026] | 0.024 [0.020-0.027] | 0.994 |
| wg-square-pert-1e4 | GHIEP | 0.064 [0.059-0.081] | 0.065 [0.061-0.068] | 1.010 | 0.064 [0.061-0.074] | 0.063 [0.060-0.080] | 0.991 |
| wg-square-1e5 | GHIEP | 0.482 [0.478-0.517] | 0.486 [0.472-0.504] | 1.007 | 0.485 [0.468-0.525] | 0.487 [0.484-0.509] | 1.003 |
| yee-pec-cube-16 | HEP | 0.208 [0.202-0.254] | 0.210 [0.202-0.254] | 1.009 | 0.209 [0.205-0.236] | 0.209 [0.201-0.264] | 0.999 |
| cav2d-rect-pert-1e5 | GHEP | 0.293 [0.290-0.322] | 0.296 [0.293-0.298] | 1.008 | 0.297 [0.295-0.299] | 0.296 [0.291-0.305] | 0.997 |

None is slower by more than 3% over 10 repetitions (geometric mean 0.999 against main, 0.995 against
blk); the screen's three were noise of 2-36 ms runs (wg-halfslab-1e5 also solves 88-98
right-hand sides from run to run in every build: its GHIEP Krylov path depends on rounding). These 2-D / small cases
are ordered by AMD in both builds; the block sizes neither help nor hurt them (blk against new within
+-4%, in both directions).

## 4. Head to head against tuned SLEPc (task 3c)

`tools/bench/compare.py --solvers hsolve,hsolve-auto,hsolve-nocert,slepc --slepc-profile best
--reps 5` (solvers interleaved with rotating order, every run its own nested guard, tools/eval on every
run): the six 1e5 configurations of `results/cmp-p5c-best*.md` with the 16-core profiles of
`tools/bench/slepc_profiles.json`, and the two 3-D 1e6 cases with `results/cmp-p9-1e6-profiles.json`
(the P9 mpi review's profiles: the 1e5 winners re-screened at 1e6, `eps_ncv = 3k`). 160 runs, all
valid. Solver time = hsolve `time_total_s` / SLEPc `-log_view` setup + solve + vectors (same scope).
Files: `results/cmp-p9int-best.md` / `.json`, `results/cmp-p9int-best-interval.md` / `.json`,
`results/cmp-p9int-1e6.md` / `.json`.

| case | hsolve s | SLEPc s | hsolve | hsolve-auto | hsolve-nocert | before (same method) |
|---|---|---|---|---|---|---|
| syn-mem27-46 | 1.99 | 3.51 | **1.77x** | 2.57x | 3.15x | P5c 1.45x / 1.96x / 2.34x |
| yee-pec-box-1e5 | 2.44 | 3.34 | **1.37x** | 1.37x | 2.07x | P5c 1.14x / 1.14x / 1.60x |
| wg-square-1e5 | 0.48 | 0.91 | **1.89x** | 1.90x | 1.89x | P5c 1.84x / 1.82x / 1.83x |
| ss-qa8fk-qa8fm | 1.49 | 1.43 | 0.96x | 0.96x | 1.31x | P5c 0.89x / 0.89x / 1.18x |
| cav2d-rect-pert-1e5 | 0.29 | 0.22 | 0.74x | 0.74x | 0.94x | P5c 0.71x / 0.71x / 0.94x |
| yee-pec-box-1e5 interval [40, 200] | 3.30 | 10.36 | **3.14x** | 3.13x | 3.13x | P5c 2.70x / 2.72x / 2.71x |
| syn-mem27-100 (1e6) | 82.46 | 116.00 | **1.41x** | 2.01x | 2.74x | main a831d79 1.04x / 1.60x / 2.13x |
| yee-pec-box-1e6 (1e6) | 69.13 | 59.30 | 0.86x | 1.08x | 1.70x | main a831d79 0.67x / 0.91x / 1.38x |

"Before": P5c = `results/cmp-p5c-best*.md` (main 34cbfd4, c003 / c005), main a831d79 =
`results/cmp-p9-1e6.md` (c005), both with the same SLEPc profiles; SLEPc's own times then and now
agree within 1% (116.00 vs 115.68 s, 59.30 vs 59.58 s, 3.51 vs 3.48 s, 3.34 vs 3.36 s).

* With the exact certificate the integrated build is faster than tuned SLEPc on 5 of the 8
  configurations (P5c: 4 of the 6 at 1e5; main at 1e6: level on syn, 0.67x on yee).
* The three it loses are the certificate's two inertia-only factorizations: ss-qa8fk-qa8fm and
  cav2d-rect-pert-1e5 factorize 3 times against SLEPc's once (without the certificate 1.31x / 0.94x;
  P8 measured every cheaper exact count negative), and on yee-pec-box-1e6 the two counts are 33.9 of
  69.1 s while SLEPc's best configuration factorizes with BLR (`cntl_7 = 1e-10`, 4 x 4 ranks,
  ParMETIS) so that its 59 single-RHS solves cost 26.8 s; hsolve's numeric factorization (17.8 s) is
  faster than SLEPc's (24.6 s), and without the certificate hsolve is 1.70x faster, with the
  randomized check (auto) 1.08x.
* `certificate = auto` changes the result on syn-mem27-46 / syn-mem27-100 / yee-pec-box-1e6 (the
  randomized exclusion check, B = I or a proven kappa(B)); elsewhere it keeps the exact counts.

## 5. Multithreading: speedup and parallel efficiency (task 3d)

**Review correction.** Part 5 of the campaign (19:54-20:14, 3 repetitions) ran each case's default
configuration at every thread count and divided by its own 1-thread time. The default's adaptive
ordering prices the parallel nested dissection by the thread count (docs/DESIGN.md §3.6-3b): at
1 thread it took SCOTCH on syn-mem27-46 and ss-qa8fk-qa8fm (AMD at 1-2 threads on
syn-near-singular-complex-1e5), from 2 threads on (4 there) the own nested dissection. Those speedups
therefore divided a slower serial ordering by the parallel one: the reported 5.39x / 4.55x at 16
threads, 1 -> 2 threads 1.97x / 2.08x (superlinear) and "ordering and analysis 7.90x / 8.31x" (SCOTCH
at 1 thread against the dissection at 16), and the statement that the 1-thread baseline was
"hsolve's own best choice at 1 thread" had not been measured - it is not (below). The 1e5 tables
are re-measured here (part 5b); part 5's raw tables stay in `results/p9_measurements_threads.json`.

**Method (part 5b).** `tools/bench/thread_scaling.py run --threads 1,2,4,8,16 --reps 0,1,2,3,4` in
the slot with configurations (`--variant`, docs in the tool): the case's default at every T; on the
three cases whose default changes ordering with T, the 16-thread runs' ordering (own ND, `ordering =
10`) fixed at every T (`nd`); and at T = 1 each of the adaptive ordering's candidates fixed (AMD 0,
own ND 10, SCOTCH 3, METIS 5, PORD 4). Each run pinned to the first T cores of the held slot (the
rest of the socket idle), certificate exact, thread counts up for even and down for odd repetitions,
the configurations forward / backward at each T; medians of 5, all 360 runs valid by tools/eval
(`results/p9_measurements_threads_orderings.json`). Same binary as everywhere in this file (`new`,
md5 e0d2ef63..., checked before and after). Two speedups:

* **against the fastest 1-thread configuration**: t_fastest(1) / t_default(T), t_fastest(1) the
  fastest of the six measured 1-thread configurations (the default and the five fixed orderings) -
  what 16 threads give a user of the default configuration over the best serial run hsolve can make;
  efficiency = that / T;
* **at equal pivot order**: t(1) / t(T) of one configuration that factorizes the same pivot order at
  every T (the 16-thread runs' ordering; the factor entries are equal at every T): the parallel
  speedup of one computation. On yee-pec-box-1e5, wg-square-1e5 and cav2d-rect-pert-1e5 that is the
  default itself (one ordering at every T). The phase speedups are taken here.

The phase columns split the report's five exclusive phases (they add up to the total): analysis
(ordering + symbolic analysis of the main object), factorization (the rest of the factor phase:
numeric factorization at sigma, the adaptive ordering's other candidate analyses), solves (the Krylov
iteration's block solves), iteration (orthogonalization, restarts), certificate (inertia counts),
rest (setup + verify).

Summary (solver time in s, medians of 5):

| case | fastest at 1 thread | default: 1 | 2 | 4 | 8 | 16 | **default 16 vs fastest 1** | efficiency | equal order 1 -> 16 | **speedup** | 1 -> 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-mem27-46 (3-D HEP) | METIS 9.06 | 11.01 (SCOTCH) | 5.58 | 3.38 | 2.33 | 2.01 | **4.51x** | 0.28 | ND 9.78 -> 1.92 | **5.09x** | 1.78x |
| yee-pec-box-1e5 (3-D GHEP, Yee) | ND 10.52 | 10.58 (ND) | 6.14 | 3.76 | 2.72 | 2.49 | **4.23x** | 0.26 | ND (default) 10.58 -> 2.49 | **4.26x** | 1.72x |
| ss-qa8fk-qa8fm (3-D GHEP, acoustic) | PORD 5.51 | 6.97 (SCOTCH) | 3.34 | 2.13 | 1.61 | 1.54 | **3.57x** | 0.22 | ND 6.25 -> 1.42 | **4.40x** | 1.91x |
| syn-near-singular-complex-1e5 (complex HEP) | AMD 4.99 | 5.02 (AMD) | 4.21 | 3.44 | 2.71 | 2.22 | 2.25x | 0.14 | ND 5.41 -> 1.92 | 2.82x | 1.29x |
| wg-square-1e5 (2-D GHIEP) | AMD 1.42 | 1.42 (AMD) | 1.02 | 0.72 | 0.55 | 0.52 | 2.71x | 0.17 | AMD (default) 1.42 -> 0.52 | 2.71x | 1.40x |
| cav2d-rect-pert-1e5 (2-D GHEP) | AMD 0.91 | 0.91 (AMD) | 0.74 | 0.57 | 0.42 | 0.34 | 2.65x | 0.17 | AMD (default) 0.91 -> 0.34 | 2.66x | 1.23x |
| yee-pec-box-1e6 (3-D GHEP, n ~ 1e6; part 5, 3 reps, from 4 threads) | not screened | - | - | 125.65 (ND) | 85.36 | 69.16 | 1.82x vs 4 | 0.45 vs 4 | ND (default) 4 -> 16: 125.65 -> 69.16 | 1.82x vs 4 | - |

The 1-thread configurations (s; min and max within 0.8% of the median everywhere; fastest in bold):

| case | default (its choice) | AMD | own ND | SCOTCH | METIS | PORD |
|---|---|---|---|---|---|---|
| syn-mem27-46 | 11.01 (SCOTCH) | 18.89 | 9.78 | 10.93 | **9.06** | 10.10 |
| yee-pec-box-1e5 | 10.58 (ND) | 42.21 | **10.52** | 12.06 | 11.23 | 17.23 |
| ss-qa8fk-qa8fm | 6.97 (SCOTCH) | 10.04 | 6.25 | 6.89 | 6.07 | **5.51** |
| syn-near-singular-complex-1e5 | 5.02 (AMD) | **4.99** | 5.41 | 10.12 | 7.71 | 8.56 |
| wg-square-1e5 | **1.42** (AMD) | 1.42 | 2.32 | 2.35 | 1.95 | 1.85 |
| cav2d-rect-pert-1e5 | 0.91 (AMD) | **0.91** | 1.55 | 1.34 | 1.36 | 1.21 |

Per case: the default configuration and, where its ordering changes with T, the equal-order one
(`*` marks a row whose ordering differs from the 1-thread row's; "vs fastest" = against the fastest
1-thread configuration; the other 1-thread configurations' tables are in the JSON):

**syn-mem27-46** (n = 97336, HEP; fastest at T = 1: metis (METIS) 9.060 s)

*default configuration (adaptive ordering)* (certificate exact, block p 8)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.007 [10.998-11.026] | 1.00x | 1.00 | 0.82x | 2.343 | 2.664 | 0.760 | 0.112 | 4.978 | 0.142 | SCOTCH | 54727876 | 5/5 | 0.68 |
| 2 | 5.584 [5.565-5.594] | 1.97x | 0.99 | 1.62x | 0.901 | 1.434 | 0.432 | 0.067 | 2.635 | 0.108 | ND * | 51738000 | 5/5 | 0.69 |
| 4 | 3.376 [3.346-3.378] | 3.26x | 0.81 | 2.68x | 0.527 | 0.871 | 0.276 | 0.044 | 1.551 | 0.096 | ND * | 51738000 | 5/5 | 0.69 |
| 8 | 2.327 [2.311-2.355] | 4.73x | 0.59 | 3.89x | 0.354 | 0.617 | 0.227 | 0.034 | 1.024 | 0.076 | ND * | 51738000 | 5/5 | 0.79 |
| 16 | 2.011 [1.985-2.072] | 5.47x | 0.34 | 4.51x | 0.290 | 0.542 | 0.229 | 0.029 | 0.864 | 0.066 | ND * | 51738000 | 5/5 | 0.86 |

Phase speedups 1 -> 16 threads: analysis 8.08x, factorization 4.92x, solves 3.33x, iteration 3.89x, certificate 5.76x, rest 2.15x - orderings differ (SCOTCH at 1, ND at 16): not the parallel speedup of one computation

*equal pivot order: the 16-thread run's ordering (own ND, ordering 10) at every T* (certificate exact, block p 8)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.776 [9.764-9.778] | 1.00x | 1.00 | 0.93x | 1.676 | 2.405 | 0.752 | 0.112 | 4.684 | 0.143 | ND | 51738000 | 5/5 | 0.65 |
| 2 | 5.506 [5.498-5.517] | 1.78x | 0.89 | 1.65x | 0.909 | 1.362 | 0.434 | 0.063 | 2.634 | 0.103 | ND | 51738000 | 5/5 | 0.65 |
| 4 | 3.283 [3.269-3.285] | 2.98x | 0.74 | 2.76x | 0.534 | 0.799 | 0.277 | 0.045 | 1.548 | 0.078 | ND | 51738000 | 5/5 | 0.69 |
| 8 | 2.242 [2.235-2.251] | 4.36x | 0.55 | 4.04x | 0.354 | 0.543 | 0.227 | 0.034 | 1.025 | 0.059 | ND | 51738000 | 5/5 | 0.79 |
| 16 | 1.922 [1.918-1.928] | 5.09x | 0.32 | 4.71x | 0.291 | 0.462 | 0.228 | 0.029 | 0.863 | 0.049 | ND | 51738000 | 5/5 | 0.84 |

Phase speedups 1 -> 16 threads: analysis 5.75x, factorization 5.20x, solves 3.29x, iteration 3.81x, certificate 5.42x, rest 2.94x

**yee-pec-box-1e5** (n = 109744, GHEP; fastest at T = 1: nd (ND) 10.523 s)

*default configuration (adaptive ordering)* (certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10.580 [10.570-10.634] | 1.00x | 1.00 | 0.99x | 1.602 | 2.332 | 1.713 | 0.462 | 4.362 | 0.107 | ND | 49568685 | 5/5 | 0.58 |
| 2 | 6.136 [6.095-6.146] | 1.72x | 0.86 | 1.71x | 0.866 | 1.380 | 1.036 | 0.261 | 2.494 | 0.096 | ND | 49568685 | 5/5 | 0.58 |
| 4 | 3.763 [3.712-3.765] | 2.81x | 0.70 | 2.80x | 0.506 | 0.850 | 0.713 | 0.162 | 1.446 | 0.080 | ND | 49568685 | 5/5 | 0.61 |
| 8 | 2.720 [2.694-2.745] | 3.89x | 0.49 | 3.87x | 0.335 | 0.610 | 0.635 | 0.111 | 0.967 | 0.064 | ND | 49568685 | 5/5 | 0.65 |
| 16 | 2.486 [2.424-2.513] | 4.26x | 0.27 | 4.23x | 0.276 | 0.565 | 0.666 | 0.088 | 0.834 | 0.048 | ND | 49568685 | 5/5 | 0.75 |

Phase speedups 1 -> 16 threads: analysis 5.80x, factorization 4.13x, solves 2.57x, iteration 5.23x, certificate 5.23x, rest 2.24x

**ss-qa8fk-qa8fm** (n = 66127, GHEP; fastest at T = 1: pord (PORD) 5.505 s)

*default configuration (adaptive ordering)* (certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.971 [6.970-6.978] | 1.00x | 1.00 | 0.79x | 1.695 | 1.516 | 1.145 | 0.391 | 2.059 | 0.163 | SCOTCH | 28441088 | 5/5 | 0.38 |
| 2 | 3.339 [3.337-3.359] | 2.09x | 1.04 | 1.65x | 0.584 | 0.803 | 0.656 | 0.218 | 0.966 | 0.114 | ND * | 24696412 | 5/5 | 0.38 |
| 4 | 2.130 [2.101-2.131] | 3.27x | 0.82 | 2.58x | 0.348 | 0.510 | 0.454 | 0.151 | 0.563 | 0.103 | ND * | 24696412 | 5/5 | 0.41 |
| 8 | 1.613 [1.589-1.638] | 4.32x | 0.54 | 3.41x | 0.242 | 0.385 | 0.394 | 0.119 | 0.393 | 0.079 | ND * | 24696412 | 5/5 | 0.44 |
| 16 | 1.544 [1.498-1.573] | 4.51x | 0.28 | 3.57x | 0.206 | 0.377 | 0.419 | 0.100 | 0.377 | 0.068 | ND * | 24696412 | 5/5 | 0.75 |

Phase speedups 1 -> 16 threads: analysis 8.24x, factorization 4.02x, solves 2.73x, iteration 3.90x, certificate 5.46x, rest 2.40x - orderings differ (SCOTCH at 1, ND at 16): not the parallel speedup of one computation

*equal pivot order: the 16-thread run's ordering (own ND, ordering 10) at every T* (certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.248 [6.241-6.266] | 1.00x | 1.00 | 0.88x | 1.086 | 1.897 | 1.078 | 0.391 | 1.636 | 0.162 | ND | 24696412 | 5/5 | 0.35 |
| 2 | 3.272 [3.270-3.273] | 1.91x | 0.95 | 1.68x | 0.590 | 0.730 | 0.660 | 0.219 | 0.965 | 0.108 | ND | 24696412 | 5/5 | 0.38 |
| 4 | 2.037 [2.035-2.040] | 3.07x | 0.77 | 2.70x | 0.351 | 0.439 | 0.454 | 0.152 | 0.562 | 0.080 | ND | 24696412 | 5/5 | 0.40 |
| 8 | 1.513 [1.510-1.523] | 4.13x | 0.52 | 3.64x | 0.247 | 0.308 | 0.395 | 0.119 | 0.392 | 0.053 | ND | 24696412 | 5/5 | 0.42 |
| 16 | 1.420 [1.415-1.487] | 4.40x | 0.27 | 3.88x | 0.204 | 0.283 | 0.417 | 0.098 | 0.374 | 0.044 | ND | 24696412 | 5/5 | 0.73 |

Phase speedups 1 -> 16 threads: analysis 5.32x, factorization 6.69x, solves 2.59x, iteration 3.98x, certificate 4.38x, rest 3.64x

**syn-near-singular-complex-1e5** (n = 100000, HEP; fastest at T = 1: amd (AMD) 4.990 s)

*default configuration (adaptive ordering)* (certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.020 [5.011-5.028] | 1.00x | 1.00 | 0.99x | 0.309 | 0.991 | 1.418 | 0.848 | 0.966 | 0.488 | AMD | 6400988 | 5/5 | 0.40 |
| 2 | 4.212 [4.172-4.230] | 1.19x | 0.60 | 1.18x | 0.300 | 0.893 | 1.376 | 0.501 | 0.858 | 0.282 | AMD | 6400988 | 5/5 | 0.39 |
| 4 | 3.438 [3.412-3.455] | 1.46x | 0.36 | 1.45x | 0.408 | 0.917 | 0.987 | 0.336 | 0.580 | 0.210 | ND * | 9598562 | 5/5 | 0.53 |
| 8 | 2.709 [2.674-2.750] | 1.85x | 0.23 | 1.84x | 0.395 | 0.738 | 0.741 | 0.277 | 0.409 | 0.158 | ND * | 9598562 | 5/5 | 0.55 |
| 16 | 2.221 [2.209-2.248] | 2.26x | 0.14 | 2.25x | 0.395 | 0.625 | 0.531 | 0.252 | 0.295 | 0.123 | ND * | 9598562 | 5/5 | 0.68 |

Phase speedups 1 -> 16 threads: analysis 0.78x, factorization 1.58x, solves 2.67x, iteration 3.36x, certificate 3.28x, rest 3.95x - orderings differ (AMD at 1, ND at 16): not the parallel speedup of one computation

*equal pivot order: the 16-thread run's ordering (own ND, ordering 10) at every T* (certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.407 [5.405-5.426] | 1.00x | 1.00 | 0.92x | 0.439 | 1.027 | 1.587 | 0.845 | 1.023 | 0.488 | ND | 9598562 | 5/5 | 0.41 |
| 2 | 4.201 [4.195-4.236] | 1.29x | 0.64 | 1.19x | 0.426 | 0.858 | 1.291 | 0.500 | 0.841 | 0.283 | ND | 9598562 | 5/5 | 0.43 |
| 4 | 3.053 [3.036-3.094] | 1.77x | 0.44 | 1.63x | 0.409 | 0.595 | 0.974 | 0.336 | 0.565 | 0.185 | ND | 9598562 | 5/5 | 0.49 |
| 8 | 2.372 [2.368-2.383] | 2.28x | 0.28 | 2.10x | 0.394 | 0.421 | 0.739 | 0.275 | 0.409 | 0.139 | ND | 9598562 | 5/5 | 0.54 |
| 16 | 1.919 [1.899-1.946] | 2.82x | 0.18 | 2.60x | 0.394 | 0.310 | 0.534 | 0.257 | 0.293 | 0.115 | ND | 9598562 | 5/5 | 0.66 |

Phase speedups 1 -> 16 threads: analysis 1.11x, factorization 3.31x, solves 2.97x, iteration 3.29x, certificate 3.49x, rest 4.26x

**wg-square-1e5** (n = 99225, GHIEP; fastest at T = 1: default (AMD) 1.421 s)

*default configuration (adaptive ordering)* (certificate none, block p 2)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.421 [1.415-1.430] | 1.00x | 1.00 | 1.00x | 0.053 | 0.206 | 0.675 | 0.290 | 0.000 | 0.197 | AMD | 7085790 | 5/5 | 0.17 |
| 2 | 1.015 [1.002-1.039] | 1.40x | 0.70 | 1.40x | 0.061 | 0.159 | 0.481 | 0.178 | 0.000 | 0.151 | AMD | 7085790 | 5/5 | 0.17 |
| 4 | 0.725 [0.686-0.736] | 1.96x | 0.49 | 1.96x | 0.067 | 0.109 | 0.308 | 0.132 | 0.000 | 0.110 | AMD | 7085790 | 5/5 | 0.18 |
| 8 | 0.554 [0.519-0.570] | 2.56x | 0.32 | 2.56x | 0.055 | 0.080 | 0.217 | 0.108 | 0.000 | 0.086 | AMD | 7085790 | 5/5 | 0.20 |
| 16 | 0.523 [0.511-0.530] | 2.71x | 0.17 | 2.71x | 0.065 | 0.078 | 0.204 | 0.097 | 0.000 | 0.079 | AMD | 7085790 | 5/5 | 0.31 |

Phase speedups 1 -> 16 threads: analysis 0.81x, factorization 2.64x, solves 3.30x, iteration 2.98x, rest 2.49x

**cav2d-rect-pert-1e5** (n = 99459, GHEP; fastest at T = 1: amd (AMD) 0.912 s)

*default configuration (adaptive ordering)* (certificate exact, block p 2)

| threads | total s [min-max] | speedup | efficiency | vs fastest | analysis | factorization | solves | iteration | certificate | rest | ordering | factor entries | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.914 [0.912-0.916] | 1.00x | 1.00 | 1.00x | 0.031 | 0.153 | 0.291 | 0.165 | 0.197 | 0.077 | AMD | 3871432 | 5/5 | 0.10 |
| 2 | 0.744 [0.717-0.746] | 1.23x | 0.61 | 1.23x | 0.044 | 0.124 | 0.251 | 0.094 | 0.162 | 0.066 | AMD | 3871432 | 5/5 | 0.13 |
| 4 | 0.568 [0.526-0.575] | 1.61x | 0.40 | 1.61x | 0.045 | 0.103 | 0.182 | 0.060 | 0.119 | 0.057 | AMD | 3871432 | 5/5 | 0.14 |
| 8 | 0.418 [0.387-0.441] | 2.19x | 0.27 | 2.18x | 0.037 | 0.083 | 0.118 | 0.045 | 0.087 | 0.046 | AMD | 3871432 | 5/5 | 0.17 |
| 16 | 0.344 [0.332-0.348] | 2.66x | 0.17 | 2.65x | 0.044 | 0.080 | 0.076 | 0.040 | 0.061 | 0.039 | AMD | 3871432 | 5/5 | 0.23 |

Phase speedups 1 -> 16 threads: analysis 0.69x, factorization 1.90x, solves 3.84x, iteration 4.10x, certificate 3.23x, rest 1.98x

**yee-pec-box-1e6** (n = 1027733, GHEP, certificate exact, block p 8)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 125.650 [125.644-125.725] | 1.00x | 1.00 | 6.175 | 35.606 | 13.059 | 2.603 | 67.591 | 0.625 | ND | 3/3 | 11.43 |
| 8 | 85.363 [85.327-85.597] | 1.47x | 0.74 | 3.794 | 23.843 | 10.747 | 2.287 | 44.107 | 0.571 | ND | 3/3 | 11.86 |
| 16 | 69.160 [69.118-69.180] | 1.82x | 0.45 | 2.734 | 18.793 | 10.911 | 2.273 | 33.908 | 0.525 | ND | 3/3 | 11.32 |

Phase speedups 4 -> 16 threads: analysis 2.26x, factorization 1.89x, solves 1.20x, iteration 1.14x, certificate 1.99x, rest 1.19x

(The 1e6 case's factor entries are 1 042 506 397 at 4, 8 and 16 threads: one pivot order. Its
4-thread baseline was not screened against the other orderings: their factors need 1.29-1.30x the own
dissection's flops on this case (SCOTCH 7.01e12, METIS 7.05e12 against 5.43e12, `results/p9_ordering.md`
§3) and the factorizations (main and counts) are 82% of the 4-thread run.)

Reading:

* **The 3-D cases scale best**: at equal pivot order 4.26-5.09x at 16 threads (1 -> 2 threads
  1.72-1.91x, efficiency 0.86-0.95); the own nested dissection orders in parallel (analysis 5.3-5.8x
  from 1 to 16 threads), the numeric factorizations run MUMPS' L0 subtrees in parallel and its large
  fronts on multithreaded BLAS (factorization 4.1-6.7x, certificate counts 4.4-5.4x). Against the
  fastest 1-thread configuration the default reaches 3.57-4.51x: it pays twice on ss-qa8fk-qa8fm
  (3.57x against 4.40x at equal order) - PORD orders this case best at 1 thread (5.51 s against the
  dissection's 6.25 s), and at 16 threads the default takes 1.54 s against the fixed dissection's
  1.42 s (syn-mem27-46 2.01 against 1.92 s, syn-near-singular-complex-1e5 2.22 against 1.92 s), the
  difference mostly (0.09 / 0.08 / 0.31 s) in the factorization column, where the adaptive
  ordering's AMD probe is counted (the open item "the AMD probe on 3-D problems" of
  docs/DESIGN.md's P9 row).
* **The solves stop at ~8 threads** (2.6-3.3x at 16; syn-mem27-46 0.227 -> 0.228 s, yee-pec-box-1e5
  0.635 -> 0.666 s from 8 to 16): a triangular solve streams the factor once per block of
  right-hand sides and is bound by the socket's memory bandwidth (P4 / docs/BLAS.md section 2).
* **2-D and low-fill cases** (AMD, serial ordering; little work per thread) reach 2.65-2.71x at 16
  threads (wg-square-1e5 8 -> 16 threads only 1.06x, cav2d-rect-pert-1e5 1.22x);
  syn-near-singular-complex-1e5's real embedding has a low-fill factor too (2.25x against its fastest
  1-thread run, AMD; 2.82x at equal order with the dissection).
* **n ~ 1e6:** 4 -> 16 threads 1.82x (factorization 1.89x, counts 1.99x, analysis 2.26x, solves
  1.20x) - the bandwidth limit of one socket that P9 factspeed measured for a single factorization
  (4 -> 16 threads 1.84x) and P9 mpi for MPI layouts on the same socket (results/p9_mpi.md §3.6).
* **The ordering at 1 thread** (open item, not the judged configuration): the adaptive ordering's
  model is fitted at 16 threads (4-thread data for its thread exponents only, docs/DESIGN.md
  §3.6-3b). At 1 thread it picks SCOTCH on syn-mem27-46 and ss-qa8fk-qa8fm, where METIS (1.21x /
  1.15x faster than the pick), PORD (1.09x / 1.27x) and the own dissection (1.13x / 1.12x) are all
  faster end to end; on the other four cases its pick is the fastest within 1%. Its decision log at
  1 thread (event `ordering`, build lane; the inputs are deterministic) prices syn-mem27-46's
  candidates at SCOTCH 7.11 s (analysed: 2.34 s), METIS 7.42, own dissection 7.78 - without the flow
  refinement there, its work ratio 4.17 falling under 4.5 at 1 thread -, PORD 8.11 (ss-qa8fk-qa8fm:
  4.23 / 4.92 / 5.18 / 5.31 s); at 16 threads it analyses the dissection (with the flow refinement,
  the configuration `nd` fixes: equal factor entries) and takes it. The model's SCOTCH ordering time
  is fitted at 16 threads, where SCOTCH itself runs multithreaded (§3.6-3b notes the model does not
  tell), so at 1 thread SCOTCH is priced low - consistent with the measured analyses (SCOTCH 2.34 s,
  METIS 1.06 s, PORD 0.51 s on syn-mem27-46), not isolated further. At 2-16 threads the other
  orderings were not timed here; the model's 16-thread replay over 38 data cases is 1.1% above the
  best ordering (results/p9_ordering.md §6). Pricing the orderings at low thread counts needs fit
  data at those counts (solver code; not part of this fix).

## 6. The default build (task 1)

* **What the build does.** `tools/dev/build.sh PRESET` (every preset; the CLI, the Python module, tests
  and benches all link MUMPS) first runs `tools/dev/mumps_cache.sh ensure` under the guard: MUMPS 5.8.2
  from the repository's `third_party/mumps/MUMPS_5.8.2.tar.gz` (the official tarball, sha256
  eb515aa6...4db7039 checked before every unpack, CeCILL-C notice and text alongside), patched with
  `patches/mumps-5.8.2-l0-blocking.patch` (patch(1), else `git apply`), built by
  `tools/dev/build_mumps.sh` with its portable defaults (`-O2 -march=nocona -mtune=haswell`:
  conda-forge's flags, the configuration P9 factspeed gated) against the environment's METIS, SCOTCH,
  OpenBLAS and gfortran, conda-forge's library names and sonames, no glibc symbol newer than 2.17.
  Installed as `$HS_MUMPS_CACHE/mumps-5.8.2-<key>` (default `build/deps/`; the cluster nodes share
  `$HOME/deps/mumps-cache`), the key the sha256 of the tarball's checksum, the patch's, the flags,
  the two build scripts, the C and Fortran compiler versions and the environment prefix
  (`CACHE-KEY.txt` in the install); any change builds a new install, repeated builds reuse it.
  Concurrent builds (work trees sharing the cache) build in their own temporary directory and rename
  the finished install into place, the first one wins (no lock on the shared NFS home). A failed
  build keeps its build directory and names its `build.log` (the whole make output; review fix - the
  temporary directory used to be removed with the log that `build_mumps.sh` had just pointed to,
  reproduced on gpu01 with a Fortran compiler that fails; `tests/py/test_mumps_cache.py`). Then
  `cmake --preset PRESET -DHSOLVE_MUMPS_ROOT=<install>`: its `lib/` goes first in the RPATH of
  executables, shared libraries and the Python module, and hsolve's default MUMPS block sizes become
  P9's (`mumps_blocking=p9`).
* **Fallbacks.** `HS_MUMPS=env tools/dev/build.sh` passes an empty `HSOLVE_MUMPS_ROOT`: the
  environment's conda-forge `mumps-seq`, MUMPS' block sizes (the pre-P9 build); `HS_MUMPS=/abs/prefix`
  links another `build_mumps.sh` install; at run time `HS_MUMPS_BLOCKING=mumps` restores MUMPS' block
  sizes in any build (section 2.2 measures exactly that). Switching reconfigures the same build
  directory: `cmake/FindMUMPS.cmake` searches again when `HSOLVE_MUMPS_ROOT` changes.
* **Build information.** `hsolve --A x --version` and `hsolve.build_info()` print
  `mumps=mumps-5.8.2-<key>` (or `mumps=env`) and `mumps_blocking=p9|mumps`.
* **Checks** (gpu01, build lane):
  * the first build of a fresh clone (`~/os2026-wt/p9int`, empty cache): MUMPS 83 s (4 jobs), then
    hsolve; 0 compiler warnings; ctest 1009/1009, pytest 376 passed (one new test failed on a test
    bug - `hsolve --version` needs `--A` to parse - fixed in ff50470);
  * the switch both ways in one build directory (`build/portable`: `HS_MUMPS=env`, then
    `HS_MUMPS=source`): CMake found the configured MUMPS each time, 0 warnings, `test_factor_mumps_build`
    and `test_factor_panel` (the counts with both block sizes against the analytic ones, the known-tree
    paths) and `tests/py/test_mumps_build.py` passed with `mumps=env` / `mumps_blocking=mumps` and with
    the source build / `p9`;
  * `tests/py/test_mumps_build.py` (new): the CLI resolves libdmumps / libsmumps / libmumps_common (and
    every other MUMPS library it loads) to the configured install (`LD_TRACE_LOADED_OBJECTS`: the
    loader's own resolution), the Python module has exactly those mapped after a solve in a fresh
    interpreter (`/proc/self/maps`), both print the configured MUMPS; the shipped tarball has the
    pinned checksum; the cache key changes with its inputs and `prefix` / `describe` never build;
  * the final state (f882138) from scratch in the work tree: no build directory and an empty MUMPS
    cache (`HS_MUMPS_CACHE=$HOME/p9int/fresh-mumps-cache`), `tools/dev/test.sh release`: MUMPS from the
    shipped tarball 84 s - the same key `df9061400e43252d` as the measured build (same inputs) -, hsolve
    211 s, 0 compiler warnings, ctest 1009/1009, pytest 380 passed; `hsolve --A x --version`:
    `mumps=mumps-5.8.2-df9061400e43252d mumps_blocking=p9`.
* **Other machines.** `env/activate.sh` used to default the guard's cores to the development
  server's NUMA node (`24-30,33-41`); on a machine without a host file (a user's, the judges') the
  guarded build and tests would have refused to start. The default is now the shell's own CPU list;
  the development server's list moved to `env/hosts/CISL-NF5468M6.sh`.

## 7. Conclusion

* The integrated build is the P9 deliverable: on the judging configuration (16 cores, 32 GiB) it
  solves the 1e6 3-D cases 1.28-1.34x faster than main a831d79 with the same exact certificate, is
  9% faster on the 1e5 data set without a regression, and beats tuned SLEPc on 5 of 8 head-to-head
  configurations (P5c: 4 of 6) - every run checked by tools/eval.
* The patched MUMPS as the default build is supported by the measurement the factspeed gate could not
  make: with the pivot order fixed by the deterministic dissection its share is 1.053-1.056x end to
  end on both 1e6 cases (4 runs each, disjoint ranges), and nothing at 1e5 (blk against new within
  +-4% on the small cases, in both directions).
* Remaining gaps against SLEPc are the exact certificate's two inertia-only factorizations
  (ss-qa8fk-qa8fm, cav2d-rect-pert-1e5, and yee-pec-box-1e6, where SLEPc's BLR factor makes its
  single-RHS solves cheap); at 1e6 the counts are half of the run. P8 measured every cheaper exact
  count negative; the certificate stays exact (user decision).
* Thread scaling on one socket is bounded by memory bandwidth (3-D 1e5 cases 4.26-5.09x at 16
  threads on one pivot order, 3.57-4.51x against the fastest 1-thread configuration; solves stop at
  ~8 threads, 1e6 4 -> 16 threads 1.82x); MPI MUMPS on the same socket is slower than threads (P9
  mpi), so more parallel speed needs more memory bandwidth (more sockets / nodes or a GPU), the P10
  work.

## 8. Commands

On gpu01. Builds and tests from the work tree (`HS_CLUSTER_NAME=p9int tools/dev/cluster_run.sh gpu01
-- ...`, `~/os2026-wt/p9int`); the measurements from a fixed copy of that tree, `$S/meas` (`S=$HOME/p9int`,
the files of b50a718 / d9f8b56 plus the two binaries in `$S/bin`: `hsolve_main` = main a831d79's
`build/release/hsolve`, `hsolve_int` = this branch's; `$S/meas/build/release/hsolve` -> `hsolve_int`
for compare.py), driven by `$S/campaign.sh`, `campaign2.sh`, `campaign3.sh` (consecutive slot calls,
each <= 2700 s; raw runs, evaluations and tables in `$S/{a1e6,b1e5,b1e5r,b1e5blk,e1e6,d1e5,d1e6,cmp}`).
`D6=$HOME/p7_p7-membudget/data`, `G="python3 tools/run/guard.py --exclusive --timeout 2700 --lock-wait
1800 --mem-gb 32 --"` (slot), `L="python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout
3600 --"` (lane), all through `env/run.sh`.

```bash
# build + whole suite (lane), and the fallback switch (build/portable, lane)
tools/dev/test.sh release
HS_MUMPS=env tools/dev/build.sh portable && ctest --test-dir build/portable -R 'test_factor_mumps_build|test_factor_panel' \
  && HSOLVE_BUILD_DIR=$PWD/build/portable python -m pytest tests/py/test_mumps_build.py      # then HS_MUMPS=source
# 2.1 1e6 ABBA, 8 repetitions (resumable; two slot calls)
HS_DATA_ROOT=$D6 $G python3 tools/bench/ab_cases.py run --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_int \
  --reps 0,1,2,3,4,5,6,7 --run-dir $S/a1e6 --timeout 1500 --deadline 2350 syn-mem27-100 yee-pec-box-1e6 wg-square-1e6
$L python3 -m tools.eval.sweep $S/a1e6 --root $D6 --out $S/a1e6/eval.md
python3 tools/bench/ab_cases.py table --base main --new new --json results/p9_measurements_1e6.json $S/a1e6
# 2.2 the block sizes' share: hsolve_int with HS_MUMPS_BLOCKING=mumps (wrapper $S/bin/hsolve_int_mumpsblk)
HS_DATA_ROOT=$D6 $G python3 tools/bench/ab_cases.py run --bin blk=$S/bin/hsolve_int_mumpsblk --bin new=$S/bin/hsolve_int \
  --reps 0,1,2,3 --run-dir $S/e1e6 --timeout 1500 --deadline 2350 syn-mem27-100 yee-pec-box-1e6
# 3 1e5 regression: all data cases but wg-square-1e6, 3 repetitions; re-check 10 repetitions (b1e5r: main
#   vs new, b1e5blk: blk vs new) of the cases not faster
$G python3 tools/bench/ab_cases.py run --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_int --reps 0,1,2 \
  --run-dir $S/b1e5 --timeout 1500 --deadline 2350 $(ls data/cases | grep -v '^wg-square-1e6$')
$G python3 tools/bench/ab_cases.py run --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_int --reps 0,...,9 \
  --run-dir $S/b1e5r --timeout 600 --deadline 2350 syn-lap2d-100 wg-halfslab-1e5 wg-square-1e3 wg-square-pert-1e4 \
  yee-pec-cube-16 cav2d-rect-pert-1e5 wg-square-1e5
python3 tools/bench/ab_cases.py table --base main --new new --json results/p9_measurements_1e5.json $S/b1e5
# 4 head to head (one slot call per 1e6 case, one for the five 1e5 nearest cases, one for the interval)
$G python3 tools/bench/compare.py syn-mem27-100 --tag cmp-p9int-1e6-syn-mem27-100 --cores 16-31 \
  --solvers hsolve,hsolve-auto,hsolve-nocert,slepc --slepc-profile best \
  --slepc-profile-file results/cmp-p9-1e6-profiles.json --reps 5 --root $D6 --mem-gb 32 --timeout 1500 --out-dir $S/cmp
$G python3 tools/bench/compare.py syn-mem27-46 yee-pec-box-1e5 wg-square-1e5 ss-qa8fk-qa8fm cav2d-rect-pert-1e5 \
  --tag cmp-p9int-best --cores 16-31 --solvers hsolve,hsolve-auto,hsolve-nocert,slepc --slepc-profile best --reps 5 \
  --mem-gb 32 --timeout 900 --out-dir $S/cmp
$G python3 tools/bench/compare.py yee-pec-box-1e5 --tag cmp-p9int-best-interval --cores 16-31 \
  --target '{"mode":"interval","lower":40,"upper":200}' --solvers hsolve,hsolve-auto,hsolve-nocert,slepc \
  --slepc-profile best --reps 5 --mem-gb 32 --timeout 900 --out-dir $S/cmp
python3 tools/bench/compare_merge.py --tag cmp-p9int-1e6 --out-dir results \
  $S/cmp/cmp-p9int-1e6-syn-mem27-100.json@gpu01 $S/cmp/cmp-p9int-1e6-yee-pec-box-1e6.json@gpu01
python3 tools/bench/compare_merge.py --tag cmp-p9int-best --out-dir results $S/cmp/cmp-p9int-best.json@gpu01   # and -interval
# 5 thread scaling
$G python3 tools/bench/thread_scaling.py run --bin $S/bin/hsolve_int --threads 1,2,4,8,16 --reps 0,1,2 \
  --run-dir $S/d1e5 --timeout 1500 --deadline 2350 syn-mem27-46 yee-pec-box-1e5 ss-qa8fk-qa8fm \
  syn-near-singular-complex-1e5 wg-square-1e5 cav2d-rect-pert-1e5
$G python3 tools/bench/thread_scaling.py run --bin $S/bin/hsolve_int --threads 4,8,16 --reps 0,1,2 \
  --run-dir $S/d1e6 --root $D6 --timeout 1500 --deadline 2350 yee-pec-box-1e6
$L python3 -m tools.eval.sweep $S/d1e5 --out $S/d1e5/eval.md            # d1e6: --root $D6
python3 tools/bench/thread_scaling.py table --order syn-mem27-46 yee-pec-box-1e5 ss-qa8fk-qa8fm \
  syn-near-singular-complex-1e5 wg-square-1e5 cav2d-rect-pert-1e5 --json d1e5.json $S/d1e5   # d1e6 likewise
# 5b review fix (F=$HOME/p9int-fix; tools from the fixed copy $F/meas = git archive of bfc1e95 (branch
#   p9int-fix), same binary; $F/campaign_fix.sh: two slot calls, resumable, evaluation in the lane)
SCREEN="--variant amd@1={\"ordering\":0} --variant scotch@1={\"ordering\":3} --variant metis@1={\"ordering\":5} \
  --variant pord@1={\"ordering\":4}"
$G python3 tools/bench/thread_scaling.py run --bin $S/bin/hsolve_int --threads 1,2,4,8,16 --reps 0,1,2,3,4 \
  --timeout 600 --deadline 2350 --run-dir $F/TA --variant default --variant 'nd={"ordering":10}' $SCREEN \
  syn-mem27-46 ss-qa8fk-qa8fm syn-near-singular-complex-1e5
$G python3 tools/bench/thread_scaling.py run --bin $S/bin/hsolve_int --threads 1,2,4,8,16 --reps 0,1,2,3,4 \
  --timeout 600 --deadline 2350 --run-dir $F/TB --variant default --variant 'nd@1={"ordering":10}' $SCREEN \
  yee-pec-box-1e5 wg-square-1e5 cav2d-rect-pert-1e5
$L python3 -m tools.eval.sweep $F/TA --out $F/TA/eval.md                 # and TB
python3 tools/bench/thread_scaling.py table --order syn-mem27-46 yee-pec-box-1e5 ss-qa8fk-qa8fm \
  syn-near-singular-complex-1e5 wg-square-1e5 cav2d-rect-pert-1e5 --md $F/table.md \
  --json results/p9_measurements_threads_orderings.json $F/TA $F/TB
# the adaptive ordering's decision at 1 and 16 threads (lane; event "ordering" of --log FILE.jsonl)
$L build/release/hsolve --A data/cases/syn-mem27-46/A.npz --config data/cases/syn-mem27-46/config.json \
  --threads 1 --out /tmp/o -q --log o.jsonl && grep '"ordering"' o.jsonl
```
