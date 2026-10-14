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
* **Multithreading** (1 -> 16 threads on one socket, 3 repetitions): 3-D n ~ 1e5 cases **4.2-5.4x**
  (efficiency 0.26-0.34; 1 -> 2 threads 1.7-2.1x), complex / 2-D cases 2.2-2.8x, yee-pec-box-1e6 4 ->
  16 threads 1.82x; ordering and analysis 5.9-8.3x, factorizations 4.0-5.8x, block solves 2.6-3.4x
  (memory bandwidth) (section 5).

## 1. Setup

* **Where.** gpu01 (2x Xeon Silver 4314, 2 x 16 cores, AVX-512, 251 GB, CentOS 7, glibc 2.17), the
  measurement slot: cores 16-31 = NUMA node 1, 16 threads, 32 GiB, one run at a time
  (`tools/run/guard.py --exclusive`, each slot call <= 2700 s, every solver run its own nested guard);
  builds, tests and evaluations in the build/test lane (cores 8-15). 2026-10-14 17:36-20:39, three
  chained scripts of consecutive slot calls (`$HOME/p9int/campaign.sh`, `campaign2.sh`,
  `campaign3.sh`), nothing else on the slot's socket.
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
* **Campaign** (slot calls in this order; 849 timed solver runs, every one valid):

  | part | slot time | content | runs |
  |---|---|---|---|
  | 2.1 | 17:36-18:25 | 1e6, main vs new, 8 reps ABBA | 48 |
  | 3 | 18:25-19:03 | 41 data cases, main vs new, 3 reps ABBA | 246 |
  | 4 | 19:03-19:54 | compare.py: syn-mem27-100, yee-pec-box-1e6, five 1e5 nearest cases, the interval, 4 solvers x 5 reps | 160 |
  | 5 | 19:54-20:14 | thread scaling: six 1e5 cases x 1/2/4/8/16 threads, yee-pec-box-1e6 x 4/8/16, 3 reps | 99 |
  | 2.2 | 20:14-20:37 | 1e6, blk vs new, 4 reps ABBA | 16 |
  | 3 (re-check) | 20:37-20:39 | seven 1e5 cases, main vs new and blk vs new, 10 reps ABBA | 280 |

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

`tools/bench/thread_scaling.py run --threads 1,2,4,8,16 --reps 0,1,2` inside one slot guard: each run
pinned to the first T cores of the held slot (16..15+T; the rest of the socket idle, so a T-thread run
has the socket's whole memory bandwidth), the case's default configuration with `--threads T`
(certificate exact), thread counts up for even and down for odd repetitions, medians of 3; all 99
runs valid (`results/p9_measurements_threads.json`). Six n ~ 1e5 cases cover HEP / GHEP / GHIEP,
real and complex, 3-D and 2-D; one 1e6 3-D case at 4 / 8 / 16 threads. Speedup = t(1) / t(T)
(1e6: t(4) / t(T)), efficiency = speedup x T_base / T. The phase columns split the report's five
exclusive phases (they add up to the total): analysis (ordering + symbolic analysis of the main
object), factorization (the rest of the factor phase: numeric factorization at sigma, other
candidate analyses), solves (the Krylov iteration's block solves), iteration (orthogonalization,
restarts), certificate (inertia counts), rest (setup + verify); "ordering" is the adaptive ordering's
choice, which depends on the thread count (the cost model prices the parallel dissection by threads).

Summary (solver time in s; speedup at 16 threads, efficiency):

| case | 1 | 2 | 4 | 8 | 16 | speedup 16 | efficiency 16 |
|---|---|---|---|---|---|---|---|
| syn-mem27-46 (3-D HEP) | 11.01 | 5.58 | 3.36 | 2.35 | 2.04 | **5.39x** | 0.34 |
| yee-pec-box-1e5 (3-D GHEP, Yee) | 10.59 | 6.13 | 3.75 | 2.74 | 2.51 | **4.21x** | 0.26 |
| ss-qa8fk-qa8fm (3-D GHEP, acoustic) | 6.97 | 3.35 | 2.14 | 1.61 | 1.53 | **4.55x** | 0.28 |
| syn-near-singular-complex-1e5 (complex HEP) | 5.03 | 4.20 | 3.44 | 2.72 | 2.25 | 2.24x | 0.14 |
| wg-square-1e5 (2-D GHIEP) | 1.44 | 1.02 | 0.69 | 0.53 | 0.53 | 2.69x | 0.17 |
| cav2d-rect-pert-1e5 (2-D GHEP) | 0.91 | 0.72 | 0.53 | 0.39 | 0.33 | 2.77x | 0.17 |
| yee-pec-box-1e6 (3-D GHEP, n ~ 1e6) | - | - | 125.65 | 85.36 | 69.16 | 1.82x vs 4 | 0.45 vs 4 |

Per case:

**syn-mem27-46** (n = 97336, HEP, certificate exact, block p 8)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.008 [10.997-11.028] | 1.00x | 1.00 | 2.345 | 2.662 | 0.763 | 0.111 | 4.978 | 0.150 | SCOTCH | 3/3 | 0.68 |
| 2 | 5.584 [5.581-5.600] | 1.97x | 0.99 | 0.901 | 1.433 | 0.432 | 0.067 | 2.634 | 0.119 | ND | 3/3 | 0.69 |
| 4 | 3.364 [3.344-3.383] | 3.27x | 0.82 | 0.527 | 0.869 | 0.276 | 0.045 | 1.548 | 0.099 | ND | 3/3 | 0.69 |
| 8 | 2.346 [2.316-2.351] | 4.69x | 0.59 | 0.354 | 0.630 | 0.227 | 0.034 | 1.026 | 0.077 | ND | 3/3 | 0.79 |
| 16 | 2.041 [2.037-2.069] | 5.39x | 0.34 | 0.297 | 0.565 | 0.228 | 0.029 | 0.865 | 0.067 | ND | 3/3 | 0.86 |

Phase speedups 1 -> 16 threads: analysis 7.90x, factorization 4.71x, solves 3.35x, iteration 3.79x, certificate 5.75x, rest 2.25x

**yee-pec-box-1e5** (n = 109744, GHEP, certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10.590 [10.581-10.598] | 1.00x | 1.00 | 1.602 | 2.330 | 1.726 | 0.463 | 4.363 | 0.107 | ND | 3/3 | 0.58 |
| 2 | 6.131 [6.089-6.142] | 1.73x | 0.86 | 0.865 | 1.377 | 1.035 | 0.259 | 2.488 | 0.101 | ND | 3/3 | 0.58 |
| 4 | 3.751 [3.709-3.762] | 2.82x | 0.71 | 0.505 | 0.836 | 0.713 | 0.163 | 1.441 | 0.087 | ND | 3/3 | 0.61 |
| 8 | 2.739 [2.695-2.741] | 3.87x | 0.48 | 0.334 | 0.628 | 0.633 | 0.110 | 0.967 | 0.062 | ND | 3/3 | 0.65 |
| 16 | 2.514 [2.497-2.532] | 4.21x | 0.26 | 0.272 | 0.585 | 0.664 | 0.090 | 0.836 | 0.057 | ND | 3/3 | 0.78 |

Phase speedups 1 -> 16 threads: analysis 5.90x, factorization 3.99x, solves 2.60x, iteration 5.16x, certificate 5.22x, rest 1.87x

**ss-qa8fk-qa8fm** (n = 66127, GHEP, certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.971 [6.955-6.979] | 1.00x | 1.00 | 1.692 | 1.515 | 1.146 | 0.392 | 2.061 | 0.162 | SCOTCH | 3/3 | 0.38 |
| 2 | 3.351 [3.338-3.355] | 2.08x | 1.04 | 0.585 | 0.801 | 0.657 | 0.219 | 0.968 | 0.122 | ND | 3/3 | 0.38 |
| 4 | 2.136 [2.096-2.137] | 3.26x | 0.82 | 0.348 | 0.512 | 0.461 | 0.154 | 0.564 | 0.094 | ND | 3/3 | 0.40 |
| 8 | 1.607 [1.581-1.616] | 4.34x | 0.54 | 0.242 | 0.389 | 0.394 | 0.119 | 0.392 | 0.073 | ND | 3/3 | 0.45 |
| 16 | 1.531 [1.517-1.534] | 4.55x | 0.28 | 0.204 | 0.357 | 0.418 | 0.097 | 0.378 | 0.069 | ND | 3/3 | 0.75 |

Phase speedups 1 -> 16 threads: analysis 8.31x, factorization 4.24x, solves 2.74x, iteration 4.02x, certificate 5.46x, rest 2.34x

**syn-near-singular-complex-1e5** (n = 100000, HEP, certificate exact, block p 4)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.031 [5.019-5.091] | 1.00x | 1.00 | 0.307 | 0.992 | 1.429 | 0.846 | 0.967 | 0.487 | AMD | 3/3 | 0.40 |
| 2 | 4.202 [4.144-4.217] | 1.20x | 0.60 | 0.298 | 0.891 | 1.363 | 0.501 | 0.855 | 0.288 | AMD | 3/3 | 0.39 |
| 4 | 3.440 [3.402-3.453] | 1.46x | 0.37 | 0.407 | 0.918 | 0.991 | 0.335 | 0.579 | 0.209 | ND | 3/3 | 0.52 |
| 8 | 2.723 [2.710-2.730] | 1.85x | 0.23 | 0.394 | 0.739 | 0.748 | 0.279 | 0.409 | 0.161 | ND | 3/3 | 0.52 |
| 16 | 2.247 [2.242-2.253] | 2.24x | 0.14 | 0.394 | 0.627 | 0.537 | 0.258 | 0.294 | 0.137 | ND | 3/3 | 0.67 |

Phase speedups 1 -> 16 threads: analysis 0.78x, factorization 1.58x, solves 2.66x, iteration 3.28x, certificate 3.29x, rest 3.55x

**wg-square-1e5** (n = 99225, GHIEP, certificate none, block p 2)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.435 [1.419-1.435] | 1.00x | 1.00 | 0.052 | 0.205 | 0.690 | 0.290 | 0.000 | 0.196 | AMD | 3/3 | 0.16 |
| 2 | 1.020 [1.009-1.043] | 1.41x | 0.70 | 0.053 | 0.159 | 0.480 | 0.178 | 0.000 | 0.150 | AMD | 3/3 | 0.17 |
| 4 | 0.690 [0.689-0.731] | 2.08x | 0.52 | 0.050 | 0.105 | 0.309 | 0.131 | 0.000 | 0.097 | AMD | 3/3 | 0.18 |
| 8 | 0.525 [0.523-0.566] | 2.73x | 0.34 | 0.049 | 0.074 | 0.218 | 0.107 | 0.000 | 0.077 | AMD | 3/3 | 0.20 |
| 16 | 0.533 [0.496-0.546] | 2.69x | 0.17 | 0.072 | 0.078 | 0.204 | 0.097 | 0.000 | 0.080 | AMD | 3/3 | 0.30 |

Phase speedups 1 -> 16 threads: analysis 0.73x, factorization 2.62x, solves 3.39x, iteration 2.99x, rest 2.45x

**cav2d-rect-pert-1e5** (n = 99459, GHEP, certificate exact, block p 2)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.912 [0.911-0.914] | 1.00x | 1.00 | 0.031 | 0.153 | 0.290 | 0.166 | 0.196 | 0.077 | AMD | 3/3 | 0.10 |
| 2 | 0.718 [0.715-0.725] | 1.27x | 0.64 | 0.036 | 0.123 | 0.252 | 0.094 | 0.163 | 0.051 | AMD | 3/3 | 0.12 |
| 4 | 0.532 [0.531-0.579] | 1.71x | 0.43 | 0.033 | 0.096 | 0.184 | 0.060 | 0.120 | 0.041 | AMD | 3/3 | 0.13 |
| 8 | 0.385 [0.382-0.435] | 2.37x | 0.30 | 0.029 | 0.078 | 0.117 | 0.046 | 0.084 | 0.030 | AMD | 3/3 | 0.17 |
| 16 | 0.330 [0.305-0.358] | 2.77x | 0.17 | 0.042 | 0.076 | 0.075 | 0.040 | 0.061 | 0.039 | AMD | 3/3 | 0.24 |

Phase speedups 1 -> 16 threads: analysis 0.74x, factorization 2.03x, solves 3.88x, iteration 4.12x, certificate 3.22x, rest 1.98x

**yee-pec-box-1e6** (n = 1027733, GHEP, certificate exact, block p 8)

| threads | total s [min-max] | speedup | efficiency | analysis | factorization | solves | iteration | certificate | rest | ordering | valid | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 125.650 [125.644-125.725] | 1.00x | 1.00 | 6.175 | 35.606 | 13.059 | 2.603 | 67.591 | 0.625 | ND | 3/3 | 11.43 |
| 8 | 85.363 [85.327-85.597] | 1.47x | 0.74 | 3.794 | 23.843 | 10.747 | 2.287 | 44.107 | 0.571 | ND | 3/3 | 11.86 |
| 16 | 69.160 [69.118-69.180] | 1.82x | 0.45 | 2.734 | 18.793 | 10.911 | 2.273 | 33.908 | 0.525 | ND | 3/3 | 11.32 |

Phase speedups 4 -> 16 threads: analysis 2.26x, factorization 1.89x, solves 1.20x, iteration 1.14x, certificate 1.99x, rest 1.19x

Reading:

* **The 3-D cases scale best** (4.2-5.4x at 16 threads on n ~ 1e5, 1 -> 2 threads 1.7-2.1x): the own
  nested dissection orders in parallel (analysis 5.9-8.3x from 1 to 16 threads), the numeric
  factorizations - main and certificate counts - run MUMPS' L0 subtrees in parallel and its large
  fronts on multithreaded BLAS (4.0-4.7x and 5.2-5.8x).
* **The solves stop at ~8 threads** (2.6-3.4x at 16; syn-mem27-46 0.227 -> 0.228 s, yee-pec-box-1e5
  0.633 -> 0.664 s from 8 to 16): a triangular solve streams the factor once per block of
  right-hand sides and is bound by the socket's memory bandwidth (P4 / docs/BLAS.md section 2).
* **2-D and low-fill cases** (AMD, serial ordering; little work per thread) saturate at 8 threads
  (2.2-2.8x); syn-near-singular-complex-1e5's real embedding has a low-fill factor too.
* **n ~ 1e6:** 4 -> 16 threads 1.82x (factorization 1.89x, counts 1.99x, analysis 2.26x, solves
  1.20x) - the bandwidth limit of one socket that P9 factspeed measured for a single factorization
  (4 -> 16 threads 1.84x) and P9 mpi for MPI layouts on the same socket (results/p9_mpi.md §3.6).
* **The ordering follows the thread count**: at 1 thread syn-mem27-46 and ss-qa8fk-qa8fm take SCOTCH
  and syn-near-singular-complex-1e5 AMD (1-2 threads), with more threads the own dissection - the
  cost model's thread exponents (docs/DESIGN.md §3.6-3b) price its parallelism. So the 1-thread
  baseline is hsolve's own best choice at 1 thread, not the 16-thread configuration run serially.

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
  the finished install into place, the first one wins (no lock on the shared NFS home). Then
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
* Thread scaling on one socket is bounded by memory bandwidth (solves stop at ~8 threads, 1e6
  4 -> 16 threads 1.82x); MPI MUMPS on the same socket is slower than threads (P9 mpi), so more
  parallel speed needs more memory bandwidth (more sockets / nodes or a GPU), the P10 work.

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
```
