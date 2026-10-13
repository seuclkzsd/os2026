# P8 "certcost": robust and cheaper certificate counts

P8 asked for (1) certificate counts whose failure never ends a solve, (2) cheaper EXACT counts
(concurrent edge counts, count settings, overlap), (3) a proven kappa for a general SPD B (optional
after the user's decision that the default stays `certificate = exact`), and (4) the measurements
below. Branch `p8-certcost` from main 34cbfd4. Every timing is from a 16-core measurement slot
(`guard.py --exclusive`, cores 16-31, 32 GiB), 16 threads, 2026-10-14.

## 1. Setup

**Code.** Branch p8-certcost. Binaries (each `tools/dev/build.sh release`: gcc 13.4, `-O3 -march=native`,
BLAS hook linked, route all; build lane, cores 8-15, `HS_JOBS=4`; copies in `~/p8_p8-certcost/bin/`):

| label | commit | md5 | used for |
|---|---|---|---|
| main | main 34cbfd4 (own clone on c005) | 8efe5c5a18b8a7f7b721fbbbe2aa1f2a | regression baseline (§5) |
| P8 (p8b) | d0035c5 | 0414844016fbd4df384675349b59f7a5 | counter A/B, regression, head-to-head nearest, first ss-3dspectralwave2 campaign |
| P8 (92d4962) | 92d4962 | - (work tree build) | head-to-head interval |
| P8 (p8e) | be2efde | 707cdb2efeebae52ffd15fb8a52fdb12 | ss-3dspectralwave2 campaigns 2 and 3 |

The commits after d0035c5 change only (a) the re-analysis of a SCOTCH analysis that does not fit the
budget (92d4962, be2efde: inert unless an estimate exceeds the cap, which happens in none of the
regression / head-to-head runs - every one of them fits) and (b) a fix in the auto / randomized modes
on general B (d645b5e), plus docs and results; the measurements of the default configuration therefore
describe the final code. compare.py ran the work tree's `build/release/hsolve` (same code).

**Nodes.** gpu01 (all slot measurements except one) and c005 (ss-3dspectralwave2 robustness runs),
both 2x Xeon Silver 4314, checked idle with `tools/dev/cluster_nodes.sh`; gpu02 was not used (P7).
Evaluations (tools/eval) ran in the build lane.

| slot campaign | node | time (2026-10-14) | content |
|---|---|---|---|
| concurrency | gpu01 | 01:48-01:51 | `tools/bench/certcost_concurrent.sh`, 8 cases x 5 reps (§3.1) |
| count settings | gpu01 | 02:03-02:05 | §3.3 |
| counter A/B | gpu01 | 02:09-02:13 | §3.4 (31 cases x 2 x 3) |
| block pair | gpu01 | 02:13-02:14 | §3.2 |
| head-to-head | gpu01 | 02:16-02:20, 02:35-02:37 | §4 (nearest; interval) |
| regression | gpu01 | 02:20-02:35 | §5 (34 cases x 2 binaries x 3) |
| ss-3dspectralwave2 | c005 | 02:03-02:36, 02:38-03:19 | §2.3 (binaries d0035c5, be2efde) |
| ss-3dspectralwave2 | gpu01 | 02:39-03:18 | §2.3 (be2efde) |

## 2. Counter robustness (task 1)

### 2.1 The failure and its analogue

P5c (results/p5c_measurements.md §2.2): ss-3dspectralwave2 at the CLI default budget (14 GiB, cap
13.49 GB) runs with out-of-core main factors; the nearest certificate then builds an InertiaCounter
whose MUMPS in-core estimate scattered 10.6-14.5 GB from run to run. At 14.5 GB its count failed
inside MUMPS (INFO(1) = -19 under ICNTL(23)), the exception ended the solve (exit 2) and the ten
converged pairs were lost; at 13.6 GB it ran with a peak RSS of 15.1 GiB, above the budget.

Small analogue (build lane, 8 threads; not timed): yee-pec-cube-32 with METIS (`ordering = 5`) at
`--memory-budget-gb 0.394`: main factors out of core (in-core estimate 636 MB, cap 287 MB), counter
estimate 300 MB. main / P5c code: exit 2, "MUMPS factorization ran out of memory: INFO(1)=-19,
INFO(2)=464894" after the Krylov run, no result. P8: the counter is not tried (300 MB above the
156 MB the budget leaves for a count; event `count_source`, warning), the counts run on the main
object (out of core), 12/12 pairs, certificate exact passed.

### 2.2 What changed (DESIGN §3.6-8 "计数失败不结束求解", FORMATS §3)

- `CountSource::count`: a count that fails (OutOfMemoryBudget - MUMPS -13/-19 -, FactorizationFailed -
  any other MUMPS error -, `std::bad_alloc`) abandons that source for the solve and retries on the other
  one (counter <-> main object); a counter whose in-core estimate exceeds `count_allowance_mb` =
  min(0.95 (memory_budget - RSS), cap) - after releasing resident main factors - is not tried.
- The certifier's own counter (single counts, their double check, `exact_counts`) checks the same
  allowance (`InertiaCounter::estimate(precision)`, new) and leaves the verdict to the exact counts
  when it fails; after a memory failure CountSource does not build a second counter on the same order.
- A randomized check whose refactorization or solves fail leaves the verdict to the counts.
- No count possible: the nearest target returns its converged pairs; the certificate is performed but
  not completed (passed = false, rigorous = false, failure_probability = 1, count_inertia = -1,
  note "not completed: ...", statement with the reason), warning, success = false (CLI exit 1, result
  written, summary line "NOT COMPLETED"). A deflated rerun that cannot refactorize at sigma returns the
  pairs found. Window targets (built on counts) still throw when no source can count.
- Fault injection `certificate_fault_count` (bits: 1 counter cap 1 MB -> genuine INFO -19, 2 main-object
  counts fail, 4 allowance 0, 8 rerun refactorization fails); tests `tests/cpp/test_certcost_faults.cpp`
  (20 tests, real and complex: each path, both sources failing, window targets, the certifier's counter,
  the genuine MUMPS -19 of a counter above its cap, `estimate(Single)`).

### 2.3 ss-3dspectralwave2 at the default budget

Repeated default runs (case config.json, `--threads 16`, CLI default budget 14 GiB = cap 13.49 GB, the
event log; `spw_runs.sh`: one slot guard call per campaign, each run under its own nested guard),
three campaigns: d0035c5 on c005 (8 runs), be2efde on c005 (8) and on gpu01 (7):

```bash
S=$HOME/p8_p8-certcost; D=$HOME/os2026/data/cases/ss-3dspectralwave2   # one command per run:
python3 tools/run/guard.py --skip-busy-check --timeout 1500 --summary $o/guard.json -- \
  $S/bin/hsolve_p8e --A $D/A.npz --config $D/config.json --threads 16 --out $o --log $o/log.jsonl -q
```

| run | exit | time (s) | peak GiB (guard) | main analysis in-core / OOC (MB) | analyses | counter estimate (MB) | counts on | certificate | error |
|---|---|---|---|---|---|---|---|---|---|
| p8b-c005-a/run1 | 0 | 328.4 | 11.27 | 28451 / 11663 | 1 | 11624 | main (counter over allowance) | exact passed (10/10) |  |
| p8b-c005-a/run2 | 0 | 334.6 | 12.791 | 28841 / 11893 | 1 | 11628 | main (counter over allowance) | exact passed (10/10) |  |
| p8b-c005-a/run3 | 0 | 306.3 | 12.614 | 27824 / 11147 | 1 | 11114 | counter | exact passed (10/10) |  |
| p8b-c005-a/run4 | 2 | - | 4.204 | 27555 / 13828 | 1 | - | - | - | OutOfMemoryBudget: main analysis OOC estimate 13828 MB > cap 13491 MB|
| p8b-c005-a/run5 | 0 | 294.0 | 12.369 | 26897 / 10912 | 1 | 10840 | counter | exact passed (10/10) |  |
| p8b-c005-a/run6 | 0 | 303.4 | 12.802 | 27560 / 10877 | 1 | 10996 | counter | exact passed (10/10) |  |
| p8b-c005-a/run7 | 0 | 316.8 | 12.245 | 27523 / 11068 | 1 | 12514 | main (counter over allowance) | exact passed (10/10) |  |
| p8b-c005-a/run8 | 2 | - | 4.176 | 27191 / 14142 | 1 | - | - | - | OutOfMemoryBudget: main analysis OOC estimate 14142 MB > cap 13491 MB|
| p8e-c005-b/run1 | 0 | 316.5 | 12.692 | 28003 / 11128 | 1 | 11093 | counter | exact passed (10/10) |  |
| p8e-c005-b/run2 | 0 | 299.3 | 12.63 | 27344 / 11076 | 1 | 10971 | counter | exact passed (10/10) |  |
| p8e-c005-b/run3 | 0 | 295.2 | 11.562 | 26698 / 10556 | 1 | 10510 | counter | exact passed (10/10) |  |
| p8e-c005-b/run4 | 0 | 303.2 | 12.291 | 27352 / 11110 | 1 | 11068 | counter | exact passed (10/10) |  |
| p8e-c005-b/run5 | 0 | 297.2 | 12.52 | 27133 / 11005 | 1 | 10972 | counter | exact passed (10/10) |  |
| p8e-c005-b/run6 | 0 | 301.9 | 12.689 | 27307 / 10870 | 1 | 10814 | counter | exact passed (10/10) |  |
| p8e-c005-b/run7 | 0 | 309.4 | 11.788 | 27233 / 11003 | 1 | 11346 | main (counter over allowance) | exact passed (10/10) |  |
| p8e-c005-b/run8 | 0 | 333.2 | 12.446 | 28683 / 11792 | 1 | 15054 | main (counter over allowance) | exact passed (10/10) |  |
| p8e-gpu01-c/run1 | 0 | 325.5 | 12.61 | 27334 / 14063; 27578 / 10994 | 2 | 10954 | counter | exact passed (10/10) |  |
| p8e-gpu01-c/run2 | 0 | 334.0 | 11.291 | 28738 / 11925 | 1 | 11868 | main (counter over allowance) | exact passed (10/10) |  |
| p8e-gpu01-c/run3 | 0 | 334.7 | 11.078 | 28373 / 11504 | 1 | 11530 | main (counter over allowance) | exact passed (10/10) |  |
| p8e-gpu01-c/run4 | 0 | 297.7 | 12.704 | 26984 / 11080 | 1 | 10942 | counter | exact passed (10/10) |  |
| p8e-gpu01-c/run5 | 0 | 343.6 | 12.933 | 28620 / 11572 | 1 | 11521 | main (counter over allowance) | exact passed (10/10) |  |
| p8e-gpu01-c/run6 | 0 | 314.0 | 11.87 | 27843 / 14352; 26955 / 10685 | 2 | 10620 | counter | exact passed (10/10) |  |
| p8e-gpu01-c/run7 | 0 | 341.1 | 12.016 | 28503 / 11504 | 1 | 11444 | main (counter over allowance) | exact passed (10/10) |  |

- **Every run that got past the main analysis returned its ten pairs certified** (21/21: success,
  every eta <= 7.8e-12, exact certificate 10 = 10; the 22 eigenvalue sets of these and the earlier run
  agree to 1.4e-19 ||A||_2 - there is no stored reference for this case). Peak RSS (guard, whole
  process tree) 11.1-12.9 GiB, below the 14 GiB budget in every run (P5c: up to 15.1 GiB).
- The counter's estimate scattered 10.5-15.1 GB (it follows SCOTCH's main order); in 9 of 21 runs it
  exceeded the count allowance (0.95 x what the budget leaves, ~11.2-11.8 GB) and the counts ran on
  the out-of-core main object: those runs took 309-344 s against 294-326 s with the counter. With the
  P5c code the 15.1 GB estimate would have failed with INFO -19 and lost the pairs.
- The main analysis' out-of-core estimate exceeded the cap in 4 of 25 analyses (13.8, 14.1, 14.1,
  14.4 GB; P5c: 2 of 17). d0035c5 ended there (runs 4 and 8 of the first campaign, exit 2 after 20 s,
  before any factorization - P5c's failure mode 2, not a count); be2efde re-analysed (gpu01 runs 1 and 6:
  14.1 -> 11.0 GB and 14.4 -> 10.7 GB, +16 s) and both runs certified. 15/15 runs of the final code
  returned certified pairs.

### 2.4 Can MUMPS' estimates be made reproducible cheaply?

The scatter comes from SCOTCH's ordering inside MUMPS' analysis, which runs multithreaded with the
analysis' OpenMP threads (ICNTL(16)). Probe (`hsolve.solve` through the Python module with
`ordering = 3`, k = 1; `SCOTCH_randomSeed(1)` + `SCOTCH_randomReset()` called through ctypes on the
libscotch MUMPS loaded; build lane):

| threads | random state | factor entries (1e6) of syn-lap3d-aniso-1e5, successive analyses | across processes |
|---|---|---|---|
| 8 | as is | 34.68, 34.92, 33.46 | differ (34.0, 34.4) |
| 8 | reset before each | 34.37, 34.63, 34.09 | - |
| 8 | seed 1 + reset | 33.91, 33.14, 34.29 | differ (34.19, 33.09) |
| 2 | seed 1 + reset | 35.96, 34.02, 34.71 | - |
| 1 | as is | 30.99, 31.51, 30.31 | first analysis reproduced (30.99) |
| 1 | seed 1 + reset | 30.99, 30.99, 30.99 | reproduced (30.99) |

With one thread the ordering is reproducible (and has less fill: 433 vs ~500 MB in-core), but the
one-thread analysis costs 2.6-3.7x: syn-lap3d-aniso-1e5 1.42 vs 0.45-0.54 s, yee-pec-box-1e5 2.27 vs
0.61 s, ss-2cubes_sphere 2.62 vs 0.74-0.80 s, ss-qa8fk-qa8fm 1.69 vs 0.46-0.53 s (8 threads, build
lane). So no cheap seed fix exists; the driver now tolerates the scatter (§2.2) instead. A one-thread
SCOTCH analysis handed on as PERM_IN to the multithreaded main analysis would make the estimates
reproducible at the price above (open issue for the factor owner).

## 3. Cheaper exact counts (task 2): measured negative

The certificate's two exact counts are inertia-only factorizations on the main pivot order. All
results below: gpu01 slot, 16 threads, medians of 5 repetitions, counts identical in every variant.

### 3.1 Concurrent edge counts (8 + 8 threads)

Feasibility: in one process impossible - sequential MUMPS 5.8.2 keeps process-global Fortran module
state (MUMPS_LOAD_INIT, communication buffers), two instances factorizing concurrently crash
(factor.hpp, P3), and hsolve serializes every MUMPS call with one lock. The upper bound of any
concurrent scheme is two processes, each pinned to 8 of the slot's cores (`two8`); `fork8` tries it
in-process by fork() (the child counts nu(b) on a fresh pthread):

| case | count 16 thr (s) | seq pair 16 thr (s) | count 8 thr (s) | one 8-thr count alone (s) | two 8-thr counts at once, slower (s) | seq pair / concurrent bound | counts agree | fork pair (s) | fork children |
|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 0.0261 | 0.0524 | 0.0374 | 0.0379 | 0.0414 | 1.26 | True | 0.0823 | -11 -11 -11 -11 -11  |
| cav2d-square-1e4 | 0.0050 | 0.0099 | 0.0054 | 0.0059 | 0.0052 | 1.90 | True | 0.0733 | ok ok ok ok ok  |
| ss-2cubes_sphere | 0.3624 | 0.7212 | 0.4211 | 0.4226 | 0.6454 | 1.12 | True | 0.6644 | -11 -11 -11 -11 -11  |
| ss-qa8fk-qa8fm | 0.1629 | 0.3253 | 0.1780 | 0.1808 | 0.2528 | 1.29 | True | 0.3425 | -11 -11 -11 -11 -11  |
| syn-lap2d-100 | 0.0053 | 0.0107 | 0.0053 | 0.0050 | 0.0053 | 2.03 | True | 0.0555 | ok ok ok ok ok  |
| syn-mem125-46 | 2.1369 | 4.2667 | 2.5841 | 2.5803 | 4.3276 | 0.99 | True | 3.7552 | -11 -11 -11 -11 -11  |
| syn-mem27-46 | 0.4127 | 0.8236 | 0.4951 | 0.4962 | 0.7759 | 1.06 | True | 0.7717 | -11 -11 -11 -11 -11  |
| yee-pec-box-1e5 | 0.4862 | 0.9690 | 0.5691 | 0.5736 | 0.8830 | 1.10 | True | 0.8499 | -11 -11 -11 -11 -11  |

- The bound is 0.99-1.29x for the 1e5 cases (the counts are memory-bandwidth bound: one 8-thread
  count alone takes 0.42-2.58 s, two at once 0.63-4.33 s each); ~2x only on the 1e4 cases, where a
  pair takes 5-10 ms. Realised fully it would save 0.07 s of ss-qa8fk-qa8fm's 1.67 s (4%) and 0.011 s
  of cav2d-rect-pert-1e5's 0.30 s (3.6%).
- fork() after OpenMP use: the child died with SIGSEGV in MUMPS on 6 of the 8 cases (every
  repetition); where it lived (the 1e4 cases) fork + wait cost 50-70 ms against a 5-10 ms pair. A
  second, independent MUMPS copy in one process (dlmopen namespace, glibc 2.17, duplicated libgomp /
  OpenBLAS / libgfortran) was not tried: high risk, gain bounded by the table.

Commands: `env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 -- bash
tools/bench/certcost_concurrent.sh $S/conc1 5 $S/cases_conc.txt` (the cases file: case, window ends
of the P8 reports at 16 threads, ordering 10 / 0 for the 2-D cases).

### 3.2 Both ends in one factorization: diag(K(b), -K(a))

MUMPS' own L0-OpenMP then factorizes the two independent trees concurrently, in one (thread-safe)
call (`bench_certpar_inertia --exp blockpair --threads 16 --reps 5`, METIS / AMD orders):

| case | seq pair (s) | block pair (s) | seq / block | block analysis (s) | block fill (s) | counts agree | INFOG(21) seq / block (MB) |
|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 0.0537 | 0.0491 | 1.09 | 0.043 | 0.0013 | True | 23 / 44 |
| cav2d-square-1e4 | 0.0095 | 0.0067 | 1.43 | 0.004 | 0.0001 | True | 3 / 5 |
| ss-2cubes_sphere | 0.7606 | 0.7562 | 1.01 | 0.123 | 0.0025 | True | 244 / 407 |
| ss-qa8fk-qa8fm | 0.3531 | 0.3320 | 1.06 | 0.072 | 0.0035 | True | 122 / 265 |
| syn-lap2d-100 | 0.0108 | 0.0074 | 1.45 | 0.004 | 0.0001 | True | 3 / 6 |
| syn-mem125-46 | 4.2327 | 4.3963 | 0.96 | 0.282 | 0.0217 | True | 988 / 2195 |
| syn-mem27-46 | 0.8203 | 0.8318 | 0.99 | 0.097 | 0.0042 | True | 244 / 619 |
| yee-pec-box-1e5 | 0.8971 | 0.8858 | 1.01 | 0.118 | 0.0027 | True | 251 / 445 |

Same time (0.96-1.09x) on the 1e5 cases plus its own analysis and 1.8-2.5x the working memory.

### 3.3 Count settings

`bench_certcost_concurrent --exp seq --threads 16 --scaling 7|0 --detection 1|0` (count nu(a)):

| case | count default (s) | repeat (s) | no scaling (s) | no detection (s) | nu equal |
|---|---|---|---|---|---|
| ss-qa8fk-qa8fm | 0.1683 | 0.1625 | 0.1606 | 0.1589 | True |
| cav2d-rect-pert-1e5 | 0.0259 | 0.0264 | 0.0247 | 0.0255 | True |
| syn-mem27-46 | 0.4122 | 0.4121 | 0.4054 | 0.4043 | True |
| yee-pec-box-1e5 | 0.4824 | 0.4838 | 0.4794 | 0.4834 | True |
| ss-2cubes_sphere | 0.3610 | 0.3631 | 0.3582 | 0.3571 | True |
| syn-mem125-46 | 2.1456 | 2.1404 | 2.1168 | 2.1162 | True |
| cav2d-square-1e4 | 0.0045 | 0.0047 | 0.0044 | 0.0047 | True |
| syn-lap2d-100 | 0.0054 | 0.0053 | 0.0048 | 0.0052 | True |

Without scaling or without null-pivot detection a count is 0-6% cheaper on the 1e5 cases
(ss-qa8fk-qa8fm 0.159-0.161 s against 0.163-0.168 s in two default runs: 0.005 s per count), and
under 1 ms on the 1e4 cases - near the run-to-run spread, and neither is sound to drop: detection is
part of the certificate's validity test, and the scaling is what the null-pivot threshold and the
count's backward stability were calibrated with. 16 threads beat 8 on every case (§3.1: count 16 thr
vs 8 thr).

### 3.4 Count source: main-object refactorization vs InertiaCounter

`counter_pays` decides per case (16 threads: the counter on the high-fill 3-D cases). ABBA, 3
repetitions, 31 HEP/GHEP cases (all with a certificate but the four largest), `inertia_counter = on`
against the rule (`tools/bench/ab_cases.py`, the variant through a config wrapper), every run valid
(tools/eval, 186/186): geometric mean on/auto **1.034**, none faster by more than 3%, 13 slower by
more than 3% (cav2d-square-1e4 1.153, syn-near-singular-complex-1e5 1.132, ...); cav2d-rect-pert-1e5
0.973 [range overlap], ss-qa8fk-qa8fm 0.995. The rule is right; table `results/p8_counter_ab.md`.

### 3.5 Overlap with other work

Between the Krylov run and the counts only the verify phase runs without MUMPS (0.010 s on
ss-qa8fk-qa8fm and cav2d-rect-pert-1e5); the Krylov run's solves need the same MUMPS lock as a count,
and the window (hence the count points) is known only after the run. Nothing to overlap.

**Conclusion.** With MUMPS 5.8.2 on this machine an exact certificate costs its two count
factorizations; no sound variant measured here is cheaper by more than noise in-process. The default
stays as it is (exact counts, counter by `counter_pays`); the gap of ss-qa8fk-qa8fm and
cav2d-rect-pert-1e5 to tuned SLEPc (§4) is the price of the deterministic proof.

## 4. Head-to-head against tuned SLEPc (task 4a)

`results/cmp-p8-best.md` / `.json` (five nearest configurations) and `results/cmp-p8-best-interval.md` /
`.json` (yee-pec-box-1e5, interval [40, 200]); the method of results/cmp-p5c-best*: tuned SLEPc profile
"best", 5 interleaved repetitions, tools/eval on every run, `--mem-gb 20` (hsolve
`--memory-budget-gb 18`), gpu01 slot. Speedup = SLEPc median solver time / hsolve median solver time.

| case | target | SLEPc s (P5c / P8) | hsolve s (P5c -> P8) | hsolve vs SLEPc (P5c -> P8) | hsolve-auto vs SLEPc (P8; auto's certificate) | hsolve-nocert vs SLEPc (P5c -> P8) |
|---|---|---|---|---|---|---|
| syn-mem27-46 | k=12 @ 0 | 3.48 / 3.48 | 2.39 -> 2.37 | 1.45x -> **1.47x** | **2.01x** (randomized) | 2.34x -> 2.35x |
| yee-pec-box-1e5 | k=10 @ 52.85 | 3.36 / 3.35 | 2.95 -> 2.97 | 1.14x -> **1.13x** | 1.14x (exact) | 1.60x -> 1.61x |
| wg-square-1e5 | k=18 @ -0.527 | 0.89 / 0.92 | 0.49 -> 0.48 | 1.84x -> **1.9x** (SLEPc valid 4/5) | 1.9x (none: GHIEP) | 1.83x -> 1.9x |
| ss-qa8fk-qa8fm | k=10 @ 12.82 | 1.50 / 1.49 | 1.68 -> 1.70 | 0.89x -> **0.87x** | 0.87x (exact) | 1.18x -> 1.15x |
| cav2d-rect-pert-1e5 | k=6 @ 0.1418 | 0.21 / 0.22 | 0.30 -> 0.30 | 0.71x -> **0.73x** | 0.72x (exact) | 0.94x -> 0.88x |
| yee-pec-box-1e5 | [40, 200] | 10.30 / 10.88 | 3.81 -> 3.84 | 2.70x -> **2.83x** | 2.82x (slicing counts) | 2.71x -> 2.81x |

- The interval row ran the binary of 92d4962 (the first re-analysis commit, identical on this case); the five nearest rows that of d0035c5.
- Every hsolve run of every variant was valid (5/5); SLEPc failed 1 of 5 runs on wg-square-1e5 (one of
  18 pairs above tol, success not claimed), so compare.py prints no speedup there; the ratio of the
  medians is given.
- P8 changes nothing on these configurations (their counts never fail, their estimates fit): the
  hsolve columns equal P5c's within the run-to-run spread (P5c ran on c003 / c005, P8 on gpu01).
- The two losses remain: ss-qa8fk-qa8fm 0.87x and cav2d-rect-pert-1e5 0.73x with the exact
  certificate, 1.15x / 0.88x without it; the certificate's two count factorizations are the gap
  (0.40 s / 0.06 s certificate time) and no cheaper exact count was found (§3). auto keeps the exact
  counts there (§6).

## 5. Regression against main 34cbfd4 (task 4b)

All 34 data cases with a certificate (HEP / GHEP nearest; ss-3dspectralwave2 is §2.3), ABBA order
(rep r: main, new for even r, new, main for odd r), 3 repetitions, each binary's default configuration
(case config.json, `--threads 16`, CLI defaults: 14 GiB budget, certificate exact), medians of
`time_total_s`; 204/204 runs valid (tools/eval: every run passes, against the stored reference or the
pooled pairs). Machine-readable: `results/p8_regression.json`.

```bash
S=$HOME/p8_p8-certcost   # from the work tree on gpu01
env/run.sh python3 tools/run/guard.py --exclusive --timeout 2700 -- python tools/bench/ab_cases.py run \
  --bin main=$S/bin/hsolve_main --bin new=$S/bin/hsolve_p8b --reps 0,1,2 --run-dir $S/A-gpu01 <34 cases>
env/run.sh python3 tools/run/guard.py --skip-busy-check --mem-gb 12 --timeout 3400 -- \
  python -m tools.eval.sweep $S/A-gpu01 --out $S/A-gpu01/eval.md
env/run.sh python tools/bench/ab_cases.py table --base main --new new --md A-table.md \
  --json results/p8_regression.json $S/A-gpu01
```

| case | n | type | node | main s [min-max] | new s [min-max] | new/main | valid main / new | numeric # main / new | block p main / new | certificate new |
|---|---|---|---|---|---|---|---|---|---|---|
| cav2d-rect-pert-1e5 | 99459 | GHEP | gpu01 | 0.295 [0.292-0.302] | 0.297 [0.293-0.298] | 1.007 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| cav2d-square-1e4 | 9976 | GHEP | gpu01 | 0.048 [0.046-0.087] | 0.049 [0.049-0.057] | 1.027 | 3/3 / 3/3 | 3 / 3 | 2 / 2 | exact |
| ss-2cubes_sphere | 101492 | HEP | gpu01 | 2.862 [2.781-2.868] | 2.852 [2.850-2.926] | 0.997 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| ss-ge87h76 | 112985 | HEP | gpu01 | 54.293 [54.034-59.888] | 55.474 [53.310-56.386] | 1.022 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-helm3d01 | 32226 | HEP | gpu01 | 0.553 [0.552-0.580] | 0.551 [0.551-0.557] | 0.997 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-qa8fk-qa8fm | 66127 | GHEP | gpu01 | 1.690 [1.676-1.701] | 1.687 [1.677-1.690] | 0.998 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| ss-si34h36 | 97569 | HEP | gpu01 | 39.172 [38.515-39.263] | 39.918 [38.870-41.427] | 1.019 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| ss-tem152078 | 152078 | GHEP | gpu01 | 5.443 [5.442-5.456] | 5.432 [5.415-5.441] | 0.998 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| ss-tem27623 | 27623 | GHEP | gpu01 | 0.599 [0.594-0.603] | 0.593 [0.586-0.595] | 0.991 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-multiplicity-real-1e4 | 10000 | GHEP | gpu01 | 0.137 [0.136-0.139] | 0.140 [0.137-0.152] | 1.027 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-ghep-near-singular-real-1e5 | 100000 | GHEP | gpu01 | 1.311 [1.295-1.316] | 1.290 [1.288-1.293] | 0.984 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-ghep-zero-cluster-complex-1e4 | 10000 | GHEP | gpu01 | 0.289 [0.287-0.289] | 0.289 [0.284-0.294] | 1.000 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap2d-100 | 10000 | HEP | gpu01 | 0.068 [0.066-0.087] | 0.067 [0.063-0.068] | 0.987 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-22 | 10648 | HEP | gpu01 | 0.146 [0.142-0.151] | 0.149 [0.140-0.154] | 1.019 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-lap3d-aniso-1e5 | 97200 | HEP | gpu01 | 2.090 [2.077-2.197] | 2.066 [2.066-2.067] | 0.989 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| syn-mem125-46 | 97336 | HEP | gpu01 | 11.322 [11.099-11.702] | 11.378 [11.132-11.727] | 1.005 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-mem27-46 | 97336 | HEP | gpu01 | 2.370 [2.365-2.416] | 2.391 [2.364-2.396] | 1.009 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| syn-multiplicity-complex-1e4 | 10000 | HEP | gpu01 | 0.207 [0.202-0.209] | 0.207 [0.205-0.211] | 1.002 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-multiplicity-real-1e4 | 10000 | HEP | gpu01 | 0.097 [0.095-0.103] | 0.100 [0.098-0.102] | 1.033 **slower** | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-near-singular-complex-1e4 | 10000 | HEP | gpu01 | 0.235 [0.232-0.239] | 0.231 [0.230-0.233] | 0.987 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-complex-1e5 | 100000 | HEP | gpu01 | 2.506 [2.486-2.514] | 2.486 [2.484-2.498] | 0.992 | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-near-singular-real-1e4 | 10000 | HEP | gpu01 | 0.104 [0.104-0.107] | 0.108 [0.106-0.113] | 1.033 **slower** | 3/3 / 3/3 | 4 / 4 | 4 / 4 | exact |
| syn-wide-range-complex-1e4 | 10000 | HEP | gpu01 | 0.185 [0.183-0.192] | 0.187 [0.184-0.194] | 1.010 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-wide-range-real-1e4 | 10000 | HEP | gpu01 | 0.091 [0.090-0.093] | 0.091 [0.090-0.098] | 1.004 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| syn-zero-cluster-complex-1e4 | 10000 | HEP | gpu01 | 0.309 [0.309-0.336] | 0.309 [0.308-0.310] | 1.000 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e4 | 10000 | HEP | gpu01 | 0.129 [0.129-0.136] | 0.131 [0.129-0.131] | 1.015 | 3/3 / 3/3 | 5 / 5 | 4 / 4 | exact |
| syn-zero-cluster-real-1e5 | 100000 | HEP | gpu01 | 1.230 [1.219-1.251] | 1.217 [1.211-1.220] | 0.990 | 3/3 / 3/3 | 6 / 6 | 4 / 4 | exact |
| yee-bloch-1e4 | 10080 | HEP | gpu01 | 0.833 [0.822-0.856] | 0.837 [0.834-0.839] | 1.005 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-pec-box-1e4 | 10064 | GHEP | gpu01 | 0.200 [0.199-0.201] | 0.202 [0.200-0.205] | 1.011 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-box-1e5 | 109744 | GHEP | gpu01 | 2.869 [2.809-2.906] | 2.893 [2.865-2.929] | 1.008 | 3/3 / 3/3 | 1 / 1 | 4 / 4 | exact |
| yee-pec-cube-16 | 10800 | HEP | gpu01 | 0.211 [0.207-0.215] | 0.209 [0.206-0.214] | 0.991 | 3/3 / 3/3 | 3 / 3 | 4 / 4 | exact |
| yee-pec-cube-32 | 92256 | HEP | gpu01 | 2.281 [2.197-2.292] | 2.251 [2.244-2.262] | 0.987 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-rods-1e4 | 10125 | HEP | gpu01 | 0.911 [0.902-0.918] | 0.920 [0.903-0.932] | 1.009 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |
| yee-photonic-spheres-1e4 | 10125 | GHEP | gpu01 | 0.971 [0.965-0.987] | 0.968 [0.968-0.974] | 0.998 | 3/3 / 3/3 | 1 / 1 | 8 / 8 | exact |

Geometric mean of new/main over the 34 cases valid in every run: **1.004**; faster by > 3%: 0; slower by > 3%: 2 (syn-multiplicity-real-1e4 1.033, syn-near-singular-real-1e4 1.033).

P8 is a robustness change for the default path: geometric mean new/main **1.004**, no case faster by
more than 3%, two 1e4 cases "slower" by 3.3% (syn-multiplicity-real-1e4 0.097 -> 0.100 s,
syn-near-singular-real-1e4 0.104 -> 0.108 s: 3-4 ms, inside the other binary's [min-max]; a 5-repetition
ABBA recheck with the final binary be2efde, 20/20 runs valid: 1.015 and 0.983); the largest
absolute differences (ss-ge87h76 +2.2%, ss-si34h36 +1.9%) are within their run-to-run spread
(53.3-56.4 s, 38.5-41.4 s). Same factorization counts and block widths everywhere.

## 6. Proven kappa of a general SPD B (task 3, optional; done before the re-prioritization)

`prove_condition_bound` (src/driver/norms.*, DESIGN §3.5b): lambda_min(W B W) >= lower from the
incomplete-Cholesky + Gershgorin proof (rigorous a-priori rounding bound), lambda_max <= upper by a
rounding-inflated Gershgorin, kappa <= upper / lower and lambda_min(B) >= lower / max w_i^2. The
certifier uses it for the randomized check's start scaling (kappa) and, through
`SingleCountRule::b_min_proven`, for single counts on a general B; only in auto / randomized modes and
only where they could beat the exact counts. Tests `tests/cpp/test_certcost_kappa.cpp` (Q1 mass
matrix, not diagonally dominant: bounds against dense eigenvalues, the single rule, randomized and
single counts certifying end to end; real and complex).

General-B GHEP data cases (B neither diagonal nor diagonally dominant): ss-qa8fk-qa8fm,
cav2d-rect-pert-1e5, ss-tem27623, ss-tem152078, syn-ghep-multiplicity-real-1e4,
syn-ghep-near-singular-real-1e5, syn-ghep-zero-cluster-complex-1e4 (cav2d-square-1e4's B is
diagonally dominant; the yee-* GHEP have diagonal B). What auto chooses on them (build lane, event
`certificate_choice`): exact counts on every one - ss-qa8fk-qa8fm: randomized check predicted no cheaper
(eps = 0.14) and single counts invalid (their eta 254 against a window of width 2; the bound itself
lower 0.0358, upper 4.22, kappa <= 118, b_min 1.26e-6, 0.084 s, is not even computed any more because
the best possible b_min = min b_ii already fails the rule); cav2d-rect-pert-1e5 and the syn-ghep-* /
ss-tem*: exact counts predicted cheapest. auto therefore never picks a slower method there; its times
equal exact's within noise (§4: hsolve-auto vs hsolve). The default stays exact.

## 7. Open issues

1. **The exact certificate still costs two factorizations.** No sound in-process way to make the exact
   counts cheaper was found (§3); ss-qa8fk-qa8fm (0.87x) and cav2d-rect-pert-1e5 (0.73x) stay behind
   tuned SLEPc with the default `certificate = exact` (1.15x / 0.88x without the certificate). What
   remains would need another factorization backend with thread-safe instances (or a second,
   dlmopen-isolated MUMPS: bounded by 1.29x of the certificate phase, ~4% of the solve).
2. **ss-3dspectralwave2 at the 14 GiB CLI default** (§2.3): the counter's estimate (10.5-15.1 GB)
   often sits above the count allowance (0.95 of what the budget leaves, ~11.2-11.8 GB): in 9 of 21 runs
   the counts ran on the out-of-core main object instead, certified but ~10% slower (309-344 s vs
   294-326 s). The main analysis' SCOTCH estimate exceeded the cap in 6 of 42 analyses (P5c and P8);
   with up to two re-analyses a run now fails only if three analyses in a row miss (~0.3% by that
   rate). P7's automatic budget (32 GiB under the guard) removes both on a 32 GB machine; on a 16 GB
   machine the budget stays near today's.
3. **SCOTCH reproducibility** (§2.4): only a one-thread SCOTCH ordering is reproducible, at 2.6-3.7x the
   analysis time; handing a one-thread order to the main analysis as PERM_IN would make estimates
   (and with them the memory decisions and the blocktune width) reproducible - a factor / anaint decision.
4. **Merging with P7** (p7-membudget): P7 rewrote the shift-invert operator construction in
   `NearestCertifier::randomized` (DriverShiftInvert) right after the refactorization this branch wraps in
   a try/catch, and much of solver.cpp; expect a small textual conflict there. P7's InexactFactorFailure
   is a std::runtime_error, not an hs::Error, so the count-failure handlers here let it through to P7's
   double-precision fallback as intended. `count_allowance_mb` reads `SolverConfig::memory_budget_gb`,
   which P7 resolves before the solve.
5. **auto on general-B GHEP** (§6): the proven kappa is in place but auto never picks the randomized check
   on the data cases (their gaps give eps 0.01-0.14); not pursued further (the default stays exact).
